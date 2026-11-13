"""Explicit, recoverable staged checkout for local workspaces.

Unlike :func:`dvc_data.index.checkout.apply`, which mutates the workspace
in place, a staged checkout freezes the result of a ``DataIndex`` compare
into a :class:`CheckoutPlan` and executes it in two distinct phases:

1. ``prepare`` copies every new/modified object into a private, same-volume
   staging directory and verifies its hash. No workspace file is removed or
   overwritten until every object is ready.
2. ``publish`` applies the plan one entry at a time. Every replacement is
   atomic and every step is recorded in an append-only journal, so a
   cancelled or crashed run can be resumed or rolled back afterwards.

Only local regular files and directories, plus the declared executable-bit
permission operation, are supported. Symlinks and root paths that cannot be
proven safe are rejected.

A cooperative workspace lock serializes writers that follow this protocol.
The lock cannot, and is not claimed to, prevent arbitrary external processes
that ignore the protocol from modifying files; every publish step therefore
re-verifies the frozen workspace pre-state and stops (preserving the file)
when an external modification is detected.
"""

from __future__ import annotations

import ctypes
import errno
import logging
import os
import shutil
import socket
import stat
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable, Optional

from attrs import define, field
from dvc_objects.fs.generic import transfer
from dvc_objects.fs.local import LocalFileSystem

from dvc_data.hashfile.hash import file_md5
from dvc_data.hashfile.hash_info import HashInfo
from dvc_data.json_compat import dumps as json_dumps
from dvc_data.json_compat import loads as json_loads

from .checkout import compare

if TYPE_CHECKING:
    from collections.abc import Iterator

    from dvc_objects.fs.base import FileSystem

    from dvc_data.hashfile.state import StateBase

    from .index import BaseDataIndex, DataIndexEntry, DataIndexKey

logger = logging.getLogger(__name__)

PLAN_VERSION = 1
STAGING_DIRNAME = ".dvc-staged-checkout"
LOCK_FNAME = "lock"
PLAN_FNAME = "plan.json"
JOURNAL_FNAME = "journal.jsonl"
OBJECTS_DIRNAME = "objects"
ROLLBACK_DIRNAME = "rollback"

OP_CREATE_FILE = "create_file"
OP_DELETE_FILE = "delete_file"
OP_MKDIR = "mkdir"
OP_RMDIR = "rmdir"
OP_CHMOD = "chmod"

# Journal events
EV_PLAN_CREATED = "plan_created"
EV_PREPARED = "prepared"
EV_PREPARE_FAILED = "prepare_failed"
EV_PUBLISH_START = "publish_start"
EV_ROLLBACK_SAVED = "rollback_saved"
EV_PUBLISHED = "published"
EV_PUBLISH_FAILED = "publish_failed"
EV_COMMITTED = "committed"
EV_ROLLBACK_START = "rollback_start"
EV_ROLLED_BACK = "rolled_back"


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


class StagedCheckoutError(Exception):
    """Base class for staged checkout errors."""

    def __init__(self, message: str, plan_id: Optional[str] = None):
        super().__init__(message)
        self.plan_id = plan_id


class StagedCheckoutSafetyError(StagedCheckoutError):
    """Unsupported filesystem object or path that cannot be proven safe."""


class StagedCheckoutLockError(StagedCheckoutError):
    """The cooperative workspace lock could not be acquired."""


class StagedCheckoutConflictError(StagedCheckoutError):
    """One or more conflicts were detected before anything was overwritten."""

    def __init__(self, conflicts: list[Conflict], plan_id: Optional[str] = None):
        self.conflicts = conflicts
        summary = "; ".join(f"{c.kind}: {c.path}" for c in conflicts)
        super().__init__(f"staged checkout conflicts: {summary}", plan_id)


class StagedCheckoutObjectMissingError(StagedCheckoutError):
    """A source object declared by the plan is not available."""


class StagedCheckoutSpaceError(StagedCheckoutError):
    """Not enough free space to stage or publish the plan."""


class StagedCheckoutExternalModificationError(StagedCheckoutError):
    """A target changed outside of the staged checkout protocol."""


class StagedCheckoutPlanError(StagedCheckoutError):
    """The plan itself is invalid or cannot be executed."""


# ---------------------------------------------------------------------------
# frozen data structures
# ---------------------------------------------------------------------------


@define
class Conflict:
    kind: str
    key: tuple[str, ...]
    path: str
    detail: str

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "key": list(self.key),
            "path": self.path,
            "detail": self.detail,
        }


@define
class PreState:
    """Frozen snapshot of a workspace path before the checkout."""

    exists: bool
    type: Optional[str] = None  # "file" | "directory" | "other"
    islink: bool = False
    size: Optional[int] = None
    mtime_ns: Optional[int] = None
    ino: Optional[int] = None
    mode: Optional[int] = None
    tracked: bool = False
    hash_info: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "exists": self.exists,
            "type": self.type,
            "islink": self.islink,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "ino": self.ino,
            "mode": self.mode,
            "tracked": self.tracked,
            "hash_info": self.hash_info,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PreState:
        return cls(**d)

    def matches_stat(self, st: Optional[os.stat_result]) -> bool:
        """Check that a fresh stat still matches the frozen pre-state."""
        if not self.exists:
            return st is None
        if st is None:
            return False
        cur_type = _stat_type(st)
        if cur_type != self.type or bool(stat.S_ISLNK(st.st_mode)) != self.islink:
            return False
        return (
            st.st_size == self.size
            and st.st_mtime_ns == self.mtime_ns
            and st.st_ino == self.ino
        )


@define
class PlanEntry:
    """A single, ordered operation of a frozen checkout plan."""

    key: tuple[str, ...]
    op: str
    pre: PreState
    target_hash: Optional[dict] = None
    target_meta: Optional[dict] = None
    isexec: bool = False
    src_fs: Optional[str] = None
    src_path: Optional[str] = None
    size: Optional[int] = None
    staged_rel: Optional[str] = None
    rollback_rel: Optional[str] = None

    @property
    def relpath(self) -> str:
        return _key_to_rel(self.key)

    def to_dict(self) -> dict:
        return {
            "key": list(self.key),
            "op": self.op,
            "pre": self.pre.to_dict(),
            "target_hash": self.target_hash,
            "target_meta": self.target_meta,
            "isexec": self.isexec,
            "src_fs": self.src_fs,
            "src_path": self.src_path,
            "size": self.size,
            "staged_rel": self.staged_rel,
            "rollback_rel": self.rollback_rel,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PlanEntry:
        d = dict(d)
        d["key"] = tuple(d["key"])
        d["pre"] = PreState.from_dict(d["pre"])
        return cls(**d)


@define
class CheckoutPlan:
    plan_id: str
    workspace: str
    created_at: str
    storage: str
    entries: list[PlanEntry] = field(factory=list)
    version: int = PLAN_VERSION

    # -- (de)serialization -------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "plan_id": self.plan_id,
            "workspace": self.workspace,
            "created_at": self.created_at,
            "storage": self.storage,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, d: dict) -> CheckoutPlan:
        if d.get("version") != PLAN_VERSION:
            raise StagedCheckoutPlanError(
                f"unsupported plan version {d.get('version')!r}",
                plan_id=d.get("plan_id"),
            )
        return cls(
            plan_id=d["plan_id"],
            workspace=d["workspace"],
            created_at=d["created_at"],
            storage=d["storage"],
            entries=[PlanEntry.from_dict(e) for e in d["entries"]],
            version=d["version"],
        )

    def save(self, staging_root: str) -> None:
        plan_dir = self.plan_dir(staging_root)
        os.makedirs(plan_dir, exist_ok=True)
        os.makedirs(os.path.join(plan_dir, OBJECTS_DIRNAME), exist_ok=True)
        os.makedirs(os.path.join(plan_dir, ROLLBACK_DIRNAME), exist_ok=True)
        path = os.path.join(plan_dir, PLAN_FNAME)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf8") as fobj:
            fobj.write(json_dumps(self.to_dict()))
            fobj.flush()
            os.fsync(fobj.fileno())
        os.replace(tmp, path)

    @classmethod
    def load(cls, staging_root: str, plan_id: str) -> CheckoutPlan:
        path = os.path.join(staging_root, plan_id, PLAN_FNAME)
        try:
            with open(path, encoding="utf8") as fobj:
                return cls.from_dict(json_loads(fobj.read()))
        except FileNotFoundError as exc:
            raise StagedCheckoutPlanError(
                f"plan {plan_id!r} not found in {staging_root!r}", plan_id
            ) from exc

    # -- paths -------------------------------------------------------------

    def plan_dir(self, staging_root: str) -> str:
        return os.path.join(staging_root, self.plan_id)

    def objects_dir(self, staging_root: str) -> str:
        return os.path.join(self.plan_dir(staging_root), OBJECTS_DIRNAME)

    def rollback_dir(self, staging_root: str) -> str:
        return os.path.join(self.plan_dir(staging_root), ROLLBACK_DIRNAME)

    def journal_path(self, staging_root: str) -> str:
        return os.path.join(self.plan_dir(staging_root), JOURNAL_FNAME)

    def dest_path(self, entry: PlanEntry) -> str:
        if not entry.key:
            return self.workspace
        return os.path.join(self.workspace, *entry.key)

    def iter_ops(self, op: str) -> Iterator[PlanEntry]:
        yield from (entry for entry in self.entries if entry.op == op)


# ---------------------------------------------------------------------------
# journal
# ---------------------------------------------------------------------------


class Journal:
    """Append-only, fsynced event log for a single plan."""

    def __init__(self, path: str):
        self.path = path

    def append(self, event: str, **fields) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **fields,
        }
        line = json_dumps(record)
        with open(self.path, "a", encoding="utf8") as fobj:
            fobj.write(line + "\n")
            fobj.flush()
            os.fsync(fobj.fileno())

    def events(self) -> list[dict]:
        try:
            with open(self.path, encoding="utf8") as fobj:
                return [json_loads(line) for line in fobj if line.strip()]
        except FileNotFoundError:
            return []

    def keys_for(self, event: str) -> set[tuple[str, ...]]:
        ret = set()
        for record in self.events():
            if record.get("event") == event and "key" in record:
                ret.add(tuple(record["key"]))
        return ret

    def has(self, event: str) -> bool:
        return any(record.get("event") == event for record in self.events())


# ---------------------------------------------------------------------------
# cooperative workspace lock
# ---------------------------------------------------------------------------


def _pid_alive(pid: int) -> bool:
    """Best-effort cross-platform process liveness check."""
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000  # noqa: N806
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return False
            kernel32.CloseHandle(handle)
            return True
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    except AttributeError:
        # Unknown platform: assume alive, never break a lock speculatively.
        return True
    return True


@define
class LockInfo:
    plan_id: Optional[str]
    pid: int
    host: str
    created_at: str

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "pid": self.pid,
            "host": self.host,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> LockInfo:
        return cls(
            plan_id=d.get("plan_id"),
            pid=int(d["pid"]),
            host=d.get("host", ""),
            created_at=d.get("created_at", ""),
        )


class WorkspaceLock:
    """Exclusive, cooperative lock for a workspace staging root.

    The lock only coordinates writers that use this protocol. It cannot stop
    an external process which ignores it from editing workspace files.
    """

    def __init__(
        self,
        staging_root: str,
        plan_id: Optional[str] = None,
        stale_seconds: float = 3600,
    ):
        self.staging_root = staging_root
        self.plan_id = plan_id
        self.stale_seconds = stale_seconds
        self.lock_path = os.path.join(staging_root, LOCK_FNAME)
        self._owned = False

    def acquire(self) -> None:
        os.makedirs(self.staging_root, exist_ok=True)
        info = LockInfo(
            plan_id=self.plan_id,
            pid=os.getpid(),
            host=socket.gethostname(),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            fd = os.open(self.lock_path, flags)
        except FileExistsError as exc:
            self._handle_stale_lock(exc)
            return
        try:
            with os.fdopen(fd, "w", encoding="utf8") as fobj:
                fobj.write(json_dumps(info.to_dict()))
                fobj.flush()
                os.fsync(fobj.fileno())
            self._owned = True
        except BaseException:
            with _suppress_file_error():
                os.unlink(self.lock_path)
            raise

    def _handle_stale_lock(self, exc: FileExistsError) -> None:
        try:
            with open(self.lock_path, encoding="utf8") as fobj:
                info = LockInfo.from_dict(json_loads(fobj.read()))
        except (OSError, ValueError):
            raise StagedCheckoutLockError(
                f"workspace is locked by an unreadable lock at {self.lock_path!r}"
            ) from exc

        age = _lock_age_seconds(info.created_at)
        if age is not None and age > self.stale_seconds and not _pid_alive(info.pid):
            logger.warning("breaking stale workspace lock %r", self.lock_path)
            with _suppress_file_error():
                os.unlink(self.lock_path)
            self.acquire()
            return

        raise StagedCheckoutLockError(
            f"workspace is locked by pid {info.pid} "
            f"(plan {info.plan_id!r}) at {self.lock_path!r}; "
            "the lock only coordinates cooperating writers and cannot stop "
            "external processes that bypass it"
        ) from exc

    def release(self) -> None:
        if not self._owned:
            return
        try:
            with open(self.lock_path, encoding="utf8") as fobj:
                info = LockInfo.from_dict(json_loads(fobj.read()))
        except (OSError, ValueError):
            info = None
        if info is None or info.pid == os.getpid():
            with _suppress_file_error():
                os.unlink(self.lock_path)
        self._owned = False

    def __enter__(self) -> WorkspaceLock:  # noqa: PYI034
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


@contextmanager
def _suppress_file_error() -> Iterator[None]:
    try:
        yield
    except FileNotFoundError:
        pass


def _lock_age_seconds(created_at: str) -> Optional[float]:
    if not created_at:
        return None
    try:
        ts = datetime.fromisoformat(created_at)
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - ts).total_seconds()


# ---------------------------------------------------------------------------
# safety helpers
# ---------------------------------------------------------------------------


def _stat_type(st: os.stat_result) -> str:
    if stat.S_ISDIR(st.st_mode):
        return "directory"
    if stat.S_ISREG(st.st_mode):
        return "file"
    return "other"


def _key_to_rel(key: DataIndexKey) -> str:
    if not key:
        return "_root_"
    parts = []
    for part in key:
        if (
            not part
            or part in (".", "..")
            or "\0" in part
            or os.sep in part
            or (os.altsep and os.altsep in part)
            or os.path.isabs(part)
        ):
            raise StagedCheckoutSafetyError(f"unsafe key component {part!r}")
        parts.append(part)
    return os.path.join(*parts)


def _ensure_local_fs(fs: FileSystem) -> None:
    if not isinstance(fs, LocalFileSystem):
        raise StagedCheckoutSafetyError(
            f"staged checkout only supports the local filesystem, got {fs.protocol!r}"
        )


def _ensure_safe_root(path: str) -> str:  # noqa: C901
    if not isinstance(path, str) or not os.path.isabs(path):
        raise StagedCheckoutSafetyError(
            f"workspace root must be an absolute local path, got {path!r}"
        )
    path = os.path.normpath(path)

    # Walk every existing component with lstat: symlinks anywhere on the
    # path mean we cannot prove the resolved target stays inside the
    # workspace.
    anchor = path
    while anchor and not os.path.lexists(anchor):
        parent = os.path.dirname(anchor)
        if parent == anchor:
            break
        anchor = parent

    cur = anchor
    for component in os.path.relpath(path, anchor).split(os.sep):
        if component in ("", "."):
            continue
        try:
            st = os.lstat(cur)
        except OSError as exc:
            raise StagedCheckoutSafetyError(
                f"cannot prove safety of root {path!r}: {exc}"
            ) from exc
        if stat.S_ISLNK(st.st_mode):
            raise StagedCheckoutSafetyError(
                f"symlinks are not supported, refusing path component {cur!r}"
            )
        if not stat.S_ISDIR(st.st_mode):
            raise StagedCheckoutSafetyError(
                f"path component {cur!r} is not a directory"
            )
        cur = os.path.join(cur, component)

    if os.path.lexists(path):
        st = os.lstat(path)
        if stat.S_ISLNK(st.st_mode):
            raise StagedCheckoutSafetyError(
                f"symlinks are not supported, refusing workspace {path!r}"
            )
        if not stat.S_ISDIR(st.st_mode):
            # A single-file checkout target (key ()) is allowed; any other
            # existing non-directory root is validated per entry below.
            logger.debug("workspace root %r is not a directory", path)
    return path


def _lstat_safe(path: str) -> Optional[os.stat_result]:
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(st.st_mode):
        raise StagedCheckoutSafetyError(
            f"symlinks are not supported, refusing {path!r}"
        )
    if not (stat.S_ISREG(st.st_mode) or stat.S_ISDIR(st.st_mode)):
        raise StagedCheckoutSafetyError(
            f"only regular files and directories are supported, refusing {path!r}"
        )
    return st


def default_staging_root(workspace: str) -> str:
    """Return the deterministic, same-volume private staging root."""
    if os.path.isdir(workspace):
        return os.path.join(workspace, STAGING_DIRNAME)
    parent = os.path.dirname(workspace) or os.curdir
    return os.path.join(parent, STAGING_DIRNAME)


def _ensure_same_volume(path_a: str, path_b: str) -> None:
    try:
        dev_a = os.stat(os.path.dirname(path_a) or os.curdir).st_dev
        dev_b = os.stat(path_b).st_dev
    except OSError:
        return
    if dev_a != dev_b:
        raise StagedCheckoutSafetyError(
            f"staging root {path_b!r} is on a different volume than "
            f"workspace {path_a!r}; same-volume staging is required"
        )


def _is_within(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([os.path.abspath(path), os.path.abspath(root)]) == (
            os.path.abspath(root)
        )
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# plan construction
# ---------------------------------------------------------------------------


def _snapshot(
    path: str,
    *,
    tracked: bool,
    expected_hash: Optional[HashInfo],
) -> PreState:
    st = _lstat_safe(path)
    if st is None:
        return PreState(exists=False, tracked=tracked)

    ptype = _stat_type(st)
    hash_dict = expected_hash.to_dict() if expected_hash else None
    return PreState(
        exists=True,
        type=ptype,
        islink=False,
        size=st.st_size,
        mtime_ns=st.st_mtime_ns,
        ino=st.st_ino,
        mode=st.st_mode,
        tracked=tracked,
        hash_info=hash_dict,
    )


def _is_tracked(old: Optional[BaseDataIndex], key: DataIndexKey) -> bool:
    if old is None:
        return False
    try:
        return old.get(key) is not None
    except Exception:  # noqa: BLE001
        return False


def _old_entry(
    old: Optional[BaseDataIndex], key: DataIndexKey
) -> Optional[DataIndexEntry]:
    if old is None:
        return None
    return old.get(key)


def _walk_children(
    path: str, staging_root: str
) -> Iterator[tuple[str, os.stat_result]]:
    for root, dirs, files in os.walk(path):
        if _is_within(root, staging_root):
            dirs[:] = []
            continue
        dirs.sort()
        files.sort()
        for name in files:
            child = os.path.join(root, name)
            st = _lstat_safe(child)
            if st is not None:
                yield child, st
        for name in dirs:
            child = os.path.join(root, name)
            st = _lstat_safe(child)
            if st is not None and stat.S_ISDIR(st.st_mode):
                yield child, st


def _resolve_source(
    index: Optional[BaseDataIndex], entry: DataIndexEntry, storage: str
) -> tuple[Optional[str], Optional[str], Optional[int]]:
    if index is None or not entry.hash_info:
        return None, None, entry.meta.size if entry.meta else None
    storage_info = index.storage_map[entry.key]
    storage_obj = getattr(storage_info, storage)
    src_fs, src_path = storage_obj.get(entry)
    size = entry.meta.size if entry.meta and entry.meta.size is not None else None
    if size is None:
        try:
            size = src_fs.size(src_path)
        except OSError:
            size = None
    return src_fs.protocol, src_path, size


def build_plan(  # noqa: C901, PLR0915
    old: Optional[BaseDataIndex],
    new: BaseDataIndex,
    path: str,
    fs: FileSystem,
    *,
    delete: bool = False,
    storage: str = "cache",
    staging_root: Optional[str] = None,
    plan_id: Optional[str] = None,
) -> CheckoutPlan:
    """Freeze a compare result into an executable, conflict-checked plan."""
    _ensure_local_fs(fs)
    workspace = _ensure_safe_root(os.path.abspath(path))
    staging_root = os.path.abspath(staging_root or default_staging_root(workspace))
    _ensure_same_volume(workspace, staging_root)

    diff = compare(old, new, delete=delete)
    conflicts: list[Conflict] = []

    create_keys = {tuple(e.key) for e in diff.files_create}
    chmod_entries = [
        entry for entry in diff.files_chmod if tuple(entry.key) not in create_keys
    ]

    # Parent-child sanity check inside the plan itself: a regular file cannot
    # have descendants created beneath it.
    for entry in diff.files_create:
        key = tuple(entry.key)
        for other in diff.files_create:
            other_key = tuple(other.key)
            if other_key != key and other_key[: len(key)] == key:
                conflicts.append(
                    Conflict(
                        kind="parent_child",
                        key=other_key,
                        path=fs.join(workspace, *other_key),
                        detail=(
                            f"cannot create {other_key!r} beneath file target {key!r}"
                        ),
                    )
                )

    plan_entries: list[PlanEntry] = []

    def _dest(key: DataIndexKey) -> str:
        return workspace if not key else os.path.join(workspace, *key)

    def _record_conflict(
        kind: str,
        key: DataIndexKey,
        target: str,
        detail: str,
    ) -> None:
        conflicts.append(
            Conflict(kind=kind, key=tuple(key), path=target, detail=detail)
        )

    def _check_existing(
        key: DataIndexKey,
        target: str,
        old_entry: Optional[DataIndexEntry],
        *,
        expect_dir: bool,
    ) -> None:
        st = _lstat_safe(target)
        if st is None:
            return
        tracked = old_entry is not None
        actual_type = _stat_type(st)
        expected_isdir = bool(old_entry and old_entry.isdir)
        if not tracked:
            _record_conflict(
                "untracked",
                key,
                target,
                "existing path is not part of the pre-state index",
            )
            return
        if actual_type != ("directory" if expected_isdir else "file"):
            _record_conflict(
                "type_mismatch",
                key,
                target,
                (
                    f"workspace has {actual_type!r}, pre-state records "
                    f"{'directory' if expected_isdir else 'file'!r}"
                ),
            )
            return
        if (
            actual_type == "file"
            and old_entry
            and old_entry.hash_info
            and old_entry.hash_info.value
        ):
            actual_hash = file_md5(target, name=old_entry.hash_info.name or "md5")
            if actual_hash != old_entry.hash_info.value:
                _record_conflict(
                    "content_mismatch",
                    key,
                    target,
                    (
                        f"content hash {actual_hash!r} does not match "
                        f"pre-state {old_entry.hash_info.value!r}"
                    ),
                )
        if expect_dir and stat.S_ISDIR(st.st_mode):
            _check_dir_children(key, target, old)

    def _check_dir_children(
        key: DataIndexKey, target: str, old_index: Optional[BaseDataIndex]
    ) -> None:
        for child, _st in _walk_children(target, staging_root):
            rel = os.path.relpath(child, target)
            child_key = (*key, *rel.split(os.sep))
            if not _is_tracked(old_index, child_key):
                _record_conflict(
                    "untracked",
                    child_key,
                    child,
                    "untracked child would be lost by directory replacement",
                )

    # Deletions first (files, deepest first), then directories.
    for entry in sorted(diff.files_delete, key=lambda e: tuple(e.key), reverse=True):
        key = tuple(entry.key)
        target = _dest(key)
        old_e = _old_entry(old, key)
        _check_existing(key, target, old_e, expect_dir=False)
        pre = _snapshot(
            target,
            tracked=old_e is not None,
            expected_hash=old_e.hash_info if old_e else None,
        )
        plan_entries.append(
            PlanEntry(
                key=key,
                op=OP_DELETE_FILE,
                pre=pre,
                rollback_rel=_key_to_rel(key),
            )
        )

    for entry in sorted(diff.dirs_delete, key=lambda e: tuple(e.key), reverse=True):
        key = tuple(entry.key)
        target = _dest(key)
        old_e = _old_entry(old, key)
        _check_existing(key, target, old_e, expect_dir=True)
        pre = _snapshot(target, tracked=old_e is not None, expected_hash=None)
        plan_entries.append(
            PlanEntry(
                key=key,
                op=OP_RMDIR,
                pre=pre,
                rollback_rel=_key_to_rel(key),
            )
        )

    # Creations: directories (shallow first), then files.
    for entry in sorted(diff.dirs_create, key=lambda e: tuple(e.key)):
        key = tuple(entry.key)
        target = _dest(key)
        old_e = _old_entry(old, key)
        _check_existing(key, target, old_e, expect_dir=True)
        pre = _snapshot(target, tracked=old_e is not None, expected_hash=None)
        plan_entries.append(PlanEntry(key=key, op=OP_MKDIR, pre=pre))

    for entry in sorted(diff.files_create, key=lambda e: tuple(e.key)):
        key = tuple(entry.key)
        target = _dest(key)
        old_e = _old_entry(old, key)
        _check_existing(key, target, old_e, expect_dir=False)
        pre = _snapshot(
            target,
            tracked=old_e is not None,
            expected_hash=old_e.hash_info if old_e else None,
        )
        src_protocol, src_path, size = _resolve_source(new, entry, storage)
        if not entry.hash_info:
            raise StagedCheckoutPlanError(
                f"entry {key!r} has no hash info; staged checkout requires "
                "a resolvable source object"
            )
        plan_entry = PlanEntry(
            key=key,
            op=OP_CREATE_FILE,
            pre=pre,
            target_hash=entry.hash_info.to_dict(),
            target_meta=entry.meta.to_dict() if entry.meta else None,
            isexec=bool(entry.meta and entry.meta.isexec),
            src_fs=src_protocol,
            src_path=src_path,
            size=size,
        )
        plan_entry.staged_rel = _key_to_rel(key)
        plan_entries.append(plan_entry)

    for entry in sorted(chmod_entries, key=lambda e: tuple(e.key)):
        key = tuple(entry.key)
        target = _dest(key)
        old_e = _old_entry(old, key)
        pre = _snapshot(
            target,
            tracked=old_e is not None,
            expected_hash=old_e.hash_info if old_e else None,
        )
        plan_entries.append(
            PlanEntry(
                key=key,
                op=OP_CHMOD,
                pre=pre,
                target_hash=entry.hash_info.to_dict() if entry.hash_info else None,
                isexec=bool(entry.meta and entry.meta.isexec),
            )
        )

    if conflicts:
        seen: set[tuple[str, str]] = set()
        unique: list[Conflict] = []
        for conflict in conflicts:
            marker = (conflict.kind, conflict.path)
            if marker not in seen:
                seen.add(marker)
                unique.append(conflict)
        raise StagedCheckoutConflictError(unique)

    return CheckoutPlan(
        plan_id=plan_id or uuid.uuid4().hex,
        workspace=workspace,
        created_at=datetime.now(timezone.utc).isoformat(),
        storage=storage,
        entries=plan_entries,
    )


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------


def _copy_file(
    src_fs: FileSystem,
    src_path: str,
    dest_fs: FileSystem,
    dest_path: str,
) -> None:
    """Copy seam. Always uses real copies, never hardlinks, so that preparing
    new content can never rewrite a shared cache object."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    transfer(
        src_fs,
        src_path,
        dest_fs,
        dest_path,
        links=["copy"],
    )


def _free_space(path: str) -> int:
    return shutil.disk_usage(path).free


def prepare(
    plan: CheckoutPlan,
    fs: FileSystem,
    staging_root: str,
    *,
    journal: Optional[Journal] = None,
) -> Journal:
    """Stage and verify every new object before touching the workspace."""
    _ensure_local_fs(fs)
    journal = journal or Journal(plan.journal_path(staging_root))
    prepared = journal.keys_for(EV_PREPARED)
    objects_dir = plan.objects_dir(staging_root)

    creates = list(plan.iter_ops(OP_CREATE_FILE))
    missing = [
        entry
        for entry in creates
        if entry.src_path is None or not fs.exists(entry.src_path)
    ]
    # src_fs may differ from the destination fs; existence is checked through
    # the source filesystem via transfer below, but local sources (the only
    # supported case) can be checked directly here.
    if missing:
        first = missing[0]
        raise StagedCheckoutObjectMissingError(
            f"source object for {first.key!r} is missing "
            f"({first.src_fs}:{first.src_path!r})",
            plan_id=plan.plan_id,
        )

    needed = sum(entry.size or 0 for entry in creates)
    rollback_needed = sum(
        entry.pre.size or 0
        for entry in plan.entries
        if entry.op in (OP_DELETE_FILE, OP_RMDIR) and entry.pre.exists
    )
    try:
        free = _free_space(plan.workspace)
    except OSError:
        free = None
    if free is not None and free < needed + rollback_needed:
        raise StagedCheckoutSpaceError(
            f"need {needed + rollback_needed} bytes but only {free} are free",
            plan_id=plan.plan_id,
        )

    for entry in creates:
        if entry.key in prepared:
            continue
        assert entry.staged_rel
        assert entry.src_path
        staged_path = os.path.join(objects_dir, entry.staged_rel)
        tmp_path = f"{staged_path}.tmp"
        try:
            _copy_file(fs, entry.src_path, fs, tmp_path)
            target_hash = HashInfo.from_dict(entry.target_hash or {})
            actual = file_md5(tmp_path, name=target_hash.name or "md5")
            if target_hash.value and actual != target_hash.value:
                raise StagedCheckoutPlanError(
                    f"staged object for {entry.key!r} hash {actual!r} does "
                    f"not match plan hash {target_hash.value!r}",
                    plan_id=plan.plan_id,
                )
            # Break any read-only protection inherited from the cache and
            # make sure the staged inode is independent from cache objects.
            mode = os.stat(tmp_path).st_mode
            os.chmod(tmp_path, mode | stat.S_IWRITE)
            os.replace(tmp_path, staged_path)
        except StagedCheckoutError:
            with _suppress_file_error():
                os.unlink(tmp_path)
            raise
        except OSError as exc:
            with _suppress_file_error():
                os.unlink(tmp_path)
            if exc.errno == errno.ENOSPC:
                journal.append(
                    EV_PREPARE_FAILED,
                    key=list(entry.key),
                    error="ENOSPC",
                )
                raise StagedCheckoutSpaceError(
                    f"ran out of space while staging {entry.key!r}",
                    plan_id=plan.plan_id,
                ) from exc
            raise
        journal.append(
            EV_PREPARED,
            key=list(entry.key),
            staged_rel=entry.staged_rel,
            hash=entry.target_hash,
            size=entry.size,
        )

    return journal


# ---------------------------------------------------------------------------
# publish
# ---------------------------------------------------------------------------


def _verify_pre_state(entry: PlanEntry, path: str, plan_id: str) -> None:
    st = _lstat_safe(path)
    if not entry.pre.matches_stat(st):
        raise StagedCheckoutExternalModificationError(
            f"{path!r} was modified outside the staged checkout since the "
            "plan was frozen; stopping and preserving the file",
            plan_id=plan_id,
        )


def _removed_keys(
    plan: CheckoutPlan, published: set[tuple[str, ...]]
) -> set[tuple[str, ...]]:
    return {
        entry.key
        for entry in plan.entries
        if entry.op in (OP_DELETE_FILE, OP_RMDIR) and entry.key in published
    }


def _is_under(key: tuple[str, ...], removed: set[tuple[str, ...]]) -> bool:
    return any(key == rk or key[: len(rk)] == rk for rk in removed)


def _verify_entry_pre_state(
    entry: PlanEntry,
    target: str,
    plan: CheckoutPlan,
    removed: set[tuple[str, ...]],
) -> None:
    """Verify the frozen pre-state, accounting for removals already applied
    by earlier operations of this same plan."""
    if _is_under(entry.key, removed):
        if _lstat_safe(target) is not None:
            raise StagedCheckoutExternalModificationError(
                f"{target!r} should have been removed by an earlier plan "
                "operation but still exists; stopping and preserving it",
                plan_id=plan.plan_id,
            )
        return

    # Deleting a descendant updates the parent's mtime, so directory entries
    # whose children were removed earlier are checked for type only.
    descendant_removed = any(
        rk != entry.key and rk[: len(entry.key)] == entry.key for rk in removed
    )
    if descendant_removed:
        st = _lstat_safe(target)
        if st is None or _stat_type(st) != entry.pre.type:
            raise StagedCheckoutExternalModificationError(
                f"{target!r} changed unexpectedly while the plan was running; "
                "stopping and preserving it",
                plan_id=plan.plan_id,
            )
        return

    _verify_pre_state(entry, target, plan.plan_id)


def _save_rollback_file(
    src: str, rollback_path: str, rel: str, journal: Journal, key: DataIndexKey
) -> None:
    os.makedirs(os.path.dirname(rollback_path), exist_ok=True)
    tmp = f"{rollback_path}.tmp"
    shutil.copyfile(src, tmp)
    os.replace(tmp, rollback_path)
    journal.append(
        EV_ROLLBACK_SAVED,
        key=list(key),
        rollback_rel=rel,
    )


def _save_dir_rollback(
    entry: PlanEntry,
    target: str,
    plan: CheckoutPlan,
    staging_root: str,
    journal: Journal,
    deleted_file_keys: set[tuple[str, ...]],
) -> None:
    rel = entry.rollback_rel or entry.relpath
    rollback_root = os.path.join(plan.rollback_dir(staging_root), rel)
    saved = []
    for child, st in _walk_children(target, staging_root):
        if not stat.S_ISREG(st.st_mode):
            continue
        child_rel = os.path.relpath(child, target)
        child_key = (*entry.key, *child_rel.split(os.sep))
        # Children with their own delete_file entry are saved separately.
        if child_key in deleted_file_keys:
            continue
        dest = os.path.join(rollback_root, child_rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        tmp = f"{dest}.tmp"
        shutil.copyfile(child, tmp)
        os.replace(tmp, dest)
        saved.append(child_rel)
    journal.append(
        EV_ROLLBACK_SAVED,
        key=list(entry.key),
        rollback_rel=rel,
        files=saved,
    )


def _post_record(st: Optional[os.stat_result]) -> Optional[dict]:
    if st is None:
        return None
    return {
        "type": _stat_type(st),
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "ino": st.st_ino,
        "mode": st.st_mode,
    }


def publish(  # noqa: C901, PLR0912, PLR0915
    plan: CheckoutPlan,
    fs: FileSystem,
    staging_root: str,
    *,
    state: Optional[StateBase] = None,
    journal: Optional[Journal] = None,
    on_op_published: Optional[Callable[[PlanEntry], None]] = None,
) -> Journal:
    """Publish a fully prepared plan, one journaled operation at a time."""
    _ensure_local_fs(fs)
    journal = journal or Journal(plan.journal_path(staging_root))
    if journal.has(EV_COMMITTED):
        return journal
    if not journal.has(EV_PUBLISH_START):
        journal.append(EV_PUBLISH_START)

    published = journal.keys_for(EV_PUBLISHED)
    objects_dir = plan.objects_dir(staging_root)
    rollback_dir = plan.rollback_dir(staging_root)
    deleted_file_keys = {
        entry.key for entry in plan.entries if entry.op == OP_DELETE_FILE
    }
    removed = _removed_keys(plan, published)

    for entry in plan.entries:
        if entry.key in published:
            continue
        target = plan.dest_path(entry)
        try:
            # Every operation re-verifies the frozen pre-state: cooperating
            # lock holders are serialized, but external processes bypassing
            # the lock must be detected instead of silently overwritten.
            _verify_entry_pre_state(entry, target, plan, removed)

            if entry.op == OP_DELETE_FILE:
                if os.path.lexists(target):
                    if entry.pre.tracked and stat.S_ISREG(os.lstat(target).st_mode):
                        rel = entry.rollback_rel or entry.relpath
                        rollback_path = os.path.join(rollback_dir, rel)
                        _save_rollback_file(
                            target, rollback_path, rel, journal, entry.key
                        )
                    os.unlink(target)
                journal.append(
                    EV_PUBLISHED, key=list(entry.key), op=entry.op, post=None
                )
                removed.add(entry.key)

            elif entry.op == OP_RMDIR:
                if os.path.isdir(target):
                    _save_dir_rollback(
                        entry,
                        target,
                        plan,
                        staging_root,
                        journal,
                        deleted_file_keys,
                    )
                    try:
                        os.rmdir(target)
                    except OSError as exc:
                        if exc.errno == errno.ENOTEMPTY:
                            raise StagedCheckoutExternalModificationError(
                                f"directory {target!r} is no longer empty; "
                                "stopping and preserving its contents",
                                plan_id=plan.plan_id,
                            ) from exc
                        raise
                journal.append(
                    EV_PUBLISHED, key=list(entry.key), op=entry.op, post=None
                )
                removed.add(entry.key)

            elif entry.op == OP_MKDIR:
                os.makedirs(target, exist_ok=True)
                journal.append(
                    EV_PUBLISHED,
                    key=list(entry.key),
                    op=entry.op,
                    post=_post_record(os.lstat(target)),
                )

            elif entry.op == OP_CREATE_FILE:
                assert entry.staged_rel
                staged_path = os.path.join(objects_dir, entry.staged_rel)
                if not os.path.isfile(staged_path):
                    raise StagedCheckoutPlanError(
                        f"staged object for {entry.key!r} is missing; run "
                        "prepare before publish",
                        plan_id=plan.plan_id,
                    )
                parent = os.path.dirname(target)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                # Same-volume os.replace is an atomic per-file replacement.
                os.replace(staged_path, target)
                st = os.lstat(target)
                journal.append(
                    EV_PUBLISHED,
                    key=list(entry.key),
                    op=entry.op,
                    post=_post_record(st),
                )
                # State cache and the executable bit are updated only after
                # the journal confirms the replacement.
                if entry.isexec:
                    try:
                        os.chmod(target, st.st_mode | stat.S_IEXEC)
                    except OSError:
                        logger.debug(
                            "failed to set exec bit on %r",
                            target,
                            exc_info=True,
                        )
                if state is not None and entry.target_hash:
                    hash_info = HashInfo.from_dict(entry.target_hash)
                    try:
                        info = fs.info(target)
                        state.save_many([(target, hash_info, info)], fs)
                    except FileNotFoundError:
                        pass

            elif entry.op == OP_CHMOD:
                if os.path.isfile(target) and entry.isexec:
                    st = os.lstat(target)
                    try:
                        os.chmod(target, st.st_mode | stat.S_IEXEC)
                    except OSError:
                        logger.debug(
                            "failed to set exec bit on %r",
                            target,
                            exc_info=True,
                        )
                journal.append(
                    EV_PUBLISHED,
                    key=list(entry.key),
                    op=entry.op,
                    post=_post_record(_lstat_safe(target)),
                )

            else:  # pragma: no cover
                raise StagedCheckoutPlanError(
                    f"unknown plan operation {entry.op!r}",
                    plan_id=plan.plan_id,
                )

            if on_op_published is not None:
                on_op_published(entry)
        except Exception as exc:
            journal.append(
                EV_PUBLISH_FAILED,
                key=list(entry.key),
                error=type(exc).__name__,
            )
            raise

    journal.append(EV_COMMITTED)
    return journal


# ---------------------------------------------------------------------------
# recovery
# ---------------------------------------------------------------------------


def _load_runtime(
    workspace: str,
    plan_id: str,
    staging_root: Optional[str],
) -> tuple[CheckoutPlan, str, Journal]:
    workspace = _ensure_safe_root(os.path.abspath(workspace))
    staging_root = os.path.abspath(staging_root or default_staging_root(workspace))
    plan = CheckoutPlan.load(staging_root, plan_id)
    if os.path.abspath(plan.workspace) != workspace:
        raise StagedCheckoutPlanError(
            f"plan workspace {plan.workspace!r} does not match {workspace!r}",
            plan_id=plan_id,
        )
    return plan, staging_root, Journal(plan.journal_path(staging_root))


def _verify_published(plan: CheckoutPlan, journal: Journal) -> None:
    published = journal.keys_for(EV_PUBLISHED)
    for entry in plan.entries:
        if entry.key not in published:
            continue
        target = plan.dest_path(entry)
        if entry.op in (OP_CREATE_FILE,):
            if not os.path.isfile(target):
                raise StagedCheckoutExternalModificationError(
                    f"published file {target!r} disappeared; stopping",
                    plan_id=plan.plan_id,
                )
            hash_info = HashInfo.from_dict(entry.target_hash or {})
            if (
                hash_info.value
                and file_md5(target, name=hash_info.name or "md5") != hash_info.value
            ):
                raise StagedCheckoutExternalModificationError(
                    f"published file {target!r} was modified externally; "
                    "stopping and preserving it",
                    plan_id=plan.plan_id,
                )
        elif entry.op == OP_MKDIR and not os.path.isdir(target):
            raise StagedCheckoutExternalModificationError(
                f"published directory {target!r} disappeared; stopping",
                plan_id=plan.plan_id,
            )


def _verify_remaining_pre_state(plan: CheckoutPlan, journal: Journal) -> None:
    published = journal.keys_for(EV_PUBLISHED)
    removed = _removed_keys(plan, published)
    for entry in plan.entries:
        if entry.key in published:
            continue
        _verify_entry_pre_state(entry, plan.dest_path(entry), plan, removed)


def resume(
    workspace: str,
    plan_id: str,
    fs: FileSystem,
    *,
    state: Optional[StateBase] = None,
    staging_root: Optional[str] = None,
) -> CheckoutPlan:
    """Continue an interrupted plan after verifying workspace contents."""
    _ensure_local_fs(fs)
    plan, staging_root, journal = _load_runtime(workspace, plan_id, staging_root)
    with WorkspaceLock(staging_root, plan_id=plan.plan_id):
        if journal.has(EV_ROLLED_BACK):
            raise StagedCheckoutPlanError(
                f"plan {plan_id!r} was rolled back", plan_id=plan_id
            )
        _verify_published(plan, journal)
        _verify_remaining_pre_state(plan, journal)
        prepare(plan, fs, staging_root, journal=journal)
        publish(plan, fs, staging_root, state=state, journal=journal)
    return plan


def rollback(  # noqa: C901, PLR0912, PLR0915
    workspace: str,
    plan_id: str,
    fs: FileSystem,
    *,
    state: Optional[StateBase] = None,
    staging_root: Optional[str] = None,
) -> CheckoutPlan:
    """Reverse every published operation, verifying content before touching."""
    _ensure_local_fs(fs)
    plan, staging_root, journal = _load_runtime(workspace, plan_id, staging_root)
    with WorkspaceLock(staging_root, plan_id=plan.plan_id):
        if journal.has(EV_ROLLED_BACK):
            return plan
        published = journal.keys_for(EV_PUBLISHED)
        if not journal.has(EV_ROLLBACK_START):
            journal.append(EV_ROLLBACK_START)

        rollback_dir = plan.rollback_dir(staging_root)
        for entry in reversed(plan.entries):
            if entry.key not in published:
                continue
            target = plan.dest_path(entry)

            if entry.op == OP_CREATE_FILE:
                if os.path.lexists(target):
                    hash_info = HashInfo.from_dict(entry.target_hash or {})
                    if hash_info.value and os.path.isfile(target):
                        actual = file_md5(target, name=hash_info.name or "md5")
                        if actual != hash_info.value:
                            raise StagedCheckoutExternalModificationError(
                                f"{target!r} was modified externally; keeping it",
                                plan_id=plan.plan_id,
                            )
                    os.unlink(target)

            elif entry.op == OP_MKDIR:
                if os.path.isdir(target):
                    try:
                        os.rmdir(target)
                    except OSError as exc:
                        if exc.errno == errno.ENOTEMPTY:
                            raise StagedCheckoutExternalModificationError(
                                f"directory {target!r} has extra contents; "
                                "stopping and preserving it",
                                plan_id=plan.plan_id,
                            ) from exc
                        raise

            elif entry.op == OP_DELETE_FILE:
                if os.path.lexists(target):
                    # Something reappeared externally; never overwrite it.
                    raise StagedCheckoutExternalModificationError(
                        f"{target!r} exists unexpectedly; stopping and preserving it",
                        plan_id=plan.plan_id,
                    )
                rollback_path = os.path.join(
                    rollback_dir, entry.rollback_rel or entry.relpath
                )
                if os.path.isfile(rollback_path):
                    parent = os.path.dirname(target)
                    if parent:
                        os.makedirs(parent, exist_ok=True)
                    os.replace(rollback_path, target)
                    if state is not None and entry.pre.hash_info:
                        old_hash = HashInfo.from_dict(entry.pre.hash_info)
                        try:
                            state.save(target, fs, old_hash, fs.info(target))
                        except FileNotFoundError:
                            pass

            elif entry.op == OP_RMDIR:
                os.makedirs(target, exist_ok=True)
                saved_root = os.path.join(
                    rollback_dir, entry.rollback_rel or entry.relpath
                )
                if os.path.isdir(saved_root):
                    for child, st in _walk_children(saved_root, staging_root):
                        if not stat.S_ISREG(st.st_mode):
                            continue
                        rel = os.path.relpath(child, saved_root)
                        dest = os.path.join(target, rel)
                        os.makedirs(os.path.dirname(dest), exist_ok=True)
                        os.replace(child, dest)

            elif (
                entry.op == OP_CHMOD
                and os.path.isfile(target)
                and entry.pre.mode is not None
            ):
                try:
                    os.chmod(target, entry.pre.mode)
                except OSError:
                    logger.debug(
                        "failed to restore mode on %r",
                        target,
                        exc_info=True,
                    )

        journal.append(EV_ROLLED_BACK)
    return plan


def cleanup(
    workspace: str,
    plan_id: str,
    *,
    staging_root: Optional[str] = None,
) -> None:
    """Remove only the given plan's staging directory.

    Staging belonging to other plans is never touched.
    """
    workspace = os.path.abspath(workspace)
    staging_root = os.path.abspath(staging_root or default_staging_root(workspace))
    plan_dir = os.path.join(staging_root, plan_id)
    if not _is_within(plan_dir, staging_root) or plan_dir == staging_root:
        raise StagedCheckoutSafetyError(
            f"refusing to clean unsafe plan path {plan_dir!r}"
        )
    shutil.rmtree(plan_dir, ignore_errors=False)


# ---------------------------------------------------------------------------
# one-shot convenience
# ---------------------------------------------------------------------------


def staged_checkout(
    old: Optional[BaseDataIndex],
    new: BaseDataIndex,
    path: str,
    fs: FileSystem,
    *,
    delete: bool = False,
    storage: str = "cache",
    staging_root: Optional[str] = None,
    state: Optional[StateBase] = None,
    plan_id: Optional[str] = None,
) -> CheckoutPlan:
    """Build, prepare and publish a plan under a single workspace lock."""
    _ensure_local_fs(fs)
    workspace = _ensure_safe_root(os.path.abspath(path))
    staging_root = os.path.abspath(staging_root or default_staging_root(workspace))
    _ensure_same_volume(workspace, staging_root)

    with WorkspaceLock(staging_root, plan_id=plan_id):
        plan = build_plan(
            old,
            new,
            workspace,
            fs,
            delete=delete,
            storage=storage,
            staging_root=staging_root,
            plan_id=plan_id,
        )
        plan.save(staging_root)
        journal = Journal(plan.journal_path(staging_root))
        journal.append(EV_PLAN_CREATED, plan_id=plan.plan_id)
        prepare(plan, fs, staging_root, journal=journal)
        publish(plan, fs, staging_root, state=state, journal=journal)
    return plan
