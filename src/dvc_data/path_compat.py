"""Declarative path compatibility planning for portable checkouts.

A :class:`PathCompatibilityPlan` maps the component paths (``tuple[str, ...]``
keys) of a :class:`~dvc_data.hashfile.tree.Tree` or a
:class:`~dvc_data.index.index.DataIndex` onto the namespace of an explicit
target :class:`Profile` and verifies that all entries can co-exist there
*before* any checkout write is attempted.

The profiles are **declared portability rule sets**.  They encode a precise,
testable subset of naming rules; they do **not** claim to fully emulate any
particular version of Windows, NTFS, or a remote filesystem (short (8.3)
names, ``MAX_PATH`` limits, reparse points, per-directory case sensitivity,
volume-specific Unicode behaviour and remote-fs semantics are intentionally
out of scope).

The planner is strictly read-only: it never touches the workspace, the object
database or any state.  The only object ever recomputed is the in-memory
digest of the remapped :class:`~dvc_data.hashfile.tree.Tree` (backed by a
memory filesystem).  File contents and their :class:`HashInfo` values are
never altered by path mapping.
"""

from __future__ import annotations

import unicodedata
from collections import defaultdict
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

import attrs
from pygtrie import Trie

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from dvc_data.hashfile.hash_info import HashInfo
    from dvc_data.hashfile.tree import Tree
    from dvc_data.index.index import BaseDataIndex

Key = tuple[str, ...]

# ---------------------------------------------------------------------------
# Profiles and normalization rules
# ---------------------------------------------------------------------------


class Profile(str, Enum):
    """Declared target namespace rules.

    ``CASE_SENSITIVE``
        Two components conflict only when they are byte-for-byte identical.
        Normalization is the identity function.

    ``CASEFOLD_NFC``
        Every component is first Unicode-normalized to NFC and then case
        folded; the folded result is NFC-normalized again.  Two components
        conflict when their normalized forms are equal, which covers both
        case aliases (``a``/``A``) and canonical Unicode aliases
        (``é``/``e + combining acute``).

    ``WINDOWS_PORTABLE``
        Uses the same NFC + casefold normalization as ``CASEFOLD_NFC`` and
        additionally enforces a precise subset of Windows naming constraints:
        separator characters, forbidden characters, reserved device names
        and trailing dot / trailing space rules.
    """

    CASE_SENSITIVE = "case-sensitive"
    CASEFOLD_NFC = "casefold-nfc"
    WINDOWS_PORTABLE = "windows-portable"


#: Reason code for an empty path component (all profiles).
REASON_EMPTY_COMPONENT = "empty-component"
#: Reason code for ``.`` / ``..`` components that would escape the root
#: (all profiles).
REASON_PATH_TRAVERSAL = "path-traversal"
#: Reason code for components that themselves contain a path separator
#: (all profiles, as keys are already split into component tuples).
REASON_SEPARATOR_IN_COMPONENT = "separator-in-component"
#: Reason code for Windows-forbidden characters ``< > : " / \\ | ? *``.
REASON_FORBIDDEN_CHARACTER = "forbidden-character"
#: Reason code for C0 control characters ``U+0000`` - ``U+001F``.
REASON_CONTROL_CHARACTER = "control-character"
#: Reason code for Windows reserved device names such as ``CON``/``nul.txt``.
REASON_RESERVED_NAME = "reserved-name"
#: Reason code for components ending with a dot.
REASON_TRAILING_DOT = "trailing-dot"
#: Reason code for components ending with a space.
REASON_TRAILING_SPACE = "trailing-space"

_WINDOWS_FORBIDDEN_CHARS = frozenset('<>:"/\\|?*')
_WINDOWS_RESERVED_BASE = frozenset(
    (
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    )
)
_STRUCTURAL_REASONS = frozenset(
    (
        REASON_EMPTY_COMPONENT,
        REASON_PATH_TRAVERSAL,
        REASON_SEPARATOR_IN_COMPONENT,
    )
)


def _nfc_casefold(part: str) -> str:
    """NFC-normalize, casefold, then NFC-normalize again.

    The final NFC pass guarantees that case folding (which may introduce
    combining marks, e.g. for some Greek/Cyrillic sequences) always lands on
    a single canonical representation, so equivalent inputs map to identical
    outputs.
    """
    return unicodedata.normalize("NFC", unicodedata.normalize("NFC", part).casefold())


def normalize_component(profile: Profile, part: str) -> str:
    """Return the canonical comparison form of one path component."""
    if profile is Profile.CASE_SENSITIVE:
        return part
    return _nfc_casefold(part)


def normalize_key(profile: Profile, key: Sequence[str]) -> Key:
    """Return the canonical comparison form of a full component key."""
    return tuple(normalize_component(profile, part) for part in key)


def _check_structural(part: str, reasons: list[str]) -> None:
    """Rules shared by every profile because keys are component tuples."""
    if part in (".", ".."):
        reasons.append(REASON_PATH_TRAVERSAL)
    if "/" in part or "\\" in part:
        reasons.append(REASON_SEPARATOR_IN_COMPONENT)


def _check_windows_portable(part: str, reasons: list[str]) -> None:
    """Declared Windows naming subset (not a full NTFS emulation)."""
    if any(ord(ch) <= 0x1F for ch in part):
        reasons.append(REASON_CONTROL_CHARACTER)
    # Separators are reported separately; only flag the remaining Windows set.
    non_separator_chars = _WINDOWS_FORBIDDEN_CHARS.difference(("/", "\\"))
    if any(ch in non_separator_chars for ch in part):
        reasons.append(REASON_FORBIDDEN_CHARACTER)
    if part[-1] == " ":
        reasons.append(REASON_TRAILING_SPACE)
    if part[-1] == ".":
        reasons.append(REASON_TRAILING_DOT)
    # Reserved device names match case-insensitively on the stem before the
    # first dot, so e.g. ``CON``, ``con.txt`` and ``NUL.log`` all qualify.
    stem = part.split(".", 1)[0].casefold()
    if stem.upper() in _WINDOWS_RESERVED_BASE:
        reasons.append(REASON_RESERVED_NAME)


def check_component(profile: Profile, part: str) -> tuple[str, ...]:
    """Validate a single path component under ``profile``.

    Returns a tuple of precise reason codes; an empty tuple means the
    component satisfies the profile.
    """
    if part == "":
        return (REASON_EMPTY_COMPONENT,)

    reasons: list[str] = []
    _check_structural(part, reasons)
    if profile is Profile.WINDOWS_PORTABLE:
        _check_windows_portable(part, reasons)
    return tuple(dict.fromkeys(reasons))


def check_key(profile: Profile, key: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    """Return per-component violation reason tuples for ``key``."""
    return tuple(check_component(profile, part) for part in key)


# ---------------------------------------------------------------------------
# Errors, conflicts and result objects
# ---------------------------------------------------------------------------


class ConflictKind(str, Enum):
    """Classification of a reported conflict group."""

    #: Distinct source paths normalize to the same target key.
    NAME_COLLISION = "name-collision"
    #: A file entry is a strict prefix of another entry.
    PARENT_CHILD = "parent-child"
    #: The same key is used by a file and a directory.
    TYPE_MISMATCH = "type-mismatch"
    #: Differently-spelled directory components merge into one directory.
    CASE_DIRECTORY_MERGE = "case-directory-merge"
    #: A source component violates a profile constraint.
    CONSTRAINT_VIOLATION = "constraint-violation"
    #: A planned target would clobber an unselected existing workspace path.
    WORKSPACE_OVERWRITE = "workspace-overwrite"
    #: The explicit rename map makes two entries land on the same target.
    RENAME_COLLISION = "rename-collision"
    #: A rename target is structurally invalid or escapes the checkout root.
    RENAME_ESCAPE = "rename-escape"


#: Kinds that prevent a plan from being applied safely.
ERROR_KINDS = frozenset(
    (
        ConflictKind.NAME_COLLISION,
        ConflictKind.PARENT_CHILD,
        ConflictKind.TYPE_MISMATCH,
        ConflictKind.CONSTRAINT_VIOLATION,
        ConflictKind.WORKSPACE_OVERWRITE,
        ConflictKind.RENAME_COLLISION,
        ConflictKind.RENAME_ESCAPE,
    )
)
#: Kinds that are reported but do not by themselves block co-existence.
WARNING_KINDS = frozenset((ConflictKind.CASE_DIRECTORY_MERGE,))

ORIGIN_TREE = "tree"
ORIGIN_INDEX = "index"
ORIGIN_WORKSPACE = "workspace"

SCOPE_OBSERVED = "observed"
SCOPE_UNKNOWN = "unknown"


class PathCompatibilityError(Exception):
    """Raised when a plan contains blocking conflict groups.

    The complete, untruncated set of conflict groups is attached as
    :attr:`conflicts` (truncation only affects the reported result object,
    never the exception payload).
    """

    def __init__(
        self,
        conflicts: Sequence[ConflictGroup],
        message: Optional[str] = None,
    ):
        self.conflicts = list(conflicts)
        if message is None:
            message = (
                f"path compatibility plan failed with "
                f"{len(self.conflicts)} blocking conflict group(s)"
            )
        super().__init__(message)


class PathPlanLimitError(PathCompatibilityError):
    """Raised when an input exceeds a declared :class:`PlanLimits` cap."""


class RenameMapError(PathCompatibilityError):
    """Raised when the caller-supplied rename map is malformed."""


@attrs.define(frozen=True)
class PlanLimits:
    """Resource caps for read-only planning.

    Exceeding any of ``max_paths``, ``max_total_bytes`` or ``max_depth``
    aborts planning with :class:`PathPlanLimitError` before any result is
    produced.  ``max_conflict_groups`` only caps the result report (the
    ``conflicts_truncated`` flag is set when it is hit); an exception still
    carries the full conflict set.
    """

    max_paths: int = 100_000
    max_total_bytes: int = 10 * 1024**4
    max_depth: int = 4096
    max_conflict_groups: int = 1_000


@attrs.define(frozen=True)
class PlanStats:
    n_paths: int
    n_files: int
    n_dirs: int
    total_bytes: int
    n_unknown_size: int
    max_depth: int


@attrs.define(frozen=True)
class PlanEntry:
    """A single planned path.

    ``source`` is the original key, ``target`` the key after applying the
    rename map.  ``hash_info`` is the unchanged object identity (``None`` for
    observed workspace entries).
    """

    source: Key
    target: Key
    isdir: bool
    origin: str
    size: Optional[int] = None
    hash_info: Optional[HashInfo] = None

    @property
    def renamed(self) -> bool:
        return self.source != self.target


@attrs.define(frozen=True)
class ConflictGroup:
    """A complete set of original paths participating in one conflict.

    Paths are always the *original* source keys (never normalized forms and
    never rename targets), sorted for deterministic reporting.
    """

    kind: ConflictKind
    paths: tuple[Key, ...]
    severity: str
    normalized_key: Optional[Key] = None
    detail: str = ""
    reasons: tuple[str, ...] = ()

    @property
    def is_error(self) -> bool:
        return self.kind in ERROR_KINDS


@attrs.define
class PlanResult:
    profile: Profile
    entries: list[PlanEntry]
    workspace_entries: list[PlanEntry]
    conflicts: list[ConflictGroup]
    stats: PlanStats
    workspace_scope: str
    conflicts_truncated: bool
    rename_map: Mapping[Key, Key]
    _targets: Mapping[Key, Key] = attrs.field(repr=False, eq=False)

    @property
    def error_conflicts(self) -> list[ConflictGroup]:
        return [group for group in self.conflicts if group.is_error]

    @property
    def warning_conflicts(self) -> list[ConflictGroup]:
        return [group for group in self.conflicts if not group.is_error]

    @property
    def is_valid(self) -> bool:
        """``True`` when no blocking conflict group was detected."""
        return not self.error_conflicts

    @property
    def has_unknown_scope(self) -> bool:
        """``True`` when the workspace could not be observed in full.

        An unknown scope means workspace safety is *not* certified; it must
        never be interpreted as "no workspace conflict".
        """
        return self.workspace_scope == SCOPE_UNKNOWN

    @property
    def workspace_certified(self) -> bool:
        return self.is_valid and not self.has_unknown_scope

    def target_for(self, source_key: Key) -> Key:
        """Return the planned target key for an original source key."""
        return self._targets[source_key]

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly representation (keys joined with ``/``)."""

        def _join(key: Key) -> str:
            return "/".join(key)

        return {
            "profile": self.profile.value,
            "workspace_scope": self.workspace_scope,
            "conflicts_truncated": self.conflicts_truncated,
            "is_valid": self.is_valid,
            "stats": attrs.asdict(self.stats),
            "entries": [
                {
                    "source": _join(entry.source),
                    "target": _join(entry.target),
                    "isdir": entry.isdir,
                    "origin": entry.origin,
                }
                for entry in self.entries
            ],
            "conflicts": [
                {
                    "kind": group.kind.value,
                    "severity": group.severity,
                    "normalized_key": (
                        _join(group.normalized_key)
                        if group.normalized_key is not None
                        else None
                    ),
                    "paths": [_join(path) for path in group.paths],
                    "reasons": list(group.reasons),
                    "detail": group.detail,
                }
                for group in self.conflicts
            ],
        }


# ---------------------------------------------------------------------------
# Internal raw input shape
# ---------------------------------------------------------------------------


@attrs.define(frozen=True)
class _RawEntry:
    key: Key
    isdir: bool
    size: Optional[int]
    hash_info: Optional[HashInfo]
    origin: str


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class PathCompatibilityPlan:
    """Plan and validate path mapping for a portable checkout.

    Construct one via :meth:`from_tree` / :meth:`from_index` (or directly with
    :class:`_RawEntry`-like tuples) and inspect :attr:`result`.  A valid plan
    can materialize a remapped :class:`~dvc_data.hashfile.tree.Tree` via
    :meth:`mapped_tree`; the new tree digest is recomputed from the new paths
    while every file :class:`HashInfo` is preserved.
    """

    def __init__(
        self,
        profile: Profile,
        raw_entries: Iterable[_RawEntry],
        *,
        rename_map: Optional[Mapping[Key, Key]] = None,
        workspace_entries: Optional[Iterable[tuple[Key, bool]]] = None,
        workspace_observed: bool = True,
        limits: Optional[PlanLimits] = None,
    ) -> None:
        self.profile = profile
        self._raw_entries = list(raw_entries)
        self._rename_map = dict(rename_map or {})
        self._workspace_input = list(workspace_entries or [])
        self._workspace_observed = workspace_observed
        self.limits = limits or PlanLimits()
        self._all_conflicts: list[ConflictGroup] = []
        self.result: PlanResult = self._plan()

    # -- constructors -----------------------------------------------------

    @classmethod
    def from_tree(
        cls,
        tree: Tree,
        profile: Profile,
        *,
        rename_map: Optional[Mapping[Key, Key]] = None,
        workspace_entries: Optional[Iterable[tuple[Key, bool]]] = None,
        workspace_observed: bool = True,
        limits: Optional[PlanLimits] = None,
    ) -> PathCompatibilityPlan:
        raw: list[_RawEntry] = []
        for key, meta, hash_info in tree:
            isdir = bool(meta is not None and meta.isdir) or bool(
                hash_info is not None and hash_info.isdir
            )
            size = meta.size if meta is not None else None
            raw.append(
                _RawEntry(
                    key=tuple(key),
                    isdir=isdir,
                    size=size,
                    hash_info=hash_info,
                    origin=ORIGIN_TREE,
                )
            )
        return cls(
            profile,
            raw,
            rename_map=rename_map,
            workspace_entries=workspace_entries,
            workspace_observed=workspace_observed,
            limits=limits,
        )

    @classmethod
    def from_index(
        cls,
        index: BaseDataIndex,
        profile: Profile,
        *,
        rename_map: Optional[Mapping[Key, Key]] = None,
        workspace_entries: Optional[Iterable[tuple[Key, bool]]] = None,
        workspace_observed: bool = True,
        limits: Optional[PlanLimits] = None,
    ) -> PathCompatibilityPlan:
        raw: list[_RawEntry] = []
        for key, entry in index.iteritems():
            isdir = bool(entry.meta is not None and entry.meta.isdir) or bool(
                entry.hash_info is not None and entry.hash_info.isdir
            )
            size = entry.meta.size if entry.meta is not None else None
            raw.append(
                _RawEntry(
                    key=tuple(key),
                    isdir=isdir,
                    size=size,
                    hash_info=entry.hash_info,
                    origin=ORIGIN_INDEX,
                )
            )
        return cls(
            profile,
            raw,
            rename_map=rename_map,
            workspace_entries=workspace_entries,
            workspace_observed=workspace_observed,
            limits=limits,
        )

    # -- public API -------------------------------------------------------

    def raise_for_conflicts(self) -> None:
        """Raise :class:`PathCompatibilityError` on blocking groups.

        The exception carries the complete conflict set even when the result
        report was truncated by :attr:`PlanLimits.max_conflict_groups`.
        """
        errors = [group for group in self._all_conflicts if group.is_error]
        if errors:
            raise PathCompatibilityError(errors)

    def mapped_tree(self, tree: Tree) -> Tree:
        """Return a new ``Tree`` with entries re-keyed to planned targets.

        File :class:`HashInfo` and meta objects are shared unchanged; the
        directory digest is recomputed from the *new* paths via
        :meth:`Tree.digest`, so the old ``.dir`` hash is never reused.  The
        input tree is not mutated.
        """
        from dvc_data.hashfile.tree import Tree

        self.raise_for_conflicts()
        targets = self.result._targets
        mapped = Tree()
        for key, meta, hash_info in tree:
            source = tuple(key)
            if source not in targets:
                raise PathCompatibilityError(
                    [],
                    f"tree path {source!r} is not part of the compatibility plan",
                )
            mapped.add(targets[source], meta, hash_info)
        mapped.digest()
        return mapped

    def mapped_index(self, index: BaseDataIndex):
        """Return a new :class:`DataIndex` re-keyed to planned targets.

        Entry meta and ``hash_info`` are preserved unchanged.  The storage
        map is deep-copied best-effort; callers that reconfigure storages
        after planning must re-attach them to the returned index.
        """
        import copy

        from dvc_data.index.index import DataIndex, DataIndexEntry

        self.raise_for_conflicts()
        targets = self.result._targets
        mapped = DataIndex()
        for key, entry in index.iteritems():
            source = tuple(key)
            mapped.add(
                DataIndexEntry(
                    key=targets[source],
                    meta=entry.meta,
                    hash_info=entry.hash_info,
                    loaded=entry.loaded,
                )
            )
        mapped.storage_map = copy.deepcopy(index.storage_map)
        return mapped

    # -- core planning ----------------------------------------------------

    def _plan(self) -> PlanResult:
        limits = self.limits
        raw = self._raw_entries

        known_sources = {entry.key for entry in raw}
        self._validate_rename_map(known_sources)

        planned: list[PlanEntry] = []
        workspace: list[PlanEntry] = []
        conflicts: list[ConflictGroup] = []

        total_bytes = 0
        n_unknown_size = 0
        n_files = 0
        n_dirs = 0
        max_depth = 0

        # Ingest planned entries while enforcing resource caps.
        for number, entry in enumerate(raw):
            if number >= limits.max_paths:
                raise PathPlanLimitError(
                    [], f"path count exceeds limit of {limits.max_paths}"
                )
            if len(entry.key) > limits.max_depth:
                raise PathPlanLimitError(
                    [],
                    f"path depth {len(entry.key)} exceeds limit of {limits.max_depth}",
                )
            if entry.isdir:
                n_dirs += 1
            else:
                n_files += 1
            if entry.size is None:
                if not entry.isdir:
                    n_unknown_size += 1
            else:
                total_bytes += entry.size
                if total_bytes > limits.max_total_bytes:
                    raise PathPlanLimitError(
                        [],
                        f"total size exceeds limit of {limits.max_total_bytes} bytes",
                    )
            max_depth = max(max_depth, len(entry.key))

            target = self._rename_map.get(entry.key, entry.key)
            planned.append(
                PlanEntry(
                    source=entry.key,
                    target=target,
                    isdir=entry.isdir,
                    origin=entry.origin,
                    size=entry.size,
                    hash_info=entry.hash_info,
                )
            )
            self._check_target_constraints(entry.key, target, conflicts)

        # Observed workspace paths join the same normalization domain.
        for key, isdir in self._workspace_input:
            workspace.append(
                PlanEntry(
                    source=tuple(key),
                    target=tuple(key),
                    isdir=isdir,
                    origin=ORIGIN_WORKSPACE,
                )
            )

        self._detect_namespace_conflicts(planned, workspace, conflicts)

        ordered = sorted(
            conflicts,
            key=lambda group: (
                group.kind.value,
                group.normalized_key or (),
                group.paths,
            ),
        )
        # Keep the full, untruncated set for exception payloads; only the
        # result object is capped.
        self._all_conflicts = ordered
        truncated = len(ordered) > limits.max_conflict_groups
        reported = ordered[: limits.max_conflict_groups]

        stats = PlanStats(
            n_paths=len(planned),
            n_files=n_files,
            n_dirs=n_dirs,
            total_bytes=total_bytes,
            n_unknown_size=n_unknown_size,
            max_depth=max_depth,
        )
        targets = {entry.source: entry.target for entry in planned}
        return PlanResult(
            profile=self.profile,
            entries=planned,
            workspace_entries=workspace,
            conflicts=reported,
            stats=stats,
            workspace_scope=(
                SCOPE_OBSERVED if self._workspace_observed else SCOPE_UNKNOWN
            ),
            conflicts_truncated=truncated,
            rename_map=dict(self._rename_map),
            targets=targets,
        )

    def _validate_rename_map(self, known_sources: set[Key]) -> None:
        for source, target in self._rename_map.items():
            if not isinstance(source, tuple) or not all(
                isinstance(part, str) for part in source
            ):
                raise RenameMapError(
                    [], f"rename map key must be a str tuple: {source!r}"
                )
            if not isinstance(target, tuple) or not all(
                isinstance(part, str) for part in target
            ):
                raise RenameMapError(
                    [], f"rename map target must be a str tuple: {target!r}"
                )
            if source not in known_sources:
                raise RenameMapError(
                    [], f"rename map references unknown source path {source!r}"
                )
            if len(target) == 0:
                raise RenameMapError(
                    [], f"rename target for {source!r} must not be the checkout root"
                )

    def _check_target_constraints(
        self, source: Key, target: Key, conflicts: list[ConflictGroup]
    ) -> None:
        renamed = source != target
        for part, reasons in zip(target, check_key(self.profile, target)):
            if not reasons:
                continue
            kind = (
                ConflictKind.RENAME_ESCAPE
                if renamed
                else ConflictKind.CONSTRAINT_VIOLATION
            )
            conflicts.append(
                ConflictGroup(
                    kind=kind,
                    paths=(source,),
                    severity="error",
                    normalized_key=normalize_key(self.profile, target),
                    detail=(
                        f"component {part!r} violates profile "
                        f"{self.profile.value}: {', '.join(reasons)}"
                    ),
                    reasons=reasons,
                )
            )

    def _detect_namespace_conflicts(
        self,
        planned: Sequence[PlanEntry],
        workspace: Sequence[PlanEntry],
        conflicts: list[ConflictGroup],
    ) -> None:
        # Normalized trie: norm key -> all entries occupying that key.
        trie: Trie = Trie()
        # (parent norm prefix, norm component) -> {raw spelling: [full keys]}
        aliases: dict[tuple[Key, str], dict[str, list[Key]]] = defaultdict(
            lambda: defaultdict(list)
        )

        all_entries = [*planned, *workspace]
        for entry in all_entries:
            norm = normalize_key(self.profile, entry.target)
            bucket = trie.get(norm)
            if bucket is None:
                bucket = []
                trie[norm] = bucket
            bucket.append(entry)
            self._register_directory_aliases(entry, aliases)

        # Exact-key collisions and type mismatches.
        for norm, bucket in trie.items():
            self._classify_exact_group(norm, bucket, conflicts)

        self._detect_prefix_conflicts(trie, all_entries, conflicts)
        self._detect_alias_conflicts(aliases, conflicts)

    def _detect_prefix_conflicts(
        self,
        trie: Trie,
        all_entries: Sequence[PlanEntry],
        conflicts: list[ConflictGroup],
    ) -> None:
        # File/descendant (prefix) conflicts.
        for entry in all_entries:
            if entry.isdir:
                continue
            norm = normalize_key(self.profile, entry.target)
            descendants: list[PlanEntry] = []
            for other_norm, other_bucket in trie.items(prefix=norm):
                if other_norm == norm or other_norm[: len(norm)] != norm:
                    continue
                descendants.extend(other_bucket)
            if descendants:
                paths = sorted({entry.source, *(other.source for other in descendants)})
                conflicts.append(
                    ConflictGroup(
                        kind=ConflictKind.PARENT_CHILD,
                        paths=tuple(paths),
                        severity="error",
                        normalized_key=norm,
                        detail=(
                            f"file {entry.source!r} is a prefix of "
                            f"{len(descendants)} descendant entry/entries"
                        ),
                    )
                )

    def _detect_alias_conflicts(
        self,
        aliases: dict[tuple[Key, str], dict[str, list[Key]]],
        conflicts: list[ConflictGroup],
    ) -> None:
        # Case / Unicode directory alias merges (warning only).
        for (parent_norm, norm_part), spellings in aliases.items():
            if len(spellings) <= 1:
                continue
            dir_paths = sorted({path for keys in spellings.values() for path in keys})
            conflicts.append(
                ConflictGroup(
                    kind=ConflictKind.CASE_DIRECTORY_MERGE,
                    paths=tuple(dir_paths),
                    severity="warning",
                    normalized_key=(*parent_norm, norm_part),
                    detail=(
                        "directory components "
                        f"{sorted(spellings)!r} normalize to the same name"
                    ),
                )
            )

    def _register_directory_aliases(
        self,
        entry: PlanEntry,
        aliases: dict[tuple[Key, str], dict[str, list[Key]]],
    ) -> None:
        target = entry.target
        norm_parts = normalize_key(self.profile, target)
        # Every strict prefix of an entry key is a directory component.
        for depth in range(1, len(target)):
            parent_norm = norm_parts[: depth - 1]
            raw_part = target[depth - 1]
            norm_part = norm_parts[depth - 1]
            aliases[(parent_norm, norm_part)][raw_part].append(target[:depth])
        # Explicit directory entries also register their own last component.
        if entry.isdir:
            parent_norm = norm_parts[:-1]
            aliases[(parent_norm, norm_parts[-1])][target[-1]].append(target)

    def _classify_exact_group(
        self,
        norm: Key,
        bucket: list[PlanEntry],
        conflicts: list[ConflictGroup],
    ) -> None:
        if len(bucket) == 1:
            return

        planned_entries = [e for e in bucket if e.origin != ORIGIN_WORKSPACE]
        workspace_entries = [e for e in bucket if e.origin == ORIGIN_WORKSPACE]

        # An intended overwrite: a planned entry re-uses the exact raw key of
        # an observed workspace entry and nothing else occupies the slot.
        if workspace_entries and len(planned_entries) == 1:
            planned_key = planned_entries[0].source
            if all(w.source == planned_key for w in workspace_entries):
                return

        paths = tuple(sorted({entry.source for entry in bucket}))
        has_renamed = any(entry.renamed for entry in planned_entries)
        isdir_values = {entry.isdir for entry in bucket}

        if workspace_entries:
            kind = ConflictKind.WORKSPACE_OVERWRITE
            detail = (
                "planned target collides with unselected existing "
                f"workspace path(s) {[w.source for w in workspace_entries]!r}"
            )
        elif len(isdir_values) > 1:
            kind = ConflictKind.TYPE_MISMATCH
            detail = f"same normalized key {norm!r} used by file and directory"
        elif has_renamed:
            kind = ConflictKind.RENAME_COLLISION
            detail = "rename map sends multiple entries to the same target"
        else:
            kind = ConflictKind.NAME_COLLISION
            detail = f"distinct source paths normalize to the same key {norm!r}"

        reasons: list[str] = []
        hash_infos = {
            entry.hash_info for entry in planned_entries if entry.hash_info is not None
        }
        if (
            kind is ConflictKind.NAME_COLLISION
            and len(planned_entries) == len(bucket)
            and len(hash_infos) == 1
        ):
            reasons.append("identical-content")

        conflicts.append(
            ConflictGroup(
                kind=kind,
                paths=paths,
                severity="error",
                normalized_key=norm,
                detail=detail,
                reasons=tuple(reasons),
            )
        )
