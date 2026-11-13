"""Read-only reachability audit for HashFile object graphs.

Given a finite set of root :class:`HashInfo` objects, :class:`ExplainReachability`
walks the directed graph of directory (``.dir``) objects and their entries down
to leaf objects using BFS, and explains *why* every reachable object is or is
not usable:

* missing objects,
* malformed objects (bad directory JSON / unsupported hash / empty entries),
* hash mismatches (``verify`` mode only, real content hashing),
* objects that were discovered but never checked (e.g. budget exhaustion).

Every problem carries a stable *shortest* ``root -> directory entry`` witness.
The audit never repairs, deletes or caches anything: content verification uses
the same streaming hashers as :meth:`HashFileDB.check`
(:func:`dvc_data.hashfile.hash.get_hash_stream`) without its destructive
side effects (no file removal, no state-cache writes).
"""

import heapq
import json
import logging
from collections import Counter
from collections.abc import Iterable
from io import BytesIO
from itertools import count
from typing import TYPE_CHECKING, Optional

from attrs import define, field
from dvc_objects.errors import ObjectFormatError

from .hash import get_hash_stream
from .hash_info import HashInfo
from .tree import Tree

if TYPE_CHECKING:
    from .db import HashFileDB

logger = logging.getLogger(__name__)

METADATA = "metadata"
VERIFY = "verify"
_MODES = (METADATA, VERIFY)

# Object states.
OK = "ok"
MISSING = "missing"
MALFORMED = "malformed"
MISMATCHED = "mismatched"
UNCHECKED = "unchecked"

_WHITE, _GRAY, _BLACK = 0, 1, 2


class ReachabilityError(Exception):
    pass


class _BudgetExhaustedError(Exception):
    """Internal control-flow signal for an early, budgeted stop."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@define(frozen=True)
class Budget:
    """Global walk limits.

    Any ``None`` limit means unlimited. When a limit is reached the audit stops
    early, reports :attr:`ReachabilityReport.incomplete` with the precise
    reason and keeps an accurate record of which objects were actually checked.
    """

    max_objects: Optional[int] = None
    max_edges: Optional[int] = None
    max_bytes: Optional[int] = None
    max_witnesses: Optional[int] = None


@define(frozen=True)
class Witness:
    """Shortest known path from a root to an object or graph edge.

    ``path`` is the concatenation of directory entry names (``relpath`` parts)
    leading from ``root`` to the referenced object. It is ``()`` for a root
    object itself.
    """

    root: HashInfo
    path: tuple[str, ...]

    def as_posix(self) -> str:
        return "/".join(self.path)


@define
class ObjectStatus:
    hash_info: HashInfo
    state: str = UNCHECKED
    witness: Optional[Witness] = None
    references: int = 0
    sources: tuple[str, ...] = ()
    actual: Optional[str] = None
    detail: Optional[str] = None


@define(frozen=True)
class CycleIssue:
    """A directory back-edge (``source -> target``) closing a cycle."""

    witness: Witness
    source: HashInfo
    target: HashInfo


@define(frozen=True)
class ContentConflict:
    """Same OID presenting different/unusable content in a fallback ODB."""

    hash_info: HashInfo
    source: str
    kind: str
    actual: Optional[str] = None
    detail: Optional[str] = None
    witness: Optional[Witness] = None


@define(frozen=True)
class MalformedEntry:
    """A directory entry without a usable HashInfo."""

    witness: Witness
    detail: str


@define
class ReachabilityStats:
    roots: int = 0
    objects_seen: int = 0
    objects_checked: int = 0
    edges_traversed: int = 0
    bytes_read: int = 0
    witnesses: int = 0
    witnesses_truncated: bool = False
    limits: Budget = field(factory=Budget)


@define
class ReachabilityReport:
    mode: str
    roots: tuple[HashInfo, ...]
    objects: dict[HashInfo, ObjectStatus] = field(factory=dict)
    cycles: list[CycleIssue] = field(factory=list)
    conflicts: list[ContentConflict] = field(factory=list)
    malformed_entries: list[MalformedEntry] = field(factory=list)
    unknown_subtrees: list[HashInfo] = field(factory=list)
    invalid_roots: list[HashInfo] = field(factory=list)
    incomplete_reasons: list[str] = field(factory=list)
    stats: ReachabilityStats = field(factory=ReachabilityStats)
    odb_labels: tuple[str, ...] = ()
    include_paths: bool = False

    def _by_state(self, state: str) -> tuple[ObjectStatus, ...]:
        return tuple(
            sorted(
                (s for s in self.objects.values() if s.state == state),
                key=lambda s: _hi_key(s.hash_info),
            )
        )

    @property
    def ok_objects(self) -> tuple[ObjectStatus, ...]:
        return self._by_state(OK)

    @property
    def missing(self) -> tuple[ObjectStatus, ...]:
        return self._by_state(MISSING)

    @property
    def malformed(self) -> tuple[ObjectStatus, ...]:
        return self._by_state(MALFORMED)

    @property
    def mismatched(self) -> tuple[ObjectStatus, ...]:
        return self._by_state(MISMATCHED)

    @property
    def unchecked(self) -> tuple[ObjectStatus, ...]:
        return self._by_state(UNCHECKED)

    @property
    def checked(self) -> tuple[ObjectStatus, ...]:
        """Exactly the objects the walk fully processed, regardless of state."""
        return tuple(
            sorted(
                (s for s in self.objects.values() if s.state != UNCHECKED),
                key=lambda s: _hi_key(s.hash_info),
            )
        )

    @property
    def incomplete(self) -> bool:
        return bool(self.incomplete_reasons)

    @property
    def complete_closure(self) -> bool:
        """True only if every reachable object could be enumerated/checked.

        A False result means the seen leaves must never be treated as the full
        closure (e.g. as GC candidates): an unloadable directory hides all of
        its descendants, and budget exhaustion leaves referenced objects
        unchecked.
        """
        return not self.incomplete_reasons and not self.unknown_subtrees

    @property
    def ok(self) -> bool:
        """Overall pass flag. Never true while incomplete or problematic."""
        return (
            self.complete_closure
            and not self.cycles
            and not self.conflicts
            and not self.malformed_entries
            and not self.invalid_roots
            and all(s.state == OK for s in self.objects.values())
        )

    def to_dict(self) -> dict:
        """JSON-safe representation.

        By default only ODB indices and filesystem protocols are emitted,
        never absolute paths or credentials. Paths are only included when the
        audit was explicitly created with ``include_paths=True``.
        """

        def _witness(w: Optional[Witness]) -> Optional[dict]:
            if w is None:
                return None
            return {
                "root": str(w.root),
                "path": list(w.path),
                "path_posix": w.as_posix(),
            }

        def _status(s: ObjectStatus) -> dict:
            return {
                "hash_info": str(s.hash_info),
                "state": s.state,
                "witness": _witness(s.witness),
                "references": s.references,
                "sources": list(s.sources),
                "actual": s.actual,
                "detail": s.detail,
            }

        return {
            "mode": self.mode,
            "complete_closure": self.complete_closure,
            "ok": self.ok,
            "incomplete": self.incomplete,
            "incomplete_reasons": list(self.incomplete_reasons),
            "odb_sources": list(self.odb_labels),
            "roots": [str(r) for r in self.roots],
            "objects": [_status(s) for s in self._sorted_statuses()],
            "unknown_subtrees": [str(h) for h in self.unknown_subtrees],
            "invalid_roots": [str(h) for h in self.invalid_roots],
            "cycles": [
                {
                    "witness": _witness(c.witness),
                    "source": str(c.source),
                    "target": str(c.target),
                }
                for c in self.cycles
            ],
            "conflicts": [
                {
                    "hash_info": str(c.hash_info),
                    "source": c.source,
                    "kind": c.kind,
                    "actual": c.actual,
                    "detail": c.detail,
                    "witness": _witness(c.witness),
                }
                for c in self.conflicts
            ],
            "malformed_entries": [
                {"witness": _witness(m.witness), "detail": m.detail}
                for m in self.malformed_entries
            ],
            "stats": {
                "roots": self.stats.roots,
                "objects_seen": self.stats.objects_seen,
                "objects_checked": self.stats.objects_checked,
                "edges_traversed": self.stats.edges_traversed,
                "bytes_read": self.stats.bytes_read,
                "witnesses": self.stats.witnesses,
                "witnesses_truncated": self.stats.witnesses_truncated,
                "limits": {
                    "max_objects": self.stats.limits.max_objects,
                    "max_edges": self.stats.limits.max_edges,
                    "max_bytes": self.stats.limits.max_bytes,
                    "max_witnesses": self.stats.limits.max_witnesses,
                },
            },
        }

    def _sorted_statuses(self) -> list[ObjectStatus]:
        return sorted(self.objects.values(), key=lambda s: _hi_key(s.hash_info))


def _hi_key(hi: HashInfo) -> tuple[str, str]:
    return hi.name or "", hi.value or ""


class ExplainReachability:
    """BFS reachability auditor over one or more fallback ODBs.

    Args:
        odbs: ordered fallback ODBs; the first ODB holding a usable copy of an
            OID wins, and per-ODB failures are recorded as conflicts.
        mode: ``"metadata"`` (existence/structure only) or ``"verify"``
            (real content hashing for every object).
        budget: global object/edge/byte/witness limits.
        labels: optional explicit ODB labels; defaults to safe
            ``odb[i]:<protocol>`` labels without paths or credentials.
        include_paths: include ODB paths in serialized output (off by default;
            credentials are never emitted).
    """

    def __init__(
        self,
        odbs: Iterable["HashFileDB"],
        *,
        mode: str = METADATA,
        budget: Optional[Budget] = None,
        labels: Optional[Iterable[str]] = None,
        include_paths: bool = False,
    ) -> None:
        self.odbs = list(odbs)
        if not self.odbs:
            raise ReachabilityError("at least one ODB is required")
        if mode not in _MODES:
            raise ReachabilityError(f"unknown mode {mode!r}, expected one of {_MODES}")
        self.mode = mode
        self.budget = budget or Budget()
        self.include_paths = include_paths

        labels = list(labels) if labels is not None else []
        self.labels: list[str] = []
        for i, odb in enumerate(self.odbs):
            if i < len(labels) and labels[i]:
                self.labels.append(labels[i])
            else:
                protocol = getattr(odb.fs, "protocol", "unknown")
                label = f"odb[{i}]:{protocol}"
                if include_paths:
                    label = f"{label}:{odb.path}"
                self.labels.append(label)

        # Per-audit mutable state (reset on every explain() call).
        self._report: Optional[ReachabilityReport] = None
        self._statuses: dict[HashInfo, ObjectStatus] = {}
        # Finalized shortest witness for processed objects.
        self._witnesses: dict[HashInfo, Witness] = {}
        # Shortest known witness for every enqueued object.
        self._known: dict[HashInfo, Witness] = {}
        self._processed: set[HashInfo] = set()
        self._heap: list[tuple] = []
        self._seq = count()
        self._refs: Counter[HashInfo] = Counter()
        self._adjacency: dict[HashInfo, list[tuple[tuple[str, ...], HashInfo]]] = {}
        self._dir_objects: set[HashInfo] = set()
        self._bytes_read = 0
        self._edges = 0
        self._objects_checked = 0
        self._witness_count = 0
        self._witnesses_truncated = False
        self._stopped = False

    def explain(self, roots: Iterable[HashInfo]) -> ReachabilityReport:
        roots = self._normalize_roots(roots)
        self._reset(roots)

        for root in roots:
            if not root.value or not root.name:
                self._report.invalid_roots.append(root)  # type: ignore[union-attr]
                continue
            self._refs[root] += 1
            self._enqueue(root, Witness(root, ()))

        self._walk()
        self._detect_cycles()
        self._finalize()
        assert self._report is not None
        return self._report

    # ------------------------------------------------------------------ #
    # setup / normalization
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_roots(roots: Iterable[HashInfo]) -> tuple[HashInfo, ...]:
        # HashInfo equality covers (name, value), so md5:x and sha256:x stay
        # distinct objects while exact duplicates collapse.
        unique = set(roots)
        return tuple(sorted(unique, key=_hi_key))

    def _reset(self, roots: tuple[HashInfo, ...]) -> None:
        self._statuses = {}
        self._witnesses = {}
        self._known = {}
        self._processed = set()
        self._heap = []
        self._seq = count()
        self._refs = Counter()
        self._adjacency = {}
        self._dir_objects = set()
        self._bytes_read = 0
        self._edges = 0
        self._objects_checked = 0
        self._witness_count = 0
        self._witnesses_truncated = False
        self._stopped = False
        self._report = ReachabilityReport(
            mode=self.mode,
            roots=roots,
            odb_labels=tuple(self.labels),
            include_paths=self.include_paths,
            stats=ReachabilityStats(roots=len(roots), limits=self.budget),
        )

    # ------------------------------------------------------------------ #
    # shortest-path walk (Dijkstra over positive entry-path weights; the
    # weight of an edge is the number of entry-name parts it spans, which
    # makes it equivalent to BFS for flat single-name trees)
    # ------------------------------------------------------------------ #

    def _enqueue(self, hi: HashInfo, witness: Witness) -> None:
        previous = self._known.get(hi)
        if previous is None or len(witness.path) < len(previous.path):
            self._known[hi] = witness
            self._statuses.setdefault(
                hi, ObjectStatus(hash_info=hi, state=UNCHECKED, witness=witness)
            )
        heapq.heappush(
            self._heap,
            (
                len(witness.path),
                witness.root.name or "",
                witness.root.value or "",
                witness.path,
                next(self._seq),
                witness,
                hi,
            ),
        )

    def _walk(self) -> None:
        while self._heap and not self._stopped:
            *_, witness, hi = heapq.heappop(self._heap)
            if hi in self._processed:
                # Stale entry: the object was already reached via a shorter
                # or equally short but deterministically ordered witness.
                continue
            try:
                self._charge_object()
                self._witnesses[hi] = witness
                # sync the finalized shortest witness onto the status
                self._statuses[hi].witness = witness
                self._process(hi)
            except _BudgetExhaustedError as exc:
                # The object was not fully processed (e.g. its children could
                # not all be enumerated): keep it honest as UNCHECKED.
                self._statuses[hi].state = UNCHECKED
                self._stop(exc.reason)
                break
            self._processed.add(hi)
            self._objects_checked += 1
        # Objects discovered but never finalized stay UNCHECKED, keeping the
        # shortest witness known at the time of the stop.
        for hi, witness in self._known.items():
            if hi in self._processed:
                continue
            status = self._statuses.setdefault(
                hi, ObjectStatus(hash_info=hi, state=UNCHECKED, witness=witness)
            )
            status.state = UNCHECKED
            status.witness = witness

    def _stop(self, reason: str) -> None:
        self._stopped = True
        assert self._report is not None
        if reason not in self._report.incomplete_reasons:
            self._report.incomplete_reasons.append(reason)

    def _charge_object(self) -> None:
        limit = self.budget.max_objects
        if limit is not None and self._objects_checked >= limit:
            raise _BudgetExhaustedError("budget:max_objects")

    def _charge_edge(self) -> None:
        limit = self.budget.max_edges
        if limit is not None and self._edges >= limit:
            raise _BudgetExhaustedError("budget:max_edges")
        self._edges += 1

    def _charge_witness(self) -> None:
        limit = self.budget.max_witnesses
        if limit is not None and self._witness_count >= limit:
            self._witnesses_truncated = True
            raise _BudgetExhaustedError("budget:max_witnesses")
        self._witness_count += 1

    def _charge_bytes(self, size: int) -> None:
        """Soft byte budget: checked *before* each read.

        The actual number of bytes consumed is always recorded even when a
        single read crosses the limit.
        """
        limit = self.budget.max_bytes
        if limit is not None and self._bytes_read >= limit:
            raise _BudgetExhaustedError("budget:max_bytes")
        self._bytes_read += size

    def _process(self, hi: HashInfo) -> None:
        if hi.isdir:
            self._dir_objects.add(hi)
            tree = self._load_directory(hi)
            if tree is None:
                # missing or unparseable in every ODB: descendants are unknown
                return
            self._expand(hi, tree)
        else:
            self._check_leaf(hi)

    # ------------------------------------------------------------------ #
    # object checks
    # ------------------------------------------------------------------ #

    def _check_leaf(self, hi: HashInfo) -> None:
        assert hi.value  # filtered at enqueue time
        if self.mode == METADATA:
            sources = self._existing_sources(hi)
            if sources:
                self._set_state(hi, OK, sources=sources)
            else:
                self._record_issue(hi, MISSING, detail="object not found in any ODB")
            return

        found_sources: list[str] = []
        conflicts: list[ContentConflict] = []
        for index, odb in enumerate(self.odbs):
            label = self.labels[index]
            try:
                if not odb.exists(hi.value):
                    continue
            except OSError as exc:
                conflicts.append(
                    ContentConflict(
                        hi,
                        label,
                        "inaccessible",
                        detail=str(exc),
                        witness=self._witnesses.get(hi),
                    )
                )
                continue
            found_sources.append(label)
            actual = self._verify_content(odb, hi, conflicts, label)
            if actual is None:
                continue
            self._set_state(hi, OK, sources=tuple(found_sources))
            self._report.conflicts.extend(conflicts)  # type: ignore[union-attr]
            return

        self._report.conflicts.extend(conflicts)  # type: ignore[union-attr]
        if found_sources:
            actual = next(
                (
                    conflict.actual
                    for conflict in reversed(conflicts)
                    if conflict.actual
                ),
                None,
            )
            self._record_issue(
                hi,
                MISMATCHED,
                sources=tuple(found_sources),
                detail="content hash does not match OID in every available ODB",
                actual=actual,
            )
        else:
            self._record_issue(hi, MISSING, detail="object not found in any ODB")

    def _verify_content(
        self,
        odb: "HashFileDB",
        hi: HashInfo,
        conflicts: list[ContentConflict],
        label: str,
    ) -> Optional[str]:
        """Stream-hash object content using the same hashers as ``check()``.

        Unlike :meth:`HashFileDB.check` this never removes the object, never
        touches the state cache and returns the actual digest instead.
        """
        assert hi.value  # filtered at enqueue time
        try:
            data = self._read_bytes(odb, hi.value)
        except FileNotFoundError:
            return None
        try:
            actual = _hash_bytes(data, hi.name)
        except (ValueError, LookupError) as exc:
            conflicts.append(
                ContentConflict(
                    hi,
                    label,
                    "unsupported_algorithm",
                    detail=str(exc),
                    witness=self._witnesses.get(hi),
                )
            )
            return None
        expected = hi.value.split(".", 1)[0]
        if actual == expected:
            return actual
        conflicts.append(
            ContentConflict(
                hi,
                label,
                "content",
                actual=actual,
                detail=f"expected {expected}, got {actual}",
                witness=self._witnesses.get(hi),
            )
        )
        return None

    def _existing_sources(self, hi: HashInfo) -> tuple[str, ...]:
        assert hi.value  # filtered at enqueue time
        sources = []
        for index, odb in enumerate(self.odbs):
            try:
                if odb.exists(hi.value):
                    sources.append(self.labels[index])
            except OSError:
                continue
        return tuple(sources)

    # ------------------------------------------------------------------ #
    # directory loading / expansion
    # ------------------------------------------------------------------ #

    def _load_directory(self, hi: HashInfo) -> Optional[Tree]:
        sources: list[str] = []
        failures: list[ContentConflict] = []
        for index, odb in enumerate(self.odbs):
            label = self.labels[index]
            tree, conflict = self._try_odb_directory(odb, label, hi)
            if conflict is not None:
                sources.append(label)
                failures.append(conflict)
                continue
            if tree is None:
                continue  # this ODB does not hold the object
            # A structurally usable copy was found; earlier fallback failures
            # remain observable as conflicts.
            sources.append(label)
            self._report.conflicts.extend(failures)  # type: ignore[union-attr]
            self._set_state(hi, OK, sources=tuple(sources))
            return tree

        self._report.conflicts.extend(failures)  # type: ignore[union-attr]
        self._record_directory_failure(hi, tuple(sources), failures)
        # The descendants of an unloadable directory cannot be enumerated:
        # mark the subtree unknown instead of pretending the walk is complete.
        self._report.unknown_subtrees.append(hi)  # type: ignore[union-attr]
        return None

    def _try_odb_directory(
        self, odb: "HashFileDB", label: str, hi: HashInfo
    ) -> tuple[Optional[Tree], Optional[ContentConflict]]:
        """Attempt loading one directory object from one fallback ODB."""
        assert hi.value  # filtered at enqueue time
        try:
            if not odb.exists(hi.value):
                return None, None
        except OSError as exc:
            return None, ContentConflict(
                hi,
                label,
                "inaccessible",
                detail=str(exc),
                witness=self._witnesses.get(hi),
            )
        try:
            raw = self._read_bytes(odb, hi.value)
            tree = _parse_tree(raw, odb, hi)
        except _BudgetExhaustedError:
            raise
        except FileNotFoundError:
            return None, None
        except (
            ObjectFormatError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
        ) as exc:
            return None, ContentConflict(
                hi,
                label,
                "format",
                detail=str(exc),
                witness=self._witnesses.get(hi),
            )
        if self.mode == VERIFY:
            conflict = self._verify_directory_bytes(hi, label, raw)
            if conflict is not None:
                return None, conflict
        return tree, None

    def _verify_directory_bytes(
        self, hi: HashInfo, label: str, raw: bytes
    ) -> Optional[ContentConflict]:
        """Hash directory bytes like HashFileDB.check, without side effects."""
        assert hi.value  # filtered at enqueue time
        try:
            actual = _hash_bytes(raw, hi.name)
        except (ValueError, LookupError) as exc:
            return ContentConflict(
                hi,
                label,
                "unsupported_algorithm",
                detail=str(exc),
                witness=self._witnesses.get(hi),
            )
        expected = hi.value.split(".", 1)[0]
        if actual == expected:
            return None
        # another ODB may still hold a good copy
        return ContentConflict(
            hi,
            label,
            "content",
            actual=actual,
            detail=f"expected {expected}, got {actual}",
            witness=self._witnesses.get(hi),
        )

    def _record_directory_failure(
        self,
        hi: HashInfo,
        sources: tuple[str, ...],
        failures: list[ContentConflict],
    ) -> None:
        if not sources:
            self._record_issue(hi, MISSING, detail="directory not found in any ODB")
            return
        kinds = {failure.kind for failure in failures}
        if "content" in kinds or "unsupported_algorithm" in kinds:
            actual = next(
                (failure.actual for failure in reversed(failures) if failure.actual),
                None,
            )
            self._record_issue(
                hi,
                MISMATCHED,
                sources=sources,
                detail="directory content hash mismatch in every available ODB",
                actual=actual,
            )
        else:
            self._record_issue(
                hi,
                MALFORMED,
                sources=sources,
                detail="directory object could not be parsed in any ODB",
            )

    def _expand(self, parent: HashInfo, tree: Tree) -> None:
        adjacency: list[tuple[tuple[str, ...], HashInfo]] = []
        parent_witness = self._witnesses[parent]
        for key in sorted(tree._dict):
            self._charge_edge()
            _, entry_hi = tree._dict[key]
            if entry_hi is None or not entry_hi.value or not entry_hi.name:
                self._report.malformed_entries.append(  # type: ignore[union-attr]
                    MalformedEntry(
                        witness=Witness(parent_witness.root, parent_witness.path + key),
                        detail="directory entry has no usable hash info",
                    )
                )
                self._charge_witness()
                continue
            self._refs[entry_hi] += 1
            adjacency.append((key, entry_hi))
            witness = Witness(parent_witness.root, parent_witness.path + key)
            self._enqueue(entry_hi, witness)
        self._adjacency[parent] = adjacency

    # ------------------------------------------------------------------ #
    # cycle detection (iterative DFS, white/gray/black, deterministic)
    # ------------------------------------------------------------------ #

    def _detect_cycles(self) -> None:
        color: dict[HashInfo, int] = {}
        for start in sorted(self._dir_objects, key=_hi_key):
            if color.get(start, _WHITE) != _WHITE:
                continue
            # stack frames: (node, iterator over its sorted outgoing edges)
            stack: list[tuple[HashInfo, Iterable[tuple[tuple[str, ...], HashInfo]]]] = [
                (start, iter(self._sorted_edges(start)))
            ]
            color[start] = _GRAY
            while stack:
                node, edges = stack[-1]
                advanced = False
                for key, child in edges:
                    node_witness = self._witnesses[node]
                    edge_witness = Witness(node_witness.root, node_witness.path + key)
                    child_color = color.get(child, _WHITE)
                    if child_color == _GRAY and child.isdir:
                        issue = CycleIssue(
                            witness=edge_witness, source=node, target=child
                        )
                        if issue not in self._report.cycles:  # type: ignore[union-attr]
                            limit = self.budget.max_witnesses
                            if limit is not None and self._witness_count >= limit:
                                self._witnesses_truncated = True
                                self._stop("budget:max_witnesses")
                                return
                            self._witness_count += 1
                            self._report.cycles.append(issue)  # type: ignore[union-attr]
                        continue
                    if child_color == _WHITE and child.isdir:
                        color[child] = _GRAY
                        stack.append((child, iter(self._sorted_edges(child))))
                        advanced = True
                        break
                if not advanced:
                    color[node] = _BLACK
                    stack.pop()

    def _sorted_edges(self, hi: HashInfo) -> list[tuple[tuple[str, ...], HashInfo]]:
        return sorted(
            self._adjacency.get(hi, ()),
            key=lambda item: (item[0], _hi_key(item[1])),
        )

    # ------------------------------------------------------------------ #
    # finalization
    # ------------------------------------------------------------------ #

    def _finalize(self) -> None:
        assert self._report is not None
        for hi, status in self._statuses.items():
            status.references = self._refs.get(hi, 0)
            if status.witness is None:
                status.witness = self._witnesses.get(hi) or self._known.get(hi)
        self._report.objects = self._statuses
        stats = self._report.stats
        stats.objects_seen = len(self._statuses)
        stats.objects_checked = self._objects_checked
        stats.edges_traversed = self._edges
        stats.bytes_read = self._bytes_read
        stats.witnesses = self._witness_count
        stats.witnesses_truncated = self._witnesses_truncated
        stats.limits = self.budget
        self._report.unknown_subtrees = sorted(
            set(self._report.unknown_subtrees), key=_hi_key
        )
        self._report.invalid_roots = sorted(
            set(self._report.invalid_roots), key=_hi_key
        )

    def _set_state(
        self, hi: HashInfo, state: str, sources: Optional[tuple[str, ...]] = None
    ) -> None:
        status = self._statuses[hi]
        status.state = state
        if sources is not None:
            status.sources = sources

    def _record_issue(
        self,
        hi: HashInfo,
        state: str,
        sources: Optional[tuple[str, ...]] = None,
        detail: Optional[str] = None,
        actual: Optional[str] = None,
    ) -> None:
        self._charge_witness()
        status = self._statuses[hi]
        status.state = state
        status.detail = detail
        status.actual = actual
        if sources is not None:
            status.sources = sources

    # ------------------------------------------------------------------ #
    # low-level, byte-counted reads (read-only; no state cache involved)
    # ------------------------------------------------------------------ #

    def _read_bytes(self, odb: "HashFileDB", oid: str) -> bytes:
        obj = odb.get(oid)
        with obj.fs.open(obj.path, "rb") as fobj:
            data = fobj.read()
        self._charge_bytes(len(data))
        return data


def _parse_tree(raw: bytes, odb: "HashFileDB", hi: HashInfo) -> Tree:
    """Parse directory JSON mirroring :meth:`Tree.load` without any writes."""
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise ObjectFormatError(f"directory {hi} is corrupted")
    hash_name = None
    if odb.hash_name == "md5-dos2unix":
        hash_name = "md5-dos2unix"
    tree = Tree.from_list(parsed, hash_name=hash_name)
    tree.hash_info = hi
    tree.oid = hi.value
    return tree


def _hash_bytes(data: bytes, name: Optional[str]) -> str:
    if not name:
        raise ValueError("missing hash algorithm name")
    stream = get_hash_stream(BytesIO(data), name)
    chunk_size = 1 << 20
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break
    return stream.hash_value


def explain_reachability(
    odbs: Iterable["HashFileDB"],
    roots: Iterable[HashInfo],
    *,
    mode: str = METADATA,
    budget: Optional[Budget] = None,
    labels: Optional[Iterable[str]] = None,
    include_paths: bool = False,
) -> ReachabilityReport:
    """Convenience wrapper around :class:`ExplainReachability`."""
    return ExplainReachability(
        odbs,
        mode=mode,
        budget=budget,
        labels=labels,
        include_paths=include_paths,
    ).explain(roots)


__all__ = [
    "MALFORMED",
    "METADATA",
    "MISMATCHED",
    "MISSING",
    "OK",
    "UNCHECKED",
    "VERIFY",
    "Budget",
    "ContentConflict",
    "CycleIssue",
    "ExplainReachability",
    "MalformedEntry",
    "ReachabilityError",
    "ReachabilityReport",
    "ReachabilityStats",
    "Witness",
    "explain_reachability",
]
