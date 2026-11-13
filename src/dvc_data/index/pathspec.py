"""Compiled, component-level path specifications for sparse DataIndex views.

This module implements :class:`PathSpecView`, a read-only view over a
:class:`~dvc_data.index.index.DataIndex` that selects a subset of entries based
on an ordered list of include/exclude rules.

Pattern semantics
-----------------
* Patterns are matched against path *components* (``DataIndexKey`` is a tuple
  of component strings), never against platform specific separator strings.
  String patterns are split on the forward slash ``"/"`` only; a backslash is
  never a separator.
* A backslash escapes the next character (``"\\*"`` matches a literal ``*``,
  ``"\\\\"`` a literal backslash, ``"\\/"`` a literal slash inside a
  component). A dangling backslash is rejected at compile time.
* Within a component, ``*`` matches any (possibly empty) sequence of
  characters, ``?`` matches a single character and ``[...]`` / ``[!...]``
  define character classes. Wildcards never cross a component boundary.
* The special component ``**`` matches zero or more full components, so
  ``("a", "**", "b")`` matches both ``("a", "b")`` (zero levels) and
  ``("a", "x", "b")``.
* The empty pattern (empty string or an empty tuple) matches the root key
  ``()`` only. Empty components (e.g. ``"a//b"``) are rejected.
* A rule matches a key either by *self match* (the pattern matches the key
  exactly) or by *descendant inheritance* (the pattern matches a strict
  ancestor directory of the key). Rules are evaluated in order and the last
  matching rule wins; keys without any match use the spec default (exclude).

The compiled spec exposes a trie/NFA plan that allows lazy traversal to skip
subtrees which can provably never contain a selected entry. A subtree is only
pruned when no include rule can possibly match below it, so excluding a parent
directory can never hide a descendant that a later rule re-includes.

Coverage
--------
The view records which directory prefixes have been fully observed. Directories
that were never loaded or failed to load stay ``unknown``: "not selected / not
loaded" is never interpreted as "the target does not contain this file".
Sparse deletions are only safe for keys that are explicitly selected and whose
every ancestor prefix is fully covered.
"""

import re
from collections import deque
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import attrs

from .index import (
    BaseDataIndex,
    DataIndexDirError,
    DataIndexEntry,
    DataIndexKey,
)

if TYPE_CHECKING:
    from .diff import Change
    from .index import DataIndex, StorageMapping

INCLUDE = "include"
EXCLUDE = "exclude"

_GLOBSTAR = "**"


class PathSpecError(ValueError):
    """Invalid path specification or exhausted compile-time budget."""


class PathSpecBudgetError(PathSpecError):
    """Compiled spec would exceed a configured budget."""


@attrs.frozen
class PathSpecBudget:
    """Upper limits for compilation and lazy traversal.

    ``max_patterns``, ``max_states`` and ``max_depth`` are enforced while
    compiling the spec. ``max_lazy_reads`` and ``max_entries`` are enforced
    while traversing a :class:`PathSpecView`; exceeding them is reported as an
    incomplete result instead of silently dropping data.
    """

    max_patterns: int = 4096
    max_states: int = 100_000
    max_depth: int = 256
    max_lazy_reads: int = 10_000
    max_entries: int = 1_000_000


@attrs.frozen
class PathSpecRule:
    """A single ordered include/exclude rule."""

    include: bool
    pattern: str
    components: tuple[str, ...]


@attrs.frozen
class CoverageReport:
    """Result of observing a :class:`PathSpecView`.

    Attributes:
        covered_prefixes: directory prefixes whose full contents were
            observed (recursively, including all child directories).
        unknown_prefixes: directories that were not loaded, failed to load or
            could not be finished because a runtime budget was exhausted.
        pruned_prefixes: prefixes that were deliberately never loaded because
            the compiled spec proves they cannot contain selected entries.
        incomplete: True if a budget (compile-time state budget or runtime
            lazy read / entry budget) was exhausted, meaning the report is not
            exhaustive.
    """

    covered_prefixes: set[DataIndexKey]
    unknown_prefixes: set[DataIndexKey]
    pruned_prefixes: set[DataIndexKey]
    incomplete: bool

    def is_covered(self, key: DataIndexKey) -> bool:
        """Return True if every prefix of ``key`` (including ``key`` itself)."""
        return all(
            key[:idx] in self.covered_prefixes for idx in range(len(key) + 1)
        )


@attrs.frozen
class _PlanState:
    selected: bool
    prunable: bool


@attrs.frozen
class _CompiledRule:
    include: bool
    components: tuple[str, ...]
    tokens: tuple[Any, ...]  # re.Pattern or the _GLOBSTAR sentinel


def parse_pattern(  # noqa: C901
    pattern: Union[str, tuple[str, ...], list[str]]
) -> tuple[str, ...]:
    """Split a pattern into component strings.

    Strings are split on unescaped ``"/"`` only. Tuples/lists are validated
    and copied, so callers may express patterns independently of the host
    platform separator.
    """
    if isinstance(pattern, (tuple, list)):
        components = tuple(pattern)
        if any(not isinstance(component, str) for component in components):
            raise PathSpecError("pattern components must be strings")
        if components and any(component == "" for component in components):
            raise PathSpecError(f"empty path component in pattern {pattern!r}")
        return components

    if not isinstance(pattern, str):
        raise PathSpecError(
            f"pattern must be a str or component tuple, got {pattern!r}"
        )

    if pattern == "":
        return ()

    parts: list[list[str]] = [[]]
    idx = 0
    while idx < len(pattern):
        char = pattern[idx]
        if char == "\\":
            if idx + 1 >= len(pattern):
                raise PathSpecError(f"dangling escape in pattern {pattern!r}")
            parts[-1].append(pattern[idx : idx + 2])
            idx += 2
            continue
        if char == "/":
            parts.append([])
            idx += 1
            continue
        parts[-1].append(char)
        idx += 1

    components = tuple("".join(part) for part in parts)
    if any(component == "" for component in components):
        raise PathSpecError(f"empty path component in pattern {pattern!r}")
    return components


def _translate_component(  # noqa: C901, PLR0912
    component: str,
) -> re.Pattern:
    """Translate a single component glob into an anchored regex."""
    out = ["^"]
    idx = 0
    while idx < len(component):
        char = component[idx]
        if char == "\\":
            idx += 1
            if idx >= len(component):
                raise PathSpecError("dangling escape in pattern component")
            out.append(re.escape(component[idx]))
            idx += 1
        elif char == "*":
            # Wildcards never cross a component boundary.
            out.append("[^/]*")
            idx += 1
        elif char == "?":
            out.append("[^/]")
            idx += 1
        elif char == "[":
            end = idx + 1
            if end < len(component) and component[end] == "!":
                end += 1
            if end < len(component) and component[end] == "]":
                end += 1
            while end < len(component) and component[end] != "]":
                if component[end] == "\\" and end + 1 < len(component):
                    end += 2
                else:
                    end += 1
            if end >= len(component):
                # No closing bracket: treat "[" as a literal character.
                out.append(re.escape("["))
                idx += 1
            else:
                inner = component[idx + 1 : end]
                if inner.startswith("!"):
                    inner = "^" + inner[1:]
                out.append(f"[{inner}]")
                idx = end + 1
        else:
            out.append(re.escape(char))
            idx += 1
    out.append("$")
    return re.compile("".join(out))


def _normalize_action(action: Any) -> bool:
    if isinstance(action, bool):
        return action
    if action == INCLUDE:
        return True
    if action == EXCLUDE:
        return False
    raise PathSpecError(
        f"rule action must be {INCLUDE!r} or {EXCLUDE!r}, got {action!r}"
    )


class CompiledPathSpec:
    """Ordered include/exclude rules compiled into a component NFA plan.

    The compiled object is reusable across many views and indexes. Plan
    states for concrete keys are materialized lazily and counted against
    ``budget.max_states``; if the state budget is exhausted the spec marks
    itself ``incomplete`` and disables subtree pruning (a safe fallback that
    only loads more data, never less).
    """

    def __init__(
        self,
        rules: Any,
        budget: Optional[PathSpecBudget] = None,
        default: bool = False,
    ) -> None:
        self.budget = budget or PathSpecBudget()
        self.default = default
        self.incomplete = False

        compiled_rules: list[_CompiledRule] = []
        raw_rules: list[PathSpecRule] = []
        for rule in rules:
            if isinstance(rule, PathSpecRule):
                include, pattern = rule.include, rule.pattern
            else:
                include, pattern = rule
            include = _normalize_action(include)
            components = parse_pattern(pattern)
            if len(components) > self.budget.max_depth:
                raise PathSpecBudgetError(
                    f"pattern {pattern!r} exceeds max depth {self.budget.max_depth}"
                )
            tokens = tuple(
                _GLOBSTAR
                if component == _GLOBSTAR
                else _translate_component(component)
                for component in components
            )
            compiled_rules.append(_CompiledRule(include, components, tokens))
            raw_rules.append(PathSpecRule(include, str(pattern), components))

        if len(compiled_rules) > self.budget.max_patterns:
            raise PathSpecBudgetError(
                f"number of patterns ({len(compiled_rules)}) exceeds "
                f"max_patterns {self.budget.max_patterns}"
            )

        self.rules: tuple[_CompiledRule, ...] = tuple(compiled_rules)
        self.raw_rules: tuple[PathSpecRule, ...] = tuple(raw_rules)
        self.n_states = 0
        self._state_cache: dict[DataIndexKey, _PlanState] = {}
        self._match_cache: dict[tuple[int, DataIndexKey], bool] = {}
        self._reachable_cache: dict[tuple[int, DataIndexKey], frozenset[int]] = {}

    def __repr__(self) -> str:
        return (
            f"CompiledPathSpec(rules={len(self.rules)}, "
            f"states={self.n_states}, incomplete={self.incomplete})"
        )

    def _epsilon(self, states: set[int], pattern: tuple[Any, ...]) -> set[int]:
        # "**" may match zero components.
        result = set(states)
        stack = deque(states)
        while stack:
            pos = stack.popleft()
            if pos < len(pattern) and pattern[pos] == _GLOBSTAR:
                nxt = pos + 1
                if nxt not in result:
                    result.add(nxt)
                    stack.append(nxt)
        return result

    def _exact_match(self, rule_idx: int, key: DataIndexKey) -> bool:
        """Return True if rule pattern matches ``key`` component-by-component."""
        cache_key = (rule_idx, key)
        cached = self._match_cache.get(cache_key)
        if cached is not None:
            return cached

        rule = self.rules[rule_idx]
        tokens = rule.tokens

        def walk(pos: int, depth: int) -> bool:
            if pos == len(tokens):
                return depth == len(key)
            if tokens[pos] == _GLOBSTAR:
                # "**" consumes zero components, or one (or more) components.
                return walk(pos + 1, depth) or (
                    depth < len(key) and walk(pos, depth + 1)
                )
            if depth >= len(key):
                return False
            token = tokens[pos]
            return bool(
                token.fullmatch(key[depth]) and walk(pos + 1, depth + 1)
            )

        result = walk(0, 0)
        self._match_cache[cache_key] = result
        return result

    def _rule_matches_key(self, rule_idx: int, key: DataIndexKey) -> bool:
        """Self match or descendant inheritance from a strict ancestor."""
        for depth in range(len(key) + 1):
            if self._exact_match(rule_idx, key[:depth]):
                return True
        return False

    def _reachable_positions(
        self, rule_idx: int, key: DataIndexKey
    ) -> frozenset[int]:
        """NFA positions reachable after consuming all components of ``key``."""
        cache_key = (rule_idx, key)
        cached = self._reachable_cache.get(cache_key)
        if cached is not None:
            return cached

        rule = self.rules[rule_idx]
        tokens = rule.tokens
        states = self._epsilon({0}, tokens)
        for component in key:
            moved: set[int] = set()
            for pos in states:
                if pos >= len(tokens):
                    continue
                token = tokens[pos]
                if token == _GLOBSTAR:
                    moved.add(pos)  # consume a component, keep looping
                    moved.add(pos + 1)  # consume and terminate the "**"
                elif token.fullmatch(component):
                    moved.add(pos + 1)
            states = self._epsilon(moved, tokens)
        result = frozenset(states)
        self._reachable_cache[cache_key] = result
        return result

    def _include_reachable_below(self, key: DataIndexKey) -> bool:
        """True if any include rule can exactly match a strict descendant.

        A rule can accept below ``key`` whenever some reachable NFA position
        still has pattern tokens left, because an arbitrary suffix of the
        right length can always be chosen. This is deliberately conservative:
        later exclude rules are ignored, so pruning can never hide a
        re-included descendant.
        """
        for rule_idx, rule in enumerate(self.rules):
            if not rule.include:
                continue
            positions = self._reachable_positions(rule_idx, key)
            if any(pos < len(rule.tokens) for pos in positions):
                return True
        return False

    def _selected(self, key: DataIndexKey) -> bool:
        selected = self.default
        for rule_idx in range(len(self.rules)):
            if self._rule_matches_key(rule_idx, key):
                selected = self.rules[rule_idx].include
        return selected

    def plan_state(self, key: DataIndexKey) -> _PlanState:
        """Return the cached selection/pruning plan state for ``key``."""
        cached = self._state_cache.get(key)
        if cached is not None:
            return cached

        selected = self._selected(key)
        prunable = not selected and not self._include_reachable_below(key)

        if self.n_states >= self.budget.max_states:
            # Safe fallback: keep selecting correctly but never prune.
            self.incomplete = True
            prunable = False

        state = _PlanState(selected=selected, prunable=prunable)
        self._state_cache[key] = state
        self.n_states += 1
        return state

    def filter_fn(self, key: DataIndexKey) -> bool:
        return self.plan_state(key).selected


def compile_pathspec(
    rules: Any,
    budget: Optional[PathSpecBudget] = None,
    default: bool = False,
) -> CompiledPathSpec:
    """Compile ordered ``(action, pattern)`` rules into a :class:`CompiledPathSpec`."""
    return CompiledPathSpec(rules, budget=budget, default=default)


class PathSpecView(BaseDataIndex):
    """Read-only, lazily traversed selection of a :class:`DataIndex`.

    The view never mutates the underlying index: writes are rejected, the
    base ``onerror`` handler is never reassigned and ``storage_map`` is
    delegated unchanged, so FileStorage/ObjectStorage routing keeps following
    the original entries. Multiple views with different specs over the same
    base index are fully independent.
    """

    def __init__(
        self,
        index: "DataIndex",
        spec: CompiledPathSpec,
        budget: Optional[PathSpecBudget] = None,
    ) -> None:
        self._index = index
        self._spec = spec
        self._budget = budget or spec.budget
        self._onerror_local: Optional[Callable] = None

        # Runtime bookkeeping.
        self.lazy_reads = 0
        self.entries_seen = 0
        self.incomplete = False
        self._loaded: set[DataIndexKey] = set()
        self._explored: set[DataIndexKey] = set()
        self._unknown: set[DataIndexKey] = set()
        self._pruned: set[DataIndexKey] = set()
        self._drained = False
        self._report: Optional[CoverageReport] = None

    @property
    def spec(self) -> CompiledPathSpec:
        return self._spec

    @property
    def storage_map(self) -> "StorageMapping":  # type: ignore[override]
        # Routing is inherited from the base index; never copy or rewrite it.
        return self._index.storage_map

    @property
    def onerror(self) -> Callable:
        if self._onerror_local is not None:
            return self._onerror_local
        return self._index.onerror

    @onerror.setter
    def onerror(self, onerror: Callable) -> None:
        # Intentionally do not propagate to the base index: a view must not
        # change global base settings.
        self._onerror_local = onerror

    def filter_fn(self, key: DataIndexKey) -> bool:
        return self._spec.plan_state(key).selected

    # -- mapping interface -------------------------------------------------

    def __getitem__(self, key: DataIndexKey) -> DataIndexEntry:
        if key == () or self.filter_fn(key):
            return self._index[key]
        raise KeyError(key)

    def __setitem__(self, key: DataIndexKey, value: DataIndexEntry) -> None:
        raise TypeError("PathSpecView is read-only")

    def __delitem__(self, key: DataIndexKey) -> None:
        raise TypeError("PathSpecView is read-only")

    def delete_node(self, key: DataIndexKey) -> None:
        raise TypeError("PathSpecView is read-only")

    def __iter__(self) -> Iterator[DataIndexKey]:
        return (key for key, _ in self._iteritems())

    def __len__(self) -> int:
        return len(list(iter(self)))

    def has_node(self, key: DataIndexKey) -> bool:
        return self.filter_fn(key) and self._index.has_node(key)

    def longest_prefix(
        self, key: DataIndexKey
    ) -> "tuple[Optional[DataIndexKey], Optional[DataIndexEntry]]":
        if self.filter_fn(key):
            return self._index.longest_prefix(key)
        return (None, None)

    def traverse(self, node_factory: Callable, **kwargs: Any) -> Any:
        def _node_factory(
            path_conv: Callable,
            key: DataIndexKey,
            children: Any,
            *args: Any,
        ):
            if not key or self.filter_fn(key):
                return node_factory(path_conv, key, children, *args)
            return None

        return self._index.traverse(_node_factory, **kwargs)

    def ls(self, root_key: DataIndexKey, detail: bool = True):
        state = self._spec.plan_state(root_key)
        if state.prunable:
            return
        self._index._ensure_loaded(root_key)

        if detail:
            yield from (
                (key, self._index._info_from_entry(key, entry))
                for key, entry in self._index._trie.ls(
                    root_key, with_values=True
                )
                if self.filter_fn(key)
            )
        else:
            yield from (
                key
                for key in self._index.ls(root_key, detail=False)
                if self.filter_fn(key)
            )

    # -- lazy, pruning-aware traversal ------------------------------------

    def _tick_entry(self) -> bool:
        self.entries_seen += 1
        if self.entries_seen > self._budget.max_entries:
            self.incomplete = True
            return False
        return True

    def _mark_incomplete(self, key: DataIndexKey) -> None:
        self.incomplete = True
        self._unknown.add(key)

    def _load_dir(self, key: DataIndexKey, entry: Optional[DataIndexEntry]) -> bool:
        """Load one directory through the base index.

        Failures (raised by the base ``onerror`` handler, or a noop handler
        that leaves the entry unloaded) are recorded as unknown instead of
        being interpreted as an empty/absent directory.
        """
        if entry is None:
            # Virtual root: nothing to load, its children come from the trie.
            return True
        if entry.loaded:
            return True

        self.lazy_reads += 1
        try:
            self._index._load(key, entry)
        except DataIndexDirError:
            self._report = None
            return False
        except Exception as exc:  # noqa: BLE001 - base onerror may re-raise anything
            if self._onerror_local is not None:
                self._onerror_local(entry, exc)
            return False
        return bool(entry.loaded)

    def _iteritems_shallow(
        self, root: DataIndexKey, ensure_loaded: bool
    ) -> Iterator[tuple[DataIndexKey, DataIndexEntry]]:
        root_entry = self._index._trie.get(root)
        if root_entry is not None and root_entry.isdir and not root_entry.loaded:
            if not ensure_loaded:
                return
            state = self._spec.plan_state(root)
            if state.prunable:
                self._pruned.add(root)
                return
            if self.lazy_reads >= self._budget.max_lazy_reads:
                self._mark_incomplete(root)
                return
            if not self._load_dir(root, root_entry):
                self._unknown.add(root)
                self._report = None
                return
            self._loaded.add(root)

        for child_key in self._index._trie.ls(root):
            state = self._spec.plan_state(child_key)
            if state.prunable:
                self._pruned.add(child_key)
                continue
            entry = self._index._trie.get(child_key)
            if entry is not None and state.selected:
                if not self._tick_entry():
                    return
                yield child_key, entry

    def _iteritems(  # noqa: C901, PLR0912
        self,
        prefix: Optional[DataIndexKey] = None,
        shallow: bool = False,
        ensure_loaded: bool = True,
    ) -> Iterator[tuple[DataIndexKey, DataIndexEntry]]:
        root: DataIndexKey = () if prefix is None else prefix
        if shallow:
            yield from self._iteritems_shallow(root, ensure_loaded)
            return

        stack: deque[DataIndexKey] = deque([root])

        while stack:
            key = stack.popleft()

            if len(key) > self._budget.max_depth:
                self._mark_incomplete(key)
                continue

            state = self._spec.plan_state(key)
            entry = self._index._trie.get(key)

            if key and entry is not None and state.selected:
                if not self._tick_entry():
                    return
                yield key, entry

            is_dir = entry is not None and entry.isdir
            if entry is not None and not is_dir:
                continue

            if entry is not None and not entry.loaded:
                if not ensure_loaded:
                    continue
                if state.prunable:
                    # Proven empty of selectable entries: do not touch storage.
                    self._pruned.add(key)
                    continue
                if self.lazy_reads >= self._budget.max_lazy_reads:
                    self._mark_incomplete(key)
                    continue
                if not self._load_dir(key, entry):
                    self._unknown.add(key)
                    self._report = None
                    continue
                self._loaded.add(key)
                # Reload after the in-place trie update.
                entry = self._index._trie.get(key)

            child_keys = list(self._index._trie.ls(key))
            self._explored.add(key)
            self._report = None

            for child_key in child_keys:
                child_state = self._spec.plan_state(child_key)
                if child_state.prunable:
                    self._pruned.add(child_key)
                    continue
                stack.append(child_key)

    def iteritems(
        self,
        prefix: Optional[DataIndexKey] = None,
        shallow: bool = False,
    ) -> Iterator[tuple[DataIndexKey, DataIndexEntry]]:
        yield from self._iteritems(prefix=prefix, shallow=shallow, ensure_loaded=True)

    # -- coverage ----------------------------------------------------------

    def _drain(self) -> None:
        if not self._drained:
            for _ in self.iteritems():
                pass
            self._drained = True

    def coverage_report(self) -> CoverageReport:  # noqa: C901
        """Compute (and cache) the coverage report for the whole view."""
        self._drain()
        if self._report is not None:
            return self._report

        covered: set[DataIndexKey] = set()
        memo: dict[DataIndexKey, bool] = {}

        def fully_covered(key: DataIndexKey) -> bool:
            if key in memo:
                return memo[key]
            if key in self._unknown or key not in self._explored:
                memo[key] = False
                return False
            ok = True
            for child_key in self._index._trie.ls(key):
                child = self._index._trie.get(child_key)
                if child is None or not child.isdir:
                    continue
                if child_key in self._pruned:
                    # Spec-deterministic exclusion: no selected entry can
                    # hide there, so it does not block ancestor coverage.
                    continue
                if not fully_covered(child_key):
                    ok = False
            if ok:
                covered.add(key)
            memo[key] = ok
            return ok

        for directory in list(self._explored):
            fully_covered(directory)

        report = CoverageReport(
            covered_prefixes=covered,
            unknown_prefixes=set(self._unknown),
            pruned_prefixes=set(self._pruned),
            incomplete=self.incomplete or self._spec.incomplete,
        )
        self._report = report
        return report

    def is_fully_covered(self, key: DataIndexKey) -> bool:
        """True if every prefix of ``key`` was completely observed."""
        report = self.coverage_report()
        return report.is_covered(key)

    def is_safe_to_delete(self, key: DataIndexKey) -> bool:
        """Sparse deletions require explicit selection and full coverage.

        Selecting a single file never licenses deleting its siblings: a
        sibling is absent from the view only because it was not selected, so
        its directory prefix is not claimed as fully covered unless the view
        actually observed every child.
        """
        if not self.filter_fn(key):
            return False
        report = self.coverage_report()
        # Every strict ancestor prefix must have been fully observed.
        for idx in range(len(key)):
            if key[:idx] not in report.covered_prefixes:
                return False
        # A directory itself must also be fully covered; files only need
        # their ancestors covered.
        entry = self._index._trie.get(key)
        return not (
            entry is not None
            and entry.isdir
            and key not in report.covered_prefixes
        )


def pathspec_view(
    index: "DataIndex",
    rules: Any,
    budget: Optional[PathSpecBudget] = None,
    default: bool = False,
) -> PathSpecView:
    """Compile ``rules`` and return a :class:`PathSpecView` over ``index``."""
    spec = compile_pathspec(rules, budget=budget, default=default)
    return PathSpecView(index, spec, budget=budget)


def sparse_diff(
    old: Optional[BaseDataIndex],
    new: PathSpecView,
    **kwargs: Any,
) -> Iterator["Change"]:
    """Diff two indexes, suppressing unsafe sparse deletions.

    A :data:`~dvc_data.index.diff.DELETE` change is only yielded when the key
    is explicitly selected by the view and all of its prefixes are fully
    covered. Adds and modifications are always passed through.
    """
    from .diff import DELETE
    from .diff import diff as _diff

    for change in _diff(old, new, **kwargs):
        if change.typ == DELETE and not new.is_safe_to_delete(change.key):
            continue
        yield change


def filter_sparse_deletes(diff_obj: Any, view: PathSpecView) -> Any:
    """Strip unsafe deletions from a :class:`~dvc_data.index.checkout.Diff`.

    This makes ``dvc_data.index.checkout.apply`` safe for sparse checkouts:
    entries outside the selected, fully covered ranges are never removed from
    the workspace.
    """
    diff_obj.files_delete = [
        entry for entry in diff_obj.files_delete if view.is_safe_to_delete(entry.key)
    ]
    diff_obj.dirs_delete = [
        entry for entry in diff_obj.dirs_delete if view.is_safe_to_delete(entry.key)
    ]
    diff_obj.changes = {
        key: change
        for key, change in diff_obj.changes.items()
        if not (change.typ == "delete" and not view.is_safe_to_delete(key))
    }
    return diff_obj
