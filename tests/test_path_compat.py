"""Tests for the declarative PathCompatibilityPlan.

Conflict expectations are cross-checked with an *independent* normalization
trie implemented below (plain nested dictionaries plus test-local
normalization rules), instead of reusing the planner's own helpers.
"""

import hashlib
import os
import unicodedata
from collections import defaultdict

import pytest
from dvc_objects.fs import MemoryFileSystem
from dvc_objects.fs.local import LocalFileSystem

from dvc_data.hashfile.checkout import checkout
from dvc_data.hashfile.db import HashFileDB
from dvc_data.hashfile.hash_info import HashInfo
from dvc_data.hashfile.meta import Meta
from dvc_data.hashfile.tree import Tree
from dvc_data.index.index import DataIndex, DataIndexEntry
from dvc_data.path_compat import (
    ConflictKind,
    PathCompatibilityError,
    PathCompatibilityPlan,
    PathPlanLimitError,
    PlanLimits,
    Profile,
    RenameMapError,
    check_component,
)

NFC_EACUTE = "\u00e9"  # "é" as a single precomposed code point
NFD_EACUTE = "e\u0301"  # "e" + combining acute accent


def _hi(content: bytes) -> HashInfo:
    return HashInfo(
        "md5",
        hashlib.md5(content).hexdigest(),  # noqa: S324
    )


# ---------------------------------------------------------------------------
# Independent reference implementation (must not import planner internals)
# ---------------------------------------------------------------------------


def _ref_fold(part: str) -> str:
    """Test-local NFC + casefold rule, written independently of the module."""
    canonical = unicodedata.normalize("NFC", part)
    folded = canonical.casefold()
    return unicodedata.normalize("NFC", folded)


def _ref_normalize(profile: Profile, part: str) -> str:
    if profile is Profile.CASE_SENSITIVE:
        return part
    return _ref_fold(part)


def _ref_norm_key(profile: Profile, key: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_ref_normalize(profile, part) for part in key)


class _RefTrie:
    """Minimal independent trie: nested dicts keyed by normalized parts."""

    def __init__(self) -> None:
        self.root: dict = {}
        self.values: dict[tuple[str, ...], list[tuple[str, ...]]] = {}

    def insert(self, key: tuple[str, ...], profile: Profile) -> None:
        node = self.root
        for part in _ref_norm_key(profile, key):
            node = node.setdefault(part, {})
        self.values.setdefault(_ref_norm_key(profile, key), []).append(key)

    def exact_collision_groups(self) -> set[frozenset]:
        return {
            frozenset(paths) for paths in self.values.values() if len(set(paths)) > 1
        }

    def prefix_groups(
        self, profile: Profile, keys: list[tuple[str, ...]]
    ) -> set[frozenset]:
        groups: set[frozenset] = set()
        for key in keys:
            norm = _ref_norm_key(profile, key)
            descendants = [
                other
                for other in keys
                if len(other) > len(key)
                and _ref_norm_key(profile, other)[: len(norm)] == norm
            ]
            if descendants:
                groups.add(frozenset([key, *descendants]))
        return groups


def _ref_directory_alias_groups(
    profile: Profile, keys: list[tuple[str, ...]]
) -> set[frozenset]:
    aliases: dict[tuple[str, ...], set[tuple[str, ...]]] = defaultdict(set)
    for key in keys:
        for depth in range(1, len(key)):
            prefix = key[:depth]
            aliases[_ref_norm_key(profile, prefix)].add(prefix)
    return {
        frozenset(spellings) for spellings in aliases.values() if len(spellings) > 1
    }


def _planner_groups(
    plan: PathCompatibilityPlan, *kinds: ConflictKind
) -> set[frozenset]:
    return {
        frozenset(group.paths) for group in plan.result.conflicts if group.kind in kinds
    }


# ---------------------------------------------------------------------------
# Profile normalization and constraint rules
# ---------------------------------------------------------------------------


def test_case_sensitive_normalization_is_identity():
    for part in ("A", "a", NFC_EACUTE, NFD_EACUTE, "CON", "foo."):
        assert check_component(Profile.CASE_SENSITIVE, part) == ()


def test_casefold_nfc_equivalence():
    assert _ref_fold("A") == _ref_fold("a")
    assert _ref_fold(NFC_EACUTE) == _ref_fold(NFD_EACUTE)
    # ligature and German sharp-s expand under casefold
    assert _ref_fold("ﬄ") == _ref_fold("FFL")
    assert _ref_fold("Straße") == _ref_fold("STRASSE")


@pytest.mark.parametrize(
    "part,reasons",
    [
        ("CON", ("reserved-name",)),
        ("con", ("reserved-name",)),
        ("nul.txt", ("reserved-name",)),
        ("LPT9.log", ("reserved-name",)),
        ("COM0", ()),  # only COM1..COM9 are in the declared set
        ("foo.", ("trailing-dot",)),
        ("bar ", ("trailing-space",)),
        ("a:b", ("forbidden-character",)),
        ("a<b", ("forbidden-character",)),
        ('q"x', ("forbidden-character",)),
        ("a/b", ("separator-in-component",)),
        ("a\\b", ("separator-in-component",)),
        ("a|b?", ("forbidden-character",)),
        ("x\ty", ("control-character",)),
        ("..", ("path-traversal", "trailing-dot")),
        (".", ("path-traversal", "trailing-dot")),
        ("", ("empty-component",)),
        ("normal name_1-2", ()),
    ],
)
def test_windows_portable_component_constraints(part, reasons):
    assert check_component(Profile.WINDOWS_PORTABLE, part) == reasons
    # the declared subset never applies under the other profiles
    if part not in ("", ".", "..", "a/b", "a\\b"):
        assert check_component(Profile.CASEFOLD_NFC, part) == ()
        assert check_component(Profile.CASE_SENSITIVE, part) == ()


# ---------------------------------------------------------------------------
# Conflict detection cross-checked against the independent trie
# ---------------------------------------------------------------------------


TRICKY_KEYS = [
    ("a",),
    ("A",),  # case alias at root -> exact collision under folding
    (NFC_EACUTE,),
    (NFD_EACUTE,),  # Unicode canonical alias -> exact collision
    ("Dir", "a"),
    ("dir", "b"),
    ("DIR", "a"),  # directory aliases + one exact child collision
    ("foo",),
    ("foo", "bar"),  # file is a prefix of a descendant
    ("d", "sub", "x"),
    ("D", "SUB", "y"),
]


@pytest.mark.parametrize(
    "profile",
    [Profile.CASE_SENSITIVE, Profile.CASEFOLD_NFC, Profile.WINDOWS_PORTABLE],
)
def test_conflict_groups_match_independent_trie(profile):
    # all entries are files with distinct content hashes
    raw = [_raw(key, str(index).encode()) for index, key in enumerate(TRICKY_KEYS)]
    plan = PathCompatibilityPlan(profile, raw)

    ref = _RefTrie()
    for key in TRICKY_KEYS:
        ref.insert(key, profile)
    assert (
        _planner_groups(
            plan,
            ConflictKind.NAME_COLLISION,
            ConflictKind.RENAME_COLLISION,
            ConflictKind.TYPE_MISMATCH,
        )
        == ref.exact_collision_groups()
    )

    assert _planner_groups(plan, ConflictKind.PARENT_CHILD) == ref.prefix_groups(
        profile, TRICKY_KEYS
    )
    assert _planner_groups(
        plan, ConflictKind.CASE_DIRECTORY_MERGE
    ) == _ref_directory_alias_groups(profile, TRICKY_KEYS)

    # whole-path comparison, not basename-only: nested aliases are detected
    # under folding profiles (case-sensitive keeps every spelling distinct)
    if profile is not Profile.CASE_SENSITIVE:
        reported = {path for group in plan.result.conflicts for path in group.paths}
        assert ("Dir", "a") in reported
        assert ("DIR", "a") in reported


def test_case_sensitive_profile_allows_aliases():
    raw = [
        _raw(("a",), b"1"),
        _raw(("A",), b"2"),
        _raw((NFC_EACUTE,), b"3"),
        _raw((NFD_EACUTE,), b"4"),
    ]
    plan = PathCompatibilityPlan(Profile.CASE_SENSITIVE, raw)
    assert plan.result.is_valid
    assert plan.result.conflicts == []


def test_case_alias_collision_returns_original_paths():
    plan = PathCompatibilityPlan(
        Profile.CASEFOLD_NFC, [_raw(("a",), b"1"), _raw(("A",), b"2")]
    )
    (group,) = plan.result.error_conflicts
    assert group.kind is ConflictKind.NAME_COLLISION
    assert group.normalized_key == ("a",)
    assert set(group.paths) == {("a",), ("A",)}


def test_unicode_composition_equivalence_collision():
    plan = PathCompatibilityPlan(
        Profile.CASEFOLD_NFC,
        [_raw((NFC_EACUTE,), b"1"), _raw((NFD_EACUTE,), b"2")],
    )
    (group,) = plan.result.error_conflicts
    assert group.kind is ConflictKind.NAME_COLLISION
    assert set(group.paths) == {(NFC_EACUTE,), (NFD_EACUTE,)}


def test_directory_alias_is_warning_but_fully_reported():
    plan = PathCompatibilityPlan(
        Profile.CASEFOLD_NFC,
        [_raw(("Dir", "a"), b"1"), _raw(("dir", "b"), b"2")],
    )
    (group,) = plan.result.conflicts
    assert group.kind is ConflictKind.CASE_DIRECTORY_MERGE
    assert group.severity == "warning"
    assert set(group.paths) == {("Dir",), ("dir",)}
    # directory merge alone does not block co-existence
    assert plan.result.is_valid


def test_directory_alias_with_same_child_is_hard_collision():
    plan = PathCompatibilityPlan(
        Profile.CASEFOLD_NFC,
        [_raw(("Dir", "a"), b"1"), _raw(("dir", "a"), b"2")],
    )
    kinds = {group.kind for group in plan.result.conflicts}
    assert ConflictKind.NAME_COLLISION in kinds
    assert ConflictKind.CASE_DIRECTORY_MERGE in kinds
    assert not plan.result.is_valid


def test_parent_file_conflicts_with_descendant():
    plan = PathCompatibilityPlan(
        Profile.CASE_SENSITIVE,
        [_raw(("foo",), b"1"), _raw(("foo", "bar"), b"2")],
    )
    (group,) = plan.result.error_conflicts
    assert group.kind is ConflictKind.PARENT_CHILD
    assert set(group.paths) == {("foo",), ("foo", "bar")}


def test_file_directory_type_mismatch():
    from dvc_data.path_compat import _RawEntry

    raw = [
        _RawEntry(("x",), True, None, None, "tree"),
        _RawEntry(("x",), False, 1, _hi(b"z"), "tree"),
    ]
    plan = PathCompatibilityPlan(Profile.CASE_SENSITIVE, raw)
    (group,) = plan.result.error_conflicts
    assert group.kind is ConflictKind.TYPE_MISMATCH


def test_windows_reserved_names_are_blocking():
    plan = PathCompatibilityPlan(
        Profile.WINDOWS_PORTABLE,
        [_raw(("CON",), b"1"), _raw(("dir", "nul.txt"), b"2"), _raw(("ok",), b"3")],
    )
    groups = plan.result.error_conflicts
    assert {group.kind for group in groups} == {ConflictKind.CONSTRAINT_VIOLATION}
    assert {group.paths[0] for group in groups} == {("CON",), ("dir", "nul.txt")}
    reasons = {reason for group in groups for reason in group.reasons}
    assert "reserved-name" in reasons


# ---------------------------------------------------------------------------
# Explicit rename map: chains, cycles, escapes and re-validation
# ---------------------------------------------------------------------------


def test_rename_chain_is_applied_simultaneously():
    # a->b, b->c, c->d, d->e: every target stays distinct (injective)
    keys = [("a",), ("b",), ("c",), ("d",)]
    plan = PathCompatibilityPlan(
        Profile.CASE_SENSITIVE,
        [_raw(key, str(i).encode()) for i, key in enumerate(keys)],
        rename_map={
            ("a",): ("b",),
            ("b",): ("c",),
            ("c",): ("d",),
            ("d",): ("e",),
        },
    )
    assert plan.result.is_valid
    assert [plan.result.target_for(key) for key in keys] == [
        ("b",),
        ("c",),
        ("d",),
        ("e",),
    ]


def test_rename_swap_cycle_is_valid_permutation():
    keys = [("a",), ("b",)]
    plan = PathCompatibilityPlan(
        Profile.CASE_SENSITIVE,
        [_raw(keys[0], b"1"), _raw(keys[1], b"2")],
        rename_map={("a",): ("b",), ("b",): ("a",)},
    )
    assert plan.result.is_valid
    assert plan.result.target_for(("a",)) == ("b",)
    assert plan.result.target_for(("b",)) == ("a",)


def test_three_node_rename_cycle_is_valid():
    keys = [("a",), ("b",), ("c",)]
    plan = PathCompatibilityPlan(
        Profile.CASE_SENSITIVE,
        [_raw(key, str(i).encode()) for i, key in enumerate(keys)],
        rename_map={("a",): ("b",), ("b",): ("c",), ("c",): ("a",)},
    )
    assert plan.result.is_valid
    assert {plan.result.target_for(key) for key in keys} == {
        ("a",),
        ("b",),
        ("c",),
    }


def test_broken_rename_chain_fails_full_injectivity_revalidation():
    plan = PathCompatibilityPlan(
        Profile.CASE_SENSITIVE,
        [_raw(("a",), b"1"), _raw(("b",), b"2")],
        rename_map={("a",): ("b",)},
    )
    (group,) = plan.result.error_conflicts
    assert group.kind is ConflictKind.RENAME_COLLISION
    assert set(group.paths) == {("a",), ("b",)}


def test_rename_cannot_escape_root():
    base = [_raw(("a",), b"1"), _raw(("b",), b"2")]
    for bad_target in (("..", "x"), ("a", "..", "b"), (".",), ("a/b",)):
        plan = PathCompatibilityPlan(
            Profile.CASE_SENSITIVE,
            base,
            rename_map={("a",): bad_target},
        )
        assert plan.result.error_conflicts[0].kind is ConflictKind.RENAME_ESCAPE
        with pytest.raises(PathCompatibilityError):
            plan.raise_for_conflicts()


def test_rename_into_reserved_name_is_rename_escape():
    plan = PathCompatibilityPlan(
        Profile.WINDOWS_PORTABLE,
        [_raw(("a",), b"1"), _raw(("b",), b"2")],
        rename_map={("a",): ("CON",)},
    )
    (group,) = plan.result.error_conflicts
    assert group.kind is ConflictKind.RENAME_ESCAPE
    assert "reserved-name" in group.reasons


def test_rename_map_unknown_source_is_rejected():
    with pytest.raises(RenameMapError):
        PathCompatibilityPlan(
            Profile.CASE_SENSITIVE,
            [_raw(("a",), b"1")],
            rename_map={("missing",): ("z",)},
        )


def test_rename_cannot_overwrite_unselected_workspace_file():
    plan = PathCompatibilityPlan(
        Profile.CASE_SENSITIVE,
        [_raw(("a",), b"1"), _raw(("b",), b"2")],
        rename_map={("a",): ("w",)},
        workspace_entries=[(("w",), False)],
    )
    (group,) = plan.result.error_conflicts
    assert group.kind is ConflictKind.WORKSPACE_OVERWRITE
    assert set(group.paths) == {("a",), ("w",)}


def test_rename_resolves_case_collision_and_revalidates():
    plan = PathCompatibilityPlan(
        Profile.WINDOWS_PORTABLE,
        [_raw(("a",), b"1"), _raw(("A",), b"2")],
        rename_map={("A",): ("b",)},
    )
    assert plan.result.is_valid
    assert plan.result.target_for(("A",)) == ("b",)
    plan.raise_for_conflicts()  # must not raise


# ---------------------------------------------------------------------------
# Same object at multiple paths
# ---------------------------------------------------------------------------


def test_same_object_multiple_paths_still_collides_under_folding():
    shared = _hi(b"same-bytes")
    from dvc_data.path_compat import _RawEntry

    raw = [
        _RawEntry(("a",), False, 9, shared, "tree"),
        _RawEntry(("A",), False, 9, shared, "tree"),
    ]
    plan = PathCompatibilityPlan(Profile.CASEFOLD_NFC, raw)
    (group,) = plan.result.error_conflicts
    assert group.kind is ConflictKind.NAME_COLLISION
    assert "identical-content" in group.reasons
    assert set(group.paths) == {("a",), ("A",)}

    # under case-sensitive rules the same object may be checked out twice
    ok = PathCompatibilityPlan(Profile.CASE_SENSITIVE, raw)
    assert ok.result.is_valid
    assert len(ok.result.entries) == 2


# ---------------------------------------------------------------------------
# Tree / Index remapping and digest recomputation
# ---------------------------------------------------------------------------


def _build_tree(entries: list[tuple[tuple[str, ...], bytes]]) -> Tree:
    tree = Tree()
    for key, content in entries:
        tree.add(key, Meta(size=len(content)), _hi(content))
    tree.digest()
    return tree


def test_mapped_tree_recomputes_digest_and_preserves_content_hashes():
    tree = _build_tree([(("a",), b"1"), (("sub", "b"), b"22")])
    original_oid = tree.oid
    original_list = tree.as_list()

    plan = PathCompatibilityPlan.from_tree(
        tree,
        Profile.CASE_SENSITIVE,
        rename_map={("a",): ("a2",), ("sub", "b"): ("sub", "b2")},
    )
    mapped = plan.mapped_tree(tree)

    # new directory digest derived from the new paths, old .dir hash dropped
    assert mapped.oid != original_oid
    assert mapped.oid.endswith(".dir")
    # the source tree is never mutated and keeps its old digest
    assert tree.oid == original_oid
    assert tree.as_list() == original_list

    relpaths = [entry["relpath"] for entry in mapped.as_list()]
    assert relpaths == ["a2", "sub/b2"]
    # file object identities are untouched by path mapping
    assert {entry[1] for entry in mapped.as_dict().values()} == {
        _hi(b"1"),
        _hi(b"22"),
    }
    # the new digest is the real hash of the re-keyed serialization
    rerun = Tree.from_list(mapped.as_list())
    rerun.digest()
    assert rerun.oid == mapped.oid


def test_mapped_tree_refuses_when_plan_invalid():
    tree = _build_tree([(("a",), b"1"), (("A",), b"2")])
    plan = PathCompatibilityPlan.from_tree(tree, Profile.WINDOWS_PORTABLE)
    assert not plan.result.is_valid
    with pytest.raises(PathCompatibilityError):
        plan.mapped_tree(tree)
    # nothing was written to the source tree
    assert tree.oid is not None


def test_from_index_and_mapped_index_rekeys_entries():
    index = DataIndex()
    index[("a",)] = DataIndexEntry(key=("a",), meta=Meta(size=1), hash_info=_hi(b"1"))
    index[("A",)] = DataIndexEntry(key=("A",), meta=Meta(size=1), hash_info=_hi(b"2"))
    index[("d",)] = DataIndexEntry(key=("d",), meta=Meta(isdir=True), loaded=True)

    bad = PathCompatibilityPlan.from_index(index, Profile.CASEFOLD_NFC)
    assert not bad.result.is_valid

    plan = PathCompatibilityPlan.from_index(
        index, Profile.CASEFOLD_NFC, rename_map={("A",): ("b",)}
    )
    mapped = plan.mapped_index(index)
    assert set(mapped.keys()) == {("a",), ("b",), ("d",)}
    assert mapped[("b",)].hash_info == _hi(b"2")
    # original index is untouched
    assert set(index.keys()) == {("a",), ("A",), ("d",)}


# ---------------------------------------------------------------------------
# Workspace normalization domain and unknown scope
# ---------------------------------------------------------------------------


def test_workspace_same_key_overwrite_is_intended():
    plan = PathCompatibilityPlan(
        Profile.CASE_SENSITIVE,
        [_raw(("a",), b"1")],
        workspace_entries=[(("a",), False)],
    )
    assert plan.result.is_valid
    assert plan.result.workspace_scope == "observed"


def test_unknown_scope_is_reported_not_silently_cleared():
    plan = PathCompatibilityPlan(
        Profile.CASE_SENSITIVE, [_raw(("a",), b"1")], workspace_observed=False
    )
    assert plan.result.has_unknown_scope
    assert not plan.result.workspace_certified
    # internal namespace is still fully validated
    assert plan.result.is_valid


def test_observed_workspace_enters_same_normalization_domain():
    plan = PathCompatibilityPlan(
        Profile.CASEFOLD_NFC,
        [_raw(("a",), b"1")],
        workspace_entries=[(("A",), False)],
    )
    (group,) = plan.result.error_conflicts
    assert group.kind is ConflictKind.WORKSPACE_OVERWRITE


# ---------------------------------------------------------------------------
# Resource caps and read-only guarantee
# ---------------------------------------------------------------------------


def test_path_count_depth_and_byte_limits_abort_before_result():
    raw = [_raw((f"f{i}",), b"x") for i in range(5)]
    with pytest.raises(PathPlanLimitError):
        PathCompatibilityPlan(
            Profile.CASE_SENSITIVE, raw, limits=PlanLimits(max_paths=3)
        )
    with pytest.raises(PathPlanLimitError):
        PathCompatibilityPlan(
            Profile.CASE_SENSITIVE,
            [_raw(("a", "b", "c"), b"x")],
            limits=PlanLimits(max_depth=2),
        )
    with pytest.raises(PathPlanLimitError):
        PathCompatibilityPlan(
            Profile.CASE_SENSITIVE,
            [_raw(("a",), b"x" * 10)],
            limits=PlanLimits(max_total_bytes=4),
        )


def test_conflict_report_is_capped_but_exception_keeps_full_set():
    raw = []
    for i in range(4):
        raw.append(_raw((f"f{i}",), str(i).encode()))
        raw.append(_raw((f"F{i}",), str(i).encode()))
    plan = PathCompatibilityPlan(
        Profile.CASEFOLD_NFC, raw, limits=PlanLimits(max_conflict_groups=2)
    )
    assert plan.result.conflicts_truncated
    assert len(plan.result.conflicts) == 2
    with pytest.raises(PathCompatibilityError) as exc:
        plan.raise_for_conflicts()
    assert len(exc.value.conflicts) == 4


def test_planning_is_read_only():
    tree = _build_tree([(("a",), b"1"), (("A",), b"2")])
    snapshot = tree.as_dict()
    oid = tree.oid
    plan = PathCompatibilityPlan.from_tree(tree, Profile.WINDOWS_PORTABLE)
    assert not plan.result.is_valid
    # failed planning leaves the in-memory tree and its digest untouched
    assert tree.as_dict() == snapshot
    assert tree.oid == oid


def test_stats_are_collected():
    raw = [
        _raw(("a",), b"1234"),
        _raw(("d", "b"), b"56"),
    ]
    plan = PathCompatibilityPlan(Profile.CASE_SENSITIVE, raw)
    stats = plan.result.stats
    assert stats.n_paths == 2
    assert stats.n_files == 2
    assert stats.total_bytes == 6
    assert stats.max_depth == 2
    assert stats.n_unknown_size == 0


# ---------------------------------------------------------------------------
# Physical checkout on filesystems that implement the target profile
# ---------------------------------------------------------------------------


def _build_odb_tree(tmp_path, entries):
    fs = LocalFileSystem()
    odb = HashFileDB(fs, str(tmp_path / "odb"))
    tree = Tree()
    for key, content in entries:
        oid = hashlib.md5(content).hexdigest()  # noqa: S324
        odb.add_bytes(oid, content)
        tree.add(key, Meta(size=len(content)), HashInfo("md5", oid))
    tree.digest()
    return fs, odb, tree


def test_case_sensitive_plan_on_case_sensitive_memory_fs(tmp_path):
    mem = MemoryFileSystem()
    odb = HashFileDB(mem, "memory:///odb")
    tree = Tree()
    for name, content in [("a", b"lower"), ("A", b"upper")]:
        oid = hashlib.md5(content).hexdigest()  # noqa: S324
        odb.add_bytes(oid, content)
        tree.add((name,), Meta(size=len(content)), HashInfo("md5", oid))
    tree.digest()

    plan = PathCompatibilityPlan.from_tree(tree, Profile.CASE_SENSITIVE)
    mapped = plan.mapped_tree(tree)

    dest = "memory:///case-sensitive-work"
    checkout(dest, mem, mapped, odb, force=True)
    listing = {path.rsplit("/", 1)[-1] for path in mem.ls(dest, detail=False)}
    assert listing == {"a", "A"}
    with mem.open(f"{dest}/a", "rb") as f:
        assert f.read() == b"lower"
    with mem.open(f"{dest}/A", "rb") as f:
        assert f.read() == b"upper"


@pytest.mark.skipif(os.name != "nt", reason="NTFS case folding is Windows-only")
def test_windows_portable_plan_prevents_case_collision_on_ntfs(tmp_path):
    entries = [(("a",), b"lower"), (("A",), b"upper")]
    fs, odb, tree = _build_odb_tree(tmp_path, entries)

    # without a plan, raw checkout on NTFS collapses both sources onto one
    raw_dest = tmp_path / "raw"
    raw_dest.mkdir()
    checkout(str(raw_dest), fs, tree, odb, force=True)
    assert len(list(raw_dest.iterdir())) == 1

    # the windows-portable profile rejects the un-mapped layout
    bad = PathCompatibilityPlan.from_tree(tree, Profile.WINDOWS_PORTABLE)
    assert not bad.result.is_valid

    # an explicit rename map makes the plan pass
    plan = PathCompatibilityPlan.from_tree(
        tree, Profile.WINDOWS_PORTABLE, rename_map={("A",): ("b",)}
    )
    assert plan.result.workspace_certified
    mapped = plan.mapped_tree(tree)

    safe_dest = tmp_path / "safe"
    safe_dest.mkdir()
    checkout(str(safe_dest), fs, mapped, odb, force=True)
    files = {path.name: path.read_bytes() for path in safe_dest.iterdir()}
    assert files == {"a": b"lower", "b": b"upper"}


@pytest.mark.skipif(os.name != "nt", reason="NTFS physical check is Windows-only")
def test_windows_portable_plan_separates_unicode_aliases_on_ntfs(tmp_path):
    entries = [((NFC_EACUTE,), b"nfc-bytes"), ((NFD_EACUTE,), b"nfd-bytes")]
    fs, odb, tree = _build_odb_tree(tmp_path, entries)

    bad = PathCompatibilityPlan.from_tree(tree, Profile.CASEFOLD_NFC)
    assert not bad.result.is_valid

    plan = PathCompatibilityPlan.from_tree(
        tree,
        Profile.CASEFOLD_NFC,
        rename_map={(NFD_EACUTE,): ("nfd-" + NFD_EACUTE,)},
    )
    mapped = plan.mapped_tree(tree)
    dest = tmp_path / "uni"
    dest.mkdir()
    checkout(str(dest), fs, mapped, odb, force=True)
    contents = {path.read_bytes() for path in dest.iterdir()}
    assert contents == {b"nfc-bytes", b"nfd-bytes"}
    assert len(list(dest.iterdir())) == 2


@pytest.mark.skipif(os.name != "nt", reason="reserved device names are Windows-only")
def test_windows_portable_plan_avoids_reserved_name_on_ntfs(tmp_path):
    entries = [(("CON",), b"device"), (("data",), b"payload")]
    fs, odb, tree = _build_odb_tree(tmp_path, entries)

    bad = PathCompatibilityPlan.from_tree(tree, Profile.WINDOWS_PORTABLE)
    assert any("reserved-name" in group.reasons for group in bad.result.error_conflicts)

    plan = PathCompatibilityPlan.from_tree(
        tree,
        Profile.WINDOWS_PORTABLE,
        rename_map={("CON",): ("con-renamed",)},
    )
    mapped = plan.mapped_tree(tree)
    dest = tmp_path / "reserved"
    dest.mkdir()
    checkout(str(dest), fs, mapped, odb, force=True)
    files = {path.name: path.read_bytes() for path in dest.iterdir()}
    assert files == {"con-renamed": b"device", "data": b"payload"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _raw(key: tuple[str, ...], content: bytes):
    from dvc_data.path_compat import _RawEntry

    return _RawEntry(key, False, len(content), _hi(content), "tree")
