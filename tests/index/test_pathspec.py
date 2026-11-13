import fnmatch
import random
from typing import TYPE_CHECKING

import pytest
from dvc_objects.fs.local import LocalFileSystem

from dvc_data.hashfile.hash_info import HashInfo
from dvc_data.hashfile.meta import Meta
from dvc_data.index import (
    DataIndex,
    DataIndexEntry,
    FileStorage,
    ObjectStorage,
    PathSpecBudget,
    PathSpecBudgetError,
    PathSpecError,
    PathSpecView,
    compile_pathspec,
    filter_sparse_deletes,
    parse_pattern,
    pathspec_view,
    sparse_diff,
)
from dvc_data.index.checkout import apply, compare

if TYPE_CHECKING:
    from dvc_data.index import CompiledPathSpec

localfs = LocalFileSystem()


# ---------------------------------------------------------------------------
# pattern parsing
# ---------------------------------------------------------------------------


class TestParsePattern:
    def test_split_on_forward_slash_only(self):
        assert parse_pattern("a/b/c") == ("a", "b", "c")
        assert parse_pattern(["a", "b"]) == ("a", "b")
        assert parse_pattern(("a", "b")) == ("a", "b")

    def test_empty_pattern_is_root(self):
        assert parse_pattern("") == ()
        assert parse_pattern(()) == ()

    def test_empty_component_rejected(self):
        for pattern in ("a//b", "/a", "a/", ( "a", "")):
            with pytest.raises(PathSpecError):
                parse_pattern(pattern)

    def test_escaped_separator_stays_in_component(self):
        components = parse_pattern("a\\/b/c")
        assert components == ("a\\/b", "c")

    def test_dangling_escape_rejected(self):
        with pytest.raises(PathSpecError):
            parse_pattern("a\\")

    def test_globstar_only_special_as_whole_component(self):
        spec = compile_pathspec([(True, "a/**/b")])
        assert spec.plan_state(("a", "b")).selected  # zero levels
        assert spec.plan_state(("a", "x", "b")).selected
        assert spec.plan_state(("a", "x", "y", "b")).selected

    def test_escaped_metachars_are_literal(self):
        spec = compile_pathspec([(True, "a/\\*.txt")])
        assert spec.plan_state(("a", "*.txt")).selected
        assert not spec.plan_state(("a", "x.txt")).selected

    def test_component_glob_does_not_cross_components(self):
        spec = compile_pathspec([(True, "*.py")])
        assert spec.plan_state(("a.py",)).selected
        assert not spec.plan_state(("dir", "a.py")).selected

        spec = compile_pathspec([(True, "a?c")])
        assert spec.plan_state(("abc",)).selected
        assert not spec.plan_state(("ac",)).selected

        spec = compile_pathspec([(True, "[!abc]")])
        assert spec.plan_state(("d",)).selected
        assert not spec.plan_state(("a",)).selected

    def test_invalid_action(self):
        with pytest.raises(PathSpecError):
            compile_pathspec([("maybe", "a")])


# ---------------------------------------------------------------------------
# ordered rules, inheritance and pruning
# ---------------------------------------------------------------------------


class TestCompiledSpec:
    def test_last_matching_rule_wins(self):
        spec = compile_pathspec(
            [(True, "a/**"), (False, "a/secret"), (True, "a/secret/keep")]
        )
        assert spec.plan_state(("a",)).selected
        assert spec.plan_state(("a", "x", "y")).selected
        assert not spec.plan_state(("a", "secret")).selected
        assert not spec.plan_state(("a", "secret", "other")).selected
        assert spec.plan_state(("a", "secret", "keep")).selected
        # re-include inherits to its own descendants
        assert spec.plan_state(("a", "secret", "keep", "x")).selected
        assert not spec.plan_state(("b",)).selected

    def test_default_exclude(self):
        spec = compile_pathspec([(True, "a")])
        assert not spec.plan_state(("b",)).selected
        spec = compile_pathspec([(False, "a")], default=True)
        assert spec.plan_state(("b",)).selected
        assert not spec.plan_state(("a",)).selected

    def test_root_rule_inherits_everything(self):
        include_all = compile_pathspec([(True, "")])
        assert include_all.plan_state(()).selected
        assert include_all.plan_state(("x", "y")).selected

        exclude_all = compile_pathspec([(False, "")])
        assert exclude_all.plan_state(("x",)).prunable

    def test_excluded_parent_with_reincluded_descendant_is_not_pruned(self):
        spec = compile_pathspec([(False, "dir"), (True, "dir/b")])
        state = spec.plan_state(("dir",))
        assert not state.selected
        assert not state.prunable  # must load to find re-included "b"

        state = spec.plan_state(("dir", "c"))
        assert not state.selected
        assert state.prunable  # nothing includeable below "dir/c"

    def test_pure_exclude_subtree_is_prunable(self):
        spec = compile_pathspec([(True, "keep")])
        state = spec.plan_state(("gone",))
        assert not state.selected
        assert state.prunable

    def test_globstar_pruning_considers_zero_level(self):
        spec = compile_pathspec([(True, "a/**/b")])
        # "a/b" itself is selectable, so "a" is not prunable
        assert not spec.plan_state(("a",)).prunable
        assert spec.plan_state(("a", "b")).selected
        # no include rule reaches below an unrelated branch
        assert spec.plan_state(("other",)).prunable

    def test_pattern_budget_errors(self):
        with pytest.raises(PathSpecBudgetError):
            compile_pathspec(
                [(True, f"p{i}") for i in range(5)],
                budget=PathSpecBudget(max_patterns=4),
            )
        with pytest.raises(PathSpecBudgetError):
            compile_pathspec(
                [(True, "a/b/c")], budget=PathSpecBudget(max_depth=2)
            )

    def test_state_budget_disables_pruning_safely(self):
        spec = compile_pathspec(
            [(True, "keep")], budget=PathSpecBudget(max_states=2)
        )
        # exhaust the state budget with unrelated planning states
        for name in ("x1", "x2", "x3"):
            spec.plan_state((name,))
        assert spec.incomplete
        # selection stays correct, pruning is switched off (safe fallback)
        state = spec.plan_state(("gone",))
        assert not state.selected
        assert not state.prunable
        assert spec.plan_state(("keep",)).selected


# ---------------------------------------------------------------------------
# independent reference matcher cross-check on generated small trees
# ---------------------------------------------------------------------------


def _ref_exact(pattern: tuple[str, ...], key: tuple[str, ...]) -> bool:
    """Independent recursive matcher, fnmatch-based per component."""

    def walk(pi: int, ki: int) -> bool:
        if pi == len(pattern):
            return ki == len(key)
        part = pattern[pi]
        if part == "**":
            # zero components ...
            if walk(pi + 1, ki):
                return True
            # ... or one/more components
            return ki < len(key) and walk(pi, ki + 1)
        if ki >= len(key):
            return False
        return fnmatch.fnmatchcase(key[ki], part) and walk(pi + 1, ki + 1)

    return walk(0, 0)


def _ref_selected(rules, key: tuple[str, ...], default: bool = False) -> bool:
    """Independent last-wins reference including ancestor inheritance."""
    action = default
    for include, pattern in rules:
        for depth in range(len(key) + 1):
            if _ref_exact(pattern, key[:depth]):
                action = include
                break
    return action


def _make_eager_index(keys: set[tuple[str, ...]]) -> DataIndex:
    index = DataIndex()
    prefixes = {
        key[:depth]
        for key in keys
        for depth in range(1, len(key))
    }
    for key in keys | prefixes:
        is_dir = key in prefixes or any(
            other[: len(key)] == key and other != key for other in keys
        )
        index[key] = DataIndexEntry(
            key=key,
            meta=Meta(isdir=is_dir),
            loaded=True if is_dir else None,
        )
    return index


@pytest.mark.parametrize("seed", range(25))
def test_selection_matches_independent_reference(seed):
    rng = random.Random(seed)  # noqa: S311 - deterministic test fixture only
    names = ["a", "b", "c", "foo", "bar"]
    globs = ["*", "*.py", "?b", "[ab]", "a*"]

    all_keys: set[tuple[str, ...]] = set()
    for _ in range(rng.randint(6, 14)):
        depth = rng.randint(1, 3)
        all_keys.add(tuple(rng.choice(names) for _ in range(depth)))

    index = _make_eager_index(all_keys)

    rules = []
    parsed_rules = []
    for _ in range(rng.randint(1, 6)):
        include = rng.choice([True, False])
        components = []
        for _ in range(rng.randint(1, 3)):
            if rng.random() < 0.25:
                components.append("**")
            elif rng.random() < 0.4:
                components.append(rng.choice(globs))
            else:
                components.append(rng.choice(names))
        pattern = "/".join(components)
        rules.append((include, pattern))
        parsed_rules.append((include, tuple(components)))

    default = rng.choice([True, False])
    view = pathspec_view(index, rules, default=default)

    index_keys = set(index.keys())
    expected = {
        key for key in index_keys if _ref_selected(parsed_rules, key, default)
    }
    assert set(view.keys()) == expected

    # every observed key (including synthesized directory prefixes) agrees
    # with the independent reference
    for key in index_keys:
        assert view.filter_fn(key) == _ref_selected(parsed_rules, key, default)

    report = view.coverage_report()
    assert not report.incomplete


# ---------------------------------------------------------------------------
# lazy loading, pruning and coverage
# ---------------------------------------------------------------------------


def _make_lazy_index(tmp_path, layout: dict[str, str]) -> DataIndex:
    """Build an index whose directories are unloaded lazy FileStorage entries."""
    src = tmp_path / "src"
    for relpath, content in layout.items():
        path = src / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    index = DataIndex()
    index.storage_map.add_data(FileStorage((), localfs, str(src)))

    top_dirs = {relpath.split("/")[0] for relpath in layout if "/" in relpath}
    top_files = {relpath for relpath in layout if "/" not in relpath}
    for name in top_dirs:
        index[(name,)] = DataIndexEntry(
            key=(name,), meta=Meta(isdir=True)
        )  # loaded is None -> lazy
    for name in top_files:
        full = src / name
        index[(name,)] = DataIndexEntry(
            key=(name,), meta=Meta(size=full.stat().st_size)
        )
    return index


class TestLazyTraversal:
    def test_excluded_subtree_is_never_loaded(self, tmp_path, mocker):
        layout = {
            "keep.txt": "keep",
            "included/keep.txt": "in",
            "excluded/secret.txt": "secret",
            "excluded/nested/deep.txt": "deep",
        }
        index = _make_lazy_index(tmp_path, layout)
        load = mocker.spy(index, "_load")

        view = pathspec_view(index, [(True, "included/**"), (True, "keep.txt")])
        selected = set(view.keys())

        assert selected == {("keep.txt",), ("included",), ("included", "keep.txt")}
        loaded_keys = [call.args[0] for call in load.call_args_list]
        assert ("excluded",) not in loaded_keys
        # excluded entries never materialized in the base trie
        assert ("excluded", "secret.txt") not in index._trie

        report = view.coverage_report()
        assert ("excluded",) in report.pruned_prefixes
        assert ("included",) in report.covered_prefixes
        assert () in report.covered_prefixes
        assert not report.unknown_prefixes
        assert not report.incomplete

    def test_reincluded_descendant_forces_parent_load(self, tmp_path, mocker):
        layout = {
            "dir/excluded.txt": "x",
            "dir/keep/yes.txt": "y",
        }
        index = _make_lazy_index(tmp_path, layout)
        load = mocker.spy(index, "_load")

        view = pathspec_view(
            index,
            [(False, "dir"), (True, "dir/keep/**")],
        )
        selected = set(view.keys())
        assert selected == {("dir", "keep"), ("dir", "keep", "yes.txt")}

        loaded_keys = [call.args[0] for call in load.call_args_list]
        assert ("dir",) in loaded_keys  # parent had to be loaded
        # dir/excluded.txt is a file sibling: visible in trie but not selected
        assert ("dir", "excluded.txt") not in selected

    def test_unknown_directory_is_not_absent(self, tmp_path):
        index = DataIndex()
        # storage points at a nonexistent path -> every load fails
        index.storage_map.add_data(
            FileStorage((), localfs, str(tmp_path / "does-not-exist"))
        )
        index[("broken",)] = DataIndexEntry(
            key=("broken",), meta=Meta(isdir=True)
        )
        index[("file",)] = DataIndexEntry(key=("file",), meta=Meta(size=1))

        view = pathspec_view(index, [(True, "**")])
        # the broken directory entry itself is known (present in the trie),
        # only its contents are unobservable
        assert set(view.keys()) == {("file",), ("broken",)}

        report = view.coverage_report()
        assert ("broken",) in report.unknown_prefixes
        assert not report.incomplete  # load failure is unknown, not a budget cut
        # unknown means "we don't know", never "target has no such file"
        assert not view.is_fully_covered(("broken", "anything"))
        assert not view.is_safe_to_delete(("broken", "anything"))

    def test_runtime_budgets_report_incomplete(self, tmp_path):
        layout = {
            "d1/f.txt": "1",
            "d2/f.txt": "2",
            "d3/f.txt": "3",
        }
        index = _make_lazy_index(tmp_path, layout)

        view = pathspec_view(
            index, [(True, "**")], budget=PathSpecBudget(max_lazy_reads=1)
        )
        list(view.iteritems())
        assert view.incomplete
        report = view.coverage_report()
        assert report.incomplete
        assert report.unknown_prefixes  # cut directories stay unknown

        index2 = _make_lazy_index(tmp_path, layout)
        view2 = pathspec_view(
            index2, [(True, "**")], budget=PathSpecBudget(max_entries=1)
        )
        assert len(list(view2.iteritems())) <= 1
        assert view2.incomplete

        index3 = _make_lazy_index(tmp_path, layout)
        view3 = pathspec_view(
            index3, [(True, "**")], budget=PathSpecBudget(max_depth=1)
        )
        list(view3.iteritems())
        assert view3.incomplete

    def test_shallow_and_prefix_iteration(self, tmp_path):
        layout = {"d/sub/f.txt": "x", "top.txt": "y"}
        index = _make_lazy_index(tmp_path, layout)
        view = pathspec_view(index, [(True, "**")])

        shallow = {key for key, _ in view.iteritems(shallow=True)}
        assert shallow == {("d",), ("top.txt",)}

        under_d = {key for key, _ in view.iteritems(prefix=("d",))}
        assert under_d == {
            ("d",),
            ("d", "sub"),
            ("d", "sub", "f.txt"),
        }


# ---------------------------------------------------------------------------
# view isolation, storage routing, read-only contract
# ---------------------------------------------------------------------------


class TestViewIsolation:
    def test_two_specs_on_same_base_are_independent(self, tmp_path):
        layout = {"a/f1.txt": "1", "b/f2.txt": "2", "top.txt": "3"}
        index = _make_lazy_index(tmp_path, layout)

        view_a = pathspec_view(index, [(True, "a/**")])
        view_b = pathspec_view(index, [(True, "b/**"), (True, "top.txt")])

        keys_a = set(view_a.keys())
        keys_b = set(view_b.keys())
        assert keys_a == {("a",), ("a", "f1.txt")}
        assert keys_b == {("b",), ("b", "f2.txt"), ("top.txt",)}
        assert keys_a.isdisjoint(keys_b)
        # both views keep working against the same base index
        assert set(view_a.keys()) == keys_a
        assert set(view_b.keys()) == keys_b

    def test_onerror_is_not_propagated_to_base(self, tmp_path):
        index = _make_lazy_index(tmp_path, {"a/f.txt": "x"})
        view = pathspec_view(index, [(True, "**")])
        original = index.onerror

        def custom_onerror(entry, exc):
            pass

        view.onerror = custom_onerror
        assert index.onerror is original
        assert view.onerror is custom_onerror

        # compare() temporarily reassigns new.onerror and restores it;
        # the base index handler must remain untouched throughout.
        compare(None, view)
        assert index.onerror is original

    def test_storage_map_is_delegated_unchanged(self, tmp_path):
        index = _make_lazy_index(tmp_path, {"a/f.txt": "x"})
        view = pathspec_view(index, [(True, "**")])
        assert view.storage_map is index.storage_map
        # FileStorage routing follows the original entry
        entry = view[("a", "f.txt")]
        fs, path = view.storage_map.get_data(entry)
        assert fs is localfs
        assert path == str(tmp_path / "src" / "a" / "f.txt")

    def test_object_storage_routing_inherited(self, odb):
        index = DataIndex(
            {
                ("data",): DataIndexEntry(
                    key=("data",),
                    meta=Meta(isdir=True),
                    hash_info=HashInfo(
                        name="md5",
                        value="1f69c66028c35037e8bf67e5bc4ceb6a.dir",
                    ),
                ),
            }
        )
        index.storage_map.add_cache(ObjectStorage((), odb))
        view = pathspec_view(index, [(True, "data/**")])

        assert view.storage_map is index.storage_map
        keys = set(view.keys())
        assert keys == {("data",), ("data", "bar"), ("data", "baz")}
        entry = view[("data", "bar")]
        fs, path = view.storage_map.get_cache(entry)
        assert fs is odb.fs

    def test_view_is_read_only(self, tmp_path):
        index = _make_lazy_index(tmp_path, {"a/f.txt": "x"})
        view = pathspec_view(index, [(True, "**")])
        with pytest.raises(TypeError):
            view[("a",)] = DataIndexEntry(key=("a",))
        with pytest.raises(TypeError):
            del view[("a",)]
        with pytest.raises(TypeError):
            view.delete_node(("a",))

    def test_mapping_interface(self, tmp_path):
        index = _make_lazy_index(tmp_path, {"a/f.txt": "x", "b/g.txt": "y"})
        view = pathspec_view(index, [(True, "a/**")])
        assert ("a", "f.txt") in view
        assert ("b", "g.txt") not in view
        assert len(view) == 2
        assert view.has_node(("a", "f.txt"))
        assert not view.has_node(("b", "g.txt"))
        with pytest.raises(KeyError):
            view[("b", "g.txt")]


# ---------------------------------------------------------------------------
# file <-> directory transitions
# ---------------------------------------------------------------------------


def test_file_dir_transition(tmp_path):
    src = tmp_path / "src"
    (src / "a").mkdir(parents=True)
    (src / "a" / "child.txt").write_text("child")

    index = DataIndex()
    index.storage_map.add_data(FileStorage((), localfs, str(src)))

    # start with a *file* entry at "a"
    index[("a",)] = DataIndexEntry(key=("a",), meta=Meta(size=5))
    view = pathspec_view(index, [(True, "a/**")])
    assert set(view.keys()) == {("a",)}

    # replace it with an unloaded directory at the same key
    index[("a",)] = DataIndexEntry(key=("a",), meta=Meta(isdir=True))
    assert set(view.keys()) == {("a",), ("a", "child.txt")}

    # and back to a file
    index[("a",)] = DataIndexEntry(key=("a",), meta=Meta(size=5))
    assert set(view.keys()) == {("a",)}


# ---------------------------------------------------------------------------
# sparse diff safety and local sparse checkout
# ---------------------------------------------------------------------------


def _build_file_index(root) -> DataIndex:
    index = DataIndex()
    index.storage_map.add_data(FileStorage((), localfs, str(root)))
    for path in root.rglob("*"):
        if path.is_file():
            key = tuple(path.relative_to(root).parts)
            index[key] = DataIndexEntry(
                key=key, meta=Meta(size=path.stat().st_size), loaded=True
            )
        else:
            key = tuple(path.relative_to(root).parts)
            index[key] = DataIndexEntry(
                key=key, meta=Meta(isdir=True), loaded=True
            )
    return index


class TestSparseDiff:
    def test_unsafe_deletes_are_suppressed(self, tmp_path):
        src = tmp_path / "src"
        for relpath, content in {
            "keep.txt": "keep",
            "data/bar": "bar",
            "data/baz": "baz",
            "skip/x": "x",
        }.items():
            path = src / relpath
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)

        index = _build_file_index(src)
        view = pathspec_view(index, [(True, "keep.txt"), (True, "data/bar")])

        diff = compare(index, view, delete=True)
        # the naive diff would indeed delete siblings and the skipped tree
        assert diff.files_delete
        naive_delete_keys = {entry.key for entry in diff.files_delete}
        assert ("data", "baz") in naive_delete_keys

        filter_sparse_deletes(diff, view)
        assert diff.files_delete == []
        assert diff.dirs_delete == []

        # generator-level check
        from dvc_data.index.diff import DELETE

        delete_changes = [
            change for change in sparse_diff(index, view) if change.typ == DELETE
        ]
        assert delete_changes == []

    def test_sparse_checkout_keeps_unselected_bytes(self, tmp_path):
        src = tmp_path / "src"
        files = {
            "keep.txt": b"keep-bytes",
            "data/bar": b"bar-bytes",
            "data/baz": b"baz-bytes",
            "skip/x": b"skip-bytes",
        }
        for relpath, content in files.items():
            path = src / relpath
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)

        index = _build_file_index(src)
        workspace = tmp_path / "ws"
        # full checkout first
        apply(
            compare(None, index),
            str(workspace),
            localfs,
            update_meta=False,
            storage="data",
        )
        original = {
            relpath: (workspace / relpath).read_bytes() for relpath in files
        }

        # sparse view selecting one file inside data/ plus keep.txt
        view = pathspec_view(index, [(True, "keep.txt"), (True, "data/bar")])
        diff = compare(index, view, delete=True)
        filter_sparse_deletes(diff, view)
        apply(
            diff,
            str(workspace),
            localfs,
            update_meta=False,
            storage="data",
        )

        # every workspace file survives with its original bytes
        for relpath, content in original.items():
            assert (workspace / relpath).read_bytes() == content

    def test_covered_selected_deletion_still_happens(self, tmp_path):
        full_src = tmp_path / "full"
        new_src = tmp_path / "new"
        for relpath, content in {
            "keep.txt": b"keep",
            "data/bar": b"bar",
            "data/baz": b"baz",
            "skip/x": b"skip",
        }.items():
            path = full_src / relpath
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        (new_src / "data").mkdir(parents=True)
        (new_src / "data" / "bar").write_bytes(b"bar")

        old = _build_file_index(full_src)
        new = _build_file_index(new_src)
        view = pathspec_view(new, [(True, "data/**")])

        workspace = tmp_path / "ws"
        apply(
            compare(None, old),
            str(workspace),
            localfs,
            update_meta=False,
            storage="data",
        )

        diff = compare(old, view, delete=True)
        naive = {entry.key for entry in diff.files_delete}
        # everything missing in the sparse new index looks like a delete
        assert ("data", "baz") in naive
        assert ("keep.txt",) in naive

        filter_sparse_deletes(diff, view)
        safe = {entry.key for entry in diff.files_delete}
        # only the selected, fully covered data subtree may lose a file
        assert safe == {("data", "baz")}

        apply(
            diff,
            str(workspace),
            localfs,
            update_meta=False,
            storage="data",
        )
        assert not (workspace / "data" / "baz").exists()
        assert (workspace / "data" / "bar").read_bytes() == b"bar"
        # unselected workspace files are untouched
        assert (workspace / "keep.txt").read_bytes() == b"keep"
        assert (workspace / "skip" / "x").read_bytes() == b"skip"

    def test_selecting_one_file_never_deletes_siblings(self, tmp_path):
        src = tmp_path / "src"
        for name in ("a", "b", "c"):
            (src / "dir").mkdir(parents=True, exist_ok=True)
            (src / "dir" / name).write_text(name)
        index = _build_file_index(src)
        view = pathspec_view(index, [(True, "dir/a")])

        report = view.coverage_report()
        # the directory itself was fully observed ...
        assert ("dir",) in report.covered_prefixes
        # ... but siblings are not selected, so deleting them is unsafe
        assert not view.is_safe_to_delete(("dir", "b"))
        assert view.is_safe_to_delete(("dir", "a"))

        diff = compare(index, view, delete=True)
        filter_sparse_deletes(diff, view)
        assert diff.files_delete == []


def test_compiled_spec_is_reusable(tmp_path):
    layout = {"a/f.txt": "1", "b/g.txt": "2"}
    index1 = _make_lazy_index(tmp_path / "one", layout)
    index2 = _make_lazy_index(tmp_path / "two", layout)
    spec: CompiledPathSpec = compile_pathspec([(True, "a/**")])

    view1 = PathSpecView(index1, spec)
    view2 = PathSpecView(index2, spec)
    assert set(view1.keys()) == set(view2.keys()) == {("a",), ("a", "f.txt")}
    # each view keeps independent runtime counters and coverage state
    assert view1.lazy_reads == view2.lazy_reads == 1
    view1.coverage_report()
    assert view1._drained is True
    assert view2._drained is False
