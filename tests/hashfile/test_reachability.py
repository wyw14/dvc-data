import hashlib
import json

import pytest
from dvc_objects.fs.local import LocalFileSystem
from dvc_objects.fs.memory import MemoryFileSystem

from dvc_data.hashfile.db import HashFileDB
from dvc_data.hashfile.db.local import LocalHashFileDB
from dvc_data.hashfile.hash_info import HashInfo
from dvc_data.hashfile.meta import Meta
from dvc_data.hashfile.reachability import (
    METADATA,
    MISSING,
    OK,
    UNCHECKED,
    VERIFY,
    Budget,
    ExplainReachability,
    explain_reachability,
)
from dvc_data.hashfile.state import StateNoop
from dvc_data.hashfile.tree import Tree

# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #


class CountingDB(HashFileDB):
    """HashFileDB recording how often each OID is probed for existence."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exists_calls: dict[str, int] = {}

    def exists(self, oid):
        self.exists_calls[oid] = self.exists_calls.get(oid, 0) + 1
        return super().exists(oid)


class RaisingState(StateNoop):
    def save(self, *args, **kwargs):
        raise AssertionError("state.save must not be called by the audit")

    def save_many(self, *args, **kwargs):
        raise AssertionError("state.save_many must not be called by the audit")


class GuardedLocalDB(LocalHashFileDB):
    """Local ODB that fails the test on any mutating/destructive call."""

    def check(self, *args, **kwargs):
        raise AssertionError("destructive HashFileDB.check must not be called")

    def add(self, *args, **kwargs):
        raise AssertionError("HashFileDB.add must not be called")

    def delete(self, oid):
        raise AssertionError("HashFileDB.delete must not be called")


def make_odb(hash_name: str = "md5", path: str | None = None) -> CountingDB:
    fs = MemoryFileSystem(global_store=False)
    return CountingDB(
        fs, path if path is not None else fs.root_marker, hash_name=hash_name
    )


def put_leaf(odb, data: bytes, name: str = "md5") -> HashInfo:
    oid = hashlib.new(name, data).hexdigest()
    odb.add_bytes(oid, data)
    return HashInfo(name, oid)


def put_tree(odb, entries, name: str = "md5") -> HashInfo:
    """entries: iterable of ((path, parts...), HashInfo|None)."""
    tree = Tree()
    for key, hash_info in entries:
        tree.add(key, Meta(), hash_info)
    tree.digest(name=name)
    odb.add_bytes(tree.oid, tree.as_bytes())
    return tree.hash_info


def missing_dir_hi(name: str = "md5") -> HashInfo:
    return HashInfo(name, "00" + "1" * 30 + ".dir")


def expected_closure(odb, roots):
    """Independent recursive reference walk used as an oracle."""
    seen: set[HashInfo] = set()
    stack = list(roots)
    while stack:
        hi = stack.pop()
        if not hi or not hi.value or hi in seen:
            continue
        seen.add(hi)
        if hi.isdir:
            tree = Tree.load(odb, hi)
            for _, _, child in tree:
                if child and child.value:
                    stack.append(child)
    return seen


def status(report, hi):
    return report.objects[hi]


# --------------------------------------------------------------------------- #
# basic happy paths
# --------------------------------------------------------------------------- #


def test_empty_roots():
    odb = make_odb()
    report = ExplainReachability([odb]).explain([])
    assert report.ok
    assert report.complete_closure
    assert report.incomplete is False
    assert report.objects == {}
    assert report.stats.objects_checked == 0


def test_leaf_and_tree_metadata_and_verify():
    odb = make_odb()
    foo = put_leaf(odb, b"foo")
    bar = put_leaf(odb, b"bar")
    root = put_tree(odb, [(("foo",), foo), (("d", "bar"), bar)])

    metadata = ExplainReachability([odb], mode=METADATA).explain([root])
    assert metadata.ok
    assert metadata.mode == METADATA
    assert {s.state for s in metadata.objects.values()} == {OK}
    # metadata mode never hashes leaf content nor reports conflicts
    assert metadata.conflicts == []
    assert all(s.actual is None for s in metadata.objects.values())
    assert status(metadata, root).witness.as_posix() == ""
    assert status(metadata, foo).witness.as_posix() == "foo"
    assert status(metadata, bar).witness.as_posix() == "d/bar"

    verify = ExplainReachability([odb], mode=VERIFY).explain([root])
    assert verify.mode == VERIFY
    assert verify.ok
    assert verify.stats.bytes_read > metadata.stats.bytes_read
    # every object, including the directory itself, was content-checked
    assert len(verify.checked) == 3


def test_witness_is_stable_across_runs():
    odb = make_odb()
    leaf = put_leaf(odb, b"x")
    root = put_tree(
        odb,
        [
            (("z",), leaf),
            (("a", "b", "c"), leaf),
            (
                (
                    "a",
                    "b",
                ),
                leaf,
            ),
        ],
    )
    first = ExplainReachability([odb]).explain([root]).to_dict()
    second = ExplainReachability([odb]).explain([root]).to_dict()
    assert first == second
    # lexicographically shortest entry name wins among equal-length paths
    assert (
        status(ExplainReachability([odb]).explain([root]), leaf).witness.as_posix()
        == "z"
    )


# --------------------------------------------------------------------------- #
# shared subtrees, multiple roots, dedup
# --------------------------------------------------------------------------- #


def test_shared_subtree_shortest_witness_and_reference_count():
    odb = make_odb()
    leaf = put_leaf(odb, b"shared-bytes")
    shared = put_tree(odb, [(("f",), leaf)])
    # R1 reaches `shared` through a 3-part key, R2 directly.
    r1 = put_tree(odb, [(("x", "y", "s"), shared)])
    r2 = put_tree(odb, [(("s",), shared)])

    report = ExplainReachability([odb]).explain([r1, r2])

    assert report.ok
    shared_status = status(report, shared)
    assert shared_status.references == 2
    assert shared_status.witness.root == r2
    assert shared_status.witness.as_posix() == "s"

    leaf_status = status(report, leaf)
    assert leaf_status.witness.root == r2
    assert leaf_status.witness.as_posix() == "s/f"

    # roots themselves are referenced once each
    assert status(report, r1).references == 1
    assert status(report, r2).references == 1


def test_same_object_checked_and_read_only_once():
    odb = make_odb()
    leaf = put_leaf(odb, b"dedup")
    shared = put_tree(odb, [(("f",), leaf)])
    r1 = put_tree(odb, [(("a",), shared), (("b",), shared)])
    r2 = put_tree(odb, [(("c",), shared)])

    report = ExplainReachability([odb], mode=VERIFY).explain([r1, r2])
    assert report.ok
    assert leaf.value
    assert shared.value

    # one existence probe and one content read per distinct object
    assert odb.exists_calls[leaf.value] == 1
    assert odb.exists_calls[shared.value] == 1
    assert report.stats.objects_seen == 4
    assert report.stats.objects_checked == 4
    assert status(report, leaf).references == 1  # one entry edge inside shared
    assert status(report, shared).references == 3  # 2 from r1 + 1 from r2

    # tree bytes for the shared directory are counted once
    with odb.get(shared.value).fs.open(odb.oid_to_path(shared.value), "rb") as fobj:
        shared_size = len(fobj.read())
    assert report.stats.bytes_read >= shared_size
    # rerunning adds no duplicate state
    again = ExplainReachability([odb], mode=VERIFY).explain([r1, r2])
    assert again.stats.objects_checked == 4


def test_md5_and_sha256_same_value_are_distinct_objects():
    odb = make_odb()
    same_hex = "ab" * 16
    odb.add_bytes(same_hex, b"colliding-oid-string")
    md5_hi = HashInfo("md5", same_hex)
    sha_hi = HashInfo("sha256", same_hex)
    root = put_tree(odb, [(("m",), md5_hi), (("s",), sha_hi)])

    report = ExplainReachability([odb]).explain([root])
    assert report.ok
    assert md5_hi in report.objects
    assert sha_hi in report.objects
    assert len(report.objects) == 3
    assert status(report, md5_hi).witness.as_posix() == "m"
    assert status(report, sha_hi).witness.as_posix() == "s"


def test_sha256_verify_end_to_end():
    odb = make_odb(hash_name="sha256")
    leaf = put_leaf(odb, b"sha content", name="sha256")
    root = put_tree(odb, [(("nested", "file"), leaf)], name="sha256")
    report = ExplainReachability([odb], mode=VERIFY).explain([root])
    assert report.ok
    assert all(s.hash_info.name == "sha256" for s in report.objects.values())


# --------------------------------------------------------------------------- #
# missing / malformed objects and unknown propagation
# --------------------------------------------------------------------------- #


def test_missing_directory_makes_descendants_unknown():
    odb = make_odb()
    good = put_leaf(odb, b"good")
    gone = missing_dir_hi()
    root = put_tree(odb, [(("good",), good), (("gone",), gone)])

    report = ExplainReachability([odb]).explain([root])

    assert not report.ok
    assert not report.complete_closure
    missing = report.missing
    assert len(missing) == 1
    assert missing[0].hash_info == gone
    assert missing[0].witness is not None
    assert missing[0].witness.as_posix() == "gone"
    assert gone in report.unknown_subtrees
    # the good leaf is still fully checked
    assert status(report, good).state == OK
    # consumers must not treat seen leaves as a complete closure
    assert report.incomplete is False  # walk itself was not budget-truncated
    assert report.complete_closure is False


def test_missing_leaf_is_reported_with_witness():
    odb = make_odb()
    missing_leaf = HashInfo("md5", "cd" + "2" * 30)
    root = put_tree(odb, [(("deep", "missing"), missing_leaf)])

    report = ExplainReachability([odb]).explain([root])
    assert len(report.missing) == 1
    result = report.missing[0]
    assert result.state == MISSING
    assert result.sources == ()
    assert result.witness is not None
    assert result.witness.as_posix() == "deep/missing"
    assert not report.ok


@pytest.mark.parametrize(
    "raw",
    [b"this is not json", b'{"not": "a list"}', b"[1, 2, 3]"],
)
def test_malformed_directory(raw):
    odb = make_odb()
    hi = HashInfo("md5", "ef" + "3" * 30 + ".dir")
    assert hi.value
    odb.add_bytes(hi.value, raw)
    report = ExplainReachability([odb]).explain([hi])

    assert len(report.malformed) == 1
    assert report.malformed[0].hash_info == hi
    assert hi in report.unknown_subtrees
    assert not report.complete_closure
    assert not report.ok


def test_malformed_tree_entry_has_witness():
    odb = make_odb()
    dir_oid = "12" + "4" * 30 + ".dir"
    payload = json.dumps([{"relpath": "empty-no-hash"}]).encode()
    odb.add_bytes(dir_oid, payload)
    hi = HashInfo("md5", dir_oid)

    report = ExplainReachability([odb]).explain([hi])
    assert len(report.malformed_entries) == 1
    entry = report.malformed_entries[0]
    assert entry.witness.as_posix() == "empty-no-hash"
    assert not report.ok


def test_invalid_root_empty_hash_info():
    odb = make_odb()
    report = ExplainReachability([odb]).explain([HashInfo()])
    assert len(report.invalid_roots) == 1
    assert not report.ok
    assert report.objects == {}


def test_root_level_failures_have_empty_path_witness():
    odb = make_odb()
    missing_root = HashInfo("md5", "aa" + "b" * 30)
    missing_report = ExplainReachability([odb]).explain([missing_root])
    result = missing_report.missing[0]
    assert result.witness is not None
    assert result.witness.root == missing_root
    assert result.witness.path == ()

    bad_dir = HashInfo("md5", "cc" + "d" * 30 + ".dir")
    assert bad_dir.value
    odb.add_bytes(bad_dir.value, b"garbage")
    malformed_report = ExplainReachability([odb]).explain([bad_dir])
    dir_result = malformed_report.malformed[0]
    assert dir_result.witness is not None
    assert dir_result.witness.path == ()
    assert bad_dir in malformed_report.unknown_subtrees


# --------------------------------------------------------------------------- #
# verify mode: real content checks, read-only behaviour
# --------------------------------------------------------------------------- #


def test_bad_leaf_modes_are_separated():
    odb = make_odb()
    good_oid = hashlib.md5(b"foo").hexdigest()  # noqa: S324
    odb.add_bytes(good_oid, b"bar")  # content does not match the OID
    hi = HashInfo("md5", good_oid)

    metadata = ExplainReachability([odb], mode=METADATA).explain([hi])
    assert metadata.ok  # existence only: object is there
    assert metadata.conflicts == []

    verify = ExplainReachability([odb], mode=VERIFY).explain([hi])
    assert not verify.ok
    assert len(verify.mismatched) == 1
    result = verify.mismatched[0]
    assert result.actual == hashlib.md5(b"bar").hexdigest()  # noqa: S324
    assert result.detail
    # corrupt object remains in place; nothing was deleted
    assert odb.exists(good_oid)


def test_verify_never_mutates_objects_or_state(tmp_path):
    fs = LocalFileSystem()
    odb = GuardedLocalDB(fs, str(tmp_path / "cache"), state=RaisingState())
    good_leaf = put_leaf(odb, b"clean")
    bad_oid = hashlib.md5(b"expected").hexdigest()  # noqa: S324
    odb.add_bytes(bad_oid, b"actually different")
    bad_hi = HashInfo("md5", bad_oid)
    root = put_tree(odb, [(("good",), good_leaf), (("bad",), bad_hi)])

    bad_path = odb.oid_to_path(bad_oid)
    with fs.open(bad_path, "rb") as fobj:
        before = fobj.read()

    report = ExplainReachability([odb], mode=VERIFY).explain([root])

    with fs.open(bad_path, "rb") as fobj:
        after = fobj.read()
    assert after == before  # not removed/repaired
    assert len(report.mismatched) == 1
    assert status(report, good_leaf).state == OK
    # state cache was never written to
    assert isinstance(odb.state, RaisingState)


def test_verify_directory_content_mismatch():
    odb = make_odb()
    leaf = put_leaf(odb, b"leaf")
    real_root = put_tree(odb, [(("f",), leaf)])
    assert real_root.value
    # store the still-parseable directory bytes under a *different* OID:
    # structure loads fine, but content hashing cannot match the claimed OID
    with odb.fs.open(odb.oid_to_path(real_root.value), "rb") as fobj:
        raw = fobj.read()
    wrong_oid = "77" + "6" * 30 + ".dir"
    odb.add_bytes(wrong_oid, raw)
    wrong_root = HashInfo("md5", wrong_oid)

    verify = ExplainReachability([odb], mode=VERIFY).explain([wrong_root])
    assert len(verify.mismatched) == 1
    assert verify.mismatched[0].hash_info == wrong_root
    assert verify.mismatched[0].actual == real_root.value[: -len(".dir")]

    # metadata mode only checks existence/structure: it stays ok
    metadata = ExplainReachability([odb], mode=METADATA).explain([wrong_root])
    assert metadata.ok
    assert metadata.conflicts == []


# --------------------------------------------------------------------------- #
# cycles
# --------------------------------------------------------------------------- #


def test_self_referencing_tree_is_an_explicit_cycle_error():
    odb = make_odb()
    leaf = put_leaf(odb, b"leaf")
    tree = Tree()
    tree.add(("leaf",), Meta(), leaf)
    tree.digest()
    oid = tree.oid
    assert oid
    # malicious self reference, stored under the same directory OID
    tree.add(("loop",), Meta(), HashInfo("md5", oid))
    odb.add_bytes(oid, tree.as_bytes())
    root = HashInfo("md5", oid)

    report = ExplainReachability([odb]).explain([root])  # must terminate
    assert len(report.cycles) == 1
    cycle = report.cycles[0]
    assert cycle.target == root
    assert cycle.source == root
    assert cycle.witness is not None
    assert cycle.witness.as_posix() == "loop"
    assert not report.ok
    # all distinct reachable objects were still checked exactly once
    assert status(report, leaf).state == OK
    assert report.stats.objects_checked == 2


def test_two_node_directory_cycle():
    odb = make_odb()
    tree_a = Tree()
    tree_a.add(("leaf",), Meta(), put_leaf(odb, b"a-leaf"))
    tree_a.digest()
    tree_b = Tree()
    tree_b.add(("leaf",), Meta(), put_leaf(odb, b"b-leaf"))
    tree_b.digest()
    a_oid, b_oid = tree_a.oid, tree_b.oid
    assert a_oid
    assert b_oid
    tree_a.add(("to_b",), Meta(), HashInfo("md5", b_oid))
    tree_b.add(("to_a",), Meta(), HashInfo("md5", a_oid))
    odb.add_bytes(a_oid, tree_a.as_bytes())
    odb.add_bytes(b_oid, tree_b.as_bytes())

    report = ExplainReachability([odb]).explain([HashInfo("md5", a_oid)])
    assert len(report.cycles) == 1
    assert report.cycles[0].target == HashInfo("md5", a_oid)
    assert report.stats.objects_checked == 4  # 2 dirs + 2 leaves
    assert not report.ok


# --------------------------------------------------------------------------- #
# budgets
# --------------------------------------------------------------------------- #


def test_budget_objects_stops_early_with_accurate_checked_set():
    odb = make_odb()
    leaves = [put_leaf(odb, f"l{i}".encode()) for i in range(2)]
    root = put_tree(odb, [(("l0",), leaves[0]), (("l1",), leaves[1])])

    stopped = ExplainReachability([odb], budget=Budget(max_objects=1)).explain([root])
    assert stopped.incomplete
    assert stopped.incomplete_reasons == ["budget:max_objects"]
    assert stopped.stats.objects_checked == 1
    assert len(stopped.unchecked) == 2
    assert {s.hash_info for s in stopped.checked} == {root}
    assert not stopped.ok

    full = ExplainReachability([odb]).explain([root])
    assert full.complete_closure
    assert full.stats.objects_checked == 3


def test_budget_edges():
    odb = make_odb()
    leaves = [put_leaf(odb, b"a"), put_leaf(odb, b"b")]
    root = put_tree(odb, [(("a",), leaves[0]), (("b",), leaves[1])])

    report = ExplainReachability([odb], budget=Budget(max_edges=1)).explain([root])
    assert report.incomplete_reasons == ["budget:max_edges"]
    assert report.stats.edges_traversed == 1
    # root could not be fully enumerated: nothing claims to be fully checked
    assert all(s.state == UNCHECKED for s in report.objects.values())
    assert not report.ok


def test_budget_bytes_in_verify_mode():
    odb = make_odb()
    leaves = [put_leaf(odb, b"alpha"), put_leaf(odb, b"beta")]
    root = put_tree(odb, [(("a",), leaves[0]), (("b",), leaves[1])])

    stopped = ExplainReachability(
        [odb], mode=VERIFY, budget=Budget(max_bytes=1)
    ).explain([root])
    assert stopped.incomplete_reasons == ["budget:max_bytes"]
    assert stopped.stats.bytes_read >= 1
    assert len(stopped.unchecked) == 2

    full = ExplainReachability([odb], mode=VERIFY).explain([root])
    assert full.complete_closure
    assert full.stats.objects_checked == 3
    # checked-set accounting is always internally consistent
    assert len(full.checked) == full.stats.objects_checked


def test_budget_witnesses():
    odb = make_odb()
    missing = [HashInfo("md5", f"{i:02x}" + "5" * 30) for i in range(2)]
    root = put_tree(odb, [(("m0",), missing[0]), (("m1",), missing[1])])

    report = ExplainReachability([odb], budget=Budget(max_witnesses=1)).explain([root])
    assert report.incomplete_reasons == ["budget:max_witnesses"]
    assert len(report.missing) == 1
    assert report.stats.witnesses == 1
    assert report.stats.witnesses_truncated
    assert any(s.state == UNCHECKED for s in report.objects.values())


def test_sufficient_budget_checks_every_real_reference():
    odb = make_odb()
    shared_leaf = put_leaf(odb, b"shared")
    shared_tree = put_tree(odb, [(("f",), shared_leaf)])
    nested = put_tree(odb, [(("s",), shared_tree), (("own",), put_leaf(odb, b"n"))])
    r1 = put_tree(odb, [(("deep", "nested"), nested), (("s",), shared_tree)])
    r2 = put_tree(odb, [(("shortcut",), shared_tree)])

    report = ExplainReachability([odb], mode=VERIFY).explain([r1, r2])
    expected = expected_closure(odb, [r1, r2])

    assert report.complete_closure
    assert {s.hash_info for s in report.checked} == expected
    assert all(s.state == OK for s in report.objects.values())
    assert report.stats.objects_seen == len(expected)


# --------------------------------------------------------------------------- #
# ODB fallback ordering and same-OID content conflicts
# --------------------------------------------------------------------------- #


def test_odb_fallback_source_order():
    odb0 = make_odb(path="/store-zero")
    odb1 = make_odb(path="/store-one")
    leaf = put_leaf(odb1, b"only in second odb")
    root = put_tree(odb1, [(("f",), leaf)])

    report = ExplainReachability([odb0, odb1], labels=["primary", "secondary"]).explain(
        [root]
    )
    assert report.ok
    assert status(report, leaf).sources == ("secondary",)
    assert status(report, root).sources == ("secondary",)


def test_same_oid_different_content_conflict_is_observable():
    oid = hashlib.md5(b"good").hexdigest()  # noqa: S324
    odb0 = make_odb(path="/store-zero")
    odb1 = make_odb(path="/store-one")
    odb0.add_bytes(oid, b"bad")
    odb1.add_bytes(oid, b"good")
    hi = HashInfo("md5", oid)

    verify = ExplainReachability([odb0, odb1], mode=VERIFY).explain([hi])
    # the object itself is usable from the fallback ODB...
    assert status(verify, hi).state == OK
    # ...but the storage conflict makes the overall report non-passing
    assert not verify.ok
    assert len(verify.conflicts) == 1
    conflict = verify.conflicts[0]
    assert conflict.kind == "content"
    assert conflict.source == "odb[0]:memory"
    assert conflict.actual == hashlib.md5(b"bad").hexdigest()  # noqa: S324
    assert any("odb[1]" in source for source in status(verify, hi).sources)

    # metadata mode cannot observe content conflicts
    metadata = ExplainReachability([odb0, odb1], mode=METADATA).explain([hi])
    assert metadata.conflicts == []
    assert metadata.ok

    # when every copy is bad the object itself is mismatched
    odb1.add_bytes(oid, b"also bad")
    failed = ExplainReachability([odb0, odb1], mode=VERIFY).explain([hi])
    assert len(failed.mismatched) == 1
    assert len(failed.conflicts) == 2


def test_report_hides_paths_and_credentials_by_default():
    private_path = "/private/credentialed-store"
    odb = make_odb(path=private_path)
    root = put_tree(odb, [(("f",), put_leaf(odb, b"x"))])

    default_report = ExplainReachability([odb]).explain([root]).to_dict()
    rendered = json.dumps(default_report)
    assert private_path not in rendered
    assert "odb[0]:memory" in rendered
    assert default_report["odb_sources"] == ["odb[0]:memory"]

    opt_in = ExplainReachability([odb], include_paths=True).explain([root]).to_dict()
    assert private_path in json.dumps(opt_in)


def test_convenience_function_equivalent():
    odb = make_odb()
    root = put_tree(odb, [(("f",), put_leaf(odb, b"x"))])
    report = explain_reachability([odb], [root])
    assert report.ok
    assert report.complete_closure
