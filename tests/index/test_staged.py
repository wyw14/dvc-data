import errno
import os
import stat
from contextlib import closing

import pytest
from dvc_objects.fs.local import LocalFileSystem

from dvc_data.hashfile.hash_info import HashInfo
from dvc_data.hashfile.meta import Meta
from dvc_data.hashfile.state import State
from dvc_data.hashfile.tree import Tree
from dvc_data.index import DataIndex, DataIndexEntry, ObjectStorage, staged
from dvc_data.index.checkout import apply, compare
from dvc_data.index.staged import (
    EV_COMMITTED,
    EV_PLAN_CREATED,
    EV_PREPARED,
    EV_PUBLISH_START,
    EV_PUBLISHED,
    EV_ROLLED_BACK,
    CheckoutPlan,
    Journal,
    StagedCheckoutConflictError,
    StagedCheckoutError,
    StagedCheckoutExternalModificationError,
    StagedCheckoutLockError,
    StagedCheckoutObjectMissingError,
    StagedCheckoutSafetyError,
    StagedCheckoutSpaceError,
    WorkspaceLock,
)

localfs = LocalFileSystem()

FOO = "d3b07384d113edec49eaa6238ad5ff00"
BAR = "c157a79031e1c40f85931829bc5fc552"
BAZ = "258622b1688250cb619f3c9ccaefb7eb"
DIR_HASH = "1f69c66028c35037e8bf67e5bc4ceb6a.dir"
MISSING = "0" * 32


def _entry(key, hash_value=None, *, isdir=False, isexec=False):
    return DataIndexEntry(
        key=tuple(key),
        meta=Meta(isdir=isdir, isexec=isexec),
        hash_info=HashInfo("md5", hash_value) if hash_value else None,
    )


def _index(entries, odb):
    index = DataIndex({})
    for entry in entries:
        index.add(entry)
    index.storage_map.add_cache(ObjectStorage((), odb))
    return index


def _add_tree(odb, items):
    """Create a .dir object in the odb; items is (relpath, md5) pairs."""
    tree = Tree.from_list(
        [{"md5": hash_value, "relpath": name} for name, hash_value in items]
    )
    tree.digest()
    odb.add_bytes(tree.hash_info.value, tree.as_bytes())
    return tree.hash_info.value


def _staging_root(ws):
    return os.path.join(str(ws), staged.STAGING_DIRNAME)


def _journal(ws, plan):
    root = _staging_root(ws)
    return Journal(plan.journal_path(root))


def _run_phased(old, new, ws, *, delete=False, state=None, plan_id=None, **pub):
    """Run the three phases explicitly, mimicking a restartable workflow."""
    root = _staging_root(ws)
    with WorkspaceLock(root, plan_id=plan_id):
        plan = staged.build_plan(
            old,
            new,
            str(ws),
            localfs,
            delete=delete,
            staging_root=root,
            plan_id=plan_id,
        )
        plan.save(root)
        journal = Journal(plan.journal_path(root))
        journal.append(EV_PLAN_CREATED, plan_id=plan.plan_id)
        staged.prepare(plan, localfs, root, journal=journal)
        staged.publish(plan, localfs, root, state=state, journal=journal, **pub)
    return plan, root


# ---------------------------------------------------------------------------
# basic behaviour
# ---------------------------------------------------------------------------


def test_staged_checkout_basic(tmp_path, odb):
    index = _index(
        [
            _entry(("foo",), FOO),
            _entry(("data",), DIR_HASH, isdir=True),
        ],
        odb,
    )
    plan = staged.staged_checkout(None, index, str(tmp_path), localfs)

    assert (tmp_path / "foo").read_text() == "foo\n"
    assert (tmp_path / "data").is_dir()
    assert (tmp_path / "data" / "bar").read_text() == "bar\n"
    assert (tmp_path / "data" / "baz").read_text() == "baz\n"

    root = _staging_root(tmp_path)
    loaded = CheckoutPlan.load(root, plan.plan_id)
    assert [e.key for e in loaded.entries] == [e.key for e in plan.entries]
    assert {e.op for e in loaded.entries} >= {"mkdir", "create_file"}

    events = [e["event"] for e in _journal(tmp_path, plan).events()]
    assert events[0] == EV_PLAN_CREATED
    assert EV_PUBLISH_START in events
    assert events[-1] == EV_COMMITTED
    assert events.index(EV_PUBLISH_START) < events.index(EV_COMMITTED)
    # every prepared object was consumed by an atomic replace
    leftover: list[str] = []
    for root_dir, _dirs, files in os.walk(plan.objects_dir(root)):
        leftover.extend(os.path.join(root_dir, name) for name in files)
    assert not leftover


def test_staged_checkout_single_file_target(tmp_path, odb):
    index = _index([_entry((), FOO)], odb)
    target = tmp_path / "foo"
    plan = staged.staged_checkout(None, index, str(target), localfs)

    assert target.read_text() == "foo\n"
    # staging lives next to the file target, on the same volume
    assert (tmp_path / staged.STAGING_DIRNAME / plan.plan_id / "plan.json").exists()


def test_staged_checkout_persists_plan_and_journal(tmp_path, odb):
    index = _index([_entry(("foo",), FOO)], odb)
    plan = staged.staged_checkout(None, index, str(tmp_path), localfs)

    root = _staging_root(tmp_path)
    reloaded = CheckoutPlan.load(root, plan.plan_id)
    assert reloaded.workspace == os.path.abspath(str(tmp_path))
    create = next(reloaded.iter_ops("create_file"))
    assert create.target_hash == {"md5": FOO}
    assert create.src_path.endswith(os.path.join(FOO[:2], FOO[2:]))
    assert create.staged_rel == "foo"
    assert create.pre.exists is False

    journal = _journal(tmp_path, plan)
    prepared = journal.keys_for(EV_PREPARED)
    published = journal.keys_for(EV_PUBLISHED)
    assert prepared == published == {("foo",)}


# ---------------------------------------------------------------------------
# conflicts: reported before any overwrite, nothing is staged
# ---------------------------------------------------------------------------


def test_untracked_file_conflict_is_reported(tmp_path, odb):
    (tmp_path / "extra").write_text("untracked\n")
    index = _index([_entry(("extra",), FOO)], odb)

    with pytest.raises(StagedCheckoutConflictError) as exc_info:
        staged.staged_checkout(None, index, str(tmp_path), localfs)

    conflict = exc_info.value.conflicts[0]
    assert conflict.kind == "untracked"
    assert conflict.key == ("extra",)
    # workspace is untouched and no plan directory was created
    assert (tmp_path / "extra").read_text() == "untracked\n"
    root = tmp_path / staged.STAGING_DIRNAME
    if root.exists():
        assert not any(p.is_dir() for p in root.iterdir())


def test_content_mismatch_conflict(tmp_path, odb):
    foo = tmp_path / "foo"
    foo.write_text("changed content\n")
    old = DataIndex({("foo",): _entry(("foo",), FOO)})
    new = _index([_entry(("foo",), BAR)], odb)

    with pytest.raises(StagedCheckoutConflictError) as exc_info:
        staged.build_plan(old, new, str(tmp_path), localfs)

    assert exc_info.value.conflicts[0].kind == "content_mismatch"
    assert foo.read_text() == "changed content\n"


def test_type_mismatch_conflict(tmp_path, odb):
    (tmp_path / "foo").mkdir()
    old = DataIndex({("foo",): _entry(("foo",), FOO)})
    new = _index([_entry(("foo",), BAR)], odb)

    with pytest.raises(StagedCheckoutConflictError) as exc_info:
        staged.build_plan(old, new, str(tmp_path), localfs)

    kinds = {c.kind for c in exc_info.value.conflicts}
    assert "type_mismatch" in kinds


def test_untracked_child_blocks_dir_to_file(tmp_path, odb):
    data = tmp_path / "data"
    data.mkdir()
    (data / "bar").write_bytes(b"bar\n")
    (data / "stray").write_text("stray\n")

    old = DataIndex(
        {
            ("data",): _entry(("data",), None, isdir=True),
            ("data", "bar"): _entry(("data", "bar"), BAR),
        }
    )
    new = _index([_entry(("data",), FOO)], odb)

    with pytest.raises(StagedCheckoutConflictError) as exc_info:
        staged.build_plan(old, new, str(tmp_path), localfs, delete=True)

    paths = {c.path for c in exc_info.value.conflicts}
    assert any(str(p).endswith("stray") for p in paths)
    assert (data / "stray").read_text() == "stray\n"
    assert (data / "bar").read_bytes() == b"bar\n"


# ---------------------------------------------------------------------------
# type switches
# ---------------------------------------------------------------------------


def test_file_to_dir_switch_is_ordered_and_recoverable(tmp_path, odb):
    (tmp_path / "a").write_bytes(b"foo\n")
    old = DataIndex({("a",): _entry(("a",), FOO)})
    nested_dir = _add_tree(odb, [("nested", BAR)])
    new = _index(
        [
            _entry(("a",), nested_dir, isdir=True),
            _entry(("a", "nested"), BAR),
        ],
        odb,
    )

    plan, root = _run_phased(old, new, tmp_path)
    ops = [(e.op, e.key) for e in plan.entries]
    assert ops.index(("delete_file", ("a",))) < ops.index(("mkdir", ("a",)))
    assert ops.index(("mkdir", ("a",))) < ops.index(("create_file", ("a", "nested")))

    assert (tmp_path / "a").is_dir()
    assert (tmp_path / "a" / "nested").read_text() == "bar\n"
    # the owned old content was saved before removal
    rollback = os.path.join(plan.rollback_dir(root), "a")
    with open(rollback, encoding="utf8") as fobj:
        assert fobj.read() == "foo\n"

    staged.rollback(str(tmp_path), plan.plan_id, localfs, staging_root=root)
    assert (tmp_path / "a").is_file()
    assert (tmp_path / "a").read_text() == "foo\n"
    assert _journal(tmp_path, plan).has(EV_ROLLED_BACK)


def test_dir_to_file_switch_restores_tree(tmp_path, odb):
    data = tmp_path / "data"
    data.mkdir()
    (data / "bar").write_bytes(b"bar\n")
    old = DataIndex(
        {
            ("data",): _entry(("data",), None, isdir=True),
            ("data", "bar"): _entry(("data", "bar"), BAR),
        }
    )
    new = _index([_entry(("data",), FOO)], odb)

    plan, root = _run_phased(old, new, tmp_path, delete=True)
    assert (tmp_path / "data").is_file()
    assert (tmp_path / "data").read_bytes() == b"foo\n"

    staged.rollback(str(tmp_path), plan.plan_id, localfs, staging_root=root)
    assert (tmp_path / "data").is_dir()
    assert (tmp_path / "data" / "bar").read_bytes() == b"bar\n"


# ---------------------------------------------------------------------------
# prepare failures: missing objects and space
# ---------------------------------------------------------------------------


def test_missing_object_aborts_before_touching_workspace(tmp_path, odb):
    (tmp_path / "foo").write_bytes(b"foo\n")
    old = DataIndex({("foo",): _entry(("foo",), FOO)})
    new = _index([_entry(("foo",), MISSING)], odb)
    root = _staging_root(tmp_path)

    with WorkspaceLock(root):
        plan = staged.build_plan(old, new, str(tmp_path), localfs, staging_root=root)
        plan.save(root)
        with pytest.raises(StagedCheckoutObjectMissingError):
            staged.prepare(plan, localfs, root)

    assert (tmp_path / "foo").read_text() == "foo\n"
    assert not any(os.scandir(plan.objects_dir(root)))


def test_space_precheck_aborts_prepare(tmp_path, odb, monkeypatch):
    index = _index([_entry(("foo",), FOO)], odb)
    root = _staging_root(tmp_path)
    with WorkspaceLock(root):
        plan = staged.build_plan(None, index, str(tmp_path), localfs, staging_root=root)
        plan.save(root)
        monkeypatch.setattr(staged, "_free_space", lambda path: 0)
        with pytest.raises(StagedCheckoutSpaceError):
            staged.prepare(plan, localfs, root)

    assert not (tmp_path / "foo").exists()


def test_enospc_during_copy_is_reported_and_cleaned(tmp_path, odb, monkeypatch):
    index = _index([_entry(("foo",), FOO)], odb)
    root = _staging_root(tmp_path)

    def _raise(src_fs, src_path, dest_fs, dest_path):
        raise OSError(errno.ENOSPC, "no space left on device")

    with WorkspaceLock(root):
        plan = staged.build_plan(None, index, str(tmp_path), localfs, staging_root=root)
        plan.save(root)
        journal = Journal(plan.journal_path(root))
        monkeypatch.setattr(staged, "_copy_file", _raise)
        with pytest.raises(StagedCheckoutSpaceError):
            staged.prepare(plan, localfs, root, journal=journal)
        assert journal.has("prepare_failed")

    assert not (tmp_path / "foo").exists()
    assert not list(os.scandir(plan.objects_dir(root)))


# ---------------------------------------------------------------------------
# interruption, restart recovery and rollback
# ---------------------------------------------------------------------------


def test_publish_interruption_resume_and_restart_recovery(tmp_path, odb):
    index = _index(
        [
            _entry(("foo",), FOO),
            _entry(("bar",), BAR),
            _entry(("baz",), BAZ),
        ],
        odb,
    )
    root = _staging_root(tmp_path)
    published_count = 0

    def fail_after_two(entry):
        nonlocal published_count
        published_count += 1
        if published_count == 2:
            raise RuntimeError("simulated crash during publish")

    with WorkspaceLock(root):
        plan = staged.build_plan(None, index, str(tmp_path), localfs, staging_root=root)
        plan.save(root)
        journal = Journal(plan.journal_path(root))
        journal.append(EV_PLAN_CREATED, plan_id=plan.plan_id)
        staged.prepare(plan, localfs, root, journal=journal)
        with pytest.raises(RuntimeError, match="simulated crash"):
            staged.publish(
                plan,
                localfs,
                root,
                journal=journal,
                on_op_published=fail_after_two,
            )

    # lock was released and two files are partially published
    assert not os.path.exists(os.path.join(root, "lock"))
    assert journal.keys_for(EV_PUBLISHED)
    assert not journal.has(EV_COMMITTED)
    assert any(e["event"] == "publish_failed" for e in journal.events())

    # "restart": recover from the on-disk plan and journal only
    resumed = staged.resume(str(tmp_path), plan.plan_id, localfs, staging_root=root)
    assert resumed.plan_id == plan.plan_id
    assert (tmp_path / "foo").read_text() == "foo\n"
    assert (tmp_path / "bar").read_text() == "bar\n"
    assert (tmp_path / "baz").read_text() == "baz\n"
    assert _journal(tmp_path, plan).has(EV_COMMITTED)

    # resuming a committed plan is an idempotent no-op
    again = staged.resume(str(tmp_path), plan.plan_id, localfs, staging_root=root)
    assert again.plan_id == plan.plan_id


def test_resume_stops_on_external_modification(tmp_path, odb):
    (tmp_path / "foo").write_bytes(b"foo\n")
    old = DataIndex({("foo",): _entry(("foo",), FOO)})
    new = _index(
        [_entry(("foo",), BAR), _entry(("bar",), BAZ)],
        odb,
    )
    root = _staging_root(tmp_path)

    def fail_after_one(entry):
        raise RuntimeError("simulated crash")

    with pytest.raises(RuntimeError):
        _run_phased(old, new, tmp_path, on_op_published=fail_after_one)

    # an external process edits a not-yet-published target
    (tmp_path / "foo").write_text("external edit\n")

    with pytest.raises(StagedCheckoutExternalModificationError):
        staged.resume(str(tmp_path), _only_plan_id(root), localfs, staging_root=root)
    # the externally modified file is preserved
    assert (tmp_path / "foo").read_text() == "external edit\n"


def test_publish_stops_when_pre_state_changed_externally(tmp_path, odb):
    (tmp_path / "foo").write_bytes(b"foo\n")
    old = DataIndex({("foo",): _entry(("foo",), FOO)})
    new = _index([_entry(("foo",), BAR)], odb)
    root = _staging_root(tmp_path)

    with WorkspaceLock(root):
        plan = staged.build_plan(old, new, str(tmp_path), localfs, staging_root=root)
        plan.save(root)
        journal = Journal(plan.journal_path(root))
        staged.prepare(plan, localfs, root, journal=journal)
        (tmp_path / "foo").write_text("external edit\n")
        with pytest.raises(StagedCheckoutExternalModificationError):
            staged.publish(plan, localfs, root, journal=journal)

    assert (tmp_path / "foo").read_text() == "external edit\n"
    assert not journal.has(EV_COMMITTED)


def test_rollback_after_interruption_restores_pre_state(tmp_path, odb):
    (tmp_path / "foo").write_bytes(b"foo\n")
    old = DataIndex({("foo",): _entry(("foo",), FOO)})
    new = _index(
        [_entry(("foo",), BAR), _entry(("bar",), BAZ)],
        odb,
    )
    root = _staging_root(tmp_path)

    def fail_after_one(entry):
        raise RuntimeError("simulated crash")

    with pytest.raises(RuntimeError):
        _run_phased(old, new, tmp_path, on_op_published=fail_after_one)

    plan_id = _only_plan_id(root)
    staged.rollback(str(tmp_path), plan_id, localfs, staging_root=root)
    assert (tmp_path / "foo").read_text() == "foo\n"
    # a file created by the interrupted publish is removed again
    assert not (tmp_path / "bar").exists()


def test_rollback_restores_state_cache(tmp_path, odb):
    (tmp_path / "foo").write_bytes(b"foo\n")
    old = DataIndex({("foo",): _entry(("foo",), FOO)})
    new = _index([_entry(("foo",), BAR)], odb)

    with closing(State(str(tmp_path), str(tmp_path / "state_tmp"))) as state:
        plan, root = _run_phased(old, new, tmp_path, state=state)
        _meta, hash_info = state.get(str(tmp_path / "foo"), localfs)
        assert hash_info == HashInfo("md5", BAR)

        staged.rollback(
            str(tmp_path), plan.plan_id, localfs, state=state, staging_root=root
        )
        _meta, hash_info = state.get(str(tmp_path / "foo"), localfs)
        assert hash_info == HashInfo("md5", FOO)
    assert (tmp_path / "foo").read_text() == "foo\n"


# ---------------------------------------------------------------------------
# state cache and exec bit timing
# ---------------------------------------------------------------------------


def test_state_updated_only_after_publish_confirmation(tmp_path, odb):
    index = _index([_entry(("foo",), FOO)], odb)
    root = _staging_root(tmp_path)

    with closing(State(str(tmp_path), str(tmp_path / "state_tmp"))) as state:
        with WorkspaceLock(root):
            plan = staged.build_plan(
                None, index, str(tmp_path), localfs, staging_root=root
            )
            plan.save(root)
            journal = Journal(plan.journal_path(root))
            staged.prepare(plan, localfs, root, journal=journal)
            # nothing is recorded in the state cache before publish
            assert state.get(str(tmp_path / "foo"), localfs) == (None, None)

            staged.publish(plan, localfs, root, state=state, journal=journal)
            _meta, hash_info = state.get(str(tmp_path / "foo"), localfs)
            assert hash_info == HashInfo("md5", FOO)


def test_exec_bit_set_after_journal_confirmation(tmp_path, odb, monkeypatch):
    index = _index([_entry(("foo",), FOO, isexec=True)], odb)
    root = _staging_root(tmp_path)
    real_chmod = os.chmod
    exec_calls = []

    def spy_chmod(path, mode, *args, **kwargs):
        if mode & stat.S_IEXEC:
            exec_calls.append(path)
        return real_chmod(path, mode, *args, **kwargs)

    monkeypatch.setattr(staged.os, "chmod", spy_chmod)

    with WorkspaceLock(root):
        plan = staged.build_plan(None, index, str(tmp_path), localfs, staging_root=root)
        plan.save(root)
        journal = Journal(plan.journal_path(root))
        staged.prepare(plan, localfs, root, journal=journal)
        staged.publish(plan, localfs, root, journal=journal)

    target = str(tmp_path / "foo")
    assert any(os.path.normpath(p) == os.path.normpath(target) for p in exec_calls)
    assert journal.keys_for(EV_PUBLISHED) == {("foo",)}


# ---------------------------------------------------------------------------
# hardlink isolation, locks, symlinks and cleanup
# ---------------------------------------------------------------------------


def test_staged_copy_never_shares_cache_inode(tmp_path, odb):
    cache_path = odb.oid_to_path(FOO)
    real_chmod = os.chmod
    real_chmod(cache_path, stat.S_IREAD)
    try:
        index = _index([_entry(("foo",), FOO)], odb)
        root = _staging_root(tmp_path)
        with WorkspaceLock(root):
            plan = staged.build_plan(
                None, index, str(tmp_path), localfs, staging_root=root
            )
            plan.save(root)
            journal = Journal(plan.journal_path(root))
            staged.prepare(plan, localfs, root, journal=journal)

            staged_path = os.path.join(plan.objects_dir(root), "foo")
            assert os.stat(staged_path).st_ino != os.stat(cache_path).st_ino
            assert os.stat(staged_path).st_mode & stat.S_IWRITE
            # rewriting the staged file must not touch the protected cache
            with open(staged_path, "wb") as fobj:
                fobj.write(b"different\n")
            assert staged.file_md5(cache_path, name="md5") == FOO
    finally:
        real_chmod(cache_path, stat.S_IWRITE | stat.S_IREAD)


def test_workspace_lock_is_exclusive_and_released(tmp_path):
    root = str(tmp_path / staged.STAGING_DIRNAME)
    lock = WorkspaceLock(root, plan_id="plan-a")
    lock.acquire()
    try:
        with pytest.raises(StagedCheckoutLockError):
            WorkspaceLock(root, plan_id="plan-b").acquire()
    finally:
        lock.release()

    assert not os.path.exists(os.path.join(root, "lock"))
    with WorkspaceLock(root, plan_id="plan-c"):
        assert os.path.exists(os.path.join(root, "lock"))
    assert not os.path.exists(os.path.join(root, "lock"))


def test_symlink_target_is_rejected(tmp_path, odb):
    target = tmp_path / "real"
    target.write_text("foo\n")
    try:
        os.symlink(str(target), str(tmp_path / "link"))
    except OSError:
        pytest.skip("symlink creation is not permitted on this platform")

    index = _index([_entry(("link",), FOO)], odb)
    with pytest.raises(StagedCheckoutSafetyError):
        staged.staged_checkout(None, index, str(tmp_path), localfs)
    assert (tmp_path / "link").is_symlink()


def test_cleanup_only_removes_named_plan(tmp_path, odb):
    index_a = _index([_entry(("foo",), FOO)], odb)
    plan_a = staged.staged_checkout(
        None, index_a, str(tmp_path), localfs, plan_id="plan-a"
    )
    index_b = _index([_entry(("bar",), BAR)], odb)
    plan_b = staged.staged_checkout(
        None, index_b, str(tmp_path), localfs, plan_id="plan-b"
    )
    root = _staging_root(tmp_path)
    assert os.path.isdir(os.path.join(root, plan_a.plan_id))
    assert os.path.isdir(os.path.join(root, plan_b.plan_id))

    staged.cleanup(str(tmp_path), plan_a.plan_id, staging_root=root)
    assert not os.path.exists(os.path.join(root, plan_a.plan_id))
    # staging belonging to the other plan is preserved
    assert os.path.isdir(os.path.join(root, plan_b.plan_id))


def test_unsafe_key_component_is_rejected(tmp_path, odb):
    entry = DataIndexEntry(
        key=("..",),
        meta=Meta(),
        hash_info=HashInfo("md5", FOO),
    )
    index = _index([entry], odb)
    with pytest.raises((StagedCheckoutSafetyError, StagedCheckoutError)):
        staged.staged_checkout(None, index, str(tmp_path), localfs)


def test_relative_root_is_rejected(tmp_path, odb):
    index = _index([_entry(("foo",), FOO)], odb)
    with pytest.raises(StagedCheckoutSafetyError):
        staged.staged_checkout(None, index, "relative/path", localfs)
    assert not (tmp_path / "foo").exists()


# ---------------------------------------------------------------------------
# default apply path remains compatible
# ---------------------------------------------------------------------------


def test_default_apply_path_still_works(tmp_path, odb):
    index = _index(
        [
            _entry(("foo",), FOO),
            _entry(("data",), DIR_HASH, isdir=True),
        ],
        odb,
    )
    diff = compare(None, index)
    apply(diff, str(tmp_path), localfs)

    assert (tmp_path / "foo").read_text() == "foo\n"
    assert (tmp_path / "data" / "bar").read_text() == "bar\n"
    assert (tmp_path / "data" / "baz").read_text() == "baz\n"
    # the default path creates no staged checkout infrastructure
    assert not (tmp_path / staged.STAGING_DIRNAME).exists()


def _only_plan_id(root):
    return next(
        name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))
    )
