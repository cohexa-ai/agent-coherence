# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""``CoherentVolume`` keys its records the way its coordinator keys file names.

At every local attach the volume reads the coordinator's fold profile from the
default ``/status`` tier, and every record it keeps per file (grants, reads,
hashes, handoffs, held paths) is keyed by that profile's fold key, so a
spelling the disk treats as the same file reaches the same record. Its managed
checks match globs as the coordinator's policy does. A refuse-mode volume
raises, with nothing sent, on a write through a link under ``managed`` to a
file outside it. A remote-endpoint volume sends its names as keys.

These tests run the real volume against a real coordinator in this process.
On a disk that folds nothing (Linux CI) the folding cases are skipped; the
macOS job runs them on a case-insensitive disk.
"""

from __future__ import annotations

import pickle
import urllib.error
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

import ccs.adapters.coherent_volume as coherent_volume_module
import ccs.cli._coherence_client as client_module
from ccs.adapters.claude_code.coordinator_server import CoordinatorHTTPServer
from ccs.adapters.claude_code.lifecycle import LifecycleConfig
from ccs.adapters.coherent_volume import CoherentVolume, install, uninstall
from ccs.cli._coherence_client import (
    CoordinatorUnavailable,
    resolve_endpoint,
    resolve_remote_endpoint,
)
from ccs.core.exceptions import (
    CoherenceError,
    CommitUnconfirmed,
    CoordinatorUnreachable,
    HandoffPathsInvalid,
    LinkOutOfManaged,
    OtherHolder,
    RefuseModeUnsupported,
    StaleView,
)
from ccs.core.path_fold import FoldProfile
from tests.adapters.test_coherent_volume_other_holder import (
    _CLAIM_ROUTE,
    _MANAGED,
    _POST_EDIT,
    _PRE_EDIT,
    _Wire,
)
from tests.coordinator_helpers import answer_with_status, serve_in_process, write_holders

_X = "data/x.txt"
_FOLD_FIELDS = ("fold_profile", "fold_table_unicode")


@pytest.fixture
def fast_cfg() -> LifecycleConfig:
    """Coordinator config tuned for fast tests (no idle shutdown)."""
    return LifecycleConfig(
        idle_shutdown_sec=0,
        port_file_retry_attempts=20,
        port_file_retry_interval_sec=0.05,
        connect_retry_attempts=10,
        connect_retry_interval_sec=0.05,
    )


@pytest.fixture
def server(tmp_path: Path) -> Iterator[CoordinatorHTTPServer]:
    """A strict coordinator for ``data/**`` running in this process."""
    (tmp_path / "data").mkdir()
    running = serve_in_process(tmp_path, "fold")
    yield running
    running.shutdown()


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch) -> _Wire:
    """Every POST the volumes send, the principal claim included."""
    recorded = _Wire()
    real_post = client_module.post

    def spy(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
        recorded.sent.append((path, dict(payload)))
        return real_post(endpoint, path, payload, **kwargs)

    monkeypatch.setattr(client_module, "post", spy)
    monkeypatch.setattr(coherent_volume_module, "_coordinator_post", spy)
    return recorded


def _keying_profile(server: CoordinatorHTTPServer) -> FoldProfile:
    """The profile the coordinator keys file names by."""
    return server.fold_index_profile or "none"


def _require_caseless(server: CoordinatorHTTPServer) -> None:
    if _keying_profile(server) != "caseless":
        pytest.skip("needs a workspace disk that folds case (the macOS job runs it)")


def _holder(root: Path, cfg: LifecycleConfig) -> CoherentVolume:
    return CoherentVolume(root, managed=_MANAGED, config=cfg)


def _refuser(root: Path, cfg: LifecycleConfig) -> CoherentVolume:
    return CoherentVolume(root, managed=_MANAGED, config=cfg, on_held="refuse")


def _releases(wire: _Wire) -> list[dict]:
    return [body for body in wire.bodies(_POST_EDIT) if body.get("success") is False]


def _status_changed_to(monkeypatch: pytest.MonkeyPatch, change: Callable[[dict], object]) -> None:
    """Pass every default-tier ``/status`` answer through ``change``; every
    other GET is untouched."""
    real_get = coherent_volume_module._coordinator_get

    def view(endpoint: object, path: str, **kwargs: object) -> object:
        doc = real_get(endpoint, path, **kwargs)
        return change(doc) if path == "/status" else doc

    monkeypatch.setattr(coherent_volume_module, "_coordinator_get", view)


def _status_raises(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    """Fail every default-tier ``/status`` GET with ``error``."""
    real_get = coherent_volume_module._coordinator_get

    def view(endpoint: object, path: str, **kwargs: object) -> object:
        if path == "/status":
            raise error
        return real_get(endpoint, path, **kwargs)

    monkeypatch.setattr(coherent_volume_module, "_coordinator_get", view)


def _without_fold_fields(doc: dict) -> dict:
    return {k: v for k, v in doc.items() if k not in _FOLD_FIELDS}


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("/status", code, "scripted", {}, None)  # type: ignore[arg-type]


# --- a refuse-mode write through a link out of managed ---------------------


@pytest.mark.parametrize(
    ("link", "target", "written"),
    [
        ("data/plan-link.md", "../plan.md", "data/plan-link.md"),
        ("data/plan-link.md", "../plan.md", "DATA/plan-link.md"),
        ("data/sub", "../elsewhere", "data/sub/plan.md"),
    ],
    ids=["file-link", "file-link-other-case", "directory-link"],
)
def test_a_refuse_mode_write_through_a_link_out_of_managed_raises_and_sends_nothing(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    wire: _Wire,
    link: str,
    target: str,
    written: str,
) -> None:
    """The name written lies under ``managed``, the file it reaches does not, so
    the opted-in refusal could not apply: the write raises before any request.
    The link is made after the volume attached, so the check runs at ``write()``.
    On a case-insensitive disk ``DATA/plan-link.md`` names the same link.

    Prevents a refuse-mode write that silently takes a file outside the
    volume's guarantee from its holder."""
    if written.startswith("DATA/"):
        _require_caseless(server)
    (tmp_path / "plan.md").write_bytes(b"seed")
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "plan.md").write_bytes(b"seed")
    volume = _refuser(tmp_path, fast_cfg)
    (tmp_path / link).symlink_to(target)
    wire.sent.clear()

    with pytest.raises(LinkOutOfManaged) as raised:
        volume.write(written, b"through-the-link")

    assert wire.sent == []
    assert (tmp_path / "plan.md").read_bytes() == b"seed"
    assert (tmp_path / "elsewhere" / "plan.md").read_bytes() == b"seed"
    refusal = raised.value
    assert refusal.reason == "link_out_of_managed"
    assert refusal.path == written
    assert refusal.link == written[: len(link)]
    assert refusal.target in ("plan.md", "elsewhere/plan.md")
    # Our own refusal, round-tripped as a process boundary would carry it.
    assert type(pickle.loads(pickle.dumps(refusal))) is LinkOutOfManaged


def test_a_displace_mode_write_through_a_link_out_of_managed_still_lands(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer
) -> None:
    """Only refuse mode raises: a displacing volume promises no holder anything.

    Pins that the check is refuse-mode only, so a change to it is seen."""
    (tmp_path / "plan.md").write_bytes(b"seed")
    (tmp_path / "data" / "plan-link.md").symlink_to("../plan.md")
    volume = _holder(tmp_path, fast_cfg)

    volume.write("data/plan-link.md", b"through-the-link")

    assert (tmp_path / "plan.md").read_bytes() == b"through-the-link"


def test_a_link_under_managed_to_a_file_under_managed_writes_as_before(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """A link that stays inside ``managed`` is the file it names: the volume
    opts in on its target and the write lands.

    Prevents the link check refusing a write the refusal covers."""
    (tmp_path / "data" / "real.md").write_bytes(b"seed")
    (tmp_path / "data" / "alias.md").symlink_to("real.md")
    volume = _refuser(tmp_path, fast_cfg)
    wire.sent.clear()

    volume.write("data/alias.md", b"through-the-link")

    [pre_edit] = wire.bodies(_PRE_EDIT)
    assert pre_edit["path"] == "data/real.md"
    assert pre_edit["if_unheld"] is True
    assert (tmp_path / "data" / "real.md").read_bytes() == b"through-the-link"


def test_a_parent_step_with_no_link_writes_as_today(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """``data/../plan.md`` crosses no link: it is ``plan.md``, outside
    ``managed``, so it takes no flag and lands, as before.

    Prevents the link check reading a ``..`` step as a link."""
    (tmp_path / "plan.md").write_bytes(b"seed")
    volume = _refuser(tmp_path, fast_cfg)
    wire.sent.clear()

    volume.write("data/../plan.md", b"from-the-volume")

    [pre_edit] = wire.bodies(_PRE_EDIT)
    assert "if_unheld" not in pre_edit
    assert (tmp_path / "plan.md").read_bytes() == b"from-the-volume"


def test_a_link_outside_managed_to_a_file_outside_it_writes_as_today(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """Only a link whose name is managed promises the refusal: a link outside
    ``managed`` to a file outside it is an ordinary unmanaged write.

    Prevents the link check refusing every write through a link."""
    (tmp_path / "plan.md").write_bytes(b"seed")
    (tmp_path / "notes-link.md").symlink_to("plan.md")
    volume = _refuser(tmp_path, fast_cfg)
    wire.sent.clear()

    volume.write("notes-link.md", b"through-the-link")

    [pre_edit] = wire.bodies(_PRE_EDIT)
    assert "if_unheld" not in pre_edit
    assert (tmp_path / "plan.md").read_bytes() == b"through-the-link"


def test_a_link_above_the_root_is_not_under_managed(
    tmp_path: Path, fast_cfg: LifecycleConfig, wire: _Wire
) -> None:
    """A path that reaches the workspace through a link to its root crosses
    no link under ``managed``, so a write of a file outside it lands.

    Prevents the link check failing on a link it cannot place in the root."""
    root = tmp_path / "workspace"
    (root / "data").mkdir(parents=True)
    (tmp_path / "workspace-link").symlink_to(root)
    (root / "plan.md").write_bytes(b"seed")
    running = serve_in_process(root, "fold-root-link")
    try:
        volume = _refuser(root, fast_cfg)
        volume.write(tmp_path / "workspace-link" / "plan.md", b"from-the-volume")
    finally:
        running.shutdown()

    assert (root / "plan.md").read_bytes() == b"from-the-volume"


def test_a_glob_meets_the_disk_spelling_of_a_managed_file(
    tmp_path: Path, fast_cfg: LifecycleConfig, wire: _Wire
) -> None:
    """``data/STRASSE.MD`` names ``data/straße.md`` on a case-insensitive disk.
    The folded name has one letter too many for ``stra?e``, but the disk's own
    spelling matches, as it does for the coordinator, so refuse mode opts in.

    Prevents a write the coordinator holds strict that the volume does not
    opt in on, which would displace the file's holder."""
    (tmp_path / "data").mkdir()
    running = serve_in_process(tmp_path, "fold-disk-spelling", tracked="data/stra?e.md")
    try:
        _require_caseless(running)
        (tmp_path / "data" / "straße.md").write_bytes(b"seed")
        volume = CoherentVolume(
            tmp_path, managed=("data/stra?e.md",), config=fast_cfg, on_held="refuse"
        )
        wire.sent.clear()
        volume.write("data/STRASSE.MD", b"from-the-volume")
    finally:
        running.shutdown()

    [pre_edit] = wire.bodies(_PRE_EDIT)
    assert pre_edit["if_unheld"] is True


# --- whether a write would ask first, read without sending -----------------

_PLANS = ("plans/**",)


@pytest.fixture
def plans_server(tmp_path: Path) -> Iterator[CoordinatorHTTPServer]:
    """A strict coordinator for ``plans/**`` over a workspace where
    ``notes/into.md`` links to ``plans/a.md`` and ``plans/link.md`` links to
    ``notes/b.md``."""
    for directory in ("plans", "notes"):
        (tmp_path / directory).mkdir()
    (tmp_path / "plans" / "a.md").write_bytes(b"seed")
    (tmp_path / "notes" / "b.md").write_bytes(b"seed")
    (tmp_path / "notes" / "into.md").symlink_to("../plans/a.md")
    (tmp_path / "plans" / "link.md").symlink_to("../notes/b.md")
    running = serve_in_process(tmp_path, "asks-first", tracked="plans/**")
    yield running
    running.shutdown()


def _plan_paths(server: CoordinatorHTTPServer) -> dict[str, bool]:
    """Each path, and whether the file it reaches is managed. ``PLANS/a.md``
    is ``plans/a.md`` only on a disk that folds case."""
    managed = {
        "plans/a.md": True,
        "./plans//a.md": True,
        "notes/into.md": True,
        "plans/link.md": False,
        "notes/b.md": False,
    }
    if _keying_profile(server) == "caseless":
        managed["PLANS/a.md"] = True
    return managed


def test_a_refuse_mode_volume_asks_first_exactly_where_its_write_does(
    tmp_path: Path, fast_cfg: LifecycleConfig, plans_server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """Every spelling of a managed file asks first, a link into ``plans/``
    too; a link in ``plans/`` to a file outside it does not, since its write
    raises ``LinkOutOfManaged``, nor does a file outside. Asking sends nothing,
    and each write then does what the answer said.

    Prevents a server judging a path by a glob match of its own, which reads
    a folded or linked spelling of a managed file as unmanaged."""
    volume = CoherentVolume(tmp_path, managed=_PLANS, config=fast_cfg, on_held="refuse")
    expected = _plan_paths(plans_server)
    wire.sent.clear()

    asks = {path: volume.asks_if_unheld(path) for path in expected}
    managed = {path: volume.is_managed(path) for path in expected}

    assert asks == expected
    assert managed == expected
    assert wire.sent == []
    with pytest.raises(LinkOutOfManaged):
        volume.write("plans/link.md", b"through the link")
    assert wire.sent == []
    for path in expected.keys() - {"plans/link.md"}:
        wire.sent.clear()
        volume.write(path, b"from the volume")
        [pre_edit] = wire.bodies(_PRE_EDIT)
        assert pre_edit.get("if_unheld", False) is asks[path], path


def test_a_displacing_volume_asks_first_nowhere_and_still_names_its_managed_files(
    tmp_path: Path, fast_cfg: LifecycleConfig, plans_server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """Only refuse mode asks first; which files are managed does not depend on
    the mode.

    Prevents a displacing server reporting a write that asked first, or
    reading its managed files as unmanaged."""
    volume = CoherentVolume(tmp_path, managed=_PLANS, config=fast_cfg)
    expected = _plan_paths(plans_server)
    wire.sent.clear()

    assert {path: volume.asks_if_unheld(path) for path in expected} == dict.fromkeys(expected, False)
    assert {path: volume.is_managed(path) for path in expected} == expected
    assert wire.sent == []


def test_a_path_python_cannot_resolve_asks_nothing_and_never_raises(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    plans_server: CoordinatorHTTPServer,
    wire: _Wire,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Python 3.11 and 3.12 raise ``RuntimeError`` resolving a link that loops,
    where 3.13 keeps its name, and the MCP server accepts such a path. There
    ``write()`` raises before it sends anything, so it does not ask first, and
    neither read raises. Reproduced on every Python by resolving that link as
    3.11 does.

    Prevents a status or write answer that fails on a looping link."""
    (tmp_path / "plans" / "loop").symlink_to("loop")
    volume = CoherentVolume(tmp_path, managed=_PLANS, config=fast_cfg, on_held="refuse")
    real_resolve = Path.resolve

    def resolve_as_python_3_11(self: Path, strict: bool = False) -> Path:
        if self.name == "loop":
            raise RuntimeError(f"Symlink loop from {str(self)!r}")
        return real_resolve(self, strict=strict)

    monkeypatch.setattr(Path, "resolve", resolve_as_python_3_11)
    wire.sent.clear()

    assert volume.asks_if_unheld("plans/loop") is False
    assert volume.is_managed("plans/loop") is False
    with pytest.raises(RuntimeError):
        volume.write("plans/loop", b"through the loop")
    assert wire.sent == []


# --- records keyed by the fold key -----------------------------------------


def test_a_path_written_under_one_spelling_releases_under_another(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """On a case-insensitive disk ``DATA/x.txt`` and ``data/x.txt`` are one
    file, so the volume's record of the grant answers either spelling, and a
    refuse-mode peer is admitted once it is released.

    Prevents a release that finds nothing for a spelling the coordinator
    keys as the file the volume holds."""
    _require_caseless(server)
    holder, peer = _holder(tmp_path, fast_cfg), _refuser(tmp_path, fast_cfg)
    holder.write("DATA/x.txt", b"from the holder")
    wire.sent.clear()

    holder.release(_X)

    assert len(_releases(wire)) == 1
    peer.write(_X, b"from the peer")
    assert write_holders(server, _X) == {peer.agent_id}


def test_another_spelling_of_a_path_the_volume_did_not_write_sends_nothing(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """Folding finds the record of the file a spelling names, and no other.

    Prevents a release sent for every path once spellings fold."""
    holder = _holder(tmp_path, fast_cfg)
    holder.write(_X, b"from the holder")
    wire.sent.clear()

    holder.release("DATA/Z.TXT")

    assert wire.sent == []
    assert write_holders(server, _X) == {holder.agent_id}


def test_a_read_under_one_spelling_guards_a_write_under_another(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer
) -> None:
    """The foreign-edit baseline a read leaves is the file's, so a write under
    another spelling of it sees an edit made on disk since the read.

    Prevents a write that overwrites an out-of-band edit because it named the
    file in another case."""
    _require_caseless(server)
    (tmp_path / _X).write_bytes(b"seed")
    volume = _holder(tmp_path, fast_cfg)
    volume.read(_X)
    (tmp_path / _X).write_bytes(b"edited behind the volume")

    with pytest.raises(StaleView):
        volume.write("DATA/X.TXT", b"from-the-volume")

    assert (tmp_path / _X).read_bytes() == b"edited behind the volume"


def test_two_spellings_of_one_file_in_one_request_are_a_duplicate(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """A transfer or a publish naming one file twice is the caller's error,
    and two spellings of one file name it twice: refused, with nothing sent.

    Prevents one file handed off or published twice in one request."""
    _require_caseless(server)
    (tmp_path / _X).write_bytes(b"seed")
    volume = _holder(tmp_path, fast_cfg)
    wire.sent.clear()

    with pytest.raises(HandoffPathsInvalid, match="more than once"):
        volume.transfer([_X, "DATA/X.TXT"], successor="5d0f3c9a8b7e4d21a6c3b2e1f0d9c8b7")
    with pytest.raises(ValueError, match="duplicate"):
        volume.atomic_publish([(_X, 1, b"one"), ("DATA/X.TXT", 1, b"two")])

    assert wire.sent == []


@pytest.fixture
def caseless_keyed_server(
    server: CoordinatorHTTPServer, monkeypatch: pytest.MonkeyPatch
) -> CoordinatorHTTPServer:
    """The ``server``, read by every volume as keyed by ``caseless``, whatever the disk."""
    _status_changed_to(monkeypatch, lambda doc: {**doc, "fold_profile": "caseless"})
    return server


def _is_write_commit(path: str, payload: dict) -> bool:
    """write()'s commit, not the release it sends on the same route."""
    return path == _POST_EDIT and payload.get("success") is True


def test_a_confirmed_release_after_an_unanswered_commit_clears_the_fold_keys_record(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    caseless_keyed_server: CoordinatorHTTPServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """write() records its grant under the fold key, so when its commit is
    answered with HTTP 503 and the release it sends first is confirmed, the
    record is cleared under that key, not under the spelling written.

    Prevents a confirmed release that leaves the grant in the volume's
    record, so a later release() or re-mint sends it again."""
    volume = _holder(tmp_path, fast_cfg)
    holder = volume._incarnation
    calls: list[tuple[str, dict, dict]] = []
    answer_with_status(monkeypatch, _is_write_commit, 503, calls=calls)

    with pytest.raises(CommitUnconfirmed):
        volume.write("data/X.txt", b"from the holder")

    assert volume._fold_profile == "caseless", "premise: the volume keys by caseless"
    released = [
        body["path"] for route, body, _ in calls
        if route == _POST_EDIT and body.get("success") is False
    ]
    assert released == ["data/X.txt"], "control: one release, sent under the spelling written"
    assert not volume._records_write_grant(_X)
    assert holder not in volume._grant_incarnations


def test_the_open_shim_mediates_another_spelling_of_a_managed_file(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    wire: _Wire,
) -> None:
    """The shim's managed check reads a name as the coordinator does, so
    ``DATA/x.txt`` is a managed file on a case-insensitive disk and its write
    goes through the volume.

    Prevents a managed file written around the coordinator under another case."""
    _require_caseless(server)
    install(tmp_path, managed=_MANAGED, config=fast_cfg)
    try:
        wire.sent.clear()
        with open(tmp_path / "DATA" / "x.txt", "wb") as handle:
            handle.write(b"through the shim")
    finally:
        uninstall()

    assert [body["path"] for body in wire.bodies(_PRE_EDIT)] == ["DATA/x.txt"]
    assert (tmp_path / _X).read_bytes() == b"through the shim"


def test_the_shims_managed_check_never_raises_on_a_link_that_does_not_resolve(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer
) -> None:
    """The disk's spelling of a looping link cannot be found, so the check
    reads the name the other ways, and answers, as the shim's check always
    has: an open() it cannot place goes on as an ordinary call would.

    Prevents an open() that raises from inside the managed check."""
    _require_caseless(server)
    (tmp_path / "data" / "loop").symlink_to("loop")
    volume = _holder(tmp_path, fast_cfg)

    assert coherent_volume_module._managed_rel(volume, tmp_path / "data" / "loop") == "data/loop"


@pytest.fixture
def none_keyed_server(server: CoordinatorHTTPServer, monkeypatch: pytest.MonkeyPatch) -> CoordinatorHTTPServer:
    """The ``server``, read by every volume as keyed by ``none``, whatever the disk."""
    _status_changed_to(monkeypatch, _without_fold_fields)
    return server


def test_the_shims_managed_check_names_a_looping_link_on_every_python(
    tmp_path: Path, fast_cfg: LifecycleConfig, none_keyed_server: CoordinatorHTTPServer
) -> None:
    """A link that loops resolves to its own name on every Python the
    package supports, so the check answers on a volume keyed by ``none``,
    where no disk spelling is read, as on every Linux disk.

    Prevents the check raising on Python 3.11 and 3.12, whose non-strict
    ``Path.resolve()`` raises RuntimeError on a loop where 3.13 keeps the
    name."""
    (tmp_path / "data" / "loop").symlink_to("loop")
    volume = _holder(tmp_path, fast_cfg)

    assert volume._fold_profile == "none"
    assert coherent_volume_module._managed_rel(volume, tmp_path / "data" / "loop") == "data/loop"


# --- what the attach reads ---------------------------------------------------


def test_an_older_coordinator_without_fold_fields_is_keyed_by_none(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    wire: _Wire,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A coordinator that advertises no profile keys names exactly, so the
    volume does too: another spelling of a held path finds no record, even
    on a disk that folds. A refuse-mode volume still attaches.

    Prevents a volume folding names its coordinator keeps apart."""
    _status_changed_to(monkeypatch, _without_fold_fields)
    holder, peer = _holder(tmp_path, fast_cfg), _refuser(tmp_path, fast_cfg)
    holder.write(_X, b"from the holder")
    wire.sent.clear()

    holder.release("data/X.TXT")

    assert _releases(wire) == []
    with pytest.raises(OtherHolder):
        peer.write(_X, b"from the peer")


@pytest.mark.parametrize(
    ("error", "raised_type"),
    [
        (CoordinatorUnavailable("scripted: no answer"), CoordinatorUnreachable),
        (_http_error(503), RefuseModeUnsupported),
        (None, RefuseModeUnsupported),
    ],
    ids=["no-answer", "http-error", "not-an-object"],
)
def test_an_unreadable_status_fails_a_refuse_mode_attach_closed(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    wire: _Wire,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception | None,
    raised_type: type[CoherenceError],
) -> None:
    """Without the profile the volume cannot tell which spellings are one file,
    so refuse mode, which promises a holder its file, fails closed before the
    claim: the coordinator receives nothing.

    Prevents a refuse-mode volume that keys a file two ways on a coordinator
    that keys it one way. A status it read but cannot rely on is the typed
    refusal a server stops on; one it got no answer to stays unreachable."""
    if error is None:
        _status_changed_to(monkeypatch, lambda doc: ["not", "an", "object"])
    else:
        _status_raises(monkeypatch, error)

    with pytest.raises(CoherenceError, match="fold profile") as raised:
        _refuser(tmp_path, fast_cfg)

    assert type(raised.value) is raised_type
    assert wire.sent == []


def test_an_unreadable_status_leaves_a_displacing_volume_keyed_by_none(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    wire: _Wire,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A volume that promises no holder anything attaches and keys exactly.

    Prevents an attach that fails for a field only refuse mode needs."""
    _status_raises(monkeypatch, _http_error(503))
    holder = _holder(tmp_path, fast_cfg)
    holder.write(_X, b"from the holder")
    wire.sent.clear()

    holder.release("data/X.TXT")

    assert _releases(wire) == []


def test_a_fold_table_of_another_unicode_version_refuses_a_refuse_mode_attach(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    wire: _Wire,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A coordinator folding by another table may key a name another way, so
    the volume keys exactly and refuse mode refuses to attach; a displacing
    volume attaches.

    Prevents a refuse-mode volume whose records and coordinator disagree on
    which spellings are one file."""
    _status_changed_to(monkeypatch, lambda doc: {**doc, "fold_table_unicode": "16.0.0"})

    with pytest.raises(CoherenceError, match="16.0.0") as raised:
        _refuser(tmp_path, fast_cfg)
    assert type(raised.value) is RefuseModeUnsupported
    assert wire.sent == []

    holder = _holder(tmp_path, fast_cfg)
    holder.write(_X, b"from the holder")
    wire.sent.clear()
    holder.release("data/X.TXT")
    assert _releases(wire) == []


@pytest.mark.parametrize("advertised", ["cannot_tell", "a-profile-from-later"])
def test_a_profile_the_volume_cannot_use_is_keyed_by_none(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    wire: _Wire,
    monkeypatch: pytest.MonkeyPatch,
    advertised: str,
) -> None:
    """``cannot_tell`` keys as ``none`` on the coordinator too, and a value the
    volume does not know is read the same way. Refuse mode still attaches.

    Prevents a volume guessing a fold."""
    _status_changed_to(monkeypatch, lambda doc: {**doc, "fold_profile": advertised})
    holder, _peer = _holder(tmp_path, fast_cfg), _refuser(tmp_path, fast_cfg)
    holder.write(_X, b"from the holder")
    wire.sent.clear()

    holder.release("data/X.TXT")

    assert _releases(wire) == []


def test_an_unmanaged_volume_reads_the_fold_profile_too(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    wire: _Wire,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every local attach reads the default ``/status`` tier, not only the
    managed volumes that read the operator view, and keys by what it says.

    Prevents an unmanaged volume keying a file two ways."""
    gets: list[str] = []
    real_get = coherent_volume_module._coordinator_get

    def recording(endpoint: object, path: str, **kwargs: object) -> object:
        gets.append(path)
        return real_get(endpoint, path, **kwargs)

    monkeypatch.setattr(coherent_volume_module, "_coordinator_get", recording)
    volume = CoherentVolume(tmp_path, config=fast_cfg)
    assert "/status" in gets
    if _keying_profile(server) != "caseless":
        return
    volume.write("DATA/x.txt", b"from the volume")
    wire.sent.clear()

    volume.release(_X)

    assert len(_releases(wire)) == 1


def test_a_remote_endpoint_volume_sends_its_names_as_keys(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    wire: _Wire,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A remote coordinator's disk is not this volume's, so the volume's paths
    are names, not its files: each request marks its path a key, and the
    volume keys its own records exactly.

    Prevents a remote coordinator folding, tidying or merging names it does
    not hold as files."""
    monkeypatch.setenv("CCS_REMOTE_COORDINATOR", "1")
    local = resolve_endpoint(tmp_path)
    remote = resolve_remote_endpoint("127.0.0.1", local.port, local.bearer)
    root = tmp_path / "remote-client"
    (root / "data").mkdir(parents=True)
    volume = CoherentVolume(root, config=fast_cfg, remote_endpoint=remote)

    volume.write(_X, b"from the remote volume")
    volume.read(_X)
    volume.release("data/X.TXT")
    assert _releases(wire) == []
    volume.release(_X)

    named = [(route, body) for route, body in wire.sent if "path" in body]
    assert {route for route, _ in named} >= {_PRE_EDIT, _POST_EDIT, "/hooks/pre-read"}
    assert all(body.get("name_kind") == "key" for _, body in named), named
    assert _CLAIM_ROUTE in wire.routes()


def test_a_remote_endpoint_volumes_best_effort_release_names_a_key(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    wire: _Wire,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The release a failed write sends on its own, outside the volume's usual
    request path, names the path a key too, so it reaches the row the grant
    was taken on.

    Prevents a release that a remote coordinator folds onto another row."""
    monkeypatch.setenv("CCS_REMOTE_COORDINATOR", "1")
    local = resolve_endpoint(tmp_path)
    remote = resolve_remote_endpoint("127.0.0.1", local.port, local.bearer)
    root = tmp_path / "remote-client"
    (root / "data").mkdir(parents=True)
    volume = CoherentVolume(root, config=fast_cfg, remote_endpoint=remote)

    def disk_full(self: CoherentVolume, abs_path: Path, data: bytes) -> None:
        raise OSError("scripted: the disk is full")

    monkeypatch.setattr(CoherentVolume, "_atomic_write", disk_full)
    with pytest.raises(OSError, match="disk is full"):
        volume.write(_X, b"from the remote volume")

    [release] = _releases(wire)
    assert release["name_kind"] == "key"
