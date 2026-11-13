# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""``swg_status`` names each path's write holder, the mode, and the paths this
session holds.

A holder is a session row of the coordinator's default ``/status`` tier that
holds the path ``EXCLUSIVE`` or ``MODIFIED``. These tests run real volumes
against a real coordinator in this process. ``/status`` is stubbed only for
answers this coordinator does not give here: one without a sessions list, a
degraded one, an unreachable coordinator, and one file listed under two
spellings, as rows an older build split are listed.
"""

from __future__ import annotations

import io
import urllib.error
from collections.abc import Callable, Iterator
from pathlib import Path
from uuid import UUID

import pytest

import ccs.adapters.coherent_volume as coherent_volume_module
from ccs.adapters.claude_code.lifecycle import LifecycleConfig
from ccs.adapters.coherent_volume import CoherentVolume
from ccs.core.exceptions import CoordinatorFailed, OtherHolder
from ccs.core.path_fold import FOLD_TABLE_UNICODE_VERSION
from ccs.mcp.session import SessionConfig
from ccs.mcp.status import build_status, status_text
from tests.coordinator_helpers import serve_in_process

_MANAGED = ("data/**",)
_MINE = "data/mine.md"
_STRANDED = "data/stranded.md"
_THEIRS = "data/theirs.md"
_SESSION_STOP = "/hooks/session-stop"
#: FROZEN copies of the starts of the text lines that say a list could not be told.
_PER_PATH_UNAVAILABLE = "per_path=unavailable: "
_HELD_UNAVAILABLE = "held=unavailable: "

VolumeFactory = Callable[..., CoherentVolume]


@pytest.fixture
def fast_cfg() -> LifecycleConfig:
    return LifecycleConfig(
        idle_shutdown_sec=0,
        port_file_retry_attempts=20,
        port_file_retry_interval_sec=0.05,
        connect_retry_attempts=10,
        connect_retry_interval_sec=0.05,
    )


@pytest.fixture
def workspace(tmp_path: Path) -> Iterator[Path]:
    """A strict coordinator for ``data/**`` running in this process."""
    (tmp_path / "data").mkdir()
    server = serve_in_process(tmp_path, "status-holders")
    yield tmp_path
    server.shutdown()


@pytest.fixture
def open_volume(workspace: Path, fast_cfg: LifecycleConfig) -> Iterator[VolumeFactory]:
    """Build volumes on the workspace; each is closed before the coordinator stops."""
    built: list[CoherentVolume] = []

    def build(on_held: str = "displace") -> CoherentVolume:
        volume = CoherentVolume(workspace, managed=_MANAGED, config=fast_cfg, on_held=on_held)
        built.append(volume)
        return volume

    yield build
    for volume in reversed(built):
        volume.close()


def _config(root: Path, on_held: str = "displace") -> SessionConfig:
    return SessionConfig(root=root, managed=_MANAGED, on_held=on_held)


def _hex(agent_id: str) -> str:
    """An agent id in the spelling a refusal names a holder by."""
    return UUID(agent_id).hex


def _strand_its_grants(volume: CoherentVolume, monkeypatch: pytest.MonkeyPatch) -> str:
    """Start a new attempt with a ``reacquire()`` whose release never reaches
    the coordinator, so every grant of the attempt before it stands. Returns
    that earlier attempt's id."""
    earlier = volume.agent_id
    real_post = coherent_volume_module._coordinator_post

    def post(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
        if path == _SESSION_STOP:
            raise urllib.error.HTTPError(path, 503, "scripted", {}, io.BytesIO(b"{}"))  # type: ignore[arg-type]
        return real_post(endpoint, path, payload, **kwargs)

    monkeypatch.setattr(coherent_volume_module, "_coordinator_post", post)
    with pytest.raises(CoordinatorFailed):
        volume.reacquire(_STRANDED)
    monkeypatch.setattr(coherent_volume_module, "_coordinator_post", real_post)
    assert volume.agent_id != earlier, "control: the reacquire started a new attempt"
    return earlier


def _listed_version(status_doc: dict, path: str) -> int:
    """The version the coordinator's ``/status`` lists for ``path``."""
    (version,) = [a["version"] for a in status_doc["tracked_artifacts"] if a["path"] == path]
    return version


# --- the three holders ------------------------------------------------------


def test_a_path_nobody_holds_keeps_todays_entry_and_nothing_is_held(
    open_volume: VolumeFactory, workspace: Path
) -> None:
    """A path read but held for writing by no one carries no holder key, and
    this session holds nothing. Prevents a holder key on every entry, which
    would change the shape a client already reads."""
    (workspace / _MINE).write_bytes(b"seed")
    volume = open_volume()
    _content, version = volume.read_with_version(_MINE)

    status = build_status(volume, _config(workspace))

    assert status["per_path"][_MINE] == {"version": version, "status": "enforced"}
    assert status["held"] == []


def test_this_servers_current_write_is_this_attempt_and_held(
    open_volume: VolumeFactory, workspace: Path
) -> None:
    """A path this server wrote is held by its current attempt, in the
    spelling refusals use, and listed as held. Prevents an agent that cannot
    find the paths it must release."""
    volume = open_volume()
    volume.write(_MINE, b"mine")

    status = build_status(volume, _config(workspace))

    assert status["per_path"][_MINE] == {
        "version": _listed_version(volume.coordinator_status(), _MINE),
        "status": "enforced",
        "holder_agent_id": _hex(volume.agent_id),
        "held_by": "this_attempt",
    }
    assert status["held"] == [_MINE]


def test_an_earlier_attempt_whose_release_did_not_land_is_this_sessions_and_held(
    open_volume: VolumeFactory, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A grant an earlier attempt kept, because its release never reached the
    coordinator, is named as this session's earlier attempt and listed as
    held. Prevents an agent waiting on its own stale grant as if another agent
    held the path."""
    volume = open_volume()
    volume.write(_STRANDED, b"first")
    earlier = _strand_its_grants(volume, monkeypatch)

    status = build_status(volume, _config(workspace))

    entry = status["per_path"][_STRANDED]
    assert entry["holder_agent_id"] == _hex(earlier)
    assert entry["held_by"] == "earlier_attempt"
    assert status["held"] == [_STRANDED]


def test_another_servers_hold_is_named_as_its_refusal_names_it_and_not_held(
    open_volume: VolumeFactory, workspace: Path
) -> None:
    """The Python coordinator's ``/status`` lists ids hyphenated; the holder
    is reported in the 32-hex spelling the refusal used, byte for byte, as
    another agent's, and is not this session's to release. Prevents an agent
    that cannot match a refusal's holder to a status entry."""
    holder, waiter = open_volume(), open_volume(on_held="refuse")
    holder.write(_THEIRS, b"theirs")
    with pytest.raises(OtherHolder) as refused:
        waiter.write(_THEIRS, b"mine")

    status = build_status(waiter, _config(workspace, "refuse"))

    entry = status["per_path"][_THEIRS]
    assert entry["holder_agent_id"] == refused.value.holder_agent_id
    assert entry["held_by"] == "another_agent"
    assert status["held"] == []


# --- the text channel -------------------------------------------------------


def test_the_text_names_the_mode_the_held_paths_and_every_hold_to_wait_on(
    open_volume: VolumeFactory, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After today's coordinator line, the text names the mode and the paths
    this session holds, then each path another agent or an earlier attempt
    holds, with its ``held_by`` and ``holder_agent_id``. A path this attempt
    holds is in the held list only. Prevents a client that shows only text
    from losing what a refused writer's wait acts on."""
    volume, peer = open_volume(on_held="refuse"), open_volume()
    volume.write(_STRANDED, b"first")
    earlier = _strand_its_grants(volume, monkeypatch)
    volume.write(_MINE, b"mine")
    peer.write(_THEIRS, b"theirs")

    text = status_text(build_status(volume, _config(workspace, "refuse")))

    assert text.splitlines() == [
        "coordinator=on",
        'on_held=refuse held=["data/mine.md", "data/stranded.md"]',
        f'path="data/stranded.md" held_by=earlier_attempt holder_agent_id={_hex(earlier)}',
        f'path="data/theirs.md" held_by=another_agent holder_agent_id={_hex(peer.agent_id)}',
    ]


@pytest.mark.parametrize("on_held", ["refuse", "displace"])
def test_on_held_is_the_servers_mode(open_volume: VolumeFactory, workspace: Path, on_held: str) -> None:
    """Both channels name the mode the server was started in."""
    volume = open_volume(on_held=on_held)

    status = build_status(volume, _config(workspace, on_held))

    assert status["on_held"] == on_held
    assert status_text(status).splitlines()[1] == f"on_held={on_held} held=[]"


# --- which entries count as managed -----------------------------------------


def test_a_stored_spelling_that_folds_into_a_managed_glob_reads_enforced(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """On a disk that folds case, the coordinator stores ``Data/x.md`` as the
    disk spells it, and this server's writes of it are coordinated under
    ``data/**``, so it reads ``enforced``. Prevents a managed file reported as
    not enforced because a glob was matched against its name as typed."""
    (tmp_path / "Data").mkdir()
    (tmp_path / "Data" / "x.md").write_bytes(b"seed")
    server = serve_in_process(tmp_path, "status-folded")
    try:
        if server.fold_index_profile != "caseless":
            pytest.skip("needs a workspace disk that folds case (the macOS job runs it)")
        volume = CoherentVolume(tmp_path, managed=_MANAGED, config=fast_cfg)
        try:
            _content, version = volume.read_with_version("data/x.md")
            per_path = build_status(volume, _config(tmp_path))["per_path"]
        finally:
            volume.close()
    finally:
        server.shutdown()

    assert per_path == {"Data/x.md": {"version": version, "status": "enforced"}}


def test_a_tracked_file_now_linked_outside_the_root_reads_not_registered(
    open_volume: VolumeFactory, workspace: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """A tracked file replaced by a link out of the workspace is one this
    server cannot write, so it reads ``not_registered``, and the rest of the
    answer still comes back. Prevents one such link failing every status
    call, which a refused writer polls while it waits."""
    (workspace / _MINE).write_bytes(b"seed")
    volume = open_volume()
    volume.read(_MINE)
    elsewhere = tmp_path_factory.mktemp("outside") / "elsewhere.md"
    elsewhere.write_bytes(b"elsewhere")
    (workspace / _MINE).unlink()
    (workspace / _MINE).symlink_to(elsewhere)

    status = build_status(volume, _config(workspace))

    assert status["per_path"][_MINE]["status"] == "not_registered"
    assert status["held"] == []


# --- one file under two spellings -------------------------------------------


def _listed_under(status_doc: dict, names: list[str], holders: list[tuple[str, str]]) -> dict:
    """``status_doc`` as a caseless coordinator lists ``names``, each a row of
    its own, with each ``(agent id, name)`` of ``holders`` holding that name
    for writing."""
    return {
        **status_doc,
        "fold_profile": "caseless",
        "fold_table_unicode": FOLD_TABLE_UNICODE_VERSION,
        "tracked_artifacts": [{"path": name, "version": 1} for name in names],
        "sessions": [
            {"agent_name": None, "agent_id": agent, "states": {name: "EXCLUSIVE"}} for agent, name in holders
        ],
    }


def _holders_of(status: dict) -> dict[str, tuple[str | None, str | None]]:
    return {path: (entry.get("holder_agent_id"), entry.get("held_by")) for path, entry in status["per_path"].items()}


def test_a_file_held_under_another_spelling_gets_its_holder(
    open_volume: VolumeFactory, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rows an older build split list one file as ``data/x.md`` and
    ``DATA/X.md``; the holder of ``DATA/X.md`` holds both, so both entries
    name it and both are held. Prevents a held file read as nobody's under
    the spelling the holder's row does not use."""
    volume = open_volume()
    answer = _listed_under(volume.coordinator_status(), ["data/x.md", "DATA/X.md"], [(volume.agent_id, "DATA/X.md")])
    monkeypatch.setattr(volume, "coordinator_status", lambda: answer)

    status = build_status(volume, _config(workspace))

    mine = (_hex(volume.agent_id), "this_attempt")
    assert _holders_of(status) == {"data/x.md": mine, "DATA/X.md": mine}
    assert status["held"] == ["DATA/X.md", "data/x.md"]


def test_a_file_two_split_rows_give_two_holders_names_only_the_first(
    open_volume: VolumeFactory, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When each split row has a holder of its own, the file reports one: the
    first the coordinator lists. Prevents one file shown with two holders."""
    volume, peer = open_volume(), open_volume()
    answer = _listed_under(
        volume.coordinator_status(),
        ["data/x.md", "DATA/X.md"],
        [(peer.agent_id, "data/x.md"), (volume.agent_id, "DATA/X.md")],
    )
    monkeypatch.setattr(volume, "coordinator_status", lambda: answer)

    status = build_status(volume, _config(workspace))

    theirs = (_hex(peer.agent_id), "another_agent")
    assert _holders_of(status) == {"data/x.md": theirs, "DATA/X.md": theirs}
    assert status["held"] == []


# --- answers that cannot list holders ---------------------------------------


def test_an_answer_without_a_sessions_list_names_no_holder_and_says_so(
    open_volume: VolumeFactory, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An answer that lists tracked paths but no sessions cannot tell who
    holds them: no entry gets a holder key, ``held`` is null, never empty,
    and the text says holders are unavailable. Prevents a held path read as
    one nobody holds."""
    volume = open_volume()
    volume.write(_MINE, b"mine")
    real = volume.coordinator_status()
    without_sessions = {key: value for key, value in real.items() if key != "sessions"}
    monkeypatch.setattr(volume, "coordinator_status", lambda: without_sessions)

    status = build_status(volume, _config(workspace))

    assert status["per_path"][_MINE] == {"version": _listed_version(real, _MINE), "status": "enforced"}
    assert status["held"] is None
    assert status_text(status).splitlines()[1].startswith(f"on_held=displace {_HELD_UNAVAILABLE}")


@pytest.mark.parametrize(
    "rows",
    [["not-a-row"], [{"agent_id": "not-an-id", "states": {}}], [{"agent_id": str(UUID(int=7)), "states": None}]],
    ids=["row-not-an-object", "id-not-an-agent-id", "states-not-an-object"],
)
def test_a_session_row_that_cannot_be_read_makes_the_holders_unknown(
    open_volume: VolumeFactory, workspace: Path, monkeypatch: pytest.MonkeyPatch, rows: list
) -> None:
    """A row that does not say whose it is or what it holds could hold any
    path, so beside a readable holder the answer still cannot tell holders:
    no holder keys and ``held`` null. Prevents an unreadable row read as one
    that holds nothing."""
    volume = open_volume()
    volume.write(_MINE, b"mine")
    real = volume.coordinator_status()
    with_unreadable_row = {**real, "sessions": [*real["sessions"], *rows]}
    monkeypatch.setattr(volume, "coordinator_status", lambda: with_unreadable_row)

    status = build_status(volume, _config(workspace))

    assert status["per_path"][_MINE] == {"version": _listed_version(real, _MINE), "status": "enforced"}
    assert status["held"] is None


@pytest.mark.parametrize(
    "lists",
    [{"tracked_artifacts": None, "sessions": None, "degraded": True}, {"tracked_artifacts": None}],
    ids=["degraded", "sessions-without-artifacts"],
)
def test_an_answer_without_an_artifact_list_keeps_per_path_null_and_cannot_tell_holders(
    open_volume: VolumeFactory, workspace: Path, monkeypatch: pytest.MonkeyPatch, lists: dict
) -> None:
    """A degraded answer, both lists null, keeps today's ``per_path: null``
    and first text line, and ``held`` is null with the text saying so; so does
    an answer that lists sessions but no tracked paths, whose held paths
    cannot be matched to an entry."""
    volume = open_volume()
    volume.write(_MINE, b"mine")
    degraded = {**volume.coordinator_status(), **lists}
    monkeypatch.setattr(volume, "coordinator_status", lambda: degraded)

    status = build_status(volume, _config(workspace))

    assert status["per_path"] is None
    assert status["held"] is None
    first, held_line = status_text(status).splitlines()
    assert first.startswith(f"coordinator=on {_PER_PATH_UNAVAILABLE}")
    assert held_line.startswith(f"on_held=displace {_HELD_UNAVAILABLE}")


def test_an_unreachable_coordinator_cannot_tell_holders(
    open_volume: VolumeFactory, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the coordinator unknown, ``per_path`` stays ``{}`` as today and
    ``held`` is null, never empty: an empty list would read as "this session
    holds nothing"."""
    volume = open_volume()
    volume.write(_MINE, b"mine")
    monkeypatch.setattr(volume, "coordinator_status", lambda: None)

    status = build_status(volume, _config(workspace))

    assert status["per_path"] == {}
    assert status["held"] is None
    first, held_line = status_text(status).splitlines()
    assert first == "coordinator=unknown"
    assert held_line.startswith(f"on_held=displace {_HELD_UNAVAILABLE}")
