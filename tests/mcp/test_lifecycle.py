"""Unit 2 — coordinator lifecycle, fail-closed binding, stdout purity, serialization.

The server owns the coordinator for its lifetime: constructing the per-session
volume self-spawns/attaches (strict-only, fail-closed); the lifespan calls
``stop_coordinator`` on exit and never double-spawns. Two invariants are
load-bearing: the JSON-RPC stdout channel must stay byte-clean (the coordinator
subprocess's stdout must not leak to fd 1), and ``import ccs.mcp`` must work
without the optional ``mcp`` SDK (lazy package).
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import ccs.adapters.coherent_volume as coherent_volume_module
import ccs.cli._coherence_client as client_module
import ccs.mcp.server as server_module
from ccs.adapters.claude_code.lifecycle import LifecycleConfig, stop_coordinator
from ccs.adapters.coherent_volume import CoherentVolume
from ccs.cli._coherence_client import CoordinatorUnavailable, escape_nonprintable
from ccs.core.exceptions import CoherenceDirRefused, CoherenceError, CoordinatorUnreachable
from ccs.mcp.server import build_server, lifespan
from ccs.mcp.session import SessionConfig, build_volume
from tests.adapters.test_coherent_volume_other_holder import _NOT_ADVERTISED, _POST_EDIT, _PRE_EDIT, _Wire
from tests.coherence_link_helpers import plant_link
from tests.coordinator_helpers import serve_in_process, write_holders

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The server's console script, run as an agent host runs it.
_MAIN = "from ccs.mcp.server import main; main()"

#: The console script against a real coordinator whose operator view loses
#: ``pre_edit_if_unheld`` on the way in: every coordinator this package starts
#: advertises it, so this is the one boundary stubbed.
_MAIN_WITHOUT_THE_REFUSAL = textwrap.dedent(
    """
    import ccs.adapters.coherent_volume as coherent_volume
    from ccs.mcp.server import main

    real_get = coherent_volume._coordinator_get

    def operator_view_without_the_refusal(endpoint, path, **kwargs):
        answer = real_get(endpoint, path, **kwargs)
        if "detail=full" in path:
            answer.pop("pre_edit_if_unheld", None)
        return answer

    coherent_volume._coordinator_get = operator_view_without_the_refusal
    main()
    """
)


@pytest.fixture
def fast_cfg() -> LifecycleConfig:
    """Coordinator config tuned for fast tests (no idle shutdown)."""
    return LifecycleConfig(
        idle_shutdown_sec=0,
        sweep_interval_sec=0.1,
        notice_evict_max_age_sec=1.0,
        port_file_retry_attempts=20,
        port_file_retry_interval_sec=0.05,
        connect_retry_attempts=10,
        connect_retry_interval_sec=0.05,
    )


def _seed(tmp_path: Path, rel: str = "data/shared.txt", content: bytes = b"v1") -> Path:
    target = tmp_path / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return target


def _subprocess_env(tmp_path: Path, managed: str = "data/**") -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT / "src"),
        "SWG_ROOT": str(tmp_path),
        "SWG_MANAGED": managed,
    }


def _run_the_server(tmp_path: Path, on_held: str, script: str = _MAIN) -> subprocess.CompletedProcess[bytes]:
    """Run the console script with ``SWG_ON_HELD=on_held`` and no client."""
    return subprocess.run(
        [sys.executable, "-c", script],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        env={**_subprocess_env(tmp_path), "SWG_ON_HELD": on_held},
        timeout=60,
    )


@pytest.fixture
def wire(monkeypatch) -> _Wire:
    """Every request the server's volume sends: each POST under its route, each
    GET as ``GET <path>``, so a request a write never sent before shows up. The
    principal claim goes out through the client module, every other request
    through the volume's aliases."""
    recorded = _Wire()
    real_post, real_get = client_module.post, client_module.get

    def post(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
        recorded.sent.append((path, dict(payload)))
        return real_post(endpoint, path, payload, **kwargs)

    def get(endpoint: object, path: str, **kwargs: object) -> object:
        recorded.sent.append((f"GET {path}", {}))
        return real_get(endpoint, path, **kwargs)

    monkeypatch.setattr(client_module, "post", post)
    monkeypatch.setattr(client_module, "get", get)
    monkeypatch.setattr(coherent_volume_module, "_coordinator_post", post)
    monkeypatch.setattr(coherent_volume_module, "_coordinator_get", get)
    return recorded


# --- happy path: attach + strict + clean stop --------------------------------


def test_lifespan_attaches_strict_and_stops(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SWG_ROOT", str(tmp_path))
    monkeypatch.setenv("SWG_MANAGED", "data/**")
    _seed(tmp_path)
    server = build_server()

    async def go() -> None:
        async with lifespan(server) as ctx:
            assert ctx.volume.is_attached
            assert ctx.volume.strict_mode_active() is True
            assert not ctx.volume.is_degraded
            assert ctx.config.root == tmp_path.resolve()
            assert ctx.config.managed == ("data/**",)

    asyncio.run(go())
    # The lifespan's finally already stopped the coordinator; a second stop in
    # THIS process reports False (nothing left for us to stop).
    assert stop_coordinator(tmp_path) is False


def test_lifespan_refuses_a_policy_file_it_cannot_load_and_keeps_it(tmp_path: Path, monkeypatch) -> None:
    """#305 at server startup, the attach that spawns the coordinator: the error names
    the file, as every other attach refusal does. Prevents startup replacing the file
    with only the managed globs."""
    monkeypatch.setenv("SWG_ROOT", str(tmp_path))
    monkeypatch.setenv("SWG_MANAGED", "data/**")
    _seed(tmp_path)
    broken = tmp_path / ".coherence" / "tracked.yaml"
    broken.parent.mkdir(mode=0o700)
    broken.write_text("- docs/**\n- *.md\n")
    server = build_server()

    async def go() -> None:
        async with lifespan(server):
            pytest.fail("attached over a policy file it cannot load")

    with pytest.raises(CoherenceError, match=r"\.coherence/tracked\.yaml is not valid YAML"):
        asyncio.run(go())

    assert broken.read_text() == "- docs/**\n- *.md\n"


def test_lifespan_refuses_a_link_in_coherence_and_names_it(tmp_path: Path, monkeypatch) -> None:
    """#498 at server startup: a linked policy file stops the server with the typed
    refusal, which names the link and the way out, and leaves the link and the file
    it names as they were. Prevents startup merging through the link (the merge
    replaced it with a regular file, so the start found no link and served) and a
    refusal a client would read as a coordinator to wait for."""
    monkeypatch.setenv("SWG_ROOT", str(tmp_path))
    monkeypatch.setenv("SWG_MANAGED", "data/**")
    _seed(tmp_path)
    coherence_dir = tmp_path / ".coherence"
    target = plant_link(coherence_dir, "tracked.yaml", "existing")
    target.write_bytes(b'- "docs/**"\n')
    server = build_server()

    async def go() -> None:
        async with lifespan(server):
            pytest.fail("attached over a link in .coherence")

    with pytest.raises(CoherenceDirRefused) as raised:
        asyncio.run(go())

    refusal = raised.value
    assert not isinstance(refusal, CoordinatorUnreachable)
    assert (refusal.kind, refusal.path) == ("coherence_link", str(tmp_path.resolve() / ".coherence" / "tracked.yaml"))
    assert "mount a directory" in str(refusal)
    assert (coherence_dir / "tracked.yaml").is_symlink()
    assert target.read_bytes() == b'- "docs/**"\n'
    assert sorted(os.listdir(coherence_dir)) == ["tracked.yaml"]
    assert stop_coordinator(tmp_path) is False


def test_the_server_refuses_a_link_in_coherence_on_one_escaped_line_and_exits_2(tmp_path: Path) -> None:
    """#498 at the console script, which an agent host runs unattended: a link in
    ``.coherence`` stops the server before it serves, on one stderr line that names
    the link with its control characters escaped, and exit 2, as the other console
    scripts answer. A cloned repository chooses the name. Prevents a traceback whose
    last line carries that name raw into the host's log or terminal, and exit 1."""
    _seed(tmp_path)
    coherence_dir = tmp_path / ".coherence"
    name = "x\n\x1b[31mIGNORE\x07"
    target = plant_link(coherence_dir, name, "dangling")

    result = subprocess.run(
        [sys.executable, "-c", "from ccs.mcp.server import main; main()"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        env=_subprocess_env(tmp_path),
        timeout=60,
    )

    assert result.returncode == 2, result.stderr
    assert result.stdout == b""
    assert not any(byte < 0x20 or byte == 0x7F for byte in result.stderr.rstrip(b"\n")), result.stderr
    assert result.stderr.count(b"\n") == 1, result.stderr
    shown = escape_nonprintable(str(tmp_path.resolve() / ".coherence" / name))
    assert result.stderr.decode("utf-8").startswith(f"stale-write-guard-fs: refusing {shown}: it is a symlink")
    assert sorted(os.listdir(coherence_dir)) == [name]
    assert not target.exists()


# --- SWG_ON_HELD: the refuse-mode opt-in --------------------------------------


@pytest.mark.parametrize(
    ("environ", "on_held"),
    [
        ({}, "displace"),
        ({"SWG_ON_HELD": ""}, "displace"),
        ({"SWG_ON_HELD": "displace"}, "displace"),
        ({"SWG_ON_HELD": "refuse"}, "refuse"),
    ],
    ids=["unset", "empty", "displace", "refuse"],
)
def test_swg_on_held_turns_refuse_mode_on_and_displace_stays_the_default(
    tmp_path: Path, environ: dict[str, str], on_held: str
) -> None:
    assert SessionConfig.from_env({"SWG_ROOT": str(tmp_path), **environ}).on_held == on_held


@pytest.mark.parametrize("value", ["Refuse", "yes", " refuse", "refuse "])
def test_swg_on_held_is_matched_exactly(tmp_path: Path, value: str) -> None:
    """Prevents a near-miss spelling starting a server whose writes displace
    holders while its operator believes they are refused."""
    with pytest.raises(ValueError, match="SWG_ON_HELD must be 'refuse' or 'displace'"):
        SessionConfig.from_env({"SWG_ROOT": str(tmp_path), "SWG_ON_HELD": value})


@pytest.mark.parametrize(
    ("value", "shown"),
    [("sometimes", "'sometimes'"), ("Refuse\n\x1b[2J", "'Refuse\\n\\x1b[2J'")],
    ids=["unknown", "control-characters"],
)
def test_the_server_refuses_another_swg_on_held_on_one_line_before_any_coordinator_starts(
    tmp_path: Path, value: str, shown: str
) -> None:
    """An agent host runs the console script unattended, so a value it cannot
    honour stops it before it serves, on one stderr line naming the accepted
    values, and exit 2, as a refused ``.coherence`` does. Prevents a traceback,
    and a coordinator started for a server that never serves."""
    _seed(tmp_path)

    result = _run_the_server(tmp_path, value)

    assert result.returncode == 2, result.stderr
    assert result.stdout == b""
    assert result.stderr.decode("utf-8") == (
        "stale-write-guard-fs: SWG_ON_HELD must be 'refuse' or 'displace' "
        f"(unset or empty means 'displace'), got {shown}\n"
    )
    assert not (tmp_path / ".coherence").exists()


def test_a_refuse_mode_start_a_coordinator_cannot_honour_ends_on_one_line_and_exit_2(tmp_path: Path) -> None:
    """``SWG_ON_HELD=refuse`` against a coordinator that does not advertise
    ``pre_edit_if_unheld`` stops the server before it serves: one stderr line
    that carries the volume's refusal unchanged, names ``SWG_ON_HELD`` and the
    fix, and exit 2. Prevents a server that silently displaces holders, and a
    traceback in the host's log."""
    _seed(tmp_path)

    result = _run_the_server(tmp_path, "refuse", _MAIN_WITHOUT_THE_REFUSAL)

    assert result.returncode == 2, result.stderr
    assert result.stdout == b""
    *logged, refusal = result.stderr.decode("utf-8").split("\n")[:-1]
    assert refusal == (
        f"stale-write-guard-fs: refusing to start with SWG_ON_HELD=refuse: {_NOT_ADVERTISED} "
        "Run a coordinator that can refuse (agent-coherence after 0.15.0, or the Claude Code "
        "plugin after 0.6.0), or unset SWG_ON_HELD to serve in displace mode."
    )
    # The attach spawned the coordinator before it read what the coordinator
    # advertises, and the coordinator logs its start as it does on every start.
    # Nothing else is printed: no traceback, and no error record.
    assert all(re.match(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3} INFO ccs\.adapters\.", line) for line in logged), logged


@pytest.mark.parametrize("on_held", [None, "", "displace"], ids=["unset", "empty", "displace"])
def test_a_displace_mode_write_sends_only_the_requests_it_always_sent(
    tmp_path: Path, monkeypatch, wire: _Wire, on_held: str | None
) -> None:
    """Without refuse mode, ``swg_write`` asks the coordinator what it asked
    before the option existed: a pre-edit with no ``if_unheld``, then the
    post-edit, and nothing else. Prevents a server started without the option
    opting its writes in, or paying an extra request for the option."""
    monkeypatch.setenv("SWG_ROOT", str(tmp_path))
    monkeypatch.setenv("SWG_MANAGED", "data/**")
    if on_held is None:
        monkeypatch.delenv("SWG_ON_HELD", raising=False)
    else:
        monkeypatch.setenv("SWG_ON_HELD", on_held)
    _seed(tmp_path)

    async def go() -> tuple[dict, list[str], dict]:
        async with lifespan(build_server()) as ctx:
            wire.sent.clear()
            result = server_module._do_write(ctx.volume, ctx.config, "data/shared.txt", "v2")
            return result.structuredContent, wire.routes(), wire.bodies(_PRE_EDIT)[0]

    structured, routes, pre_edit = asyncio.run(go())

    assert structured == {"ok": True, "path": "data/shared.txt"}
    assert routes == [_PRE_EDIT, _POST_EDIT]
    assert sorted(pre_edit) == ["agent_id", "path", "session_id"]
    assert (tmp_path / "data/shared.txt").read_bytes() == b"v2"


# --- strict-only construction (no degrade-mode volume) -----------------------


def test_build_volume_is_strict_only(tmp_path: Path) -> None:
    config = SessionConfig(root=tmp_path, managed=("data/**",))
    _seed(tmp_path)
    volume = build_volume(config)
    try:
        assert volume.is_attached
        # The server never constructs a degrade-mode volume (the fail-open hole).
        assert volume._on_error == "strict"
        assert volume.strict_mode_active() is True
        # The write-surface guard is armed too (foreign-edit clobber -> stale_view).
        assert volume._on_stale_write == "raise"
    finally:
        stop_coordinator(tmp_path)


def test_build_volume_on_a_refuse_config_builds_a_refuse_mode_volume(tmp_path: Path) -> None:
    """Refuse mode adds to the strict-only volume and takes nothing from it."""
    config = SessionConfig(root=tmp_path, managed=("data/**",), on_held="refuse")
    _seed(tmp_path)
    volume = build_volume(config)
    try:
        assert volume.asks_if_unheld("data/shared.txt") is True
        assert volume.asks_if_unheld("outside.txt") is False
        assert volume._on_error == "strict"
        assert volume._on_stale_write == "raise"
    finally:
        stop_coordinator(tmp_path)


# --- serialization: concurrent coroutines do not trip the A5 thread-guard -----


def test_concurrent_volume_access_serializes_without_a5(tmp_path: Path, monkeypatch) -> None:
    """Three overlapping coroutines each touch the one shared volume under the
    lock. The volume's A5 guard raises on a *different thread*; serialized
    main-thread access under the lock must never trip it."""
    monkeypatch.setenv("SWG_ROOT", str(tmp_path))
    monkeypatch.setenv("SWG_MANAGED", "data/**")
    _seed(tmp_path, content=b"v1")
    server = build_server()

    async def go() -> list[bytes]:
        async with lifespan(server) as ctx:

            async def op() -> bytes:
                async with ctx.lock:
                    return ctx.volume.read("data/shared.txt")

            return await asyncio.gather(op(), op(), op())

    results = asyncio.run(go())
    assert results == [b"v1", b"v1", b"v1"]
    assert stop_coordinator(tmp_path) is False


# --- shutdown closes the volume ---------------------------------------------


def _write_then_shut_down(tmp_path: Path, monkeypatch) -> str:
    """Run the lifespan against a coordinator another process could own (one in
    this process, which ``stop_coordinator`` leaves running), write a managed
    file through the server's volume, and shut down. Returns the volume's
    agent id."""
    monkeypatch.setenv("SWG_ROOT", str(tmp_path))
    monkeypatch.setenv("SWG_MANAGED", "data/**")
    server = build_server()

    async def go() -> str:
        async with lifespan(server) as ctx:
            async with ctx.lock:
                ctx.volume.write("data/shared.txt", b"from the server")
            return ctx.volume.agent_id

    return asyncio.run(go())


def test_shutting_down_the_server_gives_back_its_write_grants(tmp_path: Path, monkeypatch) -> None:
    """A normal shutdown closes the server's volume before the coordinator is
    stopped, so a refuse-mode peer is admitted at once instead of after the
    coordinator's sweep.

    Prevents every file the server wrote staying held after it exits."""
    (tmp_path / "data").mkdir()
    coordinator = serve_in_process(tmp_path, "mcp-shutdown")
    try:
        writer = _write_then_shut_down(tmp_path, monkeypatch)
        assert write_holders(coordinator, "data/shared.txt") == set(), f"{writer} still holds the file"
        peer = CoherentVolume(tmp_path, managed=("data/**",), on_held="refuse")
        peer.write("data/shared.txt", b"from the peer")
        assert write_holders(coordinator, "data/shared.txt") == {peer.agent_id}
    finally:
        coordinator.shutdown()


def test_a_close_that_fails_at_shutdown_is_logged_and_the_coordinator_still_stops(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    """Prevents a coordinator outage at shutdown leaving a coordinator this
    server started running."""
    (tmp_path / "data").mkdir()
    coordinator = serve_in_process(tmp_path, "mcp-shutdown-fails")
    real_post = coherent_volume_module._coordinator_post

    def no_answer_to_the_stop(endpoint, path, payload, **kwargs):
        if path == "/hooks/session-stop":
            raise CoordinatorUnavailable("scripted: the answer was lost")
        return real_post(endpoint, path, payload, **kwargs)

    stopped: list[Path] = []
    real_stop = server_module.stop_coordinator
    monkeypatch.setattr(coherent_volume_module, "_coordinator_post", no_answer_to_the_stop)
    monkeypatch.setattr(server_module, "stop_coordinator", lambda root: stopped.append(root) or real_stop(root))
    try:
        with caplog.at_level("WARNING", logger="ccs.mcp.server"):
            _write_then_shut_down(tmp_path, monkeypatch)
        assert stopped == [tmp_path.resolve()]
        assert any("close" in record.getMessage() for record in caplog.records)
    finally:
        coordinator.shutdown()


def test_shutdown_closes_the_volume_before_it_stops_the_coordinator_whatever_the_close_raises(
    tmp_path: Path, monkeypatch
) -> None:
    """An in-process coordinator must still answer the close, so the close
    comes first; and an error the close does not expect still lets the
    coordinator stop.

    Prevents a close sent to a stopped coordinator, and a coordinator this
    server started left running after an unexpected close error."""
    (tmp_path / "data").mkdir()
    coordinator = serve_in_process(tmp_path, "mcp-shutdown-order")
    order: list[str] = []

    def close_then_fail(volume: CoherentVolume) -> None:
        order.append("close")
        raise RuntimeError("unexpected close failure")

    real_stop = server_module.stop_coordinator
    monkeypatch.setattr(coherent_volume_module.CoherentVolume, "close", close_then_fail)
    monkeypatch.setattr(server_module, "stop_coordinator", lambda root: order.append("stop") or real_stop(root))
    try:
        with pytest.raises(RuntimeError, match="unexpected close failure"):
            _write_then_shut_down(tmp_path, monkeypatch)
        assert order == ["close", "stop"]
    finally:
        coordinator.shutdown()


# --- stdout purity (fd-level, via a real subprocess) -------------------------


def test_lifespan_keeps_stdout_byte_clean(tmp_path: Path) -> None:
    """Enter the lifespan (spawns a real coordinator) and exit, in a subprocess
    whose REAL fd 1 we capture. Nothing — not the server, not the spawned
    coordinator subprocess — may write to stdout (the JSON-RPC channel)."""
    _seed(tmp_path)
    script = textwrap.dedent(
        """
        import asyncio
        from ccs.mcp.server import build_server, lifespan

        async def go():
            async with lifespan(build_server()) as ctx:
                assert ctx.volume.is_attached

        asyncio.run(go())
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=_subprocess_env(tmp_path),
        timeout=60,
    )
    assert result.returncode == 0, f"subprocess failed:\n{result.stderr}"
    assert result.stdout == "", f"stdout polluted (JSON-RPC channel): {result.stdout!r}"


# --- lazy package: import ccs.mcp without the mcp SDK ------------------------


def test_import_ccs_mcp_does_not_require_mcp_sdk() -> None:
    """``import ccs.mcp`` must succeed even when the optional ``mcp`` SDK is
    unimportable — the package is lazy (only ``deny``/``server`` pull it in)."""
    script = textwrap.dedent(
        """
        import sys
        import importlib.abc

        class _BlockMcp(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name == "mcp" or name.startswith("mcp."):
                    raise ImportError("mcp SDK blocked for the lazy-import test")
                return None

        sys.meta_path.insert(0, _BlockMcp())
        import ccs.mcp  # must NOT import mcp
        assert "mcp" not in sys.modules, "ccs.mcp eagerly imported the mcp SDK"
        print("ok")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        timeout=30,
    )
    assert result.returncode == 0, f"lazy import failed:\n{result.stderr}"
    assert "ok" in result.stdout
