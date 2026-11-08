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
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import ccs.adapters.coherent_volume as coherent_volume_module
import ccs.mcp.server as server_module
from ccs.adapters.claude_code.lifecycle import LifecycleConfig, stop_coordinator
from ccs.adapters.coherent_volume import CoherentVolume
from ccs.cli._coherence_client import CoordinatorUnavailable, escape_nonprintable
from ccs.core.exceptions import CoherenceDirRefused, CoherenceError, CoordinatorUnreachable
from ccs.mcp.server import build_server, lifespan
from ccs.mcp.session import SessionConfig, build_volume
from tests.coherence_link_helpers import plant_link
from tests.coordinator_helpers import serve_in_process, write_holders

REPO_ROOT = Path(__file__).resolve().parents[2]


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
