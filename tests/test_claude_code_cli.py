# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Tests for the four agent-coherence-* console scripts (Unit 6).

Covers happy paths, argparse + path validation, graceful-no-coordinator
behavior, and HTTP error propagation. End-to-end smoke (real spawned
coordinator) is exercised by the lifecycle tests; here we mostly hit the
control-flow / output-rendering paths.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from ccs.adapters.claude_code.lifecycle import LifecycleConfig, ensure_coordinator, stop_coordinator
from ccs.cli import (
    coherence_coordinator,
    coherence_status,
    coherence_track,
    coherence_untrack,
)
from ccs.cli._coherence_client import (
    CoordinatorUnavailable,
    resolve_endpoint,
)


@pytest.fixture
def fast_cfg() -> LifecycleConfig:
    return LifecycleConfig(
        idle_shutdown_sec=0,
        sweep_interval_sec=0,
        port_file_retry_attempts=10,
        port_file_retry_interval_sec=0.05,
        connect_retry_attempts=10,
        connect_retry_interval_sec=0.05,
        spawn_self_probe_attempts=20,
    )


@pytest.fixture
def git_workspace(tmp_path: Path) -> Path:
    """A tmp_path with a minimal .git/ marker so find_coordinator_root resolves it."""
    (tmp_path / ".git").mkdir()
    return tmp_path


@pytest.fixture
def live_coordinator(git_workspace: Path, fast_cfg: LifecycleConfig):
    """Spawn a real coordinator and yield (workspace, port)."""
    port = ensure_coordinator(git_workspace, config=fast_cfg)
    assert port > 0
    yield git_workspace, port
    stop_coordinator(git_workspace)


# ----------------------------------------------------------------------
# coherence_coordinator
# ----------------------------------------------------------------------


def test_coordinator_not_in_git_repo_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Edge case: no .git/ ancestor → exit 1 with a clear message.

    Uses ``monkeypatch.chdir`` (not raw ``os.chdir``) so the working
    directory is restored on test teardown — otherwise the leak breaks
    every subsequent test that relies on relative paths or tmp_path.
    """
    no_git = tmp_path / "no_git"
    no_git.mkdir()
    monkeypatch.chdir(no_git)
    rc = coherence_coordinator.main([])
    captured = capsys.readouterr()
    assert rc == 1
    # ce-review P2 fix #15: errors now go to stderr (was stdout)
    assert "not in a git repository" in captured.err


def test_coordinator_spawns_and_prints_port(
    git_workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Happy path: agent-coherence-coordinator --root <git> --no-detach
    → exits 0, prints 'port=NNNN'. Uses --no-detach to keep the
    coordinator in-process so the test's stop_coordinator can find it."""
    try:
        rc = coherence_coordinator.main([
            "--root", str(git_workspace), "--no-detach",
        ])
        captured = capsys.readouterr()
        assert rc == 0
        assert captured.out.startswith("port=")
        port = int(captured.out.strip().split("=")[1])
        assert 1024 <= port <= 65535
    finally:
        stop_coordinator(git_workspace)


def test_coordinator_quiet_flag_suppresses_port_line(
    git_workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--quiet suppresses stdout but still exits 0."""
    try:
        rc = coherence_coordinator.main([
            "--root", str(git_workspace), "--quiet", "--no-detach",
        ])
        captured = capsys.readouterr()
        assert rc == 0
        assert captured.out == ""
    finally:
        stop_coordinator(git_workspace)


def test_coordinator_detached_spawn_survives_parent_exit(
    git_workspace: Path,
) -> None:
    """The load-bearing smoke-finding regression: a real detached subprocess
    must keep the coordinator alive after the launching CLI exits, so a
    subsequent agent-coherence-status invocation can reach it. This was
    broken in the initial Unit 6 implementation — coordinator was a
    daemon thread that died with its parent."""
    import subprocess
    import sys

    # Run the real CLI as a subprocess (not in-process) — replicates
    # what a user invocation does.
    proc = subprocess.run(
        [sys.executable, "-m", "ccs.cli.coherence_coordinator",
         "--root", str(git_workspace)],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert proc.stdout.startswith("port="), proc.stdout
    port = int(proc.stdout.strip().split("=")[1])

    # Critical assertion: the port must be reachable AFTER the launching
    # process has exited. This is the regression case.
    import socket as _s
    sock = _s.socket(_s.AF_INET, _s.SOCK_STREAM)
    sock.settimeout(1.0)
    try:
        sock.connect(("127.0.0.1", port))
        sock.close()
        reachable = True
    except OSError as exc:
        reachable = False
        pytest.fail(
            f"detached coordinator at port {port} not reachable after "
            f"parent exit (errno={exc.errno})"
        )

    # Clean up the detached process by reading its pid + killing it.
    pid_file = git_workspace / ".coherence" / "server.pid"
    if pid_file.exists():
        lines = pid_file.read_text().splitlines()
        if lines:
            try:
                pid = int(lines[0])
                os.kill(pid, 15)  # SIGTERM — daemon exits cleanly
            except (ValueError, ProcessLookupError):
                pass


def test_coordinator_second_invocation_reuses_existing(
    git_workspace: Path,
) -> None:
    """Once a coordinator is live, a second `agent-coherence-coordinator`
    invocation must short-circuit to its port without re-forking. The
    existing-coordinator probe in main() does the TCP check."""
    import subprocess
    import sys

    # First invocation — detached spawn.
    proc1 = subprocess.run(
        [sys.executable, "-m", "ccs.cli.coherence_coordinator",
         "--root", str(git_workspace)],
        capture_output=True, text=True, timeout=30,
    )
    assert proc1.returncode == 0
    port_1 = int(proc1.stdout.strip().split("=")[1])

    try:
        # Second invocation — must return same port via short-circuit.
        proc2 = subprocess.run(
            [sys.executable, "-m", "ccs.cli.coherence_coordinator",
             "--root", str(git_workspace)],
            capture_output=True, text=True, timeout=10,
        )
        assert proc2.returncode == 0
        port_2 = int(proc2.stdout.strip().split("=")[1])
        assert port_2 == port_1, "second invocation must reuse the live coordinator's port"
    finally:
        # Clean up
        pid_file = git_workspace / ".coherence" / "server.pid"
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().splitlines()[0])
                os.kill(pid, 15)
            except (ValueError, ProcessLookupError, IndexError):
                pass


def _capture_probe(seen: dict, *, reachable: bool):
    """Stub for coherence_coordinator._tcp_probe that records bind_host."""
    def probe(port: int, cfg: LifecycleConfig, *, bind_host: str = "127.0.0.1") -> bool:
        seen["bind_host"] = bind_host
        return reachable
    return probe


def _write_port_file(workspace: Path, port: int) -> Path:
    pid_file = workspace / ".coherence" / "server.pid"
    pid_file.parent.mkdir(exist_ok=True)
    pid_file.write_text(f"999\n{port}\n", encoding="utf-8")
    return pid_file


def _write_coherence_files(workspace: Path, port: int) -> Path:
    """pid file + hook.secret, enough for the real resolve_endpoint to succeed."""
    pid_file = _write_port_file(workspace, port)
    (workspace / ".coherence" / "hook.secret").write_text("test-token\n", encoding="utf-8")
    return pid_file


def test_coordinator_reuse_probe_targets_routed_bind_host(
    git_workspace: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Routed-bind regression: the existing-coordinator probe in main()
    must poll the --bind-host address, not loopback. Before the fix it
    dropped the kwarg, so a coordinator bound to 10.0.0.5 looked dead
    from 127.0.0.1 and every non-loopback invocation re-forked."""
    monkeypatch.setenv("CCS_REMOTE_COORDINATOR", "1")
    _write_port_file(git_workspace, 54321)
    seen: dict = {}
    monkeypatch.setattr(
        coherence_coordinator, "_tcp_probe", _capture_probe(seen, reachable=True)
    )

    rc = coherence_coordinator.main([
        "--root", str(git_workspace), "--bind-host", "10.0.0.5", "--quiet",
    ])

    assert rc == 0
    assert seen["bind_host"] == "10.0.0.5"


def test_spawn_detached_probe_targets_routed_bind_host(
    git_workspace: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Routed-bind regression: _spawn_detached's port-file poll must probe
    the child's bind host. Before the fix it polled 127.0.0.1, collected
    ECONNREFUSED for the full 10s window, exited 2 — and left the live
    daemon orphaned behind the port file."""
    _write_port_file(git_workspace, 54321)
    seen: dict = {}
    monkeypatch.setattr(
        coherence_coordinator, "_tcp_probe", _capture_probe(seen, reachable=True)
    )
    monkeypatch.setattr(
        coherence_coordinator.subprocess, "Popen",
        lambda *a, **kw: None,  # the pre-written port file stands in for the child
    )

    rc = coherence_coordinator._spawn_detached(
        git_workspace, quiet=True, bind_host="10.0.0.5",
    )

    assert rc == 0
    assert seen["bind_host"] == "10.0.0.5"


def test_spawn_detached_routed_bind_reports_loopback_conflict(
    git_workspace: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When a loopback-bound coordinator already holds the workspace, a
    routed spawn can never converge: the child joins the existing
    coordinator via the flock loser path and no routed listener appears.
    The detach wait must detect the mismatch and exit 2 immediately with
    a message naming it — not stall through the whole window with a
    generic 'did not become reachable'."""
    _write_port_file(git_workspace, 54321)

    def split_probe(port, cfg, *, bind_host="127.0.0.1"):
        return bind_host == "127.0.0.1"  # loopback answers, routed host doesn't

    monkeypatch.setattr(coherence_coordinator, "_tcp_probe", split_probe)
    monkeypatch.setattr(
        coherence_coordinator.subprocess, "Popen", lambda *a, **kw: None,
    )

    start = time.monotonic()
    rc = coherence_coordinator._spawn_detached(
        git_workspace, quiet=True, bind_host="10.0.0.5",
    )
    elapsed = time.monotonic() - start

    captured = capsys.readouterr()
    assert rc == 2
    assert "already running" in captured.err
    assert "127.0.0.1" in captured.err
    assert elapsed < 2.0, f"conflict detection took {elapsed:.1f}s — stalled through the window"


def test_spawn_detached_wait_is_wall_clock_bounded(
    git_workspace: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The detach wait must be a wall-clock deadline, not an attempt count.
    Attempt counting multiplies with the probe's internal retry budget: a
    SYN-dropping routed host makes each probe slow, inflating the '10s'
    window to minutes inside a SessionStart hook. A slow probe here must
    still hit the deadline on schedule."""
    _write_port_file(git_workspace, 54321)
    monkeypatch.setattr(coherence_coordinator, "_DETACH_PORT_WAIT_TIMEOUT_SEC", 0.5)

    def slow_dead_probe(port, cfg, *, bind_host="127.0.0.1"):
        time.sleep(0.2)  # stand-in for the probe's own retry budget
        return False

    monkeypatch.setattr(coherence_coordinator, "_tcp_probe", slow_dead_probe)
    monkeypatch.setattr(
        coherence_coordinator.subprocess, "Popen", lambda *a, **kw: None,
    )

    start = time.monotonic()
    rc = coherence_coordinator._spawn_detached(
        git_workspace, quiet=True, bind_host="127.0.0.1",
    )
    elapsed = time.monotonic() - start

    assert rc == 2
    assert "did not become reachable" in capsys.readouterr().err
    assert elapsed < 2.0, (
        f"wait ran {elapsed:.1f}s — attempt-counting, not the 0.5s wall-clock deadline"
    )


def test_wildcard_bind_host_rejected_before_any_dial(
    git_workspace: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--bind-host 0.0.0.0 is documented as rejected, but validation used
    to run only inside the forked child: the parent dialed the raw string,
    where 0.0.0.0 aliases to localhost and laundered to exit 0. The CLI
    entry must reject it before any probe dials."""
    monkeypatch.setenv("CCS_REMOTE_COORDINATOR", "1")
    _write_port_file(git_workspace, 54321)

    def must_not_dial(port, cfg, *, bind_host="127.0.0.1"):
        pytest.fail(f"probe dialed {bind_host!r} before bind-host validation")

    monkeypatch.setattr(coherence_coordinator, "_tcp_probe", must_not_dial)

    rc = coherence_coordinator.main([
        "--root", str(git_workspace), "--bind-host", "0.0.0.0",
    ])

    captured = capsys.readouterr()
    assert rc == 2
    assert "wildcard" in captured.err


def test_prepare_for_migration_targets_routed_bind_host(
    git_workspace: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Routed-bind regression: --prepare-for-migration must aim BOTH its
    release POST and its shutdown poll at the --bind-host address. The
    local resolver only reads port + secret files and always reports
    loopback, so the CLI has to rebuild the endpoint from the flag —
    exercised here through the real resolve_endpoint, not a fabricated
    CoordinatorEndpoint."""
    import ccs.cli._coherence_client as client_mod

    monkeypatch.setenv("CCS_REMOTE_COORDINATOR", "1")
    _write_coherence_files(git_workspace, 54321)
    posted: dict = {}

    def fake_post(endpoint, path, payload, **kw):
        posted["host"] = endpoint.host
        return {"ok": True, "released": 1, "errors": []}

    monkeypatch.setattr(client_mod, "post", fake_post)
    seen: dict = {}
    monkeypatch.setattr(
        coherence_coordinator, "_tcp_probe", _capture_probe(seen, reachable=False)
    )

    rc = coherence_coordinator.main([
        "--root", str(git_workspace), "--prepare-for-migration",
        "--bind-host", "10.0.0.5",
    ])

    assert rc == 0, capsys.readouterr()
    assert posted["host"] == "10.0.0.5"
    assert seen["bind_host"] == "10.0.0.5"


def test_prepare_for_migration_default_stays_loopback(
    git_workspace: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Without --bind-host the migration path must be byte-unchanged:
    POST and poll both target the resolver's loopback endpoint."""
    import ccs.cli._coherence_client as client_mod

    _write_coherence_files(git_workspace, 54321)
    posted: dict = {}

    def fake_post(endpoint, path, payload, **kw):
        posted["host"] = endpoint.host
        return {"ok": True, "released": 0, "errors": []}

    monkeypatch.setattr(client_mod, "post", fake_post)
    seen: dict = {}
    monkeypatch.setattr(
        coherence_coordinator, "_tcp_probe", _capture_probe(seen, reachable=False)
    )

    rc = coherence_coordinator.main([
        "--root", str(git_workspace), "--prepare-for-migration",
    ])

    assert rc == 0, capsys.readouterr()
    assert posted["host"] == "127.0.0.1"
    assert seen["bind_host"] == "127.0.0.1"


def test_prepare_for_migration_unreachable_post_exits_2_cleanly(
    git_workspace: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A connection failure at the release POST must map to the documented
    exit 2 with an operator message — not escape as a CoordinatorUnavailable
    traceback (interpreter exit 1, grants silently left live)."""
    import ccs.cli._coherence_client as client_mod
    from ccs.cli._coherence_client import CoordinatorUnavailable

    _write_coherence_files(git_workspace, 54321)

    def refused_post(endpoint, path, payload, **kw):
        raise CoordinatorUnavailable("could not reach coordinator at http://127.0.0.1:54321")

    monkeypatch.setattr(client_mod, "post", refused_post)

    rc = coherence_coordinator.main([
        "--root", str(git_workspace), "--prepare-for-migration",
    ])

    captured = capsys.readouterr()
    assert rc == 2
    assert "could not reach coordinator" in captured.err


# ----------------------------------------------------------------------
# coherence_status
# ----------------------------------------------------------------------


def test_status_no_coordinator_exits_0_with_graceful_message(
    git_workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Edge case: no coordinator running → exit 0 (NOT an error)."""
    rc = coherence_status.main(["--root", str(git_workspace)])
    captured = capsys.readouterr()
    assert rc == 0
    # ce-review P2 fix #15: graceful-no-coordinator message now goes to stderr
    assert "no coordinator running" in captured.err


def test_status_renders_table_against_live_coordinator(
    live_coordinator, capsys: pytest.CaptureFixture[str]
) -> None:
    """Happy path: agent-coherence-status against a live coordinator → table."""
    workspace, port = live_coordinator
    rc = coherence_status.main(["--root", str(workspace)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "Coordinator:" in captured.out
    assert f"pid={os.getpid()}" in captured.out
    # Status now shows policy state and disambiguates empty-registry from empty-policy.
    assert "Policy:" in captured.out
    assert (
        "No artifacts observed yet" in captured.out
        or "Observed artifacts:" in captured.out
        or "No tracked artifacts (policy is empty)" in captured.out
    )


def test_render_table_keeps_version_on_same_line_as_long_path(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Regression: a long tracked path used to pad every row past the screen
    edge, soft-wrapping the version onto its own line ("number below the
    filename"). Each row must now fit the terminal — long paths middle-elide,
    the version stays inline — and a legend must explain the version column."""
    monkeypatch.setenv("COLUMNS", "80")
    long_path = (
        ".claude/worktrees/feat_crash_recovery/docs/plans/"
        "2026-05-28-001-feat-c-flip-crash-recovery-default-on-plan.md"
    )
    payload = {
        "tracked_artifacts": [
            {"path": "plan.md", "version": 2},
            {"path": long_path, "version": 13},
        ],
        "sessions": [],
        "policy_summary": {},
        "coordinator_pid": 0,
    }
    coherence_status._render_table(payload)
    out = capsys.readouterr().out

    # Legend describes what the version column means.
    assert "version = artifact revision" in out
    assert "committed edit" in out

    lines = out.splitlines()
    # No row exceeds the terminal width → nothing soft-wraps.
    assert all(len(line) <= 80 for line in lines), [l for l in lines if len(l) > 80]

    # The long path is middle-elided but its filename tail survives, and its
    # version sits on the SAME line (not orphaned below it).
    long_row = next(line for line in lines if line.rstrip().endswith("13"))
    assert "…" in long_row
    assert "plan.md" in long_row

    # The short path keeps its version inline too.
    short_row = next(
        line for line in lines
        if line.strip().startswith("plan.md") and line.rstrip().endswith("2")
    )
    assert short_row  # found


def test_status_json_mode_emits_raw_payload(
    live_coordinator, capsys: pytest.CaptureFixture[str]
) -> None:
    """--json mode prints the raw response so external tools can parse it."""
    workspace, port = live_coordinator
    rc = coherence_status.main(["--root", str(workspace), "--json"])
    captured = capsys.readouterr()
    assert rc == 0
    data = json.loads(captured.out)
    assert "tracked_artifacts" in data
    assert "sessions" in data
    assert data["coordinator_pid"] == os.getpid()


def test_status_detail_metrics_renders_counter_block(
    live_coordinator, capsys: pytest.CaptureFixture[str]
) -> None:
    """KTD-J (Unit 8): --detail metrics returns the counter block only —
    no artifact/session walk in the output."""
    workspace, port = live_coordinator
    rc = coherence_status.main([
        "--root", str(workspace), "--detail", "metrics",
    ])
    captured = capsys.readouterr()
    assert rc == 0
    assert "Coordinator metrics:" in captured.out
    assert "backend=python" in captured.out
    # Endpoint counter block must be present (zero-valued is fine on a
    # fresh coordinator with no hook traffic yet).
    assert "Counters:" in captured.out
    assert "pre_read_total" in captured.out
    assert "intra_task_acquire_release_total" in captured.out
    # No artifact/session block in metrics mode.
    assert "Observed artifacts" not in captured.out
    assert "Sessions:" not in captured.out


def test_status_detail_minimal_includes_pid_redacts_abs_root(
    live_coordinator, capsys: pytest.CaptureFixture[str]
) -> None:
    """--detail minimal shows coordinator_pid (P1 #7: pid is public on
    POSIX and operators rely on it). The absolute workspace root stays
    sentinel'd to ``.`` so $HOME / directory layout never leaks at this
    tier."""
    workspace, port = live_coordinator
    rc = coherence_status.main([
        "--root", str(workspace), "--detail", "minimal",
    ])
    captured = capsys.readouterr()
    assert rc == 0
    assert "Coordinator:" in captured.out
    # P1 #7: pid IS in the minimal tier header.
    assert f"pid={os.getpid()}" in captured.out
    # Absolute root must NOT leak at this tier.
    assert str(workspace) not in captured.out


def test_status_full_default_includes_counters_below_sessions(
    live_coordinator, capsys: pytest.CaptureFixture[str]
) -> None:
    """The default --detail=full table rendering now includes a Counters
    section after the artifacts/sessions block."""
    workspace, port = live_coordinator
    rc = coherence_status.main(["--root", str(workspace)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "Counters:" in captured.out
    assert "pre_read_total" in captured.out


def test_self_test_passes_against_live_coordinator(
    live_coordinator, capsys: pytest.CaptureFixture[str]
) -> None:
    """KTD-J --self-test smoke: full pre-read → pre-edit → post-edit →
    stale pre-read chain must report OK against a healthy coordinator."""
    workspace, port = live_coordinator
    rc = coherence_status.main(["--root", str(workspace), "--self-test"])
    captured = capsys.readouterr()
    assert rc == 0, (
        f"--self-test failed unexpectedly: stdout={captured.out!r} "
        f"stderr={captured.err!r}"
    )
    assert "OK" in captured.out
    assert "pre-read STALE" in captured.out


def test_self_test_returns_3_when_no_coordinator_running(
    git_workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """If the coordinator isn't running, --self-test exits 3 with an
    actionable diagnostic — operators can distinguish 'no coordinator'
    from 'coordinator broken'."""
    rc = coherence_status.main([
        "--root", str(git_workspace), "--self-test",
    ])
    captured = capsys.readouterr()
    assert rc == 3
    assert "coordinator unreachable" in captured.err or "coordinator" in captured.err


# ----------------------------------------------------------------------
# Unit 8 — agent-coherence-coordinator --prepare-for-migration
# ----------------------------------------------------------------------


def test_prepare_for_migration_no_coordinator_running_is_noop(
    git_workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Idempotent: --prepare-for-migration on a workspace with no
    .coherence/server.pid is a no-op exit 0 so it's safe to script
    as a pre-switch step that may not always find a live coordinator."""
    rc = coherence_coordinator.main([
        "--root", str(git_workspace), "--prepare-for-migration",
    ])
    captured = capsys.readouterr()
    assert rc == 0
    assert "no coordinator running" in captured.out


def test_prepare_for_migration_releases_grants_and_shuts_down(
    live_coordinator, capsys: pytest.CaptureFixture[str]
) -> None:
    """Decision 1 contract: with an EXCLUSIVE grant held by some agent
    on some tracked artifact, --prepare-for-migration releases the
    grant (state transitions away from M/E) and the coordinator's HTTP
    server is no longer reachable when the command returns."""
    import socket
    workspace, port = live_coordinator

    # Set up an EXCLUSIVE grant via the real HTTP path so registry
    # state matches what a real client would have written.
    from ccs.cli._coherence_client import post as _post
    from ccs.cli._coherence_client import resolve_endpoint
    endpoint = resolve_endpoint(workspace)
    sid = "11111111-2222-4111-8111-aaaaaaaaaaaa"
    _post(endpoint, "/hooks/pre-edit", {"session_id": sid, "path": "plan.md"})

    # Now drive the migration helper.
    rc = coherence_coordinator.main([
        "--root", str(workspace), "--prepare-for-migration",
    ])
    captured = capsys.readouterr()
    assert rc == 0, (
        f"prepare-for-migration failed: stdout={captured.out!r} "
        f"stderr={captured.err!r}"
    )
    # Output should report at least one released grant.
    assert "released" in captured.out

    # Coordinator must no longer be TCP-reachable.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.5)
    try:
        sock.connect(("127.0.0.1", port))
        sock.close()
        pytest.fail(
            f"coordinator at port={port} still accepting connections after "
            "--prepare-for-migration"
        )
    except OSError:
        pass  # expected — connection refused after shutdown


# ----------------------------------------------------------------------
# coherence_track + coherence_untrack — validation + happy path
# ----------------------------------------------------------------------


@pytest.mark.parametrize("bad_path,reason_substr", [
    # /etc/passwd is absolute AND outside the workspace — error message
    # changed 2026-05-26 from "must be relative" to "outside workspace root"
    # because absolute paths INSIDE the workspace are now auto-normalized
    # (operator-UX fix: skill template passes absolute paths verbatim).
    ("/etc/passwd", "outside workspace root"),
    ("../../../etc/passwd", "'..'"),
    ("", "empty"),
])
def test_track_rejects_invalid_paths_without_network(
    git_workspace: Path, capsys: pytest.CaptureFixture[str],
    bad_path: str, reason_substr: str,
) -> None:
    """Pre-validation: invalid paths exit 1 without a network round-trip."""
    rc = coherence_track.main(["--root", str(git_workspace), bad_path])
    captured = capsys.readouterr()
    assert rc == 1
    # ce-review P2 fix #15: rejection messages go to stderr
    assert reason_substr in captured.err


def test_track_accepts_absolute_path_inside_workspace(
    live_coordinator, capsys: pytest.CaptureFixture[str]
) -> None:
    """Operator-UX fix 2026-05-26: skill template passes absolute paths verbatim
    (`/agent-coherence:track /abs/path/file.md`); CLI must auto-normalize to
    workspace-relative form before validation + before send-to-coordinator.
    Tracked.yaml must contain the WORKSPACE-RELATIVE form, never the absolute
    path — otherwise tracked.yaml drifts across machines / worktrees."""
    workspace, port = live_coordinator
    target = workspace / "docs" / "plan.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("plan v1")

    rc = coherence_track.main(["--root", str(workspace), str(target)])
    captured = capsys.readouterr()
    assert rc == 0, f"expected 0, got {rc}; stderr: {captured.err}"
    # Success message uses the workspace-relative form (operator sees clean output)
    assert "tracked docs/plan.md" in captured.out
    # tracked.yaml contains workspace-relative — NO absolute path leak
    tracked_yaml = workspace / ".coherence" / "tracked.yaml"
    content = tracked_yaml.read_text()
    assert "docs/plan.md" in content
    assert str(target) not in content, (
        "absolute path leaked into tracked.yaml — file is now machine-specific. "
        f"content: {content!r}"
    )


def test_untrack_accepts_absolute_path_inside_workspace(
    live_coordinator, capsys: pytest.CaptureFixture[str]
) -> None:
    """Same defense-in-depth fix for the sibling untrack CLI."""
    workspace, port = live_coordinator
    target = workspace / "docs" / "draft.md"

    rc = coherence_untrack.main(["--root", str(workspace), str(target)])
    captured = capsys.readouterr()
    assert rc == 0, f"expected 0, got {rc}; stderr: {captured.err}"
    assert "untracked docs/draft.md" in captured.out
    ignored_yaml = workspace / ".coherence" / "ignored.yaml"
    content = ignored_yaml.read_text()
    assert "docs/draft.md" in content
    assert str(target) not in content


def test_track_no_coordinator_running_exits_2(
    git_workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Valid path but coordinator down → exit 2."""
    rc = coherence_track.main(["--root", str(git_workspace), "docs/plan.md"])
    captured = capsys.readouterr()
    assert rc == 2
    # ce-review P2 fix #15: coordinator-unavailable message goes to stderr
    assert "no coordinator running" in captured.err


def test_track_against_live_coordinator(
    live_coordinator, capsys: pytest.CaptureFixture[str]
) -> None:
    """Happy path: track a valid path → exit 0, /policy/track returns added."""
    workspace, port = live_coordinator
    # Create the file so the "does not exist on disk yet" warning doesn't fire
    target = workspace / "docs" / "plan.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("plan v1")

    rc = coherence_track.main(["--root", str(workspace), "docs/plan.md"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "tracked docs/plan.md" in captured.out
    # tracked.yaml should now exist with the path
    tracked_yaml = workspace / ".coherence" / "tracked.yaml"
    assert tracked_yaml.is_file()
    assert "docs/plan.md" in tracked_yaml.read_text()


def test_track_warns_on_path_not_on_disk(
    live_coordinator, capsys: pytest.CaptureFixture[str]
) -> None:
    """Happy-ish path: tracking a path that doesn't exist on disk yet warns
    but still exits 0 (path will be seeded on first Read)."""
    workspace, port = live_coordinator
    rc = coherence_track.main(["--root", str(workspace), "docs/future.md"])
    captured = capsys.readouterr()
    assert rc == 0
    # Success ("tracked docs/future.md") on stdout; warning ("does not exist
    # on disk yet") on stderr per ce-review P2 fix #15.
    assert "tracked docs/future.md" in captured.out
    assert "does not exist on disk yet" in captured.err


def test_untrack_against_live_coordinator(
    live_coordinator, capsys: pytest.CaptureFixture[str]
) -> None:
    """Happy path: untrack a path → ignored.yaml updated, response carries removed list."""
    workspace, port = live_coordinator
    rc = coherence_untrack.main(["--root", str(workspace), "docs/draft.md"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "untracked docs/draft.md" in captured.out
    ignored_yaml = workspace / ".coherence" / "ignored.yaml"
    assert ignored_yaml.is_file()
    assert "docs/draft.md" in ignored_yaml.read_text()


@pytest.mark.parametrize("bad_path,reason_substr", [
    # See test_track_rejects_invalid_paths_without_network for rationale on
    # the 2026-05-26 message change from "must be relative" to "outside
    # workspace root" — sibling normalization fix applies to untrack too.
    ("/etc/passwd", "outside workspace root"),
    ("../escape", "'..'"),
    ("", "empty"),
])
def test_untrack_rejects_invalid_paths_without_network(
    git_workspace: Path, capsys: pytest.CaptureFixture[str],
    bad_path: str, reason_substr: str,
) -> None:
    rc = coherence_untrack.main(["--root", str(git_workspace), bad_path])
    captured = capsys.readouterr()
    assert rc == 1
    # ce-review P2 fix #15: rejection messages go to stderr
    assert reason_substr in captured.err


# ----------------------------------------------------------------------
# coherence_status --show-policy
# ----------------------------------------------------------------------


def test_show_policy_renders_pending_section_when_path_not_yet_observed(
    live_coordinator, capsys: pytest.CaptureFixture[str],
) -> None:
    """A user-added path that has never been pre-read must appear under
    'Tracked (pending first read):' — it is in user_added_patterns but
    not yet in tracked_artifacts."""
    workspace, port = live_coordinator
    from ccs.cli._coherence_client import post as _post
    from ccs.cli._coherence_client import resolve_endpoint
    endpoint = resolve_endpoint(workspace)

    # Add a path via /policy/track so it enters user_added_patterns.
    path = "pending_first_read_test.md"
    resp = _post(endpoint, "/policy/track", {"paths": [path]})
    assert path in resp.get("added", []), f"track failed: {resp}"

    rc = coherence_status.main([
        "--root", str(workspace), "--show-policy",
    ])
    captured = capsys.readouterr()
    assert rc == 0
    assert "Tracked (pending first read):" in captured.out
    assert path in captured.out


def test_show_policy_renders_none_after_path_is_observed(
    live_coordinator, capsys: pytest.CaptureFixture[str],
) -> None:
    """Once a pre-read fires for every user-added path, the pending list
    is empty and the 'none' branch renders."""
    workspace, port = live_coordinator
    import uuid as _uuid

    from ccs.cli._coherence_client import post as _post
    from ccs.cli._coherence_client import resolve_endpoint
    endpoint = resolve_endpoint(workspace)

    path = "observed_test.md"
    _post(endpoint, "/policy/track", {"paths": [path]})

    # Fire a pre-read to seed the artifact into tracked_artifacts.
    sid = str(_uuid.uuid4())
    _post(endpoint, "/hooks/pre-read", {
        "session_id": sid,
        "path": path,
        "content_hash": "a" * 64,
    })

    rc = coherence_status.main([
        "--root", str(workspace), "--show-policy",
    ])
    captured = capsys.readouterr()
    assert rc == 0
    assert "Tracked (pending first read): none" in captured.out


def test_show_policy_json_injects_pending_first_read_key(
    live_coordinator, capsys: pytest.CaptureFixture[str],
) -> None:
    """--show-policy --json injects 'policy_pending_first_read' into the
    payload so agent callers get the same data as the table renderer."""
    workspace, port = live_coordinator
    from ccs.cli._coherence_client import post as _post
    from ccs.cli._coherence_client import resolve_endpoint
    endpoint = resolve_endpoint(workspace)

    path = "json_pending_test.md"
    _post(endpoint, "/policy/track", {"paths": [path]})

    rc = coherence_status.main([
        "--root", str(workspace), "--show-policy", "--json",
    ])
    captured = capsys.readouterr()
    assert rc == 0
    data = json.loads(captured.out)
    assert "policy_pending_first_read" in data, (
        "--json --show-policy must inject 'policy_pending_first_read' key"
    )
    assert path in data["policy_pending_first_read"]


# ----------------------------------------------------------------------
# _coherence_client — endpoint resolution edge cases
# ----------------------------------------------------------------------


def test_resolve_endpoint_missing_pid_file(git_workspace: Path) -> None:
    """No port file → CoordinatorUnavailable with operator-friendly message."""
    with pytest.raises(CoordinatorUnavailable) as excinfo:
        resolve_endpoint(git_workspace)
    assert "no coordinator running" in str(excinfo.value)


def test_resolve_endpoint_missing_secret_file(git_workspace: Path) -> None:
    """Port file present but hook.secret missing → CoordinatorUnavailable."""
    (git_workspace / ".coherence").mkdir(parents=True, exist_ok=True, mode=0o700)
    (git_workspace / ".coherence" / "server.pid").write_text("12345\n50000\n")
    with pytest.raises(CoordinatorUnavailable) as excinfo:
        resolve_endpoint(git_workspace)
    assert "authentication unavailable" in str(excinfo.value)


def test_resolve_endpoint_empty_secret(git_workspace: Path) -> None:
    """hook.secret exists but is empty → CoordinatorUnavailable."""
    coh = git_workspace / ".coherence"
    coh.mkdir(parents=True, exist_ok=True, mode=0o700)
    (coh / "server.pid").write_text("12345\n50000\n")
    (coh / "hook.secret").write_text("")
    with pytest.raises(CoordinatorUnavailable) as excinfo:
        resolve_endpoint(git_workspace)
    assert "empty" in str(excinfo.value)


def test_render_table_marks_a_holder_with_no_known_name(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A holder whose grant outlived the coordinator that issued it arrives
    with ``agent_name: null``. The renderer must say so, not print "None" —
    and must not reuse the "no held grants" line, which means the opposite."""
    monkeypatch.setenv("COLUMNS", "80")
    payload = {
        "tracked_artifacts": [{"path": "docs/plan.md", "version": 2}],
        "sessions": [
            {
                "agent_name": None,
                "agent_id": "4c9625da-356c-527f-b5d7-027f181f7748",
                "states": {"docs/plan.md": "EXCLUSIVE"},
            },
        ],
        "policy_summary": {},
        "coordinator_pid": 0,
    }
    coherence_status._render_table(payload)
    out = capsys.readouterr().out

    assert "None" not in out
    assert "name unknown" in out
    assert "4c9625da" in out
    assert "docs/plan.md" in out and "EXCLUSIVE" in out
    assert "No active sessions." not in out


def test_render_table_names_a_sweep_reclaim_beside_held_states(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """#195: the operator tier's ``reclaimed`` map renders under the session,
    labelled as a reclaim with its trigger and tick — not as a held state, and
    not as the "no held grants" line a clean release prints."""
    monkeypatch.setenv("COLUMNS", "120")
    payload = {
        "tracked_artifacts": [{"path": "docs/plan.md", "version": 2}],
        "sessions": [
            {
                "agent_name": "claude-session-x",
                "agent_id": "4c9625da-356c-527f-b5d7-027f181f7748",
                "states": {"docs/spec.md": "SHARED"},
                "reclaimed": {
                    "docs/plan.md": {"trigger": "reclaim_heartbeat", "tick": 1789558656}
                },
            },
            {
                "agent_name": "claude-session-y",
                "agent_id": "5c9625da-356c-527f-b5d7-027f181f7748",
                "states": {},
                "reclaimed": {},
            },
        ],
        "policy_summary": {},
        "coordinator_pid": 0,
        "sweep_reclaims_total": 1,
    }
    coherence_status._render_table(payload)
    out = capsys.readouterr().out

    assert "reclaimed (reclaim_heartbeat at tick 1789558656)" in out
    assert "docs/spec.md" in out and "SHARED" in out
    # The released session (y) still reads as holding nothing.
    assert out.count("(no held grants)") == 1
    assert "sweep_reclaims_total" in out
