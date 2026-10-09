# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Tests for the four agent-coherence-* console scripts (Unit 6).

Covers happy paths, argparse + path validation, graceful-no-coordinator
behavior, and HTTP error propagation. End-to-end smoke (real spawned
coordinator) is exercised by the lifecycle tests; here we mostly hit the
control-flow / output-rendering paths.
"""

from __future__ import annotations

import http.server
import json
import os
import secrets
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

from ccs.adapters.claude_code.lifecycle import LifecycleConfig, ensure_coordinator, stop_coordinator
from ccs.cli import (
    coherence_coordinator,
    coherence_handoff,
    coherence_status,
    coherence_track,
    coherence_untrack,
)
from ccs.cli._coherence_client import (
    CoordinatorEndpoint,
    CoordinatorUnavailable,
    caller_principal_headers,
    claim_caller_principal,
    get,
    post,
    post_with_stored_principal,
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


def test_untrack_of_a_strict_path_is_refused_with_its_own_exit_code(
    git_workspace: Path, fast_cfg: LifecycleConfig, capsys: pytest.CaptureFixture[str]
) -> None:
    """#261: the coordinator refuses to untrack a path it enforces in strict
    mode (typed reason ``untrack_strict_path``, nothing written). The CLI
    classifies the refusal by that reason, names the strict pattern, says how
    to untrack it (restart without the strict entry), and exits 3 — distinct
    from a transport error (2)."""
    coherence_dir = git_workspace / ".coherence"
    coherence_dir.mkdir()
    (coherence_dir / "tracked.yaml").write_text("- data/**\n")
    (coherence_dir / "strict_mode.yaml").write_text("- data/**\n")
    assert ensure_coordinator(git_workspace, config=fast_cfg) > 0
    try:
        rc = coherence_untrack.main(["--root", str(git_workspace), "data/a.txt", "notes.md"])
        captured = capsys.readouterr()
        assert rc == 3, captured
        assert "refused 'data/a.txt': enforced in strict mode by data/**" in captured.err
        assert "restart the coordinator" in captured.err
        assert captured.out == ""
        assert not (coherence_dir / "ignored.yaml").exists(), "the request wrote nothing"
    finally:
        stop_coordinator(git_workspace)


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


def test_render_table_keeps_full_paths_beside_a_reclaim_at_80_columns(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """#195: a reclaim label is far wider than a state name. At the 80-column
    fallback an agent's shell tool gets, it must not shrink the path column
    until two different paths print the same."""
    monkeypatch.delenv("COLUMNS", raising=False)
    plan_a, plan_b = "docs/plans/2026-10-a-rollout-plan.md", "docs/plans/2026-10-b-rollout-plan.md"
    payload = {
        "tracked_artifacts": [{"path": plan_a, "version": 2}, {"path": plan_b, "version": 1}],
        "sessions": [
            {
                "agent_name": "claude-session-x",
                "agent_id": "4c9625da-356c-527f-b5d7-027f181f7748",
                "states": {plan_a: "SHARED", plan_b: "EXCLUSIVE"},
                "reclaimed": {plan_a: {"trigger": "reclaim_heartbeat", "tick": 1789558656}},
            },
        ],
        "policy_summary": {},
        "coordinator_pid": 0,
    }
    coherence_status._render_table(payload)
    session_lines = capsys.readouterr().out.split("Sessions:", 1)[1].splitlines()

    assert any(plan_a in ln and "reclaimed (reclaim_heartbeat" in ln for ln in session_lines)
    assert any(plan_b in ln and "EXCLUSIVE" in ln for ln in session_lines)


def test_render_table_keeps_a_reread_state_beside_its_reclaim(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """#195: a reclaimed session that re-read the path holds it SHARED and is
    still listed under ``reclaimed``. The line shows both, so the read it holds
    does not vanish behind the reclaim label."""
    monkeypatch.setenv("COLUMNS", "120")
    payload = {
        "tracked_artifacts": [{"path": "docs/plan.md", "version": 2}],
        "sessions": [
            {
                "agent_name": "claude-session-x",
                "agent_id": "4c9625da-356c-527f-b5d7-027f181f7748",
                "states": {"docs/plan.md": "SHARED"},
                "reclaimed": {
                    "docs/plan.md": {"trigger": "reclaim_heartbeat", "tick": 1789558656}
                },
            },
        ],
        "policy_summary": {},
        "coordinator_pid": 0,
    }
    coherence_status._render_table(payload)
    out = capsys.readouterr().out

    [line] = [ln for ln in out.splitlines() if "docs/plan.md" in ln and "reclaimed" in ln]
    assert "SHARED" in line
    assert "reclaimed (reclaim_heartbeat at tick 1789558656)" in line


# ----------------------------------------------------------------------
# coherence_status — the handoffs block (#185)
# ----------------------------------------------------------------------

_GIVER_AGENT = "4c9625da-356c-527f-b5d7-027f181f7748"
_SUCCESSOR_AGENT = "0f3e1a2b-5c6d-5e7f-8a9b-0c1d2e3f4a5b"


def _status_payload(**plan_entry_extra: object) -> dict:
    """A ``/status`` payload with two tracked artifacts and one session; the
    ``plan.md`` entry gains ``plan_entry_extra`` (the ``handoff`` key)."""
    return {
        "coordinator_pid": 4242,
        "coordinator_uptime_seconds": 12.0,
        "coordinator_backend": "python",
        "coordinator_version": "9.9.9",
        "policy_summary": {
            "default_pattern_count": 3, "user_added_pattern_count": 0, "ignored_pattern_count": 0,
        },
        "tracked_artifacts": [
            {"path": "plan.md", "version": 2, **plan_entry_extra},
            {"path": "spec.md", "version": 1},
        ],
        "sessions": [
            {"agent_id": _GIVER_AGENT, "agent_name": "claude-session-x", "states": {"spec.md": "SHARED"}},
        ],
    }


#: Today's text rendering of :func:`_status_payload` with no handoff key, at 80
#: columns, written out by hand.
_STATUS_TEXT_WITHOUT_HANDOFF = (
    "Coordinator: pid=4242 uptime=12s backend=python version=9.9.9\n"
    "\n"
    "Policy: 3 default pattern(s), 0 user-added, 0 ignored\n"
    "\n"
    "Observed artifacts:\n"
    "  version = artifact revision: starts at 1, +1 on every committed edit\n"
    "  (a read is flagged stale when its version is behind the current one)\n"
    "\n"
    "  path     version\n"
    "  -------  -------\n"
    "  plan.md        2\n"
    "  spec.md        1\n"
    "\n"
    "Sessions:\n"
    "  4c9625da  claude-session-x\n"
    "    spec.md  SHARED\n"
)


def _handoff_key(**extra: object) -> dict:
    return {
        "giver": _GIVER_AGENT, "successor": _SUCCESSOR_AGENT, "version_at_transfer": 2,
        "hold_shape": "SHARED", "status": "pending", "live": True, **extra,
    }


def test_status_text_lists_a_handoff_after_the_artifacts_table(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An entry carrying the ``handoff`` key gets one line in a handoffs
    block printed right after the artifacts table: the path, giver and
    successor as short session-level agent ids, the version at transfer, the
    status and the record's age from its created timestamp. Everything else
    is today's output. Prevents a pending handoff being visible only in
    ``--json``, so an operator reading the table sees a fenced giver as an
    ordinary holder."""
    monkeypatch.setenv("COLUMNS", "80")
    payload = _status_payload(handoff=_handoff_key(created_at_unix_ts=time.time() - 125))

    coherence_status._render_table(payload)

    handoffs_block = (
        "Handoffs:\n"
        "  giver → successor, by session agent id (first 8 characters, as under Sessions)\n"
        "  plan.md: 4c9625da → 0f3e1a2b at version 2 (pending, 2m ago)\n"
        "\n"
    )
    expected = _STATUS_TEXT_WITHOUT_HANDOFF.replace("\nSessions:\n", "\n" + handoffs_block + "Sessions:\n")
    assert capsys.readouterr().out == expected


def test_status_text_is_byte_identical_to_today_when_no_entry_carries_a_handoff(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no ``handoff`` key on any entry the table is today's output,
    byte for byte, with no empty handoffs heading. Prevents every workspace
    that never hands a path off seeing its status output change."""
    monkeypatch.setenv("COLUMNS", "80")

    coherence_status._render_table(_status_payload())

    assert capsys.readouterr().out == _STATUS_TEXT_WITHOUT_HANDOFF


def test_status_text_renders_a_handoff_without_an_age_when_the_entry_has_no_timestamp(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The default tier's ``handoff`` key carries no created timestamp (only the
    operator tier does): the line is printed without an age rather than with
    a made-up one. Prevents a ``--detail minimal`` table inventing how long a
    handoff has been pending."""
    monkeypatch.setenv("COLUMNS", "80")
    payload = _status_payload(handoff=_handoff_key(status="completed"))

    coherence_status._render_table(payload)

    lines = capsys.readouterr().out.splitlines()
    assert "  plan.md: 4c9625da → 0f3e1a2b at version 2 (completed)" in lines
    assert not any("ago" in line for line in lines)


def test_status_text_marks_a_handoff_that_has_ended_although_its_label_still_reads_pending(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A record ends when the path's version moves, and a write can move it
    without changing the label, so an ended record can still read
    ``pending``. The line says it has ended. Prevents an operator reading the
    table as a giver still fenced and acting on a handoff that is over."""
    monkeypatch.setenv("COLUMNS", "80")
    payload = _status_payload(handoff=_handoff_key(live=False))

    coherence_status._render_table(payload)

    lines = capsys.readouterr().out.splitlines()
    assert "  plan.md: 4c9625da → 0f3e1a2b at version 2 (pending, ended)" in lines


# ----------------------------------------------------------------------
# coherence_handoff — the four handoff verbs (#185)
# ----------------------------------------------------------------------

_SESSION_VAR = "CLAUDE_CODE_SESSION_ID"

#: The verbs' exit codes, pinned by number; 3 is the status self-test's.
_EXIT_DONE, _EXIT_USAGE, _EXIT_FAILED, _EXIT_NOT_SERVED = 0, 1, 2, 4

_NOT_HELD_HINT = (
    "hint: a Claude Code session's write grant ends when its turn ends. If an "
    "earlier transfer of plan.md may have landed, check the path's handoff in "
    "agent-coherence-status output first: a handoff from this session to that "
    "successor made at the version it held, live or ended, means it landed, so "
    "do not transfer again, and if it shows another session's handoff, ask "
    "before transferring; otherwise have the giver session read plan.md, then "
    "transfer it again (on a strict-mode path that read is denied: hand the "
    "path on in the same turn as its edit)"
)

_VERB_RUNS = {
    "transfer": lambda argv: coherence_handoff.transfer_main(["--successor", str(uuid.uuid4()), *argv]),
    "accept": coherence_handoff.accept_main,
    "decline": coherence_handoff.decline_main,
    "withdraw": coherence_handoff.withdraw_main,
}


def _session_agent(session_id: str) -> str:
    """A session's session-level agent id, by the derivation docs/guide.md
    states -- computed here, never read back from the code under test."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"ccs-agent:claude-session-{session_id}"))


def _hook(workspace: Path, route: str, body: dict) -> dict:
    """One hook event for the session ``body`` names, sent as the hook client
    sends it: presenting the session's stored principal, which its first
    event claims and stores."""
    return post_with_stored_principal(resolve_endpoint(workspace), workspace, route, body)


def _known_successor(workspace: Path) -> str:
    """A fresh session that has made one hook event, so it has claimed a
    principal and the coordinator knows it; its session-level agent id."""
    sid = str(uuid.uuid4())
    _hook(workspace, "/hooks/pre-read", {"session_id": sid, "path": "spec.md"})
    return _session_agent(sid)


def _operator_status(workspace: Path) -> dict:
    return get(
        resolve_endpoint(workspace), "/status?detail=full",
        extra_headers={"Coherence-Local-Operator": "true"},
    )


def _handoff_on(workspace: Path, path: str) -> dict | None:
    for entry in _operator_status(workspace)["tracked_artifacts"]:
        if entry["path"] == path:
            return entry.get("handoff")
    return None


def _principal_files(workspace: Path, suffix: str = "") -> list[Path]:
    return sorted((workspace / ".coherence").glob(f"caller-principal-*{suffix}"))


def _principal_file(workspace: Path, session_id: str, suffix: str) -> Path:
    key = uuid.UUID(_session_agent(session_id)).hex
    return workspace / ".coherence" / f"caller-principal-{key}{suffix}"


class _StubCoordinator(http.server.BaseHTTPRequestHandler):
    """A coordinator of another build: answers each route (the path without
    its query) from ``answers`` -- route -> (status, body) or (status, body,
    reason phrase); a ``bytes`` body is sent as is and any other as JSON, and
    a 3xx carries a ``Location`` -- every other route 404 as a coordinator
    answers an unknown one, and records each request's path in ``seen``."""

    answers: dict[str, tuple[Any, ...]] = {}
    seen: list[str] = []

    def do_POST(self) -> None:  # noqa: N802 - stdlib name
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self._answer()

    def do_GET(self) -> None:  # noqa: N802 - stdlib name
        self._answer()

    def _answer(self) -> None:
        self.seen.append(self.path)
        route = self.path.split("?", 1)[0]
        status, body, *reason = self.answers.get(route, (404, {"error": "unknown route"}))
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status, *reason)
        if 300 <= status < 400:
            self.send_header("Location", "http://127.0.0.1:9/elsewhere")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *_args: Any) -> None:
        return


@pytest.fixture
def stub_coordinator(git_workspace: Path):
    """A workspace wired to a :class:`_StubCoordinator` whose pid file has no
    backend line (the Python coordinator's format); yields (workspace, port)."""
    coherence = git_workspace / ".coherence"
    coherence.mkdir(mode=0o700)
    httpd = http.server.HTTPServer(("127.0.0.1", 0), _StubCoordinator)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]
    (coherence / "server.pid").write_text(f"999\n{port}\n")
    (coherence / "hook.secret").write_text("test-secret")
    _StubCoordinator.answers, _StubCoordinator.seen = {}, []
    try:
        yield git_workspace, port
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_transfer_with_no_session_flag_acts_as_the_session_the_harness_variable_names(
    live_coordinator, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """In a hook session's shell, with no ``--session``, the
    transfer acts as the session ``CLAUDE_CODE_SESSION_ID`` names: the
    coordinator records that session as the giver, the verb prints the
    session-level id it acted as (the giver id the answer returns) and that
    the variable named the session, and the raw session id is printed
    nowhere. The status table then lists the handoff. Prevents the default
    resolving to some other identity with nothing on screen to show it."""
    workspace, _ = live_coordinator
    giver = str(uuid.uuid4())
    successor_agent = _known_successor(workspace)
    _hook(workspace, "/hooks/pre-read", {"session_id": giver, "path": "plan.md"})
    monkeypatch.setenv(_SESSION_VAR, giver)

    rc = coherence_handoff.transfer_main([
        "--root", str(workspace), "--successor", successor_agent, "plan.md",
    ])

    captured = capsys.readouterr()
    assert rc == _EXIT_DONE, captured.err
    giver_agent = _session_agent(giver)
    assert (
        f"agent-coherence-transfer: acting as session agent {giver_agent} "
        f"(session from CLAUDE_CODE_SESSION_ID)"
    ) in captured.out
    assert giver not in captured.out + captured.err
    handoff = _handoff_on(workspace, "plan.md")
    assert handoff is not None
    assert (handoff["giver"], handoff["successor"], handoff["status"]) == (
        giver_agent, successor_agent, "pending")

    assert coherence_status.main(["--root", str(workspace)]) == 0
    status_out = capsys.readouterr().out
    assert f"  plan.md: {giver_agent[:8]} → {successor_agent[:8]} at version 1 (pending, " in status_out


@pytest.mark.parametrize("verb", sorted(_VERB_RUNS))
def test_a_verb_with_no_session_flag_and_no_harness_variable_is_a_usage_error_that_sends_nothing(
    verb: str, live_coordinator, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With neither ``--session`` nor ``CLAUDE_CODE_SESSION_ID``
    the verb refuses rather than guessing: exit 1 with its usage, and nothing
    reaches the coordinator -- no claim, no verb request, no caller-principal
    file. Prevents a verb run outside a Claude Code session acting as an
    identity it made up."""
    workspace, _ = live_coordinator
    monkeypatch.delenv(_SESSION_VAR, raising=False)
    before = _operator_status(workspace)["endpoint_counters"]

    rc = _VERB_RUNS[verb](["--root", str(workspace), "plan.md"])

    captured = capsys.readouterr()
    assert rc == _EXIT_USAGE
    assert f"usage: agent-coherence-{verb}" in captured.err
    assert "--session" in captured.err and _SESSION_VAR in captured.err
    after = _operator_status(workspace)["endpoint_counters"]
    for counter in ("principal_claim_total", f"handoff_{verb}_total"):
        assert after[counter] == before[counter], counter
    assert _principal_files(workspace) == []


def test_a_malformed_command_line_exits_1_not_argparse_2(
    git_workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exit 1 is usage. argparse's own exit for a bad command line is 2,
    which these verbs mean as a coordinator failure, so a transfer missing its
    successor exits 1. Prevents a script reading a typo as an unreachable
    coordinator."""
    with pytest.raises(SystemExit) as exc:
        coherence_handoff.transfer_main([
            "--root", str(git_workspace), "--session", str(uuid.uuid4()), "plan.md",
        ])
    assert exc.value.code == _EXIT_USAGE
    assert "--successor" in capsys.readouterr().err


@pytest.mark.parametrize("source", ["flag", "variable"])
@pytest.mark.parametrize("verb", sorted(_VERB_RUNS))
def test_each_verb_prints_the_session_agent_id_it_acted_as_and_where_the_session_came_from(
    verb: str, source: str, live_coordinator, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every verb prints the session-level agent id it acted as and
    whether ``--session`` or ``CLAUDE_CODE_SESSION_ID`` named the session (the
    flag winning over the variable), and never the raw session id. Prevents a
    shell whose variable names another session acting as that session with
    nothing on screen to show it."""
    workspace, _ = live_coordinator
    acting, other = str(uuid.uuid4()), str(uuid.uuid4())
    if source == "flag":
        monkeypatch.setenv(_SESSION_VAR, other)
        argv, named_by = ["--root", str(workspace), "--session", acting, "plan.md"], "--session"
    else:
        monkeypatch.setenv(_SESSION_VAR, acting)
        argv, named_by = ["--root", str(workspace), "plan.md"], _SESSION_VAR

    _VERB_RUNS[verb](argv)

    captured = capsys.readouterr()
    assert (
        f"agent-coherence-{verb}: acting as session agent {_session_agent(acting)} "
        f"(session from {named_by})"
    ) in captured.out
    printed = captured.out + captured.err
    assert acting not in printed
    assert other not in printed and _session_agent(other) not in printed


def test_a_verb_presents_the_stored_principal_of_a_session_its_hooks_already_claimed(
    live_coordinator, capsys: pytest.CaptureFixture[str]
) -> None:
    """The verb presents the named session's stored caller principal, as
    that session's hook events do, and the require-class route admits it on
    the first request: no principal refusal, no claim, the stored files
    untouched. Prevents the verb sending no principal for a bound session,
    which every require-class route refuses."""
    workspace, _ = live_coordinator
    giver = str(uuid.uuid4())
    successor_agent = _known_successor(workspace)
    _hook(workspace, "/hooks/pre-read", {"session_id": giver, "path": "plan.md"})
    stored = {p.name: p.read_bytes() for p in _principal_files(workspace)}
    before = _operator_status(workspace)

    rc = coherence_handoff.transfer_main([
        "--root", str(workspace), "--session", giver, "--successor", successor_agent, "plan.md",
    ])

    assert rc == _EXIT_DONE, capsys.readouterr().err
    after = _operator_status(workspace)
    assert after["caller_principal_refused_total"] == before["caller_principal_refused_total"]
    claims = "principal_claim_total"
    assert after["endpoint_counters"][claims] == before["endpoint_counters"][claims]
    assert {p.name: p.read_bytes() for p in _principal_files(workspace)} == stored


def test_a_verb_for_a_session_with_no_principal_mints_claims_and_stores_it_as_a_first_hook_event_would(
    live_coordinator, capsys: pytest.CaptureFixture[str]
) -> None:
    """The fact docs/security.md states for the handoff CLI: for a named
    session with no stored principal and no binding, the verb creates the
    session's mint nonce, claims, and stores the principal -- both files
    exist afterwards, and the stored principal is the bound one -- and the
    route then answers on its own terms. Prevents the verb binding the
    session under a nonce it keeps nowhere, which would lock that session's
    own hook events out."""
    workspace, _ = live_coordinator
    session = str(uuid.uuid4())

    rc = coherence_handoff.withdraw_main(["--root", str(workspace), "--session", session, "plan.md"])

    captured = capsys.readouterr()
    assert rc == _EXIT_FAILED
    assert "(handoff_not_live)" in captured.err
    nonce_file = _principal_file(workspace, session, ".nonce")
    principal_file = _principal_file(workspace, session, ".principal")
    assert _principal_files(workspace) == [nonce_file, principal_file]
    # The session's next require-class hook event, presenting the stored
    # principal, is admitted: the binding is the one the files hold.
    answer = post(
        resolve_endpoint(workspace), "/hooks/pre-edit", {"session_id": session, "path": "plan.md"},
        extra_headers=caller_principal_headers(principal_file.read_text().strip()),
    )
    assert answer.get("ok") is True


def test_a_verb_for_a_session_bound_under_another_nonce_is_refused_and_never_re_mints(
    live_coordinator, capsys: pytest.CaptureFixture[str]
) -> None:
    """A session bound under ANOTHER mint nonce -- claimed first by
    a direct claim, standing in for a live volume's or MCP session's id --
    cannot be acted for. Withdrawing that session's live handoff reports the
    caller-principal refusal and exits 2, stores no principal file, presents
    the same stored nonce on a second run rather than minting a new one, and
    leaves the transfer record as it was. Prevents the CLI withdrawing a
    handoff in the name of a writer it is not, and re-minting, which would
    reopen the first-claim gate."""
    workspace, _ = live_coordinator
    endpoint = resolve_endpoint(workspace)
    volume = str(uuid.uuid4())
    claim = claim_caller_principal(endpoint, volume, secrets.token_urlsafe(32))
    assert claim.outcome == "bound"
    headers = caller_principal_headers(claim.principal)
    successor_agent = _known_successor(workspace)
    post(endpoint, "/hooks/pre-read", {"session_id": volume, "path": "plan.md"}, extra_headers=headers)
    handed = post(endpoint, "/handoff/transfer", {
        "session_id": volume, "successor": successor_agent, "grants": [{"path": "plan.md"}],
    }, extra_headers=headers)
    assert handed["ok"] is True
    record = _handoff_on(workspace, "plan.md")
    nonce_file = _principal_file(workspace, volume, ".nonce")

    nonces = []
    for _ in range(2):
        rc = coherence_handoff.withdraw_main([
            "--root", str(workspace), "--session", volume, "plan.md",
        ])
        captured = capsys.readouterr()
        assert rc == _EXIT_FAILED
        assert "(caller_principal_absent)" in captured.err
        assert "different mint nonce" in captured.err
        assert claim.principal not in captured.out + captured.err
        assert not _principal_file(workspace, volume, ".principal").exists()
        nonces.append(nonce_file.read_bytes())
    assert nonces[0] == nonces[1]
    assert _handoff_on(workspace, "plan.md") == record
    assert (record["status"], record["live"]) == ("pending", True)


def test_a_transfer_after_the_givers_turn_ended_is_not_held_with_a_hint_and_a_pre_read_makes_it_transferable(
    live_coordinator, capsys: pytest.CaptureFixture[str]
) -> None:
    """A hook session's write grant ends at its turn end (its session-stop), so
    a transfer run afterwards answers not held: the verb prints the wire
    reason unchanged, adds the client-side hint (have the giver session read
    the path, then transfer) and exits 2. After the giver session's pre-read
    the same transfer goes through, giving up a SHARED claim. Prevents an
    operator meeting a bare refusal with no way forward."""
    workspace, _ = live_coordinator
    giver = str(uuid.uuid4())
    successor_agent = _known_successor(workspace)
    _hook(workspace, "/hooks/pre-edit", {"session_id": giver, "path": "plan.md"})
    _hook(workspace, "/hooks/post-edit", {
        "session_id": giver, "path": "plan.md", "content_hash": "b" * 64, "success": True,
    })
    _hook(workspace, "/hooks/session-stop", {"session_id": giver})
    argv = ["--root", str(workspace), "--session", giver, "--successor", successor_agent, "plan.md"]

    rc = coherence_handoff.transfer_main(argv)

    captured = capsys.readouterr()
    assert rc == _EXIT_FAILED
    assert "agent-coherence-transfer: plan.md not transferred (handoff_not_held)" in captured.err
    assert f"agent-coherence-transfer: {_NOT_HELD_HINT}" in captured.err
    assert _handoff_on(workspace, "plan.md") is None

    _hook(workspace, "/hooks/pre-read", {"session_id": giver, "path": "plan.md"})
    rc = coherence_handoff.transfer_main(argv)

    captured = capsys.readouterr()
    assert rc == _EXIT_DONE, captured.err
    assert (
        f"agent-coherence-transfer: transferred plan.md to {successor_agent} "
        f"at version 2 (gave up SHARED; status pending)"
    ) in captured.out
    handoff = _handoff_on(workspace, "plan.md")
    assert handoff is not None
    assert (handoff["hold_shape"], handoff["version_at_transfer"]) == ("SHARED", 2)


@pytest.mark.parametrize(("verb", "party", "taken", "status"), [
    ("accept", "successor", "accepted", "completed"),
    ("decline", "successor", "declined", "declined"),
    ("withdraw", "giver", "withdrew", "withdrawn"),
])
def test_a_settling_verb_that_lands_exits_0_and_says_what_it_did(
    live_coordinator, capsys: pytest.CaptureFixture[str],
    verb: str, party: str, taken: str, status: str,
) -> None:
    """Against a live coordinator, after a real transfer, accept and decline
    run as the successor and withdraw as the giver: each exits 0, prints the
    one line saying what it did with the record's new status, and the record
    reads that status. Fails if a verb reports success with another verb's
    wording (a decline that tells the model it accepted) or with a wrong
    exit code."""
    workspace, _ = live_coordinator
    giver, successor = str(uuid.uuid4()), str(uuid.uuid4())
    _hook(workspace, "/hooks/pre-read", {"session_id": successor, "path": "spec.md"})
    _hook(workspace, "/hooks/pre-read", {"session_id": giver, "path": "plan.md"})
    assert coherence_handoff.transfer_main([
        "--root", str(workspace), "--session", giver,
        "--successor", _session_agent(successor), "plan.md",
    ]) == _EXIT_DONE
    capsys.readouterr()

    acting = giver if party == "giver" else successor
    rc = _VERB_RUNS[verb](["--root", str(workspace), "--session", acting, "plan.md"])

    captured = capsys.readouterr()
    assert rc == _EXIT_DONE, captured.err
    assert (
        f"agent-coherence-{verb}: {taken} the handoff of plan.md (status {status})"
    ) in captured.out.splitlines(), captured.out
    assert _handoff_on(workspace, "plan.md")["status"] == status


def test_the_subagent_flag_transfers_a_grant_the_subagent_holds(
    live_coordinator, capsys: pytest.CaptureFixture[str]
) -> None:
    """A grant a subagent holds is its composite's, not the main thread's,
    and a subagent's shell names the parent session -- so without
    ``--subagent-id`` the transfer finds nothing held, and with it the
    subagent's claim is handed on, the record naming the SESSION as giver.
    Prevents a subagent's grant being out of the CLI's reach."""
    workspace, _ = live_coordinator
    giver = str(uuid.uuid4())
    successor_agent = _known_successor(workspace)
    _hook(workspace, "/hooks/pre-read", {"session_id": giver, "agent_id": "worker-1", "path": "plan.md"})
    argv = ["--root", str(workspace), "--session", giver, "--successor", successor_agent, "plan.md"]

    assert coherence_handoff.transfer_main(argv) == _EXIT_FAILED
    assert "plan.md not transferred (handoff_not_held)" in capsys.readouterr().err

    rc = coherence_handoff.transfer_main([*argv, "--subagent-id", "worker-1"])

    assert rc == _EXIT_DONE, capsys.readouterr().err
    handoff = _handoff_on(workspace, "plan.md")
    assert handoff is not None
    assert (handoff["giver"], handoff["hold_shape"]) == (_session_agent(giver), "SHARED")


@pytest.mark.parametrize("verb", sorted(_VERB_RUNS))
def test_a_verb_against_a_pid_file_naming_the_node_backend_exits_4_without_a_round_trip(
    verb: str, stub_coordinator, capsys: pytest.CaptureFixture[str]
) -> None:
    """The Node coordinator does not serve the four routes and its pid
    file says so on the backend line, so the verb exits 4 -- not on this
    backend -- before sending anything: no claim, no verb request, no
    caller-principal file. Prevents a Node workspace paying a round trip, and
    leaving principal files, for an answer the pid file already gave."""
    workspace, port = stub_coordinator
    (workspace / ".coherence" / "server.pid").write_text(f"999\n{port}\nbackend=node\n")

    rc = _VERB_RUNS[verb](["--root", str(workspace), "--session", str(uuid.uuid4()), "plan.md"])

    captured = capsys.readouterr()
    assert rc == _EXIT_NOT_SERVED
    assert "does not serve" in captured.err and "Python coordinator" in captured.err
    assert _StubCoordinator.seen == []
    assert _principal_files(workspace) == []


@pytest.mark.parametrize(("answers", "expected_rc"), [
    ({"/principal/claim": (200, {"ok": True, "principal": "P" * 43})}, _EXIT_NOT_SERVED),
    ({"/handoff/withdraw": (200, {"ok": True, "status": "withdrawn"})}, _EXIT_DONE),
], ids=["claim-served-verb-404", "claim-404-verb-served"])
def test_with_no_backend_line_not_served_is_the_verb_routes_own_404_never_the_claim_routes(
    answers: dict, expected_rc: int, stub_coordinator, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no backend line in the pid file, 'not served' is decided by
    the verb route's own 404 and never by the claim route's answer: a
    coordinator that issues principals but answers the withdraw 404 exits 4,
    and one that answers the claim 404 but serves the withdraw is answered by
    the withdraw. Prevents a coordinator without the verb reading as an
    ordinary HTTP error, and a coordinator without principals reading as one
    without the verb."""
    workspace, _ = stub_coordinator
    _StubCoordinator.answers = answers

    rc = coherence_handoff.withdraw_main([
        "--root", str(workspace), "--session", str(uuid.uuid4()), "plan.md",
    ])

    assert rc == expected_rc, capsys.readouterr().err
    assert "/handoff/withdraw" in _StubCoordinator.seen


_CLAIM_SERVED = {"/principal/claim": (200, {"ok": True, "principal": "P" * 43})}


@pytest.mark.parametrize("verb", ["transfer", "accept", "decline", "withdraw"])
def test_an_unconfirmed_answer_exits_2_saying_the_outcome_is_unknown(
    verb: str, stub_coordinator, capsys: pytest.CaptureFixture[str]
) -> None:
    """A verb the coordinator's watchdog cut short may still have landed. The
    command says the outcome is unknown and where to look before acting
    again, and exits 2. Prevents an unconfirmed transfer exiting 0, which a
    script would read as done, or reading as a refusal that changed
    nothing."""
    workspace, _ = stub_coordinator
    reason = f"handoff_{verb}_unconfirmed"
    _StubCoordinator.answers = {
        **_CLAIM_SERVED,
        f"/handoff/{verb}": (200, {"ok": False, "degraded": True, "reason": reason}),
    }

    rc = _VERB_RUNS[verb](["--root", str(workspace), "--session", str(uuid.uuid4()), "plan.md"])

    err = capsys.readouterr().err
    assert rc == _EXIT_FAILED, err
    assert (
        f"agent-coherence-{verb}: the coordinator could not confirm the {verb} ({reason}); "
        "its outcome is unknown: check the path's handoff in agent-coherence-status "
        "before acting again"
    ) in err


def test_a_transfer_answer_that_omits_a_named_path_exits_2_naming_it(
    stub_coordinator, capsys: pytest.CaptureFixture[str]
) -> None:
    """A transfer answer with no grant entry for a named path reports
    nothing about that path, so the command cannot say it transferred: it
    names the path and exits 2, even when the answer says ``ok``. Prevents
    a path the answer left out reading as handed on."""
    workspace, _ = stub_coordinator
    _StubCoordinator.answers = {
        **_CLAIM_SERVED,
        "/handoff/transfer": (200, {"ok": True, "grants": []}),
    }

    rc = _VERB_RUNS["transfer"](["--root", str(workspace), "--session", str(uuid.uuid4()), "plan.md"])

    err = capsys.readouterr().err
    assert rc == _EXIT_FAILED, err
    assert "agent-coherence-transfer: plan.md: the answer reports no outcome for it" in err


def test_a_request_the_coordinator_cannot_read_exits_2_naming_what_to_check(
    live_coordinator, capsys: pytest.CaptureFixture[str]
) -> None:
    """An HTTP error other than the verb route's 404 exits 2. A session
    id that is not a UUID is answered 400, and the verb says what the
    coordinator could not read rather than a bare status. Prevents a
    mistyped ``--session`` reading as a coordinator fault with no lead."""
    workspace, _ = live_coordinator

    rc = coherence_handoff.withdraw_main(["--root", str(workspace), "--session", "not-a-session", "plan.md"])

    assert rc == _EXIT_FAILED
    assert (
        "agent-coherence-withdraw: HTTP 400: the coordinator could not read the request; "
        "check the session id, the subagent id and the paths"
    ) in capsys.readouterr().err


# ----------------------------------------------------------------------
# #245 -- coordinator text printed escaped, and one error path (exit 2)
# ----------------------------------------------------------------------

#: One coordinator string carrying each kind of character the commands must
#: not print raw: C0 controls (ESC opening a colour sequence, CR, LF, NUL), the
#: C1 CSI, a bidi override and a bidi isolate, a lone surrogate (a JSON string
#: can carry one) and an ideographic space.
_HOSTILE = "a\x1b[31m\r\n\x00\x9b\u202e\u2066\ud800\u3000z"
#: :data:`_HOSTILE` as the commands print it: each of those characters written
#: as ``repr`` writes it, without the quotes.
_PRINTED = r"a\x1b[31m\r\n\x00\x9b\u202e\u2066\ud800\u3000z"
_RAW_CHARACTERS = ("\x1b", "\r", "\x00", "\x9b", "\u202e", "\u2066", "\ud800", "\u3000")


def _raw_characters_in(text: str) -> list[str]:
    return [c for c in _RAW_CHARACTERS if c in text]


def _hostile_status_body() -> dict:
    """An operator-tier ``/status`` body with :data:`_HOSTILE` in every value
    the table prints, each behind a marker naming the field."""
    return {
        "coordinator_pid": f"pid{_HOSTILE}",
        "coordinator_uptime_seconds": 12.0,
        "coordinator_backend": f"backend{_HOSTILE}",
        "coordinator_version": f"version{_HOSTILE}",
        "policy_summary": {
            "default_pattern_count": 3, "user_added_pattern_count": 1, "ignored_pattern_count": 0,
            "user_added_patterns": [f"pattern{_HOSTILE}"],
        },
        "tracked_artifacts": [{
            "path": f"path{_HOSTILE}", "version": f"ver{_HOSTILE}",
            "handoff": {
                "giver": "\x1b\u202e\ud800giver-agent", "successor": "\x9b\r\nsucc-agent",
                "version_at_transfer": f"vat{_HOSTILE}", "status": f"status{_HOSTILE}", "live": True,
            },
        }],
        "sessions": [{
            "agent_id": "\x1b[31m\u202e4c9625da", "agent_name": f"name{_HOSTILE}",
            "states": {f"held{_HOSTILE}": f"state{_HOSTILE}"},
            "reclaimed": {f"lost{_HOSTILE}": {"trigger": f"trigger{_HOSTILE}", "tick": f"tick{_HOSTILE}"}},
        }],
        "endpoint_counters": {"pre_read_total": f"count{_HOSTILE}"},
        "sweep_reclaims_total": f"sweep{_HOSTILE}",
    }


def test_status_prints_every_coordinator_string_with_its_control_characters_escaped(
    stub_coordinator, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every string the status table takes from the coordinator -- the header,
    a pending pattern, a path and its version, the handoff line, a session's id
    and name, a held path and its state, a reclaim's path, trigger and tick, and
    a counter -- prints with ESC, CR, LF, NUL, the C1 CSI, bidi controls, a lone
    surrogate and an ideographic space escaped as ``repr`` writes them, each on
    its own line, and the command exits 0. Prevents a coordinator answer
    driving the operator's terminal (colours, a cursor move, a reordered line)
    or crashing the print on a surrogate."""
    workspace, _ = stub_coordinator
    monkeypatch.setenv("COLUMNS", "300")
    _StubCoordinator.answers = {"/status": (200, _hostile_status_body())}

    rc = coherence_status.main(["--root", str(workspace), "--show-policy"])

    captured = capsys.readouterr()
    assert rc == 0, captured.err
    lines = captured.out.splitlines()
    assert f"Coordinator: pid=pid{_PRINTED} uptime=12s backend=backend{_PRINTED} version=version{_PRINTED}" in lines
    assert f"  pattern{_PRINTED}" in lines
    assert any(f"path{_PRINTED}" in ln and ln.endswith(f"ver{_PRINTED}") for ln in lines)
    assert (
        f"  path{_PRINTED}: \\x1b\\u202e\\ud800giver → \\x9b\\r\\nsucc- at version vat{_PRINTED} "
        f"(status{_PRINTED})"
    ) in lines
    assert f"  \\x1b[31m\\u202e4c  name{_PRINTED}" in lines
    assert any(f"held{_PRINTED}" in ln and ln.endswith(f"state{_PRINTED}") for ln in lines)
    assert any(
        f"lost{_PRINTED}" in ln and ln.endswith(f"reclaimed (trigger{_PRINTED} at tick tick{_PRINTED})")
        for ln in lines
    )
    assert any(ln.startswith("  pre_read_total") and ln.endswith(f"count{_PRINTED}") for ln in lines)
    assert any(ln.startswith("  sweep_reclaims_total") and ln.endswith(f"sweep{_PRINTED}") for ln in lines)
    assert _raw_characters_in(captured.out + captured.err) == []


def test_status_elides_an_escaped_long_path_and_keeps_its_columns_aligned(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A long path carrying an ESC is middle-elided on its escaped text, so the
    escape's four characters count toward the column: every row of the
    artifacts table is the same width within the terminal, the version stays
    on the path's line, and the session's state column starts at one offset.
    Prevents a widened escape pushing its row past the screen edge or out of
    line with the rest."""
    monkeypatch.setenv("COLUMNS", "80")
    long_path = "docs/" + "x" * 70 + "\x1b[31m.md"
    payload = {
        "tracked_artifacts": [{"path": "plan.md", "version": 2}, {"path": long_path, "version": 13}],
        "sessions": [{
            "agent_id": _GIVER_AGENT, "agent_name": "claude-session-x",
            "states": {"plan.md": "SHARED", long_path: "EXCLUSIVE"},
        }],
        "policy_summary": {},
        "coordinator_pid": 0,
    }

    coherence_status._render_table(payload)

    out = capsys.readouterr().out
    lines = out.splitlines()
    assert _raw_characters_in(out) == []
    assert all(len(ln) <= 80 for ln in lines), [ln for ln in lines if len(ln) > 80]
    table = lines[lines.index("  path" + " " * 64 + "  version"):][:4]
    assert len({len(row) for row in table}) == 1, table
    long_row = table[3]
    assert "…" in long_row and long_row.endswith(r"x\x1b[31m.md       13")
    held = [ln for ln in lines if ln.endswith(("  SHARED", "  EXCLUSIVE"))]
    assert len(held) == 2
    assert {len(ln) - len(ln.split()[-1]) for ln in held} == {4 + 64 + 2}, held
    assert any(r"\x1b[31m.md" in ln for ln in held)


def test_track_prints_coordinator_paths_escaped_and_a_rejected_path_quoted(
    stub_coordinator, capsys: pytest.CaptureFixture[str]
) -> None:
    """A tracked path the coordinator names prints escaped on the success line
    and on the not-on-disk warning; printable text, spaces, quotes, backslashes
    and non-ASCII letters included, prints unchanged, while a no-break space and
    a zero-width joiner print escaped. A path the coordinator rejects prints in
    the quoted ``repr`` form of the command's own rejections, with its reason
    escaped. Prevents a coordinator's answer writing control characters to the
    terminal through track's output."""
    workspace, _ = stub_coordinator
    printable = "docs/a b'c\\d\"é日.md"
    _StubCoordinator.answers = {"/policy/track": (200, {
        "added": [f"docs/{_HOSTILE}.md", printable, "docs/nb\u00a0sp\u200d.md"],
        "rejected": [{"path": f"r{_HOSTILE}", "reason": f"why{_HOSTILE}"}],
    })}

    rc = coherence_track.main(["--root", str(workspace), "docs/plan.md"])

    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert captured.out.splitlines() == [
        f"agent-coherence-track: tracked docs/{_PRINTED}.md",
        f"agent-coherence-track: tracked {printable}",
        "agent-coherence-track: tracked docs/nb\\xa0sp\\u200d.md",
    ]
    assert captured.err.splitlines() == [
        f"agent-coherence-track: warning: docs/{_PRINTED}.md does not exist on disk yet",
        f"agent-coherence-track: warning: {printable} does not exist on disk yet",
        "agent-coherence-track: warning: docs/nb\\xa0sp\\u200d.md does not exist on disk yet",
        f"agent-coherence-track: rejected 'r{_PRINTED}': why{_PRINTED}",
    ]


def test_untrack_prints_coordinator_paths_escaped(
    stub_coordinator, capsys: pytest.CaptureFixture[str]
) -> None:
    """An untracked path the coordinator names prints escaped on the success
    line. Prevents a coordinator's answer writing control characters to the
    terminal through untrack's output."""
    workspace, _ = stub_coordinator
    _StubCoordinator.answers = {"/policy/untrack": (200, {"removed": [f"docs/{_HOSTILE}.md"]})}

    rc = coherence_untrack.main(["--root", str(workspace), "docs/plan.md"])

    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert captured.out == f"agent-coherence-untrack: untracked docs/{_PRINTED}.md\n"
    assert captured.err == ""


def test_untrack_prints_a_strict_refusal_quoted_with_its_patterns_escaped(
    stub_coordinator, capsys: pytest.CaptureFixture[str]
) -> None:
    """A strict refusal's path prints in the quoted ``repr`` form and its strict
    patterns escaped, with the strict exit code. Prevents the refusal writing
    control characters to the terminal."""
    workspace, _ = stub_coordinator
    _StubCoordinator.answers = {"/policy/untrack": (409, {
        "reason": "untrack_strict_path",
        "refused": [{"path": f"p{_HOSTILE}", "strict_patterns": [f"g{_HOSTILE}", "data/**"]}],
    })}

    rc = coherence_untrack.main(["--root", str(workspace), "data/a.txt"])

    captured = capsys.readouterr()
    assert rc == 3, captured.err
    assert captured.err.splitlines()[0] == (
        f"agent-coherence-untrack: refused 'p{_PRINTED}': enforced in strict mode by g{_PRINTED}, data/**"
    )
    assert _raw_characters_in(captured.err) == []


def test_untrack_keeps_its_strict_exit_code_when_the_refusal_entries_are_malformed(
    stub_coordinator, capsys: pytest.CaptureFixture[str]
) -> None:
    """A strict refusal is classified by its typed reason, so entries of the
    wrong shape -- a bare string, patterns that are not a list -- still exit
    3 with the restart advice, printing what can be read and no traceback.
    Prevents a malformed refusal reaching exit 1, which means local
    validation."""
    workspace, _ = stub_coordinator
    _StubCoordinator.answers = {"/policy/untrack": (409, {
        "reason": "untrack_strict_path",
        "refused": ["data/a.txt", {"path": "data/b.txt", "strict_patterns": 5}],
    })}

    rc = coherence_untrack.main(["--root", str(workspace), "data/a.txt"])

    captured = capsys.readouterr()
    assert rc == 3, captured.err
    lines = captured.err.splitlines()
    assert lines[0] == "agent-coherence-untrack: refused 'data/b.txt': enforced in strict mode by ?"
    assert lines[1].startswith("agent-coherence-untrack: nothing was untracked.")
    assert len(lines) == 2


#: The commands the error path covers: each one's route, module and how it is
#: run against a workspace (a valid path, and a session for the handoff verb).
_ERROR_PATH_COMMANDS: dict[str, tuple[str, Any, Any]] = {
    "status": ("/status", coherence_status, lambda ws: coherence_status.main(["--root", str(ws)])),
    "track": ("/policy/track", coherence_track,
              lambda ws: coherence_track.main(["--root", str(ws), "docs/plan.md"])),
    "untrack": ("/policy/untrack", coherence_untrack,
                lambda ws: coherence_untrack.main(["--root", str(ws), "docs/plan.md"])),
    "withdraw": ("/handoff/withdraw", coherence_handoff, lambda ws: coherence_handoff.withdraw_main(
        ["--root", str(ws), "--session", str(uuid.uuid4()), "plan.md"])),
}


@pytest.mark.parametrize(("command", "answer"), [
    ("track", {"added": 5}),
    ("track", {"rejected": ["docs/plan.md"]}),
    ("untrack", {"removed": 5}),
], ids=["track-added-not-a-list", "track-rejected-entry-not-an-object", "untrack-removed-not-a-list"])
def test_a_policy_answer_with_fields_of_the_wrong_type_exits_2_with_one_line(
    command: str, answer: dict, stub_coordinator, capsys: pytest.CaptureFixture[str]
) -> None:
    """A track or untrack answer whose lists are not lists of the expected
    entries exits 2 with one line naming the answer's shape. Prevents a
    coordinator answer reaching exit 1 with a traceback."""
    workspace, _ = stub_coordinator
    _StubCoordinator.answers = {f"/policy/{command}": (200, answer)}

    rc = _ERROR_PATH_COMMANDS[command][2](workspace)

    captured = capsys.readouterr()
    assert rc == 2
    assert captured.err == f"agent-coherence-{command}: unexpected /policy/{command} answer shape\n"
    assert captured.out == ""


@pytest.mark.parametrize(("verb", "answer", "expected_rc", "expected_line"), [
    ("transfer", {"ok": True, "grants": [{
        "path": "plan.md", "transferred": True, "successor": f"s{_HOSTILE}",
        "version_at_transfer": f"v{_HOSTILE}", "hold_shape": f"h{_HOSTILE}", "status": f"st{_HOSTILE}",
    }]}, 0, (
        f"agent-coherence-transfer: transferred plan.md to s{_PRINTED} at version v{_PRINTED} "
        f"(gave up h{_PRINTED}; status st{_PRINTED})"
    )),
    ("transfer", {"ok": False, "grants": [{
        "path": "plan.md", "transferred": False, "reason": f"r{_HOSTILE}", "status": f"st{_HOSTILE}",
    }]}, 2, f"agent-coherence-transfer: plan.md not transferred (r{_PRINTED}, status st{_PRINTED})"),
    ("accept", {"ok": True, "status": f"st{_HOSTILE}", "counterparty": f"c{_HOSTILE}"}, 0, (
        f"agent-coherence-accept: accepted the handoff of plan.md "
        f"(status st{_PRINTED}, counterparty c{_PRINTED})"
    )),
    ("decline", {"ok": False, "reason": f"r{_HOSTILE}", "status": f"st{_HOSTILE}"}, 2, (
        f"agent-coherence-decline: plan.md: refused (r{_PRINTED}, status st{_PRINTED})"
    )),
    ("withdraw", {"ok": False, "degraded": True, "reason": f"r{_HOSTILE}"}, 2, (
        f"agent-coherence-withdraw: the coordinator could not confirm the withdraw (r{_PRINTED}); "
        "its outcome is unknown: check the path's handoff in agent-coherence-status before acting again"
    )),
], ids=["transferred", "not-transferred", "accepted", "declined-refused", "withdraw-unconfirmed"])
def test_a_handoff_verb_prints_the_answers_fields_escaped(
    verb: str, answer: dict, expected_rc: int, expected_line: str,
    stub_coordinator, capsys: pytest.CaptureFixture[str],
) -> None:
    """Each field a handoff verb prints from the coordinator's answer -- the
    successor, version, hold shape, status, counterparty and reason -- prints
    escaped, on stdout for a verb taken and stderr for one refused or
    unconfirmed. Prevents a coordinator answer writing control characters to
    the terminal through a handoff verb."""
    workspace, _ = stub_coordinator
    _StubCoordinator.answers = {f"/handoff/{verb}": (200, answer)}

    rc = _VERB_RUNS[verb](["--root", str(workspace), "--session", str(uuid.uuid4()), "plan.md"])

    captured = capsys.readouterr()
    assert rc == expected_rc, captured.err
    assert expected_line in (captured.out + captured.err).splitlines()
    assert _raw_characters_in(captured.out + captured.err) == []


#: Answers every command meets with exit 2 and one line: (status, body[,
#: reason phrase]) and the line after the command's name.
_ONE_LINE_FAILURES = {
    "error-body-a-list": ((500, ["x"]), "HTTP 500"),
    "error-body-a-string": ((500, "x"), "HTTP 500"),
    "error-body-a-number": ((500, 7), "HTTP 500"),
    "error-body-null": ((500, None), "HTTP 500"),
    "error-body-not-json-hostile-reason-phrase": ((500, b"<html>oops</html>", "Bad\x1b[31m"), "HTTP 500"),
    "error-string-with-esc-and-lf": ((500, {"error": "boom\x1b[31m\nsecond"}), r"HTTP 500: boom\x1b[31m\nsecond"),
    "redirect": ((302, {}), "the coordinator redirected the request (HTTP 302); not followed"),
    "ok-body-a-list": ((200, ["x"]), "the coordinator's answer is not a JSON object"),
}


def _assert_one_failure_line(command: str, captured: Any) -> None:
    """Exactly one stderr line from the command; on stdout nothing, or for the
    handoff verb only the line naming the session it acted as."""
    assert captured.err.count("\n") == 1, captured.err
    assert "Traceback" not in captured.err
    if command == "withdraw":
        assert captured.out.startswith("agent-coherence-withdraw: acting as session agent ")
        assert captured.out.count("\n") == 1
    else:
        assert captured.out == ""


@pytest.mark.parametrize("failure", sorted(_ONE_LINE_FAILURES))
@pytest.mark.parametrize("command", sorted(_ERROR_PATH_COMMANDS))
def test_a_coordinator_failure_exits_2_with_one_line(
    command: str, failure: str, stub_coordinator, capsys: pytest.CaptureFixture[str]
) -> None:
    """An error body that is not a JSON object (a list, string, number, null
    or not JSON at all), an error string carrying ESC and LF, a redirect and a
    200 body that is not an object each make every command exit 2 with one
    line: ``HTTP {code}``, plus the escaped ``error`` only when it is a string
    -- never the status line's reason phrase, which is the server's text too.
    Prevents a coordinator answer reaching exit 1 with a traceback, or a
    terminal escape reaching the screen through an error line."""
    workspace, _ = stub_coordinator
    route, _module, run = _ERROR_PATH_COMMANDS[command]
    answer, expected = _ONE_LINE_FAILURES[failure]
    _StubCoordinator.answers = {route: answer}

    rc = run(workspace)

    captured = capsys.readouterr()
    assert rc == 2, captured.err
    assert captured.err == f"agent-coherence-{command}: {expected}\n"
    _assert_one_failure_line(command, captured)


@pytest.mark.parametrize("command", sorted(_ERROR_PATH_COMMANDS))
def test_a_tls_failure_exits_2_with_one_line(
    command: str, stub_coordinator, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A TLS failure exits 2 with one line. The commands resolve a loopback
    http endpoint today, so each is handed an https one whose CA bundle does
    not exist, and the client's own TLS factory refuses it before any byte is
    sent. Prevents a TLS refusal reaching exit 1 with a traceback."""
    workspace, port = stub_coordinator
    _route, module, run = _ERROR_PATH_COMMANDS[command]
    https = CoordinatorEndpoint(
        port=port, bearer="test-secret", scheme="https", ca_file=str(tmp_path / "missing-ca.pem"),
    )
    monkeypatch.setattr(module, "resolve_endpoint", lambda _root: https)

    rc = run(workspace)

    captured = capsys.readouterr()
    assert rc == 2, captured.err
    assert captured.err.startswith(f"agent-coherence-{command}: CCS_REMOTE_CA_FILE ")
    _assert_one_failure_line(command, captured)
    assert _StubCoordinator.seen == []


@pytest.mark.parametrize("change", [
    {"coordinator_uptime_seconds": "12"},
    {"sessions": [{
        "agent_id": _GIVER_AGENT, "agent_name": "claude-session-x", "states": {"spec.md": "SHARED"},
        "reclaimed": {"plan.md": "reclaim_heartbeat"},
    }]},
], ids=["uptime-not-a-number", "reclaim-cause-not-an-object"])
def test_a_status_body_with_fields_of_the_wrong_type_exits_2_with_one_line(
    change: dict, stub_coordinator, capsys: pytest.CaptureFixture[str]
) -> None:
    """A ``/status`` body whose fields have the wrong types exits 2 with
    "unexpected /status shape" and prints none of the table -- not the half
    rendered before the bad field. Prevents a malformed answer reaching exit 1
    with a traceback, or a partial table reading as the workspace's state."""
    workspace, _ = stub_coordinator
    _StubCoordinator.answers = {"/status": (200, {**_status_payload(), **change})}

    rc = coherence_status.main(["--root", str(workspace)])

    captured = capsys.readouterr()
    assert rc == 2
    assert captured.err == "agent-coherence-status: unexpected /status shape\n"
    assert captured.out == ""


def test_detail_help_does_not_claim_the_minimal_tier_redacts_every_absolute_path() -> None:
    """``policy_summary.coordinator_root`` is an absolute path at the minimal
    tier as at the full one, so the ``--detail`` help names it rather than
    claiming the minimal tier redacts absolute paths. Prevents an operator
    pasting a minimal-tier ``--json`` body as free of their directory
    layout."""
    action = next(a for a in coherence_status.build_parser()._actions if "--detail" in a.option_strings)

    assert "redacts absolute paths" not in action.help
    assert "policy_summary.coordinator_root" in action.help


# ----------------------------------------------------------------------
# #238 -- a degraded /status answer is "cannot tell" (exit 2)
# ----------------------------------------------------------------------

#: FROZEN duplicate of the line the status table prints on a degraded answer.
_DEGRADED_STATUS_LINE = (
    "agent-coherence-status: the coordinator's registry is busy (lock contention), "
    "so tracked artifacts and sessions are unavailable; try again shortly\n"
)


def _degraded_status_body() -> dict:
    """An operator-tier ``/status`` answer whose registry read timed out: the
    normal key set, with ``policy_summary`` (counts and a user-added pattern)
    and the counters, both lists null, and ``degraded: true``."""
    return {
        **_status_payload(),
        "policy_summary": {
            "default_pattern_count": 3, "user_added_pattern_count": 1, "ignored_pattern_count": 0,
            "user_added_patterns": ["notes/**"],
        },
        "tracked_artifacts": None,
        "sessions": None,
        "endpoint_counters": {"pre_read_total": 4, "status_total": 2},
        "watchdog_timeouts_total": 1,
        "degraded": True,
    }


@pytest.mark.parametrize("flags", [[], ["--show-policy"], ["--detail", "minimal"]],
                         ids=["table", "show-policy", "minimal"])
def test_status_table_on_a_degraded_answer_exits_2_with_one_line_naming_registry_contention(
    flags: list[str], stub_coordinator, capsys: pytest.CaptureFixture[str]
) -> None:
    """A degraded ``/status`` -- the coordinator could not read its registry in
    time, so both lists are null -- prints nothing on stdout and one stderr
    line naming registry contention, and exits 2, with or without
    ``--show-policy``. Prevents a busy registry reading as an empty workspace
    ("No artifacts observed yet", "No active sessions.") or as a malformed
    answer."""
    workspace, _ = stub_coordinator
    _StubCoordinator.answers = {"/status": (200, _degraded_status_body())}

    rc = coherence_status.main(["--root", str(workspace), *flags])

    captured = capsys.readouterr()
    assert rc == 2
    assert captured.err == _DEGRADED_STATUS_LINE
    assert captured.out == ""
    assert "No active sessions." not in captured.out
    assert "No artifacts observed yet" not in captured.out


@pytest.mark.parametrize("flags", [["--json"], ["--json", "--show-policy"]], ids=["json", "json-show-policy"])
def test_status_json_on_a_degraded_answer_prints_the_body_unchanged_and_exits_2(
    flags: list[str], stub_coordinator, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--json`` prints a degraded body as the coordinator sent it and exits 2;
    with ``--show-policy`` it adds no ``policy_pending_first_read``, which
    cannot be computed without the artifact list. Prevents a script reading a
    degraded answer as a successful one, or a pending list computed against
    no observed paths."""
    workspace, _ = stub_coordinator
    body = _degraded_status_body()
    _StubCoordinator.answers = {"/status": (200, body)}

    rc = coherence_status.main(["--root", str(workspace), *flags])

    captured = capsys.readouterr()
    assert rc == 2
    assert json.loads(captured.out) == body
    assert "policy_pending_first_read" not in captured.out
    assert captured.err == ""


def test_status_table_on_an_answer_without_its_lists_and_no_marker_exits_2_as_a_bad_shape(
    stub_coordinator, capsys: pytest.CaptureFixture[str]
) -> None:
    """The lists not being lists is what makes an answer "cannot tell"; the
    ``degraded`` marker only picks the message. An answer that omits both
    lists without the marker exits 2 with the shape line and prints no table.
    Prevents an answer with no lists rendering as an empty workspace."""
    workspace, _ = stub_coordinator
    body = {k: v for k, v in _degraded_status_body().items()
            if k not in ("tracked_artifacts", "sessions", "degraded")}
    _StubCoordinator.answers = {"/status": (200, body)}

    rc = coherence_status.main(["--root", str(workspace)])

    captured = capsys.readouterr()
    assert rc == 2
    assert captured.err == "agent-coherence-status: unexpected /status shape\n"
    assert captured.out == ""
