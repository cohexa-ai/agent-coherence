# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""``CoherentVolume(on_held="refuse")``: a write that will not take a file
another agent holds.

The volume opts in on ``write()`` for a managed path and types each answer. It
attaches only to a coordinator that advertises the refusal. These tests run the
real volume against a real coordinator in this process and stub only true
boundaries: an operator view without the key, which every released build
answers, and answers no current coordinator sends but an older build does. They
cannot run an older build itself, so what one does is stated in the guide, not
proved here.
"""

from __future__ import annotations

import io
import json
import os
import urllib.error
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

import ccs.adapters.coherent_volume as coherent_volume_module
import ccs.cli._coherence_client as client_module
from ccs.adapters.claude_code.coordinator_server import CoordinatorHTTPServer
from ccs.adapters.claude_code.lifecycle import LifecycleConfig
from ccs.adapters.coherent_volume import CoherentVolume
from ccs.cli._coherence_client import CoordinatorEndpoint
from ccs.core.exceptions import CoherenceError
from tests.coordinator_helpers import serve_in_process

_MANAGED = ("data/**",)
_PATH = "data/shared.txt"
_CLAIM_ROUTE = client_module.PRINCIPAL_CLAIM_ROUTE


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
    running = serve_in_process(tmp_path, "refuse-mode")
    yield running
    running.shutdown()


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict]]:
    """Every POST the volume sends as ``(route, body)``, the principal claim
    included, each forwarded to the coordinator. The claim goes out through the
    client module, every other request through the volume's own alias."""
    record: list[tuple[str, dict]] = []
    real_post = client_module.post

    def spy(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
        record.append((path, dict(payload)))
        return real_post(endpoint, path, payload, **kwargs)

    monkeypatch.setattr(client_module, "post", spy)
    monkeypatch.setattr(coherent_volume_module, "_coordinator_post", spy)
    return record


def _routes(sent: list[tuple[str, dict]]) -> list[str]:
    return [route for route, _ in sent]


def _refuser(root: Path, cfg: LifecycleConfig, **kwargs: object) -> CoherentVolume:
    return CoherentVolume(root, managed=_MANAGED, config=cfg, on_held="refuse", **kwargs)


def _operator_view_changed_to(
    monkeypatch: pytest.MonkeyPatch, change: Callable[[dict], dict]
) -> None:
    """Pass every operator view through ``change``; every other GET is untouched."""
    real_get = coherent_volume_module._coordinator_get

    def view(endpoint: object, path: str, **kwargs: object) -> object:
        doc = real_get(endpoint, path, **kwargs)
        return change(doc) if "detail=full" in path else doc

    monkeypatch.setattr(coherent_volume_module, "_coordinator_get", view)


def _without_the_key(doc: dict) -> dict:
    return {k: v for k, v in doc.items() if k != "pre_edit_if_unheld"}


# --- what a refuse volume needs -------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"on_held": "bogus"},
        {"on_held": None},
        {"on_held": "refuse", "on_error": "degrade"},
        {"on_held": "refuse", "managed": ()},
        {
            "on_held": "refuse",
            "remote_endpoint": CoordinatorEndpoint(port=1, bearer="not-a-real-token"),
        },
    ],
    ids=["unknown-value", "none", "degrade", "no-managed-globs", "remote"],
)
def test_a_volume_that_cannot_honour_refuse_mode_is_never_built(
    tmp_path: Path, fast_cfg: LifecycleConfig, kwargs: dict
) -> None:
    """Refuse mode opts in on managed paths and must fail closed, so a value it
    cannot honour raises before anything is spawned. Each message names
    ``on_held``: without its own check a remote endpoint would still raise, from
    the remote block, and tell the caller to set an environment flag.

    Prevents a volume built with ``on_held="refuse"`` that never opts in and
    silently displaces every holder, and one that writes best-effort detached."""
    built: dict = {"managed": _MANAGED, "config": fast_cfg, **kwargs}
    with pytest.raises(ValueError, match="on_held"):
        CoherentVolume(tmp_path, **built)
    assert not (tmp_path / ".coherence").exists()


# --- the capability check at attach ---------------------------------------


@pytest.mark.parametrize(
    "change",
    [
        _without_the_key,
        lambda doc: {**doc, "pre_edit_if_unheld": "true"},
        lambda doc: {**doc, "pre_edit_if_unheld": 1},
        lambda doc: {**doc, "pre_edit_if_unheld": None},
        lambda doc: {**doc, "pre_edit_if_unheld": False},
    ],
    ids=["absent", "string", "one", "null", "false"],
)
def test_a_coordinator_that_does_not_advertise_the_refusal_is_refused_at_attach(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    sent: list[tuple[str, dict]],
    monkeypatch: pytest.MonkeyPatch,
    change: Callable[[dict], dict],
) -> None:
    """Anything but JSON ``true`` means the coordinator may answer an opted-in
    request by taking the file from its holder, so the volume fails closed
    before it claims a principal or sends a pre-edit. The error is the plain one
    a glob the coordinator does not enforce raises.

    Prevents a refuse-mode write landing over a holder on an older coordinator,
    which a version number cannot predict."""
    _operator_view_changed_to(monkeypatch, change)
    with pytest.raises(CoherenceError, match="pre_edit_if_unheld") as refused:
        _refuser(tmp_path, fast_cfg)
    assert type(refused.value) is CoherenceError
    assert sent == [], "the check runs before the claim and before any pre-edit"


def test_a_refuse_volume_adds_no_request_to_attach(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    sent: list[tuple[str, dict]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The key is read from the operator view the glob check already fetched.

    Prevents the check costing a second status read on every attach."""
    views: list[str] = []

    def counted(doc: dict) -> dict:
        views.append("full")
        return doc

    _operator_view_changed_to(monkeypatch, counted)
    volume = _refuser(tmp_path, fast_cfg)
    assert volume.is_attached
    assert len(views) == 1
    assert _routes(sent) == [_CLAIM_ROUTE]


def test_a_displace_volume_attaches_to_a_coordinator_without_the_key(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only refuse mode reads the key, so an ordinary volume attaches to an
    older coordinator as it always has."""
    _operator_view_changed_to(monkeypatch, _without_the_key)
    assert CoherentVolume(tmp_path, managed=_MANAGED, config=fast_cfg).is_attached


def test_a_forked_child_checks_the_coordinator_again(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    sent: list[tuple[str, dict]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A forked child attaches afresh on its first write, so it reads the view
    again and refuses a coordinator that no longer advertises the refusal.

    Prevents a child trusting what its parent read about another coordinator."""
    volume = _refuser(tmp_path, fast_cfg)
    _operator_view_changed_to(monkeypatch, _without_the_key)
    sent.clear()

    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # child: report through the pipe, never return into pytest
        os.close(read_fd)
        report: dict = {"raised": None}
        try:
            volume.write(_PATH, b"from-the-child")
        except BaseException as exc:  # noqa: BLE001 - reported to the parent
            report = {"raised": type(exc).__name__, "message": str(exc)}
        report["routes"] = _routes(sent)
        os.write(write_fd, json.dumps(report).encode("utf-8"))
        os.close(write_fd)
        os._exit(0)
    os.close(write_fd)
    with os.fdopen(read_fd, "rb") as pipe:
        report = json.loads(pipe.read().decode("utf-8"))
    os.waitpid(pid, 0)

    assert report["raised"] == "CoherenceError", report
    assert "pre_edit_if_unheld" in report["message"], report
    assert "/hooks/pre-edit" not in report["routes"], report
    assert not (tmp_path / _PATH).exists()


def test_a_refused_operator_view_keeps_its_own_message(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    sent: list[tuple[str, dict]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Node coordinator answers 501 to the operator view, so a managed
    volume cannot attach to it in either mode. The glob check reports that
    first, naming the status, and the capability check does not cover it up.

    Prevents a caller on Node being told to upgrade a coordinator that honours
    the flag, when what is missing is the operator view."""
    real_get = coherent_volume_module._coordinator_get

    def not_served(endpoint: object, path: str, **kwargs: object) -> object:
        if "detail=full" in path:
            raise urllib.error.HTTPError(path, 501, "not served", {}, io.BytesIO(b"{}"))  # type: ignore[arg-type]
        return real_get(endpoint, path, **kwargs)

    monkeypatch.setattr(coherent_volume_module, "_coordinator_get", not_served)
    with pytest.raises(CoherenceError, match="HTTP 501") as refused:
        _refuser(tmp_path, fast_cfg)
    assert "pre_edit_if_unheld" not in str(refused.value)
    assert sent == []
