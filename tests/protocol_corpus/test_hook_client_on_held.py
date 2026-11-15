# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The hook client's refusal of held strict files (#519), and a stale strict
reader reading the file again after one deny (#514), against the real Node
coordinator.

The rows are the ones ``tests/test_claude_code_e2e.py`` runs against the
Python coordinator, written once there: the real hook-client process, one per
event, on a workspace whose Node coordinator this test starts, so the Node
coordinator is held to the same deny texts. Each row runs both hook clients,
this checkout's Python one and the plugin's Node one, and compares what they
print parsed. The killed-coordinator row starts its Node coordinator as its own
process, so that it can kill it rather than stop it. Three rows run there
only: a caller principal refused beyond recovery (the Node coordinator issues
none), an edit the coordinator cannot decide in time (its pre-edit has no
watchdog), and an MCP server as the holder (the corpus jobs install the
package without the mcp extra).

Marked ``protocol_corpus`` -- opt-in via ``pytest -m protocol_corpus``, in the
job that builds the paired plugin branch. Without a resolvable plugin dist
every test xfails, as the Node rows of the parity corpora do."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from ccs.cli._coherence_client import get, resolve_endpoint
from tests.protocol_corpus.harness import (
    BACKEND_NODE,
    coordinator_running,
    resolve_node_dist_path,
)
from tests.test_claude_code_e2e import (
    NOTES,
    PLAN,
    check_a_change_outside_the_coordinator_after_a_deny_stays_denied,
    check_a_held_strict_file_is_refused_naming_its_holder,
    check_a_holder_that_stops_without_committing_frees_the_file,
    check_a_missing_file_after_a_deny_stays_denied,
    check_a_parent_recovers_after_its_subagent_commits,
    check_a_reader_recovers_after_a_grant_taken_and_given_back_uncommitted,
    check_a_refuse_mode_volume_holding_a_strict_file_is_named,
    check_a_refused_reader_recovers_after_the_holder_commits,
    check_a_session_that_never_read_a_changed_file_stays_denied,
    check_a_sibling_subagent_is_refused_naming_the_holding_subagent,
    check_a_stale_editor_recovers_by_reading,
    check_a_stale_reader_recovers_after_a_peer_commit,
    check_a_subagent_refused_by_its_parents_hold_is_told_to_report_back,
    check_an_edit_after_the_coordinator_is_killed_is_denied,
    check_an_edit_of_a_free_strict_file_is_admitted,
    check_an_edit_of_a_held_file_takes_it_as_today,
    check_an_unknown_setting_takes_a_held_strict_file_as_today,
    check_plain_volume_reads_after_a_peer_commit_stay_denied,
    coordinator_process,
    hook_client,
    set_hold_refusal,
    write_strict_workspace,
)

pytestmark = pytest.mark.protocol_corpus

_NODE_DIST_PATH = resolve_node_dist_path()
_CLIENTS = [pytest.param("python", id="python-client"), pytest.param("node", id="node-client")]
_UNRESOLVABLE = (
    "Node coordinator dist not resolvable. Build the plugin checkout "
    "(npm ci && npm run build) or set AGENT_COHERENCE_PLUGIN_DIST_PATH."
)


@pytest.fixture
def node_workspace(tmp_path: Path) -> Iterator[Path]:
    """``tmp_path`` with the setting on and the strict policy, served by the
    Node coordinator."""
    if _NODE_DIST_PATH is None:
        pytest.xfail(_UNRESOLVABLE)
    write_strict_workspace(tmp_path)
    with coordinator_running(BACKEND_NODE, tmp_path, _NODE_DIST_PATH):
        assert get(resolve_endpoint(tmp_path), "/status")["backend"] == "node"
        yield tmp_path


@pytest.mark.parametrize("client", _CLIENTS)
def test_an_edit_of_a_strict_file_nobody_holds_is_admitted(node_workspace: Path, client: str) -> None:
    check_an_edit_of_a_free_strict_file_is_admitted(node_workspace, hook_client(client))


@pytest.mark.parametrize("client", _CLIENTS)
@pytest.mark.parametrize("editor_read", [False, True], ids=["editor-never-read", "editor-read-during-the-hold"])
def test_an_edit_of_a_strict_file_another_session_holds_is_denied_naming_the_holder(
    node_workspace: Path, editor_read: bool, client: str,
) -> None:
    check_a_held_strict_file_is_refused_naming_its_holder(node_workspace, hook_client(client), editor_read)


@pytest.mark.parametrize("client", _CLIENTS)
def test_a_holder_that_stops_without_committing_frees_the_file_for_the_refused_editor(
    node_workspace: Path, client: str,
) -> None:
    check_a_holder_that_stops_without_committing_frees_the_file(node_workspace, hook_client(client))


@pytest.mark.parametrize("client", _CLIENTS)
def test_an_edit_of_a_held_file_that_is_not_strict_takes_it_as_today(node_workspace: Path, client: str) -> None:
    check_an_edit_of_a_held_file_takes_it_as_today(node_workspace, hook_client(client), NOTES)


@pytest.mark.parametrize("client", _CLIENTS)
def test_a_subagent_refused_by_its_own_parents_hold_is_told_to_report_back(node_workspace: Path, client: str) -> None:
    check_a_subagent_refused_by_its_parents_hold_is_told_to_report_back(node_workspace, hook_client(client))


@pytest.mark.parametrize("client", _CLIENTS)
def test_a_sibling_subagent_is_refused_naming_the_subagent_that_holds_the_file(
    node_workspace: Path, client: str,
) -> None:
    check_a_sibling_subagent_is_refused_naming_the_holding_subagent(node_workspace, hook_client(client))


@pytest.mark.parametrize("client", _CLIENTS)
@pytest.mark.parametrize("setting", [None, "displace\n"], ids=["setting-absent", "setting-displace"])
def test_with_the_setting_off_an_edit_of_a_held_strict_file_takes_it_as_today(
    node_workspace: Path, setting: str | None, client: str,
) -> None:
    set_hold_refusal(node_workspace, setting)
    check_an_edit_of_a_held_file_takes_it_as_today(node_workspace, hook_client(client), PLAN)


@pytest.mark.parametrize("client", _CLIENTS)
def test_an_edit_after_the_coordinator_is_killed_is_denied(tmp_path: Path, client: str) -> None:
    if _NODE_DIST_PATH is None:
        pytest.xfail(_UNRESOLVABLE)
    hook = hook_client(client)
    write_strict_workspace(tmp_path)
    with coordinator_process(tmp_path, ("node", str(_NODE_DIST_PATH))) as coordinator:
        assert get(resolve_endpoint(tmp_path), "/status")["backend"] == "node"
        check_an_edit_after_the_coordinator_is_killed_is_denied(tmp_path, hook, coordinator)


@pytest.mark.parametrize("later", _CLIENTS)
@pytest.mark.parametrize("first", _CLIENTS)
def test_a_refused_reader_reads_again_and_edits_after_the_holder_commits(
    node_workspace: Path, first: str, later: str,
) -> None:
    check_a_refused_reader_recovers_after_the_holder_commits(node_workspace, hook_client(first), hook_client(later))


@pytest.mark.parametrize("client", _CLIENTS)
def test_with_an_unknown_setting_an_edit_of_a_held_strict_file_takes_it_as_today(
    node_workspace: Path, client: str,
) -> None:
    check_an_unknown_setting_takes_a_held_strict_file_as_today(node_workspace, hook_client(client))


@pytest.mark.parametrize("client", _CLIENTS)
def test_a_stale_reader_reads_again_and_edits_after_a_peer_commit(node_workspace: Path, client: str) -> None:
    check_a_stale_reader_recovers_after_a_peer_commit(node_workspace, hook_client(client))


@pytest.mark.parametrize("client", _CLIENTS)
def test_a_stale_editor_reads_again_and_edits(node_workspace: Path, client: str) -> None:
    check_a_stale_editor_recovers_by_reading(node_workspace, hook_client(client))


@pytest.mark.parametrize("client", _CLIENTS)
def test_a_parent_reads_again_and_edits_after_its_subagent_commits(node_workspace: Path, client: str) -> None:
    check_a_parent_recovers_after_its_subagent_commits(node_workspace, hook_client(client))


@pytest.mark.parametrize("client", _CLIENTS)
def test_a_reader_reads_again_and_edits_after_a_grant_given_back_uncommitted(
    node_workspace: Path, client: str,
) -> None:
    check_a_reader_recovers_after_a_grant_taken_and_given_back_uncommitted(node_workspace, hook_client(client))


@pytest.mark.parametrize("client", _CLIENTS)
def test_a_change_outside_the_coordinator_after_a_deny_stays_denied(node_workspace: Path, client: str) -> None:
    check_a_change_outside_the_coordinator_after_a_deny_stays_denied(node_workspace, hook_client(client))


@pytest.mark.parametrize("client", _CLIENTS)
def test_a_missing_file_after_a_deny_stays_denied(node_workspace: Path, client: str) -> None:
    check_a_missing_file_after_a_deny_stays_denied(node_workspace, hook_client(client))


@pytest.mark.parametrize("client", _CLIENTS)
def test_a_session_that_never_read_a_changed_file_stays_denied(node_workspace: Path, client: str) -> None:
    check_a_session_that_never_read_a_changed_file_stays_denied(node_workspace, hook_client(client))


def test_plain_volume_reads_after_a_peer_commit_stay_denied(node_workspace: Path) -> None:
    check_plain_volume_reads_after_a_peer_commit_stay_denied(node_workspace)


@pytest.mark.parametrize("client", _CLIENTS)
def test_an_edit_of_a_strict_file_a_refuse_mode_volume_holds_is_denied_naming_it(
    node_workspace: Path, client: str,
) -> None:
    check_a_refuse_mode_volume_holding_a_strict_file_is_named(node_workspace, hook_client(client))
