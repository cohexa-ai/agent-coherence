# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Which answer a fold row asserts: the one for the disk under its workspace.

A fixture whose answer depends on how the workspace disk folds file names
carries ``expected_by_fold_profile``. The harness classifies the scenario's own
tmp workspace with the coordinators' probe and asserts that profile's entry,
with no fallback: a profile the fixture does not name fails the row, and a
disk the probe cannot classify is ``cannot_tell``, never ``none``.

Not marked ``protocol_corpus``: none of this starts a coordinator, so it runs
in the default suite."""

from __future__ import annotations

import copy
import os
import shutil
from pathlib import Path
from typing import Any, Iterator

import pytest

from tests.protocol_corpus import harness
from tests.protocol_corpus.harness import (
    FOLD_PROFILES,
    PRINCIPAL_SENTINEL,
    FixtureContractError,
    FoldProfileNotExpected,
    build_fixture,
    expected_for_fold_profile,
    expected_for_workspace,
    workspace_fold_profile,
)

_PATH = Path("fold-under-test.json")


def _entry(marker: str) -> dict[str, Any]:
    return {"status": 200, "body": {"ok": True, "marker": marker}}


def _by_profile(*profiles: str) -> dict[str, Any]:
    return {profile: _entry(profile) for profile in profiles}


def _fixture_data(**fields: Any) -> dict[str, Any]:
    return {
        "name": "fold-under-test",
        "request": {"method": "POST", "path": "/hooks/pre-edit", "body": {}},
        **fields,
    }


def _mapped(*profiles: str) -> harness.Fixture:
    data = _fixture_data(expected_by_fold_profile=_by_profile(*profiles))
    return build_fixture(data, _PATH)


@pytest.fixture
def unclassifiable(tmp_path: Path) -> Iterator[Path]:
    """A workspace whose probe cannot tell: a stale probe directory holding a
    directory, which the sweep leaves in place rather than descend into.
    Removed afterwards, so the suite's check for probe leftovers sees none."""
    workspace = tmp_path / "unclassifiable"
    stale = workspace / ".coherence" / "fold-probe-stale"
    (stale / "planted").mkdir(parents=True)
    yield workspace
    shutil.rmtree(stale)


def test_a_fixture_without_a_map_asserts_its_one_expectation_on_every_disk() -> None:
    fixture = build_fixture(_fixture_data(expected=_entry("one")), _PATH)

    assert fixture.expected_by_fold_profile is None
    for profile in FOLD_PROFILES:
        assert expected_for_fold_profile(fixture, profile) == _entry("one")


def test_each_profile_asserts_its_own_entry() -> None:
    fixture = _mapped("caseless", "nfd", "none", "cannot_tell")

    assert fixture.expected is None
    for profile in FOLD_PROFILES:
        assert expected_for_fold_profile(fixture, profile) == _entry(profile)


def test_a_profile_the_map_does_not_name_fails_the_row() -> None:
    """Loud: an error naming the fixture and the profile, never a fallback to
    another disk's answer and never a skip."""
    fixture = _mapped("caseless", "nfd", "none")

    with pytest.raises(FoldProfileNotExpected) as refused:
        expected_for_fold_profile(fixture, "cannot_tell")

    message = str(refused.value)
    assert "fold-under-test" in message and "'cannot_tell'" in message, message
    assert isinstance(refused.value, FixtureContractError)


@pytest.mark.parametrize("fields", [
    {"expected": _entry("one"), "expected_by_fold_profile": _by_profile("caseless", "nfd", "none")},
    {},
], ids=["both", "neither"])
def test_a_fixture_with_both_or_neither_expectation_is_refused_at_load(fields: dict[str, Any]) -> None:
    """Both is not read as "the map, else the default": a default would answer
    for a profile the map forgot."""
    with pytest.raises(FixtureContractError, match="'expected' and 'expected_by_fold_profile'"):
        build_fixture(_fixture_data(**fields), _PATH)


@pytest.mark.parametrize("profiles", [
    ("caseless", "nfd"),
    ("caseless", "none"),
    ("nfd", "none"),
    ("caseless", "nfd", "none", "caseles"),
], ids=["no-none", "no-nfd", "no-caseless", "misspelled"])
def test_a_map_without_every_disk_profile_or_with_an_unknown_one_is_refused_at_load(
    profiles: tuple[str, ...],
) -> None:
    with pytest.raises(FixtureContractError, match="expected_by_fold_profile names"):
        _mapped(*profiles)


@pytest.mark.parametrize("entry", [
    {"body": {}},
    {"status": "200", "body": {}},
    {"status": 200},
    [200, {}],
], ids=["no-status", "status-not-int", "no-body", "not-an-object"])
def test_a_map_entry_that_is_not_status_and_body_is_refused_at_load(entry: Any) -> None:
    by_profile = {**_by_profile("caseless", "nfd"), "none": entry}
    with pytest.raises(FixtureContractError, match=r"entries \['none'\]"):
        build_fixture(_fixture_data(expected_by_fold_profile=by_profile), _PATH)


def test_a_map_that_is_not_an_object_is_refused_at_load() -> None:
    with pytest.raises(FixtureContractError, match="object keyed by fold profile"):
        build_fixture(_fixture_data(expected_by_fold_profile=[_entry("caseless")]), _PATH)


def test_a_principal_in_any_profile_entry_is_refused_at_load() -> None:
    """The no-principal rule reads every entry, not only the first."""
    by_profile = _by_profile("caseless", "nfd", "none")
    by_profile["nfd"] = {"status": 200, "body": {"principal": "p-not-a-sentinel"}}
    with pytest.raises(FixtureContractError, match="carries a principal"):
        build_fixture(_fixture_data(expected_by_fold_profile=by_profile), _PATH)

    by_profile["nfd"] = {"status": 200, "body": {"principal": PRINCIPAL_SENTINEL}}
    build_fixture(_fixture_data(expected_by_fold_profile=by_profile), _PATH)


def test_an_identity_opt_in_must_be_asserted_by_every_profile_entry() -> None:
    """A key preserved for one disk's answer and absent from another's would
    pin an identity on one disk only."""
    by_profile = {p: {"status": 200, "body": {"holder": f"holder-{p}"}} for p in ("caseless", "nfd", "none")}
    build_fixture(_fixture_data(expected_by_fold_profile=by_profile, preserve_identity=["holder"]), _PATH)

    partial = copy.deepcopy(by_profile)
    partial["none"] = {"status": 200, "body": {"ok": True}}
    with pytest.raises(FixtureContractError, match="does not carry"):
        build_fixture(_fixture_data(expected_by_fold_profile=partial, preserve_identity=["holder"]), _PATH)


def _folds(workspace: Path, created: str, probed: str) -> bool:
    """Whether the disk under ``workspace`` opens ``probed`` as ``created``;
    an observation made without the library's probe."""
    target = workspace / created
    target.write_text("")
    try:
        other = workspace / probed
        return other.exists() and os.path.samefile(target, other)
    finally:
        target.unlink()


def test_the_workspace_profile_is_what_its_disk_folds_and_no_probe_file_is_left(tmp_path: Path) -> None:
    """Checked against two direct observations of the same disk rather than
    the probe's own answer: it folds case only on ``caseless``, and composed
    against decomposed accents on ``caseless`` and ``nfd``."""
    observed = tmp_path / "observed"
    observed.mkdir()
    folds_case = _folds(observed, "case.txt", "CASE.TXT")
    folds_accents = _folds(observed, "caf\u00e9.txt", "cafe\u0301.txt")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    profile = workspace_fold_profile(workspace)

    assert profile in FOLD_PROFILES
    assert (profile == "caseless") == folds_case, (profile, folds_case)
    assert (profile in ("caseless", "nfd")) == folds_accents, (profile, folds_accents)
    assert os.listdir(workspace / ".coherence") == []


def test_a_disk_the_probe_cannot_classify_is_cannot_tell_not_none(unclassifiable: Path) -> None:
    """``/status`` says ``cannot_tell`` there, so a fixture's ``none`` answer
    must not be asserted on it."""
    assert workspace_fold_profile(unclassifiable) == "cannot_tell"


def test_the_workspace_picks_the_entry_and_an_unnamed_profile_fails(
    tmp_path: Path, unclassifiable: Path
) -> None:
    classified = tmp_path / "classified"
    classified.mkdir()
    fixture = _mapped("caseless", "nfd", "none")

    assert expected_for_workspace(fixture, classified) == _entry(workspace_fold_profile(classified))

    with pytest.raises(FoldProfileNotExpected):
        expected_for_workspace(fixture, unclassifiable)
    with_cannot_tell = _mapped("caseless", "nfd", "none", "cannot_tell")
    assert expected_for_workspace(with_cannot_tell, unclassifiable) == _entry("cannot_tell")
