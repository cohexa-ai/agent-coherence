# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The plugin's copy of the held-file refusal fixtures is the library's, byte for byte.

``tests/fixtures/hook_on_held/`` is the contract both hook clients are held to
when ``.coherence/hook_on_held`` asks them to refuse edits of held strict files:
the deny texts, the decision table, the setting's value vectors and the two
stderr lines. The library is their only author. The plugin copies the JSON
files into ``src/test/fixtures/hook_on_held/`` beside a provenance note, and
its own tests hold the Node hook client to that copy, so a copy that drifted
from the library's would let the two clients disagree while both suites pass.
This compares the copy with this checkout's files: the same JSON file names on
both sides, each equal byte for byte. The provenance note is not a fixture and
is left out.

Marked ``protocol_corpus`` -- opt-in via ``pytest -m protocol_corpus``, in the
job that checks out the paired plugin branch. The plugin checkout compared is
the one whose ``dist/coordinator.js`` the corpus resolves. With none resolvable
the tests skip and say why; a resolved checkout that has no copy fails."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tests.protocol_corpus.harness import REPO_ROOT, resolve_node_dist_path

pytestmark = pytest.mark.protocol_corpus

_LIBRARY_FIXTURES = REPO_ROOT / "tests" / "fixtures" / "hook_on_held"
#: Where the plugin keeps its copy, relative to the root of its checkout.
_PLUGIN_COPY = Path("src", "test", "fixtures", "hook_on_held")
_NODE_DIST_PATH = resolve_node_dist_path()


def _json_names(folder: Path) -> set[str]:
    return {path.name for path in folder.glob("*.json")}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def plugin_copy() -> Path:
    """The fixture folder of the plugin checkout that holds the resolved
    ``dist/coordinator.js``."""
    if _NODE_DIST_PATH is None:
        pytest.skip(
            "No plugin checkout resolvable, so its copy of the hook_on_held fixtures "
            "cannot be compared. Build the plugin checkout (npm ci && npm run build) "
            "or set AGENT_COHERENCE_PLUGIN_DIST_PATH."
        )
    plugin_root = _NODE_DIST_PATH.parents[1]
    folder = plugin_root / _PLUGIN_COPY
    if not folder.is_dir():
        pytest.fail(
            f"The plugin checkout at {plugin_root} has no {_PLUGIN_COPY}, so it carries "
            f"no copy of the library's hook_on_held fixtures to compare."
        )
    return folder


def test_the_plugin_copy_has_the_same_json_files_as_the_library(plugin_copy: Path) -> None:
    library_names = _json_names(_LIBRARY_FIXTURES)
    copied_names = _json_names(plugin_copy)

    assert copied_names == library_names, (
        f"The plugin copy at {plugin_copy} does not hold the JSON files of "
        f"{_LIBRARY_FIXTURES}: missing from the copy {sorted(library_names - copied_names)}, "
        f"not in the library {sorted(copied_names - library_names)}."
    )


@pytest.mark.parametrize("name", sorted(_json_names(_LIBRARY_FIXTURES)))
def test_each_plugin_copy_is_byte_equal_to_the_library_file(plugin_copy: Path, name: str) -> None:
    library_file = _LIBRARY_FIXTURES / name
    copied_file = plugin_copy / name
    if not copied_file.is_file():
        pytest.fail(f"{name}: the plugin copy at {plugin_copy} has no such file.")

    # Compared as bytes, not parsed: the Node client is held to the exact deny
    # texts, and a byte diff of a 15 KB table would bury the file's name.
    if copied_file.read_bytes() != library_file.read_bytes():
        pytest.fail(
            f"{name}: the plugin's copy differs from the library's "
            f"(sha256 {_sha256(copied_file)} at {copied_file}, "
            f"{_sha256(library_file)} at {library_file}). "
            f"Copy it again from the library; never edit the copy by hand."
        )
