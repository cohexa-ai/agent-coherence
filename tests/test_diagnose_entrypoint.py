# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Guard the installed diagnose command's console-only entry point."""

import tomllib
from pathlib import Path


def test_console_script_targets_console_main() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as fh:
        scripts = tomllib.load(fh)["project"]["scripts"]
    assert scripts["ccs-diagnose"] == "ccs.cli.diagnose:console_main"
