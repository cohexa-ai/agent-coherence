# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Check a release tag against the package version, and classify the release.

The release workflow runs this on every ``v*`` tag before it builds or
publishes anything. It fails when the tag does not name the package version,
and when that version is a development release (``.devN``): a dev build is a
snapshot of an unreleased branch, not something to publish. Any other version
passes. A pre-release (``aN``, ``bN``, ``rcN``) is reported as one, so the
workflow can mark the GitHub release as a pre-release. PyPI still receives it,
and pip installs it only when asked (``--pre`` or an exact pin).

Usage:
    RELEASE_TAG=v0.15.0 python tools/check_release_tag.py
    python tools/check_release_tag.py --tag v0.15.0rc1

When ``GITHUB_OUTPUT`` is set, a successful run appends ``prerelease=true`` or
``prerelease=false`` to it for later steps and jobs.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from packaging.version import InvalidVersion, Version

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_INIT = _REPO_ROOT / "src" / "ccs" / "__init__.py"
_VERSION_LINE = re.compile(r'^__version__\s*=\s*"([^"]+)"', re.MULTILINE)


class ReleaseTagError(ValueError):
    """The tag must not be released; the message says why."""


@dataclass(frozen=True)
class ReleaseKind:
    """What the tag releases."""

    version: str
    is_prerelease: bool


def read_package_version(package_init: Path) -> str:
    """Return ``__version__`` as written in ``package_init``."""
    match = _VERSION_LINE.search(package_init.read_text())
    if match is None:
        raise ReleaseTagError(f"Could not find __version__ in {package_init}")
    return match.group(1)


def classify_release(tag: str, version: str) -> ReleaseKind:
    """Return the release ``tag`` makes, or raise ``ReleaseTagError``.

    A dev release is refused before the pre-release question is asked,
    because ``packaging`` counts a dev release as a pre-release too.
    """
    expected_tag = f"v{version}"
    if tag != expected_tag:
        raise ReleaseTagError(
            f"Release tag {tag} does not match package version {version} "
            f"(expected {expected_tag})"
        )
    try:
        parsed = Version(version)
    except InvalidVersion as exc:
        raise ReleaseTagError(f"Package version {version} is not a valid PEP 440 version") from exc
    if parsed.is_devrelease:
        raise ReleaseTagError(
            f"Release tag {tag} names a development version ({version}). "
            "Set the release version in src/ccs/__init__.py before tagging."
        )
    return ReleaseKind(version=version, is_prerelease=parsed.is_prerelease)


def _write_step_output(kind: ReleaseKind) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    with open(output_path, "a", encoding="utf-8") as output:
        output.write(f"prerelease={'true' if kind.is_prerelease else 'false'}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tag",
        default=os.environ.get("RELEASE_TAG"),
        help="release tag, e.g. v0.15.0 (default: $RELEASE_TAG)",
    )
    parser.add_argument("--package-init", type=Path, default=_PACKAGE_INIT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.tag:
        print("No release tag given: pass --tag or set RELEASE_TAG", file=sys.stderr)
        return 1
    try:
        kind = classify_release(args.tag, read_package_version(args.package_init))
    except ReleaseTagError as exc:
        print(exc, file=sys.stderr)
        return 1
    label = "a pre-release" if kind.is_prerelease else "a final release"
    print(f"Release tag {args.tag} matches package version {kind.version}, {label}")
    _write_step_output(kind)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
