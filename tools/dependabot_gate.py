# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Gate: decide whether a Dependabot pull request may merge itself into dev.

.github/workflows/dependabot-gate.yml runs this on every Dependabot PR, from
the copy on dev at the PR's merge base, never the PR's own copy, and records
the verdict as a commit status on the PR's head. The scheduled
.github/workflows/dependabot-merge.yml merges only a head whose status says it
qualifies.

A PR qualifies only when all of these hold:
  - it updates the Python lock (ecosystem ``uv``) or the TypeScript demo
    fixture (``npm_and_yarn``). A GitHub Actions bump brings third-party code
    that runs inside the workflows, some of which hold write or publish tokens,
    so it waits for a person;
  - it is not a grouped update. A grouped update reports the largest semver
    change among the dependencies it can classify and skips the rest, and the
    grouped security updates here carry no update type at all;
  - its update type is a semver patch or minor;
  - every changed file is that ecosystem's lockfile or manifest;
  - for uv, no package in uv.lock is lower than at the merge base, in any
    Python-version fork. CI installs from pyproject.toml, never from uv.lock,
    so a green run says nothing about the lock: a lockfile-only mcp bump once
    passed every check while moving crewai from 1.15.27 back to 1.6.1, which
    pins a vulnerable json-repair. A package the update drops waits for a
    person rather than refusing: upgrades also drop dependencies they no
    longer need;
  - for uv, every package keeps the source it had at the merge base, and a
    new package comes from PyPI; any other source waits for a person;
  - for uv, every package a locked package depends on is itself locked. A
    branch brought up to date with a git merge carries a uv.lock no resolver
    wrote, and such a merge can keep a dependency on a package the other side
    removed.

Verdicts:
    exit 0   QUALIFIES    the merger may merge this head once its checks pass
    exit 10  WAITS        a person decides; not an error
    exit 1   REFUSE       uv.lock lowers a package, or depends on one it does
                          not lock
    exit 2   CANNOT TELL  git, the lock or a version could not be read
The one-line reason goes to stdout. "Cannot tell" never becomes a pass.

Usage:
    python tools/dependabot_gate.py --base REV --head REV --ecosystem E \\
        --update-type T --group G
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from typing import Any

QUALIFIES, REFUSE, CANNOT_TELL, WAITS = 0, 1, 2, 10
FIXTURE = "examples/refactor_demo/fixture_repo_ts"
# Keys are fetch-metadata's package-ecosystem values, which it reads from the
# branch name (dependabot/uv/..., dependabot/npm_and_yarn/...).
ALLOWED_FILES = {
    "uv": frozenset({"uv.lock", "requirements-diagnose.txt"}),
    "npm_and_yarn": frozenset({f"{FIXTURE}/package.json", f"{FIXTURE}/package-lock.json"}),
}
ALLOWED_UPDATE_TYPES = frozenset({"version-update:semver-patch", "version-update:semver-minor"})
PYPI = {"registry": "https://pypi.org/simple"}
# A package name as uv.lock writes it. Names reach the verdict text, so one
# carrying a line break or anything else unexpected is not trusted.
PACKAGE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class CannotTell(Exception):
    """An input the verdict depends on could not be read."""


@dataclass(frozen=True)
class Update:
    ecosystem: str
    update_type: str
    group: str
    files: tuple[str, ...]


def wait_reason(update: Update) -> str | None:
    """Why a person must decide on this update, or None when it may qualify."""
    allowed = ALLOWED_FILES.get(update.ecosystem)
    if allowed is None:
        return f"{update.ecosystem or 'unknown ecosystem'} update"
    if update.group:
        return f"grouped update ({update.group})"
    if update.update_type not in ALLOWED_UPDATE_TYPES:
        return f"{update.update_type or 'unclassified'} update"
    if not update.files:
        return "no changed files"
    outside = sorted(set(update.files) - allowed)
    if outside:
        return f"changes {', '.join(outside)}"
    return None


def parse_version(text: str) -> Any:
    # ubuntu-24.04's system python3 ships Debian's python3-packaging.
    try:
        from packaging.version import InvalidVersion, Version
    except ImportError as error:
        raise CannotTell("no packaging module to compare versions") from error
    try:
        return Version(text)
    except InvalidVersion as error:
        raise CannotTell(f"unreadable version {text!r}") from error


def lock_packages(lock_text: str) -> list[dict[str, Any]]:
    try:
        packages = tomllib.loads(lock_text)["package"]
    except (tomllib.TOMLDecodeError, KeyError, TypeError) as error:
        raise CannotTell(f"unreadable uv.lock: {error}") from error
    if not isinstance(packages, list):
        raise CannotTell("unreadable uv.lock: package is not a list")
    names = [package.get("name") for package in packages]
    names += [dependency.get("name") for package in packages for dependency in _dependencies(package)]
    if not all(isinstance(name, str) and PACKAGE_NAME.fullmatch(name) for name in names):
        raise CannotTell("unreadable uv.lock: a package name is not a plain name")
    return packages


def locked_versions(lock_text: str) -> dict[str, dict[tuple[str, ...], Any]]:
    """Each package's locked version per Python-version fork. uv locks one
    package at several versions when Python versions need different ones, and
    keys each by its resolution markers; an unforked package has the key ()."""
    versions: dict[str, dict[tuple[str, ...], Any]] = {}
    for package in lock_packages(lock_text):
        if "version" not in package:  # the project itself, versioned dynamically
            continue
        forks = versions.setdefault(package["name"], {})
        forks[tuple(package.get("resolution-markers", ()))] = parse_version(package["version"])
    return versions


def _lowered(base: dict[tuple[str, ...], Any], head: dict[tuple[str, ...], Any]) -> bool:
    """Lower in a fork both locks share, or in the lowest or highest version
    overall (a fork that split or merged has no counterpart to compare)."""
    if any(head[markers] < base[markers] for markers in base.keys() & head.keys()):
        return True
    return min(head.values()) < min(base.values()) or max(head.values()) < max(base.values())


def _shown(forks: dict[tuple[str, ...], Any]) -> str:
    return ", ".join(str(version) for version in sorted(set(forks.values())))


def lock_regressions(base_text: str, head_text: str) -> tuple[list[str], list[str]]:
    """The packages the head locks lower than the base does, and those it drops."""
    base, head = locked_versions(base_text), locked_versions(head_text)
    lowered = [
        f"{name} {_shown(base[name])} -> {_shown(head[name])}"
        for name in sorted(base)
        if name in head and _lowered(base[name], head[name])
    ]
    removed = [f"{name} {_shown(base[name])}" for name in sorted(base) if name not in head]
    return lowered, removed


def source_changes(base_text: str, head_text: str) -> list[str]:
    """Each package whose source the head changes, and each new one not from PyPI."""
    base = {package["name"]: package.get("source") for package in lock_packages(base_text)}
    changed = {
        package["name"]
        for package in lock_packages(head_text)
        if package.get("source") != base.get(package["name"], PYPI)
    }
    return sorted(changed)


def _dependencies(package: dict[str, Any]) -> list[dict[str, Any]]:
    groups = [package.get("dependencies", [])]
    for key in ("optional-dependencies", "dev-dependencies"):
        groups.extend((package.get(key) or {}).values())
    return [dependency for group in groups for dependency in group]


def unlocked_dependencies(lock_text: str) -> list[str]:
    """Each dependency a locked package names that uv.lock does not lock, at
    the version it names when it names one."""
    packages = lock_packages(lock_text)
    locked = {(package["name"], package.get("version")) for package in packages}
    names = {name for name, _version in locked}
    missing = set()
    for package in packages:
        for dependency in _dependencies(package):
            name, version = dependency.get("name"), dependency.get("version")
            if name not in names or (version is not None and (name, version) not in locked):
                missing.add(f"{name} {version}" if version else name)
    return sorted(missing)


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise CannotTell(f"git {' '.join(args)} failed") from error


def changed_files(base: str, head: str) -> tuple[str, ...]:
    listing = git("diff", "--name-only", "--no-renames", "-z", base, head)
    return tuple(path for path in listing.split("\0") if path)


def evaluate(args: argparse.Namespace) -> tuple[int, str]:
    update = Update(args.ecosystem, args.update_type, args.group, changed_files(args.base, args.head))
    reason = wait_reason(update)
    if reason:
        return WAITS, f"waits for a person: {reason}"
    if update.ecosystem == "uv":
        return judge_lock(git("show", f"{args.base}:uv.lock"), git("show", f"{args.head}:uv.lock"))
    return QUALIFIES, "qualifies"


def judge_lock(base_lock: str, head_lock: str) -> tuple[int, str]:
    lowered, removed = lock_regressions(base_lock, head_lock)
    if lowered:
        return REFUSE, f"uv.lock lowers {len(lowered)}: {', '.join(lowered)}"
    unlocked = unlocked_dependencies(head_lock)
    if unlocked:
        return REFUSE, f"uv.lock depends on {len(unlocked)} it does not lock: {', '.join(unlocked)}"
    moved = source_changes(base_lock, head_lock)
    if moved:
        return WAITS, f"waits for a person: uv.lock takes {len(moved)} from a new source: {', '.join(moved)}"
    if removed:
        return WAITS, f"waits for a person: uv.lock removes {len(removed)}: {', '.join(removed)}"
    return QUALIFIES, "qualifies"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="the PR's merge base with dev")
    parser.add_argument("--head", required=True, help="the PR's head")
    parser.add_argument("--ecosystem", required=True, help="fetch-metadata package-ecosystem")
    parser.add_argument("--update-type", required=True, help="fetch-metadata update-type")
    parser.add_argument("--group", required=True, help="fetch-metadata dependency-group")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        code, message = evaluate(args)
    except CannotTell as error:
        code, message = CANNOT_TELL, f"cannot tell: {error}"
    print(message)
    return code


if __name__ == "__main__":
    sys.exit(main())
