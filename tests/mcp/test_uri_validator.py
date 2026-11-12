"""Unit 3 — URI→coordinator-key validation (traversal + TOCTOU guard).

The validator front-runs the adapter: it turns a client path into a safe
workspace-relative key, or rejects it. Coverage: the happy normalisation, the
dotfile-preservation regression (removeprefix vs lstrip), traversal/absolute/
control/backslash rejection, the `.coherence/**` info-disclosure surface, and
real symlink escapes (file + dir), with a within-root symlink allowed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ccs.mcp.uri import UriValidationError, validate_uri

# --- happy + canonicalisation ------------------------------------------------


@pytest.mark.parametrize(
    "uri, expected",
    [
        ("data/notes.md", "data/notes.md"),
        ("./data/x", "data/x"),  # leading './' stripped
        ("data//notes.md", "data/notes.md"),  # repeated slash canonicalised
        ("data/sub/", "data/sub"),  # trailing slash canonicalised
        # The kernel opens these as the name with its '.' segments and extra
        # slashes dropped, so they key that same file.
        (".//data/x", "data/x"),
        ("././data/x", "data/x"),
        ("data/./x", "data/x"),
    ],
)
def test_valid_uri_normalises_to_key(tmp_path: Path, uri: str, expected: str) -> None:
    assert validate_uri(uri, root=tmp_path) == expected


@pytest.mark.parametrize("uri", [".env", "subdir/.env", "a/.gitignore"])
def test_dotfiles_preserved_exactly(tmp_path: Path, uri: str) -> None:
    """The removeprefix-not-lstrip regression: a dotfile must NOT lose its dot
    (lstrip('./') would turn '.env' into 'env', silently untracking it)."""
    assert validate_uri(uri, root=tmp_path) == uri


# --- rejection: traversal / absolute / control / backslash -------------------


@pytest.mark.parametrize(
    "uri",
    [
        "../escape",
        "a/../../escape",
        "/abs/path",
        "\\windows\\path",  # leading backslash
        "..\\escape",  # backslash traversal
        "data/\x00bad",  # NUL control char
        "data/\x1bbad",  # ESC control char
        "data/\x7fbad",  # DEL control char
        "",  # empty
    ],
)
def test_unsafe_uri_rejected(tmp_path: Path, uri: str) -> None:
    with pytest.raises(UriValidationError):
        validate_uri(uri, root=tmp_path)


def test_non_string_uri_rejected(tmp_path: Path) -> None:
    with pytest.raises(UriValidationError):
        validate_uri(42, root=tmp_path)  # type: ignore[arg-type]


def test_overlong_uri_rejected(tmp_path: Path) -> None:
    with pytest.raises(UriValidationError):
        validate_uri("a/" * 700, root=tmp_path)  # exceeds MAX_PATH_LEN (1024)


# --- rejection: .coherence/** info-disclosure surface ------------------------


@pytest.mark.parametrize(
    "uri",
    [
        ".coherence",
        ".coherence/state.db",
        ".coherence/state.db-wal",
        ".coherence/server.pid",
        ".coherence/hook.secret",
        ".coherence/tracked.yaml",
        ".coherence/ignored.yaml",
        ".coherence/strict_mode.yaml",
        ".coherence/audit.log",
        "./.coherence/state.db",
        ".//.coherence/state.db",
        ".COHERENCE/state.db",  # case-variant maps to the same dir on case-insensitive FS
    ],
)
def test_coherence_state_paths_rejected(tmp_path: Path, uri: str) -> None:
    with pytest.raises(UriValidationError):
        validate_uri(uri, root=tmp_path)


# --- symlink escapes (real filesystem) ---------------------------------------


def test_symlink_out_file_rejected(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("secret")
    (ws / "link.txt").symlink_to(outside)
    with pytest.raises(UriValidationError):
        validate_uri("link.txt", root=ws)


def test_symlink_out_dir_rejected(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (outside_dir / "f.txt").write_text("x")
    (ws / "sub").symlink_to(outside_dir, target_is_directory=True)
    with pytest.raises(UriValidationError):
        validate_uri("sub/f.txt", root=ws)


def test_symlink_within_root_allowed(tmp_path: Path) -> None:
    """A symlink whose target stays INSIDE root is contained → allowed. The
    validator's security job is preventing ESCAPE; the within-root
    version-tracking ambiguity is out of v1 scope. (Note: a regular file swapped
    for an out-of-root symlink AFTER validation is the accepted same-uid TOCTOU
    residual — O_NOFOLLOW adapter open is v1.1.)"""
    ws = tmp_path / "ws"
    (ws / "data").mkdir(parents=True)
    (ws / "data" / "real.txt").write_text("x")
    (ws / "data" / "alias.txt").symlink_to(ws / "data" / "real.txt")
    assert validate_uri("data/alias.txt", root=ws) == "data/alias.txt"


def test_new_nonexistent_file_allowed(tmp_path: Path) -> None:
    """A write target that does not exist yet validates (realpath resolves the
    existing prefix and stays within root)."""
    assert validate_uri("data/brand_new.txt", root=tmp_path) == "data/brand_new.txt"


def test_symlink_to_coherence_dir_rejected(tmp_path: Path) -> None:
    """A within-root symlink whose target RESOLVES into .coherence must be rejected
    even though its key (parts[0]) is not '.coherence' and it stays within root —
    otherwise the coordinator secret/state is readable through the symlink."""
    ws = tmp_path / "ws"
    ws.mkdir()
    coherence = ws / ".coherence"
    coherence.mkdir()
    (coherence / "hook.secret").write_text("secret")
    (ws / "data").symlink_to(coherence, target_is_directory=True)
    with pytest.raises(UriValidationError):
        validate_uri("data/hook.secret", root=ws)
