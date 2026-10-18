"""The raw-bytes disk read shared by foreign-write detection and pre-edit (#278).

Pre-edit runs it on a hook path against a name a session supplied, so every
way a file can stop being an observable regular file must answer None without
following, blocking or reading past the cap.
"""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path

import pytest

from ccs.adapters.claude_code.disk_content import hash_regular_file


def test_a_regular_file_hashes_its_raw_bytes(tmp_path: Path) -> None:
    """No decode and no newline translation: CRLF stays CRLF, or the hash
    would diverge from every writer that committed one."""
    data = b"line one\r\nline two\n\xff"
    (tmp_path / "plan.md").write_bytes(data)

    assert hash_regular_file(tmp_path / "plan.md") == hashlib.sha256(data).hexdigest()


def test_a_symlink_leaf_is_not_followed(tmp_path: Path) -> None:
    (tmp_path / "outside.md").write_text("secrets\n")
    (tmp_path / "plan.md").symlink_to(tmp_path / "outside.md")

    assert hash_regular_file(tmp_path / "plan.md") is None


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs named pipes")
def test_a_named_pipe_answers_none_without_blocking(tmp_path: Path) -> None:
    """O_NONBLOCK: a coordinated name replaced by a FIFO with no writer must
    not hang the hook thread. Fails (by timing out) if the open blocks."""
    os.mkfifo(tmp_path / "plan.md")
    answers: list[str | None] = []
    reader = threading.Thread(
        target=lambda: answers.append(hash_regular_file(tmp_path / "plan.md")), daemon=True
    )

    reader.start()
    reader.join(timeout=5)

    assert not reader.is_alive(), "hash_regular_file blocked on a named pipe"
    assert answers == [None]


@pytest.mark.parametrize("make", ["missing", "directory"])
def test_a_name_that_is_not_a_regular_file_answers_none(tmp_path: Path, make: str) -> None:
    if make == "directory":
        (tmp_path / "plan.md").mkdir()

    assert hash_regular_file(tmp_path / "plan.md") is None


def test_the_size_cap_is_inclusive(tmp_path: Path) -> None:
    (tmp_path / "at.md").write_bytes(b"1234")
    (tmp_path / "over.md").write_bytes(b"12345")

    assert hash_regular_file(tmp_path / "at.md", max_bytes=4) == hashlib.sha256(b"1234").hexdigest()
    assert hash_regular_file(tmp_path / "over.md", max_bytes=4) is None
