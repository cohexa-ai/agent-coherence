# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The two path resolvers: one that follows links and keeps the case a path
was typed in, and one that answers with the disk's own spelling.

Each test works in a fresh temporary directory, named through the platform's
temporary directory as given (on macOS ``/var/...``, itself a link to
``/private/var/...``), and removed afterwards.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from ccs.core.disk_spelling import LinkDoesNotResolve, native_spelling, resolve_links

IS_DARWIN = sys.platform == "darwin"


@pytest.fixture
def scratch() -> Iterator[str]:
    """A fresh directory holding ``Repo/plan.md``, as its unresolved path."""
    base = tempfile.mkdtemp(prefix="disk-spelling-")
    if not base or not os.path.isabs(base):
        raise RuntimeError(f"refusing to use {base!r}")
    try:
        os.mkdir(os.path.join(base, "Repo"))
        Path(base, "Repo", "plan.md").write_text("plan\n", encoding="utf-8")
        yield base
    finally:
        shutil.rmtree(base)


def canonical(path: str) -> str:
    return os.path.realpath(path, strict=True)


def is_case_insensitive(directory: str) -> bool:
    return os.path.lexists(os.path.join(directory, "REPO"))


@pytest.mark.skipif(not IS_DARWIN, reason="/var is a link to /private/var on macOS")
def test_var_resolves_to_private_var_under_both_resolvers() -> None:
    assert resolve_links("/var") == "/private/var"
    assert native_spelling("/var") == "/private/var"


def test_an_existing_path_resolves_through_the_temporary_directory_link(scratch: str) -> None:
    typed = os.path.join(scratch, "Repo", "plan.md")
    expected = os.path.join(canonical(scratch), "Repo", "plan.md")
    assert resolve_links(typed) == expected
    assert native_spelling(typed) == expected


def test_a_differently_cased_path_gets_the_disk_case_from_native_spelling_only(scratch: str) -> None:
    if not is_case_insensitive(scratch):
        pytest.skip("the temporary directory's disk keeps REPO and Repo apart")
    typed = os.path.join(scratch, "REPO", "PLAN.MD")
    assert native_spelling(typed) == os.path.join(canonical(scratch), "Repo", "plan.md")
    assert resolve_links(typed) == os.path.join(canonical(scratch), "REPO", "PLAN.MD")


def test_a_missing_tail_is_kept_as_typed_after_the_deepest_existing_ancestor(scratch: str) -> None:
    typed = os.path.join(scratch, "Repo", "New Dir", "File.MD")
    expected = os.path.join(canonical(scratch), "Repo", "New Dir", "File.MD")
    assert resolve_links(typed) == expected
    assert native_spelling(typed) == expected


def test_a_missing_tail_under_a_differently_cased_ancestor_takes_the_disk_case_of_the_ancestor(
    scratch: str,
) -> None:
    if not is_case_insensitive(scratch):
        pytest.skip("the temporary directory's disk keeps REPO and Repo apart")
    typed = os.path.join(scratch, "REPO", "New.md")
    assert native_spelling(typed) == os.path.join(canonical(scratch), "Repo", "New.md")
    assert resolve_links(typed) == os.path.join(canonical(scratch), "REPO", "New.md")


def test_a_path_under_a_regular_file_keeps_its_tail_as_typed(scratch: str) -> None:
    typed = os.path.join(scratch, "Repo", "plan.md", "x")
    expected = os.path.join(canonical(scratch), "Repo", "plan.md", "x")
    assert resolve_links(typed) == expected
    assert native_spelling(typed) == expected


def test_a_link_is_followed_to_its_target(scratch: str) -> None:
    os.symlink(os.path.join(scratch, "Repo"), os.path.join(scratch, "linked"))
    typed = os.path.join(scratch, "linked", "plan.md")
    expected = os.path.join(canonical(scratch), "Repo", "plan.md")
    assert resolve_links(typed) == expected
    assert native_spelling(typed) == expected


@pytest.mark.parametrize("resolver", [resolve_links, native_spelling], ids=["resolve_links", "native_spelling"])
def test_a_parent_step_after_a_link_leaves_the_links_target(scratch: str, resolver) -> None:
    """``..`` after a link steps up from the link's target, as the kernel
    does, and a ``..`` in a tail that does not exist yet is kept as typed:
    folding either by text would name another file than the kernel opens."""
    os.makedirs(os.path.join(scratch, "other", "dir"))
    os.symlink(os.path.join(scratch, "other", "dir"), os.path.join(scratch, "Repo", "link"))
    root = canonical(scratch)

    assert resolver(os.path.join(scratch, "Repo", "link", "..", "plan.md")) == os.path.join(root, "other", "plan.md")
    assert resolver(os.path.join(scratch, "Repo", "link", "..", "New", "x.md")) == os.path.join(
        root, "other", "New", "x.md"
    )
    missing_tail = os.path.join("missing", "..", "plan.md")
    assert resolver(os.path.join(scratch, "Repo", missing_tail)) == os.path.join(root, "Repo", missing_tail)


@pytest.mark.parametrize("resolver", [resolve_links, native_spelling], ids=["resolve_links", "native_spelling"])
def test_a_dangling_link_is_reported_not_resolved(scratch: str, resolver) -> None:
    dangling = os.path.join(scratch, "dangling")
    os.symlink(os.path.join(scratch, "missing"), dangling)
    with pytest.raises(LinkDoesNotResolve) as refused:
        resolver(dangling)
    assert refused.value.filename == dangling
    with pytest.raises(LinkDoesNotResolve) as refused_below:
        resolver(os.path.join(dangling, "plan.md"))
    assert refused_below.value.filename == dangling


@pytest.mark.parametrize("resolver", [resolve_links, native_spelling], ids=["resolve_links", "native_spelling"])
def test_a_looping_link_is_reported_not_resolved(scratch: str, resolver) -> None:
    first, second = os.path.join(scratch, "first"), os.path.join(scratch, "second")
    os.symlink(second, first)
    os.symlink(first, second)
    with pytest.raises(LinkDoesNotResolve):
        resolver(os.path.join(first, "plan.md"))
    with pytest.raises(LinkDoesNotResolve) as refused:
        resolver(first)
    assert refused.value.filename == first


@pytest.mark.parametrize("resolver", [resolve_links, native_spelling], ids=["resolve_links", "native_spelling"])
@pytest.mark.parametrize("path", ["", "Repo/plan.md", "./plan.md"])
def test_a_relative_or_empty_path_is_refused(resolver, path: str) -> None:
    with pytest.raises(ValueError, match="not an absolute path"):
        resolver(path)


def test_a_path_object_is_accepted(scratch: str) -> None:
    assert resolve_links(Path(scratch, "Repo")) == os.path.join(canonical(scratch), "Repo")
