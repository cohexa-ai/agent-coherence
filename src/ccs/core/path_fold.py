# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Fold keys: one key for every spelling a disk treats as one file name.

Three profiles, one per kind of workspace disk:

- ``none``: tidying only, for a disk that keeps every spelling apart.
- ``nfd``: canonical decomposition with canonical ordering, which is what a
  case-sensitive APFS volume folds.
- ``caseless``: NFD of the full case folding of NFD, the Unicode canonical
  caseless match, which is what a case-insensitive APFS volume folds. It
  applies no Turkic mapping (dotless i stays apart from I), no compatibility
  folding (fullwidth a stays apart from a) and removes no ignorable character.

Every fold applies per path segment after :func:`tidy_path`, and never turns a
backslash into a separator.

Why a table: :mod:`unicodedata` follows the Unicode version this Python was
built with, which differs between the supported Pythons and from the plugin's
Node coordinator. Both coordinators read one generated table instead
(``fold_data/fold_table.json``, Unicode 17.0.0, a byte-identical copy of the
plugin's), so they compute the same key for the same name.

Callers check :func:`exceeds_fold_limit` first: a name over the limit is never
folded. The folds themselves stay O(n log n) at any length.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable, Iterable, Sequence
from importlib import resources
from itertools import groupby
from typing import Any, Literal

FoldProfile = Literal["caseless", "nfd", "none"]
FoldingProfile = Literal["caseless", "nfd"]

FOLD_LENGTH_LIMIT = 1024

# Hangul syllables carry no mapping in the UCD; Unicode defines them by arithmetic.
_HANGUL_S_BASE = 0xAC00
_HANGUL_L_BASE = 0x1100
_HANGUL_V_BASE = 0x1161
_HANGUL_T_BASE = 0x11A7
_HANGUL_L_COUNT = 19
_HANGUL_V_COUNT = 21
_HANGUL_T_COUNT = 28
_HANGUL_N_COUNT = _HANGUL_V_COUNT * _HANGUL_T_COUNT
_HANGUL_S_COUNT = _HANGUL_L_COUNT * _HANGUL_N_COUNT


def _load_fold_document() -> dict[str, Any]:
    source = resources.files("ccs.core").joinpath("fold_data", "fold_table.json")
    return json.loads(source.read_text(encoding="utf-8"))


def _parse_hex_sequence(sequence: str) -> tuple[int, ...]:
    return tuple(int(code_point, 16) for code_point in sequence.split(" "))


def _sequence_map(record: dict[str, str]) -> dict[int, tuple[int, ...]]:
    return {int(code_point, 16): _parse_hex_sequence(sequence) for code_point, sequence in record.items()}


_FOLD_DOCUMENT = _load_fold_document()
_FOLD_TABLE: dict[str, Any] = _FOLD_DOCUMENT["table"]
FOLD_TABLE_UNICODE_VERSION: str = _FOLD_DOCUMENT["unicode_version"]

_DECOMPOSITION = _sequence_map(_FOLD_TABLE["decomposition"])
_CASE_FOLDING = _sequence_map(_FOLD_TABLE["case_folding"])
_COMBINING_CLASS: dict[int, int] = {
    int(code_point, 16): combining_class for code_point, combining_class in _FOLD_TABLE["combining_class"].items()
}
_COMPOSITION: dict[tuple[int, ...], int] = {
    _parse_hex_sequence(pair): int(composite, 16) for pair, composite in _FOLD_TABLE["composition"].items()
}
del _FOLD_DOCUMENT, _FOLD_TABLE


def _combining_class(code_point: int) -> int:
    return _COMBINING_CLASS.get(code_point, 0)


def _from_code_points(code_points: Iterable[int]) -> str:
    return "".join(map(chr, code_points))


def _decompose_code_point(code_point: int) -> tuple[int, ...]:
    syllable_index = code_point - _HANGUL_S_BASE
    if not 0 <= syllable_index < _HANGUL_S_COUNT:
        return _DECOMPOSITION.get(code_point, (code_point,))
    leading = _HANGUL_L_BASE + syllable_index // _HANGUL_N_COUNT
    vowel = _HANGUL_V_BASE + (syllable_index % _HANGUL_N_COUNT) // _HANGUL_T_COUNT
    trailing = _HANGUL_T_BASE + syllable_index % _HANGUL_T_COUNT
    return (leading, vowel) if trailing == _HANGUL_T_BASE else (leading, vowel, trailing)


def _canonical_order(code_points: Iterable[int]) -> list[int]:
    """Each run of non-starters sorted by combining class. ``sorted`` is
    stable, which canonical ordering needs (marks of one class keep their
    order), and O(n log n), so a long run of marks in reverse order stays
    cheap."""
    ordered: list[int] = []
    for is_mark_run, run in groupby(code_points, key=lambda code_point: _combining_class(code_point) != 0):
        ordered.extend(sorted(run, key=_combining_class) if is_mark_run else run)
    return ordered


def _canonical_decomposition(code_points: Iterable[int]) -> list[int]:
    return _canonical_order(part for code_point in code_points for part in _decompose_code_point(code_point))


def _case_fold(code_points: Iterable[int]) -> list[int]:
    return [part for code_point in code_points for part in _CASE_FOLDING.get(code_point, (code_point,))]


def _compose_hangul(first: int, second: int) -> int | None:
    leading_index = first - _HANGUL_L_BASE
    vowel_index = second - _HANGUL_V_BASE
    if 0 <= leading_index < _HANGUL_L_COUNT and 0 <= vowel_index < _HANGUL_V_COUNT:
        return _HANGUL_S_BASE + (leading_index * _HANGUL_V_COUNT + vowel_index) * _HANGUL_T_COUNT
    syllable_index = first - _HANGUL_S_BASE
    trailing_index = second - _HANGUL_T_BASE
    is_lv_syllable = 0 <= syllable_index < _HANGUL_S_COUNT and syllable_index % _HANGUL_T_COUNT == 0
    return first + trailing_index if is_lv_syllable and 0 < trailing_index < _HANGUL_T_COUNT else None


def _compose_pair(starter: int, code_point: int) -> int | None:
    composite = _compose_hangul(starter, code_point)
    return composite if composite is not None else _COMPOSITION.get((starter, code_point))


def _canonical_composition(code_points: Sequence[int]) -> list[int]:
    """Canonical composition of a canonically ordered sequence. A code point
    joins the last starter unless a mark kept between them has the same or a
    higher class (any kept starter would itself be the last starter). In
    canonical order the last mark kept has the highest class of those between,
    so it alone decides."""
    composed: list[int] = []
    starter_index = -1
    last_class = 0
    for code_point in code_points:
        code_point_class = _combining_class(code_point)
        is_blocked = starter_index != len(composed) - 1 and last_class >= code_point_class
        if starter_index >= 0 and not is_blocked:
            composite = _compose_pair(composed[starter_index], code_point)
            if composite is not None:
                composed[starter_index] = composite
                continue
        if code_point_class == 0:
            starter_index = len(composed)
        last_class = code_point_class
        composed.append(code_point)
    return composed


def _segment_key(segment: str, profile: FoldingProfile) -> list[int]:
    decomposed = _canonical_decomposition(map(ord, segment))
    if profile == "nfd":
        return decomposed
    if profile == "caseless":
        # The inner decomposition comes first because U+0345 must reach its
        # canonical place among the marks before it folds to a starter.
        return _canonical_decomposition(_case_fold(decomposed))
    raise ValueError(f"not a folding profile: {profile!r}")


def _map_segments(path: str, fold: Callable[[str], str]) -> str:
    return "/".join(map(fold, path.split("/")))


def tidy_path(path: str) -> str:
    """Collapse repeated ``/`` and drop ``.`` segments, as the kernel does when
    it looks a path up. A leading and a trailing ``/`` stay, and ``..``, a
    backslash and the text of every other segment are left exactly as given:
    resolving ``..`` by text is wrong across a link, and refusing it is the
    caller's job. Hand-written because ``posixpath.normpath`` treats a trailing
    ``/`` and a leading ``//`` otherwise than the plugin's Node coordinator."""
    segments = [segment for segment in path.split("/") if segment not in ("", ".")]
    leading = "/" if path.startswith("/") else ""
    trailing = "/" if path.endswith("/") and segments else ""
    return f"{leading}{'/'.join(segments)}{trailing}"


def fold_key(path: str, profile: FoldProfile) -> str:
    """The key every spelling of ``path`` that ``profile``'s disk treats as
    one name shares."""
    tidied = tidy_path(path)
    if profile == "none":
        return tidied
    return _map_segments(tidied, lambda segment: _from_code_points(_segment_key(segment, profile)))


def policy_fold(path: str, profile: FoldingProfile) -> str:
    """The profile's key recomposed to NFC, for matching against policy globs:
    a glob's ``?`` then meets one composed letter, as it does for a name typed
    in NFC. A disk that folds nothing matches globs against the name as
    given, so ``none`` is refused."""
    return policy_fold_text(tidy_path(path), profile)


def policy_fold_text(text: str, profile: FoldingProfile) -> str:
    """:func:`policy_fold` of ``text`` exactly as given, without tidying: a
    glob's literal characters fold where they stand, and tidying a run such as
    the lone ``.`` in ``*.*`` would change what the glob matches."""
    return _map_segments(
        text, lambda segment: _from_code_points(_canonical_composition(_segment_key(segment, profile)))
    )


def fold_code_point(char: str, profile: FoldingProfile) -> str | None:
    """:func:`policy_fold_text` of the one code point ``char`` when that is
    one code point too, else None (U+00DF folds to ``ss`` caselessly). Looked
    up, not worked out: a class fold asks this of every member of every
    class, and an untrack request's entries can hold thousands of them."""
    return _single_folds(profile).get(char, char)


def code_points_folding_to(char: str, profile: FoldingProfile) -> frozenset[str]:
    """Every code point whose :func:`fold_code_point` is ``char``: caselessly,
    ``k``, ``K`` and the Kelvin sign for ``k``, and none for ``K``."""
    spellings = _fold_inverse(profile).get(char, frozenset())
    return spellings | {char} if fold_code_point(char, profile) == char else spellings


@functools.cache
def _single_folds(profile: FoldingProfile) -> dict[str, str | None]:
    """:func:`fold_code_point` over the table's domain, built on first use.
    Only a code point with a case-folding or a decomposition entry can fold
    to anything but itself (a Hangul syllable decomposes and recomposes to
    itself), so every code point outside the domain folds to itself, and the
    other runtime, reading the same table, builds the same map."""
    folds: dict[str, str | None] = {}
    for code_point in _DECOMPOSITION.keys() | _CASE_FOLDING.keys():
        char = chr(code_point)
        folded = policy_fold_text(char, profile)
        folds[char] = folded if len(folded) == 1 else None
    return folds


@functools.cache
def _fold_inverse(profile: FoldingProfile) -> dict[str, frozenset[str]]:
    """:func:`fold_code_point` inverted over the table's domain, built on
    first use from :func:`_single_folds`."""
    inverse: dict[str, set[str]] = {}
    for char, folded in _single_folds(profile).items():
        if folded is not None:
            inverse.setdefault(folded, set()).add(char)
    return {folded: frozenset(chars) for folded, chars in inverse.items()}


def exceeds_fold_limit(text: str) -> bool:
    """Whether ``text`` is over :data:`FOLD_LENGTH_LIMIT` code points, the
    length past which a name is never folded."""
    return len(text) > FOLD_LENGTH_LIMIT
