# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Tracked-path WRITE detection in Bash command strings.

Used by the ``/hooks/pre-bash`` handler to find the tracked artifacts a Bash
command would WRITE, so that the giver of a live handoff (#185) is refused a
shell write to the path it handed off just as its Edit/Write is. A model whose
edit is denied routes around the deny through the shell
(``echo '- x' >> plan.md``), which the read-side twin,
:mod:`ccs.adapters.claude_code.bash_path_detector`, never looks for.

False-negative bias, as in the read detector: a curated set of common write
forms is recognised and anything else reads as "writes nothing". A miss leaves
the command's answer what it was before this module existed; a wrong hit would
deny a command that never touches the path.

Recognised:

- output redirections -- ``>``, ``>>``, ``>|``, ``&>``, ``&>>``, ``<>``, their
  fd-prefixed forms (``1>>``) and ``>& file`` -- on any command, ``exec`` and
  ``{ ...; }`` / ``( ... )`` groups included;
- ``tee``; in-place ``sed -i`` and ``perl -i``; ``ed`` and ``ex`` when their
  script writes, or when the command does not show the script;
- ``cp`` / ``mv`` / ``install`` / ``ln`` / ``rsync`` onto the path (a directory
  destination when it ends in ``/`` or is given with ``-t``), ``mv`` / ``rm`` /
  ``unlink`` / ``shred`` / ``truncate`` of it, ``dd of=``, ``sort -o`` and
  ``patch``;
- ``git checkout`` / ``restore`` / ``rm`` / ``mv`` naming it, unless the verb
  touches only the index (``restore --staged``, ``rm --cached``);
- the body of ``eval`` and of ``bash`` / ``sh`` / ``zsh`` / ``dash`` (``-c`` or
  a heredoc), scanned as a command; and the program of an interpreter
  one-liner or heredoc (``python -c``, ``perl -e``, ``ruby -e``, ``node -e``,
  ``php -r``, ``python3 - <<EOF``) that holds a write call, from which a path
  with an extension written as a whole string literal is taken when it is a
  write call's target -- an open for writing, a write, append, delete,
  rename or move call, a copy's destination, a ``pathlib`` write method, a
  perl write-mode ``open`` -- directly or through a name bound to it
  (``p = Path('plan.md'); p.write_text(...)``), or when it is a redirection's
  target inside a shell string; a path the program only reads, copies from or
  names in text it writes is not taken;
- ``cd`` inside the command, undone at the end of a ``( ... )`` subshell.

OUT of scope -- answered as "writes nothing", or, for a relative path after an
unfollowed directory change, not resolved:

- paths built from variables or command substitution (``"$PWD/plan.md"``,
  ``$(git rev-parse --show-toplevel)/plan.md``), ``~`` paths and globs, and
  in a program body a path assembled from pieces or held inside a longer
  string, or one that reaches the write call through a container, a loop, a
  tuple assignment or a function's parameter
  (``for name in ['plan.md']: Path(name).write_text(...)``,
  ``cfg = {'out': 'plan.md'}``, ``src, dst = 'a.md', 'plan.md'``), or
  through a name bound more than 8000 characters before the write or after
  the program's first 64 bindings;
- writer tools not listed above, e.g. ``gsed``, ``awk -i inplace``, ``vim``,
  ``curl -o``, ``wget -O``, ``prettier --write``, and any script run from a
  file (``python3 fix.py``);
- a wrapper given options of its own (``sudo -u root tee plan.md``);
- file lists made at run time (``find -exec``, ``xargs``), a recursive delete
  of a directory holding the path, a directory destination spelled without a
  trailing ``/``, and git verbs that rewrite files without naming them
  (``reset --hard``, ``stash``, ``apply``, ``merge``, ``pull``);
- ``pushd`` / ``popd`` and a ``cd`` whose target the command does not spell
  out (``cd``, ``cd -``, ``cd "$DIR"``): relative paths after one are not
  resolved;
- a directory change made by an EARLIER command: relative paths resolve from
  the ``cwd`` the caller passes, and the coordinator is not told the session's
  directory, so it passes the workspace root;
- adversarial obfuscation of any kind.

A pure function: no I/O and no registry access, so a command that writes no
tracked path costs the handler a string scan and nothing else.
"""
from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Sequence

from ccs.adapters.claude_code.bash_path_detector import PATH_TOKEN_RE

# ----------------------------------------------------------------------
# Lexing: words (quotes removed, nothing expanded), operators, heredocs
# ----------------------------------------------------------------------

_Token = tuple[str, str]
"""``("word", text)`` with the quoting removed, or ``("op", text)``."""

_OPERATORS: tuple[str, ...] = (
    "&>>", "<<<", "<<-",
    "&>", ">>", ">|", ">&", "<<", "<&", "<>", "&&", "||", "|&", ";;",
    "|", "&", ";", "<", ">", "(", ")",
)
"""Longest first, so the first one that matches at a position is the operator."""

_OPERATOR_CHARS = frozenset("|&;<>()")
_FD_DIGITS = "0123456789"
_HEREDOC_OPERATORS = frozenset({"<<", "<<-"})
_HERESTRING_OPERATOR = "<<<"
_COMMAND_SEPARATORS = frozenset({"|", "||", "&&", ";", "&", "\n", "|&", ";;"})
_SUBSHELL_OPEN = "("
_SUBSHELL_CLOSE = ")"

_DOUBLE_QUOTE_ESCAPABLE = frozenset('$`"\\')
_ANSI_C_ESCAPES: dict[str, str] = {"n": "\n", "t": "\t", "r": "\r"}


@dataclass(frozen=True)
class _Lexed:
    tokens: list[_Token]
    heredocs: dict[int, str]
    """Each heredoc's body, keyed by the token index of its ``<<`` operator."""


class _Lexer:
    """Split a command into words and operators and collect heredoc bodies.

    A lexer only: it removes quoting and nothing else (no expansion), and knows
    no shell grammar beyond what the write scan needs."""

    def __init__(self, text: str) -> None:
        self._text = text
        self._pos = 0
        self._tokens: list[_Token] = []
        self._heredocs: dict[int, str] = {}
        self._chars: list[str] = []
        self._in_word = False
        # A heredoc operator still waiting for its delimiter word, then the
        # heredocs whose bodies start after the next newline.
        self._awaiting_delimiter: tuple[int, bool] | None = None
        self._queued_bodies: list[tuple[int, str, bool]] = []

    def lex(self) -> _Lexed:
        while self._pos < len(self._text):
            self._step(self._text[self._pos])
        self._end_word()
        return _Lexed(self._tokens, self._heredocs)

    def _step(self, char: str) -> None:
        if char == "\n":
            self._newline()
        elif char in " \t":
            self._end_word()
            self._pos += 1
        elif char == "#" and not self._in_word:
            self._skip_comment()
        elif char == "\\":
            self._backslash()
        elif char == "'":
            self._quoted(self._pos + 1, "'", _no_escape)
        elif char == '"':
            self._quoted(self._pos + 1, '"', _double_quote_escape)
        elif char == "$" and self._text.startswith("'", self._pos + 1):
            self._quoted(self._pos + 2, "'", _ansi_c_escape)
        elif char in _OPERATOR_CHARS:
            self._operator()
        else:
            self._append(char)
            self._pos += 1

    def _append(self, text: str) -> None:
        self._chars.append(text)
        self._in_word = True

    def _end_word(self) -> None:
        if not self._in_word:
            return
        word = "".join(self._chars)
        self._chars.clear()
        self._in_word = False
        if self._awaiting_delimiter is not None:
            operator_index, strip_tabs = self._awaiting_delimiter
            self._queued_bodies.append((operator_index, word, strip_tabs))
            self._awaiting_delimiter = None
        self._tokens.append(("word", word))

    def _newline(self) -> None:
        self._end_word()
        self._tokens.append(("op", "\n"))
        self._pos += 1
        for operator_index, delimiter, strip_tabs in self._queued_bodies:
            self._heredocs[operator_index] = self._read_heredoc(delimiter, strip_tabs)
        self._queued_bodies.clear()

    def _read_heredoc(self, delimiter: str, strip_tabs: bool) -> str:
        """Consume the lines up to the delimiter line (or the end) and return them."""
        lines: list[str] = []
        while self._pos < len(self._text):
            end = self._text.find("\n", self._pos)
            end = len(self._text) if end == -1 else end
            line = self._text[self._pos:end]
            self._pos = end + 1
            if (line.lstrip("\t") if strip_tabs else line) == delimiter:
                break
            lines.append(line)
        return "\n".join(lines)

    def _skip_comment(self) -> None:
        end = self._text.find("\n", self._pos)
        self._pos = len(self._text) if end == -1 else end

    def _backslash(self) -> None:
        following = self._text[self._pos + 1:self._pos + 2]
        # A backslash-newline joins two lines; it is not part of any word.
        if following != "\n":
            self._append(following)
        self._pos += 2

    def _quoted(self, start: int, quote: str, escape: Callable[[str], str | None]) -> None:
        """Append the text from ``start`` up to the closing ``quote`` (or the
        end of the command). ``escape`` maps the character after a backslash to
        what the pair stands for, or to ``None`` where the backslash is literal."""
        text = self._text
        pos = start
        chars: list[str] = []
        while pos < len(text) and text[pos] != quote:
            escaped = escape(text[pos + 1]) if text[pos] == "\\" and pos + 1 < len(text) else None
            if escaped is None:
                chars.append(text[pos])
                pos += 1
            else:
                chars.append(escaped)
                pos += 2
        self._append("".join(chars))
        self._pos = pos + 1

    def _operator(self) -> None:
        operator = next(op for op in _OPERATORS if self._text.startswith(op, self._pos))
        self._pos += len(operator)
        # A word of digits written against a redirection is its fd (``2>``).
        if self._in_word and operator[0] in "<>" and "".join(self._chars).isdigit():
            operator = "".join(self._chars) + operator
            self._chars.clear()
            self._in_word = False
        self._end_word()
        self._tokens.append(("op", operator))
        bare = operator.lstrip(_FD_DIGITS)
        if bare in _HEREDOC_OPERATORS:
            self._awaiting_delimiter = (len(self._tokens) - 1, bare == "<<-")


def _no_escape(char: str) -> None:
    """Inside single quotes a backslash is an ordinary character."""
    return None


def _double_quote_escape(char: str) -> str | None:
    """Inside double quotes a backslash escapes only ``$``, a backtick, ``"`` and ``\\``."""
    return char if char in _DOUBLE_QUOTE_ESCAPABLE else None


def _ansi_c_escape(char: str) -> str:
    """``$'...'`` decodes its escapes, so ``ed <<< $'a\\nw'`` reaches the
    script check as two lines."""
    return _ANSI_C_ESCAPES.get(char, char)


# ----------------------------------------------------------------------
# Simple commands: words, the files redirections write, stdin text
# ----------------------------------------------------------------------

_WRITE_REDIRECT_RE = re.compile(r"^\d*(?:>|>>|>\||&>|&>>|<>)$")
_DUPLICATE_REDIRECT_RE = re.compile(r"^\d*>&$")


@dataclass(frozen=True)
class _SimpleCommand:
    """One simple command: its words, the files its redirections write, and
    the text it reads on stdin when the command spells that text out (a
    here-string or a heredoc body)."""

    words: tuple[str, ...]
    redirected: tuple[str, ...]
    stdin: tuple[str, ...]


def _commands(lexed: _Lexed) -> Iterator[_SimpleCommand | str]:
    """Each simple command in order, with a subshell's ``(`` and ``)`` yielded
    between them so the scan can undo a ``cd`` made inside one."""
    start = 0
    for index, (kind, text) in enumerate(lexed.tokens):
        is_paren = text in (_SUBSHELL_OPEN, _SUBSHELL_CLOSE)
        if kind != "op" or not (is_paren or text in _COMMAND_SEPARATORS):
            continue
        if index > start:
            yield _simple_command(lexed, start, index)
        if is_paren:
            yield text
        start = index + 1
    if start < len(lexed.tokens):
        yield _simple_command(lexed, start, len(lexed.tokens))


def _simple_command(lexed: _Lexed, start: int, end: int) -> _SimpleCommand:
    """The simple command made of ``lexed.tokens[start:end]``: every operator
    left in that span is a redirection, whose operand is the word after it."""
    words: list[str] = []
    redirected: list[str] = []
    stdin: list[str] = []
    index = start
    while index < end:
        kind, text = lexed.tokens[index]
        if kind == "word":
            words.append(text)
            index += 1
            continue
        following = lexed.tokens[index + 1] if index + 1 < end else ("op", "")
        operand = following[1] if following[0] == "word" else None
        if operand is not None and _redirect_writes(text, operand):
            redirected.append(operand)
        elif text == _HERESTRING_OPERATOR and operand is not None:
            stdin.append(operand)
        elif text.lstrip(_FD_DIGITS) in _HEREDOC_OPERATORS:
            stdin.append(lexed.heredocs.get(index, ""))
        index += 1 if operand is None else 2
    return _SimpleCommand(tuple(words), tuple(redirected), tuple(stdin))


def _redirect_writes(operator: str, operand: str) -> bool:
    """Does the redirection ``operator operand`` write to a file named ``operand``?"""
    if _WRITE_REDIRECT_RE.match(operator):
        return True
    # ``>& file`` writes the file; ``>&2`` and ``>&-`` copy or close an fd.
    return bool(_DUPLICATE_REDIRECT_RE.match(operator)) and not operand.isdigit() and operand != "-"


# ----------------------------------------------------------------------
# Options and operands
# ----------------------------------------------------------------------


def _split_options(
    args: Sequence[str], value_letters: str = ""
) -> tuple[list[str], list[tuple[str, str]]]:
    """Split ``args`` into its operands and its ``(option, value)`` pairs.

    An arg starting with ``-`` is an option, except ``-`` alone (stdin) and
    anything after ``--``. A short-option cluster ending in one of
    ``value_letters`` takes the next arg as its value (``-c BODY``,
    ``-pe BODY``, ``-s 0``); a value written onto its letter (``-oFILE``)
    stays part of the option."""
    operands: list[str] = []
    values: list[tuple[str, str]] = []
    pending_option: str | None = None
    options_ended = False
    for arg in args:
        if pending_option is not None:
            values.append((pending_option, arg))
            pending_option = None
        elif options_ended or arg == "-" or not arg.startswith("-"):
            operands.append(arg)
        elif arg == "--":
            options_ended = True
        elif not arg.startswith("--") and arg[-1] in value_letters:
            pending_option = arg
    return operands, values


def _values_of(values: Iterable[tuple[str, str]], letters: str) -> list[str]:
    """The values of the options among ``values`` whose cluster ends in one of ``letters``."""
    return [value for option, value in values if option[-1] in letters]


def _program_on_stdin(operands: Sequence[str]) -> bool:
    """With no inline program, an interpreter or shell reads its program from
    stdin when it names no script file (``python3 - <<EOF``, ``bash <<EOF``)."""
    return not operands or operands[0] == "-"


# ----------------------------------------------------------------------
# What each recognised tool writes (raw, unresolved operands)
# ----------------------------------------------------------------------

_OPERAND_WRITERS: dict[str, str] = {
    "tee": "",
    "rm": "",
    "unlink": "",
    "shred": "ns",
    "truncate": "sr",
}
"""Tools that write every operand, each with its short options that take a value."""

_COPY_WRITERS: dict[str, str] = {
    "cp": "St",
    "ln": "St",
    "install": "mgoSt",
    # rsync's -t preserves times; it names no target directory.
    "rsync": "e",
}
"""Tools that write their destination, each with its short options that take a value."""

_IN_PLACE_RE: dict[str, re.Pattern[str]] = {
    # The letters allowed before ``i`` take no value, so ``-ni`` / ``-0pi`` /
    # ``-i.bak`` are in-place and ``-es/i/j/`` / ``-MList::Util=min`` are not.
    "sed": re.compile(r"^-[bnrsuzE]*i"),
    "perl": re.compile(r"^-[0-9acnpsStTuUvwWXl]*i"),
}


def _edits_in_place(tool: str, args: Sequence[str]) -> bool:
    if tool == "sed" and any(arg == "--in-place" or arg.startswith("--in-place=") for arg in args):
        return True
    return any(_IN_PLACE_RE[tool].match(arg) for arg in args)


def _non_empty(operands: Iterable[str]) -> list[str]:
    return [operand for operand in operands if operand]


def _sed_writes(args: Sequence[str], stdin: Sequence[str]) -> list[str]:
    if not _edits_in_place("sed", args):
        return []
    operands, values = _split_options(args, "ef")
    # BSD's ``-i ''`` leaves an empty suffix operand behind; it names no file.
    files = _non_empty(operands)
    # Without -e/-f the first operand is the script, not a file.
    return files if _values_of(values, "ef") else files[1:]


def _perl_writes(args: Sequence[str], stdin: Sequence[str]) -> list[str]:
    operands, values = _split_options(args, "eE")
    programs = _values_of(values, "eE")
    in_place: list[str] = []
    if _edits_in_place("perl", args):
        # Without -e the first operand is the program file, not an edited file.
        in_place = operands if programs else operands[1:]
    if not programs and _program_on_stdin(operands):
        programs = list(stdin)
    return in_place + _programs_writes(programs)


@dataclass(frozen=True)
class _Interpreter:
    value_letters: str
    """Short options that take a value."""
    program_letters: str
    """Those whose value is program text."""


_INTERPRETERS: dict[str, _Interpreter] = {
    "python": _Interpreter(value_letters="cmWX", program_letters="c"),
    "ruby": _Interpreter(value_letters="eIrC", program_letters="e"),
    "node": _Interpreter(value_letters="epr", program_letters="ep"),
    "php": _Interpreter(value_letters="rfdc", program_letters="r"),
}
_PYTHON_RE = re.compile(r"^python[0-9.]*$")


def _interpreter_writes(interpreter: _Interpreter, args: Sequence[str], stdin: Sequence[str]) -> list[str]:
    operands, values = _split_options(args, interpreter.value_letters)
    programs = _values_of(values, interpreter.program_letters)
    if not programs and _program_on_stdin(operands):
        programs = list(stdin)
    return _programs_writes(programs)


@dataclass(frozen=True)
class _WriteCall:
    """A call that writes a path it is given: ``head`` matches the call up to
    its first argument, ``written`` names the argument positions it writes
    (0 the first, 1 the second), and ``moded`` marks an open, which writes
    only when the mode argument after the path writes. The one list every
    program-body rule is built from -- the write-call filter, a literal's
    target test and a bound name's -- so the three cannot drift apart."""

    head: str
    written: tuple[int, ...] = (0,)
    moded: bool = False


_WRITE_CALLS: tuple[_WriteCall, ...] = (
    _WriteCall(
        r"\b(?:writeFileSync|writeFile|appendFileSync|appendFile|createWriteStream|"
        r"unlinkSync|rmSync|truncateSync|file_put_contents|File\.write|IO\.write|"
        r"File\.delete|FileUtils\.rm\w*|FileUtils\.touch|os\.remove|os\.truncate)\("
    ),
    _WriteCall(r"\bunlink(?:\(|\s+)"),
    _WriteCall(r"\b(?:os\.replace|shutil\.move|FileUtils\.mv|renameSync|rename)(?:\(|\s+)", (0, 1)),
    _WriteCall(r"\b(?:shutil\.copy\w*|copyFileSync|copyFile|FileUtils\.cp)\(", (1,)),
    # ``open(`` also matches ``os.open(``, ``io.open(``, ``codecs.open(`` and
    # ruby's ``File.open(``.
    _WriteCall(r"\b(?:open|fopen|openSync|File\.new)\(", moded=True),
    # perl's three-argument open: the mode comes before the path.
    _WriteCall(r"""\bopen\s*\(?\s*(?:my\s+)?[\$\w]+\s*,\s*['"](?:\+?>{1,2}|\+<)[^'"]*['"]\s*,\s*"""),
)
# A mode that writes: a literal with w, a, x or +, a mode held in a variable
# (``open(p, mode)``, which a read rarely spells), or os.open's write flags.
_WRITE_MODE = (
    r"""\s*,\s*(?:(?:mode\s*=\s*|flags\s*=\s*)?['"][^'"]*[wax+]|[A-Za-z_]\w*\s*\)|"""
    r"""[\w.|\s]{0,80}\bO_(?:WRONLY|RDWR|CREAT|TRUNC|APPEND)\b)"""
)
# A first argument a second, written one follows: anything up to the comma,
# one level of parentheses deep.
_FIRST_ARG = r"(?:[^,()]|\([^()]*\)){1,200}"
# A pathlib object's writing methods, and the subset a str also answers to.
_PATH_WRITE_METHOD = (
    r"""\.(?:write_text|write_bytes|unlink|touch|rename|replace|"""
    r"""open\(\s*(?:mode\s*=\s*)?['"][^'"]*[wax+])"""
)
_STR_SAFE_WRITE_METHOD = (
    r"""\.(?:write_text|write_bytes|unlink|touch|"""
    r"""open\(\s*(?:mode\s*=\s*)?['"][^'"]*[wax+])"""
)
_PATH_CALL = r"\b(?:\w+\.)?Path\("


def _heads(*, at: int, moded: bool) -> str:
    return "(?:" + "|".join(
        call.head for call in _WRITE_CALLS if at in call.written and call.moded == moded
    ) + ")"


# A write call anywhere in a program body; one makes the body's path tokens
# candidates. Built from the same table as the target rules below, plus the
# pathlib methods and a redirection inside a shell string.
_WRITE_HINT_RE = re.compile(
    "(?:" + "|".join(call.head for call in _WRITE_CALLS) + "|"
    + _PATH_WRITE_METHOD + "|"
    + r"""['"]\s*>|>>)"""
)
# A path a program can name as a file: a whole string literal, after a perl
# open mode if any (``'p'``, ``">>p"``), or a redirection's target inside a
# shell string (``"echo x >> p"``). A path inside prose is not one.
_LITERAL_START_RE = re.compile(r"""(['"])[+<>]*\s*$""")
_REDIRECT_START_RE = re.compile(r""">\s*$""")
# A literal is written only as the TARGET of a write call: reading it, copying
# from it or naming it in text the program writes is not a write.
_FIRST_TARGET_BEFORE_RE = re.compile(_heads(at=0, moded=False) + r"""\s*['"]$""")
_SECOND_TARGET_BEFORE_RE = re.compile(_heads(at=1, moded=False) + r"\s*" + _FIRST_ARG + r"""\s*,\s*['"]$""")
_OPEN_TARGET_BEFORE_RE = re.compile(_heads(at=0, moded=True) + r"""\s*(?:file\s*=\s*)?['"]$""")
_WRITE_MODE_AFTER_RE = re.compile(r"""^['"]""" + _WRITE_MODE)
_PATH_BEFORE_RE = re.compile(_PATH_CALL + r"""\s*['"]$""")
_PATH_WRITE_AFTER_RE = re.compile(r"""^['"]\s*\)\s*""" + _PATH_WRITE_METHOD)
# A literal bound to a name at the start of a statement (``p = 'x'``,
# ``p = Path('x')``, ``const f = 'x'``, ``my $f = "x"``), the statement
# ending with it: a value built from it (``'x' + '.bak'``), a tuple target and
# a keyword argument are not bindings.
_STATEMENT_START = r"(?:[;\n{]|\b(?:const|let|var|my)\b)\s*"
_ASSIGNED_BEFORE_RE = re.compile(
    _STATEMENT_START + r"""(\$?[A-Za-z_]\w*)\s*=\s*(""" + _PATH_CALL + r"""\s*)?['"]$"""
)
_ASSIGNED_AFTER_RE = re.compile(r"""^['"]\s*\)?[ \t]*(?:;|\n|$|#)""")
_CALL_CONTEXT_CHARS = 240
_CONTEXT_CHARS = 40
# The bounds that keep name tracking linear on a 16K body: a name's uses are
# looked for this far after its binding, for this many bindings per program.
_NAME_SCAN_CHARS = 8000
_MAX_TRACKED_NAMES = 64


def _programs_writes(programs: Iterable[str]) -> list[str]:
    return [path for program in programs for path in _program_writes(program)]


def _program_writes(program: str) -> list[str]:
    """The path tokens a program body writes: none unless it holds a write
    call, then each one with an occurrence that names a file and is the target
    of a write call, directly or through a name bound to it."""
    if not _WRITE_HINT_RE.search(program):
        return []
    names = _NameBudget()
    return [
        match.group(1)
        for match in PATH_TOKEN_RE.finditer(program)
        if _may_be_written(program, match, names)
    ]


class _NameBudget:
    """How many bindings one program may still follow (:data:`_MAX_TRACKED_NAMES`)."""

    def __init__(self) -> None:
        self.left = _MAX_TRACKED_NAMES

    def take(self) -> bool:
        self.left -= 1
        return self.left >= 0


def _may_be_written(program: str, match: re.Match[str], names: _NameBudget) -> bool:
    """Is this occurrence of a path token a file the program writes?"""
    start = max(0, match.start() - _CALL_CONTEXT_CHARS)
    # The body's own start counts as a statement start.
    before = ("\n" if start == 0 else "") + program[start:match.start()]
    after = program[match.end():match.end() + _CONTEXT_CHARS]
    if not _names_a_file(before[-_CONTEXT_CHARS:], after):
        return False
    return _written_in_place(before, after) or _written_through_name(program, match, before, after, names)


def _names_a_file(before: str, after: str) -> bool:
    """Is the token a whole string literal, or a redirection's target? A path
    mentioned in prose -- a summary that names the file it summarised -- is not."""
    literal = _LITERAL_START_RE.search(before)
    if literal is not None:
        return after.startswith(literal.group(1))
    return bool(_REDIRECT_START_RE.search(before))


def _written_in_place(before: str, after: str) -> bool:
    """Is the literal itself the target of a write call, or a redirection's
    target (a shell string's ``> p``, perl's ``">>p"``)? Reading it, copying
    from it, or naming it in text the program writes is not a write, so a
    one-liner that reads the path and writes elsewhere goes undetected."""
    if _REDIRECT_START_RE.search(before):
        return True
    if _FIRST_TARGET_BEFORE_RE.search(before) or _SECOND_TARGET_BEFORE_RE.search(before):
        return True
    if _OPEN_TARGET_BEFORE_RE.search(before):
        return bool(_WRITE_MODE_AFTER_RE.match(after))
    return bool(_PATH_BEFORE_RE.search(before) and _PATH_WRITE_AFTER_RE.match(after))


def _written_through_name(
    program: str, match: re.Match[str], before: str, after: str, names: _NameBudget
) -> bool:
    """Is the literal bound to a name that a write call is given, as a whole
    argument, before the name is bound again? A path that reaches the write
    through a container, a loop, a tuple, a function's parameter or a value
    built from pieces is not followed."""
    bound = _ASSIGNED_BEFORE_RE.search(before)
    if bound is None or not _ASSIGNED_AFTER_RE.match(after) or not names.take():
        return False
    name = r"(?<![\w$.])" + re.escape(bound.group(1)) + r"(?![\w$])"
    window = program[match.end():match.end() + _NAME_SCAN_CHARS]
    rebound = re.search(_STATEMENT_START + name + r"\s*=(?!=)", window)
    if rebound is not None:
        window = window[:rebound.start()]
    arg = r"(?:" + _PATH_CALL + r"\s*" + name + r"\s*\)|\bstr\(\s*" + name + r"\s*\)|" + name + r")"
    # The name is the whole argument: what follows ends it (or a perl
    # ``$f or die``), never an operator building a new value from it.
    whole = r"(?=\s*(?:[,);]|$)|\s+(?:(?:or|and)\b|\|\||&&|\{))"
    method = _PATH_WRITE_METHOD if bound.group(2) else _STR_SAFE_WRITE_METHOD
    uses = (
        name + r"\s*" + method,
        _PATH_CALL + r"\s*" + name + r"\s*\)\s*" + _PATH_WRITE_METHOD,
        _heads(at=0, moded=False) + r"\s*" + arg + whole,
        _heads(at=1, moded=False) + r"\s*" + _FIRST_ARG + r"\s*,\s*" + arg + whole,
        _heads(at=0, moded=True) + r"\s*(?:file\s*=\s*)?" + arg + _WRITE_MODE,
    )
    return any(re.search(use, window) for use in uses)


_ED_ADDRESS = r"(?:[0-9,.$;+\-]|'[a-z])*"
_ED_WRITE_RE = re.compile(rf"^{_ED_ADDRESS}[wW]q?(?:\s|$)")
_ED_INPUT_RE = re.compile(rf"^{_ED_ADDRESS}[aic]$")
# The range, when present, starts with a range character: a prefix whose two
# whitespace runs could split one run many ways went quadratic on a long one.
_EX_WRITE_RE = re.compile(
    r"^[:\s]*(?:[%0-9,.$']+\s*)?"
    r"(?:w|write|wq|wqa|wqall|wa|wall|x|xa|xall|xit|exi|exit|up|update|sav|saveas)!?(?:\s|$)"
)


def _ed_script_writes(script: str) -> bool:
    """Does an ed script hold a write command? The lines after an ``a`` /
    ``i`` / ``c`` command are text, up to a lone ``.``."""
    in_text = False
    for line in script.split("\n"):
        if in_text:
            in_text = line != "."
        elif _ED_WRITE_RE.match(line):
            return True
        else:
            in_text = bool(_ED_INPUT_RE.match(line))
    return False


def _ex_script_writes(script: str) -> bool:
    return any(_EX_WRITE_RE.match(command) for command in re.split(r"[\n|]", script))


def _ed_writes(args: Sequence[str], stdin: Sequence[str]) -> list[str]:
    files = _split_options(args, "p")[0]
    # A script the command does not show (piped in, or read from a file) is
    # taken as writing: the read-only ed scripts are the ones spelled out.
    if stdin and not any(_ed_script_writes(script) for script in stdin):
        return []
    return files[:1]


def _ex_writes(args: Sequence[str], stdin: Sequence[str]) -> list[str]:
    operands, values = _split_options(args, "c")
    commands = [value for _, value in values] + [arg[1:] for arg in operands if arg.startswith("+")]
    scripts = commands + list(stdin)
    if scripts and not any(_ex_script_writes(script) for script in scripts):
        return []
    return [arg for arg in operands if not arg.startswith("+")]


def _copy_sources_and_destinations(args: Sequence[str], value_letters: str) -> tuple[list[str], list[str]]:
    """The sources and destinations of a copy-shaped command: the last
    operand, or each source's name inside a ``-t DIR`` /
    ``--target-directory=DIR`` directory or a destination ending in ``/``."""
    operands, values = _split_options(args, value_letters)
    directory = _target_directory(args, values)
    if directory is not None:
        return operands, _inside(directory, operands)
    if len(operands) < 2:
        return operands, []
    sources, destination = operands[:-1], operands[-1]
    if destination.endswith("/"):
        return sources, _inside(destination, sources)
    return sources, [destination]


def _target_directory(args: Sequence[str], values: Iterable[tuple[str, str]]) -> str | None:
    """The directory a GNU ``-t DIR`` / ``--target-directory=DIR`` names, if any."""
    long_form = [arg.removeprefix("--target-directory=") for arg in args if arg.startswith("--target-directory=")]
    directories = _values_of(values, "t") + long_form
    return directories[0] if directories else None


def _inside(directory: str, sources: Iterable[str]) -> list[str]:
    return [posixpath.join(directory, posixpath.basename(source)) for source in sources]


def _move_writes(args: Sequence[str], stdin: Sequence[str]) -> list[str]:
    # A move writes its destination and removes its sources.
    sources, destinations = _copy_sources_and_destinations(args, "St")
    return destinations + sources


def _dd_writes(args: Sequence[str], stdin: Sequence[str]) -> list[str]:
    return [arg.removeprefix("of=") for arg in args if arg.startswith("of=")]


def _sort_writes(args: Sequence[str], stdin: Sequence[str]) -> list[str]:
    outputs = [arg.removeprefix("--output=") for arg in args if arg.startswith("--output=")]
    for index, arg in enumerate(args):
        if arg == "-o" and index + 1 < len(args):
            outputs.append(args[index + 1])
        elif arg.startswith("-o") and len(arg) > 2:
            outputs.append(arg[2:])
    return outputs


def _patch_writes(args: Sequence[str], stdin: Sequence[str]) -> list[str]:
    operands, values = _split_options(args, "iopdrDBVYzFg")
    if _values_of(values, "d"):
        return []  # -d moves to another directory first; not followed
    outputs = _values_of(values, "o")
    # -o writes the patched result there instead of over the original.
    return outputs if outputs else operands[:1]


_GIT_WORKTREE_VERBS: dict[str, str] = {
    "checkout": "bB",
    "restore": "s",
    "rm": "",
    "mv": "",
}
"""git verbs that write the files they name, each with its short options that take a value."""


def _git_writes(args: Sequence[str], stdin: Sequence[str]) -> list[str]:
    # A global option before the verb (``git -C dir ...``) is not followed.
    if not args or args[0] not in _GIT_WORKTREE_VERBS:
        return []
    verb, rest = args[0], args[1:]
    if _git_index_only(verb, rest):
        return []
    return _split_options(rest, _GIT_WORKTREE_VERBS[verb])[0]


def _git_index_only(verb: str, args: Sequence[str]) -> bool:
    """Does this ``git rm`` / ``git restore`` touch the index and leave the file alone?"""
    if verb == "rm":
        return "--cached" in args
    if verb != "restore":
        return False
    staged = any(arg in ("--staged", "-S") for arg in args)
    return staged and not any(arg in ("--worktree", "-W") for arg in args)


_ToolWrites = Callable[[Sequence[str], Sequence[str]], list[str]]

_SPECIAL_WRITERS: dict[str, _ToolWrites] = {
    "sed": _sed_writes,
    "perl": _perl_writes,
    "ed": _ed_writes,
    "ex": _ex_writes,
    "mv": _move_writes,
    "dd": _dd_writes,
    "sort": _sort_writes,
    "patch": _patch_writes,
    "git": _git_writes,
}


def _tool_writes(name: str, args: Sequence[str], stdin: Sequence[str]) -> list[str]:
    """The raw paths the tool ``name`` writes when run with ``args``."""
    if name in _OPERAND_WRITERS:
        return _split_options(args, _OPERAND_WRITERS[name])[0]
    if name in _COPY_WRITERS:
        return _copy_sources_and_destinations(args, _COPY_WRITERS[name])[1]
    interpreter = _INTERPRETERS.get("python" if _PYTHON_RE.match(name) else name)
    if interpreter is not None:
        return _interpreter_writes(interpreter, args, stdin)
    special = _SPECIAL_WRITERS.get(name)
    return special(args, stdin) if special is not None else []


# ----------------------------------------------------------------------
# The scan: commands, nested shells, the working directory
# ----------------------------------------------------------------------

_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_PREFIX_WORDS = frozenset({
    # Wrappers that run the rest of the words as the command. Unlike the read
    # detector, ``eval`` is not one: its arguments are a command string,
    # scanned as a command of its own.
    "command", "builtin", "exec", "sudo", "nohup", "time", "env",
    # Grammar words a command follows inside the same segment.
    "!", "{", "if", "then", "elif", "else", "do", "while", "until",
})
_SHELLS = frozenset({"bash", "sh", "zsh", "dash"})
_UNFOLLOWED_DIRECTORY_CHANGES = frozenset({"pushd", "popd"})
_UNRESOLVABLE_CHARS = frozenset("$`*?[")
_MAX_NESTING = 3
"""How deep ``eval`` and ``bash -c`` bodies are scanned inside one another."""


def _command_words(words: Sequence[str]) -> Sequence[str]:
    """``words`` from the command name on: leading assignments, wrappers and
    grammar words dropped."""
    index = 0
    while index < len(words) and (words[index] in _PREFIX_WORDS or _ASSIGNMENT_RE.match(words[index])):
        index += 1
    return words[index:]


def _nested_scripts(name: str, args: Sequence[str], stdin: Sequence[str]) -> list[str]:
    """The command text ``eval`` or a shell runs, scanned like the outer command."""
    if name == "eval":
        return [" ".join(args)]
    if name not in _SHELLS:
        return []
    operands, values = _split_options(args, "c")
    bodies = _values_of(values, "c")
    if bodies:
        return bodies[:1]
    return list(stdin) if _program_on_stdin(operands) else []


def _scan(command: str, cwd: str | None, roots: tuple[str, ...], depth: int) -> list[str]:
    """The workspace-relative paths ``command`` writes when run from ``cwd``
    (``None``: a directory the scan cannot name), before the tracked filter."""
    if depth > _MAX_NESTING:
        return []
    written: list[str] = []
    saved_cwds: list[str | None] = []
    for item in _commands(_Lexer(command).lex()):
        if isinstance(item, _SimpleCommand):
            paths, cwd = _simple_command_writes(item, cwd, roots, depth)
            written.extend(paths)
        elif item == _SUBSHELL_OPEN:
            saved_cwds.append(cwd)
        elif saved_cwds:
            cwd = saved_cwds.pop()
    return written


def _simple_command_writes(
    command: _SimpleCommand, cwd: str | None, roots: tuple[str, ...], depth: int
) -> tuple[list[str], str | None]:
    """The paths one simple command writes, and the directory the next one runs in."""
    written = _resolve_files(command.redirected, cwd, roots)
    words = _command_words(command.words)
    if not words:
        return written, cwd
    name, args = posixpath.basename(words[0]), words[1:]
    if name == "cd":
        return written, _cd(args, cwd, roots)
    if name in _UNFOLLOWED_DIRECTORY_CHANGES:
        return written, None
    nested = _nested_scripts(name, args, command.stdin)
    if nested:
        return written + [path for script in nested for path in _scan(script, cwd, roots, depth + 1)], cwd
    return written + _resolve_files(_tool_writes(name, args, command.stdin), cwd, roots), cwd


def _cd(args: Sequence[str], cwd: str | None, roots: tuple[str, ...]) -> str | None:
    """Where a ``cd`` goes, workspace-relative (``""`` for the root), or
    ``None`` when the command does not spell it out: ``cd`` alone goes home and
    ``cd -`` to the previous directory."""
    operands = _split_options(args)[0]
    if not operands or operands[0] == "-":
        return None
    target = _workspace_path(operands[0], cwd, roots)
    return "" if target == "." else target


def _resolve_files(raws: Iterable[str], cwd: str | None, roots: tuple[str, ...]) -> list[str]:
    resolved = (_workspace_path(raw, cwd, roots) for raw in raws)
    return [path for path in resolved if path is not None and path != "."]


def _workspace_path(raw: str, cwd: str | None, roots: tuple[str, ...]) -> str | None:
    """``raw`` as a normalized workspace-relative path (``"."`` for the root),
    or ``None`` when the command does not spell one out: an expansion, a glob,
    a home path, a path outside the workspace, or a relative path once the
    directory is unknown."""
    if not raw or raw.startswith("~") or any(char in _UNRESOLVABLE_CHARS for char in raw):
        return None
    if raw.startswith("/"):
        relative = _under_root(raw, roots)
    else:
        relative = None if cwd is None else posixpath.join(cwd, raw)
    if relative is None:
        return None
    normalized = posixpath.normpath(relative)
    if normalized == ".." or normalized.startswith(("../", "/")):
        return None
    return normalized


def _under_root(path: str, roots: tuple[str, ...]) -> str | None:
    for root in roots:
        if path == root or path.startswith(root + "/"):
            return path[len(root) + 1:]
    return None


def _root_aliases(root: str | None) -> tuple[str, ...]:
    """The spellings of the workspace root an absolute path may start with.
    On macOS ``/tmp`` and ``/var`` are links into ``/private``, and a command
    may use either spelling while the coordinator holds the resolved one."""
    if not root:
        return ()
    root = root.rstrip("/")
    aliases = {root}
    if root.startswith("/private/"):
        aliases.add(root.removeprefix("/private"))
    elif root.startswith(("/tmp/", "/var/")):
        aliases.add("/private" + root)
    return tuple(sorted(aliases))


def detect_tracked_writes(
    command: str,
    is_tracked: Callable[[str], bool],
    *,
    root: str | None = None,
    cwd: str = "",
) -> list[str]:
    """Return the tracked-artifact paths the command would write.

    Args:
        command: Raw Bash command string from CC's ``tool_input.command``.
        is_tracked: Returns True for a workspace-relative path the policy
            tracks (e.g. ``coordinator.policy.is_tracked``).
        root: The workspace root, to resolve absolute paths; without it an
            absolute path is never resolved.
        cwd: The shell's directory relative to ``root`` (``""`` for the root).
            A ``cd`` inside the command moves it. The coordinator is not told
            the session's directory and passes the root.

    Returns:
        Workspace-relative tracked paths, deduplicated, in first-occurrence
        order. Empty if none is detected; false negatives are expected for the
        forms the module docstring lists as out of scope.
    """
    start = posixpath.normpath(cwd) if cwd else ""
    found = _scan(command, "" if start == "." else start, _root_aliases(root), 0)
    return [path for path in dict.fromkeys(found) if is_tracked(path)]


__all__ = ["detect_tracked_writes"]
