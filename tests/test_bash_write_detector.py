# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Tests for the Bash WRITE detector (#185: a handoff's giver writing the
handed-off path through the shell).

- Every write form of the two labelled corpora the detector was measured
  against is detected, with exactly its labelled paths; every read-only form of
  both is not.
- ``cd`` handling: inside the command, undone by a subshell, unfollowed when the
  target is not spelled out; the ``cwd`` parameter.
- ``eval``, ``bash -c``, shell and interpreter heredocs, interpreter one-liners,
  and the nesting bound.
- Lexer edge cases: quoting, escapes, line continuations, ``>&``, fds.
- The documented misses, each pinned as NOT detected: a change that starts or
  stops detecting one is then a decision, not an accident.

The command literals are frozen copies of the labelled corpora (written before,
and independently of, this module), plus the read-only forms a code review
found answered as writes -- never derived from the code under test.
``{root}`` in a literal stands for :data:`_ROOT`.
"""
from __future__ import annotations

import shlex
import time

import pytest

from ccs.adapters.claude_code.bash_write_detector import detect_tracked_writes

_ROOT = "/private/tmp/ws-d2"
_PLAN = "docs/plans/plan.md"

#: FROZEN: the tracked set the corpora were labelled against.
_TRACKED: frozenset[str] = frozenset({
    _PLAN,
    "docs/specs/api.md",
    "docs/brainstorms/idea.md",
    "CLAUDE.md",
    "AGENTS.md",
    "task.md",
    "plan.md",
})


def _detect(command: str, cwd: str = "") -> list[str]:
    return detect_tracked_writes(
        command.replace("{root}", _ROOT), _TRACKED.__contains__, root=_ROOT, cwd=cwd
    )


# ----------------------------------------------------------------------
# The labelled corpora
# ----------------------------------------------------------------------

#: FROZEN: (command, the paths it writes, the session's directory).
_WRITE_FORMS: tuple[tuple[str, list[str], str], ...] = (
    ("echo '- gamma' >> docs/plans/plan.md", ['docs/plans/plan.md'], ''),
    ('echo x > docs/plans/plan.md', ['docs/plans/plan.md'], ''),
    ('echo "- alpha" >>docs/plans/plan.md', ['docs/plans/plan.md'], ''),
    ("printf '%s\\n' '- a' >> docs/plans/plan.md", ['docs/plans/plan.md'], ''),
    ("printf 'x\\n' > CLAUDE.md", ['CLAUDE.md'], ''),
    ("printf -- '- gamma\\n' >> docs/plans/plan.md", ['docs/plans/plan.md'], ''),
    ('echo x | tee docs/plans/plan.md', ['docs/plans/plan.md'], ''),
    ('echo x | tee -a docs/plans/plan.md', ['docs/plans/plan.md'], ''),
    ('echo x | tee -a docs/plans/plan.md > /dev/null', ['docs/plans/plan.md'], ''),
    ("tee -a docs/plans/plan.md <<< '- gamma'", ['docs/plans/plan.md'], ''),
    ('cat /tmp/x | tee docs/plans/plan.md CLAUDE.md', ['docs/plans/plan.md', 'CLAUDE.md'], ''),
    ("sed -i '' 's/a/b/' docs/plans/plan.md", ['docs/plans/plan.md'], ''),
    ("sed -i 's/a/b/' docs/plans/plan.md", ['docs/plans/plan.md'], ''),
    ("sed -i.bak 's/a/b/' docs/plans/plan.md", ['docs/plans/plan.md'], ''),
    ("sed -i -e '$a\\- gamma' docs/plans/plan.md", ['docs/plans/plan.md'], ''),
    ("sed --in-place 's/a/b/' docs/specs/api.md", ['docs/specs/api.md'], ''),
    ("perl -pi -e 's/a/b/' docs/plans/plan.md", ['docs/plans/plan.md'], ''),
    ("perl -i -pe 's/a/b/' docs/plans/plan.md", ['docs/plans/plan.md'], ''),
    ('cat <<EOF > docs/plans/plan.md\n# Plan\n- gamma\nEOF', ['docs/plans/plan.md'], ''),
    ("cat <<'EOF' >> docs/plans/plan.md\n- gamma\nEOF", ['docs/plans/plan.md'], ''),
    ('cat >> docs/plans/plan.md <<EOF\n- gamma\nEOF', ['docs/plans/plan.md'], ''),
    ('cp /tmp/new.md docs/plans/plan.md', ['docs/plans/plan.md'], ''),
    ('cp -f draft.md docs/plans/plan.md', ['docs/plans/plan.md'], ''),
    ('cp /tmp/plan.md docs/plans/', ['docs/plans/plan.md'], ''),
    ('mv /tmp/new.md docs/plans/plan.md', ['docs/plans/plan.md'], ''),
    ('mv docs/plans/plan.md.tmp docs/plans/plan.md', ['docs/plans/plan.md'], ''),
    ('mv docs/plans/plan.md docs/plans/old.txt', ['docs/plans/plan.md'], ''),
    ('python3 -c "open(\'docs/plans/plan.md\',\'a\').write(\'- gamma\\n\')"', ['docs/plans/plan.md'], ''),
    ('python3 -c \'with open("docs/plans/plan.md","w") as f: f.write("x")\'', ['docs/plans/plan.md'], ''),
    ('python -c "import pathlib; pathlib.Path(\'docs/plans/plan.md\').write_text(\'x\')"', ['docs/plans/plan.md'], ''),
    ("python3 - <<'PY'\nopen('docs/plans/plan.md','a').write('x')\nPY", ['docs/plans/plan.md'], ''),
    ('node -e "require(\'fs\').appendFileSync(\'docs/plans/plan.md\',\'x\')"', ['docs/plans/plan.md'], ''),
    ('ruby -e \'File.write("docs/plans/plan.md","x")\'', ['docs/plans/plan.md'], ''),
    ('truncate -s 0 docs/plans/plan.md', ['docs/plans/plan.md'], ''),
    ('dd if=/dev/null of=docs/plans/plan.md', ['docs/plans/plan.md'], ''),
    ('ln -sf /tmp/other.md docs/plans/plan.md', ['docs/plans/plan.md'], ''),
    ('git checkout -- docs/plans/plan.md', ['docs/plans/plan.md'], ''),
    ('git checkout HEAD docs/plans/plan.md', ['docs/plans/plan.md'], ''),
    ('git restore docs/plans/plan.md', ['docs/plans/plan.md'], ''),
    ('git rm -q docs/plans/plan.md', ['docs/plans/plan.md'], ''),
    ('git mv docs/plans/plan.md docs/plans/old.txt', ['docs/plans/plan.md'], ''),
    ('rm docs/plans/plan.md', ['docs/plans/plan.md'], ''),
    ('rm -f docs/plans/plan.md CLAUDE.md', ['docs/plans/plan.md', 'CLAUDE.md'], ''),
    ("cd docs/plans && echo '- gamma' >> plan.md", ['docs/plans/plan.md'], ''),
    ('cd docs/plans; echo x >> plan.md', ['docs/plans/plan.md'], ''),
    ('cd docs && echo x >> plans/plan.md', ['docs/plans/plan.md'], ''),
    ("tail -1 docs/plans/plan.md && echo '- gamma' >> docs/plans/plan.md", ['docs/plans/plan.md'], ''),
    ("cat docs/plans/plan.md; echo '- gamma' >> docs/plans/plan.md", ['docs/plans/plan.md'], ''),
    ("echo '- gamma' >> docs/plans/plan.md && tail -3 docs/plans/plan.md", ['docs/plans/plan.md'], ''),
    ("grep -q gamma docs/plans/plan.md || echo '- gamma' >> docs/plans/plan.md", ['docs/plans/plan.md'], ''),
    ("echo -e '\\n- gamma' >> docs/plans/plan.md", ['docs/plans/plan.md'], ''),
    ('echo x 1>> docs/plans/plan.md', ['docs/plans/plan.md'], ''),
    ('echo x >| docs/plans/plan.md', ['docs/plans/plan.md'], ''),
    ('echo x &> docs/plans/plan.md', ['docs/plans/plan.md'], ''),
    ('sort -o docs/plans/plan.md docs/plans/plan.md', ['docs/plans/plan.md'], ''),
    ("awk '{print}' docs/plans/plan.md > /tmp/x && mv /tmp/x docs/plans/plan.md", ['docs/plans/plan.md'], ''),
    ("sed 's/a/b/' docs/plans/plan.md > /tmp/x && cp /tmp/x docs/plans/plan.md", ['docs/plans/plan.md'], ''),
    ('{ echo a; echo b; } >> docs/plans/plan.md', ['docs/plans/plan.md'], ''),
    ("(echo '- gamma') >> docs/plans/plan.md", ['docs/plans/plan.md'], ''),
    ('echo x >> ./docs/plans/plan.md', ['docs/plans/plan.md'], ''),
    ('echo x >> "docs/plans/plan.md"', ['docs/plans/plan.md'], ''),
    ('bash -c "echo x >> docs/plans/plan.md"', ['docs/plans/plan.md'], ''),
    ("sh -c 'echo x >> docs/plans/plan.md'", ['docs/plans/plan.md'], ''),
    ("echo '- gamma' | cat >> docs/plans/plan.md", ['docs/plans/plan.md'], ''),
    ("ed -s docs/plans/plan.md <<< $'a\\n- gamma\\n.\\nw'", ['docs/plans/plan.md'], ''),
    ('echo x >> docs/plans/plan.md 2>/dev/null', ['docs/plans/plan.md'], ''),
    ('echo x 2>&1 >> docs/plans/plan.md', ['docs/plans/plan.md'], ''),
    ('FOO=1 echo x >> docs/plans/plan.md', ['docs/plans/plan.md'], ''),
    ('exec 3>>docs/plans/plan.md; echo x >&3', ['docs/plans/plan.md'], ''),
    ('patch docs/plans/plan.md < fix.diff', ['docs/plans/plan.md'], ''),
    ('echo x >> docs/specs/api.md', ['docs/specs/api.md'], ''),
    ('echo x > task.md', ['task.md'], ''),
    ('echo x >> CLAUDE.md', ['CLAUDE.md'], ''),
    ("echo '- gamma' >> {root}/docs/plans/plan.md", ['docs/plans/plan.md'], ''),
    ("echo '- gamma' >> {root}/docs/plans/plan.md && tail -3 {root}/docs/plans/plan.md", ['docs/plans/plan.md'], ''),
    ("echo '- alpha' >> {root}/docs/plans/plan.md && tail -3 plan.md", ['docs/plans/plan.md'], 'docs/plans'),
    ('echo x >> /tmp/ws-d2/docs/plans/plan.md', ['docs/plans/plan.md'], ''),
    ("echo '- gamma' >> plan.md", ['docs/plans/plan.md'], 'docs/plans'),
    ("echo '- gamma' >> plan.md && tail -3 plan.md", ['docs/plans/plan.md'], 'docs/plans'),
    ('echo x>>docs/plans/plan.md', ['docs/plans/plan.md'], ''),
    ('echo x >>  docs/plans//plan.md', ['docs/plans/plan.md'], ''),
    ("printf '\\n- delta\\n' | tee -a -- docs/plans/plan.md", ['docs/plans/plan.md'], ''),
    ('cat /dev/null > docs/plans/plan.md', ['docs/plans/plan.md'], ''),
    (': > docs/plans/plan.md', ['docs/plans/plan.md'], ''),
    ('> docs/plans/plan.md', ['docs/plans/plan.md'], ''),
    ('echo x >> docs/plans/plan.md; git add -A', ['docs/plans/plan.md'], ''),
    ("sed -E -i '' 's/x/y/' docs/plans/plan.md", ['docs/plans/plan.md'], ''),
    ("perl -0pi -e 's/x/y/' docs/plans/plan.md", ['docs/plans/plan.md'], ''),
    ("perl -i.orig -pe 's/x/y/' docs/plans/plan.md", ['docs/plans/plan.md'], ''),
    ('cp -p /tmp/a.md ./docs/plans/plan.md', ['docs/plans/plan.md'], ''),
    ('cp /tmp/a.md docs/plans/plan.md && git diff', ['docs/plans/plan.md'], ''),
    ('install -D /tmp/a docs/plans/plan.md', ['docs/plans/plan.md'], ''),
    ('mv -f /tmp/a docs/plans/plan.md', ['docs/plans/plan.md'], ''),
    ('rsync -a /tmp/plan.md docs/plans/plan.md', ['docs/plans/plan.md'], ''),
    ('python3 -c "from pathlib import Path; p=Path(\'docs/plans/plan.md\'); p.write_text(p.read_text()+\'x\')"', ['docs/plans/plan.md'], ''),
    ('python3 -c "import os; os.remove(\'docs/plans/plan.md\')"', ['docs/plans/plan.md'], ''),
    ("python3 <<EOF\nwith open('docs/plans/plan.md','a') as f:\n    f.write('x')\nEOF", ['docs/plans/plan.md'], ''),
    ('node -e "const fs=require(\'fs\');fs.writeFileSync(\'docs/plans/plan.md\',\'x\')"', ['docs/plans/plan.md'], ''),
    ('perl -e \'open(my $f, ">>", "docs/plans/plan.md"); print $f "x"\'', ['docs/plans/plan.md'], ''),
    ("ex -s +'$a|- gamma' +wq docs/plans/plan.md", ['docs/plans/plan.md'], ''),
    ('git checkout origin/main -- docs/plans/plan.md', ['docs/plans/plan.md'], ''),
    ('git restore --source=HEAD~1 docs/plans/plan.md', ['docs/plans/plan.md'], ''),
    ('git show HEAD~1:docs/plans/plan.md > docs/plans/plan.md', ['docs/plans/plan.md'], ''),
    ("cd docs/plans && sed -i '' 's/x/y/' plan.md", ['docs/plans/plan.md'], ''),
    ("cd docs/plans/ && printf 'x' >> ./plan.md", ['docs/plans/plan.md'], ''),
    ('(cd docs/plans && echo x >> plan.md)', ['docs/plans/plan.md'], ''),
    ('echo x | sudo tee -a docs/plans/plan.md', ['docs/plans/plan.md'], ''),
    ('xargs -I{} echo {} >> docs/plans/plan.md < /tmp/list', ['docs/plans/plan.md'], ''),
    ('for l in a b; do echo $l >> docs/plans/plan.md; done', ['docs/plans/plan.md'], ''),
    ('if true; then echo x >> docs/plans/plan.md; fi', ['docs/plans/plan.md'], ''),
    ('echo x >> docs/plans/plan.md # append', ['docs/plans/plan.md'], ''),
    ('echo x > CLAUDE.md && echo y >> AGENTS.md', ['CLAUDE.md', 'AGENTS.md'], ''),
    ('cat docs/specs/a.md >> docs/plans/plan.md', ['docs/plans/plan.md'], ''),
    ('head -3 /tmp/x >> docs/plans/plan.md', ['docs/plans/plan.md'], ''),
    ('uniq docs/plans/plan.md /tmp/u && mv /tmp/u docs/plans/plan.md', ['docs/plans/plan.md'], ''),
    ('echo x >> docs/brainstorms/idea.md', ['docs/brainstorms/idea.md'], ''),
    ("zsh -c 'print x >> docs/plans/plan.md'", ['docs/plans/plan.md'], ''),
)

#: FROZEN: (command, the session's directory) -- each writes no tracked path.
_READ_ONLY_FORMS: tuple[tuple[str, str], ...] = (
    ('cat docs/plans/plan.md', ''),
    ('head -5 docs/plans/plan.md', ''),
    ('tail -1 docs/plans/plan.md', ''),
    ('tail -c 50 docs/plans/plan.md | od -c | tail -3', ''),
    ('wc -l docs/plans/plan.md', ''),
    ('grep -n x docs/plans/plan.md', ''),
    ('rg gamma docs/plans', ''),
    ('sed -n 1,5p docs/plans/plan.md', ''),
    ("sed 's/a/b/' docs/plans/plan.md", ''),
    ("awk '{print}' docs/plans/plan.md", ''),
    ('cat docs/plans/plan.md > /tmp/copy.md', ''),
    ('cp docs/plans/plan.md /tmp/backup.md', ''),
    ('diff docs/plans/plan.md /tmp/x', ''),
    ('git diff docs/plans/plan.md', ''),
    ('git log --oneline -- docs/plans/plan.md', ''),
    ('git status', ''),
    ('ls docs/plans', ''),
    ('ls -la', ''),
    ('echo hello', ''),
    ("echo 'plan.md > done'", ''),
    ("grep '>>' docs/plans/plan.md", ''),
    ("echo 'echo x >> docs/plans/plan.md'", ''),
    ('cat docs/plans/plan.md | tee /tmp/out.md', ''),
    ('tee /tmp/log < docs/plans/plan.md', ''),
    ('python3 -c "print(open(\'docs/plans/plan.md\').read())"', ''),
    ("perl -ne 'print' docs/plans/plan.md", ''),
    ('sort docs/plans/plan.md', ''),
    ('sort docs/plans/plan.md -o /tmp/sorted', ''),
    ('md5 docs/plans/plan.md', ''),
    ('shasum docs/plans/plan.md', ''),
    ('stat docs/plans/plan.md', ''),
    ('test -f docs/plans/plan.md && echo yes', ''),
    ('[ -s docs/plans/plan.md ]', ''),
    ('file docs/plans/plan.md', ''),
    ('less docs/plans/plan.md', ''),
    ('nl docs/plans/plan.md', ''),
    ('cat -n docs/plans/plan.md', ''),
    ('echo x >> /tmp/notes.md', ''),
    ('echo x > other.txt', ''),
    ('echo x >> docs/plans/notes.txt', ''),
    ('echo x >> README.md', ''),
    ("sed -i '' 's/a/b/' README.md", ''),
    ('cp docs/plans/plan.md docs/plans/plan.md.bak', ''),
    ('mv /tmp/a /tmp/b', ''),
    ('touch /tmp/x', ''),
    ('mkdir -p docs/plans/new', ''),
    ('git add docs/plans/plan.md', ''),
    ("git commit -m 'update plan.md'", ''),
    ('git show HEAD:docs/plans/plan.md', ''),
    ('git blame docs/plans/plan.md', ''),
    ('npm test', ''),
    ('pytest -q', ''),
    ('python3 script.py > out.log', ''),
    ('agent-coherence-status', ''),
    ('agent-coherence-transfer --session G --successor 1234 docs/plans/plan.md', ''),
    ('agent-coherence-withdraw --session G docs/plans/plan.md', ''),
    ('agent-coherence-accept docs/plans/plan.md', ''),
    ('find docs -name plan.md', ''),
    ('cat docs/plans/plan.md 2>/dev/null', ''),
    ('cat docs/plans/plan.md >/dev/null', ''),
    ('wc -l docs/plans/plan.md > /tmp/count', ''),
    ('grep -c x docs/plans/plan.md >> /tmp/log', ''),
    ('cd docs/plans && cat plan.md', ''),
    ('cd docs/plans && tail -c 50 plan.md | od -c | tail -3', ''),
    ("echo '- gamma'", ''),
    ('date >> /tmp/log', ''),
    ('ls > docs/plans/listing.txt', ''),
    ('cat docs/plans/plan.md | wc -l', ''),
    ('ls -R .coherence | head -30; cat docs/plans/plan.md', ''),
    ('printenv CLAUDE_CODE_SESSION_ID', ''),
    ('cat {root}/docs/plans/plan.md', ''),
    ('tail -3 plan.md', 'docs/plans'),
    ('python3 -c "import sys; print(sys.version)" > /tmp/v.txt', ''),
    ('git checkout -b feature/x', ''),
    ('git stash list', ''),
    ('cp docs/plans/plan.md /tmp/', ''),
    ('cp docs/plans/plan.md docs/plans/plan-copy.txt', ''),
    ('mv /tmp/plan.md /tmp/plan2.md', ''),
    ('tee -a /tmp/log.txt < docs/plans/plan.md', ''),
    ("sed -n '/gamma/p' docs/plans/plan.md", ''),
    ("sed -e 's/x/y/' docs/plans/plan.md > /tmp/out", ''),
    ("perl -pe 's/x/y/' docs/plans/plan.md", ''),
    ("perl -ne 'print if /x/' docs/plans/plan.md > /tmp/y", ''),
    ('python3 -c "import json; json.dump({}, open(\'/tmp/o.json\',\'w\')); print(open(\'docs/plans/plan.md\').read())"', ''),
    ('echo "update docs/plans/plan.md > later"', ''),
    ('git diff > /tmp/patch.diff', ''),
    ('git log -p docs/plans/plan.md > /tmp/history', ''),
    ("grep -rn '>> docs/plans/plan.md' .", ''),
    ('rm -f /tmp/plan.md', ''),
    ('rm docs/plans/plan.md.bak', ''),
    ('git checkout dev', ''),
    ('git restore --staged docs/plans/plan.md', ''),
    ('git rm --cached docs/plans/plan.md', ''),
    ('cat docs/plans/plan.md | sort -o /tmp/s', ''),
    ('dd if=docs/plans/plan.md of=/tmp/x', ''),
    ('ln -s docs/plans/plan.md /tmp/link', ''),
    ('diff <(cat docs/plans/plan.md) /tmp/x', ''),
    ('cat docs/plans/plan.md > docs/plans/plan.txt', ''),
    ('echo x >> docs/plans/archive/notes.txt', ''),
    ('truncate -s 0 /tmp/x', ''),
    ('wc -c < docs/plans/plan.md', ''),
    ('head -n 3 docs/plans/plan.md >> /tmp/summary.md', ''),
    ('ls docs/plans/*.md', ''),
    ("awk 'END{print NR}' docs/plans/plan.md", ''),
    ('python3 -m json.tool package.json', ''),
    ('node -e "console.log(require(\'fs\').readFileSync(\'docs/plans/plan.md\',\'utf8\'))"', ''),
    ("bash -c 'cat docs/plans/plan.md'", ''),
    ('sh -c "wc -l docs/plans/plan.md > /tmp/c"', ''),
    ("ed -s docs/plans/plan.md <<< ',p'", ''),
    ("ex -s -c '%p' -c 'q!' docs/plans/plan.md", ''),
    ('cd docs/plans && ls -la', ''),
    ('cd /tmp && echo x >> plan.md', ''),
    ('echo "rm docs/plans/plan.md"', ''),
    ("printf '%s\\n' docs/plans/plan.md", ''),
    ('git stash', ''),
    ('agent-coherence-decline docs/plans/plan.md', ''),
    ("git add docs/plans/plan.md && git commit -m 'echo > plan.md'", ''),
    # Read-only one-liners a code review found answered as writes: a string
    # method or a flag taken for a write call, a copy's source, a summary
    # file whose text names the file it summarised.
    ("python3 -c \"from pathlib import Path; print(Path('docs/plans/plan.md').read_text().replace('a','b'))\"", ''),
    ("python3 -c \"p = 'docs/plans/plan.md'; print(open(p).read().replace('a','b'))\"", ''),
    ("python3 -c \"import shutil; shutil.copyfile('docs/plans/plan.md','/tmp/snap.md')\"", ''),
    ("python3 -c \"import subprocess; print(subprocess.check_output(['grep','-i','todo','docs/plans/plan.md']))\"", ''),
    ("python3 - <<'PY'\ntext = open('docs/plans/plan.md').read()\n"
     "open('/tmp/summary.md', 'w').write('Summary of docs/plans/plan.md: ' + str(len(text)))\nPY", ''),
)


@pytest.mark.parametrize(("command", "expected", "cwd"), _WRITE_FORMS)
def test_a_labelled_write_form_is_detected_with_its_paths(
    command: str, expected: list[str], cwd: str
) -> None:
    assert _detect(command, cwd) == expected


@pytest.mark.parametrize(("command", "cwd"), _READ_ONLY_FORMS)
def test_a_labelled_read_only_form_is_not_detected(command: str, cwd: str) -> None:
    assert _detect(command, cwd) == []


def test_the_corpora_keep_their_size() -> None:
    """Cardinality pinned apart from the lists, so a dropped row is noticed."""
    assert (len(_WRITE_FORMS), len(_READ_ONLY_FORMS)) == (117, 122)


# ----------------------------------------------------------------------
# The documented misses: not detected, by design
# ----------------------------------------------------------------------

#: FROZEN: write commands the module docstring declares out of scope.
_DOCUMENTED_MISSES: tuple[str, ...] = (
    # Writer tools the detector does not know.
    "gsed -i 's/x/y/' docs/plans/plan.md",
    "awk -i inplace '{print}' docs/plans/plan.md",
    "vim -c '$put =\"x\"' -c wq docs/plans/plan.md",
    "curl -s https://example.com/plan.md -o docs/plans/plan.md",
    "wget -O docs/plans/plan.md https://example.com/x",
    "npx prettier --write docs/plans/plan.md",
    "dos2unix docs/plans/plan.md",
    "python3 fix.py",
    "node --eval \"require('fs').writeFileSync('docs/plans/plan.md','x')\"",
    # Paths built from variables or command substitution, and globs.
    'echo "- gamma" >> "$PWD/docs/plans/plan.md"',
    "echo x >> $(git rev-parse --show-toplevel)/docs/plans/plan.md",
    "echo x > docs/plans/*.md",
    "echo x >> ~/docs/plans/plan.md",
    # Directory changes the command does not spell out.
    "pushd docs/plans && echo x >> plan.md && popd",
    'cd "$DIR" && echo x >> plan.md',
    "cd - && echo x >> plan.md",
    "cd && echo x >> plan.md",
    # File lists made at run time, recursive deletes, unspelled directories.
    "find docs -name plan.md -exec sed -i 's/a/b/' {} +",
    "ls docs/plans/*.md | xargs rm",
    "rm -rf docs/plans",
    "cp /tmp/plan.md docs/plans",
    # git verbs that rewrite files without naming them, and a global option.
    "git reset --hard",
    "git stash pop",
    "git apply fix.diff",
    "git -C docs checkout plans/plan.md",
    # A wrapper given options of its own.
    "sudo -E tee docs/plans/plan.md",
    "sudo -u root tee -a docs/plans/plan.md",
    # In a program body, a path that reaches the write through a container or
    # a loop, or is built from pieces.
    "python3 - <<'PY'\nfrom pathlib import Path\nfor name in ['CLAUDE.md', 'AGENTS.md']:\n    Path(name).write_text('x')\nPY",
    "node -e \"const cfg = { out: 'docs/plans/plan.md' }; require('fs').writeFileSync(cfg.out, 'x');\"",
    "python3 -c \"from pathlib import Path; (Path('docs') / 'plans' / 'plan.md').write_text('x')\"",
)


@pytest.mark.parametrize("command", _DOCUMENTED_MISSES)
def test_a_documented_miss_is_not_detected(command: str) -> None:
    assert _detect(command) == []


# ----------------------------------------------------------------------
# cd and the working directory
# ----------------------------------------------------------------------


@pytest.mark.parametrize(("command", "expected"), [
    ("cd docs/plans && cd .. && echo x >> plans/plan.md", [_PLAN]),
    ("cd docs && echo x >> ../CLAUDE.md", ["CLAUDE.md"]),
    # The workspace root itself, by its absolute path -- a form a model used.
    ('cd "{root}" && echo x >> docs/plans/plan.md', [_PLAN]),
    ("cd /tmp/ws-d2/docs && echo x >> plans/plan.md", [_PLAN]),
    # A subshell's cd ends with it; the second write is relative to the root.
    ("(cd docs/plans && echo x >> plan.md) && echo y >> plan.md", [_PLAN, "plan.md"]),
    ("x=$(cd docs && pwd) && echo x >> plan.md", ["plan.md"]),
    # A cd outside the workspace stops relative resolution.
    ("cd /elsewhere && echo x >> plan.md", []),
    ("cd ~ && echo x >> plan.md", []),
    ("cd ../.. && echo x >> plan.md", []),
])
def test_a_cd_inside_the_command_moves_where_relative_paths_resolve(
    command: str, expected: list[str]
) -> None:
    assert _detect(command) == expected


@pytest.mark.parametrize(("cwd", "expected"), [
    ("", ["plan.md"]),
    (".", ["plan.md"]),
    ("docs/plans", [_PLAN]),
    ("docs/plans/", [_PLAN]),
    ("./docs/plans", [_PLAN]),
])
def test_the_cwd_parameter_is_where_a_relative_path_starts(cwd: str, expected: list[str]) -> None:
    assert _detect("echo x >> plan.md", cwd) == expected


def test_an_absolute_path_resolves_only_under_the_root_or_its_private_alias() -> None:
    plan_under = "echo x >> {}/docs/plans/plan.md"
    tracked = _TRACKED.__contains__
    assert detect_tracked_writes(plan_under.format("/private/tmp/w"), tracked, root="/tmp/w") == [_PLAN]
    assert detect_tracked_writes(plan_under.format("/tmp/w"), tracked, root="/private/tmp/w/") == [_PLAN]
    assert detect_tracked_writes(plan_under.format("/tmp/other"), tracked, root="/tmp/w") == []
    assert detect_tracked_writes(plan_under.format("/tmp/w"), tracked) == []


# ----------------------------------------------------------------------
# eval, shells, interpreters
# ----------------------------------------------------------------------


@pytest.mark.parametrize(("command", "expected"), [
    # eval's arguments are a command; the prototype skipped eval as a wrapper.
    ('eval "echo x >> docs/plans/plan.md"', [_PLAN]),
    ("eval echo x '>>' docs/plans/plan.md", [_PLAN]),
    ("bash -c 'cd docs/plans && echo x >> plan.md'", [_PLAN]),
    ("bash -lc 'echo x >> docs/plans/plan.md'", [_PLAN]),
    ("sh <<'EOF'\necho x >> docs/plans/plan.md\nEOF", [_PLAN]),
    ("bash -s <<'EOF'\nsed -i 's/a/b/' docs/plans/plan.md\nEOF", [_PLAN]),
    # A shell body's cd stays inside it.
    ("bash -c 'cd docs/plans' && echo x >> plan.md", ["plan.md"]),
    # A shell running a script file shows nothing.
    ("bash fix.sh < docs/plans/plan.md", []),
])
def test_an_eval_or_shell_body_is_scanned_as_a_command(command: str, expected: list[str]) -> None:
    assert _detect(command) == expected


def test_nested_shell_bodies_are_scanned_three_levels_deep_and_no_deeper() -> None:
    command = "echo x >> docs/plans/plan.md"
    found = []
    for _ in range(5):
        command = "bash -c " + shlex.quote(command)
        found.append(_detect(command))
    assert found == [[_PLAN], [_PLAN], [_PLAN], [], []]


@pytest.mark.parametrize(("command", "expected"), [
    ("php -r \"file_put_contents('docs/plans/plan.md', 'x');\"", [_PLAN]),
    ("python3.12 -c \"open('docs/plans/plan.md', mode='w').write('x')\"", [_PLAN]),
    ("perl <<'EOF'\nopen(my $f, '>>', 'docs/plans/plan.md');\nEOF", [_PLAN]),
    # Read the path, write elsewhere: the path is only opened for reading.
    ("python3 -c \"data=open('docs/plans/plan.md').read(); open('/tmp/out.md','w').write(data)\"", []),
    ("python3 -c \"t=open('docs/plans/plan.md', 'r').read(); open('/tmp/o.md','a').write(t)\"", []),
    ("python3 -c \"from pathlib import Path; Path('/tmp/o.md').write_text(Path('docs/plans/plan.md').read_text())\"", []),
    ("ruby -e 'File.write(\"/tmp/o.md\", File.read(\"docs/plans/plan.md\"))'", []),
    # A program that writes nothing names no path.
    ("python3 -c \"print('docs/plans/plan.md')\"", []),
    # Read the path, write elsewhere, in the read forms a write-anything rule
    # took as writes: only a write call's target is written.
    ("python3 -c \"from pathlib import Path; p = Path('CLAUDE.md'); Path('/tmp/c.md').write_text(p.read_text())\"", []),
    ("python3 -c \"import json; json.dump({'t': open('plan.md', mode='r').read()}, open('/tmp/p.json', 'w'))\"", []),
    ("node -e 'const fs=require(\"fs\"); const s=fs.readFileSync(\"docs/specs/api.md\",{encoding:\"utf8\"}); fs.writeFileSync(\"/tmp/e.json\", s)'", []),
    ("node -e \"const fs=require('fs'); fs.promises.readFile('task.md','utf8').then(d => fs.promises.writeFile('/tmp/t.txt', d))\"", []),
    ("ruby -e 'File.write(\"/tmp/r.txt\", File.readlines(\"task.md\").join)'", []),
    ("perl -e 'open(my $in, \"<\", \"plan.md\") or die; open(my $o, \">\", \"/tmp/p.txt\") or die; print $o <$in>'", []),
    ("php -r 'file_put_contents(\"/tmp/a.txt\", file_get_contents(\"AGENTS.md\"));'", []),
    ("python3 - <<'PY'\nfiles = ['CLAUDE.md', 'AGENTS.md']\nwith open('/tmp/bundle.md', 'w') as out:\n    for name in files:\n        out.write(open(name).read())\nPY", []),
    # A path bound to a name, then written through the name.
    ("python3 -c \"from pathlib import Path; p = Path('CLAUDE.md'); p.write_text(p.read_text().upper())\"", ["CLAUDE.md"]),
    ("python3 - <<'PY'\npath = 'docs/plans/plan.md'\nwith open(path, 'a') as fh:\n    fh.write('x')\nPY", [_PLAN]),
    ("node -e \"const fs = require('fs'); const f = 'task.md'; fs.appendFileSync(f, 'x');\"", ["task.md"]),
    ("perl -e 'my $f = \"task.md\"; open my $out, \">\", $f or die; print $out \"x\"'", ["task.md"]),
    ("php -r '$t = \"AGENTS.md\"; file_put_contents($t, str_replace(\"a\", \"b\", file_get_contents($t)));'", ["AGENTS.md"]),
    ("python3 - <<'PY'\nmode = 'a'\ntarget = 'docs/plans/plan.md'\nwith open(target, mode) as fh:\n    fh.write('x')\nPY", [_PLAN]),
    # A name bound again: only the binding the write call sees counts.
    ("python3 - <<'PY'\npath = 'docs/specs/api.md'\nspec = open(path).read()\npath = '/tmp/out.txt'\nwith open(path, 'w') as f:\n    f.write(spec)\nPY", []),
    # Deletes, and a move of the path away.
    ("python3 -c \"import os; os.remove('task.md')\"", ["task.md"]),
    ("node -e \"require('fs').rmSync('task.md')\"", ["task.md"]),
    ("ruby -e 'File.rename(\"AGENTS.md\", \"/tmp/AGENTS.md.bak\")'", ["AGENTS.md"]),
    # A move, a rename and a copy's destination are writes; a copy's source is not.
    ("python3 -c \"import os; os.replace('/tmp/x.md', 'docs/plans/plan.md')\"", [_PLAN]),
    ("python3 -c \"import shutil; shutil.move('docs/plans/plan.md', '/tmp/x.md')\"", [_PLAN]),
    ("python3 -c \"import shutil; shutil.copyfile('/tmp/x.md', 'docs/plans/plan.md')\"", [_PLAN]),
    ("node -e \"require('fs').copyFileSync('docs/plans/plan.md', '/tmp/x.md')\"", []),
    # A redirection inside a shell string, and perl's two-argument open.
    ("python3 -c \"import os; os.system('echo x >> docs/plans/plan.md')\"", [_PLAN]),
    ("perl -e 'open(F, \">>docs/plans/plan.md\"); print F \"x\"'", [_PLAN]),
    # A path inside prose names no file, even beside a write call.
    ("python3 -c \"open('/tmp/log.md', 'a').write('touched docs/plans/plan.md')\"", []),
    # perl -M takes a module, not -i: no in-place edit.
    ("perl -MList::Util=sum -ne 'print' docs/plans/plan.md", []),
    # The heredoc is data for a script file, not the program.
    ("python3 fix.py <<'EOF'\nopen('docs/plans/plan.md','w').write('x')\nEOF", []),
])
def test_an_interpreter_program_is_scanned_for_written_paths(command: str, expected: list[str]) -> None:
    assert _detect(command) == expected


# ----------------------------------------------------------------------
# Tools: the forms the corpora do not spell
# ----------------------------------------------------------------------


@pytest.mark.parametrize(("command", "expected"), [
    ("sed --in-place=.orig 's/a/b/' docs/plans/plan.md", [_PLAN]),
    ("sed -ni 's/a/b/p' docs/plans/plan.md", [_PLAN]),
    ("sed -n 's/a/b/p' docs/plans/plan.md", []),
    ("sed -e 's/i/j/' docs/plans/plan.md", []),
    ("cp -t docs/plans /tmp/plan.md", [_PLAN]),
    ("cp --target-directory=docs/plans /tmp/plan.md", [_PLAN]),
    ("mv -t docs/plans /tmp/plan.md", [_PLAN]),
    ("install -m 644 /tmp/plan.md docs/plans/plan.md", [_PLAN]),
    # rsync's -t preserves times; it names no target directory.
    ("rsync -t /tmp/a docs/plans/plan.md", [_PLAN]),
    ("patch -p1 docs/plans/plan.md < fix.diff", [_PLAN]),
    ("patch -o /tmp/out docs/plans/plan.md < fix.diff", []),
    ("patch -d docs/plans plan.md < fix.diff", []),
    ("sort --output=docs/plans/plan.md /tmp/x", [_PLAN]),
    ("sort -odocs/plans/plan.md /tmp/x", [_PLAN]),
    ("shred -n 3 docs/plans/plan.md", [_PLAN]),
    ("unlink docs/plans/plan.md", [_PLAN]),
    ("git restore --staged --worktree docs/plans/plan.md", [_PLAN]),
    ("git checkout -b plan.md", []),
    ("ex docs/plans/plan.md <<'EOF'\n$a\n- x\n.\nwq\nEOF", [_PLAN]),
    ("ex -s -c '%s/a/b/|write' docs/plans/plan.md", [_PLAN]),
    # An ed or ex whose script the command does not show is taken as writing.
    ("printf 'a\\nx\\n.\\nw\\n' | ed -s docs/plans/plan.md", [_PLAN]),
    # ed text lines are not commands: the appended "w" line is text.
    ("ed -s docs/plans/plan.md <<< $'a\\nw\\n.\\nq'", []),
    ("if sed -i 's/a/b/' docs/plans/plan.md; then echo ok; fi", [_PLAN]),
    ("while read l; do echo $l; done < /tmp/in >> docs/plans/plan.md", [_PLAN]),
    ("! tee docs/plans/plan.md < /dev/null", [_PLAN]),
])
def test_a_tool_writes_what_its_options_say(command: str, expected: list[str]) -> None:
    assert _detect(command) == expected


# ----------------------------------------------------------------------
# Lexing
# ----------------------------------------------------------------------


@pytest.mark.parametrize(("command", "expected"), [
    # Escaped or quoted operators are words, not redirections.
    ("echo x \\>\\> docs/plans/plan.md", []),
    ("echo 'x >> docs/plans/plan.md'", []),
    ('echo "x >> docs/plans/plan.md"', []),
    # A line continuation joins the redirection to its command.
    ("echo x \\\n  >> docs/plans/plan.md", [_PLAN]),
    # >& with a file writes it; with an fd or - it copies or closes one.
    ("echo x >& docs/plans/plan.md", [_PLAN]),
    ("echo x >&2", []),
    ("echo x 2>&- >&1", []),
    ("echo x <> docs/plans/plan.md", [_PLAN]),
    ("echo x &>> docs/plans/plan.md", [_PLAN]),
    # An unterminated quote swallows the rest: one word, no redirection.
    ("echo 'x >> docs/plans/plan.md", []),
    # A comment is not a command.
    ("echo x # >> docs/plans/plan.md", []),
    # A heredoc body is not lexed as commands.
    ("cat <<'EOF' > /tmp/x\necho y >> docs/plans/plan.md\nEOF", []),
    ("cat <<-EOF > /tmp/x\n\techo y >> docs/plans/plan.md\n\tEOF\necho z >> CLAUDE.md", ["CLAUDE.md"]),
    ('echo "a \\" >> docs/plans/plan.md"', []),
])
def test_quoting_escapes_and_heredocs_are_honoured(command: str, expected: list[str]) -> None:
    assert _detect(command) == expected


def test_paths_are_deduplicated_in_first_occurrence_order() -> None:
    command = "echo a >> CLAUDE.md; tee docs/plans/plan.md CLAUDE.md < /dev/null; rm docs/plans/plan.md"
    assert _detect(command) == ["CLAUDE.md", _PLAN]


def test_only_tracked_paths_are_returned() -> None:
    assert _detect("echo x >> README.md && echo y >> docs/plans/plan.md") == [_PLAN]
    assert detect_tracked_writes("echo x >> README.md", lambda path: True) == ["README.md"]


def test_empty_and_pathological_commands_return_promptly() -> None:
    started = time.monotonic()
    for command in ("", "   ", "a" * 16000, "(" * 8000, ")" * 8000, "'" * 8000, "<<" * 4000, "\\" * 8000):
        assert _detect(command) == []
    assert time.monotonic() - started < 2.0


@pytest.mark.parametrize("command", [
    "ex <<< '" + " " * 16000 + "'",
    "ex -c '" + " " * 16000 + "' f.md",
    "ed -s f.md <<< '" + " " * 16000 + "'",
    'python3 -c "' + "open(" * 3200 + '"',
    'python3 -c "' + "open(a, " * 2000 + '"',
    'python3 -c "' + "open(," * 2600 + '"',
    'python3 -c "' + "x = 'a.md'; " * 1300 + "open('b.md', 'w')" + '"',
    ('python3 -c "' + "".join(f"x{i} = 'a.md'; " for i in range(1000))
     + "".join(f"open(x{i}, 'w'); " for i in range(60)) + '"')[:16384],
], ids=["ex-herestring-spaces", "ex-c-spaces", "ed-herestring-spaces", "open-calls",
        "open-calls-with-args", "open-calls-with-commas", "many-path-literals",
        "many-bound-names"])
def test_a_16k_script_body_is_scanned_in_linear_time(command: str) -> None:
    """Script and program checks stay linear on a 16K body: an ex prefix whose
    two whitespace runs could split one run many ways, an unbounded scan for
    an ``open(`` mode, or following every one of a thousand bound names to the
    end of the body, made one of these take seconds."""
    started = time.monotonic()
    _detect(command)
    assert time.monotonic() - started < 0.25


def test_a_literal_that_only_starts_a_built_path_is_not_the_path_written() -> None:
    """``open('plan.md' + suffix, 'w')`` writes a file whose name merely starts
    with the literal, so the literal is not the open's target and is not
    reported: a path assembled from pieces is a documented miss, however close
    the mode sits. (The ``open(`` mode scan of the write-call filter stays
    bounded; the linear-time cases cover it.)"""
    for pad in (0, 150, 250):
        command = "python3 -c \"open('docs/plans/plan.md' + '" + "x" * pad + "', 'w')\""
        assert _detect(command) == [], pad
