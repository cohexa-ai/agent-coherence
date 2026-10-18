# Contributing

## Running tests locally

Tests run under the main checkout's virtual environment, with the same extras CI installs. Set it up once:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,diagnose,mcp]'
```

Then run the whole suite, from the main checkout or any git worktree of it:

```bash
python3 tools/run_local_tests.py
```

This runs the suite the way each CI Python job does: most tests in parallel on every core, then the tests marked `serial`, which start real processes that race each other, on their own.

To run only some tests, name their files, directories or test ids. pytest options such as `-k`, `-x`, `-v` and `--lf` pass through:

```bash
python3 tools/run_local_tests.py tests/test_engine.py -x -k name
```

The command also:

- uses the main checkout's `.venv` even from a worktree that has none, and stops with the install command when that venv lacks anything CI installs;
- runs the code of the tree you started it in, including in the command-line tools the tests start;
- keeps Python's bytecode caches on, so repeated runs start quickly.

### Full runs take turns

Only one full run uses the machine at a time, across every worktree of the clone. Another full run waits, and says which worktree holds the turn and since when. A run that is stopped or killed gives the turn back when its tests exit. A run that names tests starts straight away; only its `serial` tests wait for the turn.

### When to use plain pytest

The command refuses what would make a run select different tests from CI, and points you to plain `pytest` instead. That covers:

- marker selections such as `pytest -m protocol_corpus`; the opt-in suites need extra setup, and some call paid APIs;
- custom configuration or a different root directory;
- paths outside `tests/`.
