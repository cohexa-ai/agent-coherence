# Contributing to agent-coherence

Thank you for your interest in contributing. `agent-coherence` is a coordinator
that stops one agent from silently overwriting another's work on shared state,
with its safety invariants checked in TLA+. This guide covers how to propose a
change, set up a checkout, run the checks CI runs, and get a pull request
merged.

## Before you start

- **Bugs and feature requests:** open an
  [issue](https://github.com/Cohexa-ai/agent-coherence/issues/new/choose). A
  bug report that includes a reproduction gets fixed fastest.
- **Non-trivial changes:** open an issue before you start a new feature, a
  public API change, a wire-protocol change or a large refactor, so the design
  can be agreed before you write the code.
- **Questions and ideas:** use
  [Discussions](https://github.com/Cohexa-ai/agent-coherence/discussions).
- **Security issues:** never open a public issue. Report them privately as
  [`.github/SECURITY.md`](.github/SECURITY.md) describes.

## Contribution guidelines

We prefer small, well-tested changes that solve a real problem.

- Keep each pull request to one bug, feature or documentation change. Split
  unrelated changes into separate PRs.
- Link the issue the PR fixes (`Fixes #123`) so it closes when the PR merges.
- A bug fix comes with a regression test that fails without the fix. New
  behaviour comes with tests.
- When behaviour changes, update the docs that describe it: `README.md`,
  [`docs/guide.md`](docs/guide.md) (the user guide), or
  [`docs/ccs-diagnose.md`](docs/ccs-diagnose.md) for `ccs-diagnose`.
- For a change users will notice, add an entry under `## [Unreleased]` in
  [`CHANGELOG.md`](CHANGELOG.md), under `Added`, `Changed` or `Fixed`. Lead
  with a bold sentence that says what changed and names the issue, then say
  what happened before and what happens now. If you'd rather leave it out, say
  so in the PR and a maintainer will add it.
- AI-assisted contributions are welcome, but a person must review, edit and
  test them before submitting. Bulk, low-signal generated PRs may be closed.

## Repository layout

| Path | Contents |
|---|---|
| `src/ccs/` | The library: `core` (types and errors), `coordinator`, `adapters` (framework integrations and the Claude Code coordinator server), `cli`, `mcp`, `diagnose` and the rest |
| `tests/` | The pytest suite. `tests/protocol_corpus/` holds the shared fixtures that keep the Python and Node coordinators in step |
| `docs/` | The user guide, the `ccs-diagnose` guide, the security model and the approach write-ups |
| `examples/` | Runnable examples, one directory per scenario |
| `benchmarks/` | Benchmark scenarios, the expected results the drift checks compare against, and published results |
| `formal/tla/` | The TLA+ specifications and their model-checking configs, with `tla2tools.jar` |
| `tools/` | Repository tooling: the local test runner, the architecture check, benchmark and release scripts |
| `.github/` | CI workflows, Dependabot config, issue templates and the security policy |

The Claude Code plugin, including the Node coordinator the protocol corpus
tests against, lives in a separate repository,
[`Cohexa-ai/agent-coherence-plugin`](https://github.com/Cohexa-ai/agent-coherence-plugin).

## Development setup

You need Python 3.11 or newer; CI tests on 3.11 and 3.12. Fork the
repository, clone your fork, and install the package into a virtual
environment at the root of the checkout:

```bash
git clone https://github.com/<your-username>/agent-coherence.git
cd agent-coherence
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,diagnose,mcp]'
```

These are the extras CI installs for the test jobs. Other extras, such as
`langgraph`, `crewai`, `openai-agents` or `coherent-row`, are only needed for
the integration you are working on, and `all` installs every one.

The refactor demo's TypeScript tests need Node 20 and the fixture's
dependencies. Without them those tests skip rather than fail:

```bash
(cd examples/refactor_demo/fixture_repo_ts && npm ci)
```

If you change dependencies in `pyproject.toml`, refresh `uv.lock` with
`uv lock` and commit it too. Dependabot keeps it current otherwise.

## Running tests locally

Tests run under the main checkout's virtual environment, with the same extras CI installs, as set up above. Run the whole suite from the main checkout or any git worktree of it:

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

### Marking tests

A test that starts real processes that race each other, and so is not safe
beside other parallel workers, gets `@pytest.mark.serial`. These markers
select opt-in suites that a default run leaves out:

| Marker | Needs |
|---|---|
| `protocol_corpus` | A built plugin checkout (see below) |
| `live_api` | `OPENAI_API_KEY` and/or `MISTRAL_API_KEY`; calls paid APIs |
| `real_substrate` | A real Postgres and S3, with credentials |
| `launch_gate`, `launch_gate_pilot`, `launch_gate_strict`, `launch_gate_pilot_matrix` | A live `claude` CLI; calls paid APIs |

`pyproject.toml` describes each marker in full.

## The checks CI runs

Every pull request to `dev` runs these checks, and all of them must pass
before it can merge:

| Check | What it runs | Run it locally |
|---|---|---|
| Lint | `ruff check` | `.venv/bin/ruff check src/ tests/` |
| Architecture Checks | The layer and import-cycle rules below | `.venv/bin/python tools/check_architecture.py` |
| Tests (Python 3.11), Tests (Python 3.12) | The suite, in two shards per version | `python3 tools/run_local_tests.py` |
| Benchmark Smoke | A quick benchmark regression check | `.venv/bin/python tools/benchmark_regression_smoke.py` |
| Benchmark Regression | The LangGraph benchmarks, compared with `benchmarks/expected.json` | `make benchmark benchmark-check`, with the venv activated and `pip install -e '.[langgraph,benchmark]'` |
| TLA+ Model Check | TLC over the specifications in `formal/tla/` | `make tla-check` (needs Java 17 or newer) |
| Protocol Corpus (Python ↔ Node parity) | The shared fixtures against both coordinators | See below |
| Build Package | Builds the wheel and sdist and checks them | `.venv/bin/python -m build && .venv/bin/twine check dist/*`, after `pip install build twine` |

A pull request whose changes touch no TLA+ input skips the TLC run. The model
check takes several minutes per bin, so run it when you change
`formal/tla/` or the behaviour a specification models.

### Protocol corpus

The corpus runs the same JSON fixtures against this repository's Python
coordinator and the plugin's Node coordinator, so a change to what the
coordinator sends or accepts must keep both in step. To run it, build the
plugin next to this checkout (it needs Node 22):

```bash
git clone https://github.com/Cohexa-ai/agent-coherence-plugin.git ../agent-coherence-plugin
(cd ../agent-coherence-plugin && npm ci && npm run build)
.venv/bin/pytest -m protocol_corpus
```

The harness finds `../agent-coherence-plugin/dist/coordinator.js` by itself;
set `AGENT_COHERENCE_PLUGIN_DIST_PATH` to use a build somewhere else. Without
a build, the Node scenarios report an expected failure that names the reason.
In CI, a PR is tested against the plugin branch with the same name as its own
branch when one exists, and against the plugin's `dev` otherwise, so a change
that needs both repositories can be tested together before either merges.

## Code style

- `ruff check` must pass. The configuration is in `pyproject.toml`: rules
  `E`, `F` and `I` (import order), 120-character lines.
- The repository does not use an auto-formatter. Don't reformat code your
  change doesn't touch, so the diff shows only your change.
- New Python files under `src/` and `tests/` start with the project header:

  ```python
  # Copyright (c) 2026 agent-coherence contributors.
  # The Coherence Protocol for AI Agents
  ```

- Use type hints, with `from __future__ import annotations` at the top of the
  module.
- `tools/check_architecture.py` enforces these import layers in `src/ccs/`,
  and refuses import cycles. A module may import only from its own layer and
  the layers above it in this table:

  | Layer | Packages |
  |---|---|
  | core | `core`, `artifacts` |
  | application | `coordinator`, `strategies`, `agent` |
  | infrastructure | `bus`, `transport`, `simulation` |
  | interface | `output`, `cli`, `adapters`, `validation`, `diagnose`, `replay`, `mcp`, `testing` |

  A new top-level package under `src/ccs/` must be added to the map in
  `src/ccs/hardening/architecture.py`.

## Pull requests

- Open the PR against `dev`, the default branch. `main` carries releases only.
- Write the title as a [Conventional Commit](https://www.conventionalcommits.org/):
  `<type>(<scope>): <description>`, lower case, in the imperative, with no full
  stop. The types are `feat`, `fix`, `docs`, `refactor`, `test` and `chore`;
  the scope is the area changed, such as `coordinator`, `volume`, `diagnose`
  or `corpus`. For example: `fix(coordinator): keep a read on a holder's own
  release`. PRs are squash-merged, and the title becomes the commit subject on
  `dev`.
- Tick **Allow edits by maintainers** when you open the PR. `dev` only accepts
  a branch that is up to date with it, and other PRs land there often. With
  that box ticked, a maintainer can update your branch right before merging;
  without it, you'd be asked to merge `dev` again every time it moves.
- On your first PR, a maintainer has to approve each CI run before it starts,
  so the checks can look empty for a while. That is GitHub's default for
  first-time contributors, and stops once you have a merged PR.

Before you ask for review, check that:

- the PR covers one change and links its issue;
- tests were added or updated, and `python3 tools/run_local_tests.py` passes;
- `ruff check src/ tests/` and `tools/check_architecture.py` pass;
- the docs and `CHANGELOG.md` describe any change users will notice;
- the description says what you ran, and anything you could not run.

## Releases

Maintainers cut releases. A release goes from `dev` to `main`, and pushing a
`v*` tag on `main` runs `.github/workflows/release.yml`, which builds the
package, attaches an SBOM and publishes it to PyPI with attestations. The
version lives in `src/ccs/__init__.py`. Contributors don't need to change it.

## License

`agent-coherence` is licensed under the [Apache License 2.0](LICENSE). By
contributing, you agree that your contributions are licensed under it too.

## Getting help

Ask in [Discussions](https://github.com/Cohexa-ai/agent-coherence/discussions),
or open an issue if something in this guide is wrong or missing.
