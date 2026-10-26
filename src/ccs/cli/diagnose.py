# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""CLI entrypoint for ``ccs-diagnose`` (Unit 7 — pipeline integration).

Wires Units 1-6 into a single console script. The main path imports a
user-supplied LangGraph factory (``--graph PATH:FUNCTION``), invokes the
graph under :class:`ccs.diagnose.callback.DiagnoseCallback`, runs the
classifier (Unit 3), divergence detector (Unit 4), ownership map
(Unit 5 helper), HTML renderer (Unit 5), terminal summary (Unit 6),
and optionally a JSON report.

Two subcommand-style flags bypass the main pipeline:

* ``--show-payload PATH`` reads a previously-emitted ``report.json``,
  invokes :func:`ccs.diagnose.telemetry.payload_for_from_json`, prints
  the payload, and exits 0. Incompatible with ``--graph`` (warning to
  stderr).
* ``--reset-token`` is a Unit 8 stub — prints a not-yet-implemented
  message and exits 0. Reserved for the consent-flow update.

Trust-posture flags ``--no-network`` and ``--no-telemetry`` short-circuit
the consent resolver to a denied state. ``--calibration-record`` (Unit 9)
appends the run's payload to a local JSONL file when consent is granted;
denied / kill-switched runs print a skip message and exit 0 without
writing. ``--dry-run`` prints the telemetry payload that would be
submitted (no network is involved because v0 has no submission code).

Architecture: lives in the ``interface`` layer alongside
``ccs.diagnose``. Imports flow interface → interface only.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import inspect
import json
import math
import os
import signal
import sys
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext, redirect_stdout
from dataclasses import replace
from pathlib import Path
from typing import Any

from ccs.diagnose.calibration import (
    append_calibration_entry,
    calibration_path,
)
from ccs.diagnose.callback import DiagnoseCallback
from ccs.diagnose.classifier import (
    Bucket,
    ClassifierOverrides,
    ClassifierVerdict,
    build_key_index,
    classify,
)
from ccs.diagnose.detection import (
    DetectionReport,
    ReadObservation,
    build_report_json,
    detect,
)
from ccs.diagnose.ownership import compute_ownership_map
from ccs.diagnose.render import (
    DEFAULT_BOOK_A_CALL_URL,
    DEFAULT_CONTACT_EMAIL,
    RenderOptions,
    render_html,
)
from ccs.diagnose.summary import terminal_summary
from ccs.diagnose.telemetry import (
    CURRENT_POLICY_VERSION,
    ConsentState,
    env_kill_switch_active,
    load_consent,
    payload_for,
    payload_for_from_json,
    reset_token,
    resolve_consent,
)

__all__ = ["main", "console_main", "build_parser"]


_DEFAULT_HTML_PATH = "diagnose_report.html"
_DEFAULT_JSON_PATH = "diagnose_report.json"

# -------------------------------------------------------------------- #
# Exit codes
# -------------------------------------------------------------------- #
# Distinct codes per failure category make the CLI scriptable for agents
# and CI pipelines that need to branch on *why* a run failed without
# parsing stderr. ``EXIT_USAGE_ERROR`` is the argparse default; the rest
# extend it with diagnose-specific categories.

EXIT_OK: int = 0
EXIT_GENERIC_ERROR: int = 1
EXIT_USAGE_ERROR: int = 2  # argparse default
EXIT_DEPENDENCY_MISSING: int = 3  # langgraph (or other extras) not installed
EXIT_SCHEMA_MISMATCH: int = 4  # --show-payload version not in allowlist
EXIT_IO_ERROR: int = 5  # filesystem write/read failure


SUPPORTED_SCHEMA_VERSIONS: frozenset[str] = frozenset(
    {"ccs.diagnose.v0-preview", "ccs.diagnose.v1"}
)
"""Schema versions accepted by ``--show-payload``.

The CLI emits ``v0-preview``; future units will promote to ``v1`` after
the calibration gate. ``--show-payload`` accepts both so an installation
with a tip-of-tree CLI can still inspect old report.json files (and so
v1-tagged reports load on a v0-preview CLI during the migration window).
"""

_MAX_INPUT_FILE_SIZE: int = 10 * 1024 * 1024
"""Upper bound for ``--show-payload`` and ``--state-file`` reads.

Both flags ultimately ``json.loads`` the file. 10 MB is comfortably above the
realistic shape of either artefact (verdict + report or LangGraph initial
state) while bounding worst-case memory if a user points the CLI at a
multi-gigabyte file by accident.
"""
_DEFAULT_TOKEN_COST_PER_1K: float = 0.003
"""Placeholder cost per 1K tokens. Mirrors :func:`ccs.diagnose.detection.detect`.

Calibration in later units may parameterise this from the corpus; for now
the renderer footer surfaces the assumption so users can recalibrate.
"""


# -------------------------------------------------------------------- #
# Argparse
# -------------------------------------------------------------------- #


def _finite_non_negative_float(value: str) -> float:
    """argparse ``type=`` callable that rejects ``inf`` / ``nan`` / negatives.

    Without this guard, downstream cost-extrapolation code multiplies the
    flag by other quantities and surfaces ``OverflowError`` instead of a
    clean argparse error.
    """
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected a finite, non-negative number; got {value!r}"
        ) from exc
    if not math.isfinite(result):
        raise argparse.ArgumentTypeError(
            f"expected a finite, non-negative number; got {value!r}"
        )
    if result < 0:
        raise argparse.ArgumentTypeError(
            f"expected a non-negative number; got {value!r}"
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ccs-diagnose",
        description=(
            "Diagnose a LangGraph run for divergent reads (witness-quality).\n"
            "\n"
            "ccs-diagnose attaches a passive callback to a LangGraph graph,\n"
            "classifies its write pattern (single_writer / shared_artifact /\n"
            "parallel_branch / mixed), and reports artifacts whose reads were\n"
            "handed divergent versions across nodes.\n"
            "\n"
            "v0-preview is a witness-quality surface: it observes what the\n"
            "runtime *handed* a node, not what the node read. The CCSStore\n"
            "upgrade lifts these observations into provable per-key\n"
            "attribution — same diagnose surface, no callback rewiring."
        ),
        epilog=(
            "Install: pip install \"agent-coherence[diagnose]\"\n"
            "\n"
            "Examples:\n"
            "  ccs-diagnose --graph examples/langgraph_planner/main.py:build_graph_no_store\n"
            "  ccs-diagnose --graph my/graph.py:factory --volume 50 --strict\n"
            "  ccs-diagnose --show-payload diagnose_report.json\n"
            "\n"
            "Exit codes:\n"
            "  0  success\n"
            "  1  generic error (graph load failure, JSON parse, etc.)\n"
            "  2  usage error (argparse, mutually exclusive flags, bad URL/email)\n"
            "  3  dependency missing (import error from graph or extras)\n"
            "  4  schema mismatch (--show-payload version not supported)\n"
            "  5  I/O error (write/read failed; oversized input file)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Main pipeline flags.
    parser.add_argument(
        "--graph",
        default=None,
        metavar="PATH:FUNCTION",
        help=(
            "Path to a Python file and factory function name, separated by ':'.\n"
            "Required for the main pipeline; not required when --show-payload\n"
            "or --reset-token is set. Example: path/to/graph.py:build_graph"
        ),
    )
    parser.add_argument(
        "--state-file",
        default=None,
        help=(
            "Path to a JSON or YAML file with the initial state passed to\n"
            "graph.invoke(). When omitted, the factory is invoked with no\n"
            "arguments and the graph receives an empty dict."
        ),
    )
    parser.add_argument(
        "--output-html",
        default=_DEFAULT_HTML_PATH,
        help=f"Output path for the HTML report. Default: {_DEFAULT_HTML_PATH}",
    )
    parser.add_argument(
        "--output-json",
        default=_DEFAULT_JSON_PATH,
        help=(
            "Output path for the machine-readable JSON report.\n"
            f"Default: {_DEFAULT_JSON_PATH} (always emitted unless --no-json).\n"
            "Pass '-' to write the JSON to stdout instead; the human-\n"
            "readable summary is then routed to stderr so callers can pipe\n"
            "stdout straight into ``jq`` etc."
        ),
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Suppress JSON report output.",
    )
    parser.add_argument(
        "--volume",
        type=_finite_non_negative_float,
        default=None,
        metavar="N",
        help=(
            "Interactions per hour for cost extrapolation. When supplied,\n"
            "the renderer surfaces an annualized rework-cost floor.\n"
            "Must be a finite, non-negative number."
        ),
    )
    parser.add_argument(
        "--lead-pain-type",
        choices=["cost", "auditability", "auto"],
        default="auto",
        help=(
            "Which secondary KPI rides the headline. 'auto' routes to 'cost'\n"
            "when --volume is supplied, otherwise 'auditability'."
        ),
    )
    parser.add_argument(
        "--cost-per-1k-tokens",
        type=float,
        default=_DEFAULT_TOKEN_COST_PER_1K,
        metavar="FLOAT",
        help=(
            "Token cost-per-1k assumption used for cost extrapolation.\n"
            f"Default: {_DEFAULT_TOKEN_COST_PER_1K} (placeholder pending calibration)."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Promote sequential-staleness exclusions back into the headline\n"
            "divergence count. Cold-start and append-only exclusions are not\n"
            "affected."
        ),
    )
    parser.add_argument(
        "--ignore",
        dest="ignore",
        default="",
        metavar="KEY1,KEY2",
        help="Comma-separated state keys to ignore. May include '__'-prefixed keys.",
    )
    parser.add_argument(
        "--track",
        dest="track",
        default="",
        metavar="KEY1,KEY2",
        help=(
            "Comma-separated state keys to force-track. Wins over --ignore for\n"
            "any key listed in both."
        ),
    )
    parser.add_argument(
        "--warm-lead",
        action="store_true",
        help=(
            "Switch the CTA to a warm-conversation 2-question seed (no 30-min\n"
            "walk-through ask)."
        ),
    )
    parser.add_argument(
        "--book-a-call-url",
        default=DEFAULT_BOOK_A_CALL_URL,
        metavar="URL",
        help=f"Calendar URL rendered in the CTA. Default: {DEFAULT_BOOK_A_CALL_URL}",
    )
    parser.add_argument(
        "--contact-email",
        default=DEFAULT_CONTACT_EMAIL,
        metavar="EMAIL",
        help=f"Reply-to address rendered in the CTA. Default: {DEFAULT_CONTACT_EMAIL}",
    )

    # Trust-posture flags (Unit 8 will add behaviour to most of these).
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run the pipeline and print the telemetry payload that would be\n"
            "submitted; no network code exists in v0 so this is purely a\n"
            "preview."
        ),
    )
    parser.add_argument(
        "--no-network",
        action="store_true",
        help=(
            "v0 alias for --no-telemetry (no network code exists in v0;\n"
            "flag reserved for v1 endpoint POST suppression). Suppresses\n"
            "the consent prompt and forces a denied state."
        ),
    )
    parser.add_argument(
        "--no-telemetry",
        action="store_true",
        help=(
            "Skip the consent prompt and calibration write entirely;\n"
            "suppress all telemetry-shaped output. Functionally identical\n"
            "to --no-network in v0."
        ),
    )
    parser.add_argument(
        "--yes",
        "--non-interactive",
        dest="yes",
        action="store_true",
        help=(
            "Skip the interactive consent prompt; use persisted state or\n"
            "treat as denied. Required for non-TTY agent invocations\n"
            "where the consent prompt would block forever."
        ),
    )
    parser.add_argument(
        "--calibration-record",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help=(
            "Append the run's payload to a local JSONL calibration corpus.\n"
            "When passed without a path, the default location is used:\n"
            "$XDG_DATA_HOME/ccs-diagnose/calibration.jsonl (or\n"
            "~/.local/share/ccs-diagnose/calibration.jsonl when XDG_DATA_HOME\n"
            "is unset). Pass an explicit path to override. Gated by consent —\n"
            "denied / kill-switched runs print a skip message and exit 0."
        ),
    )

    # Subcommand-style flags.
    parser.add_argument(
        "--show-payload",
        default=None,
        metavar="PATH_TO_REPORT_JSON",
        help=(
            "Bypass the pipeline; load PATH (a previously-emitted report.json)\n"
            "and print the telemetry payload. Incompatible with --graph."
        ),
    )
    parser.add_argument(
        "--reset-token",
        action="store_true",
        help=(
            "Regenerate the local installation token in consent.json and "
            "print the new UUID."
        ),
    )

    return parser


def console_main(argv: Sequence[str] | None = None) -> int:
    """Console-only entry point; embedded callers should use :func:`main`."""
    # Restore the default SIGPIPE behaviour so piping into ``head`` etc. exits
    # cleanly with status 141 instead of raising ``BrokenPipeError`` from the
    # interpreter. Windows lacks SIGPIPE; the ``hasattr`` guard keeps the
    # entrypoint portable.
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    return main(argv, _graph_context=_graph_sigpipe_guard)


class _ConsoleStdout:
    """Keep a closed console pipe quiet, including writes from graph workers."""

    def __init__(self, stream: Any) -> None:
        self._stream = stream

    def write(self, data: str) -> int:
        try:
            return self._stream.write(data)
        except BrokenPipeError:
            # signal.signal() cannot run in a worker thread. Exit immediately
            # with the conventional shell status instead: graph exception
            # handlers must not turn a closed stdout into a failure report.
            os._exit(128 + signal.SIGPIPE)

    def flush(self) -> None:
        try:
            self._stream.flush()
        except BrokenPipeError:
            os._exit(128 + signal.SIGPIPE)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


@contextmanager
def _graph_sigpipe_guard() -> Iterator[None]:
    """Protect graph sockets while preserving quiet console stdout exits.

    This process-wide window ends when graph load/build/invoke returns.
    Embedded main() never enters it. Raw file descriptors, stdout.buffer,
    captured stream aliases and graph work running after the window are not
    covered by the stdout proxy.
    """
    if not hasattr(signal, "SIGPIPE"):
        yield
        return
    previous = signal.signal(signal.SIGPIPE, signal.SIG_IGN)
    try:
        with redirect_stdout(_ConsoleStdout(sys.stdout)):
            yield
    finally:
        signal.signal(signal.SIGPIPE, previous)


def main(
    argv: Sequence[str] | None = None,
    *,
    _graph_context: Callable[[], AbstractContextManager[None]] = nullcontext,
) -> int:
    """Run diagnostics without changing SIGPIPE. Returns the exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.ignore_keys = _parse_csv(args.ignore)
    args.track_keys = _parse_csv(args.track)

    if args.show_payload and args.reset_token:
        print(
            "error: --show-payload and --reset-token are mutually exclusive",
            file=sys.stderr,
        )
        return EXIT_USAGE_ERROR

    if args.show_payload:
        return _run_show_payload(args)

    if args.reset_token:
        return _run_reset_token(args)

    if not args.graph:
        # Match argparse's own format so the exit-code path is uniform
        # across all error branches (each returns an int).
        print(
            f"{parser.prog}: error: --graph is required "
            "(or use --show-payload PATH / --reset-token)",
            file=sys.stderr,
        )
        return EXIT_USAGE_ERROR

    return _run_pipeline(args, graph_context=_graph_context)


# -------------------------------------------------------------------- #
# Subcommand handlers
# -------------------------------------------------------------------- #


def _run_show_payload(args: argparse.Namespace) -> int:
    if args.graph:
        print(
            "warning: --graph ignored when --show-payload is set",
            file=sys.stderr,
        )

    payload_path = Path(args.show_payload)
    try:
        if payload_path.stat().st_size > _MAX_INPUT_FILE_SIZE:
            print(
                f"error: --show-payload target {payload_path} exceeds "
                f"{_MAX_INPUT_FILE_SIZE} bytes",
                file=sys.stderr,
            )
            return EXIT_IO_ERROR
        loaded_text = payload_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        print(f"error: cannot read --show-payload target: {exc}", file=sys.stderr)
        return EXIT_IO_ERROR
    except OSError as exc:
        print(f"error: cannot read --show-payload target: {exc}", file=sys.stderr)
        return EXIT_IO_ERROR

    try:
        loaded = json.loads(loaded_text)
    except json.JSONDecodeError as exc:
        print(
            f"error: --show-payload target is not valid JSON: {exc}",
            file=sys.stderr,
        )
        return EXIT_GENERIC_ERROR

    if not isinstance(loaded, dict):
        print(
            "error: --show-payload target must be a JSON object (dict); "
            f"got {type(loaded).__name__}",
            file=sys.stderr,
        )
        return EXIT_GENERIC_ERROR

    schema_version = loaded.get("schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        supported_display = ", ".join(sorted(SUPPORTED_SCHEMA_VERSIONS))
        print(
            f"error: schema version mismatch in {payload_path}: "
            f"expected one of [{supported_display}], got {schema_version!r}",
            file=sys.stderr,
        )
        return EXIT_SCHEMA_MISMATCH

    # ``--show-payload`` is a read-only operation: never prompt. We use
    # the current persisted consent if any so the displayed
    # ``installation_token`` matches what would actually be submitted.
    consent = _consent_for_show_payload()
    payload = payload_for_from_json(loaded, consent=consent)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


def _consent_for_show_payload() -> ConsentState:
    """Resolve consent for the read-only ``--show-payload`` path.

    Never prompts. Honors env-var kill switches. Falls back to the
    persisted consent file when present and at the current policy
    version; otherwise returns a denied state.
    """
    if env_kill_switch_active() is not None:
        return ConsentState(
            granted=False,
            policy_version=CURRENT_POLICY_VERSION,
            installation_token=None,
        )
    existing = load_consent()
    if existing is None:
        return ConsentState(
            granted=False,
            policy_version=CURRENT_POLICY_VERSION,
            installation_token=None,
        )
    if existing.policy_version != CURRENT_POLICY_VERSION:
        return ConsentState(
            granted=False,
            policy_version=CURRENT_POLICY_VERSION,
            installation_token=None,
        )
    return existing


def _run_reset_token(args: argparse.Namespace) -> int:
    """Implement ``ccs-diagnose --reset-token``.

    Regenerates the local ``consent.json`` with a fresh UUID4 and
    ``granted=True``. When a kill-switch env var is active, prints a
    warning so users understand the regeneration won't enable
    submissions while the kill switch is set.
    """
    del args  # accepted for symmetry with subcommand handler signature
    active = env_kill_switch_active()
    if active is not None:
        print(
            f"warning: {active} is set; --reset-token regenerates the local "
            "consent.json but no submissions will occur regardless",
            file=sys.stderr,
        )
    new_token = reset_token()
    print(f"installation token regenerated: {new_token}")
    return 0


# -------------------------------------------------------------------- #
# Main pipeline
# -------------------------------------------------------------------- #


def _run_pipeline(
    args: argparse.Namespace,
    *,
    graph_context: Callable[[], AbstractContextManager[None]] = nullcontext,
) -> int:
    """Orchestrate the diagnose CLI pipeline.

    The work is broken into four named phases (each ≤ 30 lines):

    1. resolve consent
    2. load graph + invoke under callback
    3. classify, detect, build ownership map
    4. render outputs (HTML / JSON / summary / dry-run / calibration)
    """
    consent = _resolve_consent(args)

    with graph_context():
        invoke = _load_graph_and_invoke(args)
    if isinstance(invoke, int):
        return invoke
    callback, initial_state, result = invoke

    classify_result = _run_classify_and_detect(
        events=callback.events,
        args=args,
        initial_state=initial_state,
        result=result,
    )
    verdict, report, key_index = classify_result
    ownership = _compute_ownership(callback.events, verdict, key_index)

    return _write_outputs(
        verdict=verdict,
        report=report,
        ownership=ownership,
        args=args,
        consent=consent,
    )


def _resolve_consent(args: argparse.Namespace) -> ConsentState:
    """Resolve consent for the pipeline run.

    ``--no-telemetry`` and ``--no-network`` short-circuit to a denied
    state. ``--yes`` / ``--non-interactive`` skips the prompt and uses
    the persisted state at the current policy version, falling back to
    denied. The default path runs the interactive resolver.
    """
    if args.no_telemetry or args.no_network:
        return ConsentState(
            granted=False,
            policy_version=CURRENT_POLICY_VERSION,
            installation_token=None,
        )
    if args.yes:
        existing = load_consent()
        if existing is not None and existing.policy_version == CURRENT_POLICY_VERSION:
            return existing
        return ConsentState(
            granted=False,
            policy_version=CURRENT_POLICY_VERSION,
            installation_token=None,
        )
    return resolve_consent()


def _load_graph_and_invoke(
    args: argparse.Namespace,
) -> tuple[DiagnoseCallback, dict[str, Any] | None, Any] | int:
    """Load the user graph factory, build it, and invoke under a callback.

    Returns ``(callback, initial_state, result)`` on success, or an
    integer exit code when graph loading / state-file parsing fails.
    Graph-invoke exceptions surface as a warning event in the callback
    buffer (the verdict downgrades) — never as a non-zero return.
    """
    factory_loaded = _load_graph_factory(args.graph)
    if isinstance(factory_loaded, int):
        return factory_loaded
    factory, factory_module = factory_loaded

    initial_state = _load_state_file(args.state_file)
    if isinstance(initial_state, _LoadError):
        print(f"error: {initial_state.message}", file=sys.stderr)
        return EXIT_GENERIC_ERROR

    if initial_state is None and factory_module is not None:
        companion = _discover_initial_state(factory_module, factory)
        if companion is not None:
            initial_state = companion

    try:
        graph = _build_graph(factory, initial_state)
    except _PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_GENERIC_ERROR

    callback = DiagnoseCallback()
    invoke_state: dict[str, Any] = (
        dict(initial_state) if isinstance(initial_state, dict) else {}
    )
    try:
        result = graph.invoke(invoke_state, config={"callbacks": [callback]})
    except Exception as exc:  # noqa: BLE001 — graph errors are user-domain
        callback.record_warning(
            f"graph invoke failed: {type(exc).__name__}: {exc}"
        )
        result = {}
    return callback, initial_state, result


def _run_classify_and_detect(
    *,
    events: tuple[Any, ...],
    args: argparse.Namespace,
    initial_state: dict[str, Any] | None,
    result: Any,
) -> tuple[ClassifierVerdict, DetectionReport, dict]:
    """Classify the run, detect divergence, and normalize versions.

    Returns ``(verdict, report, key_index)`` so the caller can reuse the
    key index for the ownership map without re-deriving it.
    """
    candidate_names = _candidate_state_keys(
        initial_state=initial_state if isinstance(initial_state, dict) else None,
        result=result if isinstance(result, dict) else None,
        ignore=args.ignore_keys,
        track=args.track_keys,
    )
    key_index = build_key_index(candidate_names)
    overrides = ClassifierOverrides(
        ignore=tuple(args.ignore_keys),
        track=tuple(args.track_keys),
    )
    verdict = classify(events, key_index=key_index, overrides=overrides)
    _check_writers_by_key_consistency(verdict)
    report = detect(
        events,
        verdict=verdict,
        key_index=key_index,
        strict=args.strict,
        volume_per_hour=args.volume,
        # v0: no checkpointer attached, so token estimates are unavailable
        # and the renderer surfaces the cost-unmeasurable fallback.
        value_token_estimates=None,
        token_cost_per_1k=args.cost_per_1k_tokens,
    )
    return verdict, _normalize_version_strings(report), key_index


def _compute_ownership(
    events: tuple[Any, ...],
    verdict: ClassifierVerdict,
    key_index: dict,
) -> tuple:
    """Thin wrapper around :func:`compute_ownership_map` for symmetry."""
    return compute_ownership_map(events, verdict, key_index)


def _write_outputs(
    *,
    verdict: ClassifierVerdict,
    report: DetectionReport,
    ownership: tuple,
    args: argparse.Namespace,
    consent: ConsentState,
) -> int:
    """Render HTML, JSON, terminal summary, dry-run payload, calibration.

    Returns the final exit code. HTML failures degrade to a notice in
    the summary; JSON failures return ``EXIT_IO_ERROR``. ``--dry-run``
    is always non-zero only on prior failures.
    """
    try:
        options = RenderOptions(
            lead_pain_type=_resolve_lead_pain_type(args),
            warm_lead=args.warm_lead,
            book_a_call_url=args.book_a_call_url,
            contact_email=args.contact_email,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE_ERROR

    output_html = Path(args.output_html)
    html_written = _try_render_html(
        verdict=verdict,
        report=report,
        ownership=ownership,
        output_path=output_html,
        options=options,
    )

    output_json_to_stdout = (not args.no_json) and args.output_json == "-"
    if not args.no_json and not output_json_to_stdout:
        output_json = Path(args.output_json)
        try:
            _write_report_json(verdict=verdict, report=report, path=output_json)
        except OSError as exc:
            print(
                f"error: failed to write JSON report to {output_json}: {exc}",
                file=sys.stderr,
            )
            return EXIT_IO_ERROR

    _emit_summary_and_payloads(
        verdict=verdict,
        report=report,
        consent=consent,
        args=args,
        output_html=output_html,
        output_json_to_stdout=output_json_to_stdout,
        html_written=html_written,
    )

    if html_written is False and (args.no_json or output_json_to_stdout):
        return EXIT_IO_ERROR

    if args.calibration_record is not None:
        _record_calibration(
            verdict=verdict,
            report=report,
            consent=consent,
            calibration_arg=args.calibration_record,
        )
    return 0


def _try_render_html(
    *,
    verdict: ClassifierVerdict,
    report: DetectionReport,
    ownership: tuple,
    output_path: Path,
    options: RenderOptions,
) -> bool:
    """Render HTML report; on OSError print to stderr and return False."""
    try:
        render_html(
            verdict=verdict,
            report=report,
            ownership=ownership,
            output_path=output_path,
            options=options,
        )
        return True
    except OSError as exc:
        print(
            f"error: failed to write HTML report to {output_path}: {exc}",
            file=sys.stderr,
        )
        return False


def _emit_summary_and_payloads(
    *,
    verdict: ClassifierVerdict,
    report: DetectionReport,
    consent: ConsentState,
    args: argparse.Namespace,
    output_html: Path,
    output_json_to_stdout: bool,
    html_written: bool,
) -> None:
    """Print terminal summary, optional JSON-to-stdout, and dry-run payload.

    Stream routing: when ``--dry-run`` or ``--output-json -`` is in
    play, the human summary goes to stderr so stdout stays a
    machine-readable JSON artefact pipeable into ``jq``.
    """
    summary_text = terminal_summary(
        verdict=verdict, report=report, html_path=output_html
    )
    if not html_written:
        summary_text = (
            "(HTML report missing — see error above for details.)\n\n"
            + summary_text
        )
    summary_to_stderr = args.dry_run or output_json_to_stdout
    summary_stream = sys.stderr if summary_to_stderr else sys.stdout
    print(summary_text, file=summary_stream)

    if output_json_to_stdout:
        payload_dict = build_report_json(verdict, report)
        print(json.dumps(payload_dict, indent=2, sort_keys=True, default=str))
    if args.dry_run:
        payload = payload_for(verdict, consent=consent)
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _record_calibration(
    *,
    verdict: ClassifierVerdict,
    report: DetectionReport,
    consent: ConsentState,
    calibration_arg: str,
) -> None:
    """Append one calibration entry and print a status message.

    ``calibration_arg`` is the raw ``args.calibration_record`` value:
    ``""`` when ``--calibration-record`` was passed without a path
    (use the XDG default), or a non-empty path string.
    """
    cal_target = Path(calibration_arg) if calibration_arg else calibration_path()
    cal_result = append_calibration_entry(
        verdict=verdict, report=report, consent=consent, path=cal_target
    )
    if cal_result.written:
        print(f"\ncalibration entry appended to {cal_result.path}")
        return
    if cal_result.reason == "consent_not_granted":
        active = env_kill_switch_active()
        if active is not None:
            print(
                f"\ncalibration write skipped: kill switch active "
                f"({active}); unset to opt in"
            )
        else:
            print(
                "\ncalibration write skipped: consent not granted "
                "(re-run without --no-telemetry / --no-network and answer "
                "'y' to the consent prompt to opt in)"
            )
    elif cal_result.reason.startswith("io_error"):
        print(
            f"\ncalibration write failed: {cal_result.reason}",
            file=sys.stderr,
        )
    else:
        print(f"\ncalibration write skipped: {cal_result.reason}")


# -------------------------------------------------------------------- #
# Pipeline helpers — graph loading
# -------------------------------------------------------------------- #


class _PipelineError(RuntimeError):
    """Internal marker for user-facing pipeline errors."""


@dataclasses.dataclass(frozen=True)
class _LoadError:
    """Sentinel returned from :func:`_load_state_file` on failure."""

    message: str


def _load_graph_factory(graph_arg: str) -> tuple[Any, Any] | int:
    """Parse ``PATH:FUNCTION`` and return ``(factory, module)``.

    On any error prints to stderr and returns the appropriate
    :data:`EXIT_*` integer code. ``ImportError`` (i.e. the
    ``[diagnose]`` extras are missing) returns
    :data:`EXIT_DEPENDENCY_MISSING`; all other failures return
    :data:`EXIT_GENERIC_ERROR`.
    """
    if ":" not in graph_arg:
        print(
            f"error: --graph must be in PATH:FUNCTION format; got {graph_arg!r}",
            file=sys.stderr,
        )
        return EXIT_GENERIC_ERROR

    colon_idx = graph_arg.rfind(":")
    path_str = graph_arg[:colon_idx]
    fn_name = graph_arg[colon_idx + 1 :]

    graph_path = Path(path_str)
    if not graph_path.exists():
        print(f"error: graph file not found: {graph_path}", file=sys.stderr)
        return EXIT_GENERIC_ERROR

    module_dir = str(graph_path.parent.resolve())
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    spec = importlib.util.spec_from_file_location(
        f"_ccs_diagnose_graph_{uuid.uuid4().hex}", graph_path
    )
    if spec is None or spec.loader is None:
        print(f"error: cannot load module from {graph_path}", file=sys.stderr)
        return EXIT_GENERIC_ERROR

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        print(
            f"error: failed to import {graph_path}: {exc}\n"
            "Tip: install the diagnose extras: pip install \"agent-coherence[diagnose]\"",
            file=sys.stderr,
        )
        return EXIT_DEPENDENCY_MISSING
    except Exception as exc:  # noqa: BLE001 — surface user errors verbatim
        print(f"error: failed to import {graph_path}: {exc}", file=sys.stderr)
        return EXIT_GENERIC_ERROR

    factory = getattr(module, fn_name, None)
    if factory is None:
        print(
            f"error: function {fn_name!r} not found in {graph_path}",
            file=sys.stderr,
        )
        return EXIT_GENERIC_ERROR
    return factory, module


def _discover_initial_state(module: Any, factory: Any) -> dict[str, Any] | None:
    """Find a companion ``initial_state*`` callable in ``module``.

    LangGraph factories commonly ship a co-located ``initial_state``
    helper (see ``examples/langgraph_planner/main.py``). Returns its
    result when callable with no required args, otherwise ``None``.
    """
    factory_name = getattr(factory, "__name__", "")
    candidates: list[str] = []
    # Specific to a named factory (e.g. build_graph_no_store →
    # initial_state_no_store).
    if factory_name.startswith("build_graph"):
        suffix = factory_name[len("build_graph") :]
        if suffix:
            candidates.append(f"initial_state{suffix}")
    candidates.extend(
        ("initial_state", f"initial_state_for_{factory_name}", "default_initial_state")
    )

    for name in candidates:
        helper = getattr(module, name, None)
        if helper is None or not callable(helper):
            continue
        try:
            sig = inspect.signature(helper)
            required = [
                p
                for p in sig.parameters.values()
                if p.default is inspect.Parameter.empty
                and p.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
            if required:
                continue
            value = helper()
        except Exception:  # noqa: BLE001 — defensive; fall back silently
            continue
        if isinstance(value, dict):
            return value
    return None


def _load_state_file(path: str | None) -> dict[str, Any] | None | _LoadError:
    """Load ``--state-file`` content. Returns ``None`` when not supplied."""
    if path is None:
        return None
    state_path = Path(path)
    if not state_path.exists():
        return _LoadError(f"--state-file not found: {state_path}")
    try:
        if state_path.stat().st_size > _MAX_INPUT_FILE_SIZE:
            return _LoadError(
                f"--state-file {state_path} exceeds {_MAX_INPUT_FILE_SIZE} bytes"
            )
    except OSError as exc:
        return _LoadError(f"--state-file stat error: {exc}")
    text = state_path.read_text(encoding="utf-8")
    suffix = state_path.suffix.lower()
    try:
        if suffix in (".yaml", ".yml"):
            import yaml  # pyyaml is a base dep per pyproject.toml

            loaded = yaml.safe_load(text)
        else:
            loaded = json.loads(text)
    except Exception as exc:  # noqa: BLE001 — surface parse errors verbatim
        return _LoadError(f"--state-file parse error: {exc}")

    if not isinstance(loaded, dict):
        return _LoadError(
            "--state-file must contain a JSON/YAML object (dict); "
            f"got {type(loaded).__name__}"
        )
    return loaded


def _build_graph(factory: Any, initial_state: dict[str, Any] | None) -> Any:
    """Call the user's factory, threading ``initial_state`` if its signature accepts one."""
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        signature = None

    requires_arg = False
    if signature is not None:
        params = [
            p
            for p in signature.parameters.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        requires_arg = any(
            p.default is inspect.Parameter.empty for p in params
        )

    try:
        if requires_arg and initial_state is not None:
            return factory(initial_state)
        if requires_arg and initial_state is None:
            raise _PipelineError(
                f"factory {factory!r} requires a positional argument; "
                "pass --state-file to supply one."
            )
        return factory()
    except _PipelineError:
        raise
    except TypeError as exc:
        raise _PipelineError(
            f"factory raised TypeError — check its signature: {exc}"
        ) from exc
    except Exception as exc:  # noqa: BLE001 — surface user errors verbatim
        raise _PipelineError(f"factory raised an exception: {exc}") from exc


# -------------------------------------------------------------------- #
# Pipeline helpers — argument plumbing
# -------------------------------------------------------------------- #


def _parse_csv(value: str) -> tuple[str, ...]:
    """Parse a comma-separated string into a tuple of non-empty entries."""
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _resolve_lead_pain_type(args: argparse.Namespace) -> str:
    """Honour the auto-detect rule from the plan's HLTD decision matrix."""
    requested = args.lead_pain_type
    if requested != "auto":
        return requested
    if args.volume is not None and args.volume > 0:
        return "cost"
    return "auditability"


def _candidate_state_keys(
    *,
    initial_state: Mapping[str, Any] | None,
    result: Mapping[str, Any] | None,
    ignore: Sequence[str],
    track: Sequence[str],
) -> tuple[str, ...]:
    """Return the universe of top-level state-key names for ``build_key_index``.

    Combines initial-state keys, result keys, and any name surfaced via
    ``--ignore`` / ``--track`` so the classifier can match overrides
    against the observed UUIDs.
    """
    seen: set[str] = set()
    if initial_state is not None:
        seen.update(str(k) for k in initial_state)
    if result is not None:
        seen.update(str(k) for k in result)
    seen.update(ignore)
    seen.update(track)
    return tuple(sorted(seen))


# -------------------------------------------------------------------- #
# Pipeline helpers — defensive checks + normalisation
# -------------------------------------------------------------------- #


def _check_writers_by_key_consistency(verdict: ClassifierVerdict) -> None:
    """Surface a warning if ``writers_by_key`` disagrees with ``tracked_keys``.

    Tracked keys can be a strict superset (artifacts that were only ever
    read have an empty writers entry — that's fine and the v0 contract).
    The defensive case is a missing entry: a tracked key absent from the
    map. We surface that to stderr without crashing the pipeline; users
    still get a report, but they're warned the classifier and the
    downstream stages may disagree.
    """
    # Insufficient verdicts return empty ``writers_by_key`` by design (the
    # short-circuit paths skip writer attribution). Skipping the consistency
    # check avoids a guaranteed false positive on every short-circuit run.
    if verdict.bucket is Bucket.INSUFFICIENT:
        return
    expected = set(verdict.tracked_keys)
    actual = set(verdict.writers_by_key)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        print(
            "warning: classifier writers_by_key inconsistency — "
            f"missing: {sorted(missing)}, extra: {sorted(extra)}",
            file=sys.stderr,
        )


def _normalize_version_strings(report: DetectionReport) -> DetectionReport:
    """Strip a leading ``v`` from every read observation's version string.

    Some checkpointer encodings already prefix versions with ``v``; when the
    terminal summary later wraps them as ``v<earlier>→v<later>`` the result
    becomes ``vv1``. Per the Unit 6 spec gap, normalise here so downstream
    stages see clean numeric/hash strings.

    Returns a new report (the inputs are frozen dataclasses).
    """

    def _strip(version: str) -> str:
        if version.startswith("v") and len(version) > 1:
            return version[1:]
        return version

    def _strip_obs(obs: ReadObservation) -> ReadObservation:
        return replace(obs, version=_strip(obs.version))

    headline = tuple(
        replace(
            ev,
            earlier_read=_strip_obs(ev.earlier_read),
            later_read=_strip_obs(ev.later_read),
        )
        for ev in report.headline_divergence_events
    )
    excluded = tuple(
        replace(
            ev,
            earlier_read=_strip_obs(ev.earlier_read),
            later_read=_strip_obs(ev.later_read),
        )
        for ev in report.excluded_events
    )

    top_event = report.top_event
    if top_event is not None:
        top_event = replace(
            top_event,
            earlier_read=_strip_obs(top_event.earlier_read),
            later_read=_strip_obs(top_event.later_read),
        )

    return replace(
        report,
        headline_divergence_events=headline,
        excluded_events=excluded,
        top_event=top_event,
    )


# -------------------------------------------------------------------- #
# Pipeline helpers — JSON serialisation
# -------------------------------------------------------------------- #


def _write_report_json(
    *, verdict: ClassifierVerdict, report: DetectionReport, path: Path
) -> None:
    """Write the verdict + report to ``path`` as a single JSON object.

    Thin wrapper over :func:`build_report_json` that writes the result to
    disk. UUIDs and Paths are coerced via ``default=str`` for JSON safety.
    """
    payload = build_report_json(verdict, report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


if __name__ == "__main__":
    try:
        raise SystemExit(console_main())
    except BrokenPipeError:
        # Mirror the conventional shell exit status for a SIGPIPE-killed
        # process. Without this, piping into ``head`` etc. surfaces a
        # spurious traceback at interpreter shutdown.
        raise SystemExit(141) from None
