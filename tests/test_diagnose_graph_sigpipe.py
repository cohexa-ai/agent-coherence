# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The console's quiet stdout handling must not kill user graph code."""
import json
import os
import signal
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.skipif(not hasattr(signal, 'SIGPIPE'), reason='no SIGPIPE on this platform')
@pytest.mark.parametrize('phase', ['import', 'build', 'invoke'])
def test_console_graph_can_handle_broken_pipe(tmp_path, phase):
    graph = tmp_path / 'graph.py'
    graph.write_text(textwrap.dedent('''\
        import socket
        from typing import TypedDict
        from langgraph.graph import START, END, StateGraph

        def broken_pipe():
            left, right = socket.socketpair()
            right.close()
            try:
                left.sendall(b'synthetic request')
            except (BrokenPipeError, ConnectionResetError):
                return True
            finally:
                left.close()
            raise AssertionError('write unexpectedly succeeded')

        PHASE = %r
        if PHASE == 'import':
            assert broken_pipe()

        class State(TypedDict, total=False):
            plan: str

        def node(state):
            if PHASE == 'invoke':
                assert broken_pipe()
            return {'plan': 'done'}

        def build():
            if PHASE == 'build':
                assert broken_pipe()
            graph = StateGraph(State)
            graph.add_node('call', node)
            graph.add_edge(START, 'call')
            graph.add_edge('call', END)
            return graph.compile()
    ''' % phase), encoding='utf-8')
    html, report = tmp_path / 'r.html', tmp_path / 'r.json'
    env = {key: os.environ[key] for key in ('PATH', 'PYTHONPATH') if key in os.environ}
    env.update(HOME=str(tmp_path), XDG_CONFIG_HOME=str(tmp_path / 'config'), CI='1')
    result = subprocess.run([
        sys.executable, '-m', 'ccs.cli.diagnose', '--graph', f'{graph}:build',
        '--no-network', '--no-telemetry', '--output-html', str(html),
        '--output-json', str(report),
    ], capture_output=True, text=True, timeout=30, env=env)
    assert result.returncode == 0, (result.returncode, result.stderr)
    assert '<html' in html.read_text(encoding='utf-8').lower()
    assert 'report' in json.loads(report.read_text(encoding='utf-8'))
    assert 'Your write pattern' in result.stdout


@pytest.mark.skipif(not hasattr(signal, 'SIGPIPE'), reason='no SIGPIPE on this platform')
def test_graph_guard_restores_handler_on_error_in_subprocess(tmp_path):
    code = '''
import signal
from ccs.cli.diagnose import _graph_sigpipe_guard
for previous in (signal.SIG_DFL, signal.SIG_IGN, lambda *args: None):
    signal.signal(signal.SIGPIPE, previous)
    original = RuntimeError('synthetic failure')
    try:
        with _graph_sigpipe_guard():
            assert signal.getsignal(signal.SIGPIPE) == signal.SIG_IGN
            raise original
    except RuntimeError as caught:
        assert caught is original
    else:
        raise AssertionError('original error lost')
    assert signal.getsignal(signal.SIGPIPE) is previous
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True,
                            text=True, timeout=15)
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(not hasattr(signal, 'SIGPIPE'), reason='no SIGPIPE on this platform')
@pytest.mark.parametrize('unbuffered', [False, True])
def test_console_pipeline_closed_stdout_stays_quiet(tmp_path, unbuffered):
    from pathlib import Path

    graph = Path(__file__).resolve().parent.parent / 'examples/langgraph_planner/main.py'
    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    try:
        result = subprocess.run([
            sys.executable, *(['-u'] if unbuffered else []), '-m', 'ccs.cli.diagnose',
            '--graph', f'{graph}:build_graph_no_store', '--no-network', '--no-telemetry',
            '--output-html', str(tmp_path / 'r.html'), '--output-json', str(tmp_path / 'r.json'),
        ], stdout=write_fd, stderr=subprocess.PIPE, text=True, timeout=30,
            env={**os.environ, 'PYTHONUNBUFFERED': '1' if unbuffered else ''})
    finally:
        os.close(write_fd)
    assert result.returncode == -signal.SIGPIPE
    assert result.stderr == ''


def test_console_graph_failure_is_still_reported(tmp_path):
    graph = tmp_path / 'failure.py'
    graph.write_text('''
def build():
    class Graph:
        def invoke(self, state, config=None):
            raise RuntimeError('synthetic graph failure')
    return Graph()
''', encoding='utf-8')
    report = tmp_path / 'r.json'
    result = subprocess.run([
        sys.executable, '-m', 'ccs.cli.diagnose', '--graph', f'{graph}:build',
        '--no-network', '--no-telemetry', '--output-html', str(tmp_path / 'r.html'),
        '--output-json', str(report),
    ], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'graph invoke failed: RuntimeError: synthetic graph failure' in result.stderr
    assert json.loads(report.read_text(encoding='utf-8'))['verdict']['bucket'] == 'insufficient'


def test_graph_guard_without_sigpipe_does_not_change_signals(monkeypatch):
    from types import SimpleNamespace

    from ccs.cli import diagnose

    monkeypatch.setattr(diagnose, 'signal', SimpleNamespace())
    with diagnose._graph_sigpipe_guard():
        pass
