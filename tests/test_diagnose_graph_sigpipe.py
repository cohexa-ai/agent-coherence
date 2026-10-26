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

pytest.importorskip("langgraph")


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
@pytest.mark.parametrize('unrelated_warning', [False, True])
def test_console_pipeline_closed_stdout_stays_quiet(tmp_path, unbuffered, unrelated_warning):
    from pathlib import Path

    graph = Path(__file__).resolve().parent.parent / 'examples/langgraph_planner/main.py'
    if unrelated_warning:
        wrapper = tmp_path / 'warning_graph.py'
        wrapper.write_text(
            "import warnings\n"
            "warnings.simplefilter('always')\n"
            "warnings.warn('synthetic unrelated import notice', UserWarning)\n"
            "class Graph:\n"
            "    def invoke(self, state, config=None): return {'plan': 'done'}\n"
            "def build_graph_no_store(): return Graph()\n",
            encoding='utf-8',
        )
        graph = wrapper
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
    if unrelated_warning:
        assert 'synthetic unrelated import notice' in result.stderr
    for noise in ('Traceback', 'BrokenPipe', 'Broken pipe', 'DiagnoseWarning'):
        assert noise not in result.stderr


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
    before = sys.stdout
    with diagnose._graph_sigpipe_guard():
        assert sys.stdout is before
    assert sys.stdout is before


@pytest.mark.skipif(not hasattr(signal, 'SIGPIPE'), reason='no SIGPIPE on this platform')
@pytest.mark.parametrize('phase', ['import', 'build', 'invoke'])
@pytest.mark.parametrize('mode', ['unbuffered', 'flush', 'bufferfull'])
@pytest.mark.parametrize('worker', [False, True])
def test_graph_closed_stdout_exits_without_failure_report(tmp_path, phase, mode, worker):
    """A graph's prints, not only the CLI summary, use quiet pipe semantics."""
    action = {
        'unbuffered': "print('synthetic')",
        'flush': "print('synthetic', flush=True)",
        'bufferfull': "print('x' * 16384)",
    }[mode]
    graph = tmp_path / 'graph.py'
    graph.write_text(textwrap.dedent(f'''\
        from concurrent.futures import ThreadPoolExecutor
        PHASE = {phase!r}
        def write():
            {action}
        def emit():
            if {worker!r}:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    executor.submit(write).result(timeout=5)
            else:
                write()
        if PHASE == 'import': emit()
        class Graph:
            def invoke(self, state, config=None):
                if PHASE == 'invoke': emit()
                return {{'plan': 'done'}}
        def build():
            if PHASE == 'build': emit()
            return Graph()
    '''), encoding='utf-8')
    html, report = tmp_path / 'r.html', tmp_path / 'r.json'
    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    try:
        result = subprocess.run([
            sys.executable, *(['-u'] if mode == 'unbuffered' else []),
            '-m', 'ccs.cli.diagnose', '--graph', f'{graph}:build',
            '--no-network', '--no-telemetry', '--output-html', str(html),
            '--output-json', str(report),
        ], stdout=write_fd, stderr=subprocess.PIPE, text=True, timeout=30,
            env={**os.environ, 'HOME': str(tmp_path), 'CI': '1',
                 'XDG_CONFIG_HOME': str(tmp_path / 'config'), 'PYTHONUNBUFFERED': ''})
    finally:
        os.close(write_fd)
    # Thread-safe immediate exit uses the shell convention, not signal death.
    assert result.returncode == 128 + signal.SIGPIPE, result.stderr
    assert not html.exists()
    assert not report.exists()
    for noise in ('Traceback', 'BrokenPipe', 'Broken pipe', 'DiagnoseWarning'):
        assert noise not in result.stderr


@pytest.mark.skipif(not hasattr(signal, 'SIGPIPE'), reason='no SIGPIPE on this platform')
def test_graph_guard_restores_stdout_and_preserves_stream_behavior(tmp_path):
    code = '''
import io
import signal
import sys
from ccs.cli.diagnose import _graph_sigpipe_guard
original_stdout = sys.stdout
stream = io.TextIOWrapper(io.BytesIO(), encoding='utf-8')
sys.stdout = stream
previous = signal.getsignal(signal.SIGPIPE)
try:
    for fail in (False, True):
        error = RuntimeError('synthetic')
        try:
            with _graph_sigpipe_guard():
                assert sys.stdout is not stream
                assert sys.stdout.buffer is stream.buffer  # documented passthrough
                assert sys.stdout.encoding == stream.encoding
                assert sys.stdout.writable()
                assert sys.stdout.write('hello') == 5
                sys.stdout.flush()
                if fail: raise error
        except RuntimeError as caught:
            assert caught is error
        assert sys.stdout is stream
        assert signal.getsignal(signal.SIGPIPE) is previous
finally:
    sys.stdout = original_stdout
assert stream.buffer.getvalue() == b'hellohello'
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True,
                            text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_stdout_proxy_does_not_swallow_other_stream_errors():
    from ccs.cli.diagnose import _ConsoleStdout

    error = OSError('not a broken pipe')

    class Stream:
        def write(self, data):
            raise error

        def flush(self):
            raise error

    proxy = _ConsoleStdout(Stream())
    for operation in (lambda: proxy.write('test'), proxy.flush):
        with pytest.raises(OSError) as caught:
            operation()
        assert caught.value is error


def test_graph_unhandled_socket_error_is_not_a_console_exit(tmp_path):
    graph = tmp_path / 'socket_failure.py'
    graph.write_text('''
import socket
class Graph:
    def invoke(self, state, config=None):
        left, right = socket.socketpair()
        right.close()
        try:
            left.sendall(b'synthetic request')
        finally:
            left.close()
def build():
    return Graph()
''', encoding='utf-8')
    report = tmp_path / 'r.json'
    result = subprocess.run([
        sys.executable, '-m', 'ccs.cli.diagnose', '--graph', f'{graph}:build',
        '--no-network', '--no-telemetry', '--output-html', str(tmp_path / 'r.html'),
        '--output-json', str(report),
    ], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'DiagnoseWarning: graph invoke failed:' in result.stderr
    assert 'BrokenPipeError' in result.stderr or 'ConnectionResetError' in result.stderr
    assert json.loads(report.read_text(encoding='utf-8'))['verdict']['bucket'] == 'insufficient'
