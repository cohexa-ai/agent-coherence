"""Throwaway CI probe: fails on Python 3.12 only, so only that leg's gate should go red."""

import sys


def test_ci_probe_fails_only_on_python_3_12() -> None:
    assert sys.version_info[:2] != (3, 12), "deliberate CI probe failure"
# probe commit 2
