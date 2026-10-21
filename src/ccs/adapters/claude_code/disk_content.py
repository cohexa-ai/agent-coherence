# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Hash a workspace file's raw bytes on the coordinator's own host.

Two readers compare the file on disk with the content the coordinator recorded:
the sweep's foreign-write detection pass, and pre-edit's out-of-band check
(#278). Both must hash the same way every writer hashed what it committed, and
both must be safe to point at a name a session supplied, so the read lives here
once rather than in either of them.
"""

from __future__ import annotations

import logging
import os
import stat
from pathlib import Path

from ccs.core.substrate import sha256_hex

logger = logging.getLogger(__name__)

DEFAULT_MAX_HASH_BYTES = 64 * 1024 * 1024
"""Past this a file is not hashed: a check on a hook or sweep path must never
be the reason that path stalls. A skip is a coverage gap, not an answer."""


def hash_regular_file(path: Path, *, max_bytes: int = DEFAULT_MAX_HASH_BYTES) -> str | None:
    """Hash the file's raw bytes, or return None when it is not observable.

    Raw bytes and the one canonical helper: no decode, no newline translation,
    no git filter. A normalization here would be the first in the system and
    would diverge this comparison from every writer that produced a stored hash.

    One descriptor answers every question, which closes three gaps a
    stat-then-open sequence leaves open. ``O_NOFOLLOW`` refuses a symlink leaf,
    so a coordinated name cannot be pointed at a file outside the workspace.
    ``O_NONBLOCK`` means a path replaced by a named pipe fails instead of
    blocking this thread forever. And checking the size on the open descriptor,
    then reading one byte past the cap, means a file that grows between the
    check and the read is still refused rather than read unbounded.

    None covers every way an artifact can stop being an observable regular file
    — deleted, renamed away, replaced by a directory or a link, permission
    revoked, or larger than ``max_bytes``. All of those are coverage gaps.
    """
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        return None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            return None
        if info.st_size > max_bytes:
            logger.debug("%s exceeds the %d-byte hash cap; not hashed", path, max_bytes)
            return None
        with os.fdopen(fd, "rb", closefd=True) as handle:
            fd = -1  # ownership moved to the file object
            data = handle.read(max_bytes + 1)
        if len(data) > max_bytes:
            logger.debug("%s grew past the %d-byte hash cap; not hashed", path, max_bytes)
            return None
        return sha256_hex(data)
    except OSError:
        return None
    finally:
        if fd >= 0:
            os.close(fd)
