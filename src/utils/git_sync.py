"""Git synchronization helpers for run artifacts and feature changes.

These helpers stage, commit and push results to the ``main`` branch. They are
designed to be safe to call from automation: any git or network failure is
logged as a warning (including the exact error) and swallowed so a failed
commit or push never crashes the optimizer.
"""

from __future__ import annotations

import logging
import subprocess
from typing import Sequence

logger = logging.getLogger(__name__)

# Artifacts committed on every automatic run.
_DEFAULT_PATHS: tuple[str, ...] = ("HISTORY.md", "CLAUDE.md", "outputs/")

_REMOTE = "origin"
_BRANCH = "main"


def _run_git(args: Sequence[str]) -> bool:
    """Run a git command, returning ``True`` on success.

    Failures (non-zero exit, missing git binary) are logged as warnings with
    the exact error output and never raised.

    Args:
        args: Git arguments following the ``git`` executable, e.g.
            ``["add", "HISTORY.md"]``.

    Returns:
        ``True`` if git exited 0, ``False`` otherwise.
    """
    command = ["git", *args]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, FileNotFoundError) as exc:
        logger.warning("Failed to execute %s: %s", " ".join(command), exc)
        return False

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        logger.warning(
            "git command failed (%s): %s",
            " ".join(command),
            detail or f"exit code {result.returncode}",
        )
        return False

    return True


def auto_commit(timestamp: str, changed_files: list[str] | None = None) -> None:
    """Stage run artifacts, commit them and push to ``origin/main``.

    Always stages ``HISTORY.md``, ``CLAUDE.md`` and the ``outputs/`` directory
    (new HTML and CSV files included). Any extra paths in ``changed_files`` are
    staged as well.

    Args:
        timestamp: Run timestamp used in the commit message.
        changed_files: Optional additional paths to stage.

    The function never raises; git/push failures are logged and swallowed.
    """
    paths = list(_DEFAULT_PATHS)
    if changed_files:
        paths.extend(changed_files)

    # Stage the standard artifacts (and any extras). Missing paths are tolerated
    # by git only with pathspec magic, so stage them individually and warn on
    # failure rather than aborting the whole commit.
    if not _run_git(["add", *paths]):
        logger.warning("auto_commit: staging failed; skipping commit and push")
        return

    if not _run_git(["commit", "-m", f"chore: run results {timestamp}"]):
        # Most common cause: nothing to commit. Already logged by _run_git.
        logger.info("auto_commit: nothing committed; skipping push")
        return

    if not _run_git(["push", _REMOTE, _BRANCH]):
        logger.warning("auto_commit: push to %s/%s failed", _REMOTE, _BRANCH)
        return

    logger.info("auto_commit: committed and pushed run results %s", timestamp)


def commit_feature(message: str, files: list[str] | None = None) -> None:
    """Stage, commit and push a development change (feature/fix/refactor).

    Args:
        message: Commit message describing the change.
        files: Optional explicit paths to stage. When omitted, all changes in
            the working tree are staged (``git add -A``).

    The function never raises; git/push failures are logged and swallowed.
    """
    add_args = ["add", *files] if files else ["add", "-A"]

    if not _run_git(add_args):
        logger.warning("commit_feature: staging failed; skipping commit and push")
        return

    if not _run_git(["commit", "-m", message]):
        logger.info("commit_feature: nothing committed; skipping push")
        return

    if not _run_git(["push", _REMOTE, _BRANCH]):
        logger.warning("commit_feature: push to %s/%s failed", _REMOTE, _BRANCH)
        return

    logger.info("commit_feature: committed and pushed %r", message)
