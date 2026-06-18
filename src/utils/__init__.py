"""Utility package for the quantitative portfolio optimizer.

Provides:
    - history_logger: append run results to HISTORY.md and sync CLAUDE.md.
    - git_sync: stage/commit/push run artifacts and feature changes.
    - cli: command-line argument parser for the optimizer entrypoint.
"""

from src.utils.cli import build_parser
from src.utils.git_sync import auto_commit, commit_feature
from src.utils.history_logger import log_run, update_claude_md

__all__ = [
    "build_parser",
    "auto_commit",
    "commit_feature",
    "log_run",
    "update_claude_md",
]
