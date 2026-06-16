#!/usr/bin/env python3
"""Backward-compatible wrapper for the Juno Agent Harness CLI.

The implementation now lives in the `juno_agent_harness` package. This shim
keeps `import harness` and `python harness.py ...` working for existing users.
"""
from __future__ import annotations

from juno_agent_harness.cli import *  # noqa: F401,F403

if __name__ == "__main__":
    raise SystemExit(main())
