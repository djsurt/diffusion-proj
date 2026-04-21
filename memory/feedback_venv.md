---
name: Use venv for Python packages
description: User wants Python packages installed in a venv, not globally
type: feedback
---

Always create and use a virtual environment (venv) for Python package installation. Never use `pip3 install --break-system-packages` or global installs.

**Why:** User doesn't want global Python package pollution.

**How to apply:** Before installing packages, create a venv (`python3 -m venv .venv`) and activate it, then use `.venv/bin/pip install`. Reference the venv's Python for all script execution.
