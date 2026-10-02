#!/usr/bin/env python3
"""Run interactively in your VM terminal. Never paste API keys into task messages."""
import getpass
import os
from pathlib import Path
import sys

ROOT = Path("/home/azureuser/ai-swarm")
if sys.platform != "linux" or not ROOT.is_dir() or not sys.stdin.isatty():
    raise SystemExit("Run interactively in the Azure VM terminal.")
directory = ROOT / "credentials"
directory.mkdir(mode=0o700, exist_ok=True)
os.chmod(directory, 0o700)
key = getpass.getpass("OpenAI API key (hidden; saved only on this VM): ").strip()
if not key or any(c.isspace() for c in key):
    raise SystemExit("Empty or malformed input; nothing saved.")
path = directory / "openai_api_key"
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
with os.fdopen(fd, "w") as out:
    os.fchmod(out.fileno(), 0o600)
    out.write(key + "\n")
print("Credential saved privately on the VM. No API calls were made.")
