"""Bundle checksums. Kept free of training dependencies so that serving can import it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_bundle(directory: Path) -> None:
    """Refuse to serve a bundle whose files differ from what the manifest recorded."""
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"Bundle file {name} does not match its manifest checksum")
