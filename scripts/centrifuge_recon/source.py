"""Identify the specific published GiD export this reconstruction uses."""

from hashlib import sha256
from pathlib import Path

ORIGINAL_SHA256 = "5e08e9cceece819705501f3a42b14e7cd9573f9023fc419851b29dbe11f2a8b2"


def read_original(path: Path) -> str:
    data = path.read_bytes()
    if sha256(data).hexdigest() != ORIGINAL_SHA256:
        raise ValueError(f"{path} does not match the published CentrifugeExample.tcl")
    return data.decode("ascii")
