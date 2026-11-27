import os


def mojo_target_args() -> list[str]:
    """Return the optional CPU architecture used by CI builds and tests."""
    arch = os.environ.get("STRUT_MOJO_ARCH")
    return [f"--march={arch}"] if arch else []
