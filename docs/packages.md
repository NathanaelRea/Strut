# Packages

Strut uses `uv` for Python dependency management.

Note: Mojo is required to run the solver. Python is used for the harness
(JSON -> Tcl conversion, OpenSees runs, and comparisons).

## Adding Packages

Use `uv add` to add a dependency to `pyproject.toml`.

```bash
uv add <package>
```

For a dev-only dependency:

```bash
uv add --dev <package>
```
