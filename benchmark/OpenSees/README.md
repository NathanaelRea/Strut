# OpenSees Reference Workspace

Local staging area for ad hoc OpenSees Tcl examples that Strut compares
against. `scripts/run_opensees.sh` runs them with the locked reference binary
installed by `uv run scripts/setup_opensees.py` (see the main README).

## Layout

```text
OpenSees/
├── README.md                # this file
└── examples/
    └── <example>/
        ├── <example>.tcl           # Tcl script
        └── Data/                   # ignored - recorder outputs
```

Add each OpenSees example under `examples/<case_name>/` so recorder files stay
contained per problem. The native runner walks every example directory, runs
each `*.tcl` file it finds, and writes the recorder outputs into
`tests/validation/<case_name>/reference/`.

```bash
# Run every example
scripts/run_opensees.sh

# Custom subset
scripts/run_opensees.sh OpenSees/examples/my_case/MyCase.tcl
```
