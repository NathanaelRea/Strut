# OpenSees Reference Workspace

Local staging area for ad hoc OpenSees Tcl examples that Strut compares
against. The native Linux reference binary lives at
`./.build/opensees-linux/OpenSees` by default and is invoked by
`scripts/run_opensees.sh`.

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
