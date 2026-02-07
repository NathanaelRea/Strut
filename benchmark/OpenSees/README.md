# OpenSees Reference Workspace

Local staging area for OpenSees binaries and Tcl examples that Strut compares
against. Drop the official `OpenSees.exe`, supporting DLLs, and the `lib/`
directory from the upstream ZIP into this folder.

## Layout

```text
OpenSees/
├── README.md                # this file
├── OpenSees.exe             # ignored – user-provided binary
├── lib/                     # ignored – Tcl runtime from OpenSees ZIP
└── examples/
    └── <example>/
        ├── <example>.tcl           # Tcl script
        └── Data/                   # ignored - recorder outputs
```

Add each OpenSees example under `examples/<case_name>/` so recorder files stay
contained per problem. The Wine helper walks every example directory, runs each
`*.tcl` file it finds, and writes the recorder outputs into
`tests/validation/<case_name>/out/`.

```bash
# Run every example
scripts/run_opensees_wine.sh

# Custom subset
scripts/run_opensees_wine.sh OpenSees/examples/my_case/MyCase.tcl
```
