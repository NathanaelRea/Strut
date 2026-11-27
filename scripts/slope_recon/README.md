# Effective-stress slope example

The generator reconstructs the October 26, 2010 revision of
`SlopeAnalysis2Deff.tcl` from the OpenSees Wiki. The template contains the
original material definitions, constraints, loading, and analysis commands.
The compressed JSON supplies the coordinates, node numbering, connectivity,
and material assignment for the irregular GiD mesh. The generator reuses
those tables for the GiD output commands and permeability updates.

```sh
uv run python -m scripts.slope_recon.bundle --install
uv run python -m scripts.slope_recon.bundle --out-dir /tmp/opensees-slope
```

The generated Tcl matches the 581,583-byte source byte for byte. The installer
verifies SHA-256
`f7fb50b184803c7589d2de3d372198e5b244a23461679f4e070f8a250b18ff76`.
Setup also installs the published velocity time history next to the generated
script. [OpenSees Wiki source](https://opensees.berkeley.edu/wiki/index.php/File:SlopeAnalysis2Deff.tcl)
