# Model1 geometry reconstruction

`generate()` in [geometry.py](geometry.py) builds the benchmark mesh from the adjacent parameter and stencil tables. It returns node coordinates keyed by the original node IDs, ordered element node tuples keyed by the original element IDs, and diaphragm slave sets keyed by master node ID. The `element_order` mapping gives the Tcl command order for each source family. Generation uses the Python standard library and takes about 0.2 s for the full model.

Run the comparison from the repository root after installing the reference data.

```sh
uv run scripts/setup_reference_data.py
uv run scripts/megatall_recon/compare_geometry.py
```

## Geometry rules

- The base is at 0 m. The 128 increments consist of an initial seven 6 m stories, repeated three-story sets of 5.5 m at the transitions, and intervening runs of 4.5 m and 4.3 m. Every diaphragm floor has a wall-head plane 1 m below it. This accounts for all 257 elevations through 606.1 m. [Node source](../../tests/fixtures/opensees/OpenSeesExamples/MegaTallBuildingBenchmark/Model1/opensees.tcl)
- Core nodes follow repeated wall grids on ±1.5, ±5, ±8.5, ±10, ±11.5, ±13.25 and ±15 m. The lower typical core has 72 floor and wall-head points. The 270–329.5 m run uses the 12.725 m corner and a reduced 68-point wall head after the first transition. The 345–478.5 m runs use an 11.75 m corner and 60-point wall heads. The upper typical core has 48 points on both planes. At 420–478.5 m the diagonal at 14.5371 m contributes four more points inside the 15 m core boundary. [Core shell source](../../tests/fixtures/opensees/OpenSeesExamples/MegaTallBuildingBenchmark/Model1/layershell.tcl)
- The façade polygon has successive nominal circumradii of 40, 37.5, 34.5, 32, 29.5, 27, 24.5, 23 and 22 m. Endpoints on the ±5 m gridline use the circle equation and diagonal vertices use radius divided by √2. Off-grid vertices such as (13.7416, 26.1040) and (11.2406, 20.0661) lie on the 29.5 m and 23 m circles. Several other off-grid points split polygon chords. For example, (14.5180, 19.4049) is a point on the chord from (11.8184, 21.4610) to (17.3241, 17.3241). The 22.5 m projection on the ±5 m gridline gives the local 23.05 m radial measurement. [Node source](../../tests/fixtures/opensees/OpenSeesExamples/MegaTallBuildingBenchmark/Model1/opensees.tcl)
- The inner mega-column rails interpolate in physical elevation, with plateaus through the belt-truss stories. At fixed transverse gridlines, secondary members intersect chords between the moving rails and other anchors. At z = 133.5 m, for example, x = 8.5 and x = 11.5 intersect the chord from (5, 32.0868) to (15.6619, 30.7401). Near the roof the moving rails give way to fixed points. The table uses 30 circle recipes, 63 chord recipes, and linear or piecewise-linear tracks for the remaining moving families. [Beam source](../../tests/fixtures/opensees/OpenSeesExamples/MegaTallBuildingBenchmark/Model1/Hbeam.tcl)
- The eight belt-truss triples are centered at 53, 119, 189.5, 260, 335, 410, 489.5 and 566.2 m. They reuse axial and diagonal rails and split long braces at intermediate positions. Uniform scaling cannot map the triples to one another. For example, the middle planes at 53 and 119 m contain 284 and 244 nodes, and both include the (5, 16.5) and (5, 20.6155) stations. The one-to-three-floor track groups supply these changing plan details. [Brace source](../../tests/fixtures/opensees/OpenSeesExamples/MegaTallBuildingBenchmark/Model1/truss.tcl)
- Reflections account for almost all plan points. At 1 mm, 20 nodes lack the x reflection, 16 lack the y reflection, and 2,274 lack a 90°-rotated mate. Most of the rotation differences are in the core and wall-head layout or involve the perimeter geometry at transitions. Reflection masks preserve the occupancy differences and the coordinate fit absorbs the small rounding differences.
- Each diaphragm is a small selection of core and rail orbits, with 17, 19 or 23 slaves. Its master has the smallest eligible ID among the participants. The eligible orbit is the perimeter until the final eight floors, when the (5, 15) node also becomes eligible. [Diaphragm source](../../tests/fixtures/opensees/OpenSeesExamples/MegaTallBuildingBenchmark/Model1/rigiddigram.tcl)

## IDs, order and table format

The first 17,122 node IDs are on the floor or base planes, but their elevations are interleaved across the height. The wall-head IDs start at 17,123 and interleave with the remaining floor IDs. The permutation walks the generated mesh. It uses ranks among unused adjacent nodes for 8,624 IDs, ranks along the previous plan slot for 4,070 IDs, and explicit plane and unused-slot ranks for the other 14,526 IDs. The original ordering function is unknown.

Element IDs mix plan-major, span-major and irregular sequences. For example, Hbeam IDs 1–72 step through eight perimeter segments, each on the same nine floors in identical source order. Later parts of Hbeam repeat the segment list at successive elevations. Shells repeatedly connect the floor and wall-head planes. Each connection stencil uses relative plane offsets and plan slots. A repeat expands a contiguous stencil over translated planes, optionally changing its octant selectors. Of 50,350 elements, 31,742 are repeats of stored stencils. The table retains 18,608 seed or irregular stencils, the tag gaps, and the file-order permutations for the mega-columns and shells.

A track group in [params.json](params.json) is `[first, last, rows]` with inclusive floor indices. Each row starts with `[x, y]` or a shape recipe. An optional integer reflection mask selects plan copies and an optional list of `[first, last, mask]` rows overrides that mask. Floor indices start at zero for the base. Wall-head indices start at zero for the plane under floor 1. Diaphragm rows select full or partial plan orbits by track and mask.

[legacy.bin](legacy.bin) contains independent LZMA2 packets. The lengths in `packet_bytes` identify the ordered element-family packets, then the node permutation packet, then the supplemental track groups and their insertion indices. The decoder documents the integer streams. The supplemental groups share the JSON track schema and store the transitional and secondary trajectories. The JSON retains the main core, façade and inclined rail anchors. The stencils reference geometric slots. They omit materials, section assignments, loads, masses and analysis commands.

## Element tag handoff

| File and family | Section or material tag |
| --- | --- |
| `opensees.tcl` `dispBeamColumn` | Section tag = 20,000,000 + element ID on all 1,240 rows |
| `Hbeam.tcl` `dispBeamColumn` | 148 section tags from 10,000 to 1,480,000 in steps of 10,000 |
| `layershell.tcl` `ShellMITC4` | 91 section tags from 8501 to 17501 in steps of 100 |
| `truss.tcl` `elasticBeamColumn` | Explicit section constants and transformation tag 277 |
| `truss.tcl` `truss` | Uniaxial material tag 179 |

## Residual and size

All 27,220 node IDs match within 0.01 m. The maximum 3D error is 0.002904 m at the symmetric copies of the interpolated point (19.5420, 25.8646) at z = 169.5 m. The p99 error is 0.000164 m overall and 0.000774 m for perimeter floor nodes. All 50,350 element IDs, types, ordered endpoint tuples and source-family orders match exactly. All 128 diaphragm master/slave sets match.

The 3,234-byte parameter file plus the 46,717-byte stencil and numbering table total 49,951 bytes. This is 122.3× smaller than the 6,107,176 bytes of original geometry command lines, which include their original padding and element property tokens. Compared with the 3,078,225 bytes of geometry tokens alone with normalized whitespace, the reduction is 61.6×. The side table contains 19,684 bytes of element stencils and tag/order information, 24,756 bytes of node numbering, and 2,277 bytes of supplemental track groups.

No extra data is required at 0.01 m. To reproduce the original printed four-decimal coordinates, 1,180 nodes would need XY corrections. A measured sparse stream of node ID gaps and signed integer XY deltas would add 1,788 bytes with LZMA compression. The exporter’s ordering algorithm, exact formulas for the remaining interpolated tracks, and its one-off transition choices remain open.
