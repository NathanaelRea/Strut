# Third-party notices

## Purdue University Research Repository wall model

The original Tcl and publication metadata in
`tests/fixtures/opensees/OpenSeesWallExamples/PURR4348/` come from Morgan Broberg
and Amit H. Varma, *OpenSees Models for C-PSW/CF Archetype Structures*, version
1.0 (2023). The archived files are reproduced unchanged under the Creative
Commons Attribution 4.0 International license. The directory also contains
Strut test drivers that load the archived inputs.
[Dataset](https://doi.org/10.4231/9H4P-Z894) ·
[CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/)

## STAIRlab OpenSees Gallery

Copyright (c) 2024, Structural Artificial Intelligence Research Lab. The Tcl
examples and helpers in `tests/fixtures/opensees/STAIRlabGallery/` come from
commit `fd3958fde8a3d0a350321b3fbdd3f415ee16e2a2` and use the BSD 3-Clause
license. The vendored upstream Tcl files are unchanged. Strut reconstructs
the HexaExtrusion mesh with a separate Tcl generator in that directory.
[Pinned source](https://github.com/peer-open-source/opensees-gallery/tree/fd3958fde8a3d0a350321b3fbdd3f415ee16e2a2) ·
[BSD 3-Clause license](tests/fixtures/opensees/STAIRlabGallery/LICENSE)

## xara (OpenSeesRT)

Copyright (c) 2024, Claudio M. Perez. Reference setup fetches an unmodified
checkout of xara at `856acc159ca789d4796cf3234ee499a422781c14` under the BSD
2-Clause license. The dialect upgrade uses its Tcl parsers as a reference.
[Pinned source](https://github.com/peer-open-source/xara/tree/856acc159ca789d4796cf3234ee499a422781c14) ·
[BSD 2-Clause license](https://github.com/peer-open-source/xara/blob/856acc159ca789d4796cf3234ee499a422781c14/LICENSE.txt)

## OpenSees Wiki effective-stress slope model

`scripts/slope_recon` reconstructs the October 26, 2010 revision of
`SlopeAnalysis2Deff.tcl` by Chris McGann and Pedro Arduino, University of
Washington. The template and mesh table reproduce the original published Tcl
byte for byte and preserve its attribution. The published Tcl includes no
separate license notice. [OpenSees Wiki source](https://opensees.berkeley.edu/wiki/index.php/File:SlopeAnalysis2Deff.tcl)

## OpenSees specialised models

The Strut FSAM, SFI_MVLEM, ModIMKPeakOriented, HSConstraint, and CatenaryCable
implementations adapt the corresponding OpenSees 3.7.1 C++ implementations to
Mojo. The OpenSees terms are reproduced in [COPYRIGHT.md](COPYRIGHT.md). The
wall and panel models were implemented in OpenSees by Kristijan Kolozvari,
Kutay Orakcal, and John Wallace. Dimitrios G. Lignos wrote ModIMKPeakOriented,
and P. Ibanez and J. A. Abell wrote CatenaryCable.
[OpenSees 3.7.1 source](https://github.com/OpenSees/OpenSees/tree/v3.7.1/SRC)

The elastic shear section, CBDI, and thermal steel and beam implementations
adapt the corresponding OpenSees 3.7.1 C++ formulations to Mojo under the
same terms.
