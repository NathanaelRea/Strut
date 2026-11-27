# Published OpenSees models beyond the wiki examples

Research for growing the parity suite past the OpenSees wiki Examples page (see [opensees-examples-inventory.md](../opensees-examples-inventory.md)). Done 2026-11-26.

A source is useful when it has complete Tcl input that runs under the pinned OpenSees 3.7.1, a license or terms that allow use, and a page or paper to cite. "Runs" below means the pinned binary finished the script with no Tcl error.

## Summary

| Source | What it has | License | Usable now |
| --- | --- | --- | --- |
| [OpenSees repo `EXAMPLES/`](https://github.com/OpenSees/OpenSees/tree/v3.7.1/EXAMPLES) (pinned v3.7.1, already in `docs/agent-reference/OpenSees/`) | 156 Tcl scripts: `verification/` (17, closed-form answers), `ExampleScripts/` (46), `ExamplesForTesting/` (OpenSees output for regression), `MaterialModels/`, `DeveloperTestScripts/` | OpenSees license (as the rest of the repo) | Yes. Highest priority: pinned, same license as the existing fixtures, and verification answers are analytical |
| [STAIRlab OpenSees gallery](https://gallery.stairlab.io/) ([peer-open-source/opensees-gallery](https://github.com/peer-open-source/opensees-gallery) at `fd3958f`) | 418 Tcl files: `benchmarks/` with `references.bib` per case, `validation/`, ~70 examples | BSD-3-Clause | Partly. 201 files run clean on 3.7.1; many others target STAIRlab's OpenSeesRT (`verify`, `pragma`, `ForceFrame`, `PrismFrame`) or are helper files |
| [Old user manual: Simulating a Centrifuge Test](https://opensees.berkeley.edu/OpenSees/manuals/usermanual/4188.htm), A. Vytiniotis | `CentrifugeExample.tcl` (678 KB, GiD-generated, sha256 `5e08e9cc...a8b2`): 3,932 quad/quadUP elements, PIMY/PDMY02, laminar box, staged gravity then shaking | OpenSees site | Yes, with a step cap. The header says the full run takes 38.6 h |
| [MVLEM wiki page](https://opensees.berkeley.edu/wiki/index.php/MVLEM_-_Multiple-Vertical-Line-Element-Model_for_RC_Walls) | [Example_1._MVLEM.zip](https://opensees.berkeley.edu/wiki/images/d/d0/Example_1._MVLEM.zip): wall RW2 (Thomsen and Wallace 1995), cyclic displacement control | OpenSees site | Yes. Needs `MVLEM` element and `ConcreteCM`/`SteelMPF` |
| [SFI-MVLEM wiki page](https://opensees.berkeley.edu/wiki/index.php/SFI_MVLEM_-_Cyclic_Shear-Flexure_Interaction_Model_for_RC_Walls) | [Example 1](https://opensees.berkeley.edu/wiki/images/f/fd/Example_1-SFI_MVLEM.zip) (specimen RW-A15-P10-S78, Tran and Wallace 2012), [Example 2](https://opensees.berkeley.edu/wiki/images/2/23/Example_2-SFI_MVLEM.zip) (5-story wall-frame, dynamic) | OpenSees site | Yes. Needs `SFI_MVLEM`, `FSAM` nD material. Papers: Kolozvari, Orakcal, Wallace, [ASCE JSE 141(5) 04014135](https://doi.org/10.1061/(ASCE)ST.1943-541X.0001059) and [04014136](https://doi.org/10.1061/(ASCE)ST.1943-541X.0001083); [PEER 2015/12](https://peer.berkeley.edu/sites/default/files/webpeer-2015-12-kristijan_kolozvari_kutay_orakcal_john_wallace.pdf) |
| [PURR: OpenSees Models for C-PSW/CF Archetype Structures](https://purr.purdue.edu/publications/4348/1) | 8-story coupled composite plate shear wall (SpeedCore): static, cyclic, time history; ~20 Tcl files, [archive](https://purr.purdue.edu/publications/4348/serve/1?render=archive). Ground motions not included | CC BY 4.0 | Likely. Cite Broberg and Varma (2023), [doi:10.4231/9H4P-Z894](https://doi.org/10.4231/9H4P-Z894). Not run yet |
| [PEER 2003 report 0316 Tcl code](https://apps.peer.berkeley.edu/publications/peer_reports/reports_2003/0316_appCD/matkl/tcl.html) (Mackie) | Bridge demand-model zips (`sim`, `bridge`, `util`, `post`, `param`) | Public domain, research use | Doubtful. Written for OpenSees 1.3 |
| [DesignSafe PRJ-2847](https://www.designsafe-ci.org/data/browser/public/designsafe.storage.published/PRJ-2847) (controlled rocking braced frame), [PRJ-5763](https://www.designsafe-ci.org/data/browser/public/designsafe.storage.published/PRJ-5763) (SMF columns) | OpenSees simulation scripts | DesignSafe terms | Not checked: needs a DesignSafe login |

## Looked for, not found

- **LEAP liquefaction centrifuge simulations** (LEAP-UCD-2017, LEAP-ASIA-2019, PDMY02): the papers ([Springer chapter](https://link.springer.com/chapter/10.1007/978-3-031-48821-4_20)) describe OpenSees models but publish no input files.
- **UC Davis pile groups in liquefied ground** (Brandenberg et al., [eScholarship](https://escholarship.org/content/qt6z40d9td/qt6z40d9td.pdf)): models are described; the pre-processor was "still under development" and no Tcl is posted.
- **PEER blind prediction contests** ([2010 column](https://peer.berkeley.edu/node/472), [2017 bridge bent](https://peer.berkeley.edu/publications/2020-09)): entrants used OpenSees, but no models are posted.
- **FEMA P695 steel moment frame archetypes**: the [Steel SMRF database](https://journals.sagepub.com/doi/10.1177/8755293020971209) and [NIST TN 2084](https://doi.org/10.6028/NIST.TN.2084) describe the models but do not post Tcl.
- **s3hark** ([V&V](https://nheri-simcenter.github.io/s3hark-Documentation/common/technical_manual/vnv.html)): only GUI JSON, no OpenSees Tcl.
- **[OpenSees/OpenSeesExamples](https://github.com/OpenSees/OpenSeesExamples)**: 4 Tcl files, all copies of wiki Ex1a.

## STAIRlab gallery: what runs on 3.7.1

Every `*.tcl` was run with the pinned binary from its own directory (60 s timeout). Folders with at least one clean entry script and no wiki duplicate:

- Benchmarks: `frame-0002` (cantilever with transverse load, has `references.bib`), `frame-0005` (nonlinear elastic columns with shear), `model-0010`, `solid-0102` (Drucker-Prager triaxial).
- Validation: `Basic` (sdof, mdof, Newmark), `elements` (8), `Frame/Legacy` (AISC25, PortalFrame2d, beam loads, beams and sections), `Frame/Section`, `Plane` (plane strain, quad bending), `Shell` (roof meshes).
- Examples: Chopra-10.4 (static condensation, eigen), Concrete2x2x2D, ConcretePanel, corotational02 (6 cases), FrameCycling, FrameHinges, HexaExtrusion, ModelingDiaphragms2D, NewmarkFamily, Pendulum, PortalFrame, SantiagoDRM (domain reduction method), ShellCircle, ShellTwist, SteelBracedPortal, SteelBuilding2D, SteelFrame2D, thermal examples 1 to 5, triaxial-drucker, Other (11).

Gallery folders that duplicate vendored wiki examples (Ex1 to Ex9, MRF_*, ViscousDamper, ParallelSP1, SoilPyTz, SolidDam, Example1 to 8) are left out. Many gallery `index.md` pages cite the paper or textbook a case comes from, so cases built from them can link both.

## Recommended order

1. **OpenSees `EXAMPLES/verification`**, then `ExampleScripts` and `ExamplesForTesting`. These are already pinned locally, under the same license, and the verification set checks against analytical answers. `ExampleScripts/ARL360.at2` is also the record the wiki's RC Frame Earthquake example needs, so it can replace the planned synthetic motion.
2. **Gallery benchmarks and validation** that run on 3.7.1, pinned as a `git` entry in `reference-data.lock.json` at `fd3958f`.
3. **Centrifuge example** as a capped parity case plus a full-length benchmark.
4. **MVLEM and SFI-MVLEM walls**: new elements, with published experiments behind them.
5. **PURR C-PSW/CF archetype**: run it first; it needs ground motions.
