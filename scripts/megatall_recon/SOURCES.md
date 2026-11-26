# Sources and Provenance for the Mega-Tall Building Benchmark (`Model1`)

Every URL, citation, and license note below was fetched and verified directly against the live source pages, DOI registries (Crossref / DataCite / Handle), and the pinned OpenSees `v3.7.1` source tree.

---

## 1. Benchmark Model & OpenSees Distribution

1. **Mega-Tall Building Benchmark Models (OpenSeesWiki)**
   - **Authors:** Yuan Tian, Linlin Xie, and Xinzheng Lu (Department of Civil Engineering, Tsinghua University, Beijing)
   - **Year:** 2016 (page last edited 4 February 2016; `oldid=12158`)
   - **URL:** <https://opensees.berkeley.edu/wiki/index.php/Mega-Tall_Building_Benchmark_Models> (linked from the [OpenSeesWiki Examples](https://opensees.berkeley.edu/wiki/index.php/Examples) index)
   - **Pinned download URL in `tests/fixtures/opensees/reference-data.lock.json`:** <http://opensees.berkeley.edu/OpenSees/examples/Model1.rar> (SHA-256 `12bcbef50bd8a12543f198c90d8412131502cc726f97fd0b7760b023b826493b`; `Model1/CuSPSolver.dll` is excluded at extraction)
   - **What the generator takes from it:** Defines the `Model1` coarse-mesh reference benchmark (27,220 nodes, 7 structural/analysis Tcl files, `ShellMITC4` multi-layer shear walls and coupling beams, concrete modulus adjustment $E_c = 2 f_c / \varepsilon_0$, and `dynamiccpu.tcl` / `dynamicgpu.tcl` entry scripts) that `scripts/megatall_recon/` reconstructs and validates against.

2. **OpenSees models for a 606m Mega-tall Building v0.1 (Tsinghua Project Page)**
   - **Authors:** Yuan Tian, Linlin Xie, and Xinzheng Lu (Department of Civil Engineering, Tsinghua University, Beijing)
   - **Year:** 2016
   - **URL:** <http://www.luxinzheng.net/download/OpenSEES/Mega-tall_Building_Benchmark_OpenSees.htm> (companion archive URL listed on page: <http://www.thu-civil.net/download/OpenSees-Mega-tall-Building.zip>)
   - **What the generator takes from it:** Confirms the 606 m tower height, Model 1 (27,220 nodes) vs. Model 2 (81,572 nodes) mesh schemes, MSC.Marc (`Model1M`/`Model2M`) cross-validation under El-Centro scaled to 220 cm/s² (MCE), and the recorder set (`reaction.txt`, floor drift/displacement recorders, and full-mesh `dispx/y/z.txt`).

3. **OpenSees models for a 606 m Mega-tall Building v0.1 (ResearchGate Dataset)**
   - **Authors:** Xinzheng Lu and Linlin Xie
   - **Year:** 2016 (verified via DataCite DOI registry)
   - **URL:** <https://doi.org/10.13140/RG.2.1.1989.2882> (<https://www.researchgate.net/doi/10.13140/RG.2.1.1989.2882>)
   - **What the generator takes from it:** Archival DOI record for the same v0.1 OpenSees mega-tall benchmark release (`Model1`/`Model2`) used for provenance tracking.

4. **A benchmark model of mega-tall buildings and analysis of its seismic responses** (*超高层建筑结构benchmark模型及其地震反应分析*, *Structural Engineers* / 《结构工程师》, Vol. 31, No. 4, pp. 100–107, in Chinese)
   - **Authors:** Xilin Lu (吕西林), Chun Jiang (姜淳), and Huanjun Jiang (蒋欢军) (Tongji University)
   - **Year:** 2015
   - **URL:** <https://doi.org/10.15935/j.cnki.jggcs.2015.04.015> (handle verified via `doi.org/api/handles/`; cited as `Lu et al., 2015a` on both OpenSeesWiki and `luxinzheng.net`)
   - **What the generator takes from it:** Establishes the original 606.1 m mega-frame / core-tube / outrigger-truss ETABS benchmark structural layout (128 stories, 8 belt-truss/outrigger zones, mega-column layout, and perimeter/core geometry) from which Tsinghua's OpenSees `Model1` was converted.

---

## 2. GPU / CPU Solver Benchmark Context (`dynamicgpu.tcl` / `dynamiccpu.tcl`)

5. **GPU-Powered High-Performance Computing for the Analysis of Large-Scale Structures Based on OpenSees** (*Computing in Civil Engineering 2015*, ASCE, Austin, TX, pp. 411–418)
   - **Authors:** Yuan Tian, Linlin Xie, Zhen Xu, and Xinzheng Lu (Tsinghua University)
   - **Year:** 2015
   - **URL:** <https://doi.org/10.1061/9780784479247.051> (author full-text page: <http://www.luxinzheng.net/publication6/GPU-OpenSees_ASCE_Workshop_2015.htm>)
   - **What the generator takes from it:** Explains why `Model1` ships paired `dynamiccpu.tcl` (`system SparseSYM`) and `dynamicgpu.tcl` (`system CuSP`) scripts—the two files differ only on those two solver lines—and documents the GPU iterative `CuSPSolver` benchmark against CPU `SparseSYM` (9–15× speedup on large multi-layer shell / fiber-beam towers).

6. **Cusp (`CuSPSolver`) — OpenSeesWiki**
   - **Authors:** OpenSeesWiki contributors (last edited 15 May 2014; `oldid=11248`)
   - **Year:** 2014
   - **URL:** <https://opensees.berkeley.edu/wiki/index.php/Cusp>
   - **What the generator takes from it:** Documents the `system CuSP` command syntax (`-rTol`, `-mInt`, `-pre`, `-solver`) and external `CuSPSolver.dll` dependency excluded from `Model1.rar` in `reference-data.lock.json` and swapped out in favor of `dynamiccpu.tcl`.

7. **CuSP matrix solver of CUDA 9.0 version (Tsinghua Download Page)**
   - **Authors:** Xinzheng Lu's research group (Tsinghua University)
   - **Year:** 2017–2018 (CUDA 9.0 update; exact posting date not printed on page)
   - **URL:** <http://www.luxinzheng.net/download/OpenSEES/CuSP_OpenSEES.htm>
   - **What the generator takes from it:** Confirms the Windows 64-bit `CuSPSolver.dll` / OpenSees 2.4.3 runtime context of `dynamicgpu.tcl`, motivating why our generated entry script emits the portable CPU solver (`dynamiccpu.tcl`).

---

## 3. `PlaneStressUserMaterial` (`PSUMAT`) & Multi-Layer Shell Papers

8. **A shear wall element for nonlinear seismic analysis of super-tall buildings using OpenSees** (*Finite Elements in Analysis and Design*, Vol. 98, pp. 14–25)
   - **Authors:** Xinzheng Lu, Linlin Xie, Hong Guan, Yuli Huang, and Xiao Lu
   - **Year:** 2015
   - **URL:** <https://doi.org/10.1016/j.finel.2015.01.006> (author full-text page: <http://www.luxinzheng.net/publication6/Shell_wall_element_OpenSees_FEAD_2015.htm>)
   - **What the generator takes from it:** Primary modeling method (`Lu et al., 2015b`) for `layershell.tcl` (`ShellMITC4` + `LayeredShell` + `PlaneStressUserMaterial` + `PlateFromPlaneStress` + `PlateRebar` + `Steel02`), `dispBeamColumn` fiber sections, `Transformation` constraints, `SparseSYM`/`CuSP` sparse solvers, and the MSC.Marc-to-OpenSees conversion workflow.

9. **A multi-layer shell element for shear walls based on OpenSEES — THUShell**
   - **Authors:** Linlin Xie and Kaiqi Lin; supervised by Xinzheng Lu, Yuli Huang (Arup), and Lieping Ye (Tsinghua University)
   - **Year:** 2015 (references Lu et al., 2015 FEAD)
   - **URL:** <http://www.luxinzheng.net/download/OpenSEES/En_THUShell_OpenSEES.htm>
   - **What the generator takes from it:** Specifies the exact Tcl command signatures and parameter meanings (`nstatevs=40`, `nprops=7`, `fc`, `ft`, `fcu`, `epsc0`, `epscu`, `epstu`, `stc`, `PlateFromPlaneStress`, `PlateRebar` at 0°/90°, and `section LayeredShell`) reproduced in `properties.py` for `layershell.tcl`.

10. **Multi-Layer Shell Element for Shear Walls in OpenSees** (*Computing in Civil and Building Engineering (2014)*, ASCE, pp. 1190–1197)
    - **Authors:** Linlin Xie, Xiao Lu, Xinzheng Lu, Yuli Huang, and Lieping Ye
    - **Year:** 2014
    - **URL:** <https://doi.org/10.1061/9780784413616.148>
    - **What the generator takes from it:** Earlier conference formulation of the OpenSees multi-layer shell shear wall element (`LayeredShell` / `PlaneStressUserMaterial`) used throughout the core wall and coupling beam mesh.

11. **Collapse simulation of reinforced concrete high-rise building induced by extreme earthquakes** (*Earthquake Engineering & Structural Dynamics*, Vol. 42, Issue 5, pp. 705–723)
    - **Authors:** Xiao Lu, Xinzheng Lu, Hong Guan, and Lieping Ye
    - **Year:** 2013
    - **URL:** <https://doi.org/10.1002/eqe.2240> (author page: <http://www.luxinzheng.net/publication5/Collapse_Simulation_RC_High_Rise_Building_Earthquakes_EESD_2013.htm>)
    - **What the generator takes from it:** Cited directly in OpenSees `LayeredShellFiberSection.cpp` as the theoretical basis for the layered shell + fiber beam-column modeling strategy carried over from MSC.Marc to OpenSees.

12. **A High-Performance Quadrilateral Flat Shell Element for Seismic Collapse Simulation of Tall Buildings and Its Implementation in OpenSees** (*Journal of Earthquake Engineering*, Vol. 22, Issue 9, pp. 1662–1682; published online 2017, print 2018)
    - **Authors:** Xinzheng Lu, Yuan Tian, Song Cen, Hong Guan, Linlin Xie, and Lisha Wang
    - **Year:** 2018 (online 2017)
    - **URL:** <https://doi.org/10.1080/13632469.2017.1297269>
    - **What the generator takes from it:** Follow-up Tsinghua shell element (`ShellNLDKGQ`) extending the same `LayeredShell` + `PlaneStressUserMaterial` constitutive stack documented for mega-tall building simulations (`Model1` itself still uses `ShellMITC4`).

13. **OpenSees `v3.7.1` Source Files (`PlaneStressUserMaterial.cpp` & `LayeredShellFiberSection.cpp`)**
    - **Authors:** Yuli Huang and Xinzheng Lu (within OpenSees by Frank McKenna, Gregory L. Fenves, and Filip C. Filippou, UC Berkeley / PEER)
    - **Year:** 2012 (original CVS commits `2012-05-21` and `2012-06-08`; verified in OpenSees `v3.7.1`)
    - **URLs:**
      - <https://raw.githubusercontent.com/OpenSees/OpenSees/v3.7.1/SRC/material/nD/PlaneStressUserMaterial.cpp>
      - <https://raw.githubusercontent.com/OpenSees/OpenSees/v3.7.1/SRC/material/section/LayeredShellFiberSection.cpp>
    - **What the generator takes from it:** Confirms the exact `PlaneStressUserMaterial` and `LayeredShell` argument parsing and documents that the Fortran `PSUMAT` implementation is closed-source (`PSUMAT - NOT DEFINED IN THIS VERSION, SOURCE CODE RESTRICTED` when compiled without the restricted object).

---

## 4. License and Terms of Use

- **OpenSeesWiki content (including the Mega-Tall Building Benchmark Models page):** The [OpenSeesWiki General Disclaimer](https://opensees.berkeley.edu/wiki/index.php?title=OpenSeesWiki:General_disclaimer) states: *"This work is licensed under a [Creative Commons Attribution-NonCommercial 3.0 Unported License](http://creativecommons.org/licenses/by-nc/3.0/deed.en_US)"* and grants *"a limited license to copy anything from this site"* on an as-is basis.
- **`Model1.rar` archive and Tsinghua pages (`luxinzheng.net`, ResearchGate DOI):** No `LICENSE` or `README` file is included inside `Model1.rar`, none of the nine `.tcl`/`.txt` files contain copyright or license headers, the Tsinghua benchmark page (`luxinzheng.net`) states no explicit license, and the DataCite metadata for `10.13140/RG.2.1.1989.2882` has an empty `rightsList`. Consequently, the upstream `Model1` files are **not** committed or downloaded by this repository. `scripts/megatall_recon/` generates the model from compact rules into `tests/fixtures/`; verifying the rules against the original needs a copy supplied by hand (`STRUT_MEGATALL_REAL_DIR`).
- **OpenSees software & `PSUMAT`:** OpenSees's [COPYRIGHT](https://raw.githubusercontent.com/OpenSees/OpenSees/v3.7.1/COPYRIGHT) (`© 1999–2020 The Regents of the University of California`) permits free noncommercial use/redistribution by educational, research, and non-profit entities and internal use by other entities, prohibiting commercial distribution without UC Berkeley Office of Technology Licensing permission. Additionally, `PlaneStressUserMaterial.cpp` notes that the Fortran `PSUMAT` subroutine source code is restricted and omitted from the public OpenSees repository. The OpenSees git history confirms it was never public: commit [`45351742`](https://github.com/OpenSees/OpenSees/commit/45351742) (2013-08-22, "Yuli Huang & Xinzheng Lu - initial import") added only the C++ wrapper, and [`938813b6`](https://github.com/OpenSees/OpenSees/commit/938813b6) the same day replaced the external call with the "SOURCE CODE RESTRICTED" stub unless built with `_HAVE_PSUMAT`. No Fortran `psumat` file appears anywhere in the repository's history. The authors distributed it only inside their Windows builds ([THUShell](http://www.luxinzheng.net/download/OpenSEES/En_THUShell_OpenSEES.htm), `OpenSeesTHU.zip`).

---

## 5. Verification Notes & Items Not Directly Verified

- **Full Chinese text of Lu, Jiang & Jiang (2015), *Structural Engineers* 31(4): 100–107:** The CNKI DOI handle (`10.15935/j.cnki.jggcs.2015.04.015`) resolved via `doi.org/api/handles/`, and the citation is verified on both OpenSeesWiki and `luxinzheng.net`, but fetching the CNKI landing page (`link.oversea.cnki.net`) was blocked by a CAPTCHA (`安全验证`), so the Chinese article body could not be read directly.
- **Tongji University follow-up paper (`Lu XL, Jiang C, Jiang HJ, 2016, Journal of Building Structures 37(9): 1–7`):** Its journal URL (`https://www.jzjgxb.com/EN/10.14006/j.jzjgxb.2016.09.001`) failed TLS certificate verification (`certificate is valid for *.jfjyxzz.org.cn`, not `www.jzjgxb.com`), so it is omitted from the verified source list above.
- **`http://www.thu-civil.net/download/OpenSees-Mega-tall-Building.zip`:** Listed as the download link on `luxinzheng.net`, not fetched directly because `reference-data.lock.json` pins the OpenSeesWiki mirror (`http://opensees.berkeley.edu/OpenSees/examples/Model1.rar`).
- **Exact posting date of `CuSP_OpenSEES.htm`:** No date is printed on the page itself; the CUDA 9.0 release window (2017–2018) is inferred from the linked CUDA 9.0 toolkit dependency.
