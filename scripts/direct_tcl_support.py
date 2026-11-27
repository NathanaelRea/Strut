from __future__ import annotations

import json
from math import isfinite
import os
import re
import shutil
from pathlib import Path
from typing import Sequence


_RC_FRAME_EARTHQUAKE_PATH = (
    "OpenSeesExamplesBasic",
    "reinforced_concrete_frame_earthquake_analysis",
    "RCFrameEarthquake.tcl",
)


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def read_tcl_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def load_direct_tcl_manifest(manifest_path: Path) -> dict:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"invalid direct Tcl manifest: {manifest_path}")
    return data


def direct_tcl_manifest_paths(repo_root: Path) -> list[Path]:
    return sorted((repo_root / "tests" / "validation").glob("*/direct_tcl_case.json"))


def resolve_repo_relative_path(raw_path: str | Path, repo_root: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def resolve_entry_tcl_from_manifest(manifest_path: Path, repo_root: Path) -> Path:
    data = load_direct_tcl_manifest(manifest_path)
    raw_path = data.get("entry_tcl")
    if not isinstance(raw_path, str) or not raw_path:
        raise SystemExit(f"direct Tcl manifest missing entry_tcl: {manifest_path}")
    return resolve_repo_relative_path(raw_path, repo_root)


def resolve_direct_tcl_source_files(
    entry_tcl: Path, manifest_path: Path | None = None
) -> list[Path]:
    entry_path = entry_tcl.resolve()
    if manifest_path is None or not manifest_path.exists():
        return [entry_path]

    manifest = load_direct_tcl_manifest(manifest_path)
    raw_source_files = manifest.get("source_files")
    if raw_source_files is None:
        return [entry_path]
    if not isinstance(raw_source_files, list) or not raw_source_files:
        raise SystemExit(
            f"direct Tcl manifest source_files must be a non-empty list: {manifest_path}"
        )

    source_files: list[Path] = []
    for raw_path in raw_source_files:
        if not isinstance(raw_path, str) or not raw_path:
            raise SystemExit(
                f"direct Tcl manifest source_files entries must be non-empty strings: {manifest_path}"
            )
        source_path = Path(raw_path)
        if not source_path.is_absolute():
            source_path = (entry_path.parent / source_path).resolve()
        else:
            source_path = source_path.resolve()
        if not source_path.exists():
            raise SystemExit(
                f"direct Tcl manifest source_files entry does not exist: {manifest_path}: {raw_path}"
            )
        if source_path.suffix.lower() != ".tcl":
            raise SystemExit(
                f"direct Tcl manifest source_files entry is not a Tcl file: {manifest_path}: {raw_path}"
            )
        source_files.append(source_path)

    if source_files[-1] != entry_path:
        raise SystemExit(
            f"direct Tcl manifest source_files must end with entry_tcl: {manifest_path}"
        )

    return source_files


def find_direct_tcl_manifest_for_entry(entry_tcl: Path, repo_root: Path) -> Path | None:
    entry_path = entry_tcl.resolve()
    matches = [
        manifest_path
        for manifest_path in direct_tcl_manifest_paths(repo_root)
        if resolve_entry_tcl_from_manifest(manifest_path, repo_root) == entry_path
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def _should_mirror_direct_tcl_sibling(path: Path) -> bool:
    if path.name.lower() == "data":
        return False
    if path.is_file():
        return True
    if not path.is_dir():
        return False
    for child in path.iterdir():
        if child.is_file() and child.suffix.lower() == ".tcl":
            return False
    return True


def _copy_direct_tcl_shared_assets(
    script_dir: Path, mirrored_parent: Path
) -> list[Path]:
    mirrored_script_dir = mirrored_parent / script_dir.name
    mirrored_inputs: list[Path] = []
    for child in sorted(script_dir.parent.iterdir()):
        if child == script_dir or not _should_mirror_direct_tcl_sibling(child):
            continue
        source = mirrored_parent / child.name
        target = mirrored_script_dir / child.name
        if source.is_file():
            shutil.copy2(source, target)
            mirrored_inputs.append(child.resolve())
            continue
        if not source.is_dir():
            continue
        shutil.copytree(source, target, dirs_exist_ok=True)
        mirrored_inputs.extend(
            sorted(path.resolve() for path in child.rglob("*") if path.is_file())
        )
    return mirrored_inputs


def _ensure_direct_tcl_compat_aliases(
    mirrored_parent: Path, mirrored_script_dir: Path
) -> list[Path]:
    mirrored_inputs: list[Path] = []

    # Some OpenSees bundles source GeneratePeaks.tcl even when only
    # LibGeneratePeaks.tcl is present in the directory.
    lib_generate_peaks = mirrored_script_dir / "LibGeneratePeaks.tcl"
    generate_peaks = mirrored_script_dir / "GeneratePeaks.tcl"
    if lib_generate_peaks.exists() and not generate_peaks.exists():
        shutil.copy2(lib_generate_peaks, generate_peaks)
        mirrored_inputs.append(lib_generate_peaks.resolve())

    canonical_read_smd = mirrored_script_dir / "ReadSMDFile.tcl"
    legacy_read_smd = mirrored_script_dir / "ReadSMDfile.tcl"
    if canonical_read_smd.exists() and not legacy_read_smd.exists():
        shutil.copy2(canonical_read_smd, legacy_read_smd)
        mirrored_inputs.append(canonical_read_smd.resolve())

    # The wiki uploads capitalize the helper filenames used by the pile script.
    if (mirrored_script_dir / "StaticBNWFpile.tcl").is_file():
        for name in (
            "Get_pyParam.tcl",
            "Get_tzParam.tcl",
            "Get_qzParam.tcl",
            "ElasticPileSection.tcl",
        ):
            helper = mirrored_script_dir / name
            alias = helper.with_name(name[0].lower() + name[1:])
            if helper.is_file() and not alias.exists():
                shutil.copy2(helper, alias)
            if alias.is_file():
                mirrored_inputs.append(alias.resolve())

    gm_dirs = [mirrored_parent / "GMfiles", mirrored_script_dir / "GMfiles"]
    seen_dirs: set[Path] = set()
    for gm_dir in gm_dirs:
        gm_dir = gm_dir.resolve()
        if gm_dir in seen_dirs or not gm_dir.is_dir():
            continue
        seen_dirs.add(gm_dir)

        at2_files = sorted(
            path
            for path in gm_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".at2"
        )
        for at2_path in at2_files:
            dt2_path = at2_path.with_suffix(".dt2")
            if not dt2_path.exists():
                shutil.copy2(at2_path, dt2_path)
                mirrored_inputs.append(at2_path.resolve())

        canonical_peer = gm_dir / "H-e12140.at2"
        uppercase_peer = gm_dir / "H-E12140.at2"
        if canonical_peer.exists() and not uppercase_peer.exists():
            shutil.copy2(canonical_peer, uppercase_peer)
            mirrored_inputs.append(canonical_peer.resolve())
        legacy_peer = gm_dir / "H-E01140.at2"
        if canonical_peer.exists() and not legacy_peer.exists():
            shutil.copy2(canonical_peer, legacy_peer)
            mirrored_inputs.append(canonical_peer.resolve())
        if legacy_peer.exists():
            legacy_dt2 = gm_dir / "H-E01140.dt2"
            if not legacy_dt2.exists():
                shutil.copy2(legacy_peer, legacy_dt2)
                mirrored_inputs.append(legacy_peer.resolve())

    return mirrored_inputs


def _patch_direct_tcl_compatibility(
    mirrored_tcl: Path,
    original_tcl: Path,
    recorder_sample_dt: float | None = None,
    transient_step_cap: dict | None = None,
) -> None:
    try:
        text = mirrored_tcl.read_text(encoding="utf-8")
        encoding = "utf-8"
    except UnicodeDecodeError:
        text = mirrored_tcl.read_text(encoding="latin-1")
        encoding = "latin-1"
    original_text = text

    if original_tcl.parent.name == "SFI_MVLEM_FiveStory" and original_tcl.name in {
        "DisplayModel2D.tcl",
        "DisplayPlane.tcl",
    }:
        # The published display helpers may block on stdin in a batch run.
        text = f"proc {original_tcl.stem} {{args}} {{}}\n"
    if "OpenSeesSource" in original_tcl.parts:
        if original_tcl.name == "RCFrame5.tcl":
            # The serial reference executable has no MUMPS backend.
            text = re.sub(
                r"(?m)^([ \t]*)system Mumps\b",
                r"\1if {[catch {system Mumps}]} {system SparseGeneral}",
                text,
            )
        if original_tcl.name == "example1.tcl":
            # FEAP is optional in OpenSees. The fixture defines material 1
            # with the same modulus as all three FEAP elements.
            text = re.sub(
                r"(?m)^element fTruss (\S+ \S+ \S+ \S+) (\S+)$",
                r"if {[catch {element fTruss \1 \2}]} {element truss \1 1}",
                text,
            )
            text = re.sub(r"(?m)^playback [^\n]*$", "", text)
    if original_tcl.parts[-2:] == ("Concrete2x2x2D", "RCFrame5.tcl"):
        text = "file mkdir out\n" + text

    if (
        original_tcl.name == "TimeSeriesTest.tcl"
        and "OpenSeesSource" in original_tcl.parts
    ):
        # The Path sweep uses local files generated by the script. The other
        # series and remote PeerNGA lookup are outside this validation case.
        text = re.sub(r"(?m)^set records \[searchPeerNGA[^\n]*\]\s*\n", "", text)
        text = re.sub(r'(?m)^lappend seriesCommands "(?!Path\s)[^\n]*\n', "", text)

    if original_tcl.parts[-2:] == (
        "excavation_supported_by_cantilevered_sheet_pile_wall",
        "Excavation.tcl",
    ):
        # The legacy command frees its temporary tag-zero parameter while
        # leaving it registered in the domain. Remove the stale entry before
        # the next initial-state command or final wipe.
        text = re.sub(
            r"(?m)^InitialStateAnalysis (on|off)[ \t]*$",
            r"InitialStateAnalysis \1\nremove parameter 0",
            text,
        )
        text = re.sub(
            r"\b(recorder[ \t]+(?:Node|Element))(?=[ \t]+-file\b)",
            r"\1 -precision 16",
            text,
        )

    if original_tcl.parts[-2:] == (
        "one_dimensional_consolidation",
        "1Dconsolidation.tcl",
    ):
        text = re.sub(
            r"(?m)^(recorder Node[^\n]*?)\s+-node all\b",
            r"\1 -nodeRange 1 $totalNumNode",
            text,
        )

    if recorder_sample_dt is not None:

        def sample_recorder(match: re.Match[str]) -> str:
            command, options = match.groups()
            options = re.sub(r"(?<=\s)-dT[ \t]+\S+", "", options, count=1)
            return f"{command} -dT {recorder_sample_dt:g}{options}"

        text = re.sub(
            r"^([ \t]*recorder[ \t]+(?:Node|Element))\b([ \t][^\n]*)",
            sample_recorder,
            text,
            flags=re.M,
        )

    if transient_step_cap is not None:
        variable = transient_step_cap["step_variable"]
        limit = transient_step_cap["max_steps"]
        if not isinstance(variable, str) or not re.fullmatch(r"[a-zA-Z_]\w*", variable):
            raise ValueError("transient_step_cap requires a Tcl step_variable name")
        if type(limit) is not int or limit <= 0:
            raise ValueError("transient_step_cap max_steps must be a positive integer")
        pattern = (
            rf"(?m)^([ \t]*)(?=(?:set[ \t]+\w+[ \t]+\[)?analyze[ \t]+\${variable}\b)"
        )
        text, replaced = re.subn(
            pattern,
            lambda match: (
                f"{match[1]}if {{${variable} > {limit}}} {{set {variable} {limit}}}\n"
                f"{match[1]}"
            ),
            text,
            count=1,
        )
        if not replaced:
            raise ValueError(
                f"transient_step_cap could not find analyze ${variable} in {original_tcl}"
            )

    is_source_example = original_tcl.parent.parent.parts[-2:] == (
        "OpenSeesSource",
        "EXAMPLES",
    )
    if is_source_example or original_tcl.parts[-3:] == _RC_FRAME_EARTHQUAKE_PATH:
        # The pinned reference executable is headless. Full precision keeps
        # discretization and solver comparisons independent of output rounding.
        text = re.sub(
            r"^([ \t]*)(?=(?:recorder (?:plot|display)|(?:prp|vup|vpn|vrp|"
            r"viewWindow|plane|projection|fill|port|display)\b|"
            r"source (?:RCFrameDisplay|RigidFrame3Ddisplay)\.tcl))",
            r"\1# ",
            text,
            flags=re.M,
        )
        text = re.sub(
            r"^([ \t]*recorder (?:Node|Element|EnvelopeNode|EnvelopeElement))\b"
            r"(?![^\n]*-precision\b)",
            r"\1 -precision 16",
            text,
            flags=re.M,
        )
        text = text.replace("\u00a0", " ")
        # -Umfpack is an unrecognized eigen option in 3.7.1 and is ignored.
        text = text.replace("[eigen -Umfpack ", "[eigen ")

        # Sample the same responses the examples inspect with print/nodeDisp.
        static_displacements = {
            "ArcLength01.tcl": ("3 4", "1 2 3"),
            "Example3.1.tcl": ("3 4", "1 2 3"),
            "RCFrame3.tcl": ("3", "1 2 3"),
            "PortalFrame2d.tcl": ("22", "1"),
            "Test.RCFrame3.ops": ("3", "1 2 3"),
        }
        if original_tcl.name in static_displacements:
            nodes, dofs = static_displacements[original_tcl.name]
            text = re.sub(
                r"^([ \t]*)(analyze\s+\d+\b)",
                rf"\1recorder Node -precision 16 -file strut_disp.out -node {nodes} -dof {dofs} disp\n\1\2",
                text,
                count=1,
                flags=re.M,
            )
        if original_tcl.name == "PortalFrame2d.tcl":
            text = text.replace("&massType", "$massType")
            # The first eigen call selects Transformation in OpenSees when
            # no handler was specified, and the static analysis reuses it.
            text = text.replace(
                "set eigenValues [eigen $numEigen]",
                "constraints Transformation\nset eigenValues [eigen $numEigen]",
            )
            text = text.replace(
                "analyze 1\n",
                "recorder Element -precision 16 -file strut_force.out -ele 1 force\n"
                "analyze 1\n",
            )
            # The printed comparisons query live element state. Strut runs
            # the analysis after conversion and compares the recorders above.
            analysis, reports = text.split("# determine PASS/FAILURE", 1)
            text = analysis + "if {[llength [info commands eleResponse]]} {\n"
            text += "# determine PASS/FAILURE" + reports + "\n}\n"
        if original_tcl.name == "DynamicShell.tcl":
            text = text.replace(
                "analysis Transient",
                "recorder Node -precision 16 -file strut_disp.out -time -node $mid -dof 1 2 3 disp\n"
                "analysis Transient",
            )
            # The 400 s horizon is an exact multiple of the 0.01 s step.
            # Choose the last step by its nominal time, avoiding a rounding
            # dependent extra Newmark step at the end of the loop.
            text = text.replace("$tCurrent < $tFinal", "$tCurrent < ($tFinal - 0.005)")
        if original_tcl.name == "PinchedCylinder.tcl":
            # Each harness case runs one domain from the upstream sweep.
            text = (
                text.replace(
                    "foreach shellType {ShellMITC4 ShellDKGQ ShellNLDKGQ}",
                    "foreach shellType [list $strut_shell_type]",
                )
                .replace(
                    "foreach numEle {4 16 32}",
                    "foreach numEle [list $strut_shell_mesh]",
                )
                .replace(
                    "\tanalyze 1",
                    "\trecorder Node -precision 16 -file strut_disp.out -node $tipNode -dof 3 disp"
                    "\n\tanalyze 1",
                )
            )
            # The recorder supplies the observable used by parity. The
            # original multi-domain summary opens its own append-only file.
            text = text.split("\nset tol 5.0", 1)[0] + "\n"
        if original_tcl.name == "Example4.1.tcl":
            text = text.replace("[getPID]", "0")
        if original_tcl.name == "Test.Example3.1.ops":
            text = text.replace(
                "recorder Node -precision 16 nodeGravity.out disp -time -node 3 4 -dof 1 2 3",
                "recorder Node -precision 16 -file nodeGravity.out -time -node 3 4 -dof 1 2 3 disp",
            )
        if original_tcl.name == "MomentCurvature.tcl":
            text = text.replace(
                "# Define constant axial load",
                "recorder Node -precision 16 -file moment.out -node 1 -dof 3 reaction\n"
                "\t# Define constant axial load",
            )
            # Carry the constant axial load into the moment stage. Both
            # commands leave the OpenSees load factor and domain time at zero.
            text = text.replace(
                "\tanalyze 1\n", "\tanalyze 1\n\tloadConst -time 0.0\n", 1
            )
        if original_tcl.name in {"example2.tcl", "example3.tcl"}:
            # The legacy test omits max iterations. Use the same default as StaticAnalysis.
            text = re.sub(
                r"^(test NormUnbalance 1\.0e-6)[ \t]*$",
                r"\1 25",
                text,
                flags=re.M,
            )
            text = re.sub(
                r"^recorder Node -precision 16 (\S+) disp (.+)$",
                r"recorder Node -precision 16 -file \1 \2 disp",
                text,
                flags=re.M,
            )
            text = re.sub(
                r"^for [^\n]*\{playback [^\n]*$", "", text, flags=re.MULTILINE
            )
        if original_tcl.name == "PlanarTruss.tcl":
            # The published verification runs two independent domains and prints
            # active solver responses. A manifest can select a domain and record it.
            linear, nonlinear = text.split(
                'puts "\\n\\n  - NonLinear (Example2.23)"', 1
            )
            shared, linear = linear.split("# create the finite element model", 1)
            linear_model = linear.split("# determine PASS/FAILURE of test", 1)[0]
            nonlinear_model = nonlinear.split("# determine PASS/FAILURE of test", 1)[0]
            recorder = (
                "recorder Node -precision 16 -file disp.out -node 4 -dof 2 disp\n"
                "recorder Element -precision 16 -file force.out -ele 1 2 3 force\n"
            )
            linear_model = linear_model.replace("analyze 1", recorder + "analyze 1", 1)
            nonlinear_model = nonlinear_model.replace(
                "analyze 1", recorder + "analyze 1", 1
            )
            text = (
                "if {[info exists strutPlanarTrussPart]} {\n"
                + shared
                + "set dX [expr $L*tan($alphaRad)]\n"
                + 'if {$strutPlanarTrussPart eq "elastic"} {\n'
                + linear_model
                + '} elseif {$strutPlanarTrussPart eq "elastic_pp"} {\n'
                + nonlinear_model
                + "}\n} else {\n"
                + text
                + "\n}\n"
            )
        if original_tcl.name == "PlanarTruss.Extra.tcl":
            # Preserve the force and displacement checks across the script's wipes.
            text = re.sub(
                r"(?m)^([ \t]*)(analysis Static)$",
                r"\1recorder Node -file strut_disp.out -precision 17 -node 4 -dof 2 disp\n"
                r"\1recorder Element -file strut_axial.out -precision 17 -eleRange 1 3 axialForce\n"
                r"\1\2",
                text,
            )
        if original_tcl.name == "coupledZeroLength.tcl":
            # The archived recorder names nonexistent node 4, but the script prints node 2.
            text = text.replace("-nodes 4 -dof 1 2 disp", "-nodes 2 -dof 1 2 disp")
        if original_tcl.name in {"RCFrame1.tcl", "RCFrame2.tcl"}:
            # The legacy per-node -const flag was removed. These patterns
            # contain only constant loads, so use a constant time series.
            text = text.replace("pattern Plain 1 Linear", "pattern Plain 1 Constant")
            text = re.sub(r"^([ \t]*load [^\n]*) -const$", r"\1", text, flags=re.M)
            text = text.replace(
                "\nanalyze 1\n", "\nanalyze 1\nloadConst -time 1.0\n", 1
            )
        if original_tcl.name in {
            "test-HystereticSmooth.tcl",
            "test-HystereticAsym.tcl",
        }:
            # Material diagnostics remain available in the reference run. The
            # unit-area truss transfers the material stress to the support
            # reaction, which both engines record through all load reversals.
            text = re.sub(
                r"^([ \t]*recorder Element [^\n]* material (?:stress|strain|tangent))$",
                r"if {[llength [info commands eleResponse]]} {\n\1\n}",
                text,
                flags=re.M,
            )

    if original_tcl.name == "circle.tcl" and original_tcl.parent.name == "ShellCircle":
        text = text.replace(
            "analyze 40",
            "recorder Node -precision 16 -file strut_disp.out -time -node 42 -dof 3 5 disp"
            "\nanalyze 40",
        )

    if original_tcl.name in {
        "Ex2a.Canti2D.ElasticElement.Push.tcl",
        "Ex2c.Canti2D.InelasticFiberSection.Push.tcl",
        "Ex3.Canti2D.analyze.Static.Push.tcl",
        "Ex4.Portal2D.analyze.Static.Push.tcl",
        "Ex4.Portal2D.analyze.Static.Cycle.tcl",
        "Ex5.Frame2D.analyze.Static.Push.tcl",
        "Ex5.Frame2D.analyze.Static.Cycle.tcl",
    }:
        # OpenSees 3.7.1 uses the supplied vector length. These 2D examples
        # have three DOFs per node and pad the lateral load with three zeros.
        text = re.sub(
            r"^([ \t]*load[ \t]+\S+[ \t]+\S+[ \t]+0\.0[ \t]+0\.0)"
            r"(?:[ \t]+0\.0){3}(?=[ \t]*(?:;|$))",
            r"\1",
            text,
            flags=re.MULTILINE,
        )

    if original_tcl.name in {
        "Ex6.genericFrame2D.analyze.Static.Push.tcl",
        "Ex6.genericFrame2D.analyze.Static.Cycle.tcl",
    }:
        # The plot recorder is unavailable in the headless reference binary.
        text = re.sub(
            r"^([ \t]*)recorder plot ", r"\1# recorder plot ", text, flags=re.M
        )

    if original_tcl.name in {
        "pushover_concentrated.tcl",
        "pushover_distributed.tcl",
        "MRF_2Story_Concentrated.tcl",
        "MRF_2Story_Concentrated_PanelZone.tcl",
    }:
        text = re.sub(r"(?m)^[ \t]*DisplayModel2D[ \t][^\n]*$", "", text)
        text = re.sub(
            r"(?m)^([ \t]*recorder (?:Node|Element|Drift) )",
            r"\1-precision 17 ",
            text,
        )
        if original_tcl.name.startswith("MRF_2Story_Concentrated"):
            # Use the same pivoting system and DOF numbering as gravity.
            # The implicit ProfileSPD eigen system can fail its factorization
            # and still return eigenvalues that are used to compute damping.
            text = text.replace(
                "set lambdaN [eigen [expr $nEigenJ]]",
                "constraints Plain\nnumberer RCM\nsystem BandGeneral\n"
                "set lambdaN [eigen [expr $nEigenJ]]",
                1,
            )
        if original_tcl.name == "MRF_2Story_Concentrated_PanelZone.tcl":
            # Validate the bundled pushover branch. With a pivoting eigen
            # solve, the reference dynamic branch stops at 5.578 seconds.
            text = text.replace(
                'set analysisType "dynamic"', 'set analysisType "pushover"', 1
            )

    if original_tcl.name == "Ex6.genericFrame2D.build.InelasticFiberWSection.tcl":
        if "set ColSecTag" not in text:
            marker = "# define MATERIAL properties ----------------------------------------\n"
            insert = "set ColSecTag 1\nset BeamSecTag 4\n"
            if marker in text:
                text = text.replace(marker, insert + marker, 1)
            else:
                text = insert + text

    if original_tcl.name in {
        "Ex6.genericFrame2D.build.ElasticSection.tcl",
        "Ex6.genericFrame2D.build.InelasticSection.tcl",
        "Ex6.genericFrame2D.build.InelasticFiberRCSection.tcl",
        "Ex6.genericFrame2D.build.InelasticFiberWSection.tcl",
    }:
        # These 2D builders pass 6-value masses to 3-DOF nodes.
        text = text.replace(
            "mass $nodeID $MassNode 0.0 0.0 0.0 0.0 0.0;",
            "mass $nodeID $MassNode 0.0 0.0;",
        )
        text = text.replace(
            "mass $nodeID $MassNode 0. 0. 0. 0. 0.;",
            "mass $nodeID $MassNode 0. 0.;",
        )

    if original_tcl.name in {
        "Ex9b.build.WSection3D.tcl",
        "Ex9c.build.RCSection.RectUnconfinedSymm3D.tcl",
        "Ex9d.build.RCSection.RectConfinedSymm3D.tcl",
        "Ex9e.build.RCSection.Rect3D.tcl",
        "Ex9f.build.RCSection.Circ3D.tcl",
    }:
        # OpenSees 3.7.1 rejects legacy `section fiberSec` in 3D without -GJ/-torsion,
        # whereas `section Fiber` allows attaching torsion via the following `section Aggregator`.
        text = text.replace("section fiberSec", "section Fiber")

    if original_tcl.name == "Ex8.genericFrame3D.analyze.Dynamic.sine.Uniform.tcl":
        # The initial-tangent retry checks displacement increments and can
        # accept large section force residuals. Use the current tangent in
        # both engines for this example.
        text = re.sub(
            r"^source LibAnalysisDynamicParameters\.tcl;[^\n]*",
            r"\g<0>\nset algorithmTypeDynamic Newton\nalgorithm $algorithmTypeDynamic",
            text,
            count=1,
            flags=re.MULTILINE,
        )
    if original_tcl.name == "ExampleSP1.tcl":
        # ModelSP1.tcl uses np for beam integration points in the same scope.
        text = text.replace("set np  [getNP]", "set numProcesses [getNP]")
        text = text.replace("$count % $np]", "$count % $numProcesses]")
    if original_tcl.parent.name == "MVLEM_RW2":
        if original_tcl.name == "MVLEM_RW2.tcl":
            # Keep both complete cycles at the first published drift level.
            text = re.sub(
                r'^set iDmax "([^" ]+)[^"]*";', r'set iDmax "\1";', text, flags=re.M
            )
            text = text.replace("set Tol 1.0e-5", "set Tol 1.0e-10", 1)
            text = text.replace(
                "recorder Node -file", "recorder Node -precision 16 -file"
            )
            text = text.replace(
                "recorder Element -file", "recorder Element -precision 16 -file"
            )
        elif original_tcl.name == "LibAnalysisStaticParameters.tcl":
            text = text.replace(
                "variable TolStatic 1.e-5", "variable TolStatic 1.e-10", 1
            )
        elif original_tcl.name == "dispControl.tcl":
            # GeneratePeaks includes duplicate positions before each excursion.
            # Record one row for each nonzero displacement increment.
            text = text.replace(
                "set Dincr [expr $D1 - $D0]\n",
                "set Dincr [expr $D1 - $D0]\n\t\t\tif {$Dincr == 0.0} {continue}\n",
                1,
            )

    if (
        original_tcl.name == "RunCW_Planar.tcl"
        and original_tcl.parent.name == "PURR4348"
    ):
        # Allow a test entry to run one published static analysis with the
        # original PG-1B inputs. The default Run behavior stays intact.
        text = text.replace("wipe all;", "wipe;", 1)
        text = text.replace(
            '{lanalysisstep "0.005 0.01 0.02"}}',
            '{lanalysisstep "0.005 0.01 0.02"} {phase all}}',
            1,
        )
        text = text.replace(
            "set startTime [clock seconds]",
            'if {$phase eq "all"} {set startTime [clock seconds]}',
            1,
        )
        text = text.replace(
            'set fields [split $GMfile "-"]',
            'if {$phase ne "all"} {\n'
            "    set dataDirConc data\n"
            "    file mkdir $dataDirConc\n"
            '    if {$phase eq "concrete_monotonic"} {\n'
            "        source Concrete02_MonoCompression.tcl\n"
            "    } else {\n"
            "        source Concrete02_Cyclic.tcl\n"
            "    }\n"
            "    return\n"
            "}\n"
            'set fields [split $GMfile "-"]',
            1,
        )
        text = text.replace(
            "source CW_Geometry-Planar.tcl", "source CW_Geometry_Planar.tcl"
        )

    if original_tcl.parent.name == "PURR4348":
        if original_tcl.name == "Concrete02_Cyclic.tcl":
            # The archived loop goes singular at the start of its third branch.
            # Keep the complete first compression and tension branches.
            text = text.replace(
                "set disploop [list 1 1.1 2 2.2 3 3.3 5 5.5 9 9.9]",
                "set disploop [list 1 1.1]",
                1,
            )
            # Keep one Static analysis object while the displacement
            # increment changes. Rebuilding the Lagrange analysis at each
            # reversal in 3.7.1 restarts its control displacement at zero.
            text = text.replace(
                "         analysis Static\n",
                "         if {$i == 1} {analysis Static}\n",
                1,
            )
        if original_tcl.name in {
            "Concrete02_MonoCompression.tcl",
            "Concrete02_Cyclic.tcl",
        }:
            text = text.replace(
                "recorder Node -file", "recorder Node -precision 16 -file"
            )
            # These are uniaxial material checks. Prevent numerical bending
            # when the compression plateau leaves a near-zero tangent.
            text = text.replace("fix 1 1 1 1", "fix 1 1 1 1\nfix 2 0 1 1", 1)
    if "STAIRlabGallery" in original_tcl.parts:
        if original_tcl.name == "test_BeamsAndSections2D.tcl":
            # OpenSees 3.7.1's default ARPACK backend requires mass.
            text = text.replace(
                "[eigen standard 6]", "[eigen -standard -symmBandLapack 6]"
            )
        # Add file recorders where the upstream checks print displacements.
        # Use full precision for those files so rounding cannot affect parity.
        gallery_recorders = {
            "RCFrame3.tcl": "-node 3 -dof 1 2 3",
            "ZeroLength05.tcl": "-node 4 -dof 1 2 3",
            "Test01-Force-Geom01.tcl": "-nodeRange 2 9 -dof 1 2 3",
            "Test01-Prism-Geom01.tcl": "-nodeRange 2 9 -dof 1 2 3",
            "Test02-Force-Geom01.tcl": "-node 6 -dof 1 2 3",
            "Test03-Force-Geom01.tcl": "-node 9 -dof 1 2 3",
            "Test05-Force-Geom01.tcl": "-node 10 -dof 1 2 3",
            "Test05-Force-Linear.tcl": "-node 10 -dof 1 2 3",
            "Test05-Force-Linear02.tcl": "-node 10 -dof 1 2 3",
            "Test05-Prism-Geom01.tcl": "-node 10 -dof 1 2 3",
            "Test05-Prism-Linear.tcl": "-node 10 -dof 1 2 3",
            "Test06-Force-Geom01.tcl": "-node 10 -dof 1 2 3",
            "Test06-Prism-Geom01.tcl": "-node 10 -dof 1 2 3",
        }
        recorder = gallery_recorders.get(original_tcl.name)
        if recorder:
            text = re.sub(
                r"^([ \t]*)(analysis[ \t]+Static\b)",
                rf"\1recorder Node -file strut_disp.out -precision 17 {recorder} disp\n\1\2",
                text,
                count=1,
                flags=re.MULTILINE,
            )
            # Iteration diagnostics do not affect the model or its recorders.
            text = re.sub(r'^puts ".*\[numIter\].*"\s*$', "", text, flags=re.MULTILINE)
        if original_tcl.name in {"TestElement_Tri31_Patch.tcl", "Tri31.Patch.tcl"}:
            text = re.sub(
                r'^eval "(recorder Node [^"]+)"$', r"\1", text, flags=re.MULTILINE
            )
            text = text.replace(
                "recorder Node -file", "recorder Node -precision 17 -file"
            )
        if original_tcl.name in {
            "ZeroLength01.tcl",
            "ZeroLength02.tcl",
            "ZeroLength03.tcl",
            "SteelFrame1.tcl",
        }:
            # Move the legacy recorder filename and response to the modern option order.
            text = re.sub(
                r"^recorder Node (?:-file )?(\S+) disp (.+)$",
                r"recorder Node -precision 17 -file \1 \2 disp",
                text,
                flags=re.M,
            )
            text = re.sub(
                r"^recorder Element (\d+) ",
                r"recorder Element -precision 17 -ele \1 ",
                text,
                flags=re.M,
            )
        if original_tcl.name == "TestElement_CoupledZeroLength.tcl":
            text = "file mkdir out\n" + text.replace(
                "recorder Node -file", "recorder Node -precision 17 -file"
            )
        if original_tcl.name == "PortalFrame.tcl":
            # Keep the recorder targets and responses in 3.7.1 option order.
            text = re.sub(
                r"^recorder Node -file (\S+) disp (.+)$",
                r"recorder Node -precision 17 -file \1 \2 disp",
                text,
                flags=re.MULTILINE,
            )
            text = re.sub(
                r"^recorder Element (\d+) ",
                r"recorder Element -precision 17 -ele \1 ",
                text,
                flags=re.MULTILINE,
            )

    if original_tcl.name in {"bending_quad9n.tcl", "TestElement_Quad9_Bending.tcl"}:
        # The scripts print the two tip responses after the bending step.
        text = re.sub(
            r"^analyze 1\b",
            "recorder Node -file strut_tip.out -precision 16 -node 2 3 -dof 1 2 disp\n"
            "integrator LoadControl 1.0\nanalyze 1",
            text,
            count=1,
            flags=re.M,
        )
    if original_tcl.name == "QuadBending.tcl":
        # Each manifest runs one independent domain from the upstream sweep.
        text = text.replace(
            "foreach elem {{quad8n 8} {quad9n 9} {quad 4}}",
            "foreach elem [list $strut_quad_bending]",
        ).replace(
            "  analyze 1",
            "  recorder Node -file strut_tip.out -precision 16 -node 2 3 -dof 1 2 disp\n"
            "  integrator LoadControl 1.0\n  analyze 1",
        )
    if original_tcl.name == "Quad3d.tcl":
        if not is_source_example:
            # The gallery mirror has the same GUI commands as the source example.
            text = re.sub(
                r"^([ \t]*)(?=(?:recorder display|(?:prp|vup|viewWindow|display)\b))",
                r"\1# ",
                text,
                flags=re.M,
            )
        text = text.replace(
            "analysis Transient",
            "recorder Node -file strut_tip.out -precision 16 -time -node "
            "[expr {$nodeYdirnStart - 1}] $nodeNum -dof 1 2 3 disp\n"
            "analysis Transient",
        )
    if (
        "STAIRlabGallery" in original_tcl.parts
        and original_tcl.name == "PlaneStrain.tcl"
    ):
        # The parity reader consumes the same recorder columns as plain text.
        text = text.replace("recorder Node -xml", "recorder Node -precision 16 -file")
    if original_tcl.name == "drucker-old.tcl":
        # Path reads Tcl lists. The unbraced legacy spelling passes only the
        # first time and value, so the apparent loading history stays at zero.
        text = text.replace("\nprintB\n", "\n").replace("-file out/", "-file ")
        text = re.sub(
            r"(timeSeries Series \d+ -time) (0 10 100) -values (0 1 [15])",
            r"\1 {\2} -values {\3}",
            text,
        )
        # Both curves have the same factor through t=10, including the
        # script's single t=0.1 load step.
        text = re.sub(
            r"\n}\n\n#--loading object deviator stress\n"
            r"timeSeries Series 2 [^\n]+\npattern Plain 2 2 \{",
            "",
            text,
        )
        text = re.sub(
            r"^(recorder (?:Node|Element)) ", r"\1 -precision 16 ", text, flags=re.M
        )
    if original_tcl.name == "hexa_generated.tcl":
        text = text.replace(
            "analysis Static",
            "recorder Node -file strut_tip.out -precision 16 -nodeRange 12001 12120 "
            "-dof 1 2 3 disp\nanalysis Static",
        )
    if original_tcl.name == "PlanarShearWall.tcl" or original_tcl.parts[-2:] == (
        "model-0010",
        "test.tcl",
    ):
        # Each result belongs to a fresh domain. Retain one unchanged brick
        # model from each source and record the displacement used by its check.
        quads = text.index("foreach eleType {quad ")
        bricks = text.index("foreach eleType {stdBrick SSPbrick}")
        text = text[:quads] + text[bricks:]
        brick_type = "stdBrick" if original_tcl.name == "test.tcl" else "SSPbrick"
        text = text.replace(
            "foreach eleType {stdBrick SSPbrick}", f"foreach eleType {{{brick_type}}}"
        )
        text = re.sub(r"foreach numFloor \{6 3 1\}", "foreach numFloor {1}", text)
        text = re.sub(r"foreach numBay \{ *1 3 6\}", "foreach numBay {1}", text)
        text = re.sub(
            r"^([ \t]*)(analyze 1)\b",
            r"\1recorder Node -file strut_wall_top.out -precision 16 -node $nodeTop -dof 1 disp"
            "\n"
            r"\1\2"
            "\n"
            r"\1return",
            text,
            count=1,
            flags=re.M,
        )

    if text != original_text:
        mirrored_tcl.write_text(text, encoding=encoding)


def _apply_manifest_runtime_setup(
    mirrored_script_dir: Path, manifest_path: Path | None
) -> tuple[list[str], list[str], list[Path]]:
    if manifest_path is None or not manifest_path.exists():
        return [], [], []
    manifest = load_direct_tcl_manifest(manifest_path)
    extra_hash_inputs: list[Path] = []

    raw_aliases = manifest.get("file_aliases")
    if isinstance(raw_aliases, dict):
        for alias_rel, target_rel in sorted(raw_aliases.items()):
            alias_path = mirrored_script_dir / str(alias_rel)
            target_path = mirrored_script_dir / str(target_rel)
            if not target_path.exists():
                raise SystemExit(
                    f"direct Tcl manifest file_aliases target does not exist: "
                    f"{manifest_path}: {target_rel}"
                )
            alias_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target_path, alias_path)

    raw_extra_files = manifest.get("extra_files")
    if isinstance(raw_extra_files, dict):
        for rel_path, content in sorted(raw_extra_files.items()):
            dest = mirrored_script_dir / str(rel_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, list):
                text = "\n".join(str(line) for line in content) + "\n"
            else:
                text = str(content)
            dest.write_text(text, encoding="utf-8")

    raw_motions = manifest.get("synthetic_ground_motions")
    if isinstance(raw_motions, list) and raw_motions:
        import synthetic_ground_motion as sgm

        extra_hash_inputs.append(Path(sgm.__file__).resolve())
        for spec in raw_motions:
            if not isinstance(spec, dict) or not spec.get("path"):
                raise SystemExit(
                    f"direct Tcl manifest synthetic_ground_motions entry requires path: {manifest_path}"
                )
            dest = mirrored_script_dir / str(spec["path"])
            dest.parent.mkdir(parents=True, exist_ok=True)
            param_kwargs = {
                key: value
                for key, value in spec.items()
                if key not in {"path", "format"}
            }
            params = sgm.MotionParams(**param_kwargs)
            sgm.write_motion(dest, params, fmt=spec.get("format"))

    prelude = manifest.get("tcl_prelude") or []
    postlude = manifest.get("tcl_postlude") or []
    if not isinstance(prelude, list) or not isinstance(postlude, list):
        raise SystemExit(
            f"direct Tcl manifest tcl_prelude/tcl_postlude must be lists: {manifest_path}"
        )
    return (
        [str(line) for line in prelude],
        [str(line) for line in postlude],
        extra_hash_inputs,
    )


def prepare_direct_tcl_entry(
    entry_tcl: Path,
    source_files: Sequence[Path],
    mirror_root: Path,
    *,
    manifest_path: Path | None = None,
    excluded_roots: Sequence[Path] = (),
) -> tuple[Path, list[Path]]:
    entry_path = entry_tcl.resolve()
    resolved_sources = [Path(path).resolve() for path in source_files]
    if not resolved_sources:
        raise SystemExit(
            f"direct Tcl entry requires at least one source file: {entry_tcl}"
        )

    script_dir = entry_path.parent
    script_parent = script_dir.parent
    for source_path in resolved_sources:
        try:
            source_path.relative_to(script_parent)
        except ValueError as exc:
            raise SystemExit(
                "direct Tcl source_files entries must stay within the entry Tcl bundle: "
                f"{source_path}"
            ) from exc

    ensure_clean_dir(mirror_root)
    mirrored_parent = mirror_root / script_parent.name
    excluded = [Path(path).resolve() for path in excluded_roots]
    mirror_root_resolved = mirror_root.resolve()
    if mirror_root_resolved not in excluded:
        excluded.append(mirror_root_resolved)

    def _ignore(dirpath: str, names: list[str]) -> set[str]:
        resolved_dir = Path(dirpath).resolve()
        ignored: set[str] = set()
        for excluded_root in excluded:
            if excluded_root.parent == resolved_dir:
                ignored.add(excluded_root.name)
        return ignored

    shutil.copytree(script_parent, mirrored_parent, dirs_exist_ok=True, ignore=_ignore)
    mirrored_script_dir = mirrored_parent / script_dir.name
    shared_asset_inputs = _copy_direct_tcl_shared_assets(script_dir, mirrored_parent)
    compat_alias_inputs = _ensure_direct_tcl_compat_aliases(
        mirrored_parent, mirrored_script_dir
    )
    if entry_path.parts[-3:] == _RC_FRAME_EARTHQUAKE_PATH:
        source_examples = (
            script_parent.parent / "OpenSeesSource/EXAMPLES/ExampleScripts"
        )
        earthquake_inputs = (
            source_examples / "ARL360.at2",
            source_examples / "ReadRecord.tcl",
            script_parent
            / "reinforced_concrete_frame_gravity_analysis/RCFrameGravity.tcl",
        )
        for path in earthquake_inputs:
            shutil.copy2(path, mirrored_script_dir / path.name)
        compat_alias_inputs.extend(earthquake_inputs)
    patch_sources = list(resolved_sources)
    if entry_path.parts[-4:] == (
        "OpenSeesSource",
        "EXAMPLES",
        "ExampleScripts",
        "Example2.1.tcl",
    ):
        patch_sources.append(script_dir / "MomentCurvature.tcl")
    if script_dir.name == "MVLEM_RW2" and entry_path.name == "dispControl.tcl":
        patch_sources.extend(
            script_dir / name
            for name in ("MVLEM_RW2.tcl", "LibAnalysisStaticParameters.tcl")
        )
    if script_dir.name == "SFI_MVLEM_FiveStory" and entry_path.name == "Dynamic.tcl":
        patch_sources.extend(
            script_dir / name for name in ("DisplayModel2D.tcl", "DisplayPlane.tcl")
        )
    purr_concrete_scripts = {
        "concrete_monotonic.tcl": "Concrete02_MonoCompression.tcl",
        "concrete_cyclic.tcl": "Concrete02_Cyclic.tcl",
    }
    if script_dir.name == "PURR4348" and entry_path.name in purr_concrete_scripts:
        patch_sources.append(script_dir / purr_concrete_scripts[entry_path.name])
    recorder_sample_dt = None
    transient_step_cap = None
    tcl_source_version = None
    if manifest_path is not None:
        manifest = load_direct_tcl_manifest(manifest_path)
        transient_step_cap = manifest.get("transient_step_cap")
        if "tcl_source_version" in manifest:
            tcl_source_version = manifest["tcl_source_version"]
            if not isinstance(tcl_source_version, str) or not tcl_source_version:
                raise ValueError(
                    f"tcl_source_version must be a nonempty string in {manifest_path}"
                )
        if "recorder_sample_dt" in manifest:
            recorder_sample_dt = float(manifest["recorder_sample_dt"])
            if not isfinite(recorder_sample_dt) or recorder_sample_dt <= 0.0:
                raise ValueError(
                    f"recorder_sample_dt must be positive in {manifest_path}"
                )
    for source_path in patch_sources:
        relative_path = source_path.relative_to(script_dir)
        _patch_direct_tcl_compatibility(
            mirrored_script_dir / relative_path,
            source_path,
            recorder_sample_dt,
            transient_step_cap if source_path == entry_path else None,
        )
    prelude_lines, postlude_lines, manifest_hash_inputs = _apply_manifest_runtime_setup(
        mirrored_script_dir, manifest_path
    )
    dialect_hash_inputs = []
    if tcl_source_version is not None:
        from direct_tcl_dialects import upgrade_mirrored_tcl

        dialect_hash_inputs = upgrade_mirrored_tcl(
            tcl_source_version, mirrored_script_dir
        )
        # Hash the upstream sources as well as the staged rewrites.
        for staged in list(dialect_hash_inputs):
            source = script_dir / staged.relative_to(mirrored_script_dir)
            if source.is_file() and source not in dialect_hash_inputs:
                dialect_hash_inputs.append(source)

    wrapper_path = mirrored_script_dir / f"__strut_{entry_path.stem}_entry.tcl"
    wrapper_lines = list(prelude_lines)
    for source_path in resolved_sources:
        relative_path = os.path.relpath(source_path, script_dir)
        wrapper_lines.append(f"source {{{relative_path}}}")
    wrapper_lines.extend(postlude_lines)
    wrapper_path.write_text("\n".join(wrapper_lines) + "\n", encoding="utf-8")
    return wrapper_path, [
        *patch_sources,
        *shared_asset_inputs,
        *compat_alias_inputs,
        *manifest_hash_inputs,
        *dialect_hash_inputs,
    ]
