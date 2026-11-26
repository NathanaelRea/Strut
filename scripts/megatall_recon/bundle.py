"""Build the complete Model1 Tcl bundle from the geometry and property tables.

``geometry.generate()`` supplies node coordinates, element connectivity, IDs,
file order and diaphragm sets. ``properties.generate_all()`` supplies
materials, sections, masses, loads and the entry script. This module joins
them and supplies the one piece neither covers: ``geomTransf`` definitions and
each beam's transformation tag, stored in ``transf.bin``.

    uv run python -m scripts.megatall_recon.bundle --out-dir .tmp/megatall-recon
    uv run python -m scripts.megatall_recon.bundle --extract-transf  # needs fixture
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import lzma
import shutil
import tempfile
from pathlib import Path

from scripts.megatall_recon import geometry as geo
from scripts.megatall_recon import properties as prop

TRANSF_PATH = Path(__file__).with_name("transf.bin")
MEGACOL = ("opensees.tcl", "dispBeamColumn")
HBEAM = ("Hbeam.tcl", "dispBeamColumn")
SHELL = ("layershell.tcl", "ShellMITC4")
EBC = ("truss.tcl", "elasticBeamColumn")
TRUSS = ("truss.tcl", "truss")
DIAPHRAGM_PERP_DIRN = 3


def extract_transf(model_dir: Path = prop.DEFAULT_MODEL_DIR) -> bytes:
    """Encode geomTransf definitions and per-beam tags from the real model.

    Tags are stored in Tcl file order for each family, as indices into the
    list of distinct orientation vectors, so the payload stays near 3 KB.
    """
    real = prop.load_real_geometry(model_dir)
    vectors: list[list[float]] = []
    vector_index: dict[tuple[float, float, float], int] = {}
    defs = []
    for tag, kind, vx, vy, vz in real.geom_transfs:
        key = (vx, vy, vz)
        if key not in vector_index:
            vector_index[key] = len(vectors)
            vectors.append(list(key))
        defs.append([tag, kind, vector_index[key]])
    payload = {
        "vectors": vectors,
        "defs": defs,
        "megacol": [row[3] for row in real.megacol_elems],
        "hbeam": [row[3] for row in real.hbeam_elems],
        "ebc": [row[3] for row in real.ebc_elems],
    }
    raw = json.dumps(payload, separators=(",", ":")).encode()
    return lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME)


def load_transf(path: Path = TRANSF_PATH) -> dict:
    data = json.loads(lzma.decompress(path.read_bytes()))
    vectors = data["vectors"]
    data["geom_transfs"] = [
        (tag, kind, *vectors[index]) for tag, kind, index in data["defs"]
    ]
    return data


def to_property_geometry(g: geo.Geometry, transf: dict) -> prop.Geometry:
    """Convert generated geometry into the property generator's input."""

    def ordered(family):
        return [(eid, g.elements[eid][1]) for eid in g.element_order[family]]

    def with_transf(family, tags):
        rows = ordered(family)
        if len(rows) != len(tags):
            raise ValueError(
                f"{family}: {len(rows)} elements but {len(tags)} transf tags"
            )
        return [
            (eid, n[0], n[1], tag) for (eid, n), tag in zip(rows, tags, strict=True)
        ]

    return prop.Geometry(
        nodes=dict(sorted(g.nodes.items())),
        geom_transfs=transf["geom_transfs"],
        megacol_elems=with_transf(MEGACOL, transf["megacol"]),
        ebc_elems=with_transf(EBC, transf["ebc"]),
        truss_elems=[(eid, n[0], n[1]) for eid, n in ordered(TRUSS)],
        hbeam_elems=with_transf(HBEAM, transf["hbeam"]),
        shell_elems=[(eid, *n) for eid, n in ordered(SHELL)],
        diaphragms=[
            (DIAPHRAGM_PERP_DIRN, master, sorted(slaves))
            for master, slaves in sorted(g.diaphragms.items())
        ],
    )


def generated_geometry() -> prop.Geometry:
    return to_property_geometry(geo.generate(), load_transf())


def ground_motion_text() -> str:
    """Synthetic stand-in for the benchmark's ``GMX.txt``.

    Same layout as the original: 4,000 points at dt = 0.01 s, normalized to
    PGA 1.0 (dynamiccpu.tcl scales it by 2.2), one ``%.2E`` value per CRLF
    line. Drawn from the Rezaeian & Der Kiureghian (2010) predictive model for
    a reverse-fault M7 event 20 km away on Vs30 = 600 m/s (inside the
    regression data) with a fixed seed.
    """
    import random

    from scripts import synthetic_ground_motion as sgm

    scenario = sgm.RDKScenario(
        fault="reverse", magnitude=7.0, r_rup_km=20.0, vs30=600.0
    )
    rdk = sgm.rdk2010_sample_params(scenario, random.Random(GM_SCENARIO_SEED))
    values = sgm.generate_acceleration(
        sgm.MotionParams(
            seed=GM_SEED,
            dt=GM_DT,
            npts=GM_NPTS,
            pga=1.0,
            model="rdk2008",
            rdk=rdk,
        )
    )
    return "".join(f"{v:.2E}\r\n" for v in values)


GM_SEED = 1
GM_SCENARIO_SEED = 1
GM_DT = 0.01
GM_NPTS = 4000


def gpu_entry_script(cpu_script: str) -> str:
    """dynamicgpu.tcl differs from dynamiccpu.tcl only in the transient solver."""
    head, sep, tail = cpu_script.rpartition("system SparseSYM")
    if not sep or "pattern UniformExcitation" not in head:
        raise ValueError(
            "dynamiccpu.tcl no longer has a transient SparseSYM system line"
        )
    return head + "system CuSP" + tail


def build_bundle(out_dir: Path) -> dict[str, int]:
    """Write every Model1 file into ``out_dir`` without reading the fixture."""
    geom = generated_geometry()
    gen = prop.generate_all(geom, prop.load_props())
    files = prop.format_tcl_files(geom, gen)
    files["dynamicgpu.tcl"] = gpu_entry_script(files["dynamiccpu.tcl"])
    files["GMX.txt"] = ground_motion_text()
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for name, content in files.items():
        (out_dir / name).write_text(content, newline="")
        sizes[name] = len(content)
    return sizes


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = (
    REPO_ROOT
    / "tests/fixtures/opensees/OpenSeesExamples/MegaTallBuildingBenchmark/Model1"
)
STAMP_NAME = ".strut-generated"


def generator_fingerprint() -> str:
    """Hash of every input that affects the generated bundle."""
    digest = hashlib.sha256()
    here = Path(__file__).parent
    inputs = sorted(
        p
        for p in here.iterdir()
        if p.suffix in {".py", ".json", ".bin"} and p.is_file()
    )
    inputs.append(REPO_ROOT / "scripts" / "synthetic_ground_motion.py")
    for path in inputs:
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def bundle_is_current(target: Path = FIXTURE_DIR) -> bool:
    stamp = target / STAMP_NAME
    return stamp.is_file() and stamp.read_text().strip() == generator_fingerprint()


def ensure_bundle(target: Path = FIXTURE_DIR) -> bool:
    """Generate the bundle into ``target`` unless it is already current.

    Builds into a sibling temp directory and swaps it in, so an interrupted run
    never leaves a half-written model. Returns True if it regenerated.
    """
    if bundle_is_current(target):
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    # Parallel callers (pytest-xdist workers) serialize here; later ones see
    # the stamp and return.
    with open(target.parent / ".generate.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if bundle_is_current(target):
            return False
        _generate_into(target)
    return True


def _generate_into(target: Path) -> None:
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        build_bundle(staging)
        (staging / STAMP_NAME).write_text(generator_fingerprint() + "\n")
        if target.exists():
            shutil.rmtree(target)
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    # Left behind by the old pinned-archive install.
    (target.parent / ".strut-tree-source").unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, help="Write the bundle into this directory")
    ap.add_argument(
        "--install",
        action="store_true",
        help=f"Generate into the test fixture path ({FIXTURE_DIR.relative_to(REPO_ROOT)})",
    )
    ap.add_argument(
        "--model-dir",
        type=Path,
        default=prop.DEFAULT_MODEL_DIR,
        help="A copy of the real Model1 (only for --extract-transf)",
    )
    ap.add_argument(
        "--extract-transf",
        action="store_true",
        help="Regenerate transf.bin from a copy of the real model",
    )
    args = ap.parse_args()
    if args.extract_transf:
        data = extract_transf(prop.require_real_model(args.model_dir))
        TRANSF_PATH.write_bytes(data)
        print(f"Wrote {TRANSF_PATH} ({len(data):,} bytes)")
    if args.out_dir is not None:
        for name, size in build_bundle(args.out_dir).items():
            print(f"Wrote {args.out_dir / name} ({size:,} bytes)")
    if args.install:
        changed = ensure_bundle()
        print(f"{'Generated' if changed else 'Up to date:'} {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
