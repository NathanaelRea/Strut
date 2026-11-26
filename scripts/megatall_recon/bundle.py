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
import json
import lzma
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


def build_bundle(out_dir: Path) -> dict[str, int]:
    """Write every Model1 Tcl file into ``out_dir`` without reading the fixture."""
    geom = generated_geometry()
    gen = prop.generate_all(geom, prop.load_props())
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for name, content in prop.format_tcl_files(geom, gen).items():
        (out_dir / name).write_text(content)
        sizes[name] = len(content)
    return sizes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--model-dir", type=Path, default=prop.DEFAULT_MODEL_DIR)
    ap.add_argument(
        "--extract-transf",
        action="store_true",
        help="Regenerate transf.bin from the installed real model",
    )
    args = ap.parse_args()
    if args.extract_transf:
        data = extract_transf(args.model_dir)
        TRANSF_PATH.write_bytes(data)
        print(f"Wrote {TRANSF_PATH} ({len(data):,} bytes)")
    if args.out_dir is not None:
        for name, size in build_bundle(args.out_dir).items():
            print(f"Wrote {args.out_dir / name} ({size:,} bytes)")


if __name__ == "__main__":
    main()
