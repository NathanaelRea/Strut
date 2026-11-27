"""Emit the mesh, soils, laminar box and pore-pressure boundary conditions."""

from pathlib import Path

from .analysis import gravity_commands, shaking_commands
from .mesh import LOAM_TAGS, SAND_TAGS, laminar_box_nodes, read_mesh

HERE = Path(__file__).parent
LOAM = """# Yolo Loam
nDMaterial PressureIndependMultiYield 1 2 1.37 13000 65000 6. .1
set gravX 0.0
set gravY -9.81
set press 0.0
set press_h 0.0
set press_i 0.0
puts "Start Creation the quad elements"
"""
SAND = """# Dense sand and loose sand, with separate tags for either side of the model
nDMaterial PressureDependMultiYield02 2 2 2.07 130000 260000 36.5 .1 80 0.5 26. 0.013 0.0 0.3 0.0
nDMaterial PressureDependMultiYield02 3 2 2.07 130000 260000 36.5 .1 80 0.5 26. 0.013 0.0 0.3 0.0
nDMaterial PressureDependMultiYield02 4 2 1.98 90000 300000 33. .1 80 0.5 26. 0.067 0.23 0.06 0.27
nDMaterial PressureDependMultiYield02 5 2 1.98 90000 300000 33. .1 80 0.5 26. 0.067 0.23 0.06 0.27
set bulk_f 2.2e6
set n_por 0.333
set bulk [expr $bulk_f/$n_por]
"""
# Faces GiD assigns the alternate hydrostatic-pressure variables.
PRESS_H = {
    *range(722, 727),
    *range(776, 781),
    831,
    832,
    *(tag + offset for tag in range(1734, 1855, 20) for offset in range(2)),
    *(tag + offset for tag in range(1873, 1914, 20) for offset in range(3)),
    *(
        tag + offset
        for tag in range(2018, 2199, 20)
        for offset in range(2 + (tag >= 2158))
    ),
    3793,
    3813,
    3833,
    3834,
    3853,
    3854,
    3873,
    3874,
    3894,
    3895,
    3914,
    3915,
}
PRESS_I = {727, 728, *range(1736, 1917, 20), *range(2017, 2198, 20), 3893, 3913}

# GiD swaps the retained node on these nine points of the loam/sand interface.
SAND_RETAINED = {40, 64, 188, 395, 691, 1092, 1444, 1727, 4145}


def node_line(tag: int, point: tuple[int, int]) -> str:
    x, y = point
    return f"node {tag} {x / 10000:.4f} {y / 10000:.4f}"


def sand_element(tag: int, nodes: tuple[int, ...]) -> str:
    if tag <= 676:
        material = 3
    elif tag <= 1966:
        material = 5
    elif 2217 <= tag <= 2476:
        material = 2
    else:
        material = 4
    conductivity = "0.0003" if tag <= 1966 else "0.0001"
    permeability = f"[expr {conductivity}/9.81/1.]"
    pressure = (
        "$press_h" if tag in PRESS_H else "$press_i" if tag in PRESS_I else "$press"
    )
    return (
        f"element quadUP {tag} {' '.join(map(str, nodes))} 1.0 {material} $bulk 1."
        f" {permeability} {permeability} $gravX $gravY {pressure}"
    )


def pore_pressure_boundaries() -> tuple[list[str], list[int], list[int]]:
    time_series, surface, drains = [], [], []
    for line in (HERE / "pore_pressures.tsv").read_text().splitlines():
        if line.startswith("#"):
            continue
        node, pressure, boundary = line.split()
        time_series.append(f'set Timeseries{node} "Constant -factor {pressure}"')
        (surface if boundary == "surface" else drains).append(int(node))
    return time_series, surface, drains


def pressure_pattern(tag: int, nodes: list[int], motion_offset: int = 0) -> list[str]:
    commands = [f"pattern MultipleSupport {tag} {{"]
    for node in nodes:
        motion = node + motion_offset
        commands.extend(
            [
                f"    groundMotion {motion} Plain -vel $Timeseries{node}",
                f"    imposedMotion {node} 3 {motion}",
            ]
        )
    return [*commands, "}"]


def model_text() -> str:
    points, quads = read_mesh()
    loam_quads = quads[: len(LOAM_TAGS)]
    loam_nodes = {node for quad in loam_quads for node in quad}
    sand_nodes = sorted(points.keys() - loam_nodes)

    commands = [
        "# Reconstructed from Antonios Vytiniotis's OpenSees centrifuge tutorial.",
        "model BasicBuilder -ndm 2 -ndf 2",
        *(node_line(tag, points[tag]) for tag in sorted(loam_nodes)),
        *(node_line(tag, point) for tag, point in laminar_box_nodes()),
        LOAM,
    ]
    for tag, quad in zip(LOAM_TAGS, loam_quads, strict=True):
        commands.append(
            f'element quad {tag} {" ".join(map(str, quad))} 1.0 "PlaneStrain" 1 $press 1.3 $gravX $gravY'
        )
    commands.extend(f"mass {tag} 0.01015 0.0" for tag in range(4205, 4214))
    commands.append("model BasicBuilder -ndm 2 -ndf 3")
    commands.extend(node_line(tag, points[tag]) for tag in sand_nodes)
    commands.append(SAND)
    commands.extend(
        sand_element(tag, quad)
        for tag, quad in zip(SAND_TAGS, quads[len(LOAM_TAGS) :], strict=True)
    )

    # Boundary rings below the first loam node tie directly across the box.
    base_y = 17880
    commands.extend(
        f"fix {tag} 1 1 0" for tag in sand_nodes if points[tag][1] <= base_y
    )
    left = [
        tag for tag in sand_nodes if points[tag][0] <= 35110 and points[tag][1] > base_y
    ][1:]
    right = [
        tag
        for tag in sand_nodes
        if points[tag][0] == 293110 and points[tag][1] > base_y
    ][1:]
    commands.extend(f"equalDOF {tag} {tag + 8} 1 2" for tag in range(4205, 4213))
    commands.extend(f"equalDOF {a} {b} 1 2" for a, b in zip(left, right, strict=True))

    # A 0.01 m gap separates matching sand and loam nodes.
    sand_at = {points[tag]: tag for tag in sand_nodes}
    for loam in sorted(loam_nodes, key=points.__getitem__):
        x, y = points[loam]
        sand = sand_at.get((x, y - 100))
        if sand is not None:
            a, b = (sand, loam) if sand in SAND_RETAINED else (loam, sand)
            commands.append(f"equalDOF {a} {b} 1 2")

    commands += [
        "uniaxialMaterial ENT 8 1000000000",
        "uniaxialMaterial Elastic 9 10",
        "uniaxialMaterial Parallel 7 8 9",
    ]
    for side in range(2):
        for ring in range(8):
            offset = side * 8 + ring
            box, dummy = 4205 + offset, 4221 + offset
            a, b = (box, dummy) if side == 0 else (dummy, box)
            commands.append(
                f"element zeroLength {3933 + offset} {a} {b} -mat 7 7 -dir 1 2"
            )

    commands.extend(
        ["uniaxialMaterial Elastic 21 1e20", "element truss 4124 39 4144 1 21"]
    )
    boundary_loam = (1, 2, 5, 7, 12, 17, 22, 30, *range(4134, 4142))
    commands.extend(
        f"equalDOF {4221 + index} {tag} 1 2" for index, tag in enumerate(boundary_loam)
    )
    commands.extend(
        f"updateMaterialStage -material {tag} -stage 0" for tag in range(1, 7)
    )

    time_series, surface, drains = pore_pressure_boundaries()
    commands += time_series + pressure_pattern(3, surface, 30000)
    commands.extend(
        ['set Timeseries "Constant -factor -1.8"', "pattern Plain 4 $Timeseries {"]
    )
    shoulders = (1561, 1597, 1633, 1665, 1697, 1727, 2303, 2328, 2354, 2382, 2411, 2443)
    factors = (".1", ".2", ".4", ".6", ".8", "1.")
    loads = zip(shoulders, factors + factors[::-1], strict=True)
    loads = sorted([*loads, *((tag, "1.") for tag in surface)])
    commands.extend(f"    load {tag} 0. {factor} 0." for tag, factor in loads)
    commands.append("}")
    commands.extend(f"mass {tag} 0.01015 0 0" for tag in [*left, 925])
    commands.append(gravity_commands())
    commands += pressure_pattern(2, drains)
    commands.append(shaking_commands())
    return "\n".join(commands)
