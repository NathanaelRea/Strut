"""Element mass for the direct-Tcl initial stiffness eigenvalue calculation."""

import numpy as np


def _element_mass(element, ndm, ndf, node_i, node_j):
    dx = node_j["x"] - node_i["x"]
    dy = node_j["y"] - node_i["y"]
    length = np.hypot(dx, dy)
    mass = element["rho"] * length
    if element["type"] in {"truss", "corotTruss"}:
        if not element.get("cMass"):
            return np.eye(2 * ndm) * (mass / 2.0), ndm
        return np.kron([[2.0, 1.0], [1.0, 2.0]], np.eye(ndm)) * (mass / 6.0), ndm
    if element["type"] not in {
        "elasticBeamColumn2d",
        "dispBeamColumn2d",
        "forceBeamColumn2d",
    }:
        raise ValueError(f"eigen does not support mass for {element['type']}")
    if not element.get("cMass") or element["type"] == "forceBeamColumn2d":
        return np.diag([mass / 2.0, mass / 2.0, 0.0] * 2), ndf

    # OpenSees ElasticBeam2d and DispBeamColumn2d use the same local matrix.
    l2 = length * length
    local = np.array(
        [
            [140.0, 0.0, 0.0, 70.0, 0.0, 0.0],
            [0.0, 156.0, 22.0 * length, 0.0, 54.0, -13.0 * length],
            [0.0, 22.0 * length, 4.0 * l2, 0.0, 13.0 * length, -3.0 * l2],
            [70.0, 0.0, 0.0, 140.0, 0.0, 0.0],
            [0.0, 54.0, 13.0 * length, 0.0, 156.0, -22.0 * length],
            [0.0, -13.0 * length, -3.0 * l2, 0.0, -22.0 * length, 4.0 * l2],
        ]
    )
    c, s = dx / length, dy / length
    rotation = np.kron(np.eye(2), [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    return (rotation.T @ local @ rotation) * (mass / 420.0), ndf


def solve_with_element_mass(
    stiffness, nodal_mass, model, nodes, elements, active, free_count, count, largest
):
    ndm, ndf = model["ndm"], model["ndf"]
    node_index = {node: index for index, node in enumerate(sorted(nodes))}
    mass = np.diag(nodal_mass)
    for element in elements:
        if not element.get("rho"):
            continue
        i, j = element["nodes"]
        matrix, width = _element_mass(element, ndm, ndf, nodes[i], nodes[j])
        dofs = [node_index[node] * ndf + dof for node in (i, j) for dof in range(width)]
        mass[np.ix_(dofs, dofs)] += matrix

    transform = np.zeros((len(nodal_mass), free_count))
    for dof, free_index in active:
        transform[dof, free_index] = 1.0
    reduced_mass = transform.T @ mass @ transform
    reduced_stiffness = transform.T @ np.asarray(stiffness) @ transform
    masses, basis = np.linalg.eigh(reduced_mass)
    threshold = np.finfo(float).eps * free_count * np.max(masses)
    inertial = masses > threshold
    if not np.any(inertial):
        raise ValueError("eigen requires at least one positive mass")

    # Eliminate the mass nullspace before solving the symmetric eigenproblem.
    static = basis[:, ~inertial]
    dynamic = basis[:, inertial] / np.sqrt(masses[inertial])
    k_eff = dynamic.T @ reduced_stiffness @ dynamic
    if static.shape[1]:
        coupling = static.T @ reduced_stiffness @ dynamic
        try:
            k_eff -= coupling.T @ np.linalg.solve(
                static.T @ reduced_stiffness @ static, coupling
            )
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"failed to condense massless DOFs: {exc}") from exc

    values = np.linalg.eigvalsh(k_eff)
    if len(values) < count:
        raise ValueError(f"requested {count} modes, found {len(values)} finite modes")
    return (values[-count:] if largest else values[:count]).tolist()
