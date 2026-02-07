#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


def _load_case(path: Path):
    return json.loads(path.read_text())


def _node_dof_index(node_index, dof, ndf):
    return node_index * ndf + (dof - 1)


def _gaussian_elimination(A, b):
    n = len(b)
    # Forward elimination with partial pivoting
    for i in range(n):
        pivot = i
        max_val = abs(A[i][i])
        for r in range(i + 1, n):
            if abs(A[r][i]) > max_val:
                max_val = abs(A[r][i])
                pivot = r
        if max_val == 0.0:
            raise ValueError("singular matrix")
        if pivot != i:
            A[i], A[pivot] = A[pivot], A[i]
            b[i], b[pivot] = b[pivot], b[i]

        piv = A[i][i]
        for j in range(i, n):
            A[i][j] /= piv
        b[i] /= piv

        for r in range(i + 1, n):
            factor = A[r][i]
            if factor == 0.0:
                continue
            for c in range(i, n):
                A[r][c] -= factor * A[i][c]
            b[r] -= factor * b[i]

    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = b[i]
        for j in range(i + 1, n):
            s -= A[i][j] * x[j]
        x[i] = s
    return x


def _beam_local_stiffness(E, A, I, L):
    EA_L = E * A / L
    EI = E * I
    L2 = L * L
    L3 = L2 * L
    k = [[0.0 for _ in range(6)] for _ in range(6)]
    k[0][0] = EA_L
    k[0][3] = -EA_L
    k[3][0] = -EA_L
    k[3][3] = EA_L

    k[1][1] = 12 * EI / L3
    k[1][2] = 6 * EI / L2
    k[1][4] = -12 * EI / L3
    k[1][5] = 6 * EI / L2

    k[2][1] = 6 * EI / L2
    k[2][2] = 4 * EI / L
    k[2][4] = -6 * EI / L2
    k[2][5] = 2 * EI / L

    k[4][1] = -12 * EI / L3
    k[4][2] = -6 * EI / L2
    k[4][4] = 12 * EI / L3
    k[4][5] = -6 * EI / L2

    k[5][1] = 6 * EI / L2
    k[5][2] = 2 * EI / L
    k[5][4] = -6 * EI / L2
    k[5][5] = 4 * EI / L

    return k


def _matmul(A, B):
    rows = len(A)
    cols = len(B[0])
    inner = len(B)
    out = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for k in range(inner):
            aik = A[i][k]
            if aik == 0.0:
                continue
            for j in range(cols):
                out[i][j] += aik * B[k][j]
    return out


def _transpose(A):
    return [list(row) for row in zip(*A)]


def _beam_global_stiffness(E, A, I, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    L = math.hypot(dx, dy)
    if L == 0.0:
        raise ValueError("zero-length element")
    c = dx / L
    s = dy / L
    k_local = _beam_local_stiffness(E, A, I, L)

    T = [
        [c, s, 0.0, 0.0, 0.0, 0.0],
        [-s, c, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, c, s, 0.0],
        [0.0, 0.0, 0.0, -s, c, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]

    return _matmul(_transpose(T), _matmul(k_local, T))


def run_case(case_data, output_dir: Path):
    model = case_data["model"]
    ndm = model["ndm"]
    ndf = model["ndf"]
    if ndm != 2 or ndf != 3:
        raise ValueError("only 2D frame (ndm=2, ndf=3) supported in phase 1")

    nodes = case_data["nodes"]
    node_ids = [n["id"] for n in nodes]
    id_to_index = {nid: i for i, nid in enumerate(node_ids)}

    sections = {sec["id"]: sec for sec in case_data["sections"]}
    elements = case_data["elements"]

    total_dofs = len(nodes) * ndf
    K = [[0.0 for _ in range(total_dofs)] for _ in range(total_dofs)]
    F = [0.0 for _ in range(total_dofs)]

    for elem in elements:
        if elem["type"] != "elasticBeamColumn2d":
            raise ValueError(f"unsupported element type: {elem['type']}")
        n1, n2 = elem["nodes"]
        i1 = id_to_index[n1]
        i2 = id_to_index[n2]
        node1 = nodes[i1]
        node2 = nodes[i2]

        sec = sections[elem["section"]]
        params = sec["params"]
        E = params["E"]
        A = params["A"]
        I = params["I"]

        k_global = _beam_global_stiffness(E, A, I, node1["x"], node1["y"], node2["x"], node2["y"])
        dof_map = [
            _node_dof_index(i1, 1, ndf),
            _node_dof_index(i1, 2, ndf),
            _node_dof_index(i1, 3, ndf),
            _node_dof_index(i2, 1, ndf),
            _node_dof_index(i2, 2, ndf),
            _node_dof_index(i2, 3, ndf),
        ]
        for a in range(6):
            Aidx = dof_map[a]
            for b in range(6):
                Bidx = dof_map[b]
                K[Aidx][Bidx] += k_global[a][b]

    for load in case_data.get("loads", []):
        node_id = load["node"]
        dof = load["dof"]
        idx = _node_dof_index(id_to_index[node_id], dof, ndf)
        F[idx] += load["value"]

    constrained = set()
    for node in nodes:
        constraints = node.get("constraints")
        if constraints is None:
            continue
        if all(isinstance(v, bool) for v in constraints):
            for i, fixed in enumerate(constraints, start=1):
                if fixed:
                    constrained.add(_node_dof_index(id_to_index[node["id"]], i, ndf))
        else:
            for dof in constraints:
                constrained.add(_node_dof_index(id_to_index[node["id"]], dof, ndf))

    free = [i for i in range(total_dofs) if i not in constrained]
    if not free:
        raise ValueError("no free dofs")

    K_ff = [[K[i][j] for j in free] for i in free]
    F_f = [F[i] for i in free]

    u_f = _gaussian_elimination(K_ff, F_f)
    u = [0.0 for _ in range(total_dofs)]
    for idx, dof in enumerate(free):
        u[dof] = u_f[idx]

    output_dir.mkdir(parents=True, exist_ok=True)
    recorders = case_data.get("recorders", [])
    for rec in recorders:
        if rec["type"] != "node_displacement":
            raise ValueError(f"unsupported recorder type: {rec['type']}")
        dofs = rec["dofs"]
        output = rec.get("output", "node_disp")
        for node_id in rec["nodes"]:
            i = id_to_index[node_id]
            values = [u[_node_dof_index(i, dof, ndf)] for dof in dofs]
            filename = output_dir / f"{output}_node{node_id}.out"
            with filename.open("w", encoding="utf-8") as f:
                f.write(" ".join(f"{v:.16e}" for v in values))
                f.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    case_data = _load_case(Path(args.input))
    run_case(case_data, Path(args.output))


if __name__ == "__main__":
    main()
