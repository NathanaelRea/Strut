from collections import List
from math import hypot
from os import abort
from python import Python, PythonObject
from sys.arg import argv


fn _arg_value(
    args: VariadicList[StringSlice[StaticConstantOrigin]], name: String
) -> String:
    for i in range(len(args) - 1):
        if String(args[i]) == name:
            return String(args[i + 1])
    return ""


fn _py_len(obj: PythonObject) raises -> Int:
    return Int(py=obj.__len__())


fn _node_dof_index(node_index: Int, dof: Int, ndf: Int) -> Int:
    return node_index * ndf + (dof - 1)


fn _beam_local_stiffness(
    E: Float64, A: Float64, I: Float64, L: Float64
) -> List[List[Float64]]:
    var k: List[List[Float64]] = []
    for _ in range(6):
        var row: List[Float64] = []
        row.resize(6, 0.0)
        k.append(row^)

    var EA_L = E * A / L
    var EI = E * I
    var L2 = L * L
    var L3 = L2 * L

    k[0][0] = EA_L
    k[0][3] = -EA_L
    k[3][0] = -EA_L
    k[3][3] = EA_L

    k[1][1] = 12.0 * EI / L3
    k[1][2] = 6.0 * EI / L2
    k[1][4] = -12.0 * EI / L3
    k[1][5] = 6.0 * EI / L2

    k[2][1] = 6.0 * EI / L2
    k[2][2] = 4.0 * EI / L
    k[2][4] = -6.0 * EI / L2
    k[2][5] = 2.0 * EI / L

    k[4][1] = -12.0 * EI / L3
    k[4][2] = -6.0 * EI / L2
    k[4][4] = 12.0 * EI / L3
    k[4][5] = -6.0 * EI / L2

    k[5][1] = 6.0 * EI / L2
    k[5][2] = 2.0 * EI / L
    k[5][4] = -6.0 * EI / L2
    k[5][5] = 4.0 * EI / L

    return k^


fn _transpose(a: List[List[Float64]]) -> List[List[Float64]]:
    var rows = len(a)
    var cols = len(a[0])
    var out: List[List[Float64]] = []
    for _ in range(cols):
        var row: List[Float64] = []
        row.resize(rows, 0.0)
        out.append(row^)
    for i in range(rows):
        for j in range(cols):
            out[j][i] = a[i][j]
    return out^


fn _matmul(a: List[List[Float64]], b: List[List[Float64]]) -> List[List[Float64]]:
    var rows = len(a)
    var cols = len(b[0])
    var inner = len(b)
    var out: List[List[Float64]] = []
    for _ in range(rows):
        var row: List[Float64] = []
        row.resize(cols, 0.0)
        out.append(row^)
    for i in range(rows):
        for k in range(inner):
            var aik = a[i][k]
            if aik == 0.0:
                continue
            for j in range(cols):
                out[i][j] += aik * b[k][j]
    return out^


fn _beam_global_stiffness(
    E: Float64,
    A: Float64,
    I: Float64,
    x1: Float64,
    y1: Float64,
    x2: Float64,
    y2: Float64,
) -> List[List[Float64]]:
    var dx = x2 - x1
    var dy = y2 - y1
    var L = hypot(dx, dy)
    if L == 0.0:
        abort("zero-length element")
    var c = dx / L
    var s = dy / L

    var k_local = _beam_local_stiffness(E, A, I, L)

    var T: List[List[Float64]] = [
        [c, s, 0.0, 0.0, 0.0, 0.0],
        [-s, c, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, c, s, 0.0],
        [0.0, 0.0, 0.0, -s, c, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]

    return _matmul(_transpose(T), _matmul(k_local, T))


fn _truss_global_stiffness(
    E: Float64,
    A: Float64,
    x1: Float64,
    y1: Float64,
    x2: Float64,
    y2: Float64,
) -> List[List[Float64]]:
    var dx = x2 - x1
    var dy = y2 - y1
    var L = hypot(dx, dy)
    if L == 0.0:
        abort("zero-length element")
    var c = dx / L
    var s = dy / L
    var k = E * A / L

    return [
        [k * c * c, k * c * s, -k * c * c, -k * c * s],
        [k * c * s, k * s * s, -k * c * s, -k * s * s],
        [-k * c * c, -k * c * s, k * c * c, k * c * s],
        [-k * c * s, -k * s * s, k * c * s, k * s * s],
    ]


fn _gaussian_elimination(
    mut A: List[List[Float64]], mut b: List[Float64]
) -> List[Float64]:
    var n = len(b)
    for i in range(n):
        var pivot = i
        var max_val = abs(A[i][i])
        for r in range(i + 1, n):
            if abs(A[r][i]) > max_val:
                max_val = abs(A[r][i])
                pivot = r
        if max_val == 0.0:
            abort("singular matrix")
        if pivot != i:
            var tmp = A[i].copy()
            A[i] = A[pivot].copy()
            A[pivot] = tmp
            var tb = b[i]
            b[i] = b[pivot]
            b[pivot] = tb

        var piv = A[i][i]
        for j in range(i, n):
            A[i][j] /= piv
        b[i] /= piv

        for r in range(i + 1, n):
            var factor = A[r][i]
            if factor == 0.0:
                continue
            for c in range(i, n):
                A[r][c] -= factor * A[i][c]
            b[r] -= factor * b[i]

    var x: List[Float64] = []
    x.resize(n, 0.0)
    for i in range(n - 1, -1, -1):
        var s = b[i]
        for j in range(i + 1, n):
            s -= A[i][j] * x[j]
        x[i] = s
    return x^


def main() raises:
    var args = argv()
    var input_path = _arg_value(args, "--input")
    var output_path = _arg_value(args, "--output")
    if input_path == "" or output_path == "":
        abort("usage: strut.mojo -- --input <case.json> --output <dir>")

    var json = Python.import_module("json")
    var pathlib = Python.import_module("pathlib")

    var path_obj = pathlib.Path(input_path)
    var text = path_obj.read_text()
    var data = json.loads(text)

    var model = data["model"]
    var ndm = Int(py=model["ndm"])
    var ndf = Int(py=model["ndf"])
    if ndm != 2 or (ndf != 2 and ndf != 3):
        abort("only ndm=2, ndf=2/3 supported in phase 1")

    var nodes = data["nodes"]
    var node_count = _py_len(nodes)
    var node_ids: List[Int] = []
    node_ids.resize(node_count, 0)

    for i in range(node_count):
        var node = nodes[i]
        node_ids[i] = Int(py=node["id"])

    var id_to_index: List[Int] = []
    id_to_index.resize(10000, -1)
    for i in range(node_count):
        var nid = node_ids[i]
        if nid >= len(id_to_index):
            id_to_index.resize(nid + 1, -1)
        id_to_index[nid] = i

    var sections = data.get("sections", [])
    var materials = data.get("materials", [])
    var elements = data["elements"]

    var total_dofs = node_count * ndf
    var K: List[List[Float64]] = []
    for _ in range(total_dofs):
        var row: List[Float64] = []
        row.resize(total_dofs, 0.0)
        K.append(row^)
    var F: List[Float64] = []
    F.resize(total_dofs, 0.0)

    for e in range(_py_len(elements)):
        var elem = elements[e]
        var elem_type = String(py=elem["type"])
        if elem_type == "elasticBeamColumn2d":
            if ndf != 3:
                abort("elasticBeamColumn2d requires ndf=3")
            var n1 = Int(py=elem["nodes"][0])
            var n2 = Int(py=elem["nodes"][1])
            var i1 = id_to_index[n1]
            var i2 = id_to_index[n2]
            var node1 = nodes[i1]
            var node2 = nodes[i2]

            var sec_id = Int(py=elem["section"])
            var sec: PythonObject = None
            for sidx in range(_py_len(sections)):
                var candidate = sections[sidx]
                if Int(py=candidate["id"]) == sec_id:
                    sec = candidate
                    break
            if sec is None:
                abort("section not found")

            var params = sec["params"]
            var E = Float64(py=params["E"])
            var A = Float64(py=params["A"])
            var I = Float64(py=params["I"])

            var k_global = _beam_global_stiffness(
                E,
                A,
                I,
                Float64(py=node1["x"]),
                Float64(py=node1["y"]),
                Float64(py=node2["x"]),
                Float64(py=node2["y"]),
            )
            var dof_map = [
                _node_dof_index(i1, 1, ndf),
                _node_dof_index(i1, 2, ndf),
                _node_dof_index(i1, 3, ndf),
                _node_dof_index(i2, 1, ndf),
                _node_dof_index(i2, 2, ndf),
                _node_dof_index(i2, 3, ndf),
            ]
            for a in range(6):
                var Aidx = dof_map[a]
                for b in range(6):
                    var Bidx = dof_map[b]
                    K[Aidx][Bidx] += k_global[a][b]
        elif elem_type == "truss":
            if ndf != 2:
                abort("truss requires ndf=2")
            var n1 = Int(py=elem["nodes"][0])
            var n2 = Int(py=elem["nodes"][1])
            var i1 = id_to_index[n1]
            var i2 = id_to_index[n2]
            var node1 = nodes[i1]
            var node2 = nodes[i2]

            var mat_id = Int(py=elem["material"])
            var mat: PythonObject = None
            for midx in range(_py_len(materials)):
                var candidate = materials[midx]
                if Int(py=candidate["id"]) == mat_id:
                    mat = candidate
                    break
            if mat is None:
                abort("material not found")

            var params = mat["params"]
            var E = Float64(py=params["E"])
            var A = Float64(py=elem["area"])

            var k_global = _truss_global_stiffness(
                E,
                A,
                Float64(py=node1["x"]),
                Float64(py=node1["y"]),
                Float64(py=node2["x"]),
                Float64(py=node2["y"]),
            )
            var dof_map = [
                _node_dof_index(i1, 1, ndf),
                _node_dof_index(i1, 2, ndf),
                _node_dof_index(i2, 1, ndf),
                _node_dof_index(i2, 2, ndf),
            ]
            for a in range(4):
                var Aidx = dof_map[a]
                for b in range(4):
                    var Bidx = dof_map[b]
                    K[Aidx][Bidx] += k_global[a][b]
        else:
            abort("unsupported element type")

    var loads = data.get("loads", [])
    for i in range(_py_len(loads)):
        var load = loads[i]
        var node_id = Int(py=load["node"])
        var dof = Int(py=load["dof"])
        var idx = _node_dof_index(id_to_index[node_id], dof, ndf)
        F[idx] += Float64(py=load["value"])

    var constrained: List[Bool] = []
    constrained.resize(total_dofs, False)
    for i in range(node_count):
        var node = nodes[i]
        if not node.__contains__("constraints"):
            continue
        var constraints = node["constraints"]
        for j in range(_py_len(constraints)):
            var dof = Int(py=constraints[j])
            if dof <= 0:
                continue
            var idx = _node_dof_index(i, dof, ndf)
            constrained[idx] = True

    var free: List[Int] = []
    for i in range(total_dofs):
        if not constrained[i]:
            free.append(i)

    if len(free) == 0:
        abort("no free dofs")

    var K_ff: List[List[Float64]] = []
    var F_f: List[Float64] = []
    F_f.resize(len(free), 0.0)
    for i in range(len(free)):
        var row: List[Float64] = []
        row.resize(len(free), 0.0)
        K_ff.append(row^)
        F_f[i] = F[free[i]]

    for i in range(len(free)):
        for j in range(len(free)):
            K_ff[i][j] = K[free[i]][free[j]]

    var u_f = _gaussian_elimination(K_ff, F_f)
    var u: List[Float64] = []
    u.resize(total_dofs, 0.0)
    for i in range(len(free)):
        u[free[i]] = u_f[i]

    var out_dir = pathlib.Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    var recorders = data.get("recorders", [])
    for r in range(_py_len(recorders)):
        var rec = recorders[r]
        if String(py=rec["type"]) != "node_displacement":
            abort("unsupported recorder type")
        var dofs = rec["dofs"]
        var output = String(py=rec.get("output", "node_disp"))
        var nodes_out = rec["nodes"]
        for nidx in range(_py_len(nodes_out)):
            var node_id = Int(py=nodes_out[nidx])
            var i = id_to_index[node_id]
            var line = String()
            for j in range(_py_len(dofs)):
                var dof = Int(py=dofs[j])
                var value = u[_node_dof_index(i, dof, ndf)]
                if j > 0:
                    line += " "
                line += String(value)
            line += "\n"
            var filename = output + "_node" + String(node_id) + ".out"
            var file_path = out_dir.joinpath(filename)
            file_path.write_text(PythonObject(line))
