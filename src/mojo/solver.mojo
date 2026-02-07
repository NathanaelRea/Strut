from collections import List
from os import abort
from python import Python, PythonObject

from elements import beam_global_stiffness, truss_global_stiffness
from linalg import gaussian_elimination
from strut_io import py_len


fn node_dof_index(node_index: Int, dof: Int, ndf: Int) -> Int:
    return node_index * ndf + (dof - 1)


def run_case(data: PythonObject, output_path: String):
    var model = data["model"]
    var ndm = Int(model["ndm"])
    var ndf = Int(model["ndf"])
    if ndm != 2 or (ndf != 2 and ndf != 3):
        abort("only ndm=2, ndf=2/3 supported in phase 1")

    var time = Python.import_module("time")
    var t0 = Int(time.perf_counter_ns())

    var nodes = data["nodes"]
    var node_count = py_len(nodes)
    var node_ids: List[Int] = []
    node_ids.resize(node_count, 0)

    for i in range(node_count):
        var node = nodes[i]
        node_ids[i] = Int(node["id"])

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

    for e in range(py_len(elements)):
        var elem = elements[e]
        var elem_type = String(elem["type"])
        if elem_type == "elasticBeamColumn2d":
            if ndf != 3:
                abort("elasticBeamColumn2d requires ndf=3")
            var n1 = Int(elem["nodes"][0])
            var n2 = Int(elem["nodes"][1])
            var i1 = id_to_index[n1]
            var i2 = id_to_index[n2]
            var node1 = nodes[i1]
            var node2 = nodes[i2]

            var sec_id = Int(elem["section"])
            var sec: PythonObject = None
            for sidx in range(py_len(sections)):
                var candidate = sections[sidx]
                if Int(candidate["id"]) == sec_id:
                    sec = candidate
                    break
            if sec is None:
                abort("section not found")

            var params = sec["params"]
            var E = Float64(params["E"])
            var A = Float64(params["A"])
            var I = Float64(params["I"])

            var k_global = beam_global_stiffness(
                E,
                A,
                I,
                Float64(node1["x"]),
                Float64(node1["y"]),
                Float64(node2["x"]),
                Float64(node2["y"]),
            )
            var dof_map = [
                node_dof_index(i1, 1, ndf),
                node_dof_index(i1, 2, ndf),
                node_dof_index(i1, 3, ndf),
                node_dof_index(i2, 1, ndf),
                node_dof_index(i2, 2, ndf),
                node_dof_index(i2, 3, ndf),
            ]
            for a in range(6):
                var Aidx = dof_map[a]
                for b in range(6):
                    var Bidx = dof_map[b]
                    K[Aidx][Bidx] += k_global[a][b]
        elif elem_type == "truss":
            if ndf != 2:
                abort("truss requires ndf=2")
            var n1 = Int(elem["nodes"][0])
            var n2 = Int(elem["nodes"][1])
            var i1 = id_to_index[n1]
            var i2 = id_to_index[n2]
            var node1 = nodes[i1]
            var node2 = nodes[i2]

            var mat_id = Int(elem["material"])
            var mat: PythonObject = None
            for midx in range(py_len(materials)):
                var candidate = materials[midx]
                if Int(candidate["id"]) == mat_id:
                    mat = candidate
                    break
            if mat is None:
                abort("material not found")

            var params = mat["params"]
            var E = Float64(params["E"])
            var A = Float64(elem["area"])

            var k_global = truss_global_stiffness(
                E,
                A,
                Float64(node1["x"]),
                Float64(node1["y"]),
                Float64(node2["x"]),
                Float64(node2["y"]),
            )
            var dof_map = [
                node_dof_index(i1, 1, ndf),
                node_dof_index(i1, 2, ndf),
                node_dof_index(i2, 1, ndf),
                node_dof_index(i2, 2, ndf),
            ]
            for a in range(4):
                var Aidx = dof_map[a]
                for b in range(4):
                    var Bidx = dof_map[b]
                    K[Aidx][Bidx] += k_global[a][b]
        else:
            abort("unsupported element type")

    var loads = data.get("loads", [])
    for i in range(py_len(loads)):
        var load = loads[i]
        var node_id = Int(load["node"])
        var dof = Int(load["dof"])
        var idx = node_dof_index(id_to_index[node_id], dof, ndf)
        F[idx] += Float64(load["value"])

    var constrained: List[Bool] = []
    constrained.resize(total_dofs, False)
    for i in range(node_count):
        var node = nodes[i]
        if not node.__contains__("constraints"):
            continue
        var constraints = node["constraints"]
        for j in range(py_len(constraints)):
            var dof = Int(constraints[j])
            if dof <= 0:
                continue
            var idx = node_dof_index(i, dof, ndf)
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

    var u_f = gaussian_elimination(K_ff, F_f)
    var u: List[Float64] = []
    u.resize(total_dofs, 0.0)
    for i in range(len(free)):
        u[free[i]] = u_f[i]

    var t1 = Int(time.perf_counter_ns())
    var analysis_us = (t1 - t0) / 1000

    var pathlib = Python.import_module("pathlib")
    var out_dir = pathlib.Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    var analysis_path = out_dir.joinpath("analysis_time_us.txt")
    analysis_path.write_text(PythonObject(String(analysis_us) + "\n"))
    var recorders = data.get("recorders", [])
    for r in range(py_len(recorders)):
        var rec = recorders[r]
        if String(rec["type"]) != "node_displacement":
            abort("unsupported recorder type")
        var dofs = rec["dofs"]
        var output = String(rec.get("output", "node_disp"))
        var nodes_out = rec["nodes"]
        for nidx in range(py_len(nodes_out)):
            var node_id = Int(nodes_out[nidx])
            var i = id_to_index[node_id]
            var line = String()
            for j in range(py_len(dofs)):
                var dof = Int(dofs[j])
                var value = u[node_dof_index(i, dof, ndf)]
                if j > 0:
                    line += " "
                line += String(value)
            line += "\n"
            var filename = output + "_node" + String(node_id) + ".out"
            var file_path = out_dir.joinpath(filename)
            file_path.write_text(PythonObject(line))
