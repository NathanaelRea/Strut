from collections import List
from os import abort
from python import Python, PythonObject

from elements import (
    beam_global_stiffness,
    beam3d_global_stiffness,
    beam_uniform_load_global,
    link_global_stiffness,
    quad4_plane_stress_stiffness,
    shell4_mindlin_stiffness,
    truss_global_stiffness,
    truss3d_global_stiffness,
)
from linalg import gaussian_elimination
from strut_io import py_len


fn node_dof_index(node_index: Int, dof: Int, ndf: Int) -> Int:
    return node_index * ndf + (dof - 1)


fn require_dof_in_range(dof: Int, ndf: Int, context: String):
    if dof < 1 or dof > ndf:
        abort(context + " dof out of range 1.." + String(ndf))


def _write_speedscope(profile_path: String, frames: String, events: String, total_us: Int):
    var json = String()
    json += "{"
    json += "\"$schema\":\"https://www.speedscope.app/file-format-schema.json\","
    json += "\"shared\":{\"frames\":[" + frames + "]},"
    json += "\"profiles\":[{"
    json += "\"type\":\"evented\","
    json += "\"name\":\"strut\","
    json += "\"unit\":\"microseconds\","
    json += "\"startValue\":0,"
    json += "\"endValue\":" + String(total_us) + ","
    json += "\"events\":[" + events + "]"
    json += "}]}"+ "\n"

    var pathlib = Python.import_module("pathlib")
    var out_path = pathlib.Path(profile_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(PythonObject(json))


def run_case(data: PythonObject, output_path: String, profile_path: String):
    var model = data["model"]
    var ndm = Int(model["ndm"])
    var ndf = Int(model["ndf"])
    var is_2d = ndm == 2 and (ndf == 2 or ndf == 3)
    var is_3d_truss = ndm == 3 and ndf == 3
    var is_3d_shell = ndm == 3 and ndf == 6
    if not is_2d and not is_3d_truss and not is_3d_shell:
        abort("only ndm=2 ndf=2/3 and ndm=3 ndf=3/6 supported")

    var time = Python.import_module("time")
    var t0 = Int(time.perf_counter_ns())

    var nodes = data["nodes"]
    var node_count = py_len(nodes)
    var node_ids: List[Int] = []
    node_ids.resize(node_count, 0)

    for i in range(node_count):
        var node = nodes[i]
        if ndm == 3 and not node.__contains__("z"):
            abort("ndm=3 requires node z coordinate")
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
    var elem_count = py_len(elements)
    var elem_ids: List[Int] = []
    elem_ids.resize(elem_count, 0)
    var elem_id_to_index: List[Int] = []
    elem_id_to_index.resize(10000, -1)
    for i in range(elem_count):
        var elem = elements[i]
        var eid = Int(elem["id"])
        elem_ids[i] = eid
        if eid >= len(elem_id_to_index):
            elem_id_to_index.resize(eid + 1, -1)
        elem_id_to_index[eid] = i

    var total_dofs = node_count * ndf
    var K: List[List[Float64]] = []
    for _ in range(total_dofs):
        var row: List[Float64] = []
        row.resize(total_dofs, 0.0)
        K.append(row^)
    var F: List[Float64] = []
    F.resize(total_dofs, 0.0)

    for e in range(elem_count):
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
        elif elem_type == "elasticBeamColumn3d":
            if ndm != 3 or ndf != 6:
                abort("elasticBeamColumn3d requires ndm=3, ndf=6")
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
            if String(sec["type"]) != "ElasticSection3d":
                abort("elasticBeamColumn3d requires ElasticSection3d")

            var params = sec["params"]
            var E = Float64(params["E"])
            var A = Float64(params["A"])
            var Iz = Float64(params["Iz"])
            var Iy = Float64(params["Iy"])
            var G = Float64(params["G"])
            var J = Float64(params["J"])

            var k_global = beam3d_global_stiffness(
                E,
                A,
                Iy,
                Iz,
                G,
                J,
                Float64(node1["x"]),
                Float64(node1["y"]),
                Float64(node1["z"]),
                Float64(node2["x"]),
                Float64(node2["y"]),
                Float64(node2["z"]),
            )
            var dof_map = [
                node_dof_index(i1, 1, ndf),
                node_dof_index(i1, 2, ndf),
                node_dof_index(i1, 3, ndf),
                node_dof_index(i1, 4, ndf),
                node_dof_index(i1, 5, ndf),
                node_dof_index(i1, 6, ndf),
                node_dof_index(i2, 1, ndf),
                node_dof_index(i2, 2, ndf),
                node_dof_index(i2, 3, ndf),
                node_dof_index(i2, 4, ndf),
                node_dof_index(i2, 5, ndf),
                node_dof_index(i2, 6, ndf),
            ]
            for a in range(12):
                var Aidx = dof_map[a]
                for b in range(12):
                    var Bidx = dof_map[b]
                    K[Aidx][Bidx] += k_global[a][b]
        elif elem_type == "truss":
            if ndf != 2 and ndf != 3:
                abort("truss requires ndf=2 or ndf=3")
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

            if ndf == 2:
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
                var k_global = truss3d_global_stiffness(
                    E,
                    A,
                    Float64(node1["x"]),
                    Float64(node1["y"]),
                    Float64(node1["z"]),
                    Float64(node2["x"]),
                    Float64(node2["y"]),
                    Float64(node2["z"]),
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
        elif elem_type == "zeroLength" or elem_type == "twoNodeLink":
            if ndf != 2:
                abort("zeroLength/twoNodeLink requires ndf=2")
            var n1 = Int(elem["nodes"][0])
            var n2 = Int(elem["nodes"][1])
            var i1 = id_to_index[n1]
            var i2 = id_to_index[n2]

            var elem_mats = elem["materials"]
            var elem_dirs = elem["dirs"]
            if py_len(elem_mats) != py_len(elem_dirs):
                abort("zeroLength/twoNodeLink materials/dirs mismatch")

            var ks: List[Float64] = []
            ks.resize(py_len(elem_mats), 0.0)
            var dirs: List[Int] = []
            dirs.resize(py_len(elem_dirs), 0)

            for m in range(py_len(elem_mats)):
                var mat_id = Int(elem_mats[m])
                var mat: PythonObject = None
                for midx in range(py_len(materials)):
                    var candidate = materials[midx]
                    if Int(candidate["id"]) == mat_id:
                        mat = candidate
                        break
                if mat is None:
                    abort("material not found")
                if String(mat["type"]) != "Elastic":
                    abort("only Elastic uniaxial material supported")
                var params = mat["params"]
                ks[m] = Float64(params["E"])
                dirs[m] = Int(elem_dirs[m])

            var k_global = link_global_stiffness(dirs, ks)
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
        elif elem_type == "fourNodeQuad":
            if ndm != 2 or ndf != 2:
                abort("fourNodeQuad requires ndm=2, ndf=2")
            if String(elem.get("formulation", "PlaneStress")) != "PlaneStress":
                abort("fourNodeQuad only supports PlaneStress formulation")
            var n1 = Int(elem["nodes"][0])
            var n2 = Int(elem["nodes"][1])
            var n3 = Int(elem["nodes"][2])
            var n4 = Int(elem["nodes"][3])
            var i1 = id_to_index[n1]
            var i2 = id_to_index[n2]
            var i3 = id_to_index[n3]
            var i4 = id_to_index[n4]
            var node1 = nodes[i1]
            var node2 = nodes[i2]
            var node3 = nodes[i3]
            var node4 = nodes[i4]

            var mat_id = Int(elem["material"])
            var mat: PythonObject = None
            for midx in range(py_len(materials)):
                var candidate = materials[midx]
                if Int(candidate["id"]) == mat_id:
                    mat = candidate
                    break
            if mat is None:
                abort("material not found")
            if String(mat["type"]) != "ElasticIsotropic":
                abort("fourNodeQuad requires ElasticIsotropic material")
            var params = mat["params"]
            var E = Float64(params["E"])
            var nu = Float64(params["nu"])
            var t = Float64(elem["thickness"])

            var x: List[Float64] = []
            var y: List[Float64] = []
            x.resize(4, 0.0)
            y.resize(4, 0.0)
            x[0] = Float64(node1["x"])
            y[0] = Float64(node1["y"])
            x[1] = Float64(node2["x"])
            y[1] = Float64(node2["y"])
            x[2] = Float64(node3["x"])
            y[2] = Float64(node3["y"])
            x[3] = Float64(node4["x"])
            y[3] = Float64(node4["y"])

            var k_global = quad4_plane_stress_stiffness(E, nu, t, x, y)
            var dof_map = [
                node_dof_index(i1, 1, ndf),
                node_dof_index(i1, 2, ndf),
                node_dof_index(i2, 1, ndf),
                node_dof_index(i2, 2, ndf),
                node_dof_index(i3, 1, ndf),
                node_dof_index(i3, 2, ndf),
                node_dof_index(i4, 1, ndf),
                node_dof_index(i4, 2, ndf),
            ]
            for a in range(8):
                var Aidx = dof_map[a]
                for b in range(8):
                    var Bidx = dof_map[b]
                    K[Aidx][Bidx] += k_global[a][b]
        elif elem_type == "shell":
            if ndm != 3 or ndf != 6:
                abort("shell requires ndm=3, ndf=6")
            var n1 = Int(elem["nodes"][0])
            var n2 = Int(elem["nodes"][1])
            var n3 = Int(elem["nodes"][2])
            var n4 = Int(elem["nodes"][3])
            var i1 = id_to_index[n1]
            var i2 = id_to_index[n2]
            var i3 = id_to_index[n3]
            var i4 = id_to_index[n4]
            var node1 = nodes[i1]
            var node2 = nodes[i2]
            var node3 = nodes[i3]
            var node4 = nodes[i4]

            var sec_id = Int(elem["section"])
            var sec: PythonObject = None
            for sidx in range(py_len(sections)):
                var candidate = sections[sidx]
                if Int(candidate["id"]) == sec_id:
                    sec = candidate
                    break
            if sec is None:
                abort("section not found")
            if String(sec["type"]) != "ElasticMembranePlateSection":
                abort("shell requires ElasticMembranePlateSection")
            var params = sec["params"]
            var E = Float64(params["E"])
            var nu = Float64(params["nu"])
            var h = Float64(params["h"])

            var x: List[Float64] = []
            var y: List[Float64] = []
            var z: List[Float64] = []
            x.resize(4, 0.0)
            y.resize(4, 0.0)
            z.resize(4, 0.0)
            x[0] = Float64(node1["x"])
            y[0] = Float64(node1["y"])
            z[0] = Float64(node1["z"])
            x[1] = Float64(node2["x"])
            y[1] = Float64(node2["y"])
            z[1] = Float64(node2["z"])
            x[2] = Float64(node3["x"])
            y[2] = Float64(node3["y"])
            z[2] = Float64(node3["z"])
            x[3] = Float64(node4["x"])
            y[3] = Float64(node4["y"])
            z[3] = Float64(node4["z"])

            var k_global = shell4_mindlin_stiffness(E, nu, h, x, y, z)
            var dof_map = [
                node_dof_index(i1, 1, ndf),
                node_dof_index(i1, 2, ndf),
                node_dof_index(i1, 3, ndf),
                node_dof_index(i1, 4, ndf),
                node_dof_index(i1, 5, ndf),
                node_dof_index(i1, 6, ndf),
                node_dof_index(i2, 1, ndf),
                node_dof_index(i2, 2, ndf),
                node_dof_index(i2, 3, ndf),
                node_dof_index(i2, 4, ndf),
                node_dof_index(i2, 5, ndf),
                node_dof_index(i2, 6, ndf),
                node_dof_index(i3, 1, ndf),
                node_dof_index(i3, 2, ndf),
                node_dof_index(i3, 3, ndf),
                node_dof_index(i3, 4, ndf),
                node_dof_index(i3, 5, ndf),
                node_dof_index(i3, 6, ndf),
                node_dof_index(i4, 1, ndf),
                node_dof_index(i4, 2, ndf),
                node_dof_index(i4, 3, ndf),
                node_dof_index(i4, 4, ndf),
                node_dof_index(i4, 5, ndf),
                node_dof_index(i4, 6, ndf),
            ]
            for a in range(24):
                var Aidx = dof_map[a]
                for b in range(24):
                    var Bidx = dof_map[b]
                    K[Aidx][Bidx] += k_global[a][b]
        else:
            abort("unsupported element type")

    var element_loads = data.get("element_loads", [])
    for i in range(py_len(element_loads)):
        var load = element_loads[i]
        var elem_id = Int(load["element"])
        if elem_id >= len(elem_id_to_index) or elem_id_to_index[elem_id] < 0:
            abort("element load refers to unknown element")
        var elem = elements[elem_id_to_index[elem_id]]
        var elem_type = String(elem["type"])
        var load_type = String(load["type"])
        if load_type == "beamUniform":
            if elem_type != "elasticBeamColumn2d":
                abort("beamUniform requires elasticBeamColumn2d")
            if ndf != 3:
                abort("beamUniform requires ndf=3")
            var n1 = Int(elem["nodes"][0])
            var n2 = Int(elem["nodes"][1])
            var i1 = id_to_index[n1]
            var i2 = id_to_index[n2]
            var node1 = nodes[i1]
            var node2 = nodes[i2]
            var w = Float64(load["w"])
            var f_global = beam_uniform_load_global(
                Float64(node1["x"]),
                Float64(node1["y"]),
                Float64(node2["x"]),
                Float64(node2["y"]),
                w,
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
                F[dof_map[a]] += f_global[a]
        else:
            abort("unsupported element load type")

    var loads = data.get("loads", [])
    for i in range(py_len(loads)):
        var load = loads[i]
        var node_id = Int(load["node"])
        var dof = Int(load["dof"])
        require_dof_in_range(dof, ndf, "load")
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
            require_dof_in_range(dof, ndf, "constraint")
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

    var t_solve_start = Int(time.perf_counter_ns())
    var u_f = gaussian_elimination(K_ff, F_f)
    var t_solve_end = Int(time.perf_counter_ns())
    var u: List[Float64] = []
    u.resize(total_dofs, 0.0)
    for i in range(len(free)):
        u[free[i]] = u_f[i]

    var t_output_start = t_solve_end
    var t1 = Int(time.perf_counter_ns())
    var analysis_us = (t_output_start - t0) // 1000

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
                require_dof_in_range(dof, ndf, "recorder")
                var value = u[node_dof_index(i, dof, ndf)]
                if j > 0:
                    line += " "
                line += String(value)
            line += "\n"
            var filename = output + "_node" + String(node_id) + ".out"
            var file_path = out_dir.joinpath(filename)
            file_path.write_text(PythonObject(line))

    var t2 = Int(time.perf_counter_ns())
    if profile_path != "":
        var total_us = (t2 - t0) // 1000
        var assemble_end = (t_solve_start - t0) // 1000
        var solve_end = (t_solve_end - t0) // 1000
        var output_end = total_us
        var frames = (
            "{\"name\":\"total\"},"
            "{\"name\":\"assemble\"},"
            "{\"name\":\"solve\"},"
            "{\"name\":\"output\"}"
        )
        var events = (
            "{\"type\":\"O\",\"frame\":0,\"at\":0},"
            "{\"type\":\"O\",\"frame\":1,\"at\":0},"
            "{\"type\":\"C\",\"frame\":1,\"at\":" + String(assemble_end) + "},"
            "{\"type\":\"O\",\"frame\":2,\"at\":" + String(assemble_end) + "},"
            "{\"type\":\"C\",\"frame\":2,\"at\":" + String(solve_end) + "},"
            "{\"type\":\"O\",\"frame\":3,\"at\":" + String(solve_end) + "},"
            "{\"type\":\"C\",\"frame\":3,\"at\":" + String(output_end) + "},"
            "{\"type\":\"C\",\"frame\":0,\"at\":" + String(output_end) + "}"
        )
        _write_speedscope(profile_path, frames, events, total_us)
