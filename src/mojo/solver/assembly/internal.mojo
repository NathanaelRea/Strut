from collections import List
from math import hypot, sqrt
from os import abort
from python import PythonObject

from elements import (
    beam_global_stiffness,
    beam2d_corotational_global_stiffness,
    beam2d_corotational_global_internal_force,
    beam2d_pdelta_global_stiffness,
    beam3d_global_stiffness,
    force_beam_column2d_global_tangent_and_internal,
    link_global_stiffness,
    quad4_plane_stress_stiffness,
    shell4_mindlin_stiffness,
    truss_global_stiffness,
    truss3d_global_stiffness,
)
from materials import (
    UniMaterialDef,
    UniMaterialState,
    uni_mat_initial_tangent,
    uniaxial_set_trial_strain,
)
from solver.banded import banded_add, banded_matrix
from solver.dof import node_dof_index
from sections import FiberCell, FiberSection2dDef
from strut_io import py_len


fn assemble_internal_forces(
    nodes: PythonObject,
    elements: PythonObject,
    sections_by_id: List[PythonObject],
    materials_by_id: List[PythonObject],
    id_to_index: List[Int],
    node_count: Int,
    ndf: Int,
    ndm: Int,
    u: List[Float64],
    uniaxial_defs: List[UniMaterialDef],
    uniaxial_state_defs: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    elem_uniaxial_offsets: List[Int],
    elem_uniaxial_counts: List[Int],
    elem_uniaxial_state_ids: List[Int],
) raises -> List[Float64]:
    var total_dofs = node_count * ndf
    var F_int: List[Float64] = []
    F_int.resize(total_dofs, 0.0)

    var elem_count = py_len(elements)
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
            if sec_id >= len(sections_by_id):
                abort("section not found")
            var sec = sections_by_id[sec_id]
            if sec is None:
                abort("section not found")

            var params = sec["params"]
            var E = Float64(params["E"])
            var A = Float64(params["A"])
            var I = Float64(params["I"])

            var dof_map = [
                node_dof_index(i1, 1, ndf),
                node_dof_index(i1, 2, ndf),
                node_dof_index(i1, 3, ndf),
                node_dof_index(i2, 1, ndf),
                node_dof_index(i2, 2, ndf),
                node_dof_index(i2, 3, ndf),
            ]
            var geom = String(elem.get("geomTransf", "Linear"))
            var k_global: List[List[Float64]] = []
            if geom == "Linear":
                k_global = beam_global_stiffness(
                    E,
                    A,
                    I,
                    Float64(node1["x"]),
                    Float64(node1["y"]),
                    Float64(node2["x"]),
                    Float64(node2["y"]),
                )
            elif geom == "PDelta":
                var u_elem: List[Float64] = []
                u_elem.resize(6, 0.0)
                for i in range(6):
                    u_elem[i] = u[dof_map[i]]
                k_global = beam2d_pdelta_global_stiffness(
                    E,
                    A,
                    I,
                    Float64(node1["x"]),
                    Float64(node1["y"]),
                    Float64(node2["x"]),
                    Float64(node2["y"]),
                    u_elem,
                )
            elif geom == "Corotational":
                var u_elem: List[Float64] = []
                u_elem.resize(6, 0.0)
                for i in range(6):
                    u_elem[i] = u[dof_map[i]]
                k_global = beam2d_corotational_global_stiffness(
                    E,
                    A,
                    I,
                    Float64(node1["x"]),
                    Float64(node1["y"]),
                    Float64(node2["x"]),
                    Float64(node2["y"]),
                    u_elem,
                )
            else:
                abort("unsupported geomTransf: " + geom)
            if geom == "Corotational":
                var u_elem: List[Float64] = []
                u_elem.resize(6, 0.0)
                for i in range(6):
                    u_elem[i] = u[dof_map[i]]
                var f_elem = beam2d_corotational_global_internal_force(
                    E,
                    A,
                    I,
                    Float64(node1["x"]),
                    Float64(node1["y"]),
                    Float64(node2["x"]),
                    Float64(node2["y"]),
                    u_elem,
                )
                for a in range(6):
                    F_int[dof_map[a]] += f_elem[a]
            else:
                for a in range(6):
                    var sum = 0.0
                    for b in range(6):
                        sum += k_global[a][b] * u[dof_map[b]]
                    F_int[dof_map[a]] += sum
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
            if sec_id >= len(sections_by_id):
                abort("section not found")
            var sec = sections_by_id[sec_id]
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
                var sum = 0.0
                for b in range(12):
                    sum += k_global[a][b] * u[dof_map[b]]
                F_int[dof_map[a]] += sum
        elif elem_type == "truss":
            if ndf != 2 and ndf != 3:
                abort("truss requires ndf=2 or ndf=3")
            var n1 = Int(elem["nodes"][0])
            var n2 = Int(elem["nodes"][1])
            var i1 = id_to_index[n1]
            var i2 = id_to_index[n2]
            var node1 = nodes[i1]
            var node2 = nodes[i2]

            var offset = elem_uniaxial_offsets[e]
            var count = elem_uniaxial_counts[e]
            if count != 1:
                abort("truss requires one uniaxial material")
            var state_index = elem_uniaxial_state_ids[offset]
            var def_index = uniaxial_state_defs[state_index]
            var mat_def = uniaxial_defs[def_index]
            var state = uniaxial_states[state_index]
            var A = Float64(elem["area"])

            if ndf == 2:
                var dx = Float64(node2["x"]) - Float64(node1["x"])
                var dy = Float64(node2["y"]) - Float64(node1["y"])
                var L = hypot(dx, dy)
                if L == 0.0:
                    abort("zero-length element")
                var c = dx / L
                var s = dy / L
                var dof_map = [
                    node_dof_index(i1, 1, ndf),
                    node_dof_index(i1, 2, ndf),
                    node_dof_index(i2, 1, ndf),
                    node_dof_index(i2, 2, ndf),
                ]
                var du = (u[dof_map[2]] - u[dof_map[0]]) * c + (
                    u[dof_map[3]] - u[dof_map[1]]
                ) * s
                var eps = du / L
                uniaxial_set_trial_strain(mat_def, state, eps)
                uniaxial_states[state_index] = state
                var N = state.sig_t * A
                F_int[dof_map[0]] -= N * c
                F_int[dof_map[1]] -= N * s
                F_int[dof_map[2]] += N * c
                F_int[dof_map[3]] += N * s
            else:
                var dx = Float64(node2["x"]) - Float64(node1["x"])
                var dy = Float64(node2["y"]) - Float64(node1["y"])
                var dz = Float64(node2["z"]) - Float64(node1["z"])
                var L = sqrt(dx * dx + dy * dy + dz * dz)
                if L == 0.0:
                    abort("zero-length element")
                var l = dx / L
                var m = dy / L
                var n = dz / L
                var dof_map = [
                    node_dof_index(i1, 1, ndf),
                    node_dof_index(i1, 2, ndf),
                    node_dof_index(i1, 3, ndf),
                    node_dof_index(i2, 1, ndf),
                    node_dof_index(i2, 2, ndf),
                    node_dof_index(i2, 3, ndf),
                ]
                var du = (u[dof_map[3]] - u[dof_map[0]]) * l + (
                    u[dof_map[4]] - u[dof_map[1]]
                ) * m + (u[dof_map[5]] - u[dof_map[2]]) * n
                var eps = du / L
                uniaxial_set_trial_strain(mat_def, state, eps)
                uniaxial_states[state_index] = state
                var N = state.sig_t * A
                F_int[dof_map[0]] -= N * l
                F_int[dof_map[1]] -= N * m
                F_int[dof_map[2]] -= N * n
                F_int[dof_map[3]] += N * l
                F_int[dof_map[4]] += N * m
                F_int[dof_map[5]] += N * n
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
            var dof_map = [
                node_dof_index(i1, 1, ndf),
                node_dof_index(i1, 2, ndf),
                node_dof_index(i2, 1, ndf),
                node_dof_index(i2, 2, ndf),
            ]
            var offset = elem_uniaxial_offsets[e]
            var count = elem_uniaxial_counts[e]
            if count != py_len(elem_mats):
                abort("zeroLength/twoNodeLink material count mismatch")

            for m in range(count):
                var state_index = elem_uniaxial_state_ids[offset + m]
                var def_index = uniaxial_state_defs[state_index]
                var mat_def = uniaxial_defs[def_index]
                var state = uniaxial_states[state_index]
                var dir = Int(elem_dirs[m])
                var a = -1
                var b = -1
                if dir == 1:
                    a = 0
                    b = 2
                elif dir == 2:
                    a = 1
                    b = 3
                else:
                    abort("unsupported link dir")
                var delta = u[dof_map[b]] - u[dof_map[a]]
                uniaxial_set_trial_strain(mat_def, state, delta)
                uniaxial_states[state_index] = state
                var force = state.sig_t
                F_int[dof_map[a]] -= force
                F_int[dof_map[b]] += force
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
            if mat_id >= len(materials_by_id):
                abort("material not found")
            var mat = materials_by_id[mat_id]
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
                var sum = 0.0
                for b in range(8):
                    sum += k_global[a][b] * u[dof_map[b]]
                F_int[dof_map[a]] += sum
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
            if sec_id >= len(sections_by_id):
                abort("section not found")
            var sec = sections_by_id[sec_id]
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
                var sum = 0.0
                for b in range(24):
                    sum += k_global[a][b] * u[dof_map[b]]
                F_int[dof_map[a]] += sum
        else:
            abort("unsupported element type")

    return F_int^


