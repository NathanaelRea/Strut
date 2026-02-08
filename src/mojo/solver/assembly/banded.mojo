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


fn assemble_global_stiffness_banded(
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
    fiber_section_defs: List[FiberSection2dDef],
    fiber_section_cells: List[FiberCell],
    fiber_section_index_by_id: List[Int],
    free_index: List[Int],
    free_count: Int,
    bw: Int,
) raises -> List[List[Float64]]:
    var K = banded_matrix(free_count, bw)

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
            var free_map: List[Int] = []
            free_map.resize(6, -1)
            for i in range(6):
                free_map[i] = free_index[dof_map[i]]

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

            for a in range(6):
                var Aidx = free_map[a]
                if Aidx < 0:
                    continue
                for b in range(6):
                    var Bidx = free_map[b]
                    if Bidx < 0:
                        continue
                    banded_add(K, bw, Aidx, Bidx, k_global[a][b])
        elif elem_type == "forceBeamColumn2d":
            if ndf != 3:
                abort("forceBeamColumn2d requires ndf=3")
            var geom = String(elem.get("geomTransf", "Linear"))
            if geom != "Linear":
                abort("forceBeamColumn2d v1 supports geomTransf Linear only")
            var integration = String(elem.get("integration", "Lobatto"))
            if integration != "Lobatto":
                abort("forceBeamColumn2d v1 supports Lobatto integration only")
            var num_int_pts = Int(elem.get("num_int_pts", 3))
            if num_int_pts != 3:
                abort("forceBeamColumn2d v1 supports num_int_pts=3")

            var n1 = Int(elem["nodes"][0])
            var n2 = Int(elem["nodes"][1])
            var i1 = id_to_index[n1]
            var i2 = id_to_index[n2]
            var node1 = nodes[i1]
            var node2 = nodes[i2]

            var sec_id = Int(elem["section"])
            if sec_id >= len(fiber_section_index_by_id):
                abort("forceBeamColumn2d section not found")
            var sec_index = fiber_section_index_by_id[sec_id]
            if sec_index < 0 or sec_index >= len(fiber_section_defs):
                abort("forceBeamColumn2d requires FiberSection2d")
            var sec_def = fiber_section_defs[sec_index]

            var dof_map = [
                node_dof_index(i1, 1, ndf),
                node_dof_index(i1, 2, ndf),
                node_dof_index(i1, 3, ndf),
                node_dof_index(i2, 1, ndf),
                node_dof_index(i2, 2, ndf),
                node_dof_index(i2, 3, ndf),
            ]
            var free_map: List[Int] = []
            free_map.resize(6, -1)
            for i in range(6):
                free_map[i] = free_index[dof_map[i]]

            var u_elem: List[Float64] = []
            u_elem.resize(6, 0.0)
            for i in range(6):
                u_elem[i] = u[dof_map[i]]

            var elem_offset = elem_uniaxial_offsets[e]
            var elem_state_count = elem_uniaxial_counts[e]
            var k_global: List[List[Float64]] = []
            var f_dummy: List[Float64] = []
            force_beam_column2d_global_tangent_and_internal(
                Float64(node1["x"]),
                Float64(node1["y"]),
                Float64(node2["x"]),
                Float64(node2["y"]),
                u_elem,
                sec_def,
                fiber_section_cells,
                uniaxial_defs,
                uniaxial_states,
                elem_uniaxial_state_ids,
                elem_offset,
                elem_state_count,
                num_int_pts,
                k_global,
                f_dummy,
            )
            for a in range(6):
                var Aidx = free_map[a]
                if Aidx < 0:
                    continue
                for b in range(6):
                    var Bidx = free_map[b]
                    if Bidx < 0:
                        continue
                    banded_add(K, bw, Aidx, Bidx, k_global[a][b])
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
            var free_map: List[Int] = []
            free_map.resize(12, -1)
            for i in range(12):
                free_map[i] = free_index[dof_map[i]]
            for a in range(12):
                var Aidx = free_map[a]
                if Aidx < 0:
                    continue
                for b in range(12):
                    var Bidx = free_map[b]
                    if Bidx < 0:
                        continue
                    banded_add(K, bw, Aidx, Bidx, k_global[a][b])
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
            var E = uni_mat_initial_tangent(mat_def)
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
                var free_map: List[Int] = []
                free_map.resize(len(dof_map), -1)
                for i in range(len(dof_map)):
                    free_map[i] = free_index[dof_map[i]]
                for a in range(len(dof_map)):
                    var Aidx = free_map[a]
                    if Aidx < 0:
                        continue
                    for b in range(len(dof_map)):
                        var Bidx = free_map[b]
                        if Bidx < 0:
                            continue
                        banded_add(K, bw, Aidx, Bidx, k_global[a][b])
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
                var free_map: List[Int] = []
                free_map.resize(len(dof_map), -1)
                for i in range(len(dof_map)):
                    free_map[i] = free_index[dof_map[i]]
                for a in range(len(dof_map)):
                    var Aidx = free_map[a]
                    if Aidx < 0:
                        continue
                    for b in range(len(dof_map)):
                        var Bidx = free_map[b]
                        if Bidx < 0:
                            continue
                        banded_add(K, bw, Aidx, Bidx, k_global[a][b])
        elif elem_type == "twoNodeLink" or elem_type == "zeroLength":
            if ndf < 2:
                abort("twoNodeLink/zeroLength requires ndf >= 2")
            var n1 = Int(elem["nodes"][0])
            var n2 = Int(elem["nodes"][1])
            var i1 = id_to_index[n1]
            var i2 = id_to_index[n2]

            var elem_mats = elem["materials"]
            var elem_dirs = elem["dirs"]
            if py_len(elem_mats) != py_len(elem_dirs):
                abort("zeroLength/twoNodeLink materials/dirs mismatch")

            var ks: List[Float64] = []
            var offset = elem_uniaxial_offsets[e]
            var count = elem_uniaxial_counts[e]
            if count != py_len(elem_mats):
                abort("zeroLength/twoNodeLink material count mismatch")
            ks.resize(count, 0.0)
            var dirs: List[Int] = []
            dirs.resize(count, 0)

            for m in range(count):
                var state_index = elem_uniaxial_state_ids[offset + m]
                var def_index = uniaxial_state_defs[state_index]
                var mat_def = uniaxial_defs[def_index]
                ks[m] = uni_mat_initial_tangent(mat_def)
                dirs[m] = Int(elem_dirs[m])

            var k_global = link_global_stiffness(dirs, ks)
            var dof_map = [
                node_dof_index(i1, 1, ndf),
                node_dof_index(i1, 2, ndf),
                node_dof_index(i2, 1, ndf),
                node_dof_index(i2, 2, ndf),
            ]
            var free_map: List[Int] = []
            free_map.resize(4, -1)
            for i in range(4):
                free_map[i] = free_index[dof_map[i]]
            for a in range(4):
                var Aidx = free_map[a]
                if Aidx < 0:
                    continue
                for b in range(4):
                    var Bidx = free_map[b]
                    if Bidx < 0:
                        continue
                    banded_add(K, bw, Aidx, Bidx, k_global[a][b])
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
            if String(mat["type"]) != "Elastic":
                abort("fourNodeQuad only supports Elastic material")
            var params = mat["params"]
            var E = Float64(params["E"])
            var nu = Float64(params["nu"])
            var t = Float64(params["t"])

            var x = [
                Float64(node1["x"]),
                Float64(node2["x"]),
                Float64(node3["x"]),
                Float64(node4["x"]),
            ]
            var y = [
                Float64(node1["y"]),
                Float64(node2["y"]),
                Float64(node3["y"]),
                Float64(node4["y"]),
            ]

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
            var free_map: List[Int] = []
            free_map.resize(8, -1)
            for i in range(8):
                free_map[i] = free_index[dof_map[i]]
            for a in range(8):
                var Aidx = free_map[a]
                if Aidx < 0:
                    continue
                for b in range(8):
                    var Bidx = free_map[b]
                    if Bidx < 0:
                        continue
                    banded_add(K, bw, Aidx, Bidx, k_global[a][b])
        elif elem_type == "shellMITC4":
            if ndm != 3 or ndf != 6:
                abort("shellMITC4 requires ndm=3, ndf=6")
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
            if String(mat["type"]) != "Elastic":
                abort("shellMITC4 only supports Elastic material")
            var params = mat["params"]
            var E = Float64(params["E"])
            var nu = Float64(params["nu"])
            var h = Float64(params["h"])

            var x = [
                Float64(node1["x"]),
                Float64(node2["x"]),
                Float64(node3["x"]),
                Float64(node4["x"]),
            ]
            var y = [
                Float64(node1["y"]),
                Float64(node2["y"]),
                Float64(node3["y"]),
                Float64(node4["y"]),
            ]
            var z = [
                Float64(node1["z"]),
                Float64(node2["z"]),
                Float64(node3["z"]),
                Float64(node4["z"]),
            ]

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
            var free_map: List[Int] = []
            free_map.resize(24, -1)
            for i in range(24):
                free_map[i] = free_index[dof_map[i]]
            for a in range(24):
                var Aidx = free_map[a]
                if Aidx < 0:
                    continue
                for b in range(24):
                    var Bidx = free_map[b]
                    if Bidx < 0:
                        continue
                    banded_add(K, bw, Aidx, Bidx, k_global[a][b])
        else:
            abort("unsupported element type: " + elem_type)

    return K^
