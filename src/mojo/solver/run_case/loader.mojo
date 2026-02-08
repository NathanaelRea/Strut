from collections import List
from elements import beam_uniform_load_global
from os import abort
from python import PythonObject

from materials import UniMaterialDef, UniMaterialState, uni_mat_is_elastic
from solver.dof import node_dof_index, require_dof_in_range
from solver.reorder import build_node_adjacency, rcm_order
from solver.time_series import find_time_series, parse_time_series
from sections import FiberCell, FiberSection2dDef, append_fiber_section2d_from_json
from strut_io import py_len

struct RunCaseState(Movable):
    var ndm: Int
    var ndf: Int

    var nodes: PythonObject
    var node_count: Int
    var id_to_index: List[Int]

    var sections_by_id: List[PythonObject]
    var materials_by_id: List[PythonObject]

    var uniaxial_defs: List[UniMaterialDef]
    var uniaxial_state_defs: List[Int]
    var uniaxial_states: List[UniMaterialState]

    var fiber_section_defs: List[FiberSection2dDef]
    var fiber_section_cells: List[FiberCell]
    var fiber_section_index_by_id: List[Int]

    var elements: PythonObject
    var elem_count: Int
    var elem_id_to_index: List[Int]
    var elem_uniaxial_offsets: List[Int]
    var elem_uniaxial_counts: List[Int]
    var elem_uniaxial_state_ids: List[Int]

    var total_dofs: Int
    var F_total: List[Float64]
    var constrained: List[Bool]

    var analysis: PythonObject
    var analysis_type: String
    var steps: Int
    var use_banded_linear: Bool
    var use_banded_nonlinear: Bool

    var free: List[Int]
    var free_index: List[Int]

    var M_total: List[Float64]
    var time_series: PythonObject
    var ts_index: Int
    var recorders: PythonObject

    fn __init__(out self):
        self.ndm = 0
        self.ndf = 0
        self.nodes = None
        self.node_count = 0
        self.id_to_index = []
        self.sections_by_id = []
        self.materials_by_id = []
        self.uniaxial_defs = []
        self.uniaxial_state_defs = []
        self.uniaxial_states = []
        self.fiber_section_defs = []
        self.fiber_section_cells = []
        self.fiber_section_index_by_id = []
        self.elements = None
        self.elem_count = 0
        self.elem_id_to_index = []
        self.elem_uniaxial_offsets = []
        self.elem_uniaxial_counts = []
        self.elem_uniaxial_state_ids = []
        self.total_dofs = 0
        self.F_total = []
        self.constrained = []
        self.analysis = None
        self.analysis_type = ""
        self.steps = 0
        self.use_banded_linear = False
        self.use_banded_nonlinear = False
        self.free = []
        self.free_index = []
        self.M_total = []
        self.time_series = None
        self.ts_index = -1
        self.recorders = None


fn load_case_state(data: PythonObject) raises -> RunCaseState:
    var state = RunCaseState()

    var model = data["model"]
    var ndm = Int(model["ndm"])
    var ndf = Int(model["ndf"])
    var is_2d = ndm == 2 and (ndf == 2 or ndf == 3)
    var is_3d_truss = ndm == 3 and ndf == 3
    var is_3d_shell = ndm == 3 and ndf == 6
    if not is_2d and not is_3d_truss and not is_3d_shell:
        abort("only ndm=2 ndf=2/3 and ndm=3 ndf=3/6 supported")

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
    var sections_by_id: List[PythonObject] = []
    sections_by_id.resize(0, None)
    for i in range(py_len(sections)):
        var sec = sections[i]
        var sid = Int(sec["id"])
        if sid >= len(sections_by_id):
            sections_by_id.resize(sid + 1, None)
        sections_by_id[sid] = sec
    var materials_by_id: List[PythonObject] = []
    materials_by_id.resize(0, None)
    for i in range(py_len(materials)):
        var mat = materials[i]
        var mid = Int(mat["id"])
        if mid >= len(materials_by_id):
            materials_by_id.resize(mid + 1, None)
        materials_by_id[mid] = mat

    var uniaxial_defs: List[UniMaterialDef] = []
    var uniaxial_def_by_id: List[Int] = []
    uniaxial_def_by_id.resize(len(materials_by_id), -1)
    for i in range(py_len(materials)):
        var mat = materials[i]
        var mid = Int(mat["id"])
        if mid >= len(uniaxial_def_by_id):
            uniaxial_def_by_id.resize(mid + 1, -1)
        var mat_type = String(mat["type"])
        if mat_type == "Elastic":
            var params = mat["params"]
            var E = Float64(params["E"])
            if E <= 0.0:
                abort("Elastic material E must be > 0")
            var mat_def = UniMaterialDef(0, E, 0.0, 0.0, 0.0)
            uniaxial_def_by_id[mid] = len(uniaxial_defs)
            uniaxial_defs.append(mat_def)
        elif mat_type == "Steel01":
            var params = mat["params"]
            var Fy = Float64(params["Fy"])
            var E0 = Float64(params["E0"])
            var b = Float64(params["b"])
            if Fy <= 0.0:
                abort("Steel01 Fy must be > 0")
            if E0 <= 0.0:
                abort("Steel01 E0 must be > 0")
            if b < 0.0 or b >= 1.0:
                abort("Steel01 b must be in [0, 1)")
            var mat_def = UniMaterialDef(1, Fy, E0, b, 0.0)
            uniaxial_def_by_id[mid] = len(uniaxial_defs)
            uniaxial_defs.append(mat_def)
        elif mat_type == "Concrete01":
            var params = mat["params"]
            var fpc = Float64(params["fpc"])
            var epsc0 = Float64(params["epsc0"])
            var fpcu = Float64(params["fpcu"])
            var epscu = Float64(params["epscu"])
            if fpc > 0.0:
                fpc = -fpc
            if epsc0 > 0.0:
                epsc0 = -epsc0
            if fpcu > 0.0:
                fpcu = -fpcu
            if epscu > 0.0:
                epscu = -epscu
            if fpc >= 0.0:
                abort("Concrete01 fpc must be < 0")
            if epsc0 >= 0.0:
                abort("Concrete01 epsc0 must be < 0")
            if epscu >= 0.0:
                abort("Concrete01 epscu must be < 0")
            if epscu >= epsc0:
                abort("Concrete01 epscu must be < epsc0")
            if fpcu > 0.0 or fpcu < fpc:
                abort("Concrete01 fpcu must be between fpc and 0")
            var mat_def = UniMaterialDef(2, fpc, epsc0, fpcu, epscu)
            uniaxial_def_by_id[mid] = len(uniaxial_defs)
            uniaxial_defs.append(mat_def)
        elif mat_type == "Steel02":
            var params = mat["params"]
            var Fy = Float64(params["Fy"])
            var E0 = Float64(params["E0"])
            var b = Float64(params["b"])
            if Fy <= 0.0:
                abort("Steel02 Fy must be > 0")
            if E0 <= 0.0:
                abort("Steel02 E0 must be > 0")
            if b < 0.0 or b >= 1.0:
                abort("Steel02 b must be in [0, 1)")

            var has_r0 = params.__contains__("R0")
            var has_cr1 = params.__contains__("cR1")
            var has_cr2 = params.__contains__("cR2")
            var has_a1 = params.__contains__("a1")
            var has_a2 = params.__contains__("a2")
            var has_a3 = params.__contains__("a3")
            var has_a4 = params.__contains__("a4")
            var has_siginit = params.__contains__("sigInit")

            if has_r0 != has_cr1 or has_r0 != has_cr2:
                abort("Steel02 requires R0, cR1, cR2 together")
            if has_a1 != has_a2 or has_a1 != has_a3 or has_a1 != has_a4:
                abort("Steel02 requires a1, a2, a3, a4 together")
            if has_a1 and not has_r0:
                abort("Steel02 a1-a4 require R0, cR1, cR2")
            if has_siginit and not has_a1:
                abort("Steel02 sigInit requires a1-a4 and R0/cR1/cR2")

            var R0 = 15.0
            var cR1 = 0.925
            var cR2 = 0.15
            if has_r0:
                R0 = Float64(params["R0"])
                cR1 = Float64(params["cR1"])
                cR2 = Float64(params["cR2"])
            if R0 <= 0.0:
                abort("Steel02 R0 must be > 0")
            if cR2 <= 0.0:
                abort("Steel02 cR2 must be > 0")

            var a1 = 0.0
            var a2 = 1.0
            var a3 = 0.0
            var a4 = 1.0
            if has_a1:
                a1 = Float64(params["a1"])
                a2 = Float64(params["a2"])
                a3 = Float64(params["a3"])
                a4 = Float64(params["a4"])
            if a2 <= 0.0:
                abort("Steel02 a2 must be > 0")
            if a4 <= 0.0:
                abort("Steel02 a4 must be > 0")

            var sig_init = 0.0
            if has_siginit:
                sig_init = Float64(params["sigInit"])

            var mat_def = UniMaterialDef(
                3, Fy, E0, b, R0, cR1, cR2, a1, a2, a3, a4, sig_init
            )
            uniaxial_def_by_id[mid] = len(uniaxial_defs)
            uniaxial_defs.append(mat_def)
        elif mat_type == "Concrete02":
            var params = mat["params"]
            var fpc = Float64(params["fpc"])
            var epsc0 = Float64(params["epsc0"])
            var fpcu = Float64(params["fpcu"])
            var epscu = Float64(params["epscu"])
            if fpc > 0.0:
                fpc = -fpc
            if epsc0 > 0.0:
                epsc0 = -epsc0
            if fpcu > 0.0:
                fpcu = -fpcu
            if epscu > 0.0:
                epscu = -epscu
            if fpc >= 0.0:
                abort("Concrete02 fpc must be < 0")
            if epsc0 >= 0.0:
                abort("Concrete02 epsc0 must be < 0")
            if epscu >= 0.0:
                abort("Concrete02 epscu must be < 0")
            if epscu >= epsc0:
                abort("Concrete02 epscu must be < epsc0")
            if fpcu > 0.0 or fpcu < fpc:
                abort("Concrete02 fpcu must be between fpc and 0")

            var has_rat = params.__contains__("rat")
            var has_ft = params.__contains__("ft")
            var has_ets = params.__contains__("Ets")
            if has_rat != has_ft or has_rat != has_ets:
                abort("Concrete02 requires rat, ft, Ets together")

            var rat = 0.1
            var ft = 0.1 * fpc
            if ft < 0.0:
                ft = -ft
            var Ets = 0.1 * fpc / epsc0
            if has_rat:
                rat = Float64(params["rat"])
                ft = Float64(params["ft"])
                Ets = Float64(params["Ets"])

            if rat == 1.0:
                abort("Concrete02 rat must not be 1")
            if ft < 0.0:
                abort("Concrete02 ft must be >= 0")
            if Ets <= 0.0:
                abort("Concrete02 Ets must be > 0")

            var mat_def = UniMaterialDef(
                4, fpc, epsc0, fpcu, epscu, rat, ft, Ets, 0.0, 0.0, 0.0, 0.0
            )
            uniaxial_def_by_id[mid] = len(uniaxial_defs)
            uniaxial_defs.append(mat_def)
        elif mat_type == "ElasticIsotropic":
            continue
        else:
            abort("unsupported material type: " + mat_type)

    var fiber_section_defs: List[FiberSection2dDef] = []
    var fiber_section_cells: List[FiberCell] = []
    var fiber_section_index_by_id: List[Int] = []
    fiber_section_index_by_id.resize(len(sections_by_id), -1)
    for i in range(py_len(sections)):
        var sec = sections[i]
        var sec_type = String(sec["type"])
        if sec_type != "FiberSection2d":
            continue
        var sid = Int(sec["id"])
        if sid >= len(fiber_section_index_by_id):
            fiber_section_index_by_id.resize(sid + 1, -1)
        append_fiber_section2d_from_json(
            sec,
            uniaxial_def_by_id,
            fiber_section_defs,
            fiber_section_cells,
        )
        fiber_section_index_by_id[sid] = len(fiber_section_defs) - 1

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

    var has_force_beam_column2d = False
    for i in range(elem_count):
        var elem = elements[i]
        var elem_type = String(elem["type"])
        if elem_type == "elasticBeamColumn2d" or elem_type == "elasticBeamColumn3d":
            var sec_id = Int(elem["section"])
            if sec_id >= 0 and sec_id < len(fiber_section_index_by_id):
                if fiber_section_index_by_id[sec_id] >= 0:
                    abort(
                        elem_type + " with FiberSection2d requires forceBeamColumn2d"
                    )
            continue
        if elem_type != "forceBeamColumn2d":
            continue
        has_force_beam_column2d = True
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
        var sec_id = Int(elem["section"])
        if sec_id < 0 or sec_id >= len(fiber_section_index_by_id):
            abort("forceBeamColumn2d section not found")
        if fiber_section_index_by_id[sec_id] < 0:
            abort("forceBeamColumn2d requires FiberSection2d")

    var uniaxial_states: List[UniMaterialState] = []
    var uniaxial_state_defs: List[Int] = []
    var elem_uniaxial_offsets: List[Int] = []
    var elem_uniaxial_counts: List[Int] = []
    var elem_uniaxial_state_ids: List[Int] = []
    elem_uniaxial_offsets.resize(elem_count, 0)
    elem_uniaxial_counts.resize(elem_count, 0)
    var used_nonelastic_uniaxial = False
    var force_beam_has_nonelastic = False
    for e in range(elem_count):
        var elem = elements[e]
        var elem_type = String(elem["type"])
        if elem_type == "truss":
            var mat_id = Int(elem["material"])
            if mat_id >= len(uniaxial_def_by_id) or uniaxial_def_by_id[mat_id] < 0:
                abort("truss requires uniaxial material")
            var def_index = uniaxial_def_by_id[mat_id]
            var mat_def = uniaxial_defs[def_index]
            var state_index = len(uniaxial_states)
            uniaxial_states.append(UniMaterialState(mat_def))
            uniaxial_state_defs.append(def_index)
            elem_uniaxial_offsets[e] = len(elem_uniaxial_state_ids)
            elem_uniaxial_counts[e] = 1
            elem_uniaxial_state_ids.append(state_index)
            if not uni_mat_is_elastic(mat_def):
                used_nonelastic_uniaxial = True
        elif elem_type == "zeroLength" or elem_type == "twoNodeLink":
            var elem_mats = elem["materials"]
            elem_uniaxial_offsets[e] = len(elem_uniaxial_state_ids)
            elem_uniaxial_counts[e] = py_len(elem_mats)
            for m in range(py_len(elem_mats)):
                var mat_id = Int(elem_mats[m])
                if mat_id >= len(uniaxial_def_by_id) or uniaxial_def_by_id[mat_id] < 0:
                    abort("zeroLength/twoNodeLink requires uniaxial material")
                var def_index = uniaxial_def_by_id[mat_id]
                var mat_def = uniaxial_defs[def_index]
                var state_index = len(uniaxial_states)
                uniaxial_states.append(UniMaterialState(mat_def))
                uniaxial_state_defs.append(def_index)
                elem_uniaxial_state_ids.append(state_index)
                if not uni_mat_is_elastic(mat_def):
                    used_nonelastic_uniaxial = True
        elif elem_type == "forceBeamColumn2d":
            var sec_id = Int(elem["section"])
            var sec_index = fiber_section_index_by_id[sec_id]
            if sec_index < 0 or sec_index >= len(fiber_section_defs):
                abort("forceBeamColumn2d requires FiberSection2d")
            var sec_def = fiber_section_defs[sec_index]
            var num_int_pts = Int(elem.get("num_int_pts", 3))
            var state_count = num_int_pts * sec_def.fiber_count
            elem_uniaxial_offsets[e] = len(elem_uniaxial_state_ids)
            elem_uniaxial_counts[e] = state_count
            for _ in range(num_int_pts):
                for i in range(sec_def.fiber_count):
                    var cell = fiber_section_cells[sec_def.fiber_offset + i]
                    var def_index = cell.def_index
                    if def_index < 0 or def_index >= len(uniaxial_defs):
                        abort("forceBeamColumn2d fiber material definition out of range")
                    var mat_def = uniaxial_defs[def_index]
                    var state_index = len(uniaxial_states)
                    uniaxial_states.append(UniMaterialState(mat_def))
                    uniaxial_state_defs.append(def_index)
                    elem_uniaxial_state_ids.append(state_index)
                    if not uni_mat_is_elastic(mat_def):
                        used_nonelastic_uniaxial = True
                        force_beam_has_nonelastic = True
        else:
            elem_uniaxial_offsets[e] = len(elem_uniaxial_state_ids)
            elem_uniaxial_counts[e] = 0

    var total_dofs = node_count * ndf
    var F_total: List[Float64] = []
    F_total.resize(total_dofs, 0.0)

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
                F_total[dof_map[a]] += f_global[a]
        else:
            abort("unsupported element load type")

    var loads = data.get("loads", [])
    for i in range(py_len(loads)):
        var load = loads[i]
        var node_id = Int(load["node"])
        var dof = Int(load["dof"])
        require_dof_in_range(dof, ndf, "load")
        var idx = node_dof_index(id_to_index[node_id], dof, ndf)
        F_total[idx] += Float64(load["value"])

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

    var analysis = data.get("analysis", {"type": "static_linear", "steps": 1})
    var analysis_type = String(analysis.get("type", "static_linear"))
    var steps = Int(analysis.get("steps", 1))
    if steps < 1:
        abort("analysis steps must be >= 1")
    var force_beam_mode = String(analysis.get("force_beam_mode", "auto"))
    if (
        force_beam_mode != "auto"
        and force_beam_mode != "linear_if_elastic"
        and force_beam_mode != "nonlinear"
    ):
        abort(
            "unsupported force_beam_mode: "
            + force_beam_mode
            + " (expected auto|linear_if_elastic|nonlinear)"
        )
    if has_force_beam_column2d:
        if analysis_type != "static_linear" and analysis_type != "static_nonlinear":
            abort("forceBeamColumn2d requires static_linear or static_nonlinear analysis")
        if force_beam_mode == "nonlinear":
            if analysis_type != "static_nonlinear":
                abort("force_beam_mode=nonlinear requires static_nonlinear analysis")
        elif force_beam_mode == "linear_if_elastic":
            if force_beam_has_nonelastic:
                if analysis_type != "static_nonlinear":
                    abort("forceBeamColumn2d with non-elastic fibers requires static_nonlinear analysis")
            elif analysis_type != "static_linear":
                abort(
                    "force_beam_mode=linear_if_elastic requires static_linear "
                    "analysis for elastic forceBeamColumn2d"
                )
        elif analysis_type == "static_linear" and force_beam_has_nonelastic:
            abort("forceBeamColumn2d with non-elastic fibers requires static_nonlinear analysis")
    if analysis_type != "static_nonlinear" and used_nonelastic_uniaxial:
        abort("nonlinear uniaxial materials require static_nonlinear analysis")

    var solver_pref = String(analysis.get("system", analysis.get("solver", "auto")))
    if solver_pref == "":
        solver_pref = "auto"
    if solver_pref != "auto" and solver_pref != "dense" and solver_pref != "banded":
        abort("unsupported analysis system: " + solver_pref)
    var band_threshold = Int(analysis.get("band_threshold", 128))
    if band_threshold < 0:
        band_threshold = 0

    var free_count = 0
    for i in range(total_dofs):
        if not constrained[i]:
            free_count += 1

    if free_count == 0:
        abort("no free dofs")

    var use_banded_linear = False
    var use_banded_nonlinear = False
    if analysis_type == "static_linear":
        if solver_pref == "banded" or (solver_pref == "auto" and free_count > band_threshold):
            use_banded_linear = True
    elif analysis_type == "static_nonlinear":
        if solver_pref == "banded" or (solver_pref == "auto" and free_count > band_threshold):
            use_banded_nonlinear = True

    var free: List[Int] = []
    var free_index: List[Int] = []
    if use_banded_linear or use_banded_nonlinear:
        var adjacency = build_node_adjacency(elements, node_count, id_to_index)
        var node_order = rcm_order(adjacency)
        free_index.resize(total_dofs, -1)
        for i in range(len(node_order)):
            var node_idx = node_order[i]
            for dof in range(1, ndf + 1):
                var idx = node_dof_index(node_idx, dof, ndf)
                if not constrained[idx]:
                    free_index[idx] = len(free)
                    free.append(idx)
    else:
        for i in range(total_dofs):
            if not constrained[i]:
                free.append(i)

    var M_total: List[Float64] = []
    M_total.resize(total_dofs, 0.0)
    var masses = data.get("masses", [])
    for i in range(py_len(masses)):
        var mass = masses[i]
        var node_id = Int(mass["node"])
        var dof = Int(mass["dof"])
        require_dof_in_range(dof, ndf, "mass")
        var idx = node_dof_index(id_to_index[node_id], dof, ndf)
        M_total[idx] += Float64(mass["value"])

    var time_series = parse_time_series(data)
    var ts_index = -1
    if py_len(time_series) > 0:
        var ts_tag = -1
        var pattern = data.get("pattern", None)
        if pattern is None:
            if py_len(time_series) == 1:
                ts_tag = Int(time_series[0]["tag"])
            else:
                abort("pattern missing for multiple time_series")
        else:
            var pattern_type = String(pattern.get("type", "Plain"))
            if pattern_type != "Plain":
                abort("unsupported pattern type: " + pattern_type)
            if not pattern.__contains__("time_series"):
                abort("pattern missing time_series")
            ts_tag = Int(pattern["time_series"])
        ts_index = find_time_series(time_series, ts_tag)
        if ts_index < 0:
            abort("time_series tag not found")

    var recorders = data.get("recorders", [])

    state.ndm = ndm
    state.ndf = ndf
    state.nodes = nodes
    state.node_count = node_count
    state.id_to_index = id_to_index^
    state.sections_by_id = sections_by_id^
    state.materials_by_id = materials_by_id^
    state.uniaxial_defs = uniaxial_defs^
    state.uniaxial_state_defs = uniaxial_state_defs^
    state.uniaxial_states = uniaxial_states^
    state.fiber_section_defs = fiber_section_defs^
    state.fiber_section_cells = fiber_section_cells^
    state.fiber_section_index_by_id = fiber_section_index_by_id^
    state.elements = elements
    state.elem_count = elem_count
    state.elem_id_to_index = elem_id_to_index^
    state.elem_uniaxial_offsets = elem_uniaxial_offsets^
    state.elem_uniaxial_counts = elem_uniaxial_counts^
    state.elem_uniaxial_state_ids = elem_uniaxial_state_ids^
    state.total_dofs = total_dofs
    state.F_total = F_total^
    state.constrained = constrained^
    state.analysis = analysis
    state.analysis_type = analysis_type
    state.steps = steps
    state.use_banded_linear = use_banded_linear
    state.use_banded_nonlinear = use_banded_nonlinear
    state.free = free^
    state.free_index = free_index^
    state.M_total = M_total^
    state.time_series = time_series
    state.ts_index = ts_index
    state.recorders = recorders

    return state^
