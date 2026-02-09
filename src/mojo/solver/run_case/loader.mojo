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
    var modal_num_modes: Int
    var constraints_handler: String
    var use_banded_linear: Bool
    var use_banded_nonlinear: Bool
    var has_transformation_mpc: Bool

    var free: List[Int]
    var free_index: List[Int]
    var rep_dof: List[Int]
    var active_index_by_dof: List[Int]

    var M_total: List[Float64]
    var time_series: PythonObject
    var ts_index: Int
    var pattern_type: String
    var uniform_excitation_direction: Int
    var uniform_accel_ts_index: Int
    var rayleigh_alpha_m: Float64
    var rayleigh_beta_k: Float64
    var rayleigh_beta_k_init: Float64
    var rayleigh_beta_k_comm: Float64
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
        self.modal_num_modes = 0
        self.constraints_handler = "Plain"
        self.use_banded_linear = False
        self.use_banded_nonlinear = False
        self.has_transformation_mpc = False
        self.free = []
        self.free_index = []
        self.rep_dof = []
        self.active_index_by_dof = []
        self.M_total = []
        self.time_series = None
        self.ts_index = -1
        self.pattern_type = "Plain"
        self.uniform_excitation_direction = 0
        self.uniform_accel_ts_index = -1
        self.rayleigh_alpha_m = 0.0
        self.rayleigh_beta_k = 0.0
        self.rayleigh_beta_k_init = 0.0
        self.rayleigh_beta_k_comm = 0.0
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
        if geom != "Linear" and geom != "PDelta":
            abort("forceBeamColumn2d supports geomTransf Linear or PDelta")
        var integration = String(elem.get("integration", "Lobatto"))
        if integration != "Lobatto":
            abort("forceBeamColumn2d supports Lobatto integration only")
        var num_int_pts = Int(elem.get("num_int_pts", 3))
        if num_int_pts != 3 and num_int_pts != 5:
            abort("forceBeamColumn2d supports num_int_pts=3 or 5")
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
    var constraints_handler = String(analysis.get("constraints", "Plain"))
    if constraints_handler != "Plain" and constraints_handler != "Transformation":
        abort("unsupported constraints handler: " + constraints_handler)
    var steps = Int(analysis.get("steps", 1))
    var modal_num_modes = 0
    if analysis_type == "modal_eigen":
        modal_num_modes = Int(analysis.get("num_modes", 0))
        if modal_num_modes < 1:
            abort("modal_eigen requires num_modes >= 1")
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
        if (
            analysis_type != "static_linear"
            and analysis_type != "static_nonlinear"
            and analysis_type != "transient_nonlinear"
        ):
            abort(
                "forceBeamColumn2d requires static_linear, static_nonlinear, "
                "or transient_nonlinear analysis"
            )
        if analysis_type == "transient_nonlinear":
            if force_beam_mode != "auto":
                abort("force_beam_mode is only supported for static forceBeamColumn2d analyses")
        else:
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
    if (
        analysis_type != "static_nonlinear"
        and analysis_type != "transient_nonlinear"
        and analysis_type != "modal_eigen"
        and used_nonelastic_uniaxial
    ):
        abort("nonlinear uniaxial materials require static_nonlinear or transient_nonlinear analysis")

    var solver_pref = String(analysis.get("system", analysis.get("solver", "auto")))
    if solver_pref == "":
        solver_pref = "auto"
    if solver_pref != "auto" and solver_pref != "dense" and solver_pref != "banded":
        abort("unsupported analysis system: " + solver_pref)
    var band_threshold = Int(analysis.get("band_threshold", 128))
    if band_threshold < 0:
        band_threshold = 0

    var rep_dof: List[Int] = []
    rep_dof.resize(total_dofs, 0)
    for i in range(total_dofs):
        rep_dof[i] = i

    var has_transformation_mpc = False
    var mp_constraints = data.get("mp_constraints", [])
    if py_len(mp_constraints) > 0:
        if constraints_handler != "Transformation":
            abort("mp_constraints require analysis.constraints=Transformation")
        has_transformation_mpc = True
    for i in range(py_len(mp_constraints)):
        var mpc = mp_constraints[i]
        var mpc_type = String(mpc.get("type", ""))
        if mpc_type != "equalDOF":
            abort("unsupported mp constraint type: " + mpc_type)
        var retained_node = Int(mpc["retained_node"])
        var constrained_node = Int(mpc["constrained_node"])
        if retained_node >= len(id_to_index) or id_to_index[retained_node] < 0:
            abort("equalDOF retained_node not found")
        if constrained_node >= len(id_to_index) or id_to_index[constrained_node] < 0:
            abort("equalDOF constrained_node not found")
        var retained_idx = id_to_index[retained_node]
        var constrained_idx = id_to_index[constrained_node]
        var dofs = mpc.get("dofs", [])
        if py_len(dofs) == 0:
            abort("equalDOF requires non-empty dofs")
        for j in range(py_len(dofs)):
            var dof = Int(dofs[j])
            require_dof_in_range(dof, ndf, "equalDOF")
            var retained_dof = node_dof_index(retained_idx, dof, ndf)
            var constrained_dof = node_dof_index(constrained_idx, dof, ndf)
            rep_dof[constrained_dof] = retained_dof

    for i in range(total_dofs):
        var rep = i
        while rep_dof[rep] != rep:
            rep = rep_dof[rep]
        rep_dof[i] = rep

    var F_effective: List[Float64] = []
    F_effective.resize(total_dofs, 0.0)
    var M_effective: List[Float64] = []
    M_effective.resize(total_dofs, 0.0)
    for i in range(total_dofs):
        var rep = rep_dof[i]
        F_effective[rep] += F_total[i]
        M_effective[rep] += M_total[i]
    F_total = F_effective^
    M_total = M_effective^

    var active_index_by_dof: List[Int] = []
    active_index_by_dof.resize(total_dofs, -1)
    var free_count = 0
    for i in range(total_dofs):
        if rep_dof[i] != i:
            constrained[i] = True
            continue
        if constrained[i]:
            continue
        active_index_by_dof[i] = free_count
        free_count += 1

    if free_count == 0:
        abort("no free dofs")

    var use_banded_linear = False
    var use_banded_nonlinear = False
    if has_transformation_mpc:
        use_banded_linear = False
        use_banded_nonlinear = False
    elif analysis_type == "static_linear":
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
                if rep_dof[idx] != idx:
                    continue
                if not constrained[idx]:
                    free_index[idx] = len(free)
                    free.append(idx)
    else:
        free_index.resize(total_dofs, -1)
        for i in range(total_dofs):
            if rep_dof[i] != i:
                continue
            if not constrained[i]:
                free_index[i] = len(free)
                free.append(i)

    var time_series = parse_time_series(data)
    var ts_index = -1
    var pattern = data.get("pattern", None)
    var pattern_type = "Plain"
    var uniform_excitation_direction = 0
    var uniform_accel_ts_index = -1
    if pattern is not None:
        pattern_type = String(pattern.get("type", "Plain"))
        if pattern_type != "Plain" and pattern_type != "UniformExcitation":
            abort("unsupported pattern type: " + pattern_type)
    if py_len(time_series) > 0 or pattern is not None:
        if pattern is None:
            if py_len(time_series) == 1:
                var ts_tag = Int(time_series[0]["tag"])
                ts_index = find_time_series(time_series, ts_tag)
                if ts_index < 0:
                    abort("time_series tag not found")
            else:
                abort("pattern missing for multiple time_series")
        elif pattern_type == "Plain":
            if not pattern.__contains__("time_series"):
                abort("pattern missing time_series")
            var ts_tag = Int(pattern["time_series"])
            ts_index = find_time_series(time_series, ts_tag)
            if ts_index < 0:
                abort("time_series tag not found")
        else:
            if analysis_type != "transient_linear" and analysis_type != "transient_nonlinear":
                abort("UniformExcitation requires transient analysis")
            if not pattern.__contains__("direction"):
                abort("UniformExcitation pattern missing direction")
            uniform_excitation_direction = Int(pattern["direction"])
            if uniform_excitation_direction < 1 or uniform_excitation_direction > ndm:
                abort("UniformExcitation direction out of range 1..ndm")
            var accel_tag = -1
            if pattern.__contains__("accel"):
                accel_tag = Int(pattern["accel"])
            elif pattern.__contains__("time_series"):
                accel_tag = Int(pattern["time_series"])
            else:
                abort("UniformExcitation pattern missing accel time_series tag")
            uniform_accel_ts_index = find_time_series(time_series, accel_tag)
            if uniform_accel_ts_index < 0:
                abort("UniformExcitation accel time_series tag not found")
            if py_len(loads) > 0 or py_len(element_loads) > 0:
                abort("UniformExcitation does not support nodal/element loads")

    var rayleigh_alpha_m = 0.0
    var rayleigh_beta_k = 0.0
    var rayleigh_beta_k_init = 0.0
    var rayleigh_beta_k_comm = 0.0
    var rayleigh = data.get("rayleigh", None)
    if rayleigh is not None:
        rayleigh_alpha_m = Float64(rayleigh.get("alphaM", 0.0))
        rayleigh_beta_k = Float64(rayleigh.get("betaK", 0.0))
        rayleigh_beta_k_init = Float64(rayleigh.get("betaKInit", 0.0))
        rayleigh_beta_k_comm = Float64(rayleigh.get("betaKComm", 0.0))

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
    state.modal_num_modes = modal_num_modes
    state.constraints_handler = constraints_handler
    state.use_banded_linear = use_banded_linear
    state.use_banded_nonlinear = use_banded_nonlinear
    state.has_transformation_mpc = has_transformation_mpc
    state.free = free^
    state.free_index = free_index^
    state.rep_dof = rep_dof^
    state.active_index_by_dof = active_index_by_dof^
    state.M_total = M_total^
    state.time_series = time_series
    state.ts_index = ts_index
    state.pattern_type = pattern_type
    state.uniform_excitation_direction = uniform_excitation_direction
    state.uniform_accel_ts_index = uniform_accel_ts_index
    state.rayleigh_alpha_m = rayleigh_alpha_m
    state.rayleigh_beta_k = rayleigh_beta_k
    state.rayleigh_beta_k_init = rayleigh_beta_k_init
    state.rayleigh_beta_k_comm = rayleigh_beta_k_comm
    state.recorders = recorders

    return state^
