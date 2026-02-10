from collections import List
from elements import beam_uniform_load_global
from os import abort
from python import PythonObject

from materials import UniMaterialDef, UniMaterialState, uni_mat_is_elastic
from solver.dof import node_dof_index, require_dof_in_range
from solver.reorder import build_node_adjacency_typed, rcm_order
from solver.run_case.input_types import (
    AnalysisInput,
    ElementInput,
    MaterialInput,
    NodeInput,
    RecorderInput,
    SectionInput,
    parse_case_input,
)
from solver.time_series import (
    TimeSeriesInput,
    find_time_series_input,
    parse_time_series_inputs,
)
from sections import FiberCell, FiberSection2dDef, append_fiber_section2d_from_json
from strut_io import py_len
from tag_types import (
    ElementTypeTag,
    GeomTransfTag,
    LinkDirectionTag,
    UniMaterialTypeTag,
)

struct RunCaseState(Movable):
    var ndm: Int
    var ndf: Int

    var typed_nodes: List[NodeInput]
    var node_count: Int
    var id_to_index: List[Int]

    var typed_sections_by_id: List[SectionInput]
    var typed_materials_by_id: List[MaterialInput]

    var uniaxial_defs: List[UniMaterialDef]
    var uniaxial_state_defs: List[Int]
    var uniaxial_states: List[UniMaterialState]

    var fiber_section_defs: List[FiberSection2dDef]
    var fiber_section_cells: List[FiberCell]
    var fiber_section_index_by_id: List[Int]

    var typed_elements: List[ElementInput]
    var elem_count: Int
    var elem_id_to_index: List[Int]
    var elem_uniaxial_offsets: List[Int]
    var elem_uniaxial_counts: List[Int]
    var elem_uniaxial_state_ids: List[Int]
    var force_basic_offsets: List[Int]
    var force_basic_counts: List[Int]
    var force_basic_q: List[Float64]

    var total_dofs: Int
    var F_total: List[Float64]
    var constrained: List[Bool]

    var analysis: AnalysisInput
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
    var analysis_integrator_targets_pool: List[Float64]
    var time_series: List[TimeSeriesInput]
    var time_series_values: List[Float64]
    var time_series_times: List[Float64]
    var ts_index: Int
    var pattern_type: String
    var uniform_excitation_direction: Int
    var uniform_accel_ts_index: Int
    var rayleigh_alpha_m: Float64
    var rayleigh_beta_k: Float64
    var rayleigh_beta_k_init: Float64
    var rayleigh_beta_k_comm: Float64
    var recorder_nodes_pool: List[Int]
    var recorder_elements_pool: List[Int]
    var recorder_dofs_pool: List[Int]
    var recorder_modes_pool: List[Int]
    var recorders: List[RecorderInput]

    fn __init__(out self):
        self.ndm = 0
        self.ndf = 0
        self.typed_nodes = []
        self.node_count = 0
        self.id_to_index = []
        self.typed_sections_by_id = []
        self.typed_materials_by_id = []
        self.uniaxial_defs = []
        self.uniaxial_state_defs = []
        self.uniaxial_states = []
        self.fiber_section_defs = []
        self.fiber_section_cells = []
        self.fiber_section_index_by_id = []
        self.typed_elements = []
        self.elem_count = 0
        self.elem_id_to_index = []
        self.elem_uniaxial_offsets = []
        self.elem_uniaxial_counts = []
        self.elem_uniaxial_state_ids = []
        self.force_basic_offsets = []
        self.force_basic_counts = []
        self.force_basic_q = []
        self.total_dofs = 0
        self.F_total = []
        self.constrained = []
        self.analysis = AnalysisInput()
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
        self.analysis_integrator_targets_pool = []
        self.time_series = []
        self.time_series_values = []
        self.time_series_times = []
        self.ts_index = -1
        self.pattern_type = "Plain"
        self.uniform_excitation_direction = 0
        self.uniform_accel_ts_index = -1
        self.rayleigh_alpha_m = 0.0
        self.rayleigh_beta_k = 0.0
        self.rayleigh_beta_k_init = 0.0
        self.rayleigh_beta_k_comm = 0.0
        self.recorder_nodes_pool = []
        self.recorder_elements_pool = []
        self.recorder_dofs_pool = []
        self.recorder_modes_pool = []
        self.recorders = []


fn _set_elem_dof(mut elem: ElementInput, idx: Int, value: Int):
    if idx == 0:
        elem.dof_1 = value
    elif idx == 1:
        elem.dof_2 = value
    elif idx == 2:
        elem.dof_3 = value
    elif idx == 3:
        elem.dof_4 = value
    elif idx == 4:
        elem.dof_5 = value
    elif idx == 5:
        elem.dof_6 = value
    elif idx == 6:
        elem.dof_7 = value
    elif idx == 7:
        elem.dof_8 = value
    elif idx == 8:
        elem.dof_9 = value
    elif idx == 9:
        elem.dof_10 = value
    elif idx == 10:
        elem.dof_11 = value
    elif idx == 11:
        elem.dof_12 = value
    elif idx == 12:
        elem.dof_13 = value
    elif idx == 13:
        elem.dof_14 = value
    elif idx == 14:
        elem.dof_15 = value
    elif idx == 15:
        elem.dof_16 = value
    elif idx == 16:
        elem.dof_17 = value
    elif idx == 17:
        elem.dof_18 = value
    elif idx == 18:
        elem.dof_19 = value
    elif idx == 19:
        elem.dof_20 = value
    elif idx == 20:
        elem.dof_21 = value
    elif idx == 21:
        elem.dof_22 = value
    elif idx == 22:
        elem.dof_23 = value
    else:
        elem.dof_24 = value


fn _elem_material(elem: ElementInput, idx: Int) -> Int:
    if idx == 0:
        return elem.material_1
    if idx == 1:
        return elem.material_2
    if idx == 2:
        return elem.material_3
    if idx == 3:
        return elem.material_4
    if idx == 4:
        return elem.material_5
    return elem.material_6


fn _elem_dir(elem: ElementInput, idx: Int) -> Int:
    if idx == 0:
        return elem.dir_1
    if idx == 1:
        return elem.dir_2
    if idx == 2:
        return elem.dir_3
    if idx == 3:
        return elem.dir_4
    if idx == 4:
        return elem.dir_5
    return elem.dir_6


fn load_case_state(data: PythonObject) raises -> RunCaseState:
    var state = RunCaseState()
    var input = parse_case_input(data)

    var ndm = input.model.ndm
    var ndf = input.model.ndf
    var is_2d = ndm == 2 and (ndf == 2 or ndf == 3)
    var is_3d_truss = ndm == 3 and ndf == 3
    var is_3d_shell = ndm == 3 and ndf == 6
    if not is_2d and not is_3d_truss and not is_3d_shell:
        abort("only ndm=2 ndf=2/3 and ndm=3 ndf=3/6 supported")

    var node_count = len(input.nodes)
    var node_ids: List[Int] = []
    node_ids.resize(node_count, 0)

    for i in range(node_count):
        var node = input.nodes[i]
        if ndm == 3 and not node.has_z:
            abort("ndm=3 requires node z coordinate")
        node_ids[i] = node.id

    var id_to_index: List[Int] = []
    id_to_index.resize(10000, -1)
    for i in range(node_count):
        var nid = node_ids[i]
        if nid >= len(id_to_index):
            id_to_index.resize(nid + 1, -1)
        id_to_index[nid] = i

    var sections = data.get("sections", [])
    var typed_sections_by_id: List[SectionInput] = []
    for i in range(len(input.sections)):
        var sid = input.sections[i].id
        if sid >= len(typed_sections_by_id):
            typed_sections_by_id.resize(sid + 1, SectionInput())
        typed_sections_by_id[sid] = input.sections[i]
    var typed_materials_by_id: List[MaterialInput] = []
    for i in range(len(input.materials)):
        var mid = input.materials[i].id
        if mid >= len(typed_materials_by_id):
            typed_materials_by_id.resize(mid + 1, MaterialInput())
        typed_materials_by_id[mid] = input.materials[i]

    var uniaxial_defs: List[UniMaterialDef] = []
    var uniaxial_def_by_id: List[Int] = []
    uniaxial_def_by_id.resize(len(typed_materials_by_id), -1)
    for i in range(len(input.materials)):
        var mat = input.materials[i]
        var mid = mat.id
        if mid >= len(uniaxial_def_by_id):
            uniaxial_def_by_id.resize(mid + 1, -1)
        var mat_type = mat.type
        if mat_type == "Elastic":
            var E = mat.E
            if E <= 0.0:
                abort("Elastic material E must be > 0")
            var mat_def = UniMaterialDef(UniMaterialTypeTag.Elastic, E, 0.0, 0.0, 0.0)
            uniaxial_def_by_id[mid] = len(uniaxial_defs)
            uniaxial_defs.append(mat_def)
        elif mat_type == "Steel01":
            var Fy = mat.Fy
            var E0 = mat.E0
            var b = mat.b
            if Fy <= 0.0:
                abort("Steel01 Fy must be > 0")
            if E0 <= 0.0:
                abort("Steel01 E0 must be > 0")
            if b < 0.0 or b >= 1.0:
                abort("Steel01 b must be in [0, 1)")
            var mat_def = UniMaterialDef(UniMaterialTypeTag.Steel01, Fy, E0, b, 0.0)
            uniaxial_def_by_id[mid] = len(uniaxial_defs)
            uniaxial_defs.append(mat_def)
        elif mat_type == "Concrete01":
            var fpc = mat.fpc
            var epsc0 = mat.epsc0
            var fpcu = mat.fpcu
            var epscu = mat.epscu
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
            var mat_def = UniMaterialDef(UniMaterialTypeTag.Concrete01, fpc, epsc0, fpcu, epscu)
            uniaxial_def_by_id[mid] = len(uniaxial_defs)
            uniaxial_defs.append(mat_def)
        elif mat_type == "Steel02":
            var Fy = mat.Fy
            var E0 = mat.E0
            var b = mat.b
            if Fy <= 0.0:
                abort("Steel02 Fy must be > 0")
            if E0 <= 0.0:
                abort("Steel02 E0 must be > 0")
            if b < 0.0 or b >= 1.0:
                abort("Steel02 b must be in [0, 1)")

            var has_r0 = mat.has_r0
            var has_cr1 = mat.has_cr1
            var has_cr2 = mat.has_cr2
            var has_a1 = mat.has_a1
            var has_a2 = mat.has_a2
            var has_a3 = mat.has_a3
            var has_a4 = mat.has_a4
            var has_siginit = mat.has_siginit

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
                R0 = mat.R0
                cR1 = mat.cR1
                cR2 = mat.cR2
            if R0 <= 0.0:
                abort("Steel02 R0 must be > 0")
            if cR2 <= 0.0:
                abort("Steel02 cR2 must be > 0")

            var a1 = 0.0
            var a2 = 1.0
            var a3 = 0.0
            var a4 = 1.0
            if has_a1:
                a1 = mat.a1
                a2 = mat.a2
                a3 = mat.a3
                a4 = mat.a4
            if a2 <= 0.0:
                abort("Steel02 a2 must be > 0")
            if a4 <= 0.0:
                abort("Steel02 a4 must be > 0")

            var sig_init = 0.0
            if has_siginit:
                sig_init = mat.sigInit

            var mat_def = UniMaterialDef(
                UniMaterialTypeTag.Steel02,
                Fy,
                E0,
                b,
                R0,
                cR1,
                cR2,
                a1,
                a2,
                a3,
                a4,
                sig_init,
            )
            uniaxial_def_by_id[mid] = len(uniaxial_defs)
            uniaxial_defs.append(mat_def)
        elif mat_type == "Concrete02":
            var fpc = mat.fpc
            var epsc0 = mat.epsc0
            var fpcu = mat.fpcu
            var epscu = mat.epscu
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

            var has_rat = mat.has_rat
            var has_ft = mat.has_ft
            var has_ets = mat.has_ets
            if has_rat != has_ft or has_rat != has_ets:
                abort("Concrete02 requires rat, ft, Ets together")

            var rat = 0.1
            var ft = 0.1 * fpc
            if ft < 0.0:
                ft = -ft
            var Ets = 0.1 * fpc / epsc0
            if has_rat:
                rat = mat.rat
                ft = mat.ft
                Ets = mat.Ets

            if rat == 1.0:
                abort("Concrete02 rat must not be 1")
            if ft < 0.0:
                abort("Concrete02 ft must be >= 0")
            if Ets <= 0.0:
                abort("Concrete02 Ets must be > 0")

            var mat_def = UniMaterialDef(
                UniMaterialTypeTag.Concrete02,
                fpc,
                epsc0,
                fpcu,
                epscu,
                rat,
                ft,
                Ets,
                0.0,
                0.0,
                0.0,
                0.0,
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
    fiber_section_index_by_id.resize(len(typed_sections_by_id), -1)
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

    var elem_count = len(input.elements)
    var typed_elements = input.elements.copy()
    var elem_ids: List[Int] = []
    elem_ids.resize(elem_count, 0)
    var elem_id_to_index: List[Int] = []
    elem_id_to_index.resize(10000, -1)
    for i in range(elem_count):
        var eid = typed_elements[i].id
        elem_ids[i] = eid
        if eid >= len(elem_id_to_index):
            elem_id_to_index.resize(eid + 1, -1)
        elem_id_to_index[eid] = i

    var has_force_beam_column2d = False
    for i in range(elem_count):
        var elem = typed_elements[i]
        if elem.type_tag == ElementTypeTag.Unknown:
            abort("unsupported element type: " + elem.type)

        if elem.node_1 >= len(id_to_index) or id_to_index[elem.node_1] < 0:
            abort("element node not found")
        if elem.node_2 >= len(id_to_index) or id_to_index[elem.node_2] < 0:
            abort("element node not found")
        elem.node_index_1 = id_to_index[elem.node_1]
        elem.node_index_2 = id_to_index[elem.node_2]
        if elem.node_count >= 3:
            if elem.node_3 >= len(id_to_index) or id_to_index[elem.node_3] < 0:
                abort("element node not found")
            elem.node_index_3 = id_to_index[elem.node_3]
        if elem.node_count >= 4:
            if elem.node_4 >= len(id_to_index) or id_to_index[elem.node_4] < 0:
                abort("element node not found")
            elem.node_index_4 = id_to_index[elem.node_4]

        if elem.type_tag == ElementTypeTag.ElasticBeamColumn2d:
            if ndm != 2 or ndf != 3:
                abort("elasticBeamColumn2d requires ndm=2, ndf=3")
            if elem.node_count != 2:
                abort("elasticBeamColumn2d requires 2 nodes")
            if elem.section < 0 or elem.section >= len(typed_sections_by_id):
                abort("section not found")
            var sec = typed_sections_by_id[elem.section]
            if sec.id < 0:
                abort("section not found")
            if sec.type != "ElasticSection2d":
                abort("elasticBeamColumn2d requires ElasticSection2d")
            if (
                elem.geom_tag != GeomTransfTag.Linear
                and elem.geom_tag != GeomTransfTag.PDelta
                and elem.geom_tag != GeomTransfTag.Corotational
            ):
                abort("unsupported geomTransf: " + elem.geom_transf)
            if elem.section < len(fiber_section_index_by_id):
                if fiber_section_index_by_id[elem.section] >= 0:
                    abort(
                        "elasticBeamColumn2d with FiberSection2d requires forceBeamColumn2d "
                        "or dispBeamColumn2d"
                    )
            elem.dof_count = 6
            _set_elem_dof(elem, 0, node_dof_index(elem.node_index_1, 1, ndf))
            _set_elem_dof(elem, 1, node_dof_index(elem.node_index_1, 2, ndf))
            _set_elem_dof(elem, 2, node_dof_index(elem.node_index_1, 3, ndf))
            _set_elem_dof(elem, 3, node_dof_index(elem.node_index_2, 1, ndf))
            _set_elem_dof(elem, 4, node_dof_index(elem.node_index_2, 2, ndf))
            _set_elem_dof(elem, 5, node_dof_index(elem.node_index_2, 3, ndf))
        elif elem.type_tag == ElementTypeTag.ForceBeamColumn2d:
            has_force_beam_column2d = True
            if ndm != 2 or ndf != 3:
                abort("forceBeamColumn2d requires ndm=2, ndf=3")
            if elem.node_count != 2:
                abort("forceBeamColumn2d requires 2 nodes")
            if (
                elem.geom_tag != GeomTransfTag.Linear
                and elem.geom_tag != GeomTransfTag.PDelta
            ):
                abort("forceBeamColumn2d supports geomTransf Linear or PDelta")
            if elem.integration != "Lobatto":
                abort("forceBeamColumn2d supports Lobatto integration only")
            if elem.num_int_pts != 3 and elem.num_int_pts != 5:
                abort("forceBeamColumn2d supports num_int_pts=3 or 5")
            if elem.section < 0 or elem.section >= len(typed_sections_by_id):
                abort("forceBeamColumn2d section not found")
            var sec = typed_sections_by_id[elem.section]
            if sec.id < 0:
                abort("forceBeamColumn2d section not found")
            if sec.type != "FiberSection2d" and sec.type != "ElasticSection2d":
                abort("forceBeamColumn2d requires FiberSection2d or ElasticSection2d")
            elem.dof_count = 6
            _set_elem_dof(elem, 0, node_dof_index(elem.node_index_1, 1, ndf))
            _set_elem_dof(elem, 1, node_dof_index(elem.node_index_1, 2, ndf))
            _set_elem_dof(elem, 2, node_dof_index(elem.node_index_1, 3, ndf))
            _set_elem_dof(elem, 3, node_dof_index(elem.node_index_2, 1, ndf))
            _set_elem_dof(elem, 4, node_dof_index(elem.node_index_2, 2, ndf))
            _set_elem_dof(elem, 5, node_dof_index(elem.node_index_2, 3, ndf))
        elif elem.type_tag == ElementTypeTag.DispBeamColumn2d:
            has_force_beam_column2d = True
            if ndm != 2 or ndf != 3:
                abort("dispBeamColumn2d requires ndm=2, ndf=3")
            if elem.node_count != 2:
                abort("dispBeamColumn2d requires 2 nodes")
            if (
                elem.geom_tag != GeomTransfTag.Linear
                and elem.geom_tag != GeomTransfTag.PDelta
            ):
                abort("dispBeamColumn2d supports geomTransf Linear or PDelta")
            if elem.integration != "Lobatto":
                abort("dispBeamColumn2d supports Lobatto integration only")
            if elem.num_int_pts != 3 and elem.num_int_pts != 5:
                abort("dispBeamColumn2d supports num_int_pts=3 or 5")
            if elem.section < 0 or elem.section >= len(typed_sections_by_id):
                abort("dispBeamColumn2d section not found")
            var sec = typed_sections_by_id[elem.section]
            if sec.id < 0:
                abort("dispBeamColumn2d section not found")
            if sec.type != "FiberSection2d" and sec.type != "ElasticSection2d":
                abort("dispBeamColumn2d requires FiberSection2d or ElasticSection2d")
            elem.dof_count = 6
            _set_elem_dof(elem, 0, node_dof_index(elem.node_index_1, 1, ndf))
            _set_elem_dof(elem, 1, node_dof_index(elem.node_index_1, 2, ndf))
            _set_elem_dof(elem, 2, node_dof_index(elem.node_index_1, 3, ndf))
            _set_elem_dof(elem, 3, node_dof_index(elem.node_index_2, 1, ndf))
            _set_elem_dof(elem, 4, node_dof_index(elem.node_index_2, 2, ndf))
            _set_elem_dof(elem, 5, node_dof_index(elem.node_index_2, 3, ndf))
        elif elem.type_tag == ElementTypeTag.ElasticBeamColumn3d:
            if ndm != 3 or ndf != 6:
                abort("elasticBeamColumn3d requires ndm=3, ndf=6")
            if elem.node_count != 2:
                abort("elasticBeamColumn3d requires 2 nodes")
            if elem.section < 0 or elem.section >= len(typed_sections_by_id):
                abort("section not found")
            var sec = typed_sections_by_id[elem.section]
            if sec.id < 0:
                abort("section not found")
            if sec.type != "ElasticSection3d":
                abort("elasticBeamColumn3d requires ElasticSection3d")
            if elem.section < len(fiber_section_index_by_id):
                if fiber_section_index_by_id[elem.section] >= 0:
                    abort(
                        "elasticBeamColumn3d with FiberSection2d requires forceBeamColumn2d "
                        "or dispBeamColumn2d"
                    )
            elem.dof_count = 12
            for d in range(6):
                _set_elem_dof(elem, d, node_dof_index(elem.node_index_1, d + 1, ndf))
                _set_elem_dof(elem, d + 6, node_dof_index(elem.node_index_2, d + 1, ndf))
        elif elem.type_tag == ElementTypeTag.Truss:
            if elem.node_count != 2:
                abort("truss requires 2 nodes")
            if ndf != 2 and ndf != 3:
                abort("truss requires ndf=2 or ndf=3")
            if elem.area <= 0.0:
                abort("truss requires area > 0")
            if ndf == 2:
                elem.dof_count = 4
                _set_elem_dof(elem, 0, node_dof_index(elem.node_index_1, 1, ndf))
                _set_elem_dof(elem, 1, node_dof_index(elem.node_index_1, 2, ndf))
                _set_elem_dof(elem, 2, node_dof_index(elem.node_index_2, 1, ndf))
                _set_elem_dof(elem, 3, node_dof_index(elem.node_index_2, 2, ndf))
            else:
                elem.dof_count = 6
                _set_elem_dof(elem, 0, node_dof_index(elem.node_index_1, 1, ndf))
                _set_elem_dof(elem, 1, node_dof_index(elem.node_index_1, 2, ndf))
                _set_elem_dof(elem, 2, node_dof_index(elem.node_index_1, 3, ndf))
                _set_elem_dof(elem, 3, node_dof_index(elem.node_index_2, 1, ndf))
                _set_elem_dof(elem, 4, node_dof_index(elem.node_index_2, 2, ndf))
                _set_elem_dof(elem, 5, node_dof_index(elem.node_index_2, 3, ndf))
        elif elem.type_tag == ElementTypeTag.Link:
            if ndf != 2:
                abort("zeroLength/twoNodeLink requires ndf=2")
            if elem.node_count != 2:
                abort("zeroLength/twoNodeLink requires 2 nodes")
            if elem.material_count != elem.dir_count:
                abort("zeroLength/twoNodeLink materials/dirs mismatch")
            for d in range(elem.dir_count):
                var dir = _elem_dir(elem, d)
                if dir != LinkDirectionTag.UX and dir != LinkDirectionTag.UY:
                    abort("unsupported link dir")
            elem.dof_count = 4
            _set_elem_dof(elem, 0, node_dof_index(elem.node_index_1, 1, ndf))
            _set_elem_dof(elem, 1, node_dof_index(elem.node_index_1, 2, ndf))
            _set_elem_dof(elem, 2, node_dof_index(elem.node_index_2, 1, ndf))
            _set_elem_dof(elem, 3, node_dof_index(elem.node_index_2, 2, ndf))
        elif elem.type_tag == ElementTypeTag.ZeroLengthSection:
            if ndm != 2 or ndf != 3:
                abort("zeroLengthSection requires ndm=2, ndf=3")
            if elem.node_count != 2:
                abort("zeroLengthSection requires 2 nodes")
            if elem.section < 0 or elem.section >= len(typed_sections_by_id):
                abort("zeroLengthSection section not found")
            var sec = typed_sections_by_id[elem.section]
            if sec.id < 0:
                abort("zeroLengthSection section not found")
            if sec.type != "FiberSection2d" and sec.type != "ElasticSection2d":
                abort("zeroLengthSection requires FiberSection2d or ElasticSection2d")
            elem.dof_count = 6
            _set_elem_dof(elem, 0, node_dof_index(elem.node_index_1, 1, ndf))
            _set_elem_dof(elem, 1, node_dof_index(elem.node_index_1, 2, ndf))
            _set_elem_dof(elem, 2, node_dof_index(elem.node_index_1, 3, ndf))
            _set_elem_dof(elem, 3, node_dof_index(elem.node_index_2, 1, ndf))
            _set_elem_dof(elem, 4, node_dof_index(elem.node_index_2, 2, ndf))
            _set_elem_dof(elem, 5, node_dof_index(elem.node_index_2, 3, ndf))
        elif elem.type_tag == ElementTypeTag.FourNodeQuad:
            if ndm != 2 or ndf != 2:
                abort("fourNodeQuad requires ndm=2, ndf=2")
            if elem.node_count != 4:
                abort("fourNodeQuad requires 4 nodes")
            if elem.formulation != "PlaneStress":
                abort("fourNodeQuad only supports PlaneStress formulation")
            if elem.material < 0 or elem.material >= len(typed_materials_by_id):
                abort("material not found")
            var mat = typed_materials_by_id[elem.material]
            if mat.id < 0:
                abort("material not found")
            if mat.type != "ElasticIsotropic":
                abort("fourNodeQuad requires ElasticIsotropic material")
            if elem.thickness <= 0.0:
                abort("fourNodeQuad requires thickness > 0")
            elem.dof_count = 8
            _set_elem_dof(elem, 0, node_dof_index(elem.node_index_1, 1, ndf))
            _set_elem_dof(elem, 1, node_dof_index(elem.node_index_1, 2, ndf))
            _set_elem_dof(elem, 2, node_dof_index(elem.node_index_2, 1, ndf))
            _set_elem_dof(elem, 3, node_dof_index(elem.node_index_2, 2, ndf))
            _set_elem_dof(elem, 4, node_dof_index(elem.node_index_3, 1, ndf))
            _set_elem_dof(elem, 5, node_dof_index(elem.node_index_3, 2, ndf))
            _set_elem_dof(elem, 6, node_dof_index(elem.node_index_4, 1, ndf))
            _set_elem_dof(elem, 7, node_dof_index(elem.node_index_4, 2, ndf))
        elif elem.type_tag == ElementTypeTag.Shell:
            if ndm != 3 or ndf != 6:
                abort("shell requires ndm=3, ndf=6")
            if elem.node_count != 4:
                abort("shell requires 4 nodes")
            if elem.section < 0 or elem.section >= len(typed_sections_by_id):
                abort("section not found")
            var sec = typed_sections_by_id[elem.section]
            if sec.id < 0:
                abort("section not found")
            if sec.type != "ElasticMembranePlateSection":
                abort("shell requires ElasticMembranePlateSection")
            elem.dof_count = 24
            for d in range(6):
                _set_elem_dof(elem, d, node_dof_index(elem.node_index_1, d + 1, ndf))
                _set_elem_dof(elem, d + 6, node_dof_index(elem.node_index_2, d + 1, ndf))
                _set_elem_dof(elem, d + 12, node_dof_index(elem.node_index_3, d + 1, ndf))
                _set_elem_dof(elem, d + 18, node_dof_index(elem.node_index_4, d + 1, ndf))
        else:
            abort("unsupported element type: " + elem.type)

        typed_elements[i] = elem

    var uniaxial_states: List[UniMaterialState] = []
    var uniaxial_state_defs: List[Int] = []
    var elem_uniaxial_offsets: List[Int] = []
    var elem_uniaxial_counts: List[Int] = []
    var elem_uniaxial_state_ids: List[Int] = []
    var force_basic_offsets: List[Int] = []
    var force_basic_counts: List[Int] = []
    var force_basic_q: List[Float64] = []
    elem_uniaxial_offsets.resize(elem_count, 0)
    elem_uniaxial_counts.resize(elem_count, 0)
    force_basic_offsets.resize(elem_count, 0)
    force_basic_counts.resize(elem_count, 0)
    var used_nonelastic_uniaxial = False
    var force_beam_has_nonelastic = False
    for e in range(elem_count):
        var elem = typed_elements[e]
        force_basic_offsets[e] = len(force_basic_q)
        force_basic_counts[e] = 0
        if elem.type_tag == ElementTypeTag.Truss:
            var mat_id = elem.material
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
        elif elem.type_tag == ElementTypeTag.Link:
            elem_uniaxial_offsets[e] = len(elem_uniaxial_state_ids)
            elem_uniaxial_counts[e] = elem.material_count
            for m in range(elem.material_count):
                var mat_id = _elem_material(elem, m)
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
        elif elem.type_tag == ElementTypeTag.ForceBeamColumn2d:
            force_basic_offsets[e] = len(force_basic_q)
            force_basic_counts[e] = 3
            force_basic_q.append(0.0)
            force_basic_q.append(0.0)
            force_basic_q.append(0.0)
            var sec_id = elem.section
            elem_uniaxial_offsets[e] = len(elem_uniaxial_state_ids)
            var sec = typed_sections_by_id[sec_id]
            var beam_col_type = elem.type
            if sec.type == "FiberSection2d":
                var sec_index = fiber_section_index_by_id[sec_id]
                if sec_index < 0 or sec_index >= len(fiber_section_defs):
                    abort(beam_col_type + " fiber section not found")
                var sec_def = fiber_section_defs[sec_index]
                var num_int_pts = elem.num_int_pts
                var state_count = num_int_pts * sec_def.fiber_count
                elem_uniaxial_counts[e] = state_count
                for _ in range(num_int_pts):
                    for i in range(sec_def.fiber_count):
                        var cell = fiber_section_cells[sec_def.fiber_offset + i]
                        var def_index = cell.def_index
                        if def_index < 0 or def_index >= len(uniaxial_defs):
                            abort(beam_col_type + " fiber material definition out of range")
                        var mat_def = uniaxial_defs[def_index]
                        var state_index = len(uniaxial_states)
                        uniaxial_states.append(UniMaterialState(mat_def))
                        uniaxial_state_defs.append(def_index)
                        elem_uniaxial_state_ids.append(state_index)
                        if not uni_mat_is_elastic(mat_def):
                            used_nonelastic_uniaxial = True
                            force_beam_has_nonelastic = True
            elif sec.type == "ElasticSection2d":
                elem_uniaxial_counts[e] = 0
            else:
                abort(beam_col_type + " requires FiberSection2d or ElasticSection2d")
        elif elem.type_tag == ElementTypeTag.DispBeamColumn2d:
            var sec_id = elem.section
            elem_uniaxial_offsets[e] = len(elem_uniaxial_state_ids)
            var sec = typed_sections_by_id[sec_id]
            var beam_col_type = elem.type
            if sec.type == "FiberSection2d":
                var sec_index = fiber_section_index_by_id[sec_id]
                if sec_index < 0 or sec_index >= len(fiber_section_defs):
                    abort(beam_col_type + " fiber section not found")
                var sec_def = fiber_section_defs[sec_index]
                var num_int_pts = elem.num_int_pts
                var state_count = num_int_pts * sec_def.fiber_count
                elem_uniaxial_counts[e] = state_count
                for _ in range(num_int_pts):
                    for i in range(sec_def.fiber_count):
                        var cell = fiber_section_cells[sec_def.fiber_offset + i]
                        var def_index = cell.def_index
                        if def_index < 0 or def_index >= len(uniaxial_defs):
                            abort(beam_col_type + " fiber material definition out of range")
                        var mat_def = uniaxial_defs[def_index]
                        var state_index = len(uniaxial_states)
                        uniaxial_states.append(UniMaterialState(mat_def))
                        uniaxial_state_defs.append(def_index)
                        elem_uniaxial_state_ids.append(state_index)
                        if not uni_mat_is_elastic(mat_def):
                            used_nonelastic_uniaxial = True
                            force_beam_has_nonelastic = True
            elif sec.type == "ElasticSection2d":
                elem_uniaxial_counts[e] = 0
            else:
                abort(beam_col_type + " requires FiberSection2d or ElasticSection2d")
        elif elem.type_tag == ElementTypeTag.ZeroLengthSection:
            var sec_id = elem.section
            elem_uniaxial_offsets[e] = len(elem_uniaxial_state_ids)
            var sec = typed_sections_by_id[sec_id]
            if sec.type == "FiberSection2d":
                var sec_index = fiber_section_index_by_id[sec_id]
                if sec_index < 0 or sec_index >= len(fiber_section_defs):
                    abort("zeroLengthSection fiber section not found")
                var sec_def = fiber_section_defs[sec_index]
                var state_count = sec_def.fiber_count
                elem_uniaxial_counts[e] = state_count
                for i in range(sec_def.fiber_count):
                    var cell = fiber_section_cells[sec_def.fiber_offset + i]
                    var def_index = cell.def_index
                    if def_index < 0 or def_index >= len(uniaxial_defs):
                        abort("zeroLengthSection fiber material definition out of range")
                    var mat_def = uniaxial_defs[def_index]
                    var state_index = len(uniaxial_states)
                    uniaxial_states.append(UniMaterialState(mat_def))
                    uniaxial_state_defs.append(def_index)
                    elem_uniaxial_state_ids.append(state_index)
                    if not uni_mat_is_elastic(mat_def):
                        used_nonelastic_uniaxial = True
            elif sec.type == "ElasticSection2d":
                elem_uniaxial_counts[e] = 0
            else:
                abort("zeroLengthSection requires FiberSection2d or ElasticSection2d")
        else:
            elem_uniaxial_offsets[e] = len(elem_uniaxial_state_ids)
            elem_uniaxial_counts[e] = 0

    var total_dofs = node_count * ndf
    var F_total: List[Float64] = []
    F_total.resize(total_dofs, 0.0)

    for i in range(len(input.element_loads)):
        var load = input.element_loads[i]
        var elem_id = load.element
        if elem_id >= len(elem_id_to_index) or elem_id_to_index[elem_id] < 0:
            abort("element load refers to unknown element")
        var elem_index = elem_id_to_index[elem_id]
        var elem = typed_elements[elem_index]
        var elem_type = elem.type
        var load_type = load.type
        if load_type == "beamUniform":
            if elem_type != "elasticBeamColumn2d":
                abort("beamUniform requires elasticBeamColumn2d")
            if ndf != 3:
                abort("beamUniform requires ndf=3")
            var i1 = elem.node_index_1
            var i2 = elem.node_index_2
            var node1 = input.nodes[i1]
            var node2 = input.nodes[i2]
            var f_global = beam_uniform_load_global(
                node1.x,
                node1.y,
                node2.x,
                node2.y,
                load.w,
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
            elem.uniform_load_w += load.w
            typed_elements[elem_index] = elem
        else:
            abort("unsupported element load type")

    for i in range(len(input.loads)):
        var load = input.loads[i]
        var node_id = load.node
        var dof = load.dof
        require_dof_in_range(dof, ndf, "load")
        var idx = node_dof_index(id_to_index[node_id], dof, ndf)
        F_total[idx] += load.value

    var M_total: List[Float64] = []
    M_total.resize(total_dofs, 0.0)
    for i in range(len(input.masses)):
        var mass = input.masses[i]
        var node_id = mass.node
        var dof = mass.dof
        require_dof_in_range(dof, ndf, "mass")
        var idx = node_dof_index(id_to_index[node_id], dof, ndf)
        M_total[idx] += mass.value

    # Lump translational mass from fourNodeQuad/bbarQuad density when provided.
    for e in range(elem_count):
        var elem = typed_elements[e]
        if elem.type_tag != ElementTypeTag.FourNodeQuad:
            continue
        if elem.material < 0 or elem.material >= len(typed_materials_by_id):
            abort("material not found")
        var mat = typed_materials_by_id[elem.material]
        if mat.id < 0:
            abort("material not found")
        var rho = mat.rho
        if rho == 0.0:
            continue

        var n1 = input.nodes[elem.node_index_1]
        var n2 = input.nodes[elem.node_index_2]
        var n3 = input.nodes[elem.node_index_3]
        var n4 = input.nodes[elem.node_index_4]
        var twice_area = (
            n1.x * n2.y
            - n1.y * n2.x
            + n2.x * n3.y
            - n2.y * n3.x
            + n3.x * n4.y
            - n3.y * n4.x
            + n4.x * n1.y
            - n4.y * n1.x
        )
        var area = abs(twice_area) * 0.5
        if area <= 0.0:
            abort("fourNodeQuad area must be > 0")
        var lumped = rho * elem.thickness * area / 4.0
        if lumped == 0.0:
            continue
        M_total[node_dof_index(elem.node_index_1, 1, ndf)] += lumped
        M_total[node_dof_index(elem.node_index_1, 2, ndf)] += lumped
        M_total[node_dof_index(elem.node_index_2, 1, ndf)] += lumped
        M_total[node_dof_index(elem.node_index_2, 2, ndf)] += lumped
        M_total[node_dof_index(elem.node_index_3, 1, ndf)] += lumped
        M_total[node_dof_index(elem.node_index_3, 2, ndf)] += lumped
        M_total[node_dof_index(elem.node_index_4, 1, ndf)] += lumped
        M_total[node_dof_index(elem.node_index_4, 2, ndf)] += lumped

    var constrained: List[Bool] = []
    constrained.resize(total_dofs, False)
    for i in range(node_count):
        var node = input.nodes[i]
        for j in range(node.constraint_count):
            if j == 0:
                var dof = node.constraint_1
                require_dof_in_range(dof, ndf, "constraint")
                var idx = node_dof_index(i, dof, ndf)
                constrained[idx] = True
            elif j == 1:
                var dof = node.constraint_2
                require_dof_in_range(dof, ndf, "constraint")
                var idx = node_dof_index(i, dof, ndf)
                constrained[idx] = True
            elif j == 2:
                var dof = node.constraint_3
                require_dof_in_range(dof, ndf, "constraint")
                var idx = node_dof_index(i, dof, ndf)
                constrained[idx] = True
            elif j == 3:
                var dof = node.constraint_4
                require_dof_in_range(dof, ndf, "constraint")
                var idx = node_dof_index(i, dof, ndf)
                constrained[idx] = True
            elif j == 4:
                var dof = node.constraint_5
                require_dof_in_range(dof, ndf, "constraint")
                var idx = node_dof_index(i, dof, ndf)
                constrained[idx] = True
            else:
                var dof = node.constraint_6
                require_dof_in_range(dof, ndf, "constraint")
                var idx = node_dof_index(i, dof, ndf)
                constrained[idx] = True

    var analysis_input = input.analysis
    var analysis_type = analysis_input.type
    var constraints_handler = analysis_input.constraints
    if constraints_handler != "Plain" and constraints_handler != "Transformation":
        abort("unsupported constraints handler: " + constraints_handler)
    var steps = analysis_input.steps
    var modal_num_modes = 0
    if analysis_type == "modal_eigen":
        modal_num_modes = analysis_input.num_modes
        if modal_num_modes < 1:
            abort("modal_eigen requires num_modes >= 1")
    if steps < 1:
        abort("analysis steps must be >= 1")
    var force_beam_mode = analysis_input.force_beam_mode
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
                "forceBeamColumn2d/dispBeamColumn2d requires static_linear, "
                "static_nonlinear, or transient_nonlinear analysis"
            )
        if analysis_type == "transient_nonlinear":
            if force_beam_mode != "auto":
                abort(
                    "force_beam_mode is only supported for static forceBeamColumn2d/"
                    "dispBeamColumn2d analyses"
                )
        else:
            if force_beam_mode == "nonlinear":
                if analysis_type != "static_nonlinear":
                    abort("force_beam_mode=nonlinear requires static_nonlinear analysis")
            elif force_beam_mode == "linear_if_elastic":
                if force_beam_has_nonelastic:
                    if analysis_type != "static_nonlinear":
                        abort(
                            "forceBeamColumn2d/dispBeamColumn2d with non-elastic fibers "
                            "requires static_nonlinear analysis"
                        )
                elif analysis_type != "static_linear":
                    abort(
                        "force_beam_mode=linear_if_elastic requires static_linear "
                        "analysis for elastic forceBeamColumn2d/dispBeamColumn2d"
                    )
            elif analysis_type == "static_linear" and force_beam_has_nonelastic:
                abort(
                    "forceBeamColumn2d/dispBeamColumn2d with non-elastic fibers requires "
                    "static_nonlinear analysis"
                )
    if (
        analysis_type != "static_nonlinear"
        and analysis_type != "transient_nonlinear"
        and analysis_type != "modal_eigen"
        and used_nonelastic_uniaxial
    ):
        abort("nonlinear uniaxial materials require static_nonlinear or transient_nonlinear analysis")

    var solver_pref = analysis_input.solver
    if solver_pref == "":
        solver_pref = "auto"
    if solver_pref != "auto" and solver_pref != "dense" and solver_pref != "banded":
        abort("unsupported analysis system: " + solver_pref)
    var band_threshold = analysis_input.band_threshold
    if band_threshold < 0:
        band_threshold = 0

    var rep_dof: List[Int] = []
    rep_dof.resize(total_dofs, 0)
    for i in range(total_dofs):
        rep_dof[i] = i

    var has_transformation_mpc = False
    if len(input.mp_constraints) > 0:
        if constraints_handler != "Transformation":
            abort("mp_constraints require analysis.constraints=Transformation")
        has_transformation_mpc = True
    for i in range(len(input.mp_constraints)):
        var mpc = input.mp_constraints[i]
        var mpc_type = mpc.type
        if mpc_type != "equalDOF":
            abort("unsupported mp constraint type: " + mpc_type)
        var retained_node = mpc.retained_node
        var constrained_node = mpc.constrained_node
        if retained_node >= len(id_to_index) or id_to_index[retained_node] < 0:
            abort("equalDOF retained_node not found")
        if constrained_node >= len(id_to_index) or id_to_index[constrained_node] < 0:
            abort("equalDOF constrained_node not found")
        var retained_idx = id_to_index[retained_node]
        var constrained_idx = id_to_index[constrained_node]
        if mpc.dof_count == 0:
            abort("equalDOF requires non-empty dofs")
        for j in range(mpc.dof_count):
            if j == 0:
                var dof = mpc.dof_1
                require_dof_in_range(dof, ndf, "equalDOF")
                var retained_dof = node_dof_index(retained_idx, dof, ndf)
                var constrained_dof = node_dof_index(constrained_idx, dof, ndf)
                rep_dof[constrained_dof] = retained_dof
            elif j == 1:
                var dof = mpc.dof_2
                require_dof_in_range(dof, ndf, "equalDOF")
                var retained_dof = node_dof_index(retained_idx, dof, ndf)
                var constrained_dof = node_dof_index(constrained_idx, dof, ndf)
                rep_dof[constrained_dof] = retained_dof
            elif j == 2:
                var dof = mpc.dof_3
                require_dof_in_range(dof, ndf, "equalDOF")
                var retained_dof = node_dof_index(retained_idx, dof, ndf)
                var constrained_dof = node_dof_index(constrained_idx, dof, ndf)
                rep_dof[constrained_dof] = retained_dof
            elif j == 3:
                var dof = mpc.dof_4
                require_dof_in_range(dof, ndf, "equalDOF")
                var retained_dof = node_dof_index(retained_idx, dof, ndf)
                var constrained_dof = node_dof_index(constrained_idx, dof, ndf)
                rep_dof[constrained_dof] = retained_dof
            elif j == 4:
                var dof = mpc.dof_5
                require_dof_in_range(dof, ndf, "equalDOF")
                var retained_dof = node_dof_index(retained_idx, dof, ndf)
                var constrained_dof = node_dof_index(constrained_idx, dof, ndf)
                rep_dof[constrained_dof] = retained_dof
            else:
                var dof = mpc.dof_6
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
        var adjacency = build_node_adjacency_typed(typed_elements, node_count)
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

    var time_series: List[TimeSeriesInput] = []
    var time_series_values: List[Float64] = []
    var time_series_times: List[Float64] = []
    parse_time_series_inputs(data, time_series, time_series_values, time_series_times)
    var ts_index = -1
    var pattern_input = input.pattern
    var pattern_type = "Plain"
    var uniform_excitation_direction = 0
    var uniform_accel_ts_index = -1
    if pattern_input.has_pattern:
        pattern_type = pattern_input.type
        if pattern_type != "Plain" and pattern_type != "UniformExcitation":
            abort("unsupported pattern type: " + pattern_type)
    if len(time_series) > 0 or pattern_input.has_pattern:
        if not pattern_input.has_pattern:
            if len(time_series) == 1:
                var ts_tag = time_series[0].tag
                ts_index = find_time_series_input(time_series, ts_tag)
                if ts_index < 0:
                    abort("time_series tag not found")
            else:
                abort("pattern missing for multiple time_series")
        elif pattern_type == "Plain":
            if not pattern_input.has_time_series:
                abort("pattern missing time_series")
            var ts_tag = pattern_input.time_series
            ts_index = find_time_series_input(time_series, ts_tag)
            if ts_index < 0:
                abort("time_series tag not found")
        else:
            if analysis_type != "transient_linear" and analysis_type != "transient_nonlinear":
                abort("UniformExcitation requires transient analysis")
            if not pattern_input.has_direction:
                abort("UniformExcitation pattern missing direction")
            uniform_excitation_direction = pattern_input.direction
            if uniform_excitation_direction < 1 or uniform_excitation_direction > ndm:
                abort("UniformExcitation direction out of range 1..ndm")
            var accel_tag = -1
            if pattern_input.has_accel:
                accel_tag = pattern_input.accel
            elif pattern_input.has_time_series:
                accel_tag = pattern_input.time_series
            else:
                abort("UniformExcitation pattern missing accel time_series tag")
            uniform_accel_ts_index = find_time_series_input(time_series, accel_tag)
            if uniform_accel_ts_index < 0:
                abort("UniformExcitation accel time_series tag not found")
            if len(input.loads) > 0 or len(input.element_loads) > 0:
                abort("UniformExcitation does not support nodal/element loads")

    var rayleigh_alpha_m = 0.0
    var rayleigh_beta_k = 0.0
    var rayleigh_beta_k_init = 0.0
    var rayleigh_beta_k_comm = 0.0
    if input.rayleigh.has_rayleigh:
        rayleigh_alpha_m = input.rayleigh.alpha_m
        rayleigh_beta_k = input.rayleigh.beta_k
        rayleigh_beta_k_init = input.rayleigh.beta_k_init
        rayleigh_beta_k_comm = input.rayleigh.beta_k_comm

    state.ndm = ndm
    state.ndf = ndf
    state.typed_nodes = input.nodes.copy()
    state.node_count = node_count
    state.id_to_index = id_to_index^
    state.typed_sections_by_id = typed_sections_by_id^
    state.typed_materials_by_id = typed_materials_by_id^
    state.uniaxial_defs = uniaxial_defs^
    state.uniaxial_state_defs = uniaxial_state_defs^
    state.uniaxial_states = uniaxial_states^
    state.fiber_section_defs = fiber_section_defs^
    state.fiber_section_cells = fiber_section_cells^
    state.fiber_section_index_by_id = fiber_section_index_by_id^
    state.typed_elements = typed_elements^
    state.elem_count = elem_count
    state.elem_id_to_index = elem_id_to_index^
    state.elem_uniaxial_offsets = elem_uniaxial_offsets^
    state.elem_uniaxial_counts = elem_uniaxial_counts^
    state.elem_uniaxial_state_ids = elem_uniaxial_state_ids^
    state.force_basic_offsets = force_basic_offsets^
    state.force_basic_counts = force_basic_counts^
    state.force_basic_q = force_basic_q^
    state.total_dofs = total_dofs
    state.F_total = F_total^
    state.constrained = constrained^
    state.analysis = analysis_input
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
    state.analysis_integrator_targets_pool = (
        input.analysis_integrator_targets_pool.copy()
    )
    state.time_series = time_series^
    state.time_series_values = time_series_values^
    state.time_series_times = time_series_times^
    state.ts_index = ts_index
    state.pattern_type = pattern_type
    state.uniform_excitation_direction = uniform_excitation_direction
    state.uniform_accel_ts_index = uniform_accel_ts_index
    state.rayleigh_alpha_m = rayleigh_alpha_m
    state.rayleigh_beta_k = rayleigh_beta_k
    state.rayleigh_beta_k_init = rayleigh_beta_k_init
    state.rayleigh_beta_k_comm = rayleigh_beta_k_comm
    state.recorder_nodes_pool = input.recorder_nodes_pool.copy()
    state.recorder_elements_pool = input.recorder_elements_pool.copy()
    state.recorder_dofs_pool = input.recorder_dofs_pool.copy()
    state.recorder_modes_pool = input.recorder_modes_pool.copy()
    state.recorders = input.recorders.copy()

    return state^
