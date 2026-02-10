from collections import List
from os import abort
from python import Python, PythonObject

from strut_io import py_len
from tag_types import ElementTypeTag, GeomTransfTag, RecorderTypeTag


struct ModelInput(Movable, ImplicitlyCopyable):
    var ndm: Int
    var ndf: Int

    fn __init__(out self, ndm: Int, ndf: Int):
        self.ndm = ndm
        self.ndf = ndf


struct NodeInput(Movable, ImplicitlyCopyable):
    var id: Int
    var x: Float64
    var y: Float64
    var z: Float64
    var has_z: Bool
    var constraint_count: Int
    var constraint_1: Int
    var constraint_2: Int
    var constraint_3: Int
    var constraint_4: Int
    var constraint_5: Int
    var constraint_6: Int

    fn __init__(out self):
        self.id = 0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.has_z = False
        self.constraint_count = 0
        self.constraint_1 = 0
        self.constraint_2 = 0
        self.constraint_3 = 0
        self.constraint_4 = 0
        self.constraint_5 = 0
        self.constraint_6 = 0

    fn __init__(out self, id: Int, x: Float64, y: Float64, z: Float64, has_z: Bool):
        self.id = id
        self.x = x
        self.y = y
        self.z = z
        self.has_z = has_z
        self.constraint_count = 0
        self.constraint_1 = 0
        self.constraint_2 = 0
        self.constraint_3 = 0
        self.constraint_4 = 0
        self.constraint_5 = 0
        self.constraint_6 = 0


struct SectionInput(Movable, ImplicitlyCopyable):
    var id: Int
    var type: String
    var E: Float64
    var A: Float64
    var I: Float64
    var Iz: Float64
    var Iy: Float64
    var G: Float64
    var J: Float64
    var nu: Float64
    var h: Float64

    fn __init__(out self):
        self.id = -1
        self.type = ""
        self.E = 0.0
        self.A = 0.0
        self.I = 0.0
        self.Iz = 0.0
        self.Iy = 0.0
        self.G = 0.0
        self.J = 0.0
        self.nu = 0.0
        self.h = 0.0

    fn __init__(out self, id: Int, type: String):
        self.id = id
        self.type = type
        self.E = 0.0
        self.A = 0.0
        self.I = 0.0
        self.Iz = 0.0
        self.Iy = 0.0
        self.G = 0.0
        self.J = 0.0
        self.nu = 0.0
        self.h = 0.0


struct MaterialInput(Movable, ImplicitlyCopyable):
    var id: Int
    var type: String

    var E: Float64
    var Fy: Float64
    var E0: Float64
    var b: Float64
    var fpc: Float64
    var epsc0: Float64
    var fpcu: Float64
    var epscu: Float64
    var R0: Float64
    var cR1: Float64
    var cR2: Float64
    var a1: Float64
    var a2: Float64
    var a3: Float64
    var a4: Float64
    var sigInit: Float64
    var rat: Float64
    var ft: Float64
    var Ets: Float64
    var nu: Float64
    var rho: Float64

    var has_r0: Bool
    var has_cr1: Bool
    var has_cr2: Bool
    var has_a1: Bool
    var has_a2: Bool
    var has_a3: Bool
    var has_a4: Bool
    var has_siginit: Bool
    var has_rat: Bool
    var has_ft: Bool
    var has_ets: Bool

    fn __init__(out self):
        self.id = -1
        self.type = ""
        self.E = 0.0
        self.Fy = 0.0
        self.E0 = 0.0
        self.b = 0.0
        self.fpc = 0.0
        self.epsc0 = 0.0
        self.fpcu = 0.0
        self.epscu = 0.0
        self.R0 = 0.0
        self.cR1 = 0.0
        self.cR2 = 0.0
        self.a1 = 0.0
        self.a2 = 0.0
        self.a3 = 0.0
        self.a4 = 0.0
        self.sigInit = 0.0
        self.rat = 0.0
        self.ft = 0.0
        self.Ets = 0.0
        self.nu = 0.0
        self.rho = 0.0
        self.has_r0 = False
        self.has_cr1 = False
        self.has_cr2 = False
        self.has_a1 = False
        self.has_a2 = False
        self.has_a3 = False
        self.has_a4 = False
        self.has_siginit = False
        self.has_rat = False
        self.has_ft = False
        self.has_ets = False


struct ElementInput(Movable, ImplicitlyCopyable):
    var id: Int
    var type: String
    var type_tag: Int
    var geom_tag: Int
    var node_count: Int
    var node_1: Int
    var node_2: Int
    var node_3: Int
    var node_4: Int
    var node_index_1: Int
    var node_index_2: Int
    var node_index_3: Int
    var node_index_4: Int
    var section: Int
    var material: Int
    var material_count: Int
    var material_1: Int
    var material_2: Int
    var material_3: Int
    var material_4: Int
    var material_5: Int
    var material_6: Int
    var dir_count: Int
    var dir_1: Int
    var dir_2: Int
    var dir_3: Int
    var dir_4: Int
    var dir_5: Int
    var dir_6: Int
    var area: Float64
    var thickness: Float64
    var formulation: String
    var geom_transf: String
    var integration: String
    var num_int_pts: Int
    var uniform_load_w: Float64
    var dof_count: Int
    var dof_1: Int
    var dof_2: Int
    var dof_3: Int
    var dof_4: Int
    var dof_5: Int
    var dof_6: Int
    var dof_7: Int
    var dof_8: Int
    var dof_9: Int
    var dof_10: Int
    var dof_11: Int
    var dof_12: Int
    var dof_13: Int
    var dof_14: Int
    var dof_15: Int
    var dof_16: Int
    var dof_17: Int
    var dof_18: Int
    var dof_19: Int
    var dof_20: Int
    var dof_21: Int
    var dof_22: Int
    var dof_23: Int
    var dof_24: Int

    fn __init__(out self):
        self.id = 0
        self.type = ""
        self.type_tag = ElementTypeTag.Unknown
        self.geom_tag = GeomTransfTag.Unknown
        self.node_count = 0
        self.node_1 = -1
        self.node_2 = -1
        self.node_3 = -1
        self.node_4 = -1
        self.node_index_1 = -1
        self.node_index_2 = -1
        self.node_index_3 = -1
        self.node_index_4 = -1
        self.section = -1
        self.material = -1
        self.material_count = 0
        self.material_1 = -1
        self.material_2 = -1
        self.material_3 = -1
        self.material_4 = -1
        self.material_5 = -1
        self.material_6 = -1
        self.dir_count = 0
        self.dir_1 = 0
        self.dir_2 = 0
        self.dir_3 = 0
        self.dir_4 = 0
        self.dir_5 = 0
        self.dir_6 = 0
        self.area = 0.0
        self.thickness = 0.0
        self.formulation = "PlaneStress"
        self.geom_transf = "Linear"
        self.integration = "Lobatto"
        self.num_int_pts = 3
        self.uniform_load_w = 0.0
        self.dof_count = 0
        self.dof_1 = -1
        self.dof_2 = -1
        self.dof_3 = -1
        self.dof_4 = -1
        self.dof_5 = -1
        self.dof_6 = -1
        self.dof_7 = -1
        self.dof_8 = -1
        self.dof_9 = -1
        self.dof_10 = -1
        self.dof_11 = -1
        self.dof_12 = -1
        self.dof_13 = -1
        self.dof_14 = -1
        self.dof_15 = -1
        self.dof_16 = -1
        self.dof_17 = -1
        self.dof_18 = -1
        self.dof_19 = -1
        self.dof_20 = -1
        self.dof_21 = -1
        self.dof_22 = -1
        self.dof_23 = -1
        self.dof_24 = -1


struct ElementLoadInput(Movable, ImplicitlyCopyable):
    var element: Int
    var type: String
    var w: Float64

    fn __init__(out self, element: Int, type: String, w: Float64):
        self.element = element
        self.type = type
        self.w = w


struct NodalLoadInput(Movable, ImplicitlyCopyable):
    var node: Int
    var dof: Int
    var value: Float64

    fn __init__(out self, node: Int, dof: Int, value: Float64):
        self.node = node
        self.dof = dof
        self.value = value


struct MassInput(Movable, ImplicitlyCopyable):
    var node: Int
    var dof: Int
    var value: Float64

    fn __init__(out self, node: Int, dof: Int, value: Float64):
        self.node = node
        self.dof = dof
        self.value = value


struct AnalysisInput(Movable, ImplicitlyCopyable):
    var type: String
    var constraints: String
    var steps: Int
    var num_modes: Int
    var force_beam_mode: String
    var solver: String
    var band_threshold: Int
    var dt: Float64
    var algorithm: String
    var max_iters: Int
    var tol: Float64
    var rel_tol: Float64
    var fallback_algorithm: String
    var test_type: String
    var fallback_test_type: String
    var fallback_max_iters: Int
    var fallback_tol: Float64
    var fallback_rel_tol: Float64
    var integrator_type: String
    var integrator_gamma: Float64
    var integrator_beta: Float64
    var integrator_node: Int
    var integrator_dof: Int
    var integrator_cutback: Float64
    var integrator_max_cutbacks: Int
    var integrator_min_du: Float64
    var has_integrator_du: Bool
    var integrator_du: Float64
    var integrator_targets_offset: Int
    var integrator_targets_count: Int

    fn __init__(out self):
        self.type = "static_linear"
        self.constraints = "Plain"
        self.steps = 1
        self.num_modes = 0
        self.force_beam_mode = "auto"
        self.solver = "auto"
        self.band_threshold = 128
        self.dt = 0.0
        self.algorithm = "Newton"
        self.max_iters = 20
        self.tol = 1.0e-10
        self.rel_tol = 1.0e-8
        self.fallback_algorithm = ""
        self.test_type = "MaxDispIncr"
        self.fallback_test_type = "MaxDispIncr"
        self.fallback_max_iters = 20
        self.fallback_tol = 1.0e-10
        self.fallback_rel_tol = 1.0e-8
        self.integrator_type = ""
        self.integrator_gamma = 0.5
        self.integrator_beta = 0.25
        self.integrator_node = -1
        self.integrator_dof = -1
        self.integrator_cutback = 0.5
        self.integrator_max_cutbacks = 8
        self.integrator_min_du = 1.0e-10
        self.has_integrator_du = False
        self.integrator_du = 0.0
        self.integrator_targets_offset = 0
        self.integrator_targets_count = 0


struct MPConstraintInput(Movable, ImplicitlyCopyable):
    var type: String
    var retained_node: Int
    var constrained_node: Int
    var dof_count: Int
    var dof_1: Int
    var dof_2: Int
    var dof_3: Int
    var dof_4: Int
    var dof_5: Int
    var dof_6: Int

    fn __init__(out self):
        self.type = ""
        self.retained_node = -1
        self.constrained_node = -1
        self.dof_count = 0
        self.dof_1 = 0
        self.dof_2 = 0
        self.dof_3 = 0
        self.dof_4 = 0
        self.dof_5 = 0
        self.dof_6 = 0

    fn __init__(out self, type: String, retained_node: Int, constrained_node: Int):
        self.type = type
        self.retained_node = retained_node
        self.constrained_node = constrained_node
        self.dof_count = 0
        self.dof_1 = 0
        self.dof_2 = 0
        self.dof_3 = 0
        self.dof_4 = 0
        self.dof_5 = 0
        self.dof_6 = 0


struct PatternInput(Movable, ImplicitlyCopyable):
    var has_pattern: Bool
    var type: String
    var has_time_series: Bool
    var time_series: Int
    var has_direction: Bool
    var direction: Int
    var has_accel: Bool
    var accel: Int

    fn __init__(out self):
        self.has_pattern = False
        self.type = "Plain"
        self.has_time_series = False
        self.time_series = -1
        self.has_direction = False
        self.direction = 0
        self.has_accel = False
        self.accel = -1


struct RayleighInput(Movable, ImplicitlyCopyable):
    var has_rayleigh: Bool
    var alpha_m: Float64
    var beta_k: Float64
    var beta_k_init: Float64
    var beta_k_comm: Float64

    fn __init__(out self):
        self.has_rayleigh = False
        self.alpha_m = 0.0
        self.beta_k = 0.0
        self.beta_k_init = 0.0
        self.beta_k_comm = 0.0


struct RecorderInput(Movable, ImplicitlyCopyable):
    var type_tag: Int
    var output: String
    var node_offset: Int
    var node_count: Int
    var element_offset: Int
    var element_count: Int
    var dof_offset: Int
    var dof_count: Int
    var mode_offset: Int
    var mode_count: Int
    var i_node: Int
    var j_node: Int
    var drift_dof: Int
    var perp_dirn: Int

    fn __init__(out self):
        self.type_tag = RecorderTypeTag.Unknown
        self.output = ""
        self.node_offset = 0
        self.node_count = 0
        self.element_offset = 0
        self.element_count = 0
        self.dof_offset = 0
        self.dof_count = 0
        self.mode_offset = 0
        self.mode_count = 0
        self.i_node = -1
        self.j_node = -1
        self.drift_dof = -1
        self.perp_dirn = -1


struct CaseInput(Movable):
    var model: ModelInput
    var nodes: List[NodeInput]
    var sections: List[SectionInput]
    var materials: List[MaterialInput]
    var elements: List[ElementInput]
    var element_loads: List[ElementLoadInput]
    var loads: List[NodalLoadInput]
    var masses: List[MassInput]
    var analysis: AnalysisInput
    var mp_constraints: List[MPConstraintInput]
    var pattern: PatternInput
    var rayleigh: RayleighInput
    var time_series: PythonObject
    var analysis_integrator_targets_pool: List[Float64]
    var recorder_nodes_pool: List[Int]
    var recorder_elements_pool: List[Int]
    var recorder_dofs_pool: List[Int]
    var recorder_modes_pool: List[Int]
    var recorders: List[RecorderInput]

    fn __init__(out self):
        self.model = ModelInput(0, 0)
        self.nodes = []
        self.sections = []
        self.materials = []
        self.elements = []
        self.element_loads = []
        self.loads = []
        self.masses = []
        self.analysis = AnalysisInput()
        self.mp_constraints = []
        self.pattern = PatternInput()
        self.rayleigh = RayleighInput()
        self.time_series = None
        self.analysis_integrator_targets_pool = []
        self.recorder_nodes_pool = []
        self.recorder_elements_pool = []
        self.recorder_dofs_pool = []
        self.recorder_modes_pool = []
        self.recorders = []


fn element_type_tag(type_name: String) -> Int:
    if type_name == "elasticBeamColumn2d":
        return ElementTypeTag.ElasticBeamColumn2d
    if type_name == "forceBeamColumn2d":
        return ElementTypeTag.ForceBeamColumn2d
    if type_name == "dispBeamColumn2d":
        return ElementTypeTag.DispBeamColumn2d
    if type_name == "elasticBeamColumn3d":
        return ElementTypeTag.ElasticBeamColumn3d
    if type_name == "truss":
        return ElementTypeTag.Truss
    if type_name == "zeroLength" or type_name == "twoNodeLink":
        return ElementTypeTag.Link
    if type_name == "zeroLengthSection":
        return ElementTypeTag.ZeroLengthSection
    if type_name == "fourNodeQuad" or type_name == "bbarQuad":
        return ElementTypeTag.FourNodeQuad
    if type_name == "shell":
        return ElementTypeTag.Shell
    return ElementTypeTag.Unknown


fn geom_transf_tag(geom_name: String) -> Int:
    if geom_name == "Linear":
        return GeomTransfTag.Linear
    if geom_name == "PDelta":
        return GeomTransfTag.PDelta
    if geom_name == "Corotational":
        return GeomTransfTag.Corotational
    return GeomTransfTag.Unknown


fn recorder_type_tag(type_name: String) -> Int:
    if type_name == "node_displacement":
        return RecorderTypeTag.NodeDisplacement
    if type_name == "element_force":
        return RecorderTypeTag.ElementForce
    if type_name == "node_reaction":
        return RecorderTypeTag.NodeReaction
    if type_name == "drift":
        return RecorderTypeTag.Drift
    if type_name == "envelope_element_force":
        return RecorderTypeTag.EnvelopeElementForce
    if type_name == "modal_eigen":
        return RecorderTypeTag.ModalEigen
    return RecorderTypeTag.Unknown


fn parse_case_input(data: PythonObject) raises -> CaseInput:
    var case_input = CaseInput()

    var model = data["model"]
    case_input.model = ModelInput(Int(model["ndm"]), Int(model["ndf"]))

    var nodes_raw = data["nodes"]
    for i in range(py_len(nodes_raw)):
        var node = nodes_raw[i]
        var has_z = node.__contains__("z")
        var z = 0.0
        if has_z:
            z = Float64(node["z"])
        var parsed = NodeInput(
            Int(node["id"]), Float64(node["x"]), Float64(node["y"]), z, has_z
        )
        if node.__contains__("constraints"):
            var constraints = node["constraints"]
            var count = py_len(constraints)
            if count > 6:
                count = 6
            parsed.constraint_count = count
            if count > 0:
                parsed.constraint_1 = Int(constraints[0])
            if count > 1:
                parsed.constraint_2 = Int(constraints[1])
            if count > 2:
                parsed.constraint_3 = Int(constraints[2])
            if count > 3:
                parsed.constraint_4 = Int(constraints[3])
            if count > 4:
                parsed.constraint_5 = Int(constraints[4])
            if count > 5:
                parsed.constraint_6 = Int(constraints[5])
        case_input.nodes.append(parsed)

    var sections_raw = data.get("sections", [])
    for i in range(py_len(sections_raw)):
        var sec = sections_raw[i]
        var parsed = SectionInput(Int(sec["id"]), String(sec["type"]))
        var params = sec.get("params", {})
        if params.__contains__("E"):
            parsed.E = Float64(params["E"])
        if params.__contains__("A"):
            parsed.A = Float64(params["A"])
        if params.__contains__("I"):
            parsed.I = Float64(params["I"])
        if params.__contains__("Iz"):
            parsed.Iz = Float64(params["Iz"])
        if params.__contains__("Iy"):
            parsed.Iy = Float64(params["Iy"])
        if params.__contains__("G"):
            parsed.G = Float64(params["G"])
        if params.__contains__("J"):
            parsed.J = Float64(params["J"])
        if params.__contains__("nu"):
            parsed.nu = Float64(params["nu"])
        if params.__contains__("h"):
            parsed.h = Float64(params["h"])
        case_input.sections.append(parsed)

    var materials_raw = data.get("materials", [])
    for i in range(py_len(materials_raw)):
        var mat = materials_raw[i]
        var params = mat.get("params", {})
        var parsed = MaterialInput()
        parsed.id = Int(mat["id"])
        parsed.type = String(mat["type"])
        if params.__contains__("E"):
            parsed.E = Float64(params["E"])
        if params.__contains__("Fy"):
            parsed.Fy = Float64(params["Fy"])
        if params.__contains__("E0"):
            parsed.E0 = Float64(params["E0"])
        if params.__contains__("b"):
            parsed.b = Float64(params["b"])
        if params.__contains__("fpc"):
            parsed.fpc = Float64(params["fpc"])
        if params.__contains__("epsc0"):
            parsed.epsc0 = Float64(params["epsc0"])
        if params.__contains__("fpcu"):
            parsed.fpcu = Float64(params["fpcu"])
        if params.__contains__("epscu"):
            parsed.epscu = Float64(params["epscu"])
        parsed.has_r0 = params.__contains__("R0")
        parsed.has_cr1 = params.__contains__("cR1")
        parsed.has_cr2 = params.__contains__("cR2")
        if parsed.has_r0:
            parsed.R0 = Float64(params["R0"])
        if parsed.has_cr1:
            parsed.cR1 = Float64(params["cR1"])
        if parsed.has_cr2:
            parsed.cR2 = Float64(params["cR2"])
        parsed.has_a1 = params.__contains__("a1")
        parsed.has_a2 = params.__contains__("a2")
        parsed.has_a3 = params.__contains__("a3")
        parsed.has_a4 = params.__contains__("a4")
        if parsed.has_a1:
            parsed.a1 = Float64(params["a1"])
        if parsed.has_a2:
            parsed.a2 = Float64(params["a2"])
        if parsed.has_a3:
            parsed.a3 = Float64(params["a3"])
        if parsed.has_a4:
            parsed.a4 = Float64(params["a4"])
        parsed.has_siginit = params.__contains__("sigInit")
        if parsed.has_siginit:
            parsed.sigInit = Float64(params["sigInit"])
        parsed.has_rat = params.__contains__("rat")
        parsed.has_ft = params.__contains__("ft")
        parsed.has_ets = params.__contains__("Ets")
        if parsed.has_rat:
            parsed.rat = Float64(params["rat"])
        if parsed.has_ft:
            parsed.ft = Float64(params["ft"])
        if parsed.has_ets:
            parsed.Ets = Float64(params["Ets"])
        if params.__contains__("nu"):
            parsed.nu = Float64(params["nu"])
        if params.__contains__("rho"):
            parsed.rho = Float64(params["rho"])
        case_input.materials.append(parsed^)

    var elements_raw = data["elements"]
    for i in range(py_len(elements_raw)):
        var elem = elements_raw[i]
        var parsed = ElementInput()
        parsed.id = Int(elem["id"])
        parsed.type = String(elem["type"])
        var nodes = elem["nodes"]
        var node_count = py_len(nodes)
        if node_count > 4:
            node_count = 4
        parsed.node_count = node_count
        if node_count > 0:
            parsed.node_1 = Int(nodes[0])
        if node_count > 1:
            parsed.node_2 = Int(nodes[1])
        if node_count > 2:
            parsed.node_3 = Int(nodes[2])
        if node_count > 3:
            parsed.node_4 = Int(nodes[3])
        if elem.__contains__("section"):
            parsed.section = Int(elem["section"])
        if elem.__contains__("material"):
            parsed.material = Int(elem["material"])
        if elem.__contains__("materials"):
            var mat_ids = elem["materials"]
            var material_count = py_len(mat_ids)
            if material_count > 6:
                material_count = 6
            parsed.material_count = material_count
            if material_count > 0:
                parsed.material_1 = Int(mat_ids[0])
            if material_count > 1:
                parsed.material_2 = Int(mat_ids[1])
            if material_count > 2:
                parsed.material_3 = Int(mat_ids[2])
            if material_count > 3:
                parsed.material_4 = Int(mat_ids[3])
            if material_count > 4:
                parsed.material_5 = Int(mat_ids[4])
            if material_count > 5:
                parsed.material_6 = Int(mat_ids[5])
        if elem.__contains__("dirs"):
            var dirs = elem["dirs"]
            var dir_count = py_len(dirs)
            if dir_count > 6:
                dir_count = 6
            parsed.dir_count = dir_count
            if dir_count > 0:
                parsed.dir_1 = Int(dirs[0])
            if dir_count > 1:
                parsed.dir_2 = Int(dirs[1])
            if dir_count > 2:
                parsed.dir_3 = Int(dirs[2])
            if dir_count > 3:
                parsed.dir_4 = Int(dirs[3])
            if dir_count > 4:
                parsed.dir_5 = Int(dirs[4])
            if dir_count > 5:
                parsed.dir_6 = Int(dirs[5])
        if elem.__contains__("area"):
            parsed.area = Float64(elem["area"])
        if elem.__contains__("thickness"):
            parsed.thickness = Float64(elem["thickness"])
        parsed.formulation = String(elem.get("formulation", "PlaneStress"))
        parsed.geom_transf = String(elem.get("geomTransf", "Linear"))
        parsed.integration = String(elem.get("integration", "Lobatto"))
        parsed.num_int_pts = Int(elem.get("num_int_pts", 3))
        parsed.type_tag = element_type_tag(parsed.type)
        parsed.geom_tag = geom_transf_tag(parsed.geom_transf)
        case_input.elements.append(parsed^)

    var element_loads_raw = data.get("element_loads", [])
    for i in range(py_len(element_loads_raw)):
        var load = element_loads_raw[i]
        case_input.element_loads.append(
            ElementLoadInput(
                Int(load["element"]), String(load["type"]), Float64(load.get("w", 0.0))
            )
        )

    var loads_raw = data.get("loads", [])
    for i in range(py_len(loads_raw)):
        var load = loads_raw[i]
        case_input.loads.append(
            NodalLoadInput(Int(load["node"]), Int(load["dof"]), Float64(load["value"]))
        )

    var masses_raw = data.get("masses", [])
    for i in range(py_len(masses_raw)):
        var mass = masses_raw[i]
        case_input.masses.append(
            MassInput(Int(mass["node"]), Int(mass["dof"]), Float64(mass["value"]))
        )

    var analysis_raw = data.get("analysis", {"type": "static_linear", "steps": 1})
    var analysis = AnalysisInput()
    analysis.type = String(analysis_raw.get("type", "static_linear"))
    analysis.constraints = String(analysis_raw.get("constraints", "Plain"))
    analysis.steps = Int(analysis_raw.get("steps", 1))
    analysis.num_modes = Int(analysis_raw.get("num_modes", 0))
    analysis.force_beam_mode = String(analysis_raw.get("force_beam_mode", "auto"))
    analysis.dt = Float64(analysis_raw.get("dt", 0.0))
    analysis.algorithm = String(analysis_raw.get("algorithm", "Newton"))
    analysis.max_iters = Int(analysis_raw.get("max_iters", 20))
    analysis.tol = Float64(analysis_raw.get("tol", 1.0e-10))
    analysis.rel_tol = Float64(analysis_raw.get("rel_tol", 1.0e-8))
    analysis.fallback_algorithm = String(analysis_raw.get("fallback_algorithm", ""))
    analysis.test_type = String(analysis_raw.get("test_type", "MaxDispIncr"))
    analysis.fallback_test_type = String(
        analysis_raw.get("fallback_test_type", analysis.test_type)
    )
    analysis.fallback_max_iters = Int(
        analysis_raw.get("fallback_max_iters", analysis.max_iters)
    )
    analysis.fallback_tol = Float64(analysis_raw.get("fallback_tol", analysis.tol))
    analysis.fallback_rel_tol = Float64(
        analysis_raw.get("fallback_rel_tol", analysis.rel_tol)
    )
    if analysis_raw.__contains__("system"):
        analysis.solver = String(analysis_raw["system"])
    elif analysis_raw.__contains__("solver"):
        analysis.solver = String(analysis_raw["solver"])
    else:
        analysis.solver = "auto"
    analysis.band_threshold = Int(analysis_raw.get("band_threshold", 128))
    var integrator_raw = analysis_raw.get("integrator", {})
    var default_integrator_type = ""
    if analysis.type == "static_nonlinear":
        default_integrator_type = "LoadControl"
    elif analysis.type == "transient_linear" or analysis.type == "transient_nonlinear":
        default_integrator_type = "Newmark"
    analysis.integrator_type = String(
        integrator_raw.get("type", default_integrator_type)
    )
    analysis.integrator_gamma = Float64(integrator_raw.get("gamma", 0.5))
    analysis.integrator_beta = Float64(integrator_raw.get("beta", 0.25))
    if integrator_raw.__contains__("node"):
        analysis.integrator_node = Int(integrator_raw["node"])
    if integrator_raw.__contains__("dof"):
        analysis.integrator_dof = Int(integrator_raw["dof"])
    analysis.integrator_cutback = Float64(
        integrator_raw.get("cutback", analysis_raw.get("cutback", 0.5))
    )
    analysis.integrator_max_cutbacks = Int(
        integrator_raw.get("max_cutbacks", analysis_raw.get("max_cutbacks", 8))
    )
    analysis.integrator_min_du = Float64(
        integrator_raw.get("min_du", analysis_raw.get("min_du", 1.0e-10))
    )
    analysis.has_integrator_du = integrator_raw.__contains__("du")
    if analysis.has_integrator_du:
        analysis.integrator_du = Float64(integrator_raw["du"])
    if integrator_raw.__contains__("targets"):
        var targets = integrator_raw["targets"]
        analysis.integrator_targets_offset = len(
            case_input.analysis_integrator_targets_pool
        )
        analysis.integrator_targets_count = py_len(targets)
        for i in range(py_len(targets)):
            case_input.analysis_integrator_targets_pool.append(Float64(targets[i]))
    case_input.analysis = analysis^

    var mpc_raw = data.get("mp_constraints", [])
    for i in range(py_len(mpc_raw)):
        var mpc = mpc_raw[i]
        var parsed = MPConstraintInput(
            String(mpc.get("type", "")),
            Int(mpc["retained_node"]),
            Int(mpc["constrained_node"]),
        )
        if mpc.__contains__("dofs"):
            var dofs = mpc["dofs"]
            var dof_count = py_len(dofs)
            if dof_count > 6:
                dof_count = 6
            parsed.dof_count = dof_count
            if dof_count > 0:
                parsed.dof_1 = Int(dofs[0])
            if dof_count > 1:
                parsed.dof_2 = Int(dofs[1])
            if dof_count > 2:
                parsed.dof_3 = Int(dofs[2])
            if dof_count > 3:
                parsed.dof_4 = Int(dofs[3])
            if dof_count > 4:
                parsed.dof_5 = Int(dofs[4])
            if dof_count > 5:
                parsed.dof_6 = Int(dofs[5])
        case_input.mp_constraints.append(parsed)

    var pattern_raw = data.get("pattern", None)
    var pattern = PatternInput()
    if pattern_raw is not None:
        pattern.has_pattern = True
        pattern.type = String(pattern_raw.get("type", "Plain"))
        if pattern_raw.__contains__("time_series"):
            pattern.has_time_series = True
            pattern.time_series = Int(pattern_raw["time_series"])
        if pattern_raw.__contains__("direction"):
            pattern.has_direction = True
            pattern.direction = Int(pattern_raw["direction"])
        if pattern_raw.__contains__("accel"):
            pattern.has_accel = True
            pattern.accel = Int(pattern_raw["accel"])
    case_input.pattern = pattern^

    var rayleigh_raw = data.get("rayleigh", None)
    var rayleigh = RayleighInput()
    if rayleigh_raw is not None:
        rayleigh.has_rayleigh = True
        rayleigh.alpha_m = Float64(rayleigh_raw.get("alphaM", 0.0))
        rayleigh.beta_k = Float64(rayleigh_raw.get("betaK", 0.0))
        rayleigh.beta_k_init = Float64(rayleigh_raw.get("betaKInit", 0.0))
        rayleigh.beta_k_comm = Float64(rayleigh_raw.get("betaKComm", 0.0))
    case_input.rayleigh = rayleigh^

    if data.__contains__("time_series"):
        case_input.time_series = data["time_series"]
    else:
        var builtins = Python.import_module("builtins")
        case_input.time_series = builtins.list()

    var recorders_raw = data.get("recorders", [])
    for i in range(py_len(recorders_raw)):
        var rec = recorders_raw[i]
        var parsed = RecorderInput()
        parsed.type_tag = recorder_type_tag(String(rec["type"]))
        if parsed.type_tag == RecorderTypeTag.Unknown:
            abort("unsupported recorder type")
        if parsed.type_tag == RecorderTypeTag.NodeDisplacement:
            parsed.output = String(rec.get("output", "node_disp"))
            if not rec.__contains__("nodes") or not rec.__contains__("dofs"):
                abort("node_displacement recorder requires nodes and dofs")
            var nodes_raw = rec["nodes"]
            parsed.node_offset = len(case_input.recorder_nodes_pool)
            parsed.node_count = py_len(nodes_raw)
            for j in range(py_len(nodes_raw)):
                case_input.recorder_nodes_pool.append(Int(nodes_raw[j]))
            var dofs_raw = rec["dofs"]
            parsed.dof_offset = len(case_input.recorder_dofs_pool)
            parsed.dof_count = py_len(dofs_raw)
            for j in range(py_len(dofs_raw)):
                case_input.recorder_dofs_pool.append(Int(dofs_raw[j]))
        elif parsed.type_tag == RecorderTypeTag.ElementForce:
            parsed.output = String(rec.get("output", "element_force"))
            if not rec.__contains__("elements"):
                abort("element_force recorder requires elements")
            var elements_raw = rec["elements"]
            parsed.element_offset = len(case_input.recorder_elements_pool)
            parsed.element_count = py_len(elements_raw)
            for j in range(py_len(elements_raw)):
                case_input.recorder_elements_pool.append(Int(elements_raw[j]))
        elif parsed.type_tag == RecorderTypeTag.NodeReaction:
            parsed.output = String(rec.get("output", "reaction"))
            if not rec.__contains__("nodes") or not rec.__contains__("dofs"):
                abort("node_reaction recorder requires nodes and dofs")
            var nodes_raw = rec["nodes"]
            parsed.node_offset = len(case_input.recorder_nodes_pool)
            parsed.node_count = py_len(nodes_raw)
            for j in range(py_len(nodes_raw)):
                case_input.recorder_nodes_pool.append(Int(nodes_raw[j]))
            var dofs_raw = rec["dofs"]
            parsed.dof_offset = len(case_input.recorder_dofs_pool)
            parsed.dof_count = py_len(dofs_raw)
            for j in range(py_len(dofs_raw)):
                case_input.recorder_dofs_pool.append(Int(dofs_raw[j]))
        elif parsed.type_tag == RecorderTypeTag.Drift:
            parsed.output = String(rec.get("output", "drift"))
            if (
                not rec.__contains__("i_node")
                or not rec.__contains__("j_node")
                or not rec.__contains__("dof")
                or not rec.__contains__("perp_dirn")
            ):
                abort("drift recorder requires i_node, j_node, dof, perp_dirn")
            parsed.i_node = Int(rec["i_node"])
            parsed.j_node = Int(rec["j_node"])
            parsed.drift_dof = Int(rec["dof"])
            parsed.perp_dirn = Int(rec["perp_dirn"])
        elif parsed.type_tag == RecorderTypeTag.EnvelopeElementForce:
            parsed.output = String(rec.get("output", "envelope_element_force"))
            if not rec.__contains__("elements"):
                abort("envelope_element_force recorder requires elements")
            var elements_raw = rec["elements"]
            parsed.element_offset = len(case_input.recorder_elements_pool)
            parsed.element_count = py_len(elements_raw)
            for j in range(py_len(elements_raw)):
                case_input.recorder_elements_pool.append(Int(elements_raw[j]))
        else:
            parsed.output = String(rec.get("output", "modal"))
            if not rec.__contains__("nodes") or not rec.__contains__("dofs"):
                abort("modal_eigen recorder requires nodes and dofs")
            var nodes_raw = rec["nodes"]
            parsed.node_offset = len(case_input.recorder_nodes_pool)
            parsed.node_count = py_len(nodes_raw)
            for j in range(py_len(nodes_raw)):
                case_input.recorder_nodes_pool.append(Int(nodes_raw[j]))
            var dofs_raw = rec["dofs"]
            parsed.dof_offset = len(case_input.recorder_dofs_pool)
            parsed.dof_count = py_len(dofs_raw)
            for j in range(py_len(dofs_raw)):
                case_input.recorder_dofs_pool.append(Int(dofs_raw[j]))
            var modes_raw = rec.get("modes", [])
            parsed.mode_offset = len(case_input.recorder_modes_pool)
            parsed.mode_count = py_len(modes_raw)
            for j in range(py_len(modes_raw)):
                case_input.recorder_modes_pool.append(Int(modes_raw[j]))
        case_input.recorders.append(parsed^)
    return case_input^
