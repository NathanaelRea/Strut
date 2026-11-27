from solver.run_case.analysis.hs_constraint import HSConstraint
from materials.pressure_depend_multi_yield import PressureDependMultiYieldDef
from materials.pressure_depend_multi_yield_02 import PressureDependMultiYield02Parameters
from materials.pressure_independ_multi_yield import PressureIndependMultiYieldDef
from elements.continuum import continuum_nodes
from materials.drucker_prager import DruckerPragerDefinition
from collections import InlineArray, List
from pathlib import Path
from os import abort
from elements.hinge_radau2d import HingeRadau2d
from elements.force_beam_column_cbdi import CbdiSettings
from sections.thermal_profile2d import ThermalProfile2d

from materials.fsam import FSAMInput
from elements.catenary_cable import CatenaryCable
from elements.mvlem import MVLEMFiber, MVLEMGeometry
from json_native import JsonDocument, JsonValueTag
from materials.uniaxial.core import UniMaterialDef
from strut_io import CaseSourceInfo, read_text_native
from solver.time_series import TimeSeriesInput, find_time_series_input
from tag_types import (
    AnalysisAlgorithmTag,
    AnalysisTypeTag,
    AnalysisSystemTag,
    BeamIntegrationTag,
    ConstraintHandlerTag,
    ElementLoadTypeTag,
    ElementTypeTag,
    ForceBeamModeTag,
    GeomTransfTag,
    IntegratorTypeTag,
    NonlinearTestTypeTag,
    NumbererTag,
    PatternTypeTag,
    RecorderTypeTag,
    TimeSeriesTypeTag,
    UniMaterialTypeTag,
)


struct ModelInput(ImplicitlyCopyable, Movable):
    var ndm: Int
    var ndf: Int

    fn __init__(out self, ndm: Int, ndf: Int):
        self.ndm = ndm
        self.ndf = ndf


struct NodeInput(ImplicitlyCopyable, Movable):
    var id: Int
    var x: Float64
    var y: Float64
    var z: Float64
    var has_z: Bool
    var ndf: Int
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
        self.ndf = 0
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
        self.ndf = 0
        self.constraint_count = 0
        self.constraint_1 = 0
        self.constraint_2 = 0
        self.constraint_3 = 0
        self.constraint_4 = 0
        self.constraint_5 = 0
        self.constraint_6 = 0


struct SectionInput(ImplicitlyCopyable, Movable):
    var id: Int
    var type: String
    var E: Float64
    var A: Float64
    var I: Float64
    var Iz: Float64
    var Iy: Float64
    var G: Float64
    var J: Float64
    var shear_rigidity_y: Float64
    var shear_rigidity_z: Float64
    var base_torsional_rigidity: Float64
    var added_torsional_rigidity: Float64
    var nu: Float64
    var h: Float64
    var rho: Float64
    var axial_material: Int
    var flexural_material: Int
    var moment_y_material: Int
    var torsion_material: Int
    var shear_y_material: Int
    var shear_z_material: Int
    var base_section: Int
    var fiber_patch_offset: Int
    var fiber_patch_count: Int
    var fiber_layer_offset: Int
    var fiber_layer_count: Int
    var fiber_offset: Int
    var fiber_count: Int
    var plate_mod: Float64
    var plate_fiber_material: Int
    var shell_layer_offset: Int
    var shell_layer_count: Int
    var hinge_radau: HingeRadau2d
    var cbdi: CbdiSettings

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
        self.shear_rigidity_y = 0.0
        self.shear_rigidity_z = 0.0
        self.base_torsional_rigidity = 0.0
        self.added_torsional_rigidity = 0.0
        self.nu = 0.0
        self.h = 0.0
        self.rho = 0.0
        self.axial_material = -1
        self.flexural_material = -1
        self.moment_y_material = -1
        self.torsion_material = -1
        self.shear_y_material = -1
        self.shear_z_material = -1
        self.base_section = -1
        self.fiber_patch_offset = 0
        self.fiber_patch_count = 0
        self.fiber_layer_offset = 0
        self.fiber_layer_count = 0
        self.fiber_offset = 0
        self.fiber_count = 0
        self.plate_mod = 1.0
        self.plate_fiber_material = -1
        self.shell_layer_offset = 0
        self.shell_layer_count = 0
        self.hinge_radau = HingeRadau2d()
        self.cbdi = CbdiSettings(0, 0.0)

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
        self.shear_rigidity_y = 0.0
        self.shear_rigidity_z = 0.0
        self.base_torsional_rigidity = 0.0
        self.added_torsional_rigidity = 0.0
        self.nu = 0.0
        self.h = 0.0
        self.rho = 0.0
        self.axial_material = -1
        self.flexural_material = -1
        self.moment_y_material = -1
        self.torsion_material = -1
        self.shear_y_material = -1
        self.shear_z_material = -1
        self.base_section = -1
        self.fiber_patch_offset = 0
        self.fiber_patch_count = 0
        self.fiber_layer_offset = 0
        self.fiber_layer_count = 0
        self.fiber_offset = 0
        self.fiber_count = 0
        self.plate_mod = 1.0
        self.plate_fiber_material = -1
        self.shell_layer_offset = 0
        self.shell_layer_count = 0
        self.hinge_radau = HingeRadau2d()
        self.cbdi = CbdiSettings(0, 0.0)

    fn beam_shear_modulus(self, element_type: Int) -> Float64:
        if element_type == ElementTypeTag.ForceBeamColumn3d and self.added_torsional_rigidity > 0.0:
            var base = self.base_torsional_rigidity
            var added = self.added_torsional_rigidity
            return base * added / (base + added) / self.J
        return self.G


struct FiberPatchInput(ImplicitlyCopyable, Movable):
    var type: String
    var material: Int
    var num_subdiv_y: Int
    var num_subdiv_z: Int
    var y_i: Float64
    var z_i: Float64
    var y_j: Float64
    var z_j: Float64
    var y_k: Float64
    var z_k: Float64
    var y_l: Float64
    var z_l: Float64
    var num_subdiv_circ: Int
    var num_subdiv_rad: Int
    var y_center: Float64
    var z_center: Float64
    var int_rad: Float64
    var ext_rad: Float64
    var start_ang: Float64
    var end_ang: Float64

    fn __init__(out self):
        self.type = ""
        self.material = -1
        self.num_subdiv_y = 0
        self.num_subdiv_z = 0
        self.y_i = 0.0
        self.z_i = 0.0
        self.y_j = 0.0
        self.z_j = 0.0
        self.y_k = 0.0
        self.z_k = 0.0
        self.y_l = 0.0
        self.z_l = 0.0
        self.num_subdiv_circ = 0
        self.num_subdiv_rad = 0
        self.y_center = 0.0
        self.z_center = 0.0
        self.int_rad = 0.0
        self.ext_rad = 0.0
        self.start_ang = 0.0
        self.end_ang = 360.0


struct FiberLayerInput(ImplicitlyCopyable, Movable):
    var type: String
    var material: Int
    var num_bars: Int
    var bar_area: Float64
    var y_start: Float64
    var z_start: Float64
    var y_end: Float64
    var z_end: Float64
    var y_center: Float64
    var z_center: Float64
    var radius: Float64
    var start_ang: Float64
    var end_ang: Float64
    var has_angles: Bool

    fn __init__(out self):
        self.type = ""
        self.material = -1
        self.num_bars = 0
        self.bar_area = 0.0
        self.y_start = 0.0
        self.z_start = 0.0
        self.y_end = 0.0
        self.z_end = 0.0
        self.y_center = 0.0
        self.z_center = 0.0
        self.radius = 0.0
        self.start_ang = 0.0
        self.end_ang = 360.0
        self.has_angles = False


struct FiberInput(ImplicitlyCopyable, Movable):
    var y: Float64
    var z: Float64
    var area: Float64
    var material: Int

    fn __init__(out self):
        self.y = 0.0
        self.z = 0.0
        self.area = 0.0
        self.material = -1


struct ShellLayerInput(ImplicitlyCopyable, Movable):
    var material: Int
    var thickness: Float64

    fn __init__(out self):
        self.material = -1
        self.thickness = 0.0


@fieldwise_init
struct ParallelComponentInput(ImplicitlyCopyable, Movable):
    var material: Int
    var factor: Float64


struct MaterialInput(ImplicitlyCopyable, Movable):
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
    var fc: Float64
    var Es: Float64
    var nu: Float64
    var rho: Float64
    var drucker_prager: Optional[DruckerPragerDefinition]
    var cross_anisotropic: InlineArray[Float64, 5]
    var viscous_coefficient: Float64
    var viscous_exponent: Float64
    var pimy_index: Int
    var pdmy_index: Int
    var soil_dimensions: Int
    var base_material: Int
    var angle: Float64
    var gmod: Float64
    var contact_parameters: InlineArray[Float64, 4]
    var uniaxial_values: InlineArray[Float64, 24]
    var soil_type: Int
    var soil_ultimate: Float64
    var soil_disp50: Float64
    var soil_drag_or_suction: Float64
    var soil_dashpot: Float64
    var K: Float64
    var C: Float64
    var alpha: Float64
    var LGap: Float64
    var NM: Float64
    var RelTol: Float64
    var AbsTol: Float64
    var MaxHalf: Float64
    var fatigue_dmax: Float64
    var fatigue_slope: Float64
    var strain_min: Float64
    var strain_max: Float64
    var ka: Float64
    var kb: Float64
    var fbar: Float64
    var beta1: Float64
    var beta2: Float64
    var gamma: Float64

    var ent_a: Float64
    var ent_b: Float64
    var parallel_components: Optional[List[ParallelComponentInput]]
    var direct_uniaxial_def: UniMaterialDef
    var fsam: Optional[FSAMInput]

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
        self.fc = 0.0
        self.Es = 0.0
        self.nu = 0.0
        self.rho = 0.0
        self.drucker_prager = None
        self.cross_anisotropic = InlineArray[Float64, 5](fill=0.0)
        self.viscous_coefficient = 0.0
        self.viscous_exponent = 1.0
        self.pimy_index = -1
        self.pdmy_index = -1
        self.soil_dimensions = 0
        self.base_material = -1
        self.angle = 0.0
        self.gmod = 0.0
        self.contact_parameters = InlineArray[Float64, 4](fill=0.0)
        self.uniaxial_values = InlineArray[Float64, 24](fill=0.0)
        self.soil_type = 0
        self.soil_ultimate = 0.0
        self.soil_disp50 = 0.0
        self.soil_drag_or_suction = 0.0
        self.soil_dashpot = 0.0
        self.K = 0.0
        self.C = 0.0
        self.alpha = 1.0
        self.LGap = 0.0
        self.NM = 1.0
        self.RelTol = 1.0e-6
        self.AbsTol = 1.0e-10
        self.MaxHalf = 15.0
        self.fatigue_dmax = 1.0
        self.fatigue_slope = -0.458
        self.strain_min = -1.0e16
        self.strain_max = 1.0e16
        self.ka = 0.0
        self.kb = 0.0
        self.fbar = 0.0
        self.beta1 = 0.0
        self.beta2 = 0.0
        self.gamma = 0.0
        self.ent_a = 0.0
        self.ent_b = 1.0
        self.parallel_components = None
        self.direct_uniaxial_def = UniMaterialDef()
        self.fsam = None
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

    fn __copyinit__(out self, existing: Self):
        self.id = existing.id
        self.type = existing.type
        self.E = existing.E
        self.Fy = existing.Fy
        self.E0 = existing.E0
        self.b = existing.b
        self.fpc = existing.fpc
        self.epsc0 = existing.epsc0
        self.fpcu = existing.fpcu
        self.epscu = existing.epscu
        self.R0 = existing.R0
        self.cR1 = existing.cR1
        self.cR2 = existing.cR2
        self.a1 = existing.a1
        self.a2 = existing.a2
        self.a3 = existing.a3
        self.a4 = existing.a4
        self.sigInit = existing.sigInit
        self.rat = existing.rat
        self.ft = existing.ft
        self.Ets = existing.Ets
        self.fc = existing.fc
        self.Es = existing.Es
        self.nu = existing.nu
        self.rho = existing.rho
        self.drucker_prager = existing.drucker_prager
        self.cross_anisotropic = existing.cross_anisotropic
        self.viscous_coefficient = existing.viscous_coefficient
        self.viscous_exponent = existing.viscous_exponent
        self.pimy_index = existing.pimy_index
        self.pdmy_index = existing.pdmy_index
        self.soil_dimensions = existing.soil_dimensions
        self.base_material = existing.base_material
        self.angle = existing.angle
        self.gmod = existing.gmod
        self.contact_parameters = existing.contact_parameters.copy()
        self.uniaxial_values = existing.uniaxial_values.copy()
        self.soil_type = existing.soil_type
        self.soil_ultimate = existing.soil_ultimate
        self.soil_disp50 = existing.soil_disp50
        self.soil_drag_or_suction = existing.soil_drag_or_suction
        self.soil_dashpot = existing.soil_dashpot
        self.K = existing.K
        self.C = existing.C
        self.alpha = existing.alpha
        self.LGap = existing.LGap
        self.NM = existing.NM
        self.RelTol = existing.RelTol
        self.AbsTol = existing.AbsTol
        self.MaxHalf = existing.MaxHalf
        self.fatigue_dmax = existing.fatigue_dmax
        self.fatigue_slope = existing.fatigue_slope
        self.strain_min = existing.strain_min
        self.strain_max = existing.strain_max
        self.ka = existing.ka
        self.kb = existing.kb
        self.fbar = existing.fbar
        self.beta1 = existing.beta1
        self.beta2 = existing.beta2
        self.gamma = existing.gamma
        self.ent_a = existing.ent_a
        self.ent_b = existing.ent_b
        self.parallel_components = existing.parallel_components
        self.direct_uniaxial_def = existing.direct_uniaxial_def
        self.fsam = existing.fsam
        self.has_r0 = existing.has_r0
        self.has_cr1 = existing.has_cr1
        self.has_cr2 = existing.has_cr2
        self.has_a1 = existing.has_a1
        self.has_a2 = existing.has_a2
        self.has_a3 = existing.has_a3
        self.has_a4 = existing.has_a4
        self.has_siginit = existing.has_siginit
        self.has_rat = existing.has_rat
        self.has_ft = existing.has_ft
        self.has_ets = existing.has_ets


struct ElementInput(ImplicitlyCopyable, Movable):
    var id: Int
    var type: String
    var type_tag: Int
    var geom_tag: Int
    var node_count: Int
    var node_1: Int
    var node_2: Int
    var node_3: Int
    var node_4: Int
    var solid_extra_nodes: Optional[List[Int]]
    var node_index_1: Int
    var node_index_2: Int
    var node_index_3: Int
    var node_index_4: Int
    var extra_nodes: InlineArray[Int, 5]
    var extra_node_indices: InlineArray[Int, 5]
    var section: Int
    var material: Int
    var material_count: Int
    var material_1: Int
    var material_2: Int
    var material_3: Int
    var material_4: Int
    var material_5: Int
    var material_6: Int
    var has_viscous_material: Bool
    var damp_material_count: Int
    var damp_material_1: Int
    var damp_material_2: Int
    var damp_material_3: Int
    var damp_material_4: Int
    var damp_material_5: Int
    var damp_material_6: Int
    var damping_tag: Int
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
    var has_geom_vecxz: Bool
    var geom_vecxz_1: Float64
    var geom_vecxz_2: Float64
    var geom_vecxz_3: Float64
    var integration: String
    var num_int_pts: Int
    var cbdi: CbdiSettings
    var thermal_beam: Bool
    var rho: Float64
    var use_cmass: Bool
    var shell_eas: Bool
    var shell_drilling_stab: Float64
    var quad_pressure: Float64
    var quad_density: Float64
    var quad_body_x: Float64
    var quad_body_y: Float64
    var up_bulk: Float64
    var up_perm_x: Float64
    var up_perm_y: Float64
    var up_perm_z: Float64
    var up_void_ratio: Float64
    var up_stabilization: Float64
    var up_body_z: Float64
    var continuum_body_z: Float64
    var contact_gap_tol: Float64
    var contact_force_tol: Float64
    var contact_initial: Bool
    var element_mass: Float64
    var do_rayleigh: Bool
    var has_orient_x: Bool
    var orient_x_1: Float64
    var orient_x_2: Float64
    var orient_x_3: Float64
    var has_orient_y: Bool
    var orient_y_1: Float64
    var orient_y_2: Float64
    var orient_y_3: Float64
    var has_pdelta: Bool
    var pdelta_1: Float64
    var pdelta_2: Float64
    var pdelta_3: Float64
    var pdelta_4: Float64
    var cable: CatenaryCable
    var wall: MVLEMGeometry
    var wall_internal_node_start: Int
    var has_shear_dist: Bool
    var shear_dist_1: Float64
    var shear_dist_2: Float64
    var uniform_load_wy: Float64
    var uniform_load_wx: Float64
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
    var extra_dofs: InlineArray[Int, 8]

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
        self.solid_extra_nodes = None
        self.node_index_1 = -1
        self.node_index_2 = -1
        self.node_index_3 = -1
        self.node_index_4 = -1
        self.extra_nodes = InlineArray[Int, 5](fill=-1)
        self.extra_node_indices = InlineArray[Int, 5](fill=-1)
        self.section = -1
        self.material = -1
        self.material_count = 0
        self.material_1 = -1
        self.material_2 = -1
        self.material_3 = -1
        self.material_4 = -1
        self.material_5 = -1
        self.material_6 = -1
        self.has_viscous_material = False
        self.damp_material_count = 0
        self.damp_material_1 = -1
        self.damp_material_2 = -1
        self.damp_material_3 = -1
        self.damp_material_4 = -1
        self.damp_material_5 = -1
        self.damp_material_6 = -1
        self.damping_tag = -1
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
        self.has_geom_vecxz = False
        self.geom_vecxz_1 = 0.0
        self.geom_vecxz_2 = 0.0
        self.geom_vecxz_3 = 0.0
        self.integration = "Lobatto"
        self.cbdi = CbdiSettings(0, 0.0)
        self.thermal_beam = False
        self.num_int_pts = 3
        self.rho = 0.0
        self.use_cmass = False
        self.shell_eas = True
        self.shell_drilling_stab = 0.01
        self.quad_pressure = 0.0
        self.quad_density = 0.0
        self.quad_body_x = 0.0
        self.quad_body_y = 0.0
        self.up_bulk = 0.0
        self.up_perm_x = 0.0
        self.up_perm_y = 0.0
        self.up_perm_z = 0.0
        self.up_void_ratio = 0.0
        self.up_stabilization = 0.0
        self.up_body_z = 0.0
        self.continuum_body_z = 0.0
        self.contact_gap_tol = 0.0
        self.contact_force_tol = 0.0
        self.contact_initial = True
        self.element_mass = 0.0
        self.do_rayleigh = False
        self.has_orient_x = False
        self.orient_x_1 = 0.0
        self.orient_x_2 = 0.0
        self.orient_x_3 = 0.0
        self.has_orient_y = False
        self.orient_y_1 = 0.0
        self.orient_y_2 = 0.0
        self.orient_y_3 = 0.0
        self.has_pdelta = False
        self.pdelta_1 = 0.0
        self.pdelta_2 = 0.0
        self.pdelta_3 = 0.0
        self.pdelta_4 = 0.0
        self.cable = CatenaryCable()
        self.wall = MVLEMGeometry()
        self.wall_internal_node_start = -1
        self.has_shear_dist = False
        self.shear_dist_1 = 0.5
        self.shear_dist_2 = 0.5
        self.uniform_load_wy = 0.0
        self.uniform_load_wx = 0.0
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
        self.extra_dofs = InlineArray[Int, 8](fill=-1)


struct ElementLoadInput(ImplicitlyCopyable, Movable):
    var element: Int
    var type: String
    var type_tag: Int
    var wy: Float64
    var wz: Float64
    var wx: Float64
    var py: Float64
    var pz: Float64
    var px: Float64
    var x: Float64
    var thermal: ThermalProfile2d

    fn __init__(
        out self,
        element: Int,
        type: String,
        type_tag: Int,
        wy: Float64,
        wz: Float64,
        wx: Float64,
        py: Float64,
        pz: Float64,
        px: Float64,
        x: Float64,
    ):
        self.element = element
        self.type = type
        self.type_tag = type_tag
        self.wy = wy
        self.wz = wz
        self.wx = wx
        self.py = py
        self.pz = pz
        self.px = px
        self.x = x
        self.thermal = ThermalProfile2d()


struct NodalLoadInput(ImplicitlyCopyable, Movable):
    var node: Int
    var dof: Int
    var value: Float64

    fn __init__(out self, node: Int, dof: Int, value: Float64):
        self.node = node
        self.dof = dof
        self.value = value


struct MassInput(ImplicitlyCopyable, Movable):
    var node: Int
    var dof: Int
    var value: Float64

    fn __init__(out self, node: Int, dof: Int, value: Float64):
        self.node = node
        self.dof = dof
        self.value = value


@fieldwise_init
struct SupportMotionInput(ImplicitlyCopyable, Movable):
    var node: Int
    var dof: Int
    var time_series: Int
    var velocity_series: Int
    var acceleration_series: Int
    var factor: Float64


struct AnalysisConstraintsInput(ImplicitlyCopyable, Movable):
    var modal_damping: List[Float64]
    var support_motions: List[SupportMotionInput]

    fn __init__(out self):
        self.modal_damping = []
        self.support_motions = []

    fn __copyinit__(out self, existing: Self):
        self.modal_damping = existing.modal_damping.copy()
        self.support_motions = existing.support_motions.copy()


struct AnalysisInput(ImplicitlyCopyable, Movable):
    var supplemental: AnalysisConstraintsInput
    var type: String
    var type_tag: Int
    var constraints: String
    var constraints_tag: Int
    var penalty_sp: Float64
    var penalty_mp: Float64
    var numberer_tag: Int
    var steps: Int
    var num_modes: Int
    var standard_eigen: Bool
    var force_beam_mode: String
    var force_beam_mode_tag: Int
    var system: String
    var system_tag: Int
    var system_options_serialized: String
    var superlu_prefer_symmetric: Bool
    var superlu_enable_pivot: Bool
    var superlu_np_row: Int
    var superlu_np_col: Int
    var superlu_perm_spec: Int
    var umfpack_factor_once: Bool
    var umfpack_print_time: Bool
    var umfpack_lvalue_fact: Int
    var sparse_sym_ordering: Int
    var band_threshold: Int
    var dt: Float64
    var start_time: Float64
    var has_start_time: Bool
    var algorithm: String
    var algorithm_tag: Int
    var max_iters: Int
    var tol: Float64
    var test_type: String
    var test_type_tag: Int
    var step_retry_enabled: Bool
    var step_retry_restore_primary_after_success: Bool
    var step_retry_continue_after_failure: Bool
    var step_retry_continue_target_disp: Float64
    var step_retry_continue_max_steps: Int
    var has_solver_chain_override: Bool
    var solver_chain_offset: Int
    var solver_chain_count: Int
    var integrator_type: String
    var integrator_tag: Int
    var displacement_limit: Optional[Float64]
    var integrator_gamma: Float64
    var integrator_beta: Float64
    var integrator_num_iter: Int
    var has_integrator_num_iter: Bool
    var integrator_step: Float64
    var has_integrator_step: Bool
    var integrator_min_step: Float64
    var has_integrator_min_step: Bool
    var integrator_max_step: Float64
    var has_integrator_max_step: Bool
    var integrator_node: Int
    var integrator_dof: Int
    var integrator_cutback: Float64
    var integrator_max_cutbacks: Int
    var integrator_min_du: Float64
    var has_integrator_min_du: Bool
    var integrator_max_du: Float64
    var has_integrator_max_du: Bool
    var has_integrator_du: Bool
    var integrator_du: Float64
    var hs_constraint: HSConstraint
    var integrator_targets_offset: Int
    var integrator_targets_count: Int

    fn __init__(out self):
        self.type = "static_linear"
        self.type_tag = AnalysisTypeTag.StaticLinear
        self.constraints = "Plain"
        self.constraints_tag = ConstraintHandlerTag.Plain
        self.penalty_sp = 0.0
        self.penalty_mp = 0.0
        self.supplemental = AnalysisConstraintsInput()
        self.numberer_tag = NumbererTag.Unknown
        self.steps = 1
        self.num_modes = 0
        self.standard_eigen = False
        self.force_beam_mode = "auto"
        self.force_beam_mode_tag = ForceBeamModeTag.Auto
        self.system = "BandGeneral"
        self.system_tag = AnalysisSystemTag.BandGeneral
        self.system_options_serialized = ""
        self.superlu_prefer_symmetric = False
        self.superlu_enable_pivot = False
        self.superlu_np_row = -1
        self.superlu_np_col = -1
        self.superlu_perm_spec = -1
        self.umfpack_factor_once = False
        self.umfpack_print_time = False
        self.umfpack_lvalue_fact = -1
        self.sparse_sym_ordering = 0
        self.band_threshold = 128
        self.dt = 0.0
        self.start_time = 0.0
        self.has_start_time = False
        self.algorithm = "Newton"
        self.algorithm_tag = AnalysisAlgorithmTag.Newton
        self.max_iters = 20
        self.tol = 1.0e-10
        self.test_type = "NormUnbalance"
        self.test_type_tag = NonlinearTestTypeTag.NormUnbalance
        self.step_retry_enabled = False
        self.step_retry_restore_primary_after_success = True
        self.step_retry_continue_after_failure = False
        self.step_retry_continue_target_disp = 0.0
        self.step_retry_continue_max_steps = 0
        self.has_solver_chain_override = False
        self.solver_chain_offset = 0
        self.solver_chain_count = 0
        self.integrator_type = ""
        self.integrator_tag = IntegratorTypeTag.Unknown
        self.displacement_limit = None
        self.integrator_gamma = 0.5
        self.integrator_beta = 0.25
        self.integrator_num_iter = 1
        self.has_integrator_num_iter = False
        self.integrator_step = 1.0
        self.has_integrator_step = False
        self.integrator_min_step = 1.0
        self.has_integrator_min_step = False
        self.integrator_max_step = 1.0
        self.has_integrator_max_step = False
        self.integrator_node = -1
        self.integrator_dof = -1
        self.integrator_cutback = 0.5
        self.integrator_max_cutbacks = 8
        self.integrator_min_du = 1.0e-10
        self.has_integrator_min_du = False
        self.integrator_max_du = 0.0
        self.has_integrator_max_du = False
        self.has_integrator_du = False
        self.integrator_du = 0.0
        self.hs_constraint = HSConstraint()
        self.integrator_targets_offset = 0
        self.integrator_targets_count = 0


struct SolverAttemptInput(ImplicitlyCopyable, Movable):
    var algorithm: String
    var algorithm_tag: Int
    var broyden_count: Int
    var line_search_eta: Float64
    var krylov_max_dim: Int
    var test_type: String
    var test_type_tag: Int
    var max_iters: Int
    var tol: Float64

    fn __init__(out self):
        self.algorithm = ""
        self.algorithm_tag = AnalysisAlgorithmTag.Unknown
        self.broyden_count = 0
        self.line_search_eta = 1.0
        self.krylov_max_dim = 0
        self.test_type = "NormUnbalance"
        self.test_type_tag = NonlinearTestTypeTag.NormUnbalance
        self.max_iters = 20
        self.tol = 1.0e-10


struct MPConstraintInput(ImplicitlyCopyable, Movable):
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
    # Preserve the canonical rigidDiaphragm JSON shape in a fixed-size
    # allocation-light form for later loader/runtime phases.
    var rigid_perp_dirn: Int
    var rigid_constrained_dof_count: Int
    var rigid_constrained_dof_1: Int
    var rigid_constrained_dof_2: Int
    var rigid_constrained_dof_3: Int
    var rigid_retained_dof_count: Int
    var rigid_retained_dof_1: Int
    var rigid_retained_dof_2: Int
    var rigid_retained_dof_3: Int
    var rigid_matrix_row_count: Int
    var rigid_matrix_col_count: Int
    var rigid_matrix_11: Float64
    var rigid_matrix_12: Float64
    var rigid_matrix_13: Float64
    var rigid_matrix_21: Float64
    var rigid_matrix_22: Float64
    var rigid_matrix_23: Float64
    var rigid_matrix_31: Float64
    var rigid_matrix_32: Float64
    var rigid_matrix_33: Float64
    var rigid_dx: Float64
    var rigid_dy: Float64
    var rigid_dz: Float64

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
        self.rigid_perp_dirn = 0
        self.rigid_constrained_dof_count = 0
        self.rigid_constrained_dof_1 = 0
        self.rigid_constrained_dof_2 = 0
        self.rigid_constrained_dof_3 = 0
        self.rigid_retained_dof_count = 0
        self.rigid_retained_dof_1 = 0
        self.rigid_retained_dof_2 = 0
        self.rigid_retained_dof_3 = 0
        self.rigid_matrix_row_count = 0
        self.rigid_matrix_col_count = 0
        self.rigid_matrix_11 = 0.0
        self.rigid_matrix_12 = 0.0
        self.rigid_matrix_13 = 0.0
        self.rigid_matrix_21 = 0.0
        self.rigid_matrix_22 = 0.0
        self.rigid_matrix_23 = 0.0
        self.rigid_matrix_31 = 0.0
        self.rigid_matrix_32 = 0.0
        self.rigid_matrix_33 = 0.0
        self.rigid_dx = 0.0
        self.rigid_dy = 0.0
        self.rigid_dz = 0.0

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
        self.rigid_perp_dirn = 0
        self.rigid_constrained_dof_count = 0
        self.rigid_constrained_dof_1 = 0
        self.rigid_constrained_dof_2 = 0
        self.rigid_constrained_dof_3 = 0
        self.rigid_retained_dof_count = 0
        self.rigid_retained_dof_1 = 0
        self.rigid_retained_dof_2 = 0
        self.rigid_retained_dof_3 = 0
        self.rigid_matrix_row_count = 0
        self.rigid_matrix_col_count = 0
        self.rigid_matrix_11 = 0.0
        self.rigid_matrix_12 = 0.0
        self.rigid_matrix_13 = 0.0
        self.rigid_matrix_21 = 0.0
        self.rigid_matrix_22 = 0.0
        self.rigid_matrix_23 = 0.0
        self.rigid_matrix_31 = 0.0
        self.rigid_matrix_32 = 0.0
        self.rigid_matrix_33 = 0.0
        self.rigid_dx = 0.0
        self.rigid_dy = 0.0
        self.rigid_dz = 0.0


@fieldwise_init
struct UniformExcitationInput(ImplicitlyCopyable, Movable):
    var direction: Int
    var accel: Int
    var initial_velocity: Float64


struct PatternInput(ImplicitlyCopyable, Movable):
    var has_pattern: Bool
    var type: String
    var type_tag: Int
    var has_time_series: Bool
    var time_series: Int
    var excitations: List[UniformExcitationInput]
    var sp_constraints: List[NodalLoadInput]

    fn __init__(out self):
        self.has_pattern = False
        self.type = "Plain"
        self.type_tag = PatternTypeTag.Plain
        self.has_time_series = False
        self.time_series = -1
        self.excitations = []
        self.sp_constraints = []

    fn __copyinit__(out self, existing: Self):
        self.has_pattern = existing.has_pattern
        self.type = existing.type
        self.type_tag = existing.type_tag
        self.has_time_series = existing.has_time_series
        self.time_series = existing.time_series
        self.excitations = existing.excitations.copy()
        self.sp_constraints = existing.sp_constraints.copy()


@fieldwise_init
struct RayleighCoefficient(ImplicitlyCopyable, Movable):
    var tag: Int
    var coefficient: Float64


struct RayleighInput(ImplicitlyCopyable, Movable):
    var has_rayleigh: Bool
    var alpha_m: Float64
    var beta_k: Float64
    var beta_k_init: Float64
    var beta_k_comm: Float64
    var node_alpha_m: List[RayleighCoefficient]
    var element_beta_k_init: List[RayleighCoefficient]

    fn __init__(out self):
        self.has_rayleigh = False
        self.alpha_m = 0.0
        self.beta_k = 0.0
        self.beta_k_init = 0.0
        self.beta_k_comm = 0.0
        self.node_alpha_m = []
        self.element_beta_k_init = []

    fn __copyinit__(out self, existing: Self):
        self.has_rayleigh = existing.has_rayleigh
        self.alpha_m = existing.alpha_m
        self.beta_k = existing.beta_k
        self.beta_k_init = existing.beta_k_init
        self.beta_k_comm = existing.beta_k_comm
        self.node_alpha_m = existing.node_alpha_m.copy()
        self.element_beta_k_init = existing.element_beta_k_init.copy()

    fn has_regional_damping(self) -> Bool:
        return len(self.node_alpha_m) > 0 or len(self.element_beta_k_init) > 0


struct DampingInput(ImplicitlyCopyable, Movable):
    var tag: Int
    var type: String
    var beta: Float64
    var activate_time: Float64
    var deactivate_time: Float64
    var factor_ts_tag: Int
    var factor_ts_index: Int

    fn __init__(out self):
        self.tag = -1
        self.type = ""
        self.beta = 0.0
        self.activate_time = 0.0
        self.deactivate_time = 1.0e20
        self.factor_ts_tag = -1
        self.factor_ts_index = -1


@fieldwise_init
struct MaterialStageInput(ImplicitlyCopyable, Movable):
    var material: Int
    var stage: Int


@fieldwise_init
struct PermeabilityUpdate(ImplicitlyCopyable, Movable):
    var element: Int
    var axis: String
    var value: Float64


@fieldwise_init
struct PrescribedPressureInput(ImplicitlyCopyable, Movable):
    var node: Int
    var dof: Int
    var velocity: Float64
    var displacement: Float64
    var acceleration: Float64


struct StageInput(ImplicitlyCopyable, Movable):
    var constant_loads: List[NodalLoadInput]
    var constant_element_loads: List[ElementLoadInput]
    var sample_before: List[Int]
    var resume_load_control: Bool
    var material_stages: List[MaterialStageInput]
    var has_initial_state_reset: Bool
    var contact_friction: Optional[Bool]
    var capture_initial_state: Bool
    var removed_nodes: List[Int]
    var removed_elements: List[Int]
    var permeability_updates: List[PermeabilityUpdate]
    var prescribed_pressures: List[PrescribedPressureInput]
    var analysis: AnalysisInput
    var analysis_integrator_targets_pool: List[Float64]
    var analysis_solver_chain_pool: List[SolverAttemptInput]
    var pattern: PatternInput
    var rayleigh: RayleighInput
    var loads: List[NodalLoadInput]
    var element_loads: List[ElementLoadInput]
    var has_load_const: Bool
    var load_const_time: Float64
    var time_series: List[TimeSeriesInput]
    var time_series_values: List[Float64]
    var time_series_times: List[Float64]

    fn __init__(out self):
        self.constant_loads = []
        self.constant_element_loads = []
        self.sample_before = []
        self.resume_load_control = False
        self.material_stages = []
        self.has_initial_state_reset = False
        self.contact_friction = None
        self.capture_initial_state = False
        self.removed_nodes = []
        self.removed_elements = []
        self.permeability_updates = []
        self.prescribed_pressures = []
        self.analysis = AnalysisInput()
        self.analysis_integrator_targets_pool = []
        self.analysis_solver_chain_pool = []
        self.pattern = PatternInput()
        self.rayleigh = RayleighInput()
        self.loads = []
        self.element_loads = []
        self.has_load_const = False
        self.load_const_time = 0.0
        self.time_series = []
        self.time_series_values = []
        self.time_series_times = []

    fn __copyinit__(out self, existing: Self):
        self.constant_loads = existing.constant_loads.copy()
        self.constant_element_loads = existing.constant_element_loads.copy()
        self.sample_before = existing.sample_before.copy()
        self.resume_load_control = existing.resume_load_control
        self.material_stages = existing.material_stages.copy()
        self.has_initial_state_reset = existing.has_initial_state_reset
        self.contact_friction = existing.contact_friction
        self.capture_initial_state = existing.capture_initial_state
        self.removed_nodes = existing.removed_nodes.copy()
        self.removed_elements = existing.removed_elements.copy()
        self.permeability_updates = existing.permeability_updates.copy()
        self.prescribed_pressures = existing.prescribed_pressures.copy()
        self.analysis = existing.analysis
        self.analysis_integrator_targets_pool = existing.analysis_integrator_targets_pool.copy()
        self.analysis_solver_chain_pool = existing.analysis_solver_chain_pool.copy()
        self.pattern = existing.pattern
        self.rayleigh = existing.rayleigh
        self.loads = existing.loads.copy()
        self.element_loads = existing.element_loads.copy()
        self.has_load_const = existing.has_load_const
        self.load_const_time = existing.load_const_time
        self.time_series = existing.time_series.copy()
        self.time_series_values = existing.time_series_values.copy()
        self.time_series_times = existing.time_series_times.copy()


struct RecorderInput(ImplicitlyCopyable, Movable):
    var material_stress: Bool
    var integration_point: Int
    var stress_components: Int
    var type_tag: Int
    var output: String
    var start_stage: Int
    var end_stage: Int
    var delta_t: Float64
    var node_offset: Int
    var node_count: Int
    var element_offset: Int
    var element_count: Int
    var dof_offset: Int
    var dof_count: Int
    var mode_offset: Int
    var mode_count: Int
    var section_offset: Int
    var section_count: Int
    var i_node: Int
    var j_node: Int
    var drift_dof: Int
    var perp_dirn: Int
    var time_series_tag: Int

    fn __init__(out self):
        self.material_stress = False
        self.integration_point = 0
        self.stress_components = 0
        self.type_tag = RecorderTypeTag.Unknown
        self.output = ""
        self.start_stage = 0
        self.end_stage = 2147483647
        self.delta_t = 0.0
        self.node_offset = 0
        self.node_count = 0
        self.element_offset = 0
        self.element_count = 0
        self.dof_offset = 0
        self.dof_count = 0
        self.mode_offset = 0
        self.mode_count = 0
        self.section_offset = 0
        self.section_count = 0
        self.i_node = -1
        self.j_node = -1
        self.drift_dof = -1
        self.perp_dirn = -1
        self.time_series_tag = -1


struct CaseInput(Movable):
    var model: ModelInput
    var nodes: List[NodeInput]
    var sections: List[SectionInput]
    var fiber_patches: List[FiberPatchInput]
    var fiber_layers: List[FiberLayerInput]
    var fibers: List[FiberInput]
    var shell_layers: List[ShellLayerInput]
    var pimy_definitions: List[PressureIndependMultiYieldDef]
    var pdmy_definitions: List[PressureDependMultiYieldDef]
    var materials: List[MaterialInput]
    var elements: List[ElementInput]
    var element_loads: List[ElementLoadInput]
    var loads: List[NodalLoadInput]
    var masses: List[MassInput]
    var analysis: AnalysisInput
    var mp_constraints: List[MPConstraintInput]
    var pattern: PatternInput
    var rayleigh: RayleighInput
    var time_series: List[TimeSeriesInput]
    var time_series_values: List[Float64]
    var time_series_times: List[Float64]
    var dampings: List[DampingInput]
    var stages: List[StageInput]
    var analysis_integrator_targets_pool: List[Float64]
    var analysis_solver_chain_pool: List[SolverAttemptInput]
    var recorder_nodes_pool: List[Int]
    var recorder_elements_pool: List[Int]
    var recorder_dofs_pool: List[Int]
    var recorder_modes_pool: List[Int]
    var recorder_sections_pool: List[Int]
    var recorders: List[RecorderInput]

    fn __init__(out self):
        self.model = ModelInput(0, 0)
        self.nodes = []
        self.sections = []
        self.fiber_patches = []
        self.fiber_layers = []
        self.fibers = []
        self.shell_layers = []
        self.pimy_definitions = []
        self.pdmy_definitions = []
        self.materials = []
        self.elements = []
        self.element_loads = []
        self.loads = []
        self.masses = []
        self.analysis = AnalysisInput()
        self.mp_constraints = []
        self.pattern = PatternInput()
        self.rayleigh = RayleighInput()
        self.time_series = []
        self.time_series_values = []
        self.time_series_times = []
        self.dampings = []
        self.stages = []
        self.analysis_integrator_targets_pool = []
        self.analysis_solver_chain_pool = []
        self.recorder_nodes_pool = []
        self.recorder_elements_pool = []
        self.recorder_dofs_pool = []
        self.recorder_modes_pool = []
        self.recorder_sections_pool = []
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
    if type_name == "forceBeamColumn3d":
        return ElementTypeTag.ForceBeamColumn3d
    if type_name == "dispBeamColumn3d":
        return ElementTypeTag.DispBeamColumn3d
    if type_name == "truss" or type_name == "corotTruss":
        return ElementTypeTag.Truss
    if type_name == "zeroLength" or type_name == "CoupledZeroLength":
        return ElementTypeTag.ZeroLength
    if type_name == "MVLEM":
        return ElementTypeTag.MVLEM
    if type_name == "CatenaryCable":
        return ElementTypeTag.CatenaryCable
    if type_name == "SFI_MVLEM":
        return ElementTypeTag.SFIMVLEM
    if type_name == "twoNodeLink":
        return ElementTypeTag.TwoNodeLink
    if type_name == "zeroLengthSection":
        return ElementTypeTag.ZeroLengthSection
    if type_name == "SSPbrickUP":
        return ElementTypeTag.SSPBrickUP
    if type_name == "quadUP" or type_name == "9_4_QuadUP":
        return ElementTypeTag.QuadUP
    if continuum_nodes(type_name) > 0:
        return ElementTypeTag.Continuum
    if type_name == "BeamContact2D":
        return ElementTypeTag.BeamContact2D
    if type_name == "fourNodeQuad":
        return ElementTypeTag.FourNodeQuad
    if type_name == "tri31":
        return ElementTypeTag.Tri31
    if (
        type_name == "shell"
        or type_name == "ShellDKGT"
        or type_name == "ShellDKGQ"
        or type_name == "ShellNLDKGQ"
        or type_name == "ASDShellQ4"
    ):
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


fn beam_integration_tag(integration_name: String) -> Int:
    if integration_name == "HingeRadau":
        return BeamIntegrationTag.HingeRadau
    if integration_name == "Lobatto":
        return BeamIntegrationTag.Lobatto
    if integration_name == "Legendre":
        return BeamIntegrationTag.Legendre
    if integration_name == "Radau":
        return BeamIntegrationTag.Radau
    return BeamIntegrationTag.Unknown


fn numberer_tag(numberer_name: String) -> Int:
    if len(numberer_name) == 0:
        return NumbererTag.Unknown
    if numberer_name == "RCM":
        return NumbererTag.RCM
    if numberer_name == "Plain":
        return NumbererTag.Plain
    return NumbererTag.Unknown


fn analysis_system_tag(system_name: String) -> Int:
    if system_name == "BandGeneral":
        return AnalysisSystemTag.BandGeneral
    if system_name == "BandSPD":
        return AnalysisSystemTag.BandSPD
    if system_name == "ProfileSPD":
        return AnalysisSystemTag.ProfileSPD
    if system_name == "SuperLU":
        return AnalysisSystemTag.SuperLU
    if system_name == "UmfPack":
        return AnalysisSystemTag.UmfPack
    if system_name == "FullGeneral":
        return AnalysisSystemTag.FullGeneral
    if system_name == "SparseSYM":
        return AnalysisSystemTag.SparseSYM
    return AnalysisSystemTag.Unknown


fn canonical_analysis_system_name(system_name: String) -> String:
    var canonical = system_name
    if len(canonical) == 0:
        canonical = "BandGeneral"
    if canonical == "BandGEN" or canonical == "BandGen":
        canonical = "BandGeneral"
    elif canonical == "SparseGeneral" or canonical == "SparseGEN":
        canonical = "SuperLU"
    elif canonical == "SparseSPD":
        canonical = "SparseSYM"
    elif canonical == "Umfpack":
        canonical = "UmfPack"
    elif canonical == "Mumps":
        canonical = "SuperLU"
    if analysis_system_tag(canonical) == AnalysisSystemTag.Unknown:
        abort("unsupported analysis system: " + canonical)
    return canonical


fn _validate_analysis_system_options(
    system_name: String, system_tag: Int, system_options: List[String]
) raises:
    var count = len(system_options)
    if count == 0:
        return

    if (
        system_tag == AnalysisSystemTag.BandGeneral
        or system_tag == AnalysisSystemTag.BandSPD
        or system_tag == AnalysisSystemTag.ProfileSPD
        or system_tag == AnalysisSystemTag.FullGeneral
    ):
        abort("analysis system `" + system_name + "` does not accept system_options")

    if system_tag == AnalysisSystemTag.SuperLU:
        var i = 0
        while i < count:
            var option = system_options[i]
            if (
                option == "p"
                or option == "piv"
                or option == "-piv"
                or option == "s"
                or option == "symmetric"
                or option == "-symmetric"
                or option == "-symm"
                or option == "u"
                or option == "unsymmetric"
                or option == "-unsymm"
            ):
                i += 1
                continue
            if (
                option == "np"
                or option == "-np"
                or option == "npRow"
                or option == "-npRow"
                or option == "npCol"
                or option == "-npCol"
                or option == "permSpec"
                or option == "-permSpec"
            ):
                if i + 1 >= count:
                    abort(
                        "analysis system_options for `SuperLU` expects a value after `"
                        + option
                        + "`"
                    )
                _ = Int(system_options[i + 1])
                i += 2
                continue
            abort("unsupported analysis system option `" + option + "` for system `SuperLU`")
        return

    if system_tag == AnalysisSystemTag.UmfPack:
        var i = 0
        while i < count:
            var option = system_options[i]
            if (
                option == "-factorOnce"
                or option == "-FactorOnce"
                or option == "-printTime"
                or option == "-time"
            ):
                i += 1
                continue
            if option == "-lValueFact" or option == "-lvalueFact" or option == "-LVALUE":
                if i + 1 >= count:
                    abort(
                        "analysis system_options for `UmfPack` expects a value after `"
                        + option
                        + "`"
                    )
                _ = Int(system_options[i + 1])
                i += 2
                continue
            abort("unsupported analysis system option `" + option + "` for system `UmfPack`")
        return

    if system_tag == AnalysisSystemTag.SparseSYM:
        if count == 1:
            var ordering = Int(system_options[0])
            if ordering < 1 or ordering > 3:
                abort(
                    "analysis system_options for `SparseSYM` ordering must be 1 (MMD), 2 (ND), or 3"
                    " (RCM)"
                )
            return
        abort(
            "analysis system_options for `SparseSYM` expects at most one ordering value (1, 2,"
            " or 3)"
        )

    abort("unsupported analysis system tag")


fn _apply_analysis_system_options(mut analysis: AnalysisInput, system_options: List[String]) raises:
    analysis.superlu_prefer_symmetric = False
    analysis.superlu_enable_pivot = False
    analysis.superlu_np_row = -1
    analysis.superlu_np_col = -1
    analysis.superlu_perm_spec = -1
    analysis.umfpack_factor_once = False
    analysis.umfpack_print_time = False
    analysis.umfpack_lvalue_fact = -1
    analysis.sparse_sym_ordering = 0

    var count = len(system_options)
    if count == 0:
        return

    if analysis.system_tag == AnalysisSystemTag.SuperLU:
        var i = 0
        while i < count:
            var option = system_options[i]
            if option == "p" or option == "piv" or option == "-piv":
                analysis.superlu_enable_pivot = True
                i += 1
                continue
            if (
                option == "s"
                or option == "symmetric"
                or option == "-symmetric"
                or option == "-symm"
            ):
                analysis.superlu_prefer_symmetric = True
                i += 1
                continue
            if option == "u" or option == "unsymmetric" or option == "-unsymm":
                analysis.superlu_prefer_symmetric = False
                i += 1
                continue
            if option == "np" or option == "-np":
                var value = Int(system_options[i + 1])
                analysis.superlu_np_row = value
                analysis.superlu_np_col = value
                i += 2
                continue
            if option == "npRow" or option == "-npRow":
                analysis.superlu_np_row = Int(system_options[i + 1])
                i += 2
                continue
            if option == "npCol" or option == "-npCol":
                analysis.superlu_np_col = Int(system_options[i + 1])
                i += 2
                continue
            if option == "permSpec" or option == "-permSpec":
                analysis.superlu_perm_spec = Int(system_options[i + 1])
                i += 2
                continue
            i += 1
        return

    if analysis.system_tag == AnalysisSystemTag.UmfPack:
        var i = 0
        while i < count:
            var option = system_options[i]
            if option == "-factorOnce" or option == "-FactorOnce":
                analysis.umfpack_factor_once = True
                i += 1
                continue
            if option == "-printTime" or option == "-time":
                analysis.umfpack_print_time = True
                i += 1
                continue
            if option == "-lValueFact" or option == "-lvalueFact" or option == "-LVALUE":
                analysis.umfpack_lvalue_fact = Int(system_options[i + 1])
                i += 2
                continue
            i += 1
        return

    if analysis.system_tag == AnalysisSystemTag.SparseSYM:
        analysis.sparse_sym_ordering = Int(system_options[0])


fn analysis_type_tag(type_name: String) -> Int:
    if type_name == "static_linear":
        return AnalysisTypeTag.StaticLinear
    if type_name == "static_nonlinear":
        return AnalysisTypeTag.StaticNonlinear
    if type_name == "transient_linear":
        return AnalysisTypeTag.TransientLinear
    if type_name == "transient_nonlinear":
        return AnalysisTypeTag.TransientNonlinear
    if type_name == "staged":
        return AnalysisTypeTag.Staged
    if type_name == "modal_eigen":
        return AnalysisTypeTag.ModalEigen
    return AnalysisTypeTag.Unknown


fn constraint_handler_tag(handler_name: String) -> Int:
    if handler_name == "Plain":
        return ConstraintHandlerTag.Plain
    if handler_name == "Transformation":
        return ConstraintHandlerTag.Transformation
    if handler_name == "Lagrange":
        return ConstraintHandlerTag.Lagrange
    if handler_name == "Penalty":
        return ConstraintHandlerTag.Penalty
    return ConstraintHandlerTag.Unknown


fn force_beam_mode_tag(mode_name: String) -> Int:
    if mode_name == "auto":
        return ForceBeamModeTag.Auto
    if mode_name == "linear_if_elastic":
        return ForceBeamModeTag.LinearIfElastic
    if mode_name == "nonlinear":
        return ForceBeamModeTag.Nonlinear
    return ForceBeamModeTag.Unknown


fn analysis_algorithm_tag(algorithm_name: String) -> Int:
    if len(algorithm_name) == 0:
        return AnalysisAlgorithmTag.Unknown
    if algorithm_name == "Newton":
        return AnalysisAlgorithmTag.Newton
    if algorithm_name == "ModifiedNewton":
        return AnalysisAlgorithmTag.ModifiedNewton
    if algorithm_name == "ModifiedNewtonInitial":
        return AnalysisAlgorithmTag.ModifiedNewtonInitial
    if algorithm_name == "Broyden":
        return AnalysisAlgorithmTag.Broyden
    if algorithm_name == "NewtonLineSearch":
        return AnalysisAlgorithmTag.NewtonLineSearch
    if algorithm_name == "KrylovNewton":
        return AnalysisAlgorithmTag.KrylovNewton
    return AnalysisAlgorithmTag.Unknown


fn nonlinear_test_type_tag(test_type_name: String) -> Int:
    if test_type_name == "NormDispIncr":
        return NonlinearTestTypeTag.NormDispIncr
    if test_type_name == "NormUnbalance":
        return NonlinearTestTypeTag.NormUnbalance
    if test_type_name == "EnergyIncr":
        return NonlinearTestTypeTag.EnergyIncr
    return NonlinearTestTypeTag.Unknown


fn integrator_type_tag(integrator_name: String) -> Int:
    if len(integrator_name) == 0:
        return IntegratorTypeTag.Unknown
    if integrator_name == "LoadControl":
        return IntegratorTypeTag.LoadControl
    if integrator_name == "DisplacementControl":
        return IntegratorTypeTag.DisplacementControl
    if integrator_name == "HSConstraint":
        return IntegratorTypeTag.HSConstraint
    if integrator_name == "Newmark":
        return IntegratorTypeTag.Newmark
    return IntegratorTypeTag.Unknown


fn pattern_type_tag(type_name: String) -> Int:
    if type_name == "Plain":
        return PatternTypeTag.Plain
    if type_name == "UniformExcitation":
        return PatternTypeTag.UniformExcitation
    if type_name == "None":
        return PatternTypeTag.`None`
    return PatternTypeTag.Unknown


fn recorder_type_tag(type_name: String) -> Int:
    if type_name == "node_velocity":
        return RecorderTypeTag.NodeVelocity
    if type_name == "contact_secondary_force":
        return RecorderTypeTag.ContactSecondaryForce
    if type_name == "contact_friction_force":
        return RecorderTypeTag.ContactFrictionForce
    if type_name == "contact_scalar_force":
        return RecorderTypeTag.ContactScalarForce
    if type_name == "contact_primary_force":
        return RecorderTypeTag.ContactPrimaryForce
    if type_name == "quad_stress":
        return RecorderTypeTag.QuadStress
    if type_name == "quad_strain":
        return RecorderTypeTag.QuadStrain
    if type_name == "quad_state":
        return RecorderTypeTag.QuadState
    if type_name == "node_displacement":
        return RecorderTypeTag.NodeDisplacement
    if type_name == "element_force":
        return RecorderTypeTag.ElementForce
    if type_name == "element_local_force":
        return RecorderTypeTag.ElementLocalForce
    if type_name == "element_basic_force":
        return RecorderTypeTag.ElementBasicForce
    if type_name == "element_deformation":
        return RecorderTypeTag.ElementDeformation
    if type_name == "node_reaction":
        return RecorderTypeTag.NodeReaction
    if type_name == "drift":
        return RecorderTypeTag.Drift
    if type_name == "envelope_element_force":
        return RecorderTypeTag.EnvelopeElementForce
    if type_name == "envelope_element_local_force":
        return RecorderTypeTag.EnvelopeElementLocalForce
    if type_name == "envelope_node_displacement":
        return RecorderTypeTag.EnvelopeNodeDisplacement
    if type_name == "envelope_node_acceleration":
        return RecorderTypeTag.EnvelopeNodeAcceleration
    if type_name == "node_acceleration":
        return RecorderTypeTag.NodeAcceleration
    if type_name == "modal_eigen":
        return RecorderTypeTag.ModalEigen
    if type_name == "section_force":
        return RecorderTypeTag.SectionForce
    if type_name == "section_deformation":
        return RecorderTypeTag.SectionDeformation
    return RecorderTypeTag.Unknown


fn element_load_type_tag(type_name: String) -> Int:
    if type_name == "beamUniform":
        return ElementLoadTypeTag.BeamUniform
    if type_name == "beamPoint":
        return ElementLoadTypeTag.BeamPoint
    if type_name == "beamThermal":
        return ElementLoadTypeTag.BeamThermal
    return ElementLoadTypeTag.Unknown


fn _json_has_value(doc: JsonDocument, node_index: Int) -> Bool:
    return node_index >= 0 and doc.node_tag(node_index) != JsonValueTag.Null


fn _json_key(doc: JsonDocument, object_index: Int, key: StringSlice) raises -> Int:
    if object_index < 0:
        return -1
    var tag = doc.node_tag(object_index)
    if tag == JsonValueTag.Null:
        return -1
    if tag != JsonValueTag.Object:
        abort("expected object for key lookup: " + String(key))
    return doc.object_find(object_index, key)


fn _json_require_key(doc: JsonDocument, object_index: Int, key: StringSlice) raises -> Int:
    var node_index = _json_key(doc, object_index, key)
    if node_index < 0:
        abort("missing required key: " + String(key))
    return node_index


fn _json_expect_array_len(doc: JsonDocument, array_index: Int, field: String) -> Int:
    if array_index < 0 or doc.node_tag(array_index) == JsonValueTag.Null:
        return 0
    if doc.node_tag(array_index) != JsonValueTag.Array:
        abort(field + " must be an array")
    return doc.node_len(array_index)


fn _json_string_value(doc: JsonDocument, node_index: Int, field: String) -> String:
    if doc.node_tag(node_index) != JsonValueTag.String:
        abort(field + " must be a string")
    return doc.node_text(node_index)


fn _json_number_value(doc: JsonDocument, node_index: Int, field: String) -> Float64:
    if doc.node_tag(node_index) != JsonValueTag.Number:
        abort(field + " must be a number")
    return doc.node_number(node_index)


fn _required_material_param(doc: JsonDocument, params: Int, key: StringSlice) raises -> Float64:
    return _json_number_value(doc, _json_require_key(doc, params, key), String(key))


fn read_small_material_def(doc: JsonDocument, params: Int, kind: String) raises -> UniMaterialDef:
    if kind == "ElasticPP":
        var E = _required_material_param(doc, params, "E")
        var positive = abs(_required_material_param(doc, params, "epsyP"))
        var negative = -abs(_json_get_float(doc, params, "epsyN", -positive))
        var offset = _json_get_float(doc, params, "eps0", 0.0)
        if E <= 0.0:
            raise Error("ElasticPP E must be positive")
        return UniMaterialDef(UniMaterialTypeTag.ElasticPP, E, positive, negative, offset)
    if kind == "ElasticBilin":
        var first = _required_material_param(doc, params, "E1P")
        var second = _required_material_param(doc, params, "E2P")
        var breakpoint = abs(_required_material_param(doc, params, "eps2P"))
        var negative_first = first
        var negative_second = second
        var negative_breakpoint = -breakpoint
        if (
            _json_has_key(doc, params, "E1N")
            or _json_has_key(doc, params, "E2N")
            or _json_has_key(doc, params, "eps2N")
        ):
            negative_first = _required_material_param(doc, params, "E1N")
            negative_second = _required_material_param(doc, params, "E2N")
            negative_breakpoint = -abs(_required_material_param(doc, params, "eps2N"))
        return UniMaterialDef(
            UniMaterialTypeTag.ElasticBilin,
            first,
            second,
            breakpoint,
            negative_first,
            negative_second,
            negative_breakpoint,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
    var E = _required_material_param(doc, params, "E")
    var travel = _required_material_param(doc, params, "freeTravel")
    var initial = _required_material_param(doc, params, "freeTravelInitial")
    var mode = _required_material_param(doc, params, "RatType")
    if E <= 0.0 or (mode == 1.0 and travel <= 0.0):
        raise Error("Ratchet requires positive stiffness and positive travel in discrete mode")
    return UniMaterialDef(UniMaterialTypeTag.Ratchet, E, travel, initial, mode)


fn read_wall_material_def(doc: JsonDocument, params: Int, kind: String) raises -> UniMaterialDef:
    if kind == "ConcreteCM":
        var gap_close = _json_get_float(doc, params, "gap_close", 0.0)
        var monotonic = _json_get_float(doc, params, "monotonic", 0.0)
        if (gap_close != 0.0 and gap_close != 1.0) or (monotonic != 0.0 and monotonic != 1.0):
            abort("ConcreteCM gap_close and monotonic must be 0 or 1")
        return UniMaterialDef(
            UniMaterialTypeTag.ConcreteCM,
            _required_material_param(doc, params, "fpcc"),
            _required_material_param(doc, params, "epcc"),
            _required_material_param(doc, params, "Ec"),
            _required_material_param(doc, params, "rc"),
            _required_material_param(doc, params, "xcrn"),
            _required_material_param(doc, params, "ft"),
            _required_material_param(doc, params, "et"),
            _required_material_param(doc, params, "rt"),
            _required_material_param(doc, params, "xcrp"),
            gap_close,
            monotonic,
        )
    return UniMaterialDef(
        UniMaterialTypeTag.SteelMPF,
        _required_material_param(doc, params, "fyp"),
        _required_material_param(doc, params, "fyn"),
        _required_material_param(doc, params, "E0"),
        _required_material_param(doc, params, "bp"),
        _required_material_param(doc, params, "bn"),
        _required_material_param(doc, params, "R0"),
        _required_material_param(doc, params, "cR1"),
        _required_material_param(doc, params, "cR2"),
        _json_get_float(doc, params, "a1", 0.0),
        _json_get_float(doc, params, "a2", 1.0),
        _json_get_float(doc, params, "a3", 0.0),
        _json_get_float(doc, params, "a4", 1.0),
    )


fn _json_int_value(doc: JsonDocument, node_index: Int, field: String) -> Int:
    return Int(_json_number_value(doc, node_index, field))


fn _json_bool_value(doc: JsonDocument, node_index: Int, field: String) -> Bool:
    if doc.node_tag(node_index) != JsonValueTag.Bool:
        abort(field + " must be a boolean")
    return doc.node_bool(node_index)


fn _json_get_string(
    doc: JsonDocument,
    object_index: Int,
    key: StringSlice,
    default: String,
) raises -> String:
    var node_index = _json_key(doc, object_index, key)
    if node_index < 0:
        return default
    return _json_string_value(doc, node_index, String(key))


fn _json_get_float(
    doc: JsonDocument,
    object_index: Int,
    key: StringSlice,
    default: Float64,
) raises -> Float64:
    var node_index = _json_key(doc, object_index, key)
    if node_index < 0:
        return default
    return _json_number_value(doc, node_index, String(key))


fn _json_get_int(
    doc: JsonDocument, object_index: Int, key: StringSlice, default: Int
) raises -> Int:
    var node_index = _json_key(doc, object_index, key)
    if node_index < 0:
        return default
    return _json_int_value(doc, node_index, String(key))


fn _json_get_bool(
    doc: JsonDocument, object_index: Int, key: StringSlice, default: Bool
) raises -> Bool:
    var node_index = _json_key(doc, object_index, key)
    if node_index < 0:
        return default
    return _json_bool_value(doc, node_index, String(key))


fn _json_has_key(doc: JsonDocument, object_index: Int, key: StringSlice) raises -> Bool:
    return _json_key(doc, object_index, key) >= 0


fn _json_get_array_item(
    doc: JsonDocument, array_index: Int, item_index: Int, field: String
) raises -> Int:
    if doc.node_tag(array_index) != JsonValueTag.Array:
        abort(field + " must be an array")
    return doc.array_item(array_index, item_index)


fn _json_get_float_alias2(
    doc: JsonDocument,
    object_index: Int,
    key_1: StringSlice,
    key_2: StringSlice,
    default: Float64,
) raises -> Float64:
    var node_index = _json_key(doc, object_index, key_1)
    if node_index >= 0:
        return _json_number_value(doc, node_index, String(key_1))
    node_index = _json_key(doc, object_index, key_2)
    if node_index >= 0:
        return _json_number_value(doc, node_index, String(key_2))
    return default


fn _json_get_float_alias3(
    doc: JsonDocument,
    object_index: Int,
    key_1: StringSlice,
    key_2: StringSlice,
    key_3: StringSlice,
    default: Float64,
) raises -> Float64:
    var node_index = _json_key(doc, object_index, key_1)
    if node_index >= 0:
        return _json_number_value(doc, node_index, String(key_1))
    node_index = _json_key(doc, object_index, key_2)
    if node_index >= 0:
        return _json_number_value(doc, node_index, String(key_2))
    node_index = _json_key(doc, object_index, key_3)
    if node_index >= 0:
        return _json_number_value(doc, node_index, String(key_3))
    return default


fn _numeric_token_start(bytes: Span[Byte], idx: Int) -> Bool:
    var ch = bytes[idx]
    if ch >= Byte(ord("0")) and ch <= Byte(ord("9")):
        return True
    if ch == Byte(ord(".")):
        return (
            idx + 1 < len(bytes)
            and bytes[idx + 1] >= Byte(ord("0"))
            and bytes[idx + 1] <= Byte(ord("9"))
        )
    if ch == Byte(ord("+")) or ch == Byte(ord("-")):
        if idx + 1 >= len(bytes):
            return False
        var next = bytes[idx + 1]
        return (next >= Byte(ord("0")) and next <= Byte(ord("9"))) or next == Byte(ord("."))
    return False


fn _parse_numeric_values_text(text: String, values_path: String) raises -> List[Float64]:
    var values: List[Float64] = []
    var bytes = text.as_bytes()
    var i = 0
    while i < len(bytes):
        if not _numeric_token_start(bytes, i):
            i += 1
            continue

        var start = i
        if bytes[i] == Byte(ord("+")) or bytes[i] == Byte(ord("-")):
            i += 1
        var digit_count = 0
        while i < len(bytes) and bytes[i] >= Byte(ord("0")) and bytes[i] <= Byte(ord("9")):
            i += 1
            digit_count += 1
        if i < len(bytes) and bytes[i] == Byte(ord(".")):
            i += 1
            while i < len(bytes) and bytes[i] >= Byte(ord("0")) and bytes[i] <= Byte(ord("9")):
                i += 1
                digit_count += 1
        if digit_count == 0:
            i = start + 1
            continue

        if i < len(bytes) and (
            bytes[i] == Byte(ord("e"))
            or bytes[i] == Byte(ord("E"))
            or bytes[i] == Byte(ord("d"))
            or bytes[i] == Byte(ord("D"))
        ):
            var exponent_end = i + 1
            if exponent_end < len(bytes) and (
                bytes[exponent_end] == Byte(ord("+")) or bytes[exponent_end] == Byte(ord("-"))
            ):
                exponent_end += 1
            var exponent_digits = 0
            while (
                exponent_end < len(bytes)
                and bytes[exponent_end] >= Byte(ord("0"))
                and bytes[exponent_end] <= Byte(ord("9"))
            ):
                exponent_end += 1
                exponent_digits += 1
            if exponent_digits > 0:
                i = exponent_end

        var token = String(StringSlice(unsafe_from_utf8=bytes[start:i]))
        token = token.replace("D", "E").replace("d", "e")
        values.append(atof(StringSlice(token)))
    if len(values) == 0:
        abort("Path time_series values_path had no numeric values: " + values_path)
    return values^


fn _load_path_time_series_values_native(
    doc: JsonDocument, ts_index: Int, source_info: CaseSourceInfo
) raises -> List[Float64]:
    var values_path = _json_get_string(doc, ts_index, "values_path", "")
    if values_path == "":
        values_path = _json_get_string(doc, ts_index, "path", "")
    if values_path == "":
        abort("Path time_series missing values_path")

    var resolved_path = values_path
    if not values_path.startswith("/") and source_info.case_dir != "":
        var candidate = String(Path(source_info.case_dir) / values_path)
        if Path(candidate).exists():
            resolved_path = candidate
    if not Path(resolved_path).exists():
        abort("Path time_series values_path not found: " + values_path)
    return _parse_numeric_values_text(read_text_native(resolved_path), resolved_path)


fn _append_time_series_entry_native(
    doc: JsonDocument,
    ts_index: Int,
    source_info: CaseSourceInfo,
    mut parsed: List[TimeSeriesInput],
    mut values_pool: List[Float64],
    mut time_pool: List[Float64],
) raises:
    if doc.node_tag(ts_index) != JsonValueTag.Object:
        abort("time_series entry must be object")
    var parsed_entry = TimeSeriesInput()
    parsed_entry.tag = _json_get_int(doc, ts_index, "tag", -1)
    parsed_entry.factor = _json_get_float(doc, ts_index, "factor", 1.0)

    var typ = _json_get_string(doc, ts_index, "type", "")
    if typ == "PathFile":
        typ = "Path"

    if typ == "Constant":
        parsed_entry.type_tag = TimeSeriesTypeTag.Constant
        parsed.append(parsed_entry)
        return
    if typ == "Linear":
        parsed_entry.type_tag = TimeSeriesTypeTag.Linear
        parsed.append(parsed_entry)
        return
    if typ == "Path":
        parsed_entry.type_tag = TimeSeriesTypeTag.Path
        var values_index = _json_key(doc, ts_index, "values")
        if values_index >= 0 and (
            _json_has_key(doc, ts_index, "values_path") or _json_has_key(doc, ts_index, "path")
        ):
            abort("Path time_series cannot specify both values and values_path/path")

        parsed_entry.values_offset = len(values_pool)
        if values_index >= 0:
            var values_count = _json_expect_array_len(doc, values_index, "time_series values")
            parsed_entry.values_count = values_count
            for i in range(values_count):
                values_pool.append(
                    _json_number_value(
                        doc,
                        _json_get_array_item(doc, values_index, i, "time_series values"),
                        "time_series values",
                    )
                )
        else:
            var loaded_values = _load_path_time_series_values_native(doc, ts_index, source_info)
            parsed_entry.values_count = len(loaded_values)
            for i in range(len(loaded_values)):
                values_pool.append(loaded_values[i])
        parsed_entry.has_dt = _json_has_key(doc, ts_index, "dt")
        if parsed_entry.has_dt:
            parsed_entry.dt = _json_get_float(doc, ts_index, "dt", 0.0)
        parsed_entry.start_time = _json_get_float(doc, ts_index, "start_time", 0.0)
        parsed_entry.use_last = _json_get_bool(doc, ts_index, "use_last", False)
        var time_index = _json_key(doc, ts_index, "time")
        if time_index >= 0:
            parsed_entry.time_offset = len(time_pool)
            parsed_entry.time_count = _json_expect_array_len(doc, time_index, "time_series time")
            for i in range(parsed_entry.time_count):
                time_pool.append(
                    _json_number_value(
                        doc,
                        _json_get_array_item(doc, time_index, i, "time_series time"),
                        "time_series time",
                    )
                )
        parsed.append(parsed_entry)
        return
    if typ == "Trig":
        parsed_entry.type_tag = TimeSeriesTypeTag.Trig
        if not _json_has_key(doc, ts_index, "t_start") or not _json_has_key(
            doc, ts_index, "t_finish"
        ):
            abort("Trig time_series missing t_start/t_finish")
        if not _json_has_key(doc, ts_index, "period"):
            abort("Trig time_series missing period")
        parsed_entry.t_start = _json_get_float(doc, ts_index, "t_start", 0.0)
        parsed_entry.t_finish = _json_get_float(doc, ts_index, "t_finish", 0.0)
        parsed_entry.period = _json_get_float(doc, ts_index, "period", 0.0)
        parsed_entry.phase_shift = _json_get_float(doc, ts_index, "phase_shift", 0.0)
        parsed_entry.zero_shift = _json_get_float(doc, ts_index, "zero_shift", 0.0)
        parsed.append(parsed_entry)
        return
    abort("unsupported time_series type: " + typ)


fn _append_time_series_inputs_native(
    doc: JsonDocument,
    owner_index: Int,
    source_info: CaseSourceInfo,
    mut parsed: List[TimeSeriesInput],
    mut values_pool: List[Float64],
    mut time_pool: List[Float64],
) raises:
    var ts_index = _json_key(doc, owner_index, "time_series")
    if ts_index < 0:
        return
    if doc.node_tag(ts_index) == JsonValueTag.Array:
        for i in range(doc.node_len(ts_index)):
            _append_time_series_entry_native(
                doc,
                doc.array_item(ts_index, i),
                source_info,
                parsed,
                values_pool,
                time_pool,
            )
        return
    if doc.node_tag(ts_index) == JsonValueTag.Object:
        _append_time_series_entry_native(doc, ts_index, source_info, parsed, values_pool, time_pool)
        return
    if doc.node_tag(ts_index) == JsonValueTag.Null:
        return
    abort("time_series must be list or object")


fn parse_analysis_input_from_native(
    doc: JsonDocument,
    analysis_index: Int,
    mut integrator_targets_pool: List[Float64],
    mut solver_chain_pool: List[SolverAttemptInput],
) raises -> AnalysisInput:
    var analysis = AnalysisInput()
    analysis.type = _json_get_string(doc, analysis_index, "type", "static_linear")
    analysis.type_tag = analysis_type_tag(analysis.type)
    analysis.constraints = _json_get_string(doc, analysis_index, "constraints", "Plain")
    analysis.constraints_tag = constraint_handler_tag(analysis.constraints)
    analysis.penalty_sp = _json_get_float(doc, analysis_index, "penalty_sp", 0.0)
    analysis.penalty_mp = _json_get_float(doc, analysis_index, "penalty_mp", 0.0)
    var modal_damping = _json_key(doc, analysis_index, "modal_damping")
    for i in range(_json_expect_array_len(doc, modal_damping, "modal_damping")):
        analysis.supplemental.modal_damping.append(
            _json_number_value(doc, doc.array_item(modal_damping, i), "modal_damping")
        )
    var motions = _json_key(doc, analysis_index, "support_motions")
    for i in range(_json_expect_array_len(doc, motions, "support_motions")):
        var motion = doc.array_item(motions, i)
        analysis.supplemental.support_motions.append(
            SupportMotionInput(
                _json_get_int(doc, motion, "node", -1),
                _json_get_int(doc, motion, "dof", -1),
                _json_get_int(doc, motion, "time_series", -1),
                _json_get_int(doc, motion, "velocity_series", -1),
                _json_get_int(doc, motion, "acceleration_series", -1),
                _json_get_float(doc, motion, "factor", 1.0),
            )
        )
    analysis.numberer_tag = numberer_tag(_json_get_string(doc, analysis_index, "numberer", ""))
    analysis.steps = _json_get_int(doc, analysis_index, "steps", 1)
    analysis.num_modes = _json_get_int(doc, analysis_index, "num_modes", 0)
    analysis.standard_eigen = _json_get_bool(doc, analysis_index, "standard_eigen", False)
    analysis.force_beam_mode = _json_get_string(doc, analysis_index, "force_beam_mode", "auto")
    analysis.force_beam_mode_tag = force_beam_mode_tag(analysis.force_beam_mode)
    analysis.dt = _json_get_float(doc, analysis_index, "dt", 0.0)
    analysis.start_time = _json_get_float(doc, analysis_index, "start_time", 0.0)
    analysis.has_start_time = _json_has_key(doc, analysis_index, "start_time")
    analysis.algorithm = _json_get_string(doc, analysis_index, "algorithm", "Newton")
    analysis.algorithm_tag = analysis_algorithm_tag(analysis.algorithm)
    analysis.max_iters = _json_get_int(doc, analysis_index, "max_iters", 20)
    analysis.tol = _json_get_float(doc, analysis_index, "tol", 1.0e-10)
    if _json_has_key(doc, analysis_index, "rel_tol"):
        abort("analysis rel_tol is unsupported")
    analysis.test_type = _json_get_string(doc, analysis_index, "test_type", "NormUnbalance")
    analysis.test_type_tag = nonlinear_test_type_tag(analysis.test_type)

    var step_retry_index = _json_key(doc, analysis_index, "step_retry")
    analysis.step_retry_restore_primary_after_success = _json_get_bool(
        doc, step_retry_index, "restore_primary_after_success", True
    )
    analysis.step_retry_continue_after_failure = (
        _json_get_string(doc, step_retry_index, "continue_after_failure", "") != ""
    )
    analysis.step_retry_enabled = analysis.step_retry_continue_after_failure
    analysis.step_retry_continue_target_disp = _json_get_float(
        doc, step_retry_index, "continue_target_disp", 0.0
    )
    analysis.step_retry_continue_max_steps = _json_get_int(
        doc, step_retry_index, "continue_max_steps", 0
    )

    var system_name = ""
    if _json_has_key(doc, analysis_index, "system"):
        system_name = _json_get_string(doc, analysis_index, "system", "")
    elif _json_has_key(doc, analysis_index, "solver"):
        system_name = _json_get_string(doc, analysis_index, "solver", "")
    analysis.system = canonical_analysis_system_name(system_name)
    analysis.system_tag = analysis_system_tag(analysis.system)

    var system_options_index = _json_key(doc, analysis_index, "system_options")
    var system_options: List[String] = []
    if _json_has_value(doc, system_options_index):
        if doc.node_tag(system_options_index) != JsonValueTag.Array:
            abort("analysis system_options must be a list")
        for i in range(doc.node_len(system_options_index)):
            system_options.append(
                _json_string_value(
                    doc,
                    doc.array_item(system_options_index, i),
                    "analysis system_options",
                )
            )
    _validate_analysis_system_options(analysis.system, analysis.system_tag, system_options)
    _apply_analysis_system_options(analysis, system_options)
    for i in range(len(system_options)):
        if i > 0:
            analysis.system_options_serialized += "\x1f"
        analysis.system_options_serialized += system_options[i]

    analysis.band_threshold = _json_get_int(doc, analysis_index, "band_threshold", 128)

    var integrator_index = _json_key(doc, analysis_index, "integrator")
    if _json_has_key(doc, integrator_index, "displacement_limit"):
        analysis.displacement_limit = _json_get_float(
            doc, integrator_index, "displacement_limit", 0.0
        )
    var default_integrator_type = ""
    if analysis.type_tag == AnalysisTypeTag.StaticNonlinear:
        default_integrator_type = "LoadControl"
    elif (
        analysis.type_tag == AnalysisTypeTag.TransientLinear
        or analysis.type_tag == AnalysisTypeTag.TransientNonlinear
    ):
        default_integrator_type = "Newmark"
    analysis.integrator_type = _json_get_string(
        doc, integrator_index, "type", default_integrator_type
    )
    analysis.integrator_tag = integrator_type_tag(analysis.integrator_type)
    analysis.integrator_gamma = _json_get_float(doc, integrator_index, "gamma", 0.5)
    analysis.integrator_beta = _json_get_float(doc, integrator_index, "beta", 0.25)
    analysis.has_integrator_num_iter = _json_has_key(
        doc, integrator_index, "num_iter"
    ) or _json_has_key(doc, integrator_index, "numIter")
    analysis.integrator_num_iter = _json_get_int(
        doc,
        integrator_index,
        "num_iter",
        _json_get_int(doc, integrator_index, "numIter", 1),
    )
    analysis.has_integrator_step = _json_has_key(doc, integrator_index, "step")
    analysis.integrator_step = _json_get_float(doc, integrator_index, "step", 1.0)
    analysis.has_integrator_min_step = _json_has_key(doc, integrator_index, "min_step")
    analysis.integrator_min_step = _json_get_float(
        doc, integrator_index, "min_step", analysis.integrator_step
    )
    analysis.has_integrator_max_step = _json_has_key(doc, integrator_index, "max_step")
    analysis.integrator_max_step = _json_get_float(
        doc, integrator_index, "max_step", analysis.integrator_step
    )
    if _json_has_key(doc, integrator_index, "node"):
        analysis.integrator_node = _json_get_int(doc, integrator_index, "node", 0)
    if _json_has_key(doc, integrator_index, "dof"):
        analysis.integrator_dof = _json_get_int(doc, integrator_index, "dof", 0)
    analysis.integrator_cutback = _json_get_float(
        doc,
        integrator_index,
        "cutback",
        _json_get_float(doc, analysis_index, "cutback", 0.5),
    )
    analysis.integrator_max_cutbacks = _json_get_int(
        doc,
        integrator_index,
        "max_cutbacks",
        _json_get_int(doc, analysis_index, "max_cutbacks", 8),
    )
    analysis.has_integrator_min_du = _json_has_key(
        doc, integrator_index, "min_du"
    ) or _json_has_key(doc, integrator_index, "minIncrement")
    analysis.integrator_min_du = _json_get_float(
        doc,
        integrator_index,
        "min_du",
        _json_get_float(
            doc,
            integrator_index,
            "minIncrement",
            _json_get_float(doc, analysis_index, "min_du", 1.0e-10),
        ),
    )
    analysis.has_integrator_max_du = _json_has_key(
        doc, integrator_index, "max_du"
    ) or _json_has_key(doc, integrator_index, "maxIncrement")
    analysis.integrator_max_du = _json_get_float(
        doc,
        integrator_index,
        "max_du",
        _json_get_float(doc, integrator_index, "maxIncrement", 0.0),
    )
    analysis.has_integrator_du = _json_has_key(doc, integrator_index, "du")
    if analysis.has_integrator_du:
        analysis.integrator_du = _json_get_float(doc, integrator_index, "du", 0.0)
    if analysis.integrator_tag == IntegratorTypeTag.HSConstraint:
        analysis.hs_constraint = HSConstraint(
            _json_get_float(doc, integrator_index, "arc_length", 0.0),
            _json_get_float(doc, integrator_index, "psi_u", 1.0),
            _json_get_float(doc, integrator_index, "psi_f", 1.0),
            _json_get_float(doc, integrator_index, "u_ref", 1.0),
        )
    var targets_index = _json_key(doc, integrator_index, "targets")
    if _json_has_value(doc, targets_index):
        analysis.integrator_targets_offset = len(integrator_targets_pool)
        analysis.integrator_targets_count = _json_expect_array_len(
            doc, targets_index, "integrator targets"
        )
        for i in range(analysis.integrator_targets_count):
            integrator_targets_pool.append(
                _json_number_value(
                    doc,
                    doc.array_item(targets_index, i),
                    "integrator targets",
                )
            )
    var primary_algorithm_options_index = _json_key(doc, analysis_index, "algorithm_options")
    analysis.solver_chain_offset = len(solver_chain_pool)
    analysis.has_solver_chain_override = _json_has_key(doc, analysis_index, "solver_chain")
    if analysis.has_solver_chain_override:
        var solver_chain_index = _json_require_key(doc, analysis_index, "solver_chain")
        for i in range(_json_expect_array_len(doc, solver_chain_index, "solver_chain")):
            var attempt_index = doc.array_item(solver_chain_index, i)
            var attempt = SolverAttemptInput()
            var attempt_algorithm_options_index = _json_key(doc, attempt_index, "algorithm_options")
            attempt.algorithm = _json_get_string(
                doc, attempt_index, "algorithm", analysis.algorithm
            )
            attempt.algorithm_tag = analysis_algorithm_tag(attempt.algorithm)
            attempt.broyden_count = _json_get_int(
                doc,
                attempt_index,
                "broyden_count",
                _json_get_int(doc, attempt_algorithm_options_index, "max_iters", 0),
            )
            attempt.line_search_eta = _json_get_float(
                doc,
                attempt_index,
                "line_search_eta",
                _json_get_float(doc, attempt_algorithm_options_index, "alpha", 1.0),
            )
            attempt.krylov_max_dim = _json_get_int(
                doc,
                attempt_index,
                "krylov_max_dim",
                _json_get_int(doc, attempt_algorithm_options_index, "maxDim", 0),
            )
            attempt.test_type = _json_get_string(
                doc, attempt_index, "test_type", analysis.test_type
            )
            attempt.test_type_tag = nonlinear_test_type_tag(attempt.test_type)
            attempt.max_iters = _json_get_int(doc, attempt_index, "max_iters", analysis.max_iters)
            attempt.tol = _json_get_float(doc, attempt_index, "tol", analysis.tol)
            if _json_has_key(doc, attempt_index, "rel_tol"):
                abort("solver_chain rel_tol is unsupported")
            solver_chain_pool.append(attempt^)
            analysis.solver_chain_count += 1
        return analysis^

    var primary_attempt = SolverAttemptInput()
    primary_attempt.algorithm = analysis.algorithm
    primary_attempt.algorithm_tag = analysis.algorithm_tag
    primary_attempt.broyden_count = _json_get_int(
        doc, primary_algorithm_options_index, "max_iters", 0
    )
    primary_attempt.line_search_eta = _json_get_float(
        doc, primary_algorithm_options_index, "alpha", 1.0
    )
    primary_attempt.krylov_max_dim = _json_get_int(
        doc, primary_algorithm_options_index, "maxDim", 0
    )
    primary_attempt.test_type = analysis.test_type
    primary_attempt.test_type_tag = analysis.test_type_tag
    primary_attempt.max_iters = analysis.max_iters
    primary_attempt.tol = analysis.tol
    solver_chain_pool.append(primary_attempt^)
    analysis.solver_chain_count += 1
    return analysis^


fn parse_pattern_input_from_native(doc: JsonDocument, pattern_index: Int) raises -> PatternInput:
    var pattern = PatternInput()
    if not _json_has_value(doc, pattern_index):
        return pattern^
    pattern.has_pattern = True
    pattern.type = _json_get_string(doc, pattern_index, "type", "Plain")
    pattern.type_tag = pattern_type_tag(pattern.type)
    if _json_has_key(doc, pattern_index, "time_series"):
        pattern.has_time_series = True
        pattern.time_series = _json_get_int(doc, pattern_index, "time_series", -1)
    if pattern.type_tag == PatternTypeTag.UniformExcitation:
        var excitations = _json_key(doc, pattern_index, "excitations")
        if _json_has_value(doc, excitations):
            for i in range(_json_expect_array_len(doc, excitations, "excitations")):
                pattern.excitations.append(
                    _parse_uniform_excitation(doc, doc.array_item(excitations, i))
                )
        else:
            pattern.excitations.append(_parse_uniform_excitation(doc, pattern_index))
    var constraints = _json_key(doc, pattern_index, "sp_constraints")
    if _json_has_value(doc, constraints):
        for i in range(_json_expect_array_len(doc, constraints, "sp_constraints")):
            pattern.sp_constraints.append(
                parse_nodal_load_input_from_native(doc, doc.array_item(constraints, i))
            )
    return pattern^


fn _parse_uniform_excitation(doc: JsonDocument, index: Int) raises -> UniformExcitationInput:
    return UniformExcitationInput(
        _json_get_int(doc, index, "direction", 0),
        _json_get_int(doc, index, "accel", _json_get_int(doc, index, "time_series", -1)),
        _json_get_float(doc, index, "vel0", 0.0),
    )


fn _parse_rayleigh_coefficients(
    doc: JsonDocument, rayleigh_index: Int, key: StringSlice
) raises -> List[RayleighCoefficient]:
    var coefficients: List[RayleighCoefficient] = []
    var entries = _json_key(doc, rayleigh_index, key)
    if _json_has_value(doc, entries):
        for i in range(_json_expect_array_len(doc, entries, String(key))):
            var entry = doc.array_item(entries, i)
            coefficients.append(
                RayleighCoefficient(
                    _json_get_int(doc, entry, "tag", -1),
                    _json_get_float(doc, entry, "coefficient", 0.0),
                )
            )
    return coefficients^


fn parse_rayleigh_input_from_native(doc: JsonDocument, rayleigh_index: Int) raises -> RayleighInput:
    var rayleigh = RayleighInput()
    if not _json_has_value(doc, rayleigh_index):
        return rayleigh^
    rayleigh.has_rayleigh = True
    rayleigh.alpha_m = _json_get_float(doc, rayleigh_index, "alphaM", 0.0)
    rayleigh.beta_k = _json_get_float(doc, rayleigh_index, "betaK", 0.0)
    rayleigh.beta_k_init = _json_get_float(doc, rayleigh_index, "betaKInit", 0.0)
    rayleigh.beta_k_comm = _json_get_float(doc, rayleigh_index, "betaKComm", 0.0)
    rayleigh.node_alpha_m = _parse_rayleigh_coefficients(doc, rayleigh_index, "node_alpha_m")
    rayleigh.element_beta_k_init = _parse_rayleigh_coefficients(
        doc, rayleigh_index, "element_beta_k_init"
    )
    return rayleigh^


fn parse_element_load_input_from_native(
    doc: JsonDocument, load_index: Int
) raises -> ElementLoadInput:
    var load_type = _json_get_string(doc, load_index, "type", "")
    var load = ElementLoadInput(
        _json_get_int(doc, load_index, "element", 0),
        load_type,
        element_load_type_tag(load_type),
        _json_get_float_alias2(doc, load_index, "wy", "w", 0.0),
        _json_get_float(doc, load_index, "wz", 0.0),
        _json_get_float_alias3(doc, load_index, "wx", "wa", "axial", 0.0),
        _json_get_float_alias3(doc, load_index, "py", "P", "Ptrans", 0.0),
        _json_get_float(doc, load_index, "pz", 0.0),
        _json_get_float_alias3(doc, load_index, "px", "N", "Paxial", 0.0),
        _json_get_float_alias3(doc, load_index, "x", "xL", "aOverL", 0.0),
    )
    if load.type_tag == ElementLoadTypeTag.BeamThermal:
        var temperatures = _json_key(doc, load_index, "temperatures")
        var locations = _json_key(doc, load_index, "locations")
        var count = _json_expect_array_len(doc, temperatures, "beamThermal temperatures")
        if (count != 2 and count != 5 and count != 9) or count != _json_expect_array_len(
            doc, locations, "beamThermal locations"
        ):
            abort("beamThermal requires 2, 5, or 9 temperature/location pairs")
        load.thermal.count = count
        for i in range(count):
            load.thermal.temperatures[i] = _json_number_value(
                doc, doc.array_item(temperatures, i), "beamThermal temperature"
            )
            load.thermal.locations[i] = _json_number_value(
                doc, doc.array_item(locations, i), "beamThermal location"
            )
            if i > 0 and load.thermal.locations[i] <= load.thermal.locations[i - 1]:
                abort("beamThermal locations must increase")
    return load^


fn parse_nodal_load_input_from_native(doc: JsonDocument, load_index: Int) raises -> NodalLoadInput:
    return NodalLoadInput(
        _json_get_int(doc, load_index, "node", 0),
        _json_get_int(doc, load_index, "dof", 0),
        _json_get_float(doc, load_index, "value", 0.0),
    )


fn parse_damping_inputs_from_native(
    doc: JsonDocument,
    dampings_index: Int,
    time_series: List[TimeSeriesInput],
) raises -> List[DampingInput]:
    var parsed: List[DampingInput] = []
    if not _json_has_value(doc, dampings_index):
        return parsed^

    if doc.node_tag(dampings_index) == JsonValueTag.Array:
        for i in range(doc.node_len(dampings_index)):
            var raw = doc.array_item(dampings_index, i)
            var damping = DampingInput()
            damping.tag = _json_get_int(doc, raw, "id", _json_get_int(doc, raw, "tag", -1))
            if damping.tag < 0:
                abort("damping requires id")
            for j in range(len(parsed)):
                if parsed[j].tag == damping.tag:
                    abort("duplicate damping id")
            damping.type = _json_get_string(doc, raw, "type", "")
            if damping.type == "SecStiff":
                damping.type = "SecStif"
            if damping.type != "SecStif":
                abort("unsupported damping type: " + damping.type)
            damping.beta = _json_get_float(doc, raw, "beta", 0.0)
            if damping.beta <= 0.0:
                abort("SecStif damping requires beta > 0")
            damping.activate_time = _json_get_float_alias2(
                doc, raw, "activateTime", "activate_time", 0.0
            )
            damping.deactivate_time = _json_get_float_alias2(
                doc, raw, "deactivateTime", "deactivate_time", 1.0e20
            )
            damping.factor_ts_tag = _json_get_int(
                doc,
                raw,
                "factor",
                _json_get_int(doc, raw, "factor_time_series", -1),
            )
            if damping.factor_ts_tag >= 0:
                damping.factor_ts_index = find_time_series_input(time_series, damping.factor_ts_tag)
                if damping.factor_ts_index < 0:
                    abort("damping factor time_series tag not found")
            parsed.append(damping^)
        return parsed^

    if doc.node_tag(dampings_index) == JsonValueTag.Object:
        var damping = DampingInput()
        damping.tag = _json_get_int(
            doc, dampings_index, "id", _json_get_int(doc, dampings_index, "tag", -1)
        )
        if damping.tag < 0:
            abort("damping requires id")
        damping.type = _json_get_string(doc, dampings_index, "type", "")
        if damping.type == "SecStiff":
            damping.type = "SecStif"
        if damping.type != "SecStif":
            abort("unsupported damping type: " + damping.type)
        damping.beta = _json_get_float(doc, dampings_index, "beta", 0.0)
        if damping.beta <= 0.0:
            abort("SecStif damping requires beta > 0")
        damping.activate_time = _json_get_float_alias2(
            doc, dampings_index, "activateTime", "activate_time", 0.0
        )
        damping.deactivate_time = _json_get_float_alias2(
            doc, dampings_index, "deactivateTime", "deactivate_time", 1.0e20
        )
        damping.factor_ts_tag = _json_get_int(
            doc,
            dampings_index,
            "factor",
            _json_get_int(doc, dampings_index, "factor_time_series", -1),
        )
        if damping.factor_ts_tag >= 0:
            damping.factor_ts_index = find_time_series_input(time_series, damping.factor_ts_tag)
            if damping.factor_ts_index < 0:
                abort("damping factor time_series tag not found")
        parsed.append(damping^)
        return parsed^
    abort("dampings must be list or object")
    return parsed^


fn parse_stage_input_from_native(
    doc: JsonDocument, stage_index: Int, source_info: CaseSourceInfo
) raises -> StageInput:
    var stage = StageInput()
    stage.has_initial_state_reset = _json_has_key(doc, stage_index, "initial_state_reset")
    if _json_has_key(doc, stage_index, "contact_friction"):
        stage.contact_friction = _json_get_bool(doc, stage_index, "contact_friction", True)
    stage.capture_initial_state = _json_get_bool(doc, stage_index, "initial_state_reset", False)
    var removed_nodes = _json_key(doc, stage_index, "remove_nodes")
    if _json_has_value(doc, removed_nodes):
        for i in range(_json_expect_array_len(doc, removed_nodes, "removed nodes")):
            stage.removed_nodes.append(
                _json_int_value(doc, doc.array_item(removed_nodes, i), "removed node")
            )
    var removed_elements = _json_key(doc, stage_index, "remove_elements")
    if _json_has_value(doc, removed_elements):
        for i in range(_json_expect_array_len(doc, removed_elements, "removed elements")):
            stage.removed_elements.append(
                _json_int_value(doc, doc.array_item(removed_elements, i), "removed element")
            )
    var stage_analysis_index = _json_key(doc, stage_index, "analysis")
    if stage_analysis_index < 0:
        stage_analysis_index = stage_index
    stage.analysis = parse_analysis_input_from_native(
        doc,
        stage_analysis_index,
        stage.analysis_integrator_targets_pool,
        stage.analysis_solver_chain_pool,
    )
    stage.resume_load_control = _json_get_bool(doc, stage_index, "resume_load_control", False)
    var samples = _json_key(doc, stage_index, "sample_before")
    if _json_has_value(doc, samples):
        for i in range(_json_expect_array_len(doc, samples, "explicit recorder samples")):
            stage.sample_before.append(
                _json_int_value(doc, doc.array_item(samples, i), "explicit recorder index")
            )
    var updates = _json_key(doc, stage_index, "material_stages")
    if _json_has_value(doc, updates):
        for i in range(_json_expect_array_len(doc, updates, "material stages")):
            var update = doc.array_item(updates, i)
            stage.material_stages.append(
                MaterialStageInput(
                    _json_get_int(doc, update, "material", -1),
                    _json_get_int(doc, update, "stage", 0),
                )
            )
    var permeability = _json_key(doc, stage_index, "permeability_updates")
    if _json_has_value(doc, permeability):
        for i in range(_json_expect_array_len(doc, permeability, "permeability updates")):
            var update = doc.array_item(permeability, i)
            stage.permeability_updates.append(
                PermeabilityUpdate(
                    _json_get_int(doc, update, "element", -1),
                    _json_get_string(doc, update, "axis", ""),
                    _json_get_float(doc, update, "value", -1.0),
                )
            )
    var pressures = _json_key(doc, stage_index, "prescribed_pressures")
    if _json_has_value(doc, pressures):
        for i in range(_json_expect_array_len(doc, pressures, "prescribed pressures")):
            var pressure = doc.array_item(pressures, i)
            var load = parse_nodal_load_input_from_native(doc, pressure)
            stage.prescribed_pressures.append(
                PrescribedPressureInput(
                    load.node,
                    load.dof,
                    load.value,
                    _json_get_float(doc, pressure, "displacement", 0.0),
                    _json_get_float(doc, pressure, "acceleration", 0.0),
                )
            )
    stage.pattern = parse_pattern_input_from_native(doc, _json_key(doc, stage_index, "pattern"))
    stage.rayleigh = parse_rayleigh_input_from_native(doc, _json_key(doc, stage_index, "rayleigh"))

    var load_const_index = _json_key(doc, stage_index, "load_const")
    if _json_has_value(doc, load_const_index):
        if doc.node_tag(load_const_index) == JsonValueTag.Bool:
            stage.has_load_const = doc.node_bool(load_const_index)
        else:
            stage.has_load_const = True
            stage.load_const_time = _json_get_float(doc, load_const_index, "time", 0.0)

    var constant_loads = _json_key(doc, stage_index, "constant_loads")
    for i in range(_json_expect_array_len(doc, constant_loads, "constant nodal loads")):
        stage.constant_loads.append(
            parse_nodal_load_input_from_native(doc, doc.array_item(constant_loads, i))
        )
    var constant_element_loads = _json_key(doc, stage_index, "constant_element_loads")
    for i in range(_json_expect_array_len(doc, constant_element_loads, "constant element loads")):
        stage.constant_element_loads.append(
            parse_element_load_input_from_native(doc, doc.array_item(constant_element_loads, i))
        )

    var loads_index = _json_key(doc, stage_index, "loads")
    if _json_has_value(doc, loads_index):
        for i in range(_json_expect_array_len(doc, loads_index, "stage loads")):
            stage.loads.append(
                parse_nodal_load_input_from_native(doc, doc.array_item(loads_index, i))
            )

    var element_loads_index = _json_key(doc, stage_index, "element_loads")
    if _json_has_value(doc, element_loads_index):
        for i in range(_json_expect_array_len(doc, element_loads_index, "stage element_loads")):
            stage.element_loads.append(
                parse_element_load_input_from_native(doc, doc.array_item(element_loads_index, i))
            )

    _append_time_series_inputs_native(
        doc,
        stage_index,
        source_info,
        stage.time_series,
        stage.time_series_values,
        stage.time_series_times,
    )
    return stage^


fn parse_case_input_native(doc: JsonDocument) raises -> CaseInput:
    return parse_case_input_native_from_source(doc, CaseSourceInfo(), True)


fn parse_case_input_native_from_source(
    doc: JsonDocument, source_info: CaseSourceInfo
) raises -> CaseInput:
    return parse_case_input_native_from_source(doc, source_info, True)


fn parse_case_input_native_from_source(
    doc: JsonDocument, source_info: CaseSourceInfo, include_recorders: Bool
) raises -> CaseInput:
    var case_input = CaseInput()
    var root = doc.root_index

    var model = _json_require_key(doc, root, "model")
    case_input.model = ModelInput(
        _json_get_int(doc, model, "ndm", 0),
        _json_get_int(doc, model, "ndf", 0),
    )

    var nodes_raw = _json_require_key(doc, root, "nodes")
    for i in range(_json_expect_array_len(doc, nodes_raw, "nodes")):
        var node = doc.array_item(nodes_raw, i)
        var has_z = _json_has_key(doc, node, "z")
        var z = 0.0
        if has_z:
            z = _json_get_float(doc, node, "z", 0.0)
        var parsed = NodeInput(
            _json_get_int(doc, node, "id", 0),
            _json_get_float(doc, node, "x", 0.0),
            _json_get_float(doc, node, "y", 0.0),
            z,
            has_z,
        )
        var constraints = _json_key(doc, node, "constraints")
        if _json_has_value(doc, constraints):
            var count = _json_expect_array_len(doc, constraints, "node constraints")
            if count > 6:
                count = 6
            parsed.constraint_count = count
            if count > 0:
                parsed.constraint_1 = _json_int_value(
                    doc, doc.array_item(constraints, 0), "node constraints"
                )
            if count > 1:
                parsed.constraint_2 = _json_int_value(
                    doc, doc.array_item(constraints, 1), "node constraints"
                )
            if count > 2:
                parsed.constraint_3 = _json_int_value(
                    doc, doc.array_item(constraints, 2), "node constraints"
                )
            if count > 3:
                parsed.constraint_4 = _json_int_value(
                    doc, doc.array_item(constraints, 3), "node constraints"
                )
            if count > 4:
                parsed.constraint_5 = _json_int_value(
                    doc, doc.array_item(constraints, 4), "node constraints"
                )
            if count > 5:
                parsed.constraint_6 = _json_int_value(
                    doc, doc.array_item(constraints, 5), "node constraints"
                )
        parsed.ndf = _json_get_int(doc, node, "ndf", case_input.model.ndf)
        case_input.nodes.append(parsed)

    var sections_raw = _json_key(doc, root, "sections")
    if _json_has_value(doc, sections_raw):
        for i in range(_json_expect_array_len(doc, sections_raw, "sections")):
            var sec = doc.array_item(sections_raw, i)
            var parsed = SectionInput(
                _json_get_int(doc, sec, "id", -1),
                _json_get_string(doc, sec, "type", ""),
            )
            var params = _json_key(doc, sec, "params")
            if parsed.type == "ElasticMembranePlateSection":
                parsed.plate_mod = _json_get_float(doc, params, "Ep_mod", 1.0)
            if parsed.type == "PlateFiber":
                parsed.plate_fiber_material = _json_get_int(doc, params, "material", -1)
            if parsed.type == "HingeRadauSection2d":
                parsed.hinge_radau.section_i = _json_get_int(doc, params, "section_i", -1)
                parsed.hinge_radau.section_j = _json_get_int(doc, params, "section_j", -1)
                parsed.hinge_radau.length_i = _json_get_float(doc, params, "length_i", 0.0)
                parsed.hinge_radau.length_j = _json_get_float(doc, params, "length_j", 0.0)
                parsed.hinge_radau.max_iters = _json_get_int(doc, params, "max_iters", 10)
                parsed.hinge_radau.tolerance = _json_get_float(doc, params, "tol", 1.0e-10)
            if _json_has_key(doc, params, "E"):
                parsed.E = _json_get_float(doc, params, "E", 0.0)
            if _json_has_key(doc, params, "A"):
                parsed.A = _json_get_float(doc, params, "A", 0.0)
            if _json_has_key(doc, params, "I"):
                parsed.I = _json_get_float(doc, params, "I", 0.0)
            if _json_has_key(doc, params, "Iz"):
                parsed.Iz = _json_get_float(doc, params, "Iz", 0.0)
            if _json_has_key(doc, params, "Iy"):
                parsed.Iy = _json_get_float(doc, params, "Iy", 0.0)
            if _json_has_key(doc, params, "G"):
                parsed.G = _json_get_float(doc, params, "G", 0.0)
            if _json_has_key(doc, params, "J"):
                parsed.J = _json_get_float(doc, params, "J", 0.0)
            parsed.shear_rigidity_y = _json_get_float(doc, params, "shear_rigidity_y", 0.0)
            parsed.shear_rigidity_z = _json_get_float(doc, params, "shear_rigidity_z", 0.0)
            parsed.base_torsional_rigidity = _json_get_float(
                doc, params, "base_torsional_rigidity", 0.0
            )
            parsed.added_torsional_rigidity = _json_get_float(
                doc, params, "added_torsional_rigidity", 0.0
            )
            if _json_has_key(doc, params, "nu"):
                parsed.nu = _json_get_float(doc, params, "nu", 0.0)
            if _json_has_key(doc, params, "h"):
                parsed.h = _json_get_float(doc, params, "h", 0.0)
            if _json_has_key(doc, params, "rho"):
                parsed.rho = _json_get_float(doc, params, "rho", 0.0)
            if _json_has_key(doc, params, "axial_material"):
                parsed.axial_material = _json_get_int(doc, params, "axial_material", -1)
            if _json_has_key(doc, params, "flexural_material"):
                parsed.flexural_material = _json_get_int(doc, params, "flexural_material", -1)
            if _json_has_key(doc, params, "moment_y_material"):
                parsed.moment_y_material = _json_get_int(doc, params, "moment_y_material", -1)
            if _json_has_key(doc, params, "torsion_material"):
                parsed.torsion_material = _json_get_int(doc, params, "torsion_material", -1)
            if _json_has_key(doc, params, "shear_y_material"):
                parsed.shear_y_material = _json_get_int(doc, params, "shear_y_material", -1)
            if _json_has_key(doc, params, "shear_z_material"):
                parsed.shear_z_material = _json_get_int(doc, params, "shear_z_material", -1)
            if _json_has_key(doc, params, "base_section"):
                parsed.base_section = _json_get_int(doc, params, "base_section", -1)
            if parsed.type == "FiberSection2d" or parsed.type == "FiberSection3d":
                var patches_raw = _json_key(doc, params, "patches")
                parsed.fiber_patch_offset = len(case_input.fiber_patches)
                parsed.fiber_patch_count = _json_expect_array_len(
                    doc, patches_raw, "section patches"
                )
                for j in range(parsed.fiber_patch_count):
                    var patch = doc.array_item(patches_raw, j)
                    var patch_input = FiberPatchInput()
                    patch_input.type = _json_get_string(doc, patch, "type", "")
                    if patch_input.type == "quad":
                        patch_input.type = "quadr"
                    patch_input.material = _json_get_int(doc, patch, "material", -1)
                    patch_input.num_subdiv_y = _json_get_int(doc, patch, "num_subdiv_y", 0)
                    patch_input.num_subdiv_z = _json_get_int(doc, patch, "num_subdiv_z", 0)
                    if _json_has_key(doc, patch, "y_i"):
                        patch_input.y_i = _json_get_float(doc, patch, "y_i", 0.0)
                    if _json_has_key(doc, patch, "z_i"):
                        patch_input.z_i = _json_get_float(doc, patch, "z_i", 0.0)
                    if _json_has_key(doc, patch, "y_j"):
                        patch_input.y_j = _json_get_float(doc, patch, "y_j", 0.0)
                    if _json_has_key(doc, patch, "z_j"):
                        patch_input.z_j = _json_get_float(doc, patch, "z_j", 0.0)
                    if _json_has_key(doc, patch, "y_k"):
                        patch_input.y_k = _json_get_float(doc, patch, "y_k", 0.0)
                    if _json_has_key(doc, patch, "z_k"):
                        patch_input.z_k = _json_get_float(doc, patch, "z_k", 0.0)
                    if _json_has_key(doc, patch, "y_l"):
                        patch_input.y_l = _json_get_float(doc, patch, "y_l", 0.0)
                    if _json_has_key(doc, patch, "z_l"):
                        patch_input.z_l = _json_get_float(doc, patch, "z_l", 0.0)
                    if _json_has_key(doc, patch, "num_subdiv_circ"):
                        patch_input.num_subdiv_circ = _json_get_int(
                            doc, patch, "num_subdiv_circ", 0
                        )
                    if _json_has_key(doc, patch, "num_subdiv_rad"):
                        patch_input.num_subdiv_rad = _json_get_int(doc, patch, "num_subdiv_rad", 0)
                    if _json_has_key(doc, patch, "y_center"):
                        patch_input.y_center = _json_get_float(doc, patch, "y_center", 0.0)
                    if _json_has_key(doc, patch, "z_center"):
                        patch_input.z_center = _json_get_float(doc, patch, "z_center", 0.0)
                    if _json_has_key(doc, patch, "int_rad"):
                        patch_input.int_rad = _json_get_float(doc, patch, "int_rad", 0.0)
                    if _json_has_key(doc, patch, "ext_rad"):
                        patch_input.ext_rad = _json_get_float(doc, patch, "ext_rad", 0.0)
                    if _json_has_key(doc, patch, "start_ang"):
                        patch_input.start_ang = _json_get_float(doc, patch, "start_ang", 0.0)
                    if _json_has_key(doc, patch, "end_ang"):
                        patch_input.end_ang = _json_get_float(doc, patch, "end_ang", 360.0)
                    case_input.fiber_patches.append(patch_input)

                var layers_raw = _json_key(doc, params, "layers")
                parsed.fiber_layer_offset = len(case_input.fiber_layers)
                parsed.fiber_layer_count = _json_expect_array_len(doc, layers_raw, "section layers")
                for j in range(parsed.fiber_layer_count):
                    var layer = doc.array_item(layers_raw, j)
                    var layer_input = FiberLayerInput()
                    layer_input.type = _json_get_string(doc, layer, "type", "")
                    layer_input.material = _json_get_int(doc, layer, "material", -1)
                    layer_input.num_bars = _json_get_int(doc, layer, "num_bars", 0)
                    layer_input.bar_area = _json_get_float(doc, layer, "bar_area", 0.0)
                    layer_input.y_start = _json_get_float(doc, layer, "y_start", 0.0)
                    layer_input.z_start = _json_get_float(doc, layer, "z_start", 0.0)
                    layer_input.y_end = _json_get_float(doc, layer, "y_end", 0.0)
                    layer_input.z_end = _json_get_float(doc, layer, "z_end", 0.0)
                    if _json_has_key(doc, layer, "y_center"):
                        layer_input.y_center = _json_get_float(doc, layer, "y_center", 0.0)
                    if _json_has_key(doc, layer, "z_center"):
                        layer_input.z_center = _json_get_float(doc, layer, "z_center", 0.0)
                    if _json_has_key(doc, layer, "radius"):
                        layer_input.radius = _json_get_float(doc, layer, "radius", 0.0)
                    if _json_has_key(doc, layer, "start_ang") and _json_has_key(
                        doc, layer, "end_ang"
                    ):
                        layer_input.start_ang = _json_get_float(doc, layer, "start_ang", 0.0)
                        layer_input.end_ang = _json_get_float(doc, layer, "end_ang", 360.0)
                        layer_input.has_angles = True
                    case_input.fiber_layers.append(layer_input)
                var fibers_raw = _json_key(doc, params, "fibers")
                if _json_has_value(doc, fibers_raw):
                    parsed.fiber_offset = len(case_input.fibers)
                    parsed.fiber_count = _json_expect_array_len(doc, fibers_raw, "section fibers")
                    for j in range(parsed.fiber_count):
                        var fiber = doc.array_item(fibers_raw, j)
                        var fiber_input = FiberInput()
                        fiber_input.y = _json_get_float(doc, fiber, "y", 0.0)
                        fiber_input.z = _json_get_float(doc, fiber, "z", 0.0)
                        fiber_input.area = _json_get_float(doc, fiber, "area", 0.0)
                        fiber_input.material = _json_get_int(doc, fiber, "material", -1)
                        case_input.fibers.append(fiber_input)
            elif parsed.type == "LayeredShellSection":
                var layers_raw = _json_key(doc, params, "layers")
                parsed.shell_layer_offset = len(case_input.shell_layers)
                parsed.shell_layer_count = _json_expect_array_len(doc, layers_raw, "shell layers")
                for j in range(parsed.shell_layer_count):
                    var layer = doc.array_item(layers_raw, j)
                    var layer_input = ShellLayerInput()
                    layer_input.material = _json_get_int(doc, layer, "material", -1)
                    layer_input.thickness = _json_get_float(doc, layer, "thickness", 0.0)
                    case_input.shell_layers.append(layer_input)
            case_input.sections.append(parsed)

    var materials_raw = _json_key(doc, root, "materials")
    if _json_has_value(doc, materials_raw):
        for i in range(_json_expect_array_len(doc, materials_raw, "materials")):
            var mat = doc.array_item(materials_raw, i)
            var params = _json_key(doc, mat, "params")
            var parsed = MaterialInput()
            parsed.id = _json_get_int(doc, mat, "id", -1)
            parsed.type = _json_get_string(doc, mat, "type", "")
            if parsed.type == "Viscous":
                parsed.viscous_coefficient = _json_number_value(
                    doc, _json_require_key(doc, params, "C"), "Viscous C"
                )
                var exponent = _json_get_float(doc, params, "alpha", 1.0)
                parsed.viscous_exponent = exponent if exponent >= 0.0 else 1.0
                case_input.materials.append(parsed^)
                continue
            if parsed.type == "ContactMaterial2D":
                for i in range(4):
                    parsed.contact_parameters[i] = _required_material_param(
                        doc, params, ["mu", "G", "c", "t"][i]
                    )
                if parsed.contact_parameters[1] <= 0.0:
                    abort("ContactMaterial2D requires positive tangential stiffness")
            if parsed.type == "InitialStateAnalysisWrapper":
                parsed.base_material = _json_get_int(doc, params, "material", -1)
                parsed.soil_dimensions = _json_get_int(doc, params, "nd", 2)
                case_input.materials.append(parsed^)
                continue
            if (
                parsed.type == "PressureIndependMultiYield"
                or parsed.type == "PressureDependMultiYield"
                or parsed.type == "PressureDependMultiYield02"
            ):
                var variant_02 = parsed.type == "PressureDependMultiYield02"
                parsed.soil_dimensions = _json_get_int(doc, params, "nd", 2)
                var curve_data = _json_key(doc, params, "reduction_curve")
                var curve: List[Float64] = []
                if _json_has_value(doc, curve_data):
                    for j in range(_json_expect_array_len(doc, curve_data, "soil reduction curve")):
                        curve.append(
                            _json_number_value(
                                doc, doc.array_item(curve_data, j), "soil reduction point"
                            )
                        )
                parsed.rho = _json_get_float(doc, params, "rho", 0.0)
                if parsed.rho < 0.0:
                    abort("soil material requires nonnegative density")
                var shear = _json_get_float(doc, params, "G", 0.0)
                var bulk = _json_get_float(doc, params, "K", 0.0)
                var peak_strain = _json_get_float(doc, params, "peak_shear_strain", 0.0)
                var friction = _json_get_float(doc, params, "friction_angle", 0.0)
                var ref_pressure = _json_get_float(doc, params, "reference_pressure", 100.0)
                var exponent = _json_get_float(doc, params, "pressure_exponent", 0.0)
                var surfaces = _json_get_int(doc, params, "yield_surfaces", 20)
                if parsed.type == "PressureIndependMultiYield":
                    parsed.pimy_index = len(case_input.pimy_definitions)
                    case_input.pimy_definitions.append(
                        PressureIndependMultiYieldDef(
                            parsed.rho,
                            shear,
                            bulk,
                            _json_get_float(doc, params, "cohesion", 0.0),
                            peak_strain,
                            friction,
                            ref_pressure,
                            exponent,
                            surfaces,
                            curve,
                        )
                    )
                else:
                    parsed.pdmy_index = len(case_input.pdmy_definitions)
                    case_input.pdmy_definitions.append(
                        PressureDependMultiYieldDef(
                            shear,
                            bulk,
                            friction,
                            peak_strain,
                            ref_pressure,
                            exponent,
                            _json_get_float(doc, params, "phase_angle", 0.0),
                            _json_get_float(doc, params, "contraction", 0.0),
                            _json_get_float(doc, params, "dilate1", 0.0),
                            _json_get_float(doc, params, "dilate2", 3.0 if variant_02 else 0.0),
                            _json_get_float(doc, params, "liquefy_pressure", 0.0),
                            _json_get_float(doc, params, "liquefy_size", 0.0),
                            _json_get_float(doc, params, "liquefy_translation", 0.0),
                            surfaces,
                            curve,
                            _json_get_float(doc, params, "void_ratio", 0.6),
                            _json_get_float(doc, params, "vol_limit1", 0.9),
                            _json_get_float(doc, params, "vol_limit2", 0.02),
                            _json_get_float(doc, params, "vol_limit3", 0.7),
                            _json_get_float(doc, params, "atmospheric_pressure", 101.0),
                            _json_get_float(doc, params, "cohesion", 0.1 if variant_02 else 0.3),
                            _json_get_float(doc, params, "hv", 0.0),
                            _json_get_float(doc, params, "pv", 1.0),
                            PressureDependMultiYield02Parameters(
                                variant_02,
                                _json_get_float(doc, params, "contraction2", 5.0),
                                _json_get_float(doc, params, "contraction3", 0.0),
                                _json_get_float(doc, params, "dilate3", 0.0),
                                _json_get_float(doc, params, "liquefy1", 1.0),
                                _json_get_float(doc, params, "liquefy2", 0.0),
                            ),
                        )
                    )
                case_input.materials.append(parsed^)
                continue
            if (
                parsed.type == "Bilin"
                or parsed.type == "Hysteretic"
                or parsed.type == "ModIMKPeakOriented"
            ):
                var values = _json_require_key(doc, params, "values")
                var count = _json_expect_array_len(doc, values, parsed.type + " values")
                if parsed.type == "Bilin" or parsed.type == "ModIMKPeakOriented":
                    if count != 23 and count != 24:
                        raise Error(parsed.type + " expects 23 or 24 parameters")
                elif count != 12 and count != 13 and count != 16 and count != 17:
                    raise Error("Hysteretic expects 12, 13, 16 or 17 parameters")
                for p in range(count):
                    var value = _json_number_value(doc, doc.array_item(values, p), parsed.type)
                    if parsed.type == "Hysteretic" and count <= 13:
                        # Insert the collinear midpoint used by the two-point constructor.
                        var index = p if p < 2 else p + 2 if p < 6 else p + 4
                        parsed.uniaxial_values[index] = value
                    else:
                        parsed.uniaxial_values[p] = value
                if parsed.type == "Hysteretic" and count <= 13:
                    for p in range(2):
                        parsed.uniaxial_values[2 + p] = 0.5 * (
                            parsed.uniaxial_values[p] + parsed.uniaxial_values[4 + p]
                        )
                        parsed.uniaxial_values[8 + p] = 0.5 * (
                            parsed.uniaxial_values[6 + p] + parsed.uniaxial_values[10 + p]
                        )
                case_input.materials.append(parsed^)
                continue
            if (
                parsed.type == "ElasticPP"
                or parsed.type == "ElasticBilin"
                or parsed.type == "Ratchet"
            ):
                parsed.direct_uniaxial_def = read_small_material_def(doc, params, parsed.type)
            if parsed.type == "FSAM":
                parsed.fsam = FSAMInput(
                    _required_material_param(doc, params, "rho"),
                    _json_get_int(doc, params, "steel_x", -1),
                    _json_get_int(doc, params, "steel_y", -1),
                    _json_get_int(doc, params, "concrete", -1),
                    _required_material_param(doc, params, "rho_x"),
                    _required_material_param(doc, params, "rho_y"),
                    _required_material_param(doc, params, "friction"),
                    _required_material_param(doc, params, "dowel"),
                )
                case_input.materials.append(parsed^)
                continue
            if parsed.type == "ConcreteCM" or parsed.type == "SteelMPF":
                parsed.direct_uniaxial_def = read_wall_material_def(doc, params, parsed.type)
                case_input.materials.append(parsed^)
                continue
            if parsed.type == "ConcreteS":
                parsed.E = _json_number_value(
                    doc, _json_require_key(doc, params, "E"), "ConcreteS E"
                )
                parsed.nu = _json_number_value(
                    doc, _json_require_key(doc, params, "nu"), "ConcreteS nu"
                )
                parsed.fc = _json_number_value(
                    doc, _json_require_key(doc, params, "fc"), "ConcreteS fc"
                )
                parsed.ft = _json_number_value(
                    doc, _json_require_key(doc, params, "ft"), "ConcreteS ft"
                )
                parsed.Es = _json_number_value(
                    doc, _json_require_key(doc, params, "Es"), "ConcreteS Es"
                )
                case_input.materials.append(parsed^)
                continue
            if parsed.type == "HystereticSmooth" or parsed.type == "HystereticAsym":
                parsed.ka = _json_get_float(doc, params, "ka", 0.0)
                parsed.kb = _json_get_float(doc, params, "kb", 0.0)
                parsed.fbar = _json_get_float(doc, params, "fbar", 0.0)
                if parsed.type == "HystereticSmooth":
                    parsed.beta1 = _json_get_float(doc, params, "beta", 0.0)
                else:
                    parsed.beta1 = _json_get_float(doc, params, "beta1", 0.0)
                    parsed.beta2 = _json_get_float(doc, params, "beta2", 0.0)
                    parsed.gamma = _json_get_float(doc, params, "gamma", 0.0)
                case_input.materials.append(parsed^)
                continue
            if _json_has_key(doc, params, "E"):
                parsed.E = _json_get_float(doc, params, "E", 0.0)
            if parsed.type == "DruckerPrager":
                parsed.drucker_prager = DruckerPragerDefinition(
                    _json_get_float(doc, params, "bulk", 0.0),
                    _json_get_float(doc, params, "shear", 0.0),
                    _json_get_float(doc, params, "yield_stress", 0.0),
                    _json_get_float(doc, params, "friction", 0.0),
                    _json_get_float(doc, params, "dilation", 0.0),
                    _json_get_float(doc, params, "saturation", 0.0),
                    _json_get_float(doc, params, "initial_hardening", 0.0),
                    _json_get_float(doc, params, "saturation_rate", 0.0),
                    _json_get_float(doc, params, "tension_rate", 0.0),
                    _json_get_float(doc, params, "hardening", 0.0),
                    _json_get_float(doc, params, "isotropic_fraction", 1.0),
                )
            if parsed.type == "ElasticCrossAnisotropic":
                parsed.cross_anisotropic = InlineArray[Float64, 5](
                    _json_get_float(doc, params, "Eh", 0.0),
                    _json_get_float(doc, params, "Ev", 0.0),
                    _json_get_float(doc, params, "nuhv", 0.0),
                    _json_get_float(doc, params, "nuhh", 0.0),
                    _json_get_float(doc, params, "Ghv", 0.0),
                )
            if parsed.type == "PySimple1":
                parsed.soil_type = _json_get_int(doc, params, "soilType", 0)
                parsed.soil_ultimate = _json_get_float(doc, params, "pult", 0.0)
                parsed.soil_disp50 = _json_get_float(doc, params, "y50", 0.0)
                parsed.soil_drag_or_suction = _json_get_float(doc, params, "Cd", 0.0)
                parsed.soil_dashpot = _json_get_float(doc, params, "c", 0.0)
            elif parsed.type == "TzSimple1":
                parsed.soil_type = _json_get_int(doc, params, "tzType", 0)
                parsed.soil_ultimate = _json_get_float(doc, params, "tult", 0.0)
                parsed.soil_disp50 = _json_get_float(doc, params, "z50", 0.0)
                parsed.soil_dashpot = _json_get_float(doc, params, "c", 0.0)
            elif parsed.type == "QzSimple1":
                parsed.soil_type = _json_get_int(doc, params, "qzType", 0)
                parsed.soil_ultimate = _json_get_float(doc, params, "qult", 0.0)
                parsed.soil_disp50 = _json_get_float(doc, params, "z50", 0.0)
                parsed.soil_drag_or_suction = _json_get_float(doc, params, "suction", 0.0)
                parsed.soil_dashpot = _json_get_float(doc, params, "c", 0.0)
            if _json_has_key(doc, params, "Fy"):
                parsed.Fy = _json_get_float(doc, params, "Fy", 0.0)
            if _json_has_key(doc, params, "E0"):
                parsed.E0 = _json_get_float(doc, params, "E0", 0.0)
            if _json_has_key(doc, params, "b"):
                parsed.b = _json_get_float(doc, params, "b", 0.0)
            if _json_has_key(doc, params, "fpc"):
                parsed.fpc = _json_get_float(doc, params, "fpc", 0.0)
            if _json_has_key(doc, params, "epsc0"):
                parsed.epsc0 = _json_get_float(doc, params, "epsc0", 0.0)
            if _json_has_key(doc, params, "fpcu"):
                parsed.fpcu = _json_get_float(doc, params, "fpcu", 0.0)
            if _json_has_key(doc, params, "epscu"):
                parsed.epscu = _json_get_float(doc, params, "epscu", 0.0)
            parsed.has_r0 = _json_has_key(doc, params, "R0")
            parsed.has_cr1 = _json_has_key(doc, params, "cR1")
            parsed.has_cr2 = _json_has_key(doc, params, "cR2")
            if parsed.has_r0:
                parsed.R0 = _json_get_float(doc, params, "R0", 0.0)
            if parsed.has_cr1:
                parsed.cR1 = _json_get_float(doc, params, "cR1", 0.0)
            if parsed.has_cr2:
                parsed.cR2 = _json_get_float(doc, params, "cR2", 0.0)
            parsed.has_a1 = _json_has_key(doc, params, "a1")
            parsed.has_a2 = _json_has_key(doc, params, "a2")
            parsed.has_a3 = _json_has_key(doc, params, "a3")
            parsed.has_a4 = _json_has_key(doc, params, "a4")
            if parsed.has_a1:
                parsed.a1 = _json_get_float(doc, params, "a1", 0.0)
            if parsed.has_a2:
                parsed.a2 = _json_get_float(doc, params, "a2", 0.0)
            if parsed.has_a3:
                parsed.a3 = _json_get_float(doc, params, "a3", 0.0)
            if parsed.has_a4:
                parsed.a4 = _json_get_float(doc, params, "a4", 0.0)
            parsed.has_siginit = _json_has_key(doc, params, "sigInit")
            if parsed.has_siginit:
                parsed.sigInit = _json_get_float(doc, params, "sigInit", 0.0)
            parsed.has_rat = _json_has_key(doc, params, "rat")
            parsed.has_ft = _json_has_key(doc, params, "ft")
            parsed.has_ets = _json_has_key(doc, params, "Ets")
            if parsed.has_rat:
                parsed.rat = _json_get_float(doc, params, "rat", 0.0)
            if parsed.has_ft:
                parsed.ft = _json_get_float(doc, params, "ft", 0.0)
            if parsed.has_ets:
                parsed.Ets = _json_get_float(doc, params, "Ets", 0.0)
            if _json_has_key(doc, params, "nu"):
                parsed.nu = _json_get_float(doc, params, "nu", 0.0)
            if _json_has_key(doc, params, "rho"):
                parsed.rho = _json_get_float(doc, params, "rho", 0.0)
            if _json_has_key(doc, params, "material"):
                parsed.base_material = _json_get_int(doc, params, "material", -1)
            if _json_has_key(doc, params, "angle"):
                parsed.angle = _json_get_float(doc, params, "angle", 0.0)
            if _json_has_key(doc, params, "gmod"):
                parsed.gmod = _json_get_float(doc, params, "gmod", 0.0)
            if _json_has_key(doc, params, "K"):
                parsed.K = _json_get_float(doc, params, "K", 0.0)
            if _json_has_key(doc, params, "C"):
                parsed.C = _json_get_float(doc, params, "C", 0.0)
            if _json_has_key(doc, params, "alpha"):
                parsed.alpha = _json_get_float(doc, params, "alpha", 1.0)
            elif _json_has_key(doc, params, "Alpha"):
                parsed.alpha = _json_get_float(doc, params, "Alpha", 1.0)
            if _json_has_key(doc, params, "LGap"):
                parsed.LGap = _json_get_float(doc, params, "LGap", 0.0)
            if _json_has_key(doc, params, "NM"):
                parsed.NM = _json_get_float(doc, params, "NM", 1.0)
            if _json_has_key(doc, params, "RelTol"):
                parsed.RelTol = _json_get_float(doc, params, "RelTol", 1.0e-6)
            if _json_has_key(doc, params, "AbsTol"):
                parsed.AbsTol = _json_get_float(doc, params, "AbsTol", 1.0e-10)
            if _json_has_key(doc, params, "MaxHalf"):
                parsed.MaxHalf = _json_get_float(doc, params, "MaxHalf", 15.0)
            if parsed.type == "Fatigue":
                parsed.E0 = _json_get_float(doc, params, "E0", 0.191)
                parsed.fatigue_dmax = _json_get_float(doc, params, "Dmax", 1.0)
                parsed.fatigue_slope = _json_get_float(doc, params, "m", -0.458)
                parsed.strain_min = _json_get_float(doc, params, "min", -1.0e16)
                parsed.strain_max = _json_get_float(doc, params, "max", 1.0e16)
            if parsed.type == "MinMax":
                parsed.strain_min = _json_get_float(doc, params, "min", -1.0e16)
                parsed.strain_max = _json_get_float(doc, params, "max", 1.0e16)
            if parsed.type == "ENT":
                parsed.ent_a = _json_get_float(doc, params, "a", 0.0)
                parsed.ent_b = _json_get_float(doc, params, "b", 1.0)
            elif parsed.type == "Parallel":
                var children = _json_key(doc, params, "materials")
                var child_count = _json_expect_array_len(doc, children, "Parallel materials")
                if child_count == 0:
                    abort("Parallel requires at least one child material")
                var factors = _json_key(doc, params, "factors")
                var factor_count = _json_expect_array_len(doc, factors, "Parallel factors")
                if factors >= 0 and factor_count != child_count:
                    abort("Parallel requires one factor per material")
                var components = List[ParallelComponentInput](capacity=child_count)
                for j in range(child_count):
                    var child_id = _json_int_value(
                        doc, doc.array_item(children, j), "Parallel child material"
                    )
                    var factor = 1.0
                    if factor_count > 0:
                        factor = _json_number_value(
                            doc, doc.array_item(factors, j), "Parallel factor"
                        )
                    components.append(ParallelComponentInput(child_id, factor))
                parsed.parallel_components = components^
            case_input.materials.append(parsed^)

    var elements_raw = _json_require_key(doc, root, "elements")
    for i in range(_json_expect_array_len(doc, elements_raw, "elements")):
        var elem = doc.array_item(elements_raw, i)
        var parsed = ElementInput()
        parsed.id = _json_get_int(doc, elem, "id", 0)
        parsed.type = _json_get_string(doc, elem, "type", "")
        var nodes = _json_require_key(doc, elem, "nodes")
        if parsed.type == "BeamContact2D":
            if _json_expect_array_len(doc, nodes, "BeamContact2D nodes") != 4:
                abort("BeamContact2D requires four nodes")
            parsed.contact_gap_tol = _required_material_param(doc, elem, "gap_tol")
            parsed.contact_force_tol = _required_material_param(doc, elem, "force_tol")
            parsed.contact_initial = _json_get_bool(doc, elem, "initial_contact", True)
        if parsed.type == "MVLEM":
            if _json_expect_array_len(doc, nodes, "MVLEM nodes") != 2:
                abort("MVLEM requires two nodes")
            parsed.wall.c = _required_material_param(doc, elem, "c")
            parsed.material = _json_get_int(doc, elem, "shear_material", -1)
            var fibers = _json_require_key(doc, elem, "fibers")
            for f in range(_json_expect_array_len(doc, fibers, "MVLEM fibers")):
                var fiber = doc.array_item(fibers, f)
                var width = _required_material_param(doc, fiber, "width")
                var thickness = _required_material_param(doc, fiber, "thickness")
                var rho = _required_material_param(doc, fiber, "rho")
                if width <= 0.0 or thickness <= 0.0 or rho < 0.0 or rho > 1.0:
                    abort("MVLEM fiber requires positive width and thickness and rho in [0, 1]")
                parsed.wall.fibers.append(
                    MVLEMFiber(
                        width,
                        thickness,
                        rho,
                        _json_get_int(doc, fiber, "concrete", -1),
                        _json_get_int(doc, fiber, "steel", -1),
                    )
                )
        if parsed.type == "CatenaryCable":
            parsed.cable = CatenaryCable(
                _required_material_param(doc, elem, "weight"),
                _required_material_param(doc, elem, "E"),
                _required_material_param(doc, elem, "A"),
                _required_material_param(doc, elem, "L0"),
                _required_material_param(doc, elem, "alpha"),
                _required_material_param(doc, elem, "temperature_change"),
                _required_material_param(doc, elem, "rho"),
                _required_material_param(doc, elem, "error_tol"),
                _json_get_int(doc, elem, "substeps", 0),
                _json_get_int(doc, elem, "mass_type", 0),
            )
        if parsed.type == "SFI_MVLEM":
            if _json_expect_array_len(doc, nodes, "SFI_MVLEM nodes") != 2:
                abort("SFI_MVLEM requires two nodes")
            parsed.wall.c = _required_material_param(doc, elem, "c")
            var panels = _json_require_key(doc, elem, "panels")
            for i in range(_json_expect_array_len(doc, panels, "SFI_MVLEM panels")):
                var panel = doc.array_item(panels, i)
                var width = _required_material_param(doc, panel, "width")
                var thickness = _required_material_param(doc, panel, "thickness")
                if width <= 0.0 or thickness <= 0.0:
                    abort("SFI_MVLEM requires positive panel width and thickness")
                parsed.wall.fibers.append(
                    MVLEMFiber(width, thickness, 0.0, _json_get_int(doc, panel, "material", -1), -1)
                )
        var node_count = _json_expect_array_len(doc, nodes, "element nodes")
        var solid_nodes = continuum_nodes(parsed.type)
        if parsed.type == "9_4_QuadUP" or parsed.type == "quadUP" or parsed.type == "SSPbrickUP":
            var expected = 4
            if parsed.type == "9_4_QuadUP":
                expected = 9
            elif parsed.type == "SSPbrickUP":
                expected = 8
            if node_count != expected:
                abort(parsed.type + " has an invalid node count")
        elif solid_nodes > 0:
            if node_count != solid_nodes:
                abort(parsed.type + " has an invalid node count")
            if solid_nodes > 4:
                var extra_nodes: List[Int] = []
                for n in range(4, node_count):
                    extra_nodes.append(
                        _json_int_value(doc, doc.array_item(nodes, n), "element nodes")
                    )
                parsed.solid_extra_nodes = extra_nodes^
        elif node_count > 4:
            node_count = 4
        parsed.node_count = node_count
        if node_count > 0:
            parsed.node_1 = _json_int_value(doc, doc.array_item(nodes, 0), "element nodes")
        if node_count > 1:
            parsed.node_2 = _json_int_value(doc, doc.array_item(nodes, 1), "element nodes")
        if node_count > 2:
            parsed.node_3 = _json_int_value(doc, doc.array_item(nodes, 2), "element nodes")
        if node_count > 3:
            parsed.node_4 = _json_int_value(doc, doc.array_item(nodes, 3), "element nodes")
        if not parsed.solid_extra_nodes:
            for i in range(4, node_count):
                parsed.extra_nodes[i - 4] = _json_int_value(
                    doc, doc.array_item(nodes, i), "element nodes"
                )
        if _json_has_key(doc, elem, "section"):
            parsed.section = _json_get_int(doc, elem, "section", -1)
        if _json_has_key(doc, elem, "material"):
            parsed.material = _json_get_int(doc, elem, "material", -1)
        var mat_ids = _json_key(doc, elem, "materials")
        if _json_has_value(doc, mat_ids):
            var material_count = _json_expect_array_len(doc, mat_ids, "element materials")
            if material_count > 6:
                material_count = 6
            parsed.material_count = material_count
            if material_count > 0:
                parsed.material_1 = _json_int_value(
                    doc, doc.array_item(mat_ids, 0), "element materials"
                )
            if material_count > 1:
                parsed.material_2 = _json_int_value(
                    doc, doc.array_item(mat_ids, 1), "element materials"
                )
            if material_count > 2:
                parsed.material_3 = _json_int_value(
                    doc, doc.array_item(mat_ids, 2), "element materials"
                )
            if material_count > 3:
                parsed.material_4 = _json_int_value(
                    doc, doc.array_item(mat_ids, 3), "element materials"
                )
            if material_count > 4:
                parsed.material_5 = _json_int_value(
                    doc, doc.array_item(mat_ids, 4), "element materials"
                )
            if material_count > 5:
                parsed.material_6 = _json_int_value(
                    doc, doc.array_item(mat_ids, 5), "element materials"
                )
        var damp_mat_ids = _json_key(doc, elem, "dampMats")
        if _json_has_value(doc, damp_mat_ids):
            var damp_material_count = _json_expect_array_len(doc, damp_mat_ids, "element dampMats")
            if damp_material_count > 6:
                damp_material_count = 6
            parsed.damp_material_count = damp_material_count
            if damp_material_count > 0:
                parsed.damp_material_1 = _json_int_value(
                    doc, doc.array_item(damp_mat_ids, 0), "element dampMats"
                )
            if damp_material_count > 1:
                parsed.damp_material_2 = _json_int_value(
                    doc, doc.array_item(damp_mat_ids, 1), "element dampMats"
                )
            if damp_material_count > 2:
                parsed.damp_material_3 = _json_int_value(
                    doc, doc.array_item(damp_mat_ids, 2), "element dampMats"
                )
            if damp_material_count > 3:
                parsed.damp_material_4 = _json_int_value(
                    doc, doc.array_item(damp_mat_ids, 3), "element dampMats"
                )
            if damp_material_count > 4:
                parsed.damp_material_5 = _json_int_value(
                    doc, doc.array_item(damp_mat_ids, 4), "element dampMats"
                )
            if damp_material_count > 5:
                parsed.damp_material_6 = _json_int_value(
                    doc, doc.array_item(damp_mat_ids, 5), "element dampMats"
                )
        if _json_has_key(doc, elem, "damp"):
            parsed.damping_tag = _json_get_int(doc, elem, "damp", -1)
        var dirs = _json_key(doc, elem, "dirs")
        if _json_has_value(doc, dirs):
            var dir_count = _json_expect_array_len(doc, dirs, "element dirs")
            if dir_count > 6:
                dir_count = 6
            parsed.dir_count = dir_count
            if dir_count > 0:
                parsed.dir_1 = _json_int_value(doc, doc.array_item(dirs, 0), "element dirs")
            if dir_count > 1:
                parsed.dir_2 = _json_int_value(doc, doc.array_item(dirs, 1), "element dirs")
            if dir_count > 2:
                parsed.dir_3 = _json_int_value(doc, doc.array_item(dirs, 2), "element dirs")
            if dir_count > 3:
                parsed.dir_4 = _json_int_value(doc, doc.array_item(dirs, 3), "element dirs")
            if dir_count > 4:
                parsed.dir_5 = _json_int_value(doc, doc.array_item(dirs, 4), "element dirs")
            if dir_count > 5:
                parsed.dir_6 = _json_int_value(doc, doc.array_item(dirs, 5), "element dirs")
        if _json_has_key(doc, elem, "area"):
            parsed.area = _json_get_float(doc, elem, "area", 0.0)
        if _json_has_key(doc, elem, "thickness"):
            parsed.thickness = _json_get_float(doc, elem, "thickness", 0.0)
        if parsed.type == "BeamContact2D":
            parsed.thickness = _required_material_param(doc, elem, "width")
        parsed.formulation = _json_get_string(doc, elem, "formulation", "PlaneStress")
        parsed.geom_transf = _json_get_string(doc, elem, "geomTransf", "Linear")
        if parsed.type == "ASDShellQ4":
            parsed.shell_eas = _json_get_bool(doc, elem, "eas", True)
            parsed.shell_drilling_stab = _json_get_float(doc, elem, "drilling_stab", 0.01)
        var vecxz = _json_key(doc, elem, "vecxz")
        if _json_has_value(doc, vecxz):
            var vec_count = _json_expect_array_len(doc, vecxz, "element vecxz")
            if vec_count > 0:
                parsed.geom_vecxz_1 = _json_number_value(
                    doc, doc.array_item(vecxz, 0), "element vecxz"
                )
            if vec_count > 1:
                parsed.geom_vecxz_2 = _json_number_value(
                    doc, doc.array_item(vecxz, 1), "element vecxz"
                )
            if vec_count > 2:
                parsed.geom_vecxz_3 = _json_number_value(
                    doc, doc.array_item(vecxz, 2), "element vecxz"
                )
            parsed.has_geom_vecxz = True
        parsed.integration = _json_get_string(doc, elem, "integration", "Lobatto")
        parsed.num_int_pts = _json_get_int(doc, elem, "num_int_pts", 3)
        parsed.thermal_beam = _json_get_bool(doc, elem, "thermal", False)
        if _json_get_bool(doc, elem, "cbdi", False):
            var cbdi_iters = _json_get_int(doc, elem, "max_iters", 10)
            var cbdi_tol = _json_get_float(doc, elem, "tol", 1.0e-12)
            if cbdi_iters <= 0 or cbdi_tol <= 0.0:
                abort("forceBeamColumnCBDI requires positive iteration limit and tolerance")
            parsed.cbdi = CbdiSettings(cbdi_iters, cbdi_tol)
        parsed.rho = _json_get_float(doc, elem, "rho", 0.0)
        parsed.use_cmass = _json_get_bool(doc, elem, "cMass", False)
        if parsed.type == "SSPbrickUP":
            parsed.up_perm_z = _json_get_float(doc, elem, "perm_z", 0.0)
            parsed.up_void_ratio = _json_get_float(doc, elem, "void_ratio", 0.0)
            parsed.up_stabilization = _json_get_float(doc, elem, "stabilization", 0.0)
            parsed.up_body_z = _json_get_float(doc, elem, "body_z", 0.0)
        if parsed.type == "quadUP" or parsed.type == "9_4_QuadUP" or parsed.type == "SSPbrickUP":
            parsed.up_bulk = _json_get_float(doc, elem, "fluid_bulk", 0.0)
            parsed.up_perm_x = _json_get_float(doc, elem, "perm_x", 0.0)
            parsed.up_perm_y = _json_get_float(doc, elem, "perm_y", 0.0)
            parsed.quad_density = _json_get_float(doc, elem, "fluid_density", 0.0)
        if parsed.type == "fourNodeQuad" or solid_nodes > 0:
            parsed.quad_density = _json_get_float(doc, elem, "rho", 0.0)
        if (
            parsed.type == "fourNodeQuad"
            or parsed.type == "quadUP"
            or parsed.type == "9_4_QuadUP"
            or parsed.type == "SSPbrickUP"
            or solid_nodes > 0
        ):
            parsed.quad_pressure = _json_get_float(doc, elem, "pressure", 0.0)
            parsed.quad_body_x = _json_get_float(doc, elem, "body_x", 0.0)
            parsed.quad_body_y = _json_get_float(doc, elem, "body_y", 0.0)
            if solid_nodes > 0:
                parsed.continuum_body_z = _json_get_float(doc, elem, "body_z", 0.0)
        parsed.element_mass = _json_get_float(doc, elem, "mass", 0.0)
        parsed.do_rayleigh = _json_get_bool(doc, elem, "doRayleigh", False)
        if parsed.type == "MVLEM":
            _ = _required_material_param(doc, elem, "density")
            parsed.do_rayleigh = True
        if parsed.type == "SFI_MVLEM":
            parsed.do_rayleigh = True
        var orient = _json_key(doc, elem, "orient")
        if _json_has_value(doc, orient):
            var x_vals = _json_key(doc, orient, "x")
            if _json_has_value(doc, x_vals):
                var x_count = _json_expect_array_len(doc, x_vals, "element orient x")
                if x_count > 0:
                    parsed.orient_x_1 = _json_number_value(
                        doc, doc.array_item(x_vals, 0), "element orient x"
                    )
                if x_count > 1:
                    parsed.orient_x_2 = _json_number_value(
                        doc, doc.array_item(x_vals, 1), "element orient x"
                    )
                if x_count > 2:
                    parsed.orient_x_3 = _json_number_value(
                        doc, doc.array_item(x_vals, 2), "element orient x"
                    )
                parsed.has_orient_x = True
            var y_vals = _json_key(doc, orient, "y")
            if _json_has_value(doc, y_vals):
                var y_count = _json_expect_array_len(doc, y_vals, "element orient y")
                if y_count > 0:
                    parsed.orient_y_1 = _json_number_value(
                        doc, doc.array_item(y_vals, 0), "element orient y"
                    )
                if y_count > 1:
                    parsed.orient_y_2 = _json_number_value(
                        doc, doc.array_item(y_vals, 1), "element orient y"
                    )
                if y_count > 2:
                    parsed.orient_y_3 = _json_number_value(
                        doc, doc.array_item(y_vals, 2), "element orient y"
                    )
                parsed.has_orient_y = True
        var p_delta = _json_key(doc, elem, "pDelta")
        if _json_has_value(doc, p_delta):
            var p_delta_count = _json_expect_array_len(doc, p_delta, "element pDelta")
            if p_delta_count > 0:
                parsed.pdelta_1 = _json_number_value(
                    doc, doc.array_item(p_delta, 0), "element pDelta"
                )
            if p_delta_count > 1:
                parsed.pdelta_2 = _json_number_value(
                    doc, doc.array_item(p_delta, 1), "element pDelta"
                )
            if p_delta_count > 2:
                parsed.pdelta_3 = _json_number_value(
                    doc, doc.array_item(p_delta, 2), "element pDelta"
                )
            if p_delta_count > 3:
                parsed.pdelta_4 = _json_number_value(
                    doc, doc.array_item(p_delta, 3), "element pDelta"
                )
            parsed.has_pdelta = p_delta_count > 0
        var shear_dist = _json_key(doc, elem, "shearDist")
        if _json_has_value(doc, shear_dist):
            var shear_dist_count = _json_expect_array_len(doc, shear_dist, "element shearDist")
            if shear_dist_count > 0:
                parsed.shear_dist_1 = _json_number_value(
                    doc, doc.array_item(shear_dist, 0), "element shearDist"
                )
            if shear_dist_count > 1:
                parsed.shear_dist_2 = _json_number_value(
                    doc, doc.array_item(shear_dist, 1), "element shearDist"
                )
            parsed.has_shear_dist = shear_dist_count > 0
        parsed.type_tag = element_type_tag(parsed.type)
        parsed.geom_tag = geom_transf_tag(parsed.geom_transf)
        case_input.elements.append(parsed^)

    var element_loads_raw = _json_key(doc, root, "element_loads")
    if _json_has_value(doc, element_loads_raw):
        for i in range(_json_expect_array_len(doc, element_loads_raw, "element_loads")):
            case_input.element_loads.append(
                parse_element_load_input_from_native(doc, doc.array_item(element_loads_raw, i))
            )

    var loads_raw = _json_key(doc, root, "loads")
    if _json_has_value(doc, loads_raw):
        for i in range(_json_expect_array_len(doc, loads_raw, "loads")):
            case_input.loads.append(
                parse_nodal_load_input_from_native(doc, doc.array_item(loads_raw, i))
            )

    var masses_raw = _json_key(doc, root, "masses")
    if _json_has_value(doc, masses_raw):
        for i in range(_json_expect_array_len(doc, masses_raw, "masses")):
            var mass = doc.array_item(masses_raw, i)
            case_input.masses.append(
                MassInput(
                    _json_get_int(doc, mass, "node", 0),
                    _json_get_int(doc, mass, "dof", 0),
                    _json_get_float(doc, mass, "value", 0.0),
                )
            )

    var analysis_raw = _json_key(doc, root, "analysis")
    case_input.analysis = parse_analysis_input_from_native(
        doc,
        analysis_raw,
        case_input.analysis_integrator_targets_pool,
        case_input.analysis_solver_chain_pool,
    )
    if case_input.analysis.type_tag == AnalysisTypeTag.Staged:
        if not _json_has_key(doc, analysis_raw, "stages"):
            abort("staged analysis requires analysis.stages")
        var stages_raw = _json_require_key(doc, analysis_raw, "stages")
        if _json_expect_array_len(doc, stages_raw, "analysis stages") < 1:
            abort("staged analysis requires non-empty analysis.stages")
        for i in range(_json_expect_array_len(doc, stages_raw, "analysis stages")):
            case_input.stages.append(
                parse_stage_input_from_native(doc, doc.array_item(stages_raw, i), source_info)
            )

    var mpc_raw = _json_key(doc, root, "mp_constraints")
    if _json_has_value(doc, mpc_raw):
        for i in range(_json_expect_array_len(doc, mpc_raw, "mp_constraints")):
            var mpc = doc.array_item(mpc_raw, i)
            var parsed = MPConstraintInput(
                _json_get_string(doc, mpc, "type", ""),
                _json_get_int(doc, mpc, "retained_node", 0),
                _json_get_int(doc, mpc, "constrained_node", 0),
            )
            var dofs = _json_key(doc, mpc, "dofs")
            if _json_has_value(doc, dofs):
                var dof_count = _json_expect_array_len(doc, dofs, "mp_constraint dofs")
                if dof_count > 6:
                    dof_count = 6
                parsed.dof_count = dof_count
                if dof_count > 0:
                    parsed.dof_1 = _json_int_value(
                        doc, doc.array_item(dofs, 0), "mp_constraint dofs"
                    )
                if dof_count > 1:
                    parsed.dof_2 = _json_int_value(
                        doc, doc.array_item(dofs, 1), "mp_constraint dofs"
                    )
                if dof_count > 2:
                    parsed.dof_3 = _json_int_value(
                        doc, doc.array_item(dofs, 2), "mp_constraint dofs"
                    )
                if dof_count > 3:
                    parsed.dof_4 = _json_int_value(
                        doc, doc.array_item(dofs, 3), "mp_constraint dofs"
                    )
                if dof_count > 4:
                    parsed.dof_5 = _json_int_value(
                        doc, doc.array_item(dofs, 4), "mp_constraint dofs"
                    )
                if dof_count > 5:
                    parsed.dof_6 = _json_int_value(
                        doc, doc.array_item(dofs, 5), "mp_constraint dofs"
                    )
            var perp_dirn = _json_key(doc, mpc, "perp_dirn")
            if _json_has_value(doc, perp_dirn):
                parsed.rigid_perp_dirn = _json_int_value(doc, perp_dirn, "mp_constraint perp_dirn")
            var constrained_dofs = _json_key(doc, mpc, "constrained_dofs")
            if _json_has_value(doc, constrained_dofs):
                var constrained_count = _json_expect_array_len(
                    doc, constrained_dofs, "mp_constraint constrained_dofs"
                )
                if constrained_count > 3:
                    abort("mp_constraint constrained_dofs supports at most 3 entries")
                parsed.rigid_constrained_dof_count = constrained_count
                if constrained_count > 0:
                    parsed.rigid_constrained_dof_1 = _json_int_value(
                        doc,
                        doc.array_item(constrained_dofs, 0),
                        "mp_constraint constrained_dofs",
                    )
                if constrained_count > 1:
                    parsed.rigid_constrained_dof_2 = _json_int_value(
                        doc,
                        doc.array_item(constrained_dofs, 1),
                        "mp_constraint constrained_dofs",
                    )
                if constrained_count > 2:
                    parsed.rigid_constrained_dof_3 = _json_int_value(
                        doc,
                        doc.array_item(constrained_dofs, 2),
                        "mp_constraint constrained_dofs",
                    )
            var retained_dofs = _json_key(doc, mpc, "retained_dofs")
            if _json_has_value(doc, retained_dofs):
                var retained_count = _json_expect_array_len(
                    doc, retained_dofs, "mp_constraint retained_dofs"
                )
                if retained_count > 3:
                    abort("mp_constraint retained_dofs supports at most 3 entries")
                parsed.rigid_retained_dof_count = retained_count
                if retained_count > 0:
                    parsed.rigid_retained_dof_1 = _json_int_value(
                        doc,
                        doc.array_item(retained_dofs, 0),
                        "mp_constraint retained_dofs",
                    )
                if retained_count > 1:
                    parsed.rigid_retained_dof_2 = _json_int_value(
                        doc,
                        doc.array_item(retained_dofs, 1),
                        "mp_constraint retained_dofs",
                    )
                if retained_count > 2:
                    parsed.rigid_retained_dof_3 = _json_int_value(
                        doc,
                        doc.array_item(retained_dofs, 2),
                        "mp_constraint retained_dofs",
                    )
            var matrix = _json_key(doc, mpc, "matrix")
            if _json_has_value(doc, matrix):
                var row_count = _json_expect_array_len(doc, matrix, "mp_constraint matrix")
                if row_count > 3:
                    abort("mp_constraint matrix supports at most 3 rows")
                parsed.rigid_matrix_row_count = row_count
                for row in range(row_count):
                    var row_index = doc.array_item(matrix, row)
                    var col_count = _json_expect_array_len(
                        doc, row_index, "mp_constraint matrix row"
                    )
                    if col_count > 3:
                        abort("mp_constraint matrix supports at most 3 columns")
                    if row == 0:
                        parsed.rigid_matrix_col_count = col_count
                    elif col_count != parsed.rigid_matrix_col_count:
                        abort("mp_constraint matrix rows must have consistent column counts")
                    if col_count > 0:
                        var value_1 = _json_number_value(
                            doc,
                            doc.array_item(row_index, 0),
                            "mp_constraint matrix row",
                        )
                        if row == 0:
                            parsed.rigid_matrix_11 = value_1
                        elif row == 1:
                            parsed.rigid_matrix_21 = value_1
                        else:
                            parsed.rigid_matrix_31 = value_1
                    if col_count > 1:
                        var value_2 = _json_number_value(
                            doc,
                            doc.array_item(row_index, 1),
                            "mp_constraint matrix row",
                        )
                        if row == 0:
                            parsed.rigid_matrix_12 = value_2
                        elif row == 1:
                            parsed.rigid_matrix_22 = value_2
                        else:
                            parsed.rigid_matrix_32 = value_2
                    if col_count > 2:
                        var value_3 = _json_number_value(
                            doc,
                            doc.array_item(row_index, 2),
                            "mp_constraint matrix row",
                        )
                        if row == 0:
                            parsed.rigid_matrix_13 = value_3
                        elif row == 1:
                            parsed.rigid_matrix_23 = value_3
                        else:
                            parsed.rigid_matrix_33 = value_3
            if _json_has_key(doc, mpc, "dx"):
                parsed.rigid_dx = _json_get_float(doc, mpc, "dx", 0.0)
            if _json_has_key(doc, mpc, "dy"):
                parsed.rigid_dy = _json_get_float(doc, mpc, "dy", 0.0)
            if _json_has_key(doc, mpc, "dz"):
                parsed.rigid_dz = _json_get_float(doc, mpc, "dz", 0.0)
            case_input.mp_constraints.append(parsed)

    case_input.pattern = parse_pattern_input_from_native(doc, _json_key(doc, root, "pattern"))
    case_input.rayleigh = parse_rayleigh_input_from_native(doc, _json_key(doc, root, "rayleigh"))
    _append_time_series_inputs_native(
        doc,
        root,
        source_info,
        case_input.time_series,
        case_input.time_series_values,
        case_input.time_series_times,
    )
    case_input.dampings = parse_damping_inputs_from_native(
        doc,
        _json_key(doc, root, "dampings"),
        case_input.time_series,
    )

    if not include_recorders:
        return case_input^

    var recorders_raw = _json_key(doc, root, "recorders")
    if not _json_has_value(doc, recorders_raw):
        return case_input^
    for i in range(_json_expect_array_len(doc, recorders_raw, "recorders")):
        var rec = doc.array_item(recorders_raw, i)
        var parsed = RecorderInput()
        parsed.start_stage = _json_get_int(doc, rec, "start_stage", 0)
        parsed.end_stage = _json_get_int(doc, rec, "end_stage", parsed.end_stage)
        parsed.delta_t = _json_get_float_alias2(doc, rec, "delta_t", "sample_dt", 0.0)
        parsed.type_tag = recorder_type_tag(_json_get_string(doc, rec, "type", ""))
        if parsed.type_tag == RecorderTypeTag.Unknown:
            abort("unsupported recorder type")
        if (
            parsed.type_tag == RecorderTypeTag.NodeDisplacement
            or parsed.type_tag == RecorderTypeTag.NodeVelocity
            or parsed.type_tag == RecorderTypeTag.NodeAcceleration
            or parsed.type_tag == RecorderTypeTag.EnvelopeNodeDisplacement
            or parsed.type_tag == RecorderTypeTag.EnvelopeNodeAcceleration
        ):
            if parsed.type_tag == RecorderTypeTag.NodeDisplacement:
                parsed.output = _json_get_string(doc, rec, "output", "node_disp")
            elif parsed.type_tag == RecorderTypeTag.NodeVelocity:
                parsed.output = _json_get_string(doc, rec, "output", "node_velocity")
            elif parsed.type_tag == RecorderTypeTag.NodeAcceleration:
                parsed.output = _json_get_string(doc, rec, "output", "node_accel")
            elif parsed.type_tag == RecorderTypeTag.EnvelopeNodeDisplacement:
                parsed.output = _json_get_string(doc, rec, "output", "envelope_node_displacement")
            else:
                parsed.output = _json_get_string(doc, rec, "output", "envelope_node_acceleration")
            var recorder_nodes = _json_key(doc, rec, "nodes")
            var recorder_dofs = _json_key(doc, rec, "dofs")
            if not _json_has_value(doc, recorder_nodes) or not _json_has_value(doc, recorder_dofs):
                abort("node recorder requires nodes and dofs")
            parsed.node_offset = len(case_input.recorder_nodes_pool)
            parsed.node_count = _json_expect_array_len(doc, recorder_nodes, "recorder nodes")
            for j in range(parsed.node_count):
                case_input.recorder_nodes_pool.append(
                    _json_int_value(doc, doc.array_item(recorder_nodes, j), "recorder nodes")
                )
            parsed.dof_offset = len(case_input.recorder_dofs_pool)
            parsed.dof_count = _json_expect_array_len(doc, recorder_dofs, "recorder dofs")
            for j in range(parsed.dof_count):
                case_input.recorder_dofs_pool.append(
                    _json_int_value(doc, doc.array_item(recorder_dofs, j), "recorder dofs")
                )
            if _json_has_key(doc, rec, "time_series"):
                parsed.time_series_tag = _json_get_int(doc, rec, "time_series", -1)
        elif (
            parsed.type_tag == RecorderTypeTag.ElementForce
            or parsed.type_tag == RecorderTypeTag.QuadStress
            or parsed.type_tag == RecorderTypeTag.QuadStrain
            or parsed.type_tag == RecorderTypeTag.QuadState
            or parsed.type_tag == RecorderTypeTag.ContactSecondaryForce
            or parsed.type_tag == RecorderTypeTag.ContactFrictionForce
            or parsed.type_tag == RecorderTypeTag.ContactScalarForce
            or parsed.type_tag == RecorderTypeTag.ContactPrimaryForce
            or parsed.type_tag == RecorderTypeTag.ElementLocalForce
            or parsed.type_tag == RecorderTypeTag.ElementBasicForce
            or parsed.type_tag == RecorderTypeTag.ElementDeformation
            or parsed.type_tag == RecorderTypeTag.EnvelopeElementForce
            or parsed.type_tag == RecorderTypeTag.EnvelopeElementLocalForce
        ):
            if (
                parsed.type_tag == RecorderTypeTag.ContactSecondaryForce
                or parsed.type_tag == RecorderTypeTag.ContactFrictionForce
                or parsed.type_tag == RecorderTypeTag.ContactScalarForce
                or parsed.type_tag == RecorderTypeTag.ContactPrimaryForce
            ):
                parsed.output = _json_get_string(
                    doc, rec, "output", _json_get_string(doc, rec, "type", "")
                )
            elif parsed.type_tag == RecorderTypeTag.ElementForce:
                parsed.output = _json_get_string(doc, rec, "output", "element_force")
            elif (
                parsed.type_tag == RecorderTypeTag.QuadStress
                or parsed.type_tag == RecorderTypeTag.QuadStrain
                or parsed.type_tag == RecorderTypeTag.QuadState
            ):
                var output = (
                    "quad_stress" if parsed.type_tag
                    == RecorderTypeTag.QuadStress else "quad_strain"
                )
                if parsed.type_tag == RecorderTypeTag.QuadState:
                    output = "quad_state"
                parsed.output = _json_get_string(doc, rec, "output", output)
                parsed.integration_point = _json_get_int(doc, rec, "integration_point", 1) - 1
                if parsed.integration_point < 0 or parsed.integration_point >= 27:
                    abort("solid material recorder integration point must be in 1..27")
                parsed.stress_components = _json_get_int(doc, rec, "stress_components", 0)
            elif parsed.type_tag == RecorderTypeTag.ElementLocalForce:
                parsed.output = _json_get_string(doc, rec, "output", "element_local_force")
            elif parsed.type_tag == RecorderTypeTag.ElementBasicForce:
                parsed.material_stress = (
                    _json_get_string(doc, rec, "truss_response", "") == "material"
                )
                parsed.output = _json_get_string(doc, rec, "output", "element_basic_force")
            elif parsed.type_tag == RecorderTypeTag.ElementDeformation:
                parsed.output = _json_get_string(doc, rec, "output", "element_deformation")
            elif parsed.type_tag == RecorderTypeTag.EnvelopeElementForce:
                parsed.output = _json_get_string(doc, rec, "output", "envelope_element_force")
            else:
                parsed.output = _json_get_string(doc, rec, "output", "envelope_element_local_force")
            var recorder_elements = _json_key(doc, rec, "elements")
            if not _json_has_value(doc, recorder_elements):
                abort("element recorder requires elements")
            parsed.element_offset = len(case_input.recorder_elements_pool)
            parsed.element_count = _json_expect_array_len(
                doc, recorder_elements, "recorder elements"
            )
            for j in range(parsed.element_count):
                case_input.recorder_elements_pool.append(
                    _json_int_value(doc, doc.array_item(recorder_elements, j), "recorder elements")
                )
            var recorder_dofs = _json_key(doc, rec, "dofs")
            if _json_has_value(doc, recorder_dofs):
                parsed.dof_offset = len(case_input.recorder_dofs_pool)
                parsed.dof_count = _json_expect_array_len(doc, recorder_dofs, "recorder dofs")
                for j in range(parsed.dof_count):
                    case_input.recorder_dofs_pool.append(
                        _json_int_value(doc, doc.array_item(recorder_dofs, j), "recorder dofs")
                    )
        elif parsed.type_tag == RecorderTypeTag.NodeReaction:
            parsed.output = _json_get_string(doc, rec, "output", "reaction")
            var recorder_nodes = _json_key(doc, rec, "nodes")
            var recorder_dofs = _json_key(doc, rec, "dofs")
            if not _json_has_value(doc, recorder_nodes) or not _json_has_value(doc, recorder_dofs):
                abort("node_reaction recorder requires nodes and dofs")
            parsed.node_offset = len(case_input.recorder_nodes_pool)
            parsed.node_count = _json_expect_array_len(doc, recorder_nodes, "recorder nodes")
            for j in range(parsed.node_count):
                case_input.recorder_nodes_pool.append(
                    _json_int_value(doc, doc.array_item(recorder_nodes, j), "recorder nodes")
                )
            parsed.dof_offset = len(case_input.recorder_dofs_pool)
            parsed.dof_count = _json_expect_array_len(doc, recorder_dofs, "recorder dofs")
            for j in range(parsed.dof_count):
                case_input.recorder_dofs_pool.append(
                    _json_int_value(doc, doc.array_item(recorder_dofs, j), "recorder dofs")
                )
        elif parsed.type_tag == RecorderTypeTag.Drift:
            parsed.output = _json_get_string(doc, rec, "output", "drift")
            if (
                not _json_has_key(doc, rec, "i_node")
                or not _json_has_key(doc, rec, "j_node")
                or not _json_has_key(doc, rec, "dof")
                or not _json_has_key(doc, rec, "perp_dirn")
            ):
                abort("drift recorder requires i_node, j_node, dof, perp_dirn")
            parsed.i_node = _json_get_int(doc, rec, "i_node", 0)
            parsed.j_node = _json_get_int(doc, rec, "j_node", 0)
            parsed.drift_dof = _json_get_int(doc, rec, "dof", 0)
            parsed.perp_dirn = _json_get_int(doc, rec, "perp_dirn", 0)
        elif (
            parsed.type_tag == RecorderTypeTag.SectionForce
            or parsed.type_tag == RecorderTypeTag.SectionDeformation
        ):
            if parsed.type_tag == RecorderTypeTag.SectionForce:
                parsed.output = _json_get_string(doc, rec, "output", "section_force")
            else:
                parsed.output = _json_get_string(doc, rec, "output", "section_deformation")
            var recorder_elements = _json_key(doc, rec, "elements")
            if not _json_has_value(doc, recorder_elements):
                abort("section recorder requires elements")
            parsed.element_offset = len(case_input.recorder_elements_pool)
            parsed.element_count = _json_expect_array_len(
                doc, recorder_elements, "section recorder elements"
            )
            for j in range(parsed.element_count):
                case_input.recorder_elements_pool.append(
                    _json_int_value(
                        doc, doc.array_item(recorder_elements, j), "section recorder elements"
                    )
                )
            var sections_node = _json_key(doc, rec, "sections")
            if not _json_has_value(doc, sections_node):
                sections_node = _json_key(doc, rec, "section")
                if not _json_has_value(doc, sections_node):
                    abort("section recorder requires section or sections")
                parsed.section_offset = len(case_input.recorder_sections_pool)
                parsed.section_count = 1
                case_input.recorder_sections_pool.append(
                    _json_int_value(doc, sections_node, "section recorder section")
                )
            else:
                parsed.section_offset = len(case_input.recorder_sections_pool)
                parsed.section_count = _json_expect_array_len(
                    doc, sections_node, "section recorder sections"
                )
                if parsed.section_count < 1:
                    abort("section recorder requires non-empty sections")
                for j in range(parsed.section_count):
                    case_input.recorder_sections_pool.append(
                        _json_int_value(
                            doc, doc.array_item(sections_node, j), "section recorder sections"
                        )
                    )
        else:
            parsed.output = _json_get_string(doc, rec, "output", "modal")
            var recorder_nodes = _json_key(doc, rec, "nodes")
            var recorder_dofs = _json_key(doc, rec, "dofs")
            if not _json_has_value(doc, recorder_nodes) or not _json_has_value(doc, recorder_dofs):
                abort("modal_eigen recorder requires nodes and dofs")
            parsed.node_offset = len(case_input.recorder_nodes_pool)
            parsed.node_count = _json_expect_array_len(doc, recorder_nodes, "modal recorder nodes")
            for j in range(parsed.node_count):
                case_input.recorder_nodes_pool.append(
                    _json_int_value(doc, doc.array_item(recorder_nodes, j), "modal recorder nodes")
                )
            parsed.dof_offset = len(case_input.recorder_dofs_pool)
            parsed.dof_count = _json_expect_array_len(doc, recorder_dofs, "modal recorder dofs")
            for j in range(parsed.dof_count):
                case_input.recorder_dofs_pool.append(
                    _json_int_value(doc, doc.array_item(recorder_dofs, j), "modal recorder dofs")
                )
            var modes_raw = _json_key(doc, rec, "modes")
            parsed.mode_offset = len(case_input.recorder_modes_pool)
            parsed.mode_count = _json_expect_array_len(doc, modes_raw, "modal recorder modes")
            for j in range(parsed.mode_count):
                case_input.recorder_modes_pool.append(
                    _json_int_value(doc, doc.array_item(modes_raw, j), "modal recorder modes")
                )
        case_input.recorders.append(parsed^)
    return case_input^
