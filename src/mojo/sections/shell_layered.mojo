from collections import List
from elements.utils import _ensure_zero_matrix, _ensure_zero_vector, _zero_matrix
from math import cos, sin
from materials import (
    UniMaterialDef,
    UniMaterialState,
    uni_mat_is_elastic,
    uniaxial_set_trial_strain,
)
from materials.concrete_s import (
    ConcreteSDef,
    ConcreteSState,
    PlaneStressTangent,
    PlaneStressVector,
    concrete_s_set_trial_strain,
)
from os import abort
from solver.eigen import jacobi_symmetric_eigen
from solver.run_case.input_types import MaterialInput, SectionInput, ShellLayerInput


alias SHELL_LAYER_KIND_ELASTIC_PLATE = 1
alias SHELL_LAYER_KIND_REBAR = 2
alias SHELL_LAYER_KIND_CONCRETE_S = 3


struct LayeredShellSectionDef(Defaultable, ImplicitlyCopyable, Movable):
    var section_id: Int
    var layer_count: Int
    var total_thickness: Float64
    var rho_area: Float64
    var layer_kind: List[Int]
    var layer_thickness: List[Float64]
    var layer_z: List[Float64]
    var layer_rho: List[Float64]
    var elastic_E: List[Float64]
    var elastic_nu: List[Float64]
    var elastic_gmod: List[Float64]
    var rebar_cos: List[Float64]
    var rebar_sin: List[Float64]
    var rebar_slot: List[Int]
    var rebar_def_index: List[Int]
    var rebar_count: Int
    var runtime_instance_count: Int
    var concrete_slot: List[Int]
    var concrete_defs: List[ConcreteSDef]
    var concrete_states: List[ConcreteSState]
    var drill_stiffness: Float64

    fn __init__(out self):
        self.section_id = -1
        self.layer_count = 0
        self.total_thickness = 0.0
        self.rho_area = 0.0
        self.layer_kind = []
        self.layer_thickness = []
        self.layer_z = []
        self.layer_rho = []
        self.elastic_E = []
        self.elastic_nu = []
        self.elastic_gmod = []
        self.rebar_cos = []
        self.rebar_sin = []
        self.rebar_slot = []
        self.rebar_def_index = []
        self.rebar_count = 0
        self.runtime_instance_count = 0
        self.concrete_slot = []
        self.concrete_defs = []
        self.concrete_states = []
        self.drill_stiffness = 0.0

    fn __copyinit__(out self, existing: Self):
        self.section_id = existing.section_id
        self.layer_count = existing.layer_count
        self.total_thickness = existing.total_thickness
        self.rho_area = existing.rho_area
        self.layer_kind = existing.layer_kind.copy()
        self.layer_thickness = existing.layer_thickness.copy()
        self.layer_z = existing.layer_z.copy()
        self.layer_rho = existing.layer_rho.copy()
        self.elastic_E = existing.elastic_E.copy()
        self.elastic_nu = existing.elastic_nu.copy()
        self.elastic_gmod = existing.elastic_gmod.copy()
        self.rebar_cos = existing.rebar_cos.copy()
        self.rebar_sin = existing.rebar_sin.copy()
        self.rebar_slot = existing.rebar_slot.copy()
        self.rebar_def_index = existing.rebar_def_index.copy()
        self.rebar_count = existing.rebar_count
        self.runtime_instance_count = existing.runtime_instance_count
        self.concrete_slot = existing.concrete_slot.copy()
        self.concrete_defs = existing.concrete_defs.copy()
        self.concrete_states = existing.concrete_states.copy()
        self.drill_stiffness = existing.drill_stiffness


fn append_layered_shell_section_from_input(
    mut defs: List[LayeredShellSectionDef],
    sec: SectionInput,
    shell_layers: List[ShellLayerInput],
    materials_by_id: List[MaterialInput],
    uniaxial_def_by_id: List[Int],
):
    if sec.type != "LayeredShellSection":
        abort("append_layered_shell_section_from_input requires LayeredShellSection")
    if sec.shell_layer_count <= 0:
        abort("LayeredShellSection requires at least one layer")

    var sec_def = LayeredShellSectionDef()
    sec_def.section_id = sec.id
    sec_def.layer_count = sec.shell_layer_count
    sec_def.layer_kind.resize(sec.shell_layer_count, 0)
    sec_def.layer_thickness.resize(sec.shell_layer_count, 0.0)
    sec_def.layer_z.resize(sec.shell_layer_count, 0.0)
    sec_def.layer_rho.resize(sec.shell_layer_count, 0.0)
    sec_def.elastic_E.resize(sec.shell_layer_count, 0.0)
    sec_def.elastic_nu.resize(sec.shell_layer_count, 0.0)
    sec_def.elastic_gmod.resize(sec.shell_layer_count, 0.0)
    sec_def.rebar_cos.resize(sec.shell_layer_count, 1.0)
    sec_def.rebar_sin.resize(sec.shell_layer_count, 0.0)
    sec_def.rebar_slot.resize(sec.shell_layer_count, -1)
    sec_def.concrete_slot.resize(sec.shell_layer_count, -1)

    for i in range(sec.shell_layer_count):
        var layer = shell_layers[sec.shell_layer_offset + i]
        if layer.material < 0 or layer.material >= len(materials_by_id):
            abort("LayeredShellSection layer material not found")
        if layer.thickness <= 0.0:
            abort("LayeredShellSection layer thickness must be > 0")
        var mat = materials_by_id[layer.material]
        sec_def.layer_thickness[i] = layer.thickness
        sec_def.total_thickness += layer.thickness

        if mat.type == "PlateFromPlaneStress":
            if mat.base_material < 0 or mat.base_material >= len(materials_by_id):
                abort("PlateFromPlaneStress base material not found")
            var base_mat = materials_by_id[mat.base_material]
            if base_mat.type == "ConcreteS":
                sec_def.layer_kind[i] = SHELL_LAYER_KIND_CONCRETE_S
                sec_def.concrete_slot[i] = len(sec_def.concrete_defs)
                sec_def.concrete_defs.append(
                    ConcreteSDef(base_mat.E, base_mat.nu, base_mat.fc, base_mat.ft, base_mat.Es)
                )
                sec_def.elastic_gmod[i] = mat.gmod
                continue
            if base_mat.type != "ElasticIsotropic":
                abort("PlateFromPlaneStress requires ElasticIsotropic or ConcreteS")
            sec_def.layer_kind[i] = SHELL_LAYER_KIND_ELASTIC_PLATE
            sec_def.elastic_E[i] = base_mat.E
            sec_def.elastic_nu[i] = base_mat.nu
            sec_def.elastic_gmod[i] = mat.gmod
            sec_def.layer_rho[i] = base_mat.rho
            sec_def.rho_area += base_mat.rho * layer.thickness
            continue

        if mat.type == "ElasticIsotropic":
            sec_def.layer_kind[i] = SHELL_LAYER_KIND_ELASTIC_PLATE
            sec_def.elastic_E[i] = mat.E
            sec_def.elastic_nu[i] = mat.nu
            sec_def.elastic_gmod[i] = mat.E / (2.0 * (1.0 + mat.nu))
            sec_def.layer_rho[i] = mat.rho
            sec_def.rho_area += mat.rho * layer.thickness
            continue

        if mat.type == "PlateRebar":
            if mat.base_material < 0 or mat.base_material >= len(materials_by_id):
                abort("PlateRebar base material not found")
            var base_mat = materials_by_id[mat.base_material]
            if (
                mat.base_material >= len(uniaxial_def_by_id)
                or uniaxial_def_by_id[mat.base_material] < 0
            ):
                abort("PlateRebar base material must be a supported uniaxial material")
            var slot = len(sec_def.rebar_def_index)
            sec_def.rebar_def_index.append(uniaxial_def_by_id[mat.base_material])
            sec_def.layer_kind[i] = SHELL_LAYER_KIND_REBAR
            sec_def.rebar_slot[i] = slot
            if mat.angle == 0.0:
                sec_def.rebar_cos[i] = 1.0
                sec_def.rebar_sin[i] = 0.0
            elif mat.angle == 90.0:
                sec_def.rebar_cos[i] = 0.0
                sec_def.rebar_sin[i] = 1.0
            else:
                var rang = mat.angle * 3.141592653589793 / 180.0
                sec_def.rebar_cos[i] = cos(rang)
                sec_def.rebar_sin[i] = sin(rang)
            sec_def.layer_rho[i] = base_mat.rho
            sec_def.rho_area += base_mat.rho * layer.thickness
            continue

        if mat.type == "PlaneStressUserMaterial":
            abort(
                "LayeredShellSection runtime does not support PlaneStressUserMaterial because the"
                " OpenSees PSUMAT implementation is not present in the reference repo"
            )
        abort("unsupported LayeredShellSection material: " + mat.type)

    if sec_def.total_thickness <= 0.0:
        abort("LayeredShellSection total thickness must be > 0")

    var curr_loc = 0.0
    var inverse_thickness = 1.0 / sec_def.total_thickness
    for i in range(sec.shell_layer_count):
        curr_loc += sec_def.layer_thickness[i]
        sec_def.layer_z[i] = 0.5 * sec_def.total_thickness * (curr_loc * inverse_thickness - 1.0)
        curr_loc += sec_def.layer_thickness[i]
    sec_def.rebar_count = len(sec_def.rebar_def_index)

    defs.append(sec_def^)


fn layered_shell_runtime_alloc_instances(
    mut defs: List[LayeredShellSectionDef],
    instance_counts: List[Int],
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    mut uniaxial_state_defs: List[Int],
    mut section_uniaxial_offsets: List[Int],
    mut section_uniaxial_counts: List[Int],
) -> Bool:
    if len(instance_counts) != len(defs):
        abort("LayeredShellSection instance count mapping mismatch")

    section_uniaxial_offsets.resize(len(defs), 0)
    section_uniaxial_counts.resize(len(defs), 0)
    var used_nonelastic = False

    for s in range(len(defs)):
        ref sec_def = defs[s]
        sec_def.runtime_instance_count = instance_counts[s]
        if sec_def.runtime_instance_count > 0:
            sec_def.drill_stiffness = _initial_drill_stiffness(sec_def, uniaxial_defs)
            if len(sec_def.concrete_defs) > 0:
                sec_def.concrete_states.resize(
                    sec_def.runtime_instance_count * len(sec_def.concrete_defs), ConcreteSState()
                )
                used_nonelastic = True
        section_uniaxial_offsets[s] = len(uniaxial_states)
        section_uniaxial_counts[s] = sec_def.runtime_instance_count * sec_def.rebar_count

        for _ in range(sec_def.runtime_instance_count):
            for slot in range(sec_def.rebar_count):
                var def_index = sec_def.rebar_def_index[slot]
                if def_index < 0 or def_index >= len(uniaxial_defs):
                    abort("LayeredShellSection rebar material definition out of range")
                var mat_def = uniaxial_defs[def_index]
                uniaxial_states.append(UniMaterialState(mat_def))
                uniaxial_state_defs.append(def_index)
                if not uni_mat_is_elastic(mat_def):
                    used_nonelastic = True
    return used_nonelastic


fn layered_shell_commit_all(mut defs: List[LayeredShellSectionDef]):
    for section in range(len(defs)):
        ref states = defs[section].concrete_states
        for i in range(len(states)):
            states[i].commit()


fn _elastic_plane_tangent(E: Float64, nu: Float64) -> PlaneStressTangent:
    var tangent = PlaneStressTangent(fill=0.0)
    var factor = E / (1.0 - nu * nu)
    tangent[0] = factor
    tangent[1] = factor * nu
    tangent[3] = tangent[1]
    tangent[4] = factor
    tangent[8] = factor * (1.0 - nu) * 0.5
    return tangent


fn _rebar_plane_tangent(tan: Float64, c: Float64, s: Float64) -> PlaneStressTangent:
    var tangent = PlaneStressTangent(fill=0.0)
    tangent[0] = tan * c * c * c * c
    tangent[1] = tan * c * c * s * s
    tangent[2] = tan * c * c * c * s
    tangent[3] = tangent[1]
    tangent[4] = tan * s * s * s * s
    tangent[5] = tan * c * s * s * s
    tangent[6] = tangent[2]
    tangent[7] = tangent[5]
    tangent[8] = tangent[1]
    return tangent


fn _initial_drill_stiffness(
    sec_def: LayeredShellSectionDef, uniaxial_defs: List[UniMaterialDef]
) -> Float64:
    # ShellMITC4 fixes the drilling modulus at the smallest eigenvalue of
    # the initial membrane tangent.
    var membrane = _zero_matrix(3, 3)
    for i in range(sec_def.layer_count):
        var tangent: PlaneStressTangent
        if sec_def.layer_kind[i] == SHELL_LAYER_KIND_CONCRETE_S:
            tangent = sec_def.concrete_defs[sec_def.concrete_slot[i]].initial_tangent
        elif sec_def.layer_kind[i] == SHELL_LAYER_KIND_REBAR:
            var def_index = sec_def.rebar_def_index[sec_def.rebar_slot[i]]
            # OpenSees LayeredShell's initial tangent reads each fresh
            # layer's current tangent.
            var state = UniMaterialState(uniaxial_defs[def_index])
            tangent = _rebar_plane_tangent(
                state.tangent_t, sec_def.rebar_cos[i], sec_def.rebar_sin[i]
            )
        else:
            tangent = _elastic_plane_tangent(sec_def.elastic_E[i], sec_def.elastic_nu[i])
        for p in range(3):
            for q in range(3):
                membrane[p][q] += tangent[3 * p + q] * sec_def.layer_thickness[i]

    var eigenvalues: List[Float64] = []
    var eigenvectors: List[List[Float64]] = []
    jacobi_symmetric_eigen(membrane, eigenvalues, eigenvectors)
    return min(eigenvalues[0], min(eigenvalues[1], eigenvalues[2]))


fn layered_shell_set_trial_from_offset(
    mut sec_def: LayeredShellSectionDef,
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    section_state_offset: Int,
    section_state_count: Int,
    instance_slot: Int,
    strain: List[Float64],
    mut stress_out: List[Float64],
    mut tangent_out: List[List[Float64]],
) raises:
    if len(strain) != 8:
        abort("LayeredShellSection strain vector must have size 8")
    _ensure_zero_vector(stress_out, 8)
    _ensure_zero_matrix(tangent_out, 8, 8)
    if instance_slot < 0 or instance_slot >= sec_def.runtime_instance_count:
        abort("LayeredShellSection instance slot out of range")
    if section_state_offset < 0 or section_state_offset + section_state_count > len(
        uniaxial_states
    ):
        abort("LayeredShellSection state range out of bounds")
    if section_state_count != sec_def.runtime_instance_count * sec_def.rebar_count:
        abort("LayeredShellSection state count mismatch")

    for i in range(sec_def.layer_count):
        var z = sec_def.layer_z[i]
        var weight = sec_def.layer_thickness[i]
        var x = strain[0] - z * strain[3]
        var y = strain[1] - z * strain[4]
        var xy = strain[2] - z * strain[5]
        var layer_stress = PlaneStressVector(fill=0.0)
        var tangent: PlaneStressTangent
        var kind = sec_def.layer_kind[i]

        if kind == SHELL_LAYER_KIND_ELASTIC_PLATE:
            tangent = _elastic_plane_tangent(sec_def.elastic_E[i], sec_def.elastic_nu[i])
            layer_stress[0] = tangent[0] * x + tangent[1] * y
            layer_stress[1] = tangent[3] * x + tangent[4] * y
            layer_stress[2] = tangent[8] * xy
        elif kind == SHELL_LAYER_KIND_CONCRETE_S:
            var slot = sec_def.concrete_slot[i]
            var state_index = instance_slot * len(sec_def.concrete_defs) + slot
            tangent = PlaneStressTangent(fill=0.0)
            concrete_s_set_trial_strain(
                sec_def.concrete_defs[slot],
                sec_def.concrete_states[state_index],
                x,
                y,
                xy,
                layer_stress,
                tangent,
            )
        else:
            var slot = sec_def.rebar_slot[i]
            var state_index = section_state_offset + instance_slot * sec_def.rebar_count + slot
            var def_index = sec_def.rebar_def_index[slot]
            ref state = uniaxial_states[state_index]
            var c = sec_def.rebar_cos[i]
            var s = sec_def.rebar_sin[i]
            var axial_strain = x * c * c + y * s * s + xy * c * s
            uniaxial_set_trial_strain(uniaxial_defs[def_index], state, axial_strain)
            layer_stress[0] = state.sig_t * c * c
            layer_stress[1] = state.sig_t * s * s
            layer_stress[2] = state.sig_t * c * s
            tangent = _rebar_plane_tangent(state.tangent_t, c, s)

        for p in range(3):
            stress_out[p] += layer_stress[p] * weight
            stress_out[p + 3] += (z * layer_stress[p]) * weight
            for q in range(3):
                var weighted = tangent[3 * p + q] * weight
                tangent_out[p][q] += weighted
                tangent_out[p][q + 3] -= z * weighted
                tangent_out[p + 3][q] += z * weighted
                tangent_out[p + 3][q + 3] -= z * z * weighted

        # The plate wrapper supplies the two elastic transverse shear terms.
        var gmod = sec_def.elastic_gmod[i]
        for p in range(6, 8):
            stress_out[p] += (gmod * strain[p]) * weight
            tangent_out[p][p] += gmod * weight
