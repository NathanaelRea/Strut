from collections import List
from elements.utils import _ensure_zero_matrix, _ensure_zero_vector, _zero_matrix
from math import cos, sin
from materials import (
    UniMaterialDef,
    UniMaterialState,
    uni_mat_is_elastic,
    uniaxial_set_trial_strain,
)
from os import abort
from solver.run_case.input_types import MaterialInput, SectionInput, ShellLayerInput


alias SHELL_LAYER_KIND_ELASTIC_PLATE = 1
alias SHELL_LAYER_KIND_REBAR = 2


struct LayeredShellSectionDef(Defaultable, Movable, ImplicitlyCopyable):
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
    var rebar_angle_deg: List[Float64]
    var rebar_cos: List[Float64]
    var rebar_sin: List[Float64]
    var rebar_slot: List[Int]
    var rebar_def_index: List[Int]
    var rebar_count: Int
    var runtime_instance_count: Int

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
        self.rebar_angle_deg = []
        self.rebar_cos = []
        self.rebar_sin = []
        self.rebar_slot = []
        self.rebar_def_index = []
        self.rebar_count = 0
        self.runtime_instance_count = 0

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
        self.rebar_angle_deg = existing.rebar_angle_deg.copy()
        self.rebar_cos = existing.rebar_cos.copy()
        self.rebar_sin = existing.rebar_sin.copy()
        self.rebar_slot = existing.rebar_slot.copy()
        self.rebar_def_index = existing.rebar_def_index.copy()
        self.rebar_count = existing.rebar_count
        self.runtime_instance_count = existing.runtime_instance_count


fn append_layered_shell_section_from_input(
    mut defs: List[LayeredShellSectionDef],
    sec: SectionInput,
    shell_layers: List[ShellLayerInput],
    materials_by_id: List[MaterialInput],
    uniaxial_def_by_id: List[Int],
    shell_material_props: List[Float64],
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
    sec_def.rebar_angle_deg.resize(sec.shell_layer_count, 0.0)
    sec_def.rebar_cos.resize(sec.shell_layer_count, 1.0)
    sec_def.rebar_sin.resize(sec.shell_layer_count, 0.0)
    sec_def.rebar_slot.resize(sec.shell_layer_count, -1)

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
            if base_mat.type == "PlaneStressUserMaterial":
                if (
                    base_mat.props_count <= 0
                    or base_mat.props_offset < 0
                    or base_mat.props_offset + base_mat.props_count
                        > len(shell_material_props)
                ):
                    abort("PlaneStressUserMaterial surrogate requires props data")
                var nu = shell_material_props[
                    base_mat.props_offset + base_mat.props_count - 1
                ]
                sec_def.layer_kind[i] = SHELL_LAYER_KIND_ELASTIC_PLATE
                sec_def.elastic_E[i] = 2.0 * mat.gmod * (1.0 + nu)
                sec_def.elastic_nu[i] = nu
                sec_def.elastic_gmod[i] = mat.gmod
                sec_def.layer_rho[i] = base_mat.rho
                sec_def.rho_area += base_mat.rho * layer.thickness
                continue
            if base_mat.type != "ElasticIsotropic":
                abort(
                    "LayeredShellSection currently supports PlateFromPlaneStress only over ElasticIsotropic"
                )
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
            sec_def.rebar_angle_deg[i] = mat.angle
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
                "LayeredShellSection runtime does not support PlaneStressUserMaterial because the OpenSees PSUMAT implementation is not present in the reference repo"
            )
        abort("unsupported LayeredShellSection material: " + mat.type)

    if sec_def.total_thickness <= 0.0:
        abort("LayeredShellSection total thickness must be > 0")

    var curr_loc = 0.0
    for i in range(sec.shell_layer_count):
        curr_loc += sec_def.layer_thickness[i]
        sec_def.layer_z[i] = curr_loc - 0.5 * sec_def.total_thickness
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


fn layered_shell_set_trial_from_offset(
    sec_def: LayeredShellSectionDef,
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

    var layer_strain: List[Float64] = []
    layer_strain.resize(5, 0.0)
    var layer_stress: List[Float64] = []
    layer_stress.resize(5, 0.0)
    var layer_tangent = _zero_matrix(5, 5)
    var plane_tangent = _zero_matrix(3, 3)

    for i in range(sec_def.layer_count):
        var z = sec_def.layer_z[i]
        var weight = sec_def.layer_thickness[i]
        layer_strain[0] = strain[0] - z * strain[3]
        layer_strain[1] = strain[1] - z * strain[4]
        layer_strain[2] = strain[2] - z * strain[5]
        layer_strain[3] = strain[7]
        layer_strain[4] = strain[6]

        for p in range(5):
            layer_stress[p] = 0.0
        _ensure_zero_matrix(layer_tangent, 5, 5)

        if sec_def.layer_kind[i] == SHELL_LAYER_KIND_ELASTIC_PLATE:
            var E = sec_def.elastic_E[i]
            var nu = sec_def.elastic_nu[i]
            var gmod = sec_def.elastic_gmod[i]
            var factor = E / (1.0 - nu * nu)
            plane_tangent[0][0] = factor
            plane_tangent[0][1] = factor * nu
            plane_tangent[0][2] = 0.0
            plane_tangent[1][0] = factor * nu
            plane_tangent[1][1] = factor
            plane_tangent[1][2] = 0.0
            plane_tangent[2][0] = 0.0
            plane_tangent[2][1] = 0.0
            plane_tangent[2][2] = factor * (1.0 - nu) * 0.5
            for p in range(3):
                for q in range(3):
                    layer_tangent[p][q] = plane_tangent[p][q]
            layer_tangent[3][3] = gmod
            layer_tangent[4][4] = gmod
            for p in range(5):
                var sum = 0.0
                for q in range(5):
                    sum += layer_tangent[p][q] * layer_strain[q]
                layer_stress[p] = sum
        elif sec_def.layer_kind[i] == SHELL_LAYER_KIND_REBAR:
            var rebar_slot = sec_def.rebar_slot[i]
            if rebar_slot < 0:
                abort("LayeredShellSection rebar slot missing")
            var state_index = (
                section_state_offset + instance_slot * sec_def.rebar_count + rebar_slot
            )
            if state_index < section_state_offset or state_index >= (
                section_state_offset + section_state_count
            ):
                abort("LayeredShellSection rebar state index out of range")
            var def_index = sec_def.rebar_def_index[rebar_slot]
            if def_index < 0 or def_index >= len(uniaxial_defs):
                abort("LayeredShellSection rebar definition index out of range")
            var mat_def = uniaxial_defs[def_index]
            ref state = uniaxial_states[state_index]
            var c = sec_def.rebar_cos[i]
            var s = sec_def.rebar_sin[i]
            var axial_strain: Float64
            if sec_def.rebar_angle_deg[i] == 0.0:
                axial_strain = layer_strain[0]
            elif sec_def.rebar_angle_deg[i] == 90.0:
                axial_strain = layer_strain[1]
            else:
                axial_strain = layer_strain[0] * c * c + layer_strain[1] * s * s + layer_strain[2] * c * s
            uniaxial_set_trial_strain(mat_def, state, axial_strain)
            var sig = state.sig_t
            var tan = state.tangent_t
            if sec_def.rebar_angle_deg[i] == 0.0:
                layer_stress[0] = sig
                layer_tangent[0][0] = tan
            elif sec_def.rebar_angle_deg[i] == 90.0:
                layer_stress[1] = sig
                layer_tangent[1][1] = tan
            else:
                layer_stress[0] = sig * c * c
                layer_stress[1] = sig * s * s
                layer_stress[2] = sig * c * s
                layer_tangent[0][0] = tan * c * c * c * c
                layer_tangent[0][1] = tan * c * c * s * s
                layer_tangent[0][2] = tan * c * c * c * s
                layer_tangent[1][0] = layer_tangent[0][1]
                layer_tangent[1][1] = tan * s * s * s * s
                layer_tangent[1][2] = tan * c * s * s * s
                layer_tangent[2][0] = layer_tangent[0][2]
                layer_tangent[2][1] = layer_tangent[1][2]
                layer_tangent[2][2] = layer_tangent[0][1]
        else:
            abort("unsupported LayeredShellSection layer kind")

        stress_out[0] += layer_stress[0] * weight
        stress_out[1] += layer_stress[1] * weight
        stress_out[2] += layer_stress[2] * weight
        stress_out[3] += z * layer_stress[0] * weight
        stress_out[4] += z * layer_stress[1] * weight
        stress_out[5] += z * layer_stress[2] * weight
        stress_out[6] += layer_stress[4] * weight
        stress_out[7] += layer_stress[3] * weight

        tangent_out[0][0] += layer_tangent[0][0] * weight
        tangent_out[0][1] += layer_tangent[0][1] * weight
        tangent_out[0][2] += layer_tangent[0][2] * weight
        tangent_out[0][3] -= z * layer_tangent[0][0] * weight
        tangent_out[0][4] -= z * layer_tangent[0][1] * weight
        tangent_out[0][5] -= z * layer_tangent[0][2] * weight

        tangent_out[1][0] += layer_tangent[1][0] * weight
        tangent_out[1][1] += layer_tangent[1][1] * weight
        tangent_out[1][2] += layer_tangent[1][2] * weight
        tangent_out[1][3] -= z * layer_tangent[1][0] * weight
        tangent_out[1][4] -= z * layer_tangent[1][1] * weight
        tangent_out[1][5] -= z * layer_tangent[1][2] * weight

        tangent_out[2][0] += layer_tangent[2][0] * weight
        tangent_out[2][1] += layer_tangent[2][1] * weight
        tangent_out[2][2] += layer_tangent[2][2] * weight
        tangent_out[2][3] -= z * layer_tangent[2][0] * weight
        tangent_out[2][4] -= z * layer_tangent[2][1] * weight
        tangent_out[2][5] -= z * layer_tangent[2][2] * weight

        tangent_out[3][0] -= z * layer_tangent[0][0] * weight
        tangent_out[3][1] -= z * layer_tangent[0][1] * weight
        tangent_out[3][2] -= z * layer_tangent[0][2] * weight
        tangent_out[3][3] += z * z * layer_tangent[0][0] * weight
        tangent_out[3][4] += z * z * layer_tangent[0][1] * weight
        tangent_out[3][5] += z * z * layer_tangent[0][2] * weight

        tangent_out[4][0] -= z * layer_tangent[1][0] * weight
        tangent_out[4][1] -= z * layer_tangent[1][1] * weight
        tangent_out[4][2] -= z * layer_tangent[1][2] * weight
        tangent_out[4][3] += z * z * layer_tangent[1][0] * weight
        tangent_out[4][4] += z * z * layer_tangent[1][1] * weight
        tangent_out[4][5] += z * z * layer_tangent[1][2] * weight

        tangent_out[5][0] -= z * layer_tangent[2][0] * weight
        tangent_out[5][1] -= z * layer_tangent[2][1] * weight
        tangent_out[5][2] -= z * layer_tangent[2][2] * weight
        tangent_out[5][3] += z * z * layer_tangent[2][0] * weight
        tangent_out[5][4] += z * z * layer_tangent[2][1] * weight
        tangent_out[5][5] += z * z * layer_tangent[2][2] * weight

        tangent_out[6][6] += layer_tangent[4][4] * weight
        tangent_out[7][7] += layer_tangent[3][3] * weight
    for i in range(8):
        for j in range(i):
            tangent_out[i][j] = tangent_out[j][i]
