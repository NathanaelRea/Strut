from collections import InlineArray, List
from math import cos, sin
from materials import (
    UniMaterialDef,
    UniMaterialState,
    uni_mat_is_elastic,
    uniaxial_set_trial_strain,
)
from os import abort
from solver.run_case.input_types import MaterialInput, SectionInput, ShellLayerInput


struct ShellRebarLayer(ImplicitlyCopyable, Movable):
    var definition: Int
    var axis: Int
    var z: Float64
    var weight: Float64
    var moment_weight: Float64
    var bending_weight: Float64
    var direction: InlineArray[Float64, 3]
    var resultant: InlineArray[Float64, 8]
    var tangent: InlineArray[Float64, 64]

    fn __init__(
        out self, definition: Int, angle: Float64, weight: Float64, z: Float64
    ):
        self.definition = definition
        self.z = z
        self.weight = weight
        self.moment_weight = z * weight
        self.bending_weight = z * z * weight
        self.direction = InlineArray[Float64, 3](fill=0.0)
        self.resultant = InlineArray[Float64, 8](fill=0.0)
        self.tangent = InlineArray[Float64, 64](fill=0.0)
        if angle == 0.0 or angle == 90.0:
            self.axis = 0 if angle == 0.0 else 1
        else:
            self.axis = -1
            var radians = angle * 3.141592653589793 / 180.0
            var c = cos(radians)
            var s = sin(radians)
            self.direction[0] = c * c
            self.direction[1] = s * s
            self.direction[2] = c * s
            for i in range(3):
                self.resultant[i] = self.direction[i] * weight
                self.resultant[i + 3] = self.direction[i] * self.moment_weight
                for j in range(3):
                    var c_ij = self.direction[i] * self.direction[j]
                    self.tangent[i * 8 + j] = c_ij * weight
                    self.tangent[i * 8 + j + 3] = -c_ij * self.moment_weight
                    self.tangent[(i + 3) * 8 + j] = c_ij * self.moment_weight
                    self.tangent[(i + 3) * 8 + j + 3] = -c_ij * self.bending_weight


struct LayeredShellSectionDef(Defaultable, Movable, ImplicitlyCopyable):
    var rho_area: Float64
    var elastic_tangent: InlineArray[Float64, 64]
    var rebars: List[ShellRebarLayer]

    fn __init__(out self):
        self.rho_area = 0.0
        self.elastic_tangent = InlineArray[Float64, 64](fill=0.0)
        self.rebars = []

    fn __copyinit__(out self, existing: Self):
        self.rho_area = existing.rho_area
        self.elastic_tangent = existing.elastic_tangent.copy()
        self.rebars = existing.rebars.copy()


fn _add_elastic_plate(
    mut tangent: InlineArray[Float64, 64],
    E: Float64, nu: Float64, gmod: Float64, weight: Float64, z: Float64,
):
    var factor = E / (1.0 - nu * nu)
    var plane = InlineArray[Float64, 9](fill=0.0)
    plane[0] = factor
    plane[1] = factor * nu
    plane[3] = plane[1]
    plane[4] = factor
    plane[8] = factor * (1.0 - nu) * 0.5
    # OpenSees LayeredShellFiberSection uses strain e - z * kappa and
    # moment +z * stress. The section tangent is intentionally unsymmetric.
    for i in range(3):
        for j in range(3):
            var c_ij = plane[i * 3 + j]
            tangent[i * 8 + j] += c_ij * weight
            tangent[i * 8 + j + 3] -= z * c_ij * weight
            tangent[(i + 3) * 8 + j] += z * c_ij * weight
            tangent[(i + 3) * 8 + j + 3] -= z * z * c_ij * weight
    tangent[54] += gmod * weight
    tangent[63] += gmod * weight


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
    var total_thickness = 0.0
    for i in range(sec.shell_layer_count):
        var layer = shell_layers[sec.shell_layer_offset + i]
        if layer.thickness <= 0.0:
            abort("LayeredShellSection layer thickness must be > 0")
        total_thickness += layer.thickness

    var curr_loc = 0.0
    for i in range(sec.shell_layer_count):
        var layer = shell_layers[sec.shell_layer_offset + i]
        if layer.material < 0 or layer.material >= len(materials_by_id):
            abort("LayeredShellSection layer material not found")
        ref mat = materials_by_id[layer.material]
        var weight = layer.thickness
        curr_loc += weight
        var z = curr_loc - 0.5 * total_thickness
        curr_loc += weight

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
                _add_elastic_plate(
                    sec_def.elastic_tangent, 2.0 * mat.gmod * (1.0 + nu),
                    nu, mat.gmod, weight, z,
                )
                sec_def.rho_area += base_mat.rho * weight
                continue
            if base_mat.type != "ElasticIsotropic":
                abort(
                    "LayeredShellSection currently supports PlateFromPlaneStress only over ElasticIsotropic"
                )
            _add_elastic_plate(
                sec_def.elastic_tangent, base_mat.E, base_mat.nu, mat.gmod, weight, z
            )
            sec_def.rho_area += base_mat.rho * weight
            continue

        if mat.type == "ElasticIsotropic":
            _add_elastic_plate(
                sec_def.elastic_tangent, mat.E, mat.nu,
                mat.E / (2.0 * (1.0 + mat.nu)), weight, z,
            )
            sec_def.rho_area += mat.rho * weight
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
            sec_def.rebars.append(ShellRebarLayer(
                uniaxial_def_by_id[mat.base_material], mat.angle, weight, z
            ))
            sec_def.rho_area += base_mat.rho * weight
            continue

        if mat.type == "PlaneStressUserMaterial":
            abort(
                "LayeredShellSection runtime does not support PlaneStressUserMaterial because the OpenSees PSUMAT implementation is not present in the reference repo"
            )
        abort("unsupported LayeredShellSection material: " + mat.type)

    defs.append(sec_def^)


fn layered_shell_runtime_alloc_instances(
    defs: List[LayeredShellSectionDef],
    instance_counts: List[Int],
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    mut uniaxial_state_defs: List[Int],
    mut section_uniaxial_offsets: List[Int],
) -> Bool:
    if len(instance_counts) != len(defs):
        abort("LayeredShellSection instance count mapping mismatch")

    section_uniaxial_offsets.resize(len(defs), 0)
    var used_nonelastic = False

    for s in range(len(defs)):
        ref sec_def = defs[s]
        section_uniaxial_offsets[s] = len(uniaxial_states)

        for _ in range(instance_counts[s]):
            for slot in range(len(sec_def.rebars)):
                var def_index = sec_def.rebars[slot].definition
                if def_index < 0 or def_index >= len(uniaxial_defs):
                    abort("LayeredShellSection rebar material definition out of range")
                var mat_def = uniaxial_defs[def_index]
                uniaxial_states.append(UniMaterialState(mat_def))
                uniaxial_state_defs.append(def_index)
                if not uni_mat_is_elastic(mat_def):
                    used_nonelastic = True
    return used_nonelastic


fn layered_shell_set_trial_from_offset[tangent: Bool](
    sec_def: LayeredShellSectionDef,
    definitions: UnsafePointer[UniMaterialDef],
    states: UnsafePointer[UniMaterialState],
    strain: UnsafePointer[Float64],
    stress: UnsafePointer[Float64],
    d: UnsafePointer[Float64],
    need_drilling_tangent: Bool,
) -> Float64:
    """Update one Gauss point with caller-owned 8, 8, and 64-entry buffers.

    The states pointer addresses this point's first rebar state. Model setup
    validates and assigns disjoint slots. Residual-only calls leave d untouched
    and can request D[2,2] alone for the first point's drilling coefficient.
    """
    var elastic = UnsafePointer(to=sec_def.elastic_tangent[0])
    var eps = strain.load[width=8]()
    @parameter
    for i in range(8):
        var row = elastic.load[width=8](i * 8)
        stress[i] = (row * eps).reduce_add()
        @parameter
        if tangent:
            d.store(i * 8, row)

    var drilling_tangent = sec_def.elastic_tangent[18]
    for slot in range(len(sec_def.rebars)):
        ref layer = sec_def.rebars.unsafe_ptr()[slot]
        ref state = states[slot]
        ref definition = definitions[layer.definition]
        var axis = layer.axis
        if axis >= 0:
            var moment = axis + 3
            var axial_strain = strain[axis] - layer.z * strain[moment]
            uniaxial_set_trial_strain(definition, state, axial_strain)
            stress[axis] += state.sig_t * layer.weight
            stress[moment] += state.sig_t * layer.moment_weight
            @parameter
            if tangent:
                var axial_tangent = state.tangent_t
                d[axis * 8 + axis] += axial_tangent * layer.weight
                d[axis * 8 + moment] -= axial_tangent * layer.moment_weight
                d[moment * 8 + axis] += axial_tangent * layer.moment_weight
                d[moment * 8 + moment] -= axial_tangent * layer.bending_weight
        else:
            var axial_strain = (
                (strain[0] - layer.z * strain[3]) * layer.direction[0]
                + (strain[1] - layer.z * strain[4]) * layer.direction[1]
                + (strain[2] - layer.z * strain[5]) * layer.direction[2]
            )
            uniaxial_set_trial_strain(definition, state, axial_strain)
            var resultant = UnsafePointer(to=layer.resultant[0]).load[width=8]()
            stress.store(0, stress.load[width=8]() + state.sig_t * resultant)
            @parameter
            if tangent:
                var contribution = UnsafePointer(to=layer.tangent[0])
                @parameter
                for i in range(6):
                    var row = d.load[width=8](i * 8)
                    d.store(i * 8, row + state.tangent_t * contribution.load[width=8](i * 8))
            if need_drilling_tangent:
                drilling_tangent += state.tangent_t * layer.tangent[18]
    return drilling_tangent
