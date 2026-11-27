from sections.plate_fiber import elastic_plate_fiber_base
from collections import InlineArray, List
from elements.utils import _zero_matrix
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

    fn __init__(out self, definition: Int, angle: Float64, weight: Float64, z: Float64):
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


@fieldwise_init
struct ShellConcreteLayer(ImplicitlyCopyable, Movable):
    """One ConcreteS layer. Its elastic transverse shear is in `elastic_tangent`."""

    var definition: ConcreteSDef
    var z: Float64
    var weight: Float64


struct LayeredShellSectionDef(Defaultable, ImplicitlyCopyable, Movable):
    """Layer data shared by every Gauss point that uses the section.

    Elastic layers are summed into `elastic_tangent` at setup. Rebar and
    ConcreteS layers keep one state per Gauss point: rebar states in the global
    uniaxial state list, ConcreteS states in the shell assembly cache.
    """

    var rho_area: Float64
    var elastic_tangent: InlineArray[Float64, 64]
    var rebars: List[ShellRebarLayer]
    var concretes: List[ShellConcreteLayer]
    # ShellMITC4 drilling modulus, fixed from the initial membrane tangent.
    var drill_stiffness: Float64

    fn __init__(out self):
        self.rho_area = 0.0
        self.elastic_tangent = InlineArray[Float64, 64](fill=0.0)
        self.rebars = []
        self.concretes = []
        self.drill_stiffness = 0.0

    fn __copyinit__(out self, existing: Self):
        self.rho_area = existing.rho_area
        self.elastic_tangent = existing.elastic_tangent.copy()
        self.rebars = existing.rebars.copy()
        self.concretes = existing.concretes.copy()
        self.drill_stiffness = existing.drill_stiffness


fn _add_elastic_plate(
    mut tangent: InlineArray[Float64, 64],
    E: Float64,
    nu: Float64,
    gmod: Float64,
    weight: Float64,
    z: Float64,
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
    var inverse_thickness = 1.0 / total_thickness
    for i in range(sec.shell_layer_count):
        var layer = shell_layers[sec.shell_layer_offset + i]
        if layer.material < 0 or layer.material >= len(materials_by_id):
            abort("LayeredShellSection layer material not found")
        ref mat = materials_by_id[layer.material]
        var weight = layer.thickness
        curr_loc += weight
        # Same operation order as OpenSees LayeredShellFiberSection.
        var z = 0.5 * total_thickness * (curr_loc * inverse_thickness - 1.0)
        curr_loc += weight

        if mat.type == "PlateFromPlaneStress":
            if mat.base_material < 0 or mat.base_material >= len(materials_by_id):
                abort("PlateFromPlaneStress base material not found")
            var base_mat = materials_by_id[mat.base_material]
            if base_mat.type == "ConcreteS":
                sec_def.concretes.append(
                    ShellConcreteLayer(
                        ConcreteSDef(
                            base_mat.E, base_mat.nu, base_mat.fc, base_mat.ft, base_mat.Es
                        ),
                        z,
                        weight,
                    )
                )
                # The plate wrapper supplies the two elastic transverse shear terms.
                sec_def.elastic_tangent[54] += mat.gmod * weight
                sec_def.elastic_tangent[63] += mat.gmod * weight
                continue
            if base_mat.type != "ElasticIsotropic":
                abort("PlateFromPlaneStress requires ElasticIsotropic or ConcreteS")
            _add_elastic_plate(
                sec_def.elastic_tangent, base_mat.E, base_mat.nu, mat.gmod, weight, z
            )
            sec_def.rho_area += base_mat.rho * weight
            continue

        if mat.type == "ElasticIsotropic" or mat.type == "PlateFiber":
            var base_mat = elastic_plate_fiber_base(layer.material, materials_by_id)
            _add_elastic_plate(
                sec_def.elastic_tangent,
                base_mat.E,
                base_mat.nu,
                base_mat.E / (2.0 * (1.0 + base_mat.nu)),
                weight,
                z,
            )
            sec_def.rho_area += base_mat.rho * weight
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
            sec_def.rebars.append(
                ShellRebarLayer(uniaxial_def_by_id[mat.base_material], mat.angle, weight, z)
            )
            sec_def.rho_area += base_mat.rho * weight
            continue

        abort("unsupported LayeredShellSection material: " + mat.type)

    defs.append(sec_def^)


fn _initial_drill_stiffness(
    sec_def: LayeredShellSectionDef, uniaxial_defs: List[UniMaterialDef]
) -> Float64:
    # ShellMITC4 fixes the drilling modulus at the smallest eigenvalue of
    # the initial membrane tangent.
    var membrane = _zero_matrix(3, 3)
    for p in range(3):
        for q in range(3):
            membrane[p][q] = sec_def.elastic_tangent[p * 8 + q]
    for i in range(len(sec_def.concretes)):
        ref layer = sec_def.concretes[i]
        for p in range(3):
            for q in range(3):
                membrane[p][q] += layer.definition.initial_tangent[3 * p + q] * layer.weight
    for i in range(len(sec_def.rebars)):
        ref layer = sec_def.rebars[i]
        # OpenSees LayeredShell's initial tangent reads each fresh layer's
        # current tangent.
        var weighted = UniMaterialState(uniaxial_defs[layer.definition]).tangent_t * layer.weight
        if layer.axis >= 0:
            membrane[layer.axis][layer.axis] += weighted
        else:
            for p in range(3):
                for q in range(3):
                    membrane[p][q] += weighted * layer.direction[p] * layer.direction[q]

    var eigenvalues: List[Float64] = []
    var eigenvectors: List[List[Float64]] = []
    jacobi_symmetric_eigen(membrane, eigenvalues, eigenvectors)
    return min(eigenvalues[0], min(eigenvalues[1], eigenvalues[2]))


fn layered_shell_runtime_alloc_instances(
    mut defs: List[LayeredShellSectionDef],
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
        for slot in range(len(sec_def.rebars)):
            var def_index = sec_def.rebars[slot].definition
            if def_index < 0 or def_index >= len(uniaxial_defs):
                abort("LayeredShellSection rebar material definition out of range")
        if instance_counts[s] > 0:
            sec_def.drill_stiffness = _initial_drill_stiffness(sec_def, uniaxial_defs)
            if len(sec_def.concretes) > 0:
                used_nonelastic = True

        for _ in range(instance_counts[s]):
            for slot in range(len(sec_def.rebars)):
                var def_index = sec_def.rebars[slot].definition
                var mat_def = uniaxial_defs[def_index]
                uniaxial_states.append(UniMaterialState(mat_def))
                uniaxial_state_defs.append(def_index)
                if not uni_mat_is_elastic(mat_def):
                    used_nonelastic = True
    return used_nonelastic


fn layered_shell_set_trial_from_offset[
    tangent: Bool
](
    sec_def: LayeredShellSectionDef,
    definitions: UnsafePointer[UniMaterialDef],
    states: UnsafePointer[UniMaterialState],
    concrete_states: UnsafePointer[ConcreteSState],
    strain: UnsafePointer[Float64],
    stress: UnsafePointer[Float64],
    d: UnsafePointer[Float64],
):
    """Update one Gauss point with caller-owned 8, 8, and 64-entry buffers.

    The state pointers address this point's first rebar state and first
    ConcreteS state. Model setup validates and assigns disjoint slots.
    Residual-only calls leave d untouched.
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

    for slot in range(len(sec_def.concretes)):
        ref layer = sec_def.concretes.unsafe_ptr()[slot]
        var z = layer.z
        var layer_stress = PlaneStressVector(fill=0.0)
        var layer_tangent = PlaneStressTangent(fill=0.0)
        concrete_s_set_trial_strain(
            layer.definition,
            concrete_states[slot],
            strain[0] - z * strain[3],
            strain[1] - z * strain[4],
            strain[2] - z * strain[5],
            layer_stress,
            layer_tangent,
        )

        @parameter
        for p in range(3):
            stress[p] += layer_stress[p] * layer.weight
            stress[p + 3] += (z * layer_stress[p]) * layer.weight

            @parameter
            if tangent:

                @parameter
                for q in range(3):
                    var weighted = layer_tangent[3 * p + q] * layer.weight
                    d[p * 8 + q] += weighted
                    d[p * 8 + q + 3] -= z * weighted
                    d[(p + 3) * 8 + q] += z * weighted
                    d[(p + 3) * 8 + q + 3] -= z * z * weighted

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
