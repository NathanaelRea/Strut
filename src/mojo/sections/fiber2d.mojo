from collections import List
from os import abort
from python import PythonObject

from materials import (
    UniMaterialDef,
    UniMaterialState,
    uni_mat_is_elastic,
    uniaxial_commit,
    uniaxial_revert_trial,
    uniaxial_set_trial_strain,
)
from strut_io import py_len


struct FiberCell(Defaultable, Movable, ImplicitlyCopyable):
    var y: Float64
    var z: Float64
    var area: Float64
    var def_index: Int

    fn __init__(out self):
        self.y = 0.0
        self.z = 0.0
        self.area = 0.0
        self.def_index = -1

    fn __init__(
        out self, y: Float64, z: Float64, area: Float64, def_index: Int
    ):
        self.y = y
        self.z = z
        self.area = area
        self.def_index = def_index


struct FiberSection2dDef(Defaultable, Movable, ImplicitlyCopyable):
    var fiber_offset: Int
    var fiber_count: Int
    var y_bar: Float64

    fn __init__(out self):
        self.fiber_offset = 0
        self.fiber_count = 0
        self.y_bar = 0.0

    fn __init__(out self, fiber_offset: Int, fiber_count: Int, y_bar: Float64):
        self.fiber_offset = fiber_offset
        self.fiber_count = fiber_count
        self.y_bar = y_bar


struct FiberSection2dResponse(Defaultable, Movable, ImplicitlyCopyable):
    var axial_force: Float64
    var moment_z: Float64
    var k11: Float64
    var k12: Float64
    var k22: Float64

    fn __init__(out self):
        self.axial_force = 0.0
        self.moment_z = 0.0
        self.k11 = 0.0
        self.k12 = 0.0
        self.k22 = 0.0

    fn __init__(
        out self,
        axial_force: Float64,
        moment_z: Float64,
        k11: Float64,
        k12: Float64,
        k22: Float64,
    ):
        self.axial_force = axial_force
        self.moment_z = moment_z
        self.k11 = k11
        self.k12 = k12
        self.k22 = k22


fn _resolve_uniaxial_def_index(
    mat_id: Int, uniaxial_def_by_id: List[Int]
) -> Int:
    if mat_id < 0:
        abort("FiberSection2d material id must be >= 0")
    if mat_id >= len(uniaxial_def_by_id) or uniaxial_def_by_id[mat_id] < 0:
        abort("FiberSection2d requires uniaxial material for all fibers")
    return uniaxial_def_by_id[mat_id]


fn _append_rect_patch_cells(
    patch: PythonObject,
    uniaxial_def_by_id: List[Int],
    mut fibers: List[FiberCell],
    mut area_sum: Float64,
    mut qz_sum: Float64,
) raises:
    var mat_id = Int(patch["material"])
    var def_index = _resolve_uniaxial_def_index(mat_id, uniaxial_def_by_id)
    var ny = Int(patch["num_subdiv_y"])
    var nz = Int(patch["num_subdiv_z"])
    if ny <= 0 or nz <= 0:
        abort("FiberSection2d patch rect requires num_subdiv_y and num_subdiv_z > 0")

    var y_i = Float64(patch["y_i"])
    var z_i = Float64(patch["z_i"])
    var y_j = Float64(patch["y_j"])
    var z_j = Float64(patch["z_j"])

    var y_min = y_i
    var y_max = y_j
    if y_min > y_max:
        y_min = y_j
        y_max = y_i

    var z_min = z_i
    var z_max = z_j
    if z_min > z_max:
        z_min = z_j
        z_max = z_i

    var dy_total = y_max - y_min
    var dz_total = z_max - z_min
    if dy_total <= 0.0 or dz_total <= 0.0:
        abort("FiberSection2d patch rect must have non-zero side lengths")

    var dy = dy_total / Float64(ny)
    var dz = dz_total / Float64(nz)
    var area = dy * dz

    for iy in range(ny):
        var y = y_min + (Float64(iy) + 0.5) * dy
        for iz in range(nz):
            var z = z_min + (Float64(iz) + 0.5) * dz
            fibers.append(FiberCell(y, z, area, def_index))
            area_sum += area
            qz_sum += area * y


fn _append_quadr_patch_cells(
    patch: PythonObject,
    uniaxial_def_by_id: List[Int],
    mut fibers: List[FiberCell],
    mut area_sum: Float64,
    mut qz_sum: Float64,
) raises:
    var mat_id = Int(patch["material"])
    var def_index = _resolve_uniaxial_def_index(mat_id, uniaxial_def_by_id)
    var ny = Int(patch["num_subdiv_y"])
    var nz = Int(patch["num_subdiv_z"])
    if ny <= 0 or nz <= 0:
        abort("FiberSection2d patch quadr requires num_subdiv_y and num_subdiv_z > 0")

    var y_i = Float64(patch["y_i"])
    var z_i = Float64(patch["z_i"])
    var y_j = Float64(patch["y_j"])
    var z_j = Float64(patch["z_j"])
    var y_k = Float64(patch["y_k"])
    var z_k = Float64(patch["z_k"])
    var y_l = Float64(patch["y_l"])
    var z_l = Float64(patch["z_l"])

    for iy in range(ny):
        var u0 = Float64(iy) / Float64(ny)
        var u1 = Float64(iy + 1) / Float64(ny)
        var uc = 0.5 * (u0 + u1)
        for iz in range(nz):
            var v0 = Float64(iz) / Float64(nz)
            var v1 = Float64(iz + 1) / Float64(nz)
            var vc = 0.5 * (v0 + v1)

            var y00 = (1.0 - u0) * (1.0 - v0) * y_i + u0 * (1.0 - v0) * y_j + u0 * v0 * y_k + (1.0 - u0) * v0 * y_l
            var z00 = (1.0 - u0) * (1.0 - v0) * z_i + u0 * (1.0 - v0) * z_j + u0 * v0 * z_k + (1.0 - u0) * v0 * z_l
            var y10 = (1.0 - u1) * (1.0 - v0) * y_i + u1 * (1.0 - v0) * y_j + u1 * v0 * y_k + (1.0 - u1) * v0 * y_l
            var z10 = (1.0 - u1) * (1.0 - v0) * z_i + u1 * (1.0 - v0) * z_j + u1 * v0 * z_k + (1.0 - u1) * v0 * z_l
            var y11 = (1.0 - u1) * (1.0 - v1) * y_i + u1 * (1.0 - v1) * y_j + u1 * v1 * y_k + (1.0 - u1) * v1 * y_l
            var z11 = (1.0 - u1) * (1.0 - v1) * z_i + u1 * (1.0 - v1) * z_j + u1 * v1 * z_k + (1.0 - u1) * v1 * z_l
            var y01 = (1.0 - u0) * (1.0 - v1) * y_i + u0 * (1.0 - v1) * y_j + u0 * v1 * y_k + (1.0 - u0) * v1 * y_l
            var z01 = (1.0 - u0) * (1.0 - v1) * z_i + u0 * (1.0 - v1) * z_j + u0 * v1 * z_k + (1.0 - u0) * v1 * z_l

            var twice_area = (
                y00 * z10
                + y10 * z11
                + y11 * z01
                + y01 * z00
                - z00 * y10
                - z10 * y11
                - z11 * y01
                - z01 * y00
            )
            var area = abs(twice_area) * 0.5
            if area <= 0.0:
                abort("FiberSection2d patch quadr generated zero-area cell")

            var yc = (1.0 - uc) * (1.0 - vc) * y_i + uc * (1.0 - vc) * y_j + uc * vc * y_k + (1.0 - uc) * vc * y_l
            var zc = (1.0 - uc) * (1.0 - vc) * z_i + uc * (1.0 - vc) * z_j + uc * vc * z_k + (1.0 - uc) * vc * z_l
            fibers.append(FiberCell(yc, zc, area, def_index))
            area_sum += area
            qz_sum += area * yc


fn _append_straight_layer_cells(
    layer: PythonObject,
    uniaxial_def_by_id: List[Int],
    mut fibers: List[FiberCell],
    mut area_sum: Float64,
    mut qz_sum: Float64,
) raises:
    var mat_id = Int(layer["material"])
    var def_index = _resolve_uniaxial_def_index(mat_id, uniaxial_def_by_id)
    var num_bars = Int(layer["num_bars"])
    if num_bars <= 0:
        abort("FiberSection2d layer straight requires num_bars > 0")
    var bar_area = Float64(layer["bar_area"])
    if bar_area <= 0.0:
        abort("FiberSection2d layer straight requires bar_area > 0")

    var y_start = Float64(layer["y_start"])
    var z_start = Float64(layer["z_start"])
    var y_end = Float64(layer["y_end"])
    var z_end = Float64(layer["z_end"])

    if num_bars == 1:
        var y = 0.5 * (y_start + y_end)
        var z = 0.5 * (z_start + z_end)
        fibers.append(FiberCell(y, z, bar_area, def_index))
        area_sum += bar_area
        qz_sum += bar_area * y
        return

    for i in range(num_bars):
        var t = Float64(i) / Float64(num_bars - 1)
        var y = y_start + (y_end - y_start) * t
        var z = z_start + (z_end - z_start) * t
        fibers.append(FiberCell(y, z, bar_area, def_index))
        area_sum += bar_area
        qz_sum += bar_area * y


fn append_fiber_section2d_from_json(
    sec: PythonObject,
    uniaxial_def_by_id: List[Int],
    mut defs: List[FiberSection2dDef],
    mut fibers: List[FiberCell],
) raises:
    if String(sec["type"]) != "FiberSection2d":
        abort("append_fiber_section2d_from_json requires FiberSection2d")

    var params = sec["params"]
    var patches = params.get("patches", [])
    var layers = params.get("layers", [])
    if py_len(patches) == 0 and py_len(layers) == 0:
        abort("FiberSection2d requires at least one patch or layer")

    var fiber_offset = len(fibers)
    var area_sum = 0.0
    var qz_sum = 0.0

    for i in range(py_len(patches)):
        var patch = patches[i]
        var patch_type = String(patch["type"])
        if patch_type == "quad":
            patch_type = "quadr"
        if patch_type == "rect":
            _append_rect_patch_cells(
                patch, uniaxial_def_by_id, fibers, area_sum, qz_sum
            )
        elif patch_type == "quadr":
            _append_quadr_patch_cells(
                patch, uniaxial_def_by_id, fibers, area_sum, qz_sum
            )
        else:
            abort("unsupported FiberSection2d patch type: " + patch_type)

    for i in range(py_len(layers)):
        var layer = layers[i]
        var layer_type = String(layer["type"])
        if layer_type == "straight":
            _append_straight_layer_cells(
                layer, uniaxial_def_by_id, fibers, area_sum, qz_sum
            )
        else:
            abort("unsupported FiberSection2d layer type: " + layer_type)

    var fiber_count = len(fibers) - fiber_offset
    if fiber_count <= 0:
        abort("FiberSection2d produced no fibers")
    if area_sum <= 0.0:
        abort("FiberSection2d total area must be > 0")
    var y_bar = qz_sum / area_sum

    defs.append(FiberSection2dDef(fiber_offset, fiber_count, y_bar))


fn fiber_section2d_init_states(
    defs: List[FiberSection2dDef],
    fibers: List[FiberCell],
    uniaxial_defs: List[UniMaterialDef],
    mut uniaxial_states: List[UniMaterialState],
    mut uniaxial_state_defs: List[Int],
    mut section_uniaxial_offsets: List[Int],
    mut section_uniaxial_counts: List[Int],
    mut section_uniaxial_state_ids: List[Int],
) -> Bool:
    section_uniaxial_offsets.resize(len(defs), 0)
    section_uniaxial_counts.resize(len(defs), 0)
    var used_nonelastic = False

    for s in range(len(defs)):
        var sec_def = defs[s]
        section_uniaxial_offsets[s] = len(section_uniaxial_state_ids)
        section_uniaxial_counts[s] = sec_def.fiber_count
        for i in range(sec_def.fiber_count):
            var cell = fibers[sec_def.fiber_offset + i]
            var def_index = cell.def_index
            if def_index < 0 or def_index >= len(uniaxial_defs):
                abort("FiberSection2d fiber material definition out of range")
            var mat_def = uniaxial_defs[def_index]
            var state_index = len(uniaxial_states)
            uniaxial_states.append(UniMaterialState(mat_def))
            uniaxial_state_defs.append(def_index)
            section_uniaxial_state_ids.append(state_index)
            if not uni_mat_is_elastic(mat_def):
                used_nonelastic = True

    return used_nonelastic


fn fiber_section2d_set_trial(
    section_index: Int,
    defs: List[FiberSection2dDef],
    fibers: List[FiberCell],
    uniaxial_defs: List[UniMaterialDef],
    section_uniaxial_offsets: List[Int],
    section_uniaxial_counts: List[Int],
    section_uniaxial_state_ids: List[Int],
    mut uniaxial_states: List[UniMaterialState],
    eps0: Float64,
    kappa: Float64,
) -> FiberSection2dResponse:
    if section_index < 0 or section_index >= len(defs):
        abort("FiberSection2d section index out of range")
    if section_index >= len(section_uniaxial_offsets) or section_index >= len(
        section_uniaxial_counts
    ):
        abort("FiberSection2d section state mapping missing")

    var sec_def = defs[section_index]
    var offset = section_uniaxial_offsets[section_index]
    var count = section_uniaxial_counts[section_index]
    if count != sec_def.fiber_count:
        abort("FiberSection2d section state count mismatch")
    if offset + count > len(section_uniaxial_state_ids):
        abort("FiberSection2d section state ids out of range")

    var axial_force = 0.0
    var moment_z = 0.0
    var k11 = 0.0
    var k12 = 0.0
    var k22 = 0.0

    for i in range(count):
        var cell = fibers[sec_def.fiber_offset + i]
        var y_rel = cell.y - sec_def.y_bar
        var eps = eps0 - y_rel * kappa

        var state_index = section_uniaxial_state_ids[offset + i]
        if state_index < 0 or state_index >= len(uniaxial_states):
            abort("FiberSection2d uniaxial state index out of range")
        var def_index = cell.def_index
        if def_index < 0 or def_index >= len(uniaxial_defs):
            abort("FiberSection2d uniaxial definition index out of range")

        var mat_def = uniaxial_defs[def_index]
        var state = uniaxial_states[state_index]
        uniaxial_set_trial_strain(mat_def, state, eps)
        uniaxial_states[state_index] = state

        var area = cell.area
        var fs = state.sig_t * area
        var ks = state.tangent_t * area
        axial_force += fs
        moment_z += -fs * y_rel
        k11 += ks
        k12 += -ks * y_rel
        k22 += ks * y_rel * y_rel

    return FiberSection2dResponse(axial_force, moment_z, k11, k12, k22)


fn fiber_section2d_commit(
    section_index: Int,
    section_uniaxial_offsets: List[Int],
    section_uniaxial_counts: List[Int],
    section_uniaxial_state_ids: List[Int],
    mut uniaxial_states: List[UniMaterialState],
):
    if section_index < 0 or section_index >= len(section_uniaxial_offsets):
        abort("FiberSection2d section index out of range")
    if section_index >= len(section_uniaxial_counts):
        abort("FiberSection2d section state mapping missing")

    var offset = section_uniaxial_offsets[section_index]
    var count = section_uniaxial_counts[section_index]
    if offset + count > len(section_uniaxial_state_ids):
        abort("FiberSection2d section state ids out of range")

    for i in range(count):
        var state_index = section_uniaxial_state_ids[offset + i]
        var state = uniaxial_states[state_index]
        uniaxial_commit(state)
        uniaxial_states[state_index] = state


fn fiber_section2d_revert_trial(
    section_index: Int,
    section_uniaxial_offsets: List[Int],
    section_uniaxial_counts: List[Int],
    section_uniaxial_state_ids: List[Int],
    mut uniaxial_states: List[UniMaterialState],
):
    if section_index < 0 or section_index >= len(section_uniaxial_offsets):
        abort("FiberSection2d section index out of range")
    if section_index >= len(section_uniaxial_counts):
        abort("FiberSection2d section state mapping missing")

    var offset = section_uniaxial_offsets[section_index]
    var count = section_uniaxial_counts[section_index]
    if offset + count > len(section_uniaxial_state_ids):
        abort("FiberSection2d section state ids out of range")

    for i in range(count):
        var state_index = section_uniaxial_state_ids[offset + i]
        var state = uniaxial_states[state_index]
        uniaxial_revert_trial(state)
        uniaxial_states[state_index] = state


fn fiber_section2d_commit_all(
    section_uniaxial_offsets: List[Int],
    section_uniaxial_counts: List[Int],
    section_uniaxial_state_ids: List[Int],
    mut uniaxial_states: List[UniMaterialState],
):
    for i in range(len(section_uniaxial_offsets)):
        fiber_section2d_commit(
            i,
            section_uniaxial_offsets,
            section_uniaxial_counts,
            section_uniaxial_state_ids,
            uniaxial_states,
        )


fn fiber_section2d_revert_trial_all(
    section_uniaxial_offsets: List[Int],
    section_uniaxial_counts: List[Int],
    section_uniaxial_state_ids: List[Int],
    mut uniaxial_states: List[UniMaterialState],
):
    for i in range(len(section_uniaxial_offsets)):
        fiber_section2d_revert_trial(
            i,
            section_uniaxial_offsets,
            section_uniaxial_counts,
            section_uniaxial_state_ids,
            uniaxial_states,
        )
