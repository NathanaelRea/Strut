from collections import List
from os import abort
from pathlib import Path
from sys.arg import argv

from json_native import JsonDocument, JsonValueTag
from materials import UniMaterialDef, UniMaterialState
from sections import (
    FiberCell,
    FiberSection2dDef,
    FiberSection3dDef,
    append_fiber_section2d_from_input,
    append_fiber_section3d_from_input,
    fiber_section2d_commit,
    fiber_section2d_init_states,
    fiber_section2d_set_trial,
    fiber_section3d_commit,
    fiber_section3d_init_states,
    fiber_section3d_set_trial,
)
from solver.run_case.input_types import (
    FiberInput,
    FiberLayerInput,
    FiberPatchInput,
    MaterialInput,
    SectionInput,
)
from strut_io import load_json_native
from tag_types import UniMaterialTypeTag


fn arg_value(
    args: VariadicList[StringSlice[StaticConstantOrigin]], name: String
) -> String:
    for i in range(len(args) - 1):
        if String(args[i]) == name:
            return String(args[i + 1])
    return ""


fn _json_key(doc: JsonDocument, object_index: Int, key: StringSlice) raises -> Int:
    if object_index < 0:
        return -1
    if doc.node_tag(object_index) != JsonValueTag.Object:
        return -1
    return doc.object_find(object_index, key)


fn _json_has_value(doc: JsonDocument, node_index: Int) -> Bool:
    return node_index >= 0 and doc.node_tag(node_index) != JsonValueTag.Null


fn _json_require_key(
    doc: JsonDocument, object_index: Int, key: StringSlice
) raises -> Int:
    var node_index = _json_key(doc, object_index, key)
    if node_index < 0:
        abort("missing required key: " + String(key))
    return node_index


fn _json_expect_array_len(doc: JsonDocument, array_index: Int, field: String) -> Int:
    if doc.node_tag(array_index) != JsonValueTag.Array:
        abort(field + " must be an array")
    return doc.node_len(array_index)


fn _json_number_value(
    doc: JsonDocument, node_index: Int, field: String
) -> Float64:
    if doc.node_tag(node_index) != JsonValueTag.Number:
        abort(field + " must be a number")
    return doc.node_number(node_index)


fn _json_int_value(doc: JsonDocument, node_index: Int, field: String) -> Int:
    return Int(_json_number_value(doc, node_index, field))


fn _json_string_value(doc: JsonDocument, node_index: Int, field: String) -> String:
    if doc.node_tag(node_index) != JsonValueTag.String:
        abort(field + " must be a string")
    return doc.node_text(node_index)


fn _json_get_int(
    doc: JsonDocument, object_index: Int, key: StringSlice, default: Int
) raises -> Int:
    var node_index = _json_key(doc, object_index, key)
    if node_index < 0:
        return default
    return _json_int_value(doc, node_index, String(key))


fn _json_get_float(
    doc: JsonDocument, object_index: Int, key: StringSlice, default: Float64
) raises -> Float64:
    var node_index = _json_key(doc, object_index, key)
    if node_index < 0:
        return default
    return _json_number_value(doc, node_index, String(key))


fn _json_get_string(
    doc: JsonDocument, object_index: Int, key: StringSlice, default: String
) raises -> String:
    var node_index = _json_key(doc, object_index, key)
    if node_index < 0:
        return default
    return _json_string_value(doc, node_index, String(key))


fn _parse_material_input(doc: JsonDocument, material_index: Int) raises -> MaterialInput:
    var parsed = MaterialInput()
    var params_index = _json_key(doc, material_index, "params")
    parsed.id = _json_get_int(doc, material_index, "id", -1)
    parsed.type = _json_get_string(doc, material_index, "type", "")
    parsed.E = _json_get_float(doc, params_index, "E", 0.0)
    parsed.Fy = _json_get_float(doc, params_index, "Fy", 0.0)
    parsed.E0 = _json_get_float(doc, params_index, "E0", 0.0)
    parsed.b = _json_get_float(doc, params_index, "b", 0.0)
    parsed.fpc = _json_get_float(doc, params_index, "fpc", 0.0)
    parsed.epsc0 = _json_get_float(doc, params_index, "epsc0", 0.0)
    parsed.fpcu = _json_get_float(doc, params_index, "fpcu", 0.0)
    parsed.epscu = _json_get_float(doc, params_index, "epscu", 0.0)
    parsed.has_r0 = _json_key(doc, params_index, "R0") >= 0
    parsed.has_cr1 = _json_key(doc, params_index, "cR1") >= 0
    parsed.has_cr2 = _json_key(doc, params_index, "cR2") >= 0
    if parsed.has_r0:
        parsed.R0 = _json_get_float(doc, params_index, "R0", 0.0)
    if parsed.has_cr1:
        parsed.cR1 = _json_get_float(doc, params_index, "cR1", 0.0)
    if parsed.has_cr2:
        parsed.cR2 = _json_get_float(doc, params_index, "cR2", 0.0)
    parsed.has_a1 = _json_key(doc, params_index, "a1") >= 0
    parsed.has_a2 = _json_key(doc, params_index, "a2") >= 0
    parsed.has_a3 = _json_key(doc, params_index, "a3") >= 0
    parsed.has_a4 = _json_key(doc, params_index, "a4") >= 0
    if parsed.has_a1:
        parsed.a1 = _json_get_float(doc, params_index, "a1", 0.0)
    if parsed.has_a2:
        parsed.a2 = _json_get_float(doc, params_index, "a2", 0.0)
    if parsed.has_a3:
        parsed.a3 = _json_get_float(doc, params_index, "a3", 0.0)
    if parsed.has_a4:
        parsed.a4 = _json_get_float(doc, params_index, "a4", 0.0)
    parsed.has_siginit = _json_key(doc, params_index, "sigInit") >= 0
    if parsed.has_siginit:
        parsed.sigInit = _json_get_float(doc, params_index, "sigInit", 0.0)
    parsed.has_rat = _json_key(doc, params_index, "rat") >= 0
    parsed.has_ft = _json_key(doc, params_index, "ft") >= 0
    parsed.has_ets = _json_key(doc, params_index, "Ets") >= 0
    if parsed.has_rat:
        parsed.rat = _json_get_float(doc, params_index, "rat", 0.0)
    if parsed.has_ft:
        parsed.ft = _json_get_float(doc, params_index, "ft", 0.0)
    if parsed.has_ets:
        parsed.Ets = _json_get_float(doc, params_index, "Ets", 0.0)
    return parsed^


fn make_uniaxial_def(material: MaterialInput) -> UniMaterialDef:
    var mat_type = material.type
    if mat_type == "Elastic":
        var E = material.E
        if E <= 0.0:
            abort("Elastic material E must be > 0")
        return UniMaterialDef(UniMaterialTypeTag.Elastic, E, 0.0, 0.0, 0.0)
    if mat_type == "Steel01":
        var Fy = material.Fy
        var E0 = material.E0
        var b = material.b
        if Fy <= 0.0:
            abort("Steel01 Fy must be > 0")
        if E0 <= 0.0:
            abort("Steel01 E0 must be > 0")
        if b < 0.0 or b >= 1.0:
            abort("Steel01 b must be in [0, 1)")
        return UniMaterialDef(UniMaterialTypeTag.Steel01, Fy, E0, b, 0.0)
    if mat_type == "Concrete01":
        var fpc = material.fpc
        var epsc0 = material.epsc0
        var fpcu = material.fpcu
        var epscu = material.epscu
        if fpc > 0.0:
            fpc = -fpc
        if epsc0 > 0.0:
            epsc0 = -epsc0
        if fpcu > 0.0:
            fpcu = -fpcu
        if epscu > 0.0:
            epscu = -epscu
        return UniMaterialDef(UniMaterialTypeTag.Concrete01, fpc, epsc0, fpcu, epscu)
    if mat_type == "Steel02":
        var Fy = material.Fy
        var E0 = material.E0
        var b = material.b
        if Fy < 0.0:
            abort("Steel02 Fy must be >= 0")
        if E0 <= 0.0:
            abort("Steel02 E0 must be > 0")
        if b < 0.0 or b >= 1.0:
            abort("Steel02 b must be in [0, 1)")

        var R0 = 15.0
        var cR1 = 0.925
        var cR2 = 0.15
        var a1 = 0.0
        var a2 = 1.0
        var a3 = 0.0
        var a4 = 1.0
        var sig_init = 0.0
        if material.has_r0:
            R0 = material.R0
            cR1 = material.cR1
            cR2 = material.cR2
        if material.has_a1:
            a1 = material.a1
            a2 = material.a2
            a3 = material.a3
            a4 = material.a4
        if material.has_siginit:
            sig_init = material.sigInit
        return UniMaterialDef(
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
    if mat_type == "Concrete02":
        var fpc = material.fpc
        var epsc0 = material.epsc0
        var fpcu = material.fpcu
        var epscu = material.epscu
        if fpc > 0.0:
            fpc = -fpc
        if epsc0 > 0.0:
            epsc0 = -epsc0
        if fpcu > 0.0:
            fpcu = -fpcu
        if epscu > 0.0:
            epscu = -epscu
        var rat = 0.1
        var ft = 0.1 * fpc
        if ft < 0.0:
            ft = -ft
        var Ets = 0.1 * fpc / epsc0
        if material.has_rat:
            rat = material.rat
            ft = material.ft
            Ets = material.Ets
        return UniMaterialDef(
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

    abort("unsupported material type for section path: " + mat_type)
    return UniMaterialDef()


fn _parse_section_input(
    doc: JsonDocument,
    section_index: Int,
    mut section: SectionInput,
    mut patches: List[FiberPatchInput],
    mut layers: List[FiberLayerInput],
    mut fibers: List[FiberInput],
) raises:
    section = SectionInput(
        _json_get_int(doc, section_index, "id", -1),
        _json_get_string(doc, section_index, "type", ""),
    )
    var params_index = _json_require_key(doc, section_index, "params")
    patches = List[FiberPatchInput]()
    layers = List[FiberLayerInput]()
    fibers = List[FiberInput]()
    var patches_index = _json_key(doc, params_index, "patches")
    if _json_has_value(doc, patches_index):
        section.fiber_patch_count = _json_expect_array_len(
            doc, patches_index, "section patches"
        )
        for i in range(section.fiber_patch_count):
            var patch_index = doc.array_item(patches_index, i)
            var patch = FiberPatchInput()
            patch.type = _json_get_string(doc, patch_index, "type", "")
            if patch.type == "quad":
                patch.type = "quadr"
            patch.material = _json_get_int(doc, patch_index, "material", -1)
            patch.num_subdiv_y = _json_get_int(doc, patch_index, "num_subdiv_y", 0)
            patch.num_subdiv_z = _json_get_int(doc, patch_index, "num_subdiv_z", 0)
            patch.y_i = _json_get_float(doc, patch_index, "y_i", 0.0)
            patch.z_i = _json_get_float(doc, patch_index, "z_i", 0.0)
            patch.y_j = _json_get_float(doc, patch_index, "y_j", 0.0)
            patch.z_j = _json_get_float(doc, patch_index, "z_j", 0.0)
            patch.y_k = _json_get_float(doc, patch_index, "y_k", 0.0)
            patch.z_k = _json_get_float(doc, patch_index, "z_k", 0.0)
            patch.y_l = _json_get_float(doc, patch_index, "y_l", 0.0)
            patch.z_l = _json_get_float(doc, patch_index, "z_l", 0.0)
            patches.append(patch)
    var layers_index = _json_key(doc, params_index, "layers")
    if _json_has_value(doc, layers_index):
        section.fiber_layer_count = _json_expect_array_len(
            doc, layers_index, "section layers"
        )
        for i in range(section.fiber_layer_count):
            var layer_index = doc.array_item(layers_index, i)
            var layer = FiberLayerInput()
            layer.type = _json_get_string(doc, layer_index, "type", "")
            layer.material = _json_get_int(doc, layer_index, "material", -1)
            layer.num_bars = _json_get_int(doc, layer_index, "num_bars", 0)
            layer.bar_area = _json_get_float(doc, layer_index, "bar_area", 0.0)
            layer.y_start = _json_get_float(doc, layer_index, "y_start", 0.0)
            layer.z_start = _json_get_float(doc, layer_index, "z_start", 0.0)
            layer.y_end = _json_get_float(doc, layer_index, "y_end", 0.0)
            layer.z_end = _json_get_float(doc, layer_index, "z_end", 0.0)
            layers.append(layer)
    var fibers_index = _json_key(doc, params_index, "fibers")
    if _json_has_value(doc, fibers_index):
        section.fiber_count = _json_expect_array_len(doc, fibers_index, "section fibers")
        for i in range(section.fiber_count):
            var fiber_index = doc.array_item(fibers_index, i)
            var fiber = FiberInput()
            fiber.y = _json_get_float(doc, fiber_index, "y", 0.0)
            fiber.z = _json_get_float(doc, fiber_index, "z", 0.0)
            fiber.area = _json_get_float(doc, fiber_index, "area", 0.0)
            fiber.material = _json_get_int(doc, fiber_index, "material", -1)
            fibers.append(fiber)
    _ = section


def run_section_path():
    var args = argv()
    var input_path = arg_value(args, "--input")
    var output_path = arg_value(args, "--output")
    if input_path == "" or output_path == "":
        abort("usage: section_path.strut --input <json> --output <csv>")

    var doc = load_json_native(input_path)
    var root = doc.root_index
    var materials_index = _json_require_key(doc, root, "materials")
    var section_index = _json_require_key(doc, root, "section")
    var deformation_path = _json_require_key(doc, root, "deformation_path")

    var materials_by_id: List[MaterialInput] = []
    for i in range(_json_expect_array_len(doc, materials_index, "materials")):
        var material = _parse_material_input(doc, doc.array_item(materials_index, i))
        if material.id < 0:
            abort("section_path materials require id")
        if material.id >= len(materials_by_id):
            materials_by_id.resize(material.id + 1, MaterialInput())
        materials_by_id[material.id] = material

    var uniaxial_defs: List[UniMaterialDef] = []
    var uniaxial_def_by_id: List[Int] = []
    uniaxial_def_by_id.resize(len(materials_by_id), -1)
    for i in range(len(materials_by_id)):
        var material = materials_by_id[i]
        if material.id < 0:
            continue
        uniaxial_def_by_id[i] = len(uniaxial_defs)
        uniaxial_defs.append(make_uniaxial_def(material))

    var section = SectionInput(-1, "")
    var fiber_patches: List[FiberPatchInput] = []
    var fiber_layers: List[FiberLayerInput] = []
    var section_fibers: List[FiberInput] = []
    _parse_section_input(
        doc,
        section_index,
        section,
        fiber_patches,
        fiber_layers,
        section_fibers,
    )

    var out = String("")
    if section.type == "FiberSection2d":
        var defs: List[FiberSection2dDef] = []
        var fibers: List[FiberCell] = []
        append_fiber_section2d_from_input(
            section,
            fiber_patches,
            fiber_layers,
            section_fibers,
            uniaxial_def_by_id,
            uniaxial_defs,
            defs,
            fibers,
        )
        if len(defs) != 1:
            abort("section_path expects exactly one FiberSection2d")

        var uniaxial_states: List[UniMaterialState] = []
        var uniaxial_state_defs: List[Int] = []
        var section_uniaxial_offsets: List[Int] = []
        var section_uniaxial_counts: List[Int] = []
        _ = fiber_section2d_init_states(
            defs,
            fibers,
            uniaxial_defs,
            uniaxial_states,
            uniaxial_state_defs,
            section_uniaxial_offsets,
            section_uniaxial_counts,
        )

        out = String("eps0,kappa,N,Mz,k11,k12,k22\n")
        for i in range(_json_expect_array_len(doc, deformation_path, "deformation_path")):
            var step = doc.array_item(deformation_path, i)
            var eps0 = _json_get_float(doc, step, "eps0", 0.0)
            var kappa = _json_get_float(doc, step, "kappa", 0.0)
            var resp = fiber_section2d_set_trial(
                0,
                defs,
                section_uniaxial_offsets,
                section_uniaxial_counts,
                uniaxial_states,
                eps0,
                kappa,
            )
            out = (
                out
                + String(eps0)
                + ","
                + String(kappa)
                + ","
                + String(resp.axial_force)
                + ","
                + String(resp.moment_z)
                + ","
                + String(resp.k11)
                + ","
                + String(resp.k12)
                + ","
                + String(resp.k22)
                + "\n"
            )
            fiber_section2d_commit(
                0,
                defs,
                section_uniaxial_offsets,
                section_uniaxial_counts,
                uniaxial_states,
            )
    elif section.type == "FiberSection3d":
        var defs3d: List[FiberSection3dDef] = []
        var fibers3d: List[FiberCell] = []
        append_fiber_section3d_from_input(
            section,
            fiber_patches,
            fiber_layers,
            section_fibers,
            uniaxial_def_by_id,
            uniaxial_defs,
            defs3d,
            fibers3d,
        )
        if len(defs3d) != 1:
            abort("section_path expects exactly one FiberSection3d")

        var uniaxial_states3d: List[UniMaterialState] = []
        var uniaxial_state_defs3d: List[Int] = []
        var section_uniaxial_offsets3d: List[Int] = []
        var section_uniaxial_counts3d: List[Int] = []
        _ = fiber_section3d_init_states(
            defs3d,
            fibers3d,
            uniaxial_defs,
            uniaxial_states3d,
            uniaxial_state_defs3d,
            section_uniaxial_offsets3d,
            section_uniaxial_counts3d,
        )

        out = String("eps0,ky,kz,N,My,Mz,k11,k12,k13,k22,k23,k33\n")
        for i in range(_json_expect_array_len(doc, deformation_path, "deformation_path")):
            var step = doc.array_item(deformation_path, i)
            var eps0 = _json_get_float(doc, step, "eps0", 0.0)
            var ky = _json_get_float(
                doc,
                step,
                "kappa_y",
                _json_get_float(doc, step, "ky", 0.0),
            )
            var kz = _json_get_float(
                doc,
                step,
                "kappa_z",
                _json_get_float(
                    doc,
                    step,
                    "kz",
                    _json_get_float(doc, step, "kappa", 0.0),
                ),
            )
            var resp = fiber_section3d_set_trial(
                0,
                defs3d,
                section_uniaxial_offsets3d,
                section_uniaxial_counts3d,
                uniaxial_states3d,
                eps0,
                ky,
                kz,
            )
            out = (
                out
                + String(eps0)
                + ","
                + String(ky)
                + ","
                + String(kz)
                + ","
                + String(resp.axial_force)
                + ","
                + String(resp.moment_y)
                + ","
                + String(resp.moment_z)
                + ","
                + String(resp.k11)
                + ","
                + String(resp.k12)
                + ","
                + String(resp.k13)
                + ","
                + String(resp.k22)
                + ","
                + String(resp.k23)
                + ","
                + String(resp.k33)
                + "\n"
            )
            fiber_section3d_commit(
                0,
                section_uniaxial_offsets3d,
                section_uniaxial_counts3d,
                uniaxial_states3d,
            )
    else:
        abort("section_path requires FiberSection2d or FiberSection3d")

    Path(output_path).write_text(out)


fn main():
    try:
        run_section_path()
    except:
        abort("section_path failed")
