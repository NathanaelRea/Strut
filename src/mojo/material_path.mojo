from os import abort
from pathlib import Path
from sys.arg import argv

from json_native import JsonDocument, JsonValueTag
from materials import (
    UniMaterialDef,
    UniMaterialState,
    uniaxial_commit,
    uniaxial_set_trial_strain,
)
from solver.run_case.input_types import MaterialInput
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


fn _json_get_float(
    doc: JsonDocument, object_index: Int, key: StringSlice, default: Float64
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


def make_uniaxial_def(material: MaterialInput) -> UniMaterialDef:
    var mat_type = material.type
    if mat_type == "Elastic":
        return UniMaterialDef(UniMaterialTypeTag.Elastic, material.E, 0.0, 0.0, 0.0)
    if mat_type == "Steel01":
        return UniMaterialDef(
            UniMaterialTypeTag.Steel01,
            material.Fy,
            material.E0,
            material.b,
            0.0,
        )
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
            material.Fy,
            material.E0,
            material.b,
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
    abort("unsupported material type for material path: " + mat_type)
    return UniMaterialDef()


def run_material_path():
    var args = argv()
    var input_path = arg_value(args, "--input")
    var output_path = arg_value(args, "--output")
    if input_path == "" or output_path == "":
        abort("usage: material_path.strut --input <json> --output <csv>")

    var doc = load_json_native(input_path)
    var root = doc.root_index
    var material_index = _json_key(doc, root, "material")
    if material_index < 0:
        abort("material_path requires material")
    var strain_history = _json_key(doc, root, "strain_history")
    if strain_history < 0 or doc.node_tag(strain_history) != JsonValueTag.Array:
        abort("material_path requires strain_history array")

    var material = _parse_material_input(doc, material_index)
    var mat_def = make_uniaxial_def(material)
    var state = UniMaterialState(mat_def)

    var out = String("strain,stress\n")
    for i in range(doc.node_len(strain_history)):
        var strain = _json_number_value(
            doc,
            doc.array_item(strain_history, i),
            "strain_history",
        )
        uniaxial_set_trial_strain(mat_def, state, strain)
        out = out + String(strain) + "," + String(state.sig_t) + "\n"
        uniaxial_commit(state)

    Path(output_path).write_text(out)


fn main():
    try:
        run_material_path()
    except:
        abort("material_path failed")
