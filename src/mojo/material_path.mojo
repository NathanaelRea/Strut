from collections import List
from os import abort
from python import Python, PythonObject
from sys.arg import argv

from materials import (
    UniMaterialDef,
    UniMaterialState,
    uniaxial_commit,
    uniaxial_set_trial_strain,
)
from strut_io import py_len
from tag_types import UniMaterialTypeTag


fn arg_value(
    args: VariadicList[StringSlice[StaticConstantOrigin]], name: String
) -> String:
    for i in range(len(args) - 1):
        if String(args[i]) == name:
            return String(args[i + 1])
    return ""


fn load_json(path: String) raises -> PythonObject:
    var json = Python.import_module("json")
    var pathlib = Python.import_module("pathlib")
    var path_obj = pathlib.Path(path)
    var text = path_obj.read_text()
    return json.loads(text)


def make_uniaxial_def(material: PythonObject) -> UniMaterialDef:
    var mat_type = String(material["type"])
    var params = material["params"]
    if mat_type == "Elastic":
        return UniMaterialDef(UniMaterialTypeTag.Elastic, Float64(params["E"]), 0.0, 0.0, 0.0)
    if mat_type == "Steel01":
        return UniMaterialDef(
            UniMaterialTypeTag.Steel01,
            Float64(params["Fy"]),
            Float64(params["E0"]),
            Float64(params["b"]),
            0.0,
        )
    if mat_type == "Concrete01":
        var fpc = Float64(params["fpc"])
        var epsc0 = Float64(params["epsc0"])
        var fpcu = Float64(params["fpcu"])
        var epscu = Float64(params["epscu"])
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
        var Fy = Float64(params["Fy"])
        var E0 = Float64(params["E0"])
        var b = Float64(params["b"])
        var R0 = 15.0
        var cR1 = 0.925
        var cR2 = 0.15
        var a1 = 0.0
        var a2 = 1.0
        var a3 = 0.0
        var a4 = 1.0
        var sig_init = 0.0
        if params.__contains__("R0"):
            R0 = Float64(params["R0"])
            cR1 = Float64(params["cR1"])
            cR2 = Float64(params["cR2"])
        if params.__contains__("a1"):
            a1 = Float64(params["a1"])
            a2 = Float64(params["a2"])
            a3 = Float64(params["a3"])
            a4 = Float64(params["a4"])
        if params.__contains__("sigInit"):
            sig_init = Float64(params["sigInit"])
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
        var fpc = Float64(params["fpc"])
        var epsc0 = Float64(params["epsc0"])
        var fpcu = Float64(params["fpcu"])
        var epscu = Float64(params["epscu"])
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
        if params.__contains__("rat"):
            rat = Float64(params["rat"])
            ft = Float64(params["ft"])
            Ets = Float64(params["Ets"])
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

    var data = load_json(input_path)
    var material = data["material"]
    var strain_history = data["strain_history"]

    var mat_def = make_uniaxial_def(material)
    var state = UniMaterialState(mat_def)

    var out = String("strain,stress\n")
    for i in range(py_len(strain_history)):
        var strain = Float64(strain_history[i])
        uniaxial_set_trial_strain(mat_def, state, strain)
        out = out + String(strain) + "," + String(state.sig_t) + "\n"
        uniaxial_commit(state)

    var pathlib = Python.import_module("pathlib")
    pathlib.Path(output_path).write_text(out)


fn main():
    try:
        run_material_path()
    except:
        abort("material_path failed")
