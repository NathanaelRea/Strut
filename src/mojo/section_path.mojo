from collections import List
from os import abort
from python import Python, PythonObject
from sys.arg import argv

from materials import UniMaterialDef, UniMaterialState
from sections import (
    FiberCell,
    FiberSection2dDef,
    FiberSection3dDef,
    append_fiber_section2d_from_json,
    append_fiber_section3d_from_json,
    fiber_section2d_commit,
    fiber_section2d_init_states,
    fiber_section2d_set_trial,
    fiber_section3d_commit,
    fiber_section3d_init_states,
    fiber_section3d_set_trial,
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
        var E = Float64(params["E"])
        if E <= 0.0:
            abort("Elastic material E must be > 0")
        return UniMaterialDef(UniMaterialTypeTag.Elastic, E, 0.0, 0.0, 0.0)
    if mat_type == "Steel01":
        var Fy = Float64(params["Fy"])
        var E0 = Float64(params["E0"])
        var b = Float64(params["b"])
        if Fy <= 0.0:
            abort("Steel01 Fy must be > 0")
        if E0 <= 0.0:
            abort("Steel01 E0 must be > 0")
        if b < 0.0 or b >= 1.0:
            abort("Steel01 b must be in [0, 1)")
        return UniMaterialDef(UniMaterialTypeTag.Steel01, Fy, E0, b, 0.0)
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
        if Fy <= 0.0:
            abort("Steel02 Fy must be > 0")
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

    abort("unsupported material type for section path: " + mat_type)
    return UniMaterialDef()


def run_section_path():
    var args = argv()
    var input_path = arg_value(args, "--input")
    var output_path = arg_value(args, "--output")
    if input_path == "" or output_path == "":
        abort("usage: section_path.strut --input <json> --output <csv>")

    var data = load_json(input_path)
    var materials = data["materials"]
    var section = data["section"]
    var deformation_path = data["deformation_path"]

    var materials_by_id: List[PythonObject] = []
    materials_by_id.resize(0, None)
    for i in range(py_len(materials)):
        var mat = materials[i]
        var mid = Int(mat["id"])
        if mid >= len(materials_by_id):
            materials_by_id.resize(mid + 1, None)
        materials_by_id[mid] = mat

    var uniaxial_defs: List[UniMaterialDef] = []
    var uniaxial_def_by_id: List[Int] = []
    uniaxial_def_by_id.resize(len(materials_by_id), -1)
    for i in range(py_len(materials)):
        var mat = materials[i]
        var mid = Int(mat["id"])
        if mid >= len(uniaxial_def_by_id):
            uniaxial_def_by_id.resize(mid + 1, -1)
        var mat_def = make_uniaxial_def(mat)
        uniaxial_def_by_id[mid] = len(uniaxial_defs)
        uniaxial_defs.append(mat_def)

    var sec_type = String(section["type"])
    var out = String("")
    if sec_type == "FiberSection2d":
        var defs: List[FiberSection2dDef] = []
        var fibers: List[FiberCell] = []
        append_fiber_section2d_from_json(section, uniaxial_def_by_id, defs, fibers)
        if len(defs) != 1:
            abort("section_path expects exactly one FiberSection2d")

        var uniaxial_states: List[UniMaterialState] = []
        var uniaxial_state_defs: List[Int] = []
        var section_uniaxial_offsets: List[Int] = []
        var section_uniaxial_counts: List[Int] = []
        var section_uniaxial_state_ids: List[Int] = []
        _ = fiber_section2d_init_states(
            defs,
            fibers,
            uniaxial_defs,
            uniaxial_states,
            uniaxial_state_defs,
            section_uniaxial_offsets,
            section_uniaxial_counts,
            section_uniaxial_state_ids,
        )

        out = String("eps0,kappa,N,Mz,k11,k12,k22\n")
        for i in range(py_len(deformation_path)):
            var step = deformation_path[i]
            var eps0 = Float64(step["eps0"])
            var kappa = Float64(step["kappa"])
            var resp = fiber_section2d_set_trial(
                0,
                defs,
                fibers,
                uniaxial_defs,
                section_uniaxial_offsets,
                section_uniaxial_counts,
                section_uniaxial_state_ids,
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
                section_uniaxial_offsets,
                section_uniaxial_counts,
                section_uniaxial_state_ids,
                uniaxial_states,
            )
    elif sec_type == "FiberSection3d":
        var defs3d: List[FiberSection3dDef] = []
        var fibers3d: List[FiberCell] = []
        append_fiber_section3d_from_json(
            section, uniaxial_def_by_id, defs3d, fibers3d
        )
        if len(defs3d) != 1:
            abort("section_path expects exactly one FiberSection3d")

        var uniaxial_states3d: List[UniMaterialState] = []
        var uniaxial_state_defs3d: List[Int] = []
        var section_uniaxial_offsets3d: List[Int] = []
        var section_uniaxial_counts3d: List[Int] = []
        var section_uniaxial_state_ids3d: List[Int] = []
        _ = fiber_section3d_init_states(
            defs3d,
            fibers3d,
            uniaxial_defs,
            uniaxial_states3d,
            uniaxial_state_defs3d,
            section_uniaxial_offsets3d,
            section_uniaxial_counts3d,
            section_uniaxial_state_ids3d,
        )

        out = String("eps0,ky,kz,N,My,Mz,k11,k12,k13,k22,k23,k33\n")
        for i in range(py_len(deformation_path)):
            var step = deformation_path[i]
            var eps0 = Float64(step["eps0"])
            var ky = Float64(step.get("kappa_y", step.get("ky", 0.0)))
            var kz = Float64(
                step.get("kappa_z", step.get("kz", step.get("kappa", 0.0)))
            )
            var resp = fiber_section3d_set_trial(
                0,
                defs3d,
                fibers3d,
                uniaxial_defs,
                section_uniaxial_offsets3d,
                section_uniaxial_counts3d,
                section_uniaxial_state_ids3d,
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
                section_uniaxial_state_ids3d,
                uniaxial_states3d,
            )
    else:
        abort("section_path requires FiberSection2d or FiberSection3d")

    var pathlib = Python.import_module("pathlib")
    pathlib.Path(output_path).write_text(out)


fn main():
    try:
        run_section_path()
    except:
        abort("section_path failed")
