from os import abort
from sys.arg import argv

from materials import (
    UniMaterialDef,
    UniMaterialState,
    uniaxial_commit,
    uniaxial_revert_trial,
    uniaxial_set_trial_strain,
)
from tag_types import UniMaterialTypeTag


def main():
    var args = argv()
    if len(args) < 8:
        abort("usage: py_tz_qz_path <type> <commit_interval> <p0> <p1> <p2> <p3> <p4> [eps|R ...]")
    var mat_name = String(args[1])
    var mat_type = UniMaterialTypeTag.PySimple1
    if mat_name == "PySimple1":
        mat_type = UniMaterialTypeTag.PySimple1
    elif mat_name == "TzSimple1":
        mat_type = UniMaterialTypeTag.TzSimple1
    elif mat_name == "QzSimple1":
        mat_type = UniMaterialTypeTag.QzSimple1
    else:
        abort("unsupported material: " + mat_name)

    var commit_interval = atol(args[2])
    var mat_def = UniMaterialDef(
        mat_type,
        atof(args[3]),
        atof(args[4]),
        atof(args[5]),
        atof(args[6]),
        atof(args[7]),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    var state = UniMaterialState(mat_def)
    var count = 1
    for i in range(8, len(args)):
        var tok = String(args[i])
        if tok == "R":
            uniaxial_revert_trial(state)
            count = 1
            continue
        uniaxial_set_trial_strain(mat_def, state, atof(args[i]))
        if count == commit_interval:
            uniaxial_commit(state)
            count = 1
        else:
            count += 1
        print("ROW", state.sig_t, state.tangent_t)
