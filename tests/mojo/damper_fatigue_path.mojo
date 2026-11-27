from sys.arg import argv
from materials import (
    UniMaterialDef,
    UniMaterialState,
    uniaxial_commit,
    uniaxial_revert_trial,
    uniaxial_set_trial_strain_rate,
)
from materials.uniaxial.fatigue import FatigueParameters
from tag_types import UniMaterialTypeTag


def main():
    var args = argv()
    var material: UniMaterialDef
    var dt = 0.0
    if args[1] == "damper":
        dt = 0.05
        material = UniMaterialDef(
            UniMaterialTypeTag.ViscousDamper,
            120.0,
            8.0,
            0.55,
            atof(args[2]),
            atof(args[3]),
            1.0e-6,
            1.0e-10,
            15.0,
            0.0,
            0.0,
            0.0,
        )
    else:
        material = UniMaterialDef(
            UniMaterialTypeTag.Steel02,
            50.0,
            29000.0,
            0.003,
            15.0,
            0.925,
            0.15,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
        )
        material.fatigue = FatigueParameters(atof(args[2]), 0.095, -0.5, -1.0, atof(args[3]))
    var state = UniMaterialState(material)
    for i in range(4, len(args), 2):
        var strain = atof(args[i])
        var rate = atof(args[i + 1])
        uniaxial_set_trial_strain_rate(material, state, strain + 0.02, rate + 0.1, dt)
        uniaxial_revert_trial(state)
        uniaxial_set_trial_strain_rate(material, state, strain, rate, dt)
        uniaxial_commit(state)
        print("ROW", state.sig_t, state.tangent_t)
