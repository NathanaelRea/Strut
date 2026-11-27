from collections import InlineArray
from materials import UniMaterialDef, UniMaterialState, uniaxial_commit
from materials import uniaxial_revert_trial, uniaxial_set_trial_strain
from sys.arg import argv
from tag_types import UniMaterialTypeTag


fn main() raises:
    var args = argv()
    var kind = String(args[1])
    var commit_interval = Int(String(args[2]))
    var definition: UniMaterialDef
    if kind == "SteelMPF":
        definition = UniMaterialDef(
            UniMaterialTypeTag.SteelMPF,
            71.0,
            64.0,
            29000.0,
            0.012,
            0.017,
            20.0,
            0.925,
            0.15,
            0.03,
            1.2,
            0.02,
            1.3,
        )
    elif kind == "ModIMKPeakOriented":
        definition = UniMaterialDef(
            UniMaterialTypeTag.ModIMKPeakOriented,
            InlineArray[Float64, 24](
                15000.0,
                0.01,
                0.02,
                17.0,
                -14.0,
                0.3,
                0.4,
                0.5,
                0.6,
                1.0,
                1.0,
                1.0,
                1.0,
                0.006,
                0.008,
                0.02,
                0.03,
                0.2,
                0.3,
                0.04,
                0.05,
                1.0,
                0.9,
                2.0,
            ),
        )
    else:
        var gap_close = 1.0 if kind == "ConcreteCM-gap" else 0.0
        definition = UniMaterialDef(
            UniMaterialTypeTag.ConcreteCM,
            -5.6,
            -0.002,
            3900.0,
            7.0,
            1.1,
            0.5,
            0.00013,
            1.2,
            2.0,
            gap_close,
            0.0,
        )
    var state = UniMaterialState(definition)
    for i in range(3, len(args)):
        var strain = Float64(String(args[i]))
        uniaxial_set_trial_strain(definition, state, strain)
        print("ROW", state.sig_t, state.tangent_t)
        if (i - 2) % commit_interval == 0:
            uniaxial_commit(state)
        else:
            uniaxial_revert_trial(state)
