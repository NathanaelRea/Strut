from collections import InlineArray
from materials import (
    UniMaterialDef,
    UniMaterialState,
    uniaxial_commit,
    uniaxial_revert_trial,
    uniaxial_set_trial_strain,
)
from materials.uniaxial.min_max import StrainLimits
from sys.arg import argv
from tag_types import UniMaterialTypeTag
from testing import assert_equal


def main():
    var args = argv()
    var interval = atol(args[2])
    var param_count = atol(args[3])
    var params = InlineArray[Float64, 24](fill=0.0)
    for i in range(param_count):
        params[i] = atof(args[4 + i])
    var model = args[1]
    var tag = UniMaterialTypeTag.Hysteretic if model == "Hysteretic" else UniMaterialTypeTag.Bilin
    var material = UniMaterialDef(tag, params)
    if model == "ElasticPP":
        tag = UniMaterialTypeTag.ElasticPP
        material = UniMaterialDef(tag, params[0], params[1], params[2], params[3])
    elif model == "ElasticBilin":
        tag = UniMaterialTypeTag.ElasticBilin
        material = UniMaterialDef(
            tag,
            params[0],
            params[1],
            params[2],
            params[3],
            params[4],
            params[5],
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
    elif model == "MinMax":
        tag = UniMaterialTypeTag.Elastic
        material = UniMaterialDef(tag, params[0], 0.0, 0.0, 0.0)
        material.strain_limits = StrainLimits(params[1], params[2])
    elif model == "Ratchet" or model == "RatchetContinuous":
        tag = UniMaterialTypeTag.Ratchet
        material = UniMaterialDef(tag, params[0], params[1], params[2], params[3])
    var state = UniMaterialState(material)
    var first_strain = 4 + param_count
    for i in range(first_strain, len(args)):
        uniaxial_set_trial_strain(material, state, atof(args[i]))
        if (i - first_strain + 1) % interval == 0:
            uniaxial_commit(state)
        print("ROW", state.sig_t, state.tangent_t)

    var committed = UniMaterialState(material)
    uniaxial_set_trial_strain(material, committed, 0.045)
    uniaxial_commit(committed)
    uniaxial_set_trial_strain(material, committed, -0.032)
    var rejected_tangent = committed.tangent_t
    uniaxial_revert_trial(committed)
    if tag == UniMaterialTypeTag.ElasticBilin:
        assert_equal(committed.eps_t, -0.032)
        assert_equal(committed.tangent_t, rejected_tangent)
        return
    assert_equal(committed.eps_t, committed.eps_c)
    assert_equal(committed.sig_t, committed.sig_c)
    if tag == UniMaterialTypeTag.Hysteretic or tag == UniMaterialTypeTag.ElasticPP:
        assert_equal(committed.tangent_t, rejected_tangent)
    else:
        assert_equal(committed.tangent_t, committed.tangent_c)
