from sys.arg import argv
from testing import assert_almost_equal

from materials import (
    UniMaterialDef,
    UniMaterialState,
    uniaxial_commit,
    uniaxial_revert_trial,
    uniaxial_set_trial_strain,
)
from materials.uniaxial.parallel import make_parallel_def
from tag_types import UniMaterialTypeTag


fn main() raises:
    var args = argv()
    var ent = UniMaterialDef(UniMaterialTypeTag.ENT, 1500.0, 0.0, 1.0, 0.0)
    var softened = UniMaterialDef(UniMaterialTypeTag.ENT, 1500.0, 0.02, 8.0, 0.0)
    var elastic = UniMaterialDef(UniMaterialTypeTag.Elastic, 10.0, 0.0, 0.0, 0.0)
    var steel = UniMaterialDef(UniMaterialTypeTag.Steel01, 5.0, 2000.0, 0.02, 0.0)
    var spring = make_parallel_def([ent, elastic], [1.0, 1.0])
    var nested = make_parallel_def([spring, softened, steel, steel], [1.0, 0.75, 1.0, -0.25])
    var definitions = [ent, softened, spring, nested]
    for j in range(len(definitions)):
        ref definition = definitions[j]
        var state = UniMaterialState(definition)
        var untouched = state
        for i in range(1, len(args)):
            var strain = Float64(String(args[i]))
            uniaxial_set_trial_strain(definition, state, strain)
            print("ROW", state.sig_t, state.tangent_t)
            uniaxial_commit(state)
            var committed = state
            uniaxial_set_trial_strain(definition, state, 1.5 * strain - 0.01)
            uniaxial_revert_trial(state)
            assert_almost_equal(state.sig_t, committed.sig_t)
            assert_almost_equal(state.tangent_t, committed.tangent_t)
        uniaxial_set_trial_strain(definition, untouched, -0.002)
        var fresh = UniMaterialState(definition)
        uniaxial_set_trial_strain(definition, fresh, -0.002)
        assert_almost_equal(untouched.sig_t, fresh.sig_t)
        assert_almost_equal(untouched.tangent_t, fresh.tangent_t)
