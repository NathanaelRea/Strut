from elements.mvlem import MVLEMFiber, MVLEMGeometry
from elements.sfi_mvlem import sfi_mvlem_response
from materials import UniMaterialDef, UniMaterialState, uniaxial_commit, uniaxial_revert_trial
from materials.fsam import FSAMParameters, FSAMState
from materials.uniaxial.core import UniMaterialLeafState
from materials.uniaxial.parallel import make_parallel_def
from tag_types import UniMaterialTypeTag
from testing import assert_almost_equal, assert_equal


fn main() raises:
    var wall = MVLEMGeometry()
    wall.c = 0.4
    wall.fibers.append(MVLEMFiber(2.0, 1.0, 0.0, 1, -1))
    _ = wall.set_height(10.0)

    var steel = UniMaterialDef(UniMaterialTypeTag.Elastic, 29000.0, 0.0, 0.0, 0.0)
    var concrete = UniMaterialDef(
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
        0.0,
        0.0,
    )
    var monotonic = concrete
    monotonic.p10 = 1.0
    var composite = make_parallel_def(
        [steel, steel, monotonic, monotonic, concrete, concrete, concrete, concrete],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )
    var point = UniMaterialState(composite)
    assert_equal(len(point.components.value()), 8)
    point.fsam_components = composite.components
    var params = FSAMParameters(
        0.0, 0.01, 0.02, 0.4, 0.01, 3900.0, -0.002, 0.00013, 29000.0, 29000.0
    )
    point.fsam.append(FSAMState(params))
    var states = [point^]
    var dofs = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    var indices = [0]
    var zero = sfi_mvlem_response(wall, dofs, 0, indices, states, initial_tangent=True)
    assert_almost_equal(zero.tangent[6][6], (3900.0 + 290.0) * 5.0)
    assert_almost_equal(zero.tangent[4][4], (3900.0 + 580.0) * 0.2)
    assert_almost_equal(zero.tangent[0][0], 1950.0 * 0.2)
    assert_equal(zero.tangent[6][4], 0.0)

    dofs[3] = 0.006
    var uncracked = sfi_mvlem_response(wall, dofs, 0, indices, states)
    assert_equal(states[0].fsam[0].crack_count, 0)
    assert_almost_equal(uncracked.force[0], -uncracked.force[3])
    uniaxial_commit(states[0])
    assert_equal(states[0].fsam[0].crack_count, 1)
    var committed = sfi_mvlem_response(wall, dofs, 0, indices, states, committed=True)
    dofs[3] = -0.01
    _ = sfi_mvlem_response(wall, dofs, 0, indices, states)
    uniaxial_revert_trial(states[0])
    assert_almost_equal(states[0].fsam[0].trial.strain[2], 0.0006, atol=1e-15)
    var restored = sfi_mvlem_response(wall, dofs, 0, indices, states, committed=True)
    assert_equal(restored.force[0], committed.force[0])
