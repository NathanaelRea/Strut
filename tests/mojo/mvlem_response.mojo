from elements.mvlem import MVLEMFiber, MVLEMGeometry, mvlem_response
from materials import UniMaterialDef, UniMaterialState
from testing import assert_almost_equal, assert_equal
from tag_types import UniMaterialTypeTag


fn main() raises:
    var wall = MVLEMGeometry()
    wall.c = 0.4
    wall.fibers.append(MVLEMFiber(2.0, 1.0, 0.25, 1, 2))
    wall.fibers.append(MVLEMFiber(4.0, 1.0, 0.5, 1, 2))
    assert_equal(wall.set_height(10.0), 6.0)

    var defs = [
        UniMaterialDef(UniMaterialTypeTag.Elastic, 1000.0, 0.0, 0.0, 0.0),
        UniMaterialDef(UniMaterialTypeTag.Elastic, 2000.0, 0.0, 0.0, 0.0),
        UniMaterialDef(UniMaterialTypeTag.Elastic, 50.0, 0.0, 0.0, 0.0),
    ]
    var state_defs = [0, 0, 1, 1, 2]
    var state_ids = [0, 1, 2, 3, 4]
    var states: List[UniMaterialState] = []
    for material in state_defs:
        states.append(UniMaterialState(defs[material]))

    var u = [0.0, 0.0, 0.0, 0.15, -0.05, -0.025]
    var response = mvlem_response(wall, u, 0, state_ids, state_defs, defs, states)
    # Fiber centers are -2 and 1. The first fiber has zero axial strain.
    # At the second fiber, N = -0.0075 * 6000 = -45 and M = N * 1.
    # The end rotation exactly cancels the shear deformation.
    var expected = [0.0, 45.0, 45.0, 0.0, -45.0, -45.0]
    for i in range(6):
        assert_almost_equal(response.force[i], expected[i], atol=1.0e-12)

    # The exact elastic tangent at every DOF follows from finite differences.
    var delta = 1.0e-7
    for j in range(6):
        u[j] += delta
        var shifted = mvlem_response(wall, u, 0, state_ids, state_defs, defs, states)
        u[j] -= delta
        for i in range(6):
            assert_almost_equal(
                (shifted.force[i] - response.force[i]) / delta,
                response.tangent[i][j],
                atol=1.0e-6,
            )
