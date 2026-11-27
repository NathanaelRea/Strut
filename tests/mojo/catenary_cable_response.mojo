from elements.catenary_cable import CableVector, CatenaryCable, catenary_cable_response
from testing import assert_almost_equal


fn main() raises:
    var cable = CatenaryCable(-1e-5, 3e7, 1.0, 50.0, 6.5e-6, 100.0, -1e-5 / 9.81, 1e-6, 20, 0)
    # First cable at the final displacement in OpenSees TestElement_Catenary.
    var chord = CableVector(23.586934399700692, 0.0, -47.174215524324554, 0.0)
    var response = catenary_cable_response(cable, chord, CableVector(0.0, 0.0, -1.0, 0.0))
    assert_almost_equal(response.start[0], -727111.0483651919, atol=0.02)
    assert_almost_equal(response.start[2], 1454257.785329691, atol=0.02)
    assert_almost_equal(response.start[2] + response.end[2], cable.length, atol=1e-9)
    var tangent = response.tangent()
    for i in range(6):
        for j in range(6):
            assert_almost_equal(tangent[i][j], tangent[j][i], atol=1e-6)
        assert_almost_equal(tangent[i][0] + tangent[i][3], 0.0, atol=1e-9)
