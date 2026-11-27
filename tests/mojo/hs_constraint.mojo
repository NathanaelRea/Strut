from solver.run_case.analysis.hs_constraint import HSConstraint
from testing import assert_almost_equal, assert_true


fn main() raises:
    var arc = HSConstraint(0.2, 0.5, 0.25, 2.0)
    var direction = [0.3, 0.4]
    var load = [3.0, 4.0]
    var increment = arc.predict(direction, load, 0.0)
    assert_almost_equal(increment, 0.2 / (1.578125**0.5))
    assert_almost_equal(arc.predict(direction, load, -1.0), -increment)
    var correction = arc.correct(direction, [0.01, 0.0], [0.03, 0.04], load, 0.1)
    assert_true(Bool(correction))
    # The OpenSees update uses its own quadratic and angle-based root selection.
    assert_almost_equal(correction.value(), -0.00012365368655198217, atol=1e-10)
