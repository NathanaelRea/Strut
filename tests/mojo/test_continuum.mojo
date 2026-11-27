from math import sqrt
from materials.drucker_prager import (
    DruckerPragerDefinition,
    DruckerPragerHistory,
    DruckerPragerVector,
    drucker_prager_trial,
)
from elements.continuum import (
    BRICK_R,
    BRICK_S,
    BRICK_T,
    QUAD_R,
    QUAD_S,
    ContinuumVector,
    continuum_dimension,
    continuum_nodes,
    isotropic_continuum_tangent,
    prepare_continuum,
)
from materials.elastic_cross_anisotropic import cross_anisotropic_tangent
from testing import assert_almost_equal, assert_equal, assert_true


fn test_continuum(kind: String) raises:
    var ndf = continuum_dimension(kind)
    var count = continuum_nodes(kind)
    var brick = ndf == 3 and kind != "quad3d"
    var coords: List[ContinuumVector] = []
    for a in range(count):
        var r = BRICK_R[a] if brick else QUAD_R[a]
        var s = BRICK_S[a] if brick else QUAD_S[a]
        var t = BRICK_T[a] if brick else -1
        coords.append(ContinuumVector(Float64(r + 1), Float64(s + 1), Float64(t + 1), 0.0))
    if kind == "quad3d":
        for a in range(count):
            coords[a] = ContinuumVector(coords[a][0], 3.0, coords[a][1], 0.0)

    var components = 6 if brick or kind == "bbarQuad" else 3
    var constitutive = isotropic_continuum_tangent(100.0, 0.25, components, True)
    var element = prepare_continuum(
        kind, coords, 0.5, 2.0, ContinuumVector(3.0, 5.0, 7.0, 0.0), 0.0, constitutive
    )
    var u: List[Float64] = []
    u.resize(ndf * count, 0.0)
    for a in range(count):
        u[ndf * a] = 0.01 * coords[a][0]
    for point in range(len(element.strains)):
        var strain = element.response(u, point, False)
        var stress = element.response(u, point, True)
        assert_almost_equal(strain[0], 0.01, atol=1.0e-12)
        assert_almost_equal(stress[0], 1.2, atol=1.0e-12)
        for i in range(1, components):
            assert_almost_equal(strain[i], 0.0, atol=1.0e-12)
    var force = element.resisting_force(u)
    var measure = 8.0 if brick else 2.0
    var resultant = 0.0
    var mass = 0.0
    var work = 0.0
    for a in range(count):
        var row = a * ndf
        resultant += force[row]
        work += u[row] * (force[row] + element.body_force[row])
        mass += element.lumped_mass[row]
        for b in range(count):
            var col = b * ndf
            assert_almost_equal(
                element.stiffness[row * element.dofs + col],
                element.stiffness[col * element.dofs + row],
                atol=1.0e-12,
            )
            if len(element.mass) > 0:
                mass += element.mass[row * element.dofs + col]
    assert_almost_equal(resultant, -3.0 * measure, atol=1.0e-12)
    assert_almost_equal(mass, 2.0 * measure, atol=1.0e-12)
    assert_almost_equal(work, 0.012 * measure, atol=1.0e-12)
    if kind == "quad3d":
        for i in range(1, element.dofs, 3):
            assert_equal(force[i], 0.0)
            assert_equal(element.lumped_mass[i], 0.0)
    if kind == "stdBrick" or kind == "bbarBrick":
        assert_almost_equal(element.mass[0], 16.0 / 27.0, atol=1.0e-12)
        assert_almost_equal(element.mass[3], 8.0 / 27.0, atol=1.0e-12)
    if kind == "quad8n":
        assert_almost_equal(element.lumped_mass[0], -1.0 / 3.0, atol=1.0e-12)
        assert_almost_equal(element.lumped_mass[8], 4.0 / 3.0, atol=1.0e-12)
    if kind == "SSPquad" or kind == "SSPbrick":
        for a in range(count):
            u[ndf * a] = Float64(QUAD_R[a % 4] * QUAD_S[a % 4])
        force = element.resisting_force(u)
        work = 0.0
        for i in range(element.dofs):
            work += u[i] * (force[i] + element.body_force[i])
        assert_true(work > 0.0)


fn main() raises:
    test_drucker_prager_history()
    test_sspbrick_warped_mass()
    for kind in [
        "stdBrick",
        "bbarBrick",
        "Brick8N",
        "Brick27N",
        "SSPbrick",
        "SSPquad",
        "enhancedQuad",
        "bbarQuad",
        "quad8n",
        "quad9n",
        "quad3d",
    ]:
        test_continuum(kind)
    var isotropic = isotropic_continuum_tangent(100.0, 0.25, 6, True)
    var anisotropic = cross_anisotropic_tangent(100.0, 100.0, 0.25, 0.25, 40.0)
    for i in range(36):
        assert_almost_equal(isotropic[i], anisotropic[i], atol=1.0e-12)


fn test_drucker_prager_history() raises:
    var material = DruckerPragerDefinition(
        27777.78, 9259.26, 5.0, 0.398, 0.398, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
    )
    var strain = DruckerPragerVector(fill=0.0)
    strain[3] = 0.002
    var virgin = DruckerPragerHistory()
    var trial = drucker_prager_trial(material, virgin, strain)
    var gamma = (
        sqrt(2.0) * material.shear * strain[3] - sqrt(2.0 / 3.0) * material.yield_stress
    ) / (2.0 * material.shear + 9.0 * material.bulk * material.friction * material.dilation)
    assert_almost_equal(trial.history.plastic[3], sqrt(2.0) * gamma, atol=1.0e-12)
    assert_almost_equal(
        trial.stress[3], material.shear * (strain[3] - sqrt(2.0) * gamma), atol=1.0e-12
    )
    var perturbation = strain
    perturbation[3] -= 1.0e-7
    var unloaded = drucker_prager_trial(material, trial.history, perturbation)
    var retried = drucker_prager_trial(material, virgin, perturbation)
    assert_equal(unloaded.history.plastic[3], trial.history.plastic[3])
    assert_true(retried.history.plastic[3] < unloaded.history.plastic[3])
    assert_almost_equal(
        unloaded.stress[3] - trial.stress[3], -material.shear * 1.0e-7, atol=1.0e-10
    )


fn test_sspbrick_warped_mass() raises:
    var coords: List[ContinuumVector] = []
    for a in range(8):
        var r = Float64(BRICK_R[a])
        var s = Float64(BRICK_S[a])
        var t = Float64(BRICK_T[a])
        coords.append(ContinuumVector(r * (2.0 + 0.2 * t), s * (1.0 + 0.3 * t), t, 0.0))
    # J = 2 + 0.8t + 0.06t². OpenSees omits the linear term and
    # weights the even term by the center determinant.
    var tangent = isotropic_continuum_tangent(100.0, 0.25, 6, True)
    var element = prepare_continuum(
        "SSPbrick", coords, 1.0, 2.0, ContinuumVector(3.0, 0.0, 0.0, 0.0), 0.0, tangent
    )
    for a in range(8):
        assert_almost_equal(element.lumped_mass[3 * a], 4.08, atol=1.0e-12)
        assert_almost_equal(element.body_force[3 * a], 6.12, atol=1.0e-12)
