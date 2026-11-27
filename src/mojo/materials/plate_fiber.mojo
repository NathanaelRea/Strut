from collections import InlineArray
from os import abort


fn elastic_plate_fiber_tangent(E: Float64, nu: Float64) -> InlineArray[Float64, 25]:
    """Elastic isotropic PlateFiber tangent after eliminating sigma_33.

    The retained Voigt components from PlateFiberMaterial are xx, yy, xy,
    yz, zx. The Schur complement of the 33 block gives the plane-stress
    normal terms and leaves the three engineering-shear terms unchanged.
    """
    if E <= 0.0 or nu <= -1.0 or nu >= 0.5:
        abort("PlateFiber requires E > 0 and -1 < nu < 0.5")
    var tangent = InlineArray[Float64, 25](fill=0.0)
    var plane = E / (1.0 - nu * nu)
    tangent[0] = plane
    tangent[1] = nu * plane
    tangent[5] = tangent[1]
    tangent[6] = plane
    var shear = E / (2.0 * (1.0 + nu))
    tangent[12] = shear
    tangent[18] = shear
    tangent[24] = shear
    return tangent^
