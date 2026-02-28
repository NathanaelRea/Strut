from collections import List
from math import cos
from os import abort


fn _legendre_polynomial_and_derivative(n: Int, x: Float64) -> (Float64, Float64, Float64):
    if n == 0:
        return (1.0, 0.0, 0.0)
    if n == 1:
        return (x, 1.0, 1.0)

    var pnm2 = 1.0
    var pnm1 = x
    for k in range(2, n + 1):
        var kf = Float64(k)
        var pk = ((2.0 * kf - 1.0) * x * pnm1 - (kf - 1.0) * pnm2) / kf
        pnm2 = pnm1
        pnm1 = pk

    var pn = pnm1
    var pn_minus_1 = pnm2
    var denom = x * x - 1.0
    if abs(denom) <= 1.0e-18:
        var x_off = x
        if x_off >= 0.0:
            x_off -= 1.0e-12
        else:
            x_off += 1.0e-12
        var alt = _legendre_polynomial_and_derivative(n, x_off)
        return (pn, pn_minus_1, alt[2])
    var dpn = Float64(n) * (x * pn - pn_minus_1) / denom
    return (pn, pn_minus_1, dpn)


fn _sort_points(mut x: List[Float64], mut w: List[Float64]):
    var n = len(x)
    for i in range(1, n):
        var key_x = x[i]
        var key_w = w[i]
        var j = i - 1
        while j >= 0 and x[j] > key_x:
            x[j + 1] = x[j]
            w[j + 1] = w[j]
            j -= 1
        x[j + 1] = key_x
        w[j + 1] = key_w


fn _fill_lobatto(num_int_pts: Int, mut x: List[Float64], mut w: List[Float64]):
    if num_int_pts < 2:
        abort("Lobatto integration requires num_int_pts >= 2")

    x.resize(num_int_pts, 0.0)
    w.resize(num_int_pts, 0.0)

    if num_int_pts == 3:
        x[0] = -1.0
        x[1] = 0.0
        x[2] = 1.0
        w[0] = 1.0 / 3.0
        w[1] = 4.0 / 3.0
        w[2] = 1.0 / 3.0
        return

    if num_int_pts == 5:
        x[0] = -1.0
        x[1] = -0.6546536707079772
        x[2] = 0.0
        x[3] = 0.6546536707079772
        x[4] = 1.0
        w[0] = 0.1
        w[1] = 0.5444444444444444
        w[2] = 0.7111111111111111
        w[3] = 0.5444444444444444
        w[4] = 0.1
        return

    var n = num_int_pts
    var m = n - 1
    var n_float = Float64(n)

    x[0] = -1.0
    x[n - 1] = 1.0
    w[0] = 2.0 / (n_float * (n_float - 1.0))
    w[n - 1] = w[0]

    var pi = 3.141592653589793
    for i in range(1, n - 1):
        var guess_angle = pi * Float64(i) / Float64(n - 1)
        var root = cos(guess_angle)

        for _ in range(80):
            var poly = _legendre_polynomial_and_derivative(m, root)
            var pm = poly[0]
            var dpm = poly[2]
            var denom = 1.0 - root * root
            if abs(denom) <= 1.0e-18:
                break
            var ddpm = (2.0 * root * dpm - Float64(m * (m + 1)) * pm) / denom
            if abs(ddpm) <= 1.0e-24:
                break
            var delta = dpm / ddpm
            root -= delta
            if abs(delta) <= 1.0e-14:
                break

        x[i] = root
        var pm_root = _legendre_polynomial_and_derivative(m, root)[0]
        var pm2 = pm_root * pm_root
        if pm2 <= 0.0:
            abort("failed to compute Lobatto integration weights")
        w[i] = 2.0 / (n_float * (n_float - 1.0) * pm2)

    _sort_points(x, w)


fn _fill_legendre(num_int_pts: Int, mut x: List[Float64], mut w: List[Float64]):
    if num_int_pts < 1:
        abort("Legendre integration requires num_int_pts >= 1")

    x.resize(num_int_pts, 0.0)
    w.resize(num_int_pts, 0.0)

    if num_int_pts == 1:
        x[0] = 0.0
        w[0] = 2.0
        return

    var n = num_int_pts
    var m = (n + 1) // 2
    var pi = 3.141592653589793

    for i in range(m):
        var i_float = Float64(i)
        var n_float = Float64(n)
        var root = cos(pi * (i_float + 0.75) / (n_float + 0.5))
        var dpn = 0.0

        for _ in range(80):
            var poly = _legendre_polynomial_and_derivative(n, root)
            var pn = poly[0]
            dpn = poly[2]
            if abs(dpn) <= 1.0e-24:
                break
            var delta = pn / dpn
            root -= delta
            if abs(delta) <= 1.0e-14:
                break

        var one_minus = 1.0 - root * root
        if one_minus <= 1.0e-24:
            abort("failed to compute Legendre integration weights")
        var weight = 2.0 / (one_minus * dpn * dpn)

        var left = i
        var right = n - 1 - i
        x[left] = -root
        w[left] = weight
        x[right] = root
        w[right] = weight

    _sort_points(x, w)


fn _fill_radau(num_int_pts: Int, mut x: List[Float64], mut w: List[Float64]):
    if num_int_pts < 1:
        abort("Radau integration requires num_int_pts >= 1")

    x.resize(num_int_pts, 0.0)
    w.resize(num_int_pts, 0.0)

    if num_int_pts == 1:
        x[0] = -1.0
        w[0] = 2.0
        return

    var n = num_int_pts
    var n_sq = Float64(n * n)
    x[0] = -1.0
    w[0] = 2.0 / n_sq

    var pi = 3.141592653589793
    for i in range(1, n):
        var angle = Float64(2 * i - 1) * pi / Float64(2 * n - 1)
        var root = cos(angle)

        for _ in range(100):
            var pn_data = _legendre_polynomial_and_derivative(n, root)
            var pnm1_data = _legendre_polynomial_and_derivative(n - 1, root)
            var f = pn_data[0] + pnm1_data[0]
            var fp = pn_data[2] + pnm1_data[2]
            if abs(fp) <= 1.0e-24:
                break
            var delta = f / fp
            root -= delta
            if abs(delta) <= 1.0e-14:
                break

        x[i] = root
        var pnm1 = _legendre_polynomial_and_derivative(n - 1, root)[0]
        var denom = n_sq * pnm1 * pnm1
        if denom <= 1.0e-24:
            abort("failed to compute Radau integration weights")
        w[i] = (1.0 - root) / denom

    _sort_points(x, w)


fn beam_integration_rule(
    integration: String,
    num_int_pts: Int,
    mut xis: List[Float64],
    mut weights: List[Float64],
):
    if integration == "Lobatto":
        _fill_lobatto(num_int_pts, xis, weights)
    elif integration == "Legendre":
        _fill_legendre(num_int_pts, xis, weights)
    elif integration == "Radau":
        _fill_radau(num_int_pts, xis, weights)
    else:
        abort("unsupported beam integration: " + integration)

    for i in range(len(xis)):
        xis[i] = 0.5 * (xis[i] + 1.0)
        weights[i] *= 0.5


fn beam_integration_xi_weight(
    integration: String, num_int_pts: Int, ip: Int
) -> (Float64, Float64):
    if ip < 0 or ip >= num_int_pts:
        abort("beam integration point index out of range")
    var xis: List[Float64] = []
    var weights: List[Float64] = []
    beam_integration_rule(integration, num_int_pts, xis, weights)
    return (xis[ip], weights[ip])


fn beam_integration_is_supported(integration: String, num_int_pts: Int) -> Bool:
    if num_int_pts < 1:
        return False
    if integration == "Lobatto":
        return num_int_pts >= 2
    if integration == "Legendre":
        return num_int_pts >= 1
    if integration == "Radau":
        return num_int_pts >= 1
    return False


fn beam_integration_validate_or_abort(
    beam_col_type: String, integration: String, num_int_pts: Int
):
    if integration != "Lobatto" and integration != "Legendre" and integration != "Radau":
        abort(beam_col_type + " supports integration Lobatto, Legendre, or Radau")
    if integration == "Lobatto" and num_int_pts < 2:
        abort(beam_col_type + " Lobatto integration requires num_int_pts >= 2")
    if (integration == "Legendre" or integration == "Radau") and num_int_pts < 1:
        abort(beam_col_type + " " + integration + " integration requires num_int_pts >= 1")
