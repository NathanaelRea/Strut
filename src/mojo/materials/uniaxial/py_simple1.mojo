from os import abort

from materials.uniaxial.py_tz_qz_state import _abs, PyTzQzState


alias _PY_TOLERANCE: Float64 = 1.0e-12
alias _PY_MAX_ITERATIONS: Int = 20


struct _PyParams(Copyable, Movable):
    var soil_type: Int
    var pult: Float64
    var y50: Float64
    var drag: Float64
    var dashpot: Float64
    var yref: Float64
    var np: Float64
    var elast: Float64
    var far_tang: Float64
    var nf_krig: Float64

    fn __init__(
        out self,
        p0: Float64,
        p1: Float64,
        p2: Float64,
        p3: Float64,
        p4: Float64,
    ):
        var soil_type = Int(p0)
        var pult = p1
        var y50 = p2
        var drag = p3
        var dashpot = p4

        if soil_type == 0:
            pult = 1.0e-12
            y50 = 1.0e12
        if drag <= _PY_TOLERANCE:
            drag = _PY_TOLERANCE
        if dashpot < 0.0:
            dashpot = 0.0
        if pult <= 0.0 or y50 <= 0.0:
            abort("PySimple1 requires positive nonzero pult and y50")

        var yref: Float64
        var np: Float64
        var elast: Float64
        var far_tang: Float64
        if soil_type == 0 or soil_type == 1:
            yref = 10.0 * y50
            np = 5.0
            elast = 0.35
            far_tang = pult / (8.0 * (elast**2.0) * y50)
        elif soil_type == 2:
            yref = 0.5 * y50
            np = 2.0
            elast = 0.2
            far_tang = 0.542 * pult / y50
        else:
            abort("PySimple1 only accepts soilType of 1 or 2")
            yref = 0.0
            np = 1.0
            elast = 0.2
            far_tang = 0.0

        self.soil_type = soil_type
        self.pult = pult
        self.y50 = y50
        self.drag = drag
        self.dashpot = dashpot
        self.yref = yref
        self.np = np
        self.elast = elast
        self.far_tang = far_tang
        self.nf_krig = 100.0 * (0.5 * pult) / y50


fn py_simple1_initial_tangent(
    p0: Float64, p1: Float64, p2: Float64, p3: Float64, p4: Float64
) -> Float64:
    var p = _PyParams(p0, p1, p2, p3, p4)
    var nf_tang = p.nf_krig
    var drag_tang = (p.pult * p.drag) * (p.y50 / 2.0) * ((p.y50 / 2.0) ** -2.0)
    var close_yleft = -p.y50 / 100.0
    var close_yright = p.y50 / 100.0
    var close_tang = (
        1.8
        * p.pult
        * (p.y50 / 50.0)
        * (((p.y50 / 50.0 + close_yright) ** -2.0) + ((p.y50 / 50.0 - close_yleft) ** -2.0))
    )
    var gap_tang = close_tang + drag_tang
    return (1.0 / gap_tang + 1.0 / nf_tang + 1.0 / p.far_tang) ** -1.0


fn py_simple1_init_state(
    p0: Float64,
    p1: Float64,
    p2: Float64,
    p3: Float64,
    p4: Float64,
    mut s: PyTzQzState,
) -> Float64:
    var p = _PyParams(p0, p1, p2, p3, p4)
    s = PyTzQzState()
    s.py_reference_power = p.yref**p.np

    s.t_far_p = 0.0
    s.t_far_y = 0.0
    s.t_far_tang = p.far_tang

    s.t_nf_pinr = p.elast * p.pult
    s.t_nf_pinl = -s.t_nf_pinr
    s.t_nf_yinr = s.t_nf_pinr / p.nf_krig
    s.t_nf_yinl = -s.t_nf_yinr
    s.t_nf_p = 0.0
    s.t_nf_y = 0.0
    s.t_nf_tang = p.nf_krig

    s.t_drag_pin = 0.0
    s.t_drag_yin = 0.0
    s.t_drag_p = 0.0
    s.t_drag_y = 0.0
    s.t_drag_tang = (
        (p.pult * p.drag - s.t_drag_p)
        * (p.y50 / 2.0)
        * ((p.y50 / 2.0 - s.t_drag_y + s.t_drag_yin) ** -2.0)
    )

    s.t_close_yleft = -p.y50 / 100.0
    s.t_close_yright = p.y50 / 100.0
    s.t_close_p = 0.0
    s.t_close_y = 0.0
    s.t_close_tang = (
        1.8
        * p.pult
        * (p.y50 / 50.0)
        * (
            ((p.y50 / 50.0 + s.t_close_yright - s.t_close_y) ** -2.0)
            + ((p.y50 / 50.0 + s.t_close_y - s.t_close_yleft) ** -2.0)
        )
    )

    s.t_gap_y = 0.0
    s.t_gap_p = 0.0
    s.t_gap_tang = s.t_close_tang + s.t_drag_tang
    s.t_p_raw = 0.0
    s.commit()

    return (1.0 / s.t_gap_tang + 1.0 / s.t_nf_tang + 1.0 / s.t_far_tang) ** -1.0


fn _py_get_closure(p: _PyParams, mut s: PyTzQzState, ylast: Float64, dy: Float64):
    if s.t_close_yleft != s.c_close_yleft:
        s.t_close_yleft = s.c_close_yleft
    if s.t_close_yright != s.c_close_yright:
        s.t_close_yright = s.c_close_yright

    s.t_close_y = ylast + dy
    var yrebound = 1.5 * p.y50
    if s.t_nf_y + s.t_close_y > -s.t_close_yleft + yrebound:
        s.t_close_yleft = -(s.t_nf_y + s.t_close_y) + yrebound
    if s.t_nf_y + s.t_close_y < -s.t_close_yright - yrebound:
        s.t_close_yright = -(s.t_nf_y + s.t_close_y) - yrebound

    s.t_close_p = (
        1.8
        * p.pult
        * (p.y50 / 50.0)
        * (
            ((p.y50 / 50.0 + s.t_close_yright - s.t_close_y) ** -1.0)
            - ((p.y50 / 50.0 + s.t_close_y - s.t_close_yleft) ** -1.0)
        )
    )
    s.t_close_tang = (
        1.8
        * p.pult
        * (p.y50 / 50.0)
        * (
            ((p.y50 / 50.0 + s.t_close_yright - s.t_close_y) ** -2.0)
            + ((p.y50 / 50.0 + s.t_close_y - s.t_close_yleft) ** -2.0)
        )
    )
    if s.t_close_tang <= 1.0e-2 * p.pult / p.y50:
        s.t_close_tang = 1.0e-2 * p.pult / p.y50


fn _py_get_drag(p: _PyParams, mut s: PyTzQzState, ylast: Float64, dy: Float64):
    s.t_drag_y = ylast + dy
    var pmax = p.drag * p.pult
    var dy_total = s.t_drag_y - s.c_drag_y

    if _abs(dy_total * s.t_drag_tang / p.pult) < 10.0 * _PY_TOLERANCE:
        s.t_drag_p = s.t_drag_p + dy * s.t_drag_tang
        if _abs(s.t_drag_p) >= pmax:
            s.t_drag_p = (s.t_drag_p / _abs(s.t_drag_p)) * (1.0 - 1.0e-8) * pmax
        return

    if s.t_drag_pin != s.c_drag_pin:
        s.t_drag_pin = s.c_drag_pin
        s.t_drag_yin = s.c_drag_yin

    if s.c_drag_y > s.c_drag_yin and dy_total < 0.0:
        s.t_drag_pin = s.c_drag_p
        s.t_drag_yin = s.c_drag_y

    if s.c_drag_y < s.c_drag_yin and dy_total > 0.0:
        s.t_drag_pin = s.c_drag_p
        s.t_drag_yin = s.c_drag_y

    if dy_total >= 0.0:
        s.t_drag_p = pmax - (pmax - s.t_drag_pin) * (p.y50 / 2.0) * (
            (p.y50 / 2.0 + s.t_drag_y - s.t_drag_yin) ** -1.0
        )
        s.t_drag_tang = (
            (pmax - s.t_drag_pin)
            * (p.y50 / 2.0)
            * ((p.y50 / 2.0 + s.t_drag_y - s.t_drag_yin) ** -2.0)
        )

    if dy_total < 0.0:
        s.t_drag_p = -pmax + (pmax + s.t_drag_pin) * (p.y50 / 2.0) * (
            (p.y50 / 2.0 - s.t_drag_y + s.t_drag_yin) ** -1.0
        )
        s.t_drag_tang = (
            (pmax + s.t_drag_pin)
            * (p.y50 / 2.0)
            * ((p.y50 / 2.0 - s.t_drag_y + s.t_drag_yin) ** -2.0)
        )

    if _abs(s.t_drag_p) >= pmax:
        s.t_drag_p = (s.t_drag_p / _abs(s.t_drag_p)) * (1.0 - _PY_TOLERANCE) * pmax
    if s.t_drag_tang <= 1.0e-2 * p.pult / p.y50:
        s.t_drag_tang = 1.0e-2 * p.pult / p.y50


fn _py_get_gap(p: _PyParams, mut s: PyTzQzState, ylast: Float64, dy_in: Float64, dy_old: Float64):
    var dy = dy_in
    s.t_gap_y = ylast + dy
    if s.t_gap_y > s.t_close_yright:
        dy = 0.75 * (s.t_close_yright - ylast)
    if s.t_gap_y < s.t_close_yleft:
        dy = 0.75 * (s.t_close_yleft - ylast)

    if dy * dy_old < 0.0 and _abs(dy / dy_old) > 0.5:
        dy = -dy_old / 2.0

    s.t_gap_y = ylast + dy
    _py_get_closure(p, s, ylast, dy)
    _py_get_drag(p, s, ylast, dy)
    s.t_gap_p = s.t_drag_p + s.t_close_p
    s.t_gap_tang = s.t_drag_tang + s.t_close_tang

    if _abs(s.t_gap_p) >= p.pult:
        s.t_gap_p = (s.t_gap_p / _abs(s.t_gap_p)) * (1.0 - _PY_TOLERANCE) * p.pult


fn _py_get_near_field(
    p: _PyParams, mut s: PyTzQzState, ylast: Float64, dy_in: Float64, dy_old: Float64
):
    var dy = dy_in
    if dy * dy_old < 0.0 and _abs(dy / dy_old) > 0.5:
        dy = -dy_old / 2.0
    if dy * dy_old < -p.y50 * p.y50:
        dy = (s.t_nf_yinr + s.t_nf_yinl) / 2.0 - ylast

    s.t_nf_y = ylast + dy
    var nf_dy = s.t_nf_y - s.c_nf_y

    if _abs(nf_dy * s.t_nf_tang / p.pult) < 10.0 * _PY_TOLERANCE:
        s.t_nf_p = s.t_nf_p + dy * s.t_nf_tang
        if _abs(s.t_nf_p) >= p.pult:
            s.t_nf_p = (s.t_nf_p / _abs(s.t_nf_p)) * (1.0 - _PY_TOLERANCE) * p.pult
        return

    if s.t_nf_pinr != s.c_nf_pinr or s.t_nf_pinl != s.c_nf_pinl:
        s.t_nf_pinr = s.c_nf_pinr
        s.t_nf_pinl = s.c_nf_pinl
        s.t_nf_yinr = s.c_nf_yinr
        s.t_nf_yinl = s.c_nf_yinl

    var change_direction = False
    var min_e = 0.25
    if s.c_nf_p > s.c_nf_pinr and nf_dy < 0.0:
        change_direction = True
        s.t_nf_pinr = s.c_nf_p
        if _abs(s.t_nf_pinr) >= (1.0 - _PY_TOLERANCE) * p.pult:
            s.t_nf_pinr = (1.0 - 2.0 * _PY_TOLERANCE) * p.pult
        s.t_nf_pinl = s.t_nf_pinr - 2.0 * p.pult * p.elast
        if s.t_nf_pinl > -min_e * p.pult:
            s.t_nf_pinl = -min_e * p.pult
        s.t_nf_yinr = s.c_nf_y
        s.t_nf_yinl = s.t_nf_yinr - (s.t_nf_pinr - s.t_nf_pinl) / p.nf_krig

    if s.c_nf_p < s.c_nf_pinl and nf_dy > 0.0:
        change_direction = True
        s.t_nf_pinl = s.c_nf_p
        if _abs(s.t_nf_pinl) >= (1.0 - _PY_TOLERANCE) * p.pult:
            s.t_nf_pinl = (-1.0 + 2.0 * _PY_TOLERANCE) * p.pult
        s.t_nf_pinr = s.t_nf_pinl + 2.0 * p.pult * p.elast
        if s.t_nf_pinr < min_e * p.pult:
            s.t_nf_pinr = min_e * p.pult
        s.t_nf_yinl = s.c_nf_y
        s.t_nf_yinr = s.t_nf_yinl + (s.t_nf_pinr - s.t_nf_pinl) / p.nf_krig

    if change_direction:
        var maxdy = 0.25 * p.pult / p.nf_krig
        if _abs(dy) > maxdy:
            dy = (dy / _abs(dy)) * maxdy

    s.t_nf_y = ylast + dy

    if nf_dy >= 0.0:
        if s.t_nf_y <= s.t_nf_yinr:
            s.t_nf_tang = p.nf_krig
            s.t_nf_p = s.t_nf_pinl + (s.t_nf_y - s.t_nf_yinl) * p.nf_krig
        else:
            s.t_nf_tang = (
                p.np
                * (p.pult - s.t_nf_pinr)
                * s.py_reference_power
                * ((p.yref - s.t_nf_yinr + s.t_nf_y) ** (-p.np - 1.0))
            )
            s.t_nf_p = p.pult - (p.pult - s.t_nf_pinr) * (
                (p.yref / (p.yref - s.t_nf_yinr + s.t_nf_y)) ** p.np
            )

    if nf_dy < 0.0:
        if s.t_nf_y >= s.t_nf_yinl:
            s.t_nf_tang = p.nf_krig
            s.t_nf_p = s.t_nf_pinr + (s.t_nf_y - s.t_nf_yinr) * p.nf_krig
        else:
            s.t_nf_tang = (
                p.np
                * (p.pult + s.t_nf_pinl)
                * s.py_reference_power
                * ((p.yref + s.t_nf_yinl - s.t_nf_y) ** (-p.np - 1.0))
            )
            s.t_nf_p = -p.pult + (p.pult + s.t_nf_pinl) * (
                (p.yref / (p.yref + s.t_nf_yinl - s.t_nf_y)) ** p.np
            )

    if _abs(s.t_nf_p) >= p.pult:
        s.t_nf_p = (s.t_nf_p / _abs(s.t_nf_p)) * (1.0 - _PY_TOLERANCE) * p.pult
    if s.t_nf_tang <= 1.0e-2 * p.pult / p.y50:
        s.t_nf_tang = 1.0e-2 * p.pult / p.y50


fn py_simple1_set_trial(
    p0: Float64,
    p1: Float64,
    p2: Float64,
    p3: Float64,
    p4: Float64,
    mut eps_t: Float64,
    mut sig_t: Float64,
    mut tangent_t: Float64,
    mut s: PyTzQzState,
    newy: Float64,
):
    var p = _PyParams(p0, p1, p2, p3, p4)
    var ty = eps_t
    var tp = s.t_p_raw
    var ttangent = tangent_t

    var dy = newy - ty
    var dp = ttangent * dy

    var num_steps = 1
    var temp = _abs(dp / p.pult)
    if temp > 0.5:
        if temp > 50.0:
            num_steps = 100
        else:
            num_steps = 1 + Int(temp * 2.0)

    temp = _abs(dy / p.y50)
    if temp > 1.0:
        if temp > 100.0:
            num_steps = 100
        else:
            num_steps = 1 + Int(temp)

    if num_steps > 100:
        num_steps = 100
    var step_size = 1.0 / Float64(num_steps)
    dy = step_size * dy

    for _ in range(num_steps):
        ty = ty + dy
        dp = ttangent * dy

        var dy_gap_old = ((tp + dp) - s.t_gap_p) / s.t_gap_tang
        var dy_nf_old = ((tp + dp) - s.t_nf_p) / s.t_nf_tang

        for _ in range(1, _PY_MAX_ITERATIONS):
            tp = tp + dp

            var dy_nf = (tp - s.t_nf_p) / s.t_nf_tang
            _py_get_near_field(p, s, s.t_nf_y, dy_nf, dy_nf_old)
            var p_unbalance = tp - s.t_nf_p
            var yres_nf = p_unbalance / s.t_nf_tang
            dy_nf_old = dy_nf

            var dy_gap = (tp - s.t_gap_p) / s.t_gap_tang
            _py_get_gap(p, s, s.t_gap_y, dy_gap, dy_gap_old)
            var p_unbalance2 = tp - s.t_gap_p
            var yres_gap = p_unbalance2 / s.t_gap_tang
            dy_gap_old = dy_gap

            var dy_far = (tp - s.t_far_p) / s.t_far_tang
            s.t_far_y = s.t_far_y + dy_far
            s.t_far_p = s.t_far_tang * s.t_far_y
            var p_unbalance3 = tp - s.t_far_p
            var yres_far = p_unbalance3 / s.t_far_tang

            ttangent = (1.0 / s.t_gap_tang + 1.0 / s.t_nf_tang + 1.0 / s.t_far_tang) ** -1.0

            var dv = ty - (s.t_gap_y + yres_gap) - (s.t_nf_y + yres_nf) - (s.t_far_y + yres_far)
            dp = ttangent * dv

            var psum = _abs(p_unbalance) + _abs(p_unbalance2) + _abs(p_unbalance3)
            if psum / p.pult < _PY_TOLERANCE:
                break

    s.t_p_raw = tp
    eps_t = ty
    tangent_t = ttangent
    if _abs(tp) >= (1.0 - _PY_TOLERANCE) * p.pult:
        sig_t = (1.0 - _PY_TOLERANCE) * p.pult * tp / _abs(tp)
    else:
        sig_t = tp
