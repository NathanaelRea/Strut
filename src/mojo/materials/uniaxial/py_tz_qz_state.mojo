fn _abs(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x


struct PyTzQzState(Defaultable, ImplicitlyCopyable, Movable):
    # Committed internal state
    # Constant in PySimple1; unused by TzSimple1 and QzSimple1.
    var py_reference_power: Float64
    var c_p_raw: Float64
    var c_nf_pinr: Float64
    var c_nf_pinl: Float64
    var c_nf_yinr: Float64
    var c_nf_yinl: Float64
    var c_nf_p: Float64
    var c_nf_y: Float64
    var c_nf_tang: Float64

    var c_drag_pin: Float64
    var c_drag_yin: Float64
    var c_drag_p: Float64
    var c_drag_y: Float64
    var c_drag_tang: Float64

    var c_close_yleft: Float64
    var c_close_yright: Float64
    var c_close_p: Float64
    var c_close_y: Float64
    var c_close_tang: Float64

    var c_gap_y: Float64
    var c_gap_p: Float64
    var c_gap_tang: Float64

    var c_far_y: Float64
    var c_far_p: Float64
    var c_far_tang: Float64

    # Trial internal state
    var t_p_raw: Float64
    var t_nf_pinr: Float64
    var t_nf_pinl: Float64
    var t_nf_yinr: Float64
    var t_nf_yinl: Float64
    var t_nf_p: Float64
    var t_nf_y: Float64
    var t_nf_tang: Float64

    var t_drag_pin: Float64
    var t_drag_yin: Float64
    var t_drag_p: Float64
    var t_drag_y: Float64
    var t_drag_tang: Float64

    var t_close_yleft: Float64
    var t_close_yright: Float64
    var t_close_p: Float64
    var t_close_y: Float64
    var t_close_tang: Float64

    var t_gap_y: Float64
    var t_gap_p: Float64
    var t_gap_tang: Float64

    var t_far_y: Float64
    var t_far_p: Float64
    var t_far_tang: Float64

    # QzSimple1 evolves Elast in place during getNearField (not reverted on commit/revert in OpenSees)
    var qz_elast: Float64

    fn __init__(out self):
        self.py_reference_power = 0.0
        self.c_p_raw = 0.0
        self.c_nf_pinr = 0.0
        self.c_nf_pinl = 0.0
        self.c_nf_yinr = 0.0
        self.c_nf_yinl = 0.0
        self.c_nf_p = 0.0
        self.c_nf_y = 0.0
        self.c_nf_tang = 0.0
        self.c_drag_pin = 0.0
        self.c_drag_yin = 0.0
        self.c_drag_p = 0.0
        self.c_drag_y = 0.0
        self.c_drag_tang = 0.0
        self.c_close_yleft = 0.0
        self.c_close_yright = 0.0
        self.c_close_p = 0.0
        self.c_close_y = 0.0
        self.c_close_tang = 0.0
        self.c_gap_y = 0.0
        self.c_gap_p = 0.0
        self.c_gap_tang = 0.0
        self.c_far_y = 0.0
        self.c_far_p = 0.0
        self.c_far_tang = 0.0

        self.t_p_raw = 0.0
        self.t_nf_pinr = 0.0
        self.t_nf_pinl = 0.0
        self.t_nf_yinr = 0.0
        self.t_nf_yinl = 0.0
        self.t_nf_p = 0.0
        self.t_nf_y = 0.0
        self.t_nf_tang = 0.0
        self.t_drag_pin = 0.0
        self.t_drag_yin = 0.0
        self.t_drag_p = 0.0
        self.t_drag_y = 0.0
        self.t_drag_tang = 0.0
        self.t_close_yleft = 0.0
        self.t_close_yright = 0.0
        self.t_close_p = 0.0
        self.t_close_y = 0.0
        self.t_close_tang = 0.0
        self.t_gap_y = 0.0
        self.t_gap_p = 0.0
        self.t_gap_tang = 0.0
        self.t_far_y = 0.0
        self.t_far_p = 0.0
        self.t_far_tang = 0.0
        self.qz_elast = 0.0

    fn commit(mut self):
        self.c_p_raw = self.t_p_raw
        self.c_nf_pinr = self.t_nf_pinr
        self.c_nf_pinl = self.t_nf_pinl
        self.c_nf_yinr = self.t_nf_yinr
        self.c_nf_yinl = self.t_nf_yinl
        self.c_nf_p = self.t_nf_p
        self.c_nf_y = self.t_nf_y
        self.c_nf_tang = self.t_nf_tang
        self.c_drag_pin = self.t_drag_pin
        self.c_drag_yin = self.t_drag_yin
        self.c_drag_p = self.t_drag_p
        self.c_drag_y = self.t_drag_y
        self.c_drag_tang = self.t_drag_tang
        self.c_close_yleft = self.t_close_yleft
        self.c_close_yright = self.t_close_yright
        self.c_close_p = self.t_close_p
        self.c_close_y = self.t_close_y
        self.c_close_tang = self.t_close_tang
        self.c_gap_y = self.t_gap_y
        self.c_gap_p = self.t_gap_p
        self.c_gap_tang = self.t_gap_tang
        self.c_far_y = self.t_far_y
        self.c_far_p = self.t_far_p
        self.c_far_tang = self.t_far_tang

    fn revert_trial(mut self):
        self.t_p_raw = self.c_p_raw
        self.t_nf_pinr = self.c_nf_pinr
        self.t_nf_pinl = self.c_nf_pinl
        self.t_nf_yinr = self.c_nf_yinr
        self.t_nf_yinl = self.c_nf_yinl
        self.t_nf_p = self.c_nf_p
        self.t_nf_y = self.c_nf_y
        self.t_nf_tang = self.c_nf_tang
        self.t_drag_pin = self.c_drag_pin
        self.t_drag_yin = self.c_drag_yin
        self.t_drag_p = self.c_drag_p
        self.t_drag_y = self.c_drag_y
        self.t_drag_tang = self.c_drag_tang
        self.t_close_yleft = self.c_close_yleft
        self.t_close_yright = self.c_close_yright
        self.t_close_p = self.c_close_p
        self.t_close_y = self.c_close_y
        self.t_close_tang = self.c_close_tang
        self.t_gap_y = self.c_gap_y
        self.t_gap_p = self.c_gap_p
        self.t_gap_tang = self.c_gap_tang
        self.t_far_y = self.c_far_y
        self.t_far_p = self.c_far_p
        self.t_far_tang = self.c_far_tang
