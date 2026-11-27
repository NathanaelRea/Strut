from math import sqrt


struct HingeRadau2d(ImplicitlyCopyable, Movable):
    var section_i: Int
    var section_j: Int
    var length_i: Float64
    var length_j: Float64
    var axial_i: Int
    var flexural_i: Int
    var axial_j: Int
    var flexural_j: Int
    var elastic_ea: Float64
    var elastic_ei: Float64
    var elastic_end_i: (Float64, Float64, Float64)
    var elastic_end_j: (Float64, Float64, Float64)
    var max_iters: Int
    var tolerance: Float64

    fn __init__(out self):
        self.section_i = -1
        self.section_j = -1
        self.length_i = 0.0
        self.length_j = 0.0
        self.axial_i = -1
        self.flexural_i = -1
        self.axial_j = -1
        self.flexural_j = -1
        self.elastic_ea = 0.0
        self.elastic_ei = 0.0
        self.elastic_end_i = (0.0, 0.0, 0.0)
        self.elastic_end_j = (0.0, 0.0, 0.0)
        self.max_iters = 10
        self.tolerance = 1.0e-10

    fn is_defined(self) -> Bool:
        return self.section_i >= 0

    fn xi_weight(self, point: Int, length: Float64) -> (Float64, Float64):
        var i = self.length_i / length
        var j = self.length_j / length
        var alpha = 0.5 - 2.0 * (i + j)
        var beta = 0.5 + 2.0 * (i - j)
        if point == 0:
            return (0.0, i)
        if point == 1:
            return (8.0 / 3.0 * i, 3.0 * i)
        if point == 2:
            return (beta - alpha / sqrt(3.0), alpha)
        if point == 3:
            return (beta + alpha / sqrt(3.0), alpha)
        if point == 4:
            return (1.0 - 8.0 / 3.0 * j, 3.0 * j)
        return (1.0, j)

    fn material_indices(self, point: Int) -> (Int, Int):
        if point == 0:
            return (self.axial_i, self.flexural_i)
        return (self.axial_j, self.flexural_j)
