from collections import List
from os import abort
from tag_types import UniMaterialTypeTag


fn _abs(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x


fn _sign(x: Float64) -> Float64:
    if x < 0.0:
        return -1.0
    return 1.0


fn _pow_abs(x: Float64, p: Float64) -> Float64:
    return _abs(x) ** p


struct StressTangent(Defaultable, Movable, ImplicitlyCopyable):
    var stress: Float64
    var tangent: Float64

    fn __init__(out self):
        self.stress = 0.0
        self.tangent = 0.0

    fn __init__(out self, stress: Float64, tangent: Float64):
        self.stress = stress
        self.tangent = tangent


struct EndStrainSlope(Defaultable, Movable, ImplicitlyCopyable):
    var end_strain: Float64
    var unload_slope: Float64

    fn __init__(out self):
        self.end_strain = 0.0
        self.unload_slope = 0.0

    fn __init__(out self, end_strain: Float64, unload_slope: Float64):
        self.end_strain = end_strain
        self.unload_slope = unload_slope


struct UniMaterialDef(Defaultable, Movable, ImplicitlyCopyable):
    var mat_type: Int
    var p0: Float64
    var p1: Float64
    var p2: Float64
    var p3: Float64
    var p4: Float64
    var p5: Float64
    var p6: Float64
    var p7: Float64
    var p8: Float64
    var p9: Float64
    var p10: Float64

    fn __init__(out self):
        self.mat_type = UniMaterialTypeTag.Elastic
        self.p0 = 0.0
        self.p1 = 0.0
        self.p2 = 0.0
        self.p3 = 0.0
        self.p4 = 0.0
        self.p5 = 0.0
        self.p6 = 0.0
        self.p7 = 0.0
        self.p8 = 0.0
        self.p9 = 0.0
        self.p10 = 0.0

    fn __init__(
        out self, mat_type: Int, p0: Float64, p1: Float64, p2: Float64, p3: Float64
    ):
        self.mat_type = mat_type
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = 0.0
        self.p5 = 0.0
        self.p6 = 0.0
        self.p7 = 0.0
        self.p8 = 0.0
        self.p9 = 0.0
        self.p10 = 0.0

    fn __init__(
        out self,
        mat_type: Int,
        p0: Float64,
        p1: Float64,
        p2: Float64,
        p3: Float64,
        p4: Float64,
        p5: Float64,
        p6: Float64,
        p7: Float64,
        p8: Float64,
        p9: Float64,
        p10: Float64,
    ):
        self.mat_type = mat_type
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.p5 = p5
        self.p6 = p6
        self.p7 = p7
        self.p8 = p8
        self.p9 = p9
        self.p10 = p10


struct UniMaterialState(Defaultable, Movable, ImplicitlyCopyable):
    var eps_c: Float64
    var sig_c: Float64
    var tangent_c: Float64
    var eps_p_c: Float64
    var alpha_c: Float64

    var eps_t: Float64
    var sig_t: Float64
    var tangent_t: Float64
    var eps_p_t: Float64
    var alpha_t: Float64
    var min_strain_c: Float64
    var end_strain_c: Float64
    var unload_slope_c: Float64
    var min_strain_t: Float64
    var end_strain_t: Float64
    var unload_slope_t: Float64

    # Steel02 history variables
    var s2_epsmin_c: Float64
    var s2_epsmax_c: Float64
    var s2_epspl_c: Float64
    var s2_epss0_c: Float64
    var s2_sigs0_c: Float64
    var s2_epsr_c: Float64
    var s2_sigr_c: Float64
    var s2_kon_c: Int

    var s2_epsmin_t: Float64
    var s2_epsmax_t: Float64
    var s2_epspl_t: Float64
    var s2_epss0_t: Float64
    var s2_sigs0_t: Float64
    var s2_epsr_t: Float64
    var s2_sigr_t: Float64
    var s2_kon_t: Int

    # Concrete02 history variables
    var c2_ecmin_c: Float64
    var c2_dept_c: Float64
    var c2_ecmin_t: Float64
    var c2_dept_t: Float64

    fn __init__(out self):
        self.eps_c = 0.0
        self.sig_c = 0.0
        self.tangent_c = 0.0
        self.eps_p_c = 0.0
        self.alpha_c = 0.0
        self.eps_t = 0.0
        self.sig_t = 0.0
        self.tangent_t = 0.0
        self.eps_p_t = 0.0
        self.alpha_t = 0.0
        self.min_strain_c = 0.0
        self.end_strain_c = 0.0
        self.unload_slope_c = 0.0
        self.min_strain_t = 0.0
        self.end_strain_t = 0.0
        self.unload_slope_t = 0.0
        self.s2_epsmin_c = 0.0
        self.s2_epsmax_c = 0.0
        self.s2_epspl_c = 0.0
        self.s2_epss0_c = 0.0
        self.s2_sigs0_c = 0.0
        self.s2_epsr_c = 0.0
        self.s2_sigr_c = 0.0
        self.s2_kon_c = 0
        self.s2_epsmin_t = 0.0
        self.s2_epsmax_t = 0.0
        self.s2_epspl_t = 0.0
        self.s2_epss0_t = 0.0
        self.s2_sigs0_t = 0.0
        self.s2_epsr_t = 0.0
        self.s2_sigr_t = 0.0
        self.s2_kon_t = 0
        self.c2_ecmin_c = 0.0
        self.c2_dept_c = 0.0
        self.c2_ecmin_t = 0.0
        self.c2_dept_t = 0.0

    fn __init__(out self, mat_def: UniMaterialDef):
        self.eps_c = 0.0
        self.sig_c = 0.0
        self.eps_p_c = 0.0
        self.alpha_c = 0.0
        self.eps_t = 0.0
        self.sig_t = 0.0
        self.eps_p_t = 0.0
        self.alpha_t = 0.0
        self.min_strain_c = 0.0
        self.end_strain_c = 0.0
        self.unload_slope_c = 0.0
        self.min_strain_t = 0.0
        self.end_strain_t = 0.0
        self.unload_slope_t = 0.0

        self.s2_epsmin_c = 0.0
        self.s2_epsmax_c = 0.0
        self.s2_epspl_c = 0.0
        self.s2_epss0_c = 0.0
        self.s2_sigs0_c = 0.0
        self.s2_epsr_c = 0.0
        self.s2_sigr_c = 0.0
        self.s2_kon_c = 0
        self.s2_epsmin_t = 0.0
        self.s2_epsmax_t = 0.0
        self.s2_epspl_t = 0.0
        self.s2_epss0_t = 0.0
        self.s2_sigs0_t = 0.0
        self.s2_epsr_t = 0.0
        self.s2_sigr_t = 0.0
        self.s2_kon_t = 0

        self.c2_ecmin_c = 0.0
        self.c2_dept_c = 0.0
        self.c2_ecmin_t = 0.0
        self.c2_dept_t = 0.0

        var tangent = uni_mat_initial_tangent(mat_def)
        self.tangent_c = tangent
        self.tangent_t = tangent

        if mat_def.mat_type == UniMaterialTypeTag.Concrete01:
            self.unload_slope_c = tangent
            self.unload_slope_t = tangent

        if mat_def.mat_type == UniMaterialTypeTag.Steel02:
            var Fy = mat_def.p0
            var E0 = mat_def.p1
            var epsy = Fy / E0
            self.s2_epsmax_c = epsy
            self.s2_epsmax_t = epsy
            self.s2_epsmin_c = -epsy
            self.s2_epsmin_t = -epsy
            var sigini = mat_def.p10
            if sigini != 0.0:
                var epsini = sigini / E0
                self.eps_c = epsini
                self.eps_t = epsini
                self.sig_c = sigini
                self.sig_t = sigini


fn uni_mat_is_elastic(mat_def: UniMaterialDef) -> Bool:
    return mat_def.mat_type == UniMaterialTypeTag.Elastic


fn uni_mat_initial_tangent(mat_def: UniMaterialDef) -> Float64:
    if mat_def.mat_type == UniMaterialTypeTag.Elastic:
        return mat_def.p0
    if mat_def.mat_type == UniMaterialTypeTag.Steel01:
        return mat_def.p1
    if mat_def.mat_type == UniMaterialTypeTag.Concrete01:
        return (2.0 * mat_def.p0) / mat_def.p1
    if mat_def.mat_type == UniMaterialTypeTag.Steel02:
        return mat_def.p1
    if mat_def.mat_type == UniMaterialTypeTag.Concrete02:
        return (2.0 * mat_def.p0) / mat_def.p1
    abort("unsupported uniaxial material")
    return 0.0

