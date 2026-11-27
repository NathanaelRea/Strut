from collections import InlineArray, List
from materials.fsam import FSAMState
from materials.uniaxial.mod_imk_peak_oriented import IMKPeakHistory, IMKPeakParameters
from materials.uniaxial.bilin import BilinHistory, BilinParameters
from materials.uniaxial.hysteretic import HystereticHistory, HystereticParameters
from memory import ArcPointer
from os import abort
from sys.intrinsics import _type_is_eq
from materials.uniaxial.py_simple1 import py_simple1_init_state, py_simple1_initial_tangent
from materials.uniaxial.py_tz_qz_state import PyTzQzState
from materials.uniaxial.qz_simple1 import qz_simple1_init_state, qz_simple1_initial_tangent
from materials.uniaxial.tz_simple1 import tz_simple1_init_state, tz_simple1_initial_tangent
from materials.uniaxial.fatigue import FatigueHistory, FatigueParameters
from materials.uniaxial.concrete_cm import ConcreteCMHistory, ConcreteCMParameters
from materials.uniaxial.steel_mpf import SteelMPFHistory, SteelMPFParameters
from materials.uniaxial.wall_history import WallMaterialHistory
from materials.uniaxial.min_max import MinMaxHistory, StrainLimits
from materials.uniaxial.ratchet import RatchetHistory, RatchetParameters
from simd_math import pow_nonneg
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
    return pow_nonneg(_abs(x), p)


struct StressTangent(Defaultable, ImplicitlyCopyable, Movable):
    var stress: Float64
    var tangent: Float64

    fn __init__(out self):
        self.stress = 0.0
        self.tangent = 0.0

    fn __init__(out self, stress: Float64, tangent: Float64):
        self.stress = stress
        self.tangent = tangent


struct EndStrainSlope(Defaultable, ImplicitlyCopyable, Movable):
    var end_strain: Float64
    var unload_slope: Float64

    fn __init__(out self):
        self.end_strain = 0.0
        self.unload_slope = 0.0

    fn __init__(out self, end_strain: Float64, unload_slope: Float64):
        self.end_strain = end_strain
        self.unload_slope = unload_slope


# The leaf layouts exclude child storage. Specializing the extension keeps
# the usual material fields and operations shared without recursive types.
alias MaterialExtension = Defaultable & ImplicitlyCopyable & Movable


struct UniMaterialDefinition[Components: MaterialExtension](
    Defaultable, ImplicitlyCopyable, Movable
):
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
    var p11: Float64
    var bilin: BilinParameters
    var imk_peak: IMKPeakParameters
    var hysteretic: HystereticParameters
    var fatigue: FatigueParameters
    var strain_limits: Optional[StrainLimits]
    var components: Self.Components

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
        self.p11 = 0.0
        self.bilin = BilinParameters()
        self.imk_peak = IMKPeakParameters()
        self.hysteretic = HystereticParameters()
        self.fatigue = FatigueParameters()
        self.strain_limits = None
        self.components = Self.Components()

    fn __init__(out self, mat_type: Int, p0: Float64, p1: Float64, p2: Float64, p3: Float64):
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
        self.p11 = 0.0
        self.bilin = BilinParameters()
        self.imk_peak = IMKPeakParameters()
        self.hysteretic = HystereticParameters()
        self.fatigue = FatigueParameters()
        self.strain_limits = None
        self.components = Self.Components()

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
        p11: Float64 = 0.0,
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
        self.p11 = p11
        self.bilin = BilinParameters()
        self.imk_peak = IMKPeakParameters()
        self.hysteretic = HystereticParameters()
        self.fatigue = FatigueParameters()
        self.strain_limits = None
        self.components = Self.Components()

    fn ratchet_params(self) -> RatchetParameters:
        return RatchetParameters(self.p0, self.p1, self.p2, Int(self.p3))

    fn concrete_cm_params(self) -> ConcreteCMParameters:
        var params = ConcreteCMParameters()
        params.fpcc = self.p0
        params.epcc = self.p1
        params.Ec = self.p2
        params.rc = self.p3
        params.xcrn = self.p4
        params.ft = self.p5
        params.et = self.p6
        params.rt = self.p7
        params.xcrp = self.p8
        params.gap_close = self.p9
        params.monotonic = self.p10
        return params

    fn steel_mpf_params(self) -> SteelMPFParameters:
        var params = SteelMPFParameters()
        params.fyp = self.p0
        params.fyn = self.p1
        params.E0 = self.p2
        params.bp = self.p3
        params.bn = self.p4
        params.R0 = self.p5
        params.a1 = self.p6
        params.a2 = self.p7
        params.a3 = self.p8
        params.a4 = self.p9
        params.a5 = self.p10
        params.a6 = self.p11
        return params

    fn __init__(out self, mat_type: Int, values: InlineArray[Float64, 24]):
        self = Self()
        self.mat_type = mat_type
        if mat_type == UniMaterialTypeTag.Bilin:
            self.bilin = BilinParameters(values)
        elif mat_type == UniMaterialTypeTag.ModIMKPeakOriented:
            self.imk_peak = IMKPeakParameters(values)
        else:
            self.hysteretic = HystereticParameters(values)


alias UniMaterialLeafDef = UniMaterialDefinition[Bool]


@fieldwise_init
struct ParallelComponent(ImplicitlyCopyable, Movable):
    var definition: UniMaterialLeafDef
    var factor: Float64


alias UniMaterialDef = UniMaterialDefinition[Optional[ArcPointer[List[ParallelComponent]]]]


struct CoupledForceHistory(ImplicitlyCopyable, Movable):
    var committed: Tuple[Float64, Float64]
    var trial: Tuple[Float64, Float64]

    fn __init__(out self):
        self.committed = (0.0, 0.0)
        self.trial = (0.0, 0.0)


struct UniMaterialPointState[Components: MaterialExtension](
    Defaultable, ImplicitlyCopyable, Movable
):
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
    var wall_history: WallMaterialHistory
    var min_max: Optional[MinMaxHistory]
    var coupled_force: Optional[CoupledForceHistory]
    var ratchet: Optional[RatchetHistory]
    var retain_trial_tangent: Bool
    var retain_trial_response: Bool
    var fsam: List[FSAMState]
    var fsam_components: Optional[ArcPointer[List[ParallelComponent]]]

    var bilin_history: List[BilinHistory]
    var imk_peak_history: List[IMKPeakHistory]
    var imk_peak_has_commit: Bool
    var bilin_has_commit: Bool
    var hysteretic_c: HystereticHistory
    var hysteretic_t: HystereticHistory
    # PySimple1 / TzSimple1 / QzSimple1 history variables
    var has_py_tz_qz: Bool
    var py_tz_qz: PyTzQzState
    # ViscousDamper history variables
    var vd_vel_c: Float64
    var vd_pugr_c: Float64
    var vd_nugr_c: Float64
    var vd_vel_t: Float64
    var vd_pugr_t: Float64
    var vd_nugr_t: Float64
    var fatigue: FatigueHistory
    var components: Self.Components

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
        self.wall_history = WallMaterialHistory()
        self.min_max = None
        self.coupled_force = None
        self.ratchet = None
        self.retain_trial_tangent = False
        self.retain_trial_response = False
        self.fsam = []
        self.fsam_components = None
        self.bilin_history = []
        self.imk_peak_history = []
        self.imk_peak_has_commit = False
        self.bilin_has_commit = False
        self.hysteretic_c = HystereticHistory()
        self.hysteretic_t = HystereticHistory()
        self.has_py_tz_qz = False
        self.py_tz_qz = PyTzQzState()
        self.vd_vel_c = 0.0
        self.vd_pugr_c = 0.0
        self.vd_nugr_c = 0.0
        self.vd_vel_t = 0.0
        self.vd_pugr_t = 0.0
        self.vd_nugr_t = 0.0
        self.fatigue = FatigueHistory()
        self.components = Self.Components()

    fn __init__(out self, mat_def: UniMaterialDefinition):
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
        self.wall_history = WallMaterialHistory()
        self.min_max = None
        self.coupled_force = None
        self.ratchet = None
        self.fsam = []
        self.fsam_components = None
        self.bilin_history = []
        self.imk_peak_history = []
        self.imk_peak_has_commit = False
        self.bilin_has_commit = False
        self.hysteretic_c = HystereticHistory()
        self.hysteretic_t = HystereticHistory()

        if mat_def.mat_type == UniMaterialTypeTag.Bilin:
            # Committed and trial histories are allocated only for Bilin.
            self.bilin_history.append(BilinHistory())
            self.bilin_history.append(BilinHistory())
        if mat_def.mat_type == UniMaterialTypeTag.ModIMKPeakOriented:
            self.imk_peak_history = [
                IMKPeakHistory(mat_def.imk_peak),
                IMKPeakHistory(mat_def.imk_peak),
            ]

        self.has_py_tz_qz = False
        self.py_tz_qz = PyTzQzState()

        self.vd_vel_c = 0.0
        self.vd_pugr_c = 0.0
        self.vd_nugr_c = 0.0
        self.vd_vel_t = 0.0
        self.vd_pugr_t = 0.0
        self.vd_nugr_t = 0.0
        self.fatigue = FatigueHistory()
        self.components = Self.Components()
        self.fatigue.params = mat_def.fatigue

        var tangent = (
            0.0 if mat_def.mat_type
            == UniMaterialTypeTag.Bilin else uni_mat_initial_tangent(mat_def)
        )
        self.tangent_c = tangent
        self.tangent_t = tangent
        self.fatigue.initial_tangent = tangent
        self.retain_trial_tangent = (
            mat_def.mat_type == UniMaterialTypeTag.Hysteretic
            or mat_def.mat_type == UniMaterialTypeTag.ElasticPP
        )
        self.retain_trial_response = mat_def.mat_type == UniMaterialTypeTag.ElasticBilin
        if self.retain_trial_response and mat_def.p2 == 0.0:
            self.tangent_t = mat_def.p1
        if mat_def.mat_type == UniMaterialTypeTag.Ratchet:
            self.ratchet = RatchetHistory(mat_def.ratchet_params())
            self.tangent_t = mat_def.p0
            self.tangent_c = mat_def.p0
        if mat_def.strain_limits:
            self.min_max = MinMaxHistory(mat_def.strain_limits.value(), tangent)

        if mat_def.mat_type == UniMaterialTypeTag.ConcreteCM:
            self.wall_history.concrete = [ConcreteCMHistory(), ConcreteCMHistory()]
        if mat_def.mat_type == UniMaterialTypeTag.SteelMPF:
            self.wall_history.steel = [SteelMPFHistory(), SteelMPFHistory()]

        if (
            mat_def.mat_type == UniMaterialTypeTag.Concrete01
            or mat_def.mat_type == UniMaterialTypeTag.Steel01Thermal
        ):
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

        if mat_def.mat_type == UniMaterialTypeTag.PySimple1:
            self.has_py_tz_qz = True
            var t0 = py_simple1_init_state(
                mat_def.p0, mat_def.p1, mat_def.p2, mat_def.p3, mat_def.p4, self.py_tz_qz
            )
            self.tangent_c = t0
            self.tangent_t = t0
        elif mat_def.mat_type == UniMaterialTypeTag.TzSimple1:
            self.has_py_tz_qz = True
            var t0 = tz_simple1_init_state(
                mat_def.p0, mat_def.p1, mat_def.p2, mat_def.p3, self.py_tz_qz
            )
            self.tangent_c = t0
            self.tangent_t = t0
        elif mat_def.mat_type == UniMaterialTypeTag.QzSimple1:
            self.has_py_tz_qz = True
            var t0 = qz_simple1_init_state(
                mat_def.p0, mat_def.p1, mat_def.p2, mat_def.p3, mat_def.p4, self.py_tz_qz
            )
            self.tangent_c = t0
            self.tangent_t = t0

        @parameter
        if not _type_is_eq[Self.Components, Bool]():
            if mat_def.mat_type == UniMaterialTypeTag.Parallel:
                var components = rebind[Optional[ArcPointer[List[ParallelComponent]]]](
                    mat_def.components
                )
                ref definitions = components.value()[]
                var states = List[UniMaterialLeafState](capacity=len(definitions))
                for i in range(len(definitions)):
                    var child = UniMaterialLeafState(definitions[i].definition)
                    self.sig_t += definitions[i].factor * child.sig_t
                    states.append(child^)
                self.sig_c = self.sig_t
                self.components = rebind[Self.Components](Optional(states^))

    fn __copyinit__(out self, existing: Self):
        self.eps_c = existing.eps_c
        self.sig_c = existing.sig_c
        self.tangent_c = existing.tangent_c
        self.eps_p_c = existing.eps_p_c
        self.alpha_c = existing.alpha_c
        self.eps_t = existing.eps_t
        self.sig_t = existing.sig_t
        self.tangent_t = existing.tangent_t
        self.eps_p_t = existing.eps_p_t
        self.alpha_t = existing.alpha_t
        self.min_strain_c = existing.min_strain_c
        self.end_strain_c = existing.end_strain_c
        self.unload_slope_c = existing.unload_slope_c
        self.min_strain_t = existing.min_strain_t
        self.end_strain_t = existing.end_strain_t
        self.unload_slope_t = existing.unload_slope_t
        self.s2_epsmin_c = existing.s2_epsmin_c
        self.s2_epsmax_c = existing.s2_epsmax_c
        self.s2_epspl_c = existing.s2_epspl_c
        self.s2_epss0_c = existing.s2_epss0_c
        self.s2_sigs0_c = existing.s2_sigs0_c
        self.s2_epsr_c = existing.s2_epsr_c
        self.s2_sigr_c = existing.s2_sigr_c
        self.s2_kon_c = existing.s2_kon_c
        self.s2_epsmin_t = existing.s2_epsmin_t
        self.s2_epsmax_t = existing.s2_epsmax_t
        self.s2_epspl_t = existing.s2_epspl_t
        self.s2_epss0_t = existing.s2_epss0_t
        self.s2_sigs0_t = existing.s2_sigs0_t
        self.s2_epsr_t = existing.s2_epsr_t
        self.s2_sigr_t = existing.s2_sigr_t
        self.s2_kon_t = existing.s2_kon_t
        self.c2_ecmin_c = existing.c2_ecmin_c
        self.c2_dept_c = existing.c2_dept_c
        self.c2_ecmin_t = existing.c2_ecmin_t
        self.c2_dept_t = existing.c2_dept_t
        self.bilin_history = existing.bilin_history.copy()
        self.imk_peak_history = existing.imk_peak_history.copy()
        self.imk_peak_has_commit = existing.imk_peak_has_commit
        self.bilin_has_commit = existing.bilin_has_commit
        self.hysteretic_c = existing.hysteretic_c
        self.hysteretic_t = existing.hysteretic_t
        self.has_py_tz_qz = existing.has_py_tz_qz
        self.py_tz_qz = existing.py_tz_qz
        self.vd_vel_c = existing.vd_vel_c
        self.vd_pugr_c = existing.vd_pugr_c
        self.vd_nugr_c = existing.vd_nugr_c
        self.vd_vel_t = existing.vd_vel_t
        self.vd_pugr_t = existing.vd_pugr_t
        self.vd_nugr_t = existing.vd_nugr_t
        self.fatigue = existing.fatigue
        self.components = existing.components
        self.wall_history = existing.wall_history
        self.min_max = existing.min_max
        self.coupled_force = existing.coupled_force
        self.ratchet = existing.ratchet
        self.retain_trial_tangent = existing.retain_trial_tangent
        self.retain_trial_response = existing.retain_trial_response
        self.fsam = existing.fsam.copy()
        self.fsam_components = existing.fsam_components


alias UniMaterialLeafState = UniMaterialPointState[Bool]
alias UniMaterialState = UniMaterialPointState[Optional[List[UniMaterialLeafState]]]


fn uni_mat_is_elastic(mat_def: UniMaterialDef) -> Bool:
    return mat_def.mat_type == UniMaterialTypeTag.Elastic and not mat_def.strain_limits


fn uni_mat_initial_tangent(mat_def: UniMaterialDefinition) -> Float64:
    if (
        mat_def.mat_type == UniMaterialTypeTag.Elastic
        or mat_def.mat_type == UniMaterialTypeTag.ENT
        or mat_def.mat_type == UniMaterialTypeTag.ElasticPP
        or mat_def.mat_type == UniMaterialTypeTag.ElasticBilin
        or mat_def.mat_type == UniMaterialTypeTag.Parallel
    ):
        return mat_def.p0
    if (
        mat_def.mat_type == UniMaterialTypeTag.Steel01
        or mat_def.mat_type == UniMaterialTypeTag.Steel01Thermal
    ):
        return mat_def.p1
    if mat_def.mat_type == UniMaterialTypeTag.Concrete01:
        return (2.0 * mat_def.p0) / mat_def.p1
    if mat_def.mat_type == UniMaterialTypeTag.Steel02:
        return mat_def.p1
    if mat_def.mat_type == UniMaterialTypeTag.Concrete02:
        return (2.0 * mat_def.p0) / mat_def.p1
    if mat_def.mat_type == UniMaterialTypeTag.ModIMKPeakOriented:
        return mat_def.imk_peak.initial_tangent()
    if mat_def.mat_type == UniMaterialTypeTag.Bilin:
        return mat_def.bilin.Ke0 * (1.0 + mat_def.bilin.nFactor)
    if mat_def.mat_type == UniMaterialTypeTag.Hysteretic:
        return mat_def.hysteretic.positive.slope1
    if mat_def.mat_type == UniMaterialTypeTag.PySimple1:
        return py_simple1_initial_tangent(
            mat_def.p0, mat_def.p1, mat_def.p2, mat_def.p3, mat_def.p4
        )
    if mat_def.mat_type == UniMaterialTypeTag.TzSimple1:
        return tz_simple1_initial_tangent(mat_def.p0, mat_def.p1, mat_def.p2, mat_def.p3)
    if mat_def.mat_type == UniMaterialTypeTag.QzSimple1:
        return qz_simple1_initial_tangent(
            mat_def.p0, mat_def.p1, mat_def.p2, mat_def.p3, mat_def.p4
        )
    if mat_def.mat_type == UniMaterialTypeTag.ViscousDamper:
        return 0.0
    if mat_def.mat_type == UniMaterialTypeTag.HystereticSmooth:
        return mat_def.p1 - 2.0 * mat_def.p3 + 0.5 * (mat_def.p0 - mat_def.p1)
    if mat_def.mat_type == UniMaterialTypeTag.HystereticAsym:
        return 0.5 * (mat_def.p0 - mat_def.p1)
    if (
        mat_def.mat_type == UniMaterialTypeTag.ConcreteCM
        or mat_def.mat_type == UniMaterialTypeTag.SteelMPF
    ):
        return mat_def.p2
    if mat_def.mat_type == UniMaterialTypeTag.Ratchet:
        return mat_def.ratchet_params().initial_tangent()
    if mat_def.mat_type == UniMaterialTypeTag.Viscous:
        return 0.0
    abort("unsupported uniaxial material")
    return 0.0
