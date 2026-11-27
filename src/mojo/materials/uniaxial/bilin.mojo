# OpenSees - Open System for Earthquake Engineering Simulation
# Pacific Earthquake Engineering Research Center
# (C) Copyright 1999, The Regents of the University of California
# All Rights Reserved.
# Commercial use of this program without express permission of the
# University of California, Berkeley, is strictly prohibited.  See
# file 'COPYRIGHT'  in main directory for information on usage and
# redistribution,  and for a DISCLAIMER OF ALL WARRANTIES.
# Developed by:
# Frank McKenna (fmckenna@ce.berkeley.edu)
# Gregory L. Fenves (fenves@ce.berkeley.edu)
# Filip C. Filippou (filippou@ce.berkeley.edu)
#
# Port of OpenSees 3.7.1 SRC/material/uniaxial/Bilin.cpp.
# Modified Ibarra-Medina-Krawinkler model by Dimitrios G. Lignos and
# Theodore Karavasilis, adapted by Filipe Ribeiro and Andre Barbosa.
# The names of the history quantities follow Bilin.cpp for comparison.

from collections import InlineArray


struct BilinParameters(ImplicitlyCopyable, Movable):
    var Ke0: Float64
    var AsPos: Float64
    var AsNeg: Float64
    var My_pos: Float64
    var My_neg: Float64
    var LamdaS: Float64
    var LamdaD: Float64
    var LamdaA: Float64
    var LamdaK: Float64
    var Cs: Float64
    var Cd: Float64
    var Ca: Float64
    var Ck: Float64
    var Thetap_pos: Float64
    var Thetap_neg: Float64
    var Thetapc_pos: Float64
    var Thetapc_neg: Float64
    var KPos: Float64
    var KNeg: Float64
    var Thetau_pos: Float64
    var Thetau_neg: Float64
    var PDPlus: Float64
    var PDNeg: Float64
    var nFactor: Float64

    fn __init__(out self):
        self = Self(InlineArray[Float64, 24](fill=0.0))

    fn __init__(out self, values: InlineArray[Float64, 24]):
        self.Ke0 = values[0]
        self.AsPos = values[1]
        self.AsNeg = values[2]
        self.My_pos = values[3]
        self.My_neg = values[4]
        self.LamdaS = values[5]
        self.LamdaD = values[6]
        self.LamdaA = values[7]
        self.LamdaK = values[8]
        self.Cs = values[9]
        self.Cd = values[10]
        self.Ca = values[11]
        self.Ck = values[12]
        self.Thetap_pos = values[13]
        self.Thetap_neg = values[14]
        self.Thetapc_pos = values[15]
        self.Thetapc_neg = values[16]
        self.KPos = values[17]
        self.KNeg = values[18]
        self.Thetau_pos = values[19]
        self.Thetau_neg = values[20]
        self.PDPlus = values[21]
        self.PDNeg = values[22]
        self.nFactor = values[23]


struct BilinHistory(ImplicitlyCopyable, Movable):
    var U: Float64
    var Tangent: Float64
    var dNewLoadPos: Float64
    var dNewLoadNeg: Float64
    var flagdeg: Int
    var flagstopdeg: Int
    var interup: Int
    var kon: Int
    var iNoFneg: Int
    var iNoFpos: Int
    var LP: Int
    var LN: Int
    var capSlope: Float64
    var dmax: Float64
    var dmin: Float64
    var Enrgtot: Float64
    var Enrgc: Float64
    var fyPos: Float64
    var fLimNeg: Float64
    var fyNeg: Float64
    var ekP: Float64
    var ekunload: Float64
    var sp: Float64
    var sn: Float64
    var dP: Float64
    var fP: Float64
    var ek: Float64
    var dLimPos: Float64
    var dLimNeg: Float64
    var cpPos: Float64
    var cpNeg: Float64
    var fLimPos: Float64
    var ekexcurs: Float64
    var RSE: Float64
    var fPeakPos: Float64
    var fPeakNeg: Float64
    var alphaNeg: Float64
    var alphaPos: Float64
    var ekhardNeg: Float64
    var ekhardPos: Float64
    var fCapRefPos: Float64
    var fCapRefNeg: Float64
    var Enrgts: Float64
    var Enrgtk: Float64
    var Enrgtd: Float64
    var dyPos: Float64
    var dyNeg: Float64
    var resSnHor: Float64
    var fmax: Float64
    var fmin: Float64
    var resSp: Float64
    var resSn: Float64
    var fCapPos: Float64
    var fCapNeg: Float64
    var snHor: Float64
    var spHor: Float64
    var resSpHor: Float64
    var snEnv: Float64
    var resSnEnv: Float64
    var spEnv: Float64
    var resSpEnv: Float64
    var capSlopeOrig: Float64
    var capSlopeNeg: Float64
    var flagControlResponse: Int
    var capSlopeOrigNeg: Float64
    var capSlopeNegMember: Float64
    var capSlopeMember: Float64
    var Ke: Float64
    var prodBeta: Float64

    fn __init__(out self):
        self.U = 0.0
        self.Tangent = 0.0
        self.dNewLoadPos = 0.0
        self.dNewLoadNeg = 0.0
        self.flagdeg = 0
        self.flagstopdeg = 0
        self.interup = 0
        self.kon = 0
        self.iNoFneg = 0
        self.iNoFpos = 0
        self.LP = 0
        self.LN = 0
        self.capSlope = 0.0
        self.dmax = 0.0
        self.dmin = 0.0
        self.Enrgtot = 0.0
        self.Enrgc = 0.0
        self.fyPos = 0.0
        self.fLimNeg = 0.0
        self.fyNeg = 0.0
        self.ekP = 0.0
        self.ekunload = 0.0
        self.sp = 0.0
        self.sn = 0.0
        self.dP = 0.0
        self.fP = 0.0
        self.ek = 0.0
        self.dLimPos = 0.0
        self.dLimNeg = 0.0
        self.cpPos = 0.0
        self.cpNeg = 0.0
        self.fLimPos = 0.0
        self.ekexcurs = 0.0
        self.RSE = 0.0
        self.fPeakPos = 0.0
        self.fPeakNeg = 0.0
        self.alphaNeg = 0.0
        self.alphaPos = 0.0
        self.ekhardNeg = 0.0
        self.ekhardPos = 0.0
        self.fCapRefPos = 0.0
        self.fCapRefNeg = 0.0
        self.Enrgts = 0.0
        self.Enrgtk = 0.0
        self.Enrgtd = 0.0
        self.dyPos = 0.0
        self.dyNeg = 0.0
        self.resSnHor = 0.0
        self.fmax = 0.0
        self.fmin = 0.0
        self.resSp = 0.0
        self.resSn = 0.0
        self.fCapPos = 0.0
        self.fCapNeg = 0.0
        self.snHor = 0.0
        self.spHor = 0.0
        self.resSpHor = 0.0
        self.snEnv = 0.0
        self.resSnEnv = 0.0
        self.spEnv = 0.0
        self.resSpEnv = 0.0
        self.capSlopeOrig = 0.0
        self.capSlopeNeg = 0.0
        self.flagControlResponse = 0
        self.capSlopeOrigNeg = 0.0
        self.capSlopeNegMember = 0.0
        self.capSlopeMember = 0.0
        self.Ke = 0.0
        self.prodBeta = 1.0

    fn set_trial(
        mut self,
        params: BilinParameters,
        committed_strain: Float64,
        has_commit: Bool,
        strain: Float64,
    ):
        var deltaD: Float64
        var d: Float64
        var temp_1: Float64 = 0.0
        var temp: Float64 = 0.0
        var betas: Float64
        var betak: Float64
        var betad: Float64
        var dBoundPos: Float64
        var dBoundNeg: Float64
        var f1: Float64
        var f2: Float64
        var fNewLoadPos: Float64 = 0.0
        var fNewLoadNeg: Float64 = 0.0
        var f: Float64
        var xDevPos1: Float64 = 0.0
        var yDevPos1: Float64 = 0.0
        var xDevPos2: Float64 = 0.0
        var yDevPos2: Float64 = 0.0
        var xDevPos: Float64
        var yDevPos: Float64
        var xDevNeg1: Float64 = 0.0
        var yDevNeg1: Float64 = 0.0
        var xDevNeg2: Float64 = 0.0
        var yDevNeg2: Float64 = 0.0
        var xDevNeg: Float64
        var yDevNeg: Float64
        f = 0.0
        self.U = strain
        deltaD = strain - committed_strain
        d = strain
        self.dP = committed_strain
        if abs(deltaD) < 2.220446049250313e-16 and Int(has_commit) == 1:
            return
        if d > 0.0:
            intersect_lines(
                temp_1,
                temp,
                0.0,
                self.fCapRefPos,
                self.capSlope * self.Ke,
                0.0,
                params.KPos * params.My_pos,
                0.0,
            )
            if d < temp_1:
                self.iNoFpos = 0
                self.LP = 0
        else:
            intersect_lines(
                temp_1,
                temp,
                0.0,
                self.fCapRefNeg,
                self.capSlopeNeg * self.Ke,
                0.0,
                params.KNeg * params.My_neg,
                0.0,
            )
            if d > temp_1:
                self.iNoFneg = 0
                self.LN = 0
        self.flagdeg = 0
        betas = 0.0
        betak = 0.0
        betad = 0.0
        if self.kon == 0:
            self.Ke = params.Ke0 * (1 + params.nFactor)
            self.alphaNeg = params.AsNeg / (1 + params.nFactor * (1 - params.AsNeg))
            self.alphaPos = params.AsPos / (1 + params.nFactor * (1 - params.AsPos))
            self.ekhardPos = self.Ke * params.AsPos / (1 + params.nFactor * (1 - params.AsPos))
            self.ekhardNeg = self.Ke * params.AsNeg / (1 + params.nFactor * (1 - params.AsNeg))
            self.capSlopeMember = -(params.My_pos + self.ekhardPos * params.Thetap_pos) / (
                params.Thetapc_pos * params.Ke0
            )
            self.capSlope = self.capSlopeMember / (1 + params.nFactor * (1 - self.capSlopeMember))
            self.capSlopeNegMember = -(-params.My_neg + self.ekhardNeg * params.Thetap_neg) / (
                params.Thetapc_neg * params.Ke0
            )
            self.capSlopeNeg = self.capSlopeNegMember / (
                1 + params.nFactor * (1 - self.capSlopeNegMember)
            )
            self.prodBeta = 1.0
            self.ekP = self.Ke
            self.flagControlResponse = 0
            self.Tangent = self.Ke
            self.Enrgts = params.My_pos * params.LamdaS
            self.Enrgtk = 2.0 * params.My_pos * params.LamdaK
            self.Enrgtd = params.My_pos * params.LamdaD
            self.dmax = params.My_pos / self.Ke
            self.dmin = params.My_neg / self.Ke
            self.ekP = self.Ke
            self.ekunload = self.Ke
            self.ekexcurs = self.Ke
            self.Enrgtot = 0.0
            self.Enrgc = 0.0
            self.fyPos = params.My_pos
            self.fyNeg = params.My_neg
            self.dyPos = params.My_pos / self.Ke
            self.dyNeg = params.My_neg / self.Ke
            self.resSn = params.My_pos
            self.resSp = params.My_neg
            self.cpPos = params.Thetap_pos + params.My_pos / self.Ke
            self.cpNeg = -params.Thetap_neg + params.My_neg / self.Ke
            self.fPeakPos = params.My_pos + self.ekhardPos * (
                (params.Thetap_pos + params.My_pos / self.Ke) - params.My_pos / self.Ke
            )
            self.fPeakNeg = params.My_neg + self.ekhardNeg * (
                (-params.Thetap_neg + params.My_neg / self.Ke) - (params.My_neg / self.Ke)
            )
            if self.cpPos < params.My_pos / self.Ke:
                self.fPeakPos = params.My_pos * self.cpPos / (params.My_pos / self.Ke)
            if self.cpNeg > (params.My_neg / self.Ke):
                self.fPeakNeg = params.My_neg * self.cpNeg / (params.My_neg / self.Ke)
            self.fCapPos = self.fPeakPos
            self.fCapNeg = self.fPeakNeg
            self.LP = 0
            self.LN = 0
            self.fLimPos = 0
            self.fLimNeg = 0
            self.dLimPos = 0
            self.dLimNeg = 0
            self.iNoFpos = 0
            self.iNoFneg = 0
            self.interup = 0
            self.fCapRefPos = (
                -self.capSlope * self.Ke * (params.Thetap_pos + params.My_pos / self.Ke)
                + self.fPeakPos
            )
            self.fCapRefNeg = (
                -self.capSlopeNeg * self.Ke * (-params.Thetap_neg + params.My_neg / self.Ke)
                + self.fPeakNeg
            )
            self.capSlopeOrig = self.capSlope
            self.capSlopeOrigNeg = self.capSlopeNeg
            self.flagstopdeg = 0
            self.fmin = 0.0
            self.fmax = 0.0
            self.RSE = 0.0
            if deltaD >= 0.0:
                self.kon = 1
            else:
                self.kon = 2
        if deltaD >= 0.0:
            if self.iNoFpos == 1:
                intersect_lines(
                    self.dNewLoadPos,
                    fNewLoadPos,
                    self.dyNeg,
                    self.fyNeg,
                    self.ekhardNeg,
                    0.0,
                    0.0,
                    0.0,
                )
            if self.kon == 2:
                self.kon = 1
                if self.dP <= self.dmin:
                    self.fLimNeg = self.fP
                    self.dLimNeg = self.dP
                if self.resSn > 0.0:
                    self.RSE = 0.5 * self.fP * self.fP / self.ekunload
                else:
                    self.RSE = 0.5 * (self.fP + self.resSn) * (
                        self.sn - self.dP
                    ) + 0.5 * self.resSn * self.resSn / (self.Ke * self.alphaPos)
                var a2: Float64 = self.Enrgtk - (self.Enrgtot - self.RSE)
                if (a2 <= 0.0) and (self.Enrgtk != 0.0):
                    self.flagControlResponse = 1
                if (
                    (params.LamdaK != 0.0)
                    and (d < self.sp)
                    and abs(self.capSlope * (1 + params.nFactor)) >= 1.0e-3
                    and (abs(self.capSlopeNeg * (1 + params.nFactor)) >= 1.0e-3)
                ):
                    betak = (
                        ((self.Enrgc - self.RSE) / (self.Enrgtk - (self.Enrgtot - self.RSE)))
                    ) ** params.Ck
                    if ((self.Enrgtot - self.RSE) >= self.Enrgtk) or (betak >= 1.0):
                        betak = 1.0
                    if self.flagstopdeg != 1 and self.flagdeg != 1:
                        self.ekunload = self.Ke * (
                            (self.prodBeta * (1.0 - betak))
                            / (1 + params.nFactor * (1 - (self.prodBeta * (1 - betak))))
                        )
                    else:
                        self.ekunload = self.ekexcurs
                    if self.ekunload <= (0.1 / (1 + params.nFactor * (1 - 0.1))) * self.Ke:
                        self.ekunload = (0.1 / (1 + params.nFactor * (1 - 0.1))) * self.Ke
                if (self.dmin < (params.My_neg / self.Ke)) or (
                    self.dmax > (params.My_pos / self.Ke)
                ):
                    if (self.dP < self.sp) or (
                        ((self.dP > self.sp) and (self.ekP == self.ekunload))
                    ):
                        self._positive_reload_intersection(params)
                        if (abs(self.dmax - (params.My_pos / self.Ke)) >= 1.0e-10) and (
                            abs(self.sn - (params.My_pos / self.Ke)) <= 1.0e-10
                        ):
                            self.sn = (params.My_pos / self.Ke) - 1.0e-9
                if self.sn > self.dmax:
                    self.dmax = self.sn
                if (self.iNoFneg == 1) and (self.dP <= self.dNewLoadNeg):
                    self.sn = self.dNewLoadNeg - 1.0e-6
                    self.resSn = 0
            if (self.iNoFneg == 1) and (self.iNoFpos == 1) and (d < params.Thetau_pos):
                f = params.KPos * params.My_pos
                self.ek = 1.0e-7
            elif (
                (self.iNoFneg == 1) and (self.iNoFpos == 1) and (d >= params.Thetau_pos)
            ) or self.flagControlResponse == 1:
                f = 1.0e-10
                self.ek = 1.0e-7
                self.flagstopdeg = 1
            elif (self.iNoFpos == 1) and (d > self.sn) and (d < params.Thetau_pos):
                f = params.KPos * params.My_pos
                self.ek = 1.0e-7
            elif (self.iNoFpos == 1) and (d > self.sn) and (d >= params.Thetau_pos):
                f = 1.0e-10
                self.ek = 1.0e-7
                self.flagstopdeg = 1
            elif d >= params.Thetau_pos or self.flagControlResponse == 1:
                f = 1.0e-10
                self.ek = 1.0e-7
                self.flagstopdeg = 1
                self.flagControlResponse = 1
            elif (self.iNoFpos == 1) and (d < self.sn):
                self.ek = self.ekunload
                f = self.fP + self.ek * deltaD
            elif (self.iNoFneg == 1) and (d < self.dNewLoadNeg):
                f = 0
                self.ek = 1.0e-7
            elif d > self.dmax:
                f = 0.0
                _positive_envelope(
                    self.flagControlResponse,
                    params.Thetau_pos,
                    self.fyPos,
                    self.alphaPos,
                    self.capSlope,
                    self.cpPos,
                    d,
                    f,
                    self.ek,
                    self.Ke,
                    params.My_pos,
                    params.KPos,
                )
                self.dmax = d
                self.fmax = f
                dBoundPos = self._positive_bound()
                if (d > dBoundPos) or (self.ek == 1.0e-7):
                    self.iNoFpos = 1
            elif abs(self.sn) > 1.0e-10:
                if self.LP == 0:
                    if self.cpPos <= self.dyPos:
                        intersect_lines(
                            xDevPos1,
                            yDevPos1,
                            self.sn,
                            self.resSn,
                            self.ekhardPos,
                            self.cpPos,
                            self.fCapPos,
                            self.capSlope * self.Ke,
                        )
                        intersect_lines(
                            xDevPos2,
                            yDevPos2,
                            self.sn,
                            self.resSn,
                            self.ekunload,
                            self.cpPos,
                            self.fCapPos,
                            self.capSlope * self.Ke,
                        )
                        if xDevPos1 > xDevPos2:
                            xDevPos = xDevPos1
                            yDevPos = yDevPos1
                        else:
                            xDevPos = xDevPos2
                            yDevPos = yDevPos2
                        if (d <= self.sn) and (d <= xDevPos):
                            self.ek = self.ekunload
                            f = self.fP + self.ek * deltaD
                        elif (d > self.sn) and (d <= xDevPos):
                            self.ek = self.Ke * self.alphaPos
                            f2 = self.resSn + self.ek * (d - self.sn)
                            f1 = self.fP + self.ekunload * deltaD
                            if f1 < f2:
                                f = f1
                            else:
                                f = f2
                        elif d > xDevPos:
                            self.ek = self.capSlope * self.Ke
                            f2 = yDevPos + self.ek * (d - xDevPos)
                            f1 = self.fP + self.ekunload * deltaD
                            if f1 < f2:
                                f = f1
                            else:
                                f = f2
                            if self.ek != self.ekunload:
                                self._check_force_sign_change(f)
                    elif self.cpPos > self.dyPos:
                        intersect_lines(
                            xDevPos1,
                            yDevPos1,
                            self.sn,
                            self.resSn,
                            self.ekhardPos,
                            self.cpPos,
                            self.fCapPos,
                            self.capSlope * self.Ke,
                        )
                        if d <= self.sn and d <= xDevPos1:
                            self.ek = self.ekunload
                            f = self.fP + self.ek * deltaD
                        elif (d > self.sn) and (d <= self.cpPos) and (d <= xDevPos1):
                            self.ek = self.Ke * self.alphaPos
                            f2 = self.resSn + self.ek * (d - self.sn)
                            f1 = self.fP + self.ekunload * deltaD
                            if f1 < f2:
                                f = f1
                            else:
                                f = f2
                            if abs(f - f1) < 1.0e-10:
                                self.ek = self.ekunload
                        elif (d > self.cpPos) and (d < params.Thetau_pos):
                            self.ek = self.capSlope * self.Ke
                            f2 = self.fCapPos + self.ek * (d - self.cpPos)
                            f1 = self.fP + self.ekunload * deltaD
                            if f1 < f2:
                                f = f1
                            else:
                                f = f2
                            if abs(f - f1) < 1.0e-10:
                                self.ek = self.ekunload
                            if self.ek != self.ekunload:
                                self._check_force_sign_change(f)
                        elif d >= params.Thetau_pos:
                            f = 1.0e-10
                            self.ek = -1.0e-7
                            self.flagstopdeg = 1
                        if d < params.Thetau_pos and d > self.cpPos + (
                            params.KPos * params.My_pos - self.fCapPos
                        ) / (self.capSlope * self.Ke):
                            f = params.KPos * params.My_pos
                            self.ek = -1.0e-7
                        if self.flagControlResponse == 1:
                            f = 1.0e-10
                elif self.LP == 1:
                    if d <= self.sn:
                        self.ek = self.ekunload
                        f = self.fP + self.ek * deltaD
                    elif ((d > self.sn) and (self.sn == self.snEnv) and (d <= self.snHor)) or (
                        (self.iNoFneg == 1) and (d > self.sn) and (d < self.snHor)
                    ):
                        self.ek = self.Ke * self.alphaPos
                        f2 = self.resSn + self.ek * (d - self.sn)
                        f1 = self.fP + self.ekunload * deltaD
                        if f1 < f2:
                            f = f1
                        else:
                            f = f2
                        if abs(f - f1) < 1.0e-10:
                            self.ek = self.ekunload
                    else:
                        self.ek = 0
                        f1 = self.fP + self.ekunload * deltaD
                        f2 = self.fLimPos
                        if f1 < f2:
                            f = f1
                        else:
                            f = f2
                        if abs(f - f1) < 1.0e-10:
                            self.ek = self.ekunload
            else:
                if d > 0.0:
                    f = 0.0
                    _positive_envelope(
                        self.flagControlResponse,
                        params.Thetau_pos,
                        self.fyPos,
                        self.alphaPos,
                        self.capSlope,
                        self.cpPos,
                        d,
                        f,
                        self.ek,
                        self.Ke,
                        params.My_pos,
                        params.KPos,
                    )
                else:
                    f = 0.0
                    _negative_envelope(
                        self.flagControlResponse,
                        params.Thetau_neg,
                        self.fyNeg,
                        self.alphaNeg,
                        self.capSlopeNeg,
                        self.cpNeg,
                        d,
                        f,
                        self.ek,
                        self.Ke,
                        params.My_neg,
                        params.KNeg,
                    )
        else:
            if self.iNoFneg == 1:
                intersect_lines(
                    self.dNewLoadNeg,
                    fNewLoadNeg,
                    self.dyPos,
                    self.fyPos,
                    self.ekhardPos,
                    0.0,
                    0.0,
                    0.0,
                )
            if self.kon == 1:
                self.kon = 2
                if self.dP >= self.dmax:
                    self.fLimPos = self.fP
                    self.dLimPos = self.dP
                if self.resSp < 0.0:
                    self.RSE = 0.5 * self.fP * self.fP / self.ekunload
                else:
                    self.RSE = 0.5 * (self.fP + self.resSn) * (
                        self.dP - self.sp
                    ) + 0.5 * self.resSp * self.resSp / (self.Ke * self.alphaNeg)
                var a2: Float64 = self.Enrgtk - (self.Enrgtot - self.RSE)
                if (a2 <= 0.0) and (self.Enrgtk != 0.0):
                    self.flagControlResponse = 1
                if (
                    (params.LamdaK != 0.0)
                    and (d > self.sn)
                    and (self.flagstopdeg != 1)
                    and (self.flagdeg != 1)
                ):
                    betak = (
                        ((self.Enrgc - self.RSE) / (self.Enrgtk - (self.Enrgtot - self.RSE)))
                    ) ** params.Ck
                    if ((self.Enrgtot - self.RSE) >= self.Enrgtk) or (betak >= 1.0):
                        betak = 1.0
                    if self.flagstopdeg != 1 and self.flagdeg != 1:
                        self.ekunload = self.Ke * (
                            (self.prodBeta * (1.0 - betak))
                            / (1 + params.nFactor * (1 - (self.prodBeta * (1 - betak))))
                        )
                    else:
                        self.ekunload = self.ekexcurs
                    if self.ekunload <= (0.1 / (1 + params.nFactor * (1 - 0.1))) * self.Ke:
                        self.ekunload = (0.1 / (1 + params.nFactor * (1 - 0.1))) * self.Ke
                if (self.dmin < (params.My_neg / self.Ke)) or (
                    self.dmax > (params.My_pos / self.Ke)
                ):
                    if (self.dP > self.sn) or (
                        ((self.dP < self.sn) and (self.ekP == self.ekunload))
                    ):
                        self._negative_reload_intersection(params)
                        if (abs(self.dmin - (params.My_neg / self.Ke)) >= 1.0e-10) and (
                            abs(self.sp - (params.My_neg / self.Ke)) <= 1.0e-10
                        ):
                            self.sp = (params.My_neg / self.Ke) - 1.0e-9
                if self.sp < self.dmin:
                    self.dmin = self.sp
                if (self.iNoFpos == 1) and (self.dP >= self.dNewLoadPos):
                    self.sp = self.dNewLoadPos + 1.0e-6
                    self.resSp = 0
            if (self.iNoFneg == 1) and (self.iNoFpos == 1) and (d > (-params.Thetau_neg)):
                f = params.KNeg * params.My_neg
                self.ek = 1.0e-7
            elif (
                (self.iNoFneg == 1) and (self.iNoFpos == 1) and (d <= (-params.Thetau_neg))
            ) or self.flagControlResponse == 1:
                f = -1.0e-10
                self.ek = 1.0e-7
                self.flagstopdeg = 1
            elif (self.iNoFneg == 1) and (d < self.sp) and (d > (-params.Thetau_neg)):
                f = params.KNeg * params.My_neg
                self.ek = 1.0e-7
            elif (self.iNoFneg == 1) and (d < self.sp) and (d <= (-params.Thetau_neg)):
                f = -1.0e-10
                self.ek = 1.0e-7
                self.flagstopdeg = 1
            elif d <= (-params.Thetau_neg) or self.flagControlResponse == 1:
                f = -1.0e-10
                self.ek = 1.0e-7
                self.flagstopdeg = 1
                self.flagControlResponse = 1
            elif (self.iNoFneg == 1) and (d >= self.sp):
                self.ek = self.ekunload
                f = self.fP + self.ek * deltaD
            elif (self.iNoFpos == 1) and (d > self.dNewLoadPos):
                f = 0
                self.ek = 1.0e-7
            elif d < self.dmin:
                f = 0.0
                _negative_envelope(
                    self.flagControlResponse,
                    params.Thetau_neg,
                    self.fyNeg,
                    self.alphaNeg,
                    self.capSlopeNeg,
                    self.cpNeg,
                    d,
                    f,
                    self.ek,
                    self.Ke,
                    params.My_neg,
                    params.KNeg,
                )
                self.dmin = d
                self.fmin = f
                dBoundNeg = self._negative_bound()
                if (d < dBoundNeg) or (self.ek == 1.0e-7):
                    self.iNoFneg = 1
            elif abs(self.sp) > 1.0e-10:
                if self.LN == 0:
                    if self.cpNeg >= self.dyNeg:
                        intersect_lines(
                            xDevNeg1,
                            yDevNeg1,
                            self.sp,
                            self.resSp,
                            self.Ke * self.alphaNeg,
                            self.cpNeg,
                            self.fCapNeg,
                            self.capSlopeNeg * self.Ke,
                        )
                        intersect_lines(
                            xDevNeg2,
                            yDevNeg2,
                            self.sp,
                            self.resSp,
                            self.ekunload,
                            self.cpNeg,
                            self.fCapNeg,
                            self.capSlopeNeg * self.Ke,
                        )
                        if xDevNeg1 < xDevNeg2:
                            xDevNeg = xDevNeg1
                            yDevNeg = yDevNeg1
                        else:
                            xDevNeg = xDevNeg2
                            yDevNeg = yDevNeg2
                        if (d >= self.sp) and (d >= xDevNeg):
                            self.ek = self.ekunload
                            f = self.fP + self.ek * deltaD
                        elif (d < self.sp) and (d >= xDevNeg):
                            self.ek = self.Ke * self.alphaNeg
                            f2 = self.resSp + self.ek * (d - self.sp)
                            f1 = self.fP + self.ekunload * deltaD
                            if f1 > f2:
                                f = f1
                            else:
                                f = f2
                            if abs(f - f1) < 1.0e-10:
                                self.ek = self.ekunload
                        elif d < xDevNeg:
                            self.ek = self.capSlopeNeg * self.Ke
                            f2 = yDevNeg + self.ek * (d - xDevNeg)
                            f1 = self.fP + self.ekunload * deltaD
                            if f1 > f2:
                                f = f1
                            else:
                                f = f2
                            if abs(f - f1) < 1.0e-10:
                                self.ek = self.ekunload
                            if self.ek != self.ekunload:
                                self._check_force_sign_change(f)
                    elif self.cpNeg < self.dyNeg:
                        intersect_lines(
                            xDevNeg1,
                            yDevNeg1,
                            self.sp,
                            self.resSp,
                            self.Ke * self.alphaNeg,
                            self.cpNeg,
                            self.fCapNeg,
                            self.capSlopeNeg * self.Ke,
                        )
                        if d >= self.sp and d >= xDevNeg1:
                            self.ek = self.ekunload
                            f = self.fP + self.ek * deltaD
                        elif (d < self.sp) and (d >= self.cpNeg) and (d >= xDevNeg1):
                            self.ek = self.Ke * self.alphaNeg
                            f2 = self.resSp + self.ek * (d - self.sp)
                            f1 = self.fP + self.ekunload * deltaD
                            if f1 > f2:
                                f = f1
                            else:
                                f = f2
                            if abs(f - f1) < 1.0e-10:
                                self.ek = self.ekunload
                        elif (d < self.cpNeg) and (d > (-params.Thetau_neg)):
                            self.ek = self.capSlopeNeg * self.Ke
                            f2 = self.fCapNeg + self.ek * (d - self.cpNeg)
                            f1 = self.fP + self.ekunload * deltaD
                            if f1 > f2:
                                f = f1
                            else:
                                f = f2
                            if abs(f - f1) < 1.0e-10:
                                self.ek = self.ekunload
                            if self.ek != self.ekunload:
                                self._check_force_sign_change(f)
                        elif d <= (-params.Thetau_neg) or self.flagControlResponse == 1:
                            self.ek = -1.0e-7
                            f = 1.0e-10
                            self.flagstopdeg = 1
                            self.flagControlResponse = 1
                        if d > (-params.Thetau_neg) and d < self.cpNeg + (
                            params.KNeg * params.My_neg - self.fCapNeg
                        ) / (self.capSlopeNeg * self.Ke):
                            f = params.KNeg * params.My_neg
                            self.ek = -1.0e-7
                        if self.flagControlResponse == 1:
                            f = 1.0e-10
                elif self.LN == 1:
                    if d >= self.sp:
                        self.ek = self.ekunload
                        f = self.fP + self.ek * deltaD
                    elif ((d < self.sp) and (self.sp == self.spEnv) and (d > self.spHor)) or (
                        (self.iNoFpos == 1) and (d < self.sp) and (d > self.spHor)
                    ):
                        self.ek = self.Ke * self.alphaNeg
                        f2 = self.resSp + self.ek * (d - self.sp)
                        f1 = self.fP + self.ekunload * deltaD
                        if f1 > f2:
                            f = f1
                        else:
                            f = f2
                        if abs(f - f1) < 1.0e-10:
                            self.ek = self.ekunload
                    else:
                        self.ek = 1.0e-7
                        f1 = self.fP + self.ekunload * deltaD
                        f2 = self.fLimNeg
                        if f1 > f2:
                            f = f1
                        else:
                            f = f2
                        if abs(f - f1) < 1.0e-10:
                            self.ek = self.ekunload
            else:
                if d > 0.0:
                    f = 0.0
                    _positive_envelope(
                        self.flagControlResponse,
                        params.Thetau_pos,
                        self.fyPos,
                        self.alphaPos,
                        self.capSlope,
                        self.cpPos,
                        d,
                        f,
                        self.ek,
                        self.Ke,
                        params.My_pos,
                        params.KPos,
                    )
                else:
                    _negative_envelope(
                        self.flagControlResponse,
                        params.Thetau_neg,
                        self.fyNeg,
                        self.alphaNeg,
                        self.capSlopeNeg,
                        self.cpNeg,
                        d,
                        f,
                        self.ek,
                        self.Ke,
                        params.My_neg,
                        params.KNeg,
                    )
        var Enrgi: Float64
        Enrgi = 0.5 * (f + self.fP) * deltaD
        self.Enrgc = self.Enrgc + Enrgi
        self.Enrgtot = self.Enrgtot + Enrgi
        self.RSE = 0.5 * f * f / self.ekunload
        self.prodBeta = self.prodBeta * (1 - betak)
        if (f * self.fP < 0.0) and (self.interup == 0):
            if ((self.fP > 0.0) and (self.dmax > (self.dyPos))) or (
                (self.fP < 0.0) and (self.dmin < (self.dyNeg))
            ):
                self.flagdeg = 1
                self.interup = 1
        if (self.flagstopdeg == 0) and (self.flagdeg == 1):
            if (self.Enrgtot >= self.Enrgts) and (self.Enrgts != 0.0):
                betas = 1.0
            elif (self.Enrgtot >= self.Enrgtd) and (self.Enrgtd != 0.0):
                betad = 1.0
            else:
                if params.LamdaS != 0.0:
                    betas = (self.Enrgc / (self.Enrgts - self.Enrgtot)) ** params.Cs
                if params.LamdaD != 0.0:
                    betad = (self.Enrgc / (self.Enrgtd - self.Enrgtot)) ** params.Cd
                if abs(betas) >= 1.0:
                    betas = 1.0
                if abs(betad) >= 1.0:
                    betad = 1.0
            self.Enrgc = 0.0
            self.ekexcurs = self.ekunload
            if deltaD < 0.0:
                if self.flagstopdeg == 0:
                    self.fyNeg = self.fyNeg * (1 - betas * params.PDNeg)
                    self.alphaNeg = (
                        self.alphaNeg * (1 + params.nFactor) / (1 + params.nFactor * self.alphaNeg)
                    )
                    self.alphaNeg = self.alphaNeg * (1 - betas * params.PDNeg)
                    self.alphaNeg = (self.alphaNeg) / (1 + params.nFactor * (1 - self.alphaNeg))
                    self.fCapRefNeg = self.fCapRefNeg * (1 - betad * params.PDNeg)
                else:
                    self.fyNeg = self.fyNeg
                    self.alphaNeg = self.alphaNeg
                    self.fCapRefNeg = self.fCapRefNeg
                if self.fyNeg >= params.KNeg * params.My_neg:
                    self.fyNeg = params.KNeg * params.My_neg
                    self.alphaNeg = 1.0e-4
                    self.fCapRefNeg = self.fyNeg
                    self.capSlopeNeg = -1.0e-6
                    self.flagstopdeg = 1
                else:
                    self.capSlopeNeg = (
                        self.capSlopeOrigNeg
                        * (1 + params.nFactor)
                        / (1 + self.capSlopeOrigNeg * params.nFactor)
                    )
                    self.capSlopeNeg = self.capSlopeNeg * (
                        abs(
                            (params.KNeg * params.My_neg - self.fyNeg)
                            / (params.KNeg * params.My_neg - params.My_neg)
                        )
                    )
                    self.capSlopeNeg = self.capSlopeNeg / (
                        1 + params.nFactor * (1 - self.capSlopeNeg)
                    )
                    if self.capSlopeNeg >= 0:
                        self.capSlopeNeg = -1.0e-6
                self.dyNeg = self.fyNeg / self.Ke
                self.ekhardNeg = self.alphaNeg * self.Ke
                var dCap1Neg: Float64 = self.fCapRefNeg / (self.Ke - self.capSlopeOrigNeg * self.Ke)
                var dCap2Neg: Float64 = (
                    self.fCapRefNeg + self.ekhardNeg * self.dyNeg - self.fyNeg
                ) / (self.ekhardNeg - self.capSlopeOrigNeg * self.Ke)
                if dCap1Neg < dCap2Neg:
                    self.cpNeg = dCap1Neg
                else:
                    self.cpNeg = dCap2Neg
                self.fCapNeg = self.fCapRefNeg + self.capSlopeOrigNeg * self.Ke * self.cpNeg
                _negative_envelope(
                    self.flagControlResponse,
                    params.Thetau_neg,
                    self.fyNeg,
                    self.alphaNeg,
                    self.capSlopeNeg,
                    self.cpNeg,
                    self.dLimNeg,
                    self.fLimNeg,
                    self.ek,
                    self.Ke,
                    params.My_neg,
                    params.KNeg,
                )
                self._negative_reload_intersection(params)
                if self.resSp > f:
                    d = self.sp
                    f = self.resSp
                    self.ek = self.ekhardNeg
            else:
                if self.flagstopdeg == 0:
                    self.fyPos = self.fyPos * (1 - betas * params.PDPlus)
                    self.alphaPos = (
                        self.alphaPos * (1 + params.nFactor) / (1 + params.nFactor * self.alphaPos)
                    )
                    self.alphaPos = self.alphaPos * (1 - betas * params.PDPlus)
                    self.alphaPos = (self.alphaPos) / (1 + params.nFactor * (1 - self.alphaPos))
                    self.fCapRefPos = self.fCapRefPos * (1 - betad * params.PDPlus)
                else:
                    self.fyPos = self.fyPos
                    self.alphaPos = self.alphaPos
                    self.fCapRefPos = self.fCapRefPos
                if self.fyPos <= params.KPos * params.My_pos:
                    self.fyPos = params.KPos * params.My_pos
                    self.alphaPos = 1.0e-4
                    self.fCapRefPos = self.fyPos
                    self.capSlope = -1.0e-6
                    self.flagstopdeg = 1
                else:
                    self.capSlope = (
                        self.capSlopeOrig
                        * (1 + params.nFactor)
                        / (1 + self.capSlopeOrig * params.nFactor)
                    )
                    self.capSlope = self.capSlope * (
                        abs(
                            (params.KPos * params.My_pos - self.fyPos)
                            / (params.KPos * params.My_pos - params.My_pos)
                        )
                    )
                    self.capSlope = self.capSlope / (1 + params.nFactor * (1 - self.capSlope))
                    if self.capSlope >= 0:
                        self.capSlope = -1.0e-6
                self.dyPos = self.fyPos / self.Ke
                self.ekhardPos = self.alphaPos * self.Ke
                var dCap1Pos: Float64 = self.fCapRefPos / (self.Ke - self.capSlopeOrig * self.Ke)
                var dCap2Pos: Float64 = (
                    self.fCapRefPos + self.ekhardPos * self.dyPos - self.fyPos
                ) / (self.ekhardPos - self.capSlopeOrig * self.Ke)
                if dCap1Pos > dCap2Pos:
                    self.cpPos = dCap1Pos
                else:
                    self.cpPos = dCap2Pos
                self.fCapPos = self.fCapRefPos + self.capSlopeOrig * self.Ke * self.cpPos
                _positive_envelope(
                    self.flagControlResponse,
                    params.Thetau_pos,
                    self.fyPos,
                    self.alphaPos,
                    self.capSlope,
                    self.cpPos,
                    self.dLimPos,
                    self.fLimPos,
                    self.ek,
                    self.Ke,
                    params.My_pos,
                    params.KPos,
                )
                self._positive_reload_intersection(params)
                if self.resSn < f:
                    d = self.sn
                    f = self.resSn
                    self.ek = self.ekhardPos
        if (d < 0) and (abs(self.ek) <= 1.0e-7):
            self.LN = 1
        if (d > 0) and (abs(self.ek) <= 1.0e-7):
            self.LP = 1
        self.ekP = self.ek
        self.fP = f
        self.dP = d
        self.Tangent = self.ek
        if (
            (self.interup == 1 and (self.ek == self.Ke * self.alphaPos))
            or (self.ek == self.Ke * self.alphaNeg)
            or (self.ek == self.capSlope * self.Ke)
            or (self.ek == self.capSlopeNeg * self.Ke)
        ):
            self.interup = 0
        return

    fn _positive_reload_intersection(mut self, params: BilinParameters):
        var Resid: Float64 = params.KPos * self.fyPos
        var dresid: Float64 = self.cpPos + (Resid - self.fCapPos) / (self.capSlope * self.Ke)
        var ekresid: Float64 = 1.0e-10
        self.dyPos = self.fyPos / self.Ke
        var snHard: Float64 = 0.0
        var resSnHard: Float64 = 0.0
        var snLim: Float64 = 0.0
        var resSnLim: Float64 = 0.0
        var snResid: Float64 = 0.0
        var resSnResid: Float64 = 0.0
        if self.dyPos < self.cpPos:
            intersect_lines(
                snHard,
                resSnHard,
                self.dyPos,
                self.fyPos,
                self.Ke * self.alphaPos,
                self.dP,
                self.fP,
                self.ekunload,
            )
        else:
            intersect_lines(
                snHard,
                resSnHard,
                self.cpPos,
                self.fCapPos,
                self.Ke * self.alphaPos,
                self.dP,
                self.fP,
                self.ekunload,
            )
        var snCap: Float64 = 0.0
        var resSnCap: Float64 = 0.0
        intersect_lines(
            snCap,
            resSnCap,
            0.0,
            self.fCapRefPos,
            self.capSlope * self.Ke,
            self.dP,
            self.fP,
            self.ekunload,
        )
        if snHard < snCap:
            self.sn = snHard
        else:
            self.sn = snCap
        if resSnHard < resSnCap:
            self.resSn = resSnHard
        else:
            self.resSn = resSnCap
        self.snEnv = self.sn
        self.resSnEnv = self.resSn
        if (self.LP == 1) and (self.fLimPos == 0.0):
            intersect_lines(
                snLim, resSnLim, self.dLimPos, self.fLimPos, 0.0, self.dP, self.fP, self.ekunload
            )
            if snLim < self.sn:
                self.sn = snLim
                self.resSn = resSnLim
            intersect_lines(
                self.snHor,
                self.resSnHor,
                self.dLimPos,
                self.fLimPos,
                0.0,
                self.dyPos,
                self.fyPos,
                self.Ke * self.alphaPos,
            )
        if self.sn > dresid:
            intersect_lines(
                snResid, resSnResid, dresid, Resid, ekresid, self.dP, self.fP, self.ekunload
            )
            self.sn = snResid
            self.resSn = resSnResid

    fn _negative_reload_intersection(mut self, params: BilinParameters):
        var Resid: Float64 = params.KNeg * self.fyNeg
        self.dyNeg = self.fyNeg / self.Ke
        var dresid: Float64 = self.cpNeg + (Resid - self.fCapNeg) / (self.capSlopeNeg * self.Ke)
        var ekresid: Float64 = 1.0e-10
        var spHard: Float64 = 0.0
        var resSpHard: Float64 = 0.0
        var spCap: Float64 = 0.0
        var resSpCap: Float64 = 0.0
        var spLim: Float64 = 0.0
        var resSpLim: Float64 = 0.0
        var spResid: Float64 = 0.0
        var resSpResid: Float64 = 0.0
        if self.dyNeg > self.cpNeg:
            intersect_lines(
                spHard,
                resSpHard,
                self.dyNeg,
                self.fyNeg,
                self.Ke * self.alphaNeg,
                self.dP,
                self.fP,
                self.ekunload,
            )
        else:
            intersect_lines(
                spHard,
                resSpHard,
                self.cpNeg,
                self.fCapNeg,
                self.Ke * self.alphaNeg,
                self.dP,
                self.fP,
                self.ekunload,
            )
        intersect_lines(
            spCap,
            resSpCap,
            0.0,
            self.fCapRefNeg,
            self.capSlopeNeg * self.Ke,
            self.dP,
            self.fP,
            self.ekunload,
        )
        if spHard > spCap:
            self.sp = spHard
        else:
            self.sp = spCap
        if resSpHard > resSpCap:
            self.resSp = resSpHard
        else:
            self.resSp = resSpCap
        self.spEnv = self.sp
        self.resSpEnv = self.resSp
        if (self.LN == 1) and (self.fLimNeg == 0.0):
            intersect_lines(
                spLim, resSpLim, self.dLimNeg, self.fLimNeg, 0.0, self.dP, self.fP, self.ekunload
            )
            if spLim > self.sp:
                self.sp = spLim
                self.resSp = resSpLim
            intersect_lines(
                self.spHor,
                self.resSpHor,
                self.dLimNeg,
                self.fLimNeg,
                0.0,
                self.dyNeg,
                self.fyNeg,
                self.Ke * self.alphaNeg,
            )
        if self.sp < dresid:
            intersect_lines(
                spResid, resSpResid, dresid, Resid, ekresid, self.dP, self.fP, self.ekunload
            )
            self.sp = spResid
            self.resSp = resSpResid

    fn _positive_bound(mut self) -> Float64:
        var Resid: Float64 = 0.0
        var dBoundPos: Float64
        self.dyNeg = self.fyNeg / self.Ke
        var dresid: Float64 = self.cpPos + (Resid - self.fCapPos) / (self.capSlope * self.Ke)
        var ekresid: Float64 = 1.0e-10
        var d1: Float64 = 0.0
        var f1: Float64 = 0.0
        intersect_lines(
            d1,
            f1,
            self.dyNeg,
            self.fyNeg,
            self.Ke * self.alphaNeg,
            0.0,
            self.fCapRefPos,
            self.capSlope * self.Ke,
        )
        var d2: Float64 = 0.0
        var f2: Float64 = 0.0
        intersect_lines(
            d2, f2, self.dyNeg, self.fyNeg, self.Ke * self.alphaNeg, dresid, Resid, ekresid
        )
        if d1 > d2:
            dBoundPos = d1
        else:
            dBoundPos = d2
        return dBoundPos

    fn _negative_bound(mut self) -> Float64:
        var Resid: Float64 = 0
        var dBoundNeg: Float64
        self.dyPos = self.fyPos / self.Ke
        var dresid: Float64 = self.cpNeg + (Resid - self.fCapNeg) / (self.capSlopeNeg * self.Ke)
        var ekresid: Float64 = 1.0e-10
        var d1: Float64 = 0.0
        var f1: Float64 = 0.0
        var d2: Float64 = 0.0
        var f2: Float64 = 0.0
        intersect_lines(
            d1,
            f1,
            self.dyPos,
            self.fyPos,
            self.Ke * self.alphaPos,
            0.0,
            self.fCapRefNeg,
            self.capSlopeNeg * self.Ke,
        )
        intersect_lines(
            d2, f2, self.dyPos, self.fyPos, self.Ke * self.alphaPos, dresid, Resid, ekresid
        )
        if d1 < d2:
            dBoundNeg = d1
        else:
            dBoundNeg = d2
        return dBoundNeg

    fn _check_force_sign_change(mut self, mut f: Float64):
        if self.fP > 0:
            if (f * self.fP) < 0:
                f = 0
                self.ek = 1.0e-7
                self.iNoFpos = 1
                self.flagControlResponse = 1
        elif self.fP < 0:
            if (f * self.fP) < 0:
                f = 0
                self.ek = 1.0e-7
                self.iNoFneg = 1
                self.flagControlResponse = 1


fn intersect_lines(
    mut xInt: Float64,
    mut yInt: Float64,
    x1: Float64,
    y1: Float64,
    m1: Float64,
    x2: Float64,
    y2: Float64,
    m2: Float64,
):
    xInt = (-m2 * x2 + y2 + m1 * x1 - y1) / (m1 - m2)
    yInt = m1 * xInt - m1 * x1 + y1


fn _positive_envelope(
    mut flagControlResponse: Int,
    ultimate: Float64,
    fy: Float64,
    alphaPos: Float64,
    alphaCap: Float64,
    cpDsp: Float64,
    mut d: Float64,
    mut f: Float64,
    mut ek: Float64,
    elstk: Float64,
    fyieldPos: Float64,
    Resfac: Float64,
):
    var dy: Float64 = fy / elstk
    var Res: Float64
    var rcap: Float64
    var dres: Float64
    if dy <= cpDsp:
        Res = Resfac * fyieldPos
        rcap = fy + alphaPos * elstk * (cpDsp - dy)
        dres = cpDsp + (Res - rcap) / (alphaCap * elstk)
        if d < 0.0:
            f = 0.0
            ek = 1.0e-7
        elif d <= dy:
            ek = elstk
            f = ek * d
        elif d <= cpDsp:
            ek = elstk * alphaPos
            f = fy + ek * (d - dy)
        elif d <= dres:
            ek = alphaCap * elstk
            f = rcap + ek * (d - cpDsp)
        else:
            ek = 1.0e-7
            f = Res + d * ek
        if d >= ultimate:
            ek = 1.0e-7
            f = 1.0e-10
            d = ultimate
            flagControlResponse = 1
    elif dy > cpDsp:
        rcap = elstk * cpDsp
        Res = Resfac * rcap
        dres = cpDsp + (Res - rcap) / (alphaCap * elstk)
        if d < 0.0:
            f = 0.0
            ek = 1.0e-7
        elif d <= cpDsp:
            ek = elstk
            f = ek * d
        elif d <= dres:
            ek = alphaCap * elstk
            f = rcap + ek * (d - cpDsp)
        else:
            ek = 1.0e-7
            f = Res + d * ek
        if d >= ultimate:
            ek = 1.0e-7
            f = 1.0e-10
            d = ultimate
            flagControlResponse = 1


fn _negative_envelope(
    mut flagControlResponse: Int,
    ultimate: Float64,
    fy: Float64,
    alphaNeg: Float64,
    alphaCap: Float64,
    cpDsp: Float64,
    mut d: Float64,
    mut f: Float64,
    mut ek: Float64,
    elstk: Float64,
    fyieldNeg: Float64,
    Resfac: Float64,
):
    var dy: Float64 = fy / elstk
    var Res: Float64
    var rcap: Float64
    var dres: Float64
    if dy >= cpDsp:
        Res = Resfac * fyieldNeg
        rcap = fy + alphaNeg * elstk * (cpDsp - dy)
        dres = cpDsp + (Res - rcap) / (alphaCap * elstk)
        if d > 0.0:
            f = 0.0
            ek = 1.0e-7
        elif d >= dy:
            ek = elstk
            f = ek * d
        elif d >= cpDsp:
            ek = elstk * alphaNeg
            f = fy + ek * (d - dy)
        elif d >= dres:
            ek = elstk * alphaCap
            f = rcap + ek * (d - cpDsp)
        else:
            ek = 1.0e-7
            f = Res + ek * d
        if d <= (-ultimate):
            ek = 1.0e-7
            f = 1.0e-10
            d = -ultimate
            flagControlResponse = 1
    elif dy < cpDsp:
        rcap = elstk * cpDsp
        Res = Resfac * rcap
        dres = cpDsp + (Res - rcap) / (alphaCap * elstk)
        if d > 0.0:
            f = 0.0
            ek = 1.0e-7
        elif d >= cpDsp:
            ek = elstk
            f = ek * d
        elif d >= dres:
            ek = elstk * alphaCap
            f = rcap + ek * (d - cpDsp)
        else:
            ek = 1.0e-7
            f = Res + ek * d
        if d <= (-ultimate):
            ek = 1.0e-7
            f = 1.0e-10
            d = -ultimate
            flagControlResponse = 1
