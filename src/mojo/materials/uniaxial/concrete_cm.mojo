# Chang–Mander concrete implemented from OpenSees 3.7.1 ConcreteCM.cpp.
# The envelope, transition rules and history updates retain the source rule numbers.
from math import exp, log10
from sys.ffi import external_call


fn pow(base: Float64, exponent: Float64) -> Float64:
    return external_call["pow", Float64](base, exponent)


struct ConcreteCMParameters(Defaultable, ImplicitlyCopyable, Movable):
    var fpcc: Float64
    var epcc: Float64
    var Ec: Float64
    var rc: Float64
    var xcrn: Float64
    var ft: Float64
    var et: Float64
    var rt: Float64
    var xcrp: Float64
    var gap_close: Float64
    var monotonic: Float64

    fn __init__(out self):
        self.fpcc = 0.0
        self.epcc = 0.0
        self.Ec = 0.0
        self.rc = 0.0
        self.xcrn = 0.0
        self.ft = 0.0
        self.et = 0.0
        self.rt = 0.0
        self.xcrp = 0.0
        self.gap_close = 0.0
        self.monotonic = 0.0


struct ConcreteCMHistory(Defaultable, ImplicitlyCopyable, Movable):
    var eunn: Float64
    var funn: Float64
    var eunp: Float64
    var funp: Float64
    var er: Float64
    var fr: Float64
    var er0n: Float64
    var fr0n: Float64
    var er0p: Float64
    var fr0p: Float64
    var e0: Float64
    var ea: Float64
    var eb: Float64
    var ed: Float64
    var inc: Float64
    var rule: Float64
    var strain: Float64
    var stress: Float64
    var tangent: Float64

    fn __init__(out self):
        self.eunn = 0.0
        self.funn = 0.0
        self.eunp = 0.0
        self.funp = 0.0
        self.er = 0.0
        self.fr = 0.0
        self.er0n = 0.0
        self.fr0n = 0.0
        self.er0p = 0.0
        self.fr0p = 0.0
        self.e0 = 0.0
        self.ea = 0.0
        self.eb = 0.0
        self.ed = 0.0
        self.inc = 0.0
        self.rule = 0.0
        self.strain = 0.0
        self.stress = 0.0
        self.tangent = 0.0


struct ConcreteCMTrial(Movable):
    var params: ConcreteCMParameters
    var committed: ConcreteCMHistory
    var trial: ConcreteCMHistory
    var xn: Float64
    var nn: Float64
    var xsp: Float64
    var xp: Float64
    var np: Float64
    var xcrk: Float64
    var y: Float64
    var z: Float64
    var eunn: Float64
    var funn: Float64
    var espln: Float64
    var Epln: Float64
    var Esecn: Float64
    var Esectest: Float64
    var Esectest10: Float64
    var Esectest13: Float64
    var delen: Float64
    var delfn: Float64
    var fnewn: Float64
    var Enewn: Float64
    var esren: Float64
    var fren: Float64
    var Eren: Float64
    var fnewstn: Float64
    var Enewstn: Float64
    var esrestn: Float64
    var frestn: Float64
    var Erestn: Float64
    var eunp: Float64
    var funp: Float64
    var esplp: Float64
    var Eplp: Float64
    var Esecp: Float64
    var delep: Float64
    var delfp: Float64
    var fnewp: Float64
    var Enewp: Float64
    var esrep: Float64
    var frep: Float64
    var Erep: Float64
    var fnewstp: Float64
    var Enewstp: Float64
    var esrestp: Float64
    var frestp: Float64
    var Erestp: Float64
    var esi: Float64
    var esi10: Float64
    var esi13: Float64
    var fi: Float64
    var fi10: Float64
    var fi13: Float64
    var Ei: Float64
    var Ei10: Float64
    var Ei13: Float64
    var esf: Float64
    var esf10: Float64
    var esf13: Float64
    var ff: Float64
    var ff10: Float64
    var ff13: Float64
    var Ef: Float64
    var Ef10: Float64
    var Ef13: Float64
    var R: Float64
    var R10: Float64
    var R13: Float64
    var A: Float64
    var A10: Float64
    var A13: Float64
    var fc: Float64
    var Et: Float64
    var fca: Float64
    var Eta: Float64
    var fcb: Float64
    var Etb: Float64
    var fa: Float64
    var fb: Float64

    fn __init__(out self, params: ConcreteCMParameters, committed: ConcreteCMHistory):
        self.params = params
        self.committed = committed
        self.trial = committed
        self.xn = 0.0
        self.nn = 0.0
        self.xsp = 0.0
        self.xp = 0.0
        self.np = 0.0
        self.xcrk = 0.0
        self.y = 0.0
        self.z = 0.0
        self.eunn = 0.0
        self.funn = 0.0
        self.espln = 0.0
        self.Epln = 0.0
        self.Esecn = 0.0
        self.Esectest = 0.0
        self.Esectest10 = 0.0
        self.Esectest13 = 0.0
        self.delen = 0.0
        self.delfn = 0.0
        self.fnewn = 0.0
        self.Enewn = 0.0
        self.esren = 0.0
        self.fren = 0.0
        self.Eren = 0.0
        self.fnewstn = 0.0
        self.Enewstn = 0.0
        self.esrestn = 0.0
        self.frestn = 0.0
        self.Erestn = 0.0
        self.eunp = 0.0
        self.funp = 0.0
        self.esplp = 0.0
        self.Eplp = 0.0
        self.Esecp = 0.0
        self.delep = 0.0
        self.delfp = 0.0
        self.fnewp = 0.0
        self.Enewp = 0.0
        self.esrep = 0.0
        self.frep = 0.0
        self.Erep = 0.0
        self.fnewstp = 0.0
        self.Enewstp = 0.0
        self.esrestp = 0.0
        self.frestp = 0.0
        self.Erestp = 0.0
        self.esi = 0.0
        self.esi10 = 0.0
        self.esi13 = 0.0
        self.fi = 0.0
        self.fi10 = 0.0
        self.fi13 = 0.0
        self.Ei = 0.0
        self.Ei10 = 0.0
        self.Ei13 = 0.0
        self.esf = 0.0
        self.esf10 = 0.0
        self.esf13 = 0.0
        self.ff = 0.0
        self.ff10 = 0.0
        self.ff13 = 0.0
        self.Ef = 0.0
        self.Ef10 = 0.0
        self.Ef13 = 0.0
        self.R = 0.0
        self.R10 = 0.0
        self.R13 = 0.0
        self.A = 0.0
        self.A10 = 0.0
        self.A13 = 0.0
        self.fc = 0.0
        self.Et = 0.0
        self.fca = 0.0
        self.Eta = 0.0
        self.fcb = 0.0
        self.Etb = 0.0
        self.fa = 0.0
        self.fb = 0.0

    fn evaluate(mut self, strain: Float64):
        self.trial.strain = strain
        if (Int(self.params.monotonic) == 1) or (self.committed.inc == 0.0):
            if self.trial.strain < 0.0:
                self.compression_envelope(self.trial.strain)
                self.trial.inc = -1.0
            elif self.trial.strain > 0.0:
                self.tension_envelope(self.trial.strain, self.committed.e0)
                self.trial.inc = 1.0
            else:
                self.trial.stress = 0.0
                self.trial.tangent = self.params.Ec
                self.trial.rule = 0.0
                self.trial.inc = 0.0
            self.trial.eunn = 0.0
            self.trial.funn = 0.0
            self.trial.eunp = 0.0
            self.trial.funp = 0.0
            self.trial.er = 0.0
            self.trial.fr = 0.0
            self.trial.er0n = 0.0
            self.trial.fr0n = 0.0
            self.trial.er0p = 0.0
            self.trial.fr0p = 0.0
            self.trial.e0 = 0.0
            self.trial.ea = 0.0
            self.trial.eb = 0.0
            self.trial.ed = 0.0
        else:
            if self.trial.strain > self.committed.strain:
                self.trial.inc = 1.0
            elif self.trial.strain < self.committed.strain:
                self.trial.inc = -1.0
            else:
                self.trial.inc = self.committed.inc
            self.esplnf(self.trial.eunn, self.trial.funn)
            self.Eplnf(self.trial.eunn)
            self.Esecnf(self.trial.eunn, self.trial.funn)
            self.delenf(self.trial.eunn)
            self.delfnf(self.trial.eunn, self.trial.funn)
            self.fnewnf(self.trial.eunn, self.trial.funn)
            self.Enewnf(self.trial.eunn, self.trial.funn)
            self.esrenf(self.trial.eunn)
            self.freErenf(self.trial.eunn)
            self.fnewstnf(self.trial.funn, self.delfn, self.trial.eunn, self.trial.er0n, self.espln)
            self.Enewstnf(self.fnewstn, self.trial.fr0n, self.trial.eunn, self.trial.er0n)
            self.esrestnf(self.trial.eunn, self.delen, self.trial.er0n, self.espln)
            self.freErestnf(self.trial.eunn, self.trial.funn, self.trial.er0n)
            self.esplpf(self.trial.eunp, self.trial.funp, self.trial.e0, self.espln)
            self.Eplpf(self.trial.e0, self.trial.eunp)
            self.Esecpf(self.trial.e0, self.trial.eunp, self.trial.funp, self.espln)
            self.delepf(self.trial.eunp, self.trial.e0)
            self.delfpf(self.trial.funp, self.trial.eunp, self.trial.e0)
            self.fnewpf(self.trial.funp, self.trial.eunp, self.trial.e0)
            self.Enewpf(self.trial.eunp, self.trial.funp, self.trial.e0, self.espln)
            self.esrepf(self.trial.eunp, self.trial.e0)
            self.freErepf(self.trial.eunp, self.trial.e0)
            self.fnewstpf(
                self.trial.funp,
                self.delfp,
                self.trial.eunp,
                self.trial.er0p,
                self.esplp,
                self.trial.e0,
            )
            self.Enewstpf(self.fnewstp, self.trial.fr0p, self.trial.eunp, self.trial.er0p)
            self.esrestpf(self.trial.eunp, self.delep, self.trial.er0p, self.esplp)
            self.freErestpf(
                self.trial.eunp, self.trial.funp, self.trial.er0p, self.trial.e0, self.espln
            )
            if self.committed.inc == (-1.0):
                # Start reversal from negative direction to positive direction
                if self.trial.strain > self.committed.strain:
                    # Rules [3,9,8,2,6]
                    if ((self.committed.rule == 1.0) or (self.committed.rule == 5.0)) or (
                        self.committed.rule == 7.0
                    ):
                        self.trial.eunn = self.committed.strain
                        self.trial.funn = self.committed.stress
                        self.update_tension_origin(
                            self.trial.e0,
                            self.trial.eunp,
                            self.trial.funp,
                            self.trial.eunn,
                            self.trial.funn,
                        )
                        self.esplnf(self.trial.eunn, self.trial.funn)
                        self.Eplnf(self.trial.eunn)
                        self.fnewpf(self.trial.funp, self.trial.eunp, self.trial.e0)
                        self.Enewpf(self.trial.eunp, self.trial.funp, self.trial.e0, self.espln)
                        self.esrepf(self.trial.eunp, self.trial.e0)
                        self.freErepf(self.trial.eunp, self.trial.e0)
                        # Rule 3
                        if self.trial.strain <= self.espln:
                            self.rule_3(self.trial.eunn, self.trial.funn, self.espln, self.Epln)
                            self.trial.rule = 3.0
                            self.apply_new_transition()
                        # Rule 9
                        elif self.trial.strain <= self.trial.eunp:
                            self.rule_9(
                                self.espln, self.Epln, self.trial.eunp, self.fnewp, self.Enewp
                            )
                            self.trial.rule = 9.0
                            self.apply_new_transition()
                        # Rule 8
                        elif self.trial.strain <= self.esrep:
                            self.rule_8(
                                self.trial.eunp,
                                self.fnewp,
                                self.Enewp,
                                self.esrep,
                                self.frep,
                                self.Erep,
                            )
                            self.trial.rule = 8.0
                            self.apply_new_transition()
                        else:
                            self.tension_envelope(self.trial.strain, self.trial.e0)
                    elif self.committed.rule == 10.0:
                        self.trial.er = self.committed.strain
                        self.trial.fr = self.committed.stress
                        self.trial.eb = self.trial.er
                        self.fb = self.trial.fr
                        self.ea1112f(
                            self.trial.eb, self.espln, self.esplp, self.trial.eunn, self.trial.eunp
                        )
                        if self.trial.ea <= self.espln:
                            # Rule 12 targeting for ea on 3
                            if self.trial.strain <= self.trial.ea:
                                self.rule_3(self.trial.eunn, self.trial.funn, self.espln, self.Epln)
                                self.transition_coefficients(
                                    self.esi, self.fi, self.Ei, self.esf, self.ff, self.Ef
                                )
                                self.transition_response(
                                    self.trial.ea,
                                    self.esi,
                                    self.fi,
                                    self.esf,
                                    self.ff,
                                    self.Ei,
                                    self.Ef,
                                    self.A,
                                    self.R,
                                )
                                self.fca = self.fc
                                self.Eta = self.Et
                                self.esi = self.trial.er
                                self.fi = self.trial.fr
                                self.Ei = self.params.Ec
                                self.esf = self.trial.ea
                                self.transition_coefficients(
                                    self.esi, self.fi, self.Ei, self.esf, self.fca, self.Eta
                                )
                                self.rule_12(
                                    self.trial.er,
                                    self.trial.fr,
                                    self.trial.ea,
                                    self.fca,
                                    self.Eta,
                                    self.A,
                                    self.R,
                                )
                                self.trial.rule = 12.0
                                self.apply_transition()
                            # Rule 3
                            elif self.trial.strain <= self.espln:
                                self.rule_3(self.trial.eunn, self.trial.funn, self.espln, self.Epln)
                                self.trial.rule = 3.0
                                self.apply_new_transition()
                            # Rule 9
                            elif self.trial.strain <= self.trial.eunp:
                                self.rule_9(
                                    self.espln, self.Epln, self.trial.eunp, self.fnewp, self.Enewp
                                )
                                self.trial.rule = 9.0
                                self.apply_new_transition()
                            # Rule 8
                            elif self.trial.strain <= self.esrep:
                                self.rule_8(
                                    self.trial.eunp,
                                    self.fnewp,
                                    self.Enewp,
                                    self.esrep,
                                    self.frep,
                                    self.Erep,
                                )
                                self.trial.rule = 8.0
                                self.apply_new_transition()
                            else:
                                self.tension_envelope(self.trial.strain, self.trial.e0)
                        elif self.trial.ea <= self.trial.eunp:
                            # Rule 12 targeting for ea on 9
                            if self.trial.strain <= self.trial.ea:
                                self.rule_9(
                                    self.espln, self.Epln, self.trial.eunp, self.fnewp, self.Enewp
                                )
                                self.transition_coefficients(
                                    self.esi, self.fi, self.Ei, self.esf, self.ff, self.Ef
                                )
                                self.transition_response(
                                    self.trial.ea,
                                    self.esi,
                                    self.fi,
                                    self.esf,
                                    self.ff,
                                    self.Ei,
                                    self.Ef,
                                    self.A,
                                    self.R,
                                )
                                self.fca = self.fc
                                self.Eta = self.Et
                                self.esi = self.trial.er
                                self.fi = self.trial.fr
                                self.Ei = self.params.Ec
                                self.esf = self.trial.ea
                                self.transition_coefficients(
                                    self.esi, self.fi, self.Ei, self.esf, self.fca, self.Eta
                                )
                                self.rule_12(
                                    self.trial.er,
                                    self.trial.fr,
                                    self.trial.ea,
                                    self.fca,
                                    self.Eta,
                                    self.A,
                                    self.R,
                                )
                                self.trial.rule = 12.0
                                self.apply_transition()
                            # Rule 9
                            elif self.trial.strain <= self.trial.eunp:
                                self.rule_9(
                                    self.espln, self.Epln, self.trial.eunp, self.fnewp, self.Enewp
                                )
                                self.trial.rule = 9.0
                                self.apply_new_transition()
                            # Rule 8
                            elif self.trial.strain <= self.esrep:
                                self.rule_8(
                                    self.trial.eunp,
                                    self.fnewp,
                                    self.Enewp,
                                    self.esrep,
                                    self.frep,
                                    self.Erep,
                                )
                                self.trial.rule = 8.0
                                self.apply_new_transition()
                            else:
                                self.tension_envelope(self.trial.strain, self.trial.e0)
                        elif self.trial.ea <= self.esrep:
                            # Rule 12 targeting for ea on 8
                            if self.trial.strain <= self.trial.ea:
                                self.rule_8(
                                    self.trial.eunp,
                                    self.fnewp,
                                    self.Enewp,
                                    self.esrep,
                                    self.frep,
                                    self.Erep,
                                )
                                self.transition_coefficients(
                                    self.esi, self.fi, self.Ei, self.esf, self.ff, self.Ef
                                )
                                self.transition_response(
                                    self.trial.ea,
                                    self.esi,
                                    self.fi,
                                    self.esf,
                                    self.ff,
                                    self.Ei,
                                    self.Ef,
                                    self.A,
                                    self.R,
                                )
                                self.fca = self.fc
                                self.Eta = self.Et
                                self.esi = self.trial.er
                                self.fi = self.trial.fr
                                self.Ei = self.params.Ec
                                self.esf = self.trial.ea
                                self.transition_coefficients(
                                    self.esi, self.fi, self.Ei, self.esf, self.fca, self.Eta
                                )
                                self.rule_12(
                                    self.trial.er,
                                    self.trial.fr,
                                    self.trial.ea,
                                    self.fca,
                                    self.Eta,
                                    self.A,
                                    self.R,
                                )
                                self.trial.rule = 12.0
                                self.apply_transition()
                            # Rule 8
                            elif self.trial.strain <= self.esrep:
                                self.rule_8(
                                    self.trial.eunp,
                                    self.fnewp,
                                    self.Enewp,
                                    self.esrep,
                                    self.frep,
                                    self.Erep,
                                )
                                self.trial.rule = 8.0
                                self.apply_new_transition()
                            else:
                                self.tension_envelope(self.trial.strain, self.trial.e0)
                        else:
                            # Rule 12 targeting for ea on 2 or 6
                            if self.trial.strain <= self.trial.ea:
                                self.tension_envelope(self.trial.ea, self.trial.e0)
                                self.fca = self.trial.stress
                                self.Eta = self.trial.tangent
                                self.esi = self.trial.er
                                self.fi = self.trial.fr
                                self.Ei = self.params.Ec
                                self.esf = self.trial.ea
                                self.transition_coefficients(
                                    self.esi, self.fi, self.Ei, self.esf, self.fca, self.Eta
                                )
                                self.rule_12(
                                    self.trial.er,
                                    self.trial.fr,
                                    self.trial.ea,
                                    self.fca,
                                    self.Eta,
                                    self.A,
                                    self.R,
                                )
                                self.trial.rule = 12.0
                                self.apply_transition()
                            else:
                                self.tension_envelope(self.trial.strain, self.trial.e0)
                    elif self.committed.rule == 11.0:
                        self.trial.er = self.committed.strain
                        self.trial.fr = self.committed.stress
                        if self.trial.eb != self.trial.er0p:
                            if self.trial.ea <= self.espln:
                                # Rule 12 targeting for ea on 3
                                if self.trial.strain <= self.trial.ea:
                                    self.rule_3(
                                        self.trial.eunn, self.trial.funn, self.espln, self.Epln
                                    )
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.ff, self.Ef
                                    )
                                    self.transition_response(
                                        self.trial.ea,
                                        self.esi,
                                        self.fi,
                                        self.esf,
                                        self.ff,
                                        self.Ei,
                                        self.Ef,
                                        self.A,
                                        self.R,
                                    )
                                    self.fca = self.fc
                                    self.Eta = self.Et
                                    self.esi = self.trial.er
                                    self.fi = self.trial.fr
                                    self.Ei = self.params.Ec
                                    self.esf = self.trial.ea
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.fca, self.Eta
                                    )
                                    self.rule_12(
                                        self.trial.er,
                                        self.trial.fr,
                                        self.trial.ea,
                                        self.fca,
                                        self.Eta,
                                        self.A,
                                        self.R,
                                    )
                                    self.trial.rule = 12.0
                                    self.apply_transition()
                                # Rule 3
                                elif self.trial.strain <= self.espln:
                                    self.rule_3(
                                        self.trial.eunn, self.trial.funn, self.espln, self.Epln
                                    )
                                    self.trial.rule = 3.0
                                    self.apply_new_transition()
                                # Rule 9
                                elif self.trial.strain <= self.trial.eunp:
                                    self.rule_9(
                                        self.espln,
                                        self.Epln,
                                        self.trial.eunp,
                                        self.fnewp,
                                        self.Enewp,
                                    )
                                    self.trial.rule = 9.0
                                    self.apply_new_transition()
                                # Rule 8
                                elif self.trial.strain <= self.esrep:
                                    self.rule_8(
                                        self.trial.eunp,
                                        self.fnewp,
                                        self.Enewp,
                                        self.esrep,
                                        self.frep,
                                        self.Erep,
                                    )
                                    self.trial.rule = 8.0
                                    self.apply_new_transition()
                                else:
                                    self.tension_envelope(self.trial.strain, self.trial.e0)
                            elif self.trial.ea <= self.trial.eunp:
                                # Rule 12 targeting for ea on 9
                                if self.trial.strain <= self.trial.ea:
                                    self.rule_9(
                                        self.espln,
                                        self.Epln,
                                        self.trial.eunp,
                                        self.fnewp,
                                        self.Enewp,
                                    )
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.ff, self.Ef
                                    )
                                    self.transition_response(
                                        self.trial.ea,
                                        self.esi,
                                        self.fi,
                                        self.esf,
                                        self.ff,
                                        self.Ei,
                                        self.Ef,
                                        self.A,
                                        self.R,
                                    )
                                    self.fca = self.fc
                                    self.Eta = self.Et
                                    self.esi = self.trial.er
                                    self.fi = self.trial.fr
                                    self.Ei = self.params.Ec
                                    self.esf = self.trial.ea
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.fca, self.Eta
                                    )
                                    self.rule_12(
                                        self.trial.er,
                                        self.trial.fr,
                                        self.trial.ea,
                                        self.fca,
                                        self.Eta,
                                        self.A,
                                        self.R,
                                    )
                                    self.trial.rule = 12.0
                                    self.apply_transition()
                                # Rule 9
                                elif self.trial.strain <= self.trial.eunp:
                                    self.rule_9(
                                        self.espln,
                                        self.Epln,
                                        self.trial.eunp,
                                        self.fnewp,
                                        self.Enewp,
                                    )
                                    self.trial.rule = 9.0
                                    self.apply_new_transition()
                                # Rule 8
                                elif self.trial.strain <= self.esrep:
                                    self.rule_8(
                                        self.trial.eunp,
                                        self.fnewp,
                                        self.Enewp,
                                        self.esrep,
                                        self.frep,
                                        self.Erep,
                                    )
                                    self.trial.rule = 8.0
                                    self.apply_new_transition()
                                else:
                                    self.tension_envelope(self.trial.strain, self.trial.e0)
                            elif self.trial.ea <= self.esrep:
                                # Rule 12 targeting for ea on 8
                                if self.trial.strain <= self.trial.ea:
                                    self.rule_8(
                                        self.trial.eunp,
                                        self.fnewp,
                                        self.Enewp,
                                        self.esrep,
                                        self.frep,
                                        self.Erep,
                                    )
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.ff, self.Ef
                                    )
                                    self.transition_response(
                                        self.trial.ea,
                                        self.esi,
                                        self.fi,
                                        self.esf,
                                        self.ff,
                                        self.Ei,
                                        self.Ef,
                                        self.A,
                                        self.R,
                                    )
                                    self.fca = self.fc
                                    self.Eta = self.Et
                                    self.esi = self.trial.er
                                    self.fi = self.trial.fr
                                    self.Ei = self.params.Ec
                                    self.esf = self.trial.ea
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.fca, self.Eta
                                    )
                                    self.rule_12(
                                        self.trial.er,
                                        self.trial.fr,
                                        self.trial.ea,
                                        self.fca,
                                        self.Eta,
                                        self.A,
                                        self.R,
                                    )
                                    self.trial.rule = 12.0
                                    self.apply_transition()
                                # Rule 8
                                elif self.trial.strain <= self.esrep:
                                    self.rule_8(
                                        self.trial.eunp,
                                        self.fnewp,
                                        self.Enewp,
                                        self.esrep,
                                        self.frep,
                                        self.Erep,
                                    )
                                    self.trial.rule = 8.0
                                    self.apply_new_transition()
                                else:
                                    self.tension_envelope(self.trial.strain, self.trial.e0)
                            else:
                                # Rule 12 targeting for ea on 2 or 6
                                if self.trial.strain <= self.trial.ea:
                                    self.tension_envelope(self.trial.ea, self.trial.e0)
                                    self.fca = self.trial.stress
                                    self.Eta = self.trial.tangent
                                    self.esi = self.trial.er
                                    self.fi = self.trial.fr
                                    self.Ei = self.params.Ec
                                    self.esf = self.trial.ea
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.fca, self.Eta
                                    )
                                    self.rule_12(
                                        self.trial.er,
                                        self.trial.fr,
                                        self.trial.ea,
                                        self.fca,
                                        self.Eta,
                                        self.A,
                                        self.R,
                                    )
                                    self.trial.rule = 12.0
                                    self.apply_transition()
                                else:
                                    self.tension_envelope(self.trial.strain, self.trial.e0)
                        else:
                            if self.trial.ea <= self.esrestp:
                                # Rule 12 targeting for ea on 88
                                if self.trial.strain <= self.trial.ea:
                                    self.rule_88(
                                        self.trial.ea,
                                        self.trial.e0,
                                        self.trial.er0p,
                                        self.trial.fr0p,
                                        self.trial.eunp,
                                        self.fnewstp,
                                        self.Enewstp,
                                        self.esrestp,
                                        self.frestp,
                                        self.Erestp,
                                    )
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.ff, self.Ef
                                    )
                                    self.transition_response(
                                        self.trial.ea,
                                        self.esi,
                                        self.fi,
                                        self.esf,
                                        self.ff,
                                        self.Ei,
                                        self.Ef,
                                        self.A,
                                        self.R,
                                    )
                                    self.esi = self.trial.er
                                    self.fi = self.trial.fr
                                    self.Ei = self.params.Ec
                                    self.esf = self.trial.ea
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.fca, self.Eta
                                    )
                                    self.rule_12(
                                        self.trial.er,
                                        self.trial.fr,
                                        self.trial.ea,
                                        self.fca,
                                        self.Eta,
                                        self.A,
                                        self.R,
                                    )
                                    self.trial.rule = 12.0
                                    self.apply_transition()
                                # Rule 88
                                elif self.trial.strain < self.esrestp:
                                    self.rule_88(
                                        self.trial.strain,
                                        self.trial.e0,
                                        self.trial.er0p,
                                        self.trial.fr0p,
                                        self.trial.eunp,
                                        self.fnewstp,
                                        self.Enewstp,
                                        self.esrestp,
                                        self.frestp,
                                        self.Erestp,
                                    )
                                    self.trial.rule = 88.0
                                    self.apply_new_transition()
                                else:
                                    self.tension_envelope(self.trial.strain, self.trial.e0)
                            else:
                                # Rule 12 targeting for ea on 2 or 6
                                if self.trial.strain <= self.trial.ea:
                                    self.tension_envelope(self.trial.ea, self.trial.e0)
                                    self.fca = self.trial.stress
                                    self.Eta = self.trial.tangent
                                    self.esi = self.trial.er
                                    self.fi = self.trial.fr
                                    self.Ei = self.params.Ec
                                    self.esf = self.trial.ea
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.fca, self.Eta
                                    )
                                    self.rule_12(
                                        self.trial.er,
                                        self.trial.fr,
                                        self.trial.ea,
                                        self.fca,
                                        self.Eta,
                                        self.A,
                                        self.R,
                                    )
                                    self.trial.rule = 12.0
                                    self.apply_transition()
                                else:
                                    self.tension_envelope(self.trial.strain, self.trial.e0)
                    elif (self.committed.rule == 13.0) or (self.committed.rule == 15.0):
                        self.trial.er = self.committed.strain
                        self.trial.fr = self.committed.stress
                        if self.committed.rule == 13.0:
                            self.trial.ea = self.trial.er
                            self.fa = self.trial.fr
                            self.eb1415f(self.trial.ea, self.fa, self.Esecn)
                        elif self.committed.rule == 15.0:
                            self.trial.ea = self.committed.ea
                            self.trial.eb = self.committed.eb
                        # Rule 14
                        if self.trial.strain <= self.trial.eb:
                            self.rule_14(self.trial.er, self.trial.fr, self.trial.eb)
                            self.trial.rule = 14.0
                            self.apply_new_transition()
                        # Rule 66
                        elif self.trial.strain < self.trial.eunp:
                            self.rule_66(self.trial.strain, self.trial.e0)
                            self.trial.rule = 66.0
                        else:
                            self.cracked_tension_envelope(self.trial.strain, self.trial.e0)
                            self.trial.rule = 6.0
                    elif self.committed.rule == 4.0:
                        self.trial.er0p = self.committed.strain
                        self.trial.fr0p = self.committed.stress
                        self.trial.eb = self.trial.er0p
                        self.fnewstpf(
                            self.trial.funp,
                            self.delfp,
                            self.trial.eunp,
                            self.trial.er0p,
                            self.esplp,
                            self.trial.e0,
                        )
                        self.Enewstpf(
                            self.fnewstp, self.trial.fr0p, self.trial.eunp, self.trial.er0p
                        )
                        self.esrestpf(self.trial.eunp, self.delep, self.trial.er0p, self.esplp)
                        self.freErestpf(
                            self.trial.eunp,
                            self.trial.funp,
                            self.trial.er0p,
                            self.trial.e0,
                            self.espln,
                        )
                        # Rule 88
                        if self.trial.strain < self.esrestp:
                            self.rule_88(
                                self.trial.strain,
                                self.trial.e0,
                                self.trial.er0p,
                                self.trial.fr0p,
                                self.trial.eunp,
                                self.fnewstp,
                                self.Enewstp,
                                self.esrestp,
                                self.frestp,
                                self.Erestp,
                            )
                            self.trial.rule = 88.0
                            self.apply_new_transition()
                        else:
                            self.tension_envelope(self.trial.strain, self.trial.e0)
                    elif self.committed.rule == 77.0:
                        if self.committed.strain >= self.trial.eunn:
                            self.trial.er = self.committed.strain
                            self.trial.fr = self.committed.stress
                            self.trial.eb = self.trial.er
                            self.trial.ea = self.trial.er0n
                            if self.trial.ea <= self.espln:
                                # Rule 12 targeting for ea on 3
                                if self.trial.strain <= self.trial.ea:
                                    self.rule_3(
                                        self.trial.eunn, self.trial.funn, self.espln, self.Epln
                                    )
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.ff, self.Ef
                                    )
                                    self.transition_response(
                                        self.trial.ea,
                                        self.esi,
                                        self.fi,
                                        self.esf,
                                        self.ff,
                                        self.Ei,
                                        self.Ef,
                                        self.A,
                                        self.R,
                                    )
                                    self.fca = self.fc
                                    self.Eta = self.Et
                                    self.esi = self.trial.er
                                    self.fi = self.trial.fr
                                    self.Ei = self.params.Ec
                                    self.esf = self.trial.ea
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.fca, self.Eta
                                    )
                                    self.rule_12(
                                        self.trial.er,
                                        self.trial.fr,
                                        self.trial.ea,
                                        self.fca,
                                        self.Eta,
                                        self.A,
                                        self.R,
                                    )
                                    self.trial.rule = 12.0
                                    self.apply_transition()
                                # Rule 3
                                elif self.trial.strain <= self.espln:
                                    self.rule_3(
                                        self.trial.eunn, self.trial.funn, self.espln, self.Epln
                                    )
                                    self.trial.rule = 3.0
                                    self.apply_new_transition()
                                # Rule 9
                                elif self.trial.strain <= self.trial.eunp:
                                    self.rule_9(
                                        self.espln,
                                        self.Epln,
                                        self.trial.eunp,
                                        self.fnewp,
                                        self.Enewp,
                                    )
                                    self.trial.rule = 9.0
                                    self.apply_new_transition()
                                # Rule 8
                                elif self.trial.strain <= self.esrep:
                                    self.rule_8(
                                        self.trial.eunp,
                                        self.fnewp,
                                        self.Enewp,
                                        self.esrep,
                                        self.frep,
                                        self.Erep,
                                    )
                                    self.trial.rule = 8.0
                                    self.apply_new_transition()
                                else:
                                    self.tension_envelope(self.trial.strain, self.trial.e0)
                            elif self.trial.ea <= self.trial.eunp:
                                # Rule 12 targeting for ea on 9
                                if self.trial.strain <= self.trial.ea:
                                    self.rule_9(
                                        self.espln,
                                        self.Epln,
                                        self.trial.eunp,
                                        self.fnewp,
                                        self.Enewp,
                                    )
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.ff, self.Ef
                                    )
                                    self.transition_response(
                                        self.trial.ea,
                                        self.esi,
                                        self.fi,
                                        self.esf,
                                        self.ff,
                                        self.Ei,
                                        self.Ef,
                                        self.A,
                                        self.R,
                                    )
                                    self.fca = self.fc
                                    self.Eta = self.Et
                                    self.esi = self.trial.er
                                    self.fi = self.trial.fr
                                    self.Ei = self.params.Ec
                                    self.esf = self.trial.ea
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.fca, self.Eta
                                    )
                                    self.rule_12(
                                        self.trial.er,
                                        self.trial.fr,
                                        self.trial.ea,
                                        self.fca,
                                        self.Eta,
                                        self.A,
                                        self.R,
                                    )
                                    self.trial.rule = 12.0
                                    self.apply_transition()
                                # Rule 9
                                elif self.trial.strain <= self.trial.eunp:
                                    self.rule_9(
                                        self.espln,
                                        self.Epln,
                                        self.trial.eunp,
                                        self.fnewp,
                                        self.Enewp,
                                    )
                                    self.trial.rule = 9.0
                                    self.apply_new_transition()
                                # Rule 8
                                elif self.trial.strain <= self.esrep:
                                    self.rule_8(
                                        self.trial.eunp,
                                        self.fnewp,
                                        self.Enewp,
                                        self.esrep,
                                        self.frep,
                                        self.Erep,
                                    )
                                    self.trial.rule = 8.0
                                    self.apply_new_transition()
                                else:
                                    self.tension_envelope(self.trial.strain, self.trial.e0)
                            elif self.trial.ea <= self.esrep:
                                # Rule 12 targeting for ea on 8
                                if self.trial.strain <= self.trial.ea:
                                    self.rule_8(
                                        self.trial.eunp,
                                        self.fnewp,
                                        self.Enewp,
                                        self.esrep,
                                        self.frep,
                                        self.Erep,
                                    )
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.ff, self.Ef
                                    )
                                    self.transition_response(
                                        self.trial.ea,
                                        self.esi,
                                        self.fi,
                                        self.esf,
                                        self.ff,
                                        self.Ei,
                                        self.Ef,
                                        self.A,
                                        self.R,
                                    )
                                    self.fca = self.fc
                                    self.Eta = self.Et
                                    self.esi = self.trial.er
                                    self.fi = self.trial.fr
                                    self.Ei = self.params.Ec
                                    self.esf = self.trial.ea
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.fca, self.Eta
                                    )
                                    self.rule_12(
                                        self.trial.er,
                                        self.trial.fr,
                                        self.trial.ea,
                                        self.fca,
                                        self.Eta,
                                        self.A,
                                        self.R,
                                    )
                                    self.trial.rule = 12.0
                                    self.apply_transition()
                                # Rule 8
                                elif self.trial.strain <= self.esrep:
                                    self.rule_8(
                                        self.trial.eunp,
                                        self.fnewp,
                                        self.Enewp,
                                        self.esrep,
                                        self.frep,
                                        self.Erep,
                                    )
                                    self.trial.rule = 8.0
                                    self.apply_new_transition()
                                else:
                                    self.tension_envelope(self.trial.strain, self.trial.e0)
                            else:
                                # Rule 12 targeting for ea on 2 or 6
                                if self.trial.strain <= self.trial.ea:
                                    self.tension_envelope(self.trial.ea, self.trial.e0)
                                    self.fca = self.trial.stress
                                    self.Eta = self.trial.tangent
                                    self.esi = self.trial.er
                                    self.fi = self.trial.fr
                                    self.Ei = self.params.Ec
                                    self.esf = self.trial.ea
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.fca, self.Eta
                                    )
                                    self.rule_12(
                                        self.trial.er,
                                        self.trial.fr,
                                        self.trial.ea,
                                        self.fca,
                                        self.Eta,
                                        self.A,
                                        self.R,
                                    )
                                    self.trial.rule = 12.0
                                    self.apply_transition()
                                else:
                                    self.tension_envelope(self.trial.strain, self.trial.e0)
                        else:
                            self.trial.eunn = self.committed.strain
                            self.trial.funn = self.committed.stress
                            self.update_tension_origin(
                                self.trial.e0,
                                self.trial.eunp,
                                self.trial.funp,
                                self.trial.eunn,
                                self.trial.funn,
                            )
                            self.esplnf(self.trial.eunn, self.trial.funn)
                            self.Eplnf(self.trial.eunn)
                            self.fnewpf(self.trial.funp, self.trial.eunp, self.trial.e0)
                            self.Enewpf(self.trial.eunp, self.trial.funp, self.trial.e0, self.espln)
                            self.esrepf(self.trial.eunp, self.trial.e0)
                            self.freErepf(self.trial.eunp, self.trial.e0)
                            # Rule 3
                            if self.trial.strain <= self.espln:
                                self.rule_3(self.trial.eunn, self.trial.funn, self.espln, self.Epln)
                                self.trial.rule = 3.0
                                self.apply_new_transition()
                            # Rule 9
                            elif self.trial.strain <= self.trial.eunp:
                                self.rule_9(
                                    self.espln, self.Epln, self.trial.eunp, self.fnewp, self.Enewp
                                )
                                self.trial.rule = 9.0
                                self.apply_new_transition()
                            # Rule 8
                            elif self.trial.strain <= self.esrep:
                                self.rule_8(
                                    self.trial.eunp,
                                    self.fnewp,
                                    self.Enewp,
                                    self.esrep,
                                    self.frep,
                                    self.Erep,
                                )
                                self.trial.rule = 8.0
                                self.apply_new_transition()
                            else:
                                self.tension_envelope(self.trial.strain, self.trial.e0)
                else:
                    if ((self.committed.rule == 4.0) or (self.committed.rule == 10.0)) or (
                        self.committed.rule == 7.0
                    ):
                        if self.trial.strain >= self.esplp:
                            self.rule_4(self.trial.eunp, self.trial.funp, self.esplp, self.Eplp)
                            self.trial.rule = 4.0
                            self.apply_new_transition()
                        elif self.trial.strain >= self.trial.eunn:
                            self.rule_10(
                                self.esplp, self.Eplp, self.trial.eunn, self.fnewn, self.Enewn
                            )
                            self.trial.rule = 10.0
                            self.transition_coefficients(
                                self.esi, self.fi, self.Ei, self.esf, self.ff, self.Ef
                            )
                            self.transition_response(
                                self.trial.strain,
                                self.esi,
                                self.fi,
                                self.esf,
                                self.ff,
                                self.Ei,
                                self.Ef,
                                self.A,
                                self.R,
                            )
                            self.Esectest = (self.ff - self.fi) / (self.esf - self.esi)
                            if self.Et == self.Esectest:
                                if self.trial.strain >= self.espln:
                                    self.fc = 0
                                    self.Et = 0
                                    self.trial.rule = 10.0
                                else:
                                    self.Et = self.Enewn
                                    self.fc = self.Et * (self.trial.strain - self.espln)
                                    self.trial.rule = 10.0
                            self.trial.stress = self.fc
                            self.trial.tangent = self.Et
                        elif self.trial.strain >= self.esren:
                            self.rule_7(
                                self.trial.eunn,
                                self.fnewn,
                                self.Enewn,
                                self.esren,
                                self.fren,
                                self.Eren,
                            )
                            self.trial.rule = 7.0
                            self.apply_new_transition()
                        else:
                            self.compression_envelope(self.trial.strain)
                    elif (self.committed.rule == 1.0) or (self.committed.rule == 5.0):
                        self.compression_envelope(self.trial.strain)
                    # Continue on transition 77 [Rules 77,1,5]
                    elif self.committed.rule == 77.0:
                        # Rule 77
                        if self.trial.strain > self.esrestn:
                            self.rule_77(
                                self.trial.strain,
                                self.trial.e0,
                                self.trial.er0n,
                                self.trial.fr0n,
                                self.trial.eunn,
                                self.fnewstn,
                                self.Enewstn,
                                self.esrestn,
                                self.frestn,
                                self.Erestn,
                            )
                            self.trial.rule = 77.0
                            self.apply_new_transition()
                        else:
                            self.compression_envelope(self.trial.strain)
                    # Continue on transition 13 [Rules 13,7,1,5]
                    elif self.committed.rule == 13.0:
                        # Rule 13
                        if self.trial.strain >= self.trial.eunn:
                            self.rule_13(self.trial.ed, self.trial.eunn, self.fnewn, self.Enewn)
                            self.trial.rule = 13.0
                            self.transition_coefficients(
                                self.esi, self.fi, self.Ei, self.esf, self.ff, self.Ef
                            )
                            self.transition_response(
                                self.trial.strain,
                                self.esi,
                                self.fi,
                                self.esf,
                                self.ff,
                                self.Ei,
                                self.Ef,
                                self.A,
                                self.R,
                            )
                            self.Esectest = (self.ff - self.fi) / (self.esf - self.esi)
                            if self.Et == self.Esectest:
                                if self.trial.strain >= self.espln:
                                    self.fc = 0.0
                                    self.Et = 0.0
                                    self.trial.rule = 13.0
                                else:
                                    self.Et = self.Enewn
                                    self.fc = self.Et * (self.trial.strain - self.espln)
                                    self.trial.rule = 13.0
                            self.trial.stress = self.fc
                            self.trial.tangent = self.Et
                        # Rule 7
                        elif self.trial.strain >= self.esren:
                            self.rule_7(
                                self.trial.eunn,
                                self.fnewn,
                                self.Enewn,
                                self.esren,
                                self.fren,
                                self.Eren,
                            )
                            self.trial.rule = 7.0
                            self.apply_new_transition()
                        else:
                            self.compression_envelope(self.trial.strain)
                    # Continue on transition 11 [Rules 11,(4,10 or 10),7,1,5 or Rules 11,77,1,5]
                    elif self.committed.rule == 11.0:
                        if self.trial.ea != self.trial.er0n:
                            if self.trial.eb >= self.esplp:
                                # Rule 11 targeting for eb on 4
                                if self.trial.strain >= self.trial.eb:
                                    self.rule_4(
                                        self.trial.eunp, self.trial.funp, self.esplp, self.Eplp
                                    )
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.ff, self.Ef
                                    )
                                    self.transition_response(
                                        self.trial.eb,
                                        self.esi,
                                        self.fi,
                                        self.esf,
                                        self.ff,
                                        self.Ei,
                                        self.Ef,
                                        self.A,
                                        self.R,
                                    )
                                    self.fcb = self.fc
                                    self.Etb = self.Et
                                    self.esi = self.trial.er
                                    self.fi = self.trial.fr
                                    self.Ei = self.params.Ec
                                    self.esf = self.trial.eb
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.fcb, self.Etb
                                    )
                                    self.rule_11(
                                        self.trial.er,
                                        self.trial.fr,
                                        self.trial.eb,
                                        self.fcb,
                                        self.Etb,
                                        self.A,
                                        self.R,
                                    )
                                    self.trial.rule = 11.0
                                    self.apply_transition()
                                # Rule 4
                                elif self.trial.strain >= self.esplp:
                                    self.rule_4(
                                        self.trial.eunp, self.trial.funp, self.esplp, self.Eplp
                                    )
                                    self.trial.rule = 4.0
                                    self.apply_new_transition()
                                # Rule 10
                                elif self.trial.strain >= self.trial.eunn:
                                    self.rule_10(
                                        self.esplp,
                                        self.Eplp,
                                        self.trial.eunn,
                                        self.fnewn,
                                        self.Enewn,
                                    )
                                    self.trial.rule = 10.0
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.ff, self.Ef
                                    )
                                    self.transition_response(
                                        self.trial.strain,
                                        self.esi,
                                        self.fi,
                                        self.esf,
                                        self.ff,
                                        self.Ei,
                                        self.Ef,
                                        self.A,
                                        self.R,
                                    )
                                    self.Esectest = (self.ff - self.fi) / (self.esf - self.esi)
                                    if self.Et == self.Esectest:
                                        if self.trial.strain >= self.espln:
                                            self.fc = 0
                                            self.Et = 0
                                            self.trial.rule = 10.0
                                        else:
                                            self.Et = self.Enewn
                                            self.fc = self.Et * (self.trial.strain - self.espln)
                                            self.trial.rule = 10.0
                                    self.trial.stress = self.fc
                                    self.trial.tangent = self.Et
                                # Rule 7
                                elif self.trial.strain >= self.esren:
                                    self.rule_7(
                                        self.trial.eunn,
                                        self.fnewn,
                                        self.Enewn,
                                        self.esren,
                                        self.fren,
                                        self.Eren,
                                    )
                                    self.trial.rule = 7.0
                                    self.apply_new_transition()
                                else:
                                    self.compression_envelope(self.trial.strain)
                            elif self.trial.eb >= self.trial.eunn:
                                # Rule 11 targeting for eb on 10
                                if self.trial.strain >= self.trial.eb:
                                    self.rule_10(
                                        self.esplp,
                                        self.Eplp,
                                        self.trial.eunn,
                                        self.fnewn,
                                        self.Enewn,
                                    )
                                    self.esi10 = self.esi
                                    self.fi10 = self.fi
                                    self.Ei10 = self.Ei
                                    self.esf10 = self.esf
                                    self.ff10 = self.ff
                                    self.Ef10 = self.Ef
                                    self.transition_coefficients(
                                        self.esi10,
                                        self.fi10,
                                        self.Ei10,
                                        self.esf10,
                                        self.ff10,
                                        self.Ef10,
                                    )
                                    self.R10 = self.R
                                    self.A10 = self.A
                                    self.transition_response(
                                        self.trial.eb,
                                        self.esi10,
                                        self.fi10,
                                        self.esf10,
                                        self.ff10,
                                        self.Ei10,
                                        self.Ef10,
                                        self.A10,
                                        self.R10,
                                    )
                                    self.fcb = self.fc
                                    self.Etb = self.Et
                                    self.Esectest10 = (self.ff10 - self.fi10) / (
                                        self.esf10 - self.esi10
                                    )
                                    if self.Etb == self.Esectest10:
                                        if self.trial.eb >= self.espln:
                                            self.fcb = 0
                                            self.Etb = 0
                                        else:
                                            self.Etb = self.Enewn
                                            self.fcb = self.Etb * (self.trial.eb - self.espln)
                                    self.esi = self.trial.er
                                    self.fi = self.trial.fr
                                    self.Ei = self.params.Ec
                                    self.esf = self.trial.eb
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.fcb, self.Etb
                                    )
                                    self.rule_11(
                                        self.trial.er,
                                        self.trial.fr,
                                        self.trial.eb,
                                        self.fcb,
                                        self.Etb,
                                        self.A,
                                        self.R,
                                    )
                                    self.trial.rule = 11.0
                                    self.apply_transition()
                                # Rule 10
                                elif self.trial.strain >= self.trial.eunn:
                                    self.rule_10(
                                        self.esplp,
                                        self.Eplp,
                                        self.trial.eunn,
                                        self.fnewn,
                                        self.Enewn,
                                    )
                                    self.trial.rule = 10.0
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.ff, self.Ef
                                    )
                                    self.transition_response(
                                        self.trial.strain,
                                        self.esi,
                                        self.fi,
                                        self.esf,
                                        self.ff,
                                        self.Ei,
                                        self.Ef,
                                        self.A,
                                        self.R,
                                    )
                                    self.Esectest = (self.ff - self.fi) / (self.esf - self.esi)
                                    if self.Et == self.Esectest:
                                        if self.trial.strain >= self.espln:
                                            self.fc = 0
                                            self.Et = 0
                                            self.trial.rule = 10.0
                                        else:
                                            self.Et = self.Enewn
                                            self.fc = self.Et * (self.trial.strain - self.espln)
                                            self.trial.rule = 10.0
                                    self.trial.stress = self.fc
                                    self.trial.tangent = self.Et
                                # Rule 7
                                elif self.trial.strain >= self.esren:
                                    self.rule_7(
                                        self.trial.eunn,
                                        self.fnewn,
                                        self.Enewn,
                                        self.esren,
                                        self.fren,
                                        self.Eren,
                                    )
                                    self.trial.rule = 7.0
                                    self.apply_new_transition()
                                else:
                                    self.compression_envelope(self.trial.strain)
                            elif self.trial.eb >= self.esren:
                                # Rule 11 targeting for eb on 7
                                if self.trial.strain >= self.trial.eb:
                                    self.rule_7(
                                        self.trial.eunn,
                                        self.fnewn,
                                        self.Enewn,
                                        self.esren,
                                        self.fren,
                                        self.Eren,
                                    )
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.ff, self.Ef
                                    )
                                    self.transition_response(
                                        self.trial.eb,
                                        self.esi,
                                        self.fi,
                                        self.esf,
                                        self.ff,
                                        self.Ei,
                                        self.Ef,
                                        self.A,
                                        self.R,
                                    )
                                    self.fcb = self.fc
                                    self.Etb = self.Et
                                    self.esi = self.trial.er
                                    self.fi = self.trial.fr
                                    self.Ei = self.params.Ec
                                    self.esf = self.trial.eb
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.fcb, self.Etb
                                    )
                                    self.rule_11(
                                        self.trial.er,
                                        self.trial.fr,
                                        self.trial.eb,
                                        self.fcb,
                                        self.Etb,
                                        self.A,
                                        self.R,
                                    )
                                    self.trial.rule = 11.0
                                    self.apply_transition()
                                # Rule 7
                                elif self.trial.strain >= self.esren:
                                    self.rule_7(
                                        self.trial.eunn,
                                        self.fnewn,
                                        self.Enewn,
                                        self.esren,
                                        self.fren,
                                        self.Eren,
                                    )
                                    self.trial.rule = 7.0
                                    self.apply_new_transition()
                                else:
                                    self.compression_envelope(self.trial.strain)
                            else:
                                # Rule 11 targeting for eb on 1 or 5
                                if self.trial.strain >= self.trial.eb:
                                    self.compression_envelope(self.trial.eb)
                                    self.fcb = self.trial.stress
                                    self.Etb = self.trial.tangent
                                    self.esi = self.trial.er
                                    self.fi = self.trial.fr
                                    self.Ei = self.params.Ec
                                    self.esf = self.trial.eb
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.fcb, self.Etb
                                    )
                                    self.rule_11(
                                        self.trial.er,
                                        self.trial.fr,
                                        self.trial.eb,
                                        self.fcb,
                                        self.Eta,
                                        self.A,
                                        self.R,
                                    )
                                    self.trial.rule = 11.0
                                    self.apply_transition()
                                else:
                                    self.compression_envelope(self.trial.strain)
                        else:
                            if self.trial.eb >= self.esrestn:
                                # Rule 11 targeting for eb on 77
                                if self.trial.strain >= self.trial.eb:
                                    self.rule_77(
                                        self.trial.eb,
                                        self.trial.e0,
                                        self.trial.er0n,
                                        self.trial.fr0n,
                                        self.trial.eunn,
                                        self.fnewstn,
                                        self.Enewstn,
                                        self.esrestn,
                                        self.frestn,
                                        self.Erestn,
                                    )
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.ff, self.Ef
                                    )
                                    self.transition_response(
                                        self.trial.eb,
                                        self.esi,
                                        self.fi,
                                        self.esf,
                                        self.ff,
                                        self.Ei,
                                        self.Ef,
                                        self.A,
                                        self.R,
                                    )
                                    self.fcb = self.fc
                                    self.Etb = self.Et
                                    self.esi = self.trial.er
                                    self.fi = self.trial.fr
                                    self.Ei = self.params.Ec
                                    self.esf = self.trial.eb
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.fcb, self.Etb
                                    )
                                    self.rule_11(
                                        self.trial.er,
                                        self.trial.fr,
                                        self.trial.eb,
                                        self.fcb,
                                        self.Etb,
                                        self.A,
                                        self.R,
                                    )
                                    self.trial.rule = 11.0
                                    self.apply_transition()
                                # Rule 77
                                elif self.trial.strain > self.esrestn:
                                    self.rule_77(
                                        self.trial.strain,
                                        self.trial.e0,
                                        self.trial.er0n,
                                        self.trial.fr0n,
                                        self.trial.eunn,
                                        self.fnewstn,
                                        self.Enewstn,
                                        self.esrestn,
                                        self.frestn,
                                        self.Erestn,
                                    )
                                    self.trial.rule = 77.0
                                    self.apply_new_transition()
                                else:
                                    self.compression_envelope(self.trial.strain)
                            else:
                                # Rule 11 targeting for eb on 1 or 5
                                if self.trial.strain >= self.trial.eb:
                                    self.compression_envelope(self.trial.eb)
                                    self.fcb = self.trial.stress
                                    self.Etb = self.trial.tangent
                                    self.esi = self.trial.er
                                    self.fi = self.trial.fr
                                    self.Ei = self.params.Ec
                                    self.esf = self.trial.eb
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.fcb, self.Etb
                                    )
                                    self.rule_11(
                                        self.trial.er,
                                        self.trial.fr,
                                        self.trial.eb,
                                        self.fcb,
                                        self.Etb,
                                        self.A,
                                        self.R,
                                    )
                                    self.trial.rule = 11.0
                                    self.apply_transition()
                                else:
                                    self.compression_envelope(self.trial.strain)
                    # Continue on transition 15 [Rules 15,13,7,1,5]
                    elif self.committed.rule == 15.0:
                        # Rule 15 targeting for ea (ed) on 13
                        if self.trial.strain >= self.trial.ea:
                            self.rule_13(self.trial.ed, self.trial.eunn, self.fnewn, self.Enewn)
                            self.esi13 = self.esi
                            self.fi13 = self.fi
                            self.Ei13 = self.Ei
                            self.esf13 = self.esf
                            self.ff13 = self.ff
                            self.Ef13 = self.Ef
                            self.transition_coefficients(
                                self.esi13, self.fi13, self.Ei13, self.esf13, self.ff13, self.Ef13
                            )
                            self.R13 = self.R
                            self.A13 = self.A
                            self.transition_response(
                                self.trial.ea,
                                self.esi13,
                                self.fi13,
                                self.esf13,
                                self.ff13,
                                self.Ei13,
                                self.Ef13,
                                self.A13,
                                self.R13,
                            )
                            self.fca = self.fc
                            self.Eta = self.Et
                            self.Esectest13 = (self.ff13 - self.fi13) / (self.esf13 - self.esi13)
                            if self.Eta == self.Esectest13:
                                if self.trial.ea >= self.espln:
                                    self.fca = 0
                                    self.Eta = 0
                                else:
                                    self.Eta = self.Enewn
                                    self.fca = self.Et * (self.trial.ea - self.espln)
                            self.esi = self.trial.er
                            self.fi = self.trial.fr
                            self.Ei = self.params.Ec
                            self.esf = self.trial.ea
                            self.transition_coefficients(
                                self.esi, self.fi, self.Ei, self.esf, self.fca, self.Eta
                            )
                            self.rule_15(
                                self.trial.er,
                                self.trial.fr,
                                self.trial.ea,
                                self.fca,
                                self.Eta,
                                self.A,
                                self.R,
                            )
                            self.trial.rule = 15.0
                            self.apply_transition()
                        # Rule 13
                        elif self.trial.strain >= self.trial.eunn:
                            self.rule_13(self.trial.ed, self.trial.eunn, self.fnewn, self.Enewn)
                            self.trial.rule = 13.0
                            self.transition_coefficients(
                                self.esi, self.fi, self.Ei, self.esf, self.ff, self.Ef
                            )
                            self.transition_response(
                                self.trial.strain,
                                self.esi,
                                self.fi,
                                self.esf,
                                self.ff,
                                self.Ei,
                                self.Ef,
                                self.A,
                                self.R,
                            )
                            self.Esectest = (self.ff - self.fi) / (self.esf - self.esi)
                            if self.Et == self.Esectest:
                                if self.trial.strain >= self.espln:
                                    self.fc = 0
                                    self.Et = 0
                                    self.trial.rule = 13.0
                                else:
                                    self.Et = self.Enewn
                                    self.fc = self.Et * (self.trial.strain - self.espln)
                                    self.trial.rule = 13.0
                            self.trial.stress = self.fc
                            self.trial.tangent = self.Et
                        # Rule 7
                        elif self.trial.strain >= self.esren:
                            self.rule_7(
                                self.trial.eunn,
                                self.fnewn,
                                self.Enewn,
                                self.esren,
                                self.fren,
                                self.Eren,
                            )
                            self.trial.rule = 7.0
                            self.apply_new_transition()
                        else:
                            self.compression_envelope(self.trial.strain)
            else:
                if self.trial.strain < self.committed.strain:
                    if abs(self.committed.stress) == 0.0:
                        self.trial.eunp = self.committed.strain
                        self.trial.funp = self.committed.stress
                        self.compression_envelope(self.trial.eunn)
                        self.trial.funn = self.trial.stress
                        self.trial.er = self.committed.strain
                        self.trial.fr = self.committed.stress
                        self.trial.ed = self.trial.er
                        self.fnewnf(self.trial.eunn, self.trial.funn)
                        self.Enewnf(self.trial.eunn, self.trial.funn)
                        if self.trial.strain >= self.trial.eunn:
                            self.rule_13(self.trial.ed, self.trial.eunn, self.fnewn, self.Enewn)
                            self.trial.rule = 13.0
                            self.transition_coefficients(
                                self.esi, self.fi, self.Ei, self.esf, self.ff, self.Ef
                            )
                            self.transition_response(
                                self.trial.strain,
                                self.esi,
                                self.fi,
                                self.esf,
                                self.ff,
                                self.Ei,
                                self.Ef,
                                self.A,
                                self.R,
                            )
                            self.Esectest = (self.ff - self.fi) / (self.esf - self.esi)
                            if self.Et == self.Esectest:
                                if self.trial.strain >= self.espln:
                                    self.fc = 0
                                    self.Et = 0
                                    self.trial.rule = 13.0
                                else:
                                    self.Et = self.Enewn
                                    self.fc = self.Et * (self.trial.strain - self.espln)
                                    self.trial.rule = 13.0
                            self.trial.stress = self.fc
                            self.trial.tangent = self.Et
                        elif self.trial.strain >= self.esren:
                            self.rule_7(
                                self.trial.eunn,
                                self.fnewn,
                                self.Enewn,
                                self.esren,
                                self.fren,
                                self.Eren,
                            )
                            self.trial.rule = 7.0
                            self.apply_new_transition()
                        else:
                            self.compression_envelope(self.trial.strain)
                    elif (self.committed.rule == 2.0) or (self.committed.rule == 8.0):
                        self.trial.eunp = self.committed.strain
                        self.tension_envelope(self.trial.eunp, self.trial.e0)
                        self.trial.funp = self.trial.stress
                        self.Esecpf(self.trial.e0, self.trial.eunp, self.trial.funp, self.espln)
                        self.esplpf(self.trial.eunp, self.trial.funp, self.trial.e0, self.espln)
                        self.Eplpf(self.trial.e0, self.trial.eunp)
                        if self.trial.strain >= self.esplp:
                            self.rule_4(self.trial.eunp, self.trial.funp, self.esplp, self.Eplp)
                            self.trial.rule = 4.0
                            self.apply_new_transition()
                        elif self.trial.strain >= self.trial.eunn:
                            self.rule_10(
                                self.esplp, self.Eplp, self.trial.eunn, self.fnewn, self.Enewn
                            )
                            self.trial.rule = 10.0
                            self.transition_coefficients(
                                self.esi, self.fi, self.Ei, self.esf, self.ff, self.Ef
                            )
                            self.transition_response(
                                self.trial.strain,
                                self.esi,
                                self.fi,
                                self.esf,
                                self.ff,
                                self.Ei,
                                self.Ef,
                                self.A,
                                self.R,
                            )
                            self.Esectest = (self.ff - self.fi) / (self.esf - self.esi)
                            if self.Et == self.Esectest:
                                if self.trial.strain >= self.espln:
                                    self.fc = 0
                                    self.Et = 0
                                    self.trial.rule = 10.0
                                else:
                                    self.Et = self.Enewn
                                    self.fc = self.Et * (self.trial.strain - self.espln)
                                    self.trial.rule = 10.0
                            self.trial.stress = self.fc
                            self.trial.tangent = self.Et
                        elif self.trial.strain >= self.esren:
                            self.rule_7(
                                self.trial.eunn,
                                self.fnewn,
                                self.Enewn,
                                self.esren,
                                self.fren,
                                self.Eren,
                            )
                            self.trial.rule = 7.0
                            self.apply_new_transition()
                        else:
                            self.compression_envelope(self.trial.strain)
                    elif self.committed.rule == 9.0:
                        self.trial.er = self.committed.strain
                        self.trial.fr = self.committed.stress
                        self.trial.ea = self.trial.er
                        self.fa = self.trial.fr
                        self.eb1112f(
                            self.trial.ea, self.espln, self.esplp, self.trial.eunn, self.trial.eunp
                        )
                        if self.trial.eb >= self.esplp:
                            # Rule 11 targeting for eb on 4
                            if self.trial.strain >= self.trial.eb:
                                self.rule_4(self.trial.eunp, self.trial.funp, self.esplp, self.Eplp)
                                self.transition_coefficients(
                                    self.esi, self.fi, self.Ei, self.esf, self.ff, self.Ef
                                )
                                self.transition_response(
                                    self.trial.eb,
                                    self.esi,
                                    self.fi,
                                    self.esf,
                                    self.ff,
                                    self.Ei,
                                    self.Ef,
                                    self.A,
                                    self.R,
                                )
                                self.fcb = self.fc
                                self.Etb = self.Et
                                self.esi = self.trial.er
                                self.fi = self.trial.fr
                                self.Ei = self.params.Ec
                                self.esf = self.trial.eb
                                self.transition_coefficients(
                                    self.esi, self.fi, self.Ei, self.esf, self.fcb, self.Etb
                                )
                                self.rule_11(
                                    self.trial.er,
                                    self.trial.fr,
                                    self.trial.eb,
                                    self.fcb,
                                    self.Etb,
                                    self.A,
                                    self.R,
                                )
                                self.trial.rule = 11.0
                                self.apply_transition()
                            # Rule 4
                            elif self.trial.strain >= self.esplp:
                                self.rule_4(self.trial.eunp, self.trial.funp, self.esplp, self.Eplp)
                                self.trial.rule = 4.0
                                self.apply_new_transition()
                            # Rule 10
                            elif self.trial.strain >= self.trial.eunn:
                                self.rule_10(
                                    self.esplp, self.Eplp, self.trial.eunn, self.fnewn, self.Enewn
                                )
                                self.trial.rule = 10.0
                                self.transition_coefficients(
                                    self.esi, self.fi, self.Ei, self.esf, self.ff, self.Ef
                                )
                                self.transition_response(
                                    self.trial.strain,
                                    self.esi,
                                    self.fi,
                                    self.esf,
                                    self.ff,
                                    self.Ei,
                                    self.Ef,
                                    self.A,
                                    self.R,
                                )
                                self.Esectest = (self.ff - self.fi) / (self.esf - self.esi)
                                if self.Et == self.Esectest:
                                    if self.trial.strain >= self.espln:
                                        self.fc = 0
                                        self.Et = 0
                                        self.trial.rule = 10.0
                                    else:
                                        self.Et = self.Enewn
                                        self.fc = self.Et * (self.trial.strain - self.espln)
                                        self.trial.rule = 10.0
                                self.trial.stress = self.fc
                                self.trial.tangent = self.Et
                            # Rule 7
                            elif self.trial.strain >= self.esren:
                                self.rule_7(
                                    self.trial.eunn,
                                    self.fnewn,
                                    self.Enewn,
                                    self.esren,
                                    self.fren,
                                    self.Eren,
                                )
                                self.trial.rule = 7.0
                                self.apply_new_transition()
                            else:
                                self.compression_envelope(self.trial.strain)
                        elif self.trial.eb >= self.trial.eunn:
                            # Rule 11 targeting for eb on 10
                            if self.trial.strain >= self.trial.eb:
                                self.rule_10(
                                    self.esplp, self.Eplp, self.trial.eunn, self.fnewn, self.Enewn
                                )
                                self.esi10 = self.esi
                                self.fi10 = self.fi
                                self.Ei10 = self.Ei
                                self.esf10 = self.esf
                                self.ff10 = self.ff
                                self.Ef10 = self.Ef
                                self.transition_coefficients(
                                    self.esi10,
                                    self.fi10,
                                    self.Ei10,
                                    self.esf10,
                                    self.ff10,
                                    self.Ef10,
                                )
                                self.R10 = self.R
                                self.A10 = self.A
                                self.transition_response(
                                    self.trial.eb,
                                    self.esi10,
                                    self.fi10,
                                    self.esf10,
                                    self.ff10,
                                    self.Ei10,
                                    self.Ef10,
                                    self.A10,
                                    self.R10,
                                )
                                self.fcb = self.fc
                                self.Etb = self.Et
                                self.Esectest10 = (self.ff10 - self.fi10) / (
                                    self.esf10 - self.esi10
                                )
                                if self.Etb == self.Esectest10:
                                    if self.trial.eb >= self.espln:
                                        self.fcb = 0
                                        self.Etb = 0
                                    else:
                                        self.Etb = self.Enewn
                                        self.fcb = self.Etb * (self.trial.eb - self.espln)
                                self.esi = self.trial.er
                                self.fi = self.trial.fr
                                self.Ei = self.params.Ec
                                self.esf = self.trial.eb
                                self.transition_coefficients(
                                    self.esi, self.fi, self.Ei, self.esf, self.fcb, self.Etb
                                )
                                self.rule_11(
                                    self.trial.er,
                                    self.trial.fr,
                                    self.trial.eb,
                                    self.fcb,
                                    self.Etb,
                                    self.A,
                                    self.R,
                                )
                                self.trial.rule = 11.0
                                self.apply_transition()
                            # Rule 10
                            elif self.trial.strain >= self.trial.eunn:
                                self.rule_10(
                                    self.esplp, self.Eplp, self.trial.eunn, self.fnewn, self.Enewn
                                )
                                self.trial.rule = 10.0
                                self.transition_coefficients(
                                    self.esi, self.fi, self.Ei, self.esf, self.ff, self.Ef
                                )
                                self.transition_response(
                                    self.trial.strain,
                                    self.esi,
                                    self.fi,
                                    self.esf,
                                    self.ff,
                                    self.Ei,
                                    self.Ef,
                                    self.A,
                                    self.R,
                                )
                                self.Esectest = (self.ff - self.fi) / (self.esf - self.esi)
                                if self.Et == self.Esectest:
                                    if self.trial.strain >= self.espln:
                                        self.fc = 0
                                        self.Et = 0
                                        self.trial.rule = 10.0
                                    else:
                                        self.Et = self.Enewn
                                        self.fc = self.Et * (self.trial.strain - self.espln)
                                        self.trial.rule = 10.0
                                self.trial.stress = self.fc
                                self.trial.tangent = self.Et
                            # Rule 7
                            elif self.trial.strain >= self.esren:
                                self.rule_7(
                                    self.trial.eunn,
                                    self.fnewn,
                                    self.Enewn,
                                    self.esren,
                                    self.fren,
                                    self.Eren,
                                )
                                self.trial.rule = 7.0
                                self.apply_new_transition()
                            else:
                                self.compression_envelope(self.trial.strain)
                        elif self.trial.eb >= self.esren:
                            # Rule 11 targeting for eb on 7
                            if self.trial.strain >= self.trial.eb:
                                self.rule_7(
                                    self.trial.eunn,
                                    self.fnewn,
                                    self.Enewn,
                                    self.esren,
                                    self.fren,
                                    self.Eren,
                                )
                                self.transition_coefficients(
                                    self.esi, self.fi, self.Ei, self.esf, self.ff, self.Ef
                                )
                                self.transition_response(
                                    self.trial.eb,
                                    self.esi,
                                    self.fi,
                                    self.esf,
                                    self.ff,
                                    self.Ei,
                                    self.Ef,
                                    self.A,
                                    self.R,
                                )
                                self.fcb = self.fc
                                self.Etb = self.Et
                                self.esi = self.trial.er
                                self.fi = self.trial.fr
                                self.Ei = self.params.Ec
                                self.esf = self.trial.eb
                                self.transition_coefficients(
                                    self.esi, self.fi, self.Ei, self.esf, self.fcb, self.Etb
                                )
                                self.rule_11(
                                    self.trial.er,
                                    self.trial.fr,
                                    self.trial.eb,
                                    self.fcb,
                                    self.Etb,
                                    self.A,
                                    self.R,
                                )
                                self.trial.rule = 11.0
                                self.apply_transition()
                            # Rule 7
                            elif self.trial.strain >= self.esren:
                                self.rule_7(
                                    self.trial.eunn,
                                    self.fnewn,
                                    self.Enewn,
                                    self.esren,
                                    self.fren,
                                    self.Eren,
                                )
                                self.trial.rule = 7.0
                                self.apply_new_transition()
                            else:
                                self.compression_envelope(self.trial.strain)
                        else:
                            # Rule 11 targeting for eb on 1 or 5
                            if self.trial.strain >= self.trial.eb:
                                self.compression_envelope(self.trial.eb)
                                self.fcb = self.trial.stress
                                self.Etb = self.trial.tangent
                                self.esi = self.trial.er
                                self.fi = self.trial.fr
                                self.Ei = self.params.Ec
                                self.esf = self.trial.eb
                                self.transition_coefficients(
                                    self.esi, self.fi, self.Ei, self.esf, self.fcb, self.Etb
                                )
                                self.rule_11(
                                    self.trial.er,
                                    self.trial.fr,
                                    self.trial.eb,
                                    self.fcb,
                                    self.Eta,
                                    self.A,
                                    self.R,
                                )
                                self.trial.rule = 11.0
                                self.apply_transition()
                            else:
                                self.compression_envelope(self.trial.strain)
                    elif self.committed.rule == 12.0:
                        self.trial.er = self.committed.strain
                        self.trial.fr = self.committed.stress
                        if self.trial.ea != self.trial.er0n:
                            if self.trial.eb >= self.esplp:
                                # Rule 11 targeting for eb on 4
                                if self.trial.strain >= self.trial.eb:
                                    self.rule_4(
                                        self.trial.eunp, self.trial.funp, self.esplp, self.Eplp
                                    )
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.ff, self.Ef
                                    )
                                    self.transition_response(
                                        self.trial.eb,
                                        self.esi,
                                        self.fi,
                                        self.esf,
                                        self.ff,
                                        self.Ei,
                                        self.Ef,
                                        self.A,
                                        self.R,
                                    )
                                    self.fcb = self.fc
                                    self.Etb = self.Et
                                    self.esi = self.trial.er
                                    self.fi = self.trial.fr
                                    self.Ei = self.params.Ec
                                    self.esf = self.trial.eb
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.fcb, self.Etb
                                    )
                                    self.rule_11(
                                        self.trial.er,
                                        self.trial.fr,
                                        self.trial.eb,
                                        self.fcb,
                                        self.Etb,
                                        self.A,
                                        self.R,
                                    )
                                    self.trial.rule = 11.0
                                    self.apply_transition()
                                # Rule 4
                                elif self.trial.strain >= self.esplp:
                                    self.rule_4(
                                        self.trial.eunp, self.trial.funp, self.esplp, self.Eplp
                                    )
                                    self.trial.rule = 4.0
                                    self.apply_new_transition()
                                # Rule 10
                                elif self.trial.strain >= self.trial.eunn:
                                    self.rule_10(
                                        self.esplp,
                                        self.Eplp,
                                        self.trial.eunn,
                                        self.fnewn,
                                        self.Enewn,
                                    )
                                    self.trial.rule = 10.0
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.ff, self.Ef
                                    )
                                    self.transition_response(
                                        self.trial.strain,
                                        self.esi,
                                        self.fi,
                                        self.esf,
                                        self.ff,
                                        self.Ei,
                                        self.Ef,
                                        self.A,
                                        self.R,
                                    )
                                    self.Esectest = (self.ff - self.fi) / (self.esf - self.esi)
                                    if self.Et == self.Esectest:
                                        if self.trial.strain >= self.espln:
                                            self.fc = 0
                                            self.Et = 0
                                            self.trial.rule = 10.0
                                        else:
                                            self.Et = self.Enewn
                                            self.fc = self.Et * (self.trial.strain - self.espln)
                                            self.trial.rule = 10.0
                                    self.trial.stress = self.fc
                                    self.trial.tangent = self.Et
                                # Rule 7
                                elif self.trial.strain >= self.esren:
                                    self.rule_7(
                                        self.trial.eunn,
                                        self.fnewn,
                                        self.Enewn,
                                        self.esren,
                                        self.fren,
                                        self.Eren,
                                    )
                                    self.trial.rule = 7.0
                                    self.apply_new_transition()
                                else:
                                    self.compression_envelope(self.trial.strain)
                            elif self.trial.eb >= self.trial.eunn:
                                # Rule 11 targeting for eb on 10
                                if self.trial.strain >= self.trial.eb:
                                    self.rule_10(
                                        self.esplp,
                                        self.Eplp,
                                        self.trial.eunn,
                                        self.fnewn,
                                        self.Enewn,
                                    )
                                    self.esi10 = self.esi
                                    self.fi10 = self.fi
                                    self.Ei10 = self.Ei
                                    self.esf10 = self.esf
                                    self.ff10 = self.ff
                                    self.Ef10 = self.Ef
                                    self.transition_coefficients(
                                        self.esi10,
                                        self.fi10,
                                        self.Ei10,
                                        self.esf10,
                                        self.ff10,
                                        self.Ef10,
                                    )
                                    self.R10 = self.R
                                    self.A10 = self.A
                                    self.transition_response(
                                        self.trial.eb,
                                        self.esi10,
                                        self.fi10,
                                        self.esf10,
                                        self.ff10,
                                        self.Ei10,
                                        self.Ef10,
                                        self.A10,
                                        self.R10,
                                    )
                                    self.fcb = self.fc
                                    self.Etb = self.Et
                                    self.Esectest10 = (self.ff10 - self.fi10) / (
                                        self.esf10 - self.esi10
                                    )
                                    if self.Etb == self.Esectest10:
                                        if self.trial.eb >= self.espln:
                                            self.fcb = 0
                                            self.Etb = 0
                                        else:
                                            self.Etb = self.Enewn
                                            self.fcb = self.Etb * (self.trial.eb - self.espln)
                                    self.esi = self.trial.er
                                    self.fi = self.trial.fr
                                    self.Ei = self.params.Ec
                                    self.esf = self.trial.eb
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.fcb, self.Etb
                                    )
                                    self.rule_11(
                                        self.trial.er,
                                        self.trial.fr,
                                        self.trial.eb,
                                        self.fcb,
                                        self.Etb,
                                        self.A,
                                        self.R,
                                    )
                                    self.trial.rule = 11.0
                                    self.apply_transition()
                                # Rule 10
                                elif self.trial.strain >= self.trial.eunn:
                                    self.rule_10(
                                        self.esplp,
                                        self.Eplp,
                                        self.trial.eunn,
                                        self.fnewn,
                                        self.Enewn,
                                    )
                                    self.trial.rule = 10.0
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.ff, self.Ef
                                    )
                                    self.transition_response(
                                        self.trial.strain,
                                        self.esi,
                                        self.fi,
                                        self.esf,
                                        self.ff,
                                        self.Ei,
                                        self.Ef,
                                        self.A,
                                        self.R,
                                    )
                                    self.Esectest = (self.ff - self.fi) / (self.esf - self.esi)
                                    if self.Et == self.Esectest:
                                        if self.trial.strain >= self.espln:
                                            self.fc = 0
                                            self.Et = 0
                                            self.trial.rule = 10.0
                                        else:
                                            self.Et = self.Enewn
                                            self.fc = self.Et * (self.trial.strain - self.espln)
                                            self.trial.rule = 10.0
                                    self.trial.stress = self.fc
                                    self.trial.tangent = self.Et
                                # Rule 7
                                elif self.trial.strain >= self.esren:
                                    self.rule_7(
                                        self.trial.eunn,
                                        self.fnewn,
                                        self.Enewn,
                                        self.esren,
                                        self.fren,
                                        self.Eren,
                                    )
                                    self.trial.rule = 7.0
                                    self.apply_new_transition()
                                else:
                                    self.compression_envelope(self.trial.strain)
                            elif self.trial.eb >= self.esren:
                                # Rule 11 targeting for eb on 7
                                if self.trial.strain >= self.trial.eb:
                                    self.rule_7(
                                        self.trial.eunn,
                                        self.fnewn,
                                        self.Enewn,
                                        self.esren,
                                        self.fren,
                                        self.Eren,
                                    )
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.ff, self.Ef
                                    )
                                    self.transition_response(
                                        self.trial.eb,
                                        self.esi,
                                        self.fi,
                                        self.esf,
                                        self.ff,
                                        self.Ei,
                                        self.Ef,
                                        self.A,
                                        self.R,
                                    )
                                    self.fcb = self.fc
                                    self.Etb = self.Et
                                    self.esi = self.trial.er
                                    self.fi = self.trial.fr
                                    self.Ei = self.params.Ec
                                    self.esf = self.trial.eb
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.fcb, self.Etb
                                    )
                                    self.rule_11(
                                        self.trial.er,
                                        self.trial.fr,
                                        self.trial.eb,
                                        self.fcb,
                                        self.Etb,
                                        self.A,
                                        self.R,
                                    )
                                    self.trial.rule = 11.0
                                    self.apply_transition()
                                # Rule 7
                                elif self.trial.strain >= self.esren:
                                    self.rule_7(
                                        self.trial.eunn,
                                        self.fnewn,
                                        self.Enewn,
                                        self.esren,
                                        self.fren,
                                        self.Eren,
                                    )
                                    self.trial.rule = 7.0
                                    self.apply_new_transition()
                                else:
                                    self.compression_envelope(self.trial.strain)
                            else:
                                # Rule 11 targeting for eb on 1 or 5
                                if self.trial.strain >= self.trial.eb:
                                    self.compression_envelope(self.trial.eb)
                                    self.fcb = self.trial.stress
                                    self.Etb = self.trial.tangent
                                    self.esi = self.trial.er
                                    self.fi = self.trial.fr
                                    self.Ei = self.params.Ec
                                    self.esf = self.trial.eb
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.fcb, self.Etb
                                    )
                                    self.rule_11(
                                        self.trial.er,
                                        self.trial.fr,
                                        self.trial.eb,
                                        self.fcb,
                                        self.Eta,
                                        self.A,
                                        self.R,
                                    )
                                    self.trial.rule = 11.0
                                    self.apply_transition()
                                else:
                                    self.compression_envelope(self.trial.strain)
                        else:
                            if self.trial.eb >= self.esrestn:
                                # Rule 11 targeting for eb on 77
                                if self.trial.strain >= self.trial.eb:
                                    self.rule_77(
                                        self.trial.eb,
                                        self.trial.e0,
                                        self.trial.er0n,
                                        self.trial.fr0n,
                                        self.trial.eunn,
                                        self.fnewstn,
                                        self.Enewstn,
                                        self.esrestn,
                                        self.frestn,
                                        self.Erestn,
                                    )
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.ff, self.Ef
                                    )
                                    self.transition_response(
                                        self.trial.eb,
                                        self.esi,
                                        self.fi,
                                        self.esf,
                                        self.ff,
                                        self.Ei,
                                        self.Ef,
                                        self.A,
                                        self.R,
                                    )
                                    self.fcb = self.fc
                                    self.Etb = self.Et
                                    self.esi = self.trial.er
                                    self.fi = self.trial.fr
                                    self.Ei = self.params.Ec
                                    self.esf = self.trial.eb
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.fcb, self.Etb
                                    )
                                    self.rule_11(
                                        self.trial.er,
                                        self.trial.fr,
                                        self.trial.eb,
                                        self.fcb,
                                        self.Etb,
                                        self.A,
                                        self.R,
                                    )
                                    self.trial.rule = 11.0
                                    self.apply_transition()
                                # Rule 77
                                elif self.trial.strain > self.esrestn:
                                    self.rule_77(
                                        self.trial.strain,
                                        self.trial.e0,
                                        self.trial.er0n,
                                        self.trial.fr0n,
                                        self.trial.eunn,
                                        self.fnewstn,
                                        self.Enewstn,
                                        self.esrestn,
                                        self.frestn,
                                        self.Erestn,
                                    )
                                    self.trial.rule = 77.0
                                    self.apply_new_transition()
                                else:
                                    self.compression_envelope(self.trial.strain)
                            else:
                                # Rule 11 targeting for eb on 1 or 5
                                if self.trial.strain >= self.trial.eb:
                                    self.compression_envelope(self.trial.eb)
                                    self.fcb = self.trial.stress
                                    self.Etb = self.trial.tangent
                                    self.esi = self.trial.er
                                    self.fi = self.trial.fr
                                    self.Ei = self.params.Ec
                                    self.esf = self.trial.eb
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.fcb, self.Etb
                                    )
                                    self.rule_11(
                                        self.trial.er,
                                        self.trial.fr,
                                        self.trial.eb,
                                        self.fcb,
                                        self.Etb,
                                        self.A,
                                        self.R,
                                    )
                                    self.trial.rule = 11.0
                                    self.apply_transition()
                                else:
                                    self.compression_envelope(self.trial.strain)
                    elif self.committed.rule == 14.0:
                        self.trial.er = self.committed.strain
                        self.trial.fr = self.committed.stress
                        # Rule 15 targeting for ea (ed) on 13
                        if self.trial.strain >= self.trial.ea:
                            self.rule_13(self.trial.ed, self.trial.eunn, self.fnewn, self.Enewn)
                            self.esi13 = self.esi
                            self.fi13 = self.fi
                            self.Ei13 = self.Ei
                            self.esf13 = self.esf
                            self.ff13 = self.ff
                            self.Ef13 = self.Ef
                            self.transition_coefficients(
                                self.esi13, self.fi13, self.Ei13, self.esf13, self.ff13, self.Ef13
                            )
                            self.R13 = self.R
                            self.A13 = self.A
                            self.transition_response(
                                self.trial.ea,
                                self.esi13,
                                self.fi13,
                                self.esf13,
                                self.ff13,
                                self.Ei13,
                                self.Ef13,
                                self.A13,
                                self.R13,
                            )
                            self.fca = self.fc
                            self.Eta = self.Et
                            self.Esectest13 = (self.ff13 - self.fi13) / (self.esf13 - self.esi13)
                            if self.Eta == self.Esectest13:
                                if self.trial.ea >= self.espln:
                                    self.fca = 0
                                    self.Eta = 0
                                else:
                                    self.Eta = self.Enewn
                                    self.fca = self.Et * (self.trial.ea - self.espln)
                            self.esi = self.trial.er
                            self.fi = self.trial.fr
                            self.Ei = self.params.Ec
                            self.esf = self.trial.ea
                            self.transition_coefficients(
                                self.esi, self.fi, self.Ei, self.esf, self.fca, self.Eta
                            )
                            self.rule_15(
                                self.trial.er,
                                self.trial.fr,
                                self.trial.ea,
                                self.fca,
                                self.Eta,
                                self.A,
                                self.R,
                            )
                            self.trial.rule = 15.0
                            self.apply_transition()
                        # Rule 13
                        elif self.trial.strain >= self.trial.eunn:
                            self.rule_13(self.trial.ed, self.trial.eunn, self.fnewn, self.Enewn)
                            self.trial.rule = 13.0
                            self.transition_coefficients(
                                self.esi, self.fi, self.Ei, self.esf, self.ff, self.Ef
                            )
                            self.transition_response(
                                self.trial.strain,
                                self.esi,
                                self.fi,
                                self.esf,
                                self.ff,
                                self.Ei,
                                self.Ef,
                                self.A,
                                self.R,
                            )
                            self.Esectest = (self.ff - self.fi) / (self.esf - self.esi)
                            if self.Et == self.Esectest:
                                if self.trial.strain >= self.espln:
                                    self.fc = 0
                                    self.Et = 0
                                    self.trial.rule = 13.0
                                else:
                                    self.Et = self.Enewn
                                    self.fc = self.Et * (self.trial.strain - self.espln)
                                    self.trial.rule = 13.0
                            self.trial.stress = self.fc
                            self.trial.tangent = self.Et
                        # Rule 7
                        elif self.trial.strain >= self.esren:
                            self.rule_7(
                                self.trial.eunn,
                                self.fnewn,
                                self.Enewn,
                                self.esren,
                                self.fren,
                                self.Eren,
                            )
                            self.trial.rule = 7.0
                            self.apply_new_transition()
                        else:
                            self.compression_envelope(self.trial.strain)
                    elif self.committed.rule == 3.0:
                        self.trial.er0n = self.committed.strain
                        self.trial.fr0n = self.committed.stress
                        self.trial.ea = self.trial.er0n
                        self.fnewstnf(
                            self.trial.funn,
                            self.delfn,
                            self.trial.eunn,
                            self.trial.er0n,
                            self.espln,
                        )
                        self.Enewstnf(
                            self.fnewstn, self.trial.fr0n, self.trial.eunn, self.trial.er0n
                        )
                        self.esrestnf(self.trial.eunn, self.delen, self.trial.er0n, self.espln)
                        self.freErestnf(self.trial.eunn, self.trial.funn, self.trial.er0n)
                        # Rule 77
                        if self.trial.strain > self.esrestn:
                            self.rule_77(
                                self.trial.strain,
                                self.trial.e0,
                                self.trial.er0n,
                                self.trial.fr0n,
                                self.trial.eunn,
                                self.fnewstn,
                                self.Enewstn,
                                self.esrestn,
                                self.frestn,
                                self.Erestn,
                            )
                            self.trial.rule = 77.0
                            self.apply_new_transition()
                        else:
                            self.compression_envelope(self.trial.strain)
                    elif self.committed.rule == 88.0:
                        if self.committed.strain <= self.trial.eunp:
                            self.trial.er = self.committed.strain
                            self.trial.fr = self.committed.stress
                            self.trial.ea = self.trial.er
                            self.trial.eb = self.trial.er0p
                            if self.trial.eb >= self.esplp:
                                # Rule 11 targeting for eb on 4
                                if self.trial.strain >= self.trial.eb:
                                    self.rule_4(
                                        self.trial.eunp, self.trial.funp, self.esplp, self.Eplp
                                    )
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.ff, self.Ef
                                    )
                                    self.transition_response(
                                        self.trial.eb,
                                        self.esi,
                                        self.fi,
                                        self.esf,
                                        self.ff,
                                        self.Ei,
                                        self.Ef,
                                        self.A,
                                        self.R,
                                    )
                                    self.fcb = self.fc
                                    self.Etb = self.Et
                                    self.esi = self.trial.er
                                    self.fi = self.trial.fr
                                    self.Ei = self.params.Ec
                                    self.esf = self.trial.eb
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.fcb, self.Etb
                                    )
                                    self.rule_11(
                                        self.trial.er,
                                        self.trial.fr,
                                        self.trial.eb,
                                        self.fcb,
                                        self.Etb,
                                        self.A,
                                        self.R,
                                    )
                                    self.trial.rule = 11.0
                                    self.apply_transition()
                                # Rule 4
                                elif self.trial.strain >= self.esplp:
                                    self.rule_4(
                                        self.trial.eunp, self.trial.funp, self.esplp, self.Eplp
                                    )
                                    self.trial.rule = 4.0
                                    self.apply_new_transition()
                                # Rule 10
                                elif self.trial.strain >= self.trial.eunn:
                                    self.rule_10(
                                        self.esplp,
                                        self.Eplp,
                                        self.trial.eunn,
                                        self.fnewn,
                                        self.Enewn,
                                    )
                                    self.trial.rule = 10.0
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.ff, self.Ef
                                    )
                                    self.transition_response(
                                        self.trial.strain,
                                        self.esi,
                                        self.fi,
                                        self.esf,
                                        self.ff,
                                        self.Ei,
                                        self.Ef,
                                        self.A,
                                        self.R,
                                    )
                                    self.Esectest = (self.ff - self.fi) / (self.esf - self.esi)
                                    if self.Et == self.Esectest:
                                        if self.trial.strain >= self.espln:
                                            self.fc = 0
                                            self.Et = 0
                                            self.trial.rule = 10.0
                                        else:
                                            self.Et = self.Enewn
                                            self.fc = self.Et * (self.trial.strain - self.espln)
                                            self.trial.rule = 10.0
                                    self.trial.stress = self.fc
                                    self.trial.tangent = self.Et
                                # Rule 7
                                elif self.trial.strain >= self.esren:
                                    self.rule_7(
                                        self.trial.eunn,
                                        self.fnewn,
                                        self.Enewn,
                                        self.esren,
                                        self.fren,
                                        self.Eren,
                                    )
                                    self.trial.rule = 7.0
                                    self.apply_new_transition()
                                else:
                                    self.compression_envelope(self.trial.strain)
                            elif self.trial.eb >= self.trial.eunn:
                                # Rule 11 targeting for eb on 10
                                if self.trial.strain >= self.trial.eb:
                                    self.rule_10(
                                        self.esplp,
                                        self.Eplp,
                                        self.trial.eunn,
                                        self.fnewn,
                                        self.Enewn,
                                    )
                                    self.esi10 = self.esi
                                    self.fi10 = self.fi
                                    self.Ei10 = self.Ei
                                    self.esf10 = self.esf
                                    self.ff10 = self.ff
                                    self.Ef10 = self.Ef
                                    self.transition_coefficients(
                                        self.esi10,
                                        self.fi10,
                                        self.Ei10,
                                        self.esf10,
                                        self.ff10,
                                        self.Ef10,
                                    )
                                    self.R10 = self.R
                                    self.A10 = self.A
                                    self.transition_response(
                                        self.trial.eb,
                                        self.esi10,
                                        self.fi10,
                                        self.esf10,
                                        self.ff10,
                                        self.Ei10,
                                        self.Ef10,
                                        self.A10,
                                        self.R10,
                                    )
                                    self.fcb = self.fc
                                    self.Etb = self.Et
                                    self.Esectest10 = (self.ff10 - self.fi10) / (
                                        self.esf10 - self.esi10
                                    )
                                    if self.Etb == self.Esectest10:
                                        if self.trial.eb >= self.espln:
                                            self.fcb = 0
                                            self.Etb = 0
                                        else:
                                            self.Etb = self.Enewn
                                            self.fcb = self.Etb * (self.trial.eb - self.espln)
                                    self.esi = self.trial.er
                                    self.fi = self.trial.fr
                                    self.Ei = self.params.Ec
                                    self.esf = self.trial.eb
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.fcb, self.Etb
                                    )
                                    self.rule_11(
                                        self.trial.er,
                                        self.trial.fr,
                                        self.trial.ea,
                                        self.fcb,
                                        self.Etb,
                                        self.A,
                                        self.R,
                                    )
                                    self.trial.rule = 11.0
                                    self.apply_transition()
                                # Rule 10
                                elif self.trial.strain >= self.trial.eunn:
                                    self.rule_10(
                                        self.esplp,
                                        self.Eplp,
                                        self.trial.eunn,
                                        self.fnewn,
                                        self.Enewn,
                                    )
                                    self.trial.rule = 10.0
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.ff, self.Ef
                                    )
                                    self.transition_response(
                                        self.trial.strain,
                                        self.esi,
                                        self.fi,
                                        self.esf,
                                        self.ff,
                                        self.Ei,
                                        self.Ef,
                                        self.A,
                                        self.R,
                                    )
                                    self.Esectest = (self.ff - self.fi) / (self.esf - self.esi)
                                    if self.Et == self.Esectest:
                                        if self.trial.strain >= self.espln:
                                            self.fc = 0
                                            self.Et = 0
                                            self.trial.rule = 10.0
                                        else:
                                            self.Et = self.Enewn
                                            self.fc = self.Et * (self.trial.strain - self.espln)
                                            self.trial.rule = 10.0
                                    self.trial.stress = self.fc
                                    self.trial.tangent = self.Et
                                # Rule 7
                                elif self.trial.strain >= self.esren:
                                    self.rule_7(
                                        self.trial.eunn,
                                        self.fnewn,
                                        self.Enewn,
                                        self.esren,
                                        self.fren,
                                        self.Eren,
                                    )
                                    self.trial.rule = 7.0
                                    self.apply_new_transition()
                                else:
                                    self.compression_envelope(self.trial.strain)
                            elif self.trial.eb >= self.esren:
                                # Rule 11 targeting for eb on 7
                                if self.trial.strain >= self.trial.eb:
                                    self.rule_7(
                                        self.trial.eunn,
                                        self.fnewn,
                                        self.Enewn,
                                        self.esren,
                                        self.fren,
                                        self.Eren,
                                    )
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.ff, self.Ef
                                    )
                                    self.transition_response(
                                        self.trial.eb,
                                        self.esi,
                                        self.fi,
                                        self.esf,
                                        self.ff,
                                        self.Ei,
                                        self.Ef,
                                        self.A,
                                        self.R,
                                    )
                                    self.fcb = self.fc
                                    self.Etb = self.Et
                                    self.esi = self.trial.er
                                    self.fi = self.trial.fr
                                    self.Ei = self.params.Ec
                                    self.esf = self.trial.eb
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.fcb, self.Etb
                                    )
                                    self.rule_11(
                                        self.trial.er,
                                        self.trial.fr,
                                        self.trial.eb,
                                        self.fcb,
                                        self.Etb,
                                        self.A,
                                        self.R,
                                    )
                                    self.trial.rule = 11.0
                                    self.apply_transition()
                                # Rule 7
                                elif self.trial.strain >= self.esren:
                                    self.rule_7(
                                        self.trial.eunn,
                                        self.fnewn,
                                        self.Enewn,
                                        self.esren,
                                        self.fren,
                                        self.Eren,
                                    )
                                    self.trial.rule = 7.0
                                    self.apply_new_transition()
                                else:
                                    self.compression_envelope(self.trial.strain)
                            else:
                                # Rule 11 targeting for eb on 1 or 5
                                if self.trial.strain >= self.trial.eb:
                                    self.compression_envelope(self.trial.eb)
                                    self.fcb = self.trial.stress
                                    self.Etb = self.trial.tangent
                                    self.esi = self.trial.er
                                    self.fi = self.trial.fr
                                    self.Ei = self.params.Ec
                                    self.esf = self.trial.eb
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.fcb, self.Etb
                                    )
                                    self.rule_11(
                                        self.trial.er,
                                        self.trial.fr,
                                        self.trial.eb,
                                        self.fcb,
                                        self.Etb,
                                        self.A,
                                        self.R,
                                    )
                                    self.trial.rule = 11.0
                                    self.apply_transition()
                                else:
                                    self.compression_envelope(self.trial.strain)
                        else:
                            self.trial.eunp = self.committed.strain
                            self.tension_envelope(self.trial.eunp, self.trial.e0)
                            self.trial.funp = self.trial.stress
                            self.esplpf(self.trial.eunp, self.trial.funp, self.trial.e0, self.espln)
                            self.Eplpf(self.trial.e0, self.trial.eunp)
                            if self.trial.strain >= self.esplp:
                                self.rule_4(self.trial.eunp, self.trial.funp, self.esplp, self.Eplp)
                                self.trial.rule = 4.0
                                self.apply_new_transition()
                            elif self.trial.strain >= self.trial.eunn:
                                self.rule_10(
                                    self.esplp, self.Eplp, self.trial.eunn, self.fnewn, self.Enewn
                                )
                                self.trial.rule = 10.0
                                self.transition_coefficients(
                                    self.esi, self.fi, self.Ei, self.esf, self.ff, self.Ef
                                )
                                self.transition_response(
                                    self.trial.strain,
                                    self.esi,
                                    self.fi,
                                    self.esf,
                                    self.ff,
                                    self.Ei,
                                    self.Ef,
                                    self.A,
                                    self.R,
                                )
                                self.Esectest = (self.ff - self.fi) / (self.esf - self.esi)
                                if self.Et == self.Esectest:
                                    if self.trial.strain >= self.espln:
                                        self.fc = 0
                                        self.Et = 0
                                        self.trial.rule = 10.0
                                    else:
                                        self.Et = self.Enewn
                                        self.fc = self.Et * (self.trial.strain - self.espln)
                                        self.trial.rule = 10.0
                                self.trial.stress = self.fc
                                self.trial.tangent = self.Et
                            elif self.trial.strain >= self.esren:
                                self.rule_7(
                                    self.trial.eunn,
                                    self.fnewn,
                                    self.Enewn,
                                    self.esren,
                                    self.fren,
                                    self.Eren,
                                )
                                self.trial.rule = 7.0
                                self.apply_new_transition()
                            else:
                                self.compression_envelope(self.trial.strain)
                else:
                    # Continue on transition 3 or 9 or 8 [Rules 3,9,8,2,6]
                    if ((self.committed.rule == 3.0) or (self.committed.rule == 9.0)) or (
                        self.committed.rule == 8.0
                    ):
                        if self.trial.strain <= self.espln:
                            self.rule_3(self.trial.eunn, self.trial.funn, self.espln, self.Epln)
                            self.trial.rule = 3.0
                            self.apply_new_transition()
                        elif self.trial.strain <= self.trial.eunp:
                            self.rule_9(
                                self.espln, self.Epln, self.trial.eunp, self.fnewp, self.Enewp
                            )
                            self.trial.rule = 9.0
                            self.apply_new_transition()
                        elif self.trial.strain <= self.esrep:
                            self.rule_8(
                                self.trial.eunp,
                                self.fnewp,
                                self.Enewp,
                                self.esrep,
                                self.frep,
                                self.Erep,
                            )
                            self.trial.rule = 8.0
                            self.apply_new_transition()
                        else:
                            self.tension_envelope(self.trial.strain, self.trial.e0)
                    # Continue on transition 2 [Rules 2,6]
                    elif self.committed.rule == 2.0:
                        self.tension_envelope(self.trial.strain, self.trial.e0)
                    # Continue on transition 6 [Rule 6]
                    elif self.committed.rule == 6.0:
                        self.cracked_tension_envelope(self.trial.strain, self.trial.e0)
                        self.trial.rule = 6.0
                    # Continue on transition 88 [Rules 88,2,6]
                    elif self.committed.rule == 88.0:
                        # Rule 88
                        if self.trial.strain < self.esrestp:
                            self.rule_88(
                                self.trial.strain,
                                self.trial.e0,
                                self.trial.er0p,
                                self.trial.fr0p,
                                self.trial.eunp,
                                self.fnewstp,
                                self.Enewstp,
                                self.esrestp,
                                self.frestp,
                                self.Erestp,
                            )
                            self.trial.rule = 88.0
                            self.apply_new_transition()
                        else:
                            self.tension_envelope(self.trial.strain, self.trial.e0)
                    # Continue on transition 12 [Rules 12,(3,9 or 9),8,2,6] or [Rules 12,88,2,6]
                    elif self.committed.rule == 12.0:
                        if self.trial.eb != self.trial.er0p:
                            if self.trial.ea <= self.espln:
                                # Rule 12 targeting for ea on 3
                                if self.trial.strain <= self.trial.ea:
                                    self.rule_3(
                                        self.trial.eunn, self.trial.funn, self.espln, self.Epln
                                    )
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.ff, self.Ef
                                    )
                                    self.transition_response(
                                        self.trial.ea,
                                        self.esi,
                                        self.fi,
                                        self.esf,
                                        self.ff,
                                        self.Ei,
                                        self.Ef,
                                        self.A,
                                        self.R,
                                    )
                                    self.fca = self.fc
                                    self.Eta = self.Et
                                    self.esi = self.trial.er
                                    self.fi = self.trial.fr
                                    self.Ei = self.params.Ec
                                    self.esf = self.trial.ea
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.fca, self.Eta
                                    )
                                    self.rule_12(
                                        self.trial.er,
                                        self.trial.fr,
                                        self.trial.ea,
                                        self.fca,
                                        self.Eta,
                                        self.A,
                                        self.R,
                                    )
                                    self.trial.rule = 12.0
                                    self.apply_transition()
                                # Rule 3
                                elif self.trial.strain <= self.espln:
                                    self.rule_3(
                                        self.trial.eunn, self.trial.funn, self.espln, self.Epln
                                    )
                                    self.trial.rule = 3.0
                                    self.apply_new_transition()
                                # Rule 9
                                elif self.trial.strain <= self.trial.eunp:
                                    self.rule_9(
                                        self.espln,
                                        self.Epln,
                                        self.trial.eunp,
                                        self.fnewp,
                                        self.Enewp,
                                    )
                                    self.trial.rule = 9.0
                                    self.apply_new_transition()
                                # Rule 8
                                elif self.trial.strain <= self.esrep:
                                    self.rule_8(
                                        self.trial.eunp,
                                        self.fnewp,
                                        self.Enewp,
                                        self.esrep,
                                        self.frep,
                                        self.Erep,
                                    )
                                    self.trial.rule = 8.0
                                    self.apply_new_transition()
                                else:
                                    self.tension_envelope(self.trial.strain, self.trial.e0)
                            elif self.trial.ea <= self.trial.eunp:
                                # Rule 12 targeting for ea on 9
                                if self.trial.strain <= self.trial.ea:
                                    self.rule_9(
                                        self.espln,
                                        self.Epln,
                                        self.trial.eunp,
                                        self.fnewp,
                                        self.Enewp,
                                    )
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.ff, self.Ef
                                    )
                                    self.transition_response(
                                        self.trial.ea,
                                        self.esi,
                                        self.fi,
                                        self.esf,
                                        self.ff,
                                        self.Ei,
                                        self.Ef,
                                        self.A,
                                        self.R,
                                    )
                                    self.fca = self.fc
                                    self.Eta = self.Et
                                    self.esi = self.trial.er
                                    self.fi = self.trial.fr
                                    self.Ei = self.params.Ec
                                    self.esf = self.trial.ea
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.fca, self.Eta
                                    )
                                    self.rule_12(
                                        self.trial.er,
                                        self.trial.fr,
                                        self.trial.ea,
                                        self.fca,
                                        self.Eta,
                                        self.A,
                                        self.R,
                                    )
                                    self.trial.rule = 12.0
                                    self.apply_transition()
                                # Rule 9
                                elif self.trial.strain <= self.trial.eunp:
                                    self.rule_9(
                                        self.espln,
                                        self.Epln,
                                        self.trial.eunp,
                                        self.fnewp,
                                        self.Enewp,
                                    )
                                    self.trial.rule = 9.0
                                    self.apply_new_transition()
                                # Rule 8
                                elif self.trial.strain <= self.esrep:
                                    self.rule_8(
                                        self.trial.eunp,
                                        self.fnewp,
                                        self.Enewp,
                                        self.esrep,
                                        self.frep,
                                        self.Erep,
                                    )
                                    self.trial.rule = 8.0
                                    self.apply_new_transition()
                                else:
                                    self.tension_envelope(self.trial.strain, self.trial.e0)
                            elif self.trial.ea <= self.esrep:
                                # Rule 12 targeting for ea on 8
                                if self.trial.strain <= self.trial.ea:
                                    self.rule_8(
                                        self.trial.eunp,
                                        self.fnewp,
                                        self.Enewp,
                                        self.esrep,
                                        self.frep,
                                        self.Erep,
                                    )
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.ff, self.Ef
                                    )
                                    self.transition_response(
                                        self.trial.ea,
                                        self.esi,
                                        self.fi,
                                        self.esf,
                                        self.ff,
                                        self.Ei,
                                        self.Ef,
                                        self.A,
                                        self.R,
                                    )
                                    self.fca = self.fc
                                    self.Eta = self.Et
                                    self.esi = self.trial.er
                                    self.fi = self.trial.fr
                                    self.Ei = self.params.Ec
                                    self.esf = self.trial.ea
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.fca, self.Eta
                                    )
                                    self.rule_12(
                                        self.trial.er,
                                        self.trial.fr,
                                        self.trial.ea,
                                        self.fca,
                                        self.Eta,
                                        self.A,
                                        self.R,
                                    )
                                    self.trial.rule = 12.0
                                    self.apply_transition()
                                # Rule 8
                                elif self.trial.strain <= self.esrep:
                                    self.rule_8(
                                        self.trial.eunp,
                                        self.fnewp,
                                        self.Enewp,
                                        self.esrep,
                                        self.frep,
                                        self.Erep,
                                    )
                                    self.trial.rule = 8.0
                                    self.apply_new_transition()
                                else:
                                    self.tension_envelope(self.trial.strain, self.trial.e0)
                            else:
                                # Rule 12 targeting for ea on 2 or 6
                                if self.trial.strain <= self.trial.ea:
                                    self.tension_envelope(self.trial.ea, self.trial.e0)
                                    self.fca = self.trial.stress
                                    self.Eta = self.trial.tangent
                                    self.esi = self.trial.er
                                    self.fi = self.trial.fr
                                    self.Ei = self.params.Ec
                                    self.esf = self.trial.ea
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.fca, self.Eta
                                    )
                                    self.rule_12(
                                        self.trial.er,
                                        self.trial.fr,
                                        self.trial.ea,
                                        self.fca,
                                        self.Eta,
                                        self.A,
                                        self.R,
                                    )
                                    self.trial.rule = 12.0
                                    self.apply_transition()
                                else:
                                    self.tension_envelope(self.trial.strain, self.trial.e0)
                        else:
                            if self.trial.ea <= self.esrestp:
                                # Rule 12 targeting for ea on 88
                                if self.trial.strain <= self.trial.ea:
                                    self.rule_88(
                                        self.trial.ea,
                                        self.trial.e0,
                                        self.trial.er0p,
                                        self.trial.fr0p,
                                        self.trial.eunp,
                                        self.fnewstp,
                                        self.Enewstp,
                                        self.esrestp,
                                        self.frestp,
                                        self.Erestp,
                                    )
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.ff, self.Ef
                                    )
                                    self.transition_response(
                                        self.trial.ea,
                                        self.esi,
                                        self.fi,
                                        self.esf,
                                        self.ff,
                                        self.Ei,
                                        self.Ef,
                                        self.A,
                                        self.R,
                                    )
                                    self.esi = self.trial.er
                                    self.fi = self.trial.fr
                                    self.Ei = self.params.Ec
                                    self.esf = self.trial.ea
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.fca, self.Eta
                                    )
                                    self.rule_12(
                                        self.trial.er,
                                        self.trial.fr,
                                        self.trial.ea,
                                        self.fca,
                                        self.Eta,
                                        self.A,
                                        self.R,
                                    )
                                    self.trial.rule = 12.0
                                    self.apply_transition()
                                # Rule 88
                                elif self.trial.strain <= self.esrestp:
                                    self.rule_88(
                                        self.trial.strain,
                                        self.trial.e0,
                                        self.trial.er0p,
                                        self.trial.fr0p,
                                        self.trial.eunp,
                                        self.fnewstp,
                                        self.Enewstp,
                                        self.esrestp,
                                        self.frestp,
                                        self.Erestp,
                                    )
                                    self.trial.rule = 88.0
                                    self.apply_new_transition()
                                else:
                                    self.tension_envelope(self.trial.strain, self.trial.e0)
                            else:
                                # Rule 12 targeting for ea on 2 or 6
                                if self.trial.strain <= self.trial.ea:
                                    self.tension_envelope(self.trial.ea, self.trial.e0)
                                    self.fca = self.trial.stress
                                    self.Eta = self.trial.tangent
                                    self.esi = self.trial.er
                                    self.fi = self.trial.fr
                                    self.Ei = self.params.Ec
                                    self.esf = self.trial.ea
                                    self.transition_coefficients(
                                        self.esi, self.fi, self.Ei, self.esf, self.fca, self.Eta
                                    )
                                    self.rule_12(
                                        self.trial.er,
                                        self.trial.fr,
                                        self.trial.ea,
                                        self.fca,
                                        self.Eta,
                                        self.A,
                                        self.R,
                                    )
                                    self.trial.rule = 12.0
                                    self.apply_transition()
                                else:
                                    self.tension_envelope(self.trial.strain, self.trial.e0)
                    # Continue on transition 14 [Rules 14,66,6]
                    elif self.committed.rule == 14.0:
                        # Rule 14
                        if self.trial.strain <= self.trial.eb:
                            self.rule_14(self.trial.er, self.trial.fr, self.trial.eb)
                            self.trial.rule = 14.0
                            self.apply_new_transition()
                        # Rule 66
                        elif self.trial.strain < self.trial.eunp:
                            self.rule_66(self.trial.strain, self.trial.e0)
                            self.trial.rule = 66.0
                        else:
                            self.cracked_tension_envelope(self.trial.strain, self.trial.e0)
                            self.trial.rule = 6.0
                    # Continue on transition 66 [Rules 66,6]
                    elif self.committed.rule == 66:
                        if self.trial.strain < self.trial.eunp:
                            self.rule_66(self.trial.strain, self.trial.e0)
                            self.trial.rule = 66.0
                        else:
                            self.cracked_tension_envelope(self.trial.strain, self.trial.e0)
                            self.trial.rule = 6.0
        return

    fn compression_envelope(mut self, e: Float64):
        self.xn = abs(e / self.params.epcc)
        self.nn = abs((self.params.Ec * self.params.epcc) / self.params.fpcc)
        self.envelope_stress_factor(self.params.xcrn, self.nn, self.params.rc)
        self.envelope_tangent_factor(self.params.xcrn, self.nn, self.params.rc)
        self.xsp = abs(self.params.xcrn - (self.y / (self.nn * self.z)))
        if self.xn <= self.xsp:
            self.rule_1(self.xn, self.nn, self.params.rc)
            self.trial.rule = 1.0
        else:
            self.rule_5(self.xn, self.nn, self.params.rc)
            self.trial.rule = 5.0

    fn tension_envelope(mut self, e: Float64, e0: Float64):
        self.xp = abs((e - e0) / self.params.et)
        self.np = (self.params.Ec * self.params.et) / self.params.ft
        self.envelope_stress_factor(self.params.xcrp, self.np, self.params.rt)
        self.envelope_tangent_factor(self.params.xcrp, self.np, self.params.rt)
        self.xcrk = abs(self.params.xcrp - (self.y / (self.np * self.z)))
        if self.xp <= self.xcrk:
            self.rule_2(self.xp, self.np, self.params.rt)
            self.trial.rule = 2.0
        else:
            self.rule_6(self.xp, self.np, self.params.rt)
            self.trial.rule = 6.0

    fn cracked_tension_envelope(mut self, e: Float64, e0: Float64):
        self.xp = abs((e - e0) / self.params.et)
        self.np = (self.params.Ec * self.params.et) / self.params.ft
        self.rule_6(self.xp, self.np, self.params.rt)
        self.trial.rule = 6.0

    fn envelope_stress_factor(mut self, x: Float64, n: Float64, r: Float64):
        var D: Float64
        if r != 1.0:
            D = (1.0 + ((n - (r / (r - 1.0))) * x)) + (pow(x, r) / (r - 1.0))
        else:
            D = 1 + (((n - 1) + log10(x)) * x)
        self.y = (n * x) / D

    fn envelope_tangent_factor(mut self, x: Float64, n: Float64, r: Float64):
        var D: Float64
        if r != 1.0:
            D = (1.0 + ((n - (r / (r - 1.0))) * x)) + (pow(x, r) / (r - 1.0))
        else:
            D = 1.0 + (((n - 1.0) + log10(x)) * x)
        self.z = (1 - pow(x, r)) / pow(D, 2.0)

    fn esplnf(mut self, eunn: Float64, funn: Float64):
        self.Esecnf(eunn, funn)
        self.espln = eunn - (funn / self.Esecn)

    fn Eplnf(mut self, eunn: Float64):
        self.Epln = (0.1 * self.params.Ec) * exp((-2.0) * abs(eunn / self.params.epcc))

    fn Esecnf(mut self, eunn: Float64, funn: Float64):
        self.Esecn = self.params.Ec * (
            (abs(funn / (self.params.Ec * self.params.epcc)) + 0.57)
            / (abs(eunn / self.params.epcc) + 0.57)
        )

    fn delenf(mut self, eunn: Float64):
        self.delen = eunn / (1.15 + (2.75 * abs(eunn / self.params.epcc)))

    fn delfnf(mut self, eunn: Float64, funn: Float64):
        if eunn <= (self.params.epcc / 10.0):
            self.delfn = (0.09 * funn) * pow(abs(eunn / self.params.epcc), 0.5)
        else:
            self.delfn = 0.0

    fn fnewnf(mut self, eunn: Float64, funn: Float64):
        self.delfnf(eunn, funn)
        self.fnewn = funn - self.delfn

    fn Enewnf(mut self, eunn: Float64, funn: Float64):
        self.fnewnf(eunn, funn)
        self.esplnf(eunn, funn)
        self.Enewn = min(self.params.Ec, self.fnewn / (eunn - self.espln))
        if eunn == self.espln:
            self.Enewn = self.params.Ec

    fn esrenf(mut self, eunn: Float64):
        self.delenf(eunn)
        self.esren = eunn + self.delen

    fn freErenf(mut self, eunn: Float64):
        self.esrenf(eunn)
        self.xn = abs(self.esren / self.params.epcc)
        self.nn = abs((self.params.Ec * self.params.epcc) / self.params.fpcc)
        self.envelope_stress_factor(self.params.xcrn, self.nn, self.params.rc)
        self.envelope_tangent_factor(self.params.xcrn, self.nn, self.params.rc)
        self.xsp = abs(self.params.xcrn - (self.y / (self.nn * self.z)))
        if self.xn <= self.xsp:
            if self.xn < self.params.xcrn:
                self.envelope_stress_factor(self.xn, self.nn, self.params.rc)
                self.envelope_tangent_factor(self.xn, self.nn, self.params.rc)
                self.fren = self.params.fpcc * self.y
                self.Eren = self.params.Ec * self.z
            else:
                self.envelope_stress_factor(self.params.xcrn, self.nn, self.params.rc)
                self.envelope_tangent_factor(self.params.xcrn, self.nn, self.params.rc)
                self.fren = self.params.fpcc * (
                    self.y + ((self.nn * self.z) * (self.xn - self.params.xcrn))
                )
                self.Eren = self.params.Ec * self.z
        else:
            self.fren = 0.0
            self.Eren = 0.0

    fn fnewstnf(
        mut self, funn: Float64, delfn: Float64, eunn: Float64, er0n: Float64, espln: Float64
    ):
        self.fnewstn = funn - (delfn * ((eunn - er0n) / (eunn - espln)))

    fn Enewstnf(mut self, fnewstn: Float64, fr0n: Float64, eunn: Float64, er0n: Float64):
        self.Enewstn = (fnewstn - fr0n) / (eunn - er0n)

    fn esrestnf(mut self, eunn: Float64, delen: Float64, er0n: Float64, espln: Float64):
        self.esrestn = eunn + ((delen * (eunn - er0n)) / (eunn - espln))

    fn freErestnf(mut self, eunn: Float64, funn: Float64, er0n: Float64):
        self.delenf(eunn)
        self.esplnf(eunn, funn)
        self.esrestnf(eunn, self.delen, er0n, self.espln)
        self.xn = abs(self.esrestn / self.params.epcc)
        self.nn = abs((self.params.Ec * self.params.epcc) / self.params.fpcc)
        self.envelope_stress_factor(self.params.xcrn, self.nn, self.params.rc)
        self.envelope_tangent_factor(self.params.xcrn, self.nn, self.params.rc)
        self.xsp = abs(self.params.xcrn - (self.y / (self.nn * self.z)))
        if self.xn <= self.xsp:
            if self.xn < self.params.xcrn:
                self.envelope_stress_factor(self.xn, self.nn, self.params.rc)
                self.envelope_tangent_factor(self.xn, self.nn, self.params.rc)
                self.frestn = self.params.fpcc * self.y
                self.Erestn = self.params.Ec * self.z
            else:
                self.envelope_stress_factor(self.params.xcrn, self.nn, self.params.rc)
                self.envelope_tangent_factor(self.params.xcrn, self.nn, self.params.rc)
                self.frestn = self.params.fpcc * (
                    self.y + ((self.nn * self.z) * (self.xn - self.params.xcrn))
                )
                self.Erestn = self.params.Ec * self.z
        else:
            self.frestn = 0.0
            self.Erestn = 0.0

    fn esplpf(mut self, eunp: Float64, funp: Float64, e0: Float64, espln: Float64):
        self.Esecpf(e0, eunp, funp, espln)
        self.esplp = eunp - (funp / self.Esecp)

    fn Eplpf(mut self, e0: Float64, eunp: Float64):
        if Int(self.params.gap_close) == 1:
            self.Eplp = self.params.Ec / (pow(abs((eunp - e0) / self.params.et), 1.1) + 1.0)
        else:
            self.Eplp = 0.0

    fn Esecpf(mut self, e0: Float64, eunp: Float64, funp: Float64, espln: Float64):
        self.Esecp = self.params.Ec * (
            (abs(funp / (self.params.Ec * self.params.et)) + 0.67)
            / (abs((eunp - e0) / self.params.et) + 0.67)
        )
        if self.Esecp < abs(funp / abs(eunp - espln)):
            self.Esecp = abs(funp / abs(eunp - espln))

    fn delepf(mut self, eunp: Float64, e0: Float64):
        self.delep = 0.22 * abs(eunp - e0)

    fn delfpf(mut self, funp: Float64, eunp: Float64, e0: Float64):
        if eunp >= (e0 + (self.params.et / 2.0)):
            self.delfp = 0.15 * funp
        else:
            self.delfp = 0.0

    fn fnewpf(mut self, funp: Float64, eunp: Float64, e0: Float64):
        self.delfpf(funp, eunp, e0)
        self.fnewp = funp - self.delfp

    fn Enewpf(mut self, eunp: Float64, funp: Float64, e0: Float64, espln: Float64):
        self.fnewpf(funp, eunp, e0)
        self.esplpf(eunp, funp, e0, espln)
        self.Enewp = min(self.params.Ec, self.fnewp / (eunp - self.esplp))
        if eunp == self.esplp:
            self.Enewp = self.params.Ec

    fn esrepf(mut self, eunp: Float64, e0: Float64):
        self.delepf(eunp, e0)
        self.esrep = eunp + self.delep

    fn freErepf(mut self, eunp: Float64, e0: Float64):
        self.esrepf(eunp, e0)
        self.xp = abs((self.esrep - e0) / self.params.et)
        self.np = (self.params.Ec * self.params.et) / self.params.ft
        self.envelope_stress_factor(self.params.xcrp, self.np, self.params.rt)
        self.envelope_tangent_factor(self.params.xcrp, self.np, self.params.rt)
        self.xcrk = abs(self.params.xcrp - (self.y / (self.np * self.z)))
        if self.xp <= self.xcrk:
            if self.xp < self.params.xcrp:
                self.envelope_stress_factor(self.xp, self.np, self.params.rt)
                self.envelope_tangent_factor(self.xp, self.np, self.params.rt)
                self.frep = self.params.ft * self.y
                self.Erep = self.params.Ec * self.z
            else:
                self.envelope_stress_factor(self.params.xcrp, self.np, self.params.rt)
                self.envelope_tangent_factor(self.params.xcrp, self.np, self.params.rt)
                self.frep = self.params.ft * (
                    self.y + ((self.np * self.z) * (self.xp - self.params.xcrp))
                )
                self.Erep = self.params.Ec * self.z
        else:
            self.frep = 0.0
            self.Erep = 0.0

    fn fnewstpf(
        mut self,
        funp: Float64,
        delfp: Float64,
        eunp: Float64,
        er0p: Float64,
        esplp: Float64,
        e0: Float64,
    ):
        self.fnewstp = funp - (delfp * ((eunp - er0p) / (eunp - esplp)))

    fn Enewstpf(mut self, fnewstp: Float64, fr0p: Float64, eunp: Float64, er0p: Float64):
        self.Enewstp = (fnewstp - fr0p) / (eunp - er0p)

    fn esrestpf(mut self, eunp: Float64, delep: Float64, er0p: Float64, esplp: Float64):
        self.esrestp = eunp + ((delep * (eunp - er0p)) / (eunp - esplp))

    fn freErestpf(
        mut self, eunp: Float64, funp: Float64, er0p: Float64, e0: Float64, espln: Float64
    ):
        self.delepf(eunp, e0)
        self.esplpf(eunp, funp, e0, espln)
        self.esrestpf(eunp, self.delep, er0p, self.esplp)
        self.xp = abs((self.esrestp - e0) / self.params.et)
        self.np = (self.params.Ec * self.params.et) / self.params.ft
        self.envelope_stress_factor(self.params.xcrp, self.np, self.params.rt)
        self.envelope_tangent_factor(self.params.xcrp, self.np, self.params.rt)
        self.xcrk = abs(self.params.xcrp - (self.y / (self.np * self.z)))
        if self.xp <= self.xcrk:
            if self.xp < self.params.xcrp:
                self.envelope_stress_factor(self.xp, self.np, self.params.rt)
                self.envelope_tangent_factor(self.xp, self.np, self.params.rt)
                self.frestp = self.params.ft * self.y
                self.Erestp = self.params.Ec * self.z
            else:
                self.envelope_stress_factor(self.params.xcrp, self.np, self.params.rt)
                self.envelope_tangent_factor(self.params.xcrp, self.np, self.params.rt)
                self.frestp = self.params.ft * (
                    self.y + ((self.np * self.z) * (self.xp - self.params.xcrp))
                )
                self.Erestp = self.params.Ec * self.z
        else:
            self.frestp = 0.0
            self.Erestp = 0.0

    fn update_tension_origin(
        mut self, e0: Float64, eunp: Float64, funp: Float64, eunn: Float64, funn: Float64
    ):
        var xun: Float64 = abs(eunn / self.params.epcc)
        var xup: Float64 = abs((eunp - e0) / self.params.et)
        var e0ref: Float64
        var eunpref: Float64
        var funpref: Float64
        if xup < xun:
            xup = xun
            e0ref = 0.0
            eunpref = xup * self.params.et
            self.tension_envelope(eunpref, e0ref)
            funpref = self.trial.stress
        else:
            e0ref = e0
            eunpref = eunp
            funpref = funp
        self.esplnf(eunn, funn)
        self.Eplnf(eunn)
        self.Esecpf(e0ref, eunpref, funpref, self.espln)
        var dele0: Float64 = (2.0 * funpref) / (self.Esecp + self.Epln)
        self.trial.e0 = (self.espln + dele0) - (xup * self.params.et)
        self.trial.eunp = (xup * self.params.et) + self.trial.e0
        self.tension_envelope(self.trial.eunp, self.trial.e0)
        self.trial.funp = self.trial.stress

    fn rule_1(mut self, x: Float64, n: Float64, r: Float64):
        if x < self.params.xcrn:
            self.envelope_stress_factor(x, n, r)
            self.envelope_tangent_factor(x, n, r)
            self.trial.stress = self.params.fpcc * self.y
            self.trial.tangent = self.params.Ec * self.z
        else:
            self.envelope_stress_factor(self.params.xcrn, n, r)
            self.envelope_tangent_factor(self.params.xcrn, n, r)
            self.trial.stress = self.params.fpcc * (
                self.y + ((n * self.z) * (x - self.params.xcrn))
            )
            self.trial.tangent = self.params.Ec * self.z

    fn rule_5(mut self, x: Float64, n: Float64, r: Float64):
        self.trial.stress = 0.0
        self.trial.tangent = 0.0

    fn rule_2(mut self, x: Float64, n: Float64, r: Float64):
        if x < self.params.xcrp:
            self.envelope_stress_factor(x, n, r)
            self.envelope_tangent_factor(x, n, r)
            self.trial.stress = self.params.ft * self.y
            self.trial.tangent = self.params.Ec * self.z
        else:
            self.envelope_stress_factor(self.params.xcrp, n, r)
            self.envelope_tangent_factor(self.params.xcrp, n, r)
            self.trial.stress = self.params.ft * (self.y + ((n * self.z) * (x - self.params.xcrp)))
            self.trial.tangent = self.params.Ec * self.z

    fn rule_6(mut self, x: Float64, n: Float64, r: Float64):
        self.trial.stress = 0.0
        self.trial.tangent = 0.0

    fn rule_3(mut self, eunn: Float64, funn: Float64, espln: Float64, Epln: Float64):
        self.esi = eunn
        self.fi = funn
        self.Ei = self.params.Ec
        self.esf = espln
        self.ff = 0.0
        self.Ef = Epln

    fn rule_9(
        mut self, espln: Float64, Epln: Float64, eunp: Float64, fnewp: Float64, Enewp: Float64
    ):
        self.esi = espln
        self.fi = 0.0
        self.Ei = Epln
        self.esf = eunp
        self.ff = fnewp
        self.Ef = Enewp

    fn rule_8(
        mut self,
        eunp: Float64,
        fnewp: Float64,
        Enewp: Float64,
        esrep: Float64,
        frep: Float64,
        Erep: Float64,
    ):
        self.esi = eunp
        self.fi = fnewp
        self.Ei = Enewp
        self.esf = esrep
        self.ff = frep
        self.Ef = Erep

    fn rule_4(mut self, eunp: Float64, funp: Float64, esplp: Float64, Eplp: Float64):
        self.esi = eunp
        self.fi = funp
        self.Ei = self.params.Ec
        self.esf = esplp
        self.ff = 0.0
        self.Ef = Eplp

    fn rule_10(
        mut self, esplp: Float64, Eplp: Float64, eunn: Float64, fnewn: Float64, Enewn: Float64
    ):
        self.esi = esplp
        self.fi = 0.0
        self.Ei = Eplp
        self.esf = eunn
        self.ff = fnewn
        self.Ef = Enewn

    fn rule_7(
        mut self,
        eunn: Float64,
        fnewn: Float64,
        Enewn: Float64,
        esren: Float64,
        fren: Float64,
        Eren: Float64,
    ):
        self.esi = eunn
        self.fi = fnewn
        self.Ei = Enewn
        self.esf = esren
        self.ff = fren
        self.Ef = Eren

    fn rule_12(
        mut self,
        er: Float64,
        fr: Float64,
        ea: Float64,
        fca: Float64,
        Eta: Float64,
        A: Float64,
        R: Float64,
    ):
        self.esi = er
        self.fi = fr
        self.Ei = self.params.Ec
        self.esf = ea
        self.ff = fca
        self.Ef = Eta
        self.transition_response(ea, self.esi, self.fi, self.esf, self.ff, self.Ei, self.Ef, A, R)
        self.ff = self.fc
        self.Ef = self.Et

    fn rule_11(
        mut self,
        er: Float64,
        fr: Float64,
        eb: Float64,
        fcb: Float64,
        Etb: Float64,
        A: Float64,
        R: Float64,
    ):
        self.esi = er
        self.fi = fr
        self.Ei = self.params.Ec
        self.esf = eb
        self.ff = fcb
        self.Ef = Etb
        self.transition_response(eb, self.esi, self.fi, self.esf, self.ff, self.Ei, self.Ef, A, R)
        self.ff = self.fc
        self.Ef = self.Et

    fn rule_13(mut self, ed: Float64, eunn: Float64, fnewn: Float64, Enewn: Float64):
        self.esi = ed
        self.fi = 0.0
        self.Ei = 0.0
        self.esf = eunn
        self.ff = fnewn
        self.Ef = Enewn

    fn rule_14(mut self, er: Float64, fr: Float64, eb: Float64):
        self.esi = er
        self.fi = fr
        self.Ei = self.params.Ec
        self.esf = eb
        self.ff = 0.0
        self.Ef = 0.0

    fn rule_15(
        mut self,
        er: Float64,
        fr: Float64,
        ea: Float64,
        fca: Float64,
        Eta: Float64,
        A: Float64,
        R: Float64,
    ):
        self.esi = er
        self.fi = fr
        self.Ei = self.params.Ec
        self.esf = ea
        self.ff = fca
        self.Ef = Eta
        self.transition_response(ea, self.esi, self.fi, self.esf, self.ff, self.Ei, self.Ef, A, R)
        self.ff = self.fc
        self.Ef = self.Et

    fn rule_66(mut self, e: Float64, e0: Float64):
        self.trial.stress = 0.0
        self.trial.tangent = 0.0

    fn rule_88(
        mut self,
        e: Float64,
        e0: Float64,
        er0p: Float64,
        fr0p: Float64,
        eunp: Float64,
        fnewstp: Float64,
        Enewstp: Float64,
        esrestp: Float64,
        frestp: Float64,
        Erestp: Float64,
    ):
        if ((e - e0) >= (er0p - e0)) and ((e - e0) <= (eunp - e0)):
            self.esi = er0p
            self.fi = fr0p
            self.Ei = self.params.Ec
            self.esf = eunp
            self.ff = fnewstp
            self.Ef = Enewstp
        if ((e - e0) > (eunp - e0)) and ((e - e0) < (esrestp - e0)):
            self.esi = eunp
            self.fi = fnewstp
            self.Ei = Enewstp
            self.esf = esrestp
            self.ff = frestp
            self.Ef = Erestp

    fn rule_77(
        mut self,
        e: Float64,
        e0: Float64,
        er0n: Float64,
        fr0n: Float64,
        eunn: Float64,
        fnewstn: Float64,
        Enewstn: Float64,
        esrestn: Float64,
        frestn: Float64,
        Erestn: Float64,
    ):
        if (e <= er0n) and (e >= eunn):
            self.esi = er0n
            self.fi = fr0n
            self.Ei = self.params.Ec
            self.esf = eunn
            self.ff = fnewstn
            self.Ef = Enewstn
        if (e < eunn) and (e > esrestn):
            self.esi = eunn
            self.fi = fnewstn
            self.Ei = Enewstn
            self.esf = esrestn
            self.ff = frestn
            self.Ef = Erestn

    fn ea1112f(mut self, eb: Float64, espln: Float64, esplp: Float64, eunn: Float64, eunp: Float64):
        self.trial.ea = espln + (((eunn - eb) / (eunn - esplp)) * (eunp - espln))

    fn eb1112f(mut self, ea: Float64, espln: Float64, esplp: Float64, eunn: Float64, eunp: Float64):
        self.trial.eb = eunn - (((ea - espln) / (eunp - espln)) * (eunn - esplp))

    fn eb1415f(mut self, ea: Float64, fa: Float64, Esecn: Float64):
        self.trial.eb = ea - (fa / Esecn)

    fn transition_coefficients(
        mut self, esi: Float64, fi: Float64, Ei: Float64, esf: Float64, ff: Float64, Ef: Float64
    ):
        var Esec: Float64 = (ff - fi) / (esf - esi)
        self.R = (Ef - Esec) / (Esec - Ei)
        var check: Float64 = pow(abs(esf - esi), self.R)
        if (((check == 0.0) or (check > 1.797e308)) or (check < (-1.797e308))) or (Esec == Ei):
            self.A = 1.0e-300
        else:
            self.A = (Esec - Ei) / pow(abs(esf - esi), self.R)
            if (self.A > 1.797e308) or (self.A < (-1.797e308)):
                self.A = 1.0e300

    fn transition_response(
        mut self,
        es: Float64,
        esi: Float64,
        fi: Float64,
        esf: Float64,
        ff: Float64,
        Ei: Float64,
        Ef: Float64,
        A: Float64,
        R: Float64,
    ):
        var Esec: Float64 = (ff - fi) / (esf - esi)
        if (A == 1.0e300) or (A == 0.0):
            self.fc = fi + (Esec * (es - esi))
            self.Et = Esec
        elif ((pow(abs(es - esi), -R) == 0.0) or (pow(abs(es - esi), -R) > 1.797e308)) or (
            pow(abs(es - esi), -R) < (-1.797e308)
        ):
            self.fc = fi + (Esec * (es - esi))
            self.Et = Esec
        elif (Ei >= Esec) and (Ef >= Esec):
            self.fc = fi + (Esec * (es - esi))
            self.Et = Esec
        elif (Ei <= Esec) and (Ef <= Esec):
            self.fc = fi + (Esec * (es - esi))
            self.Et = Esec
        else:
            self.fc = fi + ((es - esi) * (Ei + (A * pow(abs(es - esi), R))))
            self.Et = Ei + ((A * (R + 1)) * pow(abs(es - esi), R))
            if (self.Et > 1.797e308) or (self.Et < (-1.797e308)):
                self.fc = fi + (Esec * (es - esi))
                self.Et = Esec

    fn apply_transition(mut self):
        self.transition_response(
            self.trial.strain,
            self.esi,
            self.fi,
            self.esf,
            self.ff,
            self.Ei,
            self.Ef,
            self.A,
            self.R,
        )
        self.trial.stress = self.fc
        self.trial.tangent = self.Et

    fn apply_new_transition(mut self):
        self.transition_coefficients(self.esi, self.fi, self.Ei, self.esf, self.ff, self.Ef)
        self.apply_transition()
