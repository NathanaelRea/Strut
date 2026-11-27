# Constitutive response of OpenSees ContactMaterial2D (Coulomb friction).
struct ContactMaterial2D(ImplicitlyCopyable, Movable):
    var friction: Float64
    var stiffness: Float64
    var cohesion: Float64
    var tension: Float64
    var committed_slip: Float64
    var elastic_slip: Float64
    var normal_force: Float64
    var shear_force: Float64
    var shear_tangent: Float64
    var normal_coupling: Float64

    fn __init__(
        out self, friction: Float64, stiffness: Float64, cohesion: Float64, tension: Float64
    ):
        self.friction = friction
        self.stiffness = stiffness
        self.cohesion = cohesion
        self.tension = tension
        self.committed_slip = 0.0
        self.elastic_slip = 0.0
        self.normal_force = 0.0
        self.shear_force = 0.0
        self.shear_tangent = self.stiffness
        self.normal_coupling = 0.0

    fn reset(mut self):
        self.committed_slip = 0.0
        self.elastic_slip = 0.0
        self.normal_force = 0.0
        self.shear_force = 0.0
        self.shear_tangent = self.stiffness
        self.normal_coupling = 0.0

    fn evaluate(mut self, slip: Float64, normal: Float64, friction_enabled: Bool):
        var mu = self.friction if friction_enabled else 0.0
        var cohesion = self.cohesion if friction_enabled else 0.0
        var tension = min(self.tension, cohesion / mu) if mu != 0.0 else 0.0
        self.normal_force = normal
        self.elastic_slip = self.committed_slip + slip if normal > -tension else 0.0
        var predictor = self.stiffness * self.elastic_slip
        var yield_error = abs(predictor) - mu * normal - cohesion
        self.normal_coupling = 0.0
        self.shear_tangent = self.stiffness if normal >= -tension else 0.0
        if yield_error > 0.0 and normal > -tension and abs(self.elastic_slip) > 1.0e-12:
            var direction = -1.0 if predictor < 0.0 else 1.0
            self.elastic_slip -= direction * (yield_error / self.stiffness)
            self.shear_tangent = 0.0
            self.normal_coupling = direction * mu
        self.shear_force = self.stiffness * self.elastic_slip

    fn commit(mut self):
        self.committed_slip = self.elastic_slip
