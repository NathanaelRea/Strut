from collections import InlineArray, List
from math import sqrt

from materials.contact_material2d import ContactMaterial2D

alias ContactVector = InlineArray[Float64, 10]
alias ContactDofs = InlineArray[Int, 10]
alias Point2d = SIMD[DType.float64, 2]
alias Shape = SIMD[DType.float64, 4]


fn _dot(u: Point2d, v: Point2d) -> Float64:
    return u[0] * v[0] + u[1] * v[1]


fn _perpendicular(u: Point2d) -> Point2d:
    return Point2d(-u[1], u[0])


fn _norm(u: Point2d) -> Float64:
    return sqrt(_dot(u, u))


fn _shape(x: Float64) -> Shape:
    var x2 = x * x
    var x3 = x2 * x
    return Shape(1.0 - 3.0 * x2 + 2.0 * x3, x - 2.0 * x2 + x3, 3.0 * x2 - 2.0 * x3, -x2 + x3)


fn _shape_derivative(x: Float64) -> Shape:
    var x2 = x * x
    return Shape(
        -6.0 * x + 6.0 * x2, 1.0 - 4.0 * x + 3.0 * x2, 6.0 * x - 6.0 * x2, -2.0 * x + 3.0 * x2
    )


fn _shape_curvature(x: Float64) -> Shape:
    return Shape(-6.0 + 12.0 * x, -4.0 + 6.0 * x, 6.0 - 12.0 * x, -2.0 + 6.0 * x)


# Hermite beam centerline, with projection and contact frame advanced at commit.
struct BeamContact2D(ImplicitlyCopyable, Movable):
    var element: Int
    var dofs: ContactDofs
    var initial_a: Point2d
    var initial_b: Point2d
    var initial_secondary: Point2d
    var length: Float64
    var radius: Float64
    var gap_tol: Float64
    var force_tol: Float64
    var initially_in_contact: Bool
    var material: ContactMaterial2D
    var a: Point2d
    var b: Point2d
    var secondary: Point2d
    var a_tangent: Point2d
    var b_tangent: Point2d
    var a_rotation: Float64
    var b_rotation: Float64
    var trial_a_rotation: Float64
    var trial_b_rotation: Float64
    var xi: Float64
    var normal: Point2d
    var surface_tangent: Point2d
    var surface_offset: Float64
    var shape: Shape
    var derivative: Shape
    var normal_gradient: InlineArray[Float64, 8]
    var slip_gradient: InlineArray[Float64, 8]
    var gap: Float64
    var multiplier: Float64
    var active: Bool
    var releasing: Bool
    var last_force: ContactVector
    var removed: Bool

    fn __init__(
        out self,
        element: Int,
        dofs: ContactDofs,
        a: Point2d,
        b: Point2d,
        secondary: Point2d,
        width: Float64,
        gap_tol: Float64,
        force_tol: Float64,
        initially_in_contact: Bool,
        friction: Float64,
        stiffness: Float64,
        cohesion: Float64,
        tension: Float64,
    ):
        self.element = element
        self.dofs = dofs
        self.initial_a = a
        self.initial_b = b
        self.initial_secondary = secondary
        self.a = a
        self.b = b
        self.secondary = secondary
        self.length = _norm(b - a)
        self.radius = width / 2.0
        self.gap_tol = gap_tol
        self.force_tol = force_tol
        self.initially_in_contact = initially_in_contact
        self.material = ContactMaterial2D(friction, stiffness, cohesion, tension)
        self.a_tangent = Point2d(0.0)
        self.b_tangent = Point2d(0.0)
        self.a_rotation = 0.0
        self.b_rotation = 0.0
        self.trial_a_rotation = 0.0
        self.trial_b_rotation = 0.0
        self.xi = 0.0
        self.normal = Point2d(0.0)
        self.surface_tangent = Point2d(0.0)
        self.surface_offset = 0.0
        self.shape = Shape(0.0)
        self.derivative = Shape(0.0)
        self.normal_gradient = InlineArray[Float64, 8](fill=0.0)
        self.slip_gradient = InlineArray[Float64, 8](fill=0.0)
        self.gap = 0.0
        self.multiplier = 0.0
        self.active = False
        self.releasing = False
        self.last_force = ContactVector(fill=0.0)
        self.removed = False
        self.reset()

    fn _curve(self, shape: Shape, a_tangent: Point2d, b_tangent: Point2d) -> Point2d:
        return (
            self.a * shape[0]
            + a_tangent * (self.length * shape[1])
            + self.b * shape[2]
            + b_tangent * (self.length * shape[3])
        )

    fn _update_projection(mut self):
        var xi = self.xi
        for _ in range(50):
            var offset = self.secondary - self._curve(_shape(xi), self.a_tangent, self.b_tangent)
            var tangent = self._curve(_shape_derivative(xi), self.a_tangent, self.b_tangent)
            var residual = _dot(offset, tangent)
            if abs(residual / self.length) <= self.gap_tol:
                break
            var curvature = self._curve(_shape_curvature(xi), self.a_tangent, self.b_tangent)
            xi -= residual / (_dot(offset, curvature) - _dot(tangent, tangent))
        self.xi = xi
        self.shape = _shape(xi)
        self.derivative = _shape_derivative(xi)
        var offset = self.secondary - self._curve(self.shape, self.a_tangent, self.b_tangent)
        self.normal = offset / _norm(offset)
        var tangent = self._curve(self.derivative, self.a_tangent, self.b_tangent)
        var curvature = self._curve(_shape_curvature(xi), self.a_tangent, self.b_tangent)
        var unit_tangent = tangent / _norm(tangent)
        var tangent_derivative = (curvature - _dot(unit_tangent, curvature) * unit_tangent) / _norm(
            tangent
        )
        var transverse = _perpendicular(unit_tangent)
        self.surface_offset = self.radius * _dot(self.normal, transverse)
        self.surface_tangent = (
            tangent - self.surface_offset * _dot(tangent_derivative, transverse) * unit_tangent
        )

        var a_perp = _perpendicular(self.a_tangent)
        var b_perp = _perpendicular(self.b_tangent)
        var slip_a = self.shape[0] + self.radius * self.derivative[0]
        var slip_b = self.shape[2] + self.radius * self.derivative[2]
        for i in range(2):
            self.normal_gradient[i] = -self.shape[0] * self.normal[i]
            self.normal_gradient[3 + i] = -self.shape[2] * self.normal[i]
            self.normal_gradient[6 + i] = self.normal[i]
            self.slip_gradient[i] = -self.surface_tangent[i] * slip_a
            self.slip_gradient[3 + i] = -self.surface_tangent[i] * slip_b
            self.slip_gradient[6 + i] = self.surface_tangent[i]
        self.normal_gradient[2] = -self.shape[1] * self.length * _dot(a_perp, self.normal)
        self.normal_gradient[5] = -self.shape[3] * self.length * _dot(b_perp, self.normal)
        self.slip_gradient[2] = (
            -_dot(a_perp, self.surface_tangent)
            * self.length
            * (self.shape[1] + self.radius * self.derivative[1])
        )
        self.slip_gradient[5] = (
            -_dot(b_perp, self.surface_tangent)
            * self.length
            * (self.shape[3] + self.radius * self.derivative[3])
        )

    fn reset(mut self):
        self.a = self.initial_a
        self.b = self.initial_b
        self.secondary = self.initial_secondary
        self.a_tangent = (self.b - self.a) / self.length
        self.b_tangent = self.a_tangent
        self.a_rotation = 0.0
        self.b_rotation = 0.0
        self.trial_a_rotation = 0.0
        self.trial_b_rotation = 0.0
        self.xi = _dot(self.b - self.secondary, self.b - self.a) / self.length
        self._update_projection()
        self.active = self.initially_in_contact and 0.0 < self.xi and self.xi < 1.0
        self.releasing = False
        self.material.reset()
        self.last_force = ContactVector(fill=0.0)

    fn evaluate(
        mut self,
        u: List[Float64],
        friction_enabled: Bool,
        form_tangent: Bool,
        mut tangent: List[Float64],
    ) -> ContactVector:
        self.a = self.initial_a + Point2d(u[self.dofs[0]], u[self.dofs[1]])
        self.b = self.initial_b + Point2d(u[self.dofs[3]], u[self.dofs[4]])
        self.secondary = self.initial_secondary + Point2d(u[self.dofs[6]], u[self.dofs[7]])
        self.trial_a_rotation = u[self.dofs[2]]
        self.trial_b_rotation = u[self.dofs[5]]
        self.multiplier = u[self.dofs[8]]
        var a = self.a_tangent + (self.trial_a_rotation - self.a_rotation) * _perpendicular(
            self.a_tangent
        )
        var b = self.b_tangent + (self.trial_b_rotation - self.b_rotation) * _perpendicular(
            self.b_tangent
        )
        var offset = self.secondary - self._curve(self.shape, a, b)
        self.gap = _dot(self.normal, offset) - self.radius

        var force = ContactVector(fill=0.0)
        if form_tangent:
            tangent.resize(100, 0.0)
            tangent[99] = 1.0
        if self.active:
            var centerline_tangent = self._curve(self.derivative, self.a_tangent, self.b_tangent)
            var slip = _dot(
                self.surface_tangent,
                offset - self.surface_offset * _perpendicular(centerline_tangent),
            )
            self.material.evaluate(slip, self.multiplier, friction_enabled)
            for i in range(8):
                force[i] = (
                    -self.multiplier * self.normal_gradient[i]
                    + self.material.shear_force * self.slip_gradient[i]
                )
                if form_tangent:
                    tangent[80 + i] = -self.normal_gradient[i]
                    tangent[10 * i + 8] = (
                        -self.normal_gradient[i]
                        + self.material.normal_coupling * self.slip_gradient[i]
                    )
                    for j in range(8):
                        tangent[10 * i + j] = (
                            self.material.shear_tangent
                            * self.slip_gradient[i]
                            * self.slip_gradient[j]
                        )
            force[8] = -self.gap
        else:
            if self.releasing:
                self.material.evaluate(0.0, self.multiplier, friction_enabled)
            force[8] = self.multiplier
            if form_tangent:
                tangent[88] = 1.0
        self.last_force = force
        return force

    fn commit(mut self):
        self.a_tangent += (self.trial_a_rotation - self.a_rotation) * _perpendicular(self.a_tangent)
        self.b_tangent += (self.trial_b_rotation - self.b_rotation) * _perpendicular(self.b_tangent)
        self.a_rotation = self.trial_a_rotation
        self.b_rotation = self.trial_b_rotation
        self._update_projection()
        self.releasing = self.multiplier <= -self.force_tol or self.xi <= 0.0 or self.xi >= 1.0
        self.active = self.gap < self.gap_tol and not self.releasing
        self.material.commit()

    fn secondary_force(self) -> List[Float64]:
        return [-self.last_force[6], -self.last_force[7]]

    fn friction_force(self) -> List[Float64]:
        return [
            self.material.shear_force * self.surface_tangent[0],
            self.material.shear_force * self.surface_tangent[1],
        ]

    fn contact_force(self) -> List[Float64]:
        return [self.material.normal_force, self.material.shear_force]

    fn primary_force(self) -> List[Float64]:
        var result: List[Float64] = []
        for i in range(6):
            result.append(-self.last_force[i])
        return result^
