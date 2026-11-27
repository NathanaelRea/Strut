"""Updated-basis discrete Kirchhoff quadrilateral of OpenSees."""

from collections import InlineArray, List

from elements.shell4 import SHELL4_DOFS, SHELL4_GEOMETRY_SIZE, shell4_layered_stiffness_and_residual
from elements.shell_discrete_kirchhoff import discrete_kirchhoff_updated_geometry
from materials import UniMaterialDef, UniMaterialState
from materials.concrete_s import ConcreteSState
from sections import LayeredShellSectionDef


alias _Vector = InlineArray[Float64, SHELL4_DOFS]
alias _Strains = InlineArray[Float64, 32]
alias _Point = SIMD[DType.float64, 4]


struct ShellNLDKGQState(Copyable, Movable):
    var reference: InlineArray[_Point, 4]
    var committed_u: _Vector
    var trial_u: _Vector
    var committed_e: _Strains
    var trial_e: _Strains
    var elastic_section: LayeredShellSectionDef

    fn __init__(out self, x: List[Float64], y: List[Float64], z: List[Float64]):
        self.reference = InlineArray[_Point, 4](fill=_Point(0.0))
        for n in range(4):
            self.reference[n] = _Point(x[n], y[n], z[n], 0.0)
        self.committed_u = _Vector(fill=0.0)
        self.trial_u = _Vector(fill=0.0)
        self.committed_e = _Strains(fill=0.0)
        self.trial_e = _Strains(fill=0.0)
        self.elastic_section = LayeredShellSectionDef()

    fn __copyinit__(out self, existing: Self):
        self.reference = existing.reference.copy()
        self.committed_u = existing.committed_u.copy()
        self.trial_u = existing.trial_u.copy()
        self.committed_e = existing.committed_e.copy()
        self.trial_e = existing.trial_e.copy()
        self.elastic_section = existing.elastic_section.copy()

    fn commit(mut self):
        self.committed_u = self.trial_u.copy()
        self.committed_e = self.trial_e.copy()

    fn revert(mut self):
        self.trial_u = self.committed_u.copy()
        self.trial_e = self.committed_e.copy()

    fn evaluate[
        tangent: Bool
    ](
        mut self,
        sec_def: LayeredShellSectionDef,
        definitions: UnsafePointer[UniMaterialDef],
        states: UnsafePointer[UniMaterialState],
        concrete_states: UnsafePointer[ConcreteSState],
        u: UnsafePointer[Float64],
        matrix: UnsafePointer[Float64],
        force: UnsafePointer[Float64],
        scratch: UnsafePointer[Float64],
    ):
        var geometry = InlineArray[Float64, SHELL4_GEOMETRY_SIZE](fill=0.0)
        var geometric = InlineArray[Float64, 4 * 2 * SHELL4_DOFS](fill=0.0)
        var du = _Vector(fill=0.0)
        for i in range(SHELL4_DOFS):
            self.trial_u[i] = u[i]
            du[i] = u[i] - self.committed_u[i]
        discrete_kirchhoff_updated_geometry(
            self.reference,
            UnsafePointer(to=self.committed_u[0]),
            u,
            UnsafePointer(to=geometry[0]),
            UnsafePointer(to=geometric[0]),
        )
        # OpenSees accumulates the linear displacement increment at each
        # integration point and zeros the quadratic strain increment. Its
        # tangent still includes the initial-stress contribution from BG.
        shell4_layered_stiffness_and_residual[tangent, True](
            sec_def,
            definitions,
            states,
            concrete_states,
            UnsafePointer(to=geometry[0]),
            UnsafePointer(to=du[0]),
            matrix,
            force,
            scratch,
            UnsafePointer(to=self.committed_e[0]),
            UnsafePointer(to=self.trial_e[0]),
            UnsafePointer(to=geometric[0]),
        )
