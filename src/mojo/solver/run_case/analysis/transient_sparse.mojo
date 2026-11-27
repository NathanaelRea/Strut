"""Sparse matrix set shared by the transient drivers.

`TransientSparseMatrices` owns the reduced `SparsePattern` and every value
array a Newmark driver needs; each array is a flat `List[Float64]` indexed by
pattern slot (see `solver.sparse_pattern` for the CSR layout and the
lower-triangular CSC view used by symmetric factorizations).

The Rayleigh damping matrix C is never stored. Its value at slot `s` is formed
on the fly from the arrays exactly as the dense drivers formed `C_ff[i][j]`:

    C = alphaM * M_rayleigh
      + betaK     * (k_damp      if use_k_damp      else k_tangent)
      + betaKInit * (k_init_damp if use_k_init_damp else k_init)
      + betaKComm * (k_comm_damp if use_k_comm_damp else k_comm)
      + c_zero_length

and the effective Newmark matrix is

    K_eff = k_tangent (+ k_zero_length_damp) (+ a1 * C) + a0 * mass.
"""

from collections import List

from solver.run_case.analysis.penalty import PenaltyConstraints
from os import abort

from solver.run_case.linear_solver_backend import (
    LinearSolverBackend,
    load_pattern_values,
)
from solver.sparse_pattern import SparseAssembler, SparsePattern


struct TransientSparseMatrices(Movable):
    var pattern: SparsePattern
    # Assembly target: the current (MPC-reduced) tangent of the latest assembly.
    var assembler: SparseAssembler
    # Tangent used for K_eff (frozen per algorithm mode) and its link-filtered
    # Rayleigh variant.
    var k_tangent: List[Float64]
    var k_damp: List[Float64]
    var k_current_damp: List[Float64]
    var k_init: List[Float64]
    var k_init_damp: List[Float64]
    var k_comm: List[Float64]
    var k_comm_damp: List[Float64]
    var mass: List[Float64]
    var mass_rayleigh: List[Float64]
    var c_zero_length: List[Float64]
    var c_coupled: List[Float64]
    var penalty: PenaltyConstraints
    var regional_damping: List[Float64]
    var modal_damping: List[Float64]
    var k_zero_length_damp: List[Float64]
    var k_link_all: List[Float64]
    var k_link_rayleigh: List[Float64]
    # Scratch.
    var k_effective: List[Float64]
    var damping_force: List[Float64]
    var inertia_force: List[Float64]
    # Damping definition.
    var alpha_m: Float64
    var beta_k: Float64
    var beta_k_init: Float64
    var beta_k_comm: Float64
    var use_k_damp: Bool
    var use_current_k_damp: Bool
    var use_k_init_damp: Bool
    var use_k_comm_damp: Bool
    var has_c_zero_length: Bool
    var has_effective_damping: Bool

    fn __init__(out self, var pattern: SparsePattern, var assembler: SparseAssembler):
        self.pattern = pattern^
        self.assembler = assembler^
        self.k_tangent = []
        self.k_damp = []
        self.k_current_damp = []
        self.k_init = []
        self.k_init_damp = []
        self.k_comm = []
        self.k_comm_damp = []
        self.mass = []
        self.mass_rayleigh = []
        self.c_zero_length = []
        self.c_coupled = []
        self.penalty = PenaltyConstraints()
        self.regional_damping = []
        self.modal_damping = []
        self.k_zero_length_damp = []
        self.k_link_all = []
        self.k_link_rayleigh = []
        self.k_effective = []
        self.damping_force = []
        self.inertia_force = []
        self.alpha_m = 0.0
        self.beta_k = 0.0
        self.beta_k_init = 0.0
        self.beta_k_comm = 0.0
        self.use_k_damp = False
        self.use_current_k_damp = False
        self.use_k_init_damp = False
        self.use_k_comm_damp = False
        self.has_c_zero_length = False
        self.has_effective_damping = False
        self.damping_force.resize(self.pattern.n, 0.0)
        self.inertia_force.resize(self.pattern.n, 0.0)

    @always_inline
    fn nnz(self) -> Int:
        return self.pattern.nnz()

    @always_inline
    fn damping_value(self, s: Int, use_trial_stiffness: Bool = False) -> Float64:
        var c = 0.0
        if self.alpha_m != 0.0:
            c = self.alpha_m * self.mass_rayleigh[s]
        if self.beta_k != 0.0:
            if use_trial_stiffness and self.use_current_k_damp:
                c += self.beta_k * self.k_current_damp[s]
            elif self.use_k_damp:
                c += self.beta_k * self.k_damp[s]
            else:
                c += self.beta_k * self.k_tangent[s]
        if self.beta_k_init != 0.0:
            if self.use_k_init_damp:
                c += self.beta_k_init * self.k_init_damp[s]
            else:
                c += self.beta_k_init * self.k_init[s]
        if self.beta_k_comm != 0.0:
            if self.use_k_comm_damp:
                c += self.beta_k_comm * self.k_comm_damp[s]
            else:
                c += self.beta_k_comm * self.k_comm[s]
        if self.has_c_zero_length:
            c += self.c_zero_length[s]
        if len(self.c_coupled) > 0:
            c += self.c_coupled[s]
        if len(self.regional_damping) > 0:
            c += self.regional_damping[s]
        if len(self.modal_damping) > 0:
            c += self.modal_damping[s]
        return c

    fn compute_damping_force(mut self, v: List[Float64]):
        """damping_force = C v (zero when there is no effective damping)."""
        var n = self.pattern.n
        if not self.has_effective_damping:
            for i in range(n):
                self.damping_force[i] = 0.0
            return
        for i in range(n):
            var sum = 0.0
            for s in range(self.pattern.row_start[i], self.pattern.row_start[i + 1]):
                sum += self.damping_value(s, True) * v[self.pattern.col_index[s]]
            self.damping_force[i] = sum

    fn compute_inertia_force(mut self, a: List[Float64]):
        """inertia_force = M a."""
        for i in range(self.pattern.n):
            var sum = 0.0
            for s in range(self.pattern.row_start[i], self.pattern.row_start[i + 1]):
                sum += self.mass[s] * a[self.pattern.col_index[s]]
            self.inertia_force[i] = sum

    fn load_effective(
        mut self,
        mut backend: LinearSolverBackend,
        a0: Float64,
        a1: Float64,
        include_zero_length_damp: Bool,
    ):
        """Form K_eff slot-wise and load it into the solver backend."""
        var nnz = self.pattern.nnz()
        if len(self.k_effective) != nnz:
            self.k_effective.resize(nnz, 0.0)
        if len(self.k_tangent) != nnz or len(self.mass) != nnz:
            abort("transient sparse matrices not initialized")
        for s in range(nnz):
            var value = self.k_tangent[s]
            if len(self.penalty.stiffness) > 0:
                value += self.penalty.stiffness[s]
            if include_zero_length_damp:
                value += self.k_zero_length_damp[s]
            if self.has_effective_damping:
                value += a1 * self.damping_value(s)
            value += a0 * self.mass[s]
            self.k_effective[s] = value
        load_pattern_values(backend, self.pattern, self.k_effective)
