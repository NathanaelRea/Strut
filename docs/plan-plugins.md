# Plugin Architecture Plan

This document outlines a plugin architecture for Strut's finite element
components in Mojo. The design aims to keep the plugin contract small, preserve
hot-path performance, and let third-party packages add materials and elements
without depending on the solver core.

## Goals

- Define a stable plugin contract for constitutive models and finite elements.
- Keep plugin packages independent from solver, mesh, assembly, and I/O code.
- Preserve compile-time specialization in element and material hot paths.
- Make plugin registration explicit and easy to maintain.
- Start with a minimal interface that can grow through optional traits later.

## Proposed Package Split

```text
fea-interfaces/        # tiny package: traits + shared types
fea-core/              # solver, assembly, mesh, I/O
  └── default_plugins/ # built-in materials and elements

third-party-plugin/    # depends only on fea-interfaces
```

The critical rule is that plugins do not depend on `fea-core`. Both the core
and plugin packages depend on `fea-interfaces`, and the core consumes any type
that satisfies those traits.

## Responsibility Split

| Responsibility | Owner |
| --- | --- |
| Constitutive law: stress from strain, material tangent | Material plugin |
| Shape functions, B-matrix, local stiffness and force integration | Element plugin |
| Mesh management, DOF numbering, global assembly | Core |
| Linear and nonlinear solvers | Core |
| Boundary conditions and loads | Core |
| I/O and post-processing | Core |

Material plugins should stay narrow: given strain, compute stress and tangent.
Element plugins own local formulation details: topology, interpolation,
integration, local stiffness, and local internal force evaluation.

## Interface Contract

The interface package should stay small and semver-stable. Mojo traits are the
main abstraction.

### Material Trait

```mojo
from tensor import Tensor

trait Material(Copyable, Movable):
    @staticmethod
    fn name() -> String: ...

    @staticmethod
    fn num_parameters() -> Int: ...

    fn __init__(out self, parameters: List[Float64]): ...

    fn compute_stress(
        self, strain: Tensor[DType.float64]
    ) -> Tensor[DType.float64]: ...

    fn compute_tangent(
        self, strain: Tensor[DType.float64]
    ) -> Tensor[DType.float64]: ...
```

### Element Trait

```mojo
from tensor import Tensor

trait Element(Copyable, Movable):
    @staticmethod
    fn name() -> String: ...

    @staticmethod
    fn num_nodes() -> Int: ...

    @staticmethod
    fn num_dof_per_node() -> Int: ...

    @staticmethod
    fn spatial_dim() -> Int: ...

    fn stiffness_matrix[M: Material](
        self,
        node_coords: Tensor[DType.float64],
        material: M,
    ) -> Tensor[DType.float64]: ...

    fn internal_forces[M: Material](
        self,
        node_coords: Tensor[DType.float64],
        displacements: Tensor[DType.float64],
        material: M,
    ) -> Tensor[DType.float64]: ...
```

### Design Notes

- `stiffness_matrix[M: Material]` and `internal_forces[M: Material]` keep the
  element-material inner loop monomorphized for each concrete pair.
- Flat `List[Float64]` parameter input keeps parsing and serialization simple.
- Self-documentation can be added later with `parameter_names()`.
- Path-dependent models should be added through optional traits rather than by
  expanding the base `Material` contract too early.

## Optional Traits

Optional capabilities should use trait composition so the base interfaces stay
stable.

```mojo
trait StatefulMaterial(Material):
    fn num_state_vars(self) -> Int: ...
    fn update_state(mut self, strain: Tensor[DType.float64]): ...
```

Likely follow-on traits include stateful plasticity or damage models, thermal
coupling, and other multiphysics extensions.

## Registration Model

Mojo is compiled, so plugin discovery should start as an explicit compile-time
registration step.

### Recommended Initial Approach

Users maintain a single `plugins.mojo` entry point:

```mojo
from fea_core.registry import Registry
from fea_core.default_plugins.linear_elastic import LinearElastic
from fea_core.default_plugins.tri3 import Tri3
from fea_core.default_plugins.quad4 import Quad4
from hyperelastic_plugin import MooneyRivlin

fn build_registry() -> Registry:
    var reg = Registry()
    reg.register_material[LinearElastic]()
    reg.register_material[MooneyRivlin]()
    reg.register_element[Tri3]()
    reg.register_element[Quad4]()
    return reg
```

This keeps plugin onboarding to one import and one registration line per type.
Core code and plugin code remain unchanged.

### Future Convenience Layer

Python-side discovery can be added later as a code-generation step that scans
installed packages, writes `plugins.mojo`, and then compiles the app. That is a
convenience layer, not a dependency for the first version.

## Registry Design

The registry has to store heterogeneous plugin types behind a uniform lookup
interface. The expected implementation is type erasure with function pointers.

```mojo
struct MaterialFactory:
    var name: String
    var num_params: Int
    var _create_fn: fn(List[Float64]) -> AnyMaterial

struct Registry:
    var materials: Dict[String, MaterialFactory]
    var elements: Dict[String, ElementFactory]

    fn register_material[M: Material](mut self):
        var factory = MaterialFactory(
            name=M.name(),
            num_params=M.num_parameters(),
            _create_fn=_make_material_factory[M](),
        )
        self.materials[M.name()] = factory

    fn create_material(
        self, name: String, params: List[Float64]
    ) -> AnyMaterial:
        return self.materials[name]._create_fn(params)
```

`AnyMaterial` and `AnyElement` should encapsulate the type-erased payload and
vtable-like function pointers needed for factory construction and top-level
dispatch.

Runtime dispatch should be limited to registry lookup and per-element entry
points. Quadrature-point work inside a concrete element should still benefit
from compile-time specialization.

## Third-Party Plugin Shape

A third-party material package should look roughly like this:

```text
hyperelastic-plugin/
├── mojoproject.toml
└── src/
    └── hyperelastic_plugin/
        ├── __init__.mojo
        └── mooney_rivlin.mojo
```

```toml
[project]
name = "hyperelastic-plugin"
version = "0.1.0"

[dependencies]
fea-interfaces = ">=0.2.0"
```

```mojo
from fea_interfaces.material import Material

struct MooneyRivlin(Material):
    var c1: Float64
    var c2: Float64

    @staticmethod
    fn name() -> String:
        return "mooney_rivlin"

    @staticmethod
    fn num_parameters() -> Int:
        return 2

    fn __init__(out self, parameters: List[Float64]):
        self.c1 = parameters[0]
        self.c2 = parameters[1]

    fn compute_stress(
        self, strain: Tensor[DType.float64]
    ) -> Tensor[DType.float64]:
        ...

    fn compute_tangent(
        self, strain: Tensor[DType.float64]
    ) -> Tensor[DType.float64]:
        ...
```

Installation should remain straightforward:

```bash
uv add git+https://github.com/someone/hyperelastic-plugin
```

Then the user updates `plugins.mojo`:

```mojo
from hyperelastic_plugin import MooneyRivlin
reg.register_material[MooneyRivlin]()
```

## Versioning Strategy

- `fea-interfaces` should follow semver.
- Additive interface changes should be minor releases.
- Breaking trait changes should be major releases.
- Plugins should depend on `>=major.minor.0` for the interface package.
- New capabilities should prefer optional traits over expanding mandatory base
  traits.

## Non-Goals for the First Cut

- Runtime loading of arbitrary compiled plugin binaries.
- Automatic plugin discovery at runtime.
- Sandboxing or trust isolation for third-party code.
- A large interface surface that tries to cover all future material models.

## Implementation Plan

### Phase 0: Freeze the Contract

- [ ] Confirm the package split and naming used in this repo.
  - [ ] Decide whether the public package names stay generic (`fea-*`) or map
        directly onto Strut package names.
  - [ ] Document the rule that plugins may depend on `fea-interfaces` only.
  - [ ] Define which shared math and tensor types belong in the interface
        package versus core.
- [ ] Keep the v1 interface intentionally minimal.
  - [ ] Finalize the required `Material` methods.
  - [ ] Finalize the required `Element` methods.
  - [ ] Defer stateful, thermal, and multiphysics extensions to optional traits.

### Phase 1: Create the Interface Package

- [ ] Add the `fea-interfaces` package.
  - [ ] Create `material.mojo` with the base `Material` trait.
  - [ ] Create `element.mojo` with the base `Element` trait.
  - [ ] Add shared exported types needed by both traits.
  - [ ] Add package-level documentation and version metadata.
- [ ] Validate the package is small and stable.
  - [ ] Check that it does not import solver, mesh, I/O, or assembly code.
  - [ ] Check that it exposes only contract types and shared utilities.

### Phase 2: Add Core Registry Support

- [ ] Add a registry layer in core for plugin lookup and construction.
  - [ ] Define `MaterialFactory` and `ElementFactory`.
  - [ ] Define `Registry` maps keyed by plugin name.
  - [ ] Implement `register_material[M]` and `register_element[E]`.
  - [ ] Implement `create_material(...)` and `create_element(...)`.
- [ ] Add type-erased wrappers for runtime lookup boundaries.
  - [ ] Design `AnyMaterial`.
  - [ ] Design `AnyElement`.
  - [ ] Keep runtime dispatch outside quadrature-point hot paths.
- [ ] Define failure behavior.
  - [ ] Report unknown plugin names clearly.
  - [ ] Validate parameter counts before construction.
  - [ ] Surface duplicate registrations as explicit errors.

### Phase 3: Add Built-In Default Plugins

- [ ] Implement one default material against the new trait.
  - [ ] Add `LinearElastic`.
  - [ ] Verify parameter parsing and naming are clear.
  - [ ] Validate stress and tangent outputs against known cases.
- [ ] Implement one default element against the new trait.
  - [ ] Choose `Tri3` or `Quad4` as the first reference element.
  - [ ] Implement topology metadata.
  - [ ] Implement local stiffness computation.
  - [ ] Implement internal force computation.
- [ ] Use the default plugins as the authoring benchmark.
  - [ ] Check that the trait APIs feel natural to implement.
  - [ ] Check that common math code does not leak core-only concerns into the
        interface package.

### Phase 4: Wire Registration into the Application

- [ ] Add a `plugins.mojo` entry point for compile-time registration.
  - [ ] Register built-in plugins in one place.
  - [ ] Ensure third-party packages can be imported alongside built-ins.
  - [ ] Keep plugin onboarding to one import and one registration line.
- [ ] Route input parsing through the registry.
  - [ ] Map material names in input files to material factories.
  - [ ] Map element names in input files to element factories.
  - [ ] Fail early when requested plugins are not registered.

### Phase 5: Test and Performance Validation

- [ ] Add contract-level tests for the new architecture.
  - [ ] Test registry registration and lookup.
  - [ ] Test parameter validation failures.
  - [ ] Test duplicate-name rejection.
- [ ] Add reference tests for default plugins.
  - [ ] Verify `LinearElastic` responses.
  - [ ] Verify the chosen first element's local stiffness and internal forces.
  - [ ] Exercise assembly through the core using registered plugins.
- [ ] Check performance assumptions.
  - [ ] Measure registry overhead at the element boundary.
  - [ ] Confirm no unexpected dynamic dispatch appears in hot loops.
  - [ ] Capture any compiler warnings and treat them as required fixes.

### Phase 6: External Plugin Experience

- [ ] Document the third-party plugin workflow.
  - [ ] Publish a minimal material plugin example.
  - [ ] Show package metadata and dependency expectations.
  - [ ] Show `uv add git+...` installation flow.
  - [ ] Show `plugins.mojo` registration flow.
- [ ] Prove the architecture with one external-style package.
  - [ ] Build a sample plugin outside the core package tree.
  - [ ] Verify it depends only on `fea-interfaces`.
  - [ ] Verify it can be registered without core code changes.

### Phase 7: Optional Future Work

- [ ] Add optional trait extensions only after the base contract settles.
  - [ ] Prototype `StatefulMaterial`.
  - [ ] Evaluate thermal or multiphysics capability traits.
  - [ ] Define migration guidance for plugin authors.
- [ ] Explore generated registration for convenience.
  - [ ] Evaluate Python-side plugin discovery.
  - [ ] Generate `plugins.mojo` from installed package metadata.
  - [ ] Keep explicit registration as the fallback path.

## Acceptance Criteria

- A plugin package can implement `Material` or `Element` without importing core.
- Core can register and construct built-in and third-party plugins by name.
- The first built-in material and element feel natural to implement against the
  traits.
- Runtime dispatch stays at registry and top-level element boundaries.
- The interface package remains small enough to version independently.
