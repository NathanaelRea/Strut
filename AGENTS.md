# AGENTS.md

Strut is a Mojo-based adaptation of OpenSees for high-performance finite element and earthquake analysis.

## Essentials

- Use `uv` for Python dependency management and commands.
- Never use the `mojo` command directly; always use `uv run mojo`.
- Never run `python` or `pytest` directly; always run tools via `uv run <command>` (for example, `uv run run_tests.py`).
- Always build the Mojo solver after Mojo changes and treat compiler warnings as required fixes (zero-warning target). Apply Mojo compiler recommendations, prefer allocation-light implementations, and structure for performance.
- Do not stub or mock unless the user explicitly asks for it.

## Non-Standard Commands

- Run unit tests: `uv run run_tests.py`
- Build/precompile Mojo solver: `scripts/build_mojo_solver.sh`

## Reference

When implementing OpenSees behavior, first inspect the real implementation

- [`docs/agent-reference/OpenSees/`](../agent-reference/OpenSees/): source of truth for OpenSees behavior and benchmarks.
- [`docs/agent-reference/modular/`](../agent-reference/modular/): source of truth for Mojo language/runtime docs, code, and examples.

## Mojo best practices

## 2. Common Patterns and Anti-patterns

### 2.1 Design Patterns Specific to Mojo

- **Mojo Stream Pattern:** Use Mojo streams to overlap data transfers and kernel execution.
- **Memory Pooling Pattern:** Implement memory pools to reduce the overhead of frequent memory allocations and deallocations.
- **Tiling Pattern:** Divide large data structures into smaller tiles to improve data locality and cache utilization.
- **Reduction Pattern:** Use parallel reduction algorithms to efficiently compute aggregate values.

### 2.2 Recommended Approaches for Common Tasks

- **Error Handling:** Use the Mojo error handling API to check for errors after each Mojo function call.
- **Memory Allocation:** Use `DeviceContext` APIs for memory management on GPU devices.
- **Kernel Launch:** Use the `enqueue_function` (or its type-checked variants) for launching a Mojo kernel.

### 2.3 Anti-patterns and Code Smells to Avoid

- **Synchronous Memory Transfers:** Avoid blocking memory transfers that stall the GPU.
- **Excessive Global Memory Access:** Minimize global memory access by using shared memory and registers.
- **Thread Divergence:** Avoid conditional branches that cause threads within a warp to execute different code paths.
- **Uncoalesced Memory Access:** Ensure that threads access memory in a coalesced manner to maximize memory bandwidth.
- **CPU-GPU Synchronization Bottlenecks:** Minimize the number of synchronization points between the CPU and GPU.

### 2.4 State Management Best Practices

- Encapsulate Mojo context and device management within a dedicated class or module.
- Avoid global state variables that can lead to unexpected behavior and concurrency issues.
- Use context managers (such as `DeviceContext`) to ensure that Mojo resources are properly released.

### 2.5 Error Handling Patterns

- Check the return value of every Mojo function call and handle errors appropriately.
- Implement custom error handling routines for specific error conditions.
- Log error messages with file name, line number, and a descriptive error message.

## 3. Performance Considerations

### 3.1 Optimization Techniques

- **Kernel Fusion:** Combine multiple kernels into a single kernel to reduce kernel launch overhead and data transfers.
  Some of this can happen automatically with our Graph Compiler.
- **Loop Unrolling:** Unroll loops to improve instruction-level parallelism.
- **Instruction Scheduling:** Optimize instruction scheduling to reduce pipeline stalls.
- **Constant Memory Usage:** Store frequently accessed read-only data in constant memory.
- **Texture Memory Usage:** Utilize texture memory for spatially coherent data access patterns.

### 3.2 Memory Management

- **Minimize Data Transfers:** Reduce the amount of data transferred between host and device.
- **Asynchronous Data Transfers:** Use asynchronous memory transfers with Mojo streams to overlap computation and communication.
- **Zero-Copy Memory:** Use zero-copy memory to directly access host memory from the GPU (use with caution due to performance implications).
- **Pinned Memory (Page-Locked Memory):** Use pinned memory for efficient asynchronous data transfers.

## More Instructions

- [Workflow and Planning](./docs/agent-guides/workflow-and-planning.md)
- [Mojo and OpenSees References](./docs/agent-guides/mojo-and-opensees-references.md)
- [Testing and Benchmarks](./docs/agent-guides/testing-and-benchmarks.md)
- [Dependency Management](./docs/agent-guides/dependency-management.md)
- [Knowledge Capture](./docs/agent-guides/knowledge-capture.md)
