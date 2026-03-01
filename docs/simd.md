# SIMD in Mojo

This note summarizes the SIMD guidance that is most useful for Strut from:

- `docs/agent-reference/llms-mojo.txt`
- `docs/agent-reference/modular/mojo/stdlib/std/builtin/simd.mojo`
- `docs/agent-reference/modular/mojo/docs/manual/layout/tensors.mdx`
- `docs/agent-reference/modular/max/tests/tests/torch/custom_ops/custom_ops.mojo`

The short version: keep SIMD width compile-time, derive it from the target unless you have a measured reason not to, prefer contiguous aligned loads and stores, and use `foreach` or `LayoutTensor.vectorize()` instead of hand-unrolling tensor code.

## Strut review rules

- Keep contiguous SIMD helpers and indexed gather/scatter helpers separate so irregular access is visible in review.
- Do not add new generic `SIMD[..., 4]` hot-path kernels without benchmark evidence for that specialization.
- Do not manually pack lanes from scalar indexing when data can be compacted and streamed with contiguous `load/store`.
- When a load/store API accepts alignment, either prove it and pass `align_of[SIMD[dtype, width]]()` or fall back to a safe unaligned path.

## Best practices

### 1. Make width a compile-time parameter

Mojo `SIMD[dtype, width]` widths are compile-time constants, and the width must be a positive power of two.

- Prefer generic helpers such as `fn foo[width: Int](...)`.
- Do not make width a runtime variable.
- Do not hardcode `4` or `8` unless the kernel is intentionally specialized for a known target or data layout.

For portable defaults, use `simd_width_of`:

```mojo
from sys import simd_width_of

comptime WIDTH = simd_width_of[DType.float64]()
var v = SIMD[DType.float64, WIDTH](1.0)
```

`simd_width_of[dtype]()` is the default width selector used throughout Modular's tensor and kernel APIs.

### 2. Prefer contiguous loads and stores over lane-by-lane assembly

The fastest path is usually:

- contiguous memory
- vector width matched to the target
- alignment passed explicitly when the API supports it

For pointer-style loops, use `load` and `store` instead of building vectors from scalar indexing:

```mojo
from memory import UnsafePointer
from sys import align_of, simd_width_of

comptime WIDTH = simd_width_of[DType.float64]()
comptime ALIGN = align_of[SIMD[DType.float64, WIDTH]]()

fn axpy(
    x: UnsafePointer[Float64, MutAnyOrigin],
    y: UnsafePointer[Float64, MutAnyOrigin],
    a: Float64,
    count: Int,
):
    var i = 0
    var a_vec = SIMD[DType.float64, WIDTH](a)

    while i + WIDTH <= count:
        var xv = x.load[width=WIDTH, alignment=ALIGN](i)
        var yv = y.load[width=WIDTH, alignment=ALIGN](i)
        y.store[width=WIDTH, alignment=ALIGN](i, a_vec * xv + yv)
        i += WIDTH

    while i < count:
        y[i] = a * x[i] + y[i]
        i += 1
```

For Strut, this is the main upgrade path from code that manually constructs `SIMD[..., 4](a[i], a[i+1], ...)`.

### 3. Always treat alignment as part of the implementation

The Modular docs are explicit here: vector load/store APIs accept alignment because some GPU paths require it for correct vector codegen.

- Use `align_of[SIMD[dtype, width]]()` when you know the vector type.
- Thread alignment through helper lambdas that expose it as a compile-time parameter.
- If you cannot prove alignment, fall back to a safe path instead of lying to the compiler.

Typical pattern:

```mojo
comptime alignment = align_of[SIMD[dtype, width]]()
var vec = ptr.load[width=width, alignment=alignment](idx)
ptr.store[width=width, alignment=alignment](idx, vec)
```

### 4. Use `foreach` for elementwise tensor kernels

For tensor-shaped elementwise work, `foreach` is the preferred high-level entry point. It already carries a target-specific SIMD width default, and the lambda returns a SIMD value directly.

```mojo
from tensor import InputTensor, OutputTensor, foreach
from runtime.asyncrt import DeviceContextPtr
from utils.index import IndexList

@parameter
@always_inline
fn add_kernel[
    width: Int
](idx: IndexList[C.rank]) -> SIMD[C.dtype, width]:
    var a = A.load[width](idx)
    var b = B.load[width](idx)
    return a + b

foreach[add_kernel, target=target](C, ctx)
```

If the output side needs alignment-aware stores, use the overload that provides `alignment: Int` to the output function and pass that into `store[alignment=alignment]`.

### 5. Use `load()` and `store()` instead of `tensor[...]` in vector code

`LayoutTensor.__getitem__()` returns a SIMD value whose element width may not be statically obvious at the call site. For scalar code this often forces a `[0]` extract. For vector code, prefer explicit `load[width](...)` and `store[width](...)` so width and alignment stay under your control.

Good:

```mojo
var vec = tensor.load[4](row, col)
tensor.store(row, col, vec * 2)
```

Less useful in SIMD-heavy code:

```mojo
var scalar = tensor[row, col][0]
```

### 6. Use `vectorize()` and `distribute()` for tiled GPU work

For `LayoutTensor`-based kernels, Modular's recommended pattern is:

1. tile the tensor
2. vectorize the tile
3. distribute the vectorized tile across threads
4. copy or compute on thread-local fragments

```mojo
comptime thread_layout = Layout.row_major(WARP_SIZE // simd_width, simd_width)

var global_fragment = global_tile.vectorize[
    1, simd_width
]().distribute[thread_layout](lane_id())

var shared_fragment = shared_tile.vectorize[
    1, simd_width
]().distribute[thread_layout](lane_id())

shared_fragment.copy_from_async(global_fragment)
```

This is the right shape for thread coarsening, vectorized copies, and shared-memory staging. Prefer this over hand-derived per-lane pointer arithmetic when the problem already fits `LayoutTensor`.

### 7. Keep work in registers once loaded

After data is in a SIMD register, use SIMD-native ops instead of scalar lane loops:

- arithmetic: `+`, `-`, `*`, `/`
- type conversion: `.cast[...]()`
- masking and selection: `select(...)`
- in-register rearrangement: `shuffle`, `slice`, `join`, `split`, `interleave`
- horizontal reduction: `reduce_add()`, `reduce_min()`, `reduce_max()`

Typical dot-product style pattern:

```mojo
var total = 0.0
var a_vec = ptr_a.load[width=WIDTH](i)
var b_vec = ptr_b.load[width=WIDTH](i)
total += (a_vec * b_vec).reduce_add()
```

This matches how Strut already uses SIMD in several solver loops.

### 8. Widen accumulators when precision matters

The Modular kernels frequently cast narrow inputs to a wider accumulation type before doing math.

- Integer or low-precision floating inputs often need `float32` or `float64` accumulation.
- Convert once after load, not repeatedly per lane.

Example:

```mojo
var x = input.load[width](idx).cast[DType.float32]()
var y = weights.load[width](idx).cast[DType.float32]()
var acc = x * y
```

### 9. Handle tails explicitly

A SIMD fast path is rarely the whole loop. You still need a remainder strategy when `count % WIDTH != 0`.

The two standard options are:

- bulk SIMD loop plus scalar cleanup
- masked or partial vector path when the surrounding API supports it

For reductions over variable-width vector steps, Modular exposes `reduce_add_simd(...)` specifically for cases where the step SIMD width changes across iterations.

### 10. Use gather/scatter only for genuinely irregular access

Mojo exposes masked `gather` and `scatter`, with explicit mask, passthrough, and optional alignment. Use them when you truly have indexed or sparse access patterns.

They are usually worse than contiguous vector loads and stores, so prefer:

- layout changes
- tiling
- vectorized contiguous fragments

before falling back to gather/scatter.

## Implementation patterns

### Elementwise tensor op

Use this when the problem is already expressed as tensors:

```mojo
@parameter
@always_inline
fn doit[width: Int](idx: IndexList[B.rank]) -> SIMD[B.dtype, width]:
    var a = A.load[width](idx)
    return a + type_of(a)(increment)

foreach[doit, target=target](B, ctx)
```

Why this is the default:

- width stays compile-time
- target-specific width selection is automatic
- tensor indexing and output shape traversal are delegated to the framework

### Raw 1D buffer loop

Use this when you have plain solver arrays and do not want to introduce tensor abstractions:

```mojo
comptime WIDTH = simd_width_of[DType.float64]()
comptime ALIGN = align_of[SIMD[DType.float64, WIDTH]]()

var i = 0
while i + WIDTH <= n:
    var lhs = x.load[width=WIDTH, alignment=ALIGN](i)
    var rhs = y.load[width=WIDTH, alignment=ALIGN](i)
    out.store[width=WIDTH, alignment=ALIGN](i, lhs + rhs)
    i += WIDTH

while i < n:
    out[i] = x[i] + y[i]
    i += 1
```

This is usually the simplest pattern for Strut assembly, solver, and vector update loops.

### Tiled GPU copy or compute

Use this when the kernel naturally operates on blocks or tiles:

```mojo
var global_tile = tensor.tile[block_size, block_size](
    Int(block_idx.y), Int(block_idx.x)
)

var global_fragment = global_tile.vectorize[
    1, simd_width
]().distribute[thread_layout](lane_id())

var shared_fragment = shared_tile.vectorize[
    1, simd_width
]().distribute[thread_layout](lane_id())

shared_fragment.copy_from_async(global_fragment)
```

This is the documented Modular path for combining vectorized copies with thread partitioning.

## Practical guidance for Strut

- For existing hot loops that manually assemble `SIMD[..., 4]` from scalar indexing, first switch to pointer `load/store` plus a scalar cleanup loop.
- If a loop is tensor-shaped or destined for GPU kernels, move toward `foreach` or `LayoutTensor.vectorize()` instead of adding more manual lane math.
- Keep hardcoded widths only where they are benchmarked and justified for a specific target.
- Treat alignment, remainder handling, and accumulation precision as part of the algorithm, not cleanup details.

## References

- [`docs/agent-reference/llms-mojo.txt`](./agent-reference/llms-mojo.txt)
- [`docs/agent-reference/modular/mojo/stdlib/std/builtin/simd.mojo`](./agent-reference/modular/mojo/stdlib/std/builtin/simd.mojo)
- [`docs/agent-reference/modular/mojo/stdlib/std/sys/info.mojo`](./agent-reference/modular/mojo/stdlib/std/sys/info.mojo)
- [`docs/agent-reference/modular/mojo/stdlib/std/sys/intrinsics.mojo`](./agent-reference/modular/mojo/stdlib/std/sys/intrinsics.mojo)
- [`docs/agent-reference/modular/mojo/docs/manual/layout/tensors.mdx`](./agent-reference/modular/mojo/docs/manual/layout/tensors.mdx)
- [`docs/agent-reference/modular/max/tests/tests/torch/custom_ops/custom_ops.mojo`](./agent-reference/modular/max/tests/tests/torch/custom_ops/custom_ops.mojo)
