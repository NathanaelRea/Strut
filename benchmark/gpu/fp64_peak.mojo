"""Register-resident FP64 FMA throughput for an optimistic compute estimate."""

from collections import InlineArray
from gpu import block_idx, thread_idx
from gpu.host import DeviceContext
from math import fma
from memory import UnsafePointer
from time import perf_counter_ns

alias BLOCKS = 1024
alias LANES = 256
alias ROUNDS = 4096
alias ACCUMULATORS = 16


fn fma_probe(sums: UnsafePointer[Float64], rounds: Int, multiplier: Float64):
    var index = block_idx.x * LANES + thread_idx.x
    var values = InlineArray[Float64, ACCUMULATORS](fill=0.0)

    @parameter
    for i in range(ACCUMULATORS):
        values[i] = 1.0 + 0.001 * Float64(index + i)
    for _ in range(rounds):

        @parameter
        for i in range(ACCUMULATORS):
            values[i] = fma(values[i], multiplier, Float64(i) + 0.25)
    var total = 0.0

    @parameter
    for i in range(ACCUMULATORS):
        total += values[i]
    sums[index] = total


fn main() raises:
    var ctx = DeviceContext()
    var sums = ctx.enqueue_create_buffer[DType.float64](BLOCKS * LANES)
    var kernel = ctx.compile_function[fma_probe]()
    var best = 0.0
    for run in range(5):
        var start = perf_counter_ns()
        ctx.enqueue_function(kernel, sums, ROUNDS, 1.000001, grid_dim=BLOCKS, block_dim=LANES)
        ctx.synchronize()
        var elapsed_ns = perf_counter_ns() - start
        if run > 0:
            best = max(
                best, 2e-3 * Float64(BLOCKS * LANES * ROUNDS * ACCUMULATORS) / Float64(elapsed_ns)
            )
    print(best)
