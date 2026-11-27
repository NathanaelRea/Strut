// Matrix-free FP64 Q1 plane-strain elasticity with a geometric V-cycle.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if __has_include(<omp.h>)
#include <omp.h>
#else
extern "C" void omp_set_dynamic(int);
extern "C" void omp_set_num_threads(int);
#endif

namespace {
constexpr double tolerance = 1.e-9;
constexpr int max_iterations = 100;
constexpr int vector_threads = 256;
constexpr int max_reduction_blocks = 256;
constexpr int check_interval = 4;
using Clock = std::chrono::steady_clock;

double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void check(hipError_t status) {
    if (status != hipSuccess) {
        throw std::runtime_error(hipGetErrorString(status));
    }
}

struct Vec {
    double x, y;

    __host__ __device__ Vec operator+(Vec b) const { return {x + b.x, y + b.y}; }
    __host__ __device__ Vec operator-(Vec b) const { return {x - b.x, y - b.y}; }
    __host__ __device__ Vec operator*(double a) const { return {a * x, a * y}; }
    __host__ __device__ double dot(Vec b) const { return x * b.x + y * b.y; }
};

struct Grid {
    int cells;
    __host__ __device__ int pitch() const { return cells + 1; }
    __host__ __device__ int at(int x, int y) const { return y * pitch() + x; }
    int size() const { return pitch() * pitch(); }
};

// Integrated Q1 stiffness on square cells, with unit Young's modulus and thickness.
// The 2D stiffness coefficients have no cell-size scaling.
struct Stencil {
    double center, horizontal, vertical, corner, cross, inverse_diag;

    explicit Stencil(double nu) {
        const double mu = 1.0 / (2.0 * (1.0 + nu));
        const double lambda = nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        center = (4.0 / 3.0) * (lambda + 3.0 * mu);
        horizontal = -(2.0 * lambda + 3.0 * mu) / 3.0;
        vertical = lambda / 3.0;
        corner = -(lambda + 3.0 * mu) / 6.0;
        cross = -(lambda + mu) / 4.0;
        inverse_diag = 1.0 / center;
    }

    __host__ __device__ Vec apply(Grid grid, int i, const Vec *u) const {
        const int pitch = grid.pitch();
        const Vec h = u[i - 1] + u[i + 1];
        const Vec v = u[i - pitch] + u[i + pitch];
        const Vec same = u[i - pitch - 1] + u[i + pitch + 1];
        const Vec opposite = u[i - pitch + 1] + u[i + pitch - 1];
        const Vec diag = same + opposite;
        const Vec mix = same - opposite;
        return {
            center * u[i].x + horizontal * h.x + vertical * v.x + corner * diag.x + cross * mix.y,
            center * u[i].y + vertical * h.y + horizontal * v.y + corner * diag.y + cross * mix.x,
        };
    }
};

struct Apply {
    Grid grid;
    Stencil stencil;
    const Vec *input;
    Vec *output;

    __host__ __device__ void operator()(int x, int y) const {
        const int i = grid.at(x, y);
        output[i] = stencil.apply(grid, i, input);
    }
};

struct Residual {
    Grid grid;
    Stencil stencil;
    const Vec *rhs, *u;
    Vec *out;

    __host__ __device__ void operator()(int x, int y) const {
        const int i = grid.at(x, y);
        out[i] = rhs[i] - stencil.apply(grid, i, u);
    }
};

template <bool from_zero> struct Smooth {
    Grid grid;
    Stencil stencil;
    const Vec *rhs, *u;
    Vec *out;
    double weight;

    __host__ __device__ void operator()(int x, int y) const {
        const int i = grid.at(x, y);
        const double scale = weight * stencil.inverse_diag;
        if constexpr (from_zero) {
            out[i] = rhs[i] * scale;
        } else {
            out[i] = u[i] + (rhs[i] - stencil.apply(grid, i, u)) * scale;
        }
    }
};

// R = P^T for the integrated stiffness equation. Boundary nodes always hold zero.
struct Restrict {
    Grid coarse, fine;
    const Vec *input;
    Vec *out;

    __host__ __device__ void operator()(int x, int y) const {
        const int i = fine.at(2 * x, 2 * y);
        const int p = fine.pitch();
        const Vec edge = (input[i - 1] + input[i + 1]) + (input[i - p] + input[i + p]);
        const Vec corners =
            (input[i - p - 1] + input[i - p + 1]) + (input[i + p - 1] + input[i + p + 1]);
        out[coarse.at(x, y)] = input[i] + edge * 0.5 + corners * 0.25;
    }
};

struct Prolong {
    Grid fine, coarse;
    const Vec *input;
    Vec *out;

    __host__ __device__ void operator()(int x, int y) const {
        const int i = coarse.at(x / 2, y / 2);
        const int right = x & 1;
        const int down = (y & 1) * coarse.pitch();
        const Vec top = (input[i] + input[i + right]) * 0.5;
        const Vec bottom = (input[i + down] + input[i + down + right]) * 0.5;
        out[fine.at(x, y)] = out[fine.at(x, y)] + (top + bottom) * 0.5;
    }
};

struct Control {
    double rho, beta, alpha, target;
    int iterations;
    int status; // 0 running, 1 converged, -1 breakdown
};

struct HostGrid {
    template <class Op> void operator()(Grid grid, Op op) const {
#pragma omp parallel for if (grid.cells >= 128) schedule(static)
        for (int y = 1; y < grid.cells; ++y) {
            for (int x = 1; x < grid.cells; ++x) {
                op(x, y);
            }
        }
    }
};

template <class Op> __global__ void grid_kernel(Grid grid, Op op, const Control *control) {
    if (control->status) {
        return;
    }
    const int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (x < grid.cells && y < grid.cells) {
        op(x, y);
    }
}

struct DeviceGrid {
    hipStream_t stream;
    const Control *control;

    template <class Op> void operator()(Grid grid, Op op) const {
        const dim3 threads(32, 8);
        const dim3 blocks((grid.cells + 30) / 32, (grid.cells + 6) / 8);
        grid_kernel<<<blocks, threads, 0, stream>>>(grid, op, control);
    }
};

struct Level {
    Grid grid;
    Vec *rhs, *u, *work;
};

// Two damped Jacobi sweeps in each direction. The coarsest grid has one free
// node, where the two-component diagonal block is the exact stiffness.
template <class Execute>
void v_cycle(const std::vector<Level> &levels, int index, const Vec *rhs, Stencil stencil,
             Execute execute) {
    constexpr double weight = 0.65;
    const Level level = levels[index];
    const Grid grid = level.grid;
    if (grid.cells == 2) {
        execute(grid, Smooth<true>{grid, stencil, rhs, nullptr, level.u, 1.0});
        return;
    }
    execute(grid, Smooth<true>{grid, stencil, rhs, nullptr, level.work, weight});
    execute(grid, Smooth<false>{grid, stencil, rhs, level.work, level.u, weight});
    execute(grid, Residual{grid, stencil, rhs, level.u, level.work});
    const Level coarse = levels[index + 1];
    execute(coarse.grid, Restrict{coarse.grid, grid, level.work, coarse.rhs});
    v_cycle(levels, index + 1, coarse.rhs, stencil, execute);
    execute(grid, Prolong{grid, coarse.grid, coarse.u, level.u});
    execute(grid, Smooth<false>{grid, stencil, rhs, level.u, level.work, weight});
    execute(grid, Smooth<false>{grid, stencil, rhs, level.work, level.u, weight});
}

double host_dot(int size, const Vec *a, const Vec *b) {
    double total = 0;
#pragma omp parallel for if (size >= 16'384) schedule(static) reduction(+ : total)
    for (int i = 0; i < size; ++i) {
        total += a[i].dot(b[i]);
    }
    return total;
}

struct SolveStats {
    int iterations;
    double milliseconds;
};

class CpuSolver {
  public:
    CpuSolver(Grid grid, Stencil stencil) : grid(grid), stencil(stencil) {
        const int size = grid.size();
        for (Grid level = grid; level.cells >= 2; level.cells /= 2) {
            rhs.emplace_back(level.cells == grid.cells ? 0 : level.size());
            u.emplace_back(level.size());
            work.emplace_back(level.cells == 2 ? 0 : level.size());
            levels.push_back({level, rhs.back().data(), u.back().data(), work.back().data()});
        }
        x.resize(size);
        r.resize(size);
        p.resize(size);
        ap.resize(size);
    }

    SolveStats solve(const std::vector<Vec> &b, double norm2, std::vector<Vec> &solution) {
        const auto start = Clock::now();
        const int size = grid.size();
#pragma omp parallel for if (size >= 16'384) schedule(static)
        for (int i = 0; i < size; ++i) {
            x[i] = {};
            r[i] = b[i];
            p[i] = {};
        }
        const Vec *z = levels[0].u;
        const double target = tolerance * tolerance * norm2;
        double old_rho = 1.0;
        for (int i = 1; i <= max_iterations; ++i) {
            v_cycle(levels, 0, r.data(), stencil, HostGrid{});
            const double rho = host_dot(size, r.data(), z);
            if (!(rho > 0) || !std::isfinite(rho)) {
                throw std::runtime_error("CPU PCG preconditioner breakdown");
            }
            const double beta = rho / old_rho;
            old_rho = rho;
#pragma omp parallel for if (size >= 16'384) schedule(static)
            for (int j = 0; j < size; ++j) {
                p[j] = z[j] + p[j] * beta;
            }
            HostGrid{}(grid, Apply{grid, stencil, p.data(), ap.data()});
            const double denom = host_dot(size, p.data(), ap.data());
            if (!(denom > 0) || !std::isfinite(denom)) {
                throw std::runtime_error("CPU PCG operator breakdown");
            }
            const double alpha = rho / denom;
            double norm = 0.0;
#pragma omp parallel for if (size >= 16'384) schedule(static) reduction(+ : norm)
            for (int j = 0; j < size; ++j) {
                x[j] = x[j] + p[j] * alpha;
                r[j] = r[j] - ap[j] * alpha;
                norm += r[j].dot(r[j]);
            }
            if (norm <= target) {
                std::copy(x.begin(), x.end(), solution.begin());
                return {i, milliseconds(start)};
            }
        }
        throw std::runtime_error("CPU PCG reached the iteration limit");
    }

  private:
    Grid grid;
    Stencil stencil;
    std::vector<std::vector<Vec>> rhs, u, work;
    std::vector<Level> levels;
    std::vector<Vec> x, r, p, ap;
};

template <class T> class Buffer {
  public:
    Buffer(int size) {
        if (size == 0) {
            return;
        }
        check(hipMalloc(&pointer, size * sizeof(T)));
        const auto status = hipMemset(pointer, 0, size * sizeof(T));
        if (status != hipSuccess) {
            (void)hipFree(pointer);
            check(status);
        }
    }
    Buffer(Buffer &&other) noexcept : pointer(std::exchange(other.pointer, nullptr)) {}
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    ~Buffer() {
        if (pointer) {
            (void)hipFree(pointer);
        }
    }
    T *get() const { return pointer; }

  private:
    T *pointer = nullptr;
};

__device__ double block_sum(double sum) {
    __shared__ double shared[vector_threads];
    const int lane = threadIdx.x;
    shared[lane] = sum;
    __syncthreads();
    for (int stride = vector_threads / 2; stride > 0; stride /= 2) {
        if (lane < stride) {
            shared[lane] += shared[lane + stride];
        }
        __syncthreads();
    }
    return shared[0];
}

__global__ void dot_blocks(int size, const Vec *a, const Vec *b, const Control *control,
                           double *partial) {
    if (control->status) {
        return;
    }
    double sum = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        sum += a[i].dot(b[i]);
    }
    sum = block_sum(sum);
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = sum;
    }
}

enum class Reduction { Preconditioner, Operator, Residual };

template <Reduction step>
__global__ void finish_dot(int blocks, const double *partial, Control *control) {
    if (control->status) {
        return;
    }
    const int tid = threadIdx.x;
    const double value = block_sum(tid < blocks ? partial[tid] : 0.0);
    if (tid != 0) {
        return;
    }
    if (!isfinite(value)) {
        control->status = -1;
        return;
    }
    if constexpr (step == Reduction::Preconditioner) {
        if (!(value > 0.0)) {
            control->status = -1;
            return;
        }
        control->beta = value / control->rho;
        control->rho = value;
    } else if constexpr (step == Reduction::Operator) {
        if (!(value > 0.0)) {
            control->status = -1;
            return;
        }
        control->alpha = control->rho / value;
    } else {
        ++control->iterations;
        if (value <= control->target) {
            control->status = 1;
        }
    }
}

__global__ void grow_direction(int size, const Vec *z, Vec *p, const Control *control) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size && !control->status) {
        p[i] = z[i] + p[i] * control->beta;
    }
}

__global__ void update_and_norm(int size, Vec *x, Vec *r, const Vec *p, const Vec *ap,
                                const Control *control, double *partial) {
    if (control->status) {
        return;
    }
    double sum = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        x[i] = x[i] + p[i] * control->alpha;
        r[i] = r[i] - ap[i] * control->alpha;
        sum += r[i].dot(r[i]);
    }
    sum = block_sum(sum);
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = sum;
    }
}

class DeviceSolver {
  public:
    DeviceSolver(Grid grid, Stencil stencil)
        : grid(grid), x(grid.size()), r(grid.size()), p(grid.size()), ap(grid.size()), control(1),
          partial(max_reduction_blocks) {
        for (Grid level = grid; level.cells >= 2; level.cells /= 2) {
            rhs.emplace_back(level.cells == grid.cells ? 0 : level.size());
            u.emplace_back(level.size());
            work.emplace_back(level.cells == 2 ? 0 : level.size());
            levels.push_back({level, rhs.back().get(), u.back().get(), work.back().get()});
        }
        try {
            check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            capture(stencil);
        } catch (...) {
            cleanup();
            throw;
        }
    }

    ~DeviceSolver() { cleanup(); }

    SolveStats solve(const std::vector<Vec> &b, double norm2, std::vector<Vec> &solution) {
        const auto start = Clock::now();
        const size_t bytes = b.size() * sizeof(Vec);
        check(hipMemsetAsync(x.get(), 0, bytes, stream));
        check(hipMemsetAsync(p.get(), 0, bytes, stream));
        check(hipMemcpyAsync(r.get(), b.data(), bytes, hipMemcpyHostToDevice, stream));
        Control state{1.0, 0.0, 0.0, tolerance * tolerance * norm2, 0, 0};
        check(hipMemcpyAsync(control.get(), &state, sizeof(state), hipMemcpyHostToDevice, stream));
        for (int i = 0; i < max_iterations && state.status == 0; i += check_interval) {
            for (int j = 0; j < check_interval; ++j) {
                check(hipGraphLaunch(iteration, stream));
            }
            check(hipMemcpyAsync(&state, control.get(), sizeof(state), hipMemcpyDeviceToHost,
                                 stream));
            check(hipStreamSynchronize(stream));
        }
        if (state.status != 1) {
            throw std::runtime_error(state.status < 0 ? "GPU PCG breakdown"
                                                      : "GPU PCG reached iteration limit");
        }
        check(hipMemcpyAsync(solution.data(), x.get(), bytes, hipMemcpyDeviceToHost, stream));
        check(hipStreamSynchronize(stream));
        return {state.iterations, milliseconds(start)};
    }

  private:
    Grid grid;
    Buffer<Vec> x, r, p, ap;
    Buffer<Control> control;
    Buffer<double> partial;
    std::vector<Buffer<Vec>> rhs, u, work;
    std::vector<Level> levels;
    hipStream_t stream = nullptr;
    hipGraph_t graph = nullptr;
    hipGraphExec_t iteration = nullptr;

    void cleanup() {
        if (iteration) {
            (void)hipGraphExecDestroy(iteration);
        }
        if (graph) {
            (void)hipGraphDestroy(graph);
        }
        if (stream) {
            (void)hipStreamDestroy(stream);
        }
    }

    void capture(Stencil stencil) {
        const int size = grid.size();
        const int vector_blocks = (size + vector_threads - 1) / vector_threads;
        const int blocks = std::min(max_reduction_blocks, vector_blocks);
        const DeviceGrid execute{stream, control.get()};
        const Vec *z = levels[0].u;
        check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
        v_cycle(levels, 0, r.get(), stencil, execute);
        dot_blocks<<<blocks, vector_threads, 0, stream>>>(size, r.get(), z, control.get(),
                                                          partial.get());
        finish_dot<Reduction::Preconditioner>
            <<<1, vector_threads, 0, stream>>>(blocks, partial.get(), control.get());
        grow_direction<<<vector_blocks, vector_threads, 0, stream>>>(size, z, p.get(),
                                                                     control.get());
        execute(grid, Apply{grid, stencil, p.get(), ap.get()});
        dot_blocks<<<blocks, vector_threads, 0, stream>>>(size, p.get(), ap.get(), control.get(),
                                                          partial.get());
        finish_dot<Reduction::Operator>
            <<<1, vector_threads, 0, stream>>>(blocks, partial.get(), control.get());
        update_and_norm<<<blocks, vector_threads, 0, stream>>>(
            size, x.get(), r.get(), p.get(), ap.get(), control.get(), partial.get());
        finish_dot<Reduction::Residual>
            <<<1, vector_threads, 0, stream>>>(blocks, partial.get(), control.get());
        check(hipStreamEndCapture(stream, &graph));
        check(hipGraphInstantiate(&iteration, graph, nullptr, nullptr, 0));
    }
};

// Independent 2x2 Gauss assembly on a tiny mesh verifies all coefficients,
// including the signs of the ux-uy coupling.
void verify_stencil(double nu, Stencil stencil) {
    constexpr Grid grid{4};
    constexpr std::array<int, 4> dx{0, 1, 1, 0}, dy{0, 0, 1, 1};
    const double mu = 1.0 / (2.0 * (1.0 + nu));
    const double lambda = nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    const std::array<double, 2> gauss{0.5 - 0.5 / std::sqrt(3.0), 0.5 + 0.5 / std::sqrt(3.0)};
    std::vector<Vec> u(grid.size()), expected(grid.size()), actual(grid.size());
    for (int y = 1; y < grid.cells; ++y) {
        for (int x = 1; x < grid.cells; ++x) {
            u[grid.at(x, y)] = {std::sin(x + 2 * y), std::cos(2 * x + y)};
        }
    }
    for (int y = 0; y < grid.cells; ++y) {
        for (int x = 0; x < grid.cells; ++x) {
            for (double py : gauss) {
                for (double px : gauss) {
                    const std::array<double, 4> gx{py - 1, 1 - py, py, -py};
                    const std::array<double, 4> gy{px - 1, -px, px, 1 - px};
                    double ex = 0.0, ey = 0.0, shear = 0.0;
                    for (int a = 0; a < 4; ++a) {
                        const Vec value = u[grid.at(x + dx[a], y + dy[a])];
                        ex += gx[a] * value.x;
                        ey += gy[a] * value.y;
                        shear += gy[a] * value.x + gx[a] * value.y;
                    }
                    const double sx = (lambda + 2 * mu) * ex + lambda * ey;
                    const double sy = (lambda + 2 * mu) * ey + lambda * ex;
                    const double tau = mu * shear;
                    for (int a = 0; a < 4; ++a) {
                        Vec &out = expected[grid.at(x + dx[a], y + dy[a])];
                        out.x += 0.25 * (gx[a] * sx + gy[a] * tau);
                        out.y += 0.25 * (gy[a] * sy + gx[a] * tau);
                    }
                }
            }
        }
    }
    HostGrid{}(grid, Apply{grid, stencil, u.data(), actual.data()});
    for (int y = 1; y < grid.cells; ++y) {
        for (int x = 1; x < grid.cells; ++x) {
            const int i = grid.at(x, y);
            const Vec difference = expected[i] - actual[i];
            if (std::sqrt(difference.dot(difference)) > 1.e-12 * stencil.center) {
                throw std::runtime_error("matrix-free stencil failed independent Gauss assembly");
            }
        }
    }
}

// Changing manufactured displacements generate independent RHSs at a fixed
// stiffness. The true residual and error are evaluated outside solve timing.
void manufacture(Grid grid, Stencil stencil, int load, std::vector<Vec> &exact,
                 std::vector<Vec> &rhs) {
    constexpr double pi = 3.141592653589793;
    std::vector<Vec> xshape(grid.pitch()), yshape(grid.pitch());
    const double kx = 1 + load % 5;
    const double ky = 1 + (load / 5 + load) % 7;
    for (int i = 1; i < grid.cells; ++i) {
        const double t = i * pi / grid.cells;
        xshape[i] = {std::sin(kx * t), std::sin((ky + 1) * t)};
        yshape[i] = {std::sin(ky * t), std::sin((kx + 1) * t)};
    }
#pragma omp parallel for if (grid.cells >= 128) schedule(static)
    for (int y = 1; y < grid.cells; ++y) {
        for (int x = 1; x < grid.cells; ++x) {
            exact[grid.at(x, y)] = {xshape[x].x * yshape[y].x, -0.7 * xshape[x].y * yshape[y].y};
        }
    }
    HostGrid{}(grid, Apply{grid, stencil, exact.data(), rhs.data()});
}

template <class Solver> void run(std::string_view engine, Grid grid, Stencil stencil, int loads) {
    auto start = Clock::now();
    Solver solver(grid, stencil);
    const double setup = milliseconds(start);
    std::vector<Vec> exact(grid.size()), b(grid.size()), solution(grid.size()), work(grid.size());
    std::vector<double> times;
    std::printf("%.*s,setup,%d,-1,0,%.6f,0,0\n", int(engine.size()), engine.data(), grid.cells,
                setup);
    for (int load = 0; load < loads; ++load) {
        manufacture(grid, stencil, load, exact, b);
        const double norm2 = host_dot(grid.size(), b.data(), b.data());
        const auto stats = solver.solve(b, norm2, solution);
        HostGrid{}(grid, Residual{grid, stencil, b.data(), solution.data(), work.data()});
        const double relative_residual =
            std::sqrt(host_dot(grid.size(), work.data(), work.data()) / norm2);
        double squared_error = 0.0;
#pragma omp parallel for if (grid.size() >= 16'384) schedule(static) reduction(+ : squared_error)
        for (int i = 0; i < grid.size(); ++i) {
            const Vec error = solution[i] - exact[i];
            squared_error += error.dot(error);
        }
        const double relative_error =
            std::sqrt(squared_error / host_dot(grid.size(), exact.data(), exact.data()));
        if (!(relative_residual <= 5.e-8 && relative_error <= 5.e-8)) {
            throw std::runtime_error(std::string(engine) +
                                     " failed manufactured solution validation");
        }
        std::printf("%.*s,load,%d,%d,%d,%.6f,%.9g,%.9g\n", int(engine.size()), engine.data(),
                    grid.cells, load, stats.iterations, stats.milliseconds, relative_residual,
                    relative_error);
        std::fflush(stdout);
        times.push_back(stats.milliseconds);
    }
    double total = setup;
    for (double ms : times) {
        total += ms;
    }
    std::printf("%.*s,setup_plus_first,%d,0,0,%.6f,0,0\n", int(engine.size()), engine.data(),
                grid.cells, setup + times.front());
    std::printf("%.*s,setup_plus_all,%d,-1,0,%.6f,0,0\n", int(engine.size()), engine.data(),
                grid.cells, total);
}
} // namespace

int main(int argc, char **argv) {
    try {
        const int cells = argc > 1 ? std::stoi(argv[1]) : 512;
        const int loads = argc > 2 ? std::stoi(argv[2]) : 8;
        const int threads = argc > 3 ? std::stoi(argv[3]) : 8;
        const double nu = argc > 4 ? std::stod(argv[4]) : 0.30;
        const std::string_view engine = argc > 5 ? argv[5] : "both";
        if (argc > 6 || cells < 8 || cells > 4096 || (cells & (cells - 1)) || loads < 1 ||
            threads < 1 || !(nu >= 0.0 && nu < 0.5) ||
            (engine != "both" && engine != "cpu" && engine != "gpu")) {
            throw std::runtime_error("usage: plane_strain [power-of-two cells 8..4096] [loads] "
                                     "[CPU threads] [Poisson ratio in [0, 0.5)] [both|cpu|gpu]");
        }
        omp_set_dynamic(0);
        omp_set_num_threads(threads);
        const Grid grid{cells};
        const Stencil stencil(nu);
        verify_stencil(nu, stencil);
        std::puts("engine,kind,cells,load,iterations,ms,relative_residual,relative_solution_error");
        if (engine != "gpu") {
            run<CpuSolver>("cpu", grid, stencil, loads);
        }
        if (engine != "cpu") {
            check(hipSetDevice(0));
            hipDeviceProp_t info;
            check(hipGetDeviceProperties(&info, 0));
            std::fprintf(stderr, "GPU %s (%s)\n", info.name, info.gcnArchName);
            run<DeviceSolver>("gpu", grid, stencil, loads);
        }
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
