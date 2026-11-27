// Standalone plane-strain material-point benchmark. Tensile stress is positive.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if __has_include(<omp.h>)
#include <omp.h>
#else
extern "C" void omp_set_dynamic(int);
extern "C" void omp_set_num_threads(int);
#endif

namespace {
using Clock = std::chrono::steady_clock;

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void check(hipError_t status) {
    if (status != hipSuccess) {
        throw std::runtime_error(hipGetErrorString(status));
    }
}

struct Material {
    double bulk = 10.0;
    double shear = 6.0;
    double yield_slope = 0.7;
    double flow_slope = 0.27;
    double cohesion = 0.09;
    double hardening = 0.4;
};

struct Strain {
    double xx, yy, xy;
};

struct State {
    // p = tr(stress)/3. The deviator's zz component is -sxx - syy.
    double p, sxx, syy, sxy, kappa;
};

struct Fields {
    double *p, *sxx, *syy, *sxy, *kappa;

    __host__ __device__ State get(int i) const { return {p[i], sxx[i], syy[i], sxy[i], kappa[i]}; }

    __host__ __device__ void set(int i, State state) const {
        p[i] = state.p;
        sxx[i] = state.sxx;
        syy[i] = state.syy;
        sxy[i] = state.sxy;
        kappa[i] = state.kappa;
    }
};

enum class Return { Elastic, Cone, Apex };

__host__ __device__ State predictor(Material material, State state, Strain increment,
                                    double scale) {
    const double xx = scale * increment.xx;
    const double yy = scale * increment.yy;
    const double mean = (xx + yy) / 3.0;
    return {
        state.p + material.bulk * (xx + yy),
        state.sxx + 2.0 * material.shear * (xx - mean),
        state.syy + 2.0 * material.shear * (yy - mean),
        state.sxy + 2.0 * material.shear * (scale * increment.xy),
        state.kappa,
    };
}

__host__ __device__ double equivalent_stress(State state) {
    const double szz = -state.sxx - state.syy;
    return sqrt(1.5 * (state.sxx * state.sxx + state.syy * state.syy + szz * szz +
                       2.0 * state.sxy * state.sxy));
}

// Backward Euler with f = q + a*p - (c + H*kappa) and g = q + b*p.
// For linear hardening the scalar return equation is piecewise linear.
__host__ __device__ Return correct(Material material, State &trial) {
    const double q = equivalent_stress(trial);
    const double pressure_margin =
        material.yield_slope * trial.p - material.cohesion - material.hardening * trial.kappa;
    if (q + pressure_margin <= 0.0) {
        return Return::Elastic;
    }
    const double deviator_slope = 3.0 * material.shear;
    const double pressure_slope =
        material.yield_slope * material.bulk * material.flow_slope + material.hardening;
    double multiplier = (q + pressure_margin) / (deviator_slope + pressure_slope);
    double scale = 0.0;
    const bool apex = multiplier * deviator_slope >= q;
    if (apex) {
        multiplier = pressure_margin / pressure_slope;
    } else {
        scale = 1.0 - multiplier * deviator_slope / q;
    }
    trial.p -= multiplier * material.bulk * material.flow_slope;
    trial.sxx *= scale;
    trial.syy *= scale;
    trial.sxy *= scale;
    trial.kappa += multiplier;
    return apex ? Return::Apex : Return::Cone;
}

template <bool plastic>
__global__ void step(Material material, int points, Fields fields, const double *scale,
                     Strain increment) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= points) {
        return;
    }
    State next = predictor(material, fields.get(i), increment, scale[i]);
    if constexpr (plastic) {
        correct(material, next);
    }
    fields.set(i, next);
}

bool near(State first, State second) {
    const std::array<double, 5> a{first.p, first.sxx, first.syy, first.sxy, first.kappa};
    const std::array<double, 5> b{second.p, second.sxx, second.syy, second.sxy, second.kappa};
    for (size_t i = 0; i < a.size(); ++i) {
        if (!(std::abs(a[i] - b[i]) <= 5.e-13 * std::max(1.0, std::abs(a[i]))) ||
            !std::isfinite(a[i]) || !std::isfinite(b[i])) {
            return false;
        }
    }
    return true;
}

// Independently solve the scalar yield condition by bisection, then compare the
// two corrected states. Include direct checks for elasticity, flank and apex.
State bisect(Material material, State trial) {
    const double q = equivalent_stress(trial);
    const auto yield = [&](double multiplier) {
        return std::max(q - 3.0 * material.shear * multiplier, 0.0) +
               material.yield_slope * (trial.p - material.bulk * material.flow_slope * multiplier) -
               material.cohesion - material.hardening * (trial.kappa + multiplier);
    };
    if (yield(0) <= 0) {
        return trial;
    }
    double low = 0.0, high = 1.0;
    while (yield(high) > 0) {
        high *= 2.0;
    }
    for (int i = 0; i < 80; ++i) {
        const double mid = (low + high) / 2.0;
        if (yield(mid) > 0) {
            low = mid;
        } else {
            high = mid;
        }
    }
    const double multiplier = (low + high) / 2.0;
    const double scale = q > 0 ? std::max(q - 3.0 * material.shear * multiplier, 0.0) / q : 0.0;
    return {trial.p - material.bulk * material.flow_slope * multiplier, scale * trial.sxx,
            scale * trial.syy, scale * trial.sxy, trial.kappa + multiplier};
}

void verify(Material material) {
    const std::array<State, 4> samples{
        State{-0.2, 0.01, 0.01, 0.00, 0.00},
        State{-0.2, 0.07, 0.03, 0.30, 0.10},
        State{1.00, 0.00, 0.00, 0.00, 0.03},
        State{1.00, 0.01, -0.01, 0.00, 0.03},
    };
    const std::array<Return, 4> expected{Return::Elastic, Return::Cone, Return::Apex, Return::Apex};
    for (int i = 0; i < int(samples.size()); ++i) {
        State actual = samples[i];
        if (correct(material, actual) != expected[i] ||
            !near(actual, bisect(material, samples[i]))) {
            throw std::runtime_error("Drucker–Prager return mapping check failed");
        }
    }
}

Fields fields(double *data, int points) {
    return {data, data + points, data + 2 * points, data + 3 * points, data + 4 * points};
}

template <class T> class Device {
  public:
    explicit Device(size_t size) { check(hipMalloc(&data, size * sizeof(T))); }
    Device(const Device &) = delete;
    Device &operator=(const Device &) = delete;
    ~Device() { (void)hipFree(data); }

    T *data;
};

double host_history(bool plastic, Material material, const std::vector<Strain> &history,
                    const std::vector<double> &scale, std::vector<double> &state) {
    const int points = scale.size();
    const Fields f = fields(state.data(), points);
    const auto start = Clock::now();
    if (plastic) {
        for (Strain increment : history) {
#pragma omp parallel for if (points >= 16384) schedule(static)
            for (int i = 0; i < points; ++i) {
                State next = predictor(material, f.get(i), increment, scale[i]);
                correct(material, next);
                f.set(i, next);
            }
        }
    } else {
        for (Strain increment : history) {
#pragma omp parallel for if (points >= 16384) schedule(static)
            for (int i = 0; i < points; ++i) {
                f.set(i, predictor(material, f.get(i), increment, scale[i]));
            }
        }
    }
    return elapsed(start);
}

constexpr int scale_period = 1021;

std::pair<long long, long long> verify_history(Material material,
                                               const std::vector<Strain> &history,
                                               const std::vector<double> &scale, Fields solution) {
    const int points = scale.size();
    long long cone = 0, apex = 0;
    for (int i = 0; i < std::min(points, scale_period); ++i) {
        const int copies = 1 + (points - 1 - i) / scale_period;
        State state{};
        for (Strain increment : history) {
            state = predictor(material, state, increment, scale[i]);
            const State expected = bisect(material, state);
            const Return branch = correct(material, state);
            if (!near(expected, state)) {
                throw std::runtime_error("Drucker–Prager cyclic return check failed");
            }
            cone += copies * (branch == Return::Cone);
            apex += copies * (branch == Return::Apex);
        }
        if (!near(solution.get(i), state)) {
            throw std::runtime_error("Drucker–Prager reference history mismatch");
        }
    }
    return {cone, apex};
}

double gpu_history(bool plastic, Material material, const std::vector<Strain> &history,
                   const std::vector<double> &scale, std::vector<double> &state) {
    const int points = scale.size();
    const size_t bytes = state.size() * sizeof(double);
    Device<double> d_state(state.size()), d_scale(points);
    check(hipMemcpy(d_scale.data, scale.data(), points * sizeof(double), hipMemcpyHostToDevice));
    check(hipMemset(d_state.data, 0, bytes));
    check(hipDeviceSynchronize());
    const Fields f = fields(d_state.data, points);
    const auto start = Clock::now();
    constexpr int threads = 256;
    const int blocks = (points + threads - 1) / threads;
    if (plastic) {
        for (Strain increment : history) {
            step<true><<<blocks, threads>>>(material, points, f, d_scale.data, increment);
        }
    } else {
        for (Strain increment : history) {
            step<false><<<blocks, threads>>>(material, points, f, d_scale.data, increment);
        }
    }
    check(hipGetLastError());
    check(hipDeviceSynchronize());
    const double result = elapsed(start);
    check(hipMemcpy(state.data(), d_state.data, bytes, hipMemcpyDeviceToHost));
    return result;
}
} // namespace

int main(int argc, char **argv) {
    try {
        const int points = argc > 1 ? std::stoi(argv[1]) : 1'048'576;
        const int increments = argc > 2 ? std::stoi(argv[2]) : 40;
        const int threads = argc > 3 ? std::stoi(argv[3]) : 8;
        if (argc > 4 || points < 1 || points > 16'777'216 || increments < 1 || threads < 1) {
            throw std::runtime_error(
                "usage: drucker_prager [points <= 16777216] [increments] [CPU threads]");
        }
        omp_set_dynamic(0);
        omp_set_num_threads(threads);
        check(hipSetDevice(0));
        const Material material;
        verify(material);

        // Mixed loading, unloading, shear and tension produce different yield
        // histories across points. Values are dimensionless and uncalibrated.
        constexpr std::array<Strain, 8> cycle{
            Strain{-0.007, -0.005, 0.001}, Strain{0.000, -0.003, 0.020},
            Strain{0.000, 0.000, 0.015},   Strain{0.004, 0.005, -0.015},
            Strain{0.010, 0.009, 0.000},   Strain{0.009, 0.009, 0.000},
            Strain{-0.008, -0.012, 0.000}, Strain{0.001, -0.002, -0.018},
        };
        std::vector<Strain> history(increments);
        for (int i = 0; i < increments; ++i) {
            history[i] = cycle[i % cycle.size()];
        }
        std::vector<double> scale(points);
        for (int i = 0; i < points; ++i) {
            scale[i] = 0.05 + 1.95 * ((i % scale_period * 463) % scale_period) / (scale_period - 1);
        }
        std::printf(
            "model,points,increments,cpu_ms,gpu_ms,cone_updates,apex_updates,max_difference\n");
        for (bool plastic : {false, true}) {
            std::vector<double> cpu(5 * size_t(points)), gpu(5 * size_t(points));
            const double cpu_ms = host_history(plastic, material, history, scale, cpu);
            const double gpu_ms = gpu_history(plastic, material, history, scale, gpu);
            const Fields c = fields(cpu.data(), points), g = fields(gpu.data(), points);
            double max_difference = 0.0;
            for (int i = 0; i < points; ++i) {
                const State expected = c.get(i), actual = g.get(i);
                if (!near(expected, actual)) {
                    throw std::runtime_error("Drucker–Prager CPU/GPU state mismatch");
                }
            }
            for (size_t i = 0; i < cpu.size(); ++i) {
                max_difference = std::max(max_difference, std::abs(cpu[i] - gpu[i]));
            }
            const auto [cone, apex] = plastic ? verify_history(material, history, scale, c)
                                              : std::pair<long long, long long>{};
            std::printf("%s,%d,%d,%.6f,%.6f,%lld,%lld,%.9g\n",
                        plastic ? "drucker_prager" : "elastic", points, increments, cpu_ms, gpu_ms,
                        cone, apex, max_difference);
        }
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
