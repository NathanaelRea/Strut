// Factor and solve Strut's captured symmetric tangents with STRUMPACK LU.
#include <StrumpackSparseSolver.hpp>

#include <algorithm>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using Solver = strumpack::SparseSolver<double, int>;
constexpr double max_relative_inf_error = 1e-7;
constexpr double max_backward_error = 1e-8;

double elapsed(Clock::time_point started) {
    return std::chrono::duration<double>(Clock::now() - started).count();
}

template <typename T> void read_exact(std::istream &stream, T *data, std::size_t count) {
    if (!stream.read(reinterpret_cast<char *>(data), count * sizeof(T)))
        throw std::runtime_error("truncated matrix trace");
}

std::uint64_t read_uint(std::istream &stream) {
    std::uint64_t number;
    read_exact(stream, &number, 1);
    return number;
}

void check(strumpack::ReturnCode code, std::string_view operation) {
    if (code == strumpack::ReturnCode::SUCCESS)
        return;
    std::cerr << operation << ": " << code << '\n';
    throw std::runtime_error("STRUMPACK operation failed");
}

struct SymmetricMatrix {
    int size = 0;
    std::vector<int> row_ptr;
    std::vector<int> columns;
    std::vector<int> source;
    std::vector<double> lower_values;
    std::vector<double> values;

    void read_pattern(std::istream &stream) {
        const auto n = read_uint(stream), nnz = read_uint(stream);
        const auto max_index = std::numeric_limits<int>::max();
        if (n == 0 || n >= max_index || nnz < n || nnz > (std::uint64_t(max_index) + n) / 2)
            throw std::runtime_error("matrix trace exceeds the CSR index limits");

        size = static_cast<int>(n);
        std::vector<std::int64_t> ptr(n + 1), rows(nnz);
        read_exact(stream, ptr.data(), ptr.size());
        read_exact(stream, rows.data(), rows.size());
        if (ptr.front() != 0 || ptr.back() != static_cast<std::int64_t>(nnz) ||
            !std::is_sorted(ptr.begin(), ptr.end()))
            throw std::runtime_error("invalid CSC column pointers");

        row_ptr.assign(n + 1, 0);
        for (int col = 0; col < size; ++col) {
            int previous_row = col - 1;
            if (ptr[col] == ptr[col + 1] || rows[ptr[col]] != col)
                throw std::runtime_error("CSC column has no diagonal");
            for (auto p = ptr[col]; p < ptr[col + 1]; ++p) {
                const auto row = rows[p];
                if (row <= previous_row || row >= size)
                    throw std::runtime_error("invalid lower CSC row index");
                previous_row = static_cast<int>(row);
                ++row_ptr[row + 1];
                if (row != col)
                    ++row_ptr[col + 1];
            }
        }

        std::partial_sum(row_ptr.begin(), row_ptr.end(), row_ptr.begin());
        columns.resize(row_ptr.back());
        source.resize(row_ptr.back());
        auto next = row_ptr;
        for (int col = 0; col < size; ++col) {
            for (auto p = ptr[col]; p < ptr[col + 1]; ++p) {
                const auto row = rows[p];
                const auto lower = next[row]++;
                columns[lower] = col;
                source[lower] = static_cast<int>(p);
                if (row != col) {
                    const auto upper = next[col]++;
                    columns[upper] = static_cast<int>(row);
                    source[upper] = static_cast<int>(p);
                }
            }
        }
        lower_values.resize(nnz);
        values.resize(source.size());
    }

    void read_values(std::istream &stream) {
        read_exact(stream, lower_values.data(), lower_values.size());
        for (std::size_t i = 0; i < source.size(); ++i)
            values[i] = lower_values[source[i]];
    }

    double max_scaled_residual(const std::vector<double> &result,
                               const std::vector<double> &rhs) const {
        double maximum = 0.0;
        for (int row = 0; row < size; ++row) {
            double residual = -rhs[row];
            double scale = std::abs(rhs[row]);
            for (int p = row_ptr[row]; p < row_ptr[row + 1]; ++p) {
                const double product = values[p] * result[columns[p]];
                residual += product;
                scale += std::abs(product);
            }
            if (!std::isfinite(residual) || !std::isfinite(scale))
                return std::numeric_limits<double>::infinity();
            if (scale > 0.0)
                maximum = std::max(maximum, std::abs(residual) / scale);
        }
        return maximum;
    }
};

double relative_error(const std::vector<double> &result, const std::vector<double> &reference) {
    double error = 0.0, scale = 0.0;
    for (std::size_t i = 0; i < result.size(); ++i) {
        if (!std::isfinite(result[i]) || !std::isfinite(reference[i]))
            return std::numeric_limits<double>::infinity();
        error = std::max(error, std::abs(result[i] - reference[i]));
        scale = std::max(scale, std::abs(reference[i]));
    }
    return scale > 0.0 ? error / scale : error;
}

int replay(std::istream &trace, bool gpu, int factor_limit) {
    SymmetricMatrix matrix;
    std::unique_ptr<Solver> solver;
    std::vector<double> rhs, reference, result;
    int stage = 0, factors = 0, solves = 0;
    double max_error = 0.0, max_residual = 0.0;
    double reorder_seconds = 0.0, factor_seconds = 0.0, solve_seconds = 0.0, update_seconds = 0.0;
    std::cout << "record,stage,factors,solves,seconds,max_scaled_residual,relative_inf_error\n"
              << std::setprecision(12);
    while (trace.peek() != EOF) {
        const auto record = read_uint(trace);
        if (record != 3 && factor_limit > 0 && factors >= factor_limit)
            break;
        switch (record) {
        case 1:
            solver.reset();
            matrix.read_pattern(trace);
            rhs.resize(matrix.size);
            reference.resize(matrix.size);
            result.resize(matrix.size);
            ++stage;
            break;
        case 2: {
            if (matrix.size == 0)
                throw std::runtime_error("matrix values precede the CSC pattern");
            matrix.read_values(trace);
            const bool first_factor = !solver;
            const auto update_start = Clock::now();
            if (solver) {
                solver->update_matrix_values(matrix.size, matrix.row_ptr.data(),
                                             matrix.columns.data(), matrix.values.data(), true);
                update_seconds += elapsed(update_start);
            } else {
                solver = std::make_unique<Solver>(false);
                auto &options = solver->options();
                options.set_matching(strumpack::MatchingJob::NONE);
                options.set_reordering_method(strumpack::ReorderingStrategy::METIS);
                options.set_Krylov_solver(strumpack::KrylovSolver::DIRECT);
                options.set_compression(strumpack::CompressionType::NONE);
                if (gpu)
                    options.enable_gpu();
                else
                    options.disable_gpu();
                solver->set_csr_matrix(matrix.size, matrix.row_ptr.data(), matrix.columns.data(),
                                       matrix.values.data(), true);
                check(solver->reorder(), "reorder");
                const auto duration = elapsed(update_start);
                reorder_seconds += duration;
                std::cout << "reorder," << stage << ',' << factors << ',' << solves << ','
                          << duration << ",,\n";
            }
            const auto factor_start = Clock::now();
            check(solver->factor(), "factor");
            const auto duration = elapsed(factor_start);
            factor_seconds += duration;
            if (first_factor)
                std::cerr << "stage " << stage << " full CSR entries " << matrix.values.size()
                          << ", LU factor entries " << solver->factor_nonzeros() << '\n';
            std::cout << "factor," << stage << ',' << ++factors << ',' << solves << ',' << duration
                      << ",,\n";
            break;
        }
        case 3: {
            if (!solver)
                throw std::runtime_error("RHS precedes matrix values");
            read_exact(trace, rhs.data(), rhs.size());
            read_exact(trace, reference.data(), reference.size());
            const auto solve_start = Clock::now();
            check(solver->solve(rhs.data(), result.data()), "solve");
            const auto duration = elapsed(solve_start);
            solve_seconds += duration;
            const auto residual = matrix.max_scaled_residual(result, rhs);
            const auto error = relative_error(result, reference);
            max_residual = std::max(max_residual, residual);
            max_error = std::max(max_error, error);
            std::cout << "solve," << stage << ',' << factors << ',' << ++solves << ',' << duration
                      << ',' << residual << ',' << error << '\n';
            break;
        }
        default:
            throw std::runtime_error("unknown matrix trace record");
        }
    }
    std::cout << "total_reorder," << stage << ',' << factors << ',' << solves << ','
              << reorder_seconds << ",,\n"
              << "total_update," << stage << ',' << factors << ',' << solves << ','
              << update_seconds << ",,\n"
              << "total_factor," << stage << ',' << factors << ',' << solves << ','
              << factor_seconds << ",,\n"
              << "total_solve," << stage << ',' << factors << ',' << solves << ',' << solve_seconds
              << ',' << max_residual << ',' << max_error << '\n';
    if (solves == 0 || max_error > max_relative_inf_error || max_residual > max_backward_error)
        throw std::runtime_error("replay failed numerical verification");
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    static_assert(std::endian::native == std::endian::little);
    const std::string_view mode = argc > 1 ? argv[1] : "";
    if (argc < 3 || argc > 4 || (mode != "cpu" && mode != "gpu")) {
        std::cerr << "usage: replay (cpu|gpu) TRACE [MAX_FACTORS]\n";
        return 2;
    }
    try {
        int limit = 0;
        if (argc == 4) {
            const auto text = std::string_view(argv[3]);
            const auto read = std::from_chars(text.data(), text.data() + text.size(), limit);
            if (read.ec != std::errc{} || read.ptr != text.data() + text.size() || limit < 1)
                throw std::runtime_error("MAX_FACTORS must be a positive integer");
        }
#ifndef STRUMPACK_USE_HIP
        if (mode == "gpu")
            throw std::runtime_error("STRUMPACK was built without HIP");
#endif
        std::ifstream trace(argv[2], std::ios::binary);
        if (!trace)
            throw std::runtime_error("could not open matrix trace");
        return replay(trace, mode == "gpu", limit);
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
