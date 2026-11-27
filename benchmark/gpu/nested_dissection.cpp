// Optional ordering experiment. SuiteSparse owns only its temporary CSC copy.
#include <cholmod.h>

#include <cstdint>
#include <cstring>

extern "C" int strut_nested_dissection(std::int64_t n, std::int64_t nnz,
                                       const std::int64_t *columns, const std::int64_t *rows,
                                       std::int64_t *permutation) {
    cholmod_common common;
    if (!cholmod_l_start(&common))
        return 1;
    common.nmethods = 1;
    common.method[0].ordering = CHOLMOD_NESDIS;
    // Strut postorders its own elimination tree after applying this permutation.
    common.postorder = 0;
    common.supernodal = CHOLMOD_SIMPLICIAL;
    auto *matrix = cholmod_l_allocate_sparse(n, n, nnz, 1, 1, -1, CHOLMOD_PATTERN, &common);
    if (!matrix) {
        cholmod_l_finish(&common);
        return 1;
    }
    std::memcpy(matrix->p, columns, (n + 1) * sizeof(*columns));
    std::memcpy(matrix->i, rows, nnz * sizeof(*rows));
    auto *factor = cholmod_l_analyze(matrix, &common);
    if (factor)
        std::memcpy(permutation, factor->Perm, n * sizeof(*permutation));
    int result = factor ? 0 : 1;
    cholmod_l_free_factor(&factor, &common);
    cholmod_l_free_sparse(&matrix, &common);
    cholmod_l_finish(&common);
    return result;
}
