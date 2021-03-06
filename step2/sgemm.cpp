#include "../sgemm.h"

#include <immintrin.h>

void kernel_8x8(int k, const float *a, int lda, const float *b, int ldb, float *c, int ldc);

void my_sgemm(int m, int n, int k, const float *a, int lda, const float *b, int ldb, float *c, int ldc) {
    for (int j = 0; j < n; j += 8) {
        for (int i = 0; i < m; i += 8) {
            kernel_8x8(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
        }
    }
}

void kernel_8x8(int k, const float *a, int lda, const float *b, int ldb, float *c, int ldc) {
    __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;  // for c
    __m256 ymm8;                                            // for a
    __m256 ymm9;                                            // for b

    // kernel
    ymm0 = _mm256_loadu_ps(&C(0, 0));
    ymm1 = _mm256_loadu_ps(&C(0, 1));
    ymm2 = _mm256_loadu_ps(&C(0, 2));
    ymm3 = _mm256_loadu_ps(&C(0, 3));
    ymm4 = _mm256_loadu_ps(&C(0, 4));
    ymm5 = _mm256_loadu_ps(&C(0, 5));
    ymm6 = _mm256_loadu_ps(&C(0, 6));
    ymm7 = _mm256_loadu_ps(&C(0, 7));

#define FMA(x)                            \
    ymm8 = _mm256_broadcast_ss(&B(p, x)); \
    ymm##x = _mm256_fmadd_ps(ymm8, ymm9, ymm##x);

    // dot 8 x 8
    for (int p = 0; p < k; p++) {
        ymm9 = _mm256_loadu_ps(&A(0, p));
        FMA(0)
        FMA(1)
        FMA(2)
        FMA(3)
        FMA(4)
        FMA(5)
        FMA(6)
        FMA(7)
    }

    _mm256_storeu_ps(&C(0, 0), ymm0);
    _mm256_storeu_ps(&C(0, 1), ymm1);
    _mm256_storeu_ps(&C(0, 2), ymm2);
    _mm256_storeu_ps(&C(0, 3), ymm3);
    _mm256_storeu_ps(&C(0, 4), ymm4);
    _mm256_storeu_ps(&C(0, 5), ymm5);
    _mm256_storeu_ps(&C(0, 6), ymm6);
    _mm256_storeu_ps(&C(0, 7), ymm7);
}
