#include "../sgemm.h"

#include <immintrin.h>

void kernel_8x8(int k, const float *packa, int lda, const float *b, int ldb, float *c, int ldc);

void mcxkc_sgemm(int m, int n, int k, const float *a, int lda, const float *packb, int ldb, float *c, int ldc);

void packB_Kcxn(const float *b, int ldb, int k, int nr, float *packb);

void my_sgemm(int m, int n, int k, const float *a, int lda, const float *b, int ldb, float *c, int ldc) {
    float *packb = new float[ROUND_UP(n, 8) * Kc];

    for (int p = 0; p < k; p += Kc) {
        int pb = min(Kc, k - p);

        for (int j = 0; j < n; j += 8) {
            packB_Kcxn(&B(p, j), ldb, pb, min(n - j, 8), packb + j * pb);
        }

        for (int i = 0; i < m; i += Mc) {
            int ib = min(Mc, m - i);

            mcxkc_sgemm(ib, n, pb, &A(i, p), lda, packb, ldb, &C(i, 0), ldc);
        }
    }

    delete[] packb;
}

void packB_Kcxn(const float *b, int ldb, int k, int nr, float *packb) {
    const float *pack_ptr[8];
    for (int i = 0; i < nr; i++) {
        pack_ptr[i] = b + i * ldb;
    }
    for (int i = nr; i < 8; i++) {
        pack_ptr[i] = b;
    }
    for (int p = 0; p < k; p++) {
        for (int i = 0; i < 8; i++) {
            *packb++ = *pack_ptr[i]++;
        }
    }
}

void mcxkc_sgemm(int m, int n, int k, const float *a, int lda, const float *packb, int ldb, float *c, int ldc) {
    for (int j = 0; j < n; j += 8) {
        for (int i = 0; i < m; i += 8) {
            kernel_8x8(k, &A(i, 0), lda, packb + j * k, ldb, &C(i, j), ldc);
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

#define FMA(x)                         \
    ymm8 = _mm256_broadcast_ss(b + x); \
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
        b += 8;
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
