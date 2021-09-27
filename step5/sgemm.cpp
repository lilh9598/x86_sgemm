#include "../sgemm.h"

#include <immintrin.h>

void kernel_8x8(int k, const float *packa, int lda, const float *b, int ldb, float *c, int ldc);

void mcxkc_sgemm(int m, int n, int k, const float *packa, int lda, const float *packb, int ldb, float *c, int ldc);

void packB_Kcxn(const float *b, int ldb, int k, float *packb);

void packA_KcxMc(const float *a, int lda, int k, float *packa);

void my_sgemm(int m, int n, int k, const float *a, int lda, const float *b, int ldb, float *c, int ldc) {
    float *packb = new float[n * Kc];
    float *packa = new float[Mc * Kc];

    for (int p = 0; p < k; p += Kc) {
        int pb = min(Kc, k - p);

        for (int j = 0; j < n; j += 8) {
            packB_Kcxn(&B(p, j), ldb, pb, packb + j * pb);
        }

        for (int i = 0; i < m; i += Mc) {
            int ib = min(Mc, m - i);

            for (int ii = 0; ii < ib; ii += 8) {
                packA_KcxMc(&A(ii + i, p), lda, pb, packa + ii * pb);
            }

            mcxkc_sgemm(ib, n, pb, packa, lda, packb, ldb, &C(i, 0), ldc);
        }
    }

    delete[] packb;
    delete[] packa;
}

void packA_KcxMc(const float *a, int lda, int k, float *packa) {
    const float *pack_ptr[8];
    for (int i = 0; i < 8; i++) {
        pack_ptr[i] = a + i * lda;
    }

    for (int p = 0; p < k; p++) {
        for (int i = 0; i < 8; i++) {
            *packa++ = *pack_ptr[i]++;
        }
    }
}

void packB_Kcxn(const float *b, int ldb, int k, float *packb) {
    for (int p = 0; p < k; p++) {
        auto b_ptr = b + p * ldb;
        for (int i = 0; i < 8; i++) {
            *packb++ = *b_ptr++;
        }
    }
}

void mcxkc_sgemm(int m, int n, int k, const float *packa, int lda, const float *packb, int ldb, float *c, int ldc) {
    for (int j = 0; j < n; j += 8) {
        for (int i = 0; i < m; i += 8) {
            kernel_8x8(k, packa + i * k, lda, packb + j * k, ldb, &C(i, j), ldc);
        }
    }
}

void kernel_8x8(int k, const float *packa, int lda, const float *packb, int ldb, float *c, int ldc) {
    __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;  // for c
    __m256 ymm8;                                            // for a
    __m256 ymm9;                                            // for b

    // kernel
    ymm0 = _mm256_loadu_ps(&C(0, 0));
    ymm1 = _mm256_loadu_ps(&C(1, 0));
    ymm2 = _mm256_loadu_ps(&C(2, 0));
    ymm3 = _mm256_loadu_ps(&C(3, 0));
    ymm4 = _mm256_loadu_ps(&C(4, 0));
    ymm5 = _mm256_loadu_ps(&C(5, 0));
    ymm6 = _mm256_loadu_ps(&C(6, 0));
    ymm7 = _mm256_loadu_ps(&C(7, 0));

    auto a_ptr = packa;
#define FMA(x)                             \
    ymm8 = _mm256_broadcast_ss(a_ptr + x); \
    ymm##x = _mm256_fmadd_ps(ymm8, ymm9, ymm##x);

    // dot 8 x 8
    for (int p = 0; p < k; p++) {
        ymm9 = _mm256_loadu_ps(packb);
        FMA(0)
        FMA(1)
        FMA(2)
        FMA(3)
        FMA(4)
        FMA(5)
        FMA(6)
        FMA(7)

        a_ptr += 8;
        packb += 8;
    }

    _mm256_storeu_ps(&C(0, 0), ymm0);
    _mm256_storeu_ps(&C(1, 0), ymm1);
    _mm256_storeu_ps(&C(2, 0), ymm2);
    _mm256_storeu_ps(&C(3, 0), ymm3);
    _mm256_storeu_ps(&C(4, 0), ymm4);
    _mm256_storeu_ps(&C(5, 0), ymm5);
    _mm256_storeu_ps(&C(6, 0), ymm6);
    _mm256_storeu_ps(&C(7, 0), ymm7);
}
