#include "../sgemm.h"

#include <immintrin.h>

void kernel_16x6(int k, const float *packa, const float *b, float *c, int ldc);

void mcxkc_sgemm(int m, int n, int k, const float *packa, const float *packb, float *c, int ldc);

void packB_Kcxn(const float *b, int ldb, int k, int nr, float *packb);

void packA_KcxMc(const float *a, int lda, int k, int mr, float *packa);

void my_sgemm(int m, int n, int k, const float *a, int lda, const float *b, int ldb, float *c, int ldc) {
    float *packb = new float[ROUND_UP(n, NR) * Kc];
    float *packa = new float[ROUND_UP(Mc, MR) * Kc];
    for (int p = 0; p < k; p += Kc) {
        int pb = min(Kc, k - p);

        for (int j = 0; j < n; j += NR) {
            packB_Kcxn(&B(p, j), ldb, pb, min(n - j, NR), packb + j * pb);
        }

        for (int i = 0; i < m; i += Mc) {
            int ib = min(Mc, m - i);

            for (int ii = 0; ii < ib; ii += MR) {
                packA_KcxMc(&A(ii + i, p), lda, pb, min(ib - ii, MR), packa + ii * pb);
            }

            mcxkc_sgemm(ib, n, pb, packa, packb, &C(i, 0), ldc);
        }
    }

    delete []packb;
    delete []packa;
}
void packA_KcxMc(const float *a, int lda, int k, int mr, float *packa) {
    const float *pack_ptr[MR];
    for (int i = 0; i < mr; i++) {
        pack_ptr[i] = a + i;
    }

    for (int i = mr; i < MR; i++) {
        pack_ptr[i] = a;
    }

    for (int p = 0; p < k; p++) {
        for (int i = 0; i < MR; i++) {
            *packa++ = *pack_ptr[i];
            pack_ptr[i] += lda;
        }
    }
}

void packB_Kcxn(const float *b, int ldb, int k, int nr, float *packb) {
    const float *pack_ptr[NR];
    for (int i = 0; i < nr; i++) {
        pack_ptr[i] = b + i * ldb;
    }
    for (int i = nr; i < NR; i++) {
        pack_ptr[i] = b;
    }
    for (int p = 0; p < k; p++) {
        for (int i = 0; i < NR; i++) {
            *packb++ = *pack_ptr[i]++;
        }
    }
}

void mcxkc_sgemm(int m, int n, int k, const float *packa, const float *packb, float *c, int ldc) {
    for (int j = 0; j < n; j += NR) {
        for (int i = 0; i < m; i += MR) {
            kernel_16x6(k, packa + i * k, packb + j * k, &C(i, j), ldc);
        }
    }
}
void kernel_16x6(int k, const float *packa, const float *packb, float *c, int ldc) {
    __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5;    // for c
    __m256 ymm6, ymm7, ymm8, ymm9, ymm10, ymm11;  // for c
    __m256 ymm12, ymm13;                          // for a
    __m256 ymm15;                                 // for b

    ymm0 = _mm256_loadu_ps(&C(0, 0));
    ymm1 = _mm256_loadu_ps(&C(8, 0));

    ymm2 = _mm256_loadu_ps(&C(0, 1));
    ymm3 = _mm256_loadu_ps(&C(8, 1));

    ymm4 = _mm256_loadu_ps(&C(0, 2));
    ymm5 = _mm256_loadu_ps(&C(8, 2));

    ymm6 = _mm256_loadu_ps(&C(0, 3));
    ymm7 = _mm256_loadu_ps(&C(8, 3));

    ymm8 = _mm256_loadu_ps(&C(0, 4));
    ymm9 = _mm256_loadu_ps(&C(8, 4));

    ymm10 = _mm256_loadu_ps(&C(0, 5));
    ymm11 = _mm256_loadu_ps(&C(8, 5));

    for (int p = 0; p < k; p += 1) {
        ymm12 = _mm256_loadu_ps(packa);
        ymm13 = _mm256_loadu_ps(packa + 8);

        ymm15 = _mm256_broadcast_ss(packb + 0);
        ymm0 = _mm256_fmadd_ps(ymm12, ymm15, ymm0);
        ymm1 = _mm256_fmadd_ps(ymm13, ymm15, ymm1);

        ymm15 = _mm256_broadcast_ss(packb + 1);
        ymm2 = _mm256_fmadd_ps(ymm12, ymm15, ymm2);
        ymm3 = _mm256_fmadd_ps(ymm13, ymm15, ymm3);

        ymm15 = _mm256_broadcast_ss(packb + 2);
        ymm4 = _mm256_fmadd_ps(ymm12, ymm15, ymm4);
        ymm5 = _mm256_fmadd_ps(ymm13, ymm15, ymm5);

        ymm15 = _mm256_broadcast_ss(packb + 3);
        ymm6 = _mm256_fmadd_ps(ymm12, ymm15, ymm6);
        ymm7 = _mm256_fmadd_ps(ymm13, ymm15, ymm7);

        ymm15 = _mm256_broadcast_ss(packb + 4);
        ymm8 = _mm256_fmadd_ps(ymm12, ymm15, ymm8);
        ymm9 = _mm256_fmadd_ps(ymm13, ymm15, ymm9);

        ymm15 = _mm256_broadcast_ss(packb + 5);
        ymm10 = _mm256_fmadd_ps(ymm12, ymm15, ymm10);
        ymm11 = _mm256_fmadd_ps(ymm13, ymm15, ymm11);

        packa += 16;
        packb += 6;
    }

    _mm256_storeu_ps(&C(0, 0), ymm0);
    _mm256_storeu_ps(&C(8, 0), ymm1);

    _mm256_storeu_ps(&C(0, 1), ymm2);
    _mm256_storeu_ps(&C(8, 1), ymm3);

    _mm256_storeu_ps(&C(0, 2), ymm4);
    _mm256_storeu_ps(&C(8, 2), ymm5);

    _mm256_storeu_ps(&C(0, 3), ymm6);
    _mm256_storeu_ps(&C(8, 3), ymm7);

    _mm256_storeu_ps(&C(0, 4), ymm8);
    _mm256_storeu_ps(&C(8, 4), ymm9);

    _mm256_storeu_ps(&C(0, 5), ymm10);
    _mm256_storeu_ps(&C(8, 5), ymm11);
}