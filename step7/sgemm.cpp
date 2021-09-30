#include "../sgemm.h"

#include <immintrin.h>

extern "C" void kernel_16x6(int k, const float *packa, const float *b, float *c, int ldc);

void mcxkc_sgemm(int m, int n, int k, const float *packa, const float *packb, float *c, int ldc);

void packB_Kcxn(const float *b, int ldb, int k, int nr, float *packb);

void packA_KcxMc(const float *a, int lda, int k, int mr, float *packa);

void my_sgemm(int m, int n, int k, const float *a, int lda, const float *b, int ldb, float *c, int ldc) {
    float *packb = (float*)aligned_malloc((ROUND_UP(n, NR) * Kc)*sizeof(float));
    float *packa = (float*)aligned_malloc((ROUND_UP(Mc, MR) * Kc)*sizeof(float));
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
    aligned_free(packb);
    aligned_free(packa);
}

void packA_KcxMc(const float *a, int lda, int k, int mr, float *packa) {
    int m_r16 = mr & ~15;
    int m_r8 = mr & ~7;
    for (int p = 0; p < k; p++) {
        int i = 0;
        while (i < m_r16) {
            auto v1 = _mm256_loadu_ps(a);
            auto v2 = _mm256_loadu_ps(a + 8);
            _mm256_storeu_ps(packa, v1);
            _mm256_storeu_ps(packa + 8, v2);
            packa += 16;
            i += 16;
        }

        while (i < m_r8) {
            auto v1 = _mm256_loadu_ps(a);
            _mm256_storeu_ps(packa, v1);
            packa += 8;
            i += 8;
        }
        while (i < mr) {
            packa[i] = a[i];
            i++;
        }
        a += lda;
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