#include "../sgemm.h"
#include <immintrin.h>

extern "C" void kernel_32x6(int k, const float *packa, const float *b, float *c, int ldc);

void mcxkc_sgemm(int m, int n, int k, const float *packa, const float *packb, float *c, int ldc);

void packB_Kcxn(const float *b, int ldb, int k, int nr, float *packb);

void packA_KcxMc(const float *a, int lda, int k, int mr, float *packa);

void my_sgemm(int m, int n, int k, const float *a, int lda, const float *b, int ldb, float *c, int ldc) {
    float *packb = (float *)aligned_malloc((ROUND_UP(n, NR) * Kc) * sizeof(float));
    float *packa = (float *)aligned_malloc((ROUND_UP(Mc, MR) * Kc) * sizeof(float));
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
            auto v1 = _mm512_loadu_ps(a + i);
            _mm512_storeu_ps(packa + i, v1);
            i += 16;
        }
        while (i < m_r8) {
            auto v1 = _mm256_loadu_ps(a + i);
            _mm256_storeu_ps(packa + i, v1);
            i += 8;
        }
        while (i < mr) {
            packa[i] = a[i];
            i++;
        }
        packa += MR;
        a += lda;
    }
}

void packB_Kcxn(const float *b, int ldb, int k, int nr, float *packb) {
    int nr4 = nr & ~3;
    int k4 = k & ~3;
    int p = 0;
    for (; p < k4; p += 4) {
        int i = 0;
        auto packb_ptr = packb + p * NR;
        while (i < nr4) {
            auto v0 = _mm_loadu_ps(&B(p, i + 0));
            auto v1 = _mm_loadu_ps(&B(p, i + 1));
            auto v2 = _mm_loadu_ps(&B(p, i + 2));
            auto v3 = _mm_loadu_ps(&B(p, i + 3));
            _MM_TRANSPOSE4_PS(v0, v1, v2, v3);
            _mm_storeu_ps(packb_ptr + i, v0);
            _mm_storeu_ps(packb_ptr + i + NR, v1);
            _mm_storeu_ps(packb_ptr + i + 2 * NR, v2);
            _mm_storeu_ps(packb_ptr + i + 3 * NR, v3);
            i += 4;
        }

        for (int pp = 0; pp < 4; pp++) {
            for (int j = i; j < nr; j++) {
                packb_ptr[j] = B(p + pp, j);
            }
            packb_ptr += NR;
        }
    }

    for (; p < k; p++) {
        auto packb_ptr = packb + p * NR;
        for (int i = 0; i < nr; i++) {
            packb_ptr[i] = B(p, i);
        }
    }
}

void mcxkc_sgemm(int m, int n, int k, const float *packa, const float *packb, float *c, int ldc) {
    for (int j = 0; j < n; j += NR) {
        for (int i = 0; i < m; i += MR) {
            kernel_32x6(k, packa + i * k, packb + j * k, &C(i, j), ldc);
        }
    }
}