#include "../sgemm.h"

#include <immintrin.h>

void kernel_32x6(int k, const float *packa, const float *b, float *c, int ldc);

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

    delete[] packb;
    delete[] packa;
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
            kernel_32x6(k, packa + i * k, packb + j * k, &C(i, j), ldc);
        }
    }
}
void kernel_32x6(int k, const float *packa, const float *packb, float *c, int ldc) {
#define LOAD(x)                                 \
    auto vc##x##0 = _mm512_loadu_ps(&C(0, x));  \
    auto vc##x##1 = _mm512_loadu_ps(&C(16, x));

    LOAD(0);
    LOAD(1);
    LOAD(2);
    LOAD(3);
    LOAD(4);
    LOAD(5);

    __m512 vb, va0, va1;
#define FMA(x)                                     \
    vb = _mm512_set1_ps(packb[x]);                 \
    vc##x##0 = _mm512_fmadd_ps(va0, vb, vc##x##0); \
    vc##x##1 = _mm512_fmadd_ps(va1, vb, vc##x##1);

    for (int p = 0; p < k; p += 1) {
        va0 = _mm512_loadu_ps(packa);
        va1 = _mm512_loadu_ps(packa + 16);
        FMA(0);
        FMA(1);
        FMA(2);
        FMA(3);
        FMA(4);
        FMA(5);

        packb += 6;
        packa += 32;
    }

#define SAVE(x)                            \
    _mm512_storeu_ps(&C(0, x), vc##x##0);  \
    _mm512_storeu_ps(&C(16, x), vc##x##1);

    SAVE(0);
    SAVE(1);
    SAVE(2);
    SAVE(3);
    SAVE(4);
    SAVE(5);
}