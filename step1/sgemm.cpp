#include "../sgemm.h"

void my_sgemm(int m, int n, int k, const float *a, int lda, const float *b, int ldb, float *c, int ldc) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            // dot product
            register float tmp = 0;
            for (int p = 0; p < k; p++) {
                tmp += A(i, p) * B(p, j);
            }
            C(i, j) += tmp;
        }
    }
}

