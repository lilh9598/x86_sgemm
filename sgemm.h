#include <math.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>

// M N 必须是MR NR的倍数
#define MR 32
#define NR 6

#define Mc 256
#define Kc 256

#define MEM_ALIGN 64
// A L2 : Kc * Mc * 4 < L2
// B L3 : Kc * N * 4
// packb L1 : 16 * kc * 4 < L1

#define A(i, j) a[(j)*lda + i]
#define B(i, j) b[(j)*ldb + i]
#define C(i, j) c[(j)*ldc + i]

#define ROUND_UP(x, y) ((((x) + (y)-1) / (y)) * y)
#define min(x, y) ((x) < (y) ? (x) : (y))

void my_sgemm(int m, int n, int k, const float *a, int lda, const float *b, int ldb, float *c, int ldc);

static void reference_sgemm(int m, int n, int k, const float *a, int lda, const float *b, int ldb, float *c, int ldc) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int p = 0; p < k; p++) {
                C(i, j) += A(i, p) * B(p, j);
            }
        }
    }
}

static bool consistency_check(const float *c, const float *ref_c, int m, int n, int ldc, int ldc_ref) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (std::fabs(c[j * ldc + i] - ref_c[j * ldc_ref + i]) > 0.01) {
                printf("Error! %f != %f at [%d, %d]\n", c[j * ldc + i], ref_c[j * ldc_ref + i], i, j);
                return false;
            }
        }
    }
    return true;
}

static void *aligned_malloc(size_t required_bytes, size_t alignment = MEM_ALIGN) {
    int offset = alignment - 1 + sizeof(void *);

    void *p1 = (void *)malloc(required_bytes + offset);

    if (p1 == NULL) return NULL;

    void **p2 = (void **)(((size_t)p1 + offset) & ~(alignment - 1));

    p2[-1] = p1;

    return p2;
}

static void aligned_free(void *p2) {
    void *p1 = ((void **)p2)[-1];
    free(p1);
}