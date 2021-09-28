#include <math.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>

// 不支持尾处理 Mc / MR， N / NR 必须没有余数
#define Mc 144
#define Kc 512

#define MR 6
#define NR 16

#define MEM_ALIGN 64
// A L2 : Kc * Mc * 4 < 256KB
// B L3 : Kc * N * 4
// packb L1 : 16 * kc * 4 < 32K
#define A(i, j) a[(i)*lda + j]
#define B(i, j) b[(i)*ldb + j]
#define C(i, j) c[(i)*ldc + j]

#define ROUND_UP(x, y) ((((x) + (y)-1) / (y)) * y)
#define min(x, y) ((x) < (y) ? (x) : (y))

void my_sgemm(int m, int n, int k, const float *a, int lda, const float *b, int ldb, float *c, int ldc);

void kernel_16x6(int k, const float *packa, const float *packb, float *c, int ldc);

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
            if (std::fabs(c[i * ldc + j] - ref_c[i * ldc_ref + j]) > 0.01) {
                printf("Error! %f != %f at [%d, %d]\n", c[i * ldc + j], ref_c[i * ldc_ref + j], i, j);
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