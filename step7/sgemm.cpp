#include "../sgemm.h"

#include <immintrin.h>


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
    const float *pack_ptr[MR];
    for (int i = 0; i < mr; i++) {
        pack_ptr[i] = a + i * lda;
    }

    for (int i = mr; i < MR; i++) {
        pack_ptr[i] = a;
    }

    for (int p = 0; p < k; p++) {
        for (int i = 0; i < MR; i++) {
            *packa++ = *pack_ptr[i]++;
        }
    }
}

void packB_Kcxn(const float *b, int ldb, int k, int nr, float *packb) {
    const float *pack_ptr[NR];
    for (int i = 0; i < nr; i++) {
        pack_ptr[i] = b + i;
    }
    for (int i = nr; i < NR; i++) {
        pack_ptr[i] = b;
    }
    for (int p = 0; p < k; p++) {
        for (int i = 0; i < NR; i++) {
            *packb++ = *pack_ptr[i];
            pack_ptr[i] += ldb;
        }
    }
}

void mcxkc_sgemm(int m, int n, int k, const float *packa, const float *packb, float *c, int ldc) {
    int k_iter = k / 2;
    int k_remain = k % 2;
    for (int j = 0; j < n; j += NR) {
        for (int i = 0; i < m; i += MR) {
            __asm__ volatile(
            "leaq       (, %4, 4), %%r10           \n\t" // 0 * ldc
            "leaq       (%%r10, %%r10, 2), %%r11           \n\t"
            "vmovups    (%3), %%ymm0                \n\t"  // c(0,0)
            "vmovups    32(%3), %%ymm1               \n\t"
            "vmovups    (%3, %%r10), %%ymm2           \n\t"  // c(1,0)
            "vmovups    32(%3, %%r10), %%ymm3           \n\t"
            "vmovups    (%3, %%r10, 2), %%ymm4        \n\t"  // c(2,0)
            "vmovups    32(%3, %%r10, 2), %%ymm5        \n\t"
            "vmovups    (%3, %%r11), %%ymm6        \n\t"  // c(3,0)
            "vmovups    32(%3, %%r11), %%ymm7        \n\t"
            "leaq       (%%r11, %%r10, 2), %%r11           \n\t"
            "vmovups    (%3, %%r10, 4), %%ymm8        \n\t"  // c(4,0)
            "vmovups    32(%3, %%r10, 4), %%ymm9        \n\t"
            "movq %2, %%r14                                           \n\t"
            "movq %1, %%r13                                           \n\t"
            "vmovups    (%3, %%r11), %%ymm10       \n\t"  // c(5,0)
            "vmovups    32(%3, %%r11), %%ymm11       \n\t"
            "movl %0, %%r12d                                           \n\t"
            "testl %%r12d, %%r12d                                           \n\t"
            "je .END                                           \n\t"
            ".LOOP:                                     \n\t"
            "vmovups    (%%r14), %%ymm12       \n\t"
            "vmovups    32(%%r14), %%ymm13       \n\t"

            "addq $64, %%r14                          \n\t"
            "vbroadcastss    (%%r13), %%ymm15       \n\t"
            "vfmadd231ps    %%ymm12, %%ymm15, %%ymm0       \n\t"
            "vfmadd231ps    %%ymm13, %%ymm15, %%ymm1       \n\t"

            "vbroadcastss    4(%%r13), %%ymm15       \n\t"
            "vfmadd231ps    %%ymm12, %%ymm15, %%ymm2       \n\t"
            "vfmadd231ps    %%ymm13, %%ymm15, %%ymm3       \n\t"

            "vbroadcastss    8(%%r13), %%ymm15       \n\t"
            "vfmadd231ps    %%ymm12, %%ymm15, %%ymm4       \n\t"
            "vfmadd231ps    %%ymm13, %%ymm15, %%ymm5       \n\t"

            "vbroadcastss    12(%%r13), %%ymm15       \n\t"
            "vfmadd231ps    %%ymm12, %%ymm15, %%ymm6       \n\t"
            "vfmadd231ps    %%ymm13, %%ymm15, %%ymm7       \n\t"

            "vbroadcastss    16(%%r13), %%ymm15       \n\t"
            "vfmadd231ps    %%ymm12, %%ymm15, %%ymm8       \n\t"
            "vfmadd231ps    %%ymm13, %%ymm15, %%ymm9       \n\t"

            "vbroadcastss    20(%%r13), %%ymm15       \n\t"
            "vfmadd231ps    %%ymm12, %%ymm15, %%ymm10       \n\t"
            "vfmadd231ps    %%ymm13, %%ymm15, %%ymm11       \n\t"

            "addq $24, %%r13                          \n\t"
            "vmovups    (%%r14), %%ymm12       \n\t"
            "vmovups    32(%%r14), %%ymm13       \n\t"

            "addq $64, %%r14                          \n\t"
            "vbroadcastss    (%%r13), %%ymm15       \n\t"
            "vfmadd231ps    %%ymm12, %%ymm15, %%ymm0       \n\t"
            "vfmadd231ps    %%ymm13, %%ymm15, %%ymm1       \n\t"

            "vbroadcastss    4(%%r13), %%ymm15       \n\t"
            "vfmadd231ps    %%ymm12, %%ymm15, %%ymm2       \n\t"
            "vfmadd231ps    %%ymm13, %%ymm15, %%ymm3       \n\t"

            "vbroadcastss    8(%%r13), %%ymm15       \n\t"
            "vfmadd231ps    %%ymm12, %%ymm15, %%ymm4       \n\t"
            "vfmadd231ps    %%ymm13, %%ymm15, %%ymm5       \n\t"

            "vbroadcastss    12(%%r13), %%ymm15       \n\t"
            "vfmadd231ps    %%ymm12, %%ymm15, %%ymm6       \n\t"
            "vfmadd231ps    %%ymm13, %%ymm15, %%ymm7       \n\t"

            "vbroadcastss    16(%%r13), %%ymm15       \n\t"
            "vfmadd231ps    %%ymm12, %%ymm15, %%ymm8       \n\t"
            "vfmadd231ps    %%ymm13, %%ymm15, %%ymm9       \n\t"

            "vbroadcastss    20(%%r13), %%ymm15       \n\t"
            "vfmadd231ps    %%ymm12, %%ymm15, %%ymm10       \n\t"
            "vfmadd231ps    %%ymm13, %%ymm15, %%ymm11       \n\t"

            "addq $24, %%r13                          \n\t"
            "decl %%r12d                                \n\t"
            "jne .LOOP                              \n\t"
            "testl %5, %5                              \n\t"
            "je .STORE                               \n\t"

            "vmovups    (%%r14), %%ymm12       \n\t"
            "vmovups    32(%%r14), %%ymm13       \n\t"

            "vbroadcastss    (%%r13), %%ymm15       \n\t"
            "vfmadd231ps    %%ymm12, %%ymm15, %%ymm0       \n\t"
            "vfmadd231ps    %%ymm13, %%ymm15, %%ymm1       \n\t"

            "vbroadcastss    4(%%r13), %%ymm15       \n\t"
            "vfmadd231ps    %%ymm12, %%ymm15, %%ymm2       \n\t"
            "vfmadd231ps    %%ymm13, %%ymm15, %%ymm3       \n\t"

            "vbroadcastss    8(%%r13), %%ymm15       \n\t"
            "vfmadd231ps    %%ymm12, %%ymm15, %%ymm4       \n\t"
            "vfmadd231ps    %%ymm13, %%ymm15, %%ymm5       \n\t"

            "vbroadcastss    12(%%r13), %%ymm15       \n\t"
            "vfmadd231ps    %%ymm12, %%ymm15, %%ymm6       \n\t"
            "vfmadd231ps    %%ymm13, %%ymm15, %%ymm7       \n\t"

            "vbroadcastss    16(%%r13), %%ymm15       \n\t"
            "vfmadd231ps    %%ymm12, %%ymm15, %%ymm8       \n\t"
            "vfmadd231ps    %%ymm13, %%ymm15, %%ymm9       \n\t"

            "vbroadcastss    20(%%r13), %%ymm15       \n\t"
            "vfmadd231ps    %%ymm12, %%ymm15, %%ymm10       \n\t"
            "vfmadd231ps    %%ymm13, %%ymm15, %%ymm11       \n\t"
            ".STORE:                                     \n\t"
            "vmovups    %%ymm0, (%3)                \n\t"  // c(0,0)
            "vmovups    %%ymm1, 32(%3)               \n\t"
            "movq       %%r10, %%r11               \n\t"
            "vmovups    %%ymm2, (%3, %%r11)           \n\t"  // c(1,0)
            "vmovups    %%ymm3, 32(%3, %%r11)           \n\t"
            "addq       %%r10, %%r11           \n\t"
            "vmovups    %%ymm4, (%3, %%r11)        \n\t"  // c(2,0)
            "vmovups    %%ymm5, 32(%3, %%r11)        \n\t"
            "addq       %%r10, %%r11           \n\t"
            "vmovups    %%ymm6, (%3, %%r11)        \n\t"  // c(3,0)
            "vmovups    %%ymm7, 32(%3, %%r11)        \n\t"
            "addq       %%r10, %%r11           \n\t"
            "vmovups    %%ymm8, (%3, %%r11)        \n\t"  // c(4,0)
            "vmovups    %%ymm9, 32(%3, %%r11)        \n\t"
            "addq       %%r10, %%r11           \n\t"
            "vmovups    %%ymm10, (%3, %%r11)       \n\t"  // c(5,0)
            "vmovups    %%ymm11, 32(%3, %%r11)       \n\t"
            ".END:                                     \n\t"
            :                    // output operands (none)
            :                    // input operands
            "r"(k_iter),         // 0
            "r"(packa + i * k),  // 1
            "r"(packb + j * k),  // 2
            "r"(&C(i, j)),       // 3
            "r"(ldc),             // 4
            "r"(k_remain)        // 5
            :                    // register clobber list
            "r10", "r11", "r12", "r13", "r14", "rax", "rbx", "memory", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4",
            "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13", "ymm14", "ymm15");
        }
    }
}