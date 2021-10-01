#include "../common.h"

#define K               %edi
#define K_ITER          %eax
#define K_REMAIN        %r9d
#define A               %rsi
#define B               %rdx
#define C               %rcx
#define LDC             %r8

#define vA0_0             %zmm31
#define vA0_1             %zmm30

#define vA1_0             %zmm29
#define vA1_1             %zmm28

#define vA2_0             %zmm27
#define vA2_1             %zmm26

#define vA3_0             %zmm25
#define vA3_1             %zmm24


#define vB0             %zmm23
#define vB1             %zmm22
#define vB2             %zmm21
#define vB3             %zmm20
#define vB4             %zmm19
#define vB5             %zmm18
#define vB6             %zmm16


#define vC0_0           %zmm11
#define vC0_1           %zmm10
#define vC1_0           %zmm9
#define vC1_1           %zmm8
#define vC2_0           %zmm7
#define vC2_1           %zmm6
#define vC3_0           %zmm5
#define vC3_1           %zmm4
#define vC4_0           %zmm3
#define vC4_1           %zmm2
#define vC5_0           %zmm1
#define vC5_1           %zmm0

#define     AINCREASE   addq $128, A
#define     BINCREASE   leaq 24(B), B

.macro LOAD_A a0, a1
        vmovups (A), \a0
        vmovups 64(A), \a1
        AINCREASE
.endm

.macro FMA6_LOAD_A a0, a1, na0, na1

        vbroadcastss    (B),  vB0
        vbroadcastss    4(B), vB1
        vfmadd231ps     vB0, \a0, vC0_0
        vfmadd231ps     vB0, \a1, vC0_1

        vmovups         (A),  \na0
        vbroadcastss    8(B), vB2
        vfmadd231ps     vB1, \a0, vC1_0
        vfmadd231ps     vB1, \a1, vC1_1

        vmovups         64(A), \na1
        vbroadcastss    12(B), vB3
        vfmadd231ps     vB2, \a0, vC2_0
        vfmadd231ps     vB2, \a1, vC2_1

        vbroadcastss    16(B), vB4
        vfmadd231ps     vB3, \a0, vC3_0
        vfmadd231ps     vB3, \a1, vC3_1

        AINCREASE
        vbroadcastss    20(B), vB5
        vfmadd231ps     vB4, \a0, vC4_0
        vfmadd231ps     vB4, \a1, vC4_1

        BINCREASE
        vfmadd231ps     vB5, \a0, vC5_0
        vfmadd231ps     vB5, \a1, vC5_1
.endm

.macro FMA6A a0, a1
        vbroadcastss    (B), vB0
        vbroadcastss    4(B), vB1
        vfmadd231ps     vB0, \a0, vC0_0
        vfmadd231ps     vB0, \a1, vC0_1

        vbroadcastss    8(B), vB2
        vfmadd231ps     vB1, \a0, vC1_0
        vfmadd231ps     vB1, \a1, vC1_1

        vbroadcastss    12(B), vB3
        vfmadd231ps     vB2, \a0, vC2_0
        vfmadd231ps     vB2, \a1, vC2_1

        vbroadcastss    16(B), vB4
        vfmadd231ps     vB3, \a0, vC3_0
        vfmadd231ps     vB3, \a1, vC3_1

        vbroadcastss    20(B), vB5
        vfmadd231ps     vB4, \a0, vC4_0
        vfmadd231ps     vB4, \a1, vC4_1

        BINCREASE
        vfmadd231ps     vB5, \a0, vC5_0
        vfmadd231ps     vB5, \a1, vC5_1
.endm

GLOB_FUN_START(kernel_32x6)
        PROLOGUE
        vzeroall
        movl    K, K_ITER
        shrl    $2, K_ITER   // K_ITER = k / 4
        movl    K, K_REMAIN
        andl    $3 ,K_REMAIN // K_REMAIN = k % 4

.LOOP4_INIT:
        testl  K_ITER, K_ITER // if K_ITER == 0
        jz     .LOOP1_INIT

        LOAD_A vA0_0, vA0_1
        LOAD_A vA1_0, vA1_1
        decl     K_ITER
        jz     .LOOP4_END
.LOOP4:
        FMA6_LOAD_A vA0_0, vA0_1, vA2_0, vA2_1
        FMA6_LOAD_A vA1_0, vA1_1, vA3_0, vA3_1
        FMA6_LOAD_A vA2_0, vA2_1, vA0_0, vA0_1
        FMA6_LOAD_A vA3_0, vA3_1, vA1_0, vA1_1
        decl     K_ITER
        jnz      .LOOP4

.LOOP4_END:
        FMA6_LOAD_A vA0_0, vA0_1, vA2_0, vA2_1
        FMA6_LOAD_A vA1_0, vA1_1, vA3_0, vA3_1
        FMA6A vA2_0, vA2_1
        FMA6A vA3_0, vA3_1
.LOOP1_INIT:
        testl K_REMAIN, K_REMAIN   // if (K_REMAIN == 0) go to end
        jz     .LAST
.LOOP1:                                # =>This Inner Loop Header: Depth=1
        LOAD_A vA0_0, vA0_1
        FMA6A vA0_0, vA0_1
        decl    K_REMAIN
        jnz     .LOOP1
.LAST:

.macro ADD_SW_C c0, c1
        vaddps          (C), \c0, \c0
        vmovups         \c0, (C) // c(0,0)
        vaddps          64(C), \c1, \c1
        vmovups         \c1, 64(C)  // c(8,0)
        leaq            (C, LDC, 4), C
.endm
        ADD_SW_C vC0_0, vC0_1
        ADD_SW_C vC1_0, vC1_1
        ADD_SW_C vC2_0, vC2_1
        ADD_SW_C vC3_0, vC3_1
        ADD_SW_C vC4_0, vC4_1
        ADD_SW_C vC5_0, vC5_1
        EPILOGUE
        retq

FUN_END(kernel_32x6)