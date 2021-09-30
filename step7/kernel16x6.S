.global kernel_16x6
kernel_16x6:
        pushq   %r13
        pushq   %r12
        movl    %edi, %eax
        sarl    %eax
        andl    $1, %edi
        leaq    (,%r8d,4), %r10
        leaq    (%r10,%r10,2), %r13
        vmovups (%rcx), %ymm0
        vmovups 32(%rcx), %ymm1
        vmovups (%rcx,%r10), %ymm2
        vmovups 32(%rcx,%r10), %ymm3
        vmovups (%rcx,%r10,2), %ymm4
        vmovups 32(%rcx,%r10,2), %ymm5
        leaq    (%r13,%r10,2), %r11
        vmovups (%rcx,%r13), %ymm6
        vmovups 32(%rcx,%r13), %ymm7
        vmovups (%rcx,%r10,4), %ymm8
        vmovups 32(%rcx,%r10,4), %ymm9
        vmovups (%rcx,%r11), %ymm10
        vmovups 32(%rcx,%r11), %ymm11
        movq    %rdx, %r13
        movl    %eax, %r12d
        testl   %r12d, %r12d
        je      .REMAIN
.LOOP:
        vbroadcastss    (%r13), %ymm15
        vmovups (%rsi), %ymm12
        vmovups 32(%rsi), %ymm13
        vbroadcastss    4(%r13), %ymm14
        vfmadd231ps     %ymm12, %ymm15, %ymm0   # ymm0 = (ymm15 * ymm12) + ymm0
        vfmadd231ps     %ymm13, %ymm15, %ymm1   # ymm1 = (ymm15 * ymm13) + ymm1
        vbroadcastss    8(%r13), %ymm15
        vfmadd231ps     %ymm12, %ymm14, %ymm2   # ymm2 = (ymm14 * ymm12) + ymm2
        vfmadd231ps     %ymm13, %ymm14, %ymm3   # ymm3 = (ymm14 * ymm13) + ymm3
        vbroadcastss    12(%r13), %ymm14
        vfmadd231ps     %ymm12, %ymm15, %ymm4   # ymm4 = (ymm15 * ymm12) + ymm4
        vfmadd231ps     %ymm13, %ymm15, %ymm5   # ymm5 = (ymm15 * ymm13) + ymm5
        vbroadcastss    16(%r13), %ymm15
        vfmadd231ps     %ymm12, %ymm14, %ymm6   # ymm6 = (ymm14 * ymm12) + ymm6
        vfmadd231ps     %ymm13, %ymm14, %ymm7   # ymm7 = (ymm14 * ymm13) + ymm7
        vbroadcastss    20(%r13), %ymm14
        vfmadd231ps     %ymm12, %ymm15, %ymm8   # ymm8 = (ymm15 * ymm12) + ymm8
        vfmadd231ps     %ymm13, %ymm15, %ymm9   # ymm9 = (ymm15 * ymm13) + ymm9
        vbroadcastss    24(%r13), %ymm15
        vfmadd231ps     %ymm12, %ymm14, %ymm10  # ymm10 = (ymm14 * ymm12) + ymm10
        vfmadd231ps     %ymm13, %ymm14, %ymm11  # ymm11 = (ymm14 * ymm13) + ymm11
        vmovups 64(%rsi), %ymm12
        vmovups 96(%rsi), %ymm13
        addq    $128, %rsi
        vbroadcastss    28(%r13), %ymm14
        vfmadd231ps     %ymm12, %ymm15, %ymm0   # ymm0 = (ymm15 * ymm12) + ymm0
        vfmadd231ps     %ymm13, %ymm15, %ymm1   # ymm1 = (ymm15 * ymm13) + ymm1
        vbroadcastss    32(%r13), %ymm15
        vfmadd231ps     %ymm12, %ymm14, %ymm2   # ymm2 = (ymm14 * ymm12) + ymm2
        vfmadd231ps     %ymm13, %ymm14, %ymm3   # ymm3 = (ymm14 * ymm13) + ymm3
        vbroadcastss    36(%r13), %ymm14
        vfmadd231ps     %ymm12, %ymm15, %ymm4   # ymm4 = (ymm15 * ymm12) + ymm4
        vfmadd231ps     %ymm13, %ymm15, %ymm5   # ymm5 = (ymm15 * ymm13) + ymm5
        vbroadcastss    40(%r13), %ymm15
        vfmadd231ps     %ymm12, %ymm14, %ymm6   # ymm6 = (ymm14 * ymm12) + ymm6
        vfmadd231ps     %ymm13, %ymm14, %ymm7   # ymm7 = (ymm14 * ymm13) + ymm7
        vbroadcastss    44(%r13), %ymm14
        vfmadd231ps     %ymm12, %ymm15, %ymm8   # ymm8 = (ymm15 * ymm12) + ymm8
        vfmadd231ps     %ymm13, %ymm15, %ymm9   # ymm9 = (ymm15 * ymm13) + ymm9
        addq    $48, %r13
        vfmadd231ps     %ymm12, %ymm14, %ymm10  # ymm10 = (ymm14 * ymm12) + ymm10
        vfmadd231ps     %ymm13, %ymm14, %ymm11  # ymm11 = (ymm14 * ymm13) + ymm11
        decl    %r12d
        jne     .LOOP
.REMAIN:
        testl   %edi, %edi
        je      .STORE
        vmovups (%rsi), %ymm12
        vmovups 32(%rsi), %ymm13
        vbroadcastss    (%r13), %ymm15
        vfmadd231ps     %ymm12, %ymm15, %ymm0   # ymm0 = (ymm15 * ymm12) + ymm0
        vfmadd231ps     %ymm13, %ymm15, %ymm1   # ymm1 = (ymm15 * ymm13) + ymm1
        vbroadcastss    4(%r13), %ymm15
        vfmadd231ps     %ymm12, %ymm15, %ymm2   # ymm2 = (ymm15 * ymm12) + ymm2
        vfmadd231ps     %ymm13, %ymm15, %ymm3   # ymm3 = (ymm15 * ymm13) + ymm3
        vbroadcastss    8(%r13), %ymm15
        vfmadd231ps     %ymm12, %ymm15, %ymm4   # ymm4 = (ymm15 * ymm12) + ymm4
        vfmadd231ps     %ymm13, %ymm15, %ymm5   # ymm5 = (ymm15 * ymm13) + ymm5
        vbroadcastss    12(%r13), %ymm15
        vfmadd231ps     %ymm12, %ymm15, %ymm6   # ymm6 = (ymm15 * ymm12) + ymm6
        vfmadd231ps     %ymm13, %ymm15, %ymm7   # ymm7 = (ymm15 * ymm13) + ymm7
        vbroadcastss    16(%r13), %ymm15
        vfmadd231ps     %ymm12, %ymm15, %ymm8   # ymm8 = (ymm15 * ymm12) + ymm8
        vfmadd231ps     %ymm13, %ymm15, %ymm9   # ymm9 = (ymm15 * ymm13) + ymm9
        vbroadcastss    20(%r13), %ymm15
        vfmadd231ps     %ymm12, %ymm15, %ymm10  # ymm10 = (ymm15 * ymm12) + ymm10
        vfmadd231ps     %ymm13, %ymm15, %ymm11  # ymm11 = (ymm15 * ymm13) + ymm11
.STORE:
        leaq    (%r10,%r10,2), %r11
        vmovups %ymm0, (%rcx)
        vmovups %ymm1, 32(%rcx)
        vmovups %ymm2, (%rcx,%r10)
        vmovups %ymm3, 32(%rcx,%r10)
        vmovups %ymm4, (%rcx,%r10,2)
        vmovups %ymm5, 32(%rcx,%r10,2)
        leaq    (%r11,%r10,2), %r13
        vmovups %ymm6, (%rcx,%r11)
        vmovups %ymm7, 32(%rcx,%r11)
        vmovups %ymm8, (%rcx,%r10,4)
        vmovups %ymm9, 32(%rcx,%r10,4)
        vmovups %ymm10, (%rcx,%r13)
        vmovups %ymm11, 32(%rcx,%r13)

        popq    %r12
        popq    %r13
        vzeroupper
        retq