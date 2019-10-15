#ifdef __aarch64__

#define dst         x0
#define pa          x1
#define pb          x2
#define k           x3
#define n           x4
#define ldc         x5

#ifdef GEMM_REQUANT
#define scales      x6
#define bias        x7
#endif

#define src_ori     x8
#define kd8_ori     x9
#define tmp_ptr_b   x14

#define is_kd8_even x15
#define kd4_ori     x16
#define kd2_ori     x17
#define kd1_ori     x18
#define kd          x19

#define nd4_ori     x20
#define nd2_ori     x21
#define nd1_ori     x22
#define nd          x23

#define temp        x24
#define temp32      w25

//void int8kernel_m1(int32_t* dst, const uint8_t* pa, const uint8_t* pb, size_t k, size_t n, ldc);
//void int8kernel_m1_requant(int32_t* dst, const uint8_t* pa, const uint8_t* pb, size_t k, size_t n, ldc, scales, bias);


.macro INIT
    eor v8.16b, v8.16b, v8.16b
    eor v9.16b, v9.16b, v9.16b
    eor v10.16b, v10.16b, v10.16b
    eor v11.16b, v11.16b, v11.16b
.endm

.macro DECOMPOSE_K
    // divide k into  8*kd8_ori + 4*kd4_ori + 2*kd2_ori + 1*kd1_ori
    lsr kd8_ori, k, 3  // kd8_ori = k / 8, means loopk
    lsr temp, kd8_ori, 1
    lsl is_kd8_even, temp, 1
    sub is_kd8_even, kd8_ori, is_kd8_even
    
    lsl temp, kd8_ori, 3
    sub k, k, temp
    lsr kd4_ori, k, 2  
    lsl temp, kd4_ori, 2
    sub k, k, temp
    
    lsr kd2_ori, k, 1
    lsl temp, kd2_ori, 1
    sub k, k, temp
    
    mov kd1_ori, k
.endm

.macro DECOMPOSE_N
    // divide n into 4*nd4_ori + 2*nd2_ori + 1*nd1_ori
    lsr nd4_ori, n, 2 
    lsl nd, nd4_ori, 2
    sub n, n, nd
    
    lsr nd2_ori, n, 1 
    lsl nd, nd2_ori, 1
    sub n, n, nd

    mov nd1_ori, n
.endm

.text
.align 5
#ifndef FUNCTION_NAME 
#define FUNCTION_NAME int8kernel_m1
#endif
.global FUNCTION_NAME 
.type FUNCTION_NAME, %function
FUNCTION_NAME:

    add sp, sp, #-(11 * 16)
    stp d8, d9, [sp, #(0 * 16)]
    stp d10, d11, [sp, #(1 * 16)]
    stp d12, d13, [sp, #(2 * 16)]
    stp d14, d15, [sp, #(3 * 16)]
    stp d16, d17, [sp, #(4 * 16)]
    stp x18, x19, [sp, #(5 * 16)]
    stp x20, x21, [sp, #(6 * 16)]
    stp x22, x23, [sp, #(7 * 16)]
    stp x24, x25, [sp, #(8 * 16)]
    stp x26, x27, [sp, #(9 * 16)]
    str x28, [sp, #(10 * 16)]

    DECOMPOSE_K
    DECOMPOSE_N
    
    cmp nd4_ori, #0
    beq nd2_start // means n < 4 
loopnd4:
    INIT
    mov src_ori, pa  // PanelA

    cmp kd8_ori, #0
    beq loopkd4_nd4

    mov kd, kd8_ori

    cmp is_kd8_even, #0 
    beq loopkd8_nd4_even // loop number is even 

    // start loopm1_kd8_nd4
    subs kd, kd, #1
    ld1 {v4.8b, v5.8b, v6.8b, v7.8b}, [pb], #32 // load four lines of B
    ld1 {v2.8b}, [pa], #8  // load two lines of PanelA

    smull v0.8h, v4.8b, v2.8b
    saddlp v8.4s, v0.8h
    smull v0.8h, v5.8b, v2.8b
    saddlp v9.4s, v0.8h
    smull v0.8h, v6.8b, v2.8b
    saddlp v10.4s, v0.8h
    smull v0.8h, v7.8b, v2.8b
    saddlp v11.4s, v0.8h

    cmp kd, #0
    beq loopkd8_nd4_end 

    loopkd8_nd4_even:
        ld1 {v4.8b, v5.8b, v6.8b, v7.8b}, [pb], #32
        ld1 {v12.8b, v13.8b, v14.8b, v15.8b}, [pb], #32

        ld1 {v2.8b, v3.8b}, [pa], #16

        smull v0.8h, v2.8b, v4.8b
        smlal v0.8h, v3.8b, v12.8b
        sadalp v8.4s, v0.8h

        smull v1.8h, v2.8b, v5.8b
        smlal v1.8h, v3.8b, v13.8b
        sadalp v9.4s, v1.8h

        smull v0.8h, v2.8b, v6.8b
        smlal v0.8h, v3.8b, v14.8b
        sadalp v10.4s, v0.8h

        smull v1.8h, v2.8b, v7.8b
        smlal v1.8h, v3.8b, v15.8b
        sadalp v11.4s, v1.8h

        subs kd, kd, #2
        bne loopkd8_nd4_even

    loopkd8_nd4_end:
        addp v8.4s, v8.4s, v9.4s
        addp v10.4s, v10.4s, v11.4s
        addp v8.4s, v8.4s, v10.4s

        // start process kd4 kd2 kd1 cases
    loopkd4_nd4:
        cmp kd4_ori, 0
        beq loopkd2_nd4 
        // start subkernel_m1n4k4
        ld1 {v4.8b, v5.8b}, [pb], #16  // load B4x4
        sxtl v4.8h, v4.8b
        sxtl v5.8h, v5.8b
        mov v6.d[0], v4.d[1]
        mov v7.d[0], v5.d[1]

        ld1 {v2.8b}, [pa]  // load A1x4
        add pa, pa, #4
        sxtl v2.8h, v2.8b

        smull v12.4s, v2.4h, v4.4h
        smull v13.4s, v2.4h, v6.4h
        smull v14.4s, v2.4h, v5.4h
        smull v15.4s, v2.4h, v7.4h

        addp v12.4s, v12.4s, v13.4s
        addp v14.4s, v14.4s, v15.4s
        addp v12.4s, v12.4s, v14.4s
        add v8.4s, v8.4s, v12.4s

    loopkd2_nd4:
        cmp kd2_ori, 0
        beq loopkd1_nd4 
        // start subkernel_m1n4k2
        ld1 {v4.8b}, [pa]       // load A1x2
        add pa, pa, #2
        ld1 {v0.8b}, [pb], #8   // load B2x4 

        mov v4.h[1], v4.h[0]
        mov v4.s[1], v4.s[0]
        
        smull v0.8h, v0.8b, v4.8b
        sadalp v8.4s, v0.8h

    loopkd1_nd4:
        cmp kd1_ori, 0 
        beq loopnd4_end
        // start subkernel_m1n4k1
        ld1 {v4.8b}, [pb]   // load B1x4
        add pb, pb, #4
        ld1 {v2.8b}, [pa]   // load A1x1
        add pa, pa, #1
        sxtl v4.8h, v4.8b
        sxtl v2.8h, v2.8b
        smlal v8.4s, v4.4h, v2.h[0]

    loopnd4_end:
#ifdef GEMM_REQUANT
        ldr temp32, [scales] 
        // int32 => fp32
        scvtf v8.4s, v8.4s
        // fp32 *= scale_tm
        mov v12.s[0], temp32
        fmul v8.4s, v8.4s, v12.s[0]

        cmp bias, #0
        beq loopnd4_end_requant   // skip add bias

        // fp32 += bias_tm
        ldr temp32, [bias]
        dup v15.4s, temp32
        fadd v8.4s, v8.4s, v15.4s

        loopnd4_end_requant:
            // fp32 -> int32
            fcvtas v8.4s, v8.4s
            // int32 -> int16
            sqxtn v6.4h, v8.4s
            // int16 -> int8
            sqxtn v8.8b, v6.8h
            // save
            st1 {v8.s}[0], [dst]
            add dst, dst, #4
#else
        st1 {v8.4s}, [dst], #16
#endif

        subs nd4_ori, nd4_ori, #1
        mov pa, src_ori 
        bne loopnd4 

nd2_start:
    cmp nd2_ori, #0  // means n < 2
    beq nd1_start

    INIT
    mov src_ori, pa  // PanelA

    cmp kd8_ori, #0
    beq loopkd4_nd2  // k <= 7

    mov kd, kd8_ori
    cmp is_kd8_even, #0 
    beq loopkd8_nd2_even  // loop number is even 

    // start loopmd1_kd8_nd2
    subs kd, kd, #1
    ld1 {v4.8b, v5.8b}, [pb], #16  // load two lines of B
    ld1 {v2.8b}, [pa], #8  // load two lines of PanelA
    smull v0.8h, v4.8b, v2.8b
    saddlp v8.4s, v0.8h
    smull v0.8h, v5.8b, v2.8b
    saddlp v9.4s, v0.8h

    cmp kd, #0
    beq loopkd8_nd2_end 

    loopkd8_nd2_even:
        ld1 {v4.8b, v5.8b, v6.8b, v7.8b}, [pb], #32

        ld1 {v2.8b, v3.8b}, [pa], #16

        smull v0.8h, v2.8b, v4.8b
        smlal v0.8h, v3.8b, v6.8b
        sadalp v8.4s, v0.8h

        smull v1.8h, v2.8b, v5.8b
        smlal v1.8h, v3.8b, v7.8b
        sadalp v9.4s, v1.8h

        subs kd, kd, #2
        bne loopkd8_nd2_even

    loopkd8_nd2_end:
        addp v8.4s, v8.4s, v9.4s
        addp v8.4s, v8.4s, v8.4s

        // start process kd4 kd2 kd1 cases
    loopkd4_nd2:
        cmp kd4_ori, 0
        beq loopkd2_nd2 
        // start subkernel_m1n2k4
        ld1 {v4.8b}, [pb], #8  // load B4x2
        sxtl v4.8h, v4.8b
        mov v6.d[0], v4.d[1]

        ld1 {v2.8b}, [pa]      // load A1x4
        add pa, pa, #4
        sxtl v2.8h, v2.8b

        smull v9.4s, v2.4h, v4.4h
        smull v10.4s, v2.4h, v6.4h

        addp v9.4s, v9.4s, v10.4s
        addp v9.4s, v9.4s, v9.4s
        add v8.4s, v8.4s, v9.4s

    loopkd2_nd2:
        cmp kd2_ori, 0
        beq loopkd1_nd2 
        // start subkernel_m1n2k2
        ld1 {v4.8b}, [pa]   // load A1x2
        add pa, pa, #2
        ld1 {v0.8b}, [pb]   // load B2x2
        add pb, pb, #4

        mov v4.h[1], v4.h[0]

        smull v0.8h, v4.8b, v0.8b
        saddlp v0.4s, v0.8h
        add v8.4s, v8.4s, v0.4s

    loopkd1_nd2:
        cmp kd1_ori, 0 
        beq loopnd2_end
        // start subkernel_m1n2k1
        ld1 {v4.8b}, [pb]   // load B1x2
        add pb, pb, #2
        ld1 {v2.8b}, [pa]   // load A1x1
        add pa, pa, #2
        sxtl v4.8h, v4.8b
        sxtl v2.8h, v2.8b
        smlal v8.4s, v4.4h, v2.h[0]

    loopnd2_end:
#ifdef GEMM_REQUANT
        // v12: s0 s1
        ldr temp32, [scales]
        mov v12.s[0], temp32
        mov v12.s[1], v12.s[0]

        // int32 => fp32
        scvtf v8.2s, v8.2s
        // fp32 *= scale_tm
        fmul v8.2s, v8.2s, v12.2s 

        cmp bias, #0
        beq loopnd2_end_requant   // skip add bias

        // fp32 += bias_tm
        ldr temp32, [bias]
        mov v12.s[0], temp32
        mov v12.s[1], v12.s[0]
        fadd v8.4s, v8.4s, v12.4s

        loopnd2_end_requant:
            // fp32 -> int32
            fcvtas v8.2s, v8.2s
            // int32 -> int16
            sqxtn v8.4h, v8.4s
            // int16 -> int8
            sqxtn v8.8b, v8.8h
            // save
            st1 {v8.h}[0], [dst]
            add dst, dst, #2
#else
        st1 {v8.2s}, [dst], #8
#endif
        mov pa, src_ori

nd1_start:
    cmp nd1_ori, 0  // means left n == 0
    beq finish

    INIT

    cmp kd8_ori, #0
    beq loopkd4_nd1 // k <= 7

    mov kd, kd8_ori
    
    cmp is_kd8_even, 0 
    beq loopkd8_nd1_even // loop number is even 

    // start loopkd8_nd1
    subs kd, kd, #1
    ld1 {v4.8b}, [pb], #8  // load B line
    ld1 {v2.8b}, [pa], #8  // load A line 
    smull v0.8h, v4.8b, v2.8b
    saddlp v8.4s, v0.8h

    cmp kd, #0
    beq loopkd8_nd1_end 

    loopkd8_nd1_even:
        ld1 {v4.8b, v5.8b}, [pb], #16
        ld1 {v24.8b, v25.8b}, [pa], #16
        
        smull v0.8h, v24.8b, v4.8b
        smlal v0.8h, v25.8b, v5.8b
        sadalp v8.4s, v0.8h

        subs kd, kd, #2
        bne loopkd8_nd1_even

    loopkd8_nd1_end:
        addp v8.4s, v8.4s, v8.4s
        addp v8.4s, v8.4s, v8.4s

        // start process kd4 kd2 kd1 cases
    loopkd4_nd1: 
        cmp kd4_ori, 0
        beq loopkd2_nd1 
        // start subkernel_m1n1k4
        ld1 {v4.8b}, [pb]  // load B4x1
        add pb, pb, #4
        sxtl v4.8h, v4.8b   // extend B4x1 to v4

        ld1 {v2.8b}, [pa]  // load A1x4
        add pa, pa, #4
        sxtl v2.8h, v2.8b

        smull v9.4s, v2.4h, v4.4h
        addp v9.4s, v9.4s, v9.4s
        addp v9.4s, v9.4s, v9.4s
        add v8.4s, v8.4s, v9.4s

    loopkd2_nd1:
        cmp kd2_ori, 0
        beq loopkd1_nd1
        // start subkernel_m1n1k2
        ld1 {v4.8b}, [pa]   // load A1x2
        add pa, pa, #2
        ld1 {v0.8b}, [pb]   // load B2x1
        add pb, pb, #2

        smull v0.8h, v0.8b, v4.8b
        saddlp v0.4s, v0.8h
        
        add v8.4s, v8.4s, v0.4s

    loopkd1_nd1:
        cmp kd1_ori, 0 
        beq loopnd1_end
        // start subkernel_m1n1k1

        ld1 {v0.8b}, [pb]    // load B1x1
        add pb, pb, #1

        ld1 {v1.8b}, [pa]   // load A1x1
        add pa, pa, #1
        
        sxtl v1.8h, v1.8b
        sxtl v0.8h, v0.8b

        smull v0.4s, v1.4h, v0.h[0]
        
        add v8.4s, v8.4s, v0.4s

    loopnd1_end:
#ifdef GEMM_REQUANT
        // v12: s0 s1 
        ldr temp32, [scales]
        // int32 => fp32
        scvtf v8.2s, v8.2s
        // fp32 *= scale_tm
        mov v12.s[0], temp32
        fmul v8.2s, v8.2s, v12.2s 

        cmp bias, #0
        beq loopnd1_end_requant   // skip add bias

        // fp32 += bias_tm
        ldr temp32, [bias]
        mov v12.s[0], temp32
        fadd v8.2s, v8.2s, v12.2s

        loopnd1_end_requant:
            // fp32 -> int32
            fcvtas v8.2s, v8.2s
            // int32 -> int16
            sqxtn v8.4h, v8.4s
            // int16 -> int8
            sqxtn v8.8b, v8.8h
            // save
            st1 {v8.b}[0], [dst]
#else
        // no need to add the last output pointer
        st1 {v8.s}[0], [dst]
#endif

finish:
    mov x0, #0              // set return value
    ldp d8, d9, [sp, #(0 * 16)]
    ldp d10, d11, [sp, #(1 * 16)]
    ldp d12, d13, [sp, #(2 * 16)]
    ldp d14, d15, [sp, #(3 * 16)]
    ldp d16, d17, [sp, #(4 * 16)]
    ldp x18, x19, [sp, #(5 * 16)]
    ldp x20, x21, [sp, #(6 * 16)]
    ldp x22, x23, [sp, #(7 * 16)]
    ldp x24, x25, [sp, #(8 * 16)]
    ldp x26, x27, [sp, #(9 * 16)]
    ldr x28, [sp, #(10 * 16)]
    add sp, sp, #(11*16)
    ret
#endif
