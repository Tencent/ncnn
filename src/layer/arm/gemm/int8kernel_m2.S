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
#define dst_ptr_0   x10
#define dst_ptr_1   x11
#define tmp_ptr_b   x12

#define is_kd8_even x13
#define kd4_ori     x14
#define kd2_ori     x15
#define kd1_ori     x16
#define kd          x17

#define nd4_ori     x18
#define nd2_ori     x19
#define nd1_ori     x20
#define nd          x21

#define temp        x22
#define temp1       x23
#define temp2       x24

//void int8kernel_m2(int32_t* dst, const uint8_t* pa, const uint8_t* pb, size_t k, size_t n, ldc);
//void int8kernel_m2_requant(int32_t* dst, const uint8_t* pa, const uint8_t* pb, size_t k, size_t n, ldc, scales, bias);

.macro INIT
    eor v8.16b, v8.16b, v8.16b
    eor v9.16b, v9.16b, v9.16b
    eor v10.16b, v10.16b, v10.16b
    eor v11.16b, v11.16b, v11.16b
    eor v12.16b, v12.16b, v12.16b
    eor v13.16b, v13.16b, v13.16b
    eor v14.16b, v14.16b, v14.16b
    eor v15.16b, v15.16b, v15.16b

    eor v16.16b, v16.16b, v16.16b
    eor v17.16b, v17.16b, v17.16b
    eor v18.16b, v18.16b, v18.16b
    eor v19.16b, v19.16b, v19.16b
    eor v20.16b, v20.16b, v20.16b
    eor v21.16b, v21.16b, v21.16b
    eor v22.16b, v22.16b, v22.16b
    eor v23.16b, v23.16b, v23.16b
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
#define FUNCTION_NAME int8kernel_m2
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

#ifndef GEMM_REQUANT    // if requant, typeof C = int8_t
    lsl ldc, ldc, #2    // else sizeof(int32_t) = 4
#endif
    mov dst_ptr_0, dst
    add dst_ptr_1, dst_ptr_0, ldc
    
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

    // start loopm2_kd8_nd4
    subs kd, kd, #1
    ld1 {v4.8b, v5.8b, v6.8b, v7.8b}, [pb], #32 // load four lines of B
    ld1 {v2.8b, v3.8b}, [pa], #16  // load two lines of PanelA
    smull v0.8h, v4.8b, v2.8b
    smull v1.8h, v4.8b, v3.8b
    saddlp v8.4s, v0.8h
    saddlp v12.4s, v1.8h
    smull v0.8h, v5.8b, v2.8b
    smull v1.8h, v5.8b, v3.8b
    saddlp v9.4s, v0.8h
    saddlp v13.4s, v1.8h
    smull v0.8h, v6.8b, v2.8b
    smull v1.8h, v6.8b, v3.8b
    saddlp v10.4s, v0.8h
    saddlp v14.4s, v1.8h
    smull v0.8h, v7.8b, v2.8b
    smull v1.8h, v7.8b, v3.8b
    saddlp v11.4s, v0.8h
    saddlp v15.4s, v1.8h

    cmp kd, #0
    beq loopkd8_nd4_end 

    loopkd8_nd4_even:
        add tmp_ptr_b, pb, #32 
        ld1 {v4.8b, v5.8b}, [pb], #16
        ld1 {v2.8b, v3.8b}, [pa], #16

        smull v0.8h, v4.8b, v2.8b
        smull v1.8h, v5.8b, v2.8b

        ld1 {v6.8b, v7.8b}, [tmp_ptr_b], #16
        ld1 {v24.8b, v25.8b}, [pa], #16

        smlal v0.8h, v6.8b, v24.8b
        smlal v1.8h, v7.8b, v24.8b

        sadalp v8.4s, v0.8h
        sadalp v9.4s, v1.8h
    
        smull v0.8h, v4.8b, v3.8b
        smull v1.8h, v5.8b, v3.8b
        smlal v0.8h, v6.8b, v25.8b
        smlal v1.8h, v7.8b, v25.8b

        sadalp v12.4s, v0.8h
        sadalp v13.4s, v1.8h

        // start v10v11, v14v15, v18v19, v22v23, error here!
        ld1 {v4.8b, v5.8b}, [pb], #16
        smull v0.8h, v4.8b, v2.8b
        smull v1.8h, v5.8b, v2.8b
        ld1 {v6.8b, v7.8b}, [tmp_ptr_b], #16
        smlal v0.8h, v6.8b, v24.8b
        smlal v1.8h, v7.8b, v24.8b
        sadalp v10.4s, v0.8h
        sadalp v11.4s, v1.8h

        smull v0.8h, v4.8b, v3.8b
        smull v1.8h, v5.8b, v3.8b
        smlal v0.8h, v6.8b, v25.8b
        smlal v1.8h, v7.8b, v25.8b
        sadalp v14.4s, v0.8h
        sadalp v15.4s, v1.8h

        add pb, pb, #32
        subs kd, kd, #2
        bne loopkd8_nd4_even

    loopkd8_nd4_end:
        addp v8.4s, v8.4s, v9.4s
        addp v10.4s, v10.4s, v11.4s
        addp v12.4s, v12.4s, v13.4s
        addp v14.4s, v14.4s, v15.4s

        addp v8.4s, v8.4s, v10.4s
        addp v9.4s, v12.4s, v14.4s

        // start process kd4 kd2 kd1 cases
    loopkd4_nd4:
        cmp kd4_ori, 0
        beq loopkd2_nd4 
        // start subkernel_m2n4k4
        ld1 {v4.8b, v5.8b}, [pb], #16  // load B4x4
        sxtl v4.8h, v4.8b
        sxtl v5.8h, v5.8b
        mov v6.d[0], v4.d[1]
        mov v7.d[0], v5.d[1]

        ld1 {v2.8b}, [pa], #8  // load A2x4
        sxtl v2.8h, v2.8b
        mov v3.d[0], v2.d[1]

        smull v12.4s, v2.4h, v4.4h
        smull v13.4s, v2.4h, v6.4h
        smull v14.4s, v2.4h, v5.4h
        smull v15.4s, v2.4h, v7.4h

        addp v12.4s, v12.4s, v13.4s
        addp v14.4s, v14.4s, v15.4s
        addp v12.4s, v12.4s, v14.4s
        add v8.4s, v8.4s, v12.4s

        smull v16.4s, v3.4h, v4.4h
        smull v17.4s, v3.4h, v6.4h
        smull v18.4s, v3.4h, v5.4h
        smull v19.4s, v3.4h, v7.4h

        addp v16.4s, v16.4s, v17.4s
        addp v18.4s, v18.4s, v19.4s
        addp v16.4s, v16.4s, v18.4s
        add v9.4s, v9.4s, v16.4s

    loopkd2_nd4:
        cmp kd2_ori, 0
        beq loopkd1_nd4 
        // start subkernel_m2n4k2
        ld1 {v4.8b}, [pa]       // load A2x2
        add pa, pa, #4
        ld1 {v0.8b}, [pb], #8   // load B2x4 
                            // 00 11 22 33
        rev32 v1.4h, v0.4h  // 11 00 33 22
        rev64 v2.2s, v0.2s  // 22 33 00 11
        rev64 v3.4h, v0.4h  // 33 22 11 00

        smull v12.8h, v4.8b, v0.8b
        smull v13.8h, v4.8b, v1.8b
        smull v14.8h, v4.8b, v2.8b
        smull v15.8h, v4.8b, v3.8b

        saddlp v12.4s, v12.8h
        saddlp v13.4s, v13.8h
        saddlp v14.4s, v14.8h
        saddlp v15.4s, v15.8h

        mov v16.s[0], v12.s[0]
        mov v16.s[1], v13.s[0]
        mov v16.s[2], v14.s[0]
        mov v16.s[3], v15.s[0]

        mov v17.s[0], v13.s[1]
        mov v17.s[1], v12.s[1]
        mov v17.s[2], v15.s[1]
        mov v17.s[3], v14.s[1]
        
        add v8.4s, v8.4s, v16.4s
        add v9.4s, v9.4s, v17.4s

    loopkd1_nd4:
        cmp kd1_ori, 0 
        beq loopnd4_end
        // start subkernel_m2n4k1
        ld1 {v4.8b}, [pb]   // load B1x4
        add pb, pb, #4
        ld1 {v2.8b}, [pa]   // load A2x1
        add pa, pa, #2
        sxtl v4.8h, v4.8b
        sxtl v2.8h, v2.8b
        smlal v8.4s, v4.4h, v2.h[0]
        smlal v9.4s, v4.4h, v2.h[1]

    loopnd4_end:
#ifdef GEMM_REQUANT
        ld1 {v12.2s}, [scales]
        // int32 => fp32
        scvtf v8.4s, v8.4s
        scvtf v9.4s, v9.4s
        // fp32 *= scale_tm
        fmul v8.4s, v8.4s, v12.s[0]
        fmul v9.4s, v9.4s, v12.s[1]

        cmp bias, #0
        beq loopnd4_end_requant   // skip add bias

        // fp32 += bias_tm
        ld1 {v14.2s}, [bias]
        dup v15.4s, v14.s[0]
        fadd v8.4s, v8.4s, v15.4s
        dup v15.4s, v14.s[1]
        fadd v9.4s, v9.4s, v15.4s

        loopnd4_end_requant:
            // fp32 -> int32
            fcvtas v8.4s, v8.4s
            fcvtas v9.4s, v9.4s
            // int32 -> int16
            sqxtn v6.4h, v8.4s
            sqxtn2 v6.8h, v9.4s
            // int16 -> int8
            sqxtn v8.8b, v6.8h
            // save
            st1 {v8.s}[0], [dst_ptr_0]
            add dst_ptr_0, dst_ptr_0, #4
            st1 {v8.s}[1], [dst_ptr_1]
            add dst_ptr_1, dst_ptr_1, #4
#else
        st1 {v8.4s}, [dst_ptr_0], #16
        st1 {v9.4s}, [dst_ptr_1], #16
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

    // start loopmd2_kd8_nd2
    subs kd, kd, #1
    ld1 {v4.8b, v5.8b}, [pb], #16  // load two lines of B
    ld1 {v2.8b, v3.8b}, [pa], #16  // load two lines of PanelA
    smull v0.8h, v4.8b, v2.8b
    smull v1.8h, v4.8b, v3.8b
    saddlp v8.4s, v0.8h
    saddlp v12.4s, v1.8h
    smull v0.8h, v5.8b, v2.8b
    smull v1.8h, v5.8b, v3.8b
    saddlp v9.4s, v0.8h
    saddlp v13.4s, v1.8h

    cmp kd, #0
    beq loopkd8_nd2_end 

    loopkd8_nd2_even:
        // TODO
        ld1 {v4.8b, v5.8b}, [pb], #16
        ld1 {v2.8b, v3.8b}, [pa], #16

        smull v0.8h, v4.8b, v2.8b
        ld1 {v6.8b, v7.8b}, [pb], #16
        smull v1.8h, v5.8b, v2.8b
        ld1 {v24.8b, v25.8b}, [pa], #16

        smlal v0.8h, v6.8b, v24.8b
        smlal v1.8h, v7.8b, v24.8b

        sadalp v8.4s, v0.8h
        sadalp v9.4s, v1.8h
    
        smull v0.8h, v4.8b, v3.8b
        smull v1.8h, v5.8b, v3.8b
        smlal v0.8h, v6.8b, v25.8b
        smlal v1.8h, v7.8b, v25.8b

        sadalp v12.4s, v0.8h
        sadalp v13.4s, v1.8h

        subs kd, kd, #2
        bne loopkd8_nd2_even

    loopkd8_nd2_end:
        addp v8.4s, v8.4s, v9.4s
        addp v12.4s, v12.4s, v13.4s

        addp v8.4s, v8.4s, v8.4s
        addp v12.4s, v12.4s, v12.4s

        // start process kd4 kd2 kd1 cases
    loopkd4_nd2:
        cmp kd4_ori, 0
        beq loopkd2_nd2 
        // start subkernel_m2n2k4
        ld1 {v4.8b}, [pb], #8  // load B4x2
        sxtl v4.8h, v4.8b
        mov v6.d[0], v4.d[1]

        ld1 {v2.8b}, [pa], #8  // load first A2x4
        sxtl v2.8h, v2.8b
        mov v3.d[0], v2.d[1]

        smull v9.4s, v2.4h, v4.4h
        smull v10.4s, v2.4h, v6.4h

        addp v9.4s, v9.4s, v10.4s
        addp v9.4s, v9.4s, v9.4s
        add v8.4s, v8.4s, v9.4s

        smull v13.4s, v3.4h, v4.4h
        smull v14.4s, v3.4h, v6.4h

        addp v13.4s, v13.4s, v14.4s
        addp v13.4s, v13.4s, v13.4s
        add v12.4s, v12.4s, v13.4s

    loopkd2_nd2:
        cmp kd2_ori, 0
        beq loopkd1_nd2 
        // start subkernel_m2n2k2
        ld1 {v4.8b}, [pa]   // load A2x2
        add pa, pa, #4
        ld1 {v0.8b}, [pb]   // load B2x2
        add pb, pb, #4
                            // 00 11
        rev32 v1.4h, v0.4h  // 11 00

        smull v21.8h, v4.8b, v0.8b
        smull v22.8h, v4.8b, v1.8b

        saddlp v21.4s, v21.8h
        saddlp v22.4s, v22.8h

        mov v9.s[0], v21.s[0]
        mov v9.s[1], v22.s[0]
        add v8.4s, v8.4s, v9.4s

        mov v13.s[0], v22.s[1]
        mov v13.s[1], v21.s[1]
        add v12.4s, v12.4s, v13.4s

    loopkd1_nd2:
        cmp kd1_ori, 0 
        beq loopnd2_end
        // start subkernel_m2n2k1
        ld1 {v4.8b}, [pb]   // load B1x2
        add pb, pb, #2
        ld1 {v2.8b}, [pa]   // load A4x1
        add pa, pa, #2
        sxtl v4.8h, v4.8b
        sxtl v2.8h, v2.8b
        smlal v8.4s, v4.4h, v2.h[0]
        smlal v12.4s, v4.4h, v2.h[1]

    loopnd2_end:
#ifdef GEMM_REQUANT
        mov v8.d[1], v12.d[0]

        // v12: 0 1
        ld1 {v12.2s}, [scales]
        zip1 v12.4s, v12.4s, v12.4s
        // v12: 0 0 1 1

        // int32 => fp32
        scvtf v8.4s, v8.4s
        // fp32 *= scale_tm
        fmul v8.4s, v8.4s, v12.4s 

        cmp bias, #0
        beq loopnd2_end_requant   // skip add bias

        // fp32 += bias_tm
        ld1 {v12.2s}, [bias]
        zip1 v12.4s, v12.4s, v12.4s
        fadd v8.4s, v8.4s, v12.4s

        loopnd2_end_requant:
            // fp32 -> int32
            fcvtas v8.4s, v8.4s
            // int32 -> int16
            sqxtn v8.4h, v8.4s
            // int16 -> int8
            sqxtn v8.8b, v8.8h
            // save
            st1 {v8.h}[0], [dst_ptr_0]
            add dst_ptr_0, dst_ptr_0, #2
            st1 {v8.h}[1], [dst_ptr_1]
            add dst_ptr_1, dst_ptr_1, #2
#else
        st1 {v8.2s}, [dst_ptr_0], #8
        st1 {v12.2s}, [dst_ptr_1], #8
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
    ld1 {v4.8b}, [pb], #8  // load four lines of B
    ld1 {v2.8b, v3.8b}, [pa], #16  // load two lines of PanelA
    smull v0.8h, v4.8b, v2.8b
    smull v1.8h, v4.8b, v3.8b
    saddlp v8.4s, v0.8h
    saddlp v12.4s, v1.8h

    cmp kd, #0
    beq loopkd8_nd1_end 

    loopkd8_nd1_even:
        ld1 {v4.8b, v5.8b}, [pb], #16
        ld1 {v24.8b, v25.8b, v26.8b, v27.8b}, [pa], #32
        
        smull v0.8h, v24.8b, v4.8b
        smlal v0.8h, v26.8b, v5.8b
        sadalp v8.4s, v0.8h

        smull v1.8h, v25.8b, v4.8b
        smlal v1.8h, v27.8b, v5.8b
        sadalp v12.4s, v1.8h
        
        subs kd, kd, #2
        bne loopkd8_nd1_even

    loopkd8_nd1_end:
        addp v8.4s, v8.4s, v8.4s
        addp v8.4s, v8.4s, v8.4s
        addp v12.4s, v12.4s, v12.4s
        addp v12.4s, v12.4s, v12.4s

        // start process kd4 kd2 kd1 cases
    loopkd4_nd1: 
        cmp kd4_ori, 0
        beq loopkd2_nd1 
        // start subkernel_m2n1k2
        ld1 {v4.8b}, [pb]  // load B4x1
        add pb, pb, #4
        sxtl v4.8h, v4.8b   // extend B4x1 to v4

        ld1 {v2.8b}, [pa], #8  // load A2x4
        sxtl v2.8h, v2.8b
        mov v5.d[0], v2.d[1]

        smull v9.4s, v2.4h, v4.4h
        addp v9.4s, v9.4s, v9.4s
        addp v9.4s, v9.4s, v9.4s
        add v8.4s, v8.4s, v9.4s

        smull v13.4s, v5.4h, v4.4h
        addp v13.4s, v13.4s, v13.4s
        addp v13.4s, v13.4s, v13.4s
        add v12.4s, v12.4s, v13.4s

    loopkd2_nd1:
        cmp kd2_ori, 0
        beq loopkd1_nd1
        // start subkernel_m2n1k2
        ld1 {v4.8b}, [pa]   // load A2x2
        add pa, pa, #4
        ld1 {v0.8b}, [pb]   // load B2x1
        add pb, pb, #2

        mov v0.h[1], v0.h[0]

        smull v0.8h, v0.8b, v4.8b
        saddlp v0.4s, v0.8h
        
        mov v9.s[0], v0.s[0]
        add v8.4s, v8.4s, v9.4s
        mov v13.s[0], v0.s[1]
        add v12.4s, v12.4s, v13.4s

    loopkd1_nd1:
        cmp kd1_ori, 0 
        beq loopnd1_end
        // start subkernel_m2n1k1

        ld1 {v0.8b}, [pb]    // load B1x1
        add pb, pb, #1

        ld1 {v1.8b}, [pa]   // load A2x1
        add pa, pa, #2
        
        sxtl v1.8h, v1.8b
        sxtl v0.8h, v0.8b

        smull v0.4s, v1.4h, v0.h[0]
        mov v1.s[0], v0.s[1]
        
        add v8.4s, v8.4s, v0.4s
        add v12.4s, v12.4s, v1.4s

    loopnd1_end:
#ifdef GEMM_REQUANT
        mov v8.s[1], v12.s[0]

        // v12: s0 s1 
        ld1 {v12.2s}, [scales]
        // int32 => fp32
        scvtf v8.2s, v8.2s
        // fp32 *= scale_tm
        fmul v8.2s, v8.2s, v12.2s 

        cmp bias, #0
        beq loopnd1_end_requant   // skip add bias

        // fp32 += bias_tm
        ld1 {v12.2s}, [bias]
        fadd v8.2s, v8.2s, v12.2s

        loopnd1_end_requant:
            // fp32 -> int32
            fcvtas v8.2s, v8.2s
            // int32 -> int16
            sqxtn v8.4h, v8.4s
            // int16 -> int8
            sqxtn v8.8b, v8.8h
            // save
            st1 {v8.b}[0], [dst_ptr_0]
            st1 {v8.b}[1], [dst_ptr_1]
#else
        // no need to add the last output pointer
        st1 {v8.s}[0], [dst_ptr_0]
        st1 {v12.s}[0], [dst_ptr_1]
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
