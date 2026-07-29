// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
void quantize_A_tile_wq_int8_fp16s_i8mm(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales);
void transpose_quantize_A_tile_wq_int8_fp16s_i8mm(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales);
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD && !__ARM_FEATURE_MATMUL_INT8
void quantize_A_tile_wq_int8_fp16s_asimddp(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales);
void transpose_quantize_A_tile_wq_int8_fp16s_asimddp(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales);
#endif

static void quantize_A_tile_wq_int8_fp16s(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_i8mm())
    {
        quantize_A_tile_wq_int8_fp16s_i8mm(A, AT_tile, AT_descales_tile, i, max_ii, k, max_kk, block_size, input_scales);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_asimddp())
    {
        quantize_A_tile_wq_int8_fp16s_asimddp(A, AT_tile, AT_descales_tile, i, max_ii, k, max_kk, block_size, input_scales);
        return;
    }
#endif

    const int elempack = A.elempack;
    signed char* pp = AT_tile;
    float* pd = AT_descales_tile;
    const int block_count = (max_kk + block_size - 1) / block_size;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    if (input_scales.empty())
    {
        int ii = 0;
#if __ARM_NEON
#if __aarch64__
        for (; ii + 7 < max_ii; ii += 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * elempack;

            float32x4_t _v127 = vdupq_n_f32(127.f);
            float32x4_t _zero = vdupq_n_f32(0.f);

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                if (elempack == 8)
                {
                    float32x4_t _absmax0 = vdupq_n_f32(0.f);
                    float32x4_t _absmax1 = vdupq_n_f32(0.f);
                    const unsigned short* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        uint16x8_t _p = vld1q_u16(p0a);
                        _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_p))));
                        _absmax1 = vmaxq_f32(_absmax1, vabsq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_p))));
                        p0a += 8;
                    }

                    float32x4_t _scale0 = vdivq_f32(_v127, _absmax0);
                    float32x4_t _scale1 = vdivq_f32(_v127, _absmax1);
                    _scale0 = vbslq_f32(vceqq_f32(_absmax0, _zero), _zero, _scale0);
                    _scale1 = vbslq_f32(vceqq_f32(_absmax1, _zero), _zero, _scale1);
                    vst1q_f32(pd, vmulq_n_f32(_absmax0, 1.f / 127.f));
                    vst1q_f32(pd + 4, vmulq_n_f32(_absmax1, 1.f / 127.f));

                    int kk = 0;
                    for (; kk + 7 < max_kk0; kk += 8)
                    {
                        uint16x8_t _p = vld1q_u16(p0);
                        uint16x8_t _q = vld1q_u16(p0 + 8);
                        uint16x8_t _r = vld1q_u16(p0 + 16);
                        uint16x8_t _s = vld1q_u16(p0 + 24);
                        uint16x8_t _t = vld1q_u16(p0 + 32);
                        uint16x8_t _u = vld1q_u16(p0 + 40);
                        uint16x8_t _v = vld1q_u16(p0 + 48);
                        uint16x8_t _w = vld1q_u16(p0 + 56);
                        float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vget_low_u16(_p));
                        float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vget_high_u16(_p));
                        float32x4_t _p2 = vcvt_f32_f16((float16x4_t)vget_low_u16(_q));
                        float32x4_t _p3 = vcvt_f32_f16((float16x4_t)vget_high_u16(_q));
                        float32x4_t _p4 = vcvt_f32_f16((float16x4_t)vget_low_u16(_r));
                        float32x4_t _p5 = vcvt_f32_f16((float16x4_t)vget_high_u16(_r));
                        float32x4_t _p6 = vcvt_f32_f16((float16x4_t)vget_low_u16(_s));
                        float32x4_t _p7 = vcvt_f32_f16((float16x4_t)vget_high_u16(_s));
                        float32x4_t _p8 = vcvt_f32_f16((float16x4_t)vget_low_u16(_t));
                        float32x4_t _p9 = vcvt_f32_f16((float16x4_t)vget_high_u16(_t));
                        float32x4_t _pa = vcvt_f32_f16((float16x4_t)vget_low_u16(_u));
                        float32x4_t _pb = vcvt_f32_f16((float16x4_t)vget_high_u16(_u));
                        float32x4_t _pc = vcvt_f32_f16((float16x4_t)vget_low_u16(_v));
                        float32x4_t _pd = vcvt_f32_f16((float16x4_t)vget_high_u16(_v));
                        float32x4_t _pe = vcvt_f32_f16((float16x4_t)vget_low_u16(_w));
                        float32x4_t _pf = vcvt_f32_f16((float16x4_t)vget_high_u16(_w));

                        _p0 = vmulq_f32(_p0, _scale0);
                        _p1 = vmulq_f32(_p1, _scale1);
                        _p2 = vmulq_f32(_p2, _scale0);
                        _p3 = vmulq_f32(_p3, _scale1);
                        _p4 = vmulq_f32(_p4, _scale0);
                        _p5 = vmulq_f32(_p5, _scale1);
                        _p6 = vmulq_f32(_p6, _scale0);
                        _p7 = vmulq_f32(_p7, _scale1);
                        _p8 = vmulq_f32(_p8, _scale0);
                        _p9 = vmulq_f32(_p9, _scale1);
                        _pa = vmulq_f32(_pa, _scale0);
                        _pb = vmulq_f32(_pb, _scale1);
                        _pc = vmulq_f32(_pc, _scale0);
                        _pd = vmulq_f32(_pd, _scale1);
                        _pe = vmulq_f32(_pe, _scale0);
                        _pf = vmulq_f32(_pf, _scale1);

#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                        int8x8x2_t _p04 = vzip_s8(float2int8(_p0, _p1), float2int8(_p8, _p9));
                        int8x8x2_t _p15 = vzip_s8(float2int8(_p2, _p3), float2int8(_pa, _pb));
                        int8x8x2_t _p26 = vzip_s8(float2int8(_p4, _p5), float2int8(_pc, _pd));
                        int8x8x2_t _p37 = vzip_s8(float2int8(_p6, _p7), float2int8(_pe, _pf));

                        int8x16x4_t _rr;
                        _rr.val[0] = vcombine_s8(_p04.val[0], _p04.val[1]);
                        _rr.val[1] = vcombine_s8(_p15.val[0], _p15.val[1]);
                        _rr.val[2] = vcombine_s8(_p26.val[0], _p26.val[1]);
                        _rr.val[3] = vcombine_s8(_p37.val[0], _p37.val[1]);
#else  // __ARM_FEATURE_MATMUL_INT8
                        int8x16x4_t _rr;
                        _rr.val[0] = vcombine_s8(float2int8(_p0, _p1), float2int8(_p8, _p9));
                        _rr.val[1] = vcombine_s8(float2int8(_p2, _p3), float2int8(_pa, _pb));
                        _rr.val[2] = vcombine_s8(float2int8(_p4, _p5), float2int8(_pc, _pd));
                        _rr.val[3] = vcombine_s8(float2int8(_p6, _p7), float2int8(_pe, _pf));
#endif // __ARM_FEATURE_MATMUL_INT8

                        vst4q_s8(pp, _rr);
#else  // __ARM_FEATURE_DOTPROD
                        int8x16x2_t _r01;
                        _r01.val[0] = vcombine_s8(float2int8(_p0, _p1), float2int8(_p4, _p5));
                        _r01.val[1] = vcombine_s8(float2int8(_p2, _p3), float2int8(_p6, _p7));
                        int8x16x2_t _r23;
                        _r23.val[0] = vcombine_s8(float2int8(_p8, _p9), float2int8(_pc, _pd));
                        _r23.val[1] = vcombine_s8(float2int8(_pa, _pb), float2int8(_pe, _pf));

                        vst2q_s8(pp, _r01);
                        vst2q_s8(pp + 32, _r23);
#endif // __ARM_FEATURE_DOTPROD

                        pp += 64;
                        p0 += 64;
                    }
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        uint16x8_t _p = vld1q_u16(p0);
                        uint16x8_t _q = vld1q_u16(p0 + 8);
                        uint16x8_t _r = vld1q_u16(p0 + 16);
                        uint16x8_t _s = vld1q_u16(p0 + 24);
                        float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vget_low_u16(_p));
                        float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vget_high_u16(_p));
                        float32x4_t _p2 = vcvt_f32_f16((float16x4_t)vget_low_u16(_q));
                        float32x4_t _p3 = vcvt_f32_f16((float16x4_t)vget_high_u16(_q));
                        float32x4_t _p4 = vcvt_f32_f16((float16x4_t)vget_low_u16(_r));
                        float32x4_t _p5 = vcvt_f32_f16((float16x4_t)vget_high_u16(_r));
                        float32x4_t _p6 = vcvt_f32_f16((float16x4_t)vget_low_u16(_s));
                        float32x4_t _p7 = vcvt_f32_f16((float16x4_t)vget_high_u16(_s));

                        _p0 = vmulq_f32(_p0, _scale0);
                        _p1 = vmulq_f32(_p1, _scale1);
                        _p2 = vmulq_f32(_p2, _scale0);
                        _p3 = vmulq_f32(_p3, _scale1);
                        _p4 = vmulq_f32(_p4, _scale0);
                        _p5 = vmulq_f32(_p5, _scale1);
                        _p6 = vmulq_f32(_p6, _scale0);
                        _p7 = vmulq_f32(_p7, _scale1);

#if __ARM_FEATURE_DOTPROD
                        int8x8x4_t _r0123;
                        _r0123.val[0] = float2int8(_p0, _p1);
                        _r0123.val[1] = float2int8(_p2, _p3);
                        _r0123.val[2] = float2int8(_p4, _p5);
                        _r0123.val[3] = float2int8(_p6, _p7);

                        vst4_s8(pp, _r0123);
#else  // __ARM_FEATURE_DOTPROD
                        int8x16x2_t _r01;
                        _r01.val[0] = vcombine_s8(float2int8(_p0, _p1), float2int8(_p4, _p5));
                        _r01.val[1] = vcombine_s8(float2int8(_p2, _p3), float2int8(_p6, _p7));

                        vst2q_s8(pp, _r01);
#endif // __ARM_FEATURE_DOTPROD

                        pp += 32;
                        p0 += 32;
                    }
                    for (; kk + 1 < max_kk0; kk += 2)
                    {
                        uint16x8_t _p01 = vld1q_u16(p0);
                        uint16x8_t _p23 = vld1q_u16(p0 + 8);

                        float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vget_low_u16(_p01));
                        float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vget_high_u16(_p01));
                        float32x4_t _p2 = vcvt_f32_f16((float16x4_t)vget_low_u16(_p23));
                        float32x4_t _p3 = vcvt_f32_f16((float16x4_t)vget_high_u16(_p23));

                        _p0 = vmulq_f32(_p0, _scale0);
                        _p1 = vmulq_f32(_p1, _scale1);
                        _p2 = vmulq_f32(_p2, _scale0);
                        _p3 = vmulq_f32(_p3, _scale1);

                        int8x8x2_t _r01;
                        _r01.val[0] = float2int8(_p0, _p1);
                        _r01.val[1] = float2int8(_p2, _p3);

                        vst2_s8(pp, _r01);

                        pp += 16;
                        p0 += 16;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        uint16x8_t _p01 = vld1q_u16(p0);
                        float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vget_low_u16(_p01));
                        float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vget_high_u16(_p01));

                        _p0 = vmulq_f32(_p0, _scale0);
                        _p1 = vmulq_f32(_p1, _scale1);

                        int8x8_t _r01 = float2int8(_p0, _p1);

                        vst1_s8(pp, _r01);

                        pp += 8;
                        p0 += 8;
                    }

                    pd += 8;
                }

                if (elempack == 4)
                {
                    float32x4_t _absmax0 = vdupq_n_f32(0.f);
                    float32x4_t _absmax1 = vdupq_n_f32(0.f);
                    const unsigned short* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0a))));
                        _absmax1 = vmaxq_f32(_absmax1, vabsq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0a + A_hstep * 4))));
                        p0a += 4;
                    }

                    float32x4_t _scale0 = vdivq_f32(_v127, _absmax0);
                    float32x4_t _scale1 = vdivq_f32(_v127, _absmax1);
                    _scale0 = vbslq_f32(vceqq_f32(_absmax0, _zero), _zero, _scale0);
                    _scale1 = vbslq_f32(vceqq_f32(_absmax1, _zero), _zero, _scale1);
                    vst1q_f32(pd, vmulq_n_f32(_absmax0, 1.f / 127.f));
                    vst1q_f32(pd + 4, vmulq_n_f32(_absmax1, 1.f / 127.f));

                    int kk = 0;
                    for (; kk + 7 < max_kk0; kk += 8)
                    {
#if __ARM_FEATURE_DOTPROD
                        uint16x8x4_t _p = vld4q_u16(p0);
                        uint16x8x4_t _q = vld4q_u16(p0 + A_hstep * 4);

                        float32x4_t _p0 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_p.val[0])), _scale0, 0);
                        float32x4_t _p1 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_p.val[1])), _scale0, 1);
                        float32x4_t _p2 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_p.val[2])), _scale0, 2);
                        float32x4_t _p3 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_p.val[3])), _scale0, 3);
                        float32x4_t _p4 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_p.val[0])), _scale0, 0);
                        float32x4_t _p5 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_p.val[1])), _scale0, 1);
                        float32x4_t _p6 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_p.val[2])), _scale0, 2);
                        float32x4_t _p7 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_p.val[3])), _scale0, 3);
                        float32x4_t _p8 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_q.val[0])), _scale1, 0);
                        float32x4_t _p9 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_q.val[1])), _scale1, 1);
                        float32x4_t _pa = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_q.val[2])), _scale1, 2);
                        float32x4_t _pb = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_q.val[3])), _scale1, 3);
                        float32x4_t _pc = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_q.val[0])), _scale1, 0);
                        float32x4_t _pd = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_q.val[1])), _scale1, 1);
                        float32x4_t _pe = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_q.val[2])), _scale1, 2);
                        float32x4_t _pf = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_q.val[3])), _scale1, 3);

#if __ARM_FEATURE_MATMUL_INT8
                        int8x8_t _r0 = float2int8(_p0, _p4);
                        int8x8_t _r1 = float2int8(_p1, _p5);
                        int8x8_t _r2 = float2int8(_p2, _p6);
                        int8x8_t _r3 = float2int8(_p3, _p7);
                        int8x8_t _r4 = float2int8(_p8, _pc);
                        int8x8_t _r5 = float2int8(_p9, _pd);
                        int8x8_t _r6 = float2int8(_pa, _pe);
                        int8x8_t _r7 = float2int8(_pb, _pf);
#else  // __ARM_FEATURE_MATMUL_INT8
                        int8x8_t _r0 = float2int8(_p0, _p1);
                        int8x8_t _r1 = float2int8(_p2, _p3);
                        int8x8_t _r2 = float2int8(_p8, _p9);
                        int8x8_t _r3 = float2int8(_pa, _pb);
                        int8x8_t _r4 = float2int8(_p4, _p5);
                        int8x8_t _r5 = float2int8(_p6, _p7);
                        int8x8_t _r6 = float2int8(_pc, _pd);
                        int8x8_t _r7 = float2int8(_pe, _pf);
#endif // __ARM_FEATURE_MATMUL_INT8

                        vst1q_s8(pp, vcombine_s8(_r0, _r1));
                        vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
                        vst1q_s8(pp + 32, vcombine_s8(_r4, _r5));
                        vst1q_s8(pp + 48, vcombine_s8(_r6, _r7));
#else  // __ARM_FEATURE_DOTPROD
                        uint16x8_t _p = vld1q_u16(p0);
                        uint16x8_t _q = vld1q_u16(p0 + 8);
                        uint16x8_t _r = vld1q_u16(p0 + 16);
                        uint16x8_t _s = vld1q_u16(p0 + 24);
                        uint16x8_t _t = vld1q_u16(p0 + A_hstep * 4);
                        uint16x8_t _u = vld1q_u16(p0 + A_hstep * 4 + 8);
                        uint16x8_t _v = vld1q_u16(p0 + A_hstep * 4 + 16);
                        uint16x8_t _w = vld1q_u16(p0 + A_hstep * 4 + 24);
                        float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vget_low_u16(_p));
                        float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vget_high_u16(_p));
                        float32x4_t _p2 = vcvt_f32_f16((float16x4_t)vget_low_u16(_q));
                        float32x4_t _p3 = vcvt_f32_f16((float16x4_t)vget_high_u16(_q));
                        float32x4_t _p4 = vcvt_f32_f16((float16x4_t)vget_low_u16(_r));
                        float32x4_t _p5 = vcvt_f32_f16((float16x4_t)vget_high_u16(_r));
                        float32x4_t _p6 = vcvt_f32_f16((float16x4_t)vget_low_u16(_s));
                        float32x4_t _p7 = vcvt_f32_f16((float16x4_t)vget_high_u16(_s));
                        float32x4_t _p8 = vcvt_f32_f16((float16x4_t)vget_low_u16(_t));
                        float32x4_t _p9 = vcvt_f32_f16((float16x4_t)vget_high_u16(_t));
                        float32x4_t _pa = vcvt_f32_f16((float16x4_t)vget_low_u16(_u));
                        float32x4_t _pb = vcvt_f32_f16((float16x4_t)vget_high_u16(_u));
                        float32x4_t _pc = vcvt_f32_f16((float16x4_t)vget_low_u16(_v));
                        float32x4_t _pd = vcvt_f32_f16((float16x4_t)vget_high_u16(_v));
                        float32x4_t _pe = vcvt_f32_f16((float16x4_t)vget_low_u16(_w));
                        float32x4_t _pf = vcvt_f32_f16((float16x4_t)vget_high_u16(_w));

                        _p0 = vmulq_f32(_p0, _scale0);
                        _p1 = vmulq_f32(_p1, _scale0);
                        _p2 = vmulq_f32(_p2, _scale0);
                        _p3 = vmulq_f32(_p3, _scale0);
                        _p4 = vmulq_f32(_p4, _scale0);
                        _p5 = vmulq_f32(_p5, _scale0);
                        _p6 = vmulq_f32(_p6, _scale0);
                        _p7 = vmulq_f32(_p7, _scale0);
                        _p8 = vmulq_f32(_p8, _scale1);
                        _p9 = vmulq_f32(_p9, _scale1);
                        _pa = vmulq_f32(_pa, _scale1);
                        _pb = vmulq_f32(_pb, _scale1);
                        _pc = vmulq_f32(_pc, _scale1);
                        _pd = vmulq_f32(_pd, _scale1);
                        _pe = vmulq_f32(_pe, _scale1);
                        _pf = vmulq_f32(_pf, _scale1);

                        int8x16x2_t _r01;
                        _r01.val[0] = vcombine_s8(float2int8(_p0, _p8), float2int8(_p2, _pa));
                        _r01.val[1] = vcombine_s8(float2int8(_p1, _p9), float2int8(_p3, _pb));
                        int8x16x2_t _r23;
                        _r23.val[0] = vcombine_s8(float2int8(_p4, _pc), float2int8(_p6, _pe));
                        _r23.val[1] = vcombine_s8(float2int8(_p5, _pd), float2int8(_p7, _pf));

                        vst2q_s8(pp, _r01);
                        vst2q_s8(pp + 32, _r23);
#endif // __ARM_FEATURE_DOTPROD

                        pp += 64;
                        p0 += 32;
                    }
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
#if __ARM_FEATURE_DOTPROD
                        uint16x4x4_t _p = vld4_u16(p0);
                        uint16x4x4_t _q = vld4_u16(p0 + A_hstep * 4);

                        float32x4_t _p0 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)_p.val[0]), _scale0, 0);
                        float32x4_t _p1 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)_p.val[1]), _scale0, 1);
                        float32x4_t _p2 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)_p.val[2]), _scale0, 2);
                        float32x4_t _p3 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)_p.val[3]), _scale0, 3);
                        float32x4_t _p4 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)_q.val[0]), _scale1, 0);
                        float32x4_t _p5 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)_q.val[1]), _scale1, 1);
                        float32x4_t _p6 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)_q.val[2]), _scale1, 2);
                        float32x4_t _p7 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)_q.val[3]), _scale1, 3);

                        int8x8_t _r0 = float2int8(_p0, _p1);
                        int8x8_t _r1 = float2int8(_p2, _p3);
                        int8x8_t _r2 = float2int8(_p4, _p5);
                        int8x8_t _r3 = float2int8(_p6, _p7);

                        vst1q_s8(pp, vcombine_s8(_r0, _r1));
                        vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
#else  // __ARM_FEATURE_DOTPROD
                        uint16x8_t _p = vld1q_u16(p0);
                        uint16x8_t _q = vld1q_u16(p0 + 8);
                        uint16x8_t _r = vld1q_u16(p0 + A_hstep * 4);
                        uint16x8_t _s = vld1q_u16(p0 + A_hstep * 4 + 8);
                        float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vget_low_u16(_p));
                        float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vget_high_u16(_p));
                        float32x4_t _p2 = vcvt_f32_f16((float16x4_t)vget_low_u16(_q));
                        float32x4_t _p3 = vcvt_f32_f16((float16x4_t)vget_high_u16(_q));
                        float32x4_t _p4 = vcvt_f32_f16((float16x4_t)vget_low_u16(_r));
                        float32x4_t _p5 = vcvt_f32_f16((float16x4_t)vget_high_u16(_r));
                        float32x4_t _p6 = vcvt_f32_f16((float16x4_t)vget_low_u16(_s));
                        float32x4_t _p7 = vcvt_f32_f16((float16x4_t)vget_high_u16(_s));

                        _p0 = vmulq_f32(_p0, _scale0);
                        _p1 = vmulq_f32(_p1, _scale0);
                        _p2 = vmulq_f32(_p2, _scale0);
                        _p3 = vmulq_f32(_p3, _scale0);
                        _p4 = vmulq_f32(_p4, _scale1);
                        _p5 = vmulq_f32(_p5, _scale1);
                        _p6 = vmulq_f32(_p6, _scale1);
                        _p7 = vmulq_f32(_p7, _scale1);

                        int8x16x2_t _r01;
                        _r01.val[0] = vcombine_s8(float2int8(_p0, _p4), float2int8(_p2, _p6));
                        _r01.val[1] = vcombine_s8(float2int8(_p1, _p5), float2int8(_p3, _p7));

                        vst2q_s8(pp, _r01);
#endif // __ARM_FEATURE_DOTPROD

                        pp += 32;
                        p0 += 16;
                    }
                    for (; kk + 1 < max_kk0; kk += 2)
                    {
                        uint16x8_t _p = vld1q_u16(p0);
                        uint16x8_t _q = vld1q_u16(p0 + A_hstep * 4);

                        float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vget_low_u16(_p));
                        float32x4_t _p0n = vcvt_f32_f16((float16x4_t)vget_high_u16(_p));
                        float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vget_low_u16(_q));
                        float32x4_t _p1n = vcvt_f32_f16((float16x4_t)vget_high_u16(_q));

                        _p0 = vmulq_f32(_p0, _scale0);
                        _p0n = vmulq_f32(_p0n, _scale0);
                        _p1 = vmulq_f32(_p1, _scale1);
                        _p1n = vmulq_f32(_p1n, _scale1);

                        int8x8x2_t _r01;
                        _r01.val[0] = float2int8(_p0, _p1);
                        _r01.val[1] = float2int8(_p0n, _p1n);

                        vst2_s8(pp, _r01);

                        pp += 16;
                        p0 += 8;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vld1_u16(p0));
                        float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 4));

                        _p0 = vmulq_f32(_p0, _scale0);
                        _p1 = vmulq_f32(_p1, _scale1);

                        int8x8_t _r01 = float2int8(_p0, _p1);

                        vst1_s8(pp, _r01);

                        pp += 8;
                        p0 += 4;
                    }

                    pd += 8;
                }

                if (elempack == 1)
                {
                    float32x4_t _absmax0 = vdupq_n_f32(0.f);
                    float32x4_t _absmax1 = vdupq_n_f32(0.f);
                    float32x4_t _absmax2 = vdupq_n_f32(0.f);
                    float32x4_t _absmax3 = vdupq_n_f32(0.f);
                    float32x4_t _absmax4 = vdupq_n_f32(0.f);
                    float32x4_t _absmax5 = vdupq_n_f32(0.f);
                    float32x4_t _absmax6 = vdupq_n_f32(0.f);
                    float32x4_t _absmax7 = vdupq_n_f32(0.f);
                    const unsigned short* p0a = p0;
                    int kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a));
                        _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(_p0));
                        float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a + A_hstep));
                        _absmax1 = vmaxq_f32(_absmax1, vabsq_f32(_p1));
                        float32x4_t _p2 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a + A_hstep * 2));
                        _absmax2 = vmaxq_f32(_absmax2, vabsq_f32(_p2));
                        float32x4_t _p3 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a + A_hstep * 3));
                        _absmax3 = vmaxq_f32(_absmax3, vabsq_f32(_p3));
                        float32x4_t _p4 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a + A_hstep * 4));
                        _absmax4 = vmaxq_f32(_absmax4, vabsq_f32(_p4));
                        float32x4_t _p5 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a + A_hstep * 5));
                        _absmax5 = vmaxq_f32(_absmax5, vabsq_f32(_p5));
                        float32x4_t _p6 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a + A_hstep * 6));
                        _absmax6 = vmaxq_f32(_absmax6, vabsq_f32(_p6));
                        float32x4_t _p7 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a + A_hstep * 7));
                        _absmax7 = vmaxq_f32(_absmax7, vabsq_f32(_p7));
                        p0a += 4;
                    }
                    float32x2_t _max0 = vpmax_f32(vget_low_f32(_absmax0), vget_high_f32(_absmax0));
                    float32x2_t _max1 = vpmax_f32(vget_low_f32(_absmax1), vget_high_f32(_absmax1));
                    float32x2_t _max2 = vpmax_f32(vget_low_f32(_absmax2), vget_high_f32(_absmax2));
                    float32x2_t _max3 = vpmax_f32(vget_low_f32(_absmax3), vget_high_f32(_absmax3));
                    float32x2_t _max4 = vpmax_f32(vget_low_f32(_absmax4), vget_high_f32(_absmax4));
                    float32x2_t _max5 = vpmax_f32(vget_low_f32(_absmax5), vget_high_f32(_absmax5));
                    float32x2_t _max6 = vpmax_f32(vget_low_f32(_absmax6), vget_high_f32(_absmax6));
                    float32x2_t _max7 = vpmax_f32(vget_low_f32(_absmax7), vget_high_f32(_absmax7));
                    _absmax0 = vcombine_f32(vpmax_f32(_max0, _max1), vpmax_f32(_max2, _max3));
                    _absmax1 = vcombine_f32(vpmax_f32(_max4, _max5), vpmax_f32(_max6, _max7));
                    for (; kk < max_kk0; kk++)
                    {
                        float32x4_t _p0 = vdupq_n_f32(float16_to_float32(p0a[0]));
                        _p0 = vsetq_lane_f32(float16_to_float32(p0a[A_hstep]), _p0, 1);
                        _p0 = vsetq_lane_f32(float16_to_float32(p0a[A_hstep * 2]), _p0, 2);
                        _p0 = vsetq_lane_f32(float16_to_float32(p0a[A_hstep * 3]), _p0, 3);
                        float32x4_t _p1 = vdupq_n_f32(float16_to_float32(p0a[A_hstep * 4]));
                        _p1 = vsetq_lane_f32(float16_to_float32(p0a[A_hstep * 5]), _p1, 1);
                        _p1 = vsetq_lane_f32(float16_to_float32(p0a[A_hstep * 6]), _p1, 2);
                        _p1 = vsetq_lane_f32(float16_to_float32(p0a[A_hstep * 7]), _p1, 3);
                        _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(_p0));
                        _absmax1 = vmaxq_f32(_absmax1, vabsq_f32(_p1));
                        p0a++;
                    }

                    vst1q_f32(pd, vmulq_n_f32(_absmax0, 1.f / 127.f));
                    vst1q_f32(pd + 4, vmulq_n_f32(_absmax1, 1.f / 127.f));

                    float32x4_t _v127 = vdupq_n_f32(127.f);
                    float32x4_t _zero = vdupq_n_f32(0.f);
                    float32x4_t _scale0 = vbslq_f32(vceqq_f32(_absmax0, _zero), _zero, vdivq_f32(_v127, _absmax0));
                    float32x4_t _scale1 = vbslq_f32(vceqq_f32(_absmax1, _zero), _zero, vdivq_f32(_v127, _absmax1));

                    kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                    for (; kk + 7 < max_kk0; kk += 8)
                    {
                        float32x4_t _p00 = vcvt_f32_f16((float16x4_t)vld1_u16(p0));
                        float32x4_t _p01 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 4));
                        float32x4_t _p10 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep));
                        float32x4_t _p11 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep + 4));
                        float32x4_t _p20 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 2));
                        float32x4_t _p21 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 2 + 4));
                        float32x4_t _p30 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 3));
                        float32x4_t _p31 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 3 + 4));
                        float32x4_t _p40 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 4));
                        float32x4_t _p41 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 4 + 4));
                        float32x4_t _p50 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 5));
                        float32x4_t _p51 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 5 + 4));
                        float32x4_t _p60 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 6));
                        float32x4_t _p61 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 6 + 4));
                        float32x4_t _p70 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 7));
                        float32x4_t _p71 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 7 + 4));
                        int8x8_t _r0 = float2int8(vmulq_laneq_f32(_p00, _scale0, 0), vmulq_laneq_f32(_p01, _scale0, 0));
                        int8x8_t _r1 = float2int8(vmulq_laneq_f32(_p10, _scale0, 1), vmulq_laneq_f32(_p11, _scale0, 1));
                        int8x8_t _r2 = float2int8(vmulq_laneq_f32(_p20, _scale0, 2), vmulq_laneq_f32(_p21, _scale0, 2));
                        int8x8_t _r3 = float2int8(vmulq_laneq_f32(_p30, _scale0, 3), vmulq_laneq_f32(_p31, _scale0, 3));
                        int8x8_t _r4 = float2int8(vmulq_laneq_f32(_p40, _scale1, 0), vmulq_laneq_f32(_p41, _scale1, 0));
                        int8x8_t _r5 = float2int8(vmulq_laneq_f32(_p50, _scale1, 1), vmulq_laneq_f32(_p51, _scale1, 1));
                        int8x8_t _r6 = float2int8(vmulq_laneq_f32(_p60, _scale1, 2), vmulq_laneq_f32(_p61, _scale1, 2));
                        int8x8_t _r7 = float2int8(vmulq_laneq_f32(_p70, _scale1, 3), vmulq_laneq_f32(_p71, _scale1, 3));
                        vst1q_s8(pp, vcombine_s8(_r0, _r1));
                        vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
                        vst1q_s8(pp + 32, vcombine_s8(_r4, _r5));
                        vst1q_s8(pp + 48, vcombine_s8(_r6, _r7));
                        pp += 64;
                        p0 += 8;
                    }
#endif // __ARM_FEATURE_MATMUL_INT8
#endif // __ARM_FEATURE_DOTPROD
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vld1_u16(p0));
                        float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep));
                        float32x4_t _p2 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 2));
                        float32x4_t _p3 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 3));
                        float32x4_t _p4 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 4));
                        float32x4_t _p5 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 5));
                        float32x4_t _p6 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 6));
                        float32x4_t _p7 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 7));
                        int8x8_t _r01 = float2int8(vmulq_laneq_f32(_p0, _scale0, 0), vmulq_laneq_f32(_p1, _scale0, 1));
                        int8x8_t _r23 = float2int8(vmulq_laneq_f32(_p2, _scale0, 2), vmulq_laneq_f32(_p3, _scale0, 3));
                        int8x8_t _r45 = float2int8(vmulq_laneq_f32(_p4, _scale1, 0), vmulq_laneq_f32(_p5, _scale1, 1));
                        int8x8_t _r67 = float2int8(vmulq_laneq_f32(_p6, _scale1, 2), vmulq_laneq_f32(_p7, _scale1, 3));
#if __ARM_FEATURE_DOTPROD
                        vst1q_s8(pp, vcombine_s8(_r01, _r23));
                        vst1q_s8(pp + 16, vcombine_s8(_r45, _r67));
#else
                        int16x8x2_t _r04 = vuzpq_s16(vreinterpretq_s16_s8(vcombine_s8(_r01, _r23)), vreinterpretq_s16_s8(vcombine_s8(_r45, _r67)));
                        vst1q_s16((short*)pp, _r04.val[0]);
                        vst1q_s16((short*)pp + 8, _r04.val[1]);
#endif
                        pp += 32;
                        p0 += 4;
                    }
                    for (; kk + 1 < max_kk0; kk += 2)
                    {
                        float v00 = float16_to_float32(p0[0]);
                        float v01 = float16_to_float32(p0[1]);
                        float v10 = float16_to_float32(p0[A_hstep]);
                        float v11 = float16_to_float32(p0[A_hstep + 1]);
                        float v20 = float16_to_float32(p0[A_hstep * 2]);
                        float v21 = float16_to_float32(p0[A_hstep * 2 + 1]);
                        float v30 = float16_to_float32(p0[A_hstep * 3]);
                        float v31 = float16_to_float32(p0[A_hstep * 3 + 1]);
                        float v40 = float16_to_float32(p0[A_hstep * 4]);
                        float v41 = float16_to_float32(p0[A_hstep * 4 + 1]);
                        float v50 = float16_to_float32(p0[A_hstep * 5]);
                        float v51 = float16_to_float32(p0[A_hstep * 5 + 1]);
                        float v60 = float16_to_float32(p0[A_hstep * 6]);
                        float v61 = float16_to_float32(p0[A_hstep * 6 + 1]);
                        float v70 = float16_to_float32(p0[A_hstep * 7]);
                        float v71 = float16_to_float32(p0[A_hstep * 7 + 1]);
                        *pp++ = float2int8(v00 * vgetq_lane_f32(_scale0, 0));
                        *pp++ = float2int8(v01 * vgetq_lane_f32(_scale0, 0));
                        *pp++ = float2int8(v10 * vgetq_lane_f32(_scale0, 1));
                        *pp++ = float2int8(v11 * vgetq_lane_f32(_scale0, 1));
                        *pp++ = float2int8(v20 * vgetq_lane_f32(_scale0, 2));
                        *pp++ = float2int8(v21 * vgetq_lane_f32(_scale0, 2));
                        *pp++ = float2int8(v30 * vgetq_lane_f32(_scale0, 3));
                        *pp++ = float2int8(v31 * vgetq_lane_f32(_scale0, 3));
                        *pp++ = float2int8(v40 * vgetq_lane_f32(_scale1, 0));
                        *pp++ = float2int8(v41 * vgetq_lane_f32(_scale1, 0));
                        *pp++ = float2int8(v50 * vgetq_lane_f32(_scale1, 1));
                        *pp++ = float2int8(v51 * vgetq_lane_f32(_scale1, 1));
                        *pp++ = float2int8(v60 * vgetq_lane_f32(_scale1, 2));
                        *pp++ = float2int8(v61 * vgetq_lane_f32(_scale1, 2));
                        *pp++ = float2int8(v70 * vgetq_lane_f32(_scale1, 3));
                        *pp++ = float2int8(v71 * vgetq_lane_f32(_scale1, 3));
                        p0 += 2;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        float v0 = float16_to_float32(p0[0]);
                        float v1 = float16_to_float32(p0[A_hstep]);
                        float v2 = float16_to_float32(p0[A_hstep * 2]);
                        float v3 = float16_to_float32(p0[A_hstep * 3]);
                        float v4 = float16_to_float32(p0[A_hstep * 4]);
                        float v5 = float16_to_float32(p0[A_hstep * 5]);
                        float v6 = float16_to_float32(p0[A_hstep * 6]);
                        float v7 = float16_to_float32(p0[A_hstep * 7]);
                        *pp++ = float2int8(v0 * vgetq_lane_f32(_scale0, 0));
                        *pp++ = float2int8(v1 * vgetq_lane_f32(_scale0, 1));
                        *pp++ = float2int8(v2 * vgetq_lane_f32(_scale0, 2));
                        *pp++ = float2int8(v3 * vgetq_lane_f32(_scale0, 3));
                        *pp++ = float2int8(v4 * vgetq_lane_f32(_scale1, 0));
                        *pp++ = float2int8(v5 * vgetq_lane_f32(_scale1, 1));
                        *pp++ = float2int8(v6 * vgetq_lane_f32(_scale1, 2));
                        *pp++ = float2int8(v7 * vgetq_lane_f32(_scale1, 3));
                        p0++;
                    }
                    pd += 8;
                }
            }
        }
#endif // __aarch64__
        for (; ii + 3 < max_ii; ii += 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * elempack;

            float32x4_t _v127 = vdupq_n_f32(127.f);
            float32x4_t _zero = vdupq_n_f32(0.f);

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                if (elempack == 4)
                {
                    float32x4_t _absmax = vdupq_n_f32(0.f);
                    const unsigned short* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        _absmax = vmaxq_f32(_absmax, vabsq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0a))));
                        p0a += 4;
                    }

#if __aarch64__
                    float32x4_t _scale = vdivq_f32(_v127, _absmax);
#else
                    float32x4_t _scale = div_ps(_v127, _absmax);
#endif
                    _scale = vbslq_f32(vceqq_f32(_absmax, _zero), _zero, _scale);
                    vst1q_f32(pd, vmulq_n_f32(_absmax, 1.f / 127.f));

                    int kk = 0;
                    for (; kk + 7 < max_kk0; kk += 8)
                    {
#if __ARM_FEATURE_DOTPROD
                        uint16x8x4_t _p = vld4q_u16(p0);

                        float32x4_t _p0 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_p.val[0])), _scale, 0);
                        float32x4_t _p1 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_p.val[1])), _scale, 1);
                        float32x4_t _p2 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_p.val[2])), _scale, 2);
                        float32x4_t _p3 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_p.val[3])), _scale, 3);
                        float32x4_t _p4 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_p.val[0])), _scale, 0);
                        float32x4_t _p5 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_p.val[1])), _scale, 1);
                        float32x4_t _p6 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_p.val[2])), _scale, 2);
                        float32x4_t _p7 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_p.val[3])), _scale, 3);

#if __ARM_FEATURE_MATMUL_INT8
                        int8x8_t _r0 = float2int8(_p0, _p4);
                        int8x8_t _r1 = float2int8(_p1, _p5);
                        int8x8_t _r2 = float2int8(_p2, _p6);
                        int8x8_t _r3 = float2int8(_p3, _p7);
#else  // __ARM_FEATURE_MATMUL_INT8
                        int8x8_t _r0 = float2int8(_p0, _p1);
                        int8x8_t _r1 = float2int8(_p2, _p3);
                        int8x8_t _r2 = float2int8(_p4, _p5);
                        int8x8_t _r3 = float2int8(_p6, _p7);
#endif // __ARM_FEATURE_MATMUL_INT8

                        vst1q_s8(pp, vcombine_s8(_r0, _r1));
                        vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
#else  // __ARM_FEATURE_DOTPROD
                        uint16x8_t _p = vld1q_u16(p0);
                        uint16x8_t _q = vld1q_u16(p0 + 8);
                        uint16x8_t _r = vld1q_u16(p0 + 16);
                        uint16x8_t _s = vld1q_u16(p0 + 24);
                        float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vget_low_u16(_p));
                        float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vget_high_u16(_p));
                        float32x4_t _p2 = vcvt_f32_f16((float16x4_t)vget_low_u16(_q));
                        float32x4_t _p3 = vcvt_f32_f16((float16x4_t)vget_high_u16(_q));
                        float32x4_t _p4 = vcvt_f32_f16((float16x4_t)vget_low_u16(_r));
                        float32x4_t _p5 = vcvt_f32_f16((float16x4_t)vget_high_u16(_r));
                        float32x4_t _p6 = vcvt_f32_f16((float16x4_t)vget_low_u16(_s));
                        float32x4_t _p7 = vcvt_f32_f16((float16x4_t)vget_high_u16(_s));

                        _p0 = vmulq_f32(_p0, _scale);
                        _p1 = vmulq_f32(_p1, _scale);
                        _p2 = vmulq_f32(_p2, _scale);
                        _p3 = vmulq_f32(_p3, _scale);
                        _p4 = vmulq_f32(_p4, _scale);
                        _p5 = vmulq_f32(_p5, _scale);
                        _p6 = vmulq_f32(_p6, _scale);
                        _p7 = vmulq_f32(_p7, _scale);

                        int8x16x2_t _r01;
                        _r01.val[0] = vcombine_s8(float2int8(_p0, _p2), float2int8(_p4, _p6));
                        _r01.val[1] = vcombine_s8(float2int8(_p1, _p3), float2int8(_p5, _p7));

                        vst2q_s8(pp, _r01);
#endif // __ARM_FEATURE_DOTPROD

                        pp += 32;
                        p0 += 32;
                    }
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
#if __ARM_FEATURE_DOTPROD
                        uint16x4x4_t _p = vld4_u16(p0);

                        float32x4_t _p0 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)_p.val[0]), _scale, 0);
                        float32x4_t _p1 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)_p.val[1]), _scale, 1);
                        float32x4_t _p2 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)_p.val[2]), _scale, 2);
                        float32x4_t _p3 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)_p.val[3]), _scale, 3);

                        int8x8_t _r0 = float2int8(_p0, _p1);
                        int8x8_t _r1 = float2int8(_p2, _p3);

                        vst1q_s8(pp, vcombine_s8(_r0, _r1));
#else  // __ARM_FEATURE_DOTPROD
                        uint16x8_t _p = vld1q_u16(p0);
                        uint16x8_t _q = vld1q_u16(p0 + 8);
                        float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vget_low_u16(_p));
                        float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vget_high_u16(_p));
                        float32x4_t _p2 = vcvt_f32_f16((float16x4_t)vget_low_u16(_q));
                        float32x4_t _p3 = vcvt_f32_f16((float16x4_t)vget_high_u16(_q));

                        _p0 = vmulq_f32(_p0, _scale);
                        _p1 = vmulq_f32(_p1, _scale);
                        _p2 = vmulq_f32(_p2, _scale);
                        _p3 = vmulq_f32(_p3, _scale);

                        int8x8x2_t _r01;
                        _r01.val[0] = float2int8(_p0, _p2);
                        _r01.val[1] = float2int8(_p1, _p3);

                        vst2_s8(pp, _r01);
#endif // __ARM_FEATURE_DOTPROD

                        pp += 16;
                        p0 += 16;
                    }
                    for (; kk + 1 < max_kk0; kk += 2)
                    {
                        uint16x8_t _p = vld1q_u16(p0);
                        float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vget_low_u16(_p));
                        float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vget_high_u16(_p));

                        _p0 = vmulq_f32(_p0, _scale);
                        _p1 = vmulq_f32(_p1, _scale);

                        float32x4x2_t _p01 = vzipq_f32(_p0, _p1);

                        int8x8_t _r01 = float2int8(_p01.val[0], _p01.val[1]);

                        vst1_s8(pp, _r01);

                        pp += 8;
                        p0 += 8;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vld1_u16(p0));
                        _p0 = vmulq_f32(_p0, _scale);
                        int8x8_t _r0 = float2int8(_p0, _p0);

                        pp[0] = vget_lane_s8(_r0, 0);
                        pp[1] = vget_lane_s8(_r0, 1);
                        pp[2] = vget_lane_s8(_r0, 2);
                        pp[3] = vget_lane_s8(_r0, 3);

                        pp += 4;
                        p0 += 4;
                    }

                    pd += 4;
                }

                if (elempack == 1)
                {
                    float32x4_t _absmax0 = vdupq_n_f32(0.f);
                    float32x4_t _absmax1 = vdupq_n_f32(0.f);
                    float32x4_t _absmax2 = vdupq_n_f32(0.f);
                    float32x4_t _absmax3 = vdupq_n_f32(0.f);
                    const unsigned short* p0a = p0;
                    int kk = 0;
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a));
                        _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(_p0));
                        float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a + A_hstep));
                        _absmax1 = vmaxq_f32(_absmax1, vabsq_f32(_p1));
                        float32x4_t _p2 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a + A_hstep * 2));
                        _absmax2 = vmaxq_f32(_absmax2, vabsq_f32(_p2));
                        float32x4_t _p3 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a + A_hstep * 3));
                        _absmax3 = vmaxq_f32(_absmax3, vabsq_f32(_p3));
                        p0a += 4;
                    }
                    float32x2_t _max0 = vpmax_f32(vget_low_f32(_absmax0), vget_high_f32(_absmax0));
                    float32x2_t _max1 = vpmax_f32(vget_low_f32(_absmax1), vget_high_f32(_absmax1));
                    float32x2_t _max2 = vpmax_f32(vget_low_f32(_absmax2), vget_high_f32(_absmax2));
                    float32x2_t _max3 = vpmax_f32(vget_low_f32(_absmax3), vget_high_f32(_absmax3));
                    _absmax0 = vcombine_f32(vpmax_f32(_max0, _max1), vpmax_f32(_max2, _max3));
                    for (; kk < max_kk0; kk++)
                    {
                        float32x4_t _p = vdupq_n_f32(float16_to_float32(p0a[0]));
                        _p = vsetq_lane_f32(float16_to_float32(p0a[A_hstep]), _p, 1);
                        _p = vsetq_lane_f32(float16_to_float32(p0a[A_hstep * 2]), _p, 2);
                        _p = vsetq_lane_f32(float16_to_float32(p0a[A_hstep * 3]), _p, 3);
                        _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(_p));
                        p0a++;
                    }
                    vst1q_f32(pd, vmulq_n_f32(_absmax0, 1.f / 127.f));

                    float32x4_t _zero = vdupq_n_f32(0.f);
#if __aarch64__
                    float32x4_t _scale = vdivq_f32(vdupq_n_f32(127.f), _absmax0);
#else
                    float32x4_t _scale = div_ps(vdupq_n_f32(127.f), _absmax0);
#endif
                    _scale = vbslq_f32(vceqq_f32(_absmax0, _zero), _zero, _scale);

                    kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                    for (; kk + 7 < max_kk0; kk += 8)
                    {
                        float32x4_t _p00 = vcvt_f32_f16((float16x4_t)vld1_u16(p0));
                        float32x4_t _p01 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 4));
                        float32x4_t _p10 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep));
                        float32x4_t _p11 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep + 4));
                        float32x4_t _p20 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 2));
                        float32x4_t _p21 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 2 + 4));
                        float32x4_t _p30 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 3));
                        float32x4_t _p31 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 3 + 4));
                        int8x8_t _r0 = float2int8(vmulq_laneq_f32(_p00, _scale, 0), vmulq_laneq_f32(_p01, _scale, 0));
                        int8x8_t _r1 = float2int8(vmulq_laneq_f32(_p10, _scale, 1), vmulq_laneq_f32(_p11, _scale, 1));
                        int8x8_t _r2 = float2int8(vmulq_laneq_f32(_p20, _scale, 2), vmulq_laneq_f32(_p21, _scale, 2));
                        int8x8_t _r3 = float2int8(vmulq_laneq_f32(_p30, _scale, 3), vmulq_laneq_f32(_p31, _scale, 3));
                        vst1q_s8(pp, vcombine_s8(_r0, _r1));
                        vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
                        pp += 32;
                        p0 += 8;
                    }
#endif // __ARM_FEATURE_MATMUL_INT8
#endif // __ARM_FEATURE_DOTPROD
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vld1_u16(p0));
                        float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep));
                        float32x4_t _p2 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 2));
                        float32x4_t _p3 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 3));
#if __aarch64__
                        int8x8_t _r01 = float2int8(vmulq_laneq_f32(_p0, _scale, 0), vmulq_laneq_f32(_p1, _scale, 1));
                        int8x8_t _r23 = float2int8(vmulq_laneq_f32(_p2, _scale, 2), vmulq_laneq_f32(_p3, _scale, 3));
#else
                        int8x8_t _r01 = float2int8(vmulq_lane_f32(_p0, vget_low_f32(_scale), 0), vmulq_lane_f32(_p1, vget_low_f32(_scale), 1));
                        int8x8_t _r23 = float2int8(vmulq_lane_f32(_p2, vget_high_f32(_scale), 0), vmulq_lane_f32(_p3, vget_high_f32(_scale), 1));
#endif
#if __ARM_FEATURE_DOTPROD
                        vst1q_s8(pp, vcombine_s8(_r01, _r23));
#else
                        int16x8_t _r0123 = vreinterpretq_s16_s8(vcombine_s8(_r01, _r23));
                        int16x8x2_t _r02 = vuzpq_s16(_r0123, _r0123);
                        vst1q_s8(pp, vreinterpretq_s8_s16(vcombine_s16(vget_low_s16(_r02.val[0]), vget_low_s16(_r02.val[1]))));
#endif
                        pp += 16;
                        p0 += 4;
                    }
                    for (; kk + 1 < max_kk0; kk += 2)
                    {
                        float v00 = float16_to_float32(p0[0]);
                        float v01 = float16_to_float32(p0[1]);
                        float v10 = float16_to_float32(p0[A_hstep]);
                        float v11 = float16_to_float32(p0[A_hstep + 1]);
                        float v20 = float16_to_float32(p0[A_hstep * 2]);
                        float v21 = float16_to_float32(p0[A_hstep * 2 + 1]);
                        float v30 = float16_to_float32(p0[A_hstep * 3]);
                        float v31 = float16_to_float32(p0[A_hstep * 3 + 1]);
                        *pp++ = float2int8(v00 * vgetq_lane_f32(_scale, 0));
                        *pp++ = float2int8(v01 * vgetq_lane_f32(_scale, 0));
                        *pp++ = float2int8(v10 * vgetq_lane_f32(_scale, 1));
                        *pp++ = float2int8(v11 * vgetq_lane_f32(_scale, 1));
                        *pp++ = float2int8(v20 * vgetq_lane_f32(_scale, 2));
                        *pp++ = float2int8(v21 * vgetq_lane_f32(_scale, 2));
                        *pp++ = float2int8(v30 * vgetq_lane_f32(_scale, 3));
                        *pp++ = float2int8(v31 * vgetq_lane_f32(_scale, 3));
                        p0 += 2;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        float v0 = float16_to_float32(p0[0]);
                        float v1 = float16_to_float32(p0[A_hstep]);
                        float v2 = float16_to_float32(p0[A_hstep * 2]);
                        float v3 = float16_to_float32(p0[A_hstep * 3]);
                        *pp++ = float2int8(v0 * vgetq_lane_f32(_scale, 0));
                        *pp++ = float2int8(v1 * vgetq_lane_f32(_scale, 1));
                        *pp++ = float2int8(v2 * vgetq_lane_f32(_scale, 2));
                        *pp++ = float2int8(v3 * vgetq_lane_f32(_scale, 3));
                        p0++;
                    }
                    pd += 4;
                }
            }
        }
#endif // __ARM_NEON
        for (; ii + 1 < max_ii; ii += 2)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                float absmax0 = 0.f;
                float absmax1 = 0.f;
                const unsigned short* p0a = p0;
                int kk = 0;
#if __ARM_NEON
                float32x4_t _absmax0 = vdupq_n_f32(0.f);
                float32x4_t _absmax1 = vdupq_n_f32(0.f);
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a));
                    _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(_p0));
                    float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a + A_hstep));
                    _absmax1 = vmaxq_f32(_absmax1, vabsq_f32(_p1));
                    p0a += 4;
                }
#if __aarch64__
                absmax0 = vmaxvq_f32(_absmax0);
                absmax1 = vmaxvq_f32(_absmax1);
#else
                float32x2_t _max0 = vmax_f32(vget_low_f32(_absmax0), vget_high_f32(_absmax0));
                float32x2_t _max1 = vmax_f32(vget_low_f32(_absmax1), vget_high_f32(_absmax1));
                _max0 = vpmax_f32(_max0, _max0);
                _max1 = vpmax_f32(_max1, _max1);
                absmax0 = vget_lane_f32(_max0, 0);
                absmax1 = vget_lane_f32(_max1, 0);
#endif
#endif // __ARM_NEON

                for (; kk < max_kk0; kk++)
                {
                    float v0 = float16_to_float32(p0a[0]);
                    float v1 = float16_to_float32(p0a[A_hstep]);
                    absmax0 = std::max(absmax0, fabsf(v0));
                    absmax1 = std::max(absmax1, fabsf(v1));
                    p0a++;
                }

                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
                const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;

                kk = 0;
#if __ARM_NEON
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    float32x4_t _p00 = vcvt_f32_f16((float16x4_t)vld1_u16(p0));
                    float32x4_t _p01 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 4));
                    float32x4_t _p10 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep));
                    float32x4_t _p11 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep + 4));
                    int8x8_t _r0 = float2int8(vmulq_n_f32(_p00, scale0), vmulq_n_f32(_p01, scale0));
                    int8x8_t _r1 = float2int8(vmulq_n_f32(_p10, scale1), vmulq_n_f32(_p11, scale1));
                    vst1q_s8(pp, vcombine_s8(_r0, _r1));
                    pp += 16;
                    p0 += 8;
                }
#endif // __ARM_FEATURE_MATMUL_INT8
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vld1_u16(p0));
                    float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep));
                    int8x8_t _r01 = float2int8(vmulq_n_f32(_p0, scale0), vmulq_n_f32(_p1, scale1));
#if __ARM_FEATURE_DOTPROD
                    vst1_s8(pp, _r01);
#else
                    int16x4_t _r01_s16 = vreinterpret_s16_s8(_r01);
                    int16x4_t _r10_s16 = vext_s16(_r01_s16, _r01_s16, 2);
                    vst1_s8(pp, vreinterpret_s8_s16(vzip_s16(_r01_s16, _r10_s16).val[0]));
#endif
                    pp += 8;
                    p0 += 4;
                }
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    float v00 = float16_to_float32(p0[0]);
                    float v01 = float16_to_float32(p0[1]);
                    float v10 = float16_to_float32(p0[A_hstep]);
                    float v11 = float16_to_float32(p0[A_hstep + 1]);
                    *pp++ = float2int8(v00 * scale0);
                    *pp++ = float2int8(v01 * scale0);
                    *pp++ = float2int8(v10 * scale1);
                    *pp++ = float2int8(v11 * scale1);
                    p0 += 2;
                }
#endif // __ARM_NEON
                for (; kk < max_kk0; kk++)
                {
                    float v0 = float16_to_float32(p0[0]);
                    float v1 = float16_to_float32(p0[A_hstep]);
                    *pp++ = float2int8(v0 * scale0);
                    *pp++ = float2int8(v1 * scale1);
                    p0++;
                }

                pd += 2;
            }
        }
        for (; ii < max_ii; ii++)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                const unsigned short* p0a = p0;

                float absmax = 0.f;
                int kk = 0;
#if __ARM_NEON
                float32x4_t _absmax = vdupq_n_f32(0.f);
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float32x4_t _p = vcvt_f32_f16((float16x4_t)vld1_u16(p0a));
                    _absmax = vmaxq_f32(_absmax, vabsq_f32(_p));
                    p0a += 4;
                }
#if __aarch64__
                absmax = vmaxvq_f32(_absmax);
#else
                float32x2_t _max2 = vmax_f32(vget_low_f32(_absmax), vget_high_f32(_absmax));
                _max2 = vpmax_f32(_max2, _max2);
                absmax = vget_lane_f32(_max2, 0);
#endif
#endif // __ARM_NEON
                for (; kk < max_kk0; kk++)
                {
                    float v = float16_to_float32(*p0a++);
                    absmax = std::max(absmax, fabsf(v));
                }

                if (absmax == 0.f)
                {
                    *pd++ = 0.f;
                    for (int kk0 = 0; kk0 < max_kk0; kk0++)
                        *pp++ = 0;
                    p0 += max_kk0;
                    continue;
                }

                const float scale = 127.f / absmax;
                *pd++ = absmax / 127.f;

                kk = 0;
#if __ARM_NEON
                float32x4_t _scale = vdupq_n_f32(scale);
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vld1_u16(p0));
                    float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 4));
                    vst1_s8(pp, float2int8(vmulq_f32(_p0, _scale), vmulq_f32(_p1, _scale)));
                    pp += 8;
                    p0 += 8;
                }
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float32x4_t _p = vcvt_f32_f16((float16x4_t)vld1_u16(p0));
                    int8x8_t _r = float2int8(vmulq_f32(_p, _scale), vmulq_f32(_p, _scale));
                    vst1_lane_s32((int*)pp, vreinterpret_s32_s8(_r), 0);
                    pp += 4;
                    p0 += 4;
                }
#endif // __ARM_NEON
                for (; kk < max_kk0; kk++)
                {
                    float v = float16_to_float32(*p0++);
                    *pp++ = float2int8(v * scale);
                }
            }
        }

        return;
    }

    int ii = 0;
#if __ARM_NEON
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * elempack;
        const float* ps = (const float*)input_scales + k;

        float32x4_t _v127 = vdupq_n_f32(127.f);
        float32x4_t _zero = vdupq_n_f32(0.f);

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
            if (elempack == 8)
            {
                float32x4_t _absmax0 = _zero;
                float32x4_t _absmax1 = _zero;
                const unsigned short* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a));
                    float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a + 4));
                    float32x4_t _s = vdupq_n_f32(*psa++);
                    _absmax0 = vmaxq_f32(_absmax0, vmulq_f32(vabsq_f32(_p0), _s));
                    _absmax1 = vmaxq_f32(_absmax1, vmulq_f32(vabsq_f32(_p1), _s));
                    p0a += 8;
                }

                vst1q_f32(pd, vmulq_n_f32(_absmax0, 1.f / 127.f));
                vst1q_f32(pd + 4, vmulq_n_f32(_absmax1, 1.f / 127.f));
                float32x4_t _scale0 = _v127;
                float32x4_t _scale1 = _v127;
                _scale0 = vdivq_f32(_scale0, _absmax0);
                _scale1 = vdivq_f32(_scale1, _absmax1);
                _scale0 = vbslq_f32(vceqq_f32(_absmax0, _zero), _zero, _scale0);
                _scale1 = vbslq_f32(vceqq_f32(_absmax1, _zero), _zero, _scale1);

                int kk = 0;
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    uint16x8_t _p = vld1q_u16(p0);
                    uint16x8_t _q = vld1q_u16(p0 + 8);
                    uint16x8_t _r = vld1q_u16(p0 + 16);
                    uint16x8_t _s = vld1q_u16(p0 + 24);
                    uint16x8_t _t = vld1q_u16(p0 + 32);
                    uint16x8_t _u = vld1q_u16(p0 + 40);
                    uint16x8_t _v = vld1q_u16(p0 + 48);
                    uint16x8_t _w = vld1q_u16(p0 + 56);
                    float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vget_low_u16(_p));
                    float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vget_high_u16(_p));
                    float32x4_t _p2 = vcvt_f32_f16((float16x4_t)vget_low_u16(_q));
                    float32x4_t _p3 = vcvt_f32_f16((float16x4_t)vget_high_u16(_q));
                    float32x4_t _p4 = vcvt_f32_f16((float16x4_t)vget_low_u16(_r));
                    float32x4_t _p5 = vcvt_f32_f16((float16x4_t)vget_high_u16(_r));
                    float32x4_t _p6 = vcvt_f32_f16((float16x4_t)vget_low_u16(_s));
                    float32x4_t _p7 = vcvt_f32_f16((float16x4_t)vget_high_u16(_s));
                    float32x4_t _p8 = vcvt_f32_f16((float16x4_t)vget_low_u16(_t));
                    float32x4_t _p9 = vcvt_f32_f16((float16x4_t)vget_high_u16(_t));
                    float32x4_t _pa = vcvt_f32_f16((float16x4_t)vget_low_u16(_u));
                    float32x4_t _pb = vcvt_f32_f16((float16x4_t)vget_high_u16(_u));
                    float32x4_t _pc = vcvt_f32_f16((float16x4_t)vget_low_u16(_v));
                    float32x4_t _pd = vcvt_f32_f16((float16x4_t)vget_high_u16(_v));
                    float32x4_t _pe = vcvt_f32_f16((float16x4_t)vget_low_u16(_w));
                    float32x4_t _pf = vcvt_f32_f16((float16x4_t)vget_high_u16(_w));

                    _p0 = vmulq_f32(vmulq_n_f32(_p0, ps[0]), _scale0);
                    _p1 = vmulq_f32(vmulq_n_f32(_p1, ps[0]), _scale1);
                    _p2 = vmulq_f32(vmulq_n_f32(_p2, ps[1]), _scale0);
                    _p3 = vmulq_f32(vmulq_n_f32(_p3, ps[1]), _scale1);
                    _p4 = vmulq_f32(vmulq_n_f32(_p4, ps[2]), _scale0);
                    _p5 = vmulq_f32(vmulq_n_f32(_p5, ps[2]), _scale1);
                    _p6 = vmulq_f32(vmulq_n_f32(_p6, ps[3]), _scale0);
                    _p7 = vmulq_f32(vmulq_n_f32(_p7, ps[3]), _scale1);
                    _p8 = vmulq_f32(vmulq_n_f32(_p8, ps[4]), _scale0);
                    _p9 = vmulq_f32(vmulq_n_f32(_p9, ps[4]), _scale1);
                    _pa = vmulq_f32(vmulq_n_f32(_pa, ps[5]), _scale0);
                    _pb = vmulq_f32(vmulq_n_f32(_pb, ps[5]), _scale1);
                    _pc = vmulq_f32(vmulq_n_f32(_pc, ps[6]), _scale0);
                    _pd = vmulq_f32(vmulq_n_f32(_pd, ps[6]), _scale1);
                    _pe = vmulq_f32(vmulq_n_f32(_pe, ps[7]), _scale0);
                    _pf = vmulq_f32(vmulq_n_f32(_pf, ps[7]), _scale1);

#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                    int8x8x2_t _p04 = vzip_s8(float2int8(_p0, _p1), float2int8(_p8, _p9));
                    int8x8x2_t _p15 = vzip_s8(float2int8(_p2, _p3), float2int8(_pa, _pb));
                    int8x8x2_t _p26 = vzip_s8(float2int8(_p4, _p5), float2int8(_pc, _pd));
                    int8x8x2_t _p37 = vzip_s8(float2int8(_p6, _p7), float2int8(_pe, _pf));

                    int8x16x4_t _rr;
                    _rr.val[0] = vcombine_s8(_p04.val[0], _p04.val[1]);
                    _rr.val[1] = vcombine_s8(_p15.val[0], _p15.val[1]);
                    _rr.val[2] = vcombine_s8(_p26.val[0], _p26.val[1]);
                    _rr.val[3] = vcombine_s8(_p37.val[0], _p37.val[1]);
#else  // __ARM_FEATURE_MATMUL_INT8
                    int8x16x4_t _rr;
                    _rr.val[0] = vcombine_s8(float2int8(_p0, _p1), float2int8(_p8, _p9));
                    _rr.val[1] = vcombine_s8(float2int8(_p2, _p3), float2int8(_pa, _pb));
                    _rr.val[2] = vcombine_s8(float2int8(_p4, _p5), float2int8(_pc, _pd));
                    _rr.val[3] = vcombine_s8(float2int8(_p6, _p7), float2int8(_pe, _pf));
#endif // __ARM_FEATURE_MATMUL_INT8

                    vst4q_s8(pp, _rr);
#else  // __ARM_FEATURE_DOTPROD
                    int8x16x2_t _r01;
                    _r01.val[0] = vcombine_s8(float2int8(_p0, _p1), float2int8(_p4, _p5));
                    _r01.val[1] = vcombine_s8(float2int8(_p2, _p3), float2int8(_p6, _p7));
                    int8x16x2_t _r23;
                    _r23.val[0] = vcombine_s8(float2int8(_p8, _p9), float2int8(_pc, _pd));
                    _r23.val[1] = vcombine_s8(float2int8(_pa, _pb), float2int8(_pe, _pf));

                    vst2q_s8(pp, _r01);
                    vst2q_s8(pp + 32, _r23);
#endif // __ARM_FEATURE_DOTPROD

                    pp += 64;
                    p0 += 64;
                    ps += 8;
                }
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    uint16x8_t _p = vld1q_u16(p0);
                    uint16x8_t _q = vld1q_u16(p0 + 8);
                    uint16x8_t _r = vld1q_u16(p0 + 16);
                    uint16x8_t _s = vld1q_u16(p0 + 24);
                    float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vget_low_u16(_p));
                    float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vget_high_u16(_p));
                    float32x4_t _p2 = vcvt_f32_f16((float16x4_t)vget_low_u16(_q));
                    float32x4_t _p3 = vcvt_f32_f16((float16x4_t)vget_high_u16(_q));
                    float32x4_t _p4 = vcvt_f32_f16((float16x4_t)vget_low_u16(_r));
                    float32x4_t _p5 = vcvt_f32_f16((float16x4_t)vget_high_u16(_r));
                    float32x4_t _p6 = vcvt_f32_f16((float16x4_t)vget_low_u16(_s));
                    float32x4_t _p7 = vcvt_f32_f16((float16x4_t)vget_high_u16(_s));

                    _p0 = vmulq_f32(vmulq_n_f32(_p0, ps[0]), _scale0);
                    _p1 = vmulq_f32(vmulq_n_f32(_p1, ps[0]), _scale1);
                    _p2 = vmulq_f32(vmulq_n_f32(_p2, ps[1]), _scale0);
                    _p3 = vmulq_f32(vmulq_n_f32(_p3, ps[1]), _scale1);
                    _p4 = vmulq_f32(vmulq_n_f32(_p4, ps[2]), _scale0);
                    _p5 = vmulq_f32(vmulq_n_f32(_p5, ps[2]), _scale1);
                    _p6 = vmulq_f32(vmulq_n_f32(_p6, ps[3]), _scale0);
                    _p7 = vmulq_f32(vmulq_n_f32(_p7, ps[3]), _scale1);

#if __ARM_FEATURE_DOTPROD
                    int8x8x4_t _r0123;
                    _r0123.val[0] = float2int8(_p0, _p1);
                    _r0123.val[1] = float2int8(_p2, _p3);
                    _r0123.val[2] = float2int8(_p4, _p5);
                    _r0123.val[3] = float2int8(_p6, _p7);

                    vst4_s8(pp, _r0123);
#else  // __ARM_FEATURE_DOTPROD
                    int8x16x2_t _r01;
                    _r01.val[0] = vcombine_s8(float2int8(_p0, _p1), float2int8(_p4, _p5));
                    _r01.val[1] = vcombine_s8(float2int8(_p2, _p3), float2int8(_p6, _p7));

                    vst2q_s8(pp, _r01);
#endif // __ARM_FEATURE_DOTPROD

                    pp += 32;
                    p0 += 32;
                    ps += 4;
                }
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    uint16x8_t _p01 = vld1q_u16(p0);
                    uint16x8_t _p23 = vld1q_u16(p0 + 8);

                    float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vget_low_u16(_p01));
                    float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vget_high_u16(_p01));
                    float32x4_t _p2 = vcvt_f32_f16((float16x4_t)vget_low_u16(_p23));
                    float32x4_t _p3 = vcvt_f32_f16((float16x4_t)vget_high_u16(_p23));

                    _p0 = vmulq_f32(vmulq_n_f32(_p0, ps[0]), _scale0);
                    _p1 = vmulq_f32(vmulq_n_f32(_p1, ps[0]), _scale1);
                    _p2 = vmulq_f32(vmulq_n_f32(_p2, ps[1]), _scale0);
                    _p3 = vmulq_f32(vmulq_n_f32(_p3, ps[1]), _scale1);

                    int8x8x2_t _r01;
                    _r01.val[0] = float2int8(_p0, _p1);
                    _r01.val[1] = float2int8(_p2, _p3);

                    vst2_s8(pp, _r01);

                    pp += 16;
                    p0 += 16;
                    ps += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    uint16x8_t _p01 = vld1q_u16(p0);
                    float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vget_low_u16(_p01));
                    float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vget_high_u16(_p01));

                    _p0 = vmulq_f32(vmulq_n_f32(_p0, ps[0]), _scale0);
                    _p1 = vmulq_f32(vmulq_n_f32(_p1, ps[0]), _scale1);

                    int8x8_t _r01 = float2int8(_p0, _p1);

                    vst1_s8(pp, _r01);

                    pp += 8;
                    p0 += 8;
                    ps++;
                }
                pd += 8;
            }

            if (elempack == 4)
            {
                float32x4_t _absmax0 = _zero;
                float32x4_t _absmax1 = _zero;
                const unsigned short* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a));
                    float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a + A_hstep * 4));
                    float32x4_t _s = vdupq_n_f32(*psa++);
                    _absmax0 = vmaxq_f32(_absmax0, vmulq_f32(vabsq_f32(_p0), _s));
                    _absmax1 = vmaxq_f32(_absmax1, vmulq_f32(vabsq_f32(_p1), _s));
                    p0a += 4;
                }

                vst1q_f32(pd, vmulq_n_f32(_absmax0, 1.f / 127.f));
                vst1q_f32(pd + 4, vmulq_n_f32(_absmax1, 1.f / 127.f));
                float32x4_t _scale0 = vdivq_f32(_v127, _absmax0);
                float32x4_t _scale1 = vdivq_f32(_v127, _absmax1);
                _scale0 = vbslq_f32(vceqq_f32(_absmax0, _zero), _zero, _scale0);
                _scale1 = vbslq_f32(vceqq_f32(_absmax1, _zero), _zero, _scale1);

                int kk = 0;
                for (; kk + 7 < max_kk0; kk += 8)
                {
#if __ARM_FEATURE_DOTPROD
                    uint16x8x4_t _p = vld4q_u16(p0);
                    uint16x8x4_t _q = vld4q_u16(p0 + A_hstep * 4);
                    float32x4_t _s0 = vld1q_f32(ps);
                    float32x4_t _s1 = vld1q_f32(ps + 4);

                    float32x4_t _p0 = vmulq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_p.val[0])), _s0);
                    float32x4_t _p1 = vmulq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_p.val[1])), _s0);
                    float32x4_t _p2 = vmulq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_p.val[2])), _s0);
                    float32x4_t _p3 = vmulq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_p.val[3])), _s0);
                    float32x4_t _p4 = vmulq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_p.val[0])), _s1);
                    float32x4_t _p5 = vmulq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_p.val[1])), _s1);
                    float32x4_t _p6 = vmulq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_p.val[2])), _s1);
                    float32x4_t _p7 = vmulq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_p.val[3])), _s1);
                    float32x4_t _p8 = vmulq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_q.val[0])), _s0);
                    float32x4_t _p9 = vmulq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_q.val[1])), _s0);
                    float32x4_t _pa = vmulq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_q.val[2])), _s0);
                    float32x4_t _pb = vmulq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_q.val[3])), _s0);
                    float32x4_t _pc = vmulq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_q.val[0])), _s1);
                    float32x4_t _pd = vmulq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_q.val[1])), _s1);
                    float32x4_t _pe = vmulq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_q.val[2])), _s1);
                    float32x4_t _pf = vmulq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_q.val[3])), _s1);

                    _p0 = vmulq_laneq_f32(_p0, _scale0, 0);
                    _p1 = vmulq_laneq_f32(_p1, _scale0, 1);
                    _p2 = vmulq_laneq_f32(_p2, _scale0, 2);
                    _p3 = vmulq_laneq_f32(_p3, _scale0, 3);
                    _p4 = vmulq_laneq_f32(_p4, _scale0, 0);
                    _p5 = vmulq_laneq_f32(_p5, _scale0, 1);
                    _p6 = vmulq_laneq_f32(_p6, _scale0, 2);
                    _p7 = vmulq_laneq_f32(_p7, _scale0, 3);
                    _p8 = vmulq_laneq_f32(_p8, _scale1, 0);
                    _p9 = vmulq_laneq_f32(_p9, _scale1, 1);
                    _pa = vmulq_laneq_f32(_pa, _scale1, 2);
                    _pb = vmulq_laneq_f32(_pb, _scale1, 3);
                    _pc = vmulq_laneq_f32(_pc, _scale1, 0);
                    _pd = vmulq_laneq_f32(_pd, _scale1, 1);
                    _pe = vmulq_laneq_f32(_pe, _scale1, 2);
                    _pf = vmulq_laneq_f32(_pf, _scale1, 3);

#if __ARM_FEATURE_MATMUL_INT8
                    int8x8_t _r0 = float2int8(_p0, _p4);
                    int8x8_t _r1 = float2int8(_p1, _p5);
                    int8x8_t _r2 = float2int8(_p2, _p6);
                    int8x8_t _r3 = float2int8(_p3, _p7);
                    int8x8_t _r4 = float2int8(_p8, _pc);
                    int8x8_t _r5 = float2int8(_p9, _pd);
                    int8x8_t _r6 = float2int8(_pa, _pe);
                    int8x8_t _r7 = float2int8(_pb, _pf);
#else  // __ARM_FEATURE_MATMUL_INT8
                    int8x8_t _r0 = float2int8(_p0, _p1);
                    int8x8_t _r1 = float2int8(_p2, _p3);
                    int8x8_t _r2 = float2int8(_p8, _p9);
                    int8x8_t _r3 = float2int8(_pa, _pb);
                    int8x8_t _r4 = float2int8(_p4, _p5);
                    int8x8_t _r5 = float2int8(_p6, _p7);
                    int8x8_t _r6 = float2int8(_pc, _pd);
                    int8x8_t _r7 = float2int8(_pe, _pf);
#endif // __ARM_FEATURE_MATMUL_INT8

                    vst1q_s8(pp, vcombine_s8(_r0, _r1));
                    vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
                    vst1q_s8(pp + 32, vcombine_s8(_r4, _r5));
                    vst1q_s8(pp + 48, vcombine_s8(_r6, _r7));
#else  // __ARM_FEATURE_DOTPROD
                    uint16x8_t _p = vld1q_u16(p0);
                    uint16x8_t _q = vld1q_u16(p0 + 8);
                    uint16x8_t _r = vld1q_u16(p0 + 16);
                    uint16x8_t _s = vld1q_u16(p0 + 24);
                    uint16x8_t _t = vld1q_u16(p0 + A_hstep * 4);
                    uint16x8_t _u = vld1q_u16(p0 + A_hstep * 4 + 8);
                    uint16x8_t _v = vld1q_u16(p0 + A_hstep * 4 + 16);
                    uint16x8_t _w = vld1q_u16(p0 + A_hstep * 4 + 24);
                    float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vget_low_u16(_p));
                    float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vget_high_u16(_p));
                    float32x4_t _p2 = vcvt_f32_f16((float16x4_t)vget_low_u16(_q));
                    float32x4_t _p3 = vcvt_f32_f16((float16x4_t)vget_high_u16(_q));
                    float32x4_t _p4 = vcvt_f32_f16((float16x4_t)vget_low_u16(_r));
                    float32x4_t _p5 = vcvt_f32_f16((float16x4_t)vget_high_u16(_r));
                    float32x4_t _p6 = vcvt_f32_f16((float16x4_t)vget_low_u16(_s));
                    float32x4_t _p7 = vcvt_f32_f16((float16x4_t)vget_high_u16(_s));
                    float32x4_t _p8 = vcvt_f32_f16((float16x4_t)vget_low_u16(_t));
                    float32x4_t _p9 = vcvt_f32_f16((float16x4_t)vget_high_u16(_t));
                    float32x4_t _pa = vcvt_f32_f16((float16x4_t)vget_low_u16(_u));
                    float32x4_t _pb = vcvt_f32_f16((float16x4_t)vget_high_u16(_u));
                    float32x4_t _pc = vcvt_f32_f16((float16x4_t)vget_low_u16(_v));
                    float32x4_t _pd = vcvt_f32_f16((float16x4_t)vget_high_u16(_v));
                    float32x4_t _pe = vcvt_f32_f16((float16x4_t)vget_low_u16(_w));
                    float32x4_t _pf = vcvt_f32_f16((float16x4_t)vget_high_u16(_w));

                    _p0 = vmulq_f32(vmulq_n_f32(_p0, ps[0]), _scale0);
                    _p1 = vmulq_f32(vmulq_n_f32(_p1, ps[1]), _scale0);
                    _p2 = vmulq_f32(vmulq_n_f32(_p2, ps[2]), _scale0);
                    _p3 = vmulq_f32(vmulq_n_f32(_p3, ps[3]), _scale0);
                    _p4 = vmulq_f32(vmulq_n_f32(_p4, ps[4]), _scale0);
                    _p5 = vmulq_f32(vmulq_n_f32(_p5, ps[5]), _scale0);
                    _p6 = vmulq_f32(vmulq_n_f32(_p6, ps[6]), _scale0);
                    _p7 = vmulq_f32(vmulq_n_f32(_p7, ps[7]), _scale0);
                    _p8 = vmulq_f32(vmulq_n_f32(_p8, ps[0]), _scale1);
                    _p9 = vmulq_f32(vmulq_n_f32(_p9, ps[1]), _scale1);
                    _pa = vmulq_f32(vmulq_n_f32(_pa, ps[2]), _scale1);
                    _pb = vmulq_f32(vmulq_n_f32(_pb, ps[3]), _scale1);
                    _pc = vmulq_f32(vmulq_n_f32(_pc, ps[4]), _scale1);
                    _pd = vmulq_f32(vmulq_n_f32(_pd, ps[5]), _scale1);
                    _pe = vmulq_f32(vmulq_n_f32(_pe, ps[6]), _scale1);
                    _pf = vmulq_f32(vmulq_n_f32(_pf, ps[7]), _scale1);

                    int8x16x2_t _r01;
                    _r01.val[0] = vcombine_s8(float2int8(_p0, _p8), float2int8(_p2, _pa));
                    _r01.val[1] = vcombine_s8(float2int8(_p1, _p9), float2int8(_p3, _pb));
                    int8x16x2_t _r23;
                    _r23.val[0] = vcombine_s8(float2int8(_p4, _pc), float2int8(_p6, _pe));
                    _r23.val[1] = vcombine_s8(float2int8(_p5, _pd), float2int8(_p7, _pf));

                    vst2q_s8(pp, _r01);
                    vst2q_s8(pp + 32, _r23);
#endif // __ARM_FEATURE_DOTPROD

                    pp += 64;
                    p0 += 32;
                    ps += 8;
                }
                for (; kk + 3 < max_kk0; kk += 4)
                {
#if __ARM_FEATURE_DOTPROD
                    uint16x4x4_t _p = vld4_u16(p0);
                    uint16x4x4_t _q = vld4_u16(p0 + A_hstep * 4);
                    float32x4_t _s = vld1q_f32(ps);

                    float32x4_t _p0 = vmulq_f32(vcvt_f32_f16((float16x4_t)_p.val[0]), _s);
                    float32x4_t _p1 = vmulq_f32(vcvt_f32_f16((float16x4_t)_p.val[1]), _s);
                    float32x4_t _p2 = vmulq_f32(vcvt_f32_f16((float16x4_t)_p.val[2]), _s);
                    float32x4_t _p3 = vmulq_f32(vcvt_f32_f16((float16x4_t)_p.val[3]), _s);
                    float32x4_t _p4 = vmulq_f32(vcvt_f32_f16((float16x4_t)_q.val[0]), _s);
                    float32x4_t _p5 = vmulq_f32(vcvt_f32_f16((float16x4_t)_q.val[1]), _s);
                    float32x4_t _p6 = vmulq_f32(vcvt_f32_f16((float16x4_t)_q.val[2]), _s);
                    float32x4_t _p7 = vmulq_f32(vcvt_f32_f16((float16x4_t)_q.val[3]), _s);

                    _p0 = vmulq_laneq_f32(_p0, _scale0, 0);
                    _p1 = vmulq_laneq_f32(_p1, _scale0, 1);
                    _p2 = vmulq_laneq_f32(_p2, _scale0, 2);
                    _p3 = vmulq_laneq_f32(_p3, _scale0, 3);
                    _p4 = vmulq_laneq_f32(_p4, _scale1, 0);
                    _p5 = vmulq_laneq_f32(_p5, _scale1, 1);
                    _p6 = vmulq_laneq_f32(_p6, _scale1, 2);
                    _p7 = vmulq_laneq_f32(_p7, _scale1, 3);

                    int8x8_t _r0 = float2int8(_p0, _p1);
                    int8x8_t _r1 = float2int8(_p2, _p3);
                    int8x8_t _r2 = float2int8(_p4, _p5);
                    int8x8_t _r3 = float2int8(_p6, _p7);

                    vst1q_s8(pp, vcombine_s8(_r0, _r1));
                    vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
#else  // __ARM_FEATURE_DOTPROD
                    uint16x8_t _p = vld1q_u16(p0);
                    uint16x8_t _q = vld1q_u16(p0 + 8);
                    uint16x8_t _r = vld1q_u16(p0 + A_hstep * 4);
                    uint16x8_t _s = vld1q_u16(p0 + A_hstep * 4 + 8);
                    float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vget_low_u16(_p));
                    float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vget_high_u16(_p));
                    float32x4_t _p2 = vcvt_f32_f16((float16x4_t)vget_low_u16(_q));
                    float32x4_t _p3 = vcvt_f32_f16((float16x4_t)vget_high_u16(_q));
                    float32x4_t _p4 = vcvt_f32_f16((float16x4_t)vget_low_u16(_r));
                    float32x4_t _p5 = vcvt_f32_f16((float16x4_t)vget_high_u16(_r));
                    float32x4_t _p6 = vcvt_f32_f16((float16x4_t)vget_low_u16(_s));
                    float32x4_t _p7 = vcvt_f32_f16((float16x4_t)vget_high_u16(_s));

                    _p0 = vmulq_f32(vmulq_n_f32(_p0, ps[0]), _scale0);
                    _p1 = vmulq_f32(vmulq_n_f32(_p1, ps[1]), _scale0);
                    _p2 = vmulq_f32(vmulq_n_f32(_p2, ps[2]), _scale0);
                    _p3 = vmulq_f32(vmulq_n_f32(_p3, ps[3]), _scale0);
                    _p4 = vmulq_f32(vmulq_n_f32(_p4, ps[0]), _scale1);
                    _p5 = vmulq_f32(vmulq_n_f32(_p5, ps[1]), _scale1);
                    _p6 = vmulq_f32(vmulq_n_f32(_p6, ps[2]), _scale1);
                    _p7 = vmulq_f32(vmulq_n_f32(_p7, ps[3]), _scale1);

                    int8x16x2_t _r01;
                    _r01.val[0] = vcombine_s8(float2int8(_p0, _p4), float2int8(_p2, _p6));
                    _r01.val[1] = vcombine_s8(float2int8(_p1, _p5), float2int8(_p3, _p7));

                    vst2q_s8(pp, _r01);
#endif // __ARM_FEATURE_DOTPROD

                    pp += 32;
                    p0 += 16;
                    ps += 4;
                }
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    uint16x8_t _p = vld1q_u16(p0);
                    uint16x8_t _q = vld1q_u16(p0 + A_hstep * 4);

                    float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vget_low_u16(_p));
                    float32x4_t _p0n = vcvt_f32_f16((float16x4_t)vget_high_u16(_p));
                    float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vget_low_u16(_q));
                    float32x4_t _p1n = vcvt_f32_f16((float16x4_t)vget_high_u16(_q));

                    _p0 = vmulq_f32(vmulq_n_f32(_p0, ps[0]), _scale0);
                    _p0n = vmulq_f32(vmulq_n_f32(_p0n, ps[1]), _scale0);
                    _p1 = vmulq_f32(vmulq_n_f32(_p1, ps[0]), _scale1);
                    _p1n = vmulq_f32(vmulq_n_f32(_p1n, ps[1]), _scale1);

                    int8x8x2_t _r01;
                    _r01.val[0] = float2int8(_p0, _p1);
                    _r01.val[1] = float2int8(_p0n, _p1n);

                    vst2_s8(pp, _r01);

                    pp += 16;
                    p0 += 8;
                    ps += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vld1_u16(p0));
                    float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 4));

                    _p0 = vmulq_f32(vmulq_n_f32(_p0, ps[0]), _scale0);
                    _p1 = vmulq_f32(vmulq_n_f32(_p1, ps[0]), _scale1);

                    int8x8_t _r01 = float2int8(_p0, _p1);

                    vst1_s8(pp, _r01);

                    pp += 8;
                    p0 += 4;
                    ps++;
                }
                pd += 8;
            }

            if (elempack == 1)
            {
                float32x4_t _absmax0 = _zero;
                float32x4_t _absmax1 = _zero;
                float32x4_t _absmax2 = _zero;
                float32x4_t _absmax3 = _zero;
                float32x4_t _absmax4 = _zero;
                float32x4_t _absmax5 = _zero;
                float32x4_t _absmax6 = _zero;
                float32x4_t _absmax7 = _zero;
                const unsigned short* p0a = p0;
                const float* psa = ps;
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float32x4_t _s = vld1q_f32(psa);
                    float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a));
                    _absmax0 = vmaxq_f32(_absmax0, vmulq_f32(vabsq_f32(_p0), _s));
                    float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a + A_hstep));
                    _absmax1 = vmaxq_f32(_absmax1, vmulq_f32(vabsq_f32(_p1), _s));
                    float32x4_t _p2 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a + A_hstep * 2));
                    _absmax2 = vmaxq_f32(_absmax2, vmulq_f32(vabsq_f32(_p2), _s));
                    float32x4_t _p3 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a + A_hstep * 3));
                    _absmax3 = vmaxq_f32(_absmax3, vmulq_f32(vabsq_f32(_p3), _s));
                    float32x4_t _p4 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a + A_hstep * 4));
                    _absmax4 = vmaxq_f32(_absmax4, vmulq_f32(vabsq_f32(_p4), _s));
                    float32x4_t _p5 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a + A_hstep * 5));
                    _absmax5 = vmaxq_f32(_absmax5, vmulq_f32(vabsq_f32(_p5), _s));
                    float32x4_t _p6 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a + A_hstep * 6));
                    _absmax6 = vmaxq_f32(_absmax6, vmulq_f32(vabsq_f32(_p6), _s));
                    float32x4_t _p7 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a + A_hstep * 7));
                    _absmax7 = vmaxq_f32(_absmax7, vmulq_f32(vabsq_f32(_p7), _s));
                    p0a += 4;
                    psa += 4;
                }
                float32x2_t _max0 = vpmax_f32(vget_low_f32(_absmax0), vget_high_f32(_absmax0));
                float32x2_t _max1 = vpmax_f32(vget_low_f32(_absmax1), vget_high_f32(_absmax1));
                float32x2_t _max2 = vpmax_f32(vget_low_f32(_absmax2), vget_high_f32(_absmax2));
                float32x2_t _max3 = vpmax_f32(vget_low_f32(_absmax3), vget_high_f32(_absmax3));
                float32x2_t _max4 = vpmax_f32(vget_low_f32(_absmax4), vget_high_f32(_absmax4));
                float32x2_t _max5 = vpmax_f32(vget_low_f32(_absmax5), vget_high_f32(_absmax5));
                float32x2_t _max6 = vpmax_f32(vget_low_f32(_absmax6), vget_high_f32(_absmax6));
                float32x2_t _max7 = vpmax_f32(vget_low_f32(_absmax7), vget_high_f32(_absmax7));
                _absmax0 = vcombine_f32(vpmax_f32(_max0, _max1), vpmax_f32(_max2, _max3));
                _absmax1 = vcombine_f32(vpmax_f32(_max4, _max5), vpmax_f32(_max6, _max7));
                for (; kk < max_kk0; kk++)
                {
                    const float s = *psa++;
                    float32x4_t _p0 = vdupq_n_f32(float16_to_float32(p0a[0]));
                    _p0 = vsetq_lane_f32(float16_to_float32(p0a[A_hstep]), _p0, 1);
                    _p0 = vsetq_lane_f32(float16_to_float32(p0a[A_hstep * 2]), _p0, 2);
                    _p0 = vsetq_lane_f32(float16_to_float32(p0a[A_hstep * 3]), _p0, 3);
                    float32x4_t _p1 = vdupq_n_f32(float16_to_float32(p0a[A_hstep * 4]));
                    _p1 = vsetq_lane_f32(float16_to_float32(p0a[A_hstep * 5]), _p1, 1);
                    _p1 = vsetq_lane_f32(float16_to_float32(p0a[A_hstep * 6]), _p1, 2);
                    _p1 = vsetq_lane_f32(float16_to_float32(p0a[A_hstep * 7]), _p1, 3);
                    _absmax0 = vmaxq_f32(_absmax0, vmulq_n_f32(vabsq_f32(_p0), s));
                    _absmax1 = vmaxq_f32(_absmax1, vmulq_n_f32(vabsq_f32(_p1), s));
                    p0a++;
                }

                vst1q_f32(pd, vmulq_n_f32(_absmax0, 1.f / 127.f));
                vst1q_f32(pd + 4, vmulq_n_f32(_absmax1, 1.f / 127.f));

                float32x4_t _scale0 = vbslq_f32(vceqq_f32(_absmax0, _zero), _zero, vdivq_f32(_v127, _absmax0));
                float32x4_t _scale1 = vbslq_f32(vceqq_f32(_absmax1, _zero), _zero, vdivq_f32(_v127, _absmax1));

                kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    float32x4_t _s0 = vld1q_f32(ps);
                    float32x4_t _s1 = vld1q_f32(ps + 4);
                    float32x4_t _p00 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0)), _s0);
                    float32x4_t _p01 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 4)), _s1);
                    float32x4_t _p10 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep)), _s0);
                    float32x4_t _p11 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep + 4)), _s1);
                    float32x4_t _p20 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 2)), _s0);
                    float32x4_t _p21 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 2 + 4)), _s1);
                    float32x4_t _p30 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 3)), _s0);
                    float32x4_t _p31 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 3 + 4)), _s1);
                    float32x4_t _p40 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 4)), _s0);
                    float32x4_t _p41 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 4 + 4)), _s1);
                    float32x4_t _p50 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 5)), _s0);
                    float32x4_t _p51 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 5 + 4)), _s1);
                    float32x4_t _p60 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 6)), _s0);
                    float32x4_t _p61 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 6 + 4)), _s1);
                    float32x4_t _p70 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 7)), _s0);
                    float32x4_t _p71 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 7 + 4)), _s1);
                    int8x8_t _r0 = float2int8(vmulq_laneq_f32(_p00, _scale0, 0), vmulq_laneq_f32(_p01, _scale0, 0));
                    int8x8_t _r1 = float2int8(vmulq_laneq_f32(_p10, _scale0, 1), vmulq_laneq_f32(_p11, _scale0, 1));
                    int8x8_t _r2 = float2int8(vmulq_laneq_f32(_p20, _scale0, 2), vmulq_laneq_f32(_p21, _scale0, 2));
                    int8x8_t _r3 = float2int8(vmulq_laneq_f32(_p30, _scale0, 3), vmulq_laneq_f32(_p31, _scale0, 3));
                    int8x8_t _r4 = float2int8(vmulq_laneq_f32(_p40, _scale1, 0), vmulq_laneq_f32(_p41, _scale1, 0));
                    int8x8_t _r5 = float2int8(vmulq_laneq_f32(_p50, _scale1, 1), vmulq_laneq_f32(_p51, _scale1, 1));
                    int8x8_t _r6 = float2int8(vmulq_laneq_f32(_p60, _scale1, 2), vmulq_laneq_f32(_p61, _scale1, 2));
                    int8x8_t _r7 = float2int8(vmulq_laneq_f32(_p70, _scale1, 3), vmulq_laneq_f32(_p71, _scale1, 3));
                    vst1q_s8(pp, vcombine_s8(_r0, _r1));
                    vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
                    vst1q_s8(pp + 32, vcombine_s8(_r4, _r5));
                    vst1q_s8(pp + 48, vcombine_s8(_r6, _r7));
                    pp += 64;
                    p0 += 8;
                    ps += 8;
                }
#endif // __ARM_FEATURE_MATMUL_INT8
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float32x4_t _s = vld1q_f32(ps);
                    float32x4_t _p0 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0)), _s);
                    float32x4_t _p1 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep)), _s);
                    float32x4_t _p2 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 2)), _s);
                    float32x4_t _p3 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 3)), _s);
                    float32x4_t _p4 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 4)), _s);
                    float32x4_t _p5 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 5)), _s);
                    float32x4_t _p6 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 6)), _s);
                    float32x4_t _p7 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 7)), _s);
                    int8x8_t _r01 = float2int8(vmulq_laneq_f32(_p0, _scale0, 0), vmulq_laneq_f32(_p1, _scale0, 1));
                    int8x8_t _r23 = float2int8(vmulq_laneq_f32(_p2, _scale0, 2), vmulq_laneq_f32(_p3, _scale0, 3));
                    int8x8_t _r45 = float2int8(vmulq_laneq_f32(_p4, _scale1, 0), vmulq_laneq_f32(_p5, _scale1, 1));
                    int8x8_t _r67 = float2int8(vmulq_laneq_f32(_p6, _scale1, 2), vmulq_laneq_f32(_p7, _scale1, 3));
#if __ARM_FEATURE_DOTPROD
                    vst1q_s8(pp, vcombine_s8(_r01, _r23));
                    vst1q_s8(pp + 16, vcombine_s8(_r45, _r67));
#else
                    int16x8x2_t _r04 = vuzpq_s16(vreinterpretq_s16_s8(vcombine_s8(_r01, _r23)), vreinterpretq_s16_s8(vcombine_s8(_r45, _r67)));
                    vst1q_s16((short*)pp, _r04.val[0]);
                    vst1q_s16((short*)pp + 8, _r04.val[1]);
#endif
                    pp += 32;
                    p0 += 4;
                    ps += 4;
                }
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    const float s0 = ps[0];
                    const float s1 = ps[1];
                    float v00 = float16_to_float32(p0[0]);
                    float v01 = float16_to_float32(p0[1]);
                    float v10 = float16_to_float32(p0[A_hstep]);
                    float v11 = float16_to_float32(p0[A_hstep + 1]);
                    float v20 = float16_to_float32(p0[A_hstep * 2]);
                    float v21 = float16_to_float32(p0[A_hstep * 2 + 1]);
                    float v30 = float16_to_float32(p0[A_hstep * 3]);
                    float v31 = float16_to_float32(p0[A_hstep * 3 + 1]);
                    float v40 = float16_to_float32(p0[A_hstep * 4]);
                    float v41 = float16_to_float32(p0[A_hstep * 4 + 1]);
                    float v50 = float16_to_float32(p0[A_hstep * 5]);
                    float v51 = float16_to_float32(p0[A_hstep * 5 + 1]);
                    float v60 = float16_to_float32(p0[A_hstep * 6]);
                    float v61 = float16_to_float32(p0[A_hstep * 6 + 1]);
                    float v70 = float16_to_float32(p0[A_hstep * 7]);
                    float v71 = float16_to_float32(p0[A_hstep * 7 + 1]);
                    v00 *= s0;
                    v01 *= s1;
                    v10 *= s0;
                    v11 *= s1;
                    v20 *= s0;
                    v21 *= s1;
                    v30 *= s0;
                    v31 *= s1;
                    v40 *= s0;
                    v41 *= s1;
                    v50 *= s0;
                    v51 *= s1;
                    v60 *= s0;
                    v61 *= s1;
                    v70 *= s0;
                    v71 *= s1;
                    *pp++ = float2int8(v00 * vgetq_lane_f32(_scale0, 0));
                    *pp++ = float2int8(v01 * vgetq_lane_f32(_scale0, 0));
                    *pp++ = float2int8(v10 * vgetq_lane_f32(_scale0, 1));
                    *pp++ = float2int8(v11 * vgetq_lane_f32(_scale0, 1));
                    *pp++ = float2int8(v20 * vgetq_lane_f32(_scale0, 2));
                    *pp++ = float2int8(v21 * vgetq_lane_f32(_scale0, 2));
                    *pp++ = float2int8(v30 * vgetq_lane_f32(_scale0, 3));
                    *pp++ = float2int8(v31 * vgetq_lane_f32(_scale0, 3));
                    *pp++ = float2int8(v40 * vgetq_lane_f32(_scale1, 0));
                    *pp++ = float2int8(v41 * vgetq_lane_f32(_scale1, 0));
                    *pp++ = float2int8(v50 * vgetq_lane_f32(_scale1, 1));
                    *pp++ = float2int8(v51 * vgetq_lane_f32(_scale1, 1));
                    *pp++ = float2int8(v60 * vgetq_lane_f32(_scale1, 2));
                    *pp++ = float2int8(v61 * vgetq_lane_f32(_scale1, 2));
                    *pp++ = float2int8(v70 * vgetq_lane_f32(_scale1, 3));
                    *pp++ = float2int8(v71 * vgetq_lane_f32(_scale1, 3));
                    p0 += 2;
                    ps += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    const float s = ps[0];
                    float v0 = float16_to_float32(p0[0]);
                    float v1 = float16_to_float32(p0[A_hstep]);
                    float v2 = float16_to_float32(p0[A_hstep * 2]);
                    float v3 = float16_to_float32(p0[A_hstep * 3]);
                    float v4 = float16_to_float32(p0[A_hstep * 4]);
                    float v5 = float16_to_float32(p0[A_hstep * 5]);
                    float v6 = float16_to_float32(p0[A_hstep * 6]);
                    float v7 = float16_to_float32(p0[A_hstep * 7]);
                    v0 *= s;
                    v1 *= s;
                    v2 *= s;
                    v3 *= s;
                    v4 *= s;
                    v5 *= s;
                    v6 *= s;
                    v7 *= s;
                    *pp++ = float2int8(v0 * vgetq_lane_f32(_scale0, 0));
                    *pp++ = float2int8(v1 * vgetq_lane_f32(_scale0, 1));
                    *pp++ = float2int8(v2 * vgetq_lane_f32(_scale0, 2));
                    *pp++ = float2int8(v3 * vgetq_lane_f32(_scale0, 3));
                    *pp++ = float2int8(v4 * vgetq_lane_f32(_scale1, 0));
                    *pp++ = float2int8(v5 * vgetq_lane_f32(_scale1, 1));
                    *pp++ = float2int8(v6 * vgetq_lane_f32(_scale1, 2));
                    *pp++ = float2int8(v7 * vgetq_lane_f32(_scale1, 3));
                    p0++;
                    ps++;
                }
                pd += 8;
            }
        }
    }
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * elempack;
        const float* ps = (const float*)input_scales + k;

        float32x4_t _zero = vdupq_n_f32(0.f);

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
            if (elempack == 4)
            {
                float32x4_t _absmax = _zero;
                const unsigned short* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    _absmax = vmaxq_f32(_absmax, vmulq_n_f32(vabsq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0a))), *psa++));
                    p0a += 4;
                }

                vst1q_f32(pd, vmulq_n_f32(_absmax, 1.f / 127.f));
#if __aarch64__
                float32x4_t _scale = vdivq_f32(vdupq_n_f32(127.f), _absmax);
#else
                float32x4_t _scale = div_ps(vdupq_n_f32(127.f), _absmax);
#endif
                _scale = vbslq_f32(vceqq_f32(_absmax, _zero), _zero, _scale);

                int kk = 0;
                for (; kk + 7 < max_kk0; kk += 8)
                {
#if __ARM_FEATURE_DOTPROD
                    uint16x8x4_t _p = vld4q_u16(p0);
                    float32x4_t _s0 = vld1q_f32(ps);
                    float32x4_t _s1 = vld1q_f32(ps + 4);

                    float32x4_t _p0 = vmulq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_p.val[0])), _s0);
                    float32x4_t _p1 = vmulq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_p.val[1])), _s0);
                    float32x4_t _p2 = vmulq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_p.val[2])), _s0);
                    float32x4_t _p3 = vmulq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_p.val[3])), _s0);
                    float32x4_t _p4 = vmulq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_p.val[0])), _s1);
                    float32x4_t _p5 = vmulq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_p.val[1])), _s1);
                    float32x4_t _p6 = vmulq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_p.val[2])), _s1);
                    float32x4_t _p7 = vmulq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_p.val[3])), _s1);

                    _p0 = vmulq_laneq_f32(_p0, _scale, 0);
                    _p1 = vmulq_laneq_f32(_p1, _scale, 1);
                    _p2 = vmulq_laneq_f32(_p2, _scale, 2);
                    _p3 = vmulq_laneq_f32(_p3, _scale, 3);
                    _p4 = vmulq_laneq_f32(_p4, _scale, 0);
                    _p5 = vmulq_laneq_f32(_p5, _scale, 1);
                    _p6 = vmulq_laneq_f32(_p6, _scale, 2);
                    _p7 = vmulq_laneq_f32(_p7, _scale, 3);

#if __ARM_FEATURE_MATMUL_INT8
                    int8x8_t _r0 = float2int8(_p0, _p4);
                    int8x8_t _r1 = float2int8(_p1, _p5);
                    int8x8_t _r2 = float2int8(_p2, _p6);
                    int8x8_t _r3 = float2int8(_p3, _p7);
#else  // __ARM_FEATURE_MATMUL_INT8
                    int8x8_t _r0 = float2int8(_p0, _p1);
                    int8x8_t _r1 = float2int8(_p2, _p3);
                    int8x8_t _r2 = float2int8(_p4, _p5);
                    int8x8_t _r3 = float2int8(_p6, _p7);
#endif // __ARM_FEATURE_MATMUL_INT8

                    vst1q_s8(pp, vcombine_s8(_r0, _r1));
                    vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
#else  // __ARM_FEATURE_DOTPROD
                    uint16x8_t _p = vld1q_u16(p0);
                    uint16x8_t _q = vld1q_u16(p0 + 8);
                    uint16x8_t _r = vld1q_u16(p0 + 16);
                    uint16x8_t _s = vld1q_u16(p0 + 24);
                    float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vget_low_u16(_p));
                    float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vget_high_u16(_p));
                    float32x4_t _p2 = vcvt_f32_f16((float16x4_t)vget_low_u16(_q));
                    float32x4_t _p3 = vcvt_f32_f16((float16x4_t)vget_high_u16(_q));
                    float32x4_t _p4 = vcvt_f32_f16((float16x4_t)vget_low_u16(_r));
                    float32x4_t _p5 = vcvt_f32_f16((float16x4_t)vget_high_u16(_r));
                    float32x4_t _p6 = vcvt_f32_f16((float16x4_t)vget_low_u16(_s));
                    float32x4_t _p7 = vcvt_f32_f16((float16x4_t)vget_high_u16(_s));

                    _p0 = vmulq_f32(vmulq_n_f32(_p0, ps[0]), _scale);
                    _p1 = vmulq_f32(vmulq_n_f32(_p1, ps[1]), _scale);
                    _p2 = vmulq_f32(vmulq_n_f32(_p2, ps[2]), _scale);
                    _p3 = vmulq_f32(vmulq_n_f32(_p3, ps[3]), _scale);
                    _p4 = vmulq_f32(vmulq_n_f32(_p4, ps[4]), _scale);
                    _p5 = vmulq_f32(vmulq_n_f32(_p5, ps[5]), _scale);
                    _p6 = vmulq_f32(vmulq_n_f32(_p6, ps[6]), _scale);
                    _p7 = vmulq_f32(vmulq_n_f32(_p7, ps[7]), _scale);

                    int8x16x2_t _r01;
                    _r01.val[0] = vcombine_s8(float2int8(_p0, _p2), float2int8(_p4, _p6));
                    _r01.val[1] = vcombine_s8(float2int8(_p1, _p3), float2int8(_p5, _p7));

                    vst2q_s8(pp, _r01);
#endif // __ARM_FEATURE_DOTPROD

                    pp += 32;
                    p0 += 32;
                    ps += 8;
                }
                for (; kk + 3 < max_kk0; kk += 4)
                {
#if __ARM_FEATURE_DOTPROD
                    uint16x4x4_t _p = vld4_u16(p0);
                    float32x4_t _s = vld1q_f32(ps);

                    float32x4_t _p0 = vmulq_f32(vcvt_f32_f16((float16x4_t)_p.val[0]), _s);
                    float32x4_t _p1 = vmulq_f32(vcvt_f32_f16((float16x4_t)_p.val[1]), _s);
                    float32x4_t _p2 = vmulq_f32(vcvt_f32_f16((float16x4_t)_p.val[2]), _s);
                    float32x4_t _p3 = vmulq_f32(vcvt_f32_f16((float16x4_t)_p.val[3]), _s);

                    _p0 = vmulq_laneq_f32(_p0, _scale, 0);
                    _p1 = vmulq_laneq_f32(_p1, _scale, 1);
                    _p2 = vmulq_laneq_f32(_p2, _scale, 2);
                    _p3 = vmulq_laneq_f32(_p3, _scale, 3);

                    int8x8_t _r0 = float2int8(_p0, _p1);
                    int8x8_t _r1 = float2int8(_p2, _p3);

                    vst1q_s8(pp, vcombine_s8(_r0, _r1));
#else  // __ARM_FEATURE_DOTPROD
                    uint16x8_t _p = vld1q_u16(p0);
                    uint16x8_t _q = vld1q_u16(p0 + 8);
                    float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vget_low_u16(_p));
                    float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vget_high_u16(_p));
                    float32x4_t _p2 = vcvt_f32_f16((float16x4_t)vget_low_u16(_q));
                    float32x4_t _p3 = vcvt_f32_f16((float16x4_t)vget_high_u16(_q));

                    _p0 = vmulq_f32(vmulq_n_f32(_p0, ps[0]), _scale);
                    _p1 = vmulq_f32(vmulq_n_f32(_p1, ps[1]), _scale);
                    _p2 = vmulq_f32(vmulq_n_f32(_p2, ps[2]), _scale);
                    _p3 = vmulq_f32(vmulq_n_f32(_p3, ps[3]), _scale);

                    int8x8x2_t _r01;
                    _r01.val[0] = float2int8(_p0, _p2);
                    _r01.val[1] = float2int8(_p1, _p3);

                    vst2_s8(pp, _r01);
#endif // __ARM_FEATURE_DOTPROD

                    pp += 16;
                    p0 += 16;
                    ps += 4;
                }
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    uint16x8_t _p = vld1q_u16(p0);
                    float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vget_low_u16(_p));
                    float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vget_high_u16(_p));

                    _p0 = vmulq_f32(vmulq_n_f32(_p0, ps[0]), _scale);
                    _p1 = vmulq_f32(vmulq_n_f32(_p1, ps[1]), _scale);

                    float32x4x2_t _p01 = vzipq_f32(_p0, _p1);

                    int8x8_t _r01 = float2int8(_p01.val[0], _p01.val[1]);

                    vst1_s8(pp, _r01);

                    pp += 8;
                    p0 += 8;
                    ps += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vld1_u16(p0));
                    _p0 = vmulq_f32(vmulq_n_f32(_p0, ps[0]), _scale);
                    int8x8_t _r0 = float2int8(_p0, _p0);

                    pp[0] = vget_lane_s8(_r0, 0);
                    pp[1] = vget_lane_s8(_r0, 1);
                    pp[2] = vget_lane_s8(_r0, 2);
                    pp[3] = vget_lane_s8(_r0, 3);

                    pp += 4;
                    p0 += 4;
                    ps++;
                }
                pd += 4;
            }

            if (elempack == 1)
            {
                float32x4_t _absmax0 = _zero;
                float32x4_t _absmax1 = _zero;
                float32x4_t _absmax2 = _zero;
                float32x4_t _absmax3 = _zero;
                const unsigned short* p0a = p0;
                const float* psa = ps;
                int kk = 0;
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float32x4_t _s = vld1q_f32(psa);
                    float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a));
                    _absmax0 = vmaxq_f32(_absmax0, vmulq_f32(vabsq_f32(_p0), _s));
                    float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a + A_hstep));
                    _absmax1 = vmaxq_f32(_absmax1, vmulq_f32(vabsq_f32(_p1), _s));
                    float32x4_t _p2 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a + A_hstep * 2));
                    _absmax2 = vmaxq_f32(_absmax2, vmulq_f32(vabsq_f32(_p2), _s));
                    float32x4_t _p3 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a + A_hstep * 3));
                    _absmax3 = vmaxq_f32(_absmax3, vmulq_f32(vabsq_f32(_p3), _s));
                    p0a += 4;
                    psa += 4;
                }
                float32x2_t _max0 = vpmax_f32(vget_low_f32(_absmax0), vget_high_f32(_absmax0));
                float32x2_t _max1 = vpmax_f32(vget_low_f32(_absmax1), vget_high_f32(_absmax1));
                float32x2_t _max2 = vpmax_f32(vget_low_f32(_absmax2), vget_high_f32(_absmax2));
                float32x2_t _max3 = vpmax_f32(vget_low_f32(_absmax3), vget_high_f32(_absmax3));
                _absmax0 = vcombine_f32(vpmax_f32(_max0, _max1), vpmax_f32(_max2, _max3));
                for (; kk < max_kk0; kk++)
                {
                    const float s = *psa++;
                    float32x4_t _p = vdupq_n_f32(float16_to_float32(p0a[0]));
                    _p = vsetq_lane_f32(float16_to_float32(p0a[A_hstep]), _p, 1);
                    _p = vsetq_lane_f32(float16_to_float32(p0a[A_hstep * 2]), _p, 2);
                    _p = vsetq_lane_f32(float16_to_float32(p0a[A_hstep * 3]), _p, 3);
                    _absmax0 = vmaxq_f32(_absmax0, vmulq_n_f32(vabsq_f32(_p), s));
                    p0a++;
                }
                vst1q_f32(pd, vmulq_n_f32(_absmax0, 1.f / 127.f));

#if __aarch64__
                float32x4_t _scale = vdivq_f32(vdupq_n_f32(127.f), _absmax0);
#else
                float32x4_t _scale = div_ps(vdupq_n_f32(127.f), _absmax0);
#endif
                _scale = vbslq_f32(vceqq_f32(_absmax0, _zero), _zero, _scale);

                kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    float32x4_t _s0 = vld1q_f32(ps);
                    float32x4_t _s1 = vld1q_f32(ps + 4);
                    float32x4_t _p00 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0)), _s0);
                    float32x4_t _p01 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 4)), _s1);
                    float32x4_t _p10 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep)), _s0);
                    float32x4_t _p11 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep + 4)), _s1);
                    float32x4_t _p20 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 2)), _s0);
                    float32x4_t _p21 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 2 + 4)), _s1);
                    float32x4_t _p30 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 3)), _s0);
                    float32x4_t _p31 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 3 + 4)), _s1);
                    int8x8_t _r0 = float2int8(vmulq_laneq_f32(_p00, _scale, 0), vmulq_laneq_f32(_p01, _scale, 0));
                    int8x8_t _r1 = float2int8(vmulq_laneq_f32(_p10, _scale, 1), vmulq_laneq_f32(_p11, _scale, 1));
                    int8x8_t _r2 = float2int8(vmulq_laneq_f32(_p20, _scale, 2), vmulq_laneq_f32(_p21, _scale, 2));
                    int8x8_t _r3 = float2int8(vmulq_laneq_f32(_p30, _scale, 3), vmulq_laneq_f32(_p31, _scale, 3));
                    vst1q_s8(pp, vcombine_s8(_r0, _r1));
                    vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
                    pp += 32;
                    p0 += 8;
                    ps += 8;
                }
#endif // __ARM_FEATURE_MATMUL_INT8
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float32x4_t _s = vld1q_f32(ps);
                    float32x4_t _p0 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0)), _s);
                    float32x4_t _p1 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep)), _s);
                    float32x4_t _p2 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 2)), _s);
                    float32x4_t _p3 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 3)), _s);
#if __aarch64__
                    int8x8_t _r01 = float2int8(vmulq_laneq_f32(_p0, _scale, 0), vmulq_laneq_f32(_p1, _scale, 1));
                    int8x8_t _r23 = float2int8(vmulq_laneq_f32(_p2, _scale, 2), vmulq_laneq_f32(_p3, _scale, 3));
#else
                    int8x8_t _r01 = float2int8(vmulq_lane_f32(_p0, vget_low_f32(_scale), 0), vmulq_lane_f32(_p1, vget_low_f32(_scale), 1));
                    int8x8_t _r23 = float2int8(vmulq_lane_f32(_p2, vget_high_f32(_scale), 0), vmulq_lane_f32(_p3, vget_high_f32(_scale), 1));
#endif
#if __ARM_FEATURE_DOTPROD
                    vst1q_s8(pp, vcombine_s8(_r01, _r23));
#else
                    int16x8_t _r0123 = vreinterpretq_s16_s8(vcombine_s8(_r01, _r23));
                    int16x8x2_t _r02 = vuzpq_s16(_r0123, _r0123);
                    vst1q_s8(pp, vreinterpretq_s8_s16(vcombine_s16(vget_low_s16(_r02.val[0]), vget_low_s16(_r02.val[1]))));
#endif
                    pp += 16;
                    p0 += 4;
                    ps += 4;
                }
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    const float s0 = ps[0];
                    const float s1 = ps[1];
                    float v00 = float16_to_float32(p0[0]);
                    float v01 = float16_to_float32(p0[1]);
                    float v10 = float16_to_float32(p0[A_hstep]);
                    float v11 = float16_to_float32(p0[A_hstep + 1]);
                    float v20 = float16_to_float32(p0[A_hstep * 2]);
                    float v21 = float16_to_float32(p0[A_hstep * 2 + 1]);
                    float v30 = float16_to_float32(p0[A_hstep * 3]);
                    float v31 = float16_to_float32(p0[A_hstep * 3 + 1]);
                    v00 *= s0;
                    v01 *= s1;
                    v10 *= s0;
                    v11 *= s1;
                    v20 *= s0;
                    v21 *= s1;
                    v30 *= s0;
                    v31 *= s1;
                    *pp++ = float2int8(v00 * vgetq_lane_f32(_scale, 0));
                    *pp++ = float2int8(v01 * vgetq_lane_f32(_scale, 0));
                    *pp++ = float2int8(v10 * vgetq_lane_f32(_scale, 1));
                    *pp++ = float2int8(v11 * vgetq_lane_f32(_scale, 1));
                    *pp++ = float2int8(v20 * vgetq_lane_f32(_scale, 2));
                    *pp++ = float2int8(v21 * vgetq_lane_f32(_scale, 2));
                    *pp++ = float2int8(v30 * vgetq_lane_f32(_scale, 3));
                    *pp++ = float2int8(v31 * vgetq_lane_f32(_scale, 3));
                    p0 += 2;
                    ps += 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    const float s = ps[0];
                    float v0 = float16_to_float32(p0[0]);
                    float v1 = float16_to_float32(p0[A_hstep]);
                    float v2 = float16_to_float32(p0[A_hstep * 2]);
                    float v3 = float16_to_float32(p0[A_hstep * 3]);
                    v0 *= s;
                    v1 *= s;
                    v2 *= s;
                    v3 *= s;
                    *pp++ = float2int8(v0 * vgetq_lane_f32(_scale, 0));
                    *pp++ = float2int8(v1 * vgetq_lane_f32(_scale, 1));
                    *pp++ = float2int8(v2 * vgetq_lane_f32(_scale, 2));
                    *pp++ = float2int8(v3 * vgetq_lane_f32(_scale, 3));
                    p0++;
                    ps++;
                }
                pd += 4;
            }
        }
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
        const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * elempack;
        const float* ps = (const float*)input_scales + k;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
            float absmax0 = 0.f;
            float absmax1 = 0.f;
            const unsigned short* p0a = p0;
            const float* psa = ps;
            int kk = 0;
#if __ARM_NEON
            float32x4_t _absmax0 = vdupq_n_f32(0.f);
            float32x4_t _absmax1 = vdupq_n_f32(0.f);
            for (; kk + 3 < max_kk0; kk += 4)
            {
                float32x4_t _s = vld1q_f32(psa);
                float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a));
                _absmax0 = vmaxq_f32(_absmax0, vmulq_f32(vabsq_f32(_p0), _s));
                float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a + A_hstep));
                _absmax1 = vmaxq_f32(_absmax1, vmulq_f32(vabsq_f32(_p1), _s));
                p0a += 4;
                psa += 4;
            }
#if __aarch64__
            absmax0 = vmaxvq_f32(_absmax0);
            absmax1 = vmaxvq_f32(_absmax1);
#else
            float32x2_t _max0 = vmax_f32(vget_low_f32(_absmax0), vget_high_f32(_absmax0));
            float32x2_t _max1 = vmax_f32(vget_low_f32(_absmax1), vget_high_f32(_absmax1));
            _max0 = vpmax_f32(_max0, _max0);
            _max1 = vpmax_f32(_max1, _max1);
            absmax0 = vget_lane_f32(_max0, 0);
            absmax1 = vget_lane_f32(_max1, 0);
#endif
#endif // __ARM_NEON

            for (; kk < max_kk0; kk++)
            {
                float v0 = float16_to_float32(p0a[0]);
                float v1 = float16_to_float32(p0a[A_hstep]);
                const float s = *psa++;
                absmax0 = std::max(absmax0, fabsf(v0) * s);
                absmax1 = std::max(absmax1, fabsf(v1) * s);
                p0a++;
            }

            pd[0] = absmax0 / 127.f;
            pd[1] = absmax1 / 127.f;
            const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
            const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;

            kk = 0;
#if __ARM_NEON
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
            for (; kk + 7 < max_kk0; kk += 8)
            {
                float32x4_t _s0 = vld1q_f32(ps);
                float32x4_t _s1 = vld1q_f32(ps + 4);
                float32x4_t _p00 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0)), _s0);
                float32x4_t _p01 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 4)), _s1);
                float32x4_t _p10 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep)), _s0);
                float32x4_t _p11 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep + 4)), _s1);
                int8x8_t _r0 = float2int8(vmulq_n_f32(_p00, scale0), vmulq_n_f32(_p01, scale0));
                int8x8_t _r1 = float2int8(vmulq_n_f32(_p10, scale1), vmulq_n_f32(_p11, scale1));
                vst1q_s8(pp, vcombine_s8(_r0, _r1));
                pp += 16;
                p0 += 8;
                ps += 8;
            }
#endif // __ARM_FEATURE_MATMUL_INT8
#endif // __ARM_FEATURE_DOTPROD
            for (; kk + 3 < max_kk0; kk += 4)
            {
                float32x4_t _s = vld1q_f32(ps);
                float32x4_t _p0 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0)), _s);
                float32x4_t _p1 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep)), _s);
                int8x8_t _r01 = float2int8(vmulq_n_f32(_p0, scale0), vmulq_n_f32(_p1, scale1));
#if __ARM_FEATURE_DOTPROD
                vst1_s8(pp, _r01);
#else
                int16x4_t _r01_s16 = vreinterpret_s16_s8(_r01);
                int16x4_t _r10_s16 = vext_s16(_r01_s16, _r01_s16, 2);
                vst1_s8(pp, vreinterpret_s8_s16(vzip_s16(_r01_s16, _r10_s16).val[0]));
#endif
                pp += 8;
                p0 += 4;
                ps += 4;
            }
            for (; kk + 1 < max_kk0; kk += 2)
            {
                float v00 = float16_to_float32(p0[0]);
                float v01 = float16_to_float32(p0[1]);
                float v10 = float16_to_float32(p0[A_hstep]);
                float v11 = float16_to_float32(p0[A_hstep + 1]);
                v00 *= ps[0];
                v01 *= ps[1];
                v10 *= ps[0];
                v11 *= ps[1];
                *pp++ = float2int8(v00 * scale0);
                *pp++ = float2int8(v01 * scale0);
                *pp++ = float2int8(v10 * scale1);
                *pp++ = float2int8(v11 * scale1);
                p0 += 2;
                ps += 2;
            }
#endif // __ARM_NEON
            for (; kk < max_kk0; kk++)
            {
                float v0 = float16_to_float32(p0[0]);
                float v1 = float16_to_float32(p0[A_hstep]);
                v0 *= ps[0];
                v1 *= ps[0];
                *pp++ = float2int8(v0 * scale0);
                *pp++ = float2int8(v1 * scale1);
                p0++;
                ps++;
            }

            pd += 2;
        }
    }
    for (; ii < max_ii; ii++)
    {
        const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * elempack;
        const float* ps = (const float*)input_scales + k;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
            const unsigned short* p0a = p0;
            const float* psa = ps;

            float absmax = 0.f;
            int kk = 0;
#if __ARM_NEON
            float32x4_t _absmax = vdupq_n_f32(0.f);
            for (; kk + 3 < max_kk0; kk += 4)
            {
                float32x4_t _p = vcvt_f32_f16((float16x4_t)vld1_u16(p0a));
                float32x4_t _s = vld1q_f32(psa);
                _absmax = vmaxq_f32(_absmax, vmulq_f32(vabsq_f32(_p), _s));
                p0a += 4;
                psa += 4;
            }
#if __aarch64__
            absmax = vmaxvq_f32(_absmax);
#else
            float32x2_t _max2 = vmax_f32(vget_low_f32(_absmax), vget_high_f32(_absmax));
            _max2 = vpmax_f32(_max2, _max2);
            absmax = vget_lane_f32(_max2, 0);
#endif
#endif // __ARM_NEON
            for (; kk < max_kk0; kk++)
            {
                float v = float16_to_float32(*p0a++);
                absmax = std::max(absmax, fabsf(v) * *psa++);
            }

            if (absmax == 0.f)
            {
                *pd++ = 0.f;
                for (int kk0 = 0; kk0 < max_kk0; kk0++)
                    *pp++ = 0;
                p0 += max_kk0;
                ps += max_kk0;
                continue;
            }

            const float scale = 127.f / absmax;
            *pd++ = absmax / 127.f;

            kk = 0;
#if __ARM_NEON
            float32x4_t _scale = vdupq_n_f32(scale);
            for (; kk + 7 < max_kk0; kk += 8)
            {
                float32x4_t _p0 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0)), vld1q_f32(ps));
                float32x4_t _p1 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 4)), vld1q_f32(ps + 4));
                vst1_s8(pp, float2int8(vmulq_f32(_p0, _scale), vmulq_f32(_p1, _scale)));
                pp += 8;
                p0 += 8;
                ps += 8;
            }
            for (; kk + 3 < max_kk0; kk += 4)
            {
                float32x4_t _p = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0)), vld1q_f32(ps));
                int8x8_t _r = float2int8(vmulq_f32(_p, _scale), vmulq_f32(_p, _scale));
                vst1_lane_s32((int*)pp, vreinterpret_s32_s8(_r), 0);
                pp += 4;
                p0 += 4;
                ps += 4;
            }
#endif // __ARM_NEON
            for (; kk < max_kk0; kk++)
            {
                float v = float16_to_float32(*p0++);
                v *= *ps++;
                *pp++ = float2int8(v * scale);
            }
        }
    }
}
static void transpose_quantize_A_tile_wq_int8_fp16s(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_i8mm())
    {
        transpose_quantize_A_tile_wq_int8_fp16s_i8mm(A, AT_tile, AT_descales_tile, i, max_ii, k, max_kk, block_size, input_scales);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_asimddp())
    {
        transpose_quantize_A_tile_wq_int8_fp16s_asimddp(A, AT_tile, AT_descales_tile, i, max_ii, k, max_kk, block_size, input_scales);
        return;
    }
#endif

    const int elempack = A.elempack;
    signed char* pp = AT_tile;
    float* pd = AT_descales_tile;
    const int block_count = (max_kk + block_size - 1) / block_size;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    if (input_scales.empty())
    {
        int ii = 0;
#if __ARM_NEON
#if __aarch64__
        for (; ii + 7 < max_ii; ii += 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * elempack;
            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
                if (elempack == 8)
                {
                    float32x4_t _absmax0 = vdupq_n_f32(0.f);
                    float32x4_t _absmax1 = vdupq_n_f32(0.f);
                    float32x4_t _absmax2 = vdupq_n_f32(0.f);
                    float32x4_t _absmax3 = vdupq_n_f32(0.f);
                    float32x4_t _absmax4 = vdupq_n_f32(0.f);
                    float32x4_t _absmax5 = vdupq_n_f32(0.f);
                    float32x4_t _absmax6 = vdupq_n_f32(0.f);
                    float32x4_t _absmax7 = vdupq_n_f32(0.f);
                    const unsigned short* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk += 8)
                    {
                        uint16x8_t _p = vld1q_u16(p0a);
                        uint16x8_t _q = vld1q_u16(p0a + 8);
                        uint16x8_t _r = vld1q_u16(p0a + 16);
                        uint16x8_t _s = vld1q_u16(p0a + 24);
                        uint16x8_t _t = vld1q_u16(p0a + 32);
                        uint16x8_t _u = vld1q_u16(p0a + 40);
                        uint16x8_t _v = vld1q_u16(p0a + 48);
                        uint16x8_t _w = vld1q_u16(p0a + 56);
                        _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_p))));
                        _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_p))));
                        _absmax1 = vmaxq_f32(_absmax1, vabsq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_q))));
                        _absmax1 = vmaxq_f32(_absmax1, vabsq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_q))));
                        _absmax2 = vmaxq_f32(_absmax2, vabsq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_r))));
                        _absmax2 = vmaxq_f32(_absmax2, vabsq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_r))));
                        _absmax3 = vmaxq_f32(_absmax3, vabsq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_s))));
                        _absmax3 = vmaxq_f32(_absmax3, vabsq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_s))));
                        _absmax4 = vmaxq_f32(_absmax4, vabsq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_t))));
                        _absmax4 = vmaxq_f32(_absmax4, vabsq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_t))));
                        _absmax5 = vmaxq_f32(_absmax5, vabsq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_u))));
                        _absmax5 = vmaxq_f32(_absmax5, vabsq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_u))));
                        _absmax6 = vmaxq_f32(_absmax6, vabsq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_v))));
                        _absmax6 = vmaxq_f32(_absmax6, vabsq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_v))));
                        _absmax7 = vmaxq_f32(_absmax7, vabsq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_w))));
                        _absmax7 = vmaxq_f32(_absmax7, vabsq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_w))));
                        p0a += A_hstep * 8;
                    }

                    float32x2_t _max0 = vpmax_f32(vget_low_f32(_absmax0), vget_high_f32(_absmax0));
                    float32x2_t _max1 = vpmax_f32(vget_low_f32(_absmax1), vget_high_f32(_absmax1));
                    float32x2_t _max2 = vpmax_f32(vget_low_f32(_absmax2), vget_high_f32(_absmax2));
                    float32x2_t _max3 = vpmax_f32(vget_low_f32(_absmax3), vget_high_f32(_absmax3));
                    float32x2_t _max4 = vpmax_f32(vget_low_f32(_absmax4), vget_high_f32(_absmax4));
                    float32x2_t _max5 = vpmax_f32(vget_low_f32(_absmax5), vget_high_f32(_absmax5));
                    float32x2_t _max6 = vpmax_f32(vget_low_f32(_absmax6), vget_high_f32(_absmax6));
                    float32x2_t _max7 = vpmax_f32(vget_low_f32(_absmax7), vget_high_f32(_absmax7));
                    _absmax0 = vcombine_f32(vpmax_f32(_max0, _max1), vpmax_f32(_max2, _max3));
                    _absmax1 = vcombine_f32(vpmax_f32(_max4, _max5), vpmax_f32(_max6, _max7));

                    vst1q_f32(pd, vmulq_n_f32(_absmax0, 1.f / 127.f));
                    vst1q_f32(pd + 4, vmulq_n_f32(_absmax1, 1.f / 127.f));

                    float32x4_t _zero = vdupq_n_f32(0.f);
                    float32x4_t _scale0 = vdivq_f32(vdupq_n_f32(127.f), _absmax0);
                    float32x4_t _scale1 = vdivq_f32(vdupq_n_f32(127.f), _absmax1);
                    _scale0 = vbslq_f32(vceqq_f32(_absmax0, _zero), _zero, _scale0);
                    _scale1 = vbslq_f32(vceqq_f32(_absmax1, _zero), _zero, _scale1);

                    int kk = 0;
                    for (; kk + 7 < max_kk0; kk += 8)
                    {
                        uint16x8_t _p = vld1q_u16(p0);
                        uint16x8_t _q = vld1q_u16(p0 + 8);
                        uint16x8_t _r = vld1q_u16(p0 + 16);
                        uint16x8_t _s = vld1q_u16(p0 + 24);
                        uint16x8_t _t = vld1q_u16(p0 + 32);
                        uint16x8_t _u = vld1q_u16(p0 + 40);
                        uint16x8_t _v = vld1q_u16(p0 + 48);
                        uint16x8_t _w = vld1q_u16(p0 + 56);
                        float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vget_low_u16(_p));
                        float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vget_high_u16(_p));
                        float32x4_t _p2 = vcvt_f32_f16((float16x4_t)vget_low_u16(_q));
                        float32x4_t _p3 = vcvt_f32_f16((float16x4_t)vget_high_u16(_q));
                        float32x4_t _p4 = vcvt_f32_f16((float16x4_t)vget_low_u16(_r));
                        float32x4_t _p5 = vcvt_f32_f16((float16x4_t)vget_high_u16(_r));
                        float32x4_t _p6 = vcvt_f32_f16((float16x4_t)vget_low_u16(_s));
                        float32x4_t _p7 = vcvt_f32_f16((float16x4_t)vget_high_u16(_s));
                        float32x4_t _p8 = vcvt_f32_f16((float16x4_t)vget_low_u16(_t));
                        float32x4_t _p9 = vcvt_f32_f16((float16x4_t)vget_high_u16(_t));
                        float32x4_t _pa = vcvt_f32_f16((float16x4_t)vget_low_u16(_u));
                        float32x4_t _pb = vcvt_f32_f16((float16x4_t)vget_high_u16(_u));
                        float32x4_t _pc = vcvt_f32_f16((float16x4_t)vget_low_u16(_v));
                        float32x4_t _pd = vcvt_f32_f16((float16x4_t)vget_high_u16(_v));
                        float32x4_t _pe = vcvt_f32_f16((float16x4_t)vget_low_u16(_w));
                        float32x4_t _pf = vcvt_f32_f16((float16x4_t)vget_high_u16(_w));

                        _p0 = vmulq_laneq_f32(_p0, _scale0, 0);
                        _p1 = vmulq_laneq_f32(_p1, _scale0, 0);
                        _p2 = vmulq_laneq_f32(_p2, _scale0, 1);
                        _p3 = vmulq_laneq_f32(_p3, _scale0, 1);
                        _p4 = vmulq_laneq_f32(_p4, _scale0, 2);
                        _p5 = vmulq_laneq_f32(_p5, _scale0, 2);
                        _p6 = vmulq_laneq_f32(_p6, _scale0, 3);
                        _p7 = vmulq_laneq_f32(_p7, _scale0, 3);
                        _p8 = vmulq_laneq_f32(_p8, _scale1, 0);
                        _p9 = vmulq_laneq_f32(_p9, _scale1, 0);
                        _pa = vmulq_laneq_f32(_pa, _scale1, 1);
                        _pb = vmulq_laneq_f32(_pb, _scale1, 1);
                        _pc = vmulq_laneq_f32(_pc, _scale1, 2);
                        _pd = vmulq_laneq_f32(_pd, _scale1, 2);
                        _pe = vmulq_laneq_f32(_pe, _scale1, 3);
                        _pf = vmulq_laneq_f32(_pf, _scale1, 3);

#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                        int8x8_t _r0 = float2int8(_p0, _p1);
                        int8x8_t _r1 = float2int8(_p2, _p3);
                        int8x8_t _r2 = float2int8(_p4, _p5);
                        int8x8_t _r3 = float2int8(_p6, _p7);
                        int8x8_t _r4 = float2int8(_p8, _p9);
                        int8x8_t _r5 = float2int8(_pa, _pb);
                        int8x8_t _r6 = float2int8(_pc, _pd);
                        int8x8_t _r7 = float2int8(_pe, _pf);

                        vst1q_s8(pp, vcombine_s8(_r0, _r1));
                        vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
                        vst1q_s8(pp + 32, vcombine_s8(_r4, _r5));
                        vst1q_s8(pp + 48, vcombine_s8(_r6, _r7));
#else  // __ARM_FEATURE_MATMUL_INT8
                        int8x8_t _r0 = float2int8(_p0, _p2);
                        int8x8_t _r1 = float2int8(_p4, _p6);
                        int8x8_t _r2 = float2int8(_p8, _pa);
                        int8x8_t _r3 = float2int8(_pc, _pe);
                        int8x8_t _r4 = float2int8(_p1, _p3);
                        int8x8_t _r5 = float2int8(_p5, _p7);
                        int8x8_t _r6 = float2int8(_p9, _pb);
                        int8x8_t _r7 = float2int8(_pd, _pf);

                        vst1q_s8(pp, vcombine_s8(_r0, _r1));
                        vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
                        vst1q_s8(pp + 32, vcombine_s8(_r4, _r5));
                        vst1q_s8(pp + 48, vcombine_s8(_r6, _r7));
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
                        int8x8_t _r0 = float2int8(_p0, _p2);
                        int8x8_t _r1 = float2int8(_p4, _p6);
                        int8x8_t _r2 = float2int8(_p8, _pa);
                        int8x8_t _r3 = float2int8(_pc, _pe);
                        int8x8_t _r4 = float2int8(_p1, _p3);
                        int8x8_t _r5 = float2int8(_p5, _p7);
                        int8x8_t _r6 = float2int8(_p9, _pb);
                        int8x8_t _r7 = float2int8(_pd, _pf);

                        int16x8_t _r01 = vreinterpretq_s16_s8(vcombine_s8(_r0, _r1));
                        int16x8_t _r23 = vreinterpretq_s16_s8(vcombine_s8(_r2, _r3));
                        int16x8_t _r45 = vreinterpretq_s16_s8(vcombine_s8(_r4, _r5));
                        int16x8_t _r67 = vreinterpretq_s16_s8(vcombine_s8(_r6, _r7));
                        int16x8x2_t _rr0 = vuzpq_s16(_r01, _r23);
                        int16x8x2_t _rr1 = vuzpq_s16(_r45, _r67);

                        vst1q_s8(pp, vreinterpretq_s8_s16(_rr0.val[0]));
                        vst1q_s8(pp + 16, vreinterpretq_s8_s16(_rr0.val[1]));
                        vst1q_s8(pp + 32, vreinterpretq_s8_s16(_rr1.val[0]));
                        vst1q_s8(pp + 48, vreinterpretq_s8_s16(_rr1.val[1]));
#endif // __ARM_FEATURE_DOTPROD

                        pp += 64;
                        p0 += A_hstep * 8;
                    }
                    pd += 8;
                }
                if (elempack == 4)
                {
                    float32x4_t _absmax0 = vdupq_n_f32(0.f);
                    float32x4_t _absmax1 = vdupq_n_f32(0.f);
                    float32x4_t _absmax2 = vdupq_n_f32(0.f);
                    float32x4_t _absmax3 = vdupq_n_f32(0.f);
                    float32x4_t _absmax4 = vdupq_n_f32(0.f);
                    float32x4_t _absmax5 = vdupq_n_f32(0.f);
                    float32x4_t _absmax6 = vdupq_n_f32(0.f);
                    float32x4_t _absmax7 = vdupq_n_f32(0.f);
                    const unsigned short* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk += 4)
                    {
                        _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0a))));
                        _absmax1 = vmaxq_f32(_absmax1, vabsq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0a + 4))));
                        _absmax2 = vmaxq_f32(_absmax2, vabsq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0a + 8))));
                        _absmax3 = vmaxq_f32(_absmax3, vabsq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0a + 12))));
                        _absmax4 = vmaxq_f32(_absmax4, vabsq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0a + 16))));
                        _absmax5 = vmaxq_f32(_absmax5, vabsq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0a + 20))));
                        _absmax6 = vmaxq_f32(_absmax6, vabsq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0a + 24))));
                        _absmax7 = vmaxq_f32(_absmax7, vabsq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0a + 28))));
                        p0a += A_hstep * 4;
                    }

                    float32x2_t _max0 = vpmax_f32(vget_low_f32(_absmax0), vget_high_f32(_absmax0));
                    float32x2_t _max1 = vpmax_f32(vget_low_f32(_absmax1), vget_high_f32(_absmax1));
                    float32x2_t _max2 = vpmax_f32(vget_low_f32(_absmax2), vget_high_f32(_absmax2));
                    float32x2_t _max3 = vpmax_f32(vget_low_f32(_absmax3), vget_high_f32(_absmax3));
                    float32x2_t _max4 = vpmax_f32(vget_low_f32(_absmax4), vget_high_f32(_absmax4));
                    float32x2_t _max5 = vpmax_f32(vget_low_f32(_absmax5), vget_high_f32(_absmax5));
                    float32x2_t _max6 = vpmax_f32(vget_low_f32(_absmax6), vget_high_f32(_absmax6));
                    float32x2_t _max7 = vpmax_f32(vget_low_f32(_absmax7), vget_high_f32(_absmax7));
                    _absmax0 = vcombine_f32(vpmax_f32(_max0, _max1), vpmax_f32(_max2, _max3));
                    _absmax1 = vcombine_f32(vpmax_f32(_max4, _max5), vpmax_f32(_max6, _max7));

                    vst1q_f32(pd, vmulq_n_f32(_absmax0, 1.f / 127.f));
                    vst1q_f32(pd + 4, vmulq_n_f32(_absmax1, 1.f / 127.f));

                    float32x4_t _zero = vdupq_n_f32(0.f);
                    float32x4_t _scale0 = vdivq_f32(vdupq_n_f32(127.f), _absmax0);
                    float32x4_t _scale1 = vdivq_f32(vdupq_n_f32(127.f), _absmax1);
                    _scale0 = vbslq_f32(vceqq_f32(_absmax0, _zero), _zero, _scale0);
                    _scale1 = vbslq_f32(vceqq_f32(_absmax1, _zero), _zero, _scale1);

                    int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                    for (; kk + 7 < max_kk0; kk += 8)
                    {
                        float32x4_t _p00 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0)), _scale0, 0);
                        float32x4_t _p01 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 4)), _scale0, 0);
                        float32x4_t _p10 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 4)), _scale0, 1);
                        float32x4_t _p11 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 4 + 4)), _scale0, 1);
                        int8x8_t _r0 = float2int8(_p00, _p01);
                        int8x8_t _r1 = float2int8(_p10, _p11);
                        vst1q_s8(pp, vcombine_s8(_r0, _r1));
                        float32x4_t _p20 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 8)), _scale0, 2);
                        float32x4_t _p21 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 4 + 8)), _scale0, 2);
                        float32x4_t _p30 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 12)), _scale0, 3);
                        float32x4_t _p31 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 4 + 12)), _scale0, 3);
                        int8x8_t _r2 = float2int8(_p20, _p21);
                        int8x8_t _r3 = float2int8(_p30, _p31);
                        vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
                        float32x4_t _p40 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 16)), _scale1, 0);
                        float32x4_t _p41 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 4 + 16)), _scale1, 0);
                        float32x4_t _p50 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 20)), _scale1, 1);
                        float32x4_t _p51 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 4 + 20)), _scale1, 1);
                        int8x8_t _r4 = float2int8(_p40, _p41);
                        int8x8_t _r5 = float2int8(_p50, _p51);
                        vst1q_s8(pp + 32, vcombine_s8(_r4, _r5));
                        float32x4_t _p60 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 24)), _scale1, 2);
                        float32x4_t _p61 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 4 + 24)), _scale1, 2);
                        float32x4_t _p70 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 28)), _scale1, 3);
                        float32x4_t _p71 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 4 + 28)), _scale1, 3);
                        int8x8_t _r6 = float2int8(_p60, _p61);
                        int8x8_t _r7 = float2int8(_p70, _p71);
                        vst1q_s8(pp + 48, vcombine_s8(_r6, _r7));
                        pp += 64;
                        p0 += A_hstep * 8;
                    }
#endif // __ARM_FEATURE_MATMUL_INT8
#endif // __ARM_FEATURE_DOTPROD
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        float32x4_t _p0 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0)), _scale0, 0);
                        float32x4_t _p1 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 4)), _scale0, 1);
                        float32x4_t _p2 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 8)), _scale0, 2);
                        float32x4_t _p3 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 12)), _scale0, 3);
                        float32x4_t _p4 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 16)), _scale1, 0);
                        float32x4_t _p5 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 20)), _scale1, 1);
                        float32x4_t _p6 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 24)), _scale1, 2);
                        float32x4_t _p7 = vmulq_laneq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 28)), _scale1, 3);
                        int8x8_t _r0 = float2int8(_p0, _p1);
                        int8x8_t _r1 = float2int8(_p2, _p3);
                        int8x8_t _r2 = float2int8(_p4, _p5);
                        int8x8_t _r3 = float2int8(_p6, _p7);
#if __ARM_FEATURE_DOTPROD
                        vst1q_s8(pp, vcombine_s8(_r0, _r1));
                        vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
#else
                        int16x8_t _r01 = vreinterpretq_s16_s8(vcombine_s8(_r0, _r1));
                        int16x8_t _r23 = vreinterpretq_s16_s8(vcombine_s8(_r2, _r3));
                        int16x8x2_t _rr = vuzpq_s16(_r01, _r23);
                        vst1q_s8(pp, vreinterpretq_s8_s16(_rr.val[0]));
                        vst1q_s8(pp + 16, vreinterpretq_s8_s16(_rr.val[1]));
#endif
                        pp += 32;
                        p0 += A_hstep * 4;
                    }
                    pd += 8;
                }
                if (elempack == 1)
                {
                    float32x4_t _absmax0 = vdupq_n_f32(0.f);
                    float32x4_t _absmax1 = vdupq_n_f32(0.f);
                    const unsigned short* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a));
                        _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(_p0));
                        float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a + 4));
                        _absmax1 = vmaxq_f32(_absmax1, vabsq_f32(_p1));
                        p0a += A_hstep;
                    }

                    vst1q_f32(pd, vmulq_n_f32(_absmax0, 1.f / 127.f));
                    vst1q_f32(pd + 4, vmulq_n_f32(_absmax1, 1.f / 127.f));

                    float32x4_t _v127 = vdupq_n_f32(127.f);
                    float32x4_t _zero = vdupq_n_f32(0.f);
                    float32x4_t _scale0 = vbslq_f32(vceqq_f32(_absmax0, _zero), _zero, vdivq_f32(_v127, _absmax0));
                    float32x4_t _scale1 = vbslq_f32(vceqq_f32(_absmax1, _zero), _zero, vdivq_f32(_v127, _absmax1));

                    int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                    for (; kk + 7 < max_kk0; kk += 8)
                    {
                        float32x4_t _p00 = vcvt_f32_f16((float16x4_t)vld1_u16(p0));
                        float32x4_t _p01 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 4));
                        float32x4_t _p10 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep));
                        float32x4_t _p11 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep + 4));
                        float32x4_t _p20 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 2));
                        float32x4_t _p21 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 2 + 4));
                        float32x4_t _p30 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 3));
                        float32x4_t _p31 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 3 + 4));
                        float32x4_t _p40 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 4));
                        float32x4_t _p41 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 4 + 4));
                        float32x4_t _p50 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 5));
                        float32x4_t _p51 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 5 + 4));
                        float32x4_t _p60 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 6));
                        float32x4_t _p61 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 6 + 4));
                        float32x4_t _p70 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 7));
                        float32x4_t _p71 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 7 + 4));
                        int8x8_t _r0 = float2int8(vmulq_f32(_p00, _scale0), vmulq_f32(_p01, _scale1));
                        int8x8_t _r1 = float2int8(vmulq_f32(_p10, _scale0), vmulq_f32(_p11, _scale1));
                        int8x8_t _r2 = float2int8(vmulq_f32(_p20, _scale0), vmulq_f32(_p21, _scale1));
                        int8x8_t _r3 = float2int8(vmulq_f32(_p30, _scale0), vmulq_f32(_p31, _scale1));
                        int8x8_t _r4 = float2int8(vmulq_f32(_p40, _scale0), vmulq_f32(_p41, _scale1));
                        int8x8_t _r5 = float2int8(vmulq_f32(_p50, _scale0), vmulq_f32(_p51, _scale1));
                        int8x8_t _r6 = float2int8(vmulq_f32(_p60, _scale0), vmulq_f32(_p61, _scale1));
                        int8x8_t _r7 = float2int8(vmulq_f32(_p70, _scale0), vmulq_f32(_p71, _scale1));
                        int8x8x2_t _r04 = vzip_s8(_r0, _r4);
                        int8x8x2_t _r15 = vzip_s8(_r1, _r5);
                        int8x8x2_t _r26 = vzip_s8(_r2, _r6);
                        int8x8x2_t _r37 = vzip_s8(_r3, _r7);
                        int8x8x4_t _r0123;
                        _r0123.val[0] = _r04.val[0];
                        _r0123.val[1] = _r15.val[0];
                        _r0123.val[2] = _r26.val[0];
                        _r0123.val[3] = _r37.val[0];
                        int8x8x4_t _r4567;
                        _r4567.val[0] = _r04.val[1];
                        _r4567.val[1] = _r15.val[1];
                        _r4567.val[2] = _r26.val[1];
                        _r4567.val[3] = _r37.val[1];
                        vst4_s8(pp, _r0123);
                        vst4_s8(pp + 32, _r4567);
                        pp += 64;
                        p0 += A_hstep * 8;
                    }
#endif // __ARM_FEATURE_MATMUL_INT8
#endif // __ARM_FEATURE_DOTPROD
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        float32x4_t _p00 = vcvt_f32_f16((float16x4_t)vld1_u16(p0));
                        float32x4_t _p01 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 4));
                        float32x4_t _p10 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep));
                        float32x4_t _p11 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep + 4));
                        float32x4_t _p20 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 2));
                        float32x4_t _p21 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 2 + 4));
                        float32x4_t _p30 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 3));
                        float32x4_t _p31 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 3 + 4));
                        int8x8_t _r0 = float2int8(vmulq_f32(_p00, _scale0), vmulq_f32(_p01, _scale1));
                        int8x8_t _r1 = float2int8(vmulq_f32(_p10, _scale0), vmulq_f32(_p11, _scale1));
                        int8x8_t _r2 = float2int8(vmulq_f32(_p20, _scale0), vmulq_f32(_p21, _scale1));
                        int8x8_t _r3 = float2int8(vmulq_f32(_p30, _scale0), vmulq_f32(_p31, _scale1));
#if __ARM_FEATURE_DOTPROD
                        int8x8x4_t _r0123;
                        _r0123.val[0] = _r0;
                        _r0123.val[1] = _r1;
                        _r0123.val[2] = _r2;
                        _r0123.val[3] = _r3;
                        vst4_s8(pp, _r0123);
#else
                        int8x8x2_t _r01;
                        _r01.val[0] = _r0;
                        _r01.val[1] = _r1;
                        int8x8x2_t _r23;
                        _r23.val[0] = _r2;
                        _r23.val[1] = _r3;
                        vst2_s8(pp, _r01);
                        vst2_s8(pp + 16, _r23);
#endif
                        pp += 32;
                        p0 += A_hstep * 4;
                    }
                    for (; kk + 1 < max_kk0; kk += 2)
                    {
                        float32x4_t _p00 = vcvt_f32_f16((float16x4_t)vld1_u16(p0));
                        float32x4_t _p01 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 4));
                        float32x4_t _p10 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep));
                        float32x4_t _p11 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep + 4));
                        int8x8_t _r0 = float2int8(vmulq_f32(_p00, _scale0), vmulq_f32(_p01, _scale1));
                        int8x8_t _r1 = float2int8(vmulq_f32(_p10, _scale0), vmulq_f32(_p11, _scale1));
                        int8x8x2_t _r01;
                        _r01.val[0] = _r0;
                        _r01.val[1] = _r1;
                        vst2_s8(pp, _r01);
                        pp += 16;
                        p0 += A_hstep * 2;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vld1_u16(p0));
                        float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 4));
                        vst1_s8(pp, float2int8(vmulq_f32(_p0, _scale0), vmulq_f32(_p1, _scale1)));
                        pp += 8;
                        p0 += A_hstep;
                    }
                    pd += 8;
                }
            }
        }
#endif // __aarch64__
        for (; ii + 3 < max_ii; ii += 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * elempack;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);

#if __aarch64__
                if (elempack == 8)
                {
                    float32x4_t _absmax0 = vdupq_n_f32(0.f);
                    float32x4_t _absmax1 = vdupq_n_f32(0.f);
                    float32x4_t _absmax2 = vdupq_n_f32(0.f);
                    float32x4_t _absmax3 = vdupq_n_f32(0.f);
                    const unsigned short* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk += 8)
                    {
                        uint16x8_t _p = vld1q_u16(p0a);
                        uint16x8_t _q = vld1q_u16(p0a + 8);
                        uint16x8_t _r = vld1q_u16(p0a + 16);
                        uint16x8_t _s = vld1q_u16(p0a + 24);
                        _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_p))));
                        _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_p))));
                        _absmax1 = vmaxq_f32(_absmax1, vabsq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_q))));
                        _absmax1 = vmaxq_f32(_absmax1, vabsq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_q))));
                        _absmax2 = vmaxq_f32(_absmax2, vabsq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_r))));
                        _absmax2 = vmaxq_f32(_absmax2, vabsq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_r))));
                        _absmax3 = vmaxq_f32(_absmax3, vabsq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_s))));
                        _absmax3 = vmaxq_f32(_absmax3, vabsq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_s))));
                        p0a += A_hstep * 8;
                    }

                    float32x2_t _max0 = vpmax_f32(vget_low_f32(_absmax0), vget_high_f32(_absmax0));
                    float32x2_t _max1 = vpmax_f32(vget_low_f32(_absmax1), vget_high_f32(_absmax1));
                    float32x2_t _max2 = vpmax_f32(vget_low_f32(_absmax2), vget_high_f32(_absmax2));
                    float32x2_t _max3 = vpmax_f32(vget_low_f32(_absmax3), vget_high_f32(_absmax3));
                    _absmax0 = vcombine_f32(vpmax_f32(_max0, _max1), vpmax_f32(_max2, _max3));

                    vst1q_f32(pd, vmulq_n_f32(_absmax0, 1.f / 127.f));

                    float32x4_t _zero = vdupq_n_f32(0.f);
                    float32x4_t _scale = vdivq_f32(vdupq_n_f32(127.f), _absmax0);
                    _scale = vbslq_f32(vceqq_f32(_absmax0, _zero), _zero, _scale);

                    int kk = 0;
                    for (; kk + 7 < max_kk0; kk += 8)
                    {
                        uint16x8_t _p = vld1q_u16(p0);
                        uint16x8_t _q = vld1q_u16(p0 + 8);
                        uint16x8_t _r = vld1q_u16(p0 + 16);
                        uint16x8_t _s = vld1q_u16(p0 + 24);
                        float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vget_low_u16(_p));
                        float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vget_high_u16(_p));
                        float32x4_t _p2 = vcvt_f32_f16((float16x4_t)vget_low_u16(_q));
                        float32x4_t _p3 = vcvt_f32_f16((float16x4_t)vget_high_u16(_q));
                        float32x4_t _p4 = vcvt_f32_f16((float16x4_t)vget_low_u16(_r));
                        float32x4_t _p5 = vcvt_f32_f16((float16x4_t)vget_high_u16(_r));
                        float32x4_t _p6 = vcvt_f32_f16((float16x4_t)vget_low_u16(_s));
                        float32x4_t _p7 = vcvt_f32_f16((float16x4_t)vget_high_u16(_s));

                        _p0 = vmulq_laneq_f32(_p0, _scale, 0);
                        _p1 = vmulq_laneq_f32(_p1, _scale, 0);
                        _p2 = vmulq_laneq_f32(_p2, _scale, 1);
                        _p3 = vmulq_laneq_f32(_p3, _scale, 1);
                        _p4 = vmulq_laneq_f32(_p4, _scale, 2);
                        _p5 = vmulq_laneq_f32(_p5, _scale, 2);
                        _p6 = vmulq_laneq_f32(_p6, _scale, 3);
                        _p7 = vmulq_laneq_f32(_p7, _scale, 3);

#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                        int8x8_t _r0 = float2int8(_p0, _p1);
                        int8x8_t _r1 = float2int8(_p2, _p3);
                        int8x8_t _r2 = float2int8(_p4, _p5);
                        int8x8_t _r3 = float2int8(_p6, _p7);
#else  // __ARM_FEATURE_MATMUL_INT8
                        int8x8_t _r0 = float2int8(_p0, _p2);
                        int8x8_t _r1 = float2int8(_p4, _p6);
                        int8x8_t _r2 = float2int8(_p1, _p3);
                        int8x8_t _r3 = float2int8(_p5, _p7);
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
                        int16x4_t _t0 = vreinterpret_s16_s8(float2int8(_p0, _p2));
                        int16x4_t _t1 = vreinterpret_s16_s8(float2int8(_p4, _p6));
                        int16x4_t _t2 = vreinterpret_s16_s8(float2int8(_p1, _p3));
                        int16x4_t _t3 = vreinterpret_s16_s8(float2int8(_p5, _p7));
                        int16x4x2_t _t01 = vuzp_s16(_t0, _t1);
                        int16x4x2_t _t23 = vuzp_s16(_t2, _t3);
                        int8x8_t _r0 = vreinterpret_s8_s16(_t01.val[0]);
                        int8x8_t _r1 = vreinterpret_s8_s16(_t01.val[1]);
                        int8x8_t _r2 = vreinterpret_s8_s16(_t23.val[0]);
                        int8x8_t _r3 = vreinterpret_s8_s16(_t23.val[1]);
#endif // __ARM_FEATURE_DOTPROD

                        vst1q_s8(pp, vcombine_s8(_r0, _r1));
                        vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));

                        pp += 32;
                        p0 += A_hstep * 8;
                    }
                    pd += 4;
                }
#endif // __aarch64__
                if (elempack == 4)
                {
                    float32x4_t _absmax0 = vdupq_n_f32(0.f);
                    float32x4_t _absmax1 = vdupq_n_f32(0.f);
                    float32x4_t _absmax2 = vdupq_n_f32(0.f);
                    float32x4_t _absmax3 = vdupq_n_f32(0.f);
                    const unsigned short* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk += 4)
                    {
                        float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a));
                        float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a + 4));
                        float32x4_t _p2 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a + 8));
                        float32x4_t _p3 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a + 12));
                        _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(_p0));
                        _absmax1 = vmaxq_f32(_absmax1, vabsq_f32(_p1));
                        _absmax2 = vmaxq_f32(_absmax2, vabsq_f32(_p2));
                        _absmax3 = vmaxq_f32(_absmax3, vabsq_f32(_p3));
                        p0a += A_hstep * 4;
                    }

                    float32x2_t _max0 = vpmax_f32(vget_low_f32(_absmax0), vget_high_f32(_absmax0));
                    float32x2_t _max1 = vpmax_f32(vget_low_f32(_absmax1), vget_high_f32(_absmax1));
                    float32x2_t _max2 = vpmax_f32(vget_low_f32(_absmax2), vget_high_f32(_absmax2));
                    float32x2_t _max3 = vpmax_f32(vget_low_f32(_absmax3), vget_high_f32(_absmax3));
                    _absmax0 = vcombine_f32(vpmax_f32(_max0, _max1), vpmax_f32(_max2, _max3));

                    vst1q_f32(pd, vmulq_n_f32(_absmax0, 1.f / 127.f));

                    float32x4_t _zero = vdupq_n_f32(0.f);
#if __aarch64__
                    float32x4_t _scale = vdivq_f32(vdupq_n_f32(127.f), _absmax0);
#else
                    float32x4_t _scale = div_ps(vdupq_n_f32(127.f), _absmax0);
#endif
                    _scale = vbslq_f32(vceqq_f32(_absmax0, _zero), _zero, _scale);

                    int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                    for (; kk + 7 < max_kk0; kk += 8)
                    {
                        float32x4_t _p00 = vmulq_n_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0)), vgetq_lane_f32(_scale, 0));
                        float32x4_t _p01 = vmulq_n_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 4)), vgetq_lane_f32(_scale, 0));
                        float32x4_t _p10 = vmulq_n_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 4)), vgetq_lane_f32(_scale, 1));
                        float32x4_t _p11 = vmulq_n_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 4 + 4)), vgetq_lane_f32(_scale, 1));
                        int8x8_t _r0 = float2int8(_p00, _p01);
                        int8x8_t _r1 = float2int8(_p10, _p11);
                        vst1q_s8(pp, vcombine_s8(_r0, _r1));
                        float32x4_t _p20 = vmulq_n_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 8)), vgetq_lane_f32(_scale, 2));
                        float32x4_t _p21 = vmulq_n_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 4 + 8)), vgetq_lane_f32(_scale, 2));
                        float32x4_t _p30 = vmulq_n_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 12)), vgetq_lane_f32(_scale, 3));
                        float32x4_t _p31 = vmulq_n_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 4 + 12)), vgetq_lane_f32(_scale, 3));
                        int8x8_t _r2 = float2int8(_p20, _p21);
                        int8x8_t _r3 = float2int8(_p30, _p31);
                        vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
                        pp += 32;
                        p0 += A_hstep * 8;
                    }
#endif // __ARM_FEATURE_MATMUL_INT8
#endif // __ARM_FEATURE_DOTPROD
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        float32x4_t _p0 = vmulq_n_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0)), vgetq_lane_f32(_scale, 0));
                        float32x4_t _p1 = vmulq_n_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 4)), vgetq_lane_f32(_scale, 1));
                        float32x4_t _p2 = vmulq_n_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 8)), vgetq_lane_f32(_scale, 2));
                        float32x4_t _p3 = vmulq_n_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 12)), vgetq_lane_f32(_scale, 3));
                        int8x8_t _r0 = float2int8(_p0, _p1);
                        int8x8_t _r1 = float2int8(_p2, _p3);
#if __ARM_FEATURE_DOTPROD
                        vst1q_s8(pp, vcombine_s8(_r0, _r1));
#else
                        int16x8_t _r01 = vreinterpretq_s16_s8(vcombine_s8(_r0, _r1));
                        int16x8x2_t _rr = vuzpq_s16(_r01, _r01);
                        vst1q_s8(pp, vreinterpretq_s8_s16(vcombine_s16(vget_low_s16(_rr.val[0]), vget_low_s16(_rr.val[1]))));
#endif
                        pp += 16;
                        p0 += A_hstep * 4;
                    }
                    pd += 4;
                }
                if (elempack == 1)
                {
                    float32x4_t _absmax = vdupq_n_f32(0.f);
                    const unsigned short* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        float32x4_t _p = vcvt_f32_f16((float16x4_t)vld1_u16(p0a));
                        _absmax = vmaxq_f32(_absmax, vabsq_f32(_p));
                        p0a += A_hstep;
                    }

                    vst1q_f32(pd, vmulq_n_f32(_absmax, 1.f / 127.f));
                    float32x4_t _zero = vdupq_n_f32(0.f);
#if __aarch64__
                    float32x4_t _scale = vdivq_f32(vdupq_n_f32(127.f), _absmax);
#else
                    float32x4_t _scale = div_ps(vdupq_n_f32(127.f), _absmax);
#endif
                    _scale = vbslq_f32(vceqq_f32(_absmax, _zero), _zero, _scale);

                    int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                    for (; kk + 7 < max_kk0; kk += 8)
                    {
                        float32x4_t _p0 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0)), _scale);
                        float32x4_t _p1 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep)), _scale);
                        float32x4_t _p2 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 2)), _scale);
                        float32x4_t _p3 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 3)), _scale);
                        float32x4_t _p4 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 4)), _scale);
                        float32x4_t _p5 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 5)), _scale);
                        float32x4_t _p6 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 6)), _scale);
                        float32x4_t _p7 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 7)), _scale);
                        float32x4x2_t _p04 = vzipq_f32(_p0, _p4);
                        float32x4x2_t _p15 = vzipq_f32(_p1, _p5);
                        float32x4x2_t _p26 = vzipq_f32(_p2, _p6);
                        float32x4x2_t _p37 = vzipq_f32(_p3, _p7);
                        int8x8x4_t _r0123;
                        _r0123.val[0] = float2int8(_p04.val[0], _p04.val[1]);
                        _r0123.val[1] = float2int8(_p15.val[0], _p15.val[1]);
                        _r0123.val[2] = float2int8(_p26.val[0], _p26.val[1]);
                        _r0123.val[3] = float2int8(_p37.val[0], _p37.val[1]);
                        vst4_s8(pp, _r0123);
                        pp += 32;
                        p0 += A_hstep * 8;
                    }
#endif // __ARM_FEATURE_MATMUL_INT8
#endif // __ARM_FEATURE_DOTPROD
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        float32x4_t _p0 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0)), _scale);
                        float32x4_t _p1 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep)), _scale);
                        float32x4_t _p2 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 2)), _scale);
                        float32x4_t _p3 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 3)), _scale);
#if __ARM_FEATURE_DOTPROD
                        transpose4x4_ps(_p0, _p1, _p2, _p3);
                        int8x8_t _r01 = float2int8(_p0, _p1);
                        int8x8_t _r23 = float2int8(_p2, _p3);
                        vst1q_s8(pp, vcombine_s8(_r01, _r23));
#else
                        int8x8_t _r01 = float2int8(_p0, _p1);
                        int8x8_t _r23 = float2int8(_p2, _p3);
                        int8x8_t _r10 = vext_s8(_r01, _r01, 4);
                        int8x8_t _r32 = vext_s8(_r23, _r23, 4);
                        vst1_s8(pp, vzip_s8(_r01, _r10).val[0]);
                        vst1_s8(pp + 8, vzip_s8(_r23, _r32).val[0]);
#endif
                        pp += 16;
                        p0 += A_hstep * 4;
                    }
                    for (; kk + 1 < max_kk0; kk += 2)
                    {
                        float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vld1_u16(p0));
                        float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep));
                        int8x8_t _r01 = float2int8(vmulq_f32(_p0, _scale), vmulq_f32(_p1, _scale));
                        int8x8_t _r10 = vext_s8(_r01, _r01, 4);
                        vst1_s8(pp, vzip_s8(_r01, _r10).val[0]);
                        pp += 8;
                        p0 += A_hstep * 2;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        float32x4_t _p = vcvt_f32_f16((float16x4_t)vld1_u16(p0));
                        vst1_lane_s32((int*)pp, vreinterpret_s32_s8(float2int8(vmulq_f32(_p, _scale), vmulq_f32(_p, _scale))), 0);
                        pp += 4;
                        p0 += A_hstep;
                    }
                    pd += 4;
                }
            }
        }
#endif // __ARM_NEON
        for (; ii + 1 < max_ii; ii += 2)
        {
            const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * elempack;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);
#if __ARM_NEON
#if __aarch64__
                if (elempack == 8)
                {
                    float absmax0 = 0.f;
                    float absmax1 = 0.f;
                    const unsigned short* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk += 8)
                    {
                        uint16x8_t _p = vld1q_u16(p0a);
                        uint16x8_t _q = vld1q_u16(p0a + 8);
                        absmax0 = std::max(absmax0, vmaxvq_f32(vabsq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_p)))));
                        absmax0 = std::max(absmax0, vmaxvq_f32(vabsq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_p)))));
                        absmax1 = std::max(absmax1, vmaxvq_f32(vabsq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_q)))));
                        absmax1 = std::max(absmax1, vmaxvq_f32(vabsq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_q)))));
                        p0a += A_hstep * 8;
                    }

                    pd[0] = absmax0 / 127.f;
                    pd[1] = absmax1 / 127.f;
                    float32x4_t _scale0 = vdupq_n_f32(absmax0 == 0.f ? 0.f : 127.f / absmax0);
                    float32x4_t _scale1 = vdupq_n_f32(absmax1 == 0.f ? 0.f : 127.f / absmax1);

                    for (int kk = 0; kk < max_kk0; kk += 8)
                    {
                        uint16x8_t _p = vld1q_u16(p0);
                        uint16x8_t _q = vld1q_u16(p0 + 8);
                        float32x4_t _p0 = vmulq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_p)), _scale0);
                        float32x4_t _p1 = vmulq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_p)), _scale0);
                        float32x4_t _p2 = vmulq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_q)), _scale1);
                        float32x4_t _p3 = vmulq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_q)), _scale1);

#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                        int8x8_t _r0 = float2int8(_p0, _p1);
                        int8x8_t _r1 = float2int8(_p2, _p3);
#else
                        int8x8_t _r0 = float2int8(_p0, _p2);
                        int8x8_t _r1 = float2int8(_p1, _p3);
#endif
#else
                        int16x4_t _t0 = vreinterpret_s16_s8(float2int8(_p0, _p1));
                        int16x4_t _t1 = vreinterpret_s16_s8(float2int8(_p2, _p3));
                        int16x4x2_t _t01 = vzip_s16(_t0, _t1);
                        int8x8_t _r0 = vreinterpret_s8_s16(_t01.val[0]);
                        int8x8_t _r1 = vreinterpret_s8_s16(_t01.val[1]);
#endif

                        vst1q_s8(pp, vcombine_s8(_r0, _r1));

                        pp += 16;
                        p0 += A_hstep * 8;
                    }
                    pd += 2;
                }
#endif // __aarch64__
                if (elempack == 4)
                {
                    float32x4_t _absmax0 = vdupq_n_f32(0.f);
                    float32x4_t _absmax1 = vdupq_n_f32(0.f);
                    const unsigned short* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk += 4)
                    {
                        float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a));
                        float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a + 4));
                        _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(_p0));
                        _absmax1 = vmaxq_f32(_absmax1, vabsq_f32(_p1));
                        p0a += A_hstep * 4;
                    }

#if __aarch64__
                    float absmax0 = vmaxvq_f32(_absmax0);
                    float absmax1 = vmaxvq_f32(_absmax1);
#else
                    float32x2_t _max0 = vmax_f32(vget_low_f32(_absmax0), vget_high_f32(_absmax0));
                    float32x2_t _max1 = vmax_f32(vget_low_f32(_absmax1), vget_high_f32(_absmax1));
                    _max0 = vpmax_f32(_max0, _max0);
                    _max1 = vpmax_f32(_max1, _max1);
                    float absmax0 = vget_lane_f32(_max0, 0);
                    float absmax1 = vget_lane_f32(_max1, 0);
#endif
                    pd[0] = absmax0 / 127.f;
                    pd[1] = absmax1 / 127.f;
                    float32x4_t _scale0 = vdupq_n_f32(absmax0 == 0.f ? 0.f : 127.f / absmax0);
                    float32x4_t _scale1 = vdupq_n_f32(absmax1 == 0.f ? 0.f : 127.f / absmax1);

                    int kk = 0;
                    for (; kk + 7 < max_kk0; kk += 8)
                    {
                        float32x4_t _p0 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0)), _scale0);
                        float32x4_t _p1 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 4)), _scale1);
                        float32x4_t _p2 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 4)), _scale0);
                        float32x4_t _p3 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 4 + 4)), _scale1);
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                        int8x8_t _r0 = float2int8(_p0, _p2);
                        int8x8_t _r1 = float2int8(_p1, _p3);
#else
                        int8x8_t _r0 = float2int8(_p0, _p1);
                        int8x8_t _r1 = float2int8(_p2, _p3);
#endif
#else
                        int16x4_t _t0 = vreinterpret_s16_s8(float2int8(_p0, _p2));
                        int16x4_t _t1 = vreinterpret_s16_s8(float2int8(_p1, _p3));
                        int16x4x2_t _t01 = vzip_s16(_t0, _t1);
                        int8x8_t _r0 = vreinterpret_s8_s16(_t01.val[0]);
                        int8x8_t _r1 = vreinterpret_s8_s16(_t01.val[1]);
#endif

                        vst1q_s8(pp, vcombine_s8(_r0, _r1));

                        pp += 16;
                        p0 += A_hstep * 8;
                    }
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        float32x4_t _p0 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0)), _scale0);
                        float32x4_t _p1 = vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 4)), _scale1);

#if __ARM_FEATURE_DOTPROD
                        int8x8_t _r01 = float2int8(_p0, _p1);
#else
                        float32x4_t _t0 = vcombine_f32(vget_low_f32(_p0), vget_low_f32(_p1));
                        float32x4_t _t1 = vcombine_f32(vget_high_f32(_p0), vget_high_f32(_p1));
                        int8x8_t _r01 = float2int8(_t0, _t1);
#endif

                        vst1_s8(pp, _r01);

                        pp += 8;
                        p0 += A_hstep * 4;
                    }
                    pd += 2;
                }
                if (elempack == 1)
                {
                    float32x2_t _absmax = vdup_n_f32(0.f);
                    const unsigned short* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        float32x2_t _p = vget_low_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0a)));
                        _absmax = vmax_f32(_absmax, vabs_f32(_p));
                        p0a += A_hstep;
                    }

                    vst1_f32(pd, vmul_n_f32(_absmax, 1.f / 127.f));
                    float absmax0 = vget_lane_f32(_absmax, 0);
                    float absmax1 = vget_lane_f32(_absmax, 1);
                    float32x2_t _scale = vdup_n_f32(absmax0 == 0.f ? 0.f : 127.f / absmax0);
                    _scale = vset_lane_f32(absmax1 == 0.f ? 0.f : 127.f / absmax1, _scale, 1);

                    int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                    for (; kk + 7 < max_kk0; kk += 8)
                    {
                        float32x2_t _p0 = vget_low_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0)));
                        float32x2_t _p1 = vget_low_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep)));
                        float32x2_t _p2 = vget_low_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 2)));
                        float32x2_t _p3 = vget_low_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 3)));
                        float32x2_t _p4 = vget_low_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 4)));
                        float32x2_t _p5 = vget_low_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 5)));
                        float32x2_t _p6 = vget_low_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 6)));
                        float32x2_t _p7 = vget_low_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 7)));
                        float32x4_t _scale_scale = vcombine_f32(_scale, _scale);
                        float32x4_t _p01 = vmulq_f32(vcombine_f32(_p0, _p1), _scale_scale);
                        float32x4_t _p23 = vmulq_f32(vcombine_f32(_p2, _p3), _scale_scale);
                        float32x4_t _p45 = vmulq_f32(vcombine_f32(_p4, _p5), _scale_scale);
                        float32x4_t _p67 = vmulq_f32(vcombine_f32(_p6, _p7), _scale_scale);
                        int8x8_t _r0 = float2int8(_p01, _p23);
                        int8x8_t _r1 = float2int8(_p45, _p67);
                        int8x8x2_t _r01 = vuzp_s8(_r0, _r1);
                        vst1q_s8(pp, vcombine_s8(_r01.val[0], _r01.val[1]));
                        pp += 16;
                        p0 += A_hstep * 8;
                    }
#endif // __ARM_FEATURE_MATMUL_INT8
#endif // __ARM_FEATURE_DOTPROD
                    for (; kk + 3 < max_kk0; kk += 4)
                    {
                        float32x2_t _p0 = vget_low_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0)));
                        float32x2_t _p1 = vget_low_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep)));
                        float32x2_t _p2 = vget_low_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 2)));
                        float32x2_t _p3 = vget_low_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 3)));
                        float32x4_t _scale_scale = vcombine_f32(_scale, _scale);
                        float32x4_t _p01 = vmulq_f32(vcombine_f32(_p0, _p1), _scale_scale);
                        float32x4_t _p23 = vmulq_f32(vcombine_f32(_p2, _p3), _scale_scale);
                        int8x8_t _r0 = float2int8(_p01, _p23);
#if __ARM_FEATURE_DOTPROD
                        int8x8x2_t _r01 = vuzp_s8(_r0, _r0);
                        vst1_s8(pp, vreinterpret_s8_s32(vzip_s32(vreinterpret_s32_s8(_r01.val[0]), vreinterpret_s32_s8(_r01.val[1])).val[0]));
#else
                        int8x8x2_t _r01 = vuzp_s8(_r0, _r0);
                        vst1_s8(pp, vreinterpret_s8_s16(vzip_s16(vreinterpret_s16_s8(_r01.val[0]), vreinterpret_s16_s8(_r01.val[1])).val[0]));
#endif
                        pp += 8;
                        p0 += A_hstep * 4;
                    }
                    for (; kk + 1 < max_kk0; kk += 2)
                    {
                        float32x2_t _p0 = vget_low_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0)));
                        float32x2_t _p1 = vget_low_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep)));
                        float32x4_t _scale_scale = vcombine_f32(_scale, _scale);
                        float32x4_t _p01 = vmulq_f32(vcombine_f32(_p0, _p1), _scale_scale);
                        int8x8_t _r0 = float2int8(_p01, _p01);
                        int8x8x2_t _r01 = vuzp_s8(_r0, _r0);
                        vst1_lane_s16((short*)pp, vreinterpret_s16_s8(_r01.val[0]), 0);
                        vst1_lane_s16((short*)pp + 1, vreinterpret_s16_s8(_r01.val[1]), 0);
                        pp += 4;
                        p0 += A_hstep * 2;
                    }
                    for (; kk < max_kk0; kk++)
                    {
                        float32x2_t _p0 = vget_low_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0)));
                        float32x4_t _p01 = vmulq_f32(vcombine_f32(_p0, _p0), vcombine_f32(_scale, _scale));
                        vst1_lane_s16((short*)pp, vreinterpret_s16_s8(float2int8(_p01, _p01)), 0);
                        pp += 2;
                        p0 += A_hstep;
                    }
#else
                float absmax0 = 0.f;
                float absmax1 = 0.f;
                const unsigned short* p0a = p0;

                for (int kk = 0; kk < max_kk0; kk++)
                {
                    float v0 = float16_to_float32(p0a[0]);
                    float v1 = float16_to_float32(p0a[1]);
                    absmax0 = std::max(absmax0, fabsf(v0));
                    absmax1 = std::max(absmax1, fabsf(v1));
                    p0a += A_hstep;
                }

                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
                const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;

                for (int kk = 0; kk < max_kk0; kk++)
                {
                    float v0 = float16_to_float32(p0[0]);
                    float v1 = float16_to_float32(p0[1]);
                    *pp++ = float2int8(v0 * scale0);
                    *pp++ = float2int8(v1 * scale1);
                    p0 += A_hstep;
                }
#endif // __ARM_NEON
                    pd += 2;
                }
            }
        }
        for (; ii < max_ii; ii++)
        {
            const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * elempack;

            for (int g = 0; g < block_count; g++)
            {
                const int max_kk0 = std::min(max_kk - g * block_size, block_size);

#if __ARM_NEON
#if __aarch64__
                if (elempack == 8)
                {
                    float absmax = 0.f;
                    const unsigned short* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk += 8)
                    {
                        uint16x8_t _p = vld1q_u16(p0a);
                        absmax = std::max(absmax, vmaxvq_f32(vabsq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_p)))));
                        absmax = std::max(absmax, vmaxvq_f32(vabsq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_p)))));
                        p0a += A_hstep * 8;
                    }

                    const float scale = absmax == 0.f ? 0.f : 127.f / absmax;
                    *pd++ = absmax / 127.f;
                    float32x4_t _scale = vdupq_n_f32(scale);

                    for (int kk = 0; kk < max_kk0; kk += 8)
                    {
                        uint16x8_t _p = vld1q_u16(p0);
                        float32x4_t _p0 = vmulq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_p)), _scale);
                        float32x4_t _p1 = vmulq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_p)), _scale);
                        vst1_s8(pp, float2int8(_p0, _p1));
                        pp += 8;
                        p0 += A_hstep * 8;
                    }
                }
#endif // __aarch64__
                if (elempack == 4)
                {
                    float32x4_t _absmax = vdupq_n_f32(0.f);
                    const unsigned short* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk += 4)
                    {
                        float32x4_t _p = vcvt_f32_f16((float16x4_t)vld1_u16(p0a));
                        _absmax = vmaxq_f32(_absmax, vabsq_f32(_p));
                        p0a += A_hstep * 4;
                    }
#if __aarch64__
                    float absmax = vmaxvq_f32(_absmax);
#else
                    float32x2_t _max = vmax_f32(vget_low_f32(_absmax), vget_high_f32(_absmax));
                    _max = vpmax_f32(_max, _max);
                    float absmax = vget_lane_f32(_max, 0);
#endif
                    const float scale = absmax == 0.f ? 0.f : 127.f / absmax;
                    *pd++ = absmax / 127.f;

                    float32x4_t _scale = vdupq_n_f32(scale);
                    for (int kk = 0; kk < max_kk0; kk += 4)
                    {
                        float32x4_t _p = vcvt_f32_f16((float16x4_t)vld1_u16(p0));
                        int8x8_t _r = float2int8(vmulq_f32(_p, _scale), vmulq_f32(_p, _scale));
                        vst1_lane_s32((int*)pp, vreinterpret_s32_s8(_r), 0);
                        pp += 4;
                        p0 += A_hstep * 4;
                    }
                }
#endif // __ARM_NEON
                if (elempack == 1)
                {
                    float absmax = 0.f;
                    const unsigned short* p0a = p0;
                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        float v = float16_to_float32(*p0a);
                        absmax = std::max(absmax, fabsf(v));
                        p0a += A_hstep;
                    }

                    if (absmax == 0.f)
                    {
                        *pd++ = 0.f;
                        for (int kk0 = 0; kk0 < max_kk0; kk0++)
                            *pp++ = 0;
                        p0 += (size_t)max_kk0 * A_hstep;
                        continue;
                    }

                    const float scale = 127.f / absmax;
                    *pd++ = absmax / 127.f;

                    for (int kk = 0; kk < max_kk0; kk++)
                    {
                        float v = float16_to_float32(*p0);
                        *pp++ = float2int8(v * scale);
                        p0 += A_hstep;
                    }
                }
            }
        }

        return;
    }

    int ii = 0;
#if __ARM_NEON
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * elempack;
        const float* ps = (const float*)input_scales + k;
        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
            if (elempack == 8)
            {
                float32x4_t _absmax0 = vdupq_n_f32(0.f);
                float32x4_t _absmax1 = vdupq_n_f32(0.f);
                float32x4_t _absmax2 = vdupq_n_f32(0.f);
                float32x4_t _absmax3 = vdupq_n_f32(0.f);
                float32x4_t _absmax4 = vdupq_n_f32(0.f);
                float32x4_t _absmax5 = vdupq_n_f32(0.f);
                float32x4_t _absmax6 = vdupq_n_f32(0.f);
                float32x4_t _absmax7 = vdupq_n_f32(0.f);
                const unsigned short* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk += 8)
                {
                    uint16x8_t _p = vld1q_u16(p0a);
                    uint16x8_t _q = vld1q_u16(p0a + 8);
                    uint16x8_t _r = vld1q_u16(p0a + 16);
                    uint16x8_t _s = vld1q_u16(p0a + 24);
                    uint16x8_t _t = vld1q_u16(p0a + 32);
                    uint16x8_t _u = vld1q_u16(p0a + 40);
                    uint16x8_t _v = vld1q_u16(p0a + 48);
                    uint16x8_t _w = vld1q_u16(p0a + 56);
                    float32x4_t _s0 = vld1q_f32(psa);
                    float32x4_t _s1 = vld1q_f32(psa + 4);
                    _absmax0 = vmaxq_f32(_absmax0, vmulq_f32(vabsq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_p))), _s0));
                    _absmax0 = vmaxq_f32(_absmax0, vmulq_f32(vabsq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_p))), _s1));
                    _absmax1 = vmaxq_f32(_absmax1, vmulq_f32(vabsq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_q))), _s0));
                    _absmax1 = vmaxq_f32(_absmax1, vmulq_f32(vabsq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_q))), _s1));
                    _absmax2 = vmaxq_f32(_absmax2, vmulq_f32(vabsq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_r))), _s0));
                    _absmax2 = vmaxq_f32(_absmax2, vmulq_f32(vabsq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_r))), _s1));
                    _absmax3 = vmaxq_f32(_absmax3, vmulq_f32(vabsq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_s))), _s0));
                    _absmax3 = vmaxq_f32(_absmax3, vmulq_f32(vabsq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_s))), _s1));
                    _absmax4 = vmaxq_f32(_absmax4, vmulq_f32(vabsq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_t))), _s0));
                    _absmax4 = vmaxq_f32(_absmax4, vmulq_f32(vabsq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_t))), _s1));
                    _absmax5 = vmaxq_f32(_absmax5, vmulq_f32(vabsq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_u))), _s0));
                    _absmax5 = vmaxq_f32(_absmax5, vmulq_f32(vabsq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_u))), _s1));
                    _absmax6 = vmaxq_f32(_absmax6, vmulq_f32(vabsq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_v))), _s0));
                    _absmax6 = vmaxq_f32(_absmax6, vmulq_f32(vabsq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_v))), _s1));
                    _absmax7 = vmaxq_f32(_absmax7, vmulq_f32(vabsq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_w))), _s0));
                    _absmax7 = vmaxq_f32(_absmax7, vmulq_f32(vabsq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_w))), _s1));
                    p0a += A_hstep * 8;
                    psa += 8;
                }

                float32x2_t _max0 = vpmax_f32(vget_low_f32(_absmax0), vget_high_f32(_absmax0));
                float32x2_t _max1 = vpmax_f32(vget_low_f32(_absmax1), vget_high_f32(_absmax1));
                float32x2_t _max2 = vpmax_f32(vget_low_f32(_absmax2), vget_high_f32(_absmax2));
                float32x2_t _max3 = vpmax_f32(vget_low_f32(_absmax3), vget_high_f32(_absmax3));
                float32x2_t _max4 = vpmax_f32(vget_low_f32(_absmax4), vget_high_f32(_absmax4));
                float32x2_t _max5 = vpmax_f32(vget_low_f32(_absmax5), vget_high_f32(_absmax5));
                float32x2_t _max6 = vpmax_f32(vget_low_f32(_absmax6), vget_high_f32(_absmax6));
                float32x2_t _max7 = vpmax_f32(vget_low_f32(_absmax7), vget_high_f32(_absmax7));
                _absmax0 = vcombine_f32(vpmax_f32(_max0, _max1), vpmax_f32(_max2, _max3));
                _absmax1 = vcombine_f32(vpmax_f32(_max4, _max5), vpmax_f32(_max6, _max7));

                vst1q_f32(pd, vmulq_n_f32(_absmax0, 1.f / 127.f));
                vst1q_f32(pd + 4, vmulq_n_f32(_absmax1, 1.f / 127.f));

                float32x4_t _zero = vdupq_n_f32(0.f);
                float32x4_t _scale0 = vdivq_f32(vdupq_n_f32(127.f), _absmax0);
                float32x4_t _scale1 = vdivq_f32(vdupq_n_f32(127.f), _absmax1);
                _scale0 = vbslq_f32(vceqq_f32(_absmax0, _zero), _zero, _scale0);
                _scale1 = vbslq_f32(vceqq_f32(_absmax1, _zero), _zero, _scale1);

                int kk = 0;
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    uint16x8_t _p = vld1q_u16(p0);
                    uint16x8_t _q = vld1q_u16(p0 + 8);
                    uint16x8_t _r = vld1q_u16(p0 + 16);
                    uint16x8_t _s = vld1q_u16(p0 + 24);
                    uint16x8_t _t = vld1q_u16(p0 + 32);
                    uint16x8_t _u = vld1q_u16(p0 + 40);
                    uint16x8_t _v = vld1q_u16(p0 + 48);
                    uint16x8_t _w = vld1q_u16(p0 + 56);
                    float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vget_low_u16(_p));
                    float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vget_high_u16(_p));
                    float32x4_t _p2 = vcvt_f32_f16((float16x4_t)vget_low_u16(_q));
                    float32x4_t _p3 = vcvt_f32_f16((float16x4_t)vget_high_u16(_q));
                    float32x4_t _p4 = vcvt_f32_f16((float16x4_t)vget_low_u16(_r));
                    float32x4_t _p5 = vcvt_f32_f16((float16x4_t)vget_high_u16(_r));
                    float32x4_t _p6 = vcvt_f32_f16((float16x4_t)vget_low_u16(_s));
                    float32x4_t _p7 = vcvt_f32_f16((float16x4_t)vget_high_u16(_s));
                    float32x4_t _p8 = vcvt_f32_f16((float16x4_t)vget_low_u16(_t));
                    float32x4_t _p9 = vcvt_f32_f16((float16x4_t)vget_high_u16(_t));
                    float32x4_t _pa = vcvt_f32_f16((float16x4_t)vget_low_u16(_u));
                    float32x4_t _pb = vcvt_f32_f16((float16x4_t)vget_high_u16(_u));
                    float32x4_t _pc = vcvt_f32_f16((float16x4_t)vget_low_u16(_v));
                    float32x4_t _pd = vcvt_f32_f16((float16x4_t)vget_high_u16(_v));
                    float32x4_t _pe = vcvt_f32_f16((float16x4_t)vget_low_u16(_w));
                    float32x4_t _pf = vcvt_f32_f16((float16x4_t)vget_high_u16(_w));

                    float32x4_t _s0 = vld1q_f32(ps);
                    float32x4_t _s1 = vld1q_f32(ps + 4);
                    _p0 = vmulq_f32(_p0, _s0);
                    _p1 = vmulq_f32(_p1, _s1);
                    _p2 = vmulq_f32(_p2, _s0);
                    _p3 = vmulq_f32(_p3, _s1);
                    _p4 = vmulq_f32(_p4, _s0);
                    _p5 = vmulq_f32(_p5, _s1);
                    _p6 = vmulq_f32(_p6, _s0);
                    _p7 = vmulq_f32(_p7, _s1);
                    _p8 = vmulq_f32(_p8, _s0);
                    _p9 = vmulq_f32(_p9, _s1);
                    _pa = vmulq_f32(_pa, _s0);
                    _pb = vmulq_f32(_pb, _s1);
                    _pc = vmulq_f32(_pc, _s0);
                    _pd = vmulq_f32(_pd, _s1);
                    _pe = vmulq_f32(_pe, _s0);
                    _pf = vmulq_f32(_pf, _s1);

                    _p0 = vmulq_laneq_f32(_p0, _scale0, 0);
                    _p1 = vmulq_laneq_f32(_p1, _scale0, 0);
                    _p2 = vmulq_laneq_f32(_p2, _scale0, 1);
                    _p3 = vmulq_laneq_f32(_p3, _scale0, 1);
                    _p4 = vmulq_laneq_f32(_p4, _scale0, 2);
                    _p5 = vmulq_laneq_f32(_p5, _scale0, 2);
                    _p6 = vmulq_laneq_f32(_p6, _scale0, 3);
                    _p7 = vmulq_laneq_f32(_p7, _scale0, 3);
                    _p8 = vmulq_laneq_f32(_p8, _scale1, 0);
                    _p9 = vmulq_laneq_f32(_p9, _scale1, 0);
                    _pa = vmulq_laneq_f32(_pa, _scale1, 1);
                    _pb = vmulq_laneq_f32(_pb, _scale1, 1);
                    _pc = vmulq_laneq_f32(_pc, _scale1, 2);
                    _pd = vmulq_laneq_f32(_pd, _scale1, 2);
                    _pe = vmulq_laneq_f32(_pe, _scale1, 3);
                    _pf = vmulq_laneq_f32(_pf, _scale1, 3);

#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                    int8x8_t _r0 = float2int8(_p0, _p1);
                    int8x8_t _r1 = float2int8(_p2, _p3);
                    int8x8_t _r2 = float2int8(_p4, _p5);
                    int8x8_t _r3 = float2int8(_p6, _p7);
                    int8x8_t _r4 = float2int8(_p8, _p9);
                    int8x8_t _r5 = float2int8(_pa, _pb);
                    int8x8_t _r6 = float2int8(_pc, _pd);
                    int8x8_t _r7 = float2int8(_pe, _pf);

                    vst1q_s8(pp, vcombine_s8(_r0, _r1));
                    vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
                    vst1q_s8(pp + 32, vcombine_s8(_r4, _r5));
                    vst1q_s8(pp + 48, vcombine_s8(_r6, _r7));
#else  // __ARM_FEATURE_MATMUL_INT8
                    int8x8_t _r0 = float2int8(_p0, _p2);
                    int8x8_t _r1 = float2int8(_p4, _p6);
                    int8x8_t _r2 = float2int8(_p8, _pa);
                    int8x8_t _r3 = float2int8(_pc, _pe);
                    int8x8_t _r4 = float2int8(_p1, _p3);
                    int8x8_t _r5 = float2int8(_p5, _p7);
                    int8x8_t _r6 = float2int8(_p9, _pb);
                    int8x8_t _r7 = float2int8(_pd, _pf);

                    vst1q_s8(pp, vcombine_s8(_r0, _r1));
                    vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
                    vst1q_s8(pp + 32, vcombine_s8(_r4, _r5));
                    vst1q_s8(pp + 48, vcombine_s8(_r6, _r7));
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
                    int8x8_t _r0 = float2int8(_p0, _p2);
                    int8x8_t _r1 = float2int8(_p4, _p6);
                    int8x8_t _r2 = float2int8(_p8, _pa);
                    int8x8_t _r3 = float2int8(_pc, _pe);
                    int8x8_t _r4 = float2int8(_p1, _p3);
                    int8x8_t _r5 = float2int8(_p5, _p7);
                    int8x8_t _r6 = float2int8(_p9, _pb);
                    int8x8_t _r7 = float2int8(_pd, _pf);

                    int16x8_t _r01 = vreinterpretq_s16_s8(vcombine_s8(_r0, _r1));
                    int16x8_t _r23 = vreinterpretq_s16_s8(vcombine_s8(_r2, _r3));
                    int16x8_t _r45 = vreinterpretq_s16_s8(vcombine_s8(_r4, _r5));
                    int16x8_t _r67 = vreinterpretq_s16_s8(vcombine_s8(_r6, _r7));
                    int16x8x2_t _rr0 = vuzpq_s16(_r01, _r23);
                    int16x8x2_t _rr1 = vuzpq_s16(_r45, _r67);

                    vst1q_s8(pp, vreinterpretq_s8_s16(_rr0.val[0]));
                    vst1q_s8(pp + 16, vreinterpretq_s8_s16(_rr0.val[1]));
                    vst1q_s8(pp + 32, vreinterpretq_s8_s16(_rr1.val[0]));
                    vst1q_s8(pp + 48, vreinterpretq_s8_s16(_rr1.val[1]));
#endif // __ARM_FEATURE_DOTPROD

                    pp += 64;
                    p0 += A_hstep * 8;
                    ps += 8;
                }
                pd += 8;
            }
            if (elempack == 4)
            {
                float32x4_t _absmax0 = vdupq_n_f32(0.f);
                float32x4_t _absmax1 = vdupq_n_f32(0.f);
                float32x4_t _absmax2 = vdupq_n_f32(0.f);
                float32x4_t _absmax3 = vdupq_n_f32(0.f);
                float32x4_t _absmax4 = vdupq_n_f32(0.f);
                float32x4_t _absmax5 = vdupq_n_f32(0.f);
                float32x4_t _absmax6 = vdupq_n_f32(0.f);
                float32x4_t _absmax7 = vdupq_n_f32(0.f);
                const unsigned short* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk += 4)
                {
                    float32x4_t _s = vld1q_f32(psa);
                    _absmax0 = vmaxq_f32(_absmax0, vmulq_f32(vabsq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0a))), _s));
                    _absmax1 = vmaxq_f32(_absmax1, vmulq_f32(vabsq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0a + 4))), _s));
                    _absmax2 = vmaxq_f32(_absmax2, vmulq_f32(vabsq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0a + 8))), _s));
                    _absmax3 = vmaxq_f32(_absmax3, vmulq_f32(vabsq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0a + 12))), _s));
                    _absmax4 = vmaxq_f32(_absmax4, vmulq_f32(vabsq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0a + 16))), _s));
                    _absmax5 = vmaxq_f32(_absmax5, vmulq_f32(vabsq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0a + 20))), _s));
                    _absmax6 = vmaxq_f32(_absmax6, vmulq_f32(vabsq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0a + 24))), _s));
                    _absmax7 = vmaxq_f32(_absmax7, vmulq_f32(vabsq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0a + 28))), _s));
                    p0a += A_hstep * 4;
                    psa += 4;
                }

                float32x2_t _max0 = vpmax_f32(vget_low_f32(_absmax0), vget_high_f32(_absmax0));
                float32x2_t _max1 = vpmax_f32(vget_low_f32(_absmax1), vget_high_f32(_absmax1));
                float32x2_t _max2 = vpmax_f32(vget_low_f32(_absmax2), vget_high_f32(_absmax2));
                float32x2_t _max3 = vpmax_f32(vget_low_f32(_absmax3), vget_high_f32(_absmax3));
                float32x2_t _max4 = vpmax_f32(vget_low_f32(_absmax4), vget_high_f32(_absmax4));
                float32x2_t _max5 = vpmax_f32(vget_low_f32(_absmax5), vget_high_f32(_absmax5));
                float32x2_t _max6 = vpmax_f32(vget_low_f32(_absmax6), vget_high_f32(_absmax6));
                float32x2_t _max7 = vpmax_f32(vget_low_f32(_absmax7), vget_high_f32(_absmax7));
                _absmax0 = vcombine_f32(vpmax_f32(_max0, _max1), vpmax_f32(_max2, _max3));
                _absmax1 = vcombine_f32(vpmax_f32(_max4, _max5), vpmax_f32(_max6, _max7));

                vst1q_f32(pd, vmulq_n_f32(_absmax0, 1.f / 127.f));
                vst1q_f32(pd + 4, vmulq_n_f32(_absmax1, 1.f / 127.f));

                float32x4_t _zero = vdupq_n_f32(0.f);
                float32x4_t _scale0 = vdivq_f32(vdupq_n_f32(127.f), _absmax0);
                float32x4_t _scale1 = vdivq_f32(vdupq_n_f32(127.f), _absmax1);
                _scale0 = vbslq_f32(vceqq_f32(_absmax0, _zero), _zero, _scale0);
                _scale1 = vbslq_f32(vceqq_f32(_absmax1, _zero), _zero, _scale1);

                int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    float32x4_t _s0 = vld1q_f32(ps);
                    float32x4_t _s1 = vld1q_f32(ps + 4);
                    float32x4_t _p00 = vmulq_laneq_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0)), _s0), _scale0, 0);
                    float32x4_t _p01 = vmulq_laneq_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 4)), _s1), _scale0, 0);
                    float32x4_t _p10 = vmulq_laneq_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 4)), _s0), _scale0, 1);
                    float32x4_t _p11 = vmulq_laneq_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 4 + 4)), _s1), _scale0, 1);
                    int8x8_t _r0 = float2int8(_p00, _p01);
                    int8x8_t _r1 = float2int8(_p10, _p11);
                    vst1q_s8(pp, vcombine_s8(_r0, _r1));
                    float32x4_t _p20 = vmulq_laneq_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 8)), _s0), _scale0, 2);
                    float32x4_t _p21 = vmulq_laneq_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 4 + 8)), _s1), _scale0, 2);
                    float32x4_t _p30 = vmulq_laneq_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 12)), _s0), _scale0, 3);
                    float32x4_t _p31 = vmulq_laneq_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 4 + 12)), _s1), _scale0, 3);
                    int8x8_t _r2 = float2int8(_p20, _p21);
                    int8x8_t _r3 = float2int8(_p30, _p31);
                    vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
                    float32x4_t _p40 = vmulq_laneq_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 16)), _s0), _scale1, 0);
                    float32x4_t _p41 = vmulq_laneq_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 4 + 16)), _s1), _scale1, 0);
                    float32x4_t _p50 = vmulq_laneq_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 20)), _s0), _scale1, 1);
                    float32x4_t _p51 = vmulq_laneq_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 4 + 20)), _s1), _scale1, 1);
                    int8x8_t _r4 = float2int8(_p40, _p41);
                    int8x8_t _r5 = float2int8(_p50, _p51);
                    vst1q_s8(pp + 32, vcombine_s8(_r4, _r5));
                    float32x4_t _p60 = vmulq_laneq_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 24)), _s0), _scale1, 2);
                    float32x4_t _p61 = vmulq_laneq_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 4 + 24)), _s1), _scale1, 2);
                    float32x4_t _p70 = vmulq_laneq_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 28)), _s0), _scale1, 3);
                    float32x4_t _p71 = vmulq_laneq_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 4 + 28)), _s1), _scale1, 3);
                    int8x8_t _r6 = float2int8(_p60, _p61);
                    int8x8_t _r7 = float2int8(_p70, _p71);
                    vst1q_s8(pp + 48, vcombine_s8(_r6, _r7));
                    pp += 64;
                    p0 += A_hstep * 8;
                    ps += 8;
                }
#endif // __ARM_FEATURE_MATMUL_INT8
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float32x4_t _s = vld1q_f32(ps);
                    float32x4_t _p0 = vmulq_laneq_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0)), _s), _scale0, 0);
                    float32x4_t _p1 = vmulq_laneq_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 4)), _s), _scale0, 1);
                    float32x4_t _p2 = vmulq_laneq_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 8)), _s), _scale0, 2);
                    float32x4_t _p3 = vmulq_laneq_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 12)), _s), _scale0, 3);
                    float32x4_t _p4 = vmulq_laneq_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 16)), _s), _scale1, 0);
                    float32x4_t _p5 = vmulq_laneq_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 20)), _s), _scale1, 1);
                    float32x4_t _p6 = vmulq_laneq_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 24)), _s), _scale1, 2);
                    float32x4_t _p7 = vmulq_laneq_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 28)), _s), _scale1, 3);
                    int8x8_t _r0 = float2int8(_p0, _p1);
                    int8x8_t _r1 = float2int8(_p2, _p3);
                    int8x8_t _r2 = float2int8(_p4, _p5);
                    int8x8_t _r3 = float2int8(_p6, _p7);
#if __ARM_FEATURE_DOTPROD
                    vst1q_s8(pp, vcombine_s8(_r0, _r1));
                    vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
#else
                    int16x8_t _r01 = vreinterpretq_s16_s8(vcombine_s8(_r0, _r1));
                    int16x8_t _r23 = vreinterpretq_s16_s8(vcombine_s8(_r2, _r3));
                    int16x8x2_t _rr = vuzpq_s16(_r01, _r23);
                    vst1q_s8(pp, vreinterpretq_s8_s16(_rr.val[0]));
                    vst1q_s8(pp + 16, vreinterpretq_s8_s16(_rr.val[1]));
#endif
                    pp += 32;
                    p0 += A_hstep * 4;
                    ps += 4;
                }
                pd += 8;
            }
            if (elempack == 1)
            {
                float32x4_t _absmax0 = vdupq_n_f32(0.f);
                float32x4_t _absmax1 = vdupq_n_f32(0.f);
                const unsigned short* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    const float s = *psa++;
                    float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a));
                    _absmax0 = vmaxq_f32(_absmax0, vmulq_n_f32(vabsq_f32(_p0), s));
                    float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a + 4));
                    _absmax1 = vmaxq_f32(_absmax1, vmulq_n_f32(vabsq_f32(_p1), s));
                    p0a += A_hstep;
                }

                vst1q_f32(pd, vmulq_n_f32(_absmax0, 1.f / 127.f));
                vst1q_f32(pd + 4, vmulq_n_f32(_absmax1, 1.f / 127.f));

                float32x4_t _zero = vdupq_n_f32(0.f);
                float32x4_t _scale0 = vdivq_f32(vdupq_n_f32(127.f), _absmax0);
                float32x4_t _scale1 = vdivq_f32(vdupq_n_f32(127.f), _absmax1);
                _scale0 = vbslq_f32(vceqq_f32(_absmax0, _zero), _zero, _scale0);
                _scale1 = vbslq_f32(vceqq_f32(_absmax1, _zero), _zero, _scale1);

                int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    float32x4_t _p00 = vcvt_f32_f16((float16x4_t)vld1_u16(p0));
                    float32x4_t _p01 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 4));
                    const float s0 = *ps++;
                    _p00 = vmulq_n_f32(_p00, s0);
                    _p01 = vmulq_n_f32(_p01, s0);
                    float32x4_t _p10 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep));
                    float32x4_t _p11 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep + 4));
                    const float s1 = *ps++;
                    _p10 = vmulq_n_f32(_p10, s1);
                    _p11 = vmulq_n_f32(_p11, s1);
                    float32x4_t _p20 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 2));
                    float32x4_t _p21 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 2 + 4));
                    const float s2 = *ps++;
                    _p20 = vmulq_n_f32(_p20, s2);
                    _p21 = vmulq_n_f32(_p21, s2);
                    float32x4_t _p30 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 3));
                    float32x4_t _p31 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 3 + 4));
                    const float s3 = *ps++;
                    _p30 = vmulq_n_f32(_p30, s3);
                    _p31 = vmulq_n_f32(_p31, s3);
                    float32x4_t _p40 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 4));
                    float32x4_t _p41 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 4 + 4));
                    const float s4 = *ps++;
                    _p40 = vmulq_n_f32(_p40, s4);
                    _p41 = vmulq_n_f32(_p41, s4);
                    float32x4_t _p50 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 5));
                    float32x4_t _p51 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 5 + 4));
                    const float s5 = *ps++;
                    _p50 = vmulq_n_f32(_p50, s5);
                    _p51 = vmulq_n_f32(_p51, s5);
                    float32x4_t _p60 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 6));
                    float32x4_t _p61 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 6 + 4));
                    const float s6 = *ps++;
                    _p60 = vmulq_n_f32(_p60, s6);
                    _p61 = vmulq_n_f32(_p61, s6);
                    float32x4_t _p70 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 7));
                    float32x4_t _p71 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 7 + 4));
                    const float s7 = *ps++;
                    _p70 = vmulq_n_f32(_p70, s7);
                    _p71 = vmulq_n_f32(_p71, s7);
                    int8x8_t _r0 = float2int8(vmulq_f32(_p00, _scale0), vmulq_f32(_p01, _scale1));
                    int8x8_t _r1 = float2int8(vmulq_f32(_p10, _scale0), vmulq_f32(_p11, _scale1));
                    int8x8_t _r2 = float2int8(vmulq_f32(_p20, _scale0), vmulq_f32(_p21, _scale1));
                    int8x8_t _r3 = float2int8(vmulq_f32(_p30, _scale0), vmulq_f32(_p31, _scale1));
                    int8x8_t _r4 = float2int8(vmulq_f32(_p40, _scale0), vmulq_f32(_p41, _scale1));
                    int8x8_t _r5 = float2int8(vmulq_f32(_p50, _scale0), vmulq_f32(_p51, _scale1));
                    int8x8_t _r6 = float2int8(vmulq_f32(_p60, _scale0), vmulq_f32(_p61, _scale1));
                    int8x8_t _r7 = float2int8(vmulq_f32(_p70, _scale0), vmulq_f32(_p71, _scale1));
                    int8x8x2_t _r04 = vzip_s8(_r0, _r4);
                    int8x8x2_t _r15 = vzip_s8(_r1, _r5);
                    int8x8x2_t _r26 = vzip_s8(_r2, _r6);
                    int8x8x2_t _r37 = vzip_s8(_r3, _r7);
                    int8x8x4_t _r0123;
                    _r0123.val[0] = _r04.val[0];
                    _r0123.val[1] = _r15.val[0];
                    _r0123.val[2] = _r26.val[0];
                    _r0123.val[3] = _r37.val[0];
                    int8x8x4_t _r4567;
                    _r4567.val[0] = _r04.val[1];
                    _r4567.val[1] = _r15.val[1];
                    _r4567.val[2] = _r26.val[1];
                    _r4567.val[3] = _r37.val[1];
                    vst4_s8(pp, _r0123);
                    vst4_s8(pp + 32, _r4567);
                    pp += 64;
                    p0 += A_hstep * 8;
                }
#endif // __ARM_FEATURE_MATMUL_INT8
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float32x4_t _p00 = vcvt_f32_f16((float16x4_t)vld1_u16(p0));
                    float32x4_t _p01 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 4));
                    const float s0 = *ps++;
                    _p00 = vmulq_n_f32(_p00, s0);
                    _p01 = vmulq_n_f32(_p01, s0);
                    float32x4_t _p10 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep));
                    float32x4_t _p11 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep + 4));
                    const float s1 = *ps++;
                    _p10 = vmulq_n_f32(_p10, s1);
                    _p11 = vmulq_n_f32(_p11, s1);
                    float32x4_t _p20 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 2));
                    float32x4_t _p21 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 2 + 4));
                    const float s2 = *ps++;
                    _p20 = vmulq_n_f32(_p20, s2);
                    _p21 = vmulq_n_f32(_p21, s2);
                    float32x4_t _p30 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 3));
                    float32x4_t _p31 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 3 + 4));
                    const float s3 = *ps++;
                    _p30 = vmulq_n_f32(_p30, s3);
                    _p31 = vmulq_n_f32(_p31, s3);
                    int8x8_t _r0 = float2int8(vmulq_f32(_p00, _scale0), vmulq_f32(_p01, _scale1));
                    int8x8_t _r1 = float2int8(vmulq_f32(_p10, _scale0), vmulq_f32(_p11, _scale1));
                    int8x8_t _r2 = float2int8(vmulq_f32(_p20, _scale0), vmulq_f32(_p21, _scale1));
                    int8x8_t _r3 = float2int8(vmulq_f32(_p30, _scale0), vmulq_f32(_p31, _scale1));
#if __ARM_FEATURE_DOTPROD
                    int8x8x4_t _r0123;
                    _r0123.val[0] = _r0;
                    _r0123.val[1] = _r1;
                    _r0123.val[2] = _r2;
                    _r0123.val[3] = _r3;
                    vst4_s8(pp, _r0123);
#else
                    int8x8x2_t _r01;
                    _r01.val[0] = _r0;
                    _r01.val[1] = _r1;
                    int8x8x2_t _r23;
                    _r23.val[0] = _r2;
                    _r23.val[1] = _r3;
                    vst2_s8(pp, _r01);
                    vst2_s8(pp + 16, _r23);
#endif
                    pp += 32;
                    p0 += A_hstep * 4;
                }
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    float32x4_t _p00 = vcvt_f32_f16((float16x4_t)vld1_u16(p0));
                    float32x4_t _p01 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 4));
                    const float s0 = *ps++;
                    _p00 = vmulq_n_f32(_p00, s0);
                    _p01 = vmulq_n_f32(_p01, s0);
                    float32x4_t _p10 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep));
                    float32x4_t _p11 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep + 4));
                    const float s1 = *ps++;
                    _p10 = vmulq_n_f32(_p10, s1);
                    _p11 = vmulq_n_f32(_p11, s1);
                    int8x8_t _r0 = float2int8(vmulq_f32(_p00, _scale0), vmulq_f32(_p01, _scale1));
                    int8x8_t _r1 = float2int8(vmulq_f32(_p10, _scale0), vmulq_f32(_p11, _scale1));
                    int8x8x2_t _r01;
                    _r01.val[0] = _r0;
                    _r01.val[1] = _r1;
                    vst2_s8(pp, _r01);
                    pp += 16;
                    p0 += A_hstep * 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vld1_u16(p0));
                    float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 4));
                    const float s = *ps++;
                    _p0 = vmulq_n_f32(_p0, s);
                    _p1 = vmulq_n_f32(_p1, s);
                    vst1_s8(pp, float2int8(vmulq_f32(_p0, _scale0), vmulq_f32(_p1, _scale1)));
                    pp += 8;
                    p0 += A_hstep;
                }
                pd += 8;
            }
        }
    }
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * elempack;
        const float* ps = (const float*)input_scales + k;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);

#if __aarch64__
            if (elempack == 8)
            {
                float32x4_t _absmax0 = vdupq_n_f32(0.f);
                float32x4_t _absmax1 = vdupq_n_f32(0.f);
                float32x4_t _absmax2 = vdupq_n_f32(0.f);
                float32x4_t _absmax3 = vdupq_n_f32(0.f);
                const unsigned short* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk += 8)
                {
                    float32x4_t _s0 = vld1q_f32(psa);
                    float32x4_t _s1 = vld1q_f32(psa + 4);
                    uint16x8_t _p = vld1q_u16(p0a);
                    uint16x8_t _q = vld1q_u16(p0a + 8);
                    uint16x8_t _r = vld1q_u16(p0a + 16);
                    uint16x8_t _s = vld1q_u16(p0a + 24);
                    _absmax0 = vmaxq_f32(_absmax0, vmulq_f32(vabsq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_p))), _s0));
                    _absmax0 = vmaxq_f32(_absmax0, vmulq_f32(vabsq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_p))), _s1));
                    _absmax1 = vmaxq_f32(_absmax1, vmulq_f32(vabsq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_q))), _s0));
                    _absmax1 = vmaxq_f32(_absmax1, vmulq_f32(vabsq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_q))), _s1));
                    _absmax2 = vmaxq_f32(_absmax2, vmulq_f32(vabsq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_r))), _s0));
                    _absmax2 = vmaxq_f32(_absmax2, vmulq_f32(vabsq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_r))), _s1));
                    _absmax3 = vmaxq_f32(_absmax3, vmulq_f32(vabsq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_s))), _s0));
                    _absmax3 = vmaxq_f32(_absmax3, vmulq_f32(vabsq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_s))), _s1));
                    p0a += A_hstep * 8;
                    psa += 8;
                }

                float32x2_t _max0 = vpmax_f32(vget_low_f32(_absmax0), vget_high_f32(_absmax0));
                float32x2_t _max1 = vpmax_f32(vget_low_f32(_absmax1), vget_high_f32(_absmax1));
                float32x2_t _max2 = vpmax_f32(vget_low_f32(_absmax2), vget_high_f32(_absmax2));
                float32x2_t _max3 = vpmax_f32(vget_low_f32(_absmax3), vget_high_f32(_absmax3));
                _absmax0 = vcombine_f32(vpmax_f32(_max0, _max1), vpmax_f32(_max2, _max3));

                vst1q_f32(pd, vmulq_n_f32(_absmax0, 1.f / 127.f));

                float32x4_t _zero = vdupq_n_f32(0.f);
                float32x4_t _scale = vdivq_f32(vdupq_n_f32(127.f), _absmax0);
                _scale = vbslq_f32(vceqq_f32(_absmax0, _zero), _zero, _scale);

                for (int kk = 0; kk < max_kk0; kk += 8)
                {
                    float32x4_t _s0 = vld1q_f32(ps);
                    float32x4_t _s1 = vld1q_f32(ps + 4);
                    uint16x8_t _p = vld1q_u16(p0);
                    uint16x8_t _q = vld1q_u16(p0 + 8);
                    uint16x8_t _r = vld1q_u16(p0 + 16);
                    uint16x8_t _s = vld1q_u16(p0 + 24);
                    float32x4_t _p0 = vmulq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_p)), _s0);
                    float32x4_t _p1 = vmulq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_p)), _s1);
                    float32x4_t _p2 = vmulq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_q)), _s0);
                    float32x4_t _p3 = vmulq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_q)), _s1);
                    float32x4_t _p4 = vmulq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_r)), _s0);
                    float32x4_t _p5 = vmulq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_r)), _s1);
                    float32x4_t _p6 = vmulq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_s)), _s0);
                    float32x4_t _p7 = vmulq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_s)), _s1);
                    _p0 = vmulq_laneq_f32(_p0, _scale, 0);
                    _p1 = vmulq_laneq_f32(_p1, _scale, 0);
                    _p2 = vmulq_laneq_f32(_p2, _scale, 1);
                    _p3 = vmulq_laneq_f32(_p3, _scale, 1);
                    _p4 = vmulq_laneq_f32(_p4, _scale, 2);
                    _p5 = vmulq_laneq_f32(_p5, _scale, 2);
                    _p6 = vmulq_laneq_f32(_p6, _scale, 3);
                    _p7 = vmulq_laneq_f32(_p7, _scale, 3);
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                    int8x8_t _r0 = float2int8(_p0, _p1);
                    int8x8_t _r1 = float2int8(_p2, _p3);
                    int8x8_t _r2 = float2int8(_p4, _p5);
                    int8x8_t _r3 = float2int8(_p6, _p7);
#else
                    int8x8_t _r0 = float2int8(_p0, _p2);
                    int8x8_t _r1 = float2int8(_p4, _p6);
                    int8x8_t _r2 = float2int8(_p1, _p3);
                    int8x8_t _r3 = float2int8(_p5, _p7);
#endif
#else
                    int16x4_t _t0 = vreinterpret_s16_s8(float2int8(_p0, _p2));
                    int16x4_t _t1 = vreinterpret_s16_s8(float2int8(_p4, _p6));
                    int16x4_t _t2 = vreinterpret_s16_s8(float2int8(_p1, _p3));
                    int16x4_t _t3 = vreinterpret_s16_s8(float2int8(_p5, _p7));
                    int16x4x2_t _t01 = vuzp_s16(_t0, _t1);
                    int16x4x2_t _t23 = vuzp_s16(_t2, _t3);
                    int8x8_t _r0 = vreinterpret_s8_s16(_t01.val[0]);
                    int8x8_t _r1 = vreinterpret_s8_s16(_t01.val[1]);
                    int8x8_t _r2 = vreinterpret_s8_s16(_t23.val[0]);
                    int8x8_t _r3 = vreinterpret_s8_s16(_t23.val[1]);
#endif

                    vst1q_s8(pp, vcombine_s8(_r0, _r1));
                    vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));

                    pp += 32;
                    p0 += A_hstep * 8;
                    ps += 8;
                }
                pd += 4;
            }
#endif // __aarch64__
            if (elempack == 4)
            {
                float32x4_t _absmax0 = vdupq_n_f32(0.f);
                float32x4_t _absmax1 = vdupq_n_f32(0.f);
                float32x4_t _absmax2 = vdupq_n_f32(0.f);
                float32x4_t _absmax3 = vdupq_n_f32(0.f);
                const unsigned short* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk += 4)
                {
                    float32x4_t _s = vld1q_f32(psa);
                    float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a));
                    float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a + 4));
                    float32x4_t _p2 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a + 8));
                    float32x4_t _p3 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a + 12));
                    _absmax0 = vmaxq_f32(_absmax0, vmulq_f32(vabsq_f32(_p0), _s));
                    _absmax1 = vmaxq_f32(_absmax1, vmulq_f32(vabsq_f32(_p1), _s));
                    _absmax2 = vmaxq_f32(_absmax2, vmulq_f32(vabsq_f32(_p2), _s));
                    _absmax3 = vmaxq_f32(_absmax3, vmulq_f32(vabsq_f32(_p3), _s));
                    p0a += A_hstep * 4;
                    psa += 4;
                }

                float32x2_t _max0 = vpmax_f32(vget_low_f32(_absmax0), vget_high_f32(_absmax0));
                float32x2_t _max1 = vpmax_f32(vget_low_f32(_absmax1), vget_high_f32(_absmax1));
                float32x2_t _max2 = vpmax_f32(vget_low_f32(_absmax2), vget_high_f32(_absmax2));
                float32x2_t _max3 = vpmax_f32(vget_low_f32(_absmax3), vget_high_f32(_absmax3));
                _absmax0 = vcombine_f32(vpmax_f32(_max0, _max1), vpmax_f32(_max2, _max3));

                vst1q_f32(pd, vmulq_n_f32(_absmax0, 1.f / 127.f));

                float32x4_t _zero = vdupq_n_f32(0.f);
#if __aarch64__
                float32x4_t _scale = vdivq_f32(vdupq_n_f32(127.f), _absmax0);
#else
                float32x4_t _scale = div_ps(vdupq_n_f32(127.f), _absmax0);
#endif
                _scale = vbslq_f32(vceqq_f32(_absmax0, _zero), _zero, _scale);

                int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    float32x4_t _s0 = vld1q_f32(ps);
                    float32x4_t _s1 = vld1q_f32(ps + 4);
                    float32x4_t _p00 = vmulq_n_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0)), _s0), vgetq_lane_f32(_scale, 0));
                    float32x4_t _p01 = vmulq_n_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 4)), _s1), vgetq_lane_f32(_scale, 0));
                    float32x4_t _p10 = vmulq_n_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 4)), _s0), vgetq_lane_f32(_scale, 1));
                    float32x4_t _p11 = vmulq_n_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 4 + 4)), _s1), vgetq_lane_f32(_scale, 1));
                    int8x8_t _r0 = float2int8(_p00, _p01);
                    int8x8_t _r1 = float2int8(_p10, _p11);
                    vst1q_s8(pp, vcombine_s8(_r0, _r1));
                    float32x4_t _p20 = vmulq_n_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 8)), _s0), vgetq_lane_f32(_scale, 2));
                    float32x4_t _p21 = vmulq_n_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 4 + 8)), _s1), vgetq_lane_f32(_scale, 2));
                    float32x4_t _p30 = vmulq_n_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 12)), _s0), vgetq_lane_f32(_scale, 3));
                    float32x4_t _p31 = vmulq_n_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 4 + 12)), _s1), vgetq_lane_f32(_scale, 3));
                    int8x8_t _r2 = float2int8(_p20, _p21);
                    int8x8_t _r3 = float2int8(_p30, _p31);
                    vst1q_s8(pp + 16, vcombine_s8(_r2, _r3));
                    pp += 32;
                    p0 += A_hstep * 8;
                    ps += 8;
                }
#endif // __ARM_FEATURE_MATMUL_INT8
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float32x4_t _s = vld1q_f32(ps);
                    float32x4_t _p0 = vmulq_n_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0)), _s), vgetq_lane_f32(_scale, 0));
                    float32x4_t _p1 = vmulq_n_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 4)), _s), vgetq_lane_f32(_scale, 1));
                    float32x4_t _p2 = vmulq_n_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 8)), _s), vgetq_lane_f32(_scale, 2));
                    float32x4_t _p3 = vmulq_n_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 12)), _s), vgetq_lane_f32(_scale, 3));
                    int8x8_t _r0 = float2int8(_p0, _p1);
                    int8x8_t _r1 = float2int8(_p2, _p3);
#if __ARM_FEATURE_DOTPROD
                    vst1q_s8(pp, vcombine_s8(_r0, _r1));
#else
                    int16x8_t _r01 = vreinterpretq_s16_s8(vcombine_s8(_r0, _r1));
                    int16x8x2_t _rr = vuzpq_s16(_r01, _r01);
                    vst1q_s8(pp, vreinterpretq_s8_s16(vcombine_s16(vget_low_s16(_rr.val[0]), vget_low_s16(_rr.val[1]))));
#endif
                    pp += 16;
                    p0 += A_hstep * 4;
                    ps += 4;
                }
                pd += 4;
            }
            if (elempack == 1)
            {
                float32x4_t _absmax = vdupq_n_f32(0.f);
                const unsigned short* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    float32x4_t _p = vcvt_f32_f16((float16x4_t)vld1_u16(p0a));
                    _absmax = vmaxq_f32(_absmax, vmulq_n_f32(vabsq_f32(_p), *psa++));
                    p0a += A_hstep;
                }

                vst1q_f32(pd, vmulq_n_f32(_absmax, 1.f / 127.f));
                float32x4_t _zero = vdupq_n_f32(0.f);
#if __aarch64__
                float32x4_t _scale = vdivq_f32(vdupq_n_f32(127.f), _absmax);
#else
                float32x4_t _scale = div_ps(vdupq_n_f32(127.f), _absmax);
#endif
                _scale = vbslq_f32(vceqq_f32(_absmax, _zero), _zero, _scale);

                int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    float32x4_t _p0 = vmulq_n_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0)), *ps++);
                    float32x4_t _p1 = vmulq_n_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep)), *ps++);
                    float32x4_t _p2 = vmulq_n_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 2)), *ps++);
                    float32x4_t _p3 = vmulq_n_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 3)), *ps++);
                    float32x4_t _p4 = vmulq_n_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 4)), *ps++);
                    float32x4_t _p5 = vmulq_n_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 5)), *ps++);
                    float32x4_t _p6 = vmulq_n_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 6)), *ps++);
                    float32x4_t _p7 = vmulq_n_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 7)), *ps++);
                    _p0 = vmulq_f32(_p0, _scale);
                    _p1 = vmulq_f32(_p1, _scale);
                    _p2 = vmulq_f32(_p2, _scale);
                    _p3 = vmulq_f32(_p3, _scale);
                    _p4 = vmulq_f32(_p4, _scale);
                    _p5 = vmulq_f32(_p5, _scale);
                    _p6 = vmulq_f32(_p6, _scale);
                    _p7 = vmulq_f32(_p7, _scale);
                    float32x4x2_t _p04 = vzipq_f32(_p0, _p4);
                    float32x4x2_t _p15 = vzipq_f32(_p1, _p5);
                    float32x4x2_t _p26 = vzipq_f32(_p2, _p6);
                    float32x4x2_t _p37 = vzipq_f32(_p3, _p7);
                    int8x8x4_t _r0123;
                    _r0123.val[0] = float2int8(_p04.val[0], _p04.val[1]);
                    _r0123.val[1] = float2int8(_p15.val[0], _p15.val[1]);
                    _r0123.val[2] = float2int8(_p26.val[0], _p26.val[1]);
                    _r0123.val[3] = float2int8(_p37.val[0], _p37.val[1]);
                    vst4_s8(pp, _r0123);
                    pp += 32;
                    p0 += A_hstep * 8;
                }
#endif // __ARM_FEATURE_MATMUL_INT8
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float32x4_t _p0 = vmulq_n_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0)), *ps++);
                    float32x4_t _p1 = vmulq_n_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep)), *ps++);
                    float32x4_t _p2 = vmulq_n_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 2)), *ps++);
                    float32x4_t _p3 = vmulq_n_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 3)), *ps++);
                    _p0 = vmulq_f32(_p0, _scale);
                    _p1 = vmulq_f32(_p1, _scale);
                    _p2 = vmulq_f32(_p2, _scale);
                    _p3 = vmulq_f32(_p3, _scale);
#if __ARM_FEATURE_DOTPROD
                    transpose4x4_ps(_p0, _p1, _p2, _p3);
                    int8x8_t _r01 = float2int8(_p0, _p1);
                    int8x8_t _r23 = float2int8(_p2, _p3);
                    vst1q_s8(pp, vcombine_s8(_r01, _r23));
#else
                    int8x8_t _r01 = float2int8(_p0, _p1);
                    int8x8_t _r23 = float2int8(_p2, _p3);
                    int8x8_t _r10 = vext_s8(_r01, _r01, 4);
                    int8x8_t _r32 = vext_s8(_r23, _r23, 4);
                    vst1_s8(pp, vzip_s8(_r01, _r10).val[0]);
                    vst1_s8(pp + 8, vzip_s8(_r23, _r32).val[0]);
#endif
                    pp += 16;
                    p0 += A_hstep * 4;
                }
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    float32x4_t _p0 = vmulq_n_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0)), *ps++);
                    float32x4_t _p1 = vmulq_n_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep)), *ps++);
                    int8x8_t _r01 = float2int8(vmulq_f32(_p0, _scale), vmulq_f32(_p1, _scale));
                    int8x8_t _r10 = vext_s8(_r01, _r01, 4);
                    vst1_s8(pp, vzip_s8(_r01, _r10).val[0]);
                    pp += 8;
                    p0 += A_hstep * 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    float32x4_t _p = vmulq_n_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0)), *ps++);
                    vst1_lane_s32((int*)pp, vreinterpret_s32_s8(float2int8(vmulq_f32(_p, _scale), vmulq_f32(_p, _scale))), 0);
                    pp += 4;
                    p0 += A_hstep;
                }
                pd += 4;
            }
        }
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
        const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * elempack;
        const float* ps = (const float*)input_scales + k;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);
#if __ARM_NEON
#if __aarch64__
            if (elempack == 8)
            {
                float absmax0 = 0.f;
                float absmax1 = 0.f;
                const unsigned short* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk += 8)
                {
                    float32x4_t _s0 = vld1q_f32(psa);
                    float32x4_t _s1 = vld1q_f32(psa + 4);
                    uint16x8_t _p = vld1q_u16(p0a);
                    uint16x8_t _q = vld1q_u16(p0a + 8);
                    absmax0 = std::max(absmax0, vmaxvq_f32(vmulq_f32(vabsq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_p))), _s0)));
                    absmax0 = std::max(absmax0, vmaxvq_f32(vmulq_f32(vabsq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_p))), _s1)));
                    absmax1 = std::max(absmax1, vmaxvq_f32(vmulq_f32(vabsq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_q))), _s0)));
                    absmax1 = std::max(absmax1, vmaxvq_f32(vmulq_f32(vabsq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_q))), _s1)));
                    p0a += A_hstep * 8;
                    psa += 8;
                }

                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                float32x4_t _scale0 = vdupq_n_f32(absmax0 == 0.f ? 0.f : 127.f / absmax0);
                float32x4_t _scale1 = vdupq_n_f32(absmax1 == 0.f ? 0.f : 127.f / absmax1);

                for (int kk = 0; kk < max_kk0; kk += 8)
                {
                    float32x4_t _s0 = vld1q_f32(ps);
                    float32x4_t _s1 = vld1q_f32(ps + 4);
                    uint16x8_t _p = vld1q_u16(p0);
                    uint16x8_t _q = vld1q_u16(p0 + 8);
                    float32x4_t _p0 = vmulq_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_p)), _s0), _scale0);
                    float32x4_t _p1 = vmulq_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_p)), _s1), _scale0);
                    float32x4_t _p2 = vmulq_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_q)), _s0), _scale1);
                    float32x4_t _p3 = vmulq_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_q)), _s1), _scale1);

#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                    int8x8_t _r0 = float2int8(_p0, _p1);
                    int8x8_t _r1 = float2int8(_p2, _p3);
#else
                    int8x8_t _r0 = float2int8(_p0, _p2);
                    int8x8_t _r1 = float2int8(_p1, _p3);
#endif
#else
                    int16x4_t _t0 = vreinterpret_s16_s8(float2int8(_p0, _p1));
                    int16x4_t _t1 = vreinterpret_s16_s8(float2int8(_p2, _p3));
                    int16x4x2_t _t01 = vzip_s16(_t0, _t1);
                    int8x8_t _r0 = vreinterpret_s8_s16(_t01.val[0]);
                    int8x8_t _r1 = vreinterpret_s8_s16(_t01.val[1]);
#endif

                    vst1q_s8(pp, vcombine_s8(_r0, _r1));

                    pp += 16;
                    p0 += A_hstep * 8;
                    ps += 8;
                }
                pd += 2;
            }
#endif // __aarch64__
            if (elempack == 4)
            {
                float32x4_t _absmax0 = vdupq_n_f32(0.f);
                float32x4_t _absmax1 = vdupq_n_f32(0.f);
                const unsigned short* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk += 4)
                {
                    float32x4_t _s = vld1q_f32(psa);
                    float32x4_t _p0 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a));
                    float32x4_t _p1 = vcvt_f32_f16((float16x4_t)vld1_u16(p0a + 4));
                    _absmax0 = vmaxq_f32(_absmax0, vmulq_f32(vabsq_f32(_p0), _s));
                    _absmax1 = vmaxq_f32(_absmax1, vmulq_f32(vabsq_f32(_p1), _s));
                    p0a += A_hstep * 4;
                    psa += 4;
                }

#if __aarch64__
                float absmax0 = vmaxvq_f32(_absmax0);
                float absmax1 = vmaxvq_f32(_absmax1);
#else
                float32x2_t _max0 = vmax_f32(vget_low_f32(_absmax0), vget_high_f32(_absmax0));
                float32x2_t _max1 = vmax_f32(vget_low_f32(_absmax1), vget_high_f32(_absmax1));
                _max0 = vpmax_f32(_max0, _max0);
                _max1 = vpmax_f32(_max1, _max1);
                float absmax0 = vget_lane_f32(_max0, 0);
                float absmax1 = vget_lane_f32(_max1, 0);
#endif
                pd[0] = absmax0 / 127.f;
                pd[1] = absmax1 / 127.f;
                float32x4_t _scale0 = vdupq_n_f32(absmax0 == 0.f ? 0.f : 127.f / absmax0);
                float32x4_t _scale1 = vdupq_n_f32(absmax1 == 0.f ? 0.f : 127.f / absmax1);

                int kk = 0;
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    float32x4_t _s0 = vld1q_f32(ps);
                    float32x4_t _s1 = vld1q_f32(ps + 4);
                    float32x4_t _p0 = vmulq_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0)), _s0), _scale0);
                    float32x4_t _p1 = vmulq_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 4)), _s0), _scale1);
                    float32x4_t _p2 = vmulq_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 4)), _s1), _scale0);
                    float32x4_t _p3 = vmulq_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 4 + 4)), _s1), _scale1);
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                    int8x8_t _r0 = float2int8(_p0, _p2);
                    int8x8_t _r1 = float2int8(_p1, _p3);
#else
                    int8x8_t _r0 = float2int8(_p0, _p1);
                    int8x8_t _r1 = float2int8(_p2, _p3);
#endif
#else
                    int16x4_t _t0 = vreinterpret_s16_s8(float2int8(_p0, _p2));
                    int16x4_t _t1 = vreinterpret_s16_s8(float2int8(_p1, _p3));
                    int16x4x2_t _t01 = vzip_s16(_t0, _t1);
                    int8x8_t _r0 = vreinterpret_s8_s16(_t01.val[0]);
                    int8x8_t _r1 = vreinterpret_s8_s16(_t01.val[1]);
#endif

                    vst1q_s8(pp, vcombine_s8(_r0, _r1));

                    pp += 16;
                    p0 += A_hstep * 8;
                    ps += 8;
                }
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float32x4_t _s = vld1q_f32(ps);
                    float32x4_t _p0 = vmulq_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0)), _s), _scale0);
                    float32x4_t _p1 = vmulq_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + 4)), _s), _scale1);

#if __ARM_FEATURE_DOTPROD
                    int8x8_t _r01 = float2int8(_p0, _p1);
#else
                    float32x4_t _t0 = vcombine_f32(vget_low_f32(_p0), vget_low_f32(_p1));
                    float32x4_t _t1 = vcombine_f32(vget_high_f32(_p0), vget_high_f32(_p1));
                    int8x8_t _r01 = float2int8(_t0, _t1);
#endif

                    vst1_s8(pp, _r01);

                    pp += 8;
                    p0 += A_hstep * 4;
                    ps += 4;
                }
                pd += 2;
            }
            if (elempack == 1)
            {
                float32x2_t _absmax = vdup_n_f32(0.f);
                const unsigned short* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    float32x2_t _p = vget_low_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0a)));
                    _absmax = vmax_f32(_absmax, vmul_n_f32(vabs_f32(_p), *psa++));
                    p0a += A_hstep;
                }

                vst1_f32(pd, vmul_n_f32(_absmax, 1.f / 127.f));
                float absmax0 = vget_lane_f32(_absmax, 0);
                float absmax1 = vget_lane_f32(_absmax, 1);
                float32x2_t _scale = vdup_n_f32(absmax0 == 0.f ? 0.f : 127.f / absmax0);
                _scale = vset_lane_f32(absmax1 == 0.f ? 0.f : 127.f / absmax1, _scale, 1);

                int kk = 0;
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk0; kk += 8)
                {
                    float32x2_t _p0 = vmul_n_f32(vget_low_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0))), *ps++);
                    float32x2_t _p1 = vmul_n_f32(vget_low_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep))), *ps++);
                    float32x2_t _p2 = vmul_n_f32(vget_low_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 2))), *ps++);
                    float32x2_t _p3 = vmul_n_f32(vget_low_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 3))), *ps++);
                    float32x2_t _p4 = vmul_n_f32(vget_low_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 4))), *ps++);
                    float32x2_t _p5 = vmul_n_f32(vget_low_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 5))), *ps++);
                    float32x2_t _p6 = vmul_n_f32(vget_low_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 6))), *ps++);
                    float32x2_t _p7 = vmul_n_f32(vget_low_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 7))), *ps++);
                    float32x4_t _scale_scale = vcombine_f32(_scale, _scale);
                    float32x4_t _p01 = vmulq_f32(vcombine_f32(_p0, _p1), _scale_scale);
                    float32x4_t _p23 = vmulq_f32(vcombine_f32(_p2, _p3), _scale_scale);
                    float32x4_t _p45 = vmulq_f32(vcombine_f32(_p4, _p5), _scale_scale);
                    float32x4_t _p67 = vmulq_f32(vcombine_f32(_p6, _p7), _scale_scale);
                    int8x8_t _r0 = float2int8(_p01, _p23);
                    int8x8_t _r1 = float2int8(_p45, _p67);
                    int8x8x2_t _r01 = vuzp_s8(_r0, _r1);
                    vst1q_s8(pp, vcombine_s8(_r01.val[0], _r01.val[1]));
                    pp += 16;
                    p0 += A_hstep * 8;
                }
#endif // __ARM_FEATURE_MATMUL_INT8
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk0; kk += 4)
                {
                    float32x2_t _p0 = vmul_n_f32(vget_low_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0))), *ps++);
                    float32x2_t _p1 = vmul_n_f32(vget_low_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep))), *ps++);
                    float32x2_t _p2 = vmul_n_f32(vget_low_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 2))), *ps++);
                    float32x2_t _p3 = vmul_n_f32(vget_low_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep * 3))), *ps++);
                    float32x4_t _scale_scale = vcombine_f32(_scale, _scale);
                    float32x4_t _p01 = vmulq_f32(vcombine_f32(_p0, _p1), _scale_scale);
                    float32x4_t _p23 = vmulq_f32(vcombine_f32(_p2, _p3), _scale_scale);
                    int8x8_t _r0 = float2int8(_p01, _p23);
#if __ARM_FEATURE_DOTPROD
                    int8x8x2_t _r01 = vuzp_s8(_r0, _r0);
                    vst1_s8(pp, vreinterpret_s8_s32(vzip_s32(vreinterpret_s32_s8(_r01.val[0]), vreinterpret_s32_s8(_r01.val[1])).val[0]));
#else
                    int8x8x2_t _r01 = vuzp_s8(_r0, _r0);
                    vst1_s8(pp, vreinterpret_s8_s16(vzip_s16(vreinterpret_s16_s8(_r01.val[0]), vreinterpret_s16_s8(_r01.val[1])).val[0]));
#endif
                    pp += 8;
                    p0 += A_hstep * 4;
                }
                for (; kk + 1 < max_kk0; kk += 2)
                {
                    float32x2_t _p0 = vmul_n_f32(vget_low_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0))), *ps++);
                    float32x2_t _p1 = vmul_n_f32(vget_low_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0 + A_hstep))), *ps++);
                    float32x4_t _scale_scale = vcombine_f32(_scale, _scale);
                    float32x4_t _p01 = vmulq_f32(vcombine_f32(_p0, _p1), _scale_scale);
                    int8x8_t _r0 = float2int8(_p01, _p01);
                    int8x8x2_t _r01 = vuzp_s8(_r0, _r0);
                    vst1_lane_s16((short*)pp, vreinterpret_s16_s8(_r01.val[0]), 0);
                    vst1_lane_s16((short*)pp + 1, vreinterpret_s16_s8(_r01.val[1]), 0);
                    pp += 4;
                    p0 += A_hstep * 2;
                }
                for (; kk < max_kk0; kk++)
                {
                    float32x2_t _p0 = vmul_n_f32(vget_low_f32(vcvt_f32_f16((float16x4_t)vld1_u16(p0))), *ps++);
                    float32x4_t _p01 = vmulq_f32(vcombine_f32(_p0, _p0), vcombine_f32(_scale, _scale));
                    vst1_lane_s16((short*)pp, vreinterpret_s16_s8(float2int8(_p01, _p01)), 0);
                    pp += 2;
                    p0 += A_hstep;
                }
#else
            float absmax0 = 0.f;
            float absmax1 = 0.f;
            const unsigned short* p0a = p0;
            const float* psa = ps;

            for (int kk = 0; kk < max_kk0; kk++)
            {
                const float s = *psa++;
                float v0 = float16_to_float32(p0a[0]);
                absmax0 = std::max(absmax0, fabsf(v0) * s);
                float v1 = float16_to_float32(p0a[1]);
                absmax1 = std::max(absmax1, fabsf(v1) * s);
                p0a += A_hstep;
            }

            pd[0] = absmax0 / 127.f;
            pd[1] = absmax1 / 127.f;
            const float scale0 = absmax0 == 0.f ? 0.f : 127.f / absmax0;
            const float scale1 = absmax1 == 0.f ? 0.f : 127.f / absmax1;

            for (int kk = 0; kk < max_kk0; kk++)
            {
                const float s = *ps++;
                float v0 = float16_to_float32(p0[0]) * s;
                float v1 = float16_to_float32(p0[1]) * s;
                *pp++ = float2int8(v0 * scale0);
                *pp++ = float2int8(v1 * scale1);
                p0 += A_hstep;
            }
#endif // __ARM_NEON
                pd += 2;
            }
        }
    }
    for (; ii < max_ii; ii++)
    {
        const unsigned short* p0 = (const unsigned short*)A + (size_t)k * A_hstep + (i + ii) * elempack;
        const float* ps = (const float*)input_scales + k;

        for (int g = 0; g < block_count; g++)
        {
            const int max_kk0 = std::min(max_kk - g * block_size, block_size);

#if __ARM_NEON
#if __aarch64__
            if (elempack == 8)
            {
                float absmax = 0.f;
                const unsigned short* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk += 8)
                {
                    uint16x8_t _p = vld1q_u16(p0a);
                    float32x4_t _p0 = vmulq_f32(vabsq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_p))), vld1q_f32(psa));
                    float32x4_t _p1 = vmulq_f32(vabsq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_p))), vld1q_f32(psa + 4));
                    absmax = std::max(absmax, vmaxvq_f32(_p0));
                    absmax = std::max(absmax, vmaxvq_f32(_p1));
                    p0a += A_hstep * 8;
                    psa += 8;
                }

                const float scale = absmax == 0.f ? 0.f : 127.f / absmax;
                *pd++ = absmax / 127.f;
                float32x4_t _scale = vdupq_n_f32(scale);

                for (int kk = 0; kk < max_kk0; kk += 8)
                {
                    uint16x8_t _p = vld1q_u16(p0);
                    float32x4_t _p0 = vmulq_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vget_low_u16(_p)), vld1q_f32(ps)), _scale);
                    float32x4_t _p1 = vmulq_f32(vmulq_f32(vcvt_f32_f16((float16x4_t)vget_high_u16(_p)), vld1q_f32(ps + 4)), _scale);
                    vst1_s8(pp, float2int8(_p0, _p1));
                    pp += 8;
                    p0 += A_hstep * 8;
                    ps += 8;
                }
            }
#endif // __aarch64__
            if (elempack == 4)
            {
                float32x4_t _absmax = vdupq_n_f32(0.f);
                const unsigned short* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk += 4)
                {
                    float32x4_t _p = vcvt_f32_f16((float16x4_t)vld1_u16(p0a));
                    _absmax = vmaxq_f32(_absmax, vmulq_f32(vabsq_f32(_p), vld1q_f32(psa)));
                    p0a += A_hstep * 4;
                    psa += 4;
                }
#if __aarch64__
                float absmax = vmaxvq_f32(_absmax);
#else
                float32x2_t _max = vmax_f32(vget_low_f32(_absmax), vget_high_f32(_absmax));
                _max = vpmax_f32(_max, _max);
                float absmax = vget_lane_f32(_max, 0);
#endif
                const float scale = absmax == 0.f ? 0.f : 127.f / absmax;
                *pd++ = absmax / 127.f;

                float32x4_t _scale = vdupq_n_f32(scale);
                for (int kk = 0; kk < max_kk0; kk += 4)
                {
                    float32x4_t _p = vcvt_f32_f16((float16x4_t)vld1_u16(p0));
                    _p = vmulq_f32(vmulq_f32(_p, vld1q_f32(ps)), _scale);
                    int8x8_t _r = float2int8(_p, _p);
                    vst1_lane_s32((int*)pp, vreinterpret_s32_s8(_r), 0);
                    pp += 4;
                    p0 += A_hstep * 4;
                    ps += 4;
                }
            }
#endif // __ARM_NEON
            if (elempack == 1)
            {
                float absmax = 0.f;
                const unsigned short* p0a = p0;
                const float* psa = ps;
                for (int kk = 0; kk < max_kk0; kk++)
                {
                    float v = float16_to_float32(*p0a);
                    absmax = std::max(absmax, fabsf(v) * *psa++);
                    p0a += A_hstep;
                }

                if (absmax == 0.f)
                {
                    *pd++ = 0.f;
                    for (int kk0 = 0; kk0 < max_kk0; kk0++)
                        *pp++ = 0;
                    p0 += (size_t)max_kk0 * A_hstep;
                    ps += max_kk0;
                    continue;
                }

                const float scale = 127.f / absmax;
                *pd++ = absmax / 127.f;

                for (int kk = 0; kk < max_kk0; kk++)
                {
                    float v = float16_to_float32(*p0) * *ps++;
                    *pp++ = float2int8(v * scale);
                    p0 += A_hstep;
                }
            }
        }
    }
}
