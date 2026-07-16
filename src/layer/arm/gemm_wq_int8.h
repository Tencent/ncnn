// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include <math.h>

#if NCNN_RUNTIME_CPU && NCNN_ARM86SVEI8MM && __aarch64__ && !__ARM_FEATURE_SVE_MATMUL_INT8
int pack_B_wq_int8_svei8mm(const Mat& B, const Mat& B_scales, Mat& BT, Mat& BT_descales, int N, int K, int block_size, int num_threads);
void gemm_transB_packed_tile_wq_int8_svei8mm(const Mat& AT_tile, const Mat& AT_descales_tile, const Mat& BT_tile, const Mat& BT_descales_tile, Mat& topT_tile, int max_ii, int max_jj, int K, int block_size);
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
int pack_B_wq_int8_i8mm(const Mat& B, const Mat& B_scales, Mat& BT, Mat& BT_descales, int N, int K, int block_size, int num_threads);
void quantize_A_tile_wq_int8_i8mm(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int K, int block_size, const float* input_scale_ptr);
void transpose_quantize_A_tile_wq_int8_i8mm(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int K, int block_size, const float* input_scale_ptr);
void gemm_transB_packed_tile_wq_int8_i8mm(const Mat& AT_tile, const Mat& AT_descales_tile, const Mat& BT_tile, const Mat& BT_descales_tile, Mat& topT_tile, int max_ii, int max_jj, int K, int block_size);
#endif

#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD && !__ARM_FEATURE_MATMUL_INT8
int pack_B_wq_int8_asimddp(const Mat& B, const Mat& B_scales, Mat& BT, Mat& BT_descales, int N, int K, int block_size, int num_threads);
void quantize_A_tile_wq_int8_asimddp(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int K, int block_size, const float* input_scale_ptr);
void transpose_quantize_A_tile_wq_int8_asimddp(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int K, int block_size, const float* input_scale_ptr);
void gemm_transB_packed_tile_wq_int8_asimddp(const Mat& AT_tile, const Mat& AT_descales_tile, const Mat& BT_tile, const Mat& BT_descales_tile, Mat& topT_tile, int max_ii, int max_jj, int K, int block_size);
#endif

static void quantize_A_tile_wq_int8(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int K, int block_size, const float* input_scale_ptr)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_i8mm())
    {
        quantize_A_tile_wq_int8_i8mm(A, AT_tile, AT_descales_tile, i, max_ii, K, block_size, input_scale_ptr);
        return;
    }
#endif
#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_asimddp())
    {
        quantize_A_tile_wq_int8_asimddp(A, AT_tile, AT_descales_tile, i, max_ii, K, block_size, input_scale_ptr);
        return;
    }
#endif

    signed char* outptr = AT_tile;
    const int out_hstep = AT_tile.w;
    float* descales = AT_descales_tile;
    const int descales_hstep = AT_descales_tile.w;
    const int block_count = (K + block_size - 1) / block_size;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

#if __ARM_NEON && __aarch64__
    if (max_ii >= 8)
    {
        signed char* pp = AT_tile;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);
            float absmax[8];
            float scales[8];

            for (int r = 0; r < 8; r++)
            {
                const float* ptrA = (const float*)A + (i + r) * A_hstep + k0;
                float32x4_t _absmax = vdupq_n_f32(0.f);
                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    float32x4_t _v = vld1q_f32(ptrA + kk);
                    if (input_scale_ptr)
                        _v = vmulq_f32(_v, vld1q_f32(input_scale_ptr + k0 + kk));
                    _absmax = vmaxq_f32(_absmax, vabsq_f32(_v));
                }
#if __aarch64__
                absmax[r] = vmaxvq_f32(_absmax);
#else
                float32x2_t _max2 = vmax_f32(vget_low_f32(_absmax), vget_high_f32(_absmax));
                _max2 = vpmax_f32(_max2, _max2);
                absmax[r] = vget_lane_f32(_max2, 0);
#endif
                for (; kk < max_kk; kk++)
                {
                    float v = ptrA[kk];
                    if (input_scale_ptr)
                        v *= input_scale_ptr[k0 + kk];
                    absmax[r] = std::max(absmax[r], fabsf(v));
                }

                descales[g * 8 + r] = absmax[r] / 127.f;
                volatile double scale_fp64 = absmax[r] == 0.f ? 0.0 : 127.0 / (double)absmax[r];
                scales[r] = (float)scale_fp64;
            }

            int kk = 0;
#if __ARM_FEATURE_MATMUL_INT8
            for (; kk + 7 < max_kk; kk += 8)
            {
                for (int r = 0; r < 8; r += 2)
                {
                    const float* ptrA0 = (const float*)A + (i + r) * A_hstep + k0 + kk;
                    const float* ptrA1 = ptrA0 + A_hstep;
                    float32x4_t _v00 = vld1q_f32(ptrA0);
                    float32x4_t _v01 = vld1q_f32(ptrA0 + 4);
                    float32x4_t _v10 = vld1q_f32(ptrA1);
                    float32x4_t _v11 = vld1q_f32(ptrA1 + 4);
                    if (input_scale_ptr)
                    {
                        const float32x4_t _s0 = vld1q_f32(input_scale_ptr + k0 + kk);
                        const float32x4_t _s1 = vld1q_f32(input_scale_ptr + k0 + kk + 4);
                        _v00 = vmulq_f32(_v00, _s0);
                        _v01 = vmulq_f32(_v01, _s1);
                        _v10 = vmulq_f32(_v10, _s0);
                        _v11 = vmulq_f32(_v11, _s1);
#if NCNN_GNU_INLINE_ASM
                        asm volatile(""
                                     : "+w"(_v00), "+w"(_v01), "+w"(_v10), "+w"(_v11));
#endif
                    }
                    const float32x4_t _scale0 = vdupq_n_f32(scales[r]);
                    const float32x4_t _scale1 = vdupq_n_f32(scales[r + 1]);
                    const int8x8_t _q0 = float2int8(vmulq_f32(_v00, _scale0), vmulq_f32(_v01, _scale0));
                    const int8x8_t _q1 = float2int8(vmulq_f32(_v10, _scale1), vmulq_f32(_v11, _scale1));
                    vst1q_s8(pp, vcombine_s8(_q0, _q1));
                    pp += 16;
                }
            }
#endif // __ARM_FEATURE_MATMUL_INT8
#if __ARM_FEATURE_DOTPROD
            for (; kk + 3 < max_kk; kk += 4)
            {
                for (int r = 0; r < 8; r++)
                {
                    const float* ptrA = (const float*)A + (i + r) * A_hstep + k0 + kk;
                    float32x4_t _v = vld1q_f32(ptrA);
                    if (input_scale_ptr)
                    {
                        _v = vmulq_f32(_v, vld1q_f32(input_scale_ptr + k0 + kk));
#if NCNN_GNU_INLINE_ASM
                        asm volatile(""
                                     : "+w"(_v));
#endif
                    }
                    const float32x4_t _scale = vdupq_n_f32(scales[r]);
                    const int8x8_t _q = float2int8(vmulq_f32(_v, _scale), vmulq_f32(_v, _scale));
                    vst1_lane_s32((int*)pp, vreinterpret_s32_s8(_q), 0);
                    pp += 4;
                }
            }
#endif // __ARM_FEATURE_DOTPROD
            for (; kk + 1 < max_kk; kk += 2)
            {
                for (int r = 0; r < 8; r++)
                {
                    const float* ptrA = (const float*)A + (i + r) * A_hstep + k0 + kk;
                    float v0 = ptrA[0];
                    float v1 = ptrA[1];
                    if (input_scale_ptr)
                    {
                        v0 *= input_scale_ptr[k0 + kk];
                        v1 *= input_scale_ptr[k0 + kk + 1];
#if NCNN_GNU_INLINE_ASM
                        asm volatile(""
                                     : "+w"(v0), "+w"(v1));
#endif
                    }
                    *pp++ = float2int8(v0 * scales[r]);
                    *pp++ = float2int8(v1 * scales[r]);
                }
            }
            if (kk < max_kk)
            {
                for (int r = 0; r < 8; r++)
                {
                    const float* ptrA = (const float*)A + (i + r) * A_hstep + k0 + kk;
                    float v = ptrA[0];
                    if (input_scale_ptr)
                    {
                        v *= input_scale_ptr[k0 + kk];
#if NCNN_GNU_INLINE_ASM
                        asm volatile(""
                                     : "+w"(v));
#endif
                    }
                    *pp++ = float2int8(v * scales[r]);
                }
            }
        }

        return;
    }
#endif // __ARM_NEON && __aarch64__

    int ii = 0;
#if __ARM_NEON
    for (; ii + 3 < max_ii; ii += 4)
    {
        signed char* pp = outptr + ii * out_hstep;
        float* descale_ptr = descales + ii * descales_hstep;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);
            float absmax[4];
            float scales[4];

            for (int r = 0; r < 4; r++)
            {
                const float* ptrA = (const float*)A + (i + ii + r) * A_hstep + k0;
                float32x4_t _absmax = vdupq_n_f32(0.f);
                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    float32x4_t _v = vld1q_f32(ptrA + kk);
                    if (input_scale_ptr)
                        _v = vmulq_f32(_v, vld1q_f32(input_scale_ptr + k0 + kk));
                    _absmax = vmaxq_f32(_absmax, vabsq_f32(_v));
                }
#if __aarch64__
                absmax[r] = vmaxvq_f32(_absmax);
#else
                float32x2_t _max2 = vmax_f32(vget_low_f32(_absmax), vget_high_f32(_absmax));
                _max2 = vpmax_f32(_max2, _max2);
                absmax[r] = vget_lane_f32(_max2, 0);
#endif
                for (; kk < max_kk; kk++)
                {
                    float v = ptrA[kk];
                    if (input_scale_ptr)
                        v *= input_scale_ptr[k0 + kk];
                    absmax[r] = std::max(absmax[r], fabsf(v));
                }
                descale_ptr[g * 4 + r] = absmax[r] / 127.f;
                volatile double scale_fp64 = absmax[r] == 0.f ? 0.0 : 127.0 / (double)absmax[r];
                scales[r] = (float)scale_fp64;
            }

            int kk = 0;
#if __ARM_FEATURE_MATMUL_INT8
            for (; kk + 7 < max_kk; kk += 8)
            {
                for (int r = 0; r < 4; r += 2)
                {
                    const float* ptrA0 = (const float*)A + (i + ii + r) * A_hstep + k0 + kk;
                    const float* ptrA1 = ptrA0 + A_hstep;
                    float32x4_t _v00 = vld1q_f32(ptrA0);
                    float32x4_t _v01 = vld1q_f32(ptrA0 + 4);
                    float32x4_t _v10 = vld1q_f32(ptrA1);
                    float32x4_t _v11 = vld1q_f32(ptrA1 + 4);
                    if (input_scale_ptr)
                    {
                        const float32x4_t _s0 = vld1q_f32(input_scale_ptr + k0 + kk);
                        const float32x4_t _s1 = vld1q_f32(input_scale_ptr + k0 + kk + 4);
                        _v00 = vmulq_f32(_v00, _s0);
                        _v01 = vmulq_f32(_v01, _s1);
                        _v10 = vmulq_f32(_v10, _s0);
                        _v11 = vmulq_f32(_v11, _s1);
#if NCNN_GNU_INLINE_ASM
                        asm volatile(""
                                     : "+w"(_v00), "+w"(_v01), "+w"(_v10), "+w"(_v11));
#endif
                    }
                    const float32x4_t _scale0 = vdupq_n_f32(scales[r]);
                    const float32x4_t _scale1 = vdupq_n_f32(scales[r + 1]);
                    const int8x8_t _q0 = float2int8(vmulq_f32(_v00, _scale0), vmulq_f32(_v01, _scale0));
                    const int8x8_t _q1 = float2int8(vmulq_f32(_v10, _scale1), vmulq_f32(_v11, _scale1));
                    vst1q_s8(pp, vcombine_s8(_q0, _q1));
                    pp += 16;
                }
            }
#endif // __ARM_FEATURE_MATMUL_INT8
#if __ARM_FEATURE_DOTPROD
            for (; kk + 3 < max_kk; kk += 4)
            {
                for (int r = 0; r < 4; r++)
                {
                    const float* ptrA = (const float*)A + (i + ii + r) * A_hstep + k0 + kk;
                    float32x4_t _v = vld1q_f32(ptrA);
                    if (input_scale_ptr)
                    {
                        _v = vmulq_f32(_v, vld1q_f32(input_scale_ptr + k0 + kk));
#if NCNN_GNU_INLINE_ASM
                        asm volatile(""
                                     : "+w"(_v));
#endif
                    }
                    const int8x8_t _q = float2int8(vmulq_n_f32(_v, scales[r]), vmulq_n_f32(_v, scales[r]));
                    vst1_lane_s32((int*)pp, vreinterpret_s32_s8(_q), 0);
                    pp += 4;
                }
            }
#endif // __ARM_FEATURE_DOTPROD
            for (; kk + 1 < max_kk; kk += 2)
            {
                for (int r = 0; r < 4; r++)
                {
                    const float* ptrA = (const float*)A + (i + ii + r) * A_hstep + k0 + kk;
                    float v0 = ptrA[0];
                    float v1 = ptrA[1];
                    if (input_scale_ptr)
                    {
                        v0 *= input_scale_ptr[k0 + kk];
                        v1 *= input_scale_ptr[k0 + kk + 1];
#if NCNN_GNU_INLINE_ASM
                        asm volatile(""
                                     : "+w"(v0), "+w"(v1));
#endif
                    }
                    *pp++ = float2int8(v0 * scales[r]);
                    *pp++ = float2int8(v1 * scales[r]);
                }
            }
            if (kk < max_kk)
            {
                for (int r = 0; r < 4; r++)
                {
                    const float* ptrA = (const float*)A + (i + ii + r) * A_hstep + k0 + kk;
                    float v = ptrA[0];
                    if (input_scale_ptr)
                    {
                        v *= input_scale_ptr[k0 + kk];
#if NCNN_GNU_INLINE_ASM
                        asm volatile(""
                                     : "+w"(v));
#endif
                    }
                    *pp++ = float2int8(v * scales[r]);
                }
            }
        }
    }
    for (; ii + 1 < max_ii; ii += 2)
    {
        signed char* pp = outptr + ii * out_hstep;
        float* descale_ptr = descales + ii * descales_hstep;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);
            float absmax[2];
            float scales[2];

            for (int r = 0; r < 2; r++)
            {
                const float* ptrA = (const float*)A + (i + ii + r) * A_hstep + k0;
                float32x4_t _absmax = vdupq_n_f32(0.f);
                int kk = 0;
                for (; kk + 3 < max_kk; kk += 4)
                {
                    float32x4_t _v = vld1q_f32(ptrA + kk);
                    if (input_scale_ptr)
                        _v = vmulq_f32(_v, vld1q_f32(input_scale_ptr + k0 + kk));
                    _absmax = vmaxq_f32(_absmax, vabsq_f32(_v));
                }
#if __aarch64__
                absmax[r] = vmaxvq_f32(_absmax);
#else
                float32x2_t _max2 = vmax_f32(vget_low_f32(_absmax), vget_high_f32(_absmax));
                _max2 = vpmax_f32(_max2, _max2);
                absmax[r] = vget_lane_f32(_max2, 0);
#endif
                for (; kk < max_kk; kk++)
                {
                    float v = ptrA[kk];
                    if (input_scale_ptr)
                        v *= input_scale_ptr[k0 + kk];
                    absmax[r] = std::max(absmax[r], fabsf(v));
                }
                descale_ptr[g * 2 + r] = absmax[r] / 127.f;
                volatile double scale_fp64 = absmax[r] == 0.f ? 0.0 : 127.0 / (double)absmax[r];
                scales[r] = (float)scale_fp64;
            }

            int kk = 0;
#if __ARM_FEATURE_MATMUL_INT8
            for (; kk + 7 < max_kk; kk += 8)
            {
                const float* ptrA0 = (const float*)A + (i + ii) * A_hstep + k0 + kk;
                const float* ptrA1 = ptrA0 + A_hstep;
                float32x4_t _v00 = vld1q_f32(ptrA0);
                float32x4_t _v01 = vld1q_f32(ptrA0 + 4);
                float32x4_t _v10 = vld1q_f32(ptrA1);
                float32x4_t _v11 = vld1q_f32(ptrA1 + 4);
                if (input_scale_ptr)
                {
                    const float32x4_t _s0 = vld1q_f32(input_scale_ptr + k0 + kk);
                    const float32x4_t _s1 = vld1q_f32(input_scale_ptr + k0 + kk + 4);
                    _v00 = vmulq_f32(_v00, _s0);
                    _v01 = vmulq_f32(_v01, _s1);
                    _v10 = vmulq_f32(_v10, _s0);
                    _v11 = vmulq_f32(_v11, _s1);
#if NCNN_GNU_INLINE_ASM
                    asm volatile(""
                                 : "+w"(_v00), "+w"(_v01), "+w"(_v10), "+w"(_v11));
#endif
                }
                const int8x8_t _q0 = float2int8(vmulq_n_f32(_v00, scales[0]), vmulq_n_f32(_v01, scales[0]));
                const int8x8_t _q1 = float2int8(vmulq_n_f32(_v10, scales[1]), vmulq_n_f32(_v11, scales[1]));
                vst1q_s8(pp, vcombine_s8(_q0, _q1));
                pp += 16;
            }
#endif // __ARM_FEATURE_MATMUL_INT8
#if __ARM_FEATURE_DOTPROD
            for (; kk + 3 < max_kk; kk += 4)
            {
                int32x2_t _q01 = vdup_n_s32(0);
                for (int r = 0; r < 2; r++)
                {
                    const float* ptrA = (const float*)A + (i + ii + r) * A_hstep + k0 + kk;
                    float32x4_t _v = vld1q_f32(ptrA);
                    if (input_scale_ptr)
                    {
                        _v = vmulq_f32(_v, vld1q_f32(input_scale_ptr + k0 + kk));
#if NCNN_GNU_INLINE_ASM
                        asm volatile(""
                                     : "+w"(_v));
#endif
                    }
                    const int8x8_t _q = float2int8(vmulq_n_f32(_v, scales[r]), vmulq_n_f32(_v, scales[r]));
                    _q01 = vset_lane_s32(vget_lane_s32(vreinterpret_s32_s8(_q), 0), _q01, r);
                }
                vst1_s8(pp, vreinterpret_s8_s32(_q01));
                pp += 8;
            }
#endif // __ARM_FEATURE_DOTPROD
            for (; kk + 1 < max_kk; kk += 2)
            {
                for (int r = 0; r < 2; r++)
                {
                    const float* ptrA = (const float*)A + (i + ii + r) * A_hstep + k0 + kk;
                    float v0 = ptrA[0];
                    float v1 = ptrA[1];
                    if (input_scale_ptr)
                    {
                        v0 *= input_scale_ptr[k0 + kk];
                        v1 *= input_scale_ptr[k0 + kk + 1];
                    }
                    *pp++ = float2int8(v0 * scales[r]);
                    *pp++ = float2int8(v1 * scales[r]);
                }
            }
            if (kk < max_kk)
            {
                for (int r = 0; r < 2; r++)
                {
                    const float* ptrA = (const float*)A + (i + ii + r) * A_hstep + k0 + kk;
                    float v = ptrA[0];
                    if (input_scale_ptr)
                        v *= input_scale_ptr[k0 + kk];
                    *pp++ = float2int8(v * scales[r]);
                }
            }
        }
    }
#elif __ARM_FEATURE_SIMD32 && NCNN_GNU_INLINE_ASM
    for (; ii + 1 < max_ii; ii += 2)
    {
        signed char* pp = outptr + ii * out_hstep;
        float* descale_ptr = descales + ii * descales_hstep;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);
            const float* ptrA0 = (const float*)A + (i + ii) * A_hstep + k0;
            const float* ptrA1 = ptrA0 + A_hstep;
            float absmax0 = 0.f;
            float absmax1 = 0.f;

            for (int kk = 0; kk < max_kk; kk++)
            {
                float v0 = ptrA0[kk];
                float v1 = ptrA1[kk];
                if (input_scale_ptr)
                {
                    v0 *= input_scale_ptr[k0 + kk];
                    v1 *= input_scale_ptr[k0 + kk];
                }
                absmax0 = std::max(absmax0, fabsf(v0));
                absmax1 = std::max(absmax1, fabsf(v1));
            }

            descale_ptr[g * 2] = absmax0 / 127.f;
            descale_ptr[g * 2 + 1] = absmax1 / 127.f;
            volatile double scale0_fp64 = absmax0 == 0.f ? 0.0 : 127.0 / (double)absmax0;
            volatile double scale1_fp64 = absmax1 == 0.f ? 0.0 : 127.0 / (double)absmax1;
            const float scale0 = (float)scale0_fp64;
            const float scale1 = (float)scale1_fp64;

            for (int kk = 0; kk < max_kk; kk++)
            {
                float v0 = ptrA0[kk];
                float v1 = ptrA1[kk];
                if (input_scale_ptr)
                {
                    v0 *= input_scale_ptr[k0 + kk];
                    v1 *= input_scale_ptr[k0 + kk];
                    asm volatile(""
                                 : "+w"(v0), "+w"(v1));
                }
                *pp++ = float2int8(v0 * scale0);
                *pp++ = float2int8(v1 * scale1);
            }
        }
    }
#endif // __ARM_NEON
    for (; ii < max_ii; ii++)
    {
        const float* ptrA = (const float*)A + (i + ii) * A_hstep;
        signed char* outptr0 = outptr + ii * out_hstep;
        float* descale_ptr = descales + ii * descales_hstep;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);

            float absmax = 0.f;
            int kk = 0;
#if __ARM_NEON
            float32x4_t _absmax = vdupq_n_f32(0.f);
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x4_t _v = vld1q_f32(ptrA + k0 + kk);
                if (input_scale_ptr)
                    _v = vmulq_f32(_v, vld1q_f32(input_scale_ptr + k0 + kk));
                _absmax = vmaxq_f32(_absmax, vabsq_f32(_v));
            }
#if __aarch64__
            absmax = vmaxvq_f32(_absmax);
#else
            float32x2_t _max2 = vmax_f32(vget_low_f32(_absmax), vget_high_f32(_absmax));
            _max2 = vpmax_f32(_max2, _max2);
            absmax = vget_lane_f32(_max2, 0);
#endif
#endif // __ARM_NEON
            for (; kk < max_kk; kk++)
            {
                const int k = k0 + kk;
                float v = ptrA[k];
                if (input_scale_ptr)
                    v *= input_scale_ptr[k];
                absmax = std::max(absmax, fabsf(v));
            }

            if (absmax == 0.f)
            {
                descale_ptr[g] = 0.f;
                for (int k = 0; k < max_kk; k++)
                    outptr0[k0 + k] = 0;
                continue;
            }

            volatile double scale_fp64 = 127.0 / (double)absmax;
            const float scale = (float)scale_fp64;
            descale_ptr[g] = absmax / 127.f;

            kk = 0;
#if __ARM_NEON
            const float32x4_t _scale = vdupq_n_f32(scale);
            for (; kk + 7 < max_kk; kk += 8)
            {
                float32x4_t _v0 = vld1q_f32(ptrA + k0 + kk);
                float32x4_t _v1 = vld1q_f32(ptrA + k0 + kk + 4);
                if (input_scale_ptr)
                {
                    _v0 = vmulq_f32(_v0, vld1q_f32(input_scale_ptr + k0 + kk));
                    _v1 = vmulq_f32(_v1, vld1q_f32(input_scale_ptr + k0 + kk + 4));
#if NCNN_GNU_INLINE_ASM
                    asm volatile(""
                                 : "+w"(_v0), "+w"(_v1));
#else
                    volatile float32x4_t _v0_ordered = _v0;
                    volatile float32x4_t _v1_ordered = _v1;
                    _v0 = _v0_ordered;
                    _v1 = _v1_ordered;
#endif
                }
                vst1_s8(outptr0 + k0 + kk, float2int8(vmulq_f32(_v0, _scale), vmulq_f32(_v1, _scale)));
            }
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x4_t _v = vld1q_f32(ptrA + k0 + kk);
                if (input_scale_ptr)
                {
                    _v = vmulq_f32(_v, vld1q_f32(input_scale_ptr + k0 + kk));
#if NCNN_GNU_INLINE_ASM
                    asm volatile(""
                                 : "+w"(_v));
#else
                    volatile float32x4_t _v_ordered = _v;
                    _v = _v_ordered;
#endif
                }
                const int8x8_t _q = float2int8(vmulq_f32(_v, _scale), vmulq_f32(_v, _scale));
                vst1_lane_s32((int*)(outptr0 + k0 + kk), vreinterpret_s32_s8(_q), 0);
            }
#endif // __ARM_NEON
            for (; kk < max_kk; kk++)
            {
                const int k = k0 + kk;
                float v = ptrA[k];
                if (input_scale_ptr)
                {
                    v *= input_scale_ptr[k];
#if NCNN_GNU_INLINE_ASM
                    asm volatile(""
                                 : "+w"(v));
#else
                    volatile float v_ordered = v;
                    v = v_ordered;
#endif
                }
                outptr0[k] = float2int8(v * scale);
            }
        }
    }
}

static void transpose_quantize_A_tile_wq_int8(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int K, int block_size, const float* input_scale_ptr)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_i8mm())
    {
        transpose_quantize_A_tile_wq_int8_i8mm(A, AT_tile, AT_descales_tile, i, max_ii, K, block_size, input_scale_ptr);
        return;
    }
#endif
#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_asimddp())
    {
        transpose_quantize_A_tile_wq_int8_asimddp(A, AT_tile, AT_descales_tile, i, max_ii, K, block_size, input_scale_ptr);
        return;
    }
#endif

    signed char* outptr = AT_tile;
    const int out_hstep = AT_tile.w;
    float* descales = AT_descales_tile;
    const int descales_hstep = AT_descales_tile.w;
    const int block_count = (K + block_size - 1) / block_size;
    const size_t A_hstep = A.dims == 3 ? A.cstep : (size_t)A.w;

    int ii = 0;
#if __ARM_NEON
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        const float* ptrA = (const float*)A + i + ii;
        signed char* pp = outptr + ii * out_hstep;
        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);
            float32x4_t _absmax0 = vdupq_n_f32(0.f);
            float32x4_t _absmax1 = vdupq_n_f32(0.f);
            for (int kk = 0; kk < max_kk; kk++)
            {
                const int k = k0 + kk;
                float32x4_t _v0 = vld1q_f32(ptrA + (size_t)k * A_hstep);
                float32x4_t _v1 = vld1q_f32(ptrA + (size_t)k * A_hstep + 4);
                if (input_scale_ptr)
                {
                    _v0 = vmulq_n_f32(_v0, input_scale_ptr[k]);
                    _v1 = vmulq_n_f32(_v1, input_scale_ptr[k]);
                }
                _absmax0 = vmaxq_f32(_absmax0, vabsq_f32(_v0));
                _absmax1 = vmaxq_f32(_absmax1, vabsq_f32(_v1));
            }

            float absmax[8];
            vst1q_f32(absmax, _absmax0);
            vst1q_f32(absmax + 4, _absmax1);
            descales[g * 8] = absmax[0] / 127.f;
            descales[g * 8 + 1] = absmax[1] / 127.f;
            descales[g * 8 + 2] = absmax[2] / 127.f;
            descales[g * 8 + 3] = absmax[3] / 127.f;
            descales[g * 8 + 4] = absmax[4] / 127.f;
            descales[g * 8 + 5] = absmax[5] / 127.f;
            descales[g * 8 + 6] = absmax[6] / 127.f;
            descales[g * 8 + 7] = absmax[7] / 127.f;

            float scales[8];
            for (int r = 0; r < 8; r++)
            {
                volatile double scale_fp64 = absmax[r] == 0.f ? 0.0 : 127.0 / (double)absmax[r];
                scales[r] = (float)scale_fp64;
            }
            const float32x4_t _scale0 = vld1q_f32(scales);
            const float32x4_t _scale1 = vld1q_f32(scales + 4);

            int kk = 0;
#if __ARM_FEATURE_MATMUL_INT8
            for (; kk + 7 < max_kk; kk += 8)
            {
                int8x8_t _q[8];
                for (int t = 0; t < 8; t++)
                {
                    const int k = k0 + kk + t;
                    float32x4_t _v0 = vld1q_f32(ptrA + (size_t)k * A_hstep);
                    float32x4_t _v1 = vld1q_f32(ptrA + (size_t)k * A_hstep + 4);
                    if (input_scale_ptr)
                    {
                        _v0 = vmulq_n_f32(_v0, input_scale_ptr[k]);
                        _v1 = vmulq_n_f32(_v1, input_scale_ptr[k]);
#if NCNN_GNU_INLINE_ASM
                        asm volatile(""
                                     : "+w"(_v0), "+w"(_v1));
#endif
                    }
                    _q[t] = float2int8(vmulq_f32(_v0, _scale0), vmulq_f32(_v1, _scale1));
                }
                int8x8x2_t _r04 = vzip_s8(_q[0], _q[4]);
                int8x8x2_t _r15 = vzip_s8(_q[1], _q[5]);
                int8x8x2_t _r26 = vzip_s8(_q[2], _q[6]);
                int8x8x2_t _r37 = vzip_s8(_q[3], _q[7]);
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
            }
#endif // __ARM_FEATURE_MATMUL_INT8
#if __ARM_FEATURE_DOTPROD
            for (; kk + 3 < max_kk; kk += 4)
            {
                int8x8x4_t _q;
                for (int t = 0; t < 4; t++)
                {
                    const int k = k0 + kk + t;
                    float32x4_t _v0 = vld1q_f32(ptrA + (size_t)k * A_hstep);
                    float32x4_t _v1 = vld1q_f32(ptrA + (size_t)k * A_hstep + 4);
                    if (input_scale_ptr)
                    {
                        _v0 = vmulq_n_f32(_v0, input_scale_ptr[k]);
                        _v1 = vmulq_n_f32(_v1, input_scale_ptr[k]);
#if NCNN_GNU_INLINE_ASM
                        asm volatile(""
                                     : "+w"(_v0), "+w"(_v1));
#endif
                    }
                    _q.val[t] = float2int8(vmulq_f32(_v0, _scale0), vmulq_f32(_v1, _scale1));
                }
                vst4_s8(pp, _q);
                pp += 32;
            }
#endif // __ARM_FEATURE_DOTPROD
            for (; kk + 1 < max_kk; kk += 2)
            {
                int8x8x2_t _q;
                for (int t = 0; t < 2; t++)
                {
                    const int k = k0 + kk + t;
                    float32x4_t _v0 = vld1q_f32(ptrA + (size_t)k * A_hstep);
                    float32x4_t _v1 = vld1q_f32(ptrA + (size_t)k * A_hstep + 4);
                    if (input_scale_ptr)
                    {
                        _v0 = vmulq_n_f32(_v0, input_scale_ptr[k]);
                        _v1 = vmulq_n_f32(_v1, input_scale_ptr[k]);
#if NCNN_GNU_INLINE_ASM
                        asm volatile(""
                                     : "+w"(_v0), "+w"(_v1));
#endif
                    }
                    _q.val[t] = float2int8(vmulq_f32(_v0, _scale0), vmulq_f32(_v1, _scale1));
                }
                vst2_s8(pp, _q);
                pp += 16;
            }
            if (kk < max_kk)
            {
                const int k = k0 + kk;
                float32x4_t _v0 = vld1q_f32(ptrA + (size_t)k * A_hstep);
                float32x4_t _v1 = vld1q_f32(ptrA + (size_t)k * A_hstep + 4);
                if (input_scale_ptr)
                {
                    _v0 = vmulq_n_f32(_v0, input_scale_ptr[k]);
                    _v1 = vmulq_n_f32(_v1, input_scale_ptr[k]);
#if NCNN_GNU_INLINE_ASM
                    asm volatile(""
                                 : "+w"(_v0), "+w"(_v1));
#endif
                }
                vst1_s8(pp, float2int8(vmulq_f32(_v0, _scale0), vmulq_f32(_v1, _scale1)));
                pp += 8;
            }
        }
    }
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const float* ptrA = (const float*)A + i + ii;
        signed char* pp = outptr + ii * out_hstep;
        float* descale_ptr = descales + ii * descales_hstep;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);

            float32x4_t _absmax = vdupq_n_f32(0.f);
            for (int kk = 0; kk < max_kk; kk++)
            {
                const int k = k0 + kk;
                float32x4_t _v = vld1q_f32(ptrA + (size_t)k * A_hstep);
                if (input_scale_ptr)
                    _v = vmulq_f32(_v, vdupq_n_f32(input_scale_ptr[k]));
                _absmax = vmaxq_f32(_absmax, vabsq_f32(_v));
            }

            float absmax[4];
            vst1q_f32(absmax, _absmax);
            vst1q_f32(descale_ptr + g * 4, vmulq_n_f32(_absmax, 1.f / 127.f));

            volatile double scale0_fp64 = absmax[0] == 0.f ? 0.0 : 127.0 / (double)absmax[0];
            volatile double scale1_fp64 = absmax[1] == 0.f ? 0.0 : 127.0 / (double)absmax[1];
            volatile double scale2_fp64 = absmax[2] == 0.f ? 0.0 : 127.0 / (double)absmax[2];
            volatile double scale3_fp64 = absmax[3] == 0.f ? 0.0 : 127.0 / (double)absmax[3];
            const float scales[4] = {
                (float)scale0_fp64,
                (float)scale1_fp64,
                (float)scale2_fp64,
                (float)scale3_fp64
            };
            const float32x4_t _scale = vld1q_f32(scales);

            int kk = 0;
#if __ARM_FEATURE_MATMUL_INT8
            for (; kk + 7 < max_kk; kk += 8)
            {
                int8x8_t _q[8];
                for (int t = 0; t < 8; t++)
                {
                    const int k = k0 + kk + t;
                    float32x4_t _v = vld1q_f32(ptrA + (size_t)k * A_hstep);
                    if (input_scale_ptr)
                    {
                        _v = vmulq_n_f32(_v, input_scale_ptr[k]);
#if NCNN_GNU_INLINE_ASM
                        asm volatile(""
                                     : "+w"(_v));
#else
                        volatile float32x4_t _v_ordered = _v;
                        _v = _v_ordered;
#endif
                    }
                    _q[t] = float2int8(vmulq_f32(_v, _scale), vmulq_f32(_v, _scale));
                }
                int8x8x2_t _r04 = vzip_s8(_q[0], _q[4]);
                int8x8x2_t _r15 = vzip_s8(_q[1], _q[5]);
                int8x8x2_t _r26 = vzip_s8(_q[2], _q[6]);
                int8x8x2_t _r37 = vzip_s8(_q[3], _q[7]);
                int8x8x4_t _r0123;
                _r0123.val[0] = _r04.val[0];
                _r0123.val[1] = _r15.val[0];
                _r0123.val[2] = _r26.val[0];
                _r0123.val[3] = _r37.val[0];
                vst4_s8(pp, _r0123);
                pp += 32;
            }
#endif // __ARM_FEATURE_MATMUL_INT8
#if __ARM_FEATURE_DOTPROD
            for (; kk + 3 < max_kk; kk += 4)
            {
                int8x8x4_t _q;
                for (int t = 0; t < 4; t++)
                {
                    const int k = k0 + kk + t;
                    float32x4_t _v = vld1q_f32(ptrA + (size_t)k * A_hstep);
                    if (input_scale_ptr)
                    {
                        _v = vmulq_n_f32(_v, input_scale_ptr[k]);
#if NCNN_GNU_INLINE_ASM
                        asm volatile(""
                                     : "+w"(_v));
#else
                        volatile float32x4_t _v_ordered = _v;
                        _v = _v_ordered;
#endif
                    }
                    _q.val[t] = float2int8(vmulq_f32(_v, _scale), vmulq_f32(_v, _scale));
                }
                vst4_lane_s8(pp, _q, 0);
                vst4_lane_s8(pp + 4, _q, 1);
                vst4_lane_s8(pp + 8, _q, 2);
                vst4_lane_s8(pp + 12, _q, 3);
                pp += 16;
            }
#endif // __ARM_FEATURE_DOTPROD
            for (; kk + 1 < max_kk; kk += 2)
            {
                int8x8x2_t _q;
                for (int t = 0; t < 2; t++)
                {
                    const int k = k0 + kk + t;
                    float32x4_t _v = vld1q_f32(ptrA + (size_t)k * A_hstep);
                    if (input_scale_ptr)
                    {
                        _v = vmulq_n_f32(_v, input_scale_ptr[k]);
#if NCNN_GNU_INLINE_ASM
                        asm volatile(""
                                     : "+w"(_v));
#else
                        volatile float32x4_t _v_ordered = _v;
                        _v = _v_ordered;
#endif
                    }
                    _q.val[t] = float2int8(vmulq_f32(_v, _scale), vmulq_f32(_v, _scale));
                }
                vst2_lane_s8(pp, _q, 0);
                vst2_lane_s8(pp + 2, _q, 1);
                vst2_lane_s8(pp + 4, _q, 2);
                vst2_lane_s8(pp + 6, _q, 3);
                pp += 8;
            }
            if (kk < max_kk)
            {
                const int k = k0 + kk;
                float32x4_t _v = vld1q_f32(ptrA + (size_t)k * A_hstep);
                if (input_scale_ptr)
                {
                    _v = vmulq_n_f32(_v, input_scale_ptr[k]);
#if NCNN_GNU_INLINE_ASM
                    asm volatile(""
                                 : "+w"(_v));
#else
                    volatile float32x4_t _v_ordered = _v;
                    _v = _v_ordered;
#endif
                }
                vst1_lane_s32((int*)pp, vreinterpret_s32_s8(float2int8(vmulq_f32(_v, _scale), vmulq_f32(_v, _scale))), 0);
                pp += 4;
            }
        }
    }
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* ptrA = (const float*)A + i + ii;
        signed char* pp = outptr + ii * out_hstep;
        float* descale_ptr = descales + ii * descales_hstep;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);
            float32x2_t _absmax = vdup_n_f32(0.f);
            for (int kk = 0; kk < max_kk; kk++)
            {
                const int k = k0 + kk;
                float32x2_t _v = vld1_f32(ptrA + (size_t)k * A_hstep);
                if (input_scale_ptr)
                    _v = vmul_n_f32(_v, input_scale_ptr[k]);
                _absmax = vmax_f32(_absmax, vabs_f32(_v));
            }

            vst1_f32(descale_ptr + g * 2, vmul_n_f32(_absmax, 1.f / 127.f));
            float absmax[2];
            vst1_f32(absmax, _absmax);
            float scales[2];
            for (int r = 0; r < 2; r++)
            {
                volatile double scale_fp64 = absmax[r] == 0.f ? 0.0 : 127.0 / (double)absmax[r];
                scales[r] = (float)scale_fp64;
            }
            const float32x2_t _scale = vld1_f32(scales);

            int kk = 0;
#if __ARM_FEATURE_MATMUL_INT8
            for (; kk + 7 < max_kk; kk += 8)
            {
                int8x8x4_t _q0;
                int8x8x4_t _q1;
                for (int t = 0; t < 4; t++)
                {
                    const int k0t = k0 + kk + t;
                    const int k1t = k0 + kk + 4 + t;
                    float32x2_t _v0 = vld1_f32(ptrA + (size_t)k0t * A_hstep);
                    float32x2_t _v1 = vld1_f32(ptrA + (size_t)k1t * A_hstep);
                    if (input_scale_ptr)
                    {
                        _v0 = vmul_n_f32(_v0, input_scale_ptr[k0t]);
                        _v1 = vmul_n_f32(_v1, input_scale_ptr[k1t]);
                    }
                    const float32x4_t _s = vcombine_f32(_scale, _scale);
                    const float32x4_t _v0q = vmulq_f32(vcombine_f32(_v0, _v0), _s);
                    const float32x4_t _v1q = vmulq_f32(vcombine_f32(_v1, _v1), _s);
                    _q0.val[t] = float2int8(_v0q, _v0q);
                    _q1.val[t] = float2int8(_v1q, _v1q);
                }
                vst4_lane_s8(pp, _q0, 0);
                vst4_lane_s8(pp + 4, _q1, 0);
                vst4_lane_s8(pp + 8, _q0, 1);
                vst4_lane_s8(pp + 12, _q1, 1);
                pp += 16;
            }
#endif // __ARM_FEATURE_MATMUL_INT8
#if __ARM_FEATURE_DOTPROD
            for (; kk + 3 < max_kk; kk += 4)
            {
                int8x8x4_t _q;
                for (int t = 0; t < 4; t++)
                {
                    const int k = k0 + kk + t;
                    float32x2_t _v = vld1_f32(ptrA + (size_t)k * A_hstep);
                    if (input_scale_ptr)
                        _v = vmul_n_f32(_v, input_scale_ptr[k]);
                    const float32x4_t _vq = vmulq_f32(vcombine_f32(_v, _v), vcombine_f32(_scale, _scale));
                    _q.val[t] = float2int8(_vq, _vq);
                }
                vst4_lane_s8(pp, _q, 0);
                vst4_lane_s8(pp + 4, _q, 1);
                pp += 8;
            }
#endif // __ARM_FEATURE_DOTPROD
            for (; kk + 1 < max_kk; kk += 2)
            {
                int8x8x2_t _q;
                for (int t = 0; t < 2; t++)
                {
                    const int k = k0 + kk + t;
                    float32x2_t _v = vld1_f32(ptrA + (size_t)k * A_hstep);
                    if (input_scale_ptr)
                        _v = vmul_n_f32(_v, input_scale_ptr[k]);
                    const float32x4_t _vq = vmulq_f32(vcombine_f32(_v, _v), vcombine_f32(_scale, _scale));
                    _q.val[t] = float2int8(_vq, _vq);
                }
                vst2_lane_s8(pp, _q, 0);
                vst2_lane_s8(pp + 2, _q, 1);
                pp += 4;
            }
            if (kk < max_kk)
            {
                const int k = k0 + kk;
                float32x2_t _v = vld1_f32(ptrA + (size_t)k * A_hstep);
                if (input_scale_ptr)
                    _v = vmul_n_f32(_v, input_scale_ptr[k]);
                const float32x4_t _vq = vmulq_f32(vcombine_f32(_v, _v), vcombine_f32(_scale, _scale));
                vst1_lane_s16((short*)pp, vreinterpret_s16_s8(float2int8(_vq, _vq)), 0);
                pp += 2;
            }
        }
    }
#elif __ARM_FEATURE_SIMD32 && NCNN_GNU_INLINE_ASM
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* ptrA = (const float*)A + i + ii;
        signed char* pp = outptr + ii * out_hstep;
        float* descale_ptr = descales + ii * descales_hstep;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);
            const float* ptrAk = ptrA + (size_t)k0 * A_hstep;
            float absmax0 = 0.f;
            float absmax1 = 0.f;

            for (int kk = 0; kk < max_kk; kk++)
            {
                float v0 = ptrAk[0];
                float v1 = ptrAk[1];
                if (input_scale_ptr)
                {
                    v0 *= input_scale_ptr[k0 + kk];
                    v1 *= input_scale_ptr[k0 + kk];
                }
                absmax0 = std::max(absmax0, fabsf(v0));
                absmax1 = std::max(absmax1, fabsf(v1));
                ptrAk += A_hstep;
            }

            descale_ptr[g * 2] = absmax0 / 127.f;
            descale_ptr[g * 2 + 1] = absmax1 / 127.f;
            volatile double scale0_fp64 = absmax0 == 0.f ? 0.0 : 127.0 / (double)absmax0;
            volatile double scale1_fp64 = absmax1 == 0.f ? 0.0 : 127.0 / (double)absmax1;
            const float scale0 = (float)scale0_fp64;
            const float scale1 = (float)scale1_fp64;

            ptrAk = ptrA + (size_t)k0 * A_hstep;
            for (int kk = 0; kk < max_kk; kk++)
            {
                float v0 = ptrAk[0];
                float v1 = ptrAk[1];
                if (input_scale_ptr)
                {
                    v0 *= input_scale_ptr[k0 + kk];
                    v1 *= input_scale_ptr[k0 + kk];
                    asm volatile(""
                                 : "+w"(v0), "+w"(v1));
                }
                *pp++ = float2int8(v0 * scale0);
                *pp++ = float2int8(v1 * scale1);
                ptrAk += A_hstep;
            }
        }
    }
#endif // __ARM_NEON
    for (; ii < max_ii; ii++)
    {
        const float* ptrA = (const float*)A + i + ii;
        signed char* outptr0 = outptr + ii * out_hstep;
        float* descale_ptr = descales + ii * descales_hstep;

        for (int g = 0; g < block_count; g++)
        {
            const int k0 = g * block_size;
            const int max_kk = std::min(K - k0, block_size);

            float absmax = 0.f;
            for (int kk = 0; kk < max_kk; kk++)
            {
                const int k = k0 + kk;
                float v = ptrA[(size_t)k * A_hstep];
                if (input_scale_ptr)
                    v *= input_scale_ptr[k];
                absmax = std::max(absmax, fabsf(v));
            }

            if (absmax == 0.f)
            {
                descale_ptr[g] = 0.f;
                for (int k = 0; k < max_kk; k++)
                    outptr0[k0 + k] = 0;
                continue;
            }

            volatile double scale_fp64 = 127.0 / (double)absmax;
            const float scale = (float)scale_fp64;
            descale_ptr[g] = absmax / 127.f;

            for (int kk = 0; kk < max_kk; kk++)
            {
                const int k = k0 + kk;
                float v = ptrA[(size_t)k * A_hstep];
                if (input_scale_ptr)
                {
                    v *= input_scale_ptr[k];
#if NCNN_GNU_INLINE_ASM
                    asm volatile(""
                                 : "+w"(v));
#else
                    volatile float v_ordered = v;
                    v = v_ordered;
#endif
                }
                outptr0[k] = float2int8(v * scale);
            }
        }
    }
}

// Persistent B uses the baseline gemm_int8 K2/K1 byte order inside each
// (output-column panel, quantization group). Every panel is exact and the
// address of the panel starting at output column j is always j * K.
// Two consecutive nr4 panels form the logical nr8 panel on aarch64.
static int pack_B_wq_int8(const Mat& B, const Mat& B_scales, Mat& BT, Mat& BT_descales, int N, int K, int block_size, int num_threads)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM86SVEI8MM && __aarch64__ && !__ARM_FEATURE_SVE_MATMUL_INT8
    if (ncnn::cpu_support_arm_svei8mm())
        return pack_B_wq_int8_svei8mm(B, B_scales, BT, BT_descales, N, K, block_size, num_threads);
#endif
#if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_i8mm())
        return pack_B_wq_int8_i8mm(B, B_scales, BT, BT_descales, N, K, block_size, num_threads);
#endif
#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_asimddp())
        return pack_B_wq_int8_asimddp(B, B_scales, BT, BT_descales, N, K, block_size, num_threads);
#endif

    const int block_count = (K + block_size - 1) / block_size;
    Mat BT_packed(N * K, (size_t)1u);
    Mat BT_packed_descales(N * block_count, (size_t)4u);
    if (BT_packed.empty() || BT_packed_descales.empty())
        return -100;
    BT_packed.cstep = (size_t)N * K;
    BT_packed_descales.cstep = (size_t)N * block_count;

    int panel_start = 0;
#if __ARM_NEON
    const int nn4 = (N - panel_start) / 4;
    const int panel_start4 = panel_start;
    panel_start += nn4 * 4;
#endif
    const int nn2 = (N - panel_start) / 2;
    const int panel_start2 = panel_start;
    panel_start += nn2 * 2;
    const int nn1 = N - panel_start;
    const int panel_start1 = panel_start;

    #pragma omp parallel num_threads(num_threads)
    {
#if __ARM_NEON
        #pragma omp for
        for (int p = 0; p < nn4; p++)
        {
            const int j = panel_start4 + p * 4;
            signed char* pp = (signed char*)BT_packed + j * K;
            float* pd = (float*)BT_packed_descales + j * block_count;

            for (int g = 0; g < block_count; g++)
            {
                const int k0 = g * block_size;
                const int max_kk = std::min(K - k0, block_size);
                const signed char* p0 = B.row<const signed char>(j) + k0;
                const signed char* p1 = B.row<const signed char>(j + 1) + k0;
                const signed char* p2 = B.row<const signed char>(j + 2) + k0;
                const signed char* p3 = B.row<const signed char>(j + 3) + k0;
                int kk = 0;
                for (; kk + 15 < max_kk; kk += 16)
                {
                    const int8x16_t _p0 = vld1q_s8(p0);
                    const int8x16_t _p1 = vld1q_s8(p1);
                    const int8x16_t _p2 = vld1q_s8(p2);
                    const int8x16_t _p3 = vld1q_s8(p3);
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                    int64x2x4_t _r0123;
                    _r0123.val[0] = vreinterpretq_s64_s8(_p0);
                    _r0123.val[1] = vreinterpretq_s64_s8(_p1);
                    _r0123.val[2] = vreinterpretq_s64_s8(_p2);
                    _r0123.val[3] = vreinterpretq_s64_s8(_p3);
                    vst4q_s64((int64_t*)pp, _r0123);
#else  // __ARM_FEATURE_MATMUL_INT8
                    int32x4x4_t _r0123;
                    _r0123.val[0] = vreinterpretq_s32_s8(_p0);
                    _r0123.val[1] = vreinterpretq_s32_s8(_p1);
                    _r0123.val[2] = vreinterpretq_s32_s8(_p2);
                    _r0123.val[3] = vreinterpretq_s32_s8(_p3);
                    vst4q_s32((int*)pp, _r0123);
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
                    int16x8x4_t _r0123;
                    _r0123.val[0] = vreinterpretq_s16_s8(_p0);
                    _r0123.val[1] = vreinterpretq_s16_s8(_p1);
                    _r0123.val[2] = vreinterpretq_s16_s8(_p2);
                    _r0123.val[3] = vreinterpretq_s16_s8(_p3);
                    vst4q_s16((short*)pp, _r0123);
#endif // __ARM_FEATURE_DOTPROD
                    pp += 64;
                    p0 += 16;
                    p1 += 16;
                    p2 += 16;
                    p3 += 16;
                }
                for (; kk + 7 < max_kk; kk += 8)
                {
                    const int8x8_t _p0 = vld1_s8(p0);
                    const int8x8_t _p1 = vld1_s8(p1);
                    const int8x8_t _p2 = vld1_s8(p2);
                    const int8x8_t _p3 = vld1_s8(p3);
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                    vst1q_s8(pp, vcombine_s8(_p0, _p1));
                    vst1q_s8(pp + 16, vcombine_s8(_p2, _p3));
#else  // __ARM_FEATURE_MATMUL_INT8
                    int32x2x4_t _r0123;
                    _r0123.val[0] = vreinterpret_s32_s8(_p0);
                    _r0123.val[1] = vreinterpret_s32_s8(_p1);
                    _r0123.val[2] = vreinterpret_s32_s8(_p2);
                    _r0123.val[3] = vreinterpret_s32_s8(_p3);
                    vst4_s32((int*)pp, _r0123);
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
                    int16x4x4_t _r0123;
                    _r0123.val[0] = vreinterpret_s16_s8(_p0);
                    _r0123.val[1] = vreinterpret_s16_s8(_p1);
                    _r0123.val[2] = vreinterpret_s16_s8(_p2);
                    _r0123.val[3] = vreinterpret_s16_s8(_p3);
                    vst4_s16((short*)pp, _r0123);
#endif // __ARM_FEATURE_DOTPROD
                    pp += 32;
                    p0 += 8;
                    p1 += 8;
                    p2 += 8;
                    p3 += 8;
                }
#if __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk; kk += 4)
                {
                    pp[0] = p0[0];
                    pp[1] = p0[1];
                    pp[2] = p0[2];
                    pp[3] = p0[3];
                    pp[4] = p1[0];
                    pp[5] = p1[1];
                    pp[6] = p1[2];
                    pp[7] = p1[3];
                    pp[8] = p2[0];
                    pp[9] = p2[1];
                    pp[10] = p2[2];
                    pp[11] = p2[3];
                    pp[12] = p3[0];
                    pp[13] = p3[1];
                    pp[14] = p3[2];
                    pp[15] = p3[3];
                    pp += 16;
                    p0 += 4;
                    p1 += 4;
                    p2 += 4;
                    p3 += 4;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk; kk += 2)
                {
                    pp[0] = p0[0];
                    pp[1] = p0[1];
                    pp[2] = p1[0];
                    pp[3] = p1[1];
                    pp[4] = p2[0];
                    pp[5] = p2[1];
                    pp[6] = p3[0];
                    pp[7] = p3[1];
                    pp += 8;
                    p0 += 2;
                    p1 += 2;
                    p2 += 2;
                    p3 += 2;
                }
                if (kk < max_kk)
                {
                    pp[0] = p0[0];
                    pp[1] = p1[0];
                    pp[2] = p2[0];
                    pp[3] = p3[0];
                    pp += 4;
                }

                for (int jj = 0; jj < 4; jj++)
                    pd[g * 4 + jj] = 1.f / B_scales.row(j + jj)[g];
            }
        }
#endif
        #pragma omp for
        for (int p = 0; p < nn2; p++)
        {
            const int j = panel_start2 + p * 2;
            signed char* pp = (signed char*)BT_packed + j * K;
            float* pd = (float*)BT_packed_descales + j * block_count;

            for (int g = 0; g < block_count; g++)
            {
                const int k0 = g * block_size;
                const int max_kk = std::min(K - k0, block_size);
                const signed char* p0 = B.row<const signed char>(j) + k0;
                const signed char* p1 = B.row<const signed char>(j + 1) + k0;
                int kk = 0;
#if __ARM_NEON
                for (; kk + 15 < max_kk; kk += 16)
                {
                    const int8x16_t _p0 = vld1q_s8(p0);
                    const int8x16_t _p1 = vld1q_s8(p1);
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                    int64x2x2_t _r01;
                    _r01.val[0] = vreinterpretq_s64_s8(_p0);
                    _r01.val[1] = vreinterpretq_s64_s8(_p1);
                    vst2q_s64((int64_t*)pp, _r01);
#else  // __ARM_FEATURE_MATMUL_INT8
                    int32x4x2_t _r01;
                    _r01.val[0] = vreinterpretq_s32_s8(_p0);
                    _r01.val[1] = vreinterpretq_s32_s8(_p1);
                    vst2q_s32((int*)pp, _r01);
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
                    int16x8x2_t _r01;
                    _r01.val[0] = vreinterpretq_s16_s8(_p0);
                    _r01.val[1] = vreinterpretq_s16_s8(_p1);
                    vst2q_s16((short*)pp, _r01);
#endif // __ARM_FEATURE_DOTPROD
                    pp += 32;
                    p0 += 16;
                    p1 += 16;
                }
                for (; kk + 7 < max_kk; kk += 8)
                {
                    const int8x8_t _p0 = vld1_s8(p0);
                    const int8x8_t _p1 = vld1_s8(p1);
#if __ARM_FEATURE_DOTPROD
#if __ARM_FEATURE_MATMUL_INT8
                    vst1q_s8(pp, vcombine_s8(_p0, _p1));
#else  // __ARM_FEATURE_MATMUL_INT8
                    int32x2x2_t _r01;
                    _r01.val[0] = vreinterpret_s32_s8(_p0);
                    _r01.val[1] = vreinterpret_s32_s8(_p1);
                    vst2_s32((int*)pp, _r01);
#endif // __ARM_FEATURE_MATMUL_INT8
#else  // __ARM_FEATURE_DOTPROD
                    int16x4x2_t _r01;
                    _r01.val[0] = vreinterpret_s16_s8(_p0);
                    _r01.val[1] = vreinterpret_s16_s8(_p1);
                    vst2_s16((short*)pp, _r01);
#endif // __ARM_FEATURE_DOTPROD
                    pp += 16;
                    p0 += 8;
                    p1 += 8;
                }
#endif // __ARM_NEON
#if __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk; kk += 4)
                {
                    pp[0] = p0[0];
                    pp[1] = p0[1];
                    pp[2] = p0[2];
                    pp[3] = p0[3];
                    pp[4] = p1[0];
                    pp[5] = p1[1];
                    pp[6] = p1[2];
                    pp[7] = p1[3];
                    pp += 8;
                    p0 += 4;
                    p1 += 4;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk; kk += 2)
                {
#if !__ARM_NEON && __ARM_FEATURE_SIMD32 && NCNN_GNU_INLINE_ASM
                    pp[0] = p0[0];
                    pp[1] = p1[0];
                    pp[2] = p0[1];
                    pp[3] = p1[1];
                    pp += 4;
#else
                    pp[0] = p0[0];
                    pp[1] = p0[1];
                    pp[2] = p1[0];
                    pp[3] = p1[1];
                    pp += 4;
#endif
                    p0 += 2;
                    p1 += 2;
                }
                if (kk < max_kk)
                {
                    pp[0] = p0[0];
                    pp[1] = p1[0];
                    pp += 2;
                }

                for (int jj = 0; jj < 2; jj++)
                    pd[g * 2 + jj] = 1.f / B_scales.row(j + jj)[g];
            }
        }
        #pragma omp for
        for (int p = 0; p < nn1; p++)
        {
            const int j = panel_start1 + p * 1;
            signed char* pp = (signed char*)BT_packed + j * K;
            float* pd = (float*)BT_packed_descales + j * block_count;

            for (int g = 0; g < block_count; g++)
            {
                const int k0 = g * block_size;
                const int max_kk = std::min(K - k0, block_size);
                const signed char* p0 = B.row<const signed char>(j) + k0;
                int kk = 0;
#if __ARM_NEON
                for (; kk + 15 < max_kk; kk += 16)
                {
                    vst1q_s8(pp, vld1q_s8(p0));
                    pp += 16;
                    p0 += 16;
                }
                for (; kk + 7 < max_kk; kk += 8)
                {
                    vst1_s8(pp, vld1_s8(p0));
                    pp += 8;
                    p0 += 8;
                }
#endif // __ARM_NEON
#if __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk; kk += 4)
                {
                    pp[0] = p0[0];
                    pp[1] = p0[1];
                    pp[2] = p0[2];
                    pp[3] = p0[3];
                    pp += 4;
                    p0 += 4;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk; kk += 2)
                {
                    pp[0] = p0[0];
                    pp[1] = p0[1];
                    pp += 2;
                    p0 += 2;
                }
                if (kk < max_kk)
                    *pp++ = p0[0];

                for (int jj = 0; jj < 1; jj++)
                    pd[g * 1 + jj] = 1.f / B_scales.row(j + jj)[g];
            }
        }
    }

    BT = BT_packed;
    BT_descales = BT_packed_descales;
    return 0;
}

static void gemm_transB_packed_tile_wq_int8(const Mat& AT_tile, const Mat& AT_descales_tile, const Mat& BT_tile, const Mat& BT_descales_tile, Mat& topT_tile, int max_ii, int max_jj, int K, int block_size)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM86SVEI8MM && __aarch64__ && !__ARM_FEATURE_SVE_MATMUL_INT8
    if (ncnn::cpu_support_arm_svei8mm())
    {
        gemm_transB_packed_tile_wq_int8_svei8mm(AT_tile, AT_descales_tile, BT_tile, BT_descales_tile, topT_tile, max_ii, max_jj, K, block_size);
        return;
    }
#endif
#if NCNN_RUNTIME_CPU && NCNN_ARM84I8MM && __aarch64__ && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_i8mm())
    {
        gemm_transB_packed_tile_wq_int8_i8mm(AT_tile, AT_descales_tile, BT_tile, BT_descales_tile, topT_tile, max_ii, max_jj, K, block_size);
        return;
    }
#endif
#if NCNN_RUNTIME_CPU && NCNN_ARM82DOT && __aarch64__ && !__ARM_FEATURE_DOTPROD && !__ARM_FEATURE_MATMUL_INT8
    if (ncnn::cpu_support_arm_asimddp())
    {
        gemm_transB_packed_tile_wq_int8_asimddp(AT_tile, AT_descales_tile, BT_tile, BT_descales_tile, topT_tile, max_ii, max_jj, K, block_size);
        return;
    }
#endif

    const signed char* pAT = AT_tile;
    const int A_hstep = AT_tile.w;
    const float* pAT_descales = AT_descales_tile;
    const int A_descales_hstep = AT_descales_tile.w;
    const signed char* pBT = BT_tile;
    const float* pBT_descales = BT_descales_tile;

    float* outptr = topT_tile;

    int ii = 0;
#if __ARM_NEON
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        const signed char* pA_block = pAT;
        const float* pA_descales_block = pAT_descales;
        const signed char* pB = pBT;
        const float* pB_descales = pBT_descales;

        int jj = 0;
        for (; jj + 3 < max_jj; jj += 4)
        {
            const signed char* pA = pA_block;
            const float* pA_descales = pA_descales_block;
            float32x4_t _fsum0 = vdupq_n_f32(0.f);
            float32x4_t _fsum1 = vdupq_n_f32(0.f);
            float32x4_t _fsum2 = vdupq_n_f32(0.f);
            float32x4_t _fsum3 = vdupq_n_f32(0.f);
            float32x4_t _fsum4 = vdupq_n_f32(0.f);
            float32x4_t _fsum5 = vdupq_n_f32(0.f);
            float32x4_t _fsum6 = vdupq_n_f32(0.f);
            float32x4_t _fsum7 = vdupq_n_f32(0.f);

            for (int k = 0; k < K; k += block_size)
            {
                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);
                int32x4_t _sum2 = vdupq_n_s32(0);
                int32x4_t _sum3 = vdupq_n_s32(0);
                int32x4_t _sum4 = vdupq_n_s32(0);
                int32x4_t _sum5 = vdupq_n_s32(0);
                int32x4_t _sum6 = vdupq_n_s32(0);
                int32x4_t _sum7 = vdupq_n_s32(0);
                const int max_kk = std::min(K - k, block_size);
                int kk = 0;
#if __ARM_FEATURE_MATMUL_INT8
                int32x4_t _msum0 = vdupq_n_s32(0);
                int32x4_t _msum1 = vdupq_n_s32(0);
                int32x4_t _msum2 = vdupq_n_s32(0);
                int32x4_t _msum3 = vdupq_n_s32(0);
                int32x4_t _msum4 = vdupq_n_s32(0);
                int32x4_t _msum5 = vdupq_n_s32(0);
                int32x4_t _msum6 = vdupq_n_s32(0);
                int32x4_t _msum7 = vdupq_n_s32(0);
                for (; kk + 7 < max_kk; kk += 8)
                {
                    const int8x16_t _b0 = vld1q_s8(pB);
                    const int8x16_t _b1 = vld1q_s8(pB + 16);
                    const int8x16_t _a01 = vld1q_s8(pA);
                    const int8x16_t _a23 = vld1q_s8(pA + 16);
                    const int8x16_t _a45 = vld1q_s8(pA + 32);
                    const int8x16_t _a67 = vld1q_s8(pA + 48);
                    _msum0 = vmmlaq_s32(_msum0, _a01, _b0);
                    _msum1 = vmmlaq_s32(_msum1, _a23, _b0);
                    _msum2 = vmmlaq_s32(_msum2, _a01, _b1);
                    _msum3 = vmmlaq_s32(_msum3, _a23, _b1);
                    _msum4 = vmmlaq_s32(_msum4, _a45, _b0);
                    _msum5 = vmmlaq_s32(_msum5, _a67, _b0);
                    _msum6 = vmmlaq_s32(_msum6, _a45, _b1);
                    _msum7 = vmmlaq_s32(_msum7, _a67, _b1);
                    pA += 64;
                    pB += 32;
                }
                _sum0 = vcombine_s32(vget_low_s32(_msum0), vget_low_s32(_msum2));
                _sum1 = vcombine_s32(vget_high_s32(_msum0), vget_high_s32(_msum2));
                _sum2 = vcombine_s32(vget_low_s32(_msum1), vget_low_s32(_msum3));
                _sum3 = vcombine_s32(vget_high_s32(_msum1), vget_high_s32(_msum3));
                _sum4 = vcombine_s32(vget_low_s32(_msum4), vget_low_s32(_msum6));
                _sum5 = vcombine_s32(vget_high_s32(_msum4), vget_high_s32(_msum6));
                _sum6 = vcombine_s32(vget_low_s32(_msum5), vget_low_s32(_msum7));
                _sum7 = vcombine_s32(vget_high_s32(_msum5), vget_high_s32(_msum7));
#endif // __ARM_FEATURE_MATMUL_INT8
#if __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk; kk += 4)
                {
                    const int8x16_t _b0 = vld1q_s8(pB);
                    const int8x16_t _a0 = vld1q_s8(pA);
                    const int8x16_t _a1 = vld1q_s8(pA + 16);
                    _sum0 = vdotq_laneq_s32(_sum0, _b0, _a0, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _b0, _a0, 1);
                    _sum2 = vdotq_laneq_s32(_sum2, _b0, _a0, 2);
                    _sum3 = vdotq_laneq_s32(_sum3, _b0, _a0, 3);
                    _sum4 = vdotq_laneq_s32(_sum4, _b0, _a1, 0);
                    _sum5 = vdotq_laneq_s32(_sum5, _b0, _a1, 1);
                    _sum6 = vdotq_laneq_s32(_sum6, _b0, _a1, 2);
                    _sum7 = vdotq_laneq_s32(_sum7, _b0, _a1, 3);
                    pA += 32;
                    pB += 16;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk; kk += 2)
                {
                    const int8x8_t _b0 = vld1_s8(pB);
                    const int16x8_t _a = vreinterpretq_s16_s8(vld1q_s8(pA));
                    const int16x4_t _a0 = vget_low_s16(_a);
                    const int16x4_t _a1 = vget_high_s16(_a);
                    _sum0 = vaddq_s32(_sum0, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a0, 0)))));
                    _sum1 = vaddq_s32(_sum1, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a0, 1)))));
                    _sum2 = vaddq_s32(_sum2, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a0, 2)))));
                    _sum3 = vaddq_s32(_sum3, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a0, 3)))));
                    _sum4 = vaddq_s32(_sum4, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a1, 0)))));
                    _sum5 = vaddq_s32(_sum5, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a1, 1)))));
                    _sum6 = vaddq_s32(_sum6, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a1, 2)))));
                    _sum7 = vaddq_s32(_sum7, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a1, 3)))));
                    pA += 16;
                    pB += 8;
                }
                if (kk < max_kk)
                {
                    const int8x8_t _b0 = vreinterpret_s8_s32(vld1_dup_s32((const int*)pB));
                    const int8x8_t _a = vld1_s8(pA);
                    const int16x8_t _p0 = vmull_s8(_b0, vdup_lane_s8(_a, 0));
                    const int16x8_t _p1 = vmull_s8(_b0, vdup_lane_s8(_a, 1));
                    _sum0 = vaddq_s32(_sum0, vmovl_s16(vget_low_s16(_p0)));
                    _sum1 = vaddq_s32(_sum1, vmovl_s16(vget_low_s16(_p1)));
                    const int16x8_t _p2 = vmull_s8(_b0, vdup_lane_s8(_a, 2));
                    const int16x8_t _p3 = vmull_s8(_b0, vdup_lane_s8(_a, 3));
                    _sum2 = vaddq_s32(_sum2, vmovl_s16(vget_low_s16(_p2)));
                    _sum3 = vaddq_s32(_sum3, vmovl_s16(vget_low_s16(_p3)));
                    const int16x8_t _p4 = vmull_s8(_b0, vdup_lane_s8(_a, 4));
                    const int16x8_t _p5 = vmull_s8(_b0, vdup_lane_s8(_a, 5));
                    _sum4 = vaddq_s32(_sum4, vmovl_s16(vget_low_s16(_p4)));
                    _sum5 = vaddq_s32(_sum5, vmovl_s16(vget_low_s16(_p5)));
                    const int16x8_t _p6 = vmull_s8(_b0, vdup_lane_s8(_a, 6));
                    const int16x8_t _p7 = vmull_s8(_b0, vdup_lane_s8(_a, 7));
                    _sum6 = vaddq_s32(_sum6, vmovl_s16(vget_low_s16(_p6)));
                    _sum7 = vaddq_s32(_sum7, vmovl_s16(vget_low_s16(_p7)));
                    pA += 8;
                    pB += 4;
                }

                const float32x4_t _bd0 = vld1q_f32(pB_descales);
                const float32x4_t _ad0 = vld1q_f32(pA_descales);
                const float32x4_t _ad1 = vld1q_f32(pA_descales + 4);
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_laneq_f32(_bd0, _ad0, 0));
                _fsum1 = vmlaq_f32(_fsum1, vcvtq_f32_s32(_sum1), vmulq_laneq_f32(_bd0, _ad0, 1));
                _fsum2 = vmlaq_f32(_fsum2, vcvtq_f32_s32(_sum2), vmulq_laneq_f32(_bd0, _ad0, 2));
                _fsum3 = vmlaq_f32(_fsum3, vcvtq_f32_s32(_sum3), vmulq_laneq_f32(_bd0, _ad0, 3));
                _fsum4 = vmlaq_f32(_fsum4, vcvtq_f32_s32(_sum4), vmulq_laneq_f32(_bd0, _ad1, 0));
                _fsum5 = vmlaq_f32(_fsum5, vcvtq_f32_s32(_sum5), vmulq_laneq_f32(_bd0, _ad1, 1));
                _fsum6 = vmlaq_f32(_fsum6, vcvtq_f32_s32(_sum6), vmulq_laneq_f32(_bd0, _ad1, 2));
                _fsum7 = vmlaq_f32(_fsum7, vcvtq_f32_s32(_sum7), vmulq_laneq_f32(_bd0, _ad1, 3));
                pA_descales += 8;
                pB_descales += 4;
            }

            vst1q_f32(outptr, _fsum0);
            vst1q_f32(outptr + 4, _fsum1);
            vst1q_f32(outptr + 8, _fsum2);
            vst1q_f32(outptr + 12, _fsum3);
            vst1q_f32(outptr + 16, _fsum4);
            vst1q_f32(outptr + 20, _fsum5);
            vst1q_f32(outptr + 24, _fsum6);
            vst1q_f32(outptr + 28, _fsum7);
            outptr += 32;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const signed char* pA = pA_block;
            const float* pA_descales = pA_descales_block;
            float32x4_t _fsum0 = vdupq_n_f32(0.f);
            float32x4_t _fsum1 = vdupq_n_f32(0.f);
            float32x4_t _fsum2 = vdupq_n_f32(0.f);
            float32x4_t _fsum3 = vdupq_n_f32(0.f);
            float32x4_t _fsum4 = vdupq_n_f32(0.f);
            float32x4_t _fsum5 = vdupq_n_f32(0.f);
            float32x4_t _fsum6 = vdupq_n_f32(0.f);
            float32x4_t _fsum7 = vdupq_n_f32(0.f);

            for (int k = 0; k < K; k += block_size)
            {
                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);
                int32x4_t _sum2 = vdupq_n_s32(0);
                int32x4_t _sum3 = vdupq_n_s32(0);
                int32x4_t _sum4 = vdupq_n_s32(0);
                int32x4_t _sum5 = vdupq_n_s32(0);
                int32x4_t _sum6 = vdupq_n_s32(0);
                int32x4_t _sum7 = vdupq_n_s32(0);
                const int max_kk = std::min(K - k, block_size);
                int kk = 0;
#if __ARM_FEATURE_MATMUL_INT8
                int32x4_t _msum0 = vdupq_n_s32(0);
                int32x4_t _msum1 = vdupq_n_s32(0);
                int32x4_t _msum2 = vdupq_n_s32(0);
                int32x4_t _msum3 = vdupq_n_s32(0);
                for (; kk + 7 < max_kk; kk += 8)
                {
                    const int8x16_t _b0 = vld1q_s8(pB);
                    const int8x16_t _a01 = vld1q_s8(pA);
                    const int8x16_t _a23 = vld1q_s8(pA + 16);
                    const int8x16_t _a45 = vld1q_s8(pA + 32);
                    const int8x16_t _a67 = vld1q_s8(pA + 48);
                    _msum0 = vmmlaq_s32(_msum0, _a01, _b0);
                    _msum1 = vmmlaq_s32(_msum1, _a23, _b0);
                    _msum2 = vmmlaq_s32(_msum2, _a45, _b0);
                    _msum3 = vmmlaq_s32(_msum3, _a67, _b0);
                    pA += 64;
                    pB += 16;
                }
                _sum0 = vcombine_s32(vget_low_s32(_msum0), vdup_n_s32(0));
                _sum1 = vcombine_s32(vget_high_s32(_msum0), vdup_n_s32(0));
                _sum2 = vcombine_s32(vget_low_s32(_msum1), vdup_n_s32(0));
                _sum3 = vcombine_s32(vget_high_s32(_msum1), vdup_n_s32(0));
                _sum4 = vcombine_s32(vget_low_s32(_msum2), vdup_n_s32(0));
                _sum5 = vcombine_s32(vget_high_s32(_msum2), vdup_n_s32(0));
                _sum6 = vcombine_s32(vget_low_s32(_msum3), vdup_n_s32(0));
                _sum7 = vcombine_s32(vget_high_s32(_msum3), vdup_n_s32(0));
#endif // __ARM_FEATURE_MATMUL_INT8
#if __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk; kk += 4)
                {
                    const int8x16_t _b0 = vcombine_s8(vld1_s8(pB), vdup_n_s8(0));
                    const int8x16_t _a0 = vld1q_s8(pA);
                    const int8x16_t _a1 = vld1q_s8(pA + 16);
                    _sum0 = vdotq_laneq_s32(_sum0, _b0, _a0, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _b0, _a0, 1);
                    _sum2 = vdotq_laneq_s32(_sum2, _b0, _a0, 2);
                    _sum3 = vdotq_laneq_s32(_sum3, _b0, _a0, 3);
                    _sum4 = vdotq_laneq_s32(_sum4, _b0, _a1, 0);
                    _sum5 = vdotq_laneq_s32(_sum5, _b0, _a1, 1);
                    _sum6 = vdotq_laneq_s32(_sum6, _b0, _a1, 2);
                    _sum7 = vdotq_laneq_s32(_sum7, _b0, _a1, 3);
                    pA += 32;
                    pB += 8;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk; kk += 2)
                {
                    const int8x8_t _b0 = vreinterpret_s8_s32(vld1_dup_s32((const int*)pB));
                    const int16x8_t _a = vreinterpretq_s16_s8(vld1q_s8(pA));
                    const int16x4_t _a0 = vget_low_s16(_a);
                    const int16x4_t _a1 = vget_high_s16(_a);
                    _sum0 = vaddq_s32(_sum0, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a0, 0)))));
                    _sum1 = vaddq_s32(_sum1, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a0, 1)))));
                    _sum2 = vaddq_s32(_sum2, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a0, 2)))));
                    _sum3 = vaddq_s32(_sum3, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a0, 3)))));
                    _sum4 = vaddq_s32(_sum4, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a1, 0)))));
                    _sum5 = vaddq_s32(_sum5, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a1, 1)))));
                    _sum6 = vaddq_s32(_sum6, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a1, 2)))));
                    _sum7 = vaddq_s32(_sum7, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a1, 3)))));
                    pA += 16;
                    pB += 4;
                }
                if (kk < max_kk)
                {
                    const int8x8_t _b0 = vreinterpret_s8_s16(vld1_dup_s16((const short*)pB));
                    const int8x8_t _a = vld1_s8(pA);
                    const int16x8_t _p0 = vmull_s8(_b0, vdup_lane_s8(_a, 0));
                    const int16x8_t _p1 = vmull_s8(_b0, vdup_lane_s8(_a, 1));
                    const int16x8_t _p2 = vmull_s8(_b0, vdup_lane_s8(_a, 2));
                    const int16x8_t _p3 = vmull_s8(_b0, vdup_lane_s8(_a, 3));
                    const int16x8_t _p4 = vmull_s8(_b0, vdup_lane_s8(_a, 4));
                    const int16x8_t _p5 = vmull_s8(_b0, vdup_lane_s8(_a, 5));
                    const int16x8_t _p6 = vmull_s8(_b0, vdup_lane_s8(_a, 6));
                    const int16x8_t _p7 = vmull_s8(_b0, vdup_lane_s8(_a, 7));
                    _sum0 = vaddq_s32(_sum0, vmovl_s16(vget_low_s16(_p0)));
                    _sum1 = vaddq_s32(_sum1, vmovl_s16(vget_low_s16(_p1)));
                    _sum2 = vaddq_s32(_sum2, vmovl_s16(vget_low_s16(_p2)));
                    _sum3 = vaddq_s32(_sum3, vmovl_s16(vget_low_s16(_p3)));
                    _sum4 = vaddq_s32(_sum4, vmovl_s16(vget_low_s16(_p4)));
                    _sum5 = vaddq_s32(_sum5, vmovl_s16(vget_low_s16(_p5)));
                    _sum6 = vaddq_s32(_sum6, vmovl_s16(vget_low_s16(_p6)));
                    _sum7 = vaddq_s32(_sum7, vmovl_s16(vget_low_s16(_p7)));
                    pA += 8;
                    pB += 2;
                }

                const float32x4_t _bd = vcombine_f32(vld1_f32(pB_descales), vdup_n_f32(0.f));
                const float32x4_t _ad0 = vld1q_f32(pA_descales);
                const float32x4_t _ad1 = vld1q_f32(pA_descales + 4);
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_laneq_f32(_bd, _ad0, 0));
                _fsum1 = vmlaq_f32(_fsum1, vcvtq_f32_s32(_sum1), vmulq_laneq_f32(_bd, _ad0, 1));
                _fsum2 = vmlaq_f32(_fsum2, vcvtq_f32_s32(_sum2), vmulq_laneq_f32(_bd, _ad0, 2));
                _fsum3 = vmlaq_f32(_fsum3, vcvtq_f32_s32(_sum3), vmulq_laneq_f32(_bd, _ad0, 3));
                _fsum4 = vmlaq_f32(_fsum4, vcvtq_f32_s32(_sum4), vmulq_laneq_f32(_bd, _ad1, 0));
                _fsum5 = vmlaq_f32(_fsum5, vcvtq_f32_s32(_sum5), vmulq_laneq_f32(_bd, _ad1, 1));
                _fsum6 = vmlaq_f32(_fsum6, vcvtq_f32_s32(_sum6), vmulq_laneq_f32(_bd, _ad1, 2));
                _fsum7 = vmlaq_f32(_fsum7, vcvtq_f32_s32(_sum7), vmulq_laneq_f32(_bd, _ad1, 3));
                pA_descales += 8;
                pB_descales += 2;
            }

            vst1_f32(outptr, vget_low_f32(_fsum0));
            vst1_f32(outptr + 2, vget_low_f32(_fsum1));
            vst1_f32(outptr + 4, vget_low_f32(_fsum2));
            vst1_f32(outptr + 6, vget_low_f32(_fsum3));
            vst1_f32(outptr + 8, vget_low_f32(_fsum4));
            vst1_f32(outptr + 10, vget_low_f32(_fsum5));
            vst1_f32(outptr + 12, vget_low_f32(_fsum6));
            vst1_f32(outptr + 14, vget_low_f32(_fsum7));
            outptr += 16;
        }
        for (; jj < max_jj; jj++)
        {
            const signed char* pA = pA_block;
            const float* pA_descales = pA_descales_block;
            float32x4_t _fsum0 = vdupq_n_f32(0.f);
            float32x4_t _fsum1 = vdupq_n_f32(0.f);
            float32x4_t _fsum2 = vdupq_n_f32(0.f);
            float32x4_t _fsum3 = vdupq_n_f32(0.f);
            float32x4_t _fsum4 = vdupq_n_f32(0.f);
            float32x4_t _fsum5 = vdupq_n_f32(0.f);
            float32x4_t _fsum6 = vdupq_n_f32(0.f);
            float32x4_t _fsum7 = vdupq_n_f32(0.f);

            for (int k = 0; k < K; k += block_size)
            {
                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);
                int32x4_t _sum2 = vdupq_n_s32(0);
                int32x4_t _sum3 = vdupq_n_s32(0);
                int32x4_t _sum4 = vdupq_n_s32(0);
                int32x4_t _sum5 = vdupq_n_s32(0);
                int32x4_t _sum6 = vdupq_n_s32(0);
                int32x4_t _sum7 = vdupq_n_s32(0);
                const int max_kk = std::min(K - k, block_size);
                int kk = 0;
#if __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk; kk += 8)
                {
                    const int8x16_t _a01 = vld1q_s8(pA);
                    const int8x16_t _a23 = vld1q_s8(pA + 16);
                    const int8x16_t _a45 = vld1q_s8(pA + 32);
                    const int8x16_t _a67 = vld1q_s8(pA + 48);
                    const int8x16_t _b0 = vcombine_s8(vreinterpret_s8_s32(vld1_dup_s32((const int*)pB)), vdup_n_s8(0));
                    const int8x16_t _b1 = vcombine_s8(vreinterpret_s8_s32(vld1_dup_s32((const int*)(pB + 4))), vdup_n_s8(0));
                    _sum0 = vdotq_laneq_s32(_sum0, _b0, _a01, 0);
                    _sum0 = vdotq_laneq_s32(_sum0, _b1, _a01, 1);
                    _sum1 = vdotq_laneq_s32(_sum1, _b0, _a01, 2);
                    _sum1 = vdotq_laneq_s32(_sum1, _b1, _a01, 3);
                    _sum2 = vdotq_laneq_s32(_sum2, _b0, _a23, 0);
                    _sum2 = vdotq_laneq_s32(_sum2, _b1, _a23, 1);
                    _sum3 = vdotq_laneq_s32(_sum3, _b0, _a23, 2);
                    _sum3 = vdotq_laneq_s32(_sum3, _b1, _a23, 3);
                    _sum4 = vdotq_laneq_s32(_sum4, _b0, _a45, 0);
                    _sum4 = vdotq_laneq_s32(_sum4, _b1, _a45, 1);
                    _sum5 = vdotq_laneq_s32(_sum5, _b0, _a45, 2);
                    _sum5 = vdotq_laneq_s32(_sum5, _b1, _a45, 3);
                    _sum6 = vdotq_laneq_s32(_sum6, _b0, _a67, 0);
                    _sum6 = vdotq_laneq_s32(_sum6, _b1, _a67, 1);
                    _sum7 = vdotq_laneq_s32(_sum7, _b0, _a67, 2);
                    _sum7 = vdotq_laneq_s32(_sum7, _b1, _a67, 3);
                    pA += 64;
                    pB += 8;
                }
#endif // __ARM_FEATURE_MATMUL_INT8
#if __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk; kk += 4)
                {
                    const int8x16_t _b = vcombine_s8(vreinterpret_s8_s32(vld1_dup_s32((const int*)pB)), vdup_n_s8(0));
                    const int8x16_t _a0 = vld1q_s8(pA);
                    const int8x16_t _a1 = vld1q_s8(pA + 16);
                    _sum0 = vdotq_laneq_s32(_sum0, _b, _a0, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _b, _a0, 1);
                    _sum2 = vdotq_laneq_s32(_sum2, _b, _a0, 2);
                    _sum3 = vdotq_laneq_s32(_sum3, _b, _a0, 3);
                    _sum4 = vdotq_laneq_s32(_sum4, _b, _a1, 0);
                    _sum5 = vdotq_laneq_s32(_sum5, _b, _a1, 1);
                    _sum6 = vdotq_laneq_s32(_sum6, _b, _a1, 2);
                    _sum7 = vdotq_laneq_s32(_sum7, _b, _a1, 3);
                    pA += 32;
                    pB += 4;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk; kk += 2)
                {
                    const int8x8_t _b = vreinterpret_s8_s16(vld1_dup_s16((const short*)pB));
                    const int16x8_t _a = vreinterpretq_s16_s8(vld1q_s8(pA));
                    const int16x4_t _a0 = vget_low_s16(_a);
                    const int16x4_t _a1 = vget_high_s16(_a);
                    _sum0 = vaddq_s32(_sum0, vpaddlq_s16(vmull_s8(_b, vreinterpret_s8_s16(vdup_lane_s16(_a0, 0)))));
                    _sum1 = vaddq_s32(_sum1, vpaddlq_s16(vmull_s8(_b, vreinterpret_s8_s16(vdup_lane_s16(_a0, 1)))));
                    _sum2 = vaddq_s32(_sum2, vpaddlq_s16(vmull_s8(_b, vreinterpret_s8_s16(vdup_lane_s16(_a0, 2)))));
                    _sum3 = vaddq_s32(_sum3, vpaddlq_s16(vmull_s8(_b, vreinterpret_s8_s16(vdup_lane_s16(_a0, 3)))));
                    _sum4 = vaddq_s32(_sum4, vpaddlq_s16(vmull_s8(_b, vreinterpret_s8_s16(vdup_lane_s16(_a1, 0)))));
                    _sum5 = vaddq_s32(_sum5, vpaddlq_s16(vmull_s8(_b, vreinterpret_s8_s16(vdup_lane_s16(_a1, 1)))));
                    _sum6 = vaddq_s32(_sum6, vpaddlq_s16(vmull_s8(_b, vreinterpret_s8_s16(vdup_lane_s16(_a1, 2)))));
                    _sum7 = vaddq_s32(_sum7, vpaddlq_s16(vmull_s8(_b, vreinterpret_s8_s16(vdup_lane_s16(_a1, 3)))));
                    pA += 16;
                    pB += 2;
                }
                if (kk < max_kk)
                {
                    const int8x8_t _b = vset_lane_s8(pB[0], vdup_n_s8(0), 0);
                    const int8x8_t _a = vld1_s8(pA);
                    const int16x8_t _p0 = vmull_s8(_b, vdup_lane_s8(_a, 0));
                    const int16x8_t _p1 = vmull_s8(_b, vdup_lane_s8(_a, 1));
                    const int16x8_t _p2 = vmull_s8(_b, vdup_lane_s8(_a, 2));
                    const int16x8_t _p3 = vmull_s8(_b, vdup_lane_s8(_a, 3));
                    const int16x8_t _p4 = vmull_s8(_b, vdup_lane_s8(_a, 4));
                    const int16x8_t _p5 = vmull_s8(_b, vdup_lane_s8(_a, 5));
                    const int16x8_t _p6 = vmull_s8(_b, vdup_lane_s8(_a, 6));
                    const int16x8_t _p7 = vmull_s8(_b, vdup_lane_s8(_a, 7));
                    _sum0 = vaddq_s32(_sum0, vmovl_s16(vget_low_s16(_p0)));
                    _sum1 = vaddq_s32(_sum1, vmovl_s16(vget_low_s16(_p1)));
                    _sum2 = vaddq_s32(_sum2, vmovl_s16(vget_low_s16(_p2)));
                    _sum3 = vaddq_s32(_sum3, vmovl_s16(vget_low_s16(_p3)));
                    _sum4 = vaddq_s32(_sum4, vmovl_s16(vget_low_s16(_p4)));
                    _sum5 = vaddq_s32(_sum5, vmovl_s16(vget_low_s16(_p5)));
                    _sum6 = vaddq_s32(_sum6, vmovl_s16(vget_low_s16(_p6)));
                    _sum7 = vaddq_s32(_sum7, vmovl_s16(vget_low_s16(_p7)));
                    pA += 8;
                    pB++;
                }

                const float32x4_t _bd = vdupq_n_f32(pB_descales[0]);
                const float32x4_t _ad0 = vld1q_f32(pA_descales);
                const float32x4_t _ad1 = vld1q_f32(pA_descales + 4);
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_laneq_f32(_bd, _ad0, 0));
                _fsum1 = vmlaq_f32(_fsum1, vcvtq_f32_s32(_sum1), vmulq_laneq_f32(_bd, _ad0, 1));
                _fsum2 = vmlaq_f32(_fsum2, vcvtq_f32_s32(_sum2), vmulq_laneq_f32(_bd, _ad0, 2));
                _fsum3 = vmlaq_f32(_fsum3, vcvtq_f32_s32(_sum3), vmulq_laneq_f32(_bd, _ad0, 3));
                _fsum4 = vmlaq_f32(_fsum4, vcvtq_f32_s32(_sum4), vmulq_laneq_f32(_bd, _ad1, 0));
                _fsum5 = vmlaq_f32(_fsum5, vcvtq_f32_s32(_sum5), vmulq_laneq_f32(_bd, _ad1, 1));
                _fsum6 = vmlaq_f32(_fsum6, vcvtq_f32_s32(_sum6), vmulq_laneq_f32(_bd, _ad1, 2));
                _fsum7 = vmlaq_f32(_fsum7, vcvtq_f32_s32(_sum7), vmulq_laneq_f32(_bd, _ad1, 3));
                pA_descales += 8;
                pB_descales++;
            }

            vst1q_lane_f32(outptr, _fsum0, 0);
            outptr++;
            vst1q_lane_f32(outptr, _fsum1, 0);
            outptr++;
            vst1q_lane_f32(outptr, _fsum2, 0);
            outptr++;
            vst1q_lane_f32(outptr, _fsum3, 0);
            outptr++;
            vst1q_lane_f32(outptr, _fsum4, 0);
            outptr++;
            vst1q_lane_f32(outptr, _fsum5, 0);
            outptr++;
            vst1q_lane_f32(outptr, _fsum6, 0);
            outptr++;
            vst1q_lane_f32(outptr, _fsum7, 0);
            outptr++;
        }
    }
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        const signed char* pA0_block = pAT + ii * A_hstep;
        const float* pA_descales0_block = pAT_descales + ii * A_descales_hstep;
        const signed char* pB = pBT;
        const float* pB_descales = pBT_descales;

        int jj = 0;
#if __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pB0 = pB;
            const signed char* pB1 = pB + 4 * K;
            const float* pB_descales0 = pB_descales;
            const float* pB_descales1 = pB_descales + 4 * ((K + block_size - 1) / block_size);
            float32x4_t _fsum0 = vdupq_n_f32(0.f);
            float32x4_t _fsum1 = vdupq_n_f32(0.f);
            float32x4_t _fsum2 = vdupq_n_f32(0.f);
            float32x4_t _fsum3 = vdupq_n_f32(0.f);
            float32x4_t _fsum4 = vdupq_n_f32(0.f);
            float32x4_t _fsum5 = vdupq_n_f32(0.f);
            float32x4_t _fsum6 = vdupq_n_f32(0.f);
            float32x4_t _fsum7 = vdupq_n_f32(0.f);

            const signed char* pA = pA0_block;
            const float* pA_descales = pA_descales0_block;
            for (int k = 0; k < K; k += block_size)
            {
                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);
                int32x4_t _sum2 = vdupq_n_s32(0);
                int32x4_t _sum3 = vdupq_n_s32(0);
                int32x4_t _sum4 = vdupq_n_s32(0);
                int32x4_t _sum5 = vdupq_n_s32(0);
                int32x4_t _sum6 = vdupq_n_s32(0);
                int32x4_t _sum7 = vdupq_n_s32(0);
                const int max_kk = std::min(K - k, block_size);
                int kk = 0;
#if __ARM_FEATURE_MATMUL_INT8
                int32x4_t _msum0 = vdupq_n_s32(0);
                int32x4_t _msum1 = vdupq_n_s32(0);
                int32x4_t _msum2 = vdupq_n_s32(0);
                int32x4_t _msum3 = vdupq_n_s32(0);
                int32x4_t _msum4 = vdupq_n_s32(0);
                int32x4_t _msum5 = vdupq_n_s32(0);
                int32x4_t _msum6 = vdupq_n_s32(0);
                int32x4_t _msum7 = vdupq_n_s32(0);
                for (; kk + 7 < max_kk; kk += 8)
                {
                    const int8x16_t _a01 = vld1q_s8(pA);
                    const int8x16_t _a23 = vld1q_s8(pA + 16);
                    const int8x16_t _b00 = vld1q_s8(pB0);
                    const int8x16_t _b01 = vld1q_s8(pB0 + 16);
                    const int8x16_t _b10 = vld1q_s8(pB1);
                    const int8x16_t _b11 = vld1q_s8(pB1 + 16);
                    _msum0 = vmmlaq_s32(_msum0, _a01, _b00);
                    _msum1 = vmmlaq_s32(_msum1, _a01, _b01);
                    _msum2 = vmmlaq_s32(_msum2, _a23, _b00);
                    _msum3 = vmmlaq_s32(_msum3, _a23, _b01);
                    _msum4 = vmmlaq_s32(_msum4, _a01, _b10);
                    _msum5 = vmmlaq_s32(_msum5, _a01, _b11);
                    _msum6 = vmmlaq_s32(_msum6, _a23, _b10);
                    _msum7 = vmmlaq_s32(_msum7, _a23, _b11);
                    pA += 32;
                    pB0 += 32;
                    pB1 += 32;
                }
                _sum0 = vcombine_s32(vget_low_s32(_msum0), vget_low_s32(_msum1));
                _sum1 = vcombine_s32(vget_low_s32(_msum4), vget_low_s32(_msum5));
                _sum2 = vcombine_s32(vget_high_s32(_msum0), vget_high_s32(_msum1));
                _sum3 = vcombine_s32(vget_high_s32(_msum4), vget_high_s32(_msum5));
                _sum4 = vcombine_s32(vget_low_s32(_msum2), vget_low_s32(_msum3));
                _sum5 = vcombine_s32(vget_low_s32(_msum6), vget_low_s32(_msum7));
                _sum6 = vcombine_s32(vget_high_s32(_msum2), vget_high_s32(_msum3));
                _sum7 = vcombine_s32(vget_high_s32(_msum6), vget_high_s32(_msum7));
#endif // __ARM_FEATURE_MATMUL_INT8
#if __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk; kk += 4)
                {
                    const int8x16_t _a = vld1q_s8(pA);
                    const int8x16_t _b0 = vld1q_s8(pB0);
                    const int8x16_t _b1 = vld1q_s8(pB1);
                    _sum0 = vdotq_laneq_s32(_sum0, _b0, _a, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _b1, _a, 0);
                    _sum2 = vdotq_laneq_s32(_sum2, _b0, _a, 1);
                    _sum3 = vdotq_laneq_s32(_sum3, _b1, _a, 1);
                    _sum4 = vdotq_laneq_s32(_sum4, _b0, _a, 2);
                    _sum5 = vdotq_laneq_s32(_sum5, _b1, _a, 2);
                    _sum6 = vdotq_laneq_s32(_sum6, _b0, _a, 3);
                    _sum7 = vdotq_laneq_s32(_sum7, _b1, _a, 3);
                    pA += 16;
                    pB0 += 16;
                    pB1 += 16;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk; kk += 2)
                {
                    const int16x4_t _a = vreinterpret_s16_s8(vld1_s8(pA));
                    const int8x8_t _b0 = vld1_s8(pB0);
                    const int8x8_t _b1 = vld1_s8(pB1);
                    _sum0 = vaddq_s32(_sum0, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a, 0)))));
                    _sum1 = vaddq_s32(_sum1, vpaddlq_s16(vmull_s8(_b1, vreinterpret_s8_s16(vdup_lane_s16(_a, 0)))));
                    _sum2 = vaddq_s32(_sum2, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a, 1)))));
                    _sum3 = vaddq_s32(_sum3, vpaddlq_s16(vmull_s8(_b1, vreinterpret_s8_s16(vdup_lane_s16(_a, 1)))));
                    _sum4 = vaddq_s32(_sum4, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a, 2)))));
                    _sum5 = vaddq_s32(_sum5, vpaddlq_s16(vmull_s8(_b1, vreinterpret_s8_s16(vdup_lane_s16(_a, 2)))));
                    _sum6 = vaddq_s32(_sum6, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a, 3)))));
                    _sum7 = vaddq_s32(_sum7, vpaddlq_s16(vmull_s8(_b1, vreinterpret_s8_s16(vdup_lane_s16(_a, 3)))));
                    pA += 8;
                    pB0 += 8;
                    pB1 += 8;
                }
                if (kk < max_kk)
                {
                    const int8x8_t _a = vreinterpret_s8_s32(vld1_lane_s32((const int*)pA, vdup_n_s32(0), 0));
                    const int8x8_t _b0 = vreinterpret_s8_s32(vld1_dup_s32((const int*)pB0));
                    const int8x8_t _b1 = vreinterpret_s8_s32(vld1_dup_s32((const int*)pB1));
                    const int16x8_t _p00 = vmull_s8(_b0, vdup_lane_s8(_a, 0));
                    const int16x8_t _p01 = vmull_s8(_b1, vdup_lane_s8(_a, 0));
                    const int16x8_t _p10 = vmull_s8(_b0, vdup_lane_s8(_a, 1));
                    const int16x8_t _p11 = vmull_s8(_b1, vdup_lane_s8(_a, 1));
                    const int16x8_t _p20 = vmull_s8(_b0, vdup_lane_s8(_a, 2));
                    const int16x8_t _p21 = vmull_s8(_b1, vdup_lane_s8(_a, 2));
                    const int16x8_t _p30 = vmull_s8(_b0, vdup_lane_s8(_a, 3));
                    const int16x8_t _p31 = vmull_s8(_b1, vdup_lane_s8(_a, 3));
                    _sum0 = vaddq_s32(_sum0, vmovl_s16(vget_low_s16(_p00)));
                    _sum1 = vaddq_s32(_sum1, vmovl_s16(vget_low_s16(_p01)));
                    _sum2 = vaddq_s32(_sum2, vmovl_s16(vget_low_s16(_p10)));
                    _sum3 = vaddq_s32(_sum3, vmovl_s16(vget_low_s16(_p11)));
                    _sum4 = vaddq_s32(_sum4, vmovl_s16(vget_low_s16(_p20)));
                    _sum5 = vaddq_s32(_sum5, vmovl_s16(vget_low_s16(_p21)));
                    _sum6 = vaddq_s32(_sum6, vmovl_s16(vget_low_s16(_p30)));
                    _sum7 = vaddq_s32(_sum7, vmovl_s16(vget_low_s16(_p31)));
                    pA += 4;
                    pB0 += 4;
                    pB1 += 4;
                }

                const float32x4_t _bd0 = vld1q_f32(pB_descales0);
                const float32x4_t _bd1 = vld1q_f32(pB_descales1);
                const float32x4_t _ad = vld1q_f32(pA_descales);
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_laneq_f32(_bd0, _ad, 0));
                _fsum1 = vmlaq_f32(_fsum1, vcvtq_f32_s32(_sum1), vmulq_laneq_f32(_bd1, _ad, 0));
                _fsum2 = vmlaq_f32(_fsum2, vcvtq_f32_s32(_sum2), vmulq_laneq_f32(_bd0, _ad, 1));
                _fsum3 = vmlaq_f32(_fsum3, vcvtq_f32_s32(_sum3), vmulq_laneq_f32(_bd1, _ad, 1));
                _fsum4 = vmlaq_f32(_fsum4, vcvtq_f32_s32(_sum4), vmulq_laneq_f32(_bd0, _ad, 2));
                _fsum5 = vmlaq_f32(_fsum5, vcvtq_f32_s32(_sum5), vmulq_laneq_f32(_bd1, _ad, 2));
                _fsum6 = vmlaq_f32(_fsum6, vcvtq_f32_s32(_sum6), vmulq_laneq_f32(_bd0, _ad, 3));
                _fsum7 = vmlaq_f32(_fsum7, vcvtq_f32_s32(_sum7), vmulq_laneq_f32(_bd1, _ad, 3));

                pA_descales += 4;
                pB_descales0 += 4;
                pB_descales1 += 4;
            }

            pB = pB1;
            pB_descales = pB_descales1;

            vst1q_f32(outptr, _fsum0);
            outptr += 4;
            vst1q_f32(outptr, _fsum1);
            outptr += 4;
            vst1q_f32(outptr, _fsum2);
            outptr += 4;
            vst1q_f32(outptr, _fsum3);
            outptr += 4;
            vst1q_f32(outptr, _fsum4);
            outptr += 4;
            vst1q_f32(outptr, _fsum5);
            outptr += 4;
            vst1q_f32(outptr, _fsum6);
            outptr += 4;
            vst1q_f32(outptr, _fsum7);
            outptr += 4;
        }
#endif // __aarch64__
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _fsum0 = vdupq_n_f32(0.f);
            float32x4_t _fsum1 = vdupq_n_f32(0.f);
            float32x4_t _fsum2 = vdupq_n_f32(0.f);
            float32x4_t _fsum3 = vdupq_n_f32(0.f);

            const signed char* pA = pA0_block;
            const float* pA_descales = pA_descales0_block;
            for (int k = 0; k < K; k += block_size)
            {
                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);
                int32x4_t _sum2 = vdupq_n_s32(0);
                int32x4_t _sum3 = vdupq_n_s32(0);
                const int max_kk = std::min(K - k, block_size);
                int kk = 0;
#if __ARM_FEATURE_MATMUL_INT8
                int32x4_t _msum0 = vdupq_n_s32(0);
                int32x4_t _msum1 = vdupq_n_s32(0);
                int32x4_t _msum2 = vdupq_n_s32(0);
                int32x4_t _msum3 = vdupq_n_s32(0);
                for (; kk + 7 < max_kk; kk += 8)
                {
                    const int8x16_t _b0 = vld1q_s8(pB);
                    const int8x16_t _b1 = vld1q_s8(pB + 16);
                    const int8x16_t _a01 = vld1q_s8(pA);
                    const int8x16_t _a23 = vld1q_s8(pA + 16);
                    _msum0 = vmmlaq_s32(_msum0, _a01, _b0);
                    _msum1 = vmmlaq_s32(_msum1, _a23, _b0);
                    _msum2 = vmmlaq_s32(_msum2, _a01, _b1);
                    _msum3 = vmmlaq_s32(_msum3, _a23, _b1);
                    pA += 32;
                    pB += 32;
                }
                _sum0 = vcombine_s32(vget_low_s32(_msum0), vget_low_s32(_msum2));
                _sum1 = vcombine_s32(vget_high_s32(_msum0), vget_high_s32(_msum2));
                _sum2 = vcombine_s32(vget_low_s32(_msum1), vget_low_s32(_msum3));
                _sum3 = vcombine_s32(vget_high_s32(_msum1), vget_high_s32(_msum3));
#endif // __ARM_FEATURE_MATMUL_INT8
#if __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk; kk += 4)
                {
                    const int8x16_t _b0 = vld1q_s8(pB);
                    const int8x16_t _a = vld1q_s8(pA);
                    _sum0 = vdotq_laneq_s32(_sum0, _b0, _a, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _b0, _a, 1);
                    _sum2 = vdotq_laneq_s32(_sum2, _b0, _a, 2);
                    _sum3 = vdotq_laneq_s32(_sum3, _b0, _a, 3);
                    pA += 16;
                    pB += 16;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk; kk += 2)
                {
                    const int8x8_t _b0 = vld1_s8(pB);
                    const int16x4_t _a = vreinterpret_s16_s8(vld1_s8(pA));
                    _sum0 = vaddq_s32(_sum0, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a, 0)))));
                    _sum1 = vaddq_s32(_sum1, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a, 1)))));
                    _sum2 = vaddq_s32(_sum2, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a, 2)))));
                    _sum3 = vaddq_s32(_sum3, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a, 3)))));
                    pA += 8;
                    pB += 8;
                }
                if (kk < max_kk)
                {
                    const int8x8_t _b = vreinterpret_s8_s32(vld1_dup_s32((const int*)(pB)));
                    const int8x8_t _a = vreinterpret_s8_s32(vld1_lane_s32((const int*)pA, vdup_n_s32(0), 0));
                    const int16x8_t _p0 = vmull_s8(_b, vdup_lane_s8(_a, 0));
                    const int16x8_t _p1 = vmull_s8(_b, vdup_lane_s8(_a, 1));
                    const int16x8_t _p2 = vmull_s8(_b, vdup_lane_s8(_a, 2));
                    const int16x8_t _p3 = vmull_s8(_b, vdup_lane_s8(_a, 3));
                    _sum0 = vaddq_s32(_sum0, vmovl_s16(vget_low_s16(_p0)));
                    _sum1 = vaddq_s32(_sum1, vmovl_s16(vget_low_s16(_p1)));
                    _sum2 = vaddq_s32(_sum2, vmovl_s16(vget_low_s16(_p2)));
                    _sum3 = vaddq_s32(_sum3, vmovl_s16(vget_low_s16(_p3)));
                    pA += 4;
                    pB += 4;
                }

                const float32x4_t _bd0 = vld1q_f32(pB_descales);
                const float32x4_t _ad = vld1q_f32(pA_descales);
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_n_f32(_bd0, vgetq_lane_f32(_ad, 0)));
                _fsum1 = vmlaq_f32(_fsum1, vcvtq_f32_s32(_sum1), vmulq_n_f32(_bd0, vgetq_lane_f32(_ad, 1)));
                _fsum2 = vmlaq_f32(_fsum2, vcvtq_f32_s32(_sum2), vmulq_n_f32(_bd0, vgetq_lane_f32(_ad, 2)));
                _fsum3 = vmlaq_f32(_fsum3, vcvtq_f32_s32(_sum3), vmulq_n_f32(_bd0, vgetq_lane_f32(_ad, 3)));

                pA_descales += 4;
                pB_descales += 4;
            }

            vst1q_f32(outptr, _fsum0);
            outptr += 4;
            vst1q_f32(outptr, _fsum1);
            outptr += 4;
            vst1q_f32(outptr, _fsum2);
            outptr += 4;
            vst1q_f32(outptr, _fsum3);
            outptr += 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float32x4_t _fsum0 = vdupq_n_f32(0.f);
            float32x4_t _fsum1 = vdupq_n_f32(0.f);
            float32x4_t _fsum2 = vdupq_n_f32(0.f);
            float32x4_t _fsum3 = vdupq_n_f32(0.f);

            const signed char* pA = pA0_block;
            const float* pA_descales = pA_descales0_block;
            for (int k = 0; k < K; k += block_size)
            {
                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);
                int32x4_t _sum2 = vdupq_n_s32(0);
                int32x4_t _sum3 = vdupq_n_s32(0);
                const int max_kk = std::min(K - k, block_size);
                int kk = 0;
#if __ARM_FEATURE_MATMUL_INT8
                int32x4_t _msum0 = vdupq_n_s32(0);
                int32x4_t _msum1 = vdupq_n_s32(0);
                for (; kk + 7 < max_kk; kk += 8)
                {
                    const int8x16_t _b0 = vld1q_s8(pB);
                    const int8x16_t _a01 = vld1q_s8(pA);
                    const int8x16_t _a23 = vld1q_s8(pA + 16);
                    _msum0 = vmmlaq_s32(_msum0, _a01, _b0);
                    _msum1 = vmmlaq_s32(_msum1, _a23, _b0);
                    pA += 32;
                    pB += 16;
                }
                _sum0 = vcombine_s32(vget_low_s32(_msum0), vdup_n_s32(0));
                _sum1 = vcombine_s32(vget_high_s32(_msum0), vdup_n_s32(0));
                _sum2 = vcombine_s32(vget_low_s32(_msum1), vdup_n_s32(0));
                _sum3 = vcombine_s32(vget_high_s32(_msum1), vdup_n_s32(0));
#endif // __ARM_FEATURE_MATMUL_INT8
#if __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk; kk += 4)
                {
                    const int8x16_t _b0 = vcombine_s8(vld1_s8(pB), vdup_n_s8(0));
                    const int8x16_t _a = vld1q_s8(pA);
                    _sum0 = vdotq_laneq_s32(_sum0, _b0, _a, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _b0, _a, 1);
                    _sum2 = vdotq_laneq_s32(_sum2, _b0, _a, 2);
                    _sum3 = vdotq_laneq_s32(_sum3, _b0, _a, 3);
                    pA += 16;
                    pB += 8;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk; kk += 2)
                {
                    const int8x8_t _b0 = vreinterpret_s8_s32(vld1_dup_s32((const int*)(pB)));
                    const int16x4_t _a = vreinterpret_s16_s8(vld1_s8(pA));
                    _sum0 = vaddq_s32(_sum0, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a, 0)))));
                    _sum1 = vaddq_s32(_sum1, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a, 1)))));
                    _sum2 = vaddq_s32(_sum2, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a, 2)))));
                    _sum3 = vaddq_s32(_sum3, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a, 3)))));
                    pA += 8;
                    pB += 4;
                }
                if (kk < max_kk)
                {
                    const int8x8_t _b = vreinterpret_s8_s16(vld1_dup_s16((const short*)(pB)));
                    const int8x8_t _a = vreinterpret_s8_s32(vld1_lane_s32((const int*)pA, vdup_n_s32(0), 0));
                    const int16x8_t _p0 = vmull_s8(_b, vdup_lane_s8(_a, 0));
                    const int16x8_t _p1 = vmull_s8(_b, vdup_lane_s8(_a, 1));
                    const int16x8_t _p2 = vmull_s8(_b, vdup_lane_s8(_a, 2));
                    const int16x8_t _p3 = vmull_s8(_b, vdup_lane_s8(_a, 3));
                    _sum0 = vaddq_s32(_sum0, vmovl_s16(vget_low_s16(_p0)));
                    _sum1 = vaddq_s32(_sum1, vmovl_s16(vget_low_s16(_p1)));
                    _sum2 = vaddq_s32(_sum2, vmovl_s16(vget_low_s16(_p2)));
                    _sum3 = vaddq_s32(_sum3, vmovl_s16(vget_low_s16(_p3)));
                    pA += 4;
                    pB += 2;
                }

                const float32x4_t _bd0 = vcombine_f32(vld1_f32(pB_descales), vdup_n_f32(0.f));
                const float32x4_t _ad = vld1q_f32(pA_descales);
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_n_f32(_bd0, vgetq_lane_f32(_ad, 0)));
                _fsum1 = vmlaq_f32(_fsum1, vcvtq_f32_s32(_sum1), vmulq_n_f32(_bd0, vgetq_lane_f32(_ad, 1)));
                _fsum2 = vmlaq_f32(_fsum2, vcvtq_f32_s32(_sum2), vmulq_n_f32(_bd0, vgetq_lane_f32(_ad, 2)));
                _fsum3 = vmlaq_f32(_fsum3, vcvtq_f32_s32(_sum3), vmulq_n_f32(_bd0, vgetq_lane_f32(_ad, 3)));

                pA_descales += 4;
                pB_descales += 2;
            }

            vst1_f32(outptr, vget_low_f32(_fsum0));
            outptr += 2;
            vst1_f32(outptr, vget_low_f32(_fsum1));
            outptr += 2;
            vst1_f32(outptr, vget_low_f32(_fsum2));
            outptr += 2;
            vst1_f32(outptr, vget_low_f32(_fsum3));
            outptr += 2;
        }
        for (; jj < max_jj; jj++)
        {
            float32x4_t _fsum0 = vdupq_n_f32(0.f);
            float32x4_t _fsum1 = vdupq_n_f32(0.f);
            float32x4_t _fsum2 = vdupq_n_f32(0.f);
            float32x4_t _fsum3 = vdupq_n_f32(0.f);

            const signed char* pA = pA0_block;
            const float* pA_descales = pA_descales0_block;
            for (int k = 0; k < K; k += block_size)
            {
                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);
                int32x4_t _sum2 = vdupq_n_s32(0);
                int32x4_t _sum3 = vdupq_n_s32(0);
                const int max_kk = std::min(K - k, block_size);
                int kk = 0;
#if __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk; kk += 8)
                {
                    const int8x16_t _a01 = vld1q_s8(pA);
                    const int8x16_t _a23 = vld1q_s8(pA + 16);
                    const int8x16_t _b0 = vcombine_s8(vreinterpret_s8_s32(vld1_dup_s32((const int*)pB)), vdup_n_s8(0));
                    const int8x16_t _b1 = vcombine_s8(vreinterpret_s8_s32(vld1_dup_s32((const int*)(pB + 4))), vdup_n_s8(0));
                    _sum0 = vdotq_laneq_s32(_sum0, _b0, _a01, 0);
                    _sum0 = vdotq_laneq_s32(_sum0, _b1, _a01, 1);
                    _sum1 = vdotq_laneq_s32(_sum1, _b0, _a01, 2);
                    _sum1 = vdotq_laneq_s32(_sum1, _b1, _a01, 3);
                    _sum2 = vdotq_laneq_s32(_sum2, _b0, _a23, 0);
                    _sum2 = vdotq_laneq_s32(_sum2, _b1, _a23, 1);
                    _sum3 = vdotq_laneq_s32(_sum3, _b0, _a23, 2);
                    _sum3 = vdotq_laneq_s32(_sum3, _b1, _a23, 3);
                    pA += 32;
                    pB += 8;
                }
#endif // __ARM_FEATURE_MATMUL_INT8
#if __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk; kk += 4)
                {
                    const int8x16_t _b0 = vcombine_s8(vreinterpret_s8_s32(vld1_dup_s32((const int*)pB)), vdup_n_s8(0));
                    const int8x16_t _a = vld1q_s8(pA);
                    _sum0 = vdotq_laneq_s32(_sum0, _b0, _a, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _b0, _a, 1);
                    _sum2 = vdotq_laneq_s32(_sum2, _b0, _a, 2);
                    _sum3 = vdotq_laneq_s32(_sum3, _b0, _a, 3);
                    pA += 16;
                    pB += 4;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk; kk += 2)
                {
                    const int8x8_t _b0 = vreinterpret_s8_s16(vld1_dup_s16((const short*)(pB)));
                    const int16x4_t _a = vreinterpret_s16_s8(vld1_s8(pA));
                    _sum0 = vaddq_s32(_sum0, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a, 0)))));
                    _sum1 = vaddq_s32(_sum1, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a, 1)))));
                    _sum2 = vaddq_s32(_sum2, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a, 2)))));
                    _sum3 = vaddq_s32(_sum3, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a, 3)))));
                    pA += 8;
                    pB += 2;
                }
                if (kk < max_kk)
                {
                    const int8x8_t _b = vset_lane_s8(pB[0], vdup_n_s8(0), 0);
                    const int8x8_t _a = vreinterpret_s8_s32(vld1_lane_s32((const int*)pA, vdup_n_s32(0), 0));
                    const int16x8_t _p0 = vmull_s8(_b, vdup_lane_s8(_a, 0));
                    const int16x8_t _p1 = vmull_s8(_b, vdup_lane_s8(_a, 1));
                    const int16x8_t _p2 = vmull_s8(_b, vdup_lane_s8(_a, 2));
                    const int16x8_t _p3 = vmull_s8(_b, vdup_lane_s8(_a, 3));
                    _sum0 = vaddq_s32(_sum0, vmovl_s16(vget_low_s16(_p0)));
                    _sum1 = vaddq_s32(_sum1, vmovl_s16(vget_low_s16(_p1)));
                    _sum2 = vaddq_s32(_sum2, vmovl_s16(vget_low_s16(_p2)));
                    _sum3 = vaddq_s32(_sum3, vmovl_s16(vget_low_s16(_p3)));
                    pA += 4;
                    pB += 1;
                }

                const float32x4_t _bd0 = vdupq_n_f32(pB_descales[0]);
                const float32x4_t _ad = vld1q_f32(pA_descales);
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_n_f32(_bd0, vgetq_lane_f32(_ad, 0)));
                _fsum1 = vmlaq_f32(_fsum1, vcvtq_f32_s32(_sum1), vmulq_n_f32(_bd0, vgetq_lane_f32(_ad, 1)));
                _fsum2 = vmlaq_f32(_fsum2, vcvtq_f32_s32(_sum2), vmulq_n_f32(_bd0, vgetq_lane_f32(_ad, 2)));
                _fsum3 = vmlaq_f32(_fsum3, vcvtq_f32_s32(_sum3), vmulq_n_f32(_bd0, vgetq_lane_f32(_ad, 3)));

                pA_descales += 4;
                pB_descales += 1;
            }

            vst1q_lane_f32(outptr, _fsum0, 0);
            outptr++;
            vst1q_lane_f32(outptr, _fsum1, 0);
            outptr++;
            vst1q_lane_f32(outptr, _fsum2, 0);
            outptr++;
            vst1q_lane_f32(outptr, _fsum3, 0);
            outptr++;
        }
    }
    for (; ii + 1 < max_ii; ii += 2)
    {
        const signed char* pA0_block = pAT + ii * A_hstep;
        const float* pA_descales0_block = pAT_descales + ii * A_descales_hstep;
        const signed char* pB = pBT;
        const float* pB_descales = pBT_descales;

        int jj = 0;
#if __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pB0 = pB;
            const signed char* pB1 = pB + 4 * K;
            const float* pB_descales0 = pB_descales;
            const float* pB_descales1 = pB_descales + 4 * ((K + block_size - 1) / block_size);
            float32x4_t _fsum0 = vdupq_n_f32(0.f);
            float32x4_t _fsum1 = vdupq_n_f32(0.f);
            float32x4_t _fsum2 = vdupq_n_f32(0.f);
            float32x4_t _fsum3 = vdupq_n_f32(0.f);

            const signed char* pA = pA0_block;
            const float* pA_descales = pA_descales0_block;
            for (int k = 0; k < K; k += block_size)
            {
                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);
                int32x4_t _sum2 = vdupq_n_s32(0);
                int32x4_t _sum3 = vdupq_n_s32(0);
                const int max_kk = std::min(K - k, block_size);
                int kk = 0;
#if __ARM_FEATURE_MATMUL_INT8
                int32x4_t _msum0 = vdupq_n_s32(0);
                int32x4_t _msum1 = vdupq_n_s32(0);
                int32x4_t _msum2 = vdupq_n_s32(0);
                int32x4_t _msum3 = vdupq_n_s32(0);
                for (; kk + 7 < max_kk; kk += 8)
                {
                    const int8x16_t _b0 = vld1q_s8(pB0);
                    const int8x16_t _b1 = vld1q_s8(pB0 + 16);
                    const int8x16_t _b2 = vld1q_s8(pB1);
                    const int8x16_t _b3 = vld1q_s8(pB1 + 16);
                    const int8x16_t _a0 = vld1q_s8(pA);
                    _msum0 = vmmlaq_s32(_msum0, _a0, _b0);
                    _msum1 = vmmlaq_s32(_msum1, _a0, _b1);
                    _msum2 = vmmlaq_s32(_msum2, _a0, _b2);
                    _msum3 = vmmlaq_s32(_msum3, _a0, _b3);
                    pA += 16;
                    pB0 += 32;
                    pB1 += 32;
                }
                _sum0 = vcombine_s32(vget_low_s32(_msum0), vget_low_s32(_msum1));
                _sum1 = vcombine_s32(vget_low_s32(_msum2), vget_low_s32(_msum3));
                _sum2 = vcombine_s32(vget_high_s32(_msum0), vget_high_s32(_msum1));
                _sum3 = vcombine_s32(vget_high_s32(_msum2), vget_high_s32(_msum3));
#endif // __ARM_FEATURE_MATMUL_INT8
#if __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk; kk += 4)
                {
                    const int8x16_t _b0 = vld1q_s8(pB0);
                    const int8x16_t _b1 = vld1q_s8(pB1);
                    const int8x16_t _a = vcombine_s8(vld1_s8(pA), vdup_n_s8(0));
                    _sum0 = vdotq_laneq_s32(_sum0, _b0, _a, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _b1, _a, 0);
                    _sum2 = vdotq_laneq_s32(_sum2, _b0, _a, 1);
                    _sum3 = vdotq_laneq_s32(_sum3, _b1, _a, 1);
                    pA += 8;
                    pB0 += 16;
                    pB1 += 16;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk; kk += 2)
                {
                    const int8x16_t _b = vcombine_s8(vld1_s8(pB0), vld1_s8(pB1));
                    const int8x8_t _b0 = vget_low_s8(_b);
                    const int8x8_t _b1 = vget_high_s8(_b);
                    const int16x4_t _a = vreinterpret_s16_s32(vld1_lane_s32((const int*)pA, vdup_n_s32(0), 0));
                    _sum0 = vaddq_s32(_sum0, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a, 0)))));
                    _sum1 = vaddq_s32(_sum1, vpaddlq_s16(vmull_s8(_b1, vreinterpret_s8_s16(vdup_lane_s16(_a, 0)))));
                    _sum2 = vaddq_s32(_sum2, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a, 1)))));
                    _sum3 = vaddq_s32(_sum3, vpaddlq_s16(vmull_s8(_b1, vreinterpret_s8_s16(vdup_lane_s16(_a, 1)))));
                    pA += 4;
                    pB0 += 8;
                    pB1 += 8;
                }
                if (kk < max_kk)
                {
                    const int8x8_t _b = vreinterpret_s8_s32(vld1_lane_s32((const int*)pB1, vld1_dup_s32((const int*)pB0), 1));
                    const int8x8_t _a = vreinterpret_s8_s16(vld1_lane_s16((const short*)pA, vdup_n_s16(0), 0));
                    const int16x8_t _p0 = vmull_s8(_b, vdup_lane_s8(_a, 0));
                    const int16x8_t _p1 = vmull_s8(_b, vdup_lane_s8(_a, 1));
                    _sum0 = vaddq_s32(_sum0, vmovl_s16(vget_low_s16(_p0)));
                    _sum1 = vaddq_s32(_sum1, vmovl_s16(vget_high_s16(_p0)));
                    _sum2 = vaddq_s32(_sum2, vmovl_s16(vget_low_s16(_p1)));
                    _sum3 = vaddq_s32(_sum3, vmovl_s16(vget_high_s16(_p1)));
                    pA += 2;
                    pB0 += 4;
                    pB1 += 4;
                }

                const float32x4_t _bd0 = vld1q_f32(pB_descales0);
                const float32x4_t _bd1 = vld1q_f32(pB_descales1);
                const float32x2_t _ad = vld1_f32(pA_descales);
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_lane_f32(_bd0, _ad, 0));
                _fsum1 = vmlaq_f32(_fsum1, vcvtq_f32_s32(_sum1), vmulq_lane_f32(_bd1, _ad, 0));
                _fsum2 = vmlaq_f32(_fsum2, vcvtq_f32_s32(_sum2), vmulq_lane_f32(_bd0, _ad, 1));
                _fsum3 = vmlaq_f32(_fsum3, vcvtq_f32_s32(_sum3), vmulq_lane_f32(_bd1, _ad, 1));

                pA_descales += 2;
                pB_descales0 += 4;
                pB_descales1 += 4;
            }

            pB = pB1;
            pB_descales = pB_descales1;

            vst1q_f32(outptr, _fsum0);
            outptr += 4;
            vst1q_f32(outptr, _fsum1);
            outptr += 4;
            vst1q_f32(outptr, _fsum2);
            outptr += 4;
            vst1q_f32(outptr, _fsum3);
            outptr += 4;
        }
#endif // __aarch64__
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _fsum0 = vdupq_n_f32(0.f);
            float32x4_t _fsum1 = vdupq_n_f32(0.f);

            const signed char* pA = pA0_block;
            const float* pA_descales = pA_descales0_block;
            for (int k = 0; k < K; k += block_size)
            {
                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);
                const int max_kk = std::min(K - k, block_size);
                int kk = 0;
#if __ARM_FEATURE_MATMUL_INT8
                int32x4_t _msum0 = vdupq_n_s32(0);
                int32x4_t _msum1 = vdupq_n_s32(0);
                for (; kk + 7 < max_kk; kk += 8)
                {
                    const int8x16_t _b0 = vld1q_s8(pB + 0);
                    const int8x16_t _b1 = vld1q_s8(pB + 16);
                    const int8x16_t _a0 = vld1q_s8(pA);
                    _msum0 = vmmlaq_s32(_msum0, _a0, _b0);
                    _msum1 = vmmlaq_s32(_msum1, _a0, _b1);
                    pA += 16;
                    pB += 32;
                }
                _sum0 = vcombine_s32(vget_low_s32(_msum0), vget_low_s32(_msum1));
                _sum1 = vcombine_s32(vget_high_s32(_msum0), vget_high_s32(_msum1));
#endif // __ARM_FEATURE_MATMUL_INT8
#if __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk; kk += 4)
                {
                    const int8x16_t _b0 = vld1q_s8(pB);
                    const int8x16_t _a = vcombine_s8(vld1_s8(pA), vdup_n_s8(0));
                    _sum0 = vdotq_laneq_s32(_sum0, _b0, _a, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _b0, _a, 1);
                    pA += 8;
                    pB += 16;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk; kk += 2)
                {
                    const int8x8_t _b0 = vld1_s8(pB);
                    const int16x4_t _a = vreinterpret_s16_s32(vld1_lane_s32((const int*)pA, vdup_n_s32(0), 0));
                    _sum0 = vaddq_s32(_sum0, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a, 0)))));
                    _sum1 = vaddq_s32(_sum1, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a, 1)))));
                    pA += 4;
                    pB += 8;
                }
                if (kk < max_kk)
                {
                    const int8x8_t _b = vreinterpret_s8_s32(vld1_dup_s32((const int*)(pB)));
                    const int8x8_t _a = vreinterpret_s8_s16(vld1_lane_s16((const short*)pA, vdup_n_s16(0), 0));
                    const int16x8_t _p0 = vmull_s8(_b, vdup_lane_s8(_a, 0));
                    const int16x8_t _p1 = vmull_s8(_b, vdup_lane_s8(_a, 1));
                    _sum0 = vaddq_s32(_sum0, vmovl_s16(vget_low_s16(_p0)));
                    _sum1 = vaddq_s32(_sum1, vmovl_s16(vget_low_s16(_p1)));
                    pA += 2;
                    pB += 4;
                }

                const float32x4_t _bd0 = vld1q_f32(pB_descales);
                const float32x2_t _ad = vld1_f32(pA_descales);
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_lane_f32(_bd0, _ad, 0));
                _fsum1 = vmlaq_f32(_fsum1, vcvtq_f32_s32(_sum1), vmulq_lane_f32(_bd0, _ad, 1));

                pA_descales += 2;
                pB_descales += 4;
            }

            vst1q_f32(outptr, _fsum0);
            outptr += 4;
            vst1q_f32(outptr, _fsum1);
            outptr += 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float32x4_t _fsum0 = vdupq_n_f32(0.f);
            float32x4_t _fsum1 = vdupq_n_f32(0.f);

            const signed char* pA = pA0_block;
            const float* pA_descales = pA_descales0_block;
            for (int k = 0; k < K; k += block_size)
            {
                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);
                const int max_kk = std::min(K - k, block_size);
                int kk = 0;
#if __ARM_FEATURE_MATMUL_INT8
                int32x4_t _msum0 = vdupq_n_s32(0);
                for (; kk + 7 < max_kk; kk += 8)
                {
                    const int8x16_t _b0 = vld1q_s8(pB + 0);
                    const int8x16_t _a0 = vld1q_s8(pA);
                    _msum0 = vmmlaq_s32(_msum0, _a0, _b0);
                    pA += 16;
                    pB += 16;
                }
                _sum0 = vcombine_s32(vget_low_s32(_msum0), vdup_n_s32(0));
                _sum1 = vcombine_s32(vget_high_s32(_msum0), vdup_n_s32(0));
#endif // __ARM_FEATURE_MATMUL_INT8
#if __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk; kk += 4)
                {
                    const int8x16_t _b0 = vcombine_s8(vld1_s8(pB), vdup_n_s8(0));
                    const int8x16_t _a = vcombine_s8(vld1_s8(pA), vdup_n_s8(0));
                    _sum0 = vdotq_laneq_s32(_sum0, _b0, _a, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _b0, _a, 1);
                    pA += 8;
                    pB += 8;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk; kk += 2)
                {
                    const int8x8_t _b0 = vreinterpret_s8_s32(vld1_dup_s32((const int*)(pB)));
                    const int16x4_t _a = vreinterpret_s16_s32(vld1_lane_s32((const int*)pA, vdup_n_s32(0), 0));
                    _sum0 = vaddq_s32(_sum0, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a, 0)))));
                    _sum1 = vaddq_s32(_sum1, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a, 1)))));
                    pA += 4;
                    pB += 4;
                }
                if (kk < max_kk)
                {
                    const int8x8_t _b = vreinterpret_s8_s16(vld1_dup_s16((const short*)(pB)));
                    const int8x8_t _a = vreinterpret_s8_s16(vld1_lane_s16((const short*)pA, vdup_n_s16(0), 0));
                    const int16x8_t _p0 = vmull_s8(_b, vdup_lane_s8(_a, 0));
                    const int16x8_t _p1 = vmull_s8(_b, vdup_lane_s8(_a, 1));
                    _sum0 = vaddq_s32(_sum0, vmovl_s16(vget_low_s16(_p0)));
                    _sum1 = vaddq_s32(_sum1, vmovl_s16(vget_low_s16(_p1)));
                    pA += 2;
                    pB += 2;
                }

                const float32x4_t _bd0 = vcombine_f32(vld1_f32(pB_descales), vdup_n_f32(0.f));
                const float32x2_t _ad = vld1_f32(pA_descales);
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_lane_f32(_bd0, _ad, 0));
                _fsum1 = vmlaq_f32(_fsum1, vcvtq_f32_s32(_sum1), vmulq_lane_f32(_bd0, _ad, 1));

                pA_descales += 2;
                pB_descales += 2;
            }

            vst1_f32(outptr, vget_low_f32(_fsum0));
            outptr += 2;
            vst1_f32(outptr, vget_low_f32(_fsum1));
            outptr += 2;
        }
        for (; jj < max_jj; jj++)
        {
            float32x4_t _fsum0 = vdupq_n_f32(0.f);
            float32x4_t _fsum1 = vdupq_n_f32(0.f);

            const signed char* pA = pA0_block;
            const float* pA_descales = pA_descales0_block;
            for (int k = 0; k < K; k += block_size)
            {
                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);
                const int max_kk = std::min(K - k, block_size);
                int kk = 0;
#if __ARM_FEATURE_MATMUL_INT8
                for (; kk + 7 < max_kk; kk += 8)
                {
                    const int8x16_t _a = vld1q_s8(pA);
                    const int8x16_t _b0 = vcombine_s8(vreinterpret_s8_s32(vld1_dup_s32((const int*)pB)), vdup_n_s8(0));
                    const int8x16_t _b1 = vcombine_s8(vreinterpret_s8_s32(vld1_dup_s32((const int*)(pB + 4))), vdup_n_s8(0));
                    _sum0 = vdotq_laneq_s32(_sum0, _b0, _a, 0);
                    _sum0 = vdotq_laneq_s32(_sum0, _b1, _a, 1);
                    _sum1 = vdotq_laneq_s32(_sum1, _b0, _a, 2);
                    _sum1 = vdotq_laneq_s32(_sum1, _b1, _a, 3);
                    pA += 16;
                    pB += 8;
                }
#endif // __ARM_FEATURE_MATMUL_INT8
#if __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk; kk += 4)
                {
                    const int8x16_t _b0 = vcombine_s8(vreinterpret_s8_s32(vld1_dup_s32((const int*)pB)), vdup_n_s8(0));
                    const int8x16_t _a = vcombine_s8(vld1_s8(pA), vdup_n_s8(0));
                    _sum0 = vdotq_laneq_s32(_sum0, _b0, _a, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _b0, _a, 1);
                    pA += 8;
                    pB += 4;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk; kk += 2)
                {
                    const int8x8_t _b0 = vreinterpret_s8_s16(vld1_dup_s16((const short*)(pB)));
                    const int16x4_t _a = vreinterpret_s16_s32(vld1_lane_s32((const int*)pA, vdup_n_s32(0), 0));
                    _sum0 = vaddq_s32(_sum0, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a, 0)))));
                    _sum1 = vaddq_s32(_sum1, vpaddlq_s16(vmull_s8(_b0, vreinterpret_s8_s16(vdup_lane_s16(_a, 1)))));
                    pA += 4;
                    pB += 2;
                }
                if (kk < max_kk)
                {
                    const int8x8_t _b = vset_lane_s8(pB[0], vdup_n_s8(0), 0);
                    const int8x8_t _a = vreinterpret_s8_s16(vld1_lane_s16((const short*)pA, vdup_n_s16(0), 0));
                    const int16x8_t _p0 = vmull_s8(_b, vdup_lane_s8(_a, 0));
                    const int16x8_t _p1 = vmull_s8(_b, vdup_lane_s8(_a, 1));
                    _sum0 = vaddq_s32(_sum0, vmovl_s16(vget_low_s16(_p0)));
                    _sum1 = vaddq_s32(_sum1, vmovl_s16(vget_low_s16(_p1)));
                    pA += 2;
                    pB += 1;
                }

                const float32x4_t _bd0 = vdupq_n_f32(pB_descales[0]);
                const float32x2_t _ad = vld1_f32(pA_descales);
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_lane_f32(_bd0, _ad, 0));
                _fsum1 = vmlaq_f32(_fsum1, vcvtq_f32_s32(_sum1), vmulq_lane_f32(_bd0, _ad, 1));

                pA_descales += 2;
                pB_descales += 1;
            }

            vst1q_lane_f32(outptr, _fsum0, 0);
            outptr++;
            vst1q_lane_f32(outptr, _fsum1, 0);
            outptr++;
        }
    }
    for (; ii < max_ii; ii++)
    {
        const signed char* pA0_block = pAT + ii * A_hstep;
        const float* pA_descales0_block = pAT_descales + ii * A_descales_hstep;
        const signed char* pB = pBT;
        const float* pB_descales = pBT_descales;

        int jj = 0;
#if __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            const signed char* pB0 = pB;
            const signed char* pB1 = pB + 4 * K;
            const float* pB_descales0 = pB_descales;
            const float* pB_descales1 = pB_descales + 4 * ((K + block_size - 1) / block_size);
            float32x4_t _fsum0 = vdupq_n_f32(0.f);
            float32x4_t _fsum1 = vdupq_n_f32(0.f);

            const signed char* pA0 = pA0_block;
            const float* pA_descales0 = pA_descales0_block;
            for (int k = 0; k < K; k += block_size)
            {
                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);
                const int max_kk = std::min(K - k, block_size);
                int kk = 0;
#if __ARM_FEATURE_MATMUL_INT8
                int32x4_t _msum0 = vdupq_n_s32(0);
                int32x4_t _msum1 = vdupq_n_s32(0);
                int32x4_t _msum2 = vdupq_n_s32(0);
                int32x4_t _msum3 = vdupq_n_s32(0);
                for (; kk + 7 < max_kk; kk += 8)
                {
                    const int8x16_t _b0 = vld1q_s8(pB0);
                    const int8x16_t _b1 = vld1q_s8(pB0 + 16);
                    const int8x16_t _b2 = vld1q_s8(pB1);
                    const int8x16_t _b3 = vld1q_s8(pB1 + 16);
                    const int8x16_t _a0 = vcombine_s8(vld1_s8(pA0 + kk), vdup_n_s8(0));
                    _msum0 = vmmlaq_s32(_msum0, _a0, _b0);
                    _msum1 = vmmlaq_s32(_msum1, _a0, _b1);
                    _msum2 = vmmlaq_s32(_msum2, _a0, _b2);
                    _msum3 = vmmlaq_s32(_msum3, _a0, _b3);
                    pB0 += 32;
                    pB1 += 32;
                }
                _sum0 = vcombine_s32(vget_low_s32(_msum0), vget_low_s32(_msum1));
                _sum1 = vcombine_s32(vget_low_s32(_msum2), vget_low_s32(_msum3));
#endif // __ARM_FEATURE_MATMUL_INT8
#if __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk; kk += 4)
                {
                    const int8x16_t _b0 = vld1q_s8(pB0);
                    const int8x16_t _b1 = vld1q_s8(pB1);
                    const int8x16_t _a0 = vreinterpretq_s8_s32(vdupq_lane_s32(vld1_lane_s32((const int*)(pA0 + kk), vdup_n_s32(0), 0), 0));
                    _sum0 = vdotq_s32(_sum0, _b0, _a0);
                    _sum1 = vdotq_s32(_sum1, _b1, _a0);
                    pB0 += 16;
                    pB1 += 16;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk; kk += 2)
                {
                    const int8x16_t _b = vcombine_s8(vld1_s8(pB0), vld1_s8(pB1));
                    const int8x8_t _b0 = vget_low_s8(_b);
                    const int8x8_t _b1 = vget_high_s8(_b);
                    const int8x8_t _a0 = vreinterpret_s8_s16(vdup_lane_s16(vld1_lane_s16((const short*)(pA0 + kk), vdup_n_s16(0), 0), 0));
                    _sum0 = vaddq_s32(_sum0, vpaddlq_s16(vmull_s8(_b0, _a0)));
                    _sum1 = vaddq_s32(_sum1, vpaddlq_s16(vmull_s8(_b1, _a0)));
                    pB0 += 8;
                    pB1 += 8;
                }
                if (kk < max_kk)
                {
                    const int8x8_t _b = vreinterpret_s8_s32(vld1_lane_s32((const int*)pB1, vld1_dup_s32((const int*)pB0), 1));
                    const int8x8_t _a0 = vld1_lane_s8(pA0 + kk, vdup_n_s8(0), 0);
                    const int16x8_t _p0 = vmull_s8(_b, vdup_lane_s8(_a0, 0));
                    _sum0 = vaddq_s32(_sum0, vmovl_s16(vget_low_s16(_p0)));
                    _sum1 = vaddq_s32(_sum1, vmovl_s16(vget_high_s16(_p0)));
                    pB0 += 4;
                    pB1 += 4;
                }

                const float32x4_t _bd0 = vld1q_f32(pB_descales0);
                const float32x4_t _bd1 = vld1q_f32(pB_descales1);
                const float _ad0 = pA_descales0[0];
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_n_f32(_bd0, _ad0));
                _fsum1 = vmlaq_f32(_fsum1, vcvtq_f32_s32(_sum1), vmulq_n_f32(_bd1, _ad0));

                pA0 += max_kk;
                pA_descales0++;
                pB_descales0 += 4;
                pB_descales1 += 4;
            }

            pB = pB1;
            pB_descales = pB_descales1;

            vst1q_f32(outptr, _fsum0);
            outptr += 4;
            vst1q_f32(outptr, _fsum1);
            outptr += 4;
        }
#endif // __aarch64__
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _fsum0 = vdupq_n_f32(0.f);

            const signed char* pA0 = pA0_block;
            const float* pA_descales0 = pA_descales0_block;
            for (int k = 0; k < K; k += block_size)
            {
                int32x4_t _sum0 = vdupq_n_s32(0);
                const int max_kk = std::min(K - k, block_size);
                int kk = 0;
#if __ARM_FEATURE_MATMUL_INT8
                int32x4_t _msum0 = vdupq_n_s32(0);
                int32x4_t _msum1 = vdupq_n_s32(0);
                for (; kk + 7 < max_kk; kk += 8)
                {
                    const int8x16_t _b0 = vld1q_s8(pB + 0);
                    const int8x16_t _b1 = vld1q_s8(pB + 16);
                    const int8x16_t _a0 = vcombine_s8(vld1_s8(pA0 + kk), vdup_n_s8(0));
                    _msum0 = vmmlaq_s32(_msum0, _a0, _b0);
                    _msum1 = vmmlaq_s32(_msum1, _a0, _b1);
                    pB += 32;
                }
                _sum0 = vcombine_s32(vget_low_s32(_msum0), vget_low_s32(_msum1));
#endif // __ARM_FEATURE_MATMUL_INT8
#if __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk; kk += 4)
                {
                    const int8x16_t _b0 = vld1q_s8(pB);
                    const int8x16_t _a0 = vreinterpretq_s8_s32(vdupq_lane_s32(vld1_lane_s32((const int*)(pA0 + kk), vdup_n_s32(0), 0), 0));
                    _sum0 = vdotq_s32(_sum0, _b0, _a0);
                    pB += 16;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk; kk += 2)
                {
                    const int8x8_t _b0 = vld1_s8(pB);
                    const int8x8_t _a0 = vreinterpret_s8_s16(vdup_lane_s16(vld1_lane_s16((const short*)(pA0 + kk), vdup_n_s16(0), 0), 0));
                    _sum0 = vaddq_s32(_sum0, vpaddlq_s16(vmull_s8(_b0, _a0)));
                    pB += 8;
                }
                if (kk < max_kk)
                {
                    const int8x8_t _b = vreinterpret_s8_s32(vld1_dup_s32((const int*)(pB)));
                    const int8x8_t _a0 = vld1_lane_s8(pA0 + kk, vdup_n_s8(0), 0);
                    const int16x8_t _p0 = vmull_s8(_b, vdup_lane_s8(_a0, 0));
                    _sum0 = vaddq_s32(_sum0, vmovl_s16(vget_low_s16(_p0)));
                    pB += 4;
                }

                const float32x4_t _bd0 = vld1q_f32(pB_descales);
                const float _ad0 = pA_descales0[0];
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_n_f32(_bd0, _ad0));

                pA0 += max_kk;
                pA_descales0++;
                pB_descales += 4;
            }

            vst1q_f32(outptr, _fsum0);
            outptr += 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float32x4_t _fsum0 = vdupq_n_f32(0.f);

            const signed char* pA0 = pA0_block;
            const float* pA_descales0 = pA_descales0_block;
            for (int k = 0; k < K; k += block_size)
            {
                int32x4_t _sum0 = vdupq_n_s32(0);
                const int max_kk = std::min(K - k, block_size);
                int kk = 0;
#if __ARM_FEATURE_MATMUL_INT8
                int32x4_t _msum0 = vdupq_n_s32(0);
                for (; kk + 7 < max_kk; kk += 8)
                {
                    const int8x16_t _b0 = vld1q_s8(pB + 0);
                    const int8x16_t _a0 = vcombine_s8(vld1_s8(pA0 + kk), vdup_n_s8(0));
                    _msum0 = vmmlaq_s32(_msum0, _a0, _b0);
                    pB += 16;
                }
                _sum0 = vcombine_s32(vget_low_s32(_msum0), vdup_n_s32(0));
#endif // __ARM_FEATURE_MATMUL_INT8
#if __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk; kk += 4)
                {
                    const int8x16_t _b0 = vcombine_s8(vld1_s8(pB), vdup_n_s8(0));
                    const int8x16_t _a0 = vreinterpretq_s8_s32(vdupq_lane_s32(vld1_lane_s32((const int*)(pA0 + kk), vdup_n_s32(0), 0), 0));
                    _sum0 = vdotq_s32(_sum0, _b0, _a0);
                    pB += 8;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk; kk += 2)
                {
                    const int8x8_t _b0 = vreinterpret_s8_s32(vld1_dup_s32((const int*)(pB)));
                    const int8x8_t _a0 = vreinterpret_s8_s16(vdup_lane_s16(vld1_lane_s16((const short*)(pA0 + kk), vdup_n_s16(0), 0), 0));
                    _sum0 = vaddq_s32(_sum0, vpaddlq_s16(vmull_s8(_b0, _a0)));
                    pB += 4;
                }
                if (kk < max_kk)
                {
                    const int8x8_t _b = vreinterpret_s8_s16(vld1_dup_s16((const short*)(pB)));
                    const int8x8_t _a0 = vld1_lane_s8(pA0 + kk, vdup_n_s8(0), 0);
                    const int16x8_t _p0 = vmull_s8(_b, vdup_lane_s8(_a0, 0));
                    _sum0 = vaddq_s32(_sum0, vmovl_s16(vget_low_s16(_p0)));
                    pB += 2;
                }

                const float32x4_t _bd0 = vcombine_f32(vld1_f32(pB_descales), vdup_n_f32(0.f));
                const float _ad0 = pA_descales0[0];
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_n_f32(_bd0, _ad0));

                pA0 += max_kk;
                pA_descales0++;
                pB_descales += 2;
            }

            vst1_f32(outptr, vget_low_f32(_fsum0));
            outptr += 2;
        }
        for (; jj < max_jj; jj++)
        {
            float32x4_t _fsum0 = vdupq_n_f32(0.f);

            const signed char* pA0 = pA0_block;
            const float* pA_descales0 = pA_descales0_block;
            for (int k = 0; k < K; k += block_size)
            {
                int32x4_t _sum0 = vdupq_n_s32(0);
                const int max_kk = std::min(K - k, block_size);
                int kk = 0;
#if __ARM_FEATURE_DOTPROD
                for (; kk + 3 < max_kk; kk += 4)
                {
                    const int8x16_t _b0 = vcombine_s8(vreinterpret_s8_s32(vld1_dup_s32((const int*)pB)), vdup_n_s8(0));
                    const int8x16_t _a0 = vreinterpretq_s8_s32(vdupq_lane_s32(vld1_lane_s32((const int*)(pA0 + kk), vdup_n_s32(0), 0), 0));
                    _sum0 = vdotq_s32(_sum0, _b0, _a0);
                    pB += 4;
                }
#endif // __ARM_FEATURE_DOTPROD
                for (; kk + 1 < max_kk; kk += 2)
                {
                    const int8x8_t _b0 = vreinterpret_s8_s16(vld1_dup_s16((const short*)(pB)));
                    const int8x8_t _a0 = vreinterpret_s8_s16(vdup_lane_s16(vld1_lane_s16((const short*)(pA0 + kk), vdup_n_s16(0), 0), 0));
                    _sum0 = vaddq_s32(_sum0, vpaddlq_s16(vmull_s8(_b0, _a0)));
                    pB += 2;
                }
                if (kk < max_kk)
                {
                    const int8x8_t _b = vset_lane_s8(pB[0], vdup_n_s8(0), 0);
                    const int8x8_t _a0 = vld1_lane_s8(pA0 + kk, vdup_n_s8(0), 0);
                    const int16x8_t _p0 = vmull_s8(_b, vdup_lane_s8(_a0, 0));
                    _sum0 = vaddq_s32(_sum0, vmovl_s16(vget_low_s16(_p0)));
                    pB += 1;
                }

                const float32x4_t _bd0 = vdupq_n_f32(pB_descales[0]);
                const float _ad0 = pA_descales0[0];
                _fsum0 = vmlaq_f32(_fsum0, vcvtq_f32_s32(_sum0), vmulq_n_f32(_bd0, _ad0));

                pA0 += max_kk;
                pA_descales0++;
                pB_descales += 1;
            }

            vst1q_lane_f32(outptr, _fsum0, 0);
            outptr++;
        }
    }
#else
#if __ARM_FEATURE_SIMD32 && NCNN_GNU_INLINE_ASM
    for (; ii + 1 < max_ii; ii += 2)
    {
        const signed char* pA_block = pAT + ii * A_hstep;
        const float* pA_descales_block = pAT_descales + ii * A_descales_hstep;
        const signed char* pB = pBT;
        const float* pB_descales = pBT_descales;

        int jj = 0;
        for (; jj + 1 < max_jj; jj += 2)
        {
            float fsum00 = 0.f;
            float fsum01 = 0.f;
            float fsum10 = 0.f;
            float fsum11 = 0.f;

            const signed char* pA = pA_block;
            const float* pA_descales = pA_descales_block;
            for (int k = 0; k < K; k += block_size)
            {
                int sum00 = 0;
                int sum01 = 0;
                int sum10 = 0;
                int sum11 = 0;
                const int max_kk = std::min(K - k, block_size);
                int kk = 0;
                for (; kk + 1 < max_kk; kk += 2)
                {
#if __OPTIMIZE__
                    asm volatile(
                        "ldr    r2, [%0], #4    \n"
                        "ldr    r4, [%1], #4    \n"
                        "ror    r3, r2, #8      \n"
                        "ror    r5, r4, #8      \n"
                        "sxtb16 r2, r2          \n"
                        "sxtb16 r4, r4          \n"
                        "sxtb16 r3, r3          \n"
                        "sxtb16 r5, r5          \n"
                        "smlad  %2, r2, r4, %2  \n"
                        "smlad  %3, r3, r4, %3  \n"
                        "smlad  %4, r2, r5, %4  \n"
                        "smlad  %5, r3, r5, %5  \n"
                        : "=r"(pA),
                        "=r"(pB),
                        "=r"(sum00),
                        "=r"(sum10),
                        "=r"(sum01),
                        "=r"(sum11)
                        : "0"(pA),
                        "1"(pB),
                        "2"(sum00),
                        "3"(sum10),
                        "4"(sum01),
                        "5"(sum11)
                        : "memory", "r2", "r3", "r4", "r5");
#else
                    int _pA0 = *((int*)pA);
                    int _pB0 = *((int*)pB);
                    int _pA1;
                    int _pB1;
                    asm volatile("ror %0, %1, #8"
                                 : "=r"(_pA1)
                                 : "r"(_pA0)
                                 :);
                    asm volatile("ror %0, %1, #8"
                                 : "=r"(_pB1)
                                 : "r"(_pB0)
                                 :);
                    asm volatile("sxtb16 %0, %0"
                                 : "=r"(_pA0)
                                 : "0"(_pA0)
                                 :);
                    asm volatile("sxtb16 %0, %0"
                                 : "=r"(_pA1)
                                 : "0"(_pA1)
                                 :);
                    asm volatile("sxtb16 %0, %0"
                                 : "=r"(_pB0)
                                 : "0"(_pB0)
                                 :);
                    asm volatile("sxtb16 %0, %0"
                                 : "=r"(_pB1)
                                 : "0"(_pB1)
                                 :);
                    asm volatile("smlad %0, %2, %3, %0"
                                 : "=r"(sum00)
                                 : "0"(sum00), "r"(_pA0), "r"(_pB0)
                                 :);
                    asm volatile("smlad %0, %2, %3, %0"
                                 : "=r"(sum10)
                                 : "0"(sum10), "r"(_pA1), "r"(_pB0)
                                 :);
                    asm volatile("smlad %0, %2, %3, %0"
                                 : "=r"(sum01)
                                 : "0"(sum01), "r"(_pA0), "r"(_pB1)
                                 :);
                    asm volatile("smlad %0, %2, %3, %0"
                                 : "=r"(sum11)
                                 : "0"(sum11), "r"(_pA1), "r"(_pB1)
                                 :);
                    pA += 4;
                    pB += 4;
#endif
                }
                if (kk < max_kk)
                {
                    sum00 += pA[0] * pB[0];
                    sum10 += pA[1] * pB[0];
                    sum01 += pA[0] * pB[1];
                    sum11 += pA[1] * pB[1];
                    pA += 2;
                    pB += 2;
                }

                const float bd0 = pB_descales[0];
                const float bd1 = pB_descales[1];
                const float ad0 = pA_descales[0];
                const float ad1 = pA_descales[1];
                fsum00 += sum00 * ad0 * bd0;
                fsum01 += sum01 * ad0 * bd1;
                fsum10 += sum10 * ad1 * bd0;
                fsum11 += sum11 * ad1 * bd1;

                pA_descales += 2;
                pB_descales += 2;
            }

            *outptr++ = fsum00;
            *outptr++ = fsum01;
            *outptr++ = fsum10;
            *outptr++ = fsum11;
        }
        for (; jj < max_jj; jj++)
        {
            float fsum00 = 0.f;
            float fsum10 = 0.f;

            const signed char* pA = pA_block;
            const float* pA_descales = pA_descales_block;
            for (int k = 0; k < K; k += block_size)
            {
                int sum00 = 0;
                int sum10 = 0;
                const int max_kk = std::min(K - k, block_size);
                for (int kk = 0; kk < max_kk; kk++)
                {
                    sum00 += pA[0] * pB[0];
                    sum10 += pA[1] * pB[0];
                    pA += 2;
                    pB++;
                }

                const float bd0 = pB_descales[0];
                const float ad0 = pA_descales[0];
                const float ad1 = pA_descales[1];
                fsum00 += sum00 * ad0 * bd0;
                fsum10 += sum10 * ad1 * bd0;

                pA_descales += 2;
                pB_descales++;
            }

            *outptr++ = fsum00;
            *outptr++ = fsum10;
        }
    }
#else
    for (; ii + 1 < max_ii; ii += 2)
    {
        const signed char* pA0_block = pAT + ii * A_hstep;
        const signed char* pA1_block = pA0_block + A_hstep;
        const float* pA_descales0_block = pAT_descales + ii * A_descales_hstep;
        const float* pA_descales1_block = pA_descales0_block + A_descales_hstep;
        const signed char* pB = pBT;
        const float* pB_descales = pBT_descales;

        int jj = 0;
        for (; jj + 1 < max_jj; jj += 2)
        {
            float fsum00 = 0.f;
            float fsum01 = 0.f;
            float fsum10 = 0.f;
            float fsum11 = 0.f;

            const signed char* pA0 = pA0_block;
            const signed char* pA1 = pA1_block;
            const float* pA_descales0 = pA_descales0_block;
            const float* pA_descales1 = pA_descales1_block;
            for (int k = 0; k < K; k += block_size)
            {
                int sum00 = 0;
                int sum01 = 0;
                int sum10 = 0;
                int sum11 = 0;
                const int max_kk = std::min(K - k, block_size);
                int kk = 0;
                for (; kk + 1 < max_kk; kk += 2)
                {
                    const int b00 = pB[0];
                    const int b01 = pB[1];
                    const int b10 = pB[2];
                    const int b11 = pB[3];
                    sum00 += pA0[kk] * b00 + pA0[kk + 1] * b01;
                    sum01 += pA0[kk] * b10 + pA0[kk + 1] * b11;
                    sum10 += pA1[kk] * b00 + pA1[kk + 1] * b01;
                    sum11 += pA1[kk] * b10 + pA1[kk + 1] * b11;
                    pB += 4;
                }
                if (kk < max_kk)
                {
                    const int b0 = pB[0];
                    const int b1 = pB[1];
                    sum00 += pA0[kk] * b0;
                    sum01 += pA0[kk] * b1;
                    sum10 += pA1[kk] * b0;
                    sum11 += pA1[kk] * b1;
                    pB += 2;
                }

                const float bd0 = pB_descales[0];
                const float bd1 = pB_descales[1];
                const float ad0 = pA_descales0[0];
                fsum00 += sum00 * ad0 * bd0;
                fsum01 += sum01 * ad0 * bd1;
                const float ad1 = pA_descales1[0];
                fsum10 += sum10 * ad1 * bd0;
                fsum11 += sum11 * ad1 * bd1;

                pA0 += max_kk;
                pA1 += max_kk;
                pA_descales0++;
                pA_descales1++;
                pB_descales += 2;
            }

            outptr[0] = fsum00;
            outptr++;
            outptr[0] = fsum01;
            outptr++;
            outptr[0] = fsum10;
            outptr++;
            outptr[0] = fsum11;
            outptr++;
        }
        for (; jj < max_jj; jj++)
        {
            float fsum00 = 0.f;
            float fsum10 = 0.f;

            const signed char* pA0 = pA0_block;
            const signed char* pA1 = pA1_block;
            const float* pA_descales0 = pA_descales0_block;
            const float* pA_descales1 = pA_descales1_block;
            for (int k = 0; k < K; k += block_size)
            {
                int sum00 = 0;
                int sum10 = 0;
                const int max_kk = std::min(K - k, block_size);
                int kk = 0;
                for (; kk + 1 < max_kk; kk += 2)
                {
                    const int b00 = pB[0];
                    const int b01 = pB[1];
                    sum00 += pA0[kk] * b00 + pA0[kk + 1] * b01;
                    sum10 += pA1[kk] * b00 + pA1[kk + 1] * b01;
                    pB += 2;
                }
                if (kk < max_kk)
                {
                    const int b0 = pB[0];
                    sum00 += pA0[kk] * b0;
                    sum10 += pA1[kk] * b0;
                    pB += 1;
                }

                const float bd0 = pB_descales[0];
                const float ad0 = pA_descales0[0];
                fsum00 += sum00 * ad0 * bd0;
                const float ad1 = pA_descales1[0];
                fsum10 += sum10 * ad1 * bd0;

                pA0 += max_kk;
                pA1 += max_kk;
                pA_descales0++;
                pA_descales1++;
                pB_descales += 1;
            }

            outptr[0] = fsum00;
            outptr++;
            outptr[0] = fsum10;
            outptr++;
        }
    }
#endif // __ARM_FEATURE_SIMD32 && NCNN_GNU_INLINE_ASM
    for (; ii < max_ii; ii++)
    {
        const signed char* pA0_block = pAT + ii * A_hstep;
        const float* pA_descales0_block = pAT_descales + ii * A_descales_hstep;
        const signed char* pB = pBT;
        const float* pB_descales = pBT_descales;

        int jj = 0;
        for (; jj + 1 < max_jj; jj += 2)
        {
            float fsum00 = 0.f;
            float fsum01 = 0.f;

            const signed char* pA0 = pA0_block;
            const float* pA_descales0 = pA_descales0_block;
            for (int k = 0; k < K; k += block_size)
            {
                int sum00 = 0;
                int sum01 = 0;
                const int max_kk = std::min(K - k, block_size);
                int kk = 0;
                for (; kk + 1 < max_kk; kk += 2)
                {
                    const int b00 = pB[0];
#if __ARM_FEATURE_SIMD32 && NCNN_GNU_INLINE_ASM
                    const int b01 = pB[2];
                    const int b10 = pB[1];
#else
                    const int b01 = pB[1];
                    const int b10 = pB[2];
#endif
                    const int b11 = pB[3];
                    sum00 += pA0[kk] * b00 + pA0[kk + 1] * b01;
                    sum01 += pA0[kk] * b10 + pA0[kk + 1] * b11;
                    pB += 4;
                }
                if (kk < max_kk)
                {
                    const int b0 = pB[0];
                    const int b1 = pB[1];
                    sum00 += pA0[kk] * b0;
                    sum01 += pA0[kk] * b1;
                    pB += 2;
                }

                const float bd0 = pB_descales[0];
                const float bd1 = pB_descales[1];
                const float ad0 = pA_descales0[0];
                fsum00 += sum00 * ad0 * bd0;
                fsum01 += sum01 * ad0 * bd1;

                pA0 += max_kk;
                pA_descales0++;
                pB_descales += 2;
            }

            outptr[0] = fsum00;
            outptr++;
            outptr[0] = fsum01;
            outptr++;
        }
        for (; jj < max_jj; jj++)
        {
            float fsum00 = 0.f;

            const signed char* pA0 = pA0_block;
            const float* pA_descales0 = pA_descales0_block;
            for (int k = 0; k < K; k += block_size)
            {
                int sum00 = 0;
                const int max_kk = std::min(K - k, block_size);
                int kk = 0;
                for (; kk + 1 < max_kk; kk += 2)
                {
                    const int b00 = pB[0];
                    const int b01 = pB[1];
                    sum00 += pA0[kk] * b00 + pA0[kk + 1] * b01;
                    pB += 2;
                }
                if (kk < max_kk)
                {
                    const int b0 = pB[0];
                    sum00 += pA0[kk] * b0;
                    pB += 1;
                }

                const float bd0 = pB_descales[0];
                const float ad0 = pA_descales0[0];
                fsum00 += sum00 * ad0 * bd0;

                pA0 += max_kk;
                pA_descales0++;
                pB_descales += 1;
            }

            outptr[0] = fsum00;
            outptr++;
        }
    }
#endif // __ARM_NEON
}

static void get_optimal_tile_mnk_wq_int8(int M, int N, int K, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const size_t l2_cache_size = get_cpu_level2_cache_size();

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    const int tile_size = std::max(1, (int)((float)l2_cache_size / 2 / sizeof(signed char) / std::max(1, K)));

#if __aarch64__
    TILE_M = M >= nT * 8 ? 8 : M >= nT * 4 ? 4 : M >= nT * 2 ? 2 : 1;
    TILE_N = std::max(8, tile_size / 8 * 8);
#elif __ARM_NEON
    TILE_M = M >= nT * 4 ? 4 : M >= nT * 2 ? 2 : 1;
    TILE_N = std::max(4, tile_size / 4 * 4);
#else
    TILE_M = M >= nT * 2 ? 2 : 1;
    TILE_N = std::max(2, tile_size / 2 * 2);
#endif
    TILE_K = K;

    if (N > 0)
    {
        const int nn_N = (N + TILE_N - 1) / TILE_N;
#if __aarch64__
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 7) / 8 * 8);
#elif __ARM_NEON
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4);
#else
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 1) / 2 * 2);
#endif
    }

    // always take constant TILE_M/N value when provided
    if (constant_TILE_M > 0)
    {
#if __aarch64__
        TILE_M = (constant_TILE_M + 7) / 8 * 8;
#elif __ARM_NEON
        TILE_M = (constant_TILE_M + 3) / 4 * 4;
#else
        TILE_M = (constant_TILE_M + 1) / 2 * 2;
#endif
    }

    if (constant_TILE_N > 0)
    {
#if __aarch64__
        TILE_N = (constant_TILE_N + 7) / 8 * 8;
#elif __ARM_NEON
        TILE_N = (constant_TILE_N + 3) / 4 * 4;
#else
        TILE_N = (constant_TILE_N + 1) / 2 * 2;
#endif
    }

    // one driver M tile follows the natural producer slab
#if __aarch64__
    TILE_M = std::min(TILE_M, 8);
#elif __ARM_NEON
    TILE_M = std::min(TILE_M, 4);
#else
    TILE_M = std::min(TILE_M, 2);
#endif

    (void)constant_TILE_K;
}

static void unpack_output_tile_wq_int8(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, int N, float alpha, float beta)
{
    const float* pC = C;
    const float* pp = topT;
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)N;
    const size_t c_hstep = C.dims == 3 ? C.cstep : (size_t)C.w;

    int ii = 0;
#if __ARM_NEON
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        float* outptr0 = (float*)top_blob + (size_t)(i + ii) * out_hstep + j;
        float* outptr1 = outptr0 + out_hstep;
        float* outptr2 = outptr1 + out_hstep;
        float* outptr3 = outptr2 + out_hstep;
        float* outptr4 = outptr3 + out_hstep;
        float* outptr5 = outptr4 + out_hstep;
        float* outptr6 = outptr5 + out_hstep;
        float* outptr7 = outptr6 + out_hstep;
        pC = (const float*)C;
        float c0 = 0.f;
        float c1 = 0.f;
        float c2 = 0.f;
        float c3 = 0.f;
        float c4 = 0.f;
        float c5 = 0.f;
        float c6 = 0.f;
        float c7 = 0.f;
        if (pC)
        {
            if (broadcast_type_C == 0)
                c0 = c1 = c2 = c3 = c4 = c5 = c6 = c7 = pC[0] * beta;
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
                c0 = pC[0] * beta;
                c1 = pC[1] * beta;
                c2 = pC[2] * beta;
                c3 = pC[3] * beta;
                c4 = pC[4] * beta;
                c5 = pC[5] * beta;
                c6 = pC[6] * beta;
                c7 = pC[7] * beta;
            }
            if (broadcast_type_C == 3)
                pC = (const float*)C + (size_t)(i + ii) * c_hstep + j;
            if (broadcast_type_C == 4)
                pC = (const float*)C + j;
        }

        int jj = 0;
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _out0 = vld1q_f32(pp);
            float32x4_t _out1 = vld1q_f32(pp + 4);
            float32x4_t _out2 = vld1q_f32(pp + 8);
            float32x4_t _out3 = vld1q_f32(pp + 12);
            float32x4_t _out4 = vld1q_f32(pp + 16);
            float32x4_t _out5 = vld1q_f32(pp + 20);
            float32x4_t _out6 = vld1q_f32(pp + 24);
            float32x4_t _out7 = vld1q_f32(pp + 28);
            if (pC)
            {
                if (broadcast_type_C <= 2)
                {
                    _out0 = vaddq_f32(_out0, vdupq_n_f32(c0));
                    _out1 = vaddq_f32(_out1, vdupq_n_f32(c1));
                    _out2 = vaddq_f32(_out2, vdupq_n_f32(c2));
                    _out3 = vaddq_f32(_out3, vdupq_n_f32(c3));
                    _out4 = vaddq_f32(_out4, vdupq_n_f32(c4));
                    _out5 = vaddq_f32(_out5, vdupq_n_f32(c5));
                    _out6 = vaddq_f32(_out6, vdupq_n_f32(c6));
                    _out7 = vaddq_f32(_out7, vdupq_n_f32(c7));
                }
                if (broadcast_type_C == 3)
                {
                    _out0 = beta == 1.f ? vaddq_f32(_out0, vld1q_f32(pC)) : vmlaq_n_f32(_out0, vld1q_f32(pC), beta);
                    _out1 = beta == 1.f ? vaddq_f32(_out1, vld1q_f32(pC + c_hstep)) : vmlaq_n_f32(_out1, vld1q_f32(pC + c_hstep), beta);
                    _out2 = beta == 1.f ? vaddq_f32(_out2, vld1q_f32(pC + c_hstep * 2)) : vmlaq_n_f32(_out2, vld1q_f32(pC + c_hstep * 2), beta);
                    _out3 = beta == 1.f ? vaddq_f32(_out3, vld1q_f32(pC + c_hstep * 3)) : vmlaq_n_f32(_out3, vld1q_f32(pC + c_hstep * 3), beta);
                    _out4 = beta == 1.f ? vaddq_f32(_out4, vld1q_f32(pC + c_hstep * 4)) : vmlaq_n_f32(_out4, vld1q_f32(pC + c_hstep * 4), beta);
                    _out5 = beta == 1.f ? vaddq_f32(_out5, vld1q_f32(pC + c_hstep * 5)) : vmlaq_n_f32(_out5, vld1q_f32(pC + c_hstep * 5), beta);
                    _out6 = beta == 1.f ? vaddq_f32(_out6, vld1q_f32(pC + c_hstep * 6)) : vmlaq_n_f32(_out6, vld1q_f32(pC + c_hstep * 6), beta);
                    _out7 = beta == 1.f ? vaddq_f32(_out7, vld1q_f32(pC + c_hstep * 7)) : vmlaq_n_f32(_out7, vld1q_f32(pC + c_hstep * 7), beta);
                    pC += 4;
                }
                if (broadcast_type_C == 4)
                {
                    const float32x4_t _c = beta == 1.f ? vld1q_f32(pC) : vmulq_n_f32(vld1q_f32(pC), beta);
                    _out0 = vaddq_f32(_out0, _c);
                    _out1 = vaddq_f32(_out1, _c);
                    _out2 = vaddq_f32(_out2, _c);
                    _out3 = vaddq_f32(_out3, _c);
                    _out4 = vaddq_f32(_out4, _c);
                    _out5 = vaddq_f32(_out5, _c);
                    _out6 = vaddq_f32(_out6, _c);
                    _out7 = vaddq_f32(_out7, _c);
                    pC += 4;
                }
            }
            if (alpha != 1.f)
            {
                _out0 = vmulq_n_f32(_out0, alpha);
                _out1 = vmulq_n_f32(_out1, alpha);
                _out2 = vmulq_n_f32(_out2, alpha);
                _out3 = vmulq_n_f32(_out3, alpha);
                _out4 = vmulq_n_f32(_out4, alpha);
                _out5 = vmulq_n_f32(_out5, alpha);
                _out6 = vmulq_n_f32(_out6, alpha);
                _out7 = vmulq_n_f32(_out7, alpha);
            }
            vst1q_f32(outptr0, _out0);
            vst1q_f32(outptr1, _out1);
            vst1q_f32(outptr2, _out2);
            vst1q_f32(outptr3, _out3);
            vst1q_f32(outptr4, _out4);
            vst1q_f32(outptr5, _out5);
            vst1q_f32(outptr6, _out6);
            vst1q_f32(outptr7, _out7);
            outptr0 += 4;
            outptr1 += 4;
            outptr2 += 4;
            outptr3 += 4;
            outptr4 += 4;
            outptr5 += 4;
            outptr6 += 4;
            outptr7 += 4;
            pp += 32;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float32x2_t _out0 = vld1_f32(pp);
            float32x2_t _out1 = vld1_f32(pp + 2);
            float32x2_t _out2 = vld1_f32(pp + 4);
            float32x2_t _out3 = vld1_f32(pp + 6);
            float32x2_t _out4 = vld1_f32(pp + 8);
            float32x2_t _out5 = vld1_f32(pp + 10);
            float32x2_t _out6 = vld1_f32(pp + 12);
            float32x2_t _out7 = vld1_f32(pp + 14);
            if (pC)
            {
                if (broadcast_type_C <= 2)
                {
                    _out0 = vadd_f32(_out0, vdup_n_f32(c0));
                    _out1 = vadd_f32(_out1, vdup_n_f32(c1));
                    _out2 = vadd_f32(_out2, vdup_n_f32(c2));
                    _out3 = vadd_f32(_out3, vdup_n_f32(c3));
                    _out4 = vadd_f32(_out4, vdup_n_f32(c4));
                    _out5 = vadd_f32(_out5, vdup_n_f32(c5));
                    _out6 = vadd_f32(_out6, vdup_n_f32(c6));
                    _out7 = vadd_f32(_out7, vdup_n_f32(c7));
                }
                if (broadcast_type_C == 3)
                {
                    _out0 = beta == 1.f ? vadd_f32(_out0, vld1_f32(pC)) : vmla_n_f32(_out0, vld1_f32(pC), beta);
                    _out1 = beta == 1.f ? vadd_f32(_out1, vld1_f32(pC + c_hstep)) : vmla_n_f32(_out1, vld1_f32(pC + c_hstep), beta);
                    _out2 = beta == 1.f ? vadd_f32(_out2, vld1_f32(pC + c_hstep * 2)) : vmla_n_f32(_out2, vld1_f32(pC + c_hstep * 2), beta);
                    _out3 = beta == 1.f ? vadd_f32(_out3, vld1_f32(pC + c_hstep * 3)) : vmla_n_f32(_out3, vld1_f32(pC + c_hstep * 3), beta);
                    _out4 = beta == 1.f ? vadd_f32(_out4, vld1_f32(pC + c_hstep * 4)) : vmla_n_f32(_out4, vld1_f32(pC + c_hstep * 4), beta);
                    _out5 = beta == 1.f ? vadd_f32(_out5, vld1_f32(pC + c_hstep * 5)) : vmla_n_f32(_out5, vld1_f32(pC + c_hstep * 5), beta);
                    _out6 = beta == 1.f ? vadd_f32(_out6, vld1_f32(pC + c_hstep * 6)) : vmla_n_f32(_out6, vld1_f32(pC + c_hstep * 6), beta);
                    _out7 = beta == 1.f ? vadd_f32(_out7, vld1_f32(pC + c_hstep * 7)) : vmla_n_f32(_out7, vld1_f32(pC + c_hstep * 7), beta);
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    const float32x2_t _c = beta == 1.f ? vld1_f32(pC) : vmul_n_f32(vld1_f32(pC), beta);
                    _out0 = vadd_f32(_out0, _c);
                    _out1 = vadd_f32(_out1, _c);
                    _out2 = vadd_f32(_out2, _c);
                    _out3 = vadd_f32(_out3, _c);
                    _out4 = vadd_f32(_out4, _c);
                    _out5 = vadd_f32(_out5, _c);
                    _out6 = vadd_f32(_out6, _c);
                    _out7 = vadd_f32(_out7, _c);
                    pC += 2;
                }
            }
            if (alpha != 1.f)
            {
                _out0 = vmul_n_f32(_out0, alpha);
                _out1 = vmul_n_f32(_out1, alpha);
                _out2 = vmul_n_f32(_out2, alpha);
                _out3 = vmul_n_f32(_out3, alpha);
                _out4 = vmul_n_f32(_out4, alpha);
                _out5 = vmul_n_f32(_out5, alpha);
                _out6 = vmul_n_f32(_out6, alpha);
                _out7 = vmul_n_f32(_out7, alpha);
            }
            vst1_f32(outptr0, _out0);
            vst1_f32(outptr1, _out1);
            vst1_f32(outptr2, _out2);
            vst1_f32(outptr3, _out3);
            vst1_f32(outptr4, _out4);
            vst1_f32(outptr5, _out5);
            vst1_f32(outptr6, _out6);
            vst1_f32(outptr7, _out7);
            outptr0 += 2;
            outptr1 += 2;
            outptr2 += 2;
            outptr3 += 2;
            outptr4 += 2;
            outptr5 += 2;
            outptr6 += 2;
            outptr7 += 2;
            pp += 16;
        }
        for (; jj < max_jj; jj++)
        {
            float f0 = pp[0];
            float f1 = pp[1];
            float f2 = pp[2];
            float f3 = pp[3];
            float f4 = pp[4];
            float f5 = pp[5];
            float f6 = pp[6];
            float f7 = pp[7];
            if (pC)
            {
                if (broadcast_type_C <= 2)
                {
                    f0 += c0;
                    f1 += c1;
                    f2 += c2;
                    f3 += c3;
                    f4 += c4;
                    f5 += c5;
                    f6 += c6;
                    f7 += c7;
                }
                if (broadcast_type_C == 3)
                {
                    f0 += pC[0] * beta;
                    f1 += pC[c_hstep] * beta;
                    f2 += pC[c_hstep * 2] * beta;
                    f3 += pC[c_hstep * 3] * beta;
                    f4 += pC[c_hstep * 4] * beta;
                    f5 += pC[c_hstep * 5] * beta;
                    f6 += pC[c_hstep * 6] * beta;
                    f7 += pC[c_hstep * 7] * beta;
                    pC++;
                }
                if (broadcast_type_C == 4)
                {
                    const float c = pC[0] * beta;
                    f0 += c;
                    f1 += c;
                    f2 += c;
                    f3 += c;
                    f4 += c;
                    f5 += c;
                    f6 += c;
                    f7 += c;
                    pC++;
                }
            }
            outptr0[0] = f0 * alpha;
            outptr1[0] = f1 * alpha;
            outptr2[0] = f2 * alpha;
            outptr3[0] = f3 * alpha;
            outptr4[0] = f4 * alpha;
            outptr5[0] = f5 * alpha;
            outptr6[0] = f6 * alpha;
            outptr7[0] = f7 * alpha;
            outptr0++;
            outptr1++;
            outptr2++;
            outptr3++;
            outptr4++;
            outptr5++;
            outptr6++;
            outptr7++;
            pp += 8;
        }
    }
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        pC = (const float*)C;
        float* outptr0 = (float*)top_blob + (size_t)(i + ii) * out_hstep + j;
        float* outptr1 = outptr0 + out_hstep;
        float* outptr2 = outptr1 + out_hstep;
        float* outptr3 = outptr2 + out_hstep;
        float c0 = 0.f;
        float c1 = 0.f;
        float c2 = 0.f;
        float c3 = 0.f;
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                float c = pC[0];
                if (beta != 1.f)
                    c *= beta;
                c0 = c;
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                c0 = pC[i + ii];
                if (beta != 1.f)
                    c0 *= beta;
                c1 = pC[i + ii + 1];
                if (beta != 1.f)
                    c1 *= beta;
                c2 = pC[i + ii + 2];
                if (beta != 1.f)
                    c2 *= beta;
                c3 = pC[i + ii + 3];
                if (beta != 1.f)
                    c3 *= beta;
            }
            if (broadcast_type_C == 3)
                pC = (const float*)C + (i + ii) * c_hstep + j;
            if (broadcast_type_C == 4)
                pC = (const float*)C + j;
        }

        int jj = 0;
#if __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            float32x4_t _out0 = vld1q_f32(pp + 0);
            float32x4_t _out1 = vld1q_f32(pp + 4);
            float32x4_t _out2 = vld1q_f32(pp + 8);
            float32x4_t _out3 = vld1q_f32(pp + 12);
            float32x4_t _out4 = vld1q_f32(pp + 16);
            float32x4_t _out5 = vld1q_f32(pp + 20);
            float32x4_t _out6 = vld1q_f32(pp + 24);
            float32x4_t _out7 = vld1q_f32(pp + 28);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    const float32x4_t _c = vdupq_n_f32(c0);
                    _out0 = vaddq_f32(_out0, _c);
                    _out1 = vaddq_f32(_out1, _c);
                    _out2 = vaddq_f32(_out2, _c);
                    _out3 = vaddq_f32(_out3, _c);
                    _out4 = vaddq_f32(_out4, _c);
                    _out5 = vaddq_f32(_out5, _c);
                    _out6 = vaddq_f32(_out6, _c);
                    _out7 = vaddq_f32(_out7, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _out0 = vaddq_f32(_out0, vdupq_n_f32(c0));
                    _out1 = vaddq_f32(_out1, vdupq_n_f32(c0));
                    _out2 = vaddq_f32(_out2, vdupq_n_f32(c1));
                    _out3 = vaddq_f32(_out3, vdupq_n_f32(c1));
                    _out4 = vaddq_f32(_out4, vdupq_n_f32(c2));
                    _out5 = vaddq_f32(_out5, vdupq_n_f32(c2));
                    _out6 = vaddq_f32(_out6, vdupq_n_f32(c3));
                    _out7 = vaddq_f32(_out7, vdupq_n_f32(c3));
                }
                if (broadcast_type_C == 3)
                {
                    _out0 = vaddq_f32(_out0, (beta == 1.f ? vld1q_f32(pC) : vmulq_n_f32(vld1q_f32(pC), beta)));
                    _out1 = vaddq_f32(_out1, (beta == 1.f ? vld1q_f32(pC + 4) : vmulq_n_f32(vld1q_f32(pC + 4), beta)));
                    _out2 = vaddq_f32(_out2, (beta == 1.f ? vld1q_f32(pC + c_hstep) : vmulq_n_f32(vld1q_f32(pC + c_hstep), beta)));
                    _out3 = vaddq_f32(_out3, (beta == 1.f ? vld1q_f32(pC + c_hstep + 4) : vmulq_n_f32(vld1q_f32(pC + c_hstep + 4), beta)));
                    _out4 = vaddq_f32(_out4, (beta == 1.f ? vld1q_f32(pC + c_hstep * 2) : vmulq_n_f32(vld1q_f32(pC + c_hstep * 2), beta)));
                    _out5 = vaddq_f32(_out5, (beta == 1.f ? vld1q_f32(pC + c_hstep * 2 + 4) : vmulq_n_f32(vld1q_f32(pC + c_hstep * 2 + 4), beta)));
                    _out6 = vaddq_f32(_out6, (beta == 1.f ? vld1q_f32(pC + c_hstep * 3) : vmulq_n_f32(vld1q_f32(pC + c_hstep * 3), beta)));
                    _out7 = vaddq_f32(_out7, (beta == 1.f ? vld1q_f32(pC + c_hstep * 3 + 4) : vmulq_n_f32(vld1q_f32(pC + c_hstep * 3 + 4), beta)));
                }
                if (broadcast_type_C == 4)
                {
                    const float32x4_t _cc0 = (beta == 1.f ? vld1q_f32(pC) : vmulq_n_f32(vld1q_f32(pC), beta));
                    _out0 = vaddq_f32(_out0, _cc0);
                    _out2 = vaddq_f32(_out2, _cc0);
                    _out4 = vaddq_f32(_out4, _cc0);
                    _out6 = vaddq_f32(_out6, _cc0);
                    const float32x4_t _cc1 = (beta == 1.f ? vld1q_f32(pC + 4) : vmulq_n_f32(vld1q_f32(pC + 4), beta));
                    _out1 = vaddq_f32(_out1, _cc1);
                    _out3 = vaddq_f32(_out3, _cc1);
                    _out5 = vaddq_f32(_out5, _cc1);
                    _out7 = vaddq_f32(_out7, _cc1);
                }
            }

            if (alpha != 1.f)
            {
                _out0 = vmulq_n_f32(_out0, alpha);
                _out1 = vmulq_n_f32(_out1, alpha);
                _out2 = vmulq_n_f32(_out2, alpha);
                _out3 = vmulq_n_f32(_out3, alpha);
                _out4 = vmulq_n_f32(_out4, alpha);
                _out5 = vmulq_n_f32(_out5, alpha);
                _out6 = vmulq_n_f32(_out6, alpha);
                _out7 = vmulq_n_f32(_out7, alpha);
            }

            vst1q_f32(outptr0, _out0);
            vst1q_f32(outptr0 + 4, _out1);
            vst1q_f32(outptr1, _out2);
            vst1q_f32(outptr1 + 4, _out3);
            vst1q_f32(outptr2, _out4);
            vst1q_f32(outptr2 + 4, _out5);
            vst1q_f32(outptr3, _out6);
            vst1q_f32(outptr3 + 4, _out7);

            outptr0 += 8;
            outptr1 += 8;
            outptr2 += 8;
            outptr3 += 8;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 8;
            pp += 32;
        }
#endif // __aarch64__
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _out0 = vld1q_f32(pp + 0);
            float32x4_t _out1 = vld1q_f32(pp + 4);
            float32x4_t _out2 = vld1q_f32(pp + 8);
            float32x4_t _out3 = vld1q_f32(pp + 12);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    const float32x4_t _c = vdupq_n_f32(c0);
                    _out0 = vaddq_f32(_out0, _c);
                    _out1 = vaddq_f32(_out1, _c);
                    _out2 = vaddq_f32(_out2, _c);
                    _out3 = vaddq_f32(_out3, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _out0 = vaddq_f32(_out0, vdupq_n_f32(c0));
                    _out1 = vaddq_f32(_out1, vdupq_n_f32(c1));
                    _out2 = vaddq_f32(_out2, vdupq_n_f32(c2));
                    _out3 = vaddq_f32(_out3, vdupq_n_f32(c3));
                }
                if (broadcast_type_C == 3)
                {
                    _out0 = vaddq_f32(_out0, (beta == 1.f ? vld1q_f32(pC) : vmulq_n_f32(vld1q_f32(pC), beta)));
                    _out1 = vaddq_f32(_out1, (beta == 1.f ? vld1q_f32(pC + c_hstep) : vmulq_n_f32(vld1q_f32(pC + c_hstep), beta)));
                    _out2 = vaddq_f32(_out2, (beta == 1.f ? vld1q_f32(pC + c_hstep * 2) : vmulq_n_f32(vld1q_f32(pC + c_hstep * 2), beta)));
                    _out3 = vaddq_f32(_out3, (beta == 1.f ? vld1q_f32(pC + c_hstep * 3) : vmulq_n_f32(vld1q_f32(pC + c_hstep * 3), beta)));
                }
                if (broadcast_type_C == 4)
                {
                    const float32x4_t _cc0 = (beta == 1.f ? vld1q_f32(pC) : vmulq_n_f32(vld1q_f32(pC), beta));
                    _out0 = vaddq_f32(_out0, _cc0);
                    _out1 = vaddq_f32(_out1, _cc0);
                    _out2 = vaddq_f32(_out2, _cc0);
                    _out3 = vaddq_f32(_out3, _cc0);
                }
            }

            if (alpha != 1.f)
            {
                _out0 = vmulq_n_f32(_out0, alpha);
                _out1 = vmulq_n_f32(_out1, alpha);
                _out2 = vmulq_n_f32(_out2, alpha);
                _out3 = vmulq_n_f32(_out3, alpha);
            }

            vst1q_f32(outptr0, _out0);
            vst1q_f32(outptr1, _out1);
            vst1q_f32(outptr2, _out2);
            vst1q_f32(outptr3, _out3);

            outptr0 += 4;
            outptr1 += 4;
            outptr2 += 4;
            outptr3 += 4;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 4;
            pp += 16;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float32x4_t _out0 = vld1q_f32(pp);
            float32x4_t _out1 = vld1q_f32(pp + 4);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    const float32x4_t _c = vdupq_n_f32(c0);
                    _out0 = vaddq_f32(_out0, _c);
                    _out1 = vaddq_f32(_out1, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    const float32x4_t _c01 = vcombine_f32(vdup_n_f32(c0), vdup_n_f32(c1));
                    const float32x4_t _c23 = vcombine_f32(vdup_n_f32(c2), vdup_n_f32(c3));
                    _out0 = vaddq_f32(_out0, _c01);
                    _out1 = vaddq_f32(_out1, _c23);
                }
                if (broadcast_type_C == 3)
                {
                    const float32x4_t _c01 = vcombine_f32(vld1_f32(pC), vld1_f32(pC + c_hstep));
                    const float32x4_t _c23 = vcombine_f32(vld1_f32(pC + c_hstep * 2), vld1_f32(pC + c_hstep * 3));
                    _out0 = beta == 1.f ? vaddq_f32(_out0, _c01) : vmlaq_n_f32(_out0, _c01, beta);
                    _out1 = beta == 1.f ? vaddq_f32(_out1, _c23) : vmlaq_n_f32(_out1, _c23, beta);
                }
                if (broadcast_type_C == 4)
                {
                    float32x2_t _c = vld1_f32(pC);
                    if (beta != 1.f)
                        _c = vmul_n_f32(_c, beta);
                    const float32x4_t _cc0 = vcombine_f32(_c, _c);
                    _out0 = vaddq_f32(_out0, _cc0);
                    _out1 = vaddq_f32(_out1, _cc0);
                }
            }

            if (alpha != 1.f)
            {
                _out0 = vmulq_n_f32(_out0, alpha);
                _out1 = vmulq_n_f32(_out1, alpha);
            }

            vst1_f32(outptr0, vget_low_f32(_out0));
            vst1_f32(outptr1, vget_high_f32(_out0));
            vst1_f32(outptr2, vget_low_f32(_out1));
            vst1_f32(outptr3, vget_high_f32(_out1));

            outptr0 += 2;
            outptr1 += 2;
            outptr2 += 2;
            outptr3 += 2;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 2;
            pp += 8;
        }
        for (; jj < max_jj; jj += 1)
        {
            float32x4_t _out0 = vld1q_f32(pp);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _out0 = vaddq_f32(_out0, vdupq_n_f32(c0));
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    float32x4_t _c = vdupq_n_f32(c0);
                    _c = vsetq_lane_f32(c1, _c, 1);
                    _c = vsetq_lane_f32(c2, _c, 2);
                    _c = vsetq_lane_f32(c3, _c, 3);
                    _out0 = vaddq_f32(_out0, _c);
                }
                if (broadcast_type_C == 3)
                {
                    float32x4_t _c = vdupq_n_f32(pC[0]);
                    _c = vsetq_lane_f32(pC[c_hstep], _c, 1);
                    _c = vsetq_lane_f32(pC[c_hstep * 2], _c, 2);
                    _c = vsetq_lane_f32(pC[c_hstep * 3], _c, 3);
                    _out0 = beta == 1.f ? vaddq_f32(_out0, _c) : vmlaq_n_f32(_out0, _c, beta);
                }
                if (broadcast_type_C == 4)
                {
                    const float32x4_t _cc0 = vdupq_n_f32(beta == 1.f ? pC[0] : pC[0] * beta);
                    _out0 = vaddq_f32(_out0, _cc0);
                }
            }

            if (alpha != 1.f)
            {
                _out0 = vmulq_n_f32(_out0, alpha);
            }

            vst1q_lane_f32(outptr0, _out0, 0);
            vst1q_lane_f32(outptr1, _out0, 1);
            vst1q_lane_f32(outptr2, _out0, 2);
            vst1q_lane_f32(outptr3, _out0, 3);

            outptr0++;
            outptr1++;
            outptr2++;
            outptr3++;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC++;
            pp += 4;
        }
    }
    for (; ii + 1 < max_ii; ii += 2)
    {
        pC = (const float*)C;
        float* outptr0 = (float*)top_blob + (size_t)(i + ii) * out_hstep + j;
        float* outptr1 = outptr0 + out_hstep;
        float32x4_t _c0;
        float32x4_t _c1;
        float c0 = 0.f;
        float c1 = 0.f;
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                c0 = pC[0];
                if (beta != 1.f)
                    c0 *= beta;
                _c0 = vdupq_n_f32(c0);
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                c0 = pC[i + ii];
                if (beta != 1.f)
                    c0 *= beta;
                _c0 = vdupq_n_f32(c0);
                c1 = pC[i + ii + 1];
                if (beta != 1.f)
                    c1 *= beta;
                _c1 = vdupq_n_f32(c1);
            }
            if (broadcast_type_C == 3)
                pC = (const float*)C + (i + ii) * c_hstep + j;
            if (broadcast_type_C == 4)
                pC = (const float*)C + j;
        }

        int jj = 0;
#if __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            float32x4_t _out0 = vld1q_f32(pp + 0);
            float32x4_t _out1 = vld1q_f32(pp + 4);
            float32x4_t _out2 = vld1q_f32(pp + 8);
            float32x4_t _out3 = vld1q_f32(pp + 12);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _out0 = vaddq_f32(_out0, _c0);
                    _out1 = vaddq_f32(_out1, _c0);
                    _out2 = vaddq_f32(_out2, _c0);
                    _out3 = vaddq_f32(_out3, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _out0 = vaddq_f32(_out0, _c0);
                    _out1 = vaddq_f32(_out1, _c0);
                    _out2 = vaddq_f32(_out2, _c1);
                    _out3 = vaddq_f32(_out3, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    _out0 = vaddq_f32(_out0, (beta == 1.f ? vld1q_f32(pC) : vmulq_n_f32(vld1q_f32(pC), beta)));
                    _out1 = vaddq_f32(_out1, (beta == 1.f ? vld1q_f32(pC + 4) : vmulq_n_f32(vld1q_f32(pC + 4), beta)));
                    _out2 = vaddq_f32(_out2, (beta == 1.f ? vld1q_f32(pC + c_hstep) : vmulq_n_f32(vld1q_f32(pC + c_hstep), beta)));
                    _out3 = vaddq_f32(_out3, (beta == 1.f ? vld1q_f32(pC + c_hstep + 4) : vmulq_n_f32(vld1q_f32(pC + c_hstep + 4), beta)));
                }
                if (broadcast_type_C == 4)
                {
                    const float32x4_t _c0 = (beta == 1.f ? vld1q_f32(pC) : vmulq_n_f32(vld1q_f32(pC), beta));
                    _out0 = vaddq_f32(_out0, _c0);
                    _out2 = vaddq_f32(_out2, _c0);
                    const float32x4_t _c1 = (beta == 1.f ? vld1q_f32(pC + 4) : vmulq_n_f32(vld1q_f32(pC + 4), beta));
                    _out1 = vaddq_f32(_out1, _c1);
                    _out3 = vaddq_f32(_out3, _c1);
                }
            }

            if (alpha != 1.f)
            {
                _out0 = vmulq_n_f32(_out0, alpha);
                _out1 = vmulq_n_f32(_out1, alpha);
                _out2 = vmulq_n_f32(_out2, alpha);
                _out3 = vmulq_n_f32(_out3, alpha);
            }

            vst1q_f32(outptr0, _out0);
            vst1q_f32(outptr0 + 4, _out1);
            vst1q_f32(outptr1, _out2);
            vst1q_f32(outptr1 + 4, _out3);

            outptr0 += 8;
            outptr1 += 8;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 8;
            pp += 16;
        }
#endif // __aarch64__
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _out0 = vld1q_f32(pp + 0);
            float32x4_t _out1 = vld1q_f32(pp + 4);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _out0 = vaddq_f32(_out0, _c0);
                    _out1 = vaddq_f32(_out1, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _out0 = vaddq_f32(_out0, _c0);
                    _out1 = vaddq_f32(_out1, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    _out0 = vaddq_f32(_out0, (beta == 1.f ? vld1q_f32(pC) : vmulq_n_f32(vld1q_f32(pC), beta)));
                    _out1 = vaddq_f32(_out1, (beta == 1.f ? vld1q_f32(pC + c_hstep) : vmulq_n_f32(vld1q_f32(pC + c_hstep), beta)));
                }
                if (broadcast_type_C == 4)
                {
                    const float32x4_t _c0 = (beta == 1.f ? vld1q_f32(pC) : vmulq_n_f32(vld1q_f32(pC), beta));
                    _out0 = vaddq_f32(_out0, _c0);
                    _out1 = vaddq_f32(_out1, _c0);
                }
            }

            if (alpha != 1.f)
            {
                _out0 = vmulq_n_f32(_out0, alpha);
                _out1 = vmulq_n_f32(_out1, alpha);
            }

            vst1q_f32(outptr0, _out0);
            vst1q_f32(outptr1, _out1);

            outptr0 += 4;
            outptr1 += 4;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 4;
            pp += 8;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float32x2_t _out0 = vld1_f32(pp);
            float32x2_t _out1 = vld1_f32(pp + 2);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    const float32x2_t _c = vdup_n_f32(c0);
                    _out0 = vadd_f32(_out0, _c);
                    _out1 = vadd_f32(_out1, _c);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _out0 = vadd_f32(_out0, vdup_n_f32(c0));
                    _out1 = vadd_f32(_out1, vdup_n_f32(c1));
                }
                if (broadcast_type_C == 3)
                {
                    const float32x2_t _c0 = vld1_f32(pC);
                    const float32x2_t _c1 = vld1_f32(pC + c_hstep);
                    _out0 = beta == 1.f ? vadd_f32(_out0, _c0) : vmla_n_f32(_out0, _c0, beta);
                    _out1 = beta == 1.f ? vadd_f32(_out1, _c1) : vmla_n_f32(_out1, _c1, beta);
                }
                if (broadcast_type_C == 4)
                {
                    const float32x2_t _c0 = beta == 1.f ? vld1_f32(pC) : vmul_n_f32(vld1_f32(pC), beta);
                    _out0 = vadd_f32(_out0, _c0);
                    _out1 = vadd_f32(_out1, _c0);
                }
            }

            if (alpha != 1.f)
            {
                _out0 = vmul_n_f32(_out0, alpha);
                _out1 = vmul_n_f32(_out1, alpha);
            }

            vst1_f32(outptr0, _out0);
            vst1_f32(outptr1, _out1);

            outptr0 += 2;
            outptr1 += 2;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 2;
            pp += 4;
        }
        for (; jj < max_jj; jj += 1)
        {
            float32x2_t _out0 = vld1_f32(pp);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _out0 = vadd_f32(_out0, vdup_n_f32(c0));
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    float32x2_t _c = vdup_n_f32(c0);
                    _c = vset_lane_f32(c1, _c, 1);
                    _out0 = vadd_f32(_out0, _c);
                }
                if (broadcast_type_C == 3)
                {
                    float32x2_t _c = vdup_n_f32(pC[0]);
                    _c = vset_lane_f32(pC[c_hstep], _c, 1);
                    _out0 = beta == 1.f ? vadd_f32(_out0, _c) : vmla_n_f32(_out0, _c, beta);
                }
                if (broadcast_type_C == 4)
                {
                    _out0 = vadd_f32(_out0, vdup_n_f32(beta == 1.f ? pC[0] : pC[0] * beta));
                }
            }

            if (alpha != 1.f)
            {
                _out0 = vmul_n_f32(_out0, alpha);
            }

            vst1_lane_f32(outptr0, _out0, 0);
            vst1_lane_f32(outptr1, _out0, 1);

            outptr0++;
            outptr1++;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC++;
            pp += 2;
        }
    }
    for (; ii < max_ii; ii++)
    {
        pC = (const float*)C;
        float* outptr0 = (float*)top_blob + (size_t)(i + ii) * out_hstep + j;
        float32x4_t _c0;
        float c0 = 0.f;
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                c0 = pC[0];
                if (beta != 1.f)
                    c0 *= beta;
                _c0 = vdupq_n_f32(c0);
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                c0 = pC[i + ii];
                if (beta != 1.f)
                    c0 *= beta;
                _c0 = vdupq_n_f32(c0);
            }
            if (broadcast_type_C == 3)
                pC = (const float*)C + (i + ii) * c_hstep + j;
            if (broadcast_type_C == 4)
                pC = (const float*)C + j;
        }

        int jj = 0;
#if __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            float32x4_t _out0 = vld1q_f32(pp + 0);
            float32x4_t _out1 = vld1q_f32(pp + 4);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _out0 = vaddq_f32(_out0, _c0);
                    _out1 = vaddq_f32(_out1, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _out0 = vaddq_f32(_out0, _c0);
                    _out1 = vaddq_f32(_out1, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    _out0 = vaddq_f32(_out0, (beta == 1.f ? vld1q_f32(pC) : vmulq_n_f32(vld1q_f32(pC), beta)));
                    _out1 = vaddq_f32(_out1, (beta == 1.f ? vld1q_f32(pC + 4) : vmulq_n_f32(vld1q_f32(pC + 4), beta)));
                }
                if (broadcast_type_C == 4)
                {
                    const float32x4_t _c0 = (beta == 1.f ? vld1q_f32(pC) : vmulq_n_f32(vld1q_f32(pC), beta));
                    _out0 = vaddq_f32(_out0, _c0);
                    const float32x4_t _c1 = (beta == 1.f ? vld1q_f32(pC + 4) : vmulq_n_f32(vld1q_f32(pC + 4), beta));
                    _out1 = vaddq_f32(_out1, _c1);
                }
            }

            if (alpha != 1.f)
            {
                _out0 = vmulq_n_f32(_out0, alpha);
                _out1 = vmulq_n_f32(_out1, alpha);
            }

            vst1q_f32(outptr0, _out0);
            vst1q_f32(outptr0 + 4, _out1);

            outptr0 += 8;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 8;
            pp += 8;
        }
#endif // __aarch64__
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _out0 = vld1q_f32(pp + 0);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _out0 = vaddq_f32(_out0, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _out0 = vaddq_f32(_out0, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    _out0 = vaddq_f32(_out0, (beta == 1.f ? vld1q_f32(pC) : vmulq_n_f32(vld1q_f32(pC), beta)));
                }
                if (broadcast_type_C == 4)
                {
                    const float32x4_t _c0 = (beta == 1.f ? vld1q_f32(pC) : vmulq_n_f32(vld1q_f32(pC), beta));
                    _out0 = vaddq_f32(_out0, _c0);
                }
            }

            if (alpha != 1.f)
            {
                _out0 = vmulq_n_f32(_out0, alpha);
            }

            vst1q_f32(outptr0, _out0);

            outptr0 += 4;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 4;
            pp += 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float32x2_t _out0 = vld1_f32(pp);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _out0 = vadd_f32(_out0, vdup_n_f32(c0));
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _out0 = vadd_f32(_out0, vdup_n_f32(c0));
                }
                if (broadcast_type_C == 3)
                {
                    const float32x2_t _c0 = vld1_f32(pC);
                    _out0 = beta == 1.f ? vadd_f32(_out0, _c0) : vmla_n_f32(_out0, _c0, beta);
                }
                if (broadcast_type_C == 4)
                {
                    const float32x2_t _c0 = beta == 1.f ? vld1_f32(pC) : vmul_n_f32(vld1_f32(pC), beta);
                    _out0 = vadd_f32(_out0, _c0);
                }
            }

            if (alpha != 1.f)
            {
                _out0 = vmul_n_f32(_out0, alpha);
            }

            vst1_f32(outptr0, _out0);

            outptr0 += 2;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 2;
            pp += 2;
        }
        for (; jj < max_jj; jj += 1)
        {
            float out0 = pp[0];

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    out0 += c0;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    out0 += c0;
                }
                if (broadcast_type_C == 3)
                {
                    out0 += beta == 1.f ? pC[0] : pC[0] * beta;
                }
                if (broadcast_type_C == 4)
                {
                    out0 += beta == 1.f ? pC[0] : pC[0] * beta;
                }
            }

            if (alpha != 1.f)
            {
                out0 *= alpha;
            }

            outptr0[0] = out0;

            outptr0++;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC++;
            pp += 1;
        }
    }
#else
    for (; ii + 1 < max_ii; ii += 2)
    {
        pC = (const float*)C;
        float* outptr0 = (float*)top_blob + (size_t)(i + ii) * out_hstep + j;
        float* outptr1 = outptr0 + out_hstep;
        float c0 = 0.f;
        float c1 = 0.f;
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                float c = pC[0];
                if (beta != 1.f)
                    c *= beta;
                c0 = c;
                c1 = c;
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                c0 = pC[i + ii];
                if (beta != 1.f)
                    c0 *= beta;
                c1 = pC[i + ii + 1];
                if (beta != 1.f)
                    c1 *= beta;
            }
            if (broadcast_type_C == 3)
                pC = (const float*)C + (i + ii) * c_hstep + j;
            if (broadcast_type_C == 4)
                pC = (const float*)C + j;
        }

        int jj = 0;
        for (; jj + 1 < max_jj; jj += 2)
        {
            float out00 = pp[0];
            float out01 = pp[1];
            float out10 = pp[2];
            float out11 = pp[3];

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    out00 += c0;
                    out01 += c0;
                    out10 += c0;
                    out11 += c0;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    out00 += c0;
                    out01 += c0;
                    out10 += c1;
                    out11 += c1;
                }
                if (broadcast_type_C == 3)
                {
                    out00 += beta == 1.f ? pC[0] : pC[0] * beta;
                    out01 += beta == 1.f ? pC[1] : pC[1] * beta;
                    out10 += beta == 1.f ? pC[c_hstep] : pC[c_hstep] * beta;
                    out11 += beta == 1.f ? pC[c_hstep + 1] : pC[c_hstep + 1] * beta;
                }
                if (broadcast_type_C == 4)
                {
                    out00 += beta == 1.f ? pC[0] : pC[0] * beta;
                    out01 += beta == 1.f ? pC[1] : pC[1] * beta;
                    out10 += beta == 1.f ? pC[0] : pC[0] * beta;
                    out11 += beta == 1.f ? pC[1] : pC[1] * beta;
                }
            }

            if (alpha != 1.f)
            {
                out00 *= alpha;
                out01 *= alpha;
                out10 *= alpha;
                out11 *= alpha;
            }

            outptr0[0] = out00;
            outptr0[1] = out01;
            outptr1[0] = out10;
            outptr1[1] = out11;

            outptr0 += 2;
            outptr1 += 2;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 2;
            pp += 4;
        }
        for (; jj < max_jj; jj += 1)
        {
            float out00 = pp[0];
            float out10 = pp[1];

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    out00 += c0;
                    out10 += c0;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    out00 += c0;
                    out10 += c1;
                }
                if (broadcast_type_C == 3)
                {
                    out00 += beta == 1.f ? pC[0] : pC[0] * beta;
                    out10 += beta == 1.f ? pC[c_hstep] : pC[c_hstep] * beta;
                }
                if (broadcast_type_C == 4)
                {
                    out00 += beta == 1.f ? pC[0] : pC[0] * beta;
                    out10 += beta == 1.f ? pC[0] : pC[0] * beta;
                }
            }

            if (alpha != 1.f)
            {
                out00 *= alpha;
                out10 *= alpha;
            }

            outptr0[0] = out00;
            outptr1[0] = out10;

            outptr0++;
            outptr1++;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC++;
            pp += 2;
        }
    }
    for (; ii < max_ii; ii++)
    {
        pC = (const float*)C;
        float* outptr0 = (float*)top_blob + (size_t)(i + ii) * out_hstep + j;
        float c0 = 0.f;
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                float c = pC[0];
                if (beta != 1.f)
                    c *= beta;
                c0 = c;
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                c0 = pC[i + ii];
                if (beta != 1.f)
                    c0 *= beta;
            }
            if (broadcast_type_C == 3)
                pC = (const float*)C + (i + ii) * c_hstep + j;
            if (broadcast_type_C == 4)
                pC = (const float*)C + j;
        }

        int jj = 0;
        for (; jj + 1 < max_jj; jj += 2)
        {
            float out00 = pp[0];
            float out01 = pp[1];

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    out00 += c0;
                    out01 += c0;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    out00 += c0;
                    out01 += c0;
                }
                if (broadcast_type_C == 3)
                {
                    out00 += beta == 1.f ? pC[0] : pC[0] * beta;
                    out01 += beta == 1.f ? pC[1] : pC[1] * beta;
                }
                if (broadcast_type_C == 4)
                {
                    out00 += beta == 1.f ? pC[0] : pC[0] * beta;
                    out01 += beta == 1.f ? pC[1] : pC[1] * beta;
                }
            }

            if (alpha != 1.f)
            {
                out00 *= alpha;
                out01 *= alpha;
            }

            outptr0[0] = out00;
            outptr0[1] = out01;

            outptr0 += 2;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 2;
            pp += 2;
        }
        for (; jj < max_jj; jj += 1)
        {
            float out00 = pp[0];

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    out00 += c0;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    out00 += c0;
                }
                if (broadcast_type_C == 3)
                {
                    out00 += beta == 1.f ? pC[0] : pC[0] * beta;
                }
                if (broadcast_type_C == 4)
                {
                    out00 += beta == 1.f ? pC[0] : pC[0] * beta;
                }
            }

            if (alpha != 1.f)
            {
                out00 *= alpha;
            }

            outptr0[0] = out00;

            outptr0++;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC++;
            pp += 1;
        }
    }
#endif // __ARM_NEON
}

static void transpose_unpack_output_tile_wq_int8(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, int N, float alpha, float beta)
{
    (void)N;

    const float* pC = C;
    const float* pp = topT;
    const size_t out_hstep = top_blob.dims == 3 ? top_blob.cstep : (size_t)top_blob.w;
    const size_t c_hstep = C.dims == 3 ? C.cstep : (size_t)C.w;

    int ii = 0;
#if __ARM_NEON
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        float* outptr0 = (float*)top_blob + (size_t)j * out_hstep + i + ii;
        float* outptr1 = outptr0 + out_hstep;
        float* outptr2 = outptr1 + out_hstep;
        float* outptr3 = outptr2 + out_hstep;
        pC = (const float*)C;
        float c0 = 0.f;
        float c1 = 0.f;
        float c2 = 0.f;
        float c3 = 0.f;
        float c4 = 0.f;
        float c5 = 0.f;
        float c6 = 0.f;
        float c7 = 0.f;
        if (pC)
        {
            if (broadcast_type_C == 0)
                c0 = c1 = c2 = c3 = c4 = c5 = c6 = c7 = pC[0] * beta;
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)C + i + ii;
                c0 = pC[0] * beta;
                c1 = pC[1] * beta;
                c2 = pC[2] * beta;
                c3 = pC[3] * beta;
                c4 = pC[4] * beta;
                c5 = pC[5] * beta;
                c6 = pC[6] * beta;
                c7 = pC[7] * beta;
            }
            if (broadcast_type_C == 3)
                pC = (const float*)C + (size_t)(i + ii) * c_hstep + j;
            if (broadcast_type_C == 4)
                pC = (const float*)C + j;
        }

        int jj = 0;
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _out0 = vld1q_f32(pp);
            float32x4_t _out1 = vld1q_f32(pp + 4);
            float32x4_t _out2 = vld1q_f32(pp + 8);
            float32x4_t _out3 = vld1q_f32(pp + 12);
            float32x4_t _out4 = vld1q_f32(pp + 16);
            float32x4_t _out5 = vld1q_f32(pp + 20);
            float32x4_t _out6 = vld1q_f32(pp + 24);
            float32x4_t _out7 = vld1q_f32(pp + 28);
            if (pC)
            {
                if (broadcast_type_C <= 2)
                {
                    _out0 = vaddq_f32(_out0, vdupq_n_f32(c0));
                    _out1 = vaddq_f32(_out1, vdupq_n_f32(c1));
                    _out2 = vaddq_f32(_out2, vdupq_n_f32(c2));
                    _out3 = vaddq_f32(_out3, vdupq_n_f32(c3));
                    _out4 = vaddq_f32(_out4, vdupq_n_f32(c4));
                    _out5 = vaddq_f32(_out5, vdupq_n_f32(c5));
                    _out6 = vaddq_f32(_out6, vdupq_n_f32(c6));
                    _out7 = vaddq_f32(_out7, vdupq_n_f32(c7));
                }
                if (broadcast_type_C == 3)
                {
                    _out0 = beta == 1.f ? vaddq_f32(_out0, vld1q_f32(pC)) : vmlaq_n_f32(_out0, vld1q_f32(pC), beta);
                    _out1 = beta == 1.f ? vaddq_f32(_out1, vld1q_f32(pC + c_hstep)) : vmlaq_n_f32(_out1, vld1q_f32(pC + c_hstep), beta);
                    _out2 = beta == 1.f ? vaddq_f32(_out2, vld1q_f32(pC + c_hstep * 2)) : vmlaq_n_f32(_out2, vld1q_f32(pC + c_hstep * 2), beta);
                    _out3 = beta == 1.f ? vaddq_f32(_out3, vld1q_f32(pC + c_hstep * 3)) : vmlaq_n_f32(_out3, vld1q_f32(pC + c_hstep * 3), beta);
                    _out4 = beta == 1.f ? vaddq_f32(_out4, vld1q_f32(pC + c_hstep * 4)) : vmlaq_n_f32(_out4, vld1q_f32(pC + c_hstep * 4), beta);
                    _out5 = beta == 1.f ? vaddq_f32(_out5, vld1q_f32(pC + c_hstep * 5)) : vmlaq_n_f32(_out5, vld1q_f32(pC + c_hstep * 5), beta);
                    _out6 = beta == 1.f ? vaddq_f32(_out6, vld1q_f32(pC + c_hstep * 6)) : vmlaq_n_f32(_out6, vld1q_f32(pC + c_hstep * 6), beta);
                    _out7 = beta == 1.f ? vaddq_f32(_out7, vld1q_f32(pC + c_hstep * 7)) : vmlaq_n_f32(_out7, vld1q_f32(pC + c_hstep * 7), beta);
                    pC += 4;
                }
                if (broadcast_type_C == 4)
                {
                    const float32x4_t _c = beta == 1.f ? vld1q_f32(pC) : vmulq_n_f32(vld1q_f32(pC), beta);
                    _out0 = vaddq_f32(_out0, _c);
                    _out1 = vaddq_f32(_out1, _c);
                    _out2 = vaddq_f32(_out2, _c);
                    _out3 = vaddq_f32(_out3, _c);
                    _out4 = vaddq_f32(_out4, _c);
                    _out5 = vaddq_f32(_out5, _c);
                    _out6 = vaddq_f32(_out6, _c);
                    _out7 = vaddq_f32(_out7, _c);
                    pC += 4;
                }
            }
            if (alpha != 1.f)
            {
                _out0 = vmulq_n_f32(_out0, alpha);
                _out1 = vmulq_n_f32(_out1, alpha);
                _out2 = vmulq_n_f32(_out2, alpha);
                _out3 = vmulq_n_f32(_out3, alpha);
                _out4 = vmulq_n_f32(_out4, alpha);
                _out5 = vmulq_n_f32(_out5, alpha);
                _out6 = vmulq_n_f32(_out6, alpha);
                _out7 = vmulq_n_f32(_out7, alpha);
            }
            transpose4x4_ps(_out0, _out1, _out2, _out3);
            transpose4x4_ps(_out4, _out5, _out6, _out7);
            vst1q_f32(outptr0, _out0);
            vst1q_f32(outptr0 + 4, _out4);
            vst1q_f32(outptr1, _out1);
            vst1q_f32(outptr1 + 4, _out5);
            vst1q_f32(outptr2, _out2);
            vst1q_f32(outptr2 + 4, _out6);
            vst1q_f32(outptr3, _out3);
            vst1q_f32(outptr3 + 4, _out7);
            outptr0 += out_hstep * 4;
            outptr1 += out_hstep * 4;
            outptr2 += out_hstep * 4;
            outptr3 += out_hstep * 4;
            pp += 32;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float32x2_t _out0 = vld1_f32(pp);
            float32x2_t _out1 = vld1_f32(pp + 2);
            float32x2_t _out2 = vld1_f32(pp + 4);
            float32x2_t _out3 = vld1_f32(pp + 6);
            float32x2_t _out4 = vld1_f32(pp + 8);
            float32x2_t _out5 = vld1_f32(pp + 10);
            float32x2_t _out6 = vld1_f32(pp + 12);
            float32x2_t _out7 = vld1_f32(pp + 14);
            if (pC)
            {
                if (broadcast_type_C <= 2)
                {
                    _out0 = vadd_f32(_out0, vdup_n_f32(c0));
                    _out1 = vadd_f32(_out1, vdup_n_f32(c1));
                    _out2 = vadd_f32(_out2, vdup_n_f32(c2));
                    _out3 = vadd_f32(_out3, vdup_n_f32(c3));
                    _out4 = vadd_f32(_out4, vdup_n_f32(c4));
                    _out5 = vadd_f32(_out5, vdup_n_f32(c5));
                    _out6 = vadd_f32(_out6, vdup_n_f32(c6));
                    _out7 = vadd_f32(_out7, vdup_n_f32(c7));
                }
                if (broadcast_type_C == 3)
                {
                    _out0 = beta == 1.f ? vadd_f32(_out0, vld1_f32(pC)) : vmla_n_f32(_out0, vld1_f32(pC), beta);
                    _out1 = beta == 1.f ? vadd_f32(_out1, vld1_f32(pC + c_hstep)) : vmla_n_f32(_out1, vld1_f32(pC + c_hstep), beta);
                    _out2 = beta == 1.f ? vadd_f32(_out2, vld1_f32(pC + c_hstep * 2)) : vmla_n_f32(_out2, vld1_f32(pC + c_hstep * 2), beta);
                    _out3 = beta == 1.f ? vadd_f32(_out3, vld1_f32(pC + c_hstep * 3)) : vmla_n_f32(_out3, vld1_f32(pC + c_hstep * 3), beta);
                    _out4 = beta == 1.f ? vadd_f32(_out4, vld1_f32(pC + c_hstep * 4)) : vmla_n_f32(_out4, vld1_f32(pC + c_hstep * 4), beta);
                    _out5 = beta == 1.f ? vadd_f32(_out5, vld1_f32(pC + c_hstep * 5)) : vmla_n_f32(_out5, vld1_f32(pC + c_hstep * 5), beta);
                    _out6 = beta == 1.f ? vadd_f32(_out6, vld1_f32(pC + c_hstep * 6)) : vmla_n_f32(_out6, vld1_f32(pC + c_hstep * 6), beta);
                    _out7 = beta == 1.f ? vadd_f32(_out7, vld1_f32(pC + c_hstep * 7)) : vmla_n_f32(_out7, vld1_f32(pC + c_hstep * 7), beta);
                    pC += 2;
                }
                if (broadcast_type_C == 4)
                {
                    const float32x2_t _c = beta == 1.f ? vld1_f32(pC) : vmul_n_f32(vld1_f32(pC), beta);
                    _out0 = vadd_f32(_out0, _c);
                    _out1 = vadd_f32(_out1, _c);
                    _out2 = vadd_f32(_out2, _c);
                    _out3 = vadd_f32(_out3, _c);
                    _out4 = vadd_f32(_out4, _c);
                    _out5 = vadd_f32(_out5, _c);
                    _out6 = vadd_f32(_out6, _c);
                    _out7 = vadd_f32(_out7, _c);
                    pC += 2;
                }
            }
            if (alpha != 1.f)
            {
                _out0 = vmul_n_f32(_out0, alpha);
                _out1 = vmul_n_f32(_out1, alpha);
                _out2 = vmul_n_f32(_out2, alpha);
                _out3 = vmul_n_f32(_out3, alpha);
                _out4 = vmul_n_f32(_out4, alpha);
                _out5 = vmul_n_f32(_out5, alpha);
                _out6 = vmul_n_f32(_out6, alpha);
                _out7 = vmul_n_f32(_out7, alpha);
            }
            float32x4x2_t _t0 = vuzpq_f32(vcombine_f32(_out0, _out1), vcombine_f32(_out2, _out3));
            float32x4x2_t _t1 = vuzpq_f32(vcombine_f32(_out4, _out5), vcombine_f32(_out6, _out7));
            vst1q_f32(outptr0, _t0.val[0]);
            vst1q_f32(outptr0 + 4, _t1.val[0]);
            vst1q_f32(outptr1, _t0.val[1]);
            vst1q_f32(outptr1 + 4, _t1.val[1]);
            outptr0 += out_hstep * 2;
            outptr1 += out_hstep * 2;
            pp += 16;
        }
        for (; jj < max_jj; jj++)
        {
            float32x4_t _out0 = vld1q_f32(pp);
            float32x4_t _out1 = vld1q_f32(pp + 4);
            if (pC)
            {
                if (broadcast_type_C <= 2)
                {
                    const float32x4_t _c0 = {c0, c1, c2, c3};
                    const float32x4_t _c1 = {c4, c5, c6, c7};
                    _out0 = vaddq_f32(_out0, _c0);
                    _out1 = vaddq_f32(_out1, _c1);
                }
                if (broadcast_type_C == 3)
                {
                    float32x4_t _c0 = {pC[0], pC[c_hstep], pC[c_hstep * 2], pC[c_hstep * 3]};
                    float32x4_t _c1 = {pC[c_hstep * 4], pC[c_hstep * 5], pC[c_hstep * 6], pC[c_hstep * 7]};
                    _out0 = beta == 1.f ? vaddq_f32(_out0, _c0) : vmlaq_n_f32(_out0, _c0, beta);
                    _out1 = beta == 1.f ? vaddq_f32(_out1, _c1) : vmlaq_n_f32(_out1, _c1, beta);
                    pC++;
                }
                if (broadcast_type_C == 4)
                {
                    const float32x4_t _c = vdupq_n_f32(beta == 1.f ? pC[0] : pC[0] * beta);
                    _out0 = vaddq_f32(_out0, _c);
                    _out1 = vaddq_f32(_out1, _c);
                    pC++;
                }
            }
            if (alpha != 1.f)
            {
                _out0 = vmulq_n_f32(_out0, alpha);
                _out1 = vmulq_n_f32(_out1, alpha);
            }
            vst1q_f32(outptr0, _out0);
            vst1q_f32(outptr0 + 4, _out1);
            outptr0 += out_hstep;
            pp += 8;
        }
    }
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        pC = (const float*)C;
        float* outptr0 = (float*)top_blob + (size_t)j * out_hstep + i + ii;
        float32x4_t _c0123 = vdupq_n_f32(0.f);
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                float c = pC[0];
                if (beta != 1.f)
                    c *= beta;
                _c0123 = vdupq_n_f32(c);
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                _c0123 = vld1q_f32(pC + i + ii);
                if (beta != 1.f)
                    _c0123 = vmulq_n_f32(_c0123, beta);
            }
            if (broadcast_type_C == 3)
                pC = (const float*)C + (i + ii) * c_hstep + j;
            if (broadcast_type_C == 4)
                pC = (const float*)C + j;
        }

        int jj = 0;
#if __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            float32x4_t _out0 = vld1q_f32(pp + 0);
            float32x4_t _out1 = vld1q_f32(pp + 4);
            float32x4_t _out2 = vld1q_f32(pp + 8);
            float32x4_t _out3 = vld1q_f32(pp + 12);
            float32x4_t _out4 = vld1q_f32(pp + 16);
            float32x4_t _out5 = vld1q_f32(pp + 20);
            float32x4_t _out6 = vld1q_f32(pp + 24);
            float32x4_t _out7 = vld1q_f32(pp + 28);

            if (pC)
            {
                if (broadcast_type_C == 3)
                {
                    _out0 = vaddq_f32(_out0, (beta == 1.f ? vld1q_f32(pC) : vmulq_n_f32(vld1q_f32(pC), beta)));
                    _out1 = vaddq_f32(_out1, (beta == 1.f ? vld1q_f32(pC + 4) : vmulq_n_f32(vld1q_f32(pC + 4), beta)));
                    _out2 = vaddq_f32(_out2, (beta == 1.f ? vld1q_f32(pC + c_hstep) : vmulq_n_f32(vld1q_f32(pC + c_hstep), beta)));
                    _out3 = vaddq_f32(_out3, (beta == 1.f ? vld1q_f32(pC + c_hstep + 4) : vmulq_n_f32(vld1q_f32(pC + c_hstep + 4), beta)));
                    _out4 = vaddq_f32(_out4, (beta == 1.f ? vld1q_f32(pC + c_hstep * 2) : vmulq_n_f32(vld1q_f32(pC + c_hstep * 2), beta)));
                    _out5 = vaddq_f32(_out5, (beta == 1.f ? vld1q_f32(pC + c_hstep * 2 + 4) : vmulq_n_f32(vld1q_f32(pC + c_hstep * 2 + 4), beta)));
                    _out6 = vaddq_f32(_out6, (beta == 1.f ? vld1q_f32(pC + c_hstep * 3) : vmulq_n_f32(vld1q_f32(pC + c_hstep * 3), beta)));
                    _out7 = vaddq_f32(_out7, (beta == 1.f ? vld1q_f32(pC + c_hstep * 3 + 4) : vmulq_n_f32(vld1q_f32(pC + c_hstep * 3 + 4), beta)));
                }
                if (broadcast_type_C == 4)
                {
                    const float32x4_t _cc0 = (beta == 1.f ? vld1q_f32(pC) : vmulq_n_f32(vld1q_f32(pC), beta));
                    _out0 = vaddq_f32(_out0, _cc0);
                    _out2 = vaddq_f32(_out2, _cc0);
                    _out4 = vaddq_f32(_out4, _cc0);
                    _out6 = vaddq_f32(_out6, _cc0);
                    const float32x4_t _cc1 = (beta == 1.f ? vld1q_f32(pC + 4) : vmulq_n_f32(vld1q_f32(pC + 4), beta));
                    _out1 = vaddq_f32(_out1, _cc1);
                    _out3 = vaddq_f32(_out3, _cc1);
                    _out5 = vaddq_f32(_out5, _cc1);
                    _out7 = vaddq_f32(_out7, _cc1);
                }
            }

            if (out_hstep == 4)
            {
                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _out0 = vaddq_f32(_out0, _c0123);
                        _out1 = vaddq_f32(_out1, _c0123);
                        _out2 = vaddq_f32(_out2, _c0123);
                        _out3 = vaddq_f32(_out3, _c0123);
                        _out4 = vaddq_f32(_out4, _c0123);
                        _out5 = vaddq_f32(_out5, _c0123);
                        _out6 = vaddq_f32(_out6, _c0123);
                        _out7 = vaddq_f32(_out7, _c0123);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        const float32x4_t _c0 = vdupq_lane_f32(vget_low_f32(_c0123), 0);
                        const float32x4_t _c1 = vdupq_lane_f32(vget_low_f32(_c0123), 1);
                        const float32x4_t _c2 = vdupq_lane_f32(vget_high_f32(_c0123), 0);
                        const float32x4_t _c3 = vdupq_lane_f32(vget_high_f32(_c0123), 1);
                        _out0 = vaddq_f32(_out0, _c0);
                        _out1 = vaddq_f32(_out1, _c0);
                        _out2 = vaddq_f32(_out2, _c1);
                        _out3 = vaddq_f32(_out3, _c1);
                        _out4 = vaddq_f32(_out4, _c2);
                        _out5 = vaddq_f32(_out5, _c2);
                        _out6 = vaddq_f32(_out6, _c3);
                        _out7 = vaddq_f32(_out7, _c3);
                    }
                }

                if (alpha != 1.f)
                {
                    _out0 = vmulq_n_f32(_out0, alpha);
                    _out1 = vmulq_n_f32(_out1, alpha);
                    _out2 = vmulq_n_f32(_out2, alpha);
                    _out3 = vmulq_n_f32(_out3, alpha);
                    _out4 = vmulq_n_f32(_out4, alpha);
                    _out5 = vmulq_n_f32(_out5, alpha);
                    _out6 = vmulq_n_f32(_out6, alpha);
                    _out7 = vmulq_n_f32(_out7, alpha);
                }

                float32x4x4_t _r0;
                _r0.val[0] = _out0;
                _r0.val[1] = _out2;
                _r0.val[2] = _out4;
                _r0.val[3] = _out6;
                vst4q_f32(outptr0, _r0);
                float32x4x4_t _r1;
                _r1.val[0] = _out1;
                _r1.val[1] = _out3;
                _r1.val[2] = _out5;
                _r1.val[3] = _out7;
                vst4q_f32(outptr0 + out_hstep * 4, _r1);
            }
            else
            {
                transpose4x4_ps(_out0, _out2, _out4, _out6);
                transpose4x4_ps(_out1, _out3, _out5, _out7);

                if (pC && broadcast_type_C <= 2)
                {
                    _out0 = vaddq_f32(_out0, _c0123);
                    _out1 = vaddq_f32(_out1, _c0123);
                    _out2 = vaddq_f32(_out2, _c0123);
                    _out3 = vaddq_f32(_out3, _c0123);
                    _out4 = vaddq_f32(_out4, _c0123);
                    _out5 = vaddq_f32(_out5, _c0123);
                    _out6 = vaddq_f32(_out6, _c0123);
                    _out7 = vaddq_f32(_out7, _c0123);
                }

                if (alpha != 1.f)
                {
                    _out0 = vmulq_n_f32(_out0, alpha);
                    _out1 = vmulq_n_f32(_out1, alpha);
                    _out2 = vmulq_n_f32(_out2, alpha);
                    _out3 = vmulq_n_f32(_out3, alpha);
                    _out4 = vmulq_n_f32(_out4, alpha);
                    _out5 = vmulq_n_f32(_out5, alpha);
                    _out6 = vmulq_n_f32(_out6, alpha);
                    _out7 = vmulq_n_f32(_out7, alpha);
                }

                vst1q_f32(outptr0, _out0);
                vst1q_f32(outptr0 + out_hstep, _out2);
                vst1q_f32(outptr0 + out_hstep * 2, _out4);
                vst1q_f32(outptr0 + out_hstep * 3, _out6);
                vst1q_f32(outptr0 + out_hstep * 4, _out1);
                vst1q_f32(outptr0 + out_hstep * 5, _out3);
                vst1q_f32(outptr0 + out_hstep * 6, _out5);
                vst1q_f32(outptr0 + out_hstep * 7, _out7);
            }

            outptr0 += out_hstep * 8;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 8;
            pp += 32;
        }
#endif // __aarch64__
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _out0 = vld1q_f32(pp + 0);
            float32x4_t _out1 = vld1q_f32(pp + 4);
            float32x4_t _out2 = vld1q_f32(pp + 8);
            float32x4_t _out3 = vld1q_f32(pp + 12);

            if (pC)
            {
                if (broadcast_type_C == 3)
                {
                    _out0 = vaddq_f32(_out0, (beta == 1.f ? vld1q_f32(pC) : vmulq_n_f32(vld1q_f32(pC), beta)));
                    _out1 = vaddq_f32(_out1, (beta == 1.f ? vld1q_f32(pC + c_hstep) : vmulq_n_f32(vld1q_f32(pC + c_hstep), beta)));
                    _out2 = vaddq_f32(_out2, (beta == 1.f ? vld1q_f32(pC + c_hstep * 2) : vmulq_n_f32(vld1q_f32(pC + c_hstep * 2), beta)));
                    _out3 = vaddq_f32(_out3, (beta == 1.f ? vld1q_f32(pC + c_hstep * 3) : vmulq_n_f32(vld1q_f32(pC + c_hstep * 3), beta)));
                }
                if (broadcast_type_C == 4)
                {
                    const float32x4_t _cc0 = (beta == 1.f ? vld1q_f32(pC) : vmulq_n_f32(vld1q_f32(pC), beta));
                    _out0 = vaddq_f32(_out0, _cc0);
                    _out1 = vaddq_f32(_out1, _cc0);
                    _out2 = vaddq_f32(_out2, _cc0);
                    _out3 = vaddq_f32(_out3, _cc0);
                }
            }

            if (out_hstep == 4)
            {
                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _out0 = vaddq_f32(_out0, _c0123);
                        _out1 = vaddq_f32(_out1, _c0123);
                        _out2 = vaddq_f32(_out2, _c0123);
                        _out3 = vaddq_f32(_out3, _c0123);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _out0 = vaddq_f32(_out0, vdupq_lane_f32(vget_low_f32(_c0123), 0));
                        _out1 = vaddq_f32(_out1, vdupq_lane_f32(vget_low_f32(_c0123), 1));
                        _out2 = vaddq_f32(_out2, vdupq_lane_f32(vget_high_f32(_c0123), 0));
                        _out3 = vaddq_f32(_out3, vdupq_lane_f32(vget_high_f32(_c0123), 1));
                    }
                }

                if (alpha != 1.f)
                {
                    _out0 = vmulq_n_f32(_out0, alpha);
                    _out1 = vmulq_n_f32(_out1, alpha);
                    _out2 = vmulq_n_f32(_out2, alpha);
                    _out3 = vmulq_n_f32(_out3, alpha);
                }

                float32x4x4_t _r;
                _r.val[0] = _out0;
                _r.val[1] = _out1;
                _r.val[2] = _out2;
                _r.val[3] = _out3;
                vst4q_f32(outptr0, _r);
            }
            else
            {
                transpose4x4_ps(_out0, _out1, _out2, _out3);

                if (pC && broadcast_type_C <= 2)
                {
                    _out0 = vaddq_f32(_out0, _c0123);
                    _out1 = vaddq_f32(_out1, _c0123);
                    _out2 = vaddq_f32(_out2, _c0123);
                    _out3 = vaddq_f32(_out3, _c0123);
                }

                if (alpha != 1.f)
                {
                    _out0 = vmulq_n_f32(_out0, alpha);
                    _out1 = vmulq_n_f32(_out1, alpha);
                    _out2 = vmulq_n_f32(_out2, alpha);
                    _out3 = vmulq_n_f32(_out3, alpha);
                }

                vst1q_f32(outptr0, _out0);
                vst1q_f32(outptr0 + out_hstep, _out1);
                vst1q_f32(outptr0 + out_hstep * 2, _out2);
                vst1q_f32(outptr0 + out_hstep * 3, _out3);
            }

            outptr0 += out_hstep * 4;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 4;
            pp += 16;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float32x4_t _out0 = vld1q_f32(pp);
            float32x4_t _out1 = vld1q_f32(pp + 4);

            if (pC)
            {
                if (broadcast_type_C == 3)
                {
                    const float32x4_t _c01 = vcombine_f32(vld1_f32(pC), vld1_f32(pC + c_hstep));
                    const float32x4_t _c23 = vcombine_f32(vld1_f32(pC + c_hstep * 2), vld1_f32(pC + c_hstep * 3));
                    _out0 = beta == 1.f ? vaddq_f32(_out0, _c01) : vmlaq_n_f32(_out0, _c01, beta);
                    _out1 = beta == 1.f ? vaddq_f32(_out1, _c23) : vmlaq_n_f32(_out1, _c23, beta);
                }
                if (broadcast_type_C == 4)
                {
                    float32x2_t _c = vld1_f32(pC);
                    if (beta != 1.f)
                        _c = vmul_n_f32(_c, beta);
                    const float32x4_t _cc0 = vcombine_f32(_c, _c);
                    _out0 = vaddq_f32(_out0, _cc0);
                    _out1 = vaddq_f32(_out1, _cc0);
                }
            }

            if (out_hstep == 4)
            {
                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _out0 = vaddq_f32(_out0, _c0123);
                        _out1 = vaddq_f32(_out1, _c0123);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        float32x4x2_t _c = vzipq_f32(_c0123, _c0123);
                        _out0 = vaddq_f32(_out0, _c.val[0]);
                        _out1 = vaddq_f32(_out1, _c.val[1]);
                    }
                }

                if (alpha != 1.f)
                {
                    _out0 = vmulq_n_f32(_out0, alpha);
                    _out1 = vmulq_n_f32(_out1, alpha);
                }

                float32x2x4_t _r;
                _r.val[0] = vget_low_f32(_out0);
                _r.val[1] = vget_high_f32(_out0);
                _r.val[2] = vget_low_f32(_out1);
                _r.val[3] = vget_high_f32(_out1);
                vst4_f32(outptr0, _r);
            }
            else
            {
                float32x4x2_t _t = vuzpq_f32(_out0, _out1);

                if (pC && broadcast_type_C <= 2)
                {
                    _t.val[0] = vaddq_f32(_t.val[0], _c0123);
                    _t.val[1] = vaddq_f32(_t.val[1], _c0123);
                }

                if (alpha != 1.f)
                {
                    _t.val[0] = vmulq_n_f32(_t.val[0], alpha);
                    _t.val[1] = vmulq_n_f32(_t.val[1], alpha);
                }

                vst1q_f32(outptr0, _t.val[0]);
                vst1q_f32(outptr0 + out_hstep, _t.val[1]);
            }

            outptr0 += out_hstep * 2;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 2;
            pp += 8;
        }
        for (; jj < max_jj; jj += 1)
        {
            float32x4_t _out0 = vld1q_f32(pp);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _out0 = vaddq_f32(_out0, _c0123);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _out0 = vaddq_f32(_out0, _c0123);
                }
                if (broadcast_type_C == 3)
                {
                    float32x4_t _c = vdupq_n_f32(pC[0]);
                    _c = vsetq_lane_f32(pC[c_hstep], _c, 1);
                    _c = vsetq_lane_f32(pC[c_hstep * 2], _c, 2);
                    _c = vsetq_lane_f32(pC[c_hstep * 3], _c, 3);
                    _out0 = beta == 1.f ? vaddq_f32(_out0, _c) : vmlaq_n_f32(_out0, _c, beta);
                }
                if (broadcast_type_C == 4)
                {
                    const float32x4_t _cc0 = vdupq_n_f32(beta == 1.f ? pC[0] : pC[0] * beta);
                    _out0 = vaddq_f32(_out0, _cc0);
                }
            }

            if (alpha != 1.f)
            {
                _out0 = vmulq_n_f32(_out0, alpha);
            }

            vst1q_f32(outptr0, _out0);

            outptr0 += out_hstep;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC++;
            pp += 4;
        }
    }
    for (; ii + 1 < max_ii; ii += 2)
    {
        pC = (const float*)C;
        float* outptr0 = (float*)top_blob + (size_t)j * out_hstep + i + ii;
        float32x4_t _c01 = vdupq_n_f32(0.f);
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                float c = pC[0];
                if (beta != 1.f)
                    c *= beta;
                _c01 = vdupq_n_f32(c);
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                float32x2_t _c = vld1_f32(pC + i + ii);
                if (beta != 1.f)
                    _c = vmul_n_f32(_c, beta);
                _c01 = vcombine_f32(_c, _c);
            }
            if (broadcast_type_C == 3)
                pC = (const float*)C + (i + ii) * c_hstep + j;
            if (broadcast_type_C == 4)
                pC = (const float*)C + j;
        }

        int jj = 0;
#if __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            float32x4_t _out0 = vld1q_f32(pp + 0);
            float32x4_t _out1 = vld1q_f32(pp + 4);
            float32x4_t _out2 = vld1q_f32(pp + 8);
            float32x4_t _out3 = vld1q_f32(pp + 12);

            if (pC)
            {
                if (broadcast_type_C == 3)
                {
                    _out0 = vaddq_f32(_out0, (beta == 1.f ? vld1q_f32(pC) : vmulq_n_f32(vld1q_f32(pC), beta)));
                    _out1 = vaddq_f32(_out1, (beta == 1.f ? vld1q_f32(pC + 4) : vmulq_n_f32(vld1q_f32(pC + 4), beta)));
                    _out2 = vaddq_f32(_out2, (beta == 1.f ? vld1q_f32(pC + c_hstep) : vmulq_n_f32(vld1q_f32(pC + c_hstep), beta)));
                    _out3 = vaddq_f32(_out3, (beta == 1.f ? vld1q_f32(pC + c_hstep + 4) : vmulq_n_f32(vld1q_f32(pC + c_hstep + 4), beta)));
                }
                if (broadcast_type_C == 4)
                {
                    const float32x4_t _c0 = (beta == 1.f ? vld1q_f32(pC) : vmulq_n_f32(vld1q_f32(pC), beta));
                    _out0 = vaddq_f32(_out0, _c0);
                    _out2 = vaddq_f32(_out2, _c0);
                    const float32x4_t _c1 = (beta == 1.f ? vld1q_f32(pC + 4) : vmulq_n_f32(vld1q_f32(pC + 4), beta));
                    _out1 = vaddq_f32(_out1, _c1);
                    _out3 = vaddq_f32(_out3, _c1);
                }
            }

            if (out_hstep == 2)
            {
                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _out0 = vaddq_f32(_out0, _c01);
                        _out1 = vaddq_f32(_out1, _c01);
                        _out2 = vaddq_f32(_out2, _c01);
                        _out3 = vaddq_f32(_out3, _c01);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        const float32x4_t _c0 = vdupq_lane_f32(vget_low_f32(_c01), 0);
                        const float32x4_t _c1 = vdupq_lane_f32(vget_low_f32(_c01), 1);
                        _out0 = vaddq_f32(_out0, _c0);
                        _out1 = vaddq_f32(_out1, _c0);
                        _out2 = vaddq_f32(_out2, _c1);
                        _out3 = vaddq_f32(_out3, _c1);
                    }
                }

                if (alpha != 1.f)
                {
                    _out0 = vmulq_n_f32(_out0, alpha);
                    _out1 = vmulq_n_f32(_out1, alpha);
                    _out2 = vmulq_n_f32(_out2, alpha);
                    _out3 = vmulq_n_f32(_out3, alpha);
                }

                float32x4x2_t _r0;
                _r0.val[0] = _out0;
                _r0.val[1] = _out2;
                vst2q_f32(outptr0, _r0);
                float32x4x2_t _r1;
                _r1.val[0] = _out1;
                _r1.val[1] = _out3;
                vst2q_f32(outptr0 + out_hstep * 4, _r1);
            }
            else
            {
                float32x4x2_t _t0 = vzipq_f32(_out0, _out2);
                float32x4x2_t _t1 = vzipq_f32(_out1, _out3);

                if (pC && broadcast_type_C <= 2)
                {
                    _t0.val[0] = vaddq_f32(_t0.val[0], _c01);
                    _t0.val[1] = vaddq_f32(_t0.val[1], _c01);
                    _t1.val[0] = vaddq_f32(_t1.val[0], _c01);
                    _t1.val[1] = vaddq_f32(_t1.val[1], _c01);
                }

                if (alpha == 1.f)
                {
#if __aarch64__
                    vst1q_lane_u64((uint64_t*)(outptr0), vreinterpretq_u64_f32(_t0.val[0]), 0);
                    vst1q_lane_u64((uint64_t*)(outptr0 + out_hstep), vreinterpretq_u64_f32(_t0.val[0]), 1);
                    vst1q_lane_u64((uint64_t*)(outptr0 + out_hstep * 2), vreinterpretq_u64_f32(_t0.val[1]), 0);
                    vst1q_lane_u64((uint64_t*)(outptr0 + out_hstep * 3), vreinterpretq_u64_f32(_t0.val[1]), 1);
                    vst1q_lane_u64((uint64_t*)(outptr0 + out_hstep * 4), vreinterpretq_u64_f32(_t1.val[0]), 0);
                    vst1q_lane_u64((uint64_t*)(outptr0 + out_hstep * 5), vreinterpretq_u64_f32(_t1.val[0]), 1);
                    vst1q_lane_u64((uint64_t*)(outptr0 + out_hstep * 6), vreinterpretq_u64_f32(_t1.val[1]), 0);
                    vst1q_lane_u64((uint64_t*)(outptr0 + out_hstep * 7), vreinterpretq_u64_f32(_t1.val[1]), 1);
#else
                    vst1_f32(outptr0, vget_low_f32(_t0.val[0]));
                    vst1_f32(outptr0 + out_hstep, vget_high_f32(_t0.val[0]));
                    vst1_f32(outptr0 + out_hstep * 2, vget_low_f32(_t0.val[1]));
                    vst1_f32(outptr0 + out_hstep * 3, vget_high_f32(_t0.val[1]));
                    vst1_f32(outptr0 + out_hstep * 4, vget_low_f32(_t1.val[0]));
                    vst1_f32(outptr0 + out_hstep * 5, vget_high_f32(_t1.val[0]));
                    vst1_f32(outptr0 + out_hstep * 6, vget_low_f32(_t1.val[1]));
                    vst1_f32(outptr0 + out_hstep * 7, vget_high_f32(_t1.val[1]));
#endif
                }
                else
                {
                    _t0.val[0] = vmulq_n_f32(_t0.val[0], alpha);
                    _t0.val[1] = vmulq_n_f32(_t0.val[1], alpha);
                    _t1.val[0] = vmulq_n_f32(_t1.val[0], alpha);
                    _t1.val[1] = vmulq_n_f32(_t1.val[1], alpha);

#if __aarch64__
                    vst1q_lane_u64((uint64_t*)(outptr0), vreinterpretq_u64_f32(_t0.val[0]), 0);
                    vst1q_lane_u64((uint64_t*)(outptr0 + out_hstep), vreinterpretq_u64_f32(_t0.val[0]), 1);
                    vst1q_lane_u64((uint64_t*)(outptr0 + out_hstep * 2), vreinterpretq_u64_f32(_t0.val[1]), 0);
                    vst1q_lane_u64((uint64_t*)(outptr0 + out_hstep * 3), vreinterpretq_u64_f32(_t0.val[1]), 1);
                    vst1q_lane_u64((uint64_t*)(outptr0 + out_hstep * 4), vreinterpretq_u64_f32(_t1.val[0]), 0);
                    vst1q_lane_u64((uint64_t*)(outptr0 + out_hstep * 5), vreinterpretq_u64_f32(_t1.val[0]), 1);
                    vst1q_lane_u64((uint64_t*)(outptr0 + out_hstep * 6), vreinterpretq_u64_f32(_t1.val[1]), 0);
                    vst1q_lane_u64((uint64_t*)(outptr0 + out_hstep * 7), vreinterpretq_u64_f32(_t1.val[1]), 1);
#else
                    vst1_f32(outptr0, vget_low_f32(_t0.val[0]));
                    vst1_f32(outptr0 + out_hstep, vget_high_f32(_t0.val[0]));
                    vst1_f32(outptr0 + out_hstep * 2, vget_low_f32(_t0.val[1]));
                    vst1_f32(outptr0 + out_hstep * 3, vget_high_f32(_t0.val[1]));
                    vst1_f32(outptr0 + out_hstep * 4, vget_low_f32(_t1.val[0]));
                    vst1_f32(outptr0 + out_hstep * 5, vget_high_f32(_t1.val[0]));
                    vst1_f32(outptr0 + out_hstep * 6, vget_low_f32(_t1.val[1]));
                    vst1_f32(outptr0 + out_hstep * 7, vget_high_f32(_t1.val[1]));
#endif
                }
            }

            outptr0 += out_hstep * 8;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 8;
            pp += 16;
        }
#endif // __aarch64__
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _out0 = vld1q_f32(pp + 0);
            float32x4_t _out1 = vld1q_f32(pp + 4);

            if (pC)
            {
                if (broadcast_type_C == 3)
                {
                    _out0 = vaddq_f32(_out0, (beta == 1.f ? vld1q_f32(pC) : vmulq_n_f32(vld1q_f32(pC), beta)));
                    _out1 = vaddq_f32(_out1, (beta == 1.f ? vld1q_f32(pC + c_hstep) : vmulq_n_f32(vld1q_f32(pC + c_hstep), beta)));
                }
                if (broadcast_type_C == 4)
                {
                    const float32x4_t _c0 = (beta == 1.f ? vld1q_f32(pC) : vmulq_n_f32(vld1q_f32(pC), beta));
                    _out0 = vaddq_f32(_out0, _c0);
                    _out1 = vaddq_f32(_out1, _c0);
                }
            }

            if (out_hstep == 2)
            {
                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _out0 = vaddq_f32(_out0, _c01);
                        _out1 = vaddq_f32(_out1, _c01);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _out0 = vaddq_f32(_out0, vdupq_lane_f32(vget_low_f32(_c01), 0));
                        _out1 = vaddq_f32(_out1, vdupq_lane_f32(vget_low_f32(_c01), 1));
                    }
                }

                if (alpha != 1.f)
                {
                    _out0 = vmulq_n_f32(_out0, alpha);
                    _out1 = vmulq_n_f32(_out1, alpha);
                }

                float32x4x2_t _r;
                _r.val[0] = _out0;
                _r.val[1] = _out1;
                vst2q_f32(outptr0, _r);
            }
            else
            {
                float32x4x2_t _t0 = vzipq_f32(_out0, _out1);

                if (pC && broadcast_type_C <= 2)
                {
                    _t0.val[0] = vaddq_f32(_t0.val[0], _c01);
                    _t0.val[1] = vaddq_f32(_t0.val[1], _c01);
                }

                if (alpha == 1.f)
                {
#if __aarch64__
                    vst1q_lane_u64((uint64_t*)(outptr0), vreinterpretq_u64_f32(_t0.val[0]), 0);
                    vst1q_lane_u64((uint64_t*)(outptr0 + out_hstep), vreinterpretq_u64_f32(_t0.val[0]), 1);
                    vst1q_lane_u64((uint64_t*)(outptr0 + out_hstep * 2), vreinterpretq_u64_f32(_t0.val[1]), 0);
                    vst1q_lane_u64((uint64_t*)(outptr0 + out_hstep * 3), vreinterpretq_u64_f32(_t0.val[1]), 1);
#else
                    vst1_f32(outptr0, vget_low_f32(_t0.val[0]));
                    vst1_f32(outptr0 + out_hstep, vget_high_f32(_t0.val[0]));
                    vst1_f32(outptr0 + out_hstep * 2, vget_low_f32(_t0.val[1]));
                    vst1_f32(outptr0 + out_hstep * 3, vget_high_f32(_t0.val[1]));
#endif
                }
                else
                {
                    _t0.val[0] = vmulq_n_f32(_t0.val[0], alpha);
                    _t0.val[1] = vmulq_n_f32(_t0.val[1], alpha);

#if __aarch64__
                    vst1q_lane_u64((uint64_t*)(outptr0), vreinterpretq_u64_f32(_t0.val[0]), 0);
                    vst1q_lane_u64((uint64_t*)(outptr0 + out_hstep), vreinterpretq_u64_f32(_t0.val[0]), 1);
                    vst1q_lane_u64((uint64_t*)(outptr0 + out_hstep * 2), vreinterpretq_u64_f32(_t0.val[1]), 0);
                    vst1q_lane_u64((uint64_t*)(outptr0 + out_hstep * 3), vreinterpretq_u64_f32(_t0.val[1]), 1);
#else
                    vst1_f32(outptr0, vget_low_f32(_t0.val[0]));
                    vst1_f32(outptr0 + out_hstep, vget_high_f32(_t0.val[0]));
                    vst1_f32(outptr0 + out_hstep * 2, vget_low_f32(_t0.val[1]));
                    vst1_f32(outptr0 + out_hstep * 3, vget_high_f32(_t0.val[1]));
#endif
                }
            }

            outptr0 += out_hstep * 4;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 4;
            pp += 8;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float32x2_t _out0 = vld1_f32(pp);
            float32x2_t _out1 = vld1_f32(pp + 2);

            if (pC)
            {
                if (broadcast_type_C == 3)
                {
                    const float32x2_t _c0 = vld1_f32(pC);
                    const float32x2_t _c1 = vld1_f32(pC + c_hstep);
                    _out0 = beta == 1.f ? vadd_f32(_out0, _c0) : vmla_n_f32(_out0, _c0, beta);
                    _out1 = beta == 1.f ? vadd_f32(_out1, _c1) : vmla_n_f32(_out1, _c1, beta);
                }
                if (broadcast_type_C == 4)
                {
                    const float32x2_t _c0 = beta == 1.f ? vld1_f32(pC) : vmul_n_f32(vld1_f32(pC), beta);
                    _out0 = vadd_f32(_out0, _c0);
                    _out1 = vadd_f32(_out1, _c0);
                }
            }

            if (out_hstep == 2)
            {
                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _out0 = vadd_f32(_out0, vget_low_f32(_c01));
                        _out1 = vadd_f32(_out1, vget_low_f32(_c01));
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _out0 = vadd_f32(_out0, vdup_lane_f32(vget_low_f32(_c01), 0));
                        _out1 = vadd_f32(_out1, vdup_lane_f32(vget_low_f32(_c01), 1));
                    }
                }

                if (alpha != 1.f)
                {
                    _out0 = vmul_n_f32(_out0, alpha);
                    _out1 = vmul_n_f32(_out1, alpha);
                }

                float32x2x2_t _r;
                _r.val[0] = _out0;
                _r.val[1] = _out1;
                vst2_f32(outptr0, _r);
            }
            else
            {
                float32x2x2_t _t0 = vzip_f32(_out0, _out1);

                if (pC && broadcast_type_C <= 2)
                {
                    _t0.val[0] = vadd_f32(_t0.val[0], vget_low_f32(_c01));
                    _t0.val[1] = vadd_f32(_t0.val[1], vget_low_f32(_c01));
                }

                if (alpha == 1.f)
                {
                    vst1_f32(outptr0, _t0.val[0]);
                    vst1_f32(outptr0 + out_hstep, _t0.val[1]);
                }
                else
                {
                    _t0.val[0] = vmul_n_f32(_t0.val[0], alpha);
                    _t0.val[1] = vmul_n_f32(_t0.val[1], alpha);

                    vst1_f32(outptr0, _t0.val[0]);
                    vst1_f32(outptr0 + out_hstep, _t0.val[1]);
                }
            }

            outptr0 += out_hstep * 2;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 2;
            pp += 4;
        }
        for (; jj < max_jj; jj += 1)
        {
            float32x2_t _out0 = vld1_f32(pp);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _out0 = vadd_f32(_out0, vget_low_f32(_c01));
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _out0 = vadd_f32(_out0, vget_low_f32(_c01));
                }
                if (broadcast_type_C == 3)
                {
                    float32x2_t _c = vdup_n_f32(pC[0]);
                    _c = vset_lane_f32(pC[c_hstep], _c, 1);
                    _out0 = beta == 1.f ? vadd_f32(_out0, _c) : vmla_n_f32(_out0, _c, beta);
                }
                if (broadcast_type_C == 4)
                {
                    _out0 = vadd_f32(_out0, vdup_n_f32(beta == 1.f ? pC[0] : pC[0] * beta));
                }
            }

            if (alpha != 1.f)
            {
                _out0 = vmul_n_f32(_out0, alpha);
            }

            vst1_f32(outptr0, _out0);

            outptr0 += out_hstep;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC++;
            pp += 2;
        }
    }
    for (; ii < max_ii; ii++)
    {
        pC = (const float*)C;
        float* outptr0 = (float*)top_blob + (size_t)j * out_hstep + i + ii;
        float32x4_t _c0;
        float c0 = 0.f;
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                c0 = pC[0];
                if (beta != 1.f)
                    c0 *= beta;
                _c0 = vdupq_n_f32(c0);
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                c0 = pC[i + ii];
                if (beta != 1.f)
                    c0 *= beta;
                _c0 = vdupq_n_f32(c0);
            }
            if (broadcast_type_C == 3)
                pC = (const float*)C + (i + ii) * c_hstep + j;
            if (broadcast_type_C == 4)
                pC = (const float*)C + j;
        }

        int jj = 0;
#if __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            float32x4_t _out0 = vld1q_f32(pp + 0);
            float32x4_t _out1 = vld1q_f32(pp + 4);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _out0 = vaddq_f32(_out0, _c0);
                    _out1 = vaddq_f32(_out1, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _out0 = vaddq_f32(_out0, _c0);
                    _out1 = vaddq_f32(_out1, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    _out0 = vaddq_f32(_out0, (beta == 1.f ? vld1q_f32(pC) : vmulq_n_f32(vld1q_f32(pC), beta)));
                    _out1 = vaddq_f32(_out1, (beta == 1.f ? vld1q_f32(pC + 4) : vmulq_n_f32(vld1q_f32(pC + 4), beta)));
                }
                if (broadcast_type_C == 4)
                {
                    const float32x4_t _c0 = (beta == 1.f ? vld1q_f32(pC) : vmulq_n_f32(vld1q_f32(pC), beta));
                    _out0 = vaddq_f32(_out0, _c0);
                    const float32x4_t _c1 = (beta == 1.f ? vld1q_f32(pC + 4) : vmulq_n_f32(vld1q_f32(pC + 4), beta));
                    _out1 = vaddq_f32(_out1, _c1);
                }
            }

            if (alpha != 1.f)
            {
                _out0 = vmulq_n_f32(_out0, alpha);
                _out1 = vmulq_n_f32(_out1, alpha);
            }

            if (out_hstep == 1)
            {
                vst1q_f32(outptr0, _out0);
                vst1q_f32(outptr0 + 4, _out1);
            }
            else
            {
                vst1q_lane_f32(outptr0, _out0, 0);
                vst1q_lane_f32(outptr0 + out_hstep, _out0, 1);
                vst1q_lane_f32(outptr0 + out_hstep * 2, _out0, 2);
                vst1q_lane_f32(outptr0 + out_hstep * 3, _out0, 3);
                vst1q_lane_f32(outptr0 + out_hstep * 4, _out1, 0);
                vst1q_lane_f32(outptr0 + out_hstep * 5, _out1, 1);
                vst1q_lane_f32(outptr0 + out_hstep * 6, _out1, 2);
                vst1q_lane_f32(outptr0 + out_hstep * 7, _out1, 3);
            }

            outptr0 += out_hstep * 8;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 8;
            pp += 8;
        }
#endif // __aarch64__
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _out0 = vld1q_f32(pp + 0);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _out0 = vaddq_f32(_out0, _c0);
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _out0 = vaddq_f32(_out0, _c0);
                }
                if (broadcast_type_C == 3)
                {
                    _out0 = vaddq_f32(_out0, (beta == 1.f ? vld1q_f32(pC) : vmulq_n_f32(vld1q_f32(pC), beta)));
                }
                if (broadcast_type_C == 4)
                {
                    const float32x4_t _c0 = (beta == 1.f ? vld1q_f32(pC) : vmulq_n_f32(vld1q_f32(pC), beta));
                    _out0 = vaddq_f32(_out0, _c0);
                }
            }

            if (alpha != 1.f)
            {
                _out0 = vmulq_n_f32(_out0, alpha);
            }

            if (out_hstep == 1)
            {
                vst1q_f32(outptr0, _out0);
            }
            else
            {
                vst1q_lane_f32(outptr0, _out0, 0);
                vst1q_lane_f32(outptr0 + out_hstep, _out0, 1);
                vst1q_lane_f32(outptr0 + out_hstep * 2, _out0, 2);
                vst1q_lane_f32(outptr0 + out_hstep * 3, _out0, 3);
            }

            outptr0 += out_hstep * 4;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 4;
            pp += 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float32x2_t _out0 = vld1_f32(pp);

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    _out0 = vadd_f32(_out0, vdup_n_f32(c0));
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    _out0 = vadd_f32(_out0, vdup_n_f32(c0));
                }
                if (broadcast_type_C == 3)
                {
                    const float32x2_t _c0 = vld1_f32(pC);
                    _out0 = beta == 1.f ? vadd_f32(_out0, _c0) : vmla_n_f32(_out0, _c0, beta);
                }
                if (broadcast_type_C == 4)
                {
                    const float32x2_t _c0 = beta == 1.f ? vld1_f32(pC) : vmul_n_f32(vld1_f32(pC), beta);
                    _out0 = vadd_f32(_out0, _c0);
                }
            }

            if (alpha != 1.f)
            {
                _out0 = vmul_n_f32(_out0, alpha);
            }

            if (out_hstep == 1)
            {
                vst1_f32(outptr0, _out0);
            }
            else
            {
                vst1_lane_f32(outptr0, _out0, 0);
                vst1_lane_f32(outptr0 + out_hstep, _out0, 1);
            }

            outptr0 += out_hstep * 2;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 2;
            pp += 2;
        }
        for (; jj < max_jj; jj += 1)
        {
            float out0 = pp[0];

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    out0 += c0;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    out0 += c0;
                }
                if (broadcast_type_C == 3)
                {
                    out0 += beta == 1.f ? pC[0] : pC[0] * beta;
                }
                if (broadcast_type_C == 4)
                {
                    out0 += beta == 1.f ? pC[0] : pC[0] * beta;
                }
            }

            if (alpha != 1.f)
            {
                out0 *= alpha;
            }

            outptr0[0] = out0;

            outptr0 += out_hstep;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC++;
            pp += 1;
        }
    }
#else
    for (; ii + 1 < max_ii; ii += 2)
    {
        pC = (const float*)C;
        float* outptr0 = (float*)top_blob + (size_t)j * out_hstep + i + ii;
        float c0 = 0.f;
        float c1 = 0.f;
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                float c = pC[0];
                if (beta != 1.f)
                    c *= beta;
                c0 = c;
                c1 = c;
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                c0 = pC[i + ii];
                if (beta != 1.f)
                    c0 *= beta;
                c1 = pC[i + ii + 1];
                if (beta != 1.f)
                    c1 *= beta;
            }
            if (broadcast_type_C == 3)
                pC = (const float*)C + (i + ii) * c_hstep + j;
            if (broadcast_type_C == 4)
                pC = (const float*)C + j;
        }

        int jj = 0;
        for (; jj + 1 < max_jj; jj += 2)
        {
            float out00 = pp[0];
            float out01 = pp[1];
            float out10 = pp[2];
            float out11 = pp[3];

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    out00 += c0;
                    out01 += c0;
                    out10 += c0;
                    out11 += c0;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    out00 += c0;
                    out01 += c0;
                    out10 += c1;
                    out11 += c1;
                }
                if (broadcast_type_C == 3)
                {
                    out00 += beta == 1.f ? pC[0] : pC[0] * beta;
                    out01 += beta == 1.f ? pC[1] : pC[1] * beta;
                    out10 += beta == 1.f ? pC[c_hstep] : pC[c_hstep] * beta;
                    out11 += beta == 1.f ? pC[c_hstep + 1] : pC[c_hstep + 1] * beta;
                }
                if (broadcast_type_C == 4)
                {
                    out00 += beta == 1.f ? pC[0] : pC[0] * beta;
                    out01 += beta == 1.f ? pC[1] : pC[1] * beta;
                    out10 += beta == 1.f ? pC[0] : pC[0] * beta;
                    out11 += beta == 1.f ? pC[1] : pC[1] * beta;
                }
            }

            if (alpha != 1.f)
            {
                out00 *= alpha;
                out01 *= alpha;
                out10 *= alpha;
                out11 *= alpha;
            }

            outptr0[0] = out00;
            outptr0[out_hstep] = out01;
            outptr0[1] = out10;
            outptr0[out_hstep + 1] = out11;

            outptr0 += out_hstep * 2;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 2;
            pp += 4;
        }
        for (; jj < max_jj; jj += 1)
        {
            float out00 = pp[0];
            float out10 = pp[1];

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    out00 += c0;
                    out10 += c0;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    out00 += c0;
                    out10 += c1;
                }
                if (broadcast_type_C == 3)
                {
                    out00 += beta == 1.f ? pC[0] : pC[0] * beta;
                    out10 += beta == 1.f ? pC[c_hstep] : pC[c_hstep] * beta;
                }
                if (broadcast_type_C == 4)
                {
                    out00 += beta == 1.f ? pC[0] : pC[0] * beta;
                    out10 += beta == 1.f ? pC[0] : pC[0] * beta;
                }
            }

            if (alpha != 1.f)
            {
                out00 *= alpha;
                out10 *= alpha;
            }

            outptr0[0] = out00;
            outptr0[1] = out10;

            outptr0 += out_hstep;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC++;
            pp += 2;
        }
    }
    for (; ii < max_ii; ii++)
    {
        pC = (const float*)C;
        float* outptr0 = (float*)top_blob + (size_t)j * out_hstep + i + ii;
        float c0 = 0.f;
        if (pC)
        {
            if (broadcast_type_C == 0)
            {
                float c = pC[0];
                if (beta != 1.f)
                    c *= beta;
                c0 = c;
            }
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                c0 = pC[i + ii];
                if (beta != 1.f)
                    c0 *= beta;
            }
            if (broadcast_type_C == 3)
                pC = (const float*)C + (i + ii) * c_hstep + j;
            if (broadcast_type_C == 4)
                pC = (const float*)C + j;
        }

        int jj = 0;
        for (; jj + 1 < max_jj; jj += 2)
        {
            float out00 = pp[0];
            float out01 = pp[1];

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    out00 += c0;
                    out01 += c0;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    out00 += c0;
                    out01 += c0;
                }
                if (broadcast_type_C == 3)
                {
                    out00 += beta == 1.f ? pC[0] : pC[0] * beta;
                    out01 += beta == 1.f ? pC[1] : pC[1] * beta;
                }
                if (broadcast_type_C == 4)
                {
                    out00 += beta == 1.f ? pC[0] : pC[0] * beta;
                    out01 += beta == 1.f ? pC[1] : pC[1] * beta;
                }
            }

            if (alpha != 1.f)
            {
                out00 *= alpha;
                out01 *= alpha;
            }

            outptr0[0] = out00;
            outptr0[out_hstep] = out01;

            outptr0 += out_hstep * 2;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC += 2;
            pp += 2;
        }
        for (; jj < max_jj; jj += 1)
        {
            float out00 = pp[0];

            if (pC)
            {
                if (broadcast_type_C == 0)
                {
                    out00 += c0;
                }
                if (broadcast_type_C == 1 || broadcast_type_C == 2)
                {
                    out00 += c0;
                }
                if (broadcast_type_C == 3)
                {
                    out00 += beta == 1.f ? pC[0] : pC[0] * beta;
                }
                if (broadcast_type_C == 4)
                {
                    out00 += beta == 1.f ? pC[0] : pC[0] * beta;
                }
            }

            if (alpha != 1.f)
            {
                out00 *= alpha;
            }

            outptr0[0] = out00;

            outptr0 += out_hstep;
            if (pC && (broadcast_type_C == 3 || broadcast_type_C == 4))
                pC++;
            pp += 1;
        }
    }
#endif // __ARM_NEON
}
