// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#if NCNN_RUNTIME_CPU && NCNN_ARM84BF16 && __aarch64__ && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
void cast_fp32_to_bf16_neon_bf16(const Mat& bottom_blob, Mat& top_blob, const Option& opt);
void cast_bf16_to_fp32_neon_bf16(const Mat& bottom_blob, Mat& top_blob, const Option& opt);
#endif

static void cast_fp32_to_bf16_neon(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM84BF16 && __aarch64__ && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
    if (ncnn::cpu_support_arm_bf16())
    {
        cast_fp32_to_bf16_neon_bf16(bottom_blob, top_blob, opt);
        return;
    }
#endif

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;

    const int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* ptr = bottom_blob.channel(q);
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        __bf16* outptr = top_blob.channel(q);
#else
        unsigned short* outptr = top_blob.channel(q);
#endif

        int i = 0;
#if __ARM_NEON
        for (; i + 15 < size; i += 16)
        {
#if NCNN_GNU_INLINE_ASM
#if __aarch64__
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            asm volatile(
                "prfm   pldl1keep, [%0, #512]   \n"
                "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
                "bfcvtn v0.4h, v0.4s            \n"
                "bfcvtn2 v0.8h, v1.4s           \n"
                "bfcvtn v1.4h, v2.4s            \n"
                "bfcvtn2 v1.8h, v3.4s           \n"
                "st1    {v0.8h, v1.8h}, [%1], #32 \n"
                : "=r"(ptr),   // %0
                "=r"(outptr) // %1
                : "0"(ptr),
                "1"(outptr)
                : "memory", "v0", "v1");
#else  // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            asm volatile(
                "prfm   pldl1keep, [%0, #512]   \n"
                "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
                "shrn   v0.4h, v0.4s, #16       \n"
                "shrn   v1.4h, v1.4s, #16       \n"
                "shrn   v2.4h, v2.4s, #16       \n"
                "shrn   v3.4h, v3.4s, #16       \n"
                "st1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%1], #32 \n"
                : "=r"(ptr),   // %0
                "=r"(outptr) // %1
                : "0"(ptr),
                "1"(outptr)
                : "memory", "v0", "v1", "v2", "v3");
#endif // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
#else  // __aarch64__
            asm volatile(
                "pld        [%0, #512]      \n"
                "vldm       %0!, {d0-d7}    \n"
                "vshrn.u32  d0, q0, #16     \n"
                "vshrn.u32  d1, q1, #16     \n"
                "vshrn.u32  d2, q2, #16     \n"
                "vshrn.u32  d3, q3, #16     \n"
                "vst1.u16   {d0-d3}, [%1]!  \n"
                : "=r"(ptr),   // %0
                "=r"(outptr) // %1
                : "0"(ptr),
                "1"(outptr)
                : "memory", "q0", "q1", "q2", "q3");
#endif // __aarch64__
#else  // NCNN_GNU_INLINE_ASM
            float32x4_t _p0_fp32 = vld1q_f32(ptr);
            float32x4_t _p1_fp32 = vld1q_f32(ptr + 4);
            float32x4_t _p2_fp32 = vld1q_f32(ptr + 8);
            float32x4_t _p3_fp32 = vld1q_f32(ptr + 12);
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            bfloat16x4_t _p0_bf16 = vcvt_bf16_f32(_p0_fp32);
            bfloat16x4_t _p1_bf16 = vcvt_bf16_f32(_p1_fp32);
            bfloat16x4_t _p2_bf16 = vcvt_bf16_f32(_p2_fp32);
            bfloat16x4_t _p3_bf16 = vcvt_bf16_f32(_p3_fp32);
            bfloat16x8_t _p_bf16 = vcombine_bf16(_p0_bf16, _p1_bf16);
            bfloat16x8_t _q_bf16 = vcombine_bf16(_p2_bf16, _p3_bf16);
            vst1q_bf16(outptr, _p_bf16);
            vst1q_bf16(outptr + 8, _q_bf16);
#else
            uint16x4_t _p0_bf16 = float2bfloat(_p0_fp32);
            uint16x4_t _p1_bf16 = float2bfloat(_p1_fp32);
            uint16x4_t _p2_bf16 = float2bfloat(_p2_fp32);
            uint16x4_t _p3_bf16 = float2bfloat(_p3_fp32);
            uint16x8_t _p_bf16 = vcombine_u16(_p0_bf16, _p1_bf16);
            uint16x8_t _q_bf16 = vcombine_u16(_p2_bf16, _p3_bf16);
            vst1q_u16(outptr, _p_bf16);
            vst1q_u16(outptr + 8, _q_bf16);
#endif
            ptr += 16;
            outptr += 16;
#endif // NCNN_GNU_INLINE_ASM
        }
        for (; i + 7 < size; i += 8)
        {
            float32x4_t _p0_fp32 = vld1q_f32(ptr);
            float32x4_t _p1_fp32 = vld1q_f32(ptr + 4);
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            bfloat16x4_t _p0_bf16 = vcvt_bf16_f32(_p0_fp32);
            bfloat16x4_t _p1_bf16 = vcvt_bf16_f32(_p1_fp32);
            bfloat16x8_t _p_bf16 = vcombine_bf16(_p0_bf16, _p1_bf16);
            vst1q_bf16(outptr, _p_bf16);
#else
            uint16x4_t _p0_bf16 = float2bfloat(_p0_fp32);
            uint16x4_t _p1_bf16 = float2bfloat(_p1_fp32);
            uint16x8_t _p_bf16 = vcombine_u16(_p0_bf16, _p1_bf16);
            vst1q_u16(outptr, _p_bf16);
#endif
            ptr += 8;
            outptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p_fp32 = vld1q_f32(ptr);
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            bfloat16x4_t _p_bf16 = vcvt_bf16_f32(_p_fp32);
            vst1_bf16(outptr, _p_bf16);
#else
            uint16x4_t _p_bf16 = float2bfloat(_p_fp32);
            vst1_u16(outptr, _p_bf16);
#endif
            ptr += 4;
            outptr += 4;
        }
#endif
        for (; i < size; i++)
        {
#if NCNN_GNU_INLINE_ASM && __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            asm volatile(
                "ldr    s0, [%0], #4    \n"
                "bfcvt  h0, s0          \n"
                "str    h0, [%1], #2    \n"
                : "=r"(ptr),   // %0
                "=r"(outptr) // %1
                : "0"(ptr),
                "1"(outptr)
                : "memory", "s0");
            // because intrinsic cause ndk clang crash
            // *outptr++ = vcvth_bf16_f32(*ptr++);
#else
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            *(unsigned short*)outptr = float32_to_bfloat16(*ptr++);
            outptr++;
#else
            *outptr++ = float32_to_bfloat16(*ptr++);
#endif
#endif
        }
    }
}

static void cast_bf16_to_fp32_neon(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_ARM84BF16 && __aarch64__ && !__ARM_FEATURE_BF16_VECTOR_ARITHMETIC
    if (ncnn::cpu_support_arm_bf16())
    {
        cast_bf16_to_fp32_neon_bf16(bottom_blob, top_blob, opt);
        return;
    }
#endif

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;

    const int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
        const __bf16* ptr = bottom_blob.channel(q);
#else
        const unsigned short* ptr = bottom_blob.channel(q);
#endif
        float* outptr = top_blob.channel(q);

        int i = 0;
#if __ARM_NEON
        for (; i + 15 < size; i += 16)
        {
#if NCNN_GNU_INLINE_ASM
#if __aarch64__
            asm volatile(
                "prfm   pldl1keep, [%0, #256]   \n"
                "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%0], #32 \n"
                "shll   v0.4s, v0.4h, #16       \n"
                "shll   v1.4s, v1.4h, #16       \n"
                "shll   v2.4s, v2.4h, #16       \n"
                "shll   v3.4s, v3.4h, #16       \n"
                "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%1], #64 \n"
                : "=r"(ptr),   // %0
                "=r"(outptr) // %1
                : "0"(ptr),
                "1"(outptr)
                : "memory", "v0", "v1", "v2", "v3");
#else  // __aarch64__
            asm volatile(
                "pld        [%0, #256]      \n"
                "vld1.u16   {d4-d7}, [%0]!  \n"
                "vshll.u16  q0, d4, #16     \n"
                "vshll.u16  q1, d5, #16     \n"
                "vshll.u16  q2, d6, #16     \n"
                "vshll.u16  q3, d7, #16     \n"
                "vstm       %1!, {d0-d7}    \n"
                : "=r"(ptr),   // %0
                "=r"(outptr) // %1
                : "0"(ptr),
                "1"(outptr)
                : "memory", "q0", "q1", "q2", "q3");
#endif // __aarch64__
#else  // NCNN_GNU_INLINE_ASM
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            bfloat16x8_t _p_bf16 = vld1q_bf16(ptr);
            bfloat16x8_t _q_bf16 = vld1q_bf16(ptr + 8);
            float32x4_t _p0_fp32 = vcvt_f32_bf16(vget_low_bf16(_p_bf16));
            float32x4_t _p1_fp32 = vcvt_f32_bf16(vget_high_bf16(_p_bf16));
            float32x4_t _p2_fp32 = vcvt_f32_bf16(vget_low_bf16(_q_bf16));
            float32x4_t _p3_fp32 = vcvt_f32_bf16(vget_high_bf16(_q_bf16));
#else
            uint16x8_t _p_bf16 = vld1q_u16(ptr);
            uint16x8_t _q_bf16 = vld1q_u16(ptr + 8);
            float32x4_t _p0_fp32 = bfloat2float(vget_low_u16(_p_bf16));
            float32x4_t _p1_fp32 = bfloat2float(vget_high_u16(_p_bf16));
            float32x4_t _p2_fp32 = bfloat2float(vget_low_u16(_q_bf16));
            float32x4_t _p3_fp32 = bfloat2float(vget_high_u16(_q_bf16));
#endif
            vst1q_f32(outptr, _p0_fp32);
            vst1q_f32(outptr + 4, _p1_fp32);
            vst1q_f32(outptr + 8, _p2_fp32);
            vst1q_f32(outptr + 12, _p3_fp32);
            ptr += 16;
            outptr += 16;
#endif // NCNN_GNU_INLINE_ASM
        }
        for (; i + 7 < size; i += 8)
        {
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            bfloat16x8_t _p_bf16 = vld1q_bf16(ptr);
            float32x4_t _p0_fp32 = vcvt_f32_bf16(vget_low_bf16(_p_bf16));
            float32x4_t _p1_fp32 = vcvt_f32_bf16(vget_high_bf16(_p_bf16));
#else
            uint16x8_t _p_bf16 = vld1q_u16(ptr);
            float32x4_t _p0_fp32 = bfloat2float(vget_low_u16(_p_bf16));
            float32x4_t _p1_fp32 = bfloat2float(vget_high_u16(_p_bf16));
#endif
            vst1q_f32(outptr, _p0_fp32);
            vst1q_f32(outptr + 4, _p1_fp32);
            ptr += 8;
            outptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            bfloat16x4_t _p_bf16 = vld1_bf16(ptr);
            float32x4_t _p_fp32 = vcvt_f32_bf16(_p_bf16);
#else
            uint16x4_t _p_bf16 = vld1_u16(ptr);
            float32x4_t _p_fp32 = bfloat2float(_p_bf16);
#endif
            vst1q_f32(outptr, _p_fp32);
            ptr += 4;
            outptr += 4;
        }
#endif
        for (; i < size; i++)
        {
#if __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
            *outptr++ = vcvtah_f32_bf16(*ptr++);
#else
            *outptr++ = bfloat16_to_float32(*ptr++);
#endif
        }
    }
}
