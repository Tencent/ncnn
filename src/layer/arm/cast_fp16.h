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

#if NCNN_RUNTIME_CPU && NCNN_VFPV4 && __ARM_NEON && !(__ARM_FP & 2)
void cast_fp32_to_fp16_neon_vfpv4(const Mat& bottom_blob, Mat& top_blob, const Option& opt);
void cast_fp16_to_fp32_neon_vfpv4(const Mat& bottom_blob, Mat& top_blob, const Option& opt);
#endif

static void cast_fp32_to_fp16_neon(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_VFPV4 && __ARM_NEON && !(__ARM_FP & 2)
    if (ncnn::cpu_support_arm_vfpv4())
    {
        cast_fp32_to_fp16_neon_vfpv4(bottom_blob, top_blob, opt);
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
        unsigned short* outptr = top_blob.channel(q);

        int i = 0;
#if (__ARM_FP & 2)
        for (; i + 15 < size; i += 16)
        {
#if __aarch64__
            asm volatile(
                "prfm   pldl1keep, [%0, #512]   \n"
                "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
                "fcvtn  v0.4h, v0.4s            \n"
                "fcvtn  v1.4h, v1.4s            \n"
                "fcvtn  v2.4h, v2.4s            \n"
                "fcvtn  v3.4h, v3.4s            \n"
                "st1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%1], #32 \n"
                : "=r"(ptr),   // %0
                "=r"(outptr) // %1
                : "0"(ptr),
                "1"(outptr)
                : "memory", "v0", "v1", "v2", "v3");
#else  // __aarch64__
            asm volatile(
                "pld        [%0, #512]      \n"
                "vldm       %0!, {d0-d7}    \n"
                "vcvt.f16.f32 d0, q0        \n"
                "vcvt.f16.f32 d1, q1        \n"
                "vcvt.f16.f32 d2, q2        \n"
                "vcvt.f16.f32 d3, q3        \n"
                "vst1.u16   {d0-d3}, [%1 :128]! \n"
                : "=r"(ptr),   // %0
                "=r"(outptr) // %1
                : "0"(ptr),
                "1"(outptr)
                : "memory", "q0", "q1", "q2", "q3");
#endif // __aarch64__
        }
        for (; i + 7 < size; i += 8)
        {
            float32x4_t _p0_fp32 = vld1q_f32(ptr);
            float32x4_t _p1_fp32 = vld1q_f32(ptr + 4);
            float16x4_t _p0_fp16 = vcvt_f16_f32(_p0_fp32);
            float16x4_t _p1_fp16 = vcvt_f16_f32(_p1_fp32);
            uint16x8_t _p_fp16 = vcombine_u16(vreinterpret_u16_f16(_p0_fp16), vreinterpret_u16_f16(_p1_fp16));
            vst1q_u16(outptr, _p_fp16);
            ptr += 8;
            outptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float32x4_t _p_fp32 = vld1q_f32(ptr);
            float16x4_t _p_fp16 = vcvt_f16_f32(_p_fp32);
            vst1_u16(outptr, vreinterpret_u16_f16(_p_fp16));
            ptr += 4;
            outptr += 4;
        }
#endif
        for (; i < size; i++)
        {
            *outptr++ = float32_to_float16(*ptr++);
        }
    }
}

static void cast_fp16_to_fp32_neon(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
#if NCNN_VFPV4 && __ARM_NEON && !(__ARM_FP & 2)
    if (ncnn::cpu_support_arm_vfpv4())
    {
        cast_fp16_to_fp32_neon_vfpv4(bottom_blob, top_blob, opt);
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
        const unsigned short* ptr = bottom_blob.channel(q);
        float* outptr = top_blob.channel(q);

        int i = 0;
#if (__ARM_FP & 2)
        for (; i + 15 < size; i += 16)
        {
#if __aarch64__
            asm volatile(
                "prfm   pldl1keep, [%0, #256]   \n"
                "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%0], #32 \n"
                "fcvtl  v0.4s, v0.4h            \n"
                "fcvtl  v1.4s, v1.4h            \n"
                "fcvtl  v2.4s, v2.4h            \n"
                "fcvtl  v3.4s, v3.4h            \n"
                "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%1], #64 \n"
                : "=r"(ptr),   // %0
                "=r"(outptr) // %1
                : "0"(ptr),
                "1"(outptr)
                : "memory", "v0", "v1", "v2", "v3");
#else  // __aarch64__
            asm volatile(
                "pld        [%0, #256]      \n"
                "vld1.u16   {d4-d7}, [%0 :128]! \n"
                "vcvt.f32.f16 q0, d4        \n"
                "vcvt.f32.f16 q1, d5        \n"
                "vcvt.f32.f16 q2, d6        \n"
                "vcvt.f32.f16 q3, d7        \n"
                "vstm       %1!, {d0-d7}    \n"
                : "=r"(ptr),   // %0
                "=r"(outptr) // %1
                : "0"(ptr),
                "1"(outptr)
                : "memory", "q0", "q1", "q2", "q3");
#endif // __aarch64__
        }
        for (; i + 7 < size; i += 8)
        {
            uint16x8_t _p_fp16 = vld1q_u16(ptr);
            float16x4_t _p0_fp16 = vreinterpret_f16_u16(vget_low_u16(_p_fp16));
            float16x4_t _p1_fp16 = vreinterpret_f16_u16(vget_high_u16(_p_fp16));
            float32x4_t _p0_fp32 = vcvt_f32_f16(_p0_fp16);
            float32x4_t _p1_fp32 = vcvt_f32_f16(_p1_fp16);
            vst1q_f32(outptr, _p0_fp32);
            vst1q_f32(outptr + 4, _p1_fp32);
            ptr += 8;
            outptr += 8;
        }
        for (; i + 3 < size; i += 4)
        {
            float16x4_t _p_fp16 = vreinterpret_f16_u16(vld1_u16(ptr));
            float32x4_t _p_fp32 = vcvt_f32_f16(_p_fp16);
            vst1q_f32(outptr, _p_fp32);
            ptr += 4;
            outptr += 4;
        }
#endif
        for (; i < size; i++)
        {
            *outptr++ = float16_to_float32(*ptr++);
        }
    }
}
