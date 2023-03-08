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
                "prfm   pldl1keep, [%0, #512]       \n"
                "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
                "fcvtn  v0.4h, v0.4s                \n"
                "fcvtn  v1.4h, v1.4s                \n"
                "fcvtn  v2.4h, v2.4s                \n"
                "fcvtn  v3.4h, v3.4s                \n"
                "st1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%1], #32 \n"
                : "=r"(ptr),   // %0
                "=r"(outptr) // %1
                : "0"(ptr),
                "1"(outptr)
                : "memory", "v0", "v1", "v2", "v3");
#else  // __aarch64__
            asm volatile(
                "pld        [%0, #512]          \n"
                "vldm       %0!, {d0-d7}        \n"
                "vcvt.f16.f32 d0, q0            \n"
                "vcvt.f16.f32 d1, q1            \n"
                "vcvt.f16.f32 d2, q2            \n"
                "vcvt.f16.f32 d3, q3            \n"
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
            // This is originally implemented with neon fp16 intrinsics.
            // In the new version of gcc, __ARM_FP16_FORMAT_IEEE or __ARM_FP16_FORMAT_ALTERNATIVE needs to be defined to use the float16x4_t type.
            // That leads to compiler error when compiled with -mfpu=neon-vfpv4 but without -mfp16-format=ieee flag.
            // We could add more macro conditions to differentiate between old and new versions, but that's pretty ugly!
            // Just use all inline assembly here ~
            //          --- nihui
#if __aarch64__
            asm volatile(
                "ld1    {v0.4s, v1.4s}, [%0], #32   \n"
                "fcvtn  v0.4h, v0.4s                \n"
                "fcvtn  v1.4h, v1.4s                \n"
                "st1    {v0.4h, v1.4h}, [%1], #16   \n"
                : "=r"(ptr),   // %0
                "=r"(outptr) // %1
                : "0"(ptr),
                "1"(outptr)
                : "memory", "v0", "v1");
#else  // __aarch64__
            asm volatile(
                "vld1.f32   {d0-d3}, [%0]!  \n"
                "vcvt.f16.f32 d0, q0        \n"
                "vcvt.f16.f32 d1, q1        \n"
                "vst1.u16   {d0-d1}, [%1]!  \n"
                : "=r"(ptr),   // %0
                "=r"(outptr) // %1
                : "0"(ptr),
                "1"(outptr)
                : "memory", "q0", "q1");
#endif // __aarch64__
        }
        for (; i + 3 < size; i += 4)
        {
#if __aarch64__
            asm volatile(
                "ld1    {v0.4s}, [%0], #16  \n"
                "fcvtn  v0.4h, v0.4s        \n"
                "st1    {v0.4h}, [%1], #8   \n"
                : "=r"(ptr),   // %0
                "=r"(outptr) // %1
                : "0"(ptr),
                "1"(outptr)
                : "memory", "v0");
#else  // __aarch64__
            asm volatile(
                "vld1.f32   {d0-d1}, [%0]!  \n"
                "vcvt.f16.f32 d0, q0        \n"
                "vst1.u16   {d0}, [%1]!     \n"
                : "=r"(ptr),   // %0
                "=r"(outptr) // %1
                : "0"(ptr),
                "1"(outptr)
                : "memory", "q0");
#endif // __aarch64__
        }
#endif // (__ARM_FP & 2)
        for (; i < size; i++)
        {
            *outptr++ = float32_to_float16(*ptr++);
        }
    }
}

static void cast_fp16_to_fp32_neon(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_VFPV4 && __ARM_NEON && !(__ARM_FP & 2)
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
                "prfm   pldl1keep, [%0, #256]       \n"
                "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%0], #32 \n"
                "fcvtl  v0.4s, v0.4h                \n"
                "fcvtl  v1.4s, v1.4h                \n"
                "fcvtl  v2.4s, v2.4h                \n"
                "fcvtl  v3.4s, v3.4h                \n"
                "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%1], #64 \n"
                : "=r"(ptr),   // %0
                "=r"(outptr) // %1
                : "0"(ptr),
                "1"(outptr)
                : "memory", "v0", "v1", "v2", "v3");
#else  // __aarch64__
            asm volatile(
                "pld        [%0, #256]          \n"
                "vld1.u16   {d4-d7}, [%0 :128]! \n"
                "vcvt.f32.f16 q0, d4            \n"
                "vcvt.f32.f16 q1, d5            \n"
                "vcvt.f32.f16 q2, d6            \n"
                "vcvt.f32.f16 q3, d7            \n"
                "vstm       %1!, {d0-d7}        \n"
                : "=r"(ptr),   // %0
                "=r"(outptr) // %1
                : "0"(ptr),
                "1"(outptr)
                : "memory", "q0", "q1", "q2", "q3");
#endif // __aarch64__
        }
        for (; i + 7 < size; i += 8)
        {
#if __aarch64__
            asm volatile(
                "ld1    {v0.4h, v1.4h}, [%0], #16   \n"
                "fcvtl  v0.4s, v0.4h                \n"
                "fcvtl  v1.4s, v1.4h                \n"
                "st1    {v0.4s, v1.4s}, [%1], #32   \n"
                : "=r"(ptr),   // %0
                "=r"(outptr) // %1
                : "0"(ptr),
                "1"(outptr)
                : "memory", "v0", "v1");
#else  // __aarch64__
            asm volatile(
                "vld1.u16   {d4-d5}, [%0]!  \n"
                "vcvt.f32.f16 q0, d4        \n"
                "vcvt.f32.f16 q1, d5        \n"
                "vst1.f32   {d0-d3}, [%1]!  \n"
                : "=r"(ptr),   // %0
                "=r"(outptr) // %1
                : "0"(ptr),
                "1"(outptr)
                : "memory", "q0", "q1", "q2");
#endif // __aarch64__
        }
        for (; i + 3 < size; i += 4)
        {
#if __aarch64__
            asm volatile(
                "ld1    {v0.4h}, [%0], #8   \n"
                "fcvtl  v0.4s, v0.4h        \n"
                "st1    {v0.4s}, [%1], #16  \n"
                : "=r"(ptr),   // %0
                "=r"(outptr) // %1
                : "0"(ptr),
                "1"(outptr)
                : "memory", "v0");
#else  // __aarch64__
            asm volatile(
                "vld1.u16   {d2}, [%0]!     \n"
                "vcvt.f32.f16 q0, d2        \n"
                "vst1.f32   {d0-d1}, [%1]!  \n"
                : "=r"(ptr),   // %0
                "=r"(outptr) // %1
                : "0"(ptr),
                "1"(outptr)
                : "memory", "q0", "q1");
#endif // __aarch64__
        }
#endif // (__ARM_FP & 2)
        for (; i < size; i++)
        {
            *outptr++ = float16_to_float32(*ptr++);
        }
    }
}
