// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "cast_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "cpu.h"

namespace ncnn {

#include "cast_fp16.h"

Cast_arm::Cast_arm()
{
    support_packing = true;
#if NCNN_ARM82
    support_fp16_storage = cpu_support_arm_asimdhp();
#endif

    support_bf16_storage = true;
}

int Cast_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (type_from == type_to)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    size_t out_elemsize = elemsize;
    if (type_to == 1)
    {
        if (type_from == 3)
        {
            Cast::forward(bottom_blob, top_blob, opt);
        }

        // float32
        out_elemsize = 4 * elempack;
    }
    else if (type_to == 2)
    {
        // float16
        out_elemsize = 2 * elempack;
    }
    else if (type_to == 3)
    {
        // int8
        out_elemsize = elempack;
    }
    else if (type_to == 4)
    {
        // bfloat16
        out_elemsize = 2 * elempack;
    }

    if (dims == 1)
    {
        top_blob.create(w, out_elemsize, elempack, opt.blob_allocator);
    }
    else if (dims == 2)
    {
        top_blob.create(w, h, out_elemsize, elempack, opt.blob_allocator);
    }
    else if (dims == 3)
    {
        top_blob.create(w, h, channels, out_elemsize, elempack, opt.blob_allocator);
    }
    else if (dims == 4)
    {
        top_blob.create(w, h, d, channels, out_elemsize, elempack, opt.blob_allocator);
    }
    if (top_blob.empty())
        return -100;

    int size = w * h * d * elempack;

    if (type_from == 1 && type_to == 2)
    {
        cast_fp32_to_fp16_neon(bottom_blob, top_blob, opt);
    }

    if (type_from == 2 && type_to == 1)
    {
        cast_fp16_to_fp32_neon(bottom_blob, top_blob, opt);
    }

    if (type_from == 3 && type_to == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const signed char* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                outptr[i] = (float)ptr[i];
            }
        }
    }

    if (type_from == 1 && type_to == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            unsigned short* outptr = top_blob.channel(q);

            int i = 0;
#if __ARM_NEON
            for (; i + 15 < size; i += 16)
            {
#if __aarch64__
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
#else  // __aarch64__
                asm volatile(
                    "pld        [%0, #512]      \n"
                    "vldm       %0!, {d0-d7}    \n"
                    "vshrn.u32  d0, q0, #16     \n"
                    "vshrn.u32  d1, q1, #16     \n"
                    "vshrn.u32  d2, q2, #16     \n"
                    "vshrn.u32  d3, q3, #16     \n"
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
                uint16x4_t _p0_fp16 = vcvt_bf16_f32(_p0_fp32);
                uint16x4_t _p1_fp16 = vcvt_bf16_f32(_p1_fp32);
                uint16x8_t _p_fp16 = vcombine_u16(_p0_fp16, _p1_fp16);
                vst1q_u16(outptr, _p_fp16);
                ptr += 8;
                outptr += 8;
            }
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _p_fp32 = vld1q_f32(ptr);
                uint16x4_t _p_fp16 = vcvt_bf16_f32(_p_fp32);
                vst1_u16(outptr, _p_fp16);
                ptr += 4;
                outptr += 4;
            }
#endif
            for (; i < size; i++)
            {
                *outptr++ = float32_to_bfloat16(*ptr++);
            }
        }
    }

    if (type_from == 4 && type_to == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const unsigned short* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);

            int i = 0;
#if __ARM_NEON
            for (; i + 15 < size; i += 16)
            {
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
                    "vld1.u16   {d4-d7}, [%0 :128]! \n"
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
            }
            for (; i + 7 < size; i += 8)
            {
                uint16x8_t _p_fp16 = vld1q_u16(ptr);
                float32x4_t _p0_fp32 = vcvt_f32_bf16(vget_low_u16(_p_fp16));
                float32x4_t _p1_fp32 = vcvt_f32_bf16(vget_high_u16(_p_fp16));
                vst1q_f32(outptr, _p0_fp32);
                vst1q_f32(outptr + 4, _p1_fp32);
                ptr += 8;
                outptr += 8;
            }
            for (; i + 3 < size; i += 4)
            {
                uint16x4_t _p_fp16 = vld1_u16(ptr);
                float32x4_t _p_fp32 = vcvt_f32_bf16(_p_fp16);
                vst1q_f32(outptr, _p_fp32);
                ptr += 4;
                outptr += 4;
            }
#endif
            for (; i < size; i++)
            {
                *outptr++ = bfloat16_to_float32(*ptr++);
            }
        }
    }

    return 0;
}

} // namespace ncnn
