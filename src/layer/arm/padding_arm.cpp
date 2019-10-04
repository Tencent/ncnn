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

#include "padding_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn {

DEFINE_LAYER_CREATOR(Padding_arm)

Padding_arm::Padding_arm()
{
#if __ARM_NEON
    support_packing = true;
#endif // __ARM_NEON
}

#if __ARM_NEON
static void padding_constant_pack4_neon(const Mat& src, Mat& dst, int top, int bottom, int left, int right, float v)
{
    const float* ptr = src;
    float* outptr = dst;

    int w = src.w;
    int h = src.h;

    int top_size = top * dst.w;
    int bottom_size = bottom * dst.w;

#if __aarch64__
    asm volatile(
        "dup    v0.4s, %w10             \n"
        "dup    v1.4s, %w10             \n"
        "dup    v2.4s, %w10             \n"
        "dup    v3.4s, %w10             \n"

        // fill top
        "lsr    w4, %w8, #3             \n"// w4 = nn = top_size >> 3
        "cmp    w4, #0                  \n"
        "beq    1f                      \n"

        "0:                             \n"
        "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
        "subs   w4, w4, #1              \n"
        "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
        "bne    0b                      \n"

        "1:                             \n"

        // fill top remain
        "and    w4, %w8, #7             \n"// w4 = remain = top_size & 7

        "cmp    w4, #4                  \n"// w4 >= 4
        "blt    2f                      \n"
        "sub    w4, w4, #4              \n"
        "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
        "2:                             \n"

        "cmp    w4, #2                  \n"// w4 >= 2
        "blt    3f                      \n"
        "sub    w4, w4, #2              \n"
        "st1    {v0.4s, v1.4s}, [%0], #32 \n"
        "3:                             \n"

        "cmp    w4, #0                  \n"// w4 > 0
        "beq    4f                      \n"
        "st1    {v0.4s}, [%0], #16      \n"
        "4:                             \n"

        // fill center h loop
        "cmp    %w5, #0                 \n"
        "beq    15f                     \n"
        "5:                             \n"

        // fill left
        "mov    w4, %w6                 \n"// w4 = left
        "cmp    w4, #0                  \n"
        "beq    7f                      \n"

        "6:                             \n"
        "st1    {v0.4s}, [%0], #16      \n"
        "subs   w4, w4, #1              \n"
        "bne    6b                      \n"

        "7:                             \n"

        // fill middle
        "lsr    w4, %w4, #3             \n"// w4 = nn = w >> 3
        "cmp    w4, #0                  \n"
        "beq    9f                      \n"

        "8:                             \n"
        "prfm   pldl1keep, [%1, #512]   \n"
        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%1], #64 \n"
        "prfm   pldl1keep, [%1, #512]   \n"
        "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%1], #64 \n"
        "subs   w4, w4, #1              \n"
        "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
        "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
        "bne    8b                      \n"

        "9:                             \n"

        "and    w4, %w4, #7             \n"// w4 = remain = w & 7

        "cmp    w4, #4                  \n"// w4 >= 4
        "blt    10f                     \n"
        "prfm   pldl1keep, [%1, #512]   \n"
        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%1], #64 \n"
        "sub    w4, w4, #4              \n"
        "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
        "10:                            \n"

        "cmp    w4, #2                  \n"// w4 >= 2
        "blt    11f                     \n"
        "prfm   pldl1keep, [%1, #256]   \n"
        "ld1    {v16.4s, v17.4s}, [%1], #32 \n"
        "sub    w4, w4, #2              \n"
        "st1    {v16.4s, v17.4s}, [%0], #32 \n"
        "11:                            \n"

        "cmp    w4, #0                  \n"// w4 > 0
        "beq    12f                     \n"
        "prfm   pldl1keep, [%1, #128]   \n"
        "ld1    {v16.4s}, [%1], #16     \n"
        "st1    {v16.4s}, [%0], #16     \n"
        "12:                            \n"

        // fill right
        "mov    w4, %w7                 \n"// w4 = right
        "cmp    w4, #0                  \n"
        "beq    14f                     \n"

        "13:                            \n"
        "subs   w4, w4, #1              \n"
        "st1    {v0.4s}, [%0], #16      \n"
        "bne    13b                     \n"
        "14:                            \n"

        "subs   %w5, %w5, #1            \n"
        "bne    5b                      \n"

        "15:                            \n"

        // fill bottom
        "lsr    w4, %w9, #3             \n"// w4 = nn = bottom_size >> 3
        "cmp    w4, #0                  \n"
        "beq    17f                     \n"

        "16:                            \n"
        "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
        "subs   w4, w4, #1              \n"
        "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
        "bne    16b                     \n"
        "17:                            \n"

        // fill bottom remain
        "and    w4, %w9, #7             \n"// w4 = remain = bottom_size & 7

        "cmp    w4, #4                  \n"// w4 >= 4
        "blt    18f                     \n"
        "sub    w4, w4, #4              \n"
        "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
        "18:                            \n"

        "cmp    w4, #2                  \n"// w4 >= 2
        "blt    19f                     \n"
        "sub    w4, w4, #2              \n"
        "st1    {v0.4s, v1.4s}, [%0], #32 \n"
        "19:                            \n"

        "cmp    w4, #0                  \n"// w4 > 0
        "beq    20f                     \n"
        "st1    {v0.4s}, [%0], #16      \n"
        "20:                            \n"

        : "=r"(outptr),     // %0
          "=r"(ptr)         // %1
        : "0"(outptr),
          "1"(ptr),
          "r"(w),           // %4
          "r"(h),           // %5
          "r"(left),        // %6
          "r"(right),       // %7
          "r"(top_size),    // %8
          "r"(bottom_size), // %9
          "r"(v)            // %10
        : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"
    );
#else // __aarch64__
    asm volatile(
        "vdup.f32   q0, %10             \n"
        "vdup.f32   q1, %10             \n"
        "vdup.f32   q2, %10             \n"
        "vdup.f32   q3, %10             \n"

        // fill top
        "lsr        r4, %8, #3          \n"// r4 = nn = top_size >> 3
        "cmp        r4, #0              \n"
        "beq        1f                  \n"

        "0:                             \n"
        "vstm       %0!, {d0-d7}        \n"
        "subs       r4, r4, #1          \n"
        "vstm       %0!, {d0-d7}        \n"
        "bne        0b                  \n"

        "1:                             \n"

        // fill top remain
        "and        r4, %8, #7          \n"// r4 = remain = top_size & 7

        "cmp        r4, #4              \n"// r4 >= 4
        "blt        2f                  \n"
        "sub        r4, r4, #4          \n"
        "vstm       %0!, {d0-d7}        \n"
        "2:                             \n"

        "cmp        r4, #2              \n"// r4 >= 2
        "blt        3f                  \n"
        "sub        r4, r4, #2          \n"
        "vst1.f32   {d0-d3}, [%0 :128]! \n"
        "3:                             \n"

        "cmp        r4, #0              \n"// r4 > 0
        "beq        4f                  \n"
        "vst1.f32   {d0-d1}, [%0 :128]! \n"
        "4:                             \n"

        // fill center h loop
        "cmp        %5, #0              \n"
        "beq        15f                 \n"
        "5:                             \n"

        // fill left
        "mov        r4, %6              \n"// r4 = left
        "cmp        r4, #0              \n"
        "beq        7f                  \n"

        "6:                             \n"
        "vst1.f32   {d0-d1}, [%0 :128]! \n"
        "subs       r4, r4, #1          \n"
        "bne        6b                  \n"

        "7:                             \n"

        // fill middle
        "lsr        r4, %4, #3          \n"// r4 = nn = w >> 3
        "cmp        r4, #0              \n"
        "beq        9f                  \n"

        "8:                             \n"
        "pld        [%1, #512]          \n"
        "vldm       %1!, {d16-d23}      \n"
        "pld        [%1, #512]          \n"
        "vldm       %1!, {d24-d31}      \n"
        "subs       r4, r4, #1          \n"
        "vstm       %0!, {d16-d23}      \n"
        "vstm       %0!, {d24-d31}      \n"
        "bne        8b                  \n"

        "9:                             \n"

        "and        r4, %4, #7          \n"// r4 = remain = w & 7

        "cmp        r4, #4              \n"// r4 >= 4
        "blt        10f                 \n"
        "pld        [%1, #512]          \n"
        "vldm       %1!, {d16-d23}      \n"
        "sub        r4, r4, #4          \n"
        "vstm       %0!, {d16-d23}      \n"
        "10:                            \n"

        "cmp        r4, #2              \n"// r4 >= 2
        "blt        11f                 \n"
        "pld        [%1, #256]          \n"
        "vld1.f32   {d16-d19}, [%1 :128]! \n"
        "sub        r4, r4, #2          \n"
        "vst1.f32   {d16-d19}, [%0 :128]! \n"
        "11:                            \n"

        "cmp        r4, #0              \n"// r4 > 0
        "beq        12f                 \n"
        "pld        [%1, #128]          \n"
        "vld1.f32   {d16-d17}, [%1 :128]! \n"
        "vst1.f32   {d16-d17}, [%0 :128]! \n"
        "12:                            \n"

        // fill right
        "mov        r4, %7              \n"// r4 = right
        "cmp        r4, #0              \n"
        "beq        14f                 \n"

        "13:                            \n"
        "subs       r4, r4, #1          \n"
        "vst1.f32   {d0-d1}, [%0 :128]! \n"
        "bne        13b                 \n"
        "14:                            \n"

        "subs       %5, %5, #1          \n"
        "bne        5b                  \n"

        "15:                            \n"

        // fill bottom
        "lsr        r4, %9, #3          \n"// r4 = nn = bottom_size >> 3
        "cmp        r4, #0              \n"
        "beq        17f                 \n"

        "16:                            \n"
        "vstm       %0!, {d0-d7}        \n"
        "subs       r4, r4, #1          \n"
        "vstm       %0!, {d0-d7}        \n"
        "bne        16b                 \n"
        "17:                            \n"

        // fill bottom remain
        "and        r4, %9, #7          \n"// r4 = remain = bottom_size & 7

        "cmp        r4, #4              \n"// r4 >= 4
        "blt        18f                 \n"
        "sub        r4, r4, #4          \n"
        "vstm       %0!, {d0-d7}        \n"
        "18:                            \n"

        "cmp        r4, #2              \n"// r4 >= 2
        "blt        19f                 \n"
        "sub        r4, r4, #2          \n"
        "vst1.f32   {d0-d3}, [%0 :128]! \n"
        "19:                            \n"

        "cmp        r4, #0              \n"// r4 > 0
        "beq        20f                 \n"
        "vst1.f32   {d0-d1}, [%0 :128]! \n"
        "20:                            \n"

        : "=r"(outptr),     // %0
          "=r"(ptr)         // %1
        : "0"(outptr),
          "1"(ptr),
          "r"(w),           // %4
          "r"(h),           // %5
          "r"(left),        // %6
          "r"(right),       // %7
          "r"(top_size),    // %8
          "r"(bottom_size), // %9
          "r"(v)            // %10
        : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
    );
#endif // __aarch64__
}

static void padding_replicate_pack4_neon(const Mat& src, Mat& dst, int top, int bottom, int left, int right)
{
    const float* ptr = src;
    float* outptr = dst;

    // fill top
    for (int y = 0; y < top; y++)
    {
        const float* ptr0 = ptr;
        float32x4_t _p = vld1q_f32(ptr0);
        for (int x = 0; x < left; x++)
        {
            vst1q_f32(outptr, _p);
            outptr += 4;
        }
        for (int x = 0; x < src.w; x++)
        {
            _p = vld1q_f32(ptr0);
            vst1q_f32(outptr, _p);
            ptr0 += 4;
            outptr += 4;
        }
        for (int x = 0; x < right; x++)
        {
            vst1q_f32(outptr, _p);
            outptr += 4;
        }
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        float32x4_t _p = vld1q_f32(ptr);
        for (int x = 0; x < left; x++)
        {
            vst1q_f32(outptr, _p);
            outptr += 4;
        }
        for (int x = 0; x < src.w; x++)
        {
            _p = vld1q_f32(ptr);
            vst1q_f32(outptr, _p);
            ptr += 4;
            outptr += 4;
        }
        for (int x = 0; x < right; x++)
        {
            vst1q_f32(outptr, _p);
            outptr += 4;
        }
    }
    // fill bottom
    ptr -= src.w * 4;
    for (int y = 0; y < bottom; y++)
    {
        const float* ptr0 = ptr;
        float32x4_t _p = vld1q_f32(ptr0);
        for (int x = 0; x < left; x++)
        {
            vst1q_f32(outptr, _p);
            outptr += 4;
        }
        for (int x = 0; x < src.w; x++)
        {
            _p = vld1q_f32(ptr0);
            vst1q_f32(outptr, _p);
            ptr0 += 4;
            outptr += 4;
        }
        for (int x = 0; x < right; x++)
        {
            vst1q_f32(outptr, _p);
            outptr += 4;
        }
    }
}
#endif // __ARM_NEON

int Padding_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (top == 0 && bottom == 0 && left == 0 && right == 0)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

#if __ARM_NEON
    if (opt.use_packing_layout)
    {

    if (elempack == 4)
    {
        int outw = w + left + right;

        if (dims == 1)
        {
            top_blob.create(outw, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (type == 0)
                padding_constant_pack4_neon(bottom_blob, top_blob, 0, 0, left, right, value);
            else
                padding_replicate_pack4_neon(bottom_blob, top_blob, 0, 0, left, right);

            return 0;
        }

        int outh = h + top + bottom;

        if (dims == 2)
        {
            top_blob.create(outw, outh, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (type == 0)
                padding_constant_pack4_neon(bottom_blob, top_blob, top, bottom, left, right, value);
            else
                padding_replicate_pack4_neon(bottom_blob, top_blob, top, bottom, left, right);

            return 0;
        }

        if (dims == 3)
        {
            top_blob.create(outw, outh, channels, elemsize, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const Mat m = bottom_blob.channel(q);
                Mat borderm = top_blob.channel(q);

                if (type == 0)
                    padding_constant_pack4_neon(m, borderm, top, bottom, left, right, value);
                else
                    padding_replicate_pack4_neon(m, borderm, top, bottom, left, right);
            }

            return 0;
        }

        return 0;
    }

    } // opt.use_packing_layout
#endif // __ARM_NEON

    return Padding::forward(bottom_blob, top_blob, opt);
}

} // namespace ncnn
