// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

static void padding_constant_pack8_int8_neon(const Mat& src, Mat& dst, int top, int bottom, int left, int right, int8x8_t v)
{
    const signed char* ptr = src;
    signed char* outptr = dst;

    int w = src.w;
    int h = src.h;

    int top_size = top * dst.w;
    int bottom_size = bottom * dst.w;

#if __aarch64__
    asm volatile(
        "mov    v0.8b, %10.8b           \n"
        "mov    v0.d[1], v0.d[0]        \n"
        "mov    v1.16b, v0.16b          \n"
        "mov    v2.16b, v0.16b          \n"
        "mov    v3.16b, v0.16b          \n"

        // fill top
        "lsr    w4, %w8, #3             \n" // w4 = nn = top_size >> 3
        "cmp    w4, #0                  \n"
        "beq    1f                      \n"

        "0:                             \n"
        "st1    {v0.16b, v1.16b, v2.16b, v3.16b}, [%0], #64 \n"
        "subs   w4, w4, #1              \n"
        "bne    0b                      \n"

        "1:                             \n"

        // fill top remain
        "and    w4, %w8, #7             \n" // w4 = remain = top_size & 7

        "cmp    w4, #4                  \n" // w4 >= 4
        "blt    2f                      \n"
        "sub    w4, w4, #4              \n"
        "st1    {v0.16b, v1.16b}, [%0], #32 \n"
        "2:                             \n"

        "cmp    w4, #2                  \n" // w4 >= 2
        "blt    3f                      \n"
        "sub    w4, w4, #2              \n"
        "st1    {v0.16b}, [%0], #16     \n"
        "3:                             \n"

        "cmp    w4, #0                  \n" // w4 > 0
        "beq    4f                      \n"
        "st1    {v0.8b}, [%0], #8       \n"
        "4:                             \n"

        // fill center h loop
        "cmp    %w5, #0                 \n"
        "beq    15f                     \n"
        "5:                             \n"

        // fill left
        "mov    w4, %w6                 \n" // w4 = left
        "cmp    w4, #0                  \n"
        "beq    7f                      \n"

        "6:                             \n"
        "st1    {v0.8b}, [%0], #8       \n"
        "subs   w4, w4, #1              \n"
        "bne    6b                      \n"

        "7:                             \n"

        // fill middle
        "lsr    w4, %w4, #3             \n" // w4 = nn = w >> 3
        "cmp    w4, #0                  \n"
        "beq    9f                      \n"

        "8:                             \n"
        "prfm   pldl1keep, [%1, #512]   \n"
        "ld1    {v16.16b, v17.16b, v18.16b, v19.16b}, [%1], #64 \n"
        "subs   w4, w4, #1              \n"
        "st1    {v16.16b, v17.16b, v18.16b, v19.16b}, [%0], #64 \n"
        "bne    8b                      \n"

        "9:                             \n"

        "and    w4, %w4, #7             \n" // w4 = remain = w & 7

        "cmp    w4, #4                  \n" // w4 >= 4
        "blt    10f                     \n"
        "prfm   pldl1keep, [%1, #256]   \n"
        "ld1    {v16.16b, v17.16b}, [%1], #32 \n"
        "sub    w4, w4, #4              \n"
        "st1    {v16.16b, v17.16b}, [%0], #32 \n"
        "10:                            \n"

        "cmp    w4, #2                  \n" // w4 >= 2
        "blt    11f                     \n"
        "prfm   pldl1keep, [%1, #128]   \n"
        "ld1    {v16.16b}, [%1], #16    \n"
        "sub    w4, w4, #2              \n"
        "st1    {v16.16b}, [%0], #16    \n"
        "11:                            \n"

        "cmp    w4, #0                  \n" // w4 > 0
        "beq    12f                     \n"
        "prfm   pldl1keep, [%1, #64]    \n"
        "ld1    {v16.8b}, [%1], #8      \n"
        "st1    {v16.8b}, [%0], #8      \n"
        "12:                            \n"

        // fill right
        "mov    w4, %w7                 \n" // w4 = right
        "cmp    w4, #0                  \n"
        "beq    14f                     \n"

        "13:                            \n"
        "subs   w4, w4, #1              \n"
        "st1    {v0.8b}, [%0], #8       \n"
        "bne    13b                     \n"
        "14:                            \n"

        "subs   %w5, %w5, #1            \n"
        "bne    5b                      \n"

        "15:                            \n"

        // fill bottom
        "lsr    w4, %w9, #3             \n" // w4 = nn = bottom_size >> 3
        "cmp    w4, #0                  \n"
        "beq    17f                     \n"

        "16:                            \n"
        "st1    {v0.16b, v1.16b, v2.16b, v3.16b}, [%0], #64 \n"
        "subs   w4, w4, #1              \n"
        "bne    16b                     \n"
        "17:                            \n"

        // fill bottom remain
        "and    w4, %w9, #7             \n" // w4 = remain = bottom_size & 7

        "cmp    w4, #4                  \n" // w4 >= 4
        "blt    18f                     \n"
        "sub    w4, w4, #4              \n"
        "st1    {v0.16b, v1.16b}, [%0], #32 \n"
        "18:                            \n"

        "cmp    w4, #2                  \n" // w4 >= 2
        "blt    19f                     \n"
        "sub    w4, w4, #2              \n"
        "st1    {v0.16b}, [%0], #16     \n"
        "19:                            \n"

        "cmp    w4, #0                  \n" // w4 > 0
        "beq    20f                     \n"
        "st1    {v0.8b}, [%0], #8       \n"
        "20:                            \n"

        : "=r"(outptr), // %0
        "=r"(ptr)     // %1
        : "0"(outptr),
        "1"(ptr),
        "r"(w),           // %4
        "r"(h),           // %5
        "r"(left),        // %6
        "r"(right),       // %7
        "r"(top_size),    // %8
        "r"(bottom_size), // %9
        "w"(v)            // %10
        : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v16", "v17", "v18", "v19");
#else  // __aarch64__
    asm volatile(
        "vmov       d0, %P10            \n"
        "vmov       d1, d0              \n"
        "vmov       q1, q0              \n"
        "vmov       q2, q0              \n"
        "vmov       q3, q0              \n"

        // fill top
        "lsr        r4, %8, #3          \n" // r4 = nn = top_size >> 3
        "cmp        r4, #0              \n"
        "beq        1f                  \n"

        "0:                             \n"
        "vstm       %0!, {d0-d7}        \n"
        "subs       r4, r4, #1          \n"
        "bne        0b                  \n"

        "1:                             \n"

        // fill top remain
        "and        r4, %8, #7          \n" // r4 = remain = top_size & 7

        "cmp        r4, #4              \n" // r4 >= 4
        "blt        2f                  \n"
        "sub        r4, r4, #4          \n"
        "vst1.s8    {d0-d3}, [%0 :128]! \n"
        "2:                             \n"

        "cmp        r4, #2              \n" // r4 >= 2
        "blt        3f                  \n"
        "sub        r4, r4, #2          \n"
        "vst1.s8    {d0-d1}, [%0 :128]! \n"
        "3:                             \n"

        "cmp        r4, #0              \n" // r4 > 0
        "beq        4f                  \n"
        "vst1.s8    {d0}, [%0 :64]!     \n"
        "4:                             \n"

        // fill center h loop
        "cmp        %5, #0              \n"
        "beq        15f                 \n"
        "5:                             \n"

        // fill left
        "mov        r4, %6              \n" // r4 = left
        "cmp        r4, #0              \n"
        "beq        7f                  \n"

        "6:                             \n"
        "vst1.s8    {d0}, [%0 :64]!     \n"
        "subs       r4, r4, #1          \n"
        "bne        6b                  \n"

        "7:                             \n"

        // fill middle
        "lsr        r4, %4, #3          \n" // r4 = nn = w >> 3
        "cmp        r4, #0              \n"
        "beq        9f                  \n"

        "8:                             \n"
        "pld        [%1, #512]          \n"
        "vldm       %1!, {d16-d23}      \n"
        "subs       r4, r4, #1          \n"
        "vstm       %0!, {d16-d23}      \n"
        "bne        8b                  \n"

        "9:                             \n"

        "and        r4, %4, #7          \n" // r4 = remain = w & 7

        "cmp        r4, #4              \n" // r4 >= 4
        "blt        10f                 \n"
        "pld        [%1, #256]          \n"
        "vld1.s8    {d16-d19}, [%1 :64]! \n"
        "sub        r4, r4, #4          \n"
        "vst1.s8    {d16-d19}, [%0 :64]! \n"
        "10:                            \n"

        "cmp        r4, #2              \n" // r4 >= 2
        "blt        11f                 \n"
        "pld        [%1, #128]          \n"
        "vld1.s8    {d16-d17}, [%1 :64]! \n"
        "sub        r4, r4, #2          \n"
        "vst1.s8    {d16-d17}, [%0 :64]! \n"
        "11:                            \n"

        "cmp        r4, #0              \n" // r4 > 0
        "beq        12f                 \n"
        "pld        [%1, #64]           \n"
        "vld1.s8    {d16}, [%1 :64]!    \n"
        "vst1.s8    {d16}, [%0 :64]!    \n"
        "12:                            \n"

        // fill right
        "mov        r4, %7              \n" // r4 = right
        "cmp        r4, #0              \n"
        "beq        14f                 \n"

        "13:                            \n"
        "subs       r4, r4, #1          \n"
        "vst1.s8    {d0}, [%0 :64]!     \n"
        "bne        13b                 \n"
        "14:                            \n"

        "subs       %5, %5, #1          \n"
        "bne        5b                  \n"

        "15:                            \n"

        // fill bottom
        "lsr        r4, %9, #3          \n" // r4 = nn = bottom_size >> 3
        "cmp        r4, #0              \n"
        "beq        17f                 \n"

        "16:                            \n"
        "vstm       %0!, {d0-d7}        \n"
        "subs       r4, r4, #1          \n"
        "bne        16b                 \n"
        "17:                            \n"

        // fill bottom remain
        "and        r4, %9, #7          \n" // r4 = remain = bottom_size & 7

        "cmp        r4, #4              \n" // r4 >= 4
        "blt        18f                 \n"
        "sub        r4, r4, #4          \n"
        "vst1.s8    {d0-d3}, [%0 :64]!  \n"
        "18:                            \n"

        "cmp        r4, #2              \n" // r4 >= 2
        "blt        19f                 \n"
        "sub        r4, r4, #2          \n"
        "vst1.s8    {d0-d1}, [%0 :64]!  \n"
        "19:                            \n"

        "cmp        r4, #0              \n" // r4 > 0
        "beq        20f                 \n"
        "vst1.s8    {d0}, [%0 :64]!     \n"
        "20:                            \n"

        : "=r"(outptr), // %0
        "=r"(ptr)     // %1
        : "0"(outptr),
        "1"(ptr),
        "r"(w),           // %4
        "r"(h),           // %5
        "r"(left),        // %6
        "r"(right),       // %7
        "r"(top_size),    // %8
        "r"(bottom_size), // %9
        "w"(v)            // %10
        : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11");
#endif // __aarch64__
}

static void padding_replicate_pack8_int8_neon(const Mat& src, Mat& dst, int top, int bottom, int left, int right)
{
    const signed char* ptr = src;
    signed char* outptr = dst;

    // fill top
    for (int y = 0; y < top; y++)
    {
        const signed char* ptr0 = ptr;
        int8x8_t _p = vld1_s8(ptr0);
        for (int x = 0; x < left; x++)
        {
            vst1_s8(outptr, _p);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            _p = vld1_s8(ptr0);
            vst1_s8(outptr, _p);
            ptr0 += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            vst1_s8(outptr, _p);
            outptr += 8;
        }
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        int8x8_t _p = vld1_s8(ptr);
        for (int x = 0; x < left; x++)
        {
            vst1_s8(outptr, _p);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            _p = vld1_s8(ptr);
            vst1_s8(outptr, _p);
            ptr += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            vst1_s8(outptr, _p);
            outptr += 8;
        }
    }
    // fill bottom
    ptr -= src.w * 8;
    for (int y = 0; y < bottom; y++)
    {
        const signed char* ptr0 = ptr;
        int8x8_t _p = vld1_s8(ptr0);
        for (int x = 0; x < left; x++)
        {
            vst1_s8(outptr, _p);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            _p = vld1_s8(ptr0);
            vst1_s8(outptr, _p);
            ptr0 += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            vst1_s8(outptr, _p);
            outptr += 8;
        }
    }
}

static void padding_reflect_pack8_int8_neon(const Mat& src, Mat& dst, int top, int bottom, int left, int right)
{
    const signed char* ptr = src;
    signed char* outptr = dst;

    // fill top
    ptr += top * src.w * 8;
    for (int y = 0; y < top; y++)
    {
        const signed char* ptr0 = ptr;
        for (int x = 0; x < left; x++)
        {
            int8x8_t _p = vld1_s8(ptr0 + (left - x) * 8);
            vst1_s8(outptr, _p);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            int8x8_t _p = vld1_s8(ptr0);
            vst1_s8(outptr, _p);
            ptr0 += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            int8x8_t _p = vld1_s8(ptr0 - 16 - x * 8);
            vst1_s8(outptr, _p);
            outptr += 8;
        }
        ptr -= src.w * 8;
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        for (int x = 0; x < left; x++)
        {
            int8x8_t _p = vld1_s8(ptr + (left - x) * 8);
            vst1_s8(outptr, _p);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            int8x8_t _p = vld1_s8(ptr);
            vst1_s8(outptr, _p);
            ptr += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            int8x8_t _p = vld1_s8(ptr - 16 - x * 8);
            vst1_s8(outptr, _p);
            outptr += 8;
        }
    }
    // fill bottom
    ptr -= 2 * src.w * 8;
    for (int y = 0; y < bottom; y++)
    {
        const signed char* ptr0 = ptr;
        for (int x = 0; x < left; x++)
        {
            int8x8_t _p = vld1_s8(ptr0 + (left - x) * 8);
            vst1_s8(outptr, _p);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            int8x8_t _p = vld1_s8(ptr0);
            vst1_s8(outptr, _p);
            ptr0 += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            int8x8_t _p = vld1_s8(ptr0 - 16 - x * 8);
            vst1_s8(outptr, _p);
            outptr += 8;
        }
        ptr -= src.w * 8;
    }
}
