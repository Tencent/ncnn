// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
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

static void padding_constant_pack8_fp16s_neon(const Mat& src, Mat& dst, int top, int bottom, int left, int right, uint16x8_t v)
{
    const unsigned short* ptr = src;
    unsigned short* outptr = dst;

    int w = src.w;
    int h = src.h;

    int top_size = top * dst.w;
    int bottom_size = bottom * dst.w;

#if NCNN_GNU_INLINE_ASM
    asm volatile(
        "mov    v0.16b, %10.16b         \n"
        "mov    v1.16b, %10.16b         \n"
        "mov    v2.16b, %10.16b         \n"
        "mov    v3.16b, %10.16b         \n"

        // fill top
        "lsr    w4, %w8, #2             \n" // w4 = nn = top_size >> 2
        "cmp    w4, #0                  \n"
        "beq    1f                      \n"

        "0:                             \n"
        "st1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%0], #64 \n"
        "subs   w4, w4, #1              \n"
        "bne    0b                      \n"

        "1:                             \n"

        // fill top remain
        "and    w4, %w8, #3             \n" // w4 = remain = top_size & 3

        "cmp    w4, #2                  \n" // w4 >= 2
        "blt    2f                      \n"
        "sub    w4, w4, #2              \n"
        "st1    {v0.8h, v1.8h}, [%0], #32 \n"
        "2:                             \n"

        "cmp    w4, #0                  \n" // w4 > 0
        "beq    3f                      \n"
        "st1    {v0.8h}, [%0], #16      \n"
        "3:                             \n"

        // fill center h loop
        "cmp    %w5, #0                 \n"
        "beq    13f                     \n"
        "4:                             \n"

        // fill left
        "mov    w4, %w6                 \n" // w4 = left
        "cmp    w4, #0                  \n"
        "beq    6f                      \n"

        "5:                             \n"
        "st1    {v0.8h}, [%0], #16      \n"
        "subs   w4, w4, #1              \n"
        "bne    5b                      \n"

        "6:                             \n"

        // fill middle
        "lsr    w4, %w4, #2             \n" // w4 = nn = w >> 2
        "cmp    w4, #0                  \n"
        "beq    8f                      \n"

        "7:                             \n"
        "prfm   pldl1keep, [%1, #512]   \n"
        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%1], #64 \n"
        "subs   w4, w4, #1              \n"
        "st1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%0], #64 \n"
        "bne    7b                      \n"

        "8:                             \n"

        "and    w4, %w4, #3             \n" // w4 = remain = w & 3

        "cmp    w4, #2                  \n" // w4 >= 2
        "blt    9f                      \n"
        "prfm   pldl1keep, [%1, #256]   \n"
        "ld1    {v16.8h, v17.8h}, [%1], #32 \n"
        "sub    w4, w4, #2              \n"
        "st1    {v16.8h, v17.8h}, [%0], #32 \n"
        "9:                             \n"

        "cmp    w4, #0                  \n" // w4 > 0
        "beq    10f                     \n"
        "prfm   pldl1keep, [%1, #128]   \n"
        "ld1    {v16.8h}, [%1], #16     \n"
        "st1    {v16.8h}, [%0], #16     \n"
        "10:                            \n"

        // fill right
        "mov    w4, %w7                 \n" // w4 = right
        "cmp    w4, #0                  \n"
        "beq    12f                     \n"

        "11:                            \n"
        "subs   w4, w4, #1              \n"
        "st1    {v0.8h}, [%0], #16      \n"
        "bne    11b                     \n"
        "12:                            \n"

        "subs   %w5, %w5, #1            \n"
        "bne    4b                      \n"

        "13:                            \n"

        // fill bottom
        "lsr    w4, %w9, #2             \n" // w4 = nn = bottom_size >> 2
        "cmp    w4, #0                  \n"
        "beq    15f                     \n"

        "14:                            \n"
        "st1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%0], #64 \n"
        "subs   w4, w4, #1              \n"
        "bne    14b                     \n"
        "15:                            \n"

        // fill bottom remain
        "and    w4, %w9, #3             \n" // w4 = remain = bottom_size & 3

        "cmp    w4, #2                  \n" // w4 >= 2
        "blt    16f                     \n"
        "sub    w4, w4, #2              \n"
        "st1    {v0.8h, v1.8h}, [%0], #32 \n"
        "16:                            \n"

        "cmp    w4, #0                  \n" // w4 > 0
        "beq    17f                     \n"
        "st1    {v0.8h}, [%0], #16      \n"
        "17:                            \n"

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
#else  // NCNN_GNU_INLINE_ASM

    // fill top
    {
        int x = 0;
        for (; x + 3 < top_size; x += 4)
        {
            vst1q_u16(outptr, v);
            vst1q_u16(outptr + 8, v);
            vst1q_u16(outptr + 16, v);
            vst1q_u16(outptr + 24, v);
            outptr += 32;
        }
        for (; x < top_size; x++)
        {
            vst1q_u16(outptr, v);
            outptr += 8;
        }
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        for (int x = 0; x < left; x++)
        {
            vst1q_u16(outptr, v);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            uint16x8_t _p = vld1q_u16(ptr);
            vst1q_u16(outptr, _p);
            ptr += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            vst1q_u16(outptr, v);
            outptr += 8;
        }
    }
    // fill bottom
    {
        int x = 0;
        for (; x + 3 < bottom_size; x += 4)
        {
            vst1q_u16(outptr, v);
            vst1q_u16(outptr + 8, v);
            vst1q_u16(outptr + 16, v);
            vst1q_u16(outptr + 24, v);
            outptr += 32;
        }
        for (; x < bottom_size; x++)
        {
            vst1q_u16(outptr, v);
            outptr += 8;
        }
    }
#endif // NCNN_GNU_INLINE_ASM
}

static void padding_replicate_pack8_fp16s_neon(const Mat& src, Mat& dst, int top, int bottom, int left, int right)
{
    const unsigned short* ptr = src;
    unsigned short* outptr = dst;

    // fill top
    for (int y = 0; y < top; y++)
    {
        const unsigned short* ptr0 = ptr;
        uint16x8_t _p = vld1q_u16(ptr0);
        for (int x = 0; x < left; x++)
        {
            vst1q_u16(outptr, _p);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            _p = vld1q_u16(ptr0);
            vst1q_u16(outptr, _p);
            ptr0 += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            vst1q_u16(outptr, _p);
            outptr += 8;
        }
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        uint16x8_t _p = vld1q_u16(ptr);
        for (int x = 0; x < left; x++)
        {
            vst1q_u16(outptr, _p);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            _p = vld1q_u16(ptr);
            vst1q_u16(outptr, _p);
            ptr += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            vst1q_u16(outptr, _p);
            outptr += 8;
        }
    }
    // fill bottom
    ptr -= src.w * 8;
    for (int y = 0; y < bottom; y++)
    {
        const unsigned short* ptr0 = ptr;
        uint16x8_t _p = vld1q_u16(ptr0);
        for (int x = 0; x < left; x++)
        {
            vst1q_u16(outptr, _p);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            _p = vld1q_u16(ptr0);
            vst1q_u16(outptr, _p);
            ptr0 += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            vst1q_u16(outptr, _p);
            outptr += 8;
        }
    }
}

static void padding_reflect_pack8_fp16s_neon(const Mat& src, Mat& dst, int top, int bottom, int left, int right)
{
    const unsigned short* ptr = src;
    unsigned short* outptr = dst;

    // fill top
    ptr += top * src.w * 8;
    for (int y = 0; y < top; y++)
    {
        const unsigned short* ptr0 = ptr;
        for (int x = 0; x < left; x++)
        {
            uint16x8_t _p = vld1q_u16(ptr0 + (left - x) * 8);
            vst1q_u16(outptr, _p);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            uint16x8_t _p = vld1q_u16(ptr0);
            vst1q_u16(outptr, _p);
            ptr0 += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            uint16x8_t _p = vld1q_u16(ptr0 - 16 - x * 8);
            vst1q_u16(outptr, _p);
            outptr += 8;
        }
        ptr -= src.w * 8;
    }
    // fill center
    for (int y = 0; y < src.h; y++)
    {
        for (int x = 0; x < left; x++)
        {
            uint16x8_t _p = vld1q_u16(ptr + (left - x) * 8);
            vst1q_u16(outptr, _p);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            uint16x8_t _p = vld1q_u16(ptr);
            vst1q_u16(outptr, _p);
            ptr += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            uint16x8_t _p = vld1q_u16(ptr - 16 - x * 8);
            vst1q_u16(outptr, _p);
            outptr += 8;
        }
    }
    // fill bottom
    ptr -= 2 * src.w * 8;
    for (int y = 0; y < bottom; y++)
    {
        const unsigned short* ptr0 = ptr;
        for (int x = 0; x < left; x++)
        {
            uint16x8_t _p = vld1q_u16(ptr0 + (left - x) * 8);
            vst1q_u16(outptr, _p);
            outptr += 8;
        }
        for (int x = 0; x < src.w; x++)
        {
            uint16x8_t _p = vld1q_u16(ptr0);
            vst1q_u16(outptr, _p);
            ptr0 += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++)
        {
            uint16x8_t _p = vld1q_u16(ptr0 - 16 - x * 8);
            vst1q_u16(outptr, _p);
            outptr += 8;
        }
        ptr -= src.w * 8;
    }
}
