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

#include "mat.h"
#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON
#include "platform.h"

namespace ncnn {

#if NCNN_PIXEL_ROTATE
// should be a kanna ascii art here in my local branch
// but we shall ask the original art author for permission first ...
// https://www.reddit.com/r/anime/comments/5uxjn4/i_recreated_the_kanna_ascii_art_from_kobayashisan/

static void kanna_rotate_1_c1(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int /*h*/, int stride)
{
    const int srcwgap = srcstride - srcw;
    const int wgap = stride - w;

    const unsigned char* src0 = src;
    const unsigned char* src1 = src + srcstride;
    unsigned char* dst0 = dst;
    unsigned char* dst1 = dst + stride;

    int y = 0;
    for (; y + 1 < srch; y += 2)
    {
#if __ARM_NEON
        int nn = srcw >> 5;
        int remain = srcw - (nn << 5);
#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x16_t _src0 = vld1q_u8(src0);
            uint8x16_t _src0n = vld1q_u8(src0 + 16);
            vst1q_u8(dst0, _src0);
            vst1q_u8(dst0 + 16, _src0n);

            uint8x16_t _src1 = vld1q_u8(src1);
            uint8x16_t _src1n = vld1q_u8(src1 + 16);
            vst1q_u8(dst1, _src1);
            vst1q_u8(dst1 + 16, _src1n);

            src0 += 32;
            src1 += 32;
            dst0 += 32;
            dst1 += 32;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "vld1.u8    {d0-d3}, [%1]!      \n"
                "pld        [%2, #256]          \n"
                "vld1.u8    {d4-d7}, [%2]!      \n"
                "subs       %0, #1              \n"
                "vst1.u8    {d0-d3}, [%3]!      \n"
                "vst1.u8    {d4-d7}, [%4]!      \n"
                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(src1), // %2
                "=r"(dst0), // %3
                "=r"(dst1)  // %4
                : "0"(nn),
                "1"(src0),
                "2"(src1),
                "3"(dst0),
                "4"(dst1)
                : "cc", "memory", "q0", "q1", "q2", "q3");
        }
#endif // __aarch64__
#else
        int remain = srcw;
#endif // __ARM_NEON

        for (; remain > 0; remain--)
        {
            *dst0++ = *src0++;
            *dst1++ = *src1++;
        }

        src0 += srcwgap + srcstride;
        src1 += srcwgap + srcstride;
        dst0 += wgap + stride;
        dst1 += wgap + stride;
    }

    for (; y < srch; y++)
    {
#if __ARM_NEON
        int nn = srcw >> 5;
        int remain = srcw - (nn << 5);
#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x16_t _src = vld1q_u8(src0);
            uint8x16_t _src2 = vld1q_u8(src0 + 16);
            vst1q_u8(dst0, _src);
            vst1q_u8(dst0 + 16, _src2);

            src0 += 32;
            dst0 += 32;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "vld1.u8    {d0-d3}, [%1]!      \n"
                "subs       %0, #1              \n"
                "vst1.u8    {d0-d3}, [%2]!      \n"
                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(dst0)  // %2
                : "0"(nn),
                "1"(src0),
                "2"(dst0)
                : "cc", "memory", "q0", "q1");
        }
#endif // __aarch64__
#else
        int remain = srcw;
#endif // __ARM_NEON

        for (; remain > 0; remain--)
        {
            *dst0++ = *src0++;
        }

        src0 += srcwgap;
        dst0 += wgap;
    }
}

static void kanna_rotate_1_c2(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int /*h*/, int stride)
{
    const int srcwgap = srcstride - srcw * 2;
    const int wgap = stride - w * 2;

    int size = srcw * 2;

    const unsigned char* src0 = src;
    const unsigned char* src1 = src + srcstride;
    unsigned char* dst0 = dst;
    unsigned char* dst1 = dst + stride;

    int y = 0;
    for (; y + 1 < srch; y += 2)
    {
#if __ARM_NEON
        int nn = size >> 5;
        int remain = size - (nn << 5);
#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x16_t _src0 = vld1q_u8(src0);
            uint8x16_t _src0n = vld1q_u8(src0 + 16);
            vst1q_u8(dst0, _src0);
            vst1q_u8(dst0 + 16, _src0n);

            uint8x16_t _src1 = vld1q_u8(src1);
            uint8x16_t _src1n = vld1q_u8(src1 + 16);
            vst1q_u8(dst1, _src1);
            vst1q_u8(dst1 + 16, _src1n);

            src0 += 32;
            src1 += 32;
            dst0 += 32;
            dst1 += 32;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "vld1.u8    {d0-d3}, [%1]!      \n"
                "pld        [%2, #256]          \n"
                "vld1.u8    {d4-d7}, [%2]!      \n"
                "subs       %0, #1              \n"
                "vst1.u8    {d0-d3}, [%3]!      \n"
                "vst1.u8    {d4-d7}, [%4]!      \n"
                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(src1), // %2
                "=r"(dst0), // %3
                "=r"(dst1)  // %4
                : "0"(nn),
                "1"(src0),
                "2"(src1),
                "3"(dst0),
                "4"(dst1)
                : "cc", "memory", "q0", "q1", "q2", "q3");
        }
#endif // __aarch64__
#else
        int remain = size;
#endif // __ARM_NEON

        for (; remain > 0; remain--)
        {
            *dst0++ = *src0++;
            *dst1++ = *src1++;
        }

        src0 += srcwgap + srcstride;
        src1 += srcwgap + srcstride;
        dst0 += wgap + stride;
        dst1 += wgap + stride;
    }

    for (; y < srch; y++)
    {
#if __ARM_NEON
        int nn = size >> 5;
        int remain = size - (nn << 5);
#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x16_t _src = vld1q_u8(src0);
            uint8x16_t _src2 = vld1q_u8(src0 + 16);
            vst1q_u8(dst0, _src);
            vst1q_u8(dst0 + 16, _src2);

            src0 += 32;
            dst0 += 32;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "vld1.u8    {d0-d3}, [%1]!      \n"
                "subs       %0, #1              \n"
                "vst1.u8    {d0-d3}, [%2]!      \n"
                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(dst0)  // %2
                : "0"(nn),
                "1"(src0),
                "2"(dst0)
                : "cc", "memory", "q0", "q1");
        }
#endif // __aarch64__
#else
        int remain = size;
#endif // __ARM_NEON

        for (; remain > 0; remain--)
        {
            *dst0++ = *src0++;
        }

        src0 += srcwgap;
        dst0 += wgap;
    }
}

static void kanna_rotate_1_c3(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int /*h*/, int stride)
{
    const int srcwgap = srcstride - srcw * 3;
    const int wgap = stride - w * 3;

    int size = srcw * 3;

    const unsigned char* src0 = src;
    const unsigned char* src1 = src + srcstride;
    unsigned char* dst0 = dst;
    unsigned char* dst1 = dst + stride;

    int y = 0;
    for (; y + 1 < srch; y += 2)
    {
#if __ARM_NEON
        int nn = size >> 5;
        int remain = size - (nn << 5);
#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x16_t _src0 = vld1q_u8(src0);
            uint8x16_t _src0n = vld1q_u8(src0 + 16);
            vst1q_u8(dst0, _src0);
            vst1q_u8(dst0 + 16, _src0n);

            uint8x16_t _src1 = vld1q_u8(src1);
            uint8x16_t _src1n = vld1q_u8(src1 + 16);
            vst1q_u8(dst1, _src1);
            vst1q_u8(dst1 + 16, _src1n);

            src0 += 32;
            src1 += 32;
            dst0 += 32;
            dst1 += 32;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "vld1.u8    {d0-d3}, [%1]!      \n"
                "pld        [%2, #256]          \n"
                "vld1.u8    {d4-d7}, [%2]!      \n"
                "subs       %0, #1              \n"
                "vst1.u8    {d0-d3}, [%3]!      \n"
                "vst1.u8    {d4-d7}, [%4]!      \n"
                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(src1), // %2
                "=r"(dst0), // %3
                "=r"(dst1)  // %4
                : "0"(nn),
                "1"(src0),
                "2"(src1),
                "3"(dst0),
                "4"(dst1)
                : "cc", "memory", "q0", "q1", "q2", "q3");
        }
#endif // __aarch64__
#else
        int remain = size;
#endif // __ARM_NEON

        for (; remain > 0; remain--)
        {
            *dst0++ = *src0++;
            *dst1++ = *src1++;
        }

        src0 += srcwgap + srcstride;
        src1 += srcwgap + srcstride;
        dst0 += wgap + stride;
        dst1 += wgap + stride;
    }

    for (; y < srch; y++)
    {
#if __ARM_NEON
        int nn = size >> 5;
        int remain = size - (nn << 5);
#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x16_t _src = vld1q_u8(src0);
            uint8x16_t _src2 = vld1q_u8(src0 + 16);
            vst1q_u8(dst0, _src);
            vst1q_u8(dst0 + 16, _src2);

            src0 += 32;
            dst0 += 32;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "vld1.u8    {d0-d3}, [%1]!      \n"
                "subs       %0, #1              \n"
                "vst1.u8    {d0-d3}, [%2]!      \n"
                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(dst0)  // %2
                : "0"(nn),
                "1"(src0),
                "2"(dst0)
                : "cc", "memory", "q0", "q1");
        }
#endif // __aarch64__
#else
        int remain = size;
#endif // __ARM_NEON

        for (; remain > 0; remain--)
        {
            *dst0++ = *src0++;
        }

        src0 += srcwgap;
        dst0 += wgap;
    }
}

static void kanna_rotate_1_c4(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int /*h*/, int stride)
{
    const int srcwgap = srcstride - srcw * 4;
    const int wgap = stride - w * 4;

    int size = srcw * 4;

    const unsigned char* src0 = src;
    const unsigned char* src1 = src + srcstride;
    unsigned char* dst0 = dst;
    unsigned char* dst1 = dst + stride;

    int y = 0;
    for (; y + 1 < srch; y += 2)
    {
#if __ARM_NEON
        int nn = size >> 5;
        int remain = size - (nn << 5);
#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x16_t _src0 = vld1q_u8(src0);
            uint8x16_t _src0n = vld1q_u8(src0 + 16);
            vst1q_u8(dst0, _src0);
            vst1q_u8(dst0 + 16, _src0n);

            uint8x16_t _src1 = vld1q_u8(src1);
            uint8x16_t _src1n = vld1q_u8(src1 + 16);
            vst1q_u8(dst1, _src1);
            vst1q_u8(dst1 + 16, _src1n);

            src0 += 32;
            src1 += 32;
            dst0 += 32;
            dst1 += 32;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "vld1.u8    {d0-d3}, [%1]!      \n"
                "pld        [%2, #256]          \n"
                "vld1.u8    {d4-d7}, [%2]!      \n"
                "subs       %0, #1              \n"
                "vst1.u8    {d0-d3}, [%3]!      \n"
                "vst1.u8    {d4-d7}, [%4]!      \n"
                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(src1), // %2
                "=r"(dst0), // %3
                "=r"(dst1)  // %4
                : "0"(nn),
                "1"(src0),
                "2"(src1),
                "3"(dst0),
                "4"(dst1)
                : "cc", "memory", "q0", "q1", "q2", "q3");
        }
#endif // __aarch64__
#else
        int remain = size;
#endif // __ARM_NEON

        for (; remain > 0; remain--)
        {
            *dst0++ = *src0++;
            *dst1++ = *src1++;
        }

        src0 += srcwgap + srcstride;
        src1 += srcwgap + srcstride;
        dst0 += wgap + stride;
        dst1 += wgap + stride;
    }

    for (; y < srch; y++)
    {
#if __ARM_NEON
        int nn = size >> 5;
        int remain = size - (nn << 5);
#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x16_t _src = vld1q_u8(src0);
            uint8x16_t _src2 = vld1q_u8(src0 + 16);
            vst1q_u8(dst0, _src);
            vst1q_u8(dst0 + 16, _src2);

            src0 += 32;
            dst0 += 32;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "vld1.u8    {d0-d3}, [%1]!      \n"
                "subs       %0, #1              \n"
                "vst1.u8    {d0-d3}, [%2]!      \n"
                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(dst0)  // %2
                : "0"(nn),
                "1"(src0),
                "2"(dst0)
                : "cc", "memory", "q0", "q1");
        }
#endif // __aarch64__
#else
        int remain = size;
#endif // __ARM_NEON

        for (; remain > 0; remain--)
        {
            *dst0++ = *src0++;
        }

        src0 += srcwgap;
        dst0 += wgap;
    }
}

static void kanna_rotate_2_c1(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int /*h*/, int stride)
{
    const int srcwgap = srcstride - srcw;
    const int wgap = stride + w;

    const unsigned char* src0 = src;
    unsigned char* dst0 = dst + w - 1;

    int y = 0;
    for (; y < srch; y++)
    {
#if __ARM_NEON
        dst0 -= 15;

        int nn = srcw >> 4;
        int remain = srcw - (nn << 4);

#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x8_t _src = vld1_u8(src0);
            uint8x8_t _src2 = vld1_u8(src0 + 8);

            _src = vrev64_u8(_src);
            _src2 = vrev64_u8(_src2);

            vst1_u8(dst0, _src2);
            vst1_u8(dst0 + 8, _src);

            src0 += 16;
            dst0 -= 16;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "mov        r4, #-16            \n"
                "0:                             \n"
                "pld        [%1, #128]          \n"
                "vld1.u8    {d0-d1}, [%1]!      \n"
                "vrev64.u8  d3, d0              \n"
                "vrev64.u8  d2, d1              \n"
                "subs       %0, #1              \n"
                "vst1.u8    {d2-d3}, [%2], r4   \n"
                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(dst0)  // %2
                : "0"(nn),
                "1"(src0),
                "2"(dst0)
                : "cc", "memory", "q0", "q1", "r4");
        }
#endif // __aarch64__

        dst0 += 15;
#else
        int remain = srcw;
#endif // __ARM_NEON

        for (; remain > 0; remain--)
        {
            *dst0 = *src0;

            src0 += 1;
            dst0 -= 1;
        }

        src0 += srcwgap;
        dst0 += wgap;
    }
}

static void kanna_rotate_2_c2(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int /*h*/, int stride)
{
    const int srcwgap = srcstride - srcw * 2;
    const int wgap = stride + w * 2;

    const unsigned char* src0 = src;
    unsigned char* dst0 = dst + w * 2 - 2;

    int y = 0;
    for (; y < srch; y++)
    {
#if __ARM_NEON
        dst0 -= 7 * 2;

        int nn = srcw >> 4;
        int remain = srcw - (nn << 4);

#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x8x2_t _src = vld2_u8(src0);
            uint8x8x2_t _src2 = vld2_u8(src0 + 8 * 2);

            _src.val[0] = vrev64_u8(_src.val[0]);
            _src.val[1] = vrev64_u8(_src.val[1]);

            _src2.val[0] = vrev64_u8(_src2.val[0]);
            _src2.val[1] = vrev64_u8(_src2.val[1]);

            vst2_u8(dst0, _src);
            vst2_u8(dst0 - 8 * 2, _src2);

            src0 += 16 * 2;
            dst0 -= 16 * 2;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "mov        r4, #-16            \n"
                "0:                             \n"
                "pld        [%1, #128]          \n"
                "vld2.u8    {d0-d1}, [%1]!      \n"
                "vrev64.u8  d0, d0              \n"
                "pld        [%1, #128]          \n"
                "vld2.u8    {d2-d3}, [%1]!      \n"
                "vrev64.u8  d1, d1              \n"
                "vrev64.u8  d2, d2              \n"
                "vst2.u8    {d0-d1}, [%2], r4   \n"
                "vrev64.u8  d3, d3              \n"
                "subs       %0, #1              \n"
                "vst2.u8    {d2-d3}, [%2], r4   \n"
                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(dst0)  // %2
                : "0"(nn),
                "1"(src0),
                "2"(dst0)
                : "cc", "memory", "q0", "q1", "r4");
        }
#endif // __aarch64__

        dst0 += 7 * 2;
#else
        int remain = srcw;
#endif // __ARM_NEON

        for (; remain > 0; remain--)
        {
            dst0[0] = src0[0];
            dst0[1] = src0[1];

            src0 += 2;
            dst0 -= 2;
        }

        src0 += srcwgap;
        dst0 += wgap;
    }
}

static void kanna_rotate_2_c3(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int /*h*/, int stride)
{
    const int srcwgap = srcstride - srcw * 3;
    const int wgap = stride + w * 3;

    const unsigned char* src0 = src;
    unsigned char* dst0 = dst + w * 3 - 3;

    int y = 0;
    for (; y < srch; y++)
    {
#if __ARM_NEON
        dst0 -= 7 * 3;

        int nn = srcw >> 4;
        int remain = srcw - (nn << 4);

#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x8x3_t _src = vld3_u8(src0);
            uint8x8x3_t _src2 = vld3_u8(src0 + 8 * 3);

            _src.val[0] = vrev64_u8(_src.val[0]);
            _src.val[1] = vrev64_u8(_src.val[1]);
            _src.val[2] = vrev64_u8(_src.val[2]);

            _src2.val[0] = vrev64_u8(_src2.val[0]);
            _src2.val[1] = vrev64_u8(_src2.val[1]);
            _src2.val[2] = vrev64_u8(_src2.val[2]);

            vst3_u8(dst0, _src);
            vst3_u8(dst0 - 8 * 3, _src2);

            src0 += 16 * 3;
            dst0 -= 16 * 3;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "mov        r4, #-24            \n"
                "0:                             \n"
                "pld        [%1, #192]          \n"
                "vld3.u8    {d0-d2}, [%1]!      \n"
                "vrev64.u8  d0, d0              \n"
                "vrev64.u8  d1, d1              \n"
                "pld        [%1, #192]          \n"
                "vld3.u8    {d4-d6}, [%1]!      \n"
                "vrev64.u8  d2, d2              \n"
                "vrev64.u8  d4, d4              \n"
                "vst3.u8    {d0-d2}, [%2], r4   \n"
                "vrev64.u8  d5, d5              \n"
                "vrev64.u8  d6, d6              \n"
                "subs       %0, #1              \n"
                "vst3.u8    {d4-d6}, [%2], r4   \n"
                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(dst0)  // %2
                : "0"(nn),
                "1"(src0),
                "2"(dst0)
                : "cc", "memory", "q0", "q1", "q2", "q3", "r4");
        }
#endif // __aarch64__

        dst0 += 7 * 3;
#else
        int remain = srcw;
#endif // __ARM_NEON

        for (; remain > 0; remain--)
        {
            dst0[0] = src0[0];
            dst0[1] = src0[1];
            dst0[2] = src0[2];

            src0 += 3;
            dst0 -= 3;
        }

        src0 += srcwgap;
        dst0 += wgap;
    }
}

static void kanna_rotate_2_c4(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int /*h*/, int stride)
{
    const int srcwgap = srcstride - srcw * 4;
    const int wgap = stride + w * 4;

    const unsigned char* src0 = src;
    unsigned char* dst0 = dst + w * 4 - 4;

    int y = 0;
    for (; y < srch; y++)
    {
#if __ARM_NEON
        dst0 -= 7 * 4;

        int nn = srcw >> 4;
        int remain = srcw - (nn << 4);

#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x8x4_t _src = vld4_u8(src0);
            uint8x8x4_t _src2 = vld4_u8(src0 + 8 * 4);

            _src.val[0] = vrev64_u8(_src.val[0]);
            _src.val[1] = vrev64_u8(_src.val[1]);
            _src.val[2] = vrev64_u8(_src.val[2]);
            _src.val[3] = vrev64_u8(_src.val[3]);

            _src2.val[0] = vrev64_u8(_src2.val[0]);
            _src2.val[1] = vrev64_u8(_src2.val[1]);
            _src2.val[2] = vrev64_u8(_src2.val[2]);
            _src2.val[3] = vrev64_u8(_src2.val[3]);

            vst4_u8(dst0, _src);
            vst4_u8(dst0 - 8 * 4, _src2);

            src0 += 16 * 4;
            dst0 -= 16 * 4;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "mov        r4, #-32            \n"
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "vld4.u8    {d0-d3}, [%1]!      \n"
                "vrev64.u8  d0, d0              \n"
                "vrev64.u8  d1, d1              \n"
                "vrev64.u8  d2, d2              \n"
                "pld        [%1, #256]          \n"
                "vld4.u8    {d4-d7}, [%1]!      \n"
                "vrev64.u8  d3, d3              \n"
                "vrev64.u8  d4, d4              \n"
                "vrev64.u8  d5, d5              \n"
                "vst4.u8    {d0-d3}, [%2], r4   \n"
                "vrev64.u8  d6, d6              \n"
                "vrev64.u8  d7, d7              \n"
                "subs       %0, #1              \n"
                "vst4.u8    {d4-d7}, [%2], r4   \n"
                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(dst0)  // %2
                : "0"(nn),
                "1"(src0),
                "2"(dst0)
                : "cc", "memory", "q0", "q1", "q2", "q3", "r4");
        }
#endif // __aarch64__

        dst0 += 7 * 4;
#else
        int remain = srcw;
#endif // __ARM_NEON

        for (; remain > 0; remain--)
        {
            dst0[0] = src0[0];
            dst0[1] = src0[1];
            dst0[2] = src0[2];
            dst0[3] = src0[3];

            src0 += 4;
            dst0 -= 4;
        }

        src0 += srcwgap;
        dst0 += wgap;
    }
}

static void kanna_rotate_3_c1(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    const int srcwgap = srcstride - srcw;
    const int wgap = stride - w;

    // point to the last dst pixel
    unsigned char* dstend = dst + stride * h - wgap;

    const unsigned char* src0 = src;
    unsigned char* dst0 = dstend - 1;

    int y = 0;
    for (; y < srch; y++)
    {
#if __ARM_NEON
        dst0 -= 15;

        int nn = srcw >> 4;
        int remain = srcw - (nn << 4);

#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x8_t _src = vld1_u8(src0);
            uint8x8_t _src2 = vld1_u8(src0 + 8);

            _src = vrev64_u8(_src);
            _src2 = vrev64_u8(_src2);

            vst1_u8(dst0, _src2);
            vst1_u8(dst0 + 8, _src);

            src0 += 16;
            dst0 -= 16;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "mov        r4, #-16            \n"
                "0:                             \n"
                "pld        [%1, #128]          \n"
                "vld1.u8    {d0-d1}, [%1]!      \n"
                "vrev64.u8  d3, d0              \n"
                "vrev64.u8  d2, d1              \n"
                "subs       %0, #1              \n"
                "vst1.u8    {d2-d3}, [%2], r4   \n"
                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(dst0)  // %2
                : "0"(nn),
                "1"(src0),
                "2"(dst0)
                : "cc", "memory", "q0", "q1", "r4");
        }
#endif // __aarch64__

        dst0 += 15;
#else
        int remain = srcw;
#endif // __ARM_NEON

        for (; remain > 0; remain--)
        {
            *dst0 = *src0;

            src0 += 1;
            dst0 -= 1;
        }

        src0 += srcwgap;
        dst0 -= wgap;
    }
}

static void kanna_rotate_3_c2(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    const int srcwgap = srcstride - srcw * 2;
    const int wgap = stride - w * 2;

    // point to the last dst pixel
    unsigned char* dstend = dst + stride * h - wgap;

    const unsigned char* src0 = src;
    unsigned char* dst0 = dstend - 2;

    int y = 0;
    for (; y < srch; y++)
    {
#if __ARM_NEON
        dst0 -= 7 * 2;

        int nn = srcw >> 4;
        int remain = srcw - (nn << 4);

#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x8x2_t _src = vld2_u8(src0);
            uint8x8x2_t _src2 = vld2_u8(src0 + 8 * 2);

            _src.val[0] = vrev64_u8(_src.val[0]);
            _src.val[1] = vrev64_u8(_src.val[1]);

            _src2.val[0] = vrev64_u8(_src2.val[0]);
            _src2.val[1] = vrev64_u8(_src2.val[1]);

            vst2_u8(dst0, _src);
            vst2_u8(dst0 - 8 * 2, _src2);

            src0 += 16 * 2;
            dst0 -= 16 * 2;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "mov        r4, #-16            \n"
                "0:                             \n"
                "pld        [%1, #128]          \n"
                "vld2.u8    {d0-d1}, [%1]!      \n"
                "vrev64.u8  d0, d0              \n"
                "pld        [%1, #128]          \n"
                "vld2.u8    {d2-d3}, [%1]!      \n"
                "vrev64.u8  d1, d1              \n"
                "vrev64.u8  d2, d2              \n"
                "vst2.u8    {d0-d1}, [%2], r4   \n"
                "vrev64.u8  d3, d3              \n"
                "subs       %0, #1              \n"
                "vst2.u8    {d2-d3}, [%2], r4   \n"
                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(dst0)  // %2
                : "0"(nn),
                "1"(src0),
                "2"(dst0)
                : "cc", "memory", "q0", "q1", "r4");
        }
#endif // __aarch64__

        dst0 += 7 * 2;
#else
        int remain = srcw;
#endif // __ARM_NEON

        for (; remain > 0; remain--)
        {
            dst0[0] = src0[0];
            dst0[1] = src0[1];

            src0 += 2;
            dst0 -= 2;
        }

        src0 += srcwgap;
        dst0 -= wgap;
    }
}

static void kanna_rotate_3_c3(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    const int srcwgap = srcstride - srcw * 3;
    const int wgap = stride - w * 3;

    // point to the last dst pixel
    unsigned char* dstend = dst + stride * h - wgap;

    const unsigned char* src0 = src;
    unsigned char* dst0 = dstend - 3;

    int y = 0;
    for (; y < srch; y++)
    {
#if __ARM_NEON
        dst0 -= 7 * 3;

        int nn = srcw >> 4;
        int remain = srcw - (nn << 4);

#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x8x3_t _src = vld3_u8(src0);
            uint8x8x3_t _src2 = vld3_u8(src0 + 8 * 3);

            _src.val[0] = vrev64_u8(_src.val[0]);
            _src.val[1] = vrev64_u8(_src.val[1]);
            _src.val[2] = vrev64_u8(_src.val[2]);

            _src2.val[0] = vrev64_u8(_src2.val[0]);
            _src2.val[1] = vrev64_u8(_src2.val[1]);
            _src2.val[2] = vrev64_u8(_src2.val[2]);

            vst3_u8(dst0, _src);
            vst3_u8(dst0 - 8 * 3, _src2);

            src0 += 16 * 3;
            dst0 -= 16 * 3;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "mov        r4, #-24            \n"
                "0:                             \n"
                "pld        [%1, #192]          \n"
                "vld3.u8    {d0-d2}, [%1]!      \n"
                "vrev64.u8  d0, d0              \n"
                "vrev64.u8  d1, d1              \n"
                "pld        [%1, #192]          \n"
                "vld3.u8    {d4-d6}, [%1]!      \n"
                "vrev64.u8  d2, d2              \n"
                "vrev64.u8  d4, d4              \n"
                "vst3.u8    {d0-d2}, [%2], r4   \n"
                "vrev64.u8  d5, d5              \n"
                "vrev64.u8  d6, d6              \n"
                "subs       %0, #1              \n"
                "vst3.u8    {d4-d6}, [%2], r4   \n"
                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(dst0)  // %2
                : "0"(nn),
                "1"(src0),
                "2"(dst0)
                : "cc", "memory", "q0", "q1", "q2", "q3", "r4");
        }
#endif // __aarch64__

        dst0 += 7 * 3;
#else
        int remain = srcw;
#endif // __ARM_NEON

        for (; remain > 0; remain--)
        {
            dst0[0] = src0[0];
            dst0[1] = src0[1];
            dst0[2] = src0[2];

            src0 += 3;
            dst0 -= 3;
        }

        src0 += srcwgap;
        dst0 -= wgap;
    }
}

static void kanna_rotate_3_c4(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    const int srcwgap = srcstride - srcw * 4;
    const int wgap = stride - w * 4;

    // point to the last dst pixel
    unsigned char* dstend = dst + stride * h - wgap;

    const unsigned char* src0 = src;
    unsigned char* dst0 = dstend - 4;

    int y = 0;
    for (; y < srch; y++)
    {
#if __ARM_NEON
        dst0 -= 7 * 4;

        int nn = srcw >> 4;
        int remain = srcw - (nn << 4);

#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x8x4_t _src = vld4_u8(src0);
            uint8x8x4_t _src2 = vld4_u8(src0 + 8 * 4);

            _src.val[0] = vrev64_u8(_src.val[0]);
            _src.val[1] = vrev64_u8(_src.val[1]);
            _src.val[2] = vrev64_u8(_src.val[2]);
            _src.val[3] = vrev64_u8(_src.val[3]);

            _src2.val[0] = vrev64_u8(_src2.val[0]);
            _src2.val[1] = vrev64_u8(_src2.val[1]);
            _src2.val[2] = vrev64_u8(_src2.val[2]);
            _src2.val[3] = vrev64_u8(_src2.val[3]);

            vst4_u8(dst0, _src);
            vst4_u8(dst0 - 8 * 4, _src2);

            src0 += 16 * 4;
            dst0 -= 16 * 4;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "mov        r4, #-32            \n"
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "vld4.u8    {d0-d3}, [%1]!      \n"
                "vrev64.u8  d0, d0              \n"
                "vrev64.u8  d1, d1              \n"
                "vrev64.u8  d2, d2              \n"
                "pld        [%1, #256]          \n"
                "vld4.u8    {d4-d7}, [%1]!      \n"
                "vrev64.u8  d3, d3              \n"
                "vrev64.u8  d4, d4              \n"
                "vrev64.u8  d5, d5              \n"
                "vst4.u8    {d0-d3}, [%2], r4   \n"
                "vrev64.u8  d6, d6              \n"
                "vrev64.u8  d7, d7              \n"
                "subs       %0, #1              \n"
                "vst4.u8    {d4-d7}, [%2], r4   \n"
                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(dst0)  // %2
                : "0"(nn),
                "1"(src0),
                "2"(dst0)
                : "cc", "memory", "q0", "q1", "q2", "q3", "r4");
        }
#endif // __aarch64__

        dst0 += 7 * 4;
#else
        int remain = srcw;
#endif // __ARM_NEON

        for (; remain > 0; remain--)
        {
            dst0[0] = src0[0];
            dst0[1] = src0[1];
            dst0[2] = src0[2];
            dst0[3] = src0[3];

            src0 += 4;
            dst0 -= 4;
        }

        src0 += srcwgap;
        dst0 -= wgap;
    }
}

static void kanna_rotate_4_c1(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    const int srcwgap = srcstride - srcw;
    const int wgap = stride + w;

    // point to the last dst pixel row
    unsigned char* dstend = dst + stride * (h - 1);

    const unsigned char* src0 = src;
    const unsigned char* src1 = src + srcstride;
    unsigned char* dst0 = dstend;
    unsigned char* dst1 = dstend - stride;

    int y = 0;
    for (; y + 1 < srch; y += 2)
    {
#if __ARM_NEON
        int nn = srcw >> 5;
        int remain = srcw - (nn << 5);
#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x16_t _src0 = vld1q_u8(src0);
            uint8x16_t _src0n = vld1q_u8(src0 + 16);
            vst1q_u8(dst0, _src0);
            vst1q_u8(dst0 + 16, _src0n);

            uint8x16_t _src1 = vld1q_u8(src1);
            uint8x16_t _src1n = vld1q_u8(src1 + 16);
            vst1q_u8(dst1, _src1);
            vst1q_u8(dst1 + 16, _src1n);

            src0 += 32;
            src1 += 32;
            dst0 += 32;
            dst1 += 32;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "vld1.u8    {d0-d3}, [%1]!      \n"
                "pld        [%2, #256]          \n"
                "vld1.u8    {d4-d7}, [%2]!      \n"
                "subs       %0, #1              \n"
                "vst1.u8    {d0-d3}, [%3]!      \n"
                "vst1.u8    {d4-d7}, [%4]!      \n"
                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(src1), // %2
                "=r"(dst0), // %3
                "=r"(dst1)  // %4
                : "0"(nn),
                "1"(src0),
                "2"(src1),
                "3"(dst0),
                "4"(dst1)
                : "cc", "memory", "q0", "q1", "q2", "q3");
        }
#endif // __aarch64__
#else
        int remain = srcw;
#endif // __ARM_NEON

        for (; remain > 0; remain--)
        {
            *dst0++ = *src0++;
            *dst1++ = *src1++;
        }

        src0 += srcwgap + srcstride;
        src1 += srcwgap + srcstride;
        dst0 -= wgap + stride;
        dst1 -= wgap + stride;
    }

    for (; y < srch; y++)
    {
#if __ARM_NEON
        int nn = srcw >> 5;
        int remain = srcw - (nn << 5);
#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x16_t _src = vld1q_u8(src0);
            uint8x16_t _src2 = vld1q_u8(src0 + 16);
            vst1q_u8(dst0, _src);
            vst1q_u8(dst0 + 16, _src2);

            src0 += 32;
            dst0 += 32;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "vld1.u8    {d0-d3}, [%1]!      \n"
                "subs       %0, #1              \n"
                "vst1.u8    {d0-d3}, [%2]!      \n"
                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(dst0)  // %2
                : "0"(nn),
                "1"(src0),
                "2"(dst0)
                : "cc", "memory", "q0", "q1");
        }
#endif // __aarch64__
#else
        int remain = srcw;
#endif // __ARM_NEON

        for (; remain > 0; remain--)
        {
            *dst0++ = *src0++;
        }

        src0 += srcwgap;
        dst0 -= wgap;
    }
}

static void kanna_rotate_4_c2(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    const int srcwgap = srcstride - srcw * 2;
    const int wgap = stride + w * 2;

    // point to the last dst pixel row
    unsigned char* dstend = dst + stride * (h - 1);

    int size = srcw * 2;

    const unsigned char* src0 = src;
    const unsigned char* src1 = src + srcstride;
    unsigned char* dst0 = dstend;
    unsigned char* dst1 = dstend - stride;

    int y = 0;
    for (; y + 1 < srch; y += 2)
    {
#if __ARM_NEON
        int nn = size >> 5;
        int remain = size - (nn << 5);
#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x16_t _src0 = vld1q_u8(src0);
            uint8x16_t _src0n = vld1q_u8(src0 + 16);
            vst1q_u8(dst0, _src0);
            vst1q_u8(dst0 + 16, _src0n);

            uint8x16_t _src1 = vld1q_u8(src1);
            uint8x16_t _src1n = vld1q_u8(src1 + 16);
            vst1q_u8(dst1, _src1);
            vst1q_u8(dst1 + 16, _src1n);

            src0 += 32;
            src1 += 32;
            dst0 += 32;
            dst1 += 32;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "vld1.u8    {d0-d3}, [%1]!      \n"
                "pld        [%2, #256]          \n"
                "vld1.u8    {d4-d7}, [%2]!      \n"
                "subs       %0, #1              \n"
                "vst1.u8    {d0-d3}, [%3]!      \n"
                "vst1.u8    {d4-d7}, [%4]!      \n"
                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(src1), // %2
                "=r"(dst0), // %3
                "=r"(dst1)  // %4
                : "0"(nn),
                "1"(src0),
                "2"(src1),
                "3"(dst0),
                "4"(dst1)
                : "cc", "memory", "q0", "q1", "q2", "q3");
        }
#endif // __aarch64__
#else
        int remain = size;
#endif // __ARM_NEON

        for (; remain > 0; remain--)
        {
            *dst0++ = *src0++;
            *dst1++ = *src1++;
        }

        src0 += srcwgap + srcstride;
        src1 += srcwgap + srcstride;
        dst0 -= wgap + stride;
        dst1 -= wgap + stride;
    }

    for (; y < srch; y++)
    {
#if __ARM_NEON
        int nn = size >> 5;
        int remain = size - (nn << 5);
#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x16_t _src = vld1q_u8(src0);
            uint8x16_t _src2 = vld1q_u8(src0 + 16);
            vst1q_u8(dst0, _src);
            vst1q_u8(dst0 + 16, _src2);

            src0 += 32;
            dst0 += 32;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "vld1.u8    {d0-d3}, [%1]!      \n"
                "subs       %0, #1              \n"
                "vst1.u8    {d0-d3}, [%2]!      \n"
                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(dst0)  // %2
                : "0"(nn),
                "1"(src0),
                "2"(dst0)
                : "cc", "memory", "q0", "q1");
        }
#endif // __aarch64__
#else
        int remain = size;
#endif // __ARM_NEON

        for (; remain > 0; remain--)
        {
            *dst0++ = *src0++;
        }

        src0 += srcwgap;
        dst0 -= wgap;
    }
}

static void kanna_rotate_4_c3(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    const int srcwgap = srcstride - srcw * 3;
    const int wgap = stride + w * 3;

    // point to the last dst pixel row
    unsigned char* dstend = dst + stride * (h - 1);

    int size = srcw * 3;

    const unsigned char* src0 = src;
    const unsigned char* src1 = src + srcstride;
    unsigned char* dst0 = dstend;
    unsigned char* dst1 = dstend - stride;

    int y = 0;
    for (; y + 1 < srch; y += 2)
    {
#if __ARM_NEON
        int nn = size >> 5;
        int remain = size - (nn << 5);
#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x16_t _src0 = vld1q_u8(src0);
            uint8x16_t _src0n = vld1q_u8(src0 + 16);
            vst1q_u8(dst0, _src0);
            vst1q_u8(dst0 + 16, _src0n);

            uint8x16_t _src1 = vld1q_u8(src1);
            uint8x16_t _src1n = vld1q_u8(src1 + 16);
            vst1q_u8(dst1, _src1);
            vst1q_u8(dst1 + 16, _src1n);

            src0 += 32;
            src1 += 32;
            dst0 += 32;
            dst1 += 32;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "vld1.u8    {d0-d3}, [%1]!      \n"
                "pld        [%2, #256]          \n"
                "vld1.u8    {d4-d7}, [%2]!      \n"
                "subs       %0, #1              \n"
                "vst1.u8    {d0-d3}, [%3]!      \n"
                "vst1.u8    {d4-d7}, [%4]!      \n"
                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(src1), // %2
                "=r"(dst0), // %3
                "=r"(dst1)  // %4
                : "0"(nn),
                "1"(src0),
                "2"(src1),
                "3"(dst0),
                "4"(dst1)
                : "cc", "memory", "q0", "q1", "q2", "q3");
        }
#endif // __aarch64__
#else
        int remain = size;
#endif // __ARM_NEON

        for (; remain > 0; remain--)
        {
            *dst0++ = *src0++;
            *dst1++ = *src1++;
        }

        src0 += srcwgap + srcstride;
        src1 += srcwgap + srcstride;
        dst0 -= wgap + stride;
        dst1 -= wgap + stride;
    }

    for (; y < srch; y++)
    {
#if __ARM_NEON
        int nn = size >> 5;
        int remain = size - (nn << 5);
#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x16_t _src = vld1q_u8(src0);
            uint8x16_t _src2 = vld1q_u8(src0 + 16);
            vst1q_u8(dst0, _src);
            vst1q_u8(dst0 + 16, _src2);

            src0 += 32;
            dst0 += 32;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "vld1.u8    {d0-d3}, [%1]!      \n"
                "subs       %0, #1              \n"
                "vst1.u8    {d0-d3}, [%2]!      \n"
                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(dst0)  // %2
                : "0"(nn),
                "1"(src0),
                "2"(dst0)
                : "cc", "memory", "q0", "q1");
        }
#endif // __aarch64__
#else
        int remain = size;
#endif // __ARM_NEON

        for (; remain > 0; remain--)
        {
            *dst0++ = *src0++;
        }

        src0 += srcwgap;
        dst0 -= wgap;
    }
}

static void kanna_rotate_4_c4(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    const int srcwgap = srcstride - srcw * 4;
    const int wgap = stride + w * 4;

    // point to the last dst pixel row
    unsigned char* dstend = dst + stride * (h - 1);

    int size = srcw * 4;

    const unsigned char* src0 = src;
    const unsigned char* src1 = src + srcstride;
    unsigned char* dst0 = dstend;
    unsigned char* dst1 = dstend - stride;

    int y = 0;
    for (; y + 1 < srch; y += 2)
    {
#if __ARM_NEON
        int nn = size >> 5;
        int remain = size - (nn << 5);
#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x16_t _src0 = vld1q_u8(src0);
            uint8x16_t _src0n = vld1q_u8(src0 + 16);
            vst1q_u8(dst0, _src0);
            vst1q_u8(dst0 + 16, _src0n);

            uint8x16_t _src1 = vld1q_u8(src1);
            uint8x16_t _src1n = vld1q_u8(src1 + 16);
            vst1q_u8(dst1, _src1);
            vst1q_u8(dst1 + 16, _src1n);

            src0 += 32;
            src1 += 32;
            dst0 += 32;
            dst1 += 32;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "vld1.u8    {d0-d3}, [%1]!      \n"
                "pld        [%2, #256]          \n"
                "vld1.u8    {d4-d7}, [%2]!      \n"
                "subs       %0, #1              \n"
                "vst1.u8    {d0-d3}, [%3]!      \n"
                "vst1.u8    {d4-d7}, [%4]!      \n"
                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(src1), // %2
                "=r"(dst0), // %3
                "=r"(dst1)  // %4
                : "0"(nn),
                "1"(src0),
                "2"(src1),
                "3"(dst0),
                "4"(dst1)
                : "cc", "memory", "q0", "q1", "q2", "q3");
        }
#endif // __aarch64__
#else
        int remain = size;
#endif // __ARM_NEON

        for (; remain > 0; remain--)
        {
            *dst0++ = *src0++;
            *dst1++ = *src1++;
        }

        src0 += srcwgap + srcstride;
        src1 += srcwgap + srcstride;
        dst0 -= wgap + stride;
        dst1 -= wgap + stride;
    }

    for (; y < srch; y++)
    {
#if __ARM_NEON
        int nn = size >> 5;
        int remain = size - (nn << 5);
#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x16_t _src = vld1q_u8(src0);
            uint8x16_t _src2 = vld1q_u8(src0 + 16);
            vst1q_u8(dst0, _src);
            vst1q_u8(dst0 + 16, _src2);

            src0 += 32;
            dst0 += 32;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "vld1.u8    {d0-d3}, [%1]!      \n"
                "subs       %0, #1              \n"
                "vst1.u8    {d0-d3}, [%2]!      \n"
                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(dst0)  // %2
                : "0"(nn),
                "1"(src0),
                "2"(dst0)
                : "cc", "memory", "q0", "q1");
        }
#endif // __aarch64__
#else
        int remain = size;
#endif // __ARM_NEON

        for (; remain > 0; remain--)
        {
            *dst0++ = *src0++;
        }

        src0 += srcwgap;
        dst0 -= wgap;
    }
}

static void kanna_rotate_5_c1(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int /*w*/, int /*h*/, int stride)
{
    const int srcwgap = srcstride - srcw;

    const unsigned char* src0 = src;

    int y = 0;
#if __ARM_NEON
    for (; y + 7 < srch; y += 8)
    {
        const unsigned char* src1 = src0 + srcstride;

        unsigned char* dst0 = dst + y;
        unsigned char* dst1 = dst + y + stride;

        int src_step = 2 * srcstride;
        int dst_step = 2 * stride;

        int nn = srcw >> 3;
        int remain = srcw - (nn << 3);

#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x8_t _src0 = vld1_u8(src0);
            uint8x8_t _src1 = vld1_u8(src1);

            uint8x8_t _src2 = vld1_u8(src0 + src_step);
            uint8x8_t _src3 = vld1_u8(src1 + src_step);

            uint8x8_t _src4 = vld1_u8(src0 + 2 * src_step);
            uint8x8_t _src5 = vld1_u8(src1 + 2 * src_step);

            uint8x8_t _src6 = vld1_u8(src0 + 3 * src_step);
            uint8x8_t _src7 = vld1_u8(src1 + 3 * src_step);

            uint8x8x2_t _src01t_r = vtrn_u8(_src0, _src1);
            uint8x8x2_t _src23t_r = vtrn_u8(_src2, _src3);
            uint8x8x2_t _src45t_r = vtrn_u8(_src4, _src5);
            uint8x8x2_t _src67t_r = vtrn_u8(_src6, _src7);

            uint16x4x2_t _src02tt_r = vtrn_u16(vreinterpret_u16_u8(_src01t_r.val[0]), vreinterpret_u16_u8(_src23t_r.val[0]));
            uint16x4x2_t _src13tt_r = vtrn_u16(vreinterpret_u16_u8(_src01t_r.val[1]), vreinterpret_u16_u8(_src23t_r.val[1]));
            uint16x4x2_t _src46tt_r = vtrn_u16(vreinterpret_u16_u8(_src45t_r.val[0]), vreinterpret_u16_u8(_src67t_r.val[0]));
            uint16x4x2_t _src57tt_r = vtrn_u16(vreinterpret_u16_u8(_src45t_r.val[1]), vreinterpret_u16_u8(_src67t_r.val[1]));

            uint32x2x2_t _src04ttt_r = vtrn_u32(vreinterpret_u32_u16(_src02tt_r.val[0]), vreinterpret_u32_u16(_src46tt_r.val[0]));
            uint32x2x2_t _src15ttt_r = vtrn_u32(vreinterpret_u32_u16(_src13tt_r.val[0]), vreinterpret_u32_u16(_src57tt_r.val[0]));
            uint32x2x2_t _src26ttt_r = vtrn_u32(vreinterpret_u32_u16(_src02tt_r.val[1]), vreinterpret_u32_u16(_src46tt_r.val[1]));
            uint32x2x2_t _src37ttt_r = vtrn_u32(vreinterpret_u32_u16(_src13tt_r.val[1]), vreinterpret_u32_u16(_src57tt_r.val[1]));

            uint8x8_t _dst0 = vreinterpret_u8_u32(_src04ttt_r.val[0]);
            uint8x8_t _dst1 = vreinterpret_u8_u32(_src15ttt_r.val[0]);
            uint8x8_t _dst2 = vreinterpret_u8_u32(_src26ttt_r.val[0]);
            uint8x8_t _dst3 = vreinterpret_u8_u32(_src37ttt_r.val[0]);
            uint8x8_t _dst4 = vreinterpret_u8_u32(_src04ttt_r.val[1]);
            uint8x8_t _dst5 = vreinterpret_u8_u32(_src15ttt_r.val[1]);
            uint8x8_t _dst6 = vreinterpret_u8_u32(_src26ttt_r.val[1]);
            uint8x8_t _dst7 = vreinterpret_u8_u32(_src37ttt_r.val[1]);

            vst1_u8(dst0, _dst0);
            vst1_u8(dst1, _dst1);
            vst1_u8(dst0 + dst_step, _dst2);
            vst1_u8(dst1 + dst_step, _dst3);
            vst1_u8(dst0 + 2 * dst_step, _dst4);
            vst1_u8(dst1 + 2 * dst_step, _dst5);
            vst1_u8(dst0 + 3 * dst_step, _dst6);
            vst1_u8(dst1 + 3 * dst_step, _dst7);

            src0 += 8;
            src1 += 8;

            dst0 += 4 * dst_step;
            dst1 += 4 * dst_step;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #64]           \n"
                "vld1.u8    {d0}, [%1], %10     \n"

                "pld        [%2, #64]           \n"
                "vld1.u8    {d1}, [%2], %10     \n"

                "pld        [%1, #64]           \n"
                "vld1.u8    {d2}, [%1], %10     \n"

                "vtrn.u8    d0, d1              \n" // _src01t_r

                "pld        [%2, #64]           \n"
                "vld1.u8    {d3}, [%2], %10     \n"

                "pld        [%1, #64]           \n"
                "vld1.u8    {d4}, [%1], %10     \n"

                "vtrn.u8    d2, d3              \n" // _src23t_r

                "pld        [%2, #64]           \n"
                "vld1.u8    {d5}, [%2], %10     \n"

                "pld        [%1, #64]           \n"
                "vld1.u8    {d6}, [%1], %10     \n"

                "vtrn.u8    d4, d5              \n" // _src45t_r

                "pld        [%2, #64]           \n"
                "vld1.u8    {d7}, [%2], %10     \n"

                "vtrn.u8    d6, d7              \n" // _src67t_r

                "sub        %1, %1, %10, lsl #2 \n" // restore src0

                "vtrn.u16   q0, q1              \n" // _src02tt_r _src13tt_r

                "sub        %2, %2, %10, lsl #2 \n" // restore src1

                "vtrn.u16   q2, q3              \n" // _src13tt_r _src46tt_r

                "add        %1, #8              \n" // src0 += 8

                "vtrn.u32   q0, q2              \n" // _src04ttt_r _src15ttt_r

                "add        %2, #8              \n" // src1 += 8

                "vtrn.u32   q1, q3              \n" // _src26ttt_r _src37ttt_r
                "vst1.u8    {d0}, [%3], %11     \n"
                "vst1.u8    {d1}, [%4], %11     \n"

                "subs       %0, #1              \n"

                "vst1.u8    {d2}, [%3], %11     \n"
                "vst1.u8    {d3}, [%4], %11     \n"
                "vst1.u8    {d4}, [%3], %11     \n"
                "vst1.u8    {d5}, [%4], %11     \n"
                "vst1.u8    {d6}, [%3], %11     \n"
                "vst1.u8    {d7}, [%4], %11     \n"

                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(src1), // %2
                "=r"(dst0), // %3
                "=r"(dst1)  // %4
                : "0"(nn),
                "1"(src0),
                "2"(src1),
                "3"(dst0),
                "4"(dst1),
                "r"(src_step), // %10
                "r"(dst_step)  // %11
                : "cc", "memory", "q0", "q1", "q2", "q3");
        }
#endif // __aarch64__
        for (; remain > 0; remain--)
        {
            dst0[0] = src0[0];
            dst0[1] = src1[0];
            dst0[2] = src0[0 + src_step];
            dst0[3] = src1[0 + src_step];
            dst0[4] = src0[0 + 2 * src_step];
            dst0[5] = src1[0 + 2 * src_step];
            dst0[6] = src0[0 + 3 * src_step];
            dst0[7] = src1[0 + 3 * src_step];

            src0 += 1;
            src1 += 1;

            dst0 += stride;
        }

        src0 += srcwgap + 7 * srcstride;
    }
#endif // __ARM_NEON
    for (; y < srch; y++)
    {
        unsigned char* dst0 = dst + y;

        int x = 0;
        for (; x < srcw; x++)
        {
            *dst0 = *src0;

            src0 += 1;
            dst0 += stride;
        }

        src0 += srcwgap;
    }
}

static void kanna_rotate_5_c2(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int /*w*/, int /*h*/, int stride)
{
    const int srcwgap = srcstride - srcw * 2;

    const unsigned char* src0 = src;

    int y = 0;
#if __ARM_NEON
    for (; y + 7 < srch; y += 8)
    {
        const unsigned char* src1 = src0 + srcstride;

        unsigned char* dst0 = dst + y * 2;
        unsigned char* dst1 = dst + y * 2 + stride;

        int src_step = 2 * srcstride;
        int dst_step = 2 * stride;

        int nn = srcw >> 3;
        int remain = srcw - (nn << 3);

#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x8x2_t _src0 = vld2_u8(src0);
            uint8x8x2_t _src1 = vld2_u8(src1);

            uint8x8x2_t _src2 = vld2_u8(src0 + src_step);
            uint8x8x2_t _src3 = vld2_u8(src1 + src_step);

            uint8x8x2_t _src4 = vld2_u8(src0 + 2 * src_step);
            uint8x8x2_t _src5 = vld2_u8(src1 + 2 * src_step);

            uint8x8x2_t _src6 = vld2_u8(src0 + 3 * src_step);
            uint8x8x2_t _src7 = vld2_u8(src1 + 3 * src_step);

            uint8x8x2_t _src01t_r = vtrn_u8(_src0.val[0], _src1.val[0]);
            uint8x8x2_t _src23t_r = vtrn_u8(_src2.val[0], _src3.val[0]);
            uint8x8x2_t _src45t_r = vtrn_u8(_src4.val[0], _src5.val[0]);
            uint8x8x2_t _src67t_r = vtrn_u8(_src6.val[0], _src7.val[0]);

            uint8x8x2_t _src01t_g = vtrn_u8(_src0.val[1], _src1.val[1]);
            uint8x8x2_t _src23t_g = vtrn_u8(_src2.val[1], _src3.val[1]);
            uint8x8x2_t _src45t_g = vtrn_u8(_src4.val[1], _src5.val[1]);
            uint8x8x2_t _src67t_g = vtrn_u8(_src6.val[1], _src7.val[1]);

            uint16x4x2_t _src02tt_r = vtrn_u16(vreinterpret_u16_u8(_src01t_r.val[0]), vreinterpret_u16_u8(_src23t_r.val[0]));
            uint16x4x2_t _src13tt_r = vtrn_u16(vreinterpret_u16_u8(_src01t_r.val[1]), vreinterpret_u16_u8(_src23t_r.val[1]));
            uint16x4x2_t _src46tt_r = vtrn_u16(vreinterpret_u16_u8(_src45t_r.val[0]), vreinterpret_u16_u8(_src67t_r.val[0]));
            uint16x4x2_t _src57tt_r = vtrn_u16(vreinterpret_u16_u8(_src45t_r.val[1]), vreinterpret_u16_u8(_src67t_r.val[1]));

            uint16x4x2_t _src02tt_g = vtrn_u16(vreinterpret_u16_u8(_src01t_g.val[0]), vreinterpret_u16_u8(_src23t_g.val[0]));
            uint16x4x2_t _src13tt_g = vtrn_u16(vreinterpret_u16_u8(_src01t_g.val[1]), vreinterpret_u16_u8(_src23t_g.val[1]));
            uint16x4x2_t _src46tt_g = vtrn_u16(vreinterpret_u16_u8(_src45t_g.val[0]), vreinterpret_u16_u8(_src67t_g.val[0]));
            uint16x4x2_t _src57tt_g = vtrn_u16(vreinterpret_u16_u8(_src45t_g.val[1]), vreinterpret_u16_u8(_src67t_g.val[1]));

            uint32x2x2_t _src04ttt_r = vtrn_u32(vreinterpret_u32_u16(_src02tt_r.val[0]), vreinterpret_u32_u16(_src46tt_r.val[0]));
            uint32x2x2_t _src15ttt_r = vtrn_u32(vreinterpret_u32_u16(_src13tt_r.val[0]), vreinterpret_u32_u16(_src57tt_r.val[0]));
            uint32x2x2_t _src26ttt_r = vtrn_u32(vreinterpret_u32_u16(_src02tt_r.val[1]), vreinterpret_u32_u16(_src46tt_r.val[1]));
            uint32x2x2_t _src37ttt_r = vtrn_u32(vreinterpret_u32_u16(_src13tt_r.val[1]), vreinterpret_u32_u16(_src57tt_r.val[1]));

            uint32x2x2_t _src04ttt_g = vtrn_u32(vreinterpret_u32_u16(_src02tt_g.val[0]), vreinterpret_u32_u16(_src46tt_g.val[0]));
            uint32x2x2_t _src15ttt_g = vtrn_u32(vreinterpret_u32_u16(_src13tt_g.val[0]), vreinterpret_u32_u16(_src57tt_g.val[0]));
            uint32x2x2_t _src26ttt_g = vtrn_u32(vreinterpret_u32_u16(_src02tt_g.val[1]), vreinterpret_u32_u16(_src46tt_g.val[1]));
            uint32x2x2_t _src37ttt_g = vtrn_u32(vreinterpret_u32_u16(_src13tt_g.val[1]), vreinterpret_u32_u16(_src57tt_g.val[1]));

            uint8x8x2_t _dst0;
            uint8x8x2_t _dst1;
            uint8x8x2_t _dst2;
            uint8x8x2_t _dst3;
            uint8x8x2_t _dst4;
            uint8x8x2_t _dst5;
            uint8x8x2_t _dst6;
            uint8x8x2_t _dst7;

            _dst0.val[0] = vreinterpret_u8_u32(_src04ttt_r.val[0]);
            _dst1.val[0] = vreinterpret_u8_u32(_src15ttt_r.val[0]);
            _dst2.val[0] = vreinterpret_u8_u32(_src26ttt_r.val[0]);
            _dst3.val[0] = vreinterpret_u8_u32(_src37ttt_r.val[0]);
            _dst4.val[0] = vreinterpret_u8_u32(_src04ttt_r.val[1]);
            _dst5.val[0] = vreinterpret_u8_u32(_src15ttt_r.val[1]);
            _dst6.val[0] = vreinterpret_u8_u32(_src26ttt_r.val[1]);
            _dst7.val[0] = vreinterpret_u8_u32(_src37ttt_r.val[1]);

            _dst0.val[1] = vreinterpret_u8_u32(_src04ttt_g.val[0]);
            _dst1.val[1] = vreinterpret_u8_u32(_src15ttt_g.val[0]);
            _dst2.val[1] = vreinterpret_u8_u32(_src26ttt_g.val[0]);
            _dst3.val[1] = vreinterpret_u8_u32(_src37ttt_g.val[0]);
            _dst4.val[1] = vreinterpret_u8_u32(_src04ttt_g.val[1]);
            _dst5.val[1] = vreinterpret_u8_u32(_src15ttt_g.val[1]);
            _dst6.val[1] = vreinterpret_u8_u32(_src26ttt_g.val[1]);
            _dst7.val[1] = vreinterpret_u8_u32(_src37ttt_g.val[1]);

            vst2_u8(dst0, _dst0);
            vst2_u8(dst1, _dst1);
            vst2_u8(dst0 + dst_step, _dst2);
            vst2_u8(dst1 + dst_step, _dst3);
            vst2_u8(dst0 + 2 * dst_step, _dst4);
            vst2_u8(dst1 + 2 * dst_step, _dst5);
            vst2_u8(dst0 + 3 * dst_step, _dst6);
            vst2_u8(dst1 + 3 * dst_step, _dst7);

            src0 += 2 * 8;
            src1 += 2 * 8;

            dst0 += 4 * dst_step;
            dst1 += 4 * dst_step;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #128]          \n"
                "vld2.u8    {d0-d1}, [%1], %10  \n"

                "pld        [%2, #128]          \n"
                "vld2.u8    {d2-d3}, [%2], %10  \n"

                "pld        [%1, #128]          \n"
                "vld2.u8    {d4-d5}, [%1], %10  \n"

                "vtrn.u8    q0, q1              \n" // _src01t_r

                "pld        [%2, #128]          \n"
                "vld2.u8    {d6-d7}, [%2], %10  \n"

                "pld        [%1, #128]          \n"
                "vld2.u8    {d16-d17}, [%1], %10\n"

                "vtrn.u8    q2, q3              \n" // _src23t_r

                "pld        [%2, #128]          \n"
                "vld2.u8    {d18-d19}, [%2], %10\n"

                "pld        [%1, #128]          \n"
                "vld2.u8    {d20-d21}, [%1], %10\n"

                "vtrn.u8    q8, q9              \n" // _src45t_r

                "pld        [%2, #128]          \n"
                "vld2.u8    {d22-d23}, [%2], %10\n"

                "vtrn.u8    q10, q11            \n" // _src67t_r

                "sub        %1, %1, %10, lsl #2 \n" // restore src0

                "vtrn.u16   q0, q2              \n" // _src02tt_r

                "sub        %2, %2, %10, lsl #2 \n" // restore src1

                "vtrn.u16   q1, q3              \n" // _src13tt_r

                "add        %1, #16             \n" // src0 += 16

                "vtrn.u16   q8, q10             \n" // _src46tt_r

                "add        %2, #16             \n" // src1 += 16

                "vtrn.u16   q9, q11             \n" // _src57tt_r

                "vtrn.u32   q0, q8              \n" // _src04ttt_r

                "vtrn.u32   q1, q9              \n" // _src15ttt_r
                "vst2.u8    {d0-d1}, [%3], %11  \n"

                "vtrn.u32   q2, q10             \n" // _src26ttt_r
                "vst2.u8    {d2-d3}, [%4], %11  \n"

                "vtrn.u32   q3, q11             \n" // _src37ttt_r
                "vst2.u8    {d4-d5}, [%3], %11  \n"

                "subs       %0, #1              \n"

                "vst2.u8    {d6-d7}, [%4], %11  \n"
                "vst2.u8    {d16-d17}, [%3], %11\n"
                "vst2.u8    {d18-d19}, [%4], %11\n"
                "vst2.u8    {d20-d21}, [%3], %11\n"
                "vst2.u8    {d22-d23}, [%4], %11\n"

                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(src1), // %2
                "=r"(dst0), // %3
                "=r"(dst1)  // %4
                : "0"(nn),
                "1"(src0),
                "2"(src1),
                "3"(dst0),
                "4"(dst1),
                "r"(src_step), // %10
                "r"(dst_step)  // %11
                : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11");
        }
#endif // __aarch64__
        for (; remain > 0; remain--)
        {
            dst0[0] = src0[0];
            dst0[1] = src0[1];
            dst0[2] = src1[0];
            dst0[3] = src1[1];
            dst0[4] = src0[0 + src_step];
            dst0[5] = src0[1 + src_step];
            dst0[6] = src1[0 + src_step];
            dst0[7] = src1[1 + src_step];
            dst0[8] = src0[0 + 2 * src_step];
            dst0[9] = src0[1 + 2 * src_step];
            dst0[10] = src1[0 + 2 * src_step];
            dst0[11] = src1[1 + 2 * src_step];
            dst0[12] = src0[0 + 3 * src_step];
            dst0[13] = src0[1 + 3 * src_step];
            dst0[14] = src1[0 + 3 * src_step];
            dst0[15] = src1[1 + 3 * src_step];

            src0 += 2;
            src1 += 2;

            dst0 += stride;
        }

        src0 += srcwgap + 7 * srcstride;
    }
#endif // __ARM_NEON
    for (; y < srch; y++)
    {
        unsigned char* dst0 = dst + y * 2;

        int x = 0;
        for (; x < srcw; x++)
        {
            dst0[0] = src0[0];
            dst0[1] = src0[1];

            src0 += 2;
            dst0 += stride;
        }

        src0 += srcwgap;
    }
}

static void kanna_rotate_5_c3(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int /*w*/, int /*h*/, int stride)
{
    const int srcwgap = srcstride - srcw * 3;

    const unsigned char* src0 = src;

    int y = 0;
#if __ARM_NEON
    for (; y + 7 < srch; y += 8)
    {
        const unsigned char* src1 = src0 + srcstride;

        unsigned char* dst0 = dst + y * 3;
        unsigned char* dst1 = dst + y * 3 + stride;

        int src_step = 2 * srcstride;
        int dst_step = 2 * stride;

        int nn = srcw >> 3;
        int remain = srcw - (nn << 3);

#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x8x3_t _src0 = vld3_u8(src0);
            uint8x8x3_t _src1 = vld3_u8(src1);

            uint8x8x3_t _src2 = vld3_u8(src0 + src_step);
            uint8x8x3_t _src3 = vld3_u8(src1 + src_step);

            uint8x8x3_t _src4 = vld3_u8(src0 + 2 * src_step);
            uint8x8x3_t _src5 = vld3_u8(src1 + 2 * src_step);

            uint8x8x3_t _src6 = vld3_u8(src0 + 3 * src_step);
            uint8x8x3_t _src7 = vld3_u8(src1 + 3 * src_step);

            uint8x8x2_t _src01t_r = vtrn_u8(_src0.val[0], _src1.val[0]);
            uint8x8x2_t _src23t_r = vtrn_u8(_src2.val[0], _src3.val[0]);
            uint8x8x2_t _src45t_r = vtrn_u8(_src4.val[0], _src5.val[0]);
            uint8x8x2_t _src67t_r = vtrn_u8(_src6.val[0], _src7.val[0]);

            uint8x8x2_t _src01t_g = vtrn_u8(_src0.val[1], _src1.val[1]);
            uint8x8x2_t _src23t_g = vtrn_u8(_src2.val[1], _src3.val[1]);
            uint8x8x2_t _src45t_g = vtrn_u8(_src4.val[1], _src5.val[1]);
            uint8x8x2_t _src67t_g = vtrn_u8(_src6.val[1], _src7.val[1]);

            uint8x8x2_t _src01t_b = vtrn_u8(_src0.val[2], _src1.val[2]);
            uint8x8x2_t _src23t_b = vtrn_u8(_src2.val[2], _src3.val[2]);
            uint8x8x2_t _src45t_b = vtrn_u8(_src4.val[2], _src5.val[2]);
            uint8x8x2_t _src67t_b = vtrn_u8(_src6.val[2], _src7.val[2]);

            uint16x4x2_t _src02tt_r = vtrn_u16(vreinterpret_u16_u8(_src01t_r.val[0]), vreinterpret_u16_u8(_src23t_r.val[0]));
            uint16x4x2_t _src13tt_r = vtrn_u16(vreinterpret_u16_u8(_src01t_r.val[1]), vreinterpret_u16_u8(_src23t_r.val[1]));
            uint16x4x2_t _src46tt_r = vtrn_u16(vreinterpret_u16_u8(_src45t_r.val[0]), vreinterpret_u16_u8(_src67t_r.val[0]));
            uint16x4x2_t _src57tt_r = vtrn_u16(vreinterpret_u16_u8(_src45t_r.val[1]), vreinterpret_u16_u8(_src67t_r.val[1]));

            uint16x4x2_t _src02tt_g = vtrn_u16(vreinterpret_u16_u8(_src01t_g.val[0]), vreinterpret_u16_u8(_src23t_g.val[0]));
            uint16x4x2_t _src13tt_g = vtrn_u16(vreinterpret_u16_u8(_src01t_g.val[1]), vreinterpret_u16_u8(_src23t_g.val[1]));
            uint16x4x2_t _src46tt_g = vtrn_u16(vreinterpret_u16_u8(_src45t_g.val[0]), vreinterpret_u16_u8(_src67t_g.val[0]));
            uint16x4x2_t _src57tt_g = vtrn_u16(vreinterpret_u16_u8(_src45t_g.val[1]), vreinterpret_u16_u8(_src67t_g.val[1]));

            uint16x4x2_t _src02tt_b = vtrn_u16(vreinterpret_u16_u8(_src01t_b.val[0]), vreinterpret_u16_u8(_src23t_b.val[0]));
            uint16x4x2_t _src13tt_b = vtrn_u16(vreinterpret_u16_u8(_src01t_b.val[1]), vreinterpret_u16_u8(_src23t_b.val[1]));
            uint16x4x2_t _src46tt_b = vtrn_u16(vreinterpret_u16_u8(_src45t_b.val[0]), vreinterpret_u16_u8(_src67t_b.val[0]));
            uint16x4x2_t _src57tt_b = vtrn_u16(vreinterpret_u16_u8(_src45t_b.val[1]), vreinterpret_u16_u8(_src67t_b.val[1]));

            uint32x2x2_t _src04ttt_r = vtrn_u32(vreinterpret_u32_u16(_src02tt_r.val[0]), vreinterpret_u32_u16(_src46tt_r.val[0]));
            uint32x2x2_t _src15ttt_r = vtrn_u32(vreinterpret_u32_u16(_src13tt_r.val[0]), vreinterpret_u32_u16(_src57tt_r.val[0]));
            uint32x2x2_t _src26ttt_r = vtrn_u32(vreinterpret_u32_u16(_src02tt_r.val[1]), vreinterpret_u32_u16(_src46tt_r.val[1]));
            uint32x2x2_t _src37ttt_r = vtrn_u32(vreinterpret_u32_u16(_src13tt_r.val[1]), vreinterpret_u32_u16(_src57tt_r.val[1]));

            uint32x2x2_t _src04ttt_g = vtrn_u32(vreinterpret_u32_u16(_src02tt_g.val[0]), vreinterpret_u32_u16(_src46tt_g.val[0]));
            uint32x2x2_t _src15ttt_g = vtrn_u32(vreinterpret_u32_u16(_src13tt_g.val[0]), vreinterpret_u32_u16(_src57tt_g.val[0]));
            uint32x2x2_t _src26ttt_g = vtrn_u32(vreinterpret_u32_u16(_src02tt_g.val[1]), vreinterpret_u32_u16(_src46tt_g.val[1]));
            uint32x2x2_t _src37ttt_g = vtrn_u32(vreinterpret_u32_u16(_src13tt_g.val[1]), vreinterpret_u32_u16(_src57tt_g.val[1]));

            uint32x2x2_t _src04ttt_b = vtrn_u32(vreinterpret_u32_u16(_src02tt_b.val[0]), vreinterpret_u32_u16(_src46tt_b.val[0]));
            uint32x2x2_t _src15ttt_b = vtrn_u32(vreinterpret_u32_u16(_src13tt_b.val[0]), vreinterpret_u32_u16(_src57tt_b.val[0]));
            uint32x2x2_t _src26ttt_b = vtrn_u32(vreinterpret_u32_u16(_src02tt_b.val[1]), vreinterpret_u32_u16(_src46tt_b.val[1]));
            uint32x2x2_t _src37ttt_b = vtrn_u32(vreinterpret_u32_u16(_src13tt_b.val[1]), vreinterpret_u32_u16(_src57tt_b.val[1]));

            uint8x8x3_t _dst0;
            uint8x8x3_t _dst1;
            uint8x8x3_t _dst2;
            uint8x8x3_t _dst3;
            uint8x8x3_t _dst4;
            uint8x8x3_t _dst5;
            uint8x8x3_t _dst6;
            uint8x8x3_t _dst7;

            _dst0.val[0] = vreinterpret_u8_u32(_src04ttt_r.val[0]);
            _dst1.val[0] = vreinterpret_u8_u32(_src15ttt_r.val[0]);
            _dst2.val[0] = vreinterpret_u8_u32(_src26ttt_r.val[0]);
            _dst3.val[0] = vreinterpret_u8_u32(_src37ttt_r.val[0]);
            _dst4.val[0] = vreinterpret_u8_u32(_src04ttt_r.val[1]);
            _dst5.val[0] = vreinterpret_u8_u32(_src15ttt_r.val[1]);
            _dst6.val[0] = vreinterpret_u8_u32(_src26ttt_r.val[1]);
            _dst7.val[0] = vreinterpret_u8_u32(_src37ttt_r.val[1]);

            _dst0.val[1] = vreinterpret_u8_u32(_src04ttt_g.val[0]);
            _dst1.val[1] = vreinterpret_u8_u32(_src15ttt_g.val[0]);
            _dst2.val[1] = vreinterpret_u8_u32(_src26ttt_g.val[0]);
            _dst3.val[1] = vreinterpret_u8_u32(_src37ttt_g.val[0]);
            _dst4.val[1] = vreinterpret_u8_u32(_src04ttt_g.val[1]);
            _dst5.val[1] = vreinterpret_u8_u32(_src15ttt_g.val[1]);
            _dst6.val[1] = vreinterpret_u8_u32(_src26ttt_g.val[1]);
            _dst7.val[1] = vreinterpret_u8_u32(_src37ttt_g.val[1]);

            _dst0.val[2] = vreinterpret_u8_u32(_src04ttt_b.val[0]);
            _dst1.val[2] = vreinterpret_u8_u32(_src15ttt_b.val[0]);
            _dst2.val[2] = vreinterpret_u8_u32(_src26ttt_b.val[0]);
            _dst3.val[2] = vreinterpret_u8_u32(_src37ttt_b.val[0]);
            _dst4.val[2] = vreinterpret_u8_u32(_src04ttt_b.val[1]);
            _dst5.val[2] = vreinterpret_u8_u32(_src15ttt_b.val[1]);
            _dst6.val[2] = vreinterpret_u8_u32(_src26ttt_b.val[1]);
            _dst7.val[2] = vreinterpret_u8_u32(_src37ttt_b.val[1]);

            vst3_u8(dst0, _dst0);
            vst3_u8(dst1, _dst1);
            vst3_u8(dst0 + dst_step, _dst2);
            vst3_u8(dst1 + dst_step, _dst3);
            vst3_u8(dst0 + 2 * dst_step, _dst4);
            vst3_u8(dst1 + 2 * dst_step, _dst5);
            vst3_u8(dst0 + 3 * dst_step, _dst6);
            vst3_u8(dst1 + 3 * dst_step, _dst7);

            src0 += 3 * 8;
            src1 += 3 * 8;

            dst0 += 4 * dst_step;
            dst1 += 4 * dst_step;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #192]          \n"
                "vld3.u8    {d0-d2}, [%1], %10  \n"

                "pld        [%2, #192]          \n"
                "vld3.u8    {d4-d6}, [%2], %10  \n"

                "pld        [%1, #192]          \n"
                "vld3.u8    {d8-d10}, [%1], %10 \n"

                "vtrn.u8    q0, q2              \n" // _src01t_r
                "vtrn.u8    d2, d6              \n"

                "pld        [%2, #192]          \n"
                "vld3.u8    {d12-d14}, [%2], %10\n"

                "pld        [%1, #192]          \n"
                "vld3.u8    {d16-d18}, [%1], %10\n"

                "vtrn.u8    q4, q6              \n" // _src23t_r
                "vtrn.u8    d10, d14            \n"

                "pld        [%2, #192]          \n"
                "vld3.u8    {d20-d22}, [%2], %10\n"

                "pld        [%1, #192]          \n"
                "vld3.u8    {d24-d26}, [%1], %10\n"

                "vtrn.u8    q8, q10             \n" // _src45t_r
                "vtrn.u8    d18, d22            \n"

                "pld        [%2, #192]          \n"
                "vld3.u8    {d28-d30}, [%2], %10\n"

                "vtrn.u8    q12, q14            \n" // _src67t_r
                "vtrn.u8    d26, d30            \n"

                "sub        %1, %1, %10, lsl #2 \n" // restore src0

                "vtrn.u16   q0, q4              \n" // _src02tt_r
                "vtrn.u16   d2, d10             \n"

                "sub        %2, %2, %10, lsl #2 \n" // restore src1

                "vtrn.u16   q2, q6              \n" // _src13tt_r
                "vtrn.u16   d6, d14             \n"

                "add        %1, #24             \n" // src0 += 24

                "vtrn.u16   q8, q12             \n" // _src46tt_r
                "vtrn.u16   d18, d26            \n"

                "add        %2, #24             \n" // src1 += 24

                "vtrn.u16   q10, q14            \n" // _src57tt_r
                "vtrn.u16   d22, d30            \n"

                "vtrn.u32   q0, q8              \n" // _src04ttt_r
                "vtrn.u32   d2, d18             \n"

                "vtrn.u32   q2, q10             \n" // _src15ttt_r
                "vst3.u8    {d0-d2}, [%3], %11  \n"
                "vtrn.u32   d6, d22             \n"

                "vtrn.u32   q4, q12             \n" // _src26ttt_r
                "vst3.u8    {d4-d6}, [%4], %11  \n"
                "vtrn.u32   d10, d26            \n"

                "vtrn.u32   q6, q14             \n" // _src37ttt_r
                "vst3.u8    {d8-d10}, [%3], %11 \n"
                "vtrn.u32   d14, d30            \n"

                "subs       %0, #1              \n"

                "vst3.u8    {d16-d18}, [%3], %11\n"
                "vst3.u8    {d12-d14}, [%4], %11\n"
                "vst3.u8    {d20-d22}, [%4], %11\n"
                "vst3.u8    {d24-d26}, [%3], %11\n"
                "vst3.u8    {d28-d30}, [%4], %11\n"

                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(src1), // %2
                "=r"(dst0), // %3
                "=r"(dst1)  // %4
                : "0"(nn),
                "1"(src0),
                "2"(src1),
                "3"(dst0),
                "4"(dst1),
                "r"(src_step), // %10
                "r"(dst_step)  // %11
                : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
        }
#endif // __aarch64__
        for (; remain > 0; remain--)
        {
            dst0[0] = src0[0];
            dst0[1] = src0[1];
            dst0[2] = src0[2];
            dst0[3] = src1[0];
            dst0[4] = src1[1];
            dst0[5] = src1[2];
            dst0[6] = src0[0 + src_step];
            dst0[7] = src0[1 + src_step];
            dst0[8] = src0[2 + src_step];
            dst0[9] = src1[0 + src_step];
            dst0[10] = src1[1 + src_step];
            dst0[11] = src1[2 + src_step];
            dst0[12] = src0[0 + 2 * src_step];
            dst0[13] = src0[1 + 2 * src_step];
            dst0[14] = src0[2 + 2 * src_step];
            dst0[15] = src1[0 + 2 * src_step];
            dst0[16] = src1[1 + 2 * src_step];
            dst0[17] = src1[2 + 2 * src_step];
            dst0[18] = src0[0 + 3 * src_step];
            dst0[19] = src0[1 + 3 * src_step];
            dst0[20] = src0[2 + 3 * src_step];
            dst0[21] = src1[0 + 3 * src_step];
            dst0[22] = src1[1 + 3 * src_step];
            dst0[23] = src1[2 + 3 * src_step];

            src0 += 3;
            src1 += 3;

            dst0 += stride;
        }

        src0 += srcwgap + 7 * srcstride;
    }
#endif // __ARM_NEON
    for (; y < srch; y++)
    {
        unsigned char* dst0 = dst + y * 3;

        int x = 0;
        for (; x < srcw; x++)
        {
            dst0[0] = src0[0];
            dst0[1] = src0[1];
            dst0[2] = src0[2];

            src0 += 3;
            dst0 += stride;
        }

        src0 += srcwgap;
    }
}

static void kanna_rotate_5_c4(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int /*w*/, int /*h*/, int stride)
{
    const int srcwgap = srcstride - srcw * 4;

    const unsigned char* src0 = src;

    int y = 0;
#if __ARM_NEON
    for (; y + 7 < srch; y += 8)
    {
        const unsigned char* src1 = src0 + srcstride;

        unsigned char* dst0 = dst + y * 4;
        unsigned char* dst1 = dst + y * 4 + stride;

        int src_step = 2 * srcstride;
        int dst_step = 2 * stride;

        int nn = srcw >> 3;
        int remain = srcw - (nn << 3);

#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x8x4_t _src0 = vld4_u8(src0);
            uint8x8x4_t _src1 = vld4_u8(src1);

            uint8x8x4_t _src2 = vld4_u8(src0 + src_step);
            uint8x8x4_t _src3 = vld4_u8(src1 + src_step);

            uint8x8x4_t _src4 = vld4_u8(src0 + 2 * src_step);
            uint8x8x4_t _src5 = vld4_u8(src1 + 2 * src_step);

            uint8x8x4_t _src6 = vld4_u8(src0 + 3 * src_step);
            uint8x8x4_t _src7 = vld4_u8(src1 + 3 * src_step);

            uint8x8x2_t _src01t_r = vtrn_u8(_src0.val[0], _src1.val[0]);
            uint8x8x2_t _src23t_r = vtrn_u8(_src2.val[0], _src3.val[0]);
            uint8x8x2_t _src45t_r = vtrn_u8(_src4.val[0], _src5.val[0]);
            uint8x8x2_t _src67t_r = vtrn_u8(_src6.val[0], _src7.val[0]);

            uint8x8x2_t _src01t_g = vtrn_u8(_src0.val[1], _src1.val[1]);
            uint8x8x2_t _src23t_g = vtrn_u8(_src2.val[1], _src3.val[1]);
            uint8x8x2_t _src45t_g = vtrn_u8(_src4.val[1], _src5.val[1]);
            uint8x8x2_t _src67t_g = vtrn_u8(_src6.val[1], _src7.val[1]);

            uint8x8x2_t _src01t_b = vtrn_u8(_src0.val[2], _src1.val[2]);
            uint8x8x2_t _src23t_b = vtrn_u8(_src2.val[2], _src3.val[2]);
            uint8x8x2_t _src45t_b = vtrn_u8(_src4.val[2], _src5.val[2]);
            uint8x8x2_t _src67t_b = vtrn_u8(_src6.val[2], _src7.val[2]);

            uint8x8x2_t _src01t_a = vtrn_u8(_src0.val[3], _src1.val[3]);
            uint8x8x2_t _src23t_a = vtrn_u8(_src2.val[3], _src3.val[3]);
            uint8x8x2_t _src45t_a = vtrn_u8(_src4.val[3], _src5.val[3]);
            uint8x8x2_t _src67t_a = vtrn_u8(_src6.val[3], _src7.val[3]);

            uint16x4x2_t _src02tt_r = vtrn_u16(vreinterpret_u16_u8(_src01t_r.val[0]), vreinterpret_u16_u8(_src23t_r.val[0]));
            uint16x4x2_t _src13tt_r = vtrn_u16(vreinterpret_u16_u8(_src01t_r.val[1]), vreinterpret_u16_u8(_src23t_r.val[1]));
            uint16x4x2_t _src46tt_r = vtrn_u16(vreinterpret_u16_u8(_src45t_r.val[0]), vreinterpret_u16_u8(_src67t_r.val[0]));
            uint16x4x2_t _src57tt_r = vtrn_u16(vreinterpret_u16_u8(_src45t_r.val[1]), vreinterpret_u16_u8(_src67t_r.val[1]));

            uint16x4x2_t _src02tt_g = vtrn_u16(vreinterpret_u16_u8(_src01t_g.val[0]), vreinterpret_u16_u8(_src23t_g.val[0]));
            uint16x4x2_t _src13tt_g = vtrn_u16(vreinterpret_u16_u8(_src01t_g.val[1]), vreinterpret_u16_u8(_src23t_g.val[1]));
            uint16x4x2_t _src46tt_g = vtrn_u16(vreinterpret_u16_u8(_src45t_g.val[0]), vreinterpret_u16_u8(_src67t_g.val[0]));
            uint16x4x2_t _src57tt_g = vtrn_u16(vreinterpret_u16_u8(_src45t_g.val[1]), vreinterpret_u16_u8(_src67t_g.val[1]));

            uint16x4x2_t _src02tt_b = vtrn_u16(vreinterpret_u16_u8(_src01t_b.val[0]), vreinterpret_u16_u8(_src23t_b.val[0]));
            uint16x4x2_t _src13tt_b = vtrn_u16(vreinterpret_u16_u8(_src01t_b.val[1]), vreinterpret_u16_u8(_src23t_b.val[1]));
            uint16x4x2_t _src46tt_b = vtrn_u16(vreinterpret_u16_u8(_src45t_b.val[0]), vreinterpret_u16_u8(_src67t_b.val[0]));
            uint16x4x2_t _src57tt_b = vtrn_u16(vreinterpret_u16_u8(_src45t_b.val[1]), vreinterpret_u16_u8(_src67t_b.val[1]));

            uint16x4x2_t _src02tt_a = vtrn_u16(vreinterpret_u16_u8(_src01t_a.val[0]), vreinterpret_u16_u8(_src23t_a.val[0]));
            uint16x4x2_t _src13tt_a = vtrn_u16(vreinterpret_u16_u8(_src01t_a.val[1]), vreinterpret_u16_u8(_src23t_a.val[1]));
            uint16x4x2_t _src46tt_a = vtrn_u16(vreinterpret_u16_u8(_src45t_a.val[0]), vreinterpret_u16_u8(_src67t_a.val[0]));
            uint16x4x2_t _src57tt_a = vtrn_u16(vreinterpret_u16_u8(_src45t_a.val[1]), vreinterpret_u16_u8(_src67t_a.val[1]));

            uint32x2x2_t _src04ttt_r = vtrn_u32(vreinterpret_u32_u16(_src02tt_r.val[0]), vreinterpret_u32_u16(_src46tt_r.val[0]));
            uint32x2x2_t _src15ttt_r = vtrn_u32(vreinterpret_u32_u16(_src13tt_r.val[0]), vreinterpret_u32_u16(_src57tt_r.val[0]));
            uint32x2x2_t _src26ttt_r = vtrn_u32(vreinterpret_u32_u16(_src02tt_r.val[1]), vreinterpret_u32_u16(_src46tt_r.val[1]));
            uint32x2x2_t _src37ttt_r = vtrn_u32(vreinterpret_u32_u16(_src13tt_r.val[1]), vreinterpret_u32_u16(_src57tt_r.val[1]));

            uint32x2x2_t _src04ttt_g = vtrn_u32(vreinterpret_u32_u16(_src02tt_g.val[0]), vreinterpret_u32_u16(_src46tt_g.val[0]));
            uint32x2x2_t _src15ttt_g = vtrn_u32(vreinterpret_u32_u16(_src13tt_g.val[0]), vreinterpret_u32_u16(_src57tt_g.val[0]));
            uint32x2x2_t _src26ttt_g = vtrn_u32(vreinterpret_u32_u16(_src02tt_g.val[1]), vreinterpret_u32_u16(_src46tt_g.val[1]));
            uint32x2x2_t _src37ttt_g = vtrn_u32(vreinterpret_u32_u16(_src13tt_g.val[1]), vreinterpret_u32_u16(_src57tt_g.val[1]));

            uint32x2x2_t _src04ttt_b = vtrn_u32(vreinterpret_u32_u16(_src02tt_b.val[0]), vreinterpret_u32_u16(_src46tt_b.val[0]));
            uint32x2x2_t _src15ttt_b = vtrn_u32(vreinterpret_u32_u16(_src13tt_b.val[0]), vreinterpret_u32_u16(_src57tt_b.val[0]));
            uint32x2x2_t _src26ttt_b = vtrn_u32(vreinterpret_u32_u16(_src02tt_b.val[1]), vreinterpret_u32_u16(_src46tt_b.val[1]));
            uint32x2x2_t _src37ttt_b = vtrn_u32(vreinterpret_u32_u16(_src13tt_b.val[1]), vreinterpret_u32_u16(_src57tt_b.val[1]));

            uint32x2x2_t _src04ttt_a = vtrn_u32(vreinterpret_u32_u16(_src02tt_a.val[0]), vreinterpret_u32_u16(_src46tt_a.val[0]));
            uint32x2x2_t _src15ttt_a = vtrn_u32(vreinterpret_u32_u16(_src13tt_a.val[0]), vreinterpret_u32_u16(_src57tt_a.val[0]));
            uint32x2x2_t _src26ttt_a = vtrn_u32(vreinterpret_u32_u16(_src02tt_a.val[1]), vreinterpret_u32_u16(_src46tt_a.val[1]));
            uint32x2x2_t _src37ttt_a = vtrn_u32(vreinterpret_u32_u16(_src13tt_a.val[1]), vreinterpret_u32_u16(_src57tt_a.val[1]));

            uint8x8x4_t _dst0;
            uint8x8x4_t _dst1;
            uint8x8x4_t _dst2;
            uint8x8x4_t _dst3;
            uint8x8x4_t _dst4;
            uint8x8x4_t _dst5;
            uint8x8x4_t _dst6;
            uint8x8x4_t _dst7;

            _dst0.val[0] = vreinterpret_u8_u32(_src04ttt_r.val[0]);
            _dst1.val[0] = vreinterpret_u8_u32(_src15ttt_r.val[0]);
            _dst2.val[0] = vreinterpret_u8_u32(_src26ttt_r.val[0]);
            _dst3.val[0] = vreinterpret_u8_u32(_src37ttt_r.val[0]);
            _dst4.val[0] = vreinterpret_u8_u32(_src04ttt_r.val[1]);
            _dst5.val[0] = vreinterpret_u8_u32(_src15ttt_r.val[1]);
            _dst6.val[0] = vreinterpret_u8_u32(_src26ttt_r.val[1]);
            _dst7.val[0] = vreinterpret_u8_u32(_src37ttt_r.val[1]);

            _dst0.val[1] = vreinterpret_u8_u32(_src04ttt_g.val[0]);
            _dst1.val[1] = vreinterpret_u8_u32(_src15ttt_g.val[0]);
            _dst2.val[1] = vreinterpret_u8_u32(_src26ttt_g.val[0]);
            _dst3.val[1] = vreinterpret_u8_u32(_src37ttt_g.val[0]);
            _dst4.val[1] = vreinterpret_u8_u32(_src04ttt_g.val[1]);
            _dst5.val[1] = vreinterpret_u8_u32(_src15ttt_g.val[1]);
            _dst6.val[1] = vreinterpret_u8_u32(_src26ttt_g.val[1]);
            _dst7.val[1] = vreinterpret_u8_u32(_src37ttt_g.val[1]);

            _dst0.val[2] = vreinterpret_u8_u32(_src04ttt_b.val[0]);
            _dst1.val[2] = vreinterpret_u8_u32(_src15ttt_b.val[0]);
            _dst2.val[2] = vreinterpret_u8_u32(_src26ttt_b.val[0]);
            _dst3.val[2] = vreinterpret_u8_u32(_src37ttt_b.val[0]);
            _dst4.val[2] = vreinterpret_u8_u32(_src04ttt_b.val[1]);
            _dst5.val[2] = vreinterpret_u8_u32(_src15ttt_b.val[1]);
            _dst6.val[2] = vreinterpret_u8_u32(_src26ttt_b.val[1]);
            _dst7.val[2] = vreinterpret_u8_u32(_src37ttt_b.val[1]);

            _dst0.val[3] = vreinterpret_u8_u32(_src04ttt_a.val[0]);
            _dst1.val[3] = vreinterpret_u8_u32(_src15ttt_a.val[0]);
            _dst2.val[3] = vreinterpret_u8_u32(_src26ttt_a.val[0]);
            _dst3.val[3] = vreinterpret_u8_u32(_src37ttt_a.val[0]);
            _dst4.val[3] = vreinterpret_u8_u32(_src04ttt_a.val[1]);
            _dst5.val[3] = vreinterpret_u8_u32(_src15ttt_a.val[1]);
            _dst6.val[3] = vreinterpret_u8_u32(_src26ttt_a.val[1]);
            _dst7.val[3] = vreinterpret_u8_u32(_src37ttt_a.val[1]);

            vst4_u8(dst0, _dst0);
            vst4_u8(dst1, _dst1);
            vst4_u8(dst0 + dst_step, _dst2);
            vst4_u8(dst1 + dst_step, _dst3);
            vst4_u8(dst0 + 2 * dst_step, _dst4);
            vst4_u8(dst1 + 2 * dst_step, _dst5);
            vst4_u8(dst0 + 3 * dst_step, _dst6);
            vst4_u8(dst1 + 3 * dst_step, _dst7);

            src0 += 4 * 8;
            src1 += 4 * 8;

            dst0 += 4 * dst_step;
            dst1 += 4 * dst_step;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "vld4.u8    {d0-d3}, [%1], %10  \n"

                "pld        [%2, #256]          \n"
                "vld4.u8    {d4-d7}, [%2], %10  \n"

                "pld        [%1, #256]          \n"
                "vld4.u8    {d8-d11}, [%1], %10 \n"

                "vtrn.u8    q0, q2              \n" // _src01t_r
                "vtrn.u8    q1, q3              \n"

                "pld        [%2, #256]          \n"
                "vld4.u8    {d12-d15}, [%2], %10\n"

                "pld        [%1, #256]          \n"
                "vld4.u8    {d16-d19}, [%1], %10\n"

                "vtrn.u8    q4, q6              \n" // _src23t_r
                "vtrn.u8    q5, q7              \n"

                "pld        [%2, #256]          \n"
                "vld4.u8    {d20-d23}, [%2], %10\n"

                "pld        [%1, #256]          \n"
                "vld4.u8    {d24-d27}, [%1], %10\n"

                "vtrn.u8    q8, q10             \n" // _src45t_r
                "vtrn.u8    q9, q11             \n"

                "pld        [%2, #256]          \n"
                "vld4.u8    {d28-d31}, [%2], %10\n"

                "vtrn.u8    q12, q14            \n" // _src67t_r
                "vtrn.u8    q13, q15            \n"

                "sub        %1, %1, %10, lsl #2 \n" // restore src0

                "vtrn.u16   q0, q4              \n" // _src02tt_r
                "vtrn.u16   q1, q5              \n"

                "sub        %2, %2, %10, lsl #2 \n" // restore src1

                "vtrn.u16   q2, q6              \n" // _src13tt_r
                "vtrn.u16   q3, q7              \n"

                "add        %1, #32             \n" // src0 += 32

                "vtrn.u16   q8, q12             \n" // _src46tt_r
                "vtrn.u16   q9, q13             \n"

                "add        %2, #32             \n" // src1 += 32

                "vtrn.u16   q10, q14            \n" // _src57tt_r
                "vtrn.u16   q11, q15            \n"

                "vtrn.u32   q0, q8              \n" // _src04ttt_r
                "vtrn.u32   q1, q9              \n"

                "vtrn.u32   q2, q10             \n" // _src15ttt_r
                "vst4.u8    {d0-d3}, [%3], %11  \n"
                "vtrn.u32   q3, q11             \n"

                "vtrn.u32   q4, q12             \n" // _src26ttt_r
                "vst4.u8    {d4-d7}, [%4], %11  \n"
                "vtrn.u32   q5, q13             \n"

                "vtrn.u32   q6, q14             \n" // _src37ttt_r
                "vst4.u8    {d8-d11}, [%3], %11 \n"
                "vtrn.u32   q7, q15             \n"

                "subs       %0, #1              \n"

                "vst4.u8    {d16-d19}, [%3], %11\n"
                "vst4.u8    {d12-d15}, [%4], %11\n"
                "vst4.u8    {d20-d23}, [%4], %11\n"
                "vst4.u8    {d24-d27}, [%3], %11\n"
                "vst4.u8    {d28-d31}, [%4], %11\n"

                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(src1), // %2
                "=r"(dst0), // %3
                "=r"(dst1)  // %4
                : "0"(nn),
                "1"(src0),
                "2"(src1),
                "3"(dst0),
                "4"(dst1),
                "r"(src_step), // %10
                "r"(dst_step)  // %11
                : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
        }
#endif // __aarch64__
        for (; remain > 0; remain--)
        {
            dst0[0] = src0[0];
            dst0[1] = src0[1];
            dst0[2] = src0[2];
            dst0[3] = src0[3];
            dst0[4] = src1[0];
            dst0[5] = src1[1];
            dst0[6] = src1[2];
            dst0[7] = src1[3];
            dst0[8] = src0[0 + src_step];
            dst0[9] = src0[1 + src_step];
            dst0[10] = src0[2 + src_step];
            dst0[11] = src0[3 + src_step];
            dst0[12] = src1[0 + src_step];
            dst0[13] = src1[1 + src_step];
            dst0[14] = src1[2 + src_step];
            dst0[15] = src1[3 + src_step];
            dst0[16] = src0[0 + 2 * src_step];
            dst0[17] = src0[1 + 2 * src_step];
            dst0[18] = src0[2 + 2 * src_step];
            dst0[19] = src0[3 + 2 * src_step];
            dst0[20] = src1[0 + 2 * src_step];
            dst0[21] = src1[1 + 2 * src_step];
            dst0[22] = src1[2 + 2 * src_step];
            dst0[23] = src1[3 + 2 * src_step];
            dst0[24] = src0[0 + 3 * src_step];
            dst0[25] = src0[1 + 3 * src_step];
            dst0[26] = src0[2 + 3 * src_step];
            dst0[27] = src0[3 + 3 * src_step];
            dst0[28] = src1[0 + 3 * src_step];
            dst0[29] = src1[1 + 3 * src_step];
            dst0[30] = src1[2 + 3 * src_step];
            dst0[31] = src1[3 + 3 * src_step];

            src0 += 4;
            src1 += 4;

            dst0 += stride;
        }

        src0 += srcwgap + 7 * srcstride;
    }
#endif // __ARM_NEON
    for (; y < srch; y++)
    {
        unsigned char* dst0 = dst + y * 4;

        int x = 0;
        for (; x < srcw; x++)
        {
            dst0[0] = src0[0];
            dst0[1] = src0[1];
            dst0[2] = src0[2];
            dst0[3] = src0[3];

            src0 += 4;
            dst0 += stride;
        }

        src0 += srcwgap;
    }
}

static void kanna_rotate_6_c1(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int /*h*/, int stride)
{
    const int srcwgap = srcstride - srcw;

    // point to the last dst pixel in row
    unsigned char* dstend = dst + w;

    const unsigned char* src0 = src;

    int y = 0;
#if __ARM_NEON
    for (; y + 7 < srch; y += 8)
    {
        const unsigned char* src1 = src0 + srcstride;

        unsigned char* dst0 = dstend - y - 8;
        unsigned char* dst1 = dstend - y - 8 + stride;

        int src_step = 2 * srcstride;
        int dst_step = 2 * stride;

        int nn = srcw >> 3;
        int remain = srcw - (nn << 3);

#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x8_t _src0 = vld1_u8(src0);
            uint8x8_t _src1 = vld1_u8(src1);

            uint8x8_t _src2 = vld1_u8(src0 + src_step);
            uint8x8_t _src3 = vld1_u8(src1 + src_step);

            uint8x8_t _src4 = vld1_u8(src0 + 2 * src_step);
            uint8x8_t _src5 = vld1_u8(src1 + 2 * src_step);

            uint8x8_t _src6 = vld1_u8(src0 + 3 * src_step);
            uint8x8_t _src7 = vld1_u8(src1 + 3 * src_step);

            uint8x8x2_t _src01t_r = vtrn_u8(_src1, _src0);
            uint8x8x2_t _src23t_r = vtrn_u8(_src3, _src2);
            uint8x8x2_t _src45t_r = vtrn_u8(_src5, _src4);
            uint8x8x2_t _src67t_r = vtrn_u8(_src7, _src6);

            uint16x4x2_t _src02tt_r = vtrn_u16(vreinterpret_u16_u8(_src23t_r.val[1]), vreinterpret_u16_u8(_src01t_r.val[1]));
            uint16x4x2_t _src13tt_r = vtrn_u16(vreinterpret_u16_u8(_src23t_r.val[0]), vreinterpret_u16_u8(_src01t_r.val[0]));
            uint16x4x2_t _src46tt_r = vtrn_u16(vreinterpret_u16_u8(_src67t_r.val[1]), vreinterpret_u16_u8(_src45t_r.val[1]));
            uint16x4x2_t _src57tt_r = vtrn_u16(vreinterpret_u16_u8(_src67t_r.val[0]), vreinterpret_u16_u8(_src45t_r.val[0]));

            uint32x2x2_t _src04ttt_r = vtrn_u32(vreinterpret_u32_u16(_src46tt_r.val[1]), vreinterpret_u32_u16(_src02tt_r.val[1]));
            uint32x2x2_t _src15ttt_r = vtrn_u32(vreinterpret_u32_u16(_src57tt_r.val[1]), vreinterpret_u32_u16(_src13tt_r.val[1]));
            uint32x2x2_t _src26ttt_r = vtrn_u32(vreinterpret_u32_u16(_src46tt_r.val[0]), vreinterpret_u32_u16(_src02tt_r.val[0]));
            uint32x2x2_t _src37ttt_r = vtrn_u32(vreinterpret_u32_u16(_src57tt_r.val[0]), vreinterpret_u32_u16(_src13tt_r.val[0]));

            uint8x8_t _dst0 = vreinterpret_u8_u32(_src04ttt_r.val[1]);
            uint8x8_t _dst1 = vreinterpret_u8_u32(_src15ttt_r.val[1]);
            uint8x8_t _dst2 = vreinterpret_u8_u32(_src26ttt_r.val[1]);
            uint8x8_t _dst3 = vreinterpret_u8_u32(_src37ttt_r.val[1]);
            uint8x8_t _dst4 = vreinterpret_u8_u32(_src04ttt_r.val[0]);
            uint8x8_t _dst5 = vreinterpret_u8_u32(_src15ttt_r.val[0]);
            uint8x8_t _dst6 = vreinterpret_u8_u32(_src26ttt_r.val[0]);
            uint8x8_t _dst7 = vreinterpret_u8_u32(_src37ttt_r.val[0]);

            vst1_u8(dst0, _dst7);
            vst1_u8(dst1, _dst6);
            vst1_u8(dst0 + dst_step, _dst5);
            vst1_u8(dst1 + dst_step, _dst4);
            vst1_u8(dst0 + 2 * dst_step, _dst3);
            vst1_u8(dst1 + 2 * dst_step, _dst2);
            vst1_u8(dst0 + 3 * dst_step, _dst1);
            vst1_u8(dst1 + 3 * dst_step, _dst0);

            src0 += 8;
            src1 += 8;

            dst0 += 4 * dst_step;
            dst1 += 4 * dst_step;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #64]           \n"
                "vld1.u8    {d0}, [%1], %10     \n"

                "pld        [%2, #64]           \n"
                "vld1.u8    {d1}, [%2], %10     \n"

                "pld        [%1, #64]           \n"
                "vld1.u8    {d2}, [%1], %10     \n"

                "vtrn.u8    d1, d0              \n" // _src01t_r

                "pld        [%2, #64]           \n"
                "vld1.u8    {d3}, [%2], %10     \n"

                "pld        [%1, #64]           \n"
                "vld1.u8    {d4}, [%1], %10     \n"

                "vtrn.u8    d3, d2              \n" // _src23t_r

                "pld        [%2, #64]           \n"
                "vld1.u8    {d5}, [%2], %10     \n"

                "pld        [%1, #64]           \n"
                "vld1.u8    {d6}, [%1], %10     \n"

                "vtrn.u8    d5, d4              \n" // _src45t_r

                "pld        [%2, #64]           \n"
                "vld1.u8    {d7}, [%2], %10     \n"

                "vtrn.u8    d7, d6              \n" // _src67t_r

                "sub        %1, %1, %10, lsl #2 \n" // restore src0

                "vtrn.u16   q1, q0              \n" // _src02tt_r _src13tt_r

                "sub        %2, %2, %10, lsl #2 \n" // restore src1

                "vtrn.u16   q3, q2              \n" // _src46tt_r _src57tt_r

                "add        %1, #8              \n" // src0 += 8

                "vtrn.u32   q3, q1              \n" // _src26ttt_r _src37ttt_r

                "add        %2, #8              \n" // src1 += 8

                "vtrn.u32   q2, q0              \n" // _src04ttt_r _src15ttt_r
                "vst1.u8    {d6}, [%4], %11     \n"
                "vst1.u8    {d7}, [%3], %11     \n"

                "subs       %0, #1              \n"

                "vst1.u8    {d4}, [%4], %11     \n"
                "vst1.u8    {d5}, [%3], %11     \n"
                "vst1.u8    {d2}, [%4], %11     \n"
                "vst1.u8    {d3}, [%3], %11     \n"
                "vst1.u8    {d0}, [%4], %11     \n"
                "vst1.u8    {d1}, [%3], %11     \n"

                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(src1), // %2
                "=r"(dst0), // %3
                "=r"(dst1)  // %4
                : "0"(nn),
                "1"(src0),
                "2"(src1),
                "3"(dst0),
                "4"(dst1),
                "r"(src_step), // %10
                "r"(dst_step)  // %11
                : "cc", "memory", "q0", "q1", "q2", "q3");
        }
#endif // __aarch64__
        for (; remain > 0; remain--)
        {
            dst0[0] = src1[0 + 3 * src_step];
            dst0[1] = src0[0 + 3 * src_step];
            dst0[2] = src1[0 + 2 * src_step];
            dst0[3] = src0[0 + 2 * src_step];
            dst0[4] = src1[0 + src_step];
            dst0[5] = src0[0 + src_step];
            dst0[6] = src1[0];
            dst0[7] = src0[0];

            src0 += 1;
            src1 += 1;

            dst0 += stride;
        }

        src0 += srcwgap + 7 * srcstride;
    }
#endif // __ARM_NEON
    for (; y < srch; y++)
    {
        unsigned char* dst0 = dstend - y - 1;

        int x = 0;
        for (; x < srcw; x++)
        {
            *dst0 = *src0;

            src0 += 1;
            dst0 += stride;
        }

        src0 += srcwgap;
    }
}

static void kanna_rotate_6_c2(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int /*h*/, int stride)
{
    const int srcwgap = srcstride - srcw * 2;

    // point to the last dst pixel in row
    unsigned char* dstend = dst + w * 2;

    const unsigned char* src0 = src;

    int y = 0;
#if __ARM_NEON
    for (; y + 7 < srch; y += 8)
    {
        const unsigned char* src1 = src0 + srcstride;

        unsigned char* dst0 = dstend - y * 2 - 8 * 2;
        unsigned char* dst1 = dstend - y * 2 - 8 * 2 + stride;

        int src_step = 2 * srcstride;
        int dst_step = 2 * stride;

        int nn = srcw >> 3;
        int remain = srcw - (nn << 3);

#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x8x2_t _src0 = vld2_u8(src0);
            uint8x8x2_t _src1 = vld2_u8(src1);

            uint8x8x2_t _src2 = vld2_u8(src0 + src_step);
            uint8x8x2_t _src3 = vld2_u8(src1 + src_step);

            uint8x8x2_t _src4 = vld2_u8(src0 + 2 * src_step);
            uint8x8x2_t _src5 = vld2_u8(src1 + 2 * src_step);

            uint8x8x2_t _src6 = vld2_u8(src0 + 3 * src_step);
            uint8x8x2_t _src7 = vld2_u8(src1 + 3 * src_step);

            uint8x8x2_t _src01t_r = vtrn_u8(_src1.val[0], _src0.val[0]);
            uint8x8x2_t _src23t_r = vtrn_u8(_src3.val[0], _src2.val[0]);
            uint8x8x2_t _src45t_r = vtrn_u8(_src5.val[0], _src4.val[0]);
            uint8x8x2_t _src67t_r = vtrn_u8(_src7.val[0], _src6.val[0]);

            uint8x8x2_t _src01t_g = vtrn_u8(_src1.val[1], _src0.val[1]);
            uint8x8x2_t _src23t_g = vtrn_u8(_src3.val[1], _src2.val[1]);
            uint8x8x2_t _src45t_g = vtrn_u8(_src5.val[1], _src4.val[1]);
            uint8x8x2_t _src67t_g = vtrn_u8(_src7.val[1], _src6.val[1]);

            uint16x4x2_t _src02tt_r = vtrn_u16(vreinterpret_u16_u8(_src23t_r.val[1]), vreinterpret_u16_u8(_src01t_r.val[1]));
            uint16x4x2_t _src13tt_r = vtrn_u16(vreinterpret_u16_u8(_src23t_r.val[0]), vreinterpret_u16_u8(_src01t_r.val[0]));
            uint16x4x2_t _src46tt_r = vtrn_u16(vreinterpret_u16_u8(_src67t_r.val[1]), vreinterpret_u16_u8(_src45t_r.val[1]));
            uint16x4x2_t _src57tt_r = vtrn_u16(vreinterpret_u16_u8(_src67t_r.val[0]), vreinterpret_u16_u8(_src45t_r.val[0]));

            uint16x4x2_t _src02tt_g = vtrn_u16(vreinterpret_u16_u8(_src23t_g.val[1]), vreinterpret_u16_u8(_src01t_g.val[1]));
            uint16x4x2_t _src13tt_g = vtrn_u16(vreinterpret_u16_u8(_src23t_g.val[0]), vreinterpret_u16_u8(_src01t_g.val[0]));
            uint16x4x2_t _src46tt_g = vtrn_u16(vreinterpret_u16_u8(_src67t_g.val[1]), vreinterpret_u16_u8(_src45t_g.val[1]));
            uint16x4x2_t _src57tt_g = vtrn_u16(vreinterpret_u16_u8(_src67t_g.val[0]), vreinterpret_u16_u8(_src45t_g.val[0]));

            uint32x2x2_t _src04ttt_r = vtrn_u32(vreinterpret_u32_u16(_src46tt_r.val[1]), vreinterpret_u32_u16(_src02tt_r.val[1]));
            uint32x2x2_t _src15ttt_r = vtrn_u32(vreinterpret_u32_u16(_src57tt_r.val[1]), vreinterpret_u32_u16(_src13tt_r.val[1]));
            uint32x2x2_t _src26ttt_r = vtrn_u32(vreinterpret_u32_u16(_src46tt_r.val[0]), vreinterpret_u32_u16(_src02tt_r.val[0]));
            uint32x2x2_t _src37ttt_r = vtrn_u32(vreinterpret_u32_u16(_src57tt_r.val[0]), vreinterpret_u32_u16(_src13tt_r.val[0]));

            uint32x2x2_t _src04ttt_g = vtrn_u32(vreinterpret_u32_u16(_src46tt_g.val[1]), vreinterpret_u32_u16(_src02tt_g.val[1]));
            uint32x2x2_t _src15ttt_g = vtrn_u32(vreinterpret_u32_u16(_src57tt_g.val[1]), vreinterpret_u32_u16(_src13tt_g.val[1]));
            uint32x2x2_t _src26ttt_g = vtrn_u32(vreinterpret_u32_u16(_src46tt_g.val[0]), vreinterpret_u32_u16(_src02tt_g.val[0]));
            uint32x2x2_t _src37ttt_g = vtrn_u32(vreinterpret_u32_u16(_src57tt_g.val[0]), vreinterpret_u32_u16(_src13tt_g.val[0]));

            uint8x8x2_t _dst0;
            uint8x8x2_t _dst1;
            uint8x8x2_t _dst2;
            uint8x8x2_t _dst3;
            uint8x8x2_t _dst4;
            uint8x8x2_t _dst5;
            uint8x8x2_t _dst6;
            uint8x8x2_t _dst7;

            _dst0.val[0] = vreinterpret_u8_u32(_src04ttt_r.val[1]);
            _dst1.val[0] = vreinterpret_u8_u32(_src15ttt_r.val[1]);
            _dst2.val[0] = vreinterpret_u8_u32(_src26ttt_r.val[1]);
            _dst3.val[0] = vreinterpret_u8_u32(_src37ttt_r.val[1]);
            _dst4.val[0] = vreinterpret_u8_u32(_src04ttt_r.val[0]);
            _dst5.val[0] = vreinterpret_u8_u32(_src15ttt_r.val[0]);
            _dst6.val[0] = vreinterpret_u8_u32(_src26ttt_r.val[0]);
            _dst7.val[0] = vreinterpret_u8_u32(_src37ttt_r.val[0]);

            _dst0.val[1] = vreinterpret_u8_u32(_src04ttt_g.val[1]);
            _dst1.val[1] = vreinterpret_u8_u32(_src15ttt_g.val[1]);
            _dst2.val[1] = vreinterpret_u8_u32(_src26ttt_g.val[1]);
            _dst3.val[1] = vreinterpret_u8_u32(_src37ttt_g.val[1]);
            _dst4.val[1] = vreinterpret_u8_u32(_src04ttt_g.val[0]);
            _dst5.val[1] = vreinterpret_u8_u32(_src15ttt_g.val[0]);
            _dst6.val[1] = vreinterpret_u8_u32(_src26ttt_g.val[0]);
            _dst7.val[1] = vreinterpret_u8_u32(_src37ttt_g.val[0]);

            vst2_u8(dst0, _dst7);
            vst2_u8(dst1, _dst6);
            vst2_u8(dst0 + dst_step, _dst5);
            vst2_u8(dst1 + dst_step, _dst4);
            vst2_u8(dst0 + 2 * dst_step, _dst3);
            vst2_u8(dst1 + 2 * dst_step, _dst2);
            vst2_u8(dst0 + 3 * dst_step, _dst1);
            vst2_u8(dst1 + 3 * dst_step, _dst0);

            src0 += 2 * 8;
            src1 += 2 * 8;

            dst0 += 4 * dst_step;
            dst1 += 4 * dst_step;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #128]          \n"
                "vld2.u8    {d0-d1}, [%1], %10  \n"

                "pld        [%2, #128]          \n"
                "vld2.u8    {d2-d3}, [%2], %10  \n"

                "pld        [%1, #128]          \n"
                "vld2.u8    {d4-d5}, [%1], %10  \n"

                "vtrn.u8    q1, q0              \n" // _src01t_r

                "pld        [%2, #128]          \n"
                "vld2.u8    {d6-d7}, [%2], %10  \n"

                "pld        [%1, #128]          \n"
                "vld2.u8    {d16-d17}, [%1], %10\n"

                "vtrn.u8    q3, q2              \n" // _src23t_r

                "pld        [%2, #128]          \n"
                "vld2.u8    {d18-d19}, [%2], %10\n"

                "pld        [%1, #128]          \n"
                "vld2.u8    {d20-d21}, [%1], %10\n"

                "vtrn.u8    q9, q8              \n" // _src45t_r

                "pld        [%2, #128]          \n"
                "vld2.u8    {d22-d23}, [%2], %10\n"

                "vtrn.u8    q11, q10            \n" // _src67t_r

                "sub        %1, %1, %10, lsl #2 \n" // restore src0

                "vtrn.u16   q2, q0              \n" // _src02tt_r

                "sub        %2, %2, %10, lsl #2 \n" // restore src1

                "vtrn.u16   q3, q1              \n" // _src13tt_r

                "add        %1, #16             \n" // src0 += 16

                "vtrn.u16   q10, q8             \n" // _src46tt_r

                "add        %2, #16             \n" // src1 += 16

                "vtrn.u16   q11, q9             \n" // _src57tt_r

                "vtrn.u32   q10, q2             \n" // _src26ttt_r

                "vtrn.u32   q11, q3             \n" // _src37ttt_r
                "vst2.u8    {d20-d21}, [%4], %11\n"

                "vtrn.u32   q8, q0              \n" // _src04ttt_r
                "vst2.u8    {d22-d23}, [%3], %11\n"

                "vtrn.u32   q9, q1              \n" // _src15ttt_r
                "vst2.u8    {d16-d17}, [%4], %11\n"

                "subs       %0, #1              \n"

                "vst2.u8    {d18-d19}, [%3], %11\n"
                "vst2.u8    {d4-d5}, [%4], %11  \n"
                "vst2.u8    {d6-d7}, [%3], %11  \n"
                "vst2.u8    {d0-d1}, [%4], %11  \n"
                "vst2.u8    {d2-d3}, [%3], %11  \n"

                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(src1), // %2
                "=r"(dst0), // %3
                "=r"(dst1)  // %4
                : "0"(nn),
                "1"(src0),
                "2"(src1),
                "3"(dst0),
                "4"(dst1),
                "r"(src_step), // %10
                "r"(dst_step)  // %11
                : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11");
        }
#endif // __aarch64__
        for (; remain > 0; remain--)
        {
            dst0[0] = src1[0 + 3 * src_step];
            dst0[1] = src1[1 + 3 * src_step];
            dst0[2] = src0[0 + 3 * src_step];
            dst0[3] = src0[1 + 3 * src_step];
            dst0[4] = src1[0 + 2 * src_step];
            dst0[5] = src1[1 + 2 * src_step];
            dst0[6] = src0[0 + 2 * src_step];
            dst0[7] = src0[1 + 2 * src_step];
            dst0[8] = src1[0 + src_step];
            dst0[9] = src1[1 + src_step];
            dst0[10] = src0[0 + src_step];
            dst0[11] = src0[1 + src_step];
            dst0[12] = src1[0];
            dst0[13] = src1[1];
            dst0[14] = src0[0];
            dst0[15] = src0[1];

            src0 += 2;
            src1 += 2;

            dst0 += stride;
        }

        src0 += srcwgap + 7 * srcstride;
    }
#endif // __ARM_NEON
    for (; y < srch; y++)
    {
        unsigned char* dst0 = dstend - y * 2 - 2;

        int x = 0;
        for (; x < srcw; x++)
        {
            dst0[0] = src0[0];
            dst0[1] = src0[1];

            src0 += 2;
            dst0 += stride;
        }

        src0 += srcwgap;
    }
}

static void kanna_rotate_6_c3(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int /*h*/, int stride)
{
    const int srcwgap = srcstride - srcw * 3;

    // point to the last dst pixel in row
    unsigned char* dstend = dst + w * 3;

    const unsigned char* src0 = src;

    int y = 0;
#if __ARM_NEON
    for (; y + 7 < srch; y += 8)
    {
        const unsigned char* src1 = src0 + srcstride;

        unsigned char* dst0 = dstend - y * 3 - 8 * 3;
        unsigned char* dst1 = dstend - y * 3 - 8 * 3 + stride;

        int src_step = 2 * srcstride;
        int dst_step = 2 * stride;

        int nn = srcw >> 3;
        int remain = srcw - (nn << 3);

#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x8x3_t _src0 = vld3_u8(src0);
            uint8x8x3_t _src1 = vld3_u8(src1);

            uint8x8x3_t _src2 = vld3_u8(src0 + src_step);
            uint8x8x3_t _src3 = vld3_u8(src1 + src_step);

            uint8x8x3_t _src4 = vld3_u8(src0 + 2 * src_step);
            uint8x8x3_t _src5 = vld3_u8(src1 + 2 * src_step);

            uint8x8x3_t _src6 = vld3_u8(src0 + 3 * src_step);
            uint8x8x3_t _src7 = vld3_u8(src1 + 3 * src_step);

            uint8x8x2_t _src01t_r = vtrn_u8(_src1.val[0], _src0.val[0]);
            uint8x8x2_t _src23t_r = vtrn_u8(_src3.val[0], _src2.val[0]);
            uint8x8x2_t _src45t_r = vtrn_u8(_src5.val[0], _src4.val[0]);
            uint8x8x2_t _src67t_r = vtrn_u8(_src7.val[0], _src6.val[0]);

            uint8x8x2_t _src01t_g = vtrn_u8(_src1.val[1], _src0.val[1]);
            uint8x8x2_t _src23t_g = vtrn_u8(_src3.val[1], _src2.val[1]);
            uint8x8x2_t _src45t_g = vtrn_u8(_src5.val[1], _src4.val[1]);
            uint8x8x2_t _src67t_g = vtrn_u8(_src7.val[1], _src6.val[1]);

            uint8x8x2_t _src01t_b = vtrn_u8(_src1.val[2], _src0.val[2]);
            uint8x8x2_t _src23t_b = vtrn_u8(_src3.val[2], _src2.val[2]);
            uint8x8x2_t _src45t_b = vtrn_u8(_src5.val[2], _src4.val[2]);
            uint8x8x2_t _src67t_b = vtrn_u8(_src7.val[2], _src6.val[2]);

            uint16x4x2_t _src02tt_r = vtrn_u16(vreinterpret_u16_u8(_src23t_r.val[1]), vreinterpret_u16_u8(_src01t_r.val[1]));
            uint16x4x2_t _src13tt_r = vtrn_u16(vreinterpret_u16_u8(_src23t_r.val[0]), vreinterpret_u16_u8(_src01t_r.val[0]));
            uint16x4x2_t _src46tt_r = vtrn_u16(vreinterpret_u16_u8(_src67t_r.val[1]), vreinterpret_u16_u8(_src45t_r.val[1]));
            uint16x4x2_t _src57tt_r = vtrn_u16(vreinterpret_u16_u8(_src67t_r.val[0]), vreinterpret_u16_u8(_src45t_r.val[0]));

            uint16x4x2_t _src02tt_g = vtrn_u16(vreinterpret_u16_u8(_src23t_g.val[1]), vreinterpret_u16_u8(_src01t_g.val[1]));
            uint16x4x2_t _src13tt_g = vtrn_u16(vreinterpret_u16_u8(_src23t_g.val[0]), vreinterpret_u16_u8(_src01t_g.val[0]));
            uint16x4x2_t _src46tt_g = vtrn_u16(vreinterpret_u16_u8(_src67t_g.val[1]), vreinterpret_u16_u8(_src45t_g.val[1]));
            uint16x4x2_t _src57tt_g = vtrn_u16(vreinterpret_u16_u8(_src67t_g.val[0]), vreinterpret_u16_u8(_src45t_g.val[0]));

            uint16x4x2_t _src02tt_b = vtrn_u16(vreinterpret_u16_u8(_src23t_b.val[1]), vreinterpret_u16_u8(_src01t_b.val[1]));
            uint16x4x2_t _src13tt_b = vtrn_u16(vreinterpret_u16_u8(_src23t_b.val[0]), vreinterpret_u16_u8(_src01t_b.val[0]));
            uint16x4x2_t _src46tt_b = vtrn_u16(vreinterpret_u16_u8(_src67t_b.val[1]), vreinterpret_u16_u8(_src45t_b.val[1]));
            uint16x4x2_t _src57tt_b = vtrn_u16(vreinterpret_u16_u8(_src67t_b.val[0]), vreinterpret_u16_u8(_src45t_b.val[0]));

            uint32x2x2_t _src04ttt_r = vtrn_u32(vreinterpret_u32_u16(_src46tt_r.val[1]), vreinterpret_u32_u16(_src02tt_r.val[1]));
            uint32x2x2_t _src15ttt_r = vtrn_u32(vreinterpret_u32_u16(_src57tt_r.val[1]), vreinterpret_u32_u16(_src13tt_r.val[1]));
            uint32x2x2_t _src26ttt_r = vtrn_u32(vreinterpret_u32_u16(_src46tt_r.val[0]), vreinterpret_u32_u16(_src02tt_r.val[0]));
            uint32x2x2_t _src37ttt_r = vtrn_u32(vreinterpret_u32_u16(_src57tt_r.val[0]), vreinterpret_u32_u16(_src13tt_r.val[0]));

            uint32x2x2_t _src04ttt_g = vtrn_u32(vreinterpret_u32_u16(_src46tt_g.val[1]), vreinterpret_u32_u16(_src02tt_g.val[1]));
            uint32x2x2_t _src15ttt_g = vtrn_u32(vreinterpret_u32_u16(_src57tt_g.val[1]), vreinterpret_u32_u16(_src13tt_g.val[1]));
            uint32x2x2_t _src26ttt_g = vtrn_u32(vreinterpret_u32_u16(_src46tt_g.val[0]), vreinterpret_u32_u16(_src02tt_g.val[0]));
            uint32x2x2_t _src37ttt_g = vtrn_u32(vreinterpret_u32_u16(_src57tt_g.val[0]), vreinterpret_u32_u16(_src13tt_g.val[0]));

            uint32x2x2_t _src04ttt_b = vtrn_u32(vreinterpret_u32_u16(_src46tt_b.val[1]), vreinterpret_u32_u16(_src02tt_b.val[1]));
            uint32x2x2_t _src15ttt_b = vtrn_u32(vreinterpret_u32_u16(_src57tt_b.val[1]), vreinterpret_u32_u16(_src13tt_b.val[1]));
            uint32x2x2_t _src26ttt_b = vtrn_u32(vreinterpret_u32_u16(_src46tt_b.val[0]), vreinterpret_u32_u16(_src02tt_b.val[0]));
            uint32x2x2_t _src37ttt_b = vtrn_u32(vreinterpret_u32_u16(_src57tt_b.val[0]), vreinterpret_u32_u16(_src13tt_b.val[0]));

            uint8x8x3_t _dst0;
            uint8x8x3_t _dst1;
            uint8x8x3_t _dst2;
            uint8x8x3_t _dst3;
            uint8x8x3_t _dst4;
            uint8x8x3_t _dst5;
            uint8x8x3_t _dst6;
            uint8x8x3_t _dst7;

            _dst0.val[0] = vreinterpret_u8_u32(_src04ttt_r.val[1]);
            _dst1.val[0] = vreinterpret_u8_u32(_src15ttt_r.val[1]);
            _dst2.val[0] = vreinterpret_u8_u32(_src26ttt_r.val[1]);
            _dst3.val[0] = vreinterpret_u8_u32(_src37ttt_r.val[1]);
            _dst4.val[0] = vreinterpret_u8_u32(_src04ttt_r.val[0]);
            _dst5.val[0] = vreinterpret_u8_u32(_src15ttt_r.val[0]);
            _dst6.val[0] = vreinterpret_u8_u32(_src26ttt_r.val[0]);
            _dst7.val[0] = vreinterpret_u8_u32(_src37ttt_r.val[0]);

            _dst0.val[1] = vreinterpret_u8_u32(_src04ttt_g.val[1]);
            _dst1.val[1] = vreinterpret_u8_u32(_src15ttt_g.val[1]);
            _dst2.val[1] = vreinterpret_u8_u32(_src26ttt_g.val[1]);
            _dst3.val[1] = vreinterpret_u8_u32(_src37ttt_g.val[1]);
            _dst4.val[1] = vreinterpret_u8_u32(_src04ttt_g.val[0]);
            _dst5.val[1] = vreinterpret_u8_u32(_src15ttt_g.val[0]);
            _dst6.val[1] = vreinterpret_u8_u32(_src26ttt_g.val[0]);
            _dst7.val[1] = vreinterpret_u8_u32(_src37ttt_g.val[0]);

            _dst0.val[2] = vreinterpret_u8_u32(_src04ttt_b.val[1]);
            _dst1.val[2] = vreinterpret_u8_u32(_src15ttt_b.val[1]);
            _dst2.val[2] = vreinterpret_u8_u32(_src26ttt_b.val[1]);
            _dst3.val[2] = vreinterpret_u8_u32(_src37ttt_b.val[1]);
            _dst4.val[2] = vreinterpret_u8_u32(_src04ttt_b.val[0]);
            _dst5.val[2] = vreinterpret_u8_u32(_src15ttt_b.val[0]);
            _dst6.val[2] = vreinterpret_u8_u32(_src26ttt_b.val[0]);
            _dst7.val[2] = vreinterpret_u8_u32(_src37ttt_b.val[0]);

            vst3_u8(dst0, _dst7);
            vst3_u8(dst1, _dst6);
            vst3_u8(dst0 + dst_step, _dst5);
            vst3_u8(dst1 + dst_step, _dst4);
            vst3_u8(dst0 + 2 * dst_step, _dst3);
            vst3_u8(dst1 + 2 * dst_step, _dst2);
            vst3_u8(dst0 + 3 * dst_step, _dst1);
            vst3_u8(dst1 + 3 * dst_step, _dst0);

            src0 += 3 * 8;
            src1 += 3 * 8;

            dst0 += 4 * dst_step;
            dst1 += 4 * dst_step;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #192]          \n"
                "vld3.u8    {d0-d2}, [%1], %10  \n"

                "pld        [%2, #192]          \n"
                "vld3.u8    {d4-d6}, [%2], %10  \n"

                "pld        [%1, #192]          \n"
                "vld3.u8    {d8-d10}, [%1], %10 \n"

                "vtrn.u8    q2, q0              \n" // _src01t_r
                "vtrn.u8    d6, d2              \n"

                "pld        [%2, #192]          \n"
                "vld3.u8    {d12-d14}, [%2], %10\n"

                "pld        [%1, #192]          \n"
                "vld3.u8    {d16-d18}, [%1], %10\n"

                "vtrn.u8    q6, q4              \n" // _src23t_r
                "vtrn.u8    d14, d10            \n"

                "pld        [%2, #192]          \n"
                "vld3.u8    {d20-d22}, [%2], %10\n"

                "pld        [%1, #192]          \n"
                "vld3.u8    {d24-d26}, [%1], %10\n"

                "vtrn.u8    q10, q8             \n" // _src45t_r
                "vtrn.u8    d22, d18            \n"

                "pld        [%2, #192]          \n"
                "vld3.u8    {d28-d30}, [%2], %10\n"

                "vtrn.u8    q14, q12            \n" // _src67t_r
                "vtrn.u8    d30, d26            \n"

                "sub        %1, %1, %10, lsl #2 \n" // restore src0

                "vtrn.u16   q4, q0              \n" // _src02tt_r
                "vtrn.u16   d10, d2             \n"

                "sub        %2, %2, %10, lsl #2 \n" // restore src1

                "vtrn.u16   q6, q2              \n" // _src13tt_r
                "vtrn.u16   d14, d6             \n"

                "add        %1, #24             \n" // src0 += 24

                "vtrn.u16   q12, q8             \n" // _src46tt_r
                "vtrn.u16   d26, d18            \n"

                "add        %2, #24             \n" // src1 += 24

                "vtrn.u16   q14, q10            \n" // _src57tt_r
                "vtrn.u16   d30, d22            \n"

                "vtrn.u32   q12, q4             \n" // _src26ttt_r
                "vtrn.u32   d26, d10            \n"

                "vtrn.u32   q14, q6             \n" // _src37ttt_r
                "vst3.u8    {d24-d26}, [%4], %11\n"
                "vtrn.u32   d30, d14            \n"

                "vtrn.u32   q8, q0              \n" // _src04ttt_r
                "vst3.u8    {d28-d30}, [%3], %11\n"
                "vtrn.u32   d18, d2             \n"

                "vtrn.u32   q10, q2             \n" // _src15ttt_r
                "vst3.u8    {d16-d18}, [%4], %11\n"
                "vtrn.u32   d22, d6             \n"

                "subs       %0, #1              \n"

                "vst3.u8    {d20-d22}, [%3], %11\n"
                "vst3.u8    {d8-d10}, [%4], %11 \n"
                "vst3.u8    {d12-d14}, [%3], %11\n"
                "vst3.u8    {d0-d2}, [%4], %11  \n"
                "vst3.u8    {d4-d6}, [%3], %11  \n"

                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(src1), // %2
                "=r"(dst0), // %3
                "=r"(dst1)  // %4
                : "0"(nn),
                "1"(src0),
                "2"(src1),
                "3"(dst0),
                "4"(dst1),
                "r"(src_step), // %10
                "r"(dst_step)  // %11
                : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
        }
#endif // __aarch64__
        for (; remain > 0; remain--)
        {
            dst0[0] = src1[0 + 3 * src_step];
            dst0[1] = src1[1 + 3 * src_step];
            dst0[2] = src1[2 + 3 * src_step];
            dst0[3] = src0[0 + 3 * src_step];
            dst0[4] = src0[1 + 3 * src_step];
            dst0[5] = src0[2 + 3 * src_step];
            dst0[6] = src1[0 + 2 * src_step];
            dst0[7] = src1[1 + 2 * src_step];
            dst0[8] = src1[2 + 2 * src_step];
            dst0[9] = src0[0 + 2 * src_step];
            dst0[10] = src0[1 + 2 * src_step];
            dst0[11] = src0[2 + 2 * src_step];
            dst0[12] = src1[0 + src_step];
            dst0[13] = src1[1 + src_step];
            dst0[14] = src1[2 + src_step];
            dst0[15] = src0[0 + src_step];
            dst0[16] = src0[1 + src_step];
            dst0[17] = src0[2 + src_step];
            dst0[18] = src1[0];
            dst0[19] = src1[1];
            dst0[20] = src1[2];
            dst0[21] = src0[0];
            dst0[22] = src0[1];
            dst0[23] = src0[2];

            src0 += 3;
            src1 += 3;

            dst0 += stride;
        }

        src0 += srcwgap + 7 * srcstride;
    }
#endif // __ARM_NEON
    for (; y < srch; y++)
    {
        unsigned char* dst0 = dstend - y * 3 - 3;

        int x = 0;
        for (; x < srcw; x++)
        {
            dst0[0] = src0[0];
            dst0[1] = src0[1];
            dst0[2] = src0[2];

            src0 += 3;
            dst0 += stride;
        }

        src0 += srcwgap;
    }
}

static void kanna_rotate_6_c4(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int /*h*/, int stride)
{
    const int srcwgap = srcstride - srcw * 4;

    // point to the last dst pixel in row
    unsigned char* dstend = dst + w * 4;

    const unsigned char* src0 = src;

    int y = 0;
#if __ARM_NEON
    for (; y + 7 < srch; y += 8)
    {
        const unsigned char* src1 = src0 + srcstride;

        unsigned char* dst0 = dstend - y * 4 - 8 * 4;
        unsigned char* dst1 = dstend - y * 4 - 8 * 4 + stride;

        int src_step = 2 * srcstride;
        int dst_step = 2 * stride;

        int nn = srcw >> 3;
        int remain = srcw - (nn << 3);

#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x8x4_t _src0 = vld4_u8(src0);
            uint8x8x4_t _src1 = vld4_u8(src1);

            uint8x8x4_t _src2 = vld4_u8(src0 + src_step);
            uint8x8x4_t _src3 = vld4_u8(src1 + src_step);

            uint8x8x4_t _src4 = vld4_u8(src0 + 2 * src_step);
            uint8x8x4_t _src5 = vld4_u8(src1 + 2 * src_step);

            uint8x8x4_t _src6 = vld4_u8(src0 + 3 * src_step);
            uint8x8x4_t _src7 = vld4_u8(src1 + 3 * src_step);

            uint8x8x2_t _src01t_r = vtrn_u8(_src1.val[0], _src0.val[0]);
            uint8x8x2_t _src23t_r = vtrn_u8(_src3.val[0], _src2.val[0]);
            uint8x8x2_t _src45t_r = vtrn_u8(_src5.val[0], _src4.val[0]);
            uint8x8x2_t _src67t_r = vtrn_u8(_src7.val[0], _src6.val[0]);

            uint8x8x2_t _src01t_g = vtrn_u8(_src1.val[1], _src0.val[1]);
            uint8x8x2_t _src23t_g = vtrn_u8(_src3.val[1], _src2.val[1]);
            uint8x8x2_t _src45t_g = vtrn_u8(_src5.val[1], _src4.val[1]);
            uint8x8x2_t _src67t_g = vtrn_u8(_src7.val[1], _src6.val[1]);

            uint8x8x2_t _src01t_b = vtrn_u8(_src1.val[2], _src0.val[2]);
            uint8x8x2_t _src23t_b = vtrn_u8(_src3.val[2], _src2.val[2]);
            uint8x8x2_t _src45t_b = vtrn_u8(_src5.val[2], _src4.val[2]);
            uint8x8x2_t _src67t_b = vtrn_u8(_src7.val[2], _src6.val[2]);

            uint8x8x2_t _src01t_a = vtrn_u8(_src1.val[3], _src0.val[3]);
            uint8x8x2_t _src23t_a = vtrn_u8(_src3.val[3], _src2.val[3]);
            uint8x8x2_t _src45t_a = vtrn_u8(_src5.val[3], _src4.val[3]);
            uint8x8x2_t _src67t_a = vtrn_u8(_src7.val[3], _src6.val[3]);

            uint16x4x2_t _src02tt_r = vtrn_u16(vreinterpret_u16_u8(_src23t_r.val[1]), vreinterpret_u16_u8(_src01t_r.val[1]));
            uint16x4x2_t _src13tt_r = vtrn_u16(vreinterpret_u16_u8(_src23t_r.val[0]), vreinterpret_u16_u8(_src01t_r.val[0]));
            uint16x4x2_t _src46tt_r = vtrn_u16(vreinterpret_u16_u8(_src67t_r.val[1]), vreinterpret_u16_u8(_src45t_r.val[1]));
            uint16x4x2_t _src57tt_r = vtrn_u16(vreinterpret_u16_u8(_src67t_r.val[0]), vreinterpret_u16_u8(_src45t_r.val[0]));

            uint16x4x2_t _src02tt_g = vtrn_u16(vreinterpret_u16_u8(_src23t_g.val[1]), vreinterpret_u16_u8(_src01t_g.val[1]));
            uint16x4x2_t _src13tt_g = vtrn_u16(vreinterpret_u16_u8(_src23t_g.val[0]), vreinterpret_u16_u8(_src01t_g.val[0]));
            uint16x4x2_t _src46tt_g = vtrn_u16(vreinterpret_u16_u8(_src67t_g.val[1]), vreinterpret_u16_u8(_src45t_g.val[1]));
            uint16x4x2_t _src57tt_g = vtrn_u16(vreinterpret_u16_u8(_src67t_g.val[0]), vreinterpret_u16_u8(_src45t_g.val[0]));

            uint16x4x2_t _src02tt_b = vtrn_u16(vreinterpret_u16_u8(_src23t_b.val[1]), vreinterpret_u16_u8(_src01t_b.val[1]));
            uint16x4x2_t _src13tt_b = vtrn_u16(vreinterpret_u16_u8(_src23t_b.val[0]), vreinterpret_u16_u8(_src01t_b.val[0]));
            uint16x4x2_t _src46tt_b = vtrn_u16(vreinterpret_u16_u8(_src67t_b.val[1]), vreinterpret_u16_u8(_src45t_b.val[1]));
            uint16x4x2_t _src57tt_b = vtrn_u16(vreinterpret_u16_u8(_src67t_b.val[0]), vreinterpret_u16_u8(_src45t_b.val[0]));

            uint16x4x2_t _src02tt_a = vtrn_u16(vreinterpret_u16_u8(_src23t_a.val[1]), vreinterpret_u16_u8(_src01t_a.val[1]));
            uint16x4x2_t _src13tt_a = vtrn_u16(vreinterpret_u16_u8(_src23t_a.val[0]), vreinterpret_u16_u8(_src01t_a.val[0]));
            uint16x4x2_t _src46tt_a = vtrn_u16(vreinterpret_u16_u8(_src67t_a.val[1]), vreinterpret_u16_u8(_src45t_a.val[1]));
            uint16x4x2_t _src57tt_a = vtrn_u16(vreinterpret_u16_u8(_src67t_a.val[0]), vreinterpret_u16_u8(_src45t_a.val[0]));

            uint32x2x2_t _src04ttt_r = vtrn_u32(vreinterpret_u32_u16(_src46tt_r.val[1]), vreinterpret_u32_u16(_src02tt_r.val[1]));
            uint32x2x2_t _src15ttt_r = vtrn_u32(vreinterpret_u32_u16(_src57tt_r.val[1]), vreinterpret_u32_u16(_src13tt_r.val[1]));
            uint32x2x2_t _src26ttt_r = vtrn_u32(vreinterpret_u32_u16(_src46tt_r.val[0]), vreinterpret_u32_u16(_src02tt_r.val[0]));
            uint32x2x2_t _src37ttt_r = vtrn_u32(vreinterpret_u32_u16(_src57tt_r.val[0]), vreinterpret_u32_u16(_src13tt_r.val[0]));

            uint32x2x2_t _src04ttt_g = vtrn_u32(vreinterpret_u32_u16(_src46tt_g.val[1]), vreinterpret_u32_u16(_src02tt_g.val[1]));
            uint32x2x2_t _src15ttt_g = vtrn_u32(vreinterpret_u32_u16(_src57tt_g.val[1]), vreinterpret_u32_u16(_src13tt_g.val[1]));
            uint32x2x2_t _src26ttt_g = vtrn_u32(vreinterpret_u32_u16(_src46tt_g.val[0]), vreinterpret_u32_u16(_src02tt_g.val[0]));
            uint32x2x2_t _src37ttt_g = vtrn_u32(vreinterpret_u32_u16(_src57tt_g.val[0]), vreinterpret_u32_u16(_src13tt_g.val[0]));

            uint32x2x2_t _src04ttt_b = vtrn_u32(vreinterpret_u32_u16(_src46tt_b.val[1]), vreinterpret_u32_u16(_src02tt_b.val[1]));
            uint32x2x2_t _src15ttt_b = vtrn_u32(vreinterpret_u32_u16(_src57tt_b.val[1]), vreinterpret_u32_u16(_src13tt_b.val[1]));
            uint32x2x2_t _src26ttt_b = vtrn_u32(vreinterpret_u32_u16(_src46tt_b.val[0]), vreinterpret_u32_u16(_src02tt_b.val[0]));
            uint32x2x2_t _src37ttt_b = vtrn_u32(vreinterpret_u32_u16(_src57tt_b.val[0]), vreinterpret_u32_u16(_src13tt_b.val[0]));

            uint32x2x2_t _src04ttt_a = vtrn_u32(vreinterpret_u32_u16(_src46tt_a.val[1]), vreinterpret_u32_u16(_src02tt_a.val[1]));
            uint32x2x2_t _src15ttt_a = vtrn_u32(vreinterpret_u32_u16(_src57tt_a.val[1]), vreinterpret_u32_u16(_src13tt_a.val[1]));
            uint32x2x2_t _src26ttt_a = vtrn_u32(vreinterpret_u32_u16(_src46tt_a.val[0]), vreinterpret_u32_u16(_src02tt_a.val[0]));
            uint32x2x2_t _src37ttt_a = vtrn_u32(vreinterpret_u32_u16(_src57tt_a.val[0]), vreinterpret_u32_u16(_src13tt_a.val[0]));

            uint8x8x4_t _dst0;
            uint8x8x4_t _dst1;
            uint8x8x4_t _dst2;
            uint8x8x4_t _dst3;
            uint8x8x4_t _dst4;
            uint8x8x4_t _dst5;
            uint8x8x4_t _dst6;
            uint8x8x4_t _dst7;

            _dst0.val[0] = vreinterpret_u8_u32(_src04ttt_r.val[1]);
            _dst1.val[0] = vreinterpret_u8_u32(_src15ttt_r.val[1]);
            _dst2.val[0] = vreinterpret_u8_u32(_src26ttt_r.val[1]);
            _dst3.val[0] = vreinterpret_u8_u32(_src37ttt_r.val[1]);
            _dst4.val[0] = vreinterpret_u8_u32(_src04ttt_r.val[0]);
            _dst5.val[0] = vreinterpret_u8_u32(_src15ttt_r.val[0]);
            _dst6.val[0] = vreinterpret_u8_u32(_src26ttt_r.val[0]);
            _dst7.val[0] = vreinterpret_u8_u32(_src37ttt_r.val[0]);

            _dst0.val[1] = vreinterpret_u8_u32(_src04ttt_g.val[1]);
            _dst1.val[1] = vreinterpret_u8_u32(_src15ttt_g.val[1]);
            _dst2.val[1] = vreinterpret_u8_u32(_src26ttt_g.val[1]);
            _dst3.val[1] = vreinterpret_u8_u32(_src37ttt_g.val[1]);
            _dst4.val[1] = vreinterpret_u8_u32(_src04ttt_g.val[0]);
            _dst5.val[1] = vreinterpret_u8_u32(_src15ttt_g.val[0]);
            _dst6.val[1] = vreinterpret_u8_u32(_src26ttt_g.val[0]);
            _dst7.val[1] = vreinterpret_u8_u32(_src37ttt_g.val[0]);

            _dst0.val[2] = vreinterpret_u8_u32(_src04ttt_b.val[1]);
            _dst1.val[2] = vreinterpret_u8_u32(_src15ttt_b.val[1]);
            _dst2.val[2] = vreinterpret_u8_u32(_src26ttt_b.val[1]);
            _dst3.val[2] = vreinterpret_u8_u32(_src37ttt_b.val[1]);
            _dst4.val[2] = vreinterpret_u8_u32(_src04ttt_b.val[0]);
            _dst5.val[2] = vreinterpret_u8_u32(_src15ttt_b.val[0]);
            _dst6.val[2] = vreinterpret_u8_u32(_src26ttt_b.val[0]);
            _dst7.val[2] = vreinterpret_u8_u32(_src37ttt_b.val[0]);

            _dst0.val[3] = vreinterpret_u8_u32(_src04ttt_a.val[1]);
            _dst1.val[3] = vreinterpret_u8_u32(_src15ttt_a.val[1]);
            _dst2.val[3] = vreinterpret_u8_u32(_src26ttt_a.val[1]);
            _dst3.val[3] = vreinterpret_u8_u32(_src37ttt_a.val[1]);
            _dst4.val[3] = vreinterpret_u8_u32(_src04ttt_a.val[0]);
            _dst5.val[3] = vreinterpret_u8_u32(_src15ttt_a.val[0]);
            _dst6.val[3] = vreinterpret_u8_u32(_src26ttt_a.val[0]);
            _dst7.val[3] = vreinterpret_u8_u32(_src37ttt_a.val[0]);

            vst4_u8(dst0, _dst7);
            vst4_u8(dst1, _dst6);
            vst4_u8(dst0 + dst_step, _dst5);
            vst4_u8(dst1 + dst_step, _dst4);
            vst4_u8(dst0 + 2 * dst_step, _dst3);
            vst4_u8(dst1 + 2 * dst_step, _dst2);
            vst4_u8(dst0 + 3 * dst_step, _dst1);
            vst4_u8(dst1 + 3 * dst_step, _dst0);

            src0 += 4 * 8;
            src1 += 4 * 8;

            dst0 += 4 * dst_step;
            dst1 += 4 * dst_step;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "vld4.u8    {d0-d3}, [%1], %10  \n"

                "pld        [%2, #256]          \n"
                "vld4.u8    {d4-d7}, [%2], %10  \n"

                "pld        [%1, #256]          \n"
                "vld4.u8    {d8-d11}, [%1], %10 \n"

                "vtrn.u8    q2, q0              \n" // _src01t_r
                "vtrn.u8    q3, q1              \n"

                "pld        [%2, #256]          \n"
                "vld4.u8    {d12-d15}, [%2], %10\n"

                "pld        [%1, #256]          \n"
                "vld4.u8    {d16-d19}, [%1], %10\n"

                "vtrn.u8    q6, q4              \n" // _src23t_r
                "vtrn.u8    q7, q5              \n"

                "pld        [%2, #256]          \n"
                "vld4.u8    {d20-d23}, [%2], %10\n"

                "pld        [%1, #256]          \n"
                "vld4.u8    {d24-d27}, [%1], %10\n"

                "vtrn.u8    q10, q8             \n" // _src45t_r
                "vtrn.u8    q11, q9             \n"

                "pld        [%2, #256]          \n"
                "vld4.u8    {d28-d31}, [%2], %10\n"

                "vtrn.u8    q14, q12            \n" // _src67t_r
                "vtrn.u8    q15, q13            \n"

                "sub        %1, %1, %10, lsl #2 \n" // restore src0

                "vtrn.u16   q4, q0              \n" // _src02tt_r
                "vtrn.u16   q5, q1              \n"

                "sub        %2, %2, %10, lsl #2 \n" // restore src1

                "vtrn.u16   q6, q2              \n" // _src13tt_r
                "vtrn.u16   q7, q3              \n"

                "add        %1, #32             \n" // src0 += 32

                "vtrn.u16   q12, q8             \n" // _src46tt_r
                "vtrn.u16   q13, q9             \n"

                "add        %2, #32             \n" // src1 += 32

                "vtrn.u16   q14, q10            \n" // _src57tt_r
                "vtrn.u16   q15, q11            \n"

                "vtrn.u32   q12, q4             \n" // _src26ttt_r
                "vtrn.u32   q13, q5             \n"

                "vtrn.u32   q14, q6             \n" // _src37ttt_r
                "vst4.u8    {d24-d27}, [%4], %11\n"
                "vtrn.u32   q15, q7             \n"

                "vtrn.u32   q8, q0              \n" // _src04ttt_r
                "vst4.u8    {d28-d31}, [%3], %11\n"
                "vtrn.u32   q9, q1              \n"

                "vtrn.u32   q10, q2             \n" // _src15ttt_r
                "vst4.u8    {d16-d19}, [%4], %11\n"
                "vtrn.u32   q11, q3             \n"

                "subs       %0, #1              \n"

                "vst4.u8    {d8-d11}, [%4], %11 \n"
                "vst4.u8    {d20-d23}, [%3], %11\n"
                "vst4.u8    {d12-d15}, [%3], %11\n"
                "vst4.u8    {d0-d3}, [%4], %11  \n"
                "vst4.u8    {d4-d7}, [%3], %11  \n"

                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(src1), // %2
                "=r"(dst0), // %3
                "=r"(dst1)  // %4
                : "0"(nn),
                "1"(src0),
                "2"(src1),
                "3"(dst0),
                "4"(dst1),
                "r"(src_step), // %10
                "r"(dst_step)  // %11
                : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
        }
#endif // __aarch64__
        for (; remain > 0; remain--)
        {
            dst0[0] = src1[0 + 3 * src_step];
            dst0[1] = src1[1 + 3 * src_step];
            dst0[2] = src1[2 + 3 * src_step];
            dst0[3] = src1[3 + 3 * src_step];
            dst0[4] = src0[0 + 3 * src_step];
            dst0[5] = src0[1 + 3 * src_step];
            dst0[6] = src0[2 + 3 * src_step];
            dst0[7] = src0[3 + 3 * src_step];
            dst0[8] = src1[0 + 2 * src_step];
            dst0[9] = src1[1 + 2 * src_step];
            dst0[10] = src1[2 + 2 * src_step];
            dst0[11] = src1[3 + 2 * src_step];
            dst0[12] = src0[0 + 2 * src_step];
            dst0[13] = src0[1 + 2 * src_step];
            dst0[14] = src0[2 + 2 * src_step];
            dst0[15] = src0[3 + 2 * src_step];
            dst0[16] = src1[0 + src_step];
            dst0[17] = src1[1 + src_step];
            dst0[18] = src1[2 + src_step];
            dst0[19] = src1[3 + src_step];
            dst0[20] = src0[0 + src_step];
            dst0[21] = src0[1 + src_step];
            dst0[22] = src0[2 + src_step];
            dst0[23] = src0[3 + src_step];
            dst0[24] = src1[0];
            dst0[25] = src1[1];
            dst0[26] = src1[2];
            dst0[27] = src1[3];
            dst0[28] = src0[0];
            dst0[29] = src0[1];
            dst0[30] = src0[2];
            dst0[31] = src0[3];

            src0 += 4;
            src1 += 4;

            dst0 += stride;
        }

        src0 += srcwgap + 7 * srcstride;
    }
#endif // __ARM_NEON
    for (; y < srch; y++)
    {
        unsigned char* dst0 = dstend - y * 4 - 4;

        int x = 0;
        for (; x < srcw; x++)
        {
            dst0[0] = src0[0];
            dst0[1] = src0[1];
            dst0[2] = src0[2];
            dst0[3] = src0[3];

            src0 += 4;
            dst0 += stride;
        }

        src0 += srcwgap;
    }
}

static void kanna_rotate_7_c1(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    const int srcwgap = srcstride - srcw;

    // point to the last dst pixel
    unsigned char* dstend = dst + stride * (h - 1) + w;

    const unsigned char* src0 = src;

    int y = 0;
#if __ARM_NEON
    for (; y + 7 < srch; y += 8)
    {
        const unsigned char* src1 = src0 + srcstride;

        unsigned char* dst6 = dstend - y - 8 - stride;
        unsigned char* dst7 = dstend - y - 8;

        int src_step = 2 * srcstride;
        int dst_step = -2 * stride;

        int nn = srcw >> 3;
        int remain = srcw - (nn << 3);

#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x8_t _src0 = vld1_u8(src0);
            uint8x8_t _src1 = vld1_u8(src1);

            uint8x8_t _src2 = vld1_u8(src0 + src_step);
            uint8x8_t _src3 = vld1_u8(src1 + src_step);

            uint8x8_t _src4 = vld1_u8(src0 + 2 * src_step);
            uint8x8_t _src5 = vld1_u8(src1 + 2 * src_step);

            uint8x8_t _src6 = vld1_u8(src0 + 3 * src_step);
            uint8x8_t _src7 = vld1_u8(src1 + 3 * src_step);

            uint8x8x2_t _src01t_r = vtrn_u8(_src1, _src0);
            uint8x8x2_t _src23t_r = vtrn_u8(_src3, _src2);
            uint8x8x2_t _src45t_r = vtrn_u8(_src5, _src4);
            uint8x8x2_t _src67t_r = vtrn_u8(_src7, _src6);

            uint16x4x2_t _src02tt_r = vtrn_u16(vreinterpret_u16_u8(_src23t_r.val[1]), vreinterpret_u16_u8(_src01t_r.val[1]));
            uint16x4x2_t _src13tt_r = vtrn_u16(vreinterpret_u16_u8(_src23t_r.val[0]), vreinterpret_u16_u8(_src01t_r.val[0]));
            uint16x4x2_t _src46tt_r = vtrn_u16(vreinterpret_u16_u8(_src67t_r.val[1]), vreinterpret_u16_u8(_src45t_r.val[1]));
            uint16x4x2_t _src57tt_r = vtrn_u16(vreinterpret_u16_u8(_src67t_r.val[0]), vreinterpret_u16_u8(_src45t_r.val[0]));

            uint32x2x2_t _src04ttt_r = vtrn_u32(vreinterpret_u32_u16(_src46tt_r.val[1]), vreinterpret_u32_u16(_src02tt_r.val[1]));
            uint32x2x2_t _src15ttt_r = vtrn_u32(vreinterpret_u32_u16(_src57tt_r.val[1]), vreinterpret_u32_u16(_src13tt_r.val[1]));
            uint32x2x2_t _src26ttt_r = vtrn_u32(vreinterpret_u32_u16(_src46tt_r.val[0]), vreinterpret_u32_u16(_src02tt_r.val[0]));
            uint32x2x2_t _src37ttt_r = vtrn_u32(vreinterpret_u32_u16(_src57tt_r.val[0]), vreinterpret_u32_u16(_src13tt_r.val[0]));

            uint8x8_t _dst0 = vreinterpret_u8_u32(_src04ttt_r.val[1]);
            uint8x8_t _dst1 = vreinterpret_u8_u32(_src15ttt_r.val[1]);
            uint8x8_t _dst2 = vreinterpret_u8_u32(_src26ttt_r.val[1]);
            uint8x8_t _dst3 = vreinterpret_u8_u32(_src37ttt_r.val[1]);
            uint8x8_t _dst4 = vreinterpret_u8_u32(_src04ttt_r.val[0]);
            uint8x8_t _dst5 = vreinterpret_u8_u32(_src15ttt_r.val[0]);
            uint8x8_t _dst6 = vreinterpret_u8_u32(_src26ttt_r.val[0]);
            uint8x8_t _dst7 = vreinterpret_u8_u32(_src37ttt_r.val[0]);

            vst1_u8(dst7, _dst7);
            vst1_u8(dst6, _dst6);
            vst1_u8(dst7 + dst_step, _dst5);
            vst1_u8(dst6 + dst_step, _dst4);
            vst1_u8(dst7 + 2 * dst_step, _dst3);
            vst1_u8(dst6 + 2 * dst_step, _dst2);
            vst1_u8(dst7 + 3 * dst_step, _dst1);
            vst1_u8(dst6 + 3 * dst_step, _dst0);

            src0 += 8;
            src1 += 8;

            dst7 += 4 * dst_step;
            dst6 += 4 * dst_step;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #64]           \n"
                "vld1.u8    {d0}, [%1], %10     \n"

                "pld        [%2, #64]           \n"
                "vld1.u8    {d1}, [%2], %10     \n"

                "pld        [%1, #64]           \n"
                "vld1.u8    {d2}, [%1], %10     \n"

                "vtrn.u8    d1, d0              \n" // _src01t_r

                "pld        [%2, #64]           \n"
                "vld1.u8    {d3}, [%2], %10     \n"

                "pld        [%1, #64]           \n"
                "vld1.u8    {d4}, [%1], %10     \n"

                "vtrn.u8    d3, d2              \n" // _src23t_r

                "pld        [%2, #64]           \n"
                "vld1.u8    {d5}, [%2], %10     \n"

                "pld        [%1, #64]           \n"
                "vld1.u8    {d6}, [%1], %10     \n"

                "vtrn.u8    d5, d4              \n" // _src45t_r

                "pld        [%2, #64]           \n"
                "vld1.u8    {d7}, [%2], %10     \n"

                "vtrn.u8    d7, d6              \n" // _src67t_r

                "sub        %1, %1, %10, lsl #2 \n" // restore src0

                "vtrn.u16   q1, q0              \n" // _src02tt_r _src13tt_r

                "sub        %2, %2, %10, lsl #2 \n" // restore src1

                "vtrn.u16   q3, q2              \n" // _src46tt_r _src57tt_r

                "add        %1, #8              \n" // src0 += 8

                "vtrn.u32   q3, q1              \n" // _src26ttt_r _src37ttt_r

                "add        %2, #8              \n" // src1 += 8

                "vtrn.u32   q2, q0              \n" // _src04ttt_r _src15ttt_r
                "vst1.u8    {d6}, [%4], %11     \n"
                "vst1.u8    {d7}, [%3], %11     \n"

                "subs       %0, #1              \n"

                "vst1.u8    {d4}, [%4], %11     \n"
                "vst1.u8    {d5}, [%3], %11     \n"
                "vst1.u8    {d2}, [%4], %11     \n"
                "vst1.u8    {d3}, [%3], %11     \n"
                "vst1.u8    {d0}, [%4], %11     \n"
                "vst1.u8    {d1}, [%3], %11     \n"

                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(src1), // %2
                "=r"(dst7), // %3
                "=r"(dst6)  // %4
                : "0"(nn),
                "1"(src0),
                "2"(src1),
                "3"(dst7),
                "4"(dst6),
                "r"(src_step), // %10
                "r"(dst_step)  // %11
                : "cc", "memory", "q0", "q1", "q2", "q3");
        }
#endif // __aarch64__
        for (; remain > 0; remain--)
        {
            dst7[0] = src1[0 + 3 * src_step];
            dst7[1] = src0[0 + 3 * src_step];
            dst7[2] = src1[0 + 2 * src_step];
            dst7[3] = src0[0 + 2 * src_step];
            dst7[4] = src1[0 + src_step];
            dst7[5] = src0[0 + src_step];
            dst7[6] = src1[0];
            dst7[7] = src0[0];

            src0 += 1;
            src1 += 1;

            dst7 -= stride;
        }

        src0 += srcwgap + 7 * srcstride;
    }
#endif // __ARM_NEON
    for (; y < srch; y++)
    {
        unsigned char* dst0 = dstend - y - 1;

        int x = 0;
        for (; x < srcw; x++)
        {
            *dst0 = *src0;

            src0 += 1;
            dst0 -= stride;
        }

        src0 += srcwgap;
    }
}

static void kanna_rotate_7_c2(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    const int srcwgap = srcstride - srcw * 2;

    // point to the last dst pixel
    unsigned char* dstend = dst + stride * (h - 1) + w * 2;

    const unsigned char* src0 = src;

    int y = 0;
#if __ARM_NEON
    for (; y + 7 < srch; y += 8)
    {
        const unsigned char* src1 = src0 + srcstride;

        unsigned char* dst6 = dstend - y * 2 - 8 * 2 - stride;
        unsigned char* dst7 = dstend - y * 2 - 8 * 2;

        int src_step = 2 * srcstride;
        int dst_step = -2 * stride;

        int nn = srcw >> 3;
        int remain = srcw - (nn << 3);

#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x8x2_t _src0 = vld2_u8(src0);
            uint8x8x2_t _src1 = vld2_u8(src1);

            uint8x8x2_t _src2 = vld2_u8(src0 + src_step);
            uint8x8x2_t _src3 = vld2_u8(src1 + src_step);

            uint8x8x2_t _src4 = vld2_u8(src0 + 2 * src_step);
            uint8x8x2_t _src5 = vld2_u8(src1 + 2 * src_step);

            uint8x8x2_t _src6 = vld2_u8(src0 + 3 * src_step);
            uint8x8x2_t _src7 = vld2_u8(src1 + 3 * src_step);

            uint8x8x2_t _src01t_r = vtrn_u8(_src1.val[0], _src0.val[0]);
            uint8x8x2_t _src23t_r = vtrn_u8(_src3.val[0], _src2.val[0]);
            uint8x8x2_t _src45t_r = vtrn_u8(_src5.val[0], _src4.val[0]);
            uint8x8x2_t _src67t_r = vtrn_u8(_src7.val[0], _src6.val[0]);

            uint8x8x2_t _src01t_g = vtrn_u8(_src1.val[1], _src0.val[1]);
            uint8x8x2_t _src23t_g = vtrn_u8(_src3.val[1], _src2.val[1]);
            uint8x8x2_t _src45t_g = vtrn_u8(_src5.val[1], _src4.val[1]);
            uint8x8x2_t _src67t_g = vtrn_u8(_src7.val[1], _src6.val[1]);

            uint16x4x2_t _src02tt_r = vtrn_u16(vreinterpret_u16_u8(_src23t_r.val[1]), vreinterpret_u16_u8(_src01t_r.val[1]));
            uint16x4x2_t _src13tt_r = vtrn_u16(vreinterpret_u16_u8(_src23t_r.val[0]), vreinterpret_u16_u8(_src01t_r.val[0]));
            uint16x4x2_t _src46tt_r = vtrn_u16(vreinterpret_u16_u8(_src67t_r.val[1]), vreinterpret_u16_u8(_src45t_r.val[1]));
            uint16x4x2_t _src57tt_r = vtrn_u16(vreinterpret_u16_u8(_src67t_r.val[0]), vreinterpret_u16_u8(_src45t_r.val[0]));

            uint16x4x2_t _src02tt_g = vtrn_u16(vreinterpret_u16_u8(_src23t_g.val[1]), vreinterpret_u16_u8(_src01t_g.val[1]));
            uint16x4x2_t _src13tt_g = vtrn_u16(vreinterpret_u16_u8(_src23t_g.val[0]), vreinterpret_u16_u8(_src01t_g.val[0]));
            uint16x4x2_t _src46tt_g = vtrn_u16(vreinterpret_u16_u8(_src67t_g.val[1]), vreinterpret_u16_u8(_src45t_g.val[1]));
            uint16x4x2_t _src57tt_g = vtrn_u16(vreinterpret_u16_u8(_src67t_g.val[0]), vreinterpret_u16_u8(_src45t_g.val[0]));

            uint32x2x2_t _src04ttt_r = vtrn_u32(vreinterpret_u32_u16(_src46tt_r.val[1]), vreinterpret_u32_u16(_src02tt_r.val[1]));
            uint32x2x2_t _src15ttt_r = vtrn_u32(vreinterpret_u32_u16(_src57tt_r.val[1]), vreinterpret_u32_u16(_src13tt_r.val[1]));
            uint32x2x2_t _src26ttt_r = vtrn_u32(vreinterpret_u32_u16(_src46tt_r.val[0]), vreinterpret_u32_u16(_src02tt_r.val[0]));
            uint32x2x2_t _src37ttt_r = vtrn_u32(vreinterpret_u32_u16(_src57tt_r.val[0]), vreinterpret_u32_u16(_src13tt_r.val[0]));

            uint32x2x2_t _src04ttt_g = vtrn_u32(vreinterpret_u32_u16(_src46tt_g.val[1]), vreinterpret_u32_u16(_src02tt_g.val[1]));
            uint32x2x2_t _src15ttt_g = vtrn_u32(vreinterpret_u32_u16(_src57tt_g.val[1]), vreinterpret_u32_u16(_src13tt_g.val[1]));
            uint32x2x2_t _src26ttt_g = vtrn_u32(vreinterpret_u32_u16(_src46tt_g.val[0]), vreinterpret_u32_u16(_src02tt_g.val[0]));
            uint32x2x2_t _src37ttt_g = vtrn_u32(vreinterpret_u32_u16(_src57tt_g.val[0]), vreinterpret_u32_u16(_src13tt_g.val[0]));

            uint8x8x2_t _dst0;
            uint8x8x2_t _dst1;
            uint8x8x2_t _dst2;
            uint8x8x2_t _dst3;
            uint8x8x2_t _dst4;
            uint8x8x2_t _dst5;
            uint8x8x2_t _dst6;
            uint8x8x2_t _dst7;

            _dst0.val[0] = vreinterpret_u8_u32(_src04ttt_r.val[1]);
            _dst1.val[0] = vreinterpret_u8_u32(_src15ttt_r.val[1]);
            _dst2.val[0] = vreinterpret_u8_u32(_src26ttt_r.val[1]);
            _dst3.val[0] = vreinterpret_u8_u32(_src37ttt_r.val[1]);
            _dst4.val[0] = vreinterpret_u8_u32(_src04ttt_r.val[0]);
            _dst5.val[0] = vreinterpret_u8_u32(_src15ttt_r.val[0]);
            _dst6.val[0] = vreinterpret_u8_u32(_src26ttt_r.val[0]);
            _dst7.val[0] = vreinterpret_u8_u32(_src37ttt_r.val[0]);

            _dst0.val[1] = vreinterpret_u8_u32(_src04ttt_g.val[1]);
            _dst1.val[1] = vreinterpret_u8_u32(_src15ttt_g.val[1]);
            _dst2.val[1] = vreinterpret_u8_u32(_src26ttt_g.val[1]);
            _dst3.val[1] = vreinterpret_u8_u32(_src37ttt_g.val[1]);
            _dst4.val[1] = vreinterpret_u8_u32(_src04ttt_g.val[0]);
            _dst5.val[1] = vreinterpret_u8_u32(_src15ttt_g.val[0]);
            _dst6.val[1] = vreinterpret_u8_u32(_src26ttt_g.val[0]);
            _dst7.val[1] = vreinterpret_u8_u32(_src37ttt_g.val[0]);

            vst2_u8(dst7, _dst7);
            vst2_u8(dst6, _dst6);
            vst2_u8(dst7 + dst_step, _dst5);
            vst2_u8(dst6 + dst_step, _dst4);
            vst2_u8(dst7 + 2 * dst_step, _dst3);
            vst2_u8(dst6 + 2 * dst_step, _dst2);
            vst2_u8(dst7 + 3 * dst_step, _dst1);
            vst2_u8(dst6 + 3 * dst_step, _dst0);

            src0 += 2 * 8;
            src1 += 2 * 8;

            dst7 += 4 * dst_step;
            dst6 += 4 * dst_step;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #128]          \n"
                "vld2.u8    {d0-d1}, [%1], %10  \n"

                "pld        [%2, #128]          \n"
                "vld2.u8    {d2-d3}, [%2], %10  \n"

                "pld        [%1, #128]          \n"
                "vld2.u8    {d4-d5}, [%1], %10  \n"

                "vtrn.u8    q1, q0              \n" // _src01t_r

                "pld        [%2, #128]          \n"
                "vld2.u8    {d6-d7}, [%2], %10  \n"

                "pld        [%1, #128]          \n"
                "vld2.u8    {d16-d17}, [%1], %10\n"

                "vtrn.u8    q3, q2              \n" // _src23t_r

                "pld        [%2, #128]          \n"
                "vld2.u8    {d18-d19}, [%2], %10\n"

                "pld        [%1, #128]          \n"
                "vld2.u8    {d20-d21}, [%1], %10\n"

                "vtrn.u8    q9, q8              \n" // _src45t_r

                "pld        [%2, #128]          \n"
                "vld2.u8    {d22-d23}, [%2], %10\n"

                "vtrn.u8    q11, q10            \n" // _src67t_r

                "sub        %1, %1, %10, lsl #2 \n" // restore src0

                "vtrn.u16   q2, q0              \n" // _src02tt_r

                "sub        %2, %2, %10, lsl #2 \n" // restore src1

                "vtrn.u16   q3, q1              \n" // _src13tt_r

                "add        %1, #16             \n" // src0 += 16

                "vtrn.u16   q10, q8            \n" // _src46tt_r

                "add        %2, #16             \n" // src1 += 16

                "vtrn.u16   q11, q9             \n" // _src57tt_r

                "vtrn.u32   q10, q2             \n" // _src26ttt_r

                "vtrn.u32   q11, q3             \n" // _src37ttt_r
                "vst2.u8    {d20-d21}, [%4], %11\n"

                "vtrn.u32   q8, q0              \n" // _src04ttt_r
                "vst2.u8    {d22-d23}, [%3], %11\n"

                "vtrn.u32   q9, q1              \n" // _src15ttt_r
                "vst2.u8    {d16-d17}, [%4], %11\n"

                "subs       %0, #1              \n"

                "vst2.u8    {d4-d5}, [%4], %11  \n"
                "vst2.u8    {d18-d19}, [%3], %11\n"
                "vst2.u8    {d6-d7}, [%3], %11  \n"
                "vst2.u8    {d0-d1}, [%4], %11  \n"
                "vst2.u8    {d2-d3}, [%3], %11  \n"

                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(src1), // %2
                "=r"(dst7), // %3
                "=r"(dst6)  // %4
                : "0"(nn),
                "1"(src0),
                "2"(src1),
                "3"(dst7),
                "4"(dst6),
                "r"(src_step), // %10
                "r"(dst_step)  // %11
                : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11");
        }
#endif // __aarch64__
        for (; remain > 0; remain--)
        {
            dst7[0] = src1[0 + 3 * src_step];
            dst7[1] = src1[1 + 3 * src_step];
            dst7[2] = src0[0 + 3 * src_step];
            dst7[3] = src0[1 + 3 * src_step];
            dst7[4] = src1[0 + 2 * src_step];
            dst7[5] = src1[1 + 2 * src_step];
            dst7[6] = src0[0 + 2 * src_step];
            dst7[7] = src0[1 + 2 * src_step];
            dst7[8] = src1[0 + src_step];
            dst7[9] = src1[1 + src_step];
            dst7[10] = src0[0 + src_step];
            dst7[11] = src0[1 + src_step];
            dst7[12] = src1[0];
            dst7[13] = src1[1];
            dst7[14] = src0[0];
            dst7[15] = src0[1];

            src0 += 2;
            src1 += 2;

            dst7 -= stride;
        }

        src0 += srcwgap + 7 * srcstride;
    }
#endif // __ARM_NEON
    for (; y < srch; y++)
    {
        unsigned char* dst0 = dstend - y * 2 - 2;

        int x = 0;
        for (; x < srcw; x++)
        {
            dst0[0] = src0[0];
            dst0[1] = src0[1];

            src0 += 2;
            dst0 -= stride;
        }

        src0 += srcwgap;
    }
}

static void kanna_rotate_7_c3(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    const int srcwgap = srcstride - srcw * 3;

    // point to the last dst pixel
    unsigned char* dstend = dst + stride * (h - 1) + w * 3;

    const unsigned char* src0 = src;

    int y = 0;
#if __ARM_NEON
    for (; y + 7 < srch; y += 8)
    {
        const unsigned char* src1 = src0 + srcstride;

        unsigned char* dst6 = dstend - y * 3 - 8 * 3 - stride;
        unsigned char* dst7 = dstend - y * 3 - 8 * 3;

        int src_step = 2 * srcstride;
        int dst_step = -2 * stride;

        int nn = srcw >> 3;
        int remain = srcw - (nn << 3);

#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x8x3_t _src0 = vld3_u8(src0);
            uint8x8x3_t _src1 = vld3_u8(src1);

            uint8x8x3_t _src2 = vld3_u8(src0 + src_step);
            uint8x8x3_t _src3 = vld3_u8(src1 + src_step);

            uint8x8x3_t _src4 = vld3_u8(src0 + 2 * src_step);
            uint8x8x3_t _src5 = vld3_u8(src1 + 2 * src_step);

            uint8x8x3_t _src6 = vld3_u8(src0 + 3 * src_step);
            uint8x8x3_t _src7 = vld3_u8(src1 + 3 * src_step);

            uint8x8x2_t _src01t_r = vtrn_u8(_src1.val[0], _src0.val[0]);
            uint8x8x2_t _src23t_r = vtrn_u8(_src3.val[0], _src2.val[0]);
            uint8x8x2_t _src45t_r = vtrn_u8(_src5.val[0], _src4.val[0]);
            uint8x8x2_t _src67t_r = vtrn_u8(_src7.val[0], _src6.val[0]);

            uint8x8x2_t _src01t_g = vtrn_u8(_src1.val[1], _src0.val[1]);
            uint8x8x2_t _src23t_g = vtrn_u8(_src3.val[1], _src2.val[1]);
            uint8x8x2_t _src45t_g = vtrn_u8(_src5.val[1], _src4.val[1]);
            uint8x8x2_t _src67t_g = vtrn_u8(_src7.val[1], _src6.val[1]);

            uint8x8x2_t _src01t_b = vtrn_u8(_src1.val[2], _src0.val[2]);
            uint8x8x2_t _src23t_b = vtrn_u8(_src3.val[2], _src2.val[2]);
            uint8x8x2_t _src45t_b = vtrn_u8(_src5.val[2], _src4.val[2]);
            uint8x8x2_t _src67t_b = vtrn_u8(_src7.val[2], _src6.val[2]);

            uint16x4x2_t _src02tt_r = vtrn_u16(vreinterpret_u16_u8(_src23t_r.val[1]), vreinterpret_u16_u8(_src01t_r.val[1]));
            uint16x4x2_t _src13tt_r = vtrn_u16(vreinterpret_u16_u8(_src23t_r.val[0]), vreinterpret_u16_u8(_src01t_r.val[0]));
            uint16x4x2_t _src46tt_r = vtrn_u16(vreinterpret_u16_u8(_src67t_r.val[1]), vreinterpret_u16_u8(_src45t_r.val[1]));
            uint16x4x2_t _src57tt_r = vtrn_u16(vreinterpret_u16_u8(_src67t_r.val[0]), vreinterpret_u16_u8(_src45t_r.val[0]));

            uint16x4x2_t _src02tt_g = vtrn_u16(vreinterpret_u16_u8(_src23t_g.val[1]), vreinterpret_u16_u8(_src01t_g.val[1]));
            uint16x4x2_t _src13tt_g = vtrn_u16(vreinterpret_u16_u8(_src23t_g.val[0]), vreinterpret_u16_u8(_src01t_g.val[0]));
            uint16x4x2_t _src46tt_g = vtrn_u16(vreinterpret_u16_u8(_src67t_g.val[1]), vreinterpret_u16_u8(_src45t_g.val[1]));
            uint16x4x2_t _src57tt_g = vtrn_u16(vreinterpret_u16_u8(_src67t_g.val[0]), vreinterpret_u16_u8(_src45t_g.val[0]));

            uint16x4x2_t _src02tt_b = vtrn_u16(vreinterpret_u16_u8(_src23t_b.val[1]), vreinterpret_u16_u8(_src01t_b.val[1]));
            uint16x4x2_t _src13tt_b = vtrn_u16(vreinterpret_u16_u8(_src23t_b.val[0]), vreinterpret_u16_u8(_src01t_b.val[0]));
            uint16x4x2_t _src46tt_b = vtrn_u16(vreinterpret_u16_u8(_src67t_b.val[1]), vreinterpret_u16_u8(_src45t_b.val[1]));
            uint16x4x2_t _src57tt_b = vtrn_u16(vreinterpret_u16_u8(_src67t_b.val[0]), vreinterpret_u16_u8(_src45t_b.val[0]));

            uint32x2x2_t _src04ttt_r = vtrn_u32(vreinterpret_u32_u16(_src46tt_r.val[1]), vreinterpret_u32_u16(_src02tt_r.val[1]));
            uint32x2x2_t _src15ttt_r = vtrn_u32(vreinterpret_u32_u16(_src57tt_r.val[1]), vreinterpret_u32_u16(_src13tt_r.val[1]));
            uint32x2x2_t _src26ttt_r = vtrn_u32(vreinterpret_u32_u16(_src46tt_r.val[0]), vreinterpret_u32_u16(_src02tt_r.val[0]));
            uint32x2x2_t _src37ttt_r = vtrn_u32(vreinterpret_u32_u16(_src57tt_r.val[0]), vreinterpret_u32_u16(_src13tt_r.val[0]));

            uint32x2x2_t _src04ttt_g = vtrn_u32(vreinterpret_u32_u16(_src46tt_g.val[1]), vreinterpret_u32_u16(_src02tt_g.val[1]));
            uint32x2x2_t _src15ttt_g = vtrn_u32(vreinterpret_u32_u16(_src57tt_g.val[1]), vreinterpret_u32_u16(_src13tt_g.val[1]));
            uint32x2x2_t _src26ttt_g = vtrn_u32(vreinterpret_u32_u16(_src46tt_g.val[0]), vreinterpret_u32_u16(_src02tt_g.val[0]));
            uint32x2x2_t _src37ttt_g = vtrn_u32(vreinterpret_u32_u16(_src57tt_g.val[0]), vreinterpret_u32_u16(_src13tt_g.val[0]));

            uint32x2x2_t _src04ttt_b = vtrn_u32(vreinterpret_u32_u16(_src46tt_b.val[1]), vreinterpret_u32_u16(_src02tt_b.val[1]));
            uint32x2x2_t _src15ttt_b = vtrn_u32(vreinterpret_u32_u16(_src57tt_b.val[1]), vreinterpret_u32_u16(_src13tt_b.val[1]));
            uint32x2x2_t _src26ttt_b = vtrn_u32(vreinterpret_u32_u16(_src46tt_b.val[0]), vreinterpret_u32_u16(_src02tt_b.val[0]));
            uint32x2x2_t _src37ttt_b = vtrn_u32(vreinterpret_u32_u16(_src57tt_b.val[0]), vreinterpret_u32_u16(_src13tt_b.val[0]));

            uint8x8x3_t _dst0;
            uint8x8x3_t _dst1;
            uint8x8x3_t _dst2;
            uint8x8x3_t _dst3;
            uint8x8x3_t _dst4;
            uint8x8x3_t _dst5;
            uint8x8x3_t _dst6;
            uint8x8x3_t _dst7;

            _dst0.val[0] = vreinterpret_u8_u32(_src04ttt_r.val[1]);
            _dst1.val[0] = vreinterpret_u8_u32(_src15ttt_r.val[1]);
            _dst2.val[0] = vreinterpret_u8_u32(_src26ttt_r.val[1]);
            _dst3.val[0] = vreinterpret_u8_u32(_src37ttt_r.val[1]);
            _dst4.val[0] = vreinterpret_u8_u32(_src04ttt_r.val[0]);
            _dst5.val[0] = vreinterpret_u8_u32(_src15ttt_r.val[0]);
            _dst6.val[0] = vreinterpret_u8_u32(_src26ttt_r.val[0]);
            _dst7.val[0] = vreinterpret_u8_u32(_src37ttt_r.val[0]);

            _dst0.val[1] = vreinterpret_u8_u32(_src04ttt_g.val[1]);
            _dst1.val[1] = vreinterpret_u8_u32(_src15ttt_g.val[1]);
            _dst2.val[1] = vreinterpret_u8_u32(_src26ttt_g.val[1]);
            _dst3.val[1] = vreinterpret_u8_u32(_src37ttt_g.val[1]);
            _dst4.val[1] = vreinterpret_u8_u32(_src04ttt_g.val[0]);
            _dst5.val[1] = vreinterpret_u8_u32(_src15ttt_g.val[0]);
            _dst6.val[1] = vreinterpret_u8_u32(_src26ttt_g.val[0]);
            _dst7.val[1] = vreinterpret_u8_u32(_src37ttt_g.val[0]);

            _dst0.val[2] = vreinterpret_u8_u32(_src04ttt_b.val[1]);
            _dst1.val[2] = vreinterpret_u8_u32(_src15ttt_b.val[1]);
            _dst2.val[2] = vreinterpret_u8_u32(_src26ttt_b.val[1]);
            _dst3.val[2] = vreinterpret_u8_u32(_src37ttt_b.val[1]);
            _dst4.val[2] = vreinterpret_u8_u32(_src04ttt_b.val[0]);
            _dst5.val[2] = vreinterpret_u8_u32(_src15ttt_b.val[0]);
            _dst6.val[2] = vreinterpret_u8_u32(_src26ttt_b.val[0]);
            _dst7.val[2] = vreinterpret_u8_u32(_src37ttt_b.val[0]);

            vst3_u8(dst7, _dst7);
            vst3_u8(dst6, _dst6);
            vst3_u8(dst7 + dst_step, _dst5);
            vst3_u8(dst6 + dst_step, _dst4);
            vst3_u8(dst7 + 2 * dst_step, _dst3);
            vst3_u8(dst6 + 2 * dst_step, _dst2);
            vst3_u8(dst7 + 3 * dst_step, _dst1);
            vst3_u8(dst6 + 3 * dst_step, _dst0);

            src0 += 3 * 8;
            src1 += 3 * 8;

            dst7 += 4 * dst_step;
            dst6 += 4 * dst_step;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #192]          \n"
                "vld3.u8    {d0-d2}, [%1], %10  \n"

                "pld        [%2, #192]          \n"
                "vld3.u8    {d4-d6}, [%2], %10  \n"

                "pld        [%1, #192]          \n"
                "vld3.u8    {d8-d10}, [%1], %10 \n"

                "vtrn.u8    q2, q0              \n" // _src01t_r
                "vtrn.u8    d6, d2              \n"

                "pld        [%2, #192]          \n"
                "vld3.u8    {d12-d14}, [%2], %10\n"

                "pld        [%1, #192]          \n"
                "vld3.u8    {d16-d18}, [%1], %10\n"

                "vtrn.u8    q6, q4             \n" // _src23t_r
                "vtrn.u8    d14, d10            \n"

                "pld        [%2, #192]          \n"
                "vld3.u8    {d20-d22}, [%2], %10\n"

                "pld        [%1, #192]          \n"
                "vld3.u8    {d24-d26}, [%1], %10\n"

                "vtrn.u8    q10, q8             \n" // _src45t_r
                "vtrn.u8    d22, d18            \n"

                "pld        [%2, #192]          \n"
                "vld3.u8    {d28-d30}, [%2], %10\n"

                "vtrn.u8    q14, q12            \n" // _src67t_r
                "vtrn.u8    d30, d26            \n"

                "sub        %1, %1, %10, lsl #2 \n" // restore src0

                "vtrn.u16   q4, q0              \n" // _src02tt_r
                "vtrn.u16   d10, d2             \n"

                "sub        %2, %2, %10, lsl #2 \n" // restore src1

                "vtrn.u16   q6, q2              \n" // _src13tt_r
                "vtrn.u16   d14, d6             \n"

                "add        %1, #24             \n" // src0 += 24

                "vtrn.u16   q12, q8             \n" // _src46tt_r
                "vtrn.u16   d26, d18            \n"

                "add        %2, #24             \n" // src1 += 24

                "vtrn.u16   q14, q10            \n" // _src57tt_r
                "vtrn.u16   d30, d22            \n"

                "vtrn.u32   q12, q4             \n" // _src26ttt_r
                "vtrn.u32   d26, d10            \n"

                "vtrn.u32   q14, q6             \n" // _src37ttt_r
                "vst3.u8    {d24-d26}, [%4], %11\n"
                "vtrn.u32   d30, d14            \n"

                "vtrn.u32   q8, q0             \n" // _src04ttt_r
                "vst3.u8    {d28-d30}, [%3], %11\n"
                "vtrn.u32   d18, d2             \n"

                "vtrn.u32   q10, q2             \n" // _src15ttt_r
                "vst3.u8    {d16-d18}, [%4], %11\n"
                "vtrn.u32   d22, d6             \n"

                "subs       %0, #1              \n"

                "vst3.u8    {d8-d10}, [%4], %11 \n"
                "vst3.u8    {d20-d22}, [%3], %11\n"
                "vst3.u8    {d12-d14}, [%3], %11\n"
                "vst3.u8    {d0-d2}, [%4], %11  \n"
                "vst3.u8    {d4-d6}, [%3], %11  \n"

                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(src1), // %2
                "=r"(dst7), // %3
                "=r"(dst6)  // %4
                : "0"(nn),
                "1"(src0),
                "2"(src1),
                "3"(dst7),
                "4"(dst6),
                "r"(src_step), // %10
                "r"(dst_step)  // %11
                : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
        }
#endif // __aarch64__
        for (; remain > 0; remain--)
        {
            dst7[0] = src1[0 + 3 * src_step];
            dst7[1] = src1[1 + 3 * src_step];
            dst7[2] = src1[2 + 3 * src_step];
            dst7[3] = src0[0 + 3 * src_step];
            dst7[4] = src0[1 + 3 * src_step];
            dst7[5] = src0[2 + 3 * src_step];
            dst7[6] = src1[0 + 2 * src_step];
            dst7[7] = src1[1 + 2 * src_step];
            dst7[8] = src1[2 + 2 * src_step];
            dst7[9] = src0[0 + 2 * src_step];
            dst7[10] = src0[1 + 2 * src_step];
            dst7[11] = src0[2 + 2 * src_step];
            dst7[12] = src1[0 + src_step];
            dst7[13] = src1[1 + src_step];
            dst7[14] = src1[2 + src_step];
            dst7[15] = src0[0 + src_step];
            dst7[16] = src0[1 + src_step];
            dst7[17] = src0[2 + src_step];
            dst7[18] = src1[0];
            dst7[19] = src1[1];
            dst7[20] = src1[2];
            dst7[21] = src0[0];
            dst7[22] = src0[1];
            dst7[23] = src0[2];

            src0 += 3;
            src1 += 3;

            dst7 -= stride;
        }

        src0 += srcwgap + 7 * srcstride;
    }
#endif // __ARM_NEON
    for (; y < srch; y++)
    {
        unsigned char* dst0 = dstend - y * 3 - 3;

        int x = 0;
        for (; x < srcw; x++)
        {
            dst0[0] = src0[0];
            dst0[1] = src0[1];
            dst0[2] = src0[2];

            src0 += 3;
            dst0 -= stride;
        }

        src0 += srcwgap;
    }
}

static void kanna_rotate_7_c4(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    const int srcwgap = srcstride - srcw * 4;

    // point to the last dst pixel
    unsigned char* dstend = dst + stride * (h - 1) + w * 4;

    const unsigned char* src0 = src;

    int y = 0;
#if __ARM_NEON
    for (; y + 7 < srch; y += 8)
    {
        const unsigned char* src1 = src0 + srcstride;

        unsigned char* dst6 = dstend - y * 4 - 8 * 4 - stride;
        unsigned char* dst7 = dstend - y * 4 - 8 * 4;

        int src_step = 2 * srcstride;
        int dst_step = -2 * stride;

        int nn = srcw >> 3;
        int remain = srcw - (nn << 3);

#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x8x4_t _src0 = vld4_u8(src0);
            uint8x8x4_t _src1 = vld4_u8(src1);

            uint8x8x4_t _src2 = vld4_u8(src0 + src_step);
            uint8x8x4_t _src3 = vld4_u8(src1 + src_step);

            uint8x8x4_t _src4 = vld4_u8(src0 + 2 * src_step);
            uint8x8x4_t _src5 = vld4_u8(src1 + 2 * src_step);

            uint8x8x4_t _src6 = vld4_u8(src0 + 3 * src_step);
            uint8x8x4_t _src7 = vld4_u8(src1 + 3 * src_step);

            uint8x8x2_t _src01t_r = vtrn_u8(_src1.val[0], _src0.val[0]);
            uint8x8x2_t _src23t_r = vtrn_u8(_src3.val[0], _src2.val[0]);
            uint8x8x2_t _src45t_r = vtrn_u8(_src5.val[0], _src4.val[0]);
            uint8x8x2_t _src67t_r = vtrn_u8(_src7.val[0], _src6.val[0]);

            uint8x8x2_t _src01t_g = vtrn_u8(_src1.val[1], _src0.val[1]);
            uint8x8x2_t _src23t_g = vtrn_u8(_src3.val[1], _src2.val[1]);
            uint8x8x2_t _src45t_g = vtrn_u8(_src5.val[1], _src4.val[1]);
            uint8x8x2_t _src67t_g = vtrn_u8(_src7.val[1], _src6.val[1]);

            uint8x8x2_t _src01t_b = vtrn_u8(_src1.val[2], _src0.val[2]);
            uint8x8x2_t _src23t_b = vtrn_u8(_src3.val[2], _src2.val[2]);
            uint8x8x2_t _src45t_b = vtrn_u8(_src5.val[2], _src4.val[2]);
            uint8x8x2_t _src67t_b = vtrn_u8(_src7.val[2], _src6.val[2]);

            uint8x8x2_t _src01t_a = vtrn_u8(_src1.val[3], _src0.val[3]);
            uint8x8x2_t _src23t_a = vtrn_u8(_src3.val[3], _src2.val[3]);
            uint8x8x2_t _src45t_a = vtrn_u8(_src5.val[3], _src4.val[3]);
            uint8x8x2_t _src67t_a = vtrn_u8(_src7.val[3], _src6.val[3]);

            uint16x4x2_t _src02tt_r = vtrn_u16(vreinterpret_u16_u8(_src23t_r.val[1]), vreinterpret_u16_u8(_src01t_r.val[1]));
            uint16x4x2_t _src13tt_r = vtrn_u16(vreinterpret_u16_u8(_src23t_r.val[0]), vreinterpret_u16_u8(_src01t_r.val[0]));
            uint16x4x2_t _src46tt_r = vtrn_u16(vreinterpret_u16_u8(_src67t_r.val[1]), vreinterpret_u16_u8(_src45t_r.val[1]));
            uint16x4x2_t _src57tt_r = vtrn_u16(vreinterpret_u16_u8(_src67t_r.val[0]), vreinterpret_u16_u8(_src45t_r.val[0]));

            uint16x4x2_t _src02tt_g = vtrn_u16(vreinterpret_u16_u8(_src23t_g.val[1]), vreinterpret_u16_u8(_src01t_g.val[1]));
            uint16x4x2_t _src13tt_g = vtrn_u16(vreinterpret_u16_u8(_src23t_g.val[0]), vreinterpret_u16_u8(_src01t_g.val[0]));
            uint16x4x2_t _src46tt_g = vtrn_u16(vreinterpret_u16_u8(_src67t_g.val[1]), vreinterpret_u16_u8(_src45t_g.val[1]));
            uint16x4x2_t _src57tt_g = vtrn_u16(vreinterpret_u16_u8(_src67t_g.val[0]), vreinterpret_u16_u8(_src45t_g.val[0]));

            uint16x4x2_t _src02tt_b = vtrn_u16(vreinterpret_u16_u8(_src23t_b.val[1]), vreinterpret_u16_u8(_src01t_b.val[1]));
            uint16x4x2_t _src13tt_b = vtrn_u16(vreinterpret_u16_u8(_src23t_b.val[0]), vreinterpret_u16_u8(_src01t_b.val[0]));
            uint16x4x2_t _src46tt_b = vtrn_u16(vreinterpret_u16_u8(_src67t_b.val[1]), vreinterpret_u16_u8(_src45t_b.val[1]));
            uint16x4x2_t _src57tt_b = vtrn_u16(vreinterpret_u16_u8(_src67t_b.val[0]), vreinterpret_u16_u8(_src45t_b.val[0]));

            uint16x4x2_t _src02tt_a = vtrn_u16(vreinterpret_u16_u8(_src23t_a.val[1]), vreinterpret_u16_u8(_src01t_a.val[1]));
            uint16x4x2_t _src13tt_a = vtrn_u16(vreinterpret_u16_u8(_src23t_a.val[0]), vreinterpret_u16_u8(_src01t_a.val[0]));
            uint16x4x2_t _src46tt_a = vtrn_u16(vreinterpret_u16_u8(_src67t_a.val[1]), vreinterpret_u16_u8(_src45t_a.val[1]));
            uint16x4x2_t _src57tt_a = vtrn_u16(vreinterpret_u16_u8(_src67t_a.val[0]), vreinterpret_u16_u8(_src45t_a.val[0]));

            uint32x2x2_t _src04ttt_r = vtrn_u32(vreinterpret_u32_u16(_src46tt_r.val[1]), vreinterpret_u32_u16(_src02tt_r.val[1]));
            uint32x2x2_t _src15ttt_r = vtrn_u32(vreinterpret_u32_u16(_src57tt_r.val[1]), vreinterpret_u32_u16(_src13tt_r.val[1]));
            uint32x2x2_t _src26ttt_r = vtrn_u32(vreinterpret_u32_u16(_src46tt_r.val[0]), vreinterpret_u32_u16(_src02tt_r.val[0]));
            uint32x2x2_t _src37ttt_r = vtrn_u32(vreinterpret_u32_u16(_src57tt_r.val[0]), vreinterpret_u32_u16(_src13tt_r.val[0]));

            uint32x2x2_t _src04ttt_g = vtrn_u32(vreinterpret_u32_u16(_src46tt_g.val[1]), vreinterpret_u32_u16(_src02tt_g.val[1]));
            uint32x2x2_t _src15ttt_g = vtrn_u32(vreinterpret_u32_u16(_src57tt_g.val[1]), vreinterpret_u32_u16(_src13tt_g.val[1]));
            uint32x2x2_t _src26ttt_g = vtrn_u32(vreinterpret_u32_u16(_src46tt_g.val[0]), vreinterpret_u32_u16(_src02tt_g.val[0]));
            uint32x2x2_t _src37ttt_g = vtrn_u32(vreinterpret_u32_u16(_src57tt_g.val[0]), vreinterpret_u32_u16(_src13tt_g.val[0]));

            uint32x2x2_t _src04ttt_b = vtrn_u32(vreinterpret_u32_u16(_src46tt_b.val[1]), vreinterpret_u32_u16(_src02tt_b.val[1]));
            uint32x2x2_t _src15ttt_b = vtrn_u32(vreinterpret_u32_u16(_src57tt_b.val[1]), vreinterpret_u32_u16(_src13tt_b.val[1]));
            uint32x2x2_t _src26ttt_b = vtrn_u32(vreinterpret_u32_u16(_src46tt_b.val[0]), vreinterpret_u32_u16(_src02tt_b.val[0]));
            uint32x2x2_t _src37ttt_b = vtrn_u32(vreinterpret_u32_u16(_src57tt_b.val[0]), vreinterpret_u32_u16(_src13tt_b.val[0]));

            uint32x2x2_t _src04ttt_a = vtrn_u32(vreinterpret_u32_u16(_src46tt_a.val[1]), vreinterpret_u32_u16(_src02tt_a.val[1]));
            uint32x2x2_t _src15ttt_a = vtrn_u32(vreinterpret_u32_u16(_src57tt_a.val[1]), vreinterpret_u32_u16(_src13tt_a.val[1]));
            uint32x2x2_t _src26ttt_a = vtrn_u32(vreinterpret_u32_u16(_src46tt_a.val[0]), vreinterpret_u32_u16(_src02tt_a.val[0]));
            uint32x2x2_t _src37ttt_a = vtrn_u32(vreinterpret_u32_u16(_src57tt_a.val[0]), vreinterpret_u32_u16(_src13tt_a.val[0]));

            uint8x8x4_t _dst0;
            uint8x8x4_t _dst1;
            uint8x8x4_t _dst2;
            uint8x8x4_t _dst3;
            uint8x8x4_t _dst4;
            uint8x8x4_t _dst5;
            uint8x8x4_t _dst6;
            uint8x8x4_t _dst7;

            _dst0.val[0] = vreinterpret_u8_u32(_src04ttt_r.val[1]);
            _dst1.val[0] = vreinterpret_u8_u32(_src15ttt_r.val[1]);
            _dst2.val[0] = vreinterpret_u8_u32(_src26ttt_r.val[1]);
            _dst3.val[0] = vreinterpret_u8_u32(_src37ttt_r.val[1]);
            _dst4.val[0] = vreinterpret_u8_u32(_src04ttt_r.val[0]);
            _dst5.val[0] = vreinterpret_u8_u32(_src15ttt_r.val[0]);
            _dst6.val[0] = vreinterpret_u8_u32(_src26ttt_r.val[0]);
            _dst7.val[0] = vreinterpret_u8_u32(_src37ttt_r.val[0]);

            _dst0.val[1] = vreinterpret_u8_u32(_src04ttt_g.val[1]);
            _dst1.val[1] = vreinterpret_u8_u32(_src15ttt_g.val[1]);
            _dst2.val[1] = vreinterpret_u8_u32(_src26ttt_g.val[1]);
            _dst3.val[1] = vreinterpret_u8_u32(_src37ttt_g.val[1]);
            _dst4.val[1] = vreinterpret_u8_u32(_src04ttt_g.val[0]);
            _dst5.val[1] = vreinterpret_u8_u32(_src15ttt_g.val[0]);
            _dst6.val[1] = vreinterpret_u8_u32(_src26ttt_g.val[0]);
            _dst7.val[1] = vreinterpret_u8_u32(_src37ttt_g.val[0]);

            _dst0.val[2] = vreinterpret_u8_u32(_src04ttt_b.val[1]);
            _dst1.val[2] = vreinterpret_u8_u32(_src15ttt_b.val[1]);
            _dst2.val[2] = vreinterpret_u8_u32(_src26ttt_b.val[1]);
            _dst3.val[2] = vreinterpret_u8_u32(_src37ttt_b.val[1]);
            _dst4.val[2] = vreinterpret_u8_u32(_src04ttt_b.val[0]);
            _dst5.val[2] = vreinterpret_u8_u32(_src15ttt_b.val[0]);
            _dst6.val[2] = vreinterpret_u8_u32(_src26ttt_b.val[0]);
            _dst7.val[2] = vreinterpret_u8_u32(_src37ttt_b.val[0]);

            _dst0.val[3] = vreinterpret_u8_u32(_src04ttt_a.val[1]);
            _dst1.val[3] = vreinterpret_u8_u32(_src15ttt_a.val[1]);
            _dst2.val[3] = vreinterpret_u8_u32(_src26ttt_a.val[1]);
            _dst3.val[3] = vreinterpret_u8_u32(_src37ttt_a.val[1]);
            _dst4.val[3] = vreinterpret_u8_u32(_src04ttt_a.val[0]);
            _dst5.val[3] = vreinterpret_u8_u32(_src15ttt_a.val[0]);
            _dst6.val[3] = vreinterpret_u8_u32(_src26ttt_a.val[0]);
            _dst7.val[3] = vreinterpret_u8_u32(_src37ttt_a.val[0]);

            vst4_u8(dst7, _dst7);
            vst4_u8(dst6, _dst6);
            vst4_u8(dst7 + dst_step, _dst5);
            vst4_u8(dst6 + dst_step, _dst4);
            vst4_u8(dst7 + 2 * dst_step, _dst3);
            vst4_u8(dst6 + 2 * dst_step, _dst2);
            vst4_u8(dst7 + 3 * dst_step, _dst1);
            vst4_u8(dst6 + 3 * dst_step, _dst0);

            src0 += 4 * 8;
            src1 += 4 * 8;

            dst7 += 4 * dst_step;
            dst6 += 4 * dst_step;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "vld4.u8    {d0-d3}, [%1], %10  \n"

                "pld        [%2, #256]          \n"
                "vld4.u8    {d4-d7}, [%2], %10  \n"

                "pld        [%1, #256]          \n"
                "vld4.u8    {d8-d11}, [%1], %10 \n"

                "vtrn.u8    q2, q0              \n" // _src01t_r
                "vtrn.u8    q3, q1              \n"

                "pld        [%2, #256]          \n"
                "vld4.u8    {d12-d15}, [%2], %10\n"

                "pld        [%1, #256]          \n"
                "vld4.u8    {d16-d19}, [%1], %10\n"

                "vtrn.u8    q6, q4              \n" // _src23t_r
                "vtrn.u8    q7, q5              \n"

                "pld        [%2, #256]          \n"
                "vld4.u8    {d20-d23}, [%2], %10\n"

                "pld        [%1, #256]          \n"
                "vld4.u8    {d24-d27}, [%1], %10\n"

                "vtrn.u8    q10, q8             \n" // _src45t_r
                "vtrn.u8    q11, q9             \n"

                "pld        [%2, #256]          \n"
                "vld4.u8    {d28-d31}, [%2], %10\n"

                "vtrn.u8    q14, q12            \n" // _src67t_r
                "vtrn.u8    q15, q13            \n"

                "sub        %1, %1, %10, lsl #2 \n" // restore src0

                "vtrn.u16   q4, q0              \n" // _src02tt_r
                "vtrn.u16   q5, q1              \n"

                "sub        %2, %2, %10, lsl #2 \n" // restore src1

                "vtrn.u16   q6, q2              \n" // _src13tt_r
                "vtrn.u16   q7, q3              \n"

                "add        %1, #32             \n" // src0 += 32

                "vtrn.u16   q12, q8             \n" // _src46tt_r
                "vtrn.u16   q13, q9             \n"

                "add        %2, #32             \n" // src1 += 32

                "vtrn.u16   q14, q10            \n" // _src57tt_r
                "vtrn.u16   q15, q11            \n"

                "vtrn.u32   q12, q4             \n" // _src26ttt_r
                "vtrn.u32   q13, q5             \n"

                "vtrn.u32   q14, q6             \n" // _src37ttt_r
                "vst4.u8    {d24-d27}, [%4], %11\n"
                "vtrn.u32   q15, q7             \n"

                "vtrn.u32   q8, q0              \n" // _src04ttt_r
                "vst4.u8    {d28-d31}, [%3], %11\n"
                "vtrn.u32   q9, q1              \n"

                "vtrn.u32   q10, q2             \n" // _src15ttt_r
                "vst4.u8    {d16-d19}, [%4], %11\n"
                "vtrn.u32   q11, q3             \n"

                "subs       %0, #1              \n"

                "vst4.u8    {d8-d11}, [%4], %11 \n"
                "vst4.u8    {d20-d23}, [%3], %11\n"
                "vst4.u8    {d12-d15}, [%3], %11\n"
                "vst4.u8    {d0-d3}, [%4], %11  \n"
                "vst4.u8    {d4-d7}, [%3], %11  \n"

                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(src1), // %2
                "=r"(dst7), // %3
                "=r"(dst6)  // %4
                : "0"(nn),
                "1"(src0),
                "2"(src1),
                "3"(dst7),
                "4"(dst6),
                "r"(src_step), // %10
                "r"(dst_step)  // %11
                : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
        }
#endif // __aarch64__
        for (; remain > 0; remain--)
        {
            dst7[0] = src1[0 + 3 * src_step];
            dst7[1] = src1[1 + 3 * src_step];
            dst7[2] = src1[2 + 3 * src_step];
            dst7[3] = src1[3 + 3 * src_step];
            dst7[4] = src0[0 + 3 * src_step];
            dst7[5] = src0[1 + 3 * src_step];
            dst7[6] = src0[2 + 3 * src_step];
            dst7[7] = src0[3 + 3 * src_step];
            dst7[8] = src1[0 + 2 * src_step];
            dst7[9] = src1[1 + 2 * src_step];
            dst7[10] = src1[2 + 2 * src_step];
            dst7[11] = src1[3 + 2 * src_step];
            dst7[12] = src0[0 + 2 * src_step];
            dst7[13] = src0[1 + 2 * src_step];
            dst7[14] = src0[2 + 2 * src_step];
            dst7[15] = src0[3 + 2 * src_step];
            dst7[16] = src1[0 + src_step];
            dst7[17] = src1[1 + src_step];
            dst7[18] = src1[2 + src_step];
            dst7[19] = src1[3 + src_step];
            dst7[20] = src0[0 + src_step];
            dst7[21] = src0[1 + src_step];
            dst7[22] = src0[2 + src_step];
            dst7[23] = src0[3 + src_step];
            dst7[24] = src1[0];
            dst7[25] = src1[1];
            dst7[26] = src1[2];
            dst7[27] = src1[3];
            dst7[28] = src0[0];
            dst7[29] = src0[1];
            dst7[30] = src0[2];
            dst7[31] = src0[3];

            src0 += 4;
            src1 += 4;

            dst7 -= stride;
        }

        src0 += srcwgap + 7 * srcstride;
    }
#endif // __ARM_NEON
    for (; y < srch; y++)
    {
        unsigned char* dst0 = dstend - y * 4 - 4;

        int x = 0;
        for (; x < srcw; x++)
        {
            dst0[0] = src0[0];
            dst0[1] = src0[1];
            dst0[2] = src0[2];
            dst0[3] = src0[3];

            src0 += 4;
            dst0 -= stride;
        }

        src0 += srcwgap;
    }
}

static void kanna_rotate_8_c1(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int /*w*/, int h, int stride)
{
    const int srcwgap = srcstride - srcw;

    // point to the last dst pixel row
    unsigned char* dstend = dst + stride * (h - 1);

    const unsigned char* src0 = src;

    int y = 0;
#if __ARM_NEON
    for (; y + 7 < srch; y += 8)
    {
        const unsigned char* src1 = src0 + srcstride;

        unsigned char* dst7 = dstend + y;
        unsigned char* dst6 = dstend + y - stride;

        int src_step = 2 * srcstride;
        int dst_step = -2 * stride;

        int nn = srcw >> 3;
        int remain = srcw - (nn << 3);

#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x8_t _src0 = vld1_u8(src0);
            uint8x8_t _src1 = vld1_u8(src1);

            uint8x8_t _src2 = vld1_u8(src0 + src_step);
            uint8x8_t _src3 = vld1_u8(src1 + src_step);

            uint8x8_t _src4 = vld1_u8(src0 + 2 * src_step);
            uint8x8_t _src5 = vld1_u8(src1 + 2 * src_step);

            uint8x8_t _src6 = vld1_u8(src0 + 3 * src_step);
            uint8x8_t _src7 = vld1_u8(src1 + 3 * src_step);

            uint8x8x2_t _src01t_r = vtrn_u8(_src0, _src1);
            uint8x8x2_t _src23t_r = vtrn_u8(_src2, _src3);
            uint8x8x2_t _src45t_r = vtrn_u8(_src4, _src5);
            uint8x8x2_t _src67t_r = vtrn_u8(_src6, _src7);

            uint16x4x2_t _src02tt_r = vtrn_u16(vreinterpret_u16_u8(_src01t_r.val[0]), vreinterpret_u16_u8(_src23t_r.val[0]));
            uint16x4x2_t _src13tt_r = vtrn_u16(vreinterpret_u16_u8(_src01t_r.val[1]), vreinterpret_u16_u8(_src23t_r.val[1]));
            uint16x4x2_t _src46tt_r = vtrn_u16(vreinterpret_u16_u8(_src45t_r.val[0]), vreinterpret_u16_u8(_src67t_r.val[0]));
            uint16x4x2_t _src57tt_r = vtrn_u16(vreinterpret_u16_u8(_src45t_r.val[1]), vreinterpret_u16_u8(_src67t_r.val[1]));

            uint32x2x2_t _src04ttt_r = vtrn_u32(vreinterpret_u32_u16(_src02tt_r.val[0]), vreinterpret_u32_u16(_src46tt_r.val[0]));
            uint32x2x2_t _src15ttt_r = vtrn_u32(vreinterpret_u32_u16(_src13tt_r.val[0]), vreinterpret_u32_u16(_src57tt_r.val[0]));
            uint32x2x2_t _src26ttt_r = vtrn_u32(vreinterpret_u32_u16(_src02tt_r.val[1]), vreinterpret_u32_u16(_src46tt_r.val[1]));
            uint32x2x2_t _src37ttt_r = vtrn_u32(vreinterpret_u32_u16(_src13tt_r.val[1]), vreinterpret_u32_u16(_src57tt_r.val[1]));

            uint8x8_t _dst0 = vreinterpret_u8_u32(_src04ttt_r.val[0]);
            uint8x8_t _dst1 = vreinterpret_u8_u32(_src15ttt_r.val[0]);
            uint8x8_t _dst2 = vreinterpret_u8_u32(_src26ttt_r.val[0]);
            uint8x8_t _dst3 = vreinterpret_u8_u32(_src37ttt_r.val[0]);
            uint8x8_t _dst4 = vreinterpret_u8_u32(_src04ttt_r.val[1]);
            uint8x8_t _dst5 = vreinterpret_u8_u32(_src15ttt_r.val[1]);
            uint8x8_t _dst6 = vreinterpret_u8_u32(_src26ttt_r.val[1]);
            uint8x8_t _dst7 = vreinterpret_u8_u32(_src37ttt_r.val[1]);

            vst1_u8(dst7, _dst0);
            vst1_u8(dst6, _dst1);
            vst1_u8(dst7 + dst_step, _dst2);
            vst1_u8(dst6 + dst_step, _dst3);
            vst1_u8(dst7 + 2 * dst_step, _dst4);
            vst1_u8(dst6 + 2 * dst_step, _dst5);
            vst1_u8(dst7 + 3 * dst_step, _dst6);
            vst1_u8(dst6 + 3 * dst_step, _dst7);

            src0 += 8;
            src1 += 8;

            dst7 += 4 * dst_step;
            dst6 += 4 * dst_step;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #64]           \n"
                "vld1.u8    {d0}, [%1], %10     \n"

                "pld        [%2, #64]           \n"
                "vld1.u8    {d1}, [%2], %10     \n"

                "pld        [%1, #64]           \n"
                "vld1.u8    {d2}, [%1], %10     \n"

                "vtrn.u8    d0, d1              \n" // _src01t_r

                "pld        [%2, #64]           \n"
                "vld1.u8    {d3}, [%2], %10     \n"

                "pld        [%1, #64]           \n"
                "vld1.u8    {d4}, [%1], %10     \n"

                "vtrn.u8    d2, d3              \n" // _src23t_r

                "pld        [%2, #64]           \n"
                "vld1.u8    {d5}, [%2], %10     \n"

                "pld        [%1, #64]           \n"
                "vld1.u8    {d6}, [%1], %10     \n"

                "vtrn.u8    d4, d5              \n" // _src45t_r

                "pld        [%2, #64]           \n"
                "vld1.u8    {d7}, [%2], %10     \n"

                "vtrn.u8    d6, d7              \n" // _src67t_r

                "sub        %1, %1, %10, lsl #2 \n" // restore src0

                "vtrn.u16   q0, q1              \n" // _src02tt_r _src13tt_r

                "sub        %2, %2, %10, lsl #2 \n" // restore src1

                "vtrn.u16   q2, q3              \n" // _src46tt_r _src57tt_r

                "add        %1, #8              \n" // src0 += 8

                "vtrn.u32   q0, q2              \n" // _src04ttt_r _src15ttt_r

                "add        %2, #8              \n" // src1 += 8

                "vtrn.u32   q1, q3              \n" // _src26ttt_r _src37ttt_r
                "vst1.u8    {d0}, [%3], %11     \n"
                "vst1.u8    {d1}, [%4], %11     \n"

                "subs       %0, #1              \n"

                "vst1.u8    {d2}, [%3], %11     \n"
                "vst1.u8    {d3}, [%4], %11     \n"
                "vst1.u8    {d4}, [%3], %11     \n"
                "vst1.u8    {d5}, [%4], %11     \n"
                "vst1.u8    {d6}, [%3], %11     \n"
                "vst1.u8    {d7}, [%4], %11     \n"

                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(src1), // %2
                "=r"(dst7), // %3
                "=r"(dst6)  // %4
                : "0"(nn),
                "1"(src0),
                "2"(src1),
                "3"(dst7),
                "4"(dst6),
                "r"(src_step), // %10
                "r"(dst_step)  // %11
                : "cc", "memory", "q0", "q1", "q2", "q3");
        }
#endif // __aarch64__
        for (; remain > 0; remain--)
        {
            dst7[0] = src0[0];
            dst7[1] = src1[0];
            dst7[2] = src0[0 + src_step];
            dst7[3] = src1[0 + src_step];
            dst7[4] = src0[0 + 2 * src_step];
            dst7[5] = src1[0 + 2 * src_step];
            dst7[6] = src0[0 + 3 * src_step];
            dst7[7] = src1[0 + 3 * src_step];

            src0 += 1;
            src1 += 1;

            dst7 -= stride;
        }

        src0 += srcwgap + 7 * srcstride;
    }
#endif // __ARM_NEON
    for (; y < srch; y++)
    {
        unsigned char* dst0 = dstend + y;

        int x = 0;
        for (; x < srcw; x++)
        {
            *dst0 = *src0;

            src0 += 1;
            dst0 -= stride;
        }

        src0 += srcwgap;
    }
}

static void kanna_rotate_8_c2(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int /*w*/, int h, int stride)
{
    const int srcwgap = srcstride - srcw * 2;

    // point to the last dst pixel row
    unsigned char* dstend = dst + stride * (h - 1);

    const unsigned char* src0 = src;

    int y = 0;
#if __ARM_NEON
    for (; y + 7 < srch; y += 8)
    {
        const unsigned char* src1 = src0 + srcstride;

        unsigned char* dst7 = dstend + y * 2;
        unsigned char* dst6 = dstend + y * 2 - stride;

        int src_step = 2 * srcstride;
        int dst_step = -2 * stride;

        int nn = srcw >> 3;
        int remain = srcw - (nn << 3);

#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x8x2_t _src0 = vld2_u8(src0);
            uint8x8x2_t _src1 = vld2_u8(src1);

            uint8x8x2_t _src2 = vld2_u8(src0 + src_step);
            uint8x8x2_t _src3 = vld2_u8(src1 + src_step);

            uint8x8x2_t _src4 = vld2_u8(src0 + 2 * src_step);
            uint8x8x2_t _src5 = vld2_u8(src1 + 2 * src_step);

            uint8x8x2_t _src6 = vld2_u8(src0 + 3 * src_step);
            uint8x8x2_t _src7 = vld2_u8(src1 + 3 * src_step);

            uint8x8x2_t _src01t_r = vtrn_u8(_src0.val[0], _src1.val[0]);
            uint8x8x2_t _src23t_r = vtrn_u8(_src2.val[0], _src3.val[0]);
            uint8x8x2_t _src45t_r = vtrn_u8(_src4.val[0], _src5.val[0]);
            uint8x8x2_t _src67t_r = vtrn_u8(_src6.val[0], _src7.val[0]);

            uint8x8x2_t _src01t_g = vtrn_u8(_src0.val[1], _src1.val[1]);
            uint8x8x2_t _src23t_g = vtrn_u8(_src2.val[1], _src3.val[1]);
            uint8x8x2_t _src45t_g = vtrn_u8(_src4.val[1], _src5.val[1]);
            uint8x8x2_t _src67t_g = vtrn_u8(_src6.val[1], _src7.val[1]);

            uint16x4x2_t _src02tt_r = vtrn_u16(vreinterpret_u16_u8(_src01t_r.val[0]), vreinterpret_u16_u8(_src23t_r.val[0]));
            uint16x4x2_t _src13tt_r = vtrn_u16(vreinterpret_u16_u8(_src01t_r.val[1]), vreinterpret_u16_u8(_src23t_r.val[1]));
            uint16x4x2_t _src46tt_r = vtrn_u16(vreinterpret_u16_u8(_src45t_r.val[0]), vreinterpret_u16_u8(_src67t_r.val[0]));
            uint16x4x2_t _src57tt_r = vtrn_u16(vreinterpret_u16_u8(_src45t_r.val[1]), vreinterpret_u16_u8(_src67t_r.val[1]));

            uint16x4x2_t _src02tt_g = vtrn_u16(vreinterpret_u16_u8(_src01t_g.val[0]), vreinterpret_u16_u8(_src23t_g.val[0]));
            uint16x4x2_t _src13tt_g = vtrn_u16(vreinterpret_u16_u8(_src01t_g.val[1]), vreinterpret_u16_u8(_src23t_g.val[1]));
            uint16x4x2_t _src46tt_g = vtrn_u16(vreinterpret_u16_u8(_src45t_g.val[0]), vreinterpret_u16_u8(_src67t_g.val[0]));
            uint16x4x2_t _src57tt_g = vtrn_u16(vreinterpret_u16_u8(_src45t_g.val[1]), vreinterpret_u16_u8(_src67t_g.val[1]));

            uint32x2x2_t _src04ttt_r = vtrn_u32(vreinterpret_u32_u16(_src02tt_r.val[0]), vreinterpret_u32_u16(_src46tt_r.val[0]));
            uint32x2x2_t _src15ttt_r = vtrn_u32(vreinterpret_u32_u16(_src13tt_r.val[0]), vreinterpret_u32_u16(_src57tt_r.val[0]));
            uint32x2x2_t _src26ttt_r = vtrn_u32(vreinterpret_u32_u16(_src02tt_r.val[1]), vreinterpret_u32_u16(_src46tt_r.val[1]));
            uint32x2x2_t _src37ttt_r = vtrn_u32(vreinterpret_u32_u16(_src13tt_r.val[1]), vreinterpret_u32_u16(_src57tt_r.val[1]));

            uint32x2x2_t _src04ttt_g = vtrn_u32(vreinterpret_u32_u16(_src02tt_g.val[0]), vreinterpret_u32_u16(_src46tt_g.val[0]));
            uint32x2x2_t _src15ttt_g = vtrn_u32(vreinterpret_u32_u16(_src13tt_g.val[0]), vreinterpret_u32_u16(_src57tt_g.val[0]));
            uint32x2x2_t _src26ttt_g = vtrn_u32(vreinterpret_u32_u16(_src02tt_g.val[1]), vreinterpret_u32_u16(_src46tt_g.val[1]));
            uint32x2x2_t _src37ttt_g = vtrn_u32(vreinterpret_u32_u16(_src13tt_g.val[1]), vreinterpret_u32_u16(_src57tt_g.val[1]));

            uint8x8x2_t _dst0;
            uint8x8x2_t _dst1;
            uint8x8x2_t _dst2;
            uint8x8x2_t _dst3;
            uint8x8x2_t _dst4;
            uint8x8x2_t _dst5;
            uint8x8x2_t _dst6;
            uint8x8x2_t _dst7;

            _dst0.val[0] = vreinterpret_u8_u32(_src04ttt_r.val[0]);
            _dst1.val[0] = vreinterpret_u8_u32(_src15ttt_r.val[0]);
            _dst2.val[0] = vreinterpret_u8_u32(_src26ttt_r.val[0]);
            _dst3.val[0] = vreinterpret_u8_u32(_src37ttt_r.val[0]);
            _dst4.val[0] = vreinterpret_u8_u32(_src04ttt_r.val[1]);
            _dst5.val[0] = vreinterpret_u8_u32(_src15ttt_r.val[1]);
            _dst6.val[0] = vreinterpret_u8_u32(_src26ttt_r.val[1]);
            _dst7.val[0] = vreinterpret_u8_u32(_src37ttt_r.val[1]);

            _dst0.val[1] = vreinterpret_u8_u32(_src04ttt_g.val[0]);
            _dst1.val[1] = vreinterpret_u8_u32(_src15ttt_g.val[0]);
            _dst2.val[1] = vreinterpret_u8_u32(_src26ttt_g.val[0]);
            _dst3.val[1] = vreinterpret_u8_u32(_src37ttt_g.val[0]);
            _dst4.val[1] = vreinterpret_u8_u32(_src04ttt_g.val[1]);
            _dst5.val[1] = vreinterpret_u8_u32(_src15ttt_g.val[1]);
            _dst6.val[1] = vreinterpret_u8_u32(_src26ttt_g.val[1]);
            _dst7.val[1] = vreinterpret_u8_u32(_src37ttt_g.val[1]);

            vst2_u8(dst7, _dst0);
            vst2_u8(dst6, _dst1);
            vst2_u8(dst7 + dst_step, _dst2);
            vst2_u8(dst6 + dst_step, _dst3);
            vst2_u8(dst7 + 2 * dst_step, _dst4);
            vst2_u8(dst6 + 2 * dst_step, _dst5);
            vst2_u8(dst7 + 3 * dst_step, _dst6);
            vst2_u8(dst6 + 3 * dst_step, _dst7);

            src0 += 2 * 8;
            src1 += 2 * 8;

            dst7 += 4 * dst_step;
            dst6 += 4 * dst_step;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #128]          \n"
                "vld2.u8    {d0-d1}, [%1], %10  \n"

                "pld        [%2, #128]          \n"
                "vld2.u8    {d2-d3}, [%2], %10  \n"

                "pld        [%1, #128]          \n"
                "vld2.u8    {d4-d5}, [%1], %10  \n"

                "vtrn.u8    q0, q1              \n" // _src01t_r

                "pld        [%2, #128]          \n"
                "vld2.u8    {d6-d7}, [%2], %10  \n"

                "pld        [%1, #128]          \n"
                "vld2.u8    {d16-d17}, [%1], %10\n"

                "vtrn.u8    q2, q3              \n" // _src23t_r

                "pld        [%2, #128]          \n"
                "vld2.u8    {d18-d19}, [%2], %10\n"

                "pld        [%1, #128]          \n"
                "vld2.u8    {d20-d21}, [%1], %10\n"

                "vtrn.u8    q8, q9              \n" // _src45t_r

                "pld        [%2, #128]          \n"
                "vld2.u8    {d22-d23}, [%2], %10\n"

                "vtrn.u8    q10, q11            \n" // _src67t_r

                "sub        %1, %1, %10, lsl #2 \n" // restore src0

                "vtrn.u16   q0, q2              \n" // _src02tt_r

                "sub        %2, %2, %10, lsl #2 \n" // restore src1

                "vtrn.u16   q1, q3              \n" // _src13tt_r

                "add        %1, #16             \n" // src0 += 16

                "vtrn.u16   q8, q10             \n" // _src46tt_r

                "add        %2, #16             \n" // src1 += 16

                "vtrn.u16   q9, q11             \n" // _src57tt_r

                "vtrn.u32   q0, q8              \n" // _src04ttt_r

                "vtrn.u32   q1, q9              \n" // _src15ttt_r
                "vst2.u8    {d0-d1}, [%3], %11  \n"

                "vtrn.u32   q2, q10             \n" // _src26ttt_r
                "vst2.u8    {d2-d3}, [%4], %11  \n"

                "vtrn.u32   q3, q11             \n" // _src37ttt_r
                "vst2.u8    {d4-d5}, [%3], %11  \n"

                "subs       %0, #1              \n"

                "vst2.u8    {d16-d17}, [%3], %11\n"
                "vst2.u8    {d6-d7}, [%4], %11  \n"
                "vst2.u8    {d18-d19}, [%4], %11\n"
                "vst2.u8    {d20-d21}, [%3], %11\n"
                "vst2.u8    {d22-d23}, [%4], %11\n"

                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(src1), // %2
                "=r"(dst7), // %3
                "=r"(dst6)  // %4
                : "0"(nn),
                "1"(src0),
                "2"(src1),
                "3"(dst7),
                "4"(dst6),
                "r"(src_step), // %10
                "r"(dst_step)  // %11
                : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11");
        }
#endif // __aarch64__
        for (; remain > 0; remain--)
        {
            dst7[0] = src0[0];
            dst7[1] = src0[1];
            dst7[2] = src1[0];
            dst7[3] = src1[1];
            dst7[4] = src0[0 + src_step];
            dst7[5] = src0[1 + src_step];
            dst7[6] = src1[0 + src_step];
            dst7[7] = src1[1 + src_step];
            dst7[8] = src0[0 + 2 * src_step];
            dst7[9] = src0[1 + 2 * src_step];
            dst7[10] = src1[0 + 2 * src_step];
            dst7[11] = src1[1 + 2 * src_step];
            dst7[12] = src0[0 + 3 * src_step];
            dst7[13] = src0[1 + 3 * src_step];
            dst7[14] = src1[0 + 3 * src_step];
            dst7[15] = src1[1 + 3 * src_step];

            src0 += 2;
            src1 += 2;

            dst7 -= stride;
        }

        src0 += srcwgap + 7 * srcstride;
    }
#endif // __ARM_NEON
    for (; y < srch; y++)
    {
        unsigned char* dst0 = dstend + y * 2;

        int x = 0;
        for (; x < srcw; x++)
        {
            dst0[0] = src0[0];
            dst0[1] = src0[1];

            src0 += 2;
            dst0 -= stride;
        }

        src0 += srcwgap;
    }
}

static void kanna_rotate_8_c3(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int /*w*/, int h, int stride)
{
    const int srcwgap = srcstride - srcw * 3;

    // point to the last dst pixel row
    unsigned char* dstend = dst + stride * (h - 1);

    const unsigned char* src0 = src;

    int y = 0;
#if __ARM_NEON
    for (; y + 7 < srch; y += 8)
    {
        const unsigned char* src1 = src0 + srcstride;

        unsigned char* dst7 = dstend + y * 3;
        unsigned char* dst6 = dstend + y * 3 - stride;

        int src_step = 2 * srcstride;
        int dst_step = -2 * stride;

        int nn = srcw >> 3;
        int remain = srcw - (nn << 3);

#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x8x3_t _src0 = vld3_u8(src0);
            uint8x8x3_t _src1 = vld3_u8(src1);

            uint8x8x3_t _src2 = vld3_u8(src0 + src_step);
            uint8x8x3_t _src3 = vld3_u8(src1 + src_step);

            uint8x8x3_t _src4 = vld3_u8(src0 + 2 * src_step);
            uint8x8x3_t _src5 = vld3_u8(src1 + 2 * src_step);

            uint8x8x3_t _src6 = vld3_u8(src0 + 3 * src_step);
            uint8x8x3_t _src7 = vld3_u8(src1 + 3 * src_step);

            uint8x8x2_t _src01t_r = vtrn_u8(_src0.val[0], _src1.val[0]);
            uint8x8x2_t _src23t_r = vtrn_u8(_src2.val[0], _src3.val[0]);
            uint8x8x2_t _src45t_r = vtrn_u8(_src4.val[0], _src5.val[0]);
            uint8x8x2_t _src67t_r = vtrn_u8(_src6.val[0], _src7.val[0]);

            uint8x8x2_t _src01t_g = vtrn_u8(_src0.val[1], _src1.val[1]);
            uint8x8x2_t _src23t_g = vtrn_u8(_src2.val[1], _src3.val[1]);
            uint8x8x2_t _src45t_g = vtrn_u8(_src4.val[1], _src5.val[1]);
            uint8x8x2_t _src67t_g = vtrn_u8(_src6.val[1], _src7.val[1]);

            uint8x8x2_t _src01t_b = vtrn_u8(_src0.val[2], _src1.val[2]);
            uint8x8x2_t _src23t_b = vtrn_u8(_src2.val[2], _src3.val[2]);
            uint8x8x2_t _src45t_b = vtrn_u8(_src4.val[2], _src5.val[2]);
            uint8x8x2_t _src67t_b = vtrn_u8(_src6.val[2], _src7.val[2]);

            uint16x4x2_t _src02tt_r = vtrn_u16(vreinterpret_u16_u8(_src01t_r.val[0]), vreinterpret_u16_u8(_src23t_r.val[0]));
            uint16x4x2_t _src13tt_r = vtrn_u16(vreinterpret_u16_u8(_src01t_r.val[1]), vreinterpret_u16_u8(_src23t_r.val[1]));
            uint16x4x2_t _src46tt_r = vtrn_u16(vreinterpret_u16_u8(_src45t_r.val[0]), vreinterpret_u16_u8(_src67t_r.val[0]));
            uint16x4x2_t _src57tt_r = vtrn_u16(vreinterpret_u16_u8(_src45t_r.val[1]), vreinterpret_u16_u8(_src67t_r.val[1]));

            uint16x4x2_t _src02tt_g = vtrn_u16(vreinterpret_u16_u8(_src01t_g.val[0]), vreinterpret_u16_u8(_src23t_g.val[0]));
            uint16x4x2_t _src13tt_g = vtrn_u16(vreinterpret_u16_u8(_src01t_g.val[1]), vreinterpret_u16_u8(_src23t_g.val[1]));
            uint16x4x2_t _src46tt_g = vtrn_u16(vreinterpret_u16_u8(_src45t_g.val[0]), vreinterpret_u16_u8(_src67t_g.val[0]));
            uint16x4x2_t _src57tt_g = vtrn_u16(vreinterpret_u16_u8(_src45t_g.val[1]), vreinterpret_u16_u8(_src67t_g.val[1]));

            uint16x4x2_t _src02tt_b = vtrn_u16(vreinterpret_u16_u8(_src01t_b.val[0]), vreinterpret_u16_u8(_src23t_b.val[0]));
            uint16x4x2_t _src13tt_b = vtrn_u16(vreinterpret_u16_u8(_src01t_b.val[1]), vreinterpret_u16_u8(_src23t_b.val[1]));
            uint16x4x2_t _src46tt_b = vtrn_u16(vreinterpret_u16_u8(_src45t_b.val[0]), vreinterpret_u16_u8(_src67t_b.val[0]));
            uint16x4x2_t _src57tt_b = vtrn_u16(vreinterpret_u16_u8(_src45t_b.val[1]), vreinterpret_u16_u8(_src67t_b.val[1]));

            uint32x2x2_t _src04ttt_r = vtrn_u32(vreinterpret_u32_u16(_src02tt_r.val[0]), vreinterpret_u32_u16(_src46tt_r.val[0]));
            uint32x2x2_t _src15ttt_r = vtrn_u32(vreinterpret_u32_u16(_src13tt_r.val[0]), vreinterpret_u32_u16(_src57tt_r.val[0]));
            uint32x2x2_t _src26ttt_r = vtrn_u32(vreinterpret_u32_u16(_src02tt_r.val[1]), vreinterpret_u32_u16(_src46tt_r.val[1]));
            uint32x2x2_t _src37ttt_r = vtrn_u32(vreinterpret_u32_u16(_src13tt_r.val[1]), vreinterpret_u32_u16(_src57tt_r.val[1]));

            uint32x2x2_t _src04ttt_g = vtrn_u32(vreinterpret_u32_u16(_src02tt_g.val[0]), vreinterpret_u32_u16(_src46tt_g.val[0]));
            uint32x2x2_t _src15ttt_g = vtrn_u32(vreinterpret_u32_u16(_src13tt_g.val[0]), vreinterpret_u32_u16(_src57tt_g.val[0]));
            uint32x2x2_t _src26ttt_g = vtrn_u32(vreinterpret_u32_u16(_src02tt_g.val[1]), vreinterpret_u32_u16(_src46tt_g.val[1]));
            uint32x2x2_t _src37ttt_g = vtrn_u32(vreinterpret_u32_u16(_src13tt_g.val[1]), vreinterpret_u32_u16(_src57tt_g.val[1]));

            uint32x2x2_t _src04ttt_b = vtrn_u32(vreinterpret_u32_u16(_src02tt_b.val[0]), vreinterpret_u32_u16(_src46tt_b.val[0]));
            uint32x2x2_t _src15ttt_b = vtrn_u32(vreinterpret_u32_u16(_src13tt_b.val[0]), vreinterpret_u32_u16(_src57tt_b.val[0]));
            uint32x2x2_t _src26ttt_b = vtrn_u32(vreinterpret_u32_u16(_src02tt_b.val[1]), vreinterpret_u32_u16(_src46tt_b.val[1]));
            uint32x2x2_t _src37ttt_b = vtrn_u32(vreinterpret_u32_u16(_src13tt_b.val[1]), vreinterpret_u32_u16(_src57tt_b.val[1]));

            uint8x8x3_t _dst0;
            uint8x8x3_t _dst1;
            uint8x8x3_t _dst2;
            uint8x8x3_t _dst3;
            uint8x8x3_t _dst4;
            uint8x8x3_t _dst5;
            uint8x8x3_t _dst6;
            uint8x8x3_t _dst7;

            _dst0.val[0] = vreinterpret_u8_u32(_src04ttt_r.val[0]);
            _dst1.val[0] = vreinterpret_u8_u32(_src15ttt_r.val[0]);
            _dst2.val[0] = vreinterpret_u8_u32(_src26ttt_r.val[0]);
            _dst3.val[0] = vreinterpret_u8_u32(_src37ttt_r.val[0]);
            _dst4.val[0] = vreinterpret_u8_u32(_src04ttt_r.val[1]);
            _dst5.val[0] = vreinterpret_u8_u32(_src15ttt_r.val[1]);
            _dst6.val[0] = vreinterpret_u8_u32(_src26ttt_r.val[1]);
            _dst7.val[0] = vreinterpret_u8_u32(_src37ttt_r.val[1]);

            _dst0.val[1] = vreinterpret_u8_u32(_src04ttt_g.val[0]);
            _dst1.val[1] = vreinterpret_u8_u32(_src15ttt_g.val[0]);
            _dst2.val[1] = vreinterpret_u8_u32(_src26ttt_g.val[0]);
            _dst3.val[1] = vreinterpret_u8_u32(_src37ttt_g.val[0]);
            _dst4.val[1] = vreinterpret_u8_u32(_src04ttt_g.val[1]);
            _dst5.val[1] = vreinterpret_u8_u32(_src15ttt_g.val[1]);
            _dst6.val[1] = vreinterpret_u8_u32(_src26ttt_g.val[1]);
            _dst7.val[1] = vreinterpret_u8_u32(_src37ttt_g.val[1]);

            _dst0.val[2] = vreinterpret_u8_u32(_src04ttt_b.val[0]);
            _dst1.val[2] = vreinterpret_u8_u32(_src15ttt_b.val[0]);
            _dst2.val[2] = vreinterpret_u8_u32(_src26ttt_b.val[0]);
            _dst3.val[2] = vreinterpret_u8_u32(_src37ttt_b.val[0]);
            _dst4.val[2] = vreinterpret_u8_u32(_src04ttt_b.val[1]);
            _dst5.val[2] = vreinterpret_u8_u32(_src15ttt_b.val[1]);
            _dst6.val[2] = vreinterpret_u8_u32(_src26ttt_b.val[1]);
            _dst7.val[2] = vreinterpret_u8_u32(_src37ttt_b.val[1]);

            vst3_u8(dst7, _dst0);
            vst3_u8(dst6, _dst1);
            vst3_u8(dst7 + dst_step, _dst2);
            vst3_u8(dst6 + dst_step, _dst3);
            vst3_u8(dst7 + 2 * dst_step, _dst4);
            vst3_u8(dst6 + 2 * dst_step, _dst5);
            vst3_u8(dst7 + 3 * dst_step, _dst6);
            vst3_u8(dst6 + 3 * dst_step, _dst7);

            src0 += 3 * 8;
            src1 += 3 * 8;

            dst7 += 4 * dst_step;
            dst6 += 4 * dst_step;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #192]          \n"
                "vld3.u8    {d0-d2}, [%1], %10  \n"

                "pld        [%2, #192]          \n"
                "vld3.u8    {d4-d6}, [%2], %10  \n"

                "pld        [%1, #192]          \n"
                "vld3.u8    {d8-d10}, [%1], %10 \n"

                "vtrn.u8    q0, q2              \n" // _src01t_r
                "vtrn.u8    d2, d6              \n"

                "pld        [%2, #192]          \n"
                "vld3.u8    {d12-d14}, [%2], %10\n"

                "pld        [%1, #192]          \n"
                "vld3.u8    {d16-d18}, [%1], %10\n"

                "vtrn.u8    q4, q6              \n" // _src23t_r
                "vtrn.u8    d10, d14            \n"

                "pld        [%2, #192]          \n"
                "vld3.u8    {d20-d22}, [%2], %10\n"

                "pld        [%1, #192]          \n"
                "vld3.u8    {d24-d26}, [%1], %10\n"

                "vtrn.u8    q8, q10             \n" // _src45t_r
                "vtrn.u8    d18, d22            \n"

                "pld        [%2, #192]          \n"
                "vld3.u8    {d28-d30}, [%2], %10\n"

                "vtrn.u8    q12, q14            \n" // _src67t_r
                "vtrn.u8    d26, d30            \n"

                "sub        %1, %1, %10, lsl #2 \n" // restore src0

                "vtrn.u16   q0, q4              \n" // _src02tt_r
                "vtrn.u16   d2, d10             \n"

                "sub        %2, %2, %10, lsl #2 \n" // restore src1

                "vtrn.u16   q2, q6              \n" // _src13tt_r
                "vtrn.u16   d6, d14             \n"

                "add        %1, #24             \n" // src0 += 24

                "vtrn.u16   q8, q12             \n" // _src46tt_r
                "vtrn.u16   d18, d26            \n"

                "add        %2, #24             \n" // src1 += 24

                "vtrn.u16   q10, q14            \n" // _src57tt_r
                "vtrn.u16   d22, d30            \n"

                "vtrn.u32   q0, q8              \n" // _src04ttt_r
                "vtrn.u32   d2, d18             \n"

                "vtrn.u32   q2, q10             \n" // _src15ttt_r
                "vst3.u8    {d0-d2}, [%3], %11  \n"
                "vtrn.u32   d6, d22             \n"

                "vtrn.u32   q4, q12             \n" // _src26ttt_r
                "vst3.u8    {d4-d6}, [%4], %11  \n"
                "vtrn.u32   d10, d26            \n"

                "vtrn.u32   q6, q14             \n" // _src37ttt_r
                "vst3.u8    {d8-d10}, [%3], %11 \n"
                "vtrn.u32   d14, d30            \n"

                "subs       %0, #1              \n"

                "vst3.u8    {d16-d18}, [%3], %11\n"
                "vst3.u8    {d12-d14}, [%4], %11\n"
                "vst3.u8    {d20-d22}, [%4], %11\n"
                "vst3.u8    {d24-d26}, [%3], %11\n"
                "vst3.u8    {d28-d30}, [%4], %11\n"

                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(src1), // %2
                "=r"(dst7), // %3
                "=r"(dst6)  // %4
                : "0"(nn),
                "1"(src0),
                "2"(src1),
                "3"(dst7),
                "4"(dst6),
                "r"(src_step), // %10
                "r"(dst_step)  // %11
                : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
        }
#endif // __aarch64__
        for (; remain > 0; remain--)
        {
            dst7[0] = src0[0];
            dst7[1] = src0[1];
            dst7[2] = src0[2];
            dst7[3] = src1[0];
            dst7[4] = src1[1];
            dst7[5] = src1[2];
            dst7[6] = src0[0 + src_step];
            dst7[7] = src0[1 + src_step];
            dst7[8] = src0[2 + src_step];
            dst7[9] = src1[0 + src_step];
            dst7[10] = src1[1 + src_step];
            dst7[11] = src1[2 + src_step];
            dst7[12] = src0[0 + 2 * src_step];
            dst7[13] = src0[1 + 2 * src_step];
            dst7[14] = src0[2 + 2 * src_step];
            dst7[15] = src1[0 + 2 * src_step];
            dst7[16] = src1[1 + 2 * src_step];
            dst7[17] = src1[2 + 2 * src_step];
            dst7[18] = src0[0 + 3 * src_step];
            dst7[19] = src0[1 + 3 * src_step];
            dst7[20] = src0[2 + 3 * src_step];
            dst7[21] = src1[0 + 3 * src_step];
            dst7[22] = src1[1 + 3 * src_step];
            dst7[23] = src1[2 + 3 * src_step];

            src0 += 3;
            src1 += 3;

            dst7 -= stride;
        }

        src0 += srcwgap + 7 * srcstride;
    }
#endif // __ARM_NEON
    for (; y < srch; y++)
    {
        unsigned char* dst0 = dstend + y * 3;

        int x = 0;
        for (; x < srcw; x++)
        {
            dst0[0] = src0[0];
            dst0[1] = src0[1];
            dst0[2] = src0[2];

            src0 += 3;
            dst0 -= stride;
        }

        src0 += srcwgap;
    }
}

static void kanna_rotate_8_c4(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int /*w*/, int h, int stride)
{
    const int srcwgap = srcstride - srcw * 4;

    // point to the last dst pixel row
    unsigned char* dstend = dst + stride * (h - 1);

    const unsigned char* src0 = src;

    int y = 0;
#if __ARM_NEON
    for (; y + 7 < srch; y += 8)
    {
        const unsigned char* src1 = src0 + srcstride;

        unsigned char* dst7 = dstend + y * 4;
        unsigned char* dst6 = dstend + y * 4 - stride;

        int src_step = 2 * srcstride;
        int dst_step = -2 * stride;

        int nn = srcw >> 3;
        int remain = srcw - (nn << 3);

#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x8x4_t _src0 = vld4_u8(src0);
            uint8x8x4_t _src1 = vld4_u8(src1);

            uint8x8x4_t _src2 = vld4_u8(src0 + src_step);
            uint8x8x4_t _src3 = vld4_u8(src1 + src_step);

            uint8x8x4_t _src4 = vld4_u8(src0 + 2 * src_step);
            uint8x8x4_t _src5 = vld4_u8(src1 + 2 * src_step);

            uint8x8x4_t _src6 = vld4_u8(src0 + 3 * src_step);
            uint8x8x4_t _src7 = vld4_u8(src1 + 3 * src_step);

            uint8x8x2_t _src01t_r = vtrn_u8(_src0.val[0], _src1.val[0]);
            uint8x8x2_t _src23t_r = vtrn_u8(_src2.val[0], _src3.val[0]);
            uint8x8x2_t _src45t_r = vtrn_u8(_src4.val[0], _src5.val[0]);
            uint8x8x2_t _src67t_r = vtrn_u8(_src6.val[0], _src7.val[0]);

            uint8x8x2_t _src01t_g = vtrn_u8(_src0.val[1], _src1.val[1]);
            uint8x8x2_t _src23t_g = vtrn_u8(_src2.val[1], _src3.val[1]);
            uint8x8x2_t _src45t_g = vtrn_u8(_src4.val[1], _src5.val[1]);
            uint8x8x2_t _src67t_g = vtrn_u8(_src6.val[1], _src7.val[1]);

            uint8x8x2_t _src01t_b = vtrn_u8(_src0.val[2], _src1.val[2]);
            uint8x8x2_t _src23t_b = vtrn_u8(_src2.val[2], _src3.val[2]);
            uint8x8x2_t _src45t_b = vtrn_u8(_src4.val[2], _src5.val[2]);
            uint8x8x2_t _src67t_b = vtrn_u8(_src6.val[2], _src7.val[2]);

            uint8x8x2_t _src01t_a = vtrn_u8(_src0.val[3], _src1.val[3]);
            uint8x8x2_t _src23t_a = vtrn_u8(_src2.val[3], _src3.val[3]);
            uint8x8x2_t _src45t_a = vtrn_u8(_src4.val[3], _src5.val[3]);
            uint8x8x2_t _src67t_a = vtrn_u8(_src6.val[3], _src7.val[3]);

            uint16x4x2_t _src02tt_r = vtrn_u16(vreinterpret_u16_u8(_src01t_r.val[0]), vreinterpret_u16_u8(_src23t_r.val[0]));
            uint16x4x2_t _src13tt_r = vtrn_u16(vreinterpret_u16_u8(_src01t_r.val[1]), vreinterpret_u16_u8(_src23t_r.val[1]));
            uint16x4x2_t _src46tt_r = vtrn_u16(vreinterpret_u16_u8(_src45t_r.val[0]), vreinterpret_u16_u8(_src67t_r.val[0]));
            uint16x4x2_t _src57tt_r = vtrn_u16(vreinterpret_u16_u8(_src45t_r.val[1]), vreinterpret_u16_u8(_src67t_r.val[1]));

            uint16x4x2_t _src02tt_g = vtrn_u16(vreinterpret_u16_u8(_src01t_g.val[0]), vreinterpret_u16_u8(_src23t_g.val[0]));
            uint16x4x2_t _src13tt_g = vtrn_u16(vreinterpret_u16_u8(_src01t_g.val[1]), vreinterpret_u16_u8(_src23t_g.val[1]));
            uint16x4x2_t _src46tt_g = vtrn_u16(vreinterpret_u16_u8(_src45t_g.val[0]), vreinterpret_u16_u8(_src67t_g.val[0]));
            uint16x4x2_t _src57tt_g = vtrn_u16(vreinterpret_u16_u8(_src45t_g.val[1]), vreinterpret_u16_u8(_src67t_g.val[1]));

            uint16x4x2_t _src02tt_b = vtrn_u16(vreinterpret_u16_u8(_src01t_b.val[0]), vreinterpret_u16_u8(_src23t_b.val[0]));
            uint16x4x2_t _src13tt_b = vtrn_u16(vreinterpret_u16_u8(_src01t_b.val[1]), vreinterpret_u16_u8(_src23t_b.val[1]));
            uint16x4x2_t _src46tt_b = vtrn_u16(vreinterpret_u16_u8(_src45t_b.val[0]), vreinterpret_u16_u8(_src67t_b.val[0]));
            uint16x4x2_t _src57tt_b = vtrn_u16(vreinterpret_u16_u8(_src45t_b.val[1]), vreinterpret_u16_u8(_src67t_b.val[1]));

            uint16x4x2_t _src02tt_a = vtrn_u16(vreinterpret_u16_u8(_src01t_a.val[0]), vreinterpret_u16_u8(_src23t_a.val[0]));
            uint16x4x2_t _src13tt_a = vtrn_u16(vreinterpret_u16_u8(_src01t_a.val[1]), vreinterpret_u16_u8(_src23t_a.val[1]));
            uint16x4x2_t _src46tt_a = vtrn_u16(vreinterpret_u16_u8(_src45t_a.val[0]), vreinterpret_u16_u8(_src67t_a.val[0]));
            uint16x4x2_t _src57tt_a = vtrn_u16(vreinterpret_u16_u8(_src45t_a.val[1]), vreinterpret_u16_u8(_src67t_a.val[1]));

            uint32x2x2_t _src04ttt_r = vtrn_u32(vreinterpret_u32_u16(_src02tt_r.val[0]), vreinterpret_u32_u16(_src46tt_r.val[0]));
            uint32x2x2_t _src15ttt_r = vtrn_u32(vreinterpret_u32_u16(_src13tt_r.val[0]), vreinterpret_u32_u16(_src57tt_r.val[0]));
            uint32x2x2_t _src26ttt_r = vtrn_u32(vreinterpret_u32_u16(_src02tt_r.val[1]), vreinterpret_u32_u16(_src46tt_r.val[1]));
            uint32x2x2_t _src37ttt_r = vtrn_u32(vreinterpret_u32_u16(_src13tt_r.val[1]), vreinterpret_u32_u16(_src57tt_r.val[1]));

            uint32x2x2_t _src04ttt_g = vtrn_u32(vreinterpret_u32_u16(_src02tt_g.val[0]), vreinterpret_u32_u16(_src46tt_g.val[0]));
            uint32x2x2_t _src15ttt_g = vtrn_u32(vreinterpret_u32_u16(_src13tt_g.val[0]), vreinterpret_u32_u16(_src57tt_g.val[0]));
            uint32x2x2_t _src26ttt_g = vtrn_u32(vreinterpret_u32_u16(_src02tt_g.val[1]), vreinterpret_u32_u16(_src46tt_g.val[1]));
            uint32x2x2_t _src37ttt_g = vtrn_u32(vreinterpret_u32_u16(_src13tt_g.val[1]), vreinterpret_u32_u16(_src57tt_g.val[1]));

            uint32x2x2_t _src04ttt_b = vtrn_u32(vreinterpret_u32_u16(_src02tt_b.val[0]), vreinterpret_u32_u16(_src46tt_b.val[0]));
            uint32x2x2_t _src15ttt_b = vtrn_u32(vreinterpret_u32_u16(_src13tt_b.val[0]), vreinterpret_u32_u16(_src57tt_b.val[0]));
            uint32x2x2_t _src26ttt_b = vtrn_u32(vreinterpret_u32_u16(_src02tt_b.val[1]), vreinterpret_u32_u16(_src46tt_b.val[1]));
            uint32x2x2_t _src37ttt_b = vtrn_u32(vreinterpret_u32_u16(_src13tt_b.val[1]), vreinterpret_u32_u16(_src57tt_b.val[1]));

            uint32x2x2_t _src04ttt_a = vtrn_u32(vreinterpret_u32_u16(_src02tt_a.val[0]), vreinterpret_u32_u16(_src46tt_a.val[0]));
            uint32x2x2_t _src15ttt_a = vtrn_u32(vreinterpret_u32_u16(_src13tt_a.val[0]), vreinterpret_u32_u16(_src57tt_a.val[0]));
            uint32x2x2_t _src26ttt_a = vtrn_u32(vreinterpret_u32_u16(_src02tt_a.val[1]), vreinterpret_u32_u16(_src46tt_a.val[1]));
            uint32x2x2_t _src37ttt_a = vtrn_u32(vreinterpret_u32_u16(_src13tt_a.val[1]), vreinterpret_u32_u16(_src57tt_a.val[1]));

            uint8x8x4_t _dst0;
            uint8x8x4_t _dst1;
            uint8x8x4_t _dst2;
            uint8x8x4_t _dst3;
            uint8x8x4_t _dst4;
            uint8x8x4_t _dst5;
            uint8x8x4_t _dst6;
            uint8x8x4_t _dst7;

            _dst0.val[0] = vreinterpret_u8_u32(_src04ttt_r.val[0]);
            _dst1.val[0] = vreinterpret_u8_u32(_src15ttt_r.val[0]);
            _dst2.val[0] = vreinterpret_u8_u32(_src26ttt_r.val[0]);
            _dst3.val[0] = vreinterpret_u8_u32(_src37ttt_r.val[0]);
            _dst4.val[0] = vreinterpret_u8_u32(_src04ttt_r.val[1]);
            _dst5.val[0] = vreinterpret_u8_u32(_src15ttt_r.val[1]);
            _dst6.val[0] = vreinterpret_u8_u32(_src26ttt_r.val[1]);
            _dst7.val[0] = vreinterpret_u8_u32(_src37ttt_r.val[1]);

            _dst0.val[1] = vreinterpret_u8_u32(_src04ttt_g.val[0]);
            _dst1.val[1] = vreinterpret_u8_u32(_src15ttt_g.val[0]);
            _dst2.val[1] = vreinterpret_u8_u32(_src26ttt_g.val[0]);
            _dst3.val[1] = vreinterpret_u8_u32(_src37ttt_g.val[0]);
            _dst4.val[1] = vreinterpret_u8_u32(_src04ttt_g.val[1]);
            _dst5.val[1] = vreinterpret_u8_u32(_src15ttt_g.val[1]);
            _dst6.val[1] = vreinterpret_u8_u32(_src26ttt_g.val[1]);
            _dst7.val[1] = vreinterpret_u8_u32(_src37ttt_g.val[1]);

            _dst0.val[2] = vreinterpret_u8_u32(_src04ttt_b.val[0]);
            _dst1.val[2] = vreinterpret_u8_u32(_src15ttt_b.val[0]);
            _dst2.val[2] = vreinterpret_u8_u32(_src26ttt_b.val[0]);
            _dst3.val[2] = vreinterpret_u8_u32(_src37ttt_b.val[0]);
            _dst4.val[2] = vreinterpret_u8_u32(_src04ttt_b.val[1]);
            _dst5.val[2] = vreinterpret_u8_u32(_src15ttt_b.val[1]);
            _dst6.val[2] = vreinterpret_u8_u32(_src26ttt_b.val[1]);
            _dst7.val[2] = vreinterpret_u8_u32(_src37ttt_b.val[1]);

            _dst0.val[3] = vreinterpret_u8_u32(_src04ttt_a.val[0]);
            _dst1.val[3] = vreinterpret_u8_u32(_src15ttt_a.val[0]);
            _dst2.val[3] = vreinterpret_u8_u32(_src26ttt_a.val[0]);
            _dst3.val[3] = vreinterpret_u8_u32(_src37ttt_a.val[0]);
            _dst4.val[3] = vreinterpret_u8_u32(_src04ttt_a.val[1]);
            _dst5.val[3] = vreinterpret_u8_u32(_src15ttt_a.val[1]);
            _dst6.val[3] = vreinterpret_u8_u32(_src26ttt_a.val[1]);
            _dst7.val[3] = vreinterpret_u8_u32(_src37ttt_a.val[1]);

            vst4_u8(dst7, _dst0);
            vst4_u8(dst6, _dst1);
            vst4_u8(dst7 + dst_step, _dst2);
            vst4_u8(dst6 + dst_step, _dst3);
            vst4_u8(dst7 + 2 * dst_step, _dst4);
            vst4_u8(dst6 + 2 * dst_step, _dst5);
            vst4_u8(dst7 + 3 * dst_step, _dst6);
            vst4_u8(dst6 + 3 * dst_step, _dst7);

            src0 += 4 * 8;
            src1 += 4 * 8;

            dst7 += 4 * dst_step;
            dst6 += 4 * dst_step;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "vld4.u8    {d0-d3}, [%1], %10  \n"

                "pld        [%2, #256]          \n"
                "vld4.u8    {d4-d7}, [%2], %10  \n"

                "pld        [%1, #256]          \n"
                "vld4.u8    {d8-d11}, [%1], %10 \n"

                "vtrn.u8    q0, q2              \n" // _src01t_r
                "vtrn.u8    q1, q3              \n"

                "pld        [%2, #256]          \n"
                "vld4.u8    {d12-d15}, [%2], %10\n"

                "pld        [%1, #256]          \n"
                "vld4.u8    {d16-d19}, [%1], %10\n"

                "vtrn.u8    q4, q6              \n" // _src23t_r
                "vtrn.u8    q5, q7              \n"

                "pld        [%2, #256]          \n"
                "vld4.u8    {d20-d23}, [%2], %10\n"

                "pld        [%1, #256]          \n"
                "vld4.u8    {d24-d27}, [%1], %10\n"

                "vtrn.u8    q8, q10             \n" // _src45t_r
                "vtrn.u8    q9, q11             \n"

                "pld        [%2, #256]          \n"
                "vld4.u8    {d28-d31}, [%2], %10\n"

                "vtrn.u8    q12, q14            \n" // _src67t_r
                "vtrn.u8    q13, q15            \n"

                "sub        %1, %1, %10, lsl #2 \n" // restore src0

                "vtrn.u16   q0, q4              \n" // _src02tt_r
                "vtrn.u16   q1, q5              \n"

                "sub        %2, %2, %10, lsl #2 \n" // restore src1

                "vtrn.u16   q2, q6              \n" // _src13tt_r
                "vtrn.u16   q3, q7              \n"

                "add        %1, #32             \n" // src0 += 32

                "vtrn.u16   q8, q12             \n" // _src46tt_r
                "vtrn.u16   q9, q13             \n"

                "add        %2, #32             \n" // src1 += 32

                "vtrn.u16   q10, q14            \n" // _src57tt_r
                "vtrn.u16   q11, q15            \n"

                "vtrn.u32   q0, q8              \n" // _src04ttt_r
                "vtrn.u32   q1, q9              \n"

                "vtrn.u32   q2, q10             \n" // _src15ttt_r
                "vst4.u8    {d0-d3}, [%3], %11  \n"
                "vtrn.u32   q3, q11             \n"

                "vtrn.u32   q4, q12             \n" // _src26ttt_r
                "vst4.u8    {d4-d7}, [%4], %11  \n"
                "vtrn.u32   q5, q13             \n"

                "vtrn.u32   q6, q14             \n" // _src37ttt_r
                "vst4.u8    {d8-d11}, [%3], %11 \n"
                "vtrn.u32   q7, q15             \n"

                "subs       %0, #1              \n"

                "vst4.u8    {d16-d19}, [%3], %11\n"
                "vst4.u8    {d12-d15}, [%4], %11\n"
                "vst4.u8    {d20-d23}, [%4], %11\n"
                "vst4.u8    {d24-d27}, [%3], %11\n"
                "vst4.u8    {d28-d31}, [%4], %11\n"

                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(src0), // %1
                "=r"(src1), // %2
                "=r"(dst7), // %3
                "=r"(dst6)  // %4
                : "0"(nn),
                "1"(src0),
                "2"(src1),
                "3"(dst7),
                "4"(dst6),
                "r"(src_step), // %10
                "r"(dst_step)  // %11
                : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
        }
#endif // __aarch64__
        for (; remain > 0; remain--)
        {
            dst7[0] = src0[0];
            dst7[1] = src0[1];
            dst7[2] = src0[2];
            dst7[3] = src0[3];
            dst7[4] = src1[0];
            dst7[5] = src1[1];
            dst7[6] = src1[2];
            dst7[7] = src1[3];
            dst7[8] = src0[0 + src_step];
            dst7[9] = src0[1 + src_step];
            dst7[10] = src0[2 + src_step];
            dst7[11] = src0[3 + src_step];
            dst7[12] = src1[0 + src_step];
            dst7[13] = src1[1 + src_step];
            dst7[14] = src1[2 + src_step];
            dst7[15] = src1[3 + src_step];
            dst7[16] = src0[0 + 2 * src_step];
            dst7[17] = src0[1 + 2 * src_step];
            dst7[18] = src0[2 + 2 * src_step];
            dst7[19] = src0[3 + 2 * src_step];
            dst7[20] = src1[0 + 2 * src_step];
            dst7[21] = src1[1 + 2 * src_step];
            dst7[22] = src1[2 + 2 * src_step];
            dst7[23] = src1[3 + 2 * src_step];
            dst7[24] = src0[0 + 3 * src_step];
            dst7[25] = src0[1 + 3 * src_step];
            dst7[26] = src0[2 + 3 * src_step];
            dst7[27] = src0[3 + 3 * src_step];
            dst7[28] = src1[0 + 3 * src_step];
            dst7[29] = src1[1 + 3 * src_step];
            dst7[30] = src1[2 + 3 * src_step];
            dst7[31] = src1[3 + 3 * src_step];

            src0 += 4;
            src1 += 4;

            dst7 -= stride;
        }

        src0 += srcwgap + 7 * srcstride;
    }
#endif // __ARM_NEON
    for (; y < srch; y++)
    {
        unsigned char* dst0 = dstend + y * 4;

        int x = 0;
        for (; x < srcw; x++)
        {
            dst0[0] = src0[0];
            dst0[1] = src0[1];
            dst0[2] = src0[2];
            dst0[3] = src0[3];

            src0 += 4;
            dst0 -= stride;
        }

        src0 += srcwgap;
    }
}

void kanna_rotate_c1(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, int type)
{
    return kanna_rotate_c1(src, srcw, srch, srcw, dst, w, h, w, type);
}

void kanna_rotate_c2(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, int type)
{
    return kanna_rotate_c2(src, srcw, srch, srcw * 2, dst, w, h, w * 2, type);
}

void kanna_rotate_c3(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, int type)
{
    return kanna_rotate_c3(src, srcw, srch, srcw * 3, dst, w, h, w * 3, type);
}

void kanna_rotate_c4(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, int type)
{
    return kanna_rotate_c4(src, srcw, srch, srcw * 4, dst, w, h, w * 4, type);
}

void kanna_rotate_c1(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, int type)
{
    // assert srcw == w && srch == h for type 1234
    // assert srcw == h && srch == w for type 5678

    switch (type)
    {
    case 1:
        kanna_rotate_1_c1(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 2:
        kanna_rotate_2_c1(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 3:
        kanna_rotate_3_c1(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 4:
        kanna_rotate_4_c1(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 5:
        kanna_rotate_5_c1(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 6:
        kanna_rotate_6_c1(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 7:
        kanna_rotate_7_c1(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 8:
        kanna_rotate_8_c1(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    default:
        // unsupported rotate type
        break;
    }
}

void kanna_rotate_c2(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, int type)
{
    // assert srcw == w && srch == h for type 1234
    // assert srcw == h && srch == w for type 5678

    switch (type)
    {
    case 1:
        kanna_rotate_1_c2(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 2:
        kanna_rotate_2_c2(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 3:
        kanna_rotate_3_c2(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 4:
        kanna_rotate_4_c2(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 5:
        kanna_rotate_5_c2(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 6:
        kanna_rotate_6_c2(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 7:
        kanna_rotate_7_c2(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 8:
        kanna_rotate_8_c2(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    default:
        // unsupported rotate type
        break;
    }
}

void kanna_rotate_c3(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, int type)
{
    // assert srcw == w && srch == h for type 1234
    // assert srcw == h && srch == w for type 5678

    switch (type)
    {
    case 1:
        kanna_rotate_1_c3(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 2:
        kanna_rotate_2_c3(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 3:
        kanna_rotate_3_c3(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 4:
        kanna_rotate_4_c3(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 5:
        kanna_rotate_5_c3(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 6:
        kanna_rotate_6_c3(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 7:
        kanna_rotate_7_c3(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 8:
        kanna_rotate_8_c3(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    default:
        // unsupported rotate type
        break;
    }
}

void kanna_rotate_c4(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, int type)
{
    // assert srcw == w && srch == h for type 1234
    // assert srcw == h && srch == w for type 5678

    switch (type)
    {
    case 1:
        kanna_rotate_1_c4(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 2:
        kanna_rotate_2_c4(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 3:
        kanna_rotate_3_c4(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 4:
        kanna_rotate_4_c4(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 5:
        kanna_rotate_5_c4(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 6:
        kanna_rotate_6_c4(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 7:
        kanna_rotate_7_c4(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 8:
        kanna_rotate_8_c4(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    default:
        // unsupported rotate type
        break;
    }
}

void kanna_rotate_yuv420sp(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, int type)
{
    // assert srcw % 2 == 0
    // assert srch % 2 == 0
    // assert w % 2 == 0
    // assert h % 2 == 0

    const unsigned char* srcY = src;
    unsigned char* dstY = dst;
    kanna_rotate_c1(srcY, srcw, srch, dstY, w, h, type);

    const unsigned char* srcUV = src + srcw * srch;
    unsigned char* dstUV = dst + w * h;
    kanna_rotate_c2(srcUV, srcw / 2, srch / 2, dstUV, w / 2, h / 2, type);
}
#endif // NCNN_PIXEL_ROTATE

} // namespace ncnn
