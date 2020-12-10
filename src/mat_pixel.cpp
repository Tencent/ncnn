// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

#include <limits.h>
#include <math.h>
#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON
#include "platform.h"

namespace ncnn {

#if NCNN_PIXEL
static int from_rgb(const unsigned char* rgb, int w, int h, int stride, Mat& m, Allocator* allocator)
{
    m.create(w, h, 3, 4u, allocator);
    if (m.empty())
        return -100;

    const int wgap = stride - w * 3;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    float* ptr0 = m.channel(0);
    float* ptr1 = m.channel(1);
    float* ptr2 = m.channel(2);

    for (int y = 0; y < h; y++)
    {
#if __ARM_NEON
        int nn = w >> 3;
        int remain = w - (nn << 3);
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x8x3_t _rgb = vld3_u8(rgb);
            uint16x8_t _r16 = vmovl_u8(_rgb.val[0]);
            uint16x8_t _g16 = vmovl_u8(_rgb.val[1]);
            uint16x8_t _b16 = vmovl_u8(_rgb.val[2]);

            float32x4_t _rlow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_r16)));
            float32x4_t _rhigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_r16)));
            float32x4_t _glow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_g16)));
            float32x4_t _ghigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_g16)));
            float32x4_t _blow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_b16)));
            float32x4_t _bhigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_b16)));

            vst1q_f32(ptr0, _rlow);
            vst1q_f32(ptr0 + 4, _rhigh);
            vst1q_f32(ptr1, _glow);
            vst1q_f32(ptr1 + 4, _ghigh);
            vst1q_f32(ptr2, _blow);
            vst1q_f32(ptr2 + 4, _bhigh);

            rgb += 3 * 8;
            ptr0 += 8;
            ptr1 += 8;
            ptr2 += 8;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "vld3.u8    {d0-d2}, [%1]!      \n"
                "vmovl.u8   q8, d0              \n"
                "vmovl.u8   q9, d1              \n"
                "vmovl.u8   q10, d2             \n"
                "vmovl.u16  q0, d16             \n"
                "vmovl.u16  q1, d17             \n"
                "vmovl.u16  q2, d18             \n"
                "vmovl.u16  q3, d19             \n"
                "vmovl.u16  q8, d20             \n"
                "vmovl.u16  q9, d21             \n"
                "vcvt.f32.u32   q0, q0          \n"
                "vcvt.f32.u32   q1, q1          \n"
                "vcvt.f32.u32   q2, q2          \n"
                "vcvt.f32.u32   q3, q3          \n"
                "vcvt.f32.u32   q8, q8          \n"
                "subs       %0, #1              \n"
                "vst1.f32   {d0-d3}, [%2]!      \n"
                "vcvt.f32.u32   q9, q9          \n"
                "vst1.f32   {d4-d7}, [%3]!      \n"
                "vst1.f32   {d16-d19}, [%4]!    \n"
                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(rgb),  // %1
                "=r"(ptr0), // %2
                "=r"(ptr1), // %3
                "=r"(ptr2)  // %4
                : "0"(nn),
                "1"(rgb),
                "2"(ptr0),
                "3"(ptr1),
                "4"(ptr2)
                : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10");
        }
#endif // __aarch64__
#endif // __ARM_NEON
        for (; remain > 0; remain--)
        {
            *ptr0 = rgb[0];
            *ptr1 = rgb[1];
            *ptr2 = rgb[2];

            rgb += 3;
            ptr0++;
            ptr1++;
            ptr2++;
        }

        rgb += wgap;
    }

    return 0;
}

static void to_rgb(const Mat& m, unsigned char* rgb, int stride)
{
    int w = m.w;
    int h = m.h;

    const int wgap = stride - w * 3;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    const float* ptr0 = m.channel(0);
    const float* ptr1 = m.channel(1);
    const float* ptr2 = m.channel(2);

    for (int y = 0; y < h; y++)
    {
#define SATURATE_CAST_UCHAR(X) (unsigned char)::std::min(::std::max((int)(X), 0), 255);

#if __ARM_NEON
        int nn = w >> 3;
        int remain = w - (nn << 3);
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
        for (; nn > 0; nn--)
        {
            float32x4_t _rlow = vld1q_f32(ptr0);
            float32x4_t _rhigh = vld1q_f32(ptr0 + 4);
            float32x4_t _glow = vld1q_f32(ptr1);
            float32x4_t _ghigh = vld1q_f32(ptr1 + 4);
            float32x4_t _blow = vld1q_f32(ptr2);
            float32x4_t _bhigh = vld1q_f32(ptr2 + 4);

            int16x8_t _r16 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(_rlow)), vmovn_s32(vcvtq_s32_f32(_rhigh)));
            int16x8_t _g16 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(_glow)), vmovn_s32(vcvtq_s32_f32(_ghigh)));
            int16x8_t _b16 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(_blow)), vmovn_s32(vcvtq_s32_f32(_bhigh)));

            uint8x8x3_t _rgb;
            _rgb.val[0] = vqmovun_s16(_r16);
            _rgb.val[1] = vqmovun_s16(_g16);
            _rgb.val[2] = vqmovun_s16(_b16);

            vst3_u8(rgb, _rgb);

            rgb += 3 * 8;
            ptr0 += 8;
            ptr1 += 8;
            ptr2 += 8;
        }
#endif // __ARM_NEON
        for (; remain > 0; remain--)
        {
            rgb[0] = SATURATE_CAST_UCHAR(*ptr0);
            rgb[1] = SATURATE_CAST_UCHAR(*ptr1);
            rgb[2] = SATURATE_CAST_UCHAR(*ptr2);

            rgb += 3;
            ptr0++;
            ptr1++;
            ptr2++;
        }

#undef SATURATE_CAST_UCHAR
        rgb += wgap;
    }
}

static int from_gray(const unsigned char* gray, int w, int h, int stride, Mat& m, Allocator* allocator)
{
    m.create(w, h, 1, 4u, allocator);
    if (m.empty())
        return -100;

    const int wgap = stride - w;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    float* ptr = m;

    for (int y = 0; y < h; y++)
    {
#if __ARM_NEON
        int nn = w >> 4;
        int remain = w - (nn << 4);
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x16_t _gray = vld1q_u8(gray);
            uint16x8_t _gray16_0 = vmovl_u8(vget_low_u8(_gray));
            uint16x8_t _gray16_1 = vmovl_u8(vget_high_u8(_gray));

            float32x4_t _graylow_0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_gray16_0)));
            float32x4_t _grayhigh_0 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_gray16_0)));
            float32x4_t _graylow_1 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_gray16_1)));
            float32x4_t _grayhigh_1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_gray16_1)));

            vst1q_f32(ptr, _graylow_0);
            vst1q_f32(ptr + 4, _grayhigh_0);
            vst1q_f32(ptr + 8, _graylow_1);
            vst1q_f32(ptr + 12, _grayhigh_1);

            gray += 16;
            ptr += 16;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #128]          \n"
                "vld1.u8    {d0,d1}, [%1]!      \n"
                "vmovl.u8   q8, d0              \n"
                "vmovl.u8   q9, d1              \n"
                "vmovl.u16  q0, d16             \n"
                "vmovl.u16  q1, d17             \n"
                "vmovl.u16  q2, d18             \n"
                "vmovl.u16  q3, d19             \n"
                "vcvt.f32.u32   q0, q0          \n"
                "vcvt.f32.u32   q1, q1          \n"
                "vcvt.f32.u32   q2, q2          \n"
                "vcvt.f32.u32   q3, q3          \n"
                "subs       %0, #1              \n"
                "vst1.f32   {d0-d3}, [%2]!      \n"
                "vst1.f32   {d4-d7}, [%2]!      \n"
                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(gray), // %1
                "=r"(ptr)   // %2
                : "0"(nn),
                "1"(gray),
                "2"(ptr)
                : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9");
        }
#endif // __aarch64__
#endif // __ARM_NEON
        for (; remain > 0; remain--)
        {
            *ptr = *gray;

            gray++;
            ptr++;
        }

        gray += wgap;
    }

    return 0;
}

static void to_gray(const Mat& m, unsigned char* gray, int stride)
{
    int w = m.w;
    int h = m.h;

    const int wgap = stride - w;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    const float* ptr = m;

    for (int y = 0; y < h; y++)
    {
#define SATURATE_CAST_UCHAR(X) (unsigned char)::std::min(::std::max((int)(X), 0), 255);

#if __ARM_NEON
        int nn = w >> 3;
        int remain = w - (nn << 3);
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
        for (; nn > 0; nn--)
        {
            float32x4_t _glow = vld1q_f32(ptr);
            float32x4_t _ghigh = vld1q_f32(ptr + 4);

            int16x8_t _g16 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(_glow)), vmovn_s32(vcvtq_s32_f32(_ghigh)));

            uint8x8_t _gray = vqmovun_s16(_g16);

            vst1_u8(gray, _gray);

            gray += 8;
            ptr += 8;
        }
#endif // __ARM_NEON
        for (; remain > 0; remain--)
        {
            *gray = SATURATE_CAST_UCHAR(*ptr);

            gray++;
            ptr++;
        }

#undef SATURATE_CAST_UCHAR
        gray += wgap;
    }
}

static int from_rgba(const unsigned char* rgba, int w, int h, int stride, Mat& m, Allocator* allocator)
{
    m.create(w, h, 4, 4u, allocator);
    if (m.empty())
        return -100;

    const int wgap = stride - w * 4;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    float* ptr0 = m.channel(0);
    float* ptr1 = m.channel(1);
    float* ptr2 = m.channel(2);
    float* ptr3 = m.channel(3);

    for (int y = 0; y < h; y++)
    {
#if __ARM_NEON
        int nn = w >> 3;
        int remain = w - (nn << 3);
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x8x4_t _rgba = vld4_u8(rgba);
            int16x8_t _r16 = vreinterpretq_s16_u16(vmovl_u8(_rgba.val[0]));
            int16x8_t _g16 = vreinterpretq_s16_u16(vmovl_u8(_rgba.val[1]));
            int16x8_t _b16 = vreinterpretq_s16_u16(vmovl_u8(_rgba.val[2]));
            int16x8_t _a16 = vreinterpretq_s16_u16(vmovl_u8(_rgba.val[3]));

            float32x4_t _rlow = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_r16)));
            float32x4_t _rhigh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(_r16)));
            float32x4_t _glow = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_g16)));
            float32x4_t _ghigh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(_g16)));
            float32x4_t _blow = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_b16)));
            float32x4_t _bhigh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(_b16)));
            float32x4_t _alow = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_a16)));
            float32x4_t _ahigh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(_a16)));

            vst1q_f32(ptr0, _rlow);
            vst1q_f32(ptr0 + 4, _rhigh);
            vst1q_f32(ptr1, _glow);
            vst1q_f32(ptr1 + 4, _ghigh);
            vst1q_f32(ptr2, _blow);
            vst1q_f32(ptr2 + 4, _bhigh);
            vst1q_f32(ptr3, _alow);
            vst1q_f32(ptr3 + 4, _ahigh);

            rgba += 4 * 8;
            ptr0 += 8;
            ptr1 += 8;
            ptr2 += 8;
            ptr3 += 8;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "vld4.u8    {d0-d3}, [%1]!      \n"
                "vmovl.u8   q8, d0              \n"
                "vmovl.u8   q9, d1              \n"
                "vmovl.u8   q10, d2             \n"
                "vmovl.u8   q11, d3             \n"
                "vmovl.u16  q0, d16             \n"
                "vmovl.u16  q1, d17             \n"
                "vmovl.u16  q2, d18             \n"
                "vmovl.u16  q3, d19             \n"
                "vmovl.u16  q8, d20             \n"
                "vmovl.u16  q9, d21             \n"
                "vmovl.u16  q10, d22            \n"
                "vmovl.u16  q11, d23            \n"
                "vcvt.f32.u32   q0, q0          \n"
                "vcvt.f32.u32   q1, q1          \n"
                "vcvt.f32.u32   q2, q2          \n"
                "vcvt.f32.u32   q3, q3          \n"
                "vcvt.f32.u32   q8, q8          \n"
                "vcvt.f32.u32   q9, q9          \n"
                "subs       %0, #1              \n"
                "vst1.f32   {d0-d3}, [%2]!      \n"
                "vcvt.f32.u32   q10, q10        \n"
                "vcvt.f32.u32   q11, q11        \n"
                "vst1.f32   {d4-d7}, [%3]!      \n"
                "vst1.f32   {d16-d19}, [%4]!    \n"
                "vst1.f32   {d20-d23}, [%5]!    \n"
                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(rgba), // %1
                "=r"(ptr0), // %2
                "=r"(ptr1), // %3
                "=r"(ptr2), // %4
                "=r"(ptr3)  // %5
                : "0"(nn),
                "1"(rgba),
                "2"(ptr0),
                "3"(ptr1),
                "4"(ptr2),
                "5"(ptr3)
                : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11");
        }
#endif // __aarch64__
#endif // __ARM_NEON
        for (; remain > 0; remain--)
        {
            *ptr0 = rgba[0];
            *ptr1 = rgba[1];
            *ptr2 = rgba[2];
            *ptr3 = rgba[3];

            rgba += 4;
            ptr0++;
            ptr1++;
            ptr2++;
            ptr3++;
        }

        rgba += wgap;
    }

    return 0;
}

static void to_rgba(const Mat& m, unsigned char* rgba, int stride)
{
    int w = m.w;
    int h = m.h;

    const int wgap = stride - w * 4;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    const float* ptr0 = m.channel(0);
    const float* ptr1 = m.channel(1);
    const float* ptr2 = m.channel(2);
    const float* ptr3 = m.channel(3);

    for (int y = 0; y < h; y++)
    {
#define SATURATE_CAST_UCHAR(X) (unsigned char)::std::min(::std::max((int)(X), 0), 255);

#if __ARM_NEON
        int nn = w >> 3;
        int remain = w - (nn << 3);
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
        for (; nn > 0; nn--)
        {
            float32x4_t _rlow = vld1q_f32(ptr0);
            float32x4_t _rhigh = vld1q_f32(ptr0 + 4);
            float32x4_t _glow = vld1q_f32(ptr1);
            float32x4_t _ghigh = vld1q_f32(ptr1 + 4);
            float32x4_t _blow = vld1q_f32(ptr2);
            float32x4_t _bhigh = vld1q_f32(ptr2 + 4);
            float32x4_t _alow = vld1q_f32(ptr3);
            float32x4_t _ahigh = vld1q_f32(ptr3 + 4);

            int16x8_t _r16 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(_rlow)), vmovn_s32(vcvtq_s32_f32(_rhigh)));
            int16x8_t _g16 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(_glow)), vmovn_s32(vcvtq_s32_f32(_ghigh)));
            int16x8_t _b16 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(_blow)), vmovn_s32(vcvtq_s32_f32(_bhigh)));
            int16x8_t _a16 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(_alow)), vmovn_s32(vcvtq_s32_f32(_ahigh)));

            uint8x8x4_t _rgba;
            _rgba.val[0] = vqmovun_s16(_r16);
            _rgba.val[1] = vqmovun_s16(_g16);
            _rgba.val[2] = vqmovun_s16(_b16);
            _rgba.val[3] = vqmovun_s16(_a16);

            vst4_u8(rgba, _rgba);

            rgba += 4 * 8;
            ptr0 += 8;
            ptr1 += 8;
            ptr2 += 8;
            ptr3 += 8;
        }
#endif // __ARM_NEON
        for (; remain > 0; remain--)
        {
            rgba[0] = SATURATE_CAST_UCHAR(*ptr0);
            rgba[1] = SATURATE_CAST_UCHAR(*ptr1);
            rgba[2] = SATURATE_CAST_UCHAR(*ptr2);
            rgba[3] = SATURATE_CAST_UCHAR(*ptr3);

            rgba += 4;
            ptr0++;
            ptr1++;
            ptr2++;
            ptr3++;
        }

#undef SATURATE_CAST_UCHAR
        rgba += wgap;
    }
}

static int from_rgb2bgr(const unsigned char* rgb, int w, int h, int stride, Mat& m, Allocator* allocator)
{
    m.create(w, h, 3, 4u, allocator);
    if (m.empty())
        return -100;

    const int wgap = stride - w * 3;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    float* ptr0 = m.channel(0);
    float* ptr1 = m.channel(1);
    float* ptr2 = m.channel(2);

    for (int y = 0; y < h; y++)
    {
#if __ARM_NEON
        int nn = w >> 3;
        int remain = w - (nn << 3);
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x8x3_t _rgb = vld3_u8(rgb);
            uint16x8_t _r16 = vmovl_u8(_rgb.val[0]);
            uint16x8_t _g16 = vmovl_u8(_rgb.val[1]);
            uint16x8_t _b16 = vmovl_u8(_rgb.val[2]);

            float32x4_t _rlow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_r16)));
            float32x4_t _rhigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_r16)));
            float32x4_t _glow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_g16)));
            float32x4_t _ghigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_g16)));
            float32x4_t _blow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_b16)));
            float32x4_t _bhigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_b16)));

            vst1q_f32(ptr2, _rlow);
            vst1q_f32(ptr2 + 4, _rhigh);
            vst1q_f32(ptr1, _glow);
            vst1q_f32(ptr1 + 4, _ghigh);
            vst1q_f32(ptr0, _blow);
            vst1q_f32(ptr0 + 4, _bhigh);

            rgb += 3 * 8;
            ptr0 += 8;
            ptr1 += 8;
            ptr2 += 8;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "vld3.u8    {d0-d2}, [%1]!      \n"
                "vmovl.u8   q8, d0              \n"
                "vmovl.u8   q9, d1              \n"
                "vmovl.u8   q10, d2             \n"
                "vmovl.u16  q0, d16             \n"
                "vmovl.u16  q1, d17             \n"
                "vmovl.u16  q2, d18             \n"
                "vmovl.u16  q3, d19             \n"
                "vmovl.u16  q8, d20             \n"
                "vmovl.u16  q9, d21             \n"
                "vcvt.f32.u32   q0, q0          \n"
                "vcvt.f32.u32   q1, q1          \n"
                "vcvt.f32.u32   q2, q2          \n"
                "vcvt.f32.u32   q3, q3          \n"
                "vcvt.f32.u32   q8, q8          \n"
                "subs       %0, #1              \n"
                "vst1.f32   {d0-d3}, [%4]!      \n"
                "vcvt.f32.u32   q9, q9          \n"
                "vst1.f32   {d4-d7}, [%3]!      \n"
                "vst1.f32   {d16-d19}, [%2]!    \n"
                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(rgb),  // %1
                "=r"(ptr0), // %2
                "=r"(ptr1), // %3
                "=r"(ptr2)  // %4
                : "0"(nn),
                "1"(rgb),
                "2"(ptr0),
                "3"(ptr1),
                "4"(ptr2)
                : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10");
        }
#endif // __aarch64__
#endif // __ARM_NEON
        for (; remain > 0; remain--)
        {
            *ptr0 = rgb[2];
            *ptr1 = rgb[1];
            *ptr2 = rgb[0];

            rgb += 3;
            ptr0++;
            ptr1++;
            ptr2++;
        }

        rgb += wgap;
    }

    return 0;
}

static void to_bgr2rgb(const Mat& m, unsigned char* rgb, int stride)
{
    int w = m.w;
    int h = m.h;

    const int wgap = stride - w * 3;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    const float* ptr0 = m.channel(0);
    const float* ptr1 = m.channel(1);
    const float* ptr2 = m.channel(2);

    for (int y = 0; y < h; y++)
    {
#define SATURATE_CAST_UCHAR(X) (unsigned char)::std::min(::std::max((int)(X), 0), 255);

#if __ARM_NEON
        int nn = w >> 3;
        int remain = w - (nn << 3);
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
        for (; nn > 0; nn--)
        {
            float32x4_t _rlow = vld1q_f32(ptr2);
            float32x4_t _rhigh = vld1q_f32(ptr2 + 4);
            float32x4_t _glow = vld1q_f32(ptr1);
            float32x4_t _ghigh = vld1q_f32(ptr1 + 4);
            float32x4_t _blow = vld1q_f32(ptr0);
            float32x4_t _bhigh = vld1q_f32(ptr0 + 4);

            int16x8_t _r16 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(_rlow)), vmovn_s32(vcvtq_s32_f32(_rhigh)));
            int16x8_t _g16 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(_glow)), vmovn_s32(vcvtq_s32_f32(_ghigh)));
            int16x8_t _b16 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(_blow)), vmovn_s32(vcvtq_s32_f32(_bhigh)));

            uint8x8x3_t _rgb;
            _rgb.val[0] = vqmovun_s16(_r16);
            _rgb.val[1] = vqmovun_s16(_g16);
            _rgb.val[2] = vqmovun_s16(_b16);

            vst3_u8(rgb, _rgb);

            rgb += 3 * 8;
            ptr0 += 8;
            ptr1 += 8;
            ptr2 += 8;
        }
#endif // __ARM_NEON
        for (; remain > 0; remain--)
        {
            rgb[2] = SATURATE_CAST_UCHAR(*ptr0);
            rgb[1] = SATURATE_CAST_UCHAR(*ptr1);
            rgb[0] = SATURATE_CAST_UCHAR(*ptr2);

            rgb += 3;
            ptr0++;
            ptr1++;
            ptr2++;
        }

#undef SATURATE_CAST_UCHAR
        rgb += wgap;
    }
}

static int from_rgb2gray(const unsigned char* rgb, int w, int h, int stride, Mat& m, Allocator* allocator)
{
    // coeffs for r g b = 0.299f, 0.587f, 0.114f
    const unsigned char Y_shift = 8; //14
    const unsigned char R2Y = 77;
    const unsigned char G2Y = 150;
    const unsigned char B2Y = 29;

    m.create(w, h, 1, 4u, allocator);
    if (m.empty())
        return -100;

    const int wgap = stride - w * 3;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    float* ptr = m;

    for (int y = 0; y < h; y++)
    {
#if __ARM_NEON
        int nn = w >> 3;
        int remain = w - (nn << 3);
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
        uint8x8_t _R2Y = vdup_n_u8(R2Y);
        uint8x8_t _G2Y = vdup_n_u8(G2Y);
        uint8x8_t _B2Y = vdup_n_u8(B2Y);
        for (; nn > 0; nn--)
        {
            uint8x8x3_t _rgb = vld3_u8(rgb);

            uint16x8_t _y16 = vmull_u8(_rgb.val[0], _R2Y);
            _y16 = vmlal_u8(_y16, _rgb.val[1], _G2Y);
            _y16 = vmlal_u8(_y16, _rgb.val[2], _B2Y);
            _y16 = vshrq_n_u16(_y16, Y_shift);

            float32x4_t _ylow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_y16)));
            float32x4_t _yhigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_y16)));

            vst1q_f32(ptr, _ylow);
            vst1q_f32(ptr + 4, _yhigh);

            rgb += 3 * 8;
            ptr += 8;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "vdup.u8    d16, %6             \n"
                "vdup.u8    d17, %7             \n"
                "vdup.u8    d18, %8             \n"
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "vld3.u8    {d0-d2}, [%1]!      \n"
                "vmull.u8   q2, d0, d16         \n"
                "vmlal.u8   q2, d1, d17         \n"
                "vmlal.u8   q2, d2, d18         \n"
                "vshr.u16   q2, q2, #8          \n" // Y_shift
                "vmovl.u16  q0, d4              \n"
                "vmovl.u16  q1, d5              \n"
                "vcvt.f32.u32   q0, q0          \n"
                "vcvt.f32.u32   q1, q1          \n"
                "subs       %0, #1              \n"
                "vst1.f32   {d0-d3}, [%2]!      \n"
                "bne        0b                  \n"
                : "=r"(nn),  // %0
                "=r"(rgb), // %1
                "=r"(ptr)  // %2
                : "0"(nn),
                "1"(rgb),
                "2"(ptr),
                "r"(R2Y), // %6
                "r"(G2Y), // %7
                "r"(B2Y)  // %8
                : "cc", "memory", "q0", "q1", "q2", "q8", "q9");
        }
#endif // __aarch64__
#endif // __ARM_NEON
        for (; remain > 0; remain--)
        {
            *ptr = static_cast<float>((rgb[0] * R2Y + rgb[1] * G2Y + rgb[2] * B2Y) >> Y_shift);

            rgb += 3;
            ptr++;
        }

        rgb += wgap;
    }

    return 0;
}

static int from_rgb2rgba(const unsigned char* rgb, int w, int h, int stride, Mat& m, Allocator* allocator)
{
    m.create(w, h, 4, 4u, allocator);
    if (m.empty())
        return -100;

    Mat rgb_channels = m.channel_range(0, 3);
    from_rgb(rgb, w, h, stride, rgb_channels, allocator);

    Mat alpha_channel = m.channel(3);
    alpha_channel.fill(255.f);

    return 0;
}

static void to_rgb2rgba(const Mat& m, unsigned char* rgba, int stride)
{
    int w = m.w;
    int h = m.h;

    const int wgap = stride - w * 4;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    const float* ptr0 = m.channel(0);
    const float* ptr1 = m.channel(1);
    const float* ptr2 = m.channel(2);

    for (int y = 0; y < h; y++)
    {
#define SATURATE_CAST_UCHAR(X) (unsigned char)::std::min(::std::max((int)(X), 0), 255);

#if __ARM_NEON
        int nn = w >> 3;
        int remain = w - (nn << 3);
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
        uint8x8_t _a = vdup_n_u8(255);
        for (; nn > 0; nn--)
        {
            float32x4_t _rlow = vld1q_f32(ptr0);
            float32x4_t _rhigh = vld1q_f32(ptr0 + 4);
            float32x4_t _glow = vld1q_f32(ptr1);
            float32x4_t _ghigh = vld1q_f32(ptr1 + 4);
            float32x4_t _blow = vld1q_f32(ptr2);
            float32x4_t _bhigh = vld1q_f32(ptr2 + 4);

            int16x8_t _r16 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(_rlow)), vmovn_s32(vcvtq_s32_f32(_rhigh)));
            int16x8_t _g16 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(_glow)), vmovn_s32(vcvtq_s32_f32(_ghigh)));
            int16x8_t _b16 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(_blow)), vmovn_s32(vcvtq_s32_f32(_bhigh)));

            uint8x8x4_t _rgba;
            _rgba.val[0] = vqmovun_s16(_r16);
            _rgba.val[1] = vqmovun_s16(_g16);
            _rgba.val[2] = vqmovun_s16(_b16);
            _rgba.val[3] = _a;

            vst4_u8(rgba, _rgba);

            rgba += 4 * 8;
            ptr0 += 8;
            ptr1 += 8;
            ptr2 += 8;
        }
#endif // __ARM_NEON
        for (; remain > 0; remain--)
        {
            rgba[0] = SATURATE_CAST_UCHAR(*ptr0);
            rgba[1] = SATURATE_CAST_UCHAR(*ptr1);
            rgba[2] = SATURATE_CAST_UCHAR(*ptr2);
            rgba[3] = 255;

            rgba += 4;
            ptr0++;
            ptr1++;
            ptr2++;
        }

#undef SATURATE_CAST_UCHAR
        rgba += wgap;
    }
}

static int from_bgr2gray(const unsigned char* bgr, int w, int h, int stride, Mat& m, Allocator* allocator)
{
    // coeffs for r g b = 0.299f, 0.587f, 0.114f
    const unsigned char Y_shift = 8; //14
    const unsigned char R2Y = 77;
    const unsigned char G2Y = 150;
    const unsigned char B2Y = 29;

    m.create(w, h, 1, 4u, allocator);
    if (m.empty())
        return -100;

    const int wgap = stride - w * 3;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    float* ptr = m;

    for (int y = 0; y < h; y++)
    {
#if __ARM_NEON
        int nn = w >> 3;
        int remain = w - (nn << 3);
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
        uint8x8_t _R2Y = vdup_n_u8(R2Y);
        uint8x8_t _G2Y = vdup_n_u8(G2Y);
        uint8x8_t _B2Y = vdup_n_u8(B2Y);
        for (; nn > 0; nn--)
        {
            uint8x8x3_t _rgb = vld3_u8(bgr);

            uint16x8_t _y16 = vmull_u8(_rgb.val[2], _R2Y);
            _y16 = vmlal_u8(_y16, _rgb.val[1], _G2Y);
            _y16 = vmlal_u8(_y16, _rgb.val[0], _B2Y);
            _y16 = vshrq_n_u16(_y16, Y_shift);

            float32x4_t _ylow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_y16)));
            float32x4_t _yhigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_y16)));

            vst1q_f32(ptr, _ylow);
            vst1q_f32(ptr + 4, _yhigh);

            bgr += 3 * 8;
            ptr += 8;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "vdup.u8    d16, %6             \n"
                "vdup.u8    d17, %7             \n"
                "vdup.u8    d18, %8             \n"
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "vld3.u8    {d0-d2}, [%1]!      \n"
                "vmull.u8   q2, d2, d16         \n"
                "vmlal.u8   q2, d1, d17         \n"
                "vmlal.u8   q2, d0, d18         \n"
                "vshr.u16   q2, q2, #8          \n" // Y_shift
                "vmovl.u16  q0, d4              \n"
                "vmovl.u16  q1, d5              \n"
                "vcvt.f32.u32   q0, q0          \n"
                "vcvt.f32.u32   q1, q1          \n"
                "subs       %0, #1              \n"
                "vst1.f32   {d0-d3}, [%2]!      \n"
                "bne        0b                  \n"
                : "=r"(nn),  // %0
                "=r"(bgr), // %1
                "=r"(ptr)  // %2
                : "0"(nn),
                "1"(bgr),
                "2"(ptr),
                "r"(R2Y), // %6
                "r"(G2Y), // %7
                "r"(B2Y)  // %8
                : "cc", "memory", "q0", "q1", "q2", "q8", "q9");
        }
#endif // __aarch64__
#endif // __ARM_NEON
        for (; remain > 0; remain--)
        {
            *ptr = static_cast<float>((bgr[2] * R2Y + bgr[1] * G2Y + bgr[0] * B2Y) >> Y_shift);

            bgr += 3;
            ptr++;
        }

        bgr += wgap;
    }

    return 0;
}

static int from_bgr2rgba(const unsigned char* bgr, int w, int h, int stride, Mat& m, Allocator* allocator)
{
    m.create(w, h, 4, 4u, allocator);
    if (m.empty())
        return -100;

    Mat rgb_channels = m.channel_range(0, 3);
    from_rgb2bgr(bgr, w, h, stride, rgb_channels, allocator);

    Mat alpha_channel = m.channel(3);
    alpha_channel.fill(255.f);

    return 0;
}

static void to_bgr2rgba(const Mat& m, unsigned char* rgba, int stride)
{
    int w = m.w;
    int h = m.h;

    const int wgap = stride - w * 4;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    const float* ptr0 = m.channel(0);
    const float* ptr1 = m.channel(1);
    const float* ptr2 = m.channel(2);

    for (int y = 0; y < h; y++)
    {
#define SATURATE_CAST_UCHAR(X) (unsigned char)::std::min(::std::max((int)(X), 0), 255);

#if __ARM_NEON
        int nn = w >> 3;
        int remain = w - (nn << 3);
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
        uint8x8_t _a = vdup_n_u8(255);
        for (; nn > 0; nn--)
        {
            float32x4_t _rlow = vld1q_f32(ptr2);
            float32x4_t _rhigh = vld1q_f32(ptr2 + 4);
            float32x4_t _glow = vld1q_f32(ptr1);
            float32x4_t _ghigh = vld1q_f32(ptr1 + 4);
            float32x4_t _blow = vld1q_f32(ptr0);
            float32x4_t _bhigh = vld1q_f32(ptr0 + 4);

            int16x8_t _r16 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(_rlow)), vmovn_s32(vcvtq_s32_f32(_rhigh)));
            int16x8_t _g16 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(_glow)), vmovn_s32(vcvtq_s32_f32(_ghigh)));
            int16x8_t _b16 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(_blow)), vmovn_s32(vcvtq_s32_f32(_bhigh)));

            uint8x8x4_t _rgba;
            _rgba.val[0] = vqmovun_s16(_r16);
            _rgba.val[1] = vqmovun_s16(_g16);
            _rgba.val[2] = vqmovun_s16(_b16);
            _rgba.val[3] = _a;

            vst4_u8(rgba, _rgba);

            rgba += 4 * 8;
            ptr0 += 8;
            ptr1 += 8;
            ptr2 += 8;
        }
#endif // __ARM_NEON
        for (; remain > 0; remain--)
        {
            rgba[0] = SATURATE_CAST_UCHAR(*ptr2);
            rgba[1] = SATURATE_CAST_UCHAR(*ptr1);
            rgba[2] = SATURATE_CAST_UCHAR(*ptr0);
            rgba[3] = 255;

            rgba += 4;
            ptr0++;
            ptr1++;
            ptr2++;
        }

#undef SATURATE_CAST_UCHAR
        rgba += wgap;
    }
}

static int from_gray2rgb(const unsigned char* gray, int w, int h, int stride, Mat& m, Allocator* allocator)
{
    m.create(w, h, 3, 4u, allocator);
    if (m.empty())
        return -100;

    const int wgap = stride - w;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    float* ptr0 = m.channel(0);
    float* ptr1 = m.channel(1);
    float* ptr2 = m.channel(2);

    for (int y = 0; y < h; y++)
    {
#if __ARM_NEON
        int nn = w >> 4;
        int remain = w - (nn << 4);
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x16_t _gray = vld1q_u8(gray);
            uint16x8_t _gray16_0 = vmovl_u8(vget_low_u8(_gray));
            uint16x8_t _gray16_1 = vmovl_u8(vget_high_u8(_gray));

            float32x4_t _graylow_0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_gray16_0)));
            float32x4_t _grayhigh_0 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_gray16_0)));
            float32x4_t _graylow_1 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_gray16_1)));
            float32x4_t _grayhigh_1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_gray16_1)));

            vst1q_f32(ptr0, _graylow_0);
            vst1q_f32(ptr0 + 4, _grayhigh_0);
            vst1q_f32(ptr0 + 8, _graylow_1);
            vst1q_f32(ptr0 + 12, _grayhigh_1);

            vst1q_f32(ptr1, _graylow_0);
            vst1q_f32(ptr1 + 4, _grayhigh_0);
            vst1q_f32(ptr1 + 8, _graylow_1);
            vst1q_f32(ptr1 + 12, _grayhigh_1);

            vst1q_f32(ptr2, _graylow_0);
            vst1q_f32(ptr2 + 4, _grayhigh_0);
            vst1q_f32(ptr2 + 8, _graylow_1);
            vst1q_f32(ptr2 + 12, _grayhigh_1);

            gray += 16;
            ptr0 += 16;
            ptr1 += 16;
            ptr2 += 16;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #128]          \n"
                "vld1.u8    {d0,d1}, [%1]!      \n"
                "vmovl.u8   q8, d0              \n"
                "vmovl.u8   q9, d1              \n"
                "vmovl.u16  q0, d16             \n"
                "vmovl.u16  q1, d17             \n"
                "vmovl.u16  q2, d18             \n"
                "vmovl.u16  q3, d19             \n"
                "vcvt.f32.u32   q0, q0          \n"
                "vcvt.f32.u32   q1, q1          \n"
                "vcvt.f32.u32   q2, q2          \n"
                "vcvt.f32.u32   q3, q3          \n"
                "subs       %0, #1              \n"
                "vst1.f32   {d0-d3}, [%2]!      \n"
                "vst1.f32   {d4-d7}, [%2]!      \n"
                "vst1.f32   {d0-d3}, [%3]!      \n"
                "vst1.f32   {d4-d7}, [%3]!      \n"
                "vst1.f32   {d0-d3}, [%4]!      \n"
                "vst1.f32   {d4-d7}, [%4]!      \n"
                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(gray), // %1
                "=r"(ptr0), // %2
                "=r"(ptr1), // %3
                "=r"(ptr2)  // %4
                : "0"(nn),
                "1"(gray),
                "2"(ptr0),
                "3"(ptr1),
                "4"(ptr2)
                : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9");
        }
#endif // __aarch64__
#endif // __ARM_NEON
        for (; remain > 0; remain--)
        {
            *ptr0 = *gray;
            *ptr1 = *gray;
            *ptr2 = *gray;

            gray++;
            ptr0++;
            ptr1++;
            ptr2++;
        }

        gray += wgap;
    }

    return 0;
}

static int from_gray2rgba(const unsigned char* gray, int w, int h, int stride, Mat& m, Allocator* allocator)
{
    m.create(w, h, 4, 4u, allocator);
    if (m.empty())
        return -100;

    Mat rgb_channels = m.channel_range(0, 3);
    from_gray2rgb(gray, w, h, stride, rgb_channels, allocator);

    Mat alpha_channel = m.channel(3);
    alpha_channel.fill(255.f);

    return 0;
}

static void to_gray2rgba(const Mat& m, unsigned char* rgba, int stride)
{
    int w = m.w;
    int h = m.h;

    const int wgap = stride - w * 4;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    const float* ptr = m;

    for (int y = 0; y < h; y++)
    {
#define SATURATE_CAST_UCHAR(X) (unsigned char)::std::min(::std::max((int)(X), 0), 255);

#if __ARM_NEON
        int nn = w >> 3;
        int remain = w - (nn << 3);
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
        uint8x8_t _a = vdup_n_u8(255);
        for (; nn > 0; nn--)
        {
            float32x4_t _glow = vld1q_f32(ptr);
            float32x4_t _ghigh = vld1q_f32(ptr + 4);

            int16x8_t _g16 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(_glow)), vmovn_s32(vcvtq_s32_f32(_ghigh)));

            uint8x8_t _gray = vqmovun_s16(_g16);

            uint8x8x4_t _rgba;
            _rgba.val[0] = _gray;
            _rgba.val[1] = _gray;
            _rgba.val[2] = _gray;
            _rgba.val[3] = _a;

            vst4_u8(rgba, _rgba);

            rgba += 4 * 8;
            ptr += 8;
        }
#endif // __ARM_NEON
        for (; remain > 0; remain--)
        {
            unsigned char gray = SATURATE_CAST_UCHAR(*ptr);
            rgba[0] = gray;
            rgba[1] = gray;
            rgba[2] = gray;
            rgba[3] = 255;

            rgba += 4;
            ptr++;
        }

#undef SATURATE_CAST_UCHAR
        rgba += wgap;
    }
}

static int from_rgba2rgb(const unsigned char* rgba, int w, int h, int stride, Mat& m, Allocator* allocator)
{
    m.create(w, h, 3, 4u, allocator);
    if (m.empty())
        return -100;

    const int wgap = stride - w * 4;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    float* ptr0 = m.channel(0);
    float* ptr1 = m.channel(1);
    float* ptr2 = m.channel(2);

    for (int y = 0; y < h; y++)
    {
#if __ARM_NEON
        int nn = w >> 3;
        int remain = w - (nn << 3);
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x8x4_t _rgba = vld4_u8(rgba);
            int16x8_t _r16 = vreinterpretq_s16_u16(vmovl_u8(_rgba.val[0]));
            int16x8_t _g16 = vreinterpretq_s16_u16(vmovl_u8(_rgba.val[1]));
            int16x8_t _b16 = vreinterpretq_s16_u16(vmovl_u8(_rgba.val[2]));

            float32x4_t _rlow = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_r16)));
            float32x4_t _rhigh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(_r16)));
            float32x4_t _glow = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_g16)));
            float32x4_t _ghigh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(_g16)));
            float32x4_t _blow = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_b16)));
            float32x4_t _bhigh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(_b16)));

            vst1q_f32(ptr0, _rlow);
            vst1q_f32(ptr0 + 4, _rhigh);
            vst1q_f32(ptr1, _glow);
            vst1q_f32(ptr1 + 4, _ghigh);
            vst1q_f32(ptr2, _blow);
            vst1q_f32(ptr2 + 4, _bhigh);

            rgba += 4 * 8;
            ptr0 += 8;
            ptr1 += 8;
            ptr2 += 8;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "vld4.u8    {d0-d3}, [%1]!      \n"
                "vmovl.u8   q8, d0              \n"
                "vmovl.u8   q9, d1              \n"
                "vmovl.u8   q10, d2             \n"
                "vmovl.u16  q0, d16             \n"
                "vmovl.u16  q1, d17             \n"
                "vmovl.u16  q2, d18             \n"
                "vmovl.u16  q3, d19             \n"
                "vmovl.u16  q8, d20             \n"
                "vmovl.u16  q9, d21             \n"
                "vcvt.f32.u32   q0, q0          \n"
                "vcvt.f32.u32   q1, q1          \n"
                "vcvt.f32.u32   q2, q2          \n"
                "vcvt.f32.u32   q3, q3          \n"
                "vcvt.f32.u32   q8, q8          \n"
                "subs       %0, #1              \n"
                "vst1.f32   {d0-d3}, [%2]!      \n"
                "vcvt.f32.u32   q9, q9          \n"
                "vst1.f32   {d4-d7}, [%3]!      \n"
                "vst1.f32   {d16-d19}, [%4]!    \n"
                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(rgba), // %1
                "=r"(ptr0), // %2
                "=r"(ptr1), // %3
                "=r"(ptr2)  // %4
                : "0"(nn),
                "1"(rgba),
                "2"(ptr0),
                "3"(ptr1),
                "4"(ptr2)
                : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9");
        }
#endif // __aarch64__
#endif // __ARM_NEON
        for (; remain > 0; remain--)
        {
            *ptr0 = rgba[0];
            *ptr1 = rgba[1];
            *ptr2 = rgba[2];

            rgba += 4;
            ptr0++;
            ptr1++;
            ptr2++;
        }

        rgba += wgap;
    }

    return 0;
}

static int from_rgba2bgr(const unsigned char* rgba, int w, int h, int stride, Mat& m, Allocator* allocator)
{
    m.create(w, h, 3, 4u, allocator);
    if (m.empty())
        return -100;

    const int wgap = stride - w * 4;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    float* ptr0 = m.channel(0);
    float* ptr1 = m.channel(1);
    float* ptr2 = m.channel(2);

    for (int y = 0; y < h; y++)
    {
#if __ARM_NEON
        int nn = w >> 3;
        int remain = w - (nn << 3);
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x8x4_t _rgba = vld4_u8(rgba);
            int16x8_t _r16 = vreinterpretq_s16_u16(vmovl_u8(_rgba.val[0]));
            int16x8_t _g16 = vreinterpretq_s16_u16(vmovl_u8(_rgba.val[1]));
            int16x8_t _b16 = vreinterpretq_s16_u16(vmovl_u8(_rgba.val[2]));

            float32x4_t _rlow = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_r16)));
            float32x4_t _rhigh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(_r16)));
            float32x4_t _glow = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_g16)));
            float32x4_t _ghigh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(_g16)));
            float32x4_t _blow = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_b16)));
            float32x4_t _bhigh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(_b16)));

            vst1q_f32(ptr2, _rlow);
            vst1q_f32(ptr2 + 4, _rhigh);
            vst1q_f32(ptr1, _glow);
            vst1q_f32(ptr1 + 4, _ghigh);
            vst1q_f32(ptr0, _blow);
            vst1q_f32(ptr0 + 4, _bhigh);

            rgba += 4 * 8;
            ptr0 += 8;
            ptr1 += 8;
            ptr2 += 8;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "vld4.u8    {d0-d3}, [%1]!      \n"
                "vmovl.u8   q8, d0              \n"
                "vmovl.u8   q9, d1              \n"
                "vmovl.u8   q10, d2             \n"
                "vmovl.u16  q0, d16             \n"
                "vmovl.u16  q1, d17             \n"
                "vmovl.u16  q2, d18             \n"
                "vmovl.u16  q3, d19             \n"
                "vmovl.u16  q8, d20             \n"
                "vmovl.u16  q9, d21             \n"
                "vcvt.f32.u32   q0, q0          \n"
                "vcvt.f32.u32   q1, q1          \n"
                "vcvt.f32.u32   q2, q2          \n"
                "vcvt.f32.u32   q3, q3          \n"
                "vcvt.f32.u32   q8, q8          \n"
                "subs       %0, #1              \n"
                "vst1.f32   {d0-d3}, [%4]!      \n"
                "vcvt.f32.u32   q9, q9          \n"
                "vst1.f32   {d4-d7}, [%3]!      \n"
                "vst1.f32   {d16-d19}, [%2]!    \n"
                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(rgba), // %1
                "=r"(ptr0), // %2
                "=r"(ptr1), // %3
                "=r"(ptr2)  // %4
                : "0"(nn),
                "1"(rgba),
                "2"(ptr0),
                "3"(ptr1),
                "4"(ptr2)
                : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10");
        }
#endif // __aarch64__
#endif // __ARM_NEON
        for (; remain > 0; remain--)
        {
            *ptr0 = rgba[2];
            *ptr1 = rgba[1];
            *ptr2 = rgba[0];

            rgba += 4;
            ptr0++;
            ptr1++;
            ptr2++;
        }

        rgba += wgap;
    }

    return 0;
}

static int from_rgba2gray(const unsigned char* rgba, int w, int h, int stride, Mat& m, Allocator* allocator)
{
    // coeffs for r g b = 0.299f, 0.587f, 0.114f
    const unsigned char Y_shift = 8; //14
    const unsigned char R2Y = 77;
    const unsigned char G2Y = 150;
    const unsigned char B2Y = 29;

    m.create(w, h, 1, 4u, allocator);
    if (m.empty())
        return -100;

    const int wgap = stride - w * 4;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    float* ptr = m;

    for (int y = 0; y < h; y++)
    {
#if __ARM_NEON
        int nn = w >> 3;
        int remain = w - (nn << 3);
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
        uint8x8_t _R2Y = vdup_n_u8(R2Y);
        uint8x8_t _G2Y = vdup_n_u8(G2Y);
        uint8x8_t _B2Y = vdup_n_u8(B2Y);
        for (; nn > 0; nn--)
        {
            uint8x8x4_t _rgba = vld4_u8(rgba);

            uint16x8_t _y16 = vmull_u8(_rgba.val[0], _R2Y);
            _y16 = vmlal_u8(_y16, _rgba.val[1], _G2Y);
            _y16 = vmlal_u8(_y16, _rgba.val[2], _B2Y);
            _y16 = vshrq_n_u16(_y16, Y_shift);

            float32x4_t _ylow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_y16)));
            float32x4_t _yhigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_y16)));

            vst1q_f32(ptr, _ylow);
            vst1q_f32(ptr + 4, _yhigh);

            rgba += 4 * 8;
            ptr += 8;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "vdup.u8    d16, %6             \n"
                "vdup.u8    d17, %7             \n"
                "vdup.u8    d18, %8             \n"
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "vld4.u8    {d0-d3}, [%1]!      \n"
                "vmull.u8   q2, d0, d16         \n"
                "vmlal.u8   q2, d1, d17         \n"
                "vmlal.u8   q2, d2, d18         \n"
                "vshr.u16   q2, q2, #8          \n" // Y_shift
                "vmovl.u16  q0, d4              \n"
                "vmovl.u16  q1, d5              \n"
                "vcvt.f32.u32   q0, q0          \n"
                "vcvt.f32.u32   q1, q1          \n"
                "subs       %0, #1              \n"
                "vst1.f32   {d0-d3}, [%2]!      \n"
                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(rgba), // %1
                "=r"(ptr)   // %2
                : "0"(nn),
                "1"(rgba),
                "2"(ptr),
                "r"(R2Y), // %6
                "r"(G2Y), // %7
                "r"(B2Y)  // %8
                : "cc", "memory", "q0", "q1", "q2", "q8", "q9");
        }
#endif // __aarch64__
#endif // __ARM_NEON
        for (; remain > 0; remain--)
        {
            *ptr = static_cast<float>((rgba[0] * R2Y + rgba[1] * G2Y + rgba[2] * B2Y) >> Y_shift);

            rgba += 4;
            ptr++;
        }

        rgba += wgap;
    }

    return 0;
}

static int from_rgba2bgra(const unsigned char* rgba, int w, int h, int stride, Mat& m, Allocator* allocator)
{
    m.create(w, h, 4, 4u, allocator);
    if (m.empty())
        return -100;

    const int wgap = stride - w * 4;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    float* ptr0 = m.channel(0);
    float* ptr1 = m.channel(1);
    float* ptr2 = m.channel(2);
    float* ptr3 = m.channel(3);

    for (int y = 0; y < h; y++)
    {
#if __ARM_NEON
        int nn = w >> 3;
        int remain = w - (nn << 3);
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x8x4_t _rgba = vld4_u8(rgba);
            int16x8_t _r16 = vreinterpretq_s16_u16(vmovl_u8(_rgba.val[0]));
            int16x8_t _g16 = vreinterpretq_s16_u16(vmovl_u8(_rgba.val[1]));
            int16x8_t _b16 = vreinterpretq_s16_u16(vmovl_u8(_rgba.val[2]));
            int16x8_t _a16 = vreinterpretq_s16_u16(vmovl_u8(_rgba.val[3]));

            float32x4_t _rlow = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_r16)));
            float32x4_t _rhigh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(_r16)));
            float32x4_t _glow = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_g16)));
            float32x4_t _ghigh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(_g16)));
            float32x4_t _blow = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_b16)));
            float32x4_t _bhigh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(_b16)));
            float32x4_t _alow = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_a16)));
            float32x4_t _ahigh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(_a16)));

            vst1q_f32(ptr2, _rlow);
            vst1q_f32(ptr2 + 4, _rhigh);
            vst1q_f32(ptr1, _glow);
            vst1q_f32(ptr1 + 4, _ghigh);
            vst1q_f32(ptr0, _blow);
            vst1q_f32(ptr0 + 4, _bhigh);
            vst1q_f32(ptr3, _alow);
            vst1q_f32(ptr3 + 4, _ahigh);

            rgba += 4 * 8;
            ptr0 += 8;
            ptr1 += 8;
            ptr2 += 8;
            ptr3 += 8;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "vld4.u8    {d0-d3}, [%1]!      \n"
                "vmovl.u8   q8, d0              \n"
                "vmovl.u8   q9, d1              \n"
                "vmovl.u8   q10, d2             \n"
                "vmovl.u8   q11, d3             \n"
                "vmovl.u16  q0, d16             \n"
                "vmovl.u16  q1, d17             \n"
                "vmovl.u16  q2, d18             \n"
                "vmovl.u16  q3, d19             \n"
                "vmovl.u16  q8, d20             \n"
                "vmovl.u16  q9, d21             \n"
                "vmovl.u16  q10, d22            \n"
                "vmovl.u16  q11, d23            \n"
                "vcvt.f32.u32   q0, q0          \n"
                "vcvt.f32.u32   q1, q1          \n"
                "vcvt.f32.u32   q2, q2          \n"
                "vcvt.f32.u32   q3, q3          \n"
                "vcvt.f32.u32   q8, q8          \n"
                "subs       %0, #1              \n"
                "vst1.f32   {d0-d3}, [%4]!      \n"
                "vcvt.f32.u32   q9, q9          \n"
                "vcvt.f32.u32   q10, q10        \n"
                "vst1.f32   {d4-d7}, [%3]!      \n"
                "vcvt.f32.u32   q11, q11        \n"
                "vst1.f32   {d16-d19}, [%2]!    \n"
                "vst1.f32   {d20-d23}, [%5]!    \n"
                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(rgba), // %1
                "=r"(ptr0), // %2
                "=r"(ptr1), // %3
                "=r"(ptr2), // %4
                "=r"(ptr3)  // %5
                : "0"(nn),
                "1"(rgba),
                "2"(ptr0),
                "3"(ptr1),
                "4"(ptr2),
                "5"(ptr3)
                : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11");
        }
#endif // __aarch64__
#endif // __ARM_NEON
        for (; remain > 0; remain--)
        {
            *ptr0 = rgba[2];
            *ptr1 = rgba[1];
            *ptr2 = rgba[0];
            *ptr3 = rgba[3];

            rgba += 4;
            ptr0++;
            ptr1++;
            ptr2++;
            ptr3++;
        }

        rgba += wgap;
    }

    return 0;
}

static void to_rgba2bgra(const Mat& m, unsigned char* bgra, int stride)
{
    int w = m.w;
    int h = m.h;

    const int wgap = stride - w * 4;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    const float* ptr0 = m.channel(0);
    const float* ptr1 = m.channel(1);
    const float* ptr2 = m.channel(2);
    const float* ptr3 = m.channel(3);

    for (int y = 0; y < h; y++)
    {
#define SATURATE_CAST_UCHAR(X) (unsigned char)::std::min(::std::max((int)(X), 0), 255);

#if __ARM_NEON
        int nn = w >> 3;
        int remain = w - (nn << 3);
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
        for (; nn > 0; nn--)
        {
            float32x4_t _rlow = vld1q_f32(ptr0);
            float32x4_t _rhigh = vld1q_f32(ptr0 + 4);
            float32x4_t _glow = vld1q_f32(ptr1);
            float32x4_t _ghigh = vld1q_f32(ptr1 + 4);
            float32x4_t _blow = vld1q_f32(ptr2);
            float32x4_t _bhigh = vld1q_f32(ptr2 + 4);
            float32x4_t _alow = vld1q_f32(ptr3);
            float32x4_t _ahigh = vld1q_f32(ptr3 + 4);

            int16x8_t _r16 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(_rlow)), vmovn_s32(vcvtq_s32_f32(_rhigh)));
            int16x8_t _g16 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(_glow)), vmovn_s32(vcvtq_s32_f32(_ghigh)));
            int16x8_t _b16 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(_blow)), vmovn_s32(vcvtq_s32_f32(_bhigh)));
            int16x8_t _a16 = vcombine_s16(vmovn_s32(vcvtq_s32_f32(_alow)), vmovn_s32(vcvtq_s32_f32(_ahigh)));

            uint8x8x4_t _bgra;
            _bgra.val[0] = vqmovun_s16(_b16);
            _bgra.val[1] = vqmovun_s16(_g16);
            _bgra.val[2] = vqmovun_s16(_r16);
            _bgra.val[3] = vqmovun_s16(_a16);

            vst4_u8(bgra, _bgra);

            bgra += 4 * 8;
            ptr0 += 8;
            ptr1 += 8;
            ptr2 += 8;
            ptr3 += 8;
        }
#endif // __ARM_NEON
        for (; remain > 0; remain--)
        {
            bgra[0] = SATURATE_CAST_UCHAR(*ptr2);
            bgra[1] = SATURATE_CAST_UCHAR(*ptr1);
            bgra[2] = SATURATE_CAST_UCHAR(*ptr0);
            bgra[3] = SATURATE_CAST_UCHAR(*ptr3);

            bgra += 4;
            ptr0++;
            ptr1++;
            ptr2++;
            ptr3++;
        }

#undef SATURATE_CAST_UCHAR
        bgra += wgap;
    }
}

static int from_bgra2gray(const unsigned char* bgra, int w, int h, int stride, Mat& m, Allocator* allocator)
{
    // coeffs for r g b = 0.299f, 0.587f, 0.114f
    const unsigned char Y_shift = 8; //14
    const unsigned char R2Y = 77;
    const unsigned char G2Y = 150;
    const unsigned char B2Y = 29;

    m.create(w, h, 1, 4u, allocator);
    if (m.empty())
        return -100;

    const int wgap = stride - w * 4;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    float* ptr = m;

    for (int y = 0; y < h; y++)
    {
#if __ARM_NEON
        int nn = w >> 3;
        int remain = w - (nn << 3);
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
        uint8x8_t _R2Y = vdup_n_u8(R2Y);
        uint8x8_t _G2Y = vdup_n_u8(G2Y);
        uint8x8_t _B2Y = vdup_n_u8(B2Y);
        for (; nn > 0; nn--)
        {
            uint8x8x4_t _bgra = vld4_u8(bgra);

            uint16x8_t _y16 = vmull_u8(_bgra.val[2], _R2Y);
            _y16 = vmlal_u8(_y16, _bgra.val[1], _G2Y);
            _y16 = vmlal_u8(_y16, _bgra.val[0], _B2Y);
            _y16 = vshrq_n_u16(_y16, Y_shift);

            float32x4_t _ylow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_y16)));
            float32x4_t _yhigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_y16)));

            vst1q_f32(ptr, _ylow);
            vst1q_f32(ptr + 4, _yhigh);

            bgra += 4 * 8;
            ptr += 8;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "vdup.u8    d16, %6             \n"
                "vdup.u8    d17, %7             \n"
                "vdup.u8    d18, %8             \n"
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "vld4.u8    {d0-d3}, [%1]!      \n"
                "vmull.u8   q2, d2, d16         \n"
                "vmlal.u8   q2, d1, d17         \n"
                "vmlal.u8   q2, d0, d18         \n"
                "vshr.u16   q2, q2, #8          \n" // Y_shift
                "vmovl.u16  q0, d4              \n"
                "vmovl.u16  q1, d5              \n"
                "vcvt.f32.u32   q0, q0          \n"
                "vcvt.f32.u32   q1, q1          \n"
                "subs       %0, #1              \n"
                "vst1.f32   {d0-d3}, [%2]!      \n"
                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(bgra), // %1
                "=r"(ptr)   // %2
                : "0"(nn),
                "1"(bgra),
                "2"(ptr),
                "r"(R2Y), // %6
                "r"(G2Y), // %7
                "r"(B2Y)  // %8
                : "cc", "memory", "q0", "q1", "q2", "q8", "q9");
        }
#endif // __aarch64__
#endif // __ARM_NEON
        for (; remain > 0; remain--)
        {
            *ptr = static_cast<float>((bgra[2] * R2Y + bgra[1] * G2Y + bgra[0] * B2Y) >> Y_shift);

            bgra += 4;
            ptr++;
        }

        bgra += wgap;
    }

    return 0;
}

void yuv420sp2rgb(const unsigned char* yuv420sp, int w, int h, unsigned char* rgb)
{
    const unsigned char* yptr = yuv420sp;
    const unsigned char* vuptr = yuv420sp + w * h;

#if __ARM_NEON
    uint8x8_t _v128 = vdup_n_u8(128);
    int8x8_t _v90 = vdup_n_s8(90);
    int8x8_t _v46 = vdup_n_s8(46);
    int8x8_t _v22 = vdup_n_s8(22);
    int8x8_t _v113 = vdup_n_s8(113);
#endif // __ARM_NEON

    for (int y = 0; y < h; y += 2)
    {
        const unsigned char* yptr0 = yptr;
        const unsigned char* yptr1 = yptr + w;
        unsigned char* rgb0 = rgb;
        unsigned char* rgb1 = rgb + w * 3;

#if __ARM_NEON
        int nn = w >> 3;
        int remain = w - (nn << 3);
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
        for (; nn > 0; nn--)
        {
            int16x8_t _yy0 = vreinterpretq_s16_u16(vshll_n_u8(vld1_u8(yptr0), 6));
            int16x8_t _yy1 = vreinterpretq_s16_u16(vshll_n_u8(vld1_u8(yptr1), 6));

            int8x8_t _vvuu = vreinterpret_s8_u8(vsub_u8(vld1_u8(vuptr), _v128));
            int8x8x2_t _vvvvuuuu = vtrn_s8(_vvuu, _vvuu);
            int8x8_t _vv = _vvvvuuuu.val[0];
            int8x8_t _uu = _vvvvuuuu.val[1];

            int16x8_t _r0 = vmlal_s8(_yy0, _vv, _v90);
            int16x8_t _g0 = vmlsl_s8(_yy0, _vv, _v46);
            _g0 = vmlsl_s8(_g0, _uu, _v22);
            int16x8_t _b0 = vmlal_s8(_yy0, _uu, _v113);

            int16x8_t _r1 = vmlal_s8(_yy1, _vv, _v90);
            int16x8_t _g1 = vmlsl_s8(_yy1, _vv, _v46);
            _g1 = vmlsl_s8(_g1, _uu, _v22);
            int16x8_t _b1 = vmlal_s8(_yy1, _uu, _v113);

            uint8x8x3_t _rgb0;
            _rgb0.val[0] = vqshrun_n_s16(_r0, 6);
            _rgb0.val[1] = vqshrun_n_s16(_g0, 6);
            _rgb0.val[2] = vqshrun_n_s16(_b0, 6);

            uint8x8x3_t _rgb1;
            _rgb1.val[0] = vqshrun_n_s16(_r1, 6);
            _rgb1.val[1] = vqshrun_n_s16(_g1, 6);
            _rgb1.val[2] = vqshrun_n_s16(_b1, 6);

            vst3_u8(rgb0, _rgb0);
            vst3_u8(rgb1, _rgb1);

            yptr0 += 8;
            yptr1 += 8;
            vuptr += 8;
            rgb0 += 24;
            rgb1 += 24;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "pld        [%3, #128]          \n"
                "vld1.u8    {d2}, [%3]!         \n"
                "vsub.s8    d2, d2, %12         \n"
                "0:                             \n"
                "pld        [%1, #128]          \n"
                "vld1.u8    {d0}, [%1]!         \n"
                "pld        [%2, #128]          \n"
                "vld1.u8    {d1}, [%2]!         \n"
                "vshll.u8   q2, d0, #6          \n"
                "vorr       d3, d2, d2          \n"
                "vshll.u8   q3, d1, #6          \n"
                "vorr       q9, q2, q2          \n"
                "vtrn.s8    d2, d3              \n"
                "vorr       q11, q3, q3         \n"
                "vmlsl.s8   q9, d2, %14         \n"
                "vorr       q8, q2, q2          \n"
                "vmlsl.s8   q11, d2, %14        \n"
                "vorr       q10, q3, q3         \n"
                "vmlal.s8   q8, d2, %13         \n"
                "vmlal.s8   q2, d3, %16         \n"
                "vmlal.s8   q10, d2, %13        \n"
                "vmlsl.s8   q9, d3, %15         \n"
                "vmlal.s8   q3, d3, %16         \n"
                "vmlsl.s8   q11, d3, %15        \n"
                "vqshrun.s16 d24, q8, #6        \n"
                "vqshrun.s16 d26, q2, #6        \n"
                "vqshrun.s16 d4, q10, #6        \n"
                "vqshrun.s16 d25, q9, #6        \n"
                "vqshrun.s16 d6, q3, #6         \n"
                "vqshrun.s16 d5, q11, #6        \n"
                "pld        [%3, #128]          \n"
                "vld1.u8    {d2}, [%3]!         \n"
                "subs       %0, #1              \n"
                "vst3.u8    {d24-d26}, [%4]!    \n"
                "vsub.s8    d2, d2, %12         \n"
                "vst3.u8    {d4-d6}, [%5]!      \n"
                "bne        0b                  \n"
                "sub        %3, #8              \n"
                : "=r"(nn),    // %0
                "=r"(yptr0), // %1
                "=r"(yptr1), // %2
                "=r"(vuptr), // %3
                "=r"(rgb0),  // %4
                "=r"(rgb1)   // %5
                : "0"(nn),
                "1"(yptr0),
                "2"(yptr1),
                "3"(vuptr),
                "4"(rgb0),
                "5"(rgb1),
                "w"(_v128), // %12
                "w"(_v90),  // %13
                "w"(_v46),  // %14
                "w"(_v22),  // %15
                "w"(_v113)  // %16
                : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11", "q12", "d26");
        }
#endif // __aarch64__
#endif // __ARM_NEON

#define SATURATE_CAST_UCHAR(X) (unsigned char)::std::min(::std::max((int)(X), 0), 255);
        for (; remain > 0; remain -= 2)
        {
            // R = 1.164 * yy + 1.596 * vv
            // G = 1.164 * yy - 0.813 * vv - 0.391 * uu
            // B = 1.164 * yy              + 2.018 * uu

            // R = Y + (1.370705 * (V-128))
            // G = Y - (0.698001 * (V-128)) - (0.337633 * (U-128))
            // B = Y + (1.732446 * (U-128))

            // R = ((Y << 6) + 87.72512 * (V-128)) >> 6
            // G = ((Y << 6) - 44.672064 * (V-128) - 21.608512 * (U-128)) >> 6
            // B = ((Y << 6) + 110.876544 * (U-128)) >> 6

            // R = ((Y << 6) + 90 * (V-128)) >> 6
            // G = ((Y << 6) - 46 * (V-128) - 22 * (U-128)) >> 6
            // B = ((Y << 6) + 113 * (U-128)) >> 6

            // R = (yy + 90 * vv) >> 6
            // G = (yy - 46 * vv - 22 * uu) >> 6
            // B = (yy + 113 * uu) >> 6

            int v = vuptr[0] - 128;
            int u = vuptr[1] - 128;

            int ruv = 90 * v;
            int guv = -46 * v + -22 * u;
            int buv = 113 * u;

            int y00 = yptr0[0] << 6;
            rgb0[0] = SATURATE_CAST_UCHAR((y00 + ruv) >> 6);
            rgb0[1] = SATURATE_CAST_UCHAR((y00 + guv) >> 6);
            rgb0[2] = SATURATE_CAST_UCHAR((y00 + buv) >> 6);

            int y01 = yptr0[1] << 6;
            rgb0[3] = SATURATE_CAST_UCHAR((y01 + ruv) >> 6);
            rgb0[4] = SATURATE_CAST_UCHAR((y01 + guv) >> 6);
            rgb0[5] = SATURATE_CAST_UCHAR((y01 + buv) >> 6);

            int y10 = yptr1[0] << 6;
            rgb1[0] = SATURATE_CAST_UCHAR((y10 + ruv) >> 6);
            rgb1[1] = SATURATE_CAST_UCHAR((y10 + guv) >> 6);
            rgb1[2] = SATURATE_CAST_UCHAR((y10 + buv) >> 6);

            int y11 = yptr1[1] << 6;
            rgb1[3] = SATURATE_CAST_UCHAR((y11 + ruv) >> 6);
            rgb1[4] = SATURATE_CAST_UCHAR((y11 + guv) >> 6);
            rgb1[5] = SATURATE_CAST_UCHAR((y11 + buv) >> 6);

            yptr0 += 2;
            yptr1 += 2;
            vuptr += 2;
            rgb0 += 6;
            rgb1 += 6;
        }
#undef SATURATE_CAST_UCHAR

        yptr += 2 * w;
        rgb += 2 * 3 * w;
    }
}

void yuv420sp2rgb_nv12(const unsigned char* yuv420sp, int w, int h, unsigned char* rgb)
{
    const unsigned char* yptr = yuv420sp;
    const unsigned char* uvptr = yuv420sp + w * h;

#if __ARM_NEON
    uint8x8_t _v128 = vdup_n_u8(128);
    int8x8_t _v90 = vdup_n_s8(90);
    int8x8_t _v46 = vdup_n_s8(46);
    int8x8_t _v22 = vdup_n_s8(22);
    int8x8_t _v113 = vdup_n_s8(113);
#endif // __ARM_NEON

    for (int y = 0; y < h; y += 2)
    {
        const unsigned char* yptr0 = yptr;
        const unsigned char* yptr1 = yptr + w;
        unsigned char* rgb0 = rgb;
        unsigned char* rgb1 = rgb + w * 3;

#if __ARM_NEON
        int nn = w >> 3;
        int remain = w - (nn << 3);
#else
        int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
        for (; nn > 0; nn--)
        {
            int16x8_t _yy0 = vreinterpretq_s16_u16(vshll_n_u8(vld1_u8(yptr0), 6));
            int16x8_t _yy1 = vreinterpretq_s16_u16(vshll_n_u8(vld1_u8(yptr1), 6));

            int8x8_t _uuvv = vreinterpret_s8_u8(vsub_u8(vld1_u8(uvptr), _v128));
            int8x8x2_t _uuuuvvvv = vtrn_s8(_uuvv, _uuvv);
            int8x8_t _uu = _uuuuvvvv.val[0];
            int8x8_t _vv = _uuuuvvvv.val[1];

            int16x8_t _r0 = vmlal_s8(_yy0, _vv, _v90);
            int16x8_t _g0 = vmlsl_s8(_yy0, _vv, _v46);
            _g0 = vmlsl_s8(_g0, _uu, _v22);
            int16x8_t _b0 = vmlal_s8(_yy0, _uu, _v113);

            int16x8_t _r1 = vmlal_s8(_yy1, _vv, _v90);
            int16x8_t _g1 = vmlsl_s8(_yy1, _vv, _v46);
            _g1 = vmlsl_s8(_g1, _uu, _v22);
            int16x8_t _b1 = vmlal_s8(_yy1, _uu, _v113);

            uint8x8x3_t _rgb0;
            _rgb0.val[0] = vqshrun_n_s16(_r0, 6);
            _rgb0.val[1] = vqshrun_n_s16(_g0, 6);
            _rgb0.val[2] = vqshrun_n_s16(_b0, 6);

            uint8x8x3_t _rgb1;
            _rgb1.val[0] = vqshrun_n_s16(_r1, 6);
            _rgb1.val[1] = vqshrun_n_s16(_g1, 6);
            _rgb1.val[2] = vqshrun_n_s16(_b1, 6);

            vst3_u8(rgb0, _rgb0);
            vst3_u8(rgb1, _rgb1);

            yptr0 += 8;
            yptr1 += 8;
            uvptr += 8;
            rgb0 += 24;
            rgb1 += 24;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "pld        [%3, #128]          \n"
                "vld1.u8    {d2}, [%3]!         \n"
                "vsub.s8    d2, d2, %12         \n"
                "0:                             \n"
                "pld        [%1, #128]          \n"
                "vld1.u8    {d0}, [%1]!         \n"
                "pld        [%2, #128]          \n"
                "vld1.u8    {d1}, [%2]!         \n"
                "vshll.u8   q2, d0, #6          \n"
                "vorr       d3, d2, d2          \n"
                "vshll.u8   q3, d1, #6          \n"
                "vorr       q9, q2, q2          \n"
                "vtrn.s8    d2, d3              \n"
                "vorr       q11, q3, q3         \n"
                "vmlsl.s8   q9, d3, %14         \n"
                "vorr       q8, q2, q2          \n"
                "vmlsl.s8   q11, d3, %14        \n"
                "vorr       q10, q3, q3         \n"
                "vmlal.s8   q8, d3, %13         \n"
                "vmlal.s8   q2, d2, %16         \n"
                "vmlal.s8   q10, d3, %13        \n"
                "vmlsl.s8   q9, d2, %15         \n"
                "vmlal.s8   q3, d2, %16         \n"
                "vmlsl.s8   q11, d2, %15        \n"
                "vqshrun.s16 d24, q8, #6        \n"
                "vqshrun.s16 d26, q2, #6        \n"
                "vqshrun.s16 d4, q10, #6        \n"
                "vqshrun.s16 d25, q9, #6        \n"
                "vqshrun.s16 d6, q3, #6         \n"
                "vqshrun.s16 d5, q11, #6        \n"
                "pld        [%3, #128]          \n"
                "vld1.u8    {d2}, [%3]!         \n"
                "subs       %0, #1              \n"
                "vst3.u8    {d24-d26}, [%4]!    \n"
                "vsub.s8    d2, d2, %12         \n"
                "vst3.u8    {d4-d6}, [%5]!      \n"
                "bne        0b                  \n"
                "sub        %3, #8              \n"
                : "=r"(nn),    // %0
                "=r"(yptr0), // %1
                "=r"(yptr1), // %2
                "=r"(uvptr), // %3
                "=r"(rgb0),  // %4
                "=r"(rgb1)   // %5
                : "0"(nn),
                "1"(yptr0),
                "2"(yptr1),
                "3"(uvptr),
                "4"(rgb0),
                "5"(rgb1),
                "w"(_v128), // %12
                "w"(_v90),  // %13
                "w"(_v46),  // %14
                "w"(_v22),  // %15
                "w"(_v113)  // %16
                : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11", "q12", "d26");
        }
#endif // __aarch64__
#endif // __ARM_NEON

#define SATURATE_CAST_UCHAR(X) (unsigned char)::std::min(::std::max((int)(X), 0), 255);
        for (; remain > 0; remain -= 2)
        {
            // R = 1.164 * yy + 1.596 * vv
            // G = 1.164 * yy - 0.813 * vv - 0.391 * uu
            // B = 1.164 * yy              + 2.018 * uu

            // R = Y + (1.370705 * (V-128))
            // G = Y - (0.698001 * (V-128)) - (0.337633 * (U-128))
            // B = Y + (1.732446 * (U-128))

            // R = ((Y << 6) + 87.72512 * (V-128)) >> 6
            // G = ((Y << 6) - 44.672064 * (V-128) - 21.608512 * (U-128)) >> 6
            // B = ((Y << 6) + 110.876544 * (U-128)) >> 6

            // R = ((Y << 6) + 90 * (V-128)) >> 6
            // G = ((Y << 6) - 46 * (V-128) - 22 * (U-128)) >> 6
            // B = ((Y << 6) + 113 * (U-128)) >> 6

            // R = (yy + 90 * vv) >> 6
            // G = (yy - 46 * vv - 22 * uu) >> 6
            // B = (yy + 113 * uu) >> 6

            int u = uvptr[0] - 128;
            int v = uvptr[1] - 128;

            int ruv = 90 * v;
            int guv = -46 * v + -22 * u;
            int buv = 113 * u;

            int y00 = yptr0[0] << 6;
            rgb0[0] = SATURATE_CAST_UCHAR((y00 + ruv) >> 6);
            rgb0[1] = SATURATE_CAST_UCHAR((y00 + guv) >> 6);
            rgb0[2] = SATURATE_CAST_UCHAR((y00 + buv) >> 6);

            int y01 = yptr0[1] << 6;
            rgb0[3] = SATURATE_CAST_UCHAR((y01 + ruv) >> 6);
            rgb0[4] = SATURATE_CAST_UCHAR((y01 + guv) >> 6);
            rgb0[5] = SATURATE_CAST_UCHAR((y01 + buv) >> 6);

            int y10 = yptr1[0] << 6;
            rgb1[0] = SATURATE_CAST_UCHAR((y10 + ruv) >> 6);
            rgb1[1] = SATURATE_CAST_UCHAR((y10 + guv) >> 6);
            rgb1[2] = SATURATE_CAST_UCHAR((y10 + buv) >> 6);

            int y11 = yptr1[1] << 6;
            rgb1[3] = SATURATE_CAST_UCHAR((y11 + ruv) >> 6);
            rgb1[4] = SATURATE_CAST_UCHAR((y11 + guv) >> 6);
            rgb1[5] = SATURATE_CAST_UCHAR((y11 + buv) >> 6);

            yptr0 += 2;
            yptr1 += 2;
            uvptr += 2;
            rgb0 += 6;
            rgb1 += 6;
        }
#undef SATURATE_CAST_UCHAR

        yptr += 2 * w;
        rgb += 2 * 3 * w;
    }
}

void yuv420sp2rgb_half(const unsigned char* yuv, int w, int h, unsigned char* rgb)
{
    const unsigned char* puv = yuv + w * h;
    const unsigned char *py0 = yuv, *py1 = yuv + w;
    const int hstep = h / 2;
#if __ARM_NEON
    const int wstep = w / 16, tailstep = (w - wstep * 16) / 2;
    uint8x8_t _u128 = vdup_n_u8(128);
    int8x8_t _s90 = vdup_n_s8(90);
    int8x8_t _sn46 = vdup_n_s8(-46);
    int8x8_t _s113 = vdup_n_s8(113);
    int8x8_t _sn22 = vdup_n_s8(-22);
    int16x8_t _s0 = vdupq_n_s16(0);
    int16x8_t _s16320 = vdupq_n_s16(16320); // 255 << 6
#else
    const int tailstep = w / 2;
#endif

    for (int i = 0; i < hstep; ++i)
    {
#if __ARM_NEON
        for (int j = 0; j < wstep; ++j)
        {
            uint8x16_t y0 = vld1q_u8(py0);
            uint8x16_t y1 = vld1q_u8(py1);

            // first 8 Y
            uint16x8_t low = vaddl_u8(vget_low_u8(y0), vget_low_u8(y1));
            uint16x4_t low_sum = vpadd_u16(vget_low_u16(low), vget_high_u16(low));

            // last 8 Y
            uint16x8_t high = vaddl_u8(vget_high_u8(y0), vget_high_u8(y1));
            uint16x4_t high_sum = vpadd_u16(vget_low_u16(high), vget_high_u16(high));

            uint16x8_t y8_sum = vcombine_u16(low_sum, high_sum);
            // y8 = (y8_sum >> 2) << 6 = y8_sum << 4;
            int16x8_t y8 = vreinterpretq_s16_u16(vshlq_n_u16(y8_sum, 4));

            // prepare uv
            uint8x8x2_t vu = vld2_u8(puv);
            int8x8_t v = vreinterpret_s8_u8(vsub_u8(vu.val[0], _u128));
            int8x8_t u = vreinterpret_s8_u8(vsub_u8(vu.val[1], _u128));

            int16x8_t r_acc = vmlal_s8(y8, v, _s90);
            int16x8_t g_acc = vmlal_s8(y8, v, _sn46);
            g_acc = vmlal_s8(g_acc, u, _sn22);
            int16x8_t b_acc = vmlal_s8(y8, u, _s113);

#define SHIFT_6_SATURATE(FROM, TO)                     \
    FROM = vmaxq_s16(vminq_s16((FROM), _s16320), _s0); \
    uint8x8_t TO = vshrn_n_u16(vreinterpretq_u16_s16((FROM)), 6);

            SHIFT_6_SATURATE(b_acc, b_out)
            SHIFT_6_SATURATE(g_acc, g_out)
            SHIFT_6_SATURATE(r_acc, r_out)
#undef SHIFT_6_SATURATE

            uint8x8x3_t _rgb;
            _rgb.val[0] = r_out;
            _rgb.val[1] = g_out;
            _rgb.val[2] = b_out;
            vst3_u8(rgb, _rgb);

            rgb += 24;
            py0 += 16;
            py1 += 16;
            puv += 16;
        }
#endif

        for (int idx = 0; idx < tailstep; ++idx)
        {
            int y = (static_cast<int>(py0[0]) + py0[1] + py1[2] + py1[1]) << 4;
            int v = static_cast<int>(puv[0]) - 128;
            int u = static_cast<int>(puv[1]) - 128;

            int ruv = 90 * v;
            int guv = -46 * v + -22 * u;
            int buv = 113 * u;

#define SATURATE_CAST_UCHAR(X) (unsigned char)::std::min(::std::max((int)(X), 0), 255);
            rgb[0] = SATURATE_CAST_UCHAR((y + ruv) >> 6);
            rgb[1] = SATURATE_CAST_UCHAR((y + guv) >> 6);
            rgb[2] = SATURATE_CAST_UCHAR((y + buv) >> 6);
#undef SATURATE_CAST_UCHAR

            rgb += 3;
            py0 += 2;
            py1 += 2;
            puv += 2;
        }
        // next two row
        py0 = py1;
        py1 = py0 + w;
    }
}

Mat Mat::from_pixels(const unsigned char* pixels, int type, int w, int h, Allocator* allocator)
{
    int type_from = type & PIXEL_FORMAT_MASK;

    if (type_from == PIXEL_RGB || type_from == PIXEL_BGR)
    {
        return Mat::from_pixels(pixels, type, w, h, w * 3, allocator);
    }
    else if (type_from == PIXEL_GRAY)
    {
        return Mat::from_pixels(pixels, type, w, h, w * 1, allocator);
    }
    else if (type_from == PIXEL_RGBA || type_from == PIXEL_BGRA)
    {
        return Mat::from_pixels(pixels, type, w, h, w * 4, allocator);
    }

    // unknown convert type
    NCNN_LOGE("unknown convert type %d", type);
    return Mat();
}

Mat Mat::from_pixels(const unsigned char* pixels, int type, int w, int h, int stride, Allocator* allocator)
{
    Mat m;

    if (type & PIXEL_CONVERT_MASK)
    {
        switch (type)
        {
        case PIXEL_RGB2BGR:
        case PIXEL_BGR2RGB:
            from_rgb2bgr(pixels, w, h, stride, m, allocator);
            break;
        case PIXEL_RGB2GRAY:
            from_rgb2gray(pixels, w, h, stride, m, allocator);
            break;
        case PIXEL_RGB2RGBA:
        case PIXEL_BGR2BGRA:
            from_rgb2rgba(pixels, w, h, stride, m, allocator);
            break;
        case PIXEL_BGR2GRAY:
            from_bgr2gray(pixels, w, h, stride, m, allocator);
            break;
        case PIXEL_BGR2RGBA:
        case PIXEL_RGB2BGRA:
            from_bgr2rgba(pixels, w, h, stride, m, allocator);
            break;
        case PIXEL_GRAY2RGB:
        case PIXEL_GRAY2BGR:
            from_gray2rgb(pixels, w, h, stride, m, allocator);
            break;
        case PIXEL_GRAY2RGBA:
        case PIXEL_GRAY2BGRA:
            from_gray2rgba(pixels, w, h, stride, m, allocator);
            break;
        case PIXEL_RGBA2RGB:
        case PIXEL_BGRA2BGR:
            from_rgba2rgb(pixels, w, h, stride, m, allocator);
            break;
        case PIXEL_RGBA2BGR:
        case PIXEL_BGRA2RGB:
            from_rgba2bgr(pixels, w, h, stride, m, allocator);
            break;
        case PIXEL_RGBA2GRAY:
            from_rgba2gray(pixels, w, h, stride, m, allocator);
            break;
        case PIXEL_RGBA2BGRA:
        case PIXEL_BGRA2RGBA:
            from_rgba2bgra(pixels, w, h, stride, m, allocator);
            break;
        case PIXEL_BGRA2GRAY:
            from_bgra2gray(pixels, w, h, stride, m, allocator);
            break;
        default:
            // unimplemented convert type
            NCNN_LOGE("unimplemented convert type %d", type);
            break;
        }
    }
    else
    {
        if (type == PIXEL_RGB || type == PIXEL_BGR)
            from_rgb(pixels, w, h, stride, m, allocator);

        if (type == PIXEL_GRAY)
            from_gray(pixels, w, h, stride, m, allocator);

        if (type == PIXEL_RGBA || type == PIXEL_BGRA)
            from_rgba(pixels, w, h, stride, m, allocator);
    }

    return m;
}

Mat Mat::from_pixels_resize(const unsigned char* pixels, int type, int w, int h, int target_width, int target_height, Allocator* allocator)
{
    int type_from = type & PIXEL_FORMAT_MASK;

    if (type_from == PIXEL_RGB || type_from == PIXEL_BGR)
    {
        return Mat::from_pixels_resize(pixels, type, w, h, w * 3, target_width, target_height, allocator);
    }
    else if (type_from == PIXEL_GRAY)
    {
        return Mat::from_pixels_resize(pixels, type, w, h, w * 1, target_width, target_height, allocator);
    }
    else if (type_from == PIXEL_RGBA || type_from == PIXEL_BGRA)
    {
        return Mat::from_pixels_resize(pixels, type, w, h, w * 4, target_width, target_height, allocator);
    }

    // unknown convert type
    NCNN_LOGE("unknown convert type %d", type);
    return Mat();
}

Mat Mat::from_pixels_resize(const unsigned char* pixels, int type, int w, int h, int stride, int target_width, int target_height, Allocator* allocator)
{
    if (w == target_width && h == target_height)
        return Mat::from_pixels(pixels, type, w, h, stride, allocator);

    int type_from = type & PIXEL_FORMAT_MASK;

    if (type_from == PIXEL_RGB || type_from == PIXEL_BGR)
    {
        Mat dst(target_width, target_height, (size_t)3u, 3);
        resize_bilinear_c3(pixels, w, h, stride, dst, target_width, target_height, target_width * 3);

        return Mat::from_pixels(dst, type, target_width, target_height, allocator);
    }
    else if (type_from == PIXEL_GRAY)
    {
        Mat dst(target_width, target_height, (size_t)1u, 1);
        resize_bilinear_c1(pixels, w, h, stride, dst, target_width, target_height, target_width * 1);

        return Mat::from_pixels(dst, type, target_width, target_height, allocator);
    }
    else if (type_from == PIXEL_RGBA || type_from == PIXEL_BGRA)
    {
        Mat dst(target_width, target_height, (size_t)4u, 4);
        resize_bilinear_c4(pixels, w, h, stride, dst, target_width, target_height, target_width * 4);

        return Mat::from_pixels(dst, type, target_width, target_height, allocator);
    }

    // unknown convert type
    NCNN_LOGE("unknown convert type %d", type);
    return Mat();
}

Mat Mat::from_pixels_roi(const unsigned char* pixels, int type, int w, int h, int roix, int roiy, int roiw, int roih, Allocator* allocator)
{
    if (roix < 0 || roiy < 0 || roiw <= 0 || roih <= 0 || roix + roiw > w || roiy + roih > h)
    {
        NCNN_LOGE("roi %d %d %d %d out of image %d %d", roix, roiy, roiw, roih, w, h);
        return Mat();
    }

    int type_from = type & PIXEL_FORMAT_MASK;

    if (type_from == PIXEL_RGB || type_from == PIXEL_BGR)
    {
        return from_pixels(pixels + (roiy * w + roix) * 3, type, roiw, roih, w * 3, allocator);
    }
    else if (type_from == PIXEL_GRAY)
    {
        return from_pixels(pixels + (roiy * w + roix) * 1, type, roiw, roih, w * 1, allocator);
    }
    else if (type_from == PIXEL_RGBA || type_from == PIXEL_BGRA)
    {
        return from_pixels(pixels + (roiy * w + roix) * 4, type, roiw, roih, w * 4, allocator);
    }

    // unknown convert type
    NCNN_LOGE("unknown convert type %d", type);
    return Mat();
}

Mat Mat::from_pixels_roi(const unsigned char* pixels, int type, int w, int h, int stride, int roix, int roiy, int roiw, int roih, Allocator* allocator)
{
    if (roix < 0 || roiy < 0 || roiw <= 0 || roih <= 0 || roix + roiw > w || roiy + roih > h)
    {
        NCNN_LOGE("roi %d %d %d %d out of image %d %d", roix, roiy, roiw, roih, w, h);
        return Mat();
    }

    int type_from = type & PIXEL_FORMAT_MASK;

    if (type_from == PIXEL_RGB || type_from == PIXEL_BGR)
    {
        return from_pixels(pixels + roiy * stride + roix * 3, type, roiw, roih, stride, allocator);
    }
    else if (type_from == PIXEL_GRAY)
    {
        return from_pixels(pixels + roiy * stride + roix * 1, type, roiw, roih, stride, allocator);
    }
    else if (type_from == PIXEL_RGBA || type_from == PIXEL_BGRA)
    {
        return from_pixels(pixels + roiy * stride + roix * 4, type, roiw, roih, stride, allocator);
    }

    // unknown convert type
    NCNN_LOGE("unknown convert type %d", type);
    return Mat();
}

Mat Mat::from_pixels_roi_resize(const unsigned char* pixels, int type, int w, int h, int roix, int roiy, int roiw, int roih, int target_width, int target_height, Allocator* allocator)
{
    if (roix < 0 || roiy < 0 || roiw <= 0 || roih <= 0 || roix + roiw > w || roiy + roih > h)
    {
        NCNN_LOGE("roi %d %d %d %d out of image %d %d", roix, roiy, roiw, roih, w, h);
        return Mat();
    }

    int type_from = type & PIXEL_FORMAT_MASK;

    if (type_from == PIXEL_RGB || type_from == PIXEL_BGR)
    {
        return from_pixels_resize(pixels + (roiy * w + roix) * 3, type, roiw, roih, w * 3, target_width, target_height, allocator);
    }
    else if (type_from == PIXEL_GRAY)
    {
        return from_pixels_resize(pixels + (roiy * w + roix) * 1, type, roiw, roih, w * 1, target_width, target_height, allocator);
    }
    else if (type_from == PIXEL_RGBA || type_from == PIXEL_BGRA)
    {
        return from_pixels_resize(pixels + (roiy * w + roix) * 4, type, roiw, roih, w * 4, target_width, target_height, allocator);
    }

    // unknown convert type
    NCNN_LOGE("unknown convert type %d", type);
    return Mat();
}

Mat Mat::from_pixels_roi_resize(const unsigned char* pixels, int type, int w, int h, int stride, int roix, int roiy, int roiw, int roih, int target_width, int target_height, Allocator* allocator)
{
    if (roix < 0 || roiy < 0 || roiw <= 0 || roih <= 0 || roix + roiw > w || roiy + roih > h)
    {
        NCNN_LOGE("roi %d %d %d %d out of image %d %d", roix, roiy, roiw, roih, w, h);
        return Mat();
    }

    int type_from = type & PIXEL_FORMAT_MASK;

    if (type_from == PIXEL_RGB || type_from == PIXEL_BGR)
    {
        return from_pixels_resize(pixels + roiy * stride + roix * 3, type, roiw, roih, stride, target_width, target_height, allocator);
    }
    else if (type_from == PIXEL_GRAY)
    {
        return from_pixels_resize(pixels + roiy * stride + roix * 1, type, roiw, roih, stride, target_width, target_height, allocator);
    }
    else if (type_from == PIXEL_RGBA || type_from == PIXEL_BGRA)
    {
        return from_pixels_resize(pixels + roiy * stride + roix * 4, type, roiw, roih, stride, target_width, target_height, allocator);
    }

    // unknown convert type
    NCNN_LOGE("unknown convert type %d", type);
    return Mat();
}

void Mat::to_pixels(unsigned char* pixels, int type) const
{
    int type_to = (type & PIXEL_CONVERT_MASK) ? (type >> PIXEL_CONVERT_SHIFT) : (type & PIXEL_FORMAT_MASK);

    if (type_to == PIXEL_RGB || type_to == PIXEL_BGR)
    {
        to_pixels(pixels, type, w * 3);
    }
    else if (type_to == PIXEL_GRAY)
    {
        to_pixels(pixels, type, w * 1);
    }
    else if (type_to == PIXEL_RGBA || type_to == PIXEL_BGRA)
    {
        to_pixels(pixels, type, w * 4);
    }
}

void Mat::to_pixels(unsigned char* pixels, int type, int stride) const
{
    if (type & PIXEL_CONVERT_MASK)
    {
        switch (type)
        {
        case PIXEL_RGB2BGR:
        case PIXEL_BGR2RGB:
            to_bgr2rgb(*this, pixels, stride);
            break;
        case PIXEL_RGB2RGBA:
        case PIXEL_BGR2BGRA:
            to_rgb2rgba(*this, pixels, stride);
            break;
        case PIXEL_BGR2RGBA:
        case PIXEL_RGB2BGRA:
            to_bgr2rgba(*this, pixels, stride);
            break;
        case PIXEL_GRAY2RGBA:
        case PIXEL_GRAY2BGRA:
            to_gray2rgba(*this, pixels, stride);
            break;
        case PIXEL_RGBA2BGRA:
        case PIXEL_BGRA2RGBA:
            to_rgba2bgra(*this, pixels, stride);
            break;
        default:
            // unimplemented convert type
            NCNN_LOGE("unimplemented convert type %d", type);
            break;
        }
    }
    else
    {
        if (type == PIXEL_RGB || type == PIXEL_BGR)
            to_rgb(*this, pixels, stride);

        if (type == PIXEL_GRAY)
            to_gray(*this, pixels, stride);

        if (type == PIXEL_RGBA || type == PIXEL_BGRA)
            to_rgba(*this, pixels, stride);
    }
}

void Mat::to_pixels_resize(unsigned char* pixels, int type, int target_width, int target_height) const
{
    int type_to = (type & PIXEL_CONVERT_MASK) ? (type >> PIXEL_CONVERT_SHIFT) : (type & PIXEL_FORMAT_MASK);

    if (type_to == PIXEL_RGB || type_to == PIXEL_BGR)
    {
        to_pixels_resize(pixels, type, target_width, target_height, target_width * 3);
    }
    else if (type_to == PIXEL_GRAY)
    {
        to_pixels_resize(pixels, type, target_width, target_height, target_width * 1);
    }
    else if (type_to == PIXEL_RGBA || type_to == PIXEL_BGRA)
    {
        to_pixels_resize(pixels, type, target_width, target_height, target_width * 4);
    }
}

void Mat::to_pixels_resize(unsigned char* pixels, int type, int target_width, int target_height, int target_stride) const
{
    if (w == target_width && h == target_height)
        return to_pixels(pixels, type);

    int type_to = (type & PIXEL_CONVERT_MASK) ? (type >> PIXEL_CONVERT_SHIFT) : (type & PIXEL_FORMAT_MASK);

    if (type_to == PIXEL_RGB || type_to == PIXEL_BGR)
    {
        Mat src(w, h, (size_t)3u, 3);

        to_pixels(src, type);

        resize_bilinear_c3(src, w, h, w * 3, pixels, target_width, target_height, target_stride);
    }
    else if (type_to == PIXEL_GRAY)
    {
        Mat src(w, h, (size_t)1u, 1);

        to_pixels(src, type);

        resize_bilinear_c1(src, w, h, w * 1, pixels, target_width, target_height, target_stride);
    }
    else if (type_to == PIXEL_RGBA || type_to == PIXEL_BGRA)
    {
        Mat src(w, h, (size_t)4u, 4);

        to_pixels(src, type);

        resize_bilinear_c4(src, w, h, w * 4, pixels, target_width, target_height, target_stride);
    }
}
#endif // NCNN_PIXEL

} // namespace ncnn
