// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
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

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON
#include "platform.h"

namespace ncnn {

#if NCNN_PIXEL
static void vresize_two(const short* rows0p, const short* rows1p, int wsize, unsigned char* Dp0, unsigned char* Dp1, short b0, short b1, short b2, short b3)
{
    int dx = 0;
#if __ARM_NEON
    int16x8_t _b0 = vdupq_n_s16(b0);
    int16x8_t _b1 = vdupq_n_s16(b1);
    int16x8_t _b2 = vdupq_n_s16(b2);
    int16x8_t _b3 = vdupq_n_s16(b3);
    for (; dx + 15 < wsize; dx += 16)
    {
        int16x8_t _r00 = vld1q_s16(rows0p);
        int16x8_t _r01 = vld1q_s16(rows0p + 8);
        int16x8_t _r10 = vld1q_s16(rows1p);
        int16x8_t _r11 = vld1q_s16(rows1p + 8);
        int16x8_t _acc00 = vaddq_s16(vqdmulhq_s16(_r00, _b0), vqdmulhq_s16(_r10, _b1));
        int16x8_t _acc01 = vaddq_s16(vqdmulhq_s16(_r01, _b0), vqdmulhq_s16(_r11, _b1));
        int16x8_t _acc10 = vaddq_s16(vqdmulhq_s16(_r00, _b2), vqdmulhq_s16(_r10, _b3));
        int16x8_t _acc11 = vaddq_s16(vqdmulhq_s16(_r01, _b2), vqdmulhq_s16(_r11, _b3));
        uint8x16_t _Dp0 = vcombine_u8(vqrshrun_n_s16(_acc00, 3), vqrshrun_n_s16(_acc01, 3));
        uint8x16_t _Dp1 = vcombine_u8(vqrshrun_n_s16(_acc10, 3), vqrshrun_n_s16(_acc11, 3));
        vst1q_u8(Dp0, _Dp0);
        vst1q_u8(Dp1, _Dp1);
        Dp0 += 16;
        Dp1 += 16;
        rows0p += 16;
        rows1p += 16;
    }
    for (; dx + 7 < wsize; dx += 8)
    {
        int16x8_t _r0 = vld1q_s16(rows0p);
        int16x8_t _r1 = vld1q_s16(rows1p);
        int16x8_t _acc0 = vaddq_s16(vqdmulhq_s16(_r0, _b0), vqdmulhq_s16(_r1, _b1));
        int16x8_t _acc1 = vaddq_s16(vqdmulhq_s16(_r0, _b2), vqdmulhq_s16(_r1, _b3));
        uint8x8_t _Dp0 = vqrshrun_n_s16(_acc0, 3);
        uint8x8_t _Dp1 = vqrshrun_n_s16(_acc1, 3);
        vst1_u8(Dp0, _Dp0);
        vst1_u8(Dp1, _Dp1);
        Dp0 += 8;
        Dp1 += 8;
        rows0p += 8;
        rows1p += 8;
    }
#endif // __ARM_NEON
#if __SSE2__
    __m128i _b0 = _mm_set1_epi16(b0);
    __m128i _b1 = _mm_set1_epi16(b1);
    __m128i _b2 = _mm_set1_epi16(b2);
    __m128i _b3 = _mm_set1_epi16(b3);
    __m128i _v2 = _mm_set1_epi16(2);
    for (; dx + 15 < wsize; dx += 16)
    {
        __m128i _r00 = _mm_loadu_si128((const __m128i*)rows0p);
        __m128i _r01 = _mm_loadu_si128((const __m128i*)(rows0p + 8));
        __m128i _r10 = _mm_loadu_si128((const __m128i*)rows1p);
        __m128i _r11 = _mm_loadu_si128((const __m128i*)(rows1p + 8));
        __m128i _acc00 = _mm_add_epi16(_mm_mulhi_epi16(_r00, _b0), _mm_mulhi_epi16(_r10, _b1));
        __m128i _acc01 = _mm_add_epi16(_mm_mulhi_epi16(_r01, _b0), _mm_mulhi_epi16(_r11, _b1));
        __m128i _acc10 = _mm_add_epi16(_mm_mulhi_epi16(_r00, _b2), _mm_mulhi_epi16(_r10, _b3));
        __m128i _acc11 = _mm_add_epi16(_mm_mulhi_epi16(_r01, _b2), _mm_mulhi_epi16(_r11, _b3));
        _acc00 = _mm_srai_epi16(_mm_add_epi16(_acc00, _v2), 2);
        _acc01 = _mm_srai_epi16(_mm_add_epi16(_acc01, _v2), 2);
        _acc10 = _mm_srai_epi16(_mm_add_epi16(_acc10, _v2), 2);
        _acc11 = _mm_srai_epi16(_mm_add_epi16(_acc11, _v2), 2);
        __m128i _Dp0 = _mm_packus_epi16(_acc00, _acc01);
        __m128i _Dp1 = _mm_packus_epi16(_acc10, _acc11);
        _mm_storeu_si128((__m128i*)Dp0, _Dp0);
        _mm_storeu_si128((__m128i*)Dp1, _Dp1);
        Dp0 += 16;
        Dp1 += 16;
        rows0p += 16;
        rows1p += 16;
    }
    for (; dx + 7 < wsize; dx += 8)
    {
        __m128i _r0 = _mm_loadu_si128((const __m128i*)rows0p);
        __m128i _r1 = _mm_loadu_si128((const __m128i*)rows1p);
        __m128i _acc0 = _mm_add_epi16(_mm_mulhi_epi16(_r0, _b0), _mm_mulhi_epi16(_r1, _b1));
        __m128i _acc1 = _mm_add_epi16(_mm_mulhi_epi16(_r0, _b2), _mm_mulhi_epi16(_r1, _b3));
        _acc0 = _mm_srai_epi16(_mm_add_epi16(_acc0, _v2), 2);
        _acc1 = _mm_srai_epi16(_mm_add_epi16(_acc1, _v2), 2);
        __m128i _Dp0 = _mm_packus_epi16(_acc0, _acc0);
        __m128i _Dp1 = _mm_packus_epi16(_acc1, _acc1);
        _mm_storel_epi64((__m128i*)Dp0, _Dp0);
        _mm_storel_epi64((__m128i*)Dp1, _Dp1);
        Dp0 += 8;
        Dp1 += 8;
        rows0p += 8;
        rows1p += 8;
    }
#endif // __SSE2__
    for (; dx < wsize; dx++)
    {
        short s0 = *rows0p++;
        short s1 = *rows1p++;

        *Dp0++ = (unsigned char)(((short)((b0 * s0) >> 16) + (short)((b1 * s1) >> 16) + 2) >> 2);
        *Dp1++ = (unsigned char)(((short)((b2 * s0) >> 16) + (short)((b3 * s1) >> 16) + 2) >> 2);
    }
}

static void vresize_one(const short* rows0p, const short* rows1p, int wsize, unsigned char* Dp, short b0, short b1)
{
    int dx = 0;
#if __ARM_NEON
    int16x8_t _b0 = vdupq_n_s16(b0);
    int16x8_t _b1 = vdupq_n_s16(b1);
    for (; dx + 15 < wsize; dx += 16)
    {
        int16x8_t _r00 = vld1q_s16(rows0p);
        int16x8_t _r01 = vld1q_s16(rows0p + 8);
        int16x8_t _r10 = vld1q_s16(rows1p);
        int16x8_t _r11 = vld1q_s16(rows1p + 8);
        int16x8_t _acc0 = vaddq_s16(vqdmulhq_s16(_r00, _b0), vqdmulhq_s16(_r10, _b1));
        int16x8_t _acc1 = vaddq_s16(vqdmulhq_s16(_r01, _b0), vqdmulhq_s16(_r11, _b1));
        uint8x16_t _Dp = vcombine_u8(vqrshrun_n_s16(_acc0, 3), vqrshrun_n_s16(_acc1, 3));
        vst1q_u8(Dp, _Dp);
        Dp += 16;
        rows0p += 16;
        rows1p += 16;
    }
    for (; dx + 7 < wsize; dx += 8)
    {
        int16x8_t _r0 = vld1q_s16(rows0p);
        int16x8_t _r1 = vld1q_s16(rows1p);
        int16x8_t _acc = vaddq_s16(vqdmulhq_s16(_r0, _b0), vqdmulhq_s16(_r1, _b1));
        uint8x8_t _Dp = vqrshrun_n_s16(_acc, 3);
        vst1_u8(Dp, _Dp);
        Dp += 8;
        rows0p += 8;
        rows1p += 8;
    }
#endif // __ARM_NEON
#if __SSE2__
    __m128i _b0 = _mm_set1_epi16(b0);
    __m128i _b1 = _mm_set1_epi16(b1);
    __m128i _v2 = _mm_set1_epi16(2);
    for (; dx + 15 < wsize; dx += 16)
    {
        __m128i _r00 = _mm_loadu_si128((const __m128i*)rows0p);
        __m128i _r01 = _mm_loadu_si128((const __m128i*)(rows0p + 8));
        __m128i _r10 = _mm_loadu_si128((const __m128i*)rows1p);
        __m128i _r11 = _mm_loadu_si128((const __m128i*)(rows1p + 8));
        __m128i _acc0 = _mm_add_epi16(_mm_mulhi_epi16(_r00, _b0), _mm_mulhi_epi16(_r10, _b1));
        __m128i _acc1 = _mm_add_epi16(_mm_mulhi_epi16(_r01, _b0), _mm_mulhi_epi16(_r11, _b1));
        _acc0 = _mm_srai_epi16(_mm_add_epi16(_acc0, _v2), 2);
        _acc1 = _mm_srai_epi16(_mm_add_epi16(_acc1, _v2), 2);
        __m128i _Dp = _mm_packus_epi16(_acc0, _acc1);
        _mm_storeu_si128((__m128i*)Dp, _Dp);
        Dp += 16;
        rows0p += 16;
        rows1p += 16;
    }
    for (; dx + 7 < wsize; dx += 8)
    {
        __m128i _r0 = _mm_loadu_si128((const __m128i*)rows0p);
        __m128i _r1 = _mm_loadu_si128((const __m128i*)rows1p);
        __m128i _acc = _mm_add_epi16(_mm_mulhi_epi16(_r0, _b0), _mm_mulhi_epi16(_r1, _b1));
        _acc = _mm_srai_epi16(_mm_add_epi16(_acc, _v2), 2);
        __m128i _Dp = _mm_packus_epi16(_acc, _acc);
        _mm_storel_epi64((__m128i*)Dp, _Dp);
        Dp += 8;
        rows0p += 8;
        rows1p += 8;
    }
#endif // __SSE2__
    for (; dx < wsize; dx++)
    {
        short s0 = *rows0p++;
        short s1 = *rows1p++;

        *Dp++ = (unsigned char)(((short)((b0 * s0) >> 16) + (short)((b1 * s1) >> 16) + 2) >> 2);
    }
}

void resize_bilinear_c1(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h)
{
    return resize_bilinear_c1(src, srcw, srch, srcw, dst, w, h, w);
}

void resize_bilinear_c2(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h)
{
    return resize_bilinear_c2(src, srcw, srch, srcw * 2, dst, w, h, w * 2);
}

void resize_bilinear_c3(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h)
{
    return resize_bilinear_c3(src, srcw, srch, srcw * 3, dst, w, h, w * 3);
}

void resize_bilinear_c4(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h)
{
    return resize_bilinear_c4(src, srcw, srch, srcw * 4, dst, w, h, w * 4);
}

void resize_bilinear_c1(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    const int INTER_RESIZE_COEF_BITS = 11;
    const int INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS;

    double scale_x = (double)srcw / w;
    double scale_y = (double)srch / h;

    int* buf = new int[w + h + w + h];

    int* xofs = buf;     //new int[w];
    int* yofs = buf + w; //new int[h];

    short* ialpha = (short*)(buf + w + h);    //new short[w * 2];
    short* ibeta = (short*)(buf + w + h + w); //new short[h * 2];

    float fx;
    float fy;
    int sx;
    int sy;

#define SATURATE_CAST_SHORT(X) (short)::std::min(::std::max((int)(X + (X >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), SHRT_MAX);

    for (int dx = 0; dx < w; dx++)
    {
        fx = (float)((dx + 0.5) * scale_x - 0.5);
        sx = static_cast<int>(floor(fx));
        fx -= sx;

        if (sx < 0)
        {
            sx = 0;
            fx = 0.f;
        }
        if (sx >= srcw - 1)
        {
            sx = srcw - 2;
            fx = 1.f;
        }

        xofs[dx] = sx;

        float a0 = (1.f - fx) * INTER_RESIZE_COEF_SCALE;
        float a1 = fx * INTER_RESIZE_COEF_SCALE;

        ialpha[dx * 2] = SATURATE_CAST_SHORT(a0);
        ialpha[dx * 2 + 1] = SATURATE_CAST_SHORT(a1);
    }

    for (int dy = 0; dy < h; dy++)
    {
        fy = (float)((dy + 0.5) * scale_y - 0.5);
        sy = static_cast<int>(floor(fy));
        fy -= sy;

        if (sy < 0)
        {
            sy = 0;
            fy = 0.f;
        }
        if (sy >= srch - 1)
        {
            sy = srch - 2;
            fy = 1.f;
        }

        yofs[dy] = sy;

        float b0 = (1.f - fy) * INTER_RESIZE_COEF_SCALE;
        float b1 = fy * INTER_RESIZE_COEF_SCALE;

        ibeta[dy * 2] = SATURATE_CAST_SHORT(b0);
        ibeta[dy * 2 + 1] = SATURATE_CAST_SHORT(b1);
    }

#undef SATURATE_CAST_SHORT

    // loop body
    Mat rowsbuf0(w, (size_t)2u);
    Mat rowsbuf1(w, (size_t)2u);
    short* rows0 = (short*)rowsbuf0.data;
    short* rows1 = (short*)rowsbuf1.data;

    int prev_sy1 = -2;

    for (int dy = 0; dy < h; dy++)
    {
        sy = yofs[dy];

        if (sy == prev_sy1)
        {
            // reuse all rows
        }
        else if (sy == prev_sy1 + 1)
        {
            // hresize one row
            short* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows0_old;
            const unsigned char* S1 = src + srcstride * (sy + 1);

            const short* ialphap = ialpha;
            short* rows1p = rows1;
            for (int dx = 0; dx < w; dx++)
            {
                sx = xofs[dx];
                short a0 = ialphap[0];
                short a1 = ialphap[1];

                const unsigned char* S1p = S1 + sx;
                rows1p[dx] = (S1p[0] * a0 + S1p[1] * a1) >> 4;

                ialphap += 2;
            }
        }
        else
        {
            // hresize two rows
            const unsigned char* S0 = src + srcstride * (sy);
            const unsigned char* S1 = src + srcstride * (sy + 1);

            const short* ialphap = ialpha;
            short* rows0p = rows0;
            short* rows1p = rows1;
            for (int dx = 0; dx < w; dx++)
            {
                sx = xofs[dx];
                short a0 = ialphap[0];
                short a1 = ialphap[1];

                const unsigned char* S0p = S0 + sx;
                const unsigned char* S1p = S1 + sx;
                rows0p[dx] = (S0p[0] * a0 + S0p[1] * a1) >> 4;
                rows1p[dx] = (S1p[0] * a0 + S1p[1] * a1) >> 4;

                ialphap += 2;
            }
        }

        prev_sy1 = sy;

        if (dy + 1 < h && yofs[dy + 1] == sy)
        {
            // vresize for two rows
            unsigned char* Dp0 = dst + stride * dy;
            unsigned char* Dp1 = dst + stride * (dy + 1);

            vresize_two(rows0, rows1, w, Dp0, Dp1, ibeta[0], ibeta[1], ibeta[2], ibeta[3]);

            ibeta += 4;
            dy += 1;
        }
        else
        {
            // vresize
            unsigned char* Dp = dst + stride * dy;

            vresize_one(rows0, rows1, w, Dp, ibeta[0], ibeta[1]);

            ibeta += 2;
        }
    }

    delete[] buf;
}

void resize_bilinear_c2(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    const int INTER_RESIZE_COEF_BITS = 11;
    const int INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS;

    double scale_x = (double)srcw / w;
    double scale_y = (double)srch / h;

    int* buf = new int[w + h + w + h];

    int* xofs = buf;     //new int[w];
    int* yofs = buf + w; //new int[h];

    short* ialpha = (short*)(buf + w + h);    //new short[w * 2];
    short* ibeta = (short*)(buf + w + h + w); //new short[h * 2];

    float fx;
    float fy;
    int sx;
    int sy;

#define SATURATE_CAST_SHORT(X) (short)::std::min(::std::max((int)(X + (X >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), SHRT_MAX);

    for (int dx = 0; dx < w; dx++)
    {
        fx = (float)((dx + 0.5) * scale_x - 0.5);
        sx = static_cast<int>(floor(fx));
        fx -= sx;

        if (sx < 0)
        {
            sx = 0;
            fx = 0.f;
        }
        if (sx >= srcw - 1)
        {
            sx = srcw - 2;
            fx = 1.f;
        }

        xofs[dx] = sx * 2;

        float a0 = (1.f - fx) * INTER_RESIZE_COEF_SCALE;
        float a1 = fx * INTER_RESIZE_COEF_SCALE;

        ialpha[dx * 2] = SATURATE_CAST_SHORT(a0);
        ialpha[dx * 2 + 1] = SATURATE_CAST_SHORT(a1);
    }

    for (int dy = 0; dy < h; dy++)
    {
        fy = (float)((dy + 0.5) * scale_y - 0.5);
        sy = static_cast<int>(floor(fy));
        fy -= sy;

        if (sy < 0)
        {
            sy = 0;
            fy = 0.f;
        }
        if (sy >= srch - 1)
        {
            sy = srch - 2;
            fy = 1.f;
        }

        yofs[dy] = sy;

        float b0 = (1.f - fy) * INTER_RESIZE_COEF_SCALE;
        float b1 = fy * INTER_RESIZE_COEF_SCALE;

        ibeta[dy * 2] = SATURATE_CAST_SHORT(b0);
        ibeta[dy * 2 + 1] = SATURATE_CAST_SHORT(b1);
    }

#undef SATURATE_CAST_SHORT

    // loop body
    Mat rowsbuf0(w * 2 + 2, (size_t)2u);
    Mat rowsbuf1(w * 2 + 2, (size_t)2u);
    short* rows0 = (short*)rowsbuf0.data;
    short* rows1 = (short*)rowsbuf1.data;

    int prev_sy1 = -2;

    for (int dy = 0; dy < h; dy++)
    {
        sy = yofs[dy];

        if (sy == prev_sy1)
        {
            // reuse all rows
        }
        else if (sy == prev_sy1 + 1)
        {
            // hresize one row
            short* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows0_old;
            const unsigned char* S1 = src + srcstride * (sy + 1);

            const short* ialphap = ialpha;
            short* rows1p = rows1;
            for (int dx = 0; dx < w; dx++)
            {
                sx = xofs[dx];

                const unsigned char* S1p = S1 + sx;
#if __ARM_NEON
                int16x4_t _a0a1XX = vld1_s16(ialphap);
                int16x4_t _a0a0a1a1 = vzip_s16(_a0a1XX, _a0a1XX).val[0];
                uint8x8_t _S1 = uint8x8_t();

                _S1 = vld1_lane_u8(S1p, _S1, 0);
                _S1 = vld1_lane_u8(S1p + 1, _S1, 1);
                _S1 = vld1_lane_u8(S1p + 2, _S1, 2);
                _S1 = vld1_lane_u8(S1p + 3, _S1, 3);

                int16x8_t _S116 = vreinterpretq_s16_u16(vmovl_u8(_S1));
                int16x4_t _S1lowhigh = vget_low_s16(_S116);
                int32x4_t _S1ma0a1 = vmull_s16(_S1lowhigh, _a0a0a1a1);
                int32x2_t _rows1low = vadd_s32(vget_low_s32(_S1ma0a1), vget_high_s32(_S1ma0a1));
                int32x4_t _rows1 = vcombine_s32(_rows1low, vget_high_s32(_S1ma0a1));
                int16x4_t _rows1_sr4 = vshrn_n_s32(_rows1, 4);
                vst1_s16(rows1p, _rows1_sr4);
#else
                short a0 = ialphap[0];
                short a1 = ialphap[1];

                rows1p[0] = (S1p[0] * a0 + S1p[2] * a1) >> 4;
                rows1p[1] = (S1p[1] * a0 + S1p[3] * a1) >> 4;
#endif // __ARM_NEON

                ialphap += 2;
                rows1p += 2;
            }
        }
        else
        {
            // hresize two rows
            const unsigned char* S0 = src + srcstride * (sy);
            const unsigned char* S1 = src + srcstride * (sy + 1);

            const short* ialphap = ialpha;
            short* rows0p = rows0;
            short* rows1p = rows1;
            for (int dx = 0; dx < w; dx++)
            {
                sx = xofs[dx];
                short a0 = ialphap[0];
                short a1 = ialphap[1];

                const unsigned char* S0p = S0 + sx;
                const unsigned char* S1p = S1 + sx;
#if __ARM_NEON
                int16x4_t _a0 = vdup_n_s16(a0);
                int16x4_t _a1 = vdup_n_s16(a1);
                uint8x8_t _S0 = uint8x8_t();
                uint8x8_t _S1 = uint8x8_t();

                _S0 = vld1_lane_u8(S0p, _S0, 0);
                _S0 = vld1_lane_u8(S0p + 1, _S0, 1);
                _S0 = vld1_lane_u8(S0p + 2, _S0, 2);
                _S0 = vld1_lane_u8(S0p + 3, _S0, 3);

                _S1 = vld1_lane_u8(S1p, _S1, 0);
                _S1 = vld1_lane_u8(S1p + 1, _S1, 1);
                _S1 = vld1_lane_u8(S1p + 2, _S1, 2);
                _S1 = vld1_lane_u8(S1p + 3, _S1, 3);

                int16x8_t _S016 = vreinterpretq_s16_u16(vmovl_u8(_S0));
                int16x8_t _S116 = vreinterpretq_s16_u16(vmovl_u8(_S1));
                int16x4_t _S0lowhigh = vget_low_s16(_S016);
                int16x4_t _S1lowhigh = vget_low_s16(_S116);
                int32x2x2_t _S0S1low_S0S1high = vtrn_s32(vreinterpret_s32_s16(_S0lowhigh), vreinterpret_s32_s16(_S1lowhigh));
                int32x4_t _rows01 = vmull_s16(vreinterpret_s16_s32(_S0S1low_S0S1high.val[0]), _a0);
                _rows01 = vmlal_s16(_rows01, vreinterpret_s16_s32(_S0S1low_S0S1high.val[1]), _a1);
                int16x4_t _rows01_sr4 = vshrn_n_s32(_rows01, 4);
                int16x4_t _rows1_sr4 = vext_s16(_rows01_sr4, _rows01_sr4, 2);
                vst1_s16(rows0p, _rows01_sr4);
                vst1_s16(rows1p, _rows1_sr4);
#else
                rows0p[0] = (S0p[0] * a0 + S0p[2] * a1) >> 4;
                rows0p[1] = (S0p[1] * a0 + S0p[3] * a1) >> 4;
                rows1p[0] = (S1p[0] * a0 + S1p[2] * a1) >> 4;
                rows1p[1] = (S1p[1] * a0 + S1p[3] * a1) >> 4;
#endif // __ARM_NEON

                ialphap += 2;
                rows0p += 2;
                rows1p += 2;
            }
        }

        prev_sy1 = sy;

        if (dy + 1 < h && yofs[dy + 1] == sy)
        {
            // vresize for two rows
            unsigned char* Dp0 = dst + stride * dy;
            unsigned char* Dp1 = dst + stride * (dy + 1);

            vresize_two(rows0, rows1, w * 2, Dp0, Dp1, ibeta[0], ibeta[1], ibeta[2], ibeta[3]);

            ibeta += 4;
            dy += 1;
        }
        else
        {
            // vresize
            unsigned char* Dp = dst + stride * dy;

            vresize_one(rows0, rows1, w * 2, Dp, ibeta[0], ibeta[1]);

            ibeta += 2;
        }
    }

    delete[] buf;
}

void resize_bilinear_c3(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    const int INTER_RESIZE_COEF_BITS = 11;
    const int INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS;

    double scale_x = (double)srcw / w;
    double scale_y = (double)srch / h;

    int* buf = new int[w + h + w + h];

    int* xofs = buf;     //new int[w];
    int* yofs = buf + w; //new int[h];

    short* ialpha = (short*)(buf + w + h);    //new short[w * 2];
    short* ibeta = (short*)(buf + w + h + w); //new short[h * 2];

    float fx;
    float fy;
    int sx;
    int sy;

#define SATURATE_CAST_SHORT(X) (short)::std::min(::std::max((int)(X + (X >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), SHRT_MAX);

    for (int dx = 0; dx < w; dx++)
    {
        fx = (float)((dx + 0.5) * scale_x - 0.5);
        sx = static_cast<int>(floor(fx));
        fx -= sx;

        if (sx < 0)
        {
            sx = 0;
            fx = 0.f;
        }
        if (sx >= srcw - 1)
        {
            sx = srcw - 2;
            fx = 1.f;
        }

        xofs[dx] = sx * 3;

        float a0 = (1.f - fx) * INTER_RESIZE_COEF_SCALE;
        float a1 = fx * INTER_RESIZE_COEF_SCALE;

        ialpha[dx * 2] = SATURATE_CAST_SHORT(a0);
        ialpha[dx * 2 + 1] = SATURATE_CAST_SHORT(a1);
    }

    for (int dy = 0; dy < h; dy++)
    {
        fy = (float)((dy + 0.5) * scale_y - 0.5);
        sy = static_cast<int>(floor(fy));
        fy -= sy;

        if (sy < 0)
        {
            sy = 0;
            fy = 0.f;
        }
        if (sy >= srch - 1)
        {
            sy = srch - 2;
            fy = 1.f;
        }

        yofs[dy] = sy;

        float b0 = (1.f - fy) * INTER_RESIZE_COEF_SCALE;
        float b1 = fy * INTER_RESIZE_COEF_SCALE;

        ibeta[dy * 2] = SATURATE_CAST_SHORT(b0);
        ibeta[dy * 2 + 1] = SATURATE_CAST_SHORT(b1);
    }

#undef SATURATE_CAST_SHORT

    // loop body
    Mat rowsbuf0(w * 3 + 1, (size_t)2u);
    Mat rowsbuf1(w * 3 + 1, (size_t)2u);
    short* rows0 = (short*)rowsbuf0.data;
    short* rows1 = (short*)rowsbuf1.data;

    int prev_sy1 = -2;

    for (int dy = 0; dy < h; dy++)
    {
        sy = yofs[dy];

        if (sy == prev_sy1)
        {
            // reuse all rows
        }
        else if (sy == prev_sy1 + 1)
        {
            // hresize one row
            short* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows0_old;
            const unsigned char* S1 = src + srcstride * (sy + 1);

            const short* ialphap = ialpha;
            short* rows1p = rows1;
            for (int dx = 0; dx < w; dx++)
            {
                sx = xofs[dx];
                short a0 = ialphap[0];
                short a1 = ialphap[1];

                const unsigned char* S1p = S1 + sx;
#if __ARM_NEON
                int16x4_t _a0 = vdup_n_s16(a0);
                int16x4_t _a1 = vdup_n_s16(a1);
                uint8x8_t _S1 = uint8x8_t();

                _S1 = vld1_lane_u8(S1p, _S1, 0);
                _S1 = vld1_lane_u8(S1p + 1, _S1, 1);
                _S1 = vld1_lane_u8(S1p + 2, _S1, 2);
                _S1 = vld1_lane_u8(S1p + 3, _S1, 3);
                _S1 = vld1_lane_u8(S1p + 4, _S1, 4);
                _S1 = vld1_lane_u8(S1p + 5, _S1, 5);

                int16x8_t _S116 = vreinterpretq_s16_u16(vmovl_u8(_S1));
                int16x4_t _S1low = vget_low_s16(_S116);
                int16x4_t _S1high = vext_s16(_S1low, vget_high_s16(_S116), 3);
                int32x4_t _rows1 = vmull_s16(_S1low, _a0);
                _rows1 = vmlal_s16(_rows1, _S1high, _a1);
                int16x4_t _rows1_sr4 = vshrn_n_s32(_rows1, 4);
                vst1_s16(rows1p, _rows1_sr4);
#else
                rows1p[0] = (S1p[0] * a0 + S1p[3] * a1) >> 4;
                rows1p[1] = (S1p[1] * a0 + S1p[4] * a1) >> 4;
                rows1p[2] = (S1p[2] * a0 + S1p[5] * a1) >> 4;
#endif // __ARM_NEON

                ialphap += 2;
                rows1p += 3;
            }
        }
        else
        {
            // hresize two rows
            const unsigned char* S0 = src + srcstride * (sy);
            const unsigned char* S1 = src + srcstride * (sy + 1);

            const short* ialphap = ialpha;
            short* rows0p = rows0;
            short* rows1p = rows1;
            for (int dx = 0; dx < w; dx++)
            {
                sx = xofs[dx];
                short a0 = ialphap[0];
                short a1 = ialphap[1];

                const unsigned char* S0p = S0 + sx;
                const unsigned char* S1p = S1 + sx;
#if __ARM_NEON
                int16x4_t _a0 = vdup_n_s16(a0);
                int16x4_t _a1 = vdup_n_s16(a1);
                uint8x8_t _S0 = uint8x8_t();
                uint8x8_t _S1 = uint8x8_t();

                _S0 = vld1_lane_u8(S0p, _S0, 0);
                _S0 = vld1_lane_u8(S0p + 1, _S0, 1);
                _S0 = vld1_lane_u8(S0p + 2, _S0, 2);
                _S0 = vld1_lane_u8(S0p + 3, _S0, 3);
                _S0 = vld1_lane_u8(S0p + 4, _S0, 4);
                _S0 = vld1_lane_u8(S0p + 5, _S0, 5);

                _S1 = vld1_lane_u8(S1p, _S1, 0);
                _S1 = vld1_lane_u8(S1p + 1, _S1, 1);
                _S1 = vld1_lane_u8(S1p + 2, _S1, 2);
                _S1 = vld1_lane_u8(S1p + 3, _S1, 3);
                _S1 = vld1_lane_u8(S1p + 4, _S1, 4);
                _S1 = vld1_lane_u8(S1p + 5, _S1, 5);

                int16x8_t _S016 = vreinterpretq_s16_u16(vmovl_u8(_S0));
                int16x8_t _S116 = vreinterpretq_s16_u16(vmovl_u8(_S1));
                int16x4_t _S0low = vget_low_s16(_S016);
                int16x4_t _S1low = vget_low_s16(_S116);
                int16x4_t _S0high = vext_s16(_S0low, vget_high_s16(_S016), 3);
                int16x4_t _S1high = vext_s16(_S1low, vget_high_s16(_S116), 3);
                int32x4_t _rows0 = vmull_s16(_S0low, _a0);
                int32x4_t _rows1 = vmull_s16(_S1low, _a0);
                _rows0 = vmlal_s16(_rows0, _S0high, _a1);
                _rows1 = vmlal_s16(_rows1, _S1high, _a1);
                int16x4_t _rows0_sr4 = vshrn_n_s32(_rows0, 4);
                int16x4_t _rows1_sr4 = vshrn_n_s32(_rows1, 4);
                vst1_s16(rows0p, _rows0_sr4);
                vst1_s16(rows1p, _rows1_sr4);
#else
                rows0p[0] = (S0p[0] * a0 + S0p[3] * a1) >> 4;
                rows0p[1] = (S0p[1] * a0 + S0p[4] * a1) >> 4;
                rows0p[2] = (S0p[2] * a0 + S0p[5] * a1) >> 4;
                rows1p[0] = (S1p[0] * a0 + S1p[3] * a1) >> 4;
                rows1p[1] = (S1p[1] * a0 + S1p[4] * a1) >> 4;
                rows1p[2] = (S1p[2] * a0 + S1p[5] * a1) >> 4;
#endif // __ARM_NEON

                ialphap += 2;
                rows0p += 3;
                rows1p += 3;
            }
        }

        prev_sy1 = sy;

        if (dy + 1 < h && yofs[dy + 1] == sy)
        {
            // vresize for two rows
            unsigned char* Dp0 = dst + stride * dy;
            unsigned char* Dp1 = dst + stride * (dy + 1);

            vresize_two(rows0, rows1, w * 3, Dp0, Dp1, ibeta[0], ibeta[1], ibeta[2], ibeta[3]);

            ibeta += 4;
            dy += 1;
        }
        else
        {
            // vresize
            unsigned char* Dp = dst + stride * dy;

            vresize_one(rows0, rows1, w * 3, Dp, ibeta[0], ibeta[1]);

            ibeta += 2;
        }
    }

    delete[] buf;
}

void resize_bilinear_c4(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    const int INTER_RESIZE_COEF_BITS = 11;
    const int INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS;

    double scale_x = (double)srcw / w;
    double scale_y = (double)srch / h;

    int* buf = new int[w + h + w + h];

    int* xofs = buf;     //new int[w];
    int* yofs = buf + w; //new int[h];

    short* ialpha = (short*)(buf + w + h);    //new short[w * 2];
    short* ibeta = (short*)(buf + w + h + w); //new short[h * 2];

    float fx;
    float fy;
    int sx;
    int sy;

#define SATURATE_CAST_SHORT(X) (short)::std::min(::std::max((int)(X + (X >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), SHRT_MAX);

    for (int dx = 0; dx < w; dx++)
    {
        fx = (float)((dx + 0.5) * scale_x - 0.5);
        sx = static_cast<int>(floor(fx));
        fx -= sx;

        if (sx < 0)
        {
            sx = 0;
            fx = 0.f;
        }
        if (sx >= srcw - 1)
        {
            sx = srcw - 2;
            fx = 1.f;
        }

        xofs[dx] = sx * 4;

        float a0 = (1.f - fx) * INTER_RESIZE_COEF_SCALE;
        float a1 = fx * INTER_RESIZE_COEF_SCALE;

        ialpha[dx * 2] = SATURATE_CAST_SHORT(a0);
        ialpha[dx * 2 + 1] = SATURATE_CAST_SHORT(a1);
    }

    for (int dy = 0; dy < h; dy++)
    {
        fy = (float)((dy + 0.5) * scale_y - 0.5);
        sy = static_cast<int>(floor(fy));
        fy -= sy;

        if (sy < 0)
        {
            sy = 0;
            fy = 0.f;
        }
        if (sy >= srch - 1)
        {
            sy = srch - 2;
            fy = 1.f;
        }

        yofs[dy] = sy;

        float b0 = (1.f - fy) * INTER_RESIZE_COEF_SCALE;
        float b1 = fy * INTER_RESIZE_COEF_SCALE;

        ibeta[dy * 2] = SATURATE_CAST_SHORT(b0);
        ibeta[dy * 2 + 1] = SATURATE_CAST_SHORT(b1);
    }

#undef SATURATE_CAST_SHORT

    // loop body
    Mat rowsbuf0(w * 4, (size_t)2u);
    Mat rowsbuf1(w * 4, (size_t)2u);
    short* rows0 = (short*)rowsbuf0.data;
    short* rows1 = (short*)rowsbuf1.data;

    int prev_sy1 = -2;

    for (int dy = 0; dy < h; dy++)
    {
        sy = yofs[dy];

        if (sy == prev_sy1)
        {
            // reuse all rows
        }
        else if (sy == prev_sy1 + 1)
        {
            // hresize one row
            short* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows0_old;
            const unsigned char* S1 = src + srcstride * (sy + 1);

            const short* ialphap = ialpha;
            short* rows1p = rows1;
            for (int dx = 0; dx < w; dx++)
            {
                sx = xofs[dx];
                short a0 = ialphap[0];
                short a1 = ialphap[1];

                const unsigned char* S1p = S1 + sx;
#if __ARM_NEON
                int16x4_t _a0 = vdup_n_s16(a0);
                int16x4_t _a1 = vdup_n_s16(a1);
                uint8x8_t _S1 = vld1_u8(S1p);
                int16x8_t _S116 = vreinterpretq_s16_u16(vmovl_u8(_S1));
                int16x4_t _S1low = vget_low_s16(_S116);
                int16x4_t _S1high = vget_high_s16(_S116);
                int32x4_t _rows1 = vmull_s16(_S1low, _a0);
                _rows1 = vmlal_s16(_rows1, _S1high, _a1);
                int16x4_t _rows1_sr4 = vshrn_n_s32(_rows1, 4);
                vst1_s16(rows1p, _rows1_sr4);
#else
                rows1p[0] = (S1p[0] * a0 + S1p[4] * a1) >> 4;
                rows1p[1] = (S1p[1] * a0 + S1p[5] * a1) >> 4;
                rows1p[2] = (S1p[2] * a0 + S1p[6] * a1) >> 4;
                rows1p[3] = (S1p[3] * a0 + S1p[7] * a1) >> 4;
#endif // __ARM_NEON

                ialphap += 2;
                rows1p += 4;
            }
        }
        else
        {
            // hresize two rows
            const unsigned char* S0 = src + srcstride * (sy);
            const unsigned char* S1 = src + srcstride * (sy + 1);

            const short* ialphap = ialpha;
            short* rows0p = rows0;
            short* rows1p = rows1;
            for (int dx = 0; dx < w; dx++)
            {
                sx = xofs[dx];
                short a0 = ialphap[0];
                short a1 = ialphap[1];

                const unsigned char* S0p = S0 + sx;
                const unsigned char* S1p = S1 + sx;
#if __ARM_NEON
                int16x4_t _a0 = vdup_n_s16(a0);
                int16x4_t _a1 = vdup_n_s16(a1);
                uint8x8_t _S0 = vld1_u8(S0p);
                uint8x8_t _S1 = vld1_u8(S1p);
                int16x8_t _S016 = vreinterpretq_s16_u16(vmovl_u8(_S0));
                int16x8_t _S116 = vreinterpretq_s16_u16(vmovl_u8(_S1));
                int16x4_t _S0low = vget_low_s16(_S016);
                int16x4_t _S1low = vget_low_s16(_S116);
                int16x4_t _S0high = vget_high_s16(_S016);
                int16x4_t _S1high = vget_high_s16(_S116);
                int32x4_t _rows0 = vmull_s16(_S0low, _a0);
                int32x4_t _rows1 = vmull_s16(_S1low, _a0);
                _rows0 = vmlal_s16(_rows0, _S0high, _a1);
                _rows1 = vmlal_s16(_rows1, _S1high, _a1);
                int16x4_t _rows0_sr4 = vshrn_n_s32(_rows0, 4);
                int16x4_t _rows1_sr4 = vshrn_n_s32(_rows1, 4);
                vst1_s16(rows0p, _rows0_sr4);
                vst1_s16(rows1p, _rows1_sr4);
#else
                rows0p[0] = (S0p[0] * a0 + S0p[4] * a1) >> 4;
                rows0p[1] = (S0p[1] * a0 + S0p[5] * a1) >> 4;
                rows0p[2] = (S0p[2] * a0 + S0p[6] * a1) >> 4;
                rows0p[3] = (S0p[3] * a0 + S0p[7] * a1) >> 4;
                rows1p[0] = (S1p[0] * a0 + S1p[4] * a1) >> 4;
                rows1p[1] = (S1p[1] * a0 + S1p[5] * a1) >> 4;
                rows1p[2] = (S1p[2] * a0 + S1p[6] * a1) >> 4;
                rows1p[3] = (S1p[3] * a0 + S1p[7] * a1) >> 4;
#endif // __ARM_NEON

                ialphap += 2;
                rows0p += 4;
                rows1p += 4;
            }
        }

        prev_sy1 = sy;

        if (dy + 1 < h && yofs[dy + 1] == sy)
        {
            // vresize for two rows
            unsigned char* Dp0 = dst + stride * dy;
            unsigned char* Dp1 = dst + stride * (dy + 1);

            vresize_two(rows0, rows1, w * 4, Dp0, Dp1, ibeta[0], ibeta[1], ibeta[2], ibeta[3]);

            ibeta += 4;
            dy += 1;
        }
        else
        {
            // vresize
            unsigned char* Dp = dst + stride * dy;

            vresize_one(rows0, rows1, w * 4, Dp, ibeta[0], ibeta[1]);

            ibeta += 2;
        }
    }

    delete[] buf;
}

void resize_bilinear_yuv420sp(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h)
{
    // assert srcw % 2 == 0
    // assert srch % 2 == 0
    // assert w % 2 == 0
    // assert h % 2 == 0

    const unsigned char* srcY = src;
    unsigned char* dstY = dst;
    resize_bilinear_c1(srcY, srcw, srch, dstY, w, h);

    const unsigned char* srcUV = src + srcw * srch;
    unsigned char* dstUV = dst + w * h;
    resize_bilinear_c2(srcUV, srcw / 2, srch / 2, dstUV, w / 2, h / 2);
}
#endif // NCNN_PIXEL

} // namespace ncnn
