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

#include <cstdlib>
#include <cstdio>
#include <limits.h>
#include <math.h>
#include <x86intrin.h>
#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON
#include "platform.h"

namespace ncnn {

#if NCNN_PIXEL
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
    //     const int ONE=INTER_RESIZE_COEF_SCALE;

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

        // vresize
        int b0 = (int)ibeta[0];         // 使其成为32bit数
        int b1 = (int)ibeta[1];

        short* rows0p = rows0;
        short* rows1p = rows1;
        unsigned char* Dp = dst + stride * (dy);


        int nn = w >> 2;
        int remain = w - (nn << 2);

        __m128i _b0 = _mm_set1_epi32(b0);    // 需要修改，用set指令
        __m128i _b1 = _mm_set1_epi32(b1);    // 需要修改，用set指令
        __m128i _v2 = _mm_set1_epi32(2);
        for(; nn > 0; nn--)
        {
            //__mm128i _rows0p_sr4 = _mm_loadl_epi64(rows0p);
            //__mm128i _rows1p_sr4 = _mm_loadl_epi64(rows1p);
            __m128i rows0p_0_sr4 = _mm_set_epi32(0, (int)*(rows0p+2), 0, (int)*(rows0p));
            __m128i rows0p_1_sr4 = _mm_set_epi32(0, (int)*(rows0p+3), 0, (int)*(rows0p+1));
            __m128i rows1p_0_sr4 = _mm_set_epi32(0, (int)*(rows1p+2), 0, (int)*(rows1p));
            __m128i rows1p_1_sr4 = _mm_set_epi32(0, (int)*(rows1p+3), 0, (int)*(rows1p+1));
            // __mm128i _rows0p_1_sr4 = _mm_load_epi64(rows0p + 4);
            // __mm128i _rows1p_1_sr4 = _mm_load_epi64(rows1p + 4);

            __m128i _rows0p_0_sr4_mb0 = _mm_mul_epu32(rows0p_0_sr4, _b0);
            __m128i _rows0p_1_sr4_mb0 = _mm_mul_epu32(rows0p_1_sr4, _b0);
            __m128i _rows1p_0_sr4_mb1 = _mm_mul_epu32(rows1p_0_sr4, _b1);
            __m128i _rows1p_1_sr4_mb1 = _mm_mul_epu32(rows1p_1_sr4, _b1);
            // __mm128 _rows0p_1_sr4_mb0 = _mm_mullo_epi16(_rows0p_1_sr4, _b0);
            // __mm128 _rows1p_1_sr4_mb1 = _mm_mullo_epi16(_rows1p_1_sr4, _b1);

            // right shift & pack
            // __m128i _acc = _v2;
            // _acc = _mm_add_epi32(_mm_srli_epi32(_rows0p_0_sr4_mb0, 48), _acc);
            // _acc = _mm_add_epi32(_mm_srli_epi32(_rows1p_sr4_mb1, 48), _acc); 
            // __m128i rows0p_0_unpack = _mm_unpacklo_epi32(_mm_srli_epi64(_rows0p_0_sr4_mb0, 32), _mm_srli_epi64(_rows0p_1_sr4_mb0, 32));
            // __m128i rows1p_0_unpack = _mm_unpacklo_epi32(_mm_srli_epi64(_rows1p_0_sr4_mb1, 32), _mm_srli_epi64(_rows1p_1_sr4_mb1, 32));
            // __m128i rows0p_1_unpack = _mm_unpackhi_epi32(_mm_srli_epi64(_rows0p_0_sr4_mb0, 32), _mm_srli_epi64(_rows0p_1_sr4_mb0, 32));
            // __m128i rows1p_1_unpack = _mm_unpackhi_epi32(_mm_srli_epi64(_rows1p_0_sr4_mb1, 32), _mm_srli_epi64(_rows1p_1_sr4_mb1, 32));
	        __m128i rows0p_0_unpack = _mm_unpacklo_epi32(_rows0p_0_sr4_mb0, _rows0p_1_sr4_mb0);
	        __m128i rows1p_0_unpack = _mm_unpacklo_epi32(_rows1p_0_sr4_mb1, _rows1p_1_sr4_mb1);
	        __m128i rows0p_1_unpack = _mm_unpackhi_epi32(_rows0p_0_sr4_mb0, _rows0p_1_sr4_mb0);
	        __m128i rows1p_1_unpack = _mm_unpackhi_epi32(_rows1p_0_sr4_mb1, _rows1p_1_sr4_mb1);
            __m128i rows0p_pack = _mm_unpacklo_epi64(rows0p_0_unpack, rows0p_1_unpack);
            __m128i rows1p_pack = _mm_unpacklo_epi64(rows1p_0_unpack, rows1p_1_unpack);
            __m128i _acc = _v2;
            _acc = _mm_add_epi32(rows0p_pack, _acc);
            _acc = _mm_add_epi32(rows1p_pack, _acc);
            
            // 右移指令, 并且将int32转化成int8
             __m128i _acc16 = _mm_srli_epi32(_acc, 2);
            
            int* buffer_acc = (int*)&_acc16;
	        for(size_t i = 0; i < 4; ++i){
		        // std::cout << buffer_acc[i] << std::endl;
	    	    buffer_acc[i] = (unsigned char)(buffer_acc[i] >> 16);
		        *(Dp+i) = buffer_acc[i];
	        }


            Dp += 4;
            rows0p += 4;
            rows1p += 4;
        }
        for (; remain; --remain)
        {
            //             D[x] = (rows0[x]*b0 + rows1[x]*b1) >> INTER_RESIZE_COEF_BITS;
            *Dp++ = (unsigned char)(((short)((b0 * (short)(*rows0p++)) >> 16) + (short)((b1 * (short)(*rows1p++)) >> 16) + 2) >> 2);
        }

        ibeta += 2;
    }

    delete[] buf;
}

void resize_bilinear_c2(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    const int INTER_RESIZE_COEF_BITS = 11;
    const int INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS;
    //     const int ONE=INTER_RESIZE_COEF_SCALE;

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
                __m128i _a0a0 = _mm_set_epi32(0, (int)(*(ialphap)), 0, (int)(*(ialphap)));
                __m128i _a1a1 = _mm_set_epi32(0, (int)(*(ialphap+1)), 0, (int)(*(ialphap+1)));
                __m128i _S1_0 = _mm_set_epi32(0, (int)(*(S1p+1)), 0, (int)(*(S1p)));
                __m128i _S1_1 = _mm_set_epi32(0, (int)(*(S1p+3)), 0, (int)(*(S1p+2)));
                // multiply and shift code
                __m128i _S1ma0a1_0 = _mm_mul_epu32(_S1_0, _a0a0);
                __m128i _S1ma0a1_1 = _mm_mul_epu32(_S1_1, _a1a1);
                __m128i _S1ma0a1_add = _mm_add_epi32(_S1ma0a1_0, _S1ma0a1_1);
                __m128i _rows1_sr4 = _mm_srli_epi32(_S1ma0a1_add, 4);
                // store code
                int* temp_sr4 = (int*)&_rows1_sr4;
                rows1p[0] = (short)(*(temp_sr4));
                rows1p[1] = (short)(*(temp_sr4+2));
                
                ialphap += 2;
                rows1p += 2;
            }
        }
        else
        {
            // hresize tow row
            const unsigned char* S0 = src + srcstride * sy;
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

                __m128i _a0a0 = _mm_set_epi32(0, (int)a0, 0, (int)a0);
                __m128i _a1a1 = _mm_set_epi32(0, (int)a1, 0, (int)a1);
                __m128i _S0_0 = _mm_set_epi32(0, (int)(*(S0p+1)), 0, (int)(*S0p));
                __m128i _S0_1 = _mm_set_epi32(0, (int)(*(S0p+3)), 0, (int)(*(S0p+2)));
                __m128i _S1_0 = _mm_set_epi32(0, (int)(*(S1p+1)), 0, (int)(*S1p));
                __m128i _S1_1 = _mm_set_epi32(0, (int)(*(S1p+3)), 0, (int)(*(S1p+2)));

                // multiply and shift code
                __m128i _S0ma0a1 = _mm_add_epi32(_mm_mul_epu32(_S0_0, _a0a0), _mm_mul_epu32(_S0_1, _a1a1));
                __m128i _S1ma0a1 = _mm_add_epi32(_mm_mul_epu32(_S1_0, _a0a0), _mm_mul_epu32(_S1_1, _a1a1));
                __m128i _rows0_sr4 = _mm_srli_epi32(_S0ma0a1, 4);
                __m128i _rows1_sr4 = _mm_srli_epi32(_S1ma0a1, 4);

                // store code
                int* temp0_sr4 = (int*)&_rows0_sr4;
                int* temp1_sr4 = (int*)&_rows1_sr4;
                rows0p[0] = (short)(*(temp0_sr4));
                rows0p[1] = (short)(*(temp0_sr4+2));
                rows1p[0] = (short)(*(temp1_sr4));
                rows1p[1] = (short)(*(temp1_sr4+2));

                ialphap += 2;
                rows0p += 2;
                rows1p += 2;
            }
        }

        prev_sy1 = sy;

        // vresize
        int b0 = (int)ibeta[0];
        int b1 = (int)ibeta[1];

        short* rows0p = rows0;
        short* rows1p = rows1;
        unsigned char* Dp = dst + stride * (dy);

        int nn = (w * 2) >> 2;
        int remain = (w * 2) - (nn << 2);

        __m128i _b0 = _mm_set1_epi32(b0);
        __m128i _b1 = _mm_set1_epi32(b1);
        __m128i _v2 = _mm_set1_epi32(2);
        for (; nn > 0; --nn)
        {
            __m128i rows0p_0_sr4 = _mm_set_epi32(0, (int)*(rows0p+2), 0, (int)*(rows0p));
            __m128i rows0p_1_sr4 = _mm_set_epi32(0, (int)*(rows0p+3), 0, (int)*(rows0p+1));
            __m128i rows1p_0_sr4 = _mm_set_epi32(0, (int)*(rows1p+2), 0, (int)*(rows1p));
            __m128i rows1p_1_sr4 = _mm_set_epi32(0, (int)*(rows1p+3), 0, (int)*(rows1p+1));

            __m128i _rows0p_0_sr4_mb0 = _mm_mul_epu32(rows0p_0_sr4, _b0);
            __m128i _rows0p_1_sr4_mb0 = _mm_mul_epu32(rows0p_1_sr4, _b0);
            __m128i _rows1p_0_sr4_mb1 = _mm_mul_epu32(rows1p_0_sr4, _b1);
            __m128i _rows1p_1_sr4_mb1 = _mm_mul_epu32(rows1p_1_sr4, _b1);

            __m128i rows0p_0_unpack = _mm_unpacklo_epi32(_rows0p_0_sr4_mb0, _rows0p_1_sr4_mb0);
            __m128i rows1p_0_unpack = _mm_unpacklo_epi32(_rows1p_0_sr4_mb1, _rows1p_1_sr4_mb1);
            __m128i rows0p_1_unpack = _mm_unpackhi_epi32(_rows0p_0_sr4_mb0, _rows0p_1_sr4_mb0);
            __m128i rows1p_1_unpack = _mm_unpackhi_epi32(_rows1p_0_sr4_mb1, _rows1p_1_sr4_mb1);
            __m128i rows0p_pack = _mm_unpacklo_epi64(rows0p_0_unpack, rows0p_1_unpack);
            __m128i rows1p_pack = _mm_unpacklo_epi64(rows1p_0_unpack, rows1p_1_unpack);
            __m128i _acc = _v2;
            _acc = _mm_add_epi32(rows0p_pack, _acc);
            _acc = _mm_add_epi32(rows1p_pack, _acc);

            // shift right
            __m128i _acc16 = _mm_srli_epi32(_acc, 2);
            
            int* buffer_acc = (int*)&_acc16;
	        for(size_t i = 0; i < 4; ++i){
		        // std::cout << buffer_acc[i] << std::endl;
	    	    buffer_acc[i] = (unsigned char)(buffer_acc[i] >> 16);
		        *(Dp+i) = buffer_acc[i];
	        }

            Dp += 4;
            rows0p += 4;
            rows1p += 4;
        }
        for (; remain; --remain)
        {
            //             D[x] = (rows0[x]*b0 + rows1[x]*b1) >> INTER_RESIZE_COEF_BITS;
            *Dp++ = (unsigned char)(((short)((b0 * (short)(*rows0p++)) >> 16) + (short)((b1 * (short)(*rows1p++)) >> 16) + 2) >> 2);
        }

        ibeta += 2;
    }

    delete[] buf;
}

void resize_bilinear_c3(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    const int INTER_RESIZE_COEF_BITS = 11;
    const int INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS;
    //     const int ONE=INTER_RESIZE_COEF_SCALE;

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
                
                __m128i _S1 = _mm_set_epi16(0, 0, *(S1p+5), *(S1p+2), *(S1p+4), *(S1p+1), *(S1p+3), *(S1p));
                __m128i _a0a1 = _mm_set_epi16(0, 0, a1, a0, a1, a0, a1, a0);

                __m128i _rows1 = _mm_madd_epi16(_S1, _a0a1);
                __m128i _rows1_sr4 = _mm_srli_epi32(_rows1, 4);

                int* temp_sr4 = (int*)&_rows1_sr4;
                rows1p[0] = (short)*(temp_sr4);
                rows1p[1] = (short)*(temp_sr4+1);
                rows1p[2] = (short)*(temp_sr4+2);

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

                __m128i _S0 = _mm_set_epi16(0, 0, *(S0p+5), *(S0p+2), *(S0p+4), *(S0p+1), *(S0p+3), *(S0p));
                __m128i _S1 = _mm_set_epi16(0, 0, *(S1p+5), *(S1p+2), *(S1p+4), *(S1p+1), *(S1p+3), *(S1p));
                __m128i _a0a1 = _mm_set_epi16(0, 0, a1, a0, a1, a0, a1, a0);

                __m128i _rows0 = _mm_madd_epi16(_S0, _a0a1);
                __m128i _rows1 = _mm_madd_epi16(_S1, _a0a1);
                __m128i _rows0_sr4 = _mm_srli_epi32(_rows0, 4);
                __m128i _rows1_sr4 = _mm_srli_epi32(_rows1, 4);

                int* temp0_sr4 = (int*)&_rows0_sr4;
                int* temp1_sr4 = (int*)&_rows1_sr4;
                rows0p[0] = (short)*(temp0_sr4);
                rows1p[0] = (short)*(temp1_sr4);
                rows0p[1] = (short)*(temp0_sr4+1);
                rows1p[1] = (short)*(temp1_sr4+1);
                rows0p[2] = (short)*(temp0_sr4+2);
                rows1p[2] = (short)*(temp1_sr4+2);

                ialphap += 2;
                rows0p += 3;
                rows1p += 3;
            }
        }

        prev_sy1 = sy;

        // vresize
        int b0 = (int)ibeta[0];
        int b1 = (int)ibeta[1];

        short* rows0p = rows0;
        short* rows1p = rows1;
        unsigned char* Dp = dst + stride * (dy);

        int nn = (w * 3) >> 2;
        int remain = (w * 3) - (nn << 2);

        __m128i _b0 = _mm_set1_epi32(b0);
        __m128i _b1 = _mm_set1_epi32(b1);
        __m128i _v2 = _mm_set1_epi32(2);
        for (; nn > 0; --nn)
        {
            __m128i rows0p_0_sr4 = _mm_set_epi32(0, (int)*(rows0p+2), 0, (int)*(rows0p));
            __m128i rows0p_1_sr4 = _mm_set_epi32(0, (int)*(rows0p+3), 0, (int)*(rows0p+1));
            __m128i rows1p_0_sr4 = _mm_set_epi32(0, (int)*(rows1p+2), 0, (int)*(rows1p));
            __m128i rows1p_1_sr4 = _mm_set_epi32(0, (int)*(rows1p+3), 0, (int)*(rows1p+1));

            __m128i _rows0p_0_sr4_mb0 = _mm_mul_epu32(rows0p_0_sr4, _b0);
            __m128i _rows0p_1_sr4_mb0 = _mm_mul_epu32(rows0p_1_sr4, _b0);
            __m128i _rows1p_0_sr4_mb1 = _mm_mul_epu32(rows1p_0_sr4, _b1);
            __m128i _rows1p_1_sr4_mb1 = _mm_mul_epu32(rows1p_1_sr4, _b1);

            __m128i rows0p_0_unpack = _mm_unpacklo_epi32(_rows0p_0_sr4_mb0, _rows0p_1_sr4_mb0);
            __m128i rows1p_0_unpack = _mm_unpacklo_epi32(_rows1p_0_sr4_mb1, _rows1p_1_sr4_mb1);
            __m128i rows0p_1_unpack = _mm_unpackhi_epi32(_rows0p_0_sr4_mb0, _rows0p_1_sr4_mb0);
            __m128i rows1p_1_unpack = _mm_unpackhi_epi32(_rows1p_0_sr4_mb1, _rows1p_1_sr4_mb1);
            __m128i rows0p_pack = _mm_unpacklo_epi64(rows0p_0_unpack, rows0p_1_unpack);
            __m128i rows1p_pack = _mm_unpacklo_epi64(rows1p_0_unpack, rows1p_1_unpack);
            __m128i _acc = _v2;
            _acc = _mm_add_epi32(rows0p_pack, _acc);
            _acc = _mm_add_epi32(rows1p_pack, _acc);

            // shift right
            __m128i _acc16 = _mm_srli_epi32(_acc, 2);
            
            int* buffer_acc = (int*)&_acc16;
	        for(size_t i = 0; i < 4; ++i){
		        // std::cout << buffer_acc[i] << std::endl;
	    	    buffer_acc[i] = (unsigned char)(buffer_acc[i] >> 16);
		        *(Dp+i) = buffer_acc[i];
	        }

            Dp += 4;
            rows0p += 4;
            rows1p += 4;
        }
        for (; remain; --remain)
        {
            //             D[x] = (rows0[x]*b0 + rows1[x]*b1) >> INTER_RESIZE_COEF_BITS;
            *Dp++ = (unsigned char)(((short)((b0 * (short)(*rows0p++)) >> 16) + (short)((b1 * (short)(*rows1p++)) >> 16) + 2) >> 2);
        }
        
        ibeta += 2;
    }
    
    delete[] buf;
}

void resize_bilinear_c4(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    const int INTER_RESIZE_COEF_BITS = 11;
    const int INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS;
    //     const int ONE=INTER_RESIZE_COEF_SCALE;

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

                __m128i _a0a1 = _mm_set_epi16(a1, a0, a1, a0, a1, a0, a1, a0);
                __m128i _S1 = _mm_set_epi16((short)*(S1p + 7), (short)*(S1p + 3), (short)*(S1p + 6), (short)*(S1p + 2), 
                                            (short)*(S1p + 5), (short)*(S1p + 1), (short)*(S1p + 4), (short)*(S1p));
                
                __m128i _rows1 = _mm_madd_epi16(_S1, _a0a1);
                __m128i _rows1_sr4 = _mm_srli_epi32(_rows1, 4);

                int* temp_sr4 = (int*)&_rows1_sr4;
                rows1p[0] = (short)*(temp_sr4);
                rows1p[1] = (short)*(temp_sr4+1);
                rows1p[2] = (short)*(temp_sr4+2);
                rows1p[3] = (short)*(temp_sr4+3);

		// printf("%d %d %d %d\n", rows1p[0], rows1p[1], rows1p[2], rows1p[3]);
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

                __m128i _a0a1 = _mm_set_epi16(a1, a0, a1, a0, a1, a0, a1, a0);
                __m128i _S0 = _mm_set_epi16((short)*(S0p + 7), (short)*(S0p + 3), (short)*(S0p + 6), (short)*(S0p + 2),
                                            (short)*(S0p + 5), (short)*(S0p + 1), (short)*(S0p + 4), (short)*(S0p));
                __m128i _S1 = _mm_set_epi16((short)*(S1p + 7), (short)*(S1p + 3), (short)*(S1p + 6), (short)*(S1p + 2),
                                            (short)*(S1p + 5), (short)*(S1p + 1), (short)*(S1p + 4), (short)*(S1p));

                __m128i _rows0 = _mm_madd_epi16(_S0, _a0a1);
                __m128i _rows1 = _mm_madd_epi16(_S1, _a0a1);
                __m128i _rows0_sr4 = _mm_srli_epi32(_rows0, 4);
                __m128i _rows1_sr4 = _mm_srli_epi32(_rows1, 4);

                int* temp0_sr4 = (int*)&_rows0_sr4;
                int* temp1_sr4 = (int*)&_rows1_sr4;
                rows0p[0] = (short)*(temp0_sr4);
                rows1p[0] = (short)*(temp1_sr4);
                rows0p[1] = (short)*(temp0_sr4 + 1);
                rows1p[1] = (short)*(temp1_sr4 + 1);
                rows0p[2] = (short)*(temp0_sr4 + 2);
                rows1p[2] = (short)*(temp1_sr4 + 2);
                rows0p[3] = (short)*(temp0_sr4 + 3);
                rows1p[3] = (short)*(temp1_sr4 + 3);

                /* rows0p[0] = (S0p[0] * a0 + S0p[4] * a1) >> 4;
                rows0p[1] = (S0p[1] * a0 + S0p[5] * a1) >> 4;
                rows0p[2] = (S0p[2] * a0 + S0p[6] * a1) >> 4;
                rows0p[3] = (S0p[3] * a0 + S0p[7] * a1) >> 4;
                rows1p[0] = (S1p[0] * a0 + S1p[4] * a1) >> 4;
                rows1p[1] = (S1p[1] * a0 + S1p[5] * a1) >> 4;
                rows1p[2] = (S1p[2] * a0 + S1p[6] * a1) >> 4;
                rows1p[3] = (S1p[3] * a0 + S1p[7] * a1) >> 4; */

		// printf("%d %d %d %d\n", rows1p[0], rows1p[1], rows1p[2], rows1p[3]);
                ialphap += 2;
                rows0p += 4;
                rows1p += 4;
            }
        }

        prev_sy1 = sy;

        // vresize
        short b0 = ibeta[0];
        short b1 = ibeta[1];

        short* rows0p = rows0;
        short* rows1p = rows1;
        unsigned char* Dp = dst + stride * (dy);

        int nn = (w * 4) >> 2;
        int remain = (w * 4) - (nn << 2);

        __m128i _b0 = _mm_set1_epi32(b0);
        __m128i _b1 = _mm_set1_epi32(b1);
        __m128i _v2 = _mm_set1_epi32(2);
        for (; nn > 0; --nn)
        {
            __m128i rows0p_0_sr4 = _mm_set_epi32(0, (int)*(rows0p+2), 0, (int)*(rows0p));
            __m128i rows0p_1_sr4 = _mm_set_epi32(0, (int)*(rows0p+3), 0, (int)*(rows0p+1));
            __m128i rows1p_0_sr4 = _mm_set_epi32(0, (int)*(rows1p+2), 0, (int)*(rows1p));
            __m128i rows1p_1_sr4 = _mm_set_epi32(0, (int)*(rows1p+3), 0, (int)*(rows1p+1));

            __m128i _rows0p_0_sr4_mb0 = _mm_mul_epu32(rows0p_0_sr4, _b0);
            __m128i _rows0p_1_sr4_mb0 = _mm_mul_epu32(rows0p_1_sr4, _b0);
            __m128i _rows1p_0_sr4_mb1 = _mm_mul_epu32(rows1p_0_sr4, _b1);
            __m128i _rows1p_1_sr4_mb1 = _mm_mul_epu32(rows1p_1_sr4, _b1);

            __m128i rows0p_0_unpack = _mm_unpacklo_epi32(_rows0p_0_sr4_mb0, _rows0p_1_sr4_mb0);
            __m128i rows1p_0_unpack = _mm_unpacklo_epi32(_rows1p_0_sr4_mb1, _rows1p_1_sr4_mb1);
            __m128i rows0p_1_unpack = _mm_unpackhi_epi32(_rows0p_0_sr4_mb0, _rows0p_1_sr4_mb0);
            __m128i rows1p_1_unpack = _mm_unpackhi_epi32(_rows1p_0_sr4_mb1, _rows1p_1_sr4_mb1);
            __m128i rows0p_pack = _mm_unpacklo_epi64(rows0p_0_unpack, rows0p_1_unpack);
            __m128i rows1p_pack = _mm_unpacklo_epi64(rows1p_0_unpack, rows1p_1_unpack);
            __m128i _acc = _v2;
            _acc = _mm_add_epi32(rows0p_pack, _acc);
            _acc = _mm_add_epi32(rows1p_pack, _acc);

             // shift right
            __m128i _acc16 = _mm_srli_epi32(_acc, 2);
            
            int* buffer_acc = (int*)&_acc16;
	        for(size_t i = 0; i < 4; ++i){
		        // std::cout << buffer_acc[i] << std::endl;
	    	    buffer_acc[i] = (unsigned char)(buffer_acc[i] >> 16);
		        *(Dp+i) = buffer_acc[i];
	        }

            Dp += 4;
            rows0p += 4;
            rows1p += 4;
        }
        for (; remain; --remain)
        {
            //             D[x] = (rows0[x]*b0 + rows1[x]*b1) >> INTER_RESIZE_COEF_BITS;
            *Dp++ = (unsigned char)(((short)((b0 * (short)(*rows0p++)) >> 16) + (short)((b1 * (short)(*rows1p++)) >> 16) + 2) >> 2);
        }

        ibeta += 2;
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
