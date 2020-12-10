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

#include "mat.h"
#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON
#include <math.h>
#include "platform.h"

namespace ncnn {

#if NCNN_PIXEL_AFFINE
void get_rotation_matrix(float angle, float scale, float dx, float dy, float* tm)
{
    angle *= (float)(3.14159265358979323846 / 180);
    float alpha = cos(angle) * scale;
    float beta = sin(angle) * scale;

    tm[0] = alpha;
    tm[1] = beta;
    tm[2] = (1.f - alpha) * dx - beta * dy;
    tm[3] = -beta;
    tm[4] = alpha;
    tm[5] = beta * dx + (1.f - alpha) * dy;
}

void get_affine_transform(const float* points_from, const float* points_to, int num_point, float* tm)
{
    float ma[4][4] = {{0.f}};
    float mb[4] = {0.f};
    float mm[4];

    for (int i = 0; i < num_point; i++)
    {
        ma[0][0] += points_from[0] * points_from[0] + points_from[1] * points_from[1];
        ma[0][2] += points_from[0];
        ma[0][3] += points_from[1];

        mb[0] += points_from[0] * points_to[0] + points_from[1] * points_to[1];
        mb[1] += points_from[0] * points_to[1] - points_from[1] * points_to[0];
        mb[2] += points_to[0];
        mb[3] += points_to[1];

        points_from += 2;
        points_to += 2;
    }

    ma[1][1] = ma[0][0];
    ma[2][1] = ma[1][2] = -ma[0][3];
    ma[3][1] = ma[1][3] = ma[2][0] = ma[0][2];
    ma[2][2] = ma[3][3] = (float)num_point;
    ma[3][0] = ma[0][3];

    // MM = inv(A) * B
    // matrix 4x4 invert by https://github.com/willnode/N-Matrix-Programmer
    // suppose the user provide valid points combination
    // I have not taken det == zero into account here   :>  --- nihui
    float mai[4][4];
    float det;
    // clang-format off
    // *INDENT-OFF*
    {
        float A2323 = ma[2][2] * ma[3][3] - ma[2][3] * ma[3][2];
        float A1323 = ma[2][1] * ma[3][3] - ma[2][3] * ma[3][1];
        float A1223 = ma[2][1] * ma[3][2] - ma[2][2] * ma[3][1];
        float A0323 = ma[2][0] * ma[3][3] - ma[2][3] * ma[3][0];
        float A0223 = ma[2][0] * ma[3][2] - ma[2][2] * ma[3][0];
        float A0123 = ma[2][0] * ma[3][1] - ma[2][1] * ma[3][0];
        float A2313 = ma[1][2] * ma[3][3] - ma[1][3] * ma[3][2];
        float A1313 = ma[1][1] * ma[3][3] - ma[1][3] * ma[3][1];
        float A1213 = ma[1][1] * ma[3][2] - ma[1][2] * ma[3][1];
        float A2312 = ma[1][2] * ma[2][3] - ma[1][3] * ma[2][2];
        float A1312 = ma[1][1] * ma[2][3] - ma[1][3] * ma[2][1];
        float A1212 = ma[1][1] * ma[2][2] - ma[1][2] * ma[2][1];
        float A0313 = ma[1][0] * ma[3][3] - ma[1][3] * ma[3][0];
        float A0213 = ma[1][0] * ma[3][2] - ma[1][2] * ma[3][0];
        float A0312 = ma[1][0] * ma[2][3] - ma[1][3] * ma[2][0];
        float A0212 = ma[1][0] * ma[2][2] - ma[1][2] * ma[2][0];
        float A0113 = ma[1][0] * ma[3][1] - ma[1][1] * ma[3][0];
        float A0112 = ma[1][0] * ma[2][1] - ma[1][1] * ma[2][0];

        det = ma[0][0] * (ma[1][1] * A2323 - ma[1][2] * A1323 + ma[1][3] * A1223)
            - ma[0][1] * (ma[1][0] * A2323 - ma[1][2] * A0323 + ma[1][3] * A0223)
            + ma[0][2] * (ma[1][0] * A1323 - ma[1][1] * A0323 + ma[1][3] * A0123)
            - ma[0][3] * (ma[1][0] * A1223 - ma[1][1] * A0223 + ma[1][2] * A0123);

        det = 1.f / det;

        mai[0][0] =   (ma[1][1] * A2323 - ma[1][2] * A1323 + ma[1][3] * A1223);
        mai[0][1] = - (ma[0][1] * A2323 - ma[0][2] * A1323 + ma[0][3] * A1223);
        mai[0][2] =   (ma[0][1] * A2313 - ma[0][2] * A1313 + ma[0][3] * A1213);
        mai[0][3] = - (ma[0][1] * A2312 - ma[0][2] * A1312 + ma[0][3] * A1212);
        mai[1][0] = - (ma[1][0] * A2323 - ma[1][2] * A0323 + ma[1][3] * A0223);
        mai[1][1] =   (ma[0][0] * A2323 - ma[0][2] * A0323 + ma[0][3] * A0223);
        mai[1][2] = - (ma[0][0] * A2313 - ma[0][2] * A0313 + ma[0][3] * A0213);
        mai[1][3] =   (ma[0][0] * A2312 - ma[0][2] * A0312 + ma[0][3] * A0212);
        mai[2][0] =   (ma[1][0] * A1323 - ma[1][1] * A0323 + ma[1][3] * A0123);
        mai[2][1] = - (ma[0][0] * A1323 - ma[0][1] * A0323 + ma[0][3] * A0123);
        mai[2][2] =   (ma[0][0] * A1313 - ma[0][1] * A0313 + ma[0][3] * A0113);
        mai[2][3] = - (ma[0][0] * A1312 - ma[0][1] * A0312 + ma[0][3] * A0112);
        mai[3][0] = - (ma[1][0] * A1223 - ma[1][1] * A0223 + ma[1][2] * A0123);
        mai[3][1] =   (ma[0][0] * A1223 - ma[0][1] * A0223 + ma[0][2] * A0123);
        mai[3][2] = - (ma[0][0] * A1213 - ma[0][1] * A0213 + ma[0][2] * A0113);
        mai[3][3] =   (ma[0][0] * A1212 - ma[0][1] * A0212 + ma[0][2] * A0112);
    }
    // *INDENT-ON*
    // clang-format on

    mm[0] = det * (mai[0][0] * mb[0] + mai[0][1] * mb[1] + mai[0][2] * mb[2] + mai[0][3] * mb[3]);
    mm[1] = det * (mai[1][0] * mb[0] + mai[1][1] * mb[1] + mai[1][2] * mb[2] + mai[1][3] * mb[3]);
    mm[2] = det * (mai[2][0] * mb[0] + mai[2][1] * mb[1] + mai[2][2] * mb[2] + mai[2][3] * mb[3]);
    mm[3] = det * (mai[3][0] * mb[0] + mai[3][1] * mb[1] + mai[3][2] * mb[2] + mai[3][3] * mb[3]);

    tm[0] = tm[4] = mm[0];
    tm[1] = -mm[1];
    tm[3] = mm[1];
    tm[2] = mm[2];
    tm[5] = mm[3];
}

void invert_affine_transform(const float* tm, float* tm_inv)
{
    float D = tm[0] * tm[4] - tm[1] * tm[3];
    D = D != 0.f ? 1.f / D : 0.f;

    float A11 = tm[4] * D;
    float A22 = tm[0] * D;
    float A12 = -tm[1] * D;
    float A21 = -tm[3] * D;
    float b1 = -A11 * tm[2] - A12 * tm[5];
    float b2 = -A21 * tm[2] - A22 * tm[5];

    tm_inv[0] = A11;
    tm_inv[1] = A12;
    tm_inv[2] = b1;
    tm_inv[3] = A21;
    tm_inv[4] = A22;
    tm_inv[5] = b2;
}

void warpaffine_bilinear_c1(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, const float* tm, int v)
{
    return warpaffine_bilinear_c1(src, srcw, srch, srcw, dst, w, h, w, tm, v);
}

void warpaffine_bilinear_c2(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, const float* tm, int v)
{
    return warpaffine_bilinear_c2(src, srcw, srch, srcw * 2, dst, w, h, w * 2, tm, v);
}

void warpaffine_bilinear_c3(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, const float* tm, int v)
{
    return warpaffine_bilinear_c3(src, srcw, srch, srcw * 3, dst, w, h, w * 3, tm, v);
}

void warpaffine_bilinear_c4(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, const float* tm, int v)
{
    return warpaffine_bilinear_c4(src, srcw, srch, srcw * 4, dst, w, h, w * 4, tm, v);
}

void warpaffine_bilinear_c1(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, const float* tm, int v)
{
    const unsigned char border_color = (unsigned char)v;
    const int wgap = stride - w;

    const unsigned char* src0 = src;
    unsigned char* dst0 = dst;

    int y = 0;
    for (; y < h; y++)
    {
        int x = 0;
        for (; x < w; x++)
        {
            float fx = tm[0] * x + tm[1] * y + tm[2];
            float fy = tm[3] * x + tm[4] * y + tm[5];

            if (fx < 0 || fx >= srcw - 1 || fy < 0 || fy >= srch - 1)
            {
                if (v != -233)
                {
                    *dst0 = border_color;
                }
            }
            else
            {
                // interp at fx fy
                int sx = static_cast<int>(floor(fx));
                fx -= sx;

                int sy = static_cast<int>(floor(fy));
                fy -= sy;

                const unsigned char* a0 = src0 + srcstride * sy + sx;
                const unsigned char* a1 = src0 + srcstride * sy + sx + 1;
                const unsigned char* b0 = src0 + srcstride * (sy + 1) + sx;
                const unsigned char* b1 = src0 + srcstride * (sy + 1) + sx + 1;

                *dst0 = (unsigned char)((*a0 * (1.f - fx) + *a1 * fx) * (1.f - fy) + (*b0 * (1.f - fx) + *b1 * fx) * fy);
            }

            dst0 += 1;
        }

        dst0 += wgap;
    }
}

void warpaffine_bilinear_c2(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, const float* tm, int v)
{
    const unsigned char border_color = (unsigned char)v;
    const int wgap = stride - w * 2;

    const unsigned char* src0 = src;
    unsigned char* dst0 = dst;

    int y = 0;
    for (; y < h; y++)
    {
        int x = 0;
        for (; x < w; x++)
        {
            float fx = tm[0] * x + tm[1] * y + tm[2];
            float fy = tm[3] * x + tm[4] * y + tm[5];

            if (fx < 0 || fx >= srcw - 1 || fy < 0 || fy >= srch - 1)
            {
                if (v != -233)
                {
                    dst0[0] = border_color;
                    dst0[1] = border_color;
                }
            }
            else
            {
                // interp at fx fy
                int sx = static_cast<int>(floor(fx));
                fx -= sx;

                int sy = static_cast<int>(floor(fy));
                fy -= sy;

                const unsigned char* a0 = src0 + srcstride * sy + sx * 2;
                const unsigned char* a1 = src0 + srcstride * sy + sx * 2 + 2;
                const unsigned char* b0 = src0 + srcstride * (sy + 1) + sx * 2;
                const unsigned char* b1 = src0 + srcstride * (sy + 1) + sx * 2 + 2;

                dst0[0] = (unsigned char)((a0[0] * (1.f - fx) + a1[0] * fx) * (1.f - fy) + (b0[0] * (1.f - fx) + b1[0] * fx) * fy);
                dst0[1] = (unsigned char)((a0[1] * (1.f - fx) + a1[1] * fx) * (1.f - fy) + (b0[1] * (1.f - fx) + b1[1] * fx) * fy);
            }

            dst0 += 2;
        }

        dst0 += wgap;
    }
}

void warpaffine_bilinear_c3(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, const float* tm, int v)
{
    const unsigned char border_color = (unsigned char)v;
    const int wgap = stride - w * 3;

    const unsigned char* src0 = src;
    unsigned char* dst0 = dst;

    int y = 0;
    for (; y < h; y++)
    {
        int x = 0;
        for (; x < w; x++)
        {
            float fx = tm[0] * x + tm[1] * y + tm[2];
            float fy = tm[3] * x + tm[4] * y + tm[5];

            if (fx < 0 || fx >= srcw - 1 || fy < 0 || fy >= srch - 1)
            {
                if (v != -233)
                {
                    dst0[0] = border_color;
                    dst0[1] = border_color;
                    dst0[2] = border_color;
                }
            }
            else
            {
                // interp at fx fy
                int sx = static_cast<int>(floor(fx));
                fx -= sx;

                int sy = static_cast<int>(floor(fy));
                fy -= sy;

                const unsigned char* a0 = src0 + srcstride * sy + sx * 3;
                const unsigned char* a1 = src0 + srcstride * sy + sx * 3 + 3;
                const unsigned char* b0 = src0 + srcstride * (sy + 1) + sx * 3;
                const unsigned char* b1 = src0 + srcstride * (sy + 1) + sx * 3 + 3;

                dst0[0] = (unsigned char)((a0[0] * (1.f - fx) + a1[0] * fx) * (1.f - fy) + (b0[0] * (1.f - fx) + b1[0] * fx) * fy);
                dst0[1] = (unsigned char)((a0[1] * (1.f - fx) + a1[1] * fx) * (1.f - fy) + (b0[1] * (1.f - fx) + b1[1] * fx) * fy);
                dst0[2] = (unsigned char)((a0[2] * (1.f - fx) + a1[2] * fx) * (1.f - fy) + (b0[2] * (1.f - fx) + b1[2] * fx) * fy);
            }

            dst0 += 3;
        }

        dst0 += wgap;
    }
}

void warpaffine_bilinear_c4(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, const float* tm, int v)
{
    const unsigned char border_color = (unsigned char)v;
    const int wgap = stride - w * 4;

    const unsigned char* src0 = src;
    unsigned char* dst0 = dst;

    int y = 0;
    for (; y < h; y++)
    {
        int x = 0;
        for (; x < w; x++)
        {
            float fx = tm[0] * x + tm[1] * y + tm[2];
            float fy = tm[3] * x + tm[4] * y + tm[5];

            if (fx < 0 || fx >= srcw - 1 || fy < 0 || fy >= srch - 1)
            {
                if (v != -233)
                {
                    dst0[0] = border_color;
                    dst0[1] = border_color;
                    dst0[2] = border_color;
                    dst0[3] = border_color;
                }
            }
            else
            {
                // interp at fx fy
                int sx = static_cast<int>(floor(fx));
                fx -= sx;

                int sy = static_cast<int>(floor(fy));
                fy -= sy;

                const unsigned char* a0 = src0 + srcstride * sy + sx * 4;
                const unsigned char* a1 = src0 + srcstride * sy + sx * 4 + 4;
                const unsigned char* b0 = src0 + srcstride * (sy + 1) + sx * 4;
                const unsigned char* b1 = src0 + srcstride * (sy + 1) + sx * 4 + 4;

                dst0[0] = (unsigned char)((a0[0] * (1.f - fx) + a1[0] * fx) * (1.f - fy) + (b0[0] * (1.f - fx) + b1[0] * fx) * fy);
                dst0[1] = (unsigned char)((a0[1] * (1.f - fx) + a1[1] * fx) * (1.f - fy) + (b0[1] * (1.f - fx) + b1[1] * fx) * fy);
                dst0[2] = (unsigned char)((a0[2] * (1.f - fx) + a1[2] * fx) * (1.f - fy) + (b0[2] * (1.f - fx) + b1[2] * fx) * fy);
                dst0[3] = (unsigned char)((a0[3] * (1.f - fx) + a1[3] * fx) * (1.f - fy) + (b0[3] * (1.f - fx) + b1[3] * fx) * fy);
            }

            dst0 += 4;
        }

        dst0 += wgap;
    }
}

void warpaffine_bilinear_yuv420sp(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, const float* tm, int v)
{
    // assert srcw % 2 == 0
    // assert srch % 2 == 0
    // assert w % 2 == 0
    // assert h % 2 == 0

    const unsigned char* srcY = src;
    unsigned char* dstY = dst;
    warpaffine_bilinear_c1(srcY, srcw, srch, dstY, w, h, tm, v);

    const unsigned char* srcUV = src + srcw * srch;
    unsigned char* dstUV = dst + w * h;
    warpaffine_bilinear_c2(srcUV, srcw / 2, srch / 2, dstUV, w / 2, h / 2, tm, v);
}
#endif // NCNN_PIXEL_AFFINE

} // namespace ncnn
