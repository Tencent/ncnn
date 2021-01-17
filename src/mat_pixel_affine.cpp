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
#include <limits.h>
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

void warpaffine_bilinear_c1(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, const float* tm, int type, unsigned int v)
{
    return warpaffine_bilinear_c1(src, srcw, srch, srcw, dst, w, h, w, tm, type, v);
}

void warpaffine_bilinear_c2(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, const float* tm, int type, unsigned int v)
{
    return warpaffine_bilinear_c2(src, srcw, srch, srcw * 2, dst, w, h, w * 2, tm, type, v);
}

void warpaffine_bilinear_c3(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, const float* tm, int type, unsigned int v)
{
    return warpaffine_bilinear_c3(src, srcw, srch, srcw * 3, dst, w, h, w * 3, tm, type, v);
}

void warpaffine_bilinear_c4(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, const float* tm, int type, unsigned int v)
{
    return warpaffine_bilinear_c4(src, srcw, srch, srcw * 4, dst, w, h, w * 4, tm, type, v);
}

class Boundary
{
public:
    int start_outer;
    int start_inner;
    int end_inner;
    int end_outer;
};

static void resolve_boundary(int srcw, int srch, int w, int h, const float* tm, Boundary& y_bound, std::vector<Boundary>& x_bounds)
{
    // resolve src bounding box
    float src_x[4];
    float src_y[4];
    {
        src_x[0] = tm[2];
        src_y[0] = tm[5];
        src_x[1] = tm[0] * w + tm[2];
        src_y[1] = tm[3] * w + tm[5];
        src_x[2] = tm[1] * h + tm[2];
        src_y[2] = tm[4] * h + tm[5];
        src_x[3] = tm[0] * w + tm[1] * h + tm[2];
        src_y[3] = tm[3] * w + tm[4] * h + tm[5];
    }

    if (src_x[0] >= 0 && src_x[1] >= 0 && src_x[2] >= 0 && src_x[3] >= 0
            && src_x[0] < srcw - 1 && src_x[1] < srcw - 1 && src_x[2] < srcw - 1 && src_x[3] < srcw - 1
            && src_y[0] >= 0 && src_y[1] >= 0 && src_y[2] >= 0 && src_y[3] >= 0
            && src_y[0] < srch - 1 && src_y[1] < srch - 1 && src_y[2] < srch - 1 && src_y[3] < srch - 1)
    {
        // all pixels are inside src image

        // assign start end
        y_bound.start_outer = 0;
        y_bound.start_inner = 0;
        y_bound.end_inner = h;
        y_bound.end_outer = h;

        x_bounds.resize(h);
        for (int y = 0; y < h; y++)
        {
            x_bounds[y].start_outer = 0;
            x_bounds[y].start_inner = 0;
            x_bounds[y].end_inner = w;
            x_bounds[y].end_outer = w;
        }

        return;
    }

    // resolve dst bounding box
    float eps = 0.0001f;
    float dst_outer_x[4];
    float dst_outer_y[4];
    float dst_inner_x[4];
    float dst_inner_y[4];
    {
        float tm_inv[6];
        invert_affine_transform(tm, tm_inv);

        dst_outer_x[0] = tm_inv[0] * (-1 - eps) + tm_inv[1] * (-1 - eps) + tm_inv[2];
        dst_outer_y[0] = tm_inv[3] * (-1 - eps) + tm_inv[4] * (-1 - eps) + tm_inv[5];
        dst_outer_x[1] = tm_inv[0] * (srcw + eps) + tm_inv[1] * (-1 - eps) + tm_inv[2];
        dst_outer_y[1] = tm_inv[3] * (srcw + eps) + tm_inv[4] * (-1 - eps) + tm_inv[5];
        dst_outer_x[2] = tm_inv[0] * (-1 - eps) + tm_inv[1] * (srch + eps) + tm_inv[2];
        dst_outer_y[2] = tm_inv[3] * (-1 - eps) + tm_inv[4] * (srch + eps) + tm_inv[5];
        dst_outer_x[3] = tm_inv[0] * (srcw + eps) + tm_inv[1] * (srch + eps) + tm_inv[2];
        dst_outer_y[3] = tm_inv[3] * (srcw + eps) + tm_inv[4] * (srch + eps) + tm_inv[5];

        dst_inner_x[0] = tm_inv[0] * (0 + eps) + tm_inv[1] * (0 + eps) + tm_inv[2];
        dst_inner_y[0] = tm_inv[3] * (0 + eps) + tm_inv[4] * (0 + eps) + tm_inv[5];
        dst_inner_x[1] = tm_inv[0] * (srcw - 1 - eps) + tm_inv[1] * (0 + eps) + tm_inv[2];
        dst_inner_y[1] = tm_inv[3] * (srcw - 1 - eps) + tm_inv[4] * (0 + eps) + tm_inv[5];
        dst_inner_x[2] = tm_inv[0] * (0 + eps) + tm_inv[1] * (srch - 1 - eps) + tm_inv[2];
        dst_inner_y[2] = tm_inv[3] * (0 + eps) + tm_inv[4] * (srch - 1 - eps) + tm_inv[5];
        dst_inner_x[3] = tm_inv[0] * (srcw - 1 - eps) + tm_inv[1] * (srch - 1 - eps) + tm_inv[2];
        dst_inner_y[3] = tm_inv[3] * (srcw - 1 - eps) + tm_inv[4] * (srch - 1 - eps) + tm_inv[5];
    }

    // sort by y
    for (int i = 0; i < 4; i++)
    {
        for (int j = i + 1; j < 4; j++)
        {
            if (dst_outer_y[i] > dst_outer_y[j])
            {
                std::swap(dst_outer_x[i], dst_outer_x[j]);
                std::swap(dst_outer_y[i], dst_outer_y[j]);
            }
            if (dst_inner_y[i] > dst_inner_y[j])
            {
                std::swap(dst_inner_x[i], dst_inner_x[j]);
                std::swap(dst_inner_y[i], dst_inner_y[j]);
            }
        }
    }

    float dst_outer_top_x = dst_outer_x[0];
    float dst_outer_top_y = dst_outer_y[0];
    float dst_outer_bottom_x = dst_outer_x[3];
    float dst_outer_bottom_y = dst_outer_y[3];
    float dst_inner_top_x = dst_inner_x[0];
    float dst_inner_top_y = dst_inner_y[0];
    float dst_inner_bottom_x = dst_inner_x[3];
    float dst_inner_bottom_y = dst_inner_y[3];

    // sort by x
    for (int i = 0; i < 4; i++)
    {
        for (int j = i + 1; j < 4; j++)
        {
            if (dst_outer_x[i] > dst_outer_x[j])
            {
                std::swap(dst_outer_x[i], dst_outer_x[j]);
                std::swap(dst_outer_y[i], dst_outer_y[j]);
            }
            if (dst_inner_x[i] > dst_inner_x[j])
            {
                std::swap(dst_inner_x[i], dst_inner_x[j]);
                std::swap(dst_inner_y[i], dst_inner_y[j]);
            }
        }
    }

    float dst_outer_left_x = dst_outer_x[0];
    float dst_outer_left_y = dst_outer_y[0];
    float dst_outer_right_x = dst_outer_x[3];
    float dst_outer_right_y = dst_outer_y[3];
    float dst_inner_left_x = dst_inner_x[0];
    float dst_inner_left_y = dst_inner_y[0];
    float dst_inner_right_x = dst_inner_x[3];
    float dst_inner_right_y = dst_inner_y[3];

    //     fprintf(stderr, "To %f %f\n", dst_outer_top_x, dst_outer_top_y);
    //     fprintf(stderr, "Bo %f %f\n", dst_outer_bottom_x, dst_outer_bottom_y);
    //     fprintf(stderr, "Lo %f %f\n", dst_outer_left_x, dst_outer_left_y);
    //     fprintf(stderr, "Ro %f %f\n", dst_outer_right_x, dst_outer_right_y);
    //
    //     fprintf(stderr, "Ti %f %f\n", dst_inner_top_x, dst_inner_top_y);
    //     fprintf(stderr, "Bi %f %f\n", dst_inner_bottom_x, dst_inner_bottom_y);
    //     fprintf(stderr, "Li %f %f\n", dst_inner_left_x, dst_inner_left_y);
    //     fprintf(stderr, "Ri %f %f\n", dst_inner_right_x, dst_inner_right_y);

    // assign start end
    y_bound.start_outer = std::min(std::max((int)floor(dst_outer_top_y), 0), h);
    y_bound.start_inner = std::min(std::max((int)ceil(dst_inner_top_y), 0), h);
    y_bound.end_inner = std::min(std::max((int)floor(dst_inner_bottom_y), 0), h);
    y_bound.end_outer = std::min(std::max((int)ceil(dst_outer_bottom_y), 0), h);

    int y_start = y_bound.start_outer;
    int y_end = y_bound.end_outer;

    x_bounds.resize(y_end - y_start);

    if (dst_outer_top_x == dst_outer_left_x && dst_outer_top_y == dst_outer_left_y && dst_outer_bottom_x == dst_outer_right_x && dst_outer_bottom_y == dst_outer_right_y)
    {
        // offset and resize without rotation
        for (int y = y_start; y < y_end; y++)
        {
            x_bounds[y - y_start].start_outer = std::min(std::max((int)floor(dst_outer_left_x), 0), w);
            x_bounds[y - y_start].start_inner = std::min(std::max((int)ceil(dst_inner_left_x), 0), w);
            x_bounds[y - y_start].end_inner = std::min(std::max((int)floor(dst_inner_right_x), 0), w);
            x_bounds[y - y_start].end_outer = std::min(std::max((int)ceil(dst_outer_right_x), 0), w);
        }

        return;
    }

    {
        int y = y_start;
        if (dst_outer_left_y > y_start)
        {
            for (; y < std::min(dst_outer_left_y, (float)y_end); y++)
            {
                int start_outer = floor((dst_outer_left_y - (y + 1)) / (dst_outer_left_y - dst_outer_top_y) * (dst_outer_top_x - dst_outer_left_x) + dst_outer_left_x);
                x_bounds[y - y_start].start_outer = std::min(std::max(start_outer, 0), w);
            }
        }
        for (; y < y_end; y++)
        {
            int start_outer = floor((y - dst_outer_left_y) / (dst_outer_bottom_y - dst_outer_left_y) * (dst_outer_bottom_x - dst_outer_left_x) + dst_outer_left_x);
            x_bounds[y - y_start].start_outer = std::min(std::max(start_outer, 0), w);
        }
    }
    {
        int y = y_start;
        if (dst_inner_left_y > y_start)
        {
            for (; y < std::min(dst_inner_left_y, (float)y_end); y++)
            {
                int start_inner = ceil((dst_inner_left_y - y) / (dst_inner_left_y - dst_inner_top_y) * (dst_inner_top_x - dst_inner_left_x) + dst_inner_left_x);
                x_bounds[y - y_start].start_inner = std::min(std::max(start_inner, 0), w);
            }
        }
        for (; y < y_end; y++)
        {
            int start_inner = ceil(((y + 1) - dst_inner_left_y) / (dst_inner_bottom_y - dst_inner_left_y) * (dst_inner_bottom_x - dst_inner_left_x) + dst_inner_left_x);
            x_bounds[y - y_start].start_inner = std::min(std::max(start_inner, 0), w);
        }
    }
    {
        int y = y_start;
        if (dst_outer_right_y > y_start)
        {
            for (; y < std::min(dst_outer_right_y, (float)y_end); y++)
            {
                int end_outer = ceil(dst_outer_right_x - (dst_outer_right_y - (y + 1)) / (dst_outer_right_y - dst_outer_top_y) * (dst_outer_right_x - dst_outer_top_x));
                x_bounds[y - y_start].end_outer = std::min(std::max(end_outer, 0), w);
            }
        }
        for (; y < y_end; y++)
        {
            int end_outer = ceil(dst_outer_right_x - (y - dst_outer_right_y) / (dst_outer_bottom_y - dst_outer_right_y) * (dst_outer_right_x - dst_outer_bottom_x));
            x_bounds[y - y_start].end_outer = std::min(std::max(end_outer, 0), w);
        }
    }
    {
        int y = y_start;
        if (dst_inner_right_y > y_start)
        {
            for (; y < std::min(dst_inner_right_y, (float)y_end); y++)
            {
                int end_inner = floor(dst_inner_right_x - (dst_inner_right_y - y) / (dst_inner_right_y - dst_inner_top_y) * (dst_inner_right_x - dst_inner_top_x));
                x_bounds[y - y_start].end_inner = std::min(std::max(end_inner, 0), w);
            }
        }
        for (; y < y_end; y++)
        {
            int end_inner = floor(dst_inner_right_x - ((y + 1) - dst_inner_right_y) / (dst_inner_bottom_y - dst_inner_right_y) * (dst_inner_right_x - dst_inner_bottom_x));
            x_bounds[y - y_start].end_inner = std::min(std::max(end_inner, 0), w);
        }
    }
}

void warpaffine_bilinear_c1(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, const float* tm, int type, unsigned int v)
{
    const unsigned char* border_color = (const unsigned char*)&v;
    const int wgap = stride - w;

    const unsigned char* src0 = src;
    unsigned char* dst0 = dst;

#define SATURATE_CAST_SHORT(X) (short)::std::min(::std::max((int)(X), SHRT_MIN), SHRT_MAX)
#define SATURATE_CAST_INT(X)   (int)::std::min(::std::max((int)(X + (X >= 0.f ? 0.5f : -0.5f)), INT_MIN), INT_MAX)

    std::vector<int> adelta(w);
    std::vector<int> bdelta(w);
    for (int x = 0; x < w; x++)
    {
        adelta[x] = SATURATE_CAST_INT(tm[0] * x * (1 << 10));
        bdelta[x] = SATURATE_CAST_INT(tm[3] * x * (1 << 10));
    }

    Boundary y_bound;
    std::vector<Boundary> x_bounds;
    resolve_boundary(srcw, srch, w, h, tm, y_bound, x_bounds);

    //     fprintf(stderr, "y %d %d %d %d\n", y_bound.start_outer, y_bound.start_inner, y_bound.end_inner, y_bound.end_outer);

    int y = 0;
    for (; y < y_bound.start_outer; y++)
    {
        for (int x = 0; x < w; x++)
        {
            if (type != -233)
            {
                dst0[0] = border_color[0];
            }

            dst0 += 1;
        }

        dst0 += wgap;
    }
    for (; y < y_bound.start_inner; y++)
    {
        const Boundary& x_bound = x_bounds[y - y_bound.start_outer];

        int X0 = SATURATE_CAST_INT(((tm[1] * (y) + tm[2]) * (1 << 10)));
        int Y0 = SATURATE_CAST_INT(((tm[4] * (y) + tm[5]) * (1 << 10)));

        int x = 0;
        for (; x < x_bound.start_outer; x++)
        {
            if (type != -233)
            {
                dst0[0] = border_color[0];
            }

            dst0 += 1;
        }
        for (; x < x_bound.start_inner; x++)
        {
            int X = X0 + adelta[x];
            int Y = Y0 + bdelta[x];

            short sx = SATURATE_CAST_SHORT((X >> 10));
            short sy = SATURATE_CAST_SHORT((Y >> 10));
            short fx = X & ((1 << 10) - 1);
            short fy = Y & ((1 << 10) - 1);

            short alpha0 = (1 << 10) - fx;
            short alpha1 = fx;

            short beta0 = (1 << 10) - fy;
            short beta1 = fy;

            short sx1 = sx + 1;
            short sy1 = sy + 1;

            const unsigned char* a0 = src0 + srcstride * sy + sx;
            const unsigned char* a1 = src0 + srcstride * sy + sx1;
            const unsigned char* b0 = src0 + srcstride * sy1 + sx;
            const unsigned char* b1 = src0 + srcstride * sy1 + sx1;

            if (sx < 0 || sx >= srcw || sy < 0 || sy >= srch)
            {
                a0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy < 0 || sy >= srch)
            {
                a1 = type != -233 ? border_color : dst0;
            }
            if (sx < 0 || sx >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b1 = type != -233 ? border_color : dst0;
            }

            dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);

            dst0 += 1;
        }
        for (; x < x_bound.end_inner; x++)
        {
            int X = X0 + adelta[x];
            int Y = Y0 + bdelta[x];

            short sx = SATURATE_CAST_SHORT((X >> 10));
            short sy = SATURATE_CAST_SHORT((Y >> 10));
            short fx = X & ((1 << 10) - 1);
            short fy = Y & ((1 << 10) - 1);

            short alpha0 = (1 << 10) - fx;
            short alpha1 = fx;

            short beta0 = (1 << 10) - fy;
            short beta1 = fy;

            short sx1 = sx + 1;
            short sy1 = sy + 1;

            const unsigned char* a0 = src0 + srcstride * sy + sx;
            const unsigned char* a1 = src0 + srcstride * sy + sx1;
            const unsigned char* b0 = src0 + srcstride * sy1 + sx;
            const unsigned char* b1 = src0 + srcstride * sy1 + sx1;

            if (sy < 0 || sy >= srch)
            {
                a0 = type != -233 ? border_color : dst0;
                a1 = type != -233 ? border_color : dst0;
            }
            if (sy1 < 0 || sy1 >= srch)
            {
                b0 = type != -233 ? border_color : dst0;
                b1 = type != -233 ? border_color : dst0;
            }

            dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);

            dst0 += 1;
        }
        for (; x < x_bound.end_outer; x++)
        {
            int X = X0 + adelta[x];
            int Y = Y0 + bdelta[x];

            short sx = SATURATE_CAST_SHORT((X >> 10));
            short sy = SATURATE_CAST_SHORT((Y >> 10));
            short fx = X & ((1 << 10) - 1);
            short fy = Y & ((1 << 10) - 1);

            short alpha0 = (1 << 10) - fx;
            short alpha1 = fx;

            short beta0 = (1 << 10) - fy;
            short beta1 = fy;

            short sx1 = sx + 1;
            short sy1 = sy + 1;

            const unsigned char* a0 = src0 + srcstride * sy + sx;
            const unsigned char* a1 = src0 + srcstride * sy + sx1;
            const unsigned char* b0 = src0 + srcstride * sy1 + sx;
            const unsigned char* b1 = src0 + srcstride * sy1 + sx1;

            if (sx < 0 || sx >= srcw || sy < 0 || sy >= srch)
            {
                a0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy < 0 || sy >= srch)
            {
                a1 = type != -233 ? border_color : dst0;
            }
            if (sx < 0 || sx >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b1 = type != -233 ? border_color : dst0;
            }

            dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);

            dst0 += 1;
        }
        for (; x < w; x++)
        {
            if (type != -233)
            {
                dst0[0] = border_color[0];
            }

            dst0 += 1;
        }

        dst0 += wgap;
    }
    for (; y < y_bound.end_inner; y++)
    {
        const Boundary& x_bound = x_bounds[y - y_bound.start_outer];

        //         fprintf(stderr, "x %d = %d %d %d %d\n", y, x_bound.start_outer, x_bound.start_inner, x_bound.end_inner, x_bound.end_outer);

        int X0 = SATURATE_CAST_INT(((tm[1] * (y) + tm[2]) * (1 << 10)));
        int Y0 = SATURATE_CAST_INT(((tm[4] * (y) + tm[5]) * (1 << 10)));

        int x = 0;
        for (; x < x_bound.start_outer; x++)
        {
            if (type != -233)
            {
                dst0[0] = border_color[0];
            }

            dst0 += 1;
        }
        for (; x < x_bound.start_inner; x++)
        {
            int X = X0 + adelta[x];
            int Y = Y0 + bdelta[x];

            short sx = SATURATE_CAST_SHORT((X >> 10));
            short sy = SATURATE_CAST_SHORT((Y >> 10));
            short fx = X & ((1 << 10) - 1);
            short fy = Y & ((1 << 10) - 1);

            short alpha0 = (1 << 10) - fx;
            short alpha1 = fx;

            short beta0 = (1 << 10) - fy;
            short beta1 = fy;

            short sx1 = sx + 1;
            short sy1 = sy + 1;

            const unsigned char* a0 = src0 + srcstride * sy + sx;
            const unsigned char* a1 = src0 + srcstride * sy + sx1;
            const unsigned char* b0 = src0 + srcstride * sy1 + sx;
            const unsigned char* b1 = src0 + srcstride * sy1 + sx1;

            if (sx < 0 || sx >= srcw || sy < 0 || sy >= srch)
            {
                a0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy < 0 || sy >= srch)
            {
                a1 = type != -233 ? border_color : dst0;
            }
            if (sx < 0 || sx >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b1 = type != -233 ? border_color : dst0;
            }

            dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);

            dst0 += 1;
        }
#if __ARM_NEON
        for (; x + 7 < x_bound.end_inner; x += 8)
        {
            int32x4_t _Xl = vaddq_s32(vdupq_n_s32(X0), vld1q_s32(adelta.data() + x));
            int32x4_t _Xh = vaddq_s32(vdupq_n_s32(X0), vld1q_s32(adelta.data() + x + 4));
            int32x4_t _Yl = vaddq_s32(vdupq_n_s32(Y0), vld1q_s32(bdelta.data() + x));
            int32x4_t _Yh = vaddq_s32(vdupq_n_s32(Y0), vld1q_s32(bdelta.data() + x + 4));

            int16x4_t _sxl = vqshrn_n_s32(_Xl, 10);
            int16x4_t _sxh = vqshrn_n_s32(_Xh, 10);
            int16x4_t _syl = vqshrn_n_s32(_Yl, 10);
            int16x4_t _syh = vqshrn_n_s32(_Yh, 10);

            uint32x4_t _v1024m1 = vdupq_n_u32((1 << 10) - 1);
            uint16x8_t _fx = vcombine_u16(vmovn_u32(vandq_u32(vreinterpretq_u32_s32(_Xl), _v1024m1)), vmovn_u32(vandq_u32(vreinterpretq_u32_s32(_Xh), _v1024m1)));
            uint16x8_t _fy = vcombine_u16(vmovn_u32(vandq_u32(vreinterpretq_u32_s32(_Yl), _v1024m1)), vmovn_u32(vandq_u32(vreinterpretq_u32_s32(_Yh), _v1024m1)));

            uint16x8_t _alpha0 = vsubq_u16(vdupq_n_u16(1 << 10), _fx);
            uint16x8_t _alpha1 = _fx;
            uint16x8_t _beta0 = vsubq_u16(vdupq_n_u16(1 << 10), _fy);
            uint16x8_t _beta1 = _fy;

            int16x4_t _srcstride = vdup_n_s16(srcstride);
            int16x4_t _v1 = vdup_n_s16(1);

            int32x4_t _a0l = vaddw_s16(vmull_s16(_srcstride, _syl), _sxl);
            int32x4_t _a0h = vaddw_s16(vmull_s16(_srcstride, _syh), _sxh);
            int32x4_t _b0l = vaddw_s16(_a0l, _srcstride);
            int32x4_t _b0h = vaddw_s16(_a0h, _srcstride);
            int32x4_t _a1l = vaddw_s16(_a0l, _v1);
            int32x4_t _a1h = vaddw_s16(_a0h, _v1);
            int32x4_t _b1l = vaddw_s16(_b0l, _v1);
            int32x4_t _b1h = vaddw_s16(_b0h, _v1);

            uint8x8_t _a0 = uint8x8_t();
            uint8x8_t _a1 = uint8x8_t();
            uint8x8_t _b0 = uint8x8_t();
            uint8x8_t _b1 = uint8x8_t();
            {
                _a0 = vld1_lane_u8(src0 + vgetq_lane_s32(_a0l, 0), _a0, 0);
                _a1 = vld1_lane_u8(src0 + vgetq_lane_s32(_a1l, 0), _a1, 0);
                _b0 = vld1_lane_u8(src0 + vgetq_lane_s32(_b0l, 0), _b0, 0);
                _b1 = vld1_lane_u8(src0 + vgetq_lane_s32(_b1l, 0), _b1, 0);

                _a0 = vld1_lane_u8(src0 + vgetq_lane_s32(_a0l, 1), _a0, 1);
                _a1 = vld1_lane_u8(src0 + vgetq_lane_s32(_a1l, 1), _a1, 1);
                _b0 = vld1_lane_u8(src0 + vgetq_lane_s32(_b0l, 1), _b0, 1);
                _b1 = vld1_lane_u8(src0 + vgetq_lane_s32(_b1l, 1), _b1, 1);

                _a0 = vld1_lane_u8(src0 + vgetq_lane_s32(_a0l, 2), _a0, 2);
                _a1 = vld1_lane_u8(src0 + vgetq_lane_s32(_a1l, 2), _a1, 2);
                _b0 = vld1_lane_u8(src0 + vgetq_lane_s32(_b0l, 2), _b0, 2);
                _b1 = vld1_lane_u8(src0 + vgetq_lane_s32(_b1l, 2), _b1, 2);

                _a0 = vld1_lane_u8(src0 + vgetq_lane_s32(_a0l, 3), _a0, 3);
                _a1 = vld1_lane_u8(src0 + vgetq_lane_s32(_a1l, 3), _a1, 3);
                _b0 = vld1_lane_u8(src0 + vgetq_lane_s32(_b0l, 3), _b0, 3);
                _b1 = vld1_lane_u8(src0 + vgetq_lane_s32(_b1l, 3), _b1, 3);

                _a0 = vld1_lane_u8(src0 + vgetq_lane_s32(_a0h, 0), _a0, 4);
                _a1 = vld1_lane_u8(src0 + vgetq_lane_s32(_a1h, 0), _a1, 4);
                _b0 = vld1_lane_u8(src0 + vgetq_lane_s32(_b0h, 0), _b0, 4);
                _b1 = vld1_lane_u8(src0 + vgetq_lane_s32(_b1h, 0), _b1, 4);

                _a0 = vld1_lane_u8(src0 + vgetq_lane_s32(_a0h, 1), _a0, 5);
                _a1 = vld1_lane_u8(src0 + vgetq_lane_s32(_a1h, 1), _a1, 5);
                _b0 = vld1_lane_u8(src0 + vgetq_lane_s32(_b0h, 1), _b0, 5);
                _b1 = vld1_lane_u8(src0 + vgetq_lane_s32(_b1h, 1), _b1, 5);

                _a0 = vld1_lane_u8(src0 + vgetq_lane_s32(_a0h, 2), _a0, 6);
                _a1 = vld1_lane_u8(src0 + vgetq_lane_s32(_a1h, 2), _a1, 6);
                _b0 = vld1_lane_u8(src0 + vgetq_lane_s32(_b0h, 2), _b0, 6);
                _b1 = vld1_lane_u8(src0 + vgetq_lane_s32(_b1h, 2), _b1, 6);

                _a0 = vld1_lane_u8(src0 + vgetq_lane_s32(_a0h, 3), _a0, 7);
                _a1 = vld1_lane_u8(src0 + vgetq_lane_s32(_a1h, 3), _a1, 7);
                _b0 = vld1_lane_u8(src0 + vgetq_lane_s32(_b0h, 3), _b0, 7);
                _b1 = vld1_lane_u8(src0 + vgetq_lane_s32(_b1h, 3), _b1, 7);
            }

            uint16x8_t _a0_0 = vmovl_u8(_a0);
            uint16x8_t _a1_0 = vmovl_u8(_a1);
            uint16x8_t _b0_0 = vmovl_u8(_b0);
            uint16x8_t _b1_0 = vmovl_u8(_b1);

            uint16x4_t _a00_0l = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_low_u16(_a0_0), vget_low_u16(_alpha0)), vget_low_u16(_a1_0), vget_low_u16(_alpha1)), 5);
            uint16x4_t _a00_0h = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_high_u16(_a0_0), vget_high_u16(_alpha0)), vget_high_u16(_a1_0), vget_high_u16(_alpha1)), 5);
            uint16x4_t _b00_0l = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_low_u16(_b0_0), vget_low_u16(_alpha0)), vget_low_u16(_b1_0), vget_low_u16(_alpha1)), 5);
            uint16x4_t _b00_0h = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_high_u16(_b0_0), vget_high_u16(_alpha0)), vget_high_u16(_b1_0), vget_high_u16(_alpha1)), 5);

            uint16x4_t _dst_0l = vqshrn_n_u32(vmlal_u16(vmull_u16(_a00_0l, vget_low_u16(_beta0)), _b00_0l, vget_low_u16(_beta1)), 15);
            uint16x4_t _dst_0h = vqshrn_n_u32(vmlal_u16(vmull_u16(_a00_0h, vget_high_u16(_beta0)), _b00_0h, vget_high_u16(_beta1)), 15);

            uint8x8_t _dst = vqmovn_u16(vcombine_u16(_dst_0l, _dst_0h));

            vst1_u8(dst0, _dst);

            dst0 += 8;
        }
#endif // __ARM_NEON
        for (; x < x_bound.end_inner; x++)
        {
            int X = X0 + adelta[x];
            int Y = Y0 + bdelta[x];

            short sx = SATURATE_CAST_SHORT((X >> 10));
            short sy = SATURATE_CAST_SHORT((Y >> 10));
            short fx = X & ((1 << 10) - 1);
            short fy = Y & ((1 << 10) - 1);

            short alpha0 = (1 << 10) - fx;
            short alpha1 = fx;

            short beta0 = (1 << 10) - fy;
            short beta1 = fy;

            short sx1 = sx + 1;
            short sy1 = sy + 1;

            const unsigned char* a0 = src0 + srcstride * sy + sx;
            const unsigned char* a1 = src0 + srcstride * sy + sx1;
            const unsigned char* b0 = src0 + srcstride * sy1 + sx;
            const unsigned char* b1 = src0 + srcstride * sy1 + sx1;

            dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);

            dst0 += 1;
        }
        for (; x < x_bound.end_outer; x++)
        {
            int X = X0 + adelta[x];
            int Y = Y0 + bdelta[x];

            short sx = SATURATE_CAST_SHORT((X >> 10));
            short sy = SATURATE_CAST_SHORT((Y >> 10));
            short fx = X & ((1 << 10) - 1);
            short fy = Y & ((1 << 10) - 1);

            short alpha0 = (1 << 10) - fx;
            short alpha1 = fx;

            short beta0 = (1 << 10) - fy;
            short beta1 = fy;

            short sx1 = sx + 1;
            short sy1 = sy + 1;

            const unsigned char* a0 = src0 + srcstride * sy + sx;
            const unsigned char* a1 = src0 + srcstride * sy + sx1;
            const unsigned char* b0 = src0 + srcstride * sy1 + sx;
            const unsigned char* b1 = src0 + srcstride * sy1 + sx1;

            if (sx < 0 || sx >= srcw || sy < 0 || sy >= srch)
            {
                a0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy < 0 || sy >= srch)
            {
                a1 = type != -233 ? border_color : dst0;
            }
            if (sx < 0 || sx >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b1 = type != -233 ? border_color : dst0;
            }

            dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);

            dst0 += 1;
        }
        for (; x < w; x++)
        {
            if (type != -233)
            {
                dst0[0] = border_color[0];
            }

            dst0 += 1;
        }

        dst0 += wgap;
    }
    for (; y < y_bound.end_outer; y++)
    {
        const Boundary& x_bound = x_bounds[y - y_bound.start_outer];

        int X0 = SATURATE_CAST_INT(((tm[1] * (y) + tm[2]) * (1 << 10)));
        int Y0 = SATURATE_CAST_INT(((tm[4] * (y) + tm[5]) * (1 << 10)));

        int x = 0;
        for (; x < x_bound.start_outer; x++)
        {
            if (type != -233)
            {
                dst0[0] = border_color[0];
            }

            dst0 += 1;
        }
        for (; x < x_bound.start_inner; x++)
        {
            int X = X0 + adelta[x];
            int Y = Y0 + bdelta[x];

            short sx = SATURATE_CAST_SHORT((X >> 10));
            short sy = SATURATE_CAST_SHORT((Y >> 10));
            short fx = X & ((1 << 10) - 1);
            short fy = Y & ((1 << 10) - 1);

            short alpha0 = (1 << 10) - fx;
            short alpha1 = fx;

            short beta0 = (1 << 10) - fy;
            short beta1 = fy;

            short sx1 = sx + 1;
            short sy1 = sy + 1;

            const unsigned char* a0 = src0 + srcstride * sy + sx;
            const unsigned char* a1 = src0 + srcstride * sy + sx1;
            const unsigned char* b0 = src0 + srcstride * sy1 + sx;
            const unsigned char* b1 = src0 + srcstride * sy1 + sx1;

            if (sx < 0 || sx >= srcw || sy < 0 || sy >= srch)
            {
                a0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy < 0 || sy >= srch)
            {
                a1 = type != -233 ? border_color : dst0;
            }
            if (sx < 0 || sx >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b1 = type != -233 ? border_color : dst0;
            }

            dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);

            dst0 += 1;
        }
        for (; x < x_bound.end_inner; x++)
        {
            int X = X0 + adelta[x];
            int Y = Y0 + bdelta[x];

            short sx = SATURATE_CAST_SHORT((X >> 10));
            short sy = SATURATE_CAST_SHORT((Y >> 10));
            short fx = X & ((1 << 10) - 1);
            short fy = Y & ((1 << 10) - 1);

            short alpha0 = (1 << 10) - fx;
            short alpha1 = fx;

            short beta0 = (1 << 10) - fy;
            short beta1 = fy;

            short sx1 = sx + 1;
            short sy1 = sy + 1;

            const unsigned char* a0 = src0 + srcstride * sy + sx;
            const unsigned char* a1 = src0 + srcstride * sy + sx1;
            const unsigned char* b0 = src0 + srcstride * sy1 + sx;
            const unsigned char* b1 = src0 + srcstride * sy1 + sx1;

            if (sy < 0 || sy >= srch)
            {
                a0 = type != -233 ? border_color : dst0;
                a1 = type != -233 ? border_color : dst0;
            }
            if (sy1 < 0 || sy1 >= srch)
            {
                b0 = type != -233 ? border_color : dst0;
                b1 = type != -233 ? border_color : dst0;
            }

            dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);

            dst0 += 1;
        }
        for (; x < x_bound.end_outer; x++)
        {
            int X = X0 + adelta[x];
            int Y = Y0 + bdelta[x];

            short sx = SATURATE_CAST_SHORT((X >> 10));
            short sy = SATURATE_CAST_SHORT((Y >> 10));
            short fx = X & ((1 << 10) - 1);
            short fy = Y & ((1 << 10) - 1);

            short alpha0 = (1 << 10) - fx;
            short alpha1 = fx;

            short beta0 = (1 << 10) - fy;
            short beta1 = fy;

            short sx1 = sx + 1;
            short sy1 = sy + 1;

            const unsigned char* a0 = src0 + srcstride * sy + sx;
            const unsigned char* a1 = src0 + srcstride * sy + sx1;
            const unsigned char* b0 = src0 + srcstride * sy1 + sx;
            const unsigned char* b1 = src0 + srcstride * sy1 + sx1;

            if (sx < 0 || sx >= srcw || sy < 0 || sy >= srch)
            {
                a0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy < 0 || sy >= srch)
            {
                a1 = type != -233 ? border_color : dst0;
            }
            if (sx < 0 || sx >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b1 = type != -233 ? border_color : dst0;
            }

            dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);

            dst0 += 1;
        }
        for (; x < w; x++)
        {
            if (type != -233)
            {
                dst0[0] = border_color[0];
            }

            dst0 += 1;
        }

        dst0 += wgap;
    }
    for (; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            if (type != -233)
            {
                dst0[0] = border_color[0];
            }

            dst0 += 1;
        }

        dst0 += wgap;
    }

#undef SATURATE_CAST_SHORT
#undef SATURATE_CAST_INT
}

void warpaffine_bilinear_c2(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, const float* tm, int type, unsigned int v)
{
    const unsigned char* border_color = (const unsigned char*)&v;
    const int wgap = stride - w * 2;

    const unsigned char* src0 = src;
    unsigned char* dst0 = dst;

#define SATURATE_CAST_SHORT(X) (short)::std::min(::std::max((int)(X), SHRT_MIN), SHRT_MAX)
#define SATURATE_CAST_INT(X)   (int)::std::min(::std::max((int)(X + (X >= 0.f ? 0.5f : -0.5f)), INT_MIN), INT_MAX)

    std::vector<int> adelta(w);
    std::vector<int> bdelta(w);
    for (int x = 0; x < w; x++)
    {
        adelta[x] = SATURATE_CAST_INT(tm[0] * x * (1 << 10));
        bdelta[x] = SATURATE_CAST_INT(tm[3] * x * (1 << 10));
    }

    Boundary y_bound;
    std::vector<Boundary> x_bounds;
    resolve_boundary(srcw, srch, w, h, tm, y_bound, x_bounds);

    //     fprintf(stderr, "y %d %d %d %d\n", y_bound.start_outer, y_bound.start_inner, y_bound.end_inner, y_bound.end_outer);

    int y = 0;
    for (; y < y_bound.start_outer; y++)
    {
        for (int x = 0; x < w; x++)
        {
            if (type != -233)
            {
                dst0[0] = border_color[0];
                dst0[1] = border_color[1];
            }

            dst0 += 2;
        }

        dst0 += wgap;
    }
    for (; y < y_bound.start_inner; y++)
    {
        const Boundary& x_bound = x_bounds[y - y_bound.start_outer];

        int X0 = SATURATE_CAST_INT(((tm[1] * (y) + tm[2]) * (1 << 10)));
        int Y0 = SATURATE_CAST_INT(((tm[4] * (y) + tm[5]) * (1 << 10)));

        int x = 0;
        for (; x < x_bound.start_outer; x++)
        {
            if (type != -233)
            {
                dst0[0] = border_color[0];
                dst0[1] = border_color[1];
            }

            dst0 += 2;
        }
        for (; x < x_bound.start_inner; x++)
        {
            int X = X0 + adelta[x];
            int Y = Y0 + bdelta[x];

            short sx = SATURATE_CAST_SHORT((X >> 10));
            short sy = SATURATE_CAST_SHORT((Y >> 10));
            short fx = X & ((1 << 10) - 1);
            short fy = Y & ((1 << 10) - 1);

            short alpha0 = (1 << 10) - fx;
            short alpha1 = fx;

            short beta0 = (1 << 10) - fy;
            short beta1 = fy;

            short sx1 = sx + 1;
            short sy1 = sy + 1;

            const unsigned char* a0 = src0 + srcstride * sy + sx * 2;
            const unsigned char* a1 = src0 + srcstride * sy + sx1 * 2;
            const unsigned char* b0 = src0 + srcstride * sy1 + sx * 2;
            const unsigned char* b1 = src0 + srcstride * sy1 + sx1 * 2;

            if (sx < 0 || sx >= srcw || sy < 0 || sy >= srch)
            {
                a0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy < 0 || sy >= srch)
            {
                a1 = type != -233 ? border_color : dst0;
            }
            if (sx < 0 || sx >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b1 = type != -233 ? border_color : dst0;
            }

            dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);
            dst0[1] = (unsigned char)(((((unsigned short)((a0[1] * alpha0 + a1[1] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[1] * alpha0 + b1[1] * alpha1) >> 5) * beta1))) >> 15);

            dst0 += 2;
        }
        for (; x < x_bound.end_inner; x++)
        {
            int X = X0 + adelta[x];
            int Y = Y0 + bdelta[x];

            short sx = SATURATE_CAST_SHORT((X >> 10));
            short sy = SATURATE_CAST_SHORT((Y >> 10));
            short fx = X & ((1 << 10) - 1);
            short fy = Y & ((1 << 10) - 1);

            short alpha0 = (1 << 10) - fx;
            short alpha1 = fx;

            short beta0 = (1 << 10) - fy;
            short beta1 = fy;

            short sx1 = sx + 1;
            short sy1 = sy + 1;

            const unsigned char* a0 = src0 + srcstride * sy + sx * 2;
            const unsigned char* a1 = src0 + srcstride * sy + sx1 * 2;
            const unsigned char* b0 = src0 + srcstride * sy1 + sx * 2;
            const unsigned char* b1 = src0 + srcstride * sy1 + sx1 * 2;

            if (sy < 0 || sy >= srch)
            {
                a0 = type != -233 ? border_color : dst0;
                a1 = type != -233 ? border_color : dst0;
            }
            if (sy1 < 0 || sy1 >= srch)
            {
                b0 = type != -233 ? border_color : dst0;
                b1 = type != -233 ? border_color : dst0;
            }

            dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);
            dst0[1] = (unsigned char)(((((unsigned short)((a0[1] * alpha0 + a1[1] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[1] * alpha0 + b1[1] * alpha1) >> 5) * beta1))) >> 15);

            dst0 += 2;
        }
        for (; x < x_bound.end_outer; x++)
        {
            int X = X0 + adelta[x];
            int Y = Y0 + bdelta[x];

            short sx = SATURATE_CAST_SHORT((X >> 10));
            short sy = SATURATE_CAST_SHORT((Y >> 10));
            short fx = X & ((1 << 10) - 1);
            short fy = Y & ((1 << 10) - 1);

            short alpha0 = (1 << 10) - fx;
            short alpha1 = fx;

            short beta0 = (1 << 10) - fy;
            short beta1 = fy;

            short sx1 = sx + 1;
            short sy1 = sy + 1;

            const unsigned char* a0 = src0 + srcstride * sy + sx * 2;
            const unsigned char* a1 = src0 + srcstride * sy + sx1 * 2;
            const unsigned char* b0 = src0 + srcstride * sy1 + sx * 2;
            const unsigned char* b1 = src0 + srcstride * sy1 + sx1 * 2;

            if (sx < 0 || sx >= srcw || sy < 0 || sy >= srch)
            {
                a0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy < 0 || sy >= srch)
            {
                a1 = type != -233 ? border_color : dst0;
            }
            if (sx < 0 || sx >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b1 = type != -233 ? border_color : dst0;
            }

            dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);
            dst0[1] = (unsigned char)(((((unsigned short)((a0[1] * alpha0 + a1[1] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[1] * alpha0 + b1[1] * alpha1) >> 5) * beta1))) >> 15);

            dst0 += 2;
        }
        for (; x < w; x++)
        {
            if (type != -233)
            {
                dst0[0] = border_color[0];
                dst0[1] = border_color[1];
            }

            dst0 += 2;
        }

        dst0 += wgap;
    }
    for (; y < y_bound.end_inner; y++)
    {
        const Boundary& x_bound = x_bounds[y - y_bound.start_outer];

        //         fprintf(stderr, "x %d = %d %d %d %d\n", y, x_bound.start_outer, x_bound.start_inner, x_bound.end_inner, x_bound.end_outer);

        int X0 = SATURATE_CAST_INT(((tm[1] * (y) + tm[2]) * (1 << 10)));
        int Y0 = SATURATE_CAST_INT(((tm[4] * (y) + tm[5]) * (1 << 10)));

        int x = 0;
        for (; x < x_bound.start_outer; x++)
        {
            if (type != -233)
            {
                dst0[0] = border_color[0];
                dst0[1] = border_color[1];
            }

            dst0 += 2;
        }
        for (; x < x_bound.start_inner; x++)
        {
            int X = X0 + adelta[x];
            int Y = Y0 + bdelta[x];

            short sx = SATURATE_CAST_SHORT((X >> 10));
            short sy = SATURATE_CAST_SHORT((Y >> 10));
            short fx = X & ((1 << 10) - 1);
            short fy = Y & ((1 << 10) - 1);

            short alpha0 = (1 << 10) - fx;
            short alpha1 = fx;

            short beta0 = (1 << 10) - fy;
            short beta1 = fy;

            short sx1 = sx + 1;
            short sy1 = sy + 1;

            const unsigned char* a0 = src0 + srcstride * sy + sx * 2;
            const unsigned char* a1 = src0 + srcstride * sy + sx1 * 2;
            const unsigned char* b0 = src0 + srcstride * sy1 + sx * 2;
            const unsigned char* b1 = src0 + srcstride * sy1 + sx1 * 2;

            if (sx < 0 || sx >= srcw || sy < 0 || sy >= srch)
            {
                a0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy < 0 || sy >= srch)
            {
                a1 = type != -233 ? border_color : dst0;
            }
            if (sx < 0 || sx >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b1 = type != -233 ? border_color : dst0;
            }

            dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);
            dst0[1] = (unsigned char)(((((unsigned short)((a0[1] * alpha0 + a1[1] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[1] * alpha0 + b1[1] * alpha1) >> 5) * beta1))) >> 15);

            dst0 += 2;
        }
#if __ARM_NEON
        for (; x + 7 < x_bound.end_inner; x += 8)
        {
            int32x4_t _Xl = vaddq_s32(vdupq_n_s32(X0), vld1q_s32(adelta.data() + x));
            int32x4_t _Xh = vaddq_s32(vdupq_n_s32(X0), vld1q_s32(adelta.data() + x + 4));
            int32x4_t _Yl = vaddq_s32(vdupq_n_s32(Y0), vld1q_s32(bdelta.data() + x));
            int32x4_t _Yh = vaddq_s32(vdupq_n_s32(Y0), vld1q_s32(bdelta.data() + x + 4));

            int16x4_t _sxl = vqshrn_n_s32(_Xl, 10);
            int16x4_t _sxh = vqshrn_n_s32(_Xh, 10);
            int16x4_t _syl = vqshrn_n_s32(_Yl, 10);
            int16x4_t _syh = vqshrn_n_s32(_Yh, 10);

            uint32x4_t _v1024m1 = vdupq_n_u32((1 << 10) - 1);
            uint16x8_t _fx = vcombine_u16(vmovn_u32(vandq_u32(vreinterpretq_u32_s32(_Xl), _v1024m1)), vmovn_u32(vandq_u32(vreinterpretq_u32_s32(_Xh), _v1024m1)));
            uint16x8_t _fy = vcombine_u16(vmovn_u32(vandq_u32(vreinterpretq_u32_s32(_Yl), _v1024m1)), vmovn_u32(vandq_u32(vreinterpretq_u32_s32(_Yh), _v1024m1)));

            uint16x8_t _alpha0 = vsubq_u16(vdupq_n_u16(1 << 10), _fx);
            uint16x8_t _alpha1 = _fx;
            uint16x8_t _beta0 = vsubq_u16(vdupq_n_u16(1 << 10), _fy);
            uint16x8_t _beta1 = _fy;

            int16x4_t _srcstride = vdup_n_s16(srcstride);
            int16x4_t _v2 = vdup_n_s16(2);

            int32x4_t _a0l = vmlal_s16(vmull_s16(_srcstride, _syl), _sxl, _v2);
            int32x4_t _a0h = vmlal_s16(vmull_s16(_srcstride, _syh), _sxh, _v2);
            int32x4_t _b0l = vaddw_s16(_a0l, _srcstride);
            int32x4_t _b0h = vaddw_s16(_a0h, _srcstride);
            int32x4_t _a1l = vaddw_s16(_a0l, _v2);
            int32x4_t _a1h = vaddw_s16(_a0h, _v2);
            int32x4_t _b1l = vaddw_s16(_b0l, _v2);
            int32x4_t _b1h = vaddw_s16(_b0h, _v2);

            uint8x8x2_t _a0 = uint8x8x2_t();
            uint8x8x2_t _a1 = uint8x8x2_t();
            uint8x8x2_t _b0 = uint8x8x2_t();
            uint8x8x2_t _b1 = uint8x8x2_t();
            {
                _a0 = vld2_lane_u8(src0 + vgetq_lane_s32(_a0l, 0), _a0, 0);
                _a1 = vld2_lane_u8(src0 + vgetq_lane_s32(_a1l, 0), _a1, 0);
                _b0 = vld2_lane_u8(src0 + vgetq_lane_s32(_b0l, 0), _b0, 0);
                _b1 = vld2_lane_u8(src0 + vgetq_lane_s32(_b1l, 0), _b1, 0);

                _a0 = vld2_lane_u8(src0 + vgetq_lane_s32(_a0l, 1), _a0, 1);
                _a1 = vld2_lane_u8(src0 + vgetq_lane_s32(_a1l, 1), _a1, 1);
                _b0 = vld2_lane_u8(src0 + vgetq_lane_s32(_b0l, 1), _b0, 1);
                _b1 = vld2_lane_u8(src0 + vgetq_lane_s32(_b1l, 1), _b1, 1);

                _a0 = vld2_lane_u8(src0 + vgetq_lane_s32(_a0l, 2), _a0, 2);
                _a1 = vld2_lane_u8(src0 + vgetq_lane_s32(_a1l, 2), _a1, 2);
                _b0 = vld2_lane_u8(src0 + vgetq_lane_s32(_b0l, 2), _b0, 2);
                _b1 = vld2_lane_u8(src0 + vgetq_lane_s32(_b1l, 2), _b1, 2);

                _a0 = vld2_lane_u8(src0 + vgetq_lane_s32(_a0l, 3), _a0, 3);
                _a1 = vld2_lane_u8(src0 + vgetq_lane_s32(_a1l, 3), _a1, 3);
                _b0 = vld2_lane_u8(src0 + vgetq_lane_s32(_b0l, 3), _b0, 3);
                _b1 = vld2_lane_u8(src0 + vgetq_lane_s32(_b1l, 3), _b1, 3);

                _a0 = vld2_lane_u8(src0 + vgetq_lane_s32(_a0h, 0), _a0, 4);
                _a1 = vld2_lane_u8(src0 + vgetq_lane_s32(_a1h, 0), _a1, 4);
                _b0 = vld2_lane_u8(src0 + vgetq_lane_s32(_b0h, 0), _b0, 4);
                _b1 = vld2_lane_u8(src0 + vgetq_lane_s32(_b1h, 0), _b1, 4);

                _a0 = vld2_lane_u8(src0 + vgetq_lane_s32(_a0h, 1), _a0, 5);
                _a1 = vld2_lane_u8(src0 + vgetq_lane_s32(_a1h, 1), _a1, 5);
                _b0 = vld2_lane_u8(src0 + vgetq_lane_s32(_b0h, 1), _b0, 5);
                _b1 = vld2_lane_u8(src0 + vgetq_lane_s32(_b1h, 1), _b1, 5);

                _a0 = vld2_lane_u8(src0 + vgetq_lane_s32(_a0h, 2), _a0, 6);
                _a1 = vld2_lane_u8(src0 + vgetq_lane_s32(_a1h, 2), _a1, 6);
                _b0 = vld2_lane_u8(src0 + vgetq_lane_s32(_b0h, 2), _b0, 6);
                _b1 = vld2_lane_u8(src0 + vgetq_lane_s32(_b1h, 2), _b1, 6);

                _a0 = vld2_lane_u8(src0 + vgetq_lane_s32(_a0h, 3), _a0, 7);
                _a1 = vld2_lane_u8(src0 + vgetq_lane_s32(_a1h, 3), _a1, 7);
                _b0 = vld2_lane_u8(src0 + vgetq_lane_s32(_b0h, 3), _b0, 7);
                _b1 = vld2_lane_u8(src0 + vgetq_lane_s32(_b1h, 3), _b1, 7);
            }

            uint16x8_t _a0_0 = vmovl_u8(_a0.val[0]);
            uint16x8_t _a0_1 = vmovl_u8(_a0.val[1]);
            uint16x8_t _a1_0 = vmovl_u8(_a1.val[0]);
            uint16x8_t _a1_1 = vmovl_u8(_a1.val[1]);
            uint16x8_t _b0_0 = vmovl_u8(_b0.val[0]);
            uint16x8_t _b0_1 = vmovl_u8(_b0.val[1]);
            uint16x8_t _b1_0 = vmovl_u8(_b1.val[0]);
            uint16x8_t _b1_1 = vmovl_u8(_b1.val[1]);

            uint16x4_t _a00_0l = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_low_u16(_a0_0), vget_low_u16(_alpha0)), vget_low_u16(_a1_0), vget_low_u16(_alpha1)), 5);
            uint16x4_t _a00_1l = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_low_u16(_a0_1), vget_low_u16(_alpha0)), vget_low_u16(_a1_1), vget_low_u16(_alpha1)), 5);
            uint16x4_t _a00_0h = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_high_u16(_a0_0), vget_high_u16(_alpha0)), vget_high_u16(_a1_0), vget_high_u16(_alpha1)), 5);
            uint16x4_t _a00_1h = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_high_u16(_a0_1), vget_high_u16(_alpha0)), vget_high_u16(_a1_1), vget_high_u16(_alpha1)), 5);
            uint16x4_t _b00_0l = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_low_u16(_b0_0), vget_low_u16(_alpha0)), vget_low_u16(_b1_0), vget_low_u16(_alpha1)), 5);
            uint16x4_t _b00_1l = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_low_u16(_b0_1), vget_low_u16(_alpha0)), vget_low_u16(_b1_1), vget_low_u16(_alpha1)), 5);
            uint16x4_t _b00_0h = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_high_u16(_b0_0), vget_high_u16(_alpha0)), vget_high_u16(_b1_0), vget_high_u16(_alpha1)), 5);
            uint16x4_t _b00_1h = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_high_u16(_b0_1), vget_high_u16(_alpha0)), vget_high_u16(_b1_1), vget_high_u16(_alpha1)), 5);

            uint16x4_t _dst_0l = vqshrn_n_u32(vmlal_u16(vmull_u16(_a00_0l, vget_low_u16(_beta0)), _b00_0l, vget_low_u16(_beta1)), 15);
            uint16x4_t _dst_1l = vqshrn_n_u32(vmlal_u16(vmull_u16(_a00_1l, vget_low_u16(_beta0)), _b00_1l, vget_low_u16(_beta1)), 15);
            uint16x4_t _dst_0h = vqshrn_n_u32(vmlal_u16(vmull_u16(_a00_0h, vget_high_u16(_beta0)), _b00_0h, vget_high_u16(_beta1)), 15);
            uint16x4_t _dst_1h = vqshrn_n_u32(vmlal_u16(vmull_u16(_a00_1h, vget_high_u16(_beta0)), _b00_1h, vget_high_u16(_beta1)), 15);

            uint8x8x2_t _dst;
            _dst.val[0] = vqmovn_u16(vcombine_u16(_dst_0l, _dst_0h));
            _dst.val[1] = vqmovn_u16(vcombine_u16(_dst_1l, _dst_1h));

            vst2_u8(dst0, _dst);

            dst0 += 2 * 8;
        }
#endif // __ARM_NEON
        for (; x < x_bound.end_inner; x++)
        {
            int X = X0 + adelta[x];
            int Y = Y0 + bdelta[x];

            short sx = SATURATE_CAST_SHORT((X >> 10));
            short sy = SATURATE_CAST_SHORT((Y >> 10));
            short fx = X & ((1 << 10) - 1);
            short fy = Y & ((1 << 10) - 1);

            short alpha0 = (1 << 10) - fx;
            short alpha1 = fx;

            short beta0 = (1 << 10) - fy;
            short beta1 = fy;

            short sx1 = sx + 1;
            short sy1 = sy + 1;

            const unsigned char* a0 = src0 + srcstride * sy + sx * 2;
            const unsigned char* a1 = src0 + srcstride * sy + sx1 * 2;
            const unsigned char* b0 = src0 + srcstride * sy1 + sx * 2;
            const unsigned char* b1 = src0 + srcstride * sy1 + sx1 * 2;

            dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);
            dst0[1] = (unsigned char)(((((unsigned short)((a0[1] * alpha0 + a1[1] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[1] * alpha0 + b1[1] * alpha1) >> 5) * beta1))) >> 15);

            dst0 += 2;
        }
        for (; x < x_bound.end_outer; x++)
        {
            int X = X0 + adelta[x];
            int Y = Y0 + bdelta[x];

            short sx = SATURATE_CAST_SHORT((X >> 10));
            short sy = SATURATE_CAST_SHORT((Y >> 10));
            short fx = X & ((1 << 10) - 1);
            short fy = Y & ((1 << 10) - 1);

            short alpha0 = (1 << 10) - fx;
            short alpha1 = fx;

            short beta0 = (1 << 10) - fy;
            short beta1 = fy;

            short sx1 = sx + 1;
            short sy1 = sy + 1;

            const unsigned char* a0 = src0 + srcstride * sy + sx * 2;
            const unsigned char* a1 = src0 + srcstride * sy + sx1 * 2;
            const unsigned char* b0 = src0 + srcstride * sy1 + sx * 2;
            const unsigned char* b1 = src0 + srcstride * sy1 + sx1 * 2;

            if (sx < 0 || sx >= srcw || sy < 0 || sy >= srch)
            {
                a0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy < 0 || sy >= srch)
            {
                a1 = type != -233 ? border_color : dst0;
            }
            if (sx < 0 || sx >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b1 = type != -233 ? border_color : dst0;
            }

            dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);
            dst0[1] = (unsigned char)(((((unsigned short)((a0[1] * alpha0 + a1[1] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[1] * alpha0 + b1[1] * alpha1) >> 5) * beta1))) >> 15);

            dst0 += 2;
        }
        for (; x < w; x++)
        {
            if (type != -233)
            {
                dst0[0] = border_color[0];
                dst0[1] = border_color[1];
            }

            dst0 += 2;
        }

        dst0 += wgap;
    }
    for (; y < y_bound.end_outer; y++)
    {
        const Boundary& x_bound = x_bounds[y - y_bound.start_outer];

        int X0 = SATURATE_CAST_INT(((tm[1] * (y) + tm[2]) * (1 << 10)));
        int Y0 = SATURATE_CAST_INT(((tm[4] * (y) + tm[5]) * (1 << 10)));

        int x = 0;
        for (; x < x_bound.start_outer; x++)
        {
            if (type != -233)
            {
                dst0[0] = border_color[0];
                dst0[1] = border_color[1];
            }

            dst0 += 2;
        }
        for (; x < x_bound.start_inner; x++)
        {
            int X = X0 + adelta[x];
            int Y = Y0 + bdelta[x];

            short sx = SATURATE_CAST_SHORT((X >> 10));
            short sy = SATURATE_CAST_SHORT((Y >> 10));
            short fx = X & ((1 << 10) - 1);
            short fy = Y & ((1 << 10) - 1);

            short alpha0 = (1 << 10) - fx;
            short alpha1 = fx;

            short beta0 = (1 << 10) - fy;
            short beta1 = fy;

            short sx1 = sx + 1;
            short sy1 = sy + 1;

            const unsigned char* a0 = src0 + srcstride * sy + sx * 2;
            const unsigned char* a1 = src0 + srcstride * sy + sx1 * 2;
            const unsigned char* b0 = src0 + srcstride * sy1 + sx * 2;
            const unsigned char* b1 = src0 + srcstride * sy1 + sx1 * 2;

            if (sx < 0 || sx >= srcw || sy < 0 || sy >= srch)
            {
                a0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy < 0 || sy >= srch)
            {
                a1 = type != -233 ? border_color : dst0;
            }
            if (sx < 0 || sx >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b1 = type != -233 ? border_color : dst0;
            }

            dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);
            dst0[1] = (unsigned char)(((((unsigned short)((a0[1] * alpha0 + a1[1] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[1] * alpha0 + b1[1] * alpha1) >> 5) * beta1))) >> 15);

            dst0 += 2;
        }
        for (; x < x_bound.end_inner; x++)
        {
            int X = X0 + adelta[x];
            int Y = Y0 + bdelta[x];

            short sx = SATURATE_CAST_SHORT((X >> 10));
            short sy = SATURATE_CAST_SHORT((Y >> 10));
            short fx = X & ((1 << 10) - 1);
            short fy = Y & ((1 << 10) - 1);

            short alpha0 = (1 << 10) - fx;
            short alpha1 = fx;

            short beta0 = (1 << 10) - fy;
            short beta1 = fy;

            short sx1 = sx + 1;
            short sy1 = sy + 1;

            const unsigned char* a0 = src0 + srcstride * sy + sx * 2;
            const unsigned char* a1 = src0 + srcstride * sy + sx1 * 2;
            const unsigned char* b0 = src0 + srcstride * sy1 + sx * 2;
            const unsigned char* b1 = src0 + srcstride * sy1 + sx1 * 2;

            if (sy < 0 || sy >= srch)
            {
                a0 = type != -233 ? border_color : dst0;
                a1 = type != -233 ? border_color : dst0;
            }
            if (sy1 < 0 || sy1 >= srch)
            {
                b0 = type != -233 ? border_color : dst0;
                b1 = type != -233 ? border_color : dst0;
            }

            dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);
            dst0[1] = (unsigned char)(((((unsigned short)((a0[1] * alpha0 + a1[1] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[1] * alpha0 + b1[1] * alpha1) >> 5) * beta1))) >> 15);

            dst0 += 2;
        }
        for (; x < x_bound.end_outer; x++)
        {
            int X = X0 + adelta[x];
            int Y = Y0 + bdelta[x];

            short sx = SATURATE_CAST_SHORT((X >> 10));
            short sy = SATURATE_CAST_SHORT((Y >> 10));
            short fx = X & ((1 << 10) - 1);
            short fy = Y & ((1 << 10) - 1);

            short alpha0 = (1 << 10) - fx;
            short alpha1 = fx;

            short beta0 = (1 << 10) - fy;
            short beta1 = fy;

            short sx1 = sx + 1;
            short sy1 = sy + 1;

            const unsigned char* a0 = src0 + srcstride * sy + sx * 2;
            const unsigned char* a1 = src0 + srcstride * sy + sx1 * 2;
            const unsigned char* b0 = src0 + srcstride * sy1 + sx * 2;
            const unsigned char* b1 = src0 + srcstride * sy1 + sx1 * 2;

            if (sx < 0 || sx >= srcw || sy < 0 || sy >= srch)
            {
                a0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy < 0 || sy >= srch)
            {
                a1 = type != -233 ? border_color : dst0;
            }
            if (sx < 0 || sx >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b1 = type != -233 ? border_color : dst0;
            }

            dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);
            dst0[1] = (unsigned char)(((((unsigned short)((a0[1] * alpha0 + a1[1] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[1] * alpha0 + b1[1] * alpha1) >> 5) * beta1))) >> 15);

            dst0 += 2;
        }
        for (; x < w; x++)
        {
            if (type != -233)
            {
                dst0[0] = border_color[0];
                dst0[1] = border_color[1];
            }

            dst0 += 2;
        }

        dst0 += wgap;
    }
    for (; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            if (type != -233)
            {
                dst0[0] = border_color[0];
                dst0[1] = border_color[1];
            }

            dst0 += 2;
        }

        dst0 += wgap;
    }

#undef SATURATE_CAST_SHORT
#undef SATURATE_CAST_INT
}

void warpaffine_bilinear_c3(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, const float* tm, int type, unsigned int v)
{
    const unsigned char* border_color = (const unsigned char*)&v;
    const int wgap = stride - w * 3;

    const unsigned char* src0 = src;
    unsigned char* dst0 = dst;

#define SATURATE_CAST_SHORT(X) (short)::std::min(::std::max((int)(X), SHRT_MIN), SHRT_MAX)
#define SATURATE_CAST_INT(X)   (int)::std::min(::std::max((int)(X + (X >= 0.f ? 0.5f : -0.5f)), INT_MIN), INT_MAX)

    std::vector<int> adelta(w);
    std::vector<int> bdelta(w);
    for (int x = 0; x < w; x++)
    {
        adelta[x] = SATURATE_CAST_INT(tm[0] * x * (1 << 10));
        bdelta[x] = SATURATE_CAST_INT(tm[3] * x * (1 << 10));
    }

    Boundary y_bound;
    std::vector<Boundary> x_bounds;
    resolve_boundary(srcw, srch, w, h, tm, y_bound, x_bounds);

    //     fprintf(stderr, "y %d %d %d %d\n", y_bound.start_outer, y_bound.start_inner, y_bound.end_inner, y_bound.end_outer);

    int y = 0;
    for (; y < y_bound.start_outer; y++)
    {
        for (int x = 0; x < w; x++)
        {
            if (type != -233)
            {
                dst0[0] = border_color[0];
                dst0[1] = border_color[1];
                dst0[2] = border_color[2];
            }

            dst0 += 3;
        }

        dst0 += wgap;
    }
    for (; y < y_bound.start_inner; y++)
    {
        const Boundary& x_bound = x_bounds[y - y_bound.start_outer];

        int X0 = SATURATE_CAST_INT(((tm[1] * (y) + tm[2]) * (1 << 10)));
        int Y0 = SATURATE_CAST_INT(((tm[4] * (y) + tm[5]) * (1 << 10)));

        int x = 0;
        for (; x < x_bound.start_outer; x++)
        {
            if (type != -233)
            {
                dst0[0] = border_color[0];
                dst0[1] = border_color[1];
                dst0[2] = border_color[2];
            }

            dst0 += 3;
        }
        for (; x < x_bound.start_inner; x++)
        {
            int X = X0 + adelta[x];
            int Y = Y0 + bdelta[x];

            short sx = SATURATE_CAST_SHORT((X >> 10));
            short sy = SATURATE_CAST_SHORT((Y >> 10));
            short fx = X & ((1 << 10) - 1);
            short fy = Y & ((1 << 10) - 1);

            short alpha0 = (1 << 10) - fx;
            short alpha1 = fx;

            short beta0 = (1 << 10) - fy;
            short beta1 = fy;

            short sx1 = sx + 1;
            short sy1 = sy + 1;

            const unsigned char* a0 = src0 + srcstride * sy + sx * 3;
            const unsigned char* a1 = src0 + srcstride * sy + sx1 * 3;
            const unsigned char* b0 = src0 + srcstride * sy1 + sx * 3;
            const unsigned char* b1 = src0 + srcstride * sy1 + sx1 * 3;

            if (sx < 0 || sx >= srcw || sy < 0 || sy >= srch)
            {
                a0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy < 0 || sy >= srch)
            {
                a1 = type != -233 ? border_color : dst0;
            }
            if (sx < 0 || sx >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b1 = type != -233 ? border_color : dst0;
            }

            dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);
            dst0[1] = (unsigned char)(((((unsigned short)((a0[1] * alpha0 + a1[1] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[1] * alpha0 + b1[1] * alpha1) >> 5) * beta1))) >> 15);
            dst0[2] = (unsigned char)(((((unsigned short)((a0[2] * alpha0 + a1[2] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[2] * alpha0 + b1[2] * alpha1) >> 5) * beta1))) >> 15);

            dst0 += 3;
        }
        for (; x < x_bound.end_inner; x++)
        {
            int X = X0 + adelta[x];
            int Y = Y0 + bdelta[x];

            short sx = SATURATE_CAST_SHORT((X >> 10));
            short sy = SATURATE_CAST_SHORT((Y >> 10));
            short fx = X & ((1 << 10) - 1);
            short fy = Y & ((1 << 10) - 1);

            short alpha0 = (1 << 10) - fx;
            short alpha1 = fx;

            short beta0 = (1 << 10) - fy;
            short beta1 = fy;

            short sx1 = sx + 1;
            short sy1 = sy + 1;

            const unsigned char* a0 = src0 + srcstride * sy + sx * 3;
            const unsigned char* a1 = src0 + srcstride * sy + sx1 * 3;
            const unsigned char* b0 = src0 + srcstride * sy1 + sx * 3;
            const unsigned char* b1 = src0 + srcstride * sy1 + sx1 * 3;

            if (sy < 0 || sy >= srch)
            {
                a0 = type != -233 ? border_color : dst0;
                a1 = type != -233 ? border_color : dst0;
            }
            if (sy1 < 0 || sy1 >= srch)
            {
                b0 = type != -233 ? border_color : dst0;
                b1 = type != -233 ? border_color : dst0;
            }

            dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);
            dst0[1] = (unsigned char)(((((unsigned short)((a0[1] * alpha0 + a1[1] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[1] * alpha0 + b1[1] * alpha1) >> 5) * beta1))) >> 15);
            dst0[2] = (unsigned char)(((((unsigned short)((a0[2] * alpha0 + a1[2] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[2] * alpha0 + b1[2] * alpha1) >> 5) * beta1))) >> 15);

            dst0 += 3;
        }
        for (; x < x_bound.end_outer; x++)
        {
            int X = X0 + adelta[x];
            int Y = Y0 + bdelta[x];

            short sx = SATURATE_CAST_SHORT((X >> 10));
            short sy = SATURATE_CAST_SHORT((Y >> 10));
            short fx = X & ((1 << 10) - 1);
            short fy = Y & ((1 << 10) - 1);

            short alpha0 = (1 << 10) - fx;
            short alpha1 = fx;

            short beta0 = (1 << 10) - fy;
            short beta1 = fy;

            short sx1 = sx + 1;
            short sy1 = sy + 1;

            const unsigned char* a0 = src0 + srcstride * sy + sx * 3;
            const unsigned char* a1 = src0 + srcstride * sy + sx1 * 3;
            const unsigned char* b0 = src0 + srcstride * sy1 + sx * 3;
            const unsigned char* b1 = src0 + srcstride * sy1 + sx1 * 3;

            if (sx < 0 || sx >= srcw || sy < 0 || sy >= srch)
            {
                a0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy < 0 || sy >= srch)
            {
                a1 = type != -233 ? border_color : dst0;
            }
            if (sx < 0 || sx >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b1 = type != -233 ? border_color : dst0;
            }

            dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);
            dst0[1] = (unsigned char)(((((unsigned short)((a0[1] * alpha0 + a1[1] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[1] * alpha0 + b1[1] * alpha1) >> 5) * beta1))) >> 15);
            dst0[2] = (unsigned char)(((((unsigned short)((a0[2] * alpha0 + a1[2] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[2] * alpha0 + b1[2] * alpha1) >> 5) * beta1))) >> 15);

            dst0 += 3;
        }
        for (; x < w; x++)
        {
            if (type != -233)
            {
                dst0[0] = border_color[0];
                dst0[1] = border_color[1];
                dst0[2] = border_color[2];
            }

            dst0 += 3;
        }

        dst0 += wgap;
    }
    for (; y < y_bound.end_inner; y++)
    {
        const Boundary& x_bound = x_bounds[y - y_bound.start_outer];

        //         fprintf(stderr, "x %d = %d %d %d %d\n", y, x_bound.start_outer, x_bound.start_inner, x_bound.end_inner, x_bound.end_outer);

        int X0 = SATURATE_CAST_INT(((tm[1] * (y) + tm[2]) * (1 << 10)));
        int Y0 = SATURATE_CAST_INT(((tm[4] * (y) + tm[5]) * (1 << 10)));

        int x = 0;
        for (; x < x_bound.start_outer; x++)
        {
            if (type != -233)
            {
                dst0[0] = border_color[0];
                dst0[1] = border_color[1];
                dst0[2] = border_color[2];
            }

            dst0 += 3;
        }
        for (; x < x_bound.start_inner; x++)
        {
            int X = X0 + adelta[x];
            int Y = Y0 + bdelta[x];

            short sx = SATURATE_CAST_SHORT((X >> 10));
            short sy = SATURATE_CAST_SHORT((Y >> 10));
            short fx = X & ((1 << 10) - 1);
            short fy = Y & ((1 << 10) - 1);

            short alpha0 = (1 << 10) - fx;
            short alpha1 = fx;

            short beta0 = (1 << 10) - fy;
            short beta1 = fy;

            short sx1 = sx + 1;
            short sy1 = sy + 1;

            const unsigned char* a0 = src0 + srcstride * sy + sx * 3;
            const unsigned char* a1 = src0 + srcstride * sy + sx1 * 3;
            const unsigned char* b0 = src0 + srcstride * sy1 + sx * 3;
            const unsigned char* b1 = src0 + srcstride * sy1 + sx1 * 3;

            if (sx < 0 || sx >= srcw || sy < 0 || sy >= srch)
            {
                a0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy < 0 || sy >= srch)
            {
                a1 = type != -233 ? border_color : dst0;
            }
            if (sx < 0 || sx >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b1 = type != -233 ? border_color : dst0;
            }

            dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);
            dst0[1] = (unsigned char)(((((unsigned short)((a0[1] * alpha0 + a1[1] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[1] * alpha0 + b1[1] * alpha1) >> 5) * beta1))) >> 15);
            dst0[2] = (unsigned char)(((((unsigned short)((a0[2] * alpha0 + a1[2] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[2] * alpha0 + b1[2] * alpha1) >> 5) * beta1))) >> 15);

            dst0 += 3;
        }
#if __ARM_NEON
        for (; x + 7 < x_bound.end_inner; x += 8)
        {
            int32x4_t _Xl = vaddq_s32(vdupq_n_s32(X0), vld1q_s32(adelta.data() + x));
            int32x4_t _Xh = vaddq_s32(vdupq_n_s32(X0), vld1q_s32(adelta.data() + x + 4));
            int32x4_t _Yl = vaddq_s32(vdupq_n_s32(Y0), vld1q_s32(bdelta.data() + x));
            int32x4_t _Yh = vaddq_s32(vdupq_n_s32(Y0), vld1q_s32(bdelta.data() + x + 4));

            int16x4_t _sxl = vqshrn_n_s32(_Xl, 10);
            int16x4_t _sxh = vqshrn_n_s32(_Xh, 10);
            int16x4_t _syl = vqshrn_n_s32(_Yl, 10);
            int16x4_t _syh = vqshrn_n_s32(_Yh, 10);

            uint32x4_t _v1024m1 = vdupq_n_u32((1 << 10) - 1);
            uint16x8_t _fx = vcombine_u16(vmovn_u32(vandq_u32(vreinterpretq_u32_s32(_Xl), _v1024m1)), vmovn_u32(vandq_u32(vreinterpretq_u32_s32(_Xh), _v1024m1)));
            uint16x8_t _fy = vcombine_u16(vmovn_u32(vandq_u32(vreinterpretq_u32_s32(_Yl), _v1024m1)), vmovn_u32(vandq_u32(vreinterpretq_u32_s32(_Yh), _v1024m1)));

            uint16x8_t _alpha0 = vsubq_u16(vdupq_n_u16(1 << 10), _fx);
            uint16x8_t _alpha1 = _fx;
            uint16x8_t _beta0 = vsubq_u16(vdupq_n_u16(1 << 10), _fy);
            uint16x8_t _beta1 = _fy;

            int16x4_t _srcstride = vdup_n_s16(srcstride);
            int16x4_t _v3 = vdup_n_s16(3);

            int32x4_t _a0l = vmlal_s16(vmull_s16(_srcstride, _syl), _sxl, _v3);
            int32x4_t _a0h = vmlal_s16(vmull_s16(_srcstride, _syh), _sxh, _v3);
            int32x4_t _b0l = vaddw_s16(_a0l, _srcstride);
            int32x4_t _b0h = vaddw_s16(_a0h, _srcstride);
            int32x4_t _a1l = vaddw_s16(_a0l, _v3);
            int32x4_t _a1h = vaddw_s16(_a0h, _v3);
            int32x4_t _b1l = vaddw_s16(_b0l, _v3);
            int32x4_t _b1h = vaddw_s16(_b0h, _v3);

            uint8x8x3_t _a0 = uint8x8x3_t();
            uint8x8x3_t _a1 = uint8x8x3_t();
            uint8x8x3_t _b0 = uint8x8x3_t();
            uint8x8x3_t _b1 = uint8x8x3_t();
            {
                _a0 = vld3_lane_u8(src0 + vgetq_lane_s32(_a0l, 0), _a0, 0);
                _a1 = vld3_lane_u8(src0 + vgetq_lane_s32(_a1l, 0), _a1, 0);
                _b0 = vld3_lane_u8(src0 + vgetq_lane_s32(_b0l, 0), _b0, 0);
                _b1 = vld3_lane_u8(src0 + vgetq_lane_s32(_b1l, 0), _b1, 0);

                _a0 = vld3_lane_u8(src0 + vgetq_lane_s32(_a0l, 1), _a0, 1);
                _a1 = vld3_lane_u8(src0 + vgetq_lane_s32(_a1l, 1), _a1, 1);
                _b0 = vld3_lane_u8(src0 + vgetq_lane_s32(_b0l, 1), _b0, 1);
                _b1 = vld3_lane_u8(src0 + vgetq_lane_s32(_b1l, 1), _b1, 1);

                _a0 = vld3_lane_u8(src0 + vgetq_lane_s32(_a0l, 2), _a0, 2);
                _a1 = vld3_lane_u8(src0 + vgetq_lane_s32(_a1l, 2), _a1, 2);
                _b0 = vld3_lane_u8(src0 + vgetq_lane_s32(_b0l, 2), _b0, 2);
                _b1 = vld3_lane_u8(src0 + vgetq_lane_s32(_b1l, 2), _b1, 2);

                _a0 = vld3_lane_u8(src0 + vgetq_lane_s32(_a0l, 3), _a0, 3);
                _a1 = vld3_lane_u8(src0 + vgetq_lane_s32(_a1l, 3), _a1, 3);
                _b0 = vld3_lane_u8(src0 + vgetq_lane_s32(_b0l, 3), _b0, 3);
                _b1 = vld3_lane_u8(src0 + vgetq_lane_s32(_b1l, 3), _b1, 3);

                _a0 = vld3_lane_u8(src0 + vgetq_lane_s32(_a0h, 0), _a0, 4);
                _a1 = vld3_lane_u8(src0 + vgetq_lane_s32(_a1h, 0), _a1, 4);
                _b0 = vld3_lane_u8(src0 + vgetq_lane_s32(_b0h, 0), _b0, 4);
                _b1 = vld3_lane_u8(src0 + vgetq_lane_s32(_b1h, 0), _b1, 4);

                _a0 = vld3_lane_u8(src0 + vgetq_lane_s32(_a0h, 1), _a0, 5);
                _a1 = vld3_lane_u8(src0 + vgetq_lane_s32(_a1h, 1), _a1, 5);
                _b0 = vld3_lane_u8(src0 + vgetq_lane_s32(_b0h, 1), _b0, 5);
                _b1 = vld3_lane_u8(src0 + vgetq_lane_s32(_b1h, 1), _b1, 5);

                _a0 = vld3_lane_u8(src0 + vgetq_lane_s32(_a0h, 2), _a0, 6);
                _a1 = vld3_lane_u8(src0 + vgetq_lane_s32(_a1h, 2), _a1, 6);
                _b0 = vld3_lane_u8(src0 + vgetq_lane_s32(_b0h, 2), _b0, 6);
                _b1 = vld3_lane_u8(src0 + vgetq_lane_s32(_b1h, 2), _b1, 6);

                _a0 = vld3_lane_u8(src0 + vgetq_lane_s32(_a0h, 3), _a0, 7);
                _a1 = vld3_lane_u8(src0 + vgetq_lane_s32(_a1h, 3), _a1, 7);
                _b0 = vld3_lane_u8(src0 + vgetq_lane_s32(_b0h, 3), _b0, 7);
                _b1 = vld3_lane_u8(src0 + vgetq_lane_s32(_b1h, 3), _b1, 7);
            }

            uint16x8_t _a0_0 = vmovl_u8(_a0.val[0]);
            uint16x8_t _a0_1 = vmovl_u8(_a0.val[1]);
            uint16x8_t _a0_2 = vmovl_u8(_a0.val[2]);
            uint16x8_t _a1_0 = vmovl_u8(_a1.val[0]);
            uint16x8_t _a1_1 = vmovl_u8(_a1.val[1]);
            uint16x8_t _a1_2 = vmovl_u8(_a1.val[2]);
            uint16x8_t _b0_0 = vmovl_u8(_b0.val[0]);
            uint16x8_t _b0_1 = vmovl_u8(_b0.val[1]);
            uint16x8_t _b0_2 = vmovl_u8(_b0.val[2]);
            uint16x8_t _b1_0 = vmovl_u8(_b1.val[0]);
            uint16x8_t _b1_1 = vmovl_u8(_b1.val[1]);
            uint16x8_t _b1_2 = vmovl_u8(_b1.val[2]);

            uint16x4_t _a00_0l = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_low_u16(_a0_0), vget_low_u16(_alpha0)), vget_low_u16(_a1_0), vget_low_u16(_alpha1)), 5);
            uint16x4_t _a00_1l = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_low_u16(_a0_1), vget_low_u16(_alpha0)), vget_low_u16(_a1_1), vget_low_u16(_alpha1)), 5);
            uint16x4_t _a00_2l = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_low_u16(_a0_2), vget_low_u16(_alpha0)), vget_low_u16(_a1_2), vget_low_u16(_alpha1)), 5);
            uint16x4_t _a00_0h = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_high_u16(_a0_0), vget_high_u16(_alpha0)), vget_high_u16(_a1_0), vget_high_u16(_alpha1)), 5);
            uint16x4_t _a00_1h = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_high_u16(_a0_1), vget_high_u16(_alpha0)), vget_high_u16(_a1_1), vget_high_u16(_alpha1)), 5);
            uint16x4_t _a00_2h = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_high_u16(_a0_2), vget_high_u16(_alpha0)), vget_high_u16(_a1_2), vget_high_u16(_alpha1)), 5);
            uint16x4_t _b00_0l = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_low_u16(_b0_0), vget_low_u16(_alpha0)), vget_low_u16(_b1_0), vget_low_u16(_alpha1)), 5);
            uint16x4_t _b00_1l = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_low_u16(_b0_1), vget_low_u16(_alpha0)), vget_low_u16(_b1_1), vget_low_u16(_alpha1)), 5);
            uint16x4_t _b00_2l = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_low_u16(_b0_2), vget_low_u16(_alpha0)), vget_low_u16(_b1_2), vget_low_u16(_alpha1)), 5);
            uint16x4_t _b00_0h = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_high_u16(_b0_0), vget_high_u16(_alpha0)), vget_high_u16(_b1_0), vget_high_u16(_alpha1)), 5);
            uint16x4_t _b00_1h = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_high_u16(_b0_1), vget_high_u16(_alpha0)), vget_high_u16(_b1_1), vget_high_u16(_alpha1)), 5);
            uint16x4_t _b00_2h = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_high_u16(_b0_2), vget_high_u16(_alpha0)), vget_high_u16(_b1_2), vget_high_u16(_alpha1)), 5);

            uint16x4_t _dst_0l = vqshrn_n_u32(vmlal_u16(vmull_u16(_a00_0l, vget_low_u16(_beta0)), _b00_0l, vget_low_u16(_beta1)), 15);
            uint16x4_t _dst_1l = vqshrn_n_u32(vmlal_u16(vmull_u16(_a00_1l, vget_low_u16(_beta0)), _b00_1l, vget_low_u16(_beta1)), 15);
            uint16x4_t _dst_2l = vqshrn_n_u32(vmlal_u16(vmull_u16(_a00_2l, vget_low_u16(_beta0)), _b00_2l, vget_low_u16(_beta1)), 15);
            uint16x4_t _dst_0h = vqshrn_n_u32(vmlal_u16(vmull_u16(_a00_0h, vget_high_u16(_beta0)), _b00_0h, vget_high_u16(_beta1)), 15);
            uint16x4_t _dst_1h = vqshrn_n_u32(vmlal_u16(vmull_u16(_a00_1h, vget_high_u16(_beta0)), _b00_1h, vget_high_u16(_beta1)), 15);
            uint16x4_t _dst_2h = vqshrn_n_u32(vmlal_u16(vmull_u16(_a00_2h, vget_high_u16(_beta0)), _b00_2h, vget_high_u16(_beta1)), 15);

            uint8x8x3_t _dst;
            _dst.val[0] = vqmovn_u16(vcombine_u16(_dst_0l, _dst_0h));
            _dst.val[1] = vqmovn_u16(vcombine_u16(_dst_1l, _dst_1h));
            _dst.val[2] = vqmovn_u16(vcombine_u16(_dst_2l, _dst_2h));

            vst3_u8(dst0, _dst);

            dst0 += 3 * 8;
        }
#endif // __ARM_NEON
        for (; x < x_bound.end_inner; x++)
        {
            int X = X0 + adelta[x];
            int Y = Y0 + bdelta[x];

            short sx = SATURATE_CAST_SHORT((X >> 10));
            short sy = SATURATE_CAST_SHORT((Y >> 10));
            short fx = X & ((1 << 10) - 1);
            short fy = Y & ((1 << 10) - 1);

            short alpha0 = (1 << 10) - fx;
            short alpha1 = fx;

            short beta0 = (1 << 10) - fy;
            short beta1 = fy;

            short sx1 = sx + 1;
            short sy1 = sy + 1;

            const unsigned char* a0 = src0 + srcstride * sy + sx * 3;
            const unsigned char* a1 = src0 + srcstride * sy + sx1 * 3;
            const unsigned char* b0 = src0 + srcstride * sy1 + sx * 3;
            const unsigned char* b1 = src0 + srcstride * sy1 + sx1 * 3;

            dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);
            dst0[1] = (unsigned char)(((((unsigned short)((a0[1] * alpha0 + a1[1] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[1] * alpha0 + b1[1] * alpha1) >> 5) * beta1))) >> 15);
            dst0[2] = (unsigned char)(((((unsigned short)((a0[2] * alpha0 + a1[2] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[2] * alpha0 + b1[2] * alpha1) >> 5) * beta1))) >> 15);

            dst0 += 3;
        }
        for (; x < x_bound.end_outer; x++)
        {
            int X = X0 + adelta[x];
            int Y = Y0 + bdelta[x];

            short sx = SATURATE_CAST_SHORT((X >> 10));
            short sy = SATURATE_CAST_SHORT((Y >> 10));
            short fx = X & ((1 << 10) - 1);
            short fy = Y & ((1 << 10) - 1);

            short alpha0 = (1 << 10) - fx;
            short alpha1 = fx;

            short beta0 = (1 << 10) - fy;
            short beta1 = fy;

            short sx1 = sx + 1;
            short sy1 = sy + 1;

            const unsigned char* a0 = src0 + srcstride * sy + sx * 3;
            const unsigned char* a1 = src0 + srcstride * sy + sx1 * 3;
            const unsigned char* b0 = src0 + srcstride * sy1 + sx * 3;
            const unsigned char* b1 = src0 + srcstride * sy1 + sx1 * 3;

            if (sx < 0 || sx >= srcw || sy < 0 || sy >= srch)
            {
                a0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy < 0 || sy >= srch)
            {
                a1 = type != -233 ? border_color : dst0;
            }
            if (sx < 0 || sx >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b1 = type != -233 ? border_color : dst0;
            }

            dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);
            dst0[1] = (unsigned char)(((((unsigned short)((a0[1] * alpha0 + a1[1] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[1] * alpha0 + b1[1] * alpha1) >> 5) * beta1))) >> 15);
            dst0[2] = (unsigned char)(((((unsigned short)((a0[2] * alpha0 + a1[2] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[2] * alpha0 + b1[2] * alpha1) >> 5) * beta1))) >> 15);

            dst0 += 3;
        }
        for (; x < w; x++)
        {
            if (type != -233)
            {
                dst0[0] = border_color[0];
                dst0[1] = border_color[1];
                dst0[2] = border_color[2];
            }

            dst0 += 3;
        }

        dst0 += wgap;
    }
    for (; y < y_bound.end_outer; y++)
    {
        const Boundary& x_bound = x_bounds[y - y_bound.start_outer];

        int X0 = SATURATE_CAST_INT(((tm[1] * (y) + tm[2]) * (1 << 10)));
        int Y0 = SATURATE_CAST_INT(((tm[4] * (y) + tm[5]) * (1 << 10)));

        int x = 0;
        for (; x < x_bound.start_outer; x++)
        {
            if (type != -233)
            {
                dst0[0] = border_color[0];
                dst0[1] = border_color[1];
                dst0[2] = border_color[2];
            }

            dst0 += 3;
        }
        for (; x < x_bound.start_inner; x++)
        {
            int X = X0 + adelta[x];
            int Y = Y0 + bdelta[x];

            short sx = SATURATE_CAST_SHORT((X >> 10));
            short sy = SATURATE_CAST_SHORT((Y >> 10));
            short fx = X & ((1 << 10) - 1);
            short fy = Y & ((1 << 10) - 1);

            short alpha0 = (1 << 10) - fx;
            short alpha1 = fx;

            short beta0 = (1 << 10) - fy;
            short beta1 = fy;

            short sx1 = sx + 1;
            short sy1 = sy + 1;

            const unsigned char* a0 = src0 + srcstride * sy + sx * 3;
            const unsigned char* a1 = src0 + srcstride * sy + sx1 * 3;
            const unsigned char* b0 = src0 + srcstride * sy1 + sx * 3;
            const unsigned char* b1 = src0 + srcstride * sy1 + sx1 * 3;

            if (sx < 0 || sx >= srcw || sy < 0 || sy >= srch)
            {
                a0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy < 0 || sy >= srch)
            {
                a1 = type != -233 ? border_color : dst0;
            }
            if (sx < 0 || sx >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b1 = type != -233 ? border_color : dst0;
            }

            dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);
            dst0[1] = (unsigned char)(((((unsigned short)((a0[1] * alpha0 + a1[1] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[1] * alpha0 + b1[1] * alpha1) >> 5) * beta1))) >> 15);
            dst0[2] = (unsigned char)(((((unsigned short)((a0[2] * alpha0 + a1[2] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[2] * alpha0 + b1[2] * alpha1) >> 5) * beta1))) >> 15);

            dst0 += 3;
        }
        for (; x < x_bound.end_inner; x++)
        {
            int X = X0 + adelta[x];
            int Y = Y0 + bdelta[x];

            short sx = SATURATE_CAST_SHORT((X >> 10));
            short sy = SATURATE_CAST_SHORT((Y >> 10));
            short fx = X & ((1 << 10) - 1);
            short fy = Y & ((1 << 10) - 1);

            short alpha0 = (1 << 10) - fx;
            short alpha1 = fx;

            short beta0 = (1 << 10) - fy;
            short beta1 = fy;

            short sx1 = sx + 1;
            short sy1 = sy + 1;

            const unsigned char* a0 = src0 + srcstride * sy + sx * 3;
            const unsigned char* a1 = src0 + srcstride * sy + sx1 * 3;
            const unsigned char* b0 = src0 + srcstride * sy1 + sx * 3;
            const unsigned char* b1 = src0 + srcstride * sy1 + sx1 * 3;

            if (sy < 0 || sy >= srch)
            {
                a0 = type != -233 ? border_color : dst0;
                a1 = type != -233 ? border_color : dst0;
            }
            if (sy1 < 0 || sy1 >= srch)
            {
                b0 = type != -233 ? border_color : dst0;
                b1 = type != -233 ? border_color : dst0;
            }

            dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);
            dst0[1] = (unsigned char)(((((unsigned short)((a0[1] * alpha0 + a1[1] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[1] * alpha0 + b1[1] * alpha1) >> 5) * beta1))) >> 15);
            dst0[2] = (unsigned char)(((((unsigned short)((a0[2] * alpha0 + a1[2] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[2] * alpha0 + b1[2] * alpha1) >> 5) * beta1))) >> 15);

            dst0 += 3;
        }
        for (; x < x_bound.end_outer; x++)
        {
            int X = X0 + adelta[x];
            int Y = Y0 + bdelta[x];

            short sx = SATURATE_CAST_SHORT((X >> 10));
            short sy = SATURATE_CAST_SHORT((Y >> 10));
            short fx = X & ((1 << 10) - 1);
            short fy = Y & ((1 << 10) - 1);

            short alpha0 = (1 << 10) - fx;
            short alpha1 = fx;

            short beta0 = (1 << 10) - fy;
            short beta1 = fy;

            short sx1 = sx + 1;
            short sy1 = sy + 1;

            const unsigned char* a0 = src0 + srcstride * sy + sx * 3;
            const unsigned char* a1 = src0 + srcstride * sy + sx1 * 3;
            const unsigned char* b0 = src0 + srcstride * sy1 + sx * 3;
            const unsigned char* b1 = src0 + srcstride * sy1 + sx1 * 3;

            if (sx < 0 || sx >= srcw || sy < 0 || sy >= srch)
            {
                a0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy < 0 || sy >= srch)
            {
                a1 = type != -233 ? border_color : dst0;
            }
            if (sx < 0 || sx >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b1 = type != -233 ? border_color : dst0;
            }

            dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);
            dst0[1] = (unsigned char)(((((unsigned short)((a0[1] * alpha0 + a1[1] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[1] * alpha0 + b1[1] * alpha1) >> 5) * beta1))) >> 15);
            dst0[2] = (unsigned char)(((((unsigned short)((a0[2] * alpha0 + a1[2] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[2] * alpha0 + b1[2] * alpha1) >> 5) * beta1))) >> 15);

            dst0 += 3;
        }
        for (; x < w; x++)
        {
            if (type != -233)
            {
                dst0[0] = border_color[0];
                dst0[1] = border_color[1];
                dst0[2] = border_color[2];
            }

            dst0 += 3;
        }

        dst0 += wgap;
    }
    for (; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            if (type != -233)
            {
                dst0[0] = border_color[0];
                dst0[1] = border_color[1];
                dst0[2] = border_color[2];
            }

            dst0 += 3;
        }

        dst0 += wgap;
    }

#undef SATURATE_CAST_SHORT
#undef SATURATE_CAST_INT
}

void warpaffine_bilinear_c4(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, const float* tm, int type, unsigned int v)
{
    const unsigned char* border_color = (const unsigned char*)&v;
    const int wgap = stride - w * 4;

    const unsigned char* src0 = src;
    unsigned char* dst0 = dst;

#define SATURATE_CAST_SHORT(X) (short)::std::min(::std::max((int)(X), SHRT_MIN), SHRT_MAX)
#define SATURATE_CAST_INT(X)   (int)::std::min(::std::max((int)(X + (X >= 0.f ? 0.5f : -0.5f)), INT_MIN), INT_MAX)

    std::vector<int> adelta(w);
    std::vector<int> bdelta(w);
    for (int x = 0; x < w; x++)
    {
        adelta[x] = SATURATE_CAST_INT(tm[0] * x * (1 << 10));
        bdelta[x] = SATURATE_CAST_INT(tm[3] * x * (1 << 10));
    }

    Boundary y_bound;
    std::vector<Boundary> x_bounds;
    resolve_boundary(srcw, srch, w, h, tm, y_bound, x_bounds);

    //     fprintf(stderr, "y %d %d %d %d\n", y_bound.start_outer, y_bound.start_inner, y_bound.end_inner, y_bound.end_outer);

    int y = 0;
    for (; y < y_bound.start_outer; y++)
    {
        for (int x = 0; x < w; x++)
        {
            if (type != -233)
            {
                dst0[0] = border_color[0];
                dst0[1] = border_color[1];
                dst0[2] = border_color[2];
                dst0[3] = border_color[3];
            }

            dst0 += 4;
        }

        dst0 += wgap;
    }
    for (; y < y_bound.start_inner; y++)
    {
        const Boundary& x_bound = x_bounds[y - y_bound.start_outer];

        int X0 = SATURATE_CAST_INT(((tm[1] * (y) + tm[2]) * (1 << 10)));
        int Y0 = SATURATE_CAST_INT(((tm[4] * (y) + tm[5]) * (1 << 10)));

        int x = 0;
        for (; x < x_bound.start_outer; x++)
        {
            if (type != -233)
            {
                dst0[0] = border_color[0];
                dst0[1] = border_color[1];
                dst0[2] = border_color[2];
                dst0[3] = border_color[3];
            }

            dst0 += 4;
        }
        for (; x < x_bound.start_inner; x++)
        {
            int X = X0 + adelta[x];
            int Y = Y0 + bdelta[x];

            short sx = SATURATE_CAST_SHORT((X >> 10));
            short sy = SATURATE_CAST_SHORT((Y >> 10));
            short fx = X & ((1 << 10) - 1);
            short fy = Y & ((1 << 10) - 1);

            short alpha0 = (1 << 10) - fx;
            short alpha1 = fx;

            short beta0 = (1 << 10) - fy;
            short beta1 = fy;

            short sx1 = sx + 1;
            short sy1 = sy + 1;

            const unsigned char* a0 = src0 + srcstride * sy + sx * 4;
            const unsigned char* a1 = src0 + srcstride * sy + sx1 * 4;
            const unsigned char* b0 = src0 + srcstride * sy1 + sx * 4;
            const unsigned char* b1 = src0 + srcstride * sy1 + sx1 * 4;

            if (sx < 0 || sx >= srcw || sy < 0 || sy >= srch)
            {
                a0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy < 0 || sy >= srch)
            {
                a1 = type != -233 ? border_color : dst0;
            }
            if (sx < 0 || sx >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b1 = type != -233 ? border_color : dst0;
            }

            dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);
            dst0[1] = (unsigned char)(((((unsigned short)((a0[1] * alpha0 + a1[1] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[1] * alpha0 + b1[1] * alpha1) >> 5) * beta1))) >> 15);
            dst0[2] = (unsigned char)(((((unsigned short)((a0[2] * alpha0 + a1[2] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[2] * alpha0 + b1[2] * alpha1) >> 5) * beta1))) >> 15);
            dst0[3] = (unsigned char)(((((unsigned short)((a0[3] * alpha0 + a1[3] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[3] * alpha0 + b1[3] * alpha1) >> 5) * beta1))) >> 15);

            dst0 += 4;
        }
        for (; x < x_bound.end_inner; x++)
        {
            int X = X0 + adelta[x];
            int Y = Y0 + bdelta[x];

            short sx = SATURATE_CAST_SHORT((X >> 10));
            short sy = SATURATE_CAST_SHORT((Y >> 10));
            short fx = X & ((1 << 10) - 1);
            short fy = Y & ((1 << 10) - 1);

            short alpha0 = (1 << 10) - fx;
            short alpha1 = fx;

            short beta0 = (1 << 10) - fy;
            short beta1 = fy;

            short sx1 = sx + 1;
            short sy1 = sy + 1;

            const unsigned char* a0 = src0 + srcstride * sy + sx * 4;
            const unsigned char* a1 = src0 + srcstride * sy + sx1 * 4;
            const unsigned char* b0 = src0 + srcstride * sy1 + sx * 4;
            const unsigned char* b1 = src0 + srcstride * sy1 + sx1 * 4;

            if (sy < 0 || sy >= srch)
            {
                a0 = type != -233 ? border_color : dst0;
                a1 = type != -233 ? border_color : dst0;
            }
            if (sy1 < 0 || sy1 >= srch)
            {
                b0 = type != -233 ? border_color : dst0;
                b1 = type != -233 ? border_color : dst0;
            }

            dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);
            dst0[1] = (unsigned char)(((((unsigned short)((a0[1] * alpha0 + a1[1] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[1] * alpha0 + b1[1] * alpha1) >> 5) * beta1))) >> 15);
            dst0[2] = (unsigned char)(((((unsigned short)((a0[2] * alpha0 + a1[2] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[2] * alpha0 + b1[2] * alpha1) >> 5) * beta1))) >> 15);
            dst0[3] = (unsigned char)(((((unsigned short)((a0[3] * alpha0 + a1[3] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[3] * alpha0 + b1[3] * alpha1) >> 5) * beta1))) >> 15);

            dst0 += 4;
        }
        for (; x < x_bound.end_outer; x++)
        {
            int X = X0 + adelta[x];
            int Y = Y0 + bdelta[x];

            short sx = SATURATE_CAST_SHORT((X >> 10));
            short sy = SATURATE_CAST_SHORT((Y >> 10));
            short fx = X & ((1 << 10) - 1);
            short fy = Y & ((1 << 10) - 1);

            short alpha0 = (1 << 10) - fx;
            short alpha1 = fx;

            short beta0 = (1 << 10) - fy;
            short beta1 = fy;

            short sx1 = sx + 1;
            short sy1 = sy + 1;

            const unsigned char* a0 = src0 + srcstride * sy + sx * 4;
            const unsigned char* a1 = src0 + srcstride * sy + sx1 * 4;
            const unsigned char* b0 = src0 + srcstride * sy1 + sx * 4;
            const unsigned char* b1 = src0 + srcstride * sy1 + sx1 * 4;

            if (sx < 0 || sx >= srcw || sy < 0 || sy >= srch)
            {
                a0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy < 0 || sy >= srch)
            {
                a1 = type != -233 ? border_color : dst0;
            }
            if (sx < 0 || sx >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b1 = type != -233 ? border_color : dst0;
            }

            dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);
            dst0[1] = (unsigned char)(((((unsigned short)((a0[1] * alpha0 + a1[1] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[1] * alpha0 + b1[1] * alpha1) >> 5) * beta1))) >> 15);
            dst0[2] = (unsigned char)(((((unsigned short)((a0[2] * alpha0 + a1[2] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[2] * alpha0 + b1[2] * alpha1) >> 5) * beta1))) >> 15);
            dst0[3] = (unsigned char)(((((unsigned short)((a0[3] * alpha0 + a1[3] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[3] * alpha0 + b1[3] * alpha1) >> 5) * beta1))) >> 15);

            dst0 += 4;
        }
        for (; x < w; x++)
        {
            if (type != -233)
            {
                dst0[0] = border_color[0];
                dst0[1] = border_color[1];
                dst0[2] = border_color[2];
                dst0[3] = border_color[3];
            }

            dst0 += 4;
        }

        dst0 += wgap;
    }
    for (; y < y_bound.end_inner; y++)
    {
        const Boundary& x_bound = x_bounds[y - y_bound.start_outer];

        //         fprintf(stderr, "x %d = %d %d %d %d\n", y, x_bound.start_outer, x_bound.start_inner, x_bound.end_inner, x_bound.end_outer);

        int X0 = SATURATE_CAST_INT(((tm[1] * (y) + tm[2]) * (1 << 10)));
        int Y0 = SATURATE_CAST_INT(((tm[4] * (y) + tm[5]) * (1 << 10)));

        int x = 0;
        for (; x < x_bound.start_outer; x++)
        {
            if (type != -233)
            {
                dst0[0] = border_color[0];
                dst0[1] = border_color[1];
                dst0[2] = border_color[2];
                dst0[3] = border_color[3];
            }

            dst0 += 4;
        }
        for (; x < x_bound.start_inner; x++)
        {
            int X = X0 + adelta[x];
            int Y = Y0 + bdelta[x];

            short sx = SATURATE_CAST_SHORT((X >> 10));
            short sy = SATURATE_CAST_SHORT((Y >> 10));
            short fx = X & ((1 << 10) - 1);
            short fy = Y & ((1 << 10) - 1);

            short alpha0 = (1 << 10) - fx;
            short alpha1 = fx;

            short beta0 = (1 << 10) - fy;
            short beta1 = fy;

            short sx1 = sx + 1;
            short sy1 = sy + 1;

            const unsigned char* a0 = src0 + srcstride * sy + sx * 4;
            const unsigned char* a1 = src0 + srcstride * sy + sx1 * 4;
            const unsigned char* b0 = src0 + srcstride * sy1 + sx * 4;
            const unsigned char* b1 = src0 + srcstride * sy1 + sx1 * 4;

            if (sx < 0 || sx >= srcw || sy < 0 || sy >= srch)
            {
                a0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy < 0 || sy >= srch)
            {
                a1 = type != -233 ? border_color : dst0;
            }
            if (sx < 0 || sx >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b1 = type != -233 ? border_color : dst0;
            }

            dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);
            dst0[1] = (unsigned char)(((((unsigned short)((a0[1] * alpha0 + a1[1] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[1] * alpha0 + b1[1] * alpha1) >> 5) * beta1))) >> 15);
            dst0[2] = (unsigned char)(((((unsigned short)((a0[2] * alpha0 + a1[2] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[2] * alpha0 + b1[2] * alpha1) >> 5) * beta1))) >> 15);
            dst0[3] = (unsigned char)(((((unsigned short)((a0[3] * alpha0 + a1[3] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[3] * alpha0 + b1[3] * alpha1) >> 5) * beta1))) >> 15);

            dst0 += 4;
        }
#if __ARM_NEON
        for (; x + 7 < x_bound.end_inner; x += 8)
        {
            int32x4_t _Xl = vaddq_s32(vdupq_n_s32(X0), vld1q_s32(adelta.data() + x));
            int32x4_t _Xh = vaddq_s32(vdupq_n_s32(X0), vld1q_s32(adelta.data() + x + 4));
            int32x4_t _Yl = vaddq_s32(vdupq_n_s32(Y0), vld1q_s32(bdelta.data() + x));
            int32x4_t _Yh = vaddq_s32(vdupq_n_s32(Y0), vld1q_s32(bdelta.data() + x + 4));

            int16x4_t _sxl = vqshrn_n_s32(_Xl, 10);
            int16x4_t _sxh = vqshrn_n_s32(_Xh, 10);
            int16x4_t _syl = vqshrn_n_s32(_Yl, 10);
            int16x4_t _syh = vqshrn_n_s32(_Yh, 10);

            uint32x4_t _v1024m1 = vdupq_n_u32((1 << 10) - 1);
            uint16x8_t _fx = vcombine_u16(vmovn_u32(vandq_u32(vreinterpretq_u32_s32(_Xl), _v1024m1)), vmovn_u32(vandq_u32(vreinterpretq_u32_s32(_Xh), _v1024m1)));
            uint16x8_t _fy = vcombine_u16(vmovn_u32(vandq_u32(vreinterpretq_u32_s32(_Yl), _v1024m1)), vmovn_u32(vandq_u32(vreinterpretq_u32_s32(_Yh), _v1024m1)));

            uint16x8_t _alpha0 = vsubq_u16(vdupq_n_u16(1 << 10), _fx);
            uint16x8_t _alpha1 = _fx;
            uint16x8_t _beta0 = vsubq_u16(vdupq_n_u16(1 << 10), _fy);
            uint16x8_t _beta1 = _fy;

            int16x4_t _srcstride = vdup_n_s16(srcstride);
            int16x4_t _v4 = vdup_n_s16(4);

            int32x4_t _a0l = vmlal_s16(vmull_s16(_srcstride, _syl), _sxl, _v4);
            int32x4_t _a0h = vmlal_s16(vmull_s16(_srcstride, _syh), _sxh, _v4);
            int32x4_t _b0l = vaddw_s16(_a0l, _srcstride);
            int32x4_t _b0h = vaddw_s16(_a0h, _srcstride);
            int32x4_t _a1l = vaddw_s16(_a0l, _v4);
            int32x4_t _a1h = vaddw_s16(_a0h, _v4);
            int32x4_t _b1l = vaddw_s16(_b0l, _v4);
            int32x4_t _b1h = vaddw_s16(_b0h, _v4);

            uint8x8x4_t _a0 = uint8x8x4_t();
            uint8x8x4_t _a1 = uint8x8x4_t();
            uint8x8x4_t _b0 = uint8x8x4_t();
            uint8x8x4_t _b1 = uint8x8x4_t();
            {
                _a0 = vld4_lane_u8(src0 + vgetq_lane_s32(_a0l, 0), _a0, 0);
                _a1 = vld4_lane_u8(src0 + vgetq_lane_s32(_a1l, 0), _a1, 0);
                _b0 = vld4_lane_u8(src0 + vgetq_lane_s32(_b0l, 0), _b0, 0);
                _b1 = vld4_lane_u8(src0 + vgetq_lane_s32(_b1l, 0), _b1, 0);

                _a0 = vld4_lane_u8(src0 + vgetq_lane_s32(_a0l, 1), _a0, 1);
                _a1 = vld4_lane_u8(src0 + vgetq_lane_s32(_a1l, 1), _a1, 1);
                _b0 = vld4_lane_u8(src0 + vgetq_lane_s32(_b0l, 1), _b0, 1);
                _b1 = vld4_lane_u8(src0 + vgetq_lane_s32(_b1l, 1), _b1, 1);

                _a0 = vld4_lane_u8(src0 + vgetq_lane_s32(_a0l, 2), _a0, 2);
                _a1 = vld4_lane_u8(src0 + vgetq_lane_s32(_a1l, 2), _a1, 2);
                _b0 = vld4_lane_u8(src0 + vgetq_lane_s32(_b0l, 2), _b0, 2);
                _b1 = vld4_lane_u8(src0 + vgetq_lane_s32(_b1l, 2), _b1, 2);

                _a0 = vld4_lane_u8(src0 + vgetq_lane_s32(_a0l, 3), _a0, 3);
                _a1 = vld4_lane_u8(src0 + vgetq_lane_s32(_a1l, 3), _a1, 3);
                _b0 = vld4_lane_u8(src0 + vgetq_lane_s32(_b0l, 3), _b0, 3);
                _b1 = vld4_lane_u8(src0 + vgetq_lane_s32(_b1l, 3), _b1, 3);

                _a0 = vld4_lane_u8(src0 + vgetq_lane_s32(_a0h, 0), _a0, 4);
                _a1 = vld4_lane_u8(src0 + vgetq_lane_s32(_a1h, 0), _a1, 4);
                _b0 = vld4_lane_u8(src0 + vgetq_lane_s32(_b0h, 0), _b0, 4);
                _b1 = vld4_lane_u8(src0 + vgetq_lane_s32(_b1h, 0), _b1, 4);

                _a0 = vld4_lane_u8(src0 + vgetq_lane_s32(_a0h, 1), _a0, 5);
                _a1 = vld4_lane_u8(src0 + vgetq_lane_s32(_a1h, 1), _a1, 5);
                _b0 = vld4_lane_u8(src0 + vgetq_lane_s32(_b0h, 1), _b0, 5);
                _b1 = vld4_lane_u8(src0 + vgetq_lane_s32(_b1h, 1), _b1, 5);

                _a0 = vld4_lane_u8(src0 + vgetq_lane_s32(_a0h, 2), _a0, 6);
                _a1 = vld4_lane_u8(src0 + vgetq_lane_s32(_a1h, 2), _a1, 6);
                _b0 = vld4_lane_u8(src0 + vgetq_lane_s32(_b0h, 2), _b0, 6);
                _b1 = vld4_lane_u8(src0 + vgetq_lane_s32(_b1h, 2), _b1, 6);

                _a0 = vld4_lane_u8(src0 + vgetq_lane_s32(_a0h, 3), _a0, 7);
                _a1 = vld4_lane_u8(src0 + vgetq_lane_s32(_a1h, 3), _a1, 7);
                _b0 = vld4_lane_u8(src0 + vgetq_lane_s32(_b0h, 3), _b0, 7);
                _b1 = vld4_lane_u8(src0 + vgetq_lane_s32(_b1h, 3), _b1, 7);
            }

            uint16x8_t _a0_0 = vmovl_u8(_a0.val[0]);
            uint16x8_t _a0_1 = vmovl_u8(_a0.val[1]);
            uint16x8_t _a0_2 = vmovl_u8(_a0.val[2]);
            uint16x8_t _a0_3 = vmovl_u8(_a0.val[3]);
            uint16x8_t _a1_0 = vmovl_u8(_a1.val[0]);
            uint16x8_t _a1_1 = vmovl_u8(_a1.val[1]);
            uint16x8_t _a1_2 = vmovl_u8(_a1.val[2]);
            uint16x8_t _a1_3 = vmovl_u8(_a1.val[3]);
            uint16x8_t _b0_0 = vmovl_u8(_b0.val[0]);
            uint16x8_t _b0_1 = vmovl_u8(_b0.val[1]);
            uint16x8_t _b0_2 = vmovl_u8(_b0.val[2]);
            uint16x8_t _b0_3 = vmovl_u8(_b0.val[3]);
            uint16x8_t _b1_0 = vmovl_u8(_b1.val[0]);
            uint16x8_t _b1_1 = vmovl_u8(_b1.val[1]);
            uint16x8_t _b1_2 = vmovl_u8(_b1.val[2]);
            uint16x8_t _b1_3 = vmovl_u8(_b1.val[3]);

            uint16x4_t _a00_0l = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_low_u16(_a0_0), vget_low_u16(_alpha0)), vget_low_u16(_a1_0), vget_low_u16(_alpha1)), 5);
            uint16x4_t _a00_1l = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_low_u16(_a0_1), vget_low_u16(_alpha0)), vget_low_u16(_a1_1), vget_low_u16(_alpha1)), 5);
            uint16x4_t _a00_2l = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_low_u16(_a0_2), vget_low_u16(_alpha0)), vget_low_u16(_a1_2), vget_low_u16(_alpha1)), 5);
            uint16x4_t _a00_3l = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_low_u16(_a0_3), vget_low_u16(_alpha0)), vget_low_u16(_a1_3), vget_low_u16(_alpha1)), 5);
            uint16x4_t _a00_0h = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_high_u16(_a0_0), vget_high_u16(_alpha0)), vget_high_u16(_a1_0), vget_high_u16(_alpha1)), 5);
            uint16x4_t _a00_1h = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_high_u16(_a0_1), vget_high_u16(_alpha0)), vget_high_u16(_a1_1), vget_high_u16(_alpha1)), 5);
            uint16x4_t _a00_2h = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_high_u16(_a0_2), vget_high_u16(_alpha0)), vget_high_u16(_a1_2), vget_high_u16(_alpha1)), 5);
            uint16x4_t _a00_3h = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_high_u16(_a0_3), vget_high_u16(_alpha0)), vget_high_u16(_a1_3), vget_high_u16(_alpha1)), 5);
            uint16x4_t _b00_0l = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_low_u16(_b0_0), vget_low_u16(_alpha0)), vget_low_u16(_b1_0), vget_low_u16(_alpha1)), 5);
            uint16x4_t _b00_1l = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_low_u16(_b0_1), vget_low_u16(_alpha0)), vget_low_u16(_b1_1), vget_low_u16(_alpha1)), 5);
            uint16x4_t _b00_2l = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_low_u16(_b0_2), vget_low_u16(_alpha0)), vget_low_u16(_b1_2), vget_low_u16(_alpha1)), 5);
            uint16x4_t _b00_3l = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_low_u16(_b0_3), vget_low_u16(_alpha0)), vget_low_u16(_b1_3), vget_low_u16(_alpha1)), 5);
            uint16x4_t _b00_0h = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_high_u16(_b0_0), vget_high_u16(_alpha0)), vget_high_u16(_b1_0), vget_high_u16(_alpha1)), 5);
            uint16x4_t _b00_1h = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_high_u16(_b0_1), vget_high_u16(_alpha0)), vget_high_u16(_b1_1), vget_high_u16(_alpha1)), 5);
            uint16x4_t _b00_2h = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_high_u16(_b0_2), vget_high_u16(_alpha0)), vget_high_u16(_b1_2), vget_high_u16(_alpha1)), 5);
            uint16x4_t _b00_3h = vqshrn_n_u32(vmlal_u16(vmull_u16(vget_high_u16(_b0_3), vget_high_u16(_alpha0)), vget_high_u16(_b1_3), vget_high_u16(_alpha1)), 5);

            uint16x4_t _dst_0l = vqshrn_n_u32(vmlal_u16(vmull_u16(_a00_0l, vget_low_u16(_beta0)), _b00_0l, vget_low_u16(_beta1)), 15);
            uint16x4_t _dst_1l = vqshrn_n_u32(vmlal_u16(vmull_u16(_a00_1l, vget_low_u16(_beta0)), _b00_1l, vget_low_u16(_beta1)), 15);
            uint16x4_t _dst_2l = vqshrn_n_u32(vmlal_u16(vmull_u16(_a00_2l, vget_low_u16(_beta0)), _b00_2l, vget_low_u16(_beta1)), 15);
            uint16x4_t _dst_3l = vqshrn_n_u32(vmlal_u16(vmull_u16(_a00_3l, vget_low_u16(_beta0)), _b00_3l, vget_low_u16(_beta1)), 15);
            uint16x4_t _dst_0h = vqshrn_n_u32(vmlal_u16(vmull_u16(_a00_0h, vget_high_u16(_beta0)), _b00_0h, vget_high_u16(_beta1)), 15);
            uint16x4_t _dst_1h = vqshrn_n_u32(vmlal_u16(vmull_u16(_a00_1h, vget_high_u16(_beta0)), _b00_1h, vget_high_u16(_beta1)), 15);
            uint16x4_t _dst_2h = vqshrn_n_u32(vmlal_u16(vmull_u16(_a00_2h, vget_high_u16(_beta0)), _b00_2h, vget_high_u16(_beta1)), 15);
            uint16x4_t _dst_3h = vqshrn_n_u32(vmlal_u16(vmull_u16(_a00_3h, vget_high_u16(_beta0)), _b00_3h, vget_high_u16(_beta1)), 15);

            uint8x8x4_t _dst;
            _dst.val[0] = vqmovn_u16(vcombine_u16(_dst_0l, _dst_0h));
            _dst.val[1] = vqmovn_u16(vcombine_u16(_dst_1l, _dst_1h));
            _dst.val[2] = vqmovn_u16(vcombine_u16(_dst_2l, _dst_2h));
            _dst.val[3] = vqmovn_u16(vcombine_u16(_dst_3l, _dst_3h));

            vst4_u8(dst0, _dst);

            dst0 += 4 * 8;
        }
#endif // __ARM_NEON
        for (; x < x_bound.end_inner; x++)
        {
            int X = X0 + adelta[x];
            int Y = Y0 + bdelta[x];

            short sx = SATURATE_CAST_SHORT((X >> 10));
            short sy = SATURATE_CAST_SHORT((Y >> 10));
            short fx = X & ((1 << 10) - 1);
            short fy = Y & ((1 << 10) - 1);

            short alpha0 = (1 << 10) - fx;
            short alpha1 = fx;

            short beta0 = (1 << 10) - fy;
            short beta1 = fy;

            short sx1 = sx + 1;
            short sy1 = sy + 1;

            const unsigned char* a0 = src0 + srcstride * sy + sx * 4;
            const unsigned char* a1 = src0 + srcstride * sy + sx1 * 4;
            const unsigned char* b0 = src0 + srcstride * sy1 + sx * 4;
            const unsigned char* b1 = src0 + srcstride * sy1 + sx1 * 4;

            dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);
            dst0[1] = (unsigned char)(((((unsigned short)((a0[1] * alpha0 + a1[1] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[1] * alpha0 + b1[1] * alpha1) >> 5) * beta1))) >> 15);
            dst0[2] = (unsigned char)(((((unsigned short)((a0[2] * alpha0 + a1[2] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[2] * alpha0 + b1[2] * alpha1) >> 5) * beta1))) >> 15);
            dst0[3] = (unsigned char)(((((unsigned short)((a0[3] * alpha0 + a1[3] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[3] * alpha0 + b1[3] * alpha1) >> 5) * beta1))) >> 15);

            dst0 += 4;
        }
        for (; x < x_bound.end_outer; x++)
        {
            int X = X0 + adelta[x];
            int Y = Y0 + bdelta[x];

            short sx = SATURATE_CAST_SHORT((X >> 10));
            short sy = SATURATE_CAST_SHORT((Y >> 10));
            short fx = X & ((1 << 10) - 1);
            short fy = Y & ((1 << 10) - 1);

            short alpha0 = (1 << 10) - fx;
            short alpha1 = fx;

            short beta0 = (1 << 10) - fy;
            short beta1 = fy;

            short sx1 = sx + 1;
            short sy1 = sy + 1;

            const unsigned char* a0 = src0 + srcstride * sy + sx * 4;
            const unsigned char* a1 = src0 + srcstride * sy + sx1 * 4;
            const unsigned char* b0 = src0 + srcstride * sy1 + sx * 4;
            const unsigned char* b1 = src0 + srcstride * sy1 + sx1 * 4;

            if (sx < 0 || sx >= srcw || sy < 0 || sy >= srch)
            {
                a0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy < 0 || sy >= srch)
            {
                a1 = type != -233 ? border_color : dst0;
            }
            if (sx < 0 || sx >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b1 = type != -233 ? border_color : dst0;
            }

            dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);
            dst0[1] = (unsigned char)(((((unsigned short)((a0[1] * alpha0 + a1[1] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[1] * alpha0 + b1[1] * alpha1) >> 5) * beta1))) >> 15);
            dst0[2] = (unsigned char)(((((unsigned short)((a0[2] * alpha0 + a1[2] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[2] * alpha0 + b1[2] * alpha1) >> 5) * beta1))) >> 15);
            dst0[3] = (unsigned char)(((((unsigned short)((a0[3] * alpha0 + a1[3] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[3] * alpha0 + b1[3] * alpha1) >> 5) * beta1))) >> 15);

            dst0 += 4;
        }
        for (; x < w; x++)
        {
            if (type != -233)
            {
                dst0[0] = border_color[0];
                dst0[1] = border_color[1];
                dst0[2] = border_color[2];
                dst0[3] = border_color[3];
            }

            dst0 += 4;
        }

        dst0 += wgap;
    }
    for (; y < y_bound.end_outer; y++)
    {
        const Boundary& x_bound = x_bounds[y - y_bound.start_outer];

        int X0 = SATURATE_CAST_INT(((tm[1] * (y) + tm[2]) * (1 << 10)));
        int Y0 = SATURATE_CAST_INT(((tm[4] * (y) + tm[5]) * (1 << 10)));

        int x = 0;
        for (; x < x_bound.start_outer; x++)
        {
            if (type != -233)
            {
                dst0[0] = border_color[0];
                dst0[1] = border_color[1];
                dst0[2] = border_color[2];
                dst0[3] = border_color[3];
            }

            dst0 += 4;
        }
        for (; x < x_bound.start_inner; x++)
        {
            int X = X0 + adelta[x];
            int Y = Y0 + bdelta[x];

            short sx = SATURATE_CAST_SHORT((X >> 10));
            short sy = SATURATE_CAST_SHORT((Y >> 10));
            short fx = X & ((1 << 10) - 1);
            short fy = Y & ((1 << 10) - 1);

            short alpha0 = (1 << 10) - fx;
            short alpha1 = fx;

            short beta0 = (1 << 10) - fy;
            short beta1 = fy;

            short sx1 = sx + 1;
            short sy1 = sy + 1;

            const unsigned char* a0 = src0 + srcstride * sy + sx * 4;
            const unsigned char* a1 = src0 + srcstride * sy + sx1 * 4;
            const unsigned char* b0 = src0 + srcstride * sy1 + sx * 4;
            const unsigned char* b1 = src0 + srcstride * sy1 + sx1 * 4;

            if (sx < 0 || sx >= srcw || sy < 0 || sy >= srch)
            {
                a0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy < 0 || sy >= srch)
            {
                a1 = type != -233 ? border_color : dst0;
            }
            if (sx < 0 || sx >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b1 = type != -233 ? border_color : dst0;
            }

            dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);
            dst0[1] = (unsigned char)(((((unsigned short)((a0[1] * alpha0 + a1[1] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[1] * alpha0 + b1[1] * alpha1) >> 5) * beta1))) >> 15);
            dst0[2] = (unsigned char)(((((unsigned short)((a0[2] * alpha0 + a1[2] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[2] * alpha0 + b1[2] * alpha1) >> 5) * beta1))) >> 15);
            dst0[3] = (unsigned char)(((((unsigned short)((a0[3] * alpha0 + a1[3] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[3] * alpha0 + b1[3] * alpha1) >> 5) * beta1))) >> 15);

            dst0 += 4;
        }
        for (; x < x_bound.end_inner; x++)
        {
            int X = X0 + adelta[x];
            int Y = Y0 + bdelta[x];

            short sx = SATURATE_CAST_SHORT((X >> 10));
            short sy = SATURATE_CAST_SHORT((Y >> 10));
            short fx = X & ((1 << 10) - 1);
            short fy = Y & ((1 << 10) - 1);

            short alpha0 = (1 << 10) - fx;
            short alpha1 = fx;

            short beta0 = (1 << 10) - fy;
            short beta1 = fy;

            short sx1 = sx + 1;
            short sy1 = sy + 1;

            const unsigned char* a0 = src0 + srcstride * sy + sx * 4;
            const unsigned char* a1 = src0 + srcstride * sy + sx1 * 4;
            const unsigned char* b0 = src0 + srcstride * sy1 + sx * 4;
            const unsigned char* b1 = src0 + srcstride * sy1 + sx1 * 4;

            if (sy < 0 || sy >= srch)
            {
                a0 = type != -233 ? border_color : dst0;
                a1 = type != -233 ? border_color : dst0;
            }
            if (sy1 < 0 || sy1 >= srch)
            {
                b0 = type != -233 ? border_color : dst0;
                b1 = type != -233 ? border_color : dst0;
            }

            dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);
            dst0[1] = (unsigned char)(((((unsigned short)((a0[1] * alpha0 + a1[1] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[1] * alpha0 + b1[1] * alpha1) >> 5) * beta1))) >> 15);
            dst0[2] = (unsigned char)(((((unsigned short)((a0[2] * alpha0 + a1[2] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[2] * alpha0 + b1[2] * alpha1) >> 5) * beta1))) >> 15);
            dst0[3] = (unsigned char)(((((unsigned short)((a0[3] * alpha0 + a1[3] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[3] * alpha0 + b1[3] * alpha1) >> 5) * beta1))) >> 15);

            dst0 += 4;
        }
        for (; x < x_bound.end_outer; x++)
        {
            int X = X0 + adelta[x];
            int Y = Y0 + bdelta[x];

            short sx = SATURATE_CAST_SHORT((X >> 10));
            short sy = SATURATE_CAST_SHORT((Y >> 10));
            short fx = X & ((1 << 10) - 1);
            short fy = Y & ((1 << 10) - 1);

            short alpha0 = (1 << 10) - fx;
            short alpha1 = fx;

            short beta0 = (1 << 10) - fy;
            short beta1 = fy;

            short sx1 = sx + 1;
            short sy1 = sy + 1;

            const unsigned char* a0 = src0 + srcstride * sy + sx * 4;
            const unsigned char* a1 = src0 + srcstride * sy + sx1 * 4;
            const unsigned char* b0 = src0 + srcstride * sy1 + sx * 4;
            const unsigned char* b1 = src0 + srcstride * sy1 + sx1 * 4;

            if (sx < 0 || sx >= srcw || sy < 0 || sy >= srch)
            {
                a0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy < 0 || sy >= srch)
            {
                a1 = type != -233 ? border_color : dst0;
            }
            if (sx < 0 || sx >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b0 = type != -233 ? border_color : dst0;
            }
            if (sx1 < 0 || sx1 >= srcw || sy1 < 0 || sy1 >= srch)
            {
                b1 = type != -233 ? border_color : dst0;
            }

            dst0[0] = (unsigned char)(((((unsigned short)((a0[0] * alpha0 + a1[0] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[0] * alpha0 + b1[0] * alpha1) >> 5) * beta1))) >> 15);
            dst0[1] = (unsigned char)(((((unsigned short)((a0[1] * alpha0 + a1[1] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[1] * alpha0 + b1[1] * alpha1) >> 5) * beta1))) >> 15);
            dst0[2] = (unsigned char)(((((unsigned short)((a0[2] * alpha0 + a1[2] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[2] * alpha0 + b1[2] * alpha1) >> 5) * beta1))) >> 15);
            dst0[3] = (unsigned char)(((((unsigned short)((a0[3] * alpha0 + a1[3] * alpha1) >> 5) * beta0)) + (((unsigned short)((b0[3] * alpha0 + b1[3] * alpha1) >> 5) * beta1))) >> 15);

            dst0 += 4;
        }
        for (; x < w; x++)
        {
            if (type != -233)
            {
                dst0[0] = border_color[0];
                dst0[1] = border_color[1];
                dst0[2] = border_color[2];
                dst0[3] = border_color[3];
            }

            dst0 += 4;
        }

        dst0 += wgap;
    }
    for (; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            if (type != -233)
            {
                dst0[0] = border_color[0];
                dst0[1] = border_color[1];
                dst0[2] = border_color[2];
                dst0[3] = border_color[3];
            }

            dst0 += 4;
        }

        dst0 += wgap;
    }

#undef SATURATE_CAST_SHORT
#undef SATURATE_CAST_INT
}

void warpaffine_bilinear_yuv420sp(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, const float* tm, int type, unsigned int v)
{
    // assert srcw % 2 == 0
    // assert srch % 2 == 0
    // assert w % 2 == 0
    // assert h % 2 == 0

    const unsigned char* border_color = (const unsigned char*)&v;

    unsigned int v_y;
    unsigned int v_uv;
    unsigned char* border_color_y = (unsigned char*)&v_y;
    unsigned char* border_color_uv = (unsigned char*)&v_uv;
    border_color_y[0] = border_color[0];
    border_color_uv[0] = border_color[1];
    border_color_uv[1] = border_color[2];

    const unsigned char* srcY = src;
    unsigned char* dstY = dst;
    warpaffine_bilinear_c1(srcY, srcw, srch, dstY, w, h, tm, type, v_y);

    const unsigned char* srcUV = src + srcw * srch;
    unsigned char* dstUV = dst + w * h;
    warpaffine_bilinear_c2(srcUV, srcw / 2, srch / 2, dstUV, w / 2, h / 2, tm, type, v_uv);
}
#endif // NCNN_PIXEL_AFFINE

} // namespace ncnn
