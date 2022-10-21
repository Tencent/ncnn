// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// coord compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to coord writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "gridsample.h"

#include <cmath>
#include <math.h>

namespace ncnn {

GridSample::GridSample()
{
    one_blob_only = false;
    support_inplace = false;
}

int GridSample::load_param(const ParamDict& pd)
{
    resize_type = pd.get(0, 1);
    padding_mode = pd.get(1, 1);
    align_corner = pd.get(2, 0);

    if (resize_type < 1 || resize_type > 3)
    {
        NCNN_LOGE("unsupported resize type %d", resize_type);
        return -1;
    }

    if (padding_mode < 1 || padding_mode > 3)
    {
        NCNN_LOGE("unsupported padding mode %d", padding_mode);
        return -1;
    }

    return 0;
}

#if defined(__GNUC__) && defined(__powerpc__) && defined(__ALTIVEC__)
// NOTE gcc altivec optimized version produce wrong result
// so I have to disable vectorize here  --- nihui
__attribute__((optimize("no-tree-vectorize")))
#endif

// Restore normalized location to acutal image location
//   When align_corners is true:
//     Normalized location (-1, -1) points to the top-left pixel.
//     Normalized location (1, 1) points to the bottom-tight pixel.
//   When align_corners is false [default]:
//     Normalized location (-1, -1) points to the top-left pixel minus half
//     pixel coord both directions, i.e, (-0.5, -0.5) coord acutal image space.
//     Normalized location (1, 1) points to the bottom-tight pixel plus half
//     pixel coord both directions, i.e. (H - 0.5, W - 0.5) coord acutal image space.
static float
grid_sample_unormalize(int w, float coordx, int align_corner)
{
    return align_corner ? (coordx + 1) / 2.f * (w - 1) : ((coordx + 1) * w - 1) / 2.f;
}

static float border_coord(float coord, int border)
{
    return std::min(static_cast<float>(border), std::max(coord, static_cast<float>(0)));
}

// Reflects coordinates until they fall between low and high (inclusive).
static float reflect_coord(float coord, int low, int high)
{
    if (low == high)
    {
        return 0;
    }
    float min = static_cast<int>(low) / 2;
    float span = static_cast<float>(high - low) / 2;
    coord = std::fabs(coord - min);
    // `fmod` returns same sign as `coord`, which is positive after the `fabs` above.
    float extra = std::fmod(coord, span);
    int flips = static_cast<int>(std::floor(coord / span));

    return flips % 2 ? (span - extra + min) : (extra + min);
}

static float compute_coord(float sx, int w,
                           int padding_mode, int align_corner)
{
    // correct the coordinates according to the padding_mode
    if (padding_mode == 2) // border
    {
        // clip coordinates to image borders
        sx = border_coord(sx, w - 1);
    }
    else if (padding_mode == 3) // reflection
    {
        // reflect coordinates by image borders
        if (align_corner)
        {
            sx = reflect_coord(sx, 0, 2 * (w - 1));
        }
        else
        {
            sx = reflect_coord(sx, -1, 2 * w - 1);
        }
        // clip coordinates to image borders
        sx = border_coord(sx, w - 1);
    }
    return sx;
}

static float get_coord(float x, int w, int padding_mode, int align_corner)
{
    // compute the origin coordinates
    float sx = grid_sample_unormalize(w, x, align_corner);

    // correct the coordinates according to the padding_mode
    float coord = compute_coord(sx, w, padding_mode, align_corner);

    return coord;
}

static bool in_bounds(int x, int y, int w, int h)
{
    return x >= 0 && y >= 0 && x < w && y < h;
}

static bool in_bounds_3d(int x, int y, int z, int w, int h, int d)
{
    return x >= 0 && y >= 0 && z >= 0 && x < w && y < h && z < d;
}

static float get_value_bounded(const float* data, float x, float y, int w, int h,
                               int padding_mode, int align_corner)
{
    x = compute_coord(x, w, padding_mode, align_corner);
    y = compute_coord(y, h, padding_mode, align_corner);

    int ix = static_cast<int>(x);
    int iy = static_cast<int>(y);

    return in_bounds(ix, iy, w, h) ? data[iy * w + ix] : 0.f;
}

// Based on
// https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
static float cubic_convolution1(float x, float A)
{
    return ((A + 2) * x - (A + 3)) * x * x + 1;
}

static float cubic_convolution2(float x, float A)
{
    return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
}

static void get_cubic_upsample_coefficients(float coeffs[4], float t)
{
    float A = -0.75;

    float x1 = t;
    coeffs[0] = cubic_convolution2(x1 + 1.0, A);
    coeffs[1] = cubic_convolution1(x1, A);

    // opposite coefficients
    float x2 = 1.0 - t;
    coeffs[2] = cubic_convolution1(x2, A);
    coeffs[3] = cubic_convolution2(x2 + 1.0, A);
}

static void get_cubic_coefficients(float (&coeffs)[4], const float tx)
{
    float x;
    float A = -0.75f;
    x = tx + float(1); // 1 < x = |-1 - tx| < 2
    coeffs[0] = ((A * x - float(5) * A) * x + float(8) * A) * x - float(4) * A;
    x = tx; // x = |0 - tx| <= 1
    coeffs[1] = ((A + float(2)) * x - (A + float(3))) * x * x + float(1);
    x = float(1) - tx; // x = |1 - tx| <= 1
    coeffs[2] = ((A + float(2)) * x - (A + float(3))) * x * x + float(1);
    x = float(2) - tx; // 1 < x = |2 - tx| < 2
    coeffs[3] = ((A * x - float(5) * A) * x + float(8) * A) * x - float(4) * A;
}

static float cubic_interp1d(float x0, float x1, float x2, float x3, float t)
{
    float coeffs[4];
    get_cubic_upsample_coefficients(coeffs, t);

    return x0 * coeffs[0] + x1 * coeffs[1] + x2 * coeffs[2] + x3 * coeffs[3];
}

int GridSample::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& grid = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;

    if (dims == 3)
    {
        int outw = grid.h;
        int outh = grid.c;

        top_blob.create(outw, outh, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;
        if (resize_type == 1) // bilinear
        {
            #pragma omp parallel for num_threads(opt.num_threads) // collapse(2)
            for (int row = 0; row < outh; row++)
            {
                for (int col = 0; col < outw; col++)
                {
                    // const float* gridptr = grid.depth(row).row(col);
                    const float* gridptr = grid.channel(row).row(col);

                    // get the coordinate of every output point
                    float ix = gridptr[0];
                    float iy = gridptr[1];

                    ix = get_coord(ix, w, padding_mode, align_corner);
                    iy = get_coord(iy, h, padding_mode, align_corner);

                    // for 3d, we used north-east-south-west
                    int xnw = static_cast<int>(std::floor(ix));
                    int xne = xnw + 1;
                    int xsw = xnw;
                    int xse = xne;

                    int ynw = static_cast<int>(std::floor(iy));
                    int yne = ynw;
                    int ysw = ynw + 1;
                    int yse = ysw;

                    // get the coeff of every output point
                    float fnw = (xse - ix) * (yse - iy);
                    float fne = (ix - xsw) * (ysw - iy);
                    float fsw = (xne - ix) * (iy - yne);
                    float fse = (ix - xnw) * (iy - ynw);

                    for (int q = 0; q < channels; q++)
                    {
                        const float* ptr = bottom_blob.channel(q);
                        float* outptr = top_blob.channel(q);

                        float ans = 0.f;

                        if (in_bounds(xnw, ynw, w, h))
                        {
                            ans += ptr[ynw * w + xnw] * fnw;
                        }
                        if (in_bounds(xne, yne, w, h))
                        {
                            ans += ptr[yne * w + xne] * fne;
                        }
                        if (in_bounds(xsw, ysw, w, h))
                        {
                            ans += ptr[ysw * w + xsw] * fsw;
                        }
                        if (in_bounds(xse, yse, w, h))
                        {
                            ans += ptr[yse * w + xse] * fse;
                        }

                        outptr[row * outw + col] = ans;
                    }
                }
            }
        }
        else if (resize_type == 2) //nearest
        {
            #pragma omp parallel for num_threads(opt.num_threads) // collapse(2)
            for (int row = 0; row < outh; row++)
            {
                for (int col = 0; col < outw; col++)
                {
                    // const float* gridptr = grid.depth(row).row(col);
                    const float* gridptr = grid.channel(row).row(col);

                    // get the coordinate of every output point
                    float ix = gridptr[0];
                    float iy = gridptr[1];

                    ix = get_coord(ix, w, padding_mode, align_corner);
                    iy = get_coord(iy, h, padding_mode, align_corner);

                    int x = static_cast<int>(round(ix));
                    int y = static_cast<int>(round(iy));

                    for (int q = 0; q < channels; q++)
                    {
                        const float* ptr = bottom_blob.channel(q);
                        float* outptr = top_blob.channel(q);

                        float ans = 0.f;

                        if (in_bounds(x, y, w, h))
                        {
                            ans += ptr[y * w + x];
                        }

                        outptr[row * outw + col] = ans;
                    }
                }
            }
        }
        else if (resize_type == 3) // bicubic
        {
            #pragma omp parallel for num_threads(opt.num_threads) // collapse(2)
            for (int row = 0; row < outh; row++)
            {
                for (int col = 0; col < outw; col++)
                {
                    // const float* gridptr = grid.depth(row).row(col);
                    const float* gridptr = grid.channel(row).row(col);

                    // get the coordinate of every output point
                    float ix = gridptr[0];
                    float iy = gridptr[1];

                    ix = grid_sample_unormalize(w, ix, align_corner);
                    iy = grid_sample_unormalize(h, iy, align_corner);

                    float xnw = std::floor(ix);
                    float ynw = std::floor(iy);

                    const float tx = ix - xnw;
                    const float ty = iy - ynw;

                    // float coeff_x[4];
                    // float coeff_y[4];

                    // get_cubic_coefficients(coeff_x, tx);
                    // get_cubic_coefficients(coeff_y, ty);

                    for (int q = 0; q < channels; q++)
                    {
                        const float* ptr = bottom_blob.channel(q);
                        float* outptr = top_blob.channel(q);

                        float coefficients[4];

                        // float interp_x[4];

                        // Interpolate 4 values in the x directon
                        for (int i = 0; i < 4; i++)
                        {
                            // interp_x[i] =
                            //     coeff_x[0] * get_value_bounded(ptr, xnw - 1, ynw - 1 + i, w, h, padding_mode, align_corner)+
                            //     coeff_x[1] * get_value_bounded(ptr, xnw + 0, ynw - 1 + i, w, h, padding_mode, align_corner)+
                            //     coeff_x[2] * get_value_bounded(ptr, xnw + 1, ynw - 1 + i, w, h, padding_mode, align_corner)+
                            //     coeff_x[3] * get_value_bounded(ptr, xnw + 2, ynw - 1 + i, w, h, padding_mode, align_corner);

                            coefficients[i] = cubic_interp1d(
                                                  get_value_bounded(ptr, xnw - 1, ynw - 1 + i, w, h, padding_mode, align_corner),
                                                  get_value_bounded(ptr, xnw + 0, ynw - 1 + i, w, h, padding_mode, align_corner),
                                                  get_value_bounded(ptr, xnw + 1, ynw - 1 + i, w, h, padding_mode, align_corner),
                                                  get_value_bounded(ptr, xnw + 2, ynw - 1 + i, w, h, padding_mode, align_corner),
                                                  tx);
                        }

                        // Interpolate the 4 values in the y direction
                        // outptr[row * outw + col] =  coeff_y[0] * interp_x[0] + coeff_y[1] * interp_x[1] +
                        //                             coeff_y[2] * interp_x[2] + coeff_y[3] * interp_x[3];

                        // Interpolate in the y direction
                        outptr[row * outw + col] = cubic_interp1d(
                                                       coefficients[0],
                                                       coefficients[1],
                                                       coefficients[2],
                                                       coefficients[3],
                                                       ty);
                    }
                }
            }
        }
    }

    if (dims == 4)
    {
        int outw = grid.h;
        int outh = grid.d;
        int outd = grid.c;

        top_blob.create(outw, outh, outd, channels, elemsize, opt.blob_allocator);
        if (resize_type == 1) // bilinear
        {
            #pragma omp parallel for num_threads(opt.num_threads) // collapse(2)
            for (int dep = 0; dep < outd; dep++)
            {
                for (int row = 0; row < outh; row++)
                {
                    for (int col = 0; col < outw; col++)
                    {
                        const float* gridptr = grid.channel(dep).depth(row).row(col);

                        // get the coordinate of every output point
                        float ix = gridptr[0];
                        float iy = gridptr[1];
                        float iz = gridptr[2];

                        ix = get_coord(ix, w, padding_mode, align_corner);
                        iy = get_coord(iy, h, padding_mode, align_corner);
                        iz = get_coord(iz, d, padding_mode, align_corner);

                        // for 3d, we used north-east-south-west
                        // for 4d, we add top-bottom
                        int xtnw = static_cast<int>(std::floor(ix));
                        int ytnw = static_cast<int>(std::floor(iy));
                        int ztnw = static_cast<int>(std::floor(iz));

                        int xtne = xtnw + 1;
                        int ytne = ytnw;
                        int ztne = ztnw;

                        int xtsw = xtnw;
                        int ytsw = ytnw + 1;
                        int ztsw = ztnw;

                        int xtse = xtnw + 1;
                        int ytse = ytnw + 1;
                        int ztse = ztnw;

                        int xbnw = xtnw;
                        int ybnw = ytnw;
                        int zbnw = ztnw + 1;

                        int xbne = xtnw + 1;
                        int ybne = ytnw;
                        int zbne = ztnw + 1;

                        int xbsw = xtnw;
                        int ybsw = ytnw + 1;
                        int zbsw = ztnw + 1;

                        int xbse = xtnw + 1;
                        int ybse = ytnw + 1;
                        int zbse = ztnw + 1;

                        // get surfaces to each neighbor:
                        float ftnw = (xbse - ix) * (ybse - iy) * (zbse - iz);
                        float ftne = (ix - xbsw) * (ybsw - iy) * (zbsw - iz);
                        float ftsw = (xbne - ix) * (iy - ybne) * (zbne - iz);
                        float ftse = (ix - xbnw) * (iy - ybnw) * (zbnw - iz);
                        float fbnw = (xtse - ix) * (ytse - iy) * (iz - ztse);
                        float fbne = (ix - xtsw) * (ytsw - iy) * (iz - ztsw);
                        float fbsw = (xtne - ix) * (iy - ytne) * (iz - ztne);
                        float fbse = (ix - xtnw) * (iy - ytnw) * (iz - ztnw);

                        for (int q = 0; q < channels; q++)
                        {
                            const float* ptr = bottom_blob.channel(q);
                            float* outptr = top_blob.channel(q);

                            float ans = 0.f;

                            if (in_bounds_3d(ztnw, ytnw, xtnw, d, h, w))
                            {
                                ans += ptr[ztnw * h * w + ytnw * w + xtnw] * ftnw;
                            }
                            if (in_bounds_3d(ztne, ytne, xtne, d, h, w))
                            {
                                ans += ptr[ztne * h * w + ytne * w + xtne] * ftne;
                            }
                            if (in_bounds_3d(ztsw, ytsw, xtsw, d, h, w))
                            {
                                ans += ptr[ztsw * h * w + ytsw * w + xtsw] * ftsw;
                            }
                            if (in_bounds_3d(ztse, ytse, xtse, d, h, w))
                            {
                                ans += ptr[ztse * h * w + ytse * w + xtse] * ftse;
                            }
                            if (in_bounds_3d(zbnw, ybnw, xbnw, d, h, w))
                            {
                                ans += ptr[zbnw * h * w + ybnw * w + xbnw] * fbnw;
                            }
                            if (in_bounds_3d(zbne, ybne, xbne, d, h, w))
                            {
                                ans += ptr[zbne * h * w + ybne * w + xbne] * fbne;
                            }
                            if (in_bounds_3d(zbsw, ybsw, xbsw, d, h, w))
                            {
                                ans += ptr[zbsw * h * w + ybsw * w + xbsw] * fbsw;
                            }
                            if (in_bounds_3d(zbse, ybse, xbse, d, h, w))
                            {
                                ans += ptr[zbse * h * w + ybse * w + xbse] * fbse;
                            }

                            outptr[dep * outh * outw + row * outw + col] = ans;
                        }
                    }
                }
            }
        }
        else if (resize_type == 2) //nearest
        {
            #pragma omp parallel for num_threads(opt.num_threads) // collapse(2)
            for (int dep = 0; dep < outd; dep++)
            {
                for (int row = 0; row < outh; row++)
                {
                    for (int col = 0; col < outw; col++)
                    {
                        const float* gridptr = grid.channel(dep).depth(row).row(col);

                        // get the coordinate of every output point
                        float ix = gridptr[0];
                        float iy = gridptr[1];
                        float iz = gridptr[2];

                        ix = get_coord(ix, w, padding_mode, align_corner);
                        iy = get_coord(iy, h, padding_mode, align_corner);
                        iz = get_coord(iz, d, padding_mode, align_corner);

                        int x = static_cast<int>(round(ix));
                        int y = static_cast<int>(round(iy));
                        int z = static_cast<int>(round(iz));

                        for (int q = 0; q < channels; q++)
                        {
                            const float* ptr = bottom_blob.channel(q);
                            float* outptr = top_blob.channel(q);

                            float ans = 0.f;

                            if (in_bounds_3d(x, y, z, w, h, d))
                            {
                                ans += ptr[z * h * w + y * w + x];
                            }

                            outptr[dep * outh * outw + row * outw + col] = ans;
                        }
                    }
                }
            }
        }
        else if (resize_type == 3)
        {
            NCNN_LOGE("unsupported bicubic when dims == 4");
            return -1;
        }
    }

    return 0;
}

} // namespace ncnn
