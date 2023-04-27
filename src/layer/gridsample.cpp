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

#include <math.h>

namespace ncnn {

GridSample::GridSample()
{
    one_blob_only = false;
    support_inplace = false;
}

int GridSample::load_param(const ParamDict& pd)
{
    sample_type = pd.get(0, 1);
    padding_mode = pd.get(1, 1);
    align_corner = pd.get(2, 0);

    if (sample_type < 1 || sample_type > 3)
    {
        NCNN_LOGE("unsupported sample type %d", sample_type);
        return -1;
    }

    if (padding_mode < 1 || padding_mode > 3)
    {
        NCNN_LOGE("unsupported padding mode %d", padding_mode);
        return -1;
    }

    return 0;
}

// Restore normalized location to acutal image location
//   When align_corners is true:
//     Normalized location (-1, -1) points to the top-left pixel.
//     Normalized location (1, 1) points to the bottom-tight pixel.
//   When align_corners is false [default]:
//     Normalized location (-1, -1) points to the top-left pixel minus half
//     pixel coord both directions, i.e, (-0.5, -0.5) coord acutal image space.
//     Normalized location (1, 1) points to the bottom-tight pixel plus half
//     pixel coord both directions, i.e. (H - 0.5, W - 0.5) coord acutal image space.
static float grid_sample_unormalize(int w, float coordx, int align_corner)
{
    return align_corner ? (coordx + 1) / 2.f * (w - 1) : ((coordx + 1) * w - 1) / 2.f;
}

static float border_coord(int x, int border)
{
    return std::min(border, std::max(x, 0));
}

static float reflect_coord(float x, int high)
{
    x = abs(x);
    x = high - abs(x - high);
    return x;
}

static int compute_coord(int sx, int w, int padding_mode, int align_corner)
{
    if (padding_mode == 2) // border
    {
        sx = border_coord(sx, w - 1);
    }
    else if (padding_mode == 3) // reflection
    {
        if (align_corner)
        {
            sx = reflect_coord(sx, w - 1);
        }
        else
        {
            sx = static_cast<int>(reflect_coord(sx + 0.5f, w) - 0.5f);
            sx = border_coord(sx, w - 1);
        }
    }

    return sx;
}

static bool in_bounds(const Mat& image, int x, int y)
{
    return x >= 0 && y >= 0 && x < image.w && y < image.h;
}

static bool in_bounds(const Mat& image, int x, int y, int z)
{
    return x >= 0 && y >= 0 && z >= 0 && x < image.w && y < image.h && z < image.c;
}

static float get_value_bounded(const Mat& image, int x, int y)
{
    return in_bounds(image, x, y) ? image.row(y)[x] : 0.f;
}

static float get_value_bounded(const Mat& image, int x, int y, int z)
{
    return in_bounds(image, x, y, z) ? image.channel(z).row(y)[x] : 0.f;
}

static float get_value_bounded(const Mat& image, int x, int y, int padding_mode, int align_corner)
{
    x = compute_coord(x, image.w, padding_mode, align_corner);
    y = compute_coord(y, image.h, padding_mode, align_corner);

    return get_value_bounded(image, x, y);
}

static float get_value_bounded(const Mat& image, int x, int y, int z, int padding_mode, int align_corner)
{
    x = compute_coord(x, image.w, padding_mode, align_corner);
    y = compute_coord(y, image.h, padding_mode, align_corner);
    z = compute_coord(z, image.c, padding_mode, align_corner);

    return get_value_bounded(image, x, y, z);
}

static inline void interpolate_cubic(float fx, float* coeffs)
{
    const float A = -0.75f;

    float fx0 = fx + 1;
    float fx1 = fx;
    float fx2 = 1 - fx;
    // float fx3 = 2 - fx;

    coeffs[0] = A * fx0 * fx0 * fx0 - 5 * A * fx0 * fx0 + 8 * A * fx0 - 4 * A;
    coeffs[1] = (A + 2) * fx1 * fx1 * fx1 - (A + 3) * fx1 * fx1 + 1;
    coeffs[2] = (A + 2) * fx2 * fx2 * fx2 - (A + 3) * fx2 * fx2 + 1;
    coeffs[3] = 1.f - coeffs[0] - coeffs[1] - coeffs[2];
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

        if (sample_type == 1) // bilinear
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat image = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q);

                for (int y = 0; y < outh; y++)
                {
                    const float* gridptr = grid.channel(y);

                    for (int x = 0; x < outw; x++)
                    {
                        float sample_x = gridptr[0];
                        float sample_y = gridptr[1];

                        sample_x = grid_sample_unormalize(w, sample_x, align_corner);
                        sample_y = grid_sample_unormalize(h, sample_y, align_corner);

                        // bilinear interpolate
                        float v;
                        {
                            int x0 = (int)floor(sample_x);
                            int y0 = (int)floor(sample_y);
                            int x1 = x0 + 1;
                            int y1 = y0 + 1;

                            float v00 = get_value_bounded(image, x0, y0, padding_mode, align_corner);
                            float v01 = get_value_bounded(image, x1, y0, padding_mode, align_corner);
                            float v10 = get_value_bounded(image, x0, y1, padding_mode, align_corner);
                            float v11 = get_value_bounded(image, x1, y1, padding_mode, align_corner);

                            float alpha = sample_x - x0;
                            float beta = sample_y - y0;

                            float v0 = v00 * (1 - alpha) + v01 * alpha;
                            float v1 = v10 * (1 - alpha) + v11 * alpha;

                            v = v0 * (1 - beta) + v1 * beta;
                        }

                        outptr[0] = v;
                        outptr += 1;

                        gridptr += 2;
                    }
                }
            }
        }
        else if (sample_type == 2) // nearest
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat image = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q);

                for (int y = 0; y < outh; y++)
                {
                    const float* gridptr = grid.channel(y);

                    for (int x = 0; x < outw; x++)
                    {
                        float sample_x = gridptr[0];
                        float sample_y = gridptr[1];

                        sample_x = grid_sample_unormalize(w, sample_x, align_corner);
                        sample_y = grid_sample_unormalize(h, sample_y, align_corner);

                        int x0 = static_cast<int>(round(sample_x));
                        int y0 = static_cast<int>(round(sample_y));

                        float v = get_value_bounded(image, x0, y0, padding_mode, align_corner);

                        outptr[0] = v;
                        outptr += 1;

                        gridptr += 2;
                    }
                }
            }
        }
        else if (sample_type == 3) // bicubic
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat image = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q);

                for (int y = 0; y < outh; y++)
                {
                    const float* gridptr = grid.channel(y);

                    for (int x = 0; x < outw; x++)
                    {
                        float sample_x = gridptr[0];
                        float sample_y = gridptr[1];

                        sample_x = grid_sample_unormalize(w, sample_x, align_corner);
                        sample_y = grid_sample_unormalize(h, sample_y, align_corner);

                        // bicubic interpolate
                        float v;
                        {
                            int x1 = (int)floorf(sample_x);
                            int y1 = (int)floorf(sample_y);
                            int x0 = x1 - 1;
                            int y0 = y1 - 1;
                            int x2 = x1 + 1;
                            int y2 = y1 + 1;
                            int x3 = x1 + 2;
                            int y3 = y1 + 2;

                            float v00 = get_value_bounded(image, x0, y0, padding_mode, align_corner);
                            float v01 = get_value_bounded(image, x1, y0, padding_mode, align_corner);
                            float v02 = get_value_bounded(image, x2, y0, padding_mode, align_corner);
                            float v03 = get_value_bounded(image, x3, y0, padding_mode, align_corner);
                            float v10 = get_value_bounded(image, x0, y1, padding_mode, align_corner);
                            float v11 = get_value_bounded(image, x1, y1, padding_mode, align_corner);
                            float v12 = get_value_bounded(image, x2, y1, padding_mode, align_corner);
                            float v13 = get_value_bounded(image, x3, y1, padding_mode, align_corner);
                            float v20 = get_value_bounded(image, x0, y2, padding_mode, align_corner);
                            float v21 = get_value_bounded(image, x1, y2, padding_mode, align_corner);
                            float v22 = get_value_bounded(image, x2, y2, padding_mode, align_corner);
                            float v23 = get_value_bounded(image, x3, y2, padding_mode, align_corner);
                            float v30 = get_value_bounded(image, x0, y3, padding_mode, align_corner);
                            float v31 = get_value_bounded(image, x1, y3, padding_mode, align_corner);
                            float v32 = get_value_bounded(image, x2, y3, padding_mode, align_corner);
                            float v33 = get_value_bounded(image, x3, y3, padding_mode, align_corner);

                            float x_coeffs[4];
                            float y_coeffs[4];
                            interpolate_cubic(sample_x - x1, x_coeffs);
                            interpolate_cubic(sample_y - y1, y_coeffs);

                            float v0 = v00 * x_coeffs[0] + v01 * x_coeffs[1] + v02 * x_coeffs[2] + v03 * x_coeffs[3];
                            float v1 = v10 * x_coeffs[0] + v11 * x_coeffs[1] + v12 * x_coeffs[2] + v13 * x_coeffs[3];
                            float v2 = v20 * x_coeffs[0] + v21 * x_coeffs[1] + v22 * x_coeffs[2] + v23 * x_coeffs[3];
                            float v3 = v30 * x_coeffs[0] + v31 * x_coeffs[1] + v32 * x_coeffs[2] + v33 * x_coeffs[3];

                            v = v0 * y_coeffs[0] + v1 * y_coeffs[1] + v2 * y_coeffs[2] + v3 * y_coeffs[3];
                        }

                        outptr[0] = v;
                        outptr += 1;

                        gridptr += 2;
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
        if (top_blob.empty())
            return -100;

        if (sample_type == 1) // bilinear
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat image = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q);

                for (int z = 0; z < outd; z++)
                {
                    const float* gridptr = grid.channel(z);

                    for (int y = 0; y < outh; y++)
                    {
                        for (int x = 0; x < outw; x++)
                        {
                            float sample_x = gridptr[0];
                            float sample_y = gridptr[1];
                            float sample_z = gridptr[2];

                            sample_x = grid_sample_unormalize(w, sample_x, align_corner);
                            sample_y = grid_sample_unormalize(h, sample_y, align_corner);
                            sample_z = grid_sample_unormalize(d, sample_z, align_corner);

                            // bilinear interpolate
                            float v;
                            {
                                int x0 = (int)floor(sample_x);
                                int y0 = (int)floor(sample_y);
                                int z0 = (int)floor(sample_z);
                                int x1 = x0 + 1;
                                int y1 = y0 + 1;
                                int z1 = z0 + 1;

                                float v000 = get_value_bounded(image, x0, y0, z0, padding_mode, align_corner);
                                float v001 = get_value_bounded(image, x1, y0, z0, padding_mode, align_corner);
                                float v010 = get_value_bounded(image, x0, y1, z0, padding_mode, align_corner);
                                float v011 = get_value_bounded(image, x1, y1, z0, padding_mode, align_corner);
                                float v100 = get_value_bounded(image, x0, y0, z1, padding_mode, align_corner);
                                float v101 = get_value_bounded(image, x1, y0, z1, padding_mode, align_corner);
                                float v110 = get_value_bounded(image, x0, y1, z1, padding_mode, align_corner);
                                float v111 = get_value_bounded(image, x1, y1, z1, padding_mode, align_corner);

                                float alpha = sample_x - x0;
                                float beta = sample_y - y0;
                                float gamma = sample_z - z0;

                                float v00 = v000 * (1 - alpha) + v001 * alpha;
                                float v01 = v010 * (1 - alpha) + v011 * alpha;
                                float v10 = v100 * (1 - alpha) + v101 * alpha;
                                float v11 = v110 * (1 - alpha) + v111 * alpha;

                                float v0 = v00 * (1 - beta) + v01 * beta;
                                float v1 = v10 * (1 - beta) + v11 * beta;

                                v = v0 * (1 - gamma) + v1 * gamma;
                            }

                            outptr[0] = v;
                            outptr += 1;

                            gridptr += 3;
                        }
                    }
                }
            }
        }
        else if (sample_type == 2) // nearest
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat image = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q);

                for (int z = 0; z < outd; z++)
                {
                    const float* gridptr = grid.channel(z);

                    for (int y = 0; y < outh; y++)
                    {
                        for (int x = 0; x < outw; x++)
                        {
                            float sample_x = gridptr[0];
                            float sample_y = gridptr[1];
                            float sample_z = gridptr[2];

                            sample_x = grid_sample_unormalize(w, sample_x, align_corner);
                            sample_y = grid_sample_unormalize(h, sample_y, align_corner);
                            sample_z = grid_sample_unormalize(d, sample_z, align_corner);

                            int x0 = static_cast<int>(round(sample_x));
                            int y0 = static_cast<int>(round(sample_y));
                            int z0 = static_cast<int>(round(sample_z));

                            float v = get_value_bounded(image, x0, y0, z0, padding_mode, align_corner);

                            outptr[0] = v;
                            outptr += 1;

                            gridptr += 3;
                        }
                    }
                }
            }
        }
        else if (sample_type == 3)
        {
            NCNN_LOGE("unsupported bicubic when dims == 4");
            return -1;
        }
    }

    return 0;
}

} // namespace ncnn
