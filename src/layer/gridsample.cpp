// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
<<<<<<< HEAD
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
=======
// coord compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to coord writing, software distributed
>>>>>>> master
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "gridsample.h"
<<<<<<< HEAD
#include <cmath>
#include <tuple>

namespace ncnn {
enum InterpolationMode
{
    Bilinear = 1,
    Nearest = 2,
    Bicubic = 3
};

enum PaddingMode
{
    Zeros = 1,
    Border = 2,
    Reflection = 3
};

static inline float clip_coordinates(float in, int64_t clip_limit)
{
    return std::min(static_cast<float>(clip_limit - 1), std::max(in, static_cast<float>(0)));
}

static inline float reflect_coordinates(float in, int64_t twice_low,
                                        int64_t twice_high)
{
    if (twice_low == twice_high)
    {
        return static_cast<float>(0);
    }
    float min = static_cast<float>(twice_low) / 2;
    float span = static_cast<float>(twice_high - twice_low) / 2;
    in = std::fabs(in - min);
    // `fmod` returns same sign as `in`, which is positive after the `fabs` above.
    float extra = std::fmod(in, span);
    int flips = static_cast<int>(std::floor(in / span));
    if (flips % 2 == 0)
    {
        return extra + min;
    }
    else
    {
        return span - extra + min;
    }
}

static inline float compute_coordinates(float coord, int64_t size,
                                        PaddingMode padding_mode,
                                        bool align_corners)
{
    if (padding_mode == PaddingMode::Border)
    {
        // clip coordinates to image borders
        coord = clip_coordinates(coord, size);
    }
    else if (padding_mode == PaddingMode::Reflection)
    {
        // reflect coordinates by image borders
        if (align_corners)
        {
            coord = reflect_coordinates(coord, 0, 2 * (size - 1));
        }
        else
        {
            coord = reflect_coordinates(coord, -1, 2 * size - 1);
        }
        // clip coordinates to image borders
        coord = clip_coordinates(coord, size);
    }
    return coord;
}

static inline float grid_sampler_unnormalize(float coord, int64_t size,
        bool align_corners)
{
    if (align_corners)
    {
        // unnormalize coord from [-1, 1] to [0, size - 1]
        return ((coord + 1) / 2) * (size - 1);
    }
    else
    {
        // unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
        return ((coord + 1) * size - 1) / 2;
    }
}

static inline float grid_sampler_compute_source_index(
    float coord,
    int64_t size,
    PaddingMode padding_mode,
    bool align_corners)
{
    coord = grid_sampler_unnormalize(coord, size, align_corners);
    coord = compute_coordinates(coord, size, padding_mode, align_corners);
    return coord;
}

template<InterpolationMode, PaddingMode, bool align_corners>
struct ApplyGridSample;

template<PaddingMode padding, bool align_corners>
struct ApplyGridSample<InterpolationMode::Bilinear, padding, align_corners>
{
    const bool must_in_bound = padding != PaddingMode::Zeros;
    inline std::tuple<float, float, float, float> compute_interp_params_d3(float x, float y) const
    {
        auto x_w = std::floor(x);
        auto y_n = std::floor(y);

        auto w = x - x_w;
        auto e = 1.0f - w;
        auto n = y - y_n;
        auto s = 1.0f - n;

        auto nw = s * e;
        auto ne = s * w;
        auto sw = n * e;
        auto se = n * w;

        return std::make_tuple(nw, ne, sw, se);
    }

    inline int forward(const Mat& input, const Mat& grid, Mat& output, const Option& opt)
    {
        const int dims = input.dims;
        const int w = input.w;
        const int h = input.h;
        const int outW = grid.h;
        const int outH = grid.c;
        const int channels = input.c;

        if (dims == 3)
        {
            output.create(outW, outH, input.c);
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* output_ptr = static_cast<float*>(output.channel(q).data);

                const Mat image = input.channel(q);

                //const float* gxy_ptr = static_cast<float*>(grid.data);

                for (int y = 0; y < outH; y++)
                {
                    for (int x = 0; x < outW; x++)
                    {
                        const float* gxy_ptr = grid.channel(y).row(x);
                        auto gx = grid_sampler_compute_source_index(gxy_ptr[0], w, padding, align_corners);
                        auto gy = grid_sampler_compute_source_index(gxy_ptr[1], h, padding, align_corners);

                        auto interp_params = compute_interp_params_d3(gx, gy);

                        auto nw = std::get<0>(interp_params);
                        auto ne = std::get<1>(interp_params);
                        auto sw = std::get<2>(interp_params);
                        auto se = std::get<3>(interp_params);

                        auto i_x = static_cast<int>(std::floor(gx));
                        auto i_y = static_cast<int>(std::floor(gy));

                        float v = 0.0f;
                        if (must_in_bound)
                        {
                            //out of range, val is 0 https://github.com/pytorch/pytorch/blob/435e78e5237d9fb3e433fff6ce028569db937264/aten/src/ATen/native/cpu/GridSamplerKernel.cpp#L520
                            auto nw_val = image.row(i_y)[i_x];
                            auto ne_val = i_x + 1 < w ? image.row(i_y)[i_x + 1] : 0;
                            auto sw_val = i_y + 1 < h ? image.row(i_y + 1)[i_x] : 0;
                            auto se_val = ((i_x + 1 < w) & (i_y + 1 < h)) ? image.row(i_y + 1)[i_x + 1] : 0;

                            v = nw_val * nw + ne_val * ne + sw_val * sw + se_val * se;
                        }
                        else //PaddingMode::Zeors
                        {
                            auto x0 = i_x;
                            auto x1 = i_x + 1;
                            auto y0 = i_y;
                            auto y1 = i_y + 1;

                            auto x0_in_range = (x0 > -1) & (x0 < w);
                            auto x1_in_range = (x1 > -1) & (x1 < w);
                            auto y0_in_range = (y0 > -1) & (y0 < h);
                            auto y1_in_range = (y1 > -1) & (y1 < h);

                            auto v00_in_range = x0_in_range & y0_in_range;
                            auto v01_in_range = x0_in_range & y1_in_range;
                            auto v10_in_range = x1_in_range & y0_in_range;
                            auto v11_in_range = x1_in_range & y1_in_range;

                            auto nw_val = v00_in_range ? image.row(y0)[x0] : 0;
                            auto ne_val = v10_in_range ? image.row(y0)[x1] : 0;
                            auto sw_val = v01_in_range ? image.row(y1)[x0] : 0;
                            auto se_val = v11_in_range ? image.row(y1)[x1] : 0;

                            v = nw_val * nw + ne_val * ne + sw_val * sw + se_val * se;
                        }

                        *output_ptr = v;

                        output_ptr++;
                    }
                }
            }
        }
        else if (dims == 4)
        {
        }
        else
        {
            return -100;
        }
    }
};

template<PaddingMode padding, bool align_corners>
struct ApplyGridSample<InterpolationMode::Nearest, padding, align_corners>
{
    const bool must_in_bound = padding != PaddingMode::Zeros;
    inline void forward(const Mat& input, const Mat& grid, Mat& output, const Option& opt)
    {
        const int dims = input.dims;
        const int w = input.w;
        const int h = input.h;
        const int outW = grid.h;
        const int outH = grid.c;
        const int channels = input.c;

        if (dims == 3)
        {
            output.create(outW, outH, input.c);
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* output_ptr = static_cast<float*>(output.channel(q).data);

                const Mat image = input.channel(q);

                //const float* gxy_ptr = static_cast<float*>(grid.data);

                for (int y = 0; y < outH; y++)
                {
                    for (int x = 0; x < outW; x++)
                    {
                        const float* gxy_ptr = grid.channel(y).row(x);
                        auto gx = grid_sampler_compute_source_index(gxy_ptr[0], w, padding, align_corners);
                        auto gy = grid_sampler_compute_source_index(gxy_ptr[1], h, padding, align_corners);

                        auto x_nearest = static_cast<int>(std::round(gx));
                        auto y_nearest = static_cast<int>(std::round(gy));

                        float v = image.row(y_nearest)[x_nearest];
                        if (!must_in_bound)
                        {
                            v = ((x_nearest < w) & (x_nearest > -1) & (y_nearest < h) & (y_nearest > -1)) ? v : 0;
                        }

                        *output_ptr = v;

                        output_ptr++;
                    }
                }
            }
        }
        else if (dims == 4)
        {
        }
        else
        {
        }
    }
};

template<PaddingMode padding, bool align_corners>
struct ApplyGridSample<InterpolationMode::Bicubic, padding, align_corners>
{
    inline void forward(const Mat& input, const Mat& grid, Mat& output, const Option& opt)
    {
    }
};
=======

#include <math.h>

namespace ncnn {
>>>>>>> master

GridSample::GridSample()
{
    one_blob_only = false;
    support_inplace = false;
}

int GridSample::load_param(const ParamDict& pd)
{
<<<<<<< HEAD
    mode = pd.get(0, 0);
    padding_mode = pd.get(1, 0);
    align_corners = pd.get(6, 0);
=======
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
>>>>>>> master

    return 0;
}

<<<<<<< HEAD
int GridSample::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
#define HANDLE_PADDING(interp, padding, align_corners)                     \
    case padding:                                                          \
    {                                                                      \
        ApplyGridSample<interp, padding, align_corners> func;              \
        func.forward(bottom_blobs[0], bottom_blobs[1], top_blobs[0], opt); \
        break;                                                             \
    }

#define HANDLE_INTERP(interp, align_corners)                               \
    case interp:                                                           \
    {                                                                      \
        switch (static_cast<InterpolationMode>(padding_mode))              \
        {                                                                  \
            HANDLE_PADDING(interp, PaddingMode::Zeros, align_corners)      \
            HANDLE_PADDING(interp, PaddingMode::Border, align_corners)     \
            HANDLE_PADDING(interp, PaddingMode::Reflection, align_corners) \
        }                                                                  \
        break;                                                             \
    }

    if (align_corners == true)
    {
        switch (static_cast<InterpolationMode>(mode))
        {
            HANDLE_INTERP(InterpolationMode::Bilinear, true);
            HANDLE_INTERP(InterpolationMode::Nearest, true);
            HANDLE_INTERP(InterpolationMode::Bicubic, true);
        }
    }
    else
    {
        switch (static_cast<InterpolationMode>(mode))
        {
            HANDLE_INTERP(InterpolationMode::Bilinear, false);
            HANDLE_INTERP(InterpolationMode::Nearest, false);
            HANDLE_INTERP(InterpolationMode::Bicubic, false);
        }
    }
#undef HANDLE_PADDING
#undef HANDLE_INTERP
=======
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
            sx = static_cast<int>(reflect_coord(sx + 0.5, w) - 0.5);
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
                            int x1 = floor(sample_x);
                            int y1 = floor(sample_y);
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
>>>>>>> master

    return 0;
}

} // namespace ncnn
