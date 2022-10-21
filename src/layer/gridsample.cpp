// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "gridsample.h"
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
    
    template<typename elem_type>
    static inline elem_type clip_coordinates(elem_type in, int64_t clip_limit) {
        return std::min(static_cast<elem_type>(clip_limit - static_cast<elem_type>(1)), std::max(in, static_cast<elem_type>(0)));
    }

    template<typename elem_type>
    static inline elem_type reflect_coordinates(elem_type in, int64_t twice_low,
        int64_t twice_high) {
        if (twice_low == twice_high) {
            return static_cast<elem_type>(0);
        }
        elem_type min = static_cast<elem_type>(twice_low) / 2;
        elem_type span = static_cast<elem_type>(twice_high - twice_low) / 2;
        in = std::fabs(in - min);
        // `fmod` returns same sign as `in`, which is positive after the `fabs` above.
        elem_type extra = std::fmod(in, span);
        int flips = static_cast<int>(std::floor(in / span));
        if (flips % 2 == 0) {
            return extra + min;
        }
        else {
            return span - extra + min;
        }
    }

    template<typename elem_type>
    static inline elem_type compute_coordinates(elem_type coord, int64_t size,
        PaddingMode padding_mode,
        bool align_corners) {
        if (padding_mode == PaddingMode::Border) {
            // clip coordinates to image borders
            coord = clip_coordinates(coord, size);
        }
        else if (padding_mode == PaddingMode::Reflection) {
            // reflect coordinates by image borders
            if (align_corners) {
                coord = reflect_coordinates(coord, 0, 2 * (size - 1));
            }
            else {
                coord = reflect_coordinates(coord, -1, 2 * size - 1);
            }
            // clip coordinates to image borders
            coord = clip_coordinates(coord, size);
        }
        return coord;
    }

    template <typename elem_type>
    static inline elem_type grid_sampler_unnormalize(elem_type coord, int64_t size,
        bool align_corners) {
        if (align_corners) {
            // unnormalize coord from [-1, 1] to [0, size - 1]
            return ((coord + 1) / 2) * (size - 1);
        }
        else {
            // unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
            return ((coord + 1) * size - 1) / 2;
        }
    }

    template <typename elem_type>
    static inline elem_type grid_sampler_compute_source_index(
        elem_type coord,
        int64_t size,
        PaddingMode padding_mode,
        bool align_corners) {
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
            int dims = input.dims;
            int w = input.w;
            int h = input.h;
            int channels = input.c;

            if (dims == 3)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    float* output_ptr = output.data;

                    const Mat image = input.channel(q);

                    const float* gx_ptr = grid.channel(0);
                    const float* gy_ptr = grid.channel(1);

                    for (int y = 0; y < h; y++) 
                    {
                        for (int x = 0; x < w; x++)
                        {
                            auto gx = grid_sampler_compute_source_index(*gx_ptr, w, padding, align_corners);
                            auto gy = grid_sampler_compute_source_index(*gy_ptr, h, padding, align_corners);

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
                                auto nw_val = image.row(i_x)[i_y];
                                auto ne_val = i_y + 1 < h ? image.row(i_x)[i_y + 1] : 0;
                                auto sw_val = i_x + 1 < w ? image.row(i_x + 1)[i_y] : 0;
                                auto se_val = i_x + 1 < w && i_y + 1 < h ? image.row(i_x + 1)[i_y + 1] : 0;

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
                                auto y0_in_range = (y0 > -1) & (y0 < w);
                                auto y1_in_range = (y1 > -1) & (y1 < w);

                                auto v00_in_range = x0_in_range & y0_in_range;
                                auto v01_in_range = x0_in_range & y1_in_range;
                                auto v10_in_range = x1_in_range & y0_in_range;
                                auto v11_in_range = x1_in_range & y1_in_range;

                                auto nw_val = v00_in_range ? image.row(x0)[y0] : 0;
                                auto ne_val = v01_in_range ? image.row(x0)[y1] : 0;
                                auto sw_val = v10_in_range ? image.row(x1)[y0] : 0;
                                auto se_val = v11_in_range ? image.row(x1)[y1] : 0;

                                v = nw_val * nw + ne_val * ne + sw_val * sw + se_val * se;
                            }

                            *output = v;

                            output++;
                            fxptr++;
                            fyptr++;
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
        inline void forward(const Mat& input, const Mat& grid, Mat& output)
        {

        }
    };

    template<PaddingMode padding, bool align_corners>
    struct ApplyGridSample<InterpolationMode::Bicubic, padding, align_corners>
    {
        inline void forward(const Mat& input, const Mat& grid, Mat& output)
        {

        }
    };

    GridSample::GridSample()
    {
        one_blob_only = false;
        support_inplace = false;
    }

    int GridSample::load_param(const ParamDict& pd)
    {
        mode = pd.get(0, 0);
        padding_mode = pd.get(1, 0);
        align_corners = pd.get(6, 0);

        return 0;
    }

    int GridSample::forward(const std::vector<Mat>& bottom_blobs, Mat& top_blobs, const Option& opt) const
    {
    #define HANDLE_PADDING(interp, padding, align_corners)                      \
        case padding:{                                                          \
            printf("mode: %d, padding_mode: %d, align: %d", interp, padding, align_corners); \
            break;                                                              \
        }

    #define HANDLE_INTERP(interp, align_corners)                                \
        case interp:{                                                           \
            switch(static_cast<InterpolationMode>(padding_mode)) {              \
                HANDLE_PADDING(interp, PaddingMode::Zeros, align_corners)       \
                HANDLE_PADDING(interp, PaddingMode::Border, align_corners)      \
                HANDLE_PADDING(interp, PaddingMode::Reflection, align_corners)  \
            }                                                                   \
            break;                                                              \
        }



        switch (static_cast<InterpolationMode>(mode)) {
            HANDLE_INTERP(InterpolationMode::Bilinear, align_corners);
            HANDLE_INTERP(InterpolationMode::Nearest, align_corners);
            HANDLE_INTERP(InterpolationMode::Bicubic, align_corners);
        }
#undef HANDLE_PADDING
#undef HANDLE_INTERP
    }

} // namespace ncnn
