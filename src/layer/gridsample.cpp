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

static inline int64_t clip_coordinates(int64_t in, int64_t clip_limit)
{
    return std::min(static_cast<int64_t>(clip_limit - 1), std::max(in, static_cast<int64_t>(0)));
}

static inline int64_t reflect_coordinates(int64_t in, int64_t twice_low,
        int64_t twice_high)
{
    if (twice_low == twice_high)
    {
        return static_cast<int64_t>(0);
    }
    int64_t min = static_cast<int64_t>(twice_low) / 2;
    int64_t span = static_cast<int64_t>(twice_high - twice_low) / 2;
    in = std::fabs(in - min);
    // `fmod` returns same sign as `in`, which is positive after the `fabs` above.
    int64_t extra = std::fmod(in, span);
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

static inline int64_t compute_coordinates(int64_t coord, int64_t size,
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

template<InterpolationMode, PaddingMode, bool align_corners>
struct ApplyGridSample;

template<PaddingMode padding, bool align_corners>
struct ApplyGridSample<InterpolationMode::Bilinear, padding, align_corners>
{
    inline void forward(const ncnn::Mat& input, const ncnn::Mat& grid, ncnn::Mat& output)
    {
    }
};

template<PaddingMode padding, bool align_corners>
struct ApplyGridSample<InterpolationMode::Nearest, padding, align_corners>
{
    inline void forward(const ncnn::Mat& input, const ncnn::Mat& grid, ncnn::Mat& output)
    {
    }
};

template<PaddingMode padding, bool align_corners>
struct ApplyGridSample<InterpolationMode::Bicubic, padding, align_corners>
{
    inline void forward(const ncnn::Mat& input, const ncnn::Mat& grid, ncnn::Mat& output)
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
#define HANDLE_PADDING(interp, padding, align_corners)                                   \
    case padding:                                                                        \
    {                                                                                    \
        printf("mode: %d, padding_mode: %d, align: %d", interp, padding, align_corners); \
        break;                                                                           \
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

    switch (static_cast<InterpolationMode>(mode))
    {
        HANDLE_INTERP(InterpolationMode::Bilinear, align_corners);
        HANDLE_INTERP(InterpolationMode::Nearest, align_corners);
        HANDLE_INTERP(InterpolationMode::Bicubic, align_corners);
    }
#undef HANDLE_PADDING
#undef HANDLE_INTERP
}

} // namespace ncnn
