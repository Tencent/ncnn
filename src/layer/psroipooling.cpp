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

#include "psroipooling.h"
#include <math.h>
#include <algorithm>

namespace ncnn {

DEFINE_LAYER_CREATOR(PSROIPooling)

PSROIPooling::PSROIPooling()
{
    one_blob_only = false;
    support_inplace = false;
}

int PSROIPooling::load_param(const ParamDict& pd)
{
    pooled_width = pd.get(0, 7);
    pooled_height = pd.get(1, 7);
    spatial_scale = pd.get(2, 0.0625f);
    output_dim = pd.get(3, 0);

    return 0;
}

int PSROIPooling::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    size_t elemsize = bottom_blob.elemsize;
    int channels = bottom_blob.c;

    const Mat& roi_blob = bottom_blobs[1];

    if (channels != output_dim * pooled_width * pooled_height)
    {
        // input channel number does not match layer parameters
        return -1;
    }

    Mat& top_blob = top_blobs[0];
    top_blob.create(pooled_width, pooled_height, output_dim, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // For each ROI R = [x y w h]: avg pool over R
    const float* roi_ptr = roi_blob;

    float roi_x1 = round(roi_ptr[0]) * spatial_scale;
    float roi_y1 = round(roi_ptr[1]) * spatial_scale;
    float roi_x2 = round(roi_ptr[2] + 1.f) * spatial_scale;
    float roi_y2 = round(roi_ptr[3] + 1.f) * spatial_scale;

    float roi_w = std::max(roi_x2 - roi_x1, 0.1f);
    float roi_h = std::max(roi_y2 - roi_y1, 0.1f);

    float bin_size_w = roi_w / (float)pooled_width;
    float bin_size_h = roi_h / (float)pooled_height;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q=0; q<output_dim; q++)
    {
        float* outptr = top_blob.channel(q);

        for (int ph = 0; ph < pooled_height; ph++)
        {
            for (int pw = 0; pw < pooled_width; pw++)
            {
                const float* ptr = bottom_blob.channel((q * pooled_height + ph) * pooled_width + pw);

                int hstart = floor(roi_y1 + (float)(ph) * bin_size_h);
                int wstart = floor(roi_x1 + (float)(pw) * bin_size_w);
                int hend = ceil(roi_y1 + (float)(ph + 1) * bin_size_h);
                int wend = ceil(roi_x1 + (float)(pw + 1) * bin_size_w);

                hstart = std::min(std::max(hstart, 0), h);
                wstart = std::min(std::max(wstart, 0), w);
                hend = std::min(std::max(hend, 0), h);
                wend = std::min(std::max(wend, 0), w);

                bool is_empty = (hend <= hstart) || (wend <= wstart);
                int area = (hend - hstart) * (wend - wstart);

                float sum = 0.f;
                for (int y = hstart; y < hend; y++)
                {
                    for (int x = wstart; x < wend; x++)
                    {
                        int index = y * w + x;
                        sum += ptr[index];
                    }
                }

                outptr[pw] = is_empty ? 0.f : (sum / (float)area);
            }

            outptr += pooled_width;
        }
    }

    return 0;
}

} // namespace ncnn
