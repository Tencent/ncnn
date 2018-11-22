// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "priorbox.h"
#include <algorithm>
#include <math.h>

namespace ncnn {

DEFINE_LAYER_CREATOR(PriorBox)

PriorBox::PriorBox()
{
    one_blob_only = false;
    support_inplace = false;
}

int PriorBox::load_param(const ParamDict& pd)
{
    min_sizes = pd.get(0, Mat());
    max_sizes = pd.get(1, Mat());
    aspect_ratios = pd.get(2, Mat());
    variances[0] = pd.get(3, 0.f);
    variances[1] = pd.get(4, 0.f);
    variances[2] = pd.get(5, 0.f);
    variances[3] = pd.get(6, 0.f);
    flip = pd.get(7, 1);
    clip = pd.get(8, 0);
    image_width = pd.get(9, 0);
    image_height = pd.get(10, 0);
    step_width = pd.get(11, -233.f);
    step_height = pd.get(12, -233.f);
    offset = pd.get(13, 0.f);

    return 0;
}

int PriorBox::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    int w = bottom_blobs[0].w;
    int h = bottom_blobs[0].h;

    int image_w = image_width;
    int image_h = image_height;
    if (image_w == -233)
        image_w = bottom_blobs[1].w;
    if (image_h == -233)
        image_h = bottom_blobs[1].h;

    float step_w = step_width;
    float step_h = step_height;
    if (step_w == -233)
        step_w = (float)image_w / w;
    if (step_h == -233)
        step_h = (float)image_h / h;

    int num_min_size = min_sizes.w;
    int num_max_size = max_sizes.w;
    int num_aspect_ratio = aspect_ratios.w;

    Mat min_sizes_copy = min_sizes.clone();
    Mat max_sizes_copy = max_sizes.clone();

    for (int k = 0; k < num_min_size; k++)
    {
        if (min_sizes_copy[k] < 0.f)
            min_sizes_copy[k] = -min_sizes_copy[k] * image_w;
    }
    for (int k = 0; k < num_max_size; k++)
    {
        if (max_sizes_copy[k] < 0.f)
            max_sizes_copy[k] = -max_sizes_copy[k] * image_w;
    }

    int num_prior = num_min_size * num_aspect_ratio + num_min_size + num_max_size;
    if (flip)
        num_prior += num_min_size * num_aspect_ratio;

    Mat& top_blob = top_blobs[0];
    top_blob.create(4 * w * h * num_prior, 2, 4u, opt.blob_allocator);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < h; i++)
    {
        float* box = (float*)top_blob + i * w * num_prior * 4;

        float center_x = offset * step_w;
        float center_y = offset * step_h + i * step_h;

        for (int j = 0; j < w; j++)
        {
            float box_w;
            float box_h;

            for (int k = 0; k < num_min_size; k++)
            {
                float min_size = min_sizes_copy[k];

                // min size box
                box_w = box_h = min_size;

                box[0] = (center_x - box_w * 0.5f) / image_w;
                box[1] = (center_y - box_h * 0.5f) / image_h;
                box[2] = (center_x + box_w * 0.5f) / image_w;
                box[3] = (center_y + box_h * 0.5f) / image_h;

                box += 4;

                if (num_max_size > 0)
                {
                    float max_size = max_sizes_copy[k];

                    // max size box
                    box_w = box_h = sqrt(min_size * max_size);

                    box[0] = (center_x - box_w * 0.5f) / image_w;
                    box[1] = (center_y - box_h * 0.5f) / image_h;
                    box[2] = (center_x + box_w * 0.5f) / image_w;
                    box[3] = (center_y + box_h * 0.5f) / image_h;

                    box += 4;
                }

                // all aspect_ratios
                for (int p = 0; p < num_aspect_ratio; p++)
                {
                    float ar = aspect_ratios[p];

                    box_w = min_size * sqrt(ar);
                    box_h = min_size / sqrt(ar);

                    box[0] = (center_x - box_w * 0.5f) / image_w;
                    box[1] = (center_y - box_h * 0.5f) / image_h;
                    box[2] = (center_x + box_w * 0.5f) / image_w;
                    box[3] = (center_y + box_h * 0.5f) / image_h;

                    box += 4;

                    if (flip)
                    {
                        box[0] = (center_x - box_h * 0.5f) / image_w;
                        box[1] = (center_y - box_w * 0.5f) / image_h;
                        box[2] = (center_x + box_h * 0.5f) / image_w;
                        box[3] = (center_y + box_w * 0.5f) / image_h;

                        box += 4;
                    }
                }
            }

            center_x += step_w;
        }

        center_y += step_h;
    }

    if (clip)
    {
        float* box = top_blob;
        for (int i = 0; i < top_blob.w; i++)
        {
            box[i] = std::min(std::max(box[i], 0.f), 1.f);
        }
    }

    // set variance
    float* var = top_blob.row(1);
    for (int i = 0; i < top_blob.w / 4; i++)
    {
        var[0] = variances[0];
        var[1] = variances[1];
        var[2] = variances[2];
        var[3] = variances[3];

        var += 4;
    }

    return 0;
}

} // namespace ncnn
