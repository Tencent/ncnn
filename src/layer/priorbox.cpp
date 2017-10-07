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
#include <math.h>

namespace ncnn {

DEFINE_LAYER_CREATOR(PriorBox)

PriorBox::PriorBox()
{
    one_blob_only = false;
    support_inplace = false;
}

#if NCNN_STDIO
#if NCNN_STRING
int PriorBox::load_param(FILE* paramfp)
{
    int nscan = fscanf(paramfp, "%d %d %d %f %f %f %f %d %d %d %d %f %f %f",
                       &num_min_size, &num_max_size, &num_aspect_ratio,
                       &variances[0], &variances[1], &variances[2], &variances[3],
                       &flip, &clip, &image_width, &image_height,
                       &step_width, &step_height, &offset);
    if (nscan != 14)
    {
        fprintf(stderr, "PriorBox load_param failed %d\n", nscan);
        return -1;
    }

    min_sizes.create(num_min_size);
    if (min_sizes.empty())
        return -100;
    float* min_sizes_ptr = min_sizes;
    for (int i=0; i<num_min_size; i++)
    {
        int nscan = fscanf(paramfp, "%f", &min_sizes_ptr[i]);
        if (nscan != 1)
        {
            fprintf(stderr, "PriorBox load_param failed %d\n", nscan);
            return -1;
        }
    }

    if (num_max_size > 0)
    {
        max_sizes.create(num_max_size);
        if (max_sizes.empty())
            return -100;
        float* max_sizes_ptr = max_sizes;
        for (int i=0; i<num_max_size; i++)
        {
            int nscan = fscanf(paramfp, "%f", &max_sizes_ptr[i]);
            if (nscan != 1)
            {
                fprintf(stderr, "PriorBox load_param failed %d\n", nscan);
                return -1;
            }
        }
    }

    if (num_aspect_ratio > 0)
    {
        aspect_ratios.create(num_aspect_ratio);
        if (aspect_ratios.empty())
            return -100;
        float* aspect_ratios_ptr = aspect_ratios;
        for (int i=0; i<num_aspect_ratio; i++)
        {
            int nscan = fscanf(paramfp, "%f", &aspect_ratios_ptr[i]);
            if (nscan != 1)
            {
                fprintf(stderr, "PriorBox load_param failed %d\n", nscan);
                return -1;
            }
        }
    }

    return 0;
}
#endif // NCNN_STRING
int PriorBox::load_param_bin(FILE* paramfp)
{
    fread(&num_min_size, sizeof(int), 1, paramfp);

    fread(&num_max_size, sizeof(int), 1, paramfp);

    fread(&num_aspect_ratio, sizeof(int), 1, paramfp);

    fread(&variances[0], sizeof(float), 1, paramfp);

    fread(&variances[1], sizeof(float), 1, paramfp);

    fread(&variances[2], sizeof(float), 1, paramfp);

    fread(&variances[3], sizeof(float), 1, paramfp);

    fread(&flip, sizeof(int), 1, paramfp);

    fread(&clip, sizeof(int), 1, paramfp);

    fread(&image_width, sizeof(int), 1, paramfp);

    fread(&image_height, sizeof(int), 1, paramfp);

    fread(&step_width, sizeof(float), 1, paramfp);

    fread(&step_height, sizeof(float), 1, paramfp);

    fread(&offset, sizeof(float), 1, paramfp);

    min_sizes.create(num_min_size);
    if (min_sizes.empty())
        return -100;
    float* min_sizes_ptr = min_sizes;
    fread(min_sizes_ptr, sizeof(float), num_min_size, paramfp);

    if (num_max_size > 0)
    {
        max_sizes.create(num_max_size);
        if (max_sizes.empty())
            return -100;
        float* max_sizes_ptr = max_sizes;
        fread(max_sizes_ptr, sizeof(float), num_max_size, paramfp);
    }

    if (num_aspect_ratio > 0)
    {
        aspect_ratios.create(num_aspect_ratio);
        if (aspect_ratios.empty())
            return -100;
        float* aspect_ratios_ptr = aspect_ratios;
        fread(aspect_ratios_ptr, sizeof(float), num_aspect_ratio, paramfp);
    }

    return 0;
}
#endif // NCNN_STDIO

int PriorBox::load_param(const unsigned char*& mem)
{
    num_min_size = *(int*)(mem);
    mem += 4;

    num_max_size = *(int*)(mem);
    mem += 4;

    num_aspect_ratio = *(int*)(mem);
    mem += 4;

    variances[0] = *(float*)(mem);
    mem += 4;

    variances[1] = *(float*)(mem);
    mem += 4;

    variances[2] = *(float*)(mem);
    mem += 4;

    variances[3] = *(float*)(mem);
    mem += 4;

    flip = *(int*)(mem);
    mem += 4;

    clip = *(int*)(mem);
    mem += 4;

    image_width = *(int*)(mem);
    mem += 4;

    image_height = *(int*)(mem);
    mem += 4;

    step_width = *(float*)(mem);
    mem += 4;

    step_height = *(float*)(mem);
    mem += 4;

    offset = *(float*)(mem);
    mem += 4;

    min_sizes = Mat(num_min_size, (float*)mem);
    mem += num_min_size * sizeof(float);

    if (num_max_size > 0)
    {
        max_sizes = Mat(num_max_size, (float*)mem);
        mem += num_max_size * sizeof(float);
    }

    if (num_aspect_ratio > 0)
    {
        aspect_ratios = Mat(num_aspect_ratio, (float*)mem);
        mem += num_aspect_ratio * sizeof(float);
    }

    return 0;
}

int PriorBox::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const
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

    int num_prior = num_min_size * num_aspect_ratio + num_min_size + num_max_size;
    if (flip)
        num_prior += num_min_size * num_aspect_ratio;

    Mat& top_blob = top_blobs[0];
    top_blob.create(4 * w * h * num_prior, 2);

    #pragma omp parallel for
    for (int i = 0; i < h; i++)
    {
        float* box = top_blob.data + i * w * num_prior * 4;

        float center_x = offset * step_w;
        float center_y = offset * step_h + i * step_h;

        for (int j = 0; j < w; j++)
        {
            float box_w;
            float box_h;

            for (int k = 0; k < num_min_size; k++)
            {
                float min_size = min_sizes.data[k];

                // min size box
                box_w = box_h = min_size;

                box[0] = (center_x - box_w * 0.5f) / image_w;
                box[1] = (center_y - box_h * 0.5f) / image_h;
                box[2] = (center_x + box_w * 0.5f) / image_w;
                box[3] = (center_y + box_h * 0.5f) / image_h;

                box += 4;

                if (num_max_size > 0)
                {
                    float max_size = max_sizes.data[k];

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
                        box[0] = (center_x - box_h * 0.5f) / image_h;
                        box[1] = (center_y - box_w * 0.5f) / image_w;
                        box[2] = (center_x + box_h * 0.5f) / image_h;
                        box[3] = (center_y + box_w * 0.5f) / image_w;

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
