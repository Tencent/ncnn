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
    support_vulkan = true;

#if NCNN_VULKAN
    pipeline_priorbox = 0;
    pipeline_priorbox_mxnet = 0;
#endif // NCNN_VULKAN
}

int PriorBox::load_param(const ParamDict& pd)
{
    min_sizes = pd.get(0, Mat());
    max_sizes = pd.get(1, Mat());
    aspect_ratios = pd.get(2, Mat());
    variances[0] = pd.get(3, 0.1f);
    variances[1] = pd.get(4, 0.1f);
    variances[2] = pd.get(5, 0.2f);
    variances[3] = pd.get(6, 0.2f);
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

    if (bottom_blobs.size() == 1 && image_width == -233 && image_height == -233 && max_sizes.empty())
    {
        // mxnet style _contrib_MultiBoxPrior
        float step_w = step_width;
        float step_h = step_height;
        if (step_w == -233)
            step_w = 1.f / (float)w;
        if (step_h == -233)
            step_h = 1.f / (float)h;

        int num_sizes = min_sizes.w;
        int num_ratios = aspect_ratios.w;

        int num_prior = num_sizes - 1 + num_ratios;

        Mat& top_blob = top_blobs[0];
        top_blob.create(4 * w * h * num_prior, 4u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* box = (float*)top_blob + i * w * num_prior * 4;

            float center_x = offset * step_w;
            float center_y = offset * step_h + i * step_h;

            for (int j = 0; j < w; j++)
            {
                // ratio = 1, various sizes
                for (int k = 0; k < num_sizes; k++)
                {
                    float size = min_sizes[k];
                    float cw = size * h / w / 2;
                    float ch = size / 2;

                    box[0] = center_x - cw;
                    box[1] = center_y - ch;
                    box[2] = center_x + cw;
                    box[3] = center_y + ch;
                    box += 4;
                }

                // various ratios, size = min_size = size[0]
                float size = min_sizes[0];
                for (int p = 1; p < num_ratios; p++)
                {
                    float ratio = sqrt(aspect_ratios[p]);
                    float cw = size * h / w * ratio / 2;
                    float ch = size / ratio / 2;

                    box[0] = center_x - cw;
                    box[1] = center_y - ch;
                    box[2] = center_x + cw;
                    box[3] = center_y + ch;
                    box += 4;
                }

                center_x += step_w;
            }
        }

        if (clip)
        {
            float* box = top_blob;
            for (int i = 0; i < top_blob.w; i++)
            {
                box[i] = std::min(std::max(box[i], 0.f), 1.f);
            }
        }

        return 0;
    }

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

    int num_prior = num_min_size * num_aspect_ratio + num_min_size + num_max_size;
    if (flip)
        num_prior += num_min_size * num_aspect_ratio;

    Mat& top_blob = top_blobs[0];
    top_blob.create(4 * w * h * num_prior, 2, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

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
                float min_size = min_sizes[k];

                // min size box
                box_w = box_h = min_size;

                box[0] = (center_x - box_w * 0.5f) / image_w;
                box[1] = (center_y - box_h * 0.5f) / image_h;
                box[2] = (center_x + box_w * 0.5f) / image_w;
                box[3] = (center_y + box_h * 0.5f) / image_h;

                box += 4;

                if (num_max_size > 0)
                {
                    float max_size = max_sizes[k];

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

#if NCNN_VULKAN
int PriorBox::upload_model(VkTransfer& cmd)
{
    cmd.record_upload(min_sizes, min_sizes_gpu);

    if (max_sizes.w > 0)
        cmd.record_upload(max_sizes, max_sizes_gpu);

    cmd.record_upload(aspect_ratios, aspect_ratios_gpu);

    return 0;
}

int PriorBox::create_pipeline()
{
    // caffe style
    {
        int num_min_size = min_sizes.w;
        int num_max_size = max_sizes.w;
        int num_aspect_ratio = aspect_ratios.w;

        int num_prior = num_min_size * num_aspect_ratio + num_min_size + num_max_size;
        if (flip)
            num_prior += num_min_size * num_aspect_ratio;

        std::vector<vk_specialization_type> specializations(11);
        specializations[0].i = flip;
        specializations[1].i = clip;
        specializations[2].f = offset;
        specializations[3].f = variances[0];
        specializations[4].f = variances[1];
        specializations[5].f = variances[2];
        specializations[6].f = variances[3];
        specializations[7].i = num_min_size;
        specializations[8].i = num_max_size;
        specializations[9].i = num_aspect_ratio;
        specializations[10].i = num_prior;

        pipeline_priorbox = new Pipeline(vkdev);
        pipeline_priorbox->set_optimal_local_size_xyz();
        pipeline_priorbox->create("priorbox", specializations, 4, 6);
    }

    // mxnet style
    {
        int num_sizes = min_sizes.w;
        int num_ratios = aspect_ratios.w;

        int num_prior = num_sizes - 1 + num_ratios;

        std::vector<vk_specialization_type> specializations(5);
        specializations[0].i = clip;
        specializations[1].f = offset;
        specializations[2].i = num_sizes;
        specializations[3].i = num_ratios;
        specializations[4].i = num_prior;

        pipeline_priorbox_mxnet = new Pipeline(vkdev);
        pipeline_priorbox_mxnet->set_optimal_local_size_xyz();
        pipeline_priorbox_mxnet->create("priorbox_mxnet", specializations, 3, 4);
    }

    return 0;
}

int PriorBox::destroy_pipeline()
{
    delete pipeline_priorbox;
    pipeline_priorbox = 0;

    delete pipeline_priorbox_mxnet;
    pipeline_priorbox_mxnet = 0;

    return 0;
}

int PriorBox::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_blobs[0].w;
    int h = bottom_blobs[0].h;

    if (bottom_blobs.size() == 1 && image_width == -233 && image_height == -233 && max_sizes.empty())
    {
        // mxnet style _contrib_MultiBoxPrior
        float step_w = step_width;
        float step_h = step_height;
        if (step_w == -233)
            step_w = 1.f / (float)w;
        if (step_h == -233)
            step_h = 1.f / (float)h;

        int num_sizes = min_sizes.w;
        int num_ratios = aspect_ratios.w;

        int num_prior = num_sizes - 1 + num_ratios;

        VkMat& top_blob = top_blobs[0];
        top_blob.create(4 * w * h * num_prior, 4u, opt.blob_vkallocator, opt.staging_vkallocator);
        if (top_blob.empty())
            return -100;

        std::vector<VkMat> bindings(3);
        bindings[0] = top_blob;
        bindings[1] = min_sizes_gpu;
        bindings[2] = aspect_ratios_gpu;

        std::vector<vk_constant_type> constants(4);
        constants[0].i = w;
        constants[1].i = h;
        constants[2].f = step_w;
        constants[3].f = step_h;

        // record
        VkMat dispatcher;
        dispatcher.w = num_sizes;
        dispatcher.h = w;
        dispatcher.c = h;

        cmd.record_pipeline(pipeline_priorbox_mxnet, bindings, constants, dispatcher);

        return 0;
    }

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

    int num_prior = num_min_size * num_aspect_ratio + num_min_size + num_max_size;
    if (flip)
        num_prior += num_min_size * num_aspect_ratio;

    VkMat& top_blob = top_blobs[0];
    top_blob.create(4 * w * h * num_prior, 2, 4u, opt.blob_vkallocator, opt.staging_vkallocator);
    if (top_blob.empty())
        return -100;

//     fprintf(stderr, "PriorBox::forward %p\n", top_blob.buffer());

    std::vector<VkMat> bindings(4);
    bindings[0] = top_blob;
    bindings[1] = min_sizes_gpu;
    bindings[2] = num_max_size > 0 ? max_sizes_gpu : min_sizes_gpu;
    bindings[3] = aspect_ratios_gpu;

    std::vector<vk_constant_type> constants(6);
    constants[0].i = w;
    constants[1].i = h;
    constants[2].f = image_w;
    constants[3].f = image_h;
    constants[4].f = step_w;
    constants[5].f = step_h;

    // record
    VkMat dispatcher;
    dispatcher.w = num_min_size;
    dispatcher.h = w;
    dispatcher.c = h;

    cmd.record_pipeline(pipeline_priorbox, bindings, constants, dispatcher);

    return 0;
}
#endif // NCNN_VULKAN

} // namespace ncnn
