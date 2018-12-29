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

#include "scale.h"
#include <math.h>

namespace ncnn {

DEFINE_LAYER_CREATOR(Scale)

Scale::Scale()
{
    one_blob_only = true;
    support_inplace = true;
    support_vulkan = true;
}

int Scale::load_param(const ParamDict& pd)
{
    scale_data_size = pd.get(0, 0);
    bias_term = pd.get(1, 0);

    if (scale_data_size == -233)
        one_blob_only = false;

#if NCNN_VULKAN
    if (pd.use_vulkan_compute)
    {
        local_size_z = vkdev->info.max_workgroup_size[2];
        if (scale_data_size == -233)
        {
            local_size_z = std::min(128, vkdev->info.max_workgroup_size[2]);
        }
        else
        {
            local_size_z = vkdev->info.max_workgroup_size[2];
            while (scale_data_size < local_size_z)
            {
                local_size_z /= 2;
            }
        }

        int local_size_xy = sqrt(vkdev->info.max_workgroup_invocations / local_size_z);
        int local_size_xy_prefer = 64;
        while (local_size_xy < local_size_xy_prefer)
        {
            local_size_xy_prefer /= 2;
        }
        local_size_x = local_size_xy_prefer;
        local_size_y = local_size_xy_prefer;

        fprintf(stderr, "local size = %d %d %d\n", local_size_x, local_size_y, local_size_z);

        // setup pipeline specializations
        specializations.resize(1);
        specializations[0].i = bias_term;

        binding_count = 3;
        push_constant_count = 5;
    }
#endif // NCNN_VULKAN

    return 0;
}

int Scale::load_model(const ModelBin& mb)
{
    if (scale_data_size != -233)
    {
        scale_data = mb.load(scale_data_size, 1);
        if (scale_data.empty())
            return -100;
    }

    if (bias_term)
    {
        bias_data = mb.load(scale_data_size, 1);
        if (bias_data.empty())
            return -100;
    }

#if NCNN_VULKAN
    if (mb.vk_model_loader)
    {
        // upload weight data
        if (scale_data_size != -233)
        {
            scale_data_gpu.create_like(scale_data, mb.weight_vkallocator, mb.staging_vkallocator);
            scale_data_gpu.prepare_staging_buffer();

            mb.vk_model_loader->record_upload(scale_data_gpu);
            mb.vk_model_loader->record_upload_compute_barrier(scale_data_gpu);

            scale_data_gpu.map();
            scale_data_gpu.staging_buffer_upload(scale_data);
            scale_data_gpu.unmap();
        }

        if (bias_term)
        {
            bias_data_gpu.create_like(bias_data, mb.weight_vkallocator, mb.staging_vkallocator);
            bias_data_gpu.prepare_staging_buffer();

            mb.vk_model_loader->record_upload(bias_data_gpu);
            mb.vk_model_loader->record_upload_compute_barrier(bias_data_gpu);

            bias_data_gpu.map();
            bias_data_gpu.staging_buffer_upload(bias_data);
            bias_data_gpu.unmap();
        }
    }
#endif // NCNN_VULKAN

    return 0;
}

int Scale::forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const
{
    Mat& bottom_top_blob = bottom_top_blobs[0];
    const Mat& scale_blob = bottom_top_blobs[1];

    int dims = bottom_top_blob.dims;

    if (dims == 1)
    {
        int w = bottom_top_blob.w;

        float* ptr = bottom_top_blob;

        if (bias_term)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i=0; i<w; i++)
            {
                ptr[i] = ptr[i] * scale_blob[i] + bias_data[i];
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i=0; i<w; i++)
            {
                ptr[i] *= scale_blob[i];
            }
        }
    }

    if (dims == 2)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        if (bias_term)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i=0; i<h; i++)
            {
                float* ptr = bottom_top_blob.row(i);
                float s = scale_blob[i];
                float bias = bias_data[i];

                for (int j=0; j<w; j++)
                {
                    ptr[j] = ptr[j] * s + bias;
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i=0; i<h; i++)
            {
                float* ptr = bottom_top_blob.row(i);
                float s = scale_blob[i];

                for (int j=0; j<w; j++)
                {
                    ptr[j] *= s;
                }
            }
        }
    }

    if (dims == 3)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int channels = bottom_top_blob.c;
        int size = w * h;

        if (bias_term)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);

                float s = scale_blob[q];
                float bias = bias_data[q];

                for (int i=0; i<size; i++)
                {
                    ptr[i] = ptr[i] * s + bias;
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);

                float s = scale_blob[q];

                for (int i=0; i<size; i++)
                {
                    ptr[i] *= s;
                }
            }
        }
    }

    return 0;
}

int Scale::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    std::vector<Mat> bottom_top_blobs(2);
    bottom_top_blobs[0] = bottom_top_blob;
    bottom_top_blobs[1] = scale_data;

    return forward_inplace(bottom_top_blobs, opt);
}

#if NCNN_VULKAN
int Scale::forward_inplace(std::vector<VkMat>& bottom_top_blobs, Command& cmd, const Option& opt) const
{
    VkMat& bottom_top_blob = bottom_top_blobs[0];
    const VkMat& scale_blob = bottom_top_blobs[1];

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;

    fprintf(stderr, "Scale::forward_inplace %p\n", bottom_top_blob.buffer);

    std::vector<VkMat> bindings(3);
    bindings[0] = bottom_top_blob;
    bindings[1] = scale_blob;
    bindings[2] = bias_data_gpu;

    std::vector<int> constants(5);
    constants[0] = bottom_top_blob.dims;
    constants[1] = bottom_top_blob.w;
    constants[2] = bottom_top_blob.h;
    constants[3] = bottom_top_blob.c;
    constants[4] = bottom_top_blob.cstep;

    uint32_t group_count_xyz[3];
    group_count_xyz[0] = (bottom_top_blob.w + local_size_x - 1) / local_size_x;
    group_count_xyz[1] = (bottom_top_blob.h + local_size_y - 1) / local_size_y;
    group_count_xyz[2] = (bottom_top_blob.c + local_size_z - 1) / local_size_z;

    // record
    cmd.record_bind_pipeline(pipeline);
    cmd.record_update_bindings(pipeline_layout, descriptor_update_template, bindings);
    cmd.record_push_constants(pipeline_layout, constants);
    cmd.record_dispatch(group_count_xyz);

    return 0;
}

int Scale::forward_inplace(VkMat& bottom_top_blob, Command& cmd, const Option& opt) const
{
    std::vector<VkMat> bottom_top_blobs(2);
    bottom_top_blobs[0] = bottom_top_blob;
    bottom_top_blobs[1] = scale_data_gpu;

    return forward_inplace(bottom_top_blobs, cmd, opt);
}
#endif // NCNN_VULKAN

} // namespace ncnn
