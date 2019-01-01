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

#include "batchnorm.h"
#include <math.h>

namespace ncnn {

DEFINE_LAYER_CREATOR(BatchNorm)

BatchNorm::BatchNorm()
{
    one_blob_only = true;
    support_inplace = true;
    support_vulkan = true;
}

int BatchNorm::load_param(const ParamDict& pd)
{
    channels = pd.get(0, 0);
    eps = pd.get(1, 0.f);

#if NCNN_VULKAN
    if (pd.use_vulkan_compute)
    {
        set_optimal_local_size_xyz(32, 32, channels);

        specializations.resize(0);

        binding_count = 3;
        push_constant_count = 5;
    }
#endif // NCNN_VULKAN

    return 0;
}

int BatchNorm::load_model(const ModelBin& mb)
{
    slope_data = mb.load(channels, 1);
    if (slope_data.empty())
        return -100;

    mean_data = mb.load(channels, 1);
    if (mean_data.empty())
        return -100;

    var_data = mb.load(channels, 1);
    if (var_data.empty())
        return -100;

    bias_data = mb.load(channels, 1);
    if (bias_data.empty())
        return -100;

    a_data.create(channels);
    if (a_data.empty())
        return -100;
    b_data.create(channels);
    if (b_data.empty())
        return -100;

    for (int i=0; i<channels; i++)
    {
        float sqrt_var = sqrt(var_data[i] + eps);
        a_data[i] = bias_data[i] - slope_data[i] * mean_data[i] / sqrt_var;
        b_data[i] = slope_data[i] / sqrt_var;
    }

    return 0;
}

int BatchNorm::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    // a = bias - slope * mean / sqrt(var)
    // b = slope / sqrt(var)
    // value = b * value + a

    int dims = bottom_top_blob.dims;

    if (dims == 1)
    {
        int w = bottom_top_blob.w;

        float* ptr = bottom_top_blob;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i=0; i<w; i++)
        {
            ptr[i] = b_data[i] * ptr[i] + a_data[i];
        }
    }

    if (dims == 2)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i=0; i<h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
            float a = a_data[i];
            float b = b_data[i];

            for (int j=0; j<w; j++)
            {
                ptr[j] = b * ptr[j] + a;
            }
        }
    }

    if (dims == 3)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int size = w * h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            float a = a_data[q];
            float b = b_data[q];

            for (int i=0; i<size; i++)
            {
                ptr[i] = b * ptr[i] + a;
            }
        }
    }

    return 0;
}

#if NCNN_VULKAN
int BatchNorm::upload_model(VkTransfer& cmd)
{
    cmd.record_upload(a_data, a_data_gpu);
    cmd.record_upload(b_data, b_data_gpu);

    return 0;
}

int BatchNorm::forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const
{
    fprintf(stderr, "BatchNorm::forward_inplace %p\n", bottom_top_blob.buffer);

    std::vector<VkMat> bindings(3);
    bindings[0] = bottom_top_blob;
    bindings[1] = a_data_gpu;
    bindings[2] = b_data_gpu;

    std::vector<vk_constant_type> constants(5);
    constants[0].i = bottom_top_blob.dims;
    constants[1].i = bottom_top_blob.w;
    constants[2].i = bottom_top_blob.h;
    constants[3].i = bottom_top_blob.c;
    constants[4].i = bottom_top_blob.cstep;

    uint32_t group_count_xyz[3];
    group_count_xyz[0] = (bottom_top_blob.w + local_size_x - 1) / local_size_x;
    group_count_xyz[1] = (bottom_top_blob.h + local_size_y - 1) / local_size_y;
    group_count_xyz[2] = (bottom_top_blob.c + local_size_z - 1) / local_size_z;

    // record
    cmd.record_bind_pipeline(pipeline);
    cmd.record_update_bindings(pipeline_layout, descriptorset_layout, descriptor_update_template, bindings);
    cmd.record_push_constants(pipeline_layout, constants);
    cmd.record_dispatch(group_count_xyz);

    return 0;
}
#endif // NCNN_VULKAN

} // namespace ncnn
