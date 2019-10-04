// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "scale_vulkan.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Scale_vulkan)

Scale_vulkan::Scale_vulkan()
{
    support_vulkan = true;

    pipeline_scale = 0;
    pipeline_scale_pack4 = 0;
}

int Scale_vulkan::create_pipeline(const Option& opt)
{
    if (scale_data_size == -233)
    {
        std::vector<vk_specialization_type> specializations(1);
        specializations[0].i = 0;

        pipeline_scale = new Pipeline(vkdev);
        pipeline_scale->set_optimal_local_size_xyz();
        pipeline_scale->create("scale", opt, specializations, 3, 5);

        // pack4
        {
            pipeline_scale_pack4 = new Pipeline(vkdev);
            pipeline_scale_pack4->set_optimal_local_size_xyz();
            pipeline_scale_pack4->create("scale_pack4", opt, specializations, 3, 5);
        }

        return 0;
    }

    std::vector<vk_specialization_type> specializations(1);
    specializations[0].i = bias_term;

    // pack1
    if (scale_data_size % 4 != 0)
    {
        pipeline_scale = new Pipeline(vkdev);
        pipeline_scale->set_optimal_local_size_xyz(8, 8, scale_data_size);
        pipeline_scale->create("scale", opt, specializations, 3, 5);
    }

    // pack4
    if (scale_data_size % 4 == 0)
    {
        pipeline_scale_pack4 = new Pipeline(vkdev);
        pipeline_scale_pack4->set_optimal_local_size_xyz(8, 8, scale_data_size / 4);
        pipeline_scale_pack4->create("scale_pack4", opt, specializations, 3, 5);
    }

    return 0;
}

int Scale_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_scale;
    pipeline_scale = 0;

    delete pipeline_scale_pack4;
    pipeline_scale_pack4 = 0;

    return 0;
}

int Scale_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    if (scale_data_size == -233)
        return 0;

    // pack1
    if (scale_data_size % 4 != 0)
    {
        cmd.record_upload(scale_data, scale_data_gpu, opt);
    }

    // pack4
    if (scale_data_size % 4 == 0)
    {
        Mat scale_data_pack4;
        convert_packing(scale_data, scale_data_pack4, 4);
        cmd.record_upload(scale_data_pack4, scale_data_gpu_pack4, opt);
    }

    if (bias_term)
    {
        // pack1
        if (scale_data_size % 4 != 0)
        {
            cmd.record_upload(bias_data, bias_data_gpu, opt);
        }

        // pack4
        if (scale_data_size % 4 == 0)
        {
            Mat bias_data_pack4;
            convert_packing(bias_data, bias_data_pack4, 4);
            cmd.record_upload(bias_data_pack4, bias_data_gpu_pack4, opt);
        }
    }

    return 0;
}

int Scale_vulkan::forward_inplace(std::vector<VkMat>& bottom_top_blobs, VkCompute& cmd, const Option& /*opt*/) const
{
    VkMat& bottom_top_blob = bottom_top_blobs[0];
    const VkMat& scale_blob = bottom_top_blobs[1];

    int elempack = bottom_top_blob.elempack;

    std::vector<VkMat> bindings(3);
    bindings[0] = bottom_top_blob;
    bindings[1] = scale_blob;
    bindings[2] = bias_term ? (elempack == 4 ? bias_data_gpu_pack4 : bias_data_gpu) : scale_blob;// TODO use dummy buffer

    std::vector<vk_constant_type> constants(5);
    constants[0].i = bottom_top_blob.dims;
    constants[1].i = bottom_top_blob.w;
    constants[2].i = bottom_top_blob.h;
    constants[3].i = bottom_top_blob.c;
    constants[4].i = bottom_top_blob.cstep;

    const Pipeline* pipeline = elempack == 4 ? pipeline_scale_pack4 : pipeline_scale;

    cmd.record_pipeline(pipeline, bindings, constants, bottom_top_blob);

    return 0;
}

int Scale_vulkan::forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const
{
    int elempack = bottom_top_blob.elempack;

    std::vector<VkMat> bottom_top_blobs(2);
    bottom_top_blobs[0] = bottom_top_blob;
    bottom_top_blobs[1] = elempack == 4 ? scale_data_gpu_pack4 : scale_data_gpu;

    return forward_inplace(bottom_top_blobs, cmd, opt);
}

} // namespace ncnn
