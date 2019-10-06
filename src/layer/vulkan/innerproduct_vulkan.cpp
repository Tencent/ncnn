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

#include "innerproduct_vulkan.h"
#include <algorithm>
#include "layer_type.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(InnerProduct_vulkan)

InnerProduct_vulkan::InnerProduct_vulkan()
{
    support_vulkan = true;

    flatten = 0;

    pipeline_innerproduct = 0;
    pipeline_innerproduct_pack4 = 0;
    pipeline_innerproduct_pack4_lds_64 = 0;
    pipeline_innerproduct_pack1to4 = 0;
    pipeline_innerproduct_pack4to1 = 0;
}

int InnerProduct_vulkan::create_pipeline(const Option& opt)
{
    {
        flatten = ncnn::create_layer(ncnn::LayerType::Flatten);
        flatten->vkdev = vkdev;

        ncnn::ParamDict pd;

        flatten->load_param(pd);

        flatten->create_pipeline(opt);
    }

    int num_input = weight_data_size / num_output;

    std::vector<vk_specialization_type> specializations(4);
    specializations[0].i = bias_term;
    specializations[1].i = activation_type;
    specializations[2].f = activation_params.w == 1 ? activation_params[0] : 0.f;
    specializations[3].f = activation_params.w == 2 ? activation_params[1] : 0.f;

    // pack1
    if (num_input % 4 != 0 && num_output % 4 != 0)
    {
        pipeline_innerproduct = new Pipeline(vkdev);
        pipeline_innerproduct->set_optimal_local_size_xyz(num_output, 1, 1);
        pipeline_innerproduct->create("innerproduct", opt, specializations, 4, 10);
    }

    // pack4
    if (num_input % 4 == 0 && num_output % 4 == 0)
    {
        pipeline_innerproduct_pack4 = new Pipeline(vkdev);
        pipeline_innerproduct_pack4->set_optimal_local_size_xyz(num_output / 4, 1, 1);
        pipeline_innerproduct_pack4->create("innerproduct_pack4", opt, specializations, 4, 10);

        {
            pipeline_innerproduct_pack4_lds_64 = new Pipeline(vkdev);
            pipeline_innerproduct_pack4_lds_64->set_local_size_xyz(64, 1, 1);
            pipeline_innerproduct_pack4_lds_64->create("innerproduct_pack4_lds_64", opt, specializations, 4, 10);
        }
    }

    // pack1to4
    if (num_input % 4 != 0 && num_output % 4 == 0)
    {
        pipeline_innerproduct_pack1to4 = new Pipeline(vkdev);
        pipeline_innerproduct_pack1to4->set_optimal_local_size_xyz(num_output / 4, 1, 1);
        pipeline_innerproduct_pack1to4->create("innerproduct_pack1to4", opt, specializations, 4, 10);
    }

    // pack4to1
    if (num_input % 4 == 0 && num_output % 4 != 0)
    {
        pipeline_innerproduct_pack4to1 = new Pipeline(vkdev);
        pipeline_innerproduct_pack4to1->set_optimal_local_size_xyz(num_output, 1, 1);
        pipeline_innerproduct_pack4to1->create("innerproduct_pack4to1", opt, specializations, 4, 10);
    }

    return 0;
}

int InnerProduct_vulkan::destroy_pipeline(const Option& opt)
{
    if (flatten)
    {
        flatten->destroy_pipeline(opt);
        delete flatten;
        flatten = 0;
    }

    delete pipeline_innerproduct;
    pipeline_innerproduct = 0;

    delete pipeline_innerproduct_pack4;
    pipeline_innerproduct_pack4 = 0;

    delete pipeline_innerproduct_pack4_lds_64;
    pipeline_innerproduct_pack4_lds_64 = 0;

    delete pipeline_innerproduct_pack1to4;
    pipeline_innerproduct_pack1to4 = 0;

    delete pipeline_innerproduct_pack4to1;
    pipeline_innerproduct_pack4to1 = 0;

    return 0;
}

int InnerProduct_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    int num_input = weight_data_size / num_output;

    // pack1
    if (num_input % 4 != 0 && num_output % 4 != 0)
    {
        cmd.record_upload(weight_data, weight_data_gpu, opt);
    }

    // pack4
    if (num_input % 4 == 0 && num_output % 4 == 0)
    {
        // src = inch-outch
        // dst = 4a-4b-inch/4a-outch/4b
        Mat weight_data_pack4;
        {
            Mat weight_data_r2 = weight_data.reshape(num_input, num_output);

            weight_data_pack4.create(num_input/4, num_output/4, (size_t)4*16, 16);

            for (int q=0; q+3<num_output; q+=4)
            {
                const float* k0 = weight_data_r2.row(q);
                const float* k1 = weight_data_r2.row(q+1);
                const float* k2 = weight_data_r2.row(q+2);
                const float* k3 = weight_data_r2.row(q+3);

                float* g00 = weight_data_pack4.row(q/4);

                for (int p=0; p+3<num_input; p+=4)
                {
                    g00[0] = k0[0];
                    g00[1] = k0[1];
                    g00[2] = k0[2];
                    g00[3] = k0[3];

                    g00[4] = k1[0];
                    g00[5] = k1[1];
                    g00[6] = k1[2];
                    g00[7] = k1[3];

                    g00[8] = k2[0];
                    g00[9] = k2[1];
                    g00[10] = k2[2];
                    g00[11] = k2[3];

                    g00[12] = k3[0];
                    g00[13] = k3[1];
                    g00[14] = k3[2];
                    g00[15] = k3[3];

                    k0 += 4;
                    k1 += 4;
                    k2 += 4;
                    k3 += 4;
                    g00 += 16;
                }
            }
        }

        cmd.record_upload(weight_data_pack4, weight_data_gpu_pack4, opt);
    }

    // pack1to4
    if (num_input % 4 != 0 && num_output % 4 == 0)
    {
        // src = inch-outch
        // dst = 4b-inch-outch/4b
        Mat weight_data_pack1to4;
        {
            Mat weight_data_r2 = weight_data.reshape(num_input, num_output);

            weight_data_pack1to4.create(num_input, num_output/4, (size_t)4*4, 4);

            for (int q=0; q+3<num_output; q+=4)
            {
                const float* k0 = weight_data_r2.row(q);
                const float* k1 = weight_data_r2.row(q+1);
                const float* k2 = weight_data_r2.row(q+2);
                const float* k3 = weight_data_r2.row(q+3);

                float* g00 = weight_data_pack1to4.row(q/4);

                for (int p=0; p<num_input; p++)
                {
                    g00[0] = k0[p];
                    g00[1] = k1[p];
                    g00[2] = k2[p];
                    g00[3] = k3[p];

                    g00 += 4;
                }
            }
        }

        cmd.record_upload(weight_data_pack1to4, weight_data_gpu_pack1to4, opt);
    }

    // pack4to1
    if (num_input % 4 == 0 && num_output % 4 != 0)
    {
        // src = inch-outch
        // dst = 4a-inch/4a-outch
        Mat weight_data_pack4to1;
        {
            Mat weight_data_r2 = weight_data.reshape(num_input, num_output);

            weight_data_pack4to1.create(num_input/4, num_output, (size_t)4*4, 4);

            for (int q=0; q<num_output; q++)
            {
                const float* k0 = weight_data_r2.row(q);

                float* g00 = weight_data_pack4to1.row(q);

                for (int p=0; p+3<num_input; p+=4)
                {
                    g00[0] = k0[0];
                    g00[1] = k0[1];
                    g00[2] = k0[2];
                    g00[3] = k0[3];

                    k0 += 4;
                    g00 += 4;
                }
            }
        }

        cmd.record_upload(weight_data_pack4to1, weight_data_gpu_pack4to1, opt);
    }

    if (bias_term)
    {
        if (num_output % 4 != 0)
        {
            cmd.record_upload(bias_data, bias_data_gpu, opt);
        }

        if (num_output % 4 == 0)
        {
            Mat bias_data_pack4;
            convert_packing(bias_data, bias_data_pack4, 4);
            cmd.record_upload(bias_data_pack4, bias_data_gpu_pack4, opt);
        }
    }

    return 0;
}

int InnerProduct_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    // flatten
    VkMat bottom_blob_flattened = bottom_blob;
    {
        ncnn::Option opt_flatten = opt;
        opt_flatten.blob_vkallocator = opt.workspace_vkallocator;

        flatten->forward(bottom_blob, bottom_blob_flattened, cmd, opt_flatten);
    }

    int out_elempack = num_output % 4 == 0 ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    if (opt.use_fp16_packed && !opt.use_fp16_storage)
    {
        if (out_elempack == 4) out_elemsize = 4*2u;
        if (out_elempack == 1) out_elemsize = 4u;
    }

    top_blob.create(num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator, opt.staging_vkallocator);
    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(4);
    bindings[0] = bottom_blob_flattened;
    bindings[1] = top_blob;
    if (elempack == 1 && out_elempack == 1)
    {
        bindings[2] = weight_data_gpu;
        bindings[3] = bias_term ? bias_data_gpu : bindings[2];// TODO use dummy buffer
    }
    else if (elempack == 4 && out_elempack == 4)
    {
        bindings[2] = weight_data_gpu_pack4;
        bindings[3] = bias_term ? bias_data_gpu_pack4 : bindings[2];// TODO use dummy buffer
    }
    else if (elempack == 1 && out_elempack == 4)
    {
        bindings[2] = weight_data_gpu_pack1to4;
        bindings[3] = bias_term ? bias_data_gpu_pack4 : bindings[2];// TODO use dummy buffer
    }
    else if (elempack == 4 && out_elempack == 1)
    {
        bindings[2] = weight_data_gpu_pack4to1;
        bindings[3] = bias_term ? bias_data_gpu : bindings[2];// TODO use dummy buffer
    }

    std::vector<vk_constant_type> constants(10);
    constants[0].i = bottom_blob_flattened.dims;
    constants[1].i = bottom_blob_flattened.w;
    constants[2].i = bottom_blob_flattened.h;
    constants[3].i = bottom_blob_flattened.c;
    constants[4].i = bottom_blob_flattened.cstep;
    constants[5].i = top_blob.dims;
    constants[6].i = top_blob.w;
    constants[7].i = top_blob.h;
    constants[8].i = top_blob.c;
    constants[9].i = top_blob.cstep;

    const Pipeline* pipeline = 0;
    if (elempack == 1 && out_elempack == 1)
    {
        pipeline = pipeline_innerproduct;
    }
    else if (elempack == 4 && out_elempack == 4)
    {
        pipeline = pipeline_innerproduct_pack4;

        pipeline = pipeline_innerproduct_pack4_lds_64;
    }
    else if (elempack == 1 && out_elempack == 4)
    {
        pipeline = pipeline_innerproduct_pack1to4;
    }
    else if (elempack == 4 && out_elempack == 1)
    {
        pipeline = pipeline_innerproduct_pack4to1;
    }

    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    return 0;
}

} // namespace ncnn
