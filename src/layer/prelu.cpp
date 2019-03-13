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

#include "prelu.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(PReLU)

PReLU::PReLU()
{
    one_blob_only = true;
    support_inplace = true;
    support_vulkan = true;

#if NCNN_VULKAN
    pipeline_prelu = 0;
    pipeline_prelu_pack4 = 0;
#endif // NCNN_VULKAN
}

int PReLU::load_param(const ParamDict& pd)
{
    num_slope = pd.get(0, 0);

    return 0;
}

int PReLU::load_model(const ModelBin& mb)
{
    slope_data = mb.load(num_slope, 1);
    if (slope_data.empty())
        return -100;

    return 0;
}

int PReLU::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;

    if (dims == 1)
    {
        int w = bottom_top_blob.w;

        float* ptr = bottom_top_blob;

        if (num_slope > 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i=0; i<w; i++)
            {
                if (ptr[i] < 0)
                    ptr[i] *= slope_data[i];
            }
        }
        else
        {
            float slope = slope_data[0];

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i=0; i<w; i++)
            {
                if (ptr[i] < 0)
                    ptr[i] *= slope;
            }
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
            float slope = num_slope > 1 ? slope_data[i] : slope_data[0];

            for (int j=0; j<w; j++)
            {
                if (ptr[j] < 0)
                    ptr[j] *= slope;
            }
        }
    }

    if (dims == 3)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int channels = bottom_top_blob.c;
        int size = w * h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            float slope = num_slope > 1 ? slope_data[q] : slope_data[0];

            for (int i=0; i<size; i++)
            {
                if (ptr[i] < 0)
                    ptr[i] *= slope;
            }
        }
    }

    return 0;
}

#if NCNN_VULKAN
int PReLU::upload_model(VkTransfer& cmd)
{
    if (num_slope == 1)
    {
        // dup4 for pack4
        Mat slope_data4(4);
        slope_data4.fill(slope_data[0]);
        cmd.record_upload(slope_data4, slope_data_gpu);
    }
    else
    {
        cmd.record_upload(slope_data, slope_data_gpu);
    }

    return 0;
}

int PReLU::create_pipeline()
{
    std::vector<vk_specialization_type> specializations(1);
    specializations[0].i = num_slope;

    // pack1
    if (num_slope == 1 || num_slope % 4 != 0)
    {
        pipeline_prelu = new Pipeline(vkdev);
        pipeline_prelu->set_optimal_local_size_xyz(8, 8, num_slope);
        pipeline_prelu->create("prelu", specializations, 2, 5);
    }

    // pack4
    if (num_slope == 1 || num_slope % 4 == 0)
    {
        pipeline_prelu_pack4 = new Pipeline(vkdev);
        pipeline_prelu_pack4->set_optimal_local_size_xyz(8, 8, num_slope / 4);
        pipeline_prelu_pack4->create("prelu_pack4", specializations, 2, 5);
    }

    return 0;
}

int PReLU::destroy_pipeline()
{
    delete pipeline_prelu;
    pipeline_prelu = 0;

    delete pipeline_prelu_pack4;
    pipeline_prelu_pack4 = 0;

    return 0;
}

int PReLU::forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const
{
    int packing = bottom_top_blob.packing;

//     fprintf(stderr, "PReLU::forward_inplace %p\n", bottom_top_blob.buffer());

    std::vector<VkMat> bindings(2);
    bindings[0] = bottom_top_blob;
    bindings[1] = slope_data_gpu;

    std::vector<vk_constant_type> constants(5);
    constants[0].i = bottom_top_blob.dims;
    constants[1].i = bottom_top_blob.w;
    constants[2].i = bottom_top_blob.h;
    constants[3].i = bottom_top_blob.c;
    constants[4].i = bottom_top_blob.cstep;

    const Pipeline* pipeline = packing == 4 ? pipeline_prelu_pack4 : pipeline_prelu;

    // record
    cmd.record_pipeline(pipeline, bindings, constants, bottom_top_blob);

    return 0;
}
#endif // NCNN_VULKAN

} // namespace ncnn
