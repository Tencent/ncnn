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

namespace ncnn {

DEFINE_LAYER_CREATOR(Scale)

Scale::Scale()
{
    one_blob_only = true;
    support_inplace = true;
    support_vulkan = true;

#if NCNN_VULKAN
    pipeline_scale = 0;
    pipeline_scale_pack4 = 0;
#endif // NCNN_VULKAN
}

int Scale::load_param(const ParamDict& pd)
{
    scale_data_size = pd.get(0, 0);
    bias_term = pd.get(1, 0);

    if (scale_data_size == -233)
        one_blob_only = false;

    return 0;
}

int Scale::load_model(const ModelBin& mb)
{
    if (scale_data_size == -233)
        return 0;

    scale_data = mb.load(scale_data_size, 1);
    if (scale_data.empty())
        return -100;

    if (bias_term)
    {
        bias_data = mb.load(scale_data_size, 1);
        if (bias_data.empty())
            return -100;
    }

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
int Scale::upload_model(VkTransfer& cmd)
{
    if (scale_data_size == -233)
        return 0;

    // pack1
    if (scale_data_size % 4 != 0)
    {
        cmd.record_upload(scale_data, scale_data_gpu);
    }

    // pack4
    if (scale_data_size % 4 == 0)
    {
        Mat scale_data_pack4;
        convert_packing(scale_data, scale_data_pack4, 4);
        cmd.record_upload(scale_data_pack4, scale_data_gpu_pack4);
    }

    if (bias_term)
    {
        // pack1
        if (scale_data_size % 4 != 0)
        {
            cmd.record_upload(bias_data, bias_data_gpu);
        }

        // pack4
        if (scale_data_size % 4 == 0)
        {
            Mat bias_data_pack4;
            convert_packing(bias_data, bias_data_pack4, 4);
            cmd.record_upload(bias_data_pack4, bias_data_gpu_pack4);
        }
    }

    return 0;
}

int Scale::create_pipeline()
{
    if (scale_data_size == -233)
    {
        std::vector<vk_specialization_type> specializations(1);
        specializations[0].i = 0;

        pipeline_scale = new Pipeline(vkdev);
        pipeline_scale->set_optimal_local_size_xyz();
        pipeline_scale->create("scale", specializations, 3, 5);

        // pack4
        {
            pipeline_scale_pack4 = new Pipeline(vkdev);
            pipeline_scale_pack4->set_optimal_local_size_xyz();
            pipeline_scale_pack4->create("scale_pack4", specializations, 3, 5);
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
        pipeline_scale->create("scale", specializations, 3, 5);
    }

    // pack4
    if (scale_data_size % 4 == 0)
    {
        pipeline_scale_pack4 = new Pipeline(vkdev);
        pipeline_scale_pack4->set_optimal_local_size_xyz(8, 8, scale_data_size / 4);
        pipeline_scale_pack4->create("scale_pack4", specializations, 3, 5);
    }

    return 0;
}

int Scale::destroy_pipeline()
{
    delete pipeline_scale;
    pipeline_scale = 0;

    delete pipeline_scale_pack4;
    pipeline_scale_pack4 = 0;

    return 0;
}

int Scale::forward_inplace(std::vector<VkMat>& bottom_top_blobs, VkCompute& cmd, const Option& opt) const
{
    VkMat& bottom_top_blob = bottom_top_blobs[0];
    const VkMat& scale_blob = bottom_top_blobs[1];

    int packing = bottom_top_blob.packing;

//     fprintf(stderr, "Scale::forward_inplace %p\n", bottom_top_blob.buffer());

    std::vector<VkMat> bindings(3);
    bindings[0] = bottom_top_blob;
    bindings[1] = scale_blob;
    bindings[2] = bias_term ? (packing == 4 ? bias_data_gpu_pack4 : bias_data_gpu) : scale_blob;// TODO use dummy buffer

    std::vector<vk_constant_type> constants(5);
    constants[0].i = bottom_top_blob.dims;
    constants[1].i = bottom_top_blob.w;
    constants[2].i = bottom_top_blob.h;
    constants[3].i = bottom_top_blob.c;
    constants[4].i = bottom_top_blob.cstep;

    const Pipeline* pipeline = packing == 4 ? pipeline_scale_pack4 : pipeline_scale;

    // record
    cmd.record_pipeline(pipeline, bindings, constants, bottom_top_blob);

    return 0;
}

int Scale::forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const
{
    int packing = bottom_top_blob.packing;

    std::vector<VkMat> bottom_top_blobs(2);
    bottom_top_blobs[0] = bottom_top_blob;
    bottom_top_blobs[1] = packing == 4 ? scale_data_gpu_pack4 : scale_data_gpu;

    return forward_inplace(bottom_top_blobs, cmd, opt);
}
#endif // NCNN_VULKAN

} // namespace ncnn
