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

#include "permute.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Permute)

Permute::Permute()
{
    one_blob_only = true;
    support_inplace = false;
    support_vulkan = true;

#if NCNN_VULKAN
    pipeline_permute = 0;
    pipeline_permute_pack4to1 = 0;
#endif // NCNN_VULKAN
}

int Permute::load_param(const ParamDict& pd)
{
    order_type = pd.get(0, 0);

    return 0;
}

int Permute::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    int dims = bottom_blob.dims;

    if (dims == 2)
    {
        // order_type
        // 0 = w h
        // 1 = h w

        if (order_type == 0)
        {
            top_blob = bottom_blob;
        }
        else if (order_type == 1)
        {
            top_blob.create(h, w, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            const float* ptr = bottom_blob;
            float* outptr = top_blob;

            for (int i = 0; i < w; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    outptr[i*h + j] = ptr[j*w + i];
                }
            }
        }

        return 0;
    }

    // order_type
    // 0 = w h c
    // 1 = h w c
    // 2 = w c h
    // 3 = c w h
    // 4 = h c w
    // 5 = c h w

    if (order_type == 0)
    {
        top_blob = bottom_blob;
    }
    else if (order_type == 1)
    {
        top_blob.create(h, w, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < w; i++)
            {
                for (int j = 0; j < h; j++)
                {
                    outptr[i*h + j] = ptr[j*w + i];
                }
            }
        }
    }
    else if (order_type == 2)
    {
        top_blob.create(w, channels, h, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<h; q++)
        {
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < channels; i++)
            {
                const float* ptr = bottom_blob.channel(i).row(q);

                for (int j = 0; j < w; j++)
                {
                    outptr[i*w + j] = ptr[j];
                }
            }
        }
    }
    else if (order_type == 3)
    {
        top_blob.create(channels, w, h, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<h; q++)
        {
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < w; i++)
            {
                for (int j = 0; j < channels; j++)
                {
                    const float* ptr = bottom_blob.channel(j).row(q);

                    outptr[i*channels + j] = ptr[i];
                }
            }
        }
    }
    else if (order_type == 4)
    {
        top_blob.create(h, channels, w, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<w; q++)
        {
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < channels; i++)
            {
                const float* ptr = bottom_blob.channel(i);

                for (int j = 0; j < h; j++)
                {
                    outptr[i*h + j] = ptr[j*w + q];
                }
            }
        }
    }
    else if (order_type == 5)
    {
        top_blob.create(channels, h, w, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<w; q++)
        {
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < channels; j++)
                {
                    const float* ptr = bottom_blob.channel(j);

                    outptr[i*channels + j] = ptr[i*w + q];
                }
            }
        }
    }

    return 0;
}

#if NCNN_VULKAN
int Permute::create_pipeline()
{
    std::vector<vk_specialization_type> specializations(1);
    specializations[0].i = order_type;

    pipeline_permute = new Pipeline(vkdev);
    pipeline_permute->set_optimal_local_size_xyz();
    pipeline_permute->create("permute", specializations, 2, 10);

    // pack4
    {
        pipeline_permute_pack4to1 = new Pipeline(vkdev);
        pipeline_permute_pack4to1->set_optimal_local_size_xyz();
        pipeline_permute_pack4to1->create("permute_pack4to1", specializations, 2, 10);
    }

    return 0;
}

int Permute::destroy_pipeline()
{
    delete pipeline_permute;
    pipeline_permute = 0;

    delete pipeline_permute_pack4to1;
    pipeline_permute_pack4to1 = 0;

    return 0;
}

int Permute::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int packing = bottom_blob.packing;

    int dims = bottom_blob.dims;

    int out_packing = 1;
    size_t out_elemsize = elemsize / packing;

    if (dims == 2)
    {
        // order_type
        // 0 = w h
        // 1 = h w

        h = h * packing;

        if (order_type == 0)
        {
            top_blob.create(w, h, out_elemsize, out_packing, opt.blob_vkallocator, opt.staging_vkallocator);
            if (top_blob.empty())
                return -100;
        }
        else if (order_type == 1)
        {
            top_blob.create(h, w, out_elemsize, out_packing, opt.blob_vkallocator, opt.staging_vkallocator);
            if (top_blob.empty())
                return -100;
        }
    }
    else if (dims == 3)
    {
        // order_type
        // 0 = w h c
        // 1 = h w c
        // 2 = w c h
        // 3 = c w h
        // 4 = h c w
        // 5 = c h w

        channels = channels * packing;

        if (order_type == 0)
        {
            top_blob.create(w, h, channels, out_elemsize, out_packing, opt.blob_vkallocator, opt.staging_vkallocator);
            if (top_blob.empty())
                return -100;
        }
        else if (order_type == 1)
        {
            top_blob.create(h, w, channels, out_elemsize, out_packing, opt.blob_vkallocator, opt.staging_vkallocator);
            if (top_blob.empty())
                return -100;
        }
        else if (order_type == 2)
        {
            top_blob.create(w, channels, h, out_elemsize, out_packing, opt.blob_vkallocator, opt.staging_vkallocator);
            if (top_blob.empty())
                return -100;
        }
        else if (order_type == 3)
        {
            top_blob.create(channels, w, h, out_elemsize, out_packing, opt.blob_vkallocator, opt.staging_vkallocator);
            if (top_blob.empty())
                return -100;
        }
        else if (order_type == 4)
        {
            top_blob.create(h, channels, w, out_elemsize, out_packing, opt.blob_vkallocator, opt.staging_vkallocator);
            if (top_blob.empty())
                return -100;
        }
        else if (order_type == 5)
        {
            top_blob.create(channels, h, w, out_elemsize, out_packing, opt.blob_vkallocator, opt.staging_vkallocator);
            if (top_blob.empty())
                return -100;
        }
    }

//     fprintf(stderr, "Permute::forward %p %p\n", bottom_blob.buffer(), top_blob.buffer());

    std::vector<VkMat> bindings(2);
    bindings[0] = bottom_blob;
    bindings[1] = top_blob;

    std::vector<vk_constant_type> constants(10);
    constants[0].i = bottom_blob.dims;
    constants[1].i = bottom_blob.w;
    constants[2].i = bottom_blob.h;
    constants[3].i = bottom_blob.c;
    constants[4].i = bottom_blob.cstep;
    constants[5].i = top_blob.dims;
    constants[6].i = top_blob.w;
    constants[7].i = top_blob.h;
    constants[8].i = top_blob.c;
    constants[9].i = top_blob.cstep;

    // record
    if (packing == 1)
    {
        cmd.record_pipeline(pipeline_permute, bindings, constants, top_blob);
    }

    if (packing == 4)
    {
        cmd.record_pipeline(pipeline_permute_pack4to1, bindings, constants, bottom_blob);
    }

    return 0;
}
#endif // NCNN_VULKAN

} // namespace ncnn
