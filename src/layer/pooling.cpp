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

#include "pooling.h"
#include <float.h>
#include <algorithm>
#include "layer_type.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Pooling)

Pooling::Pooling()
{
    one_blob_only = true;
    support_inplace = false;
    support_vulkan = true;

#if NCNN_VULKAN
    padding = 0;
    pipeline_pooling = 0;
    pipeline_pooling_global = 0;
    pipeline_pooling_pack4 = 0;
    pipeline_pooling_global_pack4 = 0;
#endif // NCNN_VULKAN
}

Pooling::~Pooling()
{
#if NCNN_VULKAN
    delete padding;
#endif // NCNN_VULKAN
}

int Pooling::load_param(const ParamDict& pd)
{
    pooling_type = pd.get(0, 0);
    kernel_w = pd.get(1, 0);
    kernel_h = pd.get(11, kernel_w);
    stride_w = pd.get(2, 1);
    stride_h = pd.get(12, stride_w);
    pad_left = pd.get(3, 0);
    pad_right = pd.get(14, pad_left);
    pad_top = pd.get(13, pad_left);
    pad_bottom = pd.get(15, pad_top);
    global_pooling = pd.get(4, 0);
    pad_mode = pd.get(5, 0);

#if NCNN_VULKAN
    if (pd.use_vulkan_compute)
    {
        padding = ncnn::create_layer(ncnn::LayerType::Padding);
        padding->vkdev = vkdev;

        ncnn::ParamDict pd;
        pd.set(0, pad_top);
        pd.set(1, pad_bottom);
        pd.set(2, pad_left);
        pd.set(3, pad_right);
        pd.set(4, 0);

        if (pooling_type == PoolMethod_MAX)
        {
            pd.set(5, -FLT_MAX);
        }
        else if (pooling_type == PoolMethod_AVE)
        {
            pd.set(5, 0.f);
        }

        pd.use_vulkan_compute = 1;

        padding->load_param(pd);
    }
#endif // NCNN_VULKAN

    return 0;
}

int Pooling::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // max value in NxN window
    // avg value in NxN window

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

//     fprintf(stderr, "Pooling     input %d x %d  pad = %d %d %d %d  ksize=%d %d  stride=%d %d\n", w, h, pad_left, pad_right, pad_top, pad_bottom, kernel_w, kernel_h, stride_w, stride_h);
    if (global_pooling)
    {
        top_blob.create(channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        int size = w * h;

        if (pooling_type == PoolMethod_MAX)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);

                float max = ptr[0];
                for (int i=0; i<size; i++)
                {
                    max = std::max(max, ptr[i]);
                }

                top_blob[q] = max;
            }
        }
        else if (pooling_type == PoolMethod_AVE)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);

                float sum = 0.f;
                for (int i=0; i<size; i++)
                {
                    sum += ptr[i];
                }

                top_blob[q] = sum / size;
            }
        }

        return 0;
    }

    Mat bottom_blob_bordered = bottom_blob;

    float pad_value = 0.f;
    if (pooling_type == PoolMethod_MAX)
    {
        pad_value = -FLT_MAX;
    }
    else if (pooling_type == PoolMethod_AVE)
    {
        pad_value = 0.f;
    }

    int wtailpad = 0;
    int htailpad = 0;

    if (pad_mode == 0) // full padding
    {
        int wtail = (w + pad_left + pad_right - kernel_w) % stride_w;
        int htail = (h + pad_top + pad_bottom - kernel_h) % stride_h;

        if (wtail != 0)
            wtailpad = stride_w - wtail;
        if (htail != 0)
            htailpad = stride_h - htail;

        copy_make_border(bottom_blob, bottom_blob_bordered, pad_top, pad_bottom + htailpad, pad_left, pad_right + wtailpad, BORDER_CONSTANT, pad_value, opt.workspace_allocator, opt.num_threads);
        if (bottom_blob_bordered.empty())
            return -100;

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;
    }
    else if (pad_mode == 1) // valid padding
    {
        copy_make_border(bottom_blob, bottom_blob_bordered, pad_top, pad_bottom, pad_left, pad_right, BORDER_CONSTANT, pad_value, opt.workspace_allocator, opt.num_threads);
        if (bottom_blob_bordered.empty())
            return -100;

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;
    }
    else if (pad_mode == 2) // tensorflow padding=SAME
    {
        int wpad = kernel_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            copy_make_border(bottom_blob, bottom_blob_bordered, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, BORDER_CONSTANT, pad_value, opt.workspace_allocator, opt.num_threads);
            if (bottom_blob_bordered.empty())
                return -100;
        }

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;
    }

    int outw = (w - kernel_w) / stride_w + 1;
    int outh = (h - kernel_h) / stride_h + 1;

    top_blob.create(outw, outh, channels, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w - kernel_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2++;
            }
            p2 += gap;
        }
    }

    if (pooling_type == PoolMethod_MAX)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            const Mat m = bottom_blob_bordered.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    const float* sptr = m.row(i*stride_h) + j*stride_w;

                    float max = sptr[0];

                    for (int k = 0; k < maxk; k++)
                    {
                        float val = sptr[ space_ofs[k] ];
                        max = std::max(max, val);
                    }

                    outptr[j] = max;
                }

                outptr += outw;
            }
        }
    }
    else if (pooling_type == PoolMethod_AVE)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            const Mat m = bottom_blob_bordered.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    const float* sptr = m.row(i*stride_h) + j*stride_w;

                    float sum = 0;

                    for (int k = 0; k < maxk; k++)
                    {
                        float val = sptr[ space_ofs[k] ];
                        sum += val;
                    }

                    outptr[j] = sum / maxk;
                }

                outptr += outw;
            }

            // fix pad
            if (pad_top != 0)
            {
                const float scale = (float)kernel_h / (kernel_h - pad_top);

                outptr = top_blob.channel(q).row(0);
                for (int i = 0; i < outw; i++)
                {
                    outptr[i] *= scale;
                }
            }
            if (pad_bottom + htailpad != 0)
            {
                const float scale = (float)kernel_h / (kernel_h - pad_bottom - htailpad);

                outptr = top_blob.channel(q).row(outh - 1);
                for (int i = 0; i < outw; i++)
                {
                    outptr[i] *= scale;
                }
            }
            if (pad_left != 0)
            {
                const float scale = (float)kernel_w / (kernel_w - pad_left);

                outptr = top_blob.channel(q);
                for (int i = 0; i < outh; i++)
                {
                    *outptr *= scale;
                    outptr += outw;
                }
            }
            if (pad_right + wtailpad != 0)
            {
                const float scale = (float)kernel_w / (kernel_w - pad_right - wtailpad);

                outptr = top_blob.channel(q);
                outptr += outw - 1;
                for (int i = 0; i < outh; i++)
                {
                    *outptr *= scale;
                    outptr += outw;
                }
            }
        }
    }

    return 0;
}

#if NCNN_VULKAN
int Pooling::create_pipeline()
{
    pipeline_pooling = new Pipeline(vkdev);
    pipeline_pooling->set_optimal_local_size_xyz();

    std::vector<vk_specialization_type> specializations(7);
    specializations[0].i = pooling_type;
    specializations[1].i = kernel_w;
    specializations[2].i = kernel_h;
    specializations[3].i = stride_w;
    specializations[4].i = stride_h;
    specializations[5].i = global_pooling;
    specializations[6].i = pad_mode;

    pipeline_pooling->create("pooling", specializations, 2, 10);

    padding->create_pipeline();

    // pack4
    {
        pipeline_pooling_pack4 = new Pipeline(vkdev);
        pipeline_pooling_pack4->set_optimal_local_size_xyz();
        pipeline_pooling_pack4->create("pooling_pack4", specializations, 2, 10);
    }

    if (global_pooling)
    {
        pipeline_pooling_global = new Pipeline(vkdev);

        pipeline_pooling_global->set_optimal_local_size_xyz(256, 1, 1);

        std::vector<vk_specialization_type> specializations(1);
        specializations[0].i = pooling_type;

        pipeline_pooling_global->create("pooling_global", specializations, 2, 10);

        // pack4
        {
            pipeline_pooling_global_pack4 = new Pipeline(vkdev);
            pipeline_pooling_global_pack4->set_optimal_local_size_xyz(256, 1, 1);
            pipeline_pooling_global_pack4->create("pooling_global_pack4", specializations, 2, 10);
        }
    }

    return 0;
}

int Pooling::destroy_pipeline()
{
    if (padding)
        padding->destroy_pipeline();

    delete pipeline_pooling;
    pipeline_pooling = 0;

    delete pipeline_pooling_pack4;
    pipeline_pooling_pack4 = 0;

    delete pipeline_pooling_global;
    pipeline_pooling_global = 0;

    delete pipeline_pooling_global_pack4;
    pipeline_pooling_global_pack4 = 0;

    return 0;
}

int Pooling::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int packing = bottom_blob.packing;

    if (global_pooling)
    {
        top_blob.create(channels, elemsize, packing, opt.blob_vkallocator, opt.staging_vkallocator);
        if (top_blob.empty())
            return -100;

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

        const Pipeline* pipeline = packing == 4 ? pipeline_pooling_global_pack4 : pipeline_pooling_global;

        // record
        cmd.record_pipeline(pipeline, bindings, constants, top_blob);

        return 0;
    }

    VkMat bottom_blob_bordered = bottom_blob;
    {
        ncnn::Option opt_pad = opt;
        opt_pad.blob_vkallocator = opt.workspace_vkallocator;

        padding->forward(bottom_blob, bottom_blob_bordered, cmd, opt_pad);

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;
    }

    if (pad_mode == 0) // full padding
    {
        int wtail = (w + pad_left + pad_right - kernel_w) % stride_w;
        int htail = (h + pad_top + pad_bottom - kernel_h) % stride_h;

        if (wtail != 0)
            w += stride_w - wtail;
        if (htail != 0)
            h += stride_h - htail;
    }

    // FIXME full pad and valid pad only
    int outw = (w - kernel_w) / stride_w + 1;
    int outh = (h - kernel_h) / stride_h + 1;

    top_blob.create(outw, outh, channels, elemsize, packing, opt.blob_vkallocator, opt.staging_vkallocator);
    if (top_blob.empty())
        return -100;

//     fprintf(stderr, "Pooling::forward %p %p\n", bottom_blob_bordered.buffer(), top_blob.buffer());

    std::vector<VkMat> bindings(2);
    bindings[0] = bottom_blob_bordered;
    bindings[1] = top_blob;

    std::vector<vk_constant_type> constants(10);
    constants[0].i = bottom_blob_bordered.dims;
    constants[1].i = bottom_blob_bordered.w;
    constants[2].i = bottom_blob_bordered.h;
    constants[3].i = bottom_blob_bordered.c;
    constants[4].i = bottom_blob_bordered.cstep;
    constants[5].i = top_blob.dims;
    constants[6].i = top_blob.w;
    constants[7].i = top_blob.h;
    constants[8].i = top_blob.c;
    constants[9].i = top_blob.cstep;

    const Pipeline* pipeline = packing == 4 ? pipeline_pooling_pack4 : pipeline_pooling;

    // record
    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    // TODO avgpool exclude padding

    return 0;
}
#endif // NCNN_VULKAN

} // namespace ncnn
