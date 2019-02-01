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

#include "padding.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Padding)

Padding::Padding()
{
    one_blob_only = true;
    support_inplace = false;
    support_vulkan = true;

#if NCNN_VULKAN
    pipeline_padding = 0;
    pipeline_padding_pack4 = 0;
#endif // NCNN_VULKAN
}

int Padding::load_param(const ParamDict& pd)
{
    top = pd.get(0, 0);
    bottom = pd.get(1, 0);
    left = pd.get(2, 0);
    right = pd.get(3, 0);
    type = pd.get(4, 0);
    value = pd.get(5, 0.f);

    return 0;
}

template<typename T>
static void copy_make_border_image(const Mat& src, Mat& dst, int top, int left, int type, T v)
{
    int w = dst.w;
    int h = dst.h;

    const T* ptr = src;
    T* outptr = dst;

    if (type == 0)
    {
        int y = 0;
        // fill top
        for (; y < top; y++)
        {
            int x = 0;
            for (; x < w; x++)
            {
                outptr[x] = v;
            }
            outptr += w;
        }
        // fill center
        for (; y < (top + src.h); y++)
        {
            int x = 0;
            for (; x < left; x++)
            {
                outptr[x] = v;
            }
            if (src.w < 12)
            {
                for (; x < (left + src.w); x++)
                {
                    outptr[x] = ptr[x - left];
                }
            }
            else
            {
                memcpy(outptr + left, ptr, src.w * sizeof(T));
                x += src.w;
            }
            for (; x < w; x++)
            {
                outptr[x] = v;
            }
            ptr += src.w;
            outptr += w;
        }
        // fill bottom
        for (; y < h; y++)
        {
            int x = 0;
            for (; x < w; x++)
            {
                outptr[x] = v;
            }
            outptr += w;
        }
    }
    else if (type == 1)
    {
        int y = 0;
        // fill top
        for (; y < top; y++)
        {
            int x = 0;
            for (; x < left; x++)
            {
                outptr[x] = ptr[0];
            }
            if(src.w < 12)
            {
                for (; x < (left + src.w); x++)
                {
                    outptr[x] = ptr[x - left];
                }
            }
            else
            {
                memcpy(outptr + left, ptr, src.w * sizeof(T));
                x += src.w;
            }
            for (; x < w; x++)
            {
                outptr[x] = ptr[src.w - 1];
            }
            outptr += w;
        }
        // fill center
        for (; y < (top + src.h); y++)
        {
            int x = 0;
            for (; x < left; x++)
            {
                outptr[x] = ptr[0];
            }
            if(src.w < 12)
            {
                for (; x < (left + src.w); x++)
                {
                    outptr[x] = ptr[x - left];
                }
            }
            else
            {
                memcpy(outptr + left, ptr, src.w * sizeof(T));
                x += src.w;
            }
            for (; x < w; x++)
            {
                outptr[x] = ptr[src.w - 1];
            }
            ptr += src.w;
            outptr += w;
        }
        // fill bottom
        ptr -= src.w;
        for (; y < h; y++)
        {
            int x = 0;
            for (; x < left; x++)
            {
                outptr[x] = ptr[0];
            }
            if(src.w < 12)
            {
                for (; x < (left + src.w); x++)
                {
                    outptr[x] = ptr[x - left];
                }
            }
            else
            {
                memcpy(outptr + left, ptr, src.w * sizeof(T));
                x += src.w;
            }
            for (; x < w; x++)
            {
                outptr[x] = ptr[src.w - 1];
            }
            outptr += w;
        }
    }
}

int Padding::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (top == 0 && bottom == 0 && left == 0 && right == 0)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;

    int outw = w + left + right;

    if (dims == 1)
    {
        top_blob.create(outw, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (elemsize == 1)
            copy_make_border_image<signed char>(bottom_blob, top_blob, 0, left, type, value);
        else if (elemsize == 4)
            copy_make_border_image<float>(bottom_blob, top_blob, 0, left, type, value);

        return 0;
    }

    int outh = h + top + bottom;

    if (dims == 2)
    {
        top_blob.create(outw, outh, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (elemsize == 1)
            copy_make_border_image<signed char>(bottom_blob, top_blob, top, left, type, value);
        else if (elemsize == 4)
            copy_make_border_image<float>(bottom_blob, top_blob, top, left, type, value);

        return 0;
    }

    if (dims == 3)
    {
        top_blob.create(outw, outh, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            const Mat m = bottom_blob.channel(q);
            Mat borderm = top_blob.channel(q);

            if (elemsize == 1)
                copy_make_border_image<signed char>(m, borderm, top, left, type, value);
            else if (elemsize == 4)
                copy_make_border_image<float>(m, borderm, top, left, type, value);
        }

        return 0;
    }

    return 0;
}

#if NCNN_VULKAN
int Padding::create_pipeline()
{
    pipeline_padding = new Pipeline(vkdev);
    pipeline_padding->set_optimal_local_size_xyz();

    std::vector<vk_specialization_type> specializations(6);
    specializations[0].i = top;
    specializations[1].i = bottom;
    specializations[2].i = left;
    specializations[3].i = right;
    specializations[4].i = type;
    specializations[5].f = value;

    pipeline_padding->create("padding", specializations, 2, 10);

    // pack4
    {
        pipeline_padding_pack4 = new Pipeline(vkdev);
        pipeline_padding_pack4->set_optimal_local_size_xyz();
        pipeline_padding_pack4->create("padding_pack4", specializations, 2, 10);
    }

    return 0;
}

int Padding::destroy_pipeline()
{
    delete pipeline_padding;
    pipeline_padding = 0;

    delete pipeline_padding_pack4;
    pipeline_padding_pack4 = 0;

    return 0;
}

int Padding::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int packing = bottom_blob.packing;

    // TODO vec and image padding
    int dims = bottom_blob.dims;

    int outw = w + left + right;
    int outh = h + top + bottom;

    top_blob.create(outw, outh, channels, elemsize, packing, opt.blob_vkallocator, opt.staging_vkallocator);
    if (top_blob.empty())
        return -100;

//     fprintf(stderr, "Padding::forward %p %p\n", bottom_blob.buffer(), top_blob.buffer());

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

    const Pipeline* pipeline = packing == 4 ? pipeline_padding_pack4 : pipeline_padding;

    // record
    cmd.record_prepare_compute_barrier(bottom_blob);
    cmd.record_prepare_compute_barrier(top_blob);
    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    return 0;
}
#endif // NCNN_VULKAN

} // namespace ncnn
