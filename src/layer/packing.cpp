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

#include "packing.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Packing)

Packing::Packing()
{
    one_blob_only = true;
    support_inplace = false;
    support_vulkan = false;

#if NCNN_VULKAN
    pipeline_packing_1to4 = 0;
    pipeline_packing_4to1 = 0;
#endif // NCNN_VULKAN
}

int Packing::load_param(const ParamDict& pd)
{
    out_packing = pd.get(0, 1);
    use_padding = pd.get(1, 0);

    return 0;
}

int Packing::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int packing = bottom_blob.packing;

    if (packing == out_packing)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;

    if (!use_padding)
    {
        // identity if use_padding not allowed
        if (dims == 1 && w * packing % out_packing != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
        if (dims == 2 && h * packing % out_packing != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
        if (dims == 3 && channels * packing % out_packing != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
    }

    if (dims == 1)
    {
        if (out_packing == 1)
        {
            top_blob = bottom_blob;
            top_blob.w = w * packing;
            top_blob.cstep = w * packing;
            top_blob.elemsize = elemsize / packing;
            top_blob.packing = out_packing;
            return 0;
        }

        int outw = (w * packing + out_packing - 1) / out_packing;
        size_t out_elemsize = elemsize / packing * out_packing;

        top_blob.create(outw, out_elemsize, out_packing, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        memcpy(top_blob.data, bottom_blob.data, w * elemsize);

        return 0;
    }

    if (dims == 2)
    {
        int outh = (h * packing + out_packing - 1) / out_packing;
        size_t out_elemsize = elemsize / packing * out_packing;
        size_t lane_size = out_elemsize / out_packing;

        top_blob.create(w, outh, out_elemsize, out_packing, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for
        for (int i = 0; i < outh; i++)
        {
            unsigned char* outptr = (unsigned char*)top_blob + i * w * out_elemsize;

            for (int j = 0; j < w; j++)
            {
                unsigned char* out_elem_ptr = outptr + j * out_elemsize;

                for (int k = 0; k < out_packing; k++)
                {
                    int srcy = (i * out_packing + k) / packing;
                    if (srcy >= h)
                        break;

                    int srck = (i * out_packing + k) % packing;

                    const unsigned char* ptr = (const unsigned char*)bottom_blob + srcy * w * elemsize;
                    const unsigned char* elem_ptr = ptr + j * elemsize;

                    memcpy(out_elem_ptr + k * lane_size, elem_ptr + srck * lane_size, lane_size);
                }
            }
        }

        return 0;
    }

    if (dims == 3)
    {
        int outc = (channels * packing + out_packing - 1) / out_packing;
        size_t out_elemsize = elemsize / packing * out_packing;
        size_t lane_size = out_elemsize / out_packing;

        top_blob.create(w, h, outc, out_elemsize, out_packing, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for
        for (int q = 0; q < outc; q++)
        {
            Mat out = top_blob.channel(q);

            for (int i = 0; i < h; i++)
            {
                unsigned char* outptr = (unsigned char*)out + i * w * out_elemsize;

                for (int j = 0; j < w; j++)
                {
                    unsigned char* out_elem_ptr = outptr + j * out_elemsize;

                    for (int k = 0; k < out_packing; k++)
                    {
                        int srcq = (q * out_packing + k) / packing;
                        if (srcq >= channels)
                            break;

                        int srck = (q * out_packing + k) % packing;

                        const Mat m = bottom_blob.channel(srcq);
                        const unsigned char* ptr = (const unsigned char*)m + i * w * elemsize;
                        const unsigned char* elem_ptr = ptr + j * elemsize;

                        memcpy(out_elem_ptr + k * lane_size, elem_ptr + srck * lane_size, lane_size);
                    }
                }
            }
        }

        return 0;
    }

    return 0;
}

#if NCNN_VULKAN
int Packing::create_pipeline()
{
    std::vector<vk_specialization_type> specializations;

    if (out_packing == 4)
    {
        pipeline_packing_1to4 = new Pipeline(vkdev);
        pipeline_packing_1to4->set_optimal_local_size_xyz();
        pipeline_packing_1to4->create("packing_1to4", specializations, 2, 10);
    }

    if (out_packing == 1)
    {
        pipeline_packing_4to1 = new Pipeline(vkdev);
        pipeline_packing_4to1->set_optimal_local_size_xyz();
        pipeline_packing_4to1->create("packing_4to1", specializations, 2, 10);
    }

    return 0;
}

int Packing::destroy_pipeline()
{
    delete pipeline_packing_1to4;
    pipeline_packing_1to4 = 0;

    delete pipeline_packing_4to1;
    pipeline_packing_4to1 = 0;

    return 0;
}

int Packing::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int packing = bottom_blob.packing;

    if (packing == out_packing)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;

    if (!use_padding)
    {
        // identity if use_padding not allowed
        if (dims == 1 && w * packing % out_packing != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
        if (dims == 2 && h * packing % out_packing != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
        if (dims == 3 && channels * packing % out_packing != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
    }

    if (dims == 1)
    {
        if (out_packing == 1)
        {
            top_blob = bottom_blob;
            top_blob.w = w * packing;
            top_blob.cstep = w * packing;
            top_blob.elemsize = elemsize / packing;
            top_blob.packing = out_packing;
            return 0;
        }

        int outw = (w * packing + out_packing - 1) / out_packing;
        size_t out_elemsize = elemsize / packing * out_packing;

        top_blob.create(outw, out_elemsize, out_packing, opt.blob_vkallocator, opt.staging_vkallocator);
        if (top_blob.empty())
            return -100;
    }

    if (dims == 2)
    {
        int outh = (h * packing + out_packing - 1) / out_packing;
        size_t out_elemsize = elemsize / packing * out_packing;

        top_blob.create(w, outh, out_elemsize, out_packing, opt.blob_vkallocator, opt.staging_vkallocator);
        if (top_blob.empty())
            return -100;
    }

    if (dims == 3)
    {
        int outc = (channels * packing + out_packing - 1) / out_packing;
        size_t out_elemsize = elemsize / packing * out_packing;

        top_blob.create(w, h, outc, out_elemsize, out_packing, opt.blob_vkallocator, opt.staging_vkallocator);
        if (top_blob.empty())
            return -100;
    }

//     fprintf(stderr, "Packing::forward %p %p\n", bottom_blob.buffer(), top_blob.buffer());

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
    if (packing == 1 && out_packing == 4)
    {
        cmd.record_pipeline(pipeline_packing_1to4, bindings, constants, top_blob);
    }

    if (packing == 4 && out_packing == 1)
    {
        cmd.record_pipeline(pipeline_packing_4to1, bindings, constants, bottom_blob);
    }

    return 0;
}
#endif // NCNN_VULKAN

} // namespace ncnn
