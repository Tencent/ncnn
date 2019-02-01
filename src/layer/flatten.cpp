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

#include "flatten.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Flatten)

Flatten::Flatten()
{
    one_blob_only = true;
    support_inplace = false;
    support_vulkan = true;

#if NCNN_VULKAN
    pipeline_flatten_pack4 = 0;
#endif // NCNN_VULKAN
}

int Flatten::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int size = w * h;

    top_blob.create(size * channels, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q=0; q<channels; q++)
    {
        const float* ptr = bottom_blob.channel(q);
        float* outptr = (float*)top_blob + size * q;

        for (int i=0; i<size; i++)
        {
            outptr[i] = ptr[i];
        }
    }

    return 0;
}

#if NCNN_VULKAN
int Flatten::create_pipeline()
{
    std::vector<vk_specialization_type> specializations;

    // pack4
    {
        pipeline_flatten_pack4 = new Pipeline(vkdev);
        pipeline_flatten_pack4->set_optimal_local_size_xyz();
        pipeline_flatten_pack4->create("flatten_pack4", specializations, 2, 10);
    }

    return 0;
}

int Flatten::destroy_pipeline()
{
    delete pipeline_flatten_pack4;
    pipeline_flatten_pack4 = 0;

    return 0;
}

int Flatten::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int dims = bottom_blob.dims;

    if (dims == 1 || dims == 2)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int packing = bottom_blob.packing;

    top_blob.create(w * h * channels, elemsize, packing, opt.blob_vkallocator, opt.staging_vkallocator);
    if (top_blob.empty())
        return -100;

    if (packing == 4)
    {
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
        cmd.record_prepare_compute_barrier(bottom_blob);
        cmd.record_prepare_compute_barrier(top_blob);
        cmd.record_pipeline(pipeline_flatten_pack4, bindings, constants, top_blob);

        return 0;
    }

    std::vector<VkBufferCopy> regions(channels);

    int srcOffset = 0;
    int dstOffset = 0;
    for (int q=0; q<channels; q++)
    {
        int size = w * h * elemsize;

        regions[q].srcOffset = bottom_blob.buffer_offset() + srcOffset;
        regions[q].dstOffset = top_blob.buffer_offset() + dstOffset;
        regions[q].size = size;

        srcOffset += bottom_blob.cstep * elemsize;
        dstOffset += size;
    }

    cmd.record_prepare_transfer_barrier(bottom_blob);
    cmd.record_prepare_transfer_barrier(top_blob);
    cmd.record_copy_regions(bottom_blob, top_blob, regions);

    return 0;
}
#endif // NCNN_VULKAN

} // namespace ncnn
