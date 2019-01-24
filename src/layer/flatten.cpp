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

    top_blob.create(w * h * channels, elemsize, opt.blob_vkallocator, opt.staging_vkallocator);
    if (top_blob.empty())
        return -100;

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
