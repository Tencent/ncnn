// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "reorg.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(Reorg)

Reorg::Reorg()
{
    one_blob_only = true;
    support_inplace = false;
    support_vulkan = true;

#if NCNN_VULKAN
    pipeline_reorg = 0;
    pipeline_reorg_pack4 = 0;
    pipeline_reorg_pack1to4 = 0;
#endif // NCNN_VULKAN
}

int Reorg::load_param(const ParamDict& pd)
{
    stride = pd.get(0, 0);

    return 0;
}

int Reorg::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    int outw = w / stride;
    int outh = h / stride;
    int outc = channels * stride * stride;

    top_blob.create(outw, outh, outc, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q=0; q<channels; q++)
    {
        const Mat m = bottom_blob.channel(q);

        for (int sh = 0; sh < stride; sh++)
        {
            for (int sw = 0; sw < stride; sw++)
            {
                float* outptr = top_blob.channel(q*stride*stride + sh*stride + sw);

                for (int i = 0; i < outh; i++)
                {
                    const float* sptr = m.row(i*stride + sh) + sw;
                    for (int j = 0; j < outw; j++)
                    {
                        outptr[0] = sptr[0];

                        sptr += stride;
                        outptr++;
                    }
                }
            }
        }
    }

    return 0;
}

#if NCNN_VULKAN
int Reorg::create_pipeline()
{
    std::vector<vk_specialization_type> specializations(1);
    specializations[0].i = stride;

    pipeline_reorg = new Pipeline(vkdev);
    pipeline_reorg->set_optimal_local_size_xyz();
    pipeline_reorg->create("reorg", specializations, 2, 10);

    // pack4
    {
        pipeline_reorg_pack4 = new Pipeline(vkdev);
        pipeline_reorg_pack4->set_optimal_local_size_xyz();
        pipeline_reorg_pack4->create("reorg_pack4", specializations, 2, 10);
    }

    // pack1to4
    {
        pipeline_reorg_pack1to4 = new Pipeline(vkdev);
        pipeline_reorg_pack1to4->set_optimal_local_size_xyz();
        pipeline_reorg_pack1to4->create("reorg_pack1to4", specializations, 2, 10);
    }

    return 0;
}

int Reorg::destroy_pipeline()
{
    delete pipeline_reorg;
    pipeline_reorg = 0;

    delete pipeline_reorg_pack4;
    pipeline_reorg_pack4 = 0;

    delete pipeline_reorg_pack1to4;
    pipeline_reorg_pack1to4 = 0;

    return 0;
}

int Reorg::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int packing = bottom_blob.packing;

    int outw = w / stride;
    int outh = h / stride;
    int outc = channels * packing * stride * stride;

    int out_packing = outc % 4 == 0 ? 4 : 1;
    size_t out_elemsize = elemsize / packing * out_packing;

    top_blob.create(outw, outh, outc / out_packing, out_elemsize, out_packing, opt.blob_vkallocator, opt.staging_vkallocator);
    if (top_blob.empty())
        return -100;

//     fprintf(stderr, "Reorg::forward %p %p\n", bottom_blob.buffer(), top_blob.buffer());

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

    const Pipeline* pipeline = 0;
    if (packing == 1 && out_packing == 1)
    {
        pipeline = pipeline_reorg;
    }
    else if (packing == 4) // assert out_packing == 4
    {
        pipeline = pipeline_reorg_pack4;
    }
    else if (packing == 1 && out_packing == 4)
    {
        pipeline = pipeline_reorg_pack1to4;
    }

    // record
    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    return 0;
}
#endif // NCNN_VULKAN

} // namespace ncnn
