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

#include "clip.h"

#include <float.h>

namespace ncnn {

DEFINE_LAYER_CREATOR(Clip)

Clip::Clip()
{
    one_blob_only = true;
    support_inplace = true;
    support_vulkan = true;

#if NCNN_VULKAN
    pipeline_clip = 0;
    pipeline_clip_pack4 = 0;
#endif // NCNN_VULKAN
}

int Clip::load_param(const ParamDict& pd)
{
    min = pd.get(0, -FLT_MAX);
    max = pd.get(1, FLT_MAX);

    return 0;
}

int Clip::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q=0; q<channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        for (int i=0; i<size; i++)
        {
            if (ptr[i] < min)
                ptr[i] = min;
            if (ptr[i] > max)
                ptr[i] = max;
        }
    }

    return 0;
}

#if NCNN_VULKAN
int Clip::create_pipeline()
{
    std::vector<vk_specialization_type> specializations(2);
    specializations[0].f = min;
    specializations[1].f = max;

    // pack1
    {
        pipeline_clip = new Pipeline(vkdev);
        pipeline_clip->set_optimal_local_size_xyz();
        pipeline_clip->create("clip", specializations, 1, 5);
    }

    // pack4
    {
        pipeline_clip_pack4 = new Pipeline(vkdev);
        pipeline_clip_pack4->set_optimal_local_size_xyz();
        pipeline_clip_pack4->create("clip_pack4", specializations, 1, 5);
    }

    return 0;
}

int Clip::destroy_pipeline()
{
    delete pipeline_clip;
    pipeline_clip = 0;

    delete pipeline_clip_pack4;
    pipeline_clip_pack4 = 0;

    return 0;
}

int Clip::forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const
{
    int packing = bottom_top_blob.packing;

//     fprintf(stderr, "Clip::forward_inplace %p\n", bottom_top_blob.buffer());

    std::vector<VkMat> bindings(1);
    bindings[0] = bottom_top_blob;

    std::vector<vk_constant_type> constants(5);
    constants[0].i = bottom_top_blob.dims;
    constants[1].i = bottom_top_blob.w;
    constants[2].i = bottom_top_blob.h;
    constants[3].i = bottom_top_blob.c;
    constants[4].i = bottom_top_blob.cstep;

    const Pipeline* pipeline = packing == 4 ? pipeline_clip_pack4 : pipeline_clip;

    // record
    cmd.record_pipeline(pipeline, bindings, constants, bottom_top_blob);

    return 0;
}
#endif // NCNN_VULKAN

} // namespace ncnn
