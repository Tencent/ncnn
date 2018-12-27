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

#include "relu.h"
#include <math.h>
#include <algorithm>

namespace ncnn {

DEFINE_LAYER_CREATOR(ReLU)

ReLU::ReLU()
{
    one_blob_only = true;
    support_inplace = true;
    support_vulkan = true;
}

int ReLU::load_param(const ParamDict& pd)
{
    slope = pd.get(0, 0.f);

#if NCNN_VULKAN

    local_size_z = std::min(128, pd.max_workgroup_size[2]);

    int local_size_xy = sqrt(pd.max_workgroup_invocations / local_size_z);
    int local_size_xy_prefer = 256;
    while (local_size_xy < local_size_xy_prefer)
    {
        local_size_xy_prefer /= 2;
    }
    local_size_x = local_size_xy_prefer;
    local_size_y = local_size_xy_prefer;

    // setup pipeline specializations
    specializations.resize(0);

    binding_count = 1;
    push_constant_count = 5;

#endif // NCNN_VULKAN

    return 0;
}

int ReLU::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

    if (slope == 0.f)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i=0; i<size; i++)
            {
                if (ptr[i] < 0)
                    ptr[i] = 0;
            }
        }
    }
    else
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

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
int ReLU::forward_inplace(VkMat& bottom_top_blob, Command& cmd, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;

    fprintf(stderr, "ReLU::forward_inplace %p\n", bottom_top_blob.buffer);

    std::vector<VkMat> bindings(1);
    bindings[0] = bottom_top_blob;

    std::vector<int> constants(5);
    constants[0] = bottom_top_blob.dims;
    constants[1] = bottom_top_blob.w;
    constants[2] = bottom_top_blob.h;
    constants[3] = bottom_top_blob.c;
    constants[4] = bottom_top_blob.cstep;

    uint32_t group_count_xyz[3];
    group_count_xyz[0] = (bottom_top_blob.w + local_size_x - 1) / local_size_x;
    group_count_xyz[1] = (bottom_top_blob.h + local_size_y - 1) / local_size_y;
    group_count_xyz[2] = (bottom_top_blob.c + local_size_z - 1) / local_size_z;

    // record
    cmd.record_bind_pipeline(pipeline);
    cmd.record_update_bindings(pipeline_layout, descriptor_update_template, bindings);
    cmd.record_push_constants(pipeline_layout, constants);
    cmd.record_dispatch(group_count_xyz);

    return 0;
}
#endif // NCNN_VULKAN

} // namespace ncnn
