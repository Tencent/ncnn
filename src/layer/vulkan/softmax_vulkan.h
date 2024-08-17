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

#ifndef LAYER_SOFTMAX_VULKAN_H
#define LAYER_SOFTMAX_VULKAN_H

#include "softmax.h"

namespace ncnn {

class Softmax_vulkan : public Softmax
{
public:
    Softmax_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using Softmax::forward_inplace;
    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;
    virtual int forward_inplace(VkImageMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_softmax_reduce_max;
    Pipeline* pipeline_softmax_exp_sub_max;
    Pipeline* pipeline_softmax_reduce_sum;
    Pipeline* pipeline_softmax_div_sum;

    Pipeline* pipeline_softmax_reduce_max_pack4;
    Pipeline* pipeline_softmax_exp_sub_max_pack4;
    Pipeline* pipeline_softmax_reduce_sum_pack4;
    Pipeline* pipeline_softmax_div_sum_pack4;

    Pipeline* pipeline_softmax_reduce_max_pack8;
    Pipeline* pipeline_softmax_exp_sub_max_pack8;
    Pipeline* pipeline_softmax_reduce_sum_pack8;
    Pipeline* pipeline_softmax_div_sum_pack8;
};

} // namespace ncnn

#endif // LAYER_SOFTMAX_VULKAN_H
