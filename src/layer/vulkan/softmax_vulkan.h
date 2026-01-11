// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

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

public:
    Pipeline* pipeline_softmax_reduce_max;
    Pipeline* pipeline_softmax_exp_sub_max;
    Pipeline* pipeline_softmax_reduce_sum;
    Pipeline* pipeline_softmax_div_sum;

    Pipeline* pipeline_softmax_reduce_max_pack4;
    Pipeline* pipeline_softmax_exp_sub_max_pack4;
    Pipeline* pipeline_softmax_reduce_sum_pack4;
    Pipeline* pipeline_softmax_div_sum_pack4;
};

} // namespace ncnn

#endif // LAYER_SOFTMAX_VULKAN_H
