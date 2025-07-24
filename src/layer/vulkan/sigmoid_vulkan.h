// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SIGMOID_VULKAN_H
#define LAYER_SIGMOID_VULKAN_H

#include "sigmoid.h"

namespace ncnn {

class Sigmoid_vulkan : public Sigmoid
{
public:
    Sigmoid_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using Sigmoid::forward_inplace;
    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_sigmoid;
};

} // namespace ncnn

#endif // LAYER_SIGMOID_VULKAN_H
