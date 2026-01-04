// Copyright 2026 Futz12 <pchar.cn>
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
    Pipeline* pipeline_softmax;
    Pipeline* pipeline_softmax_pack4;
};

} // namespace ncnn

#endif // LAYER_SOFTMAX_VULKAN_H
