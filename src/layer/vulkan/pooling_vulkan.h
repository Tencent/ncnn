// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_POOLING_VULKAN_H
#define LAYER_POOLING_VULKAN_H

#include "pooling.h"

namespace ncnn {

class Pooling_vulkan : public Pooling
{
public:
    Pooling_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using Pooling::forward;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_pooling;
    Pipeline* pipeline_pooling_tile;

    Pipeline* pipeline_pooling_global;
    Pipeline* pipeline_pooling_global_stage1;
    Pipeline* pipeline_pooling_global_stage2;

    Pipeline* pipeline_pooling_adaptive;
};

} // namespace ncnn

#endif // LAYER_POOLING_VULKAN_H
