// Copyright 2019 Tencent
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

    virtual int upload_model(VkTransfer& cmd, const Option& opt);

    using Pooling::forward;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;

public:
    ncnn::Layer* padding;

    Pipeline* pipeline_pooling;
    Pipeline* pipeline_pooling_pack4;

    Pipeline* pipeline_pooling_adaptive;
    Pipeline* pipeline_pooling_adaptive_pack4;

    Pipeline* pipeline_pooling_global_reduce_first;
    Pipeline* pipeline_pooling_global_reduce_first_pack4;
    Pipeline* pipeline_pooling_global_reduce;
    Pipeline* pipeline_pooling_global_reduce_pack4;
    Pipeline* pipeline_pooling_global_reduce_last;
    Pipeline* pipeline_pooling_global_reduce_last_pack4;
};

} // namespace ncnn

#endif // LAYER_POOLING_VULKAN_H
