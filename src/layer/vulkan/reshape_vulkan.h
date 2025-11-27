// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_RESHAPE_VULKAN_H
#define LAYER_RESHAPE_VULKAN_H

#include "reshape.h"

namespace ncnn {

class Reshape_vulkan : public Reshape
{
public:
    Reshape_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using Reshape::forward;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;
    virtual int forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_reshape;
    Pipeline* pipeline_reshape_pack4;
    Pipeline* pipeline_reshape_pack1to4;
    Pipeline* pipeline_reshape_pack4to1;
};

} // namespace ncnn

#endif // LAYER_RESHAPE_VULKAN_H
