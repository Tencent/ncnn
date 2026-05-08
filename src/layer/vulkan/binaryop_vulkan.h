// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_BINARYOP_VULKAN_H
#define LAYER_BINARYOP_VULKAN_H

#include "binaryop.h"

namespace ncnn {

class BinaryOp_vulkan : public BinaryOp
{
public:
    BinaryOp_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using BinaryOp::forward;
    using BinaryOp::forward_inplace;
    virtual int forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const;

    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_binaryop;
    Pipeline* pipeline_binaryop_pack4;

    // broadcast
    Pipeline* pipeline_binaryop_broadcast[2];
    Pipeline* pipeline_binaryop_broadcast_pack4[2];
    Pipeline* pipeline_binaryop_broadcast_pack1to4[2];
};

} // namespace ncnn

#endif // LAYER_BINARYOP_VULKAN_H
