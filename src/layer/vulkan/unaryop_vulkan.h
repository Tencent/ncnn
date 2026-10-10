// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_UNARYOP_VULKAN_H
#define LAYER_UNARYOP_VULKAN_H

#include "unaryop.h"

namespace ncnn {

class UnaryOp_vulkan : public UnaryOp
{
public:
    UnaryOp_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using UnaryOp::forward_inplace;
    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_unaryop;
};

} // namespace ncnn

#endif // LAYER_UNARYOP_VULKAN_H
