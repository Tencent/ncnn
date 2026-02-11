// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SHRINK_VULKAN_H
#define LAYER_SHRINK_VULKAN_H

#include "shrink.h"

namespace ncnn {

class Shrink_vulkan : public Shrink
{
public:
    Shrink_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using Shrink::forward_inplace;
    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_shrink;
};

} // namespace ncnn

#endif // LAYER_SHRINK_VULKAN_H
