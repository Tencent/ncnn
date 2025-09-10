// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_PERMUTE_VULKAN_H
#define LAYER_PERMUTE_VULKAN_H

#include "permute.h"

namespace ncnn {

class Permute_vulkan : public Permute
{
public:
    Permute_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using Permute::forward;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_permute;
    Pipeline* pipeline_permute_pack4;
    Pipeline* pipeline_permute_pack1to4;
    Pipeline* pipeline_permute_pack4to1;
};

} // namespace ncnn

#endif // LAYER_PERMUTE_VULKAN_H
