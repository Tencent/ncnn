// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_REORG_VULKAN_H
#define LAYER_REORG_VULKAN_H

#include "reorg.h"

namespace ncnn {

class Reorg_vulkan : public Reorg
{
public:
    Reorg_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using Reorg::forward;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_reorg;
    Pipeline* pipeline_reorg_pack4;
    Pipeline* pipeline_reorg_pack1to4;
};

} // namespace ncnn

#endif // LAYER_REORG_VULKAN_H
