// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_PACKING_VULKAN_H
#define LAYER_PACKING_VULKAN_H

#include "packing.h"

namespace ncnn {

class Packing_vulkan : public Packing
{
public:
    Packing_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using Packing::forward;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_packing;
    Pipeline* pipeline_packing_pack4;
    Pipeline* pipeline_packing_pack8;
    Pipeline* pipeline_packing_pack1to4;
    Pipeline* pipeline_packing_pack4to1;
    Pipeline* pipeline_packing_pack1to8;
    Pipeline* pipeline_packing_pack4to8;
    Pipeline* pipeline_packing_pack8to4;
    Pipeline* pipeline_packing_pack8to1;
};

} // namespace ncnn

#endif // LAYER_PACKING_VULKAN_H
