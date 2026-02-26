// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SLICE_VULKAN_H
#define LAYER_SLICE_VULKAN_H

#include "slice.h"

namespace ncnn {

class Slice_vulkan : public Slice
{
public:
    Slice_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using Slice::forward;
    virtual int forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_slice[2];
    Pipeline* pipeline_slice_pack4[2];
    Pipeline* pipeline_slice_pack1to4[2];
};

} // namespace ncnn

#endif // LAYER_SLICE_VULKAN_H
