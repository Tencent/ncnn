// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef NCNN_GRIDSAMPLE_VULKAN_H
#define NCNN_GRIDSAMPLE_VULKAN_H

#include "gridsample.h"

namespace ncnn {

class GridSample_vulkan : public GridSample
{
public:
    GridSample_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_gridsample;
};

} // namespace ncnn

#endif // NCNN_GRIDSAMPLE_VULKAN_H
