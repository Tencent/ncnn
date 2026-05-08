// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_PRIORBOX_VULKAN_H
#define LAYER_PRIORBOX_VULKAN_H

#include "priorbox.h"

namespace ncnn {

class PriorBox_vulkan : public PriorBox
{
public:
    PriorBox_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int upload_model(VkTransfer& cmd, const Option& opt);

    using PriorBox::forward;
    virtual int forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const;

public:
    VkMat min_sizes_gpu;
    VkMat max_sizes_gpu;
    VkMat aspect_ratios_gpu;
    Pipeline* pipeline_priorbox;
    Pipeline* pipeline_priorbox_mxnet;
};

} // namespace ncnn

#endif // LAYER_PRIORBOX_VULKAN_H
