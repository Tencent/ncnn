// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_MEMORYDATA_VULKAN_H
#define LAYER_MEMORYDATA_VULKAN_H

#include "memorydata.h"

namespace ncnn {

class MemoryData_vulkan : public MemoryData
{
public:
    MemoryData_vulkan();

    virtual int create_pipeline(const Option& opt);

    virtual int upload_model(VkTransfer& cmd, const Option& opt);

    using MemoryData::forward;
    virtual int forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const;

public:
    VkMat data_gpu;
};

} // namespace ncnn

#endif // LAYER_MEMORYDATA_VULKAN_H
