// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SDPA_VULKAN_H
#define LAYER_SDPA_VULKAN_H

#include "sdpa.h"

namespace ncnn {

class SDPA_vulkan : public SDPA
{
public:
    SDPA_vulkan();

    virtual int load_param(const ParamDict& pd);

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using SDPA::forward;
    virtual int forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const;

public:
    Layer* qk_softmax;

    Layer* kvcache_concat;

    Pipeline* pipeline_sdpa_qk_cross;

    Pipeline* pipeline_sdpa_qkv_cross;
};

} // namespace ncnn

#endif // LAYER_SDPA_VULKAN_H
