// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SDPA_VULKAN_H
#define LAYER_SDPA_VULKAN_H

#include "sdpa.h"

namespace ncnn {

class SDPA_vulkan : public SDPA
{
public:
    SDPA_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);
    virtual int load_param(const ParamDict& pd);

    using SDPA::forward;
    virtual int forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_sdpa;
    Pipeline* pipeline_sdpa_kv_concat;
};

} // namespace ncnn

#endif // LAYER_SDPA_VULKAN_H