// Copyright 2026 Futz12 <pchar.cn>
// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_ROTARYEMBED_VULKAN_H
#define LAYER_ROTARYEMBED_VULKAN_H

#include "rotaryembed.h"

namespace ncnn {

class RotaryEmbed_vulkan : public RotaryEmbed
{
public:
    RotaryEmbed_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using RotaryEmbed::forward;
    virtual int forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_rotaryembed;
    Pipeline* pipeline_rotaryembed_pack4;
};

} // namespace ncnn

#endif // LAYER_ROTARYEMBED_VULKAN_H
