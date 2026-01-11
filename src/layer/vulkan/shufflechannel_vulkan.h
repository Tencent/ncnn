// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SHUFFLECHANNEL_VULKAN_H
#define LAYER_SHUFFLECHANNEL_VULKAN_H

#include "shufflechannel.h"

namespace ncnn {

class ShuffleChannel_vulkan : public ShuffleChannel
{
public:
    ShuffleChannel_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using ShuffleChannel::forward;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_shufflechannel;
    Pipeline* pipeline_shufflechannel_pack4;
};

} // namespace ncnn

#endif // LAYER_SHUFFLECHANNEL_VULKAN_H
