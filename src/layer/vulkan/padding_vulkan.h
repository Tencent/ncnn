// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_PADDING_VULKAN_H
#define LAYER_PADDING_VULKAN_H

#include "padding.h"

namespace ncnn {

class Padding_vulkan : public Padding
{
public:
    Padding_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int upload_model(VkTransfer& cmd, const Option& opt);

    using Padding::forward;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;

    virtual int forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const;

public:
    VkMat per_channel_pad_data_gpu;

    Pipeline* pipeline_padding;
    Pipeline* pipeline_padding_pack4;
    Pipeline* pipeline_padding_pack1to4;
    Pipeline* pipeline_padding_pack4to1;

    Pipeline* pipeline_padding_3d;
    Pipeline* pipeline_padding_3d_pack4;
};

} // namespace ncnn

#endif // LAYER_PADDING_VULKAN_H
