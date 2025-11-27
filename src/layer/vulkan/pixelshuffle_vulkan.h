// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_PIXELSHUFFLE_VULKAN_H
#define LAYER_PIXELSHUFFLE_VULKAN_H

#include "pixelshuffle.h"

namespace ncnn {

class PixelShuffle_vulkan : public PixelShuffle
{
public:
    PixelShuffle_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using PixelShuffle::forward;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_pixelshuffle;
    Pipeline* pipeline_pixelshuffle_pack4;
    Pipeline* pipeline_pixelshuffle_pack4to1;
};

} // namespace ncnn

#endif // LAYER_PIXELSHUFFLE_VULKAN_H
