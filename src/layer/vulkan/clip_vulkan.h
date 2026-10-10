// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CLIP_VULKAN_H
#define LAYER_CLIP_VULKAN_H

#include "clip.h"

namespace ncnn {

class Clip_vulkan : public Clip
{
public:
    Clip_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using Clip::forward_inplace;
    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_clip;
};

} // namespace ncnn

#endif // LAYER_CLIP_VULKAN_H
