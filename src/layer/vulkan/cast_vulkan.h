// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CAST_VULKAN_H
#define LAYER_CAST_VULKAN_H

#include "cast.h"

namespace ncnn {

class Cast_vulkan : public Cast
{
public:
    Cast_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using Cast::forward;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_cast;
    Pipeline* pipeline_cast_pack4;
};

} // namespace ncnn

#endif // LAYER_CAST_VULKAN_H
