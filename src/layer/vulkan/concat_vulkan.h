// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CONCAT_VULKAN_H
#define LAYER_CONCAT_VULKAN_H

#include "concat.h"

namespace ncnn {

class Concat_vulkan : public Concat
{
public:
    Concat_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using Concat::forward;
    virtual int forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_concat[2];
    Pipeline* pipeline_concat_pack4[2];
    Pipeline* pipeline_concat_pack4to1[2];
};

} // namespace ncnn

#endif // LAYER_CONCAT_VULKAN_H
