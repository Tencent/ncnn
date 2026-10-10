// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_GELU_VULKAN_H
#define LAYER_GELU_VULKAN_H

#include "gelu.h"

namespace ncnn {

class GELU_vulkan : public GELU
{
public:
    GELU_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using GELU::forward_inplace;
    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_gelu;
};

} // namespace ncnn

#endif // LAYER_GELU_VULKAN_H
