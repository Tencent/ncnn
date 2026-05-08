// Copyright 2025 <pchar.cn> futz12
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SELU_VULKAN_H
#define LAYER_SELU_VULKAN_H

#include "selu.h"

namespace ncnn {

class SELU_vulkan : public SELU
{
public:
    SELU_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using SELU::forward_inplace;
    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_selu;
};

} // namespace ncnn

#endif // LAYER_SELU_VULKAN_H
