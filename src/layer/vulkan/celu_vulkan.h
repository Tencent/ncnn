// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CELU_VULKAN_H
#define LAYER_CELU_VULKAN_H

#include "celu.h"

namespace ncnn {

class CELU_vulkan : public CELU
{
public:
    CELU_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using CELU::forward_inplace;
    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_celu;
};

} // namespace ncnn

#endif // LAYER_CELU_VULKAN_H
