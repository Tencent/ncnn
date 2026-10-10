// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_ELU_VULKAN_H
#define LAYER_ELU_VULKAN_H

#include "elu.h"

namespace ncnn {

class ELU_vulkan : public ELU
{
public:
    ELU_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using ELU::forward_inplace;
    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_elu;
};

} // namespace ncnn

#endif // LAYER_ELU_VULKAN_H
