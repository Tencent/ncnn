// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_PRELU_VULKAN_H
#define LAYER_PRELU_VULKAN_H

#include "prelu.h"

namespace ncnn {

class PReLU_vulkan : public PReLU
{
public:
    PReLU_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int upload_model(VkTransfer& cmd, const Option& opt);

    using PReLU::forward_inplace;
    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;

public:
    VkMat slope_data_gpu;

    Pipeline* pipeline_prelu;
    Pipeline* pipeline_prelu_pack4;
};

} // namespace ncnn

#endif // LAYER_PRELU_VULKAN_H
