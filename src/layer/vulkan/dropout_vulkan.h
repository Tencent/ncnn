// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_DROPOUT_VULKAN_H
#define LAYER_DROPOUT_VULKAN_H

#include "dropout.h"

namespace ncnn {

class Dropout_vulkan : public Dropout
{
public:
    Dropout_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using Dropout::forward_inplace;
    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_dropout;
};

} // namespace ncnn

#endif // LAYER_DROPOUT_VULKAN_H
