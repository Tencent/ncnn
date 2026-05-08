// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_ERF_VULKAN_H
#define LAYER_ERF_VULKAN_H

#include "erf.h"

namespace ncnn {

class Erf_vulkan : public Erf
{
public:
    Erf_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using Erf::forward_inplace;
    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_erf;
};

} // namespace ncnn

#endif // LAYER_ERF_VULKAN_H
