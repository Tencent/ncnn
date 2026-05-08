// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_ABSVAL_VULKAN_H
#define LAYER_ABSVAL_VULKAN_H

#include "absval.h"

namespace ncnn {

class AbsVal_vulkan : public AbsVal
{
public:
    AbsVal_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using AbsVal::forward_inplace;
    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_absval;
};

} // namespace ncnn

#endif // LAYER_ABSVAL_VULKAN_H
