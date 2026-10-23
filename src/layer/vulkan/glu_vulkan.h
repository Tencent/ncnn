// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_GLU_VULKAN_H
#define LAYER_GLU_VULKAN_H

#include "glu.h"

namespace ncnn {

class GLU_vulkan : public GLU
{
public:
    GLU_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using GLU::forward;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_glu;
    Pipeline* pipeline_glu_pack4;
};

} // namespace ncnn

#endif // LAYER_GLU_VULKAN_H
