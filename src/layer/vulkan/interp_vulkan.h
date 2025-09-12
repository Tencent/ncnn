// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_INTERP_VULKAN_H
#define LAYER_INTERP_VULKAN_H

#include "interp.h"

namespace ncnn {

class Interp_vulkan : public Interp
{
public:
    Interp_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using Interp::forward;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;
    virtual int forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_interp;
    Pipeline* pipeline_interp_pack4;

    Pipeline* pipeline_interp_bicubic_coeffs_x;
    Pipeline* pipeline_interp_bicubic_coeffs_y;
    Pipeline* pipeline_interp_bicubic;
    Pipeline* pipeline_interp_bicubic_pack4;
};

} // namespace ncnn

#endif // LAYER_INTERP_VULKAN_H
