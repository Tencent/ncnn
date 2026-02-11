// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_UNFOLD_VULKAN_H
#define LAYER_UNFOLD_VULKAN_H

#include "unfold.h"

namespace ncnn {

class Unfold_vulkan : public Unfold
{
public:
    Unfold_vulkan();

    virtual int load_param(const ParamDict& pd);

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using Unfold::forward;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;

protected:
    int make_padding(const VkMat& bottom_blob, VkMat& bottom_blob_bordered, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_unfold_im2col;
    Pipeline* pipeline_unfold_padding;
};

} // namespace ncnn

#endif // LAYER_UNFOLD_VULKAN_H
