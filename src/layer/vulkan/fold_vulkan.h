// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_FOLD_VULKAN_H
#define LAYER_FOLD_VULKAN_H

#include "fold.h"

namespace ncnn {

class Fold_vulkan : public Fold
{
public:
    Fold_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using Fold::forward;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_fold_col2im;
    Pipeline* pipeline_fold_col2im_pack4;
    Pipeline* pipeline_fold_col2im_pack1to4;
    Pipeline* pipeline_fold_col2im_pack4to1;
};

} // namespace ncnn

#endif // LAYER_FOLD_VULKAN_H
