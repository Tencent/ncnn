// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_LRN_VULKAN_H
#define LAYER_LRN_VULKAN_H

#include "lrn.h"

namespace ncnn {

class LRN_vulkan : public LRN
{
public:
    LRN_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using LRN::forward_inplace;
    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_lrn_square_pad;
    Pipeline* pipeline_lrn_norm;
    Pipeline* pipeline_lrn_square_pad_across_channel_pack4;
    Pipeline* pipeline_lrn_norm_across_channel_pack4;
    Pipeline* pipeline_lrn_square_pad_within_channel_pack4;
    Pipeline* pipeline_lrn_norm_within_channel_pack4;
};

} // namespace ncnn

#endif // LAYER_LRN_VULKAN_H
