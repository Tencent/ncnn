// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SCALE_VULKAN_H
#define LAYER_SCALE_VULKAN_H

#include "scale.h"

namespace ncnn {

class Scale_vulkan : public Scale
{
public:
    Scale_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int upload_model(VkTransfer& cmd, const Option& opt);

    using Scale::forward_inplace;
    virtual int forward_inplace(std::vector<VkMat>& bottom_top_blobs, VkCompute& cmd, const Option& opt) const;
    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;

public:
    VkMat scale_data_gpu;
    VkMat bias_data_gpu;

    Pipeline* pipeline_scale;
    Pipeline* pipeline_scale_pack4;
};

} // namespace ncnn

#endif // LAYER_SCALE_VULKAN_H
