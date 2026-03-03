// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CROP_VULKAN_H
#define LAYER_CROP_VULKAN_H

#include "crop.h"

namespace ncnn {

class Crop_vulkan : public Crop
{
public:
    Crop_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using Crop::forward;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;

    virtual int forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_crop;
    Pipeline* pipeline_crop_pack4;
    Pipeline* pipeline_crop_pack1to4;
    Pipeline* pipeline_crop_pack4to1;
};

} // namespace ncnn

#endif // LAYER_CROP_VULKAN_H
