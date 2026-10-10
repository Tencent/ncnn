// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_BATCHNORM_VULKAN_H
#define LAYER_BATCHNORM_VULKAN_H

#include "batchnorm.h"

namespace ncnn {

class BatchNorm_vulkan : public BatchNorm
{
public:
    BatchNorm_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int upload_model(VkTransfer& cmd, const Option& opt);

    using BatchNorm::forward_inplace;
    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;

public:
    VkMat a_data_gpu;
    VkMat b_data_gpu;

    Pipeline* pipeline_batchnorm;
    Pipeline* pipeline_batchnorm_pack4;
};

} // namespace ncnn

#endif // LAYER_BATCHNORM_VULKAN_H
