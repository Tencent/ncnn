// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_INNERPRODUCT_VULKAN_H
#define LAYER_INNERPRODUCT_VULKAN_H

#include "innerproduct.h"

namespace ncnn {

class InnerProduct_vulkan : public InnerProduct
{
public:
    InnerProduct_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int upload_model(VkTransfer& cmd, const Option& opt);

    using InnerProduct::forward;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;

public:
    ncnn::Layer* flatten;

    Mat weight_data_packed;

    VkMat weight_data_gpu;
    VkMat bias_data_gpu;

    Pipeline* pipeline_innerproduct;

    Pipeline* pipeline_innerproduct_sum8;
    Pipeline* pipeline_innerproduct_reduce_sum8;

    Pipeline* pipeline_innerproduct_gemm;
};

} // namespace ncnn

#endif // LAYER_INNERPRODUCT_VULKAN_H
