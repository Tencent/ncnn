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

protected:
#if NCNN_INT8
    int create_pipeline_int8(const Option& opt);
    int upload_model_int8(VkTransfer& cmd, const Option& opt);
    int forward_int8(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;
#endif

public:
    ncnn::Layer* flatten;

    Mat weight_data_packed;

    VkMat weight_data_gpu;
    VkMat bias_data_gpu;

    Pipeline* pipeline_innerproduct;

    Pipeline* pipeline_innerproduct_sum8;
    Pipeline* pipeline_innerproduct_reduce_sum8;

    Pipeline* pipeline_innerproduct_gemm;

#if NCNN_INT8
    ncnn::Layer* quantize;

    Mat weight_data_int8_packed;
    Mat weight_data_int8_descales;
    Mat bias_data_int8_packed;

    VkMat weight_data_int8_descales_gpu;

    Pipeline* pipeline_innerproduct_int8;
    Pipeline* pipeline_innerproduct_sum8_int8;
    Pipeline* pipeline_innerproduct_reduce_sum8_int8;
    Pipeline* pipeline_innerproduct_gemm_int8;
#endif
};

} // namespace ncnn

#endif // LAYER_INNERPRODUCT_VULKAN_H
