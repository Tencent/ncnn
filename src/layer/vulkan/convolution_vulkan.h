// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CONVOLUTION_VULKAN_H
#define LAYER_CONVOLUTION_VULKAN_H

#include "convolution.h"

namespace ncnn {

class Convolution_vulkan : public Convolution
{
public:
    Convolution_vulkan();

    virtual int load_param(const ParamDict& pd);

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int upload_model(VkTransfer& cmd, const Option& opt);

    using Convolution::forward;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;

    int create_pipeline_int8(const Option& opt);
    int upload_model_int8(VkTransfer& cmd, const Option& opt);
    int forward_int8(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;

public:
    ncnn::Layer* padding;

    Mat weight_data_packed;
    Mat weight_data_int8_packed;
    Mat weight_winograd23_data_packed;
    Mat weight_winograd43_data_packed;
    Mat weight_winograd23_data_int8_packed;
    Mat weight_winograd43_data_int8_packed;

    VkMat weight_data_gpu;
    VkMat weight_data_int8_gpu;
    VkMat weight_data_int8_scales_gpu;
    VkMat bottom_blob_int8_scales_gpu;
    VkMat top_blob_int8_scales_gpu;
    VkMat bias_data_gpu;

    Pipeline* pipeline_convolution;
    Pipeline* pipeline_convolution_int8;
    Pipeline* pipeline_convolution_int8_cm;
    Pipeline* pipeline_convolution_1x1s1d1;
    Pipeline* pipeline_convolution_1x1s1d1_int8;

    Pipeline* pipeline_convolution_gemm;
    Pipeline* pipeline_convolution_gemm_int8;

    // winograd23 and winograd43
    VkMat weight_data_gpu_tm_winograd23;
    VkMat weight_data_int8_gpu_tm_winograd23;
    Pipeline* pipeline_convolution_3x3s1d1_winograd23_transform_input;
    Pipeline* pipeline_convolution_3x3s1d1_winograd23_gemm;
    Pipeline* pipeline_convolution_3x3s1d1_winograd23_transform_output;
    Pipeline* pipeline_convolution_3x3s1d1_winograd23_transform_input_int8;
    Pipeline* pipeline_convolution_3x3s1d1_winograd23_gemm_int8;
    Pipeline* pipeline_convolution_3x3s1d1_winograd23_transform_output_int8;

    VkMat weight_data_gpu_tm_winograd43;
    VkMat weight_data_int8_gpu_tm_winograd43;
    Pipeline* pipeline_convolution_3x3s1d1_winograd43_transform_input;
    Pipeline* pipeline_convolution_3x3s1d1_winograd43_gemm;
    Pipeline* pipeline_convolution_3x3s1d1_winograd43_transform_output;
    Pipeline* pipeline_convolution_3x3s1d1_winograd43_transform_input_int8;
    Pipeline* pipeline_convolution_3x3s1d1_winograd43_gemm_int8;
    Pipeline* pipeline_convolution_3x3s1d1_winograd43_transform_output_int8;

    // convolution as fc
    ncnn::Layer* reshape_1x1xw;
    ncnn::Layer* reshape_w;

    // cooperative matrix
    bool use_cooperative_matrix;
    bool use_int8_cooperative_matrix;
    bool use_int8_winograd_int16_packed;
    bool use_int8_winograd_int16_storage;
    int coopmat_M;
    int coopmat_N;
    int coopmat_K;
    int coopmat_subgroup_size;
    int UNROLL_SG_M;
    int UNROLL_SG_N;
    int UNROLL_SG_K;
    int UNROLL_WG_M;
    int UNROLL_WG_N;
};

} // namespace ncnn

#endif // LAYER_CONVOLUTION_VULKAN_H
