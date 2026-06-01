// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_GEMM_VULKAN_H
#define LAYER_GEMM_VULKAN_H

#include "gemm.h"

namespace ncnn {

class Gemm_vulkan : public Gemm
{
public:
    Gemm_vulkan();

    virtual int load_param(const ParamDict& pd);

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int upload_model(VkTransfer& cmd, const Option& opt);

    using Gemm::forward;
    virtual int forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Mat A_data_packed;
    Mat B_data_packed;
    Mat C_data_packed;

    VkMat A_data_gpu;
    VkMat B_data_gpu;
    VkMat C_data_gpu;

    Pipeline* pipeline_gemm;

    // subgroup
    bool use_subgroup_ops;

    // cooperative matrix
    bool use_cooperative_matrix;
    int coopmat_M;
    int coopmat_N;
    int coopmat_K;
    int coopmat_subgroup_size;
    int UNROLL_SG_M;
    int UNROLL_SG_N;
    int UNROLL_SG_K;
    int UNROLL_WG_M;
    int UNROLL_WG_N;

#if NCNN_INT8
public:
    Mat A_data_int8_packed;
    Mat B_data_int8_packed;
    Mat B_data_int8_scales;

    VkMat A_data_int8_scales_gpu;
    VkMat B_data_int8_scales_gpu;

    Pipeline* pipeline_gemm_quantize_A_int8;
    Pipeline* pipeline_gemm_quantize_B_absmax_int8;
    Pipeline* pipeline_gemm_quantize_B_scale_int8;
    Pipeline* pipeline_gemm_quantize_B_int8;

protected:
    int create_pipeline_int8(const Option& opt);
    int upload_model_int8(VkTransfer& cmd, const Option& opt);
    int forward_int8(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const;
#endif
};

} // namespace ncnn

#endif // LAYER_GEMM_VULKAN_H
