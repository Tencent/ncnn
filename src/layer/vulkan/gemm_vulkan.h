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

protected:
#if NCNN_WEIGHT_QUANT
    bool use_weight_quantized_pipeline() const;
    int get_weight_slice_rows() const;
    int create_pipeline_weight_quantized(const Option& opt);
    int create_pipeline_weight_quantized_cm(const Option& opt);
    int upload_model_weight_quantized(VkTransfer& cmd, const Option& opt);
    int create_pipeline_int8sa(const Option& opt);
    int forward_weight_quantized(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const;
    VkMat rotate_A_convrot(const VkMat& A, int M, int K, VkCompute& cmd, const Option& opt) const;
    int gemm_int8sa(const VkMat& A, const VkMat& C, int M, int broadcast_type_C, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;
#endif
#if NCNN_INT8
    int create_pipeline_int8(const Option& opt);
    int upload_model_int8(VkTransfer& cmd, const Option& opt);
    int forward_int8(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const;
#endif

public:
    Mat A_data_packed;
    Mat B_data_packed;
    Mat C_data_packed;

    VkMat A_data_gpu;
    VkMat B_data_gpu;
    VkMat C_data_gpu;

    Pipeline* pipeline_gemm;
    Pipeline* pipeline_gemm_pack4;

#if NCNN_WEIGHT_QUANT
    int weight_slice_rows;
    std::vector<VkMat> B_data_packed_gpu;
    std::vector<VkMat> B_data_descales_gpu;
    Pipeline* pipeline_gemm_weight_quantized;
    Pipeline* pipeline_gemm_weight_quantized_cm;
    Pipeline* pipeline_gemm_convrot_rotate;
    Pipeline* pipeline_gemm_quantize_A_wq_int8;
    Pipeline* pipeline_gemm_wq_int8;
    bool use_cooperative_matrix_wq_int8;
    bool fuse_convrot_quantize;
    int coopmat_M_wq;
    int coopmat_N_wq;
#endif

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
    Mat A_data_int8_packed;
    Mat B_data_int8_packed;
    Mat A_data_int8_descales;
    Mat B_data_int8_descale;

    VkMat A_data_int8_descales_gpu;
    VkMat B_data_int8_descale_gpu;

    Pipeline* pipeline_gemm_quantize_A_int8;
    Pipeline* pipeline_gemm_quantize_B_absmax_int8;
    Pipeline* pipeline_gemm_quantize_B_descale_int8;
    Pipeline* pipeline_gemm_quantize_B_int8;
#endif
};

} // namespace ncnn

#endif // LAYER_GEMM_VULKAN_H
