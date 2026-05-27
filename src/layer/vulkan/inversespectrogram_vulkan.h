// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_INVERSESPECTROGRAM_VULKAN_H
#define LAYER_INVERSESPECTROGRAM_VULKAN_H

#include "inversespectrogram.h"

namespace ncnn {

class InverseSpectrogram_vulkan : public InverseSpectrogram
{
public:
    InverseSpectrogram_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int upload_model(VkTransfer& cmd, const Option& opt);

    using InverseSpectrogram::forward;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Mat window2_data;

    Mat basis_cos_data_packed;
    Mat basis_sin_data_packed;
    Mat basis_cos_data_gemm_packed;
    Mat basis_sin_data_gemm_packed;

    VkMat basis_cos_data_gpu;
    VkMat basis_sin_data_gpu;
    VkMat basis_cos_data_gemm_gpu;
    VkMat basis_sin_data_gemm_gpu;

    VkMat window2_data_gpu;

    Pipeline* pipeline_inversespectrogram_idft;
    Pipeline* pipeline_inversespectrogram_idft_pack4;
    Pipeline* pipeline_inversespectrogram_idft_packed_gemm;
    Pipeline* pipeline_inversespectrogram_idft_gemm_cm;
    Pipeline* pipeline_inversespectrogram_ola;
    Pipeline* pipeline_inversespectrogram_ola_pack4;

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
};

} // namespace ncnn

#endif // LAYER_INVERSESPECTROGRAM_VULKAN_H
