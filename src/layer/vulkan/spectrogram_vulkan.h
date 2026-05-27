// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SPECTROGRAM_VULKAN_H
#define LAYER_SPECTROGRAM_VULKAN_H

#include "spectrogram.h"

namespace ncnn {

class Spectrogram_vulkan : public Spectrogram
{
public:
    Spectrogram_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int upload_model(VkTransfer& cmd, const Option& opt);

    using Spectrogram::forward;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;

public:
    int n_freq;

    Mat basis_data_packed;
    Mat basis_imag_data_packed;

    VkMat basis_data_gpu;
    VkMat basis_imag_data_gpu;

    Pipeline* pipeline_spectrogram_packed;
    Pipeline* pipeline_spectrogram_packed_gemm;
    Pipeline* pipeline_spectrogram_gemm_cm;

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

#endif // LAYER_SPECTROGRAM_VULKAN_H
