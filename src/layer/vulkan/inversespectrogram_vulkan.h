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

    Layer* gemm_real;
    Layer* gemm_imag;

    VkMat window2_data_gpu;

    Pipeline* pipeline_inversespectrogram_build_b;
    Pipeline* pipeline_inversespectrogram_ola;
};

} // namespace ncnn

#endif // LAYER_INVERSESPECTROGRAM_VULKAN_H
