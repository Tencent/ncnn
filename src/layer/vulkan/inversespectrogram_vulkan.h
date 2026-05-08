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

    VkMat basis_cos_data_gpu;
    VkMat basis_sin_data_gpu;

    VkMat window2_data_gpu;

    Pipeline* pipeline_inversespectrogram_idft;
    Pipeline* pipeline_inversespectrogram_idft_pack4;
    Pipeline* pipeline_inversespectrogram_ola;
    Pipeline* pipeline_inversespectrogram_ola_pack4;
};

} // namespace ncnn

#endif // LAYER_INVERSESPECTROGRAM_VULKAN_H
