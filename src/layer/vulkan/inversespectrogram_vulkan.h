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

    virtual int load_param(const ParamDict& pd);

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int upload_model(VkTransfer& cmd, const Option& opt);

    using InverseSpectrogram::forward;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;

public:
    // idft basis, 4 planes of [bin k][tap m]:
    //   plane 0: re from sp_re, plane 1: re from sp_im
    //   plane 2: im from sp_re, plane 3: im from sp_im
    Mat idft_weight;
    // window_data[k]^2 for overlap-add normalization
    Mat window_sq;

    VkMat weight_data_gpu;
    VkMat window_sq_gpu;

    Pipeline* pipeline_inversespectrogram;
};

} // namespace ncnn

#endif // LAYER_INVERSESPECTROGRAM_VULKAN_H
