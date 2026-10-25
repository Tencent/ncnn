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
    // idft basis, 2 planes of [bin k][tap m]:
    //   idft_weight = wcos, idft_weight_sin = wsin
    //   (w_re_re = wcos, w_re_im = -wsin, w_im_re = wsin, w_im_im = wcos)
    Mat idft_weight;
    Mat idft_weight_sin;
    // window_data[k]^2 for overlap-add normalization
    Mat window_sq;

    VkMat weight_data_gpu;
    VkMat weight_sin_data_gpu;
    VkMat window_sq_gpu;

    Pipeline* pipeline_inversespectrogram;
};

} // namespace ncnn

#endif // LAYER_INVERSESPECTROGRAM_VULKAN_H
