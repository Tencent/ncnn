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

    virtual int load_param(const ParamDict& pd);

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int upload_model(VkTransfer& cmd, const Option& opt);

    using Spectrogram::forward;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;

public:
    // stft as conv: kernel=n_fft stride=hoplen, dft basis as weight
    // row i = real basis of bin i, row freqs_onesided + i = imag basis
    Mat dft_weight;
    VkMat weight_data_gpu;

    // center border (constant / replicate / reflect)
    Layer* padding;

    Pipeline* pipeline_spectrogram;
};

} // namespace ncnn

#endif // LAYER_SPECTROGRAM_VULKAN_H
