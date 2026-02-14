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

    Layer* padding;
    Layer* unfold;
    Layer* gemm;

    Pipeline* pipeline_spectrogram_post;
};

} // namespace ncnn

#endif // LAYER_SPECTROGRAM_VULKAN_H
