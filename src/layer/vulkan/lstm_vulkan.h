// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_LSTM_VULKAN_H
#define LAYER_LSTM_VULKAN_H

#include "lstm.h"

namespace ncnn {

class LSTM_vulkan : public LSTM
{
public:
    LSTM_vulkan();

    virtual int load_param(const ParamDict& pd);

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int upload_model(VkTransfer& cmd, const Option& opt);

    using LSTM::forward;
    virtual int forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;

public:
    VkMat weight_xc_data_gpu;
    VkMat bias_c_data_gpu;
    VkMat weight_hc_data_gpu;
    VkMat weight_hr_data_gpu;

    Pipeline* pipeline_lstm_copy;
    Pipeline* pipeline_lstm_step;
    Pipeline* pipeline_lstm_step_h;
    Pipeline* pipeline_lstm_proj;
};

} // namespace ncnn

#endif // LAYER_LSTM_VULKAN_H
