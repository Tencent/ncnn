// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_GRU_VULKAN_H
#define LAYER_GRU_VULKAN_H

#include "gru.h"

namespace ncnn {

class GRU_vulkan : public GRU
{
public:
    GRU_vulkan();

    virtual int load_param(const ParamDict& pd);

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int upload_model(VkTransfer& cmd, const Option& opt);

    using GRU::forward;
    virtual int forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;

public:
    VkMat weight_xc_data_gpu;
    VkMat bias_c_data_gpu;
    VkMat weight_hc_data_gpu;

    VkMat weight_xc_data_gpu_pack4;
    VkMat bias_c_data_gpu_pack4;
    VkMat weight_hc_data_gpu_pack4;

    Pipeline* pipeline_gru_step;
    Pipeline* pipeline_gru_step_pack4;
};

} // namespace ncnn

#endif // LAYER_GRU_VULKAN_H
