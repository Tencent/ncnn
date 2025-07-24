// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_GEMM_VULKAN_H
#define LAYER_GEMM_VULKAN_H

#include "gemm.h"

namespace ncnn {

class Gemm_vulkan : public Gemm
{
public:
    Gemm_vulkan();

    virtual int load_param(const ParamDict& pd);

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int upload_model(VkTransfer& cmd, const Option& opt);

    using Gemm::forward;
    virtual int forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Mat A_data_packed;
    Mat B_data_packed;
    Mat C_data_packed;

    VkMat A_data_gpu;
    VkMat B_data_gpu;
    VkMat C_data_gpu;

    Pipeline* pipeline_gemm;
};

} // namespace ncnn

#endif // LAYER_GEMM_VULKAN_H
