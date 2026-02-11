// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SDPA_VULKAN_H
#define LAYER_SDPA_VULKAN_H

#include "sdpa.h"

namespace ncnn {

class SDPA_vulkan : public SDPA
{
public:
    SDPA_vulkan();

    virtual int load_param(const ParamDict& pd);

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using SDPA::forward;
    virtual int forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const;

public:
    Layer* qk_softmax;
    Layer* kvcache_concat;

    Pipeline* pipeline_sdpa_qk_cross;
    Pipeline* pipeline_sdpa_qkv_cross;

    // flash attention
    Pipeline* pipeline_sdpa_fa[8];

    bool use_flash_attention;
    int FA_coopmat_M;
    int FA_coopmat_N;
    int FA_coopmat_K;
    int FA_coopmat_subgroup_size;
    int FA_UNROLL_SG_M;
    int FA_UNROLL_WG_M;

    // cooperative matrix
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

#endif // LAYER_SDPA_VULKAN_H
