// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_MULTIHEADATTENTION_VULKAN_H
#define LAYER_MULTIHEADATTENTION_VULKAN_H

#include "multiheadattention.h"

namespace ncnn {

class MultiHeadAttention_vulkan : public MultiHeadAttention
{
public:
    MultiHeadAttention_vulkan();

    virtual int load_param(const ParamDict& pd);

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int upload_model(VkTransfer& cmd, const Option& opt);

    using MultiHeadAttention::forward;
    virtual int forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const;

public:
    Layer* q_gemm;
    Layer* k_gemm;
    Layer* v_gemm;
    Layer* o_gemm;

    Layer* qk_softmax;

    Layer* kvcache_concat;

    // Original QK^T and QKV pipelines
    Pipeline* pipeline_multiheadattention_qk_cross;
    Pipeline* pipeline_multiheadattention_qk_cross_pack4;
    Pipeline* pipeline_multiheadattention_qk_cross_pack1to4;
    Pipeline* pipeline_multiheadattention_qk_cross_pack4to1;

    Pipeline* pipeline_multiheadattention_qkv_cross;
    Pipeline* pipeline_multiheadattention_qkv_cross_pack4;
    Pipeline* pipeline_multiheadattention_qkv_cross_pack1to4;
    Pipeline* pipeline_multiheadattention_qkv_cross_pack4to1;

    // P1: Subgroup-optimized QK^T pipeline (standalone)
    Pipeline* pipeline_multiheadattention_qk_subgroup;

    // P3/P4: Fused online softmax pipelines (QK^T + softmax + QKV in one kernel)
    Pipeline* pipeline_multiheadattention_fused;
    Pipeline* pipeline_multiheadattention_fused_coopload;    // P4: cooperative K/V shared load
    Pipeline* pipeline_multiheadattention_fused_subgroup;    // P1: subgroup reduction variant
    Pipeline* pipeline_multiheadattention_fused_coopmat;     // P5: cooperative matrix variant
};

} // namespace ncnn

#endif // LAYER_MULTIHEADATTENTION_VULKAN_H