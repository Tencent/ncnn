// Copyright 2026 MYQ
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_MATMUL_VULKAN_H
#define LAYER_MATMUL_VULKAN_H

#include "matmul.h"

namespace ncnn {

class MatMul_vulkan : public MatMul
{
public:
    MatMul_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using MatMul::forward;
    virtual int forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_matmul;
    Pipeline* pipeline_matmul_sg;
    Pipeline* pipeline_matmul_cm;

    // subgroup
    bool use_subgroup_ops;

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

#endif // LAYER_MATMUL_VULKAN_H
