// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_GEMM_X86_H
#define LAYER_GEMM_X86_H

#include "gemm.h"

namespace ncnn {

class Gemm_x86 : public Gemm
{
public:
    Gemm_x86();

    virtual int create_pipeline(const Option& opt);

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

protected:
#if NCNN_INT8
    int create_pipeline_int8(const Option& opt);
    int forward_int8(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
#endif

public:
    int nT;
    Mat AT_data;
    Mat BT_data;
    Mat CT_data;
};

// expose some gemm internal routines for convolution uses
namespace Gemm_x86_utility {
#if NCNN_INT8
void pack_A_tile_int8(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk);
void gemm_transB_packed_tile_int8(const Mat& AT_tile, const Mat& BT_tile, Mat& topT_tile, int i, int max_ii, int j, int max_jj, int k, int max_kk);
#endif
} // namespace Gemm_x86_utility

} // namespace ncnn

#endif // LAYER_GEMM_X86_H
