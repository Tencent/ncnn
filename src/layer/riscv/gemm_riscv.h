// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_GEMM_RISCV_H
#define LAYER_GEMM_RISCV_H

#include "gemm.h"

namespace ncnn {

class Gemm_riscv : public Gemm
{
public:
    Gemm_riscv();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

#if NCNN_WEIGHT_QUANT && NCNN_ZFH
    static void quantize_A_tile_wq_int8_fp16s(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales);
    static void transpose_quantize_A_tile_wq_int8_fp16s(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales);
    static void unpack_output_tile_wq_int8_fp16s(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, float beta);
    static void transpose_unpack_output_tile_wq_int8_fp16s(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, float beta);
#endif // NCNN_WEIGHT_QUANT && NCNN_ZFH

protected:
#if NCNN_ZFH
    int create_pipeline_fp16s(const Option& opt);
    int forward_fp16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
#endif
#if NCNN_WEIGHT_QUANT
    int create_pipeline_wq_int8(const Option& opt);
    int forward_wq_int8(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
#endif

public:
    int nT;
    Mat AT_data;
    Mat BT_data;
    Mat CT_data;
#if NCNN_WEIGHT_QUANT
    Mat BT_data_wq_int8;
    Mat BT_data_wq_int8_descales;
#endif
};

} // namespace ncnn

#endif // LAYER_GEMM_RISCV_H
