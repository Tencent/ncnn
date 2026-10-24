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

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

#if NCNN_WEIGHT_QUANT && NCNN_ZFH
    static void quantize_A_tile_wq_int8_fp16s(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales);
    static void transpose_quantize_A_tile_wq_int8_fp16s(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales);
    static void unpack_output_tile_wq_int8_fp16s(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, float beta, int output_transpose);
#endif // NCNN_WEIGHT_QUANT && NCNN_ZFH

protected:
#if NCNN_ZFH
    int create_pipeline_fp16sa(const Option& opt);
    int forward_fp16sa(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
    int create_pipeline_fp16s(const Option& opt);
    int forward_fp16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
#endif
#if NCNN_WEIGHT_QUANT
    int create_pipeline_wq_int8(const Option& opt);
    int forward_wq_int8(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
#endif
#if NCNN_ZFH && NCNN_RISCV_SPACEMIT_IME2
    // SpacemiT K3 A100 IME2 (smt.vfwmadot) path
    int create_pipeline_ime2(const Option& opt);
    int forward_ime2(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
    /* VLEN-agnostic naive fallback when the team check fails (at the team head or inside the
     * kernel execution region): accumulate in fp32 directly from the fixed-layout BT_data_ime2 tiles. */
    int forward_ime2_fallback(const Mat& A, int M, int N, int K, float alpha, int out_elempack, std::vector<Mat>& top_blobs, const Option& opt) const;
#endif

public:
    int nT;
    Mat AT_data;
    Mat BT_data;
    Mat CT_data;
#if NCNN_ZFH && NCNN_RISCV_SPACEMIT_IME2
    int use_ime2_path = 0;
    /* whether the fp32 panel is really built: only the full-takeover branch builds the fp32
     * generic fallback pipeline at pipeline-build time (BT_data is fp32 4u). The non-takeover
     * branch builds the fp16s pipeline, whose BT_data is fp16 (2u) — the fp32 generic kernel
     * cannot read it. */
    int ime2_fp32_fallback_ready = 0;
    /* Retained reference to the original fp32 weights for the affinity fallback:
     * the generic pipeline releases B_data under lightmode, but the fallback
     * must not compute fp32 results from the fp16-rounded IME2 panel. Mat is
     * refcounted, so this costs no copy -- it only keeps the buffer alive. */
    Mat B_data_fp32_fallback;
    Mat BT_data_ime2;
    Mat BT_i8c_ime2;    // int8 column-major weight panel (integer-domain design, see REPORT 7.41)
    Mat BT_wscale_ime2; // one fp16 scale per column
#endif
#if NCNN_WEIGHT_QUANT
    Mat BT_data_wq_int8;
    Mat BT_data_wq_int8_descales;
#endif
};

} // namespace ncnn

#endif // LAYER_GEMM_RISCV_H
