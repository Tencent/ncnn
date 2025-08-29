// Copyright 2021 Xavier Hsinyuan <me@lstlx.com>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_GRU_RISCV_H
#define LAYER_GRU_RISCV_H

#include "gru.h"

namespace ncnn {

class GRU_riscv : public GRU
{
public:
    GRU_riscv();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
    virtual int create_pipeline(const Option& opt);

protected:
#if NCNN_ZFH
    int forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
    int forward_fp16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
    int forward_fp16sa(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
    int forward_fp16sa(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
    int create_pipeline_fp16sa(const Option& opt);
#endif

public:
    Mat weight_xc_data_fp16sa;
    Mat bias_c_data_fp16sa;
    Mat weight_hc_data_fp16sa;
};

} // namespace ncnn

#endif // LAYER_GRU_RISCV_H
