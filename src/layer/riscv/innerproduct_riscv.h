// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_INNERPRODUCT_RISCV_H
#define LAYER_INNERPRODUCT_RISCV_H

#include "innerproduct.h"

namespace ncnn {

class InnerProduct_riscv : public InnerProduct
{
public:
    InnerProduct_riscv();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

protected:
#if NCNN_ZFH
    int create_pipeline_fp16s(const Option& opt);
    int forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
    int forward_fp16sa(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
#endif

public:
    Layer* flatten;

    Mat weight_data_tm;

    // fp16
    Mat bias_data_fp16;
};

} // namespace ncnn

#endif // LAYER_INNERPRODUCT_RISCV_H
