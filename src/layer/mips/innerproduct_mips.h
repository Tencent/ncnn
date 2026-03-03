// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_INNERPRODUCT_MIPS_H
#define LAYER_INNERPRODUCT_MIPS_H

#include "innerproduct.h"

namespace ncnn {

class InnerProduct_mips : public InnerProduct
{
public:
    InnerProduct_mips();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

protected:
#if __mips_msa
    int create_pipeline_fp16s(const Option& opt);
    int forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
#endif
#if NCNN_INT8
    int create_pipeline_int8_mips(const Option& opt);
    int forward_int8_mips(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
#endif

public:
    Layer* flatten;

    Mat weight_data_tm;

#if NCNN_INT8
    Mat scale_in_data;
#endif
};

} // namespace ncnn

#endif // LAYER_INNERPRODUCT_MIPS_H
