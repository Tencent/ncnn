// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_INNERPRODUCT_POWERPC_H
#define LAYER_INNERPRODUCT_POWERPC_H

#include "innerproduct.h"

namespace ncnn {

class InnerProduct_powerpc : public InnerProduct
{
public:
    InnerProduct_powerpc();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

protected:
    // generic scalar int8 compute on unpacked blobs, with powerpc output packing
    int forward_int8_scalar(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

public:
    Layer* flatten;

    Mat weight_data_tm;
};

} // namespace ncnn

#endif // LAYER_INNERPRODUCT_POWERPC_H
