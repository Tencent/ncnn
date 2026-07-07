// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_RESHAPE_H
#define LAYER_RESHAPE_H

#include "layer.h"

namespace ncnn {

class Reshape : public Layer
{
public:
    Reshape();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

protected:
    int eval_shape_expr(const std::vector<Mat>& bottom_blobs, int& outw, int& outh, int& outd, int& outc) const;
#if NCNN_BATCH
    int forward_batch(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
    int resolve_batch_shape(const std::vector<Mat>& bottom_blobs,
                            Mat& input_shape, Mat& output_shape,
                            int& input_axis, int& output_axis,
                            size_t& total) const;
    void copy_batch_reshape(const Mat& bottom_blob, Mat& top_blob,
                            const Mat& input_shape, int input_axis,
                            const Mat& output_shape, int output_axis,
                            size_t total, size_t scalar_elemsize,
                            const Option& opt) const;
#endif

public:
    // reshape flag
    // 0 = copy from bottom
    // -1 = remaining
    // -233 = drop this dim (default)
    int w;
    int h;
    int d;
    int c;

    int ndim;
#if NCNN_BATCH
    // 233 = no ncnn batch axis, otherwise python-style axis in logical shape
    int input_batch_axis;
    int output_batch_axis;
#endif

    // see docs/developer-guide/expression.md
    std::string shape_expr;
};

} // namespace ncnn

#endif // LAYER_RESHAPE_H
