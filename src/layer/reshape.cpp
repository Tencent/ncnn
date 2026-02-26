// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "reshape.h"

#include "expression.h"

namespace ncnn {

Reshape::Reshape()
{
    one_blob_only = true;
    support_inplace = false;
}

int Reshape::load_param(const ParamDict& pd)
{
    w = pd.get(0, -233);
    h = pd.get(1, -233);
    d = pd.get(11, -233);
    c = pd.get(2, -233);

    ndim = 4;
    if (d == -233)
        ndim = 3;
    if (c == -233)
        ndim = 2;
    if (h == -233)
        ndim = 1;
    if (w == -233)
        ndim = 0;

    shape_expr = pd.get(6, "");

    // count reference blobs
    if (!shape_expr.empty())
    {
        const int blob_count = count_expression_blobs(shape_expr);
        if (blob_count > 1)
            one_blob_only = false;

        // resolve ndim from expression
        std::vector<Mat> blobs(blob_count);
        std::vector<int> outshape;
        int er = eval_list_expression(shape_expr, blobs, outshape);
        if (er != 0)
            return -1;

        ndim = (int)outshape.size();
    }

    return 0;
}

int Reshape::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    std::vector<Mat> bottom_blobs(1);
    bottom_blobs[0] = bottom_blob;
    std::vector<Mat> top_blobs(1);
    int ret = forward(bottom_blobs, top_blobs, opt);
    top_blob = top_blobs[0];
    return ret;
}

int Reshape::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    Mat& top_blob = top_blobs[0];

    // resolve out shape
    int outw = w;
    int outh = h;
    int outd = d;
    int outc = c;

    if (!shape_expr.empty())
    {
        eval_shape_expr(bottom_blobs, outw, outh, outd, outc);
    }

    int total = bottom_blob.w * bottom_blob.h * bottom_blob.d * bottom_blob.c;

    int dims = bottom_blob.dims;

    if (ndim == 1)
    {
        if (outw == 0)
            outw = bottom_blob.w;

        if (outw == -1)
            outw = total;

        if (dims == 1 && bottom_blob.w == outw)
        {
            top_blob = bottom_blob;
            return 0;
        }
    }
    if (ndim == 2)
    {
        if (outw == 0)
            outw = bottom_blob.w;
        if (outh == 0)
            outh = bottom_blob.h;

        if (outw == -1)
            outw = total / outh;
        if (outh == -1)
            outh = total / outw;

        if (dims == 2 && bottom_blob.h == outh)
        {
            top_blob = bottom_blob;
            return 0;
        }
    }
    if (ndim == 3)
    {
        if (outw == 0)
            outw = bottom_blob.w;
        if (outh == 0)
            outh = bottom_blob.h;
        if (outc == 0)
            outc = bottom_blob.c;

        if (outw == -1)
            outw = total / outc / outh;
        if (outh == -1)
            outh = total / outc / outw;
        if (outc == -1)
            outc = total / outh / outw;

        if (dims == 3 && bottom_blob.c == outc)
        {
            top_blob = bottom_blob;
            top_blob.w = outw;
            top_blob.h = outh;
            return 0;
        }
    }
    if (ndim == 4)
    {
        if (outw == 0)
            outw = bottom_blob.w;
        if (outh == 0)
            outh = bottom_blob.h;
        if (outc == 0)
            outc = bottom_blob.c;
        if (outd == 0)
            outd = bottom_blob.d;

        if (outw == -1)
            outw = total / outc / outd / outh;
        if (outh == -1)
            outh = total / outc / outd / outw;
        if (outd == -1)
            outd = total / outc / outh / outw;
        if (outc == -1)
            outc = total / outd / outh / outw;

        if (dims == 4 && bottom_blob.c == outc)
        {
            top_blob = bottom_blob;
            top_blob.w = outw;
            top_blob.h = outh;
            top_blob.d = outd;
            return 0;
        }
    }

    if (ndim == 1)
    {
        top_blob = bottom_blob.reshape(outw, opt.blob_allocator);
    }
    if (ndim == 2)
    {
        top_blob = bottom_blob.reshape(outw, outh, opt.blob_allocator);
    }
    if (ndim == 3)
    {
        top_blob = bottom_blob.reshape(outw, outh, outc, opt.blob_allocator);
    }
    if (ndim == 4)
    {
        top_blob = bottom_blob.reshape(outw, outh, outd, outc, opt.blob_allocator);
    }
    if (top_blob.empty())
        return -100;

    return 0;
}

int Reshape::eval_shape_expr(const std::vector<Mat>& bottom_blobs, int& outw, int& outh, int& outd, int& outc) const
{
    // [size(@0,0),size(@0,1),12,64]
    std::vector<int> shape;
    int er = eval_list_expression(shape_expr, bottom_blobs, shape);
    if (er != 0)
        return -1;

    outw = 1;
    outh = 1;
    outd = 1;
    outc = 1;
    if (shape.size() == 1)
    {
        outw = shape[0];
    }
    if (shape.size() == 2)
    {
        outw = shape[0];
        outh = shape[1];
    }
    if (shape.size() == 3)
    {
        outw = shape[0];
        outh = shape[1];
        outc = shape[2];
    }
    if (shape.size() == 4)
    {
        outw = shape[0];
        outh = shape[1];
        outd = shape[2];
        outc = shape[3];
    }

    return 0;
}

} // namespace ncnn
