// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "expanddims.h"

namespace ncnn {

ExpandDims::ExpandDims()
{
    one_blob_only = true;
    support_inplace = false;
}

int ExpandDims::load_param(const ParamDict& pd)
{
    axes = pd.get(3, Mat());

    return 0;
}

int ExpandDims::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int dims = bottom_blob.dims;

    const int outdims = dims + axes.w;

    bool expand_w = false;
    bool expand_h = false;
    bool expand_d = false;
    bool expand_c = false;

    {
        const int* axes_ptr = axes;
        for (int i = 0; i < axes.w; i++)
        {
            int axis = axes_ptr[i];
            if (axis < 0)
                axis = outdims + axis;

            if (outdims == 2)
            {
                if (axis == 0) expand_h = true;
                if (axis == 1) expand_w = true;
            }
            if (outdims == 3)
            {
                if (axis == 0) expand_c = true;
                if (axis == 1) expand_h = true;
                if (axis == 2) expand_w = true;
            }
            if (outdims == 4)
            {
                if (axis == 0) expand_c = true;
                if (axis == 1) expand_d = true;
                if (axis == 2) expand_h = true;
                if (axis == 3) expand_w = true;
            }
        }
    }

    top_blob = bottom_blob;

    if (outdims == 2)
    {
        if (expand_w)
        {
            top_blob = bottom_blob.reshape(1, w, opt.blob_allocator);
        }
        else if (expand_h)
        {
            top_blob = bottom_blob.reshape(w, 1, opt.blob_allocator);
        }
    }
    if (outdims == 3)
    {
        if (expand_w && expand_h)
        {
            top_blob = bottom_blob.reshape(1, 1, w, opt.blob_allocator);
        }
        else if (expand_w && expand_c)
        {
            top_blob = bottom_blob.reshape(1, w, 1, opt.blob_allocator);
        }
        else if (expand_h && expand_c)
        {
            top_blob = bottom_blob.reshape(w, 1, 1, opt.blob_allocator);
        }
        else if (expand_w)
        {
            top_blob = bottom_blob.reshape(1, w, h, opt.blob_allocator);
        }
        else if (expand_h)
        {
            top_blob = bottom_blob.reshape(w, 1, h, opt.blob_allocator);
        }
        else if (expand_c)
        {
            top_blob = bottom_blob.reshape(w, h, 1, opt.blob_allocator);
        }
    }
    if (outdims == 4)
    {
        if (expand_w && expand_h && expand_d)
        {
            top_blob = bottom_blob.reshape(1, 1, 1, w, opt.blob_allocator);
        }
        else if (expand_w && expand_h && expand_c)
        {
            top_blob = bottom_blob.reshape(1, 1, w, 1, opt.blob_allocator);
        }
        else if (expand_w && expand_d && expand_c)
        {
            top_blob = bottom_blob.reshape(1, w, 1, 1, opt.blob_allocator);
        }
        else if (expand_h && expand_d && expand_c)
        {
            top_blob = bottom_blob.reshape(w, 1, 1, 1, opt.blob_allocator);
        }
        else if (expand_w && expand_h)
        {
            top_blob = bottom_blob.reshape(1, 1, w, h, opt.blob_allocator);
        }
        else if (expand_w && expand_c)
        {
            top_blob = bottom_blob.reshape(1, w, h, 1, opt.blob_allocator);
        }
        else if (expand_d && expand_c)
        {
            top_blob = bottom_blob.reshape(w, h, 1, 1, opt.blob_allocator);
        }
        else if (expand_w && expand_d)
        {
            top_blob = bottom_blob.reshape(1, w, 1, h, opt.blob_allocator);
        }
        else if (expand_h && expand_c)
        {
            top_blob = bottom_blob.reshape(w, 1, h, 1, opt.blob_allocator);
        }
        else if (expand_h && expand_d)
        {
            top_blob = bottom_blob.reshape(w, 1, 1, h, opt.blob_allocator);
        }
        else if (expand_w)
        {
            top_blob = bottom_blob.reshape(1, w, h, channels, opt.blob_allocator);
        }
        else if (expand_h)
        {
            top_blob = bottom_blob.reshape(w, 1, h, channels, opt.blob_allocator);
        }
        else if (expand_d)
        {
            top_blob = bottom_blob.reshape(w, h, 1, channels, opt.blob_allocator);
        }
        else if (expand_c)
        {
            top_blob = bottom_blob.reshape(w, h, channels, 1, opt.blob_allocator);
        }
    }

    if (top_blob.empty())
        return -100;

    return 0;
}

} // namespace ncnn
