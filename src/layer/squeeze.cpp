// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "squeeze.h"

namespace ncnn {

Squeeze::Squeeze()
{
    one_blob_only = true;
    support_inplace = false;
}

int Squeeze::load_param(const ParamDict& pd)
{
    squeeze_w = pd.get(0, 0);
    squeeze_h = pd.get(1, 0);
    squeeze_d = pd.get(11, 0);
    squeeze_c = pd.get(2, 0);
    axes = pd.get(3, Mat());

    return 0;
}

int Squeeze::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;

    bool _squeeze_w = false;
    bool _squeeze_h = false;
    bool _squeeze_d = false;
    bool _squeeze_c = false;

    if (axes.empty())
    {
        _squeeze_w = w == 1 && squeeze_w;
        _squeeze_h = h == 1 && squeeze_h;
        _squeeze_d = d == 1 && squeeze_d;
        _squeeze_c = channels == 1 && squeeze_c;
    }
    else
    {
        const int* axes_ptr = axes;
        for (int i = 0; i < axes.w; i++)
        {
            int axis = axes_ptr[i];
            if (axis < 0)
                axis = dims + axis;

            if (dims == 1 && axis == 0)
            {
                _squeeze_w = w == 1;
            }
            if (dims == 2 && axis == 0)
            {
                _squeeze_h = h == 1;
            }
            if (dims == 2 && axis == 1)
            {
                _squeeze_w = w == 1;
            }
            if (dims == 3 && axis == 0)
            {
                _squeeze_c = channels == 1;
            }
            if (dims == 3 && axis == 1)
            {
                _squeeze_h = h == 1;
            }
            if (dims == 3 && axis == 2)
            {
                _squeeze_w = w == 1;
            }
            if (dims == 4 && axis == 0)
            {
                _squeeze_c = channels == 1;
            }
            if (dims == 4 && axis == 1)
            {
                _squeeze_d = d == 1;
            }
            if (dims == 4 && axis == 2)
            {
                _squeeze_h = h == 1;
            }
            if (dims == 4 && axis == 3)
            {
                _squeeze_w = w == 1;
            }
        }
    }

    top_blob = bottom_blob;

    if (dims == 1)
    {
        if (_squeeze_w)
        {
            top_blob = bottom_blob.reshape(1, opt.blob_allocator);
        }
    }

    if (dims == 2)
    {
        if (_squeeze_w && _squeeze_h)
        {
            top_blob = bottom_blob.reshape(1, opt.blob_allocator);
        }
        else if (_squeeze_w)
        {
            top_blob = bottom_blob.reshape(h, opt.blob_allocator);
        }
        else if (_squeeze_h)
        {
            top_blob = bottom_blob.reshape(w, opt.blob_allocator);
        }
    }

    if (dims == 3)
    {
        if (_squeeze_w && _squeeze_h && _squeeze_c)
        {
            top_blob = bottom_blob.reshape(1, opt.blob_allocator);
        }
        else if (_squeeze_w && _squeeze_h)
        {
            top_blob = bottom_blob.reshape(channels, opt.blob_allocator);
        }
        else if (_squeeze_h && _squeeze_c)
        {
            top_blob = bottom_blob.reshape(w, opt.blob_allocator);
        }
        else if (_squeeze_w && _squeeze_c)
        {
            top_blob = bottom_blob.reshape(h, opt.blob_allocator);
        }
        else if (_squeeze_w)
        {
            top_blob = bottom_blob.reshape(h, channels, opt.blob_allocator);
        }
        else if (_squeeze_h)
        {
            top_blob = bottom_blob.reshape(w, channels, opt.blob_allocator);
        }
        else if (_squeeze_c)
        {
            top_blob = bottom_blob.reshape(w, h, opt.blob_allocator);
        }
    }

    if (dims == 4)
    {
        if (_squeeze_w && _squeeze_h && _squeeze_d && _squeeze_c)
        {
            top_blob = bottom_blob.reshape(1, opt.blob_allocator);
        }
        else if (_squeeze_w && _squeeze_h && _squeeze_d)
        {
            top_blob = bottom_blob.reshape(channels, opt.blob_allocator);
        }
        else if (_squeeze_h && _squeeze_d && _squeeze_c)
        {
            top_blob = bottom_blob.reshape(w, opt.blob_allocator);
        }
        else if (_squeeze_w && _squeeze_d && _squeeze_c)
        {
            top_blob = bottom_blob.reshape(h, opt.blob_allocator);
        }
        else if (_squeeze_w && _squeeze_h && _squeeze_c)
        {
            top_blob = bottom_blob.reshape(d, opt.blob_allocator);
        }
        else if (_squeeze_w && _squeeze_h)
        {
            top_blob = bottom_blob.reshape(d, channels, opt.blob_allocator);
        }
        else if (_squeeze_w && _squeeze_d)
        {
            top_blob = bottom_blob.reshape(h, channels, opt.blob_allocator);
        }
        else if (_squeeze_h && _squeeze_d)
        {
            top_blob = bottom_blob.reshape(w, channels, opt.blob_allocator);
        }
        else if (_squeeze_h && _squeeze_c)
        {
            top_blob = bottom_blob.reshape(w, d, opt.blob_allocator);
        }
        else if (_squeeze_w && _squeeze_c)
        {
            top_blob = bottom_blob.reshape(h, d, opt.blob_allocator);
        }
        else if (_squeeze_d && _squeeze_c)
        {
            top_blob = bottom_blob.reshape(w, h, opt.blob_allocator);
        }
        else if (_squeeze_w)
        {
            top_blob = bottom_blob.reshape(h, d, channels, opt.blob_allocator);
        }
        else if (_squeeze_h)
        {
            top_blob = bottom_blob.reshape(w, d, channels, opt.blob_allocator);
        }
        else if (_squeeze_d)
        {
            top_blob = bottom_blob.reshape(w, h, channels, opt.blob_allocator);
        }
        else if (_squeeze_c)
        {
            top_blob = bottom_blob.reshape(w, h, d, opt.blob_allocator);
        }
    }

    if (top_blob.empty())
        return -100;

    return 0;
}

} // namespace ncnn
