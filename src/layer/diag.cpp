// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "diag.h"

namespace ncnn {

Diag::Diag()
{
    one_blob_only = true;
    support_inplace = false;
}

int Diag::load_param(const ParamDict& pd)
{
    diagonal = pd.get(0, 0);

    return 0;
}

int Diag::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;

    if (dims == 1)
    {
        int w = bottom_blob.w;
        int top_w = w + ((diagonal >= 0) ? diagonal : -diagonal);

        top_blob.create(top_w, top_w, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        top_blob.fill(0.0f);

        int bias_r = -std::min(diagonal, 0);
        int bias_c = std::max(diagonal, 0);

        for (int i = 0; i < w; i++)
        {
            top_blob.row(i + bias_r)[i + bias_c] = bottom_blob[i];
        }
    }
    if (dims == 2)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;

        int len = 0;
        int minimum = std::min(w - h, 0);
        int maximum = std::max(w - h, 0);
        if (diagonal <= maximum && diagonal >= minimum)
            len = std::min(w, h);
        else if (diagonal > -h && diagonal < minimum)
            len = diagonal + h;
        else if (diagonal > maximum && diagonal < w)
            len = -diagonal + w;

        top_blob.create(len, elemsize, opt.blob_allocator);
        if (top_blob.empty())
        {
            if (len == 0)
                return 0;
            return -100;
        }

        int bias_r = -std::min(diagonal, 0);
        int bias_c = std::max(diagonal, 0);

        for (int i = 0; i < len; i++)
        {
            top_blob[i] = bottom_blob.row(i + bias_r)[i + bias_c];
        }
    }

    return 0;
}

} // namespace ncnn
