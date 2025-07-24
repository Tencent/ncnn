// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "mish.h"

namespace ncnn {

Mish::Mish()
{
    one_blob_only = true;
    support_inplace = true;
}

int Mish::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int size = w * h * d;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        for (int i = 0; i < size; i++)
        {
            const float MISH_THRESHOLD = 20;
            float x = ptr[i], y;
            if (x > MISH_THRESHOLD)
                y = x;
            else if (x < -MISH_THRESHOLD)
                y = expf(x);
            else
                y = logf(expf(x) + 1);
            ptr[i] = x * tanhf(y);
        }
    }

    return 0;
}

} // namespace ncnn
