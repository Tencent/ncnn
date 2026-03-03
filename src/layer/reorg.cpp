// Copyright 2018 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "reorg.h"

namespace ncnn {

Reorg::Reorg()
{
    one_blob_only = true;
    support_inplace = false;
}

int Reorg::load_param(const ParamDict& pd)
{
    stride = pd.get(0, 1);
    mode = pd.get(1, 0);

    return 0;
}

int Reorg::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    int outw = w / stride;
    int outh = h / stride;
    int outc = channels * stride * stride;

    top_blob.create(outw, outh, outc, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const Mat m = bottom_blob.channel(q);

        for (int sh = 0; sh < stride; sh++)
        {
            for (int sw = 0; sw < stride; sw++)
            {
                int p;
                if (mode == 0)
                    p = q * stride * stride + sh * stride + sw;
                else // if (mode == 1)
                    p = (sh * stride + sw) * channels + q;

                float* outptr = top_blob.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    const float* sptr = m.row(i * stride + sh) + sw;
                    for (int j = 0; j < outw; j++)
                    {
                        outptr[0] = sptr[0];

                        sptr += stride;
                        outptr++;
                    }
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
