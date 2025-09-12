// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pixelshuffle.h"

namespace ncnn {

PixelShuffle::PixelShuffle()
{
    one_blob_only = true;
    support_inplace = false;
}

int PixelShuffle::load_param(const ParamDict& pd)
{
    upscale_factor = pd.get(0, 1);
    mode = pd.get(1, 0);

    return 0;
}

int PixelShuffle::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    int outw = w * upscale_factor;
    int outh = h * upscale_factor;
    int outc = channels / (upscale_factor * upscale_factor);

    top_blob.create(outw, outh, outc, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outc; p++)
    {
        Mat m = top_blob.channel(p);

        for (int sh = 0; sh < upscale_factor; sh++)
        {
            for (int sw = 0; sw < upscale_factor; sw++)
            {
                int q;
                if (mode == 0)
                    q = p * upscale_factor * upscale_factor + sh * upscale_factor + sw;
                else // if (mode == 1)
                    q = (sh * upscale_factor + sw) * outc + p;

                const float* sptr = bottom_blob.channel(q);

                for (int i = 0; i < h; i++)
                {
                    float* outptr = m.row(i * upscale_factor + sh) + sw;
                    for (int j = 0; j < w; j++)
                    {
                        outptr[0] = sptr[0];

                        sptr++;
                        outptr += upscale_factor;
                    }
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
