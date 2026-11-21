// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "unfold_powerpc.h"

#include <limits.h>

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

namespace ncnn {

Unfold_powerpc::Unfold_powerpc()
{
    // base defaults kept: one_blob_only = true
    // support_packing stays false, the generic im2col layout is reused and the
    // contiguous inner copy is vectorized when stride_w == 1
}

int Unfold_powerpc::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if __ALTIVEC__
    if (bottom_blob.elemsize != 4u || bottom_blob.elempack != 1)
        return Unfold::forward(bottom_blob, top_blob, opt);

    Mat bottom_blob_bordered;
    {
        Option opt_b = opt;
        opt_b.blob_allocator = opt.workspace_allocator;
        opt_b.use_packing_layout = false;
        make_padding(bottom_blob, bottom_blob_bordered, opt_b);
        if (bottom_blob_bordered.empty())
            return -100;
    }

    const int w = bottom_blob_bordered.w;
    const int h = bottom_blob_bordered.h;
    const int channels = bottom_blob_bordered.c;
    const size_t elemsize = bottom_blob_bordered.elemsize;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    const int outw = (w - kernel_extent_w) / stride_w + 1;
    const int outh = (h - kernel_extent_h) / stride_h + 1;

    const int size = outw * outh;
    const int maxk = kernel_w * kernel_h;

    top_blob.create(size, maxk * channels, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // im2col
    const int gap = w * stride_h - outw * stride_w;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < channels; p++)
    {
        const Mat img = bottom_blob_bordered.channel(p);
        float* ptr = top_blob.row(p * maxk);

        for (int u = 0; u < kernel_h; u++)
        {
            for (int v = 0; v < kernel_w; v++)
            {
                const float* sptr = img.row(dilation_h * u) + dilation_w * v;

                if (stride_w == 1)
                {
                    // both source and destination advance contiguously
                    for (int i = 0; i < outh; i++)
                    {
                        int j = 0;
                        for (; j + 4 <= outw; j += 4)
                        {
                            v4f32 _s = __ncnn_altivec_load(sptr);
                            __ncnn_altivec_store(ptr, _s);

                            sptr += 4;
                            ptr += 4;
                        }
                        for (; j < outw; j++)
                        {
                            ptr[0] = sptr[0];

                            sptr += 1;
                            ptr += 1;
                        }

                        sptr += gap;
                    }
                }
                else
                {
                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            ptr[0] = sptr[0];

                            sptr += stride_w;
                            ptr += 1;
                        }

                        sptr += gap;
                    }
                }
            }
        }
    }

    return 0;
#else
    return Unfold::forward(bottom_blob, top_blob, opt);
#endif // __ALTIVEC__
}

} // namespace ncnn
