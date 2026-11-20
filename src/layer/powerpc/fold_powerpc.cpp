// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "fold_powerpc.h"

#include <limits.h>

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

namespace ncnn {

Fold_powerpc::Fold_powerpc()
{
    // base defaults kept: one_blob_only = true
    // support_packing stays false, the generic col2im layout is reused and the
    // contiguous inner accumulation is vectorized when stride_w == 1
}

int Fold_powerpc::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if __ALTIVEC__
    if (bottom_blob.elemsize != 4u || bottom_blob.elempack != 1)
        return Fold::forward(bottom_blob, top_blob, opt);

    const int max_channels = bottom_blob.h;
    size_t elemsize = bottom_blob.elemsize;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    const int outw = output_w + pad_left + pad_right;
    const int outh = output_h + pad_top + pad_bottom;

    const int inw = (outw - kernel_extent_w) / stride_w + 1;
    const int inh = (outh - kernel_extent_h) / stride_h + 1;

    // assert inw * inh == size

    const int maxk = kernel_w * kernel_h;
    const int channels = max_channels / maxk;

    Mat top_blob_bordered;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0)
    {
        top_blob_bordered.create(outw, outh, channels, elemsize, opt.workspace_allocator);
    }
    else
    {
        top_blob_bordered = top_blob;
        top_blob_bordered.create(outw, outh, channels, elemsize, opt.blob_allocator);
    }
    if (top_blob_bordered.empty())
        return -100;

    // col2im
    const int gap = outw * stride_h - inw * stride_w;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < channels; p++)
    {
        const float* sptr = bottom_blob.row(p * maxk);
        Mat outm = top_blob_bordered.channel(p);

        outm.fill(0.f);

        for (int u = 0; u < kernel_h; u++)
        {
            for (int v = 0; v < kernel_w; v++)
            {
                float* ptr = outm.row(dilation_h * u) + dilation_w * v;

                if (stride_w == 1)
                {
                    // both source and destination advance contiguously
                    for (int i = 0; i < inh; i++)
                    {
                        int j = 0;
                        for (; j + 4 <= inw; j += 4)
                        {
                            v4f32 _p = __ncnn_altivec_load(ptr);
                            v4f32 _s = __ncnn_altivec_load(sptr);
                            __ncnn_altivec_store(ptr, vec_add(_p, _s));

                            ptr += 4;
                            sptr += 4;
                        }
                        for (; j < inw; j++)
                        {
                            ptr[0] += sptr[0];

                            ptr += 1;
                            sptr += 1;
                        }

                        ptr += gap;
                    }
                }
                else
                {
                    for (int i = 0; i < inh; i++)
                    {
                        for (int j = 0; j < inw; j++)
                        {
                            ptr[0] += sptr[0];

                            ptr += stride_w;
                            sptr += 1;
                        }

                        ptr += gap;
                    }
                }
            }
        }
    }

    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0)
    {
        Option opt_b = opt;
        opt_b.use_packing_layout = false;
        copy_cut_border(top_blob_bordered, top_blob, pad_top, pad_bottom, pad_left, pad_right, opt_b);
        if (top_blob.empty())
            return -100;
    }
    else
    {
        top_blob = top_blob_bordered;
    }

    return 0;
#else
    return Fold::forward(bottom_blob, top_blob, opt);
#endif // __ALTIVEC__
}

} // namespace ncnn
