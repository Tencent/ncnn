// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "reorg_powerpc.h"

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

namespace ncnn {

Reorg_powerpc::Reorg_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
#endif // __ALTIVEC__
}

#if __ALTIVEC__
// vectorized contiguous copy of n floats
static void copy_vec(float* dst, const float* src, int n)
{
    int i = 0;
    for (; i + 4 <= n; i += 4)
    {
        v4f32 _v = __ncnn_altivec_load(src + i);
        __ncnn_altivec_store(dst + i, _v);
    }
    for (; i < n; i++)
    {
        dst[i] = src[i];
    }
}
#endif // __ALTIVEC__

int Reorg_powerpc::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (bottom_blob.elempack != 1)
    {
        // reorg changes the shape, so the packing axis of the input and output
        // can differ; operate on the unpacked layout and re-pack afterwards
        Option opt_pack1 = opt;
        opt_pack1.blob_allocator = opt.workspace_allocator;

        Mat bottom_blob_unpacked;
        convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack1);
        if (bottom_blob_unpacked.empty())
            return -100;

        Mat top_blob_unpacked;
        int ret = forward(bottom_blob_unpacked, top_blob_unpacked, opt);
        if (ret != 0)
            return ret;

        if (opt.use_packing_layout && top_blob_unpacked.c % 4 == 0)
        {
            Mat top_blob_packed;
            convert_packing(top_blob_unpacked, top_blob_packed, 4, opt);
            if (!top_blob_packed.empty())
            {
                top_blob = top_blob_packed;
                return 0;
            }
        }

        top_blob = top_blob_unpacked;
        return 0;
    }

    if (bottom_blob.elembits() != 32)
    {
        // int8 / bf16 -> base implementation
        return Reorg::forward(bottom_blob, top_blob, opt);
    }

#if __ALTIVEC__
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

    if (stride == 1)
    {
        // identity, the innermost copy is contiguous, vectorize
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* sptr = bottom_blob.channel(q);
            float* outptr = top_blob.channel(q);

            copy_vec(outptr, sptr, w * h);
        }

        return 0;
    }

    // generic index math, the innermost read is strided by stride
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
#else
    // no altivec -> base implementation
    return Reorg::forward(bottom_blob, top_blob, opt);
#endif // __ALTIVEC__
}

} // namespace ncnn
