// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pixelshuffle_powerpc.h"

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

namespace ncnn {

PixelShuffle_powerpc::PixelShuffle_powerpc()
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

int PixelShuffle_powerpc::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (bottom_blob.elempack != 1)
    {
        // pixelshuffle changes the shape, so the packing axis of the input and
        // output can differ; operate on the unpacked layout and re-pack afterwards
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
        return PixelShuffle::forward(bottom_blob, top_blob, opt);
    }

#if __ALTIVEC__
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

    if (upscale_factor == 1)
    {
        // identity, the innermost copy is contiguous, vectorize
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outc; p++)
        {
            const float* sptr = bottom_blob.channel(p);
            float* outptr = top_blob.channel(p);

            copy_vec(outptr, sptr, w * h);
        }

        return 0;
    }

    // generic index math, the innermost copy is strided by upscale_factor
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
#else
    // no altivec -> base implementation
    return PixelShuffle::forward(bottom_blob, top_blob, opt);
#endif // __ALTIVEC__
}

} // namespace ncnn
