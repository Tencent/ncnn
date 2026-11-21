// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "permute_powerpc.h"

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

namespace ncnn {

Permute_powerpc::Permute_powerpc()
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

int Permute_powerpc::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (bottom_blob.elempack != 1)
    {
        // permute of packed data may move the packing axis; operate on the
        // unpacked layout and re-pack the outermost dim of the result
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

        int repack_count = 0;
        if (top_blob_unpacked.dims == 1)
            repack_count = top_blob_unpacked.w;
        if (top_blob_unpacked.dims == 2)
            repack_count = top_blob_unpacked.h;
        if (top_blob_unpacked.dims == 3 || top_blob_unpacked.dims == 4)
            repack_count = top_blob_unpacked.c;

        if (opt.use_packing_layout && repack_count % 4 == 0)
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
        return Permute::forward(bottom_blob, top_blob, opt);
    }

#if __ALTIVEC__
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    int dims = bottom_blob.dims;

    if (dims == 1)
    {
        top_blob = bottom_blob;
        return 0;
    }

    if (dims == 2 && order_type == 0)
    {
        top_blob = bottom_blob;
        return 0;
    }

    if (dims == 3 && order_type == 2)
    {
        // w c h -> the w copy is contiguous, vectorize
        top_blob.create(w, channels, h, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < h; q++)
        {
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < channels; i++)
            {
                const float* ptr = bottom_blob.channel(i).row(q);

                copy_vec(outptr, ptr, w);

                outptr += w;
            }
        }

        return 0;
    }

    if (dims == 4 && order_type == 2)
    {
        // w d h c -> the w copy is contiguous, vectorize
        top_blob.create(w, d, h, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* outptr = top_blob.channel(q);

            for (int z = 0; z < h; z++)
            {
                for (int i = 0; i < d; i++)
                {
                    const float* ptr = bottom_blob.channel(q).depth(i).row(z);

                    copy_vec(outptr, ptr, w);

                    outptr += w;
                }
            }
        }

        return 0;
    }

    if (dims == 4 && order_type == 6)
    {
        // w h c d -> the w copy is contiguous, vectorize
        top_blob.create(w, h, channels, d, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < d; q++)
        {
            float* outptr = top_blob.channel(q);

            for (int z = 0; z < channels; z++)
            {
                for (int i = 0; i < h; i++)
                {
                    const float* ptr = bottom_blob.channel(z).depth(q).row(i);

                    copy_vec(outptr, ptr, w);

                    outptr += w;
                }
            }
        }

        return 0;
    }

    if (dims == 4 && order_type == 8)
    {
        // w c h d -> the w copy is contiguous, vectorize
        top_blob.create(w, channels, h, d, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < d; q++)
        {
            float* outptr = top_blob.channel(q);

            for (int z = 0; z < h; z++)
            {
                for (int i = 0; i < channels; i++)
                {
                    const float* ptr = bottom_blob.channel(i).depth(q).row(z);

                    copy_vec(outptr, ptr, w);

                    outptr += w;
                }
            }
        }

        return 0;
    }

    if (dims == 4 && order_type == 12)
    {
        // w d c h -> the w copy is contiguous, vectorize
        top_blob.create(w, d, channels, h, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < h; q++)
        {
            float* outptr = top_blob.channel(q);

            for (int z = 0; z < channels; z++)
            {
                for (int i = 0; i < d; i++)
                {
                    const float* ptr = bottom_blob.channel(z).depth(i).row(q);

                    copy_vec(outptr, ptr, w);

                    outptr += w;
                }
            }
        }

        return 0;
    }

    if (dims == 4 && order_type == 14)
    {
        // w c d h -> the w copy is contiguous, vectorize
        top_blob.create(w, channels, d, h, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < h; q++)
        {
            float* outptr = top_blob.channel(q);

            for (int z = 0; z < d; z++)
            {
                for (int i = 0; i < channels; i++)
                {
                    const float* ptr = bottom_blob.channel(i).depth(z).row(q);

                    copy_vec(outptr, ptr, w);

                    outptr += w;
                }
            }
        }

        return 0;
    }

    // all other orders gather along a stride, the generic index math applies
    return Permute::forward(bottom_blob, top_blob, opt);
#else
    // no altivec -> base implementation
    return Permute::forward(bottom_blob, top_blob, opt);
#endif // __ALTIVEC__
}

} // namespace ncnn
