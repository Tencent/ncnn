// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cumulativesum_powerpc.h"

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

namespace ncnn {

CumulativeSum_powerpc::CumulativeSum_powerpc()
{
    // base defaults kept: one_blob_only = true, support_inplace = true
    // support_packing stays false, only pack1 fp32 is vectorized
}

#if __ALTIVEC__

// inclusive prefix sum within one 4-lane vector
static NCNN_FORCEINLINE v4f32 cumsum_prefix_add(v4f32 _v)
{
    const v4u32 _zero = vec_splats((unsigned int)0);

    // {x0, x0+x1, x1+x2, x2+x3}
    _v = vec_add(_v, (v4f32)vec_sld(_zero, (v4u32)_v, 12));
    // {x0, x0+x1, x0+x1+x2, x0+x1+x2+x3}
    _v = vec_add(_v, (v4f32)vec_sld(_zero, (v4u32)_v, 8));
    return _v;
}

// in-place cumulative sum along a contiguous row of w floats
static void cumsum_row(float* ptr, int w)
{
    int k = 0;
    float carry = 0.f;

    for (; k + 4 <= w; k += 4)
    {
        v4f32 _v = cumsum_prefix_add(__ncnn_altivec_load(ptr + k));
        _v = vec_add(_v, vec_splats(carry));
        __ncnn_altivec_store(ptr + k, _v);

        carry = vec_extract(_v, 3);
    }

    for (; k < w; k++)
    {
        ptr[k] = ptr[k] + carry;
        carry = ptr[k];
    }
}

// this_row += prev_row, contiguous
static void cumsum_add_row(float* this_row, const float* prev_row, int w)
{
    int k = 0;
    for (; k + 4 <= w; k += 4)
    {
        v4f32 _p = __ncnn_altivec_load(prev_row + k);
        v4f32 _t = __ncnn_altivec_load(this_row + k);
        __ncnn_altivec_store(this_row + k, vec_add(_t, _p));
    }
    for (; k < w; k++)
    {
        this_row[k] = this_row[k] + prev_row[k];
    }
}

// cur += prev over a contiguous region of size floats
static void cumsum_add_region(float* cur, const float* prev, int size)
{
    int k = 0;
    for (; k + 4 <= size; k += 4)
    {
        v4f32 _p = __ncnn_altivec_load(prev + k);
        v4f32 _c = __ncnn_altivec_load(cur + k);
        __ncnn_altivec_store(cur + k, vec_add(_c, _p));
    }
    for (; k < size; k++)
    {
        cur[k] = cur[k] + prev[k];
    }
}

#endif // __ALTIVEC__

int CumulativeSum_powerpc::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if __ALTIVEC__
    if (bottom_top_blob.elempack != 1 || bottom_top_blob.elemsize != 4u)
        return CumulativeSum::forward_inplace(bottom_top_blob, opt);

    int dims = bottom_top_blob.dims;
    int positive_axis = axis < 0 ? dims + axis : axis;

    if (dims == 1)
    {   // ignore axis
        int w = bottom_top_blob.w;

        float* ptr = bottom_top_blob;

        cumsum_row(ptr, w);

        return 0;
    } // if (dims == 1)

    if (dims == 2 && positive_axis == 0)
    {
        // sum over rows
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        for (int i = 1; i < h; ++i)
        {
            const float* prev_row = bottom_top_blob.row(i - 1);
            float* this_row = bottom_top_blob.row(i);

            cumsum_add_row(this_row, prev_row, w);
        }

        return 0;
    } // if (dims == 2 && positive_axis == 0)

    if (dims == 2 && positive_axis == 1)
    {
        // sum over columns
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; ++i)
        {
            float* ptr = bottom_top_blob.row(i);

            cumsum_row(ptr, w);
        }

        return 0;
    } // if (dims == 2 && positive_axis == 1)

    if (dims == 3 && positive_axis == 0)
    {
        // sum over channels
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int c = bottom_top_blob.c;

        int size = w * h;

        for (int i = 1; i < c; ++i)
        {
            const float* prev = bottom_top_blob.channel(i - 1);
            float* cur = bottom_top_blob.channel(i);

            cumsum_add_region(cur, prev, size);
        }

        return 0;
    } // if (dims == 3 && positive_axis == 0)

    if (dims == 3 && positive_axis == 1)
    {
        // sum over rows within each channel

        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int c = bottom_top_blob.c;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; ++q)
        {
            Mat this_channel = bottom_top_blob.channel(q);

            for (int i = 1; i < h; ++i)
            {
                const float* prev_row = this_channel.row(i - 1);
                float* this_row = this_channel.row(i);

                cumsum_add_row(this_row, prev_row, w);
            }
        }

        return 0;
    } // if (dims == 3 && positive_axis == 1)

    if (dims == 3 && positive_axis == 2)
    {
        // sum over columns within each channel

        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int c = bottom_top_blob.c;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; ++q)
        {
            Mat this_channel = bottom_top_blob.channel(q);

            for (int i = 0; i < h; ++i)
            {
                float* ptr = this_channel.row(i);
                cumsum_row(ptr, w);
            }
        }

        return 0;
    } // if (dims == 3 && positive_axis == 2)

    if (dims == 4 && positive_axis == 0)
    {
        // sum over channels
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int d = bottom_top_blob.d;
        int c = bottom_top_blob.c;

        int size = w * h * d;

        for (int i = 1; i < c; ++i)
        {
            const float* prev = bottom_top_blob.channel(i - 1);
            float* cur = bottom_top_blob.channel(i);

            cumsum_add_region(cur, prev, size);
        }

        return 0;
    } // if (dims == 4 && positive_axis == 0)

    if (dims == 4 && positive_axis == 1)
    {
        // sum over depth within each channel

        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int d = bottom_top_blob.d;
        int c = bottom_top_blob.c;

        int size = w * h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; ++q)
        {
            Mat this_channel = bottom_top_blob.channel(q);

            for (int z = 1; z < d; ++z)
            {
                const float* prev = this_channel.depth(z - 1);
                float* cur = this_channel.depth(z);

                cumsum_add_region(cur, prev, size);
            }
        }

        return 0;
    } // if (dims == 4 && positive_axis == 1)

    if (dims == 4 && positive_axis == 2)
    {
        // sum over rows within each depth slice

        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int d = bottom_top_blob.d;
        int c = bottom_top_blob.c;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; ++q)
        {
            Mat this_channel = bottom_top_blob.channel(q);

            for (int z = 0; z < d; ++z)
            {
                Mat this_depth = this_channel.depth(z);

                for (int i = 1; i < h; ++i)
                {
                    const float* prev_row = this_depth.row(i - 1);
                    float* this_row = this_depth.row(i);

                    cumsum_add_row(this_row, prev_row, w);
                }
            }
        }

        return 0;
    } // if (dims == 4 && positive_axis == 2)

    if (dims == 4 && positive_axis == 3)
    {
        // sum over columns within each depth slice

        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int d = bottom_top_blob.d;
        int c = bottom_top_blob.c;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; ++q)
        {
            Mat this_channel = bottom_top_blob.channel(q);

            for (int z = 0; z < d; ++z)
            {
                Mat this_depth = this_channel.depth(z);

                for (int i = 0; i < h; ++i)
                {
                    float* ptr = this_depth.row(i);
                    cumsum_row(ptr, w);
                }
            }
        }

        return 0;
    } // if (dims == 4 && positive_axis == 3)

    return -100;
#else
    return CumulativeSum::forward_inplace(bottom_top_blob, opt);
#endif // __ALTIVEC__
}

} // namespace ncnn
