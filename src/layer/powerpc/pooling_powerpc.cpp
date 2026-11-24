// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pooling_powerpc.h"

#include <algorithm>
#include <float.h>
#include <vector>

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

namespace ncnn {

Pooling_powerpc::Pooling_powerpc()
{
#if __ALTIVEC__
    support_packing = true;
#endif // __ALTIVEC__
}

int Pooling_powerpc::create_pipeline(const Option& /*opt*/)
{
    if (adaptive_pooling)
    {
        support_packing = false;
    }

    return 0;
}

#if __ALTIVEC__

// max of every adjacent float pair of two 4-lane vectors
// {a0,a1,a2,a3} , {a4,a5,a6,a7} -> {max(a0,a1), max(a2,a3), max(a4,a5), max(a6,a7)}
static NCNN_FORCEINLINE v4f32 pooling2x2s2_hmax(v4f32 _r0, v4f32 _r1)
{
    v4f32 _m0 = vec_max(__ncnn_altivec_even_lanes(_r0), __ncnn_altivec_odd_lanes(_r0));
    v4f32 _m1 = vec_max(__ncnn_altivec_even_lanes(_r1), __ncnn_altivec_odd_lanes(_r1));

    const v16u8 _perm = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
    return vec_perm(_m0, _m1, _perm);
}

// r0 = cols s..s+3, r1 = cols s+4..s+7, r2 = cols s+5..s+8
// produce max over the four 3-wide windows starting at s, s+2, s+4, s+6
static NCNN_FORCEINLINE v4f32 pooling3x3s2_hmax(v4f32 _r0, v4f32 _r1, v4f32 _r2)
{
    // window starts, lane wise
    const v16u8 _perm_x = {0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27};
    const v16u8 _perm_y = {4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31};
    const v16u8 _perm_t = {8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 0, 1, 2, 3};
    const v16u8 _perm_z = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 28, 29, 30, 31};

    // {r0[0], r0[2], r1[0], r1[2]}
    v4f32 _x = vec_perm(_r0, _r1, _perm_x);
    // {r0[1], r0[3], r1[1], r1[3]}
    v4f32 _y = vec_perm(_r0, _r1, _perm_y);
    // {r0[2], r1[0], r1[2], r2[3]}
    v4f32 _t = vec_perm(_r0, _r1, _perm_t);
    v4f32 _z = vec_perm(_t, _r2, _perm_z);

    return vec_max(vec_max(_x, _y), _z);
}

static void pooling2x2s2_max_altivec(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int tailstep = w - 2 * outw + w;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < inch; q++)
    {
        const float* img0 = bottom_blob.channel(q);
        float* outptr = top_blob.channel(q);

        const float* r0 = img0;
        const float* r1 = img0 + w;

        for (int i = 0; i < outh; i++)
        {
            int j = 0;
            for (; j + 3 < outw; j += 4)
            {
                v4f32 _r0 = vec_max(__ncnn_altivec_load(r0), __ncnn_altivec_load(r1));
                v4f32 _r1 = vec_max(__ncnn_altivec_load(r0 + 4), __ncnn_altivec_load(r1 + 4));

                v4f32 _max = pooling2x2s2_hmax(_r0, _r1);

                __ncnn_altivec_store(outptr, _max);

                r0 += 8;
                r1 += 8;
                outptr += 4;
            }

            for (; j < outw; j++)
            {
                float max0 = std::max(r0[0], r0[1]);
                float max1 = std::max(r1[0], r1[1]);

                *outptr = std::max(max0, max1);

                r0 += 2;
                r1 += 2;
                outptr++;
            }

            r0 += tailstep;
            r1 += tailstep;
        }
    }
}

static void pooling3x3s2_max_altivec(const Mat& bottom_blob, Mat& top_blob, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int tailstep = w - 2 * outw + w;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < inch; q++)
    {
        const float* img0 = bottom_blob.channel(q);
        float* outptr = top_blob.channel(q);

        const float* r0 = img0;
        const float* r1 = img0 + w;
        const float* r2 = img0 + w * 2;

        for (int i = 0; i < outh; i++)
        {
            int j = 0;
            for (; j + 3 < outw; j += 4)
            {
                v4f32 _max = pooling3x3s2_hmax(__ncnn_altivec_load(r0), __ncnn_altivec_load(r0 + 4), __ncnn_altivec_load(r0 + 5));
                _max = vec_max(_max, pooling3x3s2_hmax(__ncnn_altivec_load(r1), __ncnn_altivec_load(r1 + 4), __ncnn_altivec_load(r1 + 5)));
                _max = vec_max(_max, pooling3x3s2_hmax(__ncnn_altivec_load(r2), __ncnn_altivec_load(r2 + 4), __ncnn_altivec_load(r2 + 5)));

                __ncnn_altivec_store(outptr, _max);

                r0 += 8;
                r1 += 8;
                r2 += 8;
                outptr += 4;
            }

            for (; j < outw; j++)
            {
                float max0 = std::max(std::max(r0[0], r0[1]), r0[2]);
                float max1 = std::max(std::max(r1[0], r1[1]), r1[2]);
                float max2 = std::max(std::max(r2[0], r2[1]), r2[2]);

                *outptr = std::max(std::max(max0, max1), max2);

                r0 += 2;
                r1 += 2;
                r2 += 2;
                outptr++;
            }

            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
        }
    }
}

// the generic Padding layer does not handle packed mats, so pad pack4 data here
// the pad extent calculation mirrors Pooling::make_padding
static void make_padding_pack4(const Mat& bottom_blob, Mat& bottom_blob_bordered,
                               int kernel_w, int kernel_h, int stride_w, int stride_h,
                               int pad_left, int pad_right, int pad_top, int pad_bottom,
                               int pad_mode, float pad_value, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;

    int wtailpad = 0;
    int htailpad = 0;

    if (pad_mode == 0) // full padding
    {
        int wtail = (w + pad_left + pad_right - kernel_w) % stride_w;
        int htail = (h + pad_top + pad_bottom - kernel_h) % stride_h;

        if (wtail != 0)
            wtailpad = stride_w - wtail;
        if (htail != 0)
            htailpad = stride_h - htail;
    }
    else if (pad_mode == 2) // SAME_UPPER
    {
        int wpad = kernel_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_h + (h - 1) / stride_h * stride_h - h;

        pad_top = hpad / 2;
        pad_bottom = hpad - hpad / 2;
        pad_left = wpad / 2;
        pad_right = wpad - wpad / 2;
    }
    else if (pad_mode == 3) // SAME_LOWER
    {
        int wpad = kernel_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_h + (h - 1) / stride_h * stride_h - h;

        pad_top = hpad - hpad / 2;
        pad_bottom = hpad / 2;
        pad_left = wpad - wpad / 2;
        pad_right = wpad / 2;
    }

    if (pad_left == 0 && pad_right == 0 && pad_top == 0 && pad_bottom == 0 && wtailpad == 0 && htailpad == 0)
    {
        bottom_blob_bordered = bottom_blob;
        return;
    }

    int outw = w + pad_left + pad_right + wtailpad;
    int outh = h + pad_top + pad_bottom + htailpad;

    bottom_blob_bordered.create(outw, outh, channels, bottom_blob.elemsize, 4, opt.workspace_allocator);
    if (bottom_blob_bordered.empty())
        return;

    v4f32 _pad = vec_splats(pad_value);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* ptr = bottom_blob.channel(q);
        float* outptr = bottom_blob_bordered.channel(q);

        for (int y = 0; y < outh; y++)
        {
            int sy = y - pad_top;

            if (sy < 0 || sy >= h)
            {
                for (int x = 0; x < outw; x++)
                {
                    __ncnn_altivec_store(outptr + x * 4, _pad);
                }
            }
            else
            {
                const float* sptr = ptr + sy * w * 4;

                for (int x = 0; x < pad_left; x++)
                {
                    __ncnn_altivec_store(outptr + x * 4, _pad);
                }

                for (int x = 0; x < w; x++)
                {
                    __ncnn_altivec_store(outptr + (pad_left + x) * 4, __ncnn_altivec_load(sptr + x * 4));
                }

                for (int x = pad_left + w; x < outw; x++)
                {
                    __ncnn_altivec_store(outptr + x * 4, _pad);
                }
            }

            outptr += outw * 4;
        }
    }
}

#endif // __ALTIVEC__

int Pooling_powerpc::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // max value in NxN window
    // avg value in NxN window

    if (adaptive_pooling)
    {
        return Pooling::forward(bottom_blob, top_blob, opt);
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

#if __ALTIVEC__
    if (global_pooling && elempack == 4 && elemsize == 16u)
    {
        top_blob.create(channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        int size = w * h;

        if (pooling_type == PoolMethod_MAX)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);

                v4f32 _max = __ncnn_altivec_load(ptr);
                for (int i = 1; i < size; i++)
                {
                    _max = vec_max(_max, __ncnn_altivec_load(ptr + i * 4));
                }

                float* outptr = top_blob;
                __ncnn_altivec_store(outptr + q * 4, _max);
            }
        }
        else if (pooling_type == PoolMethod_AVE)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);

                v4f32 _sum = vec_splats(0.f);
                for (int i = 0; i < size; i++)
                {
                    _sum = vec_add(_sum, __ncnn_altivec_load(ptr + i * 4));
                }

                v4f32 _avg = __ncnn_altivec_mul(_sum, vec_splats(1.f / size));

                float* outptr = top_blob;
                __ncnn_altivec_store(outptr + q * 4, _avg);
            }
        }

        return 0;
    }

    if (global_pooling && elempack == 1 && elemsize == 4u)
    {
        top_blob.create(channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        int size = w * h;

        if (pooling_type == PoolMethod_MAX)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = top_blob;

                float max = ptr[0];
                int i = 0;
                if (size >= 4)
                {
                    v4f32 _max = __ncnn_altivec_load(ptr);
                    i = 4;
                    for (; i + 3 < size; i += 4)
                    {
                        _max = vec_max(_max, __ncnn_altivec_load(ptr + i));
                    }
                    max = __ncnn_altivec_reduce_max(_max);
                }
                for (; i < size; i++)
                {
                    max = std::max(max, ptr[i]);
                }

                outptr[q] = max;
            }
        }
        else if (pooling_type == PoolMethod_AVE)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                float* outptr = top_blob;

                int i = 0;
                v4f32 _sum = vec_splats(0.f);
                for (; i + 3 < size; i += 4)
                {
                    _sum = vec_add(_sum, __ncnn_altivec_load(ptr + i));
                }
                float sum = __ncnn_altivec_reduce_add(_sum);
                for (; i < size; i++)
                {
                    sum += ptr[i];
                }

                outptr[q] = sum / size;
            }
        }

        return 0;
    }

    if (elempack == 1 && elemsize == 4u && pooling_type == PoolMethod_MAX && !global_pooling && stride_w == 2 && stride_h == 2 && ((kernel_w == 2 && kernel_h == 2) || (kernel_w == 3 && kernel_h == 3)))
    {
        Mat bottom_blob_bordered;
        make_padding(bottom_blob, bottom_blob_bordered, opt);
        if (bottom_blob_bordered.empty())
            return -100;

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;

        int outw = (w - kernel_w) / stride_w + 1;
        int outh = (h - kernel_h) / stride_h + 1;

        top_blob.create(outw, outh, channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (kernel_w == 2)
            pooling2x2s2_max_altivec(bottom_blob_bordered, top_blob, opt);
        if (kernel_w == 3)
            pooling3x3s2_max_altivec(bottom_blob_bordered, top_blob, opt);

        return 0;
    }

    if (elempack == 4 && elemsize == 16u && !global_pooling)
    {
        Mat bottom_blob_bordered;
        float pad_value = pooling_type == PoolMethod_MAX ? -FLT_MAX : 0.f;
        make_padding_pack4(bottom_blob, bottom_blob_bordered, kernel_w, kernel_h, stride_w, stride_h, pad_left, pad_right, pad_top, pad_bottom, pad_mode, pad_value, opt);
        if (bottom_blob_bordered.empty())
            return -100;

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;

        int outw = (w - kernel_w) / stride_w + 1;
        int outh = (h - kernel_h) / stride_h + 1;

        top_blob.create(outw, outh, channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int maxk = kernel_w * kernel_h;

        // kernel offsets
        std::vector<int> _space_ofs(maxk);
        int* space_ofs = &_space_ofs[0];
        {
            int p1 = 0;
            int p2 = 0;
            int gap = w - kernel_w;
            for (int i = 0; i < kernel_h; i++)
            {
                for (int j = 0; j < kernel_w; j++)
                {
                    space_ofs[p1] = p2;
                    p1++;
                    p2++;
                }
                p2 += gap;
            }
        }

        if (pooling_type == PoolMethod_MAX)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob_bordered.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        const float* sptr = m.row(i * stride_h) + j * stride_w * 4;

                        v4f32 _max = __ncnn_altivec_load(sptr);

                        for (int k = 0; k < maxk; k++)
                        {
                            v4f32 _val = __ncnn_altivec_load(sptr + space_ofs[k] * 4);
                            _max = vec_max(_max, _val);
                        }

                        __ncnn_altivec_store(outptr + j * 4, _max);
                    }

                    outptr += outw * 4;
                }
            }
        }
        else if (pooling_type == PoolMethod_AVE)
        {
            if (avgpool_count_include_pad == 0)
            {
                int wtailpad = 0;
                int htailpad = 0;

                if (pad_mode == 0) // full padding
                {
                    wtailpad = bottom_blob_bordered.w - bottom_blob.w - pad_left - pad_right;
                    htailpad = bottom_blob_bordered.h - bottom_blob.h - pad_top - pad_bottom;
                }

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob_bordered.channel(q);
                    float* outptr = top_blob.channel(q);

                    for (int i = 0; i < outh; i++)
                    {
                        int sy0 = i * stride_h;

                        for (int j = 0; j < outw; j++)
                        {
                            int sx0 = j * stride_w;

                            v4f32 _sum = vec_splats(0.f);
                            int area = 0;

                            for (int ki = 0; ki < kernel_h; ki++)
                            {
                                int sy = sy0 + ki;

                                if (sy < pad_top)
                                    continue;

                                if (sy >= h - pad_bottom - htailpad)
                                    break;

                                for (int kj = 0; kj < kernel_w; kj++)
                                {
                                    int sx = sx0 + kj;

                                    if (sx < pad_left)
                                        continue;

                                    if (sx >= w - pad_right - wtailpad)
                                        break;

                                    v4f32 _val = __ncnn_altivec_load(m.row(sy) + sx * 4);
                                    _sum = vec_add(_sum, _val);
                                    area += 1;
                                }
                            }

                            v4f32 _avg = __ncnn_altivec_mul(_sum, vec_splats(1.f / area));
                            __ncnn_altivec_store(outptr + j * 4, _avg);
                        }

                        outptr += outw * 4;
                    }
                }
            }
            else // if (avgpool_count_include_pad == 1)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob_bordered.channel(q);
                    float* outptr = top_blob.channel(q);

                    const float inv_maxk = 1.f / maxk;

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            const float* sptr = m.row(i * stride_h) + j * stride_w * 4;

                            v4f32 _sum = vec_splats(0.f);

                            for (int k = 0; k < maxk; k++)
                            {
                                v4f32 _val = __ncnn_altivec_load(sptr + space_ofs[k] * 4);
                                _sum = vec_add(_sum, _val);
                            }

                            v4f32 _avg = __ncnn_altivec_mul(_sum, vec_splats(inv_maxk));
                            __ncnn_altivec_store(outptr + j * 4, _avg);
                        }

                        outptr += outw * 4;
                    }
                }
            }
        }

        return 0;
    }
#endif // __ALTIVEC__

    return Pooling::forward(bottom_blob, top_blob, opt);
}

} // namespace ncnn
