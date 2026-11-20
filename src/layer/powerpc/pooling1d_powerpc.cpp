// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pooling1d_powerpc.h"

#include <algorithm>
#include <float.h>

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

namespace ncnn {

Pooling1D_powerpc::Pooling1D_powerpc()
{
    // base defaults kept: one_blob_only = true, support_inplace = false
    // the base class is not packing aware, support_packing stays false
}

#if __ALTIVEC__

static void global_max_altivec(const Mat& bottom_blob, Mat& top_blob, int w, int h, int num_threads)
{
    #pragma omp parallel for num_threads(num_threads)
    for (int q = 0; q < h; q++)
    {
        const float* ptr = bottom_blob.row(q);

        int i = 0;
        float max = ptr[0];
        if (w >= 4)
        {
            v4f32 _max = __ncnn_altivec_load(ptr);
            i = 4;
            for (; i + 3 < w; i += 4)
            {
                _max = vec_max(_max, __ncnn_altivec_load(ptr + i));
            }
            max = __ncnn_altivec_reduce_max(_max);
        }
        for (; i < w; i++)
        {
            max = std::max(max, ptr[i]);
        }

        top_blob[q] = max;
    }
}

static void global_avg_altivec(const Mat& bottom_blob, Mat& top_blob, int w, int h, int num_threads)
{
    #pragma omp parallel for num_threads(num_threads)
    for (int q = 0; q < h; q++)
    {
        const float* ptr = bottom_blob.row(q);

        int i = 0;
        v4f32 _sum = vec_splats(0.f);
        for (; i + 3 < w; i += 4)
        {
            _sum = vec_add(_sum, __ncnn_altivec_load(ptr + i));
        }
        float sum = __ncnn_altivec_reduce_add(_sum);
        for (; i < w; i++)
        {
            sum += ptr[i];
        }

        top_blob[q] = sum / w;
    }
}

#endif // __ALTIVEC__

int Pooling1D_powerpc::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (bottom_blob.elembits() != 32)
    {
        // int8 / bf16, fall back to the generic implementation
        return Pooling1D::forward(bottom_blob, top_blob, opt);
    }

#if __ALTIVEC__
    if (bottom_blob.elempack == 1)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        size_t elemsize = bottom_blob.elemsize;

        if (global_pooling)
        {
            top_blob.create(h, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (pooling_type == PoolMethod_MAX)
            {
                global_max_altivec(bottom_blob, top_blob, w, h, opt.num_threads);
            }
            else if (pooling_type == PoolMethod_AVE)
            {
                global_avg_altivec(bottom_blob, top_blob, w, h, opt.num_threads);
            }

            return 0;
        }

        if (adaptive_pooling)
        {
            top_blob.create(out_w, h, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (pooling_type == PoolMethod_MAX)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < h; q++)
                {
                    const float* inptr = bottom_blob.row(q);
                    float* outptr = top_blob.row(q);

                    for (int j = 0; j < out_w; j++)
                    {
                        // floor div
                        const int iw0 = w * j / out_w;
                        // ceil div
                        const int iw1 = (w * (j + 1) + out_w - 1) / out_w;
                        const int wk = iw1 - iw0;

                        int iw = iw0;
                        float max = inptr[iw0];
                        if (wk >= 4)
                        {
                            v4f32 _max = __ncnn_altivec_load(inptr + iw0);
                            iw = iw0 + 4;
                            for (; iw + 3 < iw1; iw += 4)
                            {
                                _max = vec_max(_max, __ncnn_altivec_load(inptr + iw));
                            }
                            max = __ncnn_altivec_reduce_max(_max);
                        }
                        for (; iw < iw1; iw++)
                        {
                            max = std::max(max, inptr[iw]);
                        }

                        outptr[j] = max;
                    }
                }
            }
            else if (pooling_type == PoolMethod_AVE)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < h; q++)
                {
                    const float* inptr = bottom_blob.row(q);
                    float* outptr = top_blob.row(q);

                    for (int j = 0; j < out_w; j++)
                    {
                        // floor div
                        const int iw0 = w * j / out_w;
                        // ceil div
                        const int iw1 = (w * (j + 1) + out_w - 1) / out_w;
                        const int wk = iw1 - iw0;

                        int iw = iw0;
                        v4f32 _sum = vec_splats(0.f);
                        for (; iw + 3 < iw1; iw += 4)
                        {
                            _sum = vec_add(_sum, __ncnn_altivec_load(inptr + iw));
                        }
                        float sum = __ncnn_altivec_reduce_add(_sum);
                        for (; iw < iw1; iw++)
                        {
                            sum += inptr[iw];
                        }

                        outptr[j] = sum / wk;
                    }
                }
            }

            return 0;
        }

        Mat bottom_blob_bordered;
        make_padding(bottom_blob, bottom_blob_bordered, opt);
        if (bottom_blob_bordered.empty())
            return -100;

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;

        int outw = (w - kernel_w) / stride_w + 1;

        top_blob.create(outw, h, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (pooling_type == PoolMethod_MAX)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < h; q++)
            {
                const float* ptr = bottom_blob_bordered.row(q);
                float* outptr = top_blob.row(q);

                for (int j = 0; j < outw; j++)
                {
                    const float* sptr = ptr + j * stride_w;

                    float max = sptr[0];

                    int k = 0;
                    if (kernel_w >= 4)
                    {
                        v4f32 _max = __ncnn_altivec_load(sptr);
                        k = 4;
                        for (; k + 3 < kernel_w; k += 4)
                        {
                            _max = vec_max(_max, __ncnn_altivec_load(sptr + k));
                        }
                        max = __ncnn_altivec_reduce_max(_max);
                    }
                    for (; k < kernel_w; k++)
                    {
                        float val = sptr[k];
                        max = std::max(max, val);
                    }

                    outptr[j] = max;
                }
            }
        }
        else if (pooling_type == PoolMethod_AVE)
        {
            if (avgpool_count_include_pad == 0)
            {
                int wtailpad = 0;

                if (pad_mode == 0) // full padding
                {
                    wtailpad = bottom_blob_bordered.w - bottom_blob.w - pad_left - pad_right;
                }

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < h; q++)
                {
                    const float* ptr = bottom_blob_bordered.row(q);
                    float* outptr = top_blob.row(q);

                    for (int j = 0; j < outw; j++)
                    {
                        int sx0 = j * stride_w;

                        // the generic skip / break conditions reduce to
                        // sx in [max(sx0, pad_left), min(sx0 + kernel_w, w - pad_right - wtailpad))
                        int sx_lo = sx0 > pad_left ? sx0 : pad_left;
                        int sx_hi = sx0 + kernel_w;
                        int sx_bound = w - pad_right - wtailpad;
                        if (sx_hi > sx_bound)
                            sx_hi = sx_bound;

                        float sum = 0.f;
                        int sx = sx_lo;
                        if (sx_hi > sx_lo)
                        {
                            v4f32 _sum = vec_splats(0.f);
                            for (; sx + 3 < sx_hi; sx += 4)
                            {
                                _sum = vec_add(_sum, __ncnn_altivec_load(ptr + sx));
                            }
                            sum = __ncnn_altivec_reduce_add(_sum);
                        }
                        for (; sx < sx_hi; sx++)
                        {
                            sum += ptr[sx];
                        }
                        int area = sx_hi > sx_lo ? sx_hi - sx_lo : 0;

                        outptr[j] = sum / area;
                    }
                }
            }
            else // if (avgpool_count_include_pad == 1)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < h; q++)
                {
                    const float* ptr = bottom_blob_bordered.row(q);
                    float* outptr = top_blob.row(q);

                    for (int j = 0; j < outw; j++)
                    {
                        const float* sptr = ptr + j * stride_w;

                        float sum = 0.f;

                        int k = 0;
                        if (kernel_w >= 4)
                        {
                            v4f32 _sum = vec_splats(0.f);
                            for (; k + 3 < kernel_w; k += 4)
                            {
                                _sum = vec_add(_sum, __ncnn_altivec_load(sptr + k));
                            }
                            sum = __ncnn_altivec_reduce_add(_sum);
                        }
                        for (; k < kernel_w; k++)
                        {
                            float val = sptr[k];
                            sum += val;
                        }

                        outptr[j] = sum / kernel_w;
                    }
                }
            }
        }

        return 0;
    }

    // packed fp32 input, unpack and use the generic implementation
    if (bottom_blob.elempack != 1)
    {
        Mat bottom_blob_unpacked;
        convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt);
        if (bottom_blob_unpacked.empty())
            return -100;

        return Pooling1D::forward(bottom_blob_unpacked, top_blob, opt);
    }
#endif // __ALTIVEC__

    return Pooling1D::forward(bottom_blob, top_blob, opt);
}

} // namespace ncnn
