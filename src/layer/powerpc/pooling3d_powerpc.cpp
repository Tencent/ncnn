// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pooling3d_powerpc.h"

#include <algorithm>
#include <float.h>
#include <limits.h>

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

namespace ncnn {

Pooling3D_powerpc::Pooling3D_powerpc()
{
    // base defaults kept: one_blob_only = true, support_inplace = false
    // the base class is not packing aware, support_packing stays false
}

#if __ALTIVEC__

static void global_max3d_altivec(const Mat& bottom_blob, Mat& top_blob, int size, int channels, int num_threads)
{
    #pragma omp parallel for num_threads(num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* ptr = bottom_blob.channel(q);

        int i = 0;
        float max_value = ptr[0];
        if (size >= 4)
        {
            v4f32 _max = __ncnn_altivec_load(ptr);
            i = 4;
            for (; i + 3 < size; i += 4)
            {
                _max = vec_max(_max, __ncnn_altivec_load(ptr + i));
            }
            max_value = __ncnn_altivec_reduce_max(_max);
        }
        for (; i < size; i++)
        {
            max_value = std::max(max_value, ptr[i]);
        }

        top_blob[q] = max_value;
    }
}

static void global_avg3d_altivec(const Mat& bottom_blob, Mat& top_blob, int size, int channels, int num_threads)
{
    #pragma omp parallel for num_threads(num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* ptr = bottom_blob.channel(q);

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

        top_blob[q] = sum / size;
    }
}

#endif // __ALTIVEC__

int Pooling3D_powerpc::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (bottom_blob.elembits() != 32)
    {
        // int8 / bf16, fall back to the generic implementation
        return Pooling3D::forward(bottom_blob, top_blob, opt);
    }

#if __ALTIVEC__
    if (bottom_blob.elempack == 1)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int d = bottom_blob.d;
        int channels = bottom_blob.c;
        size_t elemsize = bottom_blob.elemsize;

        if (global_pooling)
        {
            top_blob.create(channels, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            int size = w * h * d;

            if (pooling_type == PoolMethod_MAX)
            {
                global_max3d_altivec(bottom_blob, top_blob, size, channels, opt.num_threads);
            }
            else if (pooling_type == PoolMethod_AVE)
            {
                global_avg3d_altivec(bottom_blob, top_blob, size, channels, opt.num_threads);
            }

            return 0;
        }

        if (adaptive_pooling)
        {
            int _out_w = out_w == -233 ? w : out_w;
            int _out_h = out_h == -233 ? h : out_h;
            int _out_d = out_d == -233 ? d : out_d;

            if (_out_w == w && _out_h == h && _out_d == d)
            {
                top_blob = bottom_blob;
                return 0;
            }

            top_blob.create(_out_w, _out_h, _out_d, channels, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (pooling_type == PoolMethod_MAX)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* inptr = bottom_blob.channel(q);
                    float* outptr = top_blob.channel(q);

                    for (int z = 0; z < _out_d; z++)
                    {
                        // floor div
                        const int id0 = d * z / _out_d;
                        // ceil div
                        const int id1 = (d * (z + 1) + _out_d - 1) / _out_d;
                        for (int i = 0; i < _out_h; i++)
                        {
                            // floor div
                            const int ih0 = h * i / _out_h;
                            // ceil div
                            const int ih1 = (h * (i + 1) + _out_h - 1) / _out_h;
                            for (int j = 0; j < _out_w; j++)
                            {
                                // floor div
                                const int iw0 = w * j / _out_w;
                                // ceil div
                                const int iw1 = (w * (j + 1) + _out_w - 1) / _out_w;

                                float max_value = inptr[id0 * w * h + ih0 * w + iw0];

                                for (int id = id0; id < id1; id++)
                                {
                                    for (int ih = ih0; ih < ih1; ih++)
                                    {
                                        const float* rowptr = inptr + id * w * h + ih * w;

                                        int iw = iw0;
                                        if (iw1 - iw0 >= 4)
                                        {
                                            v4f32 _max = __ncnn_altivec_load(rowptr + iw0);
                                            iw = iw0 + 4;
                                            for (; iw + 3 < iw1; iw += 4)
                                            {
                                                _max = vec_max(_max, __ncnn_altivec_load(rowptr + iw));
                                            }
                                            max_value = std::max(max_value, __ncnn_altivec_reduce_max(_max));
                                        }
                                        for (; iw < iw1; iw++)
                                        {
                                            max_value = std::max(max_value, rowptr[iw]);
                                        }
                                    }
                                }

                                outptr[j] = max_value;
                            }

                            outptr += _out_w;
                        }
                    }
                }
            }
            else if (pooling_type == PoolMethod_AVE)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* inptr = bottom_blob.channel(q);
                    float* outptr = top_blob.channel(q);

                    for (int z = 0; z < _out_d; z++)
                    {
                        // floor div
                        const int id0 = d * z / _out_d;
                        // ceil div
                        const int id1 = (d * (z + 1) + _out_d - 1) / _out_d;
                        int dk = id1 - id0;
                        for (int i = 0; i < _out_h; i++)
                        {
                            // floor div
                            const int ih0 = h * i / _out_h;
                            // ceil div
                            const int ih1 = (h * (i + 1) + _out_h - 1) / _out_h;
                            int hk = ih1 - ih0;
                            for (int j = 0; j < _out_w; j++)
                            {
                                // floor div
                                const int iw0 = w * j / _out_w;
                                // ceil div
                                const int iw1 = (w * (j + 1) + _out_w - 1) / _out_w;
                                int wk = iw1 - iw0;

                                float sum = 0;
                                for (int id = id0; id < id1; id++)
                                {
                                    for (int ih = ih0; ih < ih1; ih++)
                                    {
                                        const float* rowptr = inptr + id * w * h + ih * w;

                                        int iw = iw0;
                                        v4f32 _sum = vec_splats(0.f);
                                        for (; iw + 3 < iw1; iw += 4)
                                        {
                                            _sum = vec_add(_sum, __ncnn_altivec_load(rowptr + iw));
                                        }
                                        sum += __ncnn_altivec_reduce_add(_sum);
                                        for (; iw < iw1; iw++)
                                        {
                                            sum += rowptr[iw];
                                        }
                                    }
                                }

                                outptr[j] = sum / hk / wk / dk;
                            }

                            outptr += _out_w;
                        }
                    }
                }
            }

            return 0;
        }

        Mat bottom_blob_bordered;
        Option opt_pad = opt;
        opt_pad.use_packing_layout = false;
        make_padding(bottom_blob, bottom_blob_bordered, opt_pad);
        if (bottom_blob_bordered.empty())
            return -100;

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;
        d = bottom_blob_bordered.d;

        int outw = (w - kernel_w) / stride_w + 1;
        int outh = (h - kernel_h) / stride_h + 1;
        int outd = (d - kernel_d) / stride_d + 1;

        top_blob.create(outw, outh, outd, channels, elemsize);
        if (top_blob.empty())
            return -100;

        if (pooling_type == PoolMethod_MAX)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob_bordered.channel(q);
                float* outptr = top_blob.channel(q);
                for (int z = 0; z < outd; z++)
                {
                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            const float* sptr = m.depth(z * stride_d).row(i * stride_h) + j * stride_w;

                            float max_value = sptr[0];

                            // vectorize the contiguous kernel_w run of each kernel row
                            for (int kd = 0; kd < kernel_d; kd++)
                            {
                                for (int ki = 0; ki < kernel_h; ki++)
                                {
                                    const float* dptr = m.depth(z * stride_d + kd).row(i * stride_h + ki) + j * stride_w;

                                    int kj = 0;
                                    if (kernel_w >= 4)
                                    {
                                        v4f32 _max = __ncnn_altivec_load(dptr);
                                        kj = 4;
                                        for (; kj + 3 < kernel_w; kj += 4)
                                        {
                                            _max = vec_max(_max, __ncnn_altivec_load(dptr + kj));
                                        }
                                        max_value = std::max(max_value, __ncnn_altivec_reduce_max(_max));
                                    }
                                    for (; kj < kernel_w; kj++)
                                    {
                                        max_value = std::max(max_value, dptr[kj]);
                                    }
                                }
                            }

                            outptr[j] = max_value;
                        }

                        outptr += outw;
                    }
                }
            }
        }
        else if (pooling_type == PoolMethod_AVE)
        {
            if (avgpool_count_include_pad == 0)
            {
                int wtailpad = 0;
                int htailpad = 0;
                int dtailpad = 0;

                if (pad_mode == 0) // full padding
                {
                    wtailpad = bottom_blob_bordered.w - bottom_blob.w - pad_left - pad_right;
                    htailpad = bottom_blob_bordered.h - bottom_blob.h - pad_top - pad_bottom;
                    dtailpad = bottom_blob_bordered.d - bottom_blob.d - pad_front - pad_behind;
                }

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob_bordered.channel(q);
                    float* outptr = top_blob.channel(q);

                    for (int z = 0; z < outd; z++)
                    {
                        int sz0 = z * stride_d;

                        // the generic skip / break conditions reduce to
                        // sz in [max(sz0, pad_front), min(sz0 + kernel_d, d - pad_behind - dtailpad))
                        int sz_lo = sz0 > pad_front ? sz0 : pad_front;
                        int sz_hi = sz0 + kernel_d;
                        int sz_bound = d - pad_behind - dtailpad;
                        if (sz_hi > sz_bound)
                            sz_hi = sz_bound;

                        for (int i = 0; i < outh; i++)
                        {
                            int sy0 = i * stride_h;

                            int sy_lo = sy0 > pad_top ? sy0 : pad_top;
                            int sy_hi = sy0 + kernel_h;
                            int sy_bound = h - pad_bottom - htailpad;
                            if (sy_hi > sy_bound)
                                sy_hi = sy_bound;

                            for (int j = 0; j < outw; j++)
                            {
                                int sx0 = j * stride_w;

                                int sx_lo = sx0 > pad_left ? sx0 : pad_left;
                                int sx_hi = sx0 + kernel_w;
                                int sx_bound = w - pad_right - wtailpad;
                                if (sx_hi > sx_bound)
                                    sx_hi = sx_bound;

                                float sum = 0.f;
                                int area = 0;

                                for (int sz = sz_lo; sz < sz_hi; sz++)
                                {
                                    for (int sy = sy_lo; sy < sy_hi; sy++)
                                    {
                                        const float* rowptr = m.depth(sz).row(sy);

                                        int sx = sx_lo;
                                        v4f32 _sum = vec_splats(0.f);
                                        if (sx_hi > sx_lo)
                                        {
                                            for (; sx + 3 < sx_hi; sx += 4)
                                            {
                                                _sum = vec_add(_sum, __ncnn_altivec_load(rowptr + sx));
                                            }
                                            sum += __ncnn_altivec_reduce_add(_sum);
                                        }
                                        for (; sx < sx_hi; sx++)
                                        {
                                            sum += rowptr[sx];
                                        }
                                    }
                                }

                                if (sz_hi > sz_lo && sy_hi > sy_lo && sx_hi > sx_lo)
                                    area = (sz_hi - sz_lo) * (sy_hi - sy_lo) * (sx_hi - sx_lo);

                                outptr[j] = sum / area;
                            }

                            outptr += outw;
                        }
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

                    for (int z = 0; z < outd; z++)
                    {
                        for (int i = 0; i < outh; i++)
                        {
                            for (int j = 0; j < outw; j++)
                            {
                                // vectorize the contiguous kernel_w run of each kernel row
                                v4f32 _sum = vec_splats(0.f);
                                float sum = 0.f;

                                for (int kd = 0; kd < kernel_d; kd++)
                                {
                                    for (int ki = 0; ki < kernel_h; ki++)
                                    {
                                        const float* dptr = m.depth(z * stride_d + kd).row(i * stride_h + ki) + j * stride_w;

                                        int kj = 0;
                                        if (kernel_w >= 4)
                                        {
                                            for (; kj + 3 < kernel_w; kj += 4)
                                            {
                                                _sum = vec_add(_sum, __ncnn_altivec_load(dptr + kj));
                                            }
                                        }
                                        for (; kj < kernel_w; kj++)
                                        {
                                            sum += dptr[kj];
                                        }
                                    }
                                }

                                sum += __ncnn_altivec_reduce_add(_sum);

                                outptr[j] = sum / (kernel_w * kernel_h * kernel_d);
                            }

                            outptr += outw;
                        }
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

        return Pooling3D::forward(bottom_blob_unpacked, top_blob, opt);
    }
#endif // __ALTIVEC__

    return Pooling3D::forward(bottom_blob, top_blob, opt);
}

} // namespace ncnn
