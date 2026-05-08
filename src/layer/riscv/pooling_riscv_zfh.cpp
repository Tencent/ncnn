// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pooling_riscv.h"

#include <float.h>

#if __riscv_vector
#include <riscv_vector.h>
#include "riscv_usability.h"
#endif // __riscv_vector

namespace ncnn {

#if NCNN_ZFH
int Pooling_riscv::forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // max value in NxN window
    // avg value in NxN window

#if __riscv_zvfh
    const int packn = csrr_vlenb() / 2;
    const size_t vl = __riscv_vsetvl_e16m1(packn);
#endif // __riscv_zvfh

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    //     NCNN_LOGE("Pooling     input %d x %d  pad = %d %d %d %d  ksize=%d %d  stride=%d %d", w, h, pad_left, pad_right, pad_top, pad_bottom, kernel_w, kernel_h, stride_w, stride_h);

    if (global_pooling)
    {
        top_blob.create(channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        int size = w * h;

        if (pooling_type == PoolMethod_MAX)
        {
#if __riscv_zvfh
            if (elempack == packn)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = bottom_blob.channel(q);

                    vfloat16m1_t _max = __riscv_vfmv_v_f_f16m1((__fp16)-FLT_MAX, vl);
                    for (int i = 0; i < size; i++)
                    {
                        vfloat16m1_t _val = __riscv_vle16_v_f16m1(ptr, vl);
                        _max = __riscv_vfmax_vv_f16m1(_max, _val, vl);
                        ptr += packn;
                    }

                    __fp16* outptr = top_blob;
                    __riscv_vse16_v_f16m1(outptr + q * packn, _max, vl);
                }
            }
#endif // __riscv_zvfh

            if (elempack == 1)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = bottom_blob.channel(q);

                    __fp16 max = (__fp16)-FLT_MAX;
                    for (int i = 0; i < size; i++)
                    {
                        max = std::max(max, ptr[i]);
                    }

                    __fp16* outptr = top_blob;
                    outptr[q] = max;
                }
            }
        }

        if (pooling_type == PoolMethod_AVE)
        {
#if __riscv_zvfh
            if (elempack == packn)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = bottom_blob.channel(q);

                    vfloat32m2_t _sum = __riscv_vfmv_v_f_f32m2(0.f, vl);
                    for (int i = 0; i < size; i++)
                    {
                        vfloat32m2_t _val = __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(ptr, vl), vl);
                        _sum = __riscv_vfadd_vv_f32m2(_sum, _val, vl);
                        ptr += packn;
                    }

                    vfloat32m2_t _avg = __riscv_vfmul_vf_f32m2(_sum, 1.f / size, vl);

                    __fp16* outptr = top_blob;
                    __riscv_vse16_v_f16m1(outptr + q * packn, __riscv_vfncvt_f_f_w_f16m1(_avg, vl), vl);
                }
            }
#endif // __riscv_zvfh

            if (elempack == 1)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = bottom_blob.channel(q);

                    float sum = 0.f;
                    for (int i = 0; i < size; i++)
                    {
                        sum += (float)ptr[i];
                    }

                    __fp16* outptr = top_blob;
                    outptr[q] = (__fp16)(sum / size);
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
#if __riscv_zvfh
        if (elempack == packn)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob_bordered.channel(q);
                __fp16* outptr = top_blob.channel(q);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        const __fp16* sptr = m.row<const __fp16>(i * stride_h) + j * stride_w * packn;

                        vfloat16m1_t _max = __riscv_vfmv_v_f_f16m1((__fp16)-FLT_MAX, vl);

                        for (int k = 0; k < maxk; k++)
                        {
                            vfloat16m1_t _val = __riscv_vle16_v_f16m1(sptr + space_ofs[k] * packn, vl);
                            _max = __riscv_vfmax_vv_f16m1(_max, _val, vl);
                        }

                        __riscv_vse16_v_f16m1(outptr + j * packn, _max, vl);
                    }

                    outptr += outw * packn;
                }
            }
        }
#endif // __riscv_zvfh

        if (elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob_bordered.channel(q);
                __fp16* outptr = top_blob.channel(q);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        const __fp16* sptr = m.row<const __fp16>(i * stride_h) + j * stride_w;

                        __fp16 max = (__fp16)-FLT_MAX;

                        for (int k = 0; k < maxk; k++)
                        {
                            __fp16 val = sptr[space_ofs[k]];
                            max = std::max(max, val);
                        }

                        outptr[j] = max;
                    }

                    outptr += outw;
                }
            }
        }
    }

    if (pooling_type == PoolMethod_AVE)
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

#if __riscv_zvfh
            if (elempack == packn)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob_bordered.channel(q);
                    __fp16* outptr = top_blob.channel(q);

                    for (int i = 0; i < outh; i++)
                    {
                        int sy0 = i * stride_h;

                        for (int j = 0; j < outw; j++)
                        {
                            int sx0 = j * stride_w;

                            vfloat32m2_t _sum = __riscv_vfmv_v_f_f32m2(0.f, vl);
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

                                    vfloat32m2_t _val = __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(m.row<const __fp16>(sy) + sx * packn, vl), vl);
                                    _sum = __riscv_vfadd_vv_f32m2(_sum, _val, vl);
                                    area += 1;
                                }
                            }

                            vfloat32m2_t _avg = __riscv_vfmul_vf_f32m2(_sum, 1.f / area, vl);
                            __riscv_vse16_v_f16m1(outptr + j * packn, __riscv_vfncvt_f_f_w_f16m1(_avg, vl), vl);
                        }

                        outptr += outw * packn;
                    }
                }
            }
#endif // __riscv_zvfh

            if (elempack == 1)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob_bordered.channel(q);
                    __fp16* outptr = top_blob.channel(q);

                    for (int i = 0; i < outh; i++)
                    {
                        int sy0 = i * stride_h;

                        for (int j = 0; j < outw; j++)
                        {
                            int sx0 = j * stride_w;

                            float sum = 0.f;
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

                                    float val = (float)(m.row<const __fp16>(sy)[sx]);
                                    sum += val;
                                    area += 1;
                                }
                            }

                            outptr[j] = (__fp16)(sum / area);
                        }

                        outptr += outw;
                    }
                }
            }
        }

        if (avgpool_count_include_pad == 1)
        {
#if __riscv_zvfh
            if (elempack == packn)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob_bordered.channel(q);
                    __fp16* outptr = top_blob.channel(q);

                    const float inv_maxk = 1.f / maxk;

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            const __fp16* sptr = m.row<const __fp16>(i * stride_h) + j * stride_w * packn;

                            vfloat32m2_t _sum = __riscv_vfmv_v_f_f32m2(0.f, vl);

                            for (int k = 0; k < maxk; k++)
                            {
                                vfloat32m2_t _val = __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(sptr + space_ofs[k] * packn, vl), vl);
                                _sum = __riscv_vfadd_vv_f32m2(_sum, _val, vl);
                            }

                            vfloat32m2_t _avg = __riscv_vfmul_vf_f32m2(_sum, inv_maxk, vl);
                            __riscv_vse16_v_f16m1(outptr + j * packn, __riscv_vfncvt_f_f_w_f16m1(_avg, vl), vl);
                        }

                        outptr += outw * packn;
                    }
                }
            }
#endif // __riscv_zvfh

            if (elempack == 1)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob_bordered.channel(q);
                    __fp16* outptr = top_blob.channel(q);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            const __fp16* sptr = m.row<const __fp16>(i * stride_h) + j * stride_w;

                            float sum = 0.f;

                            for (int k = 0; k < maxk; k++)
                            {
                                float val = (float)(sptr[space_ofs[k]]);
                                sum += val;
                            }

                            outptr[j] = (__fp16)(sum / maxk);
                        }

                        outptr += outw;
                    }
                }
            }
        }
    }

    return 0;
}

int Pooling_riscv::forward_fp16sa(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // max value in NxN window
    // avg value in NxN window

    if (pooling_type == PoolMethod_MAX || global_pooling)
    {
        return forward_fp16s(bottom_blob, top_blob, opt);
    }

#if __riscv_zvfh
    const int packn = csrr_vlenb() / 2;
    const size_t vl = __riscv_vsetvl_e16m1(packn);
#endif // __riscv_zvfh

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    //     NCNN_LOGE("Pooling     input %d x %d  pad = %d %d %d %d  ksize=%d %d  stride=%d %d", w, h, pad_left, pad_right, pad_top, pad_bottom, kernel_w, kernel_h, stride_w, stride_h);

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

    if (pooling_type == PoolMethod_AVE)
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

#if __riscv_zvfh
            if (elempack == packn)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob_bordered.channel(q);
                    __fp16* outptr = top_blob.channel(q);

                    for (int i = 0; i < outh; i++)
                    {
                        int sy0 = i * stride_h;

                        for (int j = 0; j < outw; j++)
                        {
                            int sx0 = j * stride_w;

                            vfloat16m1_t _sum = __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl);
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

                                    vfloat16m1_t _val = __riscv_vle16_v_f16m1(m.row<const __fp16>(sy) + sx * packn, vl);
                                    _sum = __riscv_vfadd_vv_f16m1(_sum, _val, vl);
                                    area += 1;
                                }
                            }

                            vfloat16m1_t _avg = __riscv_vfmul_vf_f16m1(_sum, (__fp16)(1.f / area), vl);
                            __riscv_vse16_v_f16m1(outptr + j * packn, _avg, vl);
                        }

                        outptr += outw * packn;
                    }
                }
            }
#endif // __riscv_zvfh

            if (elempack == 1)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob_bordered.channel(q);
                    __fp16* outptr = top_blob.channel(q);

                    for (int i = 0; i < outh; i++)
                    {
                        int sy0 = i * stride_h;

                        for (int j = 0; j < outw; j++)
                        {
                            int sx0 = j * stride_w;

                            __fp16 sum = (__fp16)0.f;
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

                                    __fp16 val = m.row<const __fp16>(sy)[sx];
                                    sum += val;
                                    area += 1;
                                }
                            }

                            outptr[j] = sum / area;
                        }

                        outptr += outw;
                    }
                }
            }
        }

        if (avgpool_count_include_pad == 1)
        {
#if __riscv_zvfh
            if (elempack == packn)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob_bordered.channel(q);
                    __fp16* outptr = top_blob.channel(q);

                    const __fp16 inv_maxk = (__fp16)(1.f / maxk);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            const __fp16* sptr = m.row<const __fp16>(i * stride_h) + j * stride_w * packn;

                            vfloat16m1_t _sum = __riscv_vfmv_v_f_f16m1((__fp16)0.f, vl);

                            for (int k = 0; k < maxk; k++)
                            {
                                vfloat16m1_t _val = __riscv_vle16_v_f16m1(sptr + space_ofs[k] * packn, vl);
                                _sum = __riscv_vfadd_vv_f16m1(_sum, _val, vl);
                            }

                            vfloat16m1_t _avg = __riscv_vfmul_vf_f16m1(_sum, inv_maxk, vl);
                            __riscv_vse16_v_f16m1(outptr + j * packn, _avg, vl);
                        }

                        outptr += outw * packn;
                    }
                }
            }
#endif // __riscv_zvfh

            if (elempack == 1)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob_bordered.channel(q);
                    __fp16* outptr = top_blob.channel(q);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            const __fp16* sptr = m.row<const __fp16>(i * stride_h) + j * stride_w;

                            __fp16 sum = (__fp16)0.f;

                            for (int k = 0; k < maxk; k++)
                            {
                                __fp16 val = sptr[space_ofs[k]];
                                sum += val;
                            }

                            outptr[j] = sum / maxk;
                        }

                        outptr += outw;
                    }
                }
            }
        }
    }

    return 0;
}
#endif // NCNN_ZFH

} // namespace ncnn
