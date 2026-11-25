// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pooling_riscv.h"

#include <float.h>

#if __riscv_vector
#include <riscv_vector.h>
#include "riscv_usability.h"
#endif // __riscv_vector

#include "cpu.h"

namespace ncnn {

#if NCNN_BF16
#include "pooling_bf16s.h"
#endif

Pooling_riscv::Pooling_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
#if NCNN_ZFH
#if __riscv_vector
    support_fp16_storage = cpu_support_riscv_zvfh();
#else
    support_fp16_storage = cpu_support_riscv_zfh();
#endif
#endif
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int Pooling_riscv::create_pipeline(const Option& /*opt*/)
{
    if (adaptive_pooling)
    {
        support_packing = false;

        support_bf16_storage = false;
        support_fp16_storage = false;
        support_int8_storage = false;
        support_tensor_storage = false;
    }
    return 0;
}

int Pooling_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (adaptive_pooling)
    {
        return Pooling::forward(bottom_blob, top_blob, opt);
    }

#if NCNN_ZFH || NCNN_BF16
    int elembits = bottom_blob.elembits();
#endif

#if NCNN_ZFH
    if (support_fp16_storage && opt.use_fp16_storage && elembits == 16)
    {
        if (opt.use_fp16_arithmetic)
            return forward_fp16sa(bottom_blob, top_blob, opt);
        else
            return forward_fp16s(bottom_blob, top_blob, opt);
    }
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage && elembits == 16)
        return forward_bf16s(bottom_blob, top_blob, opt);
#endif

    // max value in NxN window
    // avg value in NxN window

#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m1(packn);
#endif

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

#if __riscv_vector
    //     NCNN_LOGE("Pooling     input %d x %d  pad = %d %d %d %d  ksize=%d %d  stride=%d %d", w, h, pad_left, pad_right, pad_top, pad_bottom, kernel_w, kernel_h, stride_w, stride_h);

    if (elempack == packn)
    {
        if (global_pooling)
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

                    vfloat32m1_t _max = __riscv_vle32_v_f32m1(ptr, vl);
                    for (int i = 0; i < size; i++)
                    {
                        vfloat32m1_t _val = __riscv_vle32_v_f32m1(ptr, vl);
                        _max = __riscv_vfmax_vv_f32m1(_max, _val, vl);
                        ptr += packn;
                    }

                    float* outptr = top_blob;
                    __riscv_vse32_v_f32m1(outptr + q * packn, _max, vl);
                }
            }
            else if (pooling_type == PoolMethod_AVE)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = bottom_blob.channel(q);

                    vfloat32m1_t _sum = __riscv_vfmv_v_f_f32m1(0.f, vl);
                    for (int i = 0; i < size; i++)
                    {
                        vfloat32m1_t _val = __riscv_vle32_v_f32m1(ptr, vl);
                        _sum = __riscv_vfadd_vv_f32m1(_sum, _val, vl);
                        ptr += packn;
                    }

                    vfloat32m1_t _avg = __riscv_vfmul_vf_f32m1(_sum, 1.f / size, vl);

                    float* outptr = top_blob;
                    __riscv_vse32_v_f32m1(outptr + q * packn, _avg, vl);
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
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob_bordered.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        const float* sptr = m.row(i * stride_h) + j * stride_w * packn;

                        vfloat32m1_t _max = __riscv_vle32_v_f32m1(sptr, vl);

                        for (int k = 0; k < maxk; k++)
                        {
                            vfloat32m1_t _val = __riscv_vle32_v_f32m1(sptr + space_ofs[k] * packn, vl);
                            _max = __riscv_vfmax_vv_f32m1(_max, _val, vl);
                        }

                        __riscv_vse32_v_f32m1(outptr + j * packn, _max, vl);
                    }

                    outptr += outw * packn;
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

                            vfloat32m1_t _sum = __riscv_vfmv_v_f_f32m1(0.f, vl);
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

                                    vfloat32m1_t _val = __riscv_vle32_v_f32m1(m.row(sy) + sx * packn, vl);
                                    _sum = __riscv_vfadd_vv_f32m1(_sum, _val, vl);
                                    area += 1;
                                }
                            }

                            vfloat32m1_t _avg = __riscv_vfmul_vf_f32m1(_sum, 1.f / area, vl);
                            __riscv_vse32_v_f32m1(outptr + j * packn, _avg, vl);
                        }

                        outptr += outw * packn;
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
                            const float* sptr = m.row(i * stride_h) + j * stride_w * packn;

                            vfloat32m1_t _sum = __riscv_vfmv_v_f_f32m1(0.f, vl);

                            for (int k = 0; k < maxk; k++)
                            {
                                vfloat32m1_t _val = __riscv_vle32_v_f32m1(sptr + space_ofs[k] * packn, vl);
                                _sum = __riscv_vfadd_vv_f32m1(_sum, _val, vl);
                            }

                            vfloat32m1_t _avg = __riscv_vfmul_vf_f32m1(_sum, inv_maxk, vl);
                            __riscv_vse32_v_f32m1(outptr + j * packn, _avg, vl);
                        }

                        outptr += outw * packn;
                    }
                }
            }
        }

        return 0;
    }
#endif // __riscv_vector

    return Pooling::forward(bottom_blob, top_blob, opt);
}

#if NCNN_BF16
int Pooling_riscv::forward_bf16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // max value in NxN window
    // avg value in NxN window

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    if (global_pooling)
    {
        top_blob.create(channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (pooling_type == PoolMethod_MAX)
        {
            pooling_global_max_bf16s(bottom_blob, top_blob, opt);
        }
        else if (pooling_type == PoolMethod_AVE)
        {
            pooling_global_avg_bf16s(bottom_blob, top_blob, opt);
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

    if (pooling_type == PoolMethod_MAX)
    {
        pooling_max_bf16s(bottom_blob_bordered, top_blob, kernel_w, kernel_h, stride_w, stride_h, opt);
    }
    else if (pooling_type == PoolMethod_AVE)
    {
        pooling_avg_bf16s(bottom_blob_bordered, bottom_blob, top_blob, kernel_w, kernel_h, stride_w, stride_h, pad_left, pad_right, pad_top, pad_bottom, pad_mode, avgpool_count_include_pad, opt);
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
