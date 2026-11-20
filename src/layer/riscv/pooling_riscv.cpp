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
    if (opt.use_fp16_storage && elembits == 16)
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
void Pooling_riscv::pooling_bf16s_global_scalar(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int channels = bottom_blob.c;
    const int size = bottom_blob.w * bottom_blob.h;
    const int elempack = bottom_blob.elempack;

    if (pooling_type == Pooling::PoolMethod_MAX)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const unsigned short* ptr = bottom_blob.channel(q);
            unsigned short* outptr = top_blob;

            for (int p = 0; p < elempack; p++)
            {
                float max = -FLT_MAX;
                for (int i = 0; i < size; i++)
                {
                    float val = bfloat16_to_float32(ptr[i * elempack + p]);
                    if (val > max)
                        max = val;
                }

                outptr[q * elempack + p] = float32_to_bfloat16(max);
            }
        }
    }
    else if (pooling_type == Pooling::PoolMethod_AVE)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const unsigned short* ptr = bottom_blob.channel(q);
            unsigned short* outptr = top_blob;

            for (int p = 0; p < elempack; p++)
            {
                float sum = 0.f;
                for (int i = 0; i < size; i++)
                {
                    sum += bfloat16_to_float32(ptr[i * elempack + p]);
                }

                outptr[q * elempack + p] = float32_to_bfloat16(sum / size);
            }
        }
    }
}

void Pooling_riscv::pooling_bf16s_scalar(const Mat& bottom_blob, Mat& top_blob, const Mat& bottom_blob_bordered, const Option& opt) const
{

    const int w = bottom_blob_bordered.w;
    const int h = bottom_blob_bordered.h;
    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;

    const int maxk = kernel_w * kernel_h;

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

    if (pooling_type == Pooling::PoolMethod_MAX)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const Mat m = bottom_blob_bordered.channel(q);
            unsigned short* outptr = top_blob.channel(q);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    const unsigned short* sptr = (const unsigned short*)m.row(i * stride_h) + j * stride_w * elempack;

                    for (int p = 0; p < elempack; p++)
                    {
                        float max = -FLT_MAX;

                        for (int k = 0; k < maxk; k++)
                        {
                            float val = bfloat16_to_float32(sptr[space_ofs[k] * elempack + p]);
                            if (val > max)
                                max = val;
                        }

                        outptr[j * elempack + p] = float32_to_bfloat16(max);
                    }
                }

                outptr += outw * elempack;
            }
        }
    }
    else if (pooling_type == Pooling::PoolMethod_AVE)
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
                unsigned short* outptr = top_blob.channel(q);

                for (int i = 0; i < outh; i++)
                {
                    int sy0 = i * stride_h;

                    for (int j = 0; j < outw; j++)
                    {
                        int sx0 = j * stride_w;

                        for (int p = 0; p < elempack; p++)
                        {
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

                                    float val = bfloat16_to_float32(((const unsigned short*)m.row(sy))[sx * elempack + p]);
                                    sum += val;
                                    area += 1;
                                }
                            }

                            outptr[j * elempack + p] = float32_to_bfloat16(sum / area);
                        }
                    }

                    outptr += outw * elempack;
                }
            }
        }
        else // if (avgpool_count_include_pad == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob_bordered.channel(q);
                unsigned short* outptr = top_blob.channel(q);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        const unsigned short* sptr = (const unsigned short*)m.row(i * stride_h) + j * stride_w * elempack;

                        for (int p = 0; p < elempack; p++)
                        {
                            float sum = 0.f;

                            for (int k = 0; k < maxk; k++)
                            {
                                float val = bfloat16_to_float32(sptr[space_ofs[k] * elempack + p]);
                                sum += val;
                            }

                            outptr[j * elempack + p] = float32_to_bfloat16(sum / maxk);
                        }
                    }

                    outptr += outw * elempack;
                }
            }
        }
    }
}

int Pooling_riscv::forward_bf16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{


    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

#if __riscv_vector
    const int packn = csrr_vlenb() / 2;
#endif

    if (global_pooling)
    {
        top_blob.create(channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        int size = w * h;

#if __riscv_vector
        if (elempack == packn)
        {
            const size_t vl = __riscv_vsetvl_e16m2(packn);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                unsigned long old_fxcr = csrr_fxcr();
                old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif
                const unsigned short* ptr = bottom_blob.channel(q);
                unsigned short* outptr = top_blob;

                if (pooling_type == Pooling::PoolMethod_MAX)
                {
                    vfloat32m4_t _max = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr, vl), vl);
                    for (int i = 1; i < size; i++)
                    {
                        vfloat32m4_t _val = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr + i * packn, vl), vl);
                        _max = __riscv_vfmax_vv_f32m4(_max, _val, vl);
                    }

                    __riscv_vse16_v_u16m2(outptr + q * packn, float2bfloat_rvv(_max, vl), vl);
                }
                else if (pooling_type == Pooling::PoolMethod_AVE)
                {
                    vfloat32m4_t _sum = __riscv_vfmv_v_f_f32m4(0.f, vl);
                    for (int i = 0; i < size; i++)
                    {
                        vfloat32m4_t _val = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr + i * packn, vl), vl);
                        _sum = __riscv_vfadd_vv_f32m4(_sum, _val, vl);
                    }

                    vfloat32m4_t _avg = __riscv_vfmul_vf_f32m4(_sum, 1.f / size, vl);
                    __riscv_vse16_v_u16m2(outptr + q * packn, float2bfloat_rvv(_avg, vl), vl);
                }
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
            }

            return 0;
        }

        if (elempack == 1)
        {
            const size_t vlmax = __riscv_vsetvlmax_e16m2();
            const size_t vlmax4 = __riscv_vsetvlmax_e32m4();
            const int nfull = size / (int)vlmax * (int)vlmax;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                unsigned long old_fxcr = csrr_fxcr();
                old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif
                const unsigned short* ptr = bottom_blob.channel(q);
                unsigned short* outptr = top_blob;

                if (pooling_type == Pooling::PoolMethod_MAX)
                {
                    vfloat32m4_t _max = __riscv_vfmv_v_f_f32m4(-FLT_MAX, vlmax4);
                    int i = 0;
                    for (; i < nfull; i += vlmax)
                    {
                        vfloat32m4_t _val = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr + i, vlmax), vlmax);
                        _max = __riscv_vfmax_vv_f32m4(_max, _val, vlmax);
                    }

                    vfloat32m1_t _max1 = __riscv_vfmv_s_f_f32m1(-FLT_MAX, __riscv_vsetvlmax_e32m1());
                    _max1 = __riscv_vfredmax_vs_f32m4_f32m1(_max, _max1, vlmax4);
                    float max = __riscv_vfmv_f_s_f32m1_f32(_max1);

                    for (; i < size; i++)
                    {
                        float val = bfloat16_to_float32(ptr[i]);
                        if (val > max)
                            max = val;
                    }

                    outptr[q] = float32_to_bfloat16(max);
                }
                else if (pooling_type == Pooling::PoolMethod_AVE)
                {
                    vfloat32m4_t _sum = __riscv_vfmv_v_f_f32m4(0.f, vlmax4);
                    int i = 0;
                    for (; i < nfull; i += vlmax)
                    {
                        vfloat32m4_t _val = bfloat2float_rvv(__riscv_vle16_v_u16m2(ptr + i, vlmax), vlmax);
                        _sum = __riscv_vfadd_vv_f32m4(_sum, _val, vlmax);
                    }

                    vfloat32m1_t _sum1 = __riscv_vfmv_s_f_f32m1(0.f, __riscv_vsetvlmax_e32m1());
                    _sum1 = __riscv_vfredusum_vs_f32m4_f32m1(_sum, _sum1, vlmax4);
                    float sum = __riscv_vfmv_f_s_f32m1_f32(_sum1);

                    for (; i < size; i++)
                    {
                        sum += bfloat16_to_float32(ptr[i]);
                    }

                    outptr[q] = float32_to_bfloat16(sum / size);
                }
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
            }

            return 0;
        }
#endif // __riscv_vector

        pooling_bf16s_global_scalar(bottom_blob, top_blob, opt);

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

#if __riscv_vector
    if (elempack == packn)
    {
        const size_t vl = __riscv_vsetvl_e16m2(packn);

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

        if (pooling_type == Pooling::PoolMethod_MAX)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                unsigned long old_fxcr = csrr_fxcr();
                old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif
                const Mat m = bottom_blob_bordered.channel(q);
                unsigned short* outptr = top_blob.channel(q);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        const unsigned short* sptr = (const unsigned short*)m.row(i * stride_h) + j * stride_w * packn;

                        vfloat32m4_t _max = bfloat2float_rvv(__riscv_vle16_v_u16m2(sptr, vl), vl);

                        for (int k = 1; k < maxk; k++)
                        {
                            vfloat32m4_t _val = bfloat2float_rvv(__riscv_vle16_v_u16m2(sptr + space_ofs[k] * packn, vl), vl);
                            _max = __riscv_vfmax_vv_f32m4(_max, _val, vl);
                        }

                        __riscv_vse16_v_u16m2(outptr + j * packn, float2bfloat_rvv(_max, vl), vl);
                    }

                    outptr += outw * packn;
                }
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
            }
        }
        else if (pooling_type == Pooling::PoolMethod_AVE)
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
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                    unsigned long old_fxcr = csrr_fxcr();
                    old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif
                    const Mat m = bottom_blob_bordered.channel(q);
                    unsigned short* outptr = top_blob.channel(q);

                    for (int i = 0; i < outh; i++)
                    {
                        int sy0 = i * stride_h;

                        for (int j = 0; j < outw; j++)
                        {
                            int sx0 = j * stride_w;

                            vfloat32m4_t _sum = __riscv_vfmv_v_f_f32m4(0.f, vl);
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

                                    vfloat32m4_t _val = bfloat2float_rvv(__riscv_vle16_v_u16m2((const unsigned short*)m.row(sy) + sx * packn, vl), vl);
                                    _sum = __riscv_vfadd_vv_f32m4(_sum, _val, vl);
                                    area += 1;
                                }
                            }

                            vfloat32m4_t _avg = __riscv_vfmul_vf_f32m4(_sum, 1.f / area, vl);
                            __riscv_vse16_v_u16m2(outptr + j * packn, float2bfloat_rvv(_avg, vl), vl);
                        }

                        outptr += outw * packn;
                    }
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                    csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
                }
            }
            else // if (avgpool_count_include_pad == 1)
            {
                const float inv_maxk = 1.f / maxk;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                    unsigned long old_fxcr = csrr_fxcr();
                    old_fxcr = csrrw_fxcr((old_fxcr | (1ul << 31)) & ~(7ul << 24));
#endif
                    const Mat m = bottom_blob_bordered.channel(q);
                    unsigned short* outptr = top_blob.channel(q);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            const unsigned short* sptr = (const unsigned short*)m.row(i * stride_h) + j * stride_w * packn;

                            vfloat32m4_t _sum = __riscv_vfmv_v_f_f32m4(0.f, vl);

                            for (int k = 0; k < maxk; k++)
                            {
                                vfloat32m4_t _val = bfloat2float_rvv(__riscv_vle16_v_u16m2(sptr + space_ofs[k] * packn, vl), vl);
                                _sum = __riscv_vfadd_vv_f32m4(_sum, _val, vl);
                            }

                            vfloat32m4_t _avg = __riscv_vfmul_vf_f32m4(_sum, inv_maxk, vl);
                            __riscv_vse16_v_u16m2(outptr + j * packn, float2bfloat_rvv(_avg, vl), vl);
                        }

                        outptr += outw * packn;
                    }
#if __riscv_xtheadvector && !(NCNN_ZVFBFMIN && __riscv_zvfbfmin)
                    csrrw_fxcr(old_fxcr | (csrr_fxcr() & 0x3f));
#endif
                }
            }
        }

        return 0;
    }
#endif // __riscv_vector

    pooling_bf16s_scalar(bottom_blob, top_blob, bottom_blob_bordered, opt);

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
