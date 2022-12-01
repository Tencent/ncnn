// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "pooling_riscv.h"

#include <float.h>

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_usability.h"

namespace ncnn {

Pooling_riscv::Pooling_riscv()
{
#if __riscv_vector
    support_packing = true;
#if __riscv_zfh
    support_fp16_storage = true;
#endif
#endif // __riscv_vector
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

    int elembits = bottom_blob.elembits();

#if __riscv_vector && __riscv_zfh
    if (opt.use_fp16_storage && elembits == 16)
    {
        if (opt.use_fp16_arithmetic)
            return forward_fp16sa(bottom_blob, top_blob, opt);
        else
            return forward_fp16s(bottom_blob, top_blob, opt);
    }
#endif

    // max value in NxN window
    // avg value in NxN window

#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    const size_t vl = vsetvl_e32m1(packn);
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

                    vfloat32m1_t _max = vle32_v_f32m1(ptr, vl);
                    for (int i = 0; i < size; i++)
                    {
                        vfloat32m1_t _val = vle32_v_f32m1(ptr, vl);
                        _max = vfmax_vv_f32m1(_max, _val, vl);
                        ptr += packn;
                    }

                    float* outptr = top_blob;
                    vse32_v_f32m1(outptr + q * packn, _max, vl);
                }
            }
            else if (pooling_type == PoolMethod_AVE)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = bottom_blob.channel(q);

                    vfloat32m1_t _sum = vfmv_v_f_f32m1(0.f, vl);
                    for (int i = 0; i < size; i++)
                    {
                        vfloat32m1_t _val = vle32_v_f32m1(ptr, vl);
                        _sum = vfadd_vv_f32m1(_sum, _val, vl);
                        ptr += packn;
                    }

                    vfloat32m1_t _avg = vfmul_vf_f32m1(_sum, 1.f / size, vl);

                    float* outptr = top_blob;
                    vse32_v_f32m1(outptr + q * packn, _avg, vl);
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

                        vfloat32m1_t _max = vle32_v_f32m1(sptr, vl);

                        for (int k = 0; k < maxk; k++)
                        {
                            vfloat32m1_t _val = vle32_v_f32m1(sptr + space_ofs[k] * packn, vl);
                            _max = vfmax_vv_f32m1(_max, _val, vl);
                        }

                        vse32_v_f32m1(outptr + j * packn, _max, vl);
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

                            vfloat32m1_t _sum = vfmv_v_f_f32m1(0.f, vl);
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

                                    vfloat32m1_t _val = vle32_v_f32m1(m.row(sy) + sx * packn, vl);
                                    _sum = vfadd_vv_f32m1(_sum, _val, vl);
                                    area += 1;
                                }
                            }

                            vfloat32m1_t _avg = vfmul_vf_f32m1(_sum, 1.f / area, vl);
                            vse32_v_f32m1(outptr + j * packn, _avg, vl);
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

                            vfloat32m1_t _sum = vfmv_v_f_f32m1(0.f, vl);

                            for (int k = 0; k < maxk; k++)
                            {
                                vfloat32m1_t _val = vle32_v_f32m1(sptr + space_ofs[k] * packn, vl);
                                _sum = vfadd_vv_f32m1(_sum, _val, vl);
                            }

                            vfloat32m1_t _avg = vfmul_vf_f32m1(_sum, inv_maxk, vl);
                            vse32_v_f32m1(outptr + j * packn, _avg, vl);
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

#if __riscv_vector && __riscv_zfh
int Pooling_riscv::forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // max value in NxN window
    // avg value in NxN window

    const int packn = csrr_vlenb() / 2;
    const size_t vl = vsetvl_e16m1(packn);

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
            if (elempack == packn)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = bottom_blob.channel(q);

                    vfloat16m1_t _max = vfmv_v_f_f16m1((__fp16)-FLT_MAX, vl);
                    for (int i = 0; i < size; i++)
                    {
                        vfloat16m1_t _val = vle16_v_f16m1(ptr, vl);
                        _max = vfmax_vv_f16m1(_max, _val, vl);
                        ptr += packn;
                    }

                    __fp16* outptr = top_blob;
                    vse16_v_f16m1(outptr + q * packn, _max, vl);
                }
            }

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
            if (elempack == packn)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = bottom_blob.channel(q);

                    vfloat32m2_t _sum = vfmv_v_f_f32m2(0.f, vl);
                    for (int i = 0; i < size; i++)
                    {
                        vfloat32m2_t _val = vfwcvt_f_f_v_f32m2(vle16_v_f16m1(ptr, vl), vl);
                        _sum = vfadd_vv_f32m2(_sum, _val, vl);
                        ptr += packn;
                    }

                    vfloat32m2_t _avg = vfmul_vf_f32m2(_sum, 1.f / size, vl);

                    __fp16* outptr = top_blob;
                    vse16_v_f16m1(outptr + q * packn, vfncvt_f_f_w_f16m1(_avg, vl), vl);
                }
            }

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

                        vfloat16m1_t _max = vfmv_v_f_f16m1((__fp16)-FLT_MAX, vl);

                        for (int k = 0; k < maxk; k++)
                        {
                            vfloat16m1_t _val = vle16_v_f16m1(sptr + space_ofs[k] * packn, vl);
                            _max = vfmax_vv_f16m1(_max, _val, vl);
                        }

                        vse16_v_f16m1(outptr + j * packn, _max, vl);
                    }

                    outptr += outw * packn;
                }
            }
        }

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

                            vfloat32m2_t _sum = vfmv_v_f_f32m2(0.f, vl);
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

                                    vfloat32m2_t _val = vfwcvt_f_f_v_f32m2(vle16_v_f16m1(m.row<const __fp16>(sy) + sx * packn, vl), vl);
                                    _sum = vfadd_vv_f32m2(_sum, _val, vl);
                                    area += 1;
                                }
                            }

                            vfloat32m2_t _avg = vfmul_vf_f32m2(_sum, 1.f / area, vl);
                            vse16_v_f16m1(outptr + j * packn, vfncvt_f_f_w_f16m1(_avg, vl), vl);
                        }

                        outptr += outw * packn;
                    }
                }
            }

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

                            vfloat32m2_t _sum = vfmv_v_f_f32m2(0.f, vl);

                            for (int k = 0; k < maxk; k++)
                            {
                                vfloat32m2_t _val = vfwcvt_f_f_v_f32m2(vle16_v_f16m1(sptr + space_ofs[k] * packn, vl), vl);
                                _sum = vfadd_vv_f32m2(_sum, _val, vl);
                            }

                            vfloat32m2_t _avg = vfmul_vf_f32m2(_sum, inv_maxk, vl);
                            vse16_v_f16m1(outptr + j * packn, vfncvt_f_f_w_f16m1(_avg, vl), vl);
                        }

                        outptr += outw * packn;
                    }
                }
            }

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

    const int packn = csrr_vlenb() / 2;
    const size_t vl = vsetvl_e16m1(packn);

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

                            vfloat16m1_t _sum = vfmv_v_f_f16m1(0.f, vl);
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

                                    vfloat16m1_t _val = vle16_v_f16m1(m.row<const __fp16>(sy) + sx * packn, vl);
                                    _sum = vfadd_vv_f16m1(_sum, _val, vl);
                                    area += 1;
                                }
                            }

                            vfloat16m1_t _avg = vfmul_vf_f16m1(_sum, (__fp16)(1.f / area), vl);
                            vse16_v_f16m1(outptr + j * packn, _avg, vl);
                        }

                        outptr += outw * packn;
                    }
                }
            }

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

                            vfloat16m1_t _sum = vfmv_v_f_f16m1(0.f, vl);

                            for (int k = 0; k < maxk; k++)
                            {
                                vfloat16m1_t _val = vle16_v_f16m1(sptr + space_ofs[k] * packn, vl);
                                _sum = vfadd_vv_f16m1(_sum, _val, vl);
                            }

                            vfloat16m1_t _avg = vfmul_vf_f16m1(_sum, inv_maxk, vl);
                            vse16_v_f16m1(outptr + j * packn, _avg, vl);
                        }

                        outptr += outw * packn;
                    }
                }
            }

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
#endif // __riscv_vector && __riscv_zfh

} // namespace ncnn
