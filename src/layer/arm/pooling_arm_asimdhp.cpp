// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "pooling_arm.h"

#include <float.h>

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn {

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
int Pooling_arm::forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // max value in NxN window
    // avg value in NxN window

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
            if (elempack == 8)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = bottom_blob.channel(q);

                    float16x8_t _max = vdupq_n_f16((__fp16)-FLT_MAX);
                    for (int i = 0; i < size; i++)
                    {
                        float16x8_t _val = vld1q_f16(ptr);
                        _max = vmaxq_f16(_max, _val);
                        ptr += 8;
                    }

                    __fp16* outptr = top_blob;
                    vst1q_f16(outptr + q * 8, _max);
                }
            }

            if (elempack == 4)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = bottom_blob.channel(q);

                    float16x4_t _max = vdup_n_f16((__fp16)-FLT_MAX);
                    for (int i = 0; i < size; i++)
                    {
                        float16x4_t _val = vld1_f16(ptr);
                        _max = vmax_f16(_max, _val);
                        ptr += 4;
                    }

                    __fp16* outptr = top_blob;
                    vst1_f16(outptr + q * 4, _max);
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
            if (elempack == 8)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = bottom_blob.channel(q);

                    float32x4_t _sum0 = vdupq_n_f32(0.f);
                    float32x4_t _sum1 = vdupq_n_f32(0.f);
                    for (int i = 0; i < size; i++)
                    {
                        float16x8_t _val = vld1q_f16(ptr);
                        _sum0 = vaddq_f32(_sum0, vcvt_f32_f16(vget_low_f16(_val)));
                        _sum1 = vaddq_f32(_sum1, vcvt_f32_f16(vget_high_f16(_val)));
                        ptr += 8;
                    }

                    float32x4_t _inv_size = vdupq_n_f32(1.f / size);
                    float32x4_t _avg0 = vmulq_f32(_sum0, _inv_size);
                    float32x4_t _avg1 = vmulq_f32(_sum1, _inv_size);

                    __fp16* outptr = top_blob;
                    vst1q_f16(outptr + q * 8, vcombine_f16(vcvt_f16_f32(_avg0), vcvt_f16_f32(_avg1)));
                }
            }

            if (elempack == 4)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const __fp16* ptr = bottom_blob.channel(q);

                    float32x4_t _sum = vdupq_n_f32(0.f);
                    for (int i = 0; i < size; i++)
                    {
                        float32x4_t _val = vcvt_f32_f16(vld1_f16(ptr));
                        _sum = vaddq_f32(_sum, _val);
                        ptr += 4;
                    }

                    float32x4_t _inv_size = vdupq_n_f32(1.f / size);
                    float32x4_t _avg = vmulq_f32(_sum, _inv_size);

                    __fp16* outptr = top_blob;
                    vst1_f16(outptr + q * 4, vcvt_f16_f32(_avg));
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
        if (elempack == 8)
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
                        const __fp16* sptr = m.row<const __fp16>(i * stride_h) + j * stride_w * 8;

                        float16x8_t _max = vdupq_n_f16((__fp16)-FLT_MAX);

                        for (int k = 0; k < maxk; k++)
                        {
                            float16x8_t _val = vld1q_f16(sptr + space_ofs[k] * 8);
                            _max = vmaxq_f16(_max, _val);
                        }

                        vst1q_f16(outptr + j * 8, _max);
                    }

                    outptr += outw * 8;
                }
            }
        }

        if (elempack == 4)
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
                        const __fp16* sptr = m.row<const __fp16>(i * stride_h) + j * stride_w * 4;

                        float16x4_t _max = vdup_n_f16((__fp16)-FLT_MAX);

                        for (int k = 0; k < maxk; k++)
                        {
                            float16x4_t _val = vld1_f16(sptr + space_ofs[k] * 4);
                            _max = vmax_f16(_max, _val);
                        }

                        vst1_f16(outptr + j * 4, _max);
                    }

                    outptr += outw * 4;
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

            if (elempack == 4)
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

                            float32x4_t _sum = vdupq_n_f32(0.f);
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

                                    float32x4_t _val = vcvt_f32_f16(vld1_f16(m.row<const __fp16>(sy) + sx * 4));
                                    _sum = vaddq_f32(_sum, _val);
                                    area += 1;
                                }
                            }

                            float32x4_t _inv_area = vdupq_n_f32(1.f / area);
                            float32x4_t _avg = vmulq_f32(_sum, _inv_area);
                            vst1_f16(outptr + j * 4, vcvt_f16_f32(_avg));
                        }

                        outptr += outw * 4;
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
            if (elempack == 4)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob_bordered.channel(q);
                    __fp16* outptr = top_blob.channel(q);

                    float32x4_t _inv_maxk = vdupq_n_f32(1.f / maxk);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            const __fp16* sptr = m.row<const __fp16>(i * stride_h) + j * stride_w * 4;

                            float32x4_t _sum = vdupq_n_f32(0.f);

                            for (int k = 0; k < maxk; k++)
                            {
                                float32x4_t _val = vcvt_f32_f16(vld1_f16(sptr + space_ofs[k] * 4));
                                _sum = vaddq_f32(_sum, _val);
                            }

                            float32x4_t _avg = vmulq_f32(_sum, _inv_maxk);
                            vst1_f16(outptr + j * 4, vcvt_f16_f32(_avg));
                        }

                        outptr += outw * 4;
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

int Pooling_arm::forward_fp16sa(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // max value in NxN window
    // avg value in NxN window

    if (pooling_type == PoolMethod_MAX || global_pooling)
    {
        return forward_fp16s(bottom_blob, top_blob, opt);
    }

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

            if (elempack == 8)
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

                            float16x8_t _sum = vdupq_n_f16((__fp16)0.f);
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

                                    float16x8_t _val = vld1q_f16(m.row<const __fp16>(sy) + sx * 8);
                                    _sum = vaddq_f16(_sum, _val);
                                    area += 1;
                                }
                            }

                            float16x8_t _inv_area = vdupq_n_f16((__fp16)(1.f / area));
                            float16x8_t _avg = vmulq_f16(_sum, _inv_area);
                            vst1q_f16(outptr + j * 8, _avg);
                        }

                        outptr += outw * 8;
                    }
                }
            }

            if (elempack == 4)
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

                            float16x4_t _sum = vdup_n_f16((__fp16)0.f);
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

                                    float16x4_t _val = vld1_f16(m.row<const __fp16>(sy) + sx * 4);
                                    _sum = vadd_f16(_sum, _val);
                                    area += 1;
                                }
                            }

                            float16x4_t _inv_area = vdup_n_f16((__fp16)(1.f / area));
                            float16x4_t _avg = vmul_f16(_sum, _inv_area);
                            vst1_f16(outptr + j * 4, _avg);
                        }

                        outptr += outw * 4;
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
            if (elempack == 8)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob_bordered.channel(q);
                    __fp16* outptr = top_blob.channel(q);

                    float16x8_t _inv_maxk = vdupq_n_f16((__fp16)(1.f / maxk));

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            const __fp16* sptr = m.row<const __fp16>(i * stride_h) + j * stride_w * 8;

                            float16x8_t _sum = vdupq_n_f16((__fp16)0.f);

                            for (int k = 0; k < maxk; k++)
                            {
                                float16x8_t _val = vld1q_f16(sptr + space_ofs[k] * 8);
                                _sum = vaddq_f16(_sum, _val);
                            }

                            float16x8_t _avg = vmulq_f16(_sum, _inv_maxk);
                            vst1q_f16(outptr + j * 8, _avg);
                        }

                        outptr += outw * 8;
                    }
                }
            }

            if (elempack == 4)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob_bordered.channel(q);
                    __fp16* outptr = top_blob.channel(q);

                    float16x4_t _inv_maxk = vdup_n_f16((__fp16)(1.f / maxk));

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            const __fp16* sptr = m.row<const __fp16>(i * stride_h) + j * stride_w * 4;

                            float16x4_t _sum = vdup_n_f16((__fp16)0.f);

                            for (int k = 0; k < maxk; k++)
                            {
                                float16x4_t _val = vld1_f16(sptr + space_ofs[k] * 4);
                                _sum = vadd_f16(_sum, _val);
                            }

                            float16x4_t _avg = vmul_f16(_sum, _inv_maxk);
                            vst1_f16(outptr + j * 4, _avg);
                        }

                        outptr += outw * 4;
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
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

} // namespace ncnn
