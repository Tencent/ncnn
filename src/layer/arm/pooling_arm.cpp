// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "pooling_2x2.h"
#include "pooling_3x3.h"

#if __ARM_NEON
#include "pooling_2x2_pack4.h"
#include "pooling_3x3_pack4.h"
#endif

DEFINE_LAYER_CREATOR(Pooling_arm)

Pooling_arm::Pooling_arm()
{
#if __ARM_NEON
    support_packing = true;
#endif // __ARM_NEON

    support_bf16_storage = true;
}

int Pooling_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (opt.use_bf16_storage)
        return forward_bf16s(bottom_blob, top_blob, opt);

    // max value in NxN window
    // avg value in NxN window

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

#if __ARM_NEON
//     fprintf(stderr, "Pooling     input %d x %d  pad = %d %d %d %d  ksize=%d %d  stride=%d %d\n", w, h, pad_left, pad_right, pad_top, pad_bottom, kernel_w, kernel_h, stride_w, stride_h);

    if (elempack == 4)
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
                for (int q=0; q<channels; q++)
                {
                    const float* ptr = bottom_blob.channel(q);

                    float32x4_t _max = vld1q_f32(ptr);
                    for (int i=0; i<size; i++)
                    {
                        float32x4_t _val = vld1q_f32(ptr);
                        _max = vmaxq_f32(_max, _val);
                        ptr += 4;
                    }

                    float* outptr = top_blob;
                    vst1q_f32(outptr + q * 4, _max);
                }
            }
            else if (pooling_type == PoolMethod_AVE)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q=0; q<channels; q++)
                {
                    const float* ptr = bottom_blob.channel(q);

                    float32x4_t _sum = vdupq_n_f32(0.f);
                    for (int i=0; i<size; i++)
                    {
                        float32x4_t _val = vld1q_f32(ptr);
                        _sum = vaddq_f32(_sum, _val);
                        ptr += 4;
                    }

                    float32x4_t _inv_size = vdupq_n_f32(1.f / size);
                    float32x4_t _avg = vmulq_f32(_sum, _inv_size);

                    float* outptr = top_blob;
                    vst1q_f32(outptr + q * 4, _avg);
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
            if (kernel_w == 2 && kernel_h == 2 && stride_w == 2 && stride_h == 2)
            {
                pooling2x2s2_max_pack4_neon(bottom_blob_bordered, top_blob, opt);

                return 0;
            }

            if (kernel_w == 3 && kernel_h == 3 && stride_w == 2 && stride_h == 2)
            {
                pooling3x3s2_max_pack4_neon(bottom_blob_bordered, top_blob, opt);

                return 0;
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const Mat m = bottom_blob_bordered.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        const float* sptr = m.row(i*stride_h) + j*stride_w * 4;

                        float32x4_t _max = vld1q_f32(sptr);

                        for (int k = 0; k < maxk; k++)
                        {
                            float32x4_t _val = vld1q_f32( sptr + space_ofs[k] * 4 );
                            _max = vmaxq_f32(_max, _val);
                        }

                        vst1q_f32(outptr + j * 4, _max);
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
                for (int q=0; q<channels; q++)
                {
                    const Mat m = bottom_blob_bordered.channel(q);
                    float* outptr = top_blob.channel(q);

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

                                    float32x4_t _val = vld1q_f32( m.row(sy) + sx * 4 );
                                    _sum = vaddq_f32(_sum, _val);
                                    area += 1;
                                }
                            }

                            float32x4_t _inv_area = vdupq_n_f32(1.f / area);
                            float32x4_t _avg = vmulq_f32(_sum, _inv_area);
                            vst1q_f32(outptr + j * 4, _avg);
                        }

                        outptr += outw * 4;
                    }
                }
            }
            else // if (avgpool_count_include_pad == 1)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q=0; q<channels; q++)
                {
                    const Mat m = bottom_blob_bordered.channel(q);
                    float* outptr = top_blob.channel(q);

                    float32x4_t _inv_maxk = vdupq_n_f32(1.f / maxk);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            const float* sptr = m.row(i*stride_h) + j*stride_w * 4;

                            float32x4_t _sum = vdupq_n_f32(0.f);

                            for (int k = 0; k < maxk; k++)
                            {
                                float32x4_t _val = vld1q_f32( sptr + space_ofs[k] * 4 );
                                _sum = vaddq_f32(_sum, _val);
                            }

                            float32x4_t _avg = vmulq_f32(_sum, _inv_maxk);
                            vst1q_f32(outptr + j * 4, _avg);
                        }

                        outptr += outw * 4;
                    }
                }
            }
        }

        return 0;
    }
#endif // __ARM_NEON

    if (kernel_w != kernel_h || stride_w != stride_h)
    {
        return Pooling::forward(bottom_blob, top_blob, opt);
    }

    const int kernel_size = kernel_w;
    const int stride = stride_w;

    if (pooling_type != PoolMethod_MAX || stride != 2 || global_pooling == 1)
    {
        return Pooling::forward(bottom_blob, top_blob, opt);
    }

    if (kernel_size != 2 && kernel_size != 3)
    {
        return Pooling::forward(bottom_blob, top_blob, opt);
    }

    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_w) / stride_w + 1;
    int outh = (h - kernel_h) / stride_h + 1;

    top_blob.create(outw, outh, channels, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (kernel_size == 2)
        pooling2x2s2_max_neon(bottom_blob_bordered, top_blob, opt);
    if (kernel_size == 3)
        pooling3x3s2_max_neon(bottom_blob_bordered, top_blob, opt);

    return 0;
}

int Pooling_arm::forward_bf16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // max value in NxN window
    // avg value in NxN window

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

//     fprintf(stderr, "Pooling     input %d x %d  pad = %d %d %d %d  ksize=%d %d  stride=%d %d\n", w, h, pad_left, pad_right, pad_top, pad_bottom, kernel_w, kernel_h, stride_w, stride_h);

    if (global_pooling)
    {
        top_blob.create(channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        int size = w * h;

        if (pooling_type == PoolMethod_MAX)
        {
#if __ARM_NEON
            if (elempack == 4)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q=0; q<channels; q++)
                {
                    const unsigned short* ptr = bottom_blob.channel(q);

                    float32x4_t _max = vdupq_n_f32(-FLT_MAX);
                    for (int i=0; i<size; i++)
                    {
                        float32x4_t _val = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(ptr), 16));
                        _max = vmaxq_f32(_max, _val);
                        ptr += 4;
                    }

                    unsigned short* outptr = top_blob;
                    vst1_u16(outptr + q * 4, vshrn_n_u32(vreinterpretq_u32_f32(_max), 16));
                }
            }
#endif // __ARM_NEON

            if (elempack == 1)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q=0; q<channels; q++)
                {
                    const unsigned short* ptr = bottom_blob.channel(q);

                    float max = -FLT_MAX;
                    for (int i=0; i<size; i++)
                    {
                        max = std::max(max, bfloat16_to_float32(ptr[i]));
                    }

                    unsigned short* outptr = top_blob;
                    outptr[q] = float32_to_bfloat16(max);
                }
            }
        }

        if (pooling_type == PoolMethod_AVE)
        {
#if __ARM_NEON
            if (elempack == 4)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q=0; q<channels; q++)
                {
                    const unsigned short* ptr = bottom_blob.channel(q);

                    float32x4_t _sum = vdupq_n_f32(0.f);
                    for (int i=0; i<size; i++)
                    {
                        float32x4_t _val = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(ptr), 16));
                        _sum = vaddq_f32(_sum, _val);
                        ptr += 4;
                    }

                    float32x4_t _inv_size = vdupq_n_f32(1.f / size);
                    float32x4_t _avg = vmulq_f32(_sum, _inv_size);

                    unsigned short* outptr = top_blob;
                    vst1_u16(outptr + q * 4, vshrn_n_u32(vreinterpretq_u32_f32(_avg), 16));
                }
            }
#endif // __ARM_NEON

            if (elempack == 1)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q=0; q<channels; q++)
                {
                    const unsigned short* ptr = bottom_blob.channel(q);

                    float sum = 0.f;
                    for (int i=0; i<size; i++)
                    {
                        sum += bfloat16_to_float32(ptr[i]);
                    }

                    unsigned short* outptr = top_blob;
                    outptr[q] = float32_to_bfloat16(sum / size);
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
#if __ARM_NEON
        if (elempack == 4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const Mat m = bottom_blob_bordered.channel(q);
                unsigned short* outptr = top_blob.channel(q);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        const unsigned short* sptr = m.row<const unsigned short>(i*stride_h) + j*stride_w * 4;

                        float32x4_t _max = vdupq_n_f32(-FLT_MAX);

                        for (int k = 0; k < maxk; k++)
                        {
                            float32x4_t _val = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16( sptr + space_ofs[k] * 4 ), 16));
                            _max = vmaxq_f32(_max, _val);
                        }

                        vst1_u16(outptr + j * 4, vshrn_n_u32(vreinterpretq_u32_f32(_max), 16));
                    }

                    outptr += outw * 4;
                }
            }
        }
#endif // __ARM_NEON

        if (elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const Mat m = bottom_blob_bordered.channel(q);
                unsigned short* outptr = top_blob.channel(q);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        const unsigned short* sptr = m.row<const unsigned short>(i*stride_h) + j*stride_w;

                        float max = sptr[0];

                        for (int k = 0; k < maxk; k++)
                        {
                            float val = bfloat16_to_float32(sptr[ space_ofs[k] ]);
                            max = std::max(max, val);
                        }

                        outptr[j] = float32_to_bfloat16(max);
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

#if __ARM_NEON
            if (elempack == 4)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q=0; q<channels; q++)
                {
                    const Mat m = bottom_blob_bordered.channel(q);
                    unsigned short* outptr = top_blob.channel(q);

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

                                    float32x4_t _val = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16( m.row<const unsigned short>(sy) + sx * 4 ), 16));
                                    _sum = vaddq_f32(_sum, _val);
                                    area += 1;
                                }
                            }

                            float32x4_t _inv_area = vdupq_n_f32(1.f / area);
                            float32x4_t _avg = vmulq_f32(_sum, _inv_area);
                            vst1_u16(outptr + j * 4, vshrn_n_u32(vreinterpretq_u32_f32(_avg), 16));
                        }

                        outptr += outw * 4;
                    }
                }
            }
#endif // __ARM_NEON

            if (elempack == 1)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q=0; q<channels; q++)
                {
                    const Mat m = bottom_blob_bordered.channel(q);
                    unsigned short* outptr = top_blob.channel(q);

                    for (int i = 0; i < outh; i++)
                    {
                        int sy0 = i * stride_h;

                        for (int j = 0; j < outw; j++)
                        {
                            int sx0 = j * stride_w;

                            float sum = 0;
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

                                    float val = bfloat16_to_float32(m.row<const unsigned short>(sy)[sx]);
                                    sum += val;
                                    area += 1;
                                }
                            }

                            outptr[j] = float32_to_bfloat16(sum / area);
                        }

                        outptr += outw;
                    }
                }
            }
        }

        if (avgpool_count_include_pad == 1)
        {
#if __ARM_NEON
            if (elempack == 4)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q=0; q<channels; q++)
                {
                    const Mat m = bottom_blob_bordered.channel(q);
                    unsigned short* outptr = top_blob.channel(q);

                    float32x4_t _inv_maxk = vdupq_n_f32(1.f / maxk);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            const unsigned short* sptr = m.row<const unsigned short>(i*stride_h) + j*stride_w * 4;

                            float32x4_t _sum = vdupq_n_f32(0.f);

                            for (int k = 0; k < maxk; k++)
                            {
                                float32x4_t _val = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16( sptr + space_ofs[k] * 4 ), 16));
                                _sum = vaddq_f32(_sum, _val);
                            }

                            float32x4_t _avg = vmulq_f32(_sum, _inv_maxk);
                            vst1_u16(outptr + j * 4, vshrn_n_u32(vreinterpretq_u32_f32(_avg), 16));
                        }

                        outptr += outw * 4;
                    }
                }
            }
#endif // __ARM_NEON

            if (elempack == 1)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q=0; q<channels; q++)
                {
                    const Mat m = bottom_blob_bordered.channel(q);
                    unsigned short* outptr = top_blob.channel(q);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            const unsigned short* sptr = m.row<const unsigned short>(i*stride_h) + j*stride_w;

                            float sum = 0.f;

                            for (int k = 0; k < maxk; k++)
                            {
                                float val = bfloat16_to_float32(sptr[ space_ofs[k] ]);
                                sum += val;
                            }

                            outptr[j] = float32_to_bfloat16(sum / maxk);
                        }

                        outptr += outw;
                    }
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
