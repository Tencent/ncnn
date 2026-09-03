// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#include "deconvolution_4x4_fp16s.h"

static void deconvolution_fp16s(const Mat& bottom_blob, Mat& top_blob_bordered, const Mat& weight_data_tm, const Mat& bias_data, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int num_output, int bias_term, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int outw = top_blob_bordered.w;
    const int outh = top_blob_bordered.h;
    const int out_elempack = top_blob_bordered.elempack;
    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    const int maxk = kernel_w * kernel_h;

    if (elempack == 4 && out_elempack == 4)
    {
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output / out_elempack; p++)
            {
                __fp16* outptr = top_blob_bordered.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float32x4_t _sum = vdupq_n_f32(0.f);

                        if (bias_term)
                        {
                            _sum = vld1q_f32(((const float*)bias_data) + p * 4);
                        }

                        const __fp16* kptr = weight_data_tm.channel(p);

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob.channel(q);

                            for (int y = 0; y < kernel_h; y++)
                            {
                                int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                                if (sys < 0 || sys % stride_h != 0)
                                    continue;

                                int sy = sys / stride_h;
                                if (sy >= h)
                                    continue;

                                for (int x = 0; x < kernel_w; x++)
                                {
                                    int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                    if (sxs < 0 || sxs % stride_w != 0)
                                        continue;

                                    int sx = sxs / stride_w;
                                    if (sx >= w)
                                        continue;

                                    const __fp16* sptr = m.row<const __fp16>(sy) + sx * 4;

                                    float32x4_t _val = vcvt_f32_f16(vld1_f16(sptr));

                                    int k = y * kernel_w + x;

                                    float32x4_t _w0 = vcvt_f32_f16(vld1_f16(kptr + k * 16));
                                    float32x4_t _w1 = vcvt_f32_f16(vld1_f16(kptr + k * 16 + 4));
                                    float32x4_t _w2 = vcvt_f32_f16(vld1_f16(kptr + k * 16 + 8));
                                    float32x4_t _w3 = vcvt_f32_f16(vld1_f16(kptr + k * 16 + 12));

                                    _sum = vfmaq_laneq_f32(_sum, _w0, _val, 0);
                                    _sum = vfmaq_laneq_f32(_sum, _w1, _val, 1);
                                    _sum = vfmaq_laneq_f32(_sum, _w2, _val, 2);
                                    _sum = vfmaq_laneq_f32(_sum, _w3, _val, 3);
                                }
                            }

                            kptr += maxk * 16;
                        }

                        _sum = activation_ps(_sum, activation_type, activation_params);

                        vst1_f16(outptr + j * 4, vcvt_f16_f32(_sum));
                    }

                    outptr += outw * 4;
                }
            }
        }
    }

    if (elempack == 1 && out_elempack == 4)
    {
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output / out_elempack; p++)
            {
                __fp16* outptr = top_blob_bordered.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float32x4_t _sum = vdupq_n_f32(0.f);

                        if (bias_term)
                        {
                            _sum = vld1q_f32(((const float*)bias_data) + p * 4);
                        }

                        const __fp16* kptr = weight_data_tm.channel(p);

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob.channel(q);

                            for (int y = 0; y < kernel_h; y++)
                            {
                                int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                                if (sys < 0 || sys % stride_h != 0)
                                    continue;

                                int sy = sys / stride_h;
                                if (sy >= h)
                                    continue;

                                const __fp16* sptr = m.row<const __fp16>(sy);

                                for (int x = 0; x < kernel_w; x++)
                                {
                                    int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                    if (sxs < 0 || sxs % stride_w != 0)
                                        continue;

                                    int sx = sxs / stride_w;
                                    if (sx >= w)
                                        continue;

                                    float32x4_t _val = vdupq_n_f32((float)sptr[sx]);

                                    int k = y * kernel_w + x;

                                    float32x4_t _w = vcvt_f32_f16(vld1_f16(kptr + k * 4));

                                    _sum = vfmaq_f32(_sum, _val, _w);
                                }
                            }

                            kptr += maxk * 4;
                        }

                        _sum = activation_ps(_sum, activation_type, activation_params);

                        vst1_f16(outptr + j * 4, vcvt_f16_f32(_sum));
                    }

                    outptr += outw * 4;
                }
            }
        }
    }

    if (elempack == 4 && out_elempack == 1)
    {
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output / out_elempack; p++)
            {
                __fp16* outptr = top_blob_bordered.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float sum = 0.f;

                        if (bias_term)
                        {
                            sum = bias_data[p];
                        }

                        const __fp16* kptr = weight_data_tm.channel(p);

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob.channel(q);

                            for (int y = 0; y < kernel_h; y++)
                            {
                                int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                                if (sys < 0 || sys % stride_h != 0)
                                    continue;

                                int sy = sys / stride_h;
                                if (sy >= h)
                                    continue;

                                for (int x = 0; x < kernel_w; x++)
                                {
                                    int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                    if (sxs < 0 || sxs % stride_w != 0)
                                        continue;

                                    int sx = sxs / stride_w;
                                    if (sx >= w)
                                        continue;

                                    const __fp16* sptr = m.row<const __fp16>(sy) + sx * 4;

                                    float32x4_t _val = vcvt_f32_f16(vld1_f16(sptr));

                                    int k = y * kernel_w + x;

                                    float32x4_t _w = vcvt_f32_f16(vld1_f16(kptr + k * 4));

                                    float32x4_t _s4 = vmulq_f32(_val, _w);

                                    sum += vaddvq_f32(_s4); // dot
                                }
                            }

                            kptr += maxk * 4;
                        }

                        sum = activation_ss(sum, activation_type, activation_params);

                        outptr[j] = (__fp16)sum;
                    }

                    outptr += outw;
                }
            }
        }
    }

    if (elempack == 1 && out_elempack == 1)
    {
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output; p++)
            {
                __fp16* outptr = top_blob_bordered.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float sum = 0.f;

                        if (bias_term)
                        {
                            sum = bias_data[p];
                        }

                        const __fp16* kptr = weight_data_tm.channel(p);

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob.channel(q);

                            for (int y = 0; y < kernel_h; y++)
                            {
                                int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                                if (sys < 0 || sys % stride_h != 0)
                                    continue;

                                int sy = sys / stride_h;
                                if (sy >= h)
                                    continue;

                                const __fp16* sptr = m.row<const __fp16>(sy);

                                for (int x = 0; x < kernel_w; x++)
                                {
                                    int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                    if (sxs < 0 || sxs % stride_w != 0)
                                        continue;

                                    int sx = sxs / stride_w;
                                    if (sx >= w)
                                        continue;

                                    float val = (float)sptr[sx];

                                    int k = y * kernel_w + x;

                                    float w = (float)kptr[k];

                                    sum += val * w;
                                }
                            }

                            kptr += maxk;
                        }

                        sum = activation_ss(sum, activation_type, activation_params);

                        outptr[j] = (__fp16)sum;
                    }

                    outptr += outw;
                }
            }
        }
    }
}

static void deconvolution_col2im_fp16sa(const Mat& top_col2im, Mat& top_blob_bordered, const Mat& bias_data, const Mat& bias_data_fp16, int input_w, int input_h, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
{
    const int w = input_w;
    const int h = input_h;
    const int outw = top_blob_bordered.w;
    const int out_elempack = top_blob_bordered.elempack;
    const int out_channels = top_blob_bordered.c;
    const int maxk = kernel_w * kernel_h;

// col2im
    const int gap = (outw * stride_h - w * stride_w) * out_elempack;

    if (out_elempack == 8)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < out_channels; p++)
        {
            const __fp16* sptr = top_col2im.row<const __fp16>(p * maxk);
            Mat outm = top_blob_bordered.channel(p);

            if (bias_data.empty())
            {
                outm.fill(vdupq_n_f16(0.f));
            }
            else
            {
                outm.fill(vld1q_f16((const __fp16*)bias_data_fp16 + p * 8));
            }

            for (int u = 0; u < kernel_h; u++)
            {
                for (int v = 0; v < kernel_w; v++)
                {
                    __fp16* ptr = outm.row<__fp16>(dilation_h * u) + dilation_w * v * 8;

                    for (int i = 0; i < h; i++)
                    {
                        for (int j = 0; j < w; j++)
                        {
                            float16x8_t _val = vld1q_f16(ptr);
                            float16x8_t _s = vld1q_f16(sptr);
                            _val = vaddq_f16(_val, _s);
                            vst1q_f16(ptr, _val);

                            ptr += stride_w * 8;
                            sptr += 8;
                        }

                        ptr += gap;
                    }
                }
            }
        }
    }

    if (out_elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < out_channels; p++)
        {
            const __fp16* sptr = top_col2im.row<const __fp16>(p * maxk);
            Mat outm = top_blob_bordered.channel(p);

            if (bias_data.empty())
            {
                outm.fill(vdup_n_f16(0.f));
            }
            else
            {
                outm.fill(vld1_f16((const __fp16*)bias_data_fp16 + p * 4));
            }

            for (int u = 0; u < kernel_h; u++)
            {
                for (int v = 0; v < kernel_w; v++)
                {
                    __fp16* ptr = outm.row<__fp16>(dilation_h * u) + dilation_w * v * 4;

                    for (int i = 0; i < h; i++)
                    {
                        for (int j = 0; j < w; j++)
                        {
                            float16x4_t _val = vld1_f16(ptr);
                            float16x4_t _s = vld1_f16(sptr);
                            _val = vadd_f16(_val, _s);
                            vst1_f16(ptr, _val);

                            ptr += stride_w * 4;
                            sptr += 4;
                        }

                        ptr += gap;
                    }
                }
            }
        }
    }

    if (out_elempack == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < out_channels; p++)
        {
            const __fp16* sptr = top_col2im.row<const __fp16>(p * maxk);
            Mat outm = top_blob_bordered.channel(p);

            const __fp16 bias = bias_data_fp16.empty() ? 0.f : ((const __fp16*)bias_data_fp16)[p];
            outm.fill(bias);

            for (int u = 0; u < kernel_h; u++)
            {
                for (int v = 0; v < kernel_w; v++)
                {
                    __fp16* ptr = outm.row<__fp16>(dilation_h * u) + dilation_w * v;

                    for (int i = 0; i < h; i++)
                    {
                        for (int j = 0; j < w; j++)
                        {
                            ptr[0] += sptr[0];

                            ptr += stride_w;
                            sptr += 1;
                        }

                        ptr += gap;
                    }
                }
            }
        }
    }
}

static bool deconvolution_fp16sa(const Mat& bottom_blob, Mat& top_blob_bordered, const Mat& weight_data_tm, const Mat& bias_data, const Mat& bias_data_fp16, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int num_output, int bias_term, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;
    const int outw = top_blob_bordered.w;
    const int outh = top_blob_bordered.h;
    const int out_elempack = top_blob_bordered.elempack;
    const int out_channels = top_blob_bordered.c;
    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;
    const int maxk = kernel_w * kernel_h;

if (elempack == 8 && out_elempack == 8)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < out_channels; p++)
        {
            __fp16* outptr = top_blob_bordered.channel(p);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    float16x8_t _sum = vdupq_n_f16((__fp16)0.f);

                    if (bias_term)
                    {
                        _sum = vld1q_f16((const __fp16*)bias_data_fp16 + p * 8);
                    }

                    const __fp16* kptr = weight_data_tm.channel(p);

                    // channels
                    for (int q = 0; q < channels; q++)
                    {
                        const Mat m = bottom_blob.channel(q);

                        for (int y = 0; y < kernel_h; y++)
                        {
                            int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                            if (sys < 0 || sys % stride_h != 0)
                                continue;

                            int sy = sys / stride_h;
                            if (sy >= h)
                                continue;

                            for (int x = 0; x < kernel_w; x++)
                            {
                                int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                if (sxs < 0 || sxs % stride_w != 0)
                                    continue;

                                int sx = sxs / stride_w;
                                if (sx >= w)
                                    continue;

                                const __fp16* sptr = m.row<const __fp16>(sy) + sx * 8;

                                float16x8_t _val = vld1q_f16(sptr);

                                int k = y * kernel_w + x;

                                float16x8_t _w0 = vld1q_f16(kptr + k * 64);
                                float16x8_t _w1 = vld1q_f16(kptr + k * 64 + 8);
                                float16x8_t _w2 = vld1q_f16(kptr + k * 64 + 16);
                                float16x8_t _w3 = vld1q_f16(kptr + k * 64 + 24);
                                float16x8_t _w4 = vld1q_f16(kptr + k * 64 + 32);
                                float16x8_t _w5 = vld1q_f16(kptr + k * 64 + 40);
                                float16x8_t _w6 = vld1q_f16(kptr + k * 64 + 48);
                                float16x8_t _w7 = vld1q_f16(kptr + k * 64 + 56);

                                _sum = vfmaq_laneq_f16(_sum, _w0, _val, 0);
                                _sum = vfmaq_laneq_f16(_sum, _w1, _val, 1);
                                _sum = vfmaq_laneq_f16(_sum, _w2, _val, 2);
                                _sum = vfmaq_laneq_f16(_sum, _w3, _val, 3);
                                _sum = vfmaq_laneq_f16(_sum, _w4, _val, 4);
                                _sum = vfmaq_laneq_f16(_sum, _w5, _val, 5);
                                _sum = vfmaq_laneq_f16(_sum, _w6, _val, 6);
                                _sum = vfmaq_laneq_f16(_sum, _w7, _val, 7);
                            }
                        }

                        kptr += maxk * 64;
                    }

                    _sum = activation_ps_f16(_sum, activation_type, activation_params);

                    vst1q_f16(outptr + j * 8, _sum);
                }

                outptr += outw * 8;
            }
        }
    }

    if (elempack == 1 && out_elempack == 8)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < out_channels; p++)
        {
            __fp16* outptr = top_blob_bordered.channel(p);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    float16x8_t _sum = vdupq_n_f16((__fp16)0.f);

                    if (bias_term)
                    {
                        _sum = vld1q_f16((const __fp16*)bias_data_fp16 + p * 8);
                    }

                    const __fp16* kptr = weight_data_tm.channel(p);

                    // channels
                    for (int q = 0; q < channels; q++)
                    {
                        const Mat m = bottom_blob.channel(q);

                        for (int y = 0; y < kernel_h; y++)
                        {
                            int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                            if (sys < 0 || sys % stride_h != 0)
                                continue;

                            int sy = sys / stride_h;
                            if (sy >= h)
                                continue;

                            const __fp16* sptr = m.row<const __fp16>(sy);

                            for (int x = 0; x < kernel_w; x++)
                            {
                                int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                if (sxs < 0 || sxs % stride_w != 0)
                                    continue;

                                int sx = sxs / stride_w;
                                if (sx >= w)
                                    continue;

                                float16x8_t _val = vdupq_n_f16(sptr[sx]);

                                int k = y * kernel_w + x;

                                float16x8_t _w = vld1q_f16(kptr + k * 8);

                                _sum = vfmaq_f16(_sum, _val, _w);
                            }
                        }

                        kptr += maxk * 8;
                    }

                    _sum = activation_ps_f16(_sum, activation_type, activation_params);

                    vst1q_f16(outptr + j * 8, _sum);
                }

                outptr += outw * 8;
            }
        }
    }

    if (elempack == 4 && out_elempack == 8)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < out_channels; p++)
        {
            __fp16* outptr = top_blob_bordered.channel(p);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    float16x8_t _sum = vdupq_n_f16((__fp16)0.f);

                    if (bias_term)
                    {
                        _sum = vld1q_f16((const __fp16*)bias_data_fp16 + p * 8);
                    }

                    const __fp16* kptr = weight_data_tm.channel(p);

                    // channels
                    for (int q = 0; q < channels; q++)
                    {
                        const Mat m = bottom_blob.channel(q);

                        for (int y = 0; y < kernel_h; y++)
                        {
                            int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                            if (sys < 0 || sys % stride_h != 0)
                                continue;

                            int sy = sys / stride_h;
                            if (sy >= h)
                                continue;

                            for (int x = 0; x < kernel_w; x++)
                            {
                                int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                if (sxs < 0 || sxs % stride_w != 0)
                                    continue;

                                int sx = sxs / stride_w;
                                if (sx >= w)
                                    continue;

                                const __fp16* sptr = m.row<const __fp16>(sy) + sx * 4;

                                float16x4_t _val = vld1_f16(sptr);

                                int k = y * kernel_w + x;

                                float16x8_t _w0 = vld1q_f16(kptr + k * 32);
                                float16x8_t _w1 = vld1q_f16(kptr + k * 32 + 8);
                                float16x8_t _w2 = vld1q_f16(kptr + k * 32 + 16);
                                float16x8_t _w3 = vld1q_f16(kptr + k * 32 + 24);

                                _sum = vfmaq_lane_f16(_sum, _w0, _val, 0);
                                _sum = vfmaq_lane_f16(_sum, _w1, _val, 1);
                                _sum = vfmaq_lane_f16(_sum, _w2, _val, 2);
                                _sum = vfmaq_lane_f16(_sum, _w3, _val, 3);
                            }
                        }

                        kptr += maxk * 32;
                    }

                    _sum = activation_ps_f16(_sum, activation_type, activation_params);

                    vst1q_f16(outptr + j * 8, _sum);
                }

                outptr += outw * 8;
            }
        }
    }

    if (elempack == 8 && out_elempack == 1)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < out_channels; p++)
        {
            __fp16* outptr = top_blob_bordered.channel(p);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    float sum = 0.f;

                    if (bias_term)
                    {
                        sum = bias_data[p];
                    }

                    const __fp16* kptr = weight_data_tm.channel(p);

                    // channels
                    for (int q = 0; q < channels; q++)
                    {
                        const Mat m = bottom_blob.channel(q);

                        for (int y = 0; y < kernel_h; y++)
                        {
                            int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                            if (sys < 0 || sys % stride_h != 0)
                                continue;

                            int sy = sys / stride_h;
                            if (sy >= h)
                                continue;

                            for (int x = 0; x < kernel_w; x++)
                            {
                                int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                if (sxs < 0 || sxs % stride_w != 0)
                                    continue;

                                int sx = sxs / stride_w;
                                if (sx >= w)
                                    continue;

                                const __fp16* sptr = m.row<const __fp16>(sy) + sx * 8;

                                float16x8_t _val = vld1q_f16(sptr);

                                int k = y * kernel_w + x;

                                float16x8_t _w = vld1q_f16(kptr + k * 8);

                                float16x8_t _s8 = vmulq_f16(_val, _w);

                                float16x4_t _s4 = vadd_f16(vget_low_f16(_s8), vget_high_f16(_s8));
                                sum += vaddvq_f32(vcvt_f32_f16(_s4)); // dot
                            }
                        }

                        kptr += maxk * 8;
                    }

                    sum = activation_ss_f16(sum, activation_type, activation_params);

                    outptr[j] = (__fp16)sum;
                }

                outptr += outw;
            }
        }
    }

    if (elempack == 8 && out_elempack == 4)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < out_channels; p++)
        {
            __fp16* outptr = top_blob_bordered.channel(p);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    float16x4_t _sum = vdup_n_f16((__fp16)0.f);

                    if (bias_term)
                    {
                        _sum = vld1_f16((const __fp16*)bias_data_fp16 + p * 4);
                    }

                    const __fp16* kptr = weight_data_tm.channel(p);

                    // channels
                    for (int q = 0; q < channels; q++)
                    {
                        const Mat m = bottom_blob.channel(q);

                        for (int y = 0; y < kernel_h; y++)
                        {
                            int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                            if (sys < 0 || sys % stride_h != 0)
                                continue;

                            int sy = sys / stride_h;
                            if (sy >= h)
                                continue;

                            for (int x = 0; x < kernel_w; x++)
                            {
                                int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                if (sxs < 0 || sxs % stride_w != 0)
                                    continue;

                                int sx = sxs / stride_w;
                                if (sx >= w)
                                    continue;

                                const __fp16* sptr = m.row<const __fp16>(sy) + sx * 8;

                                float16x8_t _val = vld1q_f16(sptr);

                                int k = y * kernel_w + x;

                                float16x4_t _w0 = vld1_f16(kptr + k * 32);
                                float16x4_t _w1 = vld1_f16(kptr + k * 32 + 4);
                                float16x4_t _w2 = vld1_f16(kptr + k * 32 + 8);
                                float16x4_t _w3 = vld1_f16(kptr + k * 32 + 12);
                                float16x4_t _w4 = vld1_f16(kptr + k * 32 + 16);
                                float16x4_t _w5 = vld1_f16(kptr + k * 32 + 20);
                                float16x4_t _w6 = vld1_f16(kptr + k * 32 + 24);
                                float16x4_t _w7 = vld1_f16(kptr + k * 32 + 28);

                                _sum = vfma_laneq_f16(_sum, _w0, _val, 0);
                                _sum = vfma_laneq_f16(_sum, _w1, _val, 1);
                                _sum = vfma_laneq_f16(_sum, _w2, _val, 2);
                                _sum = vfma_laneq_f16(_sum, _w3, _val, 3);
                                _sum = vfma_laneq_f16(_sum, _w4, _val, 4);
                                _sum = vfma_laneq_f16(_sum, _w5, _val, 5);
                                _sum = vfma_laneq_f16(_sum, _w6, _val, 6);
                                _sum = vfma_laneq_f16(_sum, _w7, _val, 7);
                            }
                        }

                        kptr += maxk * 32;
                    }

                    _sum = activation_ps_f16(_sum, activation_type, activation_params);

                    vst1_f16(outptr + j * 4, _sum);
                }

                outptr += outw * 4;
            }
        }
    }

    if (elempack == 4 && out_elempack == 4)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < out_channels; p++)
        {
            __fp16* outptr = top_blob_bordered.channel(p);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    float16x4_t _sum = vdup_n_f16((__fp16)0.f);

                    if (bias_term)
                    {
                        _sum = vld1_f16((const __fp16*)bias_data_fp16 + p * 4);
                    }

                    const __fp16* kptr = weight_data_tm.channel(p);

                    // channels
                    for (int q = 0; q < channels; q++)
                    {
                        const Mat m = bottom_blob.channel(q);

                        for (int y = 0; y < kernel_h; y++)
                        {
                            int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                            if (sys < 0 || sys % stride_h != 0)
                                continue;

                            int sy = sys / stride_h;
                            if (sy >= h)
                                continue;

                            for (int x = 0; x < kernel_w; x++)
                            {
                                int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                if (sxs < 0 || sxs % stride_w != 0)
                                    continue;

                                int sx = sxs / stride_w;
                                if (sx >= w)
                                    continue;

                                const __fp16* sptr = m.row<const __fp16>(sy) + sx * 4;

                                float16x4_t _val = vld1_f16(sptr);

                                int k = y * kernel_w + x;

                                float16x4_t _w0 = vld1_f16(kptr + k * 16);
                                float16x4_t _w1 = vld1_f16(kptr + k * 16 + 4);
                                float16x4_t _w2 = vld1_f16(kptr + k * 16 + 8);
                                float16x4_t _w3 = vld1_f16(kptr + k * 16 + 12);

                                _sum = vfma_lane_f16(_sum, _w0, _val, 0);
                                _sum = vfma_lane_f16(_sum, _w1, _val, 1);
                                _sum = vfma_lane_f16(_sum, _w2, _val, 2);
                                _sum = vfma_lane_f16(_sum, _w3, _val, 3);
                            }
                        }

                        kptr += maxk * 16;
                    }

                    _sum = activation_ps_f16(_sum, activation_type, activation_params);

                    vst1_f16(outptr + j * 4, _sum);
                }

                outptr += outw * 4;
            }
        }
    }

    if (elempack == 1 && out_elempack == 4)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < out_channels; p++)
        {
            __fp16* outptr = top_blob_bordered.channel(p);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    float16x4_t _sum = vdup_n_f16((__fp16)0.f);

                    if (bias_term)
                    {
                        _sum = vld1_f16((const __fp16*)bias_data_fp16 + p * 4);
                    }

                    const __fp16* kptr = weight_data_tm.channel(p);

                    // channels
                    for (int q = 0; q < channels; q++)
                    {
                        const Mat m = bottom_blob.channel(q);

                        for (int y = 0; y < kernel_h; y++)
                        {
                            int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                            if (sys < 0 || sys % stride_h != 0)
                                continue;

                            int sy = sys / stride_h;
                            if (sy >= h)
                                continue;

                            const __fp16* sptr = m.row<const __fp16>(sy);

                            for (int x = 0; x < kernel_w; x++)
                            {
                                int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                if (sxs < 0 || sxs % stride_w != 0)
                                    continue;

                                int sx = sxs / stride_w;
                                if (sx >= w)
                                    continue;

                                float16x4_t _val = vdup_n_f16(sptr[sx]);

                                int k = y * kernel_w + x;

                                float16x4_t _w = vld1_f16(kptr + k * 4);

                                _sum = vfma_f16(_sum, _val, _w);
                            }
                        }

                        kptr += maxk * 4;
                    }

                    _sum = activation_ps_f16(_sum, activation_type, activation_params);

                    vst1_f16(outptr + j * 4, _sum);
                }

                outptr += outw * 4;
            }
        }
    }

    if (elempack == 4 && out_elempack == 1)
    {
        // num_output
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < out_channels; p++)
        {
            __fp16* outptr = top_blob_bordered.channel(p);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    float sum = 0.f;

                    if (bias_term)
                    {
                        sum = bias_data[p];
                    }

                    const __fp16* kptr = weight_data_tm.channel(p);

                    // channels
                    for (int q = 0; q < channels; q++)
                    {
                        const Mat m = bottom_blob.channel(q);

                        for (int y = 0; y < kernel_h; y++)
                        {
                            int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                            if (sys < 0 || sys % stride_h != 0)
                                continue;

                            int sy = sys / stride_h;
                            if (sy >= h)
                                continue;

                            for (int x = 0; x < kernel_w; x++)
                            {
                                int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                if (sxs < 0 || sxs % stride_w != 0)
                                    continue;

                                int sx = sxs / stride_w;
                                if (sx >= w)
                                    continue;

                                const __fp16* sptr = m.row<const __fp16>(sy) + sx * 4;

                                float16x4_t _val = vld1_f16(sptr);

                                int k = y * kernel_w + x;

                                float16x4_t _w = vld1_f16(kptr + k * 4);

                                float16x4_t _s4 = vmul_f16(_val, _w);

                                sum += vaddvq_f32(vcvt_f32_f16(_s4)); // dot
                            }
                        }

                        kptr += maxk * 4;
                    }

                    sum = activation_ss_f16(sum, activation_type, activation_params);

                    outptr[j] = (__fp16)sum;
                }

                outptr += outw;
            }
        }
    }

    if (elempack == 1 && out_elempack == 1)
    {
        if (kernel_w == 4 && kernel_h == 4 && stride_w == 2 && stride_h == 2 && dilation_w == 1 && dilation_h == 1)
        {
            deconv4x4s2_fp16sa_neon(bottom_blob, top_blob_bordered, weight_data_tm, bias_data_fp16, opt);

            return true;
        }
        else
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output; p++)
            {
                __fp16* outptr = top_blob_bordered.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float sum = 0.f;

                        if (bias_term)
                        {
                            sum = bias_data[p];
                        }

                        const __fp16* kptr = weight_data_tm.channel(p);

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob.channel(q);

                            for (int y = 0; y < kernel_h; y++)
                            {
                                int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                                if (sys < 0 || sys % stride_h != 0)
                                    continue;

                                int sy = sys / stride_h;
                                if (sy >= h)
                                    continue;

                                const __fp16* sptr = m.row<const __fp16>(sy);

                                for (int x = 0; x < kernel_w; x++)
                                {
                                    int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                    if (sxs < 0 || sxs % stride_w != 0)
                                        continue;

                                    int sx = sxs / stride_w;
                                    if (sx >= w)
                                        continue;

                                    __fp16 val = sptr[sx];

                                    int k = y * kernel_w + x;

                                    __fp16 w = kptr[k];

                                    sum += val * w;
                                }
                            }

                            kptr += maxk;
                        }

                        sum = activation_ss_f16(sum, activation_type, activation_params);

                        outptr[j] = (__fp16)sum;
                    }

                    outptr += outw;
                }
            }
        }
    }

    return false;
}

#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

