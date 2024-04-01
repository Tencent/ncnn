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

#include "deconvolution_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "arm_activation.h"

namespace ncnn {

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include "deconvolution_4x4_fp16s.h"
#endif

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
int Deconvolution_arm::create_pipeline_fp16s(const Option& opt)
{
    const int maxk = kernel_w * kernel_h;
    const int num_input = weight_data_size / maxk / num_output;

    int elempack = 1;
    int out_elempack = 1;

    if (opt.use_packing_layout)
    {
        elempack = opt.use_fp16_arithmetic && num_input % 8 == 0 ? 8 : num_input % 4 == 0 ? 4 : 1;
        out_elempack = opt.use_fp16_arithmetic && num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
    }

    if (opt.use_fp16_arithmetic && opt.use_sgemm_convolution)
    {
        const int maxk = kernel_w * kernel_h;

        gemm = ncnn::create_layer_cpu(ncnn::LayerType::Gemm);

        ncnn::ParamDict pd;
        pd.set(2, 1);                 // transA
        pd.set(3, 0);                 // transB
        pd.set(4, 1);                 // constantA
        pd.set(5, 0);                 // constantB
        pd.set(6, 1);                 // constantC
        pd.set(7, maxk * num_output); // M = maxk*num_output
        pd.set(8, 0);                 // N = size
        pd.set(9, num_input);         // K = inch
        pd.set(10, -1);               // constant_broadcast_type_C = null
        pd.set(11, 0);                // output_N1M
        pd.set(12, out_elempack);

        gemm->load_param(pd);

        // maxk-inch-outch to pa-maxk-outch/pa-inch
        Mat tmp;
        {
            Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

            tmp.create(maxk * num_output, num_input);

            for (int p = 0; p < num_input; p += 1)
            {
                float* g00 = tmp.row(p);

                for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
                {
                    for (int k = 0; k < maxk; k++)
                    {
                        for (int i = 0; i < out_elempack; i++)
                        {
                            const float* k00 = weight_data_r2.channel(q + i).row(p);
                            g00[0] = k00[k];
                            g00++;
                        }
                    }
                }
            }
        }

        ncnn::Mat weights[1];
        weights[0] = tmp;

        gemm->load_model(ModelBinFromMatArray(weights));

        gemm->create_pipeline(opt);
    }
    else
    {
        Mat weight_data_transposed(weight_data.w);
        {
            float* pt = weight_data_transposed;
            const float* p = weight_data;

            for (int i = 0; i < num_input * num_output; i++)
            {
                for (int k = 0; k < maxk; k++)
                {
                    pt[maxk - 1 - k] = p[k];
                }

                p += maxk;
                pt += maxk;
            }
        }

        // src = kw-kh-inch-outch
        // dst = pb-pa-kw-kh-inch/pa-outch/pb
        Mat weight_data_r2 = weight_data_transposed.reshape(maxk, num_input, num_output);

        weight_data_tm.create(maxk, num_input / elempack, num_output / out_elempack, (size_t)2u * elempack * out_elempack, elempack * out_elempack);

        for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
        {
            __fp16* g00 = weight_data_tm.channel(q / out_elempack);

            for (int p = 0; p + (elempack - 1) < num_input; p += elempack)
            {
                for (int k = 0; k < maxk; k++)
                {
                    for (int i = 0; i < elempack; i++)
                    {
                        for (int j = 0; j < out_elempack; j++)
                        {
                            const float* k00 = weight_data_r2.channel(q + j).row(p + i);

                            g00[0] = (__fp16)k00[k];

                            g00++;
                        }
                    }
                }
            }
        }
    }

    if (elempack == 1 && out_elempack == 1 && opt.use_fp16_arithmetic)
    {
        if (kernel_w == 4 && kernel_h == 4 && stride_w == 2 && stride_h == 2 && dilation_w == 1 && dilation_h == 1)
        {
            ncnn::cast_float32_to_float16(weight_data, weight_data_tm, opt);
        }
    }

    ncnn::cast_float32_to_float16(bias_data, bias_data_fp16, opt);

    weight_data.release();

    return 0;
}

int Deconvolution_arm::forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // deconvolv with NxN kernel
    // value = value + bias

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    //     NCNN_LOGE("Deconvolution input %d x %d  pad = %d %d  ksize=%d %d  stride=%d %d", w, h, pad_w, pad_h, kernel_w, kernel_h, stride_w, stride_h);

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    int outw = (w - 1) * stride_w + kernel_extent_w + output_pad_right;
    int outh = (h - 1) * stride_h + kernel_extent_h + output_pad_bottom;
    int out_elempack = opt.use_packing_layout && num_output % 4 == 0 ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    Mat top_blob_bordered;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0 || (output_w > 0 && output_h > 0))
    {
        top_blob_bordered.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.workspace_allocator);
    }
    else
    {
        top_blob_bordered = top_blob;
        top_blob_bordered.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    }
    if (top_blob_bordered.empty())
        return -100;

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

    cut_padding(top_blob_bordered, top_blob, opt);
    if (top_blob.empty())
        return -100;

    return 0;
}

int Deconvolution_arm::forward_fp16sa(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // deconvolv with NxN kernel
    // value = value + bias

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    //     NCNN_LOGE("Deconvolution input %d x %d  pad = %d %d  ksize=%d %d  stride=%d %d", w, h, pad_w, pad_h, kernel_w, kernel_h, stride_w, stride_h);

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    int outw = (w - 1) * stride_w + kernel_extent_w + output_pad_right;
    int outh = (h - 1) * stride_h + kernel_extent_h + output_pad_bottom;
    int out_elempack = 1;
    if (opt.use_packing_layout)
    {
        out_elempack = opt.use_fp16_arithmetic && num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
    }
    size_t out_elemsize = elemsize / elempack * out_elempack;

    int out_channels = num_output / out_elempack;

    Mat top_blob_bordered;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0 || (output_w > 0 && output_h > 0))
    {
        top_blob_bordered.create(outw, outh, out_channels, out_elemsize, out_elempack, opt.workspace_allocator);
    }
    else
    {
        top_blob_bordered = top_blob;
        top_blob_bordered.create(outw, outh, out_channels, out_elemsize, out_elempack, opt.blob_allocator);
    }
    if (top_blob_bordered.empty())
        return -100;

    const int maxk = kernel_w * kernel_h;

    if (opt.use_sgemm_convolution)
    {
        // sgemm
        Mat bottom_blob_2 = bottom_blob;
        {
            bottom_blob_2.w = bottom_blob.w * bottom_blob.h;
            bottom_blob_2.h = 1;
        }
        Mat top_col2im;
        Option opt_b = opt;
        opt_b.blob_allocator = top_blob_bordered.allocator;
        gemm->forward(bottom_blob_2, top_col2im, opt_b);

        {
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

        if (activation)
        {
            activation->forward_inplace(top_blob_bordered, opt);
        }
    }
    else
    {
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

                if (activation)
                {
                    activation->forward_inplace(top_blob_bordered, opt);
                }
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
    }

    cut_padding(top_blob_bordered, top_blob, opt);
    if (top_blob.empty())
        return -100;

    return 0;
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

} // namespace ncnn
