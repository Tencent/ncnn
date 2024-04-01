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

#include "deconvolution_arm.h"

#include "layer_type.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "arm_activation.h"
#include "arm_usability.h"

#include "cpu.h"

namespace ncnn {

#include "deconvolution_3x3.h"
#include "deconvolution_4x4.h"

Deconvolution_arm::Deconvolution_arm()
{
#if __ARM_NEON
    support_packing = true;
#if NCNN_ARM82
    support_fp16_storage = cpu_support_arm_asimdhp();
#endif
#endif // __ARM_NEON

#if NCNN_BF16
    support_bf16_storage = true;
#endif

    activation = 0;
    gemm = 0;
}

int Deconvolution_arm::create_pipeline(const Option& opt)
{
    if (dynamic_weight)
        return 0;

    activation = create_activation_layer(activation_type, activation_params, opt);

#if NCNN_ARM82
    if (support_fp16_storage && opt.use_fp16_storage)
    {
        return create_pipeline_fp16s(opt);
    }
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage)
    {
        return create_pipeline_bf16s(opt);
    }
#endif

    const int maxk = kernel_w * kernel_h;
    int num_input = weight_data_size / maxk / num_output;

    int elempack = 1;
    int out_elempack = 1;
#if __ARM_NEON
    if (opt.use_packing_layout)
    {
        elempack = num_input % 4 == 0 ? 4 : 1;
        out_elempack = num_output % 4 == 0 ? 4 : 1;
    }
#endif

    if (opt.use_sgemm_convolution)
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

        Option opt1 = opt;
        opt1.use_fp16_storage = false;
        gemm->create_pipeline(opt1);
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

        weight_data_tm.create(maxk, num_input / elempack, num_output / out_elempack, (size_t)4u * elempack * out_elempack, elempack * out_elempack);

        for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
        {
            float* g00 = weight_data_tm.channel(q / out_elempack);

            for (int p = 0; p + (elempack - 1) < num_input; p += elempack)
            {
                for (int k = 0; k < maxk; k++)
                {
                    for (int i = 0; i < elempack; i++)
                    {
                        for (int j = 0; j < out_elempack; j++)
                        {
                            const float* k00 = weight_data_r2.channel(q + j).row(p + i);

                            g00[0] = k00[k];

                            g00++;
                        }
                    }
                }
            }
        }

        // pack1
        if (elempack == 1 && out_elempack == 1)
        {
            if (kernel_w == 3 && kernel_h == 3 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1)
            {
                weight_data_tm = weight_data;
            }
            else if (kernel_w == 3 && kernel_h == 3 && stride_w == 2 && stride_h == 2 && dilation_w == 1 && dilation_h == 1)
            {
                weight_data_tm = weight_data;
            }
            else if (kernel_w == 4 && kernel_h == 4 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1)
            {
                weight_data_tm = weight_data;
            }
            else if (kernel_w == 4 && kernel_h == 4 && stride_w == 2 && stride_h == 2 && dilation_w == 1 && dilation_h == 1)
            {
                weight_data_tm = weight_data;
            }
            else
            {
                weight_data_tm = weight_data_transposed;
            }
        }
    }

    weight_data.release();

    return 0;
}

int Deconvolution_arm::destroy_pipeline(const Option& opt)
{
    if (activation)
    {
        activation->destroy_pipeline(opt);
        delete activation;
        activation = 0;
    }

    if (gemm)
    {
        gemm->destroy_pipeline(opt);
        delete gemm;
        gemm = 0;
    }

    return 0;
}

int Deconvolution_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int elembits = bottom_blob.elembits();

#if NCNN_ARM82
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
#if __ARM_NEON
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % 4 == 0 ? 4 : 1;
    }
#endif
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

#if __ARM_NEON
            if (out_elempack == 4)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int p = 0; p < out_channels; p++)
                {
                    const float* sptr = top_col2im.row(p * maxk);
                    Mat outm = top_blob_bordered.channel(p);

                    if (bias_data.empty())
                    {
                        outm.fill(vdupq_n_f32(0.f));
                    }
                    else
                    {
                        outm.fill(vld1q_f32((const float*)bias_data + p * 4));
                    }

                    for (int u = 0; u < kernel_h; u++)
                    {
                        for (int v = 0; v < kernel_w; v++)
                        {
                            float* ptr = outm.row(dilation_h * u) + dilation_w * v * 4;

                            for (int i = 0; i < h; i++)
                            {
                                for (int j = 0; j < w; j++)
                                {
                                    float32x4_t _val = vld1q_f32(ptr);
                                    float32x4_t _s = vld1q_f32(sptr);
                                    _val = vaddq_f32(_val, _s);
                                    vst1q_f32(ptr, _val);

                                    ptr += stride_w * 4;
                                    sptr += 4;
                                }

                                ptr += gap;
                            }
                        }
                    }
                }
            }
#endif // __ARM_NEON

            if (out_elempack == 1)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int p = 0; p < out_channels; p++)
                {
                    const float* sptr = top_col2im.row(p * maxk);
                    Mat outm = top_blob_bordered.channel(p);

                    const float bias = bias_data.empty() ? 0.f : bias_data[p];
                    outm.fill(bias);

                    for (int u = 0; u < kernel_h; u++)
                    {
                        for (int v = 0; v < kernel_w; v++)
                        {
                            float* ptr = outm.row(dilation_h * u) + dilation_w * v;

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
#if __ARM_NEON
        if (elempack == 4 && out_elempack == 4)
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < out_channels; p++)
            {
                float* outptr = top_blob_bordered.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float32x4_t _sum = vdupq_n_f32(0.f);

                        if (bias_term)
                        {
                            _sum = vld1q_f32(((const float*)bias_data) + p * 4);
                        }

                        const float* kptr = weight_data_tm.channel(p);

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

                                    const float* sptr = m.row(sy) + sx * 4;

                                    float32x4_t _val = vld1q_f32(sptr);

                                    int k = y * kernel_w + x;

                                    float32x4_t _w0 = vld1q_f32(kptr + k * 16);
                                    float32x4_t _w1 = vld1q_f32(kptr + k * 16 + 4);
                                    float32x4_t _w2 = vld1q_f32(kptr + k * 16 + 8);
                                    float32x4_t _w3 = vld1q_f32(kptr + k * 16 + 12);

#if __aarch64__
                                    _sum = vmlaq_laneq_f32(_sum, _w0, _val, 0);
                                    _sum = vmlaq_laneq_f32(_sum, _w1, _val, 1);
                                    _sum = vmlaq_laneq_f32(_sum, _w2, _val, 2);
                                    _sum = vmlaq_laneq_f32(_sum, _w3, _val, 3);
#else
                                    _sum = vmlaq_lane_f32(_sum, _w0, vget_low_f32(_val), 0);
                                    _sum = vmlaq_lane_f32(_sum, _w1, vget_low_f32(_val), 1);
                                    _sum = vmlaq_lane_f32(_sum, _w2, vget_high_f32(_val), 0);
                                    _sum = vmlaq_lane_f32(_sum, _w3, vget_high_f32(_val), 1);
#endif
                                }
                            }

                            kptr += maxk * 16;
                        }

                        _sum = activation_ps(_sum, activation_type, activation_params);

                        vst1q_f32(outptr + j * 4, _sum);
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
                float* outptr = top_blob_bordered.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float32x4_t _sum = vdupq_n_f32(0.f);

                        if (bias_term)
                        {
                            _sum = vld1q_f32(((const float*)bias_data) + p * 4);
                        }

                        const float* kptr = weight_data_tm.channel(p);

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

                                const float* sptr = m.row(sy);

                                for (int x = 0; x < kernel_w; x++)
                                {
                                    int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                    if (sxs < 0 || sxs % stride_w != 0)
                                        continue;

                                    int sx = sxs / stride_w;
                                    if (sx >= w)
                                        continue;

                                    float32x4_t _val = vdupq_n_f32(sptr[sx]);

                                    int k = y * kernel_w + x;

                                    float32x4_t _w = vld1q_f32(kptr + k * 4);

                                    _sum = vmlaq_f32(_sum, _val, _w);
                                }
                            }

                            kptr += maxk * 4;
                        }

                        _sum = activation_ps(_sum, activation_type, activation_params);

                        vst1q_f32(outptr + j * 4, _sum);
                    }

                    outptr += outw * 4;
                }
            }
        }

        if (elempack == 4 && out_elempack == 1)
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output; p++)
            {
                float* outptr = top_blob_bordered.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float sum = 0.f;

                        if (bias_term)
                        {
                            sum = bias_data[p];
                        }

                        const float* kptr = weight_data_tm.channel(p);

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

                                    const float* sptr = m.row(sy) + sx * 4;

                                    float32x4_t _val = vld1q_f32(sptr);

                                    int k = y * kernel_w + x;

                                    float32x4_t _w = vld1q_f32(kptr + k * 4);

                                    float32x4_t _s4 = vmulq_f32(_val, _w);
#if __aarch64__
                                    sum += vaddvq_f32(_s4); // dot
#else
                                    float32x2_t _ss = vadd_f32(vget_low_f32(_s4), vget_high_f32(_s4));
                                    _ss = vpadd_f32(_ss, _ss);
                                    sum += vget_lane_f32(_ss, 0);
#endif
                                }
                            }

                            kptr += maxk * 4;
                        }

                        sum = activation_ss(sum, activation_type, activation_params);

                        outptr[j] = sum;
                    }

                    outptr += outw;
                }
            }
        }
#endif // __ARM_NEON

        if (elempack == 1 && out_elempack == 1)
        {
            if (kernel_w == 3 && kernel_h == 3 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1)
            {
                deconv3x3s1_neon(bottom_blob, top_blob_bordered, weight_data_tm, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob_bordered, opt);
                }
            }
            else if (kernel_w == 3 && kernel_h == 3 && stride_w == 2 && stride_h == 2 && dilation_w == 1 && dilation_h == 1)
            {
                deconv3x3s2_neon(bottom_blob, top_blob_bordered, weight_data_tm, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob_bordered, opt);
                }
            }
            else if (kernel_w == 4 && kernel_h == 4 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1)
            {
                deconv4x4s1_neon(bottom_blob, top_blob_bordered, weight_data_tm, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob_bordered, opt);
                }
            }
            else if (kernel_w == 4 && kernel_h == 4 && stride_w == 2 && stride_h == 2 && dilation_w == 1 && dilation_h == 1)
            {
                deconv4x4s2_neon(bottom_blob, top_blob_bordered, weight_data_tm, bias_data, opt);

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
                    float* outptr = top_blob_bordered.channel(p);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            float sum = 0.f;

                            if (bias_term)
                            {
                                sum = bias_data[p];
                            }

                            const float* kptr = (const float*)weight_data_tm + maxk * channels * p;

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

                                    const float* sptr = m.row(sy);

                                    for (int x = 0; x < kernel_w; x++)
                                    {
                                        int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                        if (sxs < 0 || sxs % stride_w != 0)
                                            continue;

                                        int sx = sxs / stride_w;
                                        if (sx >= w)
                                            continue;

                                        float val = sptr[sx];

                                        int k = y * kernel_w + x;

                                        float w = kptr[k];

                                        sum += val * w;
                                    }
                                }

                                kptr += maxk;
                            }

                            sum = activation_ss(sum, activation_type, activation_params);

                            outptr[j] = sum;
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

int Deconvolution_arm::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& _weight_data = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];

    const int _num_input = bottom_blob.c * bottom_blob.elempack;
    const int _kernel_w = _weight_data.w;
    const int _kernel_h = _weight_data.h;
    const int _num_output = _weight_data.d * 1;

    Mat weight_data_flattened;
    flatten(_weight_data, weight_data_flattened, opt);
    if (weight_data_flattened.empty())
        return -100;

#if NCNN_ARM82
    if (opt.use_fp16_storage && cpu_support_arm_asimdhp() && weight_data_flattened.elembits() == 16)
    {
        Mat weight_data_flattened_fp32;
        cast_float16_to_float32(weight_data_flattened, weight_data_flattened_fp32, opt);
        weight_data_flattened = weight_data_flattened_fp32;
    }
#endif // NCNN_ARM82
#if NCNN_BF16
    if (opt.use_bf16_storage && weight_data_flattened.elembits() == 16)
    {
        Mat weight_data_flattened_fp32;
        cast_bfloat16_to_float32(weight_data_flattened, weight_data_flattened_fp32, opt);
        weight_data_flattened = weight_data_flattened_fp32;
    }
#endif // NCNN_BF16

    // weight_data_flattened as pack1
    weight_data_flattened.w *= weight_data_flattened.elempack;
    weight_data_flattened.elemsize /= weight_data_flattened.elempack;
    weight_data_flattened.elempack = 1;

    // transpose group-inch/group-outch/group-kh-kw to group-outch/group-inch/group-kh-kw
    Mat weight_data_transposed;
    {
        weight_data_transposed.create(_kernel_w * _kernel_h * _num_output * _num_input / 1, 4u, opt.workspace_allocator);
        if (weight_data_transposed.empty())
            return -100;

        const int outch_g = _num_output / 1;
        const int inch_g = _num_input / 1;
        const int maxk = _kernel_h * _kernel_w;

        for (int g = 0; g < 1; g++)
        {
            // reorder weight from inch-outch to outch-inch
            float* wg2 = (float*)weight_data_transposed + g * outch_g * inch_g * maxk;
            const float* wg = (const float*)weight_data_flattened + g * inch_g * outch_g * maxk;
            for (int i = 0; i < outch_g; i++)
            {
                for (int j = 0; j < inch_g; j++)
                {
                    for (int k = 0; k < maxk; k++)
                    {
                        wg2[(i * inch_g + j) * maxk + k] = wg[(j * outch_g + i) * maxk + k];
                    }
                }
            }
        }
    }

    Mat bias_data_flattened;
    if (bias_term)
    {
        const Mat& _bias_data = bottom_blobs[2];
        flatten(_bias_data, bias_data_flattened, opt);
        if (bias_data_flattened.empty())
            return -100;

#if NCNN_ARM82
        if (opt.use_fp16_storage && cpu_support_arm_asimdhp() && bias_data_flattened.elembits() == 16)
        {
            Mat bias_data_flattened_fp32;
            cast_float16_to_float32(bias_data_flattened, bias_data_flattened_fp32, opt);
            bias_data_flattened = bias_data_flattened_fp32;
        }
#endif // NCNN_ARM82
#if NCNN_BF16
        if (opt.use_bf16_storage && bias_data_flattened.elembits() == 16)
        {
            Mat bias_data_flattened_fp32;
            cast_bfloat16_to_float32(bias_data_flattened, bias_data_flattened_fp32, opt);
            bias_data_flattened = bias_data_flattened_fp32;
        }
#endif // NCNN_BF16

        // bias_data_flattened as pack1
        bias_data_flattened.w *= bias_data_flattened.elempack;
        bias_data_flattened.elemsize /= bias_data_flattened.elempack;
        bias_data_flattened.elempack = 1;
    }

    ncnn::Layer* op = ncnn::create_layer_cpu(ncnn::LayerType::Deconvolution);

    ncnn::ParamDict pd;
    pd.set(0, _num_output);
    pd.set(1, _kernel_w);
    pd.set(11, _kernel_h);
    pd.set(2, dilation_w);
    pd.set(12, dilation_h);
    pd.set(3, stride_w);
    pd.set(13, stride_h);
    pd.set(4, pad_left);
    pd.set(15, pad_right);
    pd.set(14, pad_top);
    pd.set(16, pad_bottom);
    pd.set(18, output_pad_right);
    pd.set(19, output_pad_bottom);
    pd.set(20, output_w);
    pd.set(21, output_h);
    pd.set(5, bias_term);
    pd.set(6, weight_data_transposed.w);
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    op->load_param(pd);

    ncnn::Mat weights[2];
    weights[0] = weight_data_transposed;
    weights[1] = bias_data_flattened;

    op->load_model(ncnn::ModelBinFromMatArray(weights));

    op->create_pipeline(opt);

    op->forward(bottom_blob, top_blob, opt);

    op->destroy_pipeline(opt);

    delete op;

    return 0;
}

#if NCNN_BF16
int Deconvolution_arm::create_pipeline_bf16s(const Option& opt)
{
    const int maxk = kernel_w * kernel_h;
    const int num_input = weight_data_size / maxk / num_output;

    int elempack = 1;
    int out_elempack = 1;
#if __ARM_NEON
    if (opt.use_packing_layout)
    {
        elempack = num_input % 4 == 0 ? 4 : 1;
        out_elempack = num_output % 4 == 0 ? 4 : 1;
    }
#endif

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
    {
        Mat weight_data_r2 = weight_data_transposed.reshape(maxk, num_input, num_output);

        weight_data_tm.create(maxk, num_input / elempack, num_output / out_elempack, (size_t)2u * elempack * out_elempack, elempack * out_elempack);

        for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
        {
            unsigned short* g00 = weight_data_tm.channel(q / out_elempack);

            for (int p = 0; p + (elempack - 1) < num_input; p += elempack)
            {
                for (int k = 0; k < maxk; k++)
                {
                    for (int i = 0; i < elempack; i++)
                    {
                        for (int j = 0; j < out_elempack; j++)
                        {
                            const float* k00 = weight_data_r2.channel(q + j).row(p + i);

                            g00[0] = float32_to_bfloat16(k00[k]);

                            g00++;
                        }
                    }
                }
            }
        }
    }

    weight_data.release();

    return 0;
}

int Deconvolution_arm::forward_bf16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
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
#if __ARM_NEON
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % 4 == 0 ? 4 : 1;
    }
#endif
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

#if __ARM_NEON
    if (elempack == 4 && out_elempack == 4)
    {
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output / out_elempack; p++)
            {
                unsigned short* outptr = top_blob_bordered.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float32x4_t _sum = vdupq_n_f32(0.f);

                        if (bias_term)
                        {
                            _sum = vld1q_f32(((const float*)bias_data) + p * 4);
                        }

                        const unsigned short* kptr = weight_data_tm.channel(p);

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

                                    const unsigned short* sptr = m.row<const unsigned short>(sy) + sx * 4;

                                    float32x4_t _val = bfloat2float(vld1_u16(sptr));

                                    int k = y * kernel_w + x;

                                    float32x4_t _w0 = bfloat2float(vld1_u16(kptr + k * 16));
                                    float32x4_t _w1 = bfloat2float(vld1_u16(kptr + k * 16 + 4));
                                    float32x4_t _w2 = bfloat2float(vld1_u16(kptr + k * 16 + 8));
                                    float32x4_t _w3 = bfloat2float(vld1_u16(kptr + k * 16 + 12));

#if __aarch64__
                                    _sum = vmlaq_laneq_f32(_sum, _w0, _val, 0);
                                    _sum = vmlaq_laneq_f32(_sum, _w1, _val, 1);
                                    _sum = vmlaq_laneq_f32(_sum, _w2, _val, 2);
                                    _sum = vmlaq_laneq_f32(_sum, _w3, _val, 3);
#else
                                    _sum = vmlaq_lane_f32(_sum, _w0, vget_low_f32(_val), 0);
                                    _sum = vmlaq_lane_f32(_sum, _w1, vget_low_f32(_val), 1);
                                    _sum = vmlaq_lane_f32(_sum, _w2, vget_high_f32(_val), 0);
                                    _sum = vmlaq_lane_f32(_sum, _w3, vget_high_f32(_val), 1);
#endif
                                }
                            }

                            kptr += maxk * 16;
                        }

                        _sum = activation_ps(_sum, activation_type, activation_params);

                        vst1_u16(outptr + j * 4, float2bfloat(_sum));
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
                unsigned short* outptr = top_blob_bordered.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float32x4_t _sum = vdupq_n_f32(0.f);

                        if (bias_term)
                        {
                            _sum = vld1q_f32(((const float*)bias_data) + p * 4);
                        }

                        const unsigned short* kptr = weight_data_tm.channel(p);

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

                                const unsigned short* sptr = m.row<const unsigned short>(sy);

                                for (int x = 0; x < kernel_w; x++)
                                {
                                    int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                    if (sxs < 0 || sxs % stride_w != 0)
                                        continue;

                                    int sx = sxs / stride_w;
                                    if (sx >= w)
                                        continue;

                                    float32x4_t _val = vdupq_n_f32(bfloat16_to_float32(sptr[sx]));

                                    int k = y * kernel_w + x;

                                    float32x4_t _w = bfloat2float(vld1_u16(kptr + k * 4));

                                    _sum = vmlaq_f32(_sum, _val, _w);
                                }
                            }

                            kptr += maxk * 4;
                        }

                        _sum = activation_ps(_sum, activation_type, activation_params);

                        vst1_u16(outptr + j * 4, float2bfloat(_sum));
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
                unsigned short* outptr = top_blob_bordered.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float sum = 0.f;

                        if (bias_term)
                        {
                            sum = bias_data[p];
                        }

                        const unsigned short* kptr = weight_data_tm.channel(p);

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

                                    const unsigned short* sptr = m.row<const unsigned short>(sy) + sx * 4;

                                    float32x4_t _val = bfloat2float(vld1_u16(sptr));

                                    int k = y * kernel_w + x;

                                    float32x4_t _w = bfloat2float(vld1_u16(kptr + k * 4));

                                    float32x4_t _s4 = vmulq_f32(_val, _w);
#if __aarch64__
                                    sum += vaddvq_f32(_s4); // dot
#else
                                    float32x2_t _ss = vadd_f32(vget_low_f32(_s4), vget_high_f32(_s4));
                                    _ss = vpadd_f32(_ss, _ss);
                                    sum += vget_lane_f32(_ss, 0);
#endif
                                }
                            }

                            kptr += maxk * 4;
                        }

                        sum = activation_ss(sum, activation_type, activation_params);

                        outptr[j] = float32_to_bfloat16(sum);
                    }

                    outptr += outw;
                }
            }
        }
    }
#endif // __ARM_NEON

    if (elempack == 1 && out_elempack == 1)
    {
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output; p++)
            {
                unsigned short* outptr = top_blob_bordered.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float sum = 0.f;

                        if (bias_term)
                        {
                            sum = bias_data[p];
                        }

                        const unsigned short* kptr = weight_data_tm.channel(p);

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

                                const unsigned short* sptr = m.row<const unsigned short>(sy);

                                for (int x = 0; x < kernel_w; x++)
                                {
                                    int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                    if (sxs < 0 || sxs % stride_w != 0)
                                        continue;

                                    int sx = sxs / stride_w;
                                    if (sx >= w)
                                        continue;

                                    float val = bfloat16_to_float32(sptr[sx]);

                                    int k = y * kernel_w + x;

                                    float w = bfloat16_to_float32(kptr[k]);

                                    sum += val * w;
                                }
                            }

                            kptr += maxk;
                        }

                        sum = activation_ss(sum, activation_type, activation_params);

                        outptr[j] = float32_to_bfloat16(sum);
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
#endif // NCNN_BF16

} // namespace ncnn
