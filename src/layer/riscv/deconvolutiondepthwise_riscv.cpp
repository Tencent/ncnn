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

#include "deconvolutiondepthwise_riscv.h"

#include "layer_type.h"

#if __riscv_vector
#ifdef RVV_SPEC_0_7
#include "riscv_v_071_fix.h"
#else
#include <riscv_vector.h>
#endif
#endif // __riscv_vector

#include "riscv_activation.h"
#include "riscv_usability.h"

namespace ncnn {

DeconvolutionDepthWise_riscv::DeconvolutionDepthWise_riscv()
{
#if __riscv_vector
    support_packing = true;
#if __riscv_zfh
    support_fp16_storage = true;
#endif
#endif // __riscv_vector
}

int DeconvolutionDepthWise_riscv::create_pipeline(const Option& opt)
{
#if __riscv_vector && __riscv_zfh
    if (opt.use_fp16_storage)
    {
        return create_pipeline_fp16s(opt);
    }
#endif

#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
#endif

    const int maxk = kernel_w * kernel_h;
    int channels = (weight_data_size / group) / maxk / (num_output / group) * group;

    // depth-wise
    if (channels == group && group == num_output)
    {
        int elempack = 1;
#if __riscv_vector
        if (opt.use_packing_layout)
        {
            elempack = channels % packn == 0 ? packn : 1;
        }
#endif

        Mat weight_data_transposed(weight_data.w);
        {
            float* pt = weight_data_transposed;
            const float* p = weight_data;

            for (int i = 0; i < (channels / group) * (num_output / group) * group; i++)
            {
                for (int k = 0; k < maxk; k++)
                {
                    pt[maxk - 1 - k] = p[k];
                }

                p += maxk;
                pt += maxk;
            }
        }

#if __riscv_vector
        // packn
        if (elempack == packn)
        {
            Mat weight_data_r2 = weight_data_transposed.reshape(maxk, group);
            convert_packing(weight_data_r2, weight_data_packed, packn);
        }
#endif // __riscv_vector

        if (elempack == 1)
        {
            weight_data_packed = weight_data_transposed;
        }

        return 0;
    }

    // group convolution
    create_group_ops(opt);

    return 0;
}

int DeconvolutionDepthWise_riscv::create_group_ops(const Option& opt)
{
    // create Deconvolution op for each group
    const int maxk = kernel_w * kernel_h;
    int channels = (weight_data_size / group) / maxk / (num_output / group) * group;

    for (int i = 0; i < (int)group_ops.size(); i++)
        delete group_ops[i];

    group_ops.clear();

    const int channels_g = channels / group;
    const int num_output_g = num_output / group;

    group_ops.resize(group);

    for (int g = 0; g < group; g++)
    {
        Mat weight_data_g = weight_data.range(maxk * channels_g * num_output_g * g, maxk * channels_g * num_output_g);
        Mat bias_data_g;
        if (bias_term)
            bias_data_g = bias_data.range(num_output_g * g, num_output_g);

        ncnn::Layer* op = ncnn::create_layer(ncnn::LayerType::Deconvolution);

        // set param
        ncnn::ParamDict pd;
        pd.set(0, num_output_g); // num_output
        pd.set(1, kernel_w);
        pd.set(11, kernel_h);
        pd.set(2, dilation_w);
        pd.set(12, dilation_h);
        pd.set(3, stride_w);
        pd.set(13, stride_h);
        pd.set(4, 0);  // pad_w
        pd.set(14, 0); // pad_h
        pd.set(5, bias_term);
        pd.set(6, maxk * channels_g * num_output_g); // weight_data_size
        pd.set(9, activation_type);
        pd.set(10, activation_params);

        op->load_param(pd);

        // set weights
        if (bias_term)
        {
            ncnn::Mat weights[2];
            weights[0] = weight_data_g;
            weights[1] = bias_data_g;

            op->load_model(ModelBinFromMatArray(weights));
        }
        else
        {
            ncnn::Mat weights[1];
            weights[0] = weight_data_g;

            op->load_model(ModelBinFromMatArray(weights));
        }

        op->create_pipeline(opt);

        group_ops[g] = op;
    }

    return 0;
}

int DeconvolutionDepthWise_riscv::destroy_pipeline(const Option& opt)
{
    for (int i = 0; i < (int)group_ops.size(); i++)
    {
        group_ops[i]->destroy_pipeline(opt);
        delete group_ops[i];
    }
    group_ops.clear();

    return 0;
}

int DeconvolutionDepthWise_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
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

#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
    const word_type vl = vsetvl_e32m1(packn);
#endif

    // convolv with NxN kernel
    // value = value + bias

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    int outw = (w - 1) * stride_w + kernel_extent_w;
    int outh = (h - 1) * stride_h + kernel_extent_h;
    int out_elempack = 1;
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % packn == 0 ? packn : 1;
    }
#endif
    size_t out_elemsize = elemsize / elempack * out_elempack;

    Mat top_blob_bordered;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0 || output_pad_right > 0 || output_pad_bottom > 0 || (output_w > 0 && output_h > 0))
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

    // depth-wise
    if (channels * elempack == group && group == num_output)
    {
#if __riscv_vector
        if (elempack == packn)
        {
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int g = 0; g < channels; g++)
                {
                    float* outptr = top_blob_bordered.channel(g);
                    const float* kptr = (const float*)weight_data_packed + maxk * g * packn;
                    const Mat m = bottom_blob.channel(g);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            vfloat32m1_t _sum = vfmv_v_f_f32m1(0.f, vl);

                            if (bias_term)
                            {
                                _sum = vle32_v_f32m1((const float*)bias_data + g * packn, vl);
                            }

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

                                    const float* sptr = m.row(sy) + sx * packn;

                                    int k = y * kernel_w + x;

                                    vfloat32m1_t _val = vle32_v_f32m1(sptr, vl);
                                    vfloat32m1_t _w = vle32_v_f32m1(kptr + k * packn, vl);
                                    _sum = vfmacc_vv_f32m1(_sum, _val, _w, vl);
                                }
                            }

                            _sum = activation_ps(_sum, activation_type, activation_params, vl);

                            vse32_v_f32m1(outptr + j * packn, _sum, vl);
                        }

                        outptr += outw * packn;
                    }
                }
            }
        }
#endif // __riscv_vector

        if (elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int g = 0; g < channels; g++)
            {
                float* outptr = top_blob_bordered.channel(g);
                const float* kptr = (const float*)weight_data_packed + maxk * g;
                const Mat m = bottom_blob.channel(g);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float sum = 0.f;

                        if (bias_term)
                        {
                            sum = bias_data[g];
                        }

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

                        sum = activation_ss(sum, activation_type, activation_params);

                        outptr[j] = sum;
                    }

                    outptr += outw;
                }
            }
        }
    }
    else
    {
        // group deconvolution
        const int channels_g = channels * elempack / group;
        const int num_output_g = num_output / group;

        int g_elempack = 1;
        int out_g_elempack = 1;
#if __riscv_vector
        if (opt.use_packing_layout)
        {
            g_elempack = channels_g % packn == 0 ? packn : 1;
            out_g_elempack = num_output_g % packn == 0 ? packn : 1;
        }
#endif

        // unpacking
        Mat bottom_blob_unpacked = bottom_blob;
        if (elempack > g_elempack)
        {
            Option opt_p = opt;
            opt_p.blob_allocator = opt.workspace_allocator;
            convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_p);
        }

        Mat top_blob_bordered_unpacked = top_blob_bordered;
        if (out_g_elempack < out_elempack)
        {
            top_blob_bordered_unpacked.create(outw, outh, num_output, out_elemsize / out_elempack, 1, opt.workspace_allocator);
            if (top_blob_bordered_unpacked.empty())
                return -100;
        }

        for (int g = 0; g < group; g++)
        {
            const Mat bottom_blob_g = bottom_blob_unpacked.channel_range(channels_g * g / g_elempack, channels_g / g_elempack);
            Mat top_blob_bordered_g = top_blob_bordered_unpacked.channel_range(num_output_g * g / out_g_elempack, num_output_g / out_g_elempack);

            const ncnn::Layer* op = group_ops[g];

            Option opt_g = opt;
            opt_g.blob_allocator = top_blob_bordered_unpacked.allocator;

            // forward
            op->forward(bottom_blob_g, top_blob_bordered_g, opt_g);
        }

        // packing
        if (out_g_elempack < out_elempack)
        {
            convert_packing(top_blob_bordered_unpacked, top_blob_bordered, 4, opt);
        }
        else
        {
            top_blob_bordered = top_blob_bordered_unpacked;
        }
    }

    cut_padding(top_blob_bordered, top_blob, opt);
    if (top_blob.empty())
        return -100;

    return 0;
}

#if __riscv_vector && __riscv_zfh
int DeconvolutionDepthWise_riscv::create_pipeline_fp16s(const Option& opt)
{
    const int packn = csrr_vlenb() / 2;

    const int maxk = kernel_w * kernel_h;
    int channels = (weight_data_size / group) / maxk / (num_output / group) * group;

    // depth-wise
    if (channels == group && group == num_output)
    {
        int elempack = 1;
        if (opt.use_packing_layout)
        {
            elempack = channels % packn == 0 ? packn : 1;
        }

        Mat weight_data_transposed(weight_data.w);
        {
            float* pt = weight_data_transposed;
            const float* p = weight_data;

            for (int i = 0; i < (channels / group) * (num_output / group) * group; i++)
            {
                for (int k = 0; k < maxk; k++)
                {
                    pt[maxk - 1 - k] = p[k];
                }

                p += maxk;
                pt += maxk;
            }
        }

        // packn
        if (elempack == packn)
        {
            Mat weight_data_r2 = weight_data_transposed.reshape(maxk, group);
            Mat weight_data_r2_packed;
            convert_packing(weight_data_r2, weight_data_r2_packed, packn);

            ncnn::cast_float32_to_float16(weight_data_r2_packed, weight_data_fp16, opt);
        }

        if (elempack == 1)
        {
            ncnn::cast_float32_to_float16(weight_data_transposed, weight_data_fp16, opt);
        }

        ncnn::cast_float32_to_float16(bias_data, bias_data_fp16, opt);

        return 0;
    }

    // group convolution
    create_group_ops(opt);

    return 0;
}

int DeconvolutionDepthWise_riscv::forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int packn = csrr_vlenb() / 2;
    const word_type vl = vsetvl_e16m1(packn);

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    int outw = (w - 1) * stride_w + kernel_extent_w;
    int outh = (h - 1) * stride_h + kernel_extent_h;
    int out_elempack = (opt.use_packing_layout && num_output % packn == 0) ? packn : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    Mat top_blob_bordered;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0 || output_pad_right > 0 || output_pad_bottom > 0 || (output_w > 0 && output_h > 0))
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

    // depth-wise
    if (channels * elempack == group && group == num_output)
    {
        if (elempack == packn)
        {
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int g = 0; g < channels; g++)
                {
                    __fp16* outptr = top_blob_bordered.channel(g);
                    const __fp16* kptr = (const __fp16*)weight_data_fp16 + maxk * g * packn;
                    const Mat m = bottom_blob.channel(g);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            vfloat32m2_t _sum = vfmv_v_f_f32m2(0.f, vl);

                            if (bias_term)
                            {
                                _sum = vle32_v_f32m2((const float*)bias_data + g * packn, vl);
                            }

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

                                    const __fp16* sptr = m.row<const __fp16>(sy) + sx * packn;

                                    int k = y * kernel_w + x;

                                    vfloat16m1_t _val = vle16_v_f16m1(sptr, vl);
                                    vfloat16m1_t _w = vle16_v_f16m1(kptr + k * packn, vl);
                                    _sum = vfwmacc_vv_f32m2(_sum, _val, _w, vl);
                                }
                            }

                            _sum = activation_ps(_sum, activation_type, activation_params, vl);

                            vse16_v_f16m1(outptr + j * packn, vfncvt_f_f_w_f16m1(_sum, vl), vl);
                        }

                        outptr += outw * packn;
                    }
                }
            }
        }

        if (elempack == 1)
        {
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int g = 0; g < channels; g++)
                {
                    __fp16* outptr = top_blob_bordered.channel(g);
                    const __fp16* kptr = (const __fp16*)weight_data_fp16 + maxk * g;
                    const Mat m = bottom_blob.channel(g);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            float sum = 0.f;

                            if (bias_term)
                            {
                                sum = bias_data[g];
                            }

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

                            sum = activation_ss(sum, activation_type, activation_params);

                            outptr[j] = (__fp16)sum;
                        }

                        outptr += outw;
                    }
                }
            }
        }
    }
    else
    {
        // group deconvolution
        const int channels_g = channels * elempack / group;
        const int num_output_g = num_output / group;

        int g_elempack = (opt.use_packing_layout && channels_g % packn == 0) ? packn : 1;
        int out_g_elempack = (opt.use_packing_layout && num_output_g % packn == 0) ? packn : 1;

        // unpacking
        Mat bottom_blob_unpacked = bottom_blob;
        if (elempack > g_elempack)
        {
            Option opt_p = opt;
            opt_p.blob_allocator = opt.workspace_allocator;
            convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_p);
        }

        Mat top_blob_bordered_unpacked = top_blob_bordered;
        if (out_g_elempack < out_elempack)
        {
            top_blob_bordered_unpacked.create(outw, outh, num_output, out_elemsize / out_elempack, 1, opt.workspace_allocator);
            if (top_blob_bordered_unpacked.empty())
                return -100;
        }

        for (int g = 0; g < group; g++)
        {
            const Mat bottom_blob_g = bottom_blob_unpacked.channel_range(channels_g * g / g_elempack, channels_g / g_elempack);
            Mat top_blob_bordered_g = top_blob_bordered_unpacked.channel_range(num_output_g * g / out_g_elempack, num_output_g / out_g_elempack);

            const ncnn::Layer* op = group_ops[g];

            Option opt_g = opt;
            opt_g.blob_allocator = top_blob_bordered_unpacked.allocator;

            // forward
            op->forward(bottom_blob_g, top_blob_bordered_g, opt_g);
        }

        // packing
        if (out_g_elempack < out_elempack)
        {
            convert_packing(top_blob_bordered_unpacked, top_blob_bordered, out_elempack, opt);
        }
        else
        {
            top_blob_bordered = top_blob_bordered_unpacked;
        }
    }

    cut_padding(top_blob_bordered, top_blob, opt);
    if (top_blob.empty())
        return -100;

    return 0;
}

int DeconvolutionDepthWise_riscv::forward_fp16sa(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int packn = csrr_vlenb() / 2;
    const word_type vl = vsetvl_e16m1(packn);

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    int outw = (w - 1) * stride_w + kernel_extent_w;
    int outh = (h - 1) * stride_h + kernel_extent_h;
    int out_elempack = (opt.use_packing_layout && num_output % packn == 0) ? packn : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    Mat top_blob_bordered;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0 || output_pad_right > 0 || output_pad_bottom > 0 || (output_w > 0 && output_h > 0))
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

    // depth-wise
    if (channels * elempack == group && group == num_output)
    {
        if (elempack == packn)
        {
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int g = 0; g < channels; g++)
                {
                    __fp16* outptr = top_blob_bordered.channel(g);
                    const __fp16* kptr = (const __fp16*)weight_data_fp16 + maxk * g * packn;
                    const Mat m = bottom_blob.channel(g);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            vfloat16m1_t _sum = vfmv_v_f_f16m1((__fp16)0.f, vl);

                            if (bias_term)
                            {
                                _sum = vle16_v_f16m1((const __fp16*)bias_data_fp16 + g * packn, vl);
                            }

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

                                    const __fp16* sptr = m.row<const __fp16>(sy) + sx * packn;

                                    int k = y * kernel_w + x;

                                    vfloat16m1_t _val = vle16_v_f16m1(sptr, vl);
                                    vfloat16m1_t _w = vle16_v_f16m1(kptr + k * packn, vl);
                                    _sum = vfmacc_vv_f16m1(_sum, _val, _w, vl);
                                }
                            }

                            _sum = activation_ps(_sum, activation_type, activation_params, vl);

                            vse16_v_f16m1(outptr + j * packn, _sum, vl);
                        }

                        outptr += outw * packn;
                    }
                }
            }
        }

        if (elempack == 1)
        {
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int g = 0; g < channels; g++)
                {
                    __fp16* outptr = top_blob_bordered.channel(g);
                    const __fp16* kptr = (const __fp16*)weight_data_fp16 + maxk * g;
                    const Mat m = bottom_blob.channel(g);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            float sum = 0.f;

                            if (bias_term)
                            {
                                sum = bias_data[g];
                            }

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

                            sum = activation_ss(sum, activation_type, activation_params);

                            outptr[j] = (__fp16)sum;
                        }

                        outptr += outw;
                    }
                }
            }
        }
    }
    else
    {
        // group deconvolution
        const int channels_g = channels * elempack / group;
        const int num_output_g = num_output / group;

        int g_elempack = (opt.use_packing_layout && channels_g % packn == 0) ? packn : 1;
        int out_g_elempack = (opt.use_packing_layout && num_output_g % packn == 0) ? packn : 1;

        // unpacking
        Mat bottom_blob_unpacked = bottom_blob;
        if (elempack > g_elempack)
        {
            Option opt_p = opt;
            opt_p.blob_allocator = opt.workspace_allocator;
            convert_packing(bottom_blob, bottom_blob_unpacked, g_elempack, opt_p);
        }

        Mat top_blob_bordered_unpacked = top_blob_bordered;
        if (out_g_elempack < out_elempack)
        {
            top_blob_bordered_unpacked.create(outw, outh, num_output / out_g_elempack, out_elemsize / out_elempack * out_g_elempack, out_g_elempack, opt.workspace_allocator);
            if (top_blob_bordered_unpacked.empty())
                return -100;
        }

        for (int g = 0; g < group; g++)
        {
            const Mat bottom_blob_g = bottom_blob_unpacked.channel_range(channels_g * g / g_elempack, channels_g / g_elempack);
            Mat top_blob_bordered_g = top_blob_bordered_unpacked.channel_range(num_output_g * g / out_g_elempack, num_output_g / out_g_elempack);

            const ncnn::Layer* op = group_ops[g];

            Option opt_g = opt;
            opt_g.blob_allocator = top_blob_bordered_unpacked.allocator;

            // forward
            op->forward(bottom_blob_g, top_blob_bordered_g, opt_g);
        }

        // packing
        if (out_g_elempack < out_elempack)
        {
            convert_packing(top_blob_bordered_unpacked, top_blob_bordered, out_elempack, opt);
        }
        else
        {
            top_blob_bordered = top_blob_bordered_unpacked;
        }
    }

    cut_padding(top_blob_bordered, top_blob, opt);
    if (top_blob.empty())
        return -100;

    return 0;
}
#endif // __riscv_vector && __riscv_zfh

} // namespace ncnn
