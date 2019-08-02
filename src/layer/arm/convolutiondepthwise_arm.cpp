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

#include "convolutiondepthwise_arm.h"

#include "layer_type.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "neon_mathfun.h"
#endif // __ARM_NEON

namespace ncnn {

#include "convolutiondepthwise_3x3.h"
#include "convolutiondepthwise_5x5.h"

#include "convolutiondepthwise_3x3_int8.h"

DEFINE_LAYER_CREATOR(ConvolutionDepthWise_arm)

ConvolutionDepthWise_arm::ConvolutionDepthWise_arm()
{
    activation = 0;
}

int ConvolutionDepthWise_arm::create_pipeline(const Option& opt)
{
    Option opt_cpu = opt;
    opt_cpu.use_vulkan_compute = false;

    if (activation_type == 1)
    {
        activation = ncnn::create_layer(ncnn::LayerType::ReLU);

        ncnn::ParamDict pd;
        activation->load_param(pd);
    }
    else if (activation_type == 2)
    {
        activation = ncnn::create_layer(ncnn::LayerType::ReLU);

        ncnn::ParamDict pd;
        pd.set(0, activation_params[0]);// slope
        activation->load_param(pd);
    }
    else if (activation_type == 3)
    {
        activation = ncnn::create_layer(ncnn::LayerType::Clip);

        ncnn::ParamDict pd;
        pd.set(0, activation_params[0]);// min
        pd.set(1, activation_params[1]);// max
        activation->load_param(pd);
    }
    else if (activation_type == 4)
    {
        activation = ncnn::create_layer(ncnn::LayerType::Sigmoid);

        ncnn::ParamDict pd;
        activation->load_param(pd);
    }

    if (activation)
    {
        activation->create_pipeline(opt_cpu);
    }

    // create Convolution op for each group
    const int maxk = kernel_w * kernel_h;
    int channels = (weight_data_size / group) / maxk / (num_output / group) * group;

    if (opt.use_packing_layout)
    {

    // depth-wise
    if (channels == group && group == num_output)
    {
        // pack4
        if (num_output % 4 == 0)
        {
            Mat weight_data_r2 = weight_data.reshape(maxk, group);
            convert_packing(weight_data_r2, weight_data_pack4, 4);
        }
    }

    // group convolution
    const int channels_g = channels / group;
    const int num_output_g = num_output / group;

    // pack4
    if (channels_g % 4 == 0 && num_output_g % 4 == 0)
    {
        // src = kw-kh-inch-outch
        // dst = 4a-4b-kw-kh-inch/4a-outch/4b
        {
            Mat weight_data_r2_groups = weight_data.reshape(maxk, channels_g, num_output_g * group);

            weight_data_pack4_groups.create(maxk, channels_g/4, num_output_g/4 * group, (size_t)4*16, 16);

            for (int g=0; g<group; g++)
            {
                const Mat weight_data_r2 = weight_data_r2_groups.channel_range(num_output_g * g, num_output_g);

                Mat weight_data_pack4_g = weight_data_pack4_groups.channel_range(num_output_g/4 * g, num_output_g/4);

                for (int q=0; q+3<num_output_g; q+=4)
                {
                    const Mat k0 = weight_data_r2.channel(q);
                    const Mat k1 = weight_data_r2.channel(q+1);
                    const Mat k2 = weight_data_r2.channel(q+2);
                    const Mat k3 = weight_data_r2.channel(q+3);

                    Mat g0 = weight_data_pack4_g.channel(q/4);

                    for (int p=0; p+3<channels_g; p+=4)
                    {
                        const float* k00 = k0.row(p);
                        const float* k01 = k0.row(p+1);
                        const float* k02 = k0.row(p+2);
                        const float* k03 = k0.row(p+3);

                        const float* k10 = k1.row(p);
                        const float* k11 = k1.row(p+1);
                        const float* k12 = k1.row(p+2);
                        const float* k13 = k1.row(p+3);

                        const float* k20 = k2.row(p);
                        const float* k21 = k2.row(p+1);
                        const float* k22 = k2.row(p+2);
                        const float* k23 = k2.row(p+3);

                        const float* k30 = k3.row(p);
                        const float* k31 = k3.row(p+1);
                        const float* k32 = k3.row(p+2);
                        const float* k33 = k3.row(p+3);

                        float* g00 = g0.row(p/4);

                        for (int k=0; k<maxk; k++)
                        {
                            g00[0] = k00[k];
                            g00[1] = k10[k];
                            g00[2] = k20[k];
                            g00[3] = k30[k];

                            g00[4] = k01[k];
                            g00[5] = k11[k];
                            g00[6] = k21[k];
                            g00[7] = k31[k];

                            g00[8] = k02[k];
                            g00[9] = k12[k];
                            g00[10] = k22[k];
                            g00[11] = k32[k];

                            g00[12] = k03[k];
                            g00[13] = k13[k];
                            g00[14] = k23[k];
                            g00[15] = k33[k];

                            g00 += 16;
                        }
                    }
                }
            }
        }
    }

    // pack1to4
    if (channels_g % 4 != 0 && num_output_g % 4 == 0)
    {
        // src = kw-kh-inch-outch
        // dst = 4b-kw-kh-inch-outch/4b
        {
            Mat weight_data_r2_groups = weight_data.reshape(maxk, channels_g, num_output_g * group);

            weight_data_pack1to4_groups.create(maxk, channels_g, num_output_g/4 * group, (size_t)4*4, 4);

            for (int g=0; g<group; g++)
            {
                const Mat weight_data_r2 = weight_data_r2_groups.channel_range(num_output_g * g, num_output_g);

                Mat weight_data_pack1to4_g = weight_data_pack1to4_groups.channel_range(num_output_g/4 * g, num_output_g/4);

                for (int q=0; q+3<num_output_g; q+=4)
                {
                    const Mat k0 = weight_data_r2.channel(q);
                    const Mat k1 = weight_data_r2.channel(q+1);
                    const Mat k2 = weight_data_r2.channel(q+2);
                    const Mat k3 = weight_data_r2.channel(q+3);

                    Mat g0 = weight_data_pack1to4_g.channel(q/4);

                    for (int p=0; p<channels_g; p++)
                    {
                        const float* k00 = k0.row(p);
                        const float* k10 = k1.row(p);
                        const float* k20 = k2.row(p);
                        const float* k30 = k3.row(p);

                        float* g00 = g0.row(p);

                        for (int k=0; k<maxk; k++)
                        {
                            g00[0] = k00[k];
                            g00[1] = k10[k];
                            g00[2] = k20[k];
                            g00[3] = k30[k];

                            g00 += 4;
                        }
                    }
                }
            }
        }
    }

    // pack4to1
    if (channels_g % 4 == 0 && num_output_g % 4 != 0)
    {
        // src = kw-kh-inch-outch
        // dst = 4a-kw-kh-inch/4a-outch
        {
            Mat weight_data_r2_groups = weight_data.reshape(maxk, channels_g, num_output_g * group);

            weight_data_pack4to1_groups.create(maxk, channels_g/4, num_output_g * group, (size_t)4*4, 4);

            for (int g=0; g<group; g++)
            {
                const Mat weight_data_r2 = weight_data_r2_groups.channel_range(num_output_g * g, num_output_g);

                Mat weight_data_pack4to1_g = weight_data_pack4to1_groups.channel_range(num_output_g * g, num_output_g);

                for (int q=0; q<num_output_g; q++)
                {
                    const Mat k0 = weight_data_r2.channel(q);
                    Mat g0 = weight_data_pack4to1_g.channel(q);

                    for (int p=0; p+3<channels_g; p+=4)
                    {
                        const float* k00 = k0.row(p);
                        const float* k01 = k0.row(p+1);
                        const float* k02 = k0.row(p+2);
                        const float* k03 = k0.row(p+3);

                        float* g00 = g0.row(p/4);

                        for (int k=0; k<maxk; k++)
                        {
                            g00[0] = k00[k];
                            g00[1] = k01[k];
                            g00[2] = k02[k];
                            g00[3] = k03[k];

                            g00 += 4;
                        }
                    }
                }
            }
        }
    }

    } // opt.use_packing_layout

    for (int i=0; i<(int)group_ops.size(); i++)
        delete group_ops[i];

    group_ops.clear();

    if (channels == group && group == num_output)
    {
        // depth-wise specific
        if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1)
        {
            if ((stride_w == 1 && stride_h == 1) || (stride_w == 2 && stride_h == 2))
            {
                return 0;
            }
        }
        if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && use_int8_inference == false)
        {
            if ((stride_w == 1 && stride_h == 1) || (stride_w == 2 && stride_h == 2))
            {
                return 0;
            }
        }
    }

    const int channels_g = channels / group;
    const int num_output_g = num_output / group;

    group_ops.resize(group);

    for (int g=0; g<group; g++)
    {
        Mat weight_data_g = weight_data.range(maxk * channels_g * num_output_g * g, maxk * channels_g * num_output_g);
        Mat bias_data_g;
        if (bias_term)
            bias_data_g = bias_data.range(num_output_g * g, num_output_g);

        ncnn::Layer* op = ncnn::create_layer(ncnn::LayerType::Convolution);

        // set param
        ncnn::ParamDict pd;
        pd.set(0, num_output_g);// num_output
        pd.set(1, kernel_w);
        pd.set(11, kernel_h);
        pd.set(2, dilation_w);
        pd.set(12, dilation_h);
        pd.set(3, stride_w);
        pd.set(13, stride_h);
        pd.set(4, 0);// pad_w
        pd.set(14, 0);// pad_h
        pd.set(5, bias_term);
        pd.set(6, maxk * channels_g * num_output_g);// weight_data_size
        pd.set(8, int8_scale_term);

        op->load_param(pd);

        // set weights
        if (bias_term)
        {
            ncnn::Mat weights[4];
            weights[0] = weight_data_g;
            weights[1] = bias_data_g;

            if (int8_scale_term)
            {
                weights[2] = weight_data_int8_scales.range(g, 1);
                weights[3] = bottom_blob_int8_scales.range(g, 1);
            }

            op->load_model(ModelBinFromMatArray(weights));
        }
        else
        {
            ncnn::Mat weights[3];
            weights[0] = weight_data_g;

            if (int8_scale_term)
            {
                weights[1] = weight_data_int8_scales.range(g, 1);
                weights[2] = bottom_blob_int8_scales.range(g, 1);
            }

            op->load_model(ModelBinFromMatArray(weights));
        }

        op->create_pipeline(opt_cpu);

        group_ops[g] = op;
    }

    return 0;
}

int ConvolutionDepthWise_arm::destroy_pipeline(const Option& opt)
{
    Option opt_cpu = opt;
    opt_cpu.use_vulkan_compute = false;

    if (activation)
    {
        activation->destroy_pipeline(opt_cpu);
        delete activation;
        activation = 0;
    }

    for (int i=0; i<(int)group_ops.size(); i++)
    {
        group_ops[i]->destroy_pipeline(opt_cpu);
        delete group_ops[i];
    }
    group_ops.clear();

    return 0;
}

int ConvolutionDepthWise_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // convolv with NxN kernel
    // value = value + bias

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    Mat bottom_blob_unbordered = bottom_blob;
    if (use_int8_inference && elemsize != 1)
    {
        Mat bottom_blob_int8;
        bottom_blob_int8.create(w, h, channels, (size_t)1u, opt.workspace_allocator);
        if (bottom_blob_int8.empty())
            return -100;

        const int channels_g = channels / group;

        // quantize, scale and round to nearest
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g=0; g<group; g++)
        {
            ncnn::Option opt_g = opt;
            opt_g.num_threads = 1;
            opt_g.blob_allocator = bottom_blob_int8.allocator;

            const Mat bottom_blob_g = bottom_blob.channel_range(channels_g * g, channels_g);
            Mat bottom_blob_int8_g = bottom_blob_int8.channel_range(channels_g * g, channels_g);
            quantize_ops[g]->forward(bottom_blob_g, bottom_blob_int8_g, opt_g);
        }

        bottom_blob_unbordered = bottom_blob_int8;    
    }    

    Mat bottom_blob_bordered = bottom_blob_unbordered;
    if (pad_w > 0 || pad_h > 0)
    {
        Option opt_b = opt;
        opt_b.blob_allocator = opt.workspace_allocator;
        copy_make_border(bottom_blob_unbordered, bottom_blob_bordered, pad_h, pad_h, pad_w, pad_w, BORDER_CONSTANT, 0.f, opt_b);
        if (bottom_blob_bordered.empty())
            return -100;

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;
    }
    else if (pad_w == -233 && pad_h == -233)
    {
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border(bottom_blob_unbordered, bottom_blob_bordered, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, BORDER_CONSTANT, 0.f, opt_b);
            if (bottom_blob_bordered.empty())
                return -100;
        }

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;
    }

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;
    int out_elempack = num_output % 4 == 0 ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    if (opt.use_packing_layout)
    {

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // depth-wise
    if (channels == group / elempack && group / elempack == num_output / elempack)
    {
    if (elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g=0; g<group / elempack; g++)
        {
            float* outptr = top_blob.channel(g);
            const float* kptr = (const float*)weight_data_pack4 + maxk * g * 4;
            const Mat m = bottom_blob_bordered.channel(g);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    float32x4_t _sum = vdupq_n_f32(0.f);

                    if (bias_term)
                    {
                        _sum = vld1q_f32(((const float*)bias_data) + g * 4);
                    }

                    const float* sptr = m.row(i*stride_h) + j*stride_w * 4;

                    for (int k = 0; k < maxk; k++)
                    {
                        float32x4_t _val = vld1q_f32( sptr + space_ofs[k] * 4 );
                        float32x4_t _w = vld1q_f32( kptr + k * 4 );
                        _sum = vmlaq_f32(_sum, _val, _w);
                    }

                    if (activation_type == 1)
                    {
                        float32x4_t _zero = vdupq_n_f32(0.f);
                        _sum = vmaxq_f32(_sum, _zero);
                    }
                    else if (activation_type == 2)
                    {
                        float32x4_t _zero = vdupq_n_f32(0.f);
                        float32x4_t _slope = vdupq_n_f32(activation_params[0]);
                        uint32x4_t _lemask = vcleq_f32(_sum, _zero);
                        float32x4_t _ps = vmulq_f32(_sum, _slope);
                        _sum = vbslq_f32(_lemask, _ps, _sum);
                    }
                    else if (activation_type == 3)
                    {
                        float32x4_t _min = vdupq_n_f32(activation_params[0]);
                        float32x4_t _max = vdupq_n_f32(activation_params[1]);
                        _sum = vmaxq_f32(_sum, _min);
                        _sum = vminq_f32(_sum, _max);
                    }
                    else if (activation_type == 4)
                    {
                        float32x4_t _one = vdupq_n_f32(1.f);
                        _sum = vnegq_f32(_sum);
                        _sum = exp_ps(_sum);
                        _sum = vaddq_f32(_sum, _one);
                        float32x4_t _outp = vrecpeq_f32(_sum);
                        _outp = vmulq_f32(vrecpsq_f32(_sum, _outp), _outp);
//                         _outp = vmulq_f32(vrecpsq_f32(_sum, _outp), _outp);
                        _sum = _outp;
                    }

                    vst1q_f32(outptr + j * 4, _sum);
                }

                outptr += outw * 4;
            }
        }

        return 0;
    }
    }

    const int channels_g = channels * elempack / group;
    const int num_output_g = num_output / group;

    // unpacking
    Mat bottom_blob_bordered_unpacked = bottom_blob_bordered;
    if (elempack == 4 && channels_g % 4 != 0)
    {
        Option opt_p = opt;
        opt_p.blob_allocator = opt.workspace_allocator;
        convert_packing(bottom_blob_bordered, bottom_blob_bordered_unpacked, 1, opt_p);
    }

    Mat top_blob_unpacked = top_blob;
    if (num_output_g % 4 != 0 && out_elempack == 4)
    {
        top_blob_unpacked.create(outw, outh, num_output, out_elemsize / out_elempack, 1, opt.workspace_allocator);
        if (top_blob_unpacked.empty())
            return -100;
    }

    if (channels_g % 4 == 0 && num_output_g % 4 == 0)
    {
#ifdef _WIN32
        #pragma omp parallel for num_threads(opt.num_threads)
#else // _WIN32
        #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
#endif // _WIN32
        for (int g=0; g<group; g++)
        {
            for (int p=0; p<num_output_g / 4; p++)
            {
                float* outptr = top_blob_unpacked.channel(g * num_output_g / 4 + p);
                const float* weight_data_ptr = (const float*)weight_data_pack4_groups + maxk * channels_g / 4 * num_output_g / 4 * g * 16;

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float32x4_t _sum = vdupq_n_f32(0.f);

                        if (bias_term)
                        {
                            _sum = vld1q_f32(((const float*)bias_data) + num_output_g * g + p * 4);
                        }

                        const float* kptr = weight_data_ptr + maxk * channels_g / 4 * p * 16;

                        // channels_g
                        for (int q=0; q<channels_g / 4; q++)
                        {
                            const Mat m = bottom_blob_bordered.channel(channels_g / 4 * g + q);
                            const float* sptr = m.row(i*stride_h) + j*stride_w * 4;

                            for (int k = 0; k < maxk; k++)
                            {
                                float32x4_t _val = vld1q_f32( sptr + space_ofs[k] * 4 );

                                float32x4_t _w0 = vld1q_f32( kptr );
                                float32x4_t _w1 = vld1q_f32( kptr + 4 );
                                float32x4_t _w2 = vld1q_f32( kptr + 8 );
                                float32x4_t _w3 = vld1q_f32( kptr + 12 );

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

                                kptr += 16;
                            }
                        }

                        if (activation_type == 1)
                        {
                            float32x4_t _zero = vdupq_n_f32(0.f);
                            _sum = vmaxq_f32(_sum, _zero);
                        }
                        else if (activation_type == 2)
                        {
                            float32x4_t _zero = vdupq_n_f32(0.f);
                            float32x4_t _slope = vdupq_n_f32(activation_params[0]);
                            uint32x4_t _lemask = vcleq_f32(_sum, _zero);
                            float32x4_t _ps = vmulq_f32(_sum, _slope);
                            _sum = vbslq_f32(_lemask, _ps, _sum);
                        }
                        else if (activation_type == 3)
                        {
                            float32x4_t _min = vdupq_n_f32(activation_params[0]);
                            float32x4_t _max = vdupq_n_f32(activation_params[1]);
                            _sum = vmaxq_f32(_sum, _min);
                            _sum = vminq_f32(_sum, _max);
                        }
                        else if (activation_type == 4)
                        {
                            float32x4_t _one = vdupq_n_f32(1.f);
                            _sum = vnegq_f32(_sum);
                            _sum = exp_ps(_sum);
                            _sum = vaddq_f32(_sum, _one);
                            float32x4_t _outp = vrecpeq_f32(_sum);
                            _outp = vmulq_f32(vrecpsq_f32(_sum, _outp), _outp);
//                             _outp = vmulq_f32(vrecpsq_f32(_sum, _outp), _outp);
                            _sum = _outp;
                        }

                        vst1q_f32(outptr + j * 4, _sum);
                    }

                    outptr += outw * 4;
                }
            }
        }
    }

    if (channels_g % 4 != 0 && num_output_g % 4 == 0)
    {
#ifdef _WIN32
        #pragma omp parallel for num_threads(opt.num_threads)
#else // _WIN32
        #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
#endif // _WIN32
        for (int g=0; g<group; g++)
        {
            for (int p=0; p<num_output_g / 4; p++)
            {
                float* outptr = top_blob_unpacked.channel(g * num_output_g / 4 + p);
                const float* weight_data_ptr = (const float*)weight_data_pack1to4_groups + maxk * channels_g * num_output_g / 4 * g * 4;

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float32x4_t _sum = vdupq_n_f32(0.f);

                        if (bias_term)
                        {
                            _sum = vld1q_f32(((const float*)bias_data) + (num_output_g / 4 * g + p) * 4);
                        }

                        const float* kptr = weight_data_ptr + maxk * channels_g * p * 4;

                        // channels_g
                        for (int q=0; q<channels_g; q++)
                        {
                            const Mat m = bottom_blob_bordered.channel(channels_g * g + q);
                            const float* sptr = m.row(i*stride_h) + j*stride_w;

                            for (int k = 0; k < maxk; k++)
                            {
                                float32x4_t _val = vdupq_n_f32( sptr[ space_ofs[k] ] );
                                float32x4_t _w = vld1q_f32( kptr );
                                _sum = vmlaq_f32(_sum, _val, _w);

                                kptr += 4;
                            }
                        }

                        if (activation_type == 1)
                        {
                            float32x4_t _zero = vdupq_n_f32(0.f);
                            _sum = vmaxq_f32(_sum, _zero);
                        }
                        else if (activation_type == 2)
                        {
                            float32x4_t _zero = vdupq_n_f32(0.f);
                            float32x4_t _slope = vdupq_n_f32(activation_params[0]);
                            uint32x4_t _lemask = vcleq_f32(_sum, _zero);
                            float32x4_t _ps = vmulq_f32(_sum, _slope);
                            _sum = vbslq_f32(_lemask, _ps, _sum);
                        }
                        else if (activation_type == 3)
                        {
                            float32x4_t _min = vdupq_n_f32(activation_params[0]);
                            float32x4_t _max = vdupq_n_f32(activation_params[1]);
                            _sum = vmaxq_f32(_sum, _min);
                            _sum = vminq_f32(_sum, _max);
                        }
                        else if (activation_type == 4)
                        {
                            float32x4_t _one = vdupq_n_f32(1.f);
                            _sum = vnegq_f32(_sum);
                            _sum = exp_ps(_sum);
                            _sum = vaddq_f32(_sum, _one);
                            float32x4_t _outp = vrecpeq_f32(_sum);
                            _outp = vmulq_f32(vrecpsq_f32(_sum, _outp), _outp);
//                             _outp = vmulq_f32(vrecpsq_f32(_sum, _outp), _outp);
                            _sum = _outp;
                        }

                        vst1q_f32(outptr + j * 4, _sum);
                    }

                    outptr += outw * 4;
                }
            }
        }
    }

    if (channels_g % 4 == 0 && num_output_g % 4 != 0)
    {
#ifdef _WIN32
        #pragma omp parallel for num_threads(opt.num_threads)
#else // _WIN32
        #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
#endif // _WIN32
        for (int g=0; g<group; g++)
        {
            for (int p=0; p<num_output_g; p++)
            {
                float* outptr = top_blob_unpacked.channel(g * num_output_g + p);
                const float* weight_data_ptr = (const float*)weight_data_pack4to1_groups + maxk * channels_g / 4 * num_output_g * g * 4;

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float sum = 0.f;

                        if (bias_term)
                            sum = bias_data[num_output_g * g + p];

                        const float* kptr = weight_data_ptr + maxk * channels_g / 4 * p * 4;

                        // channels_g
                        for (int q=0; q<channels_g / 4; q++)
                        {
                            const Mat m = bottom_blob_bordered.channel(channels_g / 4 * g + q);
                            const float* sptr = m.row(i*stride_h) + j*stride_w * 4;

                            for (int k = 0; k < maxk; k++)
                            {
                                float32x4_t _val = vld1q_f32( sptr + space_ofs[k] * 4 );
                                float32x4_t _w = vld1q_f32( kptr );
                                float32x4_t _s4 = vmulq_f32(_val, _w);
#if __aarch64__
                                sum += vaddvq_f32(_s4); // dot
#else
                                float32x2_t _ss = vadd_f32(vget_low_f32(_s4), vget_high_f32(_s4));
                                _ss = vpadd_f32(_ss, _ss);
                                sum += vget_lane_f32(_ss, 0);
#endif

                                kptr += 4;
                            }
                        }

                        if (activation_type == 1)
                        {
                            sum = std::max(sum, 0.f);
                        }
                        else if (activation_type == 2)
                        {
                            float slope = activation_params[0];
                            sum = sum > 0.f ? sum : sum * slope;
                        }
                        else if (activation_type == 3)
                        {
                            float min = activation_params[0];
                            float max = activation_params[1];
                            if (sum < min)
                                sum = min;
                            if (sum > max)
                                sum = max;
                        }
                        else if (activation_type == 4)
                        {
                            sum = 1.f / (1.f + exp(-sum));
                        }

                        outptr[j] = sum;
                    }

                    outptr += outw;
                }
            }
        }
    }

    // packing
    if (num_output_g % 4 != 0 && out_elempack == 4)
    {
        convert_packing(top_blob_unpacked, top_blob, 4, opt);
    }
    else
    {
        top_blob = top_blob_unpacked;
    }

    return 0;

    } // opt.use_packing_layout

    // int8 
    if (use_int8_inference)
    {
        if (use_int8_requantize)
        {
            Mat top_blob_tm;
            top_blob_tm.create(outw, outh, num_output, (size_t)4u, opt.workspace_allocator);
            if (top_blob_tm.empty())
                return -100;
  
            top_blob.create(outw, outh, num_output, (size_t)1u, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            // depth-wise
            if (channels == group && group == num_output)
            {                
                if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1)
                {
                    if ((stride_w == 1 && stride_h == 1) || (stride_w == 2 && stride_h == 2))
                    {                  
                        if (stride_w == 1 && stride_h == 1)
                        {
                            convdw3x3s1_int8_requant_neon(bottom_blob_bordered, top_blob, weight_data, bias_data, requantize_scales, opt);
                        }
                        else if (stride_w == 2 && stride_h == 2)
                        {
                            convdw3x3s2_int8_requant_neon(bottom_blob_bordered, top_blob, weight_data, bias_data, requantize_scales, opt);
                        }                   

                        if (activation)
                        {
                            activation->forward_inplace(top_blob, opt);
                        }  

                        return 0;
                    }
                }

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int g=0; g<group; g++)
                {
                    const Mat bottom_blob_bordered_g = bottom_blob_bordered.channel_range(g, 1);
                    Mat top_blob_tm_g = top_blob_tm.channel_range(g, 1);

                    const ncnn::Layer* op = group_ops[g];

                    ncnn::Option opt_g = opt;
                    opt_g.num_threads = 1;
                    opt_g.blob_allocator = top_blob.allocator;

                    // forward
                    op->forward(bottom_blob_bordered_g, top_blob_tm_g, opt_g);
                }

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }  

                return 0;
            }

            const int channels_g = channels / group;
            const int num_output_g = num_output / group;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int g=0; g<group; g++)
            {
                const Mat bottom_blob_bordered_g = bottom_blob_bordered.channel_range(channels_g * g, channels_g);
                Mat top_blob_tm_g = top_blob_tm.channel_range(num_output_g * g, num_output_g);

                const ncnn::Layer* op = group_ops[g];

                ncnn::Option opt_g = opt;
                opt_g.blob_allocator = top_blob.allocator;

                // forward
                op->forward(bottom_blob_bordered_g, top_blob_tm_g, opt_g);
            }     
        }
        else
        {
            top_blob.create(outw, outh, num_output, (size_t)4u, opt.blob_allocator);
            if (top_blob.empty())
                return -100;               

            // depth-wise
            if (channels == group && group == num_output)
            {                
                if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1)
                {
                    if ((stride_w == 1 && stride_h == 1) || (stride_w == 2 && stride_h == 2))
                    {
                        if (stride_w == 1 && stride_h == 1)
                        {
                            convdw3x3s1_int8_neon(bottom_blob_bordered, top_blob, weight_data, opt);
                        }
                        else if (stride_w == 2 && stride_h == 2)
                        {
                            convdw3x3s2_int8_neon(bottom_blob_bordered, top_blob, weight_data, opt);
                        }

                        // dequantize, reverse scale inplace
                        #pragma omp parallel for num_threads(opt.num_threads)
                        for (int g=0; g<group; g++)
                        {
                            ncnn::Option opt_g = opt;
                            opt_g.num_threads = 1;
                            opt_g.blob_allocator = top_blob.allocator;

                            Mat top_blob_g = top_blob.channel(g);
                            dequantize_ops[g]->forward_inplace(top_blob_g, opt_g);
                        }

                        if (activation)
                        {
                            activation->forward_inplace(top_blob, opt);
                        }  

                        return 0;
                    }
                }

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int g=0; g<group; g++)
                {
                    const Mat bottom_blob_bordered_g = bottom_blob_bordered.channel_range(g, 1);
                    Mat top_blob_g = top_blob.channel_range(g, 1);

                    const ncnn::Layer* op = group_ops[g];

                    ncnn::Option opt_g = opt;
                    opt_g.num_threads = 1;
                    opt_g.blob_allocator = top_blob.allocator;

                    // forward
                    op->forward(bottom_blob_bordered_g, top_blob_g, opt_g);
                }

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }  

                return 0;
            }

            const int channels_g = channels / group;
            const int num_output_g = num_output / group;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int g=0; g<group; g++)
            {
                const Mat bottom_blob_bordered_g = bottom_blob_bordered.channel_range(channels_g * g, channels_g);
                Mat top_blob_g = top_blob.channel_range(num_output_g * g, num_output_g);

                const ncnn::Layer* op = group_ops[g];

                ncnn::Option opt_g = opt;
                opt_g.blob_allocator = top_blob.allocator;

                // forward
                op->forward(bottom_blob_bordered_g, top_blob_g, opt_g);
            }                    
        }

        if (activation)
        {
            activation->forward_inplace(top_blob, opt);
        }  

        return 0;
    }

    // float32
    top_blob.create(outw, outh, num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;
    
    // depth-wise
    if (channels == group && group == num_output)
    {
        if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1)
        {
            if (stride_w == 1 && stride_h == 1)
            {
                convdw3x3s1_neon(bottom_blob_bordered, top_blob, weight_data, bias_data, opt);
            }
            else if (stride_w == 2 && stride_h == 2)
            {
                convdw3x3s2_neon(bottom_blob_bordered, top_blob, weight_data, bias_data, opt);
            }

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }

            return 0;
        }
        if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1)
        {
            if (stride_w == 1 && stride_h == 1)
            {
                convdw5x5s1_neon(bottom_blob_bordered, top_blob, weight_data, bias_data, opt);
            }
            else if (stride_w == 2 && stride_h == 2)
            {
                convdw5x5s2_neon(bottom_blob_bordered, top_blob, weight_data, bias_data, opt);
            }

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }

            return 0;
        }        

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g=0; g<group; g++)
        {
            const Mat bottom_blob_bordered_g = bottom_blob_bordered.channel_range(g, 1);
            Mat top_blob_g = top_blob.channel_range(g, 1);

            const ncnn::Layer* op = group_ops[g];

            ncnn::Option opt_g = opt;
            opt_g.num_threads = 1;
            opt_g.blob_allocator = top_blob.allocator;

            // forward
            op->forward(bottom_blob_bordered_g, top_blob_g, opt_g);
        }            

        if (activation)
        {
            activation->forward_inplace(top_blob, opt);
        }

        return 0;
    }

    const int channels_g = channels / group;
    const int num_output_g = num_output / group;

    for (int g=0; g<group; g++)
    {
        const Mat bottom_blob_bordered_g = bottom_blob_bordered.channel_range(channels_g * g, channels_g);
        Mat top_blob_g = top_blob.channel_range(num_output_g * g, num_output_g);

        const ncnn::Layer* op = group_ops[g];

        ncnn::Option opt_g = opt;
        opt_g.blob_allocator = top_blob.allocator;

        // forward
        op->forward(bottom_blob_bordered_g, top_blob_g, opt_g);
    }

    if (activation)
    {
        activation->forward_inplace(top_blob, opt);
    }

    return 0;
}

} // namespace ncnn
