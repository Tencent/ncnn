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

#include "convolution_arm.h"

#include "benchmark.h"
#include "cpu.h"
#include "layer_type.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "arm_activation.h"
#include "arm_usability.h"

namespace ncnn {

#if NCNN_GNU_INLINE_ASM
#include "convolution_1x1.h"
#include "convolution_2x2.h"
#include "convolution_3x3.h"
#include "convolution_4x4.h"
#include "convolution_5x5.h"
#include "convolution_7x7.h"
#endif // NCNN_GNU_INLINE_ASM

#include "convolution_packed.h"
#include "convolution_3x3_winograd.h"
#include "convolution_im2col_gemm.h"

#if NCNN_BF16
#include "convolution_packed_bf16s.h"
#include "convolution_3x3_winograd_bf16s.h"
#include "convolution_im2col_gemm_bf16s_fp16s.h"
#include "convolution_im2col_gemm_bf16s.h"
#endif // NCNN_BF16

#if NCNN_INT8
#include "convolution_packed_int8.h"
#include "convolution_im2col_gemm_int8.h"
#include "convolution_3x3_winograd_int8.h"

// #include "convolution_3x3_int8.h"
#endif // NCNN_INT8

#if __ARM_NEON
#if NCNN_GNU_INLINE_ASM
#include "convolution_3x3_pack1to4.h"
#include "convolution_3x3_pack4.h"
#include "convolution_3x3_pack4to1.h"
#include "convolution_5x5_pack4.h"
#include "convolution_7x7_pack1to4.h"

#if NCNN_BF16
#include "convolution_3x3_pack1to4_bf16s.h"
#include "convolution_3x3_pack4_bf16s.h"
#include "convolution_5x5_pack4_bf16s.h"
#include "convolution_7x7_pack1to4_bf16s.h"
#endif // NCNN_BF16
#endif // NCNN_GNU_INLINE_ASM
#endif // __ARM_NEON

Convolution_arm::Convolution_arm()
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
    nT = 0;
    convolution_dilation1 = 0;
}

static void convolution_transform_kernel_packed_neon(const Mat& weight_data, Mat& weight_data_tm, int num_input, int num_output, int kernel_w, int kernel_h, int elempack, int out_elempack)
{
    const int maxk = kernel_w * kernel_h;

    // src = kw-kh-inch-outch
    // dst = pb-pa-kw-kh-inch/pa-outch/pb
    {
        Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

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
    }
}

int Convolution_arm::create_pipeline(const Option& opt)
{
    if (dynamic_weight)
        return 0;

    activation = create_activation_layer(activation_type, activation_params, opt);
    nT = opt.num_threads;

#if NCNN_INT8
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        return create_pipeline_int8_arm(opt);
    }
#endif

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

    if ((!support_packing || !opt.use_packing_layout) && !opt.use_bf16_storage && kernel_w == kernel_h && dilation_w != 1 && dilation_h == dilation_w && stride_w == 1 && stride_h == 1)
    {
        convolution_dilation1 = ncnn::create_layer_cpu(ncnn::LayerType::Convolution);

        // set param
        ncnn::ParamDict pd;
        pd.set(0, num_output); // num_output
        pd.set(1, kernel_w);
        pd.set(11, kernel_h);
        pd.set(2, 1);
        pd.set(12, 1);
        pd.set(3, 1);  // stride_w
        pd.set(13, 1); // stride_h
        pd.set(4, 0);  // pad_w
        pd.set(14, 0); // pad_h
        pd.set(5, bias_term);
        pd.set(6, weight_data_size);

        convolution_dilation1->load_param(pd);

        // set weights
        if (bias_term)
        {
            ncnn::Mat weights[2];
            weights[0] = weight_data;
            weights[1] = bias_data;

            convolution_dilation1->load_model(ModelBinFromMatArray(weights));
        }
        else
        {
            ncnn::Mat weights[1];
            weights[0] = weight_data;

            convolution_dilation1->load_model(ModelBinFromMatArray(weights));
        }

        convolution_dilation1->create_pipeline(opt);

        weight_data.release();

        return 0;
    }

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

    bool prefer_winograd = (opt.use_winograd23_convolution || opt.use_winograd43_convolution || opt.use_winograd63_convolution) && (num_input >= 8 || num_output >= 8);

    if (opt.use_winograd_convolution && prefer_winograd && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        // dynamic shape
        if (opt.use_winograd63_convolution && (num_input <= 128 && num_output <= 128))
            conv3x3s1_winograd63_transform_kernel(weight_data, weight_winograd63_data, num_input, num_output, opt);
        else if (opt.use_winograd43_convolution && (num_input >= 8 && num_output >= 8))
            conv3x3s1_winograd43_transform_kernel(weight_data, weight_winograd43_data, num_input, num_output, opt);
        else
            conv3x3s1_winograd23_transform_kernel(weight_data, weight_winograd23_data, num_input, num_output, opt);

        weight_data.release();

        return 0;
    }

    int l2_cache_size_fp32 = get_cpu_level2_cache_size() / sizeof(float);
    bool prefer_sgemm = num_input * num_output * kernel_w * kernel_h * dilation_w * dilation_h * stride_w * stride_h * 2 > l2_cache_size_fp32 || (num_input > 16 || num_output > 16);

#if NCNN_GNU_INLINE_ASM
    if (elempack == 4 && out_elempack == 4)
    {
        if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2 && (num_input < 4 || num_output < 32))
        {
            prefer_sgemm = false;
        }
        if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            prefer_sgemm = false;
        }
        if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2 && (num_input < 8 || num_output < 44))
        {
            prefer_sgemm = false;
        }
    }

    if (elempack == 1 && out_elempack == 4)
    {
        if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            prefer_sgemm = false;
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            prefer_sgemm = false;
        }
        else if (kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            prefer_sgemm = false;
        }
    }
#endif // NCNN_GNU_INLINE_ASM

    if ((opt.use_sgemm_convolution && prefer_sgemm) || (kernel_w == 1 && kernel_h == 1))
    {
        convolution_im2col_gemm_transform_kernel(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h, opt);

        weight_data.release();

        return 0;
    }

#if NCNN_GNU_INLINE_ASM
    if ((elempack == 4 && out_elempack == 4 && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            || (elempack == 4 && out_elempack == 4 && kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            || (elempack == 4 && out_elempack == 4 && kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            || (elempack == 1 && out_elempack == 4 && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            || (elempack == 1 && out_elempack == 4 && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            || (elempack == 1 && out_elempack == 4 && kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2))
    {
        convolution_transform_kernel_packed_neon(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
    }
    else if (elempack == 1 && out_elempack == 1 && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
    {
        conv3x3s2_transform_kernel_neon(weight_data, weight_3x3s2_data, num_input, num_output);
    }
    else if ((elempack == 1 && out_elempack == 1 && kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
             || (elempack == 1 && out_elempack == 1 && kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
             || (elempack == 1 && out_elempack == 1 && kernel_w == 4 && kernel_h == 4 && dilation_w == 1 && dilation_h == 1 && stride_w == 4 && stride_h == 4)
             || (elempack == 1 && out_elempack == 1 && kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
             || (elempack == 1 && out_elempack == 1 && kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
             || (elempack == 1 && out_elempack == 1 && kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
             || (elempack == 1 && out_elempack == 1 && kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2))
    {
        weight_data_tm = weight_data;
    }
    else
#endif // NCNN_GNU_INLINE_ASM
    {
        convolution_transform_kernel_packed(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h);
    }

    weight_data.release();

    return 0;
}

int Convolution_arm::destroy_pipeline(const Option& opt)
{
    if (activation)
    {
        activation->destroy_pipeline(opt);
        delete activation;
        activation = 0;
    }

    if (convolution_dilation1)
    {
        convolution_dilation1->destroy_pipeline(opt);
        delete convolution_dilation1;
        convolution_dilation1 = 0;
    }

    return 0;
}

int Convolution_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if NCNN_INT8
    if (opt.use_int8_inference && int8_scale_term)
    {
        return forward_int8_arm(bottom_blob, top_blob, opt);
    }
#endif

    // flattened blob, implement as InnerProduct
    if (bottom_blob.dims == 1 && kernel_w == 1 && kernel_h == 1)
    {
        Mat bottom_blob_3d;
        if (bottom_blob.elemsize % 16 == 0)
        {
            bottom_blob_3d = bottom_blob;
            bottom_blob_3d.dims = 3;
            bottom_blob_3d.w = 1;
            bottom_blob_3d.h = 1;
            bottom_blob_3d.c = bottom_blob.w;
            bottom_blob_3d.cstep = 1;
        }
        else
        {
            bottom_blob_3d = bottom_blob.reshape(1, 1, bottom_blob.w, opt.workspace_allocator);
        }

        Mat top_blob_3d;
        int ret = forward(bottom_blob_3d, top_blob_3d, opt);
        if (ret != 0)
            return ret;

        if (top_blob_3d.elemsize % 16 == 0)
        {
            top_blob = top_blob_3d;
            top_blob.dims = 1;
            top_blob.w = top_blob_3d.c;
            top_blob.h = 1;
            top_blob.c = 1;
            bottom_blob_3d.cstep = top_blob_3d.c;
        }
        else
        {
            top_blob = top_blob_3d.reshape(top_blob_3d.c, opt.blob_allocator);
        }

        return 0;
    }

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

    //     NCNN_LOGE("Convolution input %d x %d  pad = %d %d  ksize=%d %d  stride=%d %d", w, h, pad_w, pad_h, kernel_w, kernel_h, stride_w, stride_h);

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;
    int out_elempack = 1;
#if __ARM_NEON
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % 4 == 0 ? 4 : 1;
    }
#endif
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if ((!support_packing || !opt.use_packing_layout) && kernel_w == kernel_h && dilation_w != 1 && dilation_h == dilation_w && stride_w == 1 && stride_h == 1)
    {
        if (outw >= dilation_w && outh >= dilation_h)
        {
            return forwardDilation_arm(bottom_blob_bordered, top_blob, opt);
        }
    }

    const int num_input = channels * elempack;

    bool prefer_winograd = (opt.use_winograd23_convolution || opt.use_winograd43_convolution || opt.use_winograd63_convolution) && (num_input >= 8 || num_output >= 8);

    if (opt.use_winograd_convolution && prefer_winograd && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        bool prefer_winograd63 = false;
        bool prefer_winograd23 = false;
        bool prefer_winograd43 = !prefer_winograd63 && !prefer_winograd23;

        if (prefer_winograd23 && (!opt.use_winograd23_convolution || weight_winograd23_data.empty()))
        {
            // f23 fallback to f43
            prefer_winograd23 = false;
            prefer_winograd43 = true;
        }

        if (prefer_winograd63 && (!opt.use_winograd63_convolution || weight_winograd63_data.empty()))
        {
            // f63 fallback to f43
            prefer_winograd63 = false;
            prefer_winograd43 = true;
        }

        if (prefer_winograd43 && (!opt.use_winograd43_convolution || weight_winograd43_data.empty()))
        {
            // f43 fallback to f63 or f23
            prefer_winograd43 = false;
            if (opt.use_winograd63_convolution && !weight_winograd63_data.empty())
            {
                prefer_winograd63 = true;
            }
            else
            {
                prefer_winograd23 = true;
            }
        }
        // NCNN_LOGE("prefer_winograd %d %d %d", prefer_winograd23, prefer_winograd43, prefer_winograd63);

        int _nT = nT ? nT : opt.num_threads;
        if (nT != 0 && opt.num_threads != nT)
        {
            // force num_threads the same as in create_pipeline
            // so we could use pre-packed A/B from the same tile config
            NCNN_LOGE("opt.num_threads %d changed, convolution winograd will use load-time value %d", opt.num_threads, nT);
        }

        if (prefer_winograd23)
        {
            conv3x3s1_winograd23(bottom_blob_bordered, top_blob, weight_winograd23_data, bias_data, _nT, opt);
        }
        else if (prefer_winograd43)
        {
            conv3x3s1_winograd43(bottom_blob_bordered, top_blob, weight_winograd43_data, bias_data, _nT, opt);
        }
        else if (prefer_winograd63)
        {
            conv3x3s1_winograd63(bottom_blob_bordered, top_blob, weight_winograd63_data, bias_data, _nT, opt);
        }
        else
        {
            // should never reach here
        }

        if (activation)
        {
            activation->forward_inplace(top_blob, opt);
        }
        return 0;
    }

    int l2_cache_size_fp32 = get_cpu_level2_cache_size() / sizeof(float);
    bool prefer_sgemm = num_input * num_output * kernel_w * kernel_h * dilation_w * dilation_h * stride_w * stride_h * 2 > l2_cache_size_fp32 || (num_input > 16 || num_output > 16);

#if NCNN_GNU_INLINE_ASM
    if (elempack == 4 && out_elempack == 4)
    {
        if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2 && (num_input < 4 || num_output < 32))
        {
            prefer_sgemm = false;
        }
        if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            prefer_sgemm = false;
        }
        if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2 && (num_input < 8 || num_output < 44))
        {
            prefer_sgemm = false;
        }
    }

    if (elempack == 1 && out_elempack == 4)
    {
        if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            prefer_sgemm = false;
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            prefer_sgemm = false;
        }
        else if (kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            prefer_sgemm = false;
        }
    }
#endif // NCNN_GNU_INLINE_ASM

    if ((opt.use_sgemm_convolution && prefer_sgemm) || (kernel_w == 1 && kernel_h == 1))
    {
        int _nT = nT ? nT : opt.num_threads;
        if (nT != 0 && opt.num_threads != nT)
        {
            // force num_threads the same as in create_pipeline
            // so we could use pre-packed A/B from the same tile config
            NCNN_LOGE("opt.num_threads %d changed, convolution gemm will use load-time value %d", opt.num_threads, nT);
        }

        convolution_im2col_gemm(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, _nT, opt);

        if (activation)
        {
            activation->forward_inplace(top_blob, opt);
        }
        return 0;
    }

#if NCNN_GNU_INLINE_ASM
#if __ARM_NEON
    if (elempack == 4 && out_elempack == 4)
    {
        if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv3x3s2_pack4_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv5x5s1_pack4_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv5x5s2_pack4_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            convolution_packed(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }
    }

    if (elempack == 1 && out_elempack == 4)
    {
        if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv3x3s1_pack1to4_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv3x3s2_pack1to4_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv7x7s2_pack1to4_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            convolution_packed(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }
    }

    if (elempack == 4 && out_elempack == 1)
    {
        {
            convolution_packed(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }
    }
#endif // __ARM_NEON

    if (elempack == 1 && out_elempack == 1)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv1x1s2_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv3x3s2_packed_neon(bottom_blob_bordered, top_blob, weight_3x3s2_data, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 4 && kernel_h == 4 && dilation_w == 1 && dilation_h == 1 && stride_w == 4 && stride_h == 4)
        {
            conv4x4s4_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv5x5s1_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv5x5s2_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv7x7s1_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv7x7s2_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            convolution_packed(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }
    }
#else  // NCNN_GNU_INLINE_ASM
    {
        convolution_packed(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
    }
#endif // NCNN_GNU_INLINE_ASM

    return 0;
}

int Convolution_arm::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& _weight_data = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];

    const int _kernel_w = _weight_data.w;
    const int _kernel_h = _weight_data.h;
    const int _num_output = _weight_data.c * _weight_data.elempack;

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

    ncnn::Layer* op = ncnn::create_layer_cpu(ncnn::LayerType::Convolution);

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
    pd.set(18, pad_value);
    pd.set(5, bias_term);
    pd.set(6, weight_data_flattened.w);
    pd.set(8, int8_scale_term);
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    op->load_param(pd);

    ncnn::Mat weights[2];
    weights[0] = weight_data_flattened;
    weights[1] = bias_data_flattened;

    op->load_model(ncnn::ModelBinFromMatArray(weights));

    op->create_pipeline(opt);

    op->forward(bottom_blob, top_blob, opt);

    op->destroy_pipeline(opt);

    delete op;

    return 0;
}

#if NCNN_BF16
static void convolution_transform_kernel_packed_bf16s_neon(const Mat& weight_data, Mat& weight_data_tm, int num_input, int num_output, int kernel_w, int kernel_h, int elempack, int out_elempack)
{
    const int maxk = kernel_w * kernel_h;

    // src = kw-kh-inch-outch
    // dst = pb-pa-kw-kh-inch/pa-outch/pb
    {
        Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

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
}

int Convolution_arm::create_pipeline_bf16s(const Option& opt)
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

    bool prefer_winograd = (opt.use_winograd23_convolution || opt.use_winograd43_convolution || opt.use_winograd63_convolution) && (num_input >= 8 || num_output >= 8);

    if (opt.use_winograd_convolution && prefer_winograd && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        // dynamic shape
        if (opt.use_winograd63_convolution && (num_input <= 128 && num_output <= 128))
            conv3x3s1_winograd63_transform_kernel(weight_data, weight_winograd63_data, num_input, num_output, opt);
        else if (opt.use_winograd43_convolution && (num_input >= 8 && num_output >= 8))
            conv3x3s1_winograd43_transform_kernel(weight_data, weight_winograd43_data, num_input, num_output, opt);
        else
            conv3x3s1_winograd23_transform_kernel(weight_data, weight_winograd23_data, num_input, num_output, opt);

        weight_data.release();

        return 0;
    }

    int l2_cache_size_bf16 = get_cpu_level2_cache_size() / sizeof(unsigned short);
    bool prefer_sgemm = num_input * num_output * kernel_w * kernel_h * dilation_w * dilation_h * stride_w * stride_h * 2 > l2_cache_size_bf16 || (num_input > 16 || num_output > 16);

#if NCNN_GNU_INLINE_ASM
    if (elempack == 4 && out_elempack == 4)
    {
        if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2 && (num_input < 4 || num_output < 32))
        {
            prefer_sgemm = false;
        }
        if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            prefer_sgemm = false;
        }
        if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2 && (num_input < 8 || num_output < 44))
        {
            prefer_sgemm = false;
        }
    }

    if (elempack == 1 && out_elempack == 4)
    {
        if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            prefer_sgemm = false;
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            prefer_sgemm = false;
        }
        else if (kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            prefer_sgemm = false;
        }
    }
#endif // NCNN_GNU_INLINE_ASM

    if ((opt.use_sgemm_convolution && prefer_sgemm) || (kernel_w == 1 && kernel_h == 1))
    {
        convolution_im2col_gemm_transform_kernel_bf16s(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h, opt);

        weight_data.release();

        return 0;
    }

#if NCNN_GNU_INLINE_ASM
    if ((elempack == 4 && out_elempack == 4 && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            || (elempack == 4 && out_elempack == 4 && kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            || (elempack == 4 && out_elempack == 4 && kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            || (elempack == 1 && out_elempack == 4 && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            || (elempack == 1 && out_elempack == 4 && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            || (elempack == 1 && out_elempack == 4 && kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2))
    {
        convolution_transform_kernel_packed_bf16s_neon(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
    }
    else
#endif // NCNN_GNU_INLINE_ASM
    {
        convolution_transform_kernel_packed_bf16s(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h);
    }

    weight_data.release();

    return 0;
}

int Convolution_arm::forward_bf16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    //     NCNN_LOGE("Convolution input %d x %d  pad = %d %d  ksize=%d %d  stride=%d %d", w, h, pad_w, pad_h, kernel_w, kernel_h, stride_w, stride_h);

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;
    int out_elempack = 1;
#if __ARM_NEON
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % 4 == 0 ? 4 : 1;
    }
#endif
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // TODO dilated conv for bf16s
    //     if ((!support_packing || !opt.use_packing_layout) && kernel_w == kernel_h && dilation_w != 1 && dilation_h == dilation_w && stride_w == 1 && stride_h == 1)
    //     {
    //         return forwardDilation_arm(bottom_blob_bordered, top_blob, opt);
    //     }

    const int num_input = channels * elempack;

    bool prefer_winograd = (opt.use_winograd23_convolution || opt.use_winograd43_convolution || opt.use_winograd63_convolution) && (num_input >= 8 || num_output >= 8);

    if (opt.use_winograd_convolution && prefer_winograd && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        bool prefer_winograd63 = false;
        bool prefer_winograd23 = false;
        bool prefer_winograd43 = !prefer_winograd63 && !prefer_winograd23;

        if (prefer_winograd23 && (!opt.use_winograd23_convolution || weight_winograd23_data.empty()))
        {
            // f23 fallback to f43
            prefer_winograd23 = false;
            prefer_winograd43 = true;
        }

        if (prefer_winograd63 && (!opt.use_winograd63_convolution || weight_winograd63_data.empty()))
        {
            // f63 fallback to f43
            prefer_winograd63 = false;
            prefer_winograd43 = true;
        }

        if (prefer_winograd43 && (!opt.use_winograd43_convolution || weight_winograd43_data.empty()))
        {
            // f43 fallback to f63 or f23
            prefer_winograd43 = false;
            if (opt.use_winograd63_convolution && !weight_winograd63_data.empty())
            {
                prefer_winograd63 = true;
            }
            else
            {
                prefer_winograd23 = true;
            }
        }
        // NCNN_LOGE("prefer_winograd %d %d %d", prefer_winograd23, prefer_winograd43, prefer_winograd63);

        int _nT = nT ? nT : opt.num_threads;
        if (nT != 0 && opt.num_threads != nT)
        {
            // force num_threads the same as in create_pipeline
            // so we could use pre-packed A/B from the same tile config
            NCNN_LOGE("opt.num_threads %d changed, convolution winograd will use load-time value %d", opt.num_threads, nT);
        }

        if (prefer_winograd23)
        {
            conv3x3s1_winograd23_bf16s(bottom_blob_bordered, top_blob, weight_winograd23_data, bias_data, _nT, opt);
        }
        else if (prefer_winograd43)
        {
            conv3x3s1_winograd43_bf16s(bottom_blob_bordered, top_blob, weight_winograd43_data, bias_data, _nT, opt);
        }
        else if (prefer_winograd63)
        {
            conv3x3s1_winograd63_bf16s(bottom_blob_bordered, top_blob, weight_winograd63_data, bias_data, _nT, opt);
        }
        else
        {
            // should never reach here
        }

        if (activation)
        {
            activation->forward_inplace(top_blob, opt);
        }
        return 0;
    }

    int l2_cache_size_bf16 = get_cpu_level2_cache_size() / sizeof(unsigned short);
    bool prefer_sgemm = num_input * num_output * kernel_w * kernel_h * dilation_w * dilation_h * stride_w * stride_h * 2 > l2_cache_size_bf16 || (num_input > 16 || num_output > 16);

#if NCNN_GNU_INLINE_ASM
    if (elempack == 4 && out_elempack == 4)
    {
        if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2 && (num_input < 4 || num_output < 32))
        {
            prefer_sgemm = false;
        }
        if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            prefer_sgemm = false;
        }
        if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2 && (num_input < 8 || num_output < 44))
        {
            prefer_sgemm = false;
        }
    }

    if (elempack == 1 && out_elempack == 4)
    {
        if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            prefer_sgemm = false;
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            prefer_sgemm = false;
        }
        else if (kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            prefer_sgemm = false;
        }
    }
#endif // NCNN_GNU_INLINE_ASM

    if ((opt.use_sgemm_convolution && prefer_sgemm) || (kernel_w == 1 && kernel_h == 1))
    {
        int _nT = nT ? nT : opt.num_threads;
        if (nT != 0 && opt.num_threads != nT)
        {
            // force num_threads the same as in create_pipeline
            // so we could use pre-packed A/B from the same tile config
            NCNN_LOGE("opt.num_threads %d changed, convolution gemm will use load-time value %d", opt.num_threads, nT);
        }

        convolution_im2col_gemm_bf16s(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, _nT, opt);

        if (activation)
        {
            activation->forward_inplace(top_blob, opt);
        }
        return 0;
    }

#if NCNN_GNU_INLINE_ASM
#if __ARM_NEON
    if (elempack == 4 && out_elempack == 4)
    {
        if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv3x3s2_pack4_bf16s_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv5x5s1_pack4_bf16s_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv5x5s2_pack4_bf16s_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            convolution_packed_bf16s(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }
    }

    if (elempack == 1 && out_elempack == 4)
    {
        if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv3x3s1_pack1to4_bf16s_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv3x3s2_pack1to4_bf16s_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv7x7s2_pack1to4_bf16s_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            convolution_packed_bf16s(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }
    }

    if (elempack == 4 && out_elempack == 1)
    {
        {
            convolution_packed_bf16s(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }
    }
#endif // __ARM_NEON

    if (elempack == 1 && out_elempack == 1)
    {
        {
            convolution_packed_bf16s(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }
    }
#else  // NCNN_GNU_INLINE_ASM
    {
        convolution_packed_bf16s(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
    }
#endif // NCNN_GNU_INLINE_ASM

    return 0;
}
#endif // NCNN_BF16

#if NCNN_INT8
int Convolution_arm::create_pipeline_int8_arm(const Option& opt)
{
    const int maxk = kernel_w * kernel_h;
    const int num_input = weight_data_size / maxk / num_output;

    bool prefer_winograd = (opt.use_winograd23_convolution || opt.use_winograd43_convolution) && (num_input >= 8 && num_output >= 8) && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1;
#if NCNN_ARM82DOT
    if (ncnn::cpu_support_arm_asimddp())
    {
        prefer_winograd = false;
    }
#endif

    if (opt.use_winograd_convolution && prefer_winograd)
    {
        if (opt.use_winograd43_convolution)
            conv3x3s1_winograd43_transform_kernel_int8(weight_data, weight_winograd43_data, num_input, num_output, opt);
        else
            conv3x3s1_winograd23_transform_kernel_int8(weight_data, weight_winograd23_data, num_input, num_output, opt);
    }
    else if (opt.use_sgemm_convolution)
    {
        convolution_im2col_gemm_transform_kernel_int8(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h, opt);
    }
    else
    {
        convolution_transform_kernel_packed_int8(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h);
    }

    scale_in_data.create(num_output);
    for (int p = 0; p < num_output; p++)
    {
        // requantize and relu
        float scale_in;
        if (weight_data_int8_scales[p] == 0)
            scale_in = 0;
        else
            scale_in = 1.f / (bottom_blob_int8_scales[0] * weight_data_int8_scales[p]);

        scale_in_data[p] = scale_in;
    }

    weight_data.release();

    return 0;
}

int Convolution_arm::forward_int8_arm(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int elembits = bottom_blob.elembits();

    Mat bottom_blob_int8 = bottom_blob;
    if (elembits != 8)
    {
        Option opt_q = opt;
        opt_q.blob_allocator = opt.workspace_allocator;
        quantize_to_int8(bottom_blob, bottom_blob_int8, bottom_blob_int8_scales, opt_q);
    }

    //     NCNN_LOGE("Convolution_arm input %d x %d  ksize=%d %d  stride=%d %d", w, h, kernel_w, kernel_h, stride_w, stride_h);

    Mat bottom_blob_bordered;
    make_padding(bottom_blob_int8, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    int w = bottom_blob_bordered.w;
    int h = bottom_blob_bordered.h;
    int elempack = bottom_blob_bordered.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;

    bool use_int8_requantize = int8_scale_term > 100;
    int out_elempack = 1;
#if __ARM_NEON
    if (opt.use_packing_layout)
    {
        if (use_int8_requantize)
            out_elempack = num_output % 8 == 0 ? 8 : 1;
        else
            out_elempack = num_output % 4 == 0 ? 4 : 1;
    }
#endif // __ARM_NEON
    size_t out_elemsize = use_int8_requantize ? 1u * out_elempack : 4u * out_elempack;
#if NCNN_ARM82
    if (support_fp16_storage && opt.use_fp16_storage)
    {
        out_elemsize = use_int8_requantize ? 1u * out_elempack : 2u * out_elempack;
    }
#endif
    if (opt.use_bf16_storage)
        out_elemsize = use_int8_requantize ? 1u * out_elempack : 2u * out_elempack;

    //     NCNN_LOGE("forward_int8_arm %d %d %d    %d %d", w, h, bottom_blob_bordered.c, elempack, out_elempack);

    int channels = bottom_blob_bordered.c;
    const int num_input = channels * elempack;

    bool prefer_winograd = (opt.use_winograd23_convolution || opt.use_winograd43_convolution) && (num_input >= 8 && num_output >= 8) && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1;
#if NCNN_ARM82DOT
    if (ncnn::cpu_support_arm_asimddp())
    {
        prefer_winograd = false;
    }
#endif

    int out_elempack_int32 = 1;
#if __ARM_NEON
    if (opt.use_packing_layout)
    {
        out_elempack_int32 = num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
    }
#endif // __ARM_NEON

    Mat top_blob_int32;
    top_blob_int32.create(outw, outh, num_output / out_elempack_int32, (size_t)(4u * out_elempack_int32), out_elempack_int32, opt.workspace_allocator);
    if (top_blob_int32.empty())
        return -100;

    int _nT = nT ? nT : opt.num_threads;
    if (nT != 0 && opt.num_threads != nT)
    {
        // force num_threads the same as in create_pipeline
        // so we could use pre-packed A/B from the same tile config
        NCNN_LOGE("opt.num_threads %d changed, convolution gemm will use load-time value %d", opt.num_threads, nT);
    }

    if (opt.use_winograd_convolution && prefer_winograd)
    {
        if (opt.use_winograd43_convolution && !weight_winograd43_data.empty())
            conv3x3s1_winograd43_int8(bottom_blob_bordered, top_blob_int32, weight_winograd43_data, _nT, opt);
        else
            conv3x3s1_winograd23_int8(bottom_blob_bordered, top_blob_int32, weight_winograd23_data, _nT, opt);
    }
    else if (opt.use_sgemm_convolution)
    {
        convolution_im2col_gemm_int8(bottom_blob_bordered, top_blob_int32, weight_sgemm_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, _nT, opt);
    }
    else
    {
        convolution_packed_int8(bottom_blob_bordered, top_blob_int32, weight_data_tm, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);
    }

    bottom_blob_bordered.release();

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (use_int8_requantize)
    {
        requantize_from_int32_to_int8(top_blob_int32, top_blob, scale_in_data, top_blob_int8_scales, bias_data, activation_type, activation_params, opt);
    }
    else
    {
        dequantize_from_int32(top_blob_int32, top_blob, scale_in_data, bias_data, opt);

        if (activation)
        {
            activation->forward_inplace(top_blob, opt);
        }
    }

    return 0;
}
#endif // NCNN_INT8

int Convolution_arm::forwardDilation_arm(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    size_t elemsize = bottom_blob.elemsize;

    const int kernel_size = kernel_w;
    const int stride = stride_w;
    const int dilation = dilation_w;
    const int kernel_extent = dilation * (kernel_size - 1) + 1;

    int outw = (w - kernel_extent) / stride + 1;
    int outh = (h - kernel_extent) / stride + 1;

    top_blob.create(outw, outh, num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // Make (dilation * dilation) batches
    Mat inner_bottom_blob;
    Mat inner_top_blob;
    for (int x = 0; x < dilation; x++)
    {
        for (int y = 0; y < dilation; y++)
        {
            int inner_w = (w - y + dilation - 1) / dilation;
            int inner_h = (h - x + dilation - 1) / dilation;

            int inner_outw = (inner_w - kernel_size) / stride + 1;
            int inner_outh = (inner_h - kernel_size) / stride + 1;

            inner_bottom_blob.create(inner_w, inner_h, bottom_blob.c, elemsize, opt.workspace_allocator);
            if (inner_bottom_blob.empty())
                return -100;

            inner_top_blob.create(inner_outw, inner_outh, num_output, elemsize, opt.workspace_allocator);
            if (inner_top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int c = 0; c < bottom_blob.c; c++)
            {
                float* outptr = inner_bottom_blob.channel(c);

                for (int i = 0; i < inner_h; i++)
                {
                    const float* ptr = (const float*)bottom_blob.channel(c) + dilation * i * w + x * w + y;
                    for (int j = 0; j < inner_w; j++)
                    {
                        outptr[j] = ptr[j * dilation];
                    }
                    outptr += inner_w;
                }
            }

            Option opt_g = opt;
            opt_g.blob_allocator = inner_top_blob.allocator;
            convolution_dilation1->forward(inner_bottom_blob, inner_top_blob, opt_g);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int c = 0; c < num_output; c++)
            {
                float* outptr = (float*)top_blob.channel(c) + x * outw + y;
                for (int i = 0; i < inner_outh; i++)
                {
                    const float* ptr = (const float*)inner_top_blob.channel(c) + i * inner_outw;
                    for (int j = 0; j < inner_outw; j++)
                    {
                        outptr[j * dilation] = ptr[j];
                    }
                    outptr += dilation * outw;
                }
            }
        }
    }

    if (activation)
    {
        activation->forward_inplace(top_blob, opt);
    }

    return 0;
}

} // namespace ncnn
