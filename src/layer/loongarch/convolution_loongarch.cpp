// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "convolution_loongarch.h"

#include <algorithm>

#include "benchmark.h"
#include "cpu.h"
#include "layer_type.h"

#if __loongarch_sx
#include <lsxintrin.h>
#if __loongarch_asx
#include <lasxintrin.h>
#endif
#endif // __loongarch_sx

#include "loongarch_activation.h"
#include "loongarch_usability.h"

#include "cpu.h"

namespace ncnn {

#include "convolution_sgemm.h"
#include "convolution_1x1.h"
#include "convolution_3x3_winograd.h"
#include "convolution_packed.h"
#include "convolution_im2col_gemm.h"

#if NCNN_BF16
#include "convolution_packed_bf16s.h"
#include "convolution_im2col_gemm_bf16s.h"
#include "convolution_3x3_winograd_bf16s.h"
#endif

#if NCNN_INT8
#include "convolution_packed_int8.h"
#include "convolution_im2col_gemm_int8.h"

#include "convolution_3x3_winograd_int8.h"
#endif // NCNN_INT8

#if __loongarch_sx
#include "convolution_3x3_pack1to4.h"
#include "convolution_7x7_pack1to4.h"
#endif // __loongarch_sx

Convolution_loongarch::Convolution_loongarch()
{
#if __loongarch_sx
    support_packing = true;
#if NCNN_BF16
    support_bf16_storage = true;
#endif
#endif // __loongarch_sx

    activation = 0;
    nT = 0;
}

static void convolution_transform_kernel_packed_lsx(const Mat& weight_data, Mat& weight_data_tm, int num_input, int num_output, int kernel_w, int kernel_h, int elempack, int out_elempack)
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

static bool test_prefer_winograd63(int num_input, int num_output, int w, int h)
{
    // winograd selection strategy (profiled on i7-7700 single thread)

    const int minwh = std::min(w, h);

    if (num_input >= 64)
    {
        return false;
    }
    if (num_input >= 32)
    {
        if (num_output >= 64) return false;
        if (num_output >= 32) return (minwh >= 11 && minwh <= 14)
                                         || (minwh >= 19 && minwh <= 20)
                                         || (minwh >= 23 && minwh <= 44)
                                         || (minwh >= 47 && minwh <= 56)
                                         || (minwh >= 63 && minwh <= 130);
        if (num_output >= 16) return (minwh >= 13 && minwh <= 14)
                                         || (minwh >= 19 && minwh <= 20)
                                         || (minwh >= 23 && minwh <= 38)
                                         || (minwh >= 43 && minwh <= 44)
                                         || (minwh >= 47 && minwh <= 140);
        if (num_output >= 8) return (minwh >= 11 && minwh <= 14)
                                        || (minwh >= 19 && minwh <= 20)
                                        || (minwh >= 31 && minwh <= 38)
                                        || (minwh >= 43 && minwh <= 44)
                                        || (minwh >= 55 && minwh <= 162);
        return false;
    }
    if (num_input >= 16)
    {
        if (num_output >= 64) return false;
        if (num_output >= 32) return (minwh >= 11 && minwh <= 14)
                                         || (minwh >= 19 && minwh <= 20)
                                         || (minwh >= 23 && minwh <= 44)
                                         || (minwh >= 47 && minwh <= 92)
                                         || (minwh >= 95 && minwh <= 188);
        if (num_output >= 16) return (minwh >= 11 && minwh <= 14)
                                         || (minwh >= 27 && minwh <= 38)
                                         || (minwh >= 43 && minwh <= 44)
                                         || (minwh >= 47 && minwh <= 74)
                                         || (minwh >= 81 && minwh <= 110)
                                         || (minwh >= 117 && minwh <= 170)
                                         || (minwh >= 177 && minwh <= 182);
        if (num_output >= 8) return (minwh >= 19 && minwh <= 20)
                                        || (minwh >= 33 && minwh <= 38)
                                        || (minwh >= 43 && minwh <= 44)
                                        || (minwh >= 47 && minwh <= 128)
                                        || (minwh >= 155 && minwh <= 210);
        return false;
    }
    if (num_input >= 8)
    {
        if (num_output >= 64) return false;
        if (num_output >= 32) return (minwh >= 7 && minwh <= 14)
                                         || (minwh >= 17 && minwh <= 20)
                                         || (minwh >= 23 && minwh <= 26)
                                         || (minwh >= 31 && minwh <= 38)
                                         || (minwh >= 43 && minwh <= 162);
        if (num_output >= 16) return minwh == 31 || minwh == 32
                                         || (minwh >= 39 && minwh <= 44)
                                         || (minwh >= 47 && minwh <= 212);
        if (num_output >= 8) return false;
        return false;
    }

    return false;
}

static bool test_prefer_winograd23(int num_input, int num_output, int w, int h)
{
    // winograd selection strategy (profiled on i7-7700 single thread)

    const int minwh = std::min(w, h);

    if (num_input >= 64)
    {
        return false;
    }
    if (num_input >= 32)
    {
        if (num_output >= 64) return false;
        if (num_output >= 32) return minwh >= 36;
        if (num_output >= 16) return minwh >= 28;
        if (num_output >= 8) return minwh >= 20;
        return false;
    }
    if (num_input >= 16)
    {
        if (num_output >= 64) return false;
        if (num_output >= 32) return minwh >= 24;
        if (num_output >= 16) return minwh >= 16;
        if (num_output >= 8) return minwh >= 12;
        return false;
    }
    if (num_input >= 8)
    {
        if (num_output >= 64) return false;
        if (num_output >= 32) return minwh >= 16;
        if (num_output >= 16) return minwh >= 8;
        if (num_output >= 8) return false;
        return false;
    }

    return false;
}

int Convolution_loongarch::create_pipeline(const Option& opt)
{
    if (dynamic_weight)
        return 0;

    activation = create_activation_layer(activation_type, activation_params, opt);

#if NCNN_INT8
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        return create_pipeline_int8_loongarch(opt);
    }
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage)
    {
        return create_pipeline_bf16s(opt);
    }
#endif

    const int maxk = kernel_w * kernel_h;
    const int num_input = weight_data_size / maxk / num_output;

    nT = opt.num_threads;

    int elempack = 1;
    int out_elempack = 1;
#if __loongarch_sx
    if (opt.use_packing_layout)
    {
        elempack = num_input % 4 == 0 ? 4 : 1;
        out_elempack = num_output % 4 == 0 ? 4 : 1;
    }
#endif

    bool prefer_winograd = (opt.use_winograd23_convolution || opt.use_winograd43_convolution || opt.use_winograd63_convolution) && (num_input > 8 || num_output > 8);

    if (opt.use_winograd_convolution && prefer_winograd && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        if ((bottom_shapes.empty() || bottom_shapes[0].w == 0 || bottom_shapes[0].h == 0) && (top_shapes.empty() || top_shapes[0].w == 0 || top_shapes[0].h == 0))
        {
            // dynamic shape
            if ((opt.use_winograd63_convolution) && (num_input <= 32 && num_output <= 32))
                conv3x3s1_winograd63_transform_kernel(weight_data, weight_winograd63_data, num_input, num_output, opt);
            else if (opt.use_winograd43_convolution)
                conv3x3s1_winograd43_transform_kernel(weight_data, weight_winograd43_data, num_input, num_output, opt);
            else
                conv3x3s1_winograd23_transform_kernel(weight_data, weight_winograd23_data, num_input, num_output, opt);
        }
        else
        {
            int w;
            int h;
            if (top_shapes.empty() || top_shapes[0].w == 0 || top_shapes[0].h == 0)
            {
                w = bottom_shapes[0].w;
                h = bottom_shapes[0].h;

                // make padding
                if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0)
                {
                    w += pad_left + pad_right;
                    h += pad_top + pad_bottom;
                }
                else if ((pad_left == -233 && pad_right == -233 && pad_top == -233 && pad_bottom == -233)
                         || (pad_left == -234 && pad_right == -234 && pad_top == -234 && pad_bottom == -234))
                {
                    // tensorflow padding=SAME or onnx padding=SAME_UPPER/SAME_LOWER
                    w += 2;
                    h += 2;
                }
            }
            else
            {
                w = top_shapes[0].w + 2;
                h = top_shapes[0].h + 2;
            }

            bool prefer_winograd63 = test_prefer_winograd63(num_input, num_output, w, h);
            bool prefer_winograd23 = test_prefer_winograd23(num_input, num_output, w, h);
            bool prefer_winograd43 = !prefer_winograd63 && !prefer_winograd23;

            if (prefer_winograd23 && !opt.use_winograd23_convolution)
            {
                prefer_winograd23 = false;
                prefer_winograd43 = true;
            }

            if (prefer_winograd63 && !opt.use_winograd63_convolution)
            {
                prefer_winograd63 = false;
                prefer_winograd43 = true;
            }

            if (prefer_winograd43 && !opt.use_winograd43_convolution)
            {
                prefer_winograd43 = false;
                if (opt.use_winograd63_convolution)
                {
                    prefer_winograd63 = true;
                }
                else
                {
                    prefer_winograd23 = true;
                }
            }

            if (prefer_winograd23)
            {
                conv3x3s1_winograd23_transform_kernel(weight_data, weight_winograd23_data, num_input, num_output, opt);
            }
            else if (prefer_winograd43)
            {
                conv3x3s1_winograd43_transform_kernel(weight_data, weight_winograd43_data, num_input, num_output, opt);
            }
            else if (prefer_winograd63)
            {
                conv3x3s1_winograd63_transform_kernel(weight_data, weight_winograd63_data, num_input, num_output, opt);
            }
            else
            {
                // should never reach here
            }
        }

        if (opt.lightmode)
            weight_data.release();

        return 0;
    }

    int l2_cache_size = get_cpu_level2_cache_size();
    bool prefer_sgemm = num_input * num_output * kernel_w * kernel_h * dilation_w * dilation_h * stride_w * stride_h * (int)sizeof(float) * 2 > l2_cache_size || (num_input > 16 || num_output > 16);

    if ((opt.use_sgemm_convolution && prefer_sgemm) || (kernel_w == 1 && kernel_h == 1))
    {
        convolution_im2col_gemm_transform_kernel(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h, opt);

        if (opt.lightmode)
            weight_data.release();

        return 0;
    }

#if __loongarch_sx
    if ((elempack == 1 && out_elempack == 4 && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            || (elempack == 1 && out_elempack == 4 && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            || (elempack == 1 && out_elempack == 4 && kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2))
    {
        convolution_transform_kernel_packed_lsx(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
    }
    else
#endif // __loongarch_sx
    {
        convolution_transform_kernel_packed(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h);
    }

    if (opt.lightmode)
        weight_data.release();

    return 0;
}

int Convolution_loongarch::destroy_pipeline(const Option& opt)
{
    if (activation)
    {
        activation->destroy_pipeline(opt);
        delete activation;
        activation = 0;
    }

    return 0;
}

int Convolution_loongarch::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if NCNN_INT8
    if (opt.use_int8_inference && int8_scale_term)
    {
#if NCNN_BF16
        if (opt.use_bf16_storage && bottom_blob.elembits() == 16)
        {
            Mat bottom_blob_fp32;
            cast_bfloat16_to_float32(bottom_blob, bottom_blob_fp32, opt);
            if (bottom_blob_fp32.empty())
                return -100;

            return forward_int8_loongarch(bottom_blob_fp32, top_blob, opt);
        }
#endif
        return forward_int8_loongarch(bottom_blob, top_blob, opt);
    }
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_blob.elembits() == 16)
    {
        return forward_bf16s(bottom_blob, top_blob, opt);
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
            top_blob.cstep = top_blob_3d.c;
        }
        else
        {
            top_blob = top_blob_3d.reshape(top_blob_3d.c, opt.blob_allocator);
        }

        return 0;
    }

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
#if __loongarch_sx
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % 4 == 0 ? 4 : 1;
    }
#endif
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const int num_input = channels * elempack;

    bool prefer_winograd = (opt.use_winograd23_convolution || opt.use_winograd43_convolution || opt.use_winograd63_convolution) && (num_input > 8 || num_output > 8);

    if (opt.use_winograd_convolution && prefer_winograd && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        bool prefer_winograd63 = test_prefer_winograd63(num_input, num_output, w, h);
        bool prefer_winograd23 = test_prefer_winograd23(num_input, num_output, w, h);
        bool prefer_winograd43 = !prefer_winograd63 && !prefer_winograd23;

        if (prefer_winograd23 && (!opt.use_winograd23_convolution || weight_winograd23_data.empty()))
        {
            prefer_winograd23 = false;
            prefer_winograd43 = true;
        }

        if (prefer_winograd63 && (!opt.use_winograd63_convolution || weight_winograd63_data.empty()))
        {
            prefer_winograd63 = false;
            prefer_winograd43 = true;
        }

        if (prefer_winograd43 && (!opt.use_winograd43_convolution || weight_winograd43_data.empty()))
        {
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

        int _nT = nT ? nT : opt.num_threads;
        if (nT != 0 && opt.num_threads != nT)
        {
            // force num_threads the same as in create_pipeline
            // so we could use pre-packed A/B from the same tile config
            NCNN_LOGE("opt.num_threads %d changed, convolution winograd will use load-time value %d", opt.num_threads, nT);
        }

        int ret = 0;
        if (prefer_winograd23)
        {
            ret = conv3x3s1_winograd23(bottom_blob_bordered, top_blob, weight_winograd23_data, bias_data, _nT, opt);
        }
        else if (prefer_winograd43)
        {
            ret = conv3x3s1_winograd43(bottom_blob_bordered, top_blob, weight_winograd43_data, bias_data, _nT, opt);
        }
        else if (prefer_winograd63)
        {
            ret = conv3x3s1_winograd63(bottom_blob_bordered, top_blob, weight_winograd63_data, bias_data, _nT, opt);
        }
        else
        {
            // should never reach here
        }
        if (ret != 0)
            return ret;

        if (activation)
        {
            activation->forward_inplace(top_blob, opt);
        }
        return 0;
    }

    int l2_cache_size = get_cpu_level2_cache_size();
    bool prefer_sgemm = num_input * num_output * kernel_w * kernel_h * dilation_w * dilation_h * stride_w * stride_h * (int)sizeof(float) * 2 > l2_cache_size || (num_input > 16 || num_output > 16);

    if ((opt.use_sgemm_convolution && prefer_sgemm) || (kernel_w == 1 && kernel_h == 1))
    {
        int _nT = nT ? nT : opt.num_threads;
        if (nT != 0 && opt.num_threads != nT)
        {
            // force num_threads the same as in create_pipeline
            // so we could use pre-packed A/B from the same tile config
            NCNN_LOGE("opt.num_threads %d changed, convolution gemm will use load-time value %d", opt.num_threads, nT);
        }

        int ret = convolution_im2col_gemm(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, _nT, opt);
        if (ret != 0)
            return ret;

        if (activation)
        {
            activation->forward_inplace(top_blob, opt);
        }
        return 0;
    }

#if __loongarch_sx
    if (elempack == 1 && out_elempack == 4)
    {
        if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv3x3s1_pack1to4_lsx(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
            return 0;
        }
        if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv3x3s2_pack1to4_lsx(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
            return 0;
        }
        if (kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv7x7s2_pack1to4_lsx(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
            return 0;
        }
    }
#endif // __loongarch_sx

    convolution_packed(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);

    return 0;
}

int Convolution_loongarch::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
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

#if NCNN_BF16
    if (opt.use_bf16_storage && weight_data_flattened.elembits() == 16)
    {
        Mat weight_data_flattened_fp32;
        cast_bfloat16_to_float32(weight_data_flattened, weight_data_flattened_fp32, opt);
        weight_data_flattened = weight_data_flattened_fp32;
        if (weight_data_flattened.empty())
            return -100;
    }
#endif

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

#if NCNN_BF16
        if (opt.use_bf16_storage && bias_data_flattened.elembits() == 16)
        {
            Mat bias_data_flattened_fp32;
            cast_bfloat16_to_float32(bias_data_flattened, bias_data_flattened_fp32, opt);
            bias_data_flattened = bias_data_flattened_fp32;
            if (bias_data_flattened.empty())
                return -100;
        }
#endif

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

#if NCNN_INT8
int Convolution_loongarch::create_pipeline_int8_loongarch(const Option& opt)
{
    const int maxk = kernel_w * kernel_h;
    const int num_input = weight_data_size / maxk / num_output;

    nT = opt.num_threads;

    int elempack = 1;
    int out_elempack = 1;
#if __loongarch_sx
    if (opt.use_packing_layout)
    {
        elempack = num_input % 8 == 0 ? 8 : 1;
        out_elempack = num_output % 4 == 0 ? 4 : 1;
    }
#endif // __loongarch_sx

    bool prefer_winograd = (opt.use_winograd23_convolution || opt.use_winograd43_convolution) && (num_input > 8 || num_output > 8);

    if (opt.use_winograd_convolution && prefer_winograd && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
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

    if (opt.lightmode)
        weight_data.release();

    return 0;
}

int Convolution_loongarch::forward_int8_loongarch(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int elembits = bottom_blob.elembits();

    Mat bottom_blob_int8 = bottom_blob;
    if (elembits != 8)
    {
        Option opt_q = opt;
        opt_q.blob_allocator = opt.workspace_allocator;
        quantize_to_int8(bottom_blob, bottom_blob_int8, bottom_blob_int8_scales, opt_q);
    }

    Mat bottom_blob_bordered;
    make_padding(bottom_blob_int8, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    int w = bottom_blob_bordered.w;
    int h = bottom_blob_bordered.h;
    int channels = bottom_blob_bordered.c;
    int elempack = bottom_blob_bordered.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;

    bool use_int8_requantize = int8_scale_term > 100;
    int out_elempack = 1;
#if __loongarch_sx
    if (opt.use_packing_layout)
    {
        if (use_int8_requantize)
            out_elempack = num_output % 8 == 0 ? 8 : 1;
        else
            out_elempack = num_output % 4 == 0 ? 4 : 1;
    }
#endif // __loongarch_sx
    size_t out_elemsize = use_int8_requantize ? 1u * out_elempack : 4u * out_elempack;

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const int num_input = channels * elempack;

    int out_elempack_int32 = 1;
#if __loongarch_sx
    if (opt.use_packing_layout)
    {
        out_elempack_int32 = num_output % 4 == 0 ? 4 : 1;
    }
#endif // __loongarch_sx

    Mat top_blob_int32;
    top_blob_int32.create(outw, outh, num_output / out_elempack_int32, (size_t)(4u * out_elempack_int32), out_elempack_int32, opt.workspace_allocator);
    if (top_blob_int32.empty())
        return -100;

    bool prefer_winograd = (opt.use_winograd23_convolution || opt.use_winograd43_convolution) && (num_input > 8 || num_output > 8);

    if (opt.use_winograd_convolution && prefer_winograd && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        int _nT = nT ? nT : opt.num_threads;
        if (nT != 0 && opt.num_threads != nT)
        {
            // force num_threads the same as in create_pipeline
            // so we could use pre-packed A/B from the same tile config
            NCNN_LOGE("opt.num_threads %d changed, convolution gemm will use load-time value %d", opt.num_threads, nT);
        }

        int ret = 0;
        if (!weight_winograd43_data.empty())
            ret = conv3x3s1_winograd43_int8(bottom_blob_bordered, top_blob_int32, weight_winograd43_data, _nT, opt);
        else
            ret = conv3x3s1_winograd23_int8(bottom_blob_bordered, top_blob_int32, weight_winograd23_data, _nT, opt);
        if (ret != 0)
            return ret;
    }
    else if (opt.use_sgemm_convolution && !weight_sgemm_data.empty())
    {
        int _nT = nT ? nT : opt.num_threads;
        if (nT != 0 && opt.num_threads != nT)
        {
            // force num_threads the same as in create_pipeline
            // so we could use pre-packed A/B from the same tile config
            NCNN_LOGE("opt.num_threads %d changed, convolution gemm will use load-time value %d", opt.num_threads, nT);
        }

        int ret = convolution_im2col_gemm_int8(bottom_blob_bordered, top_blob_int32, weight_sgemm_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, _nT, opt);
        if (ret != 0)
            return ret;
    }
    else
    {
        convolution_packed_int8(bottom_blob_bordered, top_blob_int32, weight_data_tm, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);
    }

#if __loongarch_sx
    if (opt.use_packing_layout)
    {
        // NCNN_LOGE("top_blob_int32  %d  %d", top_blob_int32.c, top_blob_int32.elempack);
        if (use_int8_requantize)
        {
            // TODO implement winograd sgemm packed int8 pack1 output
            if (top_blob_int32.elempack == 4 && top_blob_int32.c % 2 == 1)
            {
                Mat tmp;
                convert_packing(top_blob_int32, tmp, 1, opt);
                top_blob_int32 = tmp;
            }
            if (top_blob_int32.elempack == 4 && top_blob_int32.c % 2 == 0)
            {
                Mat tmp;
                convert_packing(top_blob_int32, tmp, 8, opt);
                top_blob_int32 = tmp;
            }
        }
    }
#endif

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

#if NCNN_BF16
int Convolution_loongarch::create_pipeline_bf16s(const Option& opt)
{
    const int maxk = kernel_w * kernel_h;
    const int num_input = weight_data_size / maxk / num_output;

    nT = opt.num_threads;

    bool prefer_winograd = (opt.use_winograd23_convolution || opt.use_winograd43_convolution || opt.use_winograd63_convolution) && (num_input > 8 || num_output > 8);

    if (opt.use_winograd_convolution && prefer_winograd && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        if ((bottom_shapes.empty() || bottom_shapes[0].w == 0 || bottom_shapes[0].h == 0) && (top_shapes.empty() || top_shapes[0].w == 0 || top_shapes[0].h == 0))
        {
            if ((opt.use_winograd63_convolution) && (num_input <= 32 && num_output <= 32))
                conv3x3s1_winograd63_transform_kernel(weight_data, weight_winograd63_data, num_input, num_output, opt);
            else if (opt.use_winograd43_convolution)
                conv3x3s1_winograd43_transform_kernel(weight_data, weight_winograd43_data, num_input, num_output, opt);
            else
                conv3x3s1_winograd23_transform_kernel(weight_data, weight_winograd23_data, num_input, num_output, opt);
        }
        else
        {
            int w;
            int h;
            if (top_shapes.empty() || top_shapes[0].w == 0 || top_shapes[0].h == 0)
            {
                w = bottom_shapes[0].w;
                h = bottom_shapes[0].h;

                if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0)
                {
                    w += pad_left + pad_right;
                    h += pad_top + pad_bottom;
                }
                else if ((pad_left == -233 && pad_right == -233 && pad_top == -233 && pad_bottom == -233)
                         || (pad_left == -234 && pad_right == -234 && pad_top == -234 && pad_bottom == -234))
                {
                    w += 2;
                    h += 2;
                }
            }
            else
            {
                w = top_shapes[0].w + 2;
                h = top_shapes[0].h + 2;
            }

            bool prefer_winograd63 = test_prefer_winograd63(num_input, num_output, w, h);
            bool prefer_winograd23 = test_prefer_winograd23(num_input, num_output, w, h);
            bool prefer_winograd43 = !prefer_winograd63 && !prefer_winograd23;

            if (prefer_winograd23 && !opt.use_winograd23_convolution)
            {
                prefer_winograd23 = false;
                prefer_winograd43 = true;
            }
            if (prefer_winograd63 && !opt.use_winograd63_convolution)
            {
                prefer_winograd63 = false;
                prefer_winograd43 = true;
            }
            if (prefer_winograd43 && !opt.use_winograd43_convolution)
            {
                prefer_winograd43 = false;
                if (opt.use_winograd63_convolution)
                    prefer_winograd63 = true;
                else
                    prefer_winograd23 = true;
            }

            if (prefer_winograd23)
                conv3x3s1_winograd23_transform_kernel(weight_data, weight_winograd23_data, num_input, num_output, opt);
            else if (prefer_winograd43)
                conv3x3s1_winograd43_transform_kernel(weight_data, weight_winograd43_data, num_input, num_output, opt);
            else if (prefer_winograd63)
                conv3x3s1_winograd63_transform_kernel(weight_data, weight_winograd63_data, num_input, num_output, opt);
        }

        if (opt.lightmode)
            weight_data.release();

        return 0;
    }

    int l2_cache_size = get_cpu_level2_cache_size();
    bool prefer_sgemm = num_input * num_output * kernel_w * kernel_h * dilation_w * dilation_h * stride_w * stride_h * (int)sizeof(unsigned short) * 2 > l2_cache_size || (num_input > 16 || num_output > 16);

    if ((opt.use_sgemm_convolution && prefer_sgemm) || (kernel_w == 1 && kernel_h == 1))
    {
        convolution_im2col_gemm_transform_kernel_bf16s(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h, opt);

        if (opt.lightmode)
            weight_data.release();

        return 0;
    }

    convolution_transform_kernel_packed_bf16s(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h);

    if (opt.lightmode)
        weight_data.release();

    return 0;
}

int Convolution_loongarch::forward_bf16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
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
            if (bottom_blob_3d.empty())
                return -100;
        }

        Mat top_blob_3d;
        int ret = forward_bf16s(bottom_blob_3d, top_blob_3d, opt);
        if (ret != 0)
            return ret;

        if (top_blob_3d.elemsize % 16 == 0)
        {
            top_blob = top_blob_3d;
            top_blob.dims = 1;
            top_blob.w = top_blob_3d.c;
            top_blob.h = 1;
            top_blob.c = 1;
            top_blob.cstep = top_blob_3d.c;
        }
        else
        {
            top_blob = top_blob_3d.reshape(top_blob_3d.c, opt.blob_allocator);
            if (top_blob.empty())
                return -100;
        }

        return 0;
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

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
#if __loongarch_sx
    if (opt.use_packing_layout)
    {
#if __loongarch_asx
        out_elempack = num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
#else
        out_elempack = num_output % 4 == 0 ? 4 : 1;
#endif
    }
#endif
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const int num_input = channels * elempack;

    bool prefer_winograd = (opt.use_winograd23_convolution || opt.use_winograd43_convolution || opt.use_winograd63_convolution) && (num_input > 8 || num_output > 8);

    if (opt.use_winograd_convolution && prefer_winograd && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        bool prefer_winograd63 = test_prefer_winograd63(num_input, num_output, w, h);
        bool prefer_winograd23 = test_prefer_winograd23(num_input, num_output, w, h);
        bool prefer_winograd43 = !prefer_winograd63 && !prefer_winograd23;

        if (prefer_winograd23 && (!opt.use_winograd23_convolution || weight_winograd23_data.empty()))
        {
            prefer_winograd23 = false;
            prefer_winograd43 = true;
        }
        if (prefer_winograd63 && (!opt.use_winograd63_convolution || weight_winograd63_data.empty()))
        {
            prefer_winograd63 = false;
            prefer_winograd43 = true;
        }
        if (prefer_winograd43 && (!opt.use_winograd43_convolution || weight_winograd43_data.empty()))
        {
            prefer_winograd43 = false;
            if (opt.use_winograd63_convolution && !weight_winograd63_data.empty())
                prefer_winograd63 = true;
            else
                prefer_winograd23 = true;
        }

        int _nT = nT ? nT : opt.num_threads;
        if (nT != 0 && opt.num_threads != nT)
        {
            NCNN_LOGE("opt.num_threads %d changed, convolution winograd will use load-time value %d", opt.num_threads, nT);
        }

        int ret = 0;
        if (prefer_winograd23)
        {
            ret = conv3x3s1_winograd23_bf16s(bottom_blob_bordered, top_blob, weight_winograd23_data, bias_data, _nT, activation_type, activation_params, opt);
        }
        else if (prefer_winograd43)
        {
            ret = conv3x3s1_winograd43_bf16s(bottom_blob_bordered, top_blob, weight_winograd43_data, bias_data, _nT, activation_type, activation_params, opt);
        }
        else if (prefer_winograd63)
        {
            ret = conv3x3s1_winograd63_bf16s(bottom_blob_bordered, top_blob, weight_winograd63_data, bias_data, _nT, activation_type, activation_params, opt);
        }
        if (ret != 0)
            return ret;

        return 0;
    }

    int l2_cache_size = get_cpu_level2_cache_size();
    bool prefer_sgemm = num_input * num_output * kernel_w * kernel_h * dilation_w * dilation_h * stride_w * stride_h * (int)sizeof(unsigned short) * 2 > l2_cache_size || (num_input > 16 || num_output > 16);

    if ((opt.use_sgemm_convolution && prefer_sgemm) || (kernel_w == 1 && kernel_h == 1))
    {
        int _nT = nT ? nT : opt.num_threads;
        if (nT != 0 && opt.num_threads != nT)
        {
            NCNN_LOGE("opt.num_threads %d changed, convolution gemm will use load-time value %d", opt.num_threads, nT);
        }

        int ret = convolution_im2col_gemm_bf16s(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, _nT, opt);
        if (ret != 0)
            return ret;

        return 0;
    }

    convolution_packed_bf16s(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
