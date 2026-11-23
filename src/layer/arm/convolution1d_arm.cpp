// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convolution1d_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "arm_usability.h"
#include "arm_activation.h"

#include "cpu.h"
#include "layer_type.h"

namespace ncnn {

#include "convolution1d_packed.h"
#if NCNN_BF16
#include "convolution1d_packed_bf16s.h"
#endif // NCNN_BF16

Convolution1D_arm::Convolution1D_arm()
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

    gemm = 0;
}

int Convolution1D_arm::create_pipeline(const Option& opt)
{
    if (dynamic_weight)
        return 0;

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

    const int num_input = weight_data_size / kernel_w / num_output;

    convolution1d_transform_kernel_packed(weight_data, weight_data_tm, num_input, num_output, kernel_w);

    if (opt.lightmode)
        weight_data.release();

    return 0;
}

int Convolution1D_arm::destroy_pipeline(const Option& opt)
{
    if (gemm)
    {
        gemm->destroy_pipeline(opt);
        delete gemm;
        gemm = 0;
    }

    return 0;
}

int Convolution1D_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
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
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;

    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;

    int out_elempack = 1;
#if __ARM_NEON
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % 4 == 0 ? 4 : 1;
    }
#endif
    size_t out_elemsize = elemsize / elempack * out_elempack;

    const int outw = (w - kernel_extent_w) / stride_w + 1;
    const int outh = num_output / out_elempack;

    top_blob.create(outw, outh, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    convolution1d_packed(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, dilation_w, stride_w, activation_type, activation_params, opt);

    return 0;
}

int Convolution1D_arm::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& _weight_data = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];

    const int _kernel_w = _weight_data.w;
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

    ncnn::Layer* op = ncnn::create_layer_cpu(ncnn::LayerType::Convolution1D);

    ncnn::ParamDict pd;
    pd.set(0, _num_output);
    pd.set(1, _kernel_w);
    pd.set(2, dilation_w);
    pd.set(3, stride_w);
    pd.set(4, pad_left);
    pd.set(15, pad_right);
    pd.set(18, pad_value);
    pd.set(5, bias_term);
    pd.set(6, weight_data_flattened.w);
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
int Convolution1D_arm::create_pipeline_bf16s(const Option& opt)
{
    const int num_input = weight_data_size / kernel_w / num_output;

    if (opt.use_sgemm_convolution)
    {
        // bf16 storage with bf16 gemm avoids the fp32 im2col and weight copies
        gemm = ncnn::create_layer_cpu(ncnn::LayerType::Gemm);
        if (!gemm)
            return -100;

        const int K = kernel_w * num_input;
        int out_elempack = 1;
#if __ARM_NEON
        if (opt.use_packing_layout)
            out_elempack = num_output % 4 == 0 ? 4 : 1;
#endif

        ncnn::ParamDict pd;
        pd.set(2, 0);          // transA
        pd.set(3, 1);          // transB
        pd.set(4, 0);          // constantA
        pd.set(5, 1);          // constantB
        pd.set(6, 0);          // constantC
        pd.set(7, 0);          // M
        pd.set(8, num_output); // N
        pd.set(9, K);          // K
        pd.set(10, -1);        // constant_broadcast_type_C
        pd.set(11, 0);         // output_N1M
        pd.set(12, out_elempack);
        pd.set(14, 1); // output_transpose

        int ret = gemm->load_param(pd);
        if (ret != 0)
        {
            delete gemm;
            gemm = 0;
            return ret;
        }

        Mat weight_data_gemm(K, num_output, (size_t)4u);
        for (int p = 0; p < num_output; p++)
        {
            float* outptr = weight_data_gemm.row(p);
            for (int k = 0; k < kernel_w; k++)
            {
                for (int q = 0; q < num_input; q++)
                {
                    outptr[q * kernel_w + k] = weight_data[(p * num_input + q) * kernel_w + k];
                }
            }
        }

        ncnn::Mat weights[1];
        weights[0] = weight_data_gemm;
        ret = gemm->load_model(ModelBinFromMatArray(weights));
        if (ret != 0)
        {
            delete gemm;
            gemm = 0;
            return ret;
        }

        ret = gemm->create_pipeline(opt);
        if (ret != 0)
        {
            delete gemm;
            gemm = 0;
            return ret;
        }

        convolution1d_transform_kernel_packed_bf16s(weight_data, weight_data_tm, num_input, num_output, kernel_w);

        if (opt.lightmode)
            weight_data.release();

        return 0;
    }

    convolution1d_transform_kernel_packed_bf16s(weight_data, weight_data_tm, num_input, num_output, kernel_w);

    if (opt.lightmode)
        weight_data.release();

    return 0;
}

int Convolution1D_arm::forward_bf16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;

    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;

    int out_elempack = 1;
#if __ARM_NEON
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % 4 == 0 ? 4 : 1;
    }
#endif
    size_t out_elemsize = elemsize / elempack * out_elempack;

    const int outw = (w - kernel_extent_w) / stride_w + 1;
    const int outh = num_output / out_elempack;

    const int num_input = bottom_blob.h * bottom_blob.elempack;
    const long long work = (long long)outw * num_input * num_output * kernel_w;
    const long long large_work_threshold = 10 * 1024 * 1024;
    const long long packed_work_threshold = 6 * 1024 * 1024;
    const bool use_gemm = work >= large_work_threshold || (work >= packed_work_threshold && num_output >= 128);

    if (gemm && use_gemm)
    {
        Mat bottom_blob_pack1;
        if (bottom_blob_bordered.elempack == 1)
        {
            bottom_blob_pack1 = bottom_blob_bordered;
        }
        else
        {
            convert_packing(bottom_blob_bordered, bottom_blob_pack1, 1, opt);
            if (bottom_blob_pack1.empty())
                return -100;
        }

        const int num_input = bottom_blob_pack1.h;
        const int K = kernel_w * num_input;

        Mat im2col(K, outw, (size_t)2u, 1, opt.workspace_allocator);
        if (im2col.empty())
            return -100;

        for (int x = 0; x < outw; x++)
        {
            unsigned short* outptr = im2col.row<unsigned short>(x);

            for (int k = 0; k < kernel_w; k++)
            {
                const int sx = x * stride_w + k * dilation_w;

                for (int q = 0; q < num_input; q++)
                {
                    outptr[q * kernel_w + k] = bottom_blob_pack1.row<const unsigned short>(q)[sx];
                }
            }
        }

        Option opt_gemm = opt;
        opt_gemm.blob_allocator = opt.blob_allocator;
        opt_gemm.workspace_allocator = opt.workspace_allocator;

        int ret = gemm->forward(im2col, top_blob, opt_gemm);
        if (ret != 0)
            return ret;

        // apply bias and activation after bf16 gemm to keep the convolution core packed
        if (out_elempack == 4)
        {
            for (int p = 0; p < outh; p++)
            {
                unsigned short* outptr = top_blob.row<unsigned short>(p);

                for (int x = 0; x < outw; x++)
                {
                    float32x4_t _val = bfloat2float(vld1_u16(outptr + x * 4));
                    if (bias_term)
                        _val = vaddq_f32(_val, vld1q_f32((const float*)bias_data + p * 4));
                    _val = activation_ps(_val, activation_type, activation_params);
                    vst1_u16(outptr + x * 4, float2bfloat(_val));
                }
            }
        }

        if (out_elempack == 1)
        {
            for (int p = 0; p < outh; p++)
            {
                unsigned short* outptr = top_blob.row<unsigned short>(p);

                for (int x = 0; x < outw; x++)
                {
                    float val = bfloat16_to_float32(outptr[x]);
                    if (bias_term)
                        val += bias_data[p];
                    val = activation_ss(val, activation_type, activation_params);
                    outptr[x] = float32_to_bfloat16(val);
                }
            }
        }

        return 0;
    }

    top_blob.create(outw, outh, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    convolution1d_packed_bf16s(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, dilation_w, stride_w, activation_type, activation_params, opt);

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
