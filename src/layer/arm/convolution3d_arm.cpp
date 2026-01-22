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

#include "convolution3d_arm.h"

#include "benchmark.h"
#include "cpu.h"
#include "layer_type.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "arm_activation.h"
#include "arm_usability.h"

namespace ncnn {

#include "convolution3d_sgemm.h"

#if __ARM_NEON
#include "convolution3d_pack4.h"
#include "convolution3d_pack1to4.h"
#include "convolution3d_pack4to1.h"
#include "convolution3d_sgemm_pack4.h"
#endif // __ARM_NEON
Convolution3D_arm::Convolution3D_arm()
{
#if __ARM_NEON
    support_packing = true;

#endif // __ARM_NEON

#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int Convolution3D_arm::create_pipeline(const Option& opt)
{
    activation = create_activation_layer(activation_type, activation_params, opt);

    const int maxk = kernel_w * kernel_h * kernel_d;
    const int num_input = weight_data_size / maxk / num_output;

    int elempack = (support_packing && opt.use_packing_layout && num_input % 4 == 0) ? 4 : 1;
    int out_elempack = (support_packing && opt.use_packing_layout && num_output % 4 == 0) ? 4 : 1;
#if __ARM_NEON

    // pack4
    if (elempack == 4 && out_elempack == 4)
    {
        //Have not tested, just imitate the case of 2d
        bool prefer_sgemm = (dilation_w == 1 && dilation_h == 1 && dilation_d == 1 && stride_w == 1 && stride_h == 1 && stride_d == 1 && num_input >= 12 && num_output >= 12)
                            || (dilation_w == 1 && dilation_h == 1 && dilation_d == 1 && (stride_w >= 2 || stride_h >= 2 || stride_d >= 2) && num_input >= 16 && num_output >= 16)
                            || ((dilation_w >= 2 || dilation_h >= 2 || dilation_d >= 2) && num_input >= 16 && num_output >= 16);
        if (opt.use_sgemm_convolution && prefer_sgemm)
        {
            convolution3D_vi2col_sgemm_transform_kernel_pack4_neon(weight_data, weight_sgemm_data_pack4, num_input, num_output, kernel_w, kernel_h, kernel_d);
        }
        else
        {
            convolution3D_transform_kernel_pack4_neon(weight_data, weight_data_pack4, num_input, num_output, kernel_w, kernel_h, kernel_d);
        }
    }

    // pack1to4
    if (elempack == 1 && out_elempack == 4)
    {
        convolution3D_transform_kernel_pack1to4_neon(weight_data, weight_data_pack1to4, num_input, num_output, kernel_w, kernel_h, kernel_d);
    }

    // pack4to1
    if (elempack == 4 && out_elempack == 1)
    {
        convolution3D_transform_kernel_pack4to1_neon(weight_data, weight_data_pack4to1, num_input, num_output, kernel_w, kernel_h, kernel_d);
    }
#endif
    // pack1
    if (elempack == 1 && out_elempack == 1)
    {
        if (opt.use_sgemm_convolution && kernel_w == 3 && kernel_h == 3 && kernel_d == 3 && dilation_w == 1 && dilation_h == 1 && dilation_d == 1 && stride_w == 1 && stride_h == 1 && stride_d == 1)
        {
            convolution3D_vi2col_sgemm_transform_kernel_neon(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h, kernel_d);
        }

        if (impl_type > 0 && impl_type < 6 && impl_type != 4)
        {
            switch (impl_type)
            {
            case 1:
                // im2col
                convolution3D_vi2col_sgemm_transform_kernel_neon(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h, kernel_d);
                break;
            }
        }
    }

    return 0;
}

int Convolution3D_arm::destroy_pipeline(const Option& opt)
{
    if (activation)
    {
        activation->destroy_pipeline(opt);
        delete activation;
        activation = 0;
    }
    return 0;
}

int Convolution3D_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int elembits = bottom_blob.elembits();

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;
    const int kernel_extent_d = dilation_d * (kernel_d - 1) + 1;

    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;
    d = bottom_blob_bordered.d;
    int size = w * h * d;

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;
    int outd = (d - kernel_extent_d) / stride_d + 1;
    int out_elempack = (support_packing && opt.use_packing_layout && num_output % 4 == 0) ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(outw, outh, outd, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const int num_input = channels * elempack;

#if __ARM_NEON
    if (elempack == 4 && out_elempack == 4)
    {
        bool prefer_sgemm = (dilation_w == 1 && dilation_h == 1 && dilation_d == 1 && stride_w == 1 && stride_h == 1 && stride_d == 1 && num_input >= 12 && num_output >= 12)
                            || (dilation_w == 1 && dilation_h == 1 && dilation_d == 1 && (stride_w >= 2 || stride_h >= 2 || stride_d >= 2) && num_input >= 16 && num_output >= 16)
                            || ((dilation_w >= 2 || dilation_h >= 2 || dilation_d >= 2) && num_input >= 16 && num_output >= 16);
        if (opt.use_sgemm_convolution && prefer_sgemm)
        {
            convolution3D_vi2col_sgemm_pack4_neon(bottom_blob_bordered, top_blob, weight_sgemm_data_pack4, bias_data, kernel_w, kernel_h, kernel_d, dilation_w, dilation_h, dilation_d, stride_w, stride_h, stride_d, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            convolution3D_pack4_neon(bottom_blob_bordered, top_blob, weight_data_pack4, bias_data, kernel_w, kernel_h, kernel_d, dilation_w, dilation_h, dilation_d, stride_w, stride_h, stride_d, activation_type, activation_params, opt);
        }
    }

    if (elempack == 1 && out_elempack == 4)
    {
        convolution3D_pack1to4_neon(bottom_blob_bordered, top_blob, weight_data_pack1to4, bias_data, kernel_w, kernel_h, kernel_d, dilation_w, dilation_h, dilation_d, stride_w, stride_h, stride_d, activation_type, activation_params, opt);
    }

    if (elempack == 4 && out_elempack == 1)
    {
        convolution3D_pack4to1_neon(bottom_blob_bordered, top_blob, weight_data_pack4to1, bias_data, kernel_w, kernel_h, kernel_d, dilation_w, dilation_h, dilation_d, stride_w, stride_h, stride_d, activation_type, activation_params, opt);
    }
#endif // __ARM_NEON

    if (elempack == 1 && out_elempack == 1)
    {
        if (impl_type > 0 && impl_type < 6 && impl_type != 4)
        {
            // engineering is magic.
            switch (impl_type)
            {
            case 1:
                convolution3D_vi2col_sgemm_neon(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, kernel_w, kernel_h, kernel_d, dilation_w, dilation_h, dilation_d, stride_w, stride_h, stride_d, opt);
                break;
            }
            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }

        else
        {
            const int maxk = kernel_w * kernel_h * kernel_d;

            // kernel offsets
            std::vector<int> _space_ofs(maxk);
            int* space_ofs = &_space_ofs[0];
            {
                int p1 = 0;
                int p2 = 0;
                int gap0 = w * dilation_h - kernel_w * dilation_w;
                int gap1 = h * w * dilation_d - w * kernel_h * dilation_h;
                for (int z = 0; z < kernel_d; z++)
                {
                    for (int i = 0; i < kernel_h; i++)
                    {
                        for (int j = 0; j < kernel_w; j++)
                        {
                            space_ofs[p1] = p2;
                            p1++;
                            p2 += dilation_w;
                        }
                        p2 += gap0;
                    }
                    p2 += gap1;
                }
            }

            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output; p++)
            {
                float* outptr = top_blob.channel(p);

                for (int z = 0; z < outd; z++)
                {
                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            float sum = 0.f;

                            if (bias_term)
                                sum = bias_data[p];

                            const float* kptr = (const float*)weight_data + maxk * channels * p;

                            for (int q = 0; q < channels; q++)
                            {
                                const Mat m = bottom_blob_bordered.channel(q);
                                const float* sptr = m.depth(z * stride_d).row(i * stride_h) + j * stride_w;

                                for (int l = 0; l < maxk; l++)
                                {
                                    float val = sptr[space_ofs[l]];

                                    float wt = kptr[l];
                                    sum += val * wt;
                                }

                                kptr += maxk;
                            }

                            outptr[j] = activation_ss(sum, activation_type, activation_params);
                        }

                        outptr += outw;
                    }
                }
            }
        }
    }

    return 0;
}

int Convolution3D_arm::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& _weight_data = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];

    const int _kernel_w = _weight_data.w;
    const int _kernel_h = _weight_data.h;
    const int _kernel_d = _weight_data.d;
    const int _num_output = _weight_data.c * _weight_data.elempack;

    Mat weight_data_flattened;
    flatten(_weight_data, weight_data_flattened, opt);
    if (weight_data_flattened.empty())
        return -100;

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

        // bias_data_flattened as pack1
        bias_data_flattened.w *= bias_data_flattened.elempack;
        bias_data_flattened.elemsize /= bias_data_flattened.elempack;
        bias_data_flattened.elempack = 1;
    }

    ncnn::Layer* op = ncnn::create_layer(ncnn::LayerType::Convolution3D);

    ncnn::ParamDict pd;
    pd.set(0, _num_output);
    pd.set(1, _kernel_w);
    pd.set(11, _kernel_h);
    pd.set(21, _kernel_d);
    pd.set(2, dilation_w);
    pd.set(12, dilation_h);
    pd.set(22, dilation_d);
    pd.set(3, stride_w);
    pd.set(13, stride_h);
    pd.set(23, stride_d);
    pd.set(4, pad_left);
    pd.set(15, pad_right);
    pd.set(14, pad_top);
    pd.set(16, pad_bottom);
    pd.set(18, pad_value);
    pd.set(24, pad_left);
    pd.set(17, pad_front);
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
} //namespace ncnn
