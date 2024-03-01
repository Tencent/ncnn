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

#include "convolution_riscv.h"

#include "benchmark.h"
#include "cpu.h"
#include "layer_type.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_activation.h"
#include "riscv_usability.h"

#include "cpu.h"

namespace ncnn {

#include "convolution_sgemm.h"
#include "convolution_winograd_transform.h"
#include "convolution_winograd_dot.h"
#include "convolution_1x1.h"
#include "convolution_3x3.h"

#if __riscv_vector
#include "convolution_packn.h"
#include "convolution_pack1ton.h"
#include "convolution_packnto1.h"

#include "convolution_sgemm_packn.h"
#include "convolution_sgemm_pack1ton.h"
#include "convolution_sgemm_packnto1.h"
#include "convolution_winograd_transform_packn.h"
#include "convolution_winograd_dot_packn.h"
#include "convolution_1x1_packn.h"
#include "convolution_1x1_pack1ton.h"
#include "convolution_1x1_packnto1.h"
#include "convolution_3x3_packn.h"
#include "convolution_3x3_pack1ton.h"
#include "convolution_7x7_pack1ton.h"

#if __riscv_zfh
#include "convolution_fp16s.h"
#include "convolution_packn_fp16s.h"
#include "convolution_pack1ton_fp16s.h"
#include "convolution_packnto1_fp16s.h"

#include "convolution_sgemm_fp16s.h"
#include "convolution_sgemm_packn_fp16s.h"
#include "convolution_sgemm_pack1ton_fp16s.h"
#include "convolution_sgemm_packnto1_fp16s.h"
#include "convolution_winograd_transform_packn_fp16s.h"
#include "convolution_winograd_dot_packn_fp16s.h"
#include "convolution_1x1_fp16s.h"
#include "convolution_1x1_packn_fp16s.h"
#include "convolution_1x1_pack1ton_fp16s.h"
#include "convolution_1x1_packnto1_fp16s.h"
#include "convolution_3x3_packn_fp16s.h"
#include "convolution_3x3_pack1ton_fp16s.h"
#include "convolution_7x7_pack1ton_fp16s.h"

#endif
#endif // __riscv_vector

Convolution_riscv::Convolution_riscv()
{
#if __riscv_vector
    support_packing = true;
#if __riscv_zfh
    support_fp16_storage = true;
#endif
#endif // __riscv_vector

    activation = 0;
}

static void convolution_transform_kernel_packed_rvv(const Mat& weight_data, Mat& weight_data_tm, int num_input, int num_output, int kernel_w, int kernel_h, int elempack, int out_elempack)
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

int Convolution_riscv::create_pipeline(const Option& opt)
{
    if (dynamic_weight)
        return 0;

    activation = create_activation_layer(activation_type, activation_params, opt);

#if NCNN_INT8
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        // TODO implement int8
        return 0;
    }
#endif

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
    const int num_input = weight_data_size / maxk / num_output;

    int elempack = 1;
    int out_elempack = 1;
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        elempack = num_input % packn == 0 ? packn : 1;
        out_elempack = num_output % packn == 0 ? packn : 1;
    }
#endif

#if __riscv_vector
    // packn
    if (elempack == packn && out_elempack == packn)
    {
        if (opt.use_winograd_convolution && (opt.use_winograd23_convolution || opt.use_winograd43_convolution || opt.use_winograd63_convolution) && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            if ((opt.use_winograd63_convolution && num_input >= packn * 2 && num_output >= packn * 2 && num_input <= packn * 16 && num_output <= packn * 16) || (!opt.use_winograd43_convolution && !opt.use_winograd23_convolution))
                conv3x3s1_winograd63_transform_kernel_packn_rvv(weight_data, weight_winograd63_data, num_input, num_output, opt);
            else if ((opt.use_winograd43_convolution && num_input >= packn * 2 && num_output >= packn * 2) || (!opt.use_winograd63_convolution && !opt.use_winograd23_convolution))
                conv3x3s1_winograd43_transform_kernel_packn_rvv(weight_data, weight_winograd43_data, num_input, num_output, opt);
            else // if (opt.use_winograd23_convolution)
                conv3x3s1_winograd23_transform_kernel_packn_rvv(weight_data, weight_winograd23_data, num_input, num_output, opt);
        }
        else
        {
            convolution_transform_kernel_packed_rvv(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
        }
    }

    // pack1ton
    if (elempack == 1 && out_elempack == packn)
    {
        convolution_transform_kernel_packed_rvv(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
    }

    // packnto1
    if (elempack == packn && out_elempack == 1)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            convolution_im2col_sgemm_transform_kernel_packnto1_rvv(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h);
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            convolution_im2col_sgemm_transform_kernel_packnto1_rvv(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h);
        }
        else if (opt.use_sgemm_convolution)
        {
            convolution_im2col_sgemm_transform_kernel_packnto1_rvv(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h);
        }
        else
        {
            convolution_transform_kernel_packed_rvv(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
        }
    }
#endif // __riscv_vector

    // pack1
    if (elempack == 1 && out_elempack == 1)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            convolution_im2col_sgemm_transform_kernel_rvv(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h);
        }
        else if (opt.use_winograd_convolution && (opt.use_winograd23_convolution || opt.use_winograd43_convolution) && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            if ((opt.use_winograd43_convolution && num_input >= 16 && num_output >= 16) || !opt.use_winograd23_convolution)
            {
                conv3x3s1_winograd43_transform_kernel_rvv(weight_data, weight_winograd43_data, num_input, num_output, opt);
            }
            else if (opt.use_winograd23_convolution)
            {
                conv3x3s1_winograd23_transform_kernel_rvv(weight_data, weight_winograd23_data, num_input, num_output, opt);
            }
        }
        else if (opt.use_sgemm_convolution)
        {
            convolution_im2col_sgemm_transform_kernel_rvv(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h);
        }
        else
        {
            weight_data_tm = weight_data;
        }
    }

    weight_data.release();

    return 0;
}

int Convolution_riscv::destroy_pipeline(const Option& opt)
{
    if (activation)
    {
        activation->destroy_pipeline(opt);
        delete activation;
        activation = 0;
    }

    return 0;
}

int Convolution_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if NCNN_INT8
    if (opt.use_int8_inference && int8_scale_term)
    {
        Mat bottom_blob_unpacked = bottom_blob;
        if (bottom_blob.elempack != 1)
        {
            Option opt_pack1 = opt;
            opt_pack1.blob_allocator = opt.workspace_allocator;

            convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack1);
        }

        Mat bottom_blob_unpacked_fp32 = bottom_blob_unpacked;
        if (bottom_blob_unpacked.elembits() == 16)
        {
            Option opt_pack1 = opt;
            opt_pack1.blob_allocator = opt.workspace_allocator;

            cast_float16_to_float32(bottom_blob_unpacked, bottom_blob_unpacked_fp32, opt_pack1);
        }

        Option opt_unpacked = opt;
        opt_unpacked.use_packing_layout = false;
        return Convolution::forward_int8(bottom_blob_unpacked_fp32, top_blob, opt_unpacked);
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
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % packn == 0 ? packn : 1;
    }
#endif
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const int num_input = channels * elempack;

#if __riscv_vector
    if (elempack == packn && out_elempack == packn)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_packn_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv1x1s2_sgemm_packn_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (opt.use_winograd_convolution && (opt.use_winograd23_convolution || opt.use_winograd43_convolution || opt.use_winograd63_convolution) && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            if ((opt.use_winograd63_convolution && num_input >= packn * 2 && num_output >= packn * 2 && num_input <= packn * 16 && num_output <= packn * 16) || (!opt.use_winograd43_convolution && !opt.use_winograd23_convolution))
                conv3x3s1_winograd63_packn_rvv(bottom_blob_bordered, top_blob, weight_winograd63_data, bias_data, opt);
            else if ((opt.use_winograd43_convolution && num_input >= packn * 2 && num_output >= packn * 2) || (!opt.use_winograd63_convolution && !opt.use_winograd23_convolution))
                conv3x3s1_winograd43_packn_rvv(bottom_blob_bordered, top_blob, weight_winograd43_data, bias_data, opt);
            else // if (opt.use_winograd23_convolution)
                conv3x3s1_winograd23_packn_rvv(bottom_blob_bordered, top_blob, weight_winograd23_data, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (opt.use_sgemm_convolution)
        {
            convolution_im2col_sgemm_packn_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            convolution_packn_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }
    }

    if (elempack == 1 && out_elempack == packn)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_pack1ton_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv3x3s1_pack1ton_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv3x3s2_pack1ton_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv7x7s2_pack1ton_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (opt.use_sgemm_convolution)
        {
            convolution_im2col_sgemm_pack1ton_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            convolution_pack1ton_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }
    }

    if (elempack == packn && out_elempack == 1)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_packnto1_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv1x1s2_sgemm_packnto1_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (opt.use_sgemm_convolution)
        {
            convolution_im2col_sgemm_packnto1_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            convolution_packnto1_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }
    }
#endif // __riscv_vector

    if (elempack == 1 && out_elempack == 1)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (opt.use_winograd_convolution && (opt.use_winograd43_convolution || opt.use_winograd23_convolution) && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            if ((opt.use_winograd43_convolution && num_input >= 16 && num_output >= 16) || !opt.use_winograd23_convolution)
            {
                conv3x3s1_winograd43_rvv(bottom_blob_bordered, top_blob, weight_winograd43_data, bias_data, opt);
            }
            else if (opt.use_winograd23_convolution)
            {
                conv3x3s1_winograd23_rvv(bottom_blob_bordered, top_blob, weight_winograd23_data, bias_data, opt);
            }

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (opt.use_sgemm_convolution)
        {
            convolution_im2col_sgemm_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
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

            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output; p++)
            {
                float* outptr = top_blob.channel(p);

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
                            const Mat m = bottom_blob_bordered.channel(q);
                            const float* sptr = m.row(i * stride_h) + j * stride_w;

                            for (int k = 0; k < maxk; k++)
                            {
                                float val = sptr[space_ofs[k]];
                                float wt = kptr[k];
                                sum += val * wt;
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

    return 0;
}

int Convolution_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
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

#if NCNN_RVV
    if (opt.use_fp16_storage && cpu_support_riscv_v() && cpu_support_riscv_zfh() && weight_data_flattened.elembits() == 16)
    {
        Mat weight_data_flattened_fp32;
        cast_float16_to_float32(weight_data_flattened, weight_data_flattened_fp32, opt);
        weight_data_flattened = weight_data_flattened_fp32;
    }
#endif // NCNN_RVV

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

#if NCNN_RVV
        if (opt.use_fp16_storage && cpu_support_riscv_v() && cpu_support_riscv_zfh() && bias_data_flattened.elembits() == 16)
        {
            Mat bias_data_flattened_fp32;
            cast_float16_to_float32(bias_data_flattened, bias_data_flattened_fp32, opt);
            bias_data_flattened = bias_data_flattened_fp32;
        }
#endif // NCNN_RVV

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

#if __riscv_vector && __riscv_zfh
static void convolution_transform_kernel_packed_fp16s_rvv(const Mat& weight_data, Mat& weight_data_tm, int num_input, int num_output, int kernel_w, int kernel_h, int elempack, int out_elempack)
{
    const int maxk = kernel_w * kernel_h;

    // src = kw-kh-inch-outch
    // dst = pb-pa-kw-kh-inch/pa-outch/pb
    {
        Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

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
}

int Convolution_riscv::create_pipeline_fp16s(const Option& opt)
{
    const int packn = csrr_vlenb() / 2;

    const int maxk = kernel_w * kernel_h;
    const int num_input = weight_data_size / maxk / num_output;

    int elempack = 1;
    int out_elempack = 1;

    if (opt.use_packing_layout)
    {
        elempack = num_input % packn == 0 ? packn : 1;
        out_elempack = num_output % packn == 0 ? packn : 1;
    }

    // packn
    if (elempack == packn && out_elempack == packn)
    {
        if (opt.use_winograd_convolution && (opt.use_winograd23_convolution || opt.use_winograd43_convolution || opt.use_winograd63_convolution) && opt.use_fp16_arithmetic && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            if ((opt.use_winograd63_convolution && num_input >= packn * 2 && num_output >= packn * 2 && num_input <= packn * 16 && num_output <= packn * 16) || (!opt.use_winograd43_convolution && !opt.use_winograd23_convolution))
                conv3x3s1_winograd63_transform_kernel_packn_fp16sa_rvv(weight_data, weight_winograd63_data, num_input, num_output, opt);
            else if ((opt.use_winograd43_convolution && num_input >= packn * 2 && num_output >= packn * 2) || (!opt.use_winograd63_convolution && !opt.use_winograd23_convolution))
                conv3x3s1_winograd43_transform_kernel_packn_fp16sa_rvv(weight_data, weight_winograd43_data, num_input, num_output, opt);
            else // if (opt.use_winograd23_convolution)
                conv3x3s1_winograd23_transform_kernel_packn_fp16sa_rvv(weight_data, weight_winograd23_data, num_input, num_output, opt);
        }
        else
        {
            convolution_transform_kernel_packed_fp16s_rvv(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
        }
    }

    // pack1ton
    if (elempack == 1 && out_elempack == packn)
    {
        convolution_transform_kernel_packed_fp16s_rvv(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
    }

    // packnto1
    if (elempack == packn && out_elempack == 1)
    {
        if (opt.use_fp16_arithmetic && kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            convolution_im2col_sgemm_transform_kernel_packnto1_fp16sa_rvv(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h);
        }
        else if (opt.use_fp16_arithmetic && kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            convolution_im2col_sgemm_transform_kernel_packnto1_fp16sa_rvv(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h);
        }
        else if (opt.use_fp16_arithmetic && opt.use_sgemm_convolution)
        {
            convolution_im2col_sgemm_transform_kernel_packnto1_fp16sa_rvv(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h);
        }
        else
        {
            convolution_transform_kernel_packed_fp16s_rvv(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
        }
    }

    // pack1
    if (elempack == 1 && out_elempack == 1)
    {
        if (opt.use_fp16_arithmetic && kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            convolution_im2col_sgemm_transform_kernel_fp16sa_rvv(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h);
        }
        else if (opt.use_fp16_arithmetic && opt.use_sgemm_convolution)
        {
            convolution_im2col_sgemm_transform_kernel_fp16sa_rvv(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h);
        }
        else
        {
            convolution_transform_kernel_packed_fp16s_rvv(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
        }
    }

    if (opt.use_fp16_arithmetic)
    {
        ncnn::cast_float32_to_float16(bias_data, bias_data_fp16, opt);
    }

    weight_data.release();

    return 0;
}

int Convolution_riscv::forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int packn = csrr_vlenb() / 2;

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    //     NCNN_LOGE("Convolution forward_fp16s input %d x %d  pad = %d %d  ksize=%d %d  stride=%d %d", w, h, pad_left, pad_top, kernel_w, kernel_h, stride_w, stride_h);

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
    int out_elempack = (opt.use_packing_layout && num_output % packn == 0) ? packn : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (elempack == packn && out_elempack == packn)
    {
        {
            convolution_packn_fp16s_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }
    }

    if (elempack == 1 && out_elempack == packn)
    {
        {
            convolution_pack1ton_fp16s_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }
    }

    if (elempack == packn && out_elempack == 1)
    {
        {
            convolution_packnto1_fp16s_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }
    }

    if (elempack == 1 && out_elempack == 1)
    {
        {
            convolution_fp16s(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }
    }

    return 0;
}

int Convolution_riscv::forward_fp16sa(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int packn = csrr_vlenb() / 2;

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    // NCNN_LOGE("Convolution input %d x %d  pad = %d %d  ksize=%d %d  stride=%d %d", w, h, pad_w, pad_h, kernel_w, kernel_h, stride_w, stride_h);

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
    int out_elempack = (opt.use_packing_layout && num_output % packn == 0) ? packn : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const int num_input = channels * elempack;

    if (elempack == packn && out_elempack == packn)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_packn_fp16sa_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv1x1s2_sgemm_packn_fp16sa_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (opt.use_winograd_convolution && (opt.use_winograd23_convolution || opt.use_winograd43_convolution || opt.use_winograd63_convolution) && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            if ((opt.use_winograd63_convolution && num_input >= packn * 2 && num_output >= packn * 2 && num_input <= packn * 16 && num_output <= packn * 16) || (!opt.use_winograd43_convolution && !opt.use_winograd23_convolution))
                conv3x3s1_winograd63_packn_fp16sa_rvv(bottom_blob_bordered, top_blob, weight_winograd63_data, bias_data_fp16, opt);
            else if ((opt.use_winograd43_convolution && num_input >= packn * 2 && num_output >= packn * 2) || (!opt.use_winograd63_convolution && !opt.use_winograd23_convolution))
                conv3x3s1_winograd43_packn_fp16sa_rvv(bottom_blob_bordered, top_blob, weight_winograd43_data, bias_data_fp16, opt);
            else // if (opt.use_winograd23_convolution)
                conv3x3s1_winograd23_packn_fp16sa_rvv(bottom_blob_bordered, top_blob, weight_winograd23_data, bias_data_fp16, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (opt.use_sgemm_convolution)
        {
            convolution_im2col_sgemm_packn_fp16sa_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            convolution_packn_fp16sa_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }
    }

    if (elempack == 1 && out_elempack == packn)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_pack1ton_fp16sa_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv3x3s1_pack1ton_fp16sa_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv3x3s2_pack1ton_fp16sa_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv7x7s2_pack1ton_fp16sa_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (opt.use_sgemm_convolution)
        {
            convolution_im2col_sgemm_pack1ton_fp16sa_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            convolution_pack1ton_fp16sa_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }
    }

    if (elempack == packn && out_elempack == 1)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_packnto1_fp16sa_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv1x1s2_sgemm_packnto1_fp16sa_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (opt.use_sgemm_convolution)
        {
            convolution_im2col_sgemm_packnto1_fp16sa_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            convolution_packnto1_fp16sa_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }
    }

    if (elempack == 1 && out_elempack == 1)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_fp16sa_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (opt.use_sgemm_convolution)
        {
            convolution_im2col_sgemm_fp16sa_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            convolution_fp16s(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }
    }

    return 0;
}
#endif // __riscv_vector && __riscv_zfh

} // namespace ncnn
