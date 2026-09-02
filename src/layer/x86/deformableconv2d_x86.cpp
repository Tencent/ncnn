// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "deformableconv2d_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __SSE4_1__
#include <smmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif
#endif // __SSE4_1__
#endif // __SSE2__
#include "x86_activation.h"
#include "x86_usability.h"

#include "benchmark.h"
#include "cpu.h"
#include "layer_type.h"

namespace ncnn {

#include "deformableconv2d_im2col.h"
#include "deformableconv2d_packed.h"

DeformableConv2D_x86::DeformableConv2D_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__

    activation = 0;
    gemm = 0;
}

static int _4Dindex_to_1Dindex(int i0, int i1, int i2, int i3, int l1, int l2, int l3)
{
    return ((i0 * l1 + i1) * l2 + i2) * l3 + i3;
}

static int _6Dindex_to_1Dindex(int i0, int i1, int i2, int i3, int i4, int i5, int l1, int l2, int l3, int l4, int l5)
{
    return ((((i0 * l1 + i1) * l2 + i2) * l3 + i3) * l4 + i4) * l5 + i5;
}

static void deformableconv2d_transform_kernel_packed(const Mat& weight_data, Mat& weight_data_tm, int num_input, int num_output, int kernel_w, int kernel_h, int elempack, int out_elempack)
{
    const int maxk = kernel_w * kernel_h;

    // src = kw-kh-inch-outch
    // dst = pb-pa-inch/pa-kw-kh-outch/pb
    {
        const float* weight_ptr = weight_data;

        weight_data_tm.create(num_input * maxk * num_output / (elempack * out_elempack), (size_t)4u * elempack * out_elempack, elempack * out_elempack);
        float* ptr = weight_data_tm;
        for (int oc = 0; oc < num_output; oc++)
        {
            for (int i = 0; i < kernel_h; i++)
            {
                for (int j = 0; j < kernel_w; j++)
                {
                    for (int ic = 0; ic < num_input; ic++)
                    {
                        ptr[_6Dindex_to_1Dindex(oc / out_elempack, i, j, ic / elempack, ic % elempack, oc % out_elempack, kernel_h, kernel_w, num_input / elempack, elempack, out_elempack)] = weight_ptr[_4Dindex_to_1Dindex(oc, ic, i, j, num_input, kernel_h, kernel_w)];
                    }
                }
            }
        }
        weight_data_tm = weight_data_tm.reshape(num_input / elempack, maxk, num_output / out_elempack);
    }
}

int DeformableConv2D_x86::create_pipeline(const Option& opt)
{
    activation = create_activation_layer(activation_type, activation_params, opt);

    int kernel_size = kernel_w * kernel_h;
    int num_input = weight_data_size / kernel_size / num_output;

    int elempack = 1;
    int out_elempack = 1;

#if __SSE2__
    if (opt.use_packing_layout)
    {
#if __AVX512F__
        elempack = num_input % 16 == 0 ? 16 : num_input % 8 == 0 ? 8 : num_input % 4 == 0 ? 4 : 1;
        out_elempack = num_output % 16 == 0 ? 16 : num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
#elif __AVX__
        elempack = num_input % 8 == 0 ? 8 : num_input % 4 == 0 ? 4 : 1;
        out_elempack = num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
#else
        elempack = num_input % 4 == 0 ? 4 : 1;
        out_elempack = num_output % 4 == 0 ? 4 : 1;
#endif
    }
#endif // __SSE2__

    if (opt.use_sgemm_convolution)
    {
        const int maxk = kernel_w * kernel_h;

        gemm = ncnn::create_layer_cpu(ncnn::LayerType::Gemm);

        ncnn::ParamDict pd;
        pd.set(2, 0);                   // transA
        pd.set(3, 0);                   // transB
        pd.set(4, 1);                   // constantA
        pd.set(5, 0);                   // constantB
        pd.set(6, 1);                   // constantC
        pd.set(7, num_output);          // M = outch
        pd.set(8, 0);                   // N = size
        pd.set(9, maxk * num_input);    // K = maxk*inch
        pd.set(10, bias_term ? 1 : -1); // constant_broadcast_type_C = (M)
        pd.set(11, 1);                  // output_N1M

        gemm->load_param(pd);

        // maxk-inch-outch to pa-maxk-inch/pa-outch
        Mat tmp;
        {
            Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

            tmp.create(maxk * num_input, num_output);

            for (int q = 0; q < num_output; q += 1)
            {
                float* g00 = tmp.row(q);

                for (int p = 0; p + (elempack - 1) < num_input; p += elempack)
                {
                    for (int k = 0; k < maxk; k++)
                    {
                        for (int i = 0; i < elempack; i++)
                        {
                            const float* k00 = weight_data_r2.channel(q).row(p + i);
                            g00[0] = k00[k];
                            g00++;
                        }
                    }
                }
            }
        }

        if (bias_term)
        {
            ncnn::Mat weights[2];
            weights[0] = tmp;
            weights[1] = bias_data;

            gemm->load_model(ModelBinFromMatArray(weights));
        }
        else
        {
            ncnn::Mat weights[1];
            weights[0] = tmp;

            gemm->load_model(ModelBinFromMatArray(weights));
        }

        gemm->create_pipeline(opt);
    }
    else
    {
        deformableconv2d_transform_kernel_packed(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
    }

    if (opt.lightmode)
        weight_data.release();

    return 0;
}

int DeformableConv2D_x86::destroy_pipeline(const Option& opt)
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

int DeformableConv2D_x86::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& offset = bottom_blobs[1];
    const bool has_mask = (bottom_blobs.size() == 3);
    Mat& top_blob = top_blobs[0];

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;
    const int outw = (w + pad_left + pad_right - kernel_extent_w) / stride_w + 1;
    const int outh = (h + pad_top + pad_bottom - kernel_extent_h) / stride_h + 1;

    int out_elempack = 1;
#if __SSE2__
    if (opt.use_packing_layout)
    {
#if __AVX512F__
        out_elempack = num_output % 16 == 0 ? 16 : num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
#elif __AVX__
        out_elempack = num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
#else
        out_elempack = num_output % 4 == 0 ? 4 : 1;
#endif
    }
#endif // __SSE2__
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (opt.use_sgemm_convolution)
    {
        const int size = outw * outh;
        const int maxk = kernel_w * kernel_h;

        Mat offset_unpacked;
        convert_packing(offset, offset_unpacked, 1, opt);

        Mat mask_unpacked;
        if (has_mask)
        {
            const Mat& mask = bottom_blobs[2];
            convert_packing(mask, mask_unpacked, 1, opt);
        }

        // im2col
        Mat bottom_im2col(size, maxk * channels, elemsize, elempack, opt.workspace_allocator);
        deformableconv2d_im2col(bottom_blob, offset_unpacked, mask_unpacked, bottom_im2col, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, pad_left, pad_top, outw, outh, has_mask, opt);

        // sgemm
        {
            top_blob.w = outw * outh;
            top_blob.h = 1;
        }
        Option opt_b = opt;
        opt_b.blob_allocator = opt.workspace_allocator;
        gemm->forward(bottom_im2col, top_blob, opt_b);
        {
            top_blob.w = outw;
            top_blob.h = outh;
        }

        if (activation)
        {
            activation->forward_inplace(top_blob, opt);
        }

        return 0;
    }

    deformableconv2d_packed(bottom_blobs, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, pad_left, pad_top, activation_type, activation_params, opt);

    return 0;
}

} // namespace ncnn
