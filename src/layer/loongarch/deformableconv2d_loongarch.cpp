// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "deformableconv2d_loongarch.h"

#if __loongarch_sx
#include <lsxintrin.h>
#if __loongarch_asx
#include <lasxintrin.h>
#endif // __loongarch_asx
#endif // __loongarch_sx
#include "loongarch_activation.h"
#include "loongarch_usability.h"

#include "benchmark.h"
#include "cpu.h"
#include "layer_type.h"

namespace ncnn {

#include "deformableconv2d_packed.h"

DeformableConv2D_loongarch::DeformableConv2D_loongarch()
{
#if __loongarch_sx
    support_packing = true;
#endif // __loongarch_sx
#if NCNN_BF16
    support_bf16_storage = true;
#endif

    activation = 0;
}

static int _4Dindex_to_1Dindex(int i0, int i1, int i2, int i3, int l1, int l2, int l3)
{
    return ((i0 * l1 + i1) * l2 + i2) * l3 + i3;
}

static int _6Dindex_to_1Dindex(int i0, int i1, int i2, int i3, int i4, int i5, int l1, int l2, int l3, int l4, int l5)
{
    return ((((i0 * l1 + i1) * l2 + i2) * l3 + i3) * l4 + i4) * l5 + i5;
}

static void deformableconv2d_transform_kernel_packed_lsx(const Mat& weight_data, Mat& weight_data_tm, int num_input, int num_output, int kernel_w, int kernel_h, int elempack, int out_elempack)
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

int DeformableConv2D_loongarch::create_pipeline(const Option& opt)
{
    activation = create_activation_layer(activation_type, activation_params, opt);

    int kernel_size = kernel_w * kernel_h;
    int num_input = weight_data_size / kernel_size / num_output;

    int elempack = 1;
    int out_elempack = 1;

#if __loongarch_sx
    if (opt.use_packing_layout)
    {
#if __loongarch_asx
        elempack = num_input % 8 == 0 ? 8 : num_input % 4 == 0 ? 4 : 1;
        out_elempack = num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
#else
        elempack = num_input % 4 == 0 ? 4 : 1;
        out_elempack = num_output % 4 == 0 ? 4 : 1;
#endif
    }
#endif // __loongarch_sx

    deformableconv2d_transform_kernel_packed_lsx(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);

    if (opt.lightmode)
        weight_data.release();

    return 0;
}

int DeformableConv2D_loongarch::destroy_pipeline(const Option& opt)
{
    if (activation)
    {
        activation->destroy_pipeline(opt);
        delete activation;
        activation = 0;
    }

    return 0;
}

int DeformableConv2D_loongarch::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& offset = bottom_blobs[1];
    const bool has_mask = (bottom_blobs.size() == 3);
    Mat& top_blob = top_blobs[0];

    const int elembits = bottom_blob.elembits();

#if NCNN_BF16
    if (elembits == 16)
    {
        Option opt_fp32 = opt;
        opt_fp32.use_bf16_storage = false;

        std::vector<Mat> bottom_blobs_fp32(bottom_blobs.size());
        for (size_t i = 0; i < bottom_blobs.size(); i++)
        {
            Option opt_cast = opt;
            opt_cast.blob_allocator = opt.workspace_allocator;
            cast_bfloat16_to_float32(bottom_blobs[i], bottom_blobs_fp32[i], opt_cast);
            if (bottom_blobs_fp32[i].empty())
                return -100;
        }

        std::vector<Mat> top_blobs_fp32(1);
        int ret = forward(bottom_blobs_fp32, top_blobs_fp32, opt_fp32);
        if (ret != 0)
            return ret;

        cast_float32_to_bfloat16(top_blobs_fp32[0], top_blob, opt);
        if (top_blob.empty())
            return -100;

        return 0;
    }
#endif // NCNN_BF16

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
#if __loongarch_sx
    if (opt.use_packing_layout)
    {
#if __loongarch_asx
        out_elempack = num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
#else
        out_elempack = num_output % 4 == 0 ? 4 : 1;
#endif
    }
#endif // __loongarch_sx
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    deformableconv2d_packed(bottom_blobs, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, pad_left, pad_top, activation_type, activation_params, opt);

    return 0;
}

} // namespace ncnn
