// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convolution_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector
#include "cpu.h"
#include "riscv_activation.h"
#include "riscv_usability.h"

namespace ncnn {

#if NCNN_ZFH
#include "convolution_packed_fp16s.h"
#include "convolution_im2col_gemm_fp16s.h"
#if __riscv_zvfh
#include "convolution_3x3_winograd_fp16s.h"
#include "convolution_3x3_pack1ton_fp16s.h"
#include "convolution_7x7_pack1ton_fp16s.h"
#endif
#endif // NCNN_ZFH

#if NCNN_ZFH
int Convolution_riscv::create_pipeline_fp16s(const Option& opt)
{
#if __riscv_zvfh
    const int packn = csrr_vlenb() / 2;
#endif // __riscv_zvfh

    const int maxk = kernel_w * kernel_h;
    const int num_input = weight_data_size / maxk / num_output;

    int elempack = 1;
    int out_elempack = 1;

#if __riscv_zvfh
    if (opt.use_packing_layout)
    {
        elempack = num_input % packn == 0 ? packn : 1;
        out_elempack = num_output % packn == 0 ? packn : 1;
    }
#endif // __riscv_zvfh

    bool prefer_sgemm = opt.use_sgemm_convolution;
#if __riscv_zvfh
    if (elempack == 1 && out_elempack == packn
            && ((kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
                || (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
                || (kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)))
    {
        prefer_sgemm = false;
    }
#endif // __riscv_zvfh

#if __riscv_zvfh
    // packn
    if (elempack == packn && out_elempack == packn)
    {
        if (opt.use_winograd_convolution && (opt.use_winograd23_convolution || opt.use_winograd43_convolution || opt.use_winograd63_convolution) && opt.use_fp16_arithmetic && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            if ((opt.use_winograd63_convolution && num_input >= packn * 2 && num_output >= packn * 2 && num_input <= packn * 16 && num_output <= packn * 16) || (!opt.use_winograd43_convolution && !opt.use_winograd23_convolution))
                conv3x3s1_winograd63_transform_kernel_fp16sa_rvv(weight_data, weight_winograd63_data, num_input, num_output, opt);
            else if ((opt.use_winograd43_convolution && num_input >= packn * 2 && num_output >= packn * 2) || (!opt.use_winograd63_convolution && !opt.use_winograd23_convolution))
                conv3x3s1_winograd43_transform_kernel_fp16sa_rvv(weight_data, weight_winograd43_data, num_input, num_output, opt);
            else // if (opt.use_winograd23_convolution)
                conv3x3s1_winograd23_transform_kernel_fp16sa_rvv(weight_data, weight_winograd23_data, num_input, num_output, opt);
        }
        else if (opt.use_fp16_arithmetic && (prefer_sgemm || (kernel_w == 1 && kernel_h == 1)))
        {
            convolution_im2col_gemm_transform_kernel_fp16sa_rvv(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h, opt);
        }
        else
        {
            convolution_transform_kernel_packed_fp16s_rvv(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h);
        }
    }

    // pack1ton
    if (elempack == 1 && out_elempack == packn)
    {
        if (opt.use_fp16_arithmetic && (prefer_sgemm || (kernel_w == 1 && kernel_h == 1)))
        {
            convolution_im2col_gemm_transform_kernel_fp16sa_rvv(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h, opt);
        }
        else if (opt.use_fp16_arithmetic
                 && ((kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
                     || (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
                     || (kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)))
        {
            convolution_transform_kernel_packed_fp16s_simple_rvv(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
        }
        else
        {
            convolution_transform_kernel_packed_fp16s_rvv(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h);
        }
    }

    // packnto1
    if (elempack == packn && out_elempack == 1)
    {
        if (opt.use_fp16_arithmetic && (prefer_sgemm || (kernel_w == 1 && kernel_h == 1)))
        {
            convolution_im2col_gemm_transform_kernel_fp16sa_rvv(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h, opt);
        }
        else
        {
            convolution_transform_kernel_packed_fp16s_rvv(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h);
        }
    }
#endif // __riscv_zvfh

    // pack1
    if (elempack == 1 && out_elempack == 1)
    {
        if (opt.use_fp16_arithmetic && (prefer_sgemm || (kernel_w == 1 && kernel_h == 1)))
        {
            convolution_im2col_gemm_transform_kernel_fp16sa_rvv(weight_data, weight_sgemm_data, num_input, num_output, kernel_w, kernel_h, opt);
        }
        else
        {
            convolution_transform_kernel_packed_fp16s_rvv(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h);
        }
    }

    if (opt.use_fp16_arithmetic)
    {
        ncnn::cast_float32_to_float16(bias_data, bias_data_fp16, opt);
    }

    if (opt.lightmode)
        weight_data.release();

    return 0;
}

int Convolution_riscv::forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if __riscv_zvfh
    const int packn = csrr_vlenb() / 2;
#endif // __riscv_zvfh

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
    int out_elempack = 1;
#if __riscv_zvfh
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % packn == 0 ? packn : 1;
    }
#endif // __riscv_zvfh
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

#if __riscv_zvfh
    if (elempack == packn && out_elempack == packn)
    {
        {
            convolution_packed_fp16s_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }
    }

    if (elempack == 1 && out_elempack == packn)
    {
        {
            convolution_packed_fp16s_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }
    }

    if (elempack == packn && out_elempack == 1)
    {
        {
            convolution_packed_fp16s_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }
    }
#endif // __riscv_zvfh

    if (elempack == 1 && out_elempack == 1)
    {
        {
            convolution_packed_fp16s_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }
    }

    return 0;
}

int Convolution_riscv::forward_fp16sa(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if __riscv_zvfh
    const int packn = csrr_vlenb() / 2;
#endif // __riscv_zvfh

    int w = bottom_blob.w;
    int h = bottom_blob.h;
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
    int out_elempack = 1;
#if __riscv_zvfh
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % packn == 0 ? packn : 1;
    }
#endif // __riscv_zvfh
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

#if __riscv_zvfh
    const int num_input = bottom_blob.c * elempack;

    if (elempack == packn && out_elempack == packn && opt.use_winograd_convolution && (opt.use_winograd23_convolution || opt.use_winograd43_convolution || opt.use_winograd63_convolution) && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        int ret = 0;
        if ((opt.use_winograd63_convolution && num_input >= packn * 2 && num_output >= packn * 2 && num_input <= packn * 16 && num_output <= packn * 16) || (!opt.use_winograd43_convolution && !opt.use_winograd23_convolution))
            ret = conv3x3s1_winograd63_fp16sa_rvv(bottom_blob_bordered, top_blob, weight_winograd63_data, bias_data_fp16, nT, opt);
        else if ((opt.use_winograd43_convolution && num_input >= packn * 2 && num_output >= packn * 2) || (!opt.use_winograd63_convolution && !opt.use_winograd23_convolution))
            ret = conv3x3s1_winograd43_fp16sa_rvv(bottom_blob_bordered, top_blob, weight_winograd43_data, bias_data_fp16, nT, opt);
        else // if (opt.use_winograd23_convolution)
            ret = conv3x3s1_winograd23_fp16sa_rvv(bottom_blob_bordered, top_blob, weight_winograd23_data, bias_data_fp16, nT, opt);
        if (ret != 0)
            return ret;

        if (activation)
        {
            activation->forward_inplace(top_blob, opt);
        }

        return 0;
    }
#endif // __riscv_zvfh

    bool prefer_sgemm = opt.use_sgemm_convolution;
#if __riscv_zvfh
    if (elempack == 1 && out_elempack == packn
            && ((kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
                || (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
                || (kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)))
    {
        prefer_sgemm = false;
    }
#endif // __riscv_zvfh

    if (prefer_sgemm || (kernel_w == 1 && kernel_h == 1))
    {
        int _nT = nT ? nT : opt.num_threads;
        if (nT != 0 && opt.num_threads != nT)
        {
            // force num_threads the same as in create_pipeline
            // so we could use pre-packed A/B from the same tile config
            NCNN_LOGE("opt.num_threads %d changed, convolution gemm will use load-time value %d", opt.num_threads, nT);
        }

        int ret = convolution_im2col_gemm_fp16sa_rvv(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data_fp16, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, _nT, opt);
        if (ret != 0)
            return ret;

        if (activation)
        {
            activation->forward_inplace(top_blob, opt);
        }

        return 0;
    }

#if __riscv_zvfh
    if (elempack == packn && out_elempack == packn)
    {
        convolution_packed_fp16sa_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
    }

    if (elempack == 1 && out_elempack == packn)
    {
        if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
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
        else
        {
            convolution_packed_fp16sa_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }
    }

    if (elempack == packn && out_elempack == 1)
    {
        convolution_packed_fp16sa_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
    }
#endif // __riscv_zvfh

    if (elempack == 1 && out_elempack == 1)
    {
#if __riscv_zvfh
        convolution_packed_fp16sa_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
#else
        convolution_packed_fp16s_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
#endif
    }

    return 0;
}
#endif // NCNN_ZFH

} // namespace ncnn
