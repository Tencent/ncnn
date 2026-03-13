// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "deformableconv2d_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_activation.h"
#include "riscv_usability.h"

#include "benchmark.h"
#include "cpu.h"
#include "layer_type.h"

namespace ncnn {

#if __riscv_vector
#include "deformableconv2d_packn.h"
#include "deformableconv2d_pack1ton.h"
#include "deformableconv2d_packnto1.h"
#endif // __riscv_vector

DeformableConv2D_riscv::DeformableConv2D_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector

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

#if __riscv_vector
static void deformableconv2d_transform_kernel_packed_riscv(const Mat& weight_data, Mat& weight_data_tm, int num_input, int num_output, int kernel_w, int kernel_h, int elempack, int out_elempack)
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
#endif // __riscv_vector

int DeformableConv2D_riscv::create_pipeline(const Option& opt)
{
    activation = create_activation_layer(activation_type, activation_params, opt);

    int kernel_size = kernel_w * kernel_h;
    int num_input = weight_data_size / kernel_size / num_output;

    int elempack = 1;
    int out_elempack = 1;

#if __riscv_vector
    if (opt.use_packing_layout)
    {
        const int packn = csrr_vlenb() / 4;
        elempack = num_input % packn == 0 ? packn : 1;
        out_elempack = num_output % packn == 0 ? packn : 1;
    }
#endif // __riscv_vector

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
    else if (elempack == 1 && out_elempack == 1)
    {
        weight_data_tm = weight_data;
    }
    else
    {
#if __riscv_vector
        deformableconv2d_transform_kernel_packed_riscv(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
#endif // __riscv_vector
    }

    if (opt.lightmode)
    {
        weight_data.release();
    }

    return 0;
}

int DeformableConv2D_riscv::destroy_pipeline(const Option& opt)
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

int DeformableConv2D_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
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
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        const int packn = csrr_vlenb() / 4;
        out_elempack = num_output % packn == 0 ? packn : 1;
    }
#endif // __riscv_vector
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

#if __riscv_vector
        const int packn = csrr_vlenb() / 4;
        if (elempack == packn)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < channels; p++)
            {
                const Mat img = bottom_blob.channel(p);
                float* ptr = bottom_im2col.row(p * maxk);

                for (int u = 0; u < kernel_h; u++)
                {
                    for (int v = 0; v < kernel_w; v++)
                    {
                        const Mat offset_h_k = offset_unpacked.channel((u * kernel_w + v) * 2);
                        const Mat offset_w_k = offset_unpacked.channel((u * kernel_w + v) * 2 + 1);
                        const Mat mask_k = has_mask ? mask_unpacked.channel(u * kernel_w + v) : 0;

                        for (int i = 0; i < outh; i++)
                        {
                            for (int j = 0; j < outw; j++)
                            {
                                float offset_h = offset_h_k.row(i)[j];
                                float offset_w = offset_w_k.row(i)[j];

                                int h_in = i * stride_h - pad_top;
                                int w_in = j * stride_w - pad_left;

                                const float h_im = h_in + u * dilation_h + offset_h;
                                const float w_im = w_in + v * dilation_w + offset_w;

                                // Bilinear
                                size_t vl = __riscv_vsetvl_e32m1(packn);
                                vfloat32m1_t _val = __riscv_vfmv_v_f_f32m1(0.f, vl);
                                bool cond = h_im > -1 && w_im > -1 && h_im < h && w_im < w;
                                if (cond)
                                {
                                    int h_low = floor(h_im);
                                    int w_low = floor(w_im);
                                    int h_high = h_low + 1;
                                    int w_high = w_low + 1;

                                    float lh = h_im - h_low;
                                    float lw = w_im - w_low;
                                    float hh = 1 - lh;
                                    float hw = 1 - lw;

                                    bool v1_cond = (h_low >= 0 && w_low >= 0);
                                    bool v2_cond = (h_low >= 0 && w_high <= w - 1);
                                    bool v3_cond = (h_high <= h - 1 && w_low >= 0);
                                    bool v4_cond = (h_high <= h - 1 && w_high <= w - 1);

                                    float w1 = hh * hw;
                                    float w2 = hh * lw;
                                    float w3 = lh * hw;
                                    float w4 = lh * lw;

                                    vfloat32m1_t _v1 = v1_cond ? __riscv_vle32_v_f32m1((const float*)img.row(h_low) + w_low * packn, vl) : __riscv_vfmv_v_f_f32m1(0.f, vl);
                                    vfloat32m1_t _v2 = v2_cond ? __riscv_vle32_v_f32m1((const float*)img.row(h_low) + w_high * packn, vl) : __riscv_vfmv_v_f_f32m1(0.f, vl);
                                    vfloat32m1_t _v3 = v3_cond ? __riscv_vle32_v_f32m1((const float*)img.row(h_high) + w_low * packn, vl) : __riscv_vfmv_v_f_f32m1(0.f, vl);
                                    vfloat32m1_t _v4 = v4_cond ? __riscv_vle32_v_f32m1((const float*)img.row(h_high) + w_high * packn, vl) : __riscv_vfmv_v_f_f32m1(0.f, vl);

                                    _val = __riscv_vfmacc_vf_f32m1(_val, w1, _v1, vl);
                                    _val = __riscv_vfmacc_vf_f32m1(_val, w2, _v2, vl);
                                    _val = __riscv_vfmacc_vf_f32m1(_val, w3, _v3, vl);
                                    _val = __riscv_vfmacc_vf_f32m1(_val, w4, _v4, vl);

                                    if (has_mask)
                                        _val = __riscv_vfmul_vf_f32m1(_val, mask_k.row(i)[j], vl);
                                }

                                __riscv_vse32_v_f32m1(ptr, _val, vl);

                                ptr += packn;
                            }
                        }
                    }
                }
            }
        }
#endif // __riscv_vector

        if (elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < channels; p++)
            {
                const Mat img = bottom_blob.channel(p);
                float* ptr = bottom_im2col.row(p * maxk);

                for (int u = 0; u < kernel_h; u++)
                {
                    for (int v = 0; v < kernel_w; v++)
                    {
                        const Mat offset_h_k = offset_unpacked.channel((u * kernel_w + v) * 2);
                        const Mat offset_w_k = offset_unpacked.channel((u * kernel_w + v) * 2 + 1);
                        const Mat mask_k = has_mask ? mask_unpacked.channel(u * kernel_w + v) : 0;

                        for (int i = 0; i < outh; i++)
                        {
                            for (int j = 0; j < outw; j++)
                            {
                                float offset_h = offset_h_k.row(i)[j];
                                float offset_w = offset_w_k.row(i)[j];

                                int h_in = i * stride_h - pad_top;
                                int w_in = j * stride_w - pad_left;

                                const float h_im = h_in + u * dilation_h + offset_h;
                                const float w_im = w_in + v * dilation_w + offset_w;

                                // Bilinear
                                float val = 0.f;
                                bool cond = h_im > -1 && w_im > -1 && h_im < h && w_im < w;
                                if (cond)
                                {
                                    int h_low = (int)floorf(h_im);
                                    int w_low = (int)floorf(w_im);
                                    int h_high = h_low + 1;
                                    int w_high = w_low + 1;

                                    float lh = h_im - h_low;
                                    float lw = w_im - w_low;
                                    float hh = 1 - lh;
                                    float hw = 1 - lw;

                                    bool v1_cond = (h_low >= 0 && w_low >= 0);
                                    bool v2_cond = (h_low >= 0 && w_high <= w - 1);
                                    bool v3_cond = (h_high <= h - 1 && w_low >= 0);
                                    bool v4_cond = (h_high <= h - 1 && w_high <= w - 1);

                                    float w1 = hh * hw;
                                    float w2 = hh * lw;
                                    float w3 = lh * hw;
                                    float w4 = lh * lw;

                                    float v1 = v1_cond ? img.row(h_low)[w_low] : 0.f;
                                    float v2 = v2_cond ? img.row(h_low)[w_high] : 0.f;
                                    float v3 = v3_cond ? img.row(h_high)[w_low] : 0.f;
                                    float v4 = v4_cond ? img.row(h_high)[w_high] : 0.f;
                                    val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;

                                    if (has_mask)
                                        val *= mask_k.row(i)[j];
                                }

                                ptr[0] = val;

                                ptr += 1;
                            }
                        }
                    }
                }
            }
        }

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
    }
    else
    {
#if __riscv_vector
        const int packn = csrr_vlenb() / 4;

        if (elempack == packn && out_elempack == packn)
        {
            deformableconv2d_packn(bottom_blobs, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, pad_left, pad_top, activation_type, activation_params, opt);
        }

        if (elempack == 1 && out_elempack == packn)
        {
            deformableconv2d_pack1ton(bottom_blobs, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, pad_left, pad_top, activation_type, activation_params, opt);
        }

        if (elempack == packn && out_elempack == 1)
        {
            deformableconv2d_packnto1(bottom_blobs, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, pad_left, pad_top, activation_type, activation_params, opt);
        }
#endif // __riscv_vector
        if (elempack == 1 && out_elempack == 1)
        {
            const bool offset_not_pack = offset.elempack == 1;
            const bool mask_not_pack = has_mask ? bottom_blobs[2].elempack == 1 : true;
            const float* weight_ptr = weight_data_tm;

            // naive deformable conv
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int h_col = 0; h_col < outh; h_col++)
            {
                for (int w_col = 0; w_col < outw; w_col++)
                {
                    int h_in = h_col * stride_h - pad_top;
                    int w_in = w_col * stride_w - pad_left;
                    for (int oc = 0; oc < num_output; oc++)
                    {
                        float sum = 0.f;
                        if (bias_term)
                            sum = bias_data[oc];
                        for (int i = 0; i < kernel_h; i++)
                        {
                            for (int j = 0; j < kernel_w; j++)
                            {
                                float offset_h = 0.f;
                                float offset_w = 0.f;
                                float mask_ = 1.f;
                                if (offset_not_pack)
                                {
                                    offset_h = offset.channel((i * kernel_w + j) * 2).row(h_col)[w_col];
                                    offset_w = offset.channel((i * kernel_w + j) * 2 + 1).row(h_col)[w_col];
                                }
                                else
                                {
                                    const int y_c = (i * kernel_w + j) * 2;
                                    const int x_c = (i * kernel_w + j) * 2 + 1;
                                    offset_h = offset.channel(y_c / offset.elempack).row(h_col)[w_col * offset.elempack + y_c % offset.elempack];
                                    offset_w = offset.channel(x_c / offset.elempack).row(h_col)[w_col * offset.elempack + x_c % offset.elempack];
                                }
                                if (has_mask)
                                {
                                    const Mat& mask = bottom_blobs[2];
                                    if (mask_not_pack)
                                    {
                                        mask_ = mask.channel(i * kernel_w + j).row(h_col)[w_col];
                                    }
                                    else
                                    {
                                        const int m_c = i * kernel_w + j;
                                        mask_ = mask.channel(m_c / mask.elempack).row(h_col)[w_col * mask.elempack + m_c % mask.elempack];
                                    }
                                }
                                const float h_im = h_in + i * dilation_h + offset_h;
                                const float w_im = w_in + j * dilation_w + offset_w;

                                // Bilinear
                                const bool cond = h_im > -1 && w_im > -1 && h_im < h && w_im < w;
                                int h_low = 0;
                                int w_low = 0;
                                int h_high = 0;
                                int w_high = 0;
                                float w1 = 0.f;
                                float w2 = 0.f;
                                float w3 = 0.f;
                                float w4 = 0.f;
                                bool v1_cond = false;
                                bool v2_cond = false;
                                bool v3_cond = false;
                                bool v4_cond = false;
                                if (cond)
                                {
                                    h_low = (int)floorf(h_im);
                                    w_low = (int)floorf(w_im);
                                    h_high = h_low + 1;
                                    w_high = w_low + 1;

                                    float lh = h_im - h_low;
                                    float lw = w_im - w_low;
                                    float hh = 1 - lh;
                                    float hw = 1 - lw;

                                    v1_cond = (h_low >= 0 && w_low >= 0);
                                    v2_cond = (h_low >= 0 && w_high <= w - 1);
                                    v3_cond = (h_high <= h - 1 && w_low >= 0);
                                    v4_cond = (h_high <= h - 1 && w_high <= w - 1);

                                    w1 = hh * hw;
                                    w2 = hh * lw;
                                    w3 = lh * hw;
                                    w4 = lh * lw;
                                }

                                for (int ic = 0; ic < channels; ic++)
                                {
                                    float val = 0.f;
                                    if (cond)
                                    {
                                        float v1 = v1_cond ? bottom_blob.channel(ic).row(h_low)[w_low] : 0.f;
                                        float v2 = v2_cond ? bottom_blob.channel(ic).row(h_low)[w_high] : 0.f;
                                        float v3 = v3_cond ? bottom_blob.channel(ic).row(h_high)[w_low] : 0.f;
                                        float v4 = v4_cond ? bottom_blob.channel(ic).row(h_high)[w_high] : 0.f;
                                        val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
                                    }
                                    sum += val * mask_ * weight_ptr[((oc * channels + ic) * kernel_h + i) * kernel_w + j];
                                }
                            }
                        }
                        top_blob.channel(oc).row(h_col)[w_col] = activation_ss(sum, activation_type, activation_params);
                    }
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
