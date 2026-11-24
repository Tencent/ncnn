// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "deformableconv2d_powerpc.h"

#include <limits.h>

#include "fused_activation.h"

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

namespace ncnn {

DeformableConv2D_powerpc::DeformableConv2D_powerpc()
{
}

#if __ALTIVEC__
static void deformableconv2d_forward_fp32(const std::vector<Mat>& bottom_blobs, Mat& output, const float* weight_ptr, const float* bias_ptr, int bias_term, int num_output, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int pad_left, int pad_top, int out_w, int out_h, int activation_type, const Mat& activation_params, int num_threads)
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& offset = bottom_blobs[1];

    const bool has_mask = (bottom_blobs.size() == 3);
    const Mat& mask_blob = has_mask ? bottom_blobs[2] : Mat();

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int in_c = bottom_blob.c;

    #pragma omp parallel for num_threads(num_threads)
    for (int h_col = 0; h_col < out_h; h_col++)
    {
        for (int w_col = 0; w_col < out_w; w_col++)
        {
            int h_in = h_col * stride_h - pad_top;
            int w_in = w_col * stride_w - pad_left;
            for (int oc = 0; oc < num_output; oc++)
            {
                float sum = 0.f;
                if (bias_term)
                    sum = bias_ptr[oc];

                v4f32 _acc = __ncnn_altivec_fill_f32(0.f);

                for (int i = 0; i < kernel_h; i++)
                {
                    for (int j = 0; j < kernel_w; j++)
                    {
                        const float offset_h = offset.channel((i * kernel_w + j) * 2).row(h_col)[w_col];
                        const float offset_w = offset.channel((i * kernel_w + j) * 2 + 1).row(h_col)[w_col];
                        const float mask_ = has_mask ? mask_blob.channel(i * kernel_w + j).row(h_col)[w_col] : 1.f;
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

                        v4f32 _w1 = __ncnn_altivec_fill_f32(w1);
                        v4f32 _w2 = __ncnn_altivec_fill_f32(w2);
                        v4f32 _w3 = __ncnn_altivec_fill_f32(w3);
                        v4f32 _w4 = __ncnn_altivec_fill_f32(w4);
                        v4f32 _mask = __ncnn_altivec_fill_f32(mask_);

                        int c_im = 0;
                        for (; c_im + 3 < in_c; c_im += 4)
                        {
                            float v1[4];
                            float v2[4];
                            float v3[4];
                            float v4[4];
                            float wt[4];
                            for (int k = 0; k < 4; k++)
                            {
                                const int c = c_im + k;
                                const Mat plane = bottom_blob.channel(c);
                                const float* prow_low = plane.row(h_low);
                                const float* prow_high = plane.row(h_high);

                                v1[k] = (cond && v1_cond) ? prow_low[w_low] : 0.f;
                                v2[k] = (cond && v2_cond) ? prow_low[w_high] : 0.f;
                                v3[k] = (cond && v3_cond) ? prow_high[w_low] : 0.f;
                                v4[k] = (cond && v4_cond) ? prow_high[w_high] : 0.f;
                                wt[k] = weight_ptr[((oc * in_c + c) * kernel_h + i) * kernel_w + j];
                            }

                            v4f32 _v1 = __ncnn_altivec_load(v1);
                            v4f32 _v2 = __ncnn_altivec_load(v2);
                            v4f32 _v3 = __ncnn_altivec_load(v3);
                            v4f32 _v4 = __ncnn_altivec_load(v4);
                            v4f32 _wt = __ncnn_altivec_load(wt);

                            // val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
                            v4f32 _val = vec_add(vec_add(__ncnn_altivec_mul(_w1, _v1), __ncnn_altivec_mul(_w2, _v2)), vec_add(__ncnn_altivec_mul(_w3, _v3), __ncnn_altivec_mul(_w4, _v4)));
                            _val = __ncnn_altivec_mul(_val, _mask);

                            // sum += val * weight
                            _acc = __ncnn_altivec_fmadd(_acc, _val, _wt);
                        }

                        // tail channels
                        float tail = 0.f;
                        for (; c_im < in_c; c_im++)
                        {
                            float val = 0.f;
                            if (cond)
                            {
                                float v1 = v1_cond ? bottom_blob.channel(c_im).row(h_low)[w_low] : 0.f;
                                float v2 = v2_cond ? bottom_blob.channel(c_im).row(h_low)[w_high] : 0.f;
                                float v3 = v3_cond ? bottom_blob.channel(c_im).row(h_high)[w_low] : 0.f;
                                float v4 = v4_cond ? bottom_blob.channel(c_im).row(h_high)[w_high] : 0.f;
                                val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
                            }
                            tail += val * mask_ * weight_ptr[((oc * in_c + c_im) * kernel_h + i) * kernel_w + j];
                        }

                        sum += tail;
                    }
                }

                // reduce the vector accumulator once after all kernel positions
                sum += __ncnn_altivec_reduce_add(_acc);

                output.channel(oc).row(h_col)[w_col] = activation_ss(sum, activation_type, activation_params);
            }
        }
    }
}
#endif // __ALTIVEC__

int DeformableConv2D_powerpc::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& offset = bottom_blobs[1];
    const bool has_mask = (bottom_blobs.size() == 3);
    const Mat& mask_blob = has_mask ? bottom_blobs[2] : Mat();

    // non-fp32 or packed input: delegate to the generic implementation
    if (bottom_blob.elemsize != 4u || bottom_blob.elempack != 1 || offset.elemsize != 4u || offset.elempack != 1 || (has_mask && (mask_blob.elemsize != 4u || mask_blob.elempack != 1)))
    {
        std::vector<Mat> bottom_blobs_unpacked(bottom_blobs.size());
        for (size_t i = 0; i < bottom_blobs.size(); i++)
        {
            bottom_blobs_unpacked[i] = bottom_blobs[i];
            if (bottom_blobs[i].elempack != 1)
                convert_packing(bottom_blobs[i], bottom_blobs_unpacked[i], 1, opt);
        }

        return DeformableConv2D::forward(bottom_blobs_unpacked, top_blobs, opt);
    }

#if __ALTIVEC__
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    const int out_w = (w + pad_left + pad_right - kernel_extent_w) / stride_w + 1;
    const int out_h = (h + pad_top + pad_bottom - kernel_extent_h) / stride_h + 1;

    Mat& output = top_blobs[0];
    output.create(out_w, out_h, num_output, bottom_blob.elemsize, opt.blob_allocator);
    if (output.empty())
        return -100;

    const float* weight_ptr = weight_data;
    const float* bias_ptr = weight_data;
    if (bias_term)
        bias_ptr = bias_data;

    deformableconv2d_forward_fp32(bottom_blobs, output, weight_ptr, bias_ptr, bias_term, num_output, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, pad_left, pad_top, out_w, out_h, activation_type, activation_params, opt.num_threads);

    return 0;
#endif // __ALTIVEC__

    return DeformableConv2D::forward(bottom_blobs, top_blobs, opt);
}

} // namespace ncnn
