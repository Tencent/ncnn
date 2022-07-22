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

#include "deformableconv2d.h"

#include "fused_activation.h"

namespace ncnn {

DeformableConv2D::DeformableConv2D()
{
    one_blob_only = false;
    support_inplace = false;
}

int DeformableConv2D::load_param(const ParamDict& pd)
{
    num_output = pd.get(0, 0);
    kernel_w = pd.get(1, 0);
    kernel_h = pd.get(11, kernel_w);
    dilation_w = pd.get(2, 1);
    dilation_h = pd.get(12, dilation_w);
    stride_w = pd.get(3, 1);
    stride_h = pd.get(13, stride_w);
    pad_left = pd.get(4, 0);
    pad_right = pd.get(15, pad_left);
    pad_top = pd.get(14, pad_left);
    pad_bottom = pd.get(16, pad_top);
    bias_term = pd.get(5, 0);
    weight_data_size = pd.get(6, 0);
    activation_type = pd.get(9, 0);
    activation_params = pd.get(10, Mat());
    return 0;
}

int DeformableConv2D::load_model(const ModelBin& mb)
{
    weight_data = mb.load(weight_data_size, 0);
    if (weight_data.empty())
        return -100;

    if (bias_term)
    {
        bias_data = mb.load(num_output, 1);
        if (bias_data.empty())
            return -100;
    }
    return 0;
}

int DeformableConv2D::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& offset = bottom_blobs[1];
    const Mat& mask = bottom_blobs[2];

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int in_c = bottom_blob.c;
    const size_t elemsize = bottom_blob.elemsize;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    const int out_w = (w + pad_left + pad_right - kernel_extent_w) / stride_w + 1;
    const int out_h = (h + pad_top + pad_bottom - kernel_extent_h) / stride_h + 1;

    // output.shape is [num_output, out_h, out_w] (in python).
    Mat& output = top_blobs[0];
    output.create(out_w * out_h * num_output, elemsize, opt.blob_allocator);
    if (output.empty())
        return -100;

    // bottom_blob.shape is [in_c, h, w] (in python).
    Mat bottom_blob_flatten = bottom_blob.reshape(w * h * in_c);
    Mat offset_flatten = offset.reshape(offset.w * offset.h * offset.c);
    Mat mask_flatten = mask.reshape(mask.w * mask.h * mask.c);
    const float* data_im_ptr = bottom_blob_flatten;
    const float* data_offset_ptr = offset_flatten;
    const float* data_mask_ptr = mask_flatten;
    const float* weight_ptr = weight_data;
    const float* bias_ptr = weight_data;
    if (bias_term)
        bias_ptr = bias_data;
    float* output_ptr = output;

    // deformable conv
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int h_col = 0; h_col < out_h; h_col++)
    {
        for (int w_col = 0; w_col < out_w; w_col++)
        {
            int h_in = h_col * stride_h - pad_top;
            int w_in = w_col * stride_w - pad_left;
            float* output_hw_ptr = output_ptr + (h_col * out_w + w_col);
            for (int oc = 0; oc < num_output; oc++)
            {
                float sum = 0.f;
                if (bias_term)
                    sum = bias_ptr[oc];
                for (int i = 0; i < kernel_h; i++)
                {
                    for (int j = 0; j < kernel_w; j++)
                    {
                        const int data_offset_h_ptr = (((i * kernel_w + j) * 2) * out_h + h_col) * out_w + w_col;
                        const int data_offset_w_ptr = (((i * kernel_w + j) * 2 + 1) * out_h + h_col) * out_w + w_col;
                        const int data_mask_hw_ptr = ((i * kernel_w + j) * out_h + h_col) * out_w + w_col;

                        const float offset_h = data_offset_ptr[data_offset_h_ptr];
                        const float offset_w = data_offset_ptr[data_offset_w_ptr];
                        const float mask_ = data_mask_ptr[data_mask_hw_ptr];
                        const float h_im = h_in + i * dilation_h + offset_h;
                        const float w_im = w_in + j * dilation_w + offset_w;

                        // Bilinear
                        const bool cond = h_im > -1 && w_im > -1 && h_im < h && w_im < w;
                        float w1 = 0.f;
                        float w2 = 0.f;
                        float w3 = 0.f;
                        float w4 = 0.f;
                        bool v1_cond = false;
                        bool v2_cond = false;
                        bool v3_cond = false;
                        bool v4_cond = false;
                        int v1_pos = 0;
                        int v2_pos = 0;
                        int v3_pos = 0;
                        int v4_pos = 0;
                        if (cond) {
                            int h_low = floor(h_im);
                            int w_low = floor(w_im);
                            int h_high = h_low + 1;
                            int w_high = w_low + 1;

                            float lh = h_im - h_low;
                            float lw = w_im - w_low;
                            float hh = 1 - lh;
                            float hw = 1 - lw;

                            v1_cond = (h_low >= 0 && w_low >= 0);
                            v2_cond = (h_low >= 0 && w_high <= w - 1);
                            v3_cond = (h_high <= h - 1 && w_low >= 0);
                            v4_cond = (h_high <= h - 1 && w_high <= w - 1);
                            if (v1_cond) {
                                v1_pos = h_low * w + w_low;
                            }
                            if (v2_cond) {
                                v2_pos = h_low * w + w_high;
                            }
                            if (v3_cond) {
                                v3_pos = h_high * w + w_low;
                            }
                            if (v4_cond) {
                                v4_pos = h_high * w + w_high;
                            }

                            w1 = hh * hw;
                            w2 = hh * lw;
                            w3 = lh * hw;
                            w4 = lh * lw;
                        }

                        const float* data_im_channel_ptr = data_im_ptr;
                        for (int c_im = 0; c_im < in_c; c_im++)
                        {
                            float val = 0.f;
                            if (cond) {
                                float v1 = 0.f;
                                if (v1_cond) {
                                    v1 = data_im_channel_ptr[v1_pos];
                                }
                                float v2 = 0.f;
                                if (v2_cond) {
                                    v2 = data_im_channel_ptr[v2_pos];
                                }
                                float v3 = 0.f;
                                if (v3_cond) {
                                    v3 = data_im_channel_ptr[v3_pos];
                                }
                                float v4 = 0.f;
                                if (v4_cond) {
                                    v4 = data_im_channel_ptr[v4_pos];
                                }
                                val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
                            }
                            sum += val * mask_ * weight_ptr[((oc * in_c + c_im) * kernel_h + i) * kernel_w + j];
                            data_im_channel_ptr += h*w;
                        }
                    }
                }
                *output_hw_ptr = activation_ss(sum, activation_type, activation_params);
                output_hw_ptr += out_h * out_w;
            }
        }
    }
    output = output.reshape(out_w, out_h, num_output);
    top_blobs[0] = output;
    return 0;
}

} // namespace ncnn
