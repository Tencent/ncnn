// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "deformableconv2d_x86.h"

#include "layer_type.h"

namespace ncnn {

DeformableConv2D_x86::DeformableConv2D_x86()
{
    one_blob_only = false;
    support_inplace = false;

    inner_product = 0;
    permute = 0;
}

int DeformableConv2D_x86::create_pipeline(const Option& opt)
{
    const int in_c = weight_data_size / (num_output * kernel_h * kernel_w);
    {
        Mat weight_3d = weight_data.reshape(kernel_w * kernel_h, in_c, num_output);
        weight_data_t.create(in_c, kernel_w * kernel_h, num_output);
        if (weight_data_t.empty())
            return -100;
        for (int q = 0; q < num_output; q++)
        {
            const Mat m = weight_3d.channel(q);
            float* outptr = weight_data_t.channel(q);

            for (int i = 0; i < kernel_w * kernel_h; i++)
            {
                for (int j = 0; j < in_c; j++)
                {
                    *outptr++ = m.row(j)[i];
                }
            }
        }
        weight_3d.release();
        weight_data_t = weight_data_t.reshape(in_c * kernel_w * kernel_h, num_output);

        inner_product = ncnn::create_layer(ncnn::LayerType::InnerProduct);
        ncnn::ParamDict pd;
        pd.set(0, num_output);
        pd.set(1, bias_term);
        pd.set(2, weight_data_size);
        pd.set(9, activation_type);
        pd.set(10, activation_params);
        inner_product->load_param(pd);
        ncnn::Mat weights[2];
        weights[0] = weight_data_t;
        if (bias_term)
            weights[1] = bias_data;
        inner_product->load_model(ncnn::ModelBinFromMatArray(weights));
        inner_product->create_pipeline(opt);

        permute = ncnn::create_layer(ncnn::LayerType::Permute);
        ncnn::ParamDict permute_pd;
        permute_pd.set(0, 1);
        permute->load_param(permute_pd);
        permute->create_pipeline(opt);
    }

    return 0;
}

int DeformableConv2D_x86::destroy_pipeline(const Option& opt)
{
    if (inner_product)
    {
        inner_product->destroy_pipeline(opt);
        delete inner_product;
        inner_product = 0;
    }
    if (permute)
    {
        permute->destroy_pipeline(opt);
        delete permute;
        permute = 0;
    }

    return 0;
}

int DeformableConv2D_x86::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& offset = bottom_blobs[1];

    const bool has_mask = (bottom_blobs.size() == 3);

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int in_c = bottom_blob.c;
    const size_t elemsize = bottom_blob.elemsize;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    const int out_w = (w + pad_left + pad_right - kernel_extent_w) / stride_w + 1;
    const int out_h = (h + pad_top + pad_bottom - kernel_extent_h) / stride_h + 1;

    // output = im2col matmul weight_t, im2col.shape is [out_h * out_w, kernel_h * kernel_w * in_c] (in python),
    // weight_t.shape is [num_output, kernel_h * kernel_w * in_c] (in python),
    // output.shape   is [out_h * out_w, num_output] (in python).
    Mat im2col;
    im2col.create(kernel_h * kernel_w * in_c * out_h * out_w, elemsize, opt.blob_allocator);
    if (im2col.empty())
        return -100;

    Mat& output = top_blobs[0];
    output.create(num_output, out_h * out_w, elemsize, opt.blob_allocator);
    if (output.empty())
        return -100;

    Mat bottom_blob_flatten = bottom_blob.reshape(w * h * in_c);
    Mat offset_flatten = offset.reshape(offset.w * offset.h * offset.c);
    const float* data_im_ptr = bottom_blob_flatten;
    const float* data_offset_ptr = offset_flatten;
    float* im2col_ptr = im2col;

    // im2col
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int h_col = 0; h_col < out_h; h_col++)
    {
        for (int w_col = 0; w_col < out_w; w_col++)
        {
            int h_in = h_col * stride_h - pad_top;
            int w_in = w_col * stride_w - pad_left;
            float* data_col_ptr = im2col_ptr + (h_col * out_w + w_col) * kernel_h * kernel_w * in_c;
            for (int i = 0; i < kernel_h; i++)
            {
                for (int j = 0; j < kernel_w; j++)
                {
                    const int data_offset_h_ptr = (((i * kernel_w + j) * 2) * out_h + h_col) * out_w + w_col;
                    const int data_offset_w_ptr = (((i * kernel_w + j) * 2 + 1) * out_h + h_col) * out_w + w_col;

                    const float offset_h = data_offset_ptr[data_offset_h_ptr];
                    const float offset_w = data_offset_ptr[data_offset_w_ptr];
                    const float mask_ = has_mask ? bottom_blobs[2].channel(i * kernel_w + j).row(h_col)[w_col] : 1.f;
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

                        v1_cond = (h_low >= 0 && w_low >= 0);
                        v2_cond = (h_low >= 0 && w_high <= w - 1);
                        v3_cond = (h_high <= h - 1 && w_low >= 0);
                        v4_cond = (h_high <= h - 1 && w_high <= w - 1);
                        if (v1_cond)
                            v1_pos = h_low * w + w_low;
                        if (v2_cond)
                            v2_pos = h_low * w + w_high;
                        if (v3_cond)
                            v3_pos = h_high * w + w_low;
                        if (v4_cond)
                            v4_pos = h_high * w + w_high;

                        w1 = hh * hw;
                        w2 = hh * lw;
                        w3 = lh * hw;
                        w4 = lh * lw;
                    }

                    const float* data_im_channel_ptr = data_im_ptr;
                    for (int c_im = 0; c_im < in_c; c_im++)
                    {
                        float val = 0.f;
                        if (cond)
                        {
                            float v1 = v1_cond ? data_im_channel_ptr[v1_pos] : 0.f;
                            float v2 = v2_cond ? data_im_channel_ptr[v2_pos] : 0.f;
                            float v3 = v3_cond ? data_im_channel_ptr[v3_pos] : 0.f;
                            float v4 = v4_cond ? data_im_channel_ptr[v4_pos] : 0.f;
                            val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
                        }
                        *data_col_ptr = val * mask_;
                        data_col_ptr += 1;
                        data_im_channel_ptr += h * w;
                    }
                }
            }
        }
    }
    im2col = im2col.reshape(kernel_h * kernel_w * in_c, out_h * out_w);
    // call InnerProduct
    inner_product->forward(im2col, output, opt);
    ncnn::Mat output_t;
    // call Permute
    permute->forward(output, output_t, opt);
    output_t = output_t.reshape(out_w, out_h, num_output);
    top_blobs[0] = output_t;
    return 0;
}

} // namespace ncnn
