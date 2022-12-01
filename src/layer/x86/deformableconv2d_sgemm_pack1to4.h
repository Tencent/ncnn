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

#include "convolution_sgemm_pack1to4.h"

static void deformableconv2d_im2col_sgemm_pack1to4_sse(const std::vector<Mat>& bottom_blobs, Mat& top_blob, const Mat& kernel, const Mat& _bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int pad_left, int pad_top, const Option& opt)
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& offset = bottom_blobs[1];
    const bool has_mask = (bottom_blobs.size() == 3);
    const bool offset_not_pack = offset.elempack == 1;
    const bool mask_not_pack = has_mask ? bottom_blobs[2].elempack == 1 : true;

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    const int size = outw * outh;

    const int maxk = kernel_w * kernel_h;

    // im2col
    Mat bottom_im2col(size, maxk, inch, 4u, 1, opt.workspace_allocator);
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int h_col = 0; h_col < outh; h_col++)
        {
            for (int w_col = 0; w_col < outw; w_col++)
            {
                int h_in = h_col * stride_h - pad_top;
                int w_in = w_col * stride_w - pad_left;
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
                            h_low = floor(h_im);
                            w_low = floor(w_im);
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
                        for (int ic = 0; ic < inch; ic++)
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
                            bottom_im2col.channel(ic).row(i * kernel_w + j)[h_col * outw + w_col] = val * mask_;
                        }
                    }
                }
            }
        }
    }

    im2col_sgemm_pack1to4_sse(bottom_im2col, top_blob, kernel, _bias, opt);
}
