

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

static void deformableconv2d_packnto1(const std::vector<Mat>& bottom_blobs, Mat& top_blob, const Mat& weight_data_packed, const Mat& bias_data, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int pad_left, int pad_top, int activation_type, const Mat& activation_params, const Option& opt)
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
    int outch = top_blob.c;

    const float* bias_data_ptr = bias_data;
    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m1(packn);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int h_col = 0; h_col < outh; h_col++)
    {
        for (int w_col = 0; w_col < outw; w_col++)
        {
            int h_in = h_col * stride_h - pad_top;
            int w_in = w_col * stride_w - pad_left;
            for (int oc = 0; oc < outch; oc++)
            {
                const float* kptr = weight_data_packed.channel(oc);
                float* outptr = top_blob.channel(oc);
                float sum = 0.f;
                if (bias_data_ptr)
                    sum = bias_data_ptr[oc];

                vfloat32m1_t _sum = __riscv_vfmv_v_f_f32m1(0.f, vl);

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
                            int h_low = (int)floorf(h_im);
                            int w_low = (int)floorf(w_im);
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

                        for (int ic = 0; ic < inch; ic++)
                        {
                            const float* data_im_ptr = bottom_blob.channel(ic);

                            if (cond)
                            {
                                vfloat32m1_t _v1 = v1_cond ? __riscv_vle32_v_f32m1(data_im_ptr + v1_pos * packn, vl) : __riscv_vfmv_v_f_f32m1(0.f, vl);
                                vfloat32m1_t _v2 = v2_cond ? __riscv_vle32_v_f32m1(data_im_ptr + v2_pos * packn, vl) : __riscv_vfmv_v_f_f32m1(0.f, vl);
                                vfloat32m1_t _v3 = v3_cond ? __riscv_vle32_v_f32m1(data_im_ptr + v3_pos * packn, vl) : __riscv_vfmv_v_f_f32m1(0.f, vl);
                                vfloat32m1_t _v4 = v4_cond ? __riscv_vle32_v_f32m1(data_im_ptr + v4_pos * packn, vl) : __riscv_vfmv_v_f_f32m1(0.f, vl);

                                vfloat32m1_t _val = __riscv_vfmv_v_f_f32m1(0.f, vl);
                                _val = __riscv_vfmacc_vf_f32m1(_val, w1, _v1, vl);
                                _val = __riscv_vfmacc_vf_f32m1(_val, w2, _v2, vl);
                                _val = __riscv_vfmacc_vf_f32m1(_val, w3, _v3, vl);
                                _val = __riscv_vfmacc_vf_f32m1(_val, w4, _v4, vl);

                                if (has_mask)
                                    _val = __riscv_vfmul_vf_f32m1(_val, mask_, vl);

                                vfloat32m1_t _w = __riscv_vle32_v_f32m1(kptr, vl);
                                _sum = __riscv_vfmacc_vv_f32m1(_sum, _val, _w, vl);
                            }

                            kptr += packn;
                        }
                    }
                }

                vfloat32m1_t _v_sum = __riscv_vfredusum_vs_f32m1_f32m1(_sum, __riscv_vfmv_v_f_f32m1(0.f, vl), vl);
                sum += __riscv_vfmv_f_s_f32m1_f32(_v_sum);

                sum = activation_ss(sum, activation_type, activation_params);
                outptr[h_col * outw + w_col] = sum;
            }
        }
    }
}
