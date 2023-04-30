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

static void deformableconv2d_pack16to1_avx512(const std::vector<Mat>& bottom_blobs, Mat& top_blob, const Mat& weight_data_packed, const Mat& bias_data, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int pad_left, int pad_top, int activation_type, const Mat& activation_params, const Option& opt)
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
    const int size = outw * outh;
    const int maxk = kernel_w * kernel_h;

    const float* bias_data_ptr = bias_data;
    const int elempack = 16;
    const int out_elempack = 1;
    const int wstep = out_elempack * elempack;

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
                float _sum = 0.f;
                if (bias_data_ptr)
                    _sum = *(bias_data_ptr + oc);
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
                            float _val_channel0 = 0.f;
                            float _val_channel1 = _val_channel0;
                            float _val_channel2 = _val_channel0;
                            float _val_channel3 = _val_channel0;
                            float _val_channel4 = _val_channel0;
                            float _val_channel5 = _val_channel0;
                            float _val_channel6 = _val_channel0;
                            float _val_channel7 = _val_channel0;
                            float _val_channel8 = _val_channel0;
                            float _val_channel9 = _val_channel0;
                            float _val_channela = _val_channel0;
                            float _val_channelb = _val_channel0;
                            float _val_channelc = _val_channel0;
                            float _val_channeld = _val_channel0;
                            float _val_channele = _val_channel0;
                            float _val_channelf = _val_channel0;
                            if (cond)
                            {
                                float _v1_channel0 = _val_channel0;
                                float _v1_channel1 = _val_channel0;
                                float _v1_channel2 = _val_channel0;
                                float _v1_channel3 = _val_channel0;
                                float _v1_channel4 = _val_channel0;
                                float _v1_channel5 = _val_channel0;
                                float _v1_channel6 = _val_channel0;
                                float _v1_channel7 = _val_channel0;
                                float _v1_channel8 = _val_channel0;
                                float _v1_channel9 = _val_channel0;
                                float _v1_channela = _val_channel0;
                                float _v1_channelb = _val_channel0;
                                float _v1_channelc = _val_channel0;
                                float _v1_channeld = _val_channel0;
                                float _v1_channele = _val_channel0;
                                float _v1_channelf = _val_channel0;
                                float _v2_channel0 = _val_channel0;
                                float _v2_channel1 = _val_channel0;
                                float _v2_channel2 = _val_channel0;
                                float _v2_channel3 = _val_channel0;
                                float _v2_channel4 = _val_channel0;
                                float _v2_channel5 = _val_channel0;
                                float _v2_channel6 = _val_channel0;
                                float _v2_channel7 = _val_channel0;
                                float _v2_channel8 = _val_channel0;
                                float _v2_channel9 = _val_channel0;
                                float _v2_channela = _val_channel0;
                                float _v2_channelb = _val_channel0;
                                float _v2_channelc = _val_channel0;
                                float _v2_channeld = _val_channel0;
                                float _v2_channele = _val_channel0;
                                float _v2_channelf = _val_channel0;
                                float _v3_channel0 = _val_channel0;
                                float _v3_channel1 = _val_channel0;
                                float _v3_channel2 = _val_channel0;
                                float _v3_channel3 = _val_channel0;
                                float _v3_channel4 = _val_channel0;
                                float _v3_channel5 = _val_channel0;
                                float _v3_channel6 = _val_channel0;
                                float _v3_channel7 = _val_channel0;
                                float _v3_channel8 = _val_channel0;
                                float _v3_channel9 = _val_channel0;
                                float _v3_channela = _val_channel0;
                                float _v3_channelb = _val_channel0;
                                float _v3_channelc = _val_channel0;
                                float _v3_channeld = _val_channel0;
                                float _v3_channele = _val_channel0;
                                float _v3_channelf = _val_channel0;
                                float _v4_channel0 = _val_channel0;
                                float _v4_channel1 = _val_channel0;
                                float _v4_channel2 = _val_channel0;
                                float _v4_channel3 = _val_channel0;
                                float _v4_channel4 = _val_channel0;
                                float _v4_channel5 = _val_channel0;
                                float _v4_channel6 = _val_channel0;
                                float _v4_channel7 = _val_channel0;
                                float _v4_channel8 = _val_channel0;
                                float _v4_channel9 = _val_channel0;
                                float _v4_channela = _val_channel0;
                                float _v4_channelb = _val_channel0;
                                float _v4_channelc = _val_channel0;
                                float _v4_channeld = _val_channel0;
                                float _v4_channele = _val_channel0;
                                float _v4_channelf = _val_channel0;
                                if (v1_cond)
                                {
                                    _v1_channel0 = *(data_im_ptr + v1_pos * elempack);
                                    _v1_channel1 = *(data_im_ptr + v1_pos * elempack + 1);
                                    _v1_channel2 = *(data_im_ptr + v1_pos * elempack + 2);
                                    _v1_channel3 = *(data_im_ptr + v1_pos * elempack + 3);
                                    _v1_channel4 = *(data_im_ptr + v1_pos * elempack + 4);
                                    _v1_channel5 = *(data_im_ptr + v1_pos * elempack + 5);
                                    _v1_channel6 = *(data_im_ptr + v1_pos * elempack + 6);
                                    _v1_channel7 = *(data_im_ptr + v1_pos * elempack + 7);
                                    _v1_channel8 = *(data_im_ptr + v1_pos * elempack + 8);
                                    _v1_channel9 = *(data_im_ptr + v1_pos * elempack + 9);
                                    _v1_channela = *(data_im_ptr + v1_pos * elempack + 10);
                                    _v1_channelb = *(data_im_ptr + v1_pos * elempack + 11);
                                    _v1_channelc = *(data_im_ptr + v1_pos * elempack + 12);
                                    _v1_channeld = *(data_im_ptr + v1_pos * elempack + 13);
                                    _v1_channele = *(data_im_ptr + v1_pos * elempack + 14);
                                    _v1_channelf = *(data_im_ptr + v1_pos * elempack + 15);
                                }
                                if (v2_cond)
                                {
                                    _v2_channel0 = *(data_im_ptr + v2_pos * elempack);
                                    _v2_channel1 = *(data_im_ptr + v2_pos * elempack + 1);
                                    _v2_channel2 = *(data_im_ptr + v2_pos * elempack + 2);
                                    _v2_channel3 = *(data_im_ptr + v2_pos * elempack + 3);
                                    _v2_channel4 = *(data_im_ptr + v2_pos * elempack + 4);
                                    _v2_channel5 = *(data_im_ptr + v2_pos * elempack + 5);
                                    _v2_channel6 = *(data_im_ptr + v2_pos * elempack + 6);
                                    _v2_channel7 = *(data_im_ptr + v2_pos * elempack + 7);
                                    _v2_channel8 = *(data_im_ptr + v2_pos * elempack + 8);
                                    _v2_channel9 = *(data_im_ptr + v2_pos * elempack + 9);
                                    _v2_channela = *(data_im_ptr + v2_pos * elempack + 10);
                                    _v2_channelb = *(data_im_ptr + v2_pos * elempack + 11);
                                    _v2_channelc = *(data_im_ptr + v2_pos * elempack + 12);
                                    _v2_channeld = *(data_im_ptr + v2_pos * elempack + 13);
                                    _v2_channele = *(data_im_ptr + v2_pos * elempack + 14);
                                    _v2_channelf = *(data_im_ptr + v2_pos * elempack + 15);
                                }
                                if (v3_cond)
                                {
                                    _v3_channel0 = *(data_im_ptr + v3_pos * elempack);
                                    _v3_channel1 = *(data_im_ptr + v3_pos * elempack + 1);
                                    _v3_channel2 = *(data_im_ptr + v3_pos * elempack + 2);
                                    _v3_channel3 = *(data_im_ptr + v3_pos * elempack + 3);
                                    _v3_channel4 = *(data_im_ptr + v3_pos * elempack + 4);
                                    _v3_channel5 = *(data_im_ptr + v3_pos * elempack + 5);
                                    _v3_channel6 = *(data_im_ptr + v3_pos * elempack + 6);
                                    _v3_channel7 = *(data_im_ptr + v3_pos * elempack + 7);
                                    _v3_channel8 = *(data_im_ptr + v3_pos * elempack + 8);
                                    _v3_channel9 = *(data_im_ptr + v3_pos * elempack + 9);
                                    _v3_channela = *(data_im_ptr + v3_pos * elempack + 10);
                                    _v3_channelb = *(data_im_ptr + v3_pos * elempack + 11);
                                    _v3_channelc = *(data_im_ptr + v3_pos * elempack + 12);
                                    _v3_channeld = *(data_im_ptr + v3_pos * elempack + 13);
                                    _v3_channele = *(data_im_ptr + v3_pos * elempack + 14);
                                    _v3_channelf = *(data_im_ptr + v3_pos * elempack + 15);
                                }
                                if (v4_cond)
                                {
                                    _v4_channel0 = *(data_im_ptr + v4_pos * elempack);
                                    _v4_channel1 = *(data_im_ptr + v4_pos * elempack + 1);
                                    _v4_channel2 = *(data_im_ptr + v4_pos * elempack + 2);
                                    _v4_channel3 = *(data_im_ptr + v4_pos * elempack + 3);
                                    _v4_channel4 = *(data_im_ptr + v4_pos * elempack + 4);
                                    _v4_channel5 = *(data_im_ptr + v4_pos * elempack + 5);
                                    _v4_channel6 = *(data_im_ptr + v4_pos * elempack + 6);
                                    _v4_channel7 = *(data_im_ptr + v4_pos * elempack + 7);
                                    _v4_channel8 = *(data_im_ptr + v4_pos * elempack + 8);
                                    _v4_channel9 = *(data_im_ptr + v4_pos * elempack + 9);
                                    _v4_channela = *(data_im_ptr + v4_pos * elempack + 10);
                                    _v4_channelb = *(data_im_ptr + v4_pos * elempack + 11);
                                    _v4_channelc = *(data_im_ptr + v4_pos * elempack + 12);
                                    _v4_channeld = *(data_im_ptr + v4_pos * elempack + 13);
                                    _v4_channele = *(data_im_ptr + v4_pos * elempack + 14);
                                    _v4_channelf = *(data_im_ptr + v4_pos * elempack + 15);
                                }
                                _val_channel0 = w1 * _v1_channel0 + w2 * _v2_channel0 + w3 * _v3_channel0 + w4 * _v4_channel0;
                                _val_channel1 = w1 * _v1_channel1 + w2 * _v2_channel1 + w3 * _v3_channel1 + w4 * _v4_channel1;
                                _val_channel2 = w1 * _v1_channel2 + w2 * _v2_channel2 + w3 * _v3_channel2 + w4 * _v4_channel2;
                                _val_channel3 = w1 * _v1_channel3 + w2 * _v2_channel3 + w3 * _v3_channel3 + w4 * _v4_channel3;
                                _val_channel4 = w1 * _v1_channel4 + w2 * _v2_channel4 + w3 * _v3_channel4 + w4 * _v4_channel4;
                                _val_channel5 = w1 * _v1_channel5 + w2 * _v2_channel5 + w3 * _v3_channel5 + w4 * _v4_channel5;
                                _val_channel6 = w1 * _v1_channel6 + w2 * _v2_channel6 + w3 * _v3_channel6 + w4 * _v4_channel6;
                                _val_channel7 = w1 * _v1_channel7 + w2 * _v2_channel7 + w3 * _v3_channel7 + w4 * _v4_channel7;
                                _val_channel8 = w1 * _v1_channel8 + w2 * _v2_channel8 + w3 * _v3_channel8 + w4 * _v4_channel8;
                                _val_channel9 = w1 * _v1_channel9 + w2 * _v2_channel9 + w3 * _v3_channel9 + w4 * _v4_channel9;
                                _val_channela = w1 * _v1_channela + w2 * _v2_channela + w3 * _v3_channela + w4 * _v4_channela;
                                _val_channelb = w1 * _v1_channelb + w2 * _v2_channelb + w3 * _v3_channelb + w4 * _v4_channelb;
                                _val_channelc = w1 * _v1_channelc + w2 * _v2_channelc + w3 * _v3_channelc + w4 * _v4_channelc;
                                _val_channeld = w1 * _v1_channeld + w2 * _v2_channeld + w3 * _v3_channeld + w4 * _v4_channeld;
                                _val_channele = w1 * _v1_channele + w2 * _v2_channele + w3 * _v3_channele + w4 * _v4_channele;
                                _val_channelf = w1 * _v1_channelf + w2 * _v2_channelf + w3 * _v3_channelf + w4 * _v4_channelf;
                            }
                            if (has_mask)
                            {
                                _val_channel0 *= mask_;
                                _val_channel1 *= mask_;
                                _val_channel2 *= mask_;
                                _val_channel3 *= mask_;
                                _val_channel4 *= mask_;
                                _val_channel5 *= mask_;
                                _val_channel6 *= mask_;
                                _val_channel7 *= mask_;
                                _val_channel8 *= mask_;
                                _val_channel9 *= mask_;
                                _val_channela *= mask_;
                                _val_channelb *= mask_;
                                _val_channelc *= mask_;
                                _val_channeld *= mask_;
                                _val_channele *= mask_;
                                _val_channelf *= mask_;
                            }
                            float _conv_w0 = *(kptr);
                            float _conv_w1 = *(kptr + out_elempack); // 1 * out_elempack
                            _sum += (_val_channel0 * _conv_w0);
                            _sum += (_val_channel1 * _conv_w1);
                            float _conv_w2 = *(kptr + 2); // 2 * out_elempack
                            float _conv_w3 = *(kptr + 3); // 3 * out_elempack
                            _sum += (_val_channel2 * _conv_w2);
                            _sum += (_val_channel3 * _conv_w3);
                            float _conv_w4 = *(kptr + 4); // 4 * out_elempack
                            float _conv_w5 = *(kptr + 5); // 5 * out_elempack
                            _sum += (_val_channel4 * _conv_w4);
                            _sum += (_val_channel5 * _conv_w5);
                            float _conv_w6 = *(kptr + 6); // 6 * out_elempack
                            float _conv_w7 = *(kptr + 7); // 7 * out_elempack
                            _sum += (_val_channel6 * _conv_w6);
                            _sum += (_val_channel7 * _conv_w7);
                            float _conv_w8 = *(kptr + 8); // 8 * out_elempack
                            float _conv_w9 = *(kptr + 9); // 9 * out_elempack
                            _sum += (_val_channel8 * _conv_w8);
                            _sum += (_val_channel9 * _conv_w9);
                            float _conv_wa = *(kptr + 10); // 10 * out_elempack
                            float _conv_wb = *(kptr + 11); // 11 * out_elempack
                            _sum += (_val_channela * _conv_wa);
                            _sum += (_val_channelb * _conv_wb);
                            float _conv_wc = *(kptr + 12); // 12 * out_elempack
                            float _conv_wd = *(kptr + 13); // 13 * out_elempack
                            _sum += (_val_channelc * _conv_wc);
                            _sum += (_val_channeld * _conv_wd);
                            float _conv_we = *(kptr + 14); // 14 * out_elempack
                            float _conv_wf = *(kptr + 15); // 15 * out_elempack
                            _sum += (_val_channele * _conv_we);
                            _sum += (_val_channelf * _conv_wf);
                            kptr += wstep;
                        }
                    }
                }
                _sum = activation_ss(_sum, activation_type, activation_params);
                *(outptr + h_col * outw + w_col) = _sum;
            }
        }
    }
}
