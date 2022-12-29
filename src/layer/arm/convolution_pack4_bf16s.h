// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

static void convolution_pack4_bf16s_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_bf16, const Mat& bias_data, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int activation_type, const Mat& activation_params, const Option& opt)
{
    int w = bottom_blob.w;
    int channels = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    const float* bias_data_ptr = bias_data;

    // num_output
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        unsigned short* outptr = top_blob.channel(p);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                float32x4_t _sum = vdupq_n_f32(0.f);

                if (bias_data_ptr)
                {
                    _sum = vld1q_f32(bias_data_ptr + p * 4);
                }

                const unsigned short* kptr = weight_data_bf16.channel(p);

                // channels
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob.channel(q);
                    const unsigned short* sptr = m.row<const unsigned short>(i * stride_h) + j * stride_w * 4;

                    for (int k = 0; k < maxk; k++)
                    {
                        float32x4_t _val = bfloat2float(vld1_u16(sptr + space_ofs[k] * 4));

                        float32x4_t _w0 = bfloat2float(vld1_u16(kptr));
                        float32x4_t _w1 = bfloat2float(vld1_u16(kptr + 4));
                        float32x4_t _w2 = bfloat2float(vld1_u16(kptr + 8));
                        float32x4_t _w3 = bfloat2float(vld1_u16(kptr + 12));

#if __aarch64__
                        _sum = vmlaq_laneq_f32(_sum, _w0, _val, 0);
                        _sum = vmlaq_laneq_f32(_sum, _w1, _val, 1);
                        _sum = vmlaq_laneq_f32(_sum, _w2, _val, 2);
                        _sum = vmlaq_laneq_f32(_sum, _w3, _val, 3);
#else
                        _sum = vmlaq_lane_f32(_sum, _w0, vget_low_f32(_val), 0);
                        _sum = vmlaq_lane_f32(_sum, _w1, vget_low_f32(_val), 1);
                        _sum = vmlaq_lane_f32(_sum, _w2, vget_high_f32(_val), 0);
                        _sum = vmlaq_lane_f32(_sum, _w3, vget_high_f32(_val), 1);
#endif

                        kptr += 16;
                    }
                }

                _sum = activation_ps(_sum, activation_type, activation_params);

                vst1_u16(outptr + j * 4, float2bfloat(_sum));
            }

            outptr += outw * 4;
        }
    }
}
