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

static void convolution_transform_kernel_pack4_neon(const Mat& weight_data, Mat& weight_data_pack4, int num_input, int num_output, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // src = kw-kh-inch-outch
    // dst = 4b-4a-kw-kh-inch/4a-outch/4b
    Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

    weight_data_pack4.create(maxk, num_input / 4, num_output / 4, (size_t)4 * 16, 16);

    for (int q = 0; q + 3 < num_output; q += 4)
    {
        const Mat k0 = weight_data_r2.channel(q);
        const Mat k1 = weight_data_r2.channel(q + 1);
        const Mat k2 = weight_data_r2.channel(q + 2);
        const Mat k3 = weight_data_r2.channel(q + 3);

        Mat g0 = weight_data_pack4.channel(q / 4);

        for (int p = 0; p + 3 < num_input; p += 4)
        {
            const float* k00 = k0.row(p);
            const float* k01 = k0.row(p + 1);
            const float* k02 = k0.row(p + 2);
            const float* k03 = k0.row(p + 3);

            const float* k10 = k1.row(p);
            const float* k11 = k1.row(p + 1);
            const float* k12 = k1.row(p + 2);
            const float* k13 = k1.row(p + 3);

            const float* k20 = k2.row(p);
            const float* k21 = k2.row(p + 1);
            const float* k22 = k2.row(p + 2);
            const float* k23 = k2.row(p + 3);

            const float* k30 = k3.row(p);
            const float* k31 = k3.row(p + 1);
            const float* k32 = k3.row(p + 2);
            const float* k33 = k3.row(p + 3);

            float* g00 = g0.row(p / 4);

            for (int k = 0; k < maxk; k++)
            {
                g00[0] = k00[k];
                g00[1] = k10[k];
                g00[2] = k20[k];
                g00[3] = k30[k];

                g00[4] = k01[k];
                g00[5] = k11[k];
                g00[6] = k21[k];
                g00[7] = k31[k];

                g00[8] = k02[k];
                g00[9] = k12[k];
                g00[10] = k22[k];
                g00[11] = k32[k];

                g00[12] = k03[k];
                g00[13] = k13[k];
                g00[14] = k23[k];
                g00[15] = k33[k];

                g00 += 16;
            }
        }
    }
}

static void convolution_pack4_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_pack4, const Mat& bias_data, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int activation_type, const Mat& activation_params, const Option& opt)
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

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        float* outptr = top_blob.channel(p);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                float32x4_t _sum = vdupq_n_f32(0.f);

                if (bias_data_ptr)
                {
                    _sum = vld1q_f32(bias_data_ptr + p * 4);
                }

                const float* kptr = (const float*)weight_data_pack4 + maxk * channels * p * 16;

                // channels
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob.channel(q);
                    const float* sptr = m.row(i * stride_h) + j * stride_w * 4;

                    for (int k = 0; k < maxk; k++) // 29.23
                    {
                        float32x4_t _val = vld1q_f32(sptr + space_ofs[k] * 4);

                        float32x4_t _w0 = vld1q_f32(kptr);
                        float32x4_t _w1 = vld1q_f32(kptr + 4);
                        float32x4_t _w2 = vld1q_f32(kptr + 8);
                        float32x4_t _w3 = vld1q_f32(kptr + 12);

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

                vst1q_f32(outptr + j * 4, _sum);
            }

            outptr += outw * 4;
        }
    }
}
