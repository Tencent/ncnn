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

static void convolution_transform_kernel_pack4to1_neon(const Mat& weight_data, Mat& weight_data_pack4to1, int num_input, int num_output, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // src = kw-kh-inch-outch
    // dst = 4a-kw-kh-inch/4a-outch
    Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

    weight_data_pack4to1.create(maxk, num_input / 4, num_output, (size_t)4 * 4, 4);

    for (int q = 0; q < num_output; q++)
    {
        const Mat k0 = weight_data_r2.channel(q);
        float* g00 = weight_data_pack4to1.channel(q);

        for (int p = 0; p + 3 < num_input; p += 4)
        {
            const float* k00 = k0.row(p);
            const float* k01 = k0.row(p + 1);
            const float* k02 = k0.row(p + 2);
            const float* k03 = k0.row(p + 3);

            for (int k = 0; k < maxk; k++)
            {
                g00[0] = k00[k];
                g00[1] = k01[k];
                g00[2] = k02[k];
                g00[3] = k03[k];

                g00 += 4;
            }
        }
    }
}

static void convolution_pack4to1_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_pack4to1, const Mat& bias_data, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int activation_type, const Mat& activation_params, const Option& opt)
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
        float* outptr = top_blob.channel(p);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                float sum = 0.f;

                if (bias_data_ptr)
                {
                    sum = bias_data_ptr[p];
                }

                const float* kptr = (const float*)weight_data_pack4to1 + maxk * channels * p * 4;

                // channels
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob.channel(q);
                    const float* sptr = m.row(i * stride_h) + j * stride_w * 4;

                    for (int k = 0; k < maxk; k++) // 29.23
                    {
                        float32x4_t _val = vld1q_f32(sptr + space_ofs[k] * 4);
                        float32x4_t _w = vld1q_f32(kptr);
                        float32x4_t _s4 = vmulq_f32(_val, _w);
#if __aarch64__
                        sum += vaddvq_f32(_s4); // dot
#else
                        float32x2_t _ss = vadd_f32(vget_low_f32(_s4), vget_high_f32(_s4));
                        _ss = vpadd_f32(_ss, _ss);
                        sum += vget_lane_f32(_ss, 0);
#endif

                        kptr += 4;
                    }
                }

                sum = activation_ss(sum, activation_type, activation_params);

                outptr[j] = sum;
            }

            outptr += outw;
        }
    }
}
