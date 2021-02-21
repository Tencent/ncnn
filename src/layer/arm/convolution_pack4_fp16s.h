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

static void convolution_pack4_fp16s_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_fp16, const Mat& bias_data, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int activation_type, const Mat& activation_params, const Option& opt)
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
        __fp16* outptr = top_blob.channel(p);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                float32x4_t _sum = vdupq_n_f32(0.f);

                if (bias_data_ptr)
                {
                    _sum = vld1q_f32(bias_data_ptr + p * 4);
                }

                const __fp16* kptr = weight_data_fp16.channel(p);

                // channels
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob.channel(q);
                    const __fp16* sptr = m.row<const __fp16>(i * stride_h) + j * stride_w * 4;

                    for (int k = 0; k < maxk; k++)
                    {
                        float32x4_t _val = vcvt_f32_f16(vld1_f16(sptr + space_ofs[k] * 4));

                        float32x4_t _w0 = vcvt_f32_f16(vld1_f16(kptr));
                        float32x4_t _w1 = vcvt_f32_f16(vld1_f16(kptr + 4));
                        float32x4_t _w2 = vcvt_f32_f16(vld1_f16(kptr + 8));
                        float32x4_t _w3 = vcvt_f32_f16(vld1_f16(kptr + 12));

                        _sum = vfmaq_laneq_f32(_sum, _w0, _val, 0);
                        _sum = vfmaq_laneq_f32(_sum, _w1, _val, 1);
                        _sum = vfmaq_laneq_f32(_sum, _w2, _val, 2);
                        _sum = vfmaq_laneq_f32(_sum, _w3, _val, 3);

                        kptr += 16;
                    }
                }

                _sum = activation_ps(_sum, activation_type, activation_params);

                vst1_f16(outptr + j * 4, vcvt_f16_f32(_sum));
            }

            outptr += outw * 4;
        }
    }
}

static void convolution_pack4_fp16sa_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_fp16, const Mat& bias_data_fp16, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int activation_type, const Mat& activation_params, const Option& opt)
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

    const __fp16* bias_data_ptr = bias_data_fp16;

    // num_output
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        __fp16* outptr = top_blob.channel(p);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                float16x4_t _sum = vdup_n_f16((__fp16)0.f);

                if (bias_data_ptr)
                {
                    _sum = vld1_f16(bias_data_ptr + p * 4);
                }

                const __fp16* kptr = weight_data_fp16.channel(p);

                // channels
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob.channel(q);
                    const __fp16* sptr = m.row<const __fp16>(i * stride_h) + j * stride_w * 4;

                    for (int k = 0; k < maxk; k++)
                    {
                        float16x4_t _val = vld1_f16(sptr + space_ofs[k] * 4);

                        float16x4_t _w0 = vld1_f16(kptr);
                        float16x4_t _w1 = vld1_f16(kptr + 4);
                        float16x4_t _w2 = vld1_f16(kptr + 8);
                        float16x4_t _w3 = vld1_f16(kptr + 12);

                        _sum = vfma_lane_f16(_sum, _w0, _val, 0);
                        _sum = vfma_lane_f16(_sum, _w1, _val, 1);
                        _sum = vfma_lane_f16(_sum, _w2, _val, 2);
                        _sum = vfma_lane_f16(_sum, _w3, _val, 3);

                        kptr += 16;
                    }
                }

                _sum = activation_ps(_sum, activation_type, activation_params);

                vst1_f16(outptr + j * 4, _sum);
            }

            outptr += outw * 4;
        }
    }
}
