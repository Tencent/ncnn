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

static void deconvolution_packnto1_fp16s_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_fp16, const Mat& bias_data, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int packn = csrr_vlenb() / 2;
    const size_t vl = vsetvl_e16m1(packn);

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    const int maxk = kernel_w * kernel_h;

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
                float sum = 0.f;

                if (bias_data_ptr)
                {
                    sum = bias_data_ptr[p];
                }

                vfloat32m2_t _sum = vfmv_v_f_f32m2(0.f, vl);

                const __fp16* kptr = (const __fp16*)weight_data_fp16 + maxk * channels * p * packn;

                // channels
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob.channel(q);

                    for (int y = 0; y < kernel_h; y++)
                    {
                        int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                        if (sys < 0 || sys % stride_h != 0)
                            continue;

                        int sy = sys / stride_h;
                        if (sy >= h)
                            continue;

                        for (int x = 0; x < kernel_w; x++)
                        {
                            int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                            if (sxs < 0 || sxs % stride_w != 0)
                                continue;

                            int sx = sxs / stride_w;
                            if (sx >= w)
                                continue;

                            const __fp16* sptr = m.row<const __fp16>(sy) + sx * packn;

                            int k = y * kernel_w + x;

                            vfloat16m1_t _val = vle16_v_f16m1(sptr, vl);
                            vfloat16m1_t _w = vle16_v_f16m1(kptr + k * packn, vl);
                            _sum = vfwmacc_vv_f32m2(_sum, _val, _w, vl);
                        }
                    }

                    kptr += maxk * packn;
                }

#if C906
                // TODO
                std::vector<float> ss(packn);
                vse32_v_f32m2((float*)ss.data(), _sum, vl);
                for (int i = 0; i < packn; i++)
                {
                    sum += ss[i];
                }
#else
                sum = vfmv_f_s_f32m1_f32(vfredusum_vs_f32m2_f32m1(vfloat32m1_t(), _sum, vfmv_s_f_f32m1(vfloat32m1_t(), sum, vl), vl));
#endif

                sum = activation_ss(sum, activation_type, activation_params);

                outptr[j] = sum;
            }

            outptr += outw;
        }
    }
}

static void deconvolution_packnto1_fp16sa_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_fp16, const Mat& bias_data_fp16, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int packn = csrr_vlenb() / 2;
    const size_t vl = vsetvl_e16m1(packn);

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    const int maxk = kernel_w * kernel_h;

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
                __fp16 sum = 0.f;

                if (bias_data_ptr)
                {
                    sum = bias_data_ptr[p];
                }

                vfloat16m1_t _sum = vfmv_v_f_f16m1(0.f, vl);

                const __fp16* kptr = (const __fp16*)weight_data_fp16 + maxk * channels * p * packn;

                // channels
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob.channel(q);

                    for (int y = 0; y < kernel_h; y++)
                    {
                        int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                        if (sys < 0 || sys % stride_h != 0)
                            continue;

                        int sy = sys / stride_h;
                        if (sy >= h)
                            continue;

                        for (int x = 0; x < kernel_w; x++)
                        {
                            int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                            if (sxs < 0 || sxs % stride_w != 0)
                                continue;

                            int sx = sxs / stride_w;
                            if (sx >= w)
                                continue;

                            const __fp16* sptr = m.row<const __fp16>(sy) + sx * packn;

                            int k = y * kernel_w + x;

                            vfloat16m1_t _val = vle16_v_f16m1(sptr, vl);
                            vfloat16m1_t _w = vle16_v_f16m1(kptr + k * packn, vl);
                            _sum = vfmacc_vv_f16m1(_sum, _val, _w, vl);
                        }
                    }

                    kptr += maxk * packn;
                }

                sum = vfmv_f_s_f16m1_f16(vfredusum_vs_f16m1_f16m1(vfloat16m1_t(), _sum, vfmv_s_f_f16m1(vfloat16m1_t(), sum, vl), vl));

                sum = activation_ss(sum, activation_type, activation_params);

                outptr[j] = sum;
            }

            outptr += outw;
        }
    }
}
