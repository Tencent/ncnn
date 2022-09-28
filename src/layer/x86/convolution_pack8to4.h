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

static void convolution_pack8to4_avx(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_packed, const Mat& bias_data, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int activation_type, const Mat& activation_params, const Option& opt)
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
                __m128 _sum = _mm_setzero_ps();

                if (bias_data_ptr)
                {
                    _sum = _mm_loadu_ps(bias_data_ptr + p * 4);
                }

                const float* kptr = weight_data_packed.channel(p);

                // channels
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob.channel(q);
                    const float* sptr = m.row(i * stride_h) + j * stride_w * 8;

                    for (int k = 0; k < maxk; k++)
                    {
                        const float* slptr = sptr + space_ofs[k] * 8;

                        __m128 _val0 = _mm_broadcast_ss(slptr);
                        __m128 _val1 = _mm_broadcast_ss(slptr + 1);
                        __m128 _val2 = _mm_broadcast_ss(slptr + 2);
                        __m128 _val3 = _mm_broadcast_ss(slptr + 3);
                        __m128 _val4 = _mm_broadcast_ss(slptr + 4);
                        __m128 _val5 = _mm_broadcast_ss(slptr + 5);
                        __m128 _val6 = _mm_broadcast_ss(slptr + 6);
                        __m128 _val7 = _mm_broadcast_ss(slptr + 7);

                        __m128 _w0 = _mm_load_ps(kptr);
                        __m128 _w1 = _mm_load_ps(kptr + 4);
                        _sum = _mm_comp_fmadd_ps(_val0, _w0, _sum);
                        _sum = _mm_comp_fmadd_ps(_val1, _w1, _sum);
                        __m128 _w2 = _mm_load_ps(kptr + 8);
                        __m128 _w3 = _mm_load_ps(kptr + 12);
                        _sum = _mm_comp_fmadd_ps(_val2, _w2, _sum);
                        _sum = _mm_comp_fmadd_ps(_val3, _w3, _sum);
                        __m128 _w4 = _mm_load_ps(kptr + 16);
                        __m128 _w5 = _mm_load_ps(kptr + 20);
                        _sum = _mm_comp_fmadd_ps(_val4, _w4, _sum);
                        _sum = _mm_comp_fmadd_ps(_val5, _w5, _sum);
                        __m128 _w6 = _mm_load_ps(kptr + 24);
                        __m128 _w7 = _mm_load_ps(kptr + 28);
                        _sum = _mm_comp_fmadd_ps(_val6, _w6, _sum);
                        _sum = _mm_comp_fmadd_ps(_val7, _w7, _sum);

                        kptr += 32;
                    }
                }

                _sum = activation_sse(_sum, activation_type, activation_params);

                _mm_storeu_ps(outptr + j * 4, _sum);
            }

            outptr += outw * 4;
        }
    }
}
