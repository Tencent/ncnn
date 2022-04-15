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

static void convolution_pack4to16_avx512(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_packed, const Mat& bias_data, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int activation_type, const Mat& activation_params, const Option& opt)
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
                __m512 _sum = _mm512_setzero_ps();

                if (bias_data_ptr)
                {
                    _sum = _mm512_loadu_ps(bias_data_ptr + p * 16);
                }

                const float* kptr = weight_data_packed.channel(p);

                // channels
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob.channel(q);
                    const float* sptr = m.row(i * stride_h) + j * stride_w * 4;

                    for (int k = 0; k < maxk; k++)
                    {
                        const float* slptr = sptr + space_ofs[k] * 4;

                        __m512 _val0 = _mm512_set1_ps(slptr[0]);
                        __m512 _val1 = _mm512_set1_ps(slptr[1]);
                        __m512 _val2 = _mm512_set1_ps(slptr[2]);
                        __m512 _val3 = _mm512_set1_ps(slptr[3]);

                        __m512 _w0 = _mm512_load_ps(kptr);
                        __m512 _w1 = _mm512_load_ps(kptr + 16);
                        __m512 _w2 = _mm512_load_ps(kptr + 32);
                        __m512 _w3 = _mm512_load_ps(kptr + 48);
                        _sum = _mm512_fmadd_ps(_val0, _w0, _sum);
                        _sum = _mm512_fmadd_ps(_val1, _w1, _sum);
                        _sum = _mm512_fmadd_ps(_val2, _w2, _sum);
                        _sum = _mm512_fmadd_ps(_val3, _w3, _sum);

                        kptr += 64;
                    }
                }

                _sum = activation_avx512(_sum, activation_type, activation_params);

                _mm512_store_ps(outptr, _sum);
                outptr += 16;
            }
        }
    }
}
