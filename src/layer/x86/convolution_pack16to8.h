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

static void convolution_pack16to8_avx512(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_packed, const Mat& bias_data, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int activation_type, const Mat& activation_params, const Option& opt)
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
                __m256 _sum = _mm256_setzero_ps();

                if (bias_data_ptr)
                {
                    _sum = _mm256_loadu_ps(bias_data_ptr + p * 8);
                }

                const float* kptr = weight_data_packed.channel(p);

                // channels
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob.channel(q);
                    const float* sptr = m.row(i * stride_h) + j * stride_w * 16;

                    for (int k = 0; k < maxk; k++)
                    {
                        const float* slptr = sptr + space_ofs[k] * 16;

                        __m256 _val0 = _mm256_broadcast_ss(slptr);
                        __m256 _val1 = _mm256_broadcast_ss(slptr + 1);
                        __m256 _val2 = _mm256_broadcast_ss(slptr + 2);
                        __m256 _val3 = _mm256_broadcast_ss(slptr + 3);
                        __m256 _val4 = _mm256_broadcast_ss(slptr + 4);
                        __m256 _val5 = _mm256_broadcast_ss(slptr + 5);
                        __m256 _val6 = _mm256_broadcast_ss(slptr + 6);
                        __m256 _val7 = _mm256_broadcast_ss(slptr + 7);
                        __m256 _val8 = _mm256_broadcast_ss(slptr + 8);
                        __m256 _val9 = _mm256_broadcast_ss(slptr + 9);
                        __m256 _vala = _mm256_broadcast_ss(slptr + 10);
                        __m256 _valb = _mm256_broadcast_ss(slptr + 11);
                        __m256 _valc = _mm256_broadcast_ss(slptr + 12);
                        __m256 _vald = _mm256_broadcast_ss(slptr + 13);
                        __m256 _vale = _mm256_broadcast_ss(slptr + 14);
                        __m256 _valf = _mm256_broadcast_ss(slptr + 15);

                        __m256 _w0 = _mm256_load_ps(kptr + 8 * 0);
                        __m256 _w1 = _mm256_load_ps(kptr + 8 * 1);
                        __m256 _w2 = _mm256_load_ps(kptr + 8 * 2);
                        __m256 _w3 = _mm256_load_ps(kptr + 8 * 3);
                        __m256 _w4 = _mm256_load_ps(kptr + 8 * 4);
                        __m256 _w5 = _mm256_load_ps(kptr + 8 * 5);
                        __m256 _w6 = _mm256_load_ps(kptr + 8 * 6);
                        __m256 _w7 = _mm256_load_ps(kptr + 8 * 7);
                        __m256 _w8 = _mm256_load_ps(kptr + 8 * 8);
                        __m256 _w9 = _mm256_load_ps(kptr + 8 * 9);
                        __m256 _wa = _mm256_load_ps(kptr + 8 * 10);
                        __m256 _wb = _mm256_load_ps(kptr + 8 * 11);
                        __m256 _wc = _mm256_load_ps(kptr + 8 * 12);
                        __m256 _wd = _mm256_load_ps(kptr + 8 * 13);
                        __m256 _we = _mm256_load_ps(kptr + 8 * 14);
                        __m256 _wf = _mm256_load_ps(kptr + 8 * 15);

                        _sum = _mm256_fmadd_ps(_val0, _w0, _sum);
                        _sum = _mm256_fmadd_ps(_val1, _w1, _sum);
                        _sum = _mm256_fmadd_ps(_val2, _w2, _sum);
                        _sum = _mm256_fmadd_ps(_val3, _w3, _sum);
                        _sum = _mm256_fmadd_ps(_val4, _w4, _sum);
                        _sum = _mm256_fmadd_ps(_val5, _w5, _sum);
                        _sum = _mm256_fmadd_ps(_val6, _w6, _sum);
                        _sum = _mm256_fmadd_ps(_val7, _w7, _sum);
                        _sum = _mm256_fmadd_ps(_val8, _w8, _sum);
                        _sum = _mm256_fmadd_ps(_val9, _w9, _sum);
                        _sum = _mm256_fmadd_ps(_vala, _wa, _sum);
                        _sum = _mm256_fmadd_ps(_valb, _wb, _sum);
                        _sum = _mm256_fmadd_ps(_valc, _wc, _sum);
                        _sum = _mm256_fmadd_ps(_vald, _wd, _sum);
                        _sum = _mm256_fmadd_ps(_vale, _we, _sum);
                        _sum = _mm256_fmadd_ps(_valf, _wf, _sum);

                        kptr += 128;
                    }
                }

                _sum = activation_avx(_sum, activation_type, activation_params);

                _mm256_storeu_ps(outptr, _sum);
                outptr += 8;
            }
        }
    }
}
