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

static void convolution_pack8to1_int8_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_int8, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
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

    // num_output
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        int* outptr = top_blob.channel(p);

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                int sum = 0;

                const signed char* kptr = weight_data_int8.channel(p);

                // channels
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob.channel(q);
                    const signed char* sptr = m.row<const signed char>(i * stride_h) + j * stride_w * 8;

                    for (int k = 0; k < maxk; k++)
                    {
                        // TODO use _mm_cvtepi8_epi16 on sse4.1
                        __m128i _val = _mm_loadl_epi64((const __m128i*)(sptr + space_ofs[k] * 8));
                        _val = _mm_unpacklo_epi8(_val, _mm_cmpgt_epi8(_mm_setzero_si128(), _val));

                        __m128i _w = _mm_loadl_epi64((const __m128i*)kptr);
                        _w = _mm_unpacklo_epi8(_w, _mm_cmpgt_epi8(_mm_setzero_si128(), _w));

                        __m128i _sl = _mm_mullo_epi16(_val, _w);
                        __m128i _sh = _mm_mulhi_epi16(_val, _w);
                        __m128i _s0 = _mm_unpacklo_epi16(_sl, _sh);
                        __m128i _s1 = _mm_unpackhi_epi16(_sl, _sh);

                        __m128i _s4 = _mm_add_epi32(_s0, _s1);

                        // TODO use _mm_hadd_epi32 on ssse3
                        int s4[4];
                        _mm_storeu_si128((__m128i*)s4, _s4);
                        sum += s4[0] + s4[1] + s4[2] + s4[3]; // dot

                        kptr += 8;
                    }
                }

                outptr[j] = sum;
            }

            outptr += outw;
        }
    }
}
