// yala is pleased to support the open source community by making ncnn available.
//
//
// Copyright (C) 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>. All rights reserved.
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

static void convolution_pack8to4_int8_lsx(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_int8, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
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
                __m128i _sum0 = __lsx_vreplgr2vr_w(0);
                __m128i _sum1 = __lsx_vreplgr2vr_w(0);
                __m128i _sum2 = __lsx_vreplgr2vr_w(0);
                __m128i _sum3 = __lsx_vreplgr2vr_w(0);

                const signed char* kptr = weight_data_int8.channel(p);

                // channels
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob.channel(q);
                    const signed char* sptr = m.row<signed char>(i * stride_h) + j * stride_w * 8;

                    for (int k = 0; k < maxk; k++)
                    {
                        __m128i _val = __lsx_vld(sptr + space_ofs[k] * 8, 0);
                        __m128i _val16 = __lsx_vilvl_b(__lsx_vslti_b(_val, 0), _val);

                        __m128i _w01 = __lsx_vld(kptr, 0);
                        __m128i _w23 = __lsx_vld(kptr + 16, 0);
                        __m128i _extw01 = __lsx_vslti_b(_w01, 0);
                        __m128i _extw23 = __lsx_vslti_b(_w23, 0);
                        __m128i _w0 = __lsx_vilvl_b(_extw01, _w01);
                        __m128i _w1 = __lsx_vilvh_b(_extw01, _w01);
                        __m128i _w2 = __lsx_vilvl_b(_extw23, _w23);
                        __m128i _w3 = __lsx_vilvh_b(_extw23, _w23);

                        __m128i _s0 = __lsx_vmul_h(_val16, _w0);
                        __m128i _s1 = __lsx_vmul_h(_val16, _w1);
                        __m128i _s2 = __lsx_vmul_h(_val16, _w2);
                        __m128i _s3 = __lsx_vmul_h(_val16, _w3);

                        _sum0 = __lsx_vadd_w(_sum0, __lsx_vhaddw_w_h(_s0, _s0));
                        _sum1 = __lsx_vadd_w(_sum1, __lsx_vhaddw_w_h(_s1, _s1));
                        _sum2 = __lsx_vadd_w(_sum2, __lsx_vhaddw_w_h(_s2, _s2));
                        _sum3 = __lsx_vadd_w(_sum3, __lsx_vhaddw_w_h(_s3, _s3));

                        kptr += 32;
                    }
                }

                // transpose 4x4
                {
                    __m128i _tmp0, _tmp1, _tmp2, _tmp3;
                    _tmp0 = __lsx_vilvl_w(_sum1, _sum0);
                    _tmp1 = __lsx_vilvl_w(_sum3, _sum2);
                    _tmp2 = __lsx_vilvh_w(_sum1, _sum0);
                    _tmp3 = __lsx_vilvh_w(_sum3, _sum2);
                    _sum0 = __lsx_vilvl_d(_tmp1, _tmp0);
                    _sum1 = __lsx_vilvh_d(_tmp1, _tmp0);
                    _sum2 = __lsx_vilvl_d(_tmp3, _tmp2);
                    _sum3 = __lsx_vilvh_d(_tmp3, _tmp2);
                }

                _sum0 = __lsx_vadd_w(_sum0, _sum1);
                _sum2 = __lsx_vadd_w(_sum2, _sum3);

                _sum0 = __lsx_vadd_w(_sum0, _sum2);

                __lsx_vst(_sum0, outptr + j * 4, 0);
            }

            outptr += outw * 4;
        }
    }
}
