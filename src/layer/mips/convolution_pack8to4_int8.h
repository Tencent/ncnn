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

static void convolution_pack8to4_int8_msa(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_int8, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
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
                v4i32 _sum0 = __msa_fill_w(0);
                v4i32 _sum1 = __msa_fill_w(0);
                v4i32 _sum2 = __msa_fill_w(0);
                v4i32 _sum3 = __msa_fill_w(0);

                const signed char* kptr = weight_data_int8.channel(p);

                // channels
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob.channel(q);
                    const signed char* sptr = m.row<signed char>(i * stride_h) + j * stride_w * 8;

                    for (int k = 0; k < maxk; k++)
                    {
                        v16i8 _val = __msa_ld_b(sptr + space_ofs[k] * 8, 0);
                        v8i16 _val16 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_val, 0), _val);

                        v16i8 _w01 = __msa_ld_b(kptr, 0);
                        v16i8 _w23 = __msa_ld_b(kptr + 16, 0);
                        v16i8 _extw01 = __msa_clti_s_b(_w01, 0);
                        v16i8 _extw23 = __msa_clti_s_b(_w23, 0);
                        v8i16 _w0 = (v8i16)__msa_ilvr_b(_extw01, _w01);
                        v8i16 _w1 = (v8i16)__msa_ilvl_b(_extw01, _w01);
                        v8i16 _w2 = (v8i16)__msa_ilvr_b(_extw23, _w23);
                        v8i16 _w3 = (v8i16)__msa_ilvl_b(_extw23, _w23);

                        v8i16 _s0 = __msa_mulv_h(_val16, _w0);
                        v8i16 _s1 = __msa_mulv_h(_val16, _w1);
                        v8i16 _s2 = __msa_mulv_h(_val16, _w2);
                        v8i16 _s3 = __msa_mulv_h(_val16, _w3);

                        _sum0 = __msa_addv_w(_sum0, __msa_hadd_s_w(_s0, _s0));
                        _sum1 = __msa_addv_w(_sum1, __msa_hadd_s_w(_s1, _s1));
                        _sum2 = __msa_addv_w(_sum2, __msa_hadd_s_w(_s2, _s2));
                        _sum3 = __msa_addv_w(_sum3, __msa_hadd_s_w(_s3, _s3));

                        kptr += 32;
                    }
                }

                // transpose 4x4
                {
                    v4i32 _tmp0, _tmp1, _tmp2, _tmp3;
                    _tmp0 = __msa_ilvr_w(_sum1, _sum0);
                    _tmp1 = __msa_ilvr_w(_sum3, _sum2);
                    _tmp2 = __msa_ilvl_w(_sum1, _sum0);
                    _tmp3 = __msa_ilvl_w(_sum3, _sum2);
                    _sum0 = (v4i32)__msa_ilvr_d((v2i64)_tmp1, (v2i64)_tmp0);
                    _sum1 = (v4i32)__msa_ilvl_d((v2i64)_tmp1, (v2i64)_tmp0);
                    _sum2 = (v4i32)__msa_ilvr_d((v2i64)_tmp3, (v2i64)_tmp2);
                    _sum3 = (v4i32)__msa_ilvl_d((v2i64)_tmp3, (v2i64)_tmp2);
                }

                _sum0 = __msa_addv_w(_sum0, _sum1);
                _sum2 = __msa_addv_w(_sum2, _sum3);

                _sum0 = __msa_addv_w(_sum0, _sum2);

                __msa_st_w(_sum0, outptr + j * 4, 0);
            }

            outptr += outw * 4;
        }
    }
}
