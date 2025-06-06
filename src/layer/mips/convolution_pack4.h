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

static void convolution_pack4_msa(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_pack4, const Mat& bias_data, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int activation_type, const Mat& activation_params, const Option& opt)
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
                v4f32 _sum = (v4f32)__msa_fill_w(0);

                if (bias_data_ptr)
                {
                    _sum = (v4f32)__msa_ld_w(bias_data_ptr + p * 4, 0);
                }

                const float* kptr = (const float*)weight_data_pack4.channel(p);

                // channels
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob.channel(q);
                    const float* sptr = m.row(i * stride_h) + j * stride_w * 4;

                    for (int k = 0; k < maxk; k++) // 29.23
                    {
                        const float* slptr = sptr + space_ofs[k] * 4;

                        v4f32 _val0 = __msa_fill_w_f32(slptr[0]);
                        v4f32 _val1 = __msa_fill_w_f32(slptr[1]);
                        v4f32 _val2 = __msa_fill_w_f32(slptr[2]);
                        v4f32 _val3 = __msa_fill_w_f32(slptr[3]);

                        v4f32 _w0 = (v4f32)__msa_ld_w(kptr, 0);
                        v4f32 _w1 = (v4f32)__msa_ld_w(kptr + 4, 0);
                        v4f32 _w2 = (v4f32)__msa_ld_w(kptr + 8, 0);
                        v4f32 _w3 = (v4f32)__msa_ld_w(kptr + 12, 0);

                        _sum = __msa_fmadd_w(_sum, _val0, _w0);
                        _sum = __msa_fmadd_w(_sum, _val1, _w1);
                        _sum = __msa_fmadd_w(_sum, _val2, _w2);
                        _sum = __msa_fmadd_w(_sum, _val3, _w3);

                        kptr += 16;
                    }
                }

                _sum = activation_ps(_sum, activation_type, activation_params);

                __msa_st_w((v4i32)_sum, outptr + j * 4, 0);
            }

            outptr += outw * 4;
        }
    }
}
