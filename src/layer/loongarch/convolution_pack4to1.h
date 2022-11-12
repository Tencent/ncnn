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

static void convolution_pack4to1_lsx(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_pack4to1, const Mat& bias_data, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int activation_type, const Mat& activation_params, const Option& opt)
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

                __m128 _sum = (__m128)__lsx_vreplgr2vr_w(0);

                const float* kptr = (const float*)weight_data_pack4to1.channel(p);

                // channels
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob.channel(q);
                    const float* sptr = m.row(i * stride_h) + j * stride_w * 4;

                    for (int k = 0; k < maxk; k++)
                    {
                        __m128 _val = (__m128)__lsx_vld(sptr + space_ofs[k] * 4, 0);
                        __m128 _w = (__m128)__lsx_vld(kptr, 0);
                        _sum = __lsx_vfmadd_s(_w, _val, _sum);

                        kptr += 4;
                    }
                }

                sum += __lsx_reduce_fadd_s(_sum);

                sum = activation_ss(sum, activation_type, activation_params);

                outptr[j] = sum;
            }

            outptr += outw;
        }
    }
}
