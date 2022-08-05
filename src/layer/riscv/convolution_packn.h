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

static void convolution_packn_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_packn, const Mat& bias_data, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int packn = csrr_vlenb() / 4;
    const size_t vl = vsetvl_e32m1(packn);

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
                vfloat32m1_t _sum = vfmv_v_f_f32m1(0.f, vl);

                if (bias_data_ptr)
                {
                    _sum = vle32_v_f32m1(bias_data_ptr + p * packn, vl);
                }

                const float* kptr = (const float*)weight_data_packn.channel(p);

                // channels
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob.channel(q);
                    const float* sptr = m.row(i * stride_h) + j * stride_w * packn;

                    for (int k = 0; k < maxk; k++) // 29.23
                    {
                        const float* slptr = sptr + space_ofs[k] * packn;

                        for (int l = 0; l < packn; l++)
                        {
                            float val = *slptr++;
                            vfloat32m1_t _w0 = vle32_v_f32m1(kptr, vl);
                            _sum = vfmacc_vf_f32m1(_sum, val, _w0, vl);

                            kptr += packn;
                        }
                    }
                }

                _sum = activation_ps(_sum, activation_type, activation_params, vl);

                vse32_v_f32m1(outptr + j * packn, _sum, vl);
            }

            outptr += outw * packn;
        }
    }
}
