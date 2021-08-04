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

static void convolution_pack1to4_int8_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_int8, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
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
                int32x4_t _sum0 = vdupq_n_s32(0);

                const signed char* kptr = weight_data_int8.channel(p);

                // channels
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob.channel(q);
                    const signed char* sptr = m.row<const signed char>(i * stride_h) + j * stride_w;

                    for (int k = 0; k < maxk; k++)
                    {
                        int8x8_t _val = vdup_n_s8(sptr[space_ofs[k]]);
                        int8x8_t _w = vld1_s8(kptr);
                        int16x8_t _s0 = vmull_s8(_val, _w);
                        _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));

                        kptr += 8;
                    }
                }

                vst1q_s32(outptr + j * 4, _sum0);
            }

            outptr += outw * 4;
        }
    }
}
