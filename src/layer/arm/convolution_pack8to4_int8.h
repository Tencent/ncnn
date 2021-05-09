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

static void convolution_pack8to4_int8_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_int8, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
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
                int32x4_t _sum01 = vdupq_n_s32(0);
                int32x4_t _sum23 = vdupq_n_s32(0);

                const signed char* kptr = weight_data_int8.channel(p);

                // channels
                for (int q = 0; q < channels; q++)
                {
                    const Mat m = bottom_blob.channel(q);
                    const signed char* sptr = m.row<signed char>(i * stride_h) + j * stride_w * 8;

                    for (int k = 0; k < maxk; k++)
                    {
                        int8x8_t _val = vld1_s8(sptr + space_ofs[k] * 8);

                        int8x8_t _w0 = vld1_s8(kptr);
                        int8x8_t _w1 = vld1_s8(kptr + 8);
                        int8x8_t _w2 = vld1_s8(kptr + 16);
                        int8x8_t _w3 = vld1_s8(kptr + 24);

                        int16x8_t _wv0 = vmull_s8(_val, _w0);
                        int16x8_t _wv1 = vmull_s8(_val, _w1);
                        int16x8_t _wv2 = vmull_s8(_val, _w2);
                        int16x8_t _wv3 = vmull_s8(_val, _w3);

                        int16x4_t _wv00 = vpadd_s16(vget_low_s16(_wv0), vget_high_s16(_wv0));
                        int16x4_t _wv11 = vpadd_s16(vget_low_s16(_wv1), vget_high_s16(_wv1));
                        int16x4_t _wv22 = vpadd_s16(vget_low_s16(_wv2), vget_high_s16(_wv2));
                        int16x4_t _wv33 = vpadd_s16(vget_low_s16(_wv3), vget_high_s16(_wv3));

                        _sum01 = vpadalq_s16(_sum01, vcombine_s16(_wv00, _wv11));
                        _sum23 = vpadalq_s16(_sum23, vcombine_s16(_wv22, _wv33));

                        kptr += 32;
                    }
                }

                int32x4_t _sum0 = vcombine_s32(vpadd_s32(vget_low_s32(_sum01), vget_high_s32(_sum01)), vpadd_s32(vget_low_s32(_sum23), vget_high_s32(_sum23)));

                vst1q_s32(outptr + j * 4, _sum0);
            }

            outptr += outw * 4;
        }
    }
}
