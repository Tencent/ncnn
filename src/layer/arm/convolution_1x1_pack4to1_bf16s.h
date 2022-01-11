// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
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

static void conv1x1s1_sgemm_pack4to1_bf16s_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    const int size = w * h;

    Mat bottom_im2col = bottom_blob;
    bottom_im2col.w = size;
    bottom_im2col.h = 1;

    im2col_sgemm_pack4to1_bf16s_neon(bottom_im2col, top_blob, kernel, _bias, opt);
}

static void conv1x1s2_sgemm_pack4to1_bf16s_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int tailstep = (w - 2 * outw + w) * 4;

    Mat bottom_blob_shrinked;
    bottom_blob_shrinked.create(outw, outh, channels, elemsize, elempack, opt.workspace_allocator);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < channels; p++)
    {
        const unsigned short* r0 = bottom_blob.channel(p);
        unsigned short* outptr = bottom_blob_shrinked.channel(p);

        for (int i = 0; i < outh; i++)
        {
            int j = 0;
            for (; j + 3 < outw; j += 4)
            {
                uint16x4_t _v0 = vld1_u16(r0);
                uint16x4_t _v1 = vld1_u16(r0 + 8);
                uint16x4_t _v2 = vld1_u16(r0 + 16);
                uint16x4_t _v3 = vld1_u16(r0 + 24);
                uint16x8_t _v01 = vcombine_u16(_v0, _v1);
                uint16x8_t _v23 = vcombine_u16(_v2, _v3);
                vst1q_u16(outptr, _v01);
                vst1q_u16(outptr + 8, _v23);

                r0 += 32;
                outptr += 16;
            }
            for (; j + 1 < outw; j += 2)
            {
                uint16x4_t _v0 = vld1_u16(r0);
                uint16x4_t _v1 = vld1_u16(r0 + 8);
                uint16x8_t _v = vcombine_u16(_v0, _v1);
                vst1q_u16(outptr, _v);

                r0 += 16;
                outptr += 8;
            }
            for (; j < outw; j++)
            {
                uint16x4_t _v = vld1_u16(r0);
                vst1_u16(outptr, _v);

                r0 += 8;
                outptr += 4;
            }

            r0 += tailstep;
        }
    }

    conv1x1s1_sgemm_pack4to1_bf16s_neon(bottom_blob_shrinked, top_blob, kernel, _bias, opt);
}
