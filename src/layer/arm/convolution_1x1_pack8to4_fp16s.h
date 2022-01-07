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

static void conv1x1s1_sgemm_pack8to4_fp16sa_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    const int size = w * h;

    Mat bottom_im2col = bottom_blob;
    bottom_im2col.w = size;
    bottom_im2col.h = 1;

    im2col_sgemm_pack8to4_fp16sa_neon(bottom_im2col, top_blob, kernel, _bias, opt);
}

static void conv1x1s2_sgemm_pack8to4_fp16sa_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int tailstep = (w - 2 * outw + w) * 8;

    Mat bottom_blob_shrinked;
    bottom_blob_shrinked.create(outw, outh, channels, elemsize, elempack, opt.workspace_allocator);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < channels; p++)
    {
        const __fp16* r0 = bottom_blob.channel(p);
        __fp16* outptr = bottom_blob_shrinked.channel(p);

        for (int i = 0; i < outh; i++)
        {
            int j = 0;
            for (; j + 3 < outw; j += 4)
            {
                float16x8_t _v0 = vld1q_f16(r0);
                float16x8_t _v1 = vld1q_f16(r0 + 16);
                float16x8_t _v2 = vld1q_f16(r0 + 32);
                float16x8_t _v3 = vld1q_f16(r0 + 48);
                vst1q_f16(outptr, _v0);
                vst1q_f16(outptr + 8, _v1);
                vst1q_f16(outptr + 16, _v2);
                vst1q_f16(outptr + 24, _v3);

                r0 += 64;
                outptr += 32;
            }
            for (; j + 1 < outw; j += 2)
            {
                float16x8_t _v0 = vld1q_f16(r0);
                float16x8_t _v1 = vld1q_f16(r0 + 16);
                vst1q_f16(outptr, _v0);
                vst1q_f16(outptr + 8, _v1);

                r0 += 32;
                outptr += 16;
            }
            for (; j < outw; j++)
            {
                float16x8_t _v = vld1q_f16(r0);
                vst1q_f16(outptr, _v);

                r0 += 16;
                outptr += 8;
            }

            r0 += tailstep;
        }
    }

    conv1x1s1_sgemm_pack8to4_fp16sa_neon(bottom_blob_shrinked, top_blob, kernel, _bias, opt);
}
