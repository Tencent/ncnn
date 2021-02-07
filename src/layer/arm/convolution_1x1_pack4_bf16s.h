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

static void conv1x1s1_sgemm_transform_kernel_pack4_bf16s_neon(const Mat& kernel, Mat& kernel_tm_pack4, int inch, int outch)
{
    // interleave
    // src = inch-outch
    // dst = 4b-4a-inch/4a-outch/4b
#if __aarch64__
    kernel_tm_pack4.create(2 * 1, inch / 4, (outch / 4) / 2 + (outch / 4) % 2, (size_t)2u * 16, 16);
#else
    kernel_tm_pack4.create(1, inch / 4, outch / 4, (size_t)2u * 16, 16);
#endif

    int q = 0;
#if __aarch64__
    for (; q + 7 < outch; q += 8)
    {
        const float* k0 = (const float*)kernel + (q + 0) * inch;
        const float* k1 = (const float*)kernel + (q + 1) * inch;
        const float* k2 = (const float*)kernel + (q + 2) * inch;
        const float* k3 = (const float*)kernel + (q + 3) * inch;
        const float* k4 = (const float*)kernel + (q + 4) * inch;
        const float* k5 = (const float*)kernel + (q + 5) * inch;
        const float* k6 = (const float*)kernel + (q + 6) * inch;
        const float* k7 = (const float*)kernel + (q + 7) * inch;

        unsigned short* g0 = kernel_tm_pack4.channel(q / 8);

        for (int p = 0; p + 3 < inch; p += 4)
        {
            g0[0] = float32_to_bfloat16(k0[0]);
            g0[1] = float32_to_bfloat16(k1[0]);
            g0[2] = float32_to_bfloat16(k2[0]);
            g0[3] = float32_to_bfloat16(k3[0]);

            g0[4] = float32_to_bfloat16(k4[0]);
            g0[5] = float32_to_bfloat16(k5[0]);
            g0[6] = float32_to_bfloat16(k6[0]);
            g0[7] = float32_to_bfloat16(k7[0]);

            g0[8] = float32_to_bfloat16(k0[1]);
            g0[9] = float32_to_bfloat16(k1[1]);
            g0[10] = float32_to_bfloat16(k2[1]);
            g0[11] = float32_to_bfloat16(k3[1]);

            g0[12] = float32_to_bfloat16(k4[1]);
            g0[13] = float32_to_bfloat16(k5[1]);
            g0[14] = float32_to_bfloat16(k6[1]);
            g0[15] = float32_to_bfloat16(k7[1]);

            g0[16] = float32_to_bfloat16(k0[2]);
            g0[17] = float32_to_bfloat16(k1[2]);
            g0[18] = float32_to_bfloat16(k2[2]);
            g0[19] = float32_to_bfloat16(k3[2]);

            g0[20] = float32_to_bfloat16(k4[2]);
            g0[21] = float32_to_bfloat16(k5[2]);
            g0[22] = float32_to_bfloat16(k6[2]);
            g0[23] = float32_to_bfloat16(k7[2]);

            g0[24] = float32_to_bfloat16(k0[3]);
            g0[25] = float32_to_bfloat16(k1[3]);
            g0[26] = float32_to_bfloat16(k2[3]);
            g0[27] = float32_to_bfloat16(k3[3]);

            g0[28] = float32_to_bfloat16(k4[3]);
            g0[29] = float32_to_bfloat16(k5[3]);
            g0[30] = float32_to_bfloat16(k6[3]);
            g0[31] = float32_to_bfloat16(k7[3]);

            k0 += 4;
            k1 += 4;
            k2 += 4;
            k3 += 4;
            k4 += 4;
            k5 += 4;
            k6 += 4;
            k7 += 4;
            g0 += 32;
        }
    }
#endif // __aarch64__
    for (; q + 3 < outch; q += 4)
    {
        const float* k0 = (const float*)kernel + (q + 0) * inch;
        const float* k1 = (const float*)kernel + (q + 1) * inch;
        const float* k2 = (const float*)kernel + (q + 2) * inch;
        const float* k3 = (const float*)kernel + (q + 3) * inch;

#if __aarch64__
        unsigned short* g0 = kernel_tm_pack4.channel(q / 8 + (q % 8) / 4);
#else
        unsigned short* g0 = kernel_tm_pack4.channel(q / 4);
#endif

        for (int p = 0; p + 3 < inch; p += 4)
        {
            g0[0] = float32_to_bfloat16(k0[0]);
            g0[1] = float32_to_bfloat16(k1[0]);
            g0[2] = float32_to_bfloat16(k2[0]);
            g0[3] = float32_to_bfloat16(k3[0]);

            g0[4] = float32_to_bfloat16(k0[1]);
            g0[5] = float32_to_bfloat16(k1[1]);
            g0[6] = float32_to_bfloat16(k2[1]);
            g0[7] = float32_to_bfloat16(k3[1]);

            g0[8] = float32_to_bfloat16(k0[2]);
            g0[9] = float32_to_bfloat16(k1[2]);
            g0[10] = float32_to_bfloat16(k2[2]);
            g0[11] = float32_to_bfloat16(k3[2]);

            g0[12] = float32_to_bfloat16(k0[3]);
            g0[13] = float32_to_bfloat16(k1[3]);
            g0[14] = float32_to_bfloat16(k2[3]);
            g0[15] = float32_to_bfloat16(k3[3]);

            k0 += 4;
            k1 += 4;
            k2 += 4;
            k3 += 4;
            g0 += 16;
        }
    }
}

static void conv1x1s1_sgemm_pack4_bf16s_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    const int size = w * h;

    Mat bottom_im2col = bottom_blob;
    bottom_im2col.w = size;
    bottom_im2col.h = 1;

    im2col_sgemm_pack4_bf16s_neon(bottom_im2col, top_blob, kernel, _bias, opt);
}

static void conv1x1s2_pack4_bf16s_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
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

    conv1x1s1_sgemm_pack4_bf16s_neon(bottom_blob_shrinked, top_blob, kernel, _bias, opt);
}
