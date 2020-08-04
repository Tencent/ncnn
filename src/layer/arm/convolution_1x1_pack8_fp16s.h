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

static void conv1x1s1_sgemm_transform_kernel_pack8_fp16sa_neon(const Mat& kernel, Mat& kernel_tm_pack8, int inch, int outch)
{
    // interleave
    // src = inch-outch
    // dst = 8b-8a-inch/8a-outch/8b
    kernel_tm_pack8.create(1, inch / 8, outch / 8, (size_t)2u * 64, 64);

    int q = 0;
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

        __fp16* g0 = kernel_tm_pack8.channel(q / 8);

        for (int p = 0; p + 7 < inch; p += 8)
        {
            for (int i = 0; i < 8; i++)
            {
                g0[0] = (__fp16)k0[i];
                g0[1] = (__fp16)k1[i];
                g0[2] = (__fp16)k2[i];
                g0[3] = (__fp16)k3[i];
                g0[4] = (__fp16)k4[i];
                g0[5] = (__fp16)k5[i];
                g0[6] = (__fp16)k6[i];
                g0[7] = (__fp16)k7[i];

                g0 += 8;
            }

            k0 += 8;
            k1 += 8;
            k2 += 8;
            k3 += 8;
            k4 += 8;
            k5 += 8;
            k6 += 8;
            k7 += 8;
        }
    }
}

static void conv1x1s1_sgemm_pack8_fp16sa_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;
    int outch = top_blob.c;

    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int size = w * h;

    const __fp16* bias = _bias;

    // interleave
    Mat tmp;
    if (size >= 8)
        tmp.create(8, inch, size / 8 + (size % 8) / 4 + (size % 4) / 2 + size % 2, elemsize, elempack, opt.workspace_allocator);
    else if (size >= 4)
        tmp.create(4, inch, size / 4 + (size % 4) / 2 + size % 2, elemsize, elempack, opt.workspace_allocator);
    else if (size >= 2)
        tmp.create(2, inch, size / 2 + size % 2, elemsize, elempack, opt.workspace_allocator);
    else // if (size >= 1)
        tmp.create(1, inch, size, elemsize, elempack, opt.workspace_allocator);
    {
        int nn_size;
        int remain_size_start;

        remain_size_start = 0;

        nn_size = (size - remain_size_start) >> 3;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 8;

            const __fp16* img0 = bottom_blob.channel(0);
            img0 += i * 8;

            __fp16* tmpptr = tmp.channel(i / 8);

            for (int q = 0; q < inch; q++)
            {
                float16x8_t _v0 = vld1q_f16(img0);
                float16x8_t _v1 = vld1q_f16(img0 + 8);
                float16x8_t _v2 = vld1q_f16(img0 + 16);
                float16x8_t _v3 = vld1q_f16(img0 + 24);
                float16x8_t _v4 = vld1q_f16(img0 + 32);
                float16x8_t _v5 = vld1q_f16(img0 + 40);
                float16x8_t _v6 = vld1q_f16(img0 + 48);
                float16x8_t _v7 = vld1q_f16(img0 + 56);
                vst1q_f16(tmpptr, _v0);
                vst1q_f16(tmpptr + 8, _v1);
                vst1q_f16(tmpptr + 16, _v2);
                vst1q_f16(tmpptr + 24, _v3);
                vst1q_f16(tmpptr + 32, _v4);
                vst1q_f16(tmpptr + 40, _v5);
                vst1q_f16(tmpptr + 48, _v6);
                vst1q_f16(tmpptr + 56, _v7);

                tmpptr += 64;
                img0 += bottom_blob.cstep * 8;
            }
        }

        remain_size_start += nn_size << 3;
        nn_size = (size - remain_size_start) >> 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 4;

            const __fp16* img0 = bottom_blob.channel(0);
            img0 += i * 8;

            __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);

            for (int q = 0; q < inch; q++)
            {
                float16x8_t _v0 = vld1q_f16(img0);
                float16x8_t _v1 = vld1q_f16(img0 + 8);
                float16x8_t _v2 = vld1q_f16(img0 + 16);
                float16x8_t _v3 = vld1q_f16(img0 + 24);
                vst1q_f16(tmpptr, _v0);
                vst1q_f16(tmpptr + 8, _v1);
                vst1q_f16(tmpptr + 16, _v2);
                vst1q_f16(tmpptr + 24, _v3);

                tmpptr += 32;
                img0 += bottom_blob.cstep * 8;
            }
        }

        remain_size_start += nn_size << 2;
        nn_size = (size - remain_size_start) >> 1;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 2;

            const __fp16* img0 = bottom_blob.channel(0);
            img0 += i * 8;

            __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2);

            for (int q = 0; q < inch; q++)
            {
                float16x8_t _v0 = vld1q_f16(img0);
                float16x8_t _v1 = vld1q_f16(img0 + 8);
                vst1q_f16(tmpptr, _v0);
                vst1q_f16(tmpptr + 8, _v1);

                tmpptr += 16;
                img0 += bottom_blob.cstep * 8;
            }
        }

        remain_size_start += nn_size << 1;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < size; i++)
        {
            const __fp16* img0 = bottom_blob.channel(0);
            img0 += i * 8;

            __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);

            for (int q = 0; q < inch; q++)
            {
                float16x8_t _v = vld1q_f16(img0);
                vst1q_f16(tmpptr, _v);

                tmpptr += 8;
                img0 += bottom_blob.cstep * 8;
            }
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        __fp16* outptr0 = top_blob.channel(p);

        const __fp16 zeros[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        const __fp16* biasptr = bias ? bias + p * 8 : zeros;

        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            __fp16* tmpptr = tmp.channel(i / 8);
            const __fp16* kptr0 = kernel.channel(p);

            float16x8_t _sum0 = vld1q_f16(biasptr);
            float16x8_t _sum1 = vld1q_f16(biasptr);
            float16x8_t _sum2 = vld1q_f16(biasptr);
            float16x8_t _sum3 = vld1q_f16(biasptr);
            float16x8_t _sum4 = vld1q_f16(biasptr);
            float16x8_t _sum5 = vld1q_f16(biasptr);
            float16x8_t _sum6 = vld1q_f16(biasptr);
            float16x8_t _sum7 = vld1q_f16(biasptr);

            for (int q=0; q<inch; q++)
            {
                float16x8_t _v0 = vld1q_f16(tmpptr);
                float16x8_t _v1 = vld1q_f16(tmpptr + 8);
                float16x8_t _v2 = vld1q_f16(tmpptr + 16);
                float16x8_t _v3 = vld1q_f16(tmpptr + 24);
                float16x8_t _v4 = vld1q_f16(tmpptr + 32);
                float16x8_t _v5 = vld1q_f16(tmpptr + 40);
                float16x8_t _v6 = vld1q_f16(tmpptr + 48);
                float16x8_t _v7 = vld1q_f16(tmpptr + 56);

                float16x8_t _k0 = vld1q_f16(kptr0);
                float16x8_t _k1 = vld1q_f16(kptr0 + 8);
                float16x8_t _k2 = vld1q_f16(kptr0 + 16);
                float16x8_t _k3 = vld1q_f16(kptr0 + 24);
                float16x8_t _k4 = vld1q_f16(kptr0 + 32);
                float16x8_t _k5 = vld1q_f16(kptr0 + 40);
                float16x8_t _k6 = vld1q_f16(kptr0 + 48);
                float16x8_t _k7 = vld1q_f16(kptr0 + 56);

                _sum0 = vfmaq_laneq_f16(_sum0, _k0, _v0, 0);
                _sum0 = vfmaq_laneq_f16(_sum0, _k1, _v0, 1);
                _sum0 = vfmaq_laneq_f16(_sum0, _k2, _v0, 2);
                _sum0 = vfmaq_laneq_f16(_sum0, _k3, _v0, 3);
                _sum0 = vfmaq_laneq_f16(_sum0, _k4, _v0, 4);
                _sum0 = vfmaq_laneq_f16(_sum0, _k5, _v0, 5);
                _sum0 = vfmaq_laneq_f16(_sum0, _k6, _v0, 6);
                _sum0 = vfmaq_laneq_f16(_sum0, _k7, _v0, 7);

                _sum1 = vfmaq_laneq_f16(_sum1, _k0, _v1, 0);
                _sum1 = vfmaq_laneq_f16(_sum1, _k1, _v1, 1);
                _sum1 = vfmaq_laneq_f16(_sum1, _k2, _v1, 2);
                _sum1 = vfmaq_laneq_f16(_sum1, _k3, _v1, 3);
                _sum1 = vfmaq_laneq_f16(_sum1, _k4, _v1, 4);
                _sum1 = vfmaq_laneq_f16(_sum1, _k5, _v1, 5);
                _sum1 = vfmaq_laneq_f16(_sum1, _k6, _v1, 6);
                _sum1 = vfmaq_laneq_f16(_sum1, _k7, _v1, 7);

                _sum2 = vfmaq_laneq_f16(_sum2, _k0, _v2, 0);
                _sum2 = vfmaq_laneq_f16(_sum2, _k1, _v2, 1);
                _sum2 = vfmaq_laneq_f16(_sum2, _k2, _v2, 2);
                _sum2 = vfmaq_laneq_f16(_sum2, _k3, _v2, 3);
                _sum2 = vfmaq_laneq_f16(_sum2, _k4, _v2, 4);
                _sum2 = vfmaq_laneq_f16(_sum2, _k5, _v2, 5);
                _sum2 = vfmaq_laneq_f16(_sum2, _k6, _v2, 6);
                _sum2 = vfmaq_laneq_f16(_sum2, _k7, _v2, 7);

                _sum3 = vfmaq_laneq_f16(_sum3, _k0, _v3, 0);
                _sum3 = vfmaq_laneq_f16(_sum3, _k1, _v3, 1);
                _sum3 = vfmaq_laneq_f16(_sum3, _k2, _v3, 2);
                _sum3 = vfmaq_laneq_f16(_sum3, _k3, _v3, 3);
                _sum3 = vfmaq_laneq_f16(_sum3, _k4, _v3, 4);
                _sum3 = vfmaq_laneq_f16(_sum3, _k5, _v3, 5);
                _sum3 = vfmaq_laneq_f16(_sum3, _k6, _v3, 6);
                _sum3 = vfmaq_laneq_f16(_sum3, _k7, _v3, 7);

                _sum4 = vfmaq_laneq_f16(_sum4, _k0, _v4, 0);
                _sum4 = vfmaq_laneq_f16(_sum4, _k1, _v4, 1);
                _sum4 = vfmaq_laneq_f16(_sum4, _k2, _v4, 2);
                _sum4 = vfmaq_laneq_f16(_sum4, _k3, _v4, 3);
                _sum4 = vfmaq_laneq_f16(_sum4, _k4, _v4, 4);
                _sum4 = vfmaq_laneq_f16(_sum4, _k5, _v4, 5);
                _sum4 = vfmaq_laneq_f16(_sum4, _k6, _v4, 6);
                _sum4 = vfmaq_laneq_f16(_sum4, _k7, _v4, 7);

                _sum5 = vfmaq_laneq_f16(_sum5, _k0, _v5, 0);
                _sum5 = vfmaq_laneq_f16(_sum5, _k1, _v5, 1);
                _sum5 = vfmaq_laneq_f16(_sum5, _k2, _v5, 2);
                _sum5 = vfmaq_laneq_f16(_sum5, _k3, _v5, 3);
                _sum5 = vfmaq_laneq_f16(_sum5, _k4, _v5, 4);
                _sum5 = vfmaq_laneq_f16(_sum5, _k5, _v5, 5);
                _sum5 = vfmaq_laneq_f16(_sum5, _k6, _v5, 6);
                _sum5 = vfmaq_laneq_f16(_sum5, _k7, _v5, 7);

                _sum6 = vfmaq_laneq_f16(_sum6, _k0, _v6, 0);
                _sum6 = vfmaq_laneq_f16(_sum6, _k1, _v6, 1);
                _sum6 = vfmaq_laneq_f16(_sum6, _k2, _v6, 2);
                _sum6 = vfmaq_laneq_f16(_sum6, _k3, _v6, 3);
                _sum6 = vfmaq_laneq_f16(_sum6, _k4, _v6, 4);
                _sum6 = vfmaq_laneq_f16(_sum6, _k5, _v6, 5);
                _sum6 = vfmaq_laneq_f16(_sum6, _k6, _v6, 6);
                _sum6 = vfmaq_laneq_f16(_sum6, _k7, _v6, 7);

                _sum7 = vfmaq_laneq_f16(_sum7, _k0, _v7, 0);
                _sum7 = vfmaq_laneq_f16(_sum7, _k1, _v7, 1);
                _sum7 = vfmaq_laneq_f16(_sum7, _k2, _v7, 2);
                _sum7 = vfmaq_laneq_f16(_sum7, _k3, _v7, 3);
                _sum7 = vfmaq_laneq_f16(_sum7, _k4, _v7, 4);
                _sum7 = vfmaq_laneq_f16(_sum7, _k5, _v7, 5);
                _sum7 = vfmaq_laneq_f16(_sum7, _k6, _v7, 6);
                _sum7 = vfmaq_laneq_f16(_sum7, _k7, _v7, 7);

                tmpptr += 64;
                kptr0 += 64;
            }

            vst1q_f16(outptr0, _sum0);
            vst1q_f16(outptr0 + 8, _sum1);
            vst1q_f16(outptr0 + 16, _sum2);
            vst1q_f16(outptr0 + 24, _sum3);
            vst1q_f16(outptr0 + 32, _sum4);
            vst1q_f16(outptr0 + 40, _sum5);
            vst1q_f16(outptr0 + 48, _sum6);
            vst1q_f16(outptr0 + 56, _sum7);

            outptr0 += 64;
        }
        for (; i + 3 < size; i += 4)
        {
            __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
            const __fp16* kptr0 = kernel.channel(p);

            float16x8_t _sum0 = vld1q_f16(biasptr);
            float16x8_t _sum1 = vld1q_f16(biasptr);
            float16x8_t _sum2 = vld1q_f16(biasptr);
            float16x8_t _sum3 = vld1q_f16(biasptr);

            for (int q=0; q<inch; q++)
            {
                float16x8_t _v0 = vld1q_f16(tmpptr);
                float16x8_t _v1 = vld1q_f16(tmpptr + 8);
                float16x8_t _v2 = vld1q_f16(tmpptr + 16);
                float16x8_t _v3 = vld1q_f16(tmpptr + 24);

                float16x8_t _k0 = vld1q_f16(kptr0);
                float16x8_t _k1 = vld1q_f16(kptr0 + 8);
                float16x8_t _k2 = vld1q_f16(kptr0 + 16);
                float16x8_t _k3 = vld1q_f16(kptr0 + 24);
                float16x8_t _k4 = vld1q_f16(kptr0 + 32);
                float16x8_t _k5 = vld1q_f16(kptr0 + 40);
                float16x8_t _k6 = vld1q_f16(kptr0 + 48);
                float16x8_t _k7 = vld1q_f16(kptr0 + 56);

                _sum0 = vfmaq_laneq_f16(_sum0, _k0, _v0, 0);
                _sum0 = vfmaq_laneq_f16(_sum0, _k1, _v0, 1);
                _sum0 = vfmaq_laneq_f16(_sum0, _k2, _v0, 2);
                _sum0 = vfmaq_laneq_f16(_sum0, _k3, _v0, 3);
                _sum0 = vfmaq_laneq_f16(_sum0, _k4, _v0, 4);
                _sum0 = vfmaq_laneq_f16(_sum0, _k5, _v0, 5);
                _sum0 = vfmaq_laneq_f16(_sum0, _k6, _v0, 6);
                _sum0 = vfmaq_laneq_f16(_sum0, _k7, _v0, 7);

                _sum1 = vfmaq_laneq_f16(_sum1, _k0, _v1, 0);
                _sum1 = vfmaq_laneq_f16(_sum1, _k1, _v1, 1);
                _sum1 = vfmaq_laneq_f16(_sum1, _k2, _v1, 2);
                _sum1 = vfmaq_laneq_f16(_sum1, _k3, _v1, 3);
                _sum1 = vfmaq_laneq_f16(_sum1, _k4, _v1, 4);
                _sum1 = vfmaq_laneq_f16(_sum1, _k5, _v1, 5);
                _sum1 = vfmaq_laneq_f16(_sum1, _k6, _v1, 6);
                _sum1 = vfmaq_laneq_f16(_sum1, _k7, _v1, 7);

                _sum2 = vfmaq_laneq_f16(_sum2, _k0, _v2, 0);
                _sum2 = vfmaq_laneq_f16(_sum2, _k1, _v2, 1);
                _sum2 = vfmaq_laneq_f16(_sum2, _k2, _v2, 2);
                _sum2 = vfmaq_laneq_f16(_sum2, _k3, _v2, 3);
                _sum2 = vfmaq_laneq_f16(_sum2, _k4, _v2, 4);
                _sum2 = vfmaq_laneq_f16(_sum2, _k5, _v2, 5);
                _sum2 = vfmaq_laneq_f16(_sum2, _k6, _v2, 6);
                _sum2 = vfmaq_laneq_f16(_sum2, _k7, _v2, 7);

                _sum3 = vfmaq_laneq_f16(_sum3, _k0, _v3, 0);
                _sum3 = vfmaq_laneq_f16(_sum3, _k1, _v3, 1);
                _sum3 = vfmaq_laneq_f16(_sum3, _k2, _v3, 2);
                _sum3 = vfmaq_laneq_f16(_sum3, _k3, _v3, 3);
                _sum3 = vfmaq_laneq_f16(_sum3, _k4, _v3, 4);
                _sum3 = vfmaq_laneq_f16(_sum3, _k5, _v3, 5);
                _sum3 = vfmaq_laneq_f16(_sum3, _k6, _v3, 6);
                _sum3 = vfmaq_laneq_f16(_sum3, _k7, _v3, 7);

                tmpptr += 32;
                kptr0 += 64;
            }

            vst1q_f16(outptr0, _sum0);
            vst1q_f16(outptr0 + 8, _sum1);
            vst1q_f16(outptr0 + 16, _sum2);
            vst1q_f16(outptr0 + 24, _sum3);

            outptr0 += 32;
        }
        for (; i + 1 < size; i += 2)
        {
            __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2);
            const __fp16* kptr0 = kernel.channel(p);

            float16x8_t _sum0 = vld1q_f16(biasptr);
            float16x8_t _sum1 = vld1q_f16(biasptr);

            for (int q=0; q<inch; q++)
            {
                float16x8_t _v0 = vld1q_f16(tmpptr);
                float16x8_t _v1 = vld1q_f16(tmpptr + 8);

                float16x8_t _k0 = vld1q_f16(kptr0);
                float16x8_t _k1 = vld1q_f16(kptr0 + 8);
                float16x8_t _k2 = vld1q_f16(kptr0 + 16);
                float16x8_t _k3 = vld1q_f16(kptr0 + 24);
                float16x8_t _k4 = vld1q_f16(kptr0 + 32);
                float16x8_t _k5 = vld1q_f16(kptr0 + 40);
                float16x8_t _k6 = vld1q_f16(kptr0 + 48);
                float16x8_t _k7 = vld1q_f16(kptr0 + 56);

                _sum0 = vfmaq_laneq_f16(_sum0, _k0, _v0, 0);
                _sum0 = vfmaq_laneq_f16(_sum0, _k1, _v0, 1);
                _sum0 = vfmaq_laneq_f16(_sum0, _k2, _v0, 2);
                _sum0 = vfmaq_laneq_f16(_sum0, _k3, _v0, 3);
                _sum0 = vfmaq_laneq_f16(_sum0, _k4, _v0, 4);
                _sum0 = vfmaq_laneq_f16(_sum0, _k5, _v0, 5);
                _sum0 = vfmaq_laneq_f16(_sum0, _k6, _v0, 6);
                _sum0 = vfmaq_laneq_f16(_sum0, _k7, _v0, 7);

                _sum1 = vfmaq_laneq_f16(_sum1, _k0, _v1, 0);
                _sum1 = vfmaq_laneq_f16(_sum1, _k1, _v1, 1);
                _sum1 = vfmaq_laneq_f16(_sum1, _k2, _v1, 2);
                _sum1 = vfmaq_laneq_f16(_sum1, _k3, _v1, 3);
                _sum1 = vfmaq_laneq_f16(_sum1, _k4, _v1, 4);
                _sum1 = vfmaq_laneq_f16(_sum1, _k5, _v1, 5);
                _sum1 = vfmaq_laneq_f16(_sum1, _k6, _v1, 6);
                _sum1 = vfmaq_laneq_f16(_sum1, _k7, _v1, 7);

                tmpptr += 16;
                kptr0 += 64;
            }

            vst1q_f16(outptr0, _sum0);
            vst1q_f16(outptr0 + 8, _sum1);

            outptr0 += 16;
        }
        for (; i < size; i++)
        {
            __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);
            const __fp16* kptr0 = kernel.channel(p);

            float16x8_t _sum = vld1q_f16(biasptr);

            for (int q=0; q<inch; q++)
            {
                float16x8_t _v = vld1q_f16(tmpptr);

                float16x8_t _k0 = vld1q_f16(kptr0);
                float16x8_t _k1 = vld1q_f16(kptr0 + 8);
                float16x8_t _k2 = vld1q_f16(kptr0 + 16);
                float16x8_t _k3 = vld1q_f16(kptr0 + 24);
                float16x8_t _k4 = vld1q_f16(kptr0 + 32);
                float16x8_t _k5 = vld1q_f16(kptr0 + 40);
                float16x8_t _k6 = vld1q_f16(kptr0 + 48);
                float16x8_t _k7 = vld1q_f16(kptr0 + 56);

                _sum = vfmaq_laneq_f16(_sum, _k0, _v, 0);
                _sum = vfmaq_laneq_f16(_sum, _k1, _v, 1);
                _sum = vfmaq_laneq_f16(_sum, _k2, _v, 2);
                _sum = vfmaq_laneq_f16(_sum, _k3, _v, 3);
                _sum = vfmaq_laneq_f16(_sum, _k4, _v, 4);
                _sum = vfmaq_laneq_f16(_sum, _k5, _v, 5);
                _sum = vfmaq_laneq_f16(_sum, _k6, _v, 6);
                _sum = vfmaq_laneq_f16(_sum, _k7, _v, 7);

                tmpptr += 8;
                kptr0 += 64;
            }

            vst1q_f16(outptr0, _sum);

            outptr0 += 8;
        }
    }

    //     // NOTE sgemm
    //     for (; p<outch; p++)
    //     {
    //         Mat out0 = top_blob.channel(p);
    //
    //         const __fp16 bias0 = bias ? bias[p] : 0.f;
    //
    //         __fp16* outptr0 = out0;
    //
    //         for (int i=0; i<size; i++)
    //         {
    //             __fp16 sum = bias0;
    //
    //             const __fp16* kptr = _kernel.channel(p);
    //
    //             for (int q=0; q<inch; q++)
    //             {
    //                 const __fp16* img0 = bottom_blob.channel(q);
    //
    //                 sum += img0[i] * kptr[0];
    //                 kptr ++;
    //             }
    //
    //             outptr0[i] = sum;
    //         }
    //     }
}

static void conv1x1s2_pack8_fp16sa_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
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

    conv1x1s1_sgemm_pack8_fp16sa_neon(bottom_blob_shrinked, top_blob, kernel, _bias, opt);
}
