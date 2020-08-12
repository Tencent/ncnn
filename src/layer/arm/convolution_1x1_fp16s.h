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

static void conv1x1s1_sgemm_transform_kernel_fp16sa_neon(const Mat& _kernel, Mat& kernel_tm, int inch, int outch)
{
    const float* kernel = _kernel;

    // interleave
    kernel_tm.create(8 * 8, inch / 8 + inch % 8, outch / 8 + outch % 8, (size_t)2u, 1);

    int p = 0;
    for (; p + 7 < outch; p += 8)
    {
        const float* kernel0 = kernel + (p + 0) * inch;
        const float* kernel1 = kernel + (p + 1) * inch;
        const float* kernel2 = kernel + (p + 2) * inch;
        const float* kernel3 = kernel + (p + 3) * inch;
        const float* kernel4 = kernel + (p + 4) * inch;
        const float* kernel5 = kernel + (p + 5) * inch;
        const float* kernel6 = kernel + (p + 6) * inch;
        const float* kernel7 = kernel + (p + 7) * inch;

        __fp16* ktmp = kernel_tm.channel(p / 8);

        for (int q = 0; q < inch; q++)
        {
            // kernel0...7 0
            ktmp[0] = (__fp16)kernel0[0];
            ktmp[1] = (__fp16)kernel1[0];
            ktmp[2] = (__fp16)kernel2[0];
            ktmp[3] = (__fp16)kernel3[0];
            ktmp[4] = (__fp16)kernel4[0];
            ktmp[5] = (__fp16)kernel5[0];
            ktmp[6] = (__fp16)kernel6[0];
            ktmp[7] = (__fp16)kernel7[0];

            ktmp += 8;
            kernel0 += 1;
            kernel1 += 1;
            kernel2 += 1;
            kernel3 += 1;
            kernel4 += 1;
            kernel5 += 1;
            kernel6 += 1;
            kernel7 += 1;
        }
    }
    for (; p < outch; p++)
    {
        const float* kernel0 = kernel + p * inch;

        __fp16* ktmp = kernel_tm.channel(p / 8 + p % 8);

        for (int q = 0; q < inch; q++)
        {
            ktmp[0] = (__fp16)kernel0[0];
            ktmp++;
            kernel0++;
        }
    }
}

static void conv1x1s1_sgemm_fp16sa_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;
    int outch = top_blob.c;

    const int size = w * h;

    const __fp16* bias = _bias;

    // interleave
    Mat tmp(8 * 8, inch / 8 + inch % 8, size / 8 + size % 8, 2u, opt.workspace_allocator);
    {
        int nn_size = size >> 3;
        int remain_size_start = nn_size << 3;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = ii * 8;

            const __fp16* img0 = bottom_blob.channel(0);
            img0 += i;

            __fp16* tmpptr = tmp.channel(i / 8);

            for (int q = 0; q < inch; q++)
            {
                vst1q_f16(tmpptr, vld1q_f16(img0));

                tmpptr += 8;
                img0 += bottom_blob.cstep;
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < size; i++)
        {
            const __fp16* img0 = bottom_blob.channel(0);
            img0 += i;

            __fp16* tmpptr = tmp.channel(i / 8 + i % 8);

            for (int q = 0; q < inch; q++)
            {
                tmpptr[0] = img0[0];
                tmpptr++;
                img0 += bottom_blob.cstep;
            }
        }
    }

    int nn_outch = 0;
    int remain_outch_start = 0;

    nn_outch = outch >> 3;
    remain_outch_start = nn_outch << 3;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 8;

        __fp16* outptr0 = top_blob.channel(p);
        __fp16* outptr1 = top_blob.channel(p + 1);
        __fp16* outptr2 = top_blob.channel(p + 2);
        __fp16* outptr3 = top_blob.channel(p + 3);
        __fp16* outptr4 = top_blob.channel(p + 4);
        __fp16* outptr5 = top_blob.channel(p + 5);
        __fp16* outptr6 = top_blob.channel(p + 6);
        __fp16* outptr7 = top_blob.channel(p + 7);

        const __fp16 zeros[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        const __fp16* biasptr = bias ? bias + p : zeros;
        float16x8_t _bias0 = vld1q_f16(biasptr);

        int i = 0;

        for (; i + 7 < size; i += 8)
        {
            const __fp16* tmpptr = tmp.channel(i / 8);
            const __fp16* kptr = kernel.channel(p / 8);

            int q = 0;

            float16x8_t _sum0 = vdupq_laneq_f16(_bias0, 0);
            float16x8_t _sum1 = vdupq_laneq_f16(_bias0, 1);
            float16x8_t _sum2 = vdupq_laneq_f16(_bias0, 2);
            float16x8_t _sum3 = vdupq_laneq_f16(_bias0, 3);
            float16x8_t _sum4 = vdupq_laneq_f16(_bias0, 4);
            float16x8_t _sum5 = vdupq_laneq_f16(_bias0, 5);
            float16x8_t _sum6 = vdupq_laneq_f16(_bias0, 6);
            float16x8_t _sum7 = vdupq_laneq_f16(_bias0, 7);

            for (; q + 7 < inch; q += 8)
            {
                float16x8_t _p0 = vld1q_f16(tmpptr);
                float16x8_t _p1 = vld1q_f16(tmpptr + 8);
                float16x8_t _p2 = vld1q_f16(tmpptr + 16);
                float16x8_t _p3 = vld1q_f16(tmpptr + 24);
                float16x8_t _p4 = vld1q_f16(tmpptr + 32);
                float16x8_t _p5 = vld1q_f16(tmpptr + 40);
                float16x8_t _p6 = vld1q_f16(tmpptr + 48);
                float16x8_t _p7 = vld1q_f16(tmpptr + 56);

                float16x8_t _k0 = vld1q_f16(kptr);
                float16x8_t _k1 = vld1q_f16(kptr + 8);
                float16x8_t _k2 = vld1q_f16(kptr + 16);
                float16x8_t _k3 = vld1q_f16(kptr + 24);
                float16x8_t _k4 = vld1q_f16(kptr + 32);
                float16x8_t _k5 = vld1q_f16(kptr + 40);
                float16x8_t _k6 = vld1q_f16(kptr + 48);
                float16x8_t _k7 = vld1q_f16(kptr + 56);

                _sum0 = vfmaq_laneq_f16(_sum0, _p0, _k0, 0);
                _sum1 = vfmaq_laneq_f16(_sum1, _p0, _k0, 1);
                _sum2 = vfmaq_laneq_f16(_sum2, _p0, _k0, 2);
                _sum3 = vfmaq_laneq_f16(_sum3, _p0, _k0, 3);
                _sum4 = vfmaq_laneq_f16(_sum4, _p0, _k0, 4);
                _sum5 = vfmaq_laneq_f16(_sum5, _p0, _k0, 5);
                _sum6 = vfmaq_laneq_f16(_sum6, _p0, _k0, 6);
                _sum7 = vfmaq_laneq_f16(_sum7, _p0, _k0, 7);

                _sum0 = vfmaq_laneq_f16(_sum0, _p1, _k1, 0);
                _sum1 = vfmaq_laneq_f16(_sum1, _p1, _k1, 1);
                _sum2 = vfmaq_laneq_f16(_sum2, _p1, _k1, 2);
                _sum3 = vfmaq_laneq_f16(_sum3, _p1, _k1, 3);
                _sum4 = vfmaq_laneq_f16(_sum4, _p1, _k1, 4);
                _sum5 = vfmaq_laneq_f16(_sum5, _p1, _k1, 5);
                _sum6 = vfmaq_laneq_f16(_sum6, _p1, _k1, 6);
                _sum7 = vfmaq_laneq_f16(_sum7, _p1, _k1, 7);

                _sum0 = vfmaq_laneq_f16(_sum0, _p2, _k2, 0);
                _sum1 = vfmaq_laneq_f16(_sum1, _p2, _k2, 1);
                _sum2 = vfmaq_laneq_f16(_sum2, _p2, _k2, 2);
                _sum3 = vfmaq_laneq_f16(_sum3, _p2, _k2, 3);
                _sum4 = vfmaq_laneq_f16(_sum4, _p2, _k2, 4);
                _sum5 = vfmaq_laneq_f16(_sum5, _p2, _k2, 5);
                _sum6 = vfmaq_laneq_f16(_sum6, _p2, _k2, 6);
                _sum7 = vfmaq_laneq_f16(_sum7, _p2, _k2, 7);

                _sum0 = vfmaq_laneq_f16(_sum0, _p3, _k3, 0);
                _sum1 = vfmaq_laneq_f16(_sum1, _p3, _k3, 1);
                _sum2 = vfmaq_laneq_f16(_sum2, _p3, _k3, 2);
                _sum3 = vfmaq_laneq_f16(_sum3, _p3, _k3, 3);
                _sum4 = vfmaq_laneq_f16(_sum4, _p3, _k3, 4);
                _sum5 = vfmaq_laneq_f16(_sum5, _p3, _k3, 5);
                _sum6 = vfmaq_laneq_f16(_sum6, _p3, _k3, 6);
                _sum7 = vfmaq_laneq_f16(_sum7, _p3, _k3, 7);

                _sum0 = vfmaq_laneq_f16(_sum0, _p4, _k4, 0);
                _sum1 = vfmaq_laneq_f16(_sum1, _p4, _k4, 1);
                _sum2 = vfmaq_laneq_f16(_sum2, _p4, _k4, 2);
                _sum3 = vfmaq_laneq_f16(_sum3, _p4, _k4, 3);
                _sum4 = vfmaq_laneq_f16(_sum4, _p4, _k4, 4);
                _sum5 = vfmaq_laneq_f16(_sum5, _p4, _k4, 5);
                _sum6 = vfmaq_laneq_f16(_sum6, _p4, _k4, 6);
                _sum7 = vfmaq_laneq_f16(_sum7, _p4, _k4, 7);

                _sum0 = vfmaq_laneq_f16(_sum0, _p5, _k5, 0);
                _sum1 = vfmaq_laneq_f16(_sum1, _p5, _k5, 1);
                _sum2 = vfmaq_laneq_f16(_sum2, _p5, _k5, 2);
                _sum3 = vfmaq_laneq_f16(_sum3, _p5, _k5, 3);
                _sum4 = vfmaq_laneq_f16(_sum4, _p5, _k5, 4);
                _sum5 = vfmaq_laneq_f16(_sum5, _p5, _k5, 5);
                _sum6 = vfmaq_laneq_f16(_sum6, _p5, _k5, 6);
                _sum7 = vfmaq_laneq_f16(_sum7, _p5, _k5, 7);

                _sum0 = vfmaq_laneq_f16(_sum0, _p6, _k6, 0);
                _sum1 = vfmaq_laneq_f16(_sum1, _p6, _k6, 1);
                _sum2 = vfmaq_laneq_f16(_sum2, _p6, _k6, 2);
                _sum3 = vfmaq_laneq_f16(_sum3, _p6, _k6, 3);
                _sum4 = vfmaq_laneq_f16(_sum4, _p6, _k6, 4);
                _sum5 = vfmaq_laneq_f16(_sum5, _p6, _k6, 5);
                _sum6 = vfmaq_laneq_f16(_sum6, _p6, _k6, 6);
                _sum7 = vfmaq_laneq_f16(_sum7, _p6, _k6, 7);

                _sum0 = vfmaq_laneq_f16(_sum0, _p7, _k7, 0);
                _sum1 = vfmaq_laneq_f16(_sum1, _p7, _k7, 1);
                _sum2 = vfmaq_laneq_f16(_sum2, _p7, _k7, 2);
                _sum3 = vfmaq_laneq_f16(_sum3, _p7, _k7, 3);
                _sum4 = vfmaq_laneq_f16(_sum4, _p7, _k7, 4);
                _sum5 = vfmaq_laneq_f16(_sum5, _p7, _k7, 5);
                _sum6 = vfmaq_laneq_f16(_sum6, _p7, _k7, 6);
                _sum7 = vfmaq_laneq_f16(_sum7, _p7, _k7, 7);

                tmpptr += 64;
                kptr += 64;
            }

            for (; q < inch; q++)
            {
                float16x8_t _p0 = vld1q_f16(tmpptr);

                float16x8_t _k0 = vld1q_f16(kptr);

                _sum0 = vfmaq_laneq_f16(_sum0, _p0, _k0, 0);
                _sum1 = vfmaq_laneq_f16(_sum1, _p0, _k0, 1);
                _sum2 = vfmaq_laneq_f16(_sum2, _p0, _k0, 2);
                _sum3 = vfmaq_laneq_f16(_sum3, _p0, _k0, 3);
                _sum4 = vfmaq_laneq_f16(_sum4, _p0, _k0, 4);
                _sum5 = vfmaq_laneq_f16(_sum5, _p0, _k0, 5);
                _sum6 = vfmaq_laneq_f16(_sum6, _p0, _k0, 6);
                _sum7 = vfmaq_laneq_f16(_sum7, _p0, _k0, 7);

                tmpptr += 8;
                kptr += 8;
            }

            vst1q_f16(outptr0, _sum0);
            vst1q_f16(outptr1, _sum1);
            vst1q_f16(outptr2, _sum2);
            vst1q_f16(outptr3, _sum3);
            vst1q_f16(outptr4, _sum4);
            vst1q_f16(outptr5, _sum5);
            vst1q_f16(outptr6, _sum6);
            vst1q_f16(outptr7, _sum7);

            outptr0 += 8;
            outptr1 += 8;
            outptr2 += 8;
            outptr3 += 8;
            outptr4 += 8;
            outptr5 += 8;
            outptr6 += 8;
            outptr7 += 8;
        }

        for (; i < size; i++)
        {
            const __fp16* tmpptr = tmp.channel(i / 8 + i % 8);
            const __fp16* kptr = kernel.channel(p / 8);

            int q = 0;

            float16x8_t _sum0 = _bias0;

            for (; q + 7 < inch; q += 8)
            {
                float16x8_t _p0 = vld1q_f16(tmpptr);

                float16x8_t _k0 = vld1q_f16(kptr);
                float16x8_t _k1 = vld1q_f16(kptr + 8);
                float16x8_t _k2 = vld1q_f16(kptr + 16);
                float16x8_t _k3 = vld1q_f16(kptr + 24);
                float16x8_t _k4 = vld1q_f16(kptr + 32);
                float16x8_t _k5 = vld1q_f16(kptr + 40);
                float16x8_t _k6 = vld1q_f16(kptr + 48);
                float16x8_t _k7 = vld1q_f16(kptr + 56);

                _sum0 = vfmaq_laneq_f16(_sum0, _k0, _p0, 0);
                _sum0 = vfmaq_laneq_f16(_sum0, _k1, _p0, 1);
                _sum0 = vfmaq_laneq_f16(_sum0, _k2, _p0, 2);
                _sum0 = vfmaq_laneq_f16(_sum0, _k3, _p0, 3);
                _sum0 = vfmaq_laneq_f16(_sum0, _k4, _p0, 4);
                _sum0 = vfmaq_laneq_f16(_sum0, _k5, _p0, 5);
                _sum0 = vfmaq_laneq_f16(_sum0, _k6, _p0, 6);
                _sum0 = vfmaq_laneq_f16(_sum0, _k7, _p0, 7);

                tmpptr += 8;
                kptr += 64;
            }

            for (; q < inch; q++)
            {
                float16x8_t _p0 = vld1q_dup_f16(tmpptr);

                float16x8_t _k0 = vld1q_f16(kptr);

                _sum0 = vfmaq_f16(_sum0, _k0, _p0);

                tmpptr += 1;
                kptr += 8;
            }

            vst1q_lane_f16(outptr0, _sum0, 0);
            vst1q_lane_f16(outptr1, _sum0, 1);
            vst1q_lane_f16(outptr2, _sum0, 2);
            vst1q_lane_f16(outptr3, _sum0, 3);
            vst1q_lane_f16(outptr4, _sum0, 4);
            vst1q_lane_f16(outptr5, _sum0, 5);
            vst1q_lane_f16(outptr6, _sum0, 6);
            vst1q_lane_f16(outptr7, _sum0, 7);

            outptr0++;
            outptr1++;
            outptr2++;
            outptr3++;
            outptr4++;
            outptr5++;
            outptr6++;
            outptr7++;
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        const __fp16 bias0 = bias ? bias[p] : 0.f;

        __fp16* outptr0 = out0;

        int i = 0;

        for (; i + 7 < size; i += 8)
        {
            const __fp16* tmpptr = tmp.channel(i / 8);
            const __fp16* kptr = kernel.channel(p / 8 + p % 8);

            int q = 0;

            float16x8_t _sum0 = vdupq_n_f16(bias0);

            for (; q + 7 < inch; q += 8)
            {
                float16x8_t _p0 = vld1q_f16(tmpptr);
                float16x8_t _p1 = vld1q_f16(tmpptr + 8);
                float16x8_t _p2 = vld1q_f16(tmpptr + 16);
                float16x8_t _p3 = vld1q_f16(tmpptr + 24);
                float16x8_t _p4 = vld1q_f16(tmpptr + 32);
                float16x8_t _p5 = vld1q_f16(tmpptr + 40);
                float16x8_t _p6 = vld1q_f16(tmpptr + 48);
                float16x8_t _p7 = vld1q_f16(tmpptr + 56);

                float16x8_t _k0 = vld1q_f16(kptr);

                _sum0 = vfmaq_laneq_f16(_sum0, _p0, _k0, 0);
                _sum0 = vfmaq_laneq_f16(_sum0, _p1, _k0, 1);
                _sum0 = vfmaq_laneq_f16(_sum0, _p2, _k0, 2);
                _sum0 = vfmaq_laneq_f16(_sum0, _p3, _k0, 3);
                _sum0 = vfmaq_laneq_f16(_sum0, _p4, _k0, 4);
                _sum0 = vfmaq_laneq_f16(_sum0, _p5, _k0, 5);
                _sum0 = vfmaq_laneq_f16(_sum0, _p6, _k0, 6);
                _sum0 = vfmaq_laneq_f16(_sum0, _p7, _k0, 7);

                tmpptr += 64;
                kptr += 8;
            }

            for (; q < inch; q++)
            {
                float16x8_t _p0 = vld1q_f16(tmpptr);

                float16x8_t _k0 = vld1q_dup_f16(kptr);

                _sum0 = vfmaq_f16(_sum0, _p0, _k0);

                tmpptr += 8;
                kptr += 1;
            }

            vst1q_f16(outptr0, _sum0);

            outptr0 += 8;
        }

        for (; i < size; i++)
        {
            const __fp16* tmpptr = tmp.channel(i / 8 + i % 8);
            const __fp16* kptr = kernel.channel(p / 8 + p % 8);

            int q = 0;

            float16x8_t _sum0 = vdupq_n_f16(0.f);

            for (; q + 7 < inch; q += 8)
            {
                float16x8_t _p0 = vld1q_f16(tmpptr);

                float16x8_t _k0 = vld1q_f16(kptr);

                _sum0 = vfmaq_f16(_sum0, _p0, _k0);

                tmpptr += 8;
                kptr += 8;
            }

            __fp16 sum0 = bias0 + vaddvq_f32(vcvt_f32_f16(vadd_f16(vget_low_f16(_sum0), vget_high_f16(_sum0))));

            for (; q < inch; q++)
            {
                sum0 += tmpptr[0] * kptr[0];

                tmpptr++;
                kptr++;
            }

            outptr0[0] = sum0;

            outptr0++;
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
    //             const __fp16* kptr = _kernel.channel(p/8 + p%8);
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
