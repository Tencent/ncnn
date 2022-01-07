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

static void im2col_sgemm_pack1to4_fp16sa_neon(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    // Mat bottom_im2col(size, maxk, inch, 2u, 1, opt.workspace_allocator);

    const int size = bottom_im2col.w;
    const int maxk = bottom_im2col.h;
    const int inch = bottom_im2col.c;

    const int outch = top_blob.c;

    const __fp16* bias = _bias;

    // permute
    Mat tmp;
    if (size >= 8)
        tmp.create(8 * maxk, inch, size / 8 + (size % 8) / 4 + size % 4, 2u, 1, opt.workspace_allocator);
    else if (size >= 4)
        tmp.create(4 * maxk, inch, size / 4 + size % 4, 2u, 1, opt.workspace_allocator);
    else
        tmp.create(maxk, inch, size, 2u, 1, opt.workspace_allocator);
    {
        int nn_size = size >> 3;
        int remain_size_start = 0;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 8;

            __fp16* tmpptr = tmp.channel(i / 8);

            for (int q = 0; q < inch; q++)
            {
                const __fp16* img0 = (const __fp16*)bottom_im2col.channel(q) + i;

                for (int k = 0; k < maxk; k++)
                {
                    vst1q_f16(tmpptr, vld1q_f16(img0));
                    img0 += size;
                    tmpptr += 8;
                }
            }
        }

        remain_size_start += nn_size << 3;
        nn_size = (size - remain_size_start) >> 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 4;

            __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);

            for (int q = 0; q < inch; q++)
            {
                const __fp16* img0 = (const __fp16*)bottom_im2col.channel(q) + i;

                for (int k = 0; k < maxk; k++)
                {
                    vst1_f16(tmpptr, vld1_f16(img0));
                    img0 += size;
                    tmpptr += 4;
                }
            }
        }

        remain_size_start += nn_size << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < size; i++)
        {
            __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + i % 4);

            for (int q = 0; q < inch; q++)
            {
                const __fp16* img0 = (const __fp16*)bottom_im2col.channel(q) + i;

                for (int k = 0; k < maxk; k++)
                {
                    tmpptr[0] = img0[0];
                    img0 += size;
                    tmpptr += 1;
                }
            }
        }
    }

    int remain_outch_start = 0;

    int nn_outch = outch >> 1;
    remain_outch_start = nn_outch << 1;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 2;

        __fp16* outptr0 = top_blob.channel(p);
        __fp16* outptr1 = top_blob.channel(p + 1);

        const __fp16 zeros[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        const __fp16* biasptr = bias ? bias + p * 4 : zeros;

        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            const __fp16* tmpptr = tmp.channel(i / 8);
            const __fp16* kptr0 = kernel.channel(p / 2);

            int nn = inch * maxk; // inch always > 0

            float16x8_t _sum010 = vld1q_f16(biasptr);
            float16x8_t _sum011 = vld1q_f16(biasptr);
            float16x8_t _sum012 = vld1q_f16(biasptr);
            float16x8_t _sum013 = vld1q_f16(biasptr);
            float16x8_t _sum014 = vld1q_f16(biasptr);
            float16x8_t _sum015 = vld1q_f16(biasptr);
            float16x8_t _sum016 = vld1q_f16(biasptr);
            float16x8_t _sum017 = vld1q_f16(biasptr);

            for (int j = 0; j < nn; j++)
            {
                float16x8_t _val = vld1q_f16(tmpptr);
                float16x8_t _w01 = vld1q_f16(kptr0);

                _sum010 = vfmaq_laneq_f16(_sum010, _w01, _val, 0);
                _sum011 = vfmaq_laneq_f16(_sum011, _w01, _val, 1);
                _sum012 = vfmaq_laneq_f16(_sum012, _w01, _val, 2);
                _sum013 = vfmaq_laneq_f16(_sum013, _w01, _val, 3);
                _sum014 = vfmaq_laneq_f16(_sum014, _w01, _val, 4);
                _sum015 = vfmaq_laneq_f16(_sum015, _w01, _val, 5);
                _sum016 = vfmaq_laneq_f16(_sum016, _w01, _val, 6);
                _sum017 = vfmaq_laneq_f16(_sum017, _w01, _val, 7);

                tmpptr += 8;
                kptr0 += 8;
            }

            // TODO optimize with transpose
            vst1_f16(outptr0, vget_low_f16(_sum010));
            vst1_f16(outptr0 + 4, vget_low_f16(_sum011));
            vst1_f16(outptr0 + 8, vget_low_f16(_sum012));
            vst1_f16(outptr0 + 12, vget_low_f16(_sum013));
            vst1_f16(outptr0 + 16, vget_low_f16(_sum014));
            vst1_f16(outptr0 + 20, vget_low_f16(_sum015));
            vst1_f16(outptr0 + 24, vget_low_f16(_sum016));
            vst1_f16(outptr0 + 28, vget_low_f16(_sum017));
            vst1_f16(outptr1, vget_high_f16(_sum010));
            vst1_f16(outptr1 + 4, vget_high_f16(_sum011));
            vst1_f16(outptr1 + 8, vget_high_f16(_sum012));
            vst1_f16(outptr1 + 12, vget_high_f16(_sum013));
            vst1_f16(outptr1 + 16, vget_high_f16(_sum014));
            vst1_f16(outptr1 + 20, vget_high_f16(_sum015));
            vst1_f16(outptr1 + 24, vget_high_f16(_sum016));
            vst1_f16(outptr1 + 28, vget_high_f16(_sum017));
            outptr0 += 32;
            outptr1 += 32;
        }
        for (; i + 3 < size; i += 4)
        {
            const __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
            const __fp16* kptr0 = kernel.channel(p / 2);

            int nn = inch * maxk; // inch always > 0

            float16x8_t _sum010 = vld1q_f16(biasptr);
            float16x8_t _sum011 = vld1q_f16(biasptr);
            float16x8_t _sum012 = vld1q_f16(biasptr);
            float16x8_t _sum013 = vld1q_f16(biasptr);

            for (int j = 0; j < nn; j++)
            {
                float16x4_t _val = vld1_f16(tmpptr);
                float16x8_t _w01 = vld1q_f16(kptr0);

                _sum010 = vfmaq_lane_f16(_sum010, _w01, _val, 0);
                _sum011 = vfmaq_lane_f16(_sum011, _w01, _val, 1);
                _sum012 = vfmaq_lane_f16(_sum012, _w01, _val, 2);
                _sum013 = vfmaq_lane_f16(_sum013, _w01, _val, 3);

                tmpptr += 4;
                kptr0 += 8;
            }

            // TODO optimize with transpose
            vst1_f16(outptr0, vget_low_f16(_sum010));
            vst1_f16(outptr0 + 4, vget_low_f16(_sum011));
            vst1_f16(outptr0 + 8, vget_low_f16(_sum012));
            vst1_f16(outptr0 + 12, vget_low_f16(_sum013));
            vst1_f16(outptr1, vget_high_f16(_sum010));
            vst1_f16(outptr1 + 4, vget_high_f16(_sum011));
            vst1_f16(outptr1 + 8, vget_high_f16(_sum012));
            vst1_f16(outptr1 + 12, vget_high_f16(_sum013));
            outptr0 += 16;
            outptr1 += 16;
        }
        for (; i < size; i++)
        {
            const __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + i % 4);
            const __fp16* kptr0 = kernel.channel(p / 2);

            int nn = inch * maxk; // inch always > 0

            float16x8_t _sum01 = vld1q_f16(biasptr);

            for (int j = 0; j < nn; j++)
            {
                float16x8_t _val = vdupq_n_f16(tmpptr[0]);
                float16x8_t _w01 = vld1q_f16(kptr0);
                _sum01 = vfmaq_f16(_sum01, _val, _w01);

                tmpptr += 1;
                kptr0 += 8;
            }

            vst1_f16(outptr0, vget_low_f16(_sum01));
            vst1_f16(outptr1, vget_high_f16(_sum01));
            outptr0 += 4;
            outptr1 += 4;
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        __fp16* outptr0 = top_blob.channel(p);

        const __fp16 zeros[4] = {0.f, 0.f, 0.f, 0.f};
        const __fp16* biasptr = bias ? bias + p * 4 : zeros;

        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            const __fp16* tmpptr = tmp.channel(i / 8);
            const __fp16* kptr0 = kernel.channel(p / 2 + p % 2);

            int nn = inch * maxk; // inch always > 0

            float16x4_t _sum0 = vld1_f16(biasptr);
            float16x4_t _sum1 = vld1_f16(biasptr);
            float16x4_t _sum2 = vld1_f16(biasptr);
            float16x4_t _sum3 = vld1_f16(biasptr);
            float16x4_t _sum4 = vld1_f16(biasptr);
            float16x4_t _sum5 = vld1_f16(biasptr);
            float16x4_t _sum6 = vld1_f16(biasptr);
            float16x4_t _sum7 = vld1_f16(biasptr);

            for (int j = 0; j < nn; j++)
            {
                float16x8_t _val = vld1q_f16(tmpptr);
                float16x4_t _w0 = vld1_f16(kptr0);

                _sum0 = vfma_laneq_f16(_sum0, _w0, _val, 0);
                _sum1 = vfma_laneq_f16(_sum1, _w0, _val, 1);
                _sum2 = vfma_laneq_f16(_sum2, _w0, _val, 2);
                _sum3 = vfma_laneq_f16(_sum3, _w0, _val, 3);
                _sum4 = vfma_laneq_f16(_sum4, _w0, _val, 4);
                _sum5 = vfma_laneq_f16(_sum5, _w0, _val, 5);
                _sum6 = vfma_laneq_f16(_sum6, _w0, _val, 6);
                _sum7 = vfma_laneq_f16(_sum7, _w0, _val, 7);

                tmpptr += 8;
                kptr0 += 4;
            }

            vst1_f16(outptr0, _sum0);
            vst1_f16(outptr0 + 4, _sum1);
            vst1_f16(outptr0 + 8, _sum2);
            vst1_f16(outptr0 + 12, _sum3);
            vst1_f16(outptr0 + 16, _sum4);
            vst1_f16(outptr0 + 20, _sum5);
            vst1_f16(outptr0 + 24, _sum6);
            vst1_f16(outptr0 + 28, _sum7);
            outptr0 += 32;
        }
        for (; i + 3 < size; i += 4)
        {
            const __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
            const __fp16* kptr0 = kernel.channel(p / 2 + p % 2);

            int nn = inch * maxk; // inch always > 0

            float16x4_t _sum0 = vld1_f16(biasptr);
            float16x4_t _sum1 = vld1_f16(biasptr);
            float16x4_t _sum2 = vld1_f16(biasptr);
            float16x4_t _sum3 = vld1_f16(biasptr);

            for (int j = 0; j < nn; j++)
            {
                float16x4_t _val = vld1_f16(tmpptr);
                float16x4_t _w0 = vld1_f16(kptr0);

                _sum0 = vfma_lane_f16(_sum0, _w0, _val, 0);
                _sum1 = vfma_lane_f16(_sum1, _w0, _val, 1);
                _sum2 = vfma_lane_f16(_sum2, _w0, _val, 2);
                _sum3 = vfma_lane_f16(_sum3, _w0, _val, 3);

                tmpptr += 4;
                kptr0 += 4;
            }

            vst1_f16(outptr0, _sum0);
            vst1_f16(outptr0 + 4, _sum1);
            vst1_f16(outptr0 + 8, _sum2);
            vst1_f16(outptr0 + 12, _sum3);
            outptr0 += 16;
        }
        for (; i < size; i++)
        {
            const __fp16* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + i % 4);
            const __fp16* kptr0 = kernel.channel(p / 2 + p % 2);

            int nn = inch * maxk; // inch always > 0

            float16x4_t _sum = vld1_f16(biasptr);

            for (int j = 0; j < nn; j++)
            {
                float16x4_t _val = vdup_n_f16(tmpptr[0]);
                float16x4_t _w0 = vld1_f16(kptr0);
                _sum = vfma_f16(_sum, _val, _w0);

                tmpptr += 1;
                kptr0 += 4;
            }

            vst1_f16(outptr0, _sum);
            outptr0 += 4;
        }
    }
}

static void convolution_im2col_sgemm_transform_kernel_pack1to4_fp16sa_neon(const Mat& _kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = 4b-4a-maxk-inch/4a-outch/4b
    Mat kernel = _kernel.reshape(maxk, inch, outch);
    kernel_tm.create(8 * maxk, inch, outch / 8 + (outch % 8) / 4, (size_t)2u, 1);

    int q = 0;
    for (; q + 7 < outch; q += 8)
    {
        const Mat k0 = kernel.channel(q);
        const Mat k1 = kernel.channel(q + 1);
        const Mat k2 = kernel.channel(q + 2);
        const Mat k3 = kernel.channel(q + 3);
        const Mat k4 = kernel.channel(q + 4);
        const Mat k5 = kernel.channel(q + 5);
        const Mat k6 = kernel.channel(q + 6);
        const Mat k7 = kernel.channel(q + 7);

        __fp16* g00 = kernel_tm.channel(q / 8);

        for (int p = 0; p < inch; p++)
        {
            const float* k00 = k0.row(p);
            const float* k10 = k1.row(p);
            const float* k20 = k2.row(p);
            const float* k30 = k3.row(p);
            const float* k40 = k4.row(p);
            const float* k50 = k5.row(p);
            const float* k60 = k6.row(p);
            const float* k70 = k7.row(p);

            for (int k = 0; k < maxk; k++)
            {
                g00[0] = (__fp16)k00[k];
                g00[1] = (__fp16)k10[k];
                g00[2] = (__fp16)k20[k];
                g00[3] = (__fp16)k30[k];
                g00[4] = (__fp16)k40[k];
                g00[5] = (__fp16)k50[k];
                g00[6] = (__fp16)k60[k];
                g00[7] = (__fp16)k70[k];

                g00 += 8;
            }
        }
    }
    for (; q + 3 < outch; q += 4)
    {
        const Mat k0 = kernel.channel(q);
        const Mat k1 = kernel.channel(q + 1);
        const Mat k2 = kernel.channel(q + 2);
        const Mat k3 = kernel.channel(q + 3);

        __fp16* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4);

        for (int p = 0; p < inch; p++)
        {
            const float* k00 = k0.row(p);
            const float* k10 = k1.row(p);
            const float* k20 = k2.row(p);
            const float* k30 = k3.row(p);

            for (int k = 0; k < maxk; k++)
            {
                g00[0] = (__fp16)k00[k];
                g00[1] = (__fp16)k10[k];
                g00[2] = (__fp16)k20[k];
                g00[3] = (__fp16)k30[k];

                g00 += 4;
            }
        }
    }
}

static void convolution_im2col_sgemm_pack1to4_fp16sa_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    const int size = outw * outh;

    const int maxk = kernel_w * kernel_h;

    // im2col
    Mat bottom_im2col(size, maxk, inch, 2u, 1, opt.workspace_allocator);
    {
        const int gap = w * stride_h - outw * stride_w;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < inch; p++)
        {
            const Mat img = bottom_blob.channel(p);
            __fp16* ptr = bottom_im2col.channel(p);

            for (int u = 0; u < kernel_h; u++)
            {
                for (int v = 0; v < kernel_w; v++)
                {
                    const __fp16* sptr = img.row<const __fp16>(dilation_h * u) + dilation_w * v;

                    for (int i = 0; i < outh; i++)
                    {
                        int j = 0;
                        for (; j < outw; j++)
                        {
                            ptr[0] = sptr[0];

                            sptr += stride_w;
                            ptr += 1;
                        }

                        sptr += gap;
                    }
                }
            }
        }
    }

    im2col_sgemm_pack1to4_fp16sa_neon(bottom_im2col, top_blob, kernel, _bias, opt);
}
