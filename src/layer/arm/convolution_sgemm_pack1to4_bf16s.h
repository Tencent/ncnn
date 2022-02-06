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

static void im2col_sgemm_pack1to4_bf16s_neon(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    // Mat bottom_im2col(size, maxk, inch, 2u, 1, opt.workspace_allocator);

    const int size = bottom_im2col.w;
    const int maxk = bottom_im2col.h;
    const int inch = bottom_im2col.c;

    const int outch = top_blob.c;

    const float* bias = _bias;

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

            unsigned short* tmpptr = tmp.channel(i / 8);

            for (int q = 0; q < inch; q++)
            {
                const unsigned short* img0 = (const unsigned short*)bottom_im2col.channel(q) + i;

                for (int k = 0; k < maxk; k++)
                {
#if __ARM_NEON
                    vst1q_u16(tmpptr, vld1q_u16(img0));
#else
                    tmpptr[0] = img0[0];
                    tmpptr[1] = img0[1];
                    tmpptr[2] = img0[2];
                    tmpptr[3] = img0[3];
                    tmpptr[4] = img0[4];
                    tmpptr[5] = img0[5];
                    tmpptr[6] = img0[6];
                    tmpptr[7] = img0[7];
#endif // __ARM_NEON
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

            unsigned short* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);

            for (int q = 0; q < inch; q++)
            {
                const unsigned short* img0 = (const unsigned short*)bottom_im2col.channel(q) + i;

                for (int k = 0; k < maxk; k++)
                {
#if __ARM_NEON
                    vst1_u16(tmpptr, vld1_u16(img0));
#else
                    tmpptr[0] = img0[0];
                    tmpptr[1] = img0[1];
                    tmpptr[2] = img0[2];
                    tmpptr[3] = img0[3];

                    tmpptr += 4;
#endif // __ARM_NEON
                    img0 += size;
                    tmpptr += 4;
                }
            }
        }

        remain_size_start += nn_size << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < size; i++)
        {
            unsigned short* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + i % 4);

            for (int q = 0; q < inch; q++)
            {
                const unsigned short* img0 = (const unsigned short*)bottom_im2col.channel(q) + i;

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

#if __ARM_NEON && __aarch64__
    int nn_outch = outch >> 1;
    remain_outch_start = nn_outch << 1;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 2;

        unsigned short* outptr0 = top_blob.channel(p);
        unsigned short* outptr1 = top_blob.channel(p + 1);

        const float zeros[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        const float* biasptr = bias ? bias + p * 4 : zeros;

        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            const unsigned short* tmpptr = tmp.channel(i / 8);
            const unsigned short* kptr0 = kernel.channel(p / 2);

            int nn = inch * maxk; // inch always > 0

            float32x4_t _sum00 = vld1q_f32(biasptr);
            float32x4_t _sum01 = vld1q_f32(biasptr);
            float32x4_t _sum02 = vld1q_f32(biasptr);
            float32x4_t _sum03 = vld1q_f32(biasptr);
            float32x4_t _sum04 = vld1q_f32(biasptr);
            float32x4_t _sum05 = vld1q_f32(biasptr);
            float32x4_t _sum06 = vld1q_f32(biasptr);
            float32x4_t _sum07 = vld1q_f32(biasptr);
            float32x4_t _sum10 = vld1q_f32(biasptr + 4);
            float32x4_t _sum11 = vld1q_f32(biasptr + 4);
            float32x4_t _sum12 = vld1q_f32(biasptr + 4);
            float32x4_t _sum13 = vld1q_f32(biasptr + 4);
            float32x4_t _sum14 = vld1q_f32(biasptr + 4);
            float32x4_t _sum15 = vld1q_f32(biasptr + 4);
            float32x4_t _sum16 = vld1q_f32(biasptr + 4);
            float32x4_t _sum17 = vld1q_f32(biasptr + 4);

            for (int j = 0; j < nn; j++)
            {
                float32x4_t _val0 = vcvt_f32_bf16(vld1_u16(tmpptr));
                float32x4_t _val1 = vcvt_f32_bf16(vld1_u16(tmpptr + 4));
                float32x4_t _w0 = vcvt_f32_bf16(vld1_u16(kptr0));
                float32x4_t _w1 = vcvt_f32_bf16(vld1_u16(kptr0 + 4));

                _sum00 = vmlaq_laneq_f32(_sum00, _w0, _val0, 0);
                _sum01 = vmlaq_laneq_f32(_sum01, _w0, _val0, 1);
                _sum02 = vmlaq_laneq_f32(_sum02, _w0, _val0, 2);
                _sum03 = vmlaq_laneq_f32(_sum03, _w0, _val0, 3);
                _sum04 = vmlaq_laneq_f32(_sum04, _w0, _val1, 0);
                _sum05 = vmlaq_laneq_f32(_sum05, _w0, _val1, 1);
                _sum06 = vmlaq_laneq_f32(_sum06, _w0, _val1, 2);
                _sum07 = vmlaq_laneq_f32(_sum07, _w0, _val1, 3);
                _sum10 = vmlaq_laneq_f32(_sum10, _w1, _val0, 0);
                _sum11 = vmlaq_laneq_f32(_sum11, _w1, _val0, 1);
                _sum12 = vmlaq_laneq_f32(_sum12, _w1, _val0, 2);
                _sum13 = vmlaq_laneq_f32(_sum13, _w1, _val0, 3);
                _sum14 = vmlaq_laneq_f32(_sum14, _w1, _val1, 0);
                _sum15 = vmlaq_laneq_f32(_sum15, _w1, _val1, 1);
                _sum16 = vmlaq_laneq_f32(_sum16, _w1, _val1, 2);
                _sum17 = vmlaq_laneq_f32(_sum17, _w1, _val1, 3);

                tmpptr += 8;
                kptr0 += 8;
            }

            vst1_u16(outptr0, vcvt_bf16_f32(_sum00));
            vst1_u16(outptr0 + 4, vcvt_bf16_f32(_sum01));
            vst1_u16(outptr0 + 8, vcvt_bf16_f32(_sum02));
            vst1_u16(outptr0 + 12, vcvt_bf16_f32(_sum03));
            vst1_u16(outptr0 + 16, vcvt_bf16_f32(_sum04));
            vst1_u16(outptr0 + 20, vcvt_bf16_f32(_sum05));
            vst1_u16(outptr0 + 24, vcvt_bf16_f32(_sum06));
            vst1_u16(outptr0 + 28, vcvt_bf16_f32(_sum07));
            vst1_u16(outptr1, vcvt_bf16_f32(_sum10));
            vst1_u16(outptr1 + 4, vcvt_bf16_f32(_sum11));
            vst1_u16(outptr1 + 8, vcvt_bf16_f32(_sum12));
            vst1_u16(outptr1 + 12, vcvt_bf16_f32(_sum13));
            vst1_u16(outptr1 + 16, vcvt_bf16_f32(_sum14));
            vst1_u16(outptr1 + 20, vcvt_bf16_f32(_sum15));
            vst1_u16(outptr1 + 24, vcvt_bf16_f32(_sum16));
            vst1_u16(outptr1 + 28, vcvt_bf16_f32(_sum17));
            outptr0 += 32;
            outptr1 += 32;
        }
        for (; i + 3 < size; i += 4)
        {
            const unsigned short* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
            const unsigned short* kptr0 = kernel.channel(p / 2);

            int nn = inch * maxk; // inch always > 0

            float32x4_t _sum00 = vld1q_f32(biasptr);
            float32x4_t _sum01 = vld1q_f32(biasptr);
            float32x4_t _sum02 = vld1q_f32(biasptr);
            float32x4_t _sum03 = vld1q_f32(biasptr);
            float32x4_t _sum10 = vld1q_f32(biasptr + 4);
            float32x4_t _sum11 = vld1q_f32(biasptr + 4);
            float32x4_t _sum12 = vld1q_f32(biasptr + 4);
            float32x4_t _sum13 = vld1q_f32(biasptr + 4);

            for (int j = 0; j < nn; j++)
            {
                float32x4_t _val = vcvt_f32_bf16(vld1_u16(tmpptr));
                float32x4_t _w0 = vcvt_f32_bf16(vld1_u16(kptr0));
                float32x4_t _w1 = vcvt_f32_bf16(vld1_u16(kptr0 + 4));

                _sum00 = vmlaq_laneq_f32(_sum00, _w0, _val, 0);
                _sum01 = vmlaq_laneq_f32(_sum01, _w0, _val, 1);
                _sum02 = vmlaq_laneq_f32(_sum02, _w0, _val, 2);
                _sum03 = vmlaq_laneq_f32(_sum03, _w0, _val, 3);
                _sum10 = vmlaq_laneq_f32(_sum10, _w1, _val, 0);
                _sum11 = vmlaq_laneq_f32(_sum11, _w1, _val, 1);
                _sum12 = vmlaq_laneq_f32(_sum12, _w1, _val, 2);
                _sum13 = vmlaq_laneq_f32(_sum13, _w1, _val, 3);

                tmpptr += 4;
                kptr0 += 8;
            }

            vst1_u16(outptr0, vcvt_bf16_f32(_sum00));
            vst1_u16(outptr0 + 4, vcvt_bf16_f32(_sum01));
            vst1_u16(outptr0 + 8, vcvt_bf16_f32(_sum02));
            vst1_u16(outptr0 + 12, vcvt_bf16_f32(_sum03));
            vst1_u16(outptr1, vcvt_bf16_f32(_sum10));
            vst1_u16(outptr1 + 4, vcvt_bf16_f32(_sum11));
            vst1_u16(outptr1 + 8, vcvt_bf16_f32(_sum12));
            vst1_u16(outptr1 + 12, vcvt_bf16_f32(_sum13));
            outptr0 += 16;
            outptr1 += 16;
        }
        for (; i < size; i++)
        {
            const unsigned short* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + i % 4);
            const unsigned short* kptr0 = kernel.channel(p / 2);

            int nn = inch * maxk; // inch always > 0

            float32x4_t _sum0 = vld1q_f32(biasptr);
            float32x4_t _sum1 = vld1q_f32(biasptr + 4);

            for (int j = 0; j < nn; j++)
            {
                float32x4_t _val = vdupq_n_f32(bfloat16_to_float32(tmpptr[0]));
                float32x4_t _w0 = vcvt_f32_bf16(vld1_u16(kptr0));
                float32x4_t _w1 = vcvt_f32_bf16(vld1_u16(kptr0 + 4));
                _sum0 = vmlaq_f32(_sum0, _val, _w0);
                _sum1 = vmlaq_f32(_sum1, _val, _w1);

                tmpptr += 1;
                kptr0 += 8;
            }

            vst1_u16(outptr0, vcvt_bf16_f32(_sum0));
            vst1_u16(outptr1, vcvt_bf16_f32(_sum1));
            outptr0 += 4;
            outptr1 += 4;
        }
    }
#endif // __ARM_NEON && __aarch64__

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        unsigned short* outptr0 = top_blob.channel(p);

        const float zeros[4] = {0.f, 0.f, 0.f, 0.f};
        const float* biasptr = bias ? bias + p * 4 : zeros;

        int i = 0;
        for (; i + 7 < size; i += 8)
        {
            const unsigned short* tmpptr = tmp.channel(i / 8);
#if __ARM_NEON && __aarch64__
            const unsigned short* kptr0 = kernel.channel(p / 2 + p % 2);
#else
            const unsigned short* kptr0 = kernel.channel(p);
#endif

            int nn = inch * maxk; // inch always > 0

#if __ARM_NEON
            float32x4_t _sum0 = vld1q_f32(biasptr);
            float32x4_t _sum1 = vld1q_f32(biasptr);
            float32x4_t _sum2 = vld1q_f32(biasptr);
            float32x4_t _sum3 = vld1q_f32(biasptr);
            float32x4_t _sum4 = vld1q_f32(biasptr);
            float32x4_t _sum5 = vld1q_f32(biasptr);
            float32x4_t _sum6 = vld1q_f32(biasptr);
            float32x4_t _sum7 = vld1q_f32(biasptr);

            for (int j = 0; j < nn; j++)
            {
                float32x4_t _val0 = vcvt_f32_bf16(vld1_u16(tmpptr));
                float32x4_t _val1 = vcvt_f32_bf16(vld1_u16(tmpptr + 4));
                float32x4_t _w0 = vcvt_f32_bf16(vld1_u16(kptr0));

#if __aarch64__
                _sum0 = vmlaq_laneq_f32(_sum0, _w0, _val0, 0);
                _sum1 = vmlaq_laneq_f32(_sum1, _w0, _val0, 1);
                _sum2 = vmlaq_laneq_f32(_sum2, _w0, _val0, 2);
                _sum3 = vmlaq_laneq_f32(_sum3, _w0, _val0, 3);
                _sum4 = vmlaq_laneq_f32(_sum4, _w0, _val1, 0);
                _sum5 = vmlaq_laneq_f32(_sum5, _w0, _val1, 1);
                _sum6 = vmlaq_laneq_f32(_sum6, _w0, _val1, 2);
                _sum7 = vmlaq_laneq_f32(_sum7, _w0, _val1, 3);
#else
                _sum0 = vmlaq_lane_f32(_sum0, _w0, vget_low_f32(_val0), 0);
                _sum1 = vmlaq_lane_f32(_sum1, _w0, vget_low_f32(_val0), 1);
                _sum2 = vmlaq_lane_f32(_sum2, _w0, vget_high_f32(_val0), 0);
                _sum3 = vmlaq_lane_f32(_sum3, _w0, vget_high_f32(_val0), 1);
                _sum4 = vmlaq_lane_f32(_sum4, _w0, vget_low_f32(_val1), 0);
                _sum5 = vmlaq_lane_f32(_sum5, _w0, vget_low_f32(_val1), 1);
                _sum6 = vmlaq_lane_f32(_sum6, _w0, vget_high_f32(_val1), 0);
                _sum7 = vmlaq_lane_f32(_sum7, _w0, vget_high_f32(_val1), 1);
#endif

                tmpptr += 8;
                kptr0 += 4;
            }

            vst1_u16(outptr0, vcvt_bf16_f32(_sum0));
            vst1_u16(outptr0 + 4, vcvt_bf16_f32(_sum1));
            vst1_u16(outptr0 + 8, vcvt_bf16_f32(_sum2));
            vst1_u16(outptr0 + 12, vcvt_bf16_f32(_sum3));
            vst1_u16(outptr0 + 16, vcvt_bf16_f32(_sum4));
            vst1_u16(outptr0 + 20, vcvt_bf16_f32(_sum5));
            vst1_u16(outptr0 + 24, vcvt_bf16_f32(_sum6));
            vst1_u16(outptr0 + 28, vcvt_bf16_f32(_sum7));
            outptr0 += 32;
#else
            float sum0_0 = biasptr[0];
            float sum0_1 = biasptr[0];
            float sum0_2 = biasptr[0];
            float sum0_3 = biasptr[0];
            float sum0_4 = biasptr[0];
            float sum0_5 = biasptr[0];
            float sum0_6 = biasptr[0];
            float sum0_7 = biasptr[0];

            float sum1_0 = biasptr[1];
            float sum1_1 = biasptr[1];
            float sum1_2 = biasptr[1];
            float sum1_3 = biasptr[1];
            float sum1_4 = biasptr[1];
            float sum1_5 = biasptr[1];
            float sum1_6 = biasptr[1];
            float sum1_7 = biasptr[1];

            float sum2_0 = biasptr[2];
            float sum2_1 = biasptr[2];
            float sum2_2 = biasptr[2];
            float sum2_3 = biasptr[2];
            float sum2_4 = biasptr[2];
            float sum2_5 = biasptr[2];
            float sum2_6 = biasptr[2];
            float sum2_7 = biasptr[2];

            float sum3_0 = biasptr[3];
            float sum3_1 = biasptr[3];
            float sum3_2 = biasptr[3];
            float sum3_3 = biasptr[3];
            float sum3_4 = biasptr[3];
            float sum3_5 = biasptr[3];
            float sum3_6 = biasptr[3];
            float sum3_7 = biasptr[3];

            for (int q = 0; q < nn; q++)
            {
                sum0_0 += bfloat16_to_float32(tmpptr[0]) * bfloat16_to_float32(kptr[0]);
                sum0_1 += bfloat16_to_float32(tmpptr[1]) * bfloat16_to_float32(kptr[0]);
                sum0_2 += bfloat16_to_float32(tmpptr[2]) * bfloat16_to_float32(kptr[0]);
                sum0_3 += bfloat16_to_float32(tmpptr[3]) * bfloat16_to_float32(kptr[0]);
                sum0_4 += bfloat16_to_float32(tmpptr[4]) * bfloat16_to_float32(kptr[0]);
                sum0_5 += bfloat16_to_float32(tmpptr[5]) * bfloat16_to_float32(kptr[0]);
                sum0_6 += bfloat16_to_float32(tmpptr[6]) * bfloat16_to_float32(kptr[0]);
                sum0_7 += bfloat16_to_float32(tmpptr[7]) * bfloat16_to_float32(kptr[0]);

                sum1_0 += bfloat16_to_float32(tmpptr[0]) * bfloat16_to_float32(kptr[1]);
                sum1_1 += bfloat16_to_float32(tmpptr[1]) * bfloat16_to_float32(kptr[1]);
                sum1_2 += bfloat16_to_float32(tmpptr[2]) * bfloat16_to_float32(kptr[1]);
                sum1_3 += bfloat16_to_float32(tmpptr[3]) * bfloat16_to_float32(kptr[1]);
                sum1_4 += bfloat16_to_float32(tmpptr[4]) * bfloat16_to_float32(kptr[1]);
                sum1_5 += bfloat16_to_float32(tmpptr[5]) * bfloat16_to_float32(kptr[1]);
                sum1_6 += bfloat16_to_float32(tmpptr[6]) * bfloat16_to_float32(kptr[1]);
                sum1_7 += bfloat16_to_float32(tmpptr[7]) * bfloat16_to_float32(kptr[1]);

                sum2_0 += bfloat16_to_float32(tmpptr[0]) * bfloat16_to_float32(kptr[2]);
                sum2_1 += bfloat16_to_float32(tmpptr[1]) * bfloat16_to_float32(kptr[2]);
                sum2_2 += bfloat16_to_float32(tmpptr[2]) * bfloat16_to_float32(kptr[2]);
                sum2_3 += bfloat16_to_float32(tmpptr[3]) * bfloat16_to_float32(kptr[2]);
                sum2_4 += bfloat16_to_float32(tmpptr[4]) * bfloat16_to_float32(kptr[2]);
                sum2_5 += bfloat16_to_float32(tmpptr[5]) * bfloat16_to_float32(kptr[2]);
                sum2_6 += bfloat16_to_float32(tmpptr[6]) * bfloat16_to_float32(kptr[2]);
                sum2_7 += bfloat16_to_float32(tmpptr[7]) * bfloat16_to_float32(kptr[2]);

                sum3_0 += bfloat16_to_float32(tmpptr[0]) * bfloat16_to_float32(kptr[3]);
                sum3_1 += bfloat16_to_float32(tmpptr[1]) * bfloat16_to_float32(kptr[3]);
                sum3_2 += bfloat16_to_float32(tmpptr[2]) * bfloat16_to_float32(kptr[3]);
                sum3_3 += bfloat16_to_float32(tmpptr[3]) * bfloat16_to_float32(kptr[3]);
                sum3_4 += bfloat16_to_float32(tmpptr[4]) * bfloat16_to_float32(kptr[3]);
                sum3_5 += bfloat16_to_float32(tmpptr[5]) * bfloat16_to_float32(kptr[3]);
                sum3_6 += bfloat16_to_float32(tmpptr[6]) * bfloat16_to_float32(kptr[3]);
                sum3_7 += bfloat16_to_float32(tmpptr[7]) * bfloat16_to_float32(kptr[3]);

                tmpptr += 8;
                kptr += 4;
            }

            outptr0[0] = float32_to_bfloat16(sum0_0);
            outptr0[1] = float32_to_bfloat16(sum1_0);
            outptr0[2] = float32_to_bfloat16(sum2_0);
            outptr0[3] = float32_to_bfloat16(sum3_0);

            outptr0[4] = float32_to_bfloat16(sum0_1);
            outptr0[5] = float32_to_bfloat16(sum1_1);
            outptr0[6] = float32_to_bfloat16(sum2_1);
            outptr0[7] = float32_to_bfloat16(sum3_1);

            outptr0[8] = float32_to_bfloat16(sum0_2);
            outptr0[9] = float32_to_bfloat16(sum1_2);
            outptr0[10] = float32_to_bfloat16(sum2_2);
            outptr0[11] = float32_to_bfloat16(sum3_2);

            outptr0[12] = float32_to_bfloat16(sum0_3);
            outptr0[13] = float32_to_bfloat16(sum1_3);
            outptr0[14] = float32_to_bfloat16(sum2_3);
            outptr0[15] = float32_to_bfloat16(sum3_3);

            outptr0[16] = float32_to_bfloat16(sum0_4);
            outptr0[17] = float32_to_bfloat16(sum1_4);
            outptr0[18] = float32_to_bfloat16(sum2_4);
            outptr0[19] = float32_to_bfloat16(sum3_4);

            outptr0[20] = float32_to_bfloat16(sum0_5);
            outptr0[21] = float32_to_bfloat16(sum1_5);
            outptr0[22] = float32_to_bfloat16(sum2_5);
            outptr0[23] = float32_to_bfloat16(sum3_5);

            outptr0[24] = float32_to_bfloat16(sum0_6);
            outptr0[25] = float32_to_bfloat16(sum1_6);
            outptr0[26] = float32_to_bfloat16(sum2_6);
            outptr0[27] = float32_to_bfloat16(sum3_6);

            outptr0[28] = float32_to_bfloat16(sum0_7);
            outptr0[29] = float32_to_bfloat16(sum1_7);
            outptr0[30] = float32_to_bfloat16(sum2_7);
            outptr0[31] = float32_to_bfloat16(sum3_7);

            outptr0 += 32;
#endif // __ARM_NEON
        }
        for (; i + 3 < size; i += 4)
        {
            const unsigned short* tmpptr = tmp.channel(i / 8 + (i % 8) / 4);
#if __ARM_NEON && __aarch64__
            const unsigned short* kptr0 = kernel.channel(p / 2 + p % 2);
#else
            const unsigned short* kptr0 = kernel.channel(p);
#endif

            int nn = inch * maxk; // inch always > 0

#if __ARM_NEON
            float32x4_t _sum0 = vld1q_f32(biasptr);
            float32x4_t _sum1 = vld1q_f32(biasptr);
            float32x4_t _sum2 = vld1q_f32(biasptr);
            float32x4_t _sum3 = vld1q_f32(biasptr);

            for (int j = 0; j < nn; j++)
            {
                float32x4_t _val = vcvt_f32_bf16(vld1_u16(tmpptr));
                float32x4_t _w0 = vcvt_f32_bf16(vld1_u16(kptr0));

#if __aarch64__
                _sum0 = vmlaq_laneq_f32(_sum0, _w0, _val, 0);
                _sum1 = vmlaq_laneq_f32(_sum1, _w0, _val, 1);
                _sum2 = vmlaq_laneq_f32(_sum2, _w0, _val, 2);
                _sum3 = vmlaq_laneq_f32(_sum3, _w0, _val, 3);
#else
                _sum0 = vmlaq_lane_f32(_sum0, _w0, vget_low_f32(_val), 0);
                _sum1 = vmlaq_lane_f32(_sum1, _w0, vget_low_f32(_val), 1);
                _sum2 = vmlaq_lane_f32(_sum2, _w0, vget_high_f32(_val), 0);
                _sum3 = vmlaq_lane_f32(_sum3, _w0, vget_high_f32(_val), 1);
#endif

                tmpptr += 4;
                kptr0 += 4;
            }

            vst1_u16(outptr0, vcvt_bf16_f32(_sum0));
            vst1_u16(outptr0 + 4, vcvt_bf16_f32(_sum1));
            vst1_u16(outptr0 + 8, vcvt_bf16_f32(_sum2));
            vst1_u16(outptr0 + 12, vcvt_bf16_f32(_sum3));
            outptr0 += 16;
#else
            float sum0_0 = biasptr[0];
            float sum0_1 = biasptr[0];
            float sum0_2 = biasptr[0];
            float sum0_3 = biasptr[0];

            float sum1_0 = biasptr[1];
            float sum1_1 = biasptr[1];
            float sum1_2 = biasptr[1];
            float sum1_3 = biasptr[1];

            float sum2_0 = biasptr[2];
            float sum2_1 = biasptr[2];
            float sum2_2 = biasptr[2];
            float sum2_3 = biasptr[2];

            float sum3_0 = biasptr[3];
            float sum3_1 = biasptr[3];
            float sum3_2 = biasptr[3];
            float sum3_3 = biasptr[3];

            for (int q = 0; q < nn; q++)
            {
                sum0_0 += bfloat16_to_float32(tmpptr[0]) * bfloat16_to_float32(kptr[0]);
                sum0_1 += bfloat16_to_float32(tmpptr[1]) * bfloat16_to_float32(kptr[0]);
                sum0_2 += bfloat16_to_float32(tmpptr[2]) * bfloat16_to_float32(kptr[0]);
                sum0_3 += bfloat16_to_float32(tmpptr[3]) * bfloat16_to_float32(kptr[0]);

                sum1_0 += bfloat16_to_float32(tmpptr[0]) * bfloat16_to_float32(kptr[1]);
                sum1_1 += bfloat16_to_float32(tmpptr[1]) * bfloat16_to_float32(kptr[1]);
                sum1_2 += bfloat16_to_float32(tmpptr[2]) * bfloat16_to_float32(kptr[1]);
                sum1_3 += bfloat16_to_float32(tmpptr[3]) * bfloat16_to_float32(kptr[1]);

                sum2_0 += bfloat16_to_float32(tmpptr[0]) * bfloat16_to_float32(kptr[2]);
                sum2_1 += bfloat16_to_float32(tmpptr[1]) * bfloat16_to_float32(kptr[2]);
                sum2_2 += bfloat16_to_float32(tmpptr[2]) * bfloat16_to_float32(kptr[2]);
                sum2_3 += bfloat16_to_float32(tmpptr[3]) * bfloat16_to_float32(kptr[2]);

                sum3_0 += bfloat16_to_float32(tmpptr[0]) * bfloat16_to_float32(kptr[3]);
                sum3_1 += bfloat16_to_float32(tmpptr[1]) * bfloat16_to_float32(kptr[3]);
                sum3_2 += bfloat16_to_float32(tmpptr[2]) * bfloat16_to_float32(kptr[3]);
                sum3_3 += bfloat16_to_float32(tmpptr[3]) * bfloat16_to_float32(kptr[3]);

                tmpptr += 4;
                kptr += 4;
            }

            outptr0[0] = float32_to_bfloat16(sum0_0);
            outptr0[1] = float32_to_bfloat16(sum1_0);
            outptr0[2] = float32_to_bfloat16(sum2_0);
            outptr0[3] = float32_to_bfloat16(sum3_0);

            outptr0[4] = float32_to_bfloat16(sum0_1);
            outptr0[5] = float32_to_bfloat16(sum1_1);
            outptr0[6] = float32_to_bfloat16(sum2_1);
            outptr0[7] = float32_to_bfloat16(sum3_1);

            outptr0[8] = float32_to_bfloat16(sum0_2);
            outptr0[9] = float32_to_bfloat16(sum1_2);
            outptr0[10] = float32_to_bfloat16(sum2_2);
            outptr0[11] = float32_to_bfloat16(sum3_2);

            outptr0[12] = float32_to_bfloat16(sum0_3);
            outptr0[13] = float32_to_bfloat16(sum1_3);
            outptr0[14] = float32_to_bfloat16(sum2_3);
            outptr0[15] = float32_to_bfloat16(sum3_3);

            outptr0 += 16;
#endif // __ARM_NEON
        }
        for (; i < size; i++)
        {
            const unsigned short* tmpptr = tmp.channel(i / 8 + (i % 8) / 4 + i % 4);
#if __ARM_NEON && __aarch64__
            const unsigned short* kptr0 = kernel.channel(p / 2 + p % 2);
#else
            const unsigned short* kptr0 = kernel.channel(p);
#endif

            int nn = inch * maxk; // inch always > 0

#if __ARM_NEON
            float32x4_t _sum = vld1q_f32(biasptr);

            for (int j = 0; j < nn; j++)
            {
                float32x4_t _val = vdupq_n_f32(bfloat16_to_float32(tmpptr[0]));
                float32x4_t _w0 = vcvt_f32_bf16(vld1_u16(kptr0));
                _sum = vmlaq_f32(_sum, _val, _w0);

                tmpptr += 1;
                kptr0 += 4;
            }

            vst1_u16(outptr0, vcvt_bf16_f32(_sum));
            outptr0 += 4;
#else
            float sum0 = biasptr[0];
            float sum1 = biasptr[1];
            float sum2 = biasptr[2];
            float sum3 = biasptr[3];

            for (int q = 0; q < nn; q++)
            {
                sum0 += bfloat16_to_float32(tmpptr[0]) * bfloat16_to_float32(kptr[0]);
                sum1 += bfloat16_to_float32(tmpptr[0]) * bfloat16_to_float32(kptr[1]);
                sum2 += bfloat16_to_float32(tmpptr[0]) * bfloat16_to_float32(kptr[2]);
                sum3 += bfloat16_to_float32(tmpptr[0]) * bfloat16_to_float32(kptr[3]);

                tmpptr++;
                kptr += 4;
            }

            outptr0[0] = float32_to_bfloat16(sum0);
            outptr0[1] = float32_to_bfloat16(sum1);
            outptr0[2] = float32_to_bfloat16(sum2);
            outptr0[3] = float32_to_bfloat16(sum3);

            outptr0 += 4;
#endif // __ARM_NEON
        }
    }
}

static void convolution_im2col_sgemm_transform_kernel_pack1to4_bf16s_neon(const Mat& _kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = 4b-4a-maxk-inch/4a-outch/4b
    Mat kernel = _kernel.reshape(maxk, inch, outch);
#if __ARM_NEON && __aarch64__
    kernel_tm.create(8 * maxk, inch, outch / 8 + (outch % 8) / 4, (size_t)2u);
#else
    kernel_tm.create(4 * maxk, inch, outch / 4, (size_t)2u);
#endif

    int q = 0;
#if __ARM_NEON && __aarch64__
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

        unsigned short* g00 = kernel_tm.channel(q / 8);

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
                g00[0] = float32_to_bfloat16(k00[k]);
                g00[1] = float32_to_bfloat16(k10[k]);
                g00[2] = float32_to_bfloat16(k20[k]);
                g00[3] = float32_to_bfloat16(k30[k]);
                g00[4] = float32_to_bfloat16(k40[k]);
                g00[5] = float32_to_bfloat16(k50[k]);
                g00[6] = float32_to_bfloat16(k60[k]);
                g00[7] = float32_to_bfloat16(k70[k]);

                g00 += 8;
            }
        }
    }
#endif // __ARM_NEON && __aarch64__
    for (; q + 3 < outch; q += 4)
    {
        const Mat k0 = kernel.channel(q);
        const Mat k1 = kernel.channel(q + 1);
        const Mat k2 = kernel.channel(q + 2);
        const Mat k3 = kernel.channel(q + 3);

#if __ARM_NEON && __aarch64__
        unsigned short* g00 = kernel_tm.channel(q / 8 + (q % 8) / 4);
#else
        unsigned short* g00 = kernel_tm.channel(q / 4);
#endif

        for (int p = 0; p < inch; p++)
        {
            const float* k00 = k0.row(p);
            const float* k10 = k1.row(p);
            const float* k20 = k2.row(p);
            const float* k30 = k3.row(p);

            for (int k = 0; k < maxk; k++)
            {
                g00[0] = float32_to_bfloat16(k00[k]);
                g00[1] = float32_to_bfloat16(k10[k]);
                g00[2] = float32_to_bfloat16(k20[k]);
                g00[3] = float32_to_bfloat16(k30[k]);

                g00 += 4;
            }
        }
    }
}

static void convolution_im2col_sgemm_pack1to4_bf16s_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
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
            unsigned short* ptr = bottom_im2col.channel(p);

            for (int u = 0; u < kernel_h; u++)
            {
                for (int v = 0; v < kernel_w; v++)
                {
                    const unsigned short* sptr = img.row<const unsigned short>(dilation_h * u) + dilation_w * v;

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

    im2col_sgemm_pack1to4_bf16s_neon(bottom_im2col, top_blob, kernel, _bias, opt);
}
