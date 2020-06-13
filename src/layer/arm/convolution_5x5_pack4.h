// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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

static void conv5x5s1_pack4_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int inch = bottom_blob.c;
    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        float32x4_t _bias0 = bias ? vld1q_f32((const float*)bias + p * 4) : vdupq_n_f32(0.f);
        out0.fill(_bias0);

        for (int q = 0; q < inch; q++)
        {
            float* outptr0 = out0.row(0);

            const Mat img0 = bottom_blob.channel(q);

            const float* r0 = img0.row(0);
            const float* r1 = img0.row(1);
            const float* r2 = img0.row(2);
            const float* r3 = img0.row(3);
            const float* r4 = img0.row(4);

            const float* kptr = (const float*)kernel.channel(p).row(q);

            int i = 0;
            for (; i < outh; i++)
            {
                int j = 0;
                for (; j + 3 < outw; j += 4)
                {
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%0, #512]       \n"
                        "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0] \n" // sum0 sum1 sum2 sum3

                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%1], #64 \n" // r00 r01 r02 r03

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v16.4s, v0.s[0]     \n"
                        "fmla   v21.4s, v16.4s, v1.s[0]     \n"
                        "fmla   v22.4s, v16.4s, v2.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v3.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v0.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v1.s[1]     \n"
                        "fmla   v22.4s, v17.4s, v2.s[1]     \n"
                        "fmla   v23.4s, v17.4s, v3.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v18.4s, v0.s[2]     \n"
                        "fmla   v21.4s, v18.4s, v1.s[2]     \n"
                        "fmla   v22.4s, v18.4s, v2.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v3.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v0.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v1.s[3]     \n"
                        "fmla   v22.4s, v19.4s, v2.s[3]     \n"
                        "fmla   v23.4s, v19.4s, v3.s[3]     \n"

                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%1] \n" // r04 r05 r06 r07

                        "fmla   v20.4s, v24.4s, v1.s[0]     \n"
                        "fmla   v21.4s, v24.4s, v2.s[0]     \n"
                        "fmla   v22.4s, v24.4s, v3.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v1.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v2.s[1]     \n"
                        "fmla   v22.4s, v25.4s, v3.s[1]     \n"
                        "fmla   v23.4s, v25.4s, v4.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v26.4s, v1.s[2]     \n"
                        "fmla   v21.4s, v26.4s, v2.s[2]     \n"
                        "fmla   v22.4s, v26.4s, v3.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v4.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v1.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v2.s[3]     \n"
                        "fmla   v22.4s, v27.4s, v3.s[3]     \n"
                        "fmla   v23.4s, v27.4s, v4.s[3]     \n"

                        "fmla   v20.4s, v16.4s, v2.s[0]     \n"
                        "fmla   v21.4s, v16.4s, v3.s[0]     \n"
                        "fmla   v22.4s, v16.4s, v4.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v5.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v2.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v3.s[1]     \n"
                        "fmla   v22.4s, v17.4s, v4.s[1]     \n"
                        "fmla   v23.4s, v17.4s, v5.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v18.4s, v2.s[2]     \n"
                        "fmla   v21.4s, v18.4s, v3.s[2]     \n"
                        "fmla   v22.4s, v18.4s, v4.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v5.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v2.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v3.s[3]     \n"
                        "fmla   v22.4s, v19.4s, v4.s[3]     \n"
                        "fmla   v23.4s, v19.4s, v5.s[3]     \n"

                        "fmla   v20.4s, v24.4s, v3.s[0]     \n"
                        "fmla   v21.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v22.4s, v24.4s, v5.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v6.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v3.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v4.s[1]     \n"
                        "fmla   v22.4s, v25.4s, v5.s[1]     \n"
                        "fmla   v23.4s, v25.4s, v6.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v26.4s, v3.s[2]     \n"
                        "fmla   v21.4s, v26.4s, v4.s[2]     \n"
                        "fmla   v22.4s, v26.4s, v5.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v6.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v3.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v4.s[3]     \n"
                        "fmla   v22.4s, v27.4s, v5.s[3]     \n"
                        "fmla   v23.4s, v27.4s, v6.s[3]     \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%2], #64 \n" // r10 r11 r12 r13

                        "fmla   v20.4s, v16.4s, v4.s[0]     \n"
                        "fmla   v21.4s, v16.4s, v5.s[0]     \n"
                        "fmla   v22.4s, v16.4s, v6.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v7.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v4.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v5.s[1]     \n"
                        "fmla   v22.4s, v17.4s, v6.s[1]     \n"
                        "fmla   v23.4s, v17.4s, v7.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v18.4s, v4.s[2]     \n"
                        "fmla   v21.4s, v18.4s, v5.s[2]     \n"
                        "fmla   v22.4s, v18.4s, v6.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v7.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v4.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v5.s[3]     \n"
                        "fmla   v22.4s, v19.4s, v6.s[3]     \n"
                        "fmla   v23.4s, v19.4s, v7.s[3]     \n"

                        "fmla   v20.4s, v24.4s, v0.s[0]     \n"
                        "fmla   v21.4s, v24.4s, v1.s[0]     \n"
                        "fmla   v22.4s, v24.4s, v2.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v3.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v0.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v1.s[1]     \n"
                        "fmla   v22.4s, v25.4s, v2.s[1]     \n"
                        "fmla   v23.4s, v25.4s, v3.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v26.4s, v0.s[2]     \n"
                        "fmla   v21.4s, v26.4s, v1.s[2]     \n"
                        "fmla   v22.4s, v26.4s, v2.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v3.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v0.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v1.s[3]     \n"
                        "fmla   v22.4s, v27.4s, v2.s[3]     \n"
                        "fmla   v23.4s, v27.4s, v3.s[3]     \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%2] \n" // r14 r15 r16 r17

                        "fmla   v20.4s, v16.4s, v1.s[0]     \n"
                        "fmla   v21.4s, v16.4s, v2.s[0]     \n"
                        "fmla   v22.4s, v16.4s, v3.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v4.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v1.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v2.s[1]     \n"
                        "fmla   v22.4s, v17.4s, v3.s[1]     \n"
                        "fmla   v23.4s, v17.4s, v4.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v18.4s, v1.s[2]     \n"
                        "fmla   v21.4s, v18.4s, v2.s[2]     \n"
                        "fmla   v22.4s, v18.4s, v3.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v4.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v1.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v2.s[3]     \n"
                        "fmla   v22.4s, v19.4s, v3.s[3]     \n"
                        "fmla   v23.4s, v19.4s, v4.s[3]     \n"

                        "fmla   v20.4s, v24.4s, v2.s[0]     \n"
                        "fmla   v21.4s, v24.4s, v3.s[0]     \n"
                        "fmla   v22.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v5.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v2.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v3.s[1]     \n"
                        "fmla   v22.4s, v25.4s, v4.s[1]     \n"
                        "fmla   v23.4s, v25.4s, v5.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v26.4s, v2.s[2]     \n"
                        "fmla   v21.4s, v26.4s, v3.s[2]     \n"
                        "fmla   v22.4s, v26.4s, v4.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v5.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v2.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v3.s[3]     \n"
                        "fmla   v22.4s, v27.4s, v4.s[3]     \n"
                        "fmla   v23.4s, v27.4s, v5.s[3]     \n"

                        "fmla   v20.4s, v16.4s, v3.s[0]     \n"
                        "fmla   v21.4s, v16.4s, v4.s[0]     \n"
                        "fmla   v22.4s, v16.4s, v5.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v6.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v3.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v4.s[1]     \n"
                        "fmla   v22.4s, v17.4s, v5.s[1]     \n"
                        "fmla   v23.4s, v17.4s, v6.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v18.4s, v3.s[2]     \n"
                        "fmla   v21.4s, v18.4s, v4.s[2]     \n"
                        "fmla   v22.4s, v18.4s, v5.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v6.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v3.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v4.s[3]     \n"
                        "fmla   v22.4s, v19.4s, v5.s[3]     \n"
                        "fmla   v23.4s, v19.4s, v6.s[3]     \n"

                        "prfm   pldl1keep, [%3, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%3], #64 \n" // r20 r21 r22 r23

                        "fmla   v20.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v21.4s, v24.4s, v5.s[0]     \n"
                        "fmla   v22.4s, v24.4s, v6.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v7.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v4.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v5.s[1]     \n"
                        "fmla   v22.4s, v25.4s, v6.s[1]     \n"
                        "fmla   v23.4s, v25.4s, v7.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v26.4s, v4.s[2]     \n"
                        "fmla   v21.4s, v26.4s, v5.s[2]     \n"
                        "fmla   v22.4s, v26.4s, v6.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v7.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v4.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v5.s[3]     \n"
                        "fmla   v22.4s, v27.4s, v6.s[3]     \n"
                        "fmla   v23.4s, v27.4s, v7.s[3]     \n"

                        "fmla   v20.4s, v16.4s, v0.s[0]     \n"
                        "fmla   v21.4s, v16.4s, v1.s[0]     \n"
                        "fmla   v22.4s, v16.4s, v2.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v3.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v0.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v1.s[1]     \n"
                        "fmla   v22.4s, v17.4s, v2.s[1]     \n"
                        "fmla   v23.4s, v17.4s, v3.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v18.4s, v0.s[2]     \n"
                        "fmla   v21.4s, v18.4s, v1.s[2]     \n"
                        "fmla   v22.4s, v18.4s, v2.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v3.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v0.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v1.s[3]     \n"
                        "fmla   v22.4s, v19.4s, v2.s[3]     \n"
                        "fmla   v23.4s, v19.4s, v3.s[3]     \n"

                        "prfm   pldl1keep, [%3, #512]       \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%3] \n" // r24 r25 r26 r27

                        "fmla   v20.4s, v24.4s, v1.s[0]     \n"
                        "fmla   v21.4s, v24.4s, v2.s[0]     \n"
                        "fmla   v22.4s, v24.4s, v3.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v1.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v2.s[1]     \n"
                        "fmla   v22.4s, v25.4s, v3.s[1]     \n"
                        "fmla   v23.4s, v25.4s, v4.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v26.4s, v1.s[2]     \n"
                        "fmla   v21.4s, v26.4s, v2.s[2]     \n"
                        "fmla   v22.4s, v26.4s, v3.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v4.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v1.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v2.s[3]     \n"
                        "fmla   v22.4s, v27.4s, v3.s[3]     \n"
                        "fmla   v23.4s, v27.4s, v4.s[3]     \n"

                        "fmla   v20.4s, v16.4s, v2.s[0]     \n"
                        "fmla   v21.4s, v16.4s, v3.s[0]     \n"
                        "fmla   v22.4s, v16.4s, v4.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v5.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v2.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v3.s[1]     \n"
                        "fmla   v22.4s, v17.4s, v4.s[1]     \n"
                        "fmla   v23.4s, v17.4s, v5.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v18.4s, v2.s[2]     \n"
                        "fmla   v21.4s, v18.4s, v3.s[2]     \n"
                        "fmla   v22.4s, v18.4s, v4.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v5.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v2.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v3.s[3]     \n"
                        "fmla   v22.4s, v19.4s, v4.s[3]     \n"
                        "fmla   v23.4s, v19.4s, v5.s[3]     \n"

                        "fmla   v20.4s, v24.4s, v3.s[0]     \n"
                        "fmla   v21.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v22.4s, v24.4s, v5.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v6.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v3.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v4.s[1]     \n"
                        "fmla   v22.4s, v25.4s, v5.s[1]     \n"
                        "fmla   v23.4s, v25.4s, v6.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v26.4s, v3.s[2]     \n"
                        "fmla   v21.4s, v26.4s, v4.s[2]     \n"
                        "fmla   v22.4s, v26.4s, v5.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v6.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v3.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v4.s[3]     \n"
                        "fmla   v22.4s, v27.4s, v5.s[3]     \n"
                        "fmla   v23.4s, v27.4s, v6.s[3]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%4], #64 \n" // r30 r31 r32 r33

                        "fmla   v20.4s, v16.4s, v4.s[0]     \n"
                        "fmla   v21.4s, v16.4s, v5.s[0]     \n"
                        "fmla   v22.4s, v16.4s, v6.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v7.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v4.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v5.s[1]     \n"
                        "fmla   v22.4s, v17.4s, v6.s[1]     \n"
                        "fmla   v23.4s, v17.4s, v7.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v18.4s, v4.s[2]     \n"
                        "fmla   v21.4s, v18.4s, v5.s[2]     \n"
                        "fmla   v22.4s, v18.4s, v6.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v7.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v4.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v5.s[3]     \n"
                        "fmla   v22.4s, v19.4s, v6.s[3]     \n"
                        "fmla   v23.4s, v19.4s, v7.s[3]     \n"

                        "fmla   v20.4s, v24.4s, v0.s[0]     \n"
                        "fmla   v21.4s, v24.4s, v1.s[0]     \n"
                        "fmla   v22.4s, v24.4s, v2.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v3.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v0.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v1.s[1]     \n"
                        "fmla   v22.4s, v25.4s, v2.s[1]     \n"
                        "fmla   v23.4s, v25.4s, v3.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v26.4s, v0.s[2]     \n"
                        "fmla   v21.4s, v26.4s, v1.s[2]     \n"
                        "fmla   v22.4s, v26.4s, v2.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v3.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v0.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v1.s[3]     \n"
                        "fmla   v22.4s, v27.4s, v2.s[3]     \n"
                        "fmla   v23.4s, v27.4s, v3.s[3]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%4] \n" // r34 r35 r36 r37

                        "fmla   v20.4s, v16.4s, v1.s[0]     \n"
                        "fmla   v21.4s, v16.4s, v2.s[0]     \n"
                        "fmla   v22.4s, v16.4s, v3.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v4.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v1.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v2.s[1]     \n"
                        "fmla   v22.4s, v17.4s, v3.s[1]     \n"
                        "fmla   v23.4s, v17.4s, v4.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v18.4s, v1.s[2]     \n"
                        "fmla   v21.4s, v18.4s, v2.s[2]     \n"
                        "fmla   v22.4s, v18.4s, v3.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v4.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v1.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v2.s[3]     \n"
                        "fmla   v22.4s, v19.4s, v3.s[3]     \n"
                        "fmla   v23.4s, v19.4s, v4.s[3]     \n"

                        "fmla   v20.4s, v24.4s, v2.s[0]     \n"
                        "fmla   v21.4s, v24.4s, v3.s[0]     \n"
                        "fmla   v22.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v5.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v2.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v3.s[1]     \n"
                        "fmla   v22.4s, v25.4s, v4.s[1]     \n"
                        "fmla   v23.4s, v25.4s, v5.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v26.4s, v2.s[2]     \n"
                        "fmla   v21.4s, v26.4s, v3.s[2]     \n"
                        "fmla   v22.4s, v26.4s, v4.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v5.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v2.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v3.s[3]     \n"
                        "fmla   v22.4s, v27.4s, v4.s[3]     \n"
                        "fmla   v23.4s, v27.4s, v5.s[3]     \n"

                        "fmla   v20.4s, v16.4s, v3.s[0]     \n"
                        "fmla   v21.4s, v16.4s, v4.s[0]     \n"
                        "fmla   v22.4s, v16.4s, v5.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v6.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v3.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v4.s[1]     \n"
                        "fmla   v22.4s, v17.4s, v5.s[1]     \n"
                        "fmla   v23.4s, v17.4s, v6.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v18.4s, v3.s[2]     \n"
                        "fmla   v21.4s, v18.4s, v4.s[2]     \n"
                        "fmla   v22.4s, v18.4s, v5.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v6.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v3.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v4.s[3]     \n"
                        "fmla   v22.4s, v19.4s, v5.s[3]     \n"
                        "fmla   v23.4s, v19.4s, v6.s[3]     \n"

                        "prfm   pldl1keep, [%5, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%5], #64 \n" // r40 r41 r42 r43

                        "fmla   v20.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v21.4s, v24.4s, v5.s[0]     \n"
                        "fmla   v22.4s, v24.4s, v6.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v7.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v4.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v5.s[1]     \n"
                        "fmla   v22.4s, v25.4s, v6.s[1]     \n"
                        "fmla   v23.4s, v25.4s, v7.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v26.4s, v4.s[2]     \n"
                        "fmla   v21.4s, v26.4s, v5.s[2]     \n"
                        "fmla   v22.4s, v26.4s, v6.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v7.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v4.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v5.s[3]     \n"
                        "fmla   v22.4s, v27.4s, v6.s[3]     \n"
                        "fmla   v23.4s, v27.4s, v7.s[3]     \n"

                        "fmla   v20.4s, v16.4s, v0.s[0]     \n"
                        "fmla   v21.4s, v16.4s, v1.s[0]     \n"
                        "fmla   v22.4s, v16.4s, v2.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v3.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v0.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v1.s[1]     \n"
                        "fmla   v22.4s, v17.4s, v2.s[1]     \n"
                        "fmla   v23.4s, v17.4s, v3.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v18.4s, v0.s[2]     \n"
                        "fmla   v21.4s, v18.4s, v1.s[2]     \n"
                        "fmla   v22.4s, v18.4s, v2.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v3.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v0.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v1.s[3]     \n"
                        "fmla   v22.4s, v19.4s, v2.s[3]     \n"
                        "fmla   v23.4s, v19.4s, v3.s[3]     \n"

                        "prfm   pldl1keep, [%5, #512]       \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%5] \n" // r44 r45 r46 r47

                        "fmla   v20.4s, v24.4s, v1.s[0]     \n"
                        "fmla   v21.4s, v24.4s, v2.s[0]     \n"
                        "fmla   v22.4s, v24.4s, v3.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v1.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v2.s[1]     \n"
                        "fmla   v22.4s, v25.4s, v3.s[1]     \n"
                        "fmla   v23.4s, v25.4s, v4.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v26.4s, v1.s[2]     \n"
                        "fmla   v21.4s, v26.4s, v2.s[2]     \n"
                        "fmla   v22.4s, v26.4s, v3.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v4.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v1.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v2.s[3]     \n"
                        "fmla   v22.4s, v27.4s, v3.s[3]     \n"
                        "fmla   v23.4s, v27.4s, v4.s[3]     \n"

                        "fmla   v20.4s, v16.4s, v2.s[0]     \n"
                        "fmla   v21.4s, v16.4s, v3.s[0]     \n"
                        "fmla   v22.4s, v16.4s, v4.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v5.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v2.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v3.s[1]     \n"
                        "fmla   v22.4s, v17.4s, v4.s[1]     \n"
                        "fmla   v23.4s, v17.4s, v5.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v18.4s, v2.s[2]     \n"
                        "fmla   v21.4s, v18.4s, v3.s[2]     \n"
                        "fmla   v22.4s, v18.4s, v4.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v5.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v2.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v3.s[3]     \n"
                        "fmla   v22.4s, v19.4s, v4.s[3]     \n"
                        "fmla   v23.4s, v19.4s, v5.s[3]     \n"

                        "fmla   v20.4s, v24.4s, v3.s[0]     \n"
                        "fmla   v21.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v22.4s, v24.4s, v5.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v6.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v3.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v4.s[1]     \n"
                        "fmla   v22.4s, v25.4s, v5.s[1]     \n"
                        "fmla   v23.4s, v25.4s, v6.s[1]     \n"

                        //                         "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6] \n"

                        "fmla   v20.4s, v26.4s, v3.s[2]     \n"
                        "fmla   v21.4s, v26.4s, v4.s[2]     \n"
                        "fmla   v22.4s, v26.4s, v5.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v6.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v3.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v4.s[3]     \n"
                        "fmla   v22.4s, v27.4s, v5.s[3]     \n"
                        "fmla   v23.4s, v27.4s, v6.s[3]     \n"

                        "fmla   v20.4s, v16.4s, v4.s[0]     \n"
                        "fmla   v21.4s, v16.4s, v5.s[0]     \n"
                        "fmla   v22.4s, v16.4s, v6.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v7.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v4.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v5.s[1]     \n"
                        "fmla   v22.4s, v17.4s, v6.s[1]     \n"
                        "fmla   v23.4s, v17.4s, v7.s[1]     \n"
                        "fmla   v20.4s, v18.4s, v4.s[2]     \n"
                        "fmla   v21.4s, v18.4s, v5.s[2]     \n"
                        "fmla   v22.4s, v18.4s, v6.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v7.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v4.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v5.s[3]     \n"
                        "fmla   v22.4s, v19.4s, v6.s[3]     \n"
                        "fmla   v23.4s, v19.4s, v7.s[3]     \n"

                        "sub    %6, %6, #1536               \n" // kptr -= 24 * 16;

                        "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2),      // %3
                        "=r"(r3),      // %4
                        "=r"(r4),      // %5
                        "=r"(kptr)     // %6
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "4"(r3),
                        "5"(r4),
                        "6"(kptr)
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27");
#else  // __aarch64__
                    asm volatile(
                        "pld        [%0, #512]          \n"
                        "vldm       %0, {d24-d31}       \n" // sum0 sum1 sum2 sum3

                        "pld        [%1, #512]          \n"
                        "vldm       %1!, {d0-d7}        \n" // r00 r01 r02 r03

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q12, q8, d0[0]      \n"
                        "vmla.f32   q13, q8, d2[0]      \n"
                        "vmla.f32   q14, q8, d4[0]      \n"
                        "vmla.f32   q15, q8, d6[0]      \n"
                        "vmla.f32   q12, q9, d0[1]      \n"
                        "vmla.f32   q13, q9, d2[1]      \n"
                        "vmla.f32   q14, q9, d4[1]      \n"
                        "vmla.f32   q15, q9, d6[1]      \n"
                        "vmla.f32   q12, q10, d1[0]     \n"
                        "vmla.f32   q13, q10, d3[0]     \n"
                        "vmla.f32   q14, q10, d5[0]     \n"
                        "vmla.f32   q15, q10, d7[0]     \n"
                        "vmla.f32   q12, q11, d1[1]     \n"
                        "vmla.f32   q13, q11, d3[1]     \n"
                        "vmla.f32   q14, q11, d5[1]     \n"
                        "vmla.f32   q15, q11, d7[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%1, #512]          \n"
                        "vldm       %1, {d8-d15}        \n" // r04 r05 r06 r07

                        "vmla.f32   q12, q8, d2[0]      \n"
                        "vmla.f32   q13, q8, d4[0]      \n"
                        "vmla.f32   q14, q8, d6[0]      \n"
                        "vmla.f32   q15, q8, d8[0]      \n"
                        "vmla.f32   q12, q9, d2[1]      \n"
                        "vmla.f32   q13, q9, d4[1]      \n"
                        "vmla.f32   q14, q9, d6[1]      \n"
                        "vmla.f32   q15, q9, d8[1]      \n"
                        "vmla.f32   q12, q10, d3[0]     \n"
                        "vmla.f32   q13, q10, d5[0]     \n"
                        "vmla.f32   q14, q10, d7[0]     \n"
                        "vmla.f32   q15, q10, d9[0]     \n"
                        "vmla.f32   q12, q11, d3[1]     \n"
                        "vmla.f32   q13, q11, d5[1]     \n"
                        "vmla.f32   q14, q11, d7[1]     \n"
                        "vmla.f32   q15, q11, d9[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q12, q8, d4[0]      \n"
                        "vmla.f32   q13, q8, d6[0]      \n"
                        "vmla.f32   q14, q8, d8[0]      \n"
                        "vmla.f32   q15, q8, d10[0]     \n"
                        "vmla.f32   q12, q9, d4[1]      \n"
                        "vmla.f32   q13, q9, d6[1]      \n"
                        "vmla.f32   q14, q9, d8[1]      \n"
                        "vmla.f32   q15, q9, d10[1]     \n"
                        "vmla.f32   q12, q10, d5[0]     \n"
                        "vmla.f32   q13, q10, d7[0]     \n"
                        "vmla.f32   q14, q10, d9[0]     \n"
                        "vmla.f32   q15, q10, d11[0]    \n"
                        "vmla.f32   q12, q11, d5[1]     \n"
                        "vmla.f32   q13, q11, d7[1]     \n"
                        "vmla.f32   q14, q11, d9[1]     \n"
                        "vmla.f32   q15, q11, d11[1]    \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q12, q8, d6[0]      \n"
                        "vmla.f32   q13, q8, d8[0]      \n"
                        "vmla.f32   q14, q8, d10[0]     \n"
                        "vmla.f32   q15, q8, d12[0]     \n"
                        "vmla.f32   q12, q9, d6[1]      \n"
                        "vmla.f32   q13, q9, d8[1]      \n"
                        "vmla.f32   q14, q9, d10[1]     \n"
                        "vmla.f32   q15, q9, d12[1]     \n"
                        "vmla.f32   q12, q10, d7[0]     \n"
                        "vmla.f32   q13, q10, d9[0]     \n"
                        "vmla.f32   q14, q10, d11[0]    \n"
                        "vmla.f32   q15, q10, d13[0]    \n"
                        "vmla.f32   q12, q11, d7[1]     \n"
                        "vmla.f32   q13, q11, d9[1]     \n"
                        "vmla.f32   q14, q11, d11[1]    \n"
                        "vmla.f32   q15, q11, d13[1]    \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%2, #512]          \n"
                        "vldm       %2!, {d0-d7}        \n" // r10 r11 r12 r13

                        "vmla.f32   q12, q8, d8[0]      \n"
                        "vmla.f32   q13, q8, d10[0]     \n"
                        "vmla.f32   q14, q8, d12[0]     \n"
                        "vmla.f32   q15, q8, d14[0]     \n"
                        "vmla.f32   q12, q9, d8[1]      \n"
                        "vmla.f32   q13, q9, d10[1]     \n"
                        "vmla.f32   q14, q9, d12[1]     \n"
                        "vmla.f32   q15, q9, d14[1]     \n"
                        "vmla.f32   q12, q10, d9[0]     \n"
                        "vmla.f32   q13, q10, d11[0]    \n"
                        "vmla.f32   q14, q10, d13[0]    \n"
                        "vmla.f32   q15, q10, d15[0]    \n"
                        "vmla.f32   q12, q11, d9[1]     \n"
                        "vmla.f32   q13, q11, d11[1]    \n"
                        "vmla.f32   q14, q11, d13[1]    \n"
                        "vmla.f32   q15, q11, d15[1]    \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q12, q8, d0[0]      \n"
                        "vmla.f32   q13, q8, d2[0]      \n"
                        "vmla.f32   q14, q8, d4[0]      \n"
                        "vmla.f32   q15, q8, d6[0]      \n"
                        "vmla.f32   q12, q9, d0[1]      \n"
                        "vmla.f32   q13, q9, d2[1]      \n"
                        "vmla.f32   q14, q9, d4[1]      \n"
                        "vmla.f32   q15, q9, d6[1]      \n"
                        "vmla.f32   q12, q10, d1[0]     \n"
                        "vmla.f32   q13, q10, d3[0]     \n"
                        "vmla.f32   q14, q10, d5[0]     \n"
                        "vmla.f32   q15, q10, d7[0]     \n"
                        "vmla.f32   q12, q11, d1[1]     \n"
                        "vmla.f32   q13, q11, d3[1]     \n"
                        "vmla.f32   q14, q11, d5[1]     \n"
                        "vmla.f32   q15, q11, d7[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%2, #512]          \n"
                        "vldm       %2, {d8-d15}        \n" // r14 r15 r16 r17

                        "vmla.f32   q12, q8, d2[0]      \n"
                        "vmla.f32   q13, q8, d4[0]      \n"
                        "vmla.f32   q14, q8, d6[0]      \n"
                        "vmla.f32   q15, q8, d8[0]      \n"
                        "vmla.f32   q12, q9, d2[1]      \n"
                        "vmla.f32   q13, q9, d4[1]      \n"
                        "vmla.f32   q14, q9, d6[1]      \n"
                        "vmla.f32   q15, q9, d8[1]      \n"
                        "vmla.f32   q12, q10, d3[0]     \n"
                        "vmla.f32   q13, q10, d5[0]     \n"
                        "vmla.f32   q14, q10, d7[0]     \n"
                        "vmla.f32   q15, q10, d9[0]     \n"
                        "vmla.f32   q12, q11, d3[1]     \n"
                        "vmla.f32   q13, q11, d5[1]     \n"
                        "vmla.f32   q14, q11, d7[1]     \n"
                        "vmla.f32   q15, q11, d9[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q12, q8, d4[0]      \n"
                        "vmla.f32   q13, q8, d6[0]      \n"
                        "vmla.f32   q14, q8, d8[0]      \n"
                        "vmla.f32   q15, q8, d10[0]     \n"
                        "vmla.f32   q12, q9, d4[1]      \n"
                        "vmla.f32   q13, q9, d6[1]      \n"
                        "vmla.f32   q14, q9, d8[1]      \n"
                        "vmla.f32   q15, q9, d10[1]     \n"
                        "vmla.f32   q12, q10, d5[0]     \n"
                        "vmla.f32   q13, q10, d7[0]     \n"
                        "vmla.f32   q14, q10, d9[0]     \n"
                        "vmla.f32   q15, q10, d11[0]    \n"
                        "vmla.f32   q12, q11, d5[1]     \n"
                        "vmla.f32   q13, q11, d7[1]     \n"
                        "vmla.f32   q14, q11, d9[1]     \n"
                        "vmla.f32   q15, q11, d11[1]    \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q12, q8, d6[0]      \n"
                        "vmla.f32   q13, q8, d8[0]      \n"
                        "vmla.f32   q14, q8, d10[0]     \n"
                        "vmla.f32   q15, q8, d12[0]     \n"
                        "vmla.f32   q12, q9, d6[1]      \n"
                        "vmla.f32   q13, q9, d8[1]      \n"
                        "vmla.f32   q14, q9, d10[1]     \n"
                        "vmla.f32   q15, q9, d12[1]     \n"
                        "vmla.f32   q12, q10, d7[0]     \n"
                        "vmla.f32   q13, q10, d9[0]     \n"
                        "vmla.f32   q14, q10, d11[0]    \n"
                        "vmla.f32   q15, q10, d13[0]    \n"
                        "vmla.f32   q12, q11, d7[1]     \n"
                        "vmla.f32   q13, q11, d9[1]     \n"
                        "vmla.f32   q14, q11, d11[1]    \n"
                        "vmla.f32   q15, q11, d13[1]    \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%3, #512]          \n"
                        "vldm       %3!, {d0-d7}        \n" // r20 r21 r22 r23

                        "vmla.f32   q12, q8, d8[0]      \n"
                        "vmla.f32   q13, q8, d10[0]     \n"
                        "vmla.f32   q14, q8, d12[0]     \n"
                        "vmla.f32   q15, q8, d14[0]     \n"
                        "vmla.f32   q12, q9, d8[1]      \n"
                        "vmla.f32   q13, q9, d10[1]     \n"
                        "vmla.f32   q14, q9, d12[1]     \n"
                        "vmla.f32   q15, q9, d14[1]     \n"
                        "vmla.f32   q12, q10, d9[0]     \n"
                        "vmla.f32   q13, q10, d11[0]    \n"
                        "vmla.f32   q14, q10, d13[0]    \n"
                        "vmla.f32   q15, q10, d15[0]    \n"
                        "vmla.f32   q12, q11, d9[1]     \n"
                        "vmla.f32   q13, q11, d11[1]    \n"
                        "vmla.f32   q14, q11, d13[1]    \n"
                        "vmla.f32   q15, q11, d15[1]    \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q12, q8, d0[0]      \n"
                        "vmla.f32   q13, q8, d2[0]      \n"
                        "vmla.f32   q14, q8, d4[0]      \n"
                        "vmla.f32   q15, q8, d6[0]      \n"
                        "vmla.f32   q12, q9, d0[1]      \n"
                        "vmla.f32   q13, q9, d2[1]      \n"
                        "vmla.f32   q14, q9, d4[1]      \n"
                        "vmla.f32   q15, q9, d6[1]      \n"
                        "vmla.f32   q12, q10, d1[0]     \n"
                        "vmla.f32   q13, q10, d3[0]     \n"
                        "vmla.f32   q14, q10, d5[0]     \n"
                        "vmla.f32   q15, q10, d7[0]     \n"
                        "vmla.f32   q12, q11, d1[1]     \n"
                        "vmla.f32   q13, q11, d3[1]     \n"
                        "vmla.f32   q14, q11, d5[1]     \n"
                        "vmla.f32   q15, q11, d7[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%3, #512]          \n"
                        "vldm       %3, {d8-d15}        \n" // r24 r25 r26 r27

                        "vmla.f32   q12, q8, d2[0]      \n"
                        "vmla.f32   q13, q8, d4[0]      \n"
                        "vmla.f32   q14, q8, d6[0]      \n"
                        "vmla.f32   q15, q8, d8[0]      \n"
                        "vmla.f32   q12, q9, d2[1]      \n"
                        "vmla.f32   q13, q9, d4[1]      \n"
                        "vmla.f32   q14, q9, d6[1]      \n"
                        "vmla.f32   q15, q9, d8[1]      \n"
                        "vmla.f32   q12, q10, d3[0]     \n"
                        "vmla.f32   q13, q10, d5[0]     \n"
                        "vmla.f32   q14, q10, d7[0]     \n"
                        "vmla.f32   q15, q10, d9[0]     \n"
                        "vmla.f32   q12, q11, d3[1]     \n"
                        "vmla.f32   q13, q11, d5[1]     \n"
                        "vmla.f32   q14, q11, d7[1]     \n"
                        "vmla.f32   q15, q11, d9[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q12, q8, d4[0]      \n"
                        "vmla.f32   q13, q8, d6[0]      \n"
                        "vmla.f32   q14, q8, d8[0]      \n"
                        "vmla.f32   q15, q8, d10[0]     \n"
                        "vmla.f32   q12, q9, d4[1]      \n"
                        "vmla.f32   q13, q9, d6[1]      \n"
                        "vmla.f32   q14, q9, d8[1]      \n"
                        "vmla.f32   q15, q9, d10[1]     \n"
                        "vmla.f32   q12, q10, d5[0]     \n"
                        "vmla.f32   q13, q10, d7[0]     \n"
                        "vmla.f32   q14, q10, d9[0]     \n"
                        "vmla.f32   q15, q10, d11[0]    \n"
                        "vmla.f32   q12, q11, d5[1]     \n"
                        "vmla.f32   q13, q11, d7[1]     \n"
                        "vmla.f32   q14, q11, d9[1]     \n"
                        "vmla.f32   q15, q11, d11[1]    \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q12, q8, d6[0]      \n"
                        "vmla.f32   q13, q8, d8[0]      \n"
                        "vmla.f32   q14, q8, d10[0]     \n"
                        "vmla.f32   q15, q8, d12[0]     \n"
                        "vmla.f32   q12, q9, d6[1]      \n"
                        "vmla.f32   q13, q9, d8[1]      \n"
                        "vmla.f32   q14, q9, d10[1]     \n"
                        "vmla.f32   q15, q9, d12[1]     \n"
                        "vmla.f32   q12, q10, d7[0]     \n"
                        "vmla.f32   q13, q10, d9[0]     \n"
                        "vmla.f32   q14, q10, d11[0]    \n"
                        "vmla.f32   q15, q10, d13[0]    \n"
                        "vmla.f32   q12, q11, d7[1]     \n"
                        "vmla.f32   q13, q11, d9[1]     \n"
                        "vmla.f32   q14, q11, d11[1]    \n"
                        "vmla.f32   q15, q11, d13[1]    \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%4, #512]          \n"
                        "vldm       %4!, {d0-d7}        \n" // r30 r31 r32 r33

                        "vmla.f32   q12, q8, d8[0]      \n"
                        "vmla.f32   q13, q8, d10[0]     \n"
                        "vmla.f32   q14, q8, d12[0]     \n"
                        "vmla.f32   q15, q8, d14[0]     \n"
                        "vmla.f32   q12, q9, d8[1]      \n"
                        "vmla.f32   q13, q9, d10[1]     \n"
                        "vmla.f32   q14, q9, d12[1]     \n"
                        "vmla.f32   q15, q9, d14[1]     \n"
                        "vmla.f32   q12, q10, d9[0]     \n"
                        "vmla.f32   q13, q10, d11[0]    \n"
                        "vmla.f32   q14, q10, d13[0]    \n"
                        "vmla.f32   q15, q10, d15[0]    \n"
                        "vmla.f32   q12, q11, d9[1]     \n"
                        "vmla.f32   q13, q11, d11[1]    \n"
                        "vmla.f32   q14, q11, d13[1]    \n"
                        "vmla.f32   q15, q11, d15[1]    \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q12, q8, d0[0]      \n"
                        "vmla.f32   q13, q8, d2[0]      \n"
                        "vmla.f32   q14, q8, d4[0]      \n"
                        "vmla.f32   q15, q8, d6[0]      \n"
                        "vmla.f32   q12, q9, d0[1]      \n"
                        "vmla.f32   q13, q9, d2[1]      \n"
                        "vmla.f32   q14, q9, d4[1]      \n"
                        "vmla.f32   q15, q9, d6[1]      \n"
                        "vmla.f32   q12, q10, d1[0]     \n"
                        "vmla.f32   q13, q10, d3[0]     \n"
                        "vmla.f32   q14, q10, d5[0]     \n"
                        "vmla.f32   q15, q10, d7[0]     \n"
                        "vmla.f32   q12, q11, d1[1]     \n"
                        "vmla.f32   q13, q11, d3[1]     \n"
                        "vmla.f32   q14, q11, d5[1]     \n"
                        "vmla.f32   q15, q11, d7[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%4, #512]          \n"
                        "vldm       %4, {d8-d15}        \n" // r34 r35 r36 r37

                        "vmla.f32   q12, q8, d2[0]      \n"
                        "vmla.f32   q13, q8, d4[0]      \n"
                        "vmla.f32   q14, q8, d6[0]      \n"
                        "vmla.f32   q15, q8, d8[0]      \n"
                        "vmla.f32   q12, q9, d2[1]      \n"
                        "vmla.f32   q13, q9, d4[1]      \n"
                        "vmla.f32   q14, q9, d6[1]      \n"
                        "vmla.f32   q15, q9, d8[1]      \n"
                        "vmla.f32   q12, q10, d3[0]     \n"
                        "vmla.f32   q13, q10, d5[0]     \n"
                        "vmla.f32   q14, q10, d7[0]     \n"
                        "vmla.f32   q15, q10, d9[0]     \n"
                        "vmla.f32   q12, q11, d3[1]     \n"
                        "vmla.f32   q13, q11, d5[1]     \n"
                        "vmla.f32   q14, q11, d7[1]     \n"
                        "vmla.f32   q15, q11, d9[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q12, q8, d4[0]      \n"
                        "vmla.f32   q13, q8, d6[0]      \n"
                        "vmla.f32   q14, q8, d8[0]      \n"
                        "vmla.f32   q15, q8, d10[0]     \n"
                        "vmla.f32   q12, q9, d4[1]      \n"
                        "vmla.f32   q13, q9, d6[1]      \n"
                        "vmla.f32   q14, q9, d8[1]      \n"
                        "vmla.f32   q15, q9, d10[1]     \n"
                        "vmla.f32   q12, q10, d5[0]     \n"
                        "vmla.f32   q13, q10, d7[0]     \n"
                        "vmla.f32   q14, q10, d9[0]     \n"
                        "vmla.f32   q15, q10, d11[0]    \n"
                        "vmla.f32   q12, q11, d5[1]     \n"
                        "vmla.f32   q13, q11, d7[1]     \n"
                        "vmla.f32   q14, q11, d9[1]     \n"
                        "vmla.f32   q15, q11, d11[1]    \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q12, q8, d6[0]      \n"
                        "vmla.f32   q13, q8, d8[0]      \n"
                        "vmla.f32   q14, q8, d10[0]     \n"
                        "vmla.f32   q15, q8, d12[0]     \n"
                        "vmla.f32   q12, q9, d6[1]      \n"
                        "vmla.f32   q13, q9, d8[1]      \n"
                        "vmla.f32   q14, q9, d10[1]     \n"
                        "vmla.f32   q15, q9, d12[1]     \n"
                        "vmla.f32   q12, q10, d7[0]     \n"
                        "vmla.f32   q13, q10, d9[0]     \n"
                        "vmla.f32   q14, q10, d11[0]    \n"
                        "vmla.f32   q15, q10, d13[0]    \n"
                        "vmla.f32   q12, q11, d7[1]     \n"
                        "vmla.f32   q13, q11, d9[1]     \n"
                        "vmla.f32   q14, q11, d11[1]    \n"
                        "vmla.f32   q15, q11, d13[1]    \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%5, #512]          \n"
                        "vldm       %5!, {d0-d7}        \n" // r40 r41 r42 r43

                        "vmla.f32   q12, q8, d8[0]      \n"
                        "vmla.f32   q13, q8, d10[0]     \n"
                        "vmla.f32   q14, q8, d12[0]     \n"
                        "vmla.f32   q15, q8, d14[0]     \n"
                        "vmla.f32   q12, q9, d8[1]      \n"
                        "vmla.f32   q13, q9, d10[1]     \n"
                        "vmla.f32   q14, q9, d12[1]     \n"
                        "vmla.f32   q15, q9, d14[1]     \n"
                        "vmla.f32   q12, q10, d9[0]     \n"
                        "vmla.f32   q13, q10, d11[0]    \n"
                        "vmla.f32   q14, q10, d13[0]    \n"
                        "vmla.f32   q15, q10, d15[0]    \n"
                        "vmla.f32   q12, q11, d9[1]     \n"
                        "vmla.f32   q13, q11, d11[1]    \n"
                        "vmla.f32   q14, q11, d13[1]    \n"
                        "vmla.f32   q15, q11, d15[1]    \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q12, q8, d0[0]      \n"
                        "vmla.f32   q13, q8, d2[0]      \n"
                        "vmla.f32   q14, q8, d4[0]      \n"
                        "vmla.f32   q15, q8, d6[0]      \n"
                        "vmla.f32   q12, q9, d0[1]      \n"
                        "vmla.f32   q13, q9, d2[1]      \n"
                        "vmla.f32   q14, q9, d4[1]      \n"
                        "vmla.f32   q15, q9, d6[1]      \n"
                        "vmla.f32   q12, q10, d1[0]     \n"
                        "vmla.f32   q13, q10, d3[0]     \n"
                        "vmla.f32   q14, q10, d5[0]     \n"
                        "vmla.f32   q15, q10, d7[0]     \n"
                        "vmla.f32   q12, q11, d1[1]     \n"
                        "vmla.f32   q13, q11, d3[1]     \n"
                        "vmla.f32   q14, q11, d5[1]     \n"
                        "vmla.f32   q15, q11, d7[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%5, #512]          \n"
                        "vldm       %5, {d8-d15}        \n" // r44 r45 r46 r47

                        "vmla.f32   q12, q8, d2[0]      \n"
                        "vmla.f32   q13, q8, d4[0]      \n"
                        "vmla.f32   q14, q8, d6[0]      \n"
                        "vmla.f32   q15, q8, d8[0]      \n"
                        "vmla.f32   q12, q9, d2[1]      \n"
                        "vmla.f32   q13, q9, d4[1]      \n"
                        "vmla.f32   q14, q9, d6[1]      \n"
                        "vmla.f32   q15, q9, d8[1]      \n"
                        "vmla.f32   q12, q10, d3[0]     \n"
                        "vmla.f32   q13, q10, d5[0]     \n"
                        "vmla.f32   q14, q10, d7[0]     \n"
                        "vmla.f32   q15, q10, d9[0]     \n"
                        "vmla.f32   q12, q11, d3[1]     \n"
                        "vmla.f32   q13, q11, d5[1]     \n"
                        "vmla.f32   q14, q11, d7[1]     \n"
                        "vmla.f32   q15, q11, d9[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q12, q8, d4[0]      \n"
                        "vmla.f32   q13, q8, d6[0]      \n"
                        "vmla.f32   q14, q8, d8[0]      \n"
                        "vmla.f32   q15, q8, d10[0]     \n"
                        "vmla.f32   q12, q9, d4[1]      \n"
                        "vmla.f32   q13, q9, d6[1]      \n"
                        "vmla.f32   q14, q9, d8[1]      \n"
                        "vmla.f32   q15, q9, d10[1]     \n"
                        "vmla.f32   q12, q10, d5[0]     \n"
                        "vmla.f32   q13, q10, d7[0]     \n"
                        "vmla.f32   q14, q10, d9[0]     \n"
                        "vmla.f32   q15, q10, d11[0]    \n"
                        "vmla.f32   q12, q11, d5[1]     \n"
                        "vmla.f32   q13, q11, d7[1]     \n"
                        "vmla.f32   q14, q11, d9[1]     \n"
                        "vmla.f32   q15, q11, d11[1]    \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q12, q8, d6[0]      \n"
                        "vmla.f32   q13, q8, d8[0]      \n"
                        "vmla.f32   q14, q8, d10[0]     \n"
                        "vmla.f32   q15, q8, d12[0]     \n"
                        "vmla.f32   q12, q9, d6[1]      \n"
                        "vmla.f32   q13, q9, d8[1]      \n"
                        "vmla.f32   q14, q9, d10[1]     \n"
                        "vmla.f32   q15, q9, d12[1]     \n"
                        "vmla.f32   q12, q10, d7[0]     \n"
                        "vmla.f32   q13, q10, d9[0]     \n"
                        "vmla.f32   q14, q10, d11[0]    \n"
                        "vmla.f32   q15, q10, d13[0]    \n"
                        "vmla.f32   q12, q11, d7[1]     \n"
                        "vmla.f32   q13, q11, d9[1]     \n"
                        "vmla.f32   q14, q11, d11[1]    \n"
                        "vmla.f32   q15, q11, d13[1]    \n"

                        //                         "pld        [%6, #512]          \n"
                        "vldm       %6, {d16-d23}       \n"

                        "vmla.f32   q12, q8, d8[0]      \n"
                        "vmla.f32   q13, q8, d10[0]     \n"
                        "vmla.f32   q14, q8, d12[0]     \n"
                        "vmla.f32   q15, q8, d14[0]     \n"
                        "vmla.f32   q12, q9, d8[1]      \n"
                        "vmla.f32   q13, q9, d10[1]     \n"
                        "vmla.f32   q14, q9, d12[1]     \n"
                        "vmla.f32   q15, q9, d14[1]     \n"
                        "vmla.f32   q12, q10, d9[0]     \n"
                        "vmla.f32   q13, q10, d11[0]    \n"
                        "vmla.f32   q14, q10, d13[0]    \n"
                        "vmla.f32   q15, q10, d15[0]    \n"
                        "vmla.f32   q12, q11, d9[1]     \n"
                        "vmla.f32   q13, q11, d11[1]    \n"
                        "vmla.f32   q14, q11, d13[1]    \n"
                        "vmla.f32   q15, q11, d15[1]    \n"

                        "sub        %6, %6, #1536       \n" // kptr -= 24 * 16;

                        "vstm       %0!, {d24-d31}      \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2),      // %3
                        "=r"(r3),      // %4
                        "=r"(r4),      // %5
                        "=r"(kptr)     // %6
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "4"(r3),
                        "5"(r4),
                        "6"(kptr)
                        : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif // __aarch64__
                }
                for (; j + 1 < outw; j += 2)
                {
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%0, #256]       \n"
                        "ld1    {v20.4s, v21.4s}, [%0]      \n" // sum0 sum1

                        "prfm   pldl1keep, [%1, #256]       \n"
                        "ld1    {v0.4s, v1.4s}, [%1], #32   \n" // r00 r01

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmul   v22.4s, v16.4s, v0.s[0]     \n"
                        "fmul   v23.4s, v16.4s, v1.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v0.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v1.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v18.4s, v0.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v1.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v0.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v1.s[3]     \n"

                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v2.4s, v3.4s, v4.4s, v5.4s}, [%1] \n" // r02 r03 r04 r05

                        "fmla   v22.4s, v24.4s, v1.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v2.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v1.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v2.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v26.4s, v1.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v2.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v1.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v2.s[3]     \n"

                        "fmla   v22.4s, v16.4s, v2.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v3.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v2.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v3.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v18.4s, v2.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v3.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v2.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v3.s[3]     \n"

                        "fmla   v22.4s, v24.4s, v3.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v3.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v4.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v26.4s, v3.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v4.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v3.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v4.s[3]     \n"

                        "prfm   pldl1keep, [%2, #256]       \n"
                        "ld1    {v0.4s, v1.4s}, [%2], #32   \n" // r10 r11

                        "fmla   v22.4s, v16.4s, v4.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v5.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v4.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v5.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v18.4s, v4.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v5.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v4.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v5.s[3]     \n"

                        "fmla   v22.4s, v24.4s, v0.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v1.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v0.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v1.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v26.4s, v0.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v1.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v0.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v1.s[3]     \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v2.4s, v3.4s, v4.4s, v5.4s}, [%2] \n" // r12 r13 r14 r15

                        "fmla   v22.4s, v16.4s, v1.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v2.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v1.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v2.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v18.4s, v1.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v2.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v1.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v2.s[3]     \n"

                        "fmla   v22.4s, v24.4s, v2.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v3.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v2.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v3.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v26.4s, v2.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v3.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v2.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v3.s[3]     \n"

                        "fmla   v22.4s, v16.4s, v3.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v4.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v3.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v4.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v18.4s, v3.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v4.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v3.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v4.s[3]     \n"

                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld1    {v0.4s, v1.4s}, [%3], #32   \n" // r20 r21

                        "fmla   v22.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v5.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v4.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v5.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v26.4s, v4.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v5.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v4.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v5.s[3]     \n"

                        "fmla   v22.4s, v16.4s, v0.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v1.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v0.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v1.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v18.4s, v0.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v1.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v0.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v1.s[3]     \n"

                        "prfm   pldl1keep, [%3, #512]       \n"
                        "ld1    {v2.4s, v3.4s, v4.4s, v5.4s}, [%3] \n" // r22 r23 r24 r25

                        "fmla   v22.4s, v24.4s, v1.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v2.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v1.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v2.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v26.4s, v1.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v2.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v1.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v2.s[3]     \n"

                        "fmla   v22.4s, v16.4s, v2.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v3.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v2.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v3.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v18.4s, v2.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v3.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v2.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v3.s[3]     \n"

                        "fmla   v22.4s, v24.4s, v3.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v3.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v4.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v26.4s, v3.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v4.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v3.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v4.s[3]     \n"

                        "prfm   pldl1keep, [%4, #256]       \n"
                        "ld1    {v0.4s, v1.4s}, [%4], #32   \n" // r30 r31

                        "fmla   v22.4s, v16.4s, v4.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v5.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v4.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v5.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v18.4s, v4.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v5.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v4.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v5.s[3]     \n"

                        "fmla   v22.4s, v24.4s, v0.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v1.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v0.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v1.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v26.4s, v0.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v1.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v0.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v1.s[3]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v2.4s, v3.4s, v4.4s, v5.4s}, [%4] \n" // r32 r33 r34 r35

                        "fmla   v22.4s, v16.4s, v1.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v2.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v1.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v2.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v18.4s, v1.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v2.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v1.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v2.s[3]     \n"

                        "fmla   v22.4s, v24.4s, v2.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v3.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v2.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v3.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v26.4s, v2.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v3.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v2.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v3.s[3]     \n"

                        "fmla   v22.4s, v16.4s, v3.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v4.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v3.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v4.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v18.4s, v3.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v4.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v3.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v4.s[3]     \n"

                        "prfm   pldl1keep, [%5, #256]       \n"
                        "ld1    {v0.4s, v1.4s}, [%5], #32   \n" // r40 r41

                        "fmla   v22.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v5.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v4.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v5.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v26.4s, v4.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v5.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v4.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v5.s[3]     \n"

                        "fmla   v22.4s, v16.4s, v0.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v1.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v0.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v1.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v18.4s, v0.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v1.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v0.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v1.s[3]     \n"

                        "prfm   pldl1keep, [%5, #512]       \n"
                        "ld1    {v2.4s, v3.4s, v4.4s, v5.4s}, [%5] \n" // r42 r43 r44 r45

                        "fmla   v22.4s, v24.4s, v1.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v2.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v1.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v2.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v26.4s, v1.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v2.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v1.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v2.s[3]     \n"

                        "fmla   v22.4s, v16.4s, v2.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v3.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v2.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v3.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v18.4s, v2.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v3.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v2.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v3.s[3]     \n"

                        "fmla   v22.4s, v24.4s, v3.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v3.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v4.s[1]     \n"

                        //                         "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6] \n"

                        "fmla   v22.4s, v26.4s, v3.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v4.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v3.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v4.s[3]     \n"

                        "fmla   v22.4s, v16.4s, v4.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v5.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v4.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v5.s[1]     \n"
                        "fmla   v22.4s, v18.4s, v4.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v5.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v4.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v5.s[3]     \n"

                        "fadd   v20.4s, v20.4s, v22.4s      \n"
                        "fadd   v21.4s, v21.4s, v23.4s      \n"

                        "sub    %6, %6, #1536               \n" // kptr -= 24 * 16;

                        "st1    {v20.4s, v21.4s}, [%0], #32 \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2),      // %3
                        "=r"(r3),      // %4
                        "=r"(r4),      // %5
                        "=r"(kptr)     // %6
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "4"(r3),
                        "5"(r4),
                        "6"(kptr)
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27");
#else  // __aarch64__
                    asm volatile(
                        "pld        [%0, #256]          \n"
                        "vld1.f32   {d24-d27}, [%0 :128] \n" // sum0 sum1

                        "pld        [%1, #256]          \n"
                        "vld1.f32   {d0-d3}, [%1 :128]! \n" // r00 r01

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmul.f32   q14, q8, d0[0]      \n"
                        "vmul.f32   q15, q8, d2[0]      \n"
                        "vmla.f32   q12, q9, d0[1]      \n"
                        "vmla.f32   q13, q9, d2[1]      \n"
                        "vmla.f32   q14, q10, d1[0]     \n"
                        "vmla.f32   q15, q10, d3[0]     \n"
                        "vmla.f32   q12, q11, d1[1]     \n"
                        "vmla.f32   q13, q11, d3[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%1, #512]          \n"
                        "vldm       %1, {d4-d11}        \n" // r02 r03 r04 r05

                        "vmla.f32   q14, q8, d2[0]      \n"
                        "vmla.f32   q15, q8, d4[0]      \n"
                        "vmla.f32   q12, q9, d2[1]      \n"
                        "vmla.f32   q13, q9, d4[1]      \n"
                        "vmla.f32   q14, q10, d3[0]     \n"
                        "vmla.f32   q15, q10, d5[0]     \n"
                        "vmla.f32   q12, q11, d3[1]     \n"
                        "vmla.f32   q13, q11, d5[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q14, q8, d4[0]      \n"
                        "vmla.f32   q15, q8, d6[0]      \n"
                        "vmla.f32   q12, q9, d4[1]      \n"
                        "vmla.f32   q13, q9, d6[1]      \n"
                        "vmla.f32   q14, q10, d5[0]     \n"
                        "vmla.f32   q15, q10, d7[0]     \n"
                        "vmla.f32   q12, q11, d5[1]     \n"
                        "vmla.f32   q13, q11, d7[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q14, q8, d6[0]      \n"
                        "vmla.f32   q15, q8, d8[0]      \n"
                        "vmla.f32   q12, q9, d6[1]      \n"
                        "vmla.f32   q13, q9, d8[1]      \n"
                        "vmla.f32   q14, q10, d7[0]     \n"
                        "vmla.f32   q15, q10, d9[0]     \n"
                        "vmla.f32   q12, q11, d7[1]     \n"
                        "vmla.f32   q13, q11, d9[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%2, #256]          \n"
                        "vld1.f32   {d0-d3}, [%2 :128]! \n" // r10 r11

                        "vmla.f32   q14, q8, d8[0]      \n"
                        "vmla.f32   q15, q8, d10[0]     \n"
                        "vmla.f32   q12, q9, d8[1]      \n"
                        "vmla.f32   q13, q9, d10[1]     \n"
                        "vmla.f32   q14, q10, d9[0]     \n"
                        "vmla.f32   q15, q10, d11[0]    \n"
                        "vmla.f32   q12, q11, d9[1]     \n"
                        "vmla.f32   q13, q11, d11[1]    \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q14, q8, d0[0]      \n"
                        "vmla.f32   q15, q8, d2[0]      \n"
                        "vmla.f32   q12, q9, d0[1]      \n"
                        "vmla.f32   q13, q9, d2[1]      \n"
                        "vmla.f32   q14, q10, d1[0]     \n"
                        "vmla.f32   q15, q10, d3[0]     \n"
                        "vmla.f32   q12, q11, d1[1]     \n"
                        "vmla.f32   q13, q11, d3[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%2, #512]          \n"
                        "vldm       %2, {d4-d11}        \n" // r12 r13 r14 r15

                        "vmla.f32   q14, q8, d2[0]      \n"
                        "vmla.f32   q15, q8, d4[0]      \n"
                        "vmla.f32   q12, q9, d2[1]      \n"
                        "vmla.f32   q13, q9, d4[1]      \n"
                        "vmla.f32   q14, q10, d3[0]     \n"
                        "vmla.f32   q15, q10, d5[0]     \n"
                        "vmla.f32   q12, q11, d3[1]     \n"
                        "vmla.f32   q13, q11, d5[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q14, q8, d4[0]      \n"
                        "vmla.f32   q15, q8, d6[0]      \n"
                        "vmla.f32   q12, q9, d4[1]      \n"
                        "vmla.f32   q13, q9, d6[1]      \n"
                        "vmla.f32   q14, q10, d5[0]     \n"
                        "vmla.f32   q15, q10, d7[0]     \n"
                        "vmla.f32   q12, q11, d5[1]     \n"
                        "vmla.f32   q13, q11, d7[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q14, q8, d6[0]      \n"
                        "vmla.f32   q15, q8, d8[0]      \n"
                        "vmla.f32   q12, q9, d6[1]      \n"
                        "vmla.f32   q13, q9, d8[1]      \n"
                        "vmla.f32   q14, q10, d7[0]     \n"
                        "vmla.f32   q15, q10, d9[0]     \n"
                        "vmla.f32   q12, q11, d7[1]     \n"
                        "vmla.f32   q13, q11, d9[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%3, #256]          \n"
                        "vld1.f32   {d0-d3}, [%3 :128]! \n" // r20 r21

                        "vmla.f32   q14, q8, d8[0]      \n"
                        "vmla.f32   q15, q8, d10[0]     \n"
                        "vmla.f32   q12, q9, d8[1]      \n"
                        "vmla.f32   q13, q9, d10[1]     \n"
                        "vmla.f32   q14, q10, d9[0]     \n"
                        "vmla.f32   q15, q10, d11[0]    \n"
                        "vmla.f32   q12, q11, d9[1]     \n"
                        "vmla.f32   q13, q11, d11[1]    \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q14, q8, d0[0]      \n"
                        "vmla.f32   q15, q8, d2[0]      \n"
                        "vmla.f32   q12, q9, d0[1]      \n"
                        "vmla.f32   q13, q9, d2[1]      \n"
                        "vmla.f32   q14, q10, d1[0]     \n"
                        "vmla.f32   q15, q10, d3[0]     \n"
                        "vmla.f32   q12, q11, d1[1]     \n"
                        "vmla.f32   q13, q11, d3[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%3, #512]          \n"
                        "vldm       %3, {d4-d11}        \n" // r22 r23 r24 r25

                        "vmla.f32   q14, q8, d2[0]      \n"
                        "vmla.f32   q15, q8, d4[0]      \n"
                        "vmla.f32   q12, q9, d2[1]      \n"
                        "vmla.f32   q13, q9, d4[1]      \n"
                        "vmla.f32   q14, q10, d3[0]     \n"
                        "vmla.f32   q15, q10, d5[0]     \n"
                        "vmla.f32   q12, q11, d3[1]     \n"
                        "vmla.f32   q13, q11, d5[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q14, q8, d4[0]      \n"
                        "vmla.f32   q15, q8, d6[0]      \n"
                        "vmla.f32   q12, q9, d4[1]      \n"
                        "vmla.f32   q13, q9, d6[1]      \n"
                        "vmla.f32   q14, q10, d5[0]     \n"
                        "vmla.f32   q15, q10, d7[0]     \n"
                        "vmla.f32   q12, q11, d5[1]     \n"
                        "vmla.f32   q13, q11, d7[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q14, q8, d6[0]      \n"
                        "vmla.f32   q15, q8, d8[0]      \n"
                        "vmla.f32   q12, q9, d6[1]      \n"
                        "vmla.f32   q13, q9, d8[1]      \n"
                        "vmla.f32   q14, q10, d7[0]     \n"
                        "vmla.f32   q15, q10, d9[0]     \n"
                        "vmla.f32   q12, q11, d7[1]     \n"
                        "vmla.f32   q13, q11, d9[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%4, #256]          \n"
                        "vld1.f32   {d0-d3}, [%4 :128]! \n" // r30 r31

                        "vmla.f32   q14, q8, d8[0]      \n"
                        "vmla.f32   q15, q8, d10[0]     \n"
                        "vmla.f32   q12, q9, d8[1]      \n"
                        "vmla.f32   q13, q9, d10[1]     \n"
                        "vmla.f32   q14, q10, d9[0]     \n"
                        "vmla.f32   q15, q10, d11[0]    \n"
                        "vmla.f32   q12, q11, d9[1]     \n"
                        "vmla.f32   q13, q11, d11[1]    \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q14, q8, d0[0]      \n"
                        "vmla.f32   q15, q8, d2[0]      \n"
                        "vmla.f32   q12, q9, d0[1]      \n"
                        "vmla.f32   q13, q9, d2[1]      \n"
                        "vmla.f32   q14, q10, d1[0]     \n"
                        "vmla.f32   q15, q10, d3[0]     \n"
                        "vmla.f32   q12, q11, d1[1]     \n"
                        "vmla.f32   q13, q11, d3[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%4, #512]          \n"
                        "vldm       %4, {d4-d11}        \n" // r32 r33 r34 r35

                        "vmla.f32   q14, q8, d2[0]      \n"
                        "vmla.f32   q15, q8, d4[0]      \n"
                        "vmla.f32   q12, q9, d2[1]      \n"
                        "vmla.f32   q13, q9, d4[1]      \n"
                        "vmla.f32   q14, q10, d3[0]     \n"
                        "vmla.f32   q15, q10, d5[0]     \n"
                        "vmla.f32   q12, q11, d3[1]     \n"
                        "vmla.f32   q13, q11, d5[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q14, q8, d4[0]      \n"
                        "vmla.f32   q15, q8, d6[0]      \n"
                        "vmla.f32   q12, q9, d4[1]      \n"
                        "vmla.f32   q13, q9, d6[1]      \n"
                        "vmla.f32   q14, q10, d5[0]     \n"
                        "vmla.f32   q15, q10, d7[0]     \n"
                        "vmla.f32   q12, q11, d5[1]     \n"
                        "vmla.f32   q13, q11, d7[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q14, q8, d6[0]      \n"
                        "vmla.f32   q15, q8, d8[0]      \n"
                        "vmla.f32   q12, q9, d6[1]      \n"
                        "vmla.f32   q13, q9, d8[1]      \n"
                        "vmla.f32   q14, q10, d7[0]     \n"
                        "vmla.f32   q15, q10, d9[0]     \n"
                        "vmla.f32   q12, q11, d7[1]     \n"
                        "vmla.f32   q13, q11, d9[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%5, #256]          \n"
                        "vld1.f32   {d0-d3}, [%5 :128]! \n" // r40 r41

                        "vmla.f32   q14, q8, d8[0]      \n"
                        "vmla.f32   q15, q8, d10[0]     \n"
                        "vmla.f32   q12, q9, d8[1]      \n"
                        "vmla.f32   q13, q9, d10[1]     \n"
                        "vmla.f32   q14, q10, d9[0]     \n"
                        "vmla.f32   q15, q10, d11[0]    \n"
                        "vmla.f32   q12, q11, d9[1]     \n"
                        "vmla.f32   q13, q11, d11[1]    \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q14, q8, d0[0]      \n"
                        "vmla.f32   q15, q8, d2[0]      \n"
                        "vmla.f32   q12, q9, d0[1]      \n"
                        "vmla.f32   q13, q9, d2[1]      \n"
                        "vmla.f32   q14, q10, d1[0]     \n"
                        "vmla.f32   q15, q10, d3[0]     \n"
                        "vmla.f32   q12, q11, d1[1]     \n"
                        "vmla.f32   q13, q11, d3[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%5, #512]          \n"
                        "vldm       %5, {d4-d11}        \n" // r42 r43 r44 r45

                        "vmla.f32   q14, q8, d2[0]      \n"
                        "vmla.f32   q15, q8, d4[0]      \n"
                        "vmla.f32   q12, q9, d2[1]      \n"
                        "vmla.f32   q13, q9, d4[1]      \n"
                        "vmla.f32   q14, q10, d3[0]     \n"
                        "vmla.f32   q15, q10, d5[0]     \n"
                        "vmla.f32   q12, q11, d3[1]     \n"
                        "vmla.f32   q13, q11, d5[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q14, q8, d4[0]      \n"
                        "vmla.f32   q15, q8, d6[0]      \n"
                        "vmla.f32   q12, q9, d4[1]      \n"
                        "vmla.f32   q13, q9, d6[1]      \n"
                        "vmla.f32   q14, q10, d5[0]     \n"
                        "vmla.f32   q15, q10, d7[0]     \n"
                        "vmla.f32   q12, q11, d5[1]     \n"
                        "vmla.f32   q13, q11, d7[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q14, q8, d6[0]      \n"
                        "vmla.f32   q15, q8, d8[0]      \n"
                        "vmla.f32   q12, q9, d6[1]      \n"
                        "vmla.f32   q13, q9, d8[1]      \n"
                        "vmla.f32   q14, q10, d7[0]     \n"
                        "vmla.f32   q15, q10, d9[0]     \n"
                        "vmla.f32   q12, q11, d7[1]     \n"
                        "vmla.f32   q13, q11, d9[1]     \n"

                        //                         "pld        [%6, #512]          \n"
                        "vldm       %6, {d16-d23}       \n"

                        "vmla.f32   q14, q8, d8[0]      \n"
                        "vmla.f32   q15, q8, d10[0]     \n"
                        "vmla.f32   q12, q9, d8[1]      \n"
                        "vmla.f32   q13, q9, d10[1]     \n"
                        "vmla.f32   q14, q10, d9[0]     \n"
                        "vmla.f32   q15, q10, d11[0]    \n"
                        "vmla.f32   q12, q11, d9[1]     \n"
                        "vmla.f32   q13, q11, d11[1]    \n"

                        "vadd.f32   q12, q12, q14       \n"
                        "vadd.f32   q13, q13, q15       \n"

                        "sub        %6, %6, #1536       \n" // kptr -= 24 * 16;

                        "vst1.f32   {d24-d27}, [%0 :128]! \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2),      // %3
                        "=r"(r3),      // %4
                        "=r"(r4),      // %5
                        "=r"(kptr)     // %6
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "4"(r3),
                        "5"(r4),
                        "6"(kptr)
                        : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif // __aarch64__
                }
                for (; j < outw; j++)
                {
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%0, #128]       \n"
                        "ld1    {v20.4s}, [%0]              \n" // sum0

                        "prfm   pldl1keep, [%1, #128]       \n"
                        "ld1    {v0.4s}, [%1], #16          \n" // r00

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v1.4s, v2.4s, v3.4s, v4.4s}, [%1] \n" // r01 r02 r03 r04

                        "fmul   v21.4s, v16.4s, v0.s[0]     \n"
                        "fmul   v22.4s, v17.4s, v0.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmul   v23.4s, v18.4s, v0.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v0.s[3]     \n"

                        "fmla   v21.4s, v24.4s, v1.s[0]     \n"
                        "fmla   v22.4s, v25.4s, v1.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v23.4s, v26.4s, v1.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v1.s[3]     \n"

                        "fmla   v21.4s, v16.4s, v2.s[0]     \n"
                        "fmla   v22.4s, v17.4s, v2.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v23.4s, v18.4s, v2.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v2.s[3]     \n"

                        "fmla   v21.4s, v24.4s, v3.s[0]     \n"
                        "fmla   v22.4s, v25.4s, v3.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v23.4s, v26.4s, v3.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v3.s[3]     \n"

                        "prfm   pldl1keep, [%2, #128]       \n"
                        "ld1    {v0.4s}, [%2], #16          \n" // r10

                        "fmla   v21.4s, v16.4s, v4.s[0]     \n"
                        "fmla   v22.4s, v17.4s, v4.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v23.4s, v18.4s, v4.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v4.s[3]     \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v1.4s, v2.4s, v3.4s, v4.4s}, [%2] \n" // r11 r12 r13 r14

                        "fmla   v21.4s, v24.4s, v0.s[0]     \n"
                        "fmla   v22.4s, v25.4s, v0.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v23.4s, v26.4s, v0.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v0.s[3]     \n"

                        "fmla   v21.4s, v16.4s, v1.s[0]     \n"
                        "fmla   v22.4s, v17.4s, v1.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v23.4s, v18.4s, v1.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v1.s[3]     \n"

                        "fmla   v21.4s, v24.4s, v2.s[0]     \n"
                        "fmla   v22.4s, v25.4s, v2.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v23.4s, v26.4s, v2.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v2.s[3]     \n"

                        "fmla   v21.4s, v16.4s, v3.s[0]     \n"
                        "fmla   v22.4s, v17.4s, v3.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v23.4s, v18.4s, v3.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v3.s[3]     \n"

                        "prfm   pldl1keep, [%3, #128]       \n"
                        "ld1    {v0.4s}, [%3], #16          \n" // r20

                        "fmla   v21.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v22.4s, v25.4s, v4.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v23.4s, v26.4s, v4.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v4.s[3]     \n"

                        "prfm   pldl1keep, [%3, #512]       \n"
                        "ld1    {v1.4s, v2.4s, v3.4s, v4.4s}, [%3] \n" // r21 r22 r23 r24

                        "fmla   v21.4s, v16.4s, v0.s[0]     \n"
                        "fmla   v22.4s, v17.4s, v0.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v23.4s, v18.4s, v0.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v0.s[3]     \n"

                        "fmla   v21.4s, v24.4s, v1.s[0]     \n"
                        "fmla   v22.4s, v25.4s, v1.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v23.4s, v26.4s, v1.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v1.s[3]     \n"

                        "fmla   v21.4s, v16.4s, v2.s[0]     \n"
                        "fmla   v22.4s, v17.4s, v2.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v23.4s, v18.4s, v2.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v2.s[3]     \n"

                        "fmla   v21.4s, v24.4s, v3.s[0]     \n"
                        "fmla   v22.4s, v25.4s, v3.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v23.4s, v26.4s, v3.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v3.s[3]     \n"

                        "prfm   pldl1keep, [%4, #128]       \n"
                        "ld1    {v0.4s}, [%4], #16          \n" // r30

                        "fmla   v21.4s, v16.4s, v4.s[0]     \n"
                        "fmla   v22.4s, v17.4s, v4.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v23.4s, v18.4s, v4.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v4.s[3]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v1.4s, v2.4s, v3.4s, v4.4s}, [%4] \n" // r31 r32 r33 r34

                        "fmla   v21.4s, v24.4s, v0.s[0]     \n"
                        "fmla   v22.4s, v25.4s, v0.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v23.4s, v26.4s, v0.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v0.s[3]     \n"

                        "fmla   v21.4s, v16.4s, v1.s[0]     \n"
                        "fmla   v22.4s, v17.4s, v1.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v23.4s, v18.4s, v1.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v1.s[3]     \n"

                        "fmla   v21.4s, v24.4s, v2.s[0]     \n"
                        "fmla   v22.4s, v25.4s, v2.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v23.4s, v26.4s, v2.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v2.s[3]     \n"

                        "fmla   v21.4s, v16.4s, v3.s[0]     \n"
                        "fmla   v22.4s, v17.4s, v3.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v23.4s, v18.4s, v3.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v3.s[3]     \n"

                        "prfm   pldl1keep, [%5, #128]       \n"
                        "ld1    {v0.4s}, [%5], #16          \n" // r40

                        "fmla   v21.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v22.4s, v25.4s, v4.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v23.4s, v26.4s, v4.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v4.s[3]     \n"

                        "prfm   pldl1keep, [%5, #512]       \n"
                        "ld1    {v1.4s, v2.4s, v3.4s, v4.4s}, [%5] \n" // r41 r42 r43 r44

                        "fmla   v21.4s, v16.4s, v0.s[0]     \n"
                        "fmla   v22.4s, v17.4s, v0.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v23.4s, v18.4s, v0.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v0.s[3]     \n"

                        "fmla   v21.4s, v24.4s, v1.s[0]     \n"
                        "fmla   v22.4s, v25.4s, v1.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v23.4s, v26.4s, v1.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v1.s[3]     \n"

                        "fmla   v21.4s, v16.4s, v2.s[0]     \n"
                        "fmla   v22.4s, v17.4s, v2.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v23.4s, v18.4s, v2.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v2.s[3]     \n"

                        "fmla   v21.4s, v24.4s, v3.s[0]     \n"
                        "fmla   v22.4s, v25.4s, v3.s[1]     \n"

                        //                         "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6] \n"

                        "fmla   v23.4s, v26.4s, v3.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v3.s[3]     \n"

                        "fmla   v21.4s, v16.4s, v4.s[0]     \n"
                        "fmla   v22.4s, v17.4s, v4.s[1]     \n"
                        "fmla   v23.4s, v18.4s, v4.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v4.s[3]     \n"

                        "fadd   v22.4s, v21.4s, v22.4s      \n"
                        "fadd   v23.4s, v22.4s, v23.4s      \n"
                        "fadd   v20.4s, v20.4s, v23.4s      \n"

                        "sub    %6, %6, #1536               \n" // kptr -= 24 * 16;

                        "st1    {v20.4s}, [%0], #16         \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2),      // %3
                        "=r"(r3),      // %4
                        "=r"(r4),      // %5
                        "=r"(kptr)     // %6
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "4"(r3),
                        "5"(r4),
                        "6"(kptr)
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27");
#else  // __aarch64__
                    asm volatile(
                        "pld        [%0, #128]          \n"
                        "vld1.f32   {d24-d25}, [%0 :128] \n" // sum0

                        "pld        [%1, #128]          \n"
                        "vld1.f32   {d0-d1}, [%1 :128]! \n" // r00

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmul.f32   q13, q8, d0[0]      \n"
                        "vmul.f32   q14, q9, d0[1]      \n"
                        "vmul.f32   q15, q10, d1[0]     \n"
                        "vmla.f32   q12, q11, d1[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%1, #512]          \n"
                        "vldm       %1, {d2-d9}         \n" // r01 r02 r03 r04

                        "vmla.f32   q13, q8, d2[0]      \n"
                        "vmla.f32   q14, q9, d2[1]      \n"
                        "vmla.f32   q15, q10, d3[0]     \n"
                        "vmla.f32   q12, q11, d3[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q13, q8, d4[0]      \n"
                        "vmla.f32   q14, q9, d4[1]      \n"
                        "vmla.f32   q15, q10, d5[0]     \n"
                        "vmla.f32   q12, q11, d5[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q13, q8, d6[0]      \n"
                        "vmla.f32   q14, q9, d6[1]      \n"
                        "vmla.f32   q15, q10, d7[0]     \n"
                        "vmla.f32   q12, q11, d7[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%2, #128]          \n"
                        "vld1.f32   {d0-d1}, [%2 :128]! \n" // r10

                        "vmla.f32   q13, q8, d8[0]      \n"
                        "vmla.f32   q14, q9, d8[1]      \n"
                        "vmla.f32   q15, q10, d9[0]     \n"
                        "vmla.f32   q12, q11, d9[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q13, q8, d0[0]      \n"
                        "vmla.f32   q14, q9, d0[1]      \n"
                        "vmla.f32   q15, q10, d1[0]     \n"
                        "vmla.f32   q12, q11, d1[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%2, #512]          \n"
                        "vldm       %2, {d2-d9}         \n" // r11 r12 r13 r14

                        "vmla.f32   q13, q8, d2[0]      \n"
                        "vmla.f32   q14, q9, d2[1]      \n"
                        "vmla.f32   q15, q10, d3[0]     \n"
                        "vmla.f32   q12, q11, d3[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q13, q8, d4[0]      \n"
                        "vmla.f32   q14, q9, d4[1]      \n"
                        "vmla.f32   q15, q10, d5[0]     \n"
                        "vmla.f32   q12, q11, d5[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q13, q8, d6[0]      \n"
                        "vmla.f32   q14, q9, d6[1]      \n"
                        "vmla.f32   q15, q10, d7[0]     \n"
                        "vmla.f32   q12, q11, d7[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%3, #128]          \n"
                        "vld1.f32   {d0-d1}, [%3 :128]! \n" // r20

                        "vmla.f32   q13, q8, d8[0]      \n"
                        "vmla.f32   q14, q9, d8[1]      \n"
                        "vmla.f32   q15, q10, d9[0]     \n"
                        "vmla.f32   q12, q11, d9[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q13, q8, d0[0]      \n"
                        "vmla.f32   q14, q9, d0[1]      \n"
                        "vmla.f32   q15, q10, d1[0]     \n"
                        "vmla.f32   q12, q11, d1[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%3, #512]          \n"
                        "vldm       %3, {d2-d9}         \n" // r21 r22 r23 r24

                        "vmla.f32   q13, q8, d2[0]      \n"
                        "vmla.f32   q14, q9, d2[1]      \n"
                        "vmla.f32   q15, q10, d3[0]     \n"
                        "vmla.f32   q12, q11, d3[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q13, q8, d4[0]      \n"
                        "vmla.f32   q14, q9, d4[1]      \n"
                        "vmla.f32   q15, q10, d5[0]     \n"
                        "vmla.f32   q12, q11, d5[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q13, q8, d6[0]      \n"
                        "vmla.f32   q14, q9, d6[1]      \n"
                        "vmla.f32   q15, q10, d7[0]     \n"
                        "vmla.f32   q12, q11, d7[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%4, #128]          \n"
                        "vld1.f32   {d0-d1}, [%4 :128]! \n" // r30

                        "vmla.f32   q13, q8, d8[0]      \n"
                        "vmla.f32   q14, q9, d8[1]      \n"
                        "vmla.f32   q15, q10, d9[0]     \n"
                        "vmla.f32   q12, q11, d9[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q13, q8, d0[0]      \n"
                        "vmla.f32   q14, q9, d0[1]      \n"
                        "vmla.f32   q15, q10, d1[0]     \n"
                        "vmla.f32   q12, q11, d1[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%4, #512]          \n"
                        "vldm       %4, {d2-d9}         \n" // r31 r32 r33 r34

                        "vmla.f32   q13, q8, d2[0]      \n"
                        "vmla.f32   q14, q9, d2[1]      \n"
                        "vmla.f32   q15, q10, d3[0]     \n"
                        "vmla.f32   q12, q11, d3[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q13, q8, d4[0]      \n"
                        "vmla.f32   q14, q9, d4[1]      \n"
                        "vmla.f32   q15, q10, d5[0]     \n"
                        "vmla.f32   q12, q11, d5[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q13, q8, d6[0]      \n"
                        "vmla.f32   q14, q9, d6[1]      \n"
                        "vmla.f32   q15, q10, d7[0]     \n"
                        "vmla.f32   q12, q11, d7[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%5, #128]          \n"
                        "vld1.f32   {d0-d1}, [%5 :128]! \n" // r40

                        "vmla.f32   q13, q8, d8[0]      \n"
                        "vmla.f32   q14, q9, d8[1]      \n"
                        "vmla.f32   q15, q10, d9[0]     \n"
                        "vmla.f32   q12, q11, d9[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q13, q8, d0[0]      \n"
                        "vmla.f32   q14, q9, d0[1]      \n"
                        "vmla.f32   q15, q10, d1[0]     \n"
                        "vmla.f32   q12, q11, d1[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%5, #512]          \n"
                        "vldm       %5, {d2-d9}         \n" // r41 r42 r43 r44

                        "vmla.f32   q13, q8, d2[0]      \n"
                        "vmla.f32   q14, q9, d2[1]      \n"
                        "vmla.f32   q15, q10, d3[0]     \n"
                        "vmla.f32   q12, q11, d3[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q13, q8, d4[0]      \n"
                        "vmla.f32   q14, q9, d4[1]      \n"
                        "vmla.f32   q15, q10, d5[0]     \n"
                        "vmla.f32   q12, q11, d5[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q13, q8, d6[0]      \n"
                        "vmla.f32   q14, q9, d6[1]      \n"
                        "vmla.f32   q15, q10, d7[0]     \n"
                        "vmla.f32   q12, q11, d7[1]     \n"

                        //                         "pld        [%6, #512]          \n"
                        "vldm       %6, {d16-d23}       \n"

                        "vmla.f32   q13, q8, d8[0]      \n"
                        "vmla.f32   q14, q9, d8[1]      \n"
                        "vmla.f32   q15, q10, d9[0]     \n"
                        "vmla.f32   q12, q11, d9[1]     \n"

                        "vadd.f32   q13, q13, q14       \n"
                        "vadd.f32   q12, q12, q15       \n"
                        "vadd.f32   q12, q12, q13       \n"

                        "sub        %6, %6, #1536       \n" // kptr -= 24 * 16;

                        "vst1.f32   {d24-d25}, [%0 :128]! \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2),      // %3
                        "=r"(r3),      // %4
                        "=r"(r4),      // %5
                        "=r"(kptr)     // %6
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "4"(r3),
                        "5"(r4),
                        "6"(kptr)
                        : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif // __aarch64__
                }

                r0 += 4 * 4;
                r1 += 4 * 4;
                r2 += 4 * 4;
                r3 += 4 * 4;
                r4 += 4 * 4;
            }
        }
    }
}

static void conv5x5s2_pack4_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;
    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = (w - 2 * outw + w) * 4;

    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        float32x4_t _bias0 = bias ? vld1q_f32((const float*)bias + p * 4) : vdupq_n_f32(0.f);
        out0.fill(_bias0);

        for (int q = 0; q < inch; q++)
        {
            float* outptr0 = out0.row(0);

            const Mat img0 = bottom_blob.channel(q);

            const float* r0 = img0.row(0);
            const float* r1 = img0.row(1);
            const float* r2 = img0.row(2);
            const float* r3 = img0.row(3);
            const float* r4 = img0.row(4);

            const float* kptr = (const float*)kernel.channel(p).row(q);

            int i = 0;
            for (; i < outh; i++)
            {
                int j = 0;
                for (; j + 3 < outw; j += 4)
                {
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%0, #512]       \n"
                        "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0] \n" // sum0 sum1 sum2 sum3

                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%1], #64 \n" // r00 r01 r02 r03

                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%1], #64 \n" // r04 r05 r06 r07

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v16.4s, v0.s[0]     \n"
                        "fmla   v21.4s, v16.4s, v2.s[0]     \n"
                        "fmla   v22.4s, v16.4s, v4.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v6.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v0.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v2.s[1]     \n"
                        "fmla   v22.4s, v17.4s, v4.s[1]     \n"
                        "fmla   v23.4s, v17.4s, v6.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v18.4s, v0.s[2]     \n"
                        "fmla   v21.4s, v18.4s, v2.s[2]     \n"
                        "fmla   v22.4s, v18.4s, v4.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v6.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v0.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v2.s[3]     \n"
                        "fmla   v22.4s, v19.4s, v4.s[3]     \n"
                        "fmla   v23.4s, v19.4s, v6.s[3]     \n"

                        "prfm   pldl1keep, [%1, #384]       \n"
                        "ld1    {v28.4s, v29.4s, v30.4s}, [%1] \n" // r08 r09 r010

                        "fmla   v20.4s, v24.4s, v1.s[0]     \n"
                        "fmla   v21.4s, v24.4s, v3.s[0]     \n"
                        "fmla   v22.4s, v24.4s, v5.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v7.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v1.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v3.s[1]     \n"
                        "fmla   v22.4s, v25.4s, v5.s[1]     \n"
                        "fmla   v23.4s, v25.4s, v7.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v26.4s, v1.s[2]     \n"
                        "fmla   v21.4s, v26.4s, v3.s[2]     \n"
                        "fmla   v22.4s, v26.4s, v5.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v7.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v1.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v3.s[3]     \n"
                        "fmla   v22.4s, v27.4s, v5.s[3]     \n"
                        "fmla   v23.4s, v27.4s, v7.s[3]     \n"

                        "fmla   v20.4s, v16.4s, v2.s[0]     \n"
                        "fmla   v21.4s, v16.4s, v4.s[0]     \n"
                        "fmla   v22.4s, v16.4s, v6.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v28.s[0]    \n"
                        "fmla   v20.4s, v17.4s, v2.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v4.s[1]     \n"
                        "fmla   v22.4s, v17.4s, v6.s[1]     \n"
                        "fmla   v23.4s, v17.4s, v28.s[1]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v18.4s, v2.s[2]     \n"
                        "fmla   v21.4s, v18.4s, v4.s[2]     \n"
                        "fmla   v22.4s, v18.4s, v6.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v28.s[2]    \n"
                        "fmla   v20.4s, v19.4s, v2.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v4.s[3]     \n"
                        "fmla   v22.4s, v19.4s, v6.s[3]     \n"
                        "fmla   v23.4s, v19.4s, v28.s[3]    \n"

                        "fmla   v20.4s, v24.4s, v3.s[0]     \n"
                        "fmla   v21.4s, v24.4s, v5.s[0]     \n"
                        "fmla   v22.4s, v24.4s, v7.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v29.s[0]    \n"
                        "fmla   v20.4s, v25.4s, v3.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v5.s[1]     \n"
                        "fmla   v22.4s, v25.4s, v7.s[1]     \n"
                        "fmla   v23.4s, v25.4s, v29.s[1]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v26.4s, v3.s[2]     \n"
                        "fmla   v21.4s, v26.4s, v5.s[2]     \n"
                        "fmla   v22.4s, v26.4s, v7.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v29.s[2]    \n"
                        "fmla   v20.4s, v27.4s, v3.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v5.s[3]     \n"
                        "fmla   v22.4s, v27.4s, v7.s[3]     \n"
                        "fmla   v23.4s, v27.4s, v29.s[3]    \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%2], #64 \n" // r10 r11 r12 r13

                        "fmla   v20.4s, v16.4s, v4.s[0]     \n"
                        "fmla   v21.4s, v16.4s, v6.s[0]     \n"
                        "fmla   v22.4s, v16.4s, v28.s[0]    \n"
                        "fmla   v23.4s, v16.4s, v30.s[0]    \n"
                        "fmla   v20.4s, v17.4s, v4.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v6.s[1]     \n"
                        "fmla   v22.4s, v17.4s, v28.s[1]    \n"
                        "fmla   v23.4s, v17.4s, v30.s[1]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v18.4s, v4.s[2]     \n"
                        "fmla   v21.4s, v18.4s, v6.s[2]     \n"
                        "fmla   v22.4s, v18.4s, v28.s[2]    \n"
                        "fmla   v23.4s, v18.4s, v30.s[2]    \n"
                        "fmla   v20.4s, v19.4s, v4.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v6.s[3]     \n"
                        "fmla   v22.4s, v19.4s, v28.s[3]    \n"
                        "fmla   v23.4s, v19.4s, v30.s[3]    \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%2], #64 \n" // r14 r15 r16 r17

                        "fmla   v20.4s, v24.4s, v0.s[0]     \n"
                        "fmla   v21.4s, v24.4s, v2.s[0]     \n"
                        "fmla   v22.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v6.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v0.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v2.s[1]     \n"
                        "fmla   v22.4s, v25.4s, v4.s[1]     \n"
                        "fmla   v23.4s, v25.4s, v6.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v26.4s, v0.s[2]     \n"
                        "fmla   v21.4s, v26.4s, v2.s[2]     \n"
                        "fmla   v22.4s, v26.4s, v4.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v6.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v0.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v2.s[3]     \n"
                        "fmla   v22.4s, v27.4s, v4.s[3]     \n"
                        "fmla   v23.4s, v27.4s, v6.s[3]     \n"

                        "prfm   pldl1keep, [%2, #384]       \n"
                        "ld1    {v28.4s, v29.4s, v30.4s}, [%2] \n" // r18 r19 r110

                        "fmla   v20.4s, v16.4s, v1.s[0]     \n"
                        "fmla   v21.4s, v16.4s, v3.s[0]     \n"
                        "fmla   v22.4s, v16.4s, v5.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v7.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v1.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v3.s[1]     \n"
                        "fmla   v22.4s, v17.4s, v5.s[1]     \n"
                        "fmla   v23.4s, v17.4s, v7.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v18.4s, v1.s[2]     \n"
                        "fmla   v21.4s, v18.4s, v3.s[2]     \n"
                        "fmla   v22.4s, v18.4s, v5.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v7.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v1.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v3.s[3]     \n"
                        "fmla   v22.4s, v19.4s, v5.s[3]     \n"
                        "fmla   v23.4s, v19.4s, v7.s[3]     \n"

                        "fmla   v20.4s, v24.4s, v2.s[0]     \n"
                        "fmla   v21.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v22.4s, v24.4s, v6.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v28.s[0]    \n"
                        "fmla   v20.4s, v25.4s, v2.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v4.s[1]     \n"
                        "fmla   v22.4s, v25.4s, v6.s[1]     \n"
                        "fmla   v23.4s, v25.4s, v28.s[1]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v26.4s, v2.s[2]     \n"
                        "fmla   v21.4s, v26.4s, v4.s[2]     \n"
                        "fmla   v22.4s, v26.4s, v6.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v28.s[2]    \n"
                        "fmla   v20.4s, v27.4s, v2.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v4.s[3]     \n"
                        "fmla   v22.4s, v27.4s, v6.s[3]     \n"
                        "fmla   v23.4s, v27.4s, v28.s[3]    \n"

                        "fmla   v20.4s, v16.4s, v3.s[0]     \n"
                        "fmla   v21.4s, v16.4s, v5.s[0]     \n"
                        "fmla   v22.4s, v16.4s, v7.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v29.s[0]    \n"
                        "fmla   v20.4s, v17.4s, v3.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v5.s[1]     \n"
                        "fmla   v22.4s, v17.4s, v7.s[1]     \n"
                        "fmla   v23.4s, v17.4s, v29.s[1]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v18.4s, v3.s[2]     \n"
                        "fmla   v21.4s, v18.4s, v5.s[2]     \n"
                        "fmla   v22.4s, v18.4s, v7.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v29.s[2]    \n"
                        "fmla   v20.4s, v19.4s, v3.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v5.s[3]     \n"
                        "fmla   v22.4s, v19.4s, v7.s[3]     \n"
                        "fmla   v23.4s, v19.4s, v29.s[3]    \n"

                        "prfm   pldl1keep, [%3, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%3], #64 \n" // r20 r21 r22 r23

                        "fmla   v20.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v21.4s, v24.4s, v6.s[0]     \n"
                        "fmla   v22.4s, v24.4s, v28.s[0]    \n"
                        "fmla   v23.4s, v24.4s, v30.s[0]    \n"
                        "fmla   v20.4s, v25.4s, v4.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v6.s[1]     \n"
                        "fmla   v22.4s, v25.4s, v28.s[1]    \n"
                        "fmla   v23.4s, v25.4s, v30.s[1]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v26.4s, v4.s[2]     \n"
                        "fmla   v21.4s, v26.4s, v6.s[2]     \n"
                        "fmla   v22.4s, v26.4s, v28.s[2]    \n"
                        "fmla   v23.4s, v26.4s, v30.s[2]    \n"
                        "fmla   v20.4s, v27.4s, v4.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v6.s[3]     \n"
                        "fmla   v22.4s, v27.4s, v28.s[3]    \n"
                        "fmla   v23.4s, v27.4s, v30.s[3]    \n"

                        "prfm   pldl1keep, [%3, #512]       \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%3], #64 \n" // r24 r25 r26 r27

                        "fmla   v20.4s, v16.4s, v0.s[0]     \n"
                        "fmla   v21.4s, v16.4s, v2.s[0]     \n"
                        "fmla   v22.4s, v16.4s, v4.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v6.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v0.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v2.s[1]     \n"
                        "fmla   v22.4s, v17.4s, v4.s[1]     \n"
                        "fmla   v23.4s, v17.4s, v6.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v18.4s, v0.s[2]     \n"
                        "fmla   v21.4s, v18.4s, v2.s[2]     \n"
                        "fmla   v22.4s, v18.4s, v4.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v6.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v0.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v2.s[3]     \n"
                        "fmla   v22.4s, v19.4s, v4.s[3]     \n"
                        "fmla   v23.4s, v19.4s, v6.s[3]     \n"

                        "prfm   pldl1keep, [%3, #384]       \n"
                        "ld1    {v28.4s, v29.4s, v30.4s}, [%3] \n" // r28 r29 r210

                        "fmla   v20.4s, v24.4s, v1.s[0]     \n"
                        "fmla   v21.4s, v24.4s, v3.s[0]     \n"
                        "fmla   v22.4s, v24.4s, v5.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v7.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v1.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v3.s[1]     \n"
                        "fmla   v22.4s, v25.4s, v5.s[1]     \n"
                        "fmla   v23.4s, v25.4s, v7.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v26.4s, v1.s[2]     \n"
                        "fmla   v21.4s, v26.4s, v3.s[2]     \n"
                        "fmla   v22.4s, v26.4s, v5.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v7.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v1.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v3.s[3]     \n"
                        "fmla   v22.4s, v27.4s, v5.s[3]     \n"
                        "fmla   v23.4s, v27.4s, v7.s[3]     \n"

                        "fmla   v20.4s, v16.4s, v2.s[0]     \n"
                        "fmla   v21.4s, v16.4s, v4.s[0]     \n"
                        "fmla   v22.4s, v16.4s, v6.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v28.s[0]    \n"
                        "fmla   v20.4s, v17.4s, v2.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v4.s[1]     \n"
                        "fmla   v22.4s, v17.4s, v6.s[1]     \n"
                        "fmla   v23.4s, v17.4s, v28.s[1]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v18.4s, v2.s[2]     \n"
                        "fmla   v21.4s, v18.4s, v4.s[2]     \n"
                        "fmla   v22.4s, v18.4s, v6.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v28.s[2]    \n"
                        "fmla   v20.4s, v19.4s, v2.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v4.s[3]     \n"
                        "fmla   v22.4s, v19.4s, v6.s[3]     \n"
                        "fmla   v23.4s, v19.4s, v28.s[3]    \n"

                        "fmla   v20.4s, v24.4s, v3.s[0]     \n"
                        "fmla   v21.4s, v24.4s, v5.s[0]     \n"
                        "fmla   v22.4s, v24.4s, v7.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v29.s[0]    \n"
                        "fmla   v20.4s, v25.4s, v3.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v5.s[1]     \n"
                        "fmla   v22.4s, v25.4s, v7.s[1]     \n"
                        "fmla   v23.4s, v25.4s, v29.s[1]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v26.4s, v3.s[2]     \n"
                        "fmla   v21.4s, v26.4s, v5.s[2]     \n"
                        "fmla   v22.4s, v26.4s, v7.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v29.s[2]    \n"
                        "fmla   v20.4s, v27.4s, v3.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v5.s[3]     \n"
                        "fmla   v22.4s, v27.4s, v7.s[3]     \n"
                        "fmla   v23.4s, v27.4s, v29.s[3]    \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%4], #64 \n" // r30 r31 r32 r33

                        "fmla   v20.4s, v16.4s, v4.s[0]     \n"
                        "fmla   v21.4s, v16.4s, v6.s[0]     \n"
                        "fmla   v22.4s, v16.4s, v28.s[0]    \n"
                        "fmla   v23.4s, v16.4s, v30.s[0]    \n"
                        "fmla   v20.4s, v17.4s, v4.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v6.s[1]     \n"
                        "fmla   v22.4s, v17.4s, v28.s[1]    \n"
                        "fmla   v23.4s, v17.4s, v30.s[1]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v18.4s, v4.s[2]     \n"
                        "fmla   v21.4s, v18.4s, v6.s[2]     \n"
                        "fmla   v22.4s, v18.4s, v28.s[2]    \n"
                        "fmla   v23.4s, v18.4s, v30.s[2]    \n"
                        "fmla   v20.4s, v19.4s, v4.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v6.s[3]     \n"
                        "fmla   v22.4s, v19.4s, v28.s[3]    \n"
                        "fmla   v23.4s, v19.4s, v30.s[3]    \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%4], #64 \n" // r34 r35 r36 r37

                        "fmla   v20.4s, v24.4s, v0.s[0]     \n"
                        "fmla   v21.4s, v24.4s, v2.s[0]     \n"
                        "fmla   v22.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v6.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v0.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v2.s[1]     \n"
                        "fmla   v22.4s, v25.4s, v4.s[1]     \n"
                        "fmla   v23.4s, v25.4s, v6.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v26.4s, v0.s[2]     \n"
                        "fmla   v21.4s, v26.4s, v2.s[2]     \n"
                        "fmla   v22.4s, v26.4s, v4.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v6.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v0.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v2.s[3]     \n"
                        "fmla   v22.4s, v27.4s, v4.s[3]     \n"
                        "fmla   v23.4s, v27.4s, v6.s[3]     \n"

                        "prfm   pldl1keep, [%4, #384]       \n"
                        "ld1    {v28.4s, v29.4s, v30.4s}, [%4] \n" // r38 r39 r310

                        "fmla   v20.4s, v16.4s, v1.s[0]     \n"
                        "fmla   v21.4s, v16.4s, v3.s[0]     \n"
                        "fmla   v22.4s, v16.4s, v5.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v7.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v1.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v3.s[1]     \n"
                        "fmla   v22.4s, v17.4s, v5.s[1]     \n"
                        "fmla   v23.4s, v17.4s, v7.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v18.4s, v1.s[2]     \n"
                        "fmla   v21.4s, v18.4s, v3.s[2]     \n"
                        "fmla   v22.4s, v18.4s, v5.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v7.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v1.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v3.s[3]     \n"
                        "fmla   v22.4s, v19.4s, v5.s[3]     \n"
                        "fmla   v23.4s, v19.4s, v7.s[3]     \n"

                        "fmla   v20.4s, v24.4s, v2.s[0]     \n"
                        "fmla   v21.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v22.4s, v24.4s, v6.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v28.s[0]    \n"
                        "fmla   v20.4s, v25.4s, v2.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v4.s[1]     \n"
                        "fmla   v22.4s, v25.4s, v6.s[1]     \n"
                        "fmla   v23.4s, v25.4s, v28.s[1]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v26.4s, v2.s[2]     \n"
                        "fmla   v21.4s, v26.4s, v4.s[2]     \n"
                        "fmla   v22.4s, v26.4s, v6.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v28.s[2]    \n"
                        "fmla   v20.4s, v27.4s, v2.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v4.s[3]     \n"
                        "fmla   v22.4s, v27.4s, v6.s[3]     \n"
                        "fmla   v23.4s, v27.4s, v28.s[3]    \n"

                        "fmla   v20.4s, v16.4s, v3.s[0]     \n"
                        "fmla   v21.4s, v16.4s, v5.s[0]     \n"
                        "fmla   v22.4s, v16.4s, v7.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v29.s[0]    \n"
                        "fmla   v20.4s, v17.4s, v3.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v5.s[1]     \n"
                        "fmla   v22.4s, v17.4s, v7.s[1]     \n"
                        "fmla   v23.4s, v17.4s, v29.s[1]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v18.4s, v3.s[2]     \n"
                        "fmla   v21.4s, v18.4s, v5.s[2]     \n"
                        "fmla   v22.4s, v18.4s, v7.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v29.s[2]    \n"
                        "fmla   v20.4s, v19.4s, v3.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v5.s[3]     \n"
                        "fmla   v22.4s, v19.4s, v7.s[3]     \n"
                        "fmla   v23.4s, v19.4s, v29.s[3]    \n"

                        "prfm   pldl1keep, [%5, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%5], #64 \n" // r40 r41 r42 r43

                        "fmla   v20.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v21.4s, v24.4s, v6.s[0]     \n"
                        "fmla   v22.4s, v24.4s, v28.s[0]    \n"
                        "fmla   v23.4s, v24.4s, v30.s[0]    \n"
                        "fmla   v20.4s, v25.4s, v4.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v6.s[1]     \n"
                        "fmla   v22.4s, v25.4s, v28.s[1]    \n"
                        "fmla   v23.4s, v25.4s, v30.s[1]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v26.4s, v4.s[2]     \n"
                        "fmla   v21.4s, v26.4s, v6.s[2]     \n"
                        "fmla   v22.4s, v26.4s, v28.s[2]    \n"
                        "fmla   v23.4s, v26.4s, v30.s[2]    \n"
                        "fmla   v20.4s, v27.4s, v4.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v6.s[3]     \n"
                        "fmla   v22.4s, v27.4s, v28.s[3]    \n"
                        "fmla   v23.4s, v27.4s, v30.s[3]    \n"

                        "prfm   pldl1keep, [%5, #512]       \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%5], #64 \n" // r44 r45 r46 r47

                        "fmla   v20.4s, v16.4s, v0.s[0]     \n"
                        "fmla   v21.4s, v16.4s, v2.s[0]     \n"
                        "fmla   v22.4s, v16.4s, v4.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v6.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v0.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v2.s[1]     \n"
                        "fmla   v22.4s, v17.4s, v4.s[1]     \n"
                        "fmla   v23.4s, v17.4s, v6.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v18.4s, v0.s[2]     \n"
                        "fmla   v21.4s, v18.4s, v2.s[2]     \n"
                        "fmla   v22.4s, v18.4s, v4.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v6.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v0.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v2.s[3]     \n"
                        "fmla   v22.4s, v19.4s, v4.s[3]     \n"
                        "fmla   v23.4s, v19.4s, v6.s[3]     \n"

                        "prfm   pldl1keep, [%5, #384]       \n"
                        "ld1    {v28.4s, v29.4s, v30.4s}, [%5] \n" // r48 r49 r410

                        "fmla   v20.4s, v24.4s, v1.s[0]     \n"
                        "fmla   v21.4s, v24.4s, v3.s[0]     \n"
                        "fmla   v22.4s, v24.4s, v5.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v7.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v1.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v3.s[1]     \n"
                        "fmla   v22.4s, v25.4s, v5.s[1]     \n"
                        "fmla   v23.4s, v25.4s, v7.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v26.4s, v1.s[2]     \n"
                        "fmla   v21.4s, v26.4s, v3.s[2]     \n"
                        "fmla   v22.4s, v26.4s, v5.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v7.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v1.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v3.s[3]     \n"
                        "fmla   v22.4s, v27.4s, v5.s[3]     \n"
                        "fmla   v23.4s, v27.4s, v7.s[3]     \n"

                        "fmla   v20.4s, v16.4s, v2.s[0]     \n"
                        "fmla   v21.4s, v16.4s, v4.s[0]     \n"
                        "fmla   v22.4s, v16.4s, v6.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v28.s[0]    \n"
                        "fmla   v20.4s, v17.4s, v2.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v4.s[1]     \n"
                        "fmla   v22.4s, v17.4s, v6.s[1]     \n"
                        "fmla   v23.4s, v17.4s, v28.s[1]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v20.4s, v18.4s, v2.s[2]     \n"
                        "fmla   v21.4s, v18.4s, v4.s[2]     \n"
                        "fmla   v22.4s, v18.4s, v6.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v28.s[2]    \n"
                        "fmla   v20.4s, v19.4s, v2.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v4.s[3]     \n"
                        "fmla   v22.4s, v19.4s, v6.s[3]     \n"
                        "fmla   v23.4s, v19.4s, v28.s[3]    \n"

                        "fmla   v20.4s, v24.4s, v3.s[0]     \n"
                        "fmla   v21.4s, v24.4s, v5.s[0]     \n"
                        "fmla   v22.4s, v24.4s, v7.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v29.s[0]    \n"
                        "fmla   v20.4s, v25.4s, v3.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v5.s[1]     \n"
                        "fmla   v22.4s, v25.4s, v7.s[1]     \n"
                        "fmla   v23.4s, v25.4s, v29.s[1]    \n"

                        //                         "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6] \n"

                        "fmla   v20.4s, v26.4s, v3.s[2]     \n"
                        "fmla   v21.4s, v26.4s, v5.s[2]     \n"
                        "fmla   v22.4s, v26.4s, v7.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v29.s[2]    \n"
                        "fmla   v20.4s, v27.4s, v3.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v5.s[3]     \n"
                        "fmla   v22.4s, v27.4s, v7.s[3]     \n"
                        "fmla   v23.4s, v27.4s, v29.s[3]    \n"

                        "fmla   v20.4s, v16.4s, v4.s[0]     \n"
                        "fmla   v21.4s, v16.4s, v6.s[0]     \n"
                        "fmla   v22.4s, v16.4s, v28.s[0]    \n"
                        "fmla   v23.4s, v16.4s, v30.s[0]    \n"
                        "fmla   v20.4s, v17.4s, v4.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v6.s[1]     \n"
                        "fmla   v22.4s, v17.4s, v28.s[1]    \n"
                        "fmla   v23.4s, v17.4s, v30.s[1]    \n"
                        "fmla   v20.4s, v18.4s, v4.s[2]     \n"
                        "fmla   v21.4s, v18.4s, v6.s[2]     \n"
                        "fmla   v22.4s, v18.4s, v28.s[2]    \n"
                        "fmla   v23.4s, v18.4s, v30.s[2]    \n"
                        "fmla   v20.4s, v19.4s, v4.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v6.s[3]     \n"
                        "fmla   v22.4s, v19.4s, v28.s[3]    \n"
                        "fmla   v23.4s, v19.4s, v30.s[3]    \n"

                        "sub    %6, %6, #1536               \n" // kptr -= 24 * 16;

                        "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2),      // %3
                        "=r"(r3),      // %4
                        "=r"(r4),      // %5
                        "=r"(kptr)     // %6
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "4"(r3),
                        "5"(r4),
                        "6"(kptr)
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30");
#else  // __aarch64__
                    asm volatile(
                        "pld        [%0, #512]          \n"
                        "vldm       %0, {d24-d31}       \n" // sum0 sum1 sum2 sum3

                        "pld        [%1, #512]          \n"
                        "vldm       %1!, {d0-d7}        \n" // r00 r01 r02 r03

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%1, #512]          \n"
                        "vldm       %1!, {d8-d15}       \n" // r04 r05 r06 r07

                        "vmla.f32   q12, q8, d0[0]      \n"
                        "vmla.f32   q13, q8, d4[0]      \n"
                        "vmla.f32   q14, q8, d8[0]      \n"
                        "vmla.f32   q15, q8, d12[0]     \n"
                        "vmla.f32   q12, q9, d0[1]      \n"
                        "vmla.f32   q13, q9, d4[1]      \n"
                        "vmla.f32   q14, q9, d8[1]      \n"
                        "vmla.f32   q15, q9, d12[1]     \n"
                        "vmla.f32   q12, q10, d1[0]     \n"
                        "vmla.f32   q13, q10, d5[0]     \n"
                        "vmla.f32   q14, q10, d9[0]     \n"
                        "vmla.f32   q15, q10, d13[0]    \n"
                        "vmla.f32   q12, q11, d1[1]     \n"
                        "vmla.f32   q13, q11, d5[1]     \n"
                        "vmla.f32   q14, q11, d9[1]     \n"
                        "vmla.f32   q15, q11, d13[1]    \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q12, q8, d2[0]      \n"
                        "vmla.f32   q13, q8, d6[0]      \n"
                        "vmla.f32   q14, q8, d10[0]     \n"
                        "vmla.f32   q15, q8, d14[0]     \n"
                        "vmla.f32   q12, q9, d2[1]      \n"
                        "vmla.f32   q13, q9, d6[1]      \n"
                        "vmla.f32   q14, q9, d10[1]     \n"
                        "vmla.f32   q15, q9, d14[1]     \n"
                        "vmla.f32   q12, q10, d3[0]     \n"
                        "vmla.f32   q13, q10, d7[0]     \n"
                        "vmla.f32   q14, q10, d11[0]    \n"
                        "vmla.f32   q15, q10, d15[0]    \n"
                        "vmla.f32   q12, q11, d3[1]     \n"
                        "vmla.f32   q13, q11, d7[1]     \n"
                        "vmla.f32   q14, q11, d11[1]    \n"
                        "vmla.f32   q15, q11, d15[1]    \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%1, #256]          \n"
                        "vld1.f32   {d0-d3}, [%1 :128]! \n" // r08 r09

                        "vmla.f32   q12, q8, d4[0]      \n"
                        "vmla.f32   q13, q8, d8[0]      \n"
                        "vmla.f32   q14, q8, d12[0]     \n"
                        "vmla.f32   q15, q8, d0[0]      \n"
                        "vmla.f32   q12, q9, d4[1]      \n"
                        "vmla.f32   q13, q9, d8[1]      \n"
                        "vmla.f32   q14, q9, d12[1]     \n"
                        "vmla.f32   q15, q9, d0[1]      \n"
                        "vmla.f32   q12, q10, d5[0]     \n"
                        "vmla.f32   q13, q10, d9[0]     \n"
                        "vmla.f32   q14, q10, d13[0]    \n"
                        "vmla.f32   q15, q10, d1[0]     \n"
                        "vmla.f32   q12, q11, d5[1]     \n"
                        "vmla.f32   q13, q11, d9[1]     \n"
                        "vmla.f32   q14, q11, d13[1]    \n"
                        "vmla.f32   q15, q11, d1[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q12, q8, d6[0]      \n"
                        "vmla.f32   q13, q8, d10[0]     \n"
                        "vmla.f32   q14, q8, d14[0]     \n"
                        "vmla.f32   q15, q8, d2[0]      \n"
                        "vmla.f32   q12, q9, d6[1]      \n"
                        "vmla.f32   q13, q9, d10[1]     \n"
                        "vmla.f32   q14, q9, d14[1]     \n"
                        "vmla.f32   q15, q9, d2[1]      \n"
                        "vmla.f32   q12, q10, d7[0]     \n"
                        "vmla.f32   q13, q10, d11[0]    \n"
                        "vmla.f32   q14, q10, d15[0]    \n"
                        "vmla.f32   q15, q10, d3[0]     \n"
                        "vmla.f32   q12, q11, d7[1]     \n"
                        "vmla.f32   q13, q11, d11[1]    \n"
                        "vmla.f32   q14, q11, d15[1]    \n"
                        "vmla.f32   q15, q11, d3[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%1, #128]          \n"
                        "vld1.f32   {d4-d5}, [%1 :128]  \n" // r010

                        "vmla.f32   q12, q8, d8[0]      \n"
                        "vmla.f32   q13, q8, d12[0]     \n"
                        "vmla.f32   q14, q8, d0[0]      \n"
                        "vmla.f32   q15, q8, d4[0]      \n"
                        "vmla.f32   q12, q9, d8[1]      \n"
                        "vmla.f32   q13, q9, d12[1]     \n"
                        "vmla.f32   q14, q9, d0[1]      \n"
                        "vmla.f32   q15, q9, d4[1]      \n"
                        "vmla.f32   q12, q10, d9[0]     \n"
                        "vmla.f32   q13, q10, d13[0]    \n"
                        "vmla.f32   q14, q10, d1[0]     \n"
                        "vmla.f32   q15, q10, d5[0]     \n"
                        "vmla.f32   q12, q11, d9[1]     \n"
                        "vmla.f32   q13, q11, d13[1]    \n"

                        "pld        [%2, #512]          \n"
                        "vldm       %2!, {d8-d15}       \n" // r10 r11 r12 r13

                        "vmla.f32   q14, q11, d1[1]     \n"
                        "vmla.f32   q15, q11, d5[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%2, #512]          \n"
                        "vldm       %2!, {d0-d7}        \n" // r14 r15 r16 r17

                        "vmla.f32   q12, q8, d8[0]      \n"
                        "vmla.f32   q13, q8, d12[0]     \n"
                        "vmla.f32   q14, q8, d0[0]      \n"
                        "vmla.f32   q15, q8, d4[0]      \n"
                        "vmla.f32   q12, q9, d8[1]      \n"
                        "vmla.f32   q13, q9, d12[1]     \n"
                        "vmla.f32   q14, q9, d0[1]      \n"
                        "vmla.f32   q15, q9, d4[1]      \n"
                        "vmla.f32   q12, q10, d9[0]     \n"
                        "vmla.f32   q13, q10, d13[0]    \n"
                        "vmla.f32   q14, q10, d1[0]     \n"
                        "vmla.f32   q15, q10, d5[0]     \n"
                        "vmla.f32   q12, q11, d9[1]     \n"
                        "vmla.f32   q13, q11, d13[1]    \n"
                        "vmla.f32   q14, q11, d1[1]     \n"
                        "vmla.f32   q15, q11, d5[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q12, q8, d10[0]     \n"
                        "vmla.f32   q13, q8, d14[0]     \n"
                        "vmla.f32   q14, q8, d2[0]      \n"
                        "vmla.f32   q15, q8, d6[0]      \n"
                        "vmla.f32   q12, q9, d10[1]     \n"
                        "vmla.f32   q13, q9, d14[1]     \n"
                        "vmla.f32   q14, q9, d2[1]      \n"
                        "vmla.f32   q15, q9, d6[1]      \n"
                        "vmla.f32   q12, q10, d11[0]    \n"
                        "vmla.f32   q13, q10, d15[0]    \n"
                        "vmla.f32   q14, q10, d3[0]     \n"
                        "vmla.f32   q15, q10, d7[0]     \n"
                        "vmla.f32   q12, q11, d11[1]    \n"
                        "vmla.f32   q13, q11, d15[1]    \n"
                        "vmla.f32   q14, q11, d3[1]     \n"
                        "vmla.f32   q15, q11, d7[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%2, #256]          \n"
                        "vld1.f32   {d8-d11}, [%2 :128]! \n" // r18 r19

                        "vmla.f32   q12, q8, d12[0]     \n"
                        "vmla.f32   q13, q8, d0[0]      \n"
                        "vmla.f32   q14, q8, d4[0]      \n"
                        "vmla.f32   q15, q8, d8[0]      \n"
                        "vmla.f32   q12, q9, d12[1]     \n"
                        "vmla.f32   q13, q9, d0[1]      \n"
                        "vmla.f32   q14, q9, d4[1]      \n"
                        "vmla.f32   q15, q9, d8[1]      \n"
                        "vmla.f32   q12, q10, d13[0]    \n"
                        "vmla.f32   q13, q10, d1[0]     \n"
                        "vmla.f32   q14, q10, d5[0]     \n"
                        "vmla.f32   q15, q10, d9[0]     \n"
                        "vmla.f32   q12, q11, d13[1]    \n"
                        "vmla.f32   q13, q11, d1[1]     \n"
                        "vmla.f32   q14, q11, d5[1]     \n"
                        "vmla.f32   q15, q11, d9[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q12, q8, d14[0]     \n"
                        "vmla.f32   q13, q8, d2[0]      \n"
                        "vmla.f32   q14, q8, d6[0]      \n"
                        "vmla.f32   q15, q8, d10[0]     \n"
                        "vmla.f32   q12, q9, d14[1]     \n"
                        "vmla.f32   q13, q9, d2[1]      \n"
                        "vmla.f32   q14, q9, d6[1]      \n"
                        "vmla.f32   q15, q9, d10[1]     \n"
                        "vmla.f32   q12, q10, d15[0]    \n"
                        "vmla.f32   q13, q10, d3[0]     \n"
                        "vmla.f32   q14, q10, d7[0]     \n"
                        "vmla.f32   q15, q10, d11[0]    \n"
                        "vmla.f32   q12, q11, d15[1]    \n"
                        "vmla.f32   q13, q11, d3[1]     \n"
                        "vmla.f32   q14, q11, d7[1]     \n"
                        "vmla.f32   q15, q11, d11[1]    \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%2, #128]          \n"
                        "vld1.f32   {d12-d13}, [%2 :128] \n" // r110

                        "vmla.f32   q12, q8, d0[0]      \n"
                        "vmla.f32   q13, q8, d4[0]      \n"
                        "vmla.f32   q14, q8, d8[0]      \n"
                        "vmla.f32   q15, q8, d12[0]     \n"
                        "vmla.f32   q12, q9, d0[1]      \n"
                        "vmla.f32   q13, q9, d4[1]      \n"
                        "vmla.f32   q14, q9, d8[1]      \n"
                        "vmla.f32   q15, q9, d12[1]     \n"
                        "vmla.f32   q12, q10, d1[0]     \n"
                        "vmla.f32   q13, q10, d5[0]     \n"
                        "vmla.f32   q14, q10, d9[0]     \n"
                        "vmla.f32   q15, q10, d13[0]    \n"
                        "vmla.f32   q12, q11, d1[1]     \n"
                        "vmla.f32   q13, q11, d5[1]     \n"

                        "pld        [%3, #512]          \n"
                        "vldm       %3!, {d0-d7}        \n" // r20 r21 r22 r23

                        "vmla.f32   q14, q11, d9[1]     \n"
                        "vmla.f32   q15, q11, d13[1]    \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%3, #512]          \n"
                        "vldm       %3!, {d8-d15}       \n" // r24 r25 r26 r27

                        "vmla.f32   q12, q8, d0[0]      \n"
                        "vmla.f32   q13, q8, d4[0]      \n"
                        "vmla.f32   q14, q8, d8[0]      \n"
                        "vmla.f32   q15, q8, d12[0]     \n"
                        "vmla.f32   q12, q9, d0[1]      \n"
                        "vmla.f32   q13, q9, d4[1]      \n"
                        "vmla.f32   q14, q9, d8[1]      \n"
                        "vmla.f32   q15, q9, d12[1]     \n"
                        "vmla.f32   q12, q10, d1[0]     \n"
                        "vmla.f32   q13, q10, d5[0]     \n"
                        "vmla.f32   q14, q10, d9[0]     \n"
                        "vmla.f32   q15, q10, d13[0]    \n"
                        "vmla.f32   q12, q11, d1[1]     \n"
                        "vmla.f32   q13, q11, d5[1]     \n"
                        "vmla.f32   q14, q11, d9[1]     \n"
                        "vmla.f32   q15, q11, d13[1]    \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q12, q8, d2[0]      \n"
                        "vmla.f32   q13, q8, d6[0]      \n"
                        "vmla.f32   q14, q8, d10[0]     \n"
                        "vmla.f32   q15, q8, d14[0]     \n"
                        "vmla.f32   q12, q9, d2[1]      \n"
                        "vmla.f32   q13, q9, d6[1]      \n"
                        "vmla.f32   q14, q9, d10[1]     \n"
                        "vmla.f32   q15, q9, d14[1]     \n"
                        "vmla.f32   q12, q10, d3[0]     \n"
                        "vmla.f32   q13, q10, d7[0]     \n"
                        "vmla.f32   q14, q10, d11[0]    \n"
                        "vmla.f32   q15, q10, d15[0]    \n"
                        "vmla.f32   q12, q11, d3[1]     \n"
                        "vmla.f32   q13, q11, d7[1]     \n"
                        "vmla.f32   q14, q11, d11[1]    \n"
                        "vmla.f32   q15, q11, d15[1]    \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%3, #256]          \n"
                        "vld1.f32   {d0-d3}, [%3 :128]! \n" // r28 r29

                        "vmla.f32   q12, q8, d4[0]      \n"
                        "vmla.f32   q13, q8, d8[0]      \n"
                        "vmla.f32   q14, q8, d12[0]     \n"
                        "vmla.f32   q15, q8, d0[0]      \n"
                        "vmla.f32   q12, q9, d4[1]      \n"
                        "vmla.f32   q13, q9, d8[1]      \n"
                        "vmla.f32   q14, q9, d12[1]     \n"
                        "vmla.f32   q15, q9, d0[1]      \n"
                        "vmla.f32   q12, q10, d5[0]     \n"
                        "vmla.f32   q13, q10, d9[0]     \n"
                        "vmla.f32   q14, q10, d13[0]    \n"
                        "vmla.f32   q15, q10, d1[0]     \n"
                        "vmla.f32   q12, q11, d5[1]     \n"
                        "vmla.f32   q13, q11, d9[1]     \n"
                        "vmla.f32   q14, q11, d13[1]    \n"
                        "vmla.f32   q15, q11, d1[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q12, q8, d6[0]      \n"
                        "vmla.f32   q13, q8, d10[0]     \n"
                        "vmla.f32   q14, q8, d14[0]     \n"
                        "vmla.f32   q15, q8, d2[0]      \n"
                        "vmla.f32   q12, q9, d6[1]      \n"
                        "vmla.f32   q13, q9, d10[1]     \n"
                        "vmla.f32   q14, q9, d14[1]     \n"
                        "vmla.f32   q15, q9, d2[1]      \n"
                        "vmla.f32   q12, q10, d7[0]     \n"
                        "vmla.f32   q13, q10, d11[0]    \n"
                        "vmla.f32   q14, q10, d15[0]    \n"
                        "vmla.f32   q15, q10, d3[0]     \n"
                        "vmla.f32   q12, q11, d7[1]     \n"
                        "vmla.f32   q13, q11, d11[1]    \n"
                        "vmla.f32   q14, q11, d15[1]    \n"
                        "vmla.f32   q15, q11, d3[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%3, #128]          \n"
                        "vld1.f32   {d4-d5}, [%3 :128]  \n" // r210

                        "vmla.f32   q12, q8, d8[0]      \n"
                        "vmla.f32   q13, q8, d12[0]     \n"
                        "vmla.f32   q14, q8, d0[0]      \n"
                        "vmla.f32   q15, q8, d4[0]      \n"
                        "vmla.f32   q12, q9, d8[1]      \n"
                        "vmla.f32   q13, q9, d12[1]     \n"
                        "vmla.f32   q14, q9, d0[1]      \n"
                        "vmla.f32   q15, q9, d4[1]      \n"
                        "vmla.f32   q12, q10, d9[0]     \n"
                        "vmla.f32   q13, q10, d13[0]    \n"
                        "vmla.f32   q14, q10, d1[0]     \n"
                        "vmla.f32   q15, q10, d5[0]     \n"
                        "vmla.f32   q12, q11, d9[1]     \n"
                        "vmla.f32   q13, q11, d13[1]    \n"

                        "pld        [%4, #512]          \n"
                        "vldm       %4!, {d8-d15}       \n" // r30 r31 r32 r33

                        "vmla.f32   q14, q11, d1[1]     \n"
                        "vmla.f32   q15, q11, d5[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%4, #512]          \n"
                        "vldm       %4!, {d0-d7}        \n" // r34 r35 r36 r37

                        "vmla.f32   q12, q8, d8[0]      \n"
                        "vmla.f32   q13, q8, d12[0]     \n"
                        "vmla.f32   q14, q8, d0[0]      \n"
                        "vmla.f32   q15, q8, d4[0]      \n"
                        "vmla.f32   q12, q9, d8[1]      \n"
                        "vmla.f32   q13, q9, d12[1]     \n"
                        "vmla.f32   q14, q9, d0[1]      \n"
                        "vmla.f32   q15, q9, d4[1]      \n"
                        "vmla.f32   q12, q10, d9[0]     \n"
                        "vmla.f32   q13, q10, d13[0]    \n"
                        "vmla.f32   q14, q10, d1[0]     \n"
                        "vmla.f32   q15, q10, d5[0]     \n"
                        "vmla.f32   q12, q11, d9[1]     \n"
                        "vmla.f32   q13, q11, d13[1]    \n"
                        "vmla.f32   q14, q11, d1[1]     \n"
                        "vmla.f32   q15, q11, d5[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q12, q8, d10[0]     \n"
                        "vmla.f32   q13, q8, d14[0]     \n"
                        "vmla.f32   q14, q8, d2[0]      \n"
                        "vmla.f32   q15, q8, d6[0]      \n"
                        "vmla.f32   q12, q9, d10[1]     \n"
                        "vmla.f32   q13, q9, d14[1]     \n"
                        "vmla.f32   q14, q9, d2[1]      \n"
                        "vmla.f32   q15, q9, d6[1]      \n"
                        "vmla.f32   q12, q10, d11[0]    \n"
                        "vmla.f32   q13, q10, d15[0]    \n"
                        "vmla.f32   q14, q10, d3[0]     \n"
                        "vmla.f32   q15, q10, d7[0]     \n"
                        "vmla.f32   q12, q11, d11[1]    \n"
                        "vmla.f32   q13, q11, d15[1]    \n"
                        "vmla.f32   q14, q11, d3[1]     \n"
                        "vmla.f32   q15, q11, d7[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%4, #256]          \n"
                        "vld1.f32   {d8-d11}, [%4 :128]! \n" // r38 r39

                        "vmla.f32   q12, q8, d12[0]     \n"
                        "vmla.f32   q13, q8, d0[0]      \n"
                        "vmla.f32   q14, q8, d4[0]      \n"
                        "vmla.f32   q15, q8, d8[0]      \n"
                        "vmla.f32   q12, q9, d12[1]     \n"
                        "vmla.f32   q13, q9, d0[1]      \n"
                        "vmla.f32   q14, q9, d4[1]      \n"
                        "vmla.f32   q15, q9, d8[1]      \n"
                        "vmla.f32   q12, q10, d13[0]    \n"
                        "vmla.f32   q13, q10, d1[0]     \n"
                        "vmla.f32   q14, q10, d5[0]     \n"
                        "vmla.f32   q15, q10, d9[0]     \n"
                        "vmla.f32   q12, q11, d13[1]    \n"
                        "vmla.f32   q13, q11, d1[1]     \n"
                        "vmla.f32   q14, q11, d5[1]     \n"
                        "vmla.f32   q15, q11, d9[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q12, q8, d14[0]     \n"
                        "vmla.f32   q13, q8, d2[0]      \n"
                        "vmla.f32   q14, q8, d6[0]      \n"
                        "vmla.f32   q15, q8, d10[0]     \n"
                        "vmla.f32   q12, q9, d14[1]     \n"
                        "vmla.f32   q13, q9, d2[1]      \n"
                        "vmla.f32   q14, q9, d6[1]      \n"
                        "vmla.f32   q15, q9, d10[1]     \n"
                        "vmla.f32   q12, q10, d15[0]    \n"
                        "vmla.f32   q13, q10, d3[0]     \n"
                        "vmla.f32   q14, q10, d7[0]     \n"
                        "vmla.f32   q15, q10, d11[0]    \n"
                        "vmla.f32   q12, q11, d15[1]    \n"
                        "vmla.f32   q13, q11, d3[1]     \n"
                        "vmla.f32   q14, q11, d7[1]     \n"
                        "vmla.f32   q15, q11, d11[1]    \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%4, #128]          \n"
                        "vld1.f32   {d12-d13}, [%4 :128] \n" // r310

                        "vmla.f32   q12, q8, d0[0]      \n"
                        "vmla.f32   q13, q8, d4[0]      \n"
                        "vmla.f32   q14, q8, d8[0]      \n"
                        "vmla.f32   q15, q8, d12[0]     \n"
                        "vmla.f32   q12, q9, d0[1]      \n"
                        "vmla.f32   q13, q9, d4[1]      \n"
                        "vmla.f32   q14, q9, d8[1]      \n"
                        "vmla.f32   q15, q9, d12[1]     \n"
                        "vmla.f32   q12, q10, d1[0]     \n"
                        "vmla.f32   q13, q10, d5[0]     \n"
                        "vmla.f32   q14, q10, d9[0]     \n"
                        "vmla.f32   q15, q10, d13[0]    \n"
                        "vmla.f32   q12, q11, d1[1]     \n"
                        "vmla.f32   q13, q11, d5[1]     \n"

                        "pld        [%5, #512]          \n"
                        "vldm       %5!, {d0-d7}        \n" // r40 r41 r42 r43

                        "vmla.f32   q14, q11, d9[1]     \n"
                        "vmla.f32   q15, q11, d13[1]    \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%5, #512]          \n"
                        "vldm       %5!, {d8-d15}       \n" // r44 r45 r46 r47

                        "vmla.f32   q12, q8, d0[0]      \n"
                        "vmla.f32   q13, q8, d4[0]      \n"
                        "vmla.f32   q14, q8, d8[0]      \n"
                        "vmla.f32   q15, q8, d12[0]     \n"
                        "vmla.f32   q12, q9, d0[1]      \n"
                        "vmla.f32   q13, q9, d4[1]      \n"
                        "vmla.f32   q14, q9, d8[1]      \n"
                        "vmla.f32   q15, q9, d12[1]     \n"
                        "vmla.f32   q12, q10, d1[0]     \n"
                        "vmla.f32   q13, q10, d5[0]     \n"
                        "vmla.f32   q14, q10, d9[0]     \n"
                        "vmla.f32   q15, q10, d13[0]    \n"
                        "vmla.f32   q12, q11, d1[1]     \n"
                        "vmla.f32   q13, q11, d5[1]     \n"
                        "vmla.f32   q14, q11, d9[1]     \n"
                        "vmla.f32   q15, q11, d13[1]    \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q12, q8, d2[0]      \n"
                        "vmla.f32   q13, q8, d6[0]      \n"
                        "vmla.f32   q14, q8, d10[0]     \n"
                        "vmla.f32   q15, q8, d14[0]     \n"
                        "vmla.f32   q12, q9, d2[1]      \n"
                        "vmla.f32   q13, q9, d6[1]      \n"
                        "vmla.f32   q14, q9, d10[1]     \n"
                        "vmla.f32   q15, q9, d14[1]     \n"
                        "vmla.f32   q12, q10, d3[0]     \n"
                        "vmla.f32   q13, q10, d7[0]     \n"
                        "vmla.f32   q14, q10, d11[0]    \n"
                        "vmla.f32   q15, q10, d15[0]    \n"
                        "vmla.f32   q12, q11, d3[1]     \n"
                        "vmla.f32   q13, q11, d7[1]     \n"
                        "vmla.f32   q14, q11, d11[1]    \n"
                        "vmla.f32   q15, q11, d15[1]    \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%5, #256]          \n"
                        "vld1.f32   {d0-d3}, [%5 :128]! \n" // r48 r49

                        "vmla.f32   q12, q8, d4[0]      \n"
                        "vmla.f32   q13, q8, d8[0]      \n"
                        "vmla.f32   q14, q8, d12[0]     \n"
                        "vmla.f32   q15, q8, d0[0]      \n"
                        "vmla.f32   q12, q9, d4[1]      \n"
                        "vmla.f32   q13, q9, d8[1]      \n"
                        "vmla.f32   q14, q9, d12[1]     \n"
                        "vmla.f32   q15, q9, d0[1]      \n"
                        "vmla.f32   q12, q10, d5[0]     \n"
                        "vmla.f32   q13, q10, d9[0]     \n"
                        "vmla.f32   q14, q10, d13[0]    \n"
                        "vmla.f32   q15, q10, d1[0]     \n"
                        "vmla.f32   q12, q11, d5[1]     \n"
                        "vmla.f32   q13, q11, d9[1]     \n"
                        "vmla.f32   q14, q11, d13[1]    \n"
                        "vmla.f32   q15, q11, d1[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q12, q8, d6[0]      \n"
                        "vmla.f32   q13, q8, d10[0]     \n"
                        "vmla.f32   q14, q8, d14[0]     \n"
                        "vmla.f32   q15, q8, d2[0]      \n"
                        "vmla.f32   q12, q9, d6[1]      \n"
                        "vmla.f32   q13, q9, d10[1]     \n"
                        "vmla.f32   q14, q9, d14[1]     \n"
                        "vmla.f32   q15, q9, d2[1]      \n"
                        "vmla.f32   q12, q10, d7[0]     \n"
                        "vmla.f32   q13, q10, d11[0]    \n"
                        "vmla.f32   q14, q10, d15[0]    \n"
                        "vmla.f32   q15, q10, d3[0]     \n"
                        "vmla.f32   q12, q11, d7[1]     \n"
                        "vmla.f32   q13, q11, d11[1]    \n"
                        "vmla.f32   q14, q11, d15[1]    \n"
                        "vmla.f32   q15, q11, d3[1]     \n"

                        //                         "pld        [%6, #512]          \n"
                        "vldm       %6, {d16-d23}       \n"

                        "pld        [%5, #128]          \n"
                        "vld1.f32   {d4-d5}, [%5 :128]  \n" // r410

                        "vmla.f32   q12, q8, d8[0]      \n"
                        "vmla.f32   q13, q8, d12[0]     \n"
                        "vmla.f32   q14, q8, d0[0]      \n"
                        "vmla.f32   q15, q8, d4[0]      \n"
                        "vmla.f32   q12, q9, d8[1]      \n"
                        "vmla.f32   q13, q9, d12[1]     \n"
                        "vmla.f32   q14, q9, d0[1]      \n"
                        "vmla.f32   q15, q9, d4[1]      \n"
                        "vmla.f32   q12, q10, d9[0]     \n"
                        "vmla.f32   q13, q10, d13[0]    \n"
                        "vmla.f32   q14, q10, d1[0]     \n"
                        "vmla.f32   q15, q10, d5[0]     \n"
                        "vmla.f32   q12, q11, d9[1]     \n"
                        "vmla.f32   q13, q11, d13[1]    \n"
                        "vmla.f32   q14, q11, d1[1]     \n"
                        "vmla.f32   q15, q11, d5[1]     \n"

                        "sub        %6, %6, #1536       \n" // kptr -= 24 * 16;

                        "sub        %1, %1, #32         \n"
                        "sub        %2, %2, #32         \n"
                        "sub        %3, %3, #32         \n"
                        "sub        %4, %4, #32         \n"
                        "sub        %5, %5, #32         \n"

                        "vstm       %0!, {d24-d31}      \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2),      // %3
                        "=r"(r3),      // %4
                        "=r"(r4),      // %5
                        "=r"(kptr)     // %6
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "4"(r3),
                        "5"(r4),
                        "6"(kptr)
                        : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif // __aarch64__
                }
                for (; j + 1 < outw; j += 2)
                {
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%0, #256]       \n"
                        "ld1    {v20.4s, v21.4s}, [%0]      \n" // sum0 sum1

                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%1], #64 \n" // r00 r01 r02 r03

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmul   v22.4s, v16.4s, v0.s[0]     \n"
                        "fmul   v23.4s, v16.4s, v2.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v0.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v2.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v18.4s, v0.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v2.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v0.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v2.s[3]     \n"

                        "prfm   pldl1keep, [%1, #384]       \n"
                        "ld1    {v4.4s, v5.4s, v6.4s}, [%1] \n" // r04 r05 r06

                        "fmla   v22.4s, v24.4s, v1.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v3.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v1.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v3.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v26.4s, v1.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v3.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v1.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v3.s[3]     \n"

                        "fmla   v22.4s, v16.4s, v2.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v4.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v2.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v4.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v18.4s, v2.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v4.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v2.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v4.s[3]     \n"

                        "fmla   v22.4s, v24.4s, v3.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v5.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v3.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v5.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v26.4s, v3.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v5.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v3.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v5.s[3]     \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%2], #64 \n" // r10 r11 r12 r13

                        "fmla   v22.4s, v16.4s, v4.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v6.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v4.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v6.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v18.4s, v4.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v6.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v4.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v6.s[3]     \n"

                        "fmla   v22.4s, v24.4s, v0.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v2.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v0.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v2.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v26.4s, v0.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v2.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v0.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v2.s[3]     \n"

                        "prfm   pldl1keep, [%2, #384]       \n"
                        "ld1    {v4.4s, v5.4s, v6.4s}, [%2] \n" // r14 r15 r16

                        "fmla   v22.4s, v16.4s, v1.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v3.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v1.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v3.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v18.4s, v1.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v3.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v1.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v3.s[3]     \n"

                        "fmla   v22.4s, v24.4s, v2.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v2.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v4.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v26.4s, v2.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v4.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v2.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v4.s[3]     \n"

                        "fmla   v22.4s, v16.4s, v3.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v5.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v3.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v5.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v18.4s, v3.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v5.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v3.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v5.s[3]     \n"

                        "prfm   pldl1keep, [%3, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%3], #64 \n" // r20 r21 r22 r23

                        "fmla   v22.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v6.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v4.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v6.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v26.4s, v4.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v6.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v4.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v6.s[3]     \n"

                        "fmla   v22.4s, v16.4s, v0.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v2.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v0.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v2.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v18.4s, v0.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v2.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v0.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v2.s[3]     \n"

                        "prfm   pldl1keep, [%3, #384]       \n"
                        "ld1    {v4.4s, v5.4s, v6.4s}, [%3] \n" // r24 r25 r26

                        "fmla   v22.4s, v24.4s, v1.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v3.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v1.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v3.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v26.4s, v1.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v3.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v1.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v3.s[3]     \n"

                        "fmla   v22.4s, v16.4s, v2.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v4.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v2.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v4.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v18.4s, v2.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v4.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v2.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v4.s[3]     \n"

                        "fmla   v22.4s, v24.4s, v3.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v5.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v3.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v5.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v26.4s, v3.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v5.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v3.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v5.s[3]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%4], #64 \n" // r30 r31 r32 r33

                        "fmla   v22.4s, v16.4s, v4.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v6.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v4.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v6.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v18.4s, v4.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v6.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v4.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v6.s[3]     \n"

                        "fmla   v22.4s, v24.4s, v0.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v2.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v0.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v2.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v26.4s, v0.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v2.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v0.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v2.s[3]     \n"

                        "prfm   pldl1keep, [%4, #384]       \n"
                        "ld1    {v4.4s, v5.4s, v6.4s}, [%4] \n" // r34 r35 r36

                        "fmla   v22.4s, v16.4s, v1.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v3.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v1.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v3.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v18.4s, v1.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v3.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v1.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v3.s[3]     \n"

                        "fmla   v22.4s, v24.4s, v2.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v2.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v4.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v26.4s, v2.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v4.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v2.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v4.s[3]     \n"

                        "fmla   v22.4s, v16.4s, v3.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v5.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v3.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v5.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v18.4s, v3.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v5.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v3.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v5.s[3]     \n"

                        "prfm   pldl1keep, [%5, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%5], #64 \n" // r40 r41 r42 r43

                        "fmla   v22.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v6.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v4.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v6.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v26.4s, v4.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v6.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v4.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v6.s[3]     \n"

                        "fmla   v22.4s, v16.4s, v0.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v2.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v0.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v2.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v18.4s, v0.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v2.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v0.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v2.s[3]     \n"

                        "prfm   pldl1keep, [%5, #384]       \n"
                        "ld1    {v4.4s, v5.4s, v6.4s}, [%5] \n" // r44 r45 r46

                        "fmla   v22.4s, v24.4s, v1.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v3.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v1.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v3.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v26.4s, v1.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v3.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v1.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v3.s[3]     \n"

                        "fmla   v22.4s, v16.4s, v2.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v4.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v2.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v4.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v22.4s, v18.4s, v2.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v4.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v2.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v4.s[3]     \n"

                        "fmla   v22.4s, v24.4s, v3.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v5.s[0]     \n"
                        "fmla   v20.4s, v25.4s, v3.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v5.s[1]     \n"

                        //                         "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6] \n"

                        "fmla   v22.4s, v26.4s, v3.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v5.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v3.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v5.s[3]     \n"

                        "fmla   v22.4s, v16.4s, v4.s[0]     \n"
                        "fmla   v23.4s, v16.4s, v6.s[0]     \n"
                        "fmla   v20.4s, v17.4s, v4.s[1]     \n"
                        "fmla   v21.4s, v17.4s, v6.s[1]     \n"
                        "fmla   v22.4s, v18.4s, v4.s[2]     \n"
                        "fmla   v23.4s, v18.4s, v6.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v4.s[3]     \n"
                        "fmla   v21.4s, v19.4s, v6.s[3]     \n"

                        "fadd   v20.4s, v20.4s, v22.4s      \n"
                        "fadd   v21.4s, v21.4s, v23.4s      \n"

                        "sub    %6, %6, #1536               \n" // kptr -= 24 * 16;

                        "st1    {v20.4s, v21.4s}, [%0], #32 \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2),      // %3
                        "=r"(r3),      // %4
                        "=r"(r4),      // %5
                        "=r"(kptr)     // %6
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "4"(r3),
                        "5"(r4),
                        "6"(kptr)
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27");
#else  // __aarch64__
                    asm volatile(
                        "pld        [%0, #256]          \n"
                        "vld1.f32   {d24-d27}, [%0 :128] \n" // sum0 sum1

                        "pld        [%1, #512]          \n"
                        "vldm       %1!, {d0-d7}        \n" // r00 r01 r02 r03

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmul.f32   q14, q8, d0[0]      \n"
                        "vmul.f32   q15, q8, d4[0]      \n"
                        "vmla.f32   q12, q9, d0[1]      \n"
                        "vmla.f32   q13, q9, d4[1]      \n"
                        "vmla.f32   q14, q10, d1[0]     \n"
                        "vmla.f32   q15, q10, d5[0]     \n"
                        "vmla.f32   q12, q11, d1[1]     \n"
                        "vmla.f32   q13, q11, d5[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%1, #384]          \n"
                        "vldm       %1, {d8-d13}        \n" // r04 r05 r06

                        "vmla.f32   q14, q8, d2[0]      \n"
                        "vmla.f32   q15, q8, d6[0]      \n"
                        "vmla.f32   q12, q9, d2[1]      \n"
                        "vmla.f32   q13, q9, d6[1]      \n"
                        "vmla.f32   q14, q10, d3[0]     \n"
                        "vmla.f32   q15, q10, d7[0]     \n"
                        "vmla.f32   q12, q11, d3[1]     \n"
                        "vmla.f32   q13, q11, d7[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q14, q8, d4[0]      \n"
                        "vmla.f32   q15, q8, d8[0]      \n"
                        "vmla.f32   q12, q9, d4[1]      \n"
                        "vmla.f32   q13, q9, d8[1]      \n"
                        "vmla.f32   q14, q10, d5[0]     \n"
                        "vmla.f32   q15, q10, d9[0]     \n"
                        "vmla.f32   q12, q11, d5[1]     \n"
                        "vmla.f32   q13, q11, d9[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q14, q8, d6[0]      \n"
                        "vmla.f32   q15, q8, d10[0]     \n"
                        "vmla.f32   q12, q9, d6[1]      \n"
                        "vmla.f32   q13, q9, d10[1]     \n"
                        "vmla.f32   q14, q10, d7[0]     \n"
                        "vmla.f32   q15, q10, d11[0]    \n"
                        "vmla.f32   q12, q11, d7[1]     \n"
                        "vmla.f32   q13, q11, d11[1]    \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%2, #512]          \n"
                        "vldm       %2!, {d0-d7}        \n" // r10 r11 r12 r13

                        "vmla.f32   q14, q8, d8[0]      \n"
                        "vmla.f32   q15, q8, d12[0]     \n"
                        "vmla.f32   q12, q9, d8[1]      \n"
                        "vmla.f32   q13, q9, d12[1]     \n"
                        "vmla.f32   q14, q10, d9[0]     \n"
                        "vmla.f32   q15, q10, d13[0]    \n"
                        "vmla.f32   q12, q11, d9[1]     \n"
                        "vmla.f32   q13, q11, d13[1]    \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q14, q8, d0[0]      \n"
                        "vmla.f32   q15, q8, d4[0]      \n"
                        "vmla.f32   q12, q9, d0[1]      \n"
                        "vmla.f32   q13, q9, d4[1]      \n"
                        "vmla.f32   q14, q10, d1[0]     \n"
                        "vmla.f32   q15, q10, d5[0]     \n"
                        "vmla.f32   q12, q11, d1[1]     \n"
                        "vmla.f32   q13, q11, d5[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%2, #384]          \n"
                        "vldm       %2, {d8-d13}        \n" // r14 r15 r16

                        "vmla.f32   q14, q8, d2[0]      \n"
                        "vmla.f32   q15, q8, d6[0]      \n"
                        "vmla.f32   q12, q9, d2[1]      \n"
                        "vmla.f32   q13, q9, d6[1]      \n"
                        "vmla.f32   q14, q10, d3[0]     \n"
                        "vmla.f32   q15, q10, d7[0]     \n"
                        "vmla.f32   q12, q11, d3[1]     \n"
                        "vmla.f32   q13, q11, d7[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q14, q8, d4[0]      \n"
                        "vmla.f32   q15, q8, d8[0]      \n"
                        "vmla.f32   q12, q9, d4[1]      \n"
                        "vmla.f32   q13, q9, d8[1]      \n"
                        "vmla.f32   q14, q10, d5[0]     \n"
                        "vmla.f32   q15, q10, d9[0]     \n"
                        "vmla.f32   q12, q11, d5[1]     \n"
                        "vmla.f32   q13, q11, d9[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q14, q8, d6[0]      \n"
                        "vmla.f32   q15, q8, d10[0]     \n"
                        "vmla.f32   q12, q9, d6[1]      \n"
                        "vmla.f32   q13, q9, d10[1]     \n"
                        "vmla.f32   q14, q10, d7[0]     \n"
                        "vmla.f32   q15, q10, d11[0]    \n"
                        "vmla.f32   q12, q11, d7[1]     \n"
                        "vmla.f32   q13, q11, d11[1]    \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%3, #512]          \n"
                        "vldm       %3!, {d0-d7}        \n" // r20 r21 r22 r23

                        "vmla.f32   q14, q8, d8[0]      \n"
                        "vmla.f32   q15, q8, d12[0]     \n"
                        "vmla.f32   q12, q9, d8[1]      \n"
                        "vmla.f32   q13, q9, d12[1]     \n"
                        "vmla.f32   q14, q10, d9[0]     \n"
                        "vmla.f32   q15, q10, d13[0]    \n"
                        "vmla.f32   q12, q11, d9[1]     \n"
                        "vmla.f32   q13, q11, d13[1]    \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q14, q8, d0[0]      \n"
                        "vmla.f32   q15, q8, d4[0]      \n"
                        "vmla.f32   q12, q9, d0[1]      \n"
                        "vmla.f32   q13, q9, d4[1]      \n"
                        "vmla.f32   q14, q10, d1[0]     \n"
                        "vmla.f32   q15, q10, d5[0]     \n"
                        "vmla.f32   q12, q11, d1[1]     \n"
                        "vmla.f32   q13, q11, d5[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%3, #384]          \n"
                        "vldm       %3, {d8-d13}        \n" // r24 r25 r26

                        "vmla.f32   q14, q8, d2[0]      \n"
                        "vmla.f32   q15, q8, d6[0]      \n"
                        "vmla.f32   q12, q9, d2[1]      \n"
                        "vmla.f32   q13, q9, d6[1]      \n"
                        "vmla.f32   q14, q10, d3[0]     \n"
                        "vmla.f32   q15, q10, d7[0]     \n"
                        "vmla.f32   q12, q11, d3[1]     \n"
                        "vmla.f32   q13, q11, d7[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q14, q8, d4[0]      \n"
                        "vmla.f32   q15, q8, d8[0]      \n"
                        "vmla.f32   q12, q9, d4[1]      \n"
                        "vmla.f32   q13, q9, d8[1]      \n"
                        "vmla.f32   q14, q10, d5[0]     \n"
                        "vmla.f32   q15, q10, d9[0]     \n"
                        "vmla.f32   q12, q11, d5[1]     \n"
                        "vmla.f32   q13, q11, d9[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q14, q8, d6[0]      \n"
                        "vmla.f32   q15, q8, d10[0]     \n"
                        "vmla.f32   q12, q9, d6[1]      \n"
                        "vmla.f32   q13, q9, d10[1]     \n"
                        "vmla.f32   q14, q10, d7[0]     \n"
                        "vmla.f32   q15, q10, d11[0]    \n"
                        "vmla.f32   q12, q11, d7[1]     \n"
                        "vmla.f32   q13, q11, d11[1]    \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%4, #512]          \n"
                        "vldm       %4!, {d0-d7}        \n" // r30 r31 r32 r33

                        "vmla.f32   q14, q8, d8[0]      \n"
                        "vmla.f32   q15, q8, d12[0]     \n"
                        "vmla.f32   q12, q9, d8[1]      \n"
                        "vmla.f32   q13, q9, d12[1]     \n"
                        "vmla.f32   q14, q10, d9[0]     \n"
                        "vmla.f32   q15, q10, d13[0]    \n"
                        "vmla.f32   q12, q11, d9[1]     \n"
                        "vmla.f32   q13, q11, d13[1]    \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q14, q8, d0[0]      \n"
                        "vmla.f32   q15, q8, d4[0]      \n"
                        "vmla.f32   q12, q9, d0[1]      \n"
                        "vmla.f32   q13, q9, d4[1]      \n"
                        "vmla.f32   q14, q10, d1[0]     \n"
                        "vmla.f32   q15, q10, d5[0]     \n"
                        "vmla.f32   q12, q11, d1[1]     \n"
                        "vmla.f32   q13, q11, d5[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%4, #384]          \n"
                        "vldm       %4, {d8-d13}        \n" // r34 r35 r36

                        "vmla.f32   q14, q8, d2[0]      \n"
                        "vmla.f32   q15, q8, d6[0]      \n"
                        "vmla.f32   q12, q9, d2[1]      \n"
                        "vmla.f32   q13, q9, d6[1]      \n"
                        "vmla.f32   q14, q10, d3[0]     \n"
                        "vmla.f32   q15, q10, d7[0]     \n"
                        "vmla.f32   q12, q11, d3[1]     \n"
                        "vmla.f32   q13, q11, d7[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q14, q8, d4[0]      \n"
                        "vmla.f32   q15, q8, d8[0]      \n"
                        "vmla.f32   q12, q9, d4[1]      \n"
                        "vmla.f32   q13, q9, d8[1]      \n"
                        "vmla.f32   q14, q10, d5[0]     \n"
                        "vmla.f32   q15, q10, d9[0]     \n"
                        "vmla.f32   q12, q11, d5[1]     \n"
                        "vmla.f32   q13, q11, d9[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q14, q8, d6[0]      \n"
                        "vmla.f32   q15, q8, d10[0]     \n"
                        "vmla.f32   q12, q9, d6[1]      \n"
                        "vmla.f32   q13, q9, d10[1]     \n"
                        "vmla.f32   q14, q10, d7[0]     \n"
                        "vmla.f32   q15, q10, d11[0]    \n"
                        "vmla.f32   q12, q11, d7[1]     \n"
                        "vmla.f32   q13, q11, d11[1]    \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%5, #512]          \n"
                        "vldm       %5!, {d0-d7}        \n" // r40 r41 r42 r43

                        "vmla.f32   q14, q8, d8[0]      \n"
                        "vmla.f32   q15, q8, d12[0]     \n"
                        "vmla.f32   q12, q9, d8[1]      \n"
                        "vmla.f32   q13, q9, d12[1]     \n"
                        "vmla.f32   q14, q10, d9[0]     \n"
                        "vmla.f32   q15, q10, d13[0]    \n"
                        "vmla.f32   q12, q11, d9[1]     \n"
                        "vmla.f32   q13, q11, d13[1]    \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q14, q8, d0[0]      \n"
                        "vmla.f32   q15, q8, d4[0]      \n"
                        "vmla.f32   q12, q9, d0[1]      \n"
                        "vmla.f32   q13, q9, d4[1]      \n"
                        "vmla.f32   q14, q10, d1[0]     \n"
                        "vmla.f32   q15, q10, d5[0]     \n"
                        "vmla.f32   q12, q11, d1[1]     \n"
                        "vmla.f32   q13, q11, d5[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%5, #384]          \n"
                        "vldm       %5, {d8-d13}        \n" // r44 r45 r46

                        "vmla.f32   q14, q8, d2[0]      \n"
                        "vmla.f32   q15, q8, d6[0]      \n"
                        "vmla.f32   q12, q9, d2[1]      \n"
                        "vmla.f32   q13, q9, d6[1]      \n"
                        "vmla.f32   q14, q10, d3[0]     \n"
                        "vmla.f32   q15, q10, d7[0]     \n"
                        "vmla.f32   q12, q11, d3[1]     \n"
                        "vmla.f32   q13, q11, d7[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q14, q8, d4[0]      \n"
                        "vmla.f32   q15, q8, d8[0]      \n"
                        "vmla.f32   q12, q9, d4[1]      \n"
                        "vmla.f32   q13, q9, d8[1]      \n"
                        "vmla.f32   q14, q10, d5[0]     \n"
                        "vmla.f32   q15, q10, d9[0]     \n"
                        "vmla.f32   q12, q11, d5[1]     \n"
                        "vmla.f32   q13, q11, d9[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q14, q8, d6[0]      \n"
                        "vmla.f32   q15, q8, d10[0]     \n"
                        "vmla.f32   q12, q9, d6[1]      \n"
                        "vmla.f32   q13, q9, d10[1]     \n"
                        "vmla.f32   q14, q10, d7[0]     \n"
                        "vmla.f32   q15, q10, d11[0]    \n"
                        "vmla.f32   q12, q11, d7[1]     \n"
                        "vmla.f32   q13, q11, d11[1]    \n"

                        //                         "pld        [%6, #512]          \n"
                        "vldm       %6, {d16-d23}      \n"

                        "vmla.f32   q14, q8, d8[0]      \n"
                        "vmla.f32   q15, q8, d12[0]     \n"
                        "vmla.f32   q12, q9, d8[1]      \n"
                        "vmla.f32   q13, q9, d12[1]     \n"
                        "vmla.f32   q14, q10, d9[0]     \n"
                        "vmla.f32   q15, q10, d13[0]    \n"
                        "vmla.f32   q12, q11, d9[1]     \n"
                        "vmla.f32   q13, q11, d13[1]    \n"

                        "vadd.f32   q12, q12, q14       \n"
                        "vadd.f32   q13, q13, q15       \n"

                        "sub        %6, %6, #1536       \n" // kptr -= 24 * 16;

                        "vst1.f32   {d24-d27}, [%0 :128]! \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2),      // %3
                        "=r"(r3),      // %4
                        "=r"(r4),      // %5
                        "=r"(kptr)     // %6
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "4"(r3),
                        "5"(r4),
                        "6"(kptr)
                        : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif // __aarch64__
                }
                for (; j < outw; j++)
                {
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%0, #128]       \n"
                        "ld1    {v20.4s}, [%0]              \n" // sum0

                        "prfm   pldl1keep, [%1, #256]       \n"
                        "ld1    {v0.4s, v1.4s}, [%1], #32   \n" // r00 r01

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmul   v21.4s, v16.4s, v0.s[0]     \n"
                        "fmul   v22.4s, v17.4s, v0.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmul   v23.4s, v18.4s, v0.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v0.s[3]     \n"

                        "prfm   pldl1keep, [%1, #384]       \n"
                        "ld1    {v2.4s, v3.4s, v4.4s}, [%1] \n" // r02 r03 r04

                        "fmla   v21.4s, v24.4s, v1.s[0]     \n"
                        "fmla   v22.4s, v25.4s, v1.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v23.4s, v26.4s, v1.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v1.s[3]     \n"

                        "fmla   v21.4s, v16.4s, v2.s[0]     \n"
                        "fmla   v22.4s, v17.4s, v2.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v23.4s, v18.4s, v2.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v2.s[3]     \n"

                        "fmla   v21.4s, v24.4s, v3.s[0]     \n"
                        "fmla   v22.4s, v25.4s, v3.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v23.4s, v26.4s, v3.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v3.s[3]     \n"

                        "prfm   pldl1keep, [%2, #256]       \n"
                        "ld1    {v0.4s, v1.4s}, [%2], #32   \n" // r10 r11

                        "fmla   v21.4s, v16.4s, v4.s[0]     \n"
                        "fmla   v22.4s, v17.4s, v4.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v23.4s, v18.4s, v4.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v4.s[3]     \n"

                        "fmla   v21.4s, v24.4s, v0.s[0]     \n"
                        "fmla   v22.4s, v25.4s, v0.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v23.4s, v26.4s, v0.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v0.s[3]     \n"

                        "prfm   pldl1keep, [%2, #384]       \n"
                        "ld1    {v2.4s, v3.4s, v4.4s}, [%2] \n" // r12 r13 r14

                        "fmla   v21.4s, v16.4s, v1.s[0]     \n"
                        "fmla   v22.4s, v17.4s, v1.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v23.4s, v18.4s, v1.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v1.s[3]     \n"

                        "fmla   v21.4s, v24.4s, v2.s[0]     \n"
                        "fmla   v22.4s, v25.4s, v2.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v23.4s, v26.4s, v2.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v2.s[3]     \n"

                        "fmla   v21.4s, v16.4s, v3.s[0]     \n"
                        "fmla   v22.4s, v17.4s, v3.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v23.4s, v18.4s, v3.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v3.s[3]     \n"

                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld1    {v0.4s, v1.4s}, [%3], #32   \n" // r20 r21

                        "fmla   v21.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v22.4s, v25.4s, v4.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v23.4s, v26.4s, v4.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v4.s[3]     \n"

                        "fmla   v21.4s, v16.4s, v0.s[0]     \n"
                        "fmla   v22.4s, v17.4s, v0.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v23.4s, v18.4s, v0.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v0.s[3]     \n"

                        "prfm   pldl1keep, [%3, #384]       \n"
                        "ld1    {v2.4s, v3.4s, v4.4s}, [%3] \n" // r22 r23 r24

                        "fmla   v21.4s, v24.4s, v1.s[0]     \n"
                        "fmla   v22.4s, v25.4s, v1.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v23.4s, v26.4s, v1.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v1.s[3]     \n"

                        "fmla   v21.4s, v16.4s, v2.s[0]     \n"
                        "fmla   v22.4s, v17.4s, v2.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v23.4s, v18.4s, v2.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v2.s[3]     \n"

                        "fmla   v21.4s, v24.4s, v3.s[0]     \n"
                        "fmla   v22.4s, v25.4s, v3.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v23.4s, v26.4s, v3.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v3.s[3]     \n"

                        "prfm   pldl1keep, [%4, #256]       \n"
                        "ld1    {v0.4s, v1.4s}, [%4], #32   \n" // r30 r31

                        "fmla   v21.4s, v16.4s, v4.s[0]     \n"
                        "fmla   v22.4s, v17.4s, v4.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v23.4s, v18.4s, v4.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v4.s[3]     \n"

                        "fmla   v21.4s, v24.4s, v0.s[0]     \n"
                        "fmla   v22.4s, v25.4s, v0.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v23.4s, v26.4s, v0.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v0.s[3]     \n"

                        "prfm   pldl1keep, [%4, #384]       \n"
                        "ld1    {v2.4s, v3.4s, v4.4s}, [%4] \n" // r32 r33 r34

                        "fmla   v21.4s, v16.4s, v1.s[0]     \n"
                        "fmla   v22.4s, v17.4s, v1.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v23.4s, v18.4s, v1.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v1.s[3]     \n"

                        "fmla   v21.4s, v24.4s, v2.s[0]     \n"
                        "fmla   v22.4s, v25.4s, v2.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v23.4s, v26.4s, v2.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v2.s[3]     \n"

                        "fmla   v21.4s, v16.4s, v3.s[0]     \n"
                        "fmla   v22.4s, v17.4s, v3.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v23.4s, v18.4s, v3.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v3.s[3]     \n"

                        "prfm   pldl1keep, [%5, #256]       \n"
                        "ld1    {v0.4s, v1.4s}, [%5], #32   \n" // r40 r41

                        "fmla   v21.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v22.4s, v25.4s, v4.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v23.4s, v26.4s, v4.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v4.s[3]     \n"

                        "fmla   v21.4s, v16.4s, v0.s[0]     \n"
                        "fmla   v22.4s, v17.4s, v0.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v23.4s, v18.4s, v0.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v0.s[3]     \n"

                        "prfm   pldl1keep, [%5, #384]       \n"
                        "ld1    {v2.4s, v3.4s, v4.4s}, [%5] \n" // r42 r43 r44

                        "fmla   v21.4s, v24.4s, v1.s[0]     \n"
                        "fmla   v22.4s, v25.4s, v1.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6], #64 \n"

                        "fmla   v23.4s, v26.4s, v1.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v1.s[3]     \n"

                        "fmla   v21.4s, v16.4s, v2.s[0]     \n"
                        "fmla   v22.4s, v17.4s, v2.s[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%6], #64 \n"

                        "fmla   v23.4s, v18.4s, v2.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v2.s[3]     \n"

                        "fmla   v21.4s, v24.4s, v3.s[0]     \n"
                        "fmla   v22.4s, v25.4s, v3.s[1]     \n"

                        //                         "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%6] \n"

                        "fmla   v23.4s, v26.4s, v3.s[2]     \n"
                        "fmla   v20.4s, v27.4s, v3.s[3]     \n"

                        "fmla   v21.4s, v16.4s, v4.s[0]     \n"
                        "fmla   v22.4s, v17.4s, v4.s[1]     \n"
                        "fmla   v23.4s, v18.4s, v4.s[2]     \n"
                        "fmla   v20.4s, v19.4s, v4.s[3]     \n"

                        "fadd   v22.4s, v21.4s, v22.4s      \n"
                        "fadd   v23.4s, v22.4s, v23.4s      \n"
                        "fadd   v20.4s, v20.4s, v23.4s      \n"

                        "sub    %6, %6, #1536               \n" // kptr -= 24 * 16;

                        "st1    {v20.4s}, [%0], #16         \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2),      // %3
                        "=r"(r3),      // %4
                        "=r"(r4),      // %5
                        "=r"(kptr)     // %6
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "4"(r3),
                        "5"(r4),
                        "6"(kptr)
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27");
#else  // __aarch64__
                    asm volatile(
                        "pld        [%0, #128]          \n"
                        "vld1.f32   {d24-d25}, [%0 :128] \n" // sum0

                        "pld        [%1, #256]          \n"
                        "vld1.f32   {d0-d3}, [%1 :128]! \n" // r00 r01

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmul.f32   q13, q8, d0[0]      \n"
                        "vmul.f32   q14, q9, d0[1]      \n"
                        "vmul.f32   q15, q10, d1[0]     \n"
                        "vmla.f32   q12, q11, d1[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%1, #384]          \n"
                        "vldm       %1, {d4-d9}         \n" // r02 r03 r04

                        "vmla.f32   q13, q8, d2[0]      \n"
                        "vmla.f32   q14, q9, d2[1]      \n"
                        "vmla.f32   q15, q10, d3[0]     \n"
                        "vmla.f32   q12, q11, d3[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q13, q8, d4[0]      \n"
                        "vmla.f32   q14, q9, d4[1]      \n"
                        "vmla.f32   q15, q10, d5[0]     \n"
                        "vmla.f32   q12, q11, d5[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q13, q8, d6[0]      \n"
                        "vmla.f32   q14, q9, d6[1]      \n"
                        "vmla.f32   q15, q10, d7[0]     \n"
                        "vmla.f32   q12, q11, d7[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%2, #256]          \n"
                        "vld1.f32   {d0-d3}, [%2 :128]! \n" // r10 r11

                        "vmla.f32   q13, q8, d8[0]      \n"
                        "vmla.f32   q14, q9, d8[1]      \n"
                        "vmla.f32   q15, q10, d9[0]     \n"
                        "vmla.f32   q12, q11, d9[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q13, q8, d0[0]      \n"
                        "vmla.f32   q14, q9, d0[1]      \n"
                        "vmla.f32   q15, q10, d1[0]     \n"
                        "vmla.f32   q12, q11, d1[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%2, #384]          \n"
                        "vldm       %2, {d4-d9}         \n" // r12 r13 r14

                        "vmla.f32   q13, q8, d2[0]      \n"
                        "vmla.f32   q14, q9, d2[1]      \n"
                        "vmla.f32   q15, q10, d3[0]     \n"
                        "vmla.f32   q12, q11, d3[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q13, q8, d4[0]      \n"
                        "vmla.f32   q14, q9, d4[1]      \n"
                        "vmla.f32   q15, q10, d5[0]     \n"
                        "vmla.f32   q12, q11, d5[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q13, q8, d6[0]      \n"
                        "vmla.f32   q14, q9, d6[1]      \n"
                        "vmla.f32   q15, q10, d7[0]     \n"
                        "vmla.f32   q12, q11, d7[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%3, #256]          \n"
                        "vld1.f32   {d0-d3}, [%3 :128]! \n" // r20 r21

                        "vmla.f32   q13, q8, d8[0]      \n"
                        "vmla.f32   q14, q9, d8[1]      \n"
                        "vmla.f32   q15, q10, d9[0]     \n"
                        "vmla.f32   q12, q11, d9[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q13, q8, d0[0]      \n"
                        "vmla.f32   q14, q9, d0[1]      \n"
                        "vmla.f32   q15, q10, d1[0]     \n"
                        "vmla.f32   q12, q11, d1[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%3, #384]          \n"
                        "vldm       %3, {d4-d9}         \n" // r22 r23 r24

                        "vmla.f32   q13, q8, d2[0]      \n"
                        "vmla.f32   q14, q9, d2[1]      \n"
                        "vmla.f32   q15, q10, d3[0]     \n"
                        "vmla.f32   q12, q11, d3[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q13, q8, d4[0]      \n"
                        "vmla.f32   q14, q9, d4[1]      \n"
                        "vmla.f32   q15, q10, d5[0]     \n"
                        "vmla.f32   q12, q11, d5[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q13, q8, d6[0]      \n"
                        "vmla.f32   q14, q9, d6[1]      \n"
                        "vmla.f32   q15, q10, d7[0]     \n"
                        "vmla.f32   q12, q11, d7[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%4, #256]          \n"
                        "vld1.f32   {d0-d3}, [%4 :128]! \n" // r30 r31

                        "vmla.f32   q13, q8, d8[0]      \n"
                        "vmla.f32   q14, q9, d8[1]      \n"
                        "vmla.f32   q15, q10, d9[0]     \n"
                        "vmla.f32   q12, q11, d9[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q13, q8, d0[0]      \n"
                        "vmla.f32   q14, q9, d0[1]      \n"
                        "vmla.f32   q15, q10, d1[0]     \n"
                        "vmla.f32   q12, q11, d1[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%4, #384]          \n"
                        "vldm       %4, {d4-d9}         \n" // r32 r33 r34

                        "vmla.f32   q13, q8, d2[0]      \n"
                        "vmla.f32   q14, q9, d2[1]      \n"
                        "vmla.f32   q15, q10, d3[0]     \n"
                        "vmla.f32   q12, q11, d3[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q13, q8, d4[0]      \n"
                        "vmla.f32   q14, q9, d4[1]      \n"
                        "vmla.f32   q15, q10, d5[0]     \n"
                        "vmla.f32   q12, q11, d5[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q13, q8, d6[0]      \n"
                        "vmla.f32   q14, q9, d6[1]      \n"
                        "vmla.f32   q15, q10, d7[0]     \n"
                        "vmla.f32   q12, q11, d7[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%5, #256]          \n"
                        "vld1.f32   {d0-d3}, [%5 :128]! \n" // r40 r41

                        "vmla.f32   q13, q8, d8[0]      \n"
                        "vmla.f32   q14, q9, d8[1]      \n"
                        "vmla.f32   q15, q10, d9[0]     \n"
                        "vmla.f32   q12, q11, d9[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q13, q8, d0[0]      \n"
                        "vmla.f32   q14, q9, d0[1]      \n"
                        "vmla.f32   q15, q10, d1[0]     \n"
                        "vmla.f32   q12, q11, d1[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "pld        [%5, #384]          \n"
                        "vldm       %5, {d4-d9}         \n" // r42 r43 r44

                        "vmla.f32   q13, q8, d2[0]      \n"
                        "vmla.f32   q14, q9, d2[1]      \n"
                        "vmla.f32   q15, q10, d3[0]     \n"
                        "vmla.f32   q12, q11, d3[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q13, q8, d4[0]      \n"
                        "vmla.f32   q14, q9, d4[1]      \n"
                        "vmla.f32   q15, q10, d5[0]     \n"
                        "vmla.f32   q12, q11, d5[1]     \n"

                        "pld        [%6, #512]          \n"
                        "vldm       %6!, {d16-d23}      \n"

                        "vmla.f32   q13, q8, d6[0]      \n"
                        "vmla.f32   q14, q9, d6[1]      \n"
                        "vmla.f32   q15, q10, d7[0]     \n"
                        "vmla.f32   q12, q11, d7[1]     \n"

                        //                         "pld        [%6, #512]          \n"
                        "vldm       %6, {d16-d23}      \n"

                        "vmla.f32   q13, q8, d8[0]      \n"
                        "vmla.f32   q14, q9, d8[1]      \n"
                        "vmla.f32   q15, q10, d9[0]     \n"
                        "vmla.f32   q12, q11, d9[1]     \n"

                        "vadd.f32   q14, q13, q14       \n"
                        "vadd.f32   q15, q14, q15       \n"
                        "vadd.f32   q12, q12, q15       \n"

                        "sub        %6, %6, #1536       \n" // kptr -= 24 * 16;

                        "vst1.f32   {d24-d25}, [%0 :128]! \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2),      // %3
                        "=r"(r3),      // %4
                        "=r"(r4),      // %5
                        "=r"(kptr)     // %6
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "4"(r3),
                        "5"(r4),
                        "6"(kptr)
                        : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif // __aarch64__
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
                r3 += tailstep;
                r4 += tailstep;
            }
        }
    }
}
