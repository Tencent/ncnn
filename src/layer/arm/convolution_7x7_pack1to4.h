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

static void conv7x7s2_pack1to4_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2 * outw + w;

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
            const float* r5 = img0.row(5);
            const float* r6 = img0.row(6);

            const float* kptr = (const float*)kernel.channel(p).row(q);

            int i = 0;

            for (; i < outh; i++)
            {
                int j = 0;
#if __aarch64__
                for (; j + 7 < outw; j += 8)
                {
                    asm volatile(
                        "prfm   pldl1keep, [%0, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"

                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%1], #64 \n" // r0

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%8], #64 \n"

                        "fmla   v16.4s, v24.4s, v0.s[0]     \n"
                        "fmla   v17.4s, v24.4s, v0.s[2]     \n"
                        "fmla   v18.4s, v24.4s, v1.s[0]     \n"
                        "fmla   v19.4s, v24.4s, v1.s[2]     \n"

                        "prfm   pldl1keep, [%0, #512]       \n"
                        "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0] \n"

                        "fmla   v20.4s, v24.4s, v2.s[0]     \n"
                        "fmla   v21.4s, v24.4s, v2.s[2]     \n"
                        "fmla   v22.4s, v24.4s, v3.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v3.s[2]     \n"

                        "prfm   pldl1keep, [%1, #256]       \n"
                        "ld1    {v4.4s, v5.4s}, [%1]        \n"

                        "fmla   v16.4s, v25.4s, v0.s[1]     \n"
                        "fmla   v17.4s, v25.4s, v0.s[3]     \n"
                        "fmla   v18.4s, v25.4s, v1.s[1]     \n"
                        "fmla   v19.4s, v25.4s, v1.s[3]     \n"
                        "fmla   v20.4s, v25.4s, v2.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v2.s[3]     \n"
                        "fmla   v22.4s, v25.4s, v3.s[1]     \n"
                        "fmla   v23.4s, v25.4s, v3.s[3]     \n"

                        "prfm   pldl1keep, [%8, #384]       \n"
                        "ld1    {v28.4s, v29.4s, v30.4s}, [%8], #48 \n"

                        "fmla   v16.4s, v26.4s, v0.s[2]     \n"
                        "fmla   v17.4s, v26.4s, v1.s[0]     \n"
                        "fmla   v18.4s, v26.4s, v1.s[2]     \n"
                        "fmla   v19.4s, v26.4s, v2.s[0]     \n"
                        "fmla   v20.4s, v26.4s, v2.s[2]     \n"
                        "fmla   v21.4s, v26.4s, v3.s[0]     \n"
                        "fmla   v22.4s, v26.4s, v3.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v4.s[0]     \n"

                        "fmla   v16.4s, v27.4s, v0.s[3]     \n"
                        "fmla   v17.4s, v27.4s, v1.s[1]     \n"
                        "fmla   v18.4s, v27.4s, v1.s[3]     \n"
                        "fmla   v19.4s, v27.4s, v2.s[1]     \n"
                        "fmla   v20.4s, v27.4s, v2.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v3.s[1]     \n"
                        "fmla   v22.4s, v27.4s, v3.s[3]     \n"
                        "fmla   v23.4s, v27.4s, v4.s[1]     \n"

                        "fmla   v16.4s, v28.4s, v1.s[0]     \n"
                        "fmla   v17.4s, v28.4s, v1.s[2]     \n"
                        "fmla   v18.4s, v28.4s, v2.s[0]     \n"
                        "fmla   v19.4s, v28.4s, v2.s[2]     \n"
                        "fmla   v20.4s, v28.4s, v3.s[0]     \n"
                        "fmla   v21.4s, v28.4s, v3.s[2]     \n"
                        "fmla   v22.4s, v28.4s, v4.s[0]     \n"
                        "fmla   v23.4s, v28.4s, v4.s[2]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%8], #64 \n"

                        "fmla   v16.4s, v29.4s, v1.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v1.s[3]     \n"
                        "fmla   v18.4s, v29.4s, v2.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v2.s[3]     \n"
                        "fmla   v20.4s, v29.4s, v3.s[1]     \n"
                        "fmla   v21.4s, v29.4s, v3.s[3]     \n"
                        "fmla   v22.4s, v29.4s, v4.s[1]     \n"
                        "fmla   v23.4s, v29.4s, v4.s[3]     \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v6.4s, v7.4s, v8.4s, v9.4s}, [%2], #64 \n" // r1

                        "fmla   v16.4s, v30.4s, v1.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v2.s[0]     \n"
                        "fmla   v18.4s, v30.4s, v2.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v3.s[0]     \n"
                        "fmla   v20.4s, v30.4s, v3.s[2]     \n"
                        "fmla   v21.4s, v30.4s, v4.s[0]     \n"
                        "fmla   v22.4s, v30.4s, v4.s[2]     \n"
                        "fmla   v23.4s, v30.4s, v5.s[0]     \n"

                        "prfm   pldl1keep, [%2, #256]       \n"
                        "ld1    {v10.4s, v11.4s}, [%2]      \n"

                        "fmla   v16.4s, v24.4s, v6.s[0]     \n"
                        "fmla   v17.4s, v24.4s, v6.s[2]     \n"
                        "fmla   v18.4s, v24.4s, v7.s[0]     \n"
                        "fmla   v19.4s, v24.4s, v7.s[2]     \n"
                        "fmla   v20.4s, v24.4s, v8.s[0]     \n"
                        "fmla   v21.4s, v24.4s, v8.s[2]     \n"
                        "fmla   v22.4s, v24.4s, v9.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v9.s[2]     \n"

                        "fmla   v16.4s, v25.4s, v6.s[1]     \n"
                        "fmla   v17.4s, v25.4s, v6.s[3]     \n"
                        "fmla   v18.4s, v25.4s, v7.s[1]     \n"
                        "fmla   v19.4s, v25.4s, v7.s[3]     \n"
                        "fmla   v20.4s, v25.4s, v8.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v8.s[3]     \n"
                        "fmla   v22.4s, v25.4s, v9.s[1]     \n"
                        "fmla   v23.4s, v25.4s, v9.s[3]     \n"

                        "prfm   pldl1keep, [%8, #384]       \n"
                        "ld1    {v28.4s, v29.4s, v30.4s}, [%8], #48 \n"

                        "fmla   v16.4s, v26.4s, v6.s[2]     \n"
                        "fmla   v17.4s, v26.4s, v7.s[0]     \n"
                        "fmla   v18.4s, v26.4s, v7.s[2]     \n"
                        "fmla   v19.4s, v26.4s, v8.s[0]     \n"
                        "fmla   v20.4s, v26.4s, v8.s[2]     \n"
                        "fmla   v21.4s, v26.4s, v9.s[0]     \n"
                        "fmla   v22.4s, v26.4s, v9.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v10.s[0]    \n"

                        "fmla   v16.4s, v27.4s, v6.s[3]     \n"
                        "fmla   v17.4s, v27.4s, v7.s[1]     \n"
                        "fmla   v18.4s, v27.4s, v7.s[3]     \n"
                        "fmla   v19.4s, v27.4s, v8.s[1]     \n"
                        "fmla   v20.4s, v27.4s, v8.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v9.s[1]     \n"
                        "fmla   v22.4s, v27.4s, v9.s[3]     \n"
                        "fmla   v23.4s, v27.4s, v10.s[1]    \n"

                        "fmla   v16.4s, v28.4s, v7.s[0]     \n"
                        "fmla   v17.4s, v28.4s, v7.s[2]     \n"
                        "fmla   v18.4s, v28.4s, v8.s[0]     \n"
                        "fmla   v19.4s, v28.4s, v8.s[2]     \n"
                        "fmla   v20.4s, v28.4s, v9.s[0]     \n"
                        "fmla   v21.4s, v28.4s, v9.s[2]     \n"
                        "fmla   v22.4s, v28.4s, v10.s[0]    \n"
                        "fmla   v23.4s, v28.4s, v10.s[2]    \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%8], #64 \n"

                        "fmla   v16.4s, v29.4s, v7.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v7.s[3]     \n"
                        "fmla   v18.4s, v29.4s, v8.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v8.s[3]     \n"
                        "fmla   v20.4s, v29.4s, v9.s[1]     \n"
                        "fmla   v21.4s, v29.4s, v9.s[3]     \n"
                        "fmla   v22.4s, v29.4s, v10.s[1]    \n"
                        "fmla   v23.4s, v29.4s, v10.s[3]    \n"

                        "prfm   pldl1keep, [%3, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%3], #64 \n" // r2

                        "fmla   v16.4s, v30.4s, v7.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v8.s[0]     \n"
                        "fmla   v18.4s, v30.4s, v8.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v9.s[0]     \n"
                        "fmla   v20.4s, v30.4s, v9.s[2]     \n"
                        "fmla   v21.4s, v30.4s, v10.s[0]    \n"
                        "fmla   v22.4s, v30.4s, v10.s[2]    \n"
                        "fmla   v23.4s, v30.4s, v11.s[0]    \n"

                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld1    {v4.4s, v5.4s}, [%3]        \n"

                        "fmla   v16.4s, v24.4s, v0.s[0]     \n"
                        "fmla   v17.4s, v24.4s, v0.s[2]     \n"
                        "fmla   v18.4s, v24.4s, v1.s[0]     \n"
                        "fmla   v19.4s, v24.4s, v1.s[2]     \n"
                        "fmla   v20.4s, v24.4s, v2.s[0]     \n"
                        "fmla   v21.4s, v24.4s, v2.s[2]     \n"
                        "fmla   v22.4s, v24.4s, v3.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v3.s[2]     \n"

                        "fmla   v16.4s, v25.4s, v0.s[1]     \n"
                        "fmla   v17.4s, v25.4s, v0.s[3]     \n"
                        "fmla   v18.4s, v25.4s, v1.s[1]     \n"
                        "fmla   v19.4s, v25.4s, v1.s[3]     \n"
                        "fmla   v20.4s, v25.4s, v2.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v2.s[3]     \n"
                        "fmla   v22.4s, v25.4s, v3.s[1]     \n"
                        "fmla   v23.4s, v25.4s, v3.s[3]     \n"

                        "prfm   pldl1keep, [%8, #384]       \n"
                        "ld1    {v28.4s, v29.4s, v30.4s}, [%8], #48 \n"

                        "fmla   v16.4s, v26.4s, v0.s[2]     \n"
                        "fmla   v17.4s, v26.4s, v1.s[0]     \n"
                        "fmla   v18.4s, v26.4s, v1.s[2]     \n"
                        "fmla   v19.4s, v26.4s, v2.s[0]     \n"
                        "fmla   v20.4s, v26.4s, v2.s[2]     \n"
                        "fmla   v21.4s, v26.4s, v3.s[0]     \n"
                        "fmla   v22.4s, v26.4s, v3.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v4.s[0]     \n"

                        "fmla   v16.4s, v27.4s, v0.s[3]     \n"
                        "fmla   v17.4s, v27.4s, v1.s[1]     \n"
                        "fmla   v18.4s, v27.4s, v1.s[3]     \n"
                        "fmla   v19.4s, v27.4s, v2.s[1]     \n"
                        "fmla   v20.4s, v27.4s, v2.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v3.s[1]     \n"
                        "fmla   v22.4s, v27.4s, v3.s[3]     \n"
                        "fmla   v23.4s, v27.4s, v4.s[1]     \n"

                        "fmla   v16.4s, v28.4s, v1.s[0]     \n"
                        "fmla   v17.4s, v28.4s, v1.s[2]     \n"
                        "fmla   v18.4s, v28.4s, v2.s[0]     \n"
                        "fmla   v19.4s, v28.4s, v2.s[2]     \n"
                        "fmla   v20.4s, v28.4s, v3.s[0]     \n"
                        "fmla   v21.4s, v28.4s, v3.s[2]     \n"
                        "fmla   v22.4s, v28.4s, v4.s[0]     \n"
                        "fmla   v23.4s, v28.4s, v4.s[2]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%8], #64 \n"

                        "fmla   v16.4s, v29.4s, v1.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v1.s[3]     \n"
                        "fmla   v18.4s, v29.4s, v2.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v2.s[3]     \n"
                        "fmla   v20.4s, v29.4s, v3.s[1]     \n"
                        "fmla   v21.4s, v29.4s, v3.s[3]     \n"
                        "fmla   v22.4s, v29.4s, v4.s[1]     \n"
                        "fmla   v23.4s, v29.4s, v4.s[3]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v6.4s, v7.4s, v8.4s, v9.4s}, [%4], #64 \n" // r3

                        "fmla   v16.4s, v30.4s, v1.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v2.s[0]     \n"
                        "fmla   v18.4s, v30.4s, v2.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v3.s[0]     \n"
                        "fmla   v20.4s, v30.4s, v3.s[2]     \n"
                        "fmla   v21.4s, v30.4s, v4.s[0]     \n"
                        "fmla   v22.4s, v30.4s, v4.s[2]     \n"
                        "fmla   v23.4s, v30.4s, v5.s[0]     \n"

                        "prfm   pldl1keep, [%4, #256]       \n"
                        "ld1    {v10.4s, v11.4s}, [%4]      \n"

                        "fmla   v16.4s, v24.4s, v6.s[0]     \n"
                        "fmla   v17.4s, v24.4s, v6.s[2]     \n"
                        "fmla   v18.4s, v24.4s, v7.s[0]     \n"
                        "fmla   v19.4s, v24.4s, v7.s[2]     \n"
                        "fmla   v20.4s, v24.4s, v8.s[0]     \n"
                        "fmla   v21.4s, v24.4s, v8.s[2]     \n"
                        "fmla   v22.4s, v24.4s, v9.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v9.s[2]     \n"

                        "fmla   v16.4s, v25.4s, v6.s[1]     \n"
                        "fmla   v17.4s, v25.4s, v6.s[3]     \n"
                        "fmla   v18.4s, v25.4s, v7.s[1]     \n"
                        "fmla   v19.4s, v25.4s, v7.s[3]     \n"
                        "fmla   v20.4s, v25.4s, v8.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v8.s[3]     \n"
                        "fmla   v22.4s, v25.4s, v9.s[1]     \n"
                        "fmla   v23.4s, v25.4s, v9.s[3]     \n"

                        "prfm   pldl1keep, [%8, #384]       \n"
                        "ld1    {v28.4s, v29.4s, v30.4s}, [%8], #48 \n"

                        "fmla   v16.4s, v26.4s, v6.s[2]     \n"
                        "fmla   v17.4s, v26.4s, v7.s[0]     \n"
                        "fmla   v18.4s, v26.4s, v7.s[2]     \n"
                        "fmla   v19.4s, v26.4s, v8.s[0]     \n"
                        "fmla   v20.4s, v26.4s, v8.s[2]     \n"
                        "fmla   v21.4s, v26.4s, v9.s[0]     \n"
                        "fmla   v22.4s, v26.4s, v9.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v10.s[0]    \n"

                        "fmla   v16.4s, v27.4s, v6.s[3]     \n"
                        "fmla   v17.4s, v27.4s, v7.s[1]     \n"
                        "fmla   v18.4s, v27.4s, v7.s[3]     \n"
                        "fmla   v19.4s, v27.4s, v8.s[1]     \n"
                        "fmla   v20.4s, v27.4s, v8.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v9.s[1]     \n"
                        "fmla   v22.4s, v27.4s, v9.s[3]     \n"
                        "fmla   v23.4s, v27.4s, v10.s[1]    \n"

                        "fmla   v16.4s, v28.4s, v7.s[0]     \n"
                        "fmla   v17.4s, v28.4s, v7.s[2]     \n"
                        "fmla   v18.4s, v28.4s, v8.s[0]     \n"
                        "fmla   v19.4s, v28.4s, v8.s[2]     \n"
                        "fmla   v20.4s, v28.4s, v9.s[0]     \n"
                        "fmla   v21.4s, v28.4s, v9.s[2]     \n"
                        "fmla   v22.4s, v28.4s, v10.s[0]    \n"
                        "fmla   v23.4s, v28.4s, v10.s[2]    \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%8], #64 \n"

                        "fmla   v16.4s, v29.4s, v7.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v7.s[3]     \n"
                        "fmla   v18.4s, v29.4s, v8.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v8.s[3]     \n"
                        "fmla   v20.4s, v29.4s, v9.s[1]     \n"
                        "fmla   v21.4s, v29.4s, v9.s[3]     \n"
                        "fmla   v22.4s, v29.4s, v10.s[1]    \n"
                        "fmla   v23.4s, v29.4s, v10.s[3]    \n"

                        "prfm   pldl1keep, [%5, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%5], #64 \n" // r4

                        "fmla   v16.4s, v30.4s, v7.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v8.s[0]     \n"
                        "fmla   v18.4s, v30.4s, v8.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v9.s[0]     \n"
                        "fmla   v20.4s, v30.4s, v9.s[2]     \n"
                        "fmla   v21.4s, v30.4s, v10.s[0]    \n"
                        "fmla   v22.4s, v30.4s, v10.s[2]    \n"
                        "fmla   v23.4s, v30.4s, v11.s[0]    \n"

                        "prfm   pldl1keep, [%5, #256]       \n"
                        "ld1    {v4.4s, v5.4s}, [%5]        \n"

                        "fmla   v16.4s, v24.4s, v0.s[0]     \n"
                        "fmla   v17.4s, v24.4s, v0.s[2]     \n"
                        "fmla   v18.4s, v24.4s, v1.s[0]     \n"
                        "fmla   v19.4s, v24.4s, v1.s[2]     \n"
                        "fmla   v20.4s, v24.4s, v2.s[0]     \n"
                        "fmla   v21.4s, v24.4s, v2.s[2]     \n"
                        "fmla   v22.4s, v24.4s, v3.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v3.s[2]     \n"

                        "fmla   v16.4s, v25.4s, v0.s[1]     \n"
                        "fmla   v17.4s, v25.4s, v0.s[3]     \n"
                        "fmla   v18.4s, v25.4s, v1.s[1]     \n"
                        "fmla   v19.4s, v25.4s, v1.s[3]     \n"
                        "fmla   v20.4s, v25.4s, v2.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v2.s[3]     \n"
                        "fmla   v22.4s, v25.4s, v3.s[1]     \n"
                        "fmla   v23.4s, v25.4s, v3.s[3]     \n"

                        "prfm   pldl1keep, [%8, #384]       \n"
                        "ld1    {v28.4s, v29.4s, v30.4s}, [%8], #48 \n"

                        "fmla   v16.4s, v26.4s, v0.s[2]     \n"
                        "fmla   v17.4s, v26.4s, v1.s[0]     \n"
                        "fmla   v18.4s, v26.4s, v1.s[2]     \n"
                        "fmla   v19.4s, v26.4s, v2.s[0]     \n"
                        "fmla   v20.4s, v26.4s, v2.s[2]     \n"
                        "fmla   v21.4s, v26.4s, v3.s[0]     \n"
                        "fmla   v22.4s, v26.4s, v3.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v4.s[0]     \n"

                        "fmla   v16.4s, v27.4s, v0.s[3]     \n"
                        "fmla   v17.4s, v27.4s, v1.s[1]     \n"
                        "fmla   v18.4s, v27.4s, v1.s[3]     \n"
                        "fmla   v19.4s, v27.4s, v2.s[1]     \n"
                        "fmla   v20.4s, v27.4s, v2.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v3.s[1]     \n"
                        "fmla   v22.4s, v27.4s, v3.s[3]     \n"
                        "fmla   v23.4s, v27.4s, v4.s[1]     \n"

                        "fmla   v16.4s, v28.4s, v1.s[0]     \n"
                        "fmla   v17.4s, v28.4s, v1.s[2]     \n"
                        "fmla   v18.4s, v28.4s, v2.s[0]     \n"
                        "fmla   v19.4s, v28.4s, v2.s[2]     \n"
                        "fmla   v20.4s, v28.4s, v3.s[0]     \n"
                        "fmla   v21.4s, v28.4s, v3.s[2]     \n"
                        "fmla   v22.4s, v28.4s, v4.s[0]     \n"
                        "fmla   v23.4s, v28.4s, v4.s[2]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%8], #64 \n"

                        "fmla   v16.4s, v29.4s, v1.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v1.s[3]     \n"
                        "fmla   v18.4s, v29.4s, v2.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v2.s[3]     \n"
                        "fmla   v20.4s, v29.4s, v3.s[1]     \n"
                        "fmla   v21.4s, v29.4s, v3.s[3]     \n"
                        "fmla   v22.4s, v29.4s, v4.s[1]     \n"
                        "fmla   v23.4s, v29.4s, v4.s[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v6.4s, v7.4s, v8.4s, v9.4s}, [%6], #64 \n" // r5

                        "fmla   v16.4s, v30.4s, v1.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v2.s[0]     \n"
                        "fmla   v18.4s, v30.4s, v2.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v3.s[0]     \n"
                        "fmla   v20.4s, v30.4s, v3.s[2]     \n"
                        "fmla   v21.4s, v30.4s, v4.s[0]     \n"
                        "fmla   v22.4s, v30.4s, v4.s[2]     \n"
                        "fmla   v23.4s, v30.4s, v5.s[0]     \n"

                        "prfm   pldl1keep, [%6, #256]       \n"
                        "ld1    {v10.4s, v11.4s}, [%6]      \n"

                        "fmla   v16.4s, v24.4s, v6.s[0]     \n"
                        "fmla   v17.4s, v24.4s, v6.s[2]     \n"
                        "fmla   v18.4s, v24.4s, v7.s[0]     \n"
                        "fmla   v19.4s, v24.4s, v7.s[2]     \n"
                        "fmla   v20.4s, v24.4s, v8.s[0]     \n"
                        "fmla   v21.4s, v24.4s, v8.s[2]     \n"
                        "fmla   v22.4s, v24.4s, v9.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v9.s[2]     \n"

                        "fmla   v16.4s, v25.4s, v6.s[1]     \n"
                        "fmla   v17.4s, v25.4s, v6.s[3]     \n"
                        "fmla   v18.4s, v25.4s, v7.s[1]     \n"
                        "fmla   v19.4s, v25.4s, v7.s[3]     \n"
                        "fmla   v20.4s, v25.4s, v8.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v8.s[3]     \n"
                        "fmla   v22.4s, v25.4s, v9.s[1]     \n"
                        "fmla   v23.4s, v25.4s, v9.s[3]     \n"

                        "prfm   pldl1keep, [%8, #384]       \n"
                        "ld1    {v28.4s, v29.4s, v30.4s}, [%8], #48 \n"

                        "fmla   v16.4s, v26.4s, v6.s[2]     \n"
                        "fmla   v17.4s, v26.4s, v7.s[0]     \n"
                        "fmla   v18.4s, v26.4s, v7.s[2]     \n"
                        "fmla   v19.4s, v26.4s, v8.s[0]     \n"
                        "fmla   v20.4s, v26.4s, v8.s[2]     \n"
                        "fmla   v21.4s, v26.4s, v9.s[0]     \n"
                        "fmla   v22.4s, v26.4s, v9.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v10.s[0]    \n"

                        "fmla   v16.4s, v27.4s, v6.s[3]     \n"
                        "fmla   v17.4s, v27.4s, v7.s[1]     \n"
                        "fmla   v18.4s, v27.4s, v7.s[3]     \n"
                        "fmla   v19.4s, v27.4s, v8.s[1]     \n"
                        "fmla   v20.4s, v27.4s, v8.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v9.s[1]     \n"
                        "fmla   v22.4s, v27.4s, v9.s[3]     \n"
                        "fmla   v23.4s, v27.4s, v10.s[1]    \n"

                        "fmla   v16.4s, v28.4s, v7.s[0]     \n"
                        "fmla   v17.4s, v28.4s, v7.s[2]     \n"
                        "fmla   v18.4s, v28.4s, v8.s[0]     \n"
                        "fmla   v19.4s, v28.4s, v8.s[2]     \n"
                        "fmla   v20.4s, v28.4s, v9.s[0]     \n"
                        "fmla   v21.4s, v28.4s, v9.s[2]     \n"
                        "fmla   v22.4s, v28.4s, v10.s[0]    \n"
                        "fmla   v23.4s, v28.4s, v10.s[2]    \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%8], #64 \n"

                        "fmla   v16.4s, v29.4s, v7.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v7.s[3]     \n"
                        "fmla   v18.4s, v29.4s, v8.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v8.s[3]     \n"
                        "fmla   v20.4s, v29.4s, v9.s[1]     \n"
                        "fmla   v21.4s, v29.4s, v9.s[3]     \n"
                        "fmla   v22.4s, v29.4s, v10.s[1]    \n"
                        "fmla   v23.4s, v29.4s, v10.s[3]    \n"

                        "prfm   pldl1keep, [%7, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%7], #64 \n" // r6

                        "fmla   v16.4s, v30.4s, v7.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v8.s[0]     \n"
                        "fmla   v18.4s, v30.4s, v8.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v9.s[0]     \n"
                        "fmla   v20.4s, v30.4s, v9.s[2]     \n"
                        "fmla   v21.4s, v30.4s, v10.s[0]    \n"
                        "fmla   v22.4s, v30.4s, v10.s[2]    \n"
                        "fmla   v23.4s, v30.4s, v11.s[0]    \n"

                        "prfm   pldl1keep, [%7, #256]       \n"
                        "ld1    {v4.4s, v5.4s}, [%7]        \n"

                        "fmla   v16.4s, v24.4s, v0.s[0]     \n"
                        "fmla   v17.4s, v24.4s, v0.s[2]     \n"
                        "fmla   v18.4s, v24.4s, v1.s[0]     \n"
                        "fmla   v19.4s, v24.4s, v1.s[2]     \n"
                        "fmla   v20.4s, v24.4s, v2.s[0]     \n"
                        "fmla   v21.4s, v24.4s, v2.s[2]     \n"
                        "fmla   v22.4s, v24.4s, v3.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v3.s[2]     \n"

                        "fmla   v16.4s, v25.4s, v0.s[1]     \n"
                        "fmla   v17.4s, v25.4s, v0.s[3]     \n"
                        "fmla   v18.4s, v25.4s, v1.s[1]     \n"
                        "fmla   v19.4s, v25.4s, v1.s[3]     \n"
                        "fmla   v20.4s, v25.4s, v2.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v2.s[3]     \n"
                        "fmla   v22.4s, v25.4s, v3.s[1]     \n"
                        "fmla   v23.4s, v25.4s, v3.s[3]     \n"

                        "prfm   pldl1keep, [%8, #384]       \n"
                        "ld1    {v28.4s, v29.4s, v30.4s}, [%8], #48 \n"

                        "fmla   v16.4s, v26.4s, v0.s[2]     \n"
                        "fmla   v17.4s, v26.4s, v1.s[0]     \n"
                        "fmla   v18.4s, v26.4s, v1.s[2]     \n"
                        "fmla   v19.4s, v26.4s, v2.s[0]     \n"
                        "fmla   v20.4s, v26.4s, v2.s[2]     \n"
                        "fmla   v21.4s, v26.4s, v3.s[0]     \n"
                        "fmla   v22.4s, v26.4s, v3.s[2]     \n"
                        "fmla   v23.4s, v26.4s, v4.s[0]     \n"

                        "fmla   v16.4s, v27.4s, v0.s[3]     \n"
                        "fmla   v17.4s, v27.4s, v1.s[1]     \n"
                        "fmla   v18.4s, v27.4s, v1.s[3]     \n"
                        "fmla   v19.4s, v27.4s, v2.s[1]     \n"
                        "fmla   v20.4s, v27.4s, v2.s[3]     \n"
                        "fmla   v21.4s, v27.4s, v3.s[1]     \n"
                        "fmla   v22.4s, v27.4s, v3.s[3]     \n"
                        "fmla   v23.4s, v27.4s, v4.s[1]     \n"

                        "fmla   v16.4s, v28.4s, v1.s[0]     \n"
                        "fmla   v17.4s, v28.4s, v1.s[2]     \n"
                        "fmla   v18.4s, v28.4s, v2.s[0]     \n"
                        "fmla   v19.4s, v28.4s, v2.s[2]     \n"
                        "fmla   v20.4s, v28.4s, v3.s[0]     \n"
                        "fmla   v21.4s, v28.4s, v3.s[2]     \n"
                        "fmla   v22.4s, v28.4s, v4.s[0]     \n"
                        "fmla   v23.4s, v28.4s, v4.s[2]     \n"

                        "sub    %0, %0, #64                 \n"

                        "fmla   v16.4s, v29.4s, v1.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v1.s[3]     \n"
                        "fmla   v18.4s, v29.4s, v2.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v2.s[3]     \n"
                        "fmla   v20.4s, v29.4s, v3.s[1]     \n"
                        "fmla   v21.4s, v29.4s, v3.s[3]     \n"
                        "fmla   v22.4s, v29.4s, v4.s[1]     \n"
                        "fmla   v23.4s, v29.4s, v4.s[3]     \n"

                        "fmla   v16.4s, v30.4s, v1.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v2.s[0]     \n"
                        "fmla   v18.4s, v30.4s, v2.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v3.s[0]     \n"
                        "fmla   v20.4s, v30.4s, v3.s[2]     \n"
                        "fmla   v21.4s, v30.4s, v4.s[0]     \n"
                        "fmla   v22.4s, v30.4s, v4.s[2]     \n"
                        "fmla   v23.4s, v30.4s, v5.s[0]     \n"

                        "sub    %8, %8, #784                \n"

                        "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                        "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2),      // %3
                        "=r"(r3),      // %4
                        "=r"(r4),      // %5
                        "=r"(r5),      // %6
                        "=r"(r6),      // %7
                        "=r"(kptr)     // %8
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "4"(r3),
                        "5"(r4),
                        "6"(r5),
                        "7"(r6),
                        "8"(kptr)
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v16", "v17", "v18", "v19", "v24", "v25", "v26", "v27", "v28", "v29", "v30");
                }
#endif // __aarch64__
                for (; j + 3 < outw; j += 4)
                {
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%0, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0] \n"

                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%1] \n" // r0
                        "add    %1, %1, #32                 \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%8], #64 \n"

                        "fmla   v16.4s, v24.4s, v0.s[0]     \n"
                        "fmla   v17.4s, v24.4s, v0.s[2]     \n"
                        "fmla   v18.4s, v24.4s, v1.s[0]     \n"
                        "fmla   v19.4s, v24.4s, v1.s[2]     \n"

                        "prfm   pldl1keep, [%8, #384]       \n"
                        "ld1    {v28.4s, v29.4s, v30.4s}, [%8], #48 \n"

                        "fmla   v16.4s, v25.4s, v0.s[1]     \n"
                        "fmla   v17.4s, v25.4s, v0.s[3]     \n"
                        "fmla   v18.4s, v25.4s, v1.s[1]     \n"
                        "fmla   v19.4s, v25.4s, v1.s[3]     \n"
                        "fmla   v16.4s, v26.4s, v0.s[2]     \n"
                        "fmla   v17.4s, v26.4s, v1.s[0]     \n"
                        "fmla   v18.4s, v26.4s, v1.s[2]     \n"
                        "fmla   v19.4s, v26.4s, v2.s[0]     \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%2] \n" // r1
                        "add    %2, %2, #32                 \n"

                        "fmla   v16.4s, v27.4s, v0.s[3]     \n"
                        "fmla   v17.4s, v27.4s, v1.s[1]     \n"
                        "fmla   v18.4s, v27.4s, v1.s[3]     \n"
                        "fmla   v19.4s, v27.4s, v2.s[1]     \n"
                        "fmla   v16.4s, v28.4s, v1.s[0]     \n"
                        "fmla   v17.4s, v28.4s, v1.s[2]     \n"
                        "fmla   v18.4s, v28.4s, v2.s[0]     \n"
                        "fmla   v19.4s, v28.4s, v2.s[2]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%8], #64 \n"

                        "fmla   v16.4s, v29.4s, v1.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v1.s[3]     \n"
                        "fmla   v18.4s, v29.4s, v2.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v2.s[3]     \n"
                        "fmla   v16.4s, v30.4s, v1.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v2.s[0]     \n"
                        "fmla   v18.4s, v30.4s, v2.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v3.s[0]     \n"

                        "prfm   pldl1keep, [%8, #384]       \n"
                        "ld1    {v28.4s, v29.4s, v30.4s}, [%8], #48 \n"

                        "fmla   v16.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v17.4s, v24.4s, v4.s[2]     \n"
                        "fmla   v18.4s, v24.4s, v5.s[0]     \n"
                        "fmla   v19.4s, v24.4s, v5.s[2]     \n"
                        "fmla   v16.4s, v25.4s, v4.s[1]     \n"
                        "fmla   v17.4s, v25.4s, v4.s[3]     \n"
                        "fmla   v18.4s, v25.4s, v5.s[1]     \n"
                        "fmla   v19.4s, v25.4s, v5.s[3]     \n"
                        "fmla   v16.4s, v26.4s, v4.s[2]     \n"
                        "fmla   v17.4s, v26.4s, v5.s[0]     \n"
                        "fmla   v18.4s, v26.4s, v5.s[2]     \n"
                        "fmla   v19.4s, v26.4s, v6.s[0]     \n"

                        "prfm   pldl1keep, [%3, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%3] \n" // r2
                        "add    %3, %3, #32                 \n"

                        "fmla   v16.4s, v27.4s, v4.s[3]     \n"
                        "fmla   v17.4s, v27.4s, v5.s[1]     \n"
                        "fmla   v18.4s, v27.4s, v5.s[3]     \n"
                        "fmla   v19.4s, v27.4s, v6.s[1]     \n"
                        "fmla   v16.4s, v28.4s, v5.s[0]     \n"
                        "fmla   v17.4s, v28.4s, v5.s[2]     \n"
                        "fmla   v18.4s, v28.4s, v6.s[0]     \n"
                        "fmla   v19.4s, v28.4s, v6.s[2]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%8], #64 \n"

                        "fmla   v16.4s, v29.4s, v5.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v5.s[3]     \n"
                        "fmla   v18.4s, v29.4s, v6.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v6.s[3]     \n"
                        "fmla   v16.4s, v30.4s, v5.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v6.s[0]     \n"
                        "fmla   v18.4s, v30.4s, v6.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v7.s[0]     \n"

                        "prfm   pldl1keep, [%8, #384]       \n"
                        "ld1    {v28.4s, v29.4s, v30.4s}, [%8], #48 \n"

                        "fmla   v16.4s, v24.4s, v0.s[0]     \n"
                        "fmla   v17.4s, v24.4s, v0.s[2]     \n"
                        "fmla   v18.4s, v24.4s, v1.s[0]     \n"
                        "fmla   v19.4s, v24.4s, v1.s[2]     \n"
                        "fmla   v16.4s, v25.4s, v0.s[1]     \n"
                        "fmla   v17.4s, v25.4s, v0.s[3]     \n"
                        "fmla   v18.4s, v25.4s, v1.s[1]     \n"
                        "fmla   v19.4s, v25.4s, v1.s[3]     \n"
                        "fmla   v16.4s, v26.4s, v0.s[2]     \n"
                        "fmla   v17.4s, v26.4s, v1.s[0]     \n"
                        "fmla   v18.4s, v26.4s, v1.s[2]     \n"
                        "fmla   v19.4s, v26.4s, v2.s[0]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%4] \n" // r3
                        "add    %4, %4, #32                 \n"

                        "fmla   v16.4s, v27.4s, v0.s[3]     \n"
                        "fmla   v17.4s, v27.4s, v1.s[1]     \n"
                        "fmla   v18.4s, v27.4s, v1.s[3]     \n"
                        "fmla   v19.4s, v27.4s, v2.s[1]     \n"
                        "fmla   v16.4s, v28.4s, v1.s[0]     \n"
                        "fmla   v17.4s, v28.4s, v1.s[2]     \n"
                        "fmla   v18.4s, v28.4s, v2.s[0]     \n"
                        "fmla   v19.4s, v28.4s, v2.s[2]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%8], #64 \n"

                        "fmla   v16.4s, v29.4s, v1.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v1.s[3]     \n"
                        "fmla   v18.4s, v29.4s, v2.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v2.s[3]     \n"
                        "fmla   v16.4s, v30.4s, v1.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v2.s[0]     \n"
                        "fmla   v18.4s, v30.4s, v2.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v3.s[0]     \n"

                        "prfm   pldl1keep, [%8, #384]       \n"
                        "ld1    {v28.4s, v29.4s, v30.4s}, [%8], #48 \n"

                        "fmla   v16.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v17.4s, v24.4s, v4.s[2]     \n"
                        "fmla   v18.4s, v24.4s, v5.s[0]     \n"
                        "fmla   v19.4s, v24.4s, v5.s[2]     \n"
                        "fmla   v16.4s, v25.4s, v4.s[1]     \n"
                        "fmla   v17.4s, v25.4s, v4.s[3]     \n"
                        "fmla   v18.4s, v25.4s, v5.s[1]     \n"
                        "fmla   v19.4s, v25.4s, v5.s[3]     \n"
                        "fmla   v16.4s, v26.4s, v4.s[2]     \n"
                        "fmla   v17.4s, v26.4s, v5.s[0]     \n"
                        "fmla   v18.4s, v26.4s, v5.s[2]     \n"
                        "fmla   v19.4s, v26.4s, v6.s[0]     \n"

                        "prfm   pldl1keep, [%5, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%5] \n" // r4
                        "add    %5, %5, #32                 \n"

                        "fmla   v16.4s, v27.4s, v4.s[3]     \n"
                        "fmla   v17.4s, v27.4s, v5.s[1]     \n"
                        "fmla   v18.4s, v27.4s, v5.s[3]     \n"
                        "fmla   v19.4s, v27.4s, v6.s[1]     \n"
                        "fmla   v16.4s, v28.4s, v5.s[0]     \n"
                        "fmla   v17.4s, v28.4s, v5.s[2]     \n"
                        "fmla   v18.4s, v28.4s, v6.s[0]     \n"
                        "fmla   v19.4s, v28.4s, v6.s[2]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%8], #64 \n"

                        "fmla   v16.4s, v29.4s, v5.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v5.s[3]     \n"
                        "fmla   v18.4s, v29.4s, v6.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v6.s[3]     \n"
                        "fmla   v16.4s, v30.4s, v5.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v6.s[0]     \n"
                        "fmla   v18.4s, v30.4s, v6.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v7.s[0]     \n"

                        "prfm   pldl1keep, [%8, #384]       \n"
                        "ld1    {v28.4s, v29.4s, v30.4s}, [%8], #48 \n"

                        "fmla   v16.4s, v24.4s, v0.s[0]     \n"
                        "fmla   v17.4s, v24.4s, v0.s[2]     \n"
                        "fmla   v18.4s, v24.4s, v1.s[0]     \n"
                        "fmla   v19.4s, v24.4s, v1.s[2]     \n"
                        "fmla   v16.4s, v25.4s, v0.s[1]     \n"
                        "fmla   v17.4s, v25.4s, v0.s[3]     \n"
                        "fmla   v18.4s, v25.4s, v1.s[1]     \n"
                        "fmla   v19.4s, v25.4s, v1.s[3]     \n"
                        "fmla   v16.4s, v26.4s, v0.s[2]     \n"
                        "fmla   v17.4s, v26.4s, v1.s[0]     \n"
                        "fmla   v18.4s, v26.4s, v1.s[2]     \n"
                        "fmla   v19.4s, v26.4s, v2.s[0]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%6] \n" // r5
                        "add    %6, %6, #32                 \n"

                        "fmla   v16.4s, v27.4s, v0.s[3]     \n"
                        "fmla   v17.4s, v27.4s, v1.s[1]     \n"
                        "fmla   v18.4s, v27.4s, v1.s[3]     \n"
                        "fmla   v19.4s, v27.4s, v2.s[1]     \n"
                        "fmla   v16.4s, v28.4s, v1.s[0]     \n"
                        "fmla   v17.4s, v28.4s, v1.s[2]     \n"
                        "fmla   v18.4s, v28.4s, v2.s[0]     \n"
                        "fmla   v19.4s, v28.4s, v2.s[2]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%8], #64 \n"

                        "fmla   v16.4s, v29.4s, v1.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v1.s[3]     \n"
                        "fmla   v18.4s, v29.4s, v2.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v2.s[3]     \n"
                        "fmla   v16.4s, v30.4s, v1.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v2.s[0]     \n"
                        "fmla   v18.4s, v30.4s, v2.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v3.s[0]     \n"

                        "prfm   pldl1keep, [%8, #384]       \n"
                        "ld1    {v28.4s, v29.4s, v30.4s}, [%8], #48 \n"

                        "fmla   v16.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v17.4s, v24.4s, v4.s[2]     \n"
                        "fmla   v18.4s, v24.4s, v5.s[0]     \n"
                        "fmla   v19.4s, v24.4s, v5.s[2]     \n"
                        "fmla   v16.4s, v25.4s, v4.s[1]     \n"
                        "fmla   v17.4s, v25.4s, v4.s[3]     \n"
                        "fmla   v18.4s, v25.4s, v5.s[1]     \n"
                        "fmla   v19.4s, v25.4s, v5.s[3]     \n"
                        "fmla   v16.4s, v26.4s, v4.s[2]     \n"
                        "fmla   v17.4s, v26.4s, v5.s[0]     \n"
                        "fmla   v18.4s, v26.4s, v5.s[2]     \n"
                        "fmla   v19.4s, v26.4s, v6.s[0]     \n"

                        "prfm   pldl1keep, [%7, #512]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%7] \n" // r6
                        "add    %7, %7, #32                 \n"

                        "fmla   v16.4s, v27.4s, v4.s[3]     \n"
                        "fmla   v17.4s, v27.4s, v5.s[1]     \n"
                        "fmla   v18.4s, v27.4s, v5.s[3]     \n"
                        "fmla   v19.4s, v27.4s, v6.s[1]     \n"
                        "fmla   v16.4s, v28.4s, v5.s[0]     \n"
                        "fmla   v17.4s, v28.4s, v5.s[2]     \n"
                        "fmla   v18.4s, v28.4s, v6.s[0]     \n"
                        "fmla   v19.4s, v28.4s, v6.s[2]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%8], #64 \n"

                        "fmla   v16.4s, v29.4s, v5.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v5.s[3]     \n"
                        "fmla   v18.4s, v29.4s, v6.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v6.s[3]     \n"
                        "fmla   v16.4s, v30.4s, v5.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v6.s[0]     \n"
                        "fmla   v18.4s, v30.4s, v6.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v7.s[0]     \n"

                        "prfm   pldl1keep, [%8, #384]       \n"
                        "ld1    {v28.4s, v29.4s, v30.4s}, [%8], #48 \n"

                        "fmla   v16.4s, v24.4s, v0.s[0]     \n"
                        "fmla   v17.4s, v24.4s, v0.s[2]     \n"
                        "fmla   v18.4s, v24.4s, v1.s[0]     \n"
                        "fmla   v19.4s, v24.4s, v1.s[2]     \n"
                        "fmla   v16.4s, v25.4s, v0.s[1]     \n"
                        "fmla   v17.4s, v25.4s, v0.s[3]     \n"
                        "fmla   v18.4s, v25.4s, v1.s[1]     \n"
                        "fmla   v19.4s, v25.4s, v1.s[3]     \n"
                        "fmla   v16.4s, v26.4s, v0.s[2]     \n"
                        "fmla   v17.4s, v26.4s, v1.s[0]     \n"
                        "fmla   v18.4s, v26.4s, v1.s[2]     \n"
                        "fmla   v19.4s, v26.4s, v2.s[0]     \n"
                        "fmla   v16.4s, v27.4s, v0.s[3]     \n"
                        "fmla   v17.4s, v27.4s, v1.s[1]     \n"
                        "fmla   v18.4s, v27.4s, v1.s[3]     \n"
                        "fmla   v19.4s, v27.4s, v2.s[1]     \n"
                        "fmla   v16.4s, v28.4s, v1.s[0]     \n"
                        "fmla   v17.4s, v28.4s, v1.s[2]     \n"
                        "fmla   v18.4s, v28.4s, v2.s[0]     \n"
                        "fmla   v19.4s, v28.4s, v2.s[2]     \n"
                        "fmla   v16.4s, v29.4s, v1.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v1.s[3]     \n"
                        "fmla   v18.4s, v29.4s, v2.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v2.s[3]     \n"
                        "fmla   v16.4s, v30.4s, v1.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v2.s[0]     \n"
                        "fmla   v18.4s, v30.4s, v2.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v3.s[0]     \n"

                        "sub    %8, %8, #784                \n"

                        "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2),      // %3
                        "=r"(r3),      // %4
                        "=r"(r4),      // %5
                        "=r"(r5),      // %6
                        "=r"(r6),      // %7
                        "=r"(kptr)     // %8
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "4"(r3),
                        "5"(r4),
                        "6"(r5),
                        "7"(r6),
                        "8"(kptr)
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v24", "v25", "v26", "v27", "v28", "v29", "v30");
#else  // __aarch64__
                    asm volatile(
                        "pld        [%0, #512]      \n"
                        "vldm       %0, {d24-d31}   \n"

                        "pld        [%1, #256]      \n"
                        "vld1.f32   {d0-d3}, [%1]!  \n" // r0

                        "pld        [%8, #512]      \n"
                        "vldm       %8!, {d10-d17}  \n"

                        "vmla.f32   q12, q5, d0[0]  \n"
                        "vmla.f32   q13, q5, d1[0]  \n"
                        "vmla.f32   q14, q5, d2[0]  \n"
                        "vmla.f32   q15, q5, d3[0]  \n"

                        "pld        [%1, #192]      \n"
                        "vld1.f32   {d4-d6}, [%1]   \n"

                        "vmla.f32   q12, q6, d0[1]  \n"
                        "vmla.f32   q13, q6, d1[1]  \n"
                        "vmla.f32   q14, q6, d2[1]  \n"
                        "vmla.f32   q15, q6, d3[1]  \n"

                        "pld        [%8, #384]      \n"
                        "vldm       %8!, {d18-d23}  \n"

                        "vmla.f32   q12, q7, d1[0]  \n"
                        "vmla.f32   q13, q7, d2[0]  \n"
                        "vmla.f32   q14, q7, d3[0]  \n"
                        "vmla.f32   q15, q7, d4[0]  \n"
                        "vmla.f32   q12, q8, d1[1]  \n"
                        "vmla.f32   q13, q8, d2[1]  \n"
                        "vmla.f32   q14, q8, d3[1]  \n"
                        "vmla.f32   q15, q8, d4[1]  \n"
                        "vmla.f32   q12, q9, d2[0]  \n"
                        "vmla.f32   q13, q9, d3[0]  \n"
                        "vmla.f32   q14, q9, d4[0]  \n"
                        "vmla.f32   q15, q9, d5[0]  \n"

                        "pld        [%8, #512]      \n"
                        "vldm       %8!, {d10-d17}  \n"

                        "vmla.f32   q12, q10, d2[1] \n"
                        "vmla.f32   q13, q10, d3[1] \n"
                        "vmla.f32   q14, q10, d4[1] \n"
                        "vmla.f32   q15, q10, d5[1] \n"
                        "vmla.f32   q12, q11, d3[0] \n"
                        "vmla.f32   q13, q11, d4[0] \n"

                        "pld        [%2, #256]      \n"
                        "vld1.f32   {d0-d3}, [%2]!  \n" // r1

                        "vmla.f32   q14, q11, d5[0] \n"
                        "vmla.f32   q15, q11, d6[0] \n"

                        "vmla.f32   q12, q5, d0[0]  \n"
                        "vmla.f32   q13, q5, d1[0]  \n"
                        "vmla.f32   q14, q5, d2[0]  \n"
                        "vmla.f32   q15, q5, d3[0]  \n"

                        "pld        [%2, #192]      \n"
                        "vld1.f32   {d4-d6}, [%2]   \n"

                        "vmla.f32   q12, q6, d0[1]  \n"
                        "vmla.f32   q13, q6, d1[1]  \n"
                        "vmla.f32   q14, q6, d2[1]  \n"
                        "vmla.f32   q15, q6, d3[1]  \n"

                        "pld        [%8, #384]      \n"
                        "vldm       %8!, {d18-d23}  \n"

                        "vmla.f32   q12, q7, d1[0]  \n"
                        "vmla.f32   q13, q7, d2[0]  \n"
                        "vmla.f32   q14, q7, d3[0]  \n"
                        "vmla.f32   q15, q7, d4[0]  \n"
                        "vmla.f32   q12, q8, d1[1]  \n"
                        "vmla.f32   q13, q8, d2[1]  \n"
                        "vmla.f32   q14, q8, d3[1]  \n"
                        "vmla.f32   q15, q8, d4[1]  \n"
                        "vmla.f32   q12, q9, d2[0]  \n"
                        "vmla.f32   q13, q9, d3[0]  \n"
                        "vmla.f32   q14, q9, d4[0]  \n"
                        "vmla.f32   q15, q9, d5[0]  \n"

                        "pld        [%8, #512]      \n"
                        "vldm       %8!, {d10-d17}  \n"

                        "vmla.f32   q12, q10, d2[1] \n"
                        "vmla.f32   q13, q10, d3[1] \n"
                        "vmla.f32   q14, q10, d4[1] \n"
                        "vmla.f32   q15, q10, d5[1] \n"
                        "vmla.f32   q12, q11, d3[0] \n"
                        "vmla.f32   q13, q11, d4[0] \n"

                        "pld        [%3, #256]      \n"
                        "vld1.f32   {d0-d3}, [%3]!  \n" // r2

                        "vmla.f32   q14, q11, d5[0] \n"
                        "vmla.f32   q15, q11, d6[0] \n"

                        "vmla.f32   q12, q5, d0[0]  \n"
                        "vmla.f32   q13, q5, d1[0]  \n"
                        "vmla.f32   q14, q5, d2[0]  \n"
                        "vmla.f32   q15, q5, d3[0]  \n"

                        "pld        [%3, #192]      \n"
                        "vld1.f32   {d4-d6}, [%3]   \n"

                        "vmla.f32   q12, q6, d0[1]  \n"
                        "vmla.f32   q13, q6, d1[1]  \n"
                        "vmla.f32   q14, q6, d2[1]  \n"
                        "vmla.f32   q15, q6, d3[1]  \n"

                        "pld        [%8, #384]      \n"
                        "vldm       %8!, {d18-d23}  \n"

                        "vmla.f32   q12, q7, d1[0]  \n"
                        "vmla.f32   q13, q7, d2[0]  \n"
                        "vmla.f32   q14, q7, d3[0]  \n"
                        "vmla.f32   q15, q7, d4[0]  \n"
                        "vmla.f32   q12, q8, d1[1]  \n"
                        "vmla.f32   q13, q8, d2[1]  \n"
                        "vmla.f32   q14, q8, d3[1]  \n"
                        "vmla.f32   q15, q8, d4[1]  \n"
                        "vmla.f32   q12, q9, d2[0]  \n"
                        "vmla.f32   q13, q9, d3[0]  \n"
                        "vmla.f32   q14, q9, d4[0]  \n"
                        "vmla.f32   q15, q9, d5[0]  \n"

                        "pld        [%8, #512]      \n"
                        "vldm       %8!, {d10-d17}  \n"

                        "vmla.f32   q12, q10, d2[1] \n"
                        "vmla.f32   q13, q10, d3[1] \n"
                        "vmla.f32   q14, q10, d4[1] \n"
                        "vmla.f32   q15, q10, d5[1] \n"
                        "vmla.f32   q12, q11, d3[0] \n"
                        "vmla.f32   q13, q11, d4[0] \n"

                        "pld        [%4, #256]      \n"
                        "vld1.f32   {d0-d3}, [%4]!  \n" // r3

                        "vmla.f32   q14, q11, d5[0] \n"
                        "vmla.f32   q15, q11, d6[0] \n"

                        "vmla.f32   q12, q5, d0[0]  \n"
                        "vmla.f32   q13, q5, d1[0]  \n"
                        "vmla.f32   q14, q5, d2[0]  \n"
                        "vmla.f32   q15, q5, d3[0]  \n"

                        "pld        [%4, #192]      \n"
                        "vld1.f32   {d4-d6}, [%4]   \n"

                        "vmla.f32   q12, q6, d0[1]  \n"
                        "vmla.f32   q13, q6, d1[1]  \n"
                        "vmla.f32   q14, q6, d2[1]  \n"
                        "vmla.f32   q15, q6, d3[1]  \n"

                        "pld        [%8, #384]      \n"
                        "vldm       %8!, {d18-d23}  \n"

                        "vmla.f32   q12, q7, d1[0]  \n"
                        "vmla.f32   q13, q7, d2[0]  \n"
                        "vmla.f32   q14, q7, d3[0]  \n"
                        "vmla.f32   q15, q7, d4[0]  \n"
                        "vmla.f32   q12, q8, d1[1]  \n"
                        "vmla.f32   q13, q8, d2[1]  \n"
                        "vmla.f32   q14, q8, d3[1]  \n"
                        "vmla.f32   q15, q8, d4[1]  \n"
                        "vmla.f32   q12, q9, d2[0]  \n"
                        "vmla.f32   q13, q9, d3[0]  \n"
                        "vmla.f32   q14, q9, d4[0]  \n"
                        "vmla.f32   q15, q9, d5[0]  \n"

                        "pld        [%8, #512]      \n"
                        "vldm       %8!, {d10-d17}  \n"

                        "vmla.f32   q12, q10, d2[1] \n"
                        "vmla.f32   q13, q10, d3[1] \n"
                        "vmla.f32   q14, q10, d4[1] \n"
                        "vmla.f32   q15, q10, d5[1] \n"
                        "vmla.f32   q12, q11, d3[0] \n"
                        "vmla.f32   q13, q11, d4[0] \n"

                        "pld        [%5, #256]      \n"
                        "vld1.f32   {d0-d3}, [%5]!  \n" // r4

                        "vmla.f32   q14, q11, d5[0] \n"
                        "vmla.f32   q15, q11, d6[0] \n"

                        "vmla.f32   q12, q5, d0[0]  \n"
                        "vmla.f32   q13, q5, d1[0]  \n"
                        "vmla.f32   q14, q5, d2[0]  \n"
                        "vmla.f32   q15, q5, d3[0]  \n"

                        "pld        [%5, #192]      \n"
                        "vld1.f32   {d4-d6}, [%5]   \n"

                        "vmla.f32   q12, q6, d0[1]  \n"
                        "vmla.f32   q13, q6, d1[1]  \n"
                        "vmla.f32   q14, q6, d2[1]  \n"
                        "vmla.f32   q15, q6, d3[1]  \n"

                        "pld        [%8, #384]      \n"
                        "vldm       %8!, {d18-d23}  \n"

                        "vmla.f32   q12, q7, d1[0]  \n"
                        "vmla.f32   q13, q7, d2[0]  \n"
                        "vmla.f32   q14, q7, d3[0]  \n"
                        "vmla.f32   q15, q7, d4[0]  \n"
                        "vmla.f32   q12, q8, d1[1]  \n"
                        "vmla.f32   q13, q8, d2[1]  \n"
                        "vmla.f32   q14, q8, d3[1]  \n"
                        "vmla.f32   q15, q8, d4[1]  \n"
                        "vmla.f32   q12, q9, d2[0]  \n"
                        "vmla.f32   q13, q9, d3[0]  \n"
                        "vmla.f32   q14, q9, d4[0]  \n"
                        "vmla.f32   q15, q9, d5[0]  \n"

                        "pld        [%8, #512]      \n"
                        "vldm       %8!, {d10-d17}  \n"

                        "vmla.f32   q12, q10, d2[1] \n"
                        "vmla.f32   q13, q10, d3[1] \n"
                        "vmla.f32   q14, q10, d4[1] \n"
                        "vmla.f32   q15, q10, d5[1] \n"
                        "vmla.f32   q12, q11, d3[0] \n"
                        "vmla.f32   q13, q11, d4[0] \n"

                        "pld        [%6, #256]      \n"
                        "vld1.f32   {d0-d3}, [%6]!  \n" // r5

                        "vmla.f32   q14, q11, d5[0] \n"
                        "vmla.f32   q15, q11, d6[0] \n"

                        "vmla.f32   q12, q5, d0[0]  \n"
                        "vmla.f32   q13, q5, d1[0]  \n"
                        "vmla.f32   q14, q5, d2[0]  \n"
                        "vmla.f32   q15, q5, d3[0]  \n"

                        "pld        [%6, #192]      \n"
                        "vld1.f32   {d4-d6}, [%6]   \n"

                        "vmla.f32   q12, q6, d0[1]  \n"
                        "vmla.f32   q13, q6, d1[1]  \n"
                        "vmla.f32   q14, q6, d2[1]  \n"
                        "vmla.f32   q15, q6, d3[1]  \n"

                        "pld        [%8, #384]      \n"
                        "vldm       %8!, {d18-d23}  \n"

                        "vmla.f32   q12, q7, d1[0]  \n"
                        "vmla.f32   q13, q7, d2[0]  \n"
                        "vmla.f32   q14, q7, d3[0]  \n"
                        "vmla.f32   q15, q7, d4[0]  \n"
                        "vmla.f32   q12, q8, d1[1]  \n"
                        "vmla.f32   q13, q8, d2[1]  \n"
                        "vmla.f32   q14, q8, d3[1]  \n"
                        "vmla.f32   q15, q8, d4[1]  \n"
                        "vmla.f32   q12, q9, d2[0]  \n"
                        "vmla.f32   q13, q9, d3[0]  \n"
                        "vmla.f32   q14, q9, d4[0]  \n"
                        "vmla.f32   q15, q9, d5[0]  \n"

                        "pld        [%8, #512]      \n"
                        "vldm       %8!, {d10-d17}  \n"

                        "vmla.f32   q12, q10, d2[1] \n"
                        "vmla.f32   q13, q10, d3[1] \n"
                        "vmla.f32   q14, q10, d4[1] \n"
                        "vmla.f32   q15, q10, d5[1] \n"
                        "vmla.f32   q12, q11, d3[0] \n"
                        "vmla.f32   q13, q11, d4[0] \n"

                        "pld        [%7, #256]      \n"
                        "vld1.f32   {d0-d3}, [%7]!  \n" // r6

                        "vmla.f32   q14, q11, d5[0] \n"
                        "vmla.f32   q15, q11, d6[0] \n"

                        "vmla.f32   q12, q5, d0[0]  \n"
                        "vmla.f32   q13, q5, d1[0]  \n"
                        "vmla.f32   q14, q5, d2[0]  \n"
                        "vmla.f32   q15, q5, d3[0]  \n"

                        "pld        [%7, #192]      \n"
                        "vld1.f32   {d4-d6}, [%7]   \n"

                        "vmla.f32   q12, q6, d0[1]  \n"
                        "vmla.f32   q13, q6, d1[1]  \n"
                        "vmla.f32   q14, q6, d2[1]  \n"
                        "vmla.f32   q15, q6, d3[1]  \n"

                        "pld        [%8, #384]      \n"
                        "vldm       %8!, {d18-d23}  \n"

                        "vmla.f32   q12, q7, d1[0]  \n"
                        "vmla.f32   q13, q7, d2[0]  \n"
                        "vmla.f32   q14, q7, d3[0]  \n"
                        "vmla.f32   q15, q7, d4[0]  \n"
                        "vmla.f32   q12, q8, d1[1]  \n"
                        "vmla.f32   q13, q8, d2[1]  \n"
                        "vmla.f32   q14, q8, d3[1]  \n"
                        "vmla.f32   q15, q8, d4[1]  \n"
                        "vmla.f32   q12, q9, d2[0]  \n"
                        "vmla.f32   q13, q9, d3[0]  \n"
                        "vmla.f32   q14, q9, d4[0]  \n"
                        "vmla.f32   q15, q9, d5[0]  \n"
                        "vmla.f32   q12, q10, d2[1] \n"
                        "vmla.f32   q13, q10, d3[1] \n"
                        "vmla.f32   q14, q10, d4[1] \n"
                        "vmla.f32   q15, q10, d5[1] \n"
                        "vmla.f32   q12, q11, d3[0] \n"
                        "vmla.f32   q13, q11, d4[0] \n"
                        "vmla.f32   q14, q11, d5[0] \n"
                        "vmla.f32   q15, q11, d6[0] \n"

                        "sub        %8, %8, #784    \n"

                        "vstm       %0!, {d24-d31}  \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2),      // %3
                        "=r"(r3),      // %4
                        "=r"(r4),      // %5
                        "=r"(r5),      // %6
                        "=r"(r6),      // %7
                        "=r"(kptr)     // %8
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "4"(r3),
                        "5"(r4),
                        "6"(r5),
                        "7"(r6),
                        "8"(kptr)
                        : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif // __aarch64__
                }
                for (; j + 1 < outw; j += 2)
                {
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%0, #256]       \n"
                        "ld1    {v16.4s, v17.4s}, [%0]      \n"

                        "prfm   pldl1keep, [%1, #384]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s}, [%1] \n" // r0
                        "add    %1, %1, #16                 \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%8], #64 \n"

                        "fmul   v18.4s, v24.4s, v0.s[0]     \n"
                        "fmul   v19.4s, v24.4s, v0.s[2]     \n"

                        "prfm   pldl1keep, [%8, #384]       \n"
                        "ld1    {v28.4s, v29.4s, v30.4s}, [%8], #48 \n"

                        "fmla   v16.4s, v25.4s, v0.s[1]     \n"
                        "fmla   v17.4s, v25.4s, v0.s[3]     \n"
                        "fmla   v18.4s, v26.4s, v0.s[2]     \n"
                        "fmla   v19.4s, v26.4s, v1.s[0]     \n"

                        "prfm   pldl1keep, [%2, #384]       \n"
                        "ld1    {v4.4s, v5.4s, v6.4s}, [%2] \n" // r1
                        "add    %2, %2, #16                 \n"

                        "fmla   v16.4s, v27.4s, v0.s[3]     \n"
                        "fmla   v17.4s, v27.4s, v1.s[1]     \n"
                        "fmla   v18.4s, v28.4s, v1.s[0]     \n"
                        "fmla   v19.4s, v28.4s, v1.s[2]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%8], #64 \n"

                        "fmla   v16.4s, v29.4s, v1.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v1.s[3]     \n"
                        "fmla   v18.4s, v30.4s, v1.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v2.s[0]     \n"

                        "prfm   pldl1keep, [%8, #384]       \n"
                        "ld1    {v28.4s, v29.4s, v30.4s}, [%8], #48 \n"

                        "fmla   v16.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v17.4s, v24.4s, v4.s[2]     \n"
                        "fmla   v18.4s, v25.4s, v4.s[1]     \n"
                        "fmla   v19.4s, v25.4s, v4.s[3]     \n"
                        "fmla   v16.4s, v26.4s, v4.s[2]     \n"
                        "fmla   v17.4s, v26.4s, v5.s[0]     \n"

                        "prfm   pldl1keep, [%3, #384]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s}, [%3] \n" // r2
                        "add    %3, %3, #16                 \n"

                        "fmla   v18.4s, v27.4s, v4.s[3]     \n"
                        "fmla   v19.4s, v27.4s, v5.s[1]     \n"
                        "fmla   v16.4s, v28.4s, v5.s[0]     \n"
                        "fmla   v17.4s, v28.4s, v5.s[2]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%8], #64 \n"

                        "fmla   v18.4s, v29.4s, v5.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v5.s[3]     \n"
                        "fmla   v16.4s, v30.4s, v5.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v6.s[0]     \n"

                        "prfm   pldl1keep, [%8, #384]       \n"
                        "ld1    {v28.4s, v29.4s, v30.4s}, [%8], #48 \n"

                        "fmla   v18.4s, v24.4s, v0.s[0]     \n"
                        "fmla   v19.4s, v24.4s, v0.s[2]     \n"
                        "fmla   v16.4s, v25.4s, v0.s[1]     \n"
                        "fmla   v17.4s, v25.4s, v0.s[3]     \n"
                        "fmla   v18.4s, v26.4s, v0.s[2]     \n"
                        "fmla   v19.4s, v26.4s, v1.s[0]     \n"

                        "prfm   pldl1keep, [%4, #384]       \n"
                        "ld1    {v4.4s, v5.4s, v6.4s}, [%4] \n" // r3
                        "add    %4, %4, #16                 \n"

                        "fmla   v16.4s, v27.4s, v0.s[3]     \n"
                        "fmla   v17.4s, v27.4s, v1.s[1]     \n"
                        "fmla   v18.4s, v28.4s, v1.s[0]     \n"
                        "fmla   v19.4s, v28.4s, v1.s[2]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%8], #64 \n"

                        "fmla   v16.4s, v29.4s, v1.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v1.s[3]     \n"
                        "fmla   v18.4s, v30.4s, v1.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v2.s[0]     \n"

                        "prfm   pldl1keep, [%8, #384]       \n"
                        "ld1    {v28.4s, v29.4s, v30.4s}, [%8], #48 \n"

                        "fmla   v16.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v17.4s, v24.4s, v4.s[2]     \n"
                        "fmla   v18.4s, v25.4s, v4.s[1]     \n"
                        "fmla   v19.4s, v25.4s, v4.s[3]     \n"
                        "fmla   v16.4s, v26.4s, v4.s[2]     \n"
                        "fmla   v17.4s, v26.4s, v5.s[0]     \n"

                        "prfm   pldl1keep, [%5, #384]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s}, [%5] \n" // r4
                        "add    %5, %5, #16                 \n"

                        "fmla   v18.4s, v27.4s, v4.s[3]     \n"
                        "fmla   v19.4s, v27.4s, v5.s[1]     \n"
                        "fmla   v16.4s, v28.4s, v5.s[0]     \n"
                        "fmla   v17.4s, v28.4s, v5.s[2]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%8], #64 \n"

                        "fmla   v18.4s, v29.4s, v5.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v5.s[3]     \n"
                        "fmla   v16.4s, v30.4s, v5.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v6.s[0]     \n"

                        "prfm   pldl1keep, [%8, #384]       \n"
                        "ld1    {v28.4s, v29.4s, v30.4s}, [%8], #48 \n"

                        "fmla   v18.4s, v24.4s, v0.s[0]     \n"
                        "fmla   v19.4s, v24.4s, v0.s[2]     \n"
                        "fmla   v16.4s, v25.4s, v0.s[1]     \n"
                        "fmla   v17.4s, v25.4s, v0.s[3]     \n"
                        "fmla   v18.4s, v26.4s, v0.s[2]     \n"
                        "fmla   v19.4s, v26.4s, v1.s[0]     \n"

                        "prfm   pldl1keep, [%6, #384]       \n"
                        "ld1    {v4.4s, v5.4s, v6.4s}, [%6] \n" // r5
                        "add    %6, %6, #16                 \n"

                        "fmla   v16.4s, v27.4s, v0.s[3]     \n"
                        "fmla   v17.4s, v27.4s, v1.s[1]     \n"
                        "fmla   v18.4s, v28.4s, v1.s[0]     \n"
                        "fmla   v19.4s, v28.4s, v1.s[2]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%8], #64 \n"

                        "fmla   v16.4s, v29.4s, v1.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v1.s[3]     \n"
                        "fmla   v18.4s, v30.4s, v1.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v2.s[0]     \n"

                        "prfm   pldl1keep, [%8, #384]       \n"
                        "ld1    {v28.4s, v29.4s, v30.4s}, [%8], #48 \n"

                        "fmla   v16.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v17.4s, v24.4s, v4.s[2]     \n"
                        "fmla   v18.4s, v25.4s, v4.s[1]     \n"
                        "fmla   v19.4s, v25.4s, v4.s[3]     \n"
                        "fmla   v16.4s, v26.4s, v4.s[2]     \n"
                        "fmla   v17.4s, v26.4s, v5.s[0]     \n"

                        "prfm   pldl1keep, [%7, #384]       \n"
                        "ld1    {v0.4s, v1.4s, v2.4s}, [%7] \n" // r6
                        "add    %7, %7, #16                 \n"

                        "fmla   v18.4s, v27.4s, v4.s[3]     \n"
                        "fmla   v19.4s, v27.4s, v5.s[1]     \n"
                        "fmla   v16.4s, v28.4s, v5.s[0]     \n"
                        "fmla   v17.4s, v28.4s, v5.s[2]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%8], #64 \n"

                        "fmla   v18.4s, v29.4s, v5.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v5.s[3]     \n"
                        "fmla   v16.4s, v30.4s, v5.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v6.s[0]     \n"

                        "prfm   pldl1keep, [%8, #384]       \n"
                        "ld1    {v28.4s, v29.4s, v30.4s}, [%8], #48 \n"

                        "fmla   v18.4s, v24.4s, v0.s[0]     \n"
                        "fmla   v19.4s, v24.4s, v0.s[2]     \n"
                        "fmla   v16.4s, v25.4s, v0.s[1]     \n"
                        "fmla   v17.4s, v25.4s, v0.s[3]     \n"
                        "fmla   v18.4s, v26.4s, v0.s[2]     \n"
                        "fmla   v19.4s, v26.4s, v1.s[0]     \n"
                        "fmla   v16.4s, v27.4s, v0.s[3]     \n"
                        "fmla   v17.4s, v27.4s, v1.s[1]     \n"
                        "fmla   v18.4s, v28.4s, v1.s[0]     \n"
                        "fmla   v19.4s, v28.4s, v1.s[2]     \n"
                        "fmla   v16.4s, v29.4s, v1.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v1.s[3]     \n"
                        "fmla   v18.4s, v30.4s, v1.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v2.s[0]     \n"

                        "fadd   v16.4s, v16.4s, v18.4s      \n"
                        "fadd   v17.4s, v17.4s, v19.4s      \n"

                        "sub    %8, %8, #784                \n"

                        "st1    {v16.4s, v17.4s}, [%0], #32 \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2),      // %3
                        "=r"(r3),      // %4
                        "=r"(r4),      // %5
                        "=r"(r5),      // %6
                        "=r"(r6),      // %7
                        "=r"(kptr)     // %8
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "4"(r3),
                        "5"(r4),
                        "6"(r5),
                        "7"(r6),
                        "8"(kptr)
                        : "memory", "v0", "v1", "v2", "v4", "v5", "v6", "v16", "v17", "v18", "v19", "v24", "v25", "v26", "v27", "v28", "v29", "v30");
#else  // __aarch64__
                    asm volatile(
                        "pld        [%0, #256]      \n"
                        "vld1.f32   {d28-d31}, [%0 :128] \n"

                        "pld        [%1, #256]      \n"
                        "vld1.f32   {d0-d3}, [%1]!  \n" // r0
                        "vld1.f32   {d8[0]}, [%1]   \n"

                        "pld        [%8, #512]      \n"
                        "vldm       %8!, {d10-d17}  \n"

                        "vmul.f32   q12, q5, d0[0]  \n"
                        "vmul.f32   q13, q5, d1[0]  \n"
                        "vmla.f32   q14, q6, d0[1]  \n"
                        "vmla.f32   q15, q6, d1[1]  \n"

                        "pld        [%8, #384]      \n"
                        "vldm       %8!, {d18-d23}  \n"

                        "vmla.f32   q12, q7, d1[0]  \n"
                        "vmla.f32   q13, q7, d2[0]  \n"

                        "pld        [%2, #256]      \n"
                        "vld1.f32   {d4-d7}, [%2]!  \n" // r1
                        "vld1.f32   {d9[0]}, [%2]   \n"

                        "vmla.f32   q14, q8, d1[1]  \n"
                        "vmla.f32   q15, q8, d2[1]  \n"
                        "vmla.f32   q12, q9, d2[0]  \n"
                        "vmla.f32   q13, q9, d3[0]  \n"

                        "pld        [%8, #512]      \n"
                        "vldm       %8!, {d10-d17}  \n"

                        "vmla.f32   q14, q10, d2[1] \n"
                        "vmla.f32   q15, q10, d3[1] \n"
                        "vmla.f32   q12, q11, d3[0] \n"
                        "vmla.f32   q13, q11, d8[0] \n"

                        "pld        [%8, #384]      \n"
                        "vldm       %8!, {d18-d23}  \n"

                        "vmla.f32   q14, q5, d4[0]  \n"
                        "vmla.f32   q15, q5, d5[0]  \n"
                        "vmla.f32   q12, q6, d4[1]  \n"
                        "vmla.f32   q13, q6, d5[1]  \n"
                        "vmla.f32   q14, q7, d5[0]  \n"
                        "vmla.f32   q15, q7, d6[0]  \n"

                        "pld        [%3, #256]      \n"
                        "vld1.f32   {d0-d3}, [%3]!  \n" // r2
                        "vld1.f32   {d8[0]}, [%3]   \n"

                        "vmla.f32   q12, q8, d5[1]  \n"
                        "vmla.f32   q13, q8, d6[1]  \n"
                        "vmla.f32   q14, q9, d6[0]  \n"
                        "vmla.f32   q15, q9, d7[0]  \n"

                        "pld        [%8, #512]      \n"
                        "vldm       %8!, {d10-d17}  \n"

                        "vmla.f32   q12, q10, d6[1] \n"
                        "vmla.f32   q13, q10, d7[1] \n"
                        "vmla.f32   q14, q11, d7[0] \n"
                        "vmla.f32   q15, q11, d9[0] \n"

                        "pld        [%8, #384]      \n"
                        "vldm       %8!, {d18-d23}  \n"

                        "vmla.f32   q12, q5, d0[0]  \n"
                        "vmla.f32   q13, q5, d1[0]  \n"
                        "vmla.f32   q14, q6, d0[1]  \n"
                        "vmla.f32   q15, q6, d1[1]  \n"
                        "vmla.f32   q12, q7, d1[0]  \n"
                        "vmla.f32   q13, q7, d2[0]  \n"

                        "pld        [%4, #256]      \n"
                        "vld1.f32   {d4-d7}, [%4]!  \n" // r3
                        "vld1.f32   {d9[0]}, [%4]   \n"

                        "vmla.f32   q14, q8, d1[1]  \n"
                        "vmla.f32   q15, q8, d2[1]  \n"
                        "vmla.f32   q12, q9, d2[0]  \n"
                        "vmla.f32   q13, q9, d3[0]  \n"

                        "pld        [%8, #512]      \n"
                        "vldm       %8!, {d10-d17}  \n"

                        "vmla.f32   q14, q10, d2[1] \n"
                        "vmla.f32   q15, q10, d3[1] \n"
                        "vmla.f32   q12, q11, d3[0] \n"
                        "vmla.f32   q13, q11, d8[0] \n"

                        "pld        [%8, #384]      \n"
                        "vldm       %8!, {d18-d23}  \n"

                        "vmla.f32   q14, q5, d4[0]  \n"
                        "vmla.f32   q15, q5, d5[0]  \n"
                        "vmla.f32   q12, q6, d4[1]  \n"
                        "vmla.f32   q13, q6, d5[1]  \n"
                        "vmla.f32   q14, q7, d5[0]  \n"
                        "vmla.f32   q15, q7, d6[0]  \n"

                        "pld        [%5, #256]      \n"
                        "vld1.f32   {d0-d3}, [%5]!  \n" // r4
                        "vld1.f32   {d8[0]}, [%5]   \n"

                        "vmla.f32   q12, q8, d5[1]  \n"
                        "vmla.f32   q13, q8, d6[1]  \n"
                        "vmla.f32   q14, q9, d6[0]  \n"
                        "vmla.f32   q15, q9, d7[0]  \n"

                        "pld        [%8, #512]      \n"
                        "vldm       %8!, {d10-d17}  \n"

                        "vmla.f32   q12, q10, d6[1] \n"
                        "vmla.f32   q13, q10, d7[1] \n"
                        "vmla.f32   q14, q11, d7[0] \n"
                        "vmla.f32   q15, q11, d9[0] \n"

                        "pld        [%8, #384]      \n"
                        "vldm       %8!, {d18-d23}  \n"

                        "vmla.f32   q12, q5, d0[0]  \n"
                        "vmla.f32   q13, q5, d1[0]  \n"
                        "vmla.f32   q14, q6, d0[1]  \n"
                        "vmla.f32   q15, q6, d1[1]  \n"
                        "vmla.f32   q12, q7, d1[0]  \n"
                        "vmla.f32   q13, q7, d2[0]  \n"

                        "pld        [%6, #256]      \n"
                        "vld1.f32   {d4-d7}, [%6]!  \n" // r5
                        "vld1.f32   {d9[0]}, [%6]   \n"

                        "vmla.f32   q14, q8, d1[1]  \n"
                        "vmla.f32   q15, q8, d2[1]  \n"
                        "vmla.f32   q12, q9, d2[0]  \n"
                        "vmla.f32   q13, q9, d3[0]  \n"

                        "pld        [%8, #512]      \n"
                        "vldm       %8!, {d10-d17}  \n"

                        "vmla.f32   q14, q10, d2[1] \n"
                        "vmla.f32   q15, q10, d3[1] \n"
                        "vmla.f32   q12, q11, d3[0] \n"
                        "vmla.f32   q13, q11, d8[0] \n"

                        "pld        [%8, #384]      \n"
                        "vldm       %8!, {d18-d23}  \n"

                        "vmla.f32   q14, q5, d4[0]  \n"
                        "vmla.f32   q15, q5, d5[0]  \n"
                        "vmla.f32   q12, q6, d4[1]  \n"
                        "vmla.f32   q13, q6, d5[1]  \n"
                        "vmla.f32   q14, q7, d5[0]  \n"
                        "vmla.f32   q15, q7, d6[0]  \n"

                        "pld        [%7, #256]      \n"
                        "vld1.f32   {d0-d3}, [%7]!  \n" // r6
                        "vld1.f32   {d8[0]}, [%7]   \n"

                        "vmla.f32   q12, q8, d5[1]  \n"
                        "vmla.f32   q13, q8, d6[1]  \n"
                        "vmla.f32   q14, q9, d6[0]  \n"
                        "vmla.f32   q15, q9, d7[0]  \n"

                        "pld        [%8, #512]      \n"
                        "vldm       %8!, {d10-d17}  \n"

                        "vmla.f32   q12, q10, d6[1] \n"
                        "vmla.f32   q13, q10, d7[1] \n"
                        "vmla.f32   q14, q11, d7[0] \n"
                        "vmla.f32   q15, q11, d9[0] \n"

                        "pld        [%8, #384]      \n"
                        "vldm       %8!, {d18-d23}  \n"

                        "vmla.f32   q12, q5, d0[0]  \n"
                        "vmla.f32   q13, q5, d1[0]  \n"
                        "vmla.f32   q14, q6, d0[1]  \n"
                        "vmla.f32   q15, q6, d1[1]  \n"

                        "sub        %1, %1, #16     \n"
                        "sub        %2, %2, #16     \n"

                        "vmla.f32   q12, q7, d1[0]  \n"
                        "vmla.f32   q13, q7, d2[0]  \n"
                        "vmla.f32   q14, q8, d1[1]  \n"
                        "vmla.f32   q15, q8, d2[1]  \n"

                        "sub        %8, %8, #784    \n"

                        "vmla.f32   q12, q9, d2[0]  \n"
                        "vmla.f32   q13, q9, d3[0]  \n"
                        "vmla.f32   q14, q10, d2[1] \n"
                        "vmla.f32   q15, q10, d3[1] \n"

                        "sub        %3, %3, #16     \n"
                        "sub        %4, %4, #16     \n"

                        "vmla.f32   q12, q11, d3[0] \n"
                        "vmla.f32   q13, q11, d8[0] \n"

                        "sub        %5, %5, #16     \n"
                        "sub        %6, %6, #16     \n"

                        "vadd.f32   q14, q14, q12   \n"
                        "vadd.f32   q15, q15, q13   \n"

                        "sub        %7, %7, #16     \n"

                        "vst1.f32   {d28-d31}, [%0 :128]! \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2),      // %3
                        "=r"(r3),      // %4
                        "=r"(r4),      // %5
                        "=r"(r5),      // %6
                        "=r"(r6),      // %7
                        "=r"(kptr)     // %8
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "4"(r3),
                        "5"(r4),
                        "6"(r5),
                        "7"(r6),
                        "8"(kptr)
                        : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif // __aarch64__
                }
                for (; j < outw; j++)
                {
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%0, #128]       \n"
                        "ld1    {v16.4s}, [%0]              \n"

                        "prfm   pldl1keep, [%1, #256]       \n"
                        "ld1    {v0.4s, v1.4s}, [%1]        \n" // r0

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%8], #64 \n"

                        "fmul   v17.4s, v24.4s, v0.s[0]     \n"

                        "prfm   pldl1keep, [%8, #384]       \n"
                        "ld1    {v28.4s, v29.4s, v30.4s}, [%8], #48 \n"

                        "fmul   v18.4s, v25.4s, v0.s[1]     \n"
                        "fmul   v19.4s, v26.4s, v0.s[2]     \n"

                        "prfm   pldl1keep, [%2, #256]       \n"
                        "ld1    {v4.4s, v5.4s}, [%2]        \n" // r1

                        "fmla   v16.4s, v27.4s, v0.s[3]     \n"
                        "fmla   v17.4s, v28.4s, v1.s[0]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%8], #64 \n"

                        "fmla   v18.4s, v29.4s, v1.s[1]     \n"
                        "fmla   v19.4s, v30.4s, v1.s[2]     \n"

                        "prfm   pldl1keep, [%8, #384]       \n"
                        "ld1    {v28.4s, v29.4s, v30.4s}, [%8], #48 \n"

                        "fmla   v16.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v17.4s, v25.4s, v4.s[1]     \n"
                        "fmla   v18.4s, v26.4s, v4.s[2]     \n"

                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld1    {v0.4s, v1.4s}, [%3]        \n" // r2

                        "fmla   v19.4s, v27.4s, v4.s[3]     \n"
                        "fmla   v16.4s, v28.4s, v5.s[0]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%8], #64 \n"

                        "fmla   v17.4s, v29.4s, v5.s[1]     \n"
                        "fmla   v18.4s, v30.4s, v5.s[2]     \n"

                        "prfm   pldl1keep, [%8, #384]       \n"
                        "ld1    {v28.4s, v29.4s, v30.4s}, [%8], #48 \n"

                        "fmla   v19.4s, v24.4s, v0.s[0]     \n"
                        "fmla   v16.4s, v25.4s, v0.s[1]     \n"
                        "fmla   v17.4s, v26.4s, v0.s[2]     \n"

                        "prfm   pldl1keep, [%4, #256]       \n"
                        "ld1    {v4.4s, v5.4s}, [%4]        \n" // r3

                        "fmla   v18.4s, v27.4s, v0.s[3]     \n"
                        "fmla   v19.4s, v28.4s, v1.s[0]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%8], #64 \n"

                        "fmla   v16.4s, v29.4s, v1.s[1]     \n"
                        "fmla   v17.4s, v30.4s, v1.s[2]     \n"

                        "prfm   pldl1keep, [%8, #384]       \n"
                        "ld1    {v28.4s, v29.4s, v30.4s}, [%8], #48 \n"

                        "fmla   v18.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v19.4s, v25.4s, v4.s[1]     \n"
                        "fmla   v16.4s, v26.4s, v4.s[2]     \n"

                        "prfm   pldl1keep, [%5, #256]       \n"
                        "ld1    {v0.4s, v1.4s}, [%5]        \n" // r4

                        "fmla   v17.4s, v27.4s, v4.s[3]     \n"
                        "fmla   v18.4s, v28.4s, v5.s[0]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%8], #64 \n"

                        "fmla   v19.4s, v29.4s, v5.s[1]     \n"
                        "fmla   v16.4s, v30.4s, v5.s[2]     \n"

                        "prfm   pldl1keep, [%8, #384]       \n"
                        "ld1    {v28.4s, v29.4s, v30.4s}, [%8], #48 \n"

                        "fmla   v17.4s, v24.4s, v0.s[0]     \n"
                        "fmla   v18.4s, v25.4s, v0.s[1]     \n"
                        "fmla   v19.4s, v26.4s, v0.s[2]     \n"

                        "prfm   pldl1keep, [%6, #256]       \n"
                        "ld1    {v4.4s, v5.4s}, [%6]        \n" // r5

                        "fmla   v16.4s, v27.4s, v0.s[3]     \n"
                        "fmla   v17.4s, v28.4s, v1.s[0]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%8], #64 \n"

                        "fmla   v18.4s, v29.4s, v1.s[1]     \n"
                        "fmla   v19.4s, v30.4s, v1.s[2]     \n"

                        "prfm   pldl1keep, [%8, #384]       \n"
                        "ld1    {v28.4s, v29.4s, v30.4s}, [%8], #48 \n"

                        "fmla   v16.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v17.4s, v25.4s, v4.s[1]     \n"
                        "fmla   v18.4s, v26.4s, v4.s[2]     \n"

                        "prfm   pldl1keep, [%7, #256]       \n"
                        "ld1    {v0.4s, v1.4s}, [%7]        \n" // r6

                        "fmla   v19.4s, v27.4s, v4.s[3]     \n"
                        "fmla   v16.4s, v28.4s, v5.s[0]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%8], #64 \n"

                        "fmla   v17.4s, v29.4s, v5.s[1]     \n"
                        "fmla   v18.4s, v30.4s, v5.s[2]     \n"

                        "prfm   pldl1keep, [%8, #384]       \n"
                        "ld1    {v28.4s, v29.4s, v30.4s}, [%8], #48 \n"

                        "fmla   v19.4s, v24.4s, v0.s[0]     \n"
                        "fmla   v16.4s, v25.4s, v0.s[1]     \n"
                        "fmla   v17.4s, v26.4s, v0.s[2]     \n"

                        "add    %1, %1, #8                  \n"
                        "add    %2, %2, #8                  \n"

                        "fmla   v18.4s, v27.4s, v0.s[3]     \n"
                        "fmla   v19.4s, v28.4s, v1.s[0]     \n"
                        "fmla   v16.4s, v29.4s, v1.s[1]     \n"
                        "fmla   v17.4s, v30.4s, v1.s[2]     \n"

                        "add    %3, %3, #8                  \n"
                        "add    %4, %4, #8                  \n"

                        "fadd   v18.4s, v18.4s, v19.4s      \n"

                        "add    %5, %5, #8                  \n"

                        "fadd   v16.4s, v16.4s, v17.4s      \n"

                        "add    %6, %6, #8                  \n"
                        "add    %7, %7, #8                  \n"

                        "fadd   v16.4s, v16.4s, v18.4s      \n"

                        "sub    %8, %8, #784                \n"

                        "st1    {v16.4s}, [%0], #16         \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2),      // %3
                        "=r"(r3),      // %4
                        "=r"(r4),      // %5
                        "=r"(r5),      // %6
                        "=r"(r6),      // %7
                        "=r"(kptr)     // %8
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "4"(r3),
                        "5"(r4),
                        "6"(r5),
                        "7"(r6),
                        "8"(kptr)
                        : "memory", "v0", "v1", "v4", "v5", "v16", "v17", "v18", "v19", "v24", "v25", "v26", "v27", "v28", "v29", "v30");
#else  // __aarch64__
                    asm volatile(
                        "pld        [%0, #128]      \n"
                        "vld1.f32   {d8-d9}, [%0 :128] \n"

                        "pld        [%1, #256]      \n"
                        "vld1.f32   {d0-d3}, [%1]   \n" // r0

                        "pld        [%8, #512]      \n"
                        "vldm       %8!, {d16-d23}  \n"

                        "vmul.f32   q5, q8, d0[0]   \n"
                        "vmul.f32   q6, q9, d0[1]   \n"

                        "pld        [%8, #384]      \n"
                        "vldm       %8!, {d24-d29}  \n"

                        "vmul.f32   q7, q10, d1[0]  \n"
                        "vmla.f32   q4, q11, d1[1]  \n"

                        "pld        [%2, #256]      \n"
                        "vld1.f32   {d4-d7}, [%2]   \n" // r1

                        "vmla.f32   q5, q12, d2[0]  \n"

                        "pld        [%8, #512]      \n"
                        "vldm       %8!, {d16-d23}  \n"

                        "vmla.f32   q6, q13, d2[1]  \n"
                        "vmla.f32   q7, q14, d3[0]  \n"

                        "pld        [%8, #384]      \n"
                        "vldm       %8!, {d24-d29}  \n"

                        "vmla.f32   q4, q8, d4[0]   \n"
                        "vmla.f32   q5, q9, d4[1]   \n"
                        "vmla.f32   q6, q10, d5[0]  \n"

                        "pld        [%3, #256]      \n"
                        "vld1.f32   {d0-d3}, [%3]   \n" // r2

                        "vmla.f32   q7, q11, d5[1]  \n"
                        "vmla.f32   q4, q12, d6[0]  \n"

                        "pld        [%8, #512]      \n"
                        "vldm       %8!, {d16-d23}  \n"

                        "vmla.f32   q5, q13, d6[1]  \n"
                        "vmla.f32   q6, q14, d7[0]  \n"

                        "pld        [%8, #384]      \n"
                        "vldm       %8!, {d24-d29}  \n"

                        "vmla.f32   q7, q8, d0[0]   \n"
                        "vmla.f32   q4, q9, d0[1]   \n"
                        "vmla.f32   q5, q10, d1[0]  \n"

                        "pld        [%4, #256]      \n"
                        "vld1.f32   {d4-d7}, [%4]   \n" // r3

                        "vmla.f32   q6, q11, d1[1]  \n"
                        "vmla.f32   q7, q12, d2[0]  \n"

                        "pld        [%8, #512]      \n"
                        "vldm       %8!, {d16-d23}  \n"

                        "vmla.f32   q4, q13, d2[1]  \n"
                        "vmla.f32   q5, q14, d3[0]  \n"

                        "pld        [%8, #384]      \n"
                        "vldm       %8!, {d24-d29}  \n"

                        "vmla.f32   q6, q8, d4[0]   \n"
                        "vmla.f32   q7, q9, d4[1]   \n"
                        "vmla.f32   q4, q10, d5[0]  \n"

                        "pld        [%5, #256]      \n"
                        "vld1.f32   {d0-d3}, [%5]   \n" // r4

                        "vmla.f32   q5, q11, d5[1]  \n"
                        "vmla.f32   q6, q12, d6[0]  \n"

                        "pld        [%8, #512]      \n"
                        "vldm       %8!, {d16-d23}  \n"

                        "vmla.f32   q7, q13, d6[1]  \n"
                        "vmla.f32   q4, q14, d7[0]  \n"

                        "pld        [%8, #384]      \n"
                        "vldm       %8!, {d24-d29}  \n"

                        "vmla.f32   q5, q8, d0[0]   \n"
                        "vmla.f32   q6, q9, d0[1]   \n"
                        "vmla.f32   q7, q10, d1[0]  \n"

                        "pld        [%6, #256]      \n"
                        "vld1.f32   {d4-d7}, [%6]   \n" // r5

                        "vmla.f32   q4, q11, d1[1]  \n"
                        "vmla.f32   q5, q12, d2[0]  \n"

                        "pld        [%8, #512]      \n"
                        "vldm       %8!, {d16-d23}  \n"

                        "vmla.f32   q6, q13, d2[1]  \n"
                        "vmla.f32   q7, q14, d3[0]  \n"

                        "pld        [%8, #384]      \n"
                        "vldm       %8!, {d24-d29}  \n"

                        "vmla.f32   q4, q8, d4[0]   \n"
                        "vmla.f32   q5, q9, d4[1]   \n"
                        "vmla.f32   q6, q10, d5[0]  \n"

                        "pld        [%7, #256]      \n"
                        "vld1.f32   {d0-d3}, [%7]   \n" // r6

                        "vmla.f32   q7, q11, d5[1]  \n"
                        "vmla.f32   q4, q12, d6[0]  \n"

                        "pld        [%8, #512]      \n"
                        "vldm       %8!, {d16-d23}  \n"

                        "vmla.f32   q5, q13, d6[1]  \n"
                        "vmla.f32   q6, q14, d7[0]  \n"

                        "pld        [%8, #384]      \n"
                        "vldm       %8!, {d24-d29}  \n"

                        "vmla.f32   q7, q8, d0[0]   \n"
                        "vmla.f32   q4, q9, d0[1]   \n"

                        "add        %1, %1, #8      \n"
                        "add        %2, %2, #8      \n"

                        "vmla.f32   q5, q10, d1[0]  \n"
                        "vmla.f32   q6, q11, d1[1]  \n"

                        "sub        %8, %8, #784    \n"

                        "vmla.f32   q7, q12, d2[0]  \n"
                        "vmla.f32   q4, q13, d2[1]  \n"
                        "vmla.f32   q5, q14, d3[0]  \n"

                        "add        %3, %3, #8      \n"
                        "add        %4, %4, #8      \n"

                        "vadd.f32   q6, q6, q7      \n"

                        "add        %5, %5, #8      \n"

                        "vadd.f32   q4, q4, q5      \n"

                        "add        %6, %6, #8      \n"

                        "vadd.f32   q4, q4, q6      \n"

                        "add        %7, %7, #8      \n"

                        "vst1.f32   {d8-d9}, [%0 :128]! \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2),      // %3
                        "=r"(r3),      // %4
                        "=r"(r4),      // %5
                        "=r"(r5),      // %6
                        "=r"(r6),      // %7
                        "=r"(kptr)     // %8
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "4"(r3),
                        "5"(r4),
                        "6"(r5),
                        "7"(r6),
                        "8"(kptr)
                        : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14");
#endif // __aarch64__
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
                r3 += tailstep;
                r4 += tailstep;
                r5 += tailstep;
                r6 += tailstep;
            }
        }
    }
}
