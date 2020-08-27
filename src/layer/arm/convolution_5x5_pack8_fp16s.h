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

static void conv5x5s1_pack8_fp16sa_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int inch = bottom_blob.c;
    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const __fp16* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        float16x8_t _bias0 = bias ? vld1q_f16(bias + p * 8) : vdupq_n_f16(0.f);
        out0.fill(_bias0);

        int q = 0;
        for (; q < inch; q++)
        {
            __fp16* outptr0 = out0;

            const Mat img0 = bottom_blob.channel(q);

            const __fp16* r0 = img0.row<const __fp16>(0);
            const __fp16* r1 = img0.row<const __fp16>(1);
            const __fp16* r2 = img0.row<const __fp16>(2);
            const __fp16* r3 = img0.row<const __fp16>(3);
            const __fp16* r4 = img0.row<const __fp16>(4);

            const __fp16* kptr = kernel.channel(p).row<const __fp16>(q);

            int i = 0;
            for (; i < outh; i++)
            {
                int j = 0;
                for (; j + 3 < outw; j += 4)
                {
                    asm volatile(
                        "prfm   pldl1keep, [%0, #512]       \n"
                        "ld1    {v28.8h, v29.8h, v30.8h, v31.8h}, [%0] \n" // sum0 sum1 sum2 sum3

                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%1], #64 \n" // r00 r01 r02 r03

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v0.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v1.h[0]     \n"
                        "fmla   v30.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v31.8h, v16.8h, v3.h[0]     \n"
                        "fmla   v28.8h, v17.8h, v0.h[1]     \n"
                        "fmla   v29.8h, v17.8h, v1.h[1]     \n"
                        "fmla   v30.8h, v17.8h, v2.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v3.h[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v18.8h, v0.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v1.h[2]     \n"
                        "fmla   v30.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v31.8h, v18.8h, v3.h[2]     \n"
                        "fmla   v28.8h, v19.8h, v0.h[3]     \n"
                        "fmla   v29.8h, v19.8h, v1.h[3]     \n"
                        "fmla   v30.8h, v19.8h, v2.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v3.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v0.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v1.h[4]     \n"
                        "fmla   v30.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v31.8h, v20.8h, v3.h[4]     \n"
                        "fmla   v28.8h, v21.8h, v0.h[5]     \n"
                        "fmla   v29.8h, v21.8h, v1.h[5]     \n"
                        "fmla   v30.8h, v21.8h, v2.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v3.h[5]     \n"

                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%1] \n" // r04 r05 r06 r07

                        "fmla   v28.8h, v22.8h, v0.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v1.h[6]     \n"
                        "fmla   v30.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v31.8h, v22.8h, v3.h[6]     \n"
                        "fmla   v28.8h, v23.8h, v0.h[7]     \n"
                        "fmla   v29.8h, v23.8h, v1.h[7]     \n"
                        "fmla   v30.8h, v23.8h, v2.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v3.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v1.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v30.8h, v16.8h, v3.h[0]     \n"
                        "fmla   v31.8h, v16.8h, v4.h[0]     \n"
                        "fmla   v28.8h, v17.8h, v1.h[1]     \n"
                        "fmla   v29.8h, v17.8h, v2.h[1]     \n"
                        "fmla   v30.8h, v17.8h, v3.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v4.h[1]     \n"
                        "fmla   v28.8h, v18.8h, v1.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v30.8h, v18.8h, v3.h[2]     \n"
                        "fmla   v31.8h, v18.8h, v4.h[2]     \n"
                        "fmla   v28.8h, v19.8h, v1.h[3]     \n"
                        "fmla   v29.8h, v19.8h, v2.h[3]     \n"
                        "fmla   v30.8h, v19.8h, v3.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v4.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v1.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v30.8h, v20.8h, v3.h[4]     \n"
                        "fmla   v31.8h, v20.8h, v4.h[4]     \n"
                        "fmla   v28.8h, v21.8h, v1.h[5]     \n"
                        "fmla   v29.8h, v21.8h, v2.h[5]     \n"
                        "fmla   v30.8h, v21.8h, v3.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v4.h[5]     \n"
                        "fmla   v28.8h, v22.8h, v1.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v30.8h, v22.8h, v3.h[6]     \n"
                        "fmla   v31.8h, v22.8h, v4.h[6]     \n"
                        "fmla   v28.8h, v23.8h, v1.h[7]     \n"
                        "fmla   v29.8h, v23.8h, v2.h[7]     \n"
                        "fmla   v30.8h, v23.8h, v3.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v4.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v3.h[0]     \n"
                        "fmla   v30.8h, v16.8h, v4.h[0]     \n"
                        "fmla   v31.8h, v16.8h, v5.h[0]     \n"
                        "fmla   v28.8h, v17.8h, v2.h[1]     \n"
                        "fmla   v29.8h, v17.8h, v3.h[1]     \n"
                        "fmla   v30.8h, v17.8h, v4.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v5.h[1]     \n"
                        "fmla   v28.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v3.h[2]     \n"
                        "fmla   v30.8h, v18.8h, v4.h[2]     \n"
                        "fmla   v31.8h, v18.8h, v5.h[2]     \n"
                        "fmla   v28.8h, v19.8h, v2.h[3]     \n"
                        "fmla   v29.8h, v19.8h, v3.h[3]     \n"
                        "fmla   v30.8h, v19.8h, v4.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v5.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v3.h[4]     \n"
                        "fmla   v30.8h, v20.8h, v4.h[4]     \n"
                        "fmla   v31.8h, v20.8h, v5.h[4]     \n"
                        "fmla   v28.8h, v21.8h, v2.h[5]     \n"
                        "fmla   v29.8h, v21.8h, v3.h[5]     \n"
                        "fmla   v30.8h, v21.8h, v4.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v5.h[5]     \n"
                        "fmla   v28.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v3.h[6]     \n"
                        "fmla   v30.8h, v22.8h, v4.h[6]     \n"
                        "fmla   v31.8h, v22.8h, v5.h[6]     \n"
                        "fmla   v28.8h, v23.8h, v2.h[7]     \n"
                        "fmla   v29.8h, v23.8h, v3.h[7]     \n"
                        "fmla   v30.8h, v23.8h, v4.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v5.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v3.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v4.h[0]     \n"
                        "fmla   v30.8h, v16.8h, v5.h[0]     \n"
                        "fmla   v31.8h, v16.8h, v6.h[0]     \n"
                        "fmla   v28.8h, v17.8h, v3.h[1]     \n"
                        "fmla   v29.8h, v17.8h, v4.h[1]     \n"
                        "fmla   v30.8h, v17.8h, v5.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v6.h[1]     \n"
                        "fmla   v28.8h, v18.8h, v3.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v4.h[2]     \n"
                        "fmla   v30.8h, v18.8h, v5.h[2]     \n"
                        "fmla   v31.8h, v18.8h, v6.h[2]     \n"
                        "fmla   v28.8h, v19.8h, v3.h[3]     \n"
                        "fmla   v29.8h, v19.8h, v4.h[3]     \n"
                        "fmla   v30.8h, v19.8h, v5.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v6.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v3.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v4.h[4]     \n"
                        "fmla   v30.8h, v20.8h, v5.h[4]     \n"
                        "fmla   v31.8h, v20.8h, v6.h[4]     \n"
                        "fmla   v28.8h, v21.8h, v3.h[5]     \n"
                        "fmla   v29.8h, v21.8h, v4.h[5]     \n"
                        "fmla   v30.8h, v21.8h, v5.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v6.h[5]     \n"
                        "fmla   v28.8h, v22.8h, v3.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v4.h[6]     \n"
                        "fmla   v30.8h, v22.8h, v5.h[6]     \n"
                        "fmla   v31.8h, v22.8h, v6.h[6]     \n"
                        "fmla   v28.8h, v23.8h, v3.h[7]     \n"
                        "fmla   v29.8h, v23.8h, v4.h[7]     \n"
                        "fmla   v30.8h, v23.8h, v5.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v6.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v4.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v5.h[0]     \n"
                        "fmla   v30.8h, v16.8h, v6.h[0]     \n"
                        "fmla   v31.8h, v16.8h, v7.h[0]     \n"
                        "fmla   v28.8h, v17.8h, v4.h[1]     \n"
                        "fmla   v29.8h, v17.8h, v5.h[1]     \n"
                        "fmla   v30.8h, v17.8h, v6.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v7.h[1]     \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v8.8h, v9.8h, v10.8h, v11.8h}, [%2], #64 \n" // r10 r11 r12 r13

                        "fmla   v28.8h, v18.8h, v4.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v5.h[2]     \n"
                        "fmla   v30.8h, v18.8h, v6.h[2]     \n"
                        "fmla   v31.8h, v18.8h, v7.h[2]     \n"
                        "fmla   v28.8h, v19.8h, v4.h[3]     \n"
                        "fmla   v29.8h, v19.8h, v5.h[3]     \n"
                        "fmla   v30.8h, v19.8h, v6.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v7.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v4.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v5.h[4]     \n"
                        "fmla   v30.8h, v20.8h, v6.h[4]     \n"
                        "fmla   v31.8h, v20.8h, v7.h[4]     \n"
                        "fmla   v28.8h, v21.8h, v4.h[5]     \n"
                        "fmla   v29.8h, v21.8h, v5.h[5]     \n"
                        "fmla   v30.8h, v21.8h, v6.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v7.h[5]     \n"
                        "fmla   v28.8h, v22.8h, v4.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v5.h[6]     \n"
                        "fmla   v30.8h, v22.8h, v6.h[6]     \n"
                        "fmla   v31.8h, v22.8h, v7.h[6]     \n"
                        "fmla   v28.8h, v23.8h, v4.h[7]     \n"
                        "fmla   v29.8h, v23.8h, v5.h[7]     \n"
                        "fmla   v30.8h, v23.8h, v6.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v7.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v8.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v9.h[0]     \n"
                        "fmla   v30.8h, v16.8h, v10.h[0]    \n"
                        "fmla   v31.8h, v16.8h, v11.h[0]    \n"
                        "fmla   v28.8h, v17.8h, v8.h[1]     \n"
                        "fmla   v29.8h, v17.8h, v9.h[1]     \n"
                        "fmla   v30.8h, v17.8h, v10.h[1]    \n"
                        "fmla   v31.8h, v17.8h, v11.h[1]    \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v12.8h, v13.8h, v14.8h, v15.8h}, [%2] \n" // r14 r15 r16 r17

                        "fmla   v28.8h, v18.8h, v8.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v9.h[2]     \n"
                        "fmla   v30.8h, v18.8h, v10.h[2]    \n"
                        "fmla   v31.8h, v18.8h, v11.h[2]    \n"
                        "fmla   v28.8h, v19.8h, v8.h[3]     \n"
                        "fmla   v29.8h, v19.8h, v9.h[3]     \n"
                        "fmla   v30.8h, v19.8h, v10.h[3]    \n"
                        "fmla   v31.8h, v19.8h, v11.h[3]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v8.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v9.h[4]     \n"
                        "fmla   v30.8h, v20.8h, v10.h[4]    \n"
                        "fmla   v31.8h, v20.8h, v11.h[4]    \n"
                        "fmla   v28.8h, v21.8h, v8.h[5]     \n"
                        "fmla   v29.8h, v21.8h, v9.h[5]     \n"
                        "fmla   v30.8h, v21.8h, v10.h[5]    \n"
                        "fmla   v31.8h, v21.8h, v11.h[5]    \n"
                        "fmla   v28.8h, v22.8h, v8.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v9.h[6]     \n"
                        "fmla   v30.8h, v22.8h, v10.h[6]    \n"
                        "fmla   v31.8h, v22.8h, v11.h[6]    \n"
                        "fmla   v28.8h, v23.8h, v8.h[7]     \n"
                        "fmla   v29.8h, v23.8h, v9.h[7]     \n"
                        "fmla   v30.8h, v23.8h, v10.h[7]    \n"
                        "fmla   v31.8h, v23.8h, v11.h[7]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v9.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v10.h[0]    \n"
                        "fmla   v30.8h, v16.8h, v11.h[0]    \n"
                        "fmla   v31.8h, v16.8h, v12.h[0]    \n"
                        "fmla   v28.8h, v17.8h, v9.h[1]     \n"
                        "fmla   v29.8h, v17.8h, v10.h[1]    \n"
                        "fmla   v30.8h, v17.8h, v11.h[1]    \n"
                        "fmla   v31.8h, v17.8h, v12.h[1]    \n"
                        "fmla   v28.8h, v18.8h, v9.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v10.h[2]    \n"
                        "fmla   v30.8h, v18.8h, v11.h[2]    \n"
                        "fmla   v31.8h, v18.8h, v12.h[2]    \n"
                        "fmla   v28.8h, v19.8h, v9.h[3]     \n"
                        "fmla   v29.8h, v19.8h, v10.h[3]    \n"
                        "fmla   v30.8h, v19.8h, v11.h[3]    \n"
                        "fmla   v31.8h, v19.8h, v12.h[3]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v9.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v10.h[4]    \n"
                        "fmla   v30.8h, v20.8h, v11.h[4]    \n"
                        "fmla   v31.8h, v20.8h, v12.h[4]    \n"
                        "fmla   v28.8h, v21.8h, v9.h[5]     \n"
                        "fmla   v29.8h, v21.8h, v10.h[5]    \n"
                        "fmla   v30.8h, v21.8h, v11.h[5]    \n"
                        "fmla   v31.8h, v21.8h, v12.h[5]    \n"
                        "fmla   v28.8h, v22.8h, v9.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v10.h[6]    \n"
                        "fmla   v30.8h, v22.8h, v11.h[6]    \n"
                        "fmla   v31.8h, v22.8h, v12.h[6]    \n"
                        "fmla   v28.8h, v23.8h, v9.h[7]     \n"
                        "fmla   v29.8h, v23.8h, v10.h[7]    \n"
                        "fmla   v30.8h, v23.8h, v11.h[7]    \n"
                        "fmla   v31.8h, v23.8h, v12.h[7]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v10.h[0]    \n"
                        "fmla   v29.8h, v16.8h, v11.h[0]    \n"
                        "fmla   v30.8h, v16.8h, v12.h[0]    \n"
                        "fmla   v31.8h, v16.8h, v13.h[0]    \n"
                        "fmla   v28.8h, v17.8h, v10.h[1]    \n"
                        "fmla   v29.8h, v17.8h, v11.h[1]    \n"
                        "fmla   v30.8h, v17.8h, v12.h[1]    \n"
                        "fmla   v31.8h, v17.8h, v13.h[1]    \n"
                        "fmla   v28.8h, v18.8h, v10.h[2]    \n"
                        "fmla   v29.8h, v18.8h, v11.h[2]    \n"
                        "fmla   v30.8h, v18.8h, v12.h[2]    \n"
                        "fmla   v31.8h, v18.8h, v13.h[2]    \n"
                        "fmla   v28.8h, v19.8h, v10.h[3]    \n"
                        "fmla   v29.8h, v19.8h, v11.h[3]    \n"
                        "fmla   v30.8h, v19.8h, v12.h[3]    \n"
                        "fmla   v31.8h, v19.8h, v13.h[3]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v10.h[4]    \n"
                        "fmla   v29.8h, v20.8h, v11.h[4]    \n"
                        "fmla   v30.8h, v20.8h, v12.h[4]    \n"
                        "fmla   v31.8h, v20.8h, v13.h[4]    \n"
                        "fmla   v28.8h, v21.8h, v10.h[5]    \n"
                        "fmla   v29.8h, v21.8h, v11.h[5]    \n"
                        "fmla   v30.8h, v21.8h, v12.h[5]    \n"
                        "fmla   v31.8h, v21.8h, v13.h[5]    \n"
                        "fmla   v28.8h, v22.8h, v10.h[6]    \n"
                        "fmla   v29.8h, v22.8h, v11.h[6]    \n"
                        "fmla   v30.8h, v22.8h, v12.h[6]    \n"
                        "fmla   v31.8h, v22.8h, v13.h[6]    \n"
                        "fmla   v28.8h, v23.8h, v10.h[7]    \n"
                        "fmla   v29.8h, v23.8h, v11.h[7]    \n"
                        "fmla   v30.8h, v23.8h, v12.h[7]    \n"
                        "fmla   v31.8h, v23.8h, v13.h[7]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v11.h[0]    \n"
                        "fmla   v29.8h, v16.8h, v12.h[0]    \n"
                        "fmla   v30.8h, v16.8h, v13.h[0]    \n"
                        "fmla   v31.8h, v16.8h, v14.h[0]    \n"
                        "fmla   v28.8h, v17.8h, v11.h[1]    \n"
                        "fmla   v29.8h, v17.8h, v12.h[1]    \n"
                        "fmla   v30.8h, v17.8h, v13.h[1]    \n"
                        "fmla   v31.8h, v17.8h, v14.h[1]    \n"
                        "fmla   v28.8h, v18.8h, v11.h[2]    \n"
                        "fmla   v29.8h, v18.8h, v12.h[2]    \n"
                        "fmla   v30.8h, v18.8h, v13.h[2]    \n"
                        "fmla   v31.8h, v18.8h, v14.h[2]    \n"
                        "fmla   v28.8h, v19.8h, v11.h[3]    \n"
                        "fmla   v29.8h, v19.8h, v12.h[3]    \n"
                        "fmla   v30.8h, v19.8h, v13.h[3]    \n"
                        "fmla   v31.8h, v19.8h, v14.h[3]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v11.h[4]    \n"
                        "fmla   v29.8h, v20.8h, v12.h[4]    \n"
                        "fmla   v30.8h, v20.8h, v13.h[4]    \n"
                        "fmla   v31.8h, v20.8h, v14.h[4]    \n"
                        "fmla   v28.8h, v21.8h, v11.h[5]    \n"
                        "fmla   v29.8h, v21.8h, v12.h[5]    \n"
                        "fmla   v30.8h, v21.8h, v13.h[5]    \n"
                        "fmla   v31.8h, v21.8h, v14.h[5]    \n"
                        "fmla   v28.8h, v22.8h, v11.h[6]    \n"
                        "fmla   v29.8h, v22.8h, v12.h[6]    \n"
                        "fmla   v30.8h, v22.8h, v13.h[6]    \n"
                        "fmla   v31.8h, v22.8h, v14.h[6]    \n"
                        "fmla   v28.8h, v23.8h, v11.h[7]    \n"
                        "fmla   v29.8h, v23.8h, v12.h[7]    \n"
                        "fmla   v30.8h, v23.8h, v13.h[7]    \n"
                        "fmla   v31.8h, v23.8h, v14.h[7]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v12.h[0]    \n"
                        "fmla   v29.8h, v16.8h, v13.h[0]    \n"
                        "fmla   v30.8h, v16.8h, v14.h[0]    \n"
                        "fmla   v31.8h, v16.8h, v15.h[0]    \n"
                        "fmla   v28.8h, v17.8h, v12.h[1]    \n"
                        "fmla   v29.8h, v17.8h, v13.h[1]    \n"
                        "fmla   v30.8h, v17.8h, v14.h[1]    \n"
                        "fmla   v31.8h, v17.8h, v15.h[1]    \n"

                        "prfm   pldl1keep, [%3, #512]       \n"
                        "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%3], #64 \n" // r20 r21 r22 r23

                        "fmla   v28.8h, v18.8h, v12.h[2]    \n"
                        "fmla   v29.8h, v18.8h, v13.h[2]    \n"
                        "fmla   v30.8h, v18.8h, v14.h[2]    \n"
                        "fmla   v31.8h, v18.8h, v15.h[2]    \n"
                        "fmla   v28.8h, v19.8h, v12.h[3]    \n"
                        "fmla   v29.8h, v19.8h, v13.h[3]    \n"
                        "fmla   v30.8h, v19.8h, v14.h[3]    \n"
                        "fmla   v31.8h, v19.8h, v15.h[3]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v12.h[4]    \n"
                        "fmla   v29.8h, v20.8h, v13.h[4]    \n"
                        "fmla   v30.8h, v20.8h, v14.h[4]    \n"
                        "fmla   v31.8h, v20.8h, v15.h[4]    \n"
                        "fmla   v28.8h, v21.8h, v12.h[5]    \n"
                        "fmla   v29.8h, v21.8h, v13.h[5]    \n"
                        "fmla   v30.8h, v21.8h, v14.h[5]    \n"
                        "fmla   v31.8h, v21.8h, v15.h[5]    \n"
                        "fmla   v28.8h, v22.8h, v12.h[6]    \n"
                        "fmla   v29.8h, v22.8h, v13.h[6]    \n"
                        "fmla   v30.8h, v22.8h, v14.h[6]    \n"
                        "fmla   v31.8h, v22.8h, v15.h[6]    \n"
                        "fmla   v28.8h, v23.8h, v12.h[7]    \n"
                        "fmla   v29.8h, v23.8h, v13.h[7]    \n"
                        "fmla   v30.8h, v23.8h, v14.h[7]    \n"
                        "fmla   v31.8h, v23.8h, v15.h[7]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v0.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v1.h[0]     \n"
                        "fmla   v30.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v31.8h, v16.8h, v3.h[0]     \n"
                        "fmla   v28.8h, v17.8h, v0.h[1]     \n"
                        "fmla   v29.8h, v17.8h, v1.h[1]     \n"
                        "fmla   v30.8h, v17.8h, v2.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v3.h[1]     \n"

                        "prfm   pldl1keep, [%3, #512]       \n"
                        "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%3] \n" // r24 r25 r26 r27

                        "fmla   v28.8h, v18.8h, v0.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v1.h[2]     \n"
                        "fmla   v30.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v31.8h, v18.8h, v3.h[2]     \n"
                        "fmla   v28.8h, v19.8h, v0.h[3]     \n"
                        "fmla   v29.8h, v19.8h, v1.h[3]     \n"
                        "fmla   v30.8h, v19.8h, v2.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v3.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v0.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v1.h[4]     \n"
                        "fmla   v30.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v31.8h, v20.8h, v3.h[4]     \n"
                        "fmla   v28.8h, v21.8h, v0.h[5]     \n"
                        "fmla   v29.8h, v21.8h, v1.h[5]     \n"
                        "fmla   v30.8h, v21.8h, v2.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v3.h[5]     \n"
                        "fmla   v28.8h, v22.8h, v0.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v1.h[6]     \n"
                        "fmla   v30.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v31.8h, v22.8h, v3.h[6]     \n"
                        "fmla   v28.8h, v23.8h, v0.h[7]     \n"
                        "fmla   v29.8h, v23.8h, v1.h[7]     \n"
                        "fmla   v30.8h, v23.8h, v2.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v3.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v1.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v30.8h, v16.8h, v3.h[0]     \n"
                        "fmla   v31.8h, v16.8h, v4.h[0]     \n"
                        "fmla   v28.8h, v17.8h, v1.h[1]     \n"
                        "fmla   v29.8h, v17.8h, v2.h[1]     \n"
                        "fmla   v30.8h, v17.8h, v3.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v4.h[1]     \n"
                        "fmla   v28.8h, v18.8h, v1.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v30.8h, v18.8h, v3.h[2]     \n"
                        "fmla   v31.8h, v18.8h, v4.h[2]     \n"
                        "fmla   v28.8h, v19.8h, v1.h[3]     \n"
                        "fmla   v29.8h, v19.8h, v2.h[3]     \n"
                        "fmla   v30.8h, v19.8h, v3.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v4.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v1.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v30.8h, v20.8h, v3.h[4]     \n"
                        "fmla   v31.8h, v20.8h, v4.h[4]     \n"
                        "fmla   v28.8h, v21.8h, v1.h[5]     \n"
                        "fmla   v29.8h, v21.8h, v2.h[5]     \n"
                        "fmla   v30.8h, v21.8h, v3.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v4.h[5]     \n"
                        "fmla   v28.8h, v22.8h, v1.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v30.8h, v22.8h, v3.h[6]     \n"
                        "fmla   v31.8h, v22.8h, v4.h[6]     \n"
                        "fmla   v28.8h, v23.8h, v1.h[7]     \n"
                        "fmla   v29.8h, v23.8h, v2.h[7]     \n"
                        "fmla   v30.8h, v23.8h, v3.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v4.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v3.h[0]     \n"
                        "fmla   v30.8h, v16.8h, v4.h[0]     \n"
                        "fmla   v31.8h, v16.8h, v5.h[0]     \n"
                        "fmla   v28.8h, v17.8h, v2.h[1]     \n"
                        "fmla   v29.8h, v17.8h, v3.h[1]     \n"
                        "fmla   v30.8h, v17.8h, v4.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v5.h[1]     \n"
                        "fmla   v28.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v3.h[2]     \n"
                        "fmla   v30.8h, v18.8h, v4.h[2]     \n"
                        "fmla   v31.8h, v18.8h, v5.h[2]     \n"
                        "fmla   v28.8h, v19.8h, v2.h[3]     \n"
                        "fmla   v29.8h, v19.8h, v3.h[3]     \n"
                        "fmla   v30.8h, v19.8h, v4.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v5.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v3.h[4]     \n"
                        "fmla   v30.8h, v20.8h, v4.h[4]     \n"
                        "fmla   v31.8h, v20.8h, v5.h[4]     \n"
                        "fmla   v28.8h, v21.8h, v2.h[5]     \n"
                        "fmla   v29.8h, v21.8h, v3.h[5]     \n"
                        "fmla   v30.8h, v21.8h, v4.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v5.h[5]     \n"
                        "fmla   v28.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v3.h[6]     \n"
                        "fmla   v30.8h, v22.8h, v4.h[6]     \n"
                        "fmla   v31.8h, v22.8h, v5.h[6]     \n"
                        "fmla   v28.8h, v23.8h, v2.h[7]     \n"
                        "fmla   v29.8h, v23.8h, v3.h[7]     \n"
                        "fmla   v30.8h, v23.8h, v4.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v5.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v3.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v4.h[0]     \n"
                        "fmla   v30.8h, v16.8h, v5.h[0]     \n"
                        "fmla   v31.8h, v16.8h, v6.h[0]     \n"
                        "fmla   v28.8h, v17.8h, v3.h[1]     \n"
                        "fmla   v29.8h, v17.8h, v4.h[1]     \n"
                        "fmla   v30.8h, v17.8h, v5.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v6.h[1]     \n"
                        "fmla   v28.8h, v18.8h, v3.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v4.h[2]     \n"
                        "fmla   v30.8h, v18.8h, v5.h[2]     \n"
                        "fmla   v31.8h, v18.8h, v6.h[2]     \n"
                        "fmla   v28.8h, v19.8h, v3.h[3]     \n"
                        "fmla   v29.8h, v19.8h, v4.h[3]     \n"
                        "fmla   v30.8h, v19.8h, v5.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v6.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v3.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v4.h[4]     \n"
                        "fmla   v30.8h, v20.8h, v5.h[4]     \n"
                        "fmla   v31.8h, v20.8h, v6.h[4]     \n"
                        "fmla   v28.8h, v21.8h, v3.h[5]     \n"
                        "fmla   v29.8h, v21.8h, v4.h[5]     \n"
                        "fmla   v30.8h, v21.8h, v5.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v6.h[5]     \n"
                        "fmla   v28.8h, v22.8h, v3.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v4.h[6]     \n"
                        "fmla   v30.8h, v22.8h, v5.h[6]     \n"
                        "fmla   v31.8h, v22.8h, v6.h[6]     \n"
                        "fmla   v28.8h, v23.8h, v3.h[7]     \n"
                        "fmla   v29.8h, v23.8h, v4.h[7]     \n"
                        "fmla   v30.8h, v23.8h, v5.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v6.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v4.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v5.h[0]     \n"
                        "fmla   v30.8h, v16.8h, v6.h[0]     \n"
                        "fmla   v31.8h, v16.8h, v7.h[0]     \n"
                        "fmla   v28.8h, v17.8h, v4.h[1]     \n"
                        "fmla   v29.8h, v17.8h, v5.h[1]     \n"
                        "fmla   v30.8h, v17.8h, v6.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v7.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v8.8h, v9.8h, v10.8h, v11.8h}, [%4], #64 \n" // r30 r31 r32 r33

                        "fmla   v28.8h, v18.8h, v4.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v5.h[2]     \n"
                        "fmla   v30.8h, v18.8h, v6.h[2]     \n"
                        "fmla   v31.8h, v18.8h, v7.h[2]     \n"
                        "fmla   v28.8h, v19.8h, v4.h[3]     \n"
                        "fmla   v29.8h, v19.8h, v5.h[3]     \n"
                        "fmla   v30.8h, v19.8h, v6.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v7.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v4.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v5.h[4]     \n"
                        "fmla   v30.8h, v20.8h, v6.h[4]     \n"
                        "fmla   v31.8h, v20.8h, v7.h[4]     \n"
                        "fmla   v28.8h, v21.8h, v4.h[5]     \n"
                        "fmla   v29.8h, v21.8h, v5.h[5]     \n"
                        "fmla   v30.8h, v21.8h, v6.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v7.h[5]     \n"
                        "fmla   v28.8h, v22.8h, v4.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v5.h[6]     \n"
                        "fmla   v30.8h, v22.8h, v6.h[6]     \n"
                        "fmla   v31.8h, v22.8h, v7.h[6]     \n"
                        "fmla   v28.8h, v23.8h, v4.h[7]     \n"
                        "fmla   v29.8h, v23.8h, v5.h[7]     \n"
                        "fmla   v30.8h, v23.8h, v6.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v7.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v8.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v9.h[0]     \n"
                        "fmla   v30.8h, v16.8h, v10.h[0]    \n"
                        "fmla   v31.8h, v16.8h, v11.h[0]    \n"
                        "fmla   v28.8h, v17.8h, v8.h[1]     \n"
                        "fmla   v29.8h, v17.8h, v9.h[1]     \n"
                        "fmla   v30.8h, v17.8h, v10.h[1]    \n"
                        "fmla   v31.8h, v17.8h, v11.h[1]    \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v12.8h, v13.8h, v14.8h, v15.8h}, [%4] \n" // r34 r35 r36 r37

                        "fmla   v28.8h, v18.8h, v8.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v9.h[2]     \n"
                        "fmla   v30.8h, v18.8h, v10.h[2]    \n"
                        "fmla   v31.8h, v18.8h, v11.h[2]    \n"
                        "fmla   v28.8h, v19.8h, v8.h[3]     \n"
                        "fmla   v29.8h, v19.8h, v9.h[3]     \n"
                        "fmla   v30.8h, v19.8h, v10.h[3]    \n"
                        "fmla   v31.8h, v19.8h, v11.h[3]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v8.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v9.h[4]     \n"
                        "fmla   v30.8h, v20.8h, v10.h[4]    \n"
                        "fmla   v31.8h, v20.8h, v11.h[4]    \n"
                        "fmla   v28.8h, v21.8h, v8.h[5]     \n"
                        "fmla   v29.8h, v21.8h, v9.h[5]     \n"
                        "fmla   v30.8h, v21.8h, v10.h[5]    \n"
                        "fmla   v31.8h, v21.8h, v11.h[5]    \n"
                        "fmla   v28.8h, v22.8h, v8.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v9.h[6]     \n"
                        "fmla   v30.8h, v22.8h, v10.h[6]    \n"
                        "fmla   v31.8h, v22.8h, v11.h[6]    \n"
                        "fmla   v28.8h, v23.8h, v8.h[7]     \n"
                        "fmla   v29.8h, v23.8h, v9.h[7]     \n"
                        "fmla   v30.8h, v23.8h, v10.h[7]    \n"
                        "fmla   v31.8h, v23.8h, v11.h[7]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v9.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v10.h[0]    \n"
                        "fmla   v30.8h, v16.8h, v11.h[0]    \n"
                        "fmla   v31.8h, v16.8h, v12.h[0]    \n"
                        "fmla   v28.8h, v17.8h, v9.h[1]     \n"
                        "fmla   v29.8h, v17.8h, v10.h[1]    \n"
                        "fmla   v30.8h, v17.8h, v11.h[1]    \n"
                        "fmla   v31.8h, v17.8h, v12.h[1]    \n"
                        "fmla   v28.8h, v18.8h, v9.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v10.h[2]    \n"
                        "fmla   v30.8h, v18.8h, v11.h[2]    \n"
                        "fmla   v31.8h, v18.8h, v12.h[2]    \n"
                        "fmla   v28.8h, v19.8h, v9.h[3]     \n"
                        "fmla   v29.8h, v19.8h, v10.h[3]    \n"
                        "fmla   v30.8h, v19.8h, v11.h[3]    \n"
                        "fmla   v31.8h, v19.8h, v12.h[3]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v9.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v10.h[4]    \n"
                        "fmla   v30.8h, v20.8h, v11.h[4]    \n"
                        "fmla   v31.8h, v20.8h, v12.h[4]    \n"
                        "fmla   v28.8h, v21.8h, v9.h[5]     \n"
                        "fmla   v29.8h, v21.8h, v10.h[5]    \n"
                        "fmla   v30.8h, v21.8h, v11.h[5]    \n"
                        "fmla   v31.8h, v21.8h, v12.h[5]    \n"
                        "fmla   v28.8h, v22.8h, v9.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v10.h[6]    \n"
                        "fmla   v30.8h, v22.8h, v11.h[6]    \n"
                        "fmla   v31.8h, v22.8h, v12.h[6]    \n"
                        "fmla   v28.8h, v23.8h, v9.h[7]     \n"
                        "fmla   v29.8h, v23.8h, v10.h[7]    \n"
                        "fmla   v30.8h, v23.8h, v11.h[7]    \n"
                        "fmla   v31.8h, v23.8h, v12.h[7]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v10.h[0]    \n"
                        "fmla   v29.8h, v16.8h, v11.h[0]    \n"
                        "fmla   v30.8h, v16.8h, v12.h[0]    \n"
                        "fmla   v31.8h, v16.8h, v13.h[0]    \n"
                        "fmla   v28.8h, v17.8h, v10.h[1]    \n"
                        "fmla   v29.8h, v17.8h, v11.h[1]    \n"
                        "fmla   v30.8h, v17.8h, v12.h[1]    \n"
                        "fmla   v31.8h, v17.8h, v13.h[1]    \n"
                        "fmla   v28.8h, v18.8h, v10.h[2]    \n"
                        "fmla   v29.8h, v18.8h, v11.h[2]    \n"
                        "fmla   v30.8h, v18.8h, v12.h[2]    \n"
                        "fmla   v31.8h, v18.8h, v13.h[2]    \n"
                        "fmla   v28.8h, v19.8h, v10.h[3]    \n"
                        "fmla   v29.8h, v19.8h, v11.h[3]    \n"
                        "fmla   v30.8h, v19.8h, v12.h[3]    \n"
                        "fmla   v31.8h, v19.8h, v13.h[3]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v10.h[4]    \n"
                        "fmla   v29.8h, v20.8h, v11.h[4]    \n"
                        "fmla   v30.8h, v20.8h, v12.h[4]    \n"
                        "fmla   v31.8h, v20.8h, v13.h[4]    \n"
                        "fmla   v28.8h, v21.8h, v10.h[5]    \n"
                        "fmla   v29.8h, v21.8h, v11.h[5]    \n"
                        "fmla   v30.8h, v21.8h, v12.h[5]    \n"
                        "fmla   v31.8h, v21.8h, v13.h[5]    \n"
                        "fmla   v28.8h, v22.8h, v10.h[6]    \n"
                        "fmla   v29.8h, v22.8h, v11.h[6]    \n"
                        "fmla   v30.8h, v22.8h, v12.h[6]    \n"
                        "fmla   v31.8h, v22.8h, v13.h[6]    \n"
                        "fmla   v28.8h, v23.8h, v10.h[7]    \n"
                        "fmla   v29.8h, v23.8h, v11.h[7]    \n"
                        "fmla   v30.8h, v23.8h, v12.h[7]    \n"
                        "fmla   v31.8h, v23.8h, v13.h[7]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v11.h[0]    \n"
                        "fmla   v29.8h, v16.8h, v12.h[0]    \n"
                        "fmla   v30.8h, v16.8h, v13.h[0]    \n"
                        "fmla   v31.8h, v16.8h, v14.h[0]    \n"
                        "fmla   v28.8h, v17.8h, v11.h[1]    \n"
                        "fmla   v29.8h, v17.8h, v12.h[1]    \n"
                        "fmla   v30.8h, v17.8h, v13.h[1]    \n"
                        "fmla   v31.8h, v17.8h, v14.h[1]    \n"
                        "fmla   v28.8h, v18.8h, v11.h[2]    \n"
                        "fmla   v29.8h, v18.8h, v12.h[2]    \n"
                        "fmla   v30.8h, v18.8h, v13.h[2]    \n"
                        "fmla   v31.8h, v18.8h, v14.h[2]    \n"
                        "fmla   v28.8h, v19.8h, v11.h[3]    \n"
                        "fmla   v29.8h, v19.8h, v12.h[3]    \n"
                        "fmla   v30.8h, v19.8h, v13.h[3]    \n"
                        "fmla   v31.8h, v19.8h, v14.h[3]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v11.h[4]    \n"
                        "fmla   v29.8h, v20.8h, v12.h[4]    \n"
                        "fmla   v30.8h, v20.8h, v13.h[4]    \n"
                        "fmla   v31.8h, v20.8h, v14.h[4]    \n"
                        "fmla   v28.8h, v21.8h, v11.h[5]    \n"
                        "fmla   v29.8h, v21.8h, v12.h[5]    \n"
                        "fmla   v30.8h, v21.8h, v13.h[5]    \n"
                        "fmla   v31.8h, v21.8h, v14.h[5]    \n"
                        "fmla   v28.8h, v22.8h, v11.h[6]    \n"
                        "fmla   v29.8h, v22.8h, v12.h[6]    \n"
                        "fmla   v30.8h, v22.8h, v13.h[6]    \n"
                        "fmla   v31.8h, v22.8h, v14.h[6]    \n"
                        "fmla   v28.8h, v23.8h, v11.h[7]    \n"
                        "fmla   v29.8h, v23.8h, v12.h[7]    \n"
                        "fmla   v30.8h, v23.8h, v13.h[7]    \n"
                        "fmla   v31.8h, v23.8h, v14.h[7]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v12.h[0]    \n"
                        "fmla   v29.8h, v16.8h, v13.h[0]    \n"
                        "fmla   v30.8h, v16.8h, v14.h[0]    \n"
                        "fmla   v31.8h, v16.8h, v15.h[0]    \n"
                        "fmla   v28.8h, v17.8h, v12.h[1]    \n"
                        "fmla   v29.8h, v17.8h, v13.h[1]    \n"
                        "fmla   v30.8h, v17.8h, v14.h[1]    \n"
                        "fmla   v31.8h, v17.8h, v15.h[1]    \n"

                        "prfm   pldl1keep, [%5, #512]       \n"
                        "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%5], #64 \n" // r40 r41 r42 r43

                        "fmla   v28.8h, v18.8h, v12.h[2]    \n"
                        "fmla   v29.8h, v18.8h, v13.h[2]    \n"
                        "fmla   v30.8h, v18.8h, v14.h[2]    \n"
                        "fmla   v31.8h, v18.8h, v15.h[2]    \n"
                        "fmla   v28.8h, v19.8h, v12.h[3]    \n"
                        "fmla   v29.8h, v19.8h, v13.h[3]    \n"
                        "fmla   v30.8h, v19.8h, v14.h[3]    \n"
                        "fmla   v31.8h, v19.8h, v15.h[3]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v12.h[4]    \n"
                        "fmla   v29.8h, v20.8h, v13.h[4]    \n"
                        "fmla   v30.8h, v20.8h, v14.h[4]    \n"
                        "fmla   v31.8h, v20.8h, v15.h[4]    \n"
                        "fmla   v28.8h, v21.8h, v12.h[5]    \n"
                        "fmla   v29.8h, v21.8h, v13.h[5]    \n"
                        "fmla   v30.8h, v21.8h, v14.h[5]    \n"
                        "fmla   v31.8h, v21.8h, v15.h[5]    \n"
                        "fmla   v28.8h, v22.8h, v12.h[6]    \n"
                        "fmla   v29.8h, v22.8h, v13.h[6]    \n"
                        "fmla   v30.8h, v22.8h, v14.h[6]    \n"
                        "fmla   v31.8h, v22.8h, v15.h[6]    \n"
                        "fmla   v28.8h, v23.8h, v12.h[7]    \n"
                        "fmla   v29.8h, v23.8h, v13.h[7]    \n"
                        "fmla   v30.8h, v23.8h, v14.h[7]    \n"
                        "fmla   v31.8h, v23.8h, v15.h[7]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v0.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v1.h[0]     \n"
                        "fmla   v30.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v31.8h, v16.8h, v3.h[0]     \n"
                        "fmla   v28.8h, v17.8h, v0.h[1]     \n"
                        "fmla   v29.8h, v17.8h, v1.h[1]     \n"
                        "fmla   v30.8h, v17.8h, v2.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v3.h[1]     \n"

                        "prfm   pldl1keep, [%5, #512]       \n"
                        "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%5] \n" // r44 r45 r46 r47

                        "fmla   v28.8h, v18.8h, v0.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v1.h[2]     \n"
                        "fmla   v30.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v31.8h, v18.8h, v3.h[2]     \n"
                        "fmla   v28.8h, v19.8h, v0.h[3]     \n"
                        "fmla   v29.8h, v19.8h, v1.h[3]     \n"
                        "fmla   v30.8h, v19.8h, v2.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v3.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v0.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v1.h[4]     \n"
                        "fmla   v30.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v31.8h, v20.8h, v3.h[4]     \n"
                        "fmla   v28.8h, v21.8h, v0.h[5]     \n"
                        "fmla   v29.8h, v21.8h, v1.h[5]     \n"
                        "fmla   v30.8h, v21.8h, v2.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v3.h[5]     \n"
                        "fmla   v28.8h, v22.8h, v0.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v1.h[6]     \n"
                        "fmla   v30.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v31.8h, v22.8h, v3.h[6]     \n"
                        "fmla   v28.8h, v23.8h, v0.h[7]     \n"
                        "fmla   v29.8h, v23.8h, v1.h[7]     \n"
                        "fmla   v30.8h, v23.8h, v2.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v3.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v1.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v30.8h, v16.8h, v3.h[0]     \n"
                        "fmla   v31.8h, v16.8h, v4.h[0]     \n"
                        "fmla   v28.8h, v17.8h, v1.h[1]     \n"
                        "fmla   v29.8h, v17.8h, v2.h[1]     \n"
                        "fmla   v30.8h, v17.8h, v3.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v4.h[1]     \n"
                        "fmla   v28.8h, v18.8h, v1.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v30.8h, v18.8h, v3.h[2]     \n"
                        "fmla   v31.8h, v18.8h, v4.h[2]     \n"
                        "fmla   v28.8h, v19.8h, v1.h[3]     \n"
                        "fmla   v29.8h, v19.8h, v2.h[3]     \n"
                        "fmla   v30.8h, v19.8h, v3.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v4.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v1.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v30.8h, v20.8h, v3.h[4]     \n"
                        "fmla   v31.8h, v20.8h, v4.h[4]     \n"
                        "fmla   v28.8h, v21.8h, v1.h[5]     \n"
                        "fmla   v29.8h, v21.8h, v2.h[5]     \n"
                        "fmla   v30.8h, v21.8h, v3.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v4.h[5]     \n"
                        "fmla   v28.8h, v22.8h, v1.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v30.8h, v22.8h, v3.h[6]     \n"
                        "fmla   v31.8h, v22.8h, v4.h[6]     \n"
                        "fmla   v28.8h, v23.8h, v1.h[7]     \n"
                        "fmla   v29.8h, v23.8h, v2.h[7]     \n"
                        "fmla   v30.8h, v23.8h, v3.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v4.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v3.h[0]     \n"
                        "fmla   v30.8h, v16.8h, v4.h[0]     \n"
                        "fmla   v31.8h, v16.8h, v5.h[0]     \n"
                        "fmla   v28.8h, v17.8h, v2.h[1]     \n"
                        "fmla   v29.8h, v17.8h, v3.h[1]     \n"
                        "fmla   v30.8h, v17.8h, v4.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v5.h[1]     \n"
                        "fmla   v28.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v3.h[2]     \n"
                        "fmla   v30.8h, v18.8h, v4.h[2]     \n"
                        "fmla   v31.8h, v18.8h, v5.h[2]     \n"
                        "fmla   v28.8h, v19.8h, v2.h[3]     \n"
                        "fmla   v29.8h, v19.8h, v3.h[3]     \n"
                        "fmla   v30.8h, v19.8h, v4.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v5.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v3.h[4]     \n"
                        "fmla   v30.8h, v20.8h, v4.h[4]     \n"
                        "fmla   v31.8h, v20.8h, v5.h[4]     \n"
                        "fmla   v28.8h, v21.8h, v2.h[5]     \n"
                        "fmla   v29.8h, v21.8h, v3.h[5]     \n"
                        "fmla   v30.8h, v21.8h, v4.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v5.h[5]     \n"
                        "fmla   v28.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v3.h[6]     \n"
                        "fmla   v30.8h, v22.8h, v4.h[6]     \n"
                        "fmla   v31.8h, v22.8h, v5.h[6]     \n"
                        "fmla   v28.8h, v23.8h, v2.h[7]     \n"
                        "fmla   v29.8h, v23.8h, v3.h[7]     \n"
                        "fmla   v30.8h, v23.8h, v4.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v5.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v3.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v4.h[0]     \n"
                        "fmla   v30.8h, v16.8h, v5.h[0]     \n"
                        "fmla   v31.8h, v16.8h, v6.h[0]     \n"
                        "fmla   v28.8h, v17.8h, v3.h[1]     \n"
                        "fmla   v29.8h, v17.8h, v4.h[1]     \n"
                        "fmla   v30.8h, v17.8h, v5.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v6.h[1]     \n"
                        "fmla   v28.8h, v18.8h, v3.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v4.h[2]     \n"
                        "fmla   v30.8h, v18.8h, v5.h[2]     \n"
                        "fmla   v31.8h, v18.8h, v6.h[2]     \n"
                        "fmla   v28.8h, v19.8h, v3.h[3]     \n"
                        "fmla   v29.8h, v19.8h, v4.h[3]     \n"
                        "fmla   v30.8h, v19.8h, v5.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v6.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v3.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v4.h[4]     \n"
                        "fmla   v30.8h, v20.8h, v5.h[4]     \n"
                        "fmla   v31.8h, v20.8h, v6.h[4]     \n"
                        "fmla   v28.8h, v21.8h, v3.h[5]     \n"
                        "fmla   v29.8h, v21.8h, v4.h[5]     \n"
                        "fmla   v30.8h, v21.8h, v5.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v6.h[5]     \n"
                        "fmla   v28.8h, v22.8h, v3.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v4.h[6]     \n"
                        "fmla   v30.8h, v22.8h, v5.h[6]     \n"
                        "fmla   v31.8h, v22.8h, v6.h[6]     \n"
                        "fmla   v28.8h, v23.8h, v3.h[7]     \n"
                        "fmla   v29.8h, v23.8h, v4.h[7]     \n"
                        "fmla   v30.8h, v23.8h, v5.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v6.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6] \n"

                        "fmla   v28.8h, v16.8h, v4.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v5.h[0]     \n"
                        "fmla   v30.8h, v16.8h, v6.h[0]     \n"
                        "fmla   v31.8h, v16.8h, v7.h[0]     \n"
                        "fmla   v28.8h, v17.8h, v4.h[1]     \n"
                        "fmla   v29.8h, v17.8h, v5.h[1]     \n"
                        "fmla   v30.8h, v17.8h, v6.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v7.h[1]     \n"
                        "fmla   v28.8h, v18.8h, v4.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v5.h[2]     \n"
                        "fmla   v30.8h, v18.8h, v6.h[2]     \n"
                        "fmla   v31.8h, v18.8h, v7.h[2]     \n"
                        "fmla   v28.8h, v19.8h, v4.h[3]     \n"
                        "fmla   v29.8h, v19.8h, v5.h[3]     \n"
                        "fmla   v30.8h, v19.8h, v6.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v7.h[3]     \n"
                        "fmla   v28.8h, v20.8h, v4.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v5.h[4]     \n"
                        "fmla   v30.8h, v20.8h, v6.h[4]     \n"
                        "fmla   v31.8h, v20.8h, v7.h[4]     \n"
                        "fmla   v28.8h, v21.8h, v4.h[5]     \n"
                        "fmla   v29.8h, v21.8h, v5.h[5]     \n"
                        "fmla   v30.8h, v21.8h, v6.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v7.h[5]     \n"
                        "fmla   v28.8h, v22.8h, v4.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v5.h[6]     \n"
                        "fmla   v30.8h, v22.8h, v6.h[6]     \n"
                        "fmla   v31.8h, v22.8h, v7.h[6]     \n"
                        "fmla   v28.8h, v23.8h, v4.h[7]     \n"
                        "fmla   v29.8h, v23.8h, v5.h[7]     \n"
                        "fmla   v30.8h, v23.8h, v6.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v7.h[7]     \n"

                        "sub    %6, %6, #3136               \n" // kptr -= 24.5 * 64;

                        "st1    {v28.8h, v29.8h, v30.8h, v31.8h}, [%0], #64 \n"

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
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v28", "v29", "v30", "v31");
                }
                for (; j + 1 < outw; j += 2)
                {
                    asm volatile(
                        "prfm   pldl1keep, [%1, #256]       \n"
                        "ld1    {v0.8h, v1.8h}, [%1], #32   \n" // r00 r01

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "prfm   pldl1keep, [%0, #256]       \n"
                        "ld1    {v30.8h, v31.8h}, [%0]      \n" // sum0 sum1

                        "fmul   v28.8h, v16.8h, v0.h[0]     \n"
                        "fmul   v29.8h, v16.8h, v1.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v0.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v1.h[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v18.8h, v0.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v1.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v0.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v1.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v0.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v1.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v0.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v1.h[5]     \n"

                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v2.8h, v3.8h, v4.8h, v5.8h}, [%1] \n" // r02 r03 r04 r05

                        "fmla   v28.8h, v22.8h, v0.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v1.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v0.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v1.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v1.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v1.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v2.h[1]     \n"
                        "fmla   v28.8h, v18.8h, v1.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v1.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v2.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v1.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v1.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v2.h[5]     \n"
                        "fmla   v28.8h, v22.8h, v1.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v1.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v2.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v3.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v2.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v3.h[1]     \n"
                        "fmla   v28.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v3.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v2.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v3.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v3.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v2.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v3.h[5]     \n"
                        "fmla   v28.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v3.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v2.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v3.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v3.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v4.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v3.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v4.h[1]     \n"
                        "fmla   v28.8h, v18.8h, v3.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v4.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v3.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v4.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v3.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v4.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v3.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v4.h[5]     \n"
                        "fmla   v28.8h, v22.8h, v3.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v4.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v3.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v4.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v4.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v5.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v4.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v5.h[1]     \n"

                        "prfm   pldl1keep, [%2, #256]       \n"
                        "ld1    {v8.8h, v9.8h}, [%2], #32   \n" // r10 r11

                        "fmla   v28.8h, v18.8h, v4.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v5.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v4.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v5.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v4.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v5.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v4.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v5.h[5]     \n"
                        "fmla   v28.8h, v22.8h, v4.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v5.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v4.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v5.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v30.8h, v16.8h, v8.h[0]     \n"
                        "fmla   v31.8h, v16.8h, v9.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v8.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v9.h[1]     \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v10.8h, v11.8h, v12.8h, v13.8h}, [%2] \n" // r12 r13 r14 r15

                        "fmla   v30.8h, v18.8h, v8.h[2]     \n"
                        "fmla   v31.8h, v18.8h, v9.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v8.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v9.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v30.8h, v20.8h, v8.h[4]     \n"
                        "fmla   v31.8h, v20.8h, v9.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v8.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v9.h[5]     \n"
                        "fmla   v30.8h, v22.8h, v8.h[6]     \n"
                        "fmla   v31.8h, v22.8h, v9.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v8.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v9.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v30.8h, v16.8h, v9.h[0]     \n"
                        "fmla   v31.8h, v16.8h, v10.h[0]    \n"
                        "fmla   v30.8h, v17.8h, v9.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v10.h[1]    \n"
                        "fmla   v30.8h, v18.8h, v9.h[2]     \n"
                        "fmla   v31.8h, v18.8h, v10.h[2]    \n"
                        "fmla   v30.8h, v19.8h, v9.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v10.h[3]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v30.8h, v20.8h, v9.h[4]     \n"
                        "fmla   v31.8h, v20.8h, v10.h[4]    \n"
                        "fmla   v30.8h, v21.8h, v9.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v10.h[5]    \n"
                        "fmla   v30.8h, v22.8h, v9.h[6]     \n"
                        "fmla   v31.8h, v22.8h, v10.h[6]    \n"
                        "fmla   v30.8h, v23.8h, v9.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v10.h[7]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v30.8h, v16.8h, v10.h[0]    \n"
                        "fmla   v31.8h, v16.8h, v11.h[0]    \n"
                        "fmla   v30.8h, v17.8h, v10.h[1]    \n"
                        "fmla   v31.8h, v17.8h, v11.h[1]    \n"
                        "fmla   v30.8h, v18.8h, v10.h[2]    \n"
                        "fmla   v31.8h, v18.8h, v11.h[2]    \n"
                        "fmla   v30.8h, v19.8h, v10.h[3]    \n"
                        "fmla   v31.8h, v19.8h, v11.h[3]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v30.8h, v20.8h, v10.h[4]    \n"
                        "fmla   v31.8h, v20.8h, v11.h[4]    \n"
                        "fmla   v30.8h, v21.8h, v10.h[5]    \n"
                        "fmla   v31.8h, v21.8h, v11.h[5]    \n"
                        "fmla   v30.8h, v22.8h, v10.h[6]    \n"
                        "fmla   v31.8h, v22.8h, v11.h[6]    \n"
                        "fmla   v30.8h, v23.8h, v10.h[7]    \n"
                        "fmla   v31.8h, v23.8h, v11.h[7]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v30.8h, v16.8h, v11.h[0]    \n"
                        "fmla   v31.8h, v16.8h, v12.h[0]    \n"
                        "fmla   v30.8h, v17.8h, v11.h[1]    \n"
                        "fmla   v31.8h, v17.8h, v12.h[1]    \n"
                        "fmla   v30.8h, v18.8h, v11.h[2]    \n"
                        "fmla   v31.8h, v18.8h, v12.h[2]    \n"
                        "fmla   v30.8h, v19.8h, v11.h[3]    \n"
                        "fmla   v31.8h, v19.8h, v12.h[3]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v30.8h, v20.8h, v11.h[4]    \n"
                        "fmla   v31.8h, v20.8h, v12.h[4]    \n"
                        "fmla   v30.8h, v21.8h, v11.h[5]    \n"
                        "fmla   v31.8h, v21.8h, v12.h[5]    \n"
                        "fmla   v30.8h, v22.8h, v11.h[6]    \n"
                        "fmla   v31.8h, v22.8h, v12.h[6]    \n"
                        "fmla   v30.8h, v23.8h, v11.h[7]    \n"
                        "fmla   v31.8h, v23.8h, v12.h[7]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v30.8h, v16.8h, v12.h[0]    \n"
                        "fmla   v31.8h, v16.8h, v13.h[0]    \n"
                        "fmla   v30.8h, v17.8h, v12.h[1]    \n"
                        "fmla   v31.8h, v17.8h, v13.h[1]    \n"

                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld1    {v0.8h, v1.8h}, [%3], #32   \n" // r20 r21

                        "fmla   v30.8h, v18.8h, v12.h[2]    \n"
                        "fmla   v31.8h, v18.8h, v13.h[2]    \n"
                        "fmla   v30.8h, v19.8h, v12.h[3]    \n"
                        "fmla   v31.8h, v19.8h, v13.h[3]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v30.8h, v20.8h, v12.h[4]    \n"
                        "fmla   v31.8h, v20.8h, v13.h[4]    \n"
                        "fmla   v30.8h, v21.8h, v12.h[5]    \n"
                        "fmla   v31.8h, v21.8h, v13.h[5]    \n"
                        "fmla   v30.8h, v22.8h, v12.h[6]    \n"
                        "fmla   v31.8h, v22.8h, v13.h[6]    \n"
                        "fmla   v30.8h, v23.8h, v12.h[7]    \n"
                        "fmla   v31.8h, v23.8h, v13.h[7]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v0.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v1.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v0.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v1.h[1]     \n"

                        "prfm   pldl1keep, [%3, #512]       \n"
                        "ld1    {v2.8h, v3.8h, v4.8h, v5.8h}, [%3] \n" // r22 r23 r24 r25

                        "fmla   v28.8h, v18.8h, v0.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v1.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v0.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v1.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v0.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v1.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v0.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v1.h[5]     \n"
                        "fmla   v28.8h, v22.8h, v0.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v1.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v0.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v1.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v1.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v1.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v2.h[1]     \n"
                        "fmla   v28.8h, v18.8h, v1.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v1.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v2.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v1.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v1.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v2.h[5]     \n"
                        "fmla   v28.8h, v22.8h, v1.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v1.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v2.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v3.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v2.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v3.h[1]     \n"
                        "fmla   v28.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v3.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v2.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v3.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v3.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v2.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v3.h[5]     \n"
                        "fmla   v28.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v3.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v2.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v3.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v3.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v4.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v3.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v4.h[1]     \n"
                        "fmla   v28.8h, v18.8h, v3.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v4.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v3.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v4.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v3.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v4.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v3.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v4.h[5]     \n"
                        "fmla   v28.8h, v22.8h, v3.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v4.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v3.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v4.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v4.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v5.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v4.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v5.h[1]     \n"

                        "prfm   pldl1keep, [%4, #256]       \n"
                        "ld1    {v8.8h, v9.8h}, [%4], #32   \n" // r30 r31

                        "fmla   v28.8h, v18.8h, v4.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v5.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v4.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v5.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v4.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v5.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v4.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v5.h[5]     \n"
                        "fmla   v28.8h, v22.8h, v4.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v5.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v4.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v5.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v30.8h, v16.8h, v8.h[0]     \n"
                        "fmla   v31.8h, v16.8h, v9.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v8.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v9.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v10.8h, v11.8h, v12.8h, v13.8h}, [%4] \n" // r32 r33 r34 r35

                        "fmla   v30.8h, v18.8h, v8.h[2]     \n"
                        "fmla   v31.8h, v18.8h, v9.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v8.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v9.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v30.8h, v20.8h, v8.h[4]     \n"
                        "fmla   v31.8h, v20.8h, v9.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v8.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v9.h[5]     \n"
                        "fmla   v30.8h, v22.8h, v8.h[6]     \n"
                        "fmla   v31.8h, v22.8h, v9.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v8.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v9.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v30.8h, v16.8h, v9.h[0]     \n"
                        "fmla   v31.8h, v16.8h, v10.h[0]    \n"
                        "fmla   v30.8h, v17.8h, v9.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v10.h[1]    \n"
                        "fmla   v30.8h, v18.8h, v9.h[2]     \n"
                        "fmla   v31.8h, v18.8h, v10.h[2]    \n"
                        "fmla   v30.8h, v19.8h, v9.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v10.h[3]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v30.8h, v20.8h, v9.h[4]     \n"
                        "fmla   v31.8h, v20.8h, v10.h[4]    \n"
                        "fmla   v30.8h, v21.8h, v9.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v10.h[5]    \n"
                        "fmla   v30.8h, v22.8h, v9.h[6]     \n"
                        "fmla   v31.8h, v22.8h, v10.h[6]    \n"
                        "fmla   v30.8h, v23.8h, v9.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v10.h[7]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v30.8h, v16.8h, v10.h[0]    \n"
                        "fmla   v31.8h, v16.8h, v11.h[0]    \n"
                        "fmla   v30.8h, v17.8h, v10.h[1]    \n"
                        "fmla   v31.8h, v17.8h, v11.h[1]    \n"
                        "fmla   v30.8h, v18.8h, v10.h[2]    \n"
                        "fmla   v31.8h, v18.8h, v11.h[2]    \n"
                        "fmla   v30.8h, v19.8h, v10.h[3]    \n"
                        "fmla   v31.8h, v19.8h, v11.h[3]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v30.8h, v20.8h, v10.h[4]    \n"
                        "fmla   v31.8h, v20.8h, v11.h[4]    \n"
                        "fmla   v30.8h, v21.8h, v10.h[5]    \n"
                        "fmla   v31.8h, v21.8h, v11.h[5]    \n"
                        "fmla   v30.8h, v22.8h, v10.h[6]    \n"
                        "fmla   v31.8h, v22.8h, v11.h[6]    \n"
                        "fmla   v30.8h, v23.8h, v10.h[7]    \n"
                        "fmla   v31.8h, v23.8h, v11.h[7]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v30.8h, v16.8h, v11.h[0]    \n"
                        "fmla   v31.8h, v16.8h, v12.h[0]    \n"
                        "fmla   v30.8h, v17.8h, v11.h[1]    \n"
                        "fmla   v31.8h, v17.8h, v12.h[1]    \n"
                        "fmla   v30.8h, v18.8h, v11.h[2]    \n"
                        "fmla   v31.8h, v18.8h, v12.h[2]    \n"
                        "fmla   v30.8h, v19.8h, v11.h[3]    \n"
                        "fmla   v31.8h, v19.8h, v12.h[3]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v30.8h, v20.8h, v11.h[4]    \n"
                        "fmla   v31.8h, v20.8h, v12.h[4]    \n"
                        "fmla   v30.8h, v21.8h, v11.h[5]    \n"
                        "fmla   v31.8h, v21.8h, v12.h[5]    \n"
                        "fmla   v30.8h, v22.8h, v11.h[6]    \n"
                        "fmla   v31.8h, v22.8h, v12.h[6]    \n"
                        "fmla   v30.8h, v23.8h, v11.h[7]    \n"
                        "fmla   v31.8h, v23.8h, v12.h[7]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v30.8h, v16.8h, v12.h[0]    \n"
                        "fmla   v31.8h, v16.8h, v13.h[0]    \n"
                        "fmla   v30.8h, v17.8h, v12.h[1]    \n"
                        "fmla   v31.8h, v17.8h, v13.h[1]    \n"

                        "prfm   pldl1keep, [%5, #256]       \n"
                        "ld1    {v0.8h, v1.8h}, [%5], #32   \n" // r40 r41

                        "fmla   v30.8h, v18.8h, v12.h[2]    \n"
                        "fmla   v31.8h, v18.8h, v13.h[2]    \n"
                        "fmla   v30.8h, v19.8h, v12.h[3]    \n"
                        "fmla   v31.8h, v19.8h, v13.h[3]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v30.8h, v20.8h, v12.h[4]    \n"
                        "fmla   v31.8h, v20.8h, v13.h[4]    \n"
                        "fmla   v30.8h, v21.8h, v12.h[5]    \n"
                        "fmla   v31.8h, v21.8h, v13.h[5]    \n"
                        "fmla   v30.8h, v22.8h, v12.h[6]    \n"
                        "fmla   v31.8h, v22.8h, v13.h[6]    \n"
                        "fmla   v30.8h, v23.8h, v12.h[7]    \n"
                        "fmla   v31.8h, v23.8h, v13.h[7]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v0.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v1.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v0.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v1.h[1]     \n"

                        "prfm   pldl1keep, [%5, #512]       \n"
                        "ld1    {v2.8h, v3.8h, v4.8h, v5.8h}, [%5] \n" // r42 r43 r44 r45

                        "fmla   v28.8h, v18.8h, v0.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v1.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v0.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v1.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v0.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v1.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v0.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v1.h[5]     \n"
                        "fmla   v28.8h, v22.8h, v0.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v1.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v0.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v1.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v1.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v1.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v2.h[1]     \n"
                        "fmla   v28.8h, v18.8h, v1.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v1.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v2.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v1.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v1.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v2.h[5]     \n"
                        "fmla   v28.8h, v22.8h, v1.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v1.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v2.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v3.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v2.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v3.h[1]     \n"
                        "fmla   v28.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v3.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v2.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v3.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v3.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v2.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v3.h[5]     \n"
                        "fmla   v28.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v3.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v2.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v3.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v3.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v4.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v3.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v4.h[1]     \n"
                        "fmla   v28.8h, v18.8h, v3.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v4.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v3.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v4.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v3.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v4.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v3.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v4.h[5]     \n"
                        "fmla   v28.8h, v22.8h, v3.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v4.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v3.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v4.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6] \n"

                        "fmla   v28.8h, v16.8h, v4.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v5.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v4.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v5.h[1]     \n"
                        "fmla   v28.8h, v18.8h, v4.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v5.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v4.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v5.h[3]     \n"
                        "fmla   v28.8h, v20.8h, v4.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v5.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v4.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v5.h[5]     \n"
                        "fmla   v28.8h, v22.8h, v4.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v5.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v4.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v5.h[7]     \n"

                        "fadd   v28.8h, v28.8h, v30.8h      \n"
                        "fadd   v29.8h, v29.8h, v31.8h      \n"

                        "sub    %6, %6, #3136               \n" // kptr -= 24.5 * 64;

                        "st1    {v28.8h, v29.8h}, [%0], #32 \n"

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
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v8", "v9", "v10", "v11", "v12", "v13", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v28", "v29", "v30", "v31");
                }
                for (; j < outw; j++)
                {
                    asm volatile(
                        "prfm   pldl1keep, [%1, #128]       \n"
                        "ld1    {v0.8h}, [%1], #16          \n" // r00

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "prfm   pldl1keep, [%0, #128]       \n"
                        "ld1    {v31.8h}, [%0]              \n" // sum0

                        "fmul   v28.8h, v16.8h, v0.h[0]     \n"
                        "fmul   v29.8h, v17.8h, v0.h[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmul   v30.8h, v18.8h, v0.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v0.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v0.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v0.h[5]     \n"

                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v1.8h, v2.8h, v3.8h, v4.8h}, [%1] \n" // r01 r02 r03 r04

                        "fmla   v30.8h, v22.8h, v0.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v0.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v1.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v1.h[1]     \n"
                        "fmla   v30.8h, v18.8h, v1.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v1.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v1.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v1.h[5]     \n"
                        "fmla   v30.8h, v22.8h, v1.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v1.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v2.h[1]     \n"
                        "fmla   v30.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v2.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v2.h[5]     \n"
                        "fmla   v30.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v2.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v3.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v3.h[1]     \n"
                        "fmla   v30.8h, v18.8h, v3.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v3.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v3.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v3.h[5]     \n"
                        "fmla   v30.8h, v22.8h, v3.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v3.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v4.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v4.h[1]     \n"

                        "prfm   pldl1keep, [%2, #128]       \n"
                        "ld1    {v8.8h}, [%2], #16          \n" // r10

                        "fmla   v30.8h, v18.8h, v4.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v4.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v4.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v4.h[5]     \n"
                        "fmla   v30.8h, v22.8h, v4.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v4.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v8.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v8.h[1]     \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v9.8h, v10.8h, v11.8h, v12.8h}, [%2] \n" // r11 r12 r13 r14

                        "fmla   v30.8h, v18.8h, v8.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v8.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v8.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v8.h[5]     \n"
                        "fmla   v30.8h, v22.8h, v8.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v8.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v9.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v9.h[1]     \n"
                        "fmla   v30.8h, v18.8h, v9.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v9.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v9.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v9.h[5]     \n"
                        "fmla   v30.8h, v22.8h, v9.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v9.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v10.h[0]    \n"
                        "fmla   v29.8h, v17.8h, v10.h[1]    \n"
                        "fmla   v30.8h, v18.8h, v10.h[2]    \n"
                        "fmla   v31.8h, v19.8h, v10.h[3]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v10.h[4]    \n"
                        "fmla   v29.8h, v21.8h, v10.h[5]    \n"
                        "fmla   v30.8h, v22.8h, v10.h[6]    \n"
                        "fmla   v31.8h, v23.8h, v10.h[7]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v11.h[0]    \n"
                        "fmla   v29.8h, v17.8h, v11.h[1]    \n"
                        "fmla   v30.8h, v18.8h, v11.h[2]    \n"
                        "fmla   v31.8h, v19.8h, v11.h[3]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v11.h[4]    \n"
                        "fmla   v29.8h, v21.8h, v11.h[5]    \n"
                        "fmla   v30.8h, v22.8h, v11.h[6]    \n"
                        "fmla   v31.8h, v23.8h, v11.h[7]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v12.h[0]    \n"
                        "fmla   v29.8h, v17.8h, v12.h[1]    \n"

                        "prfm   pldl1keep, [%3, #128]       \n"
                        "ld1    {v0.8h}, [%3], #16          \n" // r20

                        "fmla   v30.8h, v18.8h, v12.h[2]    \n"
                        "fmla   v31.8h, v19.8h, v12.h[3]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v12.h[4]    \n"
                        "fmla   v29.8h, v21.8h, v12.h[5]    \n"
                        "fmla   v30.8h, v22.8h, v12.h[6]    \n"
                        "fmla   v31.8h, v23.8h, v12.h[7]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v0.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v0.h[1]     \n"

                        "prfm   pldl1keep, [%3, #512]       \n"
                        "ld1    {v1.8h, v2.8h, v3.8h, v4.8h}, [%3] \n" // r21 r22 r23 r24

                        "fmla   v30.8h, v18.8h, v0.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v0.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v0.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v0.h[5]     \n"
                        "fmla   v30.8h, v22.8h, v0.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v0.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v1.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v1.h[1]     \n"
                        "fmla   v30.8h, v18.8h, v1.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v1.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v1.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v1.h[5]     \n"
                        "fmla   v30.8h, v22.8h, v1.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v1.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v2.h[1]     \n"
                        "fmla   v30.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v2.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v2.h[5]     \n"
                        "fmla   v30.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v2.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v3.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v3.h[1]     \n"
                        "fmla   v30.8h, v18.8h, v3.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v3.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v3.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v3.h[5]     \n"
                        "fmla   v30.8h, v22.8h, v3.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v3.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v4.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v4.h[1]     \n"

                        "prfm   pldl1keep, [%4, #128]       \n"
                        "ld1    {v8.8h}, [%4], #16          \n" // r30

                        "fmla   v30.8h, v18.8h, v4.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v4.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v4.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v4.h[5]     \n"
                        "fmla   v30.8h, v22.8h, v4.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v4.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v8.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v8.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v9.8h, v10.8h, v11.8h, v12.8h}, [%4] \n" // r31 r32 r33 r34

                        "fmla   v30.8h, v18.8h, v8.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v8.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v8.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v8.h[5]     \n"
                        "fmla   v30.8h, v22.8h, v8.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v8.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v9.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v9.h[1]     \n"
                        "fmla   v30.8h, v18.8h, v9.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v9.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v9.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v9.h[5]     \n"
                        "fmla   v30.8h, v22.8h, v9.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v9.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v10.h[0]    \n"
                        "fmla   v29.8h, v17.8h, v10.h[1]    \n"
                        "fmla   v30.8h, v18.8h, v10.h[2]    \n"
                        "fmla   v31.8h, v19.8h, v10.h[3]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v10.h[4]    \n"
                        "fmla   v29.8h, v21.8h, v10.h[5]    \n"
                        "fmla   v30.8h, v22.8h, v10.h[6]    \n"
                        "fmla   v31.8h, v23.8h, v10.h[7]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v11.h[0]    \n"
                        "fmla   v29.8h, v17.8h, v11.h[1]    \n"
                        "fmla   v30.8h, v18.8h, v11.h[2]    \n"
                        "fmla   v31.8h, v19.8h, v11.h[3]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v11.h[4]    \n"
                        "fmla   v29.8h, v21.8h, v11.h[5]    \n"
                        "fmla   v30.8h, v22.8h, v11.h[6]    \n"
                        "fmla   v31.8h, v23.8h, v11.h[7]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v12.h[0]    \n"
                        "fmla   v29.8h, v17.8h, v12.h[1]    \n"

                        "prfm   pldl1keep, [%5, #128]       \n"
                        "ld1    {v0.8h}, [%5], #16          \n" // r40

                        "fmla   v30.8h, v18.8h, v12.h[2]    \n"
                        "fmla   v31.8h, v19.8h, v12.h[3]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v12.h[4]    \n"
                        "fmla   v29.8h, v21.8h, v12.h[5]    \n"
                        "fmla   v30.8h, v22.8h, v12.h[6]    \n"
                        "fmla   v31.8h, v23.8h, v12.h[7]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v0.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v0.h[1]     \n"

                        "prfm   pldl1keep, [%5, #512]       \n"
                        "ld1    {v1.8h, v2.8h, v3.8h, v4.8h}, [%5] \n" // r41 r42 r43 r44

                        "fmla   v30.8h, v18.8h, v0.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v0.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v0.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v0.h[5]     \n"
                        "fmla   v30.8h, v22.8h, v0.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v0.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v1.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v1.h[1]     \n"
                        "fmla   v30.8h, v18.8h, v1.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v1.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v1.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v1.h[5]     \n"
                        "fmla   v30.8h, v22.8h, v1.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v1.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v2.h[1]     \n"
                        "fmla   v30.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v2.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v2.h[5]     \n"
                        "fmla   v30.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v2.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v3.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v3.h[1]     \n"
                        "fmla   v30.8h, v18.8h, v3.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v3.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v3.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v3.h[5]     \n"
                        "fmla   v30.8h, v22.8h, v3.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v3.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6] \n"

                        "fmla   v28.8h, v16.8h, v4.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v4.h[1]     \n"
                        "fmla   v30.8h, v18.8h, v4.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v4.h[3]     \n"
                        "fmla   v28.8h, v20.8h, v4.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v4.h[5]     \n"
                        "fmla   v30.8h, v22.8h, v4.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v4.h[7]     \n"

                        "fadd   v28.8h, v28.8h, v29.8h      \n"
                        "fadd   v30.8h, v30.8h, v31.8h      \n"
                        "fadd   v28.8h, v28.8h, v30.8h      \n"

                        "sub    %6, %6, #3136               \n" // kptr -= 24.5 * 64;

                        "st1    {v28.8h}, [%0], #16         \n"

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
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v8", "v9", "v10", "v11", "v12", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v28", "v29", "v30", "v31");
                }

                r0 += 4 * 8;
                r1 += 4 * 8;
                r2 += 4 * 8;
                r3 += 4 * 8;
                r4 += 4 * 8;
            }
        }
    }
}

static void conv5x5s2_pack8_fp16sa_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;
    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = (w - 2 * outw + w) * 8;

    const __fp16* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        float16x8_t _bias0 = bias ? vld1q_f16(bias + p * 8) : vdupq_n_f16(0.f);
        out0.fill(_bias0);

        int q = 0;
        for (; q < inch; q++)
        {
            __fp16* outptr0 = out0;

            const Mat img0 = bottom_blob.channel(q);

            const __fp16* r0 = img0.row<const __fp16>(0);
            const __fp16* r1 = img0.row<const __fp16>(1);
            const __fp16* r2 = img0.row<const __fp16>(2);
            const __fp16* r3 = img0.row<const __fp16>(3);
            const __fp16* r4 = img0.row<const __fp16>(4);

            const __fp16* kptr = kernel.channel(p).row<const __fp16>(q);

            int i = 0;
            for (; i < outh; i++)
            {
                int j = 0;
                for (; j + 1 < outw; j += 2)
                {
                    asm volatile(
                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%1], #64 \n" // r00 r01 r02 r03

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "prfm   pldl1keep, [%0, #256]       \n"
                        "ld1    {v30.8h, v31.8h}, [%0]      \n" // sum0 sum1

                        "fmul   v28.8h, v16.8h, v0.h[0]     \n"
                        "fmul   v29.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v0.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v2.h[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v18.8h, v0.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v0.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v2.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v0.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v0.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v2.h[5]     \n"
                        "fmla   v28.8h, v22.8h, v0.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v0.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v2.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v1.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v3.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v1.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v3.h[1]     \n"

                        "prfm   pldl1keep, [%1, #384]       \n"
                        "ld1    {v4.8h, v5.8h, v6.8h}, [%1] \n" // r04 r05 r06

                        "fmla   v28.8h, v18.8h, v1.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v3.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v1.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v3.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v1.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v3.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v1.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v3.h[5]     \n"
                        "fmla   v28.8h, v22.8h, v1.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v3.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v1.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v3.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v4.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v2.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v4.h[1]     \n"
                        "fmla   v28.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v4.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v2.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v4.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v4.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v2.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v4.h[5]     \n"
                        "fmla   v28.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v4.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v2.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v4.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v3.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v5.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v3.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v5.h[1]     \n"
                        "fmla   v28.8h, v18.8h, v3.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v5.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v3.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v5.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v3.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v5.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v3.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v5.h[5]     \n"
                        "fmla   v28.8h, v22.8h, v3.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v5.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v3.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v5.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v4.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v6.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v4.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v6.h[1]     \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v8.8h, v9.8h, v10.8h, v11.8h}, [%2], #64 \n" // r10 r11 r12 r13

                        "fmla   v28.8h, v18.8h, v4.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v6.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v4.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v6.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v4.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v6.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v4.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v6.h[5]     \n"
                        "fmla   v28.8h, v22.8h, v4.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v6.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v4.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v6.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v8.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v10.h[0]    \n"
                        "fmla   v30.8h, v17.8h, v8.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v10.h[1]    \n"
                        "fmla   v28.8h, v18.8h, v8.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v10.h[2]    \n"
                        "fmla   v30.8h, v19.8h, v8.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v10.h[3]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v8.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v10.h[4]    \n"
                        "fmla   v30.8h, v21.8h, v8.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v10.h[5]    \n"
                        "fmla   v28.8h, v22.8h, v8.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v10.h[6]    \n"
                        "fmla   v30.8h, v23.8h, v8.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v10.h[7]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v9.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v11.h[0]    \n"
                        "fmla   v30.8h, v17.8h, v9.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v11.h[1]    \n"

                        "prfm   pldl1keep, [%2, #384]       \n"
                        "ld1    {v12.8h, v13.8h, v14.8h}, [%2] \n" // r14 r15 r16

                        "fmla   v28.8h, v18.8h, v9.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v11.h[2]    \n"
                        "fmla   v30.8h, v19.8h, v9.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v11.h[3]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v9.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v11.h[4]    \n"
                        "fmla   v30.8h, v21.8h, v9.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v11.h[5]    \n"
                        "fmla   v28.8h, v22.8h, v9.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v11.h[6]    \n"
                        "fmla   v30.8h, v23.8h, v9.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v11.h[7]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v10.h[0]    \n"
                        "fmla   v29.8h, v16.8h, v12.h[0]    \n"
                        "fmla   v30.8h, v17.8h, v10.h[1]    \n"
                        "fmla   v31.8h, v17.8h, v12.h[1]    \n"
                        "fmla   v28.8h, v18.8h, v10.h[2]    \n"
                        "fmla   v29.8h, v18.8h, v12.h[2]    \n"
                        "fmla   v30.8h, v19.8h, v10.h[3]    \n"
                        "fmla   v31.8h, v19.8h, v12.h[3]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v10.h[4]    \n"
                        "fmla   v29.8h, v20.8h, v12.h[4]    \n"
                        "fmla   v30.8h, v21.8h, v10.h[5]    \n"
                        "fmla   v31.8h, v21.8h, v12.h[5]    \n"
                        "fmla   v28.8h, v22.8h, v10.h[6]    \n"
                        "fmla   v29.8h, v22.8h, v12.h[6]    \n"
                        "fmla   v30.8h, v23.8h, v10.h[7]    \n"
                        "fmla   v31.8h, v23.8h, v12.h[7]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v11.h[0]    \n"
                        "fmla   v29.8h, v16.8h, v13.h[0]    \n"
                        "fmla   v30.8h, v17.8h, v11.h[1]    \n"
                        "fmla   v31.8h, v17.8h, v13.h[1]    \n"
                        "fmla   v28.8h, v18.8h, v11.h[2]    \n"
                        "fmla   v29.8h, v18.8h, v13.h[2]    \n"
                        "fmla   v30.8h, v19.8h, v11.h[3]    \n"
                        "fmla   v31.8h, v19.8h, v13.h[3]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v11.h[4]    \n"
                        "fmla   v29.8h, v20.8h, v13.h[4]    \n"
                        "fmla   v30.8h, v21.8h, v11.h[5]    \n"
                        "fmla   v31.8h, v21.8h, v13.h[5]    \n"
                        "fmla   v28.8h, v22.8h, v11.h[6]    \n"
                        "fmla   v29.8h, v22.8h, v13.h[6]    \n"
                        "fmla   v30.8h, v23.8h, v11.h[7]    \n"
                        "fmla   v31.8h, v23.8h, v13.h[7]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v12.h[0]    \n"
                        "fmla   v29.8h, v16.8h, v14.h[0]    \n"
                        "fmla   v30.8h, v17.8h, v12.h[1]    \n"
                        "fmla   v31.8h, v17.8h, v14.h[1]    \n"

                        "prfm   pldl1keep, [%3, #512]       \n"
                        "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%3], #64 \n" // r20 r21 r22 r23

                        "fmla   v28.8h, v18.8h, v12.h[2]    \n"
                        "fmla   v29.8h, v18.8h, v14.h[2]    \n"
                        "fmla   v30.8h, v19.8h, v12.h[3]    \n"
                        "fmla   v31.8h, v19.8h, v14.h[3]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v12.h[4]    \n"
                        "fmla   v29.8h, v20.8h, v14.h[4]    \n"
                        "fmla   v30.8h, v21.8h, v12.h[5]    \n"
                        "fmla   v31.8h, v21.8h, v14.h[5]    \n"
                        "fmla   v28.8h, v22.8h, v12.h[6]    \n"
                        "fmla   v29.8h, v22.8h, v14.h[6]    \n"
                        "fmla   v30.8h, v23.8h, v12.h[7]    \n"
                        "fmla   v31.8h, v23.8h, v14.h[7]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v0.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v0.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v2.h[1]     \n"
                        "fmla   v28.8h, v18.8h, v0.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v0.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v2.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v0.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v0.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v2.h[5]     \n"
                        "fmla   v28.8h, v22.8h, v0.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v0.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v2.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v1.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v3.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v1.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v3.h[1]     \n"

                        "prfm   pldl1keep, [%3, #384]       \n"
                        "ld1    {v4.8h, v5.8h, v6.8h}, [%3] \n" // r24 r25 r26

                        "fmla   v28.8h, v18.8h, v1.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v3.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v1.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v3.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v1.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v3.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v1.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v3.h[5]     \n"
                        "fmla   v28.8h, v22.8h, v1.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v3.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v1.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v3.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v4.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v2.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v4.h[1]     \n"
                        "fmla   v28.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v4.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v2.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v4.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v4.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v2.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v4.h[5]     \n"
                        "fmla   v28.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v4.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v2.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v4.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v3.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v5.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v3.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v5.h[1]     \n"
                        "fmla   v28.8h, v18.8h, v3.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v5.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v3.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v5.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v3.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v5.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v3.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v5.h[5]     \n"
                        "fmla   v28.8h, v22.8h, v3.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v5.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v3.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v5.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v4.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v6.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v4.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v6.h[1]     \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v8.8h, v9.8h, v10.8h, v11.8h}, [%4], #64 \n" // r30 r31 r32 r33

                        "fmla   v28.8h, v18.8h, v4.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v6.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v4.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v6.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v4.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v6.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v4.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v6.h[5]     \n"
                        "fmla   v28.8h, v22.8h, v4.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v6.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v4.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v6.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v8.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v10.h[0]    \n"
                        "fmla   v30.8h, v17.8h, v8.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v10.h[1]    \n"
                        "fmla   v28.8h, v18.8h, v8.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v10.h[2]    \n"
                        "fmla   v30.8h, v19.8h, v8.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v10.h[3]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v8.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v10.h[4]    \n"
                        "fmla   v30.8h, v21.8h, v8.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v10.h[5]    \n"
                        "fmla   v28.8h, v22.8h, v8.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v10.h[6]    \n"
                        "fmla   v30.8h, v23.8h, v8.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v10.h[7]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v9.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v11.h[0]    \n"
                        "fmla   v30.8h, v17.8h, v9.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v11.h[1]    \n"

                        "prfm   pldl1keep, [%4, #384]       \n"
                        "ld1    {v12.8h, v13.8h, v14.8h}, [%4] \n" // r34 r35 r36

                        "fmla   v28.8h, v18.8h, v9.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v11.h[2]    \n"
                        "fmla   v30.8h, v19.8h, v9.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v11.h[3]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v9.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v11.h[4]    \n"
                        "fmla   v30.8h, v21.8h, v9.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v11.h[5]    \n"
                        "fmla   v28.8h, v22.8h, v9.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v11.h[6]    \n"
                        "fmla   v30.8h, v23.8h, v9.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v11.h[7]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v10.h[0]    \n"
                        "fmla   v29.8h, v16.8h, v12.h[0]    \n"
                        "fmla   v30.8h, v17.8h, v10.h[1]    \n"
                        "fmla   v31.8h, v17.8h, v12.h[1]    \n"
                        "fmla   v28.8h, v18.8h, v10.h[2]    \n"
                        "fmla   v29.8h, v18.8h, v12.h[2]    \n"
                        "fmla   v30.8h, v19.8h, v10.h[3]    \n"
                        "fmla   v31.8h, v19.8h, v12.h[3]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v10.h[4]    \n"
                        "fmla   v29.8h, v20.8h, v12.h[4]    \n"
                        "fmla   v30.8h, v21.8h, v10.h[5]    \n"
                        "fmla   v31.8h, v21.8h, v12.h[5]    \n"
                        "fmla   v28.8h, v22.8h, v10.h[6]    \n"
                        "fmla   v29.8h, v22.8h, v12.h[6]    \n"
                        "fmla   v30.8h, v23.8h, v10.h[7]    \n"
                        "fmla   v31.8h, v23.8h, v12.h[7]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v11.h[0]    \n"
                        "fmla   v29.8h, v16.8h, v13.h[0]    \n"
                        "fmla   v30.8h, v17.8h, v11.h[1]    \n"
                        "fmla   v31.8h, v17.8h, v13.h[1]    \n"
                        "fmla   v28.8h, v18.8h, v11.h[2]    \n"
                        "fmla   v29.8h, v18.8h, v13.h[2]    \n"
                        "fmla   v30.8h, v19.8h, v11.h[3]    \n"
                        "fmla   v31.8h, v19.8h, v13.h[3]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v11.h[4]    \n"
                        "fmla   v29.8h, v20.8h, v13.h[4]    \n"
                        "fmla   v30.8h, v21.8h, v11.h[5]    \n"
                        "fmla   v31.8h, v21.8h, v13.h[5]    \n"
                        "fmla   v28.8h, v22.8h, v11.h[6]    \n"
                        "fmla   v29.8h, v22.8h, v13.h[6]    \n"
                        "fmla   v30.8h, v23.8h, v11.h[7]    \n"
                        "fmla   v31.8h, v23.8h, v13.h[7]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v12.h[0]    \n"
                        "fmla   v29.8h, v16.8h, v14.h[0]    \n"
                        "fmla   v30.8h, v17.8h, v12.h[1]    \n"
                        "fmla   v31.8h, v17.8h, v14.h[1]    \n"

                        "prfm   pldl1keep, [%5, #512]       \n"
                        "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%5], #64 \n" // r40 r41 r42 r43

                        "fmla   v28.8h, v18.8h, v12.h[2]    \n"
                        "fmla   v29.8h, v18.8h, v14.h[2]    \n"
                        "fmla   v30.8h, v19.8h, v12.h[3]    \n"
                        "fmla   v31.8h, v19.8h, v14.h[3]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v12.h[4]    \n"
                        "fmla   v29.8h, v20.8h, v14.h[4]    \n"
                        "fmla   v30.8h, v21.8h, v12.h[5]    \n"
                        "fmla   v31.8h, v21.8h, v14.h[5]    \n"
                        "fmla   v28.8h, v22.8h, v12.h[6]    \n"
                        "fmla   v29.8h, v22.8h, v14.h[6]    \n"
                        "fmla   v30.8h, v23.8h, v12.h[7]    \n"
                        "fmla   v31.8h, v23.8h, v14.h[7]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v0.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v0.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v2.h[1]     \n"
                        "fmla   v28.8h, v18.8h, v0.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v0.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v2.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v0.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v0.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v2.h[5]     \n"
                        "fmla   v28.8h, v22.8h, v0.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v0.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v2.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v1.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v3.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v1.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v3.h[1]     \n"

                        "prfm   pldl1keep, [%5, #384]       \n"
                        "ld1    {v4.8h, v5.8h, v6.8h}, [%5] \n" // r44 r45 r46

                        "fmla   v28.8h, v18.8h, v1.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v3.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v1.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v3.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v1.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v3.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v1.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v3.h[5]     \n"
                        "fmla   v28.8h, v22.8h, v1.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v3.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v1.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v3.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v4.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v2.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v4.h[1]     \n"
                        "fmla   v28.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v4.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v2.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v4.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v4.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v2.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v4.h[5]     \n"
                        "fmla   v28.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v4.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v2.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v4.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v3.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v5.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v3.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v5.h[1]     \n"
                        "fmla   v28.8h, v18.8h, v3.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v5.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v3.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v5.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v3.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v5.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v3.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v5.h[5]     \n"
                        "fmla   v28.8h, v22.8h, v3.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v5.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v3.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v5.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6] \n"

                        "fmla   v28.8h, v16.8h, v4.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v6.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v4.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v6.h[1]     \n"
                        "fmla   v28.8h, v18.8h, v4.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v6.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v4.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v6.h[3]     \n"
                        "fmla   v28.8h, v20.8h, v4.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v6.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v4.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v6.h[5]     \n"
                        "fmla   v28.8h, v22.8h, v4.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v6.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v4.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v6.h[7]     \n"

                        "fadd   v28.8h, v28.8h, v30.8h      \n"
                        "fadd   v29.8h, v29.8h, v31.8h      \n"

                        "sub    %6, %6, #3136               \n" // kptr -= 24.5 * 64;

                        "st1    {v28.8h, v29.8h}, [%0], #32 \n"

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
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v28", "v29", "v30", "v31");
                }
                for (; j < outw; j++)
                {
                    asm volatile(
                        "prfm   pldl1keep, [%1, #256]       \n"
                        "ld1    {v0.8h, v1.8h}, [%1], #32   \n" // r00 r01

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "prfm   pldl1keep, [%0, #128]       \n"
                        "ld1    {v31.8h}, [%0]              \n" // sum0

                        "fmul   v28.8h, v16.8h, v0.h[0]     \n"
                        "fmul   v29.8h, v17.8h, v0.h[1]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmul   v30.8h, v18.8h, v0.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v0.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v0.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v0.h[5]     \n"
                        "fmla   v30.8h, v22.8h, v0.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v0.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v1.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v1.h[1]     \n"

                        "prfm   pldl1keep, [%1, #384]       \n"
                        "ld1    {v2.8h, v3.8h, v4.8h}, [%1] \n" // r02 r03 r04

                        "fmla   v30.8h, v18.8h, v1.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v1.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v1.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v1.h[5]     \n"
                        "fmla   v30.8h, v22.8h, v1.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v1.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v2.h[1]     \n"
                        "fmla   v30.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v2.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v2.h[5]     \n"
                        "fmla   v30.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v2.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v3.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v3.h[1]     \n"
                        "fmla   v30.8h, v18.8h, v3.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v3.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v3.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v3.h[5]     \n"
                        "fmla   v30.8h, v22.8h, v3.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v3.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v4.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v4.h[1]     \n"

                        "prfm   pldl1keep, [%2, #256]       \n"
                        "ld1    {v8.8h, v9.8h}, [%2], #32   \n" // r10 r11

                        "fmla   v30.8h, v18.8h, v4.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v4.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v4.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v4.h[5]     \n"
                        "fmla   v30.8h, v22.8h, v4.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v4.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v8.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v8.h[1]     \n"
                        "fmla   v30.8h, v18.8h, v8.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v8.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v8.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v8.h[5]     \n"
                        "fmla   v30.8h, v22.8h, v8.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v8.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v9.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v9.h[1]     \n"

                        "prfm   pldl1keep, [%2, #384]       \n"
                        "ld1    {v10.8h, v11.8h, v12.8h}, [%2] \n" // r12 r13 r14

                        "fmla   v30.8h, v18.8h, v9.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v9.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v9.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v9.h[5]     \n"
                        "fmla   v30.8h, v22.8h, v9.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v9.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v10.h[0]    \n"
                        "fmla   v29.8h, v17.8h, v10.h[1]    \n"
                        "fmla   v30.8h, v18.8h, v10.h[2]    \n"
                        "fmla   v31.8h, v19.8h, v10.h[3]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v10.h[4]    \n"
                        "fmla   v29.8h, v21.8h, v10.h[5]    \n"
                        "fmla   v30.8h, v22.8h, v10.h[6]    \n"
                        "fmla   v31.8h, v23.8h, v10.h[7]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v11.h[0]    \n"
                        "fmla   v29.8h, v17.8h, v11.h[1]    \n"
                        "fmla   v30.8h, v18.8h, v11.h[2]    \n"
                        "fmla   v31.8h, v19.8h, v11.h[3]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v11.h[4]    \n"
                        "fmla   v29.8h, v21.8h, v11.h[5]    \n"
                        "fmla   v30.8h, v22.8h, v11.h[6]    \n"
                        "fmla   v31.8h, v23.8h, v11.h[7]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v12.h[0]    \n"
                        "fmla   v29.8h, v17.8h, v12.h[1]    \n"

                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld1    {v0.8h, v1.8h}, [%3], #32   \n" // r20 r21

                        "fmla   v30.8h, v18.8h, v12.h[2]    \n"
                        "fmla   v31.8h, v19.8h, v12.h[3]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v12.h[4]    \n"
                        "fmla   v29.8h, v21.8h, v12.h[5]    \n"
                        "fmla   v30.8h, v22.8h, v12.h[6]    \n"
                        "fmla   v31.8h, v23.8h, v12.h[7]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v0.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v0.h[1]     \n"
                        "fmla   v30.8h, v18.8h, v0.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v0.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v0.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v0.h[5]     \n"
                        "fmla   v30.8h, v22.8h, v0.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v0.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v1.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v1.h[1]     \n"

                        "prfm   pldl1keep, [%3, #384]       \n"
                        "ld1    {v2.8h, v3.8h, v4.8h}, [%3] \n" // r22 r23 r24

                        "fmla   v30.8h, v18.8h, v1.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v1.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v1.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v1.h[5]     \n"
                        "fmla   v30.8h, v22.8h, v1.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v1.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v2.h[1]     \n"
                        "fmla   v30.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v2.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v2.h[5]     \n"
                        "fmla   v30.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v2.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v3.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v3.h[1]     \n"
                        "fmla   v30.8h, v18.8h, v3.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v3.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v3.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v3.h[5]     \n"
                        "fmla   v30.8h, v22.8h, v3.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v3.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v4.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v4.h[1]     \n"

                        "prfm   pldl1keep, [%4, #256]       \n"
                        "ld1    {v8.8h, v9.8h}, [%4], #32   \n" // r30 r31

                        "fmla   v30.8h, v18.8h, v4.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v4.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v4.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v4.h[5]     \n"
                        "fmla   v30.8h, v22.8h, v4.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v4.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v8.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v8.h[1]     \n"
                        "fmla   v30.8h, v18.8h, v8.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v8.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v8.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v8.h[5]     \n"
                        "fmla   v30.8h, v22.8h, v8.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v8.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v9.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v9.h[1]     \n"

                        "prfm   pldl1keep, [%4, #384]       \n"
                        "ld1    {v10.8h, v11.8h, v12.8h}, [%4] \n" // r32 r33 r34

                        "fmla   v30.8h, v18.8h, v9.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v9.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v9.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v9.h[5]     \n"
                        "fmla   v30.8h, v22.8h, v9.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v9.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v10.h[0]    \n"
                        "fmla   v29.8h, v17.8h, v10.h[1]    \n"
                        "fmla   v30.8h, v18.8h, v10.h[2]    \n"
                        "fmla   v31.8h, v19.8h, v10.h[3]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v10.h[4]    \n"
                        "fmla   v29.8h, v21.8h, v10.h[5]    \n"
                        "fmla   v30.8h, v22.8h, v10.h[6]    \n"
                        "fmla   v31.8h, v23.8h, v10.h[7]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v11.h[0]    \n"
                        "fmla   v29.8h, v17.8h, v11.h[1]    \n"
                        "fmla   v30.8h, v18.8h, v11.h[2]    \n"
                        "fmla   v31.8h, v19.8h, v11.h[3]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v11.h[4]    \n"
                        "fmla   v29.8h, v21.8h, v11.h[5]    \n"
                        "fmla   v30.8h, v22.8h, v11.h[6]    \n"
                        "fmla   v31.8h, v23.8h, v11.h[7]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v12.h[0]    \n"
                        "fmla   v29.8h, v17.8h, v12.h[1]    \n"

                        "prfm   pldl1keep, [%5, #256]       \n"
                        "ld1    {v0.8h, v1.8h}, [%5], #32   \n" // r40 r41

                        "fmla   v30.8h, v18.8h, v12.h[2]    \n"
                        "fmla   v31.8h, v19.8h, v12.h[3]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v12.h[4]    \n"
                        "fmla   v29.8h, v21.8h, v12.h[5]    \n"
                        "fmla   v30.8h, v22.8h, v12.h[6]    \n"
                        "fmla   v31.8h, v23.8h, v12.h[7]    \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v0.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v0.h[1]     \n"
                        "fmla   v30.8h, v18.8h, v0.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v0.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v0.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v0.h[5]     \n"
                        "fmla   v30.8h, v22.8h, v0.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v0.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v1.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v1.h[1]     \n"

                        "prfm   pldl1keep, [%5, #384]       \n"
                        "ld1    {v2.8h, v3.8h, v4.8h}, [%5] \n" // r42 r43 r44

                        "fmla   v30.8h, v18.8h, v1.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v1.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v1.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v1.h[5]     \n"
                        "fmla   v30.8h, v22.8h, v1.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v1.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v2.h[1]     \n"
                        "fmla   v30.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v2.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v2.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v2.h[5]     \n"
                        "fmla   v30.8h, v22.8h, v2.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v2.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v16.8h, v3.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v3.h[1]     \n"
                        "fmla   v30.8h, v18.8h, v3.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v3.h[3]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n"

                        "fmla   v28.8h, v20.8h, v3.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v3.h[5]     \n"
                        "fmla   v30.8h, v22.8h, v3.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v3.h[7]     \n"

                        "prfm   pldl1keep, [%6, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6] \n"

                        "fmla   v28.8h, v16.8h, v4.h[0]     \n"
                        "fmla   v29.8h, v17.8h, v4.h[1]     \n"
                        "fmla   v30.8h, v18.8h, v4.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v4.h[3]     \n"
                        "fmla   v28.8h, v20.8h, v4.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v4.h[5]     \n"
                        "fmla   v30.8h, v22.8h, v4.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v4.h[7]     \n"

                        "fadd   v28.8h, v28.8h, v29.8h      \n"
                        "fadd   v30.8h, v30.8h, v31.8h      \n"
                        "fadd   v28.8h, v28.8h, v30.8h      \n"

                        "sub    %6, %6, #3136               \n" // kptr -= 24.5 * 64;

                        "st1    {v28.8h}, [%0], #16         \n"

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
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v8", "v9", "v10", "v11", "v12", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v28", "v29", "v30", "v31");
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
