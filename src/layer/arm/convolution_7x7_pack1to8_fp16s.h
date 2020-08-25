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

static void conv7x7s2_pack1to8_fp16sa_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2 * outw + w;

    const __fp16* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        float16x8_t _bias0 = bias ? vld1q_f16(bias + p * 8) : vdupq_n_f16(0.f);
        out0.fill(_bias0);

        for (int q = 0; q < inch; q++)
        {
            __fp16* outptr0 = out0;

            const Mat img0 = bottom_blob.channel(q);

            const __fp16* r0 = img0.row<const __fp16>(0);
            const __fp16* r1 = img0.row<const __fp16>(1);
            const __fp16* r2 = img0.row<const __fp16>(2);
            const __fp16* r3 = img0.row<const __fp16>(3);
            const __fp16* r4 = img0.row<const __fp16>(4);
            const __fp16* r5 = img0.row<const __fp16>(5);
            const __fp16* r6 = img0.row<const __fp16>(6);

            const __fp16* kptr = kernel.channel(p).row<const __fp16>(q);

            int i = 0;

            for (; i < outh; i++)
            {
                int j = 0;
                for (; j + 7 < outw; j += 8)
                {
                    asm volatile(
                        "prfm   pldl1keep, [%0, #512]       \n"
                        "ld1    {v24.8h, v25.8h, v26.8h, v27.8h}, [%0], #64 \n" // sum0

                        "prfm   pldl1keep, [%1, #384]       \n"
                        "ld1    {v0.8h, v1.8h, v2.8h}, [%1] \n" // r0

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%8], #64 \n"

                        "prfm   pldl1keep, [%0, #512]       \n"
                        "ld1    {v28.8h, v29.8h, v30.8h, v31.8h}, [%0] \n" // sum0

                        "fmla   v24.8h, v16.8h, v0.h[0]     \n"
                        "fmla   v25.8h, v16.8h, v0.h[2]     \n"
                        "fmla   v26.8h, v16.8h, v0.h[4]     \n"
                        "fmla   v27.8h, v16.8h, v0.h[6]     \n"
                        "fmla   v28.8h, v16.8h, v1.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v1.h[2]     \n"
                        "fmla   v30.8h, v16.8h, v1.h[4]     \n"
                        "fmla   v31.8h, v16.8h, v1.h[6]     \n"

                        "sub    %0, %0, #64                 \n"

                        "fmla   v24.8h, v17.8h, v0.h[1]     \n"
                        "fmla   v25.8h, v17.8h, v0.h[3]     \n"
                        "fmla   v26.8h, v17.8h, v0.h[5]     \n"
                        "fmla   v27.8h, v17.8h, v0.h[7]     \n"
                        "fmla   v28.8h, v17.8h, v1.h[1]     \n"
                        "fmla   v29.8h, v17.8h, v1.h[3]     \n"
                        "fmla   v30.8h, v17.8h, v1.h[5]     \n"
                        "fmla   v31.8h, v17.8h, v1.h[7]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%8], #64 \n"

                        "fmla   v24.8h, v18.8h, v0.h[2]     \n"
                        "fmla   v25.8h, v18.8h, v0.h[4]     \n"
                        "fmla   v26.8h, v18.8h, v0.h[6]     \n"
                        "fmla   v27.8h, v18.8h, v1.h[0]     \n"
                        "fmla   v28.8h, v18.8h, v1.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v1.h[4]     \n"
                        "fmla   v30.8h, v18.8h, v1.h[6]     \n"
                        "fmla   v31.8h, v18.8h, v2.h[0]     \n"

                        "fmla   v24.8h, v19.8h, v0.h[3]     \n"
                        "fmla   v25.8h, v19.8h, v0.h[5]     \n"
                        "fmla   v26.8h, v19.8h, v0.h[7]     \n"
                        "fmla   v27.8h, v19.8h, v1.h[1]     \n"
                        "fmla   v28.8h, v19.8h, v1.h[3]     \n"
                        "fmla   v29.8h, v19.8h, v1.h[5]     \n"
                        "fmla   v30.8h, v19.8h, v1.h[7]     \n"
                        "fmla   v31.8h, v19.8h, v2.h[1]     \n"

                        "prfm   pldl1keep, [%2, #384]       \n"
                        "ld1    {v4.8h, v5.8h, v6.8h}, [%2] \n" // r1

                        "fmla   v24.8h, v20.8h, v0.h[4]     \n"
                        "fmla   v25.8h, v20.8h, v0.h[6]     \n"
                        "fmla   v26.8h, v20.8h, v1.h[0]     \n"
                        "fmla   v27.8h, v20.8h, v1.h[2]     \n"
                        "fmla   v28.8h, v20.8h, v1.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v1.h[6]     \n"
                        "fmla   v30.8h, v20.8h, v2.h[0]     \n"
                        "fmla   v31.8h, v20.8h, v2.h[2]     \n"

                        "fmla   v24.8h, v21.8h, v0.h[5]     \n"
                        "fmla   v25.8h, v21.8h, v0.h[7]     \n"
                        "fmla   v26.8h, v21.8h, v1.h[1]     \n"
                        "fmla   v27.8h, v21.8h, v1.h[3]     \n"
                        "fmla   v28.8h, v21.8h, v1.h[5]     \n"
                        "fmla   v29.8h, v21.8h, v1.h[7]     \n"
                        "fmla   v30.8h, v21.8h, v2.h[1]     \n"
                        "fmla   v31.8h, v21.8h, v2.h[3]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%8], #64 \n"

                        "fmla   v24.8h, v22.8h, v0.h[6]     \n"
                        "fmla   v25.8h, v22.8h, v1.h[0]     \n"
                        "fmla   v26.8h, v22.8h, v1.h[2]     \n"
                        "fmla   v27.8h, v22.8h, v1.h[4]     \n"
                        "fmla   v28.8h, v22.8h, v1.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v2.h[0]     \n"
                        "fmla   v30.8h, v22.8h, v2.h[2]     \n"
                        "fmla   v31.8h, v22.8h, v2.h[4]     \n"

                        "fmla   v24.8h, v23.8h, v4.h[0]     \n"
                        "fmla   v25.8h, v23.8h, v4.h[2]     \n"
                        "fmla   v26.8h, v23.8h, v4.h[4]     \n"
                        "fmla   v27.8h, v23.8h, v4.h[6]     \n"
                        "fmla   v28.8h, v23.8h, v5.h[0]     \n"
                        "fmla   v29.8h, v23.8h, v5.h[2]     \n"
                        "fmla   v30.8h, v23.8h, v5.h[4]     \n"
                        "fmla   v31.8h, v23.8h, v5.h[6]     \n"

                        "fmla   v24.8h, v16.8h, v4.h[1]     \n"
                        "fmla   v25.8h, v16.8h, v4.h[3]     \n"
                        "fmla   v26.8h, v16.8h, v4.h[5]     \n"
                        "fmla   v27.8h, v16.8h, v4.h[7]     \n"
                        "fmla   v28.8h, v16.8h, v5.h[1]     \n"
                        "fmla   v29.8h, v16.8h, v5.h[3]     \n"
                        "fmla   v30.8h, v16.8h, v5.h[5]     \n"
                        "fmla   v31.8h, v16.8h, v5.h[7]     \n"

                        "fmla   v24.8h, v17.8h, v4.h[2]     \n"
                        "fmla   v25.8h, v17.8h, v4.h[4]     \n"
                        "fmla   v26.8h, v17.8h, v4.h[6]     \n"
                        "fmla   v27.8h, v17.8h, v5.h[0]     \n"
                        "fmla   v28.8h, v17.8h, v5.h[2]     \n"
                        "fmla   v29.8h, v17.8h, v5.h[4]     \n"
                        "fmla   v30.8h, v17.8h, v5.h[6]     \n"
                        "fmla   v31.8h, v17.8h, v6.h[0]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%8], #64 \n"

                        "fmla   v24.8h, v18.8h, v4.h[3]     \n"
                        "fmla   v25.8h, v18.8h, v4.h[5]     \n"
                        "fmla   v26.8h, v18.8h, v4.h[7]     \n"
                        "fmla   v27.8h, v18.8h, v5.h[1]     \n"
                        "fmla   v28.8h, v18.8h, v5.h[3]     \n"
                        "fmla   v29.8h, v18.8h, v5.h[5]     \n"
                        "fmla   v30.8h, v18.8h, v5.h[7]     \n"
                        "fmla   v31.8h, v18.8h, v6.h[1]     \n"

                        "fmla   v24.8h, v19.8h, v4.h[4]     \n"
                        "fmla   v25.8h, v19.8h, v4.h[6]     \n"
                        "fmla   v26.8h, v19.8h, v5.h[0]     \n"
                        "fmla   v27.8h, v19.8h, v5.h[2]     \n"
                        "fmla   v28.8h, v19.8h, v5.h[4]     \n"
                        "fmla   v29.8h, v19.8h, v5.h[6]     \n"
                        "fmla   v30.8h, v19.8h, v6.h[0]     \n"
                        "fmla   v31.8h, v19.8h, v6.h[2]     \n"

                        "prfm   pldl1keep, [%3, #384]       \n"
                        "ld1    {v0.8h, v1.8h, v2.8h}, [%3] \n" // r2

                        "fmla   v24.8h, v20.8h, v4.h[5]     \n"
                        "fmla   v25.8h, v20.8h, v4.h[7]     \n"
                        "fmla   v26.8h, v20.8h, v5.h[1]     \n"
                        "fmla   v27.8h, v20.8h, v5.h[3]     \n"
                        "fmla   v28.8h, v20.8h, v5.h[5]     \n"
                        "fmla   v29.8h, v20.8h, v5.h[7]     \n"
                        "fmla   v30.8h, v20.8h, v6.h[1]     \n"
                        "fmla   v31.8h, v20.8h, v6.h[3]     \n"

                        "fmla   v24.8h, v21.8h, v4.h[6]     \n"
                        "fmla   v25.8h, v21.8h, v5.h[0]     \n"
                        "fmla   v26.8h, v21.8h, v5.h[2]     \n"
                        "fmla   v27.8h, v21.8h, v5.h[4]     \n"
                        "fmla   v28.8h, v21.8h, v5.h[6]     \n"
                        "fmla   v29.8h, v21.8h, v6.h[0]     \n"
                        "fmla   v30.8h, v21.8h, v6.h[2]     \n"
                        "fmla   v31.8h, v21.8h, v6.h[4]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%8], #64 \n"

                        "fmla   v24.8h, v22.8h, v0.h[0]     \n"
                        "fmla   v25.8h, v22.8h, v0.h[2]     \n"
                        "fmla   v26.8h, v22.8h, v0.h[4]     \n"
                        "fmla   v27.8h, v22.8h, v0.h[6]     \n"
                        "fmla   v28.8h, v22.8h, v1.h[0]     \n"
                        "fmla   v29.8h, v22.8h, v1.h[2]     \n"
                        "fmla   v30.8h, v22.8h, v1.h[4]     \n"
                        "fmla   v31.8h, v22.8h, v1.h[6]     \n"

                        "fmla   v24.8h, v23.8h, v0.h[1]     \n"
                        "fmla   v25.8h, v23.8h, v0.h[3]     \n"
                        "fmla   v26.8h, v23.8h, v0.h[5]     \n"
                        "fmla   v27.8h, v23.8h, v0.h[7]     \n"
                        "fmla   v28.8h, v23.8h, v1.h[1]     \n"
                        "fmla   v29.8h, v23.8h, v1.h[3]     \n"
                        "fmla   v30.8h, v23.8h, v1.h[5]     \n"
                        "fmla   v31.8h, v23.8h, v1.h[7]     \n"

                        "fmla   v24.8h, v16.8h, v0.h[2]     \n"
                        "fmla   v25.8h, v16.8h, v0.h[4]     \n"
                        "fmla   v26.8h, v16.8h, v0.h[6]     \n"
                        "fmla   v27.8h, v16.8h, v1.h[0]     \n"
                        "fmla   v28.8h, v16.8h, v1.h[2]     \n"
                        "fmla   v29.8h, v16.8h, v1.h[4]     \n"
                        "fmla   v30.8h, v16.8h, v1.h[6]     \n"
                        "fmla   v31.8h, v16.8h, v2.h[0]     \n"

                        "fmla   v24.8h, v17.8h, v0.h[3]     \n"
                        "fmla   v25.8h, v17.8h, v0.h[5]     \n"
                        "fmla   v26.8h, v17.8h, v0.h[7]     \n"
                        "fmla   v27.8h, v17.8h, v1.h[1]     \n"
                        "fmla   v28.8h, v17.8h, v1.h[3]     \n"
                        "fmla   v29.8h, v17.8h, v1.h[5]     \n"
                        "fmla   v30.8h, v17.8h, v1.h[7]     \n"
                        "fmla   v31.8h, v17.8h, v2.h[1]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%8], #64 \n"

                        "fmla   v24.8h, v18.8h, v0.h[4]     \n"
                        "fmla   v25.8h, v18.8h, v0.h[6]     \n"
                        "fmla   v26.8h, v18.8h, v1.h[0]     \n"
                        "fmla   v27.8h, v18.8h, v1.h[2]     \n"
                        "fmla   v28.8h, v18.8h, v1.h[4]     \n"
                        "fmla   v29.8h, v18.8h, v1.h[6]     \n"
                        "fmla   v30.8h, v18.8h, v2.h[0]     \n"
                        "fmla   v31.8h, v18.8h, v2.h[2]     \n"

                        "prfm   pldl1keep, [%4, #384]       \n"
                        "ld1    {v4.8h, v5.8h, v6.8h}, [%4] \n" // r3

                        "fmla   v24.8h, v19.8h, v0.h[5]     \n"
                        "fmla   v25.8h, v19.8h, v0.h[7]     \n"
                        "fmla   v26.8h, v19.8h, v1.h[1]     \n"
                        "fmla   v27.8h, v19.8h, v1.h[3]     \n"
                        "fmla   v28.8h, v19.8h, v1.h[5]     \n"
                        "fmla   v29.8h, v19.8h, v1.h[7]     \n"
                        "fmla   v30.8h, v19.8h, v2.h[1]     \n"
                        "fmla   v31.8h, v19.8h, v2.h[3]     \n"

                        "fmla   v24.8h, v20.8h, v0.h[6]     \n"
                        "fmla   v25.8h, v20.8h, v1.h[0]     \n"
                        "fmla   v26.8h, v20.8h, v1.h[2]     \n"
                        "fmla   v27.8h, v20.8h, v1.h[4]     \n"
                        "fmla   v28.8h, v20.8h, v1.h[6]     \n"
                        "fmla   v29.8h, v20.8h, v2.h[0]     \n"
                        "fmla   v30.8h, v20.8h, v2.h[2]     \n"
                        "fmla   v31.8h, v20.8h, v2.h[4]     \n"

                        "fmla   v24.8h, v21.8h, v4.h[0]     \n"
                        "fmla   v25.8h, v21.8h, v4.h[2]     \n"
                        "fmla   v26.8h, v21.8h, v4.h[4]     \n"
                        "fmla   v27.8h, v21.8h, v4.h[6]     \n"
                        "fmla   v28.8h, v21.8h, v5.h[0]     \n"
                        "fmla   v29.8h, v21.8h, v5.h[2]     \n"
                        "fmla   v30.8h, v21.8h, v5.h[4]     \n"
                        "fmla   v31.8h, v21.8h, v5.h[6]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%8], #64 \n"

                        "fmla   v24.8h, v22.8h, v4.h[1]     \n"
                        "fmla   v25.8h, v22.8h, v4.h[3]     \n"
                        "fmla   v26.8h, v22.8h, v4.h[5]     \n"
                        "fmla   v27.8h, v22.8h, v4.h[7]     \n"
                        "fmla   v28.8h, v22.8h, v5.h[1]     \n"
                        "fmla   v29.8h, v22.8h, v5.h[3]     \n"
                        "fmla   v30.8h, v22.8h, v5.h[5]     \n"
                        "fmla   v31.8h, v22.8h, v5.h[7]     \n"

                        "fmla   v24.8h, v23.8h, v4.h[2]     \n"
                        "fmla   v25.8h, v23.8h, v4.h[4]     \n"
                        "fmla   v26.8h, v23.8h, v4.h[6]     \n"
                        "fmla   v27.8h, v23.8h, v5.h[0]     \n"
                        "fmla   v28.8h, v23.8h, v5.h[2]     \n"
                        "fmla   v29.8h, v23.8h, v5.h[4]     \n"
                        "fmla   v30.8h, v23.8h, v5.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v6.h[0]     \n"

                        "fmla   v24.8h, v16.8h, v4.h[3]     \n"
                        "fmla   v25.8h, v16.8h, v4.h[5]     \n"
                        "fmla   v26.8h, v16.8h, v4.h[7]     \n"
                        "fmla   v27.8h, v16.8h, v5.h[1]     \n"
                        "fmla   v28.8h, v16.8h, v5.h[3]     \n"
                        "fmla   v29.8h, v16.8h, v5.h[5]     \n"
                        "fmla   v30.8h, v16.8h, v5.h[7]     \n"
                        "fmla   v31.8h, v16.8h, v6.h[1]     \n"

                        "fmla   v24.8h, v17.8h, v4.h[4]     \n"
                        "fmla   v25.8h, v17.8h, v4.h[6]     \n"
                        "fmla   v26.8h, v17.8h, v5.h[0]     \n"
                        "fmla   v27.8h, v17.8h, v5.h[2]     \n"
                        "fmla   v28.8h, v17.8h, v5.h[4]     \n"
                        "fmla   v29.8h, v17.8h, v5.h[6]     \n"
                        "fmla   v30.8h, v17.8h, v6.h[0]     \n"
                        "fmla   v31.8h, v17.8h, v6.h[2]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%8], #64 \n"

                        "fmla   v24.8h, v18.8h, v4.h[5]     \n"
                        "fmla   v25.8h, v18.8h, v4.h[7]     \n"
                        "fmla   v26.8h, v18.8h, v5.h[1]     \n"
                        "fmla   v27.8h, v18.8h, v5.h[3]     \n"
                        "fmla   v28.8h, v18.8h, v5.h[5]     \n"
                        "fmla   v29.8h, v18.8h, v5.h[7]     \n"
                        "fmla   v30.8h, v18.8h, v6.h[1]     \n"
                        "fmla   v31.8h, v18.8h, v6.h[3]     \n"

                        "prfm   pldl1keep, [%5, #384]       \n"
                        "ld1    {v0.8h, v1.8h, v2.8h}, [%5] \n" // r4

                        "fmla   v24.8h, v19.8h, v4.h[6]     \n"
                        "fmla   v25.8h, v19.8h, v5.h[0]     \n"
                        "fmla   v26.8h, v19.8h, v5.h[2]     \n"
                        "fmla   v27.8h, v19.8h, v5.h[4]     \n"
                        "fmla   v28.8h, v19.8h, v5.h[6]     \n"
                        "fmla   v29.8h, v19.8h, v6.h[0]     \n"
                        "fmla   v30.8h, v19.8h, v6.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v6.h[4]     \n"

                        "fmla   v24.8h, v20.8h, v0.h[0]     \n"
                        "fmla   v25.8h, v20.8h, v0.h[2]     \n"
                        "fmla   v26.8h, v20.8h, v0.h[4]     \n"
                        "fmla   v27.8h, v20.8h, v0.h[6]     \n"
                        "fmla   v28.8h, v20.8h, v1.h[0]     \n"
                        "fmla   v29.8h, v20.8h, v1.h[2]     \n"
                        "fmla   v30.8h, v20.8h, v1.h[4]     \n"
                        "fmla   v31.8h, v20.8h, v1.h[6]     \n"

                        "fmla   v24.8h, v21.8h, v0.h[1]     \n"
                        "fmla   v25.8h, v21.8h, v0.h[3]     \n"
                        "fmla   v26.8h, v21.8h, v0.h[5]     \n"
                        "fmla   v27.8h, v21.8h, v0.h[7]     \n"
                        "fmla   v28.8h, v21.8h, v1.h[1]     \n"
                        "fmla   v29.8h, v21.8h, v1.h[3]     \n"
                        "fmla   v30.8h, v21.8h, v1.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v1.h[7]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%8], #64 \n"

                        "fmla   v24.8h, v22.8h, v0.h[2]     \n"
                        "fmla   v25.8h, v22.8h, v0.h[4]     \n"
                        "fmla   v26.8h, v22.8h, v0.h[6]     \n"
                        "fmla   v27.8h, v22.8h, v1.h[0]     \n"
                        "fmla   v28.8h, v22.8h, v1.h[2]     \n"
                        "fmla   v29.8h, v22.8h, v1.h[4]     \n"
                        "fmla   v30.8h, v22.8h, v1.h[6]     \n"
                        "fmla   v31.8h, v22.8h, v2.h[0]     \n"

                        "fmla   v24.8h, v23.8h, v0.h[3]     \n"
                        "fmla   v25.8h, v23.8h, v0.h[5]     \n"
                        "fmla   v26.8h, v23.8h, v0.h[7]     \n"
                        "fmla   v27.8h, v23.8h, v1.h[1]     \n"
                        "fmla   v28.8h, v23.8h, v1.h[3]     \n"
                        "fmla   v29.8h, v23.8h, v1.h[5]     \n"
                        "fmla   v30.8h, v23.8h, v1.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v2.h[1]     \n"

                        "prfm   pldl1keep, [%6, #384]       \n"
                        "ld1    {v4.8h, v5.8h, v6.8h}, [%6] \n" // r5

                        "fmla   v24.8h, v16.8h, v0.h[4]     \n"
                        "fmla   v25.8h, v16.8h, v0.h[6]     \n"
                        "fmla   v26.8h, v16.8h, v1.h[0]     \n"
                        "fmla   v27.8h, v16.8h, v1.h[2]     \n"
                        "fmla   v28.8h, v16.8h, v1.h[4]     \n"
                        "fmla   v29.8h, v16.8h, v1.h[6]     \n"
                        "fmla   v30.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v31.8h, v16.8h, v2.h[2]     \n"

                        "fmla   v24.8h, v17.8h, v0.h[5]     \n"
                        "fmla   v25.8h, v17.8h, v0.h[7]     \n"
                        "fmla   v26.8h, v17.8h, v1.h[1]     \n"
                        "fmla   v27.8h, v17.8h, v1.h[3]     \n"
                        "fmla   v28.8h, v17.8h, v1.h[5]     \n"
                        "fmla   v29.8h, v17.8h, v1.h[7]     \n"
                        "fmla   v30.8h, v17.8h, v2.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v2.h[3]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%8], #64 \n"

                        "fmla   v24.8h, v18.8h, v0.h[6]     \n"
                        "fmla   v25.8h, v18.8h, v1.h[0]     \n"
                        "fmla   v26.8h, v18.8h, v1.h[2]     \n"
                        "fmla   v27.8h, v18.8h, v1.h[4]     \n"
                        "fmla   v28.8h, v18.8h, v1.h[6]     \n"
                        "fmla   v30.8h, v18.8h, v2.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v2.h[0]     \n"
                        "fmla   v31.8h, v18.8h, v2.h[4]     \n"

                        "fmla   v24.8h, v19.8h, v4.h[0]     \n"
                        "fmla   v25.8h, v19.8h, v4.h[2]     \n"
                        "fmla   v26.8h, v19.8h, v4.h[4]     \n"
                        "fmla   v27.8h, v19.8h, v4.h[6]     \n"
                        "fmla   v28.8h, v19.8h, v5.h[0]     \n"
                        "fmla   v29.8h, v19.8h, v5.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v5.h[4]     \n"
                        "fmla   v31.8h, v19.8h, v5.h[6]     \n"

                        "fmla   v24.8h, v20.8h, v4.h[1]     \n"
                        "fmla   v25.8h, v20.8h, v4.h[3]     \n"
                        "fmla   v26.8h, v20.8h, v4.h[5]     \n"
                        "fmla   v27.8h, v20.8h, v4.h[7]     \n"
                        "fmla   v28.8h, v20.8h, v5.h[1]     \n"
                        "fmla   v29.8h, v20.8h, v5.h[3]     \n"
                        "fmla   v30.8h, v20.8h, v5.h[5]     \n"
                        "fmla   v31.8h, v20.8h, v5.h[7]     \n"

                        "fmla   v24.8h, v21.8h, v4.h[2]     \n"
                        "fmla   v25.8h, v21.8h, v4.h[4]     \n"
                        "fmla   v26.8h, v21.8h, v4.h[6]     \n"
                        "fmla   v27.8h, v21.8h, v5.h[0]     \n"
                        "fmla   v28.8h, v21.8h, v5.h[2]     \n"
                        "fmla   v29.8h, v21.8h, v5.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v5.h[6]     \n"
                        "fmla   v31.8h, v21.8h, v6.h[0]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%8], #64 \n"

                        "fmla   v24.8h, v22.8h, v4.h[3]     \n"
                        "fmla   v25.8h, v22.8h, v4.h[5]     \n"
                        "fmla   v26.8h, v22.8h, v4.h[7]     \n"
                        "fmla   v27.8h, v22.8h, v5.h[1]     \n"
                        "fmla   v28.8h, v22.8h, v5.h[3]     \n"
                        "fmla   v29.8h, v22.8h, v5.h[5]     \n"
                        "fmla   v30.8h, v22.8h, v5.h[7]     \n"
                        "fmla   v31.8h, v22.8h, v6.h[1]     \n"

                        "fmla   v24.8h, v23.8h, v4.h[4]     \n"
                        "fmla   v25.8h, v23.8h, v4.h[6]     \n"
                        "fmla   v26.8h, v23.8h, v5.h[0]     \n"
                        "fmla   v27.8h, v23.8h, v5.h[2]     \n"
                        "fmla   v28.8h, v23.8h, v5.h[4]     \n"
                        "fmla   v29.8h, v23.8h, v5.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v6.h[0]     \n"
                        "fmla   v31.8h, v23.8h, v6.h[2]     \n"

                        "prfm   pldl1keep, [%7, #384]       \n"
                        "ld1    {v0.8h, v1.8h, v2.8h}, [%7] \n" // r6

                        "fmla   v24.8h, v16.8h, v4.h[5]     \n"
                        "fmla   v25.8h, v16.8h, v4.h[7]     \n"
                        "fmla   v26.8h, v16.8h, v5.h[1]     \n"
                        "fmla   v27.8h, v16.8h, v5.h[3]     \n"
                        "fmla   v28.8h, v16.8h, v5.h[5]     \n"
                        "fmla   v29.8h, v16.8h, v5.h[7]     \n"
                        "fmla   v30.8h, v16.8h, v6.h[1]     \n"
                        "fmla   v31.8h, v16.8h, v6.h[3]     \n"

                        "fmla   v24.8h, v17.8h, v4.h[6]     \n"
                        "fmla   v25.8h, v17.8h, v5.h[0]     \n"
                        "fmla   v26.8h, v17.8h, v5.h[2]     \n"
                        "fmla   v27.8h, v17.8h, v5.h[4]     \n"
                        "fmla   v28.8h, v17.8h, v5.h[6]     \n"
                        "fmla   v29.8h, v17.8h, v6.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v6.h[2]     \n"
                        "fmla   v31.8h, v17.8h, v6.h[4]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%8], #64 \n"

                        "fmla   v24.8h, v18.8h, v0.h[0]     \n"
                        "fmla   v25.8h, v18.8h, v0.h[2]     \n"
                        "fmla   v26.8h, v18.8h, v0.h[4]     \n"
                        "fmla   v27.8h, v18.8h, v0.h[6]     \n"
                        "fmla   v28.8h, v18.8h, v1.h[0]     \n"
                        "fmla   v29.8h, v18.8h, v1.h[2]     \n"
                        "fmla   v30.8h, v18.8h, v1.h[4]     \n"
                        "fmla   v31.8h, v18.8h, v1.h[6]     \n"

                        "fmla   v24.8h, v19.8h, v0.h[1]     \n"
                        "fmla   v25.8h, v19.8h, v0.h[3]     \n"
                        "fmla   v26.8h, v19.8h, v0.h[5]     \n"
                        "fmla   v27.8h, v19.8h, v0.h[7]     \n"
                        "fmla   v28.8h, v19.8h, v1.h[1]     \n"
                        "fmla   v29.8h, v19.8h, v1.h[3]     \n"
                        "fmla   v30.8h, v19.8h, v1.h[5]     \n"
                        "fmla   v31.8h, v19.8h, v1.h[7]     \n"

                        "fmla   v24.8h, v20.8h, v0.h[2]     \n"
                        "fmla   v25.8h, v20.8h, v0.h[4]     \n"
                        "fmla   v26.8h, v20.8h, v0.h[6]     \n"
                        "fmla   v27.8h, v20.8h, v1.h[0]     \n"
                        "fmla   v28.8h, v20.8h, v1.h[2]     \n"
                        "fmla   v29.8h, v20.8h, v1.h[4]     \n"
                        "fmla   v30.8h, v20.8h, v1.h[6]     \n"
                        "fmla   v31.8h, v20.8h, v2.h[0]     \n"

                        "add    %1, %1, #32                 \n"

                        "fmla   v24.8h, v21.8h, v0.h[3]     \n"
                        "fmla   v25.8h, v21.8h, v0.h[5]     \n"
                        "fmla   v26.8h, v21.8h, v0.h[7]     \n"
                        "fmla   v27.8h, v21.8h, v1.h[1]     \n"

                        "add    %2, %2, #32                 \n"

                        "fmla   v28.8h, v21.8h, v1.h[3]     \n"
                        "fmla   v29.8h, v21.8h, v1.h[5]     \n"
                        "fmla   v30.8h, v21.8h, v1.h[7]     \n"
                        "fmla   v31.8h, v21.8h, v2.h[1]     \n"

                        "prfm   pldl1keep, [%8, #128]       \n"
                        "ld1    {v16.8h}, [%8]              \n"

                        "fmla   v24.8h, v22.8h, v0.h[4]     \n"
                        "fmla   v25.8h, v22.8h, v0.h[6]     \n"
                        "fmla   v26.8h, v22.8h, v1.h[0]     \n"
                        "fmla   v27.8h, v22.8h, v1.h[2]     \n"

                        "add    %3, %3, #32                 \n"

                        "fmla   v28.8h, v22.8h, v1.h[4]     \n"
                        "fmla   v29.8h, v22.8h, v1.h[6]     \n"
                        "fmla   v30.8h, v22.8h, v2.h[0]     \n"
                        "fmla   v31.8h, v22.8h, v2.h[2]     \n"

                        "add    %4, %4, #32                 \n"

                        "fmla   v24.8h, v23.8h, v0.h[5]     \n"
                        "fmla   v25.8h, v23.8h, v0.h[7]     \n"
                        "fmla   v26.8h, v23.8h, v1.h[1]     \n"
                        "fmla   v27.8h, v23.8h, v1.h[3]     \n"

                        "add    %5, %5, #32                 \n"

                        "fmla   v28.8h, v23.8h, v1.h[5]     \n"
                        "fmla   v29.8h, v23.8h, v1.h[7]     \n"
                        "fmla   v30.8h, v23.8h, v2.h[1]     \n"
                        "fmla   v31.8h, v23.8h, v2.h[3]     \n"

                        "add    %6, %6, #32                 \n"

                        "fmla   v24.8h, v16.8h, v0.h[6]     \n"
                        "fmla   v25.8h, v16.8h, v1.h[0]     \n"
                        "fmla   v26.8h, v16.8h, v1.h[2]     \n"
                        "fmla   v27.8h, v16.8h, v1.h[4]     \n"

                        "add    %7, %7, #32                 \n"

                        "fmla   v28.8h, v16.8h, v1.h[6]     \n"
                        "fmla   v29.8h, v16.8h, v2.h[0]     \n"
                        "fmla   v30.8h, v16.8h, v2.h[2]     \n"
                        "fmla   v31.8h, v16.8h, v2.h[4]     \n"

                        "sub    %8, %8, #768                \n" // kptr -= 48 * 8;

                        "st1    {v24.8h, v25.8h, v26.8h, v27.8h}, [%0], #64 \n"
                        "st1    {v28.8h, v29.8h, v30.8h, v31.8h}, [%0], #64 \n"

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
                        : "memory", "v0", "v1", "v2", "v4", "v5", "v6", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
                }
                for (; j + 3 < outw; j += 4)
                {
                    asm volatile(
                        "prfm   pldl1keep, [%0, #512]       \n"
                        "ld1    {v28.8h, v29.8h, v30.8h, v31.8h}, [%0] \n" // sum0

                        "prfm   pldl1keep, [%1, #256]       \n"
                        "ld1    {v0.8h, v1.8h}, [%1]        \n" // r0

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%8], #64 \n"

                        "fmla   v28.8h, v16.8h, v0.h[0]     \n"
                        "fmla   v29.8h, v16.8h, v0.h[2]     \n"
                        "fmla   v30.8h, v16.8h, v0.h[4]     \n"
                        "fmla   v31.8h, v16.8h, v0.h[6]     \n"

                        "fmla   v28.8h, v17.8h, v0.h[1]     \n"
                        "fmla   v29.8h, v17.8h, v0.h[3]     \n"
                        "fmla   v30.8h, v17.8h, v0.h[5]     \n"
                        "fmla   v31.8h, v17.8h, v0.h[7]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%8], #64 \n"

                        "fmla   v28.8h, v18.8h, v0.h[2]     \n"
                        "fmla   v29.8h, v18.8h, v0.h[4]     \n"
                        "fmla   v30.8h, v18.8h, v0.h[6]     \n"
                        "fmla   v31.8h, v18.8h, v1.h[0]     \n"

                        "fmla   v28.8h, v19.8h, v0.h[3]     \n"
                        "fmla   v29.8h, v19.8h, v0.h[5]     \n"
                        "fmla   v30.8h, v19.8h, v0.h[7]     \n"
                        "fmla   v31.8h, v19.8h, v1.h[1]     \n"

                        "prfm   pldl1keep, [%2, #256]       \n"
                        "ld1    {v2.8h, v3.8h}, [%2]        \n" // r1

                        "fmla   v28.8h, v20.8h, v0.h[4]     \n"
                        "fmla   v29.8h, v20.8h, v0.h[6]     \n"
                        "fmla   v30.8h, v20.8h, v1.h[0]     \n"
                        "fmla   v31.8h, v20.8h, v1.h[2]     \n"

                        "fmla   v28.8h, v21.8h, v0.h[5]     \n"
                        "fmla   v29.8h, v21.8h, v0.h[7]     \n"
                        "fmla   v30.8h, v21.8h, v1.h[1]     \n"
                        "fmla   v31.8h, v21.8h, v1.h[3]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%8], #64 \n"

                        "fmla   v28.8h, v22.8h, v0.h[6]     \n"
                        "fmla   v29.8h, v22.8h, v1.h[0]     \n"
                        "fmla   v30.8h, v22.8h, v1.h[2]     \n"
                        "fmla   v31.8h, v22.8h, v1.h[4]     \n"

                        "fmla   v28.8h, v23.8h, v2.h[0]     \n"
                        "fmla   v29.8h, v23.8h, v2.h[2]     \n"
                        "fmla   v30.8h, v23.8h, v2.h[4]     \n"
                        "fmla   v31.8h, v23.8h, v2.h[6]     \n"

                        "fmla   v28.8h, v16.8h, v2.h[1]     \n"
                        "fmla   v29.8h, v16.8h, v2.h[3]     \n"
                        "fmla   v30.8h, v16.8h, v2.h[5]     \n"
                        "fmla   v31.8h, v16.8h, v2.h[7]     \n"

                        "fmla   v28.8h, v17.8h, v2.h[2]     \n"
                        "fmla   v29.8h, v17.8h, v2.h[4]     \n"
                        "fmla   v30.8h, v17.8h, v2.h[6]     \n"
                        "fmla   v31.8h, v17.8h, v3.h[0]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%8], #64 \n"

                        "fmla   v28.8h, v18.8h, v2.h[3]     \n"
                        "fmla   v29.8h, v18.8h, v2.h[5]     \n"
                        "fmla   v30.8h, v18.8h, v2.h[7]     \n"
                        "fmla   v31.8h, v18.8h, v3.h[1]     \n"

                        "fmla   v28.8h, v19.8h, v2.h[4]     \n"
                        "fmla   v29.8h, v19.8h, v2.h[6]     \n"
                        "fmla   v30.8h, v19.8h, v3.h[0]     \n"
                        "fmla   v31.8h, v19.8h, v3.h[2]     \n"

                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld1    {v0.8h, v1.8h}, [%3]        \n" // r2

                        "fmla   v28.8h, v20.8h, v2.h[5]     \n"
                        "fmla   v29.8h, v20.8h, v2.h[7]     \n"
                        "fmla   v30.8h, v20.8h, v3.h[1]     \n"
                        "fmla   v31.8h, v20.8h, v3.h[3]     \n"

                        "fmla   v28.8h, v21.8h, v2.h[6]     \n"
                        "fmla   v29.8h, v21.8h, v3.h[0]     \n"
                        "fmla   v30.8h, v21.8h, v3.h[2]     \n"
                        "fmla   v31.8h, v21.8h, v3.h[4]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%8], #64 \n"

                        "fmla   v28.8h, v22.8h, v0.h[0]     \n"
                        "fmla   v29.8h, v22.8h, v0.h[2]     \n"
                        "fmla   v30.8h, v22.8h, v0.h[4]     \n"
                        "fmla   v31.8h, v22.8h, v0.h[6]     \n"

                        "fmla   v28.8h, v23.8h, v0.h[1]     \n"
                        "fmla   v29.8h, v23.8h, v0.h[3]     \n"
                        "fmla   v30.8h, v23.8h, v0.h[5]     \n"
                        "fmla   v31.8h, v23.8h, v0.h[7]     \n"

                        "fmla   v28.8h, v16.8h, v0.h[2]     \n"
                        "fmla   v29.8h, v16.8h, v0.h[4]     \n"
                        "fmla   v30.8h, v16.8h, v0.h[6]     \n"
                        "fmla   v31.8h, v16.8h, v1.h[0]     \n"

                        "fmla   v28.8h, v17.8h, v0.h[3]     \n"
                        "fmla   v29.8h, v17.8h, v0.h[5]     \n"
                        "fmla   v30.8h, v17.8h, v0.h[7]     \n"
                        "fmla   v31.8h, v17.8h, v1.h[1]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%8], #64 \n"

                        "fmla   v28.8h, v18.8h, v0.h[4]     \n"
                        "fmla   v29.8h, v18.8h, v0.h[6]     \n"
                        "fmla   v30.8h, v18.8h, v1.h[0]     \n"
                        "fmla   v31.8h, v18.8h, v1.h[2]     \n"

                        "prfm   pldl1keep, [%4, #256]       \n"
                        "ld1    {v2.8h, v3.8h}, [%4]        \n" // r3

                        "fmla   v28.8h, v19.8h, v0.h[5]     \n"
                        "fmla   v29.8h, v19.8h, v0.h[7]     \n"
                        "fmla   v30.8h, v19.8h, v1.h[1]     \n"
                        "fmla   v31.8h, v19.8h, v1.h[3]     \n"

                        "fmla   v28.8h, v20.8h, v0.h[6]     \n"
                        "fmla   v29.8h, v20.8h, v1.h[0]     \n"
                        "fmla   v30.8h, v20.8h, v1.h[2]     \n"
                        "fmla   v31.8h, v20.8h, v1.h[4]     \n"

                        "fmla   v28.8h, v21.8h, v2.h[0]     \n"
                        "fmla   v29.8h, v21.8h, v2.h[2]     \n"
                        "fmla   v30.8h, v21.8h, v2.h[4]     \n"
                        "fmla   v31.8h, v21.8h, v2.h[6]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%8], #64 \n"

                        "fmla   v28.8h, v22.8h, v2.h[1]     \n"
                        "fmla   v29.8h, v22.8h, v2.h[3]     \n"
                        "fmla   v30.8h, v22.8h, v2.h[5]     \n"
                        "fmla   v31.8h, v22.8h, v2.h[7]     \n"

                        "fmla   v28.8h, v23.8h, v2.h[2]     \n"
                        "fmla   v29.8h, v23.8h, v2.h[4]     \n"
                        "fmla   v30.8h, v23.8h, v2.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v3.h[0]     \n"

                        "fmla   v28.8h, v16.8h, v2.h[3]     \n"
                        "fmla   v29.8h, v16.8h, v2.h[5]     \n"
                        "fmla   v30.8h, v16.8h, v2.h[7]     \n"
                        "fmla   v31.8h, v16.8h, v3.h[1]     \n"

                        "prfm   pldl1keep, [%5, #256]       \n"
                        "ld1    {v0.8h, v1.8h}, [%5]        \n" // r4

                        "fmla   v28.8h, v17.8h, v2.h[4]     \n"
                        "fmla   v29.8h, v17.8h, v2.h[6]     \n"
                        "fmla   v30.8h, v17.8h, v3.h[0]     \n"
                        "fmla   v31.8h, v17.8h, v3.h[2]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%8], #64 \n"

                        "fmla   v28.8h, v18.8h, v2.h[5]     \n"
                        "fmla   v29.8h, v18.8h, v2.h[7]     \n"
                        "fmla   v30.8h, v18.8h, v3.h[1]     \n"
                        "fmla   v31.8h, v18.8h, v3.h[3]     \n"

                        "fmla   v28.8h, v19.8h, v2.h[6]     \n"
                        "fmla   v29.8h, v19.8h, v3.h[0]     \n"
                        "fmla   v30.8h, v19.8h, v3.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v3.h[4]     \n"

                        "fmla   v28.8h, v20.8h, v0.h[0]     \n"
                        "fmla   v29.8h, v20.8h, v0.h[2]     \n"
                        "fmla   v30.8h, v20.8h, v0.h[4]     \n"
                        "fmla   v31.8h, v20.8h, v0.h[6]     \n"

                        "fmla   v28.8h, v21.8h, v0.h[1]     \n"
                        "fmla   v29.8h, v21.8h, v0.h[3]     \n"
                        "fmla   v30.8h, v21.8h, v0.h[5]     \n"
                        "fmla   v31.8h, v21.8h, v0.h[7]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%8], #64 \n"

                        "fmla   v28.8h, v22.8h, v0.h[2]     \n"
                        "fmla   v29.8h, v22.8h, v0.h[4]     \n"
                        "fmla   v30.8h, v22.8h, v0.h[6]     \n"
                        "fmla   v31.8h, v22.8h, v1.h[0]     \n"

                        "fmla   v28.8h, v23.8h, v0.h[3]     \n"
                        "fmla   v29.8h, v23.8h, v0.h[5]     \n"
                        "fmla   v30.8h, v23.8h, v0.h[7]     \n"
                        "fmla   v31.8h, v23.8h, v1.h[1]     \n"

                        "prfm   pldl1keep, [%6, #256]       \n"
                        "ld1    {v2.8h, v3.8h}, [%6]        \n" // r5

                        "fmla   v28.8h, v16.8h, v0.h[4]     \n"
                        "fmla   v29.8h, v16.8h, v0.h[6]     \n"
                        "fmla   v30.8h, v16.8h, v1.h[0]     \n"
                        "fmla   v31.8h, v16.8h, v1.h[2]     \n"

                        "fmla   v28.8h, v17.8h, v0.h[5]     \n"
                        "fmla   v29.8h, v17.8h, v0.h[7]     \n"
                        "fmla   v30.8h, v17.8h, v1.h[1]     \n"
                        "fmla   v31.8h, v17.8h, v1.h[3]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%8], #64 \n"

                        "fmla   v28.8h, v18.8h, v0.h[6]     \n"
                        "fmla   v29.8h, v18.8h, v1.h[0]     \n"
                        "fmla   v30.8h, v18.8h, v1.h[2]     \n"
                        "fmla   v31.8h, v18.8h, v1.h[4]     \n"

                        "fmla   v28.8h, v19.8h, v2.h[0]     \n"
                        "fmla   v29.8h, v19.8h, v2.h[2]     \n"
                        "fmla   v30.8h, v19.8h, v2.h[4]     \n"
                        "fmla   v31.8h, v19.8h, v2.h[6]     \n"

                        "fmla   v28.8h, v20.8h, v2.h[1]     \n"
                        "fmla   v29.8h, v20.8h, v2.h[3]     \n"
                        "fmla   v30.8h, v20.8h, v2.h[5]     \n"
                        "fmla   v31.8h, v20.8h, v2.h[7]     \n"

                        "fmla   v28.8h, v21.8h, v2.h[2]     \n"
                        "fmla   v29.8h, v21.8h, v2.h[4]     \n"
                        "fmla   v30.8h, v21.8h, v2.h[6]     \n"
                        "fmla   v31.8h, v21.8h, v3.h[0]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%8], #64 \n"

                        "fmla   v28.8h, v22.8h, v2.h[3]     \n"
                        "fmla   v29.8h, v22.8h, v2.h[5]     \n"
                        "fmla   v30.8h, v22.8h, v2.h[7]     \n"
                        "fmla   v31.8h, v22.8h, v3.h[1]     \n"

                        "add    %1, %1, #16                 \n"

                        "fmla   v28.8h, v23.8h, v2.h[4]     \n"
                        "fmla   v29.8h, v23.8h, v2.h[6]     \n"
                        "fmla   v30.8h, v23.8h, v3.h[0]     \n"
                        "fmla   v31.8h, v23.8h, v3.h[2]     \n"

                        "prfm   pldl1keep, [%7, #256]       \n"
                        "ld1    {v0.8h, v1.8h}, [%7]        \n" // r6

                        "fmla   v28.8h, v16.8h, v2.h[5]     \n"
                        "fmla   v29.8h, v16.8h, v2.h[7]     \n"
                        "fmla   v30.8h, v16.8h, v3.h[1]     \n"
                        "fmla   v31.8h, v16.8h, v3.h[3]     \n"

                        "add    %2, %2, #16                 \n"

                        "fmla   v28.8h, v17.8h, v2.h[6]     \n"
                        "fmla   v29.8h, v17.8h, v3.h[0]     \n"
                        "fmla   v30.8h, v17.8h, v3.h[2]     \n"
                        "fmla   v31.8h, v17.8h, v3.h[4]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%8], #64 \n"

                        "fmla   v28.8h, v18.8h, v0.h[0]     \n"
                        "fmla   v29.8h, v18.8h, v0.h[2]     \n"
                        "fmla   v30.8h, v18.8h, v0.h[4]     \n"
                        "fmla   v31.8h, v18.8h, v0.h[6]     \n"

                        "add    %3, %3, #16                 \n"

                        "fmla   v28.8h, v19.8h, v0.h[1]     \n"
                        "fmla   v29.8h, v19.8h, v0.h[3]     \n"
                        "fmla   v30.8h, v19.8h, v0.h[5]     \n"
                        "fmla   v31.8h, v19.8h, v0.h[7]     \n"

                        "add    %4, %4, #16                 \n"

                        "fmla   v28.8h, v20.8h, v0.h[2]     \n"
                        "fmla   v29.8h, v20.8h, v0.h[4]     \n"
                        "fmla   v30.8h, v20.8h, v0.h[6]     \n"
                        "fmla   v31.8h, v20.8h, v1.h[0]     \n"

                        "add    %5, %5, #16                 \n"

                        "fmla   v28.8h, v21.8h, v0.h[3]     \n"
                        "fmla   v29.8h, v21.8h, v0.h[5]     \n"
                        "fmla   v30.8h, v21.8h, v0.h[7]     \n"
                        "fmla   v31.8h, v21.8h, v1.h[1]     \n"

                        "prfm   pldl1keep, [%8, #128]       \n"
                        "ld1    {v16.8h}, [%8]              \n"

                        "fmla   v28.8h, v22.8h, v0.h[4]     \n"
                        "fmla   v29.8h, v22.8h, v0.h[6]     \n"
                        "fmla   v30.8h, v22.8h, v1.h[0]     \n"
                        "fmla   v31.8h, v22.8h, v1.h[2]     \n"

                        "add    %6, %6, #16                 \n"

                        "fmla   v28.8h, v23.8h, v0.h[5]     \n"
                        "fmla   v29.8h, v23.8h, v0.h[7]     \n"
                        "fmla   v30.8h, v23.8h, v1.h[1]     \n"
                        "fmla   v31.8h, v23.8h, v1.h[3]     \n"

                        "add    %7, %7, #16                 \n"

                        "fmla   v28.8h, v16.8h, v0.h[6]     \n"
                        "fmla   v29.8h, v16.8h, v1.h[0]     \n"
                        "fmla   v30.8h, v16.8h, v1.h[2]     \n"
                        "fmla   v31.8h, v16.8h, v1.h[4]     \n"

                        "sub    %8, %8, #768                \n" // kptr -= 48 * 8;

                        "st1    {v28.8h, v29.8h, v30.8h, v31.8h}, [%0], #64 \n"

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
                        : "memory", "v0", "v1", "v2", "v3", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v28", "v29", "v30", "v31");
                }
                for (; j < outw; j++)
                {
                    asm volatile(
                        "prfm   pldl1keep, [%1, #128]       \n"
                        "ld1    {v0.8h}, [%1]               \n" // r0

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%8], #64 \n"

                        "prfm   pldl1keep, [%0, #128]       \n"
                        "ld1    {v31.8h}, [%0]              \n" // sum0

                        "fmul   v28.8h, v16.8h, v0.h[0]     \n"
                        "fmul   v29.8h, v17.8h, v0.h[1]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%8], #64 \n"

                        "fmul   v30.8h, v18.8h, v0.h[2]     \n"
                        "fmla   v31.8h, v19.8h, v0.h[3]     \n"

                        "prfm   pldl1keep, [%2, #128]       \n"
                        "ld1    {v1.8h}, [%2]               \n" // r1

                        "fmla   v28.8h, v20.8h, v0.h[4]     \n"
                        "fmla   v29.8h, v21.8h, v0.h[5]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%8], #64 \n"

                        "fmla   v30.8h, v22.8h, v0.h[6]     \n"
                        "fmla   v31.8h, v23.8h, v1.h[0]     \n"

                        "fmla   v28.8h, v16.8h, v1.h[1]     \n"
                        "fmla   v29.8h, v17.8h, v1.h[2]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%8], #64 \n"

                        "fmla   v30.8h, v18.8h, v1.h[3]     \n"
                        "fmla   v31.8h, v19.8h, v1.h[4]     \n"

                        "prfm   pldl1keep, [%3, #128]       \n"
                        "ld1    {v0.8h}, [%3]               \n" // r2

                        "fmla   v28.8h, v20.8h, v1.h[5]     \n"
                        "fmla   v29.8h, v21.8h, v1.h[6]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%8], #64 \n"

                        "fmla   v30.8h, v22.8h, v0.h[0]     \n"
                        "fmla   v31.8h, v23.8h, v0.h[1]     \n"

                        "prfm   pldl1keep, [%4, #128]       \n"
                        "ld1    {v1.8h}, [%4]               \n" // r3

                        "fmla   v28.8h, v16.8h, v0.h[2]     \n"
                        "fmla   v29.8h, v17.8h, v0.h[3]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%8], #64 \n"

                        "fmla   v30.8h, v18.8h, v0.h[4]     \n"
                        "fmla   v31.8h, v19.8h, v0.h[5]     \n"

                        "add    %1, %1, #4                  \n"

                        "fmla   v28.8h, v20.8h, v0.h[6]     \n"
                        "fmla   v29.8h, v21.8h, v1.h[0]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%8], #64 \n"

                        "fmla   v30.8h, v22.8h, v1.h[1]     \n"
                        "fmla   v31.8h, v23.8h, v1.h[2]     \n"

                        "prfm   pldl1keep, [%5, #128]       \n"
                        "ld1    {v0.8h}, [%5]               \n" // r4

                        "fmla   v28.8h, v16.8h, v1.h[3]     \n"
                        "fmla   v29.8h, v17.8h, v1.h[4]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%8], #64 \n"

                        "fmla   v30.8h, v18.8h, v1.h[5]     \n"
                        "fmla   v31.8h, v19.8h, v1.h[6]     \n"

                        "add    %2, %2, #4                  \n"

                        "fmla   v28.8h, v20.8h, v0.h[0]     \n"
                        "fmla   v29.8h, v21.8h, v0.h[1]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%8], #64 \n"

                        "fmla   v30.8h, v22.8h, v0.h[2]     \n"
                        "fmla   v31.8h, v23.8h, v0.h[3]     \n"

                        "prfm   pldl1keep, [%6, #128]       \n"
                        "ld1    {v1.8h}, [%6]               \n" // r5

                        "fmla   v28.8h, v16.8h, v0.h[4]     \n"
                        "fmla   v29.8h, v17.8h, v0.h[5]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%8], #64 \n"

                        "fmla   v30.8h, v18.8h, v0.h[6]     \n"
                        "fmla   v31.8h, v19.8h, v1.h[0]     \n"

                        "add    %3, %3, #4                  \n"

                        "fmla   v28.8h, v20.8h, v1.h[1]     \n"
                        "fmla   v29.8h, v21.8h, v1.h[2]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%8], #64 \n"

                        "fmla   v30.8h, v22.8h, v1.h[3]     \n"
                        "fmla   v31.8h, v23.8h, v1.h[4]     \n"

                        "prfm   pldl1keep, [%7, #128]       \n"
                        "ld1    {v0.8h}, [%7]               \n" // r6

                        "fmla   v28.8h, v16.8h, v1.h[5]     \n"
                        "fmla   v29.8h, v17.8h, v1.h[6]     \n"

                        "prfm   pldl1keep, [%8, #512]       \n"
                        "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%8], #64 \n"

                        "fmla   v30.8h, v18.8h, v0.h[0]     \n"
                        "fmla   v31.8h, v19.8h, v0.h[1]     \n"

                        "add    %4, %4, #4                  \n"

                        "fmla   v28.8h, v20.8h, v0.h[2]     \n"
                        "fmla   v29.8h, v21.8h, v0.h[3]     \n"

                        "prfm   pldl1keep, [%8, #128]       \n"
                        "ld1    {v16.8h}, [%8]              \n"

                        "fmla   v30.8h, v22.8h, v0.h[4]     \n"
                        "fmla   v31.8h, v23.8h, v0.h[5]     \n"

                        "add    %5, %5, #4                  \n"

                        "fmla   v28.8h, v16.8h, v0.h[6]     \n"

                        "add    %6, %6, #4                  \n"

                        "fadd   v29.8h, v29.8h, v30.8h      \n"
                        "fadd   v31.8h, v31.8h, v28.8h      \n"

                        "add    %7, %7, #4                  \n"

                        "fadd   v29.8h, v29.8h, v31.8h      \n"

                        "sub    %8, %8, #768                \n" // kptr -= 48 * 8;

                        "st1    {v29.8h}, [%0], #16         \n"

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
                        : "memory", "v0", "v1", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v28", "v29", "v30", "v31");
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
