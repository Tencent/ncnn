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

static void conv7x7s2_pack1to4_bf16s_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    Mat top_blob_fp32(outw, outh, opt.num_threads, (size_t)4u * 4, 4, opt.workspace_allocator);

    const int tailstep = w - 2 * outw + w;

    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out0 = top_blob_fp32.channel(get_omp_thread_num());

        float32x4_t _bias0 = bias ? vld1q_f32((const float*)bias + p * 4) : vdupq_n_f32(0.f);
        out0.fill(_bias0);

        int q = 0;
        for (; q < inch - 1; q++)
        {
            float* outptr0 = out0.row(0);

            const Mat img0 = bottom_blob.channel(q);

            const unsigned short* r0 = img0.row<const unsigned short>(0);
            const unsigned short* r1 = img0.row<const unsigned short>(1);
            const unsigned short* r2 = img0.row<const unsigned short>(2);
            const unsigned short* r3 = img0.row<const unsigned short>(3);
            const unsigned short* r4 = img0.row<const unsigned short>(4);
            const unsigned short* r5 = img0.row<const unsigned short>(5);
            const unsigned short* r6 = img0.row<const unsigned short>(6);

            const unsigned short* kptr = kernel.channel(p).row<const unsigned short>(q);

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

                        "prfm   pldl1keep, [%1, #256]       \n"
                        "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%1], #32 \n" // r0

                        "shll   v0.4s, v0.4h, #16           \n"
                        "shll   v1.4s, v1.4h, #16           \n"
                        "shll   v2.4s, v2.4h, #16           \n"
                        "shll   v3.4s, v3.4h, #16           \n"

                        "prfm   pldl1keep, [%8, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%8], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

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

                        "prfm   pldl1keep, [%1, #128]       \n"
                        "ld1    {v4.4h, v5.4h}, [%1]        \n"

                        "shll   v4.4s, v4.4h, #16           \n"
                        "shll   v5.4s, v5.4h, #16           \n"

                        "fmla   v16.4s, v25.4s, v0.s[1]     \n"
                        "fmla   v17.4s, v25.4s, v0.s[3]     \n"
                        "fmla   v18.4s, v25.4s, v1.s[1]     \n"
                        "fmla   v19.4s, v25.4s, v1.s[3]     \n"
                        "fmla   v20.4s, v25.4s, v2.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v2.s[3]     \n"
                        "fmla   v22.4s, v25.4s, v3.s[1]     \n"
                        "fmla   v23.4s, v25.4s, v3.s[3]     \n"

                        "prfm   pldl1keep, [%8, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%8], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

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

                        "prfm   pldl1keep, [%8, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%8], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v16.4s, v29.4s, v1.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v1.s[3]     \n"
                        "fmla   v18.4s, v29.4s, v2.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v2.s[3]     \n"
                        "fmla   v20.4s, v29.4s, v3.s[1]     \n"
                        "fmla   v21.4s, v29.4s, v3.s[3]     \n"
                        "fmla   v22.4s, v29.4s, v4.s[1]     \n"
                        "fmla   v23.4s, v29.4s, v4.s[3]     \n"

                        "prfm   pldl1keep, [%2, #256]       \n"
                        "ld1    {v6.4h, v7.4h, v8.4h, v9.4h}, [%2], #32 \n" // r1

                        "shll   v6.4s, v6.4h, #16           \n"
                        "shll   v7.4s, v7.4h, #16           \n"
                        "shll   v8.4s, v8.4h, #16           \n"
                        "shll   v9.4s, v9.4h, #16           \n"

                        "fmla   v16.4s, v30.4s, v1.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v2.s[0]     \n"
                        "fmla   v18.4s, v30.4s, v2.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v3.s[0]     \n"
                        "fmla   v20.4s, v30.4s, v3.s[2]     \n"
                        "fmla   v21.4s, v30.4s, v4.s[0]     \n"
                        "fmla   v22.4s, v30.4s, v4.s[2]     \n"
                        "fmla   v23.4s, v30.4s, v5.s[0]     \n"

                        "prfm   pldl1keep, [%2, #128]       \n"
                        "ld1    {v10.4h, v11.4h}, [%2]      \n"

                        "shll   v10.4s, v10.4h, #16         \n"
                        "shll   v11.4s, v11.4h, #16         \n"

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

                        "prfm   pldl1keep, [%8, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%8], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

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

                        "prfm   pldl1keep, [%8, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%8], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v16.4s, v29.4s, v7.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v7.s[3]     \n"
                        "fmla   v18.4s, v29.4s, v8.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v8.s[3]     \n"
                        "fmla   v20.4s, v29.4s, v9.s[1]     \n"
                        "fmla   v21.4s, v29.4s, v9.s[3]     \n"
                        "fmla   v22.4s, v29.4s, v10.s[1]    \n"
                        "fmla   v23.4s, v29.4s, v10.s[3]    \n"

                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%3], #32 \n" // r2

                        "shll   v0.4s, v0.4h, #16           \n"
                        "shll   v1.4s, v1.4h, #16           \n"
                        "shll   v2.4s, v2.4h, #16           \n"
                        "shll   v3.4s, v3.4h, #16           \n"

                        "fmla   v16.4s, v30.4s, v7.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v8.s[0]     \n"
                        "fmla   v18.4s, v30.4s, v8.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v9.s[0]     \n"
                        "fmla   v20.4s, v30.4s, v9.s[2]     \n"
                        "fmla   v21.4s, v30.4s, v10.s[0]    \n"
                        "fmla   v22.4s, v30.4s, v10.s[2]    \n"
                        "fmla   v23.4s, v30.4s, v11.s[0]    \n"

                        "prfm   pldl1keep, [%3, #128]       \n"
                        "ld1    {v4.4h, v5.4h}, [%3]        \n"

                        "shll   v4.4s, v4.4h, #16           \n"
                        "shll   v5.4s, v5.4h, #16           \n"

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

                        "prfm   pldl1keep, [%8, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%8], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

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

                        "prfm   pldl1keep, [%8, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%8], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v16.4s, v29.4s, v1.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v1.s[3]     \n"
                        "fmla   v18.4s, v29.4s, v2.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v2.s[3]     \n"
                        "fmla   v20.4s, v29.4s, v3.s[1]     \n"
                        "fmla   v21.4s, v29.4s, v3.s[3]     \n"
                        "fmla   v22.4s, v29.4s, v4.s[1]     \n"
                        "fmla   v23.4s, v29.4s, v4.s[3]     \n"

                        "prfm   pldl1keep, [%4, #256]       \n"
                        "ld1    {v6.4h, v7.4h, v8.4h, v9.4h}, [%4], #32 \n" // r3

                        "shll   v6.4s, v6.4h, #16           \n"
                        "shll   v7.4s, v7.4h, #16           \n"
                        "shll   v8.4s, v8.4h, #16           \n"
                        "shll   v9.4s, v9.4h, #16           \n"

                        "fmla   v16.4s, v30.4s, v1.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v2.s[0]     \n"
                        "fmla   v18.4s, v30.4s, v2.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v3.s[0]     \n"
                        "fmla   v20.4s, v30.4s, v3.s[2]     \n"
                        "fmla   v21.4s, v30.4s, v4.s[0]     \n"
                        "fmla   v22.4s, v30.4s, v4.s[2]     \n"
                        "fmla   v23.4s, v30.4s, v5.s[0]     \n"

                        "prfm   pldl1keep, [%4, #128]       \n"
                        "ld1    {v10.4h, v11.4h}, [%4]      \n"

                        "shll   v10.4s, v10.4h, #16         \n"
                        "shll   v11.4s, v11.4h, #16         \n"

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

                        "prfm   pldl1keep, [%8, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%8], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

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

                        "prfm   pldl1keep, [%8, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%8], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v16.4s, v29.4s, v7.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v7.s[3]     \n"
                        "fmla   v18.4s, v29.4s, v8.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v8.s[3]     \n"
                        "fmla   v20.4s, v29.4s, v9.s[1]     \n"
                        "fmla   v21.4s, v29.4s, v9.s[3]     \n"
                        "fmla   v22.4s, v29.4s, v10.s[1]    \n"
                        "fmla   v23.4s, v29.4s, v10.s[3]    \n"

                        "prfm   pldl1keep, [%5, #256]       \n"
                        "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%5], #32 \n" // r4

                        "shll   v0.4s, v0.4h, #16           \n"
                        "shll   v1.4s, v1.4h, #16           \n"
                        "shll   v2.4s, v2.4h, #16           \n"
                        "shll   v3.4s, v3.4h, #16           \n"

                        "fmla   v16.4s, v30.4s, v7.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v8.s[0]     \n"
                        "fmla   v18.4s, v30.4s, v8.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v9.s[0]     \n"
                        "fmla   v20.4s, v30.4s, v9.s[2]     \n"
                        "fmla   v21.4s, v30.4s, v10.s[0]    \n"
                        "fmla   v22.4s, v30.4s, v10.s[2]    \n"
                        "fmla   v23.4s, v30.4s, v11.s[0]    \n"

                        "prfm   pldl1keep, [%5, #128]       \n"
                        "ld1    {v4.4h, v5.4h}, [%5]        \n"

                        "shll   v4.4s, v4.4h, #16           \n"
                        "shll   v5.4s, v5.4h, #16           \n"

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

                        "prfm   pldl1keep, [%8, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%8], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

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

                        "prfm   pldl1keep, [%8, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%8], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v16.4s, v29.4s, v1.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v1.s[3]     \n"
                        "fmla   v18.4s, v29.4s, v2.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v2.s[3]     \n"
                        "fmla   v20.4s, v29.4s, v3.s[1]     \n"
                        "fmla   v21.4s, v29.4s, v3.s[3]     \n"
                        "fmla   v22.4s, v29.4s, v4.s[1]     \n"
                        "fmla   v23.4s, v29.4s, v4.s[3]     \n"

                        "prfm   pldl1keep, [%6, #256]       \n"
                        "ld1    {v6.4h, v7.4h, v8.4h, v9.4h}, [%6], #32 \n" // r5

                        "shll   v6.4s, v6.4h, #16           \n"
                        "shll   v7.4s, v7.4h, #16           \n"
                        "shll   v8.4s, v8.4h, #16           \n"
                        "shll   v9.4s, v9.4h, #16           \n"

                        "fmla   v16.4s, v30.4s, v1.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v2.s[0]     \n"
                        "fmla   v18.4s, v30.4s, v2.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v3.s[0]     \n"
                        "fmla   v20.4s, v30.4s, v3.s[2]     \n"
                        "fmla   v21.4s, v30.4s, v4.s[0]     \n"
                        "fmla   v22.4s, v30.4s, v4.s[2]     \n"
                        "fmla   v23.4s, v30.4s, v5.s[0]     \n"

                        "prfm   pldl1keep, [%6, #128]       \n"
                        "ld1    {v10.4h, v11.4h}, [%6]      \n"

                        "shll   v10.4s, v10.4h, #16         \n"
                        "shll   v11.4s, v11.4h, #16         \n"

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

                        "prfm   pldl1keep, [%8, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%8], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

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

                        "prfm   pldl1keep, [%8, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%8], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v16.4s, v29.4s, v7.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v7.s[3]     \n"
                        "fmla   v18.4s, v29.4s, v8.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v8.s[3]     \n"
                        "fmla   v20.4s, v29.4s, v9.s[1]     \n"
                        "fmla   v21.4s, v29.4s, v9.s[3]     \n"
                        "fmla   v22.4s, v29.4s, v10.s[1]    \n"
                        "fmla   v23.4s, v29.4s, v10.s[3]    \n"

                        "prfm   pldl1keep, [%7, #256]       \n"
                        "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%7], #32 \n" // r6

                        "shll   v0.4s, v0.4h, #16           \n"
                        "shll   v1.4s, v1.4h, #16           \n"
                        "shll   v2.4s, v2.4h, #16           \n"
                        "shll   v3.4s, v3.4h, #16           \n"

                        "fmla   v16.4s, v30.4s, v7.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v8.s[0]     \n"
                        "fmla   v18.4s, v30.4s, v8.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v9.s[0]     \n"
                        "fmla   v20.4s, v30.4s, v9.s[2]     \n"
                        "fmla   v21.4s, v30.4s, v10.s[0]    \n"
                        "fmla   v22.4s, v30.4s, v10.s[2]    \n"
                        "fmla   v23.4s, v30.4s, v11.s[0]    \n"

                        "prfm   pldl1keep, [%7, #128]       \n"
                        "ld1    {v4.4h, v5.4h}, [%7]        \n"

                        "shll   v4.4s, v4.4h, #16           \n"
                        "shll   v5.4s, v5.4h, #16           \n"

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

                        "prfm   pldl1keep, [%8, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%8], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

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

                        "sub    %8, %8, #392                \n"

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

                        "prfm   pldl1keep, [%1, #256]       \n"
                        "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%1] \n" // r0

                        "shll   v0.4s, v0.4h, #16           \n"
                        "shll   v1.4s, v1.4h, #16           \n"
                        "shll   v2.4s, v2.4h, #16           \n"
                        "shll   v3.4s, v3.4h, #16           \n"

                        "prfm   pldl1keep, [%8, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%8], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v16.4s, v24.4s, v0.s[0]     \n"
                        "fmla   v17.4s, v24.4s, v0.s[2]     \n"
                        "fmla   v18.4s, v24.4s, v1.s[0]     \n"
                        "fmla   v19.4s, v24.4s, v1.s[2]     \n"

                        "prfm   pldl1keep, [%8, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%8], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

                        "fmla   v16.4s, v25.4s, v0.s[1]     \n"
                        "fmla   v17.4s, v25.4s, v0.s[3]     \n"
                        "fmla   v18.4s, v25.4s, v1.s[1]     \n"
                        "fmla   v19.4s, v25.4s, v1.s[3]     \n"
                        "fmla   v16.4s, v26.4s, v0.s[2]     \n"
                        "fmla   v17.4s, v26.4s, v1.s[0]     \n"
                        "fmla   v18.4s, v26.4s, v1.s[2]     \n"
                        "fmla   v19.4s, v26.4s, v2.s[0]     \n"

                        "prfm   pldl1keep, [%2, #256]       \n"
                        "ld1    {v4.4h, v5.4h, v6.4h, v7.4h}, [%2] \n" // r1

                        "shll   v4.4s, v4.4h, #16           \n"
                        "shll   v5.4s, v5.4h, #16           \n"
                        "shll   v6.4s, v6.4h, #16           \n"
                        "shll   v7.4s, v7.4h, #16           \n"

                        "fmla   v16.4s, v27.4s, v0.s[3]     \n"
                        "fmla   v17.4s, v27.4s, v1.s[1]     \n"
                        "fmla   v18.4s, v27.4s, v1.s[3]     \n"
                        "fmla   v19.4s, v27.4s, v2.s[1]     \n"
                        "fmla   v16.4s, v28.4s, v1.s[0]     \n"
                        "fmla   v17.4s, v28.4s, v1.s[2]     \n"
                        "fmla   v18.4s, v28.4s, v2.s[0]     \n"
                        "fmla   v19.4s, v28.4s, v2.s[2]     \n"

                        "prfm   pldl1keep, [%8, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%8], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v16.4s, v29.4s, v1.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v1.s[3]     \n"
                        "fmla   v18.4s, v29.4s, v2.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v2.s[3]     \n"
                        "fmla   v16.4s, v30.4s, v1.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v2.s[0]     \n"
                        "fmla   v18.4s, v30.4s, v2.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v3.s[0]     \n"

                        "prfm   pldl1keep, [%8, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%8], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

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

                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%3] \n" // r2

                        "shll   v0.4s, v0.4h, #16           \n"
                        "shll   v1.4s, v1.4h, #16           \n"
                        "shll   v2.4s, v2.4h, #16           \n"
                        "shll   v3.4s, v3.4h, #16           \n"

                        "fmla   v16.4s, v27.4s, v4.s[3]     \n"
                        "fmla   v17.4s, v27.4s, v5.s[1]     \n"
                        "fmla   v18.4s, v27.4s, v5.s[3]     \n"
                        "fmla   v19.4s, v27.4s, v6.s[1]     \n"
                        "fmla   v16.4s, v28.4s, v5.s[0]     \n"
                        "fmla   v17.4s, v28.4s, v5.s[2]     \n"
                        "fmla   v18.4s, v28.4s, v6.s[0]     \n"
                        "fmla   v19.4s, v28.4s, v6.s[2]     \n"

                        "prfm   pldl1keep, [%8, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%8], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v16.4s, v29.4s, v5.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v5.s[3]     \n"
                        "fmla   v18.4s, v29.4s, v6.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v6.s[3]     \n"
                        "fmla   v16.4s, v30.4s, v5.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v6.s[0]     \n"
                        "fmla   v18.4s, v30.4s, v6.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v7.s[0]     \n"

                        "prfm   pldl1keep, [%8, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%8], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

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

                        "prfm   pldl1keep, [%4, #256]       \n"
                        "ld1    {v4.4h, v5.4h, v6.4h, v7.4h}, [%4] \n" // r3

                        "shll   v4.4s, v4.4h, #16           \n"
                        "shll   v5.4s, v5.4h, #16           \n"
                        "shll   v6.4s, v6.4h, #16           \n"
                        "shll   v7.4s, v7.4h, #16           \n"

                        "fmla   v16.4s, v27.4s, v0.s[3]     \n"
                        "fmla   v17.4s, v27.4s, v1.s[1]     \n"
                        "fmla   v18.4s, v27.4s, v1.s[3]     \n"
                        "fmla   v19.4s, v27.4s, v2.s[1]     \n"
                        "fmla   v16.4s, v28.4s, v1.s[0]     \n"
                        "fmla   v17.4s, v28.4s, v1.s[2]     \n"
                        "fmla   v18.4s, v28.4s, v2.s[0]     \n"
                        "fmla   v19.4s, v28.4s, v2.s[2]     \n"

                        "prfm   pldl1keep, [%8, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%8], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v16.4s, v29.4s, v1.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v1.s[3]     \n"
                        "fmla   v18.4s, v29.4s, v2.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v2.s[3]     \n"
                        "fmla   v16.4s, v30.4s, v1.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v2.s[0]     \n"
                        "fmla   v18.4s, v30.4s, v2.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v3.s[0]     \n"

                        "prfm   pldl1keep, [%8, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%8], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

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

                        "prfm   pldl1keep, [%5, #256]       \n"
                        "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%5] \n" // r4

                        "shll   v0.4s, v0.4h, #16           \n"
                        "shll   v1.4s, v1.4h, #16           \n"
                        "shll   v2.4s, v2.4h, #16           \n"
                        "shll   v3.4s, v3.4h, #16           \n"

                        "fmla   v16.4s, v27.4s, v4.s[3]     \n"
                        "fmla   v17.4s, v27.4s, v5.s[1]     \n"
                        "fmla   v18.4s, v27.4s, v5.s[3]     \n"
                        "fmla   v19.4s, v27.4s, v6.s[1]     \n"
                        "fmla   v16.4s, v28.4s, v5.s[0]     \n"
                        "fmla   v17.4s, v28.4s, v5.s[2]     \n"
                        "fmla   v18.4s, v28.4s, v6.s[0]     \n"
                        "fmla   v19.4s, v28.4s, v6.s[2]     \n"

                        "prfm   pldl1keep, [%8, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%8], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v16.4s, v29.4s, v5.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v5.s[3]     \n"
                        "fmla   v18.4s, v29.4s, v6.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v6.s[3]     \n"
                        "fmla   v16.4s, v30.4s, v5.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v6.s[0]     \n"
                        "fmla   v18.4s, v30.4s, v6.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v7.s[0]     \n"

                        "prfm   pldl1keep, [%8, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%8], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

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

                        "prfm   pldl1keep, [%6, #256]       \n"
                        "ld1    {v4.4h, v5.4h, v6.4h, v7.4h}, [%6] \n" // r5

                        "shll   v4.4s, v4.4h, #16           \n"
                        "shll   v5.4s, v5.4h, #16           \n"
                        "shll   v6.4s, v6.4h, #16           \n"
                        "shll   v7.4s, v7.4h, #16           \n"

                        "fmla   v16.4s, v27.4s, v0.s[3]     \n"
                        "fmla   v17.4s, v27.4s, v1.s[1]     \n"
                        "fmla   v18.4s, v27.4s, v1.s[3]     \n"
                        "fmla   v19.4s, v27.4s, v2.s[1]     \n"
                        "fmla   v16.4s, v28.4s, v1.s[0]     \n"
                        "fmla   v17.4s, v28.4s, v1.s[2]     \n"
                        "fmla   v18.4s, v28.4s, v2.s[0]     \n"
                        "fmla   v19.4s, v28.4s, v2.s[2]     \n"

                        "prfm   pldl1keep, [%8, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%8], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v16.4s, v29.4s, v1.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v1.s[3]     \n"
                        "fmla   v18.4s, v29.4s, v2.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v2.s[3]     \n"
                        "fmla   v16.4s, v30.4s, v1.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v2.s[0]     \n"
                        "fmla   v18.4s, v30.4s, v2.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v3.s[0]     \n"

                        "prfm   pldl1keep, [%8, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%8], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

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

                        "prfm   pldl1keep, [%7, #256]       \n"
                        "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%7] \n" // r6

                        "shll   v0.4s, v0.4h, #16           \n"
                        "shll   v1.4s, v1.4h, #16           \n"
                        "shll   v2.4s, v2.4h, #16           \n"
                        "shll   v3.4s, v3.4h, #16           \n"

                        "fmla   v16.4s, v27.4s, v4.s[3]     \n"
                        "fmla   v17.4s, v27.4s, v5.s[1]     \n"
                        "fmla   v18.4s, v27.4s, v5.s[3]     \n"
                        "fmla   v19.4s, v27.4s, v6.s[1]     \n"
                        "fmla   v16.4s, v28.4s, v5.s[0]     \n"
                        "fmla   v17.4s, v28.4s, v5.s[2]     \n"
                        "fmla   v18.4s, v28.4s, v6.s[0]     \n"
                        "fmla   v19.4s, v28.4s, v6.s[2]     \n"

                        "prfm   pldl1keep, [%8, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%8], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v16.4s, v29.4s, v5.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v5.s[3]     \n"
                        "fmla   v18.4s, v29.4s, v6.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v6.s[3]     \n"
                        "fmla   v16.4s, v30.4s, v5.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v6.s[0]     \n"
                        "fmla   v18.4s, v30.4s, v6.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v7.s[0]     \n"

                        "prfm   pldl1keep, [%8, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%8], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

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

                        "add    %1, %1, #16                 \n"
                        "add    %2, %2, #16                 \n"
                        "add    %3, %3, #16                 \n"
                        "add    %4, %4, #16                 \n"
                        "add    %5, %5, #16                 \n"
                        "add    %6, %6, #16                 \n"
                        "add    %7, %7, #16                 \n"

                        "sub    %8, %8, #392                \n"

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
                        "pld        [%0, #512]          \n"
                        "vldm       %0, {d24-d31}       \n"

                        "pld        [%1, #128]          \n"
                        "vld1.u16   {d2-d3}, [%1]!      \n" // r0

                        "vshll.u16  q0, d2, #16         \n"
                        "vshll.u16  q1, d3, #16         \n"

                        "pld        [%8, #256]          \n"
                        "vld1.u16   {d14-d17}, [%8]!    \n"

                        "vshll.u16  q5, d14, #16        \n"
                        "vshll.u16  q6, d15, #16        \n"
                        "vshll.u16  q7, d16, #16        \n"
                        "vshll.u16  q8, d17, #16        \n"

                        "vmla.f32   q12, q5, d0[0]      \n"
                        "vmla.f32   q13, q5, d1[0]      \n"
                        "vmla.f32   q14, q5, d2[0]      \n"
                        "vmla.f32   q15, q5, d3[0]      \n"

                        "pld        [%1, #128]          \n"
                        "vld1.u16   {d5-d6}, [%1]       \n"

                        "vshll.u16  q2, d5, #16         \n"
                        "vshl.u32   d6, d6, #16         \n"

                        "vmla.f32   q12, q6, d0[1]      \n"
                        "vmla.f32   q13, q6, d1[1]      \n"
                        "vmla.f32   q14, q6, d2[1]      \n"
                        "vmla.f32   q15, q6, d3[1]      \n"

                        "pld        [%8, #192]          \n"
                        "vld1.u16   {d20-d22}, [%8]!    \n"

                        "vshll.u16  q9, d20, #16        \n"
                        "vshll.u16  q10, d21, #16       \n"
                        "vshll.u16  q11, d22, #16       \n"

                        "vmla.f32   q12, q7, d1[0]      \n"
                        "vmla.f32   q13, q7, d2[0]      \n"
                        "vmla.f32   q14, q7, d3[0]      \n"
                        "vmla.f32   q15, q7, d4[0]      \n"
                        "vmla.f32   q12, q8, d1[1]      \n"
                        "vmla.f32   q13, q8, d2[1]      \n"
                        "vmla.f32   q14, q8, d3[1]      \n"
                        "vmla.f32   q15, q8, d4[1]      \n"
                        "vmla.f32   q12, q9, d2[0]      \n"
                        "vmla.f32   q13, q9, d3[0]      \n"
                        "vmla.f32   q14, q9, d4[0]      \n"
                        "vmla.f32   q15, q9, d5[0]      \n"

                        "pld        [%8, #256]          \n"
                        "vld1.u16   {d14-d17}, [%8]!    \n"

                        "vshll.u16  q5, d14, #16        \n"
                        "vshll.u16  q6, d15, #16        \n"
                        "vshll.u16  q7, d16, #16        \n"
                        "vshll.u16  q8, d17, #16        \n"

                        "vmla.f32   q12, q10, d2[1]     \n"
                        "vmla.f32   q13, q10, d3[1]     \n"
                        "vmla.f32   q14, q10, d4[1]     \n"
                        "vmla.f32   q15, q10, d5[1]     \n"
                        "vmla.f32   q12, q11, d3[0]     \n"
                        "vmla.f32   q13, q11, d4[0]     \n"

                        "pld        [%2, #128]          \n"
                        "vld1.u16   {d2-d3}, [%2]!      \n" // r1

                        "vshll.u16  q0, d2, #16         \n"
                        "vshll.u16  q1, d3, #16         \n"

                        "vmla.f32   q14, q11, d5[0]     \n"
                        "vmla.f32   q15, q11, d6[0]     \n"

                        "vmla.f32   q12, q5, d0[0]      \n"
                        "vmla.f32   q13, q5, d1[0]      \n"
                        "vmla.f32   q14, q5, d2[0]      \n"
                        "vmla.f32   q15, q5, d3[0]      \n"

                        "pld        [%2, #128]          \n"
                        "vld1.u16   {d5-d6}, [%2]       \n"

                        "vshll.u16  q2, d5, #16         \n"
                        "vshl.u32   d6, d6, #16         \n"

                        "vmla.f32   q12, q6, d0[1]      \n"
                        "vmla.f32   q13, q6, d1[1]      \n"
                        "vmla.f32   q14, q6, d2[1]      \n"
                        "vmla.f32   q15, q6, d3[1]      \n"

                        "pld        [%8, #192]          \n"
                        "vld1.u16   {d20-d22}, [%8]!    \n"

                        "vshll.u16  q9, d20, #16        \n"
                        "vshll.u16  q10, d21, #16       \n"
                        "vshll.u16  q11, d22, #16       \n"

                        "vmla.f32   q12, q7, d1[0]      \n"
                        "vmla.f32   q13, q7, d2[0]      \n"
                        "vmla.f32   q14, q7, d3[0]      \n"
                        "vmla.f32   q15, q7, d4[0]      \n"
                        "vmla.f32   q12, q8, d1[1]      \n"
                        "vmla.f32   q13, q8, d2[1]      \n"
                        "vmla.f32   q14, q8, d3[1]      \n"
                        "vmla.f32   q15, q8, d4[1]      \n"
                        "vmla.f32   q12, q9, d2[0]      \n"
                        "vmla.f32   q13, q9, d3[0]      \n"
                        "vmla.f32   q14, q9, d4[0]      \n"
                        "vmla.f32   q15, q9, d5[0]      \n"

                        "pld        [%8, #256]          \n"
                        "vld1.u16   {d14-d17}, [%8]!    \n"

                        "vshll.u16  q5, d14, #16        \n"
                        "vshll.u16  q6, d15, #16        \n"
                        "vshll.u16  q7, d16, #16        \n"
                        "vshll.u16  q8, d17, #16        \n"

                        "vmla.f32   q12, q10, d2[1]     \n"
                        "vmla.f32   q13, q10, d3[1]     \n"
                        "vmla.f32   q14, q10, d4[1]     \n"
                        "vmla.f32   q15, q10, d5[1]     \n"
                        "vmla.f32   q12, q11, d3[0]     \n"
                        "vmla.f32   q13, q11, d4[0]     \n"

                        "pld        [%3, #128]          \n"
                        "vld1.u16   {d2-d3}, [%3]!      \n" // r2

                        "vshll.u16  q0, d2, #16         \n"
                        "vshll.u16  q1, d3, #16         \n"

                        "vmla.f32   q14, q11, d5[0]     \n"
                        "vmla.f32   q15, q11, d6[0]     \n"

                        "vmla.f32   q12, q5, d0[0]      \n"
                        "vmla.f32   q13, q5, d1[0]      \n"
                        "vmla.f32   q14, q5, d2[0]      \n"
                        "vmla.f32   q15, q5, d3[0]      \n"

                        "pld        [%3, #128]          \n"
                        "vld1.u16   {d5-d6}, [%3]       \n"

                        "vshll.u16  q2, d5, #16         \n"
                        "vshl.u32   d6, d6, #16         \n"

                        "vmla.f32   q12, q6, d0[1]      \n"
                        "vmla.f32   q13, q6, d1[1]      \n"
                        "vmla.f32   q14, q6, d2[1]      \n"
                        "vmla.f32   q15, q6, d3[1]      \n"

                        "pld        [%8, #192]          \n"
                        "vld1.u16   {d20-d22}, [%8]!    \n"

                        "vshll.u16  q9, d20, #16        \n"
                        "vshll.u16  q10, d21, #16       \n"
                        "vshll.u16  q11, d22, #16       \n"

                        "vmla.f32   q12, q7, d1[0]      \n"
                        "vmla.f32   q13, q7, d2[0]      \n"
                        "vmla.f32   q14, q7, d3[0]      \n"
                        "vmla.f32   q15, q7, d4[0]      \n"
                        "vmla.f32   q12, q8, d1[1]      \n"
                        "vmla.f32   q13, q8, d2[1]      \n"
                        "vmla.f32   q14, q8, d3[1]      \n"
                        "vmla.f32   q15, q8, d4[1]      \n"
                        "vmla.f32   q12, q9, d2[0]      \n"
                        "vmla.f32   q13, q9, d3[0]      \n"
                        "vmla.f32   q14, q9, d4[0]      \n"
                        "vmla.f32   q15, q9, d5[0]      \n"

                        "pld        [%8, #256]          \n"
                        "vld1.u16   {d14-d17}, [%8]!    \n"

                        "vshll.u16  q5, d14, #16        \n"
                        "vshll.u16  q6, d15, #16        \n"
                        "vshll.u16  q7, d16, #16        \n"
                        "vshll.u16  q8, d17, #16        \n"

                        "vmla.f32   q12, q10, d2[1]     \n"
                        "vmla.f32   q13, q10, d3[1]     \n"
                        "vmla.f32   q14, q10, d4[1]     \n"
                        "vmla.f32   q15, q10, d5[1]     \n"
                        "vmla.f32   q12, q11, d3[0]     \n"
                        "vmla.f32   q13, q11, d4[0]     \n"

                        "pld        [%4, #128]          \n"
                        "vld1.u16   {d2-d3}, [%4]!      \n" // r3

                        "vshll.u16  q0, d2, #16         \n"
                        "vshll.u16  q1, d3, #16         \n"

                        "vmla.f32   q14, q11, d5[0]     \n"
                        "vmla.f32   q15, q11, d6[0]     \n"

                        "vmla.f32   q12, q5, d0[0]      \n"
                        "vmla.f32   q13, q5, d1[0]      \n"
                        "vmla.f32   q14, q5, d2[0]      \n"
                        "vmla.f32   q15, q5, d3[0]      \n"

                        "pld        [%4, #128]          \n"
                        "vld1.u16   {d5-d6}, [%4]       \n"

                        "vshll.u16  q2, d5, #16         \n"
                        "vshl.u32   d6, d6, #16         \n"

                        "vmla.f32   q12, q6, d0[1]      \n"
                        "vmla.f32   q13, q6, d1[1]      \n"
                        "vmla.f32   q14, q6, d2[1]      \n"
                        "vmla.f32   q15, q6, d3[1]      \n"

                        "pld        [%8, #192]          \n"
                        "vld1.u16   {d20-d22}, [%8]!    \n"

                        "vshll.u16  q9, d20, #16        \n"
                        "vshll.u16  q10, d21, #16       \n"
                        "vshll.u16  q11, d22, #16       \n"

                        "vmla.f32   q12, q7, d1[0]      \n"
                        "vmla.f32   q13, q7, d2[0]      \n"
                        "vmla.f32   q14, q7, d3[0]      \n"
                        "vmla.f32   q15, q7, d4[0]      \n"
                        "vmla.f32   q12, q8, d1[1]      \n"
                        "vmla.f32   q13, q8, d2[1]      \n"
                        "vmla.f32   q14, q8, d3[1]      \n"
                        "vmla.f32   q15, q8, d4[1]      \n"
                        "vmla.f32   q12, q9, d2[0]      \n"
                        "vmla.f32   q13, q9, d3[0]      \n"
                        "vmla.f32   q14, q9, d4[0]      \n"
                        "vmla.f32   q15, q9, d5[0]      \n"

                        "pld        [%8, #256]          \n"
                        "vld1.u16   {d14-d17}, [%8]!    \n"

                        "vshll.u16  q5, d14, #16        \n"
                        "vshll.u16  q6, d15, #16        \n"
                        "vshll.u16  q7, d16, #16        \n"
                        "vshll.u16  q8, d17, #16        \n"

                        "vmla.f32   q12, q10, d2[1]     \n"
                        "vmla.f32   q13, q10, d3[1]     \n"
                        "vmla.f32   q14, q10, d4[1]     \n"
                        "vmla.f32   q15, q10, d5[1]     \n"
                        "vmla.f32   q12, q11, d3[0]     \n"
                        "vmla.f32   q13, q11, d4[0]     \n"

                        "pld        [%5, #128]          \n"
                        "vld1.u16   {d2-d3}, [%5]!      \n" // r4

                        "vshll.u16  q0, d2, #16         \n"
                        "vshll.u16  q1, d3, #16         \n"

                        "vmla.f32   q14, q11, d5[0]     \n"
                        "vmla.f32   q15, q11, d6[0]     \n"

                        "vmla.f32   q12, q5, d0[0]      \n"
                        "vmla.f32   q13, q5, d1[0]      \n"
                        "vmla.f32   q14, q5, d2[0]      \n"
                        "vmla.f32   q15, q5, d3[0]      \n"

                        "pld        [%5, #128]          \n"
                        "vld1.u16   {d5-d6}, [%5]       \n"

                        "vshll.u16  q2, d5, #16         \n"
                        "vshl.u32   d6, d6, #16         \n"

                        "vmla.f32   q12, q6, d0[1]      \n"
                        "vmla.f32   q13, q6, d1[1]      \n"
                        "vmla.f32   q14, q6, d2[1]      \n"
                        "vmla.f32   q15, q6, d3[1]      \n"

                        "pld        [%8, #192]          \n"
                        "vld1.u16   {d20-d22}, [%8]!    \n"

                        "vshll.u16  q9, d20, #16        \n"
                        "vshll.u16  q10, d21, #16       \n"
                        "vshll.u16  q11, d22, #16       \n"

                        "vmla.f32   q12, q7, d1[0]      \n"
                        "vmla.f32   q13, q7, d2[0]      \n"
                        "vmla.f32   q14, q7, d3[0]      \n"
                        "vmla.f32   q15, q7, d4[0]      \n"
                        "vmla.f32   q12, q8, d1[1]      \n"
                        "vmla.f32   q13, q8, d2[1]      \n"
                        "vmla.f32   q14, q8, d3[1]      \n"
                        "vmla.f32   q15, q8, d4[1]      \n"
                        "vmla.f32   q12, q9, d2[0]      \n"
                        "vmla.f32   q13, q9, d3[0]      \n"
                        "vmla.f32   q14, q9, d4[0]      \n"
                        "vmla.f32   q15, q9, d5[0]      \n"

                        "pld        [%8, #256]          \n"
                        "vld1.u16   {d14-d17}, [%8]!    \n"

                        "vshll.u16  q5, d14, #16        \n"
                        "vshll.u16  q6, d15, #16        \n"
                        "vshll.u16  q7, d16, #16        \n"
                        "vshll.u16  q8, d17, #16        \n"

                        "vmla.f32   q12, q10, d2[1]     \n"
                        "vmla.f32   q13, q10, d3[1]     \n"
                        "vmla.f32   q14, q10, d4[1]     \n"
                        "vmla.f32   q15, q10, d5[1]     \n"
                        "vmla.f32   q12, q11, d3[0]     \n"
                        "vmla.f32   q13, q11, d4[0]     \n"

                        "pld        [%6, #128]          \n"
                        "vld1.u16   {d2-d3}, [%6]!      \n" // r5

                        "vshll.u16  q0, d2, #16         \n"
                        "vshll.u16  q1, d3, #16         \n"

                        "vmla.f32   q14, q11, d5[0]     \n"
                        "vmla.f32   q15, q11, d6[0]     \n"

                        "vmla.f32   q12, q5, d0[0]      \n"
                        "vmla.f32   q13, q5, d1[0]      \n"
                        "vmla.f32   q14, q5, d2[0]      \n"
                        "vmla.f32   q15, q5, d3[0]      \n"

                        "pld        [%6, #128]          \n"
                        "vld1.u16   {d5-d6}, [%6]       \n"

                        "vshll.u16  q2, d5, #16         \n"
                        "vshl.u32   d6, d6, #16         \n"

                        "vmla.f32   q12, q6, d0[1]      \n"
                        "vmla.f32   q13, q6, d1[1]      \n"
                        "vmla.f32   q14, q6, d2[1]      \n"
                        "vmla.f32   q15, q6, d3[1]      \n"

                        "pld        [%8, #192]          \n"
                        "vld1.u16   {d20-d22}, [%8]!    \n"

                        "vshll.u16  q9, d20, #16        \n"
                        "vshll.u16  q10, d21, #16       \n"
                        "vshll.u16  q11, d22, #16       \n"

                        "vmla.f32   q12, q7, d1[0]      \n"
                        "vmla.f32   q13, q7, d2[0]      \n"
                        "vmla.f32   q14, q7, d3[0]      \n"
                        "vmla.f32   q15, q7, d4[0]      \n"
                        "vmla.f32   q12, q8, d1[1]      \n"
                        "vmla.f32   q13, q8, d2[1]      \n"
                        "vmla.f32   q14, q8, d3[1]      \n"
                        "vmla.f32   q15, q8, d4[1]      \n"
                        "vmla.f32   q12, q9, d2[0]      \n"
                        "vmla.f32   q13, q9, d3[0]      \n"
                        "vmla.f32   q14, q9, d4[0]      \n"
                        "vmla.f32   q15, q9, d5[0]      \n"

                        "pld        [%8, #256]          \n"
                        "vld1.u16   {d14-d17}, [%8]!    \n"

                        "vshll.u16  q5, d14, #16        \n"
                        "vshll.u16  q6, d15, #16        \n"
                        "vshll.u16  q7, d16, #16        \n"
                        "vshll.u16  q8, d17, #16        \n"

                        "vmla.f32   q12, q10, d2[1]     \n"
                        "vmla.f32   q13, q10, d3[1]     \n"
                        "vmla.f32   q14, q10, d4[1]     \n"
                        "vmla.f32   q15, q10, d5[1]     \n"
                        "vmla.f32   q12, q11, d3[0]     \n"
                        "vmla.f32   q13, q11, d4[0]     \n"

                        "pld        [%7, #128]          \n"
                        "vld1.u16   {d2-d3}, [%7]!      \n" // r6

                        "vshll.u16  q0, d2, #16         \n"
                        "vshll.u16  q1, d3, #16         \n"

                        "vmla.f32   q14, q11, d5[0]     \n"
                        "vmla.f32   q15, q11, d6[0]     \n"

                        "vmla.f32   q12, q5, d0[0]      \n"
                        "vmla.f32   q13, q5, d1[0]      \n"
                        "vmla.f32   q14, q5, d2[0]      \n"
                        "vmla.f32   q15, q5, d3[0]      \n"

                        "pld        [%7, #128]          \n"
                        "vld1.u16   {d5-d6}, [%7]       \n"

                        "vshll.u16  q2, d5, #16         \n"
                        "vshl.u32   d6, d6, #16         \n"

                        "vmla.f32   q12, q6, d0[1]      \n"
                        "vmla.f32   q13, q6, d1[1]      \n"
                        "vmla.f32   q14, q6, d2[1]      \n"
                        "vmla.f32   q15, q6, d3[1]      \n"

                        "pld        [%8, #192]          \n"
                        "vld1.u16   {d20-d22}, [%8]!    \n"

                        "vshll.u16  q9, d20, #16        \n"
                        "vshll.u16  q10, d21, #16       \n"
                        "vshll.u16  q11, d22, #16       \n"

                        "vmla.f32   q12, q7, d1[0]      \n"
                        "vmla.f32   q13, q7, d2[0]      \n"
                        "vmla.f32   q14, q7, d3[0]      \n"
                        "vmla.f32   q15, q7, d4[0]      \n"
                        "vmla.f32   q12, q8, d1[1]      \n"
                        "vmla.f32   q13, q8, d2[1]      \n"
                        "vmla.f32   q14, q8, d3[1]      \n"
                        "vmla.f32   q15, q8, d4[1]      \n"
                        "vmla.f32   q12, q9, d2[0]      \n"
                        "vmla.f32   q13, q9, d3[0]      \n"
                        "vmla.f32   q14, q9, d4[0]      \n"
                        "vmla.f32   q15, q9, d5[0]      \n"
                        "vmla.f32   q12, q10, d2[1]     \n"
                        "vmla.f32   q13, q10, d3[1]     \n"
                        "vmla.f32   q14, q10, d4[1]     \n"
                        "vmla.f32   q15, q10, d5[1]     \n"
                        "vmla.f32   q12, q11, d3[0]     \n"
                        "vmla.f32   q13, q11, d4[0]     \n"
                        "vmla.f32   q14, q11, d5[0]     \n"
                        "vmla.f32   q15, q11, d6[0]     \n"

                        "sub        %8, %8, #392        \n"

                        "vstm       %0!, {d24-d31}      \n"

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

                        "prfm   pldl1keep, [%1, #192]       \n"
                        "ld1    {v0.4h, v1.4h, v2.4h}, [%1] \n" // r0

                        "shll   v0.4s, v0.4h, #16           \n"
                        "shll   v1.4s, v1.4h, #16           \n"
                        "shll   v2.4s, v2.4h, #16           \n"

                        "prfm   pldl1keep, [%8, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%8], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmul   v18.4s, v24.4s, v0.s[0]     \n"
                        "fmul   v19.4s, v24.4s, v0.s[2]     \n"

                        "prfm   pldl1keep, [%8, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%8], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

                        "fmla   v16.4s, v25.4s, v0.s[1]     \n"
                        "fmla   v17.4s, v25.4s, v0.s[3]     \n"
                        "fmla   v18.4s, v26.4s, v0.s[2]     \n"
                        "fmla   v19.4s, v26.4s, v1.s[0]     \n"

                        "prfm   pldl1keep, [%2, #192]       \n"
                        "ld1    {v4.4h, v5.4h, v6.4h}, [%2] \n" // r1

                        "shll   v4.4s, v4.4h, #16           \n"
                        "shll   v5.4s, v5.4h, #16           \n"
                        "shll   v6.4s, v6.4h, #16           \n"

                        "fmla   v16.4s, v27.4s, v0.s[3]     \n"
                        "fmla   v17.4s, v27.4s, v1.s[1]     \n"
                        "fmla   v18.4s, v28.4s, v1.s[0]     \n"
                        "fmla   v19.4s, v28.4s, v1.s[2]     \n"

                        "prfm   pldl1keep, [%8, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%8], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v16.4s, v29.4s, v1.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v1.s[3]     \n"
                        "fmla   v18.4s, v30.4s, v1.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v2.s[0]     \n"

                        "prfm   pldl1keep, [%8, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%8], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

                        "fmla   v16.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v17.4s, v24.4s, v4.s[2]     \n"
                        "fmla   v18.4s, v25.4s, v4.s[1]     \n"
                        "fmla   v19.4s, v25.4s, v4.s[3]     \n"
                        "fmla   v16.4s, v26.4s, v4.s[2]     \n"
                        "fmla   v17.4s, v26.4s, v5.s[0]     \n"

                        "prfm   pldl1keep, [%3, #192]       \n"
                        "ld1    {v0.4h, v1.4h, v2.4h}, [%3] \n" // r2

                        "shll   v0.4s, v0.4h, #16           \n"
                        "shll   v1.4s, v1.4h, #16           \n"
                        "shll   v2.4s, v2.4h, #16           \n"

                        "fmla   v18.4s, v27.4s, v4.s[3]     \n"
                        "fmla   v19.4s, v27.4s, v5.s[1]     \n"
                        "fmla   v16.4s, v28.4s, v5.s[0]     \n"
                        "fmla   v17.4s, v28.4s, v5.s[2]     \n"

                        "prfm   pldl1keep, [%8, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%8], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v18.4s, v29.4s, v5.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v5.s[3]     \n"
                        "fmla   v16.4s, v30.4s, v5.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v6.s[0]     \n"

                        "prfm   pldl1keep, [%8, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%8], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

                        "fmla   v18.4s, v24.4s, v0.s[0]     \n"
                        "fmla   v19.4s, v24.4s, v0.s[2]     \n"
                        "fmla   v16.4s, v25.4s, v0.s[1]     \n"
                        "fmla   v17.4s, v25.4s, v0.s[3]     \n"
                        "fmla   v18.4s, v26.4s, v0.s[2]     \n"
                        "fmla   v19.4s, v26.4s, v1.s[0]     \n"

                        "prfm   pldl1keep, [%4, #192]       \n"
                        "ld1    {v4.4h, v5.4h, v6.4h}, [%4] \n" // r3

                        "shll   v4.4s, v4.4h, #16           \n"
                        "shll   v5.4s, v5.4h, #16           \n"
                        "shll   v6.4s, v6.4h, #16           \n"

                        "fmla   v16.4s, v27.4s, v0.s[3]     \n"
                        "fmla   v17.4s, v27.4s, v1.s[1]     \n"
                        "fmla   v18.4s, v28.4s, v1.s[0]     \n"
                        "fmla   v19.4s, v28.4s, v1.s[2]     \n"

                        "prfm   pldl1keep, [%8, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%8], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v16.4s, v29.4s, v1.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v1.s[3]     \n"
                        "fmla   v18.4s, v30.4s, v1.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v2.s[0]     \n"

                        "prfm   pldl1keep, [%8, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%8], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

                        "fmla   v16.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v17.4s, v24.4s, v4.s[2]     \n"
                        "fmla   v18.4s, v25.4s, v4.s[1]     \n"
                        "fmla   v19.4s, v25.4s, v4.s[3]     \n"
                        "fmla   v16.4s, v26.4s, v4.s[2]     \n"
                        "fmla   v17.4s, v26.4s, v5.s[0]     \n"

                        "prfm   pldl1keep, [%5, #192]       \n"
                        "ld1    {v0.4h, v1.4h, v2.4h}, [%5] \n" // r4

                        "shll   v0.4s, v0.4h, #16           \n"
                        "shll   v1.4s, v1.4h, #16           \n"
                        "shll   v2.4s, v2.4h, #16           \n"

                        "fmla   v18.4s, v27.4s, v4.s[3]     \n"
                        "fmla   v19.4s, v27.4s, v5.s[1]     \n"
                        "fmla   v16.4s, v28.4s, v5.s[0]     \n"
                        "fmla   v17.4s, v28.4s, v5.s[2]     \n"

                        "prfm   pldl1keep, [%8, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%8], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v18.4s, v29.4s, v5.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v5.s[3]     \n"
                        "fmla   v16.4s, v30.4s, v5.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v6.s[0]     \n"

                        "prfm   pldl1keep, [%8, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%8], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

                        "fmla   v18.4s, v24.4s, v0.s[0]     \n"
                        "fmla   v19.4s, v24.4s, v0.s[2]     \n"
                        "fmla   v16.4s, v25.4s, v0.s[1]     \n"
                        "fmla   v17.4s, v25.4s, v0.s[3]     \n"
                        "fmla   v18.4s, v26.4s, v0.s[2]     \n"
                        "fmla   v19.4s, v26.4s, v1.s[0]     \n"

                        "prfm   pldl1keep, [%6, #192]       \n"
                        "ld1    {v4.4h, v5.4h, v6.4h}, [%6] \n" // r5

                        "shll   v4.4s, v4.4h, #16           \n"
                        "shll   v5.4s, v5.4h, #16           \n"
                        "shll   v6.4s, v6.4h, #16           \n"

                        "fmla   v16.4s, v27.4s, v0.s[3]     \n"
                        "fmla   v17.4s, v27.4s, v1.s[1]     \n"
                        "fmla   v18.4s, v28.4s, v1.s[0]     \n"
                        "fmla   v19.4s, v28.4s, v1.s[2]     \n"

                        "prfm   pldl1keep, [%8, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%8], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v16.4s, v29.4s, v1.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v1.s[3]     \n"
                        "fmla   v18.4s, v30.4s, v1.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v2.s[0]     \n"

                        "prfm   pldl1keep, [%8, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%8], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

                        "fmla   v16.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v17.4s, v24.4s, v4.s[2]     \n"
                        "fmla   v18.4s, v25.4s, v4.s[1]     \n"
                        "fmla   v19.4s, v25.4s, v4.s[3]     \n"
                        "fmla   v16.4s, v26.4s, v4.s[2]     \n"
                        "fmla   v17.4s, v26.4s, v5.s[0]     \n"

                        "prfm   pldl1keep, [%7, #192]       \n"
                        "ld1    {v0.4h, v1.4h, v2.4h}, [%7] \n" // r6

                        "shll   v0.4s, v0.4h, #16           \n"
                        "shll   v1.4s, v1.4h, #16           \n"
                        "shll   v2.4s, v2.4h, #16           \n"

                        "fmla   v18.4s, v27.4s, v4.s[3]     \n"
                        "fmla   v19.4s, v27.4s, v5.s[1]     \n"
                        "fmla   v16.4s, v28.4s, v5.s[0]     \n"
                        "fmla   v17.4s, v28.4s, v5.s[2]     \n"

                        "prfm   pldl1keep, [%8, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%8], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v18.4s, v29.4s, v5.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v5.s[3]     \n"
                        "fmla   v16.4s, v30.4s, v5.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v6.s[0]     \n"

                        "prfm   pldl1keep, [%8, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%8], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

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

                        "add    %1, %1, #8                  \n"
                        "add    %2, %2, #8                  \n"
                        "add    %3, %3, #8                  \n"
                        "add    %4, %4, #8                  \n"
                        "add    %5, %5, #8                  \n"
                        "add    %6, %6, #8                  \n"
                        "add    %7, %7, #8                  \n"

                        "fadd   v16.4s, v16.4s, v18.4s      \n"
                        "fadd   v17.4s, v17.4s, v19.4s      \n"

                        "sub    %8, %8, #392                \n"

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
                        "pld        [%0, #256]          \n"
                        "vld1.f32   {d28-d31}, [%0 :128] \n"

                        "pld        [%1, #128]          \n"
                        "vld1.u16   {d2-d3}, [%1]!      \n" // r0
                        "vld1.u16   {d8[0]}, [%1]       \n"

                        "vshll.u16  q0, d2, #16         \n"
                        "vshll.u16  q1, d3, #16         \n"
                        "vshl.u32   d8, d8, #16         \n"

                        "pld        [%8, #256]          \n"
                        "vld1.u16   {d14-d17}, [%8]!    \n"

                        "vshll.u16  q5, d14, #16        \n"
                        "vshll.u16  q6, d15, #16        \n"
                        "vshll.u16  q7, d16, #16        \n"
                        "vshll.u16  q8, d17, #16        \n"

                        "vmul.f32   q12, q5, d0[0]      \n"
                        "vmul.f32   q13, q5, d1[0]      \n"
                        "vmla.f32   q14, q6, d0[1]      \n"
                        "vmla.f32   q15, q6, d1[1]      \n"

                        "pld        [%8, #192]          \n"
                        "vld1.u16   {d20-d22}, [%8]!    \n"

                        "vshll.u16  q9, d20, #16        \n"
                        "vshll.u16  q10, d21, #16       \n"
                        "vshll.u16  q11, d22, #16       \n"

                        "vmla.f32   q12, q7, d1[0]      \n"
                        "vmla.f32   q13, q7, d2[0]      \n"

                        "pld        [%2, #128]          \n"
                        "vld1.u16   {d6-d7}, [%2]!      \n" // r1
                        "vld1.u16   {d9[0]}, [%2]       \n"

                        "vshll.u16  q2, d6, #16         \n"
                        "vshll.u16  q3, d7, #16         \n"
                        "vshl.u32   d9, d9, #16         \n"

                        "vmla.f32   q14, q8, d1[1]      \n"
                        "vmla.f32   q15, q8, d2[1]      \n"
                        "vmla.f32   q12, q9, d2[0]      \n"
                        "vmla.f32   q13, q9, d3[0]      \n"

                        "pld        [%8, #256]          \n"
                        "vld1.u16   {d14-d17}, [%8]!    \n"

                        "vshll.u16  q5, d14, #16        \n"
                        "vshll.u16  q6, d15, #16        \n"
                        "vshll.u16  q7, d16, #16        \n"
                        "vshll.u16  q8, d17, #16        \n"

                        "vmla.f32   q14, q10, d2[1]     \n"
                        "vmla.f32   q15, q10, d3[1]     \n"
                        "vmla.f32   q12, q11, d3[0]     \n"
                        "vmla.f32   q13, q11, d8[0]     \n"

                        "pld        [%8, #192]          \n"
                        "vld1.u16   {d20-d22}, [%8]!    \n"

                        "vshll.u16  q9, d20, #16        \n"
                        "vshll.u16  q10, d21, #16       \n"
                        "vshll.u16  q11, d22, #16       \n"

                        "vmla.f32   q14, q5, d4[0]      \n"
                        "vmla.f32   q15, q5, d5[0]      \n"
                        "vmla.f32   q12, q6, d4[1]      \n"
                        "vmla.f32   q13, q6, d5[1]      \n"
                        "vmla.f32   q14, q7, d5[0]      \n"
                        "vmla.f32   q15, q7, d6[0]      \n"

                        "pld        [%3, #128]          \n"
                        "vld1.u16   {d2-d3}, [%3]!      \n" // r2
                        "vld1.u16   {d8[0]}, [%3]       \n"

                        "vshll.u16  q0, d2, #16         \n"
                        "vshll.u16  q1, d3, #16         \n"
                        "vshl.u32   d8, d8, #16         \n"

                        "vmla.f32   q12, q8, d5[1]      \n"
                        "vmla.f32   q13, q8, d6[1]      \n"
                        "vmla.f32   q14, q9, d6[0]      \n"
                        "vmla.f32   q15, q9, d7[0]      \n"

                        "pld        [%8, #256]          \n"
                        "vld1.u16   {d14-d17}, [%8]!    \n"

                        "vshll.u16  q5, d14, #16        \n"
                        "vshll.u16  q6, d15, #16        \n"
                        "vshll.u16  q7, d16, #16        \n"
                        "vshll.u16  q8, d17, #16        \n"

                        "vmla.f32   q12, q10, d6[1]     \n"
                        "vmla.f32   q13, q10, d7[1]     \n"
                        "vmla.f32   q14, q11, d7[0]     \n"
                        "vmla.f32   q15, q11, d9[0]     \n"

                        "pld        [%8, #192]          \n"
                        "vld1.u16   {d20-d22}, [%8]!    \n"

                        "vshll.u16  q9, d20, #16        \n"
                        "vshll.u16  q10, d21, #16       \n"
                        "vshll.u16  q11, d22, #16       \n"

                        "vmla.f32   q12, q5, d0[0]      \n"
                        "vmla.f32   q13, q5, d1[0]      \n"
                        "vmla.f32   q14, q6, d0[1]      \n"
                        "vmla.f32   q15, q6, d1[1]      \n"
                        "vmla.f32   q12, q7, d1[0]      \n"
                        "vmla.f32   q13, q7, d2[0]      \n"

                        "pld        [%4, #128]          \n"
                        "vld1.u16   {d6-d7}, [%4]!      \n" // r3
                        "vld1.u16   {d9[0]}, [%4]       \n"

                        "vshll.u16  q2, d6, #16         \n"
                        "vshll.u16  q3, d7, #16         \n"
                        "vshl.u32   d9, d9, #16         \n"

                        "vmla.f32   q14, q8, d1[1]      \n"
                        "vmla.f32   q15, q8, d2[1]      \n"
                        "vmla.f32   q12, q9, d2[0]      \n"
                        "vmla.f32   q13, q9, d3[0]      \n"

                        "pld        [%8, #256]          \n"
                        "vld1.u16   {d14-d17}, [%8]!    \n"

                        "vshll.u16  q5, d14, #16        \n"
                        "vshll.u16  q6, d15, #16        \n"
                        "vshll.u16  q7, d16, #16        \n"
                        "vshll.u16  q8, d17, #16        \n"

                        "vmla.f32   q14, q10, d2[1]     \n"
                        "vmla.f32   q15, q10, d3[1]     \n"
                        "vmla.f32   q12, q11, d3[0]     \n"
                        "vmla.f32   q13, q11, d8[0]     \n"

                        "pld        [%8, #192]          \n"
                        "vld1.u16   {d20-d22}, [%8]!    \n"

                        "vshll.u16  q9, d20, #16        \n"
                        "vshll.u16  q10, d21, #16       \n"
                        "vshll.u16  q11, d22, #16       \n"

                        "vmla.f32   q14, q5, d4[0]      \n"
                        "vmla.f32   q15, q5, d5[0]      \n"
                        "vmla.f32   q12, q6, d4[1]      \n"
                        "vmla.f32   q13, q6, d5[1]      \n"
                        "vmla.f32   q14, q7, d5[0]      \n"
                        "vmla.f32   q15, q7, d6[0]      \n"

                        "pld        [%5, #128]          \n"
                        "vld1.u16   {d2-d3}, [%5]!      \n" // r4
                        "vld1.u16   {d8[0]}, [%5]       \n"

                        "vshll.u16  q0, d2, #16         \n"
                        "vshll.u16  q1, d3, #16         \n"
                        "vshl.u32   d8, d8, #16         \n"

                        "vmla.f32   q12, q8, d5[1]      \n"
                        "vmla.f32   q13, q8, d6[1]      \n"
                        "vmla.f32   q14, q9, d6[0]      \n"
                        "vmla.f32   q15, q9, d7[0]      \n"

                        "pld        [%8, #256]          \n"
                        "vld1.u16   {d14-d17}, [%8]!    \n"

                        "vshll.u16  q5, d14, #16        \n"
                        "vshll.u16  q6, d15, #16        \n"
                        "vshll.u16  q7, d16, #16        \n"
                        "vshll.u16  q8, d17, #16        \n"

                        "vmla.f32   q12, q10, d6[1]     \n"
                        "vmla.f32   q13, q10, d7[1]     \n"
                        "vmla.f32   q14, q11, d7[0]     \n"
                        "vmla.f32   q15, q11, d9[0]     \n"

                        "pld        [%8, #192]          \n"
                        "vld1.u16   {d20-d22}, [%8]!    \n"

                        "vshll.u16  q9, d20, #16        \n"
                        "vshll.u16  q10, d21, #16       \n"
                        "vshll.u16  q11, d22, #16       \n"

                        "vmla.f32   q12, q5, d0[0]      \n"
                        "vmla.f32   q13, q5, d1[0]      \n"
                        "vmla.f32   q14, q6, d0[1]      \n"
                        "vmla.f32   q15, q6, d1[1]      \n"
                        "vmla.f32   q12, q7, d1[0]      \n"
                        "vmla.f32   q13, q7, d2[0]      \n"

                        "pld        [%6, #128]          \n"
                        "vld1.u16   {d6-d7}, [%6]!      \n" // r5
                        "vld1.u16   {d9[0]}, [%6]       \n"

                        "vshll.u16  q2, d6, #16         \n"
                        "vshll.u16  q3, d7, #16         \n"
                        "vshl.u32   d9, d9, #16         \n"

                        "vmla.f32   q14, q8, d1[1]      \n"
                        "vmla.f32   q15, q8, d2[1]      \n"
                        "vmla.f32   q12, q9, d2[0]      \n"
                        "vmla.f32   q13, q9, d3[0]      \n"

                        "pld        [%8, #256]          \n"
                        "vld1.u16   {d14-d17}, [%8]!    \n"

                        "vshll.u16  q5, d14, #16        \n"
                        "vshll.u16  q6, d15, #16        \n"
                        "vshll.u16  q7, d16, #16        \n"
                        "vshll.u16  q8, d17, #16        \n"

                        "vmla.f32   q14, q10, d2[1]     \n"
                        "vmla.f32   q15, q10, d3[1]     \n"
                        "vmla.f32   q12, q11, d3[0]     \n"
                        "vmla.f32   q13, q11, d8[0]     \n"

                        "pld        [%8, #192]          \n"
                        "vld1.u16   {d20-d22}, [%8]!    \n"

                        "vshll.u16  q9, d20, #16        \n"
                        "vshll.u16  q10, d21, #16       \n"
                        "vshll.u16  q11, d22, #16       \n"

                        "vmla.f32   q14, q5, d4[0]      \n"
                        "vmla.f32   q15, q5, d5[0]      \n"
                        "vmla.f32   q12, q6, d4[1]      \n"
                        "vmla.f32   q13, q6, d5[1]      \n"
                        "vmla.f32   q14, q7, d5[0]      \n"
                        "vmla.f32   q15, q7, d6[0]      \n"

                        "pld        [%7, #128]          \n"
                        "vld1.u16   {d2-d3}, [%7]!      \n" // r6
                        "vld1.u16   {d8[0]}, [%7]       \n"

                        "vshll.u16  q0, d2, #16         \n"
                        "vshll.u16  q1, d3, #16         \n"
                        "vshl.u32   d8, d8, #16         \n"

                        "vmla.f32   q12, q8, d5[1]      \n"
                        "vmla.f32   q13, q8, d6[1]      \n"
                        "vmla.f32   q14, q9, d6[0]      \n"
                        "vmla.f32   q15, q9, d7[0]      \n"

                        "pld        [%8, #256]          \n"
                        "vld1.u16   {d14-d17}, [%8]!    \n"

                        "vshll.u16  q5, d14, #16        \n"
                        "vshll.u16  q6, d15, #16        \n"
                        "vshll.u16  q7, d16, #16        \n"
                        "vshll.u16  q8, d17, #16        \n"

                        "vmla.f32   q12, q10, d6[1]     \n"
                        "vmla.f32   q13, q10, d7[1]     \n"
                        "vmla.f32   q14, q11, d7[0]     \n"
                        "vmla.f32   q15, q11, d9[0]     \n"

                        "pld        [%8, #192]          \n"
                        "vld1.u16   {d20-d22}, [%8]!    \n"

                        "vshll.u16  q9, d20, #16        \n"
                        "vshll.u16  q10, d21, #16       \n"
                        "vshll.u16  q11, d22, #16       \n"

                        "vmla.f32   q12, q5, d0[0]      \n"
                        "vmla.f32   q13, q5, d1[0]      \n"
                        "vmla.f32   q14, q6, d0[1]      \n"
                        "vmla.f32   q15, q6, d1[1]      \n"

                        "sub        %1, %1, #8          \n"
                        "sub        %2, %2, #8          \n"

                        "vmla.f32   q12, q7, d1[0]      \n"
                        "vmla.f32   q13, q7, d2[0]      \n"
                        "vmla.f32   q14, q8, d1[1]      \n"
                        "vmla.f32   q15, q8, d2[1]      \n"

                        "sub        %8, %8, #392        \n"

                        "vmla.f32   q12, q9, d2[0]      \n"
                        "vmla.f32   q13, q9, d3[0]      \n"
                        "vmla.f32   q14, q10, d2[1]     \n"
                        "vmla.f32   q15, q10, d3[1]     \n"

                        "sub        %3, %3, #8          \n"
                        "sub        %4, %4, #8          \n"

                        "vmla.f32   q12, q11, d3[0]     \n"
                        "vmla.f32   q13, q11, d8[0]     \n"

                        "sub        %5, %5, #8          \n"
                        "sub        %6, %6, #8          \n"

                        "vadd.f32   q14, q14, q12       \n"
                        "vadd.f32   q15, q15, q13       \n"

                        "sub        %7, %7, #8          \n"

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

                        "prfm   pldl1keep, [%1, #128]       \n"
                        "ld1    {v0.4h, v1.4h}, [%1]        \n" // r0

                        "shll   v0.4s, v0.4h, #16           \n"
                        "shll   v1.4s, v1.4h, #16           \n"

                        "prfm   pldl1keep, [%8, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%8], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmul   v17.4s, v24.4s, v0.s[0]     \n"

                        "prfm   pldl1keep, [%8, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%8], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

                        "fmul   v18.4s, v25.4s, v0.s[1]     \n"
                        "fmul   v19.4s, v26.4s, v0.s[2]     \n"

                        "prfm   pldl1keep, [%2, #128]       \n"
                        "ld1    {v4.4h, v5.4h}, [%2]        \n" // r1

                        "shll   v4.4s, v4.4h, #16           \n"
                        "shll   v5.4s, v5.4h, #16           \n"

                        "fmla   v16.4s, v27.4s, v0.s[3]     \n"
                        "fmla   v17.4s, v28.4s, v1.s[0]     \n"

                        "prfm   pldl1keep, [%8, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%8], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v18.4s, v29.4s, v1.s[1]     \n"
                        "fmla   v19.4s, v30.4s, v1.s[2]     \n"

                        "prfm   pldl1keep, [%8, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%8], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

                        "fmla   v16.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v17.4s, v25.4s, v4.s[1]     \n"
                        "fmla   v18.4s, v26.4s, v4.s[2]     \n"

                        "prfm   pldl1keep, [%3, #128]       \n"
                        "ld1    {v0.4h, v1.4h}, [%3]        \n" // r2

                        "shll   v0.4s, v0.4h, #16           \n"
                        "shll   v1.4s, v1.4h, #16           \n"

                        "fmla   v19.4s, v27.4s, v4.s[3]     \n"
                        "fmla   v16.4s, v28.4s, v5.s[0]     \n"

                        "prfm   pldl1keep, [%8, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%8], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v17.4s, v29.4s, v5.s[1]     \n"
                        "fmla   v18.4s, v30.4s, v5.s[2]     \n"

                        "prfm   pldl1keep, [%8, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%8], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

                        "fmla   v19.4s, v24.4s, v0.s[0]     \n"
                        "fmla   v16.4s, v25.4s, v0.s[1]     \n"
                        "fmla   v17.4s, v26.4s, v0.s[2]     \n"

                        "prfm   pldl1keep, [%4, #128]       \n"
                        "ld1    {v4.4h, v5.4h}, [%4]        \n" // r3

                        "shll   v4.4s, v4.4h, #16           \n"
                        "shll   v5.4s, v5.4h, #16           \n"

                        "fmla   v18.4s, v27.4s, v0.s[3]     \n"
                        "fmla   v19.4s, v28.4s, v1.s[0]     \n"

                        "prfm   pldl1keep, [%8, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%8], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v16.4s, v29.4s, v1.s[1]     \n"
                        "fmla   v17.4s, v30.4s, v1.s[2]     \n"

                        "prfm   pldl1keep, [%8, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%8], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

                        "fmla   v18.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v19.4s, v25.4s, v4.s[1]     \n"
                        "fmla   v16.4s, v26.4s, v4.s[2]     \n"

                        "prfm   pldl1keep, [%5, #128]       \n"
                        "ld1    {v0.4h, v1.4h}, [%5]        \n" // r4

                        "shll   v0.4s, v0.4h, #16           \n"
                        "shll   v1.4s, v1.4h, #16           \n"

                        "fmla   v17.4s, v27.4s, v4.s[3]     \n"
                        "fmla   v18.4s, v28.4s, v5.s[0]     \n"

                        "prfm   pldl1keep, [%8, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%8], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v19.4s, v29.4s, v5.s[1]     \n"
                        "fmla   v16.4s, v30.4s, v5.s[2]     \n"

                        "prfm   pldl1keep, [%8, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%8], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

                        "fmla   v17.4s, v24.4s, v0.s[0]     \n"
                        "fmla   v18.4s, v25.4s, v0.s[1]     \n"
                        "fmla   v19.4s, v26.4s, v0.s[2]     \n"

                        "prfm   pldl1keep, [%6, #128]       \n"
                        "ld1    {v4.4h, v5.4h}, [%6]        \n" // r5

                        "shll   v4.4s, v4.4h, #16           \n"
                        "shll   v5.4s, v5.4h, #16           \n"

                        "fmla   v16.4s, v27.4s, v0.s[3]     \n"
                        "fmla   v17.4s, v28.4s, v1.s[0]     \n"

                        "prfm   pldl1keep, [%8, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%8], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v18.4s, v29.4s, v1.s[1]     \n"
                        "fmla   v19.4s, v30.4s, v1.s[2]     \n"

                        "prfm   pldl1keep, [%8, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%8], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

                        "fmla   v16.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v17.4s, v25.4s, v4.s[1]     \n"
                        "fmla   v18.4s, v26.4s, v4.s[2]     \n"

                        "prfm   pldl1keep, [%7, #128]       \n"
                        "ld1    {v0.4h, v1.4h}, [%7]        \n" // r6

                        "shll   v0.4s, v0.4h, #16           \n"
                        "shll   v1.4s, v1.4h, #16           \n"

                        "fmla   v19.4s, v27.4s, v4.s[3]     \n"
                        "fmla   v16.4s, v28.4s, v5.s[0]     \n"

                        "prfm   pldl1keep, [%8, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%8], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v17.4s, v29.4s, v5.s[1]     \n"
                        "fmla   v18.4s, v30.4s, v5.s[2]     \n"

                        "prfm   pldl1keep, [%8, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%8], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

                        "fmla   v19.4s, v24.4s, v0.s[0]     \n"
                        "fmla   v16.4s, v25.4s, v0.s[1]     \n"
                        "fmla   v17.4s, v26.4s, v0.s[2]     \n"

                        "add    %1, %1, #4                  \n"
                        "add    %2, %2, #4                  \n"

                        "fmla   v18.4s, v27.4s, v0.s[3]     \n"
                        "fmla   v19.4s, v28.4s, v1.s[0]     \n"
                        "fmla   v16.4s, v29.4s, v1.s[1]     \n"
                        "fmla   v17.4s, v30.4s, v1.s[2]     \n"

                        "add    %3, %3, #4                  \n"
                        "add    %4, %4, #4                  \n"

                        "fadd   v18.4s, v18.4s, v19.4s      \n"

                        "add    %5, %5, #4                  \n"

                        "fadd   v16.4s, v16.4s, v17.4s      \n"

                        "add    %6, %6, #4                  \n"
                        "add    %7, %7, #4                  \n"

                        "fadd   v16.4s, v16.4s, v18.4s      \n"

                        "sub    %8, %8, #392                \n"

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
                        "pld        [%0, #128]          \n"
                        "vld1.f32   {d8-d9}, [%0 :128]  \n"

                        "pld        [%1, #128]          \n"
                        "vld1.u16   {d2-d3}, [%1]       \n" // r0

                        "vshll.u16  q0, d2, #16         \n"
                        "vshll.u16  q1, d3, #16         \n"

                        "pld        [%8, #256]          \n"
                        "vld1.u16   {d20-d23}, [%8]!    \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmul.f32   q5, q8, d0[0]       \n"
                        "vmul.f32   q6, q9, d0[1]       \n"

                        "pld        [%8, #192]          \n"
                        "vld1.u16   {d26-d28}, [%8]!    \n"

                        "vshll.u16  q12, d26, #16       \n"
                        "vshll.u16  q13, d27, #16       \n"
                        "vshll.u16  q14, d28, #16       \n"

                        "vmul.f32   q7, q10, d1[0]      \n"
                        "vmla.f32   q4, q11, d1[1]      \n"

                        "pld        [%2, #128]          \n"
                        "vld1.u16   {d6-d7}, [%2]       \n" // r1

                        "vshll.u16  q2, d6, #16         \n"
                        "vshll.u16  q3, d7, #16         \n"

                        "vmla.f32   q5, q12, d2[0]      \n"

                        "pld        [%8, #256]          \n"
                        "vld1.u16   {d20-d23}, [%8]!    \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q6, q13, d2[1]      \n"
                        "vmla.f32   q7, q14, d3[0]      \n"

                        "pld        [%8, #192]          \n"
                        "vld1.u16   {d26-d28}, [%8]!    \n"

                        "vshll.u16  q12, d26, #16       \n"
                        "vshll.u16  q13, d27, #16       \n"
                        "vshll.u16  q14, d28, #16       \n"

                        "vmla.f32   q4, q8, d4[0]       \n"
                        "vmla.f32   q5, q9, d4[1]       \n"
                        "vmla.f32   q6, q10, d5[0]      \n"

                        "pld        [%3, #128]          \n"
                        "vld1.u16   {d2-d3}, [%3]       \n" // r2

                        "vshll.u16  q0, d2, #16         \n"
                        "vshll.u16  q1, d3, #16         \n"

                        "vmla.f32   q7, q11, d5[1]      \n"
                        "vmla.f32   q4, q12, d6[0]      \n"

                        "pld        [%8, #256]          \n"
                        "vld1.u16   {d20-d23}, [%8]!    \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q5, q13, d6[1]      \n"
                        "vmla.f32   q6, q14, d7[0]      \n"

                        "pld        [%8, #192]          \n"
                        "vld1.u16   {d26-d28}, [%8]!    \n"

                        "vshll.u16  q12, d26, #16       \n"
                        "vshll.u16  q13, d27, #16       \n"
                        "vshll.u16  q14, d28, #16       \n"

                        "vmla.f32   q7, q8, d0[0]       \n"
                        "vmla.f32   q4, q9, d0[1]       \n"
                        "vmla.f32   q5, q10, d1[0]      \n"

                        "pld        [%4, #128]          \n"
                        "vld1.u16   {d6-d7}, [%4]       \n" // r3

                        "vshll.u16  q2, d6, #16         \n"
                        "vshll.u16  q3, d7, #16         \n"

                        "vmla.f32   q6, q11, d1[1]      \n"
                        "vmla.f32   q7, q12, d2[0]      \n"

                        "pld        [%8, #256]          \n"
                        "vld1.u16   {d20-d23}, [%8]!    \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q4, q13, d2[1]      \n"
                        "vmla.f32   q5, q14, d3[0]      \n"

                        "pld        [%8, #192]          \n"
                        "vld1.u16   {d26-d28}, [%8]!    \n"

                        "vshll.u16  q12, d26, #16       \n"
                        "vshll.u16  q13, d27, #16       \n"
                        "vshll.u16  q14, d28, #16       \n"

                        "vmla.f32   q6, q8, d4[0]       \n"
                        "vmla.f32   q7, q9, d4[1]       \n"
                        "vmla.f32   q4, q10, d5[0]      \n"

                        "pld        [%5, #128]          \n"
                        "vld1.u16   {d2-d3}, [%5]       \n" // r4

                        "vshll.u16  q0, d2, #16         \n"
                        "vshll.u16  q1, d3, #16         \n"

                        "vmla.f32   q5, q11, d5[1]      \n"
                        "vmla.f32   q6, q12, d6[0]      \n"

                        "pld        [%8, #256]          \n"
                        "vld1.u16   {d20-d23}, [%8]!    \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q7, q13, d6[1]      \n"
                        "vmla.f32   q4, q14, d7[0]      \n"

                        "pld        [%8, #192]          \n"
                        "vld1.u16   {d26-d28}, [%8]!    \n"

                        "vshll.u16  q12, d26, #16       \n"
                        "vshll.u16  q13, d27, #16       \n"
                        "vshll.u16  q14, d28, #16       \n"

                        "vmla.f32   q5, q8, d0[0]       \n"
                        "vmla.f32   q6, q9, d0[1]       \n"
                        "vmla.f32   q7, q10, d1[0]      \n"

                        "pld        [%6, #128]          \n"
                        "vld1.u16   {d6-d7}, [%6]       \n" // r5

                        "vshll.u16  q2, d6, #16         \n"
                        "vshll.u16  q3, d7, #16         \n"

                        "vmla.f32   q4, q11, d1[1]      \n"
                        "vmla.f32   q5, q12, d2[0]      \n"

                        "pld        [%8, #256]          \n"
                        "vld1.u16   {d20-d23}, [%8]!    \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q6, q13, d2[1]      \n"
                        "vmla.f32   q7, q14, d3[0]      \n"

                        "pld        [%8, #192]          \n"
                        "vld1.u16   {d26-d28}, [%8]!    \n"

                        "vshll.u16  q12, d26, #16       \n"
                        "vshll.u16  q13, d27, #16       \n"
                        "vshll.u16  q14, d28, #16       \n"

                        "vmla.f32   q4, q8, d4[0]       \n"
                        "vmla.f32   q5, q9, d4[1]       \n"
                        "vmla.f32   q6, q10, d5[0]      \n"

                        "pld        [%7, #128]          \n"
                        "vld1.u16   {d2-d3}, [%7]       \n" // r6

                        "vshll.u16  q0, d2, #16         \n"
                        "vshll.u16  q1, d3, #16         \n"

                        "vmla.f32   q7, q11, d5[1]      \n"
                        "vmla.f32   q4, q12, d6[0]      \n"

                        "pld        [%8, #256]          \n"
                        "vld1.u16   {d20-d23}, [%8]!    \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q5, q13, d6[1]      \n"
                        "vmla.f32   q6, q14, d7[0]      \n"

                        "pld        [%8, #192]          \n"
                        "vld1.u16   {d26-d28}, [%8]!    \n"

                        "vshll.u16  q12, d26, #16       \n"
                        "vshll.u16  q13, d27, #16       \n"
                        "vshll.u16  q14, d28, #16       \n"

                        "vmla.f32   q7, q8, d0[0]       \n"
                        "vmla.f32   q4, q9, d0[1]       \n"

                        "add        %1, %1, #4          \n"
                        "add        %2, %2, #4          \n"

                        "vmla.f32   q5, q10, d1[0]      \n"
                        "vmla.f32   q6, q11, d1[1]      \n"

                        "sub        %8, %8, #392        \n"

                        "vmla.f32   q7, q12, d2[0]      \n"
                        "vmla.f32   q4, q13, d2[1]      \n"
                        "vmla.f32   q5, q14, d3[0]      \n"

                        "add        %3, %3, #4          \n"
                        "add        %4, %4, #4          \n"

                        "vadd.f32   q6, q6, q7          \n"

                        "add        %5, %5, #4          \n"

                        "vadd.f32   q4, q4, q5          \n"

                        "add        %6, %6, #4          \n"

                        "vadd.f32   q4, q4, q6          \n"

                        "add        %7, %7, #4          \n"

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
        for (; q < inch; q++)
        {
            unsigned short* outptr0_bf16 = top_blob.channel(p);

            float* outptr0 = out0.row(0);

            const Mat img0 = bottom_blob.channel(q);

            const unsigned short* r0 = img0.row<const unsigned short>(0);
            const unsigned short* r1 = img0.row<const unsigned short>(1);
            const unsigned short* r2 = img0.row<const unsigned short>(2);
            const unsigned short* r3 = img0.row<const unsigned short>(3);
            const unsigned short* r4 = img0.row<const unsigned short>(4);
            const unsigned short* r5 = img0.row<const unsigned short>(5);
            const unsigned short* r6 = img0.row<const unsigned short>(6);

            const unsigned short* kptr = kernel.channel(p).row<const unsigned short>(q);

            int i = 0;

            for (; i < outh; i++)
            {
                int j = 0;
#if __aarch64__
                for (; j + 7 < outw; j += 8)
                {
                    asm volatile(
                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%1], #64 \n"

                        "prfm   pldl1keep, [%2, #256]       \n"
                        "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%2], #32 \n" // r0

                        "shll   v0.4s, v0.4h, #16           \n"
                        "shll   v1.4s, v1.4h, #16           \n"
                        "shll   v2.4s, v2.4h, #16           \n"
                        "shll   v3.4s, v3.4h, #16           \n"

                        "prfm   pldl1keep, [%9, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%9], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v16.4s, v24.4s, v0.s[0]     \n"
                        "fmla   v17.4s, v24.4s, v0.s[2]     \n"
                        "fmla   v18.4s, v24.4s, v1.s[0]     \n"
                        "fmla   v19.4s, v24.4s, v1.s[2]     \n"

                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%1], #64 \n"

                        "fmla   v20.4s, v24.4s, v2.s[0]     \n"
                        "fmla   v21.4s, v24.4s, v2.s[2]     \n"
                        "fmla   v22.4s, v24.4s, v3.s[0]     \n"
                        "fmla   v23.4s, v24.4s, v3.s[2]     \n"

                        "prfm   pldl1keep, [%2, #128]       \n"
                        "ld1    {v4.4h, v5.4h}, [%2]        \n"

                        "shll   v4.4s, v4.4h, #16           \n"
                        "shll   v5.4s, v5.4h, #16           \n"

                        "fmla   v16.4s, v25.4s, v0.s[1]     \n"
                        "fmla   v17.4s, v25.4s, v0.s[3]     \n"
                        "fmla   v18.4s, v25.4s, v1.s[1]     \n"
                        "fmla   v19.4s, v25.4s, v1.s[3]     \n"
                        "fmla   v20.4s, v25.4s, v2.s[1]     \n"
                        "fmla   v21.4s, v25.4s, v2.s[3]     \n"
                        "fmla   v22.4s, v25.4s, v3.s[1]     \n"
                        "fmla   v23.4s, v25.4s, v3.s[3]     \n"

                        "prfm   pldl1keep, [%9, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%9], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

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

                        "prfm   pldl1keep, [%9, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%9], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v16.4s, v29.4s, v1.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v1.s[3]     \n"
                        "fmla   v18.4s, v29.4s, v2.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v2.s[3]     \n"
                        "fmla   v20.4s, v29.4s, v3.s[1]     \n"
                        "fmla   v21.4s, v29.4s, v3.s[3]     \n"
                        "fmla   v22.4s, v29.4s, v4.s[1]     \n"
                        "fmla   v23.4s, v29.4s, v4.s[3]     \n"

                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld1    {v6.4h, v7.4h, v8.4h, v9.4h}, [%3], #32 \n" // r1

                        "shll   v6.4s, v6.4h, #16           \n"
                        "shll   v7.4s, v7.4h, #16           \n"
                        "shll   v8.4s, v8.4h, #16           \n"
                        "shll   v9.4s, v9.4h, #16           \n"

                        "fmla   v16.4s, v30.4s, v1.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v2.s[0]     \n"
                        "fmla   v18.4s, v30.4s, v2.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v3.s[0]     \n"
                        "fmla   v20.4s, v30.4s, v3.s[2]     \n"
                        "fmla   v21.4s, v30.4s, v4.s[0]     \n"
                        "fmla   v22.4s, v30.4s, v4.s[2]     \n"
                        "fmla   v23.4s, v30.4s, v5.s[0]     \n"

                        "prfm   pldl1keep, [%3, #128]       \n"
                        "ld1    {v10.4h, v11.4h}, [%3]      \n"

                        "shll   v10.4s, v10.4h, #16         \n"
                        "shll   v11.4s, v11.4h, #16         \n"

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

                        "prfm   pldl1keep, [%9, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%9], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

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

                        "prfm   pldl1keep, [%9, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%9], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v16.4s, v29.4s, v7.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v7.s[3]     \n"
                        "fmla   v18.4s, v29.4s, v8.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v8.s[3]     \n"
                        "fmla   v20.4s, v29.4s, v9.s[1]     \n"
                        "fmla   v21.4s, v29.4s, v9.s[3]     \n"
                        "fmla   v22.4s, v29.4s, v10.s[1]    \n"
                        "fmla   v23.4s, v29.4s, v10.s[3]    \n"

                        "prfm   pldl1keep, [%4, #256]       \n"
                        "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%4], #32 \n" // r2

                        "shll   v0.4s, v0.4h, #16           \n"
                        "shll   v1.4s, v1.4h, #16           \n"
                        "shll   v2.4s, v2.4h, #16           \n"
                        "shll   v3.4s, v3.4h, #16           \n"

                        "fmla   v16.4s, v30.4s, v7.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v8.s[0]     \n"
                        "fmla   v18.4s, v30.4s, v8.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v9.s[0]     \n"
                        "fmla   v20.4s, v30.4s, v9.s[2]     \n"
                        "fmla   v21.4s, v30.4s, v10.s[0]    \n"
                        "fmla   v22.4s, v30.4s, v10.s[2]    \n"
                        "fmla   v23.4s, v30.4s, v11.s[0]    \n"

                        "prfm   pldl1keep, [%4, #128]       \n"
                        "ld1    {v4.4h, v5.4h}, [%4]        \n"

                        "shll   v4.4s, v4.4h, #16           \n"
                        "shll   v5.4s, v5.4h, #16           \n"

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

                        "prfm   pldl1keep, [%9, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%9], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

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

                        "prfm   pldl1keep, [%9, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%9], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v16.4s, v29.4s, v1.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v1.s[3]     \n"
                        "fmla   v18.4s, v29.4s, v2.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v2.s[3]     \n"
                        "fmla   v20.4s, v29.4s, v3.s[1]     \n"
                        "fmla   v21.4s, v29.4s, v3.s[3]     \n"
                        "fmla   v22.4s, v29.4s, v4.s[1]     \n"
                        "fmla   v23.4s, v29.4s, v4.s[3]     \n"

                        "prfm   pldl1keep, [%5, #256]       \n"
                        "ld1    {v6.4h, v7.4h, v8.4h, v9.4h}, [%5], #32 \n" // r3

                        "shll   v6.4s, v6.4h, #16           \n"
                        "shll   v7.4s, v7.4h, #16           \n"
                        "shll   v8.4s, v8.4h, #16           \n"
                        "shll   v9.4s, v9.4h, #16           \n"

                        "fmla   v16.4s, v30.4s, v1.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v2.s[0]     \n"
                        "fmla   v18.4s, v30.4s, v2.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v3.s[0]     \n"
                        "fmla   v20.4s, v30.4s, v3.s[2]     \n"
                        "fmla   v21.4s, v30.4s, v4.s[0]     \n"
                        "fmla   v22.4s, v30.4s, v4.s[2]     \n"
                        "fmla   v23.4s, v30.4s, v5.s[0]     \n"

                        "prfm   pldl1keep, [%5, #128]       \n"
                        "ld1    {v10.4h, v11.4h}, [%5]      \n"

                        "shll   v10.4s, v10.4h, #16         \n"
                        "shll   v11.4s, v11.4h, #16         \n"

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

                        "prfm   pldl1keep, [%9, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%9], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

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

                        "prfm   pldl1keep, [%9, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%9], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v16.4s, v29.4s, v7.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v7.s[3]     \n"
                        "fmla   v18.4s, v29.4s, v8.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v8.s[3]     \n"
                        "fmla   v20.4s, v29.4s, v9.s[1]     \n"
                        "fmla   v21.4s, v29.4s, v9.s[3]     \n"
                        "fmla   v22.4s, v29.4s, v10.s[1]    \n"
                        "fmla   v23.4s, v29.4s, v10.s[3]    \n"

                        "prfm   pldl1keep, [%6, #256]       \n"
                        "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%6], #32 \n" // r4

                        "shll   v0.4s, v0.4h, #16           \n"
                        "shll   v1.4s, v1.4h, #16           \n"
                        "shll   v2.4s, v2.4h, #16           \n"
                        "shll   v3.4s, v3.4h, #16           \n"

                        "fmla   v16.4s, v30.4s, v7.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v8.s[0]     \n"
                        "fmla   v18.4s, v30.4s, v8.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v9.s[0]     \n"
                        "fmla   v20.4s, v30.4s, v9.s[2]     \n"
                        "fmla   v21.4s, v30.4s, v10.s[0]    \n"
                        "fmla   v22.4s, v30.4s, v10.s[2]    \n"
                        "fmla   v23.4s, v30.4s, v11.s[0]    \n"

                        "prfm   pldl1keep, [%6, #128]       \n"
                        "ld1    {v4.4h, v5.4h}, [%6]        \n"

                        "shll   v4.4s, v4.4h, #16           \n"
                        "shll   v5.4s, v5.4h, #16           \n"

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

                        "prfm   pldl1keep, [%9, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%9], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

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

                        "prfm   pldl1keep, [%9, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%9], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v16.4s, v29.4s, v1.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v1.s[3]     \n"
                        "fmla   v18.4s, v29.4s, v2.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v2.s[3]     \n"
                        "fmla   v20.4s, v29.4s, v3.s[1]     \n"
                        "fmla   v21.4s, v29.4s, v3.s[3]     \n"
                        "fmla   v22.4s, v29.4s, v4.s[1]     \n"
                        "fmla   v23.4s, v29.4s, v4.s[3]     \n"

                        "prfm   pldl1keep, [%7, #256]       \n"
                        "ld1    {v6.4h, v7.4h, v8.4h, v9.4h}, [%7], #32 \n" // r5

                        "shll   v6.4s, v6.4h, #16           \n"
                        "shll   v7.4s, v7.4h, #16           \n"
                        "shll   v8.4s, v8.4h, #16           \n"
                        "shll   v9.4s, v9.4h, #16           \n"

                        "fmla   v16.4s, v30.4s, v1.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v2.s[0]     \n"
                        "fmla   v18.4s, v30.4s, v2.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v3.s[0]     \n"
                        "fmla   v20.4s, v30.4s, v3.s[2]     \n"
                        "fmla   v21.4s, v30.4s, v4.s[0]     \n"
                        "fmla   v22.4s, v30.4s, v4.s[2]     \n"
                        "fmla   v23.4s, v30.4s, v5.s[0]     \n"

                        "prfm   pldl1keep, [%7, #128]       \n"
                        "ld1    {v10.4h, v11.4h}, [%7]      \n"

                        "shll   v10.4s, v10.4h, #16         \n"
                        "shll   v11.4s, v11.4h, #16         \n"

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

                        "prfm   pldl1keep, [%9, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%9], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

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

                        "prfm   pldl1keep, [%9, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%9], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v16.4s, v29.4s, v7.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v7.s[3]     \n"
                        "fmla   v18.4s, v29.4s, v8.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v8.s[3]     \n"
                        "fmla   v20.4s, v29.4s, v9.s[1]     \n"
                        "fmla   v21.4s, v29.4s, v9.s[3]     \n"
                        "fmla   v22.4s, v29.4s, v10.s[1]    \n"
                        "fmla   v23.4s, v29.4s, v10.s[3]    \n"

                        "prfm   pldl1keep, [%8, #256]       \n"
                        "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%8], #32 \n" // r6

                        "shll   v0.4s, v0.4h, #16           \n"
                        "shll   v1.4s, v1.4h, #16           \n"
                        "shll   v2.4s, v2.4h, #16           \n"
                        "shll   v3.4s, v3.4h, #16           \n"

                        "fmla   v16.4s, v30.4s, v7.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v8.s[0]     \n"
                        "fmla   v18.4s, v30.4s, v8.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v9.s[0]     \n"
                        "fmla   v20.4s, v30.4s, v9.s[2]     \n"
                        "fmla   v21.4s, v30.4s, v10.s[0]    \n"
                        "fmla   v22.4s, v30.4s, v10.s[2]    \n"
                        "fmla   v23.4s, v30.4s, v11.s[0]    \n"

                        "prfm   pldl1keep, [%8, #128]       \n"
                        "ld1    {v4.4h, v5.4h}, [%8]        \n"

                        "shll   v4.4s, v4.4h, #16           \n"
                        "shll   v5.4s, v5.4h, #16           \n"

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

                        "prfm   pldl1keep, [%9, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%9], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

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

                        "sub    %9, %9, #392                \n"

                        "shrn   v16.4h, v16.4s, #16         \n"
                        "shrn   v17.4h, v17.4s, #16         \n"
                        "shrn   v18.4h, v18.4s, #16         \n"
                        "shrn   v19.4h, v19.4s, #16         \n"
                        "shrn   v20.4h, v20.4s, #16         \n"
                        "shrn   v21.4h, v21.4s, #16         \n"
                        "shrn   v22.4h, v22.4s, #16         \n"
                        "shrn   v23.4h, v23.4s, #16         \n"

                        "st1    {v16.4h, v17.4h, v18.4h, v19.4h}, [%0], #32 \n"
                        "st1    {v20.4h, v21.4h, v22.4h, v23.4h}, [%0], #32 \n"

                        : "=r"(outptr0_bf16), // %0
                        "=r"(outptr0),      // %1
                        "=r"(r0),           // %2
                        "=r"(r1),           // %3
                        "=r"(r2),           // %4
                        "=r"(r3),           // %5
                        "=r"(r4),           // %6
                        "=r"(r5),           // %7
                        "=r"(r6),           // %8
                        "=r"(kptr)          // %9
                        : "0"(outptr0_bf16),
                        "1"(outptr0),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "5"(r3),
                        "6"(r4),
                        "7"(r5),
                        "8"(r6),
                        "9"(kptr)
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v16", "v17", "v18", "v19", "v24", "v25", "v26", "v27", "v28", "v29", "v30");
                }
#endif // __aarch64__
                for (; j + 3 < outw; j += 4)
                {
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%1], #64 \n"

                        "prfm   pldl1keep, [%2, #256]       \n"
                        "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%2] \n" // r0

                        "shll   v0.4s, v0.4h, #16           \n"
                        "shll   v1.4s, v1.4h, #16           \n"
                        "shll   v2.4s, v2.4h, #16           \n"
                        "shll   v3.4s, v3.4h, #16           \n"

                        "prfm   pldl1keep, [%9, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%9], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v16.4s, v24.4s, v0.s[0]     \n"
                        "fmla   v17.4s, v24.4s, v0.s[2]     \n"
                        "fmla   v18.4s, v24.4s, v1.s[0]     \n"
                        "fmla   v19.4s, v24.4s, v1.s[2]     \n"

                        "prfm   pldl1keep, [%9, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%9], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

                        "fmla   v16.4s, v25.4s, v0.s[1]     \n"
                        "fmla   v17.4s, v25.4s, v0.s[3]     \n"
                        "fmla   v18.4s, v25.4s, v1.s[1]     \n"
                        "fmla   v19.4s, v25.4s, v1.s[3]     \n"
                        "fmla   v16.4s, v26.4s, v0.s[2]     \n"
                        "fmla   v17.4s, v26.4s, v1.s[0]     \n"
                        "fmla   v18.4s, v26.4s, v1.s[2]     \n"
                        "fmla   v19.4s, v26.4s, v2.s[0]     \n"

                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld1    {v4.4h, v5.4h, v6.4h, v7.4h}, [%3] \n" // r1

                        "shll   v4.4s, v4.4h, #16           \n"
                        "shll   v5.4s, v5.4h, #16           \n"
                        "shll   v6.4s, v6.4h, #16           \n"
                        "shll   v7.4s, v7.4h, #16           \n"

                        "fmla   v16.4s, v27.4s, v0.s[3]     \n"
                        "fmla   v17.4s, v27.4s, v1.s[1]     \n"
                        "fmla   v18.4s, v27.4s, v1.s[3]     \n"
                        "fmla   v19.4s, v27.4s, v2.s[1]     \n"
                        "fmla   v16.4s, v28.4s, v1.s[0]     \n"
                        "fmla   v17.4s, v28.4s, v1.s[2]     \n"
                        "fmla   v18.4s, v28.4s, v2.s[0]     \n"
                        "fmla   v19.4s, v28.4s, v2.s[2]     \n"

                        "prfm   pldl1keep, [%9, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%9], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v16.4s, v29.4s, v1.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v1.s[3]     \n"
                        "fmla   v18.4s, v29.4s, v2.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v2.s[3]     \n"
                        "fmla   v16.4s, v30.4s, v1.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v2.s[0]     \n"
                        "fmla   v18.4s, v30.4s, v2.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v3.s[0]     \n"

                        "prfm   pldl1keep, [%9, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%9], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

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

                        "prfm   pldl1keep, [%4, #256]       \n"
                        "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%4] \n" // r2

                        "shll   v0.4s, v0.4h, #16           \n"
                        "shll   v1.4s, v1.4h, #16           \n"
                        "shll   v2.4s, v2.4h, #16           \n"
                        "shll   v3.4s, v3.4h, #16           \n"

                        "fmla   v16.4s, v27.4s, v4.s[3]     \n"
                        "fmla   v17.4s, v27.4s, v5.s[1]     \n"
                        "fmla   v18.4s, v27.4s, v5.s[3]     \n"
                        "fmla   v19.4s, v27.4s, v6.s[1]     \n"
                        "fmla   v16.4s, v28.4s, v5.s[0]     \n"
                        "fmla   v17.4s, v28.4s, v5.s[2]     \n"
                        "fmla   v18.4s, v28.4s, v6.s[0]     \n"
                        "fmla   v19.4s, v28.4s, v6.s[2]     \n"

                        "prfm   pldl1keep, [%9, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%9], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v16.4s, v29.4s, v5.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v5.s[3]     \n"
                        "fmla   v18.4s, v29.4s, v6.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v6.s[3]     \n"
                        "fmla   v16.4s, v30.4s, v5.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v6.s[0]     \n"
                        "fmla   v18.4s, v30.4s, v6.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v7.s[0]     \n"

                        "prfm   pldl1keep, [%9, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%9], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

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

                        "prfm   pldl1keep, [%5, #256]       \n"
                        "ld1    {v4.4h, v5.4h, v6.4h, v7.4h}, [%5] \n" // r3

                        "shll   v4.4s, v4.4h, #16           \n"
                        "shll   v5.4s, v5.4h, #16           \n"
                        "shll   v6.4s, v6.4h, #16           \n"
                        "shll   v7.4s, v7.4h, #16           \n"

                        "fmla   v16.4s, v27.4s, v0.s[3]     \n"
                        "fmla   v17.4s, v27.4s, v1.s[1]     \n"
                        "fmla   v18.4s, v27.4s, v1.s[3]     \n"
                        "fmla   v19.4s, v27.4s, v2.s[1]     \n"
                        "fmla   v16.4s, v28.4s, v1.s[0]     \n"
                        "fmla   v17.4s, v28.4s, v1.s[2]     \n"
                        "fmla   v18.4s, v28.4s, v2.s[0]     \n"
                        "fmla   v19.4s, v28.4s, v2.s[2]     \n"

                        "prfm   pldl1keep, [%9, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%9], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v16.4s, v29.4s, v1.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v1.s[3]     \n"
                        "fmla   v18.4s, v29.4s, v2.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v2.s[3]     \n"
                        "fmla   v16.4s, v30.4s, v1.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v2.s[0]     \n"
                        "fmla   v18.4s, v30.4s, v2.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v3.s[0]     \n"

                        "prfm   pldl1keep, [%9, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%9], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

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

                        "prfm   pldl1keep, [%6, #256]       \n"
                        "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%6] \n" // r4

                        "shll   v0.4s, v0.4h, #16           \n"
                        "shll   v1.4s, v1.4h, #16           \n"
                        "shll   v2.4s, v2.4h, #16           \n"
                        "shll   v3.4s, v3.4h, #16           \n"

                        "fmla   v16.4s, v27.4s, v4.s[3]     \n"
                        "fmla   v17.4s, v27.4s, v5.s[1]     \n"
                        "fmla   v18.4s, v27.4s, v5.s[3]     \n"
                        "fmla   v19.4s, v27.4s, v6.s[1]     \n"
                        "fmla   v16.4s, v28.4s, v5.s[0]     \n"
                        "fmla   v17.4s, v28.4s, v5.s[2]     \n"
                        "fmla   v18.4s, v28.4s, v6.s[0]     \n"
                        "fmla   v19.4s, v28.4s, v6.s[2]     \n"

                        "prfm   pldl1keep, [%9, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%9], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v16.4s, v29.4s, v5.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v5.s[3]     \n"
                        "fmla   v18.4s, v29.4s, v6.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v6.s[3]     \n"
                        "fmla   v16.4s, v30.4s, v5.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v6.s[0]     \n"
                        "fmla   v18.4s, v30.4s, v6.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v7.s[0]     \n"

                        "prfm   pldl1keep, [%9, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%9], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

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

                        "prfm   pldl1keep, [%7, #256]       \n"
                        "ld1    {v4.4h, v5.4h, v6.4h, v7.4h}, [%7] \n" // r5

                        "shll   v4.4s, v4.4h, #16           \n"
                        "shll   v5.4s, v5.4h, #16           \n"
                        "shll   v6.4s, v6.4h, #16           \n"
                        "shll   v7.4s, v7.4h, #16           \n"

                        "fmla   v16.4s, v27.4s, v0.s[3]     \n"
                        "fmla   v17.4s, v27.4s, v1.s[1]     \n"
                        "fmla   v18.4s, v27.4s, v1.s[3]     \n"
                        "fmla   v19.4s, v27.4s, v2.s[1]     \n"
                        "fmla   v16.4s, v28.4s, v1.s[0]     \n"
                        "fmla   v17.4s, v28.4s, v1.s[2]     \n"
                        "fmla   v18.4s, v28.4s, v2.s[0]     \n"
                        "fmla   v19.4s, v28.4s, v2.s[2]     \n"

                        "prfm   pldl1keep, [%9, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%9], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v16.4s, v29.4s, v1.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v1.s[3]     \n"
                        "fmla   v18.4s, v29.4s, v2.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v2.s[3]     \n"
                        "fmla   v16.4s, v30.4s, v1.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v2.s[0]     \n"
                        "fmla   v18.4s, v30.4s, v2.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v3.s[0]     \n"

                        "prfm   pldl1keep, [%9, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%9], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

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

                        "prfm   pldl1keep, [%8, #256]       \n"
                        "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%8] \n" // r6

                        "shll   v0.4s, v0.4h, #16           \n"
                        "shll   v1.4s, v1.4h, #16           \n"
                        "shll   v2.4s, v2.4h, #16           \n"
                        "shll   v3.4s, v3.4h, #16           \n"

                        "fmla   v16.4s, v27.4s, v4.s[3]     \n"
                        "fmla   v17.4s, v27.4s, v5.s[1]     \n"
                        "fmla   v18.4s, v27.4s, v5.s[3]     \n"
                        "fmla   v19.4s, v27.4s, v6.s[1]     \n"
                        "fmla   v16.4s, v28.4s, v5.s[0]     \n"
                        "fmla   v17.4s, v28.4s, v5.s[2]     \n"
                        "fmla   v18.4s, v28.4s, v6.s[0]     \n"
                        "fmla   v19.4s, v28.4s, v6.s[2]     \n"

                        "prfm   pldl1keep, [%9, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%9], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v16.4s, v29.4s, v5.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v5.s[3]     \n"
                        "fmla   v18.4s, v29.4s, v6.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v6.s[3]     \n"
                        "fmla   v16.4s, v30.4s, v5.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v6.s[0]     \n"
                        "fmla   v18.4s, v30.4s, v6.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v7.s[0]     \n"

                        "prfm   pldl1keep, [%9, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%9], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

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

                        "add    %2, %2, #16                 \n"
                        "add    %3, %3, #16                 \n"
                        "add    %4, %4, #16                 \n"
                        "add    %5, %5, #16                 \n"
                        "add    %6, %6, #16                 \n"
                        "add    %7, %7, #16                 \n"
                        "add    %8, %8, #16                 \n"

                        "sub    %9, %9, #392                \n"

                        "shrn   v16.4h, v16.4s, #16         \n"
                        "shrn   v17.4h, v17.4s, #16         \n"
                        "shrn   v18.4h, v18.4s, #16         \n"
                        "shrn   v19.4h, v19.4s, #16         \n"

                        "st1    {v16.4h, v17.4h, v18.4h, v19.4h}, [%0], #32 \n"

                        : "=r"(outptr0_bf16), // %0
                        "=r"(outptr0),      // %1
                        "=r"(r0),           // %2
                        "=r"(r1),           // %3
                        "=r"(r2),           // %4
                        "=r"(r3),           // %5
                        "=r"(r4),           // %6
                        "=r"(r5),           // %7
                        "=r"(r6),           // %8
                        "=r"(kptr)          // %9
                        : "0"(outptr0_bf16),
                        "1"(outptr0),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "5"(r3),
                        "6"(r4),
                        "7"(r5),
                        "8"(r6),
                        "9"(kptr)
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v24", "v25", "v26", "v27", "v28", "v29", "v30");
#else  // __aarch64__
                    asm volatile(
                        "pld        [%1, #512]          \n"
                        "vldm       %1!, {d24-d31}      \n"

                        "pld        [%2, #128]          \n"
                        "vld1.u16   {d2-d3}, [%2]!      \n" // r0

                        "vshll.u16  q0, d2, #16         \n"
                        "vshll.u16  q1, d3, #16         \n"

                        "pld        [%9, #256]          \n"
                        "vld1.u16   {d14-d17}, [%9]!    \n"

                        "vshll.u16  q5, d14, #16        \n"
                        "vshll.u16  q6, d15, #16        \n"
                        "vshll.u16  q7, d16, #16        \n"
                        "vshll.u16  q8, d17, #16        \n"

                        "vmla.f32   q12, q5, d0[0]      \n"
                        "vmla.f32   q13, q5, d1[0]      \n"
                        "vmla.f32   q14, q5, d2[0]      \n"
                        "vmla.f32   q15, q5, d3[0]      \n"

                        "pld        [%2, #128]          \n"
                        "vld1.u16   {d5-d6}, [%2]       \n"

                        "vshll.u16  q2, d5, #16         \n"
                        "vshl.u32   d6, d6, #16         \n"

                        "vmla.f32   q12, q6, d0[1]      \n"
                        "vmla.f32   q13, q6, d1[1]      \n"
                        "vmla.f32   q14, q6, d2[1]      \n"
                        "vmla.f32   q15, q6, d3[1]      \n"

                        "pld        [%9, #192]          \n"
                        "vld1.u16   {d20-d22}, [%9]!    \n"

                        "vshll.u16  q9, d20, #16        \n"
                        "vshll.u16  q10, d21, #16       \n"
                        "vshll.u16  q11, d22, #16       \n"

                        "vmla.f32   q12, q7, d1[0]      \n"
                        "vmla.f32   q13, q7, d2[0]      \n"
                        "vmla.f32   q14, q7, d3[0]      \n"
                        "vmla.f32   q15, q7, d4[0]      \n"
                        "vmla.f32   q12, q8, d1[1]      \n"
                        "vmla.f32   q13, q8, d2[1]      \n"
                        "vmla.f32   q14, q8, d3[1]      \n"
                        "vmla.f32   q15, q8, d4[1]      \n"
                        "vmla.f32   q12, q9, d2[0]      \n"
                        "vmla.f32   q13, q9, d3[0]      \n"
                        "vmla.f32   q14, q9, d4[0]      \n"
                        "vmla.f32   q15, q9, d5[0]      \n"

                        "pld        [%9, #256]          \n"
                        "vld1.u16   {d14-d17}, [%9]!    \n"

                        "vshll.u16  q5, d14, #16        \n"
                        "vshll.u16  q6, d15, #16        \n"
                        "vshll.u16  q7, d16, #16        \n"
                        "vshll.u16  q8, d17, #16        \n"

                        "vmla.f32   q12, q10, d2[1]     \n"
                        "vmla.f32   q13, q10, d3[1]     \n"
                        "vmla.f32   q14, q10, d4[1]     \n"
                        "vmla.f32   q15, q10, d5[1]     \n"
                        "vmla.f32   q12, q11, d3[0]     \n"
                        "vmla.f32   q13, q11, d4[0]     \n"

                        "pld        [%3, #128]          \n"
                        "vld1.u16   {d2-d3}, [%3]!      \n" // r1

                        "vshll.u16  q0, d2, #16         \n"
                        "vshll.u16  q1, d3, #16         \n"

                        "vmla.f32   q14, q11, d5[0]     \n"
                        "vmla.f32   q15, q11, d6[0]     \n"

                        "vmla.f32   q12, q5, d0[0]      \n"
                        "vmla.f32   q13, q5, d1[0]      \n"
                        "vmla.f32   q14, q5, d2[0]      \n"
                        "vmla.f32   q15, q5, d3[0]      \n"

                        "pld        [%3, #128]          \n"
                        "vld1.u16   {d5-d6}, [%3]       \n"

                        "vshll.u16  q2, d5, #16         \n"
                        "vshl.u32   d6, d6, #16         \n"

                        "vmla.f32   q12, q6, d0[1]      \n"
                        "vmla.f32   q13, q6, d1[1]      \n"
                        "vmla.f32   q14, q6, d2[1]      \n"
                        "vmla.f32   q15, q6, d3[1]      \n"

                        "pld        [%9, #192]          \n"
                        "vld1.u16   {d20-d22}, [%9]!    \n"

                        "vshll.u16  q9, d20, #16        \n"
                        "vshll.u16  q10, d21, #16       \n"
                        "vshll.u16  q11, d22, #16       \n"

                        "vmla.f32   q12, q7, d1[0]      \n"
                        "vmla.f32   q13, q7, d2[0]      \n"
                        "vmla.f32   q14, q7, d3[0]      \n"
                        "vmla.f32   q15, q7, d4[0]      \n"
                        "vmla.f32   q12, q8, d1[1]      \n"
                        "vmla.f32   q13, q8, d2[1]      \n"
                        "vmla.f32   q14, q8, d3[1]      \n"
                        "vmla.f32   q15, q8, d4[1]      \n"
                        "vmla.f32   q12, q9, d2[0]      \n"
                        "vmla.f32   q13, q9, d3[0]      \n"
                        "vmla.f32   q14, q9, d4[0]      \n"
                        "vmla.f32   q15, q9, d5[0]      \n"

                        "pld        [%9, #256]          \n"
                        "vld1.u16   {d14-d17}, [%9]!    \n"

                        "vshll.u16  q5, d14, #16        \n"
                        "vshll.u16  q6, d15, #16        \n"
                        "vshll.u16  q7, d16, #16        \n"
                        "vshll.u16  q8, d17, #16        \n"

                        "vmla.f32   q12, q10, d2[1]     \n"
                        "vmla.f32   q13, q10, d3[1]     \n"
                        "vmla.f32   q14, q10, d4[1]     \n"
                        "vmla.f32   q15, q10, d5[1]     \n"
                        "vmla.f32   q12, q11, d3[0]     \n"
                        "vmla.f32   q13, q11, d4[0]     \n"

                        "pld        [%4, #128]          \n"
                        "vld1.u16   {d2-d3}, [%4]!      \n" // r2

                        "vshll.u16  q0, d2, #16         \n"
                        "vshll.u16  q1, d3, #16         \n"

                        "vmla.f32   q14, q11, d5[0]     \n"
                        "vmla.f32   q15, q11, d6[0]     \n"

                        "vmla.f32   q12, q5, d0[0]      \n"
                        "vmla.f32   q13, q5, d1[0]      \n"
                        "vmla.f32   q14, q5, d2[0]      \n"
                        "vmla.f32   q15, q5, d3[0]      \n"

                        "pld        [%4, #128]          \n"
                        "vld1.u16   {d5-d6}, [%4]       \n"

                        "vshll.u16  q2, d5, #16         \n"
                        "vshl.u32   d6, d6, #16         \n"

                        "vmla.f32   q12, q6, d0[1]      \n"
                        "vmla.f32   q13, q6, d1[1]      \n"
                        "vmla.f32   q14, q6, d2[1]      \n"
                        "vmla.f32   q15, q6, d3[1]      \n"

                        "pld        [%9, #192]          \n"
                        "vld1.u16   {d20-d22}, [%9]!    \n"

                        "vshll.u16  q9, d20, #16        \n"
                        "vshll.u16  q10, d21, #16       \n"
                        "vshll.u16  q11, d22, #16       \n"

                        "vmla.f32   q12, q7, d1[0]      \n"
                        "vmla.f32   q13, q7, d2[0]      \n"
                        "vmla.f32   q14, q7, d3[0]      \n"
                        "vmla.f32   q15, q7, d4[0]      \n"
                        "vmla.f32   q12, q8, d1[1]      \n"
                        "vmla.f32   q13, q8, d2[1]      \n"
                        "vmla.f32   q14, q8, d3[1]      \n"
                        "vmla.f32   q15, q8, d4[1]      \n"
                        "vmla.f32   q12, q9, d2[0]      \n"
                        "vmla.f32   q13, q9, d3[0]      \n"
                        "vmla.f32   q14, q9, d4[0]      \n"
                        "vmla.f32   q15, q9, d5[0]      \n"

                        "pld        [%9, #256]          \n"
                        "vld1.u16   {d14-d17}, [%9]!    \n"

                        "vshll.u16  q5, d14, #16        \n"
                        "vshll.u16  q6, d15, #16        \n"
                        "vshll.u16  q7, d16, #16        \n"
                        "vshll.u16  q8, d17, #16        \n"

                        "vmla.f32   q12, q10, d2[1]     \n"
                        "vmla.f32   q13, q10, d3[1]     \n"
                        "vmla.f32   q14, q10, d4[1]     \n"
                        "vmla.f32   q15, q10, d5[1]     \n"
                        "vmla.f32   q12, q11, d3[0]     \n"
                        "vmla.f32   q13, q11, d4[0]     \n"

                        "pld        [%5, #128]          \n"
                        "vld1.u16   {d2-d3}, [%5]!      \n" // r3

                        "vshll.u16  q0, d2, #16         \n"
                        "vshll.u16  q1, d3, #16         \n"

                        "vmla.f32   q14, q11, d5[0]     \n"
                        "vmla.f32   q15, q11, d6[0]     \n"

                        "vmla.f32   q12, q5, d0[0]      \n"
                        "vmla.f32   q13, q5, d1[0]      \n"
                        "vmla.f32   q14, q5, d2[0]      \n"
                        "vmla.f32   q15, q5, d3[0]      \n"

                        "pld        [%5, #128]          \n"
                        "vld1.u16   {d5-d6}, [%5]       \n"

                        "vshll.u16  q2, d5, #16         \n"
                        "vshl.u32   d6, d6, #16         \n"

                        "vmla.f32   q12, q6, d0[1]      \n"
                        "vmla.f32   q13, q6, d1[1]      \n"
                        "vmla.f32   q14, q6, d2[1]      \n"
                        "vmla.f32   q15, q6, d3[1]      \n"

                        "pld        [%9, #192]          \n"
                        "vld1.u16   {d20-d22}, [%9]!    \n"

                        "vshll.u16  q9, d20, #16        \n"
                        "vshll.u16  q10, d21, #16       \n"
                        "vshll.u16  q11, d22, #16       \n"

                        "vmla.f32   q12, q7, d1[0]      \n"
                        "vmla.f32   q13, q7, d2[0]      \n"
                        "vmla.f32   q14, q7, d3[0]      \n"
                        "vmla.f32   q15, q7, d4[0]      \n"
                        "vmla.f32   q12, q8, d1[1]      \n"
                        "vmla.f32   q13, q8, d2[1]      \n"
                        "vmla.f32   q14, q8, d3[1]      \n"
                        "vmla.f32   q15, q8, d4[1]      \n"
                        "vmla.f32   q12, q9, d2[0]      \n"
                        "vmla.f32   q13, q9, d3[0]      \n"
                        "vmla.f32   q14, q9, d4[0]      \n"
                        "vmla.f32   q15, q9, d5[0]      \n"

                        "pld        [%9, #256]          \n"
                        "vld1.u16   {d14-d17}, [%9]!    \n"

                        "vshll.u16  q5, d14, #16        \n"
                        "vshll.u16  q6, d15, #16        \n"
                        "vshll.u16  q7, d16, #16        \n"
                        "vshll.u16  q8, d17, #16        \n"

                        "vmla.f32   q12, q10, d2[1]     \n"
                        "vmla.f32   q13, q10, d3[1]     \n"
                        "vmla.f32   q14, q10, d4[1]     \n"
                        "vmla.f32   q15, q10, d5[1]     \n"
                        "vmla.f32   q12, q11, d3[0]     \n"
                        "vmla.f32   q13, q11, d4[0]     \n"

                        "pld        [%6, #128]          \n"
                        "vld1.u16   {d2-d3}, [%6]!      \n" // r4

                        "vshll.u16  q0, d2, #16         \n"
                        "vshll.u16  q1, d3, #16         \n"

                        "vmla.f32   q14, q11, d5[0]     \n"
                        "vmla.f32   q15, q11, d6[0]     \n"

                        "vmla.f32   q12, q5, d0[0]      \n"
                        "vmla.f32   q13, q5, d1[0]      \n"
                        "vmla.f32   q14, q5, d2[0]      \n"
                        "vmla.f32   q15, q5, d3[0]      \n"

                        "pld        [%6, #128]          \n"
                        "vld1.u16   {d5-d6}, [%6]       \n"

                        "vshll.u16  q2, d5, #16         \n"
                        "vshl.u32   d6, d6, #16         \n"

                        "vmla.f32   q12, q6, d0[1]      \n"
                        "vmla.f32   q13, q6, d1[1]      \n"
                        "vmla.f32   q14, q6, d2[1]      \n"
                        "vmla.f32   q15, q6, d3[1]      \n"

                        "pld        [%9, #192]          \n"
                        "vld1.u16   {d20-d22}, [%9]!    \n"

                        "vshll.u16  q9, d20, #16        \n"
                        "vshll.u16  q10, d21, #16       \n"
                        "vshll.u16  q11, d22, #16       \n"

                        "vmla.f32   q12, q7, d1[0]      \n"
                        "vmla.f32   q13, q7, d2[0]      \n"
                        "vmla.f32   q14, q7, d3[0]      \n"
                        "vmla.f32   q15, q7, d4[0]      \n"
                        "vmla.f32   q12, q8, d1[1]      \n"
                        "vmla.f32   q13, q8, d2[1]      \n"
                        "vmla.f32   q14, q8, d3[1]      \n"
                        "vmla.f32   q15, q8, d4[1]      \n"
                        "vmla.f32   q12, q9, d2[0]      \n"
                        "vmla.f32   q13, q9, d3[0]      \n"
                        "vmla.f32   q14, q9, d4[0]      \n"
                        "vmla.f32   q15, q9, d5[0]      \n"

                        "pld        [%9, #256]          \n"
                        "vld1.u16   {d14-d17}, [%9]!    \n"

                        "vshll.u16  q5, d14, #16        \n"
                        "vshll.u16  q6, d15, #16        \n"
                        "vshll.u16  q7, d16, #16        \n"
                        "vshll.u16  q8, d17, #16        \n"

                        "vmla.f32   q12, q10, d2[1]     \n"
                        "vmla.f32   q13, q10, d3[1]     \n"
                        "vmla.f32   q14, q10, d4[1]     \n"
                        "vmla.f32   q15, q10, d5[1]     \n"
                        "vmla.f32   q12, q11, d3[0]     \n"
                        "vmla.f32   q13, q11, d4[0]     \n"

                        "pld        [%7, #128]          \n"
                        "vld1.u16   {d2-d3}, [%7]!      \n" // r5

                        "vshll.u16  q0, d2, #16         \n"
                        "vshll.u16  q1, d3, #16         \n"

                        "vmla.f32   q14, q11, d5[0]     \n"
                        "vmla.f32   q15, q11, d6[0]     \n"

                        "vmla.f32   q12, q5, d0[0]      \n"
                        "vmla.f32   q13, q5, d1[0]      \n"
                        "vmla.f32   q14, q5, d2[0]      \n"
                        "vmla.f32   q15, q5, d3[0]      \n"

                        "pld        [%7, #128]          \n"
                        "vld1.u16   {d5-d6}, [%7]       \n"

                        "vshll.u16  q2, d5, #16         \n"
                        "vshl.u32   d6, d6, #16         \n"

                        "vmla.f32   q12, q6, d0[1]      \n"
                        "vmla.f32   q13, q6, d1[1]      \n"
                        "vmla.f32   q14, q6, d2[1]      \n"
                        "vmla.f32   q15, q6, d3[1]      \n"

                        "pld        [%9, #192]          \n"
                        "vld1.u16   {d20-d22}, [%9]!    \n"

                        "vshll.u16  q9, d20, #16        \n"
                        "vshll.u16  q10, d21, #16       \n"
                        "vshll.u16  q11, d22, #16       \n"

                        "vmla.f32   q12, q7, d1[0]      \n"
                        "vmla.f32   q13, q7, d2[0]      \n"
                        "vmla.f32   q14, q7, d3[0]      \n"
                        "vmla.f32   q15, q7, d4[0]      \n"
                        "vmla.f32   q12, q8, d1[1]      \n"
                        "vmla.f32   q13, q8, d2[1]      \n"
                        "vmla.f32   q14, q8, d3[1]      \n"
                        "vmla.f32   q15, q8, d4[1]      \n"
                        "vmla.f32   q12, q9, d2[0]      \n"
                        "vmla.f32   q13, q9, d3[0]      \n"
                        "vmla.f32   q14, q9, d4[0]      \n"
                        "vmla.f32   q15, q9, d5[0]      \n"

                        "pld        [%9, #256]          \n"
                        "vld1.u16   {d14-d17}, [%9]!    \n"

                        "vshll.u16  q5, d14, #16        \n"
                        "vshll.u16  q6, d15, #16        \n"
                        "vshll.u16  q7, d16, #16        \n"
                        "vshll.u16  q8, d17, #16        \n"

                        "vmla.f32   q12, q10, d2[1]     \n"
                        "vmla.f32   q13, q10, d3[1]     \n"
                        "vmla.f32   q14, q10, d4[1]     \n"
                        "vmla.f32   q15, q10, d5[1]     \n"
                        "vmla.f32   q12, q11, d3[0]     \n"
                        "vmla.f32   q13, q11, d4[0]     \n"

                        "pld        [%8, #128]          \n"
                        "vld1.u16   {d2-d3}, [%8]!      \n" // r6

                        "vshll.u16  q0, d2, #16         \n"
                        "vshll.u16  q1, d3, #16         \n"

                        "vmla.f32   q14, q11, d5[0]     \n"
                        "vmla.f32   q15, q11, d6[0]     \n"

                        "vmla.f32   q12, q5, d0[0]      \n"
                        "vmla.f32   q13, q5, d1[0]      \n"
                        "vmla.f32   q14, q5, d2[0]      \n"
                        "vmla.f32   q15, q5, d3[0]      \n"

                        "pld        [%8, #128]          \n"
                        "vld1.u16   {d5-d6}, [%8]       \n"

                        "vshll.u16  q2, d5, #16         \n"
                        "vshl.u32   d6, d6, #16         \n"

                        "vmla.f32   q12, q6, d0[1]      \n"
                        "vmla.f32   q13, q6, d1[1]      \n"
                        "vmla.f32   q14, q6, d2[1]      \n"
                        "vmla.f32   q15, q6, d3[1]      \n"

                        "pld        [%9, #192]          \n"
                        "vld1.u16   {d20-d22}, [%9]!    \n"

                        "vshll.u16  q9, d20, #16        \n"
                        "vshll.u16  q10, d21, #16       \n"
                        "vshll.u16  q11, d22, #16       \n"

                        "vmla.f32   q12, q7, d1[0]      \n"
                        "vmla.f32   q13, q7, d2[0]      \n"
                        "vmla.f32   q14, q7, d3[0]      \n"
                        "vmla.f32   q15, q7, d4[0]      \n"
                        "vmla.f32   q12, q8, d1[1]      \n"
                        "vmla.f32   q13, q8, d2[1]      \n"
                        "vmla.f32   q14, q8, d3[1]      \n"
                        "vmla.f32   q15, q8, d4[1]      \n"
                        "vmla.f32   q12, q9, d2[0]      \n"
                        "vmla.f32   q13, q9, d3[0]      \n"
                        "vmla.f32   q14, q9, d4[0]      \n"
                        "vmla.f32   q15, q9, d5[0]      \n"
                        "vmla.f32   q12, q10, d2[1]     \n"
                        "vmla.f32   q13, q10, d3[1]     \n"
                        "vmla.f32   q14, q10, d4[1]     \n"
                        "vmla.f32   q15, q10, d5[1]     \n"
                        "vmla.f32   q12, q11, d3[0]     \n"
                        "vmla.f32   q13, q11, d4[0]     \n"
                        "vmla.f32   q14, q11, d5[0]     \n"
                        "vmla.f32   q15, q11, d6[0]     \n"

                        "sub        %9, %9, #392        \n"

                        "vshrn.u32  d24, q12, #16       \n"
                        "vshrn.u32  d25, q13, #16       \n"
                        "vshrn.u32  d26, q14, #16       \n"
                        "vshrn.u32  d27, q15, #16       \n"

                        "vst1.u16   {d24-d27}, [%0]!    \n"

                        : "=r"(outptr0_bf16), // %0
                        "=r"(outptr0),      // %1
                        "=r"(r0),           // %2
                        "=r"(r1),           // %3
                        "=r"(r2),           // %4
                        "=r"(r3),           // %5
                        "=r"(r4),           // %6
                        "=r"(r5),           // %7
                        "=r"(r6),           // %8
                        "=r"(kptr)          // %9
                        : "0"(outptr0_bf16),
                        "1"(outptr0),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "5"(r3),
                        "6"(r4),
                        "7"(r5),
                        "8"(r6),
                        "9"(kptr)
                        : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif // __aarch64__
                }
                for (; j + 1 < outw; j += 2)
                {
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%1, #256]       \n"
                        "ld1    {v16.4s, v17.4s}, [%1], #32 \n"

                        "prfm   pldl1keep, [%2, #192]       \n"
                        "ld1    {v0.4h, v1.4h, v2.4h}, [%2] \n" // r0

                        "shll   v0.4s, v0.4h, #16           \n"
                        "shll   v1.4s, v1.4h, #16           \n"
                        "shll   v2.4s, v2.4h, #16           \n"

                        "prfm   pldl1keep, [%9, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%9], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmul   v18.4s, v24.4s, v0.s[0]     \n"
                        "fmul   v19.4s, v24.4s, v0.s[2]     \n"

                        "prfm   pldl1keep, [%9, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%9], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

                        "fmla   v16.4s, v25.4s, v0.s[1]     \n"
                        "fmla   v17.4s, v25.4s, v0.s[3]     \n"
                        "fmla   v18.4s, v26.4s, v0.s[2]     \n"
                        "fmla   v19.4s, v26.4s, v1.s[0]     \n"

                        "prfm   pldl1keep, [%3, #192]       \n"
                        "ld1    {v4.4h, v5.4h, v6.4h}, [%3] \n" // r1

                        "shll   v4.4s, v4.4h, #16           \n"
                        "shll   v5.4s, v5.4h, #16           \n"
                        "shll   v6.4s, v6.4h, #16           \n"

                        "fmla   v16.4s, v27.4s, v0.s[3]     \n"
                        "fmla   v17.4s, v27.4s, v1.s[1]     \n"
                        "fmla   v18.4s, v28.4s, v1.s[0]     \n"
                        "fmla   v19.4s, v28.4s, v1.s[2]     \n"

                        "prfm   pldl1keep, [%9, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%9], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v16.4s, v29.4s, v1.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v1.s[3]     \n"
                        "fmla   v18.4s, v30.4s, v1.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v2.s[0]     \n"

                        "prfm   pldl1keep, [%9, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%9], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

                        "fmla   v16.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v17.4s, v24.4s, v4.s[2]     \n"
                        "fmla   v18.4s, v25.4s, v4.s[1]     \n"
                        "fmla   v19.4s, v25.4s, v4.s[3]     \n"
                        "fmla   v16.4s, v26.4s, v4.s[2]     \n"
                        "fmla   v17.4s, v26.4s, v5.s[0]     \n"

                        "prfm   pldl1keep, [%4, #192]       \n"
                        "ld1    {v0.4h, v1.4h, v2.4h}, [%4] \n" // r2

                        "shll   v0.4s, v0.4h, #16           \n"
                        "shll   v1.4s, v1.4h, #16           \n"
                        "shll   v2.4s, v2.4h, #16           \n"

                        "fmla   v18.4s, v27.4s, v4.s[3]     \n"
                        "fmla   v19.4s, v27.4s, v5.s[1]     \n"
                        "fmla   v16.4s, v28.4s, v5.s[0]     \n"
                        "fmla   v17.4s, v28.4s, v5.s[2]     \n"

                        "prfm   pldl1keep, [%9, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%9], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v18.4s, v29.4s, v5.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v5.s[3]     \n"
                        "fmla   v16.4s, v30.4s, v5.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v6.s[0]     \n"

                        "prfm   pldl1keep, [%9, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%9], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

                        "fmla   v18.4s, v24.4s, v0.s[0]     \n"
                        "fmla   v19.4s, v24.4s, v0.s[2]     \n"
                        "fmla   v16.4s, v25.4s, v0.s[1]     \n"
                        "fmla   v17.4s, v25.4s, v0.s[3]     \n"
                        "fmla   v18.4s, v26.4s, v0.s[2]     \n"
                        "fmla   v19.4s, v26.4s, v1.s[0]     \n"

                        "prfm   pldl1keep, [%5, #192]       \n"
                        "ld1    {v4.4h, v5.4h, v6.4h}, [%5] \n" // r3

                        "shll   v4.4s, v4.4h, #16           \n"
                        "shll   v5.4s, v5.4h, #16           \n"
                        "shll   v6.4s, v6.4h, #16           \n"

                        "fmla   v16.4s, v27.4s, v0.s[3]     \n"
                        "fmla   v17.4s, v27.4s, v1.s[1]     \n"
                        "fmla   v18.4s, v28.4s, v1.s[0]     \n"
                        "fmla   v19.4s, v28.4s, v1.s[2]     \n"

                        "prfm   pldl1keep, [%9, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%9], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v16.4s, v29.4s, v1.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v1.s[3]     \n"
                        "fmla   v18.4s, v30.4s, v1.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v2.s[0]     \n"

                        "prfm   pldl1keep, [%9, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%9], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

                        "fmla   v16.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v17.4s, v24.4s, v4.s[2]     \n"
                        "fmla   v18.4s, v25.4s, v4.s[1]     \n"
                        "fmla   v19.4s, v25.4s, v4.s[3]     \n"
                        "fmla   v16.4s, v26.4s, v4.s[2]     \n"
                        "fmla   v17.4s, v26.4s, v5.s[0]     \n"

                        "prfm   pldl1keep, [%6, #192]       \n"
                        "ld1    {v0.4h, v1.4h, v2.4h}, [%6] \n" // r4

                        "shll   v0.4s, v0.4h, #16           \n"
                        "shll   v1.4s, v1.4h, #16           \n"
                        "shll   v2.4s, v2.4h, #16           \n"

                        "fmla   v18.4s, v27.4s, v4.s[3]     \n"
                        "fmla   v19.4s, v27.4s, v5.s[1]     \n"
                        "fmla   v16.4s, v28.4s, v5.s[0]     \n"
                        "fmla   v17.4s, v28.4s, v5.s[2]     \n"

                        "prfm   pldl1keep, [%9, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%9], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v18.4s, v29.4s, v5.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v5.s[3]     \n"
                        "fmla   v16.4s, v30.4s, v5.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v6.s[0]     \n"

                        "prfm   pldl1keep, [%9, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%9], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

                        "fmla   v18.4s, v24.4s, v0.s[0]     \n"
                        "fmla   v19.4s, v24.4s, v0.s[2]     \n"
                        "fmla   v16.4s, v25.4s, v0.s[1]     \n"
                        "fmla   v17.4s, v25.4s, v0.s[3]     \n"
                        "fmla   v18.4s, v26.4s, v0.s[2]     \n"
                        "fmla   v19.4s, v26.4s, v1.s[0]     \n"

                        "prfm   pldl1keep, [%7, #192]       \n"
                        "ld1    {v4.4h, v5.4h, v6.4h}, [%7] \n" // r5

                        "shll   v4.4s, v4.4h, #16           \n"
                        "shll   v5.4s, v5.4h, #16           \n"
                        "shll   v6.4s, v6.4h, #16           \n"

                        "fmla   v16.4s, v27.4s, v0.s[3]     \n"
                        "fmla   v17.4s, v27.4s, v1.s[1]     \n"
                        "fmla   v18.4s, v28.4s, v1.s[0]     \n"
                        "fmla   v19.4s, v28.4s, v1.s[2]     \n"

                        "prfm   pldl1keep, [%9, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%9], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v16.4s, v29.4s, v1.s[1]     \n"
                        "fmla   v17.4s, v29.4s, v1.s[3]     \n"
                        "fmla   v18.4s, v30.4s, v1.s[2]     \n"
                        "fmla   v19.4s, v30.4s, v2.s[0]     \n"

                        "prfm   pldl1keep, [%9, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%9], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

                        "fmla   v16.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v17.4s, v24.4s, v4.s[2]     \n"
                        "fmla   v18.4s, v25.4s, v4.s[1]     \n"
                        "fmla   v19.4s, v25.4s, v4.s[3]     \n"
                        "fmla   v16.4s, v26.4s, v4.s[2]     \n"
                        "fmla   v17.4s, v26.4s, v5.s[0]     \n"

                        "prfm   pldl1keep, [%8, #192]       \n"
                        "ld1    {v0.4h, v1.4h, v2.4h}, [%8] \n" // r6

                        "shll   v0.4s, v0.4h, #16           \n"
                        "shll   v1.4s, v1.4h, #16           \n"
                        "shll   v2.4s, v2.4h, #16           \n"

                        "fmla   v18.4s, v27.4s, v4.s[3]     \n"
                        "fmla   v19.4s, v27.4s, v5.s[1]     \n"
                        "fmla   v16.4s, v28.4s, v5.s[0]     \n"
                        "fmla   v17.4s, v28.4s, v5.s[2]     \n"

                        "prfm   pldl1keep, [%9, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%9], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v18.4s, v29.4s, v5.s[1]     \n"
                        "fmla   v19.4s, v29.4s, v5.s[3]     \n"
                        "fmla   v16.4s, v30.4s, v5.s[2]     \n"
                        "fmla   v17.4s, v30.4s, v6.s[0]     \n"

                        "prfm   pldl1keep, [%9, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%9], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

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

                        "add    %2, %2, #8                  \n"
                        "add    %3, %3, #8                  \n"
                        "add    %4, %4, #8                  \n"
                        "add    %5, %5, #8                  \n"
                        "add    %6, %6, #8                  \n"
                        "add    %7, %7, #8                  \n"
                        "add    %8, %8, #8                  \n"

                        "fadd   v16.4s, v16.4s, v18.4s      \n"
                        "fadd   v17.4s, v17.4s, v19.4s      \n"

                        "sub    %9, %9, #392                \n"

                        "shrn   v16.4h, v16.4s, #16         \n"
                        "shrn   v17.4h, v17.4s, #16         \n"

                        "st1    {v16.4h, v17.4h}, [%0], #16 \n"

                        : "=r"(outptr0_bf16), // %0
                        "=r"(outptr0),      // %1
                        "=r"(r0),           // %2
                        "=r"(r1),           // %3
                        "=r"(r2),           // %4
                        "=r"(r3),           // %5
                        "=r"(r4),           // %6
                        "=r"(r5),           // %7
                        "=r"(r6),           // %8
                        "=r"(kptr)          // %9
                        : "0"(outptr0_bf16),
                        "1"(outptr0),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "5"(r3),
                        "6"(r4),
                        "7"(r5),
                        "8"(r6),
                        "9"(kptr)
                        : "memory", "v0", "v1", "v2", "v4", "v5", "v6", "v16", "v17", "v18", "v19", "v24", "v25", "v26", "v27", "v28", "v29", "v30");
#else  // __aarch64__
                    asm volatile(
                        "pld        [%1, #256]          \n"
                        "vld1.f32   {d28-d31}, [%1 :128]! \n"

                        "pld        [%2, #128]          \n"
                        "vld1.u16   {d2-d3}, [%2]!      \n" // r0
                        "vld1.u16   {d8[0]}, [%2]       \n"

                        "vshll.u16  q0, d2, #16         \n"
                        "vshll.u16  q1, d3, #16         \n"
                        "vshl.u32   d8, d8, #16         \n"

                        "pld        [%9, #256]          \n"
                        "vld1.u16   {d14-d17}, [%9]!    \n"

                        "vshll.u16  q5, d14, #16        \n"
                        "vshll.u16  q6, d15, #16        \n"
                        "vshll.u16  q7, d16, #16        \n"
                        "vshll.u16  q8, d17, #16        \n"

                        "vmul.f32   q12, q5, d0[0]      \n"
                        "vmul.f32   q13, q5, d1[0]      \n"
                        "vmla.f32   q14, q6, d0[1]      \n"
                        "vmla.f32   q15, q6, d1[1]      \n"

                        "pld        [%9, #192]          \n"
                        "vld1.u16   {d20-d22}, [%9]!    \n"

                        "vshll.u16  q9, d20, #16        \n"
                        "vshll.u16  q10, d21, #16       \n"
                        "vshll.u16  q11, d22, #16       \n"

                        "vmla.f32   q12, q7, d1[0]      \n"
                        "vmla.f32   q13, q7, d2[0]      \n"

                        "pld        [%3, #128]          \n"
                        "vld1.u16   {d6-d7}, [%3]!      \n" // r1
                        "vld1.u16   {d9[0]}, [%3]       \n"

                        "vshll.u16  q2, d6, #16         \n"
                        "vshll.u16  q3, d7, #16         \n"
                        "vshl.u32   d9, d9, #16         \n"

                        "vmla.f32   q14, q8, d1[1]      \n"
                        "vmla.f32   q15, q8, d2[1]      \n"
                        "vmla.f32   q12, q9, d2[0]      \n"
                        "vmla.f32   q13, q9, d3[0]      \n"

                        "pld        [%9, #256]          \n"
                        "vld1.u16   {d14-d17}, [%9]!    \n"

                        "vshll.u16  q5, d14, #16        \n"
                        "vshll.u16  q6, d15, #16        \n"
                        "vshll.u16  q7, d16, #16        \n"
                        "vshll.u16  q8, d17, #16        \n"

                        "vmla.f32   q14, q10, d2[1]     \n"
                        "vmla.f32   q15, q10, d3[1]     \n"
                        "vmla.f32   q12, q11, d3[0]     \n"
                        "vmla.f32   q13, q11, d8[0]     \n"

                        "pld        [%9, #192]          \n"
                        "vld1.u16   {d20-d22}, [%9]!    \n"

                        "vshll.u16  q9, d20, #16        \n"
                        "vshll.u16  q10, d21, #16       \n"
                        "vshll.u16  q11, d22, #16       \n"

                        "vmla.f32   q14, q5, d4[0]      \n"
                        "vmla.f32   q15, q5, d5[0]      \n"
                        "vmla.f32   q12, q6, d4[1]      \n"
                        "vmla.f32   q13, q6, d5[1]      \n"
                        "vmla.f32   q14, q7, d5[0]      \n"
                        "vmla.f32   q15, q7, d6[0]      \n"

                        "pld        [%4, #128]          \n"
                        "vld1.u16   {d2-d3}, [%4]!      \n" // r2
                        "vld1.u16   {d8[0]}, [%4]       \n"

                        "vshll.u16  q0, d2, #16         \n"
                        "vshll.u16  q1, d3, #16         \n"
                        "vshl.u32   d8, d8, #16         \n"

                        "vmla.f32   q12, q8, d5[1]      \n"
                        "vmla.f32   q13, q8, d6[1]      \n"
                        "vmla.f32   q14, q9, d6[0]      \n"
                        "vmla.f32   q15, q9, d7[0]      \n"

                        "pld        [%9, #256]          \n"
                        "vld1.u16   {d14-d17}, [%9]!    \n"

                        "vshll.u16  q5, d14, #16        \n"
                        "vshll.u16  q6, d15, #16        \n"
                        "vshll.u16  q7, d16, #16        \n"
                        "vshll.u16  q8, d17, #16        \n"

                        "vmla.f32   q12, q10, d6[1]     \n"
                        "vmla.f32   q13, q10, d7[1]     \n"
                        "vmla.f32   q14, q11, d7[0]     \n"
                        "vmla.f32   q15, q11, d9[0]     \n"

                        "pld        [%9, #192]          \n"
                        "vld1.u16   {d20-d22}, [%9]!    \n"

                        "vshll.u16  q9, d20, #16        \n"
                        "vshll.u16  q10, d21, #16       \n"
                        "vshll.u16  q11, d22, #16       \n"

                        "vmla.f32   q12, q5, d0[0]      \n"
                        "vmla.f32   q13, q5, d1[0]      \n"
                        "vmla.f32   q14, q6, d0[1]      \n"
                        "vmla.f32   q15, q6, d1[1]      \n"
                        "vmla.f32   q12, q7, d1[0]      \n"
                        "vmla.f32   q13, q7, d2[0]      \n"

                        "pld        [%5, #128]          \n"
                        "vld1.u16   {d6-d7}, [%5]!      \n" // r3
                        "vld1.u16   {d9[0]}, [%5]       \n"

                        "vshll.u16  q2, d6, #16         \n"
                        "vshll.u16  q3, d7, #16         \n"
                        "vshl.u32   d9, d9, #16         \n"

                        "vmla.f32   q14, q8, d1[1]      \n"
                        "vmla.f32   q15, q8, d2[1]      \n"
                        "vmla.f32   q12, q9, d2[0]      \n"
                        "vmla.f32   q13, q9, d3[0]      \n"

                        "pld        [%9, #256]          \n"
                        "vld1.u16   {d14-d17}, [%9]!    \n"

                        "vshll.u16  q5, d14, #16        \n"
                        "vshll.u16  q6, d15, #16        \n"
                        "vshll.u16  q7, d16, #16        \n"
                        "vshll.u16  q8, d17, #16        \n"

                        "vmla.f32   q14, q10, d2[1]     \n"
                        "vmla.f32   q15, q10, d3[1]     \n"
                        "vmla.f32   q12, q11, d3[0]     \n"
                        "vmla.f32   q13, q11, d8[0]     \n"

                        "pld        [%9, #192]          \n"
                        "vld1.u16   {d20-d22}, [%9]!    \n"

                        "vshll.u16  q9, d20, #16        \n"
                        "vshll.u16  q10, d21, #16       \n"
                        "vshll.u16  q11, d22, #16       \n"

                        "vmla.f32   q14, q5, d4[0]      \n"
                        "vmla.f32   q15, q5, d5[0]      \n"
                        "vmla.f32   q12, q6, d4[1]      \n"
                        "vmla.f32   q13, q6, d5[1]      \n"
                        "vmla.f32   q14, q7, d5[0]      \n"
                        "vmla.f32   q15, q7, d6[0]      \n"

                        "pld        [%6, #128]          \n"
                        "vld1.u16   {d2-d3}, [%6]!      \n" // r4
                        "vld1.u16   {d8[0]}, [%6]       \n"

                        "vshll.u16  q0, d2, #16         \n"
                        "vshll.u16  q1, d3, #16         \n"
                        "vshl.u32   d8, d8, #16         \n"

                        "vmla.f32   q12, q8, d5[1]      \n"
                        "vmla.f32   q13, q8, d6[1]      \n"
                        "vmla.f32   q14, q9, d6[0]      \n"
                        "vmla.f32   q15, q9, d7[0]      \n"

                        "pld        [%9, #256]          \n"
                        "vld1.u16   {d14-d17}, [%9]!    \n"

                        "vshll.u16  q5, d14, #16        \n"
                        "vshll.u16  q6, d15, #16        \n"
                        "vshll.u16  q7, d16, #16        \n"
                        "vshll.u16  q8, d17, #16        \n"

                        "vmla.f32   q12, q10, d6[1]     \n"
                        "vmla.f32   q13, q10, d7[1]     \n"
                        "vmla.f32   q14, q11, d7[0]     \n"
                        "vmla.f32   q15, q11, d9[0]     \n"

                        "pld        [%9, #192]          \n"
                        "vld1.u16   {d20-d22}, [%9]!    \n"

                        "vshll.u16  q9, d20, #16        \n"
                        "vshll.u16  q10, d21, #16       \n"
                        "vshll.u16  q11, d22, #16       \n"

                        "vmla.f32   q12, q5, d0[0]      \n"
                        "vmla.f32   q13, q5, d1[0]      \n"
                        "vmla.f32   q14, q6, d0[1]      \n"
                        "vmla.f32   q15, q6, d1[1]      \n"
                        "vmla.f32   q12, q7, d1[0]      \n"
                        "vmla.f32   q13, q7, d2[0]      \n"

                        "pld        [%7, #128]          \n"
                        "vld1.u16   {d6-d7}, [%7]!      \n" // r5
                        "vld1.u16   {d9[0]}, [%7]       \n"

                        "vshll.u16  q2, d6, #16         \n"
                        "vshll.u16  q3, d7, #16         \n"
                        "vshl.u32   d9, d9, #16         \n"

                        "vmla.f32   q14, q8, d1[1]      \n"
                        "vmla.f32   q15, q8, d2[1]      \n"
                        "vmla.f32   q12, q9, d2[0]      \n"
                        "vmla.f32   q13, q9, d3[0]      \n"

                        "pld        [%9, #256]          \n"
                        "vld1.u16   {d14-d17}, [%9]!    \n"

                        "vshll.u16  q5, d14, #16        \n"
                        "vshll.u16  q6, d15, #16        \n"
                        "vshll.u16  q7, d16, #16        \n"
                        "vshll.u16  q8, d17, #16        \n"

                        "vmla.f32   q14, q10, d2[1]     \n"
                        "vmla.f32   q15, q10, d3[1]     \n"
                        "vmla.f32   q12, q11, d3[0]     \n"
                        "vmla.f32   q13, q11, d8[0]     \n"

                        "pld        [%9, #192]          \n"
                        "vld1.u16   {d20-d22}, [%9]!    \n"

                        "vshll.u16  q9, d20, #16        \n"
                        "vshll.u16  q10, d21, #16       \n"
                        "vshll.u16  q11, d22, #16       \n"

                        "vmla.f32   q14, q5, d4[0]      \n"
                        "vmla.f32   q15, q5, d5[0]      \n"
                        "vmla.f32   q12, q6, d4[1]      \n"
                        "vmla.f32   q13, q6, d5[1]      \n"
                        "vmla.f32   q14, q7, d5[0]      \n"
                        "vmla.f32   q15, q7, d6[0]      \n"

                        "pld        [%8, #128]          \n"
                        "vld1.u16   {d2-d3}, [%8]!      \n" // r6
                        "vld1.u16   {d8[0]}, [%8]       \n"

                        "vshll.u16  q0, d2, #16         \n"
                        "vshll.u16  q1, d3, #16         \n"
                        "vshl.u32   d8, d8, #16         \n"

                        "vmla.f32   q12, q8, d5[1]      \n"
                        "vmla.f32   q13, q8, d6[1]      \n"
                        "vmla.f32   q14, q9, d6[0]      \n"
                        "vmla.f32   q15, q9, d7[0]      \n"

                        "pld        [%9, #256]          \n"
                        "vld1.u16   {d14-d17}, [%9]!    \n"

                        "vshll.u16  q5, d14, #16        \n"
                        "vshll.u16  q6, d15, #16        \n"
                        "vshll.u16  q7, d16, #16        \n"
                        "vshll.u16  q8, d17, #16        \n"

                        "vmla.f32   q12, q10, d6[1]     \n"
                        "vmla.f32   q13, q10, d7[1]     \n"
                        "vmla.f32   q14, q11, d7[0]     \n"
                        "vmla.f32   q15, q11, d9[0]     \n"

                        "pld        [%9, #192]          \n"
                        "vld1.u16   {d20-d22}, [%9]!    \n"

                        "vshll.u16  q9, d20, #16        \n"
                        "vshll.u16  q10, d21, #16       \n"
                        "vshll.u16  q11, d22, #16       \n"

                        "vmla.f32   q12, q5, d0[0]      \n"
                        "vmla.f32   q13, q5, d1[0]      \n"
                        "vmla.f32   q14, q6, d0[1]      \n"
                        "vmla.f32   q15, q6, d1[1]      \n"

                        "sub        %2, %2, #8          \n"
                        "sub        %3, %3, #8          \n"

                        "vmla.f32   q12, q7, d1[0]      \n"
                        "vmla.f32   q13, q7, d2[0]      \n"
                        "vmla.f32   q14, q8, d1[1]      \n"
                        "vmla.f32   q15, q8, d2[1]      \n"

                        "sub        %9, %9, #392        \n"

                        "vmla.f32   q12, q9, d2[0]      \n"
                        "vmla.f32   q13, q9, d3[0]      \n"
                        "vmla.f32   q14, q10, d2[1]     \n"
                        "vmla.f32   q15, q10, d3[1]     \n"

                        "sub        %4, %4, #8          \n"
                        "sub        %5, %5, #8          \n"

                        "vmla.f32   q12, q11, d3[0]     \n"
                        "vmla.f32   q13, q11, d8[0]     \n"

                        "sub        %6, %6, #8          \n"
                        "sub        %7, %7, #8          \n"

                        "vadd.f32   q14, q14, q12       \n"
                        "vadd.f32   q15, q15, q13       \n"

                        "sub        %8, %8, #8          \n"

                        "vshrn.u32  d28, q14, #16       \n"
                        "vshrn.u32  d29, q15, #16       \n"

                        "vst1.u16   {d28-d29}, [%0 :64]! \n"

                        : "=r"(outptr0_bf16), // %0
                        "=r"(outptr0),      // %1
                        "=r"(r0),           // %2
                        "=r"(r1),           // %3
                        "=r"(r2),           // %4
                        "=r"(r3),           // %5
                        "=r"(r4),           // %6
                        "=r"(r5),           // %7
                        "=r"(r6),           // %8
                        "=r"(kptr)          // %9
                        : "0"(outptr0_bf16),
                        "1"(outptr0),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "5"(r3),
                        "6"(r4),
                        "7"(r5),
                        "8"(r6),
                        "9"(kptr)
                        : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif // __aarch64__
                }
                for (; j < outw; j++)
                {
#if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%1, #128]       \n"
                        "ld1    {v16.4s}, [%1], #16         \n"

                        "prfm   pldl1keep, [%2, #128]       \n"
                        "ld1    {v0.4h, v1.4h}, [%2]        \n" // r0

                        "shll   v0.4s, v0.4h, #16           \n"
                        "shll   v1.4s, v1.4h, #16           \n"

                        "prfm   pldl1keep, [%9, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%9], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmul   v17.4s, v24.4s, v0.s[0]     \n"

                        "prfm   pldl1keep, [%9, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%9], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

                        "fmul   v18.4s, v25.4s, v0.s[1]     \n"
                        "fmul   v19.4s, v26.4s, v0.s[2]     \n"

                        "prfm   pldl1keep, [%3, #128]       \n"
                        "ld1    {v4.4h, v5.4h}, [%3]        \n" // r1

                        "shll   v4.4s, v4.4h, #16           \n"
                        "shll   v5.4s, v5.4h, #16           \n"

                        "fmla   v16.4s, v27.4s, v0.s[3]     \n"
                        "fmla   v17.4s, v28.4s, v1.s[0]     \n"

                        "prfm   pldl1keep, [%9, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%9], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v18.4s, v29.4s, v1.s[1]     \n"
                        "fmla   v19.4s, v30.4s, v1.s[2]     \n"

                        "prfm   pldl1keep, [%9, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%9], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

                        "fmla   v16.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v17.4s, v25.4s, v4.s[1]     \n"
                        "fmla   v18.4s, v26.4s, v4.s[2]     \n"

                        "prfm   pldl1keep, [%4, #128]       \n"
                        "ld1    {v0.4h, v1.4h}, [%4]        \n" // r2

                        "shll   v0.4s, v0.4h, #16           \n"
                        "shll   v1.4s, v1.4h, #16           \n"

                        "fmla   v19.4s, v27.4s, v4.s[3]     \n"
                        "fmla   v16.4s, v28.4s, v5.s[0]     \n"

                        "prfm   pldl1keep, [%9, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%9], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v17.4s, v29.4s, v5.s[1]     \n"
                        "fmla   v18.4s, v30.4s, v5.s[2]     \n"

                        "prfm   pldl1keep, [%9, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%9], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

                        "fmla   v19.4s, v24.4s, v0.s[0]     \n"
                        "fmla   v16.4s, v25.4s, v0.s[1]     \n"
                        "fmla   v17.4s, v26.4s, v0.s[2]     \n"

                        "prfm   pldl1keep, [%5, #128]       \n"
                        "ld1    {v4.4h, v5.4h}, [%5]        \n" // r3

                        "shll   v4.4s, v4.4h, #16           \n"
                        "shll   v5.4s, v5.4h, #16           \n"

                        "fmla   v18.4s, v27.4s, v0.s[3]     \n"
                        "fmla   v19.4s, v28.4s, v1.s[0]     \n"

                        "prfm   pldl1keep, [%9, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%9], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v16.4s, v29.4s, v1.s[1]     \n"
                        "fmla   v17.4s, v30.4s, v1.s[2]     \n"

                        "prfm   pldl1keep, [%9, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%9], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

                        "fmla   v18.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v19.4s, v25.4s, v4.s[1]     \n"
                        "fmla   v16.4s, v26.4s, v4.s[2]     \n"

                        "prfm   pldl1keep, [%6, #128]       \n"
                        "ld1    {v0.4h, v1.4h}, [%6]        \n" // r4

                        "shll   v0.4s, v0.4h, #16           \n"
                        "shll   v1.4s, v1.4h, #16           \n"

                        "fmla   v17.4s, v27.4s, v4.s[3]     \n"
                        "fmla   v18.4s, v28.4s, v5.s[0]     \n"

                        "prfm   pldl1keep, [%9, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%9], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v19.4s, v29.4s, v5.s[1]     \n"
                        "fmla   v16.4s, v30.4s, v5.s[2]     \n"

                        "prfm   pldl1keep, [%9, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%9], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

                        "fmla   v17.4s, v24.4s, v0.s[0]     \n"
                        "fmla   v18.4s, v25.4s, v0.s[1]     \n"
                        "fmla   v19.4s, v26.4s, v0.s[2]     \n"

                        "prfm   pldl1keep, [%7, #128]       \n"
                        "ld1    {v4.4h, v5.4h}, [%7]        \n" // r5

                        "shll   v4.4s, v4.4h, #16           \n"
                        "shll   v5.4s, v5.4h, #16           \n"

                        "fmla   v16.4s, v27.4s, v0.s[3]     \n"
                        "fmla   v17.4s, v28.4s, v1.s[0]     \n"

                        "prfm   pldl1keep, [%9, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%9], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v18.4s, v29.4s, v1.s[1]     \n"
                        "fmla   v19.4s, v30.4s, v1.s[2]     \n"

                        "prfm   pldl1keep, [%9, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%9], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

                        "fmla   v16.4s, v24.4s, v4.s[0]     \n"
                        "fmla   v17.4s, v25.4s, v4.s[1]     \n"
                        "fmla   v18.4s, v26.4s, v4.s[2]     \n"

                        "prfm   pldl1keep, [%8, #128]       \n"
                        "ld1    {v0.4h, v1.4h}, [%8]        \n" // r6

                        "shll   v0.4s, v0.4h, #16           \n"
                        "shll   v1.4s, v1.4h, #16           \n"

                        "fmla   v19.4s, v27.4s, v4.s[3]     \n"
                        "fmla   v16.4s, v28.4s, v5.s[0]     \n"

                        "prfm   pldl1keep, [%9, #256]       \n"
                        "ld1    {v24.4h, v25.4h, v26.4h, v27.4h}, [%9], #32 \n"

                        "shll   v24.4s, v24.4h, #16         \n"
                        "shll   v25.4s, v25.4h, #16         \n"
                        "shll   v26.4s, v26.4h, #16         \n"
                        "shll   v27.4s, v27.4h, #16         \n"

                        "fmla   v17.4s, v29.4s, v5.s[1]     \n"
                        "fmla   v18.4s, v30.4s, v5.s[2]     \n"

                        "prfm   pldl1keep, [%9, #192]       \n"
                        "ld1    {v28.4h, v29.4h, v30.4h}, [%9], #24 \n"

                        "shll   v28.4s, v28.4h, #16         \n"
                        "shll   v29.4s, v29.4h, #16         \n"
                        "shll   v30.4s, v30.4h, #16         \n"

                        "fmla   v19.4s, v24.4s, v0.s[0]     \n"
                        "fmla   v16.4s, v25.4s, v0.s[1]     \n"
                        "fmla   v17.4s, v26.4s, v0.s[2]     \n"

                        "add    %2, %2, #4                  \n"
                        "add    %3, %3, #4                  \n"

                        "fmla   v18.4s, v27.4s, v0.s[3]     \n"
                        "fmla   v19.4s, v28.4s, v1.s[0]     \n"
                        "fmla   v16.4s, v29.4s, v1.s[1]     \n"
                        "fmla   v17.4s, v30.4s, v1.s[2]     \n"

                        "add    %4, %4, #4                  \n"
                        "add    %5, %5, #4                  \n"

                        "fadd   v18.4s, v18.4s, v19.4s      \n"

                        "add    %6, %6, #4                  \n"

                        "fadd   v16.4s, v16.4s, v17.4s      \n"

                        "add    %7, %7, #4                  \n"
                        "add    %8, %8, #4                  \n"

                        "fadd   v16.4s, v16.4s, v18.4s      \n"

                        "sub    %9, %9, #392                \n"

                        "shrn   v16.4h, v16.4s, #16         \n"

                        "st1    {v16.4h}, [%0], #8          \n"

                        : "=r"(outptr0_bf16), // %0
                        "=r"(outptr0),      // %1
                        "=r"(r0),           // %2
                        "=r"(r1),           // %3
                        "=r"(r2),           // %4
                        "=r"(r3),           // %5
                        "=r"(r4),           // %6
                        "=r"(r5),           // %7
                        "=r"(r6),           // %8
                        "=r"(kptr)          // %9
                        : "0"(outptr0_bf16),
                        "1"(outptr0),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "5"(r3),
                        "6"(r4),
                        "7"(r5),
                        "8"(r6),
                        "9"(kptr)
                        : "memory", "v0", "v1", "v4", "v5", "v16", "v17", "v18", "v19", "v24", "v25", "v26", "v27", "v28", "v29", "v30");
#else  // __aarch64__
                    asm volatile(
                        "pld        [%1, #128]          \n"
                        "vld1.f32   {d8-d9}, [%1 :128]! \n"

                        "pld        [%2, #128]          \n"
                        "vld1.u16   {d2-d3}, [%2]       \n" // r0

                        "vshll.u16  q0, d2, #16         \n"
                        "vshll.u16  q1, d3, #16         \n"

                        "pld        [%9, #256]          \n"
                        "vld1.u16   {d20-d23}, [%9]!    \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmul.f32   q5, q8, d0[0]       \n"
                        "vmul.f32   q6, q9, d0[1]       \n"

                        "pld        [%9, #192]          \n"
                        "vld1.u16   {d26-d28}, [%9]!    \n"

                        "vshll.u16  q12, d26, #16       \n"
                        "vshll.u16  q13, d27, #16       \n"
                        "vshll.u16  q14, d28, #16       \n"

                        "vmul.f32   q7, q10, d1[0]      \n"
                        "vmla.f32   q4, q11, d1[1]      \n"

                        "pld        [%3, #128]          \n"
                        "vld1.u16   {d6-d7}, [%3]       \n" // r1

                        "vshll.u16  q2, d6, #16         \n"
                        "vshll.u16  q3, d7, #16         \n"

                        "vmla.f32   q5, q12, d2[0]      \n"

                        "pld        [%9, #256]          \n"
                        "vld1.u16   {d20-d23}, [%9]!    \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q6, q13, d2[1]      \n"
                        "vmla.f32   q7, q14, d3[0]      \n"

                        "pld        [%9, #192]          \n"
                        "vld1.u16   {d26-d28}, [%9]!    \n"

                        "vshll.u16  q12, d26, #16       \n"
                        "vshll.u16  q13, d27, #16       \n"
                        "vshll.u16  q14, d28, #16       \n"

                        "vmla.f32   q4, q8, d4[0]       \n"
                        "vmla.f32   q5, q9, d4[1]       \n"
                        "vmla.f32   q6, q10, d5[0]      \n"

                        "pld        [%4, #128]          \n"
                        "vld1.u16   {d2-d3}, [%4]       \n" // r2

                        "vshll.u16  q0, d2, #16         \n"
                        "vshll.u16  q1, d3, #16         \n"

                        "vmla.f32   q7, q11, d5[1]      \n"
                        "vmla.f32   q4, q12, d6[0]      \n"

                        "pld        [%9, #256]          \n"
                        "vld1.u16   {d20-d23}, [%9]!    \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q5, q13, d6[1]      \n"
                        "vmla.f32   q6, q14, d7[0]      \n"

                        "pld        [%9, #192]          \n"
                        "vld1.u16   {d26-d28}, [%9]!    \n"

                        "vshll.u16  q12, d26, #16       \n"
                        "vshll.u16  q13, d27, #16       \n"
                        "vshll.u16  q14, d28, #16       \n"

                        "vmla.f32   q7, q8, d0[0]       \n"
                        "vmla.f32   q4, q9, d0[1]       \n"
                        "vmla.f32   q5, q10, d1[0]      \n"

                        "pld        [%5, #128]          \n"
                        "vld1.u16   {d6-d7}, [%5]       \n" // r3

                        "vshll.u16  q2, d6, #16         \n"
                        "vshll.u16  q3, d7, #16         \n"

                        "vmla.f32   q6, q11, d1[1]      \n"
                        "vmla.f32   q7, q12, d2[0]      \n"

                        "pld        [%9, #256]          \n"
                        "vld1.u16   {d20-d23}, [%9]!    \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q4, q13, d2[1]      \n"
                        "vmla.f32   q5, q14, d3[0]      \n"

                        "pld        [%9, #192]          \n"
                        "vld1.u16   {d26-d28}, [%9]!    \n"

                        "vshll.u16  q12, d26, #16       \n"
                        "vshll.u16  q13, d27, #16       \n"
                        "vshll.u16  q14, d28, #16       \n"

                        "vmla.f32   q6, q8, d4[0]       \n"
                        "vmla.f32   q7, q9, d4[1]       \n"
                        "vmla.f32   q4, q10, d5[0]      \n"

                        "pld        [%6, #128]          \n"
                        "vld1.u16   {d2-d3}, [%6]       \n" // r4

                        "vshll.u16  q0, d2, #16         \n"
                        "vshll.u16  q1, d3, #16         \n"

                        "vmla.f32   q5, q11, d5[1]      \n"
                        "vmla.f32   q6, q12, d6[0]      \n"

                        "pld        [%9, #256]          \n"
                        "vld1.u16   {d20-d23}, [%9]!    \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q7, q13, d6[1]      \n"
                        "vmla.f32   q4, q14, d7[0]      \n"

                        "pld        [%9, #192]          \n"
                        "vld1.u16   {d26-d28}, [%9]!    \n"

                        "vshll.u16  q12, d26, #16       \n"
                        "vshll.u16  q13, d27, #16       \n"
                        "vshll.u16  q14, d28, #16       \n"

                        "vmla.f32   q5, q8, d0[0]       \n"
                        "vmla.f32   q6, q9, d0[1]       \n"
                        "vmla.f32   q7, q10, d1[0]      \n"

                        "pld        [%7, #128]          \n"
                        "vld1.u16   {d6-d7}, [%7]       \n" // r5

                        "vshll.u16  q2, d6, #16         \n"
                        "vshll.u16  q3, d7, #16         \n"

                        "vmla.f32   q4, q11, d1[1]      \n"
                        "vmla.f32   q5, q12, d2[0]      \n"

                        "pld        [%9, #256]          \n"
                        "vld1.u16   {d20-d23}, [%9]!    \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q6, q13, d2[1]      \n"
                        "vmla.f32   q7, q14, d3[0]      \n"

                        "pld        [%9, #192]          \n"
                        "vld1.u16   {d26-d28}, [%9]!    \n"

                        "vshll.u16  q12, d26, #16       \n"
                        "vshll.u16  q13, d27, #16       \n"
                        "vshll.u16  q14, d28, #16       \n"

                        "vmla.f32   q4, q8, d4[0]       \n"
                        "vmla.f32   q5, q9, d4[1]       \n"
                        "vmla.f32   q6, q10, d5[0]      \n"

                        "pld        [%8, #128]          \n"
                        "vld1.u16   {d2-d3}, [%8]       \n" // r6

                        "vshll.u16  q0, d2, #16         \n"
                        "vshll.u16  q1, d3, #16         \n"

                        "vmla.f32   q7, q11, d5[1]      \n"
                        "vmla.f32   q4, q12, d6[0]      \n"

                        "pld        [%9, #256]          \n"
                        "vld1.u16   {d20-d23}, [%9]!    \n"

                        "vshll.u16  q8, d20, #16        \n"
                        "vshll.u16  q9, d21, #16        \n"
                        "vshll.u16  q10, d22, #16       \n"
                        "vshll.u16  q11, d23, #16       \n"

                        "vmla.f32   q5, q13, d6[1]      \n"
                        "vmla.f32   q6, q14, d7[0]      \n"

                        "pld        [%9, #192]          \n"
                        "vld1.u16   {d26-d28}, [%9]!    \n"

                        "vshll.u16  q12, d26, #16       \n"
                        "vshll.u16  q13, d27, #16       \n"
                        "vshll.u16  q14, d28, #16       \n"

                        "vmla.f32   q7, q8, d0[0]       \n"
                        "vmla.f32   q4, q9, d0[1]       \n"

                        "add        %2, %2, #4          \n"
                        "add        %3, %3, #4          \n"

                        "vmla.f32   q5, q10, d1[0]      \n"
                        "vmla.f32   q6, q11, d1[1]      \n"

                        "sub        %9, %9, #392        \n"

                        "vmla.f32   q7, q12, d2[0]      \n"
                        "vmla.f32   q4, q13, d2[1]      \n"
                        "vmla.f32   q5, q14, d3[0]      \n"

                        "add        %4, %4, #4          \n"
                        "add        %5, %5, #4          \n"

                        "vadd.f32   q6, q6, q7          \n"

                        "add        %6, %6, #4          \n"

                        "vadd.f32   q4, q4, q5          \n"

                        "add        %7, %7, #4          \n"

                        "vadd.f32   q4, q4, q6          \n"

                        "add        %8, %8, #4          \n"

                        "vshrn.u32  d8, q4, #16         \n"

                        "vst1.u16   {d8}, [%0 :64]!     \n"

                        : "=r"(outptr0_bf16), // %0
                        "=r"(outptr0),      // %1
                        "=r"(r0),           // %2
                        "=r"(r1),           // %3
                        "=r"(r2),           // %4
                        "=r"(r3),           // %5
                        "=r"(r4),           // %6
                        "=r"(r5),           // %7
                        "=r"(r6),           // %8
                        "=r"(kptr)          // %9
                        : "0"(outptr0_bf16),
                        "1"(outptr0),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "5"(r3),
                        "6"(r4),
                        "7"(r5),
                        "8"(r6),
                        "9"(kptr)
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
