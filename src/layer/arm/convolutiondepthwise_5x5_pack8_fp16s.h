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

static void convdw5x5s1_pack8_fp16sa_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int group = bottom_blob.c;

    const __fp16* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g = 0; g < group; g++)
    {
        Mat out = top_blob.channel(g);

        __fp16 bias0_data[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

        const __fp16* k0 = kernel.row<const __fp16>(g);

        __fp16* outptr0 = out.row<__fp16>(0);
        __fp16* outptr1 = out.row<__fp16>(1);

        const Mat img0 = bottom_blob.channel(g);

        const __fp16* r0 = img0.row<const __fp16>(0);
        const __fp16* r1 = img0.row<const __fp16>(1);
        const __fp16* r2 = img0.row<const __fp16>(2);
        const __fp16* r3 = img0.row<const __fp16>(3);
        const __fp16* r4 = img0.row<const __fp16>(4);
        const __fp16* r5 = img0.row<const __fp16>(5);

        int i = 0;
        for (; i + 1 < outh; i += 2)
        {
            int j = 0;
            for (; j + 3 < outw; j += 4)
            {
                const __fp16* bias0_data_ptr = bias ? bias + g * 8 : bias0_data;

                asm volatile(
                    "prfm   pldl1keep, [%18, #512]      \n"
                    "ld1    {v31.8h}, [%18]             \n" // sum13

                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%2], #64 \n" // r0_0123

                    "prfm   pldl1keep, [%8, #512]       \n"
                    "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%8], #64 \n" // w0_0123

                    "mov    v24.16b, v31.16b            \n" // sum00
                    "mov    v25.16b, v31.16b            \n" // sum01
                    "mov    v26.16b, v31.16b            \n" // sum02
                    "mov    v27.16b, v31.16b            \n" // sum03

                    "fmla   v24.8h, v16.8h, v0.8h       \n"
                    "fmla   v25.8h, v17.8h, v0.8h       \n"
                    "fmla   v26.8h, v18.8h, v0.8h       \n"
                    "fmla   v27.8h, v19.8h, v0.8h       \n"

                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%2] \n" // r0_4567

                    "fmla   v24.8h, v17.8h, v1.8h       \n"
                    "fmla   v25.8h, v18.8h, v1.8h       \n"
                    "fmla   v26.8h, v19.8h, v1.8h       \n"
                    "fmla   v27.8h, v20.8h, v1.8h       \n"

                    "mov    v28.16b, v31.16b            \n" // sum10

                    "fmla   v24.8h, v18.8h, v2.8h       \n"
                    "fmla   v25.8h, v19.8h, v2.8h       \n"
                    "fmla   v26.8h, v20.8h, v2.8h       \n"
                    "fmla   v27.8h, v21.8h, v2.8h       \n"

                    "prfm   pldl1keep, [%8, #512]       \n"
                    "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%8], #64 \n" // w04 w1_012

                    "fmla   v24.8h, v19.8h, v3.8h       \n"
                    "fmla   v25.8h, v20.8h, v3.8h       \n"
                    "fmla   v26.8h, v21.8h, v3.8h       \n"
                    "fmla   v27.8h, v22.8h, v3.8h       \n"

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v8.8h, v9.8h, v10.8h, v11.8h}, [%3], #64 \n" // r1_0123

                    "fmla   v24.8h, v20.8h, v4.8h       \n"
                    "fmla   v25.8h, v21.8h, v4.8h       \n"
                    "fmla   v26.8h, v22.8h, v4.8h       \n"
                    "fmla   v27.8h, v23.8h, v4.8h       \n"

                    "mov    v29.16b, v31.16b            \n" // sum11
                    "mov    v30.16b, v31.16b            \n" // sum12

                    "fmla   v28.8h, v8.8h, v0.8h        \n"
                    "fmla   v29.8h, v9.8h, v0.8h        \n"
                    "fmla   v30.8h, v10.8h, v0.8h       \n"
                    "fmla   v31.8h, v11.8h, v0.8h       \n"

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v12.8h, v13.8h, v14.8h, v15.8h}, [%3] \n" // r1_4567

                    "fmla   v28.8h, v9.8h, v1.8h        \n"
                    "fmla   v29.8h, v10.8h, v1.8h       \n"
                    "fmla   v30.8h, v11.8h, v1.8h       \n"
                    "fmla   v31.8h, v12.8h, v1.8h       \n"

                    "fmla   v28.8h, v10.8h, v2.8h       \n"
                    "fmla   v29.8h, v11.8h, v2.8h       \n"
                    "fmla   v30.8h, v12.8h, v2.8h       \n"
                    "fmla   v31.8h, v13.8h, v2.8h       \n"

                    "fmla   v28.8h, v11.8h, v3.8h       \n"
                    "fmla   v29.8h, v12.8h, v3.8h       \n"
                    "fmla   v30.8h, v13.8h, v3.8h       \n"
                    "fmla   v31.8h, v14.8h, v3.8h       \n"

                    "fmla   v28.8h, v12.8h, v4.8h       \n"
                    "fmla   v29.8h, v13.8h, v4.8h       \n"
                    "fmla   v30.8h, v14.8h, v4.8h       \n"
                    "fmla   v31.8h, v15.8h, v4.8h       \n"

                    "fmla   v24.8h, v8.8h, v5.8h        \n"
                    "fmla   v25.8h, v9.8h, v5.8h        \n"
                    "fmla   v26.8h, v10.8h, v5.8h       \n"
                    "fmla   v27.8h, v11.8h, v5.8h       \n"

                    "fmla   v24.8h, v9.8h, v6.8h        \n"
                    "fmla   v25.8h, v10.8h, v6.8h       \n"
                    "fmla   v26.8h, v11.8h, v6.8h       \n"
                    "fmla   v27.8h, v12.8h, v6.8h       \n"

                    "prfm   pldl1keep, [%8, #512]       \n"
                    "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%8], #64 \n" // w1_34 w2_01

                    "fmla   v24.8h, v10.8h, v7.8h       \n"
                    "fmla   v25.8h, v11.8h, v7.8h       \n"
                    "fmla   v26.8h, v12.8h, v7.8h       \n"
                    "fmla   v27.8h, v13.8h, v7.8h       \n"

                    "fmla   v24.8h, v11.8h, v0.8h       \n"
                    "fmla   v25.8h, v12.8h, v0.8h       \n"
                    "fmla   v26.8h, v13.8h, v0.8h       \n"
                    "fmla   v27.8h, v14.8h, v0.8h       \n"

                    "prfm   pldl1keep, [%4, #512]       \n"
                    "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%4], #64 \n" // r2_0123

                    "fmla   v24.8h, v12.8h, v1.8h       \n"
                    "fmla   v25.8h, v13.8h, v1.8h       \n"
                    "fmla   v26.8h, v14.8h, v1.8h       \n"
                    "fmla   v27.8h, v15.8h, v1.8h       \n"

                    "fmla   v28.8h, v16.8h, v5.8h       \n"
                    "fmla   v29.8h, v17.8h, v5.8h       \n"
                    "fmla   v30.8h, v18.8h, v5.8h       \n"
                    "fmla   v31.8h, v19.8h, v5.8h       \n"

                    "prfm   pldl1keep, [%4, #512]       \n"
                    "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4] \n" // r2_4567

                    "fmla   v28.8h, v17.8h, v6.8h       \n"
                    "fmla   v29.8h, v18.8h, v6.8h       \n"
                    "fmla   v30.8h, v19.8h, v6.8h       \n"
                    "fmla   v31.8h, v20.8h, v6.8h       \n"

                    "fmla   v28.8h, v18.8h, v7.8h       \n"
                    "fmla   v29.8h, v19.8h, v7.8h       \n"
                    "fmla   v30.8h, v20.8h, v7.8h       \n"
                    "fmla   v31.8h, v21.8h, v7.8h       \n"

                    "fmla   v28.8h, v19.8h, v0.8h       \n"
                    "fmla   v29.8h, v20.8h, v0.8h       \n"
                    "fmla   v30.8h, v21.8h, v0.8h       \n"
                    "fmla   v31.8h, v22.8h, v0.8h       \n"

                    "fmla   v28.8h, v20.8h, v1.8h       \n"
                    "fmla   v29.8h, v21.8h, v1.8h       \n"
                    "fmla   v30.8h, v22.8h, v1.8h       \n"
                    "fmla   v31.8h, v23.8h, v1.8h       \n"

                    "fmla   v24.8h, v16.8h, v2.8h       \n"
                    "fmla   v25.8h, v17.8h, v2.8h       \n"
                    "fmla   v26.8h, v18.8h, v2.8h       \n"
                    "fmla   v27.8h, v19.8h, v2.8h       \n"

                    "prfm   pldl1keep, [%8, #512]       \n"
                    "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%8], #64 \n" // w2_234 w30

                    "fmla   v24.8h, v17.8h, v3.8h       \n"
                    "fmla   v25.8h, v18.8h, v3.8h       \n"
                    "fmla   v26.8h, v19.8h, v3.8h       \n"
                    "fmla   v27.8h, v20.8h, v3.8h       \n"

                    "fmla   v24.8h, v18.8h, v4.8h       \n"
                    "fmla   v25.8h, v19.8h, v4.8h       \n"
                    "fmla   v26.8h, v20.8h, v4.8h       \n"
                    "fmla   v27.8h, v21.8h, v4.8h       \n"

                    "fmla   v24.8h, v19.8h, v5.8h       \n"
                    "fmla   v25.8h, v20.8h, v5.8h       \n"
                    "fmla   v26.8h, v21.8h, v5.8h       \n"
                    "fmla   v27.8h, v22.8h, v5.8h       \n"

                    "prfm   pldl1keep, [%5, #512]       \n"
                    "ld1    {v8.8h, v9.8h, v10.8h, v11.8h}, [%5], #64 \n" // r3_0123

                    "fmla   v24.8h, v20.8h, v6.8h       \n"
                    "fmla   v25.8h, v21.8h, v6.8h       \n"
                    "fmla   v26.8h, v22.8h, v6.8h       \n"
                    "fmla   v27.8h, v23.8h, v6.8h       \n"

                    "fmla   v28.8h, v8.8h, v2.8h        \n"
                    "fmla   v29.8h, v9.8h, v2.8h        \n"
                    "fmla   v30.8h, v10.8h, v2.8h       \n"
                    "fmla   v31.8h, v11.8h, v2.8h       \n"

                    "prfm   pldl1keep, [%5, #512]       \n"
                    "ld1    {v12.8h, v13.8h, v14.8h, v15.8h}, [%5] \n" // r3_4567

                    "fmla   v28.8h, v9.8h, v3.8h        \n"
                    "fmla   v29.8h, v10.8h, v3.8h       \n"
                    "fmla   v30.8h, v11.8h, v3.8h       \n"
                    "fmla   v31.8h, v12.8h, v3.8h       \n"

                    "fmla   v28.8h, v10.8h, v4.8h       \n"
                    "fmla   v29.8h, v11.8h, v4.8h       \n"
                    "fmla   v30.8h, v12.8h, v4.8h       \n"
                    "fmla   v31.8h, v13.8h, v4.8h       \n"

                    "fmla   v28.8h, v11.8h, v5.8h       \n"
                    "fmla   v29.8h, v12.8h, v5.8h       \n"
                    "fmla   v30.8h, v13.8h, v5.8h       \n"
                    "fmla   v31.8h, v14.8h, v5.8h       \n"

                    "fmla   v28.8h, v12.8h, v6.8h       \n"
                    "fmla   v29.8h, v13.8h, v6.8h       \n"
                    "fmla   v30.8h, v14.8h, v6.8h       \n"
                    "fmla   v31.8h, v15.8h, v6.8h       \n"

                    "prfm   pldl1keep, [%8, #512]       \n"
                    "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%8], #64 \n" // w3_1234

                    "fmla   v24.8h, v8.8h, v7.8h        \n"
                    "fmla   v25.8h, v9.8h, v7.8h        \n"
                    "fmla   v26.8h, v10.8h, v7.8h       \n"
                    "fmla   v27.8h, v11.8h, v7.8h       \n"

                    "fmla   v24.8h, v9.8h, v0.8h        \n"
                    "fmla   v25.8h, v10.8h, v0.8h       \n"
                    "fmla   v26.8h, v11.8h, v0.8h       \n"
                    "fmla   v27.8h, v12.8h, v0.8h       \n"

                    "fmla   v24.8h, v10.8h, v1.8h       \n"
                    "fmla   v25.8h, v11.8h, v1.8h       \n"
                    "fmla   v26.8h, v12.8h, v1.8h       \n"
                    "fmla   v27.8h, v13.8h, v1.8h       \n"

                    "fmla   v24.8h, v11.8h, v2.8h       \n"
                    "fmla   v25.8h, v12.8h, v2.8h       \n"
                    "fmla   v26.8h, v13.8h, v2.8h       \n"
                    "fmla   v27.8h, v14.8h, v2.8h       \n"

                    "prfm   pldl1keep, [%6, #512]       \n"
                    "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%6], #64 \n" // r4_0123

                    "fmla   v24.8h, v12.8h, v3.8h       \n"
                    "fmla   v25.8h, v13.8h, v3.8h       \n"
                    "fmla   v26.8h, v14.8h, v3.8h       \n"
                    "fmla   v27.8h, v15.8h, v3.8h       \n"

                    "fmla   v28.8h, v16.8h, v7.8h       \n"
                    "fmla   v29.8h, v17.8h, v7.8h       \n"
                    "fmla   v30.8h, v18.8h, v7.8h       \n"
                    "fmla   v31.8h, v19.8h, v7.8h       \n"

                    "prfm   pldl1keep, [%6, #512]       \n"
                    "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%6] \n" // r4_4567

                    "fmla   v28.8h, v17.8h, v0.8h       \n"
                    "fmla   v29.8h, v18.8h, v0.8h       \n"
                    "fmla   v30.8h, v19.8h, v0.8h       \n"
                    "fmla   v31.8h, v20.8h, v0.8h       \n"

                    "fmla   v28.8h, v18.8h, v1.8h       \n"
                    "fmla   v29.8h, v19.8h, v1.8h       \n"
                    "fmla   v30.8h, v20.8h, v1.8h       \n"
                    "fmla   v31.8h, v21.8h, v1.8h       \n"

                    "fmla   v28.8h, v19.8h, v2.8h       \n"
                    "fmla   v29.8h, v20.8h, v2.8h       \n"
                    "fmla   v30.8h, v21.8h, v2.8h       \n"
                    "fmla   v31.8h, v22.8h, v2.8h       \n"

                    "prfm   pldl1keep, [%8, #512]       \n"
                    "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%8], #64 \n" // w4_0123

                    "fmla   v28.8h, v20.8h, v3.8h       \n"
                    "fmla   v29.8h, v21.8h, v3.8h       \n"
                    "fmla   v30.8h, v22.8h, v3.8h       \n"
                    "fmla   v31.8h, v23.8h, v3.8h       \n"

                    "fmla   v24.8h, v16.8h, v4.8h       \n"
                    "fmla   v25.8h, v17.8h, v4.8h       \n"
                    "fmla   v26.8h, v18.8h, v4.8h       \n"
                    "fmla   v27.8h, v19.8h, v4.8h       \n"

                    "fmla   v24.8h, v17.8h, v5.8h       \n"
                    "fmla   v25.8h, v18.8h, v5.8h       \n"
                    "fmla   v26.8h, v19.8h, v5.8h       \n"
                    "fmla   v27.8h, v20.8h, v5.8h       \n"

                    "fmla   v24.8h, v18.8h, v6.8h       \n"
                    "fmla   v25.8h, v19.8h, v6.8h       \n"
                    "fmla   v26.8h, v20.8h, v6.8h       \n"
                    "fmla   v27.8h, v21.8h, v6.8h       \n"

                    "prfm   pldl1keep, [%8, #128]       \n"
                    "ld1    {v0.8h}, [%8]               \n" // w44

                    "fmla   v24.8h, v19.8h, v7.8h       \n"
                    "fmla   v25.8h, v20.8h, v7.8h       \n"
                    "fmla   v26.8h, v21.8h, v7.8h       \n"
                    "fmla   v27.8h, v22.8h, v7.8h       \n"

                    "prfm   pldl1keep, [%7, #512]       \n"
                    "ld1    {v8.8h, v9.8h, v10.8h, v11.8h}, [%7], #64 \n" // r5_0123

                    "fmla   v24.8h, v20.8h, v0.8h       \n"
                    "fmla   v25.8h, v21.8h, v0.8h       \n"
                    "fmla   v26.8h, v22.8h, v0.8h       \n"
                    "fmla   v27.8h, v23.8h, v0.8h       \n"

                    "fmla   v28.8h, v8.8h, v4.8h        \n"
                    "fmla   v29.8h, v9.8h, v4.8h        \n"
                    "fmla   v30.8h, v10.8h, v4.8h       \n"
                    "fmla   v31.8h, v11.8h, v4.8h       \n"

                    "prfm   pldl1keep, [%7, #512]       \n"
                    "ld1    {v12.8h, v13.8h, v14.8h, v15.8h}, [%7] \n" // r5_4567

                    "fmla   v28.8h, v9.8h, v5.8h        \n"
                    "fmla   v29.8h, v10.8h, v5.8h       \n"
                    "fmla   v30.8h, v11.8h, v5.8h       \n"
                    "fmla   v31.8h, v12.8h, v5.8h       \n"

                    "fmla   v28.8h, v10.8h, v6.8h       \n"
                    "fmla   v29.8h, v11.8h, v6.8h       \n"
                    "fmla   v30.8h, v12.8h, v6.8h       \n"
                    "fmla   v31.8h, v13.8h, v6.8h       \n"

                    "fmla   v28.8h, v11.8h, v7.8h       \n"
                    "fmla   v29.8h, v12.8h, v7.8h       \n"
                    "fmla   v30.8h, v13.8h, v7.8h       \n"
                    "fmla   v31.8h, v14.8h, v7.8h       \n"

                    "fmla   v28.8h, v12.8h, v0.8h       \n"
                    "fmla   v29.8h, v13.8h, v0.8h       \n"
                    "fmla   v30.8h, v14.8h, v0.8h       \n"
                    "fmla   v31.8h, v15.8h, v0.8h       \n"

                    "sub    %8, %8, #384                \n" // k0 -= 24 * 8

                    "st1    {v24.8h, v25.8h, v26.8h, v27.8h}, [%0], #64 \n"
                    "st1    {v28.8h, v29.8h, v30.8h, v31.8h}, [%1], #64 \n"

                    : "=r"(outptr0), // %0
                    "=r"(outptr1), // %1
                    "=r"(r0),      // %2
                    "=r"(r1),      // %3
                    "=r"(r2),      // %4
                    "=r"(r3),      // %5
                    "=r"(r4),      // %6
                    "=r"(r5),      // %7
                    "=r"(k0)       // %8
                    : "0"(outptr0),
                    "1"(outptr1),
                    "2"(r0),
                    "3"(r1),
                    "4"(r2),
                    "5"(r3),
                    "6"(r4),
                    "7"(r5),
                    "8"(k0),
                    "r"(bias0_data_ptr) // %18
                    : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
            }

            float16x8_t _bias0 = bias ? vld1q_f16(bias + g * 8) : vdupq_n_f16((__fp16)0.f);

            for (; j + 1 < outw; j += 2)
            {
                asm volatile(
                    "prfm   pldl1keep, [%2, #256]       \n"
                    "ld1    {v16.8h, v17.8h}, [%2], #32 \n" // r0_01

                    "prfm   pldl1keep, [%8, #512]       \n"
                    "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%8], #64 \n" // w0_0123

                    "mov    v28.16b, %18.16b            \n" // sum00
                    "mov    v29.16b, %18.16b            \n" // sum01

                    "fmla   v28.8h, v16.8h, v0.8h       \n"
                    "fmla   v29.8h, v17.8h, v0.8h       \n"

                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v18.8h, v19.8h, v20.8h, v21.8h}, [%2] \n" // r0_2345

                    "mov    v30.16b, %18.16b            \n" // sum10
                    "mov    v31.16b, %18.16b            \n" // sum11

                    "fmla   v28.8h, v17.8h, v1.8h       \n"
                    "fmla   v29.8h, v18.8h, v1.8h       \n"

                    "prfm   pldl1keep, [%8, #512]       \n"
                    "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%8], #64 \n" // w04 w1_012

                    "fmla   v28.8h, v18.8h, v2.8h       \n"
                    "fmla   v29.8h, v19.8h, v2.8h       \n"
                    "fmla   v28.8h, v19.8h, v3.8h       \n"
                    "fmla   v29.8h, v20.8h, v3.8h       \n"

                    "prfm   pldl1keep, [%3, #256]       \n"
                    "ld1    {v22.8h, v23.8h}, [%3], #32 \n" // r1_01

                    "fmla   v28.8h, v20.8h, v4.8h       \n"
                    "fmla   v29.8h, v21.8h, v4.8h       \n"

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v24.8h, v25.8h, v26.8h, v27.8h}, [%3] \n" // r1_2345

                    "fmla   v30.8h, v22.8h, v0.8h       \n"
                    "fmla   v31.8h, v23.8h, v0.8h       \n"
                    "fmla   v30.8h, v23.8h, v1.8h       \n"
                    "fmla   v31.8h, v24.8h, v1.8h       \n"
                    "fmla   v30.8h, v24.8h, v2.8h       \n"
                    "fmla   v31.8h, v25.8h, v2.8h       \n"
                    "fmla   v30.8h, v25.8h, v3.8h       \n"
                    "fmla   v31.8h, v26.8h, v3.8h       \n"
                    "fmla   v30.8h, v26.8h, v4.8h       \n"
                    "fmla   v31.8h, v27.8h, v4.8h       \n"

                    "prfm   pldl1keep, [%8, #512]       \n"
                    "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%8], #64 \n" // w1_34 w2_01

                    "fmla   v28.8h, v22.8h, v5.8h       \n"
                    "fmla   v29.8h, v23.8h, v5.8h       \n"
                    "fmla   v28.8h, v23.8h, v6.8h       \n"
                    "fmla   v29.8h, v24.8h, v6.8h       \n"
                    "fmla   v28.8h, v24.8h, v7.8h       \n"
                    "fmla   v29.8h, v25.8h, v7.8h       \n"
                    "fmla   v28.8h, v25.8h, v0.8h       \n"
                    "fmla   v29.8h, v26.8h, v0.8h       \n"

                    "prfm   pldl1keep, [%4, #256]       \n"
                    "ld1    {v16.8h, v17.8h}, [%4], #32 \n" // r2_01

                    "fmla   v28.8h, v26.8h, v1.8h       \n"
                    "fmla   v29.8h, v27.8h, v1.8h       \n"

                    "prfm   pldl1keep, [%4, #512]       \n"
                    "ld1    {v18.8h, v19.8h, v20.8h, v21.8h}, [%4] \n" // r2_2345

                    "fmla   v30.8h, v16.8h, v5.8h       \n"
                    "fmla   v31.8h, v17.8h, v5.8h       \n"
                    "fmla   v30.8h, v17.8h, v6.8h       \n"
                    "fmla   v31.8h, v18.8h, v6.8h       \n"
                    "fmla   v30.8h, v18.8h, v7.8h       \n"
                    "fmla   v31.8h, v19.8h, v7.8h       \n"
                    "fmla   v30.8h, v19.8h, v0.8h       \n"
                    "fmla   v31.8h, v20.8h, v0.8h       \n"
                    "fmla   v30.8h, v20.8h, v1.8h       \n"
                    "fmla   v31.8h, v21.8h, v1.8h       \n"

                    "prfm   pldl1keep, [%8, #512]       \n"
                    "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%8], #64 \n" // w2_234 w30

                    "fmla   v28.8h, v16.8h, v2.8h       \n"
                    "fmla   v29.8h, v17.8h, v2.8h       \n"
                    "fmla   v28.8h, v17.8h, v3.8h       \n"
                    "fmla   v29.8h, v18.8h, v3.8h       \n"
                    "fmla   v28.8h, v18.8h, v4.8h       \n"
                    "fmla   v29.8h, v19.8h, v4.8h       \n"
                    "fmla   v28.8h, v19.8h, v5.8h       \n"
                    "fmla   v29.8h, v20.8h, v5.8h       \n"

                    "prfm   pldl1keep, [%5, #256]       \n"
                    "ld1    {v22.8h, v23.8h}, [%5], #32 \n" // r3_01

                    "fmla   v28.8h, v20.8h, v6.8h       \n"
                    "fmla   v29.8h, v21.8h, v6.8h       \n"

                    "prfm   pldl1keep, [%5, #512]       \n"
                    "ld1    {v24.8h, v25.8h, v26.8h, v27.8h}, [%5] \n" // r3_2345

                    "fmla   v30.8h, v22.8h, v2.8h       \n"
                    "fmla   v31.8h, v23.8h, v2.8h       \n"
                    "fmla   v30.8h, v23.8h, v3.8h       \n"
                    "fmla   v31.8h, v24.8h, v3.8h       \n"
                    "fmla   v30.8h, v24.8h, v4.8h       \n"
                    "fmla   v31.8h, v25.8h, v4.8h       \n"
                    "fmla   v30.8h, v25.8h, v5.8h       \n"
                    "fmla   v31.8h, v26.8h, v5.8h       \n"

                    "prfm   pldl1keep, [%8, #512]       \n"
                    "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%8], #64 \n" // w3_1234

                    "fmla   v30.8h, v26.8h, v6.8h       \n"
                    "fmla   v31.8h, v27.8h, v6.8h       \n"

                    "fmla   v28.8h, v22.8h, v7.8h       \n"
                    "fmla   v29.8h, v23.8h, v7.8h       \n"
                    "fmla   v28.8h, v23.8h, v0.8h       \n"
                    "fmla   v29.8h, v24.8h, v0.8h       \n"
                    "fmla   v28.8h, v24.8h, v1.8h       \n"
                    "fmla   v29.8h, v25.8h, v1.8h       \n"
                    "fmla   v28.8h, v25.8h, v2.8h       \n"
                    "fmla   v29.8h, v26.8h, v2.8h       \n"

                    "prfm   pldl1keep, [%6, #256]       \n"
                    "ld1    {v16.8h, v17.8h}, [%6], #32 \n" // r4_01

                    "fmla   v28.8h, v26.8h, v3.8h       \n"
                    "fmla   v29.8h, v27.8h, v3.8h       \n"

                    "prfm   pldl1keep, [%6, #512]       \n"
                    "ld1    {v18.8h, v19.8h, v20.8h, v21.8h}, [%6] \n" // r4_2345

                    "fmla   v30.8h, v16.8h, v7.8h       \n"
                    "fmla   v31.8h, v17.8h, v7.8h       \n"
                    "fmla   v30.8h, v17.8h, v0.8h       \n"
                    "fmla   v31.8h, v18.8h, v0.8h       \n"
                    "fmla   v30.8h, v18.8h, v1.8h       \n"
                    "fmla   v31.8h, v19.8h, v1.8h       \n"

                    "prfm   pldl1keep, [%8, #512]       \n"
                    "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%8], #64 \n" // w4_0123

                    "fmla   v30.8h, v19.8h, v2.8h       \n"
                    "fmla   v31.8h, v20.8h, v2.8h       \n"
                    "fmla   v30.8h, v20.8h, v3.8h       \n"
                    "fmla   v31.8h, v21.8h, v3.8h       \n"

                    "fmla   v28.8h, v16.8h, v4.8h       \n"
                    "fmla   v29.8h, v17.8h, v4.8h       \n"
                    "fmla   v28.8h, v17.8h, v5.8h       \n"
                    "fmla   v29.8h, v18.8h, v5.8h       \n"

                    "prfm   pldl1keep, [%8, #128]       \n"
                    "ld1    {v0.8h}, [%8]               \n" // w44

                    "fmla   v28.8h, v18.8h, v6.8h       \n"
                    "fmla   v29.8h, v19.8h, v6.8h       \n"
                    "fmla   v28.8h, v19.8h, v7.8h       \n"
                    "fmla   v29.8h, v20.8h, v7.8h       \n"

                    "prfm   pldl1keep, [%7, #256]       \n"
                    "ld1    {v22.8h, v23.8h}, [%7], #32 \n" // r5_01

                    "fmla   v28.8h, v20.8h, v0.8h       \n"
                    "fmla   v29.8h, v21.8h, v0.8h       \n"

                    "prfm   pldl1keep, [%7, #512]       \n"
                    "ld1    {v24.8h, v25.8h, v26.8h, v27.8h}, [%7] \n" // r5_2345

                    "fmla   v30.8h, v22.8h, v4.8h       \n"
                    "fmla   v31.8h, v23.8h, v4.8h       \n"
                    "fmla   v30.8h, v23.8h, v5.8h       \n"
                    "fmla   v31.8h, v24.8h, v5.8h       \n"
                    "fmla   v30.8h, v24.8h, v6.8h       \n"
                    "fmla   v31.8h, v25.8h, v6.8h       \n"
                    "fmla   v30.8h, v25.8h, v7.8h       \n"
                    "fmla   v31.8h, v26.8h, v7.8h       \n"
                    "fmla   v30.8h, v26.8h, v0.8h       \n"
                    "fmla   v31.8h, v27.8h, v0.8h       \n"

                    "sub    %8, %8, #384                \n" // k0 -= 24 * 8

                    "st1    {v28.8h, v29.8h}, [%0], #32 \n"
                    "st1    {v30.8h, v31.8h}, [%1], #32 \n"

                    : "=r"(outptr0), // %0
                    "=r"(outptr1), // %1
                    "=r"(r0),      // %2
                    "=r"(r1),      // %3
                    "=r"(r2),      // %4
                    "=r"(r3),      // %5
                    "=r"(r4),      // %6
                    "=r"(r5),      // %7
                    "=r"(k0)       // %8
                    : "0"(outptr0),
                    "1"(outptr1),
                    "2"(r0),
                    "3"(r1),
                    "4"(r2),
                    "5"(r3),
                    "6"(r4),
                    "7"(r5),
                    "8"(k0),
                    "w"(_bias0) // %18
                    : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
            }
            for (; j < outw; j++)
            {
                asm volatile(
                    "prfm   pldl1keep, [%2, #128]       \n"
                    "ld1    {v16.8h}, [%2], #16         \n" // r0_0

                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v17.8h, v18.8h, v19.8h, v20.8h}, [%2] \n" // r0_1234

                    "prfm   pldl1keep, [%8, #512]       \n"
                    "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%8], #64 \n" // w0_0123

                    "mov    v30.16b, %18.16b            \n" // sum00
                    "mov    v31.16b, %18.16b            \n" // sum10

                    "fmla   v30.8h, v16.8h, v0.8h       \n"
                    "fmla   v30.8h, v17.8h, v1.8h       \n"

                    "prfm   pldl1keep, [%8, #512]       \n"
                    "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%8], #64 \n" // w04 w1_012

                    "fmla   v30.8h, v18.8h, v2.8h       \n"

                    "prfm   pldl1keep, [%3, #128]       \n"
                    "ld1    {v21.8h}, [%3], #16         \n" // r1_0

                    "fmla   v30.8h, v19.8h, v3.8h       \n"
                    "fmla   v30.8h, v20.8h, v4.8h       \n"

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v22.8h, v23.8h, v24.8h, v25.8h}, [%3] \n" // r1_1234

                    "fmla   v31.8h, v21.8h, v0.8h       \n"
                    "fmla   v31.8h, v22.8h, v1.8h       \n"
                    "fmla   v31.8h, v23.8h, v2.8h       \n"
                    "fmla   v31.8h, v24.8h, v3.8h       \n"
                    "fmla   v31.8h, v25.8h, v4.8h       \n"

                    "prfm   pldl1keep, [%8, #512]       \n"
                    "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%8], #64 \n" // w1_34 w2_01

                    "fmla   v30.8h, v21.8h, v5.8h       \n"
                    "fmla   v30.8h, v22.8h, v6.8h       \n"
                    "fmla   v30.8h, v23.8h, v7.8h       \n"

                    "prfm   pldl1keep, [%4, #128]       \n"
                    "ld1    {v16.8h}, [%4], #16         \n" // r2_0

                    "fmla   v30.8h, v24.8h, v0.8h       \n"
                    "fmla   v30.8h, v25.8h, v1.8h       \n"

                    "prfm   pldl1keep, [%4, #512]       \n"
                    "ld1    {v17.8h, v18.8h, v19.8h, v20.8h}, [%4] \n" // r2_1234

                    "fmla   v31.8h, v16.8h, v5.8h       \n"
                    "fmla   v31.8h, v17.8h, v6.8h       \n"
                    "fmla   v31.8h, v18.8h, v7.8h       \n"
                    "fmla   v31.8h, v19.8h, v0.8h       \n"
                    "fmla   v31.8h, v20.8h, v1.8h       \n"

                    "prfm   pldl1keep, [%8, #512]       \n"
                    "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%8], #64 \n" // w2_234 w30

                    "fmla   v30.8h, v16.8h, v2.8h       \n"
                    "fmla   v30.8h, v17.8h, v3.8h       \n"
                    "fmla   v30.8h, v18.8h, v4.8h       \n"

                    "prfm   pldl1keep, [%5, #128]       \n"
                    "ld1    {v21.8h}, [%5], #16         \n" // r3_0

                    "fmla   v30.8h, v19.8h, v5.8h       \n"
                    "fmla   v30.8h, v20.8h, v6.8h       \n"

                    "prfm   pldl1keep, [%5, #512]       \n"
                    "ld1    {v22.8h, v23.8h, v24.8h, v25.8h}, [%5] \n" // r3_1234

                    "fmla   v31.8h, v21.8h, v2.8h       \n"
                    "fmla   v31.8h, v22.8h, v3.8h       \n"
                    "fmla   v31.8h, v23.8h, v4.8h       \n"

                    "prfm   pldl1keep, [%8, #512]       \n"
                    "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%8], #64 \n" // w3_1234

                    "fmla   v31.8h, v24.8h, v5.8h       \n"
                    "fmla   v31.8h, v25.8h, v6.8h       \n"

                    "fmla   v30.8h, v21.8h, v7.8h       \n"
                    "fmla   v30.8h, v22.8h, v0.8h       \n"
                    "fmla   v30.8h, v23.8h, v1.8h       \n"

                    "prfm   pldl1keep, [%6, #128]       \n"
                    "ld1    {v16.8h}, [%6], #16         \n" // r4_0

                    "fmla   v30.8h, v24.8h, v2.8h       \n"
                    "fmla   v30.8h, v25.8h, v3.8h       \n"

                    "prfm   pldl1keep, [%6, #512]       \n"
                    "ld1    {v17.8h, v18.8h, v19.8h, v20.8h}, [%6] \n" // r4_1234

                    "fmla   v31.8h, v16.8h, v7.8h       \n"
                    "fmla   v31.8h, v17.8h, v0.8h       \n"
                    "fmla   v31.8h, v18.8h, v1.8h       \n"

                    "prfm   pldl1keep, [%8, #512]       \n"
                    "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%8], #64 \n" // w4_0123

                    "fmla   v31.8h, v19.8h, v2.8h       \n"
                    "fmla   v31.8h, v20.8h, v3.8h       \n"

                    "fmla   v30.8h, v16.8h, v4.8h       \n"
                    "fmla   v30.8h, v17.8h, v5.8h       \n"

                    "prfm   pldl1keep, [%8, #128]       \n"
                    "ld1    {v0.8h}, [%8]               \n" // w44

                    "fmla   v30.8h, v18.8h, v6.8h       \n"

                    "prfm   pldl1keep, [%7, #128]       \n"
                    "ld1    {v21.8h}, [%7], #16         \n" // r5_0

                    "fmla   v30.8h, v19.8h, v7.8h       \n"
                    "fmla   v30.8h, v20.8h, v0.8h       \n"

                    "prfm   pldl1keep, [%7, #512]       \n"
                    "ld1    {v22.8h, v23.8h, v24.8h, v25.8h}, [%7] \n" // r5_1234

                    "fmla   v31.8h, v21.8h, v4.8h       \n"
                    "fmla   v31.8h, v22.8h, v5.8h       \n"
                    "fmla   v31.8h, v23.8h, v6.8h       \n"
                    "fmla   v31.8h, v24.8h, v7.8h       \n"
                    "fmla   v31.8h, v25.8h, v0.8h       \n"

                    "sub    %8, %8, #384                \n" // k0 -= 24 * 8

                    "st1    {v30.8h}, [%0], #16         \n"
                    "st1    {v31.8h}, [%1], #16         \n"

                    : "=r"(outptr0), // %0
                    "=r"(outptr1), // %1
                    "=r"(r0),      // %2
                    "=r"(r1),      // %3
                    "=r"(r2),      // %4
                    "=r"(r3),      // %5
                    "=r"(r4),      // %6
                    "=r"(r5),      // %7
                    "=r"(k0)       // %8
                    : "0"(outptr0),
                    "1"(outptr1),
                    "2"(r0),
                    "3"(r1),
                    "4"(r2),
                    "5"(r3),
                    "6"(r4),
                    "7"(r5),
                    "8"(k0),
                    "w"(_bias0) // %18
                    : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v30", "v31");
            }

            r0 += 4 * 8 + w * 8;
            r1 += 4 * 8 + w * 8;
            r2 += 4 * 8 + w * 8;
            r3 += 4 * 8 + w * 8;
            r4 += 4 * 8 + w * 8;
            r5 += 4 * 8 + w * 8;

            outptr0 += outw * 8;
            outptr1 += outw * 8;
        }

        float16x8_t _bias0 = bias ? vld1q_f16(bias + g * 8) : vdupq_n_f16((__fp16)0.f);

        for (; i < outh; i++)
        {
            int j = 0;
            for (; j + 3 < outw; j += 4)
            {
                asm volatile(
                    "prfm   pldl1keep, [%1, #512]       \n"
                    "ld1    {v12.8h, v13.8h, v14.8h, v15.8h}, [%1], #64 \n" // r0_0123

                    "prfm   pldl1keep, [%6, #512]       \n"
                    "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%6], #64 \n" // w0_0123

                    "mov    v28.16b, %14.16b            \n" // sum00
                    "mov    v29.16b, %14.16b            \n" // sum01
                    "mov    v30.16b, %14.16b            \n" // sum02
                    "mov    v31.16b, %14.16b            \n" // sum03

                    "fmla   v28.8h, v12.8h, v0.8h       \n"
                    "fmla   v29.8h, v13.8h, v0.8h       \n"
                    "fmla   v30.8h, v14.8h, v0.8h       \n"
                    "fmla   v31.8h, v15.8h, v0.8h       \n"

                    "prfm   pldl1keep, [%1, #512]       \n"
                    "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%1] \n" // r0_4567

                    "fmla   v28.8h, v13.8h, v1.8h       \n"
                    "fmla   v29.8h, v14.8h, v1.8h       \n"
                    "fmla   v30.8h, v15.8h, v1.8h       \n"
                    "fmla   v31.8h, v16.8h, v1.8h       \n"

                    "fmla   v28.8h, v14.8h, v2.8h       \n"
                    "fmla   v29.8h, v15.8h, v2.8h       \n"
                    "fmla   v30.8h, v16.8h, v2.8h       \n"
                    "fmla   v31.8h, v17.8h, v2.8h       \n"

                    "prfm   pldl1keep, [%6, #512]       \n"
                    "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%6], #64 \n" // w04 w1_012

                    "fmla   v28.8h, v15.8h, v3.8h       \n"
                    "fmla   v29.8h, v16.8h, v3.8h       \n"
                    "fmla   v30.8h, v17.8h, v3.8h       \n"
                    "fmla   v31.8h, v18.8h, v3.8h       \n"

                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%2], #64 \n" // r1_0123

                    "fmla   v28.8h, v16.8h, v4.8h       \n"
                    "fmla   v29.8h, v17.8h, v4.8h       \n"
                    "fmla   v30.8h, v18.8h, v4.8h       \n"
                    "fmla   v31.8h, v19.8h, v4.8h       \n"

                    "fmla   v28.8h, v20.8h, v5.8h       \n"
                    "fmla   v29.8h, v21.8h, v5.8h       \n"
                    "fmla   v30.8h, v22.8h, v5.8h       \n"
                    "fmla   v31.8h, v23.8h, v5.8h       \n"

                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v24.8h, v25.8h, v26.8h, v27.8h}, [%2] \n" // r1_4567

                    "fmla   v28.8h, v21.8h, v6.8h       \n"
                    "fmla   v29.8h, v22.8h, v6.8h       \n"
                    "fmla   v30.8h, v23.8h, v6.8h       \n"
                    "fmla   v31.8h, v24.8h, v6.8h       \n"

                    "prfm   pldl1keep, [%6, #512]       \n"
                    "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%6], #64 \n" // w1_34 w2_01

                    "fmla   v28.8h, v22.8h, v7.8h       \n"
                    "fmla   v29.8h, v23.8h, v7.8h       \n"
                    "fmla   v30.8h, v24.8h, v7.8h       \n"
                    "fmla   v31.8h, v25.8h, v7.8h       \n"

                    "fmla   v28.8h, v23.8h, v0.8h       \n"
                    "fmla   v29.8h, v24.8h, v0.8h       \n"
                    "fmla   v30.8h, v25.8h, v0.8h       \n"
                    "fmla   v31.8h, v26.8h, v0.8h       \n"

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v12.8h, v13.8h, v14.8h, v15.8h}, [%3], #64 \n" // r2_0123

                    "fmla   v28.8h, v24.8h, v1.8h       \n"
                    "fmla   v29.8h, v25.8h, v1.8h       \n"
                    "fmla   v30.8h, v26.8h, v1.8h       \n"
                    "fmla   v31.8h, v27.8h, v1.8h       \n"

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%3] \n" // r2_4567

                    "fmla   v28.8h, v12.8h, v2.8h       \n"
                    "fmla   v29.8h, v13.8h, v2.8h       \n"
                    "fmla   v30.8h, v14.8h, v2.8h       \n"
                    "fmla   v31.8h, v15.8h, v2.8h       \n"

                    "prfm   pldl1keep, [%6, #512]       \n"
                    "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%6], #64 \n" // w2_234 w30

                    "fmla   v28.8h, v13.8h, v3.8h       \n"
                    "fmla   v29.8h, v14.8h, v3.8h       \n"
                    "fmla   v30.8h, v15.8h, v3.8h       \n"
                    "fmla   v31.8h, v16.8h, v3.8h       \n"

                    "fmla   v28.8h, v14.8h, v4.8h       \n"
                    "fmla   v29.8h, v15.8h, v4.8h       \n"
                    "fmla   v30.8h, v16.8h, v4.8h       \n"
                    "fmla   v31.8h, v17.8h, v4.8h       \n"

                    "fmla   v28.8h, v15.8h, v5.8h       \n"
                    "fmla   v29.8h, v16.8h, v5.8h       \n"
                    "fmla   v30.8h, v17.8h, v5.8h       \n"
                    "fmla   v31.8h, v18.8h, v5.8h       \n"

                    "prfm   pldl1keep, [%4, #512]       \n"
                    "ld1    {v20.8h, v21.8h, v22.8h, v23.8h}, [%4], #64 \n" // r3_0123

                    "fmla   v28.8h, v16.8h, v6.8h       \n"
                    "fmla   v29.8h, v17.8h, v6.8h       \n"
                    "fmla   v30.8h, v18.8h, v6.8h       \n"
                    "fmla   v31.8h, v19.8h, v6.8h       \n"

                    "prfm   pldl1keep, [%6, #512]       \n"
                    "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%6], #64 \n" // w3_1234

                    "fmla   v28.8h, v20.8h, v7.8h       \n"
                    "fmla   v29.8h, v21.8h, v7.8h       \n"
                    "fmla   v30.8h, v22.8h, v7.8h       \n"
                    "fmla   v31.8h, v23.8h, v7.8h       \n"

                    "prfm   pldl1keep, [%4, #512]       \n"
                    "ld1    {v24.8h, v25.8h, v26.8h, v27.8h}, [%4] \n" // r3_4567

                    "fmla   v28.8h, v21.8h, v0.8h       \n"
                    "fmla   v29.8h, v22.8h, v0.8h       \n"
                    "fmla   v30.8h, v23.8h, v0.8h       \n"
                    "fmla   v31.8h, v24.8h, v0.8h       \n"

                    "fmla   v28.8h, v22.8h, v1.8h       \n"
                    "fmla   v29.8h, v23.8h, v1.8h       \n"
                    "fmla   v30.8h, v24.8h, v1.8h       \n"
                    "fmla   v31.8h, v25.8h, v1.8h       \n"

                    "prfm   pldl1keep, [%5, #512]       \n"
                    "ld1    {v12.8h, v13.8h, v14.8h, v15.8h}, [%5], #64 \n" // r4_0123

                    "fmla   v28.8h, v23.8h, v2.8h       \n"
                    "fmla   v29.8h, v24.8h, v2.8h       \n"
                    "fmla   v30.8h, v25.8h, v2.8h       \n"
                    "fmla   v31.8h, v26.8h, v2.8h       \n"

                    "prfm   pldl1keep, [%6, #512]       \n"
                    "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%6], #64 \n" // w4_0123

                    "fmla   v28.8h, v24.8h, v3.8h       \n"
                    "fmla   v29.8h, v25.8h, v3.8h       \n"
                    "fmla   v30.8h, v26.8h, v3.8h       \n"
                    "fmla   v31.8h, v27.8h, v3.8h       \n"

                    "fmla   v28.8h, v12.8h, v4.8h       \n"
                    "fmla   v29.8h, v13.8h, v4.8h       \n"
                    "fmla   v30.8h, v14.8h, v4.8h       \n"
                    "fmla   v31.8h, v15.8h, v4.8h       \n"

                    "prfm   pldl1keep, [%5, #512]       \n"
                    "ld1    {v16.8h, v17.8h, v18.8h, v19.8h}, [%5] \n" // r4_4567

                    "fmla   v28.8h, v13.8h, v5.8h       \n"
                    "fmla   v29.8h, v14.8h, v5.8h       \n"
                    "fmla   v30.8h, v15.8h, v5.8h       \n"
                    "fmla   v31.8h, v16.8h, v5.8h       \n"

                    "fmla   v28.8h, v14.8h, v6.8h       \n"
                    "fmla   v29.8h, v15.8h, v6.8h       \n"
                    "fmla   v30.8h, v16.8h, v6.8h       \n"
                    "fmla   v31.8h, v17.8h, v6.8h       \n"

                    "prfm   pldl1keep, [%6, #128]       \n"
                    "ld1    {v0.8h}, [%6]               \n" // w44

                    "fmla   v28.8h, v15.8h, v7.8h       \n"
                    "fmla   v29.8h, v16.8h, v7.8h       \n"
                    "fmla   v30.8h, v17.8h, v7.8h       \n"
                    "fmla   v31.8h, v18.8h, v7.8h       \n"

                    "fmla   v28.8h, v16.8h, v0.8h       \n"
                    "fmla   v29.8h, v17.8h, v0.8h       \n"
                    "fmla   v30.8h, v18.8h, v0.8h       \n"
                    "fmla   v31.8h, v19.8h, v0.8h       \n"

                    "sub    %6, %6, #384                \n" // k0 -= 24 * 8

                    "st1    {v28.8h, v29.8h, v30.8h, v31.8h}, [%0], #64 \n"

                    : "=r"(outptr0), // %0
                    "=r"(r0),      // %1
                    "=r"(r1),      // %2
                    "=r"(r2),      // %3
                    "=r"(r3),      // %4
                    "=r"(r4),      // %5
                    "=r"(k0)       // %6
                    : "0"(outptr0),
                    "1"(r0),
                    "2"(r1),
                    "3"(r2),
                    "4"(r3),
                    "5"(r4),
                    "6"(k0),
                    "w"(_bias0) // %14
                    : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
            }
            for (; j + 1 < outw; j += 2)
            {
                asm volatile(
                    "prfm   pldl1keep, [%1, #256]       \n"
                    "ld1    {v16.8h, v17.8h}, [%1], #32 \n" // r0_01

                    "prfm   pldl1keep, [%6, #512]       \n"
                    "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%6], #64 \n" // w0_0123

                    "mov    v30.16b, %14.16b            \n" // sum00
                    "mov    v31.16b, %14.16b            \n" // sum01

                    "fmla   v30.8h, v16.8h, v0.8h       \n"
                    "fmla   v31.8h, v17.8h, v0.8h       \n"

                    "prfm   pldl1keep, [%1, #512]       \n"
                    "ld1    {v18.8h, v19.8h, v20.8h, v21.8h}, [%1] \n" // r0_2345

                    "fmla   v30.8h, v17.8h, v1.8h       \n"
                    "fmla   v31.8h, v18.8h, v1.8h       \n"
                    "fmla   v30.8h, v18.8h, v2.8h       \n"
                    "fmla   v31.8h, v19.8h, v2.8h       \n"

                    "prfm   pldl1keep, [%6, #512]       \n"
                    "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%6], #64 \n" // w04 w1_012

                    "fmla   v30.8h, v19.8h, v3.8h       \n"
                    "fmla   v31.8h, v20.8h, v3.8h       \n"

                    "prfm   pldl1keep, [%2, #256]       \n"
                    "ld1    {v22.8h, v23.8h}, [%2], #32 \n" // r1_01

                    "fmla   v30.8h, v20.8h, v4.8h       \n"
                    "fmla   v31.8h, v21.8h, v4.8h       \n"

                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v24.8h, v25.8h, v26.8h, v27.8h}, [%2] \n" // r1_2345

                    "fmla   v30.8h, v22.8h, v5.8h       \n"
                    "fmla   v31.8h, v23.8h, v5.8h       \n"
                    "fmla   v30.8h, v23.8h, v6.8h       \n"
                    "fmla   v31.8h, v24.8h, v6.8h       \n"

                    "prfm   pldl1keep, [%6, #512]       \n"
                    "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%6], #64 \n" // w1_34 w2_01

                    "fmla   v30.8h, v24.8h, v7.8h       \n"
                    "fmla   v31.8h, v25.8h, v7.8h       \n"
                    "fmla   v30.8h, v25.8h, v0.8h       \n"
                    "fmla   v31.8h, v26.8h, v0.8h       \n"

                    "prfm   pldl1keep, [%3, #256]       \n"
                    "ld1    {v16.8h, v17.8h}, [%3], #32 \n" // r2_01

                    "fmla   v30.8h, v26.8h, v1.8h       \n"
                    "fmla   v31.8h, v27.8h, v1.8h       \n"

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v18.8h, v19.8h, v20.8h, v21.8h}, [%3] \n" // r2_2345

                    "fmla   v30.8h, v16.8h, v2.8h       \n"
                    "fmla   v31.8h, v17.8h, v2.8h       \n"

                    "prfm   pldl1keep, [%6, #512]       \n"
                    "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%6], #64 \n" // w2_234 w30

                    "fmla   v30.8h, v17.8h, v3.8h       \n"
                    "fmla   v31.8h, v18.8h, v3.8h       \n"
                    "fmla   v30.8h, v18.8h, v4.8h       \n"
                    "fmla   v31.8h, v19.8h, v4.8h       \n"

                    "prfm   pldl1keep, [%4, #256]       \n"
                    "ld1    {v22.8h, v23.8h}, [%4], #32 \n" // r3_01

                    "fmla   v30.8h, v19.8h, v5.8h       \n"
                    "fmla   v31.8h, v20.8h, v5.8h       \n"

                    "prfm   pldl1keep, [%6, #512]       \n"
                    "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%6], #64 \n" // w3_1234

                    "fmla   v30.8h, v20.8h, v6.8h       \n"
                    "fmla   v31.8h, v21.8h, v6.8h       \n"

                    "prfm   pldl1keep, [%4, #512]       \n"
                    "ld1    {v24.8h, v25.8h, v26.8h, v27.8h}, [%4] \n" // r3_2345

                    "fmla   v30.8h, v22.8h, v7.8h       \n"
                    "fmla   v31.8h, v23.8h, v7.8h       \n"
                    "fmla   v30.8h, v23.8h, v0.8h       \n"
                    "fmla   v31.8h, v24.8h, v0.8h       \n"
                    "fmla   v30.8h, v24.8h, v1.8h       \n"
                    "fmla   v31.8h, v25.8h, v1.8h       \n"

                    "prfm   pldl1keep, [%5, #256]       \n"
                    "ld1    {v16.8h, v17.8h}, [%5], #32 \n" // r4_01

                    "fmla   v30.8h, v25.8h, v2.8h       \n"
                    "fmla   v31.8h, v26.8h, v2.8h       \n"

                    "prfm   pldl1keep, [%6, #512]       \n"
                    "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%6], #64 \n" // w4_0123

                    "fmla   v30.8h, v26.8h, v3.8h       \n"
                    "fmla   v31.8h, v27.8h, v3.8h       \n"

                    "prfm   pldl1keep, [%5, #512]       \n"
                    "ld1    {v18.8h, v19.8h, v20.8h, v21.8h}, [%5] \n" // r4_2345

                    "fmla   v30.8h, v16.8h, v4.8h       \n"
                    "fmla   v31.8h, v17.8h, v4.8h       \n"
                    "fmla   v30.8h, v17.8h, v5.8h       \n"
                    "fmla   v31.8h, v18.8h, v5.8h       \n"

                    "prfm   pldl1keep, [%6, #128]       \n"
                    "ld1    {v0.8h}, [%6]               \n" // w44

                    "fmla   v30.8h, v18.8h, v6.8h       \n"
                    "fmla   v31.8h, v19.8h, v6.8h       \n"
                    "fmla   v30.8h, v19.8h, v7.8h       \n"
                    "fmla   v31.8h, v20.8h, v7.8h       \n"
                    "fmla   v30.8h, v20.8h, v0.8h       \n"
                    "fmla   v31.8h, v21.8h, v0.8h       \n"

                    "sub    %6, %6, #384                \n" // k0 -= 24 * 8

                    "st1    {v30.8h, v31.8h}, [%0], #32 \n"

                    : "=r"(outptr0), // %0
                    "=r"(r0),      // %1
                    "=r"(r1),      // %2
                    "=r"(r2),      // %3
                    "=r"(r3),      // %4
                    "=r"(r4),      // %5
                    "=r"(k0)       // %6
                    : "0"(outptr0),
                    "1"(r0),
                    "2"(r1),
                    "3"(r2),
                    "4"(r3),
                    "5"(r4),
                    "6"(k0),
                    "w"(_bias0) // %14
                    : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v30", "v31");
            }
            for (; j < outw; j++)
            {
                asm volatile(
                    "prfm   pldl1keep, [%1, #128]       \n"
                    "ld1    {v16.8h}, [%1], #16         \n" // r0_0

                    "prfm   pldl1keep, [%6, #512]       \n"
                    "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%6], #64 \n" // w0_0123

                    "mov    v30.16b, %14.16b            \n" // sum00

                    "prfm   pldl1keep, [%1, #512]       \n"
                    "ld1    {v17.8h, v18.8h, v19.8h, v20.8h}, [%1] \n" // r0_1234

                    "fmla   v30.8h, v16.8h, v0.8h       \n"

                    "prfm   pldl1keep, [%6, #512]       \n"
                    "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%6], #64 \n" // w04 w1_012

                    "fmla   v30.8h, v17.8h, v1.8h       \n"

                    "fmla   v30.8h, v18.8h, v2.8h       \n"

                    "prfm   pldl1keep, [%2, #128]       \n"
                    "ld1    {v21.8h}, [%2], #16         \n" // r1_0

                    "fmla   v30.8h, v19.8h, v3.8h       \n"

                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v22.8h, v23.8h, v24.8h, v25.8h}, [%2] \n" // r1_1234

                    "fmla   v30.8h, v20.8h, v4.8h       \n"

                    "fmla   v30.8h, v21.8h, v5.8h       \n"

                    "prfm   pldl1keep, [%6, #512]       \n"
                    "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%6], #64 \n" // w1_34 w2_01

                    "fmla   v30.8h, v22.8h, v6.8h       \n"

                    "fmla   v30.8h, v23.8h, v7.8h       \n"

                    "prfm   pldl1keep, [%3, #128]       \n"
                    "ld1    {v16.8h}, [%3], #16         \n" // r2_0

                    "fmla   v30.8h, v24.8h, v0.8h       \n"

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v17.8h, v18.8h, v19.8h, v20.8h}, [%3] \n" // r2_1234

                    "fmla   v30.8h, v25.8h, v1.8h       \n"

                    "prfm   pldl1keep, [%6, #512]       \n"
                    "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%6], #64 \n" // w2_234 w30

                    "fmla   v30.8h, v16.8h, v2.8h       \n"
                    "fmla   v30.8h, v17.8h, v3.8h       \n"

                    "prfm   pldl1keep, [%4, #128]       \n"
                    "ld1    {v21.8h}, [%4], #16         \n" // r3_0

                    "fmla   v30.8h, v18.8h, v4.8h       \n"

                    "prfm   pldl1keep, [%4, #512]       \n"
                    "ld1    {v22.8h, v23.8h, v24.8h, v25.8h}, [%4] \n" // r3_1234

                    "fmla   v30.8h, v19.8h, v5.8h       \n"

                    "prfm   pldl1keep, [%6, #512]       \n"
                    "ld1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%6], #64 \n" // w3_1234

                    "fmla   v30.8h, v20.8h, v6.8h       \n"

                    "fmla   v30.8h, v21.8h, v7.8h       \n"
                    "fmla   v30.8h, v22.8h, v0.8h       \n"

                    "prfm   pldl1keep, [%5, #128]       \n"
                    "ld1    {v16.8h}, [%5], #16         \n" // r4_0

                    "fmla   v30.8h, v23.8h, v1.8h       \n"

                    "prfm   pldl1keep, [%6, #512]       \n"
                    "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%6], #64 \n" // w4_0123

                    "fmla   v30.8h, v24.8h, v2.8h       \n"

                    "prfm   pldl1keep, [%5, #512]       \n"
                    "ld1    {v17.8h, v18.8h, v19.8h, v20.8h}, [%5] \n" // r4_1234

                    "fmla   v30.8h, v25.8h, v3.8h       \n"

                    "fmla   v30.8h, v16.8h, v4.8h       \n"
                    "fmla   v30.8h, v17.8h, v5.8h       \n"

                    "prfm   pldl1keep, [%6, #128]       \n"
                    "ld1    {v0.8h}, [%6]               \n" // w44

                    "fmla   v30.8h, v18.8h, v6.8h       \n"
                    "fmla   v30.8h, v19.8h, v7.8h       \n"
                    "fmla   v30.8h, v20.8h, v0.8h       \n"

                    "sub    %6, %6, #384                \n" // k0 -= 24 * 8

                    "st1    {v30.8h}, [%0], #16         \n"

                    : "=r"(outptr0), // %0
                    "=r"(r0),      // %1
                    "=r"(r1),      // %2
                    "=r"(r2),      // %3
                    "=r"(r3),      // %4
                    "=r"(r4),      // %5
                    "=r"(k0)       // %6
                    : "0"(outptr0),
                    "1"(r0),
                    "2"(r1),
                    "3"(r2),
                    "4"(r3),
                    "5"(r4),
                    "6"(k0),
                    "w"(_bias0) // %14
                    : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v30");
            }

            r0 += 4 * 8;
            r1 += 4 * 8;
            r2 += 4 * 8;
            r3 += 4 * 8;
            r4 += 4 * 8;
        }
    }
}

static void convdw5x5s2_pack8_fp16sa_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int group = bottom_blob.c;

    const int tailstep = (w - 2 * outw + w) * 8;

    const __fp16* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g = 0; g < group; g++)
    {
        Mat out = top_blob.channel(g);

        float16x8_t _bias0 = bias ? vld1q_f16(bias + g * 8) : vdupq_n_f16((__fp16)0.f);

        const __fp16* k0 = kernel.row<const __fp16>(g);

        __fp16* outptr0 = out.row<__fp16>(0);

        const Mat img0 = bottom_blob.channel(g);

        const __fp16* r0 = img0.row<const __fp16>(0);
        const __fp16* r1 = img0.row<const __fp16>(1);
        const __fp16* r2 = img0.row<const __fp16>(2);
        const __fp16* r3 = img0.row<const __fp16>(3);
        const __fp16* r4 = img0.row<const __fp16>(4);

        int i = 0;
        for (; i < outh; i++)
        {
            int j = 0;
            for (; j < outw; j++)
            {
                float16x8_t _sum0 = _bias0;

                float16x8_t _r00 = vld1q_f16(r0);
                float16x8_t _r01 = vld1q_f16(r0 + 8);
                float16x8_t _r02 = vld1q_f16(r0 + 16);
                float16x8_t _r03 = vld1q_f16(r0 + 24);
                float16x8_t _r04 = vld1q_f16(r0 + 32);

                float16x8_t _k00 = vld1q_f16(k0);
                float16x8_t _k01 = vld1q_f16(k0 + 8);
                float16x8_t _k02 = vld1q_f16(k0 + 16);
                float16x8_t _k03 = vld1q_f16(k0 + 24);
                float16x8_t _k04 = vld1q_f16(k0 + 32);
                k0 += 40;

                _sum0 = vfmaq_f16(_sum0, _k00, _r00);
                _sum0 = vfmaq_f16(_sum0, _k01, _r01);
                _sum0 = vfmaq_f16(_sum0, _k02, _r02);
                _sum0 = vfmaq_f16(_sum0, _k03, _r03);
                _sum0 = vfmaq_f16(_sum0, _k04, _r04);

                float16x8_t _r10 = vld1q_f16(r1);
                float16x8_t _r11 = vld1q_f16(r1 + 8);
                float16x8_t _r12 = vld1q_f16(r1 + 16);
                float16x8_t _r13 = vld1q_f16(r1 + 24);
                float16x8_t _r14 = vld1q_f16(r1 + 32);

                float16x8_t _k10 = vld1q_f16(k0);
                float16x8_t _k11 = vld1q_f16(k0 + 8);
                float16x8_t _k12 = vld1q_f16(k0 + 16);
                float16x8_t _k13 = vld1q_f16(k0 + 24);
                float16x8_t _k14 = vld1q_f16(k0 + 32);
                k0 += 40;

                _sum0 = vfmaq_f16(_sum0, _k10, _r10);
                _sum0 = vfmaq_f16(_sum0, _k11, _r11);
                _sum0 = vfmaq_f16(_sum0, _k12, _r12);
                _sum0 = vfmaq_f16(_sum0, _k13, _r13);
                _sum0 = vfmaq_f16(_sum0, _k14, _r14);

                float16x8_t _r20 = vld1q_f16(r2);
                float16x8_t _r21 = vld1q_f16(r2 + 8);
                float16x8_t _r22 = vld1q_f16(r2 + 16);
                float16x8_t _r23 = vld1q_f16(r2 + 24);
                float16x8_t _r24 = vld1q_f16(r2 + 32);

                float16x8_t _k20 = vld1q_f16(k0);
                float16x8_t _k21 = vld1q_f16(k0 + 8);
                float16x8_t _k22 = vld1q_f16(k0 + 16);
                float16x8_t _k23 = vld1q_f16(k0 + 24);
                float16x8_t _k24 = vld1q_f16(k0 + 32);
                k0 += 40;

                _sum0 = vfmaq_f16(_sum0, _k20, _r20);
                _sum0 = vfmaq_f16(_sum0, _k21, _r21);
                _sum0 = vfmaq_f16(_sum0, _k22, _r22);
                _sum0 = vfmaq_f16(_sum0, _k23, _r23);
                _sum0 = vfmaq_f16(_sum0, _k24, _r24);

                float16x8_t _r30 = vld1q_f16(r3);
                float16x8_t _r31 = vld1q_f16(r3 + 8);
                float16x8_t _r32 = vld1q_f16(r3 + 16);
                float16x8_t _r33 = vld1q_f16(r3 + 24);
                float16x8_t _r34 = vld1q_f16(r3 + 32);

                float16x8_t _k30 = vld1q_f16(k0);
                float16x8_t _k31 = vld1q_f16(k0 + 8);
                float16x8_t _k32 = vld1q_f16(k0 + 16);
                float16x8_t _k33 = vld1q_f16(k0 + 24);
                float16x8_t _k34 = vld1q_f16(k0 + 32);
                k0 += 40;

                _sum0 = vfmaq_f16(_sum0, _k30, _r30);
                _sum0 = vfmaq_f16(_sum0, _k31, _r31);
                _sum0 = vfmaq_f16(_sum0, _k32, _r32);
                _sum0 = vfmaq_f16(_sum0, _k33, _r33);
                _sum0 = vfmaq_f16(_sum0, _k34, _r34);

                float16x8_t _r40 = vld1q_f16(r4);
                float16x8_t _r41 = vld1q_f16(r4 + 8);
                float16x8_t _r42 = vld1q_f16(r4 + 16);
                float16x8_t _r43 = vld1q_f16(r4 + 24);
                float16x8_t _r44 = vld1q_f16(r4 + 32);

                float16x8_t _k40 = vld1q_f16(k0);
                float16x8_t _k41 = vld1q_f16(k0 + 8);
                float16x8_t _k42 = vld1q_f16(k0 + 16);
                float16x8_t _k43 = vld1q_f16(k0 + 24);
                float16x8_t _k44 = vld1q_f16(k0 + 32);
                k0 -= 160;

                _sum0 = vfmaq_f16(_sum0, _k40, _r40);
                _sum0 = vfmaq_f16(_sum0, _k41, _r41);
                _sum0 = vfmaq_f16(_sum0, _k42, _r42);
                _sum0 = vfmaq_f16(_sum0, _k43, _r43);
                _sum0 = vfmaq_f16(_sum0, _k44, _r44);

                vst1q_f16(outptr0, _sum0);

                outptr0 += 8;

                r0 += 16;
                r1 += 16;
                r2 += 16;
                r3 += 16;
                r4 += 16;
            }

            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
            r3 += tailstep;
            r4 += tailstep;
        }
    }
}
