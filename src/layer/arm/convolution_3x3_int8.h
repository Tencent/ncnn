// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

static void conv3x3s2_transform_kernel_int8_neon(const Mat& _kernel, Mat& kernel_tm, int inch, int outch)
{
    kernel_tm.create(8 * 9, inch, outch / 8 + outch % 8, (size_t)1u);

    const signed char* kernel = _kernel;

    int p = 0;
    for (; p + 7 < outch; p += 8)
    {
        const signed char* k0 = kernel + (p + 0) * inch * 9;
        const signed char* k1 = kernel + (p + 1) * inch * 9;
        const signed char* k2 = kernel + (p + 2) * inch * 9;
        const signed char* k3 = kernel + (p + 3) * inch * 9;
        const signed char* k4 = kernel + (p + 4) * inch * 9;
        const signed char* k5 = kernel + (p + 5) * inch * 9;
        const signed char* k6 = kernel + (p + 6) * inch * 9;
        const signed char* k7 = kernel + (p + 7) * inch * 9;

        signed char* ktmp = kernel_tm.channel(p / 8);

        for (int q = 0; q < inch; q++)
        {
            for (int k = 0; k < 9; k++)
            {
                ktmp[0] = k0[k];
                ktmp[1] = k1[k];
                ktmp[2] = k2[k];
                ktmp[3] = k3[k];
                ktmp[4] = k4[k];
                ktmp[5] = k5[k];
                ktmp[6] = k6[k];
                ktmp[7] = k7[k];
                ktmp += 8;
            }

            k0 += 9;
            k1 += 9;
            k2 += 9;
            k3 += 9;
            k4 += 9;
            k5 += 9;
            k6 += 9;
            k7 += 9;
        }
    }
    for (; p < outch; p++)
    {
        const signed char* k0 = kernel + (p + 0) * inch * 9;

        signed char* ktmp = kernel_tm.channel(p / 8 + p % 8);

        for (int q = 0; q < inch; q++)
        {
            for (int k = 0; k < 9; k++)
            {
                ktmp[k] = k0[k];
            }
            ktmp += 9;

            k0 += 9;
        }
    }
}

static void conv3x3s2_packed_int8_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2 * outw + w;

    int nn_outch = outch >> 3;
    int remain_outch_start = nn_outch << 3;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp = 0; pp < nn_outch; pp++)
    {
        int p = pp * 8;

        Mat out0 = top_blob.channel(p + 0);
        Mat out1 = top_blob.channel(p + 1);
        Mat out2 = top_blob.channel(p + 2);
        Mat out3 = top_blob.channel(p + 3);
        Mat out4 = top_blob.channel(p + 4);
        Mat out5 = top_blob.channel(p + 5);
        Mat out6 = top_blob.channel(p + 6);
        Mat out7 = top_blob.channel(p + 7);

        out0.fill(0);
        out1.fill(0);
        out2.fill(0);
        out3.fill(0);
        out4.fill(0);
        out5.fill(0);
        out6.fill(0);
        out7.fill(0);

        const signed char* ktmp = _kernel.channel(p / 8);

        for (int q = 0; q < inch; q++)
        {
            int* outptr0 = out0;
            int* outptr1 = out1;
            int* outptr2 = out2;
            int* outptr3 = out3;
            int* outptr4 = out4;
            int* outptr5 = out5;
            int* outptr6 = out6;
            int* outptr7 = out7;

            const signed char* img0 = bottom_blob.channel(q);

            const signed char* r0 = img0;
            const signed char* r1 = img0 + w;
            const signed char* r2 = img0 + w * 2;

            int i = 0;

            for (; i < outh; i++)
            {
#if __ARM_NEON
#if __aarch64__
                int nn = outw >> 3;
                int remain = outw & 7;
#else
                int nn = outw >> 2;
                int remain = outw & 3;
#endif // __aarch64__
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                if (nn > 0)
                {
                    asm volatile(
                        "0:                                   \n"

                        "ld1    {v0.8b, v1.8b, v2.8b}, [%12], #24  \n" //ktmp
                        "ld2    {v3.8b, v4.8b}, [%9], #16     \n"      //r0-r2
                        "ld2    {v5.8b, v6.8b}, [%9]          \n"

                        "ld1    {v8.4s, v9.4s}, [%1]          \n" //out0
                        "ld1    {v10.4s, v11.4s}, [%2]        \n" //out1
                        "ld1    {v12.4s, v13.4s}, [%3]        \n" //out2
                        "ld1    {v14.4s, v15.4s}, [%4]        \n" //out3
                        "ld1    {v16.4s, v17.4s}, [%5]        \n" //out4
                        "ld1    {v18.4s, v19.4s}, [%6]        \n" //out5
                        "ld1    {v20.4s, v21.4s}, [%7]        \n" //out6
                        "ld1    {v22.4s, v23.4s}, [%8]        \n" //out7

                        "ext    v7.8b, v3.8b, v5.8b, #1       \n"

                        "sshll  v0.8h, v0.8b, #0              \n" //(k00-k70)
                        "sshll  v1.8h, v1.8b, #0              \n" //(k01-k71)
                        "sshll  v2.8h, v2.8b, #0              \n" //(k02-k72)
                        "sshll  v3.8h, v3.8b, #0              \n" // r0
                        "sshll  v4.8h, v4.8b, #0              \n" // r1
                        "sshll  v7.8h, v7.8b, #0              \n" // r2

                        // r0
                        "smlal  v8.4s, v3.4h, v0.h[0]         \n" // out0 += (r00-r07)*k00
                        "smlal2  v9.4s, v3.8h, v0.h[0]        \n"
                        "smlal  v10.4s, v3.4h, v0.h[1]        \n" // out1 += (r00-r07)*k10
                        "smlal2  v11.4s, v3.8h, v0.h[1]       \n"
                        "smlal  v12.4s, v3.4h, v0.h[2]        \n" // out2 += (r00-r07)*k20
                        "smlal2  v13.4s, v3.8h, v0.h[2]       \n"
                        "smlal  v14.4s, v3.4h, v0.h[3]        \n" // out3 += (r00-r07)*k30
                        "smlal2  v15.4s, v3.8h, v0.h[3]       \n"
                        "smlal  v16.4s, v3.4h, v0.h[4]        \n" // out4 += (r00-r07)*k40
                        "smlal2  v17.4s, v3.8h, v0.h[4]       \n"
                        "smlal  v18.4s, v3.4h, v0.h[5]        \n" // out5 += (r00-r07)*k50
                        "smlal2  v19.4s, v3.8h, v0.h[5]       \n"
                        "smlal  v20.4s, v3.4h, v0.h[6]        \n" // out6 += (r00-r07)*k60
                        "smlal2  v21.4s, v3.8h, v0.h[6]       \n"
                        "smlal  v22.4s, v3.4h, v0.h[7]        \n" // out7 += (r00-r07)*k70
                        "smlal2  v23.4s, v3.8h, v0.h[7]       \n"
                        // r1
                        "smlal  v8.4s, v4.4h, v1.h[0]         \n" // out0 += (r10-r17)*k01
                        "smlal2  v9.4s, v4.8h, v1.h[0]        \n"
                        "smlal  v10.4s, v4.4h, v1.h[1]        \n" // out1 += (r10-r17)*k11
                        "smlal2  v11.4s, v4.8h, v1.h[1]       \n"
                        "smlal  v12.4s, v4.4h, v1.h[2]        \n" // out2 += (r10-r17)*k21
                        "smlal2  v13.4s, v4.8h, v1.h[2]       \n"
                        "smlal  v14.4s, v4.4h, v1.h[3]        \n" // out3 += (r10-r17)*k31
                        "smlal2  v15.4s, v4.8h, v1.h[3]       \n"
                        "smlal  v16.4s, v4.4h, v1.h[4]        \n" // out4 += (r10-r17)*k41
                        "smlal2  v17.4s, v4.8h, v1.h[4]       \n"
                        "smlal  v18.4s, v4.4h, v1.h[5]        \n" // out5 += (r10-r17)*k51
                        "smlal2  v19.4s, v4.8h, v1.h[5]       \n"
                        "smlal  v20.4s, v4.4h, v1.h[6]        \n" // out6 += (r10-r17)*k61
                        "smlal2  v21.4s, v4.8h, v1.h[6]       \n"
                        "smlal  v22.4s, v4.4h, v1.h[7]        \n" // out7 += (r10-r17)*k71
                        "smlal2  v23.4s, v4.8h, v1.h[7]       \n"
                        // r2
                        "smlal  v8.4s, v7.4h, v2.h[0]         \n" // out0 += (r20-r27)*k02
                        "smlal2  v9.4s, v7.8h, v2.h[0]        \n"
                        "smlal  v10.4s, v7.4h, v2.h[1]        \n" // out1 += (r20-r27)*k12
                        "smlal2  v11.4s, v7.8h, v2.h[1]       \n"
                        "smlal  v12.4s, v7.4h, v2.h[2]        \n" // out2 += (r20-r27)*k22
                        "smlal2  v13.4s, v7.8h, v2.h[2]       \n"
                        "smlal  v14.4s, v7.4h, v2.h[3]        \n" // out3 += (r20-r27)*k32
                        "smlal2  v15.4s, v7.8h, v2.h[3]       \n"
                        "smlal  v16.4s, v7.4h, v2.h[4]        \n" // out4 += (r20-r27)*k42
                        "smlal2  v17.4s, v7.8h, v2.h[4]       \n"
                        "smlal  v18.4s, v7.4h, v2.h[5]        \n" // out5 += (r20-r27)*k52
                        "smlal2  v19.4s, v7.8h, v2.h[5]       \n"
                        "smlal  v20.4s, v7.4h, v2.h[6]        \n" // out6 += (r20-r27)*k62
                        "smlal2  v21.4s, v7.8h, v2.h[6]       \n"
                        "smlal  v22.4s, v7.4h, v2.h[7]        \n" // out7 += (r20-r27)*k72
                        "smlal2  v23.4s, v7.8h, v2.h[7]       \n"

                        "ld1    {v0.8b, v1.8b, v2.8b}, [%12], #24  \n" //ktmp
                        "ld2    {v3.8b, v4.8b}, [%10], #16    \n"      //r3-r5
                        "ld2    {v5.8b, v6.8b}, [%10]         \n"

                        "ext    v7.8b, v3.8b, v5.8b, #1       \n"

                        "sshll  v0.8h, v0.8b, #0              \n" //(k03-k73)
                        "sshll  v1.8h, v1.8b, #0              \n" //(k04-k74)
                        "sshll  v2.8h, v2.8b, #0              \n" //(k05-k75)
                        "sshll  v3.8h, v3.8b, #0              \n" // r3
                        "sshll  v4.8h, v4.8b, #0              \n" // r4
                        "sshll  v7.8h, v7.8b, #0              \n" // r5

                        // r3
                        "smlal  v8.4s, v3.4h, v0.h[0]         \n" // out0 += (r30-r37)*k03
                        "smlal2  v9.4s, v3.8h, v0.h[0]        \n"
                        "smlal  v10.4s, v3.4h, v0.h[1]        \n" // out1 += (r30-r37)*k13
                        "smlal2  v11.4s, v3.8h, v0.h[1]       \n"
                        "smlal  v12.4s, v3.4h, v0.h[2]        \n" // out2 += (r30-r37)*k23
                        "smlal2  v13.4s, v3.8h, v0.h[2]       \n"
                        "smlal  v14.4s, v3.4h, v0.h[3]        \n" // out3 += (r30-r37)*k33
                        "smlal2  v15.4s, v3.8h, v0.h[3]       \n"
                        "smlal  v16.4s, v3.4h, v0.h[4]        \n" // out4 += (r30-r37)*k43
                        "smlal2  v17.4s, v3.8h, v0.h[4]       \n"
                        "smlal  v18.4s, v3.4h, v0.h[5]        \n" // out5 += (r30-r37)*k53
                        "smlal2  v19.4s, v3.8h, v0.h[5]       \n"
                        "smlal  v20.4s, v3.4h, v0.h[6]        \n" // out6 += (r30-r37)*k63
                        "smlal2  v21.4s, v3.8h, v0.h[6]       \n"
                        "smlal  v22.4s, v3.4h, v0.h[7]        \n" // out7 += (r30-r37)*k73
                        "smlal2  v23.4s, v3.8h, v0.h[7]       \n"
                        // r4
                        "smlal  v8.4s, v4.4h, v1.h[0]         \n" // out0 += (r40-r47)*k04
                        "smlal2  v9.4s, v4.8h, v1.h[0]        \n"
                        "smlal  v10.4s, v4.4h, v1.h[1]        \n" // out1 += (r40-r47)*k14
                        "smlal2  v11.4s, v4.8h, v1.h[1]       \n"
                        "smlal  v12.4s, v4.4h, v1.h[2]        \n" // out2 += (r40-r47)*k24
                        "smlal2  v13.4s, v4.8h, v1.h[2]       \n"
                        "smlal  v14.4s, v4.4h, v1.h[3]        \n" // out3 += (r40-r47)*k34
                        "smlal2  v15.4s, v4.8h, v1.h[3]       \n"
                        "smlal  v16.4s, v4.4h, v1.h[4]        \n" // out4 += (r40-r47)*k44
                        "smlal2  v17.4s, v4.8h, v1.h[4]       \n"
                        "smlal  v18.4s, v4.4h, v1.h[5]        \n" // out5 += (r40-r47)*k54
                        "smlal2  v19.4s, v4.8h, v1.h[5]       \n"
                        "smlal  v20.4s, v4.4h, v1.h[6]        \n" // out6 += (r40-r47)*k64
                        "smlal2  v21.4s, v4.8h, v1.h[6]       \n"
                        "smlal  v22.4s, v4.4h, v1.h[7]        \n" // out7 += (r40-r47)*k74
                        "smlal2  v23.4s, v4.8h, v1.h[7]       \n"
                        // r5
                        "smlal  v8.4s, v7.4h, v2.h[0]         \n" // out0 += (r50-r57)*k05
                        "smlal2  v9.4s, v7.8h, v2.h[0]        \n"
                        "smlal  v10.4s, v7.4h, v2.h[1]        \n" // out1 += (r50-r57)*k15
                        "smlal2  v11.4s, v7.8h, v2.h[1]       \n"
                        "smlal  v12.4s, v7.4h, v2.h[2]        \n" // out2 += (r50-r57)*k25
                        "smlal2  v13.4s, v7.8h, v2.h[2]       \n"
                        "smlal  v14.4s, v7.4h, v2.h[3]        \n" // out3 += (r50-r57)*k35
                        "smlal2  v15.4s, v7.8h, v2.h[3]       \n"
                        "smlal  v16.4s, v7.4h, v2.h[4]        \n" // out4 += (r50-r57)*k45
                        "smlal2  v17.4s, v7.8h, v2.h[4]       \n"
                        "smlal  v18.4s, v7.4h, v2.h[5]        \n" // out5 += (r50-r57)*k55
                        "smlal2  v19.4s, v7.8h, v2.h[5]       \n"
                        "smlal  v20.4s, v7.4h, v2.h[6]        \n" // out6 += (r50-r57)*k65
                        "smlal2  v21.4s, v7.8h, v2.h[6]       \n"
                        "smlal  v22.4s, v7.4h, v2.h[7]        \n" // out7 += (r50-r57)*k75
                        "smlal2  v23.4s, v7.8h, v2.h[7]       \n"

                        "ld1    {v0.8b, v1.8b, v2.8b}, [%12], #24  \n" //ktmp
                        "ld2    {v3.8b, v4.8b}, [%11], #16    \n"      //r6-r8
                        "ld2    {v5.8b, v6.8b}, [%11]         \n"

                        "ext    v7.8b, v3.8b, v5.8b, #1       \n"

                        "sshll  v0.8h, v0.8b, #0              \n" //(k06-k76)
                        "sshll  v1.8h, v1.8b, #0              \n" //(k07-k77)
                        "sshll  v2.8h, v2.8b, #0              \n" //(k08-k78)
                        "sshll  v3.8h, v3.8b, #0              \n" // r6
                        "sshll  v4.8h, v4.8b, #0              \n" // r7
                        "sshll  v7.8h, v7.8b, #0              \n" // r8

                        // r6
                        "smlal  v8.4s, v3.4h, v0.h[0]         \n" // out0 += (r60-r67)*k06
                        "smlal2  v9.4s, v3.8h, v0.h[0]        \n"
                        "smlal  v10.4s, v3.4h, v0.h[1]        \n" // out1 += (r60-r67)*k16
                        "smlal2  v11.4s, v3.8h, v0.h[1]       \n"
                        "smlal  v12.4s, v3.4h, v0.h[2]        \n" // out2 += (r60-r67)*k26
                        "smlal2  v13.4s, v3.8h, v0.h[2]       \n"
                        "smlal  v14.4s, v3.4h, v0.h[3]        \n" // out3 += (r60-r67)*k36
                        "smlal2  v15.4s, v3.8h, v0.h[3]       \n"
                        "smlal  v16.4s, v3.4h, v0.h[4]        \n" // out4 += (r60-r67)*k46
                        "smlal2  v17.4s, v3.8h, v0.h[4]       \n"
                        "smlal  v18.4s, v3.4h, v0.h[5]        \n" // out5 += (r60-r67)*k56
                        "smlal2  v19.4s, v3.8h, v0.h[5]       \n"
                        "smlal  v20.4s, v3.4h, v0.h[6]        \n" // out6 += (r60-r67)*k66
                        "smlal2  v21.4s, v3.8h, v0.h[6]       \n"
                        "smlal  v22.4s, v3.4h, v0.h[7]        \n" // out7 += (r60-r67)*k76
                        "smlal2  v23.4s, v3.8h, v0.h[7]       \n"
                        // r7
                        "smlal  v8.4s, v4.4h, v1.h[0]         \n" // out0 += (r70-r77)*k07
                        "smlal2  v9.4s, v4.8h, v1.h[0]        \n"
                        "smlal  v10.4s, v4.4h, v1.h[1]        \n" // out1 += (r70-r77)*k17
                        "smlal2  v11.4s, v4.8h, v1.h[1]       \n"
                        "smlal  v12.4s, v4.4h, v1.h[2]        \n" // out2 += (r70-r77)*k27
                        "smlal2  v13.4s, v4.8h, v1.h[2]       \n"
                        "smlal  v14.4s, v4.4h, v1.h[3]        \n" // out3 += (r70-r77)*k37
                        "smlal2  v15.4s, v4.8h, v1.h[3]       \n"
                        "smlal  v16.4s, v4.4h, v1.h[4]        \n" // out4 += (r70-r77)*k47
                        "smlal2  v17.4s, v4.8h, v1.h[4]       \n"
                        "smlal  v18.4s, v4.4h, v1.h[5]        \n" // out5 += (r70-r77)*k57
                        "smlal2  v19.4s, v4.8h, v1.h[5]       \n"
                        "smlal  v20.4s, v4.4h, v1.h[6]        \n" // out6 += (r70-r77)*k67
                        "smlal2  v21.4s, v4.8h, v1.h[6]       \n"
                        "smlal  v22.4s, v4.4h, v1.h[7]        \n" // out7 += (r70-r77)*k77
                        "smlal2  v23.4s, v4.8h, v1.h[7]       \n"
                        // r8
                        "smlal  v8.4s, v7.4h, v2.h[0]         \n" // out0 += (r80-r87)*k08
                        "smlal2  v9.4s, v7.8h, v2.h[0]        \n"
                        "smlal  v10.4s, v7.4h, v2.h[1]        \n" // out1 += (r80-r87)*k18
                        "smlal2  v11.4s, v7.8h, v2.h[1]       \n"
                        "smlal  v12.4s, v7.4h, v2.h[2]        \n" // out2 += (r80-r87)*k28
                        "smlal2  v13.4s, v7.8h, v2.h[2]       \n"
                        "smlal  v14.4s, v7.4h, v2.h[3]        \n" // out3 += (r80-r87)*k38
                        "smlal2  v15.4s, v7.8h, v2.h[3]       \n"
                        "smlal  v16.4s, v7.4h, v2.h[4]        \n" // out4 += (r80-r87)*k48
                        "smlal2  v17.4s, v7.8h, v2.h[4]       \n"
                        "smlal  v18.4s, v7.4h, v2.h[5]        \n" // out5 += (r80-r87)*k58
                        "smlal2  v19.4s, v7.8h, v2.h[5]       \n"
                        "smlal  v20.4s, v7.4h, v2.h[6]        \n" // out6 += (r80-r87)*k68
                        "smlal2  v21.4s, v7.8h, v2.h[6]       \n"
                        "smlal  v22.4s, v7.4h, v2.h[7]        \n" // out7 += (r80-r87)*k78
                        "smlal2  v23.4s, v7.8h, v2.h[7]       \n"

                        "st1    {v8.4s, v9.4s}, [%1], #32     \n"
                        "st1    {v10.4s, v11.4s}, [%2], #32   \n"
                        "st1    {v12.4s, v13.4s}, [%3], #32   \n"
                        "st1    {v14.4s, v15.4s}, [%4], #32   \n"
                        "st1    {v16.4s, v17.4s}, [%5], #32   \n"
                        "st1    {v18.4s, v19.4s}, [%6], #32   \n"
                        "st1    {v20.4s, v21.4s}, [%7], #32   \n"
                        "st1    {v22.4s, v23.4s}, [%8], #32   \n"

                        "subs   %w0, %w0, #1                  \n"
                        "sub    %12, %12, #72                 \n" // reset ktmp

                        "bne    0b                            \n"

                        : "=r"(nn),      // %0
                        "=r"(outptr0), // %1
                        "=r"(outptr1), // %2
                        "=r"(outptr2), // %3
                        "=r"(outptr3), // %4
                        "=r"(outptr4), // %5
                        "=r"(outptr5), // %6
                        "=r"(outptr6), // %7
                        "=r"(outptr7), // %8
                        "=r"(r0),      // %9
                        "=r"(r1),      // %10
                        "=r"(r2),      // %11
                        "=r"(ktmp)     // %12
                        : "0"(nn),
                        "1"(outptr0),
                        "2"(outptr1),
                        "3"(outptr2),
                        "4"(outptr3),
                        "5"(outptr4),
                        "6"(outptr5),
                        "7"(outptr6),
                        "8"(outptr7),
                        "9"(r0),
                        "10"(r1),
                        "11"(r2),
                        "12"(ktmp)
                        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
                }
#else  // __aarch64__
                if (nn > 0)
                {
                    asm volatile(
                        "0:                             \n"
                        "pld        [%1, #128]          \n"
                        "vld1.s32   {d16-d17}, [%1]     \n" // out0
                        "pld        [%2, #128]          \n"
                        "vld1.s32   {d18-d19}, [%2]     \n" // out1
                        "pld        [%3, #128]          \n"
                        "vld1.s32   {d20-d21}, [%3]     \n" // out2
                        "pld        [%4, #128]          \n"
                        "vld1.s32   {d22-d23}, [%4]     \n" // out3

                        // r0
                        "pld        [%9, #64]          \n"
                        "vld2.s8    {d8-d9}, [%9]       \n" // d8(a00 a02 a04 a06 a08 a010 a012 a014), d9(a01 a03 a05 a07 a09 a011 a013 a015)
                        "add        %9, #8              \n"
                        "pld        [%12, #64]         \n"
                        "vld1.s8    {d0-d2}, [%12]!     \n" // d0(k00-k70) d1(k01-k71) d2(k02-k72)

                        "pld        [%5, #128]          \n"
                        "vld1.s32   {d24-d25}, [%5]     \n" // out4
                        "pld        [%6, #128]          \n"
                        "vld1.s32   {d26-d27}, [%6]     \n" // out5

                        "vmovl.s8   q2, d2              \n" // q2(k02-k72)
                        "vmovl.s8   q1, d1              \n" // q1(k01-k71)
                        "vmovl.s8   q0, d0              \n" // q0(k00-k70)
                        "vext.s8    d12, d8, d8, #1     \n" // d12(a02 a04 a06 a08 x x x x)

                        "pld        [%7, #128]          \n"
                        "vld1.s32   {d28-d29}, [%7]     \n" // out6

                        "vmovl.s8   q5, d9              \n" // q5(a01 a03 a05 a07 a09 a011 a013 a015) d11
                        "vmovl.s8   q4, d8              \n" // q4(a00 a02 a04 a06 a08 a010 a012 a014) d9
                        "vmovl.s8   q6, d12             \n" // q6(a02 a04 a06 a08 a010 a012 a014 a016) d13

                        "pld        [%8, #128]          \n"
                        "vld1.s32   {d30-d31}, [%8]     \n" // out7

                        "vmlal.s16  q8, d8, d0[0]       \n" // sum0 += (a00 a02 a04 a06) * k00
                        "vmlal.s16  q9, d8, d0[1]       \n" // sum1 += (a00 a02 a04 a06) * k10
                        "vmlal.s16  q10, d8, d0[2]      \n" // sum2 += (a00 a02 a04 a06) * k20
                        "vmlal.s16  q11, d8, d0[3]      \n" // sum3 += (a00 a02 a04 a06) * k30
                        "vmlal.s16  q12, d8, d1[0]      \n" // sum4 += (a00 a02 a04 a06) * k40
                        "vmlal.s16  q13, d8, d1[1]      \n" // sum5 += (a00 a02 a04 a06) * k50
                        "vmlal.s16  q14, d8, d1[2]      \n" // sum6 += (a00 a02 a04 a06) * k60
                        "vmlal.s16  q15, d8, d1[3]      \n" // sum7 += (a00 a02 a04 a06) * k70

                        "vmlal.s16  q8, d10, d2[0]      \n" // sum0 += (a01-a07) * k01
                        "vmlal.s16  q9, d10, d2[1]      \n" // sum1 += (a01-a07) * k11
                        "vmlal.s16  q10, d10, d2[2]     \n" // sum2 += (a01-a07) * k21
                        "vmlal.s16  q11, d10, d2[3]     \n" // sum3 += (a01-a07) * k31
                        "vmlal.s16  q12, d10, d3[0]     \n" // sum4 += (a01-a07) * k41
                        "vmlal.s16  q13, d10, d3[1]     \n" // sum5 += (a01-a07) * k51
                        "vmlal.s16  q14, d10, d3[2]     \n" // sum6 += (a01-a07) * k61
                        "vmlal.s16  q15, d10, d3[3]     \n" // sum7 += (a01-a07) * k71

                        "pld        [%10, #64]         \n"
                        "vld2.s8    {d8-d9}, [%10]      \n" // d8(a10 a12 a14 a16 a18 a110 a112 a114), d9(a11 a13 a15 a17 a19 a111 a113 a115)
                        "add        %10, #8             \n"

                        "vmlal.s16  q8, d12, d4[0]      \n" // sum0 += (a02-a08) * k02
                        "vmlal.s16  q9, d12, d4[1]      \n" // sum1 += (a02-a08) * k12
                        "vmlal.s16  q10, d12, d4[2]     \n" // sum2 += (a02-a08) * k22
                        "vmlal.s16  q11, d12, d4[3]     \n" // sum3 += (a02-a08) * k32

                        "pld        [%12, #64]         \n"
                        "vld1.s8    {d0-d2}, [%12]!     \n" // d0(k03-k73) d1(k04-k74) d2(k05-k75)

                        "vmlal.s16  q12, d12, d5[0]     \n" // sum4 += (a02-a08) * k42
                        "vmlal.s16  q13, d12, d5[1]     \n" // sum5 += (a02-a08) * k52
                        "vmlal.s16  q14, d12, d5[2]     \n" // sum6 += (a02-a08) * k62
                        "vmlal.s16  q15, d12, d5[3]     \n" // sum7 += (a02-a08) * k72

                        // r1
                        "vext.s8    d12, d8, d8, #1     \n" // d12(a12 a14 a16 a18 x x x x)

                        "vmovl.s8   q2, d2              \n" // q2(k05-k75)
                        "vmovl.s8   q1, d1              \n" // q1(k04-k74)
                        "vmovl.s8   q0, d0              \n" // q0(k03-k73)
                        "vmovl.s8   q5, d9              \n" // q5(a11-a115)
                        "vmovl.s8   q4, d8              \n" // q4(a10-a114)
                        "vmovl.s8   q6, d12             \n" // q6(a12-a116)

                        "vmlal.s16  q8, d8, d0[0]       \n" // sum0 += (a10-a16) * k03
                        "vmlal.s16  q9, d8, d0[1]       \n" // sum1 += (a10-a16) * k13
                        "vmlal.s16  q10, d8, d0[2]      \n" // sum2 += (a10-a16) * k23
                        "vmlal.s16  q11, d8, d0[3]      \n" // sum3 += (a10-a16) * k33
                        "vmlal.s16  q12, d8, d1[0]      \n" // sum4 += (a10-a16) * k43
                        "vmlal.s16  q13, d8, d1[1]      \n" // sum5 += (a10-a16) * k53
                        "vmlal.s16  q14, d8, d1[2]      \n" // sum6 += (a10-a16) * k63
                        "vmlal.s16  q15, d8, d1[3]      \n" // sum7 += (a10-a16) * k73

                        "vmlal.s16  q8, d10, d2[0]      \n" // sum0 += (a11-a17) * k04
                        "vmlal.s16  q9, d10, d2[1]      \n" // sum1 += (a11-a17) * k14
                        "vmlal.s16  q10, d10, d2[2]     \n" // sum2 += (a11-a17) * k24
                        "vmlal.s16  q11, d10, d2[3]     \n" // sum3 += (a11-a17) * k34
                        "vmlal.s16  q12, d10, d3[0]     \n" // sum4 += (a11-a17) * k44
                        "vmlal.s16  q13, d10, d3[1]     \n" // sum5 += (a11-a17) * k54
                        "vmlal.s16  q14, d10, d3[2]     \n" // sum6 += (a11-a17) * k64
                        "vmlal.s16  q15, d10, d3[3]     \n" // sum7 += (a11-a17) * k74

                        "pld        [%11, #64]         \n"
                        "vld2.s8    {d8-d9}, [%11]      \n" // d8(a20 a22 a24 a26 a28 a210 a212 a214), d9(a21 a23 a25 a27 a29 a211 a213 a215)
                        "add        %11, #8             \n"

                        "vmlal.s16  q8, d12, d4[0]      \n" // sum0 += (a12-a18) * k05
                        "vmlal.s16  q9, d12, d4[1]      \n" // sum1 += (a12-a18) * k15
                        "vmlal.s16  q10, d12, d4[2]     \n" // sum2 += (a12-a18) * k25
                        "vmlal.s16  q11, d12, d4[3]     \n" // sum3 += (a12-a18) * k35

                        "pld        [%12, #64]         \n"
                        "vld1.s8    {d0-d2}, [%12]!     \n" // d0(k06-k76) d1(k07-k77) d2(k08-k78)

                        "vmlal.s16  q12, d12, d5[0]     \n" // sum4 += (a12-a18) * k45
                        "vmlal.s16  q13, d12, d5[1]     \n" // sum5 += (a12-a18) * k55
                        "vmlal.s16  q14, d12, d5[2]     \n" // sum6 += (a12-a18) * k65
                        "vmlal.s16  q15, d12, d5[3]     \n" // sum7 += (a12-a18) * k75

                        // r2
                        "vext.s8    d12, d8, d8, #1     \n" // d12(a22 a24 a26 a28 x x x x)

                        "vmovl.s8   q2, d2              \n" // q2(k08-k78)
                        "vmovl.s8   q1, d1              \n" // q1(k07-k77)
                        "vmovl.s8   q0, d0              \n" // q0(k06-k76)
                        "vmovl.s8   q5, d9              \n" // q5(a21-a215)
                        "vmovl.s8   q4, d8              \n" // q4(a20-a214)
                        "vmovl.s8   q6, d12             \n" // q6(a22-a216)

                        "vmlal.s16  q8, d8, d0[0]       \n" // sum0 += (a20-a26) * k06
                        "vmlal.s16  q9, d8, d0[1]       \n" // sum1 += (a20-a26) * k16
                        "vmlal.s16  q10, d8, d0[2]      \n" // sum2 += (a20-a26) * k26
                        "vmlal.s16  q11, d8, d0[3]      \n" // sum3 += (a20-a26) * k36
                        "vmlal.s16  q12, d8, d1[0]      \n" // sum4 += (a20-a26) * k46
                        "vmlal.s16  q13, d8, d1[1]      \n" // sum5 += (a20-a26) * k56
                        "vmlal.s16  q14, d8, d1[2]      \n" // sum6 += (a20-a26) * k66
                        "vmlal.s16  q15, d8, d1[3]      \n" // sum7 += (a20-a26) * k76

                        "vmlal.s16  q8, d10, d2[0]      \n" // sum0 += (a21-a27) * k07
                        "vmlal.s16  q9, d10, d2[1]      \n" // sum1 += (a21-a27) * k17
                        "vmlal.s16  q10, d10, d2[2]     \n" // sum2 += (a21-a27) * k27
                        "vmlal.s16  q11, d10, d2[3]     \n" // sum3 += (a21-a27) * k37
                        "vmlal.s16  q12, d10, d3[0]     \n" // sum4 += (a21-a27) * k47
                        "vmlal.s16  q13, d10, d3[1]     \n" // sum5 += (a21-a27) * k57
                        "vmlal.s16  q14, d10, d3[2]     \n" // sum6 += (a21-a27) * k67
                        "vmlal.s16  q15, d10, d3[3]     \n" // sum7 += (a21-a27) * k77

                        "vmlal.s16  q8, d12, d4[0]      \n" // sum0 += (a22-a28) * k08
                        "vmlal.s16  q9, d12, d4[1]      \n" // sum1 += (a22-a28) * k18
                        "vmlal.s16  q10, d12, d4[2]     \n" // sum2 += (a22-a28) * k28
                        "vmlal.s16  q11, d12, d4[3]     \n" // sum3 += (a22-a28) * k38
                        "vmlal.s16  q12, d12, d5[0]     \n" // sum4 += (a22-a28) * k48
                        "vmlal.s16  q13, d12, d5[1]     \n" // sum5 += (a22-a28) * k58
                        "vmlal.s16  q14, d12, d5[2]     \n" // sum6 += (a22-a28) * k68
                        "vmlal.s16  q15, d12, d5[3]     \n" // sum7 += (a22-a28) * k78

                        // save s32 to memory
                        "sub        %12, %12, #72       \n"
                        "vst1.s32   {d16-d17}, [%1]!    \n" // out0
                        "vst1.s32   {d18-d19}, [%2]!    \n" // out1
                        "vst1.s32   {d20-d21}, [%3]!    \n" // out2
                        "vst1.s32   {d22-d23}, [%4]!    \n" // out3
                        "subs       %0, #1              \n"
                        "vst1.s32   {d24-d25}, [%5]!    \n" // out4
                        "vst1.s32   {d26-d27}, [%6]!    \n" // out5
                        "vst1.s32   {d28-d29}, [%7]!    \n" // out6
                        "vst1.s32   {d30-d31}, [%8]!    \n" // out7

                        "bne        0b                  \n"
                        : "=r"(nn),      // %0
                        "=r"(outptr0), // %1
                        "=r"(outptr1), // %2
                        "=r"(outptr2), // %3
                        "=r"(outptr3), // %4
                        "=r"(outptr4), // %5
                        "=r"(outptr5), // %6
                        "=r"(outptr6), // %7
                        "=r"(outptr7), // %8
                        "=r"(r0),      // %9
                        "=r"(r1),      // %10
                        "=r"(r2),      // %11
                        "=r"(ktmp)     // %12
                        : "0"(nn),
                        "1"(outptr0),
                        "2"(outptr1),
                        "3"(outptr2),
                        "4"(outptr3),
                        "5"(outptr4),
                        "6"(outptr5),
                        "7"(outptr6),
                        "8"(outptr7),
                        "9"(r0),
                        "10"(r1),
                        "11"(r2),
                        "12"(ktmp)
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain > 0; remain--)
                {
#if __ARM_NEON
#if __aarch64__
                    int8x8_t _r0_s8 = vld1_s8(r0); // (a00 a01 a02 ....)
                    int8x8_t _r1_s8 = vld1_s8(r1); // (a10 a11 a12 ....)
                    int8x8_t _r2_s8 = vld1_s8(r2); // (a20 a21 a22 ....)

                    int16x8_t _r0 = vmovl_s8(_r0_s8);
                    int16x8_t _r1 = vmovl_s8(_r1_s8);
                    int16x8_t _r2 = vmovl_s8(_r2_s8);

                    int32x4_t _sum03 = {};
                    int32x4_t _sum47 = {};

                    _sum03 = vld1q_lane_s32(outptr0, _sum03, 0); // out0
                    _sum03 = vld1q_lane_s32(outptr1, _sum03, 1); // out1
                    _sum03 = vld1q_lane_s32(outptr2, _sum03, 2); // out2
                    _sum03 = vld1q_lane_s32(outptr3, _sum03, 3); // out3
                    _sum47 = vld1q_lane_s32(outptr4, _sum47, 0); // out4
                    _sum47 = vld1q_lane_s32(outptr5, _sum47, 1); // out5
                    _sum47 = vld1q_lane_s32(outptr6, _sum47, 2); // out6
                    _sum47 = vld1q_lane_s32(outptr7, _sum47, 3); // out7

                    // k0 - k2
                    int8x8_t _k0_8 = vld1_s8(ktmp);      //(k00-k70)
                    int8x8_t _k1_8 = vld1_s8(ktmp + 8);  //(k01-k71)
                    int8x8_t _k2_8 = vld1_s8(ktmp + 16); //(k02-k72)

                    int16x8_t _k0 = vmovl_s8(_k0_8);
                    int16x8_t _k1 = vmovl_s8(_k1_8);
                    int16x8_t _k2 = vmovl_s8(_k2_8);

                    int32x4_t _sum0 = vmull_laneq_s16(vget_low_s16(_k0), _r0, 0);
                    int32x4_t _sum0n = vmull_laneq_s16(vget_high_s16(_k0), _r0, 0);
                    int32x4_t _sum1 = vmull_laneq_s16(vget_low_s16(_k1), _r0, 1);
                    int32x4_t _sum1n = vmull_laneq_s16(vget_high_s16(_k1), _r0, 1);
                    _sum03 = vmlal_laneq_s16(_sum03, vget_low_s16(_k2), _r0, 2);
                    _sum47 = vmlal_laneq_s16(_sum47, vget_high_s16(_k2), _r0, 2);

                    // k3 - k5
                    _k0_8 = vld1_s8(ktmp + 24); //(k03-k73)
                    _k1_8 = vld1_s8(ktmp + 32); //(k04-k74)
                    _k2_8 = vld1_s8(ktmp + 40); //(k05-k75)

                    _k0 = vmovl_s8(_k0_8);
                    _k1 = vmovl_s8(_k1_8);
                    _k2 = vmovl_s8(_k2_8);

                    _sum0 = vmlal_laneq_s16(_sum0, vget_low_s16(_k0), _r1, 0);
                    _sum0n = vmlal_laneq_s16(_sum0n, vget_high_s16(_k0), _r1, 0);
                    _sum1 = vmlal_laneq_s16(_sum1, vget_low_s16(_k1), _r1, 1);
                    _sum1n = vmlal_laneq_s16(_sum1n, vget_high_s16(_k1), _r1, 1);
                    _sum03 = vmlal_laneq_s16(_sum03, vget_low_s16(_k2), _r1, 2);
                    _sum47 = vmlal_laneq_s16(_sum47, vget_high_s16(_k2), _r1, 2);

                    // k6 - k8
                    _k0_8 = vld1_s8(ktmp + 48); //(k06-k76)
                    _k1_8 = vld1_s8(ktmp + 56); //(k07-k77)
                    _k2_8 = vld1_s8(ktmp + 64); //(k08-k78)

                    _k0 = vmovl_s8(_k0_8);
                    _k1 = vmovl_s8(_k1_8);
                    _k2 = vmovl_s8(_k2_8);

                    _sum0 = vmlal_laneq_s16(_sum0, vget_low_s16(_k0), _r2, 0);
                    _sum0n = vmlal_laneq_s16(_sum0n, vget_high_s16(_k0), _r2, 0);
                    _sum1 = vmlal_laneq_s16(_sum1, vget_low_s16(_k1), _r2, 1);
                    _sum1n = vmlal_laneq_s16(_sum1n, vget_high_s16(_k1), _r2, 1);
                    _sum03 = vmlal_laneq_s16(_sum03, vget_low_s16(_k2), _r2, 2);
                    _sum47 = vmlal_laneq_s16(_sum47, vget_high_s16(_k2), _r2, 2);

                    _sum0 = vaddq_s32(_sum0, _sum1);
                    _sum0n = vaddq_s32(_sum0n, _sum1n);
                    _sum03 = vaddq_s32(_sum03, _sum0);
                    _sum47 = vaddq_s32(_sum47, _sum0n);

                    vst1q_lane_s32(outptr0, _sum03, 0);
                    vst1q_lane_s32(outptr1, _sum03, 1);
                    vst1q_lane_s32(outptr2, _sum03, 2);
                    vst1q_lane_s32(outptr3, _sum03, 3);
                    vst1q_lane_s32(outptr4, _sum47, 0);
                    vst1q_lane_s32(outptr5, _sum47, 1);
                    vst1q_lane_s32(outptr6, _sum47, 2);
                    vst1q_lane_s32(outptr7, _sum47, 3);

                    outptr0++;
                    outptr1++;
                    outptr2++;
                    outptr3++;
                    outptr4++;
                    outptr5++;
                    outptr6++;
                    outptr7++;
#else  // __aarch64__
                    asm volatile(
                        "pld        [%8, #64]          \n"
                        "vld1.s8    {d0}, [%8]         \n" // d0(a00 a01 a02 ....)
                        "pld        [%9, #64]          \n"
                        "vld1.s8    {d2}, [%9]         \n" // d2(a10 a11 a12 ....)
                        "pld        [%10, #64]         \n"
                        "vld1.s8    {d4}, [%10]        \n" // d4(a20 a21 a22 ....)

                        "pld        [%11, #64]         \n"
                        "vld1.s8    {d6-d8}, [%11]!    \n" // d6(k00-k70) d7(k01-k71) d8(k02-k72)

                        "vmovl.s8   q0, d0             \n" // d0(a00 a01 a02 x)
                        "vmovl.s8   q1, d2             \n" // d2(a10 a11 a12 x)
                        "vmovl.s8   q2, d4             \n" // d4(a20 a21 a22 x)

                        "vmovl.s8   q5, d8             \n" // d10(k02-k32) d11(k42-k72)
                        "vmovl.s8   q4, d7             \n" // d8(k01-k31) d9(k41-k71)
                        "vmovl.s8   q3, d6             \n" // d6(k00-k30) d7(k40-k70)

                        "vld1.s32   {d20[0]}, [%0]     \n" // out0 q10
                        "vld1.s32   {d20[1]}, [%1]     \n" // out1
                        "vld1.s32   {d21[0]}, [%2]     \n" // out2
                        "vld1.s32   {d21[1]}, [%3]     \n" // out3

                        "pld        [%11, #64]         \n"
                        "vld1.s8    {d24-d26}, [%11]!  \n"
                        "vmovl.s8   q14, d26           \n" // d28(k05-k35) d29(k45-k75)
                        "vmovl.s8   q13, d25           \n" // d26(k04-k34) d27(k44-k74)
                        "vmovl.s8   q12, d24           \n" // d24(k03-k33) d25(k43-k73)

                        "vld1.s32   {d22[0]}, [%4]     \n" // out4 q11
                        "vld1.s32   {d22[1]}, [%5]     \n" // out5
                        "vld1.s32   {d23[0]}, [%6]     \n" // out6
                        "vld1.s32   {d23[1]}, [%7]     \n" // out7

                        "vmull.s16  q6, d6, d0[0]      \n" // a00 x (k00-k30)
                        "vmull.s16  q7, d7, d0[0]      \n" // a00 x (k40-k70)
                        "vmull.s16  q8, d8, d0[1]      \n" // a01 x (k01-k31)
                        "vmull.s16  q9, d9, d0[1]      \n" // a01 x (k41-k71)
                        "vmlal.s16  q10, d10, d0[2]    \n" // a02 x (k02-k32)
                        "vmlal.s16  q11, d11, d0[2]    \n" // a02 x (k42-k72)

                        "pld        [%11, #64]         \n"
                        "vld1.s8    {d6-d8}, [%11]!    \n"
                        "vmovl.s8   q5, d8             \n" // d10(k08-k38) d11(k48-k78)
                        "vmovl.s8   q4, d7             \n" // d8(k07-k37) d9(k47-k77)
                        "vmovl.s8   q3, d6             \n" // d6(k06-k36) d7(k46-k76)

                        "vmlal.s16  q6, d24, d2[0]     \n" // a10 x (k03-k33)
                        "vmlal.s16  q7, d25, d2[0]     \n" // a10 x (k43-k73)
                        "vmlal.s16  q8, d26, d2[1]     \n" // a11 x (k04-k34)
                        "vmlal.s16  q9, d27, d2[1]     \n" // a11 x (k44-k74)
                        "vmlal.s16  q10, d28, d2[2]    \n" // a12 x (k05-k35)
                        "vmlal.s16  q11, d29, d2[2]    \n" // a12 x (k45-k75)

                        "vmlal.s16  q6, d6, d4[0]      \n" // a20 x (k06-k36)
                        "vmlal.s16  q7, d7, d4[0]      \n" // a20 x (k46-k76)
                        "vmlal.s16  q8, d8, d4[1]      \n" // a21 x (k07-k37)
                        "vmlal.s16  q9, d9, d4[1]      \n" // a21 x (k47-k77)
                        "vmlal.s16  q10, d10, d4[2]    \n" // a22 x (k08-k38)
                        "vmlal.s16  q11, d11, d4[2]    \n" // a22 x (k48-k78)

                        "vadd.s32   q8, q8, q6         \n"
                        "vadd.s32   q9, q9, q7         \n"

                        "sub        %11, %11, #72      \n"

                        "vadd.s32   q10, q10, q8       \n"
                        "vadd.s32   q11, q11, q9       \n"

                        "vst1.s32   {d20[0]}, [%0]!    \n" // out0
                        "vst1.s32   {d20[1]}, [%1]!    \n" // out1
                        "vst1.s32   {d21[0]}, [%2]!    \n" // out2
                        "vst1.s32   {d21[1]}, [%3]!    \n" // out3
                        "vst1.s32   {d22[0]}, [%4]!    \n" // out4
                        "vst1.s32   {d22[1]}, [%5]!    \n" // out5
                        "vst1.s32   {d23[0]}, [%6]!    \n" // out6
                        "vst1.s32   {d23[1]}, [%7]!    \n" // out7

                        : "=r"(outptr0), // %0
                        "=r"(outptr1), // %1
                        "=r"(outptr2), // %2
                        "=r"(outptr3), // %3
                        "=r"(outptr4), // %4
                        "=r"(outptr5), // %5
                        "=r"(outptr6), // %6
                        "=r"(outptr7), // %7
                        "=r"(r0),      // %8
                        "=r"(r1),      // %9
                        "=r"(r2),      // %10
                        "=r"(ktmp)     // %11
                        : "0"(outptr0),
                        "1"(outptr1),
                        "2"(outptr2),
                        "3"(outptr3),
                        "4"(outptr4),
                        "5"(outptr5),
                        "6"(outptr6),
                        "7"(outptr7),
                        "8"(r0),
                        "9"(r1),
                        "10"(r2),
                        "11"(ktmp)
                        : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif // __aarch64__
#else  // __ARM_NEON
                    int sum0 = 0;
                    int sum1 = 0;
                    int sum2 = 0;
                    int sum3 = 0;
                    int sum4 = 0;
                    int sum5 = 0;
                    int sum6 = 0;
                    int sum7 = 0;

                    sum0 += (int)r0[0] * ktmp[0];
                    sum1 += (int)r0[0] * ktmp[1];
                    sum2 += (int)r0[0] * ktmp[2];
                    sum3 += (int)r0[0] * ktmp[3];
                    sum4 += (int)r0[0] * ktmp[4];
                    sum5 += (int)r0[0] * ktmp[5];
                    sum6 += (int)r0[0] * ktmp[6];
                    sum7 += (int)r0[0] * ktmp[7];
                    ktmp += 8;

                    sum0 += (int)r0[1] * ktmp[0];
                    sum1 += (int)r0[1] * ktmp[1];
                    sum2 += (int)r0[1] * ktmp[2];
                    sum3 += (int)r0[1] * ktmp[3];
                    sum4 += (int)r0[1] * ktmp[4];
                    sum5 += (int)r0[1] * ktmp[5];
                    sum6 += (int)r0[1] * ktmp[6];
                    sum7 += (int)r0[1] * ktmp[7];
                    ktmp += 8;

                    sum0 += (int)r0[2] * ktmp[0];
                    sum1 += (int)r0[2] * ktmp[1];
                    sum2 += (int)r0[2] * ktmp[2];
                    sum3 += (int)r0[2] * ktmp[3];
                    sum4 += (int)r0[2] * ktmp[4];
                    sum5 += (int)r0[2] * ktmp[5];
                    sum6 += (int)r0[2] * ktmp[6];
                    sum7 += (int)r0[2] * ktmp[7];
                    ktmp += 8;

                    sum0 += (int)r1[0] * ktmp[0];
                    sum1 += (int)r1[0] * ktmp[1];
                    sum2 += (int)r1[0] * ktmp[2];
                    sum3 += (int)r1[0] * ktmp[3];
                    sum4 += (int)r1[0] * ktmp[4];
                    sum5 += (int)r1[0] * ktmp[5];
                    sum6 += (int)r1[0] * ktmp[6];
                    sum7 += (int)r1[0] * ktmp[7];
                    ktmp += 8;

                    sum0 += (int)r1[1] * ktmp[0];
                    sum1 += (int)r1[1] * ktmp[1];
                    sum2 += (int)r1[1] * ktmp[2];
                    sum3 += (int)r1[1] * ktmp[3];
                    sum4 += (int)r1[1] * ktmp[4];
                    sum5 += (int)r1[1] * ktmp[5];
                    sum6 += (int)r1[1] * ktmp[6];
                    sum7 += (int)r1[1] * ktmp[7];
                    ktmp += 8;

                    sum0 += (int)r1[2] * ktmp[0];
                    sum1 += (int)r1[2] * ktmp[1];
                    sum2 += (int)r1[2] * ktmp[2];
                    sum3 += (int)r1[2] * ktmp[3];
                    sum4 += (int)r1[2] * ktmp[4];
                    sum5 += (int)r1[2] * ktmp[5];
                    sum6 += (int)r1[2] * ktmp[6];
                    sum7 += (int)r1[2] * ktmp[7];
                    ktmp += 8;

                    sum0 += (int)r2[0] * ktmp[0];
                    sum1 += (int)r2[0] * ktmp[1];
                    sum2 += (int)r2[0] * ktmp[2];
                    sum3 += (int)r2[0] * ktmp[3];
                    sum4 += (int)r2[0] * ktmp[4];
                    sum5 += (int)r2[0] * ktmp[5];
                    sum6 += (int)r2[0] * ktmp[6];
                    sum7 += (int)r2[0] * ktmp[7];
                    ktmp += 8;

                    sum0 += (int)r2[1] * ktmp[0];
                    sum1 += (int)r2[1] * ktmp[1];
                    sum2 += (int)r2[1] * ktmp[2];
                    sum3 += (int)r2[1] * ktmp[3];
                    sum4 += (int)r2[1] * ktmp[4];
                    sum5 += (int)r2[1] * ktmp[5];
                    sum6 += (int)r2[1] * ktmp[6];
                    sum7 += (int)r2[1] * ktmp[7];
                    ktmp += 8;

                    sum0 += (int)r2[2] * ktmp[0];
                    sum1 += (int)r2[2] * ktmp[1];
                    sum2 += (int)r2[2] * ktmp[2];
                    sum3 += (int)r2[2] * ktmp[3];
                    sum4 += (int)r2[2] * ktmp[4];
                    sum5 += (int)r2[2] * ktmp[5];
                    sum6 += (int)r2[2] * ktmp[6];
                    sum7 += (int)r2[2] * ktmp[7];
                    ktmp += 8;

                    *outptr0 += sum0;
                    *outptr1 += sum1;
                    *outptr2 += sum2;
                    *outptr3 += sum3;
                    *outptr4 += sum4;
                    *outptr5 += sum5;
                    *outptr6 += sum6;
                    *outptr7 += sum7;

                    ktmp -= 8 * 9;

                    outptr0++;
                    outptr1++;
                    outptr2++;
                    outptr3++;
                    outptr4++;
                    outptr5++;
                    outptr6++;
                    outptr7++;
#endif // __ARM_NEON
                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }

            ktmp += 8 * 9;
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = remain_outch_start; p < outch; p++)
    {
        Mat out = top_blob.channel(p);

        out.fill(0);

        const signed char* ktmp = _kernel.channel(p / 8 + p % 8);

        for (int q = 0; q < inch; q++)
        {
            int* outptr = out;

            const signed char* img0 = bottom_blob.channel(q);

            const signed char* r0 = img0;
            const signed char* r1 = img0 + w;
            const signed char* r2 = img0 + w * 2;

            int i = 0;

            for (; i < outh; i++)
            {
#if __ARM_NEON
                int nn = outw >> 3;
                int remain = outw & 7;
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                if (nn > 0)
                {
                    asm volatile(
                        "0:                                   \n"

                        "ld1    {v0.8b, v1.8b}, [%5]          \n" //ktmp
                        "ld2    {v2.8b, v3.8b}, [%2], #16     \n" //r0-r2
                        "ld2    {v4.8b, v5.8b}, [%2]          \n"

                        "ld2    {v6.8b, v7.8b}, [%3], #16     \n" //r3-r5
                        "ld2    {v8.8b, v9.8b}, [%3]          \n"

                        "ld2    {v10.8b, v11.8b}, [%4], #16   \n" //r6-r8
                        "ld2    {v12.8b, v13.8b}, [%4]        \n"

                        "ld1    {v14.4s, v15.4s}, [%1]        \n" //out0

                        "ext    v4.8b, v2.8b, v4.8b, #1       \n"
                        "ext    v8.8b, v6.8b, v8.8b, #1       \n"
                        "ext    v12.8b, v10.8b, v12.8b, #1    \n"

                        "sshll  v0.8h, v0.8b, #0              \n" //(k0-k7)
                        "sshll  v1.8h, v1.8b, #0              \n" //(k8)
                        "sshll  v2.8h, v2.8b, #0              \n" // r0
                        "sshll  v3.8h, v3.8b, #0              \n" // r1
                        "sshll  v4.8h, v4.8b, #0              \n" // r2
                        "sshll  v6.8h, v6.8b, #0              \n" // r3
                        "sshll  v7.8h, v7.8b, #0              \n" // r4
                        "sshll  v8.8h, v8.8b, #0              \n" // r5
                        "sshll  v10.8h, v10.8b, #0            \n" // r6
                        "sshll  v11.8h, v11.8b, #0            \n" // r7
                        "sshll  v12.8h, v12.8b, #0            \n" // r8

                        // r0
                        "smull  v16.4s, v2.4h, v0.h[0]        \n" // out = r0*k0
                        "smull2  v17.4s, v2.8h, v0.h[0]       \n"
                        "smull  v18.4s, v3.4h, v0.h[1]        \n" // outn = r1*k1
                        "smull2  v19.4s, v3.8h, v0.h[1]       \n"
                        "smlal  v16.4s, v4.4h, v0.h[2]        \n" // out = r2*k2
                        "smlal2  v17.4s, v4.8h, v0.h[2]       \n"
                        "smlal  v18.4s, v6.4h, v0.h[3]        \n" // outn = r3*k3
                        "smlal2  v19.4s, v6.8h, v0.h[3]       \n"
                        "smlal  v16.4s, v7.4h, v0.h[4]        \n" // out = r4*k4
                        "smlal2  v17.4s, v7.8h, v0.h[4]       \n"
                        "smlal  v18.4s, v8.4h, v0.h[5]        \n" // outn = r5*k5
                        "smlal2  v19.4s, v8.8h, v0.h[5]       \n"
                        "smlal  v16.4s, v10.4h, v0.h[6]       \n" // out = r6*k6
                        "smlal2  v17.4s, v10.8h, v0.h[6]      \n"
                        "smlal  v18.4s, v11.4h, v0.h[7]       \n" // outn = r7*k7
                        "smlal2  v19.4s, v11.8h, v0.h[7]      \n"
                        "smlal  v16.4s, v12.4h, v1.h[0]       \n" // out = r8*k8
                        "smlal2  v17.4s, v12.8h, v1.h[0]      \n"

                        "add    v8.4s, v16.4s, v18.4s         \n"
                        "add    v9.4s, v17.4s, v19.4s         \n"

                        "st1    {v8.4s, v9.4s}, [%1], #32     \n"

                        "subs   %w0, %w0, #1                  \n"

                        "bne    0b                            \n"

                        : "=r"(nn),     // %0
                        "=r"(outptr), // %1
                        "=r"(r0),     // %2
                        "=r"(r1),     // %3
                        "=r"(r2),     // %4
                        "=r"(ktmp)    // %5
                        : "0"(nn),
                        "1"(outptr),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "5"(ktmp)
                        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19");
                }
#else
                if (nn > 0)
                {
                    asm volatile(
                        "vld1.s8    {d0-d1}, [%5]       \n" // d0(k0 - k7) d1(k8 ...)
                        "vmovl.s8   q1, d1              \n" // d2(k8 ...)
                        "vmovl.s8   q0, d0              \n" // d0(k0 - k3) d1(k4 - k7)
                        "0:                             \n"
                        "pld        [%2, #192]          \n"
                        "vld2.s8    {d4-d5}, [%2]!      \n" // r0 d4(a00 a02 ... a014) d5(a01 a03 ... a015)
                        "vld2.s8    {d8-d9}, [%2]       \n" //    d8(a016 ....)
                        "vld2.s8    {d10-d11}, [%3]!    \n" // r1 d10(a10 a12 ... a114) d11(a11 a13 ... a115)
                        "vld2.s8    {d14-d15}, [%3]     \n" //    d14(a116 ....)
                        "vld2.s8    {d16-d17}, [%4]!    \n" // r2 d16(a20 a22 ... a214) d17(a21 a23 ... a215)
                        "vld2.s8    {d20-d21}, [%4]     \n" //    d20(a216 ....)
                        "vld1.s32   {d22-d25}, [%1]     \n" // q11(out0 - out3) q12(out4 - out7)

                        "vext.s8    d8, d4, d8, #1      \n" //  d8(a02 a04 ... a016)
                        "vext.s8    d14, d10, d14, #1   \n" // d14(a12 a14 ... a116)
                        "vext.s8    d20, d16, d20, #1   \n" // d20(a22 a24 ... a216)

                        "vmovl.s8   q3, d5              \n" // q3(a01 a03 ... a015)
                        "vmovl.s8   q2, d4              \n" // q2(a00 a02 ... a014)
                        "vmovl.s8   q4, d8              \n" // q4(a02 a04 ... a016)

                        "vmovl.s8   q6, d11             \n" // q6(a11 a13 ... a115)
                        "vmovl.s8   q5, d10             \n" // q5(a10 a12 ... a114)
                        "vmovl.s8   q7, d14             \n" // q7(a12 a14 ... a116)

                        "vmovl.s8   q9, d17             \n" // q9(a21 a23 ... a215)
                        "vmovl.s8   q8, d16             \n" // q8(a20 a22 ... a214)
                        "vmovl.s8   q10, d20            \n" // q10(a22 a24 ... a216)

                        "vmlal.s16  q11, d4, d0[0]      \n" // k0
                        "vmlal.s16  q12, d5, d0[0]      \n"
                        "vmull.s16  q13, d6, d0[1]      \n" // k1
                        "vmull.s16  q14, d7, d0[1]      \n"
                        "vmlal.s16  q11, d8, d0[2]      \n" // k2
                        "vmlal.s16  q12, d9, d0[2]      \n"

                        "vmlal.s16  q13, d12, d1[0]     \n" // k4
                        "vmlal.s16  q14, d13, d1[0]     \n"
                        "vmlal.s16  q11, d10, d0[3]     \n" // k3
                        "vmlal.s16  q12, d11, d0[3]     \n"
                        "vmlal.s16  q13, d14, d1[1]     \n" // k5
                        "vmlal.s16  q14, d15, d1[1]     \n"

                        "vmlal.s16  q11, d16, d1[2]     \n" // k6
                        "vmlal.s16  q12, d17, d1[2]     \n"
                        "vmlal.s16  q13, d18, d1[3]     \n" // k7
                        "vmlal.s16  q14, d19, d1[3]     \n"
                        "vmlal.s16  q11, d20, d2[0]     \n" // k8
                        "vmlal.s16  q12, d21, d2[0]     \n"

                        "vadd.s32   q11, q11, q13       \n"
                        "vadd.s32   q12, q12, q14       \n"

                        "vst1.32    {d22-d25}, [%1]!    \n"

                        "subs       %0, #1              \n"
                        "bne        0b                  \n"
                        : "=r"(nn),     // %0
                        "=r"(outptr), // %1
                        "=r"(r0),     // %2
                        "=r"(r1),     // %3
                        "=r"(r2),     // %4
                        "=r"(ktmp)    // %5
                        : "0"(nn),
                        "1"(outptr),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "5"(ktmp)
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
                }
#endif // __aarch64__
#endif // __ARM_NEON
                if (remain > 0)
                {
#if __ARM_NEON
                    int8x8_t _k01234567s8 = vld1_s8(ktmp);
                    int8x8_t _k8xxxxxxxs8 = vld1_s8(ktmp + 8);
                    int8x8_t _k34567xxxs8 = vext_s8(_k01234567s8, _k01234567s8, 3);
                    int8x8_t _k678xxxxxs8 = vext_s8(_k01234567s8, _k8xxxxxxxs8, 6);
                    int16x8_t _k0123_s16 = vmovl_s8(_k01234567s8);
                    int16x8_t _k3456_s16 = vmovl_s8(_k34567xxxs8);
                    int16x8_t _k678x_s16 = vmovl_s8(_k678xxxxxs8);
#endif
                    for (; remain > 0; remain--)
                    {
#if __ARM_NEON
                        int8x8_t _r00s8 = vld1_s8(r0);
                        int8x8_t _r10s8 = vld1_s8(r1);
                        int8x8_t _r20s8 = vld1_s8(r2);

                        int16x8_t _r00s16 = vmovl_s8(_r00s8);
                        int16x8_t _r10s16 = vmovl_s8(_r10s8);
                        int16x8_t _r20s16 = vmovl_s8(_r20s8);

                        int32x4_t _sum = vmull_s16(vget_low_s16(_r00s16), vget_low_s16(_k0123_s16));
                        _sum = vmlal_s16(_sum, vget_low_s16(_r10s16), vget_low_s16(_k3456_s16));
                        _sum = vmlal_s16(_sum, vget_low_s16(_r20s16), vget_low_s16(_k678x_s16));

                        _sum = vsetq_lane_s32(*outptr, _sum, 3);

#if __aarch64__
                        *outptr = vaddvq_s32(_sum);
#else
                        int32x2_t _ss = vadd_s32(vget_low_s32(_sum), vget_high_s32(_sum));
                        _ss = vpadd_s32(_ss, _ss);

                        *outptr = vget_lane_s32(_ss, 0);
#endif // __aarch64__
#else
                        int sum = 0;

                        sum += (int)r0[0] * ktmp[0];
                        sum += (int)r0[1] * ktmp[1];
                        sum += (int)r0[2] * ktmp[2];
                        sum += (int)r1[0] * ktmp[3];
                        sum += (int)r1[1] * ktmp[4];
                        sum += (int)r1[2] * ktmp[5];
                        sum += (int)r2[0] * ktmp[6];
                        sum += (int)r2[1] * ktmp[7];
                        sum += (int)r2[2] * ktmp[8];

                        *outptr += sum;
#endif // __ARM_NEON
                        r0 += 2;
                        r1 += 2;
                        r2 += 2;
                        outptr++;
                    }
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }

            ktmp += 9;
        }
    }
}
