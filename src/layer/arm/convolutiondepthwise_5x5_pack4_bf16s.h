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

static void convdw5x5s2_pack4_bf16s_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int group = bottom_blob.c;

    const int tailstep = (w - 2*outw + w) * 4;

    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g=0; g<group; g++)
    {
        Mat out = top_blob.channel(g);

#if __aarch64__
        float32x4_t _bias0 = bias ? vld1q_f32((const float*)bias + g * 4) : vdupq_n_f32(0.f);
#endif // __aarch64__

        const unsigned short* kptr = kernel.row<const unsigned short>(g);

        unsigned short* outptr0 = out;

        const Mat img0 = bottom_blob.channel(g);

        const unsigned short* r0 = img0.row<const unsigned short>(0);
        const unsigned short* r1 = img0.row<const unsigned short>(1);
        const unsigned short* r2 = img0.row<const unsigned short>(2);
        const unsigned short* r3 = img0.row<const unsigned short>(3);
        const unsigned short* r4 = img0.row<const unsigned short>(4);

#if __aarch64__
        // 4 * 25
        uint16x8_t _k00_01 = vld1q_u16(kptr);
        uint16x8_t _k02_03 = vld1q_u16(kptr+8);
        uint16x8_t _k04_10 = vld1q_u16(kptr+16);
        uint16x8_t _k11_12 = vld1q_u16(kptr+24);
        uint16x8_t _k13_14 = vld1q_u16(kptr+32);
        uint16x8_t _k20_21 = vld1q_u16(kptr+40);
        uint16x8_t _k22_23 = vld1q_u16(kptr+48);
        uint16x8_t _k24_30 = vld1q_u16(kptr+56);
        uint16x8_t _k31_32 = vld1q_u16(kptr+64);
        uint16x8_t _k33_34 = vld1q_u16(kptr+72);
        uint16x8_t _k40_41 = vld1q_u16(kptr+80);
        uint16x8_t _k42_43 = vld1q_u16(kptr+88);
        uint16x4_t _k44 = vld1_u16(kptr+96);
#else // __aarch64__
        float bias0_data[4];
        if (bias)
        {
            bias0_data[0] = bias[g * 4 + 0];
            bias0_data[1] = bias[g * 4 + 1];
            bias0_data[2] = bias[g * 4 + 2];
            bias0_data[3] = bias[g * 4 + 3];
        }
        else
        {
            bias0_data[0] = 0.f;
            bias0_data[1] = 0.f;
            bias0_data[2] = 0.f;
            bias0_data[3] = 0.f;
        }
        const float* bias0_data_ptr = bias0_data;
#endif // __aarch64__

        int i = 0;

        for (; i < outh; i++)
        {
            int j = 0;

            for (; j+3 < outw; j+=4)
            {
#if __aarch64__
                asm volatile(
                    "prfm   pldl1keep, [%1, #256]       \n"
                    "ld1    {v16.4h, v17.4h, v18.4h, v19.4h}, [%1], #32 \n"// r00 r01 r02 r03

                    "prfm   pldl1keep, [%1, #256]       \n"
                    "ld1    {v20.4h, v21.4h, v22.4h, v23.4h}, [%1], #32 \n"// r04 r05 r06 r07

                    "shll   v14.4s, %12.4h, #16         \n"

                    "mov    v28.16b, %25.16b            \n"// sum00
                    "mov    v29.16b, %25.16b            \n"// sum01
                    "mov    v30.16b, %25.16b            \n"// sum02
                    "mov    v31.16b, %25.16b            \n"// sum03

                    "shll   v16.4s, v16.4h, #16         \n"
                    "shll   v17.4s, v17.4h, #16         \n"
                    "shll   v18.4s, v18.4h, #16         \n"

                    "prfm   pldl1keep, [%1, #192]       \n"
                    "ld1    {v24.4h, v25.4h, v26.4h}, [%1] \n"// r08 r09 r010

                    "shll2  v15.4s, %12.8h, #16         \n"
                    "fmla   v28.4s, v14.4s, v16.4s      \n"
                    "shll   v20.4s, v20.4h, #16         \n"
                    "fmla   v29.4s, v14.4s, v18.4s      \n"
                    "shll   v22.4s, v22.4h, #16         \n"
                    "fmla   v30.4s, v14.4s, v20.4s      \n"
                    "shll   v19.4s, v19.4h, #16         \n"
                    "fmla   v31.4s, v14.4s, v22.4s      \n"
                    "shll   v14.4s, %13.4h, #16         \n"
                    "fmla   v28.4s, v15.4s, v17.4s      \n"
                    "shll   v21.4s, v21.4h, #16         \n"
                    "fmla   v29.4s, v15.4s, v19.4s      \n"
                    "shll   v23.4s, v23.4h, #16         \n"
                    "fmla   v30.4s, v15.4s, v21.4s      \n"
                    "fmla   v31.4s, v15.4s, v23.4s      \n"
                    "shll2  v15.4s, %13.8h, #16         \n"
                    "fmla   v28.4s, v14.4s, v18.4s      \n"
                    "fmla   v29.4s, v14.4s, v20.4s      \n"
                    "shll   v24.4s, v24.4h, #16         \n"
                    "fmla   v30.4s, v14.4s, v22.4s      \n"
                    "fmla   v31.4s, v14.4s, v24.4s      \n"
                    "shll   v14.4s, %14.4h, #16         \n"
                    "fmla   v28.4s, v15.4s, v19.4s      \n"
                    "prfm   pldl1keep, [%2, #256]       \n"
                    "ld1    {v16.4h, v17.4h, v18.4h, v19.4h}, [%2], #32 \n"// r10 r11 r12 r13

                    "fmla   v29.4s, v15.4s, v21.4s      \n"
                    "shll   v25.4s, v25.4h, #16         \n"
                    "fmla   v30.4s, v15.4s, v23.4s      \n"
                    "shll   v16.4s, v16.4h, #16         \n"
                    "fmla   v31.4s, v15.4s, v25.4s      \n"
                    "shll2  v15.4s, %14.8h, #16         \n"
                    "fmla   v28.4s, v14.4s, v20.4s      \n"
                    "shll   v17.4s, v17.4h, #16         \n"
                    "fmla   v29.4s, v14.4s, v22.4s      \n"
                    "prfm   pldl1keep, [%2, #256]       \n"
                    "ld1    {v20.4h, v21.4h, v22.4h, v23.4h}, [%2], #32 \n"// r14 r15 r16 r17

                    "shll   v26.4s, v26.4h, #16         \n"
                    "fmla   v30.4s, v14.4s, v24.4s      \n"
                    "shll   v18.4s, v18.4h, #16         \n"
                    "fmla   v31.4s, v14.4s, v26.4s      \n"

                    "prfm   pldl1keep, [%2, #192]       \n"
                    "ld1    {v24.4h, v25.4h, v26.4h}, [%2] \n"// r18 r19 r110

                    "shll   v14.4s, %15.4h, #16         \n"
                    "fmla   v28.4s, v15.4s, v16.4s      \n"
                    "shll   v20.4s, v20.4h, #16         \n"
                    "fmla   v29.4s, v15.4s, v18.4s      \n"
                    "shll   v22.4s, v22.4h, #16         \n"
                    "fmla   v30.4s, v15.4s, v20.4s      \n"
                    "shll   v19.4s, v19.4h, #16         \n"
                    "fmla   v31.4s, v15.4s, v22.4s      \n"
                    "shll2  v15.4s, %15.8h, #16         \n"
                    "fmla   v28.4s, v14.4s, v17.4s      \n"
                    "shll   v21.4s, v21.4h, #16         \n"
                    "fmla   v29.4s, v14.4s, v19.4s      \n"
                    "shll   v23.4s, v23.4h, #16         \n"
                    "fmla   v30.4s, v14.4s, v21.4s      \n"
                    "fmla   v31.4s, v14.4s, v23.4s      \n"
                    "shll   v14.4s, %16.4h, #16         \n"
                    "fmla   v28.4s, v15.4s, v18.4s      \n"
                    "fmla   v29.4s, v15.4s, v20.4s      \n"
                    "shll   v24.4s, v24.4h, #16         \n"
                    "fmla   v30.4s, v15.4s, v22.4s      \n"
                    "fmla   v31.4s, v15.4s, v24.4s      \n"
                    "shll2  v15.4s, %16.8h, #16         \n"
                    "fmla   v28.4s, v14.4s, v19.4s      \n"
                    "prfm   pldl1keep, [%3, #256]       \n"
                    "ld1    {v16.4h, v17.4h, v18.4h, v19.4h}, [%3], #32 \n"// r20 r21 r22 r23

                    "fmla   v29.4s, v14.4s, v21.4s      \n"
                    "shll   v25.4s, v25.4h, #16         \n"
                    "fmla   v30.4s, v14.4s, v23.4s      \n"
                    "shll   v16.4s, v16.4h, #16         \n"
                    "fmla   v31.4s, v14.4s, v25.4s      \n"
                    "shll   v14.4s, %17.4h, #16         \n"
                    "fmla   v28.4s, v15.4s, v20.4s      \n"
                    "shll   v17.4s, v17.4h, #16         \n"
                    "fmla   v29.4s, v15.4s, v22.4s      \n"
                    "prfm   pldl1keep, [%3, #256]       \n"
                    "ld1    {v20.4h, v21.4h, v22.4h, v23.4h}, [%3], #32 \n"// r24 r25 r26 r27

                    "shll   v26.4s, v26.4h, #16         \n"
                    "fmla   v30.4s, v15.4s, v24.4s      \n"
                    "shll   v18.4s, v18.4h, #16         \n"
                    "fmla   v31.4s, v15.4s, v26.4s      \n"

                    "prfm   pldl1keep, [%3, #192]       \n"
                    "ld1    {v24.4h, v25.4h, v26.4h}, [%3] \n"// r28 r29 r210

                    "shll2  v15.4s, %17.8h, #16         \n"
                    "fmla   v28.4s, v14.4s, v16.4s      \n"
                    "shll   v20.4s, v20.4h, #16         \n"
                    "fmla   v29.4s, v14.4s, v18.4s      \n"
                    "shll   v22.4s, v22.4h, #16         \n"
                    "fmla   v30.4s, v14.4s, v20.4s      \n"
                    "shll   v19.4s, v19.4h, #16         \n"
                    "fmla   v31.4s, v14.4s, v22.4s      \n"
                    "shll   v14.4s, %18.4h, #16         \n"
                    "fmla   v28.4s, v15.4s, v17.4s      \n"
                    "shll   v21.4s, v21.4h, #16         \n"
                    "fmla   v29.4s, v15.4s, v19.4s      \n"
                    "shll   v23.4s, v23.4h, #16         \n"
                    "fmla   v30.4s, v15.4s, v21.4s      \n"
                    "fmla   v31.4s, v15.4s, v23.4s      \n"
                    "shll2  v15.4s, %18.8h, #16         \n"
                    "fmla   v28.4s, v14.4s, v18.4s      \n"
                    "fmla   v29.4s, v14.4s, v20.4s      \n"
                    "shll   v24.4s, v24.4h, #16         \n"
                    "fmla   v30.4s, v14.4s, v22.4s      \n"
                    "fmla   v31.4s, v14.4s, v24.4s      \n"
                    "shll   v14.4s, %19.4h, #16         \n"
                    "fmla   v28.4s, v15.4s, v19.4s      \n"
                    "prfm   pldl1keep, [%4, #256]       \n"
                    "ld1    {v16.4h, v17.4h, v18.4h, v19.4h}, [%4], #32 \n"// r30 r31 r32 r33

                    "fmla   v29.4s, v15.4s, v21.4s      \n"
                    "shll   v25.4s, v25.4h, #16         \n"
                    "fmla   v30.4s, v15.4s, v23.4s      \n"
                    "shll   v16.4s, v16.4h, #16         \n"
                    "fmla   v31.4s, v15.4s, v25.4s      \n"
                    "shll2  v15.4s, %19.8h, #16         \n"
                    "fmla   v28.4s, v14.4s, v20.4s      \n"
                    "shll   v17.4s, v17.4h, #16         \n"
                    "fmla   v29.4s, v14.4s, v22.4s      \n"
                    "prfm   pldl1keep, [%4, #256]       \n"
                    "ld1    {v20.4h, v21.4h, v22.4h, v23.4h}, [%4], #32 \n"// r34 r35 r36 r37

                    "shll   v26.4s, v26.4h, #16         \n"
                    "fmla   v30.4s, v14.4s, v24.4s      \n"
                    "shll   v18.4s, v18.4h, #16         \n"
                    "fmla   v31.4s, v14.4s, v26.4s      \n"

                    "prfm   pldl1keep, [%4, #192]       \n"
                    "ld1    {v24.4h, v25.4h, v26.4h}, [%4] \n"// r38 r39 r310

                    "shll   v14.4s, %20.4h, #16         \n"
                    "fmla   v28.4s, v15.4s, v16.4s      \n"
                    "shll   v20.4s, v20.4h, #16         \n"
                    "fmla   v29.4s, v15.4s, v18.4s      \n"
                    "shll   v22.4s, v22.4h, #16         \n"
                    "fmla   v30.4s, v15.4s, v20.4s      \n"
                    "shll   v19.4s, v19.4h, #16         \n"
                    "fmla   v31.4s, v15.4s, v22.4s      \n"
                    "shll2  v15.4s, %20.8h, #16         \n"
                    "fmla   v28.4s, v14.4s, v17.4s      \n"
                    "shll   v21.4s, v21.4h, #16         \n"
                    "fmla   v29.4s, v14.4s, v19.4s      \n"
                    "shll   v23.4s, v23.4h, #16         \n"
                    "fmla   v30.4s, v14.4s, v21.4s      \n"
                    "fmla   v31.4s, v14.4s, v23.4s      \n"
                    "shll   v14.4s, %21.4h, #16         \n"
                    "fmla   v28.4s, v15.4s, v18.4s      \n"
                    "fmla   v29.4s, v15.4s, v20.4s      \n"
                    "shll   v24.4s, v24.4h, #16         \n"
                    "fmla   v30.4s, v15.4s, v22.4s      \n"
                    "fmla   v31.4s, v15.4s, v24.4s      \n"
                    "shll2  v15.4s, %21.8h, #16         \n"
                    "fmla   v28.4s, v14.4s, v19.4s      \n"
                    "prfm   pldl1keep, [%5, #256]       \n"
                    "ld1    {v16.4h, v17.4h, v18.4h, v19.4h}, [%5], #32 \n"// r40 r41 r42 r43

                    "fmla   v29.4s, v14.4s, v21.4s      \n"
                    "shll   v25.4s, v25.4h, #16         \n"
                    "fmla   v30.4s, v14.4s, v23.4s      \n"
                    "shll   v16.4s, v16.4h, #16         \n"
                    "fmla   v31.4s, v14.4s, v25.4s      \n"
                    "shll   v14.4s, %22.4h, #16         \n"
                    "fmla   v28.4s, v15.4s, v20.4s      \n"
                    "shll   v17.4s, v17.4h, #16         \n"
                    "fmla   v29.4s, v15.4s, v22.4s      \n"
                    "prfm   pldl1keep, [%5, #256]       \n"
                    "ld1    {v20.4h, v21.4h, v22.4h, v23.4h}, [%5], #32 \n"// r44 r45 r46 r47

                    "shll   v26.4s, v26.4h, #16         \n"
                    "fmla   v30.4s, v15.4s, v24.4s      \n"
                    "shll   v18.4s, v18.4h, #16         \n"
                    "fmla   v31.4s, v15.4s, v26.4s      \n"

                    "prfm   pldl1keep, [%5, #192]       \n"
                    "ld1    {v24.4h, v25.4h, v26.4h}, [%5] \n"// r48 r49 r410

                    "shll2  v15.4s, %22.8h, #16         \n"
                    "fmla   v28.4s, v14.4s, v16.4s      \n"
                    "shll   v20.4s, v20.4h, #16         \n"
                    "fmla   v29.4s, v14.4s, v18.4s      \n"
                    "shll   v22.4s, v22.4h, #16         \n"
                    "fmla   v30.4s, v14.4s, v20.4s      \n"
                    "shll   v19.4s, v19.4h, #16         \n"
                    "fmla   v31.4s, v14.4s, v22.4s      \n"
                    "shll   v14.4s, %23.4h, #16         \n"
                    "fmla   v28.4s, v15.4s, v17.4s      \n"
                    "shll   v21.4s, v21.4h, #16         \n"
                    "fmla   v29.4s, v15.4s, v19.4s      \n"
                    "shll   v23.4s, v23.4h, #16         \n"
                    "fmla   v30.4s, v15.4s, v21.4s      \n"
                    "fmla   v31.4s, v15.4s, v23.4s      \n"
                    "shll2  v15.4s, %23.8h, #16         \n"
                    "fmla   v28.4s, v14.4s, v18.4s      \n"
                    "fmla   v29.4s, v14.4s, v20.4s      \n"
                    "shll   v24.4s, v24.4h, #16         \n"
                    "fmla   v30.4s, v14.4s, v22.4s      \n"
                    "fmla   v31.4s, v14.4s, v24.4s      \n"
                    "shll   v14.4s, %24.4h, #16         \n"
                    "fmla   v28.4s, v15.4s, v19.4s      \n"
                    "fmla   v29.4s, v15.4s, v21.4s      \n"
                    "shll   v25.4s, v25.4h, #16         \n"
                    "fmla   v30.4s, v15.4s, v23.4s      \n"
                    "fmla   v31.4s, v15.4s, v25.4s      \n"
                    "fmla   v28.4s, v14.4s, v20.4s      \n"
                    "fmla   v29.4s, v14.4s, v22.4s      \n"
                    "shll   v26.4s, v26.4h, #16         \n"
                    "fmla   v30.4s, v14.4s, v24.4s      \n"
                    "fmla   v31.4s, v14.4s, v26.4s      \n"

                    "shrn   v28.4h, v28.4s, #16         \n"
                    "shrn   v29.4h, v29.4s, #16         \n"
                    "shrn   v30.4h, v30.4s, #16         \n"
                    "shrn   v31.4h, v31.4s, #16         \n"

                    "st1    {v28.4h, v29.4h, v30.4h, v31.4h}, [%0], #32 \n"

                    : "=r"(outptr0),    // %0
                      "=r"(r0),         // %1
                      "=r"(r1),         // %2
                      "=r"(r2),         // %3
                      "=r"(r3),         // %4
                      "=r"(r4)          // %5
                    : "0"(outptr0),
                      "1"(r0),
                      "2"(r1),
                      "3"(r2),
                      "4"(r3),
                      "5"(r4),
                      "w"(_k00_01),     // %12
                      "w"(_k02_03),     // %13
                      "w"(_k04_10),     // %14
                      "w"(_k11_12),     // %15
                      "w"(_k13_14),     // %16
                      "w"(_k20_21),     // %17
                      "w"(_k22_23),     // %18
                      "w"(_k24_30),     // %19
                      "w"(_k31_32),     // %20
                      "w"(_k33_34),     // %21
                      "w"(_k40_41),     // %22
                      "w"(_k42_43),     // %23
                      "w"(_k44),        // %24
                      "w"(_bias0)       // %25
                    : "memory", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
                );
#else // __aarch64__
                asm volatile(
                    "pld        [%7, #256]          \n"
                    "vld1.u16   {d20-d23}, [%7 :64]! \n"

                    "pld        [%1, #128]          \n"
                    "vld1.f32   {d24-d25}, [%1]     \n"
                    "vmov       q13, q12            \n"// sum0 sum1

                    "vshll.u16  q8, d20, #16        \n"// k00

                    "pld        [%2, #256]          \n"
                    "vld1.u16   {d4-d7}, [%2 :64]!  \n"// r00 r01 r02 r03

                    "vmov       q14, q12            \n"
                    "vmov       q15, q12            \n"// sum2 sum3

                    "vshll.u16  q9, d21, #16        \n"// k01

                    "pld        [%2, #256]          \n"
                    "vld1.u16   {d12-d15}, [%2 :64]! \n"// r04 r05 r06 r07

                    "vshll.u16  q0, d4, #16         \n"
                    "vshll.u16  q1, d5, #16         \n"
                    "vshll.u16  q2, d6, #16         \n"
                    "vshll.u16  q3, d7, #16         \n"

                    "vshll.u16  q4, d12, #16        \n"
                    "vshll.u16  q5, d13, #16        \n"

                    "vmla.f32   q12, q8, q0         \n"
                    "vmla.f32   q13, q8, q2         \n"
                    "vshll.u16  q6, d14, #16        \n"
                    "vmla.f32   q14, q8, q4         \n"
                    "vmla.f32   q15, q8, q6         \n"

                    "vshll.u16  q10, d22, #16       \n"// k02

                    "vmla.f32   q12, q9, q1         \n"
                    "vmla.f32   q13, q9, q3         \n"
                    "vshll.u16  q7, d15, #16        \n"
                    "vmla.f32   q14, q9, q5         \n"
                    "vmla.f32   q15, q9, q7         \n"

                    "pld        [%2, #128]          \n"
                    "vld1.u16   {d2-d3}, [%2 :64]!  \n"// r08 r09

                    "pld        [%7, #256]          \n"
                    "vld1.u16   {d16-d19}, [%7 :64]! \n"

                    "vmla.f32   q12, q10, q2        \n"
                    "vshll.u16  q11, d23, #16       \n"// k03
                    "vmla.f32   q13, q10, q4        \n"
                    "vshll.u16  q0, d2, #16         \n"
                    "vmla.f32   q14, q10, q6        \n"
                    "vmla.f32   q15, q10, q0        \n"

                    "vshll.u16  q10, d16, #16       \n"// k04

                    "vmla.f32   q12, q11, q3        \n"
                    "vmla.f32   q13, q11, q5        \n"
                    "vshll.u16  q1, d3, #16         \n"
                    "vmla.f32   q14, q11, q7        \n"
                    "vmla.f32   q15, q11, q1        \n"

                    "pld        [%2, #64]           \n"
                    "vld1.u16   {d5}, [%2 :64]      \n"// r010

                    "vmla.f32   q12, q10, q4        \n"
                    "vshll.u16  q11, d17, #16       \n"// k10
                    "vmla.f32   q13, q10, q6        \n"
                    "vshll.u16  q2, d5, #16         \n"
                    "vmla.f32   q14, q10, q0        \n"
                    "pld        [%3, #256]          \n"
                    "vld1.u16   {d12-d15}, [%3 :64]! \n"// r10 r11 r12 r13

                    "vmla.f32   q15, q10, q2        \n"

                    "vshll.u16  q8, d18, #16        \n"// k11

                    "pld        [%3, #256]          \n"
                    "vld1.u16   {d4-d7}, [%3 :64]!  \n"// r14 r15 r16 r17

                    "vshll.u16  q4, d12, #16        \n"
                    "vshll.u16  q5, d13, #16        \n"
                    "vshll.u16  q6, d14, #16        \n"
                    "vshll.u16  q7, d15, #16        \n"

                    "vshll.u16  q0, d4, #16         \n"
                    "vshll.u16  q1, d5, #16         \n"

                    "vmla.f32   q12, q11, q4        \n"
                    "vmla.f32   q13, q11, q6        \n"
                    "vshll.u16  q2, d6, #16         \n"
                    "vmla.f32   q14, q11, q0        \n"
                    "vmla.f32   q15, q11, q2        \n"

                    "pld        [%7, #256]          \n"
                    "vld1.u16   {d20-d23}, [%7 :64]! \n"

                    "vmla.f32   q12, q8, q5         \n"
                    "vshll.u16  q9, d19, #16        \n"// k12
                    "vmla.f32   q13, q8, q7         \n"
                    "vshll.u16  q3, d7, #16         \n"
                    "vmla.f32   q14, q8, q1         \n"
                    "vmla.f32   q15, q8, q3         \n"

                    "pld        [%3, #128]          \n"
                    "vld1.u16   {d10-d11}, [%3 :64]! \n"// r18 r19

                    "vmla.f32   q12, q9, q6         \n"
                    "vshll.u16  q8, d20, #16        \n"// k13
                    "vmla.f32   q13, q9, q0         \n"
                    "vshll.u16  q4, d10, #16        \n"
                    "vmla.f32   q14, q9, q2         \n"
                    "vmla.f32   q15, q9, q4         \n"

                    "vshll.u16  q9, d21, #16        \n"// k14

                    "vmla.f32   q12, q8, q7         \n"
                    "vmla.f32   q13, q8, q1         \n"
                    "vshll.u16  q5, d11, #16        \n"
                    "vmla.f32   q14, q8, q3         \n"
                    "vmla.f32   q15, q8, q5         \n"

                    "pld        [%3, #64]           \n"
                    "vld1.u16   {d13}, [%3 :64]     \n"// r110

                    "vmla.f32   q12, q9, q0         \n"
                    "vshll.u16  q10, d22, #16       \n"// k20
                    "vmla.f32   q13, q9, q2         \n"
                    "vshll.u16  q6, d13, #16        \n"
                    "vmla.f32   q14, q9, q4         \n"
                    "pld        [%4, #256]          \n"
                    "vld1.u16   {d4-d7}, [%4 :64]!  \n"// r20 r21 r22 r23

                    "vmla.f32   q15, q9, q6         \n"

                    "vshll.u16  q11, d23, #16       \n"// k21

                    "pld        [%4, #256]          \n"
                    "vld1.u16   {d12-d15}, [%4 :64]! \n"// r24 r25 r26 r27

                    "vshll.u16  q0, d4, #16         \n"
                    "vshll.u16  q1, d5, #16         \n"
                    "vshll.u16  q2, d6, #16         \n"
                    "vshll.u16  q3, d7, #16         \n"

                    "vshll.u16  q4, d12, #16        \n"
                    "vshll.u16  q5, d13, #16        \n"

                    "pld        [%7, #256]          \n"
                    "vld1.u16   {d16-d19}, [%7 :64]! \n"

                    "vmla.f32   q12, q10, q0        \n"
                    "vmla.f32   q13, q10, q2        \n"
                    "vshll.u16  q6, d14, #16        \n"
                    "vmla.f32   q14, q10, q4        \n"
                    "vmla.f32   q15, q10, q6        \n"

                    "vshll.u16  q10, d16, #16       \n"// k22

                    "vmla.f32   q12, q11, q1        \n"
                    "vmla.f32   q13, q11, q3        \n"
                    "vshll.u16  q7, d15, #16        \n"
                    "vmla.f32   q14, q11, q5        \n"
                    "vmla.f32   q15, q11, q7        \n"

                    "pld        [%4, #128]          \n"
                    "vld1.u16   {d2-d3}, [%4 :64]!  \n"// r28 r29

                    "vmla.f32   q12, q10, q2        \n"
                    "vshll.u16  q11, d17, #16       \n"// k23
                    "vmla.f32   q13, q10, q4        \n"
                    "vshll.u16  q0, d2, #16         \n"
                    "vmla.f32   q14, q10, q6        \n"
                    "vmla.f32   q15, q10, q0        \n"

                    "vshll.u16  q8, d18, #16        \n"// k24

                    "vmla.f32   q12, q11, q3        \n"
                    "vmla.f32   q13, q11, q5        \n"
                    "vshll.u16  q1, d3, #16         \n"
                    "vmla.f32   q14, q11, q7        \n"
                    "vmla.f32   q15, q11, q1        \n"

                    "pld        [%4, #64]           \n"
                    "vld1.u16   {d5}, [%4 :64]      \n"// r210

                    "pld        [%7, #256]          \n"
                    "vld1.u16   {d20-d23}, [%7 :64]! \n"

                    "vmla.f32   q12, q8, q4         \n"
                    "vshll.u16  q9, d19, #16        \n"// k30
                    "vmla.f32   q13, q8, q6         \n"
                    "vshll.u16  q2, d5, #16         \n"
                    "vmla.f32   q14, q8, q0         \n"
                    "pld        [%5, #256]          \n"
                    "vld1.u16   {d12-d15}, [%5 :64]! \n"// r30 r31 r32 r33

                    "vmla.f32   q15, q8, q2         \n"

                    "vshll.u16  q8, d20, #16        \n"// k31

                    "pld        [%5, #256]          \n"
                    "vld1.u16   {d4-d7}, [%5 :64]!  \n"// r34 r35 r36 r37

                    "vshll.u16  q4, d12, #16        \n"
                    "vshll.u16  q5, d13, #16        \n"
                    "vshll.u16  q6, d14, #16        \n"
                    "vshll.u16  q7, d15, #16        \n"

                    "vshll.u16  q0, d4, #16         \n"
                    "vshll.u16  q1, d5, #16         \n"

                    "vmla.f32   q12, q9, q4         \n"
                    "vmla.f32   q13, q9, q6         \n"
                    "vshll.u16  q2, d6, #16         \n"
                    "vmla.f32   q14, q9, q0         \n"
                    "vmla.f32   q15, q9, q2         \n"

                    "vshll.u16  q9, d21, #16        \n"// k32

                    "vmla.f32   q12, q8, q5         \n"
                    "vmla.f32   q13, q8, q7         \n"
                    "vshll.u16  q3, d7, #16         \n"
                    "vmla.f32   q14, q8, q1         \n"
                    "vmla.f32   q15, q8, q3         \n"

                    "pld        [%5, #128]          \n"
                    "vld1.u16   {d10-d11}, [%5 :64]! \n"// r38 r39

                    "vmla.f32   q12, q9, q6         \n"
                    "vshll.u16  q10, d22, #16       \n"// k33
                    "vmla.f32   q13, q9, q0         \n"
                    "vshll.u16  q4, d10, #16        \n"
                    "vmla.f32   q14, q9, q2         \n"
                    "vmla.f32   q15, q9, q4         \n"

                    "pld        [%7, #256]          \n"
                    "vld1.u16   {d16-d19}, [%7 :64]! \n"

                    "vmla.f32   q12, q10, q7        \n"
                    "vshll.u16  q11, d23, #16       \n"// k34
                    "vmla.f32   q13, q10, q1        \n"
                    "vshll.u16  q5, d11, #16        \n"
                    "vmla.f32   q14, q10, q3        \n"
                    "vmla.f32   q15, q10, q5        \n"

                    "pld        [%5, #64]           \n"
                    "vld1.u16   {d13}, [%5 :64]     \n"// r310

                    "vmla.f32   q12, q11, q0        \n"
                    "vshll.u16  q10, d16, #16       \n"// k40
                    "vmla.f32   q13, q11, q2        \n"
                    "vshll.u16  q6, d13, #16        \n"
                    "vmla.f32   q14, q11, q4        \n"
                    "pld        [%6, #256]          \n"
                    "vld1.u16   {d4-d7}, [%6 :64]!  \n"// r40 r41 r42 r43

                    "vmla.f32   q15, q11, q6        \n"

                    "vshll.u16  q11, d17, #16       \n"// k41

                    "pld        [%6, #256]          \n"
                    "vld1.u16   {d12-d15}, [%6 :64]! \n"// r44 r45 r46 r47

                    "vshll.u16  q0, d4, #16         \n"
                    "vshll.u16  q1, d5, #16         \n"
                    "vshll.u16  q2, d6, #16         \n"
                    "vshll.u16  q3, d7, #16         \n"

                    "vshll.u16  q4, d12, #16        \n"
                    "vshll.u16  q5, d13, #16        \n"

                    "vmla.f32   q12, q10, q0        \n"
                    "vmla.f32   q13, q10, q2        \n"
                    "vshll.u16  q6, d14, #16        \n"
                    "vmla.f32   q14, q10, q4        \n"
                    "vmla.f32   q15, q10, q6        \n"

                    "vshll.u16  q8, d18, #16        \n"// k42

                    "vmla.f32   q12, q11, q1        \n"
                    "vmla.f32   q13, q11, q3        \n"
                    "vshll.u16  q7, d15, #16        \n"
                    "vmla.f32   q14, q11, q5        \n"
                    "pld        [%7, #64]           \n"
                    "vld1.u16   {d20}, [%7 :64]     \n"

                    "vmla.f32   q15, q11, q7        \n"

                    "pld        [%6, #128]          \n"
                    "vld1.u16   {d2-d3}, [%6 :64]!  \n"// r48 r49

                    "vmla.f32   q12, q8, q2         \n"
                    "vshll.u16  q9, d19, #16        \n"// k43
                    "vmla.f32   q13, q8, q4         \n"
                    "vshll.u16  q0, d2, #16         \n"
                    "vmla.f32   q14, q8, q6         \n"
                    "vmla.f32   q15, q8, q0         \n"

                    "vshll.u16  q8, d20, #16        \n"// k44

                    "vmla.f32   q12, q9, q3         \n"
                    "vmla.f32   q13, q9, q5         \n"
                    "vshll.u16  q1, d3, #16         \n"
                    "vmla.f32   q14, q9, q7         \n"
                    "vmla.f32   q15, q9, q1         \n"

                    "pld        [%6, #64]           \n"
                    "vld1.u16   {d5}, [%6 :64]      \n"// r410

                    "vmla.f32   q12, q8, q4         \n"
                    "vmla.f32   q13, q8, q6         \n"
                    "vshll.u16  q2, d5, #16         \n"
                    "vmla.f32   q14, q8, q0         \n"
                    "vmla.f32   q15, q8, q2         \n"

                    "sub        %7, %7, #192        \n"// kptr -= 24 * 4;

                    "sub        %2, %2, #16         \n"
                    "sub        %3, %3, #16         \n"
                    "sub        %4, %4, #16         \n"
                    "sub        %5, %5, #16         \n"
                    "sub        %6, %6, #16         \n"

                    "vshrn.u32  d24, q12, #16       \n"
                    "vshrn.u32  d25, q13, #16       \n"
                    "vshrn.u32  d26, q14, #16       \n"
                    "vshrn.u32  d27, q15, #16       \n"

                    "vst1.u16   {d24-d27}, [%0 :64]! \n"

                    : "=r"(outptr0),    // %0
                      "=r"(bias0_data_ptr), // %1
                      "=r"(r0),         // %2
                      "=r"(r1),         // %3
                      "=r"(r2),         // %4
                      "=r"(r3),         // %5
                      "=r"(r4),         // %6
                      "=r"(kptr)        // %7
                    : "0"(outptr0),
                      "1"(bias0_data_ptr),
                      "2"(r0),
                      "3"(r1),
                      "4"(r2),
                      "5"(r3),
                      "6"(r4),
                      "7"(kptr)
                    : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
#endif // __aarch64__
            }
            for (; j+1 < outw; j+=2)
            {
#if __aarch64__
                asm volatile(
                    "prfm   pldl1keep, [%1, #256]       \n"
                    "ld1    {v16.4h, v17.4h, v18.4h, v19.4h}, [%1], #32 \n"// r00 r01 r02 r03

                    "prfm   pldl1keep, [%1, #192]       \n"
                    "ld1    {v20.4h, v21.4h, v22.4h}, [%1] \n"// r04 r05 r06

                    "shll   v14.4s, %12.4h, #16         \n"
                    "shll2  v15.4s, %12.8h, #16         \n"

                    "shll   v16.4s, v16.4h, #16         \n"

                    "mov    v30.16b, %25.16b            \n"// sum00
                    "mov    v31.16b, %25.16b            \n"// sum01

                    "shll   v17.4s, v17.4h, #16         \n"
                    "shll   v18.4s, v18.4h, #16         \n"

                    "fmul   v28.4s, v14.4s, v16.4s      \n"
                    "shll   v19.4s, v19.4h, #16         \n"
                    "fmul   v29.4s, v14.4s, v18.4s      \n"
                    "shll   v14.4s, %13.4h, #16         \n"
                    "fmla   v30.4s, v15.4s, v17.4s      \n"
                    "shll   v20.4s, v20.4h, #16         \n"
                    "fmla   v31.4s, v15.4s, v19.4s      \n"
                    "shll2  v15.4s, %13.8h, #16         \n"
                    "fmla   v28.4s, v14.4s, v18.4s      \n"
                    "shll   v21.4s, v21.4h, #16         \n"
                    "fmla   v29.4s, v14.4s, v20.4s      \n"
                    "shll   v14.4s, %14.4h, #16         \n"
                    "fmla   v30.4s, v15.4s, v19.4s      \n"
                    "prfm   pldl1keep, [%2, #256]       \n"
                    "ld1    {v16.4h, v17.4h, v18.4h, v19.4h}, [%2], #32 \n"// r10 r11 r12 r13

                    "shll   v22.4s, v22.4h, #16         \n"
                    "fmla   v31.4s, v15.4s, v21.4s      \n"
                    "shll2  v15.4s, %14.8h, #16         \n"
                    "fmla   v28.4s, v14.4s, v20.4s      \n"
                    "shll   v16.4s, v16.4h, #16         \n"
                    "fmla   v29.4s, v14.4s, v22.4s      \n"

                    "prfm   pldl1keep, [%2, #192]       \n"
                    "ld1    {v20.4h, v21.4h, v22.4h}, [%2] \n"// r14 r15 r16

                    "shll   v14.4s, %15.4h, #16         \n"
                    "shll   v17.4s, v17.4h, #16         \n"
                    "shll   v18.4s, v18.4h, #16         \n"

                    "fmla   v30.4s, v15.4s, v16.4s      \n"
                    "shll   v19.4s, v19.4h, #16         \n"
                    "fmla   v31.4s, v15.4s, v18.4s      \n"
                    "shll2  v15.4s, %15.8h, #16         \n"
                    "fmla   v28.4s, v14.4s, v17.4s      \n"
                    "shll   v20.4s, v20.4h, #16         \n"
                    "fmla   v29.4s, v14.4s, v19.4s      \n"
                    "shll   v14.4s, %16.4h, #16         \n"
                    "fmla   v30.4s, v15.4s, v18.4s      \n"
                    "shll   v21.4s, v21.4h, #16         \n"
                    "fmla   v31.4s, v15.4s, v20.4s      \n"
                    "shll2  v15.4s, %16.8h, #16         \n"
                    "fmla   v28.4s, v14.4s, v19.4s      \n"
                    "prfm   pldl1keep, [%3, #256]       \n"
                    "ld1    {v16.4h, v17.4h, v18.4h, v19.4h}, [%3], #32 \n"// r20 r21 r22 r23

                    "shll   v22.4s, v22.4h, #16         \n"
                    "fmla   v29.4s, v14.4s, v21.4s      \n"
                    "shll   v14.4s, %17.4h, #16         \n"
                    "fmla   v30.4s, v15.4s, v20.4s      \n"
                    "shll   v16.4s, v16.4h, #16         \n"
                    "fmla   v31.4s, v15.4s, v22.4s      \n"

                    "prfm   pldl1keep, [%3, #192]       \n"
                    "ld1    {v20.4h, v21.4h, v22.4h}, [%3] \n"// r24 r25 r26

                    "shll2  v15.4s, %17.8h, #16         \n"
                    "shll   v17.4s, v17.4h, #16         \n"
                    "shll   v18.4s, v18.4h, #16         \n"

                    "fmla   v28.4s, v14.4s, v16.4s      \n"
                    "shll   v19.4s, v19.4h, #16         \n"
                    "fmla   v29.4s, v14.4s, v18.4s      \n"
                    "shll   v14.4s, %18.4h, #16         \n"
                    "fmla   v30.4s, v15.4s, v17.4s      \n"
                    "shll   v20.4s, v20.4h, #16         \n"
                    "fmla   v31.4s, v15.4s, v19.4s      \n"
                    "shll2  v15.4s, %18.8h, #16         \n"
                    "fmla   v28.4s, v14.4s, v18.4s      \n"
                    "shll   v21.4s, v21.4h, #16         \n"
                    "fmla   v29.4s, v14.4s, v20.4s      \n"
                    "shll   v14.4s, %19.4h, #16         \n"
                    "fmla   v30.4s, v15.4s, v19.4s      \n"
                    "prfm   pldl1keep, [%4, #256]       \n"
                    "ld1    {v16.4h, v17.4h, v18.4h, v19.4h}, [%4], #32 \n"// r30 r31 r32 r33

                    "shll   v22.4s, v22.4h, #16         \n"
                    "fmla   v31.4s, v15.4s, v21.4s      \n"
                    "shll2  v15.4s, %19.8h, #16         \n"
                    "fmla   v28.4s, v14.4s, v20.4s      \n"
                    "shll   v16.4s, v16.4h, #16         \n"
                    "fmla   v29.4s, v14.4s, v22.4s      \n"

                    "prfm   pldl1keep, [%4, #192]       \n"
                    "ld1    {v20.4h, v21.4h, v22.4h}, [%4] \n"// r34 r35 r36

                    "shll   v14.4s, %20.4h, #16         \n"
                    "shll   v17.4s, v17.4h, #16         \n"
                    "shll   v18.4s, v18.4h, #16         \n"

                    "fmla   v30.4s, v15.4s, v16.4s      \n"
                    "shll   v19.4s, v19.4h, #16         \n"
                    "fmla   v31.4s, v15.4s, v18.4s      \n"
                    "shll2  v15.4s, %20.8h, #16         \n"
                    "fmla   v28.4s, v14.4s, v17.4s      \n"
                    "shll   v20.4s, v20.4h, #16         \n"
                    "fmla   v29.4s, v14.4s, v19.4s      \n"
                    "shll   v14.4s, %21.4h, #16         \n"
                    "fmla   v30.4s, v15.4s, v18.4s      \n"
                    "shll   v21.4s, v21.4h, #16         \n"
                    "fmla   v31.4s, v15.4s, v20.4s      \n"
                    "shll2  v15.4s, %21.8h, #16         \n"
                    "fmla   v28.4s, v14.4s, v19.4s      \n"
                    "prfm   pldl1keep, [%5, #256]       \n"
                    "ld1    {v16.4h, v17.4h, v18.4h, v19.4h}, [%5], #32 \n"// r40 r41 r42 r43

                    "shll   v22.4s, v22.4h, #16         \n"
                    "fmla   v29.4s, v14.4s, v21.4s      \n"
                    "shll   v14.4s, %22.4h, #16         \n"
                    "fmla   v30.4s, v15.4s, v20.4s      \n"
                    "shll   v16.4s, v16.4h, #16         \n"
                    "fmla   v31.4s, v15.4s, v22.4s      \n"

                    "prfm   pldl1keep, [%5, #192]       \n"
                    "ld1    {v20.4h, v21.4h, v22.4h}, [%5] \n"// r44 r45 r46

                    "shll2  v15.4s, %22.8h, #16         \n"
                    "shll   v17.4s, v17.4h, #16         \n"
                    "shll   v18.4s, v18.4h, #16         \n"

                    "fmla   v28.4s, v14.4s, v16.4s      \n"
                    "shll   v19.4s, v19.4h, #16         \n"
                    "fmla   v29.4s, v14.4s, v18.4s      \n"
                    "shll   v14.4s, %23.4h, #16         \n"
                    "fmla   v30.4s, v15.4s, v17.4s      \n"
                    "shll   v20.4s, v20.4h, #16         \n"
                    "fmla   v31.4s, v15.4s, v19.4s      \n"
                    "shll2  v15.4s, %23.8h, #16         \n"
                    "fmla   v28.4s, v14.4s, v18.4s      \n"
                    "shll   v21.4s, v21.4h, #16         \n"
                    "fmla   v29.4s, v14.4s, v20.4s      \n"
                    "shll   v14.4s, %24.4h, #16         \n"
                    "fmla   v30.4s, v15.4s, v19.4s      \n"
                    "fmla   v31.4s, v15.4s, v21.4s      \n"
                    "shll   v22.4s, v22.4h, #16         \n"
                    "fmla   v28.4s, v14.4s, v20.4s      \n"
                    "fmla   v29.4s, v14.4s, v22.4s      \n"

                    "fadd   v30.4s, v30.4s, v28.4s      \n"
                    "fadd   v31.4s, v31.4s, v29.4s      \n"

                    "shrn   v30.4h, v30.4s, #16         \n"
                    "shrn   v31.4h, v31.4s, #16         \n"

                    "st1    {v30.4h, v31.4h}, [%0], #16 \n"

                    : "=r"(outptr0),    // %0
                      "=r"(r0),         // %1
                      "=r"(r1),         // %2
                      "=r"(r2),         // %3
                      "=r"(r3),         // %4
                      "=r"(r4)          // %5
                    : "0"(outptr0),
                      "1"(r0),
                      "2"(r1),
                      "3"(r2),
                      "4"(r3),
                      "5"(r4),
                      "w"(_k00_01),     // %12
                      "w"(_k02_03),     // %13
                      "w"(_k04_10),     // %14
                      "w"(_k11_12),     // %15
                      "w"(_k13_14),     // %16
                      "w"(_k20_21),     // %17
                      "w"(_k22_23),     // %18
                      "w"(_k24_30),     // %19
                      "w"(_k31_32),     // %20
                      "w"(_k33_34),     // %21
                      "w"(_k40_41),     // %22
                      "w"(_k42_43),     // %23
                      "w"(_k44),        // %24
                      "w"(_bias0)       // %25
                    : "memory", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
                );
#else // __aarch64__
                asm volatile(
                    "pld        [%7, #256]          \n"
                    "vld1.u16   {d20-d23}, [%7 :64]! \n"

                    "pld        [%1, #128]          \n"
                    "vld1.f32   {d24-d25}, [%1]     \n"

                    "pld        [%2, #256]          \n"
                    "vld1.u16   {d4-d7}, [%2 :64]!  \n"// r00 r01 r02 r03

                    "vshll.u16  q8, d20, #16        \n"// k00

                    "pld        [%2, #256]          \n"
                    "vld1.u16   {d10-d12}, [%2 :64] \n"// r04 r05 r06

                    "vmov       q13, q12            \n"// sum0 sum1

                    "vshll.u16  q0, d4, #16         \n"
                    "vshll.u16  q1, d5, #16         \n"
                    "vshll.u16  q2, d6, #16         \n"
                    "vshll.u16  q3, d7, #16         \n"

                    "vshll.u16  q9, d21, #16        \n"// k01
                    "vmul.f32   q14, q8, q0         \n"
                    "vshll.u16  q4, d10, #16        \n"
                    "vmul.f32   q15, q8, q2         \n"
                    "vshll.u16  q10, d22, #16       \n"// k02
                    "vmla.f32   q12, q9, q1         \n"
                    "vshll.u16  q5, d11, #16        \n"
                    "vmla.f32   q13, q9, q3         \n"
                    "pld        [%7, #256]          \n"
                    "vld1.u16   {d16-d19}, [%7 :64]! \n"
                    "vmla.f32   q14, q10, q2        \n"
                    "vshll.u16  q11, d23, #16       \n"// k03
                    "vmla.f32   q15, q10, q4        \n"
                    "vshll.u16  q10, d16, #16       \n"// k04
                    "vmla.f32   q12, q11, q3        \n"
                    "vshll.u16  q6, d12, #16        \n"
                    "vmla.f32   q13, q11, q5        \n"
                    "vshll.u16  q11, d17, #16       \n"// k10
                    "vmla.f32   q14, q10, q4        \n"
                    "vmla.f32   q15, q10, q6        \n"

                    "pld        [%3, #256]          \n"
                    "vld1.u16   {d4-d7}, [%3 :64]!  \n"// r10 r11 r12 r13

                    "vshll.u16  q8, d18, #16        \n"// k11

                    "pld        [%3, #256]          \n"
                    "vld1.u16   {d10-d12}, [%3 :64] \n"// r14 r15 r16

                    "vshll.u16  q0, d4, #16         \n"
                    "vshll.u16  q1, d5, #16         \n"
                    "vshll.u16  q2, d6, #16         \n"
                    "vshll.u16  q3, d7, #16         \n"

                    "vmla.f32   q12, q11, q0        \n"
                    "vshll.u16  q4, d10, #16        \n"
                    "vmla.f32   q13, q11, q2        \n"
                    "pld        [%7, #256]          \n"
                    "vld1.u16   {d20-d23}, [%7 :64]! \n"
                    "vmla.f32   q14, q8, q1         \n"
                    "vshll.u16  q9, d19, #16        \n"// k12
                    "vmla.f32   q15, q8, q3         \n"
                    "vshll.u16  q8, d20, #16        \n"// k13
                    "vmla.f32   q12, q9, q2         \n"
                    "vshll.u16  q5, d11, #16        \n"
                    "vmla.f32   q13, q9, q4         \n"
                    "vshll.u16  q9, d21, #16        \n"// k14
                    "vmla.f32   q14, q8, q3         \n"
                    "vshll.u16  q6, d12, #16        \n"
                    "vmla.f32   q15, q8, q5         \n"
                    "vshll.u16  q10, d22, #16       \n"// k20
                    "vmla.f32   q12, q9, q4         \n"
                    "vmla.f32   q13, q9, q6         \n"
                    "pld        [%7, #256]          \n"
                    "vld1.u16   {d16-d19}, [%7 :64]! \n"

                    "pld        [%4, #256]          \n"
                    "vld1.u16   {d4-d7}, [%4 :64]!  \n"// r20 r21 r22 r23

                    "vshll.u16  q11, d23, #16       \n"// k21

                    "pld        [%4, #256]          \n"
                    "vld1.u16   {d10-d12}, [%4 :64] \n"// r24 r25 r26

                    "vshll.u16  q0, d4, #16         \n"
                    "vshll.u16  q1, d5, #16         \n"
                    "vshll.u16  q2, d6, #16         \n"
                    "vshll.u16  q3, d7, #16         \n"

                    "vmla.f32   q14, q10, q0        \n"
                    "vmla.f32   q15, q10, q2        \n"
                    "vshll.u16  q10, d16, #16       \n"// k22
                    "vmla.f32   q12, q11, q1        \n"
                    "vshll.u16  q4, d10, #16        \n"
                    "vmla.f32   q13, q11, q3        \n"
                    "vshll.u16  q11, d17, #16       \n"// k23
                    "vmla.f32   q14, q10, q2        \n"
                    "vshll.u16  q5, d11, #16        \n"
                    "vmla.f32   q15, q10, q4        \n"
                    "vshll.u16  q8, d18, #16        \n"// k24
                    "vmla.f32   q12, q11, q3        \n"
                    "vshll.u16  q6, d12, #16        \n"
                    "vmla.f32   q13, q11, q5        \n"
                    "pld        [%7, #256]          \n"
                    "vld1.u16   {d20-d23}, [%7 :64]! \n"
                    "vmla.f32   q14, q8, q4         \n"
                    "vshll.u16  q9, d19, #16        \n"// k30
                    "vmla.f32   q15, q8, q6         \n"

                    "pld        [%5, #256]          \n"
                    "vld1.u16   {d4-d7}, [%5 :64]!  \n"// r30 r31 r32 r33

                    "vshll.u16  q8, d20, #16        \n"// k31

                    "pld        [%5, #256]          \n"
                    "vld1.u16   {d10-d12}, [%5 :64] \n"// r34 r35 r36

                    "vshll.u16  q0, d4, #16         \n"
                    "vshll.u16  q1, d5, #16         \n"
                    "vshll.u16  q2, d6, #16         \n"
                    "vshll.u16  q3, d7, #16         \n"

                    "vmla.f32   q12, q9, q0         \n"
                    "vshll.u16  q4, d10, #16        \n"
                    "vmla.f32   q13, q9, q2         \n"
                    "vshll.u16  q9, d21, #16        \n"// k32
                    "vmla.f32   q14, q8, q1         \n"
                    "vshll.u16  q5, d11, #16        \n"
                    "vmla.f32   q15, q8, q3         \n"
                    "vshll.u16  q10, d22, #16       \n"// k33
                    "vmla.f32   q12, q9, q2         \n"
                    "vshll.u16  q6, d12, #16        \n"
                    "vmla.f32   q13, q9, q4         \n"
                    "pld        [%7, #256]          \n"
                    "vld1.u16   {d16-d19}, [%7 :64]! \n"
                    "vmla.f32   q14, q10, q3        \n"
                    "vshll.u16  q11, d23, #16       \n"// k34
                    "vmla.f32   q15, q10, q5        \n"
                    "vshll.u16  q10, d16, #16       \n"// k40
                    "vmla.f32   q12, q11, q4        \n"
                    "vmla.f32   q13, q11, q6        \n"

                    "pld        [%6, #256]          \n"
                    "vld1.u16   {d4-d7}, [%6 :64]!  \n"// r40 r41 r42 r43

                    "vshll.u16  q11, d17, #16       \n"// k41

                    "pld        [%6, #256]          \n"
                    "vld1.u16   {d10-d12}, [%6 :64] \n"// r44 r45 r46

                    "vshll.u16  q0, d4, #16         \n"
                    "vshll.u16  q1, d5, #16         \n"
                    "vshll.u16  q2, d6, #16         \n"
                    "vshll.u16  q3, d7, #16         \n"

                    "vmla.f32   q14, q10, q0        \n"
                    "vshll.u16  q4, d10, #16        \n"
                    "vmla.f32   q15, q10, q2        \n"
                    "vshll.u16  q8, d18, #16        \n"// k42
                    "vmla.f32   q12, q11, q1        \n"
                    "vshll.u16  q5, d11, #16        \n"
                    "vmla.f32   q13, q11, q3        \n"
                    "pld        [%7, #64]           \n"
                    "vld1.u16   {d20}, [%7 :64]     \n"
                    "vmla.f32   q14, q8, q2         \n"
                    "vshll.u16  q9, d19, #16        \n"// k43
                    "vmla.f32   q15, q8, q4         \n"
                    "vshll.u16  q8, d20, #16        \n"// k44
                    "vmla.f32   q12, q9, q3         \n"
                    "vshll.u16  q6, d12, #16        \n"
                    "vmla.f32   q13, q9, q5         \n"
                    "vmla.f32   q14, q8, q4         \n"
                    "vmla.f32   q15, q8, q6         \n"

                    "vadd.f32   q12, q12, q14       \n"
                    "vadd.f32   q13, q13, q15       \n"

                    "sub        %7, %7, #192        \n"// kptr -= 24 * 4;

                    "vshrn.u32  d24, q12, #16       \n"
                    "vshrn.u32  d25, q13, #16       \n"

                    "vst1.u16   {d24-d25}, [%0 :64]! \n"

                    : "=r"(outptr0),    // %0
                      "=r"(bias0_data_ptr), // %1
                      "=r"(r0),         // %2
                      "=r"(r1),         // %3
                      "=r"(r2),         // %4
                      "=r"(r3),         // %5
                      "=r"(r4),         // %6
                      "=r"(kptr)        // %7
                    : "0"(outptr0),
                      "1"(bias0_data_ptr),
                      "2"(r0),
                      "3"(r1),
                      "4"(r2),
                      "5"(r3),
                      "6"(r4),
                      "7"(kptr)
                    : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
#endif // __aarch64__
            }
            for (; j < outw; j++)
            {
#if __aarch64__
                asm volatile(
                    "prfm   pldl1keep, [%1, #128]       \n"
                    "ld1    {v16.4h, v17.4h}, [%1], #16 \n"// r00 r01

                    "prfm   pldl1keep, [%1, #192]       \n"
                    "ld1    {v18.4h, v19.4h, v20.4h}, [%1] \n"// r02 r03 r04

                    "shll   v14.4s, %12.4h, #16         \n"

                    "mov    v31.16b, %25.16b            \n"// sum00

                    "shll   v16.4s, v16.4h, #16         \n"
                    "shll   v17.4s, v17.4h, #16         \n"

                    "shll   v18.4s, v18.4h, #16         \n"
                    "shll   v19.4s, v19.4h, #16         \n"
                    "shll   v20.4s, v20.4h, #16         \n"

                    "shll2  v15.4s, %12.8h, #16         \n"
                    "fmul   v28.4s, v14.4s, v16.4s      \n"
                    "shll   v14.4s, %13.4h, #16         \n"
                    "fmul   v29.4s, v15.4s, v17.4s      \n"
                    "shll2  v15.4s, %13.8h, #16         \n"
                    "fmul   v30.4s, v14.4s, v18.4s      \n"
                    "shll   v14.4s, %14.4h, #16         \n"
                    "fmla   v31.4s, v15.4s, v19.4s      \n"
                    "shll2  v15.4s, %14.8h, #16         \n"
                    "fmla   v28.4s, v14.4s, v20.4s      \n"

                    "prfm   pldl1keep, [%2, #128]       \n"
                    "ld1    {v16.4h, v17.4h}, [%2], #16 \n"// r10 r11

                    "prfm   pldl1keep, [%2, #192]       \n"
                    "ld1    {v18.4h, v19.4h, v20.4h}, [%2] \n"// r12 r13 r14

                    "shll   v14.4s, %15.4h, #16         \n"

                    "shll   v16.4s, v16.4h, #16         \n"
                    "shll   v17.4s, v17.4h, #16         \n"

                    "shll   v18.4s, v18.4h, #16         \n"
                    "shll   v19.4s, v19.4h, #16         \n"
                    "shll   v20.4s, v20.4h, #16         \n"

                    "fmla   v29.4s, v15.4s, v16.4s      \n"
                    "shll2  v15.4s, %15.8h, #16         \n"
                    "fmla   v30.4s, v14.4s, v17.4s      \n"
                    "shll   v14.4s, %16.4h, #16         \n"
                    "fmla   v31.4s, v15.4s, v18.4s      \n"
                    "shll2  v15.4s, %16.8h, #16         \n"
                    "fmla   v28.4s, v14.4s, v19.4s      \n"
                    "shll   v14.4s, %17.4h, #16         \n"
                    "fmla   v29.4s, v15.4s, v20.4s      \n"

                    "prfm   pldl1keep, [%3, #128]       \n"
                    "ld1    {v16.4h, v17.4h}, [%3], #16 \n"// r20 r21

                    "prfm   pldl1keep, [%3, #192]       \n"
                    "ld1    {v18.4h, v19.4h, v20.4h}, [%3] \n"// r22 r23 r24

                    "shll2  v15.4s, %17.8h, #16         \n"

                    "shll   v16.4s, v16.4h, #16         \n"
                    "shll   v17.4s, v17.4h, #16         \n"

                    "shll   v18.4s, v18.4h, #16         \n"
                    "shll   v19.4s, v19.4h, #16         \n"
                    "shll   v20.4s, v20.4h, #16         \n"

                    "fmla   v30.4s, v14.4s, v16.4s      \n"
                    "shll   v14.4s, %18.4h, #16         \n"
                    "fmla   v31.4s, v15.4s, v17.4s      \n"
                    "shll2  v15.4s, %18.8h, #16         \n"
                    "fmla   v28.4s, v14.4s, v18.4s      \n"
                    "shll   v14.4s, %19.4h, #16         \n"
                    "fmla   v29.4s, v15.4s, v19.4s      \n"
                    "shll2  v15.4s, %19.8h, #16         \n"
                    "fmla   v30.4s, v14.4s, v20.4s      \n"

                    "prfm   pldl1keep, [%4, #128]       \n"
                    "ld1    {v16.4h, v17.4h}, [%4], #16 \n"// r30 r31

                    "prfm   pldl1keep, [%4, #192]       \n"
                    "ld1    {v18.4h, v19.4h, v20.4h}, [%4] \n"// r32 r33 r34

                    "shll   v14.4s, %20.4h, #16         \n"

                    "shll   v16.4s, v16.4h, #16         \n"
                    "shll   v17.4s, v17.4h, #16         \n"

                    "shll   v18.4s, v18.4h, #16         \n"
                    "shll   v19.4s, v19.4h, #16         \n"
                    "shll   v20.4s, v20.4h, #16         \n"

                    "fmla   v31.4s, v15.4s, v16.4s      \n"
                    "shll2  v15.4s, %20.8h, #16         \n"
                    "fmla   v28.4s, v14.4s, v17.4s      \n"
                    "shll   v14.4s, %21.4h, #16         \n"
                    "fmla   v29.4s, v15.4s, v18.4s      \n"
                    "shll2  v15.4s, %21.8h, #16         \n"
                    "fmla   v30.4s, v14.4s, v19.4s      \n"
                    "shll   v14.4s, %22.4h, #16         \n"
                    "fmla   v31.4s, v15.4s, v20.4s      \n"

                    "prfm   pldl1keep, [%5, #128]       \n"
                    "ld1    {v16.4h, v17.4h}, [%5], #16 \n"// r40 r41

                    "prfm   pldl1keep, [%5, #192]       \n"
                    "ld1    {v18.4h, v19.4h, v20.4h}, [%5] \n"// r42 r43 r44

                    "shll2  v15.4s, %22.8h, #16         \n"

                    "shll   v16.4s, v16.4h, #16         \n"
                    "shll   v17.4s, v17.4h, #16         \n"

                    "shll   v18.4s, v18.4h, #16         \n"
                    "shll   v19.4s, v19.4h, #16         \n"
                    "shll   v20.4s, v20.4h, #16         \n"

                    "fmla   v28.4s, v14.4s, v16.4s      \n"
                    "shll   v14.4s, %23.4h, #16         \n"
                    "fmla   v29.4s, v15.4s, v17.4s      \n"
                    "shll2  v15.4s, %23.8h, #16         \n"
                    "fmla   v30.4s, v14.4s, v18.4s      \n"
                    "shll   v14.4s, %24.4h, #16         \n"
                    "fmla   v31.4s, v15.4s, v19.4s      \n"
                    "fmla   v28.4s, v14.4s, v20.4s      \n"

                    "fadd   v29.4s, v29.4s, v30.4s      \n"
                    "fadd   v31.4s, v31.4s, v28.4s      \n"
                    "fadd   v31.4s, v31.4s, v29.4s      \n"

                    "shrn   v31.4h, v31.4s, #16         \n"

                    "st1    {v31.4h}, [%0], #8          \n"

                    : "=r"(outptr0),    // %0
                      "=r"(r0),         // %1
                      "=r"(r1),         // %2
                      "=r"(r2),         // %3
                      "=r"(r3),         // %4
                      "=r"(r4)          // %5
                    : "0"(outptr0),
                      "1"(r0),
                      "2"(r1),
                      "3"(r2),
                      "4"(r3),
                      "5"(r4),
                      "w"(_k00_01),     // %12
                      "w"(_k02_03),     // %13
                      "w"(_k04_10),     // %14
                      "w"(_k11_12),     // %15
                      "w"(_k13_14),     // %16
                      "w"(_k20_21),     // %17
                      "w"(_k22_23),     // %18
                      "w"(_k24_30),     // %19
                      "w"(_k31_32),     // %20
                      "w"(_k33_34),     // %21
                      "w"(_k40_41),     // %22
                      "w"(_k42_43),     // %23
                      "w"(_k44),        // %24
                      "w"(_bias0)       // %25
                    : "memory", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
                );
#else // __aarch64__
                asm volatile(
                    "pld        [%2, #128]          \n"
                    "vld1.u16   {d2-d3}, [%2 :64]!  \n"// r00 r01

                    "pld        [%2, #192]          \n"
                    "vld1.u16   {d6-d8}, [%2 :64]   \n"// r02 r03 r04

                    "vshll.u16  q0, d2, #16         \n"
                    "vshll.u16  q1, d3, #16         \n"

                    "pld        [%7, #256]          \n"
                    "vld1.u16   {d20-d23}, [%7 :64]! \n"

                    "vshll.u16  q8, d20, #16        \n"// k00

                    "pld        [%1, #128]          \n"
                    "vld1.f32   {d24-d25}, [%1]     \n"// sum0

                    "vshll.u16  q9, d21, #16        \n"// k01
                    "vmul.f32   q13, q8, q0         \n"
                    "vshll.u16  q10, d22, #16       \n"// k02
                    "vmul.f32   q14, q9, q1         \n"

                    "pld        [%3, #128]          \n"
                    "vld1.u16   {d14-d15}, [%3 :64]! \n"// r10 r11

                    "vshll.u16  q2, d6, #16         \n"
                    "vshll.u16  q3, d7, #16         \n"
                    "vshll.u16  q4, d8, #16         \n"

                    "pld        [%7, #256]          \n"
                    "vld1.u16   {d16-d19}, [%7 :64]! \n"
                    "vshll.u16  q11, d23, #16       \n"// k03
                    "vmul.f32   q15, q10, q2        \n"
                    "vshll.u16  q10, d16, #16       \n"// k04
                    "vmla.f32   q12, q11, q3        \n"
                    "vshll.u16  q11, d17, #16       \n"// k10
                    "vmla.f32   q13, q10, q4        \n"

                    "pld        [%3, #192]          \n"
                    "vld1.u16   {d8-d10}, [%3 :64]  \n"// r12 r13 r14

                    "vshll.u16  q6, d14, #16        \n"
                    "vshll.u16  q7, d15, #16        \n"

                    "vshll.u16  q8, d18, #16        \n"// k11
                    "vmla.f32   q14, q11, q6        \n"
                    "pld        [%7, #256]          \n"
                    "vld1.u16   {d20-d23}, [%7 :64]! \n"
                    "vshll.u16  q9, d19, #16        \n"// k12
                    "vmla.f32   q15, q8, q7         \n"

                    "pld        [%4, #128]          \n"
                    "vld1.u16   {d2-d3}, [%4 :64]!  \n"// r20 r21

                    "vshll.u16  q3, d8, #16         \n"
                    "vshll.u16  q4, d9, #16         \n"
                    "vshll.u16  q5, d10, #16        \n"

                    "vshll.u16  q8, d20, #16        \n"// k13
                    "vmla.f32   q12, q9, q3         \n"
                    "vshll.u16  q9, d21, #16        \n"// k14
                    "vmla.f32   q13, q8, q4         \n"
                    "vshll.u16  q10, d22, #16       \n"// k20
                    "vmla.f32   q14, q9, q5         \n"

                    "pld        [%4, #192]          \n"
                    "vld1.u16   {d6-d8}, [%4 :64]   \n"// r22 r23 r24

                    "vshll.u16  q0, d2, #16         \n"
                    "vshll.u16  q1, d3, #16         \n"

                    "pld        [%7, #256]          \n"
                    "vld1.u16   {d16-d19}, [%7 :64]! \n"
                    "vshll.u16  q11, d23, #16       \n"// k21
                    "vmla.f32   q15, q10, q0        \n"
                    "vshll.u16  q10, d16, #16       \n"// k22
                    "vmla.f32   q12, q11, q1        \n"

                    "pld        [%5, #128]          \n"
                    "vld1.u16   {d14-d15}, [%5 :64]! \n"// r30 r31

                    "vshll.u16  q2, d6, #16         \n"
                    "vshll.u16  q3, d7, #16         \n"
                    "vshll.u16  q4, d8, #16         \n"

                    "vshll.u16  q11, d17, #16       \n"// k23
                    "vmla.f32   q13, q10, q2        \n"
                    "vshll.u16  q8, d18, #16        \n"// k24
                    "vmla.f32   q14, q11, q3        \n"
                    "pld        [%7, #256]          \n"
                    "vld1.u16   {d20-d23}, [%7 :64]! \n"
                    "vshll.u16  q9, d19, #16        \n"// k30
                    "vmla.f32   q15, q8, q4         \n"

                    "pld        [%5, #192]          \n"
                    "vld1.u16   {d8-d10}, [%5 :64]  \n"// r32 r33 r34

                    "vshll.u16  q6, d14, #16        \n"
                    "vshll.u16  q7, d15, #16        \n"

                    "vshll.u16  q8, d20, #16        \n"// k31
                    "vmla.f32   q12, q9, q6         \n"

                    "vshll.u16  q9, d21, #16        \n"// k32
                    "vmla.f32   q13, q8, q7         \n"

                    "pld        [%6, #128]          \n"
                    "vld1.u16   {d2-d3}, [%6 :64]!  \n"// r40 r41

                    "vshll.u16  q3, d8, #16         \n"
                    "vshll.u16  q4, d9, #16         \n"
                    "vshll.u16  q5, d10, #16        \n"

                    "vshll.u16  q10, d22, #16       \n"// k33
                    "vmla.f32   q14, q9, q3         \n"
                    "pld        [%7, #256]          \n"
                    "vld1.u16   {d16-d19}, [%7 :64]! \n"
                    "vshll.u16  q11, d23, #16       \n"// k34
                    "vmla.f32   q15, q10, q4        \n"
                    "vshll.u16  q10, d16, #16       \n"// k40
                    "vmla.f32   q12, q11, q5        \n"

                    "pld        [%6, #192]          \n"
                    "vld1.u16   {d6-d8}, [%6 :64]   \n"// r42 r43 r44

                    "vshll.u16  q0, d2, #16         \n"
                    "vshll.u16  q1, d3, #16         \n"

                    "vshll.u16  q11, d17, #16       \n"// k41
                    "vmla.f32   q13, q10, q0        \n"
                    "vshll.u16  q8, d18, #16        \n"// k42
                    "vmla.f32   q14, q11, q1        \n"

                    "vshll.u16  q2, d6, #16         \n"
                    "vshll.u16  q3, d7, #16         \n"
                    "vshll.u16  q4, d8, #16         \n"

                    "pld        [%7, #64]           \n"
                    "vld1.u16   {d20}, [%7 :64]     \n"
                    "vshll.u16  q9, d19, #16        \n"// k43
                    "vmla.f32   q15, q8, q2         \n"
                    "vshll.u16  q8, d20, #16        \n"// k44
                    "vmla.f32   q12, q9, q3         \n"

                    "vmla.f32   q13, q8, q4         \n"

                    "vadd.f32   q14, q14, q15       \n"
                    "vadd.f32   q12, q12, q13       \n"

                    "sub        %7, %7, #192        \n"// kptr -= 24 * 4;

                    "vadd.f32   q12, q12, q14       \n"

                    "vshrn.u32  d24, q12, #16       \n"

                    "vst1.u16   {d24}, [%0 :64]!    \n"

                    : "=r"(outptr0),    // %0
                      "=r"(bias0_data_ptr), // %1
                      "=r"(r0),         // %2
                      "=r"(r1),         // %3
                      "=r"(r2),         // %4
                      "=r"(r3),         // %5
                      "=r"(r4),         // %6
                      "=r"(kptr)        // %7
                    : "0"(outptr0),
                      "1"(bias0_data_ptr),
                      "2"(r0),
                      "3"(r1),
                      "4"(r2),
                      "5"(r3),
                      "6"(r4),
                      "7"(kptr)
                    : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
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
