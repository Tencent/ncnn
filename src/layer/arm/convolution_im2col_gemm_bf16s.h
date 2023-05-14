// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
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

static void convolution_gemm_transB_packed_tile_bf16s(const Mat& AT_tile, const Mat& BT_tile, const Mat& CT_tile, Mat& topT_tile, Mat& top_blob, int i, int max_ii, int j, int max_jj, int k, int max_kk, bool k_end, int use_a53_a55_optimized_kernel)
{
    // NCNN_LOGE("convolution_gemm_transB_packed_tile_bf16s %d %d %d %d %d %d", i, max_ii, j, max_jj, k, max_kk);

    const int out_elempack = top_blob.elempack;
    const int out_hstep = (int)top_blob.cstep;

    const unsigned short* pAT = AT_tile;
    const unsigned short* pBT = BT_tile;
    const float* pC = CT_tile;

    float* outptr = topT_tile;

    int ii = 0;
#if __ARM_NEON
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        unsigned short* outptr0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const unsigned short* pB = pBT;

        if (pC)
        {
            pC = (const float*)CT_tile + i + ii;
        }

        int jj = 0;
        for (; jj + 11 < max_jj; jj += 12)
        {
            const unsigned short* pA = pAT;

#if NCNN_GNU_INLINE_ASM
            if (use_a53_a55_optimized_kernel && cpu_support_arm_asimdhp())
            {
                // a55
                asm volatile(
                    "cbz    %w10, 0f                    \n"

                    "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%0], #64   \n"
                    "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%0], #64 \n"
                    "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                    "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                    "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                    "ld1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0]      \n"
                    "subs   %0, %0, #320                \n"
                    "b      3f                          \n"

                    "0:                                 \n"
                    // if pC
                    "cbz    %8, 1f                      \n"

                    "add    x4, %8, #16                 \n"
                    "ld1    {v8.4s}, [%8]               \n"
                    "ld1    {v20.4s}, [x4]              \n"
                    "b      2f                          \n"

                    // else
                    "1:                                 \n"
                    "eor    v8.16b, v8.16b, v8.16b      \n"
                    "eor    v20.16b, v20.16b, v20.16b   \n"

                    "2:                                 \n"
                    "mov    v9.16b, v8.16b              \n"
                    "mov    v10.16b, v8.16b             \n"
                    "mov    v11.16b, v8.16b             \n"
                    "mov    v12.16b, v8.16b             \n"
                    "mov    v13.16b, v8.16b             \n"
                    "mov    v14.16b, v8.16b             \n"
                    "mov    v15.16b, v8.16b             \n"
                    "mov    v16.16b, v8.16b             \n"
                    "mov    v17.16b, v8.16b             \n"
                    "mov    v18.16b, v8.16b             \n"
                    "mov    v19.16b, v8.16b             \n"

                    "mov    v21.16b, v20.16b            \n"
                    "mov    v22.16b, v20.16b            \n"
                    "mov    v23.16b, v20.16b            \n"
                    "mov    v24.16b, v20.16b            \n"
                    "mov    v25.16b, v20.16b            \n"
                    "mov    v26.16b, v20.16b            \n"
                    "mov    v27.16b, v20.16b            \n"
                    "mov    v28.16b, v20.16b            \n"
                    "mov    v29.16b, v20.16b            \n"
                    "mov    v30.16b, v20.16b            \n"
                    "mov    v31.16b, v20.16b            \n"

                    "3:                                 \n"
                    "lsr    w4, %w9, #2                 \n" // w4 = max_kk >> 2
                    "cmp    w4, #0                      \n"
                    "beq    5f                          \n"

                    "prfm   pldl1keep, [%1, #512]       \n"
                    "ld1    {v4.4h, v5.4h}, [%1], #16   \n"

                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v0.4h, v1.4h, v2.4h}, [%2], #24 \n"

                    "shll   v4.4s, v4.4h, #16           \n"
                    "shll   v0.4s, v0.4h, #16           \n"

                    ".align 4                           \n"
                    "4:                                 \n"
                    "shll   v5.4s, v5.4h, #16           \n"
                    "fmla   v8.4s, v4.4s, v0.s[0]       \n"
                    "ldr    d6, [%1], #8                \n"
                    "fmla   v9.4s, v4.4s, v0.s[1]       \n"
                    "ldr    d3, [%2], #8                \n"
                    "fmla   v10.4s, v4.4s, v0.s[2]      \n"
                    "ldr    d7, [%1], #8                \n"
                    "fmla   v11.4s, v4.4s, v0.s[3]      \n"
                    "shll   v1.4s, v1.4h, #16           \n"
                    "fmla   v20.4s, v5.4s, v0.s[0]      \n"
                    "fmla   v21.4s, v5.4s, v0.s[1]      \n"
                    "fmla   v22.4s, v5.4s, v0.s[2]      \n"
                    "fmla   v23.4s, v5.4s, v0.s[3]      \n"
                    "fmla   v12.4s, v4.4s, v1.s[0]      \n"
                    "ldr    d0, [%2], #8                \n"
                    "fmla   v13.4s, v4.4s, v1.s[1]      \n"
                    "fmla   v14.4s, v4.4s, v1.s[2]      \n"
                    "fmla   v15.4s, v4.4s, v1.s[3]      \n"
                    "shll   v2.4s, v2.4h, #16           \n"
                    "fmla   v24.4s, v5.4s, v1.s[0]      \n"
                    "fmla   v25.4s, v5.4s, v1.s[1]      \n"
                    "fmla   v26.4s, v5.4s, v1.s[2]      \n"
                    "fmla   v27.4s, v5.4s, v1.s[3]      \n"
                    "shll   v6.4s, v6.4h, #16           \n"
                    "fmla   v16.4s, v4.4s, v2.s[0]      \n"
                    "ldr    d1, [%2], #8                \n"
                    "fmla   v17.4s, v4.4s, v2.s[1]      \n"
                    "fmla   v18.4s, v4.4s, v2.s[2]      \n"
                    "fmla   v19.4s, v4.4s, v2.s[3]      \n"
                    "shll   v3.4s, v3.4h, #16           \n"
                    "fmla   v28.4s, v5.4s, v2.s[0]      \n"
                    "ldr    d4, [%1], #8                \n"
                    "fmla   v29.4s, v5.4s, v2.s[1]      \n"
                    "fmla   v30.4s, v5.4s, v2.s[2]      \n"
                    "fmla   v31.4s, v5.4s, v2.s[3]      \n"
                    "shll   v7.4s, v7.4h, #16           \n"
                    "fmla   v8.4s, v6.4s, v3.s[0]       \n"
                    "ldr    d2, [%2], #8                \n"
                    "fmla   v9.4s, v6.4s, v3.s[1]       \n"
                    "ldr    d5, [%1], #8                \n"
                    "fmla   v10.4s, v6.4s, v3.s[2]      \n"
                    "fmla   v11.4s, v6.4s, v3.s[3]      \n"
                    "shll   v0.4s, v0.4h, #16           \n"
                    "fmla   v20.4s, v7.4s, v3.s[0]      \n"
                    "fmla   v21.4s, v7.4s, v3.s[1]      \n"
                    "fmla   v22.4s, v7.4s, v3.s[2]      \n"
                    "fmla   v23.4s, v7.4s, v3.s[3]      \n"
                    "fmla   v12.4s, v6.4s, v0.s[0]      \n"
                    "ldr    d3, [%2], #8                \n"
                    "fmla   v13.4s, v6.4s, v0.s[1]      \n"
                    "fmla   v14.4s, v6.4s, v0.s[2]      \n"
                    "fmla   v15.4s, v6.4s, v0.s[3]      \n"
                    "shll   v1.4s, v1.4h, #16           \n"
                    "fmla   v24.4s, v7.4s, v0.s[0]      \n"
                    "fmla   v25.4s, v7.4s, v0.s[1]      \n"
                    "fmla   v26.4s, v7.4s, v0.s[2]      \n"
                    "fmla   v27.4s, v7.4s, v0.s[3]      \n"
                    "shll   v4.4s, v4.4h, #16           \n"
                    "fmla   v16.4s, v6.4s, v1.s[0]      \n"
                    "ldr    d0, [%2], #8                \n"
                    "fmla   v17.4s, v6.4s, v1.s[1]      \n"
                    "fmla   v18.4s, v6.4s, v1.s[2]      \n"
                    "fmla   v19.4s, v6.4s, v1.s[3]      \n"
                    "shll   v2.4s, v2.4h, #16           \n"
                    "fmla   v28.4s, v7.4s, v1.s[0]      \n"
                    "ldr    d6, [%1], #8                \n"
                    "fmla   v29.4s, v7.4s, v1.s[1]      \n"
                    "fmla   v30.4s, v7.4s, v1.s[2]      \n"
                    "fmla   v31.4s, v7.4s, v1.s[3]      \n"
                    "shll   v5.4s, v5.4h, #16           \n"
                    "fmla   v8.4s, v4.4s, v2.s[0]       \n"
                    "ldr    d1, [%2], #8                \n"
                    "fmla   v9.4s, v4.4s, v2.s[1]       \n"
                    "ldr    d7, [%1], #8                \n"
                    "fmla   v10.4s, v4.4s, v2.s[2]      \n"
                    "fmla   v11.4s, v4.4s, v2.s[3]      \n"
                    "shll   v3.4s, v3.4h, #16           \n"
                    "fmla   v20.4s, v5.4s, v2.s[0]      \n"
                    "fmla   v21.4s, v5.4s, v2.s[1]      \n"
                    "fmla   v22.4s, v5.4s, v2.s[2]      \n"
                    "fmla   v23.4s, v5.4s, v2.s[3]      \n"
                    "fmla   v12.4s, v4.4s, v3.s[0]      \n"
                    "ldr    d2, [%2], #8                \n"
                    "fmla   v13.4s, v4.4s, v3.s[1]      \n"
                    "fmla   v14.4s, v4.4s, v3.s[2]      \n"
                    "fmla   v15.4s, v4.4s, v3.s[3]      \n"
                    "shll   v0.4s, v0.4h, #16           \n"
                    "fmla   v24.4s, v5.4s, v3.s[0]      \n"
                    "fmla   v25.4s, v5.4s, v3.s[1]      \n"
                    "fmla   v26.4s, v5.4s, v3.s[2]      \n"
                    "fmla   v27.4s, v5.4s, v3.s[3]      \n"
                    "shll   v6.4s, v6.4h, #16           \n"
                    "fmla   v16.4s, v4.4s, v0.s[0]      \n"
                    "ldr    d3, [%2], #8                \n"
                    "fmla   v17.4s, v4.4s, v0.s[1]      \n"
                    "fmla   v18.4s, v4.4s, v0.s[2]      \n"
                    "prfm   pldl1keep, [%1, #512]       \n" // NOTE PRELOAD
                    "fmla   v19.4s, v4.4s, v0.s[3]      \n"
                    "shll   v1.4s, v1.4h, #16           \n"
                    "fmla   v28.4s, v5.4s, v0.s[0]      \n"
                    "ldr    d4, [%1], #8                \n"
                    "fmla   v29.4s, v5.4s, v0.s[1]      \n"
                    "fmla   v30.4s, v5.4s, v0.s[2]      \n"
                    "prfm   pldl1keep, [%2, #512]       \n" // NOTE PRELOAD
                    "fmla   v31.4s, v5.4s, v0.s[3]      \n"
                    "shll   v7.4s, v7.4h, #16           \n"
                    "fmla   v8.4s, v6.4s, v1.s[0]       \n"
                    "ldr    d0, [%2], #8                \n"
                    "fmla   v9.4s, v6.4s, v1.s[1]       \n"
                    "ldr    d5, [%1], #8                 \n"
                    "fmla   v10.4s, v6.4s, v1.s[2]      \n"
                    "fmla   v11.4s, v6.4s, v1.s[3]      \n"
                    "shll   v2.4s, v2.4h, #16           \n"
                    "fmla   v20.4s, v7.4s, v1.s[0]      \n"
                    "fmla   v21.4s, v7.4s, v1.s[1]      \n"
                    "fmla   v22.4s, v7.4s, v1.s[2]      \n"
                    "fmla   v23.4s, v7.4s, v1.s[3]      \n"
                    "fmla   v12.4s, v6.4s, v2.s[0]      \n"
                    "ldr    d1, [%2], #8                 \n"
                    "fmla   v13.4s, v6.4s, v2.s[1]      \n"
                    "fmla   v14.4s, v6.4s, v2.s[2]      \n"
                    "fmla   v15.4s, v6.4s, v2.s[3]      \n"
                    "shll   v3.4s, v3.4h, #16           \n"
                    "fmla   v24.4s, v7.4s, v2.s[0]      \n"
                    "fmla   v25.4s, v7.4s, v2.s[1]      \n"
                    "fmla   v26.4s, v7.4s, v2.s[2]      \n"
                    "fmla   v27.4s, v7.4s, v2.s[3]      \n"
                    "shll   v4.4s, v4.4h, #16           \n"
                    "fmla   v16.4s, v6.4s, v3.s[0]      \n"
                    "ldr    d2, [%2], #8                \n"
                    "fmla   v17.4s, v6.4s, v3.s[1]      \n"
                    "fmla   v18.4s, v6.4s, v3.s[2]      \n"
                    "fmla   v19.4s, v6.4s, v3.s[3]      \n"
                    "shll   v0.4s, v0.4h, #16           \n"
                    "fmla   v28.4s, v7.4s, v3.s[0]      \n"
                    "fmla   v29.4s, v7.4s, v3.s[1]      \n"
                    "subs   w4, w4, #1                  \n"
                    "fmla   v30.4s, v7.4s, v3.s[2]      \n"
                    "fmla   v31.4s, v7.4s, v3.s[3]      \n"
                    "bne    4b                          \n"

                    "sub    %1, %1, #16                 \n"
                    "sub    %2, %2, #24                 \n"

                    "5:                                 \n"
                    "and    w4, %w9, #3                 \n" // w4 = remain = max_kk & 3
                    "cmp    w4, #0                      \n"
                    "beq    7f                          \n"

                    "6:                                 \n"
                    "ld1    {v0.4h, v1.4h, v2.4h}, [%2], #24 \n"

                    "shll   v0.4s, v0.4h, #16           \n"
                    "shll   v1.4s, v1.4h, #16           \n"
                    "shll   v2.4s, v2.4h, #16           \n"

                    "ld1    {v4.4h, v5.4h}, [%1], #16   \n"

                    "shll   v4.4s, v4.4h, #16           \n"
                    "shll   v5.4s, v5.4h, #16           \n"

                    "fmla   v8.4s, v4.4s, v0.s[0]       \n"
                    "fmla   v9.4s, v4.4s, v0.s[1]       \n"
                    "fmla   v10.4s, v4.4s, v0.s[2]      \n"
                    "fmla   v11.4s, v4.4s, v0.s[3]      \n"
                    "fmla   v12.4s, v4.4s, v1.s[0]      \n"
                    "fmla   v13.4s, v4.4s, v1.s[1]      \n"
                    "fmla   v14.4s, v4.4s, v1.s[2]      \n"
                    "fmla   v15.4s, v4.4s, v1.s[3]      \n"
                    "fmla   v16.4s, v4.4s, v2.s[0]      \n"
                    "fmla   v17.4s, v4.4s, v2.s[1]      \n"
                    "fmla   v18.4s, v4.4s, v2.s[2]      \n"
                    "fmla   v19.4s, v4.4s, v2.s[3]      \n"

                    "subs   w4, w4, #1                  \n"

                    "fmla   v20.4s, v5.4s, v0.s[0]      \n"
                    "fmla   v21.4s, v5.4s, v0.s[1]      \n"
                    "fmla   v22.4s, v5.4s, v0.s[2]      \n"
                    "fmla   v23.4s, v5.4s, v0.s[3]      \n"
                    "fmla   v24.4s, v5.4s, v1.s[0]      \n"
                    "fmla   v25.4s, v5.4s, v1.s[1]      \n"
                    "fmla   v26.4s, v5.4s, v1.s[2]      \n"
                    "fmla   v27.4s, v5.4s, v1.s[3]      \n"
                    "fmla   v28.4s, v5.4s, v2.s[0]      \n"
                    "fmla   v29.4s, v5.4s, v2.s[1]      \n"
                    "fmla   v30.4s, v5.4s, v2.s[2]      \n"
                    "fmla   v31.4s, v5.4s, v2.s[3]      \n"

                    "bne    6b                          \n"

                    "7:                                 \n"
                    "shrn   v0.4h, v8.4s, #16           \n"
                    "shrn2  v0.8h, v9.4s, #16           \n"
                    "shrn   v1.4h, v10.4s, #16          \n"
                    "shrn2  v1.8h, v11.4s, #16          \n"
                    "shrn   v2.4h, v12.4s, #16          \n"
                    "shrn2  v2.8h, v13.4s, #16          \n"
                    "shrn   v3.4h, v14.4s, #16          \n"
                    "shrn2  v3.8h, v15.4s, #16          \n"
                    "shrn   v4.4h, v16.4s, #16          \n"
                    "shrn2  v4.8h, v17.4s, #16          \n"
                    "shrn   v5.4h, v18.4s, #16          \n"
                    "shrn2  v5.8h, v19.4s, #16          \n"
                    "shrn   v6.4h, v20.4s, #16          \n"
                    "shrn2  v6.8h, v21.4s, #16          \n"
                    "shrn   v7.4h, v22.4s, #16          \n"
                    "shrn2  v7.8h, v23.4s, #16          \n"
                    "shrn   v8.4h, v24.4s, #16          \n"
                    "shrn2  v8.8h, v25.4s, #16          \n"
                    "shrn   v9.4h, v26.4s, #16          \n"
                    "shrn2  v9.8h, v27.4s, #16          \n"
                    "shrn   v10.4h, v28.4s, #16         \n"
                    "shrn2  v10.8h, v29.4s, #16         \n"
                    "shrn   v11.4h, v30.4s, #16         \n"
                    "shrn2  v11.8h, v31.4s, #16         \n"
                    "tst    %w11, #255                  \n"
                    "beq    10f                         \n"

                    // if out_elempack == 4
                    "cmp    %w12, #4                    \n"
                    "bne    8f                          \n"

                    "lsl    w4, %w13, #2                \n"
                    "add    x4, %3, w4, sxtw 1          \n"
                    "st1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%3], #64 \n"
                    "st1    {v4.8h, v5.8h}, [%3], #32 \n"
                    "st1    {v6.8h, v7.8h, v8.8h, v9.8h}, [x4], #64 \n"
                    "st1    {v10.8h, v11.8h}, [x4]      \n"
                    "b      9f                          \n"

                    // if out_elempack == 1
                    "8:                                 \n"
                    // transpose8x12
                    "uzp1   v20.8h, v0.8h, v1.8h        \n"
                    "uzp2   v21.8h, v0.8h, v1.8h        \n"
                    "uzp1   v22.8h, v2.8h, v3.8h        \n"
                    "uzp2   v23.8h, v2.8h, v3.8h        \n"
                    "uzp1   v24.8h, v4.8h, v5.8h        \n"
                    "uzp2   v25.8h, v4.8h, v5.8h        \n"
                    "uzp1   v26.8h, v6.8h, v7.8h        \n"
                    "uzp2   v27.8h, v6.8h, v7.8h        \n"
                    "uzp1   v28.8h, v8.8h, v9.8h        \n"
                    "uzp2   v29.8h, v8.8h, v9.8h        \n"
                    "uzp1   v30.8h, v10.8h, v11.8h      \n"
                    "uzp2   v31.8h, v10.8h, v11.8h      \n"

                    "uzp1   v0.8h, v20.8h, v22.8h       \n"
                    "uzp2   v6.8h, v20.8h, v22.8h       \n"
                    "uzp1   v3.8h, v21.8h, v23.8h       \n"
                    "uzp2   v9.8h, v21.8h, v23.8h       \n"
                    "mov    v1.d[0], v0.d[1]            \n"
                    "mov    v7.d[0], v6.d[1]            \n"
                    "mov    v4.d[0], v3.d[1]            \n"
                    "mov    v10.d[0], v9.d[1]           \n"
                    "uzp1   v2.8h, v24.8h, v24.8h       \n"
                    "uzp2   v8.8h, v24.8h, v24.8h       \n"
                    "uzp1   v5.8h, v25.8h, v25.8h       \n"
                    "uzp2   v11.8h, v25.8h, v25.8h      \n"

                    "uzp1   v12.8h, v26.8h, v28.8h      \n"
                    "uzp2   v18.8h, v26.8h, v28.8h      \n"
                    "uzp1   v15.8h, v27.8h, v29.8h      \n"
                    "uzp2   v21.8h, v27.8h, v29.8h      \n"
                    "mov    v13.d[0], v12.d[1]          \n"
                    "mov    v19.d[0], v18.d[1]          \n"
                    "mov    v16.d[0], v15.d[1]          \n"
                    "mov    v22.d[0], v21.d[1]          \n"
                    "uzp1   v14.8h, v30.8h, v30.8h      \n"
                    "uzp2   v20.8h, v30.8h, v30.8h      \n"
                    "uzp1   v17.8h, v31.8h, v31.8h      \n"
                    "uzp2   v23.8h, v31.8h, v31.8h      \n"

                    "add    x4, %3, %w13, sxtw 1        \n"
                    "st1    {v0.4h, v1.4h, v2.4h}, [%3], #24 \n"
                    "st1    {v3.4h, v4.4h, v5.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v6.4h, v7.4h, v8.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v9.4h, v10.4h, v11.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v12.4h, v13.4h, v14.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v15.4h, v16.4h, v17.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v18.4h, v19.4h, v20.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v21.4h, v22.4h, v23.4h}, [x4] \n"

                    "9:                                 \n"
                    "add    %0, %0, #384                \n"
                    "b      11f                         \n"

                    "10:                                \n"
                    "st1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%0], #64   \n"
                    "st1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%0], #64 \n"
                    "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                    "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                    "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                    "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0], #64 \n"

                    "11:                                \n"

                    : "=r"(outptr), // %0
                    "=r"(pA),     // %1
                    "=r"(pB),     // %2
                    "=r"(outptr0) // %3
                    : "0"(outptr),
                    "1"(pA),
                    "2"(pB),
                    "3"(outptr0),
                    "r"(pC),           // %8
                    "r"(max_kk),       // %9
                    "r"(k),            // %10
                    "r"(k_end),        // %11
                    "r"(out_elempack), // %12
                    "r"(out_hstep)     // %13
                    : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
            }
            else if (use_a53_a55_optimized_kernel && !cpu_support_arm_asimdhp())
            {
                // a53
                asm volatile(
                    "cbz    %w10, 0f                    \n"

                    "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%0], #64   \n"
                    "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%0], #64 \n"
                    "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                    "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                    "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                    "ld1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0]      \n"
                    "subs   %0, %0, #320                \n"
                    "b      3f                          \n"

                    "0:                                 \n"
                    // if pC
                    "cbz    %8, 1f                      \n"

                    "add    x4, %8, #16                 \n"
                    "ld1    {v8.4s}, [%8]               \n"
                    "ld1    {v20.4s}, [x4]              \n"
                    "b      2f                          \n"

                    // else
                    "1:                                 \n"
                    "eor    v8.16b, v8.16b, v8.16b      \n"
                    "eor    v20.16b, v20.16b, v20.16b   \n"

                    "2:                                 \n"
                    "mov    v9.16b, v8.16b              \n"
                    "mov    v10.16b, v8.16b             \n"
                    "mov    v11.16b, v8.16b             \n"
                    "mov    v12.16b, v8.16b             \n"
                    "mov    v13.16b, v8.16b             \n"
                    "mov    v14.16b, v8.16b             \n"
                    "mov    v15.16b, v8.16b             \n"
                    "mov    v16.16b, v8.16b             \n"
                    "mov    v17.16b, v8.16b             \n"
                    "mov    v18.16b, v8.16b             \n"
                    "mov    v19.16b, v8.16b             \n"

                    "mov    v21.16b, v20.16b            \n"
                    "mov    v22.16b, v20.16b            \n"
                    "mov    v23.16b, v20.16b            \n"
                    "mov    v24.16b, v20.16b            \n"
                    "mov    v25.16b, v20.16b            \n"
                    "mov    v26.16b, v20.16b            \n"
                    "mov    v27.16b, v20.16b            \n"
                    "mov    v28.16b, v20.16b            \n"
                    "mov    v29.16b, v20.16b            \n"
                    "mov    v30.16b, v20.16b            \n"
                    "mov    v31.16b, v20.16b            \n"

                    "3:                                 \n"
                    "lsr    w4, %w9, #2                 \n" // w4 = max_kk >> 2
                    "cmp    w4, #0                      \n"
                    "beq    5f                          \n"

                    "prfm   pldl1keep, [%1, #256]       \n"
                    "ld1    {v4.4h}, [%1], #8           \n"

                    "prfm   pldl1keep, [%2, #384]       \n"
                    "ld1    {v0.4h, v1.4h, v2.4h}, [%2], #24 \n"

                    "ldr    x25, [%1]                   \n"
                    "add    %1, %1, #8                  \n"

                    "shll   v4.4s, v4.4h, #16           \n"
                    "shll   v0.4s, v0.4h, #16           \n"

                    ".align 4                           \n"
                    "4:                                 \n"

                    "shll   v1.4s, v1.4h, #16           \n"
                    "fmla   v8.4s, v4.4s, v0.s[0]       \n"
                    "ldr    x26, [%1]                   \n"
                    "fmla   v9.4s, v4.4s, v0.s[1]       \n"
                    "add    %1, %1, #8                  \n"
                    "fmla   v10.4s, v4.4s, v0.s[2]      \n"

                    "shll   v2.4s, v2.4h, #16           \n"
                    "fmla   v11.4s, v4.4s, v0.s[3]      \n"
                    "ldr    x23, [%2]                   \n"
                    "fmla   v12.4s, v4.4s, v1.s[0]      \n"
                    "add    %2, %2, #8                  \n"
                    "fmla   v13.4s, v4.4s, v1.s[1]      \n"

                    "nop                                \n"
                    "ins    v5.d[0], x25                \n"
                    "fmla   v14.4s, v4.4s, v1.s[2]      \n"
                    "ldr    x20, [%2]                   \n"
                    "fmla   v15.4s, v4.4s, v1.s[3]      \n"
                    "add    %2, %2, #8                  \n"
                    "fmla   v16.4s, v4.4s, v2.s[0]      \n"

                    "shll   v5.4s, v5.4h, #16           \n"
                    "fmla   v17.4s, v4.4s, v2.s[1]      \n"
                    "ldr    x21, [%2]                   \n"
                    "fmla   v18.4s, v4.4s, v2.s[2]      \n"
                    "add    %2, %2, #8                  \n"
                    "fmla   v19.4s, v4.4s, v2.s[3]      \n"

                    "ins    v6.d[0], x26                \n"
                    "ins    v3.d[0], x23                \n"
                    "fmla   v20.4s, v5.4s, v0.s[0]      \n"
                    "prfm   pldl1keep, [%2, #384]       \n" // NOTE PRELOAD
                    "fmla   v21.4s, v5.4s, v0.s[1]      \n"
                    "nop                                \n"
                    "fmla   v22.4s, v5.4s, v0.s[2]      \n"

                    "shll   v6.4s, v6.4h, #16           \n"
                    "fmla   v23.4s, v5.4s, v0.s[3]      \n"
                    "ldr    x27, [%1]                   \n"
                    "fmla   v24.4s, v5.4s, v1.s[0]      \n"
                    "add    %1, %1, #8                  \n"
                    "fmla   v25.4s, v5.4s, v1.s[1]      \n"

                    "shll   v3.4s, v3.4h, #16           \n"
                    "fmla   v26.4s, v5.4s, v1.s[2]      \n"
                    "prfm   pldl1keep, [%1, #256]       \n" // NOTE PRELOAD
                    "fmla   v27.4s, v5.4s, v1.s[3]      \n"
                    "nop                                \n"
                    "fmla   v28.4s, v5.4s, v2.s[0]      \n"

                    "ins    v0.d[0], x20                \n"
                    "ins    v1.d[0], x21                \n"
                    "fmla   v29.4s, v5.4s, v2.s[1]      \n"
                    "nop                                \n"
                    "fmla   v30.4s, v5.4s, v2.s[2]      \n"
                    "nop                                \n"
                    "fmla   v31.4s, v5.4s, v2.s[3]      \n"

                    "shll   v0.4s, v0.4h, #16           \n"
                    "fmla   v8.4s, v6.4s, v3.s[0]       \n"
                    "ldr    x24, [%1]                   \n"
                    "fmla   v9.4s, v6.4s, v3.s[1]       \n"
                    "add    %1, %1, #8                  \n"
                    "fmla   v10.4s, v6.4s, v3.s[2]      \n"

                    "shll   v1.4s, v1.4h, #16           \n"
                    "fmla   v11.4s, v6.4s, v3.s[3]      \n"
                    "ldr    x22, [%2]                   \n"
                    "fmla   v12.4s, v6.4s, v0.s[0]      \n"
                    "add    %2, %2, #8                  \n"
                    "fmla   v13.4s, v6.4s, v0.s[1]      \n"

                    "nop                                \n"
                    "ins    v7.d[0], x27                \n"
                    "fmla   v14.4s, v6.4s, v0.s[2]      \n"
                    "ldr    x23, [%2]                   \n"
                    "fmla   v15.4s, v6.4s, v0.s[3]      \n"
                    "add    %2, %2, #8                  \n"
                    "fmla   v16.4s, v6.4s, v1.s[0]      \n"

                    "shll   v7.4s, v7.4h, #16           \n"
                    "fmla   v17.4s, v6.4s, v1.s[1]      \n"
                    "ldr    x20, [%2]                   \n"
                    "fmla   v18.4s, v6.4s, v1.s[2]      \n"
                    "add    %2, %2, #8                  \n"
                    "fmla   v19.4s, v6.4s, v1.s[3]      \n"

                    "ins    v4.d[0], x24                \n"
                    "ins    v2.d[0], x22                \n"
                    "fmla   v20.4s, v7.4s, v3.s[0]      \n"
                    "nop                                \n"
                    "fmla   v21.4s, v7.4s, v3.s[1]      \n"
                    "nop                                \n"
                    "fmla   v22.4s, v7.4s, v3.s[2]      \n"

                    "shll   v4.4s, v4.4h, #16           \n"
                    "fmla   v23.4s, v7.4s, v3.s[3]      \n"
                    "ldr    x25, [%1]                   \n"
                    "fmla   v24.4s, v7.4s, v0.s[0]      \n"
                    "add    %1, %1, #8                  \n"
                    "fmla   v25.4s, v7.4s, v0.s[1]      \n"

                    "shll   v2.4s, v2.4h, #16           \n"
                    "fmla   v26.4s, v7.4s, v0.s[2]      \n"
                    "nop                                \n"
                    "fmla   v27.4s, v7.4s, v0.s[3]      \n"
                    "nop                                \n"
                    "fmla   v28.4s, v7.4s, v1.s[0]      \n"

                    "ins    v3.d[0], x23                \n"
                    "ins    v0.d[0], x20                \n"
                    "fmla   v29.4s, v7.4s, v1.s[1]      \n"
                    "nop                                \n"
                    "fmla   v30.4s, v7.4s, v1.s[2]      \n"
                    "nop                                \n"
                    "fmla   v31.4s, v7.4s, v1.s[3]      \n"

                    "shll   v3.4s, v3.4h, #16           \n"
                    "fmla   v8.4s, v4.4s, v2.s[0]       \n"
                    "ldr    x26, [%1]                   \n"
                    "fmla   v9.4s, v4.4s, v2.s[1]       \n"
                    "add    %1, %1, #8                  \n"
                    "fmla   v10.4s, v4.4s, v2.s[2]      \n"

                    "shll   v0.4s, v0.4h, #16           \n"
                    "fmla   v11.4s, v4.4s, v2.s[3]      \n"
                    "ldr    x21, [%2]                   \n"
                    "fmla   v12.4s, v4.4s, v3.s[0]      \n"
                    "add    %2, %2, #8                  \n"
                    "fmla   v13.4s, v4.4s, v3.s[1]      \n"

                    "nop                                \n"
                    "ins    v5.d[0], x25                \n"
                    "fmla   v14.4s, v4.4s, v3.s[2]      \n"
                    "ldr    x22, [%2]                   \n"
                    "fmla   v15.4s, v4.4s, v3.s[3]      \n"
                    "add    %2, %2, #8                  \n"
                    "fmla   v16.4s, v4.4s, v0.s[0]      \n"

                    "shll   v5.4s, v5.4h, #16           \n"
                    "fmla   v17.4s, v4.4s, v0.s[1]      \n"
                    "ldr    x23, [%2]                   \n"
                    "fmla   v18.4s, v4.4s, v0.s[2]      \n"
                    "add    %2, %2, #8                  \n"
                    "fmla   v19.4s, v4.4s, v0.s[3]      \n"

                    "ins    v6.d[0], x26                \n"
                    "ins    v1.d[0], x21                \n"
                    "fmla   v20.4s, v5.4s, v2.s[0]      \n"
                    "prfm   pldl1keep, [%2, #384]       \n" // NOTE PRELOAD
                    "fmla   v21.4s, v5.4s, v2.s[1]      \n"
                    "nop                                \n"
                    "fmla   v22.4s, v5.4s, v2.s[2]      \n"

                    "shll   v6.4s, v6.4h, #16           \n"
                    "fmla   v23.4s, v5.4s, v2.s[3]      \n"
                    "ldr    x27, [%1]                   \n"
                    "fmla   v24.4s, v5.4s, v3.s[0]      \n"
                    "add    %1, %1, #8                  \n"
                    "fmla   v25.4s, v5.4s, v3.s[1]      \n"

                    "shll   v1.4s, v1.4h, #16           \n"
                    "fmla   v26.4s, v5.4s, v3.s[2]      \n"
                    "prfm   pldl1keep, [%1, #256]       \n" // NOTE PRELOAD
                    "fmla   v27.4s, v5.4s, v3.s[3]      \n"
                    "nop                                \n"
                    "fmla   v28.4s, v5.4s, v0.s[0]      \n"

                    "ins    v2.d[0], x22                \n"
                    "ins    v3.d[0], x23                \n"
                    "fmla   v29.4s, v5.4s, v0.s[1]      \n"
                    "nop                                \n"
                    "fmla   v30.4s, v5.4s, v0.s[2]      \n"
                    "nop                                \n"
                    "fmla   v31.4s, v5.4s, v0.s[3]      \n"

                    "shll   v2.4s, v2.4h, #16           \n"
                    "fmla   v8.4s, v6.4s, v1.s[0]       \n"
                    "ldr    x24, [%1]                   \n"
                    "fmla   v9.4s, v6.4s, v1.s[1]       \n"
                    "add    %1, %1, #8                  \n"
                    "fmla   v10.4s, v6.4s, v1.s[2]      \n"

                    "shll   v3.4s, v3.4h, #16           \n"
                    "fmla   v11.4s, v6.4s, v1.s[3]      \n"
                    "ldr    x20, [%2]                   \n"
                    "fmla   v12.4s, v6.4s, v2.s[0]      \n"
                    "add    %2, %2, #8                  \n"
                    "fmla   v13.4s, v6.4s, v2.s[1]      \n"

                    "nop                                \n"
                    "ins    v7.d[0], x27                \n"
                    "fmla   v14.4s, v6.4s, v2.s[2]      \n"
                    "ldr    x21, [%2]                   \n"
                    "fmla   v15.4s, v6.4s, v2.s[3]      \n"
                    "add    %2, %2, #8                  \n"
                    "fmla   v16.4s, v6.4s, v3.s[0]      \n"

                    "shll   v7.4s, v7.4h, #16           \n"
                    "fmla   v17.4s, v6.4s, v3.s[1]      \n"
                    "ldr    x22, [%2]                   \n"
                    "fmla   v18.4s, v6.4s, v3.s[2]      \n"
                    "add    %2, %2, #8                  \n"
                    "fmla   v19.4s, v6.4s, v3.s[3]      \n"

                    "ins    v4.d[0], x24                \n"
                    "ins    v0.d[0], x20                \n"
                    "fmla   v20.4s, v7.4s, v1.s[0]      \n"
                    "nop                                \n"
                    "fmla   v21.4s, v7.4s, v1.s[1]      \n"
                    "nop                                \n"
                    "fmla   v22.4s, v7.4s, v1.s[2]      \n"

                    "shll   v4.4s, v4.4h, #16           \n"
                    "fmla   v23.4s, v7.4s, v1.s[3]      \n"
                    "ldr    x25, [%1]                   \n"
                    "fmla   v24.4s, v7.4s, v2.s[0]      \n"
                    "add    %1, %1, #8                  \n"
                    "fmla   v25.4s, v7.4s, v2.s[1]      \n"

                    "shll   v0.4s, v0.4h, #16           \n"
                    "fmla   v26.4s, v7.4s, v2.s[2]      \n"
                    "nop                                \n"
                    "fmla   v27.4s, v7.4s, v2.s[3]      \n"
                    "nop                                \n"
                    "fmla   v28.4s, v7.4s, v3.s[0]      \n"

                    "ins    v1.d[0], x21                \n"
                    "ins    v2.d[0], x22                \n"
                    "fmla   v29.4s, v7.4s, v3.s[1]      \n"
                    "subs   w4, w4, #1                  \n"
                    "fmla   v30.4s, v7.4s, v3.s[2]      \n"
                    "nop                                \n"
                    "fmla   v31.4s, v7.4s, v3.s[3]      \n"

                    "bne    4b                          \n"

                    "sub    %1, %1, #16                 \n"
                    "sub    %2, %2, #24                 \n"

                    "5:                                 \n"
                    "and    w4, %w9, #3                 \n" // w4 = remain = max_kk & 3
                    "cmp    w4, #0                      \n"
                    "beq    7f                          \n"

                    "6:                                 \n"
                    "ld1    {v0.4h, v1.4h, v2.4h}, [%2], #24 \n"

                    "shll   v0.4s, v0.4h, #16           \n"
                    "shll   v1.4s, v1.4h, #16           \n"
                    "shll   v2.4s, v2.4h, #16           \n"

                    "ld1    {v4.4h, v5.4h}, [%1], #16   \n"

                    "shll   v4.4s, v4.4h, #16           \n"
                    "shll   v5.4s, v5.4h, #16           \n"

                    "fmla   v8.4s, v4.4s, v0.s[0]       \n"
                    "fmla   v9.4s, v4.4s, v0.s[1]       \n"
                    "fmla   v10.4s, v4.4s, v0.s[2]      \n"
                    "fmla   v11.4s, v4.4s, v0.s[3]      \n"
                    "fmla   v12.4s, v4.4s, v1.s[0]      \n"
                    "fmla   v13.4s, v4.4s, v1.s[1]      \n"
                    "fmla   v14.4s, v4.4s, v1.s[2]      \n"
                    "fmla   v15.4s, v4.4s, v1.s[3]      \n"
                    "fmla   v16.4s, v4.4s, v2.s[0]      \n"
                    "fmla   v17.4s, v4.4s, v2.s[1]      \n"
                    "fmla   v18.4s, v4.4s, v2.s[2]      \n"
                    "fmla   v19.4s, v4.4s, v2.s[3]      \n"

                    "subs   w4, w4, #1                  \n"

                    "fmla   v20.4s, v5.4s, v0.s[0]      \n"
                    "fmla   v21.4s, v5.4s, v0.s[1]      \n"
                    "fmla   v22.4s, v5.4s, v0.s[2]      \n"
                    "fmla   v23.4s, v5.4s, v0.s[3]      \n"
                    "fmla   v24.4s, v5.4s, v1.s[0]      \n"
                    "fmla   v25.4s, v5.4s, v1.s[1]      \n"
                    "fmla   v26.4s, v5.4s, v1.s[2]      \n"
                    "fmla   v27.4s, v5.4s, v1.s[3]      \n"
                    "fmla   v28.4s, v5.4s, v2.s[0]      \n"
                    "fmla   v29.4s, v5.4s, v2.s[1]      \n"
                    "fmla   v30.4s, v5.4s, v2.s[2]      \n"
                    "fmla   v31.4s, v5.4s, v2.s[3]      \n"

                    "bne    6b                          \n"

                    "7:                                 \n"
                    "shrn   v0.4h, v8.4s, #16           \n"
                    "shrn2  v0.8h, v9.4s, #16           \n"
                    "shrn   v1.4h, v10.4s, #16          \n"
                    "shrn2  v1.8h, v11.4s, #16          \n"
                    "shrn   v2.4h, v12.4s, #16          \n"
                    "shrn2  v2.8h, v13.4s, #16          \n"
                    "shrn   v3.4h, v14.4s, #16          \n"
                    "shrn2  v3.8h, v15.4s, #16          \n"
                    "shrn   v4.4h, v16.4s, #16          \n"
                    "shrn2  v4.8h, v17.4s, #16          \n"
                    "shrn   v5.4h, v18.4s, #16          \n"
                    "shrn2  v5.8h, v19.4s, #16          \n"
                    "shrn   v6.4h, v20.4s, #16          \n"
                    "shrn2  v6.8h, v21.4s, #16          \n"
                    "shrn   v7.4h, v22.4s, #16          \n"
                    "shrn2  v7.8h, v23.4s, #16          \n"
                    "shrn   v8.4h, v24.4s, #16          \n"
                    "shrn2  v8.8h, v25.4s, #16          \n"
                    "shrn   v9.4h, v26.4s, #16          \n"
                    "shrn2  v9.8h, v27.4s, #16          \n"
                    "shrn   v10.4h, v28.4s, #16         \n"
                    "shrn2  v10.8h, v29.4s, #16         \n"
                    "shrn   v11.4h, v30.4s, #16         \n"
                    "shrn2  v11.8h, v31.4s, #16         \n"
                    "tst    %w11, #255                  \n"
                    "beq    10f                         \n"

                    // if out_elempack == 4
                    "cmp    %w12, #4                    \n"
                    "bne    8f                          \n"

                    "lsl    w4, %w13, #2                \n"
                    "add    x4, %3, w4, sxtw 1          \n"
                    "st1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%3], #64 \n"
                    "st1    {v4.8h, v5.8h}, [%3], #32 \n"
                    "st1    {v6.8h, v7.8h, v8.8h, v9.8h}, [x4], #64 \n"
                    "st1    {v10.8h, v11.8h}, [x4]      \n"
                    "b      9f                          \n"

                    // if out_elempack == 1
                    "8:                                 \n"
                    // transpose8x12
                    "uzp1   v20.8h, v0.8h, v1.8h        \n"
                    "uzp2   v21.8h, v0.8h, v1.8h        \n"
                    "uzp1   v22.8h, v2.8h, v3.8h        \n"
                    "uzp2   v23.8h, v2.8h, v3.8h        \n"
                    "uzp1   v24.8h, v4.8h, v5.8h        \n"
                    "uzp2   v25.8h, v4.8h, v5.8h        \n"
                    "uzp1   v26.8h, v6.8h, v7.8h        \n"
                    "uzp2   v27.8h, v6.8h, v7.8h        \n"
                    "uzp1   v28.8h, v8.8h, v9.8h        \n"
                    "uzp2   v29.8h, v8.8h, v9.8h        \n"
                    "uzp1   v30.8h, v10.8h, v11.8h      \n"
                    "uzp2   v31.8h, v10.8h, v11.8h      \n"

                    "uzp1   v0.8h, v20.8h, v22.8h       \n"
                    "uzp2   v6.8h, v20.8h, v22.8h       \n"
                    "uzp1   v3.8h, v21.8h, v23.8h       \n"
                    "uzp2   v9.8h, v21.8h, v23.8h       \n"
                    "mov    v1.d[0], v0.d[1]            \n"
                    "mov    v7.d[0], v6.d[1]            \n"
                    "mov    v4.d[0], v3.d[1]            \n"
                    "mov    v10.d[0], v9.d[1]           \n"
                    "uzp1   v2.8h, v24.8h, v24.8h       \n"
                    "uzp2   v8.8h, v24.8h, v24.8h       \n"
                    "uzp1   v5.8h, v25.8h, v25.8h       \n"
                    "uzp2   v11.8h, v25.8h, v25.8h      \n"

                    "uzp1   v12.8h, v26.8h, v28.8h      \n"
                    "uzp2   v18.8h, v26.8h, v28.8h      \n"
                    "uzp1   v15.8h, v27.8h, v29.8h      \n"
                    "uzp2   v21.8h, v27.8h, v29.8h      \n"
                    "mov    v13.d[0], v12.d[1]          \n"
                    "mov    v19.d[0], v18.d[1]          \n"
                    "mov    v16.d[0], v15.d[1]          \n"
                    "mov    v22.d[0], v21.d[1]          \n"
                    "uzp1   v14.8h, v30.8h, v30.8h      \n"
                    "uzp2   v20.8h, v30.8h, v30.8h      \n"
                    "uzp1   v17.8h, v31.8h, v31.8h      \n"
                    "uzp2   v23.8h, v31.8h, v31.8h      \n"

                    "add    x4, %3, %w13, sxtw 1        \n"
                    "st1    {v0.4h, v1.4h, v2.4h}, [%3], #24 \n"
                    "st1    {v3.4h, v4.4h, v5.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v6.4h, v7.4h, v8.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v9.4h, v10.4h, v11.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v12.4h, v13.4h, v14.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v15.4h, v16.4h, v17.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v18.4h, v19.4h, v20.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v21.4h, v22.4h, v23.4h}, [x4] \n"

                    "9:                                 \n"
                    "add    %0, %0, #384                \n"
                    "b      11f                         \n"

                    "10:                                \n"
                    "st1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%0], #64   \n"
                    "st1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%0], #64 \n"
                    "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                    "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                    "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                    "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0], #64 \n"

                    "11:                                \n"

                    : "=r"(outptr), // %0
                    "=r"(pA),     // %1
                    "=r"(pB),     // %2
                    "=r"(outptr0) // %3
                    : "0"(outptr),
                    "1"(pA),
                    "2"(pB),
                    "3"(outptr0),
                    "r"(pC),           // %8
                    "r"(max_kk),       // %9
                    "r"(k),            // %10
                    "r"(k_end),        // %11
                    "r"(out_elempack), // %12
                    "r"(out_hstep)     // %13
                    : "cc", "memory", "x4", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
            }
            else
            {
                asm volatile(
                    "cbz    %w10, 0f                    \n"

                    "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%0], #64   \n"
                    "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%0], #64 \n"
                    "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                    "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                    "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                    "ld1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0]      \n"
                    "subs   %0, %0, #320                \n"
                    "b      3f                          \n"

                    "0:                                 \n"
                    // if pC
                    "cbz    %8, 1f                      \n"

                    "add    x4, %8, #16                 \n"
                    "ld1    {v8.4s}, [%8]               \n"
                    "ld1    {v20.4s}, [x4]              \n"
                    "b      2f                          \n"

                    // else
                    "1:                                 \n"
                    "eor    v8.16b, v8.16b, v8.16b      \n"
                    "eor    v20.16b, v20.16b, v20.16b   \n"

                    "2:                                 \n"
                    "mov    v9.16b, v8.16b              \n"
                    "mov    v10.16b, v8.16b             \n"
                    "mov    v11.16b, v8.16b             \n"
                    "mov    v12.16b, v8.16b             \n"
                    "mov    v13.16b, v8.16b             \n"
                    "mov    v14.16b, v8.16b             \n"
                    "mov    v15.16b, v8.16b             \n"
                    "mov    v16.16b, v8.16b             \n"
                    "mov    v17.16b, v8.16b             \n"
                    "mov    v18.16b, v8.16b             \n"
                    "mov    v19.16b, v8.16b             \n"

                    "mov    v21.16b, v20.16b            \n"
                    "mov    v22.16b, v20.16b            \n"
                    "mov    v23.16b, v20.16b            \n"
                    "mov    v24.16b, v20.16b            \n"
                    "mov    v25.16b, v20.16b            \n"
                    "mov    v26.16b, v20.16b            \n"
                    "mov    v27.16b, v20.16b            \n"
                    "mov    v28.16b, v20.16b            \n"
                    "mov    v29.16b, v20.16b            \n"
                    "mov    v30.16b, v20.16b            \n"
                    "mov    v31.16b, v20.16b            \n"

                    "3:                                 \n"
                    "lsr    w4, %w9, #2                 \n" // w4 = max_kk >> 2
                    "cmp    w4, #0                      \n"
                    "beq    5f                          \n"

                    "4:                                 \n"
                    "prfm   pldl1keep, [%1, #256]       \n"
                    "ld1    {v4.4h, v5.4h, v6.4h, v7.4h}, [%1], #32 \n"

                    "shll   v4.4s, v4.4h, #16           \n"
                    "shll   v5.4s, v5.4h, #16           \n"
                    "shll   v6.4s, v6.4h, #16           \n"
                    "shll   v7.4s, v7.4h, #16           \n"

                    "prfm   pldl1keep, [%2, #256]       \n"
                    "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%2], #32 \n"

                    "shll   v0.4s, v0.4h, #16           \n"
                    "shll   v1.4s, v1.4h, #16           \n"
                    "shll   v2.4s, v2.4h, #16           \n"
                    "shll   v3.4s, v3.4h, #16           \n"

                    "fmla   v8.4s, v4.4s, v0.s[0]       \n"
                    "fmla   v9.4s, v4.4s, v0.s[1]       \n"
                    "fmla   v10.4s, v4.4s, v0.s[2]      \n"
                    "fmla   v11.4s, v4.4s, v0.s[3]      \n"
                    "fmla   v20.4s, v5.4s, v0.s[0]      \n"
                    "fmla   v21.4s, v5.4s, v0.s[1]      \n"
                    "fmla   v22.4s, v5.4s, v0.s[2]      \n"
                    "fmla   v23.4s, v5.4s, v0.s[3]      \n"

                    "fmla   v12.4s, v4.4s, v1.s[0]      \n"
                    "fmla   v13.4s, v4.4s, v1.s[1]      \n"
                    "fmla   v14.4s, v4.4s, v1.s[2]      \n"
                    "fmla   v15.4s, v4.4s, v1.s[3]      \n"
                    "fmla   v24.4s, v5.4s, v1.s[0]      \n"
                    "fmla   v25.4s, v5.4s, v1.s[1]      \n"
                    "fmla   v26.4s, v5.4s, v1.s[2]      \n"
                    "fmla   v27.4s, v5.4s, v1.s[3]      \n"

                    "fmla   v16.4s, v4.4s, v2.s[0]      \n"
                    "fmla   v17.4s, v4.4s, v2.s[1]      \n"
                    "fmla   v18.4s, v4.4s, v2.s[2]      \n"
                    "fmla   v19.4s, v4.4s, v2.s[3]      \n"
                    "fmla   v28.4s, v5.4s, v2.s[0]      \n"
                    "fmla   v29.4s, v5.4s, v2.s[1]      \n"
                    "fmla   v30.4s, v5.4s, v2.s[2]      \n"
                    "fmla   v31.4s, v5.4s, v2.s[3]      \n"

                    "fmla   v8.4s, v6.4s, v3.s[0]       \n"
                    "fmla   v9.4s, v6.4s, v3.s[1]       \n"
                    "fmla   v10.4s, v6.4s, v3.s[2]      \n"
                    "fmla   v11.4s, v6.4s, v3.s[3]      \n"
                    "fmla   v20.4s, v7.4s, v3.s[0]      \n"
                    "fmla   v21.4s, v7.4s, v3.s[1]      \n"
                    "fmla   v22.4s, v7.4s, v3.s[2]      \n"
                    "fmla   v23.4s, v7.4s, v3.s[3]      \n"

                    "prfm   pldl1keep, [%2, #256]       \n"
                    "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%2], #32 \n"

                    "shll   v0.4s, v0.4h, #16           \n"
                    "shll   v1.4s, v1.4h, #16           \n"
                    "shll   v2.4s, v2.4h, #16           \n"
                    "shll   v3.4s, v3.4h, #16           \n"

                    "fmla   v12.4s, v6.4s, v0.s[0]      \n"
                    "fmla   v13.4s, v6.4s, v0.s[1]      \n"
                    "fmla   v14.4s, v6.4s, v0.s[2]      \n"
                    "fmla   v15.4s, v6.4s, v0.s[3]      \n"
                    "fmla   v24.4s, v7.4s, v0.s[0]      \n"
                    "fmla   v25.4s, v7.4s, v0.s[1]      \n"
                    "fmla   v26.4s, v7.4s, v0.s[2]      \n"
                    "fmla   v27.4s, v7.4s, v0.s[3]      \n"

                    "fmla   v16.4s, v6.4s, v1.s[0]      \n"
                    "fmla   v17.4s, v6.4s, v1.s[1]      \n"
                    "fmla   v18.4s, v6.4s, v1.s[2]      \n"
                    "fmla   v19.4s, v6.4s, v1.s[3]      \n"
                    "fmla   v28.4s, v7.4s, v1.s[0]      \n"
                    "fmla   v29.4s, v7.4s, v1.s[1]      \n"
                    "fmla   v30.4s, v7.4s, v1.s[2]      \n"
                    "fmla   v31.4s, v7.4s, v1.s[3]      \n"

                    "prfm   pldl1keep, [%1, #256]       \n"
                    "ld1    {v4.4h, v5.4h, v6.4h, v7.4h}, [%1], #32 \n"

                    "shll   v4.4s, v4.4h, #16           \n"
                    "shll   v5.4s, v5.4h, #16           \n"
                    "shll   v6.4s, v6.4h, #16           \n"
                    "shll   v7.4s, v7.4h, #16           \n"

                    "fmla   v8.4s, v4.4s, v2.s[0]       \n"
                    "fmla   v9.4s, v4.4s, v2.s[1]       \n"
                    "fmla   v10.4s, v4.4s, v2.s[2]      \n"
                    "fmla   v11.4s, v4.4s, v2.s[3]      \n"
                    "fmla   v20.4s, v5.4s, v2.s[0]      \n"
                    "fmla   v21.4s, v5.4s, v2.s[1]      \n"
                    "fmla   v22.4s, v5.4s, v2.s[2]      \n"
                    "fmla   v23.4s, v5.4s, v2.s[3]      \n"

                    "fmla   v12.4s, v4.4s, v3.s[0]      \n"
                    "fmla   v13.4s, v4.4s, v3.s[1]      \n"
                    "fmla   v14.4s, v4.4s, v3.s[2]      \n"
                    "fmla   v15.4s, v4.4s, v3.s[3]      \n"
                    "fmla   v24.4s, v5.4s, v3.s[0]      \n"
                    "fmla   v25.4s, v5.4s, v3.s[1]      \n"
                    "fmla   v26.4s, v5.4s, v3.s[2]      \n"
                    "fmla   v27.4s, v5.4s, v3.s[3]      \n"

                    "prfm   pldl1keep, [%2, #256]       \n"
                    "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%2], #32 \n"

                    "shll   v0.4s, v0.4h, #16           \n"
                    "shll   v1.4s, v1.4h, #16           \n"
                    "shll   v2.4s, v2.4h, #16           \n"
                    "shll   v3.4s, v3.4h, #16           \n"

                    "fmla   v16.4s, v4.4s, v0.s[0]      \n"
                    "fmla   v17.4s, v4.4s, v0.s[1]      \n"
                    "fmla   v18.4s, v4.4s, v0.s[2]      \n"
                    "fmla   v19.4s, v4.4s, v0.s[3]      \n"
                    "fmla   v28.4s, v5.4s, v0.s[0]      \n"
                    "fmla   v29.4s, v5.4s, v0.s[1]      \n"
                    "fmla   v30.4s, v5.4s, v0.s[2]      \n"
                    "fmla   v31.4s, v5.4s, v0.s[3]      \n"

                    "fmla   v8.4s, v6.4s, v1.s[0]       \n"
                    "fmla   v9.4s, v6.4s, v1.s[1]       \n"
                    "fmla   v10.4s, v6.4s, v1.s[2]      \n"
                    "fmla   v11.4s, v6.4s, v1.s[3]      \n"
                    "fmla   v20.4s, v7.4s, v1.s[0]      \n"
                    "fmla   v21.4s, v7.4s, v1.s[1]      \n"
                    "fmla   v22.4s, v7.4s, v1.s[2]      \n"
                    "fmla   v23.4s, v7.4s, v1.s[3]      \n"

                    "fmla   v12.4s, v6.4s, v2.s[0]      \n"
                    "fmla   v13.4s, v6.4s, v2.s[1]      \n"
                    "fmla   v14.4s, v6.4s, v2.s[2]      \n"
                    "fmla   v15.4s, v6.4s, v2.s[3]      \n"
                    "fmla   v24.4s, v7.4s, v2.s[0]      \n"
                    "fmla   v25.4s, v7.4s, v2.s[1]      \n"
                    "fmla   v26.4s, v7.4s, v2.s[2]      \n"
                    "fmla   v27.4s, v7.4s, v2.s[3]      \n"

                    "subs   w4, w4, #1                  \n"

                    "fmla   v16.4s, v6.4s, v3.s[0]      \n"
                    "fmla   v17.4s, v6.4s, v3.s[1]      \n"
                    "fmla   v18.4s, v6.4s, v3.s[2]      \n"
                    "fmla   v19.4s, v6.4s, v3.s[3]      \n"
                    "fmla   v28.4s, v7.4s, v3.s[0]      \n"
                    "fmla   v29.4s, v7.4s, v3.s[1]      \n"
                    "fmla   v30.4s, v7.4s, v3.s[2]      \n"
                    "fmla   v31.4s, v7.4s, v3.s[3]      \n"

                    "bne    4b                          \n"

                    "5:                                 \n"
                    "and    w4, %w9, #3                 \n" // w4 = remain = max_kk & 3
                    "cmp    w4, #0                      \n"
                    "beq    7f                          \n"

                    "6:                                 \n"
                    "ld1    {v0.4h, v1.4h, v2.4h}, [%2], #24 \n"

                    "shll   v0.4s, v0.4h, #16           \n"
                    "shll   v1.4s, v1.4h, #16           \n"
                    "shll   v2.4s, v2.4h, #16           \n"

                    "ld1    {v4.4h, v5.4h}, [%1], #16   \n"

                    "shll   v4.4s, v4.4h, #16           \n"
                    "shll   v5.4s, v5.4h, #16           \n"

                    "fmla   v8.4s, v4.4s, v0.s[0]       \n"
                    "fmla   v9.4s, v4.4s, v0.s[1]       \n"
                    "fmla   v10.4s, v4.4s, v0.s[2]      \n"
                    "fmla   v11.4s, v4.4s, v0.s[3]      \n"
                    "fmla   v12.4s, v4.4s, v1.s[0]      \n"
                    "fmla   v13.4s, v4.4s, v1.s[1]      \n"
                    "fmla   v14.4s, v4.4s, v1.s[2]      \n"
                    "fmla   v15.4s, v4.4s, v1.s[3]      \n"
                    "fmla   v16.4s, v4.4s, v2.s[0]      \n"
                    "fmla   v17.4s, v4.4s, v2.s[1]      \n"
                    "fmla   v18.4s, v4.4s, v2.s[2]      \n"
                    "fmla   v19.4s, v4.4s, v2.s[3]      \n"

                    "subs   w4, w4, #1                  \n"

                    "fmla   v20.4s, v5.4s, v0.s[0]      \n"
                    "fmla   v21.4s, v5.4s, v0.s[1]      \n"
                    "fmla   v22.4s, v5.4s, v0.s[2]      \n"
                    "fmla   v23.4s, v5.4s, v0.s[3]      \n"
                    "fmla   v24.4s, v5.4s, v1.s[0]      \n"
                    "fmla   v25.4s, v5.4s, v1.s[1]      \n"
                    "fmla   v26.4s, v5.4s, v1.s[2]      \n"
                    "fmla   v27.4s, v5.4s, v1.s[3]      \n"
                    "fmla   v28.4s, v5.4s, v2.s[0]      \n"
                    "fmla   v29.4s, v5.4s, v2.s[1]      \n"
                    "fmla   v30.4s, v5.4s, v2.s[2]      \n"
                    "fmla   v31.4s, v5.4s, v2.s[3]      \n"

                    "bne    6b                          \n"

                    "7:                                 \n"
                    "shrn   v0.4h, v8.4s, #16           \n"
                    "shrn2  v0.8h, v9.4s, #16           \n"
                    "shrn   v1.4h, v10.4s, #16          \n"
                    "shrn2  v1.8h, v11.4s, #16          \n"
                    "shrn   v2.4h, v12.4s, #16          \n"
                    "shrn2  v2.8h, v13.4s, #16          \n"
                    "shrn   v3.4h, v14.4s, #16          \n"
                    "shrn2  v3.8h, v15.4s, #16          \n"
                    "shrn   v4.4h, v16.4s, #16          \n"
                    "shrn2  v4.8h, v17.4s, #16          \n"
                    "shrn   v5.4h, v18.4s, #16          \n"
                    "shrn2  v5.8h, v19.4s, #16          \n"
                    "shrn   v6.4h, v20.4s, #16          \n"
                    "shrn2  v6.8h, v21.4s, #16          \n"
                    "shrn   v7.4h, v22.4s, #16          \n"
                    "shrn2  v7.8h, v23.4s, #16          \n"
                    "shrn   v8.4h, v24.4s, #16          \n"
                    "shrn2  v8.8h, v25.4s, #16          \n"
                    "shrn   v9.4h, v26.4s, #16          \n"
                    "shrn2  v9.8h, v27.4s, #16          \n"
                    "shrn   v10.4h, v28.4s, #16         \n"
                    "shrn2  v10.8h, v29.4s, #16         \n"
                    "shrn   v11.4h, v30.4s, #16         \n"
                    "shrn2  v11.8h, v31.4s, #16         \n"
                    "tst    %w11, #255                  \n"
                    "beq    10f                         \n"

                    // if out_elempack == 4
                    "cmp    %w12, #4                    \n"
                    "bne    8f                          \n"

                    "lsl    w4, %w13, #2                \n"
                    "add    x4, %3, w4, sxtw 1          \n"
                    "st1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%3], #64 \n"
                    "st1    {v4.8h, v5.8h}, [%3], #32 \n"
                    "st1    {v6.8h, v7.8h, v8.8h, v9.8h}, [x4], #64 \n"
                    "st1    {v10.8h, v11.8h}, [x4]      \n"
                    "b      9f                          \n"

                    // if out_elempack == 1
                    "8:                                 \n"
                    // transpose8x12
                    "uzp1   v20.8h, v0.8h, v1.8h        \n"
                    "uzp2   v21.8h, v0.8h, v1.8h        \n"
                    "uzp1   v22.8h, v2.8h, v3.8h        \n"
                    "uzp2   v23.8h, v2.8h, v3.8h        \n"
                    "uzp1   v24.8h, v4.8h, v5.8h        \n"
                    "uzp2   v25.8h, v4.8h, v5.8h        \n"
                    "uzp1   v26.8h, v6.8h, v7.8h        \n"
                    "uzp2   v27.8h, v6.8h, v7.8h        \n"
                    "uzp1   v28.8h, v8.8h, v9.8h        \n"
                    "uzp2   v29.8h, v8.8h, v9.8h        \n"
                    "uzp1   v30.8h, v10.8h, v11.8h      \n"
                    "uzp2   v31.8h, v10.8h, v11.8h      \n"

                    "uzp1   v0.8h, v20.8h, v22.8h       \n"
                    "uzp2   v6.8h, v20.8h, v22.8h       \n"
                    "uzp1   v3.8h, v21.8h, v23.8h       \n"
                    "uzp2   v9.8h, v21.8h, v23.8h       \n"
                    "mov    v1.d[0], v0.d[1]            \n"
                    "mov    v7.d[0], v6.d[1]            \n"
                    "mov    v4.d[0], v3.d[1]            \n"
                    "mov    v10.d[0], v9.d[1]           \n"
                    "uzp1   v2.8h, v24.8h, v24.8h       \n"
                    "uzp2   v8.8h, v24.8h, v24.8h       \n"
                    "uzp1   v5.8h, v25.8h, v25.8h       \n"
                    "uzp2   v11.8h, v25.8h, v25.8h      \n"

                    "uzp1   v12.8h, v26.8h, v28.8h      \n"
                    "uzp2   v18.8h, v26.8h, v28.8h      \n"
                    "uzp1   v15.8h, v27.8h, v29.8h      \n"
                    "uzp2   v21.8h, v27.8h, v29.8h      \n"
                    "mov    v13.d[0], v12.d[1]          \n"
                    "mov    v19.d[0], v18.d[1]          \n"
                    "mov    v16.d[0], v15.d[1]          \n"
                    "mov    v22.d[0], v21.d[1]          \n"
                    "uzp1   v14.8h, v30.8h, v30.8h      \n"
                    "uzp2   v20.8h, v30.8h, v30.8h      \n"
                    "uzp1   v17.8h, v31.8h, v31.8h      \n"
                    "uzp2   v23.8h, v31.8h, v31.8h      \n"

                    "add    x4, %3, %w13, sxtw 1        \n"
                    "st1    {v0.4h, v1.4h, v2.4h}, [%3], #24 \n"
                    "st1    {v3.4h, v4.4h, v5.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v6.4h, v7.4h, v8.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v9.4h, v10.4h, v11.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v12.4h, v13.4h, v14.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v15.4h, v16.4h, v17.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v18.4h, v19.4h, v20.4h}, [x4] \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v21.4h, v22.4h, v23.4h}, [x4] \n"

                    "9:                                 \n"
                    "add    %0, %0, #384                \n"
                    "b      11f                         \n"

                    "10:                                \n"
                    "st1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%0], #64   \n"
                    "st1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%0], #64 \n"
                    "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                    "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                    "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                    "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0], #64 \n"

                    "11:                                \n"

                    : "=r"(outptr), // %0
                    "=r"(pA),     // %1
                    "=r"(pB),     // %2
                    "=r"(outptr0) // %3
                    : "0"(outptr),
                    "1"(pA),
                    "2"(pB),
                    "3"(outptr0),
                    "r"(pC),           // %8
                    "r"(max_kk),       // %9
                    "r"(k),            // %10
                    "r"(k_end),        // %11
                    "r"(out_elempack), // %12
                    "r"(out_hstep)     // %13
                    : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
            }
#else  // NCNN_GNU_INLINE_ASM
            float32x4_t _sum00;
            float32x4_t _sum01;
            float32x4_t _sum10;
            float32x4_t _sum11;
            float32x4_t _sum20;
            float32x4_t _sum21;
            float32x4_t _sum30;
            float32x4_t _sum31;
            float32x4_t _sum40;
            float32x4_t _sum41;
            float32x4_t _sum50;
            float32x4_t _sum51;
            float32x4_t _sum60;
            float32x4_t _sum61;
            float32x4_t _sum70;
            float32x4_t _sum71;
            float32x4_t _sum80;
            float32x4_t _sum81;
            float32x4_t _sum90;
            float32x4_t _sum91;
            float32x4_t _suma0;
            float32x4_t _suma1;
            float32x4_t _sumb0;
            float32x4_t _sumb1;

            if (k == 0)
            {
                if (pC)
                {
                    _sum00 = vld1q_f32(pC);
                    _sum01 = vld1q_f32(pC + 4);
                    _sum10 = _sum00;
                    _sum11 = _sum01;
                    _sum20 = _sum00;
                    _sum21 = _sum01;
                    _sum30 = _sum00;
                    _sum31 = _sum01;
                    _sum40 = _sum00;
                    _sum41 = _sum01;
                    _sum50 = _sum00;
                    _sum51 = _sum01;
                    _sum60 = _sum00;
                    _sum61 = _sum01;
                    _sum70 = _sum00;
                    _sum71 = _sum01;
                    _sum80 = _sum00;
                    _sum81 = _sum01;
                    _sum90 = _sum00;
                    _sum91 = _sum01;
                    _suma0 = _sum00;
                    _suma1 = _sum01;
                    _sumb0 = _sum00;
                    _sumb1 = _sum01;
                }
                else
                {
                    _sum00 = vdupq_n_f32(0.f);
                    _sum01 = vdupq_n_f32(0.f);
                    _sum10 = vdupq_n_f32(0.f);
                    _sum11 = vdupq_n_f32(0.f);
                    _sum20 = vdupq_n_f32(0.f);
                    _sum21 = vdupq_n_f32(0.f);
                    _sum30 = vdupq_n_f32(0.f);
                    _sum31 = vdupq_n_f32(0.f);
                    _sum40 = vdupq_n_f32(0.f);
                    _sum41 = vdupq_n_f32(0.f);
                    _sum50 = vdupq_n_f32(0.f);
                    _sum51 = vdupq_n_f32(0.f);
                    _sum60 = vdupq_n_f32(0.f);
                    _sum61 = vdupq_n_f32(0.f);
                    _sum70 = vdupq_n_f32(0.f);
                    _sum71 = vdupq_n_f32(0.f);
                    _sum80 = vdupq_n_f32(0.f);
                    _sum81 = vdupq_n_f32(0.f);
                    _sum90 = vdupq_n_f32(0.f);
                    _sum91 = vdupq_n_f32(0.f);
                    _suma0 = vdupq_n_f32(0.f);
                    _suma1 = vdupq_n_f32(0.f);
                    _sumb0 = vdupq_n_f32(0.f);
                    _sumb1 = vdupq_n_f32(0.f);
                }
            }
            else
            {
                _sum00 = vld1q_f32(outptr);
                _sum01 = vld1q_f32(outptr + 4 * 1);
                _sum10 = vld1q_f32(outptr + 4 * 2);
                _sum11 = vld1q_f32(outptr + 4 * 3);
                _sum20 = vld1q_f32(outptr + 4 * 4);
                _sum21 = vld1q_f32(outptr + 4 * 5);
                _sum30 = vld1q_f32(outptr + 4 * 6);
                _sum31 = vld1q_f32(outptr + 4 * 7);
                _sum40 = vld1q_f32(outptr + 4 * 8);
                _sum41 = vld1q_f32(outptr + 4 * 9);
                _sum50 = vld1q_f32(outptr + 4 * 10);
                _sum51 = vld1q_f32(outptr + 4 * 11);
                _sum60 = vld1q_f32(outptr + 4 * 12);
                _sum61 = vld1q_f32(outptr + 4 * 13);
                _sum70 = vld1q_f32(outptr + 4 * 14);
                _sum71 = vld1q_f32(outptr + 4 * 15);
                _sum80 = vld1q_f32(outptr + 4 * 16);
                _sum81 = vld1q_f32(outptr + 4 * 17);
                _sum90 = vld1q_f32(outptr + 4 * 18);
                _sum91 = vld1q_f32(outptr + 4 * 19);
                _suma0 = vld1q_f32(outptr + 4 * 20);
                _suma1 = vld1q_f32(outptr + 4 * 21);
                _sumb0 = vld1q_f32(outptr + 4 * 22);
                _sumb1 = vld1q_f32(outptr + 4 * 23);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA0 = bfloat2float(vld1_u16(pA));
                float32x4_t _pA1 = bfloat2float(vld1_u16(pA + 4));

                float32x4_t _pB0 = bfloat2float(vld1_u16(pB));
                float32x4_t _pB1 = bfloat2float(vld1_u16(pB + 4));
                float32x4_t _pB2 = bfloat2float(vld1_u16(pB + 8));

                _sum00 = vfmaq_laneq_f32(_sum00, _pA0, _pB0, 0);
                _sum01 = vfmaq_laneq_f32(_sum01, _pA1, _pB0, 0);
                _sum10 = vfmaq_laneq_f32(_sum10, _pA0, _pB0, 1);
                _sum11 = vfmaq_laneq_f32(_sum11, _pA1, _pB0, 1);
                _sum20 = vfmaq_laneq_f32(_sum20, _pA0, _pB0, 2);
                _sum21 = vfmaq_laneq_f32(_sum21, _pA1, _pB0, 2);
                _sum30 = vfmaq_laneq_f32(_sum30, _pA0, _pB0, 3);
                _sum31 = vfmaq_laneq_f32(_sum31, _pA1, _pB0, 3);
                _sum40 = vfmaq_laneq_f32(_sum40, _pA0, _pB1, 0);
                _sum41 = vfmaq_laneq_f32(_sum41, _pA1, _pB1, 0);
                _sum50 = vfmaq_laneq_f32(_sum50, _pA0, _pB1, 1);
                _sum51 = vfmaq_laneq_f32(_sum51, _pA1, _pB1, 1);
                _sum60 = vfmaq_laneq_f32(_sum60, _pA0, _pB1, 2);
                _sum61 = vfmaq_laneq_f32(_sum61, _pA1, _pB1, 2);
                _sum70 = vfmaq_laneq_f32(_sum70, _pA0, _pB1, 3);
                _sum71 = vfmaq_laneq_f32(_sum71, _pA1, _pB1, 3);
                _sum80 = vfmaq_laneq_f32(_sum80, _pA0, _pB2, 0);
                _sum81 = vfmaq_laneq_f32(_sum81, _pA1, _pB2, 0);
                _sum90 = vfmaq_laneq_f32(_sum90, _pA0, _pB2, 1);
                _sum91 = vfmaq_laneq_f32(_sum91, _pA1, _pB2, 1);
                _suma0 = vfmaq_laneq_f32(_suma0, _pA0, _pB2, 2);
                _suma1 = vfmaq_laneq_f32(_suma1, _pA1, _pB2, 2);
                _sumb0 = vfmaq_laneq_f32(_sumb0, _pA0, _pB2, 3);
                _sumb1 = vfmaq_laneq_f32(_sumb1, _pA1, _pB2, 3);

                pA += 8;
                pB += 12;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1_u16(outptr0, float2bfloat(_sum00));
                    vst1_u16(outptr0 + 4, float2bfloat(_sum10));
                    vst1_u16(outptr0 + 4 * 2, float2bfloat(_sum20));
                    vst1_u16(outptr0 + 4 * 3, float2bfloat(_sum30));
                    vst1_u16(outptr0 + 4 * 4, float2bfloat(_sum40));
                    vst1_u16(outptr0 + 4 * 5, float2bfloat(_sum50));
                    vst1_u16(outptr0 + 4 * 6, float2bfloat(_sum60));
                    vst1_u16(outptr0 + 4 * 7, float2bfloat(_sum70));
                    vst1_u16(outptr0 + 4 * 8, float2bfloat(_sum80));
                    vst1_u16(outptr0 + 4 * 9, float2bfloat(_sum90));
                    vst1_u16(outptr0 + 4 * 10, float2bfloat(_suma0));
                    vst1_u16(outptr0 + 4 * 11, float2bfloat(_sumb0));

                    vst1_u16(outptr0 + out_hstep * 4, float2bfloat(_sum01));
                    vst1_u16(outptr0 + out_hstep * 4 + 4, float2bfloat(_sum11));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 2, float2bfloat(_sum21));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 3, float2bfloat(_sum31));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 4, float2bfloat(_sum41));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 5, float2bfloat(_sum51));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 6, float2bfloat(_sum61));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 7, float2bfloat(_sum71));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 8, float2bfloat(_sum81));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 9, float2bfloat(_sum91));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 10, float2bfloat(_suma1));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 11, float2bfloat(_sumb1));

                    outptr0 += 48;
                }
                if (out_elempack == 1)
                {
                    uint16x8_t _t0 = vcombine_u16(float2bfloat(_sum00), float2bfloat(_sum01));
                    uint16x8_t _t1 = vcombine_u16(float2bfloat(_sum10), float2bfloat(_sum11));
                    uint16x8_t _t2 = vcombine_u16(float2bfloat(_sum20), float2bfloat(_sum21));
                    uint16x8_t _t3 = vcombine_u16(float2bfloat(_sum30), float2bfloat(_sum31));
                    uint16x8_t _t4 = vcombine_u16(float2bfloat(_sum40), float2bfloat(_sum41));
                    uint16x8_t _t5 = vcombine_u16(float2bfloat(_sum50), float2bfloat(_sum51));
                    uint16x8_t _t6 = vcombine_u16(float2bfloat(_sum60), float2bfloat(_sum61));
                    uint16x8_t _t7 = vcombine_u16(float2bfloat(_sum70), float2bfloat(_sum71));
                    uint16x8_t _t8 = vcombine_u16(float2bfloat(_sum80), float2bfloat(_sum81));
                    uint16x8_t _t9 = vcombine_u16(float2bfloat(_sum90), float2bfloat(_sum91));
                    uint16x8_t _ta = vcombine_u16(float2bfloat(_suma0), float2bfloat(_suma1));
                    uint16x8_t _tb = vcombine_u16(float2bfloat(_sumb0), float2bfloat(_sumb1));
                    transpose8x12_u16(_t0, _t1, _t2, _t3, _t4, _t5, _t6, _t7, _t8, _t9, _ta, _tb);

                    vst1_u16(outptr0, vget_low_u16(_t0));
                    vst1_u16(outptr0 + 4, vget_high_u16(_t0));
                    vst1_u16(outptr0 + 8, vget_low_u16(_t1));
                    vst1_u16(outptr0 + out_hstep, vget_high_u16(_t1));
                    vst1_u16(outptr0 + out_hstep + 4, vget_low_u16(_t2));
                    vst1_u16(outptr0 + out_hstep + 8, vget_high_u16(_t2));
                    vst1_u16(outptr0 + out_hstep * 2, vget_low_u16(_t3));
                    vst1_u16(outptr0 + out_hstep * 2 + 4, vget_high_u16(_t3));
                    vst1_u16(outptr0 + out_hstep * 2 + 8, vget_low_u16(_t4));
                    vst1_u16(outptr0 + out_hstep * 3, vget_high_u16(_t4));
                    vst1_u16(outptr0 + out_hstep * 3 + 4, vget_low_u16(_t5));
                    vst1_u16(outptr0 + out_hstep * 3 + 8, vget_high_u16(_t5));
                    vst1_u16(outptr0 + out_hstep * 4, vget_low_u16(_t6));
                    vst1_u16(outptr0 + out_hstep * 4 + 4, vget_high_u16(_t6));
                    vst1_u16(outptr0 + out_hstep * 4 + 8, vget_low_u16(_t7));
                    vst1_u16(outptr0 + out_hstep * 5, vget_high_u16(_t7));
                    vst1_u16(outptr0 + out_hstep * 5 + 4, vget_low_u16(_t8));
                    vst1_u16(outptr0 + out_hstep * 5 + 8, vget_high_u16(_t8));
                    vst1_u16(outptr0 + out_hstep * 6, vget_low_u16(_t9));
                    vst1_u16(outptr0 + out_hstep * 6 + 4, vget_high_u16(_t9));
                    vst1_u16(outptr0 + out_hstep * 6 + 8, vget_low_u16(_ta));
                    vst1_u16(outptr0 + out_hstep * 7, vget_high_u16(_ta));
                    vst1_u16(outptr0 + out_hstep * 7 + 4, vget_low_u16(_tb));
                    vst1_u16(outptr0 + out_hstep * 7 + 8, vget_high_u16(_tb));

                    outptr0 += 12;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum00);
                vst1q_f32(outptr + 4, _sum01);
                vst1q_f32(outptr + 4 * 2, _sum10);
                vst1q_f32(outptr + 4 * 3, _sum11);
                vst1q_f32(outptr + 4 * 4, _sum20);
                vst1q_f32(outptr + 4 * 5, _sum21);
                vst1q_f32(outptr + 4 * 6, _sum30);
                vst1q_f32(outptr + 4 * 7, _sum31);
                vst1q_f32(outptr + 4 * 8, _sum40);
                vst1q_f32(outptr + 4 * 9, _sum41);
                vst1q_f32(outptr + 4 * 10, _sum50);
                vst1q_f32(outptr + 4 * 11, _sum51);
                vst1q_f32(outptr + 4 * 12, _sum60);
                vst1q_f32(outptr + 4 * 13, _sum61);
                vst1q_f32(outptr + 4 * 14, _sum70);
                vst1q_f32(outptr + 4 * 15, _sum71);
                vst1q_f32(outptr + 4 * 16, _sum80);
                vst1q_f32(outptr + 4 * 17, _sum81);
                vst1q_f32(outptr + 4 * 18, _sum90);
                vst1q_f32(outptr + 4 * 19, _sum91);
                vst1q_f32(outptr + 4 * 20, _suma0);
                vst1q_f32(outptr + 4 * 21, _suma1);
                vst1q_f32(outptr + 4 * 22, _sumb0);
                vst1q_f32(outptr + 4 * 23, _sumb1);
            }

            outptr += 96;
#endif // NCNN_GNU_INLINE_ASM
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            const unsigned short* pA = pAT;

#if NCNN_GNU_INLINE_ASM
            if (use_a53_a55_optimized_kernel && cpu_support_arm_asimdhp())
            {
                // a55
                asm volatile(
                    "cbz    %w10, 0f                    \n"

                    "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                    "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                    "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                    "ld1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0]      \n"
                    "subs   %0, %0, #192                \n"
                    "b      3f                          \n"

                    "0:                                 \n"
                    // if pC
                    "cbz    %8, 1f                      \n"

                    "add    x4, %8, #16                 \n"
                    "ld1    {v16.4s}, [%8]              \n"
                    "ld1    {v24.4s}, [x4]              \n"
                    "b      2f                          \n"

                    // else
                    "1:                                 \n"
                    "eor    v16.16b, v16.16b, v16.16b   \n"
                    "eor    v24.16b, v24.16b, v24.16b   \n"

                    "2:                                 \n"
                    "mov    v17.16b, v16.16b            \n"
                    "mov    v18.16b, v16.16b            \n"
                    "mov    v19.16b, v16.16b            \n"
                    "mov    v20.16b, v16.16b            \n"
                    "mov    v21.16b, v16.16b            \n"
                    "mov    v22.16b, v16.16b            \n"
                    "mov    v23.16b, v16.16b            \n"

                    "mov    v25.16b, v24.16b            \n"
                    "mov    v26.16b, v24.16b            \n"
                    "mov    v27.16b, v24.16b            \n"
                    "mov    v28.16b, v24.16b            \n"
                    "mov    v29.16b, v24.16b            \n"
                    "mov    v30.16b, v24.16b            \n"
                    "mov    v31.16b, v24.16b            \n"

                    "3:                                 \n"
                    "lsr    w4, %w9, #2                 \n" // w4 = max_kk >> 2
                    "cmp    w4, #0                      \n"
                    "beq    5f                          \n"

                    "prfm   pldl1keep, [%1, #512]       \n"
                    "ld1    {v8.4h, v9.4h}, [%1], #16   \n"

                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v0.4h, v1.4h, v2.4h}, [%2], #24 \n"

                    "shll   v0.4s, v0.4h, #16           \n"
                    "shll   v8.4s, v8.4h, #16           \n"

                    ".align 4                           \n"
                    "4:                                 \n"
                    "shll   v1.4s, v1.4h, #16           \n"
                    "fmla   v16.4s, v8.4s, v0.s[0]      \n"
                    "ldr    d10, [%1], #8               \n"
                    "fmla   v17.4s, v8.4s, v0.s[1]      \n"
                    "ldr    d3, [%2], #8                \n"
                    "fmla   v18.4s, v8.4s, v0.s[2]      \n"
                    "ldr    d11, [%1], #8               \n"
                    "fmla   v19.4s, v8.4s, v0.s[3]      \n"
                    "shll   v9.4s, v9.4h, #16           \n"
                    "fmla   v20.4s, v8.4s, v1.s[0]      \n"
                    "ldr    d4, [%2], #8                \n"
                    "fmla   v21.4s, v8.4s, v1.s[1]      \n"
                    "ldr    d12, [%1], #8               \n"
                    "fmla   v22.4s, v8.4s, v1.s[2]      \n"
                    "ldr    d5, [%2], #8                \n"
                    "fmla   v23.4s, v8.4s, v1.s[3]      \n"
                    "shll   v2.4s, v2.4h, #16           \n"
                    "fmla   v24.4s, v9.4s, v0.s[0]      \n"
                    "ldr    d13, [%1], #8               \n"
                    "fmla   v25.4s, v9.4s, v0.s[1]      \n"
                    "ldr    d6, [%2], #8                \n"
                    "fmla   v26.4s, v9.4s, v0.s[2]      \n"
                    "ldr    d14, [%1], #8               \n"
                    "fmla   v27.4s, v9.4s, v0.s[3]      \n"
                    "shll   v10.4s, v10.4h, #16         \n"
                    "fmla   v28.4s, v9.4s, v1.s[0]      \n"
                    "ldr    d7, [%2], #8                \n"
                    "fmla   v29.4s, v9.4s, v1.s[1]      \n"
                    "ldr    d15, [%1], #8               \n"
                    "fmla   v30.4s, v9.4s, v1.s[2]      \n"
                    "prfm   pldl1keep, [%1, #512]       \n" // NOTE PRELOAD
                    "fmla   v31.4s, v9.4s, v1.s[3]      \n"
                    "shll   v3.4s, v3.4h, #16           \n"
                    "fmla   v16.4s, v10.4s, v2.s[0]     \n"
                    "ldr    d8, [%1], #8                \n"
                    "fmla   v17.4s, v10.4s, v2.s[1]     \n"
                    "prfm   pldl1keep, [%2, #512]       \n" // NOTE PRELOAD
                    "fmla   v18.4s, v10.4s, v2.s[2]     \n"
                    "ldr    d0, [%2], #8                \n"
                    "fmla   v19.4s, v10.4s, v2.s[3]     \n"
                    "shll   v11.4s, v11.4h, #16         \n"
                    "fmla   v20.4s, v10.4s, v3.s[0]     \n"
                    "ldr    d1, [%2], #8                \n"
                    "fmla   v21.4s, v10.4s, v3.s[1]     \n"
                    "ldr    d9, [%1], #8                \n"
                    "fmla   v22.4s, v10.4s, v3.s[2]     \n"
                    "fmla   v23.4s, v10.4s, v3.s[3]     \n"
                    "shll   v4.4s, v4.4h, #16           \n"
                    "fmla   v24.4s, v11.4s, v2.s[0]     \n"
                    "fmla   v25.4s, v11.4s, v2.s[1]     \n"
                    "fmla   v26.4s, v11.4s, v2.s[2]     \n"
                    "fmla   v27.4s, v11.4s, v2.s[3]     \n"
                    "shll   v12.4s, v12.4h, #16         \n"
                    "fmla   v28.4s, v11.4s, v3.s[0]     \n"
                    "ldr    d2, [%2], #8                \n"
                    "fmla   v29.4s, v11.4s, v3.s[1]     \n"
                    "fmla   v30.4s, v11.4s, v3.s[2]     \n"
                    "fmla   v31.4s, v11.4s, v3.s[3]     \n"
                    "shll   v5.4s, v5.4h, #16           \n"
                    "fmla   v16.4s, v12.4s, v4.s[0]     \n"
                    "fmla   v17.4s, v12.4s, v4.s[1]     \n"
                    "fmla   v18.4s, v12.4s, v4.s[2]     \n"
                    "fmla   v19.4s, v12.4s, v4.s[3]     \n"
                    "shll   v13.4s, v13.4h, #16         \n"
                    "fmla   v20.4s, v12.4s, v5.s[0]     \n"
                    "fmla   v21.4s, v12.4s, v5.s[1]     \n"
                    "fmla   v22.4s, v12.4s, v5.s[2]     \n"
                    "fmla   v23.4s, v12.4s, v5.s[3]     \n"
                    "shll   v6.4s, v6.4h, #16           \n"
                    "fmla   v24.4s, v13.4s, v4.s[0]     \n"
                    "fmla   v25.4s, v13.4s, v4.s[1]     \n"
                    "fmla   v26.4s, v13.4s, v4.s[2]     \n"
                    "fmla   v27.4s, v13.4s, v4.s[3]     \n"
                    "shll   v14.4s, v14.4h, #16         \n"
                    "fmla   v28.4s, v13.4s, v5.s[0]     \n"
                    "fmla   v29.4s, v13.4s, v5.s[1]     \n"
                    "fmla   v30.4s, v13.4s, v5.s[2]     \n"
                    "fmla   v31.4s, v13.4s, v5.s[3]     \n"
                    "shll   v7.4s, v7.4h, #16           \n"
                    "fmla   v16.4s, v14.4s, v6.s[0]     \n"
                    "fmla   v17.4s, v14.4s, v6.s[1]     \n"
                    "fmla   v18.4s, v14.4s, v6.s[2]     \n"
                    "fmla   v19.4s, v14.4s, v6.s[3]     \n"
                    "shll   v15.4s, v15.4h, #16         \n"
                    "fmla   v20.4s, v14.4s, v7.s[0]     \n"
                    "fmla   v21.4s, v14.4s, v7.s[1]     \n"
                    "fmla   v22.4s, v14.4s, v7.s[2]     \n"
                    "fmla   v23.4s, v14.4s, v7.s[3]     \n"
                    "shll   v8.4s, v8.4h, #16           \n"
                    "fmla   v24.4s, v15.4s, v6.s[0]     \n"
                    "fmla   v25.4s, v15.4s, v6.s[1]     \n"
                    "fmla   v26.4s, v15.4s, v6.s[2]     \n"
                    "fmla   v27.4s, v15.4s, v6.s[3]     \n"
                    "shll   v0.4s, v0.4h, #16           \n"
                    "fmla   v28.4s, v15.4s, v7.s[0]     \n"
                    "fmla   v29.4s, v15.4s, v7.s[1]     \n"
                    "subs   w4, w4, #1                  \n"
                    "fmla   v30.4s, v15.4s, v7.s[2]     \n"
                    "fmla   v31.4s, v15.4s, v7.s[3]     \n"
                    "bne    4b                          \n"

                    "sub    %1, %1, #16                 \n"
                    "sub    %2, %2, #24                 \n"

                    "5:                                 \n"
                    "and    w4, %w9, #3                 \n" // w4 = remain = max_kk & 3
                    "cmp    w4, #0                      \n"
                    "beq    7f                          \n"

                    "6:                                 \n"
                    "ld1    {v2.8h}, [%2], #16          \n"
                    "shll   v0.4s, v2.4h, #16           \n"
                    "shll2  v1.4s, v2.8h, #16           \n"
                    "ld1    {v3.8h}, [%1], #16          \n"
                    "shll   v4.4s, v3.4h, #16           \n"
                    "shll2  v5.4s, v3.8h, #16           \n"

                    "fmla   v16.4s, v4.4s, v0.s[0]      \n"
                    "fmla   v17.4s, v4.4s, v0.s[1]      \n"
                    "fmla   v18.4s, v4.4s, v0.s[2]      \n"
                    "fmla   v19.4s, v4.4s, v0.s[3]      \n"
                    "fmla   v20.4s, v4.4s, v1.s[0]      \n"
                    "fmla   v21.4s, v4.4s, v1.s[1]      \n"
                    "fmla   v22.4s, v4.4s, v1.s[2]      \n"
                    "fmla   v23.4s, v4.4s, v1.s[3]      \n"

                    "subs   w4, w4, #1                  \n"

                    "fmla   v24.4s, v5.4s, v0.s[0]      \n"
                    "fmla   v25.4s, v5.4s, v0.s[1]      \n"
                    "fmla   v26.4s, v5.4s, v0.s[2]      \n"
                    "fmla   v27.4s, v5.4s, v0.s[3]      \n"
                    "fmla   v28.4s, v5.4s, v1.s[0]      \n"
                    "fmla   v29.4s, v5.4s, v1.s[1]      \n"
                    "fmla   v30.4s, v5.4s, v1.s[2]      \n"
                    "fmla   v31.4s, v5.4s, v1.s[3]      \n"

                    "bne    6b                          \n"

                    "7:                                 \n"
                    "shrn   v0.4h, v16.4s, #16          \n"
                    "shrn2  v0.8h, v17.4s, #16          \n"
                    "shrn   v1.4h, v18.4s, #16          \n"
                    "shrn2  v1.8h, v19.4s, #16          \n"
                    "shrn   v2.4h, v20.4s, #16          \n"
                    "shrn2  v2.8h, v21.4s, #16          \n"
                    "shrn   v3.4h, v22.4s, #16          \n"
                    "shrn2  v3.8h, v23.4s, #16          \n"
                    "shrn   v4.4h, v24.4s, #16          \n"
                    "shrn2  v4.8h, v25.4s, #16          \n"
                    "shrn   v5.4h, v26.4s, #16          \n"
                    "shrn2  v5.8h, v27.4s, #16          \n"
                    "shrn   v6.4h, v28.4s, #16          \n"
                    "shrn2  v6.8h, v29.4s, #16          \n"
                    "shrn   v7.4h, v30.4s, #16          \n"
                    "shrn2  v7.8h, v31.4s, #16          \n"
                    "tst    %w11, #255                  \n"
                    "beq    10f                         \n"

                    // if out_elempack == 4
                    "cmp    %w12, #4                    \n"
                    "bne    8f                          \n"

                    "lsl    w4, %w13, #2                \n"
                    "add    x4, %3, w4, sxtw 1          \n"
                    "st1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%3], #64 \n"
                    "st1    {v4.8h, v5.8h, v6.8h, v7.8h}, [x4] \n"
                    "b      9f                          \n"

                    // if out_elempack == 1
                    "8:                                 \n"
                    // transpose8x8
                    "uzp1   v24.8h, v0.8h, v1.8h        \n"
                    "uzp2   v25.8h, v0.8h, v1.8h        \n"
                    "uzp1   v26.8h, v2.8h, v3.8h        \n"
                    "uzp2   v27.8h, v2.8h, v3.8h        \n"
                    "uzp1   v28.8h, v4.8h, v5.8h        \n"
                    "uzp2   v29.8h, v4.8h, v5.8h        \n"
                    "uzp1   v30.8h, v6.8h, v7.8h        \n"
                    "uzp2   v31.8h, v6.8h, v7.8h        \n"

                    "uzp1   v0.8h, v24.8h, v26.8h       \n"
                    "uzp2   v2.8h, v24.8h, v26.8h       \n"
                    "uzp1   v1.8h, v25.8h, v27.8h       \n"
                    "uzp2   v3.8h, v25.8h, v27.8h       \n"

                    "uzp1   v4.8h, v28.8h, v30.8h       \n"
                    "uzp2   v6.8h, v28.8h, v30.8h       \n"
                    "uzp1   v5.8h, v29.8h, v31.8h       \n"
                    "uzp2   v7.8h, v29.8h, v31.8h       \n"

                    "add    x4, %3, %w13, sxtw 1        \n"
                    "st1    {v0.8h}, [%3], #16          \n"
                    "st1    {v1.8h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v2.8h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v3.8h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v4.8h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v5.8h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v6.8h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v7.8h}, [x4]               \n"

                    "9:                                 \n"
                    "add    %0, %0, #256                \n"
                    "b      11f                         \n"

                    "10:                                \n"
                    "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                    "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                    "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                    "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0], #64 \n"

                    "11:                                \n"

                    : "=r"(outptr), // %0
                    "=r"(pA),     // %1
                    "=r"(pB),     // %2
                    "=r"(outptr0) // %3
                    : "0"(outptr),
                    "1"(pA),
                    "2"(pB),
                    "3"(outptr0),
                    "r"(pC),           // %8
                    "r"(max_kk),       // %9
                    "r"(k),            // %10
                    "r"(k_end),        // %11
                    "r"(out_elempack), // %12
                    "r"(out_hstep)     // %13
                    : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
            }
            else
            {
                asm volatile(
                    "cbz    %w10, 0f                    \n"

                    "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                    "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                    "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                    "ld1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0]      \n"
                    "subs   %0, %0, #192                \n"
                    "b      3f                          \n"

                    "0:                                 \n"
                    // if pC
                    "cbz    %8, 1f                      \n"

                    "add    x4, %8, #16                 \n"
                    "ld1    {v16.4s}, [%8]              \n"
                    "ld1    {v24.4s}, [x4]              \n"
                    "b      2f                          \n"

                    // else
                    "1:                                 \n"
                    "eor    v16.16b, v16.16b, v16.16b   \n"
                    "eor    v24.16b, v24.16b, v24.16b   \n"

                    "2:                                 \n"
                    "mov    v17.16b, v16.16b            \n"
                    "mov    v18.16b, v16.16b            \n"
                    "mov    v19.16b, v16.16b            \n"
                    "mov    v20.16b, v16.16b            \n"
                    "mov    v21.16b, v16.16b            \n"
                    "mov    v22.16b, v16.16b            \n"
                    "mov    v23.16b, v16.16b            \n"

                    "mov    v25.16b, v24.16b            \n"
                    "mov    v26.16b, v24.16b            \n"
                    "mov    v27.16b, v24.16b            \n"
                    "mov    v28.16b, v24.16b            \n"
                    "mov    v29.16b, v24.16b            \n"
                    "mov    v30.16b, v24.16b            \n"
                    "mov    v31.16b, v24.16b            \n"

                    "3:                                 \n"
                    "lsr    w4, %w9, #2                 \n" // w4 = max_kk >> 2
                    "cmp    w4, #0                      \n"
                    "beq    5f                          \n"

                    "4:                                 \n"
                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%2], #64 \n"
                    "shll   v0.4s, v4.4h, #16           \n"
                    "shll2  v1.4s, v4.8h, #16           \n"
                    "shll   v2.4s, v5.4h, #16           \n"
                    "shll2  v3.4s, v5.8h, #16           \n"
                    "prfm   pldl1keep, [%1, #512]       \n"
                    "ld1    {v12.8h, v13.8h, v14.8h, v15.8h}, [%1], #64 \n"
                    "shll   v8.4s, v12.4h, #16          \n"
                    "shll2  v9.4s, v12.8h, #16          \n"
                    "shll   v10.4s, v13.4h, #16         \n"
                    "shll2  v11.4s, v13.8h, #16         \n"
                    "fmla   v16.4s, v8.4s, v0.s[0]      \n"
                    "fmla   v17.4s, v8.4s, v0.s[1]      \n"
                    "fmla   v18.4s, v8.4s, v0.s[2]      \n"
                    "fmla   v19.4s, v8.4s, v0.s[3]      \n"
                    "fmla   v20.4s, v8.4s, v1.s[0]      \n"
                    "fmla   v21.4s, v8.4s, v1.s[1]      \n"
                    "fmla   v22.4s, v8.4s, v1.s[2]      \n"
                    "fmla   v23.4s, v8.4s, v1.s[3]      \n"
                    "fmla   v24.4s, v9.4s, v0.s[0]      \n"
                    "fmla   v25.4s, v9.4s, v0.s[1]      \n"
                    "fmla   v26.4s, v9.4s, v0.s[2]      \n"
                    "fmla   v27.4s, v9.4s, v0.s[3]      \n"
                    "fmla   v28.4s, v9.4s, v1.s[0]      \n"
                    "fmla   v29.4s, v9.4s, v1.s[1]      \n"
                    "fmla   v30.4s, v9.4s, v1.s[2]      \n"
                    "fmla   v31.4s, v9.4s, v1.s[3]      \n"
                    "fmla   v16.4s, v10.4s, v2.s[0]     \n"
                    "fmla   v17.4s, v10.4s, v2.s[1]     \n"
                    "fmla   v18.4s, v10.4s, v2.s[2]     \n"
                    "fmla   v19.4s, v10.4s, v2.s[3]     \n"
                    "fmla   v20.4s, v10.4s, v3.s[0]     \n"
                    "fmla   v21.4s, v10.4s, v3.s[1]     \n"
                    "fmla   v22.4s, v10.4s, v3.s[2]     \n"
                    "fmla   v23.4s, v10.4s, v3.s[3]     \n"
                    "fmla   v24.4s, v11.4s, v2.s[0]     \n"
                    "fmla   v25.4s, v11.4s, v2.s[1]     \n"
                    "fmla   v26.4s, v11.4s, v2.s[2]     \n"
                    "fmla   v27.4s, v11.4s, v2.s[3]     \n"
                    "fmla   v28.4s, v11.4s, v3.s[0]     \n"
                    "fmla   v29.4s, v11.4s, v3.s[1]     \n"
                    "fmla   v30.4s, v11.4s, v3.s[2]     \n"
                    "fmla   v31.4s, v11.4s, v3.s[3]     \n"
                    "shll   v4.4s, v6.4h, #16           \n"
                    "shll2  v5.4s, v6.8h, #16           \n"
                    "shll   v6.4s, v7.4h, #16           \n"
                    "shll2  v7.4s, v7.8h, #16           \n"
                    "shll   v12.4s, v14.4h, #16         \n"
                    "shll2  v13.4s, v14.8h, #16         \n"
                    "shll   v14.4s, v15.4h, #16         \n"
                    "shll2  v15.4s, v15.8h, #16         \n"
                    "fmla   v16.4s, v12.4s, v4.s[0]     \n"
                    "fmla   v17.4s, v12.4s, v4.s[1]     \n"
                    "fmla   v18.4s, v12.4s, v4.s[2]     \n"
                    "fmla   v19.4s, v12.4s, v4.s[3]     \n"
                    "fmla   v20.4s, v12.4s, v5.s[0]     \n"
                    "fmla   v21.4s, v12.4s, v5.s[1]     \n"
                    "fmla   v22.4s, v12.4s, v5.s[2]     \n"
                    "fmla   v23.4s, v12.4s, v5.s[3]     \n"
                    "fmla   v24.4s, v13.4s, v4.s[0]     \n"
                    "fmla   v25.4s, v13.4s, v4.s[1]     \n"
                    "fmla   v26.4s, v13.4s, v4.s[2]     \n"
                    "fmla   v27.4s, v13.4s, v4.s[3]     \n"
                    "fmla   v28.4s, v13.4s, v5.s[0]     \n"
                    "fmla   v29.4s, v13.4s, v5.s[1]     \n"
                    "fmla   v30.4s, v13.4s, v5.s[2]     \n"
                    "fmla   v31.4s, v13.4s, v5.s[3]     \n"
                    "fmla   v16.4s, v14.4s, v6.s[0]     \n"
                    "fmla   v17.4s, v14.4s, v6.s[1]     \n"
                    "fmla   v18.4s, v14.4s, v6.s[2]     \n"
                    "fmla   v19.4s, v14.4s, v6.s[3]     \n"
                    "fmla   v20.4s, v14.4s, v7.s[0]     \n"
                    "fmla   v21.4s, v14.4s, v7.s[1]     \n"
                    "fmla   v22.4s, v14.4s, v7.s[2]     \n"
                    "fmla   v23.4s, v14.4s, v7.s[3]     \n"
                    "subs   w4, w4, #1                  \n"
                    "fmla   v24.4s, v15.4s, v6.s[0]     \n"
                    "fmla   v25.4s, v15.4s, v6.s[1]     \n"
                    "fmla   v26.4s, v15.4s, v6.s[2]     \n"
                    "fmla   v27.4s, v15.4s, v6.s[3]     \n"
                    "fmla   v28.4s, v15.4s, v7.s[0]     \n"
                    "fmla   v29.4s, v15.4s, v7.s[1]     \n"
                    "fmla   v30.4s, v15.4s, v7.s[2]     \n"
                    "fmla   v31.4s, v15.4s, v7.s[3]     \n"
                    "bne    4b                          \n"

                    "5:                                 \n"
                    "and    w4, %w9, #3                 \n" // w4 = remain = max_kk & 3
                    "cmp    w4, #0                      \n"
                    "beq    7f                          \n"

                    "6:                                 \n"
                    "ld1    {v2.8h}, [%2], #16          \n"
                    "shll   v0.4s, v2.4h, #16           \n"
                    "shll2  v1.4s, v2.8h, #16           \n"
                    "ld1    {v3.8h}, [%1], #16          \n"
                    "shll   v4.4s, v3.4h, #16           \n"
                    "shll2  v5.4s, v3.8h, #16           \n"

                    "fmla   v16.4s, v4.4s, v0.s[0]      \n"
                    "fmla   v17.4s, v4.4s, v0.s[1]      \n"
                    "fmla   v18.4s, v4.4s, v0.s[2]      \n"
                    "fmla   v19.4s, v4.4s, v0.s[3]      \n"
                    "fmla   v20.4s, v4.4s, v1.s[0]      \n"
                    "fmla   v21.4s, v4.4s, v1.s[1]      \n"
                    "fmla   v22.4s, v4.4s, v1.s[2]      \n"
                    "fmla   v23.4s, v4.4s, v1.s[3]      \n"

                    "subs   w4, w4, #1                  \n"

                    "fmla   v24.4s, v5.4s, v0.s[0]      \n"
                    "fmla   v25.4s, v5.4s, v0.s[1]      \n"
                    "fmla   v26.4s, v5.4s, v0.s[2]      \n"
                    "fmla   v27.4s, v5.4s, v0.s[3]      \n"
                    "fmla   v28.4s, v5.4s, v1.s[0]      \n"
                    "fmla   v29.4s, v5.4s, v1.s[1]      \n"
                    "fmla   v30.4s, v5.4s, v1.s[2]      \n"
                    "fmla   v31.4s, v5.4s, v1.s[3]      \n"

                    "bne    6b                          \n"

                    "7:                                 \n"
                    "shrn   v0.4h, v16.4s, #16          \n"
                    "shrn2  v0.8h, v17.4s, #16          \n"
                    "shrn   v1.4h, v18.4s, #16          \n"
                    "shrn2  v1.8h, v19.4s, #16          \n"
                    "shrn   v2.4h, v20.4s, #16          \n"
                    "shrn2  v2.8h, v21.4s, #16          \n"
                    "shrn   v3.4h, v22.4s, #16          \n"
                    "shrn2  v3.8h, v23.4s, #16          \n"
                    "shrn   v4.4h, v24.4s, #16          \n"
                    "shrn2  v4.8h, v25.4s, #16          \n"
                    "shrn   v5.4h, v26.4s, #16          \n"
                    "shrn2  v5.8h, v27.4s, #16          \n"
                    "shrn   v6.4h, v28.4s, #16          \n"
                    "shrn2  v6.8h, v29.4s, #16          \n"
                    "shrn   v7.4h, v30.4s, #16          \n"
                    "shrn2  v7.8h, v31.4s, #16          \n"
                    "tst    %w11, #255                  \n"
                    "beq    10f                         \n"

                    // if out_elempack == 4
                    "cmp    %w12, #4                    \n"
                    "bne    8f                          \n"

                    "lsl    w4, %w13, #2                \n"
                    "add    x4, %3, w4, sxtw 1          \n"
                    "st1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%3], #64 \n"
                    "st1    {v4.8h, v5.8h, v6.8h, v7.8h}, [x4] \n"
                    "b      9f                          \n"

                    // if out_elempack == 1
                    "8:                                 \n"
                    // transpose8x8
                    "uzp1   v24.8h, v0.8h, v1.8h        \n"
                    "uzp2   v25.8h, v0.8h, v1.8h        \n"
                    "uzp1   v26.8h, v2.8h, v3.8h        \n"
                    "uzp2   v27.8h, v2.8h, v3.8h        \n"
                    "uzp1   v28.8h, v4.8h, v5.8h        \n"
                    "uzp2   v29.8h, v4.8h, v5.8h        \n"
                    "uzp1   v30.8h, v6.8h, v7.8h        \n"
                    "uzp2   v31.8h, v6.8h, v7.8h        \n"

                    "uzp1   v0.8h, v24.8h, v26.8h       \n"
                    "uzp2   v2.8h, v24.8h, v26.8h       \n"
                    "uzp1   v1.8h, v25.8h, v27.8h       \n"
                    "uzp2   v3.8h, v25.8h, v27.8h       \n"

                    "uzp1   v4.8h, v28.8h, v30.8h       \n"
                    "uzp2   v6.8h, v28.8h, v30.8h       \n"
                    "uzp1   v5.8h, v29.8h, v31.8h       \n"
                    "uzp2   v7.8h, v29.8h, v31.8h       \n"

                    "add    x4, %3, %w13, sxtw 1        \n"
                    "st1    {v0.8h}, [%3], #16          \n"
                    "st1    {v1.8h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v2.8h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v3.8h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v4.8h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v5.8h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v6.8h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v7.8h}, [x4]               \n"

                    "9:                                 \n"
                    "add    %0, %0, #256                \n"
                    "b      11f                         \n"

                    "10:                                \n"
                    "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                    "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                    "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                    "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0], #64 \n"

                    "11:                                \n"

                    : "=r"(outptr), // %0
                    "=r"(pA),     // %1
                    "=r"(pB),     // %2
                    "=r"(outptr0) // %3
                    : "0"(outptr),
                    "1"(pA),
                    "2"(pB),
                    "3"(outptr0),
                    "r"(pC),           // %8
                    "r"(max_kk),       // %9
                    "r"(k),            // %10
                    "r"(k_end),        // %11
                    "r"(out_elempack), // %12
                    "r"(out_hstep)     // %13
                    : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
            }
#else  // NCNN_GNU_INLINE_ASM
            float32x4_t _sum00;
            float32x4_t _sum01;
            float32x4_t _sum10;
            float32x4_t _sum11;
            float32x4_t _sum20;
            float32x4_t _sum21;
            float32x4_t _sum30;
            float32x4_t _sum31;
            float32x4_t _sum40;
            float32x4_t _sum41;
            float32x4_t _sum50;
            float32x4_t _sum51;
            float32x4_t _sum60;
            float32x4_t _sum61;
            float32x4_t _sum70;
            float32x4_t _sum71;

            if (k == 0)
            {
                if (pC)
                {
                    _sum00 = vld1q_f32(pC);
                    _sum01 = vld1q_f32(pC + 4);
                    _sum10 = _sum00;
                    _sum11 = _sum01;
                    _sum20 = _sum00;
                    _sum21 = _sum01;
                    _sum30 = _sum00;
                    _sum31 = _sum01;
                    _sum40 = _sum00;
                    _sum41 = _sum01;
                    _sum50 = _sum00;
                    _sum51 = _sum01;
                    _sum60 = _sum00;
                    _sum61 = _sum01;
                    _sum70 = _sum00;
                    _sum71 = _sum01;
                }
                else
                {
                    _sum00 = vdupq_n_f32(0.f);
                    _sum01 = vdupq_n_f32(0.f);
                    _sum10 = vdupq_n_f32(0.f);
                    _sum11 = vdupq_n_f32(0.f);
                    _sum20 = vdupq_n_f32(0.f);
                    _sum21 = vdupq_n_f32(0.f);
                    _sum30 = vdupq_n_f32(0.f);
                    _sum31 = vdupq_n_f32(0.f);
                    _sum40 = vdupq_n_f32(0.f);
                    _sum41 = vdupq_n_f32(0.f);
                    _sum50 = vdupq_n_f32(0.f);
                    _sum51 = vdupq_n_f32(0.f);
                    _sum60 = vdupq_n_f32(0.f);
                    _sum61 = vdupq_n_f32(0.f);
                    _sum70 = vdupq_n_f32(0.f);
                    _sum71 = vdupq_n_f32(0.f);
                }
            }
            else
            {
                _sum00 = vld1q_f32(outptr);
                _sum01 = vld1q_f32(outptr + 4 * 1);
                _sum10 = vld1q_f32(outptr + 4 * 2);
                _sum11 = vld1q_f32(outptr + 4 * 3);
                _sum20 = vld1q_f32(outptr + 4 * 4);
                _sum21 = vld1q_f32(outptr + 4 * 5);
                _sum30 = vld1q_f32(outptr + 4 * 6);
                _sum31 = vld1q_f32(outptr + 4 * 7);
                _sum40 = vld1q_f32(outptr + 4 * 8);
                _sum41 = vld1q_f32(outptr + 4 * 9);
                _sum50 = vld1q_f32(outptr + 4 * 10);
                _sum51 = vld1q_f32(outptr + 4 * 11);
                _sum60 = vld1q_f32(outptr + 4 * 12);
                _sum61 = vld1q_f32(outptr + 4 * 13);
                _sum70 = vld1q_f32(outptr + 4 * 14);
                _sum71 = vld1q_f32(outptr + 4 * 15);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA0 = bfloat2float(vld1_u16(pA));
                float32x4_t _pA1 = bfloat2float(vld1_u16(pA + 4));

                float32x4_t _pB0 = bfloat2float(vld1_u16(pB));
                float32x4_t _pB1 = bfloat2float(vld1_u16(pB + 4));

                _sum00 = vfmaq_laneq_f32(_sum00, _pA0, _pB0, 0);
                _sum01 = vfmaq_laneq_f32(_sum01, _pA1, _pB0, 0);
                _sum10 = vfmaq_laneq_f32(_sum10, _pA0, _pB0, 1);
                _sum11 = vfmaq_laneq_f32(_sum11, _pA1, _pB0, 1);
                _sum20 = vfmaq_laneq_f32(_sum20, _pA0, _pB0, 2);
                _sum21 = vfmaq_laneq_f32(_sum21, _pA1, _pB0, 2);
                _sum30 = vfmaq_laneq_f32(_sum30, _pA0, _pB0, 3);
                _sum31 = vfmaq_laneq_f32(_sum31, _pA1, _pB0, 3);
                _sum40 = vfmaq_laneq_f32(_sum40, _pA0, _pB1, 0);
                _sum41 = vfmaq_laneq_f32(_sum41, _pA1, _pB1, 0);
                _sum50 = vfmaq_laneq_f32(_sum50, _pA0, _pB1, 1);
                _sum51 = vfmaq_laneq_f32(_sum51, _pA1, _pB1, 1);
                _sum60 = vfmaq_laneq_f32(_sum60, _pA0, _pB1, 2);
                _sum61 = vfmaq_laneq_f32(_sum61, _pA1, _pB1, 2);
                _sum70 = vfmaq_laneq_f32(_sum70, _pA0, _pB1, 3);
                _sum71 = vfmaq_laneq_f32(_sum71, _pA1, _pB1, 3);

                pA += 8;
                pB += 8;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1_u16(outptr0, float2bfloat(_sum00));
                    vst1_u16(outptr0 + 4, float2bfloat(_sum10));
                    vst1_u16(outptr0 + 4 * 2, float2bfloat(_sum20));
                    vst1_u16(outptr0 + 4 * 3, float2bfloat(_sum30));
                    vst1_u16(outptr0 + 4 * 4, float2bfloat(_sum40));
                    vst1_u16(outptr0 + 4 * 5, float2bfloat(_sum50));
                    vst1_u16(outptr0 + 4 * 6, float2bfloat(_sum60));
                    vst1_u16(outptr0 + 4 * 7, float2bfloat(_sum70));

                    vst1_u16(outptr0 + out_hstep * 4, float2bfloat(_sum01));
                    vst1_u16(outptr0 + out_hstep * 4 + 4, float2bfloat(_sum11));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 2, float2bfloat(_sum21));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 3, float2bfloat(_sum31));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 4, float2bfloat(_sum41));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 5, float2bfloat(_sum51));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 6, float2bfloat(_sum61));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 7, float2bfloat(_sum71));

                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    uint16x8_t _t0 = vcombine_u16(float2bfloat(_sum00), float2bfloat(_sum01));
                    uint16x8_t _t1 = vcombine_u16(float2bfloat(_sum10), float2bfloat(_sum11));
                    uint16x8_t _t2 = vcombine_u16(float2bfloat(_sum20), float2bfloat(_sum21));
                    uint16x8_t _t3 = vcombine_u16(float2bfloat(_sum30), float2bfloat(_sum31));
                    uint16x8_t _t4 = vcombine_u16(float2bfloat(_sum40), float2bfloat(_sum41));
                    uint16x8_t _t5 = vcombine_u16(float2bfloat(_sum50), float2bfloat(_sum51));
                    uint16x8_t _t6 = vcombine_u16(float2bfloat(_sum60), float2bfloat(_sum61));
                    uint16x8_t _t7 = vcombine_u16(float2bfloat(_sum70), float2bfloat(_sum71));
                    transpose8x8_u16(_t0, _t1, _t2, _t3, _t4, _t5, _t6, _t7);

                    vst1q_u16(outptr0, _t0);
                    vst1q_u16(outptr0 + out_hstep, _t1);
                    vst1q_u16(outptr0 + out_hstep * 2, _t2);
                    vst1q_u16(outptr0 + out_hstep * 3, _t3);
                    vst1q_u16(outptr0 + out_hstep * 4, _t4);
                    vst1q_u16(outptr0 + out_hstep * 5, _t5);
                    vst1q_u16(outptr0 + out_hstep * 6, _t6);
                    vst1q_u16(outptr0 + out_hstep * 7, _t7);

                    outptr0 += 8;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum00);
                vst1q_f32(outptr + 4, _sum01);
                vst1q_f32(outptr + 4 * 2, _sum10);
                vst1q_f32(outptr + 4 * 3, _sum11);
                vst1q_f32(outptr + 4 * 4, _sum20);
                vst1q_f32(outptr + 4 * 5, _sum21);
                vst1q_f32(outptr + 4 * 6, _sum30);
                vst1q_f32(outptr + 4 * 7, _sum31);
                vst1q_f32(outptr + 4 * 8, _sum40);
                vst1q_f32(outptr + 4 * 9, _sum41);
                vst1q_f32(outptr + 4 * 10, _sum50);
                vst1q_f32(outptr + 4 * 11, _sum51);
                vst1q_f32(outptr + 4 * 12, _sum60);
                vst1q_f32(outptr + 4 * 13, _sum61);
                vst1q_f32(outptr + 4 * 14, _sum70);
                vst1q_f32(outptr + 4 * 15, _sum71);
            }

            outptr += 64;
#endif // NCNN_GNU_INLINE_ASM
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            const unsigned short* pA = pAT;

#if NCNN_GNU_INLINE_ASM
            if (use_a53_a55_optimized_kernel && cpu_support_arm_asimdhp())
            {
                // a55
                asm volatile(
                    "cbz    %w10, 0f                    \n"

                    "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                    "ld1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0]      \n"
                    "subs   %0, %0, #64                 \n"
                    "b      3f                          \n"

                    "0:                                 \n"
                    // if pC
                    "cbz    %8, 1f                      \n"

                    "add    x4, %8, #16                 \n"
                    "ld1    {v24.4s}, [%8]              \n"
                    "ld1    {v28.4s}, [x4]              \n"
                    "b      2f                          \n"

                    // else
                    "1:                                 \n"
                    "eor    v24.16b, v24.16b, v24.16b   \n"
                    "eor    v28.16b, v28.16b, v28.16b   \n"

                    "2:                                 \n"
                    "mov    v25.16b, v24.16b            \n"
                    "mov    v26.16b, v24.16b            \n"
                    "mov    v27.16b, v24.16b            \n"

                    "mov    v29.16b, v28.16b            \n"
                    "mov    v30.16b, v28.16b            \n"
                    "mov    v31.16b, v28.16b            \n"

                    "3:                                 \n"
                    "lsr    w4, %w9, #2                 \n" // w4 = max_kk >> 2
                    "cmp    w4, #0                      \n"
                    "beq    5f                          \n"

                    "prfm   pldl1keep, [%1, #512]       \n"
                    "ld1    {v4.4h, v5.4h, v6.4h}, [%1], #24 \n"

                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v0.4h, v1.4h}, [%2], #16   \n"

                    "shll   v0.4s, v0.4h, #16           \n"
                    "shll   v4.4s, v4.4h, #16           \n"

                    ".align 4                           \n"
                    "4:                                 \n"
                    "shll   v5.4s, v5.4h, #16           \n"
                    "fmla   v24.4s, v4.4s, v0.s[0]      \n"
                    "ldr    d7, [%1], #8                \n"
                    "fmla   v25.4s, v4.4s, v0.s[1]      \n"
                    "shll   v1.4s, v1.4h, #16           \n"
                    "fmla   v26.4s, v4.4s, v0.s[2]      \n"
                    "ldr    d2, [%2], #8                \n"
                    "fmla   v27.4s, v4.4s, v0.s[3]      \n"
                    "shll   v6.4s, v6.4h, #16           \n"
                    "fmla   v28.4s, v5.4s, v0.s[0]      \n"
                    "ldr    d8, [%1], #8                \n"
                    "fmla   v29.4s, v5.4s, v0.s[1]      \n"
                    "ldr    d9, [%1], #8                \n"
                    "fmla   v30.4s, v5.4s, v0.s[2]      \n"
                    "ldr    d3, [%2], #8                \n"
                    "fmla   v31.4s, v5.4s, v0.s[3]      \n"
                    "shll   v7.4s, v7.4h, #16           \n"
                    "fmla   v24.4s, v6.4s, v1.s[0]      \n"
                    "ldr    d10, [%1], #8               \n"
                    "fmla   v25.4s, v6.4s, v1.s[1]      \n"
                    "shll   v2.4s, v2.4h, #16           \n"
                    "fmla   v26.4s, v6.4s, v1.s[2]      \n"
                    "ldr    d11, [%1], #8               \n"
                    "fmla   v27.4s, v6.4s, v1.s[3]      \n"
                    "shll   v8.4s, v8.4h, #16           \n"
                    "fmla   v28.4s, v7.4s, v1.s[0]      \n"
                    "prfm   pldl1keep, [%1, #512]       \n" // NOTE PRELOAD
                    "fmla   v29.4s, v7.4s, v1.s[1]      \n"
                    "ldr    d4, [%1], #8                \n"
                    "fmla   v30.4s, v7.4s, v1.s[2]      \n"
                    "prfm   pldl1keep, [%2, #256]       \n" // NOTE PRELOAD
                    "fmla   v31.4s, v7.4s, v1.s[3]      \n"
                    "shll   v9.4s, v9.4h, #16           \n"
                    "fmla   v24.4s, v8.4s, v2.s[0]      \n"
                    "ldr    d0, [%2], #8                \n"
                    "fmla   v25.4s, v8.4s, v2.s[1]      \n"
                    "shll   v3.4s, v3.4h, #16           \n"
                    "fmla   v26.4s, v8.4s, v2.s[2]      \n"
                    "ldr    d5, [%1], #8                \n"
                    "fmla   v27.4s, v8.4s, v2.s[3]      \n"
                    "shll   v10.4s, v10.4h, #16         \n"
                    "fmla   v28.4s, v9.4s, v2.s[0]      \n"
                    "ldr    d1, [%2], #8                \n"
                    "fmla   v29.4s, v9.4s, v2.s[1]      \n"
                    "ldr    d6, [%1], #8                \n"
                    "fmla   v30.4s, v9.4s, v2.s[2]      \n"
                    "fmla   v31.4s, v9.4s, v2.s[3]      \n"
                    "shll   v11.4s, v11.4h, #16         \n"
                    "fmla   v24.4s, v10.4s, v3.s[0]     \n"
                    "fmla   v25.4s, v10.4s, v3.s[1]     \n"
                    "shll   v4.4s, v4.4h, #16           \n"
                    "fmla   v26.4s, v10.4s, v3.s[2]     \n"
                    "fmla   v27.4s, v10.4s, v3.s[3]     \n"
                    "shll   v0.4s, v0.4h, #16           \n"
                    "fmla   v28.4s, v11.4s, v3.s[0]     \n"
                    "fmla   v29.4s, v11.4s, v3.s[1]     \n"
                    "subs   w4, w4, #1                  \n"
                    "fmla   v30.4s, v11.4s, v3.s[2]     \n"
                    "fmla   v31.4s, v11.4s, v3.s[3]     \n"
                    "bne    4b                          \n"

                    "sub    %1, %1, #24                 \n"
                    "sub    %2, %2, #16                 \n"

                    "5:                                 \n"
                    "and    w4, %w9, #3                 \n" // w4 = remain = max_kk & 3
                    "cmp    w4, #0                      \n"
                    "beq    7f                          \n"

                    "6:                                 \n"
                    "ld1    {v0.4h}, [%2], #8           \n"
                    "shll   v0.4s, v0.4h, #16           \n"
                    "ld1    {v3.8h}, [%1], #16          \n"
                    "shll   v4.4s, v3.4h, #16           \n"
                    "shll2  v5.4s, v3.8h, #16           \n"
                    "fmla   v24.4s, v4.4s, v0.s[0]      \n"
                    "fmla   v25.4s, v4.4s, v0.s[1]      \n"
                    "fmla   v26.4s, v4.4s, v0.s[2]      \n"
                    "fmla   v27.4s, v4.4s, v0.s[3]      \n"
                    "subs   w4, w4, #1                  \n"
                    "fmla   v28.4s, v5.4s, v0.s[0]      \n"
                    "fmla   v29.4s, v5.4s, v0.s[1]      \n"
                    "fmla   v30.4s, v5.4s, v0.s[2]      \n"
                    "fmla   v31.4s, v5.4s, v0.s[3]      \n"
                    "bne    6b                          \n"

                    "7:                                 \n"
                    "shrn   v0.4h, v24.4s, #16          \n"
                    "shrn2  v0.8h, v25.4s, #16          \n"
                    "shrn   v1.4h, v26.4s, #16          \n"
                    "shrn2  v1.8h, v27.4s, #16          \n"
                    "shrn   v2.4h, v28.4s, #16          \n"
                    "shrn2  v2.8h, v29.4s, #16          \n"
                    "shrn   v3.4h, v30.4s, #16          \n"
                    "shrn2  v3.8h, v31.4s, #16          \n"
                    "tst    %w11, #255                  \n"
                    "beq    10f                         \n"

                    // if out_elempack == 4
                    "cmp    %w12, #4                    \n"
                    "bne    8f                          \n"

                    "lsl    w4, %w13, #2                \n"
                    "add    x4, %3, w4, sxtw 1          \n"
                    "st1    {v0.8h, v1.8h}, [%3], #32   \n"
                    "st1    {v2.8h, v3.8h}, [x4]        \n"
                    "b      9f                          \n"

                    // if out_elempack == 1
                    "8:                                 \n"
                    // transpose8x4
                    "uzp1   v28.8h, v0.8h, v1.8h        \n"
                    "uzp2   v29.8h, v0.8h, v1.8h        \n"
                    "uzp1   v30.8h, v2.8h, v3.8h        \n"
                    "uzp2   v31.8h, v2.8h, v3.8h        \n"

                    "uzp1   v0.8h, v28.8h, v29.8h       \n"
                    "uzp2   v2.8h, v28.8h, v29.8h       \n"
                    "uzp1   v4.8h, v30.8h, v31.8h       \n"
                    "uzp2   v6.8h, v30.8h, v31.8h       \n"

                    "mov    v1.d[0], v0.d[1]            \n"
                    "mov    v3.d[0], v2.d[1]            \n"
                    "mov    v5.d[0], v4.d[1]            \n"
                    "mov    v7.d[0], v6.d[1]            \n"

                    "add    x4, %3, %w13, sxtw 1        \n"
                    "st1    {v0.4h}, [%3], #8           \n"
                    "st1    {v1.4h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v2.4h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v3.4h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v4.4h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v5.4h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v6.4h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v7.4h}, [x4]               \n"

                    "9:                                 \n"
                    "add    %0, %0, #128                \n"
                    "b      11f                         \n"

                    "10:                                \n"
                    "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                    "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0], #64 \n"

                    "11:                                \n"

                    : "=r"(outptr), // %0
                    "=r"(pA),     // %1
                    "=r"(pB),     // %2
                    "=r"(outptr0) // %3
                    : "0"(outptr),
                    "1"(pA),
                    "2"(pB),
                    "3"(outptr0),
                    "r"(pC),           // %8
                    "r"(max_kk),       // %9
                    "r"(k),            // %10
                    "r"(k_end),        // %11
                    "r"(out_elempack), // %12
                    "r"(out_hstep)     // %13
                    : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
            }
            else
            {
                asm volatile(
                    "cbz    %w10, 0f                    \n"

                    "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                    "ld1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0]      \n"
                    "subs   %0, %0, #64                 \n"
                    "b      3f                          \n"

                    "0:                                 \n"
                    // if pC
                    "cbz    %8, 1f                      \n"

                    "add    x4, %8, #16                 \n"
                    "ld1    {v24.4s}, [%8]              \n"
                    "ld1    {v28.4s}, [x4]              \n"
                    "b      2f                          \n"

                    // else
                    "1:                                 \n"
                    "eor    v24.16b, v24.16b, v24.16b   \n"
                    "eor    v28.16b, v28.16b, v28.16b   \n"

                    "2:                                 \n"
                    "mov    v25.16b, v24.16b            \n"
                    "mov    v26.16b, v24.16b            \n"
                    "mov    v27.16b, v24.16b            \n"

                    "mov    v29.16b, v28.16b            \n"
                    "mov    v30.16b, v28.16b            \n"
                    "mov    v31.16b, v28.16b            \n"

                    "3:                                 \n"
                    "lsr    w4, %w9, #2                 \n" // w4 = max_kk >> 2
                    "cmp    w4, #0                      \n"
                    "beq    5f                          \n"

                    "4:                                 \n"
                    "prfm   pldl1keep, [%2, #256]       \n"
                    "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%2], #32 \n"
                    "shll   v0.4s, v0.4h, #16           \n"
                    "shll   v1.4s, v1.4h, #16           \n"
                    "shll   v2.4s, v2.4h, #16           \n"
                    "shll   v3.4s, v3.4h, #16           \n"
                    "prfm   pldl1keep, [%1, #512]       \n"
                    "ld1    {v12.8h, v13.8h, v14.8h, v15.8h}, [%1], #64 \n"
                    "shll   v4.4s, v12.4h, #16          \n"
                    "shll2  v5.4s, v12.8h, #16          \n"
                    "shll   v6.4s, v13.4h, #16          \n"
                    "shll2  v7.4s, v13.8h, #16          \n"
                    "fmla   v24.4s, v4.4s, v0.s[0]      \n"
                    "fmla   v25.4s, v4.4s, v0.s[1]      \n"
                    "fmla   v26.4s, v4.4s, v0.s[2]      \n"
                    "fmla   v27.4s, v4.4s, v0.s[3]      \n"
                    "fmla   v28.4s, v5.4s, v0.s[0]      \n"
                    "fmla   v29.4s, v5.4s, v0.s[1]      \n"
                    "fmla   v30.4s, v5.4s, v0.s[2]      \n"
                    "fmla   v31.4s, v5.4s, v0.s[3]      \n"
                    "fmla   v24.4s, v6.4s, v1.s[0]      \n"
                    "fmla   v25.4s, v6.4s, v1.s[1]      \n"
                    "fmla   v26.4s, v6.4s, v1.s[2]      \n"
                    "fmla   v27.4s, v6.4s, v1.s[3]      \n"
                    "fmla   v28.4s, v7.4s, v1.s[0]      \n"
                    "fmla   v29.4s, v7.4s, v1.s[1]      \n"
                    "fmla   v30.4s, v7.4s, v1.s[2]      \n"
                    "fmla   v31.4s, v7.4s, v1.s[3]      \n"
                    "shll   v8.4s, v14.4h, #16          \n"
                    "shll2  v9.4s, v14.8h, #16          \n"
                    "shll   v10.4s, v15.4h, #16         \n"
                    "shll2  v11.4s, v15.8h, #16         \n"
                    "fmla   v24.4s, v8.4s, v2.s[0]      \n"
                    "fmla   v25.4s, v8.4s, v2.s[1]      \n"
                    "fmla   v26.4s, v8.4s, v2.s[2]      \n"
                    "fmla   v27.4s, v8.4s, v2.s[3]      \n"
                    "fmla   v28.4s, v9.4s, v2.s[0]      \n"
                    "fmla   v29.4s, v9.4s, v2.s[1]      \n"
                    "fmla   v30.4s, v9.4s, v2.s[2]      \n"
                    "fmla   v31.4s, v9.4s, v2.s[3]      \n"
                    "subs   w4, w4, #1                  \n"
                    "fmla   v24.4s, v10.4s, v3.s[0]     \n"
                    "fmla   v25.4s, v10.4s, v3.s[1]     \n"
                    "fmla   v26.4s, v10.4s, v3.s[2]     \n"
                    "fmla   v27.4s, v10.4s, v3.s[3]     \n"
                    "fmla   v28.4s, v11.4s, v3.s[0]     \n"
                    "fmla   v29.4s, v11.4s, v3.s[1]     \n"
                    "fmla   v30.4s, v11.4s, v3.s[2]     \n"
                    "fmla   v31.4s, v11.4s, v3.s[3]     \n"
                    "bne    4b                          \n"

                    "5:                                 \n"
                    "and    w4, %w9, #3                 \n" // w4 = remain = max_kk & 3
                    "cmp    w4, #0                      \n"
                    "beq    7f                          \n"

                    "6:                                 \n"
                    "ld1    {v0.4h}, [%2], #8           \n"
                    "shll   v0.4s, v0.4h, #16           \n"
                    "ld1    {v3.8h}, [%1], #16          \n"
                    "shll   v4.4s, v3.4h, #16           \n"
                    "shll2  v5.4s, v3.8h, #16           \n"
                    "fmla   v24.4s, v4.4s, v0.s[0]      \n"
                    "fmla   v25.4s, v4.4s, v0.s[1]      \n"
                    "fmla   v26.4s, v4.4s, v0.s[2]      \n"
                    "fmla   v27.4s, v4.4s, v0.s[3]      \n"
                    "subs   w4, w4, #1                  \n"
                    "fmla   v28.4s, v5.4s, v0.s[0]      \n"
                    "fmla   v29.4s, v5.4s, v0.s[1]      \n"
                    "fmla   v30.4s, v5.4s, v0.s[2]      \n"
                    "fmla   v31.4s, v5.4s, v0.s[3]      \n"
                    "bne    6b                          \n"

                    "7:                                 \n"
                    "shrn   v0.4h, v24.4s, #16          \n"
                    "shrn2  v0.8h, v25.4s, #16          \n"
                    "shrn   v1.4h, v26.4s, #16          \n"
                    "shrn2  v1.8h, v27.4s, #16          \n"
                    "shrn   v2.4h, v28.4s, #16          \n"
                    "shrn2  v2.8h, v29.4s, #16          \n"
                    "shrn   v3.4h, v30.4s, #16          \n"
                    "shrn2  v3.8h, v31.4s, #16          \n"
                    "tst    %w11, #255                  \n"
                    "beq    10f                         \n"

                    // if out_elempack == 4
                    "cmp    %w12, #4                    \n"
                    "bne    8f                          \n"

                    "lsl    w4, %w13, #2                \n"
                    "add    x4, %3, w4, sxtw 1          \n"
                    "st1    {v0.8h, v1.8h}, [%3], #32   \n"
                    "st1    {v2.8h, v3.8h}, [x4]        \n"
                    "b      9f                          \n"

                    // if out_elempack == 1
                    "8:                                 \n"
                    // transpose8x4
                    "uzp1   v28.8h, v0.8h, v1.8h        \n"
                    "uzp2   v29.8h, v0.8h, v1.8h        \n"
                    "uzp1   v30.8h, v2.8h, v3.8h        \n"
                    "uzp2   v31.8h, v2.8h, v3.8h        \n"

                    "uzp1   v0.8h, v28.8h, v29.8h       \n"
                    "uzp2   v2.8h, v28.8h, v29.8h       \n"
                    "uzp1   v4.8h, v30.8h, v31.8h       \n"
                    "uzp2   v6.8h, v30.8h, v31.8h       \n"

                    "mov    v1.d[0], v0.d[1]            \n"
                    "mov    v3.d[0], v2.d[1]            \n"
                    "mov    v5.d[0], v4.d[1]            \n"
                    "mov    v7.d[0], v6.d[1]            \n"

                    "add    x4, %3, %w13, sxtw 1        \n"
                    "st1    {v0.4h}, [%3], #8           \n"
                    "st1    {v1.4h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v2.4h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v3.4h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v4.4h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v5.4h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v6.4h}, [x4]               \n"
                    "add    x4, x4, %w13, sxtw 1        \n"
                    "st1    {v7.4h}, [x4]               \n"

                    "9:                                 \n"
                    "add    %0, %0, #128                \n"
                    "b      11f                         \n"

                    "10:                                \n"
                    "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                    "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0], #64 \n"

                    "11:                                \n"

                    : "=r"(outptr), // %0
                    "=r"(pA),     // %1
                    "=r"(pB),     // %2
                    "=r"(outptr0) // %3
                    : "0"(outptr),
                    "1"(pA),
                    "2"(pB),
                    "3"(outptr0),
                    "r"(pC),           // %8
                    "r"(max_kk),       // %9
                    "r"(k),            // %10
                    "r"(k_end),        // %11
                    "r"(out_elempack), // %12
                    "r"(out_hstep)     // %13
                    : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
            }
#else  // NCNN_GNU_INLINE_ASM
            float32x4_t _sum00;
            float32x4_t _sum01;
            float32x4_t _sum10;
            float32x4_t _sum11;
            float32x4_t _sum20;
            float32x4_t _sum21;
            float32x4_t _sum30;
            float32x4_t _sum31;

            if (k == 0)
            {
                if (pC)
                {
                    _sum00 = vld1q_f32(pC);
                    _sum01 = vld1q_f32(pC + 4);
                    _sum10 = _sum00;
                    _sum11 = _sum01;
                    _sum20 = _sum00;
                    _sum21 = _sum01;
                    _sum30 = _sum00;
                    _sum31 = _sum01;
                }
                else
                {
                    _sum00 = vdupq_n_f32(0.f);
                    _sum01 = vdupq_n_f32(0.f);
                    _sum10 = vdupq_n_f32(0.f);
                    _sum11 = vdupq_n_f32(0.f);
                    _sum20 = vdupq_n_f32(0.f);
                    _sum21 = vdupq_n_f32(0.f);
                    _sum30 = vdupq_n_f32(0.f);
                    _sum31 = vdupq_n_f32(0.f);
                }
            }
            else
            {
                _sum00 = vld1q_f32(outptr);
                _sum01 = vld1q_f32(outptr + 4 * 1);
                _sum10 = vld1q_f32(outptr + 4 * 2);
                _sum11 = vld1q_f32(outptr + 4 * 3);
                _sum20 = vld1q_f32(outptr + 4 * 4);
                _sum21 = vld1q_f32(outptr + 4 * 5);
                _sum30 = vld1q_f32(outptr + 4 * 6);
                _sum31 = vld1q_f32(outptr + 4 * 7);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA0 = bfloat2float(vld1_u16(pA));
                float32x4_t _pA1 = bfloat2float(vld1_u16(pA + 4));

                float32x4_t _pB0 = bfloat2float(vld1_u16(pB));

                _sum00 = vfmaq_laneq_f32(_sum00, _pA0, _pB0, 0);
                _sum01 = vfmaq_laneq_f32(_sum01, _pA1, _pB0, 0);
                _sum10 = vfmaq_laneq_f32(_sum10, _pA0, _pB0, 1);
                _sum11 = vfmaq_laneq_f32(_sum11, _pA1, _pB0, 1);
                _sum20 = vfmaq_laneq_f32(_sum20, _pA0, _pB0, 2);
                _sum21 = vfmaq_laneq_f32(_sum21, _pA1, _pB0, 2);
                _sum30 = vfmaq_laneq_f32(_sum30, _pA0, _pB0, 3);
                _sum31 = vfmaq_laneq_f32(_sum31, _pA1, _pB0, 3);

                pA += 8;
                pB += 4;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1_u16(outptr0, float2bfloat(_sum00));
                    vst1_u16(outptr0 + 4, float2bfloat(_sum10));
                    vst1_u16(outptr0 + 4 * 2, float2bfloat(_sum20));
                    vst1_u16(outptr0 + 4 * 3, float2bfloat(_sum30));

                    vst1_u16(outptr0 + out_hstep * 4, float2bfloat(_sum01));
                    vst1_u16(outptr0 + out_hstep * 4 + 4, float2bfloat(_sum11));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 2, float2bfloat(_sum21));
                    vst1_u16(outptr0 + out_hstep * 4 + 4 * 3, float2bfloat(_sum31));

                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    uint16x8_t _t0 = vcombine_u16(float2bfloat(_sum00), float2bfloat(_sum01));
                    uint16x8_t _t1 = vcombine_u16(float2bfloat(_sum10), float2bfloat(_sum11));
                    uint16x8_t _t2 = vcombine_u16(float2bfloat(_sum20), float2bfloat(_sum21));
                    uint16x8_t _t3 = vcombine_u16(float2bfloat(_sum30), float2bfloat(_sum31));
                    transpose8x4_u16(_t0, _t1, _t2, _t3);

                    vst1_u16(outptr0, vget_low_u16(_t0));
                    vst1_u16(outptr0 + out_hstep * 1, vget_high_u16(_t0));
                    vst1_u16(outptr0 + out_hstep * 2, vget_low_u16(_t1));
                    vst1_u16(outptr0 + out_hstep * 3, vget_high_u16(_t1));
                    vst1_u16(outptr0 + out_hstep * 4, vget_low_u16(_t2));
                    vst1_u16(outptr0 + out_hstep * 5, vget_high_u16(_t2));
                    vst1_u16(outptr0 + out_hstep * 6, vget_low_u16(_t3));
                    vst1_u16(outptr0 + out_hstep * 7, vget_high_u16(_t3));

                    outptr0 += 4;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum00);
                vst1q_f32(outptr + 4, _sum01);
                vst1q_f32(outptr + 4 * 2, _sum10);
                vst1q_f32(outptr + 4 * 3, _sum11);
                vst1q_f32(outptr + 4 * 4, _sum20);
                vst1q_f32(outptr + 4 * 5, _sum21);
                vst1q_f32(outptr + 4 * 6, _sum30);
                vst1q_f32(outptr + 4 * 7, _sum31);
            }

            outptr += 32;
#endif // NCNN_GNU_INLINE_ASM
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const unsigned short* pA = pAT;

#if NCNN_GNU_INLINE_ASM
            asm volatile(
                "cbz    %w10, 0f                    \n"

                "ld1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0]      \n"
                "b      3f                          \n"

                "0:                                 \n"
                // if pC
                "cbz    %8, 1f                      \n"

                "add    x4, %8, #16                 \n"
                "ld1    {v28.4s}, [%8]              \n"
                "ld1    {v30.4s}, [x4]              \n"
                "b      2f                          \n"

                // else
                "1:                                 \n"
                "eor    v28.16b, v28.16b, v28.16b   \n"
                "eor    v30.16b, v30.16b, v30.16b   \n"

                "2:                                 \n"
                "mov    v29.16b, v28.16b            \n"
                "mov    v31.16b, v30.16b            \n"

                "3:                                 \n"
                "lsr    w4, %w9, #2                 \n" // w4 = max_kk >> 2
                "cmp    w4, #0                      \n"
                "beq    5f                          \n"

                "4:                                 \n"
                "prfm   pldl1keep, [%2, #128]       \n"
                "ld1    {v0.4h, v1.4h}, [%2], #16   \n"
                "shll   v0.4s, v0.4h, #16           \n"
                "shll   v1.4s, v1.4h, #16           \n"
                "prfm   pldl1keep, [%1, #512]       \n"
                "ld1    {v12.8h, v13.8h, v14.8h, v15.8h}, [%1], #64 \n"
                "shll   v4.4s, v12.4h, #16          \n"
                "shll2  v5.4s, v12.8h, #16          \n"
                "shll   v6.4s, v13.4h, #16          \n"
                "shll2  v7.4s, v13.8h, #16          \n"
                "fmla   v28.4s, v4.4s, v0.s[0]      \n"
                "fmla   v29.4s, v4.4s, v0.s[1]      \n"
                "fmla   v30.4s, v5.4s, v0.s[0]      \n"
                "fmla   v31.4s, v5.4s, v0.s[1]      \n"
                "fmla   v28.4s, v6.4s, v0.s[2]      \n"
                "fmla   v29.4s, v6.4s, v0.s[3]      \n"
                "fmla   v30.4s, v7.4s, v0.s[2]      \n"
                "fmla   v31.4s, v7.4s, v0.s[3]      \n"
                "shll   v8.4s, v14.4h, #16          \n"
                "shll2  v9.4s, v14.8h, #16          \n"
                "shll   v10.4s, v15.4h, #16         \n"
                "shll2  v11.4s, v15.8h, #16         \n"
                "fmla   v28.4s, v8.4s, v1.s[0]      \n"
                "fmla   v29.4s, v8.4s, v1.s[1]      \n"
                "fmla   v30.4s, v9.4s, v1.s[0]      \n"
                "fmla   v31.4s, v9.4s, v1.s[1]      \n"
                "subs   w4, w4, #1                  \n"
                "fmla   v28.4s, v10.4s, v1.s[2]     \n"
                "fmla   v29.4s, v10.4s, v1.s[3]     \n"
                "fmla   v30.4s, v11.4s, v1.s[2]     \n"
                "fmla   v31.4s, v11.4s, v1.s[3]     \n"
                "bne    4b                          \n"

                "5:                                 \n"
                "and    w4, %w9, #3                 \n" // w4 = remain = max_kk & 3
                "cmp    w4, #0                      \n"
                "beq    7f                          \n"

                "6:                                 \n"
                "ld1    {v0.s}[0], [%2], #4         \n"
                "shll   v0.4s, v0.4h, #16           \n"
                "ld1    {v3.8h}, [%1], #16          \n"
                "shll   v4.4s, v3.4h, #16           \n"
                "shll2  v5.4s, v3.8h, #16           \n"
                "fmla   v28.4s, v4.4s, v0.s[0]      \n"
                "fmla   v29.4s, v4.4s, v0.s[1]      \n"
                "subs   w4, w4, #1                  \n"
                "fmla   v30.4s, v5.4s, v0.s[0]      \n"
                "fmla   v31.4s, v5.4s, v0.s[1]      \n"
                "bne    6b                          \n"

                "7:                                 \n"
                "shrn   v0.4h, v28.4s, #16          \n"
                "shrn2  v0.8h, v29.4s, #16          \n"
                "shrn   v1.4h, v30.4s, #16          \n"
                "shrn2  v1.8h, v31.4s, #16          \n"
                "tst    %w11, #255                  \n"
                "beq    10f                         \n"

                // if out_elempack == 4
                "cmp    %w12, #4                    \n"
                "bne    8f                          \n"

                "lsl    w4, %w13, #2                \n"
                "add    x4, %3, w4, sxtw 1          \n"
                "st1    {v0.8h}, [%3], #16          \n"
                "st1    {v1.8h}, [x4]               \n"
                "b      9f                          \n"

                // if out_elempack == 1
                "8:                                 \n"
                // transpose8x2
                "uzp1   v2.8h, v0.8h, v1.8h         \n"
                "uzp2   v3.8h, v0.8h, v1.8h         \n"
                "uzp1   v0.8h, v2.8h, v3.8h         \n"
                "uzp2   v1.8h, v2.8h, v3.8h         \n"

                "add    x4, %3, %w13, sxtw 1        \n"
                "st1    {v0.s}[0], [%3], #4         \n"
                "st1    {v0.s}[2], [x4]             \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v1.s}[0], [x4]             \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v1.s}[2], [x4]             \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v0.s}[1], [x4]             \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v0.s}[3], [x4]             \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v1.s}[1], [x4]             \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v1.s}[3], [x4]             \n"

                "9:                                 \n"
                "add    %0, %0, #64                 \n"
                "b      11f                         \n"

                "10:                                \n"
                "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0], #64 \n"

                "11:                                \n"

                : "=r"(outptr), // %0
                "=r"(pA),     // %1
                "=r"(pB),     // %2
                "=r"(outptr0) // %3
                : "0"(outptr),
                "1"(pA),
                "2"(pB),
                "3"(outptr0),
                "r"(pC),           // %8
                "r"(max_kk),       // %9
                "r"(k),            // %10
                "r"(k_end),        // %11
                "r"(out_elempack), // %12
                "r"(out_hstep)     // %13
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v28", "v29", "v30", "v31");
#else  // NCNN_GNU_INLINE_ASM
            float32x4_t _sum00;
            float32x4_t _sum01;
            float32x4_t _sum10;
            float32x4_t _sum11;

            if (k == 0)
            {
                if (pC)
                {
                    _sum00 = vld1q_f32(pC);
                    _sum01 = vld1q_f32(pC + 4);
                    _sum10 = _sum00;
                    _sum11 = _sum01;
                }
                else
                {
                    _sum00 = vdupq_n_f32(0.f);
                    _sum01 = vdupq_n_f32(0.f);
                    _sum10 = vdupq_n_f32(0.f);
                    _sum11 = vdupq_n_f32(0.f);
                }
            }
            else
            {
                _sum00 = vld1q_f32(outptr);
                _sum01 = vld1q_f32(outptr + 4 * 1);
                _sum10 = vld1q_f32(outptr + 4 * 2);
                _sum11 = vld1q_f32(outptr + 4 * 3);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA0 = bfloat2float(vld1_u16(pA));
                float32x4_t _pA1 = bfloat2float(vld1_u16(pA + 4));

                float32x2_t _pB0 = vget_low_f32(bfloat2float(vld1_u16(pB)));

                _sum00 = vfmaq_lane_f32(_sum00, _pA0, _pB0, 0);
                _sum01 = vfmaq_lane_f32(_sum01, _pA1, _pB0, 0);
                _sum10 = vfmaq_lane_f32(_sum10, _pA0, _pB0, 1);
                _sum11 = vfmaq_lane_f32(_sum11, _pA1, _pB0, 1);

                pA += 8;
                pB += 2;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1_u16(outptr0, float2bfloat(_sum00));
                    vst1_u16(outptr0 + 4, float2bfloat(_sum10));

                    vst1_u16(outptr0 + out_hstep * 4, float2bfloat(_sum01));
                    vst1_u16(outptr0 + out_hstep * 4 + 4, float2bfloat(_sum11));
                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
                    unsigned short sum0[8];
                    unsigned short sum1[8];
                    vst1_u16(sum0, float2bfloat(_sum00));
                    vst1_u16(sum0 + 4, float2bfloat(_sum01));
                    vst1_u16(sum1, float2bfloat(_sum10));
                    vst1_u16(sum1 + 4, float2bfloat(_sum11));

                    outptr0[0] = sum0[0];
                    outptr0[out_hstep] = sum0[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0[out_hstep * 4] = sum0[4];
                    outptr0[out_hstep * 5] = sum0[5];
                    outptr0[out_hstep * 6] = sum0[6];
                    outptr0[out_hstep * 7] = sum0[7];

                    outptr0[1] = sum1[0];
                    outptr0[out_hstep + 1] = sum1[1];
                    outptr0[out_hstep * 2 + 1] = sum1[2];
                    outptr0[out_hstep * 3 + 1] = sum1[3];
                    outptr0[out_hstep * 4 + 1] = sum1[4];
                    outptr0[out_hstep * 5 + 1] = sum1[5];
                    outptr0[out_hstep * 6 + 1] = sum1[6];
                    outptr0[out_hstep * 7 + 1] = sum1[7];
                    outptr0 += 2;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum00);
                vst1q_f32(outptr + 4, _sum01);
                vst1q_f32(outptr + 4 * 2, _sum10);
                vst1q_f32(outptr + 4 * 3, _sum11);
            }

            outptr += 16;
#endif // NCNN_GNU_INLINE_ASM
        }
        for (; jj < max_jj; jj += 1)
        {
            const unsigned short* pA = pAT;

#if NCNN_GNU_INLINE_ASM
            asm volatile(
                "cbz    %w10, 0f                    \n"

                "ld1    {v30.4s, v31.4s}, [%0]      \n"
                "b      3f                          \n"

                "0:                                 \n"
                // if pC
                "cbz    %8, 1f                      \n"

                "ld1    {v30.4s, v31.4s}, [%8]      \n"
                "b      2f                          \n"

                // else
                "1:                                 \n"
                "eor    v30.16b, v30.16b, v30.16b   \n"
                "eor    v31.16b, v31.16b, v31.16b   \n"

                "2:                                 \n"

                "3:                                 \n"
                "lsr    w4, %w9, #2                 \n" // w4 = max_kk >> 2
                "cmp    w4, #0                      \n"
                "beq    5f                          \n"

                "eor    v28.16b, v28.16b, v28.16b   \n"
                "eor    v29.16b, v29.16b, v29.16b   \n"
                "4:                                 \n"
                "prfm   pldl1keep, [%2, #64]        \n"
                "ld1    {v0.4h}, [%2], #8           \n"
                "shll   v0.4s, v0.4h, #16           \n"
                "prfm   pldl1keep, [%1, #512]       \n"
                "ld1    {v12.8h, v13.8h, v14.8h, v15.8h}, [%1], #64 \n"
                "shll   v4.4s, v12.4h, #16          \n"
                "shll2  v5.4s, v12.8h, #16          \n"
                "shll   v6.4s, v13.4h, #16          \n"
                "shll2  v7.4s, v13.8h, #16          \n"
                "fmla   v28.4s, v4.4s, v0.s[0]      \n"
                "fmla   v29.4s, v5.4s, v0.s[0]      \n"
                "fmla   v30.4s, v6.4s, v0.s[1]      \n"
                "fmla   v31.4s, v7.4s, v0.s[1]      \n"
                "shll   v8.4s, v14.4h, #16          \n"
                "shll2  v9.4s, v14.8h, #16          \n"
                "shll   v10.4s, v15.4h, #16         \n"
                "shll2  v11.4s, v15.8h, #16         \n"
                "fmla   v28.4s, v8.4s, v0.s[2]      \n"
                "fmla   v29.4s, v9.4s, v0.s[2]      \n"
                "subs   w4, w4, #1                  \n"
                "fmla   v30.4s, v10.4s, v0.s[3]     \n"
                "fmla   v31.4s, v11.4s, v0.s[3]     \n"
                "bne    4b                          \n"
                "fadd   v30.4s, v30.4s, v28.4s      \n"
                "fadd   v31.4s, v31.4s, v29.4s      \n"

                "5:                                 \n"
                "and    w4, %w9, #3                 \n" // w4 = remain = max_kk & 3
                "cmp    w4, #0                      \n"
                "beq    7f                          \n"

                "6:                                 \n"
                "ld1r   {v0.4h}, [%2], #2           \n"
                "shll   v0.4s, v0.4h, #16           \n"
                "ld1    {v3.8h}, [%1], #16          \n"
                "shll   v4.4s, v3.4h, #16           \n"
                "shll2  v5.4s, v3.8h, #16           \n"
                "subs   w4, w4, #1                  \n"
                "fmla   v30.4s, v4.4s, v0.4s        \n"
                "fmla   v31.4s, v5.4s, v0.4s        \n"
                "bne    6b                          \n"

                "7:                                 \n"
                "shrn   v30.4h, v30.4s, #16         \n"
                "shrn   v31.4h, v31.4s, #16         \n"
                "tst    %w11, #255                  \n"
                "beq    10f                         \n"

                // if out_elempack == 4
                "cmp    %w12, #4                    \n"
                "bne    8f                          \n"

                "lsl    w4, %w13, #2                \n"
                "add    x4, %3, w4, sxtw 1          \n"
                "st1    {v30.4h}, [%3], #8          \n"
                "st1    {v31.4h}, [x4]              \n"
                "b      9f                          \n"

                // if out_elempack == 1
                "8:                                 \n"
                "add    x4, %3, %w13, sxtw 1        \n"
                "st1    {v30.h}[0], [%3], #2        \n"
                "st1    {v30.h}[1], [x4]            \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v30.h}[2], [x4]            \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v30.h}[3], [x4]            \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v31.h}[0], [x4]            \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v31.h}[1], [x4]            \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v31.h}[2], [x4]            \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v31.h}[3], [x4]            \n"

                "9:                                 \n"
                "add    %0, %0, #32                 \n"
                "b      11f                         \n"

                "10:                                \n"
                "st1    {v30.4s, v31.4s}, [%0], #32 \n"

                "11:                                \n"

                : "=r"(outptr), // %0
                "=r"(pA),     // %1
                "=r"(pB),     // %2
                "=r"(outptr0) // %3
                : "0"(outptr),
                "1"(pA),
                "2"(pB),
                "3"(outptr0),
                "r"(pC),           // %8
                "r"(max_kk),       // %9
                "r"(k),            // %10
                "r"(k_end),        // %11
                "r"(out_elempack), // %12
                "r"(out_hstep)     // %13
                : "cc", "memory", "x4", "v0", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v28", "v29", "v30", "v31");
#else  // NCNN_GNU_INLINE_ASM
            float32x4_t _sum00;
            float32x4_t _sum01;

            if (k == 0)
            {
                if (pC)
                {
                    _sum00 = vld1q_f32(pC);
                    _sum01 = vld1q_f32(pC + 4);
                }
                else
                {
                    _sum00 = vdupq_n_f32(0.f);
                    _sum01 = vdupq_n_f32(0.f);
                }
            }
            else
            {
                _sum00 = vld1q_f32(outptr);
                _sum01 = vld1q_f32(outptr + 4 * 1);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA0 = bfloat2float(vld1_u16(pA));
                float32x4_t _pA1 = bfloat2float(vld1_u16(pA + 4));

                float32x4_t _pB = bfloat2float(vld1_dup_u16(pB));

                _sum00 = vfmaq_f32(_sum00, _pA0, _pB);
                _sum01 = vfmaq_f32(_sum01, _pA1, _pB);

                pA += 8;
                pB += 1;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1_u16(outptr0, float2bfloat(_sum00));
                    vst1_u16(outptr0 + out_hstep * 4, float2bfloat(_sum01));
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
                    unsigned short sum0[8];
                    vst1_u16(sum0, float2bfloat(_sum00));
                    vst1_u16(sum0 + 4, float2bfloat(_sum01));

                    outptr0[0] = sum0[0];
                    outptr0[out_hstep * 1] = sum0[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0[out_hstep * 4] = sum0[4];
                    outptr0[out_hstep * 5] = sum0[5];
                    outptr0[out_hstep * 6] = sum0[6];
                    outptr0[out_hstep * 7] = sum0[7];
                    outptr0++;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum00);
                vst1q_f32(outptr + 4, _sum01);
            }

            outptr += 8;
#endif // NCNN_GNU_INLINE_ASM
        }

        pAT += max_kk * 8;
    }
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        unsigned short* outptr0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const unsigned short* pB = pBT;

        if (pC)
        {
            pC = (const float*)CT_tile + i + ii;
        }

        int jj = 0;
#if __aarch64__
        for (; jj + 11 < max_jj; jj += 12)
        {
            const unsigned short* pA = pAT;

#if NCNN_GNU_INLINE_ASM
            asm volatile(
                "cbz    %w10, 0f                    \n"

                "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                "ld1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0]      \n"
                "subs   %0, %0, #128                \n"
                "b      3f                          \n"

                "0:                                 \n"
                // if pC
                "cbz    %8, 1f                      \n"

                "ld1    {v20.4s}, [%8]              \n"
                "b      2f                          \n"

                // else
                "1:                                 \n"
                "eor    v20.16b, v20.16b, v20.16b   \n"

                "2:                                 \n"
                "mov    v21.16b, v20.16b            \n"
                "mov    v22.16b, v20.16b            \n"
                "mov    v23.16b, v20.16b            \n"
                "mov    v24.16b, v20.16b            \n"
                "mov    v25.16b, v20.16b            \n"
                "mov    v26.16b, v20.16b            \n"
                "mov    v27.16b, v20.16b            \n"
                "mov    v28.16b, v20.16b            \n"
                "mov    v29.16b, v20.16b            \n"
                "mov    v30.16b, v20.16b            \n"
                "mov    v31.16b, v20.16b            \n"

                "3:                                 \n"
                "lsr    w4, %w9, #2                 \n" // w4 = max_kk >> 2
                "cmp    w4, #0                      \n"
                "beq    5f                          \n"

                "4:                                 \n"
                "prfm   pldl1keep, [%2, #512]       \n"
                "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%2], #64 \n"
                "shll   v0.4s, v4.4h, #16           \n"
                "shll2  v1.4s, v4.8h, #16           \n"
                "shll   v2.4s, v5.4h, #16           \n"
                "shll2  v3.4s, v5.8h, #16           \n"
                "prfm   pldl1keep, [%1, #256]       \n"
                "ld1    {v16.4h, v17.4h, v18.4h, v19.4h}, [%1], #32 \n"
                "shll   v16.4s, v16.4h, #16         \n"
                "shll   v17.4s, v17.4h, #16         \n"
                "shll   v18.4s, v18.4h, #16         \n"
                "shll   v19.4s, v19.4h, #16         \n"
                "fmla   v20.4s, v16.4s, v0.s[0]     \n"
                "fmla   v21.4s, v16.4s, v0.s[1]     \n"
                "fmla   v22.4s, v16.4s, v0.s[2]     \n"
                "fmla   v23.4s, v16.4s, v0.s[3]     \n"
                "fmla   v24.4s, v16.4s, v1.s[0]     \n"
                "fmla   v25.4s, v16.4s, v1.s[1]     \n"
                "fmla   v26.4s, v16.4s, v1.s[2]     \n"
                "fmla   v27.4s, v16.4s, v1.s[3]     \n"
                "fmla   v28.4s, v16.4s, v2.s[0]     \n"
                "fmla   v29.4s, v16.4s, v2.s[1]     \n"
                "fmla   v30.4s, v16.4s, v2.s[2]     \n"
                "fmla   v31.4s, v16.4s, v2.s[3]     \n"
                "shll   v4.4s, v6.4h, #16           \n"
                "shll2  v5.4s, v6.8h, #16           \n"
                "shll   v6.4s, v7.4h, #16           \n"
                "shll2  v7.4s, v7.8h, #16           \n"
                "fmla   v20.4s, v17.4s, v3.s[0]     \n"
                "fmla   v21.4s, v17.4s, v3.s[1]     \n"
                "fmla   v22.4s, v17.4s, v3.s[2]     \n"
                "fmla   v23.4s, v17.4s, v3.s[3]     \n"
                "fmla   v24.4s, v17.4s, v4.s[0]     \n"
                "fmla   v25.4s, v17.4s, v4.s[1]     \n"
                "fmla   v26.4s, v17.4s, v4.s[2]     \n"
                "fmla   v27.4s, v17.4s, v4.s[3]     \n"
                "fmla   v28.4s, v17.4s, v5.s[0]     \n"
                "fmla   v29.4s, v17.4s, v5.s[1]     \n"
                "fmla   v30.4s, v17.4s, v5.s[2]     \n"
                "fmla   v31.4s, v17.4s, v5.s[3]     \n"
                "prfm   pldl1keep, [%2, #256]       \n"
                "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%2], #32 \n"
                "shll   v0.4s, v0.4h, #16           \n"
                "shll   v1.4s, v1.4h, #16           \n"
                "shll   v2.4s, v2.4h, #16           \n"
                "shll   v3.4s, v3.4h, #16           \n"
                "fmla   v20.4s, v18.4s, v6.s[0]     \n"
                "fmla   v21.4s, v18.4s, v6.s[1]     \n"
                "fmla   v22.4s, v18.4s, v6.s[2]     \n"
                "fmla   v23.4s, v18.4s, v6.s[3]     \n"
                "fmla   v24.4s, v18.4s, v7.s[0]     \n"
                "fmla   v25.4s, v18.4s, v7.s[1]     \n"
                "fmla   v26.4s, v18.4s, v7.s[2]     \n"
                "fmla   v27.4s, v18.4s, v7.s[3]     \n"
                "fmla   v28.4s, v18.4s, v0.s[0]     \n"
                "fmla   v29.4s, v18.4s, v0.s[1]     \n"
                "fmla   v30.4s, v18.4s, v0.s[2]     \n"
                "fmla   v31.4s, v18.4s, v0.s[3]     \n"
                "subs   w4, w4, #1                  \n"
                "fmla   v20.4s, v19.4s, v1.s[0]     \n"
                "fmla   v21.4s, v19.4s, v1.s[1]     \n"
                "fmla   v22.4s, v19.4s, v1.s[2]     \n"
                "fmla   v23.4s, v19.4s, v1.s[3]     \n"
                "fmla   v24.4s, v19.4s, v2.s[0]     \n"
                "fmla   v25.4s, v19.4s, v2.s[1]     \n"
                "fmla   v26.4s, v19.4s, v2.s[2]     \n"
                "fmla   v27.4s, v19.4s, v2.s[3]     \n"
                "fmla   v28.4s, v19.4s, v3.s[0]     \n"
                "fmla   v29.4s, v19.4s, v3.s[1]     \n"
                "fmla   v30.4s, v19.4s, v3.s[2]     \n"
                "fmla   v31.4s, v19.4s, v3.s[3]     \n"
                "bne    4b                          \n"

                "5:                                 \n"
                "and    w4, %w9, #3                 \n" // w4 = remain = max_kk & 3
                "cmp    w4, #0                      \n"
                "beq    7f                          \n"

                "6:                                 \n"
                "ld1    {v0.4h, v1.4h, v2.4h}, [%2], #24 \n"
                "shll   v0.4s, v0.4h, #16           \n"
                "shll   v1.4s, v1.4h, #16           \n"
                "shll   v2.4s, v2.4h, #16           \n"
                "ld1    {v16.4h}, [%1], #8          \n"
                "shll   v16.4s, v16.4h, #16         \n"
                "fmla   v20.4s, v16.4s, v0.s[0]     \n"
                "fmla   v21.4s, v16.4s, v0.s[1]     \n"
                "fmla   v22.4s, v16.4s, v0.s[2]     \n"
                "fmla   v23.4s, v16.4s, v0.s[3]     \n"
                "fmla   v24.4s, v16.4s, v1.s[0]     \n"
                "fmla   v25.4s, v16.4s, v1.s[1]     \n"
                "fmla   v26.4s, v16.4s, v1.s[2]     \n"
                "fmla   v27.4s, v16.4s, v1.s[3]     \n"
                "subs   w4, w4, #1                  \n"
                "fmla   v28.4s, v16.4s, v2.s[0]     \n"
                "fmla   v29.4s, v16.4s, v2.s[1]     \n"
                "fmla   v30.4s, v16.4s, v2.s[2]     \n"
                "fmla   v31.4s, v16.4s, v2.s[3]     \n"
                "bne    6b                          \n"

                "7:                                 \n"
                "shrn   v0.4h, v20.4s, #16          \n"
                "shrn2  v0.8h, v21.4s, #16          \n"
                "shrn   v1.4h, v22.4s, #16          \n"
                "shrn2  v1.8h, v23.4s, #16          \n"
                "shrn   v2.4h, v24.4s, #16          \n"
                "shrn2  v2.8h, v25.4s, #16          \n"
                "shrn   v3.4h, v26.4s, #16          \n"
                "shrn2  v3.8h, v27.4s, #16          \n"
                "shrn   v4.4h, v28.4s, #16          \n"
                "shrn2  v4.8h, v29.4s, #16          \n"
                "shrn   v5.4h, v30.4s, #16          \n"
                "shrn2  v5.8h, v31.4s, #16          \n"
                "tst    %w11, #255                  \n"
                "beq    10f                         \n"

                // if out_elempack == 4
                "cmp    %w12, #4                    \n"
                "bne    8f                          \n"

                "st1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%3], #64 \n"
                "st1    {v4.8h, v5.8h}, [%3], #32   \n"
                "b      9f                          \n"

                // if out_elempack == 1
                "8:                                 \n"
                // transpose4x12
                "uzp1   v20.8h, v0.8h, v1.8h        \n"
                "uzp2   v21.8h, v0.8h, v1.8h        \n"
                "uzp1   v22.8h, v2.8h, v3.8h        \n"
                "uzp2   v23.8h, v2.8h, v3.8h        \n"
                "uzp1   v24.8h, v4.8h, v5.8h        \n"
                "uzp2   v25.8h, v4.8h, v5.8h        \n"

                "uzp1   v0.8h, v20.8h, v21.8h       \n"
                "uzp2   v6.8h, v20.8h, v21.8h       \n"
                "uzp1   v1.8h, v22.8h, v23.8h       \n"
                "uzp2   v7.8h, v22.8h, v23.8h       \n"
                "uzp1   v2.8h, v24.8h, v25.8h       \n"
                "uzp2   v8.8h, v24.8h, v25.8h       \n"

                "mov    v3.d[0], v0.d[1]            \n"
                "mov    v4.d[0], v1.d[1]            \n"
                "mov    v5.d[0], v2.d[1]            \n"
                "mov    v9.d[0], v6.d[1]            \n"
                "mov    v10.d[0], v7.d[1]           \n"
                "mov    v11.d[0], v8.d[1]           \n"

                "add    x4, %3, %w13, sxtw 1        \n"
                "st1    {v0.4h, v1.4h, v2.4h}, [%3], #24 \n"
                "st1    {v3.4h, v4.4h, v5.4h}, [x4] \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v6.4h, v7.4h, v8.4h}, [x4] \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v9.4h, v10.4h, v11.4h}, [x4] \n"

                "9:                                 \n"
                "add    %0, %0, #192                \n"
                "b      11f                         \n"

                "10:                                \n"
                "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0], #64 \n"

                "11:                                \n"

                : "=r"(outptr), // %0
                "=r"(pA),     // %1
                "=r"(pB),     // %2
                "=r"(outptr0) // %3
                : "0"(outptr),
                "1"(pA),
                "2"(pB),
                "3"(outptr0),
                "r"(pC),           // %8
                "r"(max_kk),       // %9
                "r"(k),            // %10
                "r"(k_end),        // %11
                "r"(out_elempack), // %12
                "r"(out_hstep)     // %13
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
#else  // NCNN_GNU_INLINE_ASM
            float32x4_t _sum0;
            float32x4_t _sum1;
            float32x4_t _sum2;
            float32x4_t _sum3;
            float32x4_t _sum4;
            float32x4_t _sum5;
            float32x4_t _sum6;
            float32x4_t _sum7;
            float32x4_t _sum8;
            float32x4_t _sum9;
            float32x4_t _suma;
            float32x4_t _sumb;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = vld1q_f32(pC);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
                    _sum4 = _sum0;
                    _sum5 = _sum0;
                    _sum6 = _sum0;
                    _sum7 = _sum0;
                    _sum8 = _sum0;
                    _sum9 = _sum0;
                    _suma = _sum0;
                    _sumb = _sum0;
                }
                else
                {
                    _sum0 = vdupq_n_f32(0.f);
                    _sum1 = vdupq_n_f32(0.f);
                    _sum2 = vdupq_n_f32(0.f);
                    _sum3 = vdupq_n_f32(0.f);
                    _sum4 = vdupq_n_f32(0.f);
                    _sum5 = vdupq_n_f32(0.f);
                    _sum6 = vdupq_n_f32(0.f);
                    _sum7 = vdupq_n_f32(0.f);
                    _sum8 = vdupq_n_f32(0.f);
                    _sum9 = vdupq_n_f32(0.f);
                    _suma = vdupq_n_f32(0.f);
                    _sumb = vdupq_n_f32(0.f);
                }
            }
            else
            {
                _sum0 = vld1q_f32(outptr);
                _sum1 = vld1q_f32(outptr + 4 * 1);
                _sum2 = vld1q_f32(outptr + 4 * 2);
                _sum3 = vld1q_f32(outptr + 4 * 3);
                _sum4 = vld1q_f32(outptr + 4 * 4);
                _sum5 = vld1q_f32(outptr + 4 * 5);
                _sum6 = vld1q_f32(outptr + 4 * 6);
                _sum7 = vld1q_f32(outptr + 4 * 7);
                _sum8 = vld1q_f32(outptr + 4 * 8);
                _sum9 = vld1q_f32(outptr + 4 * 9);
                _suma = vld1q_f32(outptr + 4 * 10);
                _sumb = vld1q_f32(outptr + 4 * 11);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA = bfloat2float(vld1_u16(pA));
                float32x4_t _pB0 = bfloat2float(vld1_u16(pB));
                float32x4_t _pB1 = bfloat2float(vld1_u16(pB + 4));
                float32x4_t _pB2 = bfloat2float(vld1_u16(pB + 8));

                _sum0 = vfmaq_laneq_f32(_sum0, _pA, _pB0, 0);
                _sum1 = vfmaq_laneq_f32(_sum1, _pA, _pB0, 1);
                _sum2 = vfmaq_laneq_f32(_sum2, _pA, _pB0, 2);
                _sum3 = vfmaq_laneq_f32(_sum3, _pA, _pB0, 3);
                _sum4 = vfmaq_laneq_f32(_sum4, _pA, _pB1, 0);
                _sum5 = vfmaq_laneq_f32(_sum5, _pA, _pB1, 1);
                _sum6 = vfmaq_laneq_f32(_sum6, _pA, _pB1, 2);
                _sum7 = vfmaq_laneq_f32(_sum7, _pA, _pB1, 3);
                _sum8 = vfmaq_laneq_f32(_sum8, _pA, _pB2, 0);
                _sum9 = vfmaq_laneq_f32(_sum9, _pA, _pB2, 1);
                _suma = vfmaq_laneq_f32(_suma, _pA, _pB2, 2);
                _sumb = vfmaq_laneq_f32(_sumb, _pA, _pB2, 3);

                pA += 4;
                pB += 12;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1_u16(outptr0, float2bfloat(_sum0));
                    vst1_u16(outptr0 + 4, float2bfloat(_sum1));
                    vst1_u16(outptr0 + 4 * 2, float2bfloat(_sum2));
                    vst1_u16(outptr0 + 4 * 3, float2bfloat(_sum3));
                    vst1_u16(outptr0 + 4 * 4, float2bfloat(_sum4));
                    vst1_u16(outptr0 + 4 * 5, float2bfloat(_sum5));
                    vst1_u16(outptr0 + 4 * 6, float2bfloat(_sum6));
                    vst1_u16(outptr0 + 4 * 7, float2bfloat(_sum7));
                    vst1_u16(outptr0 + 4 * 8, float2bfloat(_sum8));
                    vst1_u16(outptr0 + 4 * 9, float2bfloat(_sum9));
                    vst1_u16(outptr0 + 4 * 10, float2bfloat(_suma));
                    vst1_u16(outptr0 + 4 * 11, float2bfloat(_sumb));
                    outptr0 += 48;
                }
                if (out_elempack == 1)
                {
                    uint16x4_t _t0 = float2bfloat(_sum0);
                    uint16x4_t _t1 = float2bfloat(_sum1);
                    uint16x4_t _t2 = float2bfloat(_sum2);
                    uint16x4_t _t3 = float2bfloat(_sum3);
                    uint16x4_t _t4 = float2bfloat(_sum4);
                    uint16x4_t _t5 = float2bfloat(_sum5);
                    uint16x4_t _t6 = float2bfloat(_sum6);
                    uint16x4_t _t7 = float2bfloat(_sum7);
                    uint16x4_t _t8 = float2bfloat(_sum8);
                    uint16x4_t _t9 = float2bfloat(_sum9);
                    uint16x4_t _ta = float2bfloat(_suma);
                    uint16x4_t _tb = float2bfloat(_sumb);
                    transpose4x12_u16(_t0, _t1, _t2, _t3, _t4, _t5, _t6, _t7, _t8, _t9, _ta, _tb);

                    vst1_u16(outptr0, _t0);
                    vst1_u16(outptr0 + 4, _t1);
                    vst1_u16(outptr0 + 8, _t2);
                    vst1_u16(outptr0 + out_hstep, _t3);
                    vst1_u16(outptr0 + out_hstep + 4, _t4);
                    vst1_u16(outptr0 + out_hstep + 8, _t5);
                    vst1_u16(outptr0 + out_hstep * 2, _t6);
                    vst1_u16(outptr0 + out_hstep * 2 + 4, _t7);
                    vst1_u16(outptr0 + out_hstep * 2 + 8, _t8);
                    vst1_u16(outptr0 + out_hstep * 3, _t9);
                    vst1_u16(outptr0 + out_hstep * 3 + 4, _ta);
                    vst1_u16(outptr0 + out_hstep * 3 + 8, _tb);
                    outptr0 += 12;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum0);
                vst1q_f32(outptr + 4, _sum1);
                vst1q_f32(outptr + 4 * 2, _sum2);
                vst1q_f32(outptr + 4 * 3, _sum3);
                vst1q_f32(outptr + 4 * 4, _sum4);
                vst1q_f32(outptr + 4 * 5, _sum5);
                vst1q_f32(outptr + 4 * 6, _sum6);
                vst1q_f32(outptr + 4 * 7, _sum7);
                vst1q_f32(outptr + 4 * 8, _sum8);
                vst1q_f32(outptr + 4 * 9, _sum9);
                vst1q_f32(outptr + 4 * 10, _suma);
                vst1q_f32(outptr + 4 * 11, _sumb);
            }

            outptr += 48;
#endif // NCNN_GNU_INLINE_ASM
        }
#endif // __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            const unsigned short* pA = pAT;

#if NCNN_GNU_INLINE_ASM
#if __aarch64__
            asm volatile(
                "cbz    %w10, 0f                    \n"

                "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                "ld1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0]      \n"
                "subs   %0, %0, #64                 \n"
                "b      3f                          \n"

                "0:                                 \n"
                // if pC
                "cbz    %8, 1f                      \n"

                "ld1    {v24.4s}, [%8]              \n"
                "b      2f                          \n"

                // else
                "1:                                 \n"
                "eor    v24.16b, v24.16b, v24.16b   \n"

                "2:                                 \n"
                "mov    v25.16b, v24.16b            \n"
                "mov    v26.16b, v24.16b            \n"
                "mov    v27.16b, v24.16b            \n"
                "mov    v28.16b, v24.16b            \n"
                "mov    v29.16b, v24.16b            \n"
                "mov    v30.16b, v24.16b            \n"
                "mov    v31.16b, v24.16b            \n"

                "3:                                 \n"
                "lsr    w4, %w9, #2                 \n" // w4 = max_kk >> 2
                "cmp    w4, #0                      \n"
                "beq    5f                          \n"

                "4:                                 \n"
                "prfm   pldl1keep, [%2, #512]       \n"
                "ld1    {v4.8h, v5.8h, v6.8h, v7.8h}, [%2], #64 \n"
                "shll   v0.4s, v4.4h, #16           \n"
                "shll2  v1.4s, v4.8h, #16           \n"
                "shll   v2.4s, v5.4h, #16           \n"
                "shll2  v3.4s, v5.8h, #16           \n"
                "prfm   pldl1keep, [%1, #256]       \n"
                "ld1    {v16.4h, v17.4h, v18.4h, v19.4h}, [%1], #32 \n"
                "shll   v16.4s, v16.4h, #16         \n"
                "shll   v17.4s, v17.4h, #16         \n"
                "shll   v18.4s, v18.4h, #16         \n"
                "shll   v19.4s, v19.4h, #16         \n"
                "fmla   v24.4s, v16.4s, v0.s[0]     \n"
                "fmla   v25.4s, v16.4s, v0.s[1]     \n"
                "fmla   v26.4s, v16.4s, v0.s[2]     \n"
                "fmla   v27.4s, v16.4s, v0.s[3]     \n"
                "fmla   v28.4s, v16.4s, v1.s[0]     \n"
                "fmla   v29.4s, v16.4s, v1.s[1]     \n"
                "fmla   v30.4s, v16.4s, v1.s[2]     \n"
                "fmla   v31.4s, v16.4s, v1.s[3]     \n"
                "fmla   v24.4s, v17.4s, v2.s[0]     \n"
                "fmla   v25.4s, v17.4s, v2.s[1]     \n"
                "fmla   v26.4s, v17.4s, v2.s[2]     \n"
                "fmla   v27.4s, v17.4s, v2.s[3]     \n"
                "fmla   v28.4s, v17.4s, v3.s[0]     \n"
                "fmla   v29.4s, v17.4s, v3.s[1]     \n"
                "fmla   v30.4s, v17.4s, v3.s[2]     \n"
                "fmla   v31.4s, v17.4s, v3.s[3]     \n"
                "shll   v4.4s, v6.4h, #16           \n"
                "shll2  v5.4s, v6.8h, #16           \n"
                "shll   v6.4s, v7.4h, #16           \n"
                "shll2  v7.4s, v7.8h, #16           \n"
                "fmla   v24.4s, v18.4s, v4.s[0]     \n"
                "fmla   v25.4s, v18.4s, v4.s[1]     \n"
                "fmla   v26.4s, v18.4s, v4.s[2]     \n"
                "fmla   v27.4s, v18.4s, v4.s[3]     \n"
                "fmla   v28.4s, v18.4s, v5.s[0]     \n"
                "fmla   v29.4s, v18.4s, v5.s[1]     \n"
                "fmla   v30.4s, v18.4s, v5.s[2]     \n"
                "fmla   v31.4s, v18.4s, v5.s[3]     \n"
                "subs   w4, w4, #1                  \n"
                "fmla   v24.4s, v19.4s, v6.s[0]     \n"
                "fmla   v25.4s, v19.4s, v6.s[1]     \n"
                "fmla   v26.4s, v19.4s, v6.s[2]     \n"
                "fmla   v27.4s, v19.4s, v6.s[3]     \n"
                "fmla   v28.4s, v19.4s, v7.s[0]     \n"
                "fmla   v29.4s, v19.4s, v7.s[1]     \n"
                "fmla   v30.4s, v19.4s, v7.s[2]     \n"
                "fmla   v31.4s, v19.4s, v7.s[3]     \n"
                "bne    4b                          \n"

                "5:                                 \n"
                "and    w4, %w9, #3                 \n" // w4 = remain = max_kk & 3
                "cmp    w4, #0                      \n"
                "beq    7f                          \n"

                "6:                                 \n"
                "ld1    {v0.4h, v1.4h}, [%2], #16   \n"
                "shll   v0.4s, v0.4h, #16           \n"
                "shll   v1.4s, v1.4h, #16           \n"
                "ld1    {v16.4h}, [%1], #8          \n"
                "shll   v16.4s, v16.4h, #16         \n"
                "fmla   v24.4s, v16.4s, v0.s[0]     \n"
                "fmla   v25.4s, v16.4s, v0.s[1]     \n"
                "fmla   v26.4s, v16.4s, v0.s[2]     \n"
                "fmla   v27.4s, v16.4s, v0.s[3]     \n"
                "subs   w4, w4, #1                  \n"
                "fmla   v28.4s, v16.4s, v1.s[0]     \n"
                "fmla   v29.4s, v16.4s, v1.s[1]     \n"
                "fmla   v30.4s, v16.4s, v1.s[2]     \n"
                "fmla   v31.4s, v16.4s, v1.s[3]     \n"
                "bne    6b                          \n"

                "7:                                 \n"
                "shrn   v0.4h, v24.4s, #16          \n"
                "shrn2  v0.8h, v25.4s, #16          \n"
                "shrn   v1.4h, v26.4s, #16          \n"
                "shrn2  v1.8h, v27.4s, #16          \n"
                "shrn   v2.4h, v28.4s, #16          \n"
                "shrn2  v2.8h, v29.4s, #16          \n"
                "shrn   v3.4h, v30.4s, #16          \n"
                "shrn2  v3.8h, v31.4s, #16          \n"
                "tst    %w11, #255                  \n"
                "beq    10f                         \n"

                // if out_elempack == 4
                "cmp    %w12, #4                    \n"
                "bne    8f                          \n"

                "st1    {v0.8h, v1.8h, v2.8h, v3.8h}, [%3], #64 \n"
                "b      9f                          \n"

                // if out_elempack == 1
                "8:                                 \n"
                // transpose4x8
                "uzp1   v20.8h, v0.8h, v1.8h        \n"
                "uzp2   v21.8h, v0.8h, v1.8h        \n"
                "uzp1   v22.8h, v2.8h, v3.8h        \n"
                "uzp2   v23.8h, v2.8h, v3.8h        \n"

                "uzp1   v0.8h, v20.8h, v22.8h       \n"
                "uzp2   v2.8h, v20.8h, v22.8h       \n"
                "uzp1   v1.8h, v21.8h, v23.8h       \n"
                "uzp2   v3.8h, v21.8h, v23.8h       \n"

                "add    x4, %3, %w13, sxtw 1        \n"
                "st1    {v0.8h}, [%3], #16          \n"
                "st1    {v1.8h}, [x4]               \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v2.8h}, [x4]               \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v3.8h}, [x4]               \n"

                "9:                                 \n"
                "add    %0, %0, #128                \n"
                "b      11f                         \n"

                "10:                                \n"
                "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0], #64 \n"

                "11:                                \n"

                : "=r"(outptr), // %0
                "=r"(pA),     // %1
                "=r"(pB),     // %2
                "=r"(outptr0) // %3
                : "0"(outptr),
                "1"(pA),
                "2"(pB),
                "3"(outptr0),
                "r"(pC),           // %8
                "r"(max_kk),       // %9
                "r"(k),            // %10
                "r"(k_end),        // %11
                "r"(out_elempack), // %12
                "r"(out_hstep)     // %13
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
#else  // __aarch64__
            asm volatile(
                "cmp        %10, #0             \n"
                "beq        0f                  \n"

                "vldm       %0!, {d16-d23}      \n"
                "vldm       %0, {d24-d31}       \n"
                "sub        %0, %0, #64         \n"
                "b          3f                  \n"

                "0:                             \n"
                // if pC
                "cmp        %8, #0              \n"
                "beq        1f                  \n"

                "vld1.f32   {d16-d17}, [%8]     \n"
                "b          2f                  \n"

                // else
                "1:                             \n"
                "veor       q8, q8              \n"

                "2:                             \n"
                "vmov       q9, q8              \n"
                "vmov       q10, q8             \n"
                "vmov       q11, q8             \n"
                "vmov       q12, q8             \n"
                "vmov       q13, q8             \n"
                "vmov       q14, q8             \n"
                "vmov       q15, q8             \n"

                "3:                             \n"
                "lsr        r4, %9, #2          \n" // r4 = max_kk >> 2
                "cmp        r4, #0              \n"
                "beq        5f                  \n"

                "4:                             \n"
                "pld        [%2, #256]          \n"
                "vld1.u16   {d4-d7}, [%2 :64]!  \n"
                "pld        [%1, #256]          \n"
                "vld1.u16   {d12-d15}, [%1 :64]! \n"
                "vshll.u16  q0, d4, #16         \n"
                "vshll.u16  q1, d5, #16         \n"
                "vshll.u16  q2, d6, #16         \n"
                "vshll.u16  q3, d7, #16         \n"
                "vshll.u16  q4, d12, #16        \n"
                "vshll.u16  q5, d13, #16        \n"
                "vshll.u16  q6, d14, #16        \n"
                "vshll.u16  q7, d15, #16        \n"
                "vmla.f32   q8, q4, d0[0]       \n"
                "vmla.f32   q9, q4, d0[1]       \n"
                "vmla.f32   q10, q4, d1[0]      \n"
                "vmla.f32   q11, q4, d1[1]      \n"
                "vmla.f32   q12, q4, d2[0]      \n"
                "vmla.f32   q13, q4, d2[1]      \n"
                "vmla.f32   q14, q4, d3[0]      \n"
                "vmla.f32   q15, q4, d3[1]      \n"
                "vmla.f32   q8, q5, d4[0]       \n"
                "vmla.f32   q9, q5, d4[1]       \n"
                "vmla.f32   q10, q5, d5[0]      \n"
                "vmla.f32   q11, q5, d5[1]      \n"
                "vmla.f32   q12, q5, d6[0]      \n"
                "vmla.f32   q13, q5, d6[1]      \n"
                "vmla.f32   q14, q5, d7[0]      \n"
                "vmla.f32   q15, q5, d7[1]      \n"
                "pld        [%2, #256]          \n"
                "vld1.u16   {d4-d7}, [%2 :64]!  \n"
                "vshll.u16  q0, d4, #16         \n"
                "vshll.u16  q1, d5, #16         \n"
                "vshll.u16  q2, d6, #16         \n"
                "vshll.u16  q3, d7, #16         \n"
                "vmla.f32   q8, q6, d0[0]       \n"
                "vmla.f32   q9, q6, d0[1]       \n"
                "vmla.f32   q10, q6, d1[0]      \n"
                "vmla.f32   q11, q6, d1[1]      \n"
                "vmla.f32   q12, q6, d2[0]      \n"
                "vmla.f32   q13, q6, d2[1]      \n"
                "vmla.f32   q14, q6, d3[0]      \n"
                "vmla.f32   q15, q6, d3[1]      \n"
                "subs       r4, r4, #1          \n"
                "vmla.f32   q8, q7, d4[0]       \n"
                "vmla.f32   q9, q7, d4[1]       \n"
                "vmla.f32   q10, q7, d5[0]      \n"
                "vmla.f32   q11, q7, d5[1]      \n"
                "vmla.f32   q12, q7, d6[0]      \n"
                "vmla.f32   q13, q7, d6[1]      \n"
                "vmla.f32   q14, q7, d7[0]      \n"
                "vmla.f32   q15, q7, d7[1]      \n"
                "bne        4b                  \n"

                "5:                             \n"
                "and        r4, %9, #3          \n" // r4 = remain = max_kk & 3
                "cmp        r4, #0              \n"
                "beq        7f                  \n"

                "6:                             \n"
                "vld1.u16   {d2-d3}, [%2 :64]!  \n"
                "vshll.u16  q0, d2, #16         \n"
                "vshll.u16  q1, d3, #16         \n"
                "vld1.u16   {d9}, [%1 :64]!     \n"
                "vshll.u16  q4, d9, #16         \n"
                "vmla.f32   q8, q4, d0[0]       \n"
                "vmla.f32   q9, q4, d0[1]       \n"
                "vmla.f32   q10, q4, d1[0]      \n"
                "vmla.f32   q11, q4, d1[1]      \n"
                "subs       r4, r4, #1          \n"
                "vmla.f32   q12, q4, d2[0]      \n"
                "vmla.f32   q13, q4, d2[1]      \n"
                "vmla.f32   q14, q4, d3[0]      \n"
                "vmla.f32   q15, q4, d3[1]      \n"
                "bne        6b                  \n"

                "7:                             \n"
                "vshrn.u32  d16, q8, #16        \n"
                "vshrn.u32  d17, q9, #16        \n"
                "vshrn.u32  d18, q10, #16       \n"
                "vshrn.u32  d19, q11, #16       \n"
                "vshrn.u32  d20, q12, #16       \n"
                "vshrn.u32  d21, q13, #16       \n"
                "vshrn.u32  d22, q14, #16       \n"
                "vshrn.u32  d23, q15, #16       \n"
                "cmp        %11, #0             \n"
                "beq        10f                 \n"

                // if out_elempack == 4
                "cmp        %12, #4             \n"
                "bne        8f                  \n"

                "vstm       %3!, {d16-d23}      \n"
                "b          9f                  \n"

                // if out_elempack == 1
                "8:                             \n"
                // transpose4x8
                "vuzp.16    q8, q9              \n"
                "vuzp.16    q10, q11            \n"
                "vuzp.16    q8, q10             \n"
                "vuzp.16    q9, q11             \n"

                "add        r4, %3, %13, lsl #1 \n"
                "vst1.u16   {d16-d17}, [%3 :64]! \n"
                "vst1.u16   {d18-d19}, [r4 :64] \n"
                "add        r4, r4, %13, lsl #1 \n"
                "vst1.u16   {d20-d21}, [r4 :64] \n"
                "add        r4, r4, %13, lsl #1 \n"
                "vst1.u16   {d22-d23}, [r4 :64] \n"

                "9:                             \n"
                "add        %0, %0, #128        \n"
                "b          11f                 \n"

                "10:                            \n"
                "vstm       %0!, {d16-d23}      \n"
                "vstm       %0!, {d24-d31}      \n"

                "11:                            \n"

                : "=r"(outptr), // %0
                "=r"(pA),     // %1
                "=r"(pB),     // %2
                "=r"(outptr0) // %3
                : "0"(outptr),
                "1"(pA),
                "2"(pB),
                "3"(outptr0),
                "r"(pC),           // %8
                "r"(max_kk),       // %9
                "r"(k),            // %10
                "r"(k_end),        // %11
                "r"(out_elempack), // %12
                "r"(out_hstep)     // %13
                : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif // __aarch64__
#else  // NCNN_GNU_INLINE_ASM
            float32x4_t _sum0;
            float32x4_t _sum1;
            float32x4_t _sum2;
            float32x4_t _sum3;
            float32x4_t _sum4;
            float32x4_t _sum5;
            float32x4_t _sum6;
            float32x4_t _sum7;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = vld1q_f32(pC);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
                    _sum4 = _sum0;
                    _sum5 = _sum0;
                    _sum6 = _sum0;
                    _sum7 = _sum0;
                }
                else
                {
                    _sum0 = vdupq_n_f32(0.f);
                    _sum1 = vdupq_n_f32(0.f);
                    _sum2 = vdupq_n_f32(0.f);
                    _sum3 = vdupq_n_f32(0.f);
                    _sum4 = vdupq_n_f32(0.f);
                    _sum5 = vdupq_n_f32(0.f);
                    _sum6 = vdupq_n_f32(0.f);
                    _sum7 = vdupq_n_f32(0.f);
                }
            }
            else
            {
                _sum0 = vld1q_f32(outptr);
                _sum1 = vld1q_f32(outptr + 4 * 1);
                _sum2 = vld1q_f32(outptr + 4 * 2);
                _sum3 = vld1q_f32(outptr + 4 * 3);
                _sum4 = vld1q_f32(outptr + 4 * 4);
                _sum5 = vld1q_f32(outptr + 4 * 5);
                _sum6 = vld1q_f32(outptr + 4 * 6);
                _sum7 = vld1q_f32(outptr + 4 * 7);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA = bfloat2float(vld1_u16(pA));
                float32x4_t _pB0 = bfloat2float(vld1_u16(pB));
                float32x4_t _pB1 = bfloat2float(vld1_u16(pB + 4));

#if __aarch64__
                _sum0 = vfmaq_laneq_f32(_sum0, _pA, _pB0, 0);
                _sum1 = vfmaq_laneq_f32(_sum1, _pA, _pB0, 1);
                _sum2 = vfmaq_laneq_f32(_sum2, _pA, _pB0, 2);
                _sum3 = vfmaq_laneq_f32(_sum3, _pA, _pB0, 3);
                _sum4 = vfmaq_laneq_f32(_sum4, _pA, _pB1, 0);
                _sum5 = vfmaq_laneq_f32(_sum5, _pA, _pB1, 1);
                _sum6 = vfmaq_laneq_f32(_sum6, _pA, _pB1, 2);
                _sum7 = vfmaq_laneq_f32(_sum7, _pA, _pB1, 3);
#else
                _sum0 = vmlaq_lane_f32(_sum0, _pA, vget_low_f32(_pB0), 0);
                _sum1 = vmlaq_lane_f32(_sum1, _pA, vget_low_f32(_pB0), 1);
                _sum2 = vmlaq_lane_f32(_sum2, _pA, vget_high_f32(_pB0), 0);
                _sum3 = vmlaq_lane_f32(_sum3, _pA, vget_high_f32(_pB0), 1);
                _sum4 = vmlaq_lane_f32(_sum4, _pA, vget_low_f32(_pB1), 0);
                _sum5 = vmlaq_lane_f32(_sum5, _pA, vget_low_f32(_pB1), 1);
                _sum6 = vmlaq_lane_f32(_sum6, _pA, vget_high_f32(_pB1), 0);
                _sum7 = vmlaq_lane_f32(_sum7, _pA, vget_high_f32(_pB1), 1);
#endif

                pA += 4;
                pB += 8;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1_u16(outptr0, float2bfloat(_sum0));
                    vst1_u16(outptr0 + 4, float2bfloat(_sum1));
                    vst1_u16(outptr0 + 4 * 2, float2bfloat(_sum2));
                    vst1_u16(outptr0 + 4 * 3, float2bfloat(_sum3));
                    vst1_u16(outptr0 + 4 * 4, float2bfloat(_sum4));
                    vst1_u16(outptr0 + 4 * 5, float2bfloat(_sum5));
                    vst1_u16(outptr0 + 4 * 6, float2bfloat(_sum6));
                    vst1_u16(outptr0 + 4 * 7, float2bfloat(_sum7));
                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    uint16x4_t _t0 = float2bfloat(_sum0);
                    uint16x4_t _t1 = float2bfloat(_sum1);
                    uint16x4_t _t2 = float2bfloat(_sum2);
                    uint16x4_t _t3 = float2bfloat(_sum3);
                    uint16x4_t _t4 = float2bfloat(_sum4);
                    uint16x4_t _t5 = float2bfloat(_sum5);
                    uint16x4_t _t6 = float2bfloat(_sum6);
                    uint16x4_t _t7 = float2bfloat(_sum7);
                    transpose4x8_u16(_t0, _t1, _t2, _t3, _t4, _t5, _t6, _t7);

                    vst1_u16(outptr0, _t0);
                    vst1_u16(outptr0 + 4, _t1);
                    vst1_u16(outptr0 + out_hstep, _t2);
                    vst1_u16(outptr0 + out_hstep + 4, _t3);
                    vst1_u16(outptr0 + out_hstep * 2, _t4);
                    vst1_u16(outptr0 + out_hstep * 2 + 4, _t5);
                    vst1_u16(outptr0 + out_hstep * 3, _t6);
                    vst1_u16(outptr0 + out_hstep * 3 + 4, _t7);
                    outptr0 += 8;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum0);
                vst1q_f32(outptr + 4, _sum1);
                vst1q_f32(outptr + 4 * 2, _sum2);
                vst1q_f32(outptr + 4 * 3, _sum3);
                vst1q_f32(outptr + 4 * 4, _sum4);
                vst1q_f32(outptr + 4 * 5, _sum5);
                vst1q_f32(outptr + 4 * 6, _sum6);
                vst1q_f32(outptr + 4 * 7, _sum7);
            }

            outptr += 32;
#endif // NCNN_GNU_INLINE_ASM
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            const unsigned short* pA = pAT;

#if NCNN_GNU_INLINE_ASM
#if __aarch64__
            asm volatile(
                "cbz    %w10, 0f                    \n"

                "ld1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0]      \n"
                "b      3f                          \n"

                "0:                                 \n"
                // if pC
                "cbz    %8, 1f                      \n"

                "ld1    {v28.4s}, [%8]              \n"
                "b      2f                          \n"

                // else
                "1:                                 \n"
                "eor    v28.16b, v28.16b, v28.16b   \n"

                "2:                                 \n"
                "mov    v29.16b, v28.16b            \n"
                "mov    v30.16b, v28.16b            \n"
                "mov    v31.16b, v28.16b            \n"

                "3:                                 \n"
                "lsr    w4, %w9, #2                 \n" // w4 = max_kk >> 2
                "cmp    w4, #0                      \n"
                "beq    5f                          \n"

                "4:                                 \n"
                "prfm   pldl1keep, [%2, #256]       \n"
                "ld1    {v0.4h, v1.4h, v2.4h, v3.4h}, [%2], #32 \n"
                "shll   v0.4s, v0.4h, #16           \n"
                "shll   v1.4s, v1.4h, #16           \n"
                "shll   v2.4s, v2.4h, #16           \n"
                "shll   v3.4s, v3.4h, #16           \n"
                "prfm   pldl1keep, [%1, #256]       \n"
                "ld1    {v16.4h, v17.4h, v18.4h, v19.4h}, [%1], #32 \n"
                "shll   v16.4s, v16.4h, #16         \n"
                "shll   v17.4s, v17.4h, #16         \n"
                "shll   v18.4s, v18.4h, #16         \n"
                "shll   v19.4s, v19.4h, #16         \n"
                "fmla   v28.4s, v16.4s, v0.s[0]     \n"
                "fmla   v29.4s, v16.4s, v0.s[1]     \n"
                "fmla   v30.4s, v16.4s, v0.s[2]     \n"
                "fmla   v31.4s, v16.4s, v0.s[3]     \n"
                "fmla   v28.4s, v17.4s, v1.s[0]     \n"
                "fmla   v29.4s, v17.4s, v1.s[1]     \n"
                "fmla   v30.4s, v17.4s, v1.s[2]     \n"
                "fmla   v31.4s, v17.4s, v1.s[3]     \n"
                "subs   w4, w4, #1                  \n"
                "fmla   v28.4s, v18.4s, v2.s[0]     \n"
                "fmla   v29.4s, v18.4s, v2.s[1]     \n"
                "fmla   v30.4s, v18.4s, v2.s[2]     \n"
                "fmla   v31.4s, v18.4s, v2.s[3]     \n"
                "fmla   v28.4s, v19.4s, v3.s[0]     \n"
                "fmla   v29.4s, v19.4s, v3.s[1]     \n"
                "fmla   v30.4s, v19.4s, v3.s[2]     \n"
                "fmla   v31.4s, v19.4s, v3.s[3]     \n"
                "bne    4b                          \n"

                "5:                                 \n"
                "and    w4, %w9, #3                 \n" // w4 = remain = max_kk & 3
                "cmp    w4, #0                      \n"
                "beq    7f                          \n"

                "6:                                 \n"
                "ld1    {v0.4h}, [%2], #8           \n"
                "shll   v0.4s, v0.4h, #16           \n"
                "ld1    {v16.4h}, [%1], #8          \n"
                "shll   v16.4s, v16.4h, #16         \n"
                "fmla   v28.4s, v16.4s, v0.s[0]     \n"
                "fmla   v29.4s, v16.4s, v0.s[1]     \n"
                "subs   w4, w4, #1                  \n"
                "fmla   v30.4s, v16.4s, v0.s[2]     \n"
                "fmla   v31.4s, v16.4s, v0.s[3]     \n"
                "bne    6b                          \n"

                "7:                                 \n"
                "shrn   v0.4h, v28.4s, #16          \n"
                "shrn2  v0.8h, v29.4s, #16          \n"
                "shrn   v1.4h, v30.4s, #16          \n"
                "shrn2  v1.8h, v31.4s, #16          \n"
                "tst    %w11, #255                  \n"
                "beq    10f                         \n"

                // if out_elempack == 4
                "cmp    %w12, #4                    \n"
                "bne    8f                          \n"

                "st1    {v0.8h, v1.8h}, [%3], #32   \n"
                "b      9f                          \n"

                // if out_elempack == 1
                "8:                                 \n"
                // transpose4x4
                "uzp1   v20.8h, v0.8h, v1.8h        \n"
                "uzp2   v21.8h, v0.8h, v1.8h        \n"

                "uzp1   v0.8h, v20.8h, v21.8h       \n"
                "uzp2   v1.8h, v20.8h, v21.8h       \n"

                "add    x4, %3, %w13, sxtw 1        \n"
                "st1    {v0.d}[0], [%3], #8         \n"
                "st1    {v0.d}[1], [x4]             \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v1.d}[0], [x4]             \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v1.d}[1], [x4]             \n"

                "9:                                 \n"
                "add    %0, %0, #64                 \n"
                "b      11f                         \n"

                "10:                                \n"
                "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0], #64 \n"

                "11:                                \n"

                : "=r"(outptr), // %0
                "=r"(pA),     // %1
                "=r"(pB),     // %2
                "=r"(outptr0) // %3
                : "0"(outptr),
                "1"(pA),
                "2"(pB),
                "3"(outptr0),
                "r"(pC),           // %8
                "r"(max_kk),       // %9
                "r"(k),            // %10
                "r"(k_end),        // %11
                "r"(out_elempack), // %12
                "r"(out_hstep)     // %13
                : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v16", "v17", "v18", "v19", "v20", "v21", "v28", "v29", "v30", "v31");
#else  // __aarch64__
            asm volatile(
                "cmp        %10, #0             \n"
                "beq        0f                  \n"

                "vldm       %0, {d24-d31}       \n"
                "b          3f                  \n"

                "0:                             \n"
                // if pC
                "cmp        %8, #0              \n"
                "beq        1f                  \n"

                "vld1.f32   {d24-d25}, [%8]     \n"
                "b          2f                  \n"

                // else
                "1:                             \n"
                "veor       q12, q12            \n"

                "2:                             \n"
                "vmov       q13, q12            \n"
                "vmov       q14, q12            \n"
                "vmov       q15, q12            \n"

                "3:                             \n"
                "lsr        r4, %9, #2          \n" // r4 = max_kk >> 2
                "cmp        r4, #0              \n"
                "beq        5f                  \n"

                "4:                             \n"
                "pld        [%2, #256]          \n"
                "vld1.u16   {d4-d7}, [%2 :64]!  \n"
                "pld        [%1, #256]          \n"
                "vld1.u16   {d12-d15}, [%1 :64]! \n"
                "vshll.u16  q0, d4, #16         \n"
                "vshll.u16  q1, d5, #16         \n"
                "vshll.u16  q2, d6, #16         \n"
                "vshll.u16  q3, d7, #16         \n"
                "vshll.u16  q4, d12, #16        \n"
                "vshll.u16  q5, d13, #16        \n"
                "vshll.u16  q6, d14, #16        \n"
                "vshll.u16  q7, d15, #16        \n"
                "vmla.f32   q12, q4, d0[0]      \n"
                "vmla.f32   q13, q4, d0[1]      \n"
                "vmla.f32   q14, q4, d1[0]      \n"
                "vmla.f32   q15, q4, d1[1]      \n"
                "vmla.f32   q12, q5, d2[0]      \n"
                "vmla.f32   q13, q5, d2[1]      \n"
                "vmla.f32   q14, q5, d3[0]      \n"
                "vmla.f32   q15, q5, d3[1]      \n"
                "subs       r4, r4, #1          \n"
                "vmla.f32   q12, q6, d4[0]      \n"
                "vmla.f32   q13, q6, d4[1]      \n"
                "vmla.f32   q14, q6, d5[0]      \n"
                "vmla.f32   q15, q6, d5[1]      \n"
                "vmla.f32   q12, q7, d6[0]      \n"
                "vmla.f32   q13, q7, d6[1]      \n"
                "vmla.f32   q14, q7, d7[0]      \n"
                "vmla.f32   q15, q7, d7[1]      \n"
                "bne        4b                  \n"

                "5:                             \n"
                "and        r4, %9, #3          \n" // r4 = remain = max_kk & 3
                "cmp        r4, #0              \n"
                "beq        7f                  \n"

                "6:                             \n"
                "vld1.u16   {d0}, [%2 :64]!     \n"
                "vshll.u16  q0, d0, #16         \n"
                "vld1.u16   {d8}, [%1 :64]!     \n"
                "vshll.u16  q4, d8, #16         \n"
                "subs       r4, r4, #1          \n"
                "vmla.f32   q12, q4, d0[0]      \n"
                "vmla.f32   q13, q4, d0[1]      \n"
                "vmla.f32   q14, q4, d1[0]      \n"
                "vmla.f32   q15, q4, d1[1]      \n"
                "bne        6b                  \n"

                "7:                             \n"
                "vshrn.u32  d24, q12, #16       \n"
                "vshrn.u32  d25, q13, #16       \n"
                "vshrn.u32  d26, q14, #16       \n"
                "vshrn.u32  d27, q15, #16       \n"
                "cmp        %11, #0             \n"
                "beq        10f                 \n"

                // if out_elempack == 4
                "cmp        %12, #4             \n"
                "bne        8f                  \n"

                "vst1.u16   {d24-d27}, [%3]!    \n"
                "b          9f                  \n"

                // if out_elempack == 1
                "8:                             \n"
                // transpose4x4
                "vuzp.16    q12, q13            \n"
                "vuzp.16    q12, q13            \n"

                "add        r4, %3, %13, lsl #1 \n"
                "vst1.u16   {d24}, [%3]!        \n"
                "vst1.u16   {d25}, [r4]         \n"
                "add        r4, r4, %13, lsl #1 \n"
                "vst1.u16   {d26}, [r4]         \n"
                "add        r4, r4, %13, lsl #1 \n"
                "vst1.u16   {d27}, [r4]         \n"

                "9:                             \n"
                "add        %0, %0, #64         \n"
                "b          11f                 \n"

                "10:                            \n"
                "vstm       %0!, {d24-d31}      \n"

                "11:                            \n"

                : "=r"(outptr), // %0
                "=r"(pA),     // %1
                "=r"(pB),     // %2
                "=r"(outptr0) // %3
                : "0"(outptr),
                "1"(pA),
                "2"(pB),
                "3"(outptr0),
                "r"(pC),           // %8
                "r"(max_kk),       // %9
                "r"(k),            // %10
                "r"(k_end),        // %11
                "r"(out_elempack), // %12
                "r"(out_hstep)     // %13
                : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q12", "q13", "q14", "q15");
#endif // __aarch64__
#else  // NCNN_GNU_INLINE_ASM
            float32x4_t _sum0;
            float32x4_t _sum1;
            float32x4_t _sum2;
            float32x4_t _sum3;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = vld1q_f32(pC);
                    _sum1 = _sum0;
                    _sum2 = _sum0;
                    _sum3 = _sum0;
                }
                else
                {
                    _sum0 = vdupq_n_f32(0.f);
                    _sum1 = vdupq_n_f32(0.f);
                    _sum2 = vdupq_n_f32(0.f);
                    _sum3 = vdupq_n_f32(0.f);
                }
            }
            else
            {
                _sum0 = vld1q_f32(outptr);
                _sum1 = vld1q_f32(outptr + 4 * 1);
                _sum2 = vld1q_f32(outptr + 4 * 2);
                _sum3 = vld1q_f32(outptr + 4 * 3);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA = bfloat2float(vld1_u16(pA));
                float32x4_t _pB = bfloat2float(vld1_u16(pB));

#if __aarch64__
                _sum0 = vfmaq_laneq_f32(_sum0, _pA, _pB, 0);
                _sum1 = vfmaq_laneq_f32(_sum1, _pA, _pB, 1);
                _sum2 = vfmaq_laneq_f32(_sum2, _pA, _pB, 2);
                _sum3 = vfmaq_laneq_f32(_sum3, _pA, _pB, 3);
#else
                _sum0 = vmlaq_lane_f32(_sum0, _pA, vget_low_f32(_pB), 0);
                _sum1 = vmlaq_lane_f32(_sum1, _pA, vget_low_f32(_pB), 1);
                _sum2 = vmlaq_lane_f32(_sum2, _pA, vget_high_f32(_pB), 0);
                _sum3 = vmlaq_lane_f32(_sum3, _pA, vget_high_f32(_pB), 1);
#endif

                pA += 4;
                pB += 4;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1_u16(outptr0, float2bfloat(_sum0));
                    vst1_u16(outptr0 + 4, float2bfloat(_sum1));
                    vst1_u16(outptr0 + 4 * 2, float2bfloat(_sum2));
                    vst1_u16(outptr0 + 4 * 3, float2bfloat(_sum3));
                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    uint16x4_t _t0 = float2bfloat(_sum0);
                    uint16x4_t _t1 = float2bfloat(_sum1);
                    uint16x4_t _t2 = float2bfloat(_sum2);
                    uint16x4_t _t3 = float2bfloat(_sum3);
                    transpose4x4_u16(_t0, _t1, _t2, _t3);

                    vst1_u16(outptr0, _t0);
                    vst1_u16(outptr0 + out_hstep * 1, _t1);
                    vst1_u16(outptr0 + out_hstep * 2, _t2);
                    vst1_u16(outptr0 + out_hstep * 3, _t3);
                    outptr0 += 4;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum0);
                vst1q_f32(outptr + 4, _sum1);
                vst1q_f32(outptr + 4 * 2, _sum2);
                vst1q_f32(outptr + 4 * 3, _sum3);
            }

            outptr += 16;
#endif // NCNN_GNU_INLINE_ASM
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            const unsigned short* pA = pAT;

#if NCNN_GNU_INLINE_ASM
#if __aarch64__
            asm volatile(
                "cbz    %w10, 0f                    \n"

                "ld1    {v30.4s, v31.4s}, [%0]      \n"
                "b      3f                          \n"

                "0:                                 \n"
                // if pC
                "cbz    %8, 1f                      \n"

                "ld1    {v30.4s}, [%8]              \n"
                "b      2f                          \n"

                // else
                "1:                                 \n"
                "eor    v30.16b, v30.16b, v30.16b   \n"

                "2:                                 \n"
                "mov    v31.16b, v30.16b            \n"

                "3:                                 \n"
                "lsr    w4, %w9, #2                 \n" // w4 = max_kk >> 2
                "cmp    w4, #0                      \n"
                "beq    5f                          \n"

                "eor    v28.16b, v28.16b, v28.16b   \n"
                "eor    v29.16b, v29.16b, v29.16b   \n"
                "4:                                 \n"
                "prfm   pldl1keep, [%2, #128]       \n"
                "ld1    {v0.4h, v1.4h}, [%2], #16   \n"
                "shll   v0.4s, v0.4h, #16           \n"
                "shll   v1.4s, v1.4h, #16           \n"
                "prfm   pldl1keep, [%1, #256]       \n"
                "ld1    {v16.4h, v17.4h, v18.4h, v19.4h}, [%1], #32 \n"
                "shll   v16.4s, v16.4h, #16         \n"
                "shll   v17.4s, v17.4h, #16         \n"
                "shll   v18.4s, v18.4h, #16         \n"
                "shll   v19.4s, v19.4h, #16         \n"
                "fmla   v28.4s, v16.4s, v0.s[0]     \n"
                "fmla   v29.4s, v16.4s, v0.s[1]     \n"
                "fmla   v30.4s, v17.4s, v0.s[2]     \n"
                "fmla   v31.4s, v17.4s, v0.s[3]     \n"
                "subs   w4, w4, #1                  \n"
                "fmla   v28.4s, v18.4s, v1.s[0]     \n"
                "fmla   v29.4s, v18.4s, v1.s[1]     \n"
                "fmla   v30.4s, v19.4s, v1.s[2]     \n"
                "fmla   v31.4s, v19.4s, v1.s[3]     \n"
                "bne    4b                          \n"
                "fadd   v30.4s, v30.4s, v28.4s      \n"
                "fadd   v31.4s, v31.4s, v29.4s      \n"

                "5:                                 \n"
                "and    w4, %w9, #3                 \n" // w4 = remain = max_kk & 3
                "cmp    w4, #0                      \n"
                "beq    7f                          \n"

                "6:                                 \n"
                "ld1    {v0.s}[0], [%2], #4         \n"
                "shll   v0.4s, v0.4h, #16           \n"
                "ld1    {v16.4h}, [%1], #8          \n"
                "shll   v16.4s, v16.4h, #16         \n"
                "subs   w4, w4, #1                  \n"
                "fmla   v30.4s, v16.4s, v0.s[0]     \n"
                "fmla   v31.4s, v16.4s, v0.s[1]     \n"
                "bne    6b                          \n"

                "7:                                 \n"
                "shrn   v0.4h, v30.4s, #16          \n"
                "shrn   v1.4h, v31.4s, #16          \n"
                "tst    %w11, #255                  \n"
                "beq    10f                         \n"

                // if out_elempack == 4
                "cmp    %w12, #4                    \n"
                "bne    8f                          \n"

                "st1    {v0.4h, v1.4h}, [%3], #16   \n"
                "b      9f                          \n"

                // if out_elempack == 1
                "8:                                 \n"
                // transpose4x2
                "zip1   v30.4h, v0.4h, v1.4h        \n"
                "zip2   v31.4h, v0.4h, v1.4h        \n"

                "add    x4, %3, %w13, sxtw 1        \n"
                "st1    {v30.s}[0], [%3], #4        \n"
                "st1    {v30.s}[1], [x4]            \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v31.s}[0], [x4]            \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v31.s}[1], [x4]            \n"

                "9:                                 \n"
                "add    %0, %0, #32                 \n"
                "b      11f                         \n"

                "10:                                \n"
                "st1    {v30.4s, v31.4s}, [%0], #32 \n"

                "11:                                \n"

                : "=r"(outptr), // %0
                "=r"(pA),     // %1
                "=r"(pB),     // %2
                "=r"(outptr0) // %3
                : "0"(outptr),
                "1"(pA),
                "2"(pB),
                "3"(outptr0),
                "r"(pC),           // %8
                "r"(max_kk),       // %9
                "r"(k),            // %10
                "r"(k_end),        // %11
                "r"(out_elempack), // %12
                "r"(out_hstep)     // %13
                : "cc", "memory", "x4", "v0", "v1", "v16", "v17", "v18", "v19", "v28", "v29", "v30", "v31");
#else  // __aarch64__
            asm volatile(
                "cmp        %10, #0             \n"
                "beq        0f                  \n"

                "vld1.f32   {d28-d31}, [%0 :128] \n"
                "b          3f                  \n"

                "0:                             \n"
                // if pC
                "cmp        %8, #0              \n"
                "beq        1f                  \n"

                "vld1.f32   {d28-d29}, [%8]     \n"
                "b          2f                  \n"

                // else
                "1:                             \n"
                "veor       q14, q14            \n"

                "2:                             \n"
                "vmov       q15, q14            \n"

                "3:                             \n"
                "lsr        r4, %9, #2          \n" // r4 = max_kk >> 2
                "cmp        r4, #0              \n"
                "beq        5f                  \n"

                "veor       q12, q12            \n"
                "veor       q13, q13            \n"
                "4:                             \n"
                "pld        [%2, #128]          \n"
                "vld1.u16   {d2-d3}, [%2 :64]!  \n"
                "pld        [%1, #256]          \n"
                "vld1.u16   {d12-d15}, [%1 :64]! \n"
                "vshll.u16  q0, d2, #16         \n"
                "vshll.u16  q1, d3, #16         \n"
                "vshll.u16  q4, d12, #16        \n"
                "vshll.u16  q5, d13, #16        \n"
                "vshll.u16  q6, d14, #16        \n"
                "vshll.u16  q7, d15, #16        \n"
                "vmla.f32   q12, q4, d0[0]      \n"
                "vmla.f32   q13, q4, d0[1]      \n"
                "vmla.f32   q14, q5, d1[0]      \n"
                "vmla.f32   q15, q5, d1[1]      \n"
                "subs       r4, r4, #1          \n"
                "vmla.f32   q12, q6, d2[0]      \n"
                "vmla.f32   q13, q6, d2[1]      \n"
                "vmla.f32   q14, q7, d3[0]      \n"
                "vmla.f32   q15, q7, d3[1]      \n"
                "bne        4b                  \n"
                "vadd.f32   q14, q14, q12       \n"
                "vadd.f32   q15, q15, q13       \n"

                "5:                             \n"
                "and        r4, %9, #3          \n" // r4 = remain = max_kk & 3
                "cmp        r4, #0              \n"
                "beq        7f                  \n"

                "6:                             \n"
                "vld1.u32   {d0[0]}, [%2]!      \n"
                "vshll.u16  q0, d0, #16         \n"
                "vld1.u16   {d8}, [%1 :64]!     \n"
                "vshll.u16  q4, d8, #16         \n"
                "subs       r4, r4, #1          \n"
                "vmla.f32   q14, q4, d0[0]      \n"
                "vmla.f32   q15, q4, d0[1]      \n"
                "bne        6b                  \n"

                "7:                             \n"
                "vshrn.u32  d28, q14, #16       \n"
                "vshrn.u32  d29, q15, #16       \n"
                "cmp        %11, #0             \n"
                "beq        10f                 \n"

                // if out_elempack == 4
                "cmp        %12, #4             \n"
                "bne        8f                  \n"

                "vst1.u16   {d28-d29}, [%3]!    \n"
                "b          9f                  \n"

                // if out_elempack == 1
                "8:                             \n"
                // transpose4x2
                "vzip.16    d28, d29            \n"

                "add        r4, %3, %13, lsl #1 \n"
                "vst1.u32   {d28[0]}, [%3]!     \n"
                "vst1.u32   {d28[1]}, [r4]      \n"
                "add        r4, r4, %13, lsl #1 \n"
                "vst1.u32   {d29[0]}, [r4]      \n"
                "add        r4, r4, %13, lsl #1 \n"
                "vst1.u32   {d29[1]}, [r4]      \n"

                "9:                             \n"
                "add        %0, %0, #32         \n"
                "b          11f                 \n"

                "10:                            \n"
                "vst1.f32   {d28-d31}, [%0 :128]! \n"

                "11:                            \n"

                : "=r"(outptr), // %0
                "=r"(pA),     // %1
                "=r"(pB),     // %2
                "=r"(outptr0) // %3
                : "0"(outptr),
                "1"(pA),
                "2"(pB),
                "3"(outptr0),
                "r"(pC),           // %8
                "r"(max_kk),       // %9
                "r"(k),            // %10
                "r"(k_end),        // %11
                "r"(out_elempack), // %12
                "r"(out_hstep)     // %13
                : "cc", "memory", "r4", "q0", "q1", "q4", "q5", "q6", "q7", "q12", "q13", "q14", "q15");
#endif // __aarch64__
#else  // NCNN_GNU_INLINE_ASM
            float32x4_t _sum0;
            float32x4_t _sum1;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = vld1q_f32(pC);
                    _sum1 = _sum0;
                }
                else
                {
                    _sum0 = vdupq_n_f32(0.f);
                    _sum1 = vdupq_n_f32(0.f);
                }
            }
            else
            {
                _sum0 = vld1q_f32(outptr);
                _sum1 = vld1q_f32(outptr + 4);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA = bfloat2float(vld1_u16(pA));
                float32x2_t _pB = vget_low_f32(bfloat2float(vld1_u16(pB)));

#if __aarch64__
                _sum0 = vfmaq_lane_f32(_sum0, _pA, _pB, 0);
                _sum1 = vfmaq_lane_f32(_sum1, _pA, _pB, 1);
#else
                _sum0 = vmlaq_lane_f32(_sum0, _pA, _pB, 0);
                _sum1 = vmlaq_lane_f32(_sum1, _pA, _pB, 1);
#endif

                pA += 4;
                pB += 2;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1_u16(outptr0, float2bfloat(_sum0));
                    vst1_u16(outptr0 + 4, float2bfloat(_sum1));
                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
                    unsigned short sum0[4];
                    unsigned short sum1[4];
                    vst1_u16(sum0, float2bfloat(_sum0));
                    vst1_u16(sum1, float2bfloat(_sum1));

                    outptr0[0] = sum0[0];
                    outptr0[out_hstep] = sum0[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0[1] = sum1[0];
                    outptr0[out_hstep + 1] = sum1[1];
                    outptr0[out_hstep * 2 + 1] = sum1[2];
                    outptr0[out_hstep * 3 + 1] = sum1[3];
                    outptr0 += 2;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum0);
                vst1q_f32(outptr + 4, _sum1);
            }

            outptr += 8;
#endif // NCNN_GNU_INLINE_ASM
        }
        for (; jj < max_jj; jj += 1)
        {
            const unsigned short* pA = pAT;

#if NCNN_GNU_INLINE_ASM
#if __aarch64__
            asm volatile(
                "cbz    %w10, 0f                    \n"

                "ld1    {v31.4s}, [%0]              \n"
                "b      2f                          \n"

                "0:                                 \n"
                // if pC
                "cbz    %8, 1f                      \n"

                "ld1    {v31.4s}, [%8]              \n"
                "b      2f                          \n"

                // else
                "1:                                 \n"
                "eor    v31.16b, v31.16b, v31.16b   \n"

                "2:                                 \n"
                "lsr    w4, %w9, #2                 \n" // w4 = max_kk >> 2
                "cmp    w4, #0                      \n"
                "beq    4f                          \n"

                "eor    v28.16b, v28.16b, v28.16b   \n"
                "eor    v29.16b, v29.16b, v29.16b   \n"
                "eor    v30.16b, v30.16b, v30.16b   \n"
                "3:                                 \n"
                "prfm   pldl1keep, [%2, #64]        \n"
                "ld1    {v0.4h}, [%2], #8           \n"
                "shll   v0.4s, v0.4h, #16           \n"
                "prfm   pldl1keep, [%1, #256]       \n"
                "ld1    {v16.4h, v17.4h, v18.4h, v19.4h}, [%1], #32 \n"
                "shll   v16.4s, v16.4h, #16         \n"
                "shll   v17.4s, v17.4h, #16         \n"
                "shll   v18.4s, v18.4h, #16         \n"
                "shll   v19.4s, v19.4h, #16         \n"
                "fmla   v28.4s, v16.4s, v0.s[0]     \n"
                "fmla   v29.4s, v17.4s, v0.s[1]     \n"
                "subs   w4, w4, #1                  \n"
                "fmla   v30.4s, v18.4s, v0.s[2]     \n"
                "fmla   v31.4s, v19.4s, v0.s[3]     \n"
                "bne    3b                          \n"
                "fadd   v30.4s, v30.4s, v28.4s      \n"
                "fadd   v31.4s, v31.4s, v29.4s      \n"
                "fadd   v31.4s, v31.4s, v30.4s      \n"

                "4:                                 \n"
                "and    w4, %w9, #3                 \n" // w4 = remain = max_kk & 3
                "cmp    w4, #0                      \n"
                "beq    6f                          \n"

                "5:                                 \n"
                "ld1r   {v0.4h}, [%2], #2           \n"
                "shll   v0.4s, v0.4h, #16           \n"
                "ld1    {v16.4h}, [%1], #8          \n"
                "shll   v16.4s, v16.4h, #16         \n"
                "subs   w4, w4, #1                  \n"
                "fmla   v31.4s, v16.4s, v0.4s       \n"
                "bne    5b                          \n"

                "6:                                 \n"
                "shrn   v0.4h, v31.4s, #16          \n"
                "tst    %w11, #255                  \n"
                "beq    9f                          \n"

                // if out_elempack == 4
                "cmp    %w12, #4                    \n"
                "bne    7f                          \n"

                "st1    {v0.4h}, [%3], #8           \n"
                "b      8f                          \n"

                // if out_elempack == 1
                "7:                                 \n"
                "add    x4, %3, %w13, sxtw 1        \n"
                "st1    {v0.h}[0], [%3], #2         \n"
                "st1    {v0.h}[1], [x4]             \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v0.h}[2], [x4]             \n"
                "add    x4, x4, %w13, sxtw 1        \n"
                "st1    {v0.h}[3], [x4]             \n"

                "8:                                 \n"
                "add    %0, %0, #16                 \n"
                "b      10f                         \n"

                "9:                                 \n"
                "st1    {v31.4s}, [%0], #16         \n"

                "10:                                \n"

                : "=r"(outptr), // %0
                "=r"(pA),     // %1
                "=r"(pB),     // %2
                "=r"(outptr0) // %3
                : "0"(outptr),
                "1"(pA),
                "2"(pB),
                "3"(outptr0),
                "r"(pC),           // %8
                "r"(max_kk),       // %9
                "r"(k),            // %10
                "r"(k_end),        // %11
                "r"(out_elempack), // %12
                "r"(out_hstep)     // %13
                : "cc", "memory", "x4", "v0", "v16", "v17", "v18", "v19", "v28", "v29", "v30", "v31");
#else  // __aarch64__
            asm volatile(
                "cmp        %10, #0             \n"
                "beq        0f                  \n"

                "vld1.f32   {d30-d31}, [%0 :64] \n"
                "b          2f                  \n"

                "0:                             \n"
                // if pC
                "cmp        %8, #0              \n"
                "beq        1f                  \n"

                "vld1.f32   {d30-d31}, [%8]     \n"
                "b          2f                  \n"

                // else
                "1:                             \n"
                "veor       q15, q15            \n"

                "2:                             \n"
                "lsr        r4, %9, #2          \n" // r4 = max_kk >> 2
                "cmp        r4, #0              \n"
                "beq        4f                  \n"

                "veor       q12, q12            \n"
                "veor       q13, q13            \n"
                "veor       q14, q14            \n"
                "3:                             \n"
                "pld        [%2, #64]           \n"
                "vld1.u16   {d1}, [%2]!         \n"
                "pld        [%1, #256]          \n"
                "vld1.u16   {d12-d15}, [%1 :64]! \n"
                "vshll.u16  q0, d1, #16         \n"
                "vshll.u16  q4, d12, #16        \n"
                "vshll.u16  q5, d13, #16        \n"
                "vshll.u16  q6, d14, #16        \n"
                "vshll.u16  q7, d15, #16        \n"
                "vmla.f32   q12, q4, d0[0]      \n"
                "vmla.f32   q13, q5, d0[1]      \n"
                "vmla.f32   q14, q6, d1[0]      \n"
                "vmla.f32   q15, q7, d1[1]      \n"
                "subs       r4, r4, #1          \n"
                "bne        3b                  \n"
                "vadd.f32   q14, q14, q12       \n"
                "vadd.f32   q15, q15, q13       \n"
                "vadd.f32   q15, q15, q14       \n"

                "4:                             \n"
                "and        r4, %9, #3          \n" // r4 = remain = max_kk & 3
                "cmp        r4, #0              \n"
                "beq        6f                  \n"

                "5:                             \n"
                "vld1.u16   {d0[]}, [%2]!       \n"
                "vshll.u16  q0, d0, #16         \n"
                "vld1.u16   {d8}, [%1 :64]!     \n"
                "vshll.u16  q4, d8, #16         \n"
                "subs       r4, r4, #1          \n"
                "vmla.f32   q15, q4, q0         \n"
                "bne        5b                  \n"

                "6:                             \n"
                "vshrn.u32  d30, q15, #16       \n"
                "cmp        %11, #0             \n"
                "beq        9f                  \n"

                // if out_elempack == 4
                "cmp        %12, #4             \n"
                "bne        7f                  \n"

                "vst1.u16   {d30}, [%3]!        \n"
                "b          8f                  \n"

                // if out_elempack == 1
                "7:                             \n"

                "add        r4, %3, %13, lsl #1 \n"
                "vst1.u16   {d30[0]}, [%3]!     \n"
                "vst1.u16   {d30[1]}, [r4]      \n"
                "add        r4, r4, %13, lsl #1 \n"
                "vst1.u16   {d30[2]}, [r4]      \n"
                "add        r4, r4, %13, lsl #1 \n"
                "vst1.u16   {d30[3]}, [r4]      \n"

                "8:                             \n"
                "add        %0, %0, #16         \n"
                "b          10f                 \n"

                "9:                             \n"
                "vst1.f32   {d30-d31}, [%0 :64]! \n"

                "10:                            \n"

                : "=r"(outptr), // %0
                "=r"(pA),     // %1
                "=r"(pB),     // %2
                "=r"(outptr0) // %3
                : "0"(outptr),
                "1"(pA),
                "2"(pB),
                "3"(outptr0),
                "r"(pC),           // %8
                "r"(max_kk),       // %9
                "r"(k),            // %10
                "r"(k_end),        // %11
                "r"(out_elempack), // %12
                "r"(out_hstep)     // %13
                : "cc", "memory", "r4", "q0", "q4", "q5", "q6", "q7", "q12", "q13", "q14", "q15");
#endif // __aarch64__
#else  // NCNN_GNU_INLINE_ASM
            float32x4_t _sum0;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = vld1q_f32(pC);
                }
                else
                {
                    _sum0 = vdupq_n_f32(0.f);
                }
            }
            else
            {
                _sum0 = vld1q_f32(outptr);
            }

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA = bfloat2float(vld1_u16(pA));
                float32x4_t _pB = bfloat2float(vdup_n_u16(pB[0]));

#if __aarch64__
                _sum0 = vfmaq_f32(_sum0, _pA, _pB);
#else
                _sum0 = vmlaq_f32(_sum0, _pA, _pB);
#endif

                pA += 4;
                pB += 1;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1_u16(outptr0, float2bfloat(_sum0));
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
                    unsigned short sum0[4];
                    vst1_u16(sum0, float2bfloat(_sum0));

                    outptr0[0] = sum0[0];
                    outptr0[out_hstep] = sum0[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0++;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum0);
            }

            outptr += 4;
#endif // NCNN_GNU_INLINE_ASM
        }

        pAT += max_kk * 4;
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
        unsigned short* outptr0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j;

        const unsigned short* pB = pBT;

        if (pC)
        {
            pC = (const float*)CT_tile + i + ii;
        }

        int jj = 0;
#if __ARM_NEON
#if __aarch64__
        for (; jj + 11 < max_jj; jj += 12)
        {
            float32x4_t _sum00;
            float32x4_t _sum01;
            float32x4_t _sum02;
            float32x4_t _sum10;
            float32x4_t _sum11;
            float32x4_t _sum12;

            if (k == 0)
            {
                if (pC)
                {
                    _sum00 = vdupq_n_f32(pC[0]);
                    _sum01 = vdupq_n_f32(pC[0]);
                    _sum02 = vdupq_n_f32(pC[0]);
                    _sum10 = vdupq_n_f32(pC[1]);
                    _sum11 = vdupq_n_f32(pC[1]);
                    _sum12 = vdupq_n_f32(pC[1]);
                }
                else
                {
                    _sum00 = vdupq_n_f32(0.f);
                    _sum01 = vdupq_n_f32(0.f);
                    _sum02 = vdupq_n_f32(0.f);
                    _sum10 = vdupq_n_f32(0.f);
                    _sum11 = vdupq_n_f32(0.f);
                    _sum12 = vdupq_n_f32(0.f);
                }
            }
            else
            {
                float32x4x2_t _tmp01 = vld2q_f32(outptr);
                float32x4x2_t _tmp23 = vld2q_f32(outptr + 8);
                float32x4x2_t _tmp45 = vld2q_f32(outptr + 16);
                _sum00 = _tmp01.val[0];
                _sum01 = _tmp23.val[0];
                _sum02 = _tmp45.val[0];
                _sum10 = _tmp01.val[1];
                _sum11 = _tmp23.val[1];
                _sum12 = _tmp45.val[1];
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pB0 = bfloat2float(vld1_u16(pB));
                float32x4_t _pB1 = bfloat2float(vld1_u16(pB + 4));
                float32x4_t _pB2 = bfloat2float(vld1_u16(pB + 8));

                float32x2_t _pA = vget_low_f32(bfloat2float(vld1_u16(pA)));

                _sum00 = vfmaq_lane_f32(_sum00, _pB0, _pA, 0);
                _sum01 = vfmaq_lane_f32(_sum01, _pB1, _pA, 0);
                _sum02 = vfmaq_lane_f32(_sum02, _pB2, _pA, 0);
                _sum10 = vfmaq_lane_f32(_sum10, _pB0, _pA, 1);
                _sum11 = vfmaq_lane_f32(_sum11, _pB1, _pA, 1);
                _sum12 = vfmaq_lane_f32(_sum12, _pB2, _pA, 1);

                pA += 2;
                pB += 12;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1_u16(outptr0, float2bfloat(_sum00));
                    vst1_u16(outptr0 + 4, float2bfloat(_sum01));
                    vst1_u16(outptr0 + 8, float2bfloat(_sum02));
                    vst1_u16(outptr0 + out_hstep, float2bfloat(_sum10));
                    vst1_u16(outptr0 + out_hstep + 4, float2bfloat(_sum11));
                    vst1_u16(outptr0 + out_hstep + 8, float2bfloat(_sum12));
                    outptr0 += 12;
                }
            }
            else
            {
                float32x4x2_t _tmp01;
                _tmp01.val[0] = _sum00;
                _tmp01.val[1] = _sum10;
                float32x4x2_t _tmp23;
                _tmp23.val[0] = _sum01;
                _tmp23.val[1] = _sum11;
                float32x4x2_t _tmp45;
                _tmp45.val[0] = _sum02;
                _tmp45.val[1] = _sum12;
                vst2q_f32(outptr, _tmp01);
                vst2q_f32(outptr + 8, _tmp23);
                vst2q_f32(outptr + 16, _tmp45);
            }

            outptr += 24;
        }
#endif // __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            float32x4_t _sum00;
            float32x4_t _sum01;
            float32x4_t _sum10;
            float32x4_t _sum11;

            if (k == 0)
            {
                if (pC)
                {
                    _sum00 = vdupq_n_f32(pC[0]);
                    _sum01 = vdupq_n_f32(pC[0]);
                    _sum10 = vdupq_n_f32(pC[1]);
                    _sum11 = vdupq_n_f32(pC[1]);
                }
                else
                {
                    _sum00 = vdupq_n_f32(0.f);
                    _sum01 = vdupq_n_f32(0.f);
                    _sum10 = vdupq_n_f32(0.f);
                    _sum11 = vdupq_n_f32(0.f);
                }
            }
            else
            {
                float32x4x2_t _tmp01 = vld2q_f32(outptr);
                float32x4x2_t _tmp23 = vld2q_f32(outptr + 8);
                _sum00 = _tmp01.val[0];
                _sum01 = _tmp23.val[0];
                _sum10 = _tmp01.val[1];
                _sum11 = _tmp23.val[1];
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pB0 = bfloat2float(vld1_u16(pB));
                float32x4_t _pB1 = bfloat2float(vld1_u16(pB + 4));

                float32x2_t _pA = vget_low_f32(bfloat2float(vld1_u16(pA)));
#if __aarch64__
                _sum00 = vfmaq_lane_f32(_sum00, _pB0, _pA, 0);
                _sum01 = vfmaq_lane_f32(_sum01, _pB1, _pA, 0);
                _sum10 = vfmaq_lane_f32(_sum10, _pB0, _pA, 1);
                _sum11 = vfmaq_lane_f32(_sum11, _pB1, _pA, 1);
#else
                _sum00 = vmlaq_lane_f32(_sum00, _pB0, _pA, 0);
                _sum01 = vmlaq_lane_f32(_sum01, _pB1, _pA, 0);
                _sum10 = vmlaq_lane_f32(_sum10, _pB0, _pA, 1);
                _sum11 = vmlaq_lane_f32(_sum11, _pB1, _pA, 1);
#endif

                pA += 2;
                pB += 8;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1_u16(outptr0, float2bfloat(_sum00));
                    vst1_u16(outptr0 + 4, float2bfloat(_sum01));
                    vst1_u16(outptr0 + out_hstep, float2bfloat(_sum10));
                    vst1_u16(outptr0 + out_hstep + 4, float2bfloat(_sum11));
                    outptr0 += 8;
                }
            }
            else
            {
                float32x4x2_t _tmp01;
                _tmp01.val[0] = _sum00;
                _tmp01.val[1] = _sum10;
                float32x4x2_t _tmp23;
                _tmp23.val[0] = _sum01;
                _tmp23.val[1] = _sum11;
                vst2q_f32(outptr, _tmp01);
                vst2q_f32(outptr + 8, _tmp23);
            }

            outptr += 16;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _sum0;
            float32x4_t _sum1;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = vdupq_n_f32(pC[0]);
                    _sum1 = vdupq_n_f32(pC[1]);
                }
                else
                {
                    _sum0 = vdupq_n_f32(0.f);
                    _sum1 = vdupq_n_f32(0.f);
                }
            }
            else
            {
                float32x4x2_t _tmp01 = vld2q_f32(outptr);
                _sum0 = _tmp01.val[0];
                _sum1 = _tmp01.val[1];
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pB = bfloat2float(vld1_u16(pB));

                float32x2_t _pA = vget_low_f32(bfloat2float(vld1_u16(pA)));
#if __aarch64__
                _sum0 = vfmaq_lane_f32(_sum0, _pB, _pA, 0);
                _sum1 = vfmaq_lane_f32(_sum1, _pB, _pA, 1);
#else
                _sum0 = vmlaq_lane_f32(_sum0, _pB, _pA, 0);
                _sum1 = vmlaq_lane_f32(_sum1, _pB, _pA, 1);
#endif

                pA += 2;
                pB += 4;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1_u16(outptr0, float2bfloat(_sum0));
                    vst1_u16(outptr0 + out_hstep, float2bfloat(_sum1));
                    outptr0 += 4;
                }
            }
            else
            {
                float32x4x2_t _tmp01;
                _tmp01.val[0] = _sum0;
                _tmp01.val[1] = _sum1;
                vst2q_f32(outptr, _tmp01);
            }

            outptr += 8;
        }
#endif // __ARM_NEON
        for (; jj + 1 < max_jj; jj += 2)
        {
            float sum00;
            float sum01;
            float sum10;
            float sum11;

            if (k == 0)
            {
                if (pC)
                {
                    sum00 = pC[0];
                    sum01 = pC[1];
                    sum10 = pC[0];
                    sum11 = pC[1];
                }
                else
                {
                    sum00 = 0.f;
                    sum01 = 0.f;
                    sum10 = 0.f;
                    sum11 = 0.f;
                }
            }
            else
            {
                sum00 = outptr[0];
                sum01 = outptr[1];
                sum10 = outptr[2];
                sum11 = outptr[3];
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum00 += bfloat16_to_float32(pA[0]) * bfloat16_to_float32(pB[0]);
                sum01 += bfloat16_to_float32(pA[1]) * bfloat16_to_float32(pB[0]);
                sum10 += bfloat16_to_float32(pA[0]) * bfloat16_to_float32(pB[1]);
                sum11 += bfloat16_to_float32(pA[1]) * bfloat16_to_float32(pB[1]);

                pA += 2;
                pB += 2;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = float32_to_bfloat16(sum00);
                    outptr0[1] = float32_to_bfloat16(sum10);
                    outptr0[out_hstep] = float32_to_bfloat16(sum01);
                    outptr0[out_hstep + 1] = float32_to_bfloat16(sum11);
                    outptr0 += 2;
                }
            }
            else
            {
                outptr[0] = sum00;
                outptr[1] = sum01;
                outptr[2] = sum10;
                outptr[3] = sum11;
            }

            outptr += 4;
        }
        for (; jj < max_jj; jj += 1)
        {
            float sum0;
            float sum1;

            if (k == 0)
            {
                if (pC)
                {
                    sum0 = pC[0];
                    sum1 = pC[1];
                }
                else
                {
                    sum0 = 0.f;
                    sum1 = 0.f;
                }
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum0 += bfloat16_to_float32(pA[0]) * bfloat16_to_float32(pB[0]);
                sum1 += bfloat16_to_float32(pA[1]) * bfloat16_to_float32(pB[0]);
                pA += 2;
                pB += 1;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = float32_to_bfloat16(sum0);
                    outptr0[out_hstep] = float32_to_bfloat16(sum1);
                    outptr0++;
                }
            }
            else
            {
                outptr[0] = sum0;
                outptr[1] = sum1;
            }

            outptr += 2;
        }

        pAT += max_kk * 2;
    }
    for (; ii < max_ii; ii += 1)
    {
        unsigned short* outptr0 = (unsigned short*)top_blob + (i + ii) * out_hstep + j;

        const unsigned short* pB = pBT;

        if (pC)
        {
            pC = (const float*)CT_tile + i + ii;
        }

        int jj = 0;
#if __ARM_NEON
#if __aarch64__
        for (; jj + 11 < max_jj; jj += 12)
        {
            float32x4_t _sum0;
            float32x4_t _sum1;
            float32x4_t _sum2;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = vdupq_n_f32(pC[0]);
                    _sum1 = vdupq_n_f32(pC[0]);
                    _sum2 = vdupq_n_f32(pC[0]);
                }
                else
                {
                    _sum0 = vdupq_n_f32(0.f);
                    _sum1 = vdupq_n_f32(0.f);
                    _sum2 = vdupq_n_f32(0.f);
                }
            }
            else
            {
                _sum0 = vld1q_f32(outptr);
                _sum1 = vld1q_f32(outptr + 4);
                _sum2 = vld1q_f32(outptr + 8);
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pB0 = bfloat2float(vld1_u16(pB));
                float32x4_t _pB1 = bfloat2float(vld1_u16(pB + 4));
                float32x4_t _pB2 = bfloat2float(vld1_u16(pB + 8));

                float32x4_t _pA0 = bfloat2float(vdup_n_u16(pA[0]));

                _sum0 = vfmaq_f32(_sum0, _pA0, _pB0);
                _sum1 = vfmaq_f32(_sum1, _pA0, _pB1);
                _sum2 = vfmaq_f32(_sum2, _pA0, _pB2);

                pA += 1;
                pB += 12;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1_u16(outptr0, float2bfloat(_sum0));
                    vst1_u16(outptr0 + 4, float2bfloat(_sum1));
                    vst1_u16(outptr0 + 8, float2bfloat(_sum2));
                    outptr0 += 12;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum0);
                vst1q_f32(outptr + 4, _sum1);
                vst1q_f32(outptr + 8, _sum2);
            }

            outptr += 12;
        }
#endif // __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            float32x4_t _sum0;
            float32x4_t _sum1;

            if (k == 0)
            {
                if (pC)
                {
                    _sum0 = vdupq_n_f32(pC[0]);
                    _sum1 = vdupq_n_f32(pC[0]);
                }
                else
                {
                    _sum0 = vdupq_n_f32(0.f);
                    _sum1 = vdupq_n_f32(0.f);
                }
            }
            else
            {
                _sum0 = vld1q_f32(outptr);
                _sum1 = vld1q_f32(outptr + 4);
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pB0 = bfloat2float(vld1_u16(pB));
                float32x4_t _pB1 = bfloat2float(vld1_u16(pB + 4));

                float32x4_t _pA0 = bfloat2float(vdup_n_u16(pA[0]));
#if __aarch64__
                _sum0 = vfmaq_f32(_sum0, _pA0, _pB0);
                _sum1 = vfmaq_f32(_sum1, _pA0, _pB1);
#else
                _sum0 = vmlaq_f32(_sum0, _pA0, _pB0);
                _sum1 = vmlaq_f32(_sum1, _pA0, _pB1);
#endif

                pA += 1;
                pB += 8;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1_u16(outptr0, float2bfloat(_sum0));
                    vst1_u16(outptr0 + 4, float2bfloat(_sum1));
                    outptr0 += 8;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum0);
                vst1q_f32(outptr + 4, _sum1);
            }

            outptr += 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _sum;

            if (k == 0)
            {
                if (pC)
                {
                    _sum = vdupq_n_f32(pC[0]);
                }
                else
                {
                    _sum = vdupq_n_f32(0.f);
                }
            }
            else
            {
                _sum = vld1q_f32(outptr);
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pB = bfloat2float(vld1_u16(pB));
                float32x4_t _pA = bfloat2float(vdup_n_u16(pA[0]));

#if __aarch64__
                _sum = vfmaq_f32(_sum, _pA, _pB);
#else
                _sum = vmlaq_f32(_sum, _pA, _pB);
#endif

                pA += 1;
                pB += 4;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1_u16(outptr0, float2bfloat(_sum));
                    outptr0 += 4;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum);
            }

            outptr += 4;
        }
#endif // __ARM_NEON
        for (; jj + 1 < max_jj; jj += 2)
        {
            float sum0;
            float sum1;

            if (k == 0)
            {
                if (pC)
                {
                    sum0 = pC[0];
                    sum1 = pC[0];
                }
                else
                {
                    sum0 = 0.f;
                    sum1 = 0.f;
                }
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum0 += bfloat16_to_float32(pA[0]) * bfloat16_to_float32(pB[0]);
                sum1 += bfloat16_to_float32(pA[0]) * bfloat16_to_float32(pB[1]);

                pA += 1;
                pB += 2;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = float32_to_bfloat16(sum0);
                    outptr0[1] = float32_to_bfloat16(sum1);
                    outptr0 += 2;
                }
            }
            else
            {
                outptr[0] = sum0;
                outptr[1] = sum1;
            }

            outptr += 2;
        }
        for (; jj < max_jj; jj += 1)
        {
            float sum;

            if (k == 0)
            {
                if (pC)
                {
                    sum = pC[0];
                }
                else
                {
                    sum = 0.f;
                }
            }
            else
            {
                sum = outptr[0];
            }

            const unsigned short* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum += bfloat16_to_float32(pA[0]) * bfloat16_to_float32(pB[0]);

                pA += 1;
                pB += 1;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = float32_to_bfloat16(sum);
                    outptr0++;
                }
            }
            else
            {
                outptr[0] = sum;
            }

            outptr += 1;
        }

        pAT += max_kk;
    }
}

static void convolution_im2col_gemm_get_optimal_tile_mnk_bf16s(int M, int N, int K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    // resolve optimal tile size from cache size
    const int l2_cache_size_bf16 = (int)(get_cpu_level2_cache_size() / sizeof(unsigned short));

    if (nT == 0)
        nT = get_physical_big_cpu_count();

    // solve K
    {
        // try not to split K
#if __aarch64__
        int tile_size = (l2_cache_size_bf16 - 32) / 12;
#elif __ARM_NEON
        int tile_size = (l2_cache_size_bf16 - 16) / 8;
#else
        int tile_size = (l2_cache_size_bf16 - 2) / 3;
#endif

#if __aarch64__
        TILE_K = std::max(8, tile_size / 8 * 8);
#elif __ARM_NEON
        TILE_K = std::max(4, tile_size / 4 * 4);
#else
        TILE_K = std::max(2, tile_size / 2 * 2);
#endif

        int nn_K = (K + TILE_K - 1) / TILE_K;
#if __aarch64__
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 7) / 8 * 8);
#elif __ARM_NEON
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 3) / 4 * 4);
#else
        TILE_K = std::min(TILE_K, ((K + nn_K - 1) / nn_K + 1) / 2 * 2);
#endif
    }

    // solve M
    {
#if __aarch64__
        int nn_M = (M + 31) / 32;
#elif __ARM_NEON
        int nn_M = (M + 15) / 16;
#else
        int nn_M = (M + 7) / 8;
#endif

#if __aarch64__
        TILE_M = std::max(8, ((M + nn_M - 1) / nn_M + 7) / 8 * 8);
#elif __ARM_NEON
        TILE_M = std::max(4, ((M + nn_M - 1) / nn_M + 3) / 4 * 4);
#else
        TILE_M = std::max(2, ((M + nn_M - 1) / nn_M + 1) / 2 * 2);
#endif
    }

    {
        TILE_M *= std::min(nT, get_physical_cpu_count());

        int nn_M = (M + TILE_M - 1) / TILE_M;
#if __aarch64__
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 7) / 8 * 8);
#elif __ARM_NEON
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 3) / 4 * 4);
#else
        TILE_M = std::min(TILE_M, ((M + nn_M - 1) / nn_M + 1) / 2 * 2);
#endif

        if (nT > 1)
        {
#if __aarch64__
            TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 7) / 8 * 8);
#elif __ARM_NEON
            TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 3) / 4 * 4);
#else
            TILE_M = std::min(TILE_M, (std::max(1, TILE_M / nT) + 1) / 2 * 2);
#endif
        }
    }

    if (N > 0)
    {
        int tile_size;
        if (TILE_K >= K)
        {
            tile_size = (l2_cache_size_bf16 - TILE_M * TILE_K) / TILE_K;
        }
        else
        {
            tile_size = (l2_cache_size_bf16 - TILE_M * TILE_K) / (TILE_M * 2 + TILE_K);
        }

#if __aarch64__
        TILE_N = std::max(4, tile_size / 4 * 4);
#elif __ARM_NEON
        TILE_N = std::max(4, tile_size / 4 * 4);
#else
        TILE_N = std::max(1, tile_size);
#endif

        int nn_N = (N + TILE_N - 1) / TILE_N;
#if __aarch64__
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4);
#elif __ARM_NEON
        TILE_N = std::min(TILE_N, ((N + nn_N - 1) / nn_N + 3) / 4 * 4);
#else
        TILE_N = std::min(TILE_N, (N + nn_N - 1) / nn_N);
#endif
    }
}

static void convolution_im2col_gemm_transform_kernel_bf16s(const Mat& kernel, Mat& AT, int inch, int outch, int kernel_w, int kernel_h, const Option& opt)
{
    // NCNN_LOGE("convolution_im2col_gemm_transform_kernel");
    const int maxk = kernel_w * kernel_h;

    const int M = outch;
    const int K = inch * maxk;

    int TILE_M, TILE_N, TILE_K;
    convolution_im2col_gemm_get_optimal_tile_mnk_bf16s(M, 0, K, TILE_M, TILE_N, TILE_K, opt.num_threads);

    const int nn_M = (M + TILE_M - 1) / TILE_M;

    int elempack = 1;
#if __ARM_NEON
    if (opt.use_packing_layout)
    {
        elempack = inch % 4 == 0 ? 4 : 1;
    }
#endif // __ARM_NEON

    // maxk-inch-outch to pa-maxk-inch/pa-outch
    Mat A_data;
    if (maxk == 1)
    {
        cast_float32_to_bfloat16(kernel, A_data);
        A_data = A_data.reshape(maxk * inch, outch);
    }
    else
    {
        Mat weight_data_r2 = kernel.reshape(maxk, inch, outch);

        A_data.create(maxk * inch, outch, (size_t)2u);

        for (int q = 0; q < outch; q += 1)
        {
            unsigned short* g00 = A_data.row<unsigned short>(q);

            for (int p = 0; p + (elempack - 1) < inch; p += elempack)
            {
                for (int k = 0; k < maxk; k++)
                {
                    for (int i = 0; i < elempack; i++)
                    {
                        const float* k00 = weight_data_r2.channel(q).row(p + i);
                        g00[0] = float32_to_bfloat16(k00[k]);
                        g00++;
                    }
                }
            }
        }
    }

    AT.create(TILE_K * TILE_M, (K + TILE_K - 1) / TILE_K, (M + TILE_M - 1) / TILE_M, (size_t)2u);

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        const int max_ii = std::min((M - i), TILE_M);

        for (int k = 0; k < K; k += TILE_K)
        {
            const int max_kk = std::min((K - k), TILE_K);

            Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

            convolution_im2col_pack_A_tile_bf16_fp16(A_data, AT_tile, i, max_ii, k, max_kk);
        }
    }
}

static void convolution_im2col_gemm_bf16s(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, const Mat& bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int nT, const Option& opt)
{
    const int maxk = kernel_w * kernel_h;

    const int M = top_blob.c * top_blob.elempack;
    const int N = top_blob.w * top_blob.h;
    const int K = bottom_blob.c * bottom_blob.elempack * maxk;

    int TILE_M, TILE_N, TILE_K;
    convolution_im2col_gemm_get_optimal_tile_mnk_bf16s(M, N, K, TILE_M, TILE_N, TILE_K, nT);

    const int nn_M = (M + TILE_M - 1) / TILE_M;
    const int nn_N = (N + TILE_N - 1) / TILE_N;
    const int nn_K = (K + TILE_K - 1) / TILE_K;

    // NCNN_LOGE("TILE M/N/K = %d %d %d -> %d %d %d", M, N, K, TILE_M, TILE_N, TILE_K);

    Mat BT(TILE_K * TILE_N, (K + TILE_K - 1) / TILE_K, (N + TILE_N - 1) / TILE_N, 2u, opt.workspace_allocator);

    const int nn_NK = nn_N * nn_K;

    #pragma omp parallel for num_threads(nT)
    for (int ppjk = 0; ppjk < nn_NK; ppjk++)
    {
        const int ppj = ppjk / nn_K;
        const int ppk = ppjk % nn_K;

        const int j = ppj * TILE_N;
        const int k = ppk * TILE_K;

        const int max_jj = std::min((N - j), TILE_N);
        const int max_kk = std::min((K - k), TILE_K);

        Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

        // im2col
        convolution_im2col_input_tile_bf16_fp16(bottom_blob, BT_tile, j, max_jj, k, max_kk, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);
    }

    Mat topT_tileX;
    if (K > TILE_K)
        topT_tileX.create(TILE_N * TILE_M, 1, nT, 4u, opt.workspace_allocator);

    #pragma omp parallel for num_threads(nT)
    for (int ppj = 0; ppj < nn_M; ppj++)
    {
        const int i = ppj * TILE_M;

        Mat topT_tile;
        if (K > TILE_K)
            topT_tile = topT_tileX.channel(get_omp_thread_num());

        const int max_ii = std::min((M - i), TILE_M);

        for (int j = 0; j < N; j += TILE_N)
        {
            const int max_jj = std::min((N - j), TILE_N);

            for (int k = 0; k < K; k += TILE_K)
            {
                const int max_kk = std::min((K - k), TILE_K);

                const Mat AT_tile = AT.channel(i / TILE_M).row_range(k / TILE_K, 1);

                const Mat BT_tile = BT.channel(j / TILE_N).row_range(k / TILE_K, 1);

                bool k_end = k + TILE_K >= K;

                convolution_gemm_transB_packed_tile_bf16s(AT_tile, BT_tile, bias, topT_tile, top_blob, i, max_ii, j, max_jj, k, max_kk, k_end, opt.use_a53_a55_optimized_kernel);
            }
        }
    }
}
