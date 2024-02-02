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


static void print_f32_m2(vfloat32m2_t _val, size_t l)
{
    float* ptr = (float*)malloc(l * sizeof(float));
    vse32_v_f32m2(ptr, _val, l);
    for (int i = 0; i < l; i++)
    {
        fprintf(stderr, "%f ", ptr[i]);
    }
    fprintf(stderr, "\n");
    free(ptr);
}

static void print_f16_m1(vfloat16m1_t _val, size_t l)
{
    __fp16* ptr = (__fp16*)malloc(l * sizeof(__fp16));
    vse16_v_f16m1(ptr, _val, l);
    for (int i = 0; i < l; i++)
    {
        fprintf(stderr, "%f ", (float)ptr[i]);
    }
    fprintf(stderr, "\n");
    free(ptr);
}

static void pack_A_tile_fp32_to_fp16(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    int vl;
    const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;

    __fp16* pp = AT;

    int ii = 0;
#if __riscv_vector
    for (; ii + 7 < max_ii; ii += 8)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
        const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;
        const float* p2 = (const float*)A + (i + ii + 2) * A_hstep + k;
        const float* p3 = (const float*)A + (i + ii + 3) * A_hstep + k;
        const float* p4 = (const float*)A + (i + ii + 4) * A_hstep + k;
        const float* p5 = (const float*)A + (i + ii + 5) * A_hstep + k;
        const float* p6 = (const float*)A + (i + ii + 6) * A_hstep + k;
        const float* p7 = (const float*)A + (i + ii + 7) * A_hstep + k;

        int kk = 0;

        int n = max_kk;
        while (n > 0)
        {
            vl = vsetvl_e32m2(n);
            vfloat16m1_t _r0 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p0, vl), vl);
            vfloat16m1_t _r1 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p1, vl), vl);
            vfloat16m1_t _r2 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p2, vl), vl);
            vfloat16m1_t _r3 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p3, vl), vl);
            vfloat16m1_t _r4 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p4, vl), vl);
            vfloat16m1_t _r5 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p5, vl), vl);
            vfloat16m1_t _r6 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p6, vl), vl);
            vfloat16m1_t _r7 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p7, vl), vl);
            vsseg8e16_v_f16m1(pp, _r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, vl);
            pp += 8 * vl;
            p0 += vl;
            p1 += vl;
            p2 += vl;
            p3 += vl;
            p4 += vl;
            p5 += vl;
            p6 += vl;
            p7 += vl;
            n -= vl;
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
        const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;
        const float* p2 = (const float*)A + (i + ii + 2) * A_hstep + k;
        const float* p3 = (const float*)A + (i + ii + 3) * A_hstep + k;

        int kk = 0;

        int n = max_kk;
        while (n > 0)
        {
            vl = vsetvl_e32m2(n);
            vfloat16m1_t _r0 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p0, vl), vl);
            vfloat16m1_t _r1 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p1, vl), vl);
            vfloat16m1_t _r2 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p2, vl), vl);
            vfloat16m1_t _r3 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p3, vl), vl);
            vsseg4e16_v_f16m1(pp, _r0, _r1, _r2, _r3, vl);
            pp += 4 * vl;
            p0 += vl;
            p1 += vl;
            p2 += vl;
            p3 += vl;
            n -= vl;
        }
    }
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
        const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;

        int kk = 0;
        int n = max_kk;
        while (n > 0)
        {
            vl = vsetvl_e32m2(n);
            vfloat16m1_t _r0 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p0, vl), vl);
            vfloat16m1_t _r1 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p1, vl), vl);
            vsseg2e16_v_f16m1(pp, _r0, _r1, vl);
            pp += 2 * vl;
            p0 += vl;
            p1 += vl;
            n -= vl;
        }
    }
    for (; ii < max_ii; ii += 1)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;

        int kk = 0;
        int n = max_kk;

        while (n > 0)
        {
            vl = vsetvl_e32m2(n);
            vfloat16m1_t _r0 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p0, vl), vl);
            vse16_v_f16m1(pp, _r0, vl);
            pp += vl;
            p0 += vl;
            n -= vl;
        }
    }
#endif // __riscv_vector
}

static void transpose_pack_A_tile_fp32_to_fp16(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    int vl;
    const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;

    __fp16* pp = AT;

    int ii = 0;
#if __riscv_vector
        for (; ii + 7 < max_ii; ii += 8)
    {
        vl = 8;
        const float* p0 = (const float*)A + k * A_hstep + (i + ii);

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            vfloat16m1_t _r0 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p0, vl), vl);
            vse16_v_f16m1(pp, _r0, vl);
            pp += vl;
            p0 += A_hstep;
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        vl = 4;
        const float* p0 = (const float*)A + k * A_hstep + (i + ii);

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            vfloat16m1_t _r0 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p0, vl), vl);
            vse16_v_f16m1(pp, _r0, vl);
            pp += vl;
            p0 += A_hstep;
        }
    }

    for (; ii + 1 < max_ii; ii += 2)
    {
        vl = 2;
        const float* p0 = (const float*)A + k * A_hstep + (i + ii);

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            vfloat16m1_t _r0 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p0, vl), vl);
            vse16_v_f16m1(pp, _r0, vl);
            pp += vl;
            p0 += A_hstep;
        }
    }

    for (; ii < max_ii; ii += 1)
    {
        vl = 1;
        const float* p0 = (const float*)A + k * A_hstep + (i + ii);

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            vfloat16m1_t _r0 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p0, vl), vl);
            vse16_v_f16m1(pp, _r0, vl);
            pp += vl;
            p0 += A_hstep;
        }
    }
#endif // __riscv_vector
}

static void pack_B_tile_fp32_to_fp16(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    int vl;
    const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;

    __fp16* pp = BT;

    int jj = 0;
#if __riscv_vector
    for (; jj + 11 < max_jj; jj += 12)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
        const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;
        const float* p2 = (const float*)B + (j + jj + 2) * B_hstep + k;
        const float* p3 = (const float*)B + (j + jj + 3) * B_hstep + k;
        const float* p4 = (const float*)B + (j + jj + 4) * B_hstep + k;
        const float* p5 = (const float*)B + (j + jj + 5) * B_hstep + k;
        const float* p6 = (const float*)B + (j + jj + 6) * B_hstep + k;
        const float* p7 = (const float*)B + (j + jj + 7) * B_hstep + k;
        const float* p8 = (const float*)B + (j + jj + 8) * B_hstep + k;
        const float* p9 = (const float*)B + (j + jj + 9) * B_hstep + k;
        const float* pa = (const float*)B + (j + jj + 10) * B_hstep + k;
        const float* pb = (const float*)B + (j + jj + 11) * B_hstep + k;

        int kk = 0;

        int n = max_kk;
        while (n > 0)
        {
            vl = vsetvl_e32m2(n);
            vfloat16m1_t _r0 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p0, vl), vl);
            vfloat16m1_t _r1 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p1, vl), vl);
            vfloat16m1_t _r2 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p2, vl), vl);
            vfloat16m1_t _r3 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p3, vl), vl);
            vfloat16m1_t _r4 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p4, vl), vl);
            vfloat16m1_t _r5 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p5, vl), vl);
            vfloat16m1_t _r6 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p6, vl), vl);
            vfloat16m1_t _r7 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p7, vl), vl);
            vfloat16m1_t _r8 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p8, vl), vl);
            vfloat16m1_t _r9 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p9, vl), vl);
            vfloat16m1_t _ra = vfncvt_f_f_w_f16m1(vle32_v_f32m2(pa, vl), vl);
            vfloat16m1_t _rb = vfncvt_f_f_w_f16m1(vle32_v_f32m2(pb, vl), vl);

            vsse16_v_f16m1(pp + 0, 12 * sizeof(__fp16), _r0, vl);
            vsse16_v_f16m1(pp + 1, 12 * sizeof(__fp16), _r1, vl);
            vsse16_v_f16m1(pp + 2, 12 * sizeof(__fp16), _r2, vl);
            vsse16_v_f16m1(pp + 3, 12 * sizeof(__fp16), _r3, vl);
            vsse16_v_f16m1(pp + 4, 12 * sizeof(__fp16), _r4, vl);
            vsse16_v_f16m1(pp + 5, 12 * sizeof(__fp16), _r5, vl);
            vsse16_v_f16m1(pp + 6, 12 * sizeof(__fp16), _r6, vl);
            vsse16_v_f16m1(pp + 7, 12 * sizeof(__fp16), _r7, vl);
            vsse16_v_f16m1(pp + 8, 12 * sizeof(__fp16), _r8, vl);
            vsse16_v_f16m1(pp + 9, 12 * sizeof(__fp16), _r9, vl);
            vsse16_v_f16m1(pp + 10, 12 * sizeof(__fp16), _ra, vl);
            vsse16_v_f16m1(pp + 11, 12 * sizeof(__fp16), _rb, vl);
            pp += 12 * vl;
            p0 += vl;
            p1 += vl;
            p2 += vl;
            p3 += vl;
            p4 += vl;
            p5 += vl;
            p6 += vl;
            p7 += vl;
            p8 += vl;
            p9 += vl;
            pa += vl;
            pb += vl;

            n -= vl;
        }
    }
#endif // __riscv_vector
    for (; jj + 7 < max_jj; jj += 8)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
        const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;
        const float* p2 = (const float*)B + (j + jj + 2) * B_hstep + k;
        const float* p3 = (const float*)B + (j + jj + 3) * B_hstep + k;
        const float* p4 = (const float*)B + (j + jj + 4) * B_hstep + k;
        const float* p5 = (const float*)B + (j + jj + 5) * B_hstep + k;
        const float* p6 = (const float*)B + (j + jj + 6) * B_hstep + k;
        const float* p7 = (const float*)B + (j + jj + 7) * B_hstep + k;

        int kk = 0;

        int n = max_kk;
        while (n > 0)
        {
            vl = vsetvl_e32m2(n);
            vfloat16m1_t _r0 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p0, vl), vl);
            vfloat16m1_t _r1 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p1, vl), vl);
            vfloat16m1_t _r2 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p2, vl), vl);
            vfloat16m1_t _r3 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p3, vl), vl);
            vfloat16m1_t _r4 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p4, vl), vl);
            vfloat16m1_t _r5 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p5, vl), vl);
            vfloat16m1_t _r6 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p6, vl), vl);
            vfloat16m1_t _r7 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p7, vl), vl);

            vsseg8e16_v_f16m1(pp, _r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, vl);

            pp += 8 * vl;
            p0 += vl;
            p1 += vl;
            p2 += vl;
            p3 += vl;
            p4 += vl;
            p5 += vl;
            p6 += vl;
            p7 += vl;
            n -= vl;
            print_f16_m1(_r0, vl);
            print_f16_m1(_r1, vl);
            print_f16_m1(_r2, vl);
            print_f16_m1(_r3, vl);
            print_f16_m1(_r4, vl);
            print_f16_m1(_r5, vl);
            print_f16_m1(_r6, vl);
            print_f16_m1(_r7, vl);
        }
    }
    for (; jj + 3 < max_jj; jj += 4)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
        const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;
        const float* p2 = (const float*)B + (j + jj + 2) * B_hstep + k;
        const float* p3 = (const float*)B + (j + jj + 3) * B_hstep + k;

        int kk = 0;

        int n = max_kk;

        while (n > 0)
        {
            vl = vsetvl_e32m2(n);
            vfloat16m1_t _r0 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p0, vl), vl);
            vfloat16m1_t _r1 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p1, vl), vl);
            vfloat16m1_t _r2 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p2, vl), vl);
            vfloat16m1_t _r3 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p3, vl), vl);

            vsseg4e16_v_f16m1(pp, _r0, _r1, _r2, _r3, vl);

            pp += 4 * vl;
            p0 += vl;
            p1 += vl;
            p2 += vl;
            p3 += vl;
            n -= vl;
        }
    }
    for (; jj + 1 < max_jj; jj += 2)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
        const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;

        int kk = 0;

        int n = max_kk;
        while (n > 0)
        {
            vl = vsetvl_e32m2(n);
            vfloat16m1_t _r0 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p0, vl), vl);
            vfloat16m1_t _r1 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p1, vl), vl);

            vsseg2e16_v_f16m1(pp, _r0, _r1, vl);

            pp += 2 * vl;
            p0 += vl;
            p1 += vl;
            n -= vl;
        }
    }
    for (; jj < max_jj; jj += 1)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k;

        int kk = 0;

        int n = max_kk;
        while (n > 0)
        {
            vl = vsetvl_e32m2(n);
            vfloat16m1_t _r0 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p0, vl), vl);

            vse16_v_f16m1(pp, _r0, vl);

            pp += 1 * vl;
            p0 += vl;
            n -= vl;
        }
    }
}

static void transpose_pack_B_tile_fp32_to_fp16(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    int vl;
    const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;

    __fp16* pp = BT;

    int jj = 0;

    for (; jj + 11 < max_jj; jj += 12)
    {
        vl = 12;
        const float* p0 = (const float*)B + k * B_hstep + (j + jj);

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            vfloat16m2_t _r0 = vfncvt_f_f_w_f16m2(vle32_v_f32m4(p0, vl), vl);
            vse16_v_f16m2(pp, _r0, vl);
            pp += vl;
            p0 += B_hstep;
        }
    }

    for (; jj + 7 < max_jj; jj += 8)
    {
        vl = 8;
        const float* p0 = (const float*)B + k * B_hstep + (j + jj);

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            vfloat16m1_t _r0 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p0, vl), vl);
            vse16_v_f16m1(pp, _r0, vl);
            print_f16_m1(_r0, vl);
            pp += vl;
            p0 += B_hstep;
        }
    }

    for (; jj + 3 < max_jj; jj += 4)
    {
        vl = 4;
        const float* p0 = (const float*)B + k * B_hstep + (j + jj);

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            vfloat16m1_t _r0 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p0, vl), vl);
            vse16_v_f16m1(pp, _r0, vl);
            pp += vl;
            p0 += B_hstep;
        }
    }

    for (; jj + 1 < max_jj; jj += 2)
    {
        vl = 2;
        const float* p0 = (const float*)B + k * B_hstep + (j + jj);

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            vfloat16m1_t _r0 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p0, vl), vl);
            vse16_v_f16m1(pp, _r0, vl);
            pp += vl;
            p0 += B_hstep;
        }
    }

    for (; jj < max_jj; jj += 1)
    {
        vl = 1;
        const float* p0 = (const float*)B + k * B_hstep + (j + jj);

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            vfloat16m1_t _r0 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p0, vl), vl);
            vse16_v_f16m1(pp, _r0, vl);
            pp += vl;
            p0 += B_hstep;
        }
    }
}

static void transpose_unpack_output_tile_fp32_to_fp16(const Mat& topT, Mat& top_blob, int i, int max_ii, int j, int max_jj)
{
    int vl;
    const int out_elempack = top_blob.elempack;
    const int out_hstep = top_blob.dims == 3 ? (int)top_blob.cstep : top_blob.w;

    const float* pp = topT;

    int ii = 0;
#if __riscv_vector
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (out_elempack == 4)
        {
            __fp16* p0 = (__fp16*)top_blob + j * out_hstep + (i + ii) * 4;

            for (int jj = 0; jj + 3 < max_jj; jj += 4)
            {
                vl = 8;
                vfloat16m1_t _r0 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(pp, vl), vl);
                vfloat16m1_t _r1 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(pp + 8, vl), vl);
                vfloat16m1_t _r2 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(pp + 16, vl), vl);
                vfloat16m1_t _r3 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(pp + 24, vl), vl);

                vsseg4e16_v_f16m1(p0, _r0, _r1, _r2, _r3, vl);

                pp += 32;
                p0 += out_hstep * 4;
            }
        }
        if (out_elempack == 1)
        {
            __fp16* p0 = (__fp16*)top_blob + j * out_hstep + (i + ii);

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                vl = 8;
                vfloat16m1_t _r0 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(pp, vl), vl);
                vse16_v_f16m1(p0, _r0, vl);

                pp += 8;
                p0 += out_hstep;
            }
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        if (out_elempack == 4)
        {
            __fp16* p0 = (__fp16*)top_blob + j * out_hstep + (i + ii) * 4;

            for (int jj = 0; jj + 3 < max_jj; jj += 4)
            {
                vl = 4;
                vfloat16m1_t _r0 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(pp, vl), vl);
                vfloat16m1_t _r1 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(pp + 4, vl), vl);
                vfloat16m1_t _r2 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(pp + 8, vl), vl);
                vfloat16m1_t _r3 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(pp + 12, vl), vl);

                vsseg4e16_v_f16m1(p0, _r0, _r1, _r2, _r3, vl);

                pp += 16;
                p0 += out_hstep * 4;
            }
        }
        if (out_elempack == 1)
        {
            __fp16* p0 = (__fp16*)top_blob + j * out_hstep + (i + ii);

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                vl = 4;
                vfloat16m1_t _r0 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(pp, vl), vl);
                vse16_v_f16m1(p0, _r0, vl);

                pp += 4;
                p0 += out_hstep;
            }
        }
    }
#endif // __riscv_vector
    for (; ii + 1 < max_ii; ii += 2)
    {
#if __riscv_vector
        if (out_elempack == 4)
        {
            __fp16* p0 = (__fp16*)top_blob + j * out_hstep + (i + ii) * 4;

            for (int jj = 0; jj + 3 < max_jj; jj += 4)
            {
                p0[0] = (__fp16)(pp[0]);
                p0[1] = (__fp16)(pp[2]);
                p0[2] = (__fp16)(pp[4]);
                p0[3] = (__fp16)(pp[6]);
                p0[4] = (__fp16)(pp[1]);
                p0[5] = (__fp16)(pp[3]);
                p0[6] = (__fp16)(pp[5]);
                p0[7] = (__fp16)(pp[7]);
                pp += 8;
                p0 += out_hstep * 4;
            }
        }
#endif // __riscv_vector
        if (out_elempack == 1)
        {
            __fp16* p0 = (__fp16*)top_blob + j * out_hstep + (i + ii);

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                p0[0] = (__fp16)(pp[0]);
                p0[1] = (__fp16)(pp[1]);
                pp += 2;
                p0 += out_hstep;
            }
        }
    }
    for (; ii < max_ii; ii += 1)
    {
#if __riscv_vector
        if (out_elempack == 4)
        {
            __fp16* p0 = (__fp16*)top_blob + j * out_hstep + (i + ii) * 4;

            for (int jj = 0; jj + 3 < max_jj; jj += 4)
            {
                vl = 4;
                vfloat16m1_t _r0 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(pp, vl), vl);
                vse16_v_f16m1(p0, _r0, vl);

                pp += 4;
                p0 += out_hstep * 4;
            }
        }
#endif // __riscv_vector
        if (out_elempack == 1)
        {
            __fp16* p0 = (__fp16*)top_blob + j * out_hstep + (i + ii);

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                p0[0] = (__fp16)(pp[0]);
                pp += 1;
                p0 += out_hstep;
            }
        }
    }
}

static void gemm_transB_packed_tile_fp16s(const Mat& AT_tile, const Mat& BT_tile, const Mat& CT_tile, Mat& topT_tile, Mat& top_blob, int broadcast_type_C, float alpha, int i, int max_ii, int j, int max_jj, int k, int max_kk, bool k_end)
{
    int vl;

    const int out_elempack = top_blob.elempack;
    const int out_hstep = top_blob.dims == 3 ? (int)top_blob.cstep : top_blob.w;

    const __fp16* pAT = AT_tile;
    const __fp16* pBT = BT_tile;
    const float* pBF = BT_tile;

    const float* pC = CT_tile;

    float* outptr = topT_tile;

    int ii = 0;
#if __riscv_vector
    for (; ii + 7 < max_ii; ii += 8)
    {
        __fp16* outptr0 = (__fp16*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const __fp16* pB = pBT;

        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)CT_tile + i + ii;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)CT_tile + j;
            }
        }

        int jj = 0;
        for (; jj + 11 < max_jj; jj += 12)
        {
            vfloat32m2_t _sum0;
            vfloat32m2_t _sum1;
            vfloat32m2_t _sum2;
            vfloat32m2_t _sum3;
            vfloat32m2_t _sum4;
            vfloat32m2_t _sum5;
            vfloat32m2_t _sum6;
            vfloat32m2_t _sum7;
            vfloat32m2_t _sum8;
            vfloat32m2_t _sum9;
            vfloat32m2_t _suma;
            vfloat32m2_t _sumb;

            vl = 8;

            if (k == 0)
            {
                _sum0 = vfmv_v_f_f32m2(0.f, vl);
                _sum1 = vfmv_v_f_f32m2(0.f, vl);
                _sum2 = vfmv_v_f_f32m2(0.f, vl);
                _sum3 = vfmv_v_f_f32m2(0.f, vl);
                _sum4 = vfmv_v_f_f32m2(0.f, vl);
                _sum5 = vfmv_v_f_f32m2(0.f, vl);
                _sum6 = vfmv_v_f_f32m2(0.f, vl);
                _sum7 = vfmv_v_f_f32m2(0.f, vl);
                _sum8 = vfmv_v_f_f32m2(0.f, vl);
                _sum9 = vfmv_v_f_f32m2(0.f, vl);
                _suma = vfmv_v_f_f32m2(0.f, vl);
                _sumb = vfmv_v_f_f32m2(0.f, vl);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vfmv_v_f_f32m2(pC[0], vl);
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
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vle32_v_f32m2(pC, vl);
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
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = vle32_v_f32m2(pC, vl);
                        _sum1 = vle32_v_f32m2(pC + 4 * 2, vl);
                        _sum2 = vle32_v_f32m2(pC + 4 * 4, vl);
                        _sum3 = vle32_v_f32m2(pC + 4 * 6, vl);
                        _sum4 = vle32_v_f32m2(pC + 4 * 8, vl);
                        _sum5 = vle32_v_f32m2(pC + 4 * 10, vl);
                        _sum6 = vle32_v_f32m2(pC + 4 * 12, vl);
                        _sum7 = vle32_v_f32m2(pC + 4 * 14, vl);
                        _sum8 = vle32_v_f32m2(pC + 4 * 16, vl);
                        _sum9 = vle32_v_f32m2(pC + 4 * 18, vl);
                        _suma = vle32_v_f32m2(pC + 4 * 20, vl);
                        _sumb = vle32_v_f32m2(pC + 4 * 22, vl);
                        pC += 96;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vfmv_v_f_f32m2(pC[0], vl);
                        _sum1 = vfmv_v_f_f32m2(pC[1], vl);
                        _sum2 = vfmv_v_f_f32m2(pC[2], vl);
                        _sum3 = vfmv_v_f_f32m2(pC[3], vl);
                        _sum4 = vfmv_v_f_f32m2(pC[4], vl);
                        _sum5 = vfmv_v_f_f32m2(pC[5], vl);
                        _sum6 = vfmv_v_f_f32m2(pC[6], vl);
                        _sum7 = vfmv_v_f_f32m2(pC[7], vl);
                        _sum8 = vfmv_v_f_f32m2(pC[8], vl);
                        _sum9 = vfmv_v_f_f32m2(pC[9], vl);
                        _suma = vfmv_v_f_f32m2(pC[10], vl);
                        _sumb = vfmv_v_f_f32m2(pC[11], vl);
                        pC += 12;
                    }
                }
            }
            else
            {
                _sum0 = vle32_v_f32m2(outptr, vl);
                _sum1 = vle32_v_f32m2(outptr + 4 * 2, vl);
                _sum2 = vle32_v_f32m2(outptr + 4 * 4, vl);
                _sum3 = vle32_v_f32m2(outptr + 4 * 6, vl);
                _sum4 = vle32_v_f32m2(outptr + 4 * 8, vl);
                _sum5 = vle32_v_f32m2(outptr + 4 * 10, vl);
                _sum6 = vle32_v_f32m2(outptr + 4 * 12, vl);
                _sum7 = vle32_v_f32m2(outptr + 4 * 14, vl);
                _sum8 = vle32_v_f32m2(outptr + 4 * 16, vl);
                _sum9 = vle32_v_f32m2(outptr + 4 * 18, vl);
                _suma = vle32_v_f32m2(outptr + 4 * 20, vl);
                _sumb = vle32_v_f32m2(outptr + 4 * 22, vl);
            }

            const __fp16* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat16m1_t _pA = vle16_v_f16m1(pA, vl);

                _sum0 = vfwmacc_vf_f32m2(_sum0, pB[0], _pA, vl);
                _sum1 = vfwmacc_vf_f32m2(_sum1, pB[1], _pA, vl);
                _sum2 = vfwmacc_vf_f32m2(_sum2, pB[2], _pA, vl);
                _sum3 = vfwmacc_vf_f32m2(_sum3, pB[3], _pA, vl);
                _sum4 = vfwmacc_vf_f32m2(_sum4, pB[4], _pA, vl);
                _sum5 = vfwmacc_vf_f32m2(_sum5, pB[5], _pA, vl);
                _sum6 = vfwmacc_vf_f32m2(_sum6, pB[6], _pA, vl);
                _sum7 = vfwmacc_vf_f32m2(_sum7, pB[7], _pA, vl);
                _sum8 = vfwmacc_vf_f32m2(_sum8, pB[8], _pA, vl);
                _sum9 = vfwmacc_vf_f32m2(_sum9, pB[9], _pA, vl);
                _suma = vfwmacc_vf_f32m2(_suma, pB[10], _pA, vl);
                _sumb = vfwmacc_vf_f32m2(_sumb, pB[11], _pA, vl);

                pA += 8;
                pB += 12;
            }

            if (alpha != 1.f)
            {
                _sum0 = vfmul_vf_f32m2(_sum0, alpha, vl);
                _sum1 = vfmul_vf_f32m2(_sum1, alpha, vl);
                _sum2 = vfmul_vf_f32m2(_sum2, alpha, vl);
                _sum3 = vfmul_vf_f32m2(_sum3, alpha, vl);
                _sum4 = vfmul_vf_f32m2(_sum4, alpha, vl);
                _sum5 = vfmul_vf_f32m2(_sum5, alpha, vl);
                _sum6 = vfmul_vf_f32m2(_sum6, alpha, vl);
                _sum7 = vfmul_vf_f32m2(_sum7, alpha, vl);
                _sum8 = vfmul_vf_f32m2(_sum8, alpha, vl);
                _sum9 = vfmul_vf_f32m2(_sum9, alpha, vl);
                _suma = vfmul_vf_f32m2(_suma, alpha, vl);
                _sumb = vfmul_vf_f32m2(_sumb, alpha, vl);
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vl = 4;
                    print_f32_m2(_sum0, 8);
                    print_f32_m2(_sum1, 8);
                    print_f32_m2(_sum2, 8);
                    print_f32_m2(_sum3, 8);
                    print_f32_m2(_sum4, 8);
                    print_f32_m2(_sum5, 8);
                    print_f32_m2(_sum6, 8);
                    print_f32_m2(_sum7, 8);
                    print_f32_m2(_sum8, 8);
                    print_f32_m2(_sum9, 8);
                    print_f32_m2(_suma, 8);
                    print_f32_m2(_sumb, 8);

                    vse16_v_f16m1(outptr0, vfncvt_f_f_w_f16m1(_sum0, vl), vl);
                    vse16_v_f16m1(outptr0 + 4, vfncvt_f_f_w_f16m1(_sum1, vl), vl);
                    vse16_v_f16m1(outptr0 + 4 * 2, vfncvt_f_f_w_f16m1(_sum2, vl), vl);
                    vse16_v_f16m1(outptr0 + 4 * 3, vfncvt_f_f_w_f16m1(_sum3, vl), vl);
                    vse16_v_f16m1(outptr0 + 4 * 4, vfncvt_f_f_w_f16m1(_sum4, vl), vl);
                    vse16_v_f16m1(outptr0 + 4 * 5, vfncvt_f_f_w_f16m1(_sum5, vl), vl);
                    vse16_v_f16m1(outptr0 + 4 * 6, vfncvt_f_f_w_f16m1(_sum6, vl), vl);
                    vse16_v_f16m1(outptr0 + 4 * 7, vfncvt_f_f_w_f16m1(_sum7, vl), vl);
                    vse16_v_f16m1(outptr0 + 4 * 8, vfncvt_f_f_w_f16m1(_sum8, vl), vl);
                    vse16_v_f16m1(outptr0 + 4 * 9, vfncvt_f_f_w_f16m1(_sum9, vl), vl);
                    vse16_v_f16m1(outptr0 + 4 * 10, vfncvt_f_f_w_f16m1(_suma, vl), vl);
                    vse16_v_f16m1(outptr0 + 4 * 11, vfncvt_f_f_w_f16m1(_sumb, vl), vl);

                    _sum0 = vslidedown_vx_f32m2(_sum0, _sum0, 4, vl);
                    _sum1 = vslidedown_vx_f32m2(_sum1, _sum1, 4, vl);
                    _sum2 = vslidedown_vx_f32m2(_sum2, _sum2, 4, vl);
                    _sum3 = vslidedown_vx_f32m2(_sum3, _sum3, 4, vl);
                    _sum4 = vslidedown_vx_f32m2(_sum4, _sum4, 4, vl);
                    _sum5 = vslidedown_vx_f32m2(_sum5, _sum5, 4, vl);
                    _sum6 = vslidedown_vx_f32m2(_sum6, _sum6, 4, vl);
                    _sum7 = vslidedown_vx_f32m2(_sum7, _sum7, 4, vl);
                    _sum8 = vslidedown_vx_f32m2(_sum8, _sum8, 4, vl);
                    _sum9 = vslidedown_vx_f32m2(_sum9, _sum9, 4, vl);
                    _suma = vslidedown_vx_f32m2(_suma, _suma, 4, vl);
                    _sumb = vslidedown_vx_f32m2(_sumb, _sumb, 4, vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4, vfncvt_f_f_w_f16m1(_sum0, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4 + 4, vfncvt_f_f_w_f16m1(_sum1, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4 + 4 * 2, vfncvt_f_f_w_f16m1(_sum2, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4 + 4 * 3, vfncvt_f_f_w_f16m1(_sum3, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4 + 4 * 4, vfncvt_f_f_w_f16m1(_sum4, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4 + 4 * 5, vfncvt_f_f_w_f16m1(_sum5, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4 + 4 * 6, vfncvt_f_f_w_f16m1(_sum6, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4 + 4 * 7, vfncvt_f_f_w_f16m1(_sum7, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4 + 4 * 8, vfncvt_f_f_w_f16m1(_sum8, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4 + 4 * 9, vfncvt_f_f_w_f16m1(_sum9, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4 + 4 * 10, vfncvt_f_f_w_f16m1(_suma, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4 + 4 * 11, vfncvt_f_f_w_f16m1(_sumb, vl), vl);
                    outptr0 += 48;
                }
                if (out_elempack == 1)
                {
                    vl = 8;
                    vfloat16m1_t _sum0_f16 = vfncvt_f_f_w_f16m1(_sum0, vl);
                    vfloat16m1_t _sum1_f16 = vfncvt_f_f_w_f16m1(_sum1, vl);
                    vfloat16m1_t _sum2_f16 = vfncvt_f_f_w_f16m1(_sum2, vl);
                    vfloat16m1_t _sum3_f16 = vfncvt_f_f_w_f16m1(_sum3, vl);
                    vfloat16m1_t _sum4_f16 = vfncvt_f_f_w_f16m1(_sum4, vl);
                    vfloat16m1_t _sum5_f16 = vfncvt_f_f_w_f16m1(_sum5, vl);
                    vfloat16m1_t _sum6_f16 = vfncvt_f_f_w_f16m1(_sum6, vl);
                    vfloat16m1_t _sum7_f16 = vfncvt_f_f_w_f16m1(_sum7, vl);
                    vfloat16m1_t _sum8_f16 = vfncvt_f_f_w_f16m1(_sum8, vl);
                    vfloat16m1_t _sum9_f16 = vfncvt_f_f_w_f16m1(_sum9, vl);
                    vfloat16m1_t _suma_f16 = vfncvt_f_f_w_f16m1(_suma, vl);
                    vfloat16m1_t _sumb_f16 = vfncvt_f_f_w_f16m1(_sumb, vl);

                    vsse16_v_f16m1(outptr0, out_hstep * sizeof(__fp16), _sum0_f16, vl);
                    vsse16_v_f16m1(outptr0 + 1, out_hstep * sizeof(__fp16), _sum1_f16, vl);
                    vsse16_v_f16m1(outptr0 + 2, out_hstep * sizeof(__fp16), _sum2_f16, vl);
                    vsse16_v_f16m1(outptr0 + 3, out_hstep * sizeof(__fp16), _sum3_f16, vl);
                    vsse16_v_f16m1(outptr0 + 4, out_hstep * sizeof(__fp16), _sum4_f16, vl);
                    vsse16_v_f16m1(outptr0 + 5, out_hstep * sizeof(__fp16), _sum5_f16, vl);
                    vsse16_v_f16m1(outptr0 + 6, out_hstep * sizeof(__fp16), _sum6_f16, vl);
                    vsse16_v_f16m1(outptr0 + 7, out_hstep * sizeof(__fp16), _sum7_f16, vl);
                    vsse16_v_f16m1(outptr0 + 8, out_hstep * sizeof(__fp16), _sum8_f16, vl);
                    vsse16_v_f16m1(outptr0 + 9, out_hstep * sizeof(__fp16), _sum9_f16, vl);
                    vsse16_v_f16m1(outptr0 + 10, out_hstep * sizeof(__fp16), _suma_f16, vl);
                    vsse16_v_f16m1(outptr0 + 11, out_hstep * sizeof(__fp16), _sumb_f16, vl);
                    outptr0 += 12;
                }
            }
            else
            {
                vl = 8;
                vse32_v_f32m2(outptr, _sum0, vl);
                vse32_v_f32m2(outptr + 8 * 1, _sum1, vl);
                vse32_v_f32m2(outptr + 8 * 2, _sum2, vl);
                vse32_v_f32m2(outptr + 8 * 3, _sum3, vl);
                vse32_v_f32m2(outptr + 8 * 4, _sum4, vl);
                vse32_v_f32m2(outptr + 8 * 5, _sum5, vl);
                vse32_v_f32m2(outptr + 8 * 6, _sum6, vl);
                vse32_v_f32m2(outptr + 8 * 7, _sum7, vl);
                vse32_v_f32m2(outptr + 8 * 8, _sum8, vl);
                vse32_v_f32m2(outptr + 8 * 9, _sum9, vl);
                vse32_v_f32m2(outptr + 8 * 10, _suma, vl);
                vse32_v_f32m2(outptr + 8 * 11, _sumb, vl);

            }

            outptr += 96;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            vfloat32m2_t _sum0;
            vfloat32m2_t _sum1;
            vfloat32m2_t _sum2;
            vfloat32m2_t _sum3;
            vfloat32m2_t _sum4;
            vfloat32m2_t _sum5;
            vfloat32m2_t _sum6;
            vfloat32m2_t _sum7;
            vl = 8;

            if (k == 0)
            {
                _sum0 = vfmv_v_f_f32m2(0.f, vl);
                _sum1 = vfmv_v_f_f32m2(0.f, vl);
                _sum2 = vfmv_v_f_f32m2(0.f, vl);
                _sum3 = vfmv_v_f_f32m2(0.f, vl);
                _sum4 = vfmv_v_f_f32m2(0.f, vl);
                _sum5 = vfmv_v_f_f32m2(0.f, vl);
                _sum6 = vfmv_v_f_f32m2(0.f, vl);
                _sum7 = vfmv_v_f_f32m2(0.f, vl);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vfmv_v_f_f32m2(pC[0], vl);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                        _sum4 = _sum0;
                        _sum5 = _sum0;
                        _sum6 = _sum0;
                        _sum7 = _sum0;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vle32_v_f32m2(pC, vl);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                        _sum4 = _sum0;
                        _sum5 = _sum0;
                        _sum6 = _sum0;
                        _sum7 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {

                        _sum0 = vle32_v_f32m2(pC, vl);
                        _sum1 = vle32_v_f32m2(pC + 4 * 2, vl);
                        _sum2 = vle32_v_f32m2(pC + 4 * 4, vl);
                        _sum3 = vle32_v_f32m2(pC + 4 * 6, vl);
                        _sum4 = vle32_v_f32m2(pC + 4 * 8, vl);
                        _sum5 = vle32_v_f32m2(pC + 4 * 10, vl);
                        _sum6 = vle32_v_f32m2(pC + 4 * 12, vl);
                        _sum7 = vle32_v_f32m2(pC + 4 * 14, vl);
                        pC += 64;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vfmv_v_f_f32m2(pC[0], vl);
                        _sum1 = vfmv_v_f_f32m2(pC[1], vl);
                        _sum2 = vfmv_v_f_f32m2(pC[2], vl);
                        _sum3 = vfmv_v_f_f32m2(pC[3], vl);
                        _sum4 = vfmv_v_f_f32m2(pC[4], vl);
                        _sum5 = vfmv_v_f_f32m2(pC[5], vl);
                        _sum6 = vfmv_v_f_f32m2(pC[6], vl);
                        _sum7 = vfmv_v_f_f32m2(pC[7], vl);
                        pC += 8;
                    }
                }
            }
            else
            {
                _sum0 = vle32_v_f32m2(outptr, vl);
                _sum1 = vle32_v_f32m2(outptr + 4 * 2, vl);
                _sum2 = vle32_v_f32m2(outptr + 4 * 4, vl);
                _sum3 = vle32_v_f32m2(outptr + 4 * 6, vl);
                _sum4 = vle32_v_f32m2(outptr + 4 * 8, vl);
                _sum5 = vle32_v_f32m2(outptr + 4 * 10, vl);
                _sum6 = vle32_v_f32m2(outptr + 4 * 12, vl);
                _sum7 = vle32_v_f32m2(outptr + 4 * 14, vl);
            }
            const __fp16* pA = pAT;

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat16m1_t _pA = vle16_v_f16m1(pA, vl);

                _sum0 = vfwmacc_vf_f32m2(_sum0, pB[0], _pA, vl);
                _sum1 = vfwmacc_vf_f32m2(_sum1, pB[1], _pA, vl);
                _sum2 = vfwmacc_vf_f32m2(_sum2, pB[2], _pA, vl);
                _sum3 = vfwmacc_vf_f32m2(_sum3, pB[3], _pA, vl);
                _sum4 = vfwmacc_vf_f32m2(_sum4, pB[4], _pA, vl);
                _sum5 = vfwmacc_vf_f32m2(_sum5, pB[5], _pA, vl);
                _sum6 = vfwmacc_vf_f32m2(_sum6, pB[6], _pA, vl);
                _sum7 = vfwmacc_vf_f32m2(_sum7, pB[7], _pA, vl);

                pA += 8;
                pB += 8;
            }

            if (alpha != 1.f)
            {
                _sum0 = vfmul_vf_f32m2(_sum0, alpha, vl);
                _sum1 = vfmul_vf_f32m2(_sum1, alpha, vl);
                _sum2 = vfmul_vf_f32m2(_sum2, alpha, vl);
                _sum3 = vfmul_vf_f32m2(_sum3, alpha, vl);
                _sum4 = vfmul_vf_f32m2(_sum4, alpha, vl);
                _sum5 = vfmul_vf_f32m2(_sum5, alpha, vl);
                _sum6 = vfmul_vf_f32m2(_sum6, alpha, vl);
                _sum7 = vfmul_vf_f32m2(_sum7, alpha, vl);
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    print_f32_m2(_sum0, 8);
                    print_f32_m2(_sum1, 8);
                    print_f32_m2(_sum2, 8);
                    print_f32_m2(_sum3, 8);
                    print_f32_m2(_sum4, 8);
                    print_f32_m2(_sum5, 8);
                    print_f32_m2(_sum6, 8);
                    print_f32_m2(_sum7, 8);

                    vl = 4;

                    vse16_v_f16m1(outptr0, vfncvt_f_f_w_f16m1(_sum0, vl), vl);
                    vse16_v_f16m1(outptr0 + 4, vfncvt_f_f_w_f16m1(_sum1, vl), vl);
                    vse16_v_f16m1(outptr0 + 4 * 2, vfncvt_f_f_w_f16m1(_sum2, vl), vl);
                    vse16_v_f16m1(outptr0 + 4 * 3, vfncvt_f_f_w_f16m1(_sum3, vl), vl);
                    vse16_v_f16m1(outptr0 + 4 * 4, vfncvt_f_f_w_f16m1(_sum4, vl), vl);
                    vse16_v_f16m1(outptr0 + 4 * 5, vfncvt_f_f_w_f16m1(_sum5, vl), vl);
                    vse16_v_f16m1(outptr0 + 4 * 6, vfncvt_f_f_w_f16m1(_sum6, vl), vl);
                    vse16_v_f16m1(outptr0 + 4 * 7, vfncvt_f_f_w_f16m1(_sum7, vl), vl);

                    vl = 8;

                    _sum0 = vslidedown_vx_f32m2(_sum0, _sum0, 4, vl);
                    _sum1 = vslidedown_vx_f32m2(_sum1, _sum1, 4, vl);
                    _sum2 = vslidedown_vx_f32m2(_sum2, _sum2, 4, vl);
                    _sum3 = vslidedown_vx_f32m2(_sum3, _sum3, 4, vl);
                    _sum4 = vslidedown_vx_f32m2(_sum4, _sum4, 4, vl);
                    _sum5 = vslidedown_vx_f32m2(_sum5, _sum5, 4, vl);
                    _sum6 = vslidedown_vx_f32m2(_sum6, _sum6, 4, vl);
                    _sum7 = vslidedown_vx_f32m2(_sum7, _sum7, 4, vl);


                    vl = 4;
                    vse16_v_f16m1(outptr0 + out_hstep * 4, vfncvt_f_f_w_f16m1(_sum0, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4 + 4, vfncvt_f_f_w_f16m1(_sum1, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4 + 4 * 2, vfncvt_f_f_w_f16m1(_sum2, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4 + 4 * 3, vfncvt_f_f_w_f16m1(_sum3, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4 + 4 * 4, vfncvt_f_f_w_f16m1(_sum4, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4 + 4 * 5, vfncvt_f_f_w_f16m1(_sum5, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4 + 4 * 6, vfncvt_f_f_w_f16m1(_sum6, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4 + 4 * 7, vfncvt_f_f_w_f16m1(_sum7, vl), vl);

                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    vl = 8;
                    vfloat16m1_t _sum0_f16 = vfncvt_f_f_w_f16m1(_sum0, vl);
                    vfloat16m1_t _sum1_f16 = vfncvt_f_f_w_f16m1(_sum1, vl);
                    vfloat16m1_t _sum2_f16 = vfncvt_f_f_w_f16m1(_sum2, vl);
                    vfloat16m1_t _sum3_f16 = vfncvt_f_f_w_f16m1(_sum3, vl);
                    vfloat16m1_t _sum4_f16 = vfncvt_f_f_w_f16m1(_sum4, vl);
                    vfloat16m1_t _sum5_f16 = vfncvt_f_f_w_f16m1(_sum5, vl);
                    vfloat16m1_t _sum6_f16 = vfncvt_f_f_w_f16m1(_sum6, vl);
                    vfloat16m1_t _sum7_f16 = vfncvt_f_f_w_f16m1(_sum7, vl);

                    vsse16_v_f16m1(outptr0, out_hstep * sizeof(__fp16), _sum0_f16, vl);
                    vsse16_v_f16m1(outptr0 + 1, out_hstep * sizeof(__fp16), _sum1_f16, vl);
                    vsse16_v_f16m1(outptr0 + 2, out_hstep * sizeof(__fp16), _sum2_f16, vl);
                    vsse16_v_f16m1(outptr0 + 3, out_hstep * sizeof(__fp16), _sum3_f16, vl);
                    vsse16_v_f16m1(outptr0 + 4, out_hstep * sizeof(__fp16), _sum4_f16, vl);
                    vsse16_v_f16m1(outptr0 + 5, out_hstep * sizeof(__fp16), _sum5_f16, vl);
                    vsse16_v_f16m1(outptr0 + 6, out_hstep * sizeof(__fp16), _sum6_f16, vl);
                    vsse16_v_f16m1(outptr0 + 7, out_hstep * sizeof(__fp16), _sum7_f16, vl);
        
                    outptr0 += 8;
                }
            }
            else
            {
                vl = 8;
                vse32_v_f32m2(outptr, _sum0, vl);
                vse32_v_f32m2(outptr + 4 * 2, _sum1, vl);
                vse32_v_f32m2(outptr + 4 * 4, _sum2, vl);
                vse32_v_f32m2(outptr + 4 * 6, _sum3, vl);
                vse32_v_f32m2(outptr + 4 * 8, _sum4, vl);
                vse32_v_f32m2(outptr + 4 * 10, _sum5, vl);
                vse32_v_f32m2(outptr + 4 * 12, _sum6, vl);
                vse32_v_f32m2(outptr + 4 * 14, _sum7, vl);
            }

            outptr += 64;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            vfloat32m2_t _sum0;
            vfloat32m2_t _sum1;
            vfloat32m2_t _sum2;
            vfloat32m2_t _sum3;
            vl = 8;

            if (k == 0)
            {
                _sum0 = vfmv_v_f_f32m2(0.f, vl);
                _sum1 = vfmv_v_f_f32m2(0.f, vl);
                _sum2 = vfmv_v_f_f32m2(0.f, vl);
                _sum3 = vfmv_v_f_f32m2(0.f, vl);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vfmv_v_f_f32m2(pC[0], vl);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vle32_v_f32m2(pC, vl);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = vle32_v_f32m2(pC, vl);
                        _sum1 = vle32_v_f32m2(pC + 4 * 2, vl);
                        _sum2 = vle32_v_f32m2(pC + 4 * 4, vl);
                        _sum3 = vle32_v_f32m2(pC + 4 * 6, vl);
                        pC += 32;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vfmv_v_f_f32m2(pC[0], vl);
                        _sum1 = vfmv_v_f_f32m2(pC[1], vl);
                        _sum2 = vfmv_v_f_f32m2(pC[2], vl);
                        _sum3 = vfmv_v_f_f32m2(pC[3], vl);
                        pC += 4;
                    }
                }
            }
            else
            {
                _sum0 = vle32_v_f32m2(outptr, vl);
                _sum1 = vle32_v_f32m2(outptr + 4 * 2, vl);
                _sum2 = vle32_v_f32m2(outptr + 4 * 4, vl);
                _sum3 = vle32_v_f32m2(outptr + 4 * 6, vl);
            }

            const __fp16* pA = pAT;

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat16m1_t _pA = vle16_v_f16m1(pA, vl);
                _sum0 = vfwmacc_vf_f32m2(_sum0, pB[0], _pA, vl);
                _sum1 = vfwmacc_vf_f32m2(_sum1, pB[1], _pA, vl);
                _sum2 = vfwmacc_vf_f32m2(_sum2, pB[2], _pA, vl);
                _sum3 = vfwmacc_vf_f32m2(_sum3, pB[3], _pA, vl);


                pA += 8;
                pB += 4;
            }

            if (alpha != 1.f)
            {
                _sum0 = vfmul_vf_f32m2(_sum0, alpha, vl);
                _sum1 = vfmul_vf_f32m2(_sum1, alpha, vl);
                _sum2 = vfmul_vf_f32m2(_sum2, alpha, vl);
                _sum3 = vfmul_vf_f32m2(_sum3, alpha, vl);

            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vl = 4;

                    vse16_v_f16m1(outptr0, vfncvt_f_f_w_f16m1(_sum0, vl), vl);
                    vse16_v_f16m1(outptr0 + 4, vfncvt_f_f_w_f16m1(_sum1, vl), vl);
                    vse16_v_f16m1(outptr0 + 4 * 2, vfncvt_f_f_w_f16m1(_sum2, vl), vl);
                    vse16_v_f16m1(outptr0 + 4 * 3, vfncvt_f_f_w_f16m1(_sum3, vl), vl);

                    vl = 8;

                    _sum0 = vslidedown_vx_f32m2(_sum0, _sum0, 4, vl);
                    _sum1 = vslidedown_vx_f32m2(_sum1, _sum1, 4, vl);
                    _sum2 = vslidedown_vx_f32m2(_sum2, _sum2, 4, vl);
                    _sum3 = vslidedown_vx_f32m2(_sum3, _sum3, 4, vl);

                    vl = 4;
                    vse16_v_f16m1(outptr0 + out_hstep * 4, vfncvt_f_f_w_f16m1(_sum0, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4 + 4, vfncvt_f_f_w_f16m1(_sum1, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4 + 4 * 2, vfncvt_f_f_w_f16m1(_sum2, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4 + 4 * 3, vfncvt_f_f_w_f16m1(_sum3, vl), vl);
                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    vl = 8;
                    vfloat16m1_t _sum0_f16 = vfncvt_f_f_w_f16m1(_sum0, vl);
                    vfloat16m1_t _sum1_f16 = vfncvt_f_f_w_f16m1(_sum1, vl);
                    vfloat16m1_t _sum2_f16 = vfncvt_f_f_w_f16m1(_sum2, vl);
                    vfloat16m1_t _sum3_f16 = vfncvt_f_f_w_f16m1(_sum3, vl);

                    vsse16_v_f16m1(outptr0, out_hstep * sizeof(__fp16), _sum0_f16, vl);
                    vsse16_v_f16m1(outptr0 + 1, out_hstep * sizeof(__fp16), _sum1_f16, vl);
                    vsse16_v_f16m1(outptr0 + 2, out_hstep * sizeof(__fp16), _sum2_f16, vl);
                    vsse16_v_f16m1(outptr0 + 3, out_hstep * sizeof(__fp16), _sum3_f16, vl);
                    outptr0 += 4;
                }
            }
            else
            {
                vl = 8;
                vse32_v_f32m2(outptr, _sum0, vl);
                vse32_v_f32m2(outptr + 4 * 2, _sum1, vl);
                vse32_v_f32m2(outptr + 4 * 4, _sum2, vl);
                vse32_v_f32m2(outptr + 4 * 6, _sum3, vl);
            }

            outptr += 32;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            vfloat32m2_t _sum0;
            vfloat32m2_t _sum1;

            vl = 8;

            if (k == 0)
            {
                _sum0 = vfmv_v_f_f32m2(0.f, vl);
                _sum1 = vfmv_v_f_f32m2(0.f, vl);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vfmv_v_f_f32m2(pC[0], vl);
                        _sum1 = _sum0;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vle32_v_f32m2(pC, vl);
                        _sum1 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = vle32_v_f32m2(pC, vl);
                        _sum1 = vle32_v_f32m2(pC + 4 * 2, vl);
                        pC += 16;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vfmv_v_f_f32m2(pC[0], vl);
                        _sum1 = vfmv_v_f_f32m2(pC[1], vl);
                        pC += 2;
                    }
                }
            }
            else
            {
                _sum0 = vle32_v_f32m2(outptr, vl);
                _sum1 = vle32_v_f32m2(outptr + 4 * 2, vl);
            }

            const __fp16* pA = pAT;

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat16m1_t _pA = vle16_v_f16m1(pA, vl);
                _sum0 = vfwmacc_vf_f32m2(_sum0, pB[0], _pA, vl);
                _sum1 = vfwmacc_vf_f32m2(_sum1, pB[1], _pA, vl);
                pA += 8;
                pB += 2;
            }

            if (alpha != 1.f)
            {
                _sum0 = vfmul_vf_f32m2(_sum0, alpha, vl);
                _sum1 = vfmul_vf_f32m2(_sum1, alpha, vl);

            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vl = 4;
                    vse16_v_f16m1(outptr0, vfncvt_f_f_w_f16m1(_sum0, vl), vl);
                    vse16_v_f16m1(outptr0 + 4, vfncvt_f_f_w_f16m1(_sum1, vl), vl);

                    vl = 8;

                    _sum0 = vslidedown_vx_f32m2(_sum0, _sum0, 4, vl);
                    _sum1 = vslidedown_vx_f32m2(_sum1, _sum1, 4, vl);

                    vl = 4;
                    vse16_v_f16m1(outptr0 + out_hstep * 4, vfncvt_f_f_w_f16m1(_sum0, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep * 4 + 4, vfncvt_f_f_w_f16m1(_sum1, vl), vl);

                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
                    vl = 8;
                    vfloat16m1_t _sum0_f16 = vfncvt_f_f_w_f16m1(_sum0, vl);
                    vfloat16m1_t _sum1_f16 = vfncvt_f_f_w_f16m1(_sum1, vl);

                    vsse16_v_f16m1(outptr0, out_hstep * sizeof(__fp16), _sum0_f16, vl);
                    vsse16_v_f16m1(outptr0 + 1, out_hstep * sizeof(__fp16), _sum1_f16, vl);

                    outptr0 += 2;
                }
            }
            else
            {
                vl = 8;
                vse32_v_f32m2(outptr, _sum0, vl);
                vse32_v_f32m2(outptr + 4 * 2, _sum1, vl);
            }

            outptr += 16;
        }
        for (; jj < max_jj; jj += 1)
        {
            vfloat32m2_t _sum0;
            vl = 8;

            if (k == 0)
            {
                _sum0 = vfmv_v_f_f32m2(0.f, vl);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vfmv_v_f_f32m2(pC[0], vl);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vle32_v_f32m2(pC, vl);
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = vle32_v_f32m2(pC, vl);
                        pC += 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vfmv_v_f_f32m2(pC[0], vl);
                        pC += 1;
                    }
                }
            }
            else
            {
                _sum0 = vle32_v_f32m2(outptr, vl);
            }

            const __fp16* pA = pAT;

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat16m1_t _pA = vle16_v_f16m1(pA, vl);

                _sum0 = vfwmacc_vf_f32m2(_sum0, pB[0], _pA, vl);
                pA += 8;
                pB += 1;
            }

            if (alpha != 1.f)
            {
                _sum0 = vfmul_vf_f32m2(_sum0, alpha, vl);
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vl = 4;
                    vse16_v_f16m1(outptr0, vfncvt_f_f_w_f16m1(_sum0, vl), vl);
                    vl = 8;
                    _sum0 = vslidedown_vx_f32m2(_sum0, _sum0, 4, vl);
                    vl = 4;
                    vse16_v_f16m1(outptr0 + out_hstep * 4, vfncvt_f_f_w_f16m1(_sum0, vl), vl);
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
                    vl = 8;
                    vfloat16m1_t _sum0_f16 = vfncvt_f_f_w_f16m1(_sum0, vl);
                    vsse16_v_f16m1(outptr0, out_hstep * sizeof(__fp16), _sum0_f16, vl);

                    outptr0++;
                }
            }
            else
            {
                vse32_v_f32m2(outptr, _sum0, vl);
            }

            outptr += 8;
        }

        pAT += max_kk * 8;
    }
#endif // __riscv_vector
    for (; ii + 3 < max_ii; ii += 4)
    {
        __fp16* outptr0 = (__fp16*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const __fp16* pB = pBT;

        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)CT_tile + i + ii;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)CT_tile + j;
            }
        }

        int jj = 0;
#if __riscv_vector
        for (; jj + 11 < max_jj; jj += 12)
        {
            vfloat32m2_t _sum0;
            vfloat32m2_t _sum1;
            vfloat32m2_t _sum2;
            vfloat32m2_t _sum3;
            vfloat32m2_t _sum4;
            vfloat32m2_t _sum5;
            vfloat32m2_t _sum6;
            vfloat32m2_t _sum7;
            vfloat32m2_t _sum8;
            vfloat32m2_t _sum9;
            vfloat32m2_t _suma;
            vfloat32m2_t _sumb;
            vl = 4;

            if (k == 0)
            {
                _sum0 = vfmv_v_f_f32m2(0.f, vl);
                _sum1 = vfmv_v_f_f32m2(0.f, vl);
                _sum2 = vfmv_v_f_f32m2(0.f, vl);
                _sum3 = vfmv_v_f_f32m2(0.f, vl);
                _sum4 = vfmv_v_f_f32m2(0.f, vl);
                _sum5 = vfmv_v_f_f32m2(0.f, vl);
                _sum6 = vfmv_v_f_f32m2(0.f, vl);
                _sum7 = vfmv_v_f_f32m2(0.f, vl);
                _sum8 = vfmv_v_f_f32m2(0.f, vl);
                _sum9 = vfmv_v_f_f32m2(0.f, vl);
                _suma = vfmv_v_f_f32m2(0.f, vl);
                _sumb = vfmv_v_f_f32m2(0.f, vl);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vfmv_v_f_f32m2(pC[0], vl);
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
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vle32_v_f32m2(pC, vl);
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
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = vle32_v_f32m2(pC, vl);
                        _sum1 = vle32_v_f32m2(pC + 4, vl);
                        _sum2 = vle32_v_f32m2(pC + 8, vl);
                        _sum3 = vle32_v_f32m2(pC + 12, vl);
                        _sum4 = vle32_v_f32m2(pC + 16, vl);
                        _sum5 = vle32_v_f32m2(pC + 20, vl);
                        _sum6 = vle32_v_f32m2(pC + 24, vl);
                        _sum7 = vle32_v_f32m2(pC + 28, vl);
                        _sum8 = vle32_v_f32m2(pC + 32, vl);
                        _sum9 = vle32_v_f32m2(pC + 36, vl);
                        _suma = vle32_v_f32m2(pC + 40, vl);
                        _sumb = vle32_v_f32m2(pC + 44, vl);
                        pC += 48;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vfmv_v_f_f32m2(pC[0], vl);
                        _sum1 = vfmv_v_f_f32m2(pC[1], vl);
                        _sum2 = vfmv_v_f_f32m2(pC[2], vl);
                        _sum3 = vfmv_v_f_f32m2(pC[3], vl);
                        _sum4 = vfmv_v_f_f32m2(pC[4], vl);
                        _sum5 = vfmv_v_f_f32m2(pC[5], vl);
                        _sum6 = vfmv_v_f_f32m2(pC[6], vl);
                        _sum7 = vfmv_v_f_f32m2(pC[7], vl);
                        _sum8 = vfmv_v_f_f32m2(pC[8], vl);
                        _sum9 = vfmv_v_f_f32m2(pC[9], vl);
                        _suma = vfmv_v_f_f32m2(pC[10], vl);
                        _sumb = vfmv_v_f_f32m2(pC[11], vl);
                        pC += 12;
                    }
                }
            }
            else
            {
                _sum0 = vle32_v_f32m2(outptr, vl);
                _sum1 = vle32_v_f32m2(outptr + 4 * 1, vl);
                _sum2 = vle32_v_f32m2(outptr + 4 * 2, vl);
                _sum3 = vle32_v_f32m2(outptr + 4 * 3, vl);
                _sum4 = vle32_v_f32m2(outptr + 4 * 4, vl);
                _sum5 = vle32_v_f32m2(outptr + 4 * 5, vl);
                _sum6 = vle32_v_f32m2(outptr + 4 * 6, vl);
                _sum7 = vle32_v_f32m2(outptr + 4 * 7, vl);
                _sum8 = vle32_v_f32m2(outptr + 4 * 8, vl);
                _sum9 = vle32_v_f32m2(outptr + 4 * 9, vl);
                _suma = vle32_v_f32m2(outptr + 4 * 10, vl);
                _sumb = vle32_v_f32m2(outptr + 4 * 11, vl);
            }

            const __fp16* pA = pAT;

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat16m1_t _pA = vle16_v_f16m1(pA, vl);

                _sum0 = vfwmacc_vf_f32m2(_sum0, pB[0], _pA, vl);
                _sum1 = vfwmacc_vf_f32m2(_sum1, pB[1], _pA, vl);
                _sum2 = vfwmacc_vf_f32m2(_sum2, pB[2], _pA, vl);
                _sum3 = vfwmacc_vf_f32m2(_sum3, pB[3], _pA, vl);
                _sum4 = vfwmacc_vf_f32m2(_sum4, pB[4], _pA, vl);
                _sum5 = vfwmacc_vf_f32m2(_sum5, pB[5], _pA, vl);
                _sum6 = vfwmacc_vf_f32m2(_sum6, pB[6], _pA, vl);
                _sum7 = vfwmacc_vf_f32m2(_sum7, pB[7], _pA, vl);
                _sum8 = vfwmacc_vf_f32m2(_sum8, pB[8], _pA, vl);
                _sum9 = vfwmacc_vf_f32m2(_sum9, pB[9], _pA, vl);
                _suma = vfwmacc_vf_f32m2(_suma, pB[10], _pA, vl);
                _sumb = vfwmacc_vf_f32m2(_sumb, pB[11], _pA, vl);

                pA += 4;
                pB += 12;
            }

            if (alpha != 1.f)
            {
                _sum0 = vfmul_vf_f32m2(_sum0, alpha, vl);
                _sum1 = vfmul_vf_f32m2(_sum1, alpha, vl);
                _sum2 = vfmul_vf_f32m2(_sum2, alpha, vl);
                _sum3 = vfmul_vf_f32m2(_sum3, alpha, vl);
                _sum4 = vfmul_vf_f32m2(_sum4, alpha, vl);
                _sum5 = vfmul_vf_f32m2(_sum5, alpha, vl);
                _sum6 = vfmul_vf_f32m2(_sum6, alpha, vl);
                _sum7 = vfmul_vf_f32m2(_sum7, alpha, vl);
                _sum8 = vfmul_vf_f32m2(_sum8, alpha, vl);
                _sum9 = vfmul_vf_f32m2(_sum9, alpha, vl);
                _suma = vfmul_vf_f32m2(_suma, alpha, vl);
                _sumb = vfmul_vf_f32m2(_sumb, alpha, vl);

            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vse16_v_f16m1(outptr0, vfncvt_f_f_w_f16m1(_sum0, vl), vl);
                    vse16_v_f16m1(outptr0 + 4, vfncvt_f_f_w_f16m1(_sum1, vl), vl);
                    vse16_v_f16m1(outptr0 + 4 * 2, vfncvt_f_f_w_f16m1(_sum2, vl), vl);
                    vse16_v_f16m1(outptr0 + 4 * 3, vfncvt_f_f_w_f16m1(_sum3, vl), vl);
                    vse16_v_f16m1(outptr0 + 4 * 4, vfncvt_f_f_w_f16m1(_sum4, vl), vl);
                    vse16_v_f16m1(outptr0 + 4 * 5, vfncvt_f_f_w_f16m1(_sum5, vl), vl);
                    vse16_v_f16m1(outptr0 + 4 * 6, vfncvt_f_f_w_f16m1(_sum6, vl), vl);
                    vse16_v_f16m1(outptr0 + 4 * 7, vfncvt_f_f_w_f16m1(_sum7, vl), vl);
                    vse16_v_f16m1(outptr0 + 4 * 8, vfncvt_f_f_w_f16m1(_sum8, vl), vl);
                    vse16_v_f16m1(outptr0 + 4 * 9, vfncvt_f_f_w_f16m1(_sum9, vl), vl);
                    vse16_v_f16m1(outptr0 + 4 * 10, vfncvt_f_f_w_f16m1(_suma, vl), vl);
                    vse16_v_f16m1(outptr0 + 4 * 11, vfncvt_f_f_w_f16m1(_sumb, vl), vl);

                    outptr0 += 48;
                }
                if (out_elempack == 1)
                {
                    vfloat16m1_t _sum0_f16 = vfncvt_f_f_w_f16m1(_sum0, vl);
                    vfloat16m1_t _sum1_f16 = vfncvt_f_f_w_f16m1(_sum1, vl);
                    vfloat16m1_t _sum2_f16 = vfncvt_f_f_w_f16m1(_sum2, vl);
                    vfloat16m1_t _sum3_f16 = vfncvt_f_f_w_f16m1(_sum3, vl);
                    vfloat16m1_t _sum4_f16 = vfncvt_f_f_w_f16m1(_sum4, vl);
                    vfloat16m1_t _sum5_f16 = vfncvt_f_f_w_f16m1(_sum5, vl);
                    vfloat16m1_t _sum6_f16 = vfncvt_f_f_w_f16m1(_sum6, vl);
                    vfloat16m1_t _sum7_f16 = vfncvt_f_f_w_f16m1(_sum7, vl);
                    vfloat16m1_t _sum8_f16 = vfncvt_f_f_w_f16m1(_sum8, vl);
                    vfloat16m1_t _sum9_f16 = vfncvt_f_f_w_f16m1(_sum9, vl);
                    vfloat16m1_t _suma_f16 = vfncvt_f_f_w_f16m1(_suma, vl);
                    vfloat16m1_t _sumb_f16 = vfncvt_f_f_w_f16m1(_sumb, vl);

                    vsse16_v_f16m1(outptr0, out_hstep * sizeof(__fp16), _sum0_f16, vl);
                    vsse16_v_f16m1(outptr0 + 1, out_hstep * sizeof(__fp16), _sum1_f16, vl);
                    vsse16_v_f16m1(outptr0 + 2, out_hstep * sizeof(__fp16), _sum2_f16, vl);
                    vsse16_v_f16m1(outptr0 + 3, out_hstep * sizeof(__fp16), _sum3_f16, vl);
                    vsse16_v_f16m1(outptr0 + 4, out_hstep * sizeof(__fp16), _sum4_f16, vl);
                    vsse16_v_f16m1(outptr0 + 5, out_hstep * sizeof(__fp16), _sum5_f16, vl);
                    vsse16_v_f16m1(outptr0 + 6, out_hstep * sizeof(__fp16), _sum6_f16, vl);
                    vsse16_v_f16m1(outptr0 + 7, out_hstep * sizeof(__fp16), _sum7_f16, vl);
                    vsse16_v_f16m1(outptr0 + 8, out_hstep * sizeof(__fp16), _sum8_f16, vl);
                    vsse16_v_f16m1(outptr0 + 9, out_hstep * sizeof(__fp16), _sum9_f16, vl);
                    vsse16_v_f16m1(outptr0 + 10, out_hstep * sizeof(__fp16), _suma_f16, vl);
                    vsse16_v_f16m1(outptr0 + 11, out_hstep * sizeof(__fp16), _sumb_f16, vl);

                    outptr0 += 12;
                }
            }
            else
            {
                vse32_v_f32m2(outptr, _sum0, vl);
                vse32_v_f32m2(outptr + 4, _sum1, vl);
                vse32_v_f32m2(outptr + 4 * 2, _sum2, vl);
                vse32_v_f32m2(outptr + 4 * 3, _sum3, vl);
                vse32_v_f32m2(outptr + 4 * 4, _sum4, vl);
                vse32_v_f32m2(outptr + 4 * 5, _sum5, vl);
                vse32_v_f32m2(outptr + 4 * 6, _sum6, vl);
                vse32_v_f32m2(outptr + 4 * 7, _sum7, vl);
                vse32_v_f32m2(outptr + 4 * 8, _sum8, vl);
                vse32_v_f32m2(outptr + 4 * 9, _sum9, vl);
                vse32_v_f32m2(outptr + 4 * 10, _suma, vl);
                vse32_v_f32m2(outptr + 4 * 11, _sumb, vl);
            }

            outptr += 48;
        }
#endif // __riscv_vector
        for (; jj + 7 < max_jj; jj += 8)
        {
            vfloat32m2_t _sum0;
            vfloat32m2_t _sum1;
            vfloat32m2_t _sum2;
            vfloat32m2_t _sum3;
            vfloat32m2_t _sum4;
            vfloat32m2_t _sum5;
            vfloat32m2_t _sum6;
            vfloat32m2_t _sum7;
            vl = 4;

            if (k == 0)
            {
                _sum0 = vfmv_v_f_f32m2(0.f, vl);
                _sum1 = vfmv_v_f_f32m2(0.f, vl);
                _sum2 = vfmv_v_f_f32m2(0.f, vl);
                _sum3 = vfmv_v_f_f32m2(0.f, vl);
                _sum4 = vfmv_v_f_f32m2(0.f, vl);
                _sum5 = vfmv_v_f_f32m2(0.f, vl);
                _sum6 = vfmv_v_f_f32m2(0.f, vl);
                _sum7 = vfmv_v_f_f32m2(0.f, vl);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vfmv_v_f_f32m2(pC[0], vl);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                        _sum4 = _sum0;
                        _sum5 = _sum0;
                        _sum6 = _sum0;
                        _sum7 = _sum0;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vle32_v_f32m2(pC, vl);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                        _sum4 = _sum0;
                        _sum5 = _sum0;
                        _sum6 = _sum0;
                        _sum7 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = vle32_v_f32m2(pC, vl);
                        _sum1 = vle32_v_f32m2(pC + 4, vl);
                        _sum2 = vle32_v_f32m2(pC + 8, vl);
                        _sum3 = vle32_v_f32m2(pC + 12, vl);
                        _sum4 = vle32_v_f32m2(pC + 16, vl);
                        _sum5 = vle32_v_f32m2(pC + 20, vl);
                        _sum6 = vle32_v_f32m2(pC + 24, vl);
                        _sum7 = vle32_v_f32m2(pC + 28, vl);
                        pC += 32;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vfmv_v_f_f32m2(pC[0], vl);
                        _sum1 = vfmv_v_f_f32m2(pC[1], vl);
                        _sum2 = vfmv_v_f_f32m2(pC[2], vl);
                        _sum3 = vfmv_v_f_f32m2(pC[3], vl);
                        _sum4 = vfmv_v_f_f32m2(pC[4], vl);
                        _sum5 = vfmv_v_f_f32m2(pC[5], vl);
                        _sum6 = vfmv_v_f_f32m2(pC[6], vl);
                        _sum7 = vfmv_v_f_f32m2(pC[7], vl);
                        pC += 8;
                    }
                }
            }
            else
            {
                _sum0 = vle32_v_f32m2(outptr, vl);
                _sum1 = vle32_v_f32m2(outptr + 4 * 1, vl);
                _sum2 = vle32_v_f32m2(outptr + 4 * 2, vl);
                _sum3 = vle32_v_f32m2(outptr + 4 * 3, vl);
                _sum4 = vle32_v_f32m2(outptr + 4 * 4, vl);
                _sum5 = vle32_v_f32m2(outptr + 4 * 5, vl);
                _sum6 = vle32_v_f32m2(outptr + 4 * 6, vl);
                _sum7 = vle32_v_f32m2(outptr + 4 * 7, vl);
            }

            const __fp16* pA = pAT;

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat16m1_t _pA = vle16_v_f16m1(pA, vl);

                _sum0 = vfwmacc_vf_f32m2(_sum0, pB[0], _pA, vl);
                _sum1 = vfwmacc_vf_f32m2(_sum1, pB[1], _pA, vl);
                _sum2 = vfwmacc_vf_f32m2(_sum2, pB[2], _pA, vl);
                _sum3 = vfwmacc_vf_f32m2(_sum3, pB[3], _pA, vl);
                _sum4 = vfwmacc_vf_f32m2(_sum4, pB[4], _pA, vl);
                _sum5 = vfwmacc_vf_f32m2(_sum5, pB[5], _pA, vl);
                _sum6 = vfwmacc_vf_f32m2(_sum6, pB[6], _pA, vl);
                _sum7 = vfwmacc_vf_f32m2(_sum7, pB[7], _pA, vl);

                pA += 4;
                pB += 8;
            }

            if (alpha != 1.f)
            {
                _sum0 = vfmul_vf_f32m2(_sum0, alpha, vl);
                _sum1 = vfmul_vf_f32m2(_sum1, alpha, vl);
                _sum2 = vfmul_vf_f32m2(_sum2, alpha, vl);
                _sum3 = vfmul_vf_f32m2(_sum3, alpha, vl);
                _sum4 = vfmul_vf_f32m2(_sum4, alpha, vl);
                _sum5 = vfmul_vf_f32m2(_sum5, alpha, vl);
                _sum6 = vfmul_vf_f32m2(_sum6, alpha, vl);
                _sum7 = vfmul_vf_f32m2(_sum7, alpha, vl);

            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vse16_v_f16m1(outptr0, vfncvt_f_f_w_f16m1(_sum0, vl), vl);
                    vse16_v_f16m1(outptr0 + 4, vfncvt_f_f_w_f16m1(_sum1, vl), vl);
                    vse16_v_f16m1(outptr0 + 4 * 2, vfncvt_f_f_w_f16m1(_sum2, vl), vl);
                    vse16_v_f16m1(outptr0 + 4 * 3, vfncvt_f_f_w_f16m1(_sum3, vl), vl);
                    vse16_v_f16m1(outptr0 + 4 * 4, vfncvt_f_f_w_f16m1(_sum4, vl), vl);
                    vse16_v_f16m1(outptr0 + 4 * 5, vfncvt_f_f_w_f16m1(_sum5, vl), vl);
                    vse16_v_f16m1(outptr0 + 4 * 6, vfncvt_f_f_w_f16m1(_sum6, vl), vl);
                    vse16_v_f16m1(outptr0 + 4 * 7, vfncvt_f_f_w_f16m1(_sum7, vl), vl);

                                                                                                                                            outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    vfloat16m1_t _sum0_f16 = vfncvt_f_f_w_f16m1(_sum0, vl);
                    vfloat16m1_t _sum1_f16 = vfncvt_f_f_w_f16m1(_sum1, vl);
                    vfloat16m1_t _sum2_f16 = vfncvt_f_f_w_f16m1(_sum2, vl);
                    vfloat16m1_t _sum3_f16 = vfncvt_f_f_w_f16m1(_sum3, vl);
                    vfloat16m1_t _sum4_f16 = vfncvt_f_f_w_f16m1(_sum4, vl);
                    vfloat16m1_t _sum5_f16 = vfncvt_f_f_w_f16m1(_sum5, vl);
                    vfloat16m1_t _sum6_f16 = vfncvt_f_f_w_f16m1(_sum6, vl);
                    vfloat16m1_t _sum7_f16 = vfncvt_f_f_w_f16m1(_sum7, vl);

                    vsse16_v_f16m1(outptr0, out_hstep * sizeof(__fp16), _sum0_f16, vl);
                    vsse16_v_f16m1(outptr0 + 1, out_hstep * sizeof(__fp16), _sum1_f16, vl);
                    vsse16_v_f16m1(outptr0 + 2, out_hstep * sizeof(__fp16), _sum2_f16, vl);
                    vsse16_v_f16m1(outptr0 + 3, out_hstep * sizeof(__fp16), _sum3_f16, vl);
                    vsse16_v_f16m1(outptr0 + 4, out_hstep * sizeof(__fp16), _sum4_f16, vl);
                    vsse16_v_f16m1(outptr0 + 5, out_hstep * sizeof(__fp16), _sum5_f16, vl);
                    vsse16_v_f16m1(outptr0 + 6, out_hstep * sizeof(__fp16), _sum6_f16, vl);
                    vsse16_v_f16m1(outptr0 + 7, out_hstep * sizeof(__fp16), _sum7_f16, vl);

                    // transpose4x8_ps(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7);

                                                                                                                                                                                    outptr0 += 8;
                }
            }
            else
            {
                vse32_v_f32m2(outptr, _sum0, vl);
                vse32_v_f32m2(outptr + 4, _sum1, vl);
                vse32_v_f32m2(outptr + 4 * 2, _sum2, vl);
                vse32_v_f32m2(outptr + 4 * 3, _sum3, vl);
                vse32_v_f32m2(outptr + 4 * 4, _sum4, vl);
                vse32_v_f32m2(outptr + 4 * 5, _sum5, vl);
                vse32_v_f32m2(outptr + 4 * 6, _sum6, vl);
                vse32_v_f32m2(outptr + 4 * 7, _sum7, vl);
            }

            outptr += 32;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            vfloat32m2_t _sum0;
            vfloat32m2_t _sum1;
            vfloat32m2_t _sum2;
            vfloat32m2_t _sum3;

            vl = 4;

            if (k == 0)
            {
                _sum0 = vfmv_v_f_f32m2(0.f, vl);
                _sum1 = vfmv_v_f_f32m2(0.f, vl);
                _sum2 = vfmv_v_f_f32m2(0.f, vl);
                _sum3 = vfmv_v_f_f32m2(0.f, vl);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vfmv_v_f_f32m2(pC[0], vl);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vle32_v_f32m2(pC, vl);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = vle32_v_f32m2(pC, vl);
                        _sum1 = vle32_v_f32m2(pC + 4, vl);
                        _sum2 = vle32_v_f32m2(pC + 8, vl);
                        _sum3 = vle32_v_f32m2(pC + 12, vl);
                        pC += 16;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vfmv_v_f_f32m2(pC[0], vl);
                        _sum1 = vfmv_v_f_f32m2(pC[1], vl);
                        _sum2 = vfmv_v_f_f32m2(pC[2], vl);
                        _sum3 = vfmv_v_f_f32m2(pC[3], vl);
                        pC += 4;
                    }
                }
            }
            else
            {
                _sum0 = vle32_v_f32m2(outptr, vl);
                _sum1 = vle32_v_f32m2(outptr + 4 * 1, vl);
                _sum2 = vle32_v_f32m2(outptr + 4 * 2, vl);
                _sum3 = vle32_v_f32m2(outptr + 4 * 3, vl);
            }
            // __fp16 tmp_result0[4];
            // __fp16 tmp_result1[4];
            // __fp16 tmp_result2[4];
            // __fp16 tmp_result3[4];

                                                
            // // printf 4 array
            // printf("tmp_result0: %f, %f, %f, %f\n", tmp_result0[0], tmp_result0[1], tmp_result0[2], tmp_result0[3]);
            // printf("tmp_result1: %f, %f, %f, %f\n", tmp_result1[0], tmp_result1[1], tmp_result1[2], tmp_result1[3]);
            // printf("tmp_result2: %f, %f, %f, %f\n", tmp_result2[0], tmp_result2[1], tmp_result2[2], tmp_result2[3]);
            // printf("tmp_result3: %f, %f, %f, %f\n", tmp_result3[0], tmp_result3[1], tmp_result3[2], tmp_result3[3]);

            const __fp16* pA = pAT;

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat16m1_t _pA = vle16_v_f16m1(pA, vl);

                _sum0 = vfwmacc_vf_f32m2(_sum0, pB[0], _pA, vl);
                _sum1 = vfwmacc_vf_f32m2(_sum1, pB[1], _pA, vl);
                _sum2 = vfwmacc_vf_f32m2(_sum2, pB[2], _pA, vl);
                _sum3 = vfwmacc_vf_f32m2(_sum3, pB[3], _pA, vl);

                                
                
                                                                
                pA += 4;
                pB += 4;
            }

            if (alpha != 1.f)
            {
                _sum0 = vfmul_vf_f32m2(_sum0, alpha, vl);
                _sum1 = vfmul_vf_f32m2(_sum1, alpha, vl);
                _sum2 = vfmul_vf_f32m2(_sum2, alpha, vl);
                _sum3 = vfmul_vf_f32m2(_sum3, alpha, vl);
                                                                                            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vse16_v_f16m1(outptr0, vfncvt_f_f_w_f16m1(_sum0, vl), vl);
                    vse16_v_f16m1(outptr0 + 4, vfncvt_f_f_w_f16m1(_sum1, vl), vl);
                    vse16_v_f16m1(outptr0 + 4 * 2, vfncvt_f_f_w_f16m1(_sum2, vl), vl);
                    vse16_v_f16m1(outptr0 + 4 * 3, vfncvt_f_f_w_f16m1(_sum3, vl), vl);

                                                                                                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    vfloat16m1_t _sum0_f16 = vfncvt_f_f_w_f16m1(_sum0, vl);
                    vfloat16m1_t _sum1_f16 = vfncvt_f_f_w_f16m1(_sum1, vl);
                    vfloat16m1_t _sum2_f16 = vfncvt_f_f_w_f16m1(_sum2, vl);
                    vfloat16m1_t _sum3_f16 = vfncvt_f_f_w_f16m1(_sum3, vl);

                    vsse16_v_f16m1(outptr0, out_hstep * sizeof(__fp16), _sum0_f16, vl);
                    vsse16_v_f16m1(outptr0 + 1, out_hstep * sizeof(__fp16), _sum1_f16, vl);
                    vsse16_v_f16m1(outptr0 + 2, out_hstep * sizeof(__fp16), _sum2_f16, vl);
                    vsse16_v_f16m1(outptr0 + 3, out_hstep * sizeof(__fp16), _sum3_f16, vl);
                    // transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);

                                                                                                    outptr0 += 4;
                }
            }
            else
            {
                vse32_v_f32m2(outptr, _sum0, vl);
                vse32_v_f32m2(outptr + 4, _sum1, vl);
                vse32_v_f32m2(outptr + 4 * 2, _sum2, vl);
                vse32_v_f32m2(outptr + 4 * 3, _sum3, vl);
            }

            outptr += 16;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            vfloat32m2_t _sum0;
            vfloat32m2_t _sum1;

            vl = 4;

            if (k == 0)
            {
                _sum0 = vfmv_v_f_f32m2(0.f, vl);
                _sum1 = vfmv_v_f_f32m2(0.f, vl);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vfmv_v_f_f32m2(pC[0], vl);
                        _sum1 = _sum0;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vle32_v_f32m2(pC, vl);
                        _sum1 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = vle32_v_f32m2(pC, vl);
                        _sum1 = vle32_v_f32m2(pC + 4, vl);
                        pC += 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vfmv_v_f_f32m2(pC[0], vl);
                        _sum1 = vfmv_v_f_f32m2(pC[1], vl);
                        pC += 2;
                    }
                }
            }
            else
            {
                _sum0 = vle32_v_f32m2(outptr, vl);
                _sum1 = vle32_v_f32m2(outptr + 4, vl);
            }

            const __fp16* pA = pAT;

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat16m1_t _pA = vle16_v_f16m1(pA, vl);

                _sum0 = vfwmacc_vf_f32m2(_sum0, pB[0], _pA, vl);
                _sum1 = vfwmacc_vf_f32m2(_sum1, pB[1], _pA, vl);
                                
                                                
                                
                pA += 4;
                pB += 2;
            }

            if (alpha != 1.f)
            {
                _sum0 = vfmul_vf_f32m2(_sum0, alpha, vl);
                _sum1 = vfmul_vf_f32m2(_sum1, alpha, vl);

                                                            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vse16_v_f16m1(outptr0, vfncvt_f_f_w_f16m1(_sum0, vl), vl);
                    vse16_v_f16m1(outptr0 + 4, vfncvt_f_f_w_f16m1(_sum1, vl), vl);
                                                            outptr0 += 8;
                }
                if (out_elempack == 1)
                {
                    vfloat16m1_t _sum0_f16 = vfncvt_f_f_w_f16m1(_sum0, vl);
                    vfloat16m1_t _sum1_f16 = vfncvt_f_f_w_f16m1(_sum1, vl);
                    vsse16_v_f16m1(outptr0, out_hstep * sizeof(__fp16), _sum0_f16, vl);
                    vsse16_v_f16m1(outptr0 + 1, out_hstep * sizeof(__fp16), _sum1_f16, vl);
                    outptr0 += 2;
                }
            }
            else
            {
                vse32_v_f32m2(outptr, _sum0, vl);
                vse32_v_f32m2(outptr + 4, _sum1, vl);
            }

            outptr += 8;
        }
        for (; jj < max_jj; jj += 1)
        {
            vfloat32m2_t _sum0;

            vl = 4;

            if (k == 0)
            {
                _sum0 = vfmv_v_f_f32m2(0.f, vl);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vfmv_v_f_f32m2(pC[0], vl);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vle32_v_f32m2(pC, vl);
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = vle32_v_f32m2(pC, vl);
                        pC += 4;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vfmv_v_f_f32m2(pC[0], vl);
                        pC += 1;
                    }
                }
            }
            else
            {
                _sum0 = vle32_v_f32m2(outptr, vl);
            }

            const __fp16* pA = pAT;

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat16m1_t _pA = vle16_v_f16m1(pA, vl);

                _sum0 = vfwmacc_vf_f32m2(_sum0, pB[0], _pA, vl);

                                
                
                
                pA += 4;
                pB += 1;
            }

            if (alpha != 1.f)
            {
                _sum0 = vfmul_vf_f32m2(_sum0, alpha, vl);
                                            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vse16_v_f16m1(outptr0, vfncvt_f_f_w_f16m1(_sum0, vl), vl);
                                        outptr0 += 4;
                }
                if (out_elempack == 1)
                {
                    vfloat16m1_t _sum0_f16 = vfncvt_f_f_w_f16m1(_sum0, vl);
                    vsse16_v_f16m1(outptr0, out_hstep * sizeof(__fp16), _sum0_f16, vl);

                    outptr0++;
                }
            }
            else
            {
                vse32_v_f32m2(outptr, _sum0, vl);
            }

            outptr += 4;
        }

        pAT += max_kk * 4;
    }
    for (; ii + 1 < max_ii; ii += 2)
    {
        __fp16* outptr0 = (__fp16*)top_blob + (i + ii) * out_hstep + j;

        const __fp16* pB = pBT;

        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)CT_tile + i + ii;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)CT_tile + j;
            }
        }

        int jj = 0;
#if __riscv_vector
        for (; jj + 11 < max_jj; jj += 12)
        {
            vfloat32m2_t _sum00;
            vfloat32m2_t _sum01;
            vfloat32m2_t _sum02;
            vfloat32m2_t _sum10;
            vfloat32m2_t _sum11;
            vfloat32m2_t _sum12;

            vl = 4;

            if (k == 0)
            {
                _sum00 = vfmv_v_f_f32m2(0.f, vl);
                _sum01 = vfmv_v_f_f32m2(0.f, vl);
                _sum02 = vfmv_v_f_f32m2(0.f, vl);
                _sum10 = vfmv_v_f_f32m2(0.f, vl);
                _sum11 = vfmv_v_f_f32m2(0.f, vl);
                _sum12 = vfmv_v_f_f32m2(0.f, vl);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum00 = vfmv_v_f_f32m2(pC[0], vl);
                        _sum01 = _sum00;
                        _sum02 = _sum00;
                        _sum10 = _sum00;
                        _sum11 = _sum00;
                        _sum12 = _sum00;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum00 = vfmv_v_f_f32m2(pC[0], vl);
                        _sum01 = _sum00;
                        _sum02 = _sum00;
                        _sum10 = vfmv_v_f_f32m2(pC[1], vl);
                        _sum11 = _sum10;
                        _sum12 = _sum10;
                    }
                    if (broadcast_type_C == 3)
                    {
                        vlseg2e32_v_f32m2(&_sum00, &_sum10, pC, vl);
                        vlseg2e32_v_f32m2(&_sum01, &_sum11, pC + 8, vl);
                        vlseg2e32_v_f32m2(&_sum02, &_sum12, pC + 16, vl);

                                                                                                                                                                        pC += 24;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum00 = vle32_v_f32m2(pC, vl);
                        _sum01 = vle32_v_f32m2(pC + 4, vl);
                        _sum02 = vle32_v_f32m2(pC + 8, vl);
                        _sum10 = _sum00;
                        _sum11 = _sum01;
                        _sum12 = _sum02;
                        pC += 12;
                    }
                }
            }
            else
            {
                vlseg2e32_v_f32m2(&_sum00, &_sum10, outptr, vl);
                vlseg2e32_v_f32m2(&_sum01, &_sum11, outptr + 8, vl);
                vlseg2e32_v_f32m2(&_sum02, &_sum12, outptr + 16, vl);
                // float32x4x2_t _tmp01 = vld2q_f32(outptr);
                // float32x4x2_t _tmp23 = vld2q_f32(outptr + 8);
                // float32x4x2_t _tmp45 = vld2q_f32(outptr + 16);
                                                                                                            }

            const __fp16* pA = pAT;

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat16m1_t _pB0 = vle16_v_f16m1(pB, vl);
                vfloat16m1_t _pB1 = vle16_v_f16m1(pB + 4, vl);
                vfloat16m1_t _pB2 = vle16_v_f16m1(pB + 8, vl);

                _sum00 = vfwmacc_vf_f32m2(_sum00, pA[0], _pB0, vl);
                _sum01 = vfwmacc_vf_f32m2(_sum01, pA[0], _pB1, vl);
                _sum02 = vfwmacc_vf_f32m2(_sum02, pA[0], _pB2, vl);
                _sum10 = vfwmacc_vf_f32m2(_sum10, pA[1], _pB0, vl);
                _sum11 = vfwmacc_vf_f32m2(_sum11, pA[1], _pB1, vl);
                _sum12 = vfwmacc_vf_f32m2(_sum12, pA[1], _pB2, vl);

                                                
                                                                
                                                                                                
                pA += 2;
                pB += 12;
            }

            if (alpha != 1.f)
            {
                _sum00 = vfmul_vf_f32m2(_sum00, alpha, vl);
                _sum01 = vfmul_vf_f32m2(_sum01, alpha, vl);
                _sum02 = vfmul_vf_f32m2(_sum02, alpha, vl);
                _sum10 = vfmul_vf_f32m2(_sum10, alpha, vl);
                _sum11 = vfmul_vf_f32m2(_sum11, alpha, vl);
                _sum12 = vfmul_vf_f32m2(_sum12, alpha, vl);
                                                                                                                            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vse16_v_f16m1(outptr0, vfncvt_f_f_w_f16m1(_sum00, vl), vl);
                    vse16_v_f16m1(outptr0 + 4, vfncvt_f_f_w_f16m1(_sum01, vl), vl);
                    vse16_v_f16m1(outptr0 + 8, vfncvt_f_f_w_f16m1(_sum02, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep, vfncvt_f_f_w_f16m1(_sum10, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep + 4, vfncvt_f_f_w_f16m1(_sum11, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep + 8, vfncvt_f_f_w_f16m1(_sum12, vl), vl);
                                                                                                                                            outptr0 += 12;
                }
            }
            else
            {
                vsseg2e32_v_f32m1(outptr, vget_v_f32m2_f32m1(_sum00, 0), vget_v_f32m2_f32m1(_sum10, 0), vl);
                vsseg2e32_v_f32m1(outptr + 8, vget_v_f32m2_f32m1(_sum01, 0), vget_v_f32m2_f32m1(_sum11, 0), vl);
                vsseg2e32_v_f32m1(outptr + 16, vget_v_f32m2_f32m1(_sum02, 0), vget_v_f32m2_f32m1(_sum12, 0), vl);

                //                 vsseg2e16_v_f32m1(outptr, vget_v_f32m2_f32m1(_sum00, 0), vget_v_f32m2_f32m1(_sum10, 0), vl);
                                                // float32x4x2_t _tmp01;
                // _tmp01.val[0] = _sum0;
                // _tmp01.val[1] = _sum1;
                // float32x4x2_t _tmp23;
                // _tmp23.val[0] = _sum01;
                // _tmp23.val[1] = _sum11;
                // float32x4x2_t _tmp45;
                // _tmp45.val[0] = _sum02;
                // _tmp45.val[1] = _sum12;
                                                            }

            outptr += 24;
        }
#endif // __riscv_vector
        for (; jj + 7 < max_jj; jj += 8)
        {
            vfloat32m2_t _sum0;
            vfloat32m2_t _sum1;

            vl = 8;

            if (k == 0)
            {
                _sum0 = vfmv_v_f_f32m2(0.f, vl);
                _sum1 = vfmv_v_f_f32m2(0.f, vl);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vfmv_v_f_f32m2(pC[0], vl);
                        _sum1 = _sum0;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vfmv_v_f_f32m2(pC[0], vl);
                        _sum1 = vfmv_v_f_f32m2(pC[1], vl);
                    }
                    if (broadcast_type_C == 3)
                    {
                        vfloat32m1_t _tmp0;
                        vfloat32m1_t _tmp1;
                        vfloat32m1_t _tmp2;
                        vfloat32m1_t _tmp3;

                        vlseg2e32_v_f32m1(&_tmp0, &_tmp1, pC, vl);
                        vlseg2e32_v_f32m1(&_tmp2, &_tmp3, pC + 8, vl);

                        _sum0 = vset_v_f32m1_f32m2(_sum0, 0, _tmp0);
                        _sum0 = vset_v_f32m1_f32m2(_sum0, 1, _tmp2);
                        _sum1 = vset_v_f32m1_f32m2(_sum1, 0, _tmp1);
                        _sum1 = vset_v_f32m1_f32m2(_sum1, 1, _tmp3);
                        // float32x4x2_t _tmp01 = vld2q_f32(pC);
                        // float32x4x2_t _tmp23 = vld2q_f32(pC + 8);
                                                                                                                        pC += 16;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vle32_v_f32m2(pC, vl);
                        _sum1 = _sum0;
                                                                                                                        pC += 8;
                    }
                }
            }
            else
            {
                vl = 4;
                vfloat32m1_t _tmp0;
                vfloat32m1_t _tmp1;
                vfloat32m1_t _tmp2;
                vfloat32m1_t _tmp3;

                vlseg2e32_v_f32m1(&_tmp0, &_tmp1, outptr, vl);
                vlseg2e32_v_f32m1(&_tmp2, &_tmp3, outptr + 8, vl);

                _sum0 = vset_v_f32m1_f32m2(_sum0, 0, _tmp0);
                _sum0 = vset_v_f32m1_f32m2(_sum0, 1, _tmp2);
                _sum1 = vset_v_f32m1_f32m2(_sum1, 0, _tmp1);
                _sum1 = vset_v_f32m1_f32m2(_sum1, 1, _tmp3);
                // float32x4x2_t _tmp01 = vld2q_f32(outptr);
                // float32x4x2_t _tmp23 = vld2q_f32(outptr + 8);
                                                                            }
            vl = 8;

            const __fp16* pA = pAT;

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat16m1_t _pB0 = vle16_v_f16m1(pB, vl);
                _sum0 = vfwmacc_vf_f32m2(_sum0, pA[0], _pB0, vl);
                _sum1 = vfwmacc_vf_f32m2(_sum1, pA[1], _pB0, vl);
                
                                                                
                                                                
                pA += 2;
                pB += 8;
            }

            if (alpha != 1.f)
            {
                _sum0 = vfmul_vf_f32m2(_sum0, alpha, vl);
                _sum1 = vfmul_vf_f32m2(_sum1, alpha, vl);
                                                                                            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vse16_v_f16m1(outptr0, vfncvt_f_f_w_f16m1(_sum0, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep, vfncvt_f_f_w_f16m1(_sum1, vl), vl);

                                                                                                    outptr0 += 8;
                }
            }
            else
            {
                // spmm? size => 32x32x micro kernel
                vfloat32m1_t _tmp00 = vget_v_f32m2_f32m1(_sum0, 0);
                vfloat32m1_t _tmp01 = vget_v_f32m2_f32m1(_sum1, 0);
                vfloat32m1_t _tmp10 = vget_v_f32m2_f32m1(_sum0, 1);
                vfloat32m1_t _tmp11 = vget_v_f32m2_f32m1(_sum1, 1);
                vl = 4;
                vsseg2e32_v_f32m1(outptr, _tmp00, _tmp01, vl);
                vsseg2e32_v_f32m1(outptr + 8, _tmp10, _tmp11, vl);
                // float32x4x2_t _tmp01;
                // _tmp01.val[0] = _sum00;
                // _tmp01.val[1] = _sum10;
                // float32x4x2_t _tmp23;
                // _tmp23.val[0] = _sum01;
                // _tmp23.val[1] = _sum11;
                                            }
            outptr += 16;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            vfloat32m2_t _sum0;
            vfloat32m2_t _sum1;

            vl = 4;

            if (k == 0)
            {
                _sum0 = vfmv_v_f_f32m2(0.f, vl);
                _sum1 = vfmv_v_f_f32m2(0.f, vl);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vfmv_v_f_f32m2(pC[0], vl);
                        _sum1 = _sum0;
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vfmv_v_f_f32m2(pC[0], vl);
                        _sum1 = vfmv_v_f_f32m2(pC[1], vl);
                    }
                    if (broadcast_type_C == 3)
                    {
                        vfloat32m1_t _tmp0;
                        vfloat32m1_t _tmp1;
                        vlseg2e32_v_f32m1(&_tmp0, &_tmp1, pC, vl);
                        _sum0 = vset_v_f32m1_f32m2(_sum0, 0, _tmp0);
                        _sum1 = vset_v_f32m1_f32m2(_sum1, 0, _tmp1);
                        // float32x4x2_t _tmp01 = vld2q_f32(pC);
                                                                        pC += 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vle32_v_f32m2(pC, vl);
                        _sum1 = _sum0;
                        pC += 4;
                    }
                }
            }
            else
            {
                vfloat32m1_t _tmp0;
                vfloat32m1_t _tmp1;
                vlseg2e32_v_f32m1(&_tmp0, &_tmp1, outptr, vl);
                _sum0 = vset_v_f32m1_f32m2(_sum0, 0, _tmp0);
                _sum1 = vset_v_f32m1_f32m2(_sum1, 0, _tmp1);
                                                // float32x4x2_t _tmp01 = vuzpq_f32(_tmp0, _tmp1);
                                            }

            const __fp16* pA = pAT;

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat16m1_t _pB = vle16_v_f16m1(pB, vl);
                
                _sum0 = vfwmacc_vf_f32m2(_sum0, pA[0], _pB, vl);
                _sum1 = vfwmacc_vf_f32m2(_sum1, pA[1], _pB, vl);
                // _pB0 = vslideup_vx_f16m1(_pB0, 4, vl);

                                
                                                
                                
                pA += 2;
                pB += 4;
            }

            if (alpha != 1.f)
            {
                _sum0 = vfmul_vf_f32m2(_sum0, alpha, vl);
                _sum1 = vfmul_vf_f32m2(_sum1, alpha, vl);
                                                            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vse16_v_f16m1(outptr0, vfncvt_f_f_w_f16m1(_sum0, vl), vl);
                    vse16_v_f16m1(outptr0 + out_hstep, vfncvt_f_f_w_f16m1(_sum1, vl), vl);
                                                            outptr0 += 4;
                }
            }
            else
            {
                vsseg2e32_v_f32m2(outptr, _sum0, _sum1, vl);
                // float32x4x2_t _tmp01;
                // _tmp01.val[0] = _sum0;
                // _tmp01.val[1] = _sum1;
                            }

            outptr += 8;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float sum00;
            float sum01;
            float sum10;
            float sum11;

            if (k == 0)
            {
                sum00 = 0.f;
                sum01 = 0.f;
                sum10 = 0.f;
                sum11 = 0.f;

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        sum00 = pC[0];
                        sum01 = pC[0];
                        sum10 = pC[0];
                        sum11 = pC[0];
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        sum00 = pC[0];
                        sum01 = pC[1];
                        sum10 = pC[0];
                        sum11 = pC[1];
                    }
                    if (broadcast_type_C == 3)
                    {
                        sum00 = pC[0];
                        sum01 = pC[1];
                        sum10 = pC[2];
                        sum11 = pC[3];
                        pC += 4;
                    }
                    if (broadcast_type_C == 4)
                    {
                        sum00 = pC[0];
                        sum01 = pC[0];
                        sum10 = pC[1];
                        sum11 = pC[1];
                        pC += 2;
                    }
                }
            }
            else
            {
                sum00 = outptr[0];
                sum01 = outptr[1];
                sum10 = outptr[2];
                sum11 = outptr[3];
            }

            const __fp16* pA = pAT;

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __fp16 pA0 = pA[0];
                __fp16 pA1 = pA[1];
                __fp16 pB0 = pB[0];
                __fp16 pB1 = pB[1];

                sum00 += pA0 * pB0;
                sum01 += pA1 * pB0;
                sum10 += pA0 * pB1;
                sum11 += pA1 * pB1;

                pA += 2;
                pB += 2;
            }

            if (alpha != 1.f)
            {
                sum00 *= alpha;
                sum01 *= alpha;
                sum10 *= alpha;
                sum11 *= alpha;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = (__fp16)(sum00);
                    outptr0[1] = (__fp16)(sum10);
                    outptr0[out_hstep] = (__fp16)(sum01);
                    outptr0[out_hstep + 1] = (__fp16)(sum11);
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
            float _sum0;
            float _sum1;

            if (k == 0)
            {
                _sum0 = 0.f;
                _sum1 = 0.f;

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = pC[0];
                        _sum1 = pC[0];
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = pC[0];
                        _sum1 = pC[1];
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = pC[0];
                        _sum1 = pC[1];
                        pC += 2;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = pC[0];
                        _sum1 = pC[0];
                        pC += 1;
                    }
                }
            }
            else
            {
                _sum0 = outptr[0];
                _sum1 = outptr[1];
            }

            const __fp16* pA = pAT;

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __fp16 pA0 = pA[0];
                __fp16 pA1 = pA[1];
                __fp16 pB0 = pB[0];

                _sum0 += pA0 * pB0;
                _sum1 += pA1 * pB0;
                pA += 2;
                pB += 1;
            }

            if (alpha != 1.f)
            {
                _sum0 *= alpha;
                _sum1 *= alpha;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = (__fp16)(_sum0);
                    outptr0[out_hstep] = (__fp16)(_sum1);
                    outptr0++;
                }
            }
            else
            {
                outptr[0] = _sum0;
                outptr[1] = _sum1;
            }

            outptr += 2;
        }

        pAT += max_kk * 2;
    }
    for (; ii < max_ii; ii += 1)
    {
        __fp16* outptr0 = (__fp16*)top_blob + (i + ii) * out_hstep + j;

        const __fp16* pB = pBT;

        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)CT_tile + i + ii;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)CT_tile + j;
            }
        }

        int jj = 0;
#if __riscv_vector
        for (; jj + 11 < max_jj; jj += 12)
        {
            vfloat32m2_t _sum0;
            vfloat32m2_t _sum1;
            vfloat32m2_t _sum2;

            vl = 4;

            if (k == 0)
            {
                _sum0 = vfmv_v_f_f32m2(0.f, vl);
                _sum1 = vfmv_v_f_f32m2(0.f, vl);
                _sum2 = vfmv_v_f_f32m2(0.f, vl);

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vfmv_v_f_f32m2(pC[0], vl);
                        _sum1 = vfmv_v_f_f32m2(pC[0], vl);
                        _sum2 = vfmv_v_f_f32m2(pC[0], vl);
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        _sum0 = vle32_v_f32m2(pC, vl);
                        _sum1 = vle32_v_f32m2(pC + 4, vl);
                        _sum2 = vle32_v_f32m2(pC + 8, vl);
                        pC += 12;
                    }
                }
            }
            else
            {
                _sum0 = vle32_v_f32m2(outptr, vl);
                _sum1 = vle32_v_f32m2(outptr + 4, vl);
                _sum2 = vle32_v_f32m2(outptr + 8, vl);
            }

            const __fp16* pA = pAT;

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat16m1_t _pB0 = vle16_v_f16m1(pB, vl);
                vfloat16m1_t _pB1 = vle16_v_f16m1(pB + 4, vl);
                vfloat16m1_t _pB2 = vle16_v_f16m1(pB + 8, vl);

                _sum0 = vfwmacc_vf_f32m2(_sum0, pA[0], _pB0, vl);
                _sum1 = vfwmacc_vf_f32m2(_sum1, pA[0], _pB1, vl);
                _sum2 = vfwmacc_vf_f32m2(_sum2, pA[0], _pB2, vl);

                                                
                
                                                
                pA += 1;
                pB += 12;
            }

            if (alpha != 1.f)
            {
                _sum0 = vfmul_vf_f32m2(_sum0, alpha, vl);
                _sum1 = vfmul_vf_f32m2(_sum1, alpha, vl);
                _sum2 = vfmul_vf_f32m2(_sum2, alpha, vl);
                                                                            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vse16_v_f16m1(outptr0, vfncvt_f_f_w_f16m1(_sum0, vl), vl);
                    vse16_v_f16m1(outptr0 + 4, vfncvt_f_f_w_f16m1(_sum1, vl), vl);
                    vse16_v_f16m1(outptr0 + 8, vfncvt_f_f_w_f16m1(_sum2, vl), vl);
                                                                                outptr0 += 12;
                }
            }
            else
            {
                vse32_v_f32m2(outptr, _sum0, vl);
                vse32_v_f32m2(outptr + 4, _sum1, vl);
                vse32_v_f32m2(outptr + 8, _sum2, vl);
            }

            outptr += 12;
        }
#endif // __riscv_vector
        for (; jj + 7 < max_jj; jj += 8)
        {
            vfloat32m2_t _sum0;
            vfloat32m2_t _sum1;
            vl = 4;

            if (k == 0)
            {
                _sum0 = vfmv_v_f_f32m2(0.f, vl);
                _sum1 = vfmv_v_f_f32m2(0.f, vl);

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vfmv_v_f_f32m2(pC[0], vl);
                        _sum1 = vfmv_v_f_f32m2(pC[0], vl);
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        _sum0 = vle32_v_f32m2(pC, vl);
                        _sum1 = vle32_v_f32m2(pC + 4, vl);
                        pC += 8;
                    }
                }
            }
            else
            {
                _sum0 = vle32_v_f32m2(outptr, vl);
                _sum1 = vle32_v_f32m2(outptr + 4, vl);
            }

            const __fp16* pA = pAT;

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat16m1_t _pB0 = vle16_v_f16m1(pB, vl);
                vfloat16m1_t _pB1 = vle16_v_f16m1(pB + 4, vl);

                _sum0 = vfwmacc_vf_f32m2(_sum0, pA[0], _pB0, vl);
                _sum1 = vfwmacc_vf_f32m2(_sum1, pA[0], _pB1, vl);

                                
                                
                pA += 1;
                pB += 8;
            }

            if (alpha != 1.f)
            {
                _sum0 = vfmul_vf_f32m2(_sum0, alpha, vl);
                _sum1 = vfmul_vf_f32m2(_sum1, alpha, vl);
                                                            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vse16_v_f16m1(outptr0, vfncvt_f_f_w_f16m1(_sum0, vl), vl);
                    vse16_v_f16m1(outptr0 + 4, vfncvt_f_f_w_f16m1(_sum1, vl), vl);
                                                            outptr0 += 8;
                }
            }
            else
            {
                vse32_v_f32m2(outptr, _sum0, vl);
                vse32_v_f32m2(outptr + 4, _sum1, vl);
            }

            outptr += 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            vfloat32m2_t _sum;
            vl = 4;

            if (k == 0)
            {
                _sum = vfmv_v_f_f32m2(0.f, vl);

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum = vfmv_v_f_f32m2(pC[0], vl);
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        _sum = vle32_v_f32m2(pC, vl);
                        pC += 4;
                    }
                }
            }
            else
            {
                _sum = vle32_v_f32m2(outptr, vl);
            }

            const __fp16* pA = pAT;

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat16m1_t _pB = vle16_v_f16m1(pB, vl);

                _sum = vfwmacc_vf_f32m2(_sum, pA[0], _pB, vl);
                                                
                
                pA += 1;
                pB += 4;
            }

            if (alpha != 1.f)
            {
                _sum = vfmul_vf_f32m2(_sum, alpha, vl);
                                            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vse16_v_f16m1(outptr0, vfncvt_f_f_w_f16m1(_sum, vl), vl);
                                        outptr0 += 4;
                }
            }
            else
            {
                vse32_v_f32m2(outptr, _sum, vl);
            }

            outptr += 4;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float _sum0;
            float _sum1;

            if (k == 0)
            {
                _sum0 = 0.f;
                _sum1 = 0.f;

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = pC[0];
                        _sum1 = pC[0];
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        _sum0 = pC[0];
                        _sum1 = pC[1];
                        pC += 2;
                    }
                }
            }
            else
            {
                _sum0 = outptr[0];
                _sum1 = outptr[1];
            }

            const __fp16* pA = pAT;

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __fp16 pA0 = pA[0];
                __fp16 pB0 = pB[0];
                __fp16 pB1 = pB[1];

                _sum0 += pA0 * pB0;
                _sum1 += pA0 * pB1;

                pA += 1;
                pB += 2;
            }

            if (alpha != 1.f)
            {
                _sum0 *= alpha;
                _sum1 *= alpha;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = (__fp16)(_sum0);
                    outptr0[1] = (__fp16)(_sum1);
                    outptr0 += 2;
                }
            }
            else
            {
                outptr[0] = _sum0;
                outptr[1] = _sum1;
            }

            outptr += 2;
        }
        for (; jj < max_jj; jj += 1)
        {
            float sum;

            if (k == 0)
            {
                sum = 0.f;

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        sum = pC[0];
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        sum = pC[0];
                        pC += 1;
                    }
                }
            }
            else
            {
                sum = outptr[0];
            }

            const __fp16* pA = pAT;

            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                __fp16 pA0 = pA[0];
                __fp16 pB0 = pB[0];

                sum += pA0 * pB0;
                pA += 1;
                pB += 1;
            }

            if (alpha != 1.f)
            {
                sum *= alpha;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = (__fp16)(sum);
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
