// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static void convdw3x3s1_packn_int8_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Option& opt)
{
    const int packn = csrr_vlenb();
    const int out_elempack = top_blob.elempack;
    const size_t vl = __riscv_vsetvl_e8m1(packn);
    const size_t vl32 = __riscv_vsetvl_e32m1(out_elempack);

    int outw = top_blob.w;
    int outh = top_blob.h;
    const int group = bottom_blob.c;

    // int8 packn accumulates into four int32 packs
    const int total_gi = group * outh;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int gi = 0; gi < total_gi; gi++)
    {
        int g = gi / outh;
        int i = gi % outh;

        int* outptr0 = top_blob.channel(g * 4).row<int>(i);
        int* outptr1 = top_blob.channel(g * 4 + 1).row<int>(i);
        int* outptr2 = top_blob.channel(g * 4 + 2).row<int>(i);
        int* outptr3 = top_blob.channel(g * 4 + 3).row<int>(i);

        const Mat img = bottom_blob.channel(g);
        const signed char* r0 = img.row<const signed char>(i);
        const signed char* r1 = img.row<const signed char>(i + 1);
        const signed char* r2 = img.row<const signed char>(i + 2);
        const signed char* k0 = kernel.row<const signed char>(g);

        vint16m2_t _k00 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(k0, vl), 0, vl);
        vint16m2_t _k01 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(k0 + packn, vl), 0, vl);
        vint16m2_t _k02 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(k0 + packn * 2, vl), 0, vl);
        vint16m2_t _k10 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(k0 + packn * 3, vl), 0, vl);
        vint16m2_t _k11 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(k0 + packn * 4, vl), 0, vl);
        vint16m2_t _k12 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(k0 + packn * 5, vl), 0, vl);
        vint16m2_t _k20 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(k0 + packn * 6, vl), 0, vl);
        vint16m2_t _k21 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(k0 + packn * 7, vl), 0, vl);
        vint16m2_t _k22 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(k0 + packn * 8, vl), 0, vl);

        int j = 0;
        for (; j + 1 < outw; j += 2)
        {
            vint32m4_t _sum0 = __riscv_vmv_v_x_i32m4(0, vl);
            vint32m4_t _sum1 = __riscv_vmv_v_x_i32m4(0, vl);

            vint16m2_t _r00 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(r0, vl), 0, vl);
            _sum0 = __riscv_vwmacc_vv_i32m4(_sum0, _k00, _r00, vl);
            vint16m2_t _r01 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(r0 + packn, vl), 0, vl);
            _sum0 = __riscv_vwmacc_vv_i32m4(_sum0, _k01, _r01, vl);
            _sum1 = __riscv_vwmacc_vv_i32m4(_sum1, _k00, _r01, vl);
            vint16m2_t _r02 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(r0 + packn * 2, vl), 0, vl);
            _sum0 = __riscv_vwmacc_vv_i32m4(_sum0, _k02, _r02, vl);
            _sum1 = __riscv_vwmacc_vv_i32m4(_sum1, _k01, _r02, vl);
            vint16m2_t _r03 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(r0 + packn * 3, vl), 0, vl);
            _sum1 = __riscv_vwmacc_vv_i32m4(_sum1, _k02, _r03, vl);

            vint16m2_t _r10 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(r1, vl), 0, vl);
            _sum0 = __riscv_vwmacc_vv_i32m4(_sum0, _k10, _r10, vl);
            vint16m2_t _r11 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(r1 + packn, vl), 0, vl);
            _sum0 = __riscv_vwmacc_vv_i32m4(_sum0, _k11, _r11, vl);
            _sum1 = __riscv_vwmacc_vv_i32m4(_sum1, _k10, _r11, vl);
            vint16m2_t _r12 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(r1 + packn * 2, vl), 0, vl);
            _sum0 = __riscv_vwmacc_vv_i32m4(_sum0, _k12, _r12, vl);
            _sum1 = __riscv_vwmacc_vv_i32m4(_sum1, _k11, _r12, vl);
            vint16m2_t _r13 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(r1 + packn * 3, vl), 0, vl);
            _sum1 = __riscv_vwmacc_vv_i32m4(_sum1, _k12, _r13, vl);

            vint16m2_t _r20 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(r2, vl), 0, vl);
            _sum0 = __riscv_vwmacc_vv_i32m4(_sum0, _k20, _r20, vl);
            vint16m2_t _r21 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(r2 + packn, vl), 0, vl);
            _sum0 = __riscv_vwmacc_vv_i32m4(_sum0, _k21, _r21, vl);
            _sum1 = __riscv_vwmacc_vv_i32m4(_sum1, _k20, _r21, vl);
            vint16m2_t _r22 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(r2 + packn * 2, vl), 0, vl);
            _sum0 = __riscv_vwmacc_vv_i32m4(_sum0, _k22, _r22, vl);
            _sum1 = __riscv_vwmacc_vv_i32m4(_sum1, _k21, _r22, vl);
            vint16m2_t _r23 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(r2 + packn * 3, vl), 0, vl);
            _sum1 = __riscv_vwmacc_vv_i32m4(_sum1, _k22, _r23, vl);

            __riscv_vse32_v_i32m1(outptr0, __riscv_vget_v_i32m4_i32m1(_sum0, 0), vl32);
            __riscv_vse32_v_i32m1(outptr0 + out_elempack, __riscv_vget_v_i32m4_i32m1(_sum1, 0), vl32);
            __riscv_vse32_v_i32m1(outptr1, __riscv_vget_v_i32m4_i32m1(_sum0, 1), vl32);
            __riscv_vse32_v_i32m1(outptr1 + out_elempack, __riscv_vget_v_i32m4_i32m1(_sum1, 1), vl32);
            __riscv_vse32_v_i32m1(outptr2, __riscv_vget_v_i32m4_i32m1(_sum0, 2), vl32);
            __riscv_vse32_v_i32m1(outptr2 + out_elempack, __riscv_vget_v_i32m4_i32m1(_sum1, 2), vl32);
            __riscv_vse32_v_i32m1(outptr3, __riscv_vget_v_i32m4_i32m1(_sum0, 3), vl32);
            __riscv_vse32_v_i32m1(outptr3 + out_elempack, __riscv_vget_v_i32m4_i32m1(_sum1, 3), vl32);

            outptr0 += out_elempack * 2;
            outptr1 += out_elempack * 2;
            outptr2 += out_elempack * 2;
            outptr3 += out_elempack * 2;
            r0 += packn * 2;
            r1 += packn * 2;
            r2 += packn * 2;
        }
        for (; j < outw; j++)
        {
            vint32m4_t _sum = __riscv_vmv_v_x_i32m4(0, vl);

            vint16m2_t _r00 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(r0, vl), 0, vl);
            _sum = __riscv_vwmacc_vv_i32m4(_sum, _k00, _r00, vl);
            vint16m2_t _r01 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(r0 + packn, vl), 0, vl);
            _sum = __riscv_vwmacc_vv_i32m4(_sum, _k01, _r01, vl);
            vint16m2_t _r02 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(r0 + packn * 2, vl), 0, vl);
            _sum = __riscv_vwmacc_vv_i32m4(_sum, _k02, _r02, vl);

            vint16m2_t _r10 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(r1, vl), 0, vl);
            _sum = __riscv_vwmacc_vv_i32m4(_sum, _k10, _r10, vl);
            vint16m2_t _r11 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(r1 + packn, vl), 0, vl);
            _sum = __riscv_vwmacc_vv_i32m4(_sum, _k11, _r11, vl);
            vint16m2_t _r12 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(r1 + packn * 2, vl), 0, vl);
            _sum = __riscv_vwmacc_vv_i32m4(_sum, _k12, _r12, vl);

            vint16m2_t _r20 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(r2, vl), 0, vl);
            _sum = __riscv_vwmacc_vv_i32m4(_sum, _k20, _r20, vl);
            vint16m2_t _r21 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(r2 + packn, vl), 0, vl);
            _sum = __riscv_vwmacc_vv_i32m4(_sum, _k21, _r21, vl);
            vint16m2_t _r22 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(r2 + packn * 2, vl), 0, vl);
            _sum = __riscv_vwmacc_vv_i32m4(_sum, _k22, _r22, vl);

            __riscv_vse32_v_i32m1(outptr0, __riscv_vget_v_i32m4_i32m1(_sum, 0), vl32);
            __riscv_vse32_v_i32m1(outptr1, __riscv_vget_v_i32m4_i32m1(_sum, 1), vl32);
            __riscv_vse32_v_i32m1(outptr2, __riscv_vget_v_i32m4_i32m1(_sum, 2), vl32);
            __riscv_vse32_v_i32m1(outptr3, __riscv_vget_v_i32m4_i32m1(_sum, 3), vl32);

            outptr0 += out_elempack;
            outptr1 += out_elempack;
            outptr2 += out_elempack;
            outptr3 += out_elempack;
            r0 += packn;
            r1 += packn;
            r2 += packn;
        }
    }
}

static void convdw3x3s2_packn_int8_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Option& opt)
{
    const int packn = csrr_vlenb();
    const int out_elempack = top_blob.elempack;
    const size_t vl = __riscv_vsetvl_e8m1(packn);
    const size_t vl32 = __riscv_vsetvl_e32m1(out_elempack);

    int outw = top_blob.w;
    int outh = top_blob.h;
    const int group = bottom_blob.c;

    // int8 packn accumulates into four int32 packs
    const int total_gi = group * outh;
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int gi = 0; gi < total_gi; gi++)
    {
        int g = gi / outh;
        int i = gi % outh;

        int* outptr0 = top_blob.channel(g * 4).row<int>(i);
        int* outptr1 = top_blob.channel(g * 4 + 1).row<int>(i);
        int* outptr2 = top_blob.channel(g * 4 + 2).row<int>(i);
        int* outptr3 = top_blob.channel(g * 4 + 3).row<int>(i);

        const Mat img = bottom_blob.channel(g);
        const signed char* r0 = img.row<const signed char>(i * 2);
        const signed char* r1 = img.row<const signed char>(i * 2 + 1);
        const signed char* r2 = img.row<const signed char>(i * 2 + 2);
        const signed char* k0 = kernel.row<const signed char>(g);

        vint16m2_t _k00 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(k0, vl), 0, vl);
        vint16m2_t _k01 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(k0 + packn, vl), 0, vl);
        vint16m2_t _k02 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(k0 + packn * 2, vl), 0, vl);
        vint16m2_t _k10 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(k0 + packn * 3, vl), 0, vl);
        vint16m2_t _k11 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(k0 + packn * 4, vl), 0, vl);
        vint16m2_t _k12 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(k0 + packn * 5, vl), 0, vl);
        vint16m2_t _k20 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(k0 + packn * 6, vl), 0, vl);
        vint16m2_t _k21 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(k0 + packn * 7, vl), 0, vl);
        vint16m2_t _k22 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(k0 + packn * 8, vl), 0, vl);

        int j = 0;
        for (; j < outw; j++)
        {
            vint32m4_t _sum = __riscv_vmv_v_x_i32m4(0, vl);

            vint16m2_t _r00 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(r0, vl), 0, vl);
            _sum = __riscv_vwmacc_vv_i32m4(_sum, _k00, _r00, vl);
            vint16m2_t _r01 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(r0 + packn, vl), 0, vl);
            _sum = __riscv_vwmacc_vv_i32m4(_sum, _k01, _r01, vl);
            vint16m2_t _r02 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(r0 + packn * 2, vl), 0, vl);
            _sum = __riscv_vwmacc_vv_i32m4(_sum, _k02, _r02, vl);

            vint16m2_t _r10 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(r1, vl), 0, vl);
            _sum = __riscv_vwmacc_vv_i32m4(_sum, _k10, _r10, vl);
            vint16m2_t _r11 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(r1 + packn, vl), 0, vl);
            _sum = __riscv_vwmacc_vv_i32m4(_sum, _k11, _r11, vl);
            vint16m2_t _r12 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(r1 + packn * 2, vl), 0, vl);
            _sum = __riscv_vwmacc_vv_i32m4(_sum, _k12, _r12, vl);

            vint16m2_t _r20 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(r2, vl), 0, vl);
            _sum = __riscv_vwmacc_vv_i32m4(_sum, _k20, _r20, vl);
            vint16m2_t _r21 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(r2 + packn, vl), 0, vl);
            _sum = __riscv_vwmacc_vv_i32m4(_sum, _k21, _r21, vl);
            vint16m2_t _r22 = __riscv_vwadd_vx_i16m2(__riscv_vle8_v_i8m1(r2 + packn * 2, vl), 0, vl);
            _sum = __riscv_vwmacc_vv_i32m4(_sum, _k22, _r22, vl);

            __riscv_vse32_v_i32m1(outptr0, __riscv_vget_v_i32m4_i32m1(_sum, 0), vl32);
            __riscv_vse32_v_i32m1(outptr1, __riscv_vget_v_i32m4_i32m1(_sum, 1), vl32);
            __riscv_vse32_v_i32m1(outptr2, __riscv_vget_v_i32m4_i32m1(_sum, 2), vl32);
            __riscv_vse32_v_i32m1(outptr3, __riscv_vget_v_i32m4_i32m1(_sum, 3), vl32);

            outptr0 += out_elempack;
            outptr1 += out_elempack;
            outptr2 += out_elempack;
            outptr3 += out_elempack;
            r0 += packn * 2;
            r1 += packn * 2;
            r2 += packn * 2;
        }
    }
}
