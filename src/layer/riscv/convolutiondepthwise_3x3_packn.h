// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

static void convdw3x3s1_packn_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    const int packn = csrr_vlenb() / 4;
    const size_t vl = vsetvl_e32m1(packn);

    int w = bottom_blob.w;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int group = bottom_blob.c;

    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g = 0; g < group; g++)
    {
        Mat out = top_blob.channel(g);

        vfloat32m1_t _bias0 = bias ? vle32_v_f32m1(bias + g * packn, vl) : vfmv_v_f_f32m1(0.f, vl);

        const float* k0 = kernel.row(g);

        float* outptr0 = out.row(0);
        float* outptr1 = out.row(1);

        const Mat img0 = bottom_blob.channel(g);

        const float* r0 = img0.row(0);
        const float* r1 = img0.row(1);
        const float* r2 = img0.row(2);
        const float* r3 = img0.row(3);

        vfloat32m1_t _k00 = vle32_v_f32m1(k0, vl);
        vfloat32m1_t _k01 = vle32_v_f32m1(k0 + packn, vl);
        vfloat32m1_t _k02 = vle32_v_f32m1(k0 + packn * 2, vl);
        vfloat32m1_t _k10 = vle32_v_f32m1(k0 + packn * 3, vl);
        vfloat32m1_t _k11 = vle32_v_f32m1(k0 + packn * 4, vl);
        vfloat32m1_t _k12 = vle32_v_f32m1(k0 + packn * 5, vl);
        vfloat32m1_t _k20 = vle32_v_f32m1(k0 + packn * 6, vl);
        vfloat32m1_t _k21 = vle32_v_f32m1(k0 + packn * 7, vl);
        vfloat32m1_t _k22 = vle32_v_f32m1(k0 + packn * 8, vl);

        int i = 0;
        for (; i + 1 < outh; i += 2)
        {
            int j = 0;
            for (; j + 1 < outw; j += 2)
            {
                vfloat32m1_t _sum00 = _bias0;
                vfloat32m1_t _sum01 = _bias0;
                vfloat32m1_t _sum10 = _bias0;
                vfloat32m1_t _sum11 = _bias0;

                vfloat32m1_t _r00 = vle32_v_f32m1(r0, vl);
                vfloat32m1_t _r01 = vle32_v_f32m1(r0 + packn, vl);
                vfloat32m1_t _r02 = vle32_v_f32m1(r0 + packn * 2, vl);
                vfloat32m1_t _r03 = vle32_v_f32m1(r0 + packn * 3, vl);

                _sum00 = vfmacc_vv_f32m1(_sum00, _k00, _r00, vl);
                _sum00 = vfmacc_vv_f32m1(_sum00, _k01, _r01, vl);
                _sum00 = vfmacc_vv_f32m1(_sum00, _k02, _r02, vl);
                _sum01 = vfmacc_vv_f32m1(_sum01, _k00, _r01, vl);
                _sum01 = vfmacc_vv_f32m1(_sum01, _k01, _r02, vl);
                _sum01 = vfmacc_vv_f32m1(_sum01, _k02, _r03, vl);

                vfloat32m1_t _r10 = vle32_v_f32m1(r1, vl);
                vfloat32m1_t _r11 = vle32_v_f32m1(r1 + packn, vl);
                vfloat32m1_t _r12 = vle32_v_f32m1(r1 + packn * 2, vl);
                vfloat32m1_t _r13 = vle32_v_f32m1(r1 + packn * 3, vl);

                _sum00 = vfmacc_vv_f32m1(_sum00, _k10, _r10, vl);
                _sum00 = vfmacc_vv_f32m1(_sum00, _k11, _r11, vl);
                _sum00 = vfmacc_vv_f32m1(_sum00, _k12, _r12, vl);
                _sum01 = vfmacc_vv_f32m1(_sum01, _k10, _r11, vl);
                _sum01 = vfmacc_vv_f32m1(_sum01, _k11, _r12, vl);
                _sum01 = vfmacc_vv_f32m1(_sum01, _k12, _r13, vl);
                _sum10 = vfmacc_vv_f32m1(_sum10, _k00, _r10, vl);
                _sum10 = vfmacc_vv_f32m1(_sum10, _k01, _r11, vl);
                _sum10 = vfmacc_vv_f32m1(_sum10, _k02, _r12, vl);
                _sum11 = vfmacc_vv_f32m1(_sum11, _k00, _r11, vl);
                _sum11 = vfmacc_vv_f32m1(_sum11, _k01, _r12, vl);
                _sum11 = vfmacc_vv_f32m1(_sum11, _k02, _r13, vl);

                vfloat32m1_t _r20 = vle32_v_f32m1(r2, vl);
                vfloat32m1_t _r21 = vle32_v_f32m1(r2 + packn, vl);
                vfloat32m1_t _r22 = vle32_v_f32m1(r2 + packn * 2, vl);
                vfloat32m1_t _r23 = vle32_v_f32m1(r2 + packn * 3, vl);

                _sum00 = vfmacc_vv_f32m1(_sum00, _k20, _r20, vl);
                _sum00 = vfmacc_vv_f32m1(_sum00, _k21, _r21, vl);
                _sum00 = vfmacc_vv_f32m1(_sum00, _k22, _r22, vl);
                _sum01 = vfmacc_vv_f32m1(_sum01, _k20, _r21, vl);
                _sum01 = vfmacc_vv_f32m1(_sum01, _k21, _r22, vl);
                _sum01 = vfmacc_vv_f32m1(_sum01, _k22, _r23, vl);
                _sum10 = vfmacc_vv_f32m1(_sum10, _k10, _r20, vl);
                _sum10 = vfmacc_vv_f32m1(_sum10, _k11, _r21, vl);
                _sum10 = vfmacc_vv_f32m1(_sum10, _k12, _r22, vl);
                _sum11 = vfmacc_vv_f32m1(_sum11, _k10, _r21, vl);
                _sum11 = vfmacc_vv_f32m1(_sum11, _k11, _r22, vl);
                _sum11 = vfmacc_vv_f32m1(_sum11, _k12, _r23, vl);

                vfloat32m1_t _r30 = vle32_v_f32m1(r3, vl);
                vfloat32m1_t _r31 = vle32_v_f32m1(r3 + packn, vl);
                vfloat32m1_t _r32 = vle32_v_f32m1(r3 + packn * 2, vl);
                vfloat32m1_t _r33 = vle32_v_f32m1(r3 + packn * 3, vl);

                _sum10 = vfmacc_vv_f32m1(_sum10, _k20, _r30, vl);
                _sum10 = vfmacc_vv_f32m1(_sum10, _k21, _r31, vl);
                _sum10 = vfmacc_vv_f32m1(_sum10, _k22, _r32, vl);
                _sum11 = vfmacc_vv_f32m1(_sum11, _k20, _r31, vl);
                _sum11 = vfmacc_vv_f32m1(_sum11, _k21, _r32, vl);
                _sum11 = vfmacc_vv_f32m1(_sum11, _k22, _r33, vl);

                vse32_v_f32m1(outptr0, _sum00, vl);
                vse32_v_f32m1(outptr0 + packn, _sum01, vl);
                vse32_v_f32m1(outptr1, _sum10, vl);
                vse32_v_f32m1(outptr1 + packn, _sum11, vl);

                outptr0 += packn * 2;
                outptr1 += packn * 2;

                r0 += packn * 2;
                r1 += packn * 2;
                r2 += packn * 2;
                r3 += packn * 2;
            }
            for (; j < outw; j++)
            {
                vfloat32m1_t _sum0 = _bias0;
                vfloat32m1_t _sum1 = _bias0;

                vfloat32m1_t _r00 = vle32_v_f32m1(r0, vl);
                vfloat32m1_t _r01 = vle32_v_f32m1(r0 + packn, vl);
                vfloat32m1_t _r02 = vle32_v_f32m1(r0 + packn * 2, vl);

                _sum0 = vfmacc_vv_f32m1(_sum0, _k00, _r00, vl);
                _sum0 = vfmacc_vv_f32m1(_sum0, _k01, _r01, vl);
                _sum0 = vfmacc_vv_f32m1(_sum0, _k02, _r02, vl);

                vfloat32m1_t _r10 = vle32_v_f32m1(r1, vl);
                vfloat32m1_t _r11 = vle32_v_f32m1(r1 + packn, vl);
                vfloat32m1_t _r12 = vle32_v_f32m1(r1 + packn * 2, vl);

                _sum0 = vfmacc_vv_f32m1(_sum0, _k10, _r10, vl);
                _sum0 = vfmacc_vv_f32m1(_sum0, _k11, _r11, vl);
                _sum0 = vfmacc_vv_f32m1(_sum0, _k12, _r12, vl);
                _sum1 = vfmacc_vv_f32m1(_sum1, _k00, _r10, vl);
                _sum1 = vfmacc_vv_f32m1(_sum1, _k01, _r11, vl);
                _sum1 = vfmacc_vv_f32m1(_sum1, _k02, _r12, vl);

                vfloat32m1_t _r20 = vle32_v_f32m1(r2, vl);
                vfloat32m1_t _r21 = vle32_v_f32m1(r2 + packn, vl);
                vfloat32m1_t _r22 = vle32_v_f32m1(r2 + packn * 2, vl);

                _sum0 = vfmacc_vv_f32m1(_sum0, _k20, _r20, vl);
                _sum0 = vfmacc_vv_f32m1(_sum0, _k21, _r21, vl);
                _sum0 = vfmacc_vv_f32m1(_sum0, _k22, _r22, vl);
                _sum1 = vfmacc_vv_f32m1(_sum1, _k10, _r20, vl);
                _sum1 = vfmacc_vv_f32m1(_sum1, _k11, _r21, vl);
                _sum1 = vfmacc_vv_f32m1(_sum1, _k12, _r22, vl);

                vfloat32m1_t _r30 = vle32_v_f32m1(r3, vl);
                vfloat32m1_t _r31 = vle32_v_f32m1(r3 + packn, vl);
                vfloat32m1_t _r32 = vle32_v_f32m1(r3 + packn * 2, vl);

                _sum1 = vfmacc_vv_f32m1(_sum1, _k20, _r30, vl);
                _sum1 = vfmacc_vv_f32m1(_sum1, _k21, _r31, vl);
                _sum1 = vfmacc_vv_f32m1(_sum1, _k22, _r32, vl);

                vse32_v_f32m1(outptr0, _sum0, vl);
                vse32_v_f32m1(outptr1, _sum1, vl);

                outptr0 += packn;
                outptr1 += packn;

                r0 += packn;
                r1 += packn;
                r2 += packn;
                r3 += packn;
            }

            r0 += 2 * packn + w * packn;
            r1 += 2 * packn + w * packn;
            r2 += 2 * packn + w * packn;
            r3 += 2 * packn + w * packn;

            outptr0 += outw * packn;
            outptr1 += outw * packn;
        }
        for (; i < outh; i++)
        {
            int j = 0;
            for (; j + 1 < outw; j += 2)
            {
                vfloat32m1_t _sum00 = _bias0;
                vfloat32m1_t _sum01 = _bias0;

                vfloat32m1_t _r00 = vle32_v_f32m1(r0, vl);
                vfloat32m1_t _r01 = vle32_v_f32m1(r0 + packn, vl);
                vfloat32m1_t _r02 = vle32_v_f32m1(r0 + packn * 2, vl);
                vfloat32m1_t _r03 = vle32_v_f32m1(r0 + packn * 3, vl);

                _sum00 = vfmacc_vv_f32m1(_sum00, _k00, _r00, vl);
                _sum00 = vfmacc_vv_f32m1(_sum00, _k01, _r01, vl);
                _sum00 = vfmacc_vv_f32m1(_sum00, _k02, _r02, vl);
                _sum01 = vfmacc_vv_f32m1(_sum01, _k00, _r01, vl);
                _sum01 = vfmacc_vv_f32m1(_sum01, _k01, _r02, vl);
                _sum01 = vfmacc_vv_f32m1(_sum01, _k02, _r03, vl);

                vfloat32m1_t _r10 = vle32_v_f32m1(r1, vl);
                vfloat32m1_t _r11 = vle32_v_f32m1(r1 + packn, vl);
                vfloat32m1_t _r12 = vle32_v_f32m1(r1 + packn * 2, vl);
                vfloat32m1_t _r13 = vle32_v_f32m1(r1 + packn * 3, vl);

                _sum00 = vfmacc_vv_f32m1(_sum00, _k10, _r10, vl);
                _sum00 = vfmacc_vv_f32m1(_sum00, _k11, _r11, vl);
                _sum00 = vfmacc_vv_f32m1(_sum00, _k12, _r12, vl);
                _sum01 = vfmacc_vv_f32m1(_sum01, _k10, _r11, vl);
                _sum01 = vfmacc_vv_f32m1(_sum01, _k11, _r12, vl);
                _sum01 = vfmacc_vv_f32m1(_sum01, _k12, _r13, vl);

                vfloat32m1_t _r20 = vle32_v_f32m1(r2, vl);
                vfloat32m1_t _r21 = vle32_v_f32m1(r2 + packn, vl);
                vfloat32m1_t _r22 = vle32_v_f32m1(r2 + packn * 2, vl);
                vfloat32m1_t _r23 = vle32_v_f32m1(r2 + packn * 3, vl);

                _sum00 = vfmacc_vv_f32m1(_sum00, _k20, _r20, vl);
                _sum00 = vfmacc_vv_f32m1(_sum00, _k21, _r21, vl);
                _sum00 = vfmacc_vv_f32m1(_sum00, _k22, _r22, vl);
                _sum01 = vfmacc_vv_f32m1(_sum01, _k20, _r21, vl);
                _sum01 = vfmacc_vv_f32m1(_sum01, _k21, _r22, vl);
                _sum01 = vfmacc_vv_f32m1(_sum01, _k22, _r23, vl);

                vse32_v_f32m1(outptr0, _sum00, vl);
                vse32_v_f32m1(outptr0 + packn, _sum01, vl);

                outptr0 += packn * 2;

                r0 += packn * 2;
                r1 += packn * 2;
                r2 += packn * 2;
            }
            for (; j < outw; j++)
            {
                vfloat32m1_t _sum0 = _bias0;

                vfloat32m1_t _r00 = vle32_v_f32m1(r0, vl);
                vfloat32m1_t _r01 = vle32_v_f32m1(r0 + packn, vl);
                vfloat32m1_t _r02 = vle32_v_f32m1(r0 + packn * 2, vl);

                _sum0 = vfmacc_vv_f32m1(_sum0, _k00, _r00, vl);
                _sum0 = vfmacc_vv_f32m1(_sum0, _k01, _r01, vl);
                _sum0 = vfmacc_vv_f32m1(_sum0, _k02, _r02, vl);

                vfloat32m1_t _r10 = vle32_v_f32m1(r1, vl);
                vfloat32m1_t _r11 = vle32_v_f32m1(r1 + packn, vl);
                vfloat32m1_t _r12 = vle32_v_f32m1(r1 + packn * 2, vl);

                _sum0 = vfmacc_vv_f32m1(_sum0, _k10, _r10, vl);
                _sum0 = vfmacc_vv_f32m1(_sum0, _k11, _r11, vl);
                _sum0 = vfmacc_vv_f32m1(_sum0, _k12, _r12, vl);

                vfloat32m1_t _r20 = vle32_v_f32m1(r2, vl);
                vfloat32m1_t _r21 = vle32_v_f32m1(r2 + packn, vl);
                vfloat32m1_t _r22 = vle32_v_f32m1(r2 + packn * 2, vl);

                _sum0 = vfmacc_vv_f32m1(_sum0, _k20, _r20, vl);
                _sum0 = vfmacc_vv_f32m1(_sum0, _k21, _r21, vl);
                _sum0 = vfmacc_vv_f32m1(_sum0, _k22, _r22, vl);

                vse32_v_f32m1(outptr0, _sum0, vl);

                outptr0 += packn;

                r0 += packn;
                r1 += packn;
                r2 += packn;
            }

            r0 += 2 * packn;
            r1 += 2 * packn;
            r2 += 2 * packn;
        }
    }
}

static void convdw3x3s2_packn_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    const int packn = csrr_vlenb() / 4;
    const size_t vl = vsetvl_e32m1(packn);

    int w = bottom_blob.w;

    int outw = top_blob.w;
    int outh = top_blob.h;

    const int group = bottom_blob.c;

    const int tailstep = (w - 2 * outw + w) * packn;

    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g = 0; g < group; g++)
    {
        Mat out = top_blob.channel(g);

        vfloat32m1_t _bias0 = bias ? vle32_v_f32m1(bias + g * packn, vl) : vfmv_v_f_f32m1(0.f, vl);

        const float* k0 = kernel.row(g);

        float* outptr0 = out;

        const Mat img0 = bottom_blob.channel(g);

        const float* r0 = img0.row(0);
        const float* r1 = img0.row(1);
        const float* r2 = img0.row(2);

        vfloat32m1_t _k00 = vle32_v_f32m1(k0, vl);
        vfloat32m1_t _k01 = vle32_v_f32m1(k0 + packn, vl);
        vfloat32m1_t _k02 = vle32_v_f32m1(k0 + packn * 2, vl);
        vfloat32m1_t _k10 = vle32_v_f32m1(k0 + packn * 3, vl);
        vfloat32m1_t _k11 = vle32_v_f32m1(k0 + packn * 4, vl);
        vfloat32m1_t _k12 = vle32_v_f32m1(k0 + packn * 5, vl);
        vfloat32m1_t _k20 = vle32_v_f32m1(k0 + packn * 6, vl);
        vfloat32m1_t _k21 = vle32_v_f32m1(k0 + packn * 7, vl);
        vfloat32m1_t _k22 = vle32_v_f32m1(k0 + packn * 8, vl);

        int i = 0;
        for (; i < outh; i++)
        {
            int j = 0;
            for (; j + 1 < outw; j += 2)
            {
                vfloat32m1_t _sum00 = _bias0;
                vfloat32m1_t _sum01 = _bias0;

                vfloat32m1_t _r00 = vle32_v_f32m1(r0, vl);
                vfloat32m1_t _r01 = vle32_v_f32m1(r0 + packn, vl);
                vfloat32m1_t _r02 = vle32_v_f32m1(r0 + packn * 2, vl);
                vfloat32m1_t _r03 = vle32_v_f32m1(r0 + packn * 3, vl);
                vfloat32m1_t _r04 = vle32_v_f32m1(r0 + packn * 4, vl);

                _sum00 = vfmacc_vv_f32m1(_sum00, _k00, _r00, vl);
                _sum00 = vfmacc_vv_f32m1(_sum00, _k01, _r01, vl);
                _sum00 = vfmacc_vv_f32m1(_sum00, _k02, _r02, vl);
                _sum01 = vfmacc_vv_f32m1(_sum01, _k00, _r02, vl);
                _sum01 = vfmacc_vv_f32m1(_sum01, _k01, _r03, vl);
                _sum01 = vfmacc_vv_f32m1(_sum01, _k02, _r04, vl);

                vfloat32m1_t _r10 = vle32_v_f32m1(r1, vl);
                vfloat32m1_t _r11 = vle32_v_f32m1(r1 + packn, vl);
                vfloat32m1_t _r12 = vle32_v_f32m1(r1 + packn * 2, vl);
                vfloat32m1_t _r13 = vle32_v_f32m1(r1 + packn * 3, vl);
                vfloat32m1_t _r14 = vle32_v_f32m1(r1 + packn * 4, vl);

                _sum00 = vfmacc_vv_f32m1(_sum00, _k10, _r10, vl);
                _sum00 = vfmacc_vv_f32m1(_sum00, _k11, _r11, vl);
                _sum00 = vfmacc_vv_f32m1(_sum00, _k12, _r12, vl);
                _sum01 = vfmacc_vv_f32m1(_sum01, _k10, _r12, vl);
                _sum01 = vfmacc_vv_f32m1(_sum01, _k11, _r13, vl);
                _sum01 = vfmacc_vv_f32m1(_sum01, _k12, _r14, vl);

                vfloat32m1_t _r20 = vle32_v_f32m1(r2, vl);
                vfloat32m1_t _r21 = vle32_v_f32m1(r2 + packn, vl);
                vfloat32m1_t _r22 = vle32_v_f32m1(r2 + packn * 2, vl);
                vfloat32m1_t _r23 = vle32_v_f32m1(r2 + packn * 3, vl);
                vfloat32m1_t _r24 = vle32_v_f32m1(r2 + packn * 4, vl);

                _sum00 = vfmacc_vv_f32m1(_sum00, _k20, _r20, vl);
                _sum00 = vfmacc_vv_f32m1(_sum00, _k21, _r21, vl);
                _sum00 = vfmacc_vv_f32m1(_sum00, _k22, _r22, vl);
                _sum01 = vfmacc_vv_f32m1(_sum01, _k20, _r22, vl);
                _sum01 = vfmacc_vv_f32m1(_sum01, _k21, _r23, vl);
                _sum01 = vfmacc_vv_f32m1(_sum01, _k22, _r24, vl);

                vse32_v_f32m1(outptr0, _sum00, vl);
                vse32_v_f32m1(outptr0 + packn, _sum01, vl);

                outptr0 += packn * 2;

                r0 += packn * 4;
                r1 += packn * 4;
                r2 += packn * 4;
            }
            for (; j < outw; j++)
            {
                vfloat32m1_t _sum0 = _bias0;

                vfloat32m1_t _r00 = vle32_v_f32m1(r0, vl);
                vfloat32m1_t _r01 = vle32_v_f32m1(r0 + packn, vl);
                vfloat32m1_t _r02 = vle32_v_f32m1(r0 + packn * 2, vl);

                _sum0 = vfmacc_vv_f32m1(_sum0, _k00, _r00, vl);
                _sum0 = vfmacc_vv_f32m1(_sum0, _k01, _r01, vl);
                _sum0 = vfmacc_vv_f32m1(_sum0, _k02, _r02, vl);

                vfloat32m1_t _r10 = vle32_v_f32m1(r1, vl);
                vfloat32m1_t _r11 = vle32_v_f32m1(r1 + packn, vl);
                vfloat32m1_t _r12 = vle32_v_f32m1(r1 + packn * 2, vl);

                _sum0 = vfmacc_vv_f32m1(_sum0, _k10, _r10, vl);
                _sum0 = vfmacc_vv_f32m1(_sum0, _k11, _r11, vl);
                _sum0 = vfmacc_vv_f32m1(_sum0, _k12, _r12, vl);

                vfloat32m1_t _r20 = vle32_v_f32m1(r2, vl);
                vfloat32m1_t _r21 = vle32_v_f32m1(r2 + packn, vl);
                vfloat32m1_t _r22 = vle32_v_f32m1(r2 + packn * 2, vl);

                _sum0 = vfmacc_vv_f32m1(_sum0, _k20, _r20, vl);
                _sum0 = vfmacc_vv_f32m1(_sum0, _k21, _r21, vl);
                _sum0 = vfmacc_vv_f32m1(_sum0, _k22, _r22, vl);

                vse32_v_f32m1(outptr0, _sum0, vl);

                outptr0 += packn;

                r0 += packn * 2;
                r1 += packn * 2;
                r2 += packn * 2;
            }

            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
        }
    }
}
