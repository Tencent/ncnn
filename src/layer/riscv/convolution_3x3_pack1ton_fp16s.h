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

static void conv3x3s1_pack1ton_fp16sa_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    const int packn = csrr_vlenb() / 2;
    const size_t vl = vsetvl_e16m1(packn);

    int inch = bottom_blob.c;
    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const __fp16* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        vfloat16m1_t _bias0 = bias ? vle16_v_f16m1(bias + p * packn, vl) : vfmv_v_f_f16m1(0.f, vl);
        out0.fill(_bias0);

        const __fp16* k0 = kernel.channel(p);

        int q = 0;
        for (; q < inch; q++)
        {
            __fp16* outptr0 = out0;

            const Mat img0 = bottom_blob.channel(q);

            const __fp16* r0 = img0.row<const __fp16>(0);
            const __fp16* r1 = img0.row<const __fp16>(1);
            const __fp16* r2 = img0.row<const __fp16>(2);

            vfloat16m1_t _k00 = vle16_v_f16m1(k0, vl);
            vfloat16m1_t _k01 = vle16_v_f16m1(k0 + packn, vl);
            vfloat16m1_t _k02 = vle16_v_f16m1(k0 + packn * 2, vl);
            vfloat16m1_t _k10 = vle16_v_f16m1(k0 + packn * 3, vl);
            vfloat16m1_t _k11 = vle16_v_f16m1(k0 + packn * 4, vl);
            vfloat16m1_t _k12 = vle16_v_f16m1(k0 + packn * 5, vl);
            vfloat16m1_t _k20 = vle16_v_f16m1(k0 + packn * 6, vl);
            vfloat16m1_t _k21 = vle16_v_f16m1(k0 + packn * 7, vl);
            vfloat16m1_t _k22 = vle16_v_f16m1(k0 + packn * 8, vl);

            int i = 0;
            for (; i < outh; i++)
            {
                int j = 0;
                for (; j + 7 < outw; j += 8)
                {
                    vfloat16m1_t _sum0 = vle16_v_f16m1(outptr0, vl);
                    vfloat16m1_t _sum1 = vle16_v_f16m1(outptr0 + packn, vl);
                    vfloat16m1_t _sum2 = vle16_v_f16m1(outptr0 + packn * 2, vl);
                    vfloat16m1_t _sum3 = vle16_v_f16m1(outptr0 + packn * 3, vl);
                    vfloat16m1_t _sum4 = vle16_v_f16m1(outptr0 + packn * 4, vl);
                    vfloat16m1_t _sum5 = vle16_v_f16m1(outptr0 + packn * 5, vl);
                    vfloat16m1_t _sum6 = vle16_v_f16m1(outptr0 + packn * 6, vl);
                    vfloat16m1_t _sum7 = vle16_v_f16m1(outptr0 + packn * 7, vl);

                    _sum0 = vfmacc_vf_f16m1(_sum0, r0[0], _k00, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r0[1], _k00, vl);
                    _sum2 = vfmacc_vf_f16m1(_sum2, r0[2], _k00, vl);
                    _sum3 = vfmacc_vf_f16m1(_sum3, r0[3], _k00, vl);
                    _sum4 = vfmacc_vf_f16m1(_sum4, r0[4], _k00, vl);
                    _sum5 = vfmacc_vf_f16m1(_sum5, r0[5], _k00, vl);
                    _sum6 = vfmacc_vf_f16m1(_sum6, r0[6], _k00, vl);
                    _sum7 = vfmacc_vf_f16m1(_sum7, r0[7], _k00, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r0[1], _k01, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r0[2], _k01, vl);
                    _sum2 = vfmacc_vf_f16m1(_sum2, r0[3], _k01, vl);
                    _sum3 = vfmacc_vf_f16m1(_sum3, r0[4], _k01, vl);
                    _sum4 = vfmacc_vf_f16m1(_sum4, r0[5], _k01, vl);
                    _sum5 = vfmacc_vf_f16m1(_sum5, r0[6], _k01, vl);
                    _sum6 = vfmacc_vf_f16m1(_sum6, r0[7], _k01, vl);
                    _sum7 = vfmacc_vf_f16m1(_sum7, r0[8], _k01, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r0[2], _k02, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r0[3], _k02, vl);
                    _sum2 = vfmacc_vf_f16m1(_sum2, r0[4], _k02, vl);
                    _sum3 = vfmacc_vf_f16m1(_sum3, r0[5], _k02, vl);
                    _sum4 = vfmacc_vf_f16m1(_sum4, r0[6], _k02, vl);
                    _sum5 = vfmacc_vf_f16m1(_sum5, r0[7], _k02, vl);
                    _sum6 = vfmacc_vf_f16m1(_sum6, r0[8], _k02, vl);
                    _sum7 = vfmacc_vf_f16m1(_sum7, r0[9], _k02, vl);

                    _sum0 = vfmacc_vf_f16m1(_sum0, r1[0], _k10, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r1[1], _k10, vl);
                    _sum2 = vfmacc_vf_f16m1(_sum2, r1[2], _k10, vl);
                    _sum3 = vfmacc_vf_f16m1(_sum3, r1[3], _k10, vl);
                    _sum4 = vfmacc_vf_f16m1(_sum4, r1[4], _k10, vl);
                    _sum5 = vfmacc_vf_f16m1(_sum5, r1[5], _k10, vl);
                    _sum6 = vfmacc_vf_f16m1(_sum6, r1[6], _k10, vl);
                    _sum7 = vfmacc_vf_f16m1(_sum7, r1[7], _k10, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r1[1], _k11, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r1[2], _k11, vl);
                    _sum2 = vfmacc_vf_f16m1(_sum2, r1[3], _k11, vl);
                    _sum3 = vfmacc_vf_f16m1(_sum3, r1[4], _k11, vl);
                    _sum4 = vfmacc_vf_f16m1(_sum4, r1[5], _k11, vl);
                    _sum5 = vfmacc_vf_f16m1(_sum5, r1[6], _k11, vl);
                    _sum6 = vfmacc_vf_f16m1(_sum6, r1[7], _k11, vl);
                    _sum7 = vfmacc_vf_f16m1(_sum7, r1[8], _k11, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r1[2], _k12, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r1[3], _k12, vl);
                    _sum2 = vfmacc_vf_f16m1(_sum2, r1[4], _k12, vl);
                    _sum3 = vfmacc_vf_f16m1(_sum3, r1[5], _k12, vl);
                    _sum4 = vfmacc_vf_f16m1(_sum4, r1[6], _k12, vl);
                    _sum5 = vfmacc_vf_f16m1(_sum5, r1[7], _k12, vl);
                    _sum6 = vfmacc_vf_f16m1(_sum6, r1[8], _k12, vl);
                    _sum7 = vfmacc_vf_f16m1(_sum7, r1[9], _k12, vl);

                    _sum0 = vfmacc_vf_f16m1(_sum0, r2[0], _k20, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r2[1], _k20, vl);
                    _sum2 = vfmacc_vf_f16m1(_sum2, r2[2], _k20, vl);
                    _sum3 = vfmacc_vf_f16m1(_sum3, r2[3], _k20, vl);
                    _sum4 = vfmacc_vf_f16m1(_sum4, r2[4], _k20, vl);
                    _sum5 = vfmacc_vf_f16m1(_sum5, r2[5], _k20, vl);
                    _sum6 = vfmacc_vf_f16m1(_sum6, r2[6], _k20, vl);
                    _sum7 = vfmacc_vf_f16m1(_sum7, r2[7], _k20, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r2[1], _k21, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r2[2], _k21, vl);
                    _sum2 = vfmacc_vf_f16m1(_sum2, r2[3], _k21, vl);
                    _sum3 = vfmacc_vf_f16m1(_sum3, r2[4], _k21, vl);
                    _sum4 = vfmacc_vf_f16m1(_sum4, r2[5], _k21, vl);
                    _sum5 = vfmacc_vf_f16m1(_sum5, r2[6], _k21, vl);
                    _sum6 = vfmacc_vf_f16m1(_sum6, r2[7], _k21, vl);
                    _sum7 = vfmacc_vf_f16m1(_sum7, r2[8], _k21, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r2[2], _k22, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r2[3], _k22, vl);
                    _sum2 = vfmacc_vf_f16m1(_sum2, r2[4], _k22, vl);
                    _sum3 = vfmacc_vf_f16m1(_sum3, r2[5], _k22, vl);
                    _sum4 = vfmacc_vf_f16m1(_sum4, r2[6], _k22, vl);
                    _sum5 = vfmacc_vf_f16m1(_sum5, r2[7], _k22, vl);
                    _sum6 = vfmacc_vf_f16m1(_sum6, r2[8], _k22, vl);
                    _sum7 = vfmacc_vf_f16m1(_sum7, r2[9], _k22, vl);

                    vse16_v_f16m1(outptr0, _sum0, vl);
                    vse16_v_f16m1(outptr0 + packn, _sum1, vl);
                    vse16_v_f16m1(outptr0 + packn * 2, _sum2, vl);
                    vse16_v_f16m1(outptr0 + packn * 3, _sum3, vl);
                    vse16_v_f16m1(outptr0 + packn * 4, _sum4, vl);
                    vse16_v_f16m1(outptr0 + packn * 5, _sum5, vl);
                    vse16_v_f16m1(outptr0 + packn * 6, _sum6, vl);
                    vse16_v_f16m1(outptr0 + packn * 7, _sum7, vl);

                    outptr0 += packn * 8;

                    r0 += 8;
                    r1 += 8;
                    r2 += 8;
                }
                for (; j + 3 < outw; j += 4)
                {
                    vfloat16m1_t _sum0 = vle16_v_f16m1(outptr0, vl);
                    vfloat16m1_t _sum1 = vle16_v_f16m1(outptr0 + packn, vl);
                    vfloat16m1_t _sum2 = vle16_v_f16m1(outptr0 + packn * 2, vl);
                    vfloat16m1_t _sum3 = vle16_v_f16m1(outptr0 + packn * 3, vl);

                    _sum0 = vfmacc_vf_f16m1(_sum0, r0[0], _k00, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r0[1], _k00, vl);
                    _sum2 = vfmacc_vf_f16m1(_sum2, r0[2], _k00, vl);
                    _sum3 = vfmacc_vf_f16m1(_sum3, r0[3], _k00, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r0[1], _k01, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r0[2], _k01, vl);
                    _sum2 = vfmacc_vf_f16m1(_sum2, r0[3], _k01, vl);
                    _sum3 = vfmacc_vf_f16m1(_sum3, r0[4], _k01, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r0[2], _k02, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r0[3], _k02, vl);
                    _sum2 = vfmacc_vf_f16m1(_sum2, r0[4], _k02, vl);
                    _sum3 = vfmacc_vf_f16m1(_sum3, r0[5], _k02, vl);

                    _sum0 = vfmacc_vf_f16m1(_sum0, r1[0], _k10, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r1[1], _k10, vl);
                    _sum2 = vfmacc_vf_f16m1(_sum2, r1[2], _k10, vl);
                    _sum3 = vfmacc_vf_f16m1(_sum3, r1[3], _k10, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r1[1], _k11, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r1[2], _k11, vl);
                    _sum2 = vfmacc_vf_f16m1(_sum2, r1[3], _k11, vl);
                    _sum3 = vfmacc_vf_f16m1(_sum3, r1[4], _k11, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r1[2], _k12, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r1[3], _k12, vl);
                    _sum2 = vfmacc_vf_f16m1(_sum2, r1[4], _k12, vl);
                    _sum3 = vfmacc_vf_f16m1(_sum3, r1[5], _k12, vl);

                    _sum0 = vfmacc_vf_f16m1(_sum0, r2[0], _k20, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r2[1], _k20, vl);
                    _sum2 = vfmacc_vf_f16m1(_sum2, r2[2], _k20, vl);
                    _sum3 = vfmacc_vf_f16m1(_sum3, r2[3], _k20, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r2[1], _k21, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r2[2], _k21, vl);
                    _sum2 = vfmacc_vf_f16m1(_sum2, r2[3], _k21, vl);
                    _sum3 = vfmacc_vf_f16m1(_sum3, r2[4], _k21, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r2[2], _k22, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r2[3], _k22, vl);
                    _sum2 = vfmacc_vf_f16m1(_sum2, r2[4], _k22, vl);
                    _sum3 = vfmacc_vf_f16m1(_sum3, r2[5], _k22, vl);

                    vse16_v_f16m1(outptr0, _sum0, vl);
                    vse16_v_f16m1(outptr0 + packn, _sum1, vl);
                    vse16_v_f16m1(outptr0 + packn * 2, _sum2, vl);
                    vse16_v_f16m1(outptr0 + packn * 3, _sum3, vl);

                    outptr0 += packn * 4;

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                }
                for (; j + 1 < outw; j += 2)
                {
                    vfloat16m1_t _sum0 = vle16_v_f16m1(outptr0, vl);
                    vfloat16m1_t _sum1 = vle16_v_f16m1(outptr0 + packn, vl);

                    _sum0 = vfmacc_vf_f16m1(_sum0, r0[0], _k00, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r0[1], _k00, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r0[1], _k01, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r0[2], _k01, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r0[2], _k02, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r0[3], _k02, vl);

                    _sum0 = vfmacc_vf_f16m1(_sum0, r1[0], _k10, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r1[1], _k10, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r1[1], _k11, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r1[2], _k11, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r1[2], _k12, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r1[3], _k12, vl);

                    _sum0 = vfmacc_vf_f16m1(_sum0, r2[0], _k20, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r2[1], _k20, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r2[1], _k21, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r2[2], _k21, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r2[2], _k22, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r2[3], _k22, vl);

                    vse16_v_f16m1(outptr0, _sum0, vl);
                    vse16_v_f16m1(outptr0 + packn, _sum1, vl);

                    outptr0 += packn * 2;

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                }
                for (; j < outw; j++)
                {
                    vfloat16m1_t _sum0 = vle16_v_f16m1(outptr0, vl);

                    _sum0 = vfmacc_vf_f16m1(_sum0, r0[0], _k00, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r0[1], _k01, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r0[2], _k02, vl);

                    _sum0 = vfmacc_vf_f16m1(_sum0, r1[0], _k10, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r1[1], _k11, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r1[2], _k12, vl);

                    _sum0 = vfmacc_vf_f16m1(_sum0, r2[0], _k20, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r2[1], _k21, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r2[2], _k22, vl);

                    vse16_v_f16m1(outptr0, _sum0, vl);

                    outptr0 += packn;

                    r0 += 1;
                    r1 += 1;
                    r2 += 1;
                }

                r0 += 2;
                r1 += 2;
                r2 += 2;
            }

            k0 += 9 * packn;
        }
    }
}

static void conv3x3s2_pack1ton_fp16sa_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    const int packn = csrr_vlenb() / 2;
    const size_t vl = vsetvl_e16m1(packn);

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

        vfloat16m1_t _bias0 = bias ? vle16_v_f16m1(bias + p * packn, vl) : vfmv_v_f_f16m1(0.f, vl);
        out0.fill(_bias0);

        const __fp16* k0 = kernel.channel(p);

        int q = 0;
        for (; q < inch; q++)
        {
            __fp16* outptr0 = out0;

            const Mat img0 = bottom_blob.channel(q);

            const __fp16* r0 = img0.row<const __fp16>(0);
            const __fp16* r1 = img0.row<const __fp16>(1);
            const __fp16* r2 = img0.row<const __fp16>(2);

            vfloat16m1_t _k00 = vle16_v_f16m1(k0, vl);
            vfloat16m1_t _k01 = vle16_v_f16m1(k0 + packn, vl);
            vfloat16m1_t _k02 = vle16_v_f16m1(k0 + packn * 2, vl);
            vfloat16m1_t _k10 = vle16_v_f16m1(k0 + packn * 3, vl);
            vfloat16m1_t _k11 = vle16_v_f16m1(k0 + packn * 4, vl);
            vfloat16m1_t _k12 = vle16_v_f16m1(k0 + packn * 5, vl);
            vfloat16m1_t _k20 = vle16_v_f16m1(k0 + packn * 6, vl);
            vfloat16m1_t _k21 = vle16_v_f16m1(k0 + packn * 7, vl);
            vfloat16m1_t _k22 = vle16_v_f16m1(k0 + packn * 8, vl);

            int i = 0;
            for (; i < outh; i++)
            {
                int j = 0;
                for (; j + 7 < outw; j += 8)
                {
                    vfloat16m1_t _sum0 = vle16_v_f16m1(outptr0, vl);
                    vfloat16m1_t _sum1 = vle16_v_f16m1(outptr0 + packn, vl);
                    vfloat16m1_t _sum2 = vle16_v_f16m1(outptr0 + packn * 2, vl);
                    vfloat16m1_t _sum3 = vle16_v_f16m1(outptr0 + packn * 3, vl);
                    vfloat16m1_t _sum4 = vle16_v_f16m1(outptr0 + packn * 4, vl);
                    vfloat16m1_t _sum5 = vle16_v_f16m1(outptr0 + packn * 5, vl);
                    vfloat16m1_t _sum6 = vle16_v_f16m1(outptr0 + packn * 6, vl);
                    vfloat16m1_t _sum7 = vle16_v_f16m1(outptr0 + packn * 7, vl);

                    _sum0 = vfmacc_vf_f16m1(_sum0, r0[0], _k00, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r0[2], _k00, vl);
                    _sum2 = vfmacc_vf_f16m1(_sum2, r0[4], _k00, vl);
                    _sum3 = vfmacc_vf_f16m1(_sum3, r0[6], _k00, vl);
                    _sum4 = vfmacc_vf_f16m1(_sum4, r0[8], _k00, vl);
                    _sum5 = vfmacc_vf_f16m1(_sum5, r0[10], _k00, vl);
                    _sum6 = vfmacc_vf_f16m1(_sum6, r0[12], _k00, vl);
                    _sum7 = vfmacc_vf_f16m1(_sum7, r0[14], _k00, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r0[1], _k01, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r0[3], _k01, vl);
                    _sum2 = vfmacc_vf_f16m1(_sum2, r0[5], _k01, vl);
                    _sum3 = vfmacc_vf_f16m1(_sum3, r0[7], _k01, vl);
                    _sum4 = vfmacc_vf_f16m1(_sum4, r0[9], _k01, vl);
                    _sum5 = vfmacc_vf_f16m1(_sum5, r0[11], _k01, vl);
                    _sum6 = vfmacc_vf_f16m1(_sum6, r0[13], _k01, vl);
                    _sum7 = vfmacc_vf_f16m1(_sum7, r0[15], _k01, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r0[2], _k02, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r0[4], _k02, vl);
                    _sum2 = vfmacc_vf_f16m1(_sum2, r0[6], _k02, vl);
                    _sum3 = vfmacc_vf_f16m1(_sum3, r0[8], _k02, vl);
                    _sum4 = vfmacc_vf_f16m1(_sum4, r0[10], _k02, vl);
                    _sum5 = vfmacc_vf_f16m1(_sum5, r0[12], _k02, vl);
                    _sum6 = vfmacc_vf_f16m1(_sum6, r0[14], _k02, vl);
                    _sum7 = vfmacc_vf_f16m1(_sum7, r0[16], _k02, vl);

                    _sum0 = vfmacc_vf_f16m1(_sum0, r1[0], _k10, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r1[2], _k10, vl);
                    _sum2 = vfmacc_vf_f16m1(_sum2, r1[4], _k10, vl);
                    _sum3 = vfmacc_vf_f16m1(_sum3, r1[6], _k10, vl);
                    _sum4 = vfmacc_vf_f16m1(_sum4, r1[8], _k10, vl);
                    _sum5 = vfmacc_vf_f16m1(_sum5, r1[10], _k10, vl);
                    _sum6 = vfmacc_vf_f16m1(_sum6, r1[12], _k10, vl);
                    _sum7 = vfmacc_vf_f16m1(_sum7, r1[14], _k10, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r1[1], _k11, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r1[3], _k11, vl);
                    _sum2 = vfmacc_vf_f16m1(_sum2, r1[5], _k11, vl);
                    _sum3 = vfmacc_vf_f16m1(_sum3, r1[7], _k11, vl);
                    _sum4 = vfmacc_vf_f16m1(_sum4, r1[9], _k11, vl);
                    _sum5 = vfmacc_vf_f16m1(_sum5, r1[11], _k11, vl);
                    _sum6 = vfmacc_vf_f16m1(_sum6, r1[13], _k11, vl);
                    _sum7 = vfmacc_vf_f16m1(_sum7, r1[15], _k11, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r1[2], _k12, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r1[4], _k12, vl);
                    _sum2 = vfmacc_vf_f16m1(_sum2, r1[6], _k12, vl);
                    _sum3 = vfmacc_vf_f16m1(_sum3, r1[8], _k12, vl);
                    _sum4 = vfmacc_vf_f16m1(_sum4, r1[10], _k12, vl);
                    _sum5 = vfmacc_vf_f16m1(_sum5, r1[12], _k12, vl);
                    _sum6 = vfmacc_vf_f16m1(_sum6, r1[14], _k12, vl);
                    _sum7 = vfmacc_vf_f16m1(_sum7, r1[16], _k12, vl);

                    _sum0 = vfmacc_vf_f16m1(_sum0, r2[0], _k20, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r2[2], _k20, vl);
                    _sum2 = vfmacc_vf_f16m1(_sum2, r2[4], _k20, vl);
                    _sum3 = vfmacc_vf_f16m1(_sum3, r2[6], _k20, vl);
                    _sum4 = vfmacc_vf_f16m1(_sum4, r2[8], _k20, vl);
                    _sum5 = vfmacc_vf_f16m1(_sum5, r2[10], _k20, vl);
                    _sum6 = vfmacc_vf_f16m1(_sum6, r2[12], _k20, vl);
                    _sum7 = vfmacc_vf_f16m1(_sum7, r2[14], _k20, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r2[1], _k21, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r2[3], _k21, vl);
                    _sum2 = vfmacc_vf_f16m1(_sum2, r2[5], _k21, vl);
                    _sum3 = vfmacc_vf_f16m1(_sum3, r2[7], _k21, vl);
                    _sum4 = vfmacc_vf_f16m1(_sum4, r2[9], _k21, vl);
                    _sum5 = vfmacc_vf_f16m1(_sum5, r2[11], _k21, vl);
                    _sum6 = vfmacc_vf_f16m1(_sum6, r2[13], _k21, vl);
                    _sum7 = vfmacc_vf_f16m1(_sum7, r2[15], _k21, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r2[2], _k22, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r2[4], _k22, vl);
                    _sum2 = vfmacc_vf_f16m1(_sum2, r2[6], _k22, vl);
                    _sum3 = vfmacc_vf_f16m1(_sum3, r2[8], _k22, vl);
                    _sum4 = vfmacc_vf_f16m1(_sum4, r2[10], _k22, vl);
                    _sum5 = vfmacc_vf_f16m1(_sum5, r2[12], _k22, vl);
                    _sum6 = vfmacc_vf_f16m1(_sum6, r2[14], _k22, vl);
                    _sum7 = vfmacc_vf_f16m1(_sum7, r2[16], _k22, vl);

                    vse16_v_f16m1(outptr0, _sum0, vl);
                    vse16_v_f16m1(outptr0 + packn, _sum1, vl);
                    vse16_v_f16m1(outptr0 + packn * 2, _sum2, vl);
                    vse16_v_f16m1(outptr0 + packn * 3, _sum3, vl);
                    vse16_v_f16m1(outptr0 + packn * 4, _sum4, vl);
                    vse16_v_f16m1(outptr0 + packn * 5, _sum5, vl);
                    vse16_v_f16m1(outptr0 + packn * 6, _sum6, vl);
                    vse16_v_f16m1(outptr0 + packn * 7, _sum7, vl);

                    outptr0 += packn * 8;

                    r0 += 16;
                    r1 += 16;
                    r2 += 16;
                }
                for (; j + 3 < outw; j += 4)
                {
                    vfloat16m1_t _sum0 = vle16_v_f16m1(outptr0, vl);
                    vfloat16m1_t _sum1 = vle16_v_f16m1(outptr0 + packn, vl);
                    vfloat16m1_t _sum2 = vle16_v_f16m1(outptr0 + packn * 2, vl);
                    vfloat16m1_t _sum3 = vle16_v_f16m1(outptr0 + packn * 3, vl);

                    _sum0 = vfmacc_vf_f16m1(_sum0, r0[0], _k00, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r0[2], _k00, vl);
                    _sum2 = vfmacc_vf_f16m1(_sum2, r0[4], _k00, vl);
                    _sum3 = vfmacc_vf_f16m1(_sum3, r0[6], _k00, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r0[1], _k01, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r0[3], _k01, vl);
                    _sum2 = vfmacc_vf_f16m1(_sum2, r0[5], _k01, vl);
                    _sum3 = vfmacc_vf_f16m1(_sum3, r0[7], _k01, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r0[2], _k02, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r0[4], _k02, vl);
                    _sum2 = vfmacc_vf_f16m1(_sum2, r0[6], _k02, vl);
                    _sum3 = vfmacc_vf_f16m1(_sum3, r0[8], _k02, vl);

                    _sum0 = vfmacc_vf_f16m1(_sum0, r1[0], _k10, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r1[2], _k10, vl);
                    _sum2 = vfmacc_vf_f16m1(_sum2, r1[4], _k10, vl);
                    _sum3 = vfmacc_vf_f16m1(_sum3, r1[6], _k10, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r1[1], _k11, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r1[3], _k11, vl);
                    _sum2 = vfmacc_vf_f16m1(_sum2, r1[5], _k11, vl);
                    _sum3 = vfmacc_vf_f16m1(_sum3, r1[7], _k11, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r1[2], _k12, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r1[4], _k12, vl);
                    _sum2 = vfmacc_vf_f16m1(_sum2, r1[6], _k12, vl);
                    _sum3 = vfmacc_vf_f16m1(_sum3, r1[8], _k12, vl);

                    _sum0 = vfmacc_vf_f16m1(_sum0, r2[0], _k20, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r2[2], _k20, vl);
                    _sum2 = vfmacc_vf_f16m1(_sum2, r2[4], _k20, vl);
                    _sum3 = vfmacc_vf_f16m1(_sum3, r2[6], _k20, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r2[1], _k21, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r2[3], _k21, vl);
                    _sum2 = vfmacc_vf_f16m1(_sum2, r2[5], _k21, vl);
                    _sum3 = vfmacc_vf_f16m1(_sum3, r2[7], _k21, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r2[2], _k22, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r2[4], _k22, vl);
                    _sum2 = vfmacc_vf_f16m1(_sum2, r2[6], _k22, vl);
                    _sum3 = vfmacc_vf_f16m1(_sum3, r2[8], _k22, vl);

                    vse16_v_f16m1(outptr0, _sum0, vl);
                    vse16_v_f16m1(outptr0 + packn, _sum1, vl);
                    vse16_v_f16m1(outptr0 + packn * 2, _sum2, vl);
                    vse16_v_f16m1(outptr0 + packn * 3, _sum3, vl);

                    outptr0 += packn * 4;

                    r0 += 8;
                    r1 += 8;
                    r2 += 8;
                }
                for (; j + 1 < outw; j += 2)
                {
                    vfloat16m1_t _sum0 = vle16_v_f16m1(outptr0, vl);
                    vfloat16m1_t _sum1 = vle16_v_f16m1(outptr0 + packn, vl);

                    _sum0 = vfmacc_vf_f16m1(_sum0, r0[0], _k00, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r0[2], _k00, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r0[1], _k01, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r0[3], _k01, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r0[2], _k02, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r0[4], _k02, vl);

                    _sum0 = vfmacc_vf_f16m1(_sum0, r1[0], _k10, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r1[2], _k10, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r1[1], _k11, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r1[3], _k11, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r1[2], _k12, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r1[4], _k12, vl);

                    _sum0 = vfmacc_vf_f16m1(_sum0, r2[0], _k20, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r2[2], _k20, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r2[1], _k21, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r2[3], _k21, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r2[2], _k22, vl);
                    _sum1 = vfmacc_vf_f16m1(_sum1, r2[4], _k22, vl);

                    vse16_v_f16m1(outptr0, _sum0, vl);
                    vse16_v_f16m1(outptr0 + packn, _sum1, vl);

                    outptr0 += packn * 2;

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                }
                for (; j < outw; j++)
                {
                    vfloat16m1_t _sum0 = vle16_v_f16m1(outptr0, vl);

                    _sum0 = vfmacc_vf_f16m1(_sum0, r0[0], _k00, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r0[1], _k01, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r0[2], _k02, vl);

                    _sum0 = vfmacc_vf_f16m1(_sum0, r1[0], _k10, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r1[1], _k11, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r1[2], _k12, vl);

                    _sum0 = vfmacc_vf_f16m1(_sum0, r2[0], _k20, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r2[1], _k21, vl);
                    _sum0 = vfmacc_vf_f16m1(_sum0, r2[2], _k22, vl);

                    vse16_v_f16m1(outptr0, _sum0, vl);

                    outptr0 += packn;

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }

            k0 += 9 * packn;
        }
    }
}
