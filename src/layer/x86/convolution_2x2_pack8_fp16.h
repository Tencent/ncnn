// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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
static void conv2x2s1_weight_fp16_pack8_avx(const Mat& kernel, Mat& kernel_tm_pack8, int num_input, int num_output)
{
    // src = kw-kh-inch-outch
    // dst = 8b-8a-kw-kh-inch/8a-outch/8b
    Mat weight_data_r2 = kernel.reshape(4, num_input, num_output);

    kernel_tm_pack8.create(4, num_input / 8, num_output / 8, (size_t)2 * 64, 64);

    for (int q = 0; q + 7 < num_output; q += 8)
    {
        const Mat k0 = weight_data_r2.channel(q);
        const Mat k1 = weight_data_r2.channel(q + 1);
        const Mat k2 = weight_data_r2.channel(q + 2);
        const Mat k3 = weight_data_r2.channel(q + 3);
        const Mat k4 = weight_data_r2.channel(q + 4);
        const Mat k5 = weight_data_r2.channel(q + 5);
        const Mat k6 = weight_data_r2.channel(q + 6);
        const Mat k7 = weight_data_r2.channel(q + 7);

        Mat g0 = kernel_tm_pack8.channel(q / 8);

        for (int p = 0; p + 7 < num_input; p += 8)
        {
            const float* k00 = k0.row(p);
            const float* k01 = k0.row(p + 1);
            const float* k02 = k0.row(p + 2);
            const float* k03 = k0.row(p + 3);
            const float* k04 = k0.row(p + 4);
            const float* k05 = k0.row(p + 5);
            const float* k06 = k0.row(p + 6);
            const float* k07 = k0.row(p + 7);

            const float* k10 = k1.row(p);
            const float* k11 = k1.row(p + 1);
            const float* k12 = k1.row(p + 2);
            const float* k13 = k1.row(p + 3);
            const float* k14 = k1.row(p + 4);
            const float* k15 = k1.row(p + 5);
            const float* k16 = k1.row(p + 6);
            const float* k17 = k1.row(p + 7);

            const float* k20 = k2.row(p);
            const float* k21 = k2.row(p + 1);
            const float* k22 = k2.row(p + 2);
            const float* k23 = k2.row(p + 3);
            const float* k24 = k2.row(p + 4);
            const float* k25 = k2.row(p + 5);
            const float* k26 = k2.row(p + 6);
            const float* k27 = k2.row(p + 7);

            const float* k30 = k3.row(p);
            const float* k31 = k3.row(p + 1);
            const float* k32 = k3.row(p + 2);
            const float* k33 = k3.row(p + 3);
            const float* k34 = k3.row(p + 4);
            const float* k35 = k3.row(p + 5);
            const float* k36 = k3.row(p + 6);
            const float* k37 = k3.row(p + 7);

            const float* k40 = k4.row(p);
            const float* k41 = k4.row(p + 1);
            const float* k42 = k4.row(p + 2);
            const float* k43 = k4.row(p + 3);
            const float* k44 = k4.row(p + 4);
            const float* k45 = k4.row(p + 5);
            const float* k46 = k4.row(p + 6);
            const float* k47 = k4.row(p + 7);

            const float* k50 = k5.row(p);
            const float* k51 = k5.row(p + 1);
            const float* k52 = k5.row(p + 2);
            const float* k53 = k5.row(p + 3);
            const float* k54 = k5.row(p + 4);
            const float* k55 = k5.row(p + 5);
            const float* k56 = k5.row(p + 6);
            const float* k57 = k5.row(p + 7);

            const float* k60 = k6.row(p);
            const float* k61 = k6.row(p + 1);
            const float* k62 = k6.row(p + 2);
            const float* k63 = k6.row(p + 3);
            const float* k64 = k6.row(p + 4);
            const float* k65 = k6.row(p + 5);
            const float* k66 = k6.row(p + 6);
            const float* k67 = k6.row(p + 7);

            const float* k70 = k7.row(p);
            const float* k71 = k7.row(p + 1);
            const float* k72 = k7.row(p + 2);
            const float* k73 = k7.row(p + 3);
            const float* k74 = k7.row(p + 4);
            const float* k75 = k7.row(p + 5);
            const float* k76 = k7.row(p + 6);
            const float* k77 = k7.row(p + 7);

            unsigned short* g00 = (unsigned short*)g0.row(p / 8);

            for (int k = 0; k < 4; k++)
            {
                g00[0] = float32_to_float16(k00[k]);
                g00[1] = float32_to_float16(k10[k]);
                g00[2] = float32_to_float16(k20[k]);
                g00[3] = float32_to_float16(k30[k]);
                g00[4] = float32_to_float16(k40[k]);
                g00[5] = float32_to_float16(k50[k]);
                g00[6] = float32_to_float16(k60[k]);
                g00[7] = float32_to_float16(k70[k]);
                g00 += 8;
                g00[0] = float32_to_float16(k01[k]);
                g00[1] = float32_to_float16(k11[k]);
                g00[2] = float32_to_float16(k21[k]);
                g00[3] = float32_to_float16(k31[k]);
                g00[4] = float32_to_float16(k41[k]);
                g00[5] = float32_to_float16(k51[k]);
                g00[6] = float32_to_float16(k61[k]);
                g00[7] = float32_to_float16(k71[k]);

                g00 += 8;
                g00[0] = float32_to_float16(k02[k]);
                g00[1] = float32_to_float16(k12[k]);
                g00[2] = float32_to_float16(k22[k]);
                g00[3] = float32_to_float16(k32[k]);
                g00[4] = float32_to_float16(k42[k]);
                g00[5] = float32_to_float16(k52[k]);
                g00[6] = float32_to_float16(k62[k]);
                g00[7] = float32_to_float16(k72[k]);

                g00 += 8;
                g00[0] = float32_to_float16(k03[k]);
                g00[1] = float32_to_float16(k13[k]);
                g00[2] = float32_to_float16(k23[k]);
                g00[3] = float32_to_float16(k33[k]);
                g00[4] = float32_to_float16(k43[k]);
                g00[5] = float32_to_float16(k53[k]);
                g00[6] = float32_to_float16(k63[k]);
                g00[7] = float32_to_float16(k73[k]);

                g00 += 8;
                g00[0] = float32_to_float16(k04[k]);
                g00[1] = float32_to_float16(k14[k]);
                g00[2] = float32_to_float16(k24[k]);
                g00[3] = float32_to_float16(k34[k]);
                g00[4] = float32_to_float16(k44[k]);
                g00[5] = float32_to_float16(k54[k]);
                g00[6] = float32_to_float16(k64[k]);
                g00[7] = float32_to_float16(k74[k]);

                g00 += 8;
                g00[0] = float32_to_float16(k05[k]);
                g00[1] = float32_to_float16(k15[k]);
                g00[2] = float32_to_float16(k25[k]);
                g00[3] = float32_to_float16(k35[k]);
                g00[4] = float32_to_float16(k45[k]);
                g00[5] = float32_to_float16(k55[k]);
                g00[6] = float32_to_float16(k65[k]);
                g00[7] = float32_to_float16(k75[k]);

                g00 += 8;
                g00[0] = float32_to_float16(k06[k]);
                g00[1] = float32_to_float16(k16[k]);
                g00[2] = float32_to_float16(k26[k]);
                g00[3] = float32_to_float16(k36[k]);
                g00[4] = float32_to_float16(k46[k]);
                g00[5] = float32_to_float16(k56[k]);
                g00[6] = float32_to_float16(k66[k]);
                g00[7] = float32_to_float16(k76[k]);

                g00 += 8;
                g00[0] = float32_to_float16(k07[k]);
                g00[1] = float32_to_float16(k17[k]);
                g00[2] = float32_to_float16(k27[k]);
                g00[3] = float32_to_float16(k37[k]);
                g00[4] = float32_to_float16(k47[k]);
                g00[5] = float32_to_float16(k57[k]);
                g00[6] = float32_to_float16(k67[k]);
                g00[7] = float32_to_float16(k77[k]);

                g00 += 8;
            }
        }
    }
}
static void conv2x2s1_fp16_pack8_avx(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int inch = bottom_blob.c;
    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;
    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        __m256 _bias0 = bias ? _mm256_loadu_ps((const float*)bias + p * 8) : _mm256_set1_ps(0.f);
        out0.fill(_bias0);

        for (int q = 0; q < inch; q++)
        {
            float* outptr0 = out0.row(0);

            const Mat img0 = bottom_blob.channel(q);

            const float* r0 = img0.row(0);
            const float* r1 = img0.row(1);

            const unsigned short* kptr = (const unsigned short*)kernel.channel(p).row(q);
            // const float* kptr = (const float*)kernel + 4 * inch * p * 64;

            int i = 0;
            for (; i < outh; i++)
            {
                int j = 0;

                for (; j + 1 < outw; j += 2)
                {
                    __m256 _sum0 = _mm256_loadu_ps(outptr0);
                    __m256 _sum1 = _mm256_loadu_ps(outptr0 + 8);

                    __m256 _r00 = _mm256_broadcast_ss(r0);
                    __m256 _r01 = _mm256_broadcast_ss(r0 + 1);
                    __m256 _r02 = _mm256_broadcast_ss(r0 + 2);
                    __m256 _r03 = _mm256_broadcast_ss(r0 + 3);
                    __m256 _r04 = _mm256_broadcast_ss(r0 + 4);
                    __m256 _r05 = _mm256_broadcast_ss(r0 + 5);
                    __m256 _r06 = _mm256_broadcast_ss(r0 + 6);
                    __m256 _r07 = _mm256_broadcast_ss(r0 + 7);
                    r0 += 8;

                    __m256 _k00 = loadfp16(kptr);
                    __m256 _k01 = loadfp16(kptr + 8);
                    __m256 _k02 = loadfp16(kptr + 16);
                    __m256 _k03 = loadfp16(kptr + 24);
                    kptr += 32;

                    _sum0 = _mm256_fmadd_ps(_k00, _r00, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k01, _r01, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k02, _r02, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k03, _r03, _sum0);

                    __m256 _k04 = loadfp16(kptr);
                    __m256 _k05 = loadfp16(kptr + 8);
                    __m256 _k06 = loadfp16(kptr + 16);
                    __m256 _k07 = loadfp16(kptr + 24);
                    kptr += 32;

                    _sum0 = _mm256_fmadd_ps(_k04, _r04, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k05, _r05, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k06, _r06, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k07, _r07, _sum0);

                    //========================================

                    _r00 = _mm256_broadcast_ss(r0);
                    _r01 = _mm256_broadcast_ss(r0 + 1);
                    _r02 = _mm256_broadcast_ss(r0 + 2);
                    _r03 = _mm256_broadcast_ss(r0 + 3);
                    _r04 = _mm256_broadcast_ss(r0 + 4);
                    _r05 = _mm256_broadcast_ss(r0 + 5);
                    _r06 = _mm256_broadcast_ss(r0 + 6);
                    _r07 = _mm256_broadcast_ss(r0 + 7);
                    r0 += 8;

                    _sum1 = _mm256_fmadd_ps(_k00, _r00, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k01, _r01, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k02, _r02, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k03, _r03, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k04, _r04, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k05, _r05, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k06, _r06, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k07, _r07, _sum1);

                    _k00 = loadfp16(kptr);
                    _k01 = loadfp16(kptr + 8);
                    _k02 = loadfp16(kptr + 16);
                    _k03 = loadfp16(kptr + 24);
                    kptr += 32;

                    _sum0 = _mm256_fmadd_ps(_k00, _r00, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k01, _r01, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k02, _r02, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k03, _r03, _sum0);

                    _k04 = loadfp16(kptr);
                    _k05 = loadfp16(kptr + 8);
                    _k06 = loadfp16(kptr + 16);
                    _k07 = loadfp16(kptr + 24);
                    kptr += 32;

                    _sum0 = _mm256_fmadd_ps(_k04, _r04, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k05, _r05, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k06, _r06, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k07, _r07, _sum0);

                    _r00 = _mm256_broadcast_ss(r0);
                    _r01 = _mm256_broadcast_ss(r0 + 1);
                    _r02 = _mm256_broadcast_ss(r0 + 2);
                    _r03 = _mm256_broadcast_ss(r0 + 3);
                    _r04 = _mm256_broadcast_ss(r0 + 4);
                    _r05 = _mm256_broadcast_ss(r0 + 5);
                    _r06 = _mm256_broadcast_ss(r0 + 6);
                    _r07 = _mm256_broadcast_ss(r0 + 7);

                    _sum1 = _mm256_fmadd_ps(_k00, _r00, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k01, _r01, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k02, _r02, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k03, _r03, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k04, _r04, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k05, _r05, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k06, _r06, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k07, _r07, _sum1);
                    //===============

                    __m256 _r10 = _mm256_broadcast_ss(r1);
                    __m256 _r11 = _mm256_broadcast_ss(r1 + 1);
                    __m256 _r12 = _mm256_broadcast_ss(r1 + 2);
                    __m256 _r13 = _mm256_broadcast_ss(r1 + 3);
                    __m256 _r14 = _mm256_broadcast_ss(r1 + 4);
                    __m256 _r15 = _mm256_broadcast_ss(r1 + 5);
                    __m256 _r16 = _mm256_broadcast_ss(r1 + 6);
                    __m256 _r17 = _mm256_broadcast_ss(r1 + 7);

                    __m256 _k10 = loadfp16(kptr);
                    __m256 _k11 = loadfp16(kptr + 8);
                    __m256 _k12 = loadfp16(kptr + 16);
                    __m256 _k13 = loadfp16(kptr + 24);
                    kptr += 32;

                    _sum0 = _mm256_fmadd_ps(_k10, _r10, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k11, _r11, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k12, _r12, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k13, _r13, _sum0);

                    __m256 _k14 = loadfp16(kptr);
                    __m256 _k15 = loadfp16(kptr + 8);
                    __m256 _k16 = loadfp16(kptr + 16);
                    __m256 _k17 = loadfp16(kptr + 24);
                    kptr += 32;

                    _sum0 = _mm256_fmadd_ps(_k14, _r14, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k15, _r15, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k16, _r16, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k17, _r17, _sum0);

                    //=======================================
                    r1 += 8;
                    _r10 = _mm256_broadcast_ss(r1);
                    _r11 = _mm256_broadcast_ss(r1 + 1);
                    _r12 = _mm256_broadcast_ss(r1 + 2);
                    _r13 = _mm256_broadcast_ss(r1 + 3);
                    _r14 = _mm256_broadcast_ss(r1 + 4);
                    _r15 = _mm256_broadcast_ss(r1 + 5);
                    _r16 = _mm256_broadcast_ss(r1 + 6);
                    _r17 = _mm256_broadcast_ss(r1 + 7);

                    _sum1 = _mm256_fmadd_ps(_k10, _r10, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k11, _r11, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k12, _r12, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k13, _r13, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k14, _r14, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k15, _r15, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k16, _r16, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k17, _r17, _sum1);

                    _k10 = loadfp16(kptr);
                    _k11 = loadfp16(kptr + 8);
                    _k12 = loadfp16(kptr + 16);
                    _k13 = loadfp16(kptr + 24);
                    kptr += 32;

                    _sum0 = _mm256_fmadd_ps(_k10, _r10, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k11, _r11, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k12, _r12, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k13, _r13, _sum0);

                    _k14 = loadfp16(kptr);
                    _k15 = loadfp16(kptr + 8);
                    _k16 = loadfp16(kptr + 16);
                    _k17 = loadfp16(kptr + 24);
                    _sum0 = _mm256_fmadd_ps(_k14, _r14, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k15, _r15, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k16, _r16, _sum0);
                    _sum0 = _mm256_fmadd_ps(_k17, _r17, _sum0);

                    r1 += 8;
                    _r10 = _mm256_broadcast_ss(r1);
                    _r11 = _mm256_broadcast_ss(r1 + 1);
                    _r12 = _mm256_broadcast_ss(r1 + 2);
                    _r13 = _mm256_broadcast_ss(r1 + 3);
                    _r14 = _mm256_broadcast_ss(r1 + 4);
                    _r15 = _mm256_broadcast_ss(r1 + 5);
                    _r16 = _mm256_broadcast_ss(r1 + 6);
                    _r17 = _mm256_broadcast_ss(r1 + 7);

                    _sum1 = _mm256_fmadd_ps(_k10, _r10, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k11, _r11, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k12, _r12, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k13, _r13, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k14, _r14, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k15, _r15, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k16, _r16, _sum1);
                    _sum1 = _mm256_fmadd_ps(_k17, _r17, _sum1);

                    kptr -= 224;
                    _mm256_storeu_ps(outptr0, _sum0);
                    _mm256_storeu_ps(outptr0 + 8, _sum1);
                    outptr0 += 16;
                }

                for (; j < outw; j++)
                {
                    __m256 _sum = _mm256_loadu_ps(outptr0);

                    __m256 _r00 = _mm256_broadcast_ss(r0);
                    __m256 _r01 = _mm256_broadcast_ss(r0 + 1);
                    __m256 _r02 = _mm256_broadcast_ss(r0 + 2);
                    __m256 _r03 = _mm256_broadcast_ss(r0 + 3);
                    __m256 _r04 = _mm256_broadcast_ss(r0 + 4);
                    __m256 _r05 = _mm256_broadcast_ss(r0 + 5);
                    __m256 _r06 = _mm256_broadcast_ss(r0 + 6);
                    __m256 _r07 = _mm256_broadcast_ss(r0 + 7);

                    __m256 _k00 = loadfp16(kptr);
                    __m256 _k01 = loadfp16(kptr + 8);
                    __m256 _k02 = loadfp16(kptr + 16);
                    __m256 _k03 = loadfp16(kptr + 24);
                    kptr += 32;

                    _sum = _mm256_fmadd_ps(_k00, _r00, _sum);
                    _sum = _mm256_fmadd_ps(_k01, _r01, _sum);
                    _sum = _mm256_fmadd_ps(_k02, _r02, _sum);
                    _sum = _mm256_fmadd_ps(_k03, _r03, _sum);

                    __m256 _k04 = loadfp16(kptr);
                    __m256 _k05 = loadfp16(kptr + 8);
                    __m256 _k06 = loadfp16(kptr + 16);
                    __m256 _k07 = loadfp16(kptr + 24);
                    kptr += 32;

                    _sum = _mm256_fmadd_ps(_k04, _r04, _sum);
                    _sum = _mm256_fmadd_ps(_k05, _r05, _sum);
                    _sum = _mm256_fmadd_ps(_k06, _r06, _sum);
                    _sum = _mm256_fmadd_ps(_k07, _r07, _sum);

                    //========================================
                    r0 += 8;
                    _r00 = _mm256_broadcast_ss(r0);
                    _r01 = _mm256_broadcast_ss(r0 + 1);
                    _r02 = _mm256_broadcast_ss(r0 + 2);
                    _r03 = _mm256_broadcast_ss(r0 + 3);
                    _r04 = _mm256_broadcast_ss(r0 + 4);
                    _r05 = _mm256_broadcast_ss(r0 + 5);
                    _r06 = _mm256_broadcast_ss(r0 + 6);
                    _r07 = _mm256_broadcast_ss(r0 + 7);

                    _k00 = loadfp16(kptr);
                    _k01 = loadfp16(kptr + 8);
                    _k02 = loadfp16(kptr + 16);
                    _k03 = loadfp16(kptr + 24);
                    kptr += 32;

                    _sum = _mm256_fmadd_ps(_k00, _r00, _sum);
                    _sum = _mm256_fmadd_ps(_k01, _r01, _sum);
                    _sum = _mm256_fmadd_ps(_k02, _r02, _sum);
                    _sum = _mm256_fmadd_ps(_k03, _r03, _sum);

                    _k04 = loadfp16(kptr);
                    _k05 = loadfp16(kptr + 8);
                    _k06 = loadfp16(kptr + 16);
                    _k07 = loadfp16(kptr + 24);
                    kptr += 32;

                    _sum = _mm256_fmadd_ps(_k04, _r04, _sum);
                    _sum = _mm256_fmadd_ps(_k05, _r05, _sum);
                    _sum = _mm256_fmadd_ps(_k06, _r06, _sum);
                    _sum = _mm256_fmadd_ps(_k07, _r07, _sum);
                    //===============

                    __m256 _r10 = _mm256_broadcast_ss(r1);
                    __m256 _r11 = _mm256_broadcast_ss(r1 + 1);
                    __m256 _r12 = _mm256_broadcast_ss(r1 + 2);
                    __m256 _r13 = _mm256_broadcast_ss(r1 + 3);
                    __m256 _r14 = _mm256_broadcast_ss(r1 + 4);
                    __m256 _r15 = _mm256_broadcast_ss(r1 + 5);
                    __m256 _r16 = _mm256_broadcast_ss(r1 + 6);
                    __m256 _r17 = _mm256_broadcast_ss(r1 + 7);

                    __m256 _k10 = loadfp16(kptr);
                    __m256 _k11 = loadfp16(kptr + 8);
                    __m256 _k12 = loadfp16(kptr + 16);
                    __m256 _k13 = loadfp16(kptr + 24);
                    kptr += 32;

                    _sum = _mm256_fmadd_ps(_k10, _r10, _sum);
                    _sum = _mm256_fmadd_ps(_k11, _r11, _sum);
                    _sum = _mm256_fmadd_ps(_k12, _r12, _sum);
                    _sum = _mm256_fmadd_ps(_k13, _r13, _sum);

                    __m256 _k14 = loadfp16(kptr);
                    __m256 _k15 = loadfp16(kptr + 8);
                    __m256 _k16 = loadfp16(kptr + 16);
                    __m256 _k17 = loadfp16(kptr + 24);
                    kptr += 32;

                    _sum = _mm256_fmadd_ps(_k14, _r14, _sum);
                    _sum = _mm256_fmadd_ps(_k15, _r15, _sum);
                    _sum = _mm256_fmadd_ps(_k16, _r16, _sum);
                    _sum = _mm256_fmadd_ps(_k17, _r17, _sum);

                    //=======================================
                    r1 += 8;
                    _r10 = _mm256_broadcast_ss(r1);
                    _r11 = _mm256_broadcast_ss(r1 + 1);
                    _r12 = _mm256_broadcast_ss(r1 + 2);
                    _r13 = _mm256_broadcast_ss(r1 + 3);
                    _r14 = _mm256_broadcast_ss(r1 + 4);
                    _r15 = _mm256_broadcast_ss(r1 + 5);
                    _r16 = _mm256_broadcast_ss(r1 + 6);
                    _r17 = _mm256_broadcast_ss(r1 + 7);

                    _k10 = loadfp16(kptr);
                    _k11 = loadfp16(kptr + 8);
                    _k12 = loadfp16(kptr + 16);
                    _k13 = loadfp16(kptr + 24);
                    kptr += 32;

                    _sum = _mm256_fmadd_ps(_k10, _r10, _sum);
                    _sum = _mm256_fmadd_ps(_k11, _r11, _sum);
                    _sum = _mm256_fmadd_ps(_k12, _r12, _sum);
                    _sum = _mm256_fmadd_ps(_k13, _r13, _sum);

                    _k14 = loadfp16(kptr);
                    _k15 = loadfp16(kptr + 8);
                    _k16 = loadfp16(kptr + 16);
                    _k17 = loadfp16(kptr + 24);
                    _sum = _mm256_fmadd_ps(_k14, _r14, _sum);
                    _sum = _mm256_fmadd_ps(_k15, _r15, _sum);
                    _sum = _mm256_fmadd_ps(_k16, _r16, _sum);
                    _sum = _mm256_fmadd_ps(_k17, _r17, _sum);

                    kptr -= 224;
                    _mm256_storeu_ps(outptr0, _sum);
                    outptr0 += 8;
                }

                r0 += 8;
                r1 += 8;
            }
        }
    }
}
