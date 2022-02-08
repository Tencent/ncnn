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

static void conv3x3s1_pack8_avx(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out = top_blob.channel(p);

        __m256 _bias0 = bias ? _mm256_loadu_ps(bias + p * 8) : _mm256_setzero_ps();
        out.fill(_bias0);

        for (int q = 0; q < inch; q++)
        {
            float* outptr = out;

            const Mat img0 = bottom_blob.channel(q);

            const float* r0 = img0.row(0);
            const float* r1 = img0.row(1);
            const float* r2 = img0.row(2);

            const float* kptr = kernel.channel(p).row(q);

            int i = 0;
            for (; i < outh; i++)
            {
                int j = 0;
                for (; j + 1 < outw; j += 2)
                {
                    __m256 _sum00 = _mm256_loadu_ps(outptr);
                    __m256 _sum01 = _mm256_setzero_ps();
                    __m256 _sum10 = _mm256_loadu_ps(outptr + 8);
                    __m256 _sum11 = _mm256_setzero_ps();

                    __m256 _r000 = _mm256_broadcast_ss(r0 + 0);
                    __m256 _r001 = _mm256_broadcast_ss(r0 + 1);
                    __m256 _r002 = _mm256_broadcast_ss(r0 + 2);
                    __m256 _r003 = _mm256_broadcast_ss(r0 + 3);
                    __m256 _r004 = _mm256_broadcast_ss(r0 + 4);
                    __m256 _r005 = _mm256_broadcast_ss(r0 + 5);
                    __m256 _r006 = _mm256_broadcast_ss(r0 + 6);
                    __m256 _r007 = _mm256_broadcast_ss(r0 + 7);

                    __m256 _k00 = _mm256_loadu_ps(kptr);
                    __m256 _k01 = _mm256_loadu_ps(kptr + 8);
                    __m256 _k02 = _mm256_loadu_ps(kptr + 16);
                    __m256 _k03 = _mm256_loadu_ps(kptr + 24);
                    __m256 _k04 = _mm256_loadu_ps(kptr + 32);
                    __m256 _k05 = _mm256_loadu_ps(kptr + 40);
                    __m256 _k06 = _mm256_loadu_ps(kptr + 48);
                    __m256 _k07 = _mm256_loadu_ps(kptr + 56);

                    kptr += 64;

                    _sum00 = _mm256_comp_fmadd_ps(_r000, _k00, _sum00);
                    _sum01 = _mm256_comp_fmadd_ps(_r001, _k01, _sum01);
                    _sum00 = _mm256_comp_fmadd_ps(_r002, _k02, _sum00);
                    _sum01 = _mm256_comp_fmadd_ps(_r003, _k03, _sum01);
                    _sum00 = _mm256_comp_fmadd_ps(_r004, _k04, _sum00);
                    _sum01 = _mm256_comp_fmadd_ps(_r005, _k05, _sum01);
                    _sum00 = _mm256_comp_fmadd_ps(_r006, _k06, _sum00);
                    _sum01 = _mm256_comp_fmadd_ps(_r007, _k07, _sum01);

                    __m256 _r010 = _mm256_broadcast_ss(r0 + 8);
                    __m256 _r011 = _mm256_broadcast_ss(r0 + 9);
                    __m256 _r012 = _mm256_broadcast_ss(r0 + 10);
                    __m256 _r013 = _mm256_broadcast_ss(r0 + 11);
                    __m256 _r014 = _mm256_broadcast_ss(r0 + 12);
                    __m256 _r015 = _mm256_broadcast_ss(r0 + 13);
                    __m256 _r016 = _mm256_broadcast_ss(r0 + 14);
                    __m256 _r017 = _mm256_broadcast_ss(r0 + 15);

                    _sum10 = _mm256_comp_fmadd_ps(_r010, _k00, _sum10);
                    _sum11 = _mm256_comp_fmadd_ps(_r011, _k01, _sum11);
                    _sum10 = _mm256_comp_fmadd_ps(_r012, _k02, _sum10);
                    _sum11 = _mm256_comp_fmadd_ps(_r013, _k03, _sum11);
                    _sum10 = _mm256_comp_fmadd_ps(_r014, _k04, _sum10);
                    _sum11 = _mm256_comp_fmadd_ps(_r015, _k05, _sum11);
                    _sum10 = _mm256_comp_fmadd_ps(_r016, _k06, _sum10);
                    _sum11 = _mm256_comp_fmadd_ps(_r017, _k07, _sum11);

                    __m256 _k10 = _mm256_loadu_ps(kptr);
                    __m256 _k11 = _mm256_loadu_ps(kptr + 8);
                    __m256 _k12 = _mm256_loadu_ps(kptr + 16);
                    __m256 _k13 = _mm256_loadu_ps(kptr + 24);
                    __m256 _k14 = _mm256_loadu_ps(kptr + 32);
                    __m256 _k15 = _mm256_loadu_ps(kptr + 40);
                    __m256 _k16 = _mm256_loadu_ps(kptr + 48);
                    __m256 _k17 = _mm256_loadu_ps(kptr + 56);

                    kptr += 64;

                    _sum00 = _mm256_comp_fmadd_ps(_r010, _k10, _sum00);
                    _sum01 = _mm256_comp_fmadd_ps(_r011, _k11, _sum01);
                    _sum00 = _mm256_comp_fmadd_ps(_r012, _k12, _sum00);
                    _sum01 = _mm256_comp_fmadd_ps(_r013, _k13, _sum01);
                    _sum00 = _mm256_comp_fmadd_ps(_r014, _k14, _sum00);
                    _sum01 = _mm256_comp_fmadd_ps(_r015, _k15, _sum01);
                    _sum00 = _mm256_comp_fmadd_ps(_r016, _k16, _sum00);
                    _sum01 = _mm256_comp_fmadd_ps(_r017, _k17, _sum01);

                    __m256 _r020 = _mm256_broadcast_ss(r0 + 16);
                    __m256 _r021 = _mm256_broadcast_ss(r0 + 17);
                    __m256 _r022 = _mm256_broadcast_ss(r0 + 18);
                    __m256 _r023 = _mm256_broadcast_ss(r0 + 19);
                    __m256 _r024 = _mm256_broadcast_ss(r0 + 20);
                    __m256 _r025 = _mm256_broadcast_ss(r0 + 21);
                    __m256 _r026 = _mm256_broadcast_ss(r0 + 22);
                    __m256 _r027 = _mm256_broadcast_ss(r0 + 23);

                    _sum10 = _mm256_comp_fmadd_ps(_r020, _k10, _sum10);
                    _sum11 = _mm256_comp_fmadd_ps(_r021, _k11, _sum11);
                    _sum10 = _mm256_comp_fmadd_ps(_r022, _k12, _sum10);
                    _sum11 = _mm256_comp_fmadd_ps(_r023, _k13, _sum11);
                    _sum10 = _mm256_comp_fmadd_ps(_r024, _k14, _sum10);
                    _sum11 = _mm256_comp_fmadd_ps(_r025, _k15, _sum11);
                    _sum10 = _mm256_comp_fmadd_ps(_r026, _k16, _sum10);
                    _sum11 = _mm256_comp_fmadd_ps(_r027, _k17, _sum11);

                    __m256 _k20 = _mm256_loadu_ps(kptr);
                    __m256 _k21 = _mm256_loadu_ps(kptr + 8);
                    __m256 _k22 = _mm256_loadu_ps(kptr + 16);
                    __m256 _k23 = _mm256_loadu_ps(kptr + 24);
                    __m256 _k24 = _mm256_loadu_ps(kptr + 32);
                    __m256 _k25 = _mm256_loadu_ps(kptr + 40);
                    __m256 _k26 = _mm256_loadu_ps(kptr + 48);
                    __m256 _k27 = _mm256_loadu_ps(kptr + 56);

                    kptr += 64;

                    _sum00 = _mm256_comp_fmadd_ps(_r020, _k20, _sum00);
                    _sum01 = _mm256_comp_fmadd_ps(_r021, _k21, _sum01);
                    _sum00 = _mm256_comp_fmadd_ps(_r022, _k22, _sum00);
                    _sum01 = _mm256_comp_fmadd_ps(_r023, _k23, _sum01);
                    _sum00 = _mm256_comp_fmadd_ps(_r024, _k24, _sum00);
                    _sum01 = _mm256_comp_fmadd_ps(_r025, _k25, _sum01);
                    _sum00 = _mm256_comp_fmadd_ps(_r026, _k26, _sum00);
                    _sum01 = _mm256_comp_fmadd_ps(_r027, _k27, _sum01);

                    __m256 _r030 = _mm256_broadcast_ss(r0 + 24);
                    __m256 _r031 = _mm256_broadcast_ss(r0 + 25);
                    __m256 _r032 = _mm256_broadcast_ss(r0 + 26);
                    __m256 _r033 = _mm256_broadcast_ss(r0 + 27);
                    __m256 _r034 = _mm256_broadcast_ss(r0 + 28);
                    __m256 _r035 = _mm256_broadcast_ss(r0 + 29);
                    __m256 _r036 = _mm256_broadcast_ss(r0 + 30);
                    __m256 _r037 = _mm256_broadcast_ss(r0 + 31);

                    _sum10 = _mm256_comp_fmadd_ps(_r030, _k20, _sum10);
                    _sum11 = _mm256_comp_fmadd_ps(_r031, _k21, _sum11);
                    _sum10 = _mm256_comp_fmadd_ps(_r032, _k22, _sum10);
                    _sum11 = _mm256_comp_fmadd_ps(_r033, _k23, _sum11);
                    _sum10 = _mm256_comp_fmadd_ps(_r034, _k24, _sum10);
                    _sum11 = _mm256_comp_fmadd_ps(_r035, _k25, _sum11);
                    _sum10 = _mm256_comp_fmadd_ps(_r036, _k26, _sum10);
                    _sum11 = _mm256_comp_fmadd_ps(_r037, _k27, _sum11);

                    __m256 _r100 = _mm256_broadcast_ss(r1 + 0);
                    __m256 _r101 = _mm256_broadcast_ss(r1 + 1);
                    __m256 _r102 = _mm256_broadcast_ss(r1 + 2);
                    __m256 _r103 = _mm256_broadcast_ss(r1 + 3);
                    __m256 _r104 = _mm256_broadcast_ss(r1 + 4);
                    __m256 _r105 = _mm256_broadcast_ss(r1 + 5);
                    __m256 _r106 = _mm256_broadcast_ss(r1 + 6);
                    __m256 _r107 = _mm256_broadcast_ss(r1 + 7);

                    __m256 _k30 = _mm256_loadu_ps(kptr);
                    __m256 _k31 = _mm256_loadu_ps(kptr + 8);
                    __m256 _k32 = _mm256_loadu_ps(kptr + 16);
                    __m256 _k33 = _mm256_loadu_ps(kptr + 24);
                    __m256 _k34 = _mm256_loadu_ps(kptr + 32);
                    __m256 _k35 = _mm256_loadu_ps(kptr + 40);
                    __m256 _k36 = _mm256_loadu_ps(kptr + 48);
                    __m256 _k37 = _mm256_loadu_ps(kptr + 56);

                    kptr += 64;

                    _sum00 = _mm256_comp_fmadd_ps(_r100, _k30, _sum00);
                    _sum01 = _mm256_comp_fmadd_ps(_r101, _k31, _sum01);
                    _sum00 = _mm256_comp_fmadd_ps(_r102, _k32, _sum00);
                    _sum01 = _mm256_comp_fmadd_ps(_r103, _k33, _sum01);
                    _sum00 = _mm256_comp_fmadd_ps(_r104, _k34, _sum00);
                    _sum01 = _mm256_comp_fmadd_ps(_r105, _k35, _sum01);
                    _sum00 = _mm256_comp_fmadd_ps(_r106, _k36, _sum00);
                    _sum01 = _mm256_comp_fmadd_ps(_r107, _k37, _sum01);

                    __m256 _r110 = _mm256_broadcast_ss(r1 + 8);
                    __m256 _r111 = _mm256_broadcast_ss(r1 + 9);
                    __m256 _r112 = _mm256_broadcast_ss(r1 + 10);
                    __m256 _r113 = _mm256_broadcast_ss(r1 + 11);
                    __m256 _r114 = _mm256_broadcast_ss(r1 + 12);
                    __m256 _r115 = _mm256_broadcast_ss(r1 + 13);
                    __m256 _r116 = _mm256_broadcast_ss(r1 + 14);
                    __m256 _r117 = _mm256_broadcast_ss(r1 + 15);

                    _sum10 = _mm256_comp_fmadd_ps(_r110, _k30, _sum10);
                    _sum11 = _mm256_comp_fmadd_ps(_r111, _k31, _sum11);
                    _sum10 = _mm256_comp_fmadd_ps(_r112, _k32, _sum10);
                    _sum11 = _mm256_comp_fmadd_ps(_r113, _k33, _sum11);
                    _sum10 = _mm256_comp_fmadd_ps(_r114, _k34, _sum10);
                    _sum11 = _mm256_comp_fmadd_ps(_r115, _k35, _sum11);
                    _sum10 = _mm256_comp_fmadd_ps(_r116, _k36, _sum10);
                    _sum11 = _mm256_comp_fmadd_ps(_r117, _k37, _sum11);

                    __m256 _k40 = _mm256_loadu_ps(kptr);
                    __m256 _k41 = _mm256_loadu_ps(kptr + 8);
                    __m256 _k42 = _mm256_loadu_ps(kptr + 16);
                    __m256 _k43 = _mm256_loadu_ps(kptr + 24);
                    __m256 _k44 = _mm256_loadu_ps(kptr + 32);
                    __m256 _k45 = _mm256_loadu_ps(kptr + 40);
                    __m256 _k46 = _mm256_loadu_ps(kptr + 48);
                    __m256 _k47 = _mm256_loadu_ps(kptr + 56);

                    kptr += 64;

                    _sum00 = _mm256_comp_fmadd_ps(_r110, _k40, _sum00);
                    _sum01 = _mm256_comp_fmadd_ps(_r111, _k41, _sum01);
                    _sum00 = _mm256_comp_fmadd_ps(_r112, _k42, _sum00);
                    _sum01 = _mm256_comp_fmadd_ps(_r113, _k43, _sum01);
                    _sum00 = _mm256_comp_fmadd_ps(_r114, _k44, _sum00);
                    _sum01 = _mm256_comp_fmadd_ps(_r115, _k45, _sum01);
                    _sum00 = _mm256_comp_fmadd_ps(_r116, _k46, _sum00);
                    _sum01 = _mm256_comp_fmadd_ps(_r117, _k47, _sum01);

                    __m256 _r120 = _mm256_broadcast_ss(r1 + 16);
                    __m256 _r121 = _mm256_broadcast_ss(r1 + 17);
                    __m256 _r122 = _mm256_broadcast_ss(r1 + 18);
                    __m256 _r123 = _mm256_broadcast_ss(r1 + 19);
                    __m256 _r124 = _mm256_broadcast_ss(r1 + 20);
                    __m256 _r125 = _mm256_broadcast_ss(r1 + 21);
                    __m256 _r126 = _mm256_broadcast_ss(r1 + 22);
                    __m256 _r127 = _mm256_broadcast_ss(r1 + 23);

                    _sum10 = _mm256_comp_fmadd_ps(_r120, _k40, _sum10);
                    _sum11 = _mm256_comp_fmadd_ps(_r121, _k41, _sum11);
                    _sum10 = _mm256_comp_fmadd_ps(_r122, _k42, _sum10);
                    _sum11 = _mm256_comp_fmadd_ps(_r123, _k43, _sum11);
                    _sum10 = _mm256_comp_fmadd_ps(_r124, _k44, _sum10);
                    _sum11 = _mm256_comp_fmadd_ps(_r125, _k45, _sum11);
                    _sum10 = _mm256_comp_fmadd_ps(_r126, _k46, _sum10);
                    _sum11 = _mm256_comp_fmadd_ps(_r127, _k47, _sum11);

                    __m256 _k50 = _mm256_loadu_ps(kptr);
                    __m256 _k51 = _mm256_loadu_ps(kptr + 8);
                    __m256 _k52 = _mm256_loadu_ps(kptr + 16);
                    __m256 _k53 = _mm256_loadu_ps(kptr + 24);
                    __m256 _k54 = _mm256_loadu_ps(kptr + 32);
                    __m256 _k55 = _mm256_loadu_ps(kptr + 40);
                    __m256 _k56 = _mm256_loadu_ps(kptr + 48);
                    __m256 _k57 = _mm256_loadu_ps(kptr + 56);

                    kptr += 64;

                    _sum00 = _mm256_comp_fmadd_ps(_r120, _k50, _sum00);
                    _sum01 = _mm256_comp_fmadd_ps(_r121, _k51, _sum01);
                    _sum00 = _mm256_comp_fmadd_ps(_r122, _k52, _sum00);
                    _sum01 = _mm256_comp_fmadd_ps(_r123, _k53, _sum01);
                    _sum00 = _mm256_comp_fmadd_ps(_r124, _k54, _sum00);
                    _sum01 = _mm256_comp_fmadd_ps(_r125, _k55, _sum01);
                    _sum00 = _mm256_comp_fmadd_ps(_r126, _k56, _sum00);
                    _sum01 = _mm256_comp_fmadd_ps(_r127, _k57, _sum01);

                    __m256 _r130 = _mm256_broadcast_ss(r1 + 24);
                    __m256 _r131 = _mm256_broadcast_ss(r1 + 25);
                    __m256 _r132 = _mm256_broadcast_ss(r1 + 26);
                    __m256 _r133 = _mm256_broadcast_ss(r1 + 27);
                    __m256 _r134 = _mm256_broadcast_ss(r1 + 28);
                    __m256 _r135 = _mm256_broadcast_ss(r1 + 29);
                    __m256 _r136 = _mm256_broadcast_ss(r1 + 30);
                    __m256 _r137 = _mm256_broadcast_ss(r1 + 31);

                    _sum10 = _mm256_comp_fmadd_ps(_r130, _k50, _sum10);
                    _sum11 = _mm256_comp_fmadd_ps(_r131, _k51, _sum11);
                    _sum10 = _mm256_comp_fmadd_ps(_r132, _k52, _sum10);
                    _sum11 = _mm256_comp_fmadd_ps(_r133, _k53, _sum11);
                    _sum10 = _mm256_comp_fmadd_ps(_r134, _k54, _sum10);
                    _sum11 = _mm256_comp_fmadd_ps(_r135, _k55, _sum11);
                    _sum10 = _mm256_comp_fmadd_ps(_r136, _k56, _sum10);
                    _sum11 = _mm256_comp_fmadd_ps(_r137, _k57, _sum11);

                    __m256 _r200 = _mm256_broadcast_ss(r2 + 0);
                    __m256 _r201 = _mm256_broadcast_ss(r2 + 1);
                    __m256 _r202 = _mm256_broadcast_ss(r2 + 2);
                    __m256 _r203 = _mm256_broadcast_ss(r2 + 3);
                    __m256 _r204 = _mm256_broadcast_ss(r2 + 4);
                    __m256 _r205 = _mm256_broadcast_ss(r2 + 5);
                    __m256 _r206 = _mm256_broadcast_ss(r2 + 6);
                    __m256 _r207 = _mm256_broadcast_ss(r2 + 7);

                    __m256 _k60 = _mm256_loadu_ps(kptr);
                    __m256 _k61 = _mm256_loadu_ps(kptr + 8);
                    __m256 _k62 = _mm256_loadu_ps(kptr + 16);
                    __m256 _k63 = _mm256_loadu_ps(kptr + 24);
                    __m256 _k64 = _mm256_loadu_ps(kptr + 32);
                    __m256 _k65 = _mm256_loadu_ps(kptr + 40);
                    __m256 _k66 = _mm256_loadu_ps(kptr + 48);
                    __m256 _k67 = _mm256_loadu_ps(kptr + 56);

                    kptr += 64;

                    _sum00 = _mm256_comp_fmadd_ps(_r200, _k60, _sum00);
                    _sum01 = _mm256_comp_fmadd_ps(_r201, _k61, _sum01);
                    _sum00 = _mm256_comp_fmadd_ps(_r202, _k62, _sum00);
                    _sum01 = _mm256_comp_fmadd_ps(_r203, _k63, _sum01);
                    _sum00 = _mm256_comp_fmadd_ps(_r204, _k64, _sum00);
                    _sum01 = _mm256_comp_fmadd_ps(_r205, _k65, _sum01);
                    _sum00 = _mm256_comp_fmadd_ps(_r206, _k66, _sum00);
                    _sum01 = _mm256_comp_fmadd_ps(_r207, _k67, _sum01);

                    __m256 _r210 = _mm256_broadcast_ss(r2 + 8);
                    __m256 _r211 = _mm256_broadcast_ss(r2 + 9);
                    __m256 _r212 = _mm256_broadcast_ss(r2 + 10);
                    __m256 _r213 = _mm256_broadcast_ss(r2 + 11);
                    __m256 _r214 = _mm256_broadcast_ss(r2 + 12);
                    __m256 _r215 = _mm256_broadcast_ss(r2 + 13);
                    __m256 _r216 = _mm256_broadcast_ss(r2 + 14);
                    __m256 _r217 = _mm256_broadcast_ss(r2 + 15);

                    _sum10 = _mm256_comp_fmadd_ps(_r210, _k60, _sum10);
                    _sum11 = _mm256_comp_fmadd_ps(_r211, _k61, _sum11);
                    _sum10 = _mm256_comp_fmadd_ps(_r212, _k62, _sum10);
                    _sum11 = _mm256_comp_fmadd_ps(_r213, _k63, _sum11);
                    _sum10 = _mm256_comp_fmadd_ps(_r214, _k64, _sum10);
                    _sum11 = _mm256_comp_fmadd_ps(_r215, _k65, _sum11);
                    _sum10 = _mm256_comp_fmadd_ps(_r216, _k66, _sum10);
                    _sum11 = _mm256_comp_fmadd_ps(_r217, _k67, _sum11);

                    __m256 _k70 = _mm256_loadu_ps(kptr);
                    __m256 _k71 = _mm256_loadu_ps(kptr + 8);
                    __m256 _k72 = _mm256_loadu_ps(kptr + 16);
                    __m256 _k73 = _mm256_loadu_ps(kptr + 24);
                    __m256 _k74 = _mm256_loadu_ps(kptr + 32);
                    __m256 _k75 = _mm256_loadu_ps(kptr + 40);
                    __m256 _k76 = _mm256_loadu_ps(kptr + 48);
                    __m256 _k77 = _mm256_loadu_ps(kptr + 56);

                    kptr += 64;

                    _sum00 = _mm256_comp_fmadd_ps(_r210, _k70, _sum00);
                    _sum01 = _mm256_comp_fmadd_ps(_r211, _k71, _sum01);
                    _sum00 = _mm256_comp_fmadd_ps(_r212, _k72, _sum00);
                    _sum01 = _mm256_comp_fmadd_ps(_r213, _k73, _sum01);
                    _sum00 = _mm256_comp_fmadd_ps(_r214, _k74, _sum00);
                    _sum01 = _mm256_comp_fmadd_ps(_r215, _k75, _sum01);
                    _sum00 = _mm256_comp_fmadd_ps(_r216, _k76, _sum00);
                    _sum01 = _mm256_comp_fmadd_ps(_r217, _k77, _sum01);

                    __m256 _r220 = _mm256_broadcast_ss(r2 + 16);
                    __m256 _r221 = _mm256_broadcast_ss(r2 + 17);
                    __m256 _r222 = _mm256_broadcast_ss(r2 + 18);
                    __m256 _r223 = _mm256_broadcast_ss(r2 + 19);
                    __m256 _r224 = _mm256_broadcast_ss(r2 + 20);
                    __m256 _r225 = _mm256_broadcast_ss(r2 + 21);
                    __m256 _r226 = _mm256_broadcast_ss(r2 + 22);
                    __m256 _r227 = _mm256_broadcast_ss(r2 + 23);

                    _sum10 = _mm256_comp_fmadd_ps(_r220, _k70, _sum10);
                    _sum11 = _mm256_comp_fmadd_ps(_r221, _k71, _sum11);
                    _sum10 = _mm256_comp_fmadd_ps(_r222, _k72, _sum10);
                    _sum11 = _mm256_comp_fmadd_ps(_r223, _k73, _sum11);
                    _sum10 = _mm256_comp_fmadd_ps(_r224, _k74, _sum10);
                    _sum11 = _mm256_comp_fmadd_ps(_r225, _k75, _sum11);
                    _sum10 = _mm256_comp_fmadd_ps(_r226, _k76, _sum10);
                    _sum11 = _mm256_comp_fmadd_ps(_r227, _k77, _sum11);

                    __m256 _k80 = _mm256_loadu_ps(kptr);
                    __m256 _k81 = _mm256_loadu_ps(kptr + 8);
                    __m256 _k82 = _mm256_loadu_ps(kptr + 16);
                    __m256 _k83 = _mm256_loadu_ps(kptr + 24);
                    __m256 _k84 = _mm256_loadu_ps(kptr + 32);
                    __m256 _k85 = _mm256_loadu_ps(kptr + 40);
                    __m256 _k86 = _mm256_loadu_ps(kptr + 48);
                    __m256 _k87 = _mm256_loadu_ps(kptr + 56);

                    _sum00 = _mm256_comp_fmadd_ps(_r220, _k80, _sum00);
                    _sum01 = _mm256_comp_fmadd_ps(_r221, _k81, _sum01);
                    _sum00 = _mm256_comp_fmadd_ps(_r222, _k82, _sum00);
                    _sum01 = _mm256_comp_fmadd_ps(_r223, _k83, _sum01);
                    _sum00 = _mm256_comp_fmadd_ps(_r224, _k84, _sum00);
                    _sum01 = _mm256_comp_fmadd_ps(_r225, _k85, _sum01);
                    _sum00 = _mm256_comp_fmadd_ps(_r226, _k86, _sum00);
                    _sum01 = _mm256_comp_fmadd_ps(_r227, _k87, _sum01);

                    __m256 _r230 = _mm256_broadcast_ss(r2 + 24);
                    __m256 _r231 = _mm256_broadcast_ss(r2 + 25);
                    __m256 _r232 = _mm256_broadcast_ss(r2 + 26);
                    __m256 _r233 = _mm256_broadcast_ss(r2 + 27);
                    __m256 _r234 = _mm256_broadcast_ss(r2 + 28);
                    __m256 _r235 = _mm256_broadcast_ss(r2 + 29);
                    __m256 _r236 = _mm256_broadcast_ss(r2 + 30);
                    __m256 _r237 = _mm256_broadcast_ss(r2 + 31);

                    _sum10 = _mm256_comp_fmadd_ps(_r230, _k80, _sum10);
                    _sum11 = _mm256_comp_fmadd_ps(_r231, _k81, _sum11);
                    _sum10 = _mm256_comp_fmadd_ps(_r232, _k82, _sum10);
                    _sum11 = _mm256_comp_fmadd_ps(_r233, _k83, _sum11);
                    _sum10 = _mm256_comp_fmadd_ps(_r234, _k84, _sum10);
                    _sum11 = _mm256_comp_fmadd_ps(_r235, _k85, _sum11);
                    _sum10 = _mm256_comp_fmadd_ps(_r236, _k86, _sum10);
                    _sum11 = _mm256_comp_fmadd_ps(_r237, _k87, _sum11);

                    kptr -= 64 * 8;

                    _sum00 = _mm256_add_ps(_sum00, _sum01);
                    _sum10 = _mm256_add_ps(_sum10, _sum11);

                    _mm256_storeu_ps(outptr, _sum00);
                    _mm256_storeu_ps(outptr + 8, _sum10);

                    r0 += 16;
                    r1 += 16;
                    r2 += 16;
                    outptr += 16;
                }
                for (; j < outw; j++)
                {
                    __m256 _sum0 = _mm256_loadu_ps(outptr);
                    __m256 _sum1 = _mm256_setzero_ps();

                    __m256 _r000 = _mm256_broadcast_ss(r0 + 0);
                    __m256 _r001 = _mm256_broadcast_ss(r0 + 1);
                    __m256 _r002 = _mm256_broadcast_ss(r0 + 2);
                    __m256 _r003 = _mm256_broadcast_ss(r0 + 3);
                    __m256 _r004 = _mm256_broadcast_ss(r0 + 4);
                    __m256 _r005 = _mm256_broadcast_ss(r0 + 5);
                    __m256 _r006 = _mm256_broadcast_ss(r0 + 6);
                    __m256 _r007 = _mm256_broadcast_ss(r0 + 7);

                    __m256 _k00 = _mm256_loadu_ps(kptr);
                    __m256 _k01 = _mm256_loadu_ps(kptr + 8);
                    __m256 _k02 = _mm256_loadu_ps(kptr + 16);
                    __m256 _k03 = _mm256_loadu_ps(kptr + 24);
                    __m256 _k04 = _mm256_loadu_ps(kptr + 32);
                    __m256 _k05 = _mm256_loadu_ps(kptr + 40);
                    __m256 _k06 = _mm256_loadu_ps(kptr + 48);
                    __m256 _k07 = _mm256_loadu_ps(kptr + 56);

                    kptr += 64;

                    _sum0 = _mm256_comp_fmadd_ps(_r000, _k00, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_r001, _k01, _sum1);
                    _sum0 = _mm256_comp_fmadd_ps(_r002, _k02, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_r003, _k03, _sum1);
                    _sum0 = _mm256_comp_fmadd_ps(_r004, _k04, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_r005, _k05, _sum1);
                    _sum0 = _mm256_comp_fmadd_ps(_r006, _k06, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_r007, _k07, _sum1);

                    __m256 _r010 = _mm256_broadcast_ss(r0 + 8);
                    __m256 _r011 = _mm256_broadcast_ss(r0 + 9);
                    __m256 _r012 = _mm256_broadcast_ss(r0 + 10);
                    __m256 _r013 = _mm256_broadcast_ss(r0 + 11);
                    __m256 _r014 = _mm256_broadcast_ss(r0 + 12);
                    __m256 _r015 = _mm256_broadcast_ss(r0 + 13);
                    __m256 _r016 = _mm256_broadcast_ss(r0 + 14);
                    __m256 _r017 = _mm256_broadcast_ss(r0 + 15);

                    __m256 _k10 = _mm256_loadu_ps(kptr);
                    __m256 _k11 = _mm256_loadu_ps(kptr + 8);
                    __m256 _k12 = _mm256_loadu_ps(kptr + 16);
                    __m256 _k13 = _mm256_loadu_ps(kptr + 24);
                    __m256 _k14 = _mm256_loadu_ps(kptr + 32);
                    __m256 _k15 = _mm256_loadu_ps(kptr + 40);
                    __m256 _k16 = _mm256_loadu_ps(kptr + 48);
                    __m256 _k17 = _mm256_loadu_ps(kptr + 56);

                    kptr += 64;

                    _sum0 = _mm256_comp_fmadd_ps(_r010, _k10, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_r011, _k11, _sum1);
                    _sum0 = _mm256_comp_fmadd_ps(_r012, _k12, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_r013, _k13, _sum1);
                    _sum0 = _mm256_comp_fmadd_ps(_r014, _k14, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_r015, _k15, _sum1);
                    _sum0 = _mm256_comp_fmadd_ps(_r016, _k16, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_r017, _k17, _sum1);

                    __m256 _r020 = _mm256_broadcast_ss(r0 + 16);
                    __m256 _r021 = _mm256_broadcast_ss(r0 + 17);
                    __m256 _r022 = _mm256_broadcast_ss(r0 + 18);
                    __m256 _r023 = _mm256_broadcast_ss(r0 + 19);
                    __m256 _r024 = _mm256_broadcast_ss(r0 + 20);
                    __m256 _r025 = _mm256_broadcast_ss(r0 + 21);
                    __m256 _r026 = _mm256_broadcast_ss(r0 + 22);
                    __m256 _r027 = _mm256_broadcast_ss(r0 + 23);

                    __m256 _k20 = _mm256_loadu_ps(kptr);
                    __m256 _k21 = _mm256_loadu_ps(kptr + 8);
                    __m256 _k22 = _mm256_loadu_ps(kptr + 16);
                    __m256 _k23 = _mm256_loadu_ps(kptr + 24);
                    __m256 _k24 = _mm256_loadu_ps(kptr + 32);
                    __m256 _k25 = _mm256_loadu_ps(kptr + 40);
                    __m256 _k26 = _mm256_loadu_ps(kptr + 48);
                    __m256 _k27 = _mm256_loadu_ps(kptr + 56);

                    kptr += 64;

                    _sum0 = _mm256_comp_fmadd_ps(_r020, _k20, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_r021, _k21, _sum1);
                    _sum0 = _mm256_comp_fmadd_ps(_r022, _k22, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_r023, _k23, _sum1);
                    _sum0 = _mm256_comp_fmadd_ps(_r024, _k24, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_r025, _k25, _sum1);
                    _sum0 = _mm256_comp_fmadd_ps(_r026, _k26, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_r027, _k27, _sum1);

                    __m256 _r100 = _mm256_broadcast_ss(r1 + 0);
                    __m256 _r101 = _mm256_broadcast_ss(r1 + 1);
                    __m256 _r102 = _mm256_broadcast_ss(r1 + 2);
                    __m256 _r103 = _mm256_broadcast_ss(r1 + 3);
                    __m256 _r104 = _mm256_broadcast_ss(r1 + 4);
                    __m256 _r105 = _mm256_broadcast_ss(r1 + 5);
                    __m256 _r106 = _mm256_broadcast_ss(r1 + 6);
                    __m256 _r107 = _mm256_broadcast_ss(r1 + 7);

                    __m256 _k30 = _mm256_loadu_ps(kptr);
                    __m256 _k31 = _mm256_loadu_ps(kptr + 8);
                    __m256 _k32 = _mm256_loadu_ps(kptr + 16);
                    __m256 _k33 = _mm256_loadu_ps(kptr + 24);
                    __m256 _k34 = _mm256_loadu_ps(kptr + 32);
                    __m256 _k35 = _mm256_loadu_ps(kptr + 40);
                    __m256 _k36 = _mm256_loadu_ps(kptr + 48);
                    __m256 _k37 = _mm256_loadu_ps(kptr + 56);

                    kptr += 64;

                    _sum0 = _mm256_comp_fmadd_ps(_r100, _k30, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_r101, _k31, _sum1);
                    _sum0 = _mm256_comp_fmadd_ps(_r102, _k32, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_r103, _k33, _sum1);
                    _sum0 = _mm256_comp_fmadd_ps(_r104, _k34, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_r105, _k35, _sum1);
                    _sum0 = _mm256_comp_fmadd_ps(_r106, _k36, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_r107, _k37, _sum1);

                    __m256 _r110 = _mm256_broadcast_ss(r1 + 8);
                    __m256 _r111 = _mm256_broadcast_ss(r1 + 9);
                    __m256 _r112 = _mm256_broadcast_ss(r1 + 10);
                    __m256 _r113 = _mm256_broadcast_ss(r1 + 11);
                    __m256 _r114 = _mm256_broadcast_ss(r1 + 12);
                    __m256 _r115 = _mm256_broadcast_ss(r1 + 13);
                    __m256 _r116 = _mm256_broadcast_ss(r1 + 14);
                    __m256 _r117 = _mm256_broadcast_ss(r1 + 15);

                    __m256 _k40 = _mm256_loadu_ps(kptr);
                    __m256 _k41 = _mm256_loadu_ps(kptr + 8);
                    __m256 _k42 = _mm256_loadu_ps(kptr + 16);
                    __m256 _k43 = _mm256_loadu_ps(kptr + 24);
                    __m256 _k44 = _mm256_loadu_ps(kptr + 32);
                    __m256 _k45 = _mm256_loadu_ps(kptr + 40);
                    __m256 _k46 = _mm256_loadu_ps(kptr + 48);
                    __m256 _k47 = _mm256_loadu_ps(kptr + 56);

                    kptr += 64;

                    _sum0 = _mm256_comp_fmadd_ps(_r110, _k40, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_r111, _k41, _sum1);
                    _sum0 = _mm256_comp_fmadd_ps(_r112, _k42, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_r113, _k43, _sum1);
                    _sum0 = _mm256_comp_fmadd_ps(_r114, _k44, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_r115, _k45, _sum1);
                    _sum0 = _mm256_comp_fmadd_ps(_r116, _k46, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_r117, _k47, _sum1);

                    __m256 _r120 = _mm256_broadcast_ss(r1 + 16);
                    __m256 _r121 = _mm256_broadcast_ss(r1 + 17);
                    __m256 _r122 = _mm256_broadcast_ss(r1 + 18);
                    __m256 _r123 = _mm256_broadcast_ss(r1 + 19);
                    __m256 _r124 = _mm256_broadcast_ss(r1 + 20);
                    __m256 _r125 = _mm256_broadcast_ss(r1 + 21);
                    __m256 _r126 = _mm256_broadcast_ss(r1 + 22);
                    __m256 _r127 = _mm256_broadcast_ss(r1 + 23);

                    __m256 _k50 = _mm256_loadu_ps(kptr);
                    __m256 _k51 = _mm256_loadu_ps(kptr + 8);
                    __m256 _k52 = _mm256_loadu_ps(kptr + 16);
                    __m256 _k53 = _mm256_loadu_ps(kptr + 24);
                    __m256 _k54 = _mm256_loadu_ps(kptr + 32);
                    __m256 _k55 = _mm256_loadu_ps(kptr + 40);
                    __m256 _k56 = _mm256_loadu_ps(kptr + 48);
                    __m256 _k57 = _mm256_loadu_ps(kptr + 56);

                    kptr += 64;

                    _sum0 = _mm256_comp_fmadd_ps(_r120, _k50, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_r121, _k51, _sum1);
                    _sum0 = _mm256_comp_fmadd_ps(_r122, _k52, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_r123, _k53, _sum1);
                    _sum0 = _mm256_comp_fmadd_ps(_r124, _k54, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_r125, _k55, _sum1);
                    _sum0 = _mm256_comp_fmadd_ps(_r126, _k56, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_r127, _k57, _sum1);

                    __m256 _r200 = _mm256_broadcast_ss(r2 + 0);
                    __m256 _r201 = _mm256_broadcast_ss(r2 + 1);
                    __m256 _r202 = _mm256_broadcast_ss(r2 + 2);
                    __m256 _r203 = _mm256_broadcast_ss(r2 + 3);
                    __m256 _r204 = _mm256_broadcast_ss(r2 + 4);
                    __m256 _r205 = _mm256_broadcast_ss(r2 + 5);
                    __m256 _r206 = _mm256_broadcast_ss(r2 + 6);
                    __m256 _r207 = _mm256_broadcast_ss(r2 + 7);

                    __m256 _k60 = _mm256_loadu_ps(kptr);
                    __m256 _k61 = _mm256_loadu_ps(kptr + 8);
                    __m256 _k62 = _mm256_loadu_ps(kptr + 16);
                    __m256 _k63 = _mm256_loadu_ps(kptr + 24);
                    __m256 _k64 = _mm256_loadu_ps(kptr + 32);
                    __m256 _k65 = _mm256_loadu_ps(kptr + 40);
                    __m256 _k66 = _mm256_loadu_ps(kptr + 48);
                    __m256 _k67 = _mm256_loadu_ps(kptr + 56);

                    kptr += 64;

                    _sum0 = _mm256_comp_fmadd_ps(_r200, _k60, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_r201, _k61, _sum1);
                    _sum0 = _mm256_comp_fmadd_ps(_r202, _k62, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_r203, _k63, _sum1);
                    _sum0 = _mm256_comp_fmadd_ps(_r204, _k64, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_r205, _k65, _sum1);
                    _sum0 = _mm256_comp_fmadd_ps(_r206, _k66, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_r207, _k67, _sum1);

                    __m256 _r210 = _mm256_broadcast_ss(r2 + 8);
                    __m256 _r211 = _mm256_broadcast_ss(r2 + 9);
                    __m256 _r212 = _mm256_broadcast_ss(r2 + 10);
                    __m256 _r213 = _mm256_broadcast_ss(r2 + 11);
                    __m256 _r214 = _mm256_broadcast_ss(r2 + 12);
                    __m256 _r215 = _mm256_broadcast_ss(r2 + 13);
                    __m256 _r216 = _mm256_broadcast_ss(r2 + 14);
                    __m256 _r217 = _mm256_broadcast_ss(r2 + 15);

                    __m256 _k70 = _mm256_loadu_ps(kptr);
                    __m256 _k71 = _mm256_loadu_ps(kptr + 8);
                    __m256 _k72 = _mm256_loadu_ps(kptr + 16);
                    __m256 _k73 = _mm256_loadu_ps(kptr + 24);
                    __m256 _k74 = _mm256_loadu_ps(kptr + 32);
                    __m256 _k75 = _mm256_loadu_ps(kptr + 40);
                    __m256 _k76 = _mm256_loadu_ps(kptr + 48);
                    __m256 _k77 = _mm256_loadu_ps(kptr + 56);

                    kptr += 64;

                    _sum0 = _mm256_comp_fmadd_ps(_r210, _k70, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_r211, _k71, _sum1);
                    _sum0 = _mm256_comp_fmadd_ps(_r212, _k72, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_r213, _k73, _sum1);
                    _sum0 = _mm256_comp_fmadd_ps(_r214, _k74, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_r215, _k75, _sum1);
                    _sum0 = _mm256_comp_fmadd_ps(_r216, _k76, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_r217, _k77, _sum1);

                    __m256 _r220 = _mm256_broadcast_ss(r2 + 16);
                    __m256 _r221 = _mm256_broadcast_ss(r2 + 17);
                    __m256 _r222 = _mm256_broadcast_ss(r2 + 18);
                    __m256 _r223 = _mm256_broadcast_ss(r2 + 19);
                    __m256 _r224 = _mm256_broadcast_ss(r2 + 20);
                    __m256 _r225 = _mm256_broadcast_ss(r2 + 21);
                    __m256 _r226 = _mm256_broadcast_ss(r2 + 22);
                    __m256 _r227 = _mm256_broadcast_ss(r2 + 23);

                    __m256 _k80 = _mm256_loadu_ps(kptr);
                    __m256 _k81 = _mm256_loadu_ps(kptr + 8);
                    __m256 _k82 = _mm256_loadu_ps(kptr + 16);
                    __m256 _k83 = _mm256_loadu_ps(kptr + 24);
                    __m256 _k84 = _mm256_loadu_ps(kptr + 32);
                    __m256 _k85 = _mm256_loadu_ps(kptr + 40);
                    __m256 _k86 = _mm256_loadu_ps(kptr + 48);
                    __m256 _k87 = _mm256_loadu_ps(kptr + 56);

                    _sum0 = _mm256_comp_fmadd_ps(_r220, _k80, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_r221, _k81, _sum1);
                    _sum0 = _mm256_comp_fmadd_ps(_r222, _k82, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_r223, _k83, _sum1);
                    _sum0 = _mm256_comp_fmadd_ps(_r224, _k84, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_r225, _k85, _sum1);
                    _sum0 = _mm256_comp_fmadd_ps(_r226, _k86, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_r227, _k87, _sum1);

                    kptr -= 64 * 8;

                    _sum0 = _mm256_add_ps(_sum0, _sum1);

                    _mm256_storeu_ps(outptr, _sum0);

                    r0 += 8;
                    r1 += 8;
                    r2 += 8;
                    outptr += 8;
                }

                r0 += 16;
                r1 += 16;
                r2 += 16;
            }
        }
    }
}

static void conv3x3s1_winograd64_transform_kernel_pack8_avx(const Mat& kernel, Mat& kernel_tm_pack8, int inch, int outch, const Option& opt)
{
    // winograd63 transform kernel
    Mat kernel_tm;
    kernel_tm.create(8 * 8, inch, outch);

    const float ktm[8][3] = {
        {1.0f, 0.0f, 0.0f},
        {-2.0f / 9, -2.0f / 9, -2.0f / 9},
        {-2.0f / 9, 2.0f / 9, -2.0f / 9},
        {1.0f / 90, 1.0f / 45, 2.0f / 45},
        {1.0f / 90, -1.0f / 45, 2.0f / 45},
        {1.0f / 45, 1.0f / 90, 1.0f / 180},
        {1.0f / 45, -1.0f / 90, 1.0f / 180},
        {0.0f, 0.0f, 1.0f}
    };

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        for (int q = 0; q < inch; q++)
        {
            const float* kernel0 = (const float*)kernel + p * inch * 9 + q * 9;
            float* kernel_tm0 = kernel_tm.channel(p).row(q);

            // transform kernel, transposed
            const float* k0 = kernel0;
            const float* k1 = kernel0 + 3;
            const float* k2 = kernel0 + 6;

            // h
            float tmp[8][3];
            for (int i = 0; i < 8; i++)
            {
                tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
            }

            // v
            for (int j = 0; j < 8; j++)
            {
                float* tmpp = &tmp[j][0];

                for (int i = 0; i < 8; i++)
                {
                    kernel_tm0[j * 8 + i] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                }
            }
        }
    }
    // interleave
    // src = 64-inch-outch
    // dst = 8b-8a-inch/8a-64-outch/8b
    kernel_tm_pack8.create(inch / 8, 64, outch / 8, (size_t)4u * 64, 64);

    int q = 0;
    for (; q + 7 < outch; q += 8)
    {
        const Mat k0 = kernel_tm.channel(q);
        const Mat k1 = kernel_tm.channel(q + 1);
        const Mat k2 = kernel_tm.channel(q + 2);
        const Mat k3 = kernel_tm.channel(q + 3);
        const Mat k4 = kernel_tm.channel(q + 4);
        const Mat k5 = kernel_tm.channel(q + 5);
        const Mat k6 = kernel_tm.channel(q + 6);
        const Mat k7 = kernel_tm.channel(q + 7);

        Mat g0 = kernel_tm_pack8.channel(q / 8);

        for (int k = 0; k < 64; k++)
        {
            float* g00 = g0.row(k);

            for (int p = 0; p + 7 < inch; p += 8)
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

                g00[0] = k00[k];
                g00[1] = k10[k];
                g00[2] = k20[k];
                g00[3] = k30[k];
                g00[4] = k40[k];
                g00[5] = k50[k];
                g00[6] = k60[k];
                g00[7] = k70[k];

                g00[8] = k01[k];
                g00[9] = k11[k];
                g00[10] = k21[k];
                g00[11] = k31[k];
                g00[12] = k41[k];
                g00[13] = k51[k];
                g00[14] = k61[k];
                g00[15] = k71[k];

                g00[16] = k02[k];
                g00[17] = k12[k];
                g00[18] = k22[k];
                g00[19] = k32[k];
                g00[20] = k42[k];
                g00[21] = k52[k];
                g00[22] = k62[k];
                g00[23] = k72[k];

                g00[24] = k03[k];
                g00[25] = k13[k];
                g00[26] = k23[k];
                g00[27] = k33[k];
                g00[28] = k43[k];
                g00[29] = k53[k];
                g00[30] = k63[k];
                g00[31] = k73[k];

                g00[32] = k04[k];
                g00[33] = k14[k];
                g00[34] = k24[k];
                g00[35] = k34[k];
                g00[36] = k44[k];
                g00[37] = k54[k];
                g00[38] = k64[k];
                g00[39] = k74[k];

                g00[40] = k05[k];
                g00[41] = k15[k];
                g00[42] = k25[k];
                g00[43] = k35[k];
                g00[44] = k45[k];
                g00[45] = k55[k];
                g00[46] = k65[k];
                g00[47] = k75[k];

                g00[48] = k06[k];
                g00[49] = k16[k];
                g00[50] = k26[k];
                g00[51] = k36[k];
                g00[52] = k46[k];
                g00[53] = k56[k];
                g00[54] = k66[k];
                g00[55] = k76[k];

                g00[56] = k07[k];
                g00[57] = k17[k];
                g00[58] = k27[k];
                g00[59] = k37[k];
                g00[60] = k47[k];
                g00[61] = k57[k];
                g00[62] = k67[k];
                g00[63] = k77[k];

                g00 += 64;
            }
        }
    }
}

static void conv3x3s1_winograd64_pack8_avx(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;
    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    // pad to 6n+2
    Mat bottom_blob_bordered = bottom_blob;

    outw = (outw + 5) / 6 * 6;
    outh = (outh + 5) / 6 * 6;

    w = outw + 2;
    h = outh + 2;
    copy_make_border(bottom_blob, bottom_blob_bordered, 0, h - bottom_blob.h, 0, w - bottom_blob.w, BORDER_CONSTANT, 0.f, opt);

    const float* bias = _bias;
    // BEGIN transform input
    Mat bottom_blob_tm;
    {
        int w_tm = outw / 6 * 8;
        int h_tm = outh / 6 * 8;

        const int tiles = w_tm / 8 * h_tm / 8;

        bottom_blob_tm.create(tiles, 64, inch, elemsize, elempack, opt.workspace_allocator);

        //         const float itm[8][8] = {
        //             {1.0f,  0.0f, -5.25f,  0.00f,  5.25f,  0.00f, -1.0f, 0.0f},
        //
        //             {0.0f,  1.0f,  1.00f, -4.25f, -4.25f,  1.00f,  1.0f, 0.0f},
        //             {0.0f, -1.0f,  1.00f,  4.25f, -4.25f, -1.00f,  1.0f, 0.0f},
        //
        //             {0.0f,  0.5f,  0.25f, -2.50f, -1.25f,  2.00f,  1.0f, 0.0f},
        //             {0.0f, -0.5f,  0.25f,  2.50f, -1.25f, -2.00f,  1.0f, 0.0f},
        //
        //             {0.0f,  2.0f,  4.00f, -2.50f, -5.00f,  0.50f,  1.0f, 0.0f},
        //             {0.0f, -2.0f,  4.00f,  2.50f, -5.00f, -0.50f,  1.0f, 0.0f},
        //
        //             {0.0f, -1.0f,  0.00f,  5.25f,  0.00f, -5.25f,  0.0f, 1.0f}
        //         };

        // 0 = r00 - r06 + (r04 - r02) * 5.25
        // 7 = r07 - r01 + (r03 - r05) * 5.25

        // 1 = (r02 + r06 - r04 * 4.25) + (r01 - r03 * 4.25 + r05)
        // 2 = (r02 + r06 - r04 * 4.25) - (r01 - r03 * 4.25 + r05)

        // 3 = (r06 + r02 * 0.25 - r04 * 1.25) + (r01 * 0.5 - r03 * 2.5 + r05 * 2)
        // 4 = (r06 + r02 * 0.25 - r04 * 1.25) - (r01 * 0.5 - r03 * 2.5 + r05 * 2)

        // reuse r04 * 1.25
        // reuse r03 * 2.5
        // 5 = (r06 + (r02 - r04 * 1.25) * 4) + (r01 * 2 - r03 * 2.5 + r05 * 0.5)
        // 6 = (r06 + (r02 - r04 * 1.25) * 4) - (r01 * 2 - r03 * 2.5 + r05 * 0.5)

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < inch; q++)
        {
            const Mat img0 = bottom_blob_bordered.channel(q);
            Mat img0_tm = bottom_blob_tm.channel(q);

#ifdef _MSC_VER
            __declspec(align(32))
#else
            __attribute__((aligned(32)))
#endif
            float tmp[8][8][8];

            // tile
            for (int i = 0; i < h_tm / 8; i++)
            {
                for (int j = 0; j < w_tm / 8; j++)
                {
                    const float* r0 = img0.row(i * 6) + (j * 6) * 8;

                    for (int m = 0; m < 8; m++)
                    {
                        __m256 _r00 = _mm256_load_ps(r0);
                        __m256 _r01 = _mm256_load_ps(r0 + 8);
                        __m256 _r02 = _mm256_load_ps(r0 + 16);
                        __m256 _r03 = _mm256_load_ps(r0 + 24);
                        __m256 _r04 = _mm256_load_ps(r0 + 32);
                        __m256 _r05 = _mm256_load_ps(r0 + 40);
                        __m256 _r06 = _mm256_load_ps(r0 + 48);
                        __m256 _r07 = _mm256_load_ps(r0 + 56);

                        __m256 _tmp0m = _mm256_comp_fmadd_ps(_mm256_set1_ps(5.25f), _mm256_sub_ps(_r04, _r02), _mm256_sub_ps(_r00, _r06));
                        __m256 _tmp7m = _mm256_comp_fmadd_ps(_mm256_set1_ps(5.25f), _mm256_sub_ps(_r03, _r05), _mm256_sub_ps(_r07, _r01));
                        _mm256_store_ps(tmp[0][m], _tmp0m);
                        _mm256_store_ps(tmp[7][m], _tmp7m);

                        __m256 _tmp12a = _mm256_comp_fmadd_ps(_mm256_set1_ps(-4.25f), _r04, _mm256_add_ps(_r02, _r06));
                        __m256 _tmp12b = _mm256_comp_fmadd_ps(_mm256_set1_ps(-4.25f), _r03, _mm256_add_ps(_r01, _r05));

                        __m256 _tmp1m = _mm256_add_ps(_tmp12a, _tmp12b);
                        __m256 _tmp2m = _mm256_sub_ps(_tmp12a, _tmp12b);
                        _mm256_store_ps(tmp[1][m], _tmp1m);
                        _mm256_store_ps(tmp[2][m], _tmp2m);

                        __m256 _tmp34a = _mm256_comp_fmadd_ps(_mm256_set1_ps(-1.25f), _r04, _mm256_comp_fmadd_ps(_mm256_set1_ps(0.25f), _r02, _r06));
                        __m256 _tmp34b = _mm256_comp_fmadd_ps(_mm256_set1_ps(2.f), _r05, _mm256_comp_fmadd_ps(_mm256_set1_ps(-2.5f), _r03, _mm256_mul_ps(_r01, _mm256_set1_ps(0.5f))));

                        __m256 _tmp3m = _mm256_add_ps(_tmp34a, _tmp34b);
                        __m256 _tmp4m = _mm256_sub_ps(_tmp34a, _tmp34b);
                        _mm256_store_ps(tmp[3][m], _tmp3m);
                        _mm256_store_ps(tmp[4][m], _tmp4m);

                        __m256 _tmp56a = _mm256_comp_fmadd_ps(_mm256_set1_ps(4.f), _mm256_comp_fmadd_ps(_mm256_set1_ps(-1.25f), _r04, _r02), _r06);
                        __m256 _tmp56b = _mm256_comp_fmadd_ps(_mm256_set1_ps(0.5f), _r05, _mm256_comp_fmadd_ps(_mm256_set1_ps(-2.5f), _r03, _mm256_mul_ps(_r01, _mm256_set1_ps(2.f))));

                        __m256 _tmp5m = _mm256_add_ps(_tmp56a, _tmp56b);
                        __m256 _tmp6m = _mm256_sub_ps(_tmp56a, _tmp56b);
                        _mm256_store_ps(tmp[5][m], _tmp5m);
                        _mm256_store_ps(tmp[6][m], _tmp6m);

                        r0 += w * 8;
                    }

                    float* r0_tm_0 = (float*)img0_tm + (i * w_tm / 8 + j) * 8;
                    float* r0_tm_1 = r0_tm_0 + tiles * 8;
                    float* r0_tm_2 = r0_tm_0 + tiles * 16;
                    float* r0_tm_3 = r0_tm_0 + tiles * 24;
                    float* r0_tm_4 = r0_tm_0 + tiles * 32;
                    float* r0_tm_5 = r0_tm_0 + tiles * 40;
                    float* r0_tm_6 = r0_tm_0 + tiles * 48;
                    float* r0_tm_7 = r0_tm_0 + tiles * 56;

                    for (int m = 0; m < 8; m++)
                    {
                        __m256 _tmp00 = _mm256_load_ps(tmp[m][0]);
                        __m256 _tmp01 = _mm256_load_ps(tmp[m][1]);
                        __m256 _tmp02 = _mm256_load_ps(tmp[m][2]);
                        __m256 _tmp03 = _mm256_load_ps(tmp[m][3]);
                        __m256 _tmp04 = _mm256_load_ps(tmp[m][4]);
                        __m256 _tmp05 = _mm256_load_ps(tmp[m][5]);
                        __m256 _tmp06 = _mm256_load_ps(tmp[m][6]);
                        __m256 _tmp07 = _mm256_load_ps(tmp[m][7]);

                        __m256 _r0tm0 = _mm256_comp_fmadd_ps(_mm256_set1_ps(5.25f), _mm256_sub_ps(_tmp04, _tmp02), _mm256_sub_ps(_tmp00, _tmp06));
                        __m256 _r0tm7 = _mm256_comp_fmadd_ps(_mm256_set1_ps(5.25f), _mm256_sub_ps(_tmp03, _tmp05), _mm256_sub_ps(_tmp07, _tmp01));

                        __m256 _tmp12a = _mm256_comp_fmadd_ps(_mm256_set1_ps(-4.25f), _tmp04, _mm256_add_ps(_tmp02, _tmp06));
                        __m256 _tmp12b = _mm256_comp_fmadd_ps(_mm256_set1_ps(-4.25f), _tmp03, _mm256_add_ps(_tmp01, _tmp05));

                        __m256 _r0tm1 = _mm256_add_ps(_tmp12a, _tmp12b);
                        __m256 _r0tm2 = _mm256_sub_ps(_tmp12a, _tmp12b);

                        __m256 _tmp34a = _mm256_comp_fmadd_ps(_mm256_set1_ps(-1.25f), _tmp04, _mm256_comp_fmadd_ps(_mm256_set1_ps(0.25f), _tmp02, _tmp06));
                        __m256 _tmp34b = _mm256_comp_fmadd_ps(_mm256_set1_ps(2.f), _tmp05, _mm256_comp_fmadd_ps(_mm256_set1_ps(-2.5f), _tmp03, _mm256_mul_ps(_tmp01, _mm256_set1_ps(0.5f))));

                        __m256 _r0tm3 = _mm256_add_ps(_tmp34a, _tmp34b);
                        __m256 _r0tm4 = _mm256_sub_ps(_tmp34a, _tmp34b);

                        __m256 _tmp56a = _mm256_comp_fmadd_ps(_mm256_set1_ps(4.f), _mm256_comp_fmadd_ps(_mm256_set1_ps(-1.25f), _tmp04, _tmp02), _tmp06);
                        __m256 _tmp56b = _mm256_comp_fmadd_ps(_mm256_set1_ps(0.5f), _tmp05, _mm256_comp_fmadd_ps(_mm256_set1_ps(-2.5f), _tmp03, _mm256_mul_ps(_tmp01, _mm256_set1_ps(2.f))));

                        __m256 _r0tm5 = _mm256_add_ps(_tmp56a, _tmp56b);
                        __m256 _r0tm6 = _mm256_sub_ps(_tmp56a, _tmp56b);

                        _mm256_store_ps(r0_tm_0, _r0tm0);
                        _mm256_store_ps(r0_tm_1, _r0tm1);
                        _mm256_store_ps(r0_tm_2, _r0tm2);
                        _mm256_store_ps(r0_tm_3, _r0tm3);
                        _mm256_store_ps(r0_tm_4, _r0tm4);
                        _mm256_store_ps(r0_tm_5, _r0tm5);
                        _mm256_store_ps(r0_tm_6, _r0tm6);
                        _mm256_store_ps(r0_tm_7, _r0tm7);

                        r0_tm_0 += tiles * 64;
                        r0_tm_1 += tiles * 64;
                        r0_tm_2 += tiles * 64;
                        r0_tm_3 += tiles * 64;
                        r0_tm_4 += tiles * 64;
                        r0_tm_5 += tiles * 64;
                        r0_tm_6 += tiles * 64;
                        r0_tm_7 += tiles * 64;
                    }
                }
            }
        }
    }
    bottom_blob_bordered = Mat();
    // END transform input
    // BEGIN dot
    Mat top_blob_tm;
    {
        int w_tm = outw / 6 * 8;
        int h_tm = outh / 6 * 8;

        const int tiles = h_tm / 8 * w_tm / 8;

        Mat bottom_blob_tm2;

        if (tiles >= 12)
            bottom_blob_tm2.create(12 * inch, tiles / 12 + (tiles % 12) / 8 + (tiles % 12 % 8) / 4 + (tiles % 12 % 4) / 2 + tiles % 12 % 2, 64, elemsize, elempack, opt.workspace_allocator);
        else if (tiles >= 8)
            bottom_blob_tm2.create(8 * inch, tiles / 8 + (tiles % 8) / 4 + (tiles % 4) / 2 + tiles % 2, 64, elemsize, elempack, opt.workspace_allocator);
        else if (tiles >= 4)
            bottom_blob_tm2.create(4 * inch, tiles / 4 + (tiles % 4) / 2 + tiles % 2, 64, elemsize, elempack, opt.workspace_allocator);
        else if (tiles >= 2)
            bottom_blob_tm2.create(2 * inch, tiles / 2 + tiles % 2, 64, elemsize, elempack, opt.workspace_allocator);
        else // if (tiles >= 1)
            bottom_blob_tm2.create(1 * inch, tiles, 64, elemsize, elempack, opt.workspace_allocator);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int r = 0; r < 64; r++)
        {
            Mat tm2 = bottom_blob_tm2.channel(r);

            // tile
            int i = 0;

            for (; i + 11 < tiles; i += 12)
            {
                float* tmpptr = tm2.row(i / 12);

                const float* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * 8;

                for (int q = 0; q < inch; q++)
                {
                    // transpose 8x12
                    __m256 _r0 = _mm256_load_ps(r0);
                    __m256 _r1 = _mm256_load_ps(r0 + 8);
                    __m256 _r2 = _mm256_load_ps(r0 + 16);
                    __m256 _r3 = _mm256_load_ps(r0 + 24);
                    __m256 _r4 = _mm256_load_ps(r0 + 32);
                    __m256 _r5 = _mm256_load_ps(r0 + 40);
                    __m256 _r6 = _mm256_load_ps(r0 + 48);
                    __m256 _r7 = _mm256_load_ps(r0 + 56);
                    __m256 _r8 = _mm256_load_ps(r0 + 64);
                    __m256 _r9 = _mm256_load_ps(r0 + 72);
                    __m256 _ra = _mm256_load_ps(r0 + 80);
                    __m256 _rb = _mm256_load_ps(r0 + 88);

                    __m256 _tmp0 = _mm256_unpacklo_ps(_r0, _r1);
                    __m256 _tmp1 = _mm256_unpackhi_ps(_r0, _r1);
                    __m256 _tmp2 = _mm256_unpacklo_ps(_r2, _r3);
                    __m256 _tmp3 = _mm256_unpackhi_ps(_r2, _r3);
                    __m256 _tmp4 = _mm256_unpacklo_ps(_r4, _r5);
                    __m256 _tmp5 = _mm256_unpackhi_ps(_r4, _r5);
                    __m256 _tmp6 = _mm256_unpacklo_ps(_r6, _r7);
                    __m256 _tmp7 = _mm256_unpackhi_ps(_r6, _r7);
                    __m256 _tmp8 = _mm256_unpacklo_ps(_r8, _r9);
                    __m256 _tmp9 = _mm256_unpackhi_ps(_r8, _r9);
                    __m256 _tmpa = _mm256_unpacklo_ps(_ra, _rb);
                    __m256 _tmpb = _mm256_unpackhi_ps(_ra, _rb);
                    __m256 _tmpc = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 _tmpd = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(3, 2, 3, 2));
                    __m256 _tmpe = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 _tmpf = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(3, 2, 3, 2));
                    __m256 _tmpg = _mm256_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 _tmph = _mm256_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(3, 2, 3, 2));
                    __m256 _tmpi = _mm256_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 _tmpj = _mm256_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(3, 2, 3, 2));
                    __m256 _tmpk = _mm256_shuffle_ps(_tmp8, _tmpa, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 _tmpl = _mm256_shuffle_ps(_tmp8, _tmpa, _MM_SHUFFLE(3, 2, 3, 2));
                    __m256 _tmpm = _mm256_shuffle_ps(_tmp9, _tmpb, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 _tmpn = _mm256_shuffle_ps(_tmp9, _tmpb, _MM_SHUFFLE(3, 2, 3, 2));
                    _r0 = _mm256_permute2f128_ps(_tmpc, _tmpg, _MM_SHUFFLE(0, 2, 0, 0));
                    _r1 = _mm256_permute2f128_ps(_tmpk, _tmpd, _MM_SHUFFLE(0, 2, 0, 0));
                    _r2 = _mm256_permute2f128_ps(_tmph, _tmpl, _MM_SHUFFLE(0, 2, 0, 0));
                    _r3 = _mm256_permute2f128_ps(_tmpe, _tmpi, _MM_SHUFFLE(0, 2, 0, 0));
                    _r4 = _mm256_permute2f128_ps(_tmpm, _tmpf, _MM_SHUFFLE(0, 2, 0, 0));
                    _r5 = _mm256_permute2f128_ps(_tmpj, _tmpn, _MM_SHUFFLE(0, 2, 0, 0));
                    _r6 = _mm256_permute2f128_ps(_tmpc, _tmpg, _MM_SHUFFLE(0, 3, 0, 1));
                    _r7 = _mm256_permute2f128_ps(_tmpk, _tmpd, _MM_SHUFFLE(0, 3, 0, 1));
                    _r8 = _mm256_permute2f128_ps(_tmph, _tmpl, _MM_SHUFFLE(0, 3, 0, 1));
                    _r9 = _mm256_permute2f128_ps(_tmpe, _tmpi, _MM_SHUFFLE(0, 3, 0, 1));
                    _ra = _mm256_permute2f128_ps(_tmpm, _tmpf, _MM_SHUFFLE(0, 3, 0, 1));
                    _rb = _mm256_permute2f128_ps(_tmpj, _tmpn, _MM_SHUFFLE(0, 3, 0, 1));

                    _mm256_store_ps(tmpptr, _r0);
                    _mm256_store_ps(tmpptr + 8, _r1);
                    _mm256_store_ps(tmpptr + 8 * 2, _r2);
                    _mm256_store_ps(tmpptr + 8 * 3, _r3);
                    _mm256_store_ps(tmpptr + 8 * 4, _r4);
                    _mm256_store_ps(tmpptr + 8 * 5, _r5);
                    _mm256_store_ps(tmpptr + 8 * 6, _r6);
                    _mm256_store_ps(tmpptr + 8 * 7, _r7);
                    _mm256_store_ps(tmpptr + 8 * 8, _r8);
                    _mm256_store_ps(tmpptr + 8 * 9, _r9);
                    _mm256_store_ps(tmpptr + 8 * 10, _ra);
                    _mm256_store_ps(tmpptr + 8 * 11, _rb);

                    tmpptr += 96;
                    r0 += bottom_blob_tm.cstep * 8;
                }
            }
            for (; i + 7 < tiles; i += 8)
            {
                float* tmpptr = tm2.row(i / 12 + (i % 12) / 8);

                const float* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * 8;

                for (int q = 0; q < inch; q++)
                {
                    // transpose 8x8
                    __m256 _r0 = _mm256_load_ps(r0);
                    __m256 _r1 = _mm256_load_ps(r0 + 8);
                    __m256 _r2 = _mm256_load_ps(r0 + 8 * 2);
                    __m256 _r3 = _mm256_load_ps(r0 + 8 * 3);
                    __m256 _r4 = _mm256_load_ps(r0 + 8 * 4);
                    __m256 _r5 = _mm256_load_ps(r0 + 8 * 5);
                    __m256 _r6 = _mm256_load_ps(r0 + 8 * 6);
                    __m256 _r7 = _mm256_load_ps(r0 + 8 * 7);

                    __m256 _tmp0 = _mm256_unpacklo_ps(_r0, _r1);
                    __m256 _tmp1 = _mm256_unpackhi_ps(_r0, _r1);
                    __m256 _tmp2 = _mm256_unpacklo_ps(_r2, _r3);
                    __m256 _tmp3 = _mm256_unpackhi_ps(_r2, _r3);
                    __m256 _tmp4 = _mm256_unpacklo_ps(_r4, _r5);
                    __m256 _tmp5 = _mm256_unpackhi_ps(_r4, _r5);
                    __m256 _tmp6 = _mm256_unpacklo_ps(_r6, _r7);
                    __m256 _tmp7 = _mm256_unpackhi_ps(_r6, _r7);
                    __m256 _tmp8 = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 _tmp9 = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(3, 2, 3, 2));
                    __m256 _tmpa = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 _tmpb = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(3, 2, 3, 2));
                    __m256 _tmpc = _mm256_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 _tmpd = _mm256_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(3, 2, 3, 2));
                    __m256 _tmpe = _mm256_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 _tmpf = _mm256_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(3, 2, 3, 2));
                    _r0 = _mm256_permute2f128_ps(_tmp8, _tmpc, _MM_SHUFFLE(0, 2, 0, 0));
                    _r1 = _mm256_permute2f128_ps(_tmp9, _tmpd, _MM_SHUFFLE(0, 2, 0, 0));
                    _r2 = _mm256_permute2f128_ps(_tmpa, _tmpe, _MM_SHUFFLE(0, 2, 0, 0));
                    _r3 = _mm256_permute2f128_ps(_tmpb, _tmpf, _MM_SHUFFLE(0, 2, 0, 0));
                    _r4 = _mm256_permute2f128_ps(_tmp8, _tmpc, _MM_SHUFFLE(0, 3, 0, 1));
                    _r5 = _mm256_permute2f128_ps(_tmp9, _tmpd, _MM_SHUFFLE(0, 3, 0, 1));
                    _r6 = _mm256_permute2f128_ps(_tmpa, _tmpe, _MM_SHUFFLE(0, 3, 0, 1));
                    _r7 = _mm256_permute2f128_ps(_tmpb, _tmpf, _MM_SHUFFLE(0, 3, 0, 1));

                    _mm256_store_ps(tmpptr, _r0);
                    _mm256_store_ps(tmpptr + 8, _r1);
                    _mm256_store_ps(tmpptr + 8 * 2, _r2);
                    _mm256_store_ps(tmpptr + 8 * 3, _r3);
                    _mm256_store_ps(tmpptr + 8 * 4, _r4);
                    _mm256_store_ps(tmpptr + 8 * 5, _r5);
                    _mm256_store_ps(tmpptr + 8 * 6, _r6);
                    _mm256_store_ps(tmpptr + 8 * 7, _r7);

                    tmpptr += 64;
                    r0 += bottom_blob_tm.cstep * 8;
                }
            }
            for (; i + 3 < tiles; i += 4)
            {
                float* tmpptr = tm2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4);

                const float* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * 8;

                for (int q = 0; q < inch; q++)
                {
                    // transpose 8x4
                    __m256 _r0 = _mm256_load_ps(r0);
                    __m256 _r1 = _mm256_load_ps(r0 + 8);
                    __m256 _r2 = _mm256_load_ps(r0 + 8 * 2);
                    __m256 _r3 = _mm256_load_ps(r0 + 8 * 3);

                    __m256 _tmp0 = _mm256_unpacklo_ps(_r0, _r1);
                    __m256 _tmp1 = _mm256_unpackhi_ps(_r0, _r1);
                    __m256 _tmp2 = _mm256_unpacklo_ps(_r2, _r3);
                    __m256 _tmp3 = _mm256_unpackhi_ps(_r2, _r3);
                    __m256 _tmp4 = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 _tmp5 = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(3, 2, 3, 2));
                    __m256 _tmp6 = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 _tmp7 = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(3, 2, 3, 2));
                    _r0 = _mm256_permute2f128_ps(_tmp4, _tmp5, _MM_SHUFFLE(0, 2, 0, 0));
                    _r1 = _mm256_permute2f128_ps(_tmp6, _tmp7, _MM_SHUFFLE(0, 2, 0, 0));
                    _r2 = _mm256_permute2f128_ps(_tmp4, _tmp5, _MM_SHUFFLE(0, 3, 0, 1));
                    _r3 = _mm256_permute2f128_ps(_tmp6, _tmp7, _MM_SHUFFLE(0, 3, 0, 1));

                    _mm256_store_ps(tmpptr, _r0);
                    _mm256_store_ps(tmpptr + 8, _r1);
                    _mm256_store_ps(tmpptr + 8 * 2, _r2);
                    _mm256_store_ps(tmpptr + 8 * 3, _r3);

                    tmpptr += 32;
                    r0 += bottom_blob_tm.cstep * 8;
                }
            }
            for (; i + 1 < tiles; i += 2)
            {
                float* tmpptr = tm2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2);

                const float* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * 8;

                for (int q = 0; q < inch; q++)
                {
                    // transpose 8x2
                    __m256 _r0 = _mm256_load_ps(r0);
                    __m256 _r1 = _mm256_load_ps(r0 + 8);

                    __m256 _tmp0 = _mm256_unpacklo_ps(_r0, _r1);
                    __m256 _tmp1 = _mm256_unpackhi_ps(_r0, _r1);
                    _r0 = _mm256_permute2f128_ps(_tmp0, _tmp1, _MM_SHUFFLE(0, 2, 0, 0));
                    _r1 = _mm256_permute2f128_ps(_tmp0, _tmp1, _MM_SHUFFLE(0, 3, 0, 1));

                    _mm256_store_ps(tmpptr, _r0);
                    _mm256_store_ps(tmpptr + 8, _r1);

                    tmpptr += 16;
                    r0 += bottom_blob_tm.cstep * 8;
                }
            }

            for (; i < tiles; i++)
            {
                float* tmpptr = tm2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2);

                const float* r0 = bottom_blob_tm;
                r0 += (r * tiles + i) * 8;

                for (int q = 0; q < inch; q++)
                {
                    __m256 _val = _mm256_load_ps(r0);
                    _mm256_store_ps(tmpptr, _val);

                    tmpptr += 8;
                    r0 += bottom_blob_tm.cstep * 8;
                }
            }
        }

        bottom_blob_tm = Mat();
        // permute end

        top_blob_tm.create(tiles, 64, outch, elemsize, elempack, opt.workspace_allocator);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outch; p++)
        {
            float* output0_tm = top_blob_tm.channel(p);

            const Mat kernel0_tm = kernel_tm.channel(p);

            for (int r = 0; r < 64; r++)
            {
                const Mat bb2 = bottom_blob_tm2.channel(r);

                int i = 0;
                for (; i + 11 < tiles; i += 12)
                {
                    const float* r0 = bb2.row(i / 12);
                    const float* k0 = kernel0_tm.row(r);

                    int nn = inch * 8; // inch always > 0

                    __m256 _sum0 = _mm256_setzero_ps();
                    __m256 _sum1 = _mm256_setzero_ps();
                    __m256 _sum2 = _mm256_setzero_ps();
                    __m256 _sum3 = _mm256_setzero_ps();
                    __m256 _sum4 = _mm256_setzero_ps();
                    __m256 _sum5 = _mm256_setzero_ps();
                    __m256 _sum6 = _mm256_setzero_ps();
                    __m256 _sum7 = _mm256_setzero_ps();
                    __m256 _sum8 = _mm256_setzero_ps();
                    __m256 _sum9 = _mm256_setzero_ps();
                    __m256 _suma = _mm256_setzero_ps();
                    __m256 _sumb = _mm256_setzero_ps();

                    for (int j = 0; j < nn; j++)
                    {
                        __m256 _w0 = _mm256_load_ps(k0);

                        __m256 _val0 = _mm256_broadcast_ss(r0);
                        __m256 _val1 = _mm256_broadcast_ss(r0 + 1);
                        _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val1, _w0, _sum1);
                        __m256 _val2 = _mm256_broadcast_ss(r0 + 2);
                        __m256 _val3 = _mm256_broadcast_ss(r0 + 3);
                        _sum2 = _mm256_comp_fmadd_ps(_val2, _w0, _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val3, _w0, _sum3);
                        __m256 _val4 = _mm256_broadcast_ss(r0 + 4);
                        __m256 _val5 = _mm256_broadcast_ss(r0 + 5);
                        _sum4 = _mm256_comp_fmadd_ps(_val4, _w0, _sum4);
                        _sum5 = _mm256_comp_fmadd_ps(_val5, _w0, _sum5);
                        __m256 _val6 = _mm256_broadcast_ss(r0 + 6);
                        __m256 _val7 = _mm256_broadcast_ss(r0 + 7);
                        _sum6 = _mm256_comp_fmadd_ps(_val6, _w0, _sum6);
                        _sum7 = _mm256_comp_fmadd_ps(_val7, _w0, _sum7);
                        __m256 _val8 = _mm256_broadcast_ss(r0 + 8);
                        __m256 _val9 = _mm256_broadcast_ss(r0 + 9);
                        _sum8 = _mm256_comp_fmadd_ps(_val8, _w0, _sum8);
                        _sum9 = _mm256_comp_fmadd_ps(_val9, _w0, _sum9);
                        __m256 _vala = _mm256_broadcast_ss(r0 + 10);
                        __m256 _valb = _mm256_broadcast_ss(r0 + 11);
                        _suma = _mm256_comp_fmadd_ps(_vala, _w0, _suma);
                        _sumb = _mm256_comp_fmadd_ps(_valb, _w0, _sumb);

                        r0 += 12;
                        k0 += 8;
                    }

                    _mm256_store_ps(output0_tm, _sum0);
                    _mm256_store_ps(output0_tm + 8, _sum1);
                    _mm256_store_ps(output0_tm + 8 * 2, _sum2);
                    _mm256_store_ps(output0_tm + 8 * 3, _sum3);
                    _mm256_store_ps(output0_tm + 8 * 4, _sum4);
                    _mm256_store_ps(output0_tm + 8 * 5, _sum5);
                    _mm256_store_ps(output0_tm + 8 * 6, _sum6);
                    _mm256_store_ps(output0_tm + 8 * 7, _sum7);
                    _mm256_store_ps(output0_tm + 8 * 8, _sum8);
                    _mm256_store_ps(output0_tm + 8 * 9, _sum9);
                    _mm256_store_ps(output0_tm + 8 * 10, _suma);
                    _mm256_store_ps(output0_tm + 8 * 11, _sumb);

                    output0_tm += 8 * 12;
                }
                for (; i + 7 < tiles; i += 8)
                {
                    const float* r0 = bb2.row(i / 12 + (i % 12) / 8);
                    const float* k0 = kernel0_tm.row(r);

                    int nn = inch * 8; // inch always > 0

                    __m256 _sum0 = _mm256_setzero_ps();
                    __m256 _sum1 = _mm256_setzero_ps();
                    __m256 _sum2 = _mm256_setzero_ps();
                    __m256 _sum3 = _mm256_setzero_ps();
                    __m256 _sum4 = _mm256_setzero_ps();
                    __m256 _sum5 = _mm256_setzero_ps();
                    __m256 _sum6 = _mm256_setzero_ps();
                    __m256 _sum7 = _mm256_setzero_ps();

                    for (int j = 0; j < nn; j++)
                    {
                        __m256 _w0 = _mm256_load_ps(k0);

                        __m256 _val0 = _mm256_broadcast_ss(r0);
                        __m256 _val1 = _mm256_broadcast_ss(r0 + 1);
                        _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val1, _w0, _sum1);
                        __m256 _val2 = _mm256_broadcast_ss(r0 + 2);
                        __m256 _val3 = _mm256_broadcast_ss(r0 + 3);
                        _sum2 = _mm256_comp_fmadd_ps(_val2, _w0, _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val3, _w0, _sum3);
                        __m256 _val4 = _mm256_broadcast_ss(r0 + 4);
                        __m256 _val5 = _mm256_broadcast_ss(r0 + 5);
                        _sum4 = _mm256_comp_fmadd_ps(_val4, _w0, _sum4);
                        _sum5 = _mm256_comp_fmadd_ps(_val5, _w0, _sum5);
                        __m256 _val6 = _mm256_broadcast_ss(r0 + 6);
                        __m256 _val7 = _mm256_broadcast_ss(r0 + 7);
                        _sum6 = _mm256_comp_fmadd_ps(_val6, _w0, _sum6);
                        _sum7 = _mm256_comp_fmadd_ps(_val7, _w0, _sum7);

                        r0 += 8;
                        k0 += 8;
                    }

                    _mm256_store_ps(output0_tm, _sum0);
                    _mm256_store_ps(output0_tm + 8, _sum1);
                    _mm256_store_ps(output0_tm + 8 * 2, _sum2);
                    _mm256_store_ps(output0_tm + 8 * 3, _sum3);
                    _mm256_store_ps(output0_tm + 8 * 4, _sum4);
                    _mm256_store_ps(output0_tm + 8 * 5, _sum5);
                    _mm256_store_ps(output0_tm + 8 * 6, _sum6);
                    _mm256_store_ps(output0_tm + 8 * 7, _sum7);

                    output0_tm += 8 * 8;
                }
                for (; i + 3 < tiles; i += 4)
                {
                    const float* r0 = bb2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4);
                    const float* k0 = kernel0_tm.row(r);

                    int nn = inch * 8; // inch always > 0

                    __m256 _sum0 = _mm256_setzero_ps();
                    __m256 _sum1 = _mm256_setzero_ps();
                    __m256 _sum2 = _mm256_setzero_ps();
                    __m256 _sum3 = _mm256_setzero_ps();

                    for (int j = 0; j < nn; j++)
                    {
                        __m256 _w0 = _mm256_load_ps(k0);

                        __m256 _val0 = _mm256_broadcast_ss(r0);
                        __m256 _val1 = _mm256_broadcast_ss(r0 + 1);
                        _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val1, _w0, _sum1);
                        __m256 _val2 = _mm256_broadcast_ss(r0 + 2);
                        __m256 _val3 = _mm256_broadcast_ss(r0 + 3);
                        _sum2 = _mm256_comp_fmadd_ps(_val2, _w0, _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val3, _w0, _sum3);

                        r0 += 4;
                        k0 += 8;
                    }

                    _mm256_store_ps(output0_tm, _sum0);
                    _mm256_store_ps(output0_tm + 8, _sum1);
                    _mm256_store_ps(output0_tm + 8 * 2, _sum2);
                    _mm256_store_ps(output0_tm + 8 * 3, _sum3);

                    output0_tm += 8 * 4;
                }
                for (; i + 1 < tiles; i += 2)
                {
                    const float* r0 = bb2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2);
                    const float* k0 = kernel0_tm.row(r);

                    int nn = inch * 8; // inch always > 0

                    __m256 _sum0 = _mm256_setzero_ps();
                    __m256 _sum1 = _mm256_setzero_ps();

                    for (int j = 0; j < nn; j++)
                    {
                        __m256 _w0 = _mm256_load_ps(k0);

                        __m256 _val0 = _mm256_broadcast_ss(r0);
                        __m256 _val1 = _mm256_broadcast_ss(r0 + 1);
                        _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val1, _w0, _sum1);

                        r0 += 2;
                        k0 += 8;
                    }

                    _mm256_store_ps(output0_tm, _sum0);
                    _mm256_store_ps(output0_tm + 8, _sum1);

                    output0_tm += 8 * 2;
                }

                for (; i < tiles; i++)
                {
                    const float* r0 = bb2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2);
                    const float* k0 = kernel0_tm.row(r);

                    int nn = inch * 8; // inch always > 0

                    __m256 _sum0 = _mm256_setzero_ps();

                    for (int j = 0; j < nn; j++)
                    {
                        __m256 _w0 = _mm256_load_ps(k0);
                        __m256 _val0 = _mm256_broadcast_ss(r0);
                        _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);

                        r0 += 1;
                        k0 += 8;
                    }

                    _mm256_store_ps(output0_tm, _sum0);

                    output0_tm += 8;
                }
            }
        }
    }
    bottom_blob_tm = Mat();
    // END dot

    // BEGIN transform output
    Mat top_blob_bordered;
    if (outw == top_blob.w && outh == top_blob.h)
    {
        top_blob_bordered = top_blob;
    }
    else
    {
        top_blob_bordered.create(outw, outh, outch, elemsize, elempack, opt.workspace_allocator);
    }
    {
        //         const float otm[6][8] = {
        //             {1.0f,  1.0f,   1.0f,   1.0f,   1.0f,  32.0f, 32.0f, 0.0f},
        //             {0.0f,  1.0f,  -1.0f,   2.0f,  -2.0f,  16.0f,-16.0f, 0.0f},
        //             {0.0f,  1.0f,   1.0f,   4.0f,   4.0f,   8.0f,  8.0f, 0.0f},
        //             {0.0f,  1.0f,  -1.0f,   8.0f,  -8.0f,   4.0f, -4.0f, 0.0f},
        //             {0.0f,  1.0f,   1.0f,  16.0f,  16.0f,   2.0f,  2.0f, 0.0f},
        //             {0.0f,  1.0f,  -1.0f,  32.0f, -32.0f,   1.0f, -1.0f, 1.0f}
        //         };

        // 0 = r0 + (r1 + r2) + (r3 + r4)     + (r5 + r6) * 32
        // 1 =      (r1 - r2) + (r3 - r4) * 2 + (r5 - r6) * 16
        // 2 =      (r1 + r2) + (r3 + r4) * 4 + (r5 + r6) * 8
        // 3 =      (r1 - r2) + (r3 - r4) * 8 + (r5 - r6) * 4
        // 4 =      (r1 + r2) + (r3 + r4) * 16+ (r5 + r6) * 2
        // 5 = r7 + (r1 - r2) + (r3 - r4) * 32+ (r5 - r6)

        int w_tm = outw / 6 * 8;
        int h_tm = outh / 6 * 8;
        const int tiles = w_tm / 8 * h_tm / 8;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outch; p++)
        {
            const Mat out0_tm = top_blob_tm.channel(p);
            Mat out0 = top_blob_bordered.channel(p);

            //             const float bias0 = bias ? bias[p] : 0.f;
            __m256 _bias0 = bias ? _mm256_loadu_ps((const float*)bias + p * 8) : _mm256_setzero_ps();

#ifdef _MSC_VER
            __declspec(align(32))
#else
            __attribute__((aligned(32)))
#endif
            float tmp[6][8][8];

            // tile
            for (int i = 0; i < outh / 6; i++)
            {
                for (int j = 0; j < outw / 6; j++)
                {
                    //                     top_blob_tm.create(tiles, 64, outch, elemsize, elempack);

                    const float* output0_tm_0 = (const float*)out0_tm + (i * w_tm / 8 + j) * 8;
                    const float* output0_tm_1 = output0_tm_0 + tiles * 8;
                    const float* output0_tm_2 = output0_tm_0 + tiles * 16;
                    const float* output0_tm_3 = output0_tm_0 + tiles * 24;
                    const float* output0_tm_4 = output0_tm_0 + tiles * 32;
                    const float* output0_tm_5 = output0_tm_0 + tiles * 40;
                    const float* output0_tm_6 = output0_tm_0 + tiles * 48;
                    const float* output0_tm_7 = output0_tm_0 + tiles * 56;

                    float* output0 = out0.row(i * 6) + (j * 6) * 8;

                    // TODO neon optimize
                    for (int m = 0; m < 8; m++)
                    {
                        __m256 _out0tm0 = _mm256_load_ps(output0_tm_0);
                        __m256 _out0tm1 = _mm256_load_ps(output0_tm_1);
                        __m256 _out0tm2 = _mm256_load_ps(output0_tm_2);
                        __m256 _out0tm3 = _mm256_load_ps(output0_tm_3);
                        __m256 _out0tm4 = _mm256_load_ps(output0_tm_4);
                        __m256 _out0tm5 = _mm256_load_ps(output0_tm_5);
                        __m256 _out0tm6 = _mm256_load_ps(output0_tm_6);
                        __m256 _out0tm7 = _mm256_load_ps(output0_tm_7);

                        __m256 _tmp024a = _mm256_add_ps(_out0tm1, _out0tm2);
                        __m256 _tmp135a = _mm256_sub_ps(_out0tm1, _out0tm2);

                        __m256 _tmp024b = _mm256_add_ps(_out0tm3, _out0tm4);
                        __m256 _tmp135b = _mm256_sub_ps(_out0tm3, _out0tm4);

                        __m256 _tmp024c = _mm256_add_ps(_out0tm5, _out0tm6);
                        __m256 _tmp135c = _mm256_sub_ps(_out0tm5, _out0tm6);

                        __m256 _tmp0m = _mm256_add_ps(_mm256_add_ps(_out0tm0, _tmp024a), _mm256_comp_fmadd_ps(_mm256_set1_ps(32.f), _tmp024c, _tmp024b));
                        __m256 _tmp2m = _mm256_comp_fmadd_ps(_mm256_set1_ps(8.f), _tmp024c, _mm256_comp_fmadd_ps(_mm256_set1_ps(4.f), _tmp024b, _tmp024a));
                        __m256 _tmp4m = _mm256_comp_fmadd_ps(_mm256_set1_ps(2.f), _tmp024c, _mm256_comp_fmadd_ps(_mm256_set1_ps(16.f), _tmp024b, _tmp024a));
                        _mm256_store_ps(tmp[0][m], _tmp0m);
                        _mm256_store_ps(tmp[2][m], _tmp2m);
                        _mm256_store_ps(tmp[4][m], _tmp4m);

                        __m256 _tmp1m = _mm256_comp_fmadd_ps(_mm256_set1_ps(16.f), _tmp135c, _mm256_comp_fmadd_ps(_mm256_set1_ps(2.f), _tmp135b, _tmp135a));
                        __m256 _tmp3m = _mm256_comp_fmadd_ps(_mm256_set1_ps(4.f), _tmp135c, _mm256_comp_fmadd_ps(_mm256_set1_ps(8.f), _tmp135b, _tmp135a));
                        __m256 _tmp5m = _mm256_add_ps(_mm256_add_ps(_out0tm7, _tmp135a), _mm256_comp_fmadd_ps(_mm256_set1_ps(32.f), _tmp135b, _tmp135c));
                        _mm256_store_ps(tmp[1][m], _tmp1m);
                        _mm256_store_ps(tmp[3][m], _tmp3m);
                        _mm256_store_ps(tmp[5][m], _tmp5m);

                        output0_tm_0 += tiles * 64;
                        output0_tm_1 += tiles * 64;
                        output0_tm_2 += tiles * 64;
                        output0_tm_3 += tiles * 64;
                        output0_tm_4 += tiles * 64;
                        output0_tm_5 += tiles * 64;
                        output0_tm_6 += tiles * 64;
                        output0_tm_7 += tiles * 64;
                    }

                    for (int m = 0; m < 6; m++)
                    {
                        __m256 _tmp00 = _mm256_load_ps(tmp[m][0]);
                        __m256 _tmp01 = _mm256_load_ps(tmp[m][1]);
                        __m256 _tmp02 = _mm256_load_ps(tmp[m][2]);
                        __m256 _tmp03 = _mm256_load_ps(tmp[m][3]);
                        __m256 _tmp04 = _mm256_load_ps(tmp[m][4]);
                        __m256 _tmp05 = _mm256_load_ps(tmp[m][5]);
                        __m256 _tmp06 = _mm256_load_ps(tmp[m][6]);
                        __m256 _tmp07 = _mm256_load_ps(tmp[m][7]);

                        __m256 _tmp024a = _mm256_add_ps(_tmp01, _tmp02);
                        __m256 _tmp135a = _mm256_sub_ps(_tmp01, _tmp02);

                        __m256 _tmp024b = _mm256_add_ps(_tmp03, _tmp04);
                        __m256 _tmp135b = _mm256_sub_ps(_tmp03, _tmp04);

                        __m256 _tmp024c = _mm256_add_ps(_tmp05, _tmp06);
                        __m256 _tmp135c = _mm256_sub_ps(_tmp05, _tmp06);

                        __m256 _out00 = _mm256_add_ps(_bias0, _mm256_add_ps(_mm256_add_ps(_tmp00, _tmp024a), _mm256_comp_fmadd_ps(_mm256_set1_ps(32.f), _tmp024c, _tmp024b)));
                        __m256 _out02 = _mm256_add_ps(_bias0, _mm256_comp_fmadd_ps(_mm256_set1_ps(8.f), _tmp024c, _mm256_comp_fmadd_ps(_mm256_set1_ps(4.f), _tmp024b, _tmp024a)));
                        __m256 _out04 = _mm256_add_ps(_bias0, _mm256_comp_fmadd_ps(_mm256_set1_ps(2.f), _tmp024c, _mm256_comp_fmadd_ps(_mm256_set1_ps(16.f), _tmp024b, _tmp024a)));
                        _mm256_store_ps(output0, _out00);
                        _mm256_store_ps(output0 + 16, _out02);
                        _mm256_store_ps(output0 + 32, _out04);

                        __m256 _out01 = _mm256_add_ps(_bias0, _mm256_comp_fmadd_ps(_mm256_set1_ps(16.f), _tmp135c, _mm256_comp_fmadd_ps(_mm256_set1_ps(2.f), _tmp135b, _tmp135a)));
                        __m256 _out03 = _mm256_add_ps(_bias0, _mm256_comp_fmadd_ps(_mm256_set1_ps(4.f), _tmp135c, _mm256_comp_fmadd_ps(_mm256_set1_ps(8.f), _tmp135b, _tmp135a)));
                        __m256 _out05 = _mm256_add_ps(_bias0, _mm256_add_ps(_mm256_add_ps(_tmp07, _tmp135a), _mm256_comp_fmadd_ps(_mm256_set1_ps(32.f), _tmp135b, _tmp135c)));
                        _mm256_store_ps(output0 + 8, _out01);
                        _mm256_store_ps(output0 + 24, _out03);
                        _mm256_store_ps(output0 + 40, _out05);

                        output0 += outw * 8;
                    }
                }
            }
        }
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt);
}

static void conv3x3s1_winograd42_transform_kernel_pack8_avx(const Mat& kernel, Mat& kernel_tm_pack4, int inch, int outch, const Option& opt)
{
    // winograd42 transform kernel
    Mat kernel_tm(6 * 6, inch, outch);

    const float ktm[6][3] = {
        {1.0f / 4, 0.0f, 0.0f},
        {-1.0f / 6, -1.0f / 6, -1.0f / 6},
        {-1.0f / 6, 1.0f / 6, -1.0f / 6},
        {1.0f / 24, 1.0f / 12, 1.0f / 6},
        {1.0f / 24, -1.0f / 12, 1.0f / 6},
        {0.0f, 0.0f, 1.0f}
    };

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        for (int q = 0; q < inch; q++)
        {
            const float* kernel0 = (const float*)kernel + p * inch * 9 + q * 9;
            float* kernel_tm0 = kernel_tm.channel(p).row(q);

            // transform kernel
            const float* k0 = kernel0;
            const float* k1 = kernel0 + 3;
            const float* k2 = kernel0 + 6;

            // h
            float tmp[6][3];
            for (int i = 0; i < 6; i++)
            {
                tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
            }

            // U
            for (int j = 0; j < 6; j++)
            {
                float* tmpp = &tmp[j][0];

                for (int i = 0; i < 6; i++)
                {
                    kernel_tm0[j * 6 + i] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                }
            }
        }
    }

    // interleave
    // src = 36-inch-outch
    // dst = 8b-8a-inch/8a-36-outch/8b
    kernel_tm_pack4.create(inch / 8, 36, outch / 8, (size_t)4u * 64, 64);

    for (int q = 0; q + (8 - 1) < outch; q += 8)
    {
        Mat g0 = kernel_tm_pack4.channel(q / 8);

        for (int k = 0; k < 36; k++)
        {
            float* g00 = g0.row<float>(k);

            for (int p = 0; p + (8 - 1) < inch; p += 8)
            {
                for (int i = 0; i < 8; i++)
                {
                    for (int j = 0; j < 8; j++)
                    {
                        const float* k00 = kernel_tm.channel(q + j).row(p + i);
                        g00[0] = k00[k];
                        g00++;
                    }
                }
            }
        }
    }
}

static void conv3x3s1_winograd42_pack8_avx(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    // pad to 4n+2
    Mat bottom_blob_bordered = bottom_blob;

    outw = (outw + 3) / 4 * 4;
    outh = (outh + 3) / 4 * 4;

    w = outw + 2;
    h = outh + 2;
    copy_make_border(bottom_blob, bottom_blob_bordered, 0, h - bottom_blob.h, 0, w - bottom_blob.w, BORDER_CONSTANT, 0.f, opt);

    const float* bias = _bias;

    // BEGIN transform input
    Mat bottom_blob_tm;
    {
        int w_tm = outw / 4 * 6;
        int h_tm = outh / 4 * 6;

        const int tiles = w_tm / 6 * h_tm / 6;

        bottom_blob_tm.create(tiles, 36, inch, 4u * elempack, elempack, opt.workspace_allocator);

        // const float itm[4][4] = {
        //     {4.0f, 0.0f, -5.0f, 0.0f, 1.0f, 0.0f},
        //     {0.0f,-4.0f, -4.0f, 1.0f, 1.0f, 0.0f},
        //     {0.0f, 4.0f, -4.0f,-1.0f, 1.0f, 0.0f},
        //     {0.0f,-2.0f, -1.0f, 2.0f, 1.0f, 0.0f},
        //     {0.0f, 2.0f, -1.0f,-2.0f, 1.0f, 0.0f},
        //     {0.0f, 4.0f,  0.0f,-5.0f, 0.0f, 1.0f}
        // };

        // 0 =  4 * r00 - 5 * r02 + r04
        // 1 = -4 * (r01 + r02) + r04 + r03
        // 2 =  4 * (r01 - r02) + r04 - r03
        // 3 = -2 * (r01 - r03) + r04 - r02
        // 4 =  2 * (r01 - r03) + r04 - r02
        // 5 =  4 * r01 - 5 * r03 + r05

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < inch; q++)
        {
            const Mat img0 = bottom_blob_bordered.channel(q);
            Mat img0_tm = bottom_blob_tm.channel(q);

#ifdef _MSC_VER
            __declspec(align(32))
#else
            __attribute__((aligned(32)))
#endif
            float tmp[6][6][8];

            // tile
            for (int i = 0; i < h_tm / 6; i++)
            {
                for (int j = 0; j < w_tm / 6; j++)
                {
                    const float* r0 = img0.row(i * 4) + (j * 4) * 8;

                    for (int m = 0; m < 6; m++)
                    {
                        __m256 _r00 = _mm256_load_ps(r0);
                        __m256 _r01 = _mm256_load_ps(r0 + 8);
                        __m256 _r02 = _mm256_load_ps(r0 + 8 * 2);
                        __m256 _r03 = _mm256_load_ps(r0 + 8 * 3);
                        __m256 _r04 = _mm256_load_ps(r0 + 8 * 4);
                        __m256 _r05 = _mm256_load_ps(r0 + 8 * 5);

                        __m256 _tmp0m = _mm256_comp_fmadd_ps(_mm256_set1_ps(-5.f), _r02, _mm256_comp_fmadd_ps(_mm256_set1_ps(4.f), _r00, _r04));
                        __m256 _tmp1m = _mm256_comp_fmadd_ps(_mm256_set1_ps(-4.f), _mm256_add_ps(_r01, _r02), _mm256_add_ps(_r04, _r03));
                        __m256 _tmp2m = _mm256_comp_fmadd_ps(_mm256_set1_ps(4.f), _mm256_sub_ps(_r01, _r02), _mm256_sub_ps(_r04, _r03));
                        __m256 _tmp3m = _mm256_comp_fmadd_ps(_mm256_set1_ps(-2.f), _mm256_sub_ps(_r01, _r03), _mm256_sub_ps(_r04, _r02));
                        __m256 _tmp4m = _mm256_comp_fmadd_ps(_mm256_set1_ps(2.f), _mm256_sub_ps(_r01, _r03), _mm256_sub_ps(_r04, _r02));
                        __m256 _tmp5m = _mm256_comp_fmadd_ps(_mm256_set1_ps(-5.f), _r03, _mm256_comp_fmadd_ps(_mm256_set1_ps(4.f), _r01, _r05));

                        _mm256_store_ps(tmp[0][m], _tmp0m);
                        _mm256_store_ps(tmp[1][m], _tmp1m);
                        _mm256_store_ps(tmp[2][m], _tmp2m);
                        _mm256_store_ps(tmp[3][m], _tmp3m);
                        _mm256_store_ps(tmp[4][m], _tmp4m);
                        _mm256_store_ps(tmp[5][m], _tmp5m);

                        r0 += w * 8;
                    }

                    float* r0_tm_0 = (float*)img0_tm + (i * w_tm / 6 + j) * 8;
                    float* r0_tm_1 = r0_tm_0 + tiles * 8;
                    float* r0_tm_2 = r0_tm_0 + tiles * 8 * 2;
                    float* r0_tm_3 = r0_tm_0 + tiles * 8 * 3;
                    float* r0_tm_4 = r0_tm_0 + tiles * 8 * 4;
                    float* r0_tm_5 = r0_tm_0 + tiles * 8 * 5;

                    for (int m = 0; m < 6; m++)
                    {
                        __m256 _tmp00 = _mm256_load_ps(tmp[m][0]);
                        __m256 _tmp01 = _mm256_load_ps(tmp[m][1]);
                        __m256 _tmp02 = _mm256_load_ps(tmp[m][2]);
                        __m256 _tmp03 = _mm256_load_ps(tmp[m][3]);
                        __m256 _tmp04 = _mm256_load_ps(tmp[m][4]);
                        __m256 _tmp05 = _mm256_load_ps(tmp[m][5]);

                        __m256 _r0tm0 = _mm256_comp_fmadd_ps(_mm256_set1_ps(-5.f), _tmp02, _mm256_comp_fmadd_ps(_mm256_set1_ps(4.f), _tmp00, _tmp04));
                        __m256 _r0tm1 = _mm256_comp_fmadd_ps(_mm256_set1_ps(-4.f), _mm256_add_ps(_tmp01, _tmp02), _mm256_add_ps(_tmp04, _tmp03));
                        __m256 _r0tm2 = _mm256_comp_fmadd_ps(_mm256_set1_ps(4.f), _mm256_sub_ps(_tmp01, _tmp02), _mm256_sub_ps(_tmp04, _tmp03));
                        __m256 _r0tm3 = _mm256_comp_fmadd_ps(_mm256_set1_ps(-2.f), _mm256_sub_ps(_tmp01, _tmp03), _mm256_sub_ps(_tmp04, _tmp02));
                        __m256 _r0tm4 = _mm256_comp_fmadd_ps(_mm256_set1_ps(2.f), _mm256_sub_ps(_tmp01, _tmp03), _mm256_sub_ps(_tmp04, _tmp02));
                        __m256 _r0tm5 = _mm256_comp_fmadd_ps(_mm256_set1_ps(-5.f), _tmp03, _mm256_comp_fmadd_ps(_mm256_set1_ps(4.f), _tmp01, _tmp05));

                        _mm256_store_ps(r0_tm_0, _r0tm0);
                        _mm256_store_ps(r0_tm_1, _r0tm1);
                        _mm256_store_ps(r0_tm_2, _r0tm2);
                        _mm256_store_ps(r0_tm_3, _r0tm3);
                        _mm256_store_ps(r0_tm_4, _r0tm4);
                        _mm256_store_ps(r0_tm_5, _r0tm5);

                        r0_tm_0 += tiles * 8 * 6;
                        r0_tm_1 += tiles * 8 * 6;
                        r0_tm_2 += tiles * 8 * 6;
                        r0_tm_3 += tiles * 8 * 6;
                        r0_tm_4 += tiles * 8 * 6;
                        r0_tm_5 += tiles * 8 * 6;
                    }
                }
            }
        }
    }
    bottom_blob_bordered = Mat();
    // END transform input

    // BEGIN dot
    Mat top_blob_tm;
    {
        int w_tm = outw / 4 * 6;
        int h_tm = outh / 4 * 6;

        const int tiles = h_tm / 6 * w_tm / 6;

        // permute
        //         bottom_blob_tm.create(tiles, 36, inch, elemsize, elempack, opt.workspace_allocator);
        Mat bottom_blob_tm2;
        if (tiles >= 12)
            bottom_blob_tm2.create(12 * inch, tiles / 12 + (tiles % 12) / 8 + (tiles % 12 % 8) / 4 + (tiles % 12 % 4) / 2 + tiles % 12 % 2, 36, 4u * elempack, elempack, opt.workspace_allocator);
        else if (tiles >= 8)
            bottom_blob_tm2.create(8 * inch, tiles / 8 + (tiles % 8) / 4 + (tiles % 4) / 2 + tiles % 2, 36, 4u * elempack, elempack, opt.workspace_allocator);
        else if (tiles >= 4)
            bottom_blob_tm2.create(4 * inch, tiles / 4 + (tiles % 4) / 2 + tiles % 2, 36, 4u * elempack, elempack, opt.workspace_allocator);
        else if (tiles >= 2)
            bottom_blob_tm2.create(2 * inch, tiles / 2 + tiles % 2, 36, 4u * elempack, elempack, opt.workspace_allocator);
        else // if (tiles >= 1)
            bottom_blob_tm2.create(1 * inch, tiles, 36, 4u * elempack, elempack, opt.workspace_allocator);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int r = 0; r < 36; r++)
        {
            Mat tm2 = bottom_blob_tm2.channel(r);

            // tile
            int i = 0;
            for (; i + 11 < tiles; i += 12)
            {
                float* tmpptr = tm2.row(i / 12);

                const float* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * 8;

                for (int q = 0; q < inch; q++)
                {
                    // transpose 8x12
                    __m256 _r0 = _mm256_load_ps(r0);
                    __m256 _r1 = _mm256_load_ps(r0 + 8);
                    __m256 _r2 = _mm256_load_ps(r0 + 8 * 2);
                    __m256 _r3 = _mm256_load_ps(r0 + 8 * 3);
                    __m256 _r4 = _mm256_load_ps(r0 + 8 * 4);
                    __m256 _r5 = _mm256_load_ps(r0 + 8 * 5);
                    __m256 _r6 = _mm256_load_ps(r0 + 8 * 6);
                    __m256 _r7 = _mm256_load_ps(r0 + 8 * 7);
                    __m256 _r8 = _mm256_load_ps(r0 + 8 * 8);
                    __m256 _r9 = _mm256_load_ps(r0 + 8 * 9);
                    __m256 _ra = _mm256_load_ps(r0 + 8 * 10);
                    __m256 _rb = _mm256_load_ps(r0 + 8 * 11);

                    __m256 _tmp0 = _mm256_unpacklo_ps(_r0, _r1);
                    __m256 _tmp1 = _mm256_unpackhi_ps(_r0, _r1);
                    __m256 _tmp2 = _mm256_unpacklo_ps(_r2, _r3);
                    __m256 _tmp3 = _mm256_unpackhi_ps(_r2, _r3);
                    __m256 _tmp4 = _mm256_unpacklo_ps(_r4, _r5);
                    __m256 _tmp5 = _mm256_unpackhi_ps(_r4, _r5);
                    __m256 _tmp6 = _mm256_unpacklo_ps(_r6, _r7);
                    __m256 _tmp7 = _mm256_unpackhi_ps(_r6, _r7);
                    __m256 _tmp8 = _mm256_unpacklo_ps(_r8, _r9);
                    __m256 _tmp9 = _mm256_unpackhi_ps(_r8, _r9);
                    __m256 _tmpa = _mm256_unpacklo_ps(_ra, _rb);
                    __m256 _tmpb = _mm256_unpackhi_ps(_ra, _rb);
                    __m256 _tmpc = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 _tmpd = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(3, 2, 3, 2));
                    __m256 _tmpe = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 _tmpf = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(3, 2, 3, 2));
                    __m256 _tmpg = _mm256_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 _tmph = _mm256_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(3, 2, 3, 2));
                    __m256 _tmpi = _mm256_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 _tmpj = _mm256_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(3, 2, 3, 2));
                    __m256 _tmpk = _mm256_shuffle_ps(_tmp8, _tmpa, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 _tmpl = _mm256_shuffle_ps(_tmp8, _tmpa, _MM_SHUFFLE(3, 2, 3, 2));
                    __m256 _tmpm = _mm256_shuffle_ps(_tmp9, _tmpb, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 _tmpn = _mm256_shuffle_ps(_tmp9, _tmpb, _MM_SHUFFLE(3, 2, 3, 2));
                    _r0 = _mm256_permute2f128_ps(_tmpc, _tmpg, _MM_SHUFFLE(0, 2, 0, 0));
                    _r1 = _mm256_permute2f128_ps(_tmpk, _tmpd, _MM_SHUFFLE(0, 2, 0, 0));
                    _r2 = _mm256_permute2f128_ps(_tmph, _tmpl, _MM_SHUFFLE(0, 2, 0, 0));
                    _r3 = _mm256_permute2f128_ps(_tmpe, _tmpi, _MM_SHUFFLE(0, 2, 0, 0));
                    _r4 = _mm256_permute2f128_ps(_tmpm, _tmpf, _MM_SHUFFLE(0, 2, 0, 0));
                    _r5 = _mm256_permute2f128_ps(_tmpj, _tmpn, _MM_SHUFFLE(0, 2, 0, 0));
                    _r6 = _mm256_permute2f128_ps(_tmpc, _tmpg, _MM_SHUFFLE(0, 3, 0, 1));
                    _r7 = _mm256_permute2f128_ps(_tmpk, _tmpd, _MM_SHUFFLE(0, 3, 0, 1));
                    _r8 = _mm256_permute2f128_ps(_tmph, _tmpl, _MM_SHUFFLE(0, 3, 0, 1));
                    _r9 = _mm256_permute2f128_ps(_tmpe, _tmpi, _MM_SHUFFLE(0, 3, 0, 1));
                    _ra = _mm256_permute2f128_ps(_tmpm, _tmpf, _MM_SHUFFLE(0, 3, 0, 1));
                    _rb = _mm256_permute2f128_ps(_tmpj, _tmpn, _MM_SHUFFLE(0, 3, 0, 1));

                    _mm256_store_ps(tmpptr, _r0);
                    _mm256_store_ps(tmpptr + 8, _r1);
                    _mm256_store_ps(tmpptr + 8 * 2, _r2);
                    _mm256_store_ps(tmpptr + 8 * 3, _r3);
                    _mm256_store_ps(tmpptr + 8 * 4, _r4);
                    _mm256_store_ps(tmpptr + 8 * 5, _r5);
                    _mm256_store_ps(tmpptr + 8 * 6, _r6);
                    _mm256_store_ps(tmpptr + 8 * 7, _r7);
                    _mm256_store_ps(tmpptr + 8 * 8, _r8);
                    _mm256_store_ps(tmpptr + 8 * 9, _r9);
                    _mm256_store_ps(tmpptr + 8 * 10, _ra);
                    _mm256_store_ps(tmpptr + 8 * 11, _rb);

                    r0 += bottom_blob_tm.cstep * 8;
                    tmpptr += 96;
                }
            }
            for (; i + 7 < tiles; i += 8)
            {
                float* tmpptr = tm2.row(i / 12 + (i % 12) / 8);

                const float* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * 8;

                for (int q = 0; q < inch; q++)
                {
                    // transpose 8x8
                    __m256 _r0 = _mm256_load_ps(r0);
                    __m256 _r1 = _mm256_load_ps(r0 + 8);
                    __m256 _r2 = _mm256_load_ps(r0 + 8 * 2);
                    __m256 _r3 = _mm256_load_ps(r0 + 8 * 3);
                    __m256 _r4 = _mm256_load_ps(r0 + 8 * 4);
                    __m256 _r5 = _mm256_load_ps(r0 + 8 * 5);
                    __m256 _r6 = _mm256_load_ps(r0 + 8 * 6);
                    __m256 _r7 = _mm256_load_ps(r0 + 8 * 7);

                    __m256 _tmp0 = _mm256_unpacklo_ps(_r0, _r1);
                    __m256 _tmp1 = _mm256_unpackhi_ps(_r0, _r1);
                    __m256 _tmp2 = _mm256_unpacklo_ps(_r2, _r3);
                    __m256 _tmp3 = _mm256_unpackhi_ps(_r2, _r3);
                    __m256 _tmp4 = _mm256_unpacklo_ps(_r4, _r5);
                    __m256 _tmp5 = _mm256_unpackhi_ps(_r4, _r5);
                    __m256 _tmp6 = _mm256_unpacklo_ps(_r6, _r7);
                    __m256 _tmp7 = _mm256_unpackhi_ps(_r6, _r7);
                    __m256 _tmp8 = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 _tmp9 = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(3, 2, 3, 2));
                    __m256 _tmpa = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 _tmpb = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(3, 2, 3, 2));
                    __m256 _tmpc = _mm256_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 _tmpd = _mm256_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(3, 2, 3, 2));
                    __m256 _tmpe = _mm256_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 _tmpf = _mm256_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(3, 2, 3, 2));
                    _r0 = _mm256_permute2f128_ps(_tmp8, _tmpc, _MM_SHUFFLE(0, 2, 0, 0));
                    _r1 = _mm256_permute2f128_ps(_tmp9, _tmpd, _MM_SHUFFLE(0, 2, 0, 0));
                    _r2 = _mm256_permute2f128_ps(_tmpa, _tmpe, _MM_SHUFFLE(0, 2, 0, 0));
                    _r3 = _mm256_permute2f128_ps(_tmpb, _tmpf, _MM_SHUFFLE(0, 2, 0, 0));
                    _r4 = _mm256_permute2f128_ps(_tmp8, _tmpc, _MM_SHUFFLE(0, 3, 0, 1));
                    _r5 = _mm256_permute2f128_ps(_tmp9, _tmpd, _MM_SHUFFLE(0, 3, 0, 1));
                    _r6 = _mm256_permute2f128_ps(_tmpa, _tmpe, _MM_SHUFFLE(0, 3, 0, 1));
                    _r7 = _mm256_permute2f128_ps(_tmpb, _tmpf, _MM_SHUFFLE(0, 3, 0, 1));

                    _mm256_store_ps(tmpptr, _r0);
                    _mm256_store_ps(tmpptr + 8, _r1);
                    _mm256_store_ps(tmpptr + 8 * 2, _r2);
                    _mm256_store_ps(tmpptr + 8 * 3, _r3);
                    _mm256_store_ps(tmpptr + 8 * 4, _r4);
                    _mm256_store_ps(tmpptr + 8 * 5, _r5);
                    _mm256_store_ps(tmpptr + 8 * 6, _r6);
                    _mm256_store_ps(tmpptr + 8 * 7, _r7);

                    r0 += bottom_blob_tm.cstep * 8;
                    tmpptr += 64;
                }
            }
            for (; i + 3 < tiles; i += 4)
            {
                float* tmpptr = tm2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4);

                const float* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * 8;

                for (int q = 0; q < inch; q++)
                {
                    // transpose 8x4
                    __m256 _r0 = _mm256_load_ps(r0);
                    __m256 _r1 = _mm256_load_ps(r0 + 8);
                    __m256 _r2 = _mm256_load_ps(r0 + 8 * 2);
                    __m256 _r3 = _mm256_load_ps(r0 + 8 * 3);

                    __m256 _tmp0 = _mm256_unpacklo_ps(_r0, _r1);
                    __m256 _tmp1 = _mm256_unpackhi_ps(_r0, _r1);
                    __m256 _tmp2 = _mm256_unpacklo_ps(_r2, _r3);
                    __m256 _tmp3 = _mm256_unpackhi_ps(_r2, _r3);
                    __m256 _tmp4 = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 _tmp5 = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(3, 2, 3, 2));
                    __m256 _tmp6 = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 _tmp7 = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(3, 2, 3, 2));
                    _r0 = _mm256_permute2f128_ps(_tmp4, _tmp5, _MM_SHUFFLE(0, 2, 0, 0));
                    _r1 = _mm256_permute2f128_ps(_tmp6, _tmp7, _MM_SHUFFLE(0, 2, 0, 0));
                    _r2 = _mm256_permute2f128_ps(_tmp4, _tmp5, _MM_SHUFFLE(0, 3, 0, 1));
                    _r3 = _mm256_permute2f128_ps(_tmp6, _tmp7, _MM_SHUFFLE(0, 3, 0, 1));

                    _mm256_store_ps(tmpptr, _r0);
                    _mm256_store_ps(tmpptr + 8, _r1);
                    _mm256_store_ps(tmpptr + 8 * 2, _r2);
                    _mm256_store_ps(tmpptr + 8 * 3, _r3);

                    r0 += bottom_blob_tm.cstep * 8;
                    tmpptr += 32;
                }
            }
            for (; i + 1 < tiles; i += 2)
            {
                float* tmpptr = tm2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2);

                const float* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * 8;

                for (int q = 0; q < inch; q++)
                {
                    // transpose 8x2
                    __m256 _r0 = _mm256_load_ps(r0);
                    __m256 _r1 = _mm256_load_ps(r0 + 8);

                    __m256 _tmp0 = _mm256_unpacklo_ps(_r0, _r1);
                    __m256 _tmp1 = _mm256_unpackhi_ps(_r0, _r1);
                    _r0 = _mm256_permute2f128_ps(_tmp0, _tmp1, _MM_SHUFFLE(0, 2, 0, 0));
                    _r1 = _mm256_permute2f128_ps(_tmp0, _tmp1, _MM_SHUFFLE(0, 3, 0, 1));

                    _mm256_store_ps(tmpptr, _r0);
                    _mm256_store_ps(tmpptr + 8, _r1);

                    r0 += bottom_blob_tm.cstep * 8;
                    tmpptr += 16;
                }
            }
            for (; i < tiles; i++)
            {
                float* tmpptr = tm2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2);

                const float* r0 = bottom_blob_tm;

                r0 += (r * tiles + i) * 8;

                for (int q = 0; q < inch; q++)
                {
                    __m256 _val = _mm256_load_ps(r0);
                    _mm256_store_ps(tmpptr, _val);

                    r0 += bottom_blob_tm.cstep * 8;
                    tmpptr += 8;
                }
            }
        }

        bottom_blob_tm = Mat();
        // permute end

        top_blob_tm.create(tiles, 36, outch, 4u * elempack, elempack, opt.workspace_allocator);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outch; p++)
        {
            float* output0_tm = top_blob_tm.channel(p);

            const Mat kernel0_tm = kernel_tm.channel(p);

            for (int r = 0; r < 36; r++)
            {
                const Mat bb2 = bottom_blob_tm2.channel(r);

                int i = 0;
                for (; i + 11 < tiles; i += 12)
                {
                    const float* r0 = bb2.row(i / 12);
                    const float* k0 = kernel0_tm.row(r);

                    int nn = inch * 8; // inch always > 0

                    __m256 _sum0 = _mm256_setzero_ps();
                    __m256 _sum1 = _mm256_setzero_ps();
                    __m256 _sum2 = _mm256_setzero_ps();
                    __m256 _sum3 = _mm256_setzero_ps();
                    __m256 _sum4 = _mm256_setzero_ps();
                    __m256 _sum5 = _mm256_setzero_ps();
                    __m256 _sum6 = _mm256_setzero_ps();
                    __m256 _sum7 = _mm256_setzero_ps();
                    __m256 _sum8 = _mm256_setzero_ps();
                    __m256 _sum9 = _mm256_setzero_ps();
                    __m256 _suma = _mm256_setzero_ps();
                    __m256 _sumb = _mm256_setzero_ps();

                    for (int j = 0; j < nn; j++)
                    {
                        __m256 _w0 = _mm256_load_ps(k0);

                        __m256 _val0 = _mm256_broadcast_ss(r0);
                        __m256 _val1 = _mm256_broadcast_ss(r0 + 1);
                        _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val1, _w0, _sum1);
                        __m256 _val2 = _mm256_broadcast_ss(r0 + 2);
                        __m256 _val3 = _mm256_broadcast_ss(r0 + 3);
                        _sum2 = _mm256_comp_fmadd_ps(_val2, _w0, _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val3, _w0, _sum3);
                        __m256 _val4 = _mm256_broadcast_ss(r0 + 4);
                        __m256 _val5 = _mm256_broadcast_ss(r0 + 5);
                        _sum4 = _mm256_comp_fmadd_ps(_val4, _w0, _sum4);
                        _sum5 = _mm256_comp_fmadd_ps(_val5, _w0, _sum5);
                        __m256 _val6 = _mm256_broadcast_ss(r0 + 6);
                        __m256 _val7 = _mm256_broadcast_ss(r0 + 7);
                        _sum6 = _mm256_comp_fmadd_ps(_val6, _w0, _sum6);
                        _sum7 = _mm256_comp_fmadd_ps(_val7, _w0, _sum7);
                        __m256 _val8 = _mm256_broadcast_ss(r0 + 8);
                        __m256 _val9 = _mm256_broadcast_ss(r0 + 9);
                        _sum8 = _mm256_comp_fmadd_ps(_val8, _w0, _sum8);
                        _sum9 = _mm256_comp_fmadd_ps(_val9, _w0, _sum9);
                        __m256 _vala = _mm256_broadcast_ss(r0 + 10);
                        __m256 _valb = _mm256_broadcast_ss(r0 + 11);
                        _suma = _mm256_comp_fmadd_ps(_vala, _w0, _suma);
                        _sumb = _mm256_comp_fmadd_ps(_valb, _w0, _sumb);

                        r0 += 12;
                        k0 += 8;
                    }

                    _mm256_store_ps(output0_tm, _sum0);
                    _mm256_store_ps(output0_tm + 8, _sum1);
                    _mm256_store_ps(output0_tm + 8 * 2, _sum2);
                    _mm256_store_ps(output0_tm + 8 * 3, _sum3);
                    _mm256_store_ps(output0_tm + 8 * 4, _sum4);
                    _mm256_store_ps(output0_tm + 8 * 5, _sum5);
                    _mm256_store_ps(output0_tm + 8 * 6, _sum6);
                    _mm256_store_ps(output0_tm + 8 * 7, _sum7);
                    _mm256_store_ps(output0_tm + 8 * 8, _sum8);
                    _mm256_store_ps(output0_tm + 8 * 9, _sum9);
                    _mm256_store_ps(output0_tm + 8 * 10, _suma);
                    _mm256_store_ps(output0_tm + 8 * 11, _sumb);

                    output0_tm += 8 * 12;
                }
                for (; i + 7 < tiles; i += 8)
                {
                    const float* r0 = bb2.row(i / 12 + (i % 12) / 8);
                    const float* k0 = kernel0_tm.row(r);

                    int nn = inch * 8; // inch always > 0

                    __m256 _sum0 = _mm256_setzero_ps();
                    __m256 _sum1 = _mm256_setzero_ps();
                    __m256 _sum2 = _mm256_setzero_ps();
                    __m256 _sum3 = _mm256_setzero_ps();
                    __m256 _sum4 = _mm256_setzero_ps();
                    __m256 _sum5 = _mm256_setzero_ps();
                    __m256 _sum6 = _mm256_setzero_ps();
                    __m256 _sum7 = _mm256_setzero_ps();

                    for (int j = 0; j < nn; j++)
                    {
                        __m256 _w0 = _mm256_load_ps(k0);

                        __m256 _val0 = _mm256_broadcast_ss(r0);
                        __m256 _val1 = _mm256_broadcast_ss(r0 + 1);
                        _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val1, _w0, _sum1);
                        __m256 _val2 = _mm256_broadcast_ss(r0 + 2);
                        __m256 _val3 = _mm256_broadcast_ss(r0 + 3);
                        _sum2 = _mm256_comp_fmadd_ps(_val2, _w0, _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val3, _w0, _sum3);
                        __m256 _val4 = _mm256_broadcast_ss(r0 + 4);
                        __m256 _val5 = _mm256_broadcast_ss(r0 + 5);
                        _sum4 = _mm256_comp_fmadd_ps(_val4, _w0, _sum4);
                        _sum5 = _mm256_comp_fmadd_ps(_val5, _w0, _sum5);
                        __m256 _val6 = _mm256_broadcast_ss(r0 + 6);
                        __m256 _val7 = _mm256_broadcast_ss(r0 + 7);
                        _sum6 = _mm256_comp_fmadd_ps(_val6, _w0, _sum6);
                        _sum7 = _mm256_comp_fmadd_ps(_val7, _w0, _sum7);

                        r0 += 8;
                        k0 += 8;
                    }

                    _mm256_store_ps(output0_tm, _sum0);
                    _mm256_store_ps(output0_tm + 8, _sum1);
                    _mm256_store_ps(output0_tm + 8 * 2, _sum2);
                    _mm256_store_ps(output0_tm + 8 * 3, _sum3);
                    _mm256_store_ps(output0_tm + 8 * 4, _sum4);
                    _mm256_store_ps(output0_tm + 8 * 5, _sum5);
                    _mm256_store_ps(output0_tm + 8 * 6, _sum6);
                    _mm256_store_ps(output0_tm + 8 * 7, _sum7);

                    output0_tm += 8 * 8;
                }
                for (; i + 3 < tiles; i += 4)
                {
                    const float* r0 = bb2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4);
                    const float* k0 = kernel0_tm.row(r);

                    int nn = inch * 8; // inch always > 0

                    __m256 _sum0 = _mm256_setzero_ps();
                    __m256 _sum1 = _mm256_setzero_ps();
                    __m256 _sum2 = _mm256_setzero_ps();
                    __m256 _sum3 = _mm256_setzero_ps();

                    for (int j = 0; j < nn; j++)
                    {
                        __m256 _w0 = _mm256_load_ps(k0);

                        __m256 _val0 = _mm256_broadcast_ss(r0);
                        __m256 _val1 = _mm256_broadcast_ss(r0 + 1);
                        _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val1, _w0, _sum1);
                        __m256 _val2 = _mm256_broadcast_ss(r0 + 2);
                        __m256 _val3 = _mm256_broadcast_ss(r0 + 3);
                        _sum2 = _mm256_comp_fmadd_ps(_val2, _w0, _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val3, _w0, _sum3);

                        r0 += 4;
                        k0 += 8;
                    }

                    _mm256_store_ps(output0_tm, _sum0);
                    _mm256_store_ps(output0_tm + 8, _sum1);
                    _mm256_store_ps(output0_tm + 8 * 2, _sum2);
                    _mm256_store_ps(output0_tm + 8 * 3, _sum3);

                    output0_tm += 8 * 4;
                }
                for (; i + 1 < tiles; i += 2)
                {
                    const float* r0 = bb2.row(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2);
                    const float* k0 = kernel0_tm.row(r);

                    int nn = inch * 8; // inch always > 0

                    __m256 _sum0 = _mm256_setzero_ps();
                    __m256 _sum1 = _mm256_setzero_ps();

                    for (int j = 0; j < nn; j++)
                    {
                        __m256 _w0 = _mm256_load_ps(k0);

                        __m256 _val0 = _mm256_broadcast_ss(r0);
                        __m256 _val1 = _mm256_broadcast_ss(r0 + 1);
                        _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val1, _w0, _sum1);

                        r0 += 2;
                        k0 += 8;
                    }

                    _mm256_store_ps(output0_tm, _sum0);
                    _mm256_store_ps(output0_tm + 8, _sum1);

                    output0_tm += 8 * 2;
                }
                for (; i < tiles; i++)
                {
                    const float* r0 = bb2.row<const float>(i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2);
                    const float* k0 = kernel0_tm.row<const float>(r);

                    int nn = inch * 8; // inch always > 0

                    __m256 _sum0 = _mm256_setzero_ps();

                    for (int j = 0; j < nn; j++)
                    {
                        __m256 _w0 = _mm256_load_ps(k0);
                        __m256 _val0 = _mm256_broadcast_ss(r0);
                        _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);

                        r0 += 1;
                        k0 += 8;
                    }

                    _mm256_store_ps(output0_tm, _sum0);

                    output0_tm += 8;
                }
            }
        }
    }
    bottom_blob_tm = Mat();
    // END dot

    // BEGIN transform output
    Mat top_blob_bordered;
    if (outw == top_blob.w && outh == top_blob.h)
    {
        top_blob_bordered = top_blob;
    }
    else
    {
        top_blob_bordered.create(outw, outh, outch, elemsize, elempack, opt.workspace_allocator);
    }
    {
        // const float otm[4][6] = {
        //     {1.0f, 1.0f,  1.0f, 1.0f,  1.0f, 0.0f},
        //     {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.0f},
        //     {0.0f, 1.0f,  1.0f, 4.0f,  4.0f, 0.0f},
        //     {0.0f, 1.0f, -1.0f, 8.0f, -8.0f, 1.0f}
        // };

        // 0 = r00 + (r01 + r02) + (r03 + r04)
        // 1 =       (r01 - r02) + (r03 - r04) * 2
        // 2 =       (r01 + r02) + (r03 + r04) * 4
        // 3 = r05 + (r01 - r02) + (r03 - r04) * 8

        int w_tm = outw / 4 * 6;
        int h_tm = outh / 4 * 6;
        const int tiles = w_tm / 6 * h_tm / 6;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outch; p++)
        {
            const Mat out0_tm = top_blob_tm.channel(p);
            Mat out0 = top_blob_bordered.channel(p);

            // const float bias0 = bias ? bias[p] : 0.f;
            __m256 _bias0 = bias ? _mm256_loadu_ps((const float*)bias + p * 8) : _mm256_setzero_ps();

#ifdef _MSC_VER
            __declspec(align(32))
#else
            __attribute__((aligned(32)))
#endif
            float tmp[4][6][8];

            // tile
            for (int i = 0; i < outh / 4; i++)
            {
                for (int j = 0; j < outw / 4; j++)
                {
                    // top_blob_tm.create(tiles, 36, outch, elemsize, elempack);

                    const float* output0_tm_0 = (const float*)out0_tm + (i * w_tm / 6 + j) * 8;
                    const float* output0_tm_1 = output0_tm_0 + tiles * 8;
                    const float* output0_tm_2 = output0_tm_0 + tiles * 8 * 2;
                    const float* output0_tm_3 = output0_tm_0 + tiles * 8 * 3;
                    const float* output0_tm_4 = output0_tm_0 + tiles * 8 * 4;
                    const float* output0_tm_5 = output0_tm_0 + tiles * 8 * 5;

                    float* output0 = out0.row<float>(i * 4) + (j * 4) * 8;

                    // TODO msa optimize
                    for (int m = 0; m < 6; m++)
                    {
                        __m256 _out0tm0 = _mm256_load_ps(output0_tm_0);
                        __m256 _out0tm1 = _mm256_load_ps(output0_tm_1);
                        __m256 _out0tm2 = _mm256_load_ps(output0_tm_2);
                        __m256 _out0tm3 = _mm256_load_ps(output0_tm_3);
                        __m256 _out0tm4 = _mm256_load_ps(output0_tm_4);
                        __m256 _out0tm5 = _mm256_load_ps(output0_tm_5);

                        __m256 _tmp02a = _mm256_add_ps(_out0tm1, _out0tm2);
                        __m256 _tmp13a = _mm256_sub_ps(_out0tm1, _out0tm2);

                        __m256 _tmp02b = _mm256_add_ps(_out0tm3, _out0tm4);
                        __m256 _tmp13b = _mm256_sub_ps(_out0tm3, _out0tm4);

                        __m256 _tmp0m = _mm256_add_ps(_mm256_add_ps(_out0tm0, _tmp02a), _tmp02b);
                        __m256 _tmp1m = _mm256_comp_fmadd_ps(_mm256_set1_ps(2.f), _tmp13b, _tmp13a);
                        __m256 _tmp2m = _mm256_comp_fmadd_ps(_mm256_set1_ps(4.f), _tmp02b, _tmp02a);
                        __m256 _tmp3m = _mm256_comp_fmadd_ps(_mm256_set1_ps(8.f), _tmp13b, _mm256_add_ps(_out0tm5, _tmp13a));

                        _mm256_store_ps(tmp[0][m], _tmp0m);
                        _mm256_store_ps(tmp[1][m], _tmp1m);
                        _mm256_store_ps(tmp[2][m], _tmp2m);
                        _mm256_store_ps(tmp[3][m], _tmp3m);

                        output0_tm_0 += tiles * 8 * 6;
                        output0_tm_1 += tiles * 8 * 6;
                        output0_tm_2 += tiles * 8 * 6;
                        output0_tm_3 += tiles * 8 * 6;
                        output0_tm_4 += tiles * 8 * 6;
                        output0_tm_5 += tiles * 8 * 6;
                    }

                    for (int m = 0; m < 4; m++)
                    {
                        __m256 _tmp00 = _mm256_load_ps(tmp[m][0]);
                        __m256 _tmp01 = _mm256_load_ps(tmp[m][1]);
                        __m256 _tmp02 = _mm256_load_ps(tmp[m][2]);
                        __m256 _tmp03 = _mm256_load_ps(tmp[m][3]);
                        __m256 _tmp04 = _mm256_load_ps(tmp[m][4]);
                        __m256 _tmp05 = _mm256_load_ps(tmp[m][5]);

                        __m256 _tmp02a = _mm256_add_ps(_tmp01, _tmp02);
                        __m256 _tmp13a = _mm256_sub_ps(_tmp01, _tmp02);

                        __m256 _tmp02b = _mm256_add_ps(_tmp03, _tmp04);
                        __m256 _tmp13b = _mm256_sub_ps(_tmp03, _tmp04);

                        __m256 _out00 = _mm256_add_ps(_bias0, _mm256_add_ps(_mm256_add_ps(_tmp00, _tmp02a), _tmp02b));
                        __m256 _out01 = _mm256_add_ps(_bias0, _mm256_comp_fmadd_ps(_mm256_set1_ps(2.f), _tmp13b, _tmp13a));
                        __m256 _out02 = _mm256_add_ps(_bias0, _mm256_comp_fmadd_ps(_mm256_set1_ps(4.f), _tmp02b, _tmp02a));
                        __m256 _out03 = _mm256_add_ps(_bias0, _mm256_comp_fmadd_ps(_mm256_set1_ps(8.f), _tmp13b, _mm256_add_ps(_tmp05, _tmp13a)));

                        _mm256_store_ps(output0, _out00);
                        _mm256_store_ps(output0 + 8, _out01);
                        _mm256_store_ps(output0 + 8 * 2, _out02);
                        _mm256_store_ps(output0 + 8 * 3, _out03);

                        output0 += outw * 8;
                    }
                }
            }
        }
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt);
}
