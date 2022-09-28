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

static void conv3x3s1_winograd63_transform_kernel_pack8_avx(const Mat& kernel, Mat& kernel_tm_pack8, int inch, int outch, const Option& opt)
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
    for (int q = 0; q + 7 < outch; q += 8)
    {
        Mat g0 = kernel_tm_pack8.channel(q / 8);

        for (int k = 0; k < 64; k++)
        {
            float* g00 = g0.row<float>(k);

            for (int p = 0; p + 7 < inch; p += 8)
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

static void conv3x3s1_winograd63_pack8_avx(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& bias, const Option& opt)
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

    // BEGIN transform input
    Mat bottom_blob_tm;
    {
        int w_tiles = outw / 6;
        int h_tiles = outh / 6;
        const int tiles = w_tiles * h_tiles;

        bottom_blob_tm.create(tiles, 64, inch, elemsize, elempack, opt.workspace_allocator);
        conv3x3s1_winograd63_transform_input_pack8_avx(bottom_blob_bordered, bottom_blob_tm, opt);
    }
    bottom_blob_bordered = Mat();
    // END transform input

    // BEGIN dot
    Mat top_blob_tm;
    convolution_winograd_dot_pack8_avx(bottom_blob_tm, outch, kernel_tm, top_blob_tm, opt);
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
        conv3x3s1_winograd63_transform_output_pack8_avx(top_blob_tm, top_blob_bordered, bias, opt);
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt);
}

static void conv3x3s1_winograd43_transform_kernel_pack8_avx(const Mat& kernel, Mat& kernel_tm_pack8, int inch, int outch, const Option& opt)
{
    // winograd43 transform kernel
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
    kernel_tm_pack8.create(inch / 8, 36, outch / 8, (size_t)4u * 64, 64);
    for (int q = 0; q + 7 < outch; q += 8)
    {
        Mat g0 = kernel_tm_pack8.channel(q / 8);

        for (int k = 0; k < 36; k++)
        {
            float* g00 = g0.row<float>(k);

            for (int p = 0; p + 7 < inch; p += 8)
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

static void conv3x3s1_winograd43_pack8_avx(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& bias, const Option& opt)
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

    // BEGIN transform input
    Mat bottom_blob_tm;
    {
        int w_tiles = outw / 4;
        int h_tiles = outh / 4;
        const int tiles = w_tiles * h_tiles;

        bottom_blob_tm.create(tiles, 36, inch, elemsize, elempack, opt.workspace_allocator);
        conv3x3s1_winograd43_transform_input_pack8_avx(bottom_blob_bordered, bottom_blob_tm, opt);
    }
    bottom_blob_bordered = Mat();
    // END transform input

    // BEGIN dot
    Mat top_blob_tm;
    convolution_winograd_dot_pack8_avx(bottom_blob_tm, outch, kernel_tm, top_blob_tm, opt);
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
        conv3x3s1_winograd43_transform_output_pack8_avx(top_blob_tm, top_blob_bordered, bias, opt);
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt);
}

static void conv3x3s1_winograd23_transform_kernel_pack8_avx(const Mat& kernel, Mat& kernel_tm_pack8, int inch, int outch, const Option& opt)
{
    // winograd23 transform kernel
    Mat kernel_tm(4 * 4, inch, outch);

    const float ktm[4][3] = {
        {1.0f, 0.0f, 0.0f},
        {1.0f / 2, 1.0f / 2, 1.0f / 2},
        {1.0f / 2, -1.0f / 2, 1.0f / 2},
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
            float tmp[4][3];
            for (int i = 0; i < 4; i++)
            {
                tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
            }

            // U
            for (int j = 0; j < 4; j++)
            {
                float* tmpp = &tmp[j][0];

                for (int i = 0; i < 4; i++)
                {
                    kernel_tm0[j * 4 + i] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                }
            }
        }
    }

    // interleave
    // src = 16-inch-outch
    // dst = pb-pa-inch/pa-16-outch/pb
    kernel_tm_pack8.create(inch / 8, 16, outch / 8, (size_t)4u * 8 * 8, 8 * 8);
    for (int q = 0; q + 7 < outch; q += 8)
    {
        Mat g0 = kernel_tm_pack8.channel(q / 8);

        for (int k = 0; k < 16; k++)
        {
            float* g00 = g0.row<float>(k);

            for (int p = 0; p + 7 < inch; p += 8)
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

static void conv3x3s1_winograd23_pack8_avx(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& bias, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    // pad to 2n+2
    Mat bottom_blob_bordered = bottom_blob;

    outw = (outw + 1) / 2 * 2;
    outh = (outh + 1) / 2 * 2;

    w = outw + 2;
    h = outh + 2;
    copy_make_border(bottom_blob, bottom_blob_bordered, 0, h - bottom_blob.h, 0, w - bottom_blob.w, BORDER_CONSTANT, 0.f, opt);

    // BEGIN transform input
    Mat bottom_blob_tm;
    {
        int w_tiles = outw / 2;
        int h_tiles = outh / 2;
        int tiles = w_tiles * h_tiles;

        bottom_blob_tm.create(tiles, 16, inch, elemsize, elempack, opt.workspace_allocator);
        conv3x3s1_winograd23_transform_input_pack8_avx(bottom_blob_bordered, bottom_blob_tm, opt);
    }
    bottom_blob_bordered = Mat();
    // END transform input

    // BEGIN dot
    Mat top_blob_tm;
    convolution_winograd_dot_pack8_avx(bottom_blob_tm, outch, kernel_tm, top_blob_tm, opt);
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
        conv3x3s1_winograd23_transform_output_pack8_avx(top_blob_tm, top_blob_bordered, bias, opt);
    }
    // END transform output

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt);
}
