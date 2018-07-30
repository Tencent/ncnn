// SenseNets is pleased to support the open source community by supporting ncnn available.
//
// Copyright (C) 2018 SenseNets Technology Ltd. All rights reserved.
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

static void conv1x1s1_int8_sse(const Mat &bottom_blob, Mat &top_blob, const Mat &_kernel, const Option& opt)
{
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const float *kernel = _kernel;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        out0.fill(0);

        int q = 0;

        for (; q+7<inch; q+=8)
        {
            int* outptr0 = out0;

            const signed char *kernel0 = (const signed char *)kernel + p * inch + q;

            const signed char *r0 = bottom_blob.channel(q);
            const signed char *r1 = bottom_blob.channel(q + 1);
            const signed char *r2 = bottom_blob.channel(q + 2);
            const signed char *r3 = bottom_blob.channel(q + 3);
            const signed char *r4 = bottom_blob.channel(q + 4);
            const signed char *r5 = bottom_blob.channel(q + 5);
            const signed char *r6 = bottom_blob.channel(q + 6);
            const signed char *r7 = bottom_blob.channel(q + 7);

            int size = outw * outh;
            int remain = size;

            for (; remain > 0; remain--)
            {
                //ToDo Neon
                int sum0 = (int)*r0 * (int)kernel0[0] + (int)*r1 * (int)kernel0[1] +
                           (int)*r2 * (int)kernel0[2] + (int)*r3 * (int)kernel0[3] +
                           (int)*r4 * (int)kernel0[4] + (int)*r5 * (int)kernel0[5] +
                           (int)*r6 * (int)kernel0[6] + (int)*r7 * (int)kernel0[7];

                *outptr0 += sum0;

                r0++;
                r1++;
                r2++;
                r3++;
                r4++;
                r5++;
                r6++;
                r7++;
                outptr0++;
            }
        }

        for (; q<inch; q++)
        {
            int* outptr0 = out0;

            const signed char *r0 = bottom_blob.channel(q);

            const signed char *kernel0 = (const signed char *)kernel + p * inch + q;
            const signed char k0 = kernel0[0];

            int size = outw * outh;
            int remain = size;

            for (; remain > 0; remain--)
            {
                int sum0 = (int)(*r0) * (int)k0;

                *outptr0 += sum0;

                r0++;
                outptr0++;
            }
        }
    }
}

static void conv1x1s2_int8_sse(const Mat &bottom_blob, Mat &top_blob, const Mat &_kernel, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2*outw + w;
    const signed char *kernel = _kernel;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        out0.fill(0);

        int q = 0;

        for (; q+7<inch; q+=8)
        {
            int* outptr0 = out0;

            const signed char *kernel0 = (const signed char *)kernel + p * inch + q;

            const signed char *r0 = bottom_blob.channel(q);
            const signed char *r1 = bottom_blob.channel(q + 1);
            const signed char *r2 = bottom_blob.channel(q + 2);
            const signed char *r3 = bottom_blob.channel(q + 3);
            const signed char *r4 = bottom_blob.channel(q + 4);
            const signed char *r5 = bottom_blob.channel(q + 5);
            const signed char *r6 = bottom_blob.channel(q + 6);
            const signed char *r7 = bottom_blob.channel(q + 7);

            for(int i = 0; i < outh; i++)
            {
                int remain = outw;

                for (; remain > 0; remain--)
                {
                    //ToDo Neon
                    int sum0 = (int)*r0 * (int)kernel0[0] + (int)*r1 * (int)kernel0[1] +
                            (int)*r2 * (int)kernel0[2] + (int)*r3 * (int)kernel0[3] +
                            (int)*r4 * (int)kernel0[4] + (int)*r5 * (int)kernel0[5] +
                            (int)*r6 * (int)kernel0[6] + (int)*r7 * (int)kernel0[7];

                    *outptr0 += sum0;

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    r3 += 2;
                    r4 += 2;
                    r5 += 2;
                    r6 += 2;
                    r7 += 2;
                    outptr0++;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
                r3 += tailstep;
                r4 += tailstep;
                r5 += tailstep;
                r6 += tailstep;
                r7 += tailstep;
            }
        }

        for (; q<inch; q++)
        {
            int* outptr0 = out0;

            const signed char *r0 = bottom_blob.channel(q);

            const signed char *kernel0 = (const signed char *)kernel + p * inch + q;

            for(int i = 0; i < outh; i++)
            {
                int remain = outw;

                for (; remain > 0; remain--)
                {
                    //ToDo Neon
                    int sum0 = (int)*r0 * (int)kernel0[0];

                    *outptr0 += sum0;

                    r0 += 2;
                    outptr0++;
                }

                r0 += tailstep;
            }
        }
    }
}
