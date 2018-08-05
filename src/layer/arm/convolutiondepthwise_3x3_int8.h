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

static void convdw3x3s1_int8_neon(const Mat &bottom_blob, Mat &top_blob, const Mat &_kernel, const Option& opt)
{
    int w = bottom_blob.w;
    //int h = bottom_blob.h;
    //int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const signed char *kernel = _kernel;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out = top_blob.channel(p);

        out.fill(0);

        const signed char *kernel0 = (const signed char *)kernel + p * 9;

        int *outptr = out;

        const signed char *img0 = bottom_blob.channel(p);

        const signed char *r0 = img0;
        const signed char *r1 = img0 + w;
        const signed char *r2 = img0 + w * 2;

        int i = 0;
        for (; i < outh; i++)
        {
            int remain = outw;

            for (; remain > 0; remain--)
            {

                int sum = 0;

                sum += (int)r0[0] * (int)kernel0[0];
                sum += (int)r0[1] * (int)kernel0[1];
                sum += (int)r0[2] * (int)kernel0[2];
                sum += (int)r1[0] * (int)kernel0[3];
                sum += (int)r1[1] * (int)kernel0[4];
                sum += (int)r1[2] * (int)kernel0[5];
                sum += (int)r2[0] * (int)kernel0[6];
                sum += (int)r2[1] * (int)kernel0[7];
                sum += (int)r2[2] * (int)kernel0[8];

                *outptr += sum;

                r0++;
                r1++;
                r2++;
                outptr++;
            }

            r0 += 2;
            r1 += 2;
            r2 += 2;
        }
    }
}

static void convdw3x3s2_int8_neon(const Mat &bottom_blob, Mat &top_blob, const Mat &_kernel, const Option& opt)
{
    int w = bottom_blob.w;
    //int h = bottom_blob.h;
    //int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2 * outw + w;

    const signed char *kernel = _kernel;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out = top_blob.channel(p);
        out.fill(0);

        const signed char *kernel0 = (const signed char *)kernel + p * 9;

        int *outptr = out;

        const signed char *img0 = bottom_blob.channel(p);

        const signed char *r0 = img0;
        const signed char *r1 = img0 + w;
        const signed char *r2 = img0 + w * 2;

        int i = 0;

        for (; i < outh; i++)
        {
            int remain = outw;

            for (; remain > 0; remain--)
            {
                int sum = 0;

                sum += (int)r0[0] * (int)kernel0[0];
                sum += (int)r0[1] * (int)kernel0[1];
                sum += (int)r0[2] * (int)kernel0[2];
                sum += (int)r1[0] * (int)kernel0[3];
                sum += (int)r1[1] * (int)kernel0[4];
                sum += (int)r1[2] * (int)kernel0[5];
                sum += (int)r2[0] * (int)kernel0[6];
                sum += (int)r2[1] * (int)kernel0[7];
                sum += (int)r2[2] * (int)kernel0[8];

                *outptr += sum;

                r0 += 2;
                r1 += 2;
                r2 += 2;
                outptr++;
            }

            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
        }
    }
}
