// BUG1989 is pleased to support the open source community by supporting ncnn available.
//
// Copyright (C) 2019 BUG1989. All rights reserved.
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

static inline signed char float2int8(float v)
{
    int int32 = static_cast<int>(round(v));
    if (int32 > 127) return 127;
    if (int32 < -127) return -127;
    return (signed char)int32;
}

static void conv_im2col_sgemm_int8_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel,
                                       const int kernel_w, const int kernel_h, const int stride_w, const int stride_h, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const signed char* kernel = _kernel;

    // im2row
    Mat bottom_im2row(kernel_h * kernel_w * inch, outw * outh, 1UL, opt.workspace_allocator);
    {
        signed char* ret = (signed char*)bottom_im2row;
        int retID = 0;

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                for (int p = 0; p < inch; p++)
                {
                    const signed char* input = bottom_blob.channel(p);
                    for (int u = 0; u < kernel_h; u++)
                    {
                        for (int v = 0; v < kernel_w; v++)
                        {
                            int row = u + i * stride_h;
                            int col = v + j * stride_w;
                            int index = row * w + col;
                            ret[retID] = input[index];
                            retID++;
                        }
                    }
                }
            }
        }
    }

    int kernel_size = kernel_w * kernel_h;
    int out_size = outw * outh;

    // int M = outch;  // outch
    int N = outw * outh;                // outsize or out stride
    int K = kernel_w * kernel_h * inch; // ksize * inch

    // bottom_im2row memory packed 4 x 4
    Mat bottom_tm(4 * kernel_size, inch, out_size / 4 + out_size % 4, (size_t)1u, opt.workspace_allocator);
    {
        int nn_size = out_size >> 2;
        int remain_size_start = nn_size << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = ii * 4;

            const signed char* img0 = bottom_im2row.row<signed char>(i);
            const signed char* img1 = bottom_im2row.row<signed char>(i + 1);
            const signed char* img2 = bottom_im2row.row<signed char>(i + 2);
            const signed char* img3 = bottom_im2row.row<signed char>(i + 3);

            signed char* tmpptr = bottom_tm.channel(i / 4);

            int q = 0;
            for (; q + 1 < inch * kernel_size; q = q + 2)
            {
                tmpptr[0] = img0[0];
                tmpptr[1] = img0[1];
                tmpptr[2] = img1[0];
                tmpptr[3] = img1[1];
                tmpptr[4] = img2[0];
                tmpptr[5] = img2[1];
                tmpptr[6] = img3[0];
                tmpptr[7] = img3[1];

                tmpptr += 8;
                img0 += 2;
                img1 += 2;
                img2 += 2;
                img3 += 2;
            }

            for (; q < inch * kernel_size; q++)
            {
                tmpptr[0] = img0[0];
                tmpptr[1] = img1[0];
                tmpptr[2] = img2[0];
                tmpptr[3] = img3[0];

                tmpptr += 4;
                img0 += 1;
                img1 += 1;
                img2 += 1;
                img3 += 1;
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < out_size; i++)
        {
            const signed char* img0 = bottom_im2row.row<signed char>(i);

            signed char* tmpptr = bottom_tm.channel(i / 4 + i % 4);

            int q = 0;
            for (; q + 1 < inch * kernel_size; q = q + 2)
            {
                tmpptr[0] = img0[0];
                tmpptr[1] = img0[1];

                tmpptr += 2;
                img0 += 2;
            }

            for (; q < inch * kernel_size; q++)
            {
                tmpptr[0] = img0[0];

                tmpptr += 1;
                img0 += 1;
            }
        }
    }

    // kernel memory packed 4 x 4
    Mat kernel_tm(4 * kernel_size, inch, outch / 4 + outch % 4, (size_t)1u, opt.workspace_allocator);
    {
        int nn_outch = 0;
        int remain_outch_start = 0;

        nn_outch = outch >> 2;
        remain_outch_start = nn_outch << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp = 0; pp < nn_outch; pp++)
        {
            int p = pp * 4;

            const signed char* k0 = kernel + (p + 0) * inch * kernel_size;
            const signed char* k1 = kernel + (p + 1) * inch * kernel_size;
            const signed char* k2 = kernel + (p + 2) * inch * kernel_size;
            const signed char* k3 = kernel + (p + 3) * inch * kernel_size;

            signed char* ktmp = kernel_tm.channel(p / 4);

            int q = 0;
            for (; q + 1 < inch * kernel_size; q += 2)
            {
                ktmp[0] = k0[0];
                ktmp[1] = k0[1];
                ktmp[2] = k1[0];
                ktmp[3] = k1[1];
                ktmp[4] = k2[0];
                ktmp[5] = k2[1];
                ktmp[6] = k3[0];
                ktmp[7] = k3[1];

                ktmp += 8;

                k0 += 2;
                k1 += 2;
                k2 += 2;
                k3 += 2;
            }

            for (; q < inch * kernel_size; q++)
            {
                ktmp[0] = k0[0];
                ktmp[1] = k1[0];
                ktmp[2] = k2[0];
                ktmp[3] = k3[0];
                ktmp += 4;

                k0 += 1;
                k1 += 1;
                k2 += 1;
                k3 += 1;
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = remain_outch_start; p < outch; p++)
        {
            const signed char* k0 = kernel + (p + 0) * inch * kernel_size;

            signed char* ktmp = kernel_tm.channel(p / 4 + p % 4);

            int q = 0;
            for (; q + 1 < inch * kernel_size; q = q + 2)
            {
                ktmp[0] = k0[0];
                ktmp[1] = k0[1];
                ktmp += 2;
                k0 += 2;
            }

            for (; q < inch * kernel_size; q++)
            {
                ktmp[0] = k0[0];
                ktmp++;
                k0++;
            }
        }
    }

    // 4x4
    // sgemm(int M, int N, int K, float* A, float* B, float* C)
    {
        // int M = outch;  // outch
        // int N = outw * outh; // outsize or out stride
        // int L = kernel_w * kernel_h * inch; // ksize * inch

        int nn_outch = 0;
        int remain_outch_start = 0;

        nn_outch = outch >> 2;
        remain_outch_start = nn_outch << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp = 0; pp < nn_outch; pp++)
        {
            int i = pp * 4;

            int* output0 = top_blob.channel(i);
            int* output1 = top_blob.channel(i + 1);
            int* output2 = top_blob.channel(i + 2);
            int* output3 = top_blob.channel(i + 3);

            int j = 0;
            for (; j + 3 < N; j = j + 4)
            {
                signed char* vb = bottom_tm.channel(j / 4);
                signed char* va = kernel_tm.channel(i / 4);

                int sum0[4] = {0};
                int sum1[4] = {0};
                int sum2[4] = {0};
                int sum3[4] = {0};

                int k = 0;

                for (; k + 1 < K; k = k + 2)
                {
                    for (int n = 0; n < 4; n++)
                    {
                        sum0[n] += (int)va[0] * vb[2 * n]; // k0
                        sum0[n] += (int)va[1] * vb[2 * n + 1];

                        sum1[n] += (int)va[2] * vb[2 * n]; // k1
                        sum1[n] += (int)va[3] * vb[2 * n + 1];

                        sum2[n] += (int)va[4] * vb[2 * n]; // k2
                        sum2[n] += (int)va[5] * vb[2 * n + 1];

                        sum3[n] += (int)va[6] * vb[2 * n]; // k3
                        sum3[n] += (int)va[7] * vb[2 * n + 1];
                    }

                    va += 8;
                    vb += 8;
                }

                for (; k < K; k++)
                {
                    for (int n = 0; n < 4; n++)
                    {
                        sum0[n] += (int)va[0] * vb[n];
                        sum1[n] += (int)va[1] * vb[n];
                        sum2[n] += (int)va[2] * vb[n];
                        sum3[n] += (int)va[3] * vb[n];
                    }

                    va += 4;
                    vb += 4;
                }

                for (int n = 0; n < 4; n++)
                {
                    output0[n] = sum0[n];
                    output1[n] = sum1[n];
                    output2[n] = sum2[n];
                    output3[n] = sum3[n];
                }
                output0 += 4;
                output1 += 4;
                output2 += 4;
                output3 += 4;
            }

            for (; j < N; j++)
            {
                int sum0 = 0;
                int sum1 = 0;
                int sum2 = 0;
                int sum3 = 0;

                signed char* vb = bottom_tm.channel(j / 4 + j % 4);
                signed char* va = kernel_tm.channel(i / 4);

                int k = 0;

                for (; k + 1 < K; k = k + 2)
                {
                    sum0 += (int)va[0] * vb[0];
                    sum0 += (int)va[1] * vb[1];

                    sum1 += (int)va[2] * vb[0];
                    sum1 += (int)va[3] * vb[1];

                    sum2 += (int)va[4] * vb[0];
                    sum2 += (int)va[5] * vb[1];

                    sum3 += (int)va[6] * vb[0];
                    sum3 += (int)va[7] * vb[1];

                    va += 8;
                    vb += 2;
                }

                for (; k < K; k++)
                {
                    sum0 += (int)va[0] * vb[0];
                    sum1 += (int)va[1] * vb[0];
                    sum2 += (int)va[2] * vb[0];
                    sum3 += (int)va[3] * vb[0];

                    va += 4;
                    vb += 1;
                }

                output0[0] = sum0;
                output1[0] = sum1;
                output2[0] = sum2;
                output3[0] = sum3;

                output0++;
                output1++;
                output2++;
                output3++;
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_outch_start; i < outch; i++)
        {
            int* output = top_blob.channel(i);

            int j = 0;
            for (; j + 3 < N; j = j + 4)
            {
                signed char* vb = bottom_tm.channel(j / 4);
                signed char* va = kernel_tm.channel(i / 4 + i % 4);
                int sum[4] = {0};

                int k = 0;
                for (; k + 1 < K; k = k + 2)
                {
                    for (int n = 0; n < 4; n++)
                    {
                        sum[n] += (int)va[0] * vb[2 * n];
                        sum[n] += (int)va[1] * vb[2 * n + 1];
                    }
                    va += 2;
                    vb += 8;
                }

                for (; k < K; k++)
                {
                    for (int n = 0; n < 4; n++)
                    {
                        sum[n] += (int)va[0] * vb[n];
                    }
                    va += 1;
                    vb += 4;
                }

                for (int n = 0; n < 4; n++)
                {
                    output[n] = sum[n];
                }
                output += 4;
            }

            for (; j < N; j++)
            {
                int sum = 0;

                signed char* vb = bottom_tm.channel(j / 4 + j % 4);
                signed char* va = kernel_tm.channel(i / 4 + i % 4);

                for (int k = 0; k < K; k++)
                {
                    sum += (int)va[0] * vb[0];

                    va += 1;
                    vb += 1;
                }
                output[0] = sum;

                output++;
            }
        }
    }

    // // sgemm(int M, int N, int K, float* A, float* B, float* C)
    // {
    //     for (int i=0; i<M; i++)
    //     {
    //         int* output = top_blob.channel(i);

    //         for (int j=0; j<N; j++)
    //         {
    //             int sum = 0;

    //             signed char* vb = (signed char*)bottom_im2row + K * j;
    //             const signed char* va = kernel + K * i;

    //             for (int k=0; k<K; k++)
    //             {
    //                 sum += (int)va[0] * vb[0];

    //                 va += 1;
    //                 vb += 1;
    //             }
    //             output[0] = sum;

    //             output++;
    //         }
    //     }
    // }
}

static void conv_im2col_sgemm_int8_dequant_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel,
        const int kernel_w, const int kernel_h, const int stride_w, const int stride_h, const Mat& _bias, std::vector<float> scale_dequant, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const signed char* kernel = _kernel;
    const float* bias = _bias;

    // im2row
    Mat bottom_im2row(kernel_h * kernel_w * inch, outw * outh, 1UL, opt.workspace_allocator);
    {
        signed char* ret = (signed char*)bottom_im2row;
        int retID = 0;

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                for (int p = 0; p < inch; p++)
                {
                    const signed char* input = bottom_blob.channel(p);
                    for (int u = 0; u < kernel_h; u++)
                    {
                        for (int v = 0; v < kernel_w; v++)
                        {
                            int row = u + i * stride_h;
                            int col = v + j * stride_w;
                            int index = row * w + col;
                            ret[retID] = input[index];
                            retID++;
                        }
                    }
                }
            }
        }
    }

    int kernel_size = kernel_w * kernel_h;
    int out_size = outw * outh;

    // int M = outch;  // outch
    int N = outw * outh;                // outsize or out stride
    int K = kernel_w * kernel_h * inch; // ksize * inch

    // bottom_im2row memory packed 4 x 4
    Mat bottom_tm(4 * kernel_size, inch, out_size / 4 + out_size % 4, (size_t)1u, opt.workspace_allocator);
    {
        int nn_size = out_size >> 2;
        int remain_size_start = nn_size << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = ii * 4;

            const signed char* img0 = bottom_im2row.row<signed char>(i);
            const signed char* img1 = bottom_im2row.row<signed char>(i + 1);
            const signed char* img2 = bottom_im2row.row<signed char>(i + 2);
            const signed char* img3 = bottom_im2row.row<signed char>(i + 3);

            signed char* tmpptr = bottom_tm.channel(i / 4);

            int q = 0;
            for (; q + 1 < inch * kernel_size; q = q + 2)
            {
                tmpptr[0] = img0[0];
                tmpptr[1] = img0[1];
                tmpptr[2] = img1[0];
                tmpptr[3] = img1[1];
                tmpptr[4] = img2[0];
                tmpptr[5] = img2[1];
                tmpptr[6] = img3[0];
                tmpptr[7] = img3[1];

                tmpptr += 8;
                img0 += 2;
                img1 += 2;
                img2 += 2;
                img3 += 2;
            }

            for (; q < inch * kernel_size; q++)
            {
                tmpptr[0] = img0[0];
                tmpptr[1] = img1[0];
                tmpptr[2] = img2[0];
                tmpptr[3] = img3[0];

                tmpptr += 4;
                img0 += 1;
                img1 += 1;
                img2 += 1;
                img3 += 1;
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < out_size; i++)
        {
            const signed char* img0 = bottom_im2row.row<signed char>(i);

            signed char* tmpptr = bottom_tm.channel(i / 4 + i % 4);

            int q = 0;
            for (; q + 1 < inch * kernel_size; q = q + 2)
            {
                tmpptr[0] = img0[0];
                tmpptr[1] = img0[1];

                tmpptr += 2;
                img0 += 2;
            }

            for (; q < inch * kernel_size; q++)
            {
                tmpptr[0] = img0[0];

                tmpptr += 1;
                img0 += 1;
            }
        }
    }

    // kernel memory packed 4 x 4
    Mat kernel_tm(4 * kernel_size, inch, outch / 4 + outch % 4, (size_t)1u, opt.workspace_allocator);
    {
        int nn_outch = 0;
        int remain_outch_start = 0;

        nn_outch = outch >> 2;
        remain_outch_start = nn_outch << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp = 0; pp < nn_outch; pp++)
        {
            int p = pp * 4;

            const signed char* k0 = kernel + (p + 0) * inch * kernel_size;
            const signed char* k1 = kernel + (p + 1) * inch * kernel_size;
            const signed char* k2 = kernel + (p + 2) * inch * kernel_size;
            const signed char* k3 = kernel + (p + 3) * inch * kernel_size;

            signed char* ktmp = kernel_tm.channel(p / 4);

            int q = 0;
            for (; q + 1 < inch * kernel_size; q += 2)
            {
                ktmp[0] = k0[0];
                ktmp[1] = k0[1];
                ktmp[2] = k1[0];
                ktmp[3] = k1[1];
                ktmp[4] = k2[0];
                ktmp[5] = k2[1];
                ktmp[6] = k3[0];
                ktmp[7] = k3[1];

                ktmp += 8;

                k0 += 2;
                k1 += 2;
                k2 += 2;
                k3 += 2;
            }

            for (; q < inch * kernel_size; q++)
            {
                ktmp[0] = k0[0];
                ktmp[1] = k1[0];
                ktmp[2] = k2[0];
                ktmp[3] = k3[0];
                ktmp += 4;

                k0 += 1;
                k1 += 1;
                k2 += 1;
                k3 += 1;
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = remain_outch_start; p < outch; p++)
        {
            const signed char* k0 = kernel + (p + 0) * inch * kernel_size;

            signed char* ktmp = kernel_tm.channel(p / 4 + p % 4);

            int q = 0;
            for (; q + 1 < inch * kernel_size; q = q + 2)
            {
                ktmp[0] = k0[0];
                ktmp[1] = k0[1];
                ktmp += 2;
                k0 += 2;
            }

            for (; q < inch * kernel_size; q++)
            {
                ktmp[0] = k0[0];
                ktmp++;
                k0++;
            }
        }
    }

    // 4x4
    // sgemm(int M, int N, int K, float* A, float* B, float* C)
    {
        // int M = outch;  // outch
        // int N = outw * outh; // outsize or out stride
        // int L = kernel_w * kernel_h * inch; // ksize * inch

        int nn_outch = 0;
        int remain_outch_start = 0;

        nn_outch = outch >> 2;
        remain_outch_start = nn_outch << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp = 0; pp < nn_outch; pp++)
        {
            int i = pp * 4;

            const float bias0 = bias ? bias[i] : 0.f;
            const float bias1 = bias ? bias[i + 1] : 0.f;
            const float bias2 = bias ? bias[i + 2] : 0.f;
            const float bias3 = bias ? bias[i + 3] : 0.f;

            const float scale_dequant0 = scale_dequant[i];
            const float scale_dequant1 = scale_dequant[i + 1];
            const float scale_dequant2 = scale_dequant[i + 2];
            const float scale_dequant3 = scale_dequant[i + 3];

            float* output0 = top_blob.channel(i);
            float* output1 = top_blob.channel(i + 1);
            float* output2 = top_blob.channel(i + 2);
            float* output3 = top_blob.channel(i + 3);

            int j = 0;
            for (; j + 3 < N; j = j + 4)
            {
                signed char* vb = bottom_tm.channel(j / 4);
                signed char* va = kernel_tm.channel(i / 4);

                int sum0[4] = {0};
                int sum1[4] = {0};
                int sum2[4] = {0};
                int sum3[4] = {0};

                int k = 0;

                for (; k + 1 < K; k = k + 2)
                {
                    for (int n = 0; n < 4; n++)
                    {
                        sum0[n] += (int)va[0] * vb[2 * n]; // k0
                        sum0[n] += (int)va[1] * vb[2 * n + 1];

                        sum1[n] += (int)va[2] * vb[2 * n]; // k1
                        sum1[n] += (int)va[3] * vb[2 * n + 1];

                        sum2[n] += (int)va[4] * vb[2 * n]; // k2
                        sum2[n] += (int)va[5] * vb[2 * n + 1];

                        sum3[n] += (int)va[6] * vb[2 * n]; // k3
                        sum3[n] += (int)va[7] * vb[2 * n + 1];
                    }

                    va += 8;
                    vb += 8;
                }

                for (; k < K; k++)
                {
                    for (int n = 0; n < 4; n++)
                    {
                        sum0[n] += (int)va[0] * vb[n];
                        sum1[n] += (int)va[1] * vb[n];
                        sum2[n] += (int)va[2] * vb[n];
                        sum3[n] += (int)va[3] * vb[n];
                    }

                    va += 4;
                    vb += 4;
                }

                for (int n = 0; n < 4; n++)
                {
                    output0[n] = (float)sum0[n] * scale_dequant0 + bias0;
                    output1[n] = (float)sum1[n] * scale_dequant1 + bias1;
                    output2[n] = (float)sum2[n] * scale_dequant2 + bias2;
                    output3[n] = (float)sum3[n] * scale_dequant3 + bias3;
                }
                output0 += 4;
                output1 += 4;
                output2 += 4;
                output3 += 4;
            }

            for (; j < N; j++)
            {
                int sum0 = 0;
                int sum1 = 0;
                int sum2 = 0;
                int sum3 = 0;

                signed char* vb = bottom_tm.channel(j / 4 + j % 4);
                signed char* va = kernel_tm.channel(i / 4);

                int k = 0;

                for (; k + 1 < K; k = k + 2)
                {
                    sum0 += (int)va[0] * vb[0];
                    sum0 += (int)va[1] * vb[1];

                    sum1 += (int)va[2] * vb[0];
                    sum1 += (int)va[3] * vb[1];

                    sum2 += (int)va[4] * vb[0];
                    sum2 += (int)va[5] * vb[1];

                    sum3 += (int)va[6] * vb[0];
                    sum3 += (int)va[7] * vb[1];

                    va += 8;
                    vb += 2;
                }

                for (; k < K; k++)
                {
                    sum0 += (int)va[0] * vb[0];
                    sum1 += (int)va[1] * vb[0];
                    sum2 += (int)va[2] * vb[0];
                    sum3 += (int)va[3] * vb[0];

                    va += 4;
                    vb += 1;
                }

                output0[0] = (float)sum0 * scale_dequant0 + bias0;
                output1[0] = (float)sum1 * scale_dequant1 + bias1;
                output2[0] = (float)sum2 * scale_dequant2 + bias2;
                output3[0] = (float)sum3 * scale_dequant3 + bias3;

                output0++;
                output1++;
                output2++;
                output3++;
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_outch_start; i < outch; i++)
        {
            float* output = top_blob.channel(i);

            const float bias0 = bias ? bias[i] : 0.f;
            const float scale_dequant0 = scale_dequant[i];

            int j = 0;
            for (; j + 3 < N; j = j + 4)
            {
                signed char* vb = bottom_tm.channel(j / 4);
                signed char* va = kernel_tm.channel(i / 4 + i % 4);
                int sum[4] = {0};

                int k = 0;
                for (; k + 1 < K; k = k + 2)
                {
                    for (int n = 0; n < 4; n++)
                    {
                        sum[n] += (int)va[0] * vb[2 * n];
                        sum[n] += (int)va[1] * vb[2 * n + 1];
                    }
                    va += 2;
                    vb += 8;
                }

                for (; k < K; k++)
                {
                    for (int n = 0; n < 4; n++)
                    {
                        sum[n] += (int)va[0] * vb[n];
                    }
                    va += 1;
                    vb += 4;
                }

                for (int n = 0; n < 4; n++)
                {
                    output[n] = (float)sum[n] * scale_dequant0 + bias0;
                }
                output += 4;
            }

            for (; j < N; j++)
            {
                int sum = 0;

                signed char* vb = bottom_tm.channel(j / 4 + j % 4);
                signed char* va = kernel_tm.channel(i / 4 + i % 4);

                for (int k = 0; k < K; k++)
                {
                    sum += (int)va[0] * vb[0];

                    va += 1;
                    vb += 1;
                }
                output[0] = (float)sum * scale_dequant0 + bias0;

                output++;
            }
        }
    }

    // // sgemm(int M, int N, int K, float* A, float* B, float* C)
    // {
    //     for (int i=0; i<M; i++)
    //     {
    //         int* output = top_blob.channel(i);

    //         for (int j=0; j<N; j++)
    //         {
    //             int sum = 0;

    //             signed char* vb = (signed char*)bottom_im2row + K * j;
    //             const signed char* va = kernel + K * i;

    //             for (int k=0; k<K; k++)
    //             {
    //                 sum += (int)va[0] * vb[0];

    //                 va += 1;
    //                 vb += 1;
    //             }
    //             output[0] = sum;

    //             output++;
    //         }
    //     }
    // }
}

static void conv_im2col_sgemm_int8_requant_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel,
        const int kernel_w, const int kernel_h, const int stride_w, const int stride_h, const Mat& _bias, std::vector<float> scale_requant, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const signed char* kernel = _kernel;
    const float* bias = _bias;

    // im2row
    Mat bottom_im2row(kernel_h * kernel_w * inch, outw * outh, 1UL, opt.workspace_allocator);
    {
        signed char* ret = (signed char*)bottom_im2row;
        int retID = 0;

        for (int i = 0; i < outh; i++)
        {
            for (int j = 0; j < outw; j++)
            {
                for (int p = 0; p < inch; p++)
                {
                    const signed char* input = bottom_blob.channel(p);
                    for (int u = 0; u < kernel_h; u++)
                    {
                        for (int v = 0; v < kernel_w; v++)
                        {
                            int row = u + i * stride_h;
                            int col = v + j * stride_w;
                            int index = row * w + col;
                            ret[retID] = input[index];
                            retID++;
                        }
                    }
                }
            }
        }
    }

    int kernel_size = kernel_w * kernel_h;
    int out_size = outw * outh;

    // int M = outch;  // outch
    int N = outw * outh;                // outsize or out stride
    int K = kernel_w * kernel_h * inch; // ksize * inch

    // bottom_im2row memory packed 4 x 4
    Mat bottom_tm(4 * kernel_size, inch, out_size / 4 + out_size % 4, (size_t)1u, opt.workspace_allocator);
    {
        int nn_size = out_size >> 2;
        int remain_size_start = nn_size << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = ii * 4;

            const signed char* img0 = bottom_im2row.row<signed char>(i);
            const signed char* img1 = bottom_im2row.row<signed char>(i + 1);
            const signed char* img2 = bottom_im2row.row<signed char>(i + 2);
            const signed char* img3 = bottom_im2row.row<signed char>(i + 3);

            signed char* tmpptr = bottom_tm.channel(i / 4);

            int q = 0;
            for (; q + 1 < inch * kernel_size; q = q + 2)
            {
                tmpptr[0] = img0[0];
                tmpptr[1] = img0[1];
                tmpptr[2] = img1[0];
                tmpptr[3] = img1[1];
                tmpptr[4] = img2[0];
                tmpptr[5] = img2[1];
                tmpptr[6] = img3[0];
                tmpptr[7] = img3[1];

                tmpptr += 8;
                img0 += 2;
                img1 += 2;
                img2 += 2;
                img3 += 2;
            }

            for (; q < inch * kernel_size; q++)
            {
                tmpptr[0] = img0[0];
                tmpptr[1] = img1[0];
                tmpptr[2] = img2[0];
                tmpptr[3] = img3[0];

                tmpptr += 4;
                img0 += 1;
                img1 += 1;
                img2 += 1;
                img3 += 1;
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < out_size; i++)
        {
            const signed char* img0 = bottom_im2row.row<signed char>(i);

            signed char* tmpptr = bottom_tm.channel(i / 4 + i % 4);

            int q = 0;
            for (; q + 1 < inch * kernel_size; q = q + 2)
            {
                tmpptr[0] = img0[0];
                tmpptr[1] = img0[1];

                tmpptr += 2;
                img0 += 2;
            }

            for (; q < inch * kernel_size; q++)
            {
                tmpptr[0] = img0[0];

                tmpptr += 1;
                img0 += 1;
            }
        }
    }

    // kernel memory packed 4 x 4
    Mat kernel_tm(4 * kernel_size, inch, outch / 4 + outch % 4, (size_t)1u, opt.workspace_allocator);
    {
        int nn_outch = 0;
        int remain_outch_start = 0;

        nn_outch = outch >> 2;
        remain_outch_start = nn_outch << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp = 0; pp < nn_outch; pp++)
        {
            int p = pp * 4;

            const signed char* k0 = kernel + (p + 0) * inch * kernel_size;
            const signed char* k1 = kernel + (p + 1) * inch * kernel_size;
            const signed char* k2 = kernel + (p + 2) * inch * kernel_size;
            const signed char* k3 = kernel + (p + 3) * inch * kernel_size;

            signed char* ktmp = kernel_tm.channel(p / 4);

            int q = 0;
            for (; q + 1 < inch * kernel_size; q += 2)
            {
                ktmp[0] = k0[0];
                ktmp[1] = k0[1];
                ktmp[2] = k1[0];
                ktmp[3] = k1[1];
                ktmp[4] = k2[0];
                ktmp[5] = k2[1];
                ktmp[6] = k3[0];
                ktmp[7] = k3[1];

                ktmp += 8;

                k0 += 2;
                k1 += 2;
                k2 += 2;
                k3 += 2;
            }

            for (; q < inch * kernel_size; q++)
            {
                ktmp[0] = k0[0];
                ktmp[1] = k1[0];
                ktmp[2] = k2[0];
                ktmp[3] = k3[0];
                ktmp += 4;

                k0 += 1;
                k1 += 1;
                k2 += 1;
                k3 += 1;
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = remain_outch_start; p < outch; p++)
        {
            const signed char* k0 = kernel + (p + 0) * inch * kernel_size;

            signed char* ktmp = kernel_tm.channel(p / 4 + p % 4);

            int q = 0;
            for (; q + 1 < inch * kernel_size; q = q + 2)
            {
                ktmp[0] = k0[0];
                ktmp[1] = k0[1];
                ktmp += 2;
                k0 += 2;
            }

            for (; q < inch * kernel_size; q++)
            {
                ktmp[0] = k0[0];
                ktmp++;
                k0++;
            }
        }
    }

    // 4x4
    // sgemm(int M, int N, int K, float* A, float* B, float* C)
    {
        // int M = outch;  // outch
        // int N = outw * outh; // outsize or out stride
        // int L = kernel_w * kernel_h * inch; // ksize * inch

        int nn_outch = 0;
        int remain_outch_start = 0;

        nn_outch = outch >> 2;
        remain_outch_start = nn_outch << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp = 0; pp < nn_outch; pp++)
        {
            int i = pp * 4;

            signed char* output0 = top_blob.channel(i);
            signed char* output1 = top_blob.channel(i + 1);
            signed char* output2 = top_blob.channel(i + 2);
            signed char* output3 = top_blob.channel(i + 3);

            const float bias0 = bias ? bias[i] : 0.f;
            const float bias1 = bias ? bias[i + 1] : 0.f;
            const float bias2 = bias ? bias[i + 2] : 0.f;
            const float bias3 = bias ? bias[i + 3] : 0.f;

            const float scale_requant_in0 = scale_requant[2 * i];
            const float scale_requant_out0 = scale_requant[2 * i + 1];
            const float scale_requant_in1 = scale_requant[2 * (i + 1)];
            const float scale_requant_out1 = scale_requant[2 * (i + 1) + 1];
            const float scale_requant_in2 = scale_requant[2 * (i + 2)];
            const float scale_requant_out2 = scale_requant[2 * (i + 2) + 1];
            const float scale_requant_in3 = scale_requant[2 * (i + 3)];
            const float scale_requant_out3 = scale_requant[2 * (i + 3) + 1];

            int j = 0;
            for (; j + 3 < N; j = j + 4)
            {
                signed char* vb = bottom_tm.channel(j / 4);
                signed char* va = kernel_tm.channel(i / 4);

                int sum0[4] = {0};
                int sum1[4] = {0};
                int sum2[4] = {0};
                int sum3[4] = {0};

                int k = 0;

                for (; k + 1 < K; k = k + 2)
                {
                    for (int n = 0; n < 4; n++)
                    {
                        sum0[n] += (int)va[0] * vb[2 * n]; // k0
                        sum0[n] += (int)va[1] * vb[2 * n + 1];

                        sum1[n] += (int)va[2] * vb[2 * n]; // k1
                        sum1[n] += (int)va[3] * vb[2 * n + 1];

                        sum2[n] += (int)va[4] * vb[2 * n]; // k2
                        sum2[n] += (int)va[5] * vb[2 * n + 1];

                        sum3[n] += (int)va[6] * vb[2 * n]; // k3
                        sum3[n] += (int)va[7] * vb[2 * n + 1];
                    }

                    va += 8;
                    vb += 8;
                }

                for (; k < K; k++)
                {
                    for (int n = 0; n < 4; n++)
                    {
                        sum0[n] += (int)va[0] * vb[n];
                        sum1[n] += (int)va[1] * vb[n];
                        sum2[n] += (int)va[2] * vb[n];
                        sum3[n] += (int)va[3] * vb[n];
                    }

                    va += 4;
                    vb += 4;
                }

                for (int n = 0; n < 4; n++)
                {
                    output0[n] = float2int8(((float)sum0[n] * scale_requant_in0 + bias0) * scale_requant_out0);
                    output1[n] = float2int8(((float)sum1[n] * scale_requant_in1 + bias1) * scale_requant_out1);
                    output2[n] = float2int8(((float)sum2[n] * scale_requant_in2 + bias2) * scale_requant_out2);
                    output3[n] = float2int8(((float)sum3[n] * scale_requant_in3 + bias3) * scale_requant_out3);
                }
                output0 += 4;
                output1 += 4;
                output2 += 4;
                output3 += 4;
            }

            for (; j < N; j++)
            {
                int sum0 = 0;
                int sum1 = 0;
                int sum2 = 0;
                int sum3 = 0;

                signed char* vb = bottom_tm.channel(j / 4 + j % 4);
                signed char* va = kernel_tm.channel(i / 4);

                int k = 0;

                for (; k + 1 < K; k = k + 2)
                {
                    sum0 += (int)va[0] * vb[0];
                    sum0 += (int)va[1] * vb[1];

                    sum1 += (int)va[2] * vb[0];
                    sum1 += (int)va[3] * vb[1];

                    sum2 += (int)va[4] * vb[0];
                    sum2 += (int)va[5] * vb[1];

                    sum3 += (int)va[6] * vb[0];
                    sum3 += (int)va[7] * vb[1];

                    va += 8;
                    vb += 2;
                }

                for (; k < K; k++)
                {
                    sum0 += (int)va[0] * vb[0];
                    sum1 += (int)va[1] * vb[0];
                    sum2 += (int)va[2] * vb[0];
                    sum3 += (int)va[3] * vb[0];

                    va += 4;
                    vb += 1;
                }

                output0[0] = float2int8(((float)sum0 * scale_requant_in0 + bias0) * scale_requant_out0);
                output1[0] = float2int8(((float)sum1 * scale_requant_in1 + bias1) * scale_requant_out1);
                output2[0] = float2int8(((float)sum2 * scale_requant_in2 + bias2) * scale_requant_out2);
                output3[0] = float2int8(((float)sum3 * scale_requant_in3 + bias3) * scale_requant_out3);

                output0++;
                output1++;
                output2++;
                output3++;
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_outch_start; i < outch; i++)
        {
            signed char* output = top_blob.channel(i);

            const float bias0 = bias ? bias[i] : 0.f;

            const float scale_requant_in0 = scale_requant[2 * i];
            const float scale_requant_out0 = scale_requant[2 * i + 1];

            int j = 0;
            for (; j + 3 < N; j = j + 4)
            {
                signed char* vb = bottom_tm.channel(j / 4);
                signed char* va = kernel_tm.channel(i / 4 + i % 4);
                int sum[4] = {0};

                int k = 0;
                for (; k + 1 < K; k = k + 2)
                {
                    for (int n = 0; n < 4; n++)
                    {
                        sum[n] += (int)va[0] * vb[2 * n];
                        sum[n] += (int)va[1] * vb[2 * n + 1];
                    }
                    va += 2;
                    vb += 8;
                }

                for (; k < K; k++)
                {
                    for (int n = 0; n < 4; n++)
                    {
                        sum[n] += (int)va[0] * vb[n];
                    }
                    va += 1;
                    vb += 4;
                }

                for (int n = 0; n < 4; n++)
                {
                    output[n] = float2int8(((float)sum[n] * scale_requant_in0 + bias0) * scale_requant_out0);
                }
                output += 4;
            }

            for (; j < N; j++)
            {
                int sum = 0;

                signed char* vb = bottom_tm.channel(j / 4 + j % 4);
                signed char* va = kernel_tm.channel(i / 4 + i % 4);

                for (int k = 0; k < K; k++)
                {
                    sum += (int)va[0] * vb[0];

                    va += 1;
                    vb += 1;
                }
                output[0] = float2int8(((float)sum * scale_requant_in0 + bias0) * scale_requant_out0);

                output++;
            }
        }
    }

    // // sgemm(int M, int N, int K, float* A, float* B, float* C)
    // {
    //     for (int i=0; i<M; i++)
    //     {
    //         int* output = top_blob.channel(i);

    //         for (int j=0; j<N; j++)
    //         {
    //             int sum = 0;

    //             signed char* vb = (signed char*)bottom_im2row + K * j;
    //             const signed char* va = kernel + K * i;

    //             for (int k=0; k<K; k++)
    //             {
    //                 sum += (int)va[0] * vb[0];

    //                 va += 1;
    //                 vb += 1;
    //             }
    //             output[0] = sum;

    //             output++;
    //         }
    //     }
    // }
}
