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

static void conv_im2col_sgemm_int8_sse(const Mat &bottom_blob, Mat &top_blob, const Mat &_kernel, \
            const int kernel_w, const int kernel_h, const int stride_w, const int stride_h, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const signed char *kernel = _kernel;

    // im2col
    Mat bottom_im2col(outw*outh, kernel_h*kernel_w*inch, 1UL, opt.workspace_allocator);
    {
        const int stride = kernel_h*kernel_w*outw*outh;
        signed char* ret = (signed char*)bottom_im2col;
    
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p=0; p<inch; p++)
        {
            const signed char* input = bottom_blob.channel(p);
            int retID = stride * p;
            for (int u=0; u<kernel_h; u++)
            {
                for (int v=0; v<kernel_w; v++)
                {
                    for (int i=0; i<outh; i++)
                    {
                        for (int j=0; j<outw; j++)
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

    // bottom_im2col memory packed 4 x 8
    Mat bottom_tm(8*kernel_size, inch, out_size/8 + out_size%8, (size_t)1u, opt.workspace_allocator);
    {
        int nn_size = out_size >> 3;
        int remain_size_start = nn_size << 3;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii=0; ii<nn_size; ii++)
        {
            int i = ii * 8;

            const signed char* img0 = bottom_im2col.channel(0);
            img0 += i;

            signed char* tmpptr = bottom_tm.channel(i/8);

            for (int q=0; q<inch*kernel_size; q++)
            {
                tmpptr[0] = img0[0];
                tmpptr[1] = img0[1];
                tmpptr[2] = img0[2];
                tmpptr[3] = img0[3];
                tmpptr[4] = img0[4];
                tmpptr[5] = img0[5];
                tmpptr[6] = img0[6];
                tmpptr[7] = img0[7];

                tmpptr += 8;
                img0 += out_size;
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i=remain_size_start; i<out_size; i++)
        {
            const signed char* img0 = bottom_im2col.channel(0);
            img0 += i;

            signed char* tmpptr = bottom_tm.channel(i/8 + i%8);

            for (int q=0; q<inch*kernel_size; q++)
            {
                tmpptr[0] = img0[0];

                tmpptr += 1;
                img0 += out_size;
            }
        }       
    }

    // kernel memory packed 4 x 8
    Mat kernel_tm(4*kernel_size, inch, outch/4 + outch%4, (size_t)1u, opt.workspace_allocator);
    {
        int nn_outch = 0;
        int remain_outch_start = 0;

        nn_outch = outch >> 2;
        remain_outch_start = nn_outch << 2;
        
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp=0; pp<nn_outch; pp++)
        {
            int p = pp * 4;

            const signed char* k0 = kernel + (p+0)*inch*kernel_size;
            const signed char* k1 = kernel + (p+1)*inch*kernel_size;
            const signed char* k2 = kernel + (p+2)*inch*kernel_size;
            const signed char* k3 = kernel + (p+3)*inch*kernel_size;

            signed char* ktmp = kernel_tm.channel(p/4);

            for (int q=0; q<inch*kernel_size; q++)
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
        for (int p=remain_outch_start; p<outch; p++)
        {
            const signed char* k0 = kernel + (p+0)*inch*kernel_size;

            signed char* ktmp = kernel_tm.channel(p/4 + p%4);

            for (int q=0; q<inch*kernel_size; q++)
            {
                ktmp[0] = k0[0];
                ktmp++;
                k0++;
            }
        }
    }

    // sgemm(int M, int N, int L, float* A, float* B, float* C)
    {
        // int M = outch;  // outch
        int N = outw * outh; // outsize or out stride
        int L = kernel_w * kernel_h * inch; // ksize * inch

        int nn_outch = 0;
        int remain_outch_start = 0;

        nn_outch = outch >> 2;
        remain_outch_start = nn_outch << 2;
        
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp=0; pp<nn_outch; pp++)
        {
            int i = pp * 4;

            int* output0 = top_blob.channel(i);
            int* output1 = top_blob.channel(i+1);
            int* output2 = top_blob.channel(i+2);
            int* output3 = top_blob.channel(i+3);

            int j=0;
            for (; j+7<N; j=j+8)
            {
                signed char* vb = bottom_tm.channel(j/8);
                signed char* va = kernel_tm.channel(i/4);
                
                int sum0[8] = {0};
                int sum1[8] = {0};
                int sum2[8] = {0};
                int sum3[8] = {0};
               
                int k=0;
                for (; k+7<L; k=k+8)
                {
                    for (int n=0; n<8; n++)
                    {
                        sum0[n] += (int)va[0] * vb[n];
                        sum1[n] += (int)va[1] * vb[n];
                        sum2[n] += (int)va[2] * vb[n];
                        sum3[n] += (int)va[3] * vb[n];
                        va += 4;

                        sum0[n] += (int)va[0] * vb[n+8];
                        sum1[n] += (int)va[1] * vb[n+8];
                        sum2[n] += (int)va[2] * vb[n+8];
                        sum3[n] += (int)va[3] * vb[n+8];
                        va += 4;

                        sum0[n] += (int)va[0] * vb[n+16];
                        sum1[n] += (int)va[1] * vb[n+16];
                        sum2[n] += (int)va[2] * vb[n+16];
                        sum3[n] += (int)va[3] * vb[n+16];
                        va += 4;

                        sum0[n] += (int)va[0] * vb[n+24];
                        sum1[n] += (int)va[1] * vb[n+24];
                        sum2[n] += (int)va[2] * vb[n+24];
                        sum3[n] += (int)va[3] * vb[n+24];
                        va += 4;

                        sum0[n] += (int)va[0] * vb[n+32];
                        sum1[n] += (int)va[1] * vb[n+32];
                        sum2[n] += (int)va[2] * vb[n+32];
                        sum3[n] += (int)va[3] * vb[n+32];
                        va += 4;

                        sum0[n] += (int)va[0] * vb[n+40];
                        sum1[n] += (int)va[1] * vb[n+40];
                        sum2[n] += (int)va[2] * vb[n+40];
                        sum3[n] += (int)va[3] * vb[n+40];
                        va += 4;

                        sum0[n] += (int)va[0] * vb[n+48];
                        sum1[n] += (int)va[1] * vb[n+48];
                        sum2[n] += (int)va[2] * vb[n+48];
                        sum3[n] += (int)va[3] * vb[n+48];
                        va += 4;

                        sum0[n] += (int)va[0] * vb[n+56];
                        sum1[n] += (int)va[1] * vb[n+56];
                        sum2[n] += (int)va[2] * vb[n+56];
                        sum3[n] += (int)va[3] * vb[n+56];
                        va -= 28;
                    }

                    va += 32;
                    vb += 64;
                }

                for (; k<L; k++)
                {
                    for (int n=0; n<8; n++)
                    {
                        sum0[n] += (int)va[0] * vb[n];
                        sum1[n] += (int)va[1] * vb[n];
                        sum2[n] += (int)va[2] * vb[n];
                        sum3[n] += (int)va[3] * vb[n];
                    }
                    
                    va += 4;
                    vb += 8;
                }

                for (int n=0; n<8; n++)
                {
                    output0[n] = sum0[n];
                    output1[n] = sum1[n];
                    output2[n] = sum2[n];
                    output3[n] = sum3[n];
                }
                output0 += 8;
                output1 += 8;
                output2 += 8;
                output3 += 8;
            }

            for (; j<N; j++)
            {                
                int sum0 = 0;
                int sum1 = 0;
                int sum2 = 0;
                int sum3 = 0;

                signed char* vb = bottom_tm.channel(j/8 + j%8);
                signed char* va = kernel_tm.channel(i/4);

                for (int k=0; k<L; k++)
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
        for (int i=remain_outch_start; i<outch; i++)
        {
            int* output = top_blob.channel(i);

            int j=0;
            for (; j+7<N; j=j+8)
            {
                signed char* vb = bottom_tm.channel(j/8);
                signed char* va = kernel_tm.channel(i/4 + i%4);
                int sum[8] = {0};

                int k=0;
                for (; k+7<L; k=k+8)
                {
                    for (int n=0; n<8; n++)
                    {
                        sum[n] += (int)va[0] * vb[n];
                        sum[n] += (int)va[1] * vb[n+8];
                        sum[n] += (int)va[2] * vb[n+16];
                        sum[n] += (int)va[3] * vb[n+24];
                        sum[n] += (int)va[4] * vb[n+32];
                        sum[n] += (int)va[5] * vb[n+40];
                        sum[n] += (int)va[6] * vb[n+48];
                        sum[n] += (int)va[7] * vb[n+56];
                    }
                    va += 8;
                    vb += 64;
                }

                for (; k<L; k++)
                {
                    for (int n=0; n<8; n++)
                    {
                        sum[n] += (int)va[0] * vb[n];
                    }
                    va += 1;
                    vb += 8;
                }

                for (int n=0; n<8; n++)
                {
                    output[n] = sum[n];
                }
                output += 8;
            }

            for (; j<N; j++)
            {
                int sum = 0;

                signed char* vb = bottom_tm.channel(j/8 + j%8);
                signed char* va = kernel_tm.channel(i/4 + i%4);

                for (int k=0; k<L; k++)
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
}
