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

static void conv7x7s2_int8_sse(const Mat &bottom_blob, Mat &top_blob, const Mat &_kernel, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    int kernel_w = 7;
    int kernel_h = 7;

    int stride_w = 2;
    int stride_h = 2;

    const signed char *kernel = _kernel;

    double start = ncnn::get_current_time();
    // im2col
    Mat bottom_im2col(outw*outh, kernel_h*kernel_w*inch, 1UL, opt.workspace_allocator);
    {
        const int stride = kernel_h*kernel_w*outw*outh;
        signed char* ret = (signed char*)bottom_im2col;
    
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

    // sgemm(int M, int N, int L, float* A, float* B, float* C)  4x4_sgemm
    {
        int M = outch;  // outch
        int N = outw * outh; // outsize or out stride
        int L = kernel_w * kernel_h * inch; // ksize * inch

        const signed char* A = kernel;
        const signed char* B = bottom_im2col;

        int i=0;
        for (; i+3<M; i=i+4)
        {
            int* output0 = top_blob.channel(i);
            int* output1 = top_blob.channel(i+1);
            int* output2 = top_blob.channel(i+2);
            int* output3 = top_blob.channel(i+3);

            int j=0;
            for (; j+3<N; j=j+4)
            {
                int sum0[4] = {0};
                int sum1[4] = {0};
                int sum2[4] = {0};
                int sum3[4] = {0};

                for (int k=0; k<L; k++)
                {
                    for (int n=0; n<4; n++)
                    {
                       sum0[n] += (int)A[i*L+k] * B[k*N+j+n]; 
                       sum1[n] += (int)A[(i+1)*L+k] * B[k*N+j+n];
                       sum2[n] += (int)A[(i+2)*L+k] * B[k*N+j+n];
                       sum3[n] += (int)A[(i+3)*L+k] * B[k*N+j+n];
                    }                                                       
                }

                for (int n=0; n<4; n++)
                {
                    output0[j+n] = sum0[n];
                    output1[j+n] = sum1[n];
                    output2[j+n] = sum2[n];
                    output3[j+n] = sum3[n];
                }                
            }

            for (; j<N; j++)
            {
                int sum0 = 0;
                int sum1 = 0;
                int sum2 = 0;
                int sum3 = 0;

                for (int k=0; k<L; k++)
                {
                    sum0 += (int)A[i*L+k] * B[k*N+j];
                    sum1 += (int)A[(i+1)*L+k] * B[k*N+j];
                    sum2 += (int)A[(i+2)*L+k] * B[k*N+j];
                    sum3 += (int)A[(i+3)*L+k] * B[k*N+j];
                }
                output0[j] = sum0;
                output1[j] = sum1;
                output2[j] = sum2;
                output3[j] = sum3;
            }
        }

        for (; i<M; i++)
        {
            int* output = top_blob.channel(i);

            int j=0;
            for (; j+3<N; j=j+4)
            {
                int sum0 = 0;
                int sum1 = 0;
                int sum2 = 0;
                int sum3 = 0;

                for (int k=0; k<L; k++)
                {
                    sum0 += (int)A[i*L+k] * B[k*N+j];
                    sum1 += (int)A[i*L+k] * B[k*N+j+1];
                    sum2 += (int)A[i*L+k] * B[k*N+j+2];
                    sum3 += (int)A[i*L+k] * B[k*N+j+3];
                }
                output[j] = sum0;
                output[j+1] = sum1;
                output[j+2] = sum2;
                output[j+3] = sum3;
            }

            for (; j<N; j++)
            {
                int sum = 0;

                for (int k=0; k<L; k++)
                {
                    sum += (int)A[i*L+k] * B[k*N+j];
                }
                output[j] = sum;
            }
        }
    }
}
