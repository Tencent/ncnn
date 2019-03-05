// SenseNets is pleased to support the open source community by supporting ncnn available.
//
// Copyright (C) 2018 SenseNets Technology Ltd. All rights reserved.
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
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

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

static void conv3x3s1_transform_kernel_int8_neon(const Mat& _kernel, Mat& kernel_tm, int inch, int outch)
{
    kernel_tm.create(4*9, inch, outch/4 + outch%4, (size_t)1u);

    const signed char* kernel = _kernel;

    int p=0;
    for (; p+3<outch; p+=4)
    {
        const signed char* k0 = kernel + (p+0)*inch*9;
        const signed char* k1 = kernel + (p+1)*inch*9;
        const signed char* k2 = kernel + (p+2)*inch*9;
        const signed char* k3 = kernel + (p+3)*inch*9;

        signed char* ktmp = kernel_tm.channel(p/4);

        for (int q=0; q<inch; q++)
        {
            for (int k=0; k<9; k++)
            {
                ktmp[0] = k0[k];
                ktmp[1] = k1[k];
                ktmp[2] = k2[k];
                ktmp[3] = k3[k];
                ktmp += 4;
            }

            k0 += 9;
            k1 += 9;
            k2 += 9;
            k3 += 9;
        }
    }
    for (; p<outch; p++)
    {
        const signed char* k0 = kernel + (p+0)*inch*9;

        signed char* ktmp = kernel_tm.channel(p/4 + p%4);

        for (int q=0; q<inch; q++)
        {
            for (int k=0; k<9; k++)
            {
                ktmp[k] = k0[k];
            }
            ktmp += 9;

            k0 += 9;
        }
    }
}

static void conv3x3s1_winograd23_transform_kernel_int8_neon(const Mat& kernel, std::vector<Mat> &kernel_tm2, int inch, int outch)
{
    Mat kernel_tm(4*4, inch, outch, 2ul);  

    // G
    const short ktm[4][3] = {
        {   2,     0,     0},
        {   1,     1,     1},
        {   1,    -1,     1},
        {   0,     0,     2}
    };

    #pragma omp parallel for
    for (int p = 0; p<outch; p++)
    {
        for (int q = 0; q<inch; q++)
        {
            const signed char* kernel0 = (const signed char*)kernel + p*inch * 9 + q * 9;
            short* kernel_tm0 = kernel_tm.channel(p).row<short>(q);

            // transform kernel
            const signed char* k0 = kernel0;
            const signed char* k1 = kernel0 + 3;
            const signed char* k2 = kernel0 + 6;

            // h
            short tmp[4][3];
            for (int i=0; i<4; i++)
            {
                tmp[i][0] = (short)k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                tmp[i][1] = (short)k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                tmp[i][2] = (short)k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
            }

            // U
            for (int j=0; j<4; j++)
            {
                short* tmpp = &tmp[j][0];

                for (int i=0; i<4; i++)
                {
                    kernel_tm0[j*4 + i] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                }
            }
        }
    }

    for (int r=0; r<4; r++)
    {
        Mat kernel_tm_test(4*8, inch, outch/8 + (outch%8)/4 + outch%4, 2u);

        int p = 0;
        for (; p+7<outch; p+=8)
        {
            const short* kernel0 = (const short*)kernel_tm + (p+0)*inch*16;
            const short* kernel1 = (const short*)kernel_tm + (p+1)*inch*16;
            const short* kernel2 = (const short*)kernel_tm + (p+2)*inch*16;
            const short* kernel3 = (const short*)kernel_tm + (p+3)*inch*16;
            const short* kernel4 = (const short*)kernel_tm + (p+4)*inch*16;
            const short* kernel5 = (const short*)kernel_tm + (p+5)*inch*16;
            const short* kernel6 = (const short*)kernel_tm + (p+6)*inch*16;
            const short* kernel7 = (const short*)kernel_tm + (p+7)*inch*16;

            short* ktmp = kernel_tm_test.channel(p/8);

            for (int q=0; q<inch; q++)
            {
                ktmp[0] = kernel0[r*4+0];
                ktmp[1] = kernel0[r*4+1];
                ktmp[2] = kernel0[r*4+2];
                ktmp[3] = kernel0[r*4+3];

                ktmp[4] = kernel1[r*4+0];
                ktmp[5] = kernel1[r*4+1];
                ktmp[6] = kernel1[r*4+2];
                ktmp[7] = kernel1[r*4+3];

                ktmp[8] = kernel2[r*4+0];
                ktmp[9] = kernel2[r*4+1];
                ktmp[10] = kernel2[r*4+2];
                ktmp[11] = kernel2[r*4+3];

                ktmp[12] = kernel3[r*4+0];
                ktmp[13] = kernel3[r*4+1];
                ktmp[14] = kernel3[r*4+2];
                ktmp[15] = kernel3[r*4+3];

                ktmp[16] = kernel4[r*4+0];
                ktmp[17] = kernel4[r*4+1];
                ktmp[18] = kernel4[r*4+2];
                ktmp[19] = kernel4[r*4+3];

                ktmp[20] = kernel5[r*4+0];
                ktmp[21] = kernel5[r*4+1];
                ktmp[22] = kernel5[r*4+2];
                ktmp[23] = kernel5[r*4+3];

                ktmp[24] = kernel6[r*4+0];
                ktmp[25] = kernel6[r*4+1];
                ktmp[26] = kernel6[r*4+2];
                ktmp[27] = kernel6[r*4+3];

                ktmp[28] = kernel7[r*4+0];
                ktmp[29] = kernel7[r*4+1];
                ktmp[30] = kernel7[r*4+2];
                ktmp[31] = kernel7[r*4+3];

                ktmp += 32;
                kernel0 += 16;
                kernel1 += 16;
                kernel2 += 16;
                kernel3 += 16;
                kernel4 += 16;
                kernel5 += 16;
                kernel6 += 16;
                kernel7 += 16;
            }
        }

        for (; p+3<outch; p+=4)
        {
            const short* kernel0 = (const short*)kernel_tm + (p+0)*inch*16;
            const short* kernel1 = (const short*)kernel_tm + (p+1)*inch*16;
            const short* kernel2 = (const short*)kernel_tm + (p+2)*inch*16;
            const short* kernel3 = (const short*)kernel_tm + (p+3)*inch*16;

            short* ktmp = kernel_tm_test.channel(p/8 + (p%8)/4);

            for (int q=0; q<inch; q++)
            {
                ktmp[0] = kernel0[r*4+0];
                ktmp[1] = kernel0[r*4+1];
                ktmp[2] = kernel0[r*4+2];
                ktmp[3] = kernel0[r*4+3];

                ktmp[4] = kernel1[r*4+0];
                ktmp[5] = kernel1[r*4+1];
                ktmp[6] = kernel1[r*4+2];
                ktmp[7] = kernel1[r*4+3];

                ktmp[8] = kernel2[r*4+0];
                ktmp[9] = kernel2[r*4+1];
                ktmp[10] = kernel2[r*4+2];
                ktmp[11] = kernel2[r*4+3];

                ktmp[12] = kernel3[r*4+0];
                ktmp[13] = kernel3[r*4+1];
                ktmp[14] = kernel3[r*4+2];
                ktmp[15] = kernel3[r*4+3];                             

                ktmp += 16;
                kernel0 += 16;
                kernel1 += 16;
                kernel2 += 16;
                kernel3 += 16;                
            }
        }

        for (; p<outch; p++)
        {
            const short* kernel0 = (const short*)kernel_tm + p*inch*16;

            short* ktmp = kernel_tm_test.channel(p/8 + (p%8)/4 + p%4);

            for (int q=0; q<inch; q++)
            {
                ktmp[0] = kernel0[r*4+0];
                ktmp[1] = kernel0[r*4+1];
                ktmp[2] = kernel0[r*4+2];
                ktmp[3] = kernel0[r*4+3];

                ktmp += 4;
                kernel0 += 16;
            }        
        }
        kernel_tm2.push_back(kernel_tm_test);
    }
}

static void conv3x3s2_transform_kernel_int8_neon(const Mat& _kernel, Mat& kernel_tm, int inch, int outch)
{
    kernel_tm.create(8*9, inch, outch/8 + outch%8, (size_t)1u);

    const signed char* kernel = _kernel;

    int p=0;
    for (; p+7<outch; p+=8)
    {
        const signed char* k0 = kernel + (p+0)*inch*9;
        const signed char* k1 = kernel + (p+1)*inch*9;
        const signed char* k2 = kernel + (p+2)*inch*9;
        const signed char* k3 = kernel + (p+3)*inch*9;
        const signed char* k4 = kernel + (p+4)*inch*9;
        const signed char* k5 = kernel + (p+5)*inch*9;
        const signed char* k6 = kernel + (p+6)*inch*9;
        const signed char* k7 = kernel + (p+7)*inch*9;

        signed char* ktmp = kernel_tm.channel(p/8);

        for (int q=0; q<inch; q++)
        {
            for (int k=0; k<9; k++)
            {
                ktmp[0] = k0[k];
                ktmp[1] = k1[k];
                ktmp[2] = k2[k];
                ktmp[3] = k3[k];
                ktmp[4] = k4[k];
                ktmp[5] = k5[k];
                ktmp[6] = k6[k];
                ktmp[7] = k7[k];
                ktmp += 8;
            }

            k0 += 9;
            k1 += 9;
            k2 += 9;
            k3 += 9;
            k4 += 9;
            k5 += 9;
            k6 += 9;
            k7 += 9;
        }
    }
    for (; p<outch; p++)
    {
        const signed char* k0 = kernel + (p+0)*inch*9;

        signed char* ktmp = kernel_tm.channel(p/8 + p%8);

        for (int q=0; q<inch; q++)
        {
            for (int k=0; k<9; k++)
            {
                ktmp[k] = k0[k];
            }
            ktmp += 9;

            k0 += 9;
        }
    }
}

#if __aarch64__
static void conv3x3s1_winograd23_int8_neon(const Mat& bottom_blob, Mat& top_blob, const std::vector<Mat> &kernel_tm_test, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    // pad to 2n+2, winograd F(2,3)
    Mat bottom_blob_bordered = bottom_blob;

    outw = (outw + 1) / 2 * 2;
    outh = (outh + 1) / 2 * 2;

    w = outw + 2;
    h = outh + 2;
    copy_make_border(bottom_blob, bottom_blob_bordered, 0, h - bottom_blob.h, 0, w - bottom_blob.w, 0, 0.f, opt.workspace_allocator, opt.num_threads);  
    
    // double start = ncnn::get_current_time();
    // BEGIN transform input
    Mat bottom_blob_tm;
    {
        int w_tm = outw / 2 * 4;
        int h_tm = outh / 2 * 4;

        int nColBlocks = h_tm/4; // may be the block num in FeatherCNN
        int nRowBlocks = w_tm/4;

        const int tiles = nColBlocks * nRowBlocks;

        bottom_blob_tm.create(4, inch, tiles*4, 2u, opt.workspace_allocator);

        // BT
        // const float itm[4][4] = {
        //     {1.0f,  0.0f, -1.0f,  0.0f},
        //     {0.0f,  1.0f,  1.00f, 0.0f},
        //     {0.0f, -1.0f,  1.00f, 0.0f},
        //     {0.0f, -1.0f,  0.00f, 1.0f}
        // };        

        for (int q=0; q<inch; q++)
        {
            const signed char* img = bottom_blob_bordered.channel(q);

            for (int j=0; j<nColBlocks; j++)
            {
                const signed char* r0 = img + w * j * 2;
                const signed char* r1 = r0 + w;
                const signed char* r2 = r1 + w;
                const signed char* r3 = r2 + w;

                for (int i = 0; i<nRowBlocks; i++)
                {
                    short* out_tm0 = bottom_blob_tm.channel(tiles*0+j*nRowBlocks+i).row<short>(q);
                    short* out_tm1 = bottom_blob_tm.channel(tiles*1+j*nRowBlocks+i).row<short>(q);
                    short* out_tm2 = bottom_blob_tm.channel(tiles*2+j*nRowBlocks+i).row<short>(q);
                    short* out_tm3 = bottom_blob_tm.channel(tiles*3+j*nRowBlocks+i).row<short>(q);

                    short d0[4],d1[4],d2[4],d3[4];
                    short w0[4],w1[4],w2[4],w3[4];
                    short t0[4],t1[4],t2[4],t3[4];
                    // load 
                    for (int n = 0; n < 4; n++)
                    {
                        d0[n] = r0[n];
                        d1[n] = r1[n];
                        d2[n] = r2[n];
                        d3[n] = r3[n];
                    }
                    // w = B_t * d
                    for (int n = 0; n < 4; n++)
                    {   
                        w0[n] = d0[n] - d2[n];
                        w1[n] = d1[n] + d2[n];
                        w2[n] = d2[n] - d1[n];
                        w3[n] = d3[n] - d1[n];
                    } 
                    // transpose d to d_t
                    {
                        t0[0]=w0[0]; t1[0]=w0[1]; t2[0]=w0[2]; t3[0]=w0[3];
                        t0[1]=w1[0]; t1[1]=w1[1]; t2[1]=w1[2]; t3[1]=w1[3];
                        t0[2]=w2[0]; t1[2]=w2[1]; t2[2]=w2[2]; t3[2]=w2[3];
                        t0[3]=w3[0]; t1[3]=w3[1]; t2[3]=w3[2]; t3[3]=w3[3];
                    }
                    // U = B_t * d_t
                    for (int n = 0; n < 4; n++)
                    {   
                        d0[n] = t0[n] - t2[n];
                        d1[n] = t1[n] + t2[n];
                        d2[n] = t2[n] - t1[n];
                        d3[n] = t3[n] - t1[n];
                    }                
                    // save to out_tm
                    for (int n = 0; n < 4; n++)
                    {
                        out_tm0[n] = d0[n];
                        out_tm1[n] = d1[n];
                        out_tm2[n] = d2[n];
                        out_tm3[n] = d3[n];
                    }
                        
                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    r3 += 2;    
                }
            }
        }
    }
    bottom_blob_bordered = Mat();

    // double end = ncnn::get_current_time();
    // printf("trans A : %.3f ms\n", end - start);
    // start = ncnn::get_current_time();

    // BEGIN dot
    Mat top_blob_tm;
    {
        int w_tm = outw / 2 * 4;
        int h_tm = outh / 2 * 4;

        int nColBlocks = h_tm/4; // may be the block num in FeatherCNN
        int nRowBlocks = w_tm/4;

        const int tiles = nColBlocks * nRowBlocks; 

        top_blob_tm.create(16, tiles, outch, 4u, opt.workspace_allocator);

        for (int r=0; r<4; r++)
        {
            int nn_outch = 0;
            int remain_outch_start = 0;

            nn_outch = outch >> 3;
            remain_outch_start = nn_outch << 3;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int pp=0; pp<nn_outch; pp++)
            {
                int p = pp * 8;

                int* output0_tm = top_blob_tm.channel(p);
                int* output1_tm = top_blob_tm.channel(p+1);
                int* output2_tm = top_blob_tm.channel(p+2);
                int* output3_tm = top_blob_tm.channel(p+3);
                int* output4_tm = top_blob_tm.channel(p+4);
                int* output5_tm = top_blob_tm.channel(p+5);
                int* output6_tm = top_blob_tm.channel(p+6);
                int* output7_tm = top_blob_tm.channel(p+7);

                output0_tm = output0_tm + r*4;
                output1_tm = output1_tm + r*4;
                output2_tm = output2_tm + r*4;
                output3_tm = output3_tm + r*4;
                output4_tm = output4_tm + r*4;
                output5_tm = output5_tm + r*4;
                output6_tm = output6_tm + r*4;
                output7_tm = output7_tm + r*4;

                for (int i=0; i<tiles; i++)
                {
                    const short* kptr = kernel_tm_test[r].channel(p/8);
                    const short* r0 = bottom_blob_tm.channel(tiles*r+i);

                    int sum0[4] = {0};
                    int sum1[4] = {0};
                    int sum2[4] = {0};
                    int sum3[4] = {0};
                    int sum4[4] = {0};
                    int sum5[4] = {0};
                    int sum6[4] = {0};
                    int sum7[4] = {0};

                    for (int q=0; q<inch; q++)
                    {
                        for (int n=0; n<4; n++)
                        {
                            sum0[n] += (int)r0[n] * kptr[n];
                            sum1[n] += (int)r0[n] * kptr[n+4];
                            sum2[n] += (int)r0[n] * kptr[n+8];
                            sum3[n] += (int)r0[n] * kptr[n+12];
                            sum4[n] += (int)r0[n] * kptr[n+16];
                            sum5[n] += (int)r0[n] * kptr[n+20];
                            sum6[n] += (int)r0[n] * kptr[n+24];
                            sum7[n] += (int)r0[n] * kptr[n+28];
                        }
                        kptr += 32;
                        r0 += 4;
                    }

                    for (int n=0; n<4; n++)
                    {
                        output0_tm[n] = sum0[n];
                        output1_tm[n] = sum1[n];
                        output2_tm[n] = sum2[n];
                        output3_tm[n] = sum3[n];
                        output4_tm[n] = sum4[n];
                        output5_tm[n] = sum5[n];
                        output6_tm[n] = sum6[n];
                        output7_tm[n] = sum7[n];
                    }

                    output0_tm += 16;
                    output1_tm += 16;
                    output2_tm += 16;
                    output3_tm += 16;
                    output4_tm += 16;
                    output5_tm += 16;
                    output6_tm += 16;
                    output7_tm += 16;
                }
            }

            nn_outch = (outch - remain_outch_start) >> 2;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int pp=0; pp<nn_outch; pp++)
            {
                int p = remain_outch_start + pp * 4;

                int* output0_tm = top_blob_tm.channel(p);
                int* output1_tm = top_blob_tm.channel(p+1);
                int* output2_tm = top_blob_tm.channel(p+2);
                int* output3_tm = top_blob_tm.channel(p+3);

                output0_tm = output0_tm + r*4;
                output1_tm = output1_tm + r*4;
                output2_tm = output2_tm + r*4;
                output3_tm = output3_tm + r*4;

                for (int i=0; i<tiles; i++)
                {
                    const short* kptr = kernel_tm_test[r].channel(p/8 + (p%8)/4);
                    const short* r0 = bottom_blob_tm.channel(tiles*r+i);

                    int sum0[4] = {0};
                    int sum1[4] = {0};
                    int sum2[4] = {0};
                    int sum3[4] = {0};

                    for (int q=0; q<inch; q++)
                    {   
                        for (int n=0; n<4; n++)
                        {
                            sum0[n] += (int)r0[n] * kptr[n];
                            sum1[n] += (int)r0[n] * kptr[n+4];
                            sum2[n] += (int)r0[n] * kptr[n+8];
                            sum3[n] += (int)r0[n] * kptr[n+12];
                        }
                        kptr += 16;
                        r0 += 4;
                    }

                    for (int n=0; n<4; n++)
                    {
                        output0_tm[n] = sum0[n];
                        output1_tm[n] = sum1[n];
                        output2_tm[n] = sum2[n];
                        output3_tm[n] = sum3[n];
                    }

                    output0_tm += 16;
                    output1_tm += 16;
                    output2_tm += 16;
                    output3_tm += 16;
                }
            }

            remain_outch_start += nn_outch << 2;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p=remain_outch_start; p<outch; p++)
            {
                int* output0_tm = top_blob_tm.channel(p);

                output0_tm = output0_tm + r*4;

                for (int i=0; i<tiles; i++)
                {
                    const short* kptr = kernel_tm_test[r].channel(p/8 + (p%8)/4 + p%4);
                    const short* r0 = bottom_blob_tm.channel(tiles*r+i);

                    int sum0[4] = {0};

                    for (int q=0; q<inch; q++)
                    {
                        for (int n=0; n<4; n++)
                        {
                            sum0[n] += (int)r0[n] * kptr[n];
                        }
                        kptr += 4; 
                        r0 += 4;
                    }

                    for (int n=0; n<4; n++)
                    {
                        output0_tm[n] = sum0[n];
                    }
                    output0_tm += 16;
                }
            }
        }   
    }
    bottom_blob_tm = Mat();
    // END dot    

    // end = ncnn::get_current_time();
    // printf("dot B   : %.3f ms\n", end - start);
    // start = ncnn::get_current_time();
    // BEGIN transform output
    Mat top_blob_bordered;
    top_blob_bordered.create(outw, outh, outch, 4u, opt.workspace_allocator);
    {
        // AT
        // const float itm[2][4] = {
        //     {1.0f,  1.0f,  1.0f,  0.0f},
        //     {0.0f,  1.0f, -1.0f,  1.0f}
        // }; 

        int w_tm = outw / 2 * 4;
        int h_tm = outh / 2 * 4;

        int nColBlocks = h_tm/4; // may be the block num in FeatherCNN
        int nRowBlocks = w_tm/4;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p=0; p<outch; p++)
        {
            int* out_tile = top_blob_tm.channel(p);
            int* outRow0 = top_blob_bordered.channel(p);
            int* outRow1 = outRow0 + outw;     

            for (int j=0; j<nColBlocks; j++)
            {
                for(int i=0; i<nRowBlocks; i++)
                {
                    int s0[4],s1[4],s2[4],s3[4];
                    int w0[4],w1[4];
                    int d0[2],d1[2],d2[2],d3[2];
                    int o0[2],o1[2];
                    // load
                    for (int n = 0; n < 4; n++)
                    {
                        s0[n] = out_tile[n];
                        s1[n] = out_tile[n+ 4];
                        s2[n] = out_tile[n+ 8];
                        s3[n] = out_tile[n+12];
                    }
                    // w = A_T * W
                    for (int n = 0; n < 4; n++)
                    {
                        w0[n] = s0[n] + s1[n] + s2[n];
                        w1[n] = s1[n] - s2[n] + s3[n];
                    }
                    // transpose w to w_t
                    {
                        d0[0] = w0[0]; d0[1] = w1[0];
                        d1[0] = w0[1]; d1[1] = w1[1];
                        d2[0] = w0[2]; d2[1] = w1[2];
                        d3[0] = w0[3]; d3[1] = w1[3];
                    }
                    // Y = A_T * w_t
                    for (int n = 0; n < 2; n++)
                    {
                        o0[n] = d0[n] + d1[n] + d2[n];
                        o1[n] = d1[n] - d2[n] + d3[n];
                    }
                    // save to top blob tm,why right 2,because the G' = G*2
                    outRow0[0] = o0[0] >> 2;
                    outRow0[1] = o0[1] >> 2;
                    outRow1[0] = o1[0] >> 2;
                    outRow1[1] = o1[1] >> 2;

                    out_tile += 16;

                    outRow0 += 2;
                    outRow1 += 2;
                }

                outRow0 += outw;
                outRow1 += outw;
            }
        }        
    }
    // END transform output 
    // end = ncnn::get_current_time();
    // printf("trans C : %.3f ms\n", end - start);
    
    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt.blob_allocator, opt.num_threads);  
}

static void conv3x3s1_packed_int8_neon(const Mat &bottom_blob, Mat &top_blob, const Mat &_kernel, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    int nn_outch = outch >> 2;
    int remain_outch_start = nn_outch << 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp=0; pp < nn_outch; pp++)
    {
        int p = pp * 4;

        Mat out0 = top_blob.channel(p+0);
        Mat out1 = top_blob.channel(p+1);
        Mat out2 = top_blob.channel(p+2);
        Mat out3 = top_blob.channel(p+3);

        out0.fill(0);
        out1.fill(0);
        out2.fill(0);
        out3.fill(0);

        const signed char* ktmp = _kernel.channel(p/4);

        for (int q = 0; q < inch; q++)
        {
            int* outptr0 = out0;
            int* outptr1 = out1;
            int* outptr2 = out2;
            int* outptr3 = out3;

            const signed char *img0 = bottom_blob.channel(q);

            const signed char *r0 = img0;
            const signed char *r1 = img0 + w;
            const signed char *r2 = img0 + w * 2;

            int i = 0;    

            for (; i < outh; i++)
            {
#if 0 //__ARM_NEON
                int nn = outw >> 3;
                int remain = outw & 7;
#else
                int remain = outw;
#endif // __ARM_NEON

#if 0 //__ARM_NEON
                if (nn > 0)
                {
                asm volatile(
                    "0:                         \n"
                    "vld1.s8    {d0-d3}, [%8]!  \n"// d0=(k00-k30 k01-k31) d1=(k02-k32 k03-k33) d2=(k04-k34 k05-k35) d3=(k06-k36 k07-k37)
                    // r0
                    "pld        [%5, #128]      \n"
                    "vld1.s8    {d8-d9}, [%5]   \n"// d8=r00-r07 d9=r08-r015 q4
                    "add        %5, #8          \n"
                    "pld        [%1, #128]      \n"
                    "vld1.s32   {d12-d15}, [%1] \n"// sum00-sum07 q6 q7
                    "pld        [%2, #128]      \n"
                    "vld1.s32   {d16-d19}, [%2] \n"// sum10-sum17 q8 q9
                    "pld        [%3, #128]      \n"
                    "vld1.s32   {d20-d23}, [%3] \n"// sum20-sum27 q10 q11
                    "pld        [%4, #128]      \n"
                    "vld1.s32   {d24-d27}, [%4] \n"// sum30-sum37 q12 q13
                    
                    "vmovl.s8   q3, d3          \n"// d6(k06-k36) d7(k07-k37) 
                    "vmovl.s8   q2, d2          \n"// d4(k04-k34) d5(k05-k35)
                    "vmovl.s8   q1, d1          \n"// d2(k02-k32) d3(k03-k33)
                    "vmovl.s8   q0, d0          \n"// d0(k00-k30) d1(k01-k31)
                    "vmovl.s8   q5, d8          \n"// d10(r00-r03) d11(r04-r07)
                    
                    "vmlal.s16  q6, d10, d0[0]  \n"// sum(00-07) += (r00-r07) * k00
                    "vmlal.s16  q7, d11, d0[0]  \n"
                    "vmlal.s16  q8, d10, d0[1]  \n"// sum(10-17) += (r00-r07) * k10
                    "vmlal.s16  q9, d11, d0[1]  \n"     
                    "vmlal.s16  q10, d10, d0[2] \n"// sum(20-27) += (r00-r07) * k20
                    "vmlal.s16  q11, d11, d0[2] \n"
                    "vmlal.s16  q12, d10, d0[3] \n"// sum(30-37) += (r00-r07) * k30
                    "vmlal.s16  q13, d11, d0[3] \n"

                    "vext.s8    q4, q4, #1      \n"// d8=r01-r08 q4
                    "vmovl.s8   q5, d8          \n"// d10(r01-r04) d11(r05-r08)
                    
                    "vmlal.s16  q6, d10, d1[0]  \n"// sum(00-07) += (r01-r08) * k01
                    "vmlal.s16  q7, d11, d1[0]  \n"
                    "vmlal.s16  q8, d10, d1[1]  \n"// sum(10-17) += (r01-r08) * k11
                    "vmlal.s16  q9, d11, d1[1]  \n"     
                    "vmlal.s16  q10, d10, d1[2] \n"// sum(20-27) += (r01-r08) * k21
                    "vmlal.s16  q11, d11, d1[2] \n"
                    "vmlal.s16  q12, d10, d1[3] \n"// sum(30-37) += (r01-r08) * k31
                    "vmlal.s16  q13, d11, d1[3] \n"
                    
                    "vext.s8    q4, q4, #1      \n"// d8=r02-r09 q4
                    "vmovl.s8   q5, d8          \n"// d10(r02-r05) d11(r06-r09)
                    
                    "vmlal.s16  q6, d10, d2[0]  \n"// sum(00-07) += (r02-r09) * k02
                    "vmlal.s16  q7, d11, d2[0]  \n"
                    "vmlal.s16  q8, d10, d2[1]  \n"// sum(10-17) += (r02-r09) * k12
                    "vmlal.s16  q9, d11, d2[1]  \n"     
                    "vmlal.s16  q10, d10, d2[2] \n"// sum(20-27) += (r02-r09) * k22
                    "vmlal.s16  q11, d11, d2[2] \n"
                    "vmlal.s16  q12, d10, d2[3] \n"// sum(30-37) += (r02-r09) * k32
                    "vmlal.s16  q13, d11, d2[3] \n"                 
                    
                    // r1
                    "pld        [%6, #128]      \n"
                    "vld1.s8    {d8-d9}, [%6]   \n"// d8=r10-r17 d9=r18-r115 q4
                    "add        %6, #8          \n"
                    "vmovl.s8   q5, d8          \n"// d10(r10-r13) d11(r14-r17)
                    
                    "vmlal.s16  q6, d10, d3[0]  \n"// sum(00-07) += (r10-r17) * k03
                    "vmlal.s16  q7, d11, d3[0]  \n"
                    "vmlal.s16  q8, d10, d3[1]  \n"// sum(10-17) += (r10-r17) * k13
                    "vmlal.s16  q9, d11, d3[1]  \n"     
                    "vmlal.s16  q10, d10, d3[2] \n"// sum(20-27) += (r10-r17) * k23
                    "vmlal.s16  q11, d11, d3[2] \n"
                    "vmlal.s16  q12, d10, d3[3] \n"// sum(30-37) += (r10-r17) * k33
                    "vmlal.s16  q13, d11, d3[3] \n"
                    
                    "vext.s8    q4, q4, #1      \n"// d8=r11-r18 q4
                    "vmovl.s8   q5, d8          \n"// d10(r11-r14) d11(r15-r18)

                    "vmlal.s16  q6, d10, d4[0]  \n"// sum(00-07) += (r11-r18) * k04
                    "vmlal.s16  q7, d11, d4[0]  \n"
                    "vmlal.s16  q8, d10, d4[1]  \n"// sum(10-17) += (r11-r18) * k14
                    "vmlal.s16  q9, d11, d4[1]  \n"     
                    "vmlal.s16  q10, d10, d4[2] \n"// sum(20-27) += (r11-r18) * k24
                    "vmlal.s16  q11, d11, d4[2] \n"
                    "vmlal.s16  q12, d10, d4[3] \n"// sum(30-37) += (r11-r18) * k34
                    "vmlal.s16  q13, d11, d4[3] \n"
                    
                    "vext.s8    q4, q4, #1      \n"// d8=r12-r19 q4
                    "vmovl.s8   q5, d8          \n"// d10(r12-r15) d11(r16-r19)

                    "vmlal.s16  q6, d10, d5[0]  \n"// sum(00-07) += (r12-r19) * k05
                    "vmlal.s16  q7, d11, d5[0]  \n"
                    "vmlal.s16  q8, d10, d5[1]  \n"// sum(10-17) += (r12-r19) * k15
                    "vmlal.s16  q9, d11, d5[1]  \n"     
                    "vmlal.s16  q10, d10, d5[2] \n"// sum(20-27) += (r12-r19) * k25
                    "vmlal.s16  q11, d11, d5[2] \n"
                    "vmlal.s16  q12, d10, d5[3] \n"// sum(30-37) += (r12-r19) * k35
                    "vmlal.s16  q13, d11, d5[3] \n"

                    // r2
                    "pld        [%7, #128]      \n"
                    "vld1.s8    {d8-d9}, [%7]   \n"// d8=r20-r27 d9=r28-r215 q4
                    "add        %7, #8          \n"
                    "vmovl.s8   q5, d8          \n"// d10(r20-r23) d11(r24-r27)

                    "vmlal.s16  q6, d10, d6[0]  \n"// sum(00-07) += (r20-r27) * k06
                    "vmlal.s16  q7, d11, d6[0]  \n"
                    "vmlal.s16  q8, d10, d6[1]  \n"// sum(10-17) += (r20-r27) * k16
                    "vmlal.s16  q9, d11, d6[1]  \n"     
                    "vmlal.s16  q10, d10, d6[2] \n"// sum(20-27) += (r20-r27) * k26
                    "vmlal.s16  q11, d11, d6[2] \n"
                    "vmlal.s16  q12, d10, d6[3] \n"// sum(30-37) += (r20-r27) * k36
                    "vmlal.s16  q13, d11, d6[3] \n"
                    
                    "vext.s8    q4, q4, #1      \n"// d8=r21-r28 q4
                    "vmovl.s8   q5, d8          \n"// d10(r21-r24) d11(r25-r28)

                    "vmlal.s16  q6, d10, d7[0]  \n"// sum(00-07) += (r21-r28) * k07
                    "vmlal.s16  q7, d11, d7[0]  \n"
                    "vmlal.s16  q8, d10, d7[1]  \n"// sum(10-17) += (r21-r28) * k17
                    "vmlal.s16  q9, d11, d7[1]  \n"     
                    "vmlal.s16  q10, d10, d7[2] \n"// sum(20-27) += (r21-r28) * k27
                    "vmlal.s16  q11, d11, d7[2] \n"
                    "vmlal.s16  q12, d10, d7[3] \n"// sum(30-37) += (r21-r28) * k37
                    "vmlal.s16  q13, d11, d7[3] \n"
                    
                    "vld1.s8    {d0}, [%8]      \n"// d0(k08-k38 xx-xx)
                    "add        %8, #4          \n"
                    "vmovl.s8   q0, d0          \n"// d0(k08-k38) d1(xx-xx)

                    "vext.s8    q4, q4, #1      \n"// d8=r22-r29 q4
                    "vmovl.s8   q5, d8          \n"// d10(r22-r25) d11(r26-r29)

                    "vmlal.s16  q6, d10, d0[0]  \n"// sum(00-07) += (r22-r29) * k08
                    "vmlal.s16  q7, d11, d0[0]  \n"
                    "vmlal.s16  q8, d10, d0[1]  \n"// sum(10-17) += (r22-r29) * k18
                    "vmlal.s16  q9, d11, d0[1]  \n"     
                    "vmlal.s16  q10, d10, d0[2] \n"// sum(20-27) += (r22-r29) * k28
                    "vmlal.s16  q11, d11, d0[2] \n"
                    "vmlal.s16  q12, d10, d0[3] \n"// sum(30-37) += (r22-r29) * k38
                    "vmlal.s16  q13, d11, d0[3] \n"
                    
                    "vst1.s32   {d12-d15}, [%1]! \n"// sum00-sum07 q6 q7
                    "vst1.s32   {d16-d19}, [%2]! \n"// sum10-sum17 q8 q9
                    "vst1.s32   {d20-d23}, [%3]! \n"// sum20-sum27 q10 q11
                    "vst1.s32   {d24-d27}, [%4]! \n"// sum30-sum37 q12 q13

                    "sub        %8, #36          \n"
                    "subs       %0, #1           \n"

                    "bne        0b               \n"

                    : "=r"(nn),         // %0
                      "=r"(outptr0),    // %1
                      "=r"(outptr1),    // %2
                      "=r"(outptr2),    // %3
                      "=r"(outptr3),    // %4
                      "=r"(r0),         // %5
                      "=r"(r1),         // %6
                      "=r"(r2),         // %7
                      "=r"(ktmp)        // %8
                    : "0"(nn),
                      "1"(outptr0),
                      "2"(outptr1),
                      "3"(outptr2),
                      "4"(outptr3),
                      "5"(r0),
                      "6"(r1),
                      "7"(r2),
                      "8"(ktmp)
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15" //q14 q15 not be used...
                );
                }
#endif
#if 0 //__ARM_NEON
                if (remain >= 4)
                {
                    remain -= 4;
                asm volatile(
                    "vld1.s8    {d0-d3}, [%7]!  \n"// d0=(k00-k30 k01-k31) d1=(k02-k32 k03-k33) d2=(k04-k34 k05-k35) d3=(k06-k36 k07-k37)
                    // r0
                    "vld1.s8    {d8}, [%4]      \n"// d8=r00-r07
                    "add        %4, #4          \n"
                    "vld1.s32   {d12-d13}, [%0] \n"// sum00-sum03 q6
                    "vld1.s32   {d16-d17}, [%1] \n"// sum10-sum13 q8
                    "vld1.s32   {d20-d21}, [%2] \n"// sum20-sum23 q10
                    "vld1.s32   {d24-d25}, [%3] \n"// sum30-sum33 q12
                    
                    "vmovl.s8   q3, d3          \n"// d6(k06-k36) d7(k07-k37) 
                    "vmovl.s8   q2, d2          \n"// d4(k04-k34) d5(k05-k35)
                    "vmovl.s8   q1, d1          \n"// d2(k02-k32) d3(k03-k33)
                    "vmovl.s8   q0, d0          \n"// d0(k00-k30) d1(k01-k31)
                    "vmovl.s8   q5, d8          \n"// d10(r00-r03)
                    
                    "vmlal.s16  q6, d10, d0[0]  \n"// sum(00-03) += (r00-r03) * k00
                    "vmlal.s16  q8, d10, d0[1]  \n"// sum(10-13) += (r00-r03) * k10
                    "vmlal.s16  q10, d10, d0[2] \n"// sum(20-23) += (r00-r03) * k20
                    "vmlal.s16  q12, d10, d0[3] \n"// sum(30-33) += (r00-r03) * k30

                    "vext.s8    d8, d8, #1      \n"// d8=r01-r08
                    "vmovl.s8   q5, d8          \n"// d10(r01-r04)
                    
                    "vmlal.s16  q6, d10, d1[0]  \n"// sum(00-03) += (r01-r04) * k01
                    "vmlal.s16  q8, d10, d1[1]  \n"// sum(10-13) += (r01-r04) * k11
                    "vmlal.s16  q10, d10, d1[2] \n"// sum(20-23) += (r01-r04) * k21
                    "vmlal.s16  q12, d10, d1[3] \n"// sum(30-33) += (r01-r04) * k31
                    
                    "vext.s8    d8, d8, #1      \n"// d8=r02-r09
                    "vmovl.s8   q5, d8          \n"// d10(r02-r05)
                    
                    "vmlal.s16  q6, d10, d2[0]  \n"// sum(00-03) += (r02-r05) * k02
                    "vmlal.s16  q8, d10, d2[1]  \n"// sum(10-13) += (r02-r05) * k12
                    "vmlal.s16  q10, d10, d2[2] \n"// sum(20-23) += (r02-r05) * k22
                    "vmlal.s16  q12, d10, d2[3] \n"// sum(30-33) += (r02-r05) * k32
                    
                    // r1
                    "vld1.s8    {d8}, [%5]      \n"// d8=r10-r17
                    "add        %5, #4          \n"
                    "vmovl.s8   q5, d8          \n"// d10(r10-r13)
                    
                    "vmlal.s16  q6, d10, d3[0]  \n"// sum(00-03) += (r10-r13) * k03
                    "vmlal.s16  q8, d10, d3[1]  \n"// sum(10-13) += (r10-r13) * k13
                    "vmlal.s16  q10, d10, d3[2] \n"// sum(20-23) += (r10-r13) * k23
                    "vmlal.s16  q12, d10, d3[3] \n"// sum(30-33) += (r10-r13) * k33
                    
                    "vext.s8    d8, d8, #1      \n"// d8=r11-r18
                    "vmovl.s8   q5, d8          \n"// d10(r11-r14)

                    "vmlal.s16  q6, d10, d4[0]  \n"// sum(00-03) += (r11-r14) * k04
                    "vmlal.s16  q8, d10, d4[1]  \n"// sum(10-13) += (r11-r14) * k14
                    "vmlal.s16  q10, d10, d4[2] \n"// sum(20-23) += (r11-r14) * k24
                    "vmlal.s16  q12, d10, d4[3] \n"// sum(30-33) += (r11-r14) * k34
                    
                    "vext.s8    d8, d8, #1      \n"// d8=r12-r19 q4
                    "vmovl.s8   q5, d8          \n"// d10(r12-r15)

                    "vmlal.s16  q6, d10, d5[0]  \n"// sum(00-03) += (r12-r15) * k05
                    "vmlal.s16  q8, d10, d5[1]  \n"// sum(10-13) += (r12-r15) * k15
                    "vmlal.s16  q10, d10, d5[2] \n"// sum(20-23) += (r12-r15) * k25
                    "vmlal.s16  q12, d10, d5[3] \n"// sum(30-33) += (r12-r15) * k35

                    // r2
                    "vld1.s8    {d8}, [%6]      \n"// d8=r20-r27
                    "add        %6, #4          \n"
                    "vmovl.s8   q5, d8          \n"// d10(r20-r23)

                    "vmlal.s16  q6, d10, d6[0]  \n"// sum(00-03) += (r20-r23) * k06
                    "vmlal.s16  q8, d10, d6[1]  \n"// sum(10-13) += (r20-r23) * k16
                    "vmlal.s16  q10, d10, d6[2] \n"// sum(20-23) += (r20-r23) * k26
                    "vmlal.s16  q12, d10, d6[3] \n"// sum(30-33) += (r20-r23) * k36
                    
                    "vext.s8    q4, q4, #1      \n"// d8=r21-r28 q4
                    "vmovl.s8   q5, d8          \n"// d10(r21-r24)

                    "vmlal.s16  q6, d10, d7[0]  \n"// sum(00-03) += (r21-r24) * k07
                    "vmlal.s16  q8, d10, d7[1]  \n"// sum(10-13) += (r21-r24) * k17
                    "vmlal.s16  q10, d10, d7[2] \n"// sum(20-23) += (r21-r24) * k27
                    "vmlal.s16  q12, d10, d7[3] \n"// sum(30-33) += (r21-r24) * k37
                    
                    "vld1.s8    {d0}, [%7]      \n"// d0(k08-k38 xx-xx)
                    "add        %7, #4          \n"
                    "vmovl.s8   q0, d0          \n"// d0(k08-k38) d1(xx-xx)

                    "vext.s8    d8, d8, #1      \n"// d8=r22-r25
                    "vmovl.s8   q5, d8          \n"// d10(r22-r25)

                    "vmlal.s16  q6, d10, d0[0]  \n"// sum(00-03) += (r22-r25) * k08
                    "vmlal.s16  q8, d10, d0[1]  \n"// sum(10-13) += (r22-r25) * k18
                    "vmlal.s16  q10, d10, d0[2] \n"// sum(20-23) += (r22-r25) * k28
                    "vmlal.s16  q12, d10, d0[3] \n"// sum(30-33) += (r22-r25) * k38
                    
                    "vst1.s32   {d12-d13}, [%0]! \n"// sum00-sum03 q6
                    "vst1.s32   {d16-d17}, [%1]! \n"// sum10-sum13 q8
                    "vst1.s32   {d20-d21}, [%2]! \n"// sum20-sum23 q10
                    "vst1.s32   {d24-d25}, [%3]! \n"// sum30-sum33 q12 

                    "sub        %7, #36          \n"

                    : "=r"(outptr0),    // %0
                      "=r"(outptr1),    // %1
                      "=r"(outptr2),    // %2
                      "=r"(outptr3),    // %3
                      "=r"(r0),         // %4
                      "=r"(r1),         // %5
                      "=r"(r2),         // %6
                      "=r"(ktmp)        // %7
                    : "0"(outptr0),
                      "1"(outptr1),
                      "2"(outptr2),
                      "3"(outptr3),
                      "4"(r0),
                      "5"(r1),
                      "6"(r2),
                      "7"(ktmp)
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13" //q14 q15 not be used...
                );
                }
#endif
                for (; remain>0; remain--)
                {
#if 0 //__ARM_NEON
                    asm volatile(
                        "vld1.s8    {d0[]}, [%4]!       \n"// d0(r00)
                        "vld1.s8    {d1[]}, [%4]!       \n"// d1(r01)

                        "vld1.s8    {d4-d7}, [%7]!      \n"// d4(k00-k30 k01-k31) d5(k02-k32 k03-k33) d6(k04-k34 k05-k35) d7(k06-k36 k07-k37)

                        "vsli.64    d0, d1, #32         \n"// d0(r00 r00 r00 r00 r01 r01 r01 r01)

                        "vld1.s8    {d2[]}, [%4]        \n"// d2(r02 r02 r02 r02 r02 r02 r02 r02)
                        "sub        %4, %4, #2          \n"
                        "vld1.s8    {d3[]}, [%5]!       \n"// d3(r10 r10 r10 r10 r10 r10 r10 r10)
                        
                        "vmovl.s8   q5, d7              \n"// d10(k06-k36) d11(k07-k37)
                        "vmovl.s8   q4, d6              \n"// d8(k04-k34) d9(k05-k35)
                        "vmovl.s8   q3, d5              \n"// d6(k02-k32) d7(k03-k33)
                        "vmovl.s8   q2, d4              \n"// d4(k00-k30) d5(k01-k31)
                        
                        "vmovl.s8   q0, d0              \n"// d0(r00 r00 r00 r00) d1(r01 r01 r01 r01)

                        "vsli.64    d2, d3, #32         \n"// d2(r02 r02 r02 r02 r10 r10 r10 r10)
                        
                        "vmull.s16  q8, d0, d4          \n"// (r00) * (k00-k30)
                        "vmull.s16  q9, d1, d5          \n"// (r01) * (k01-k31)
                        
                        "vmovl.s8   q10, d2             \n"// d20(r02 r02 r02 r02) d21(r10 r10 r10 r10)

                        "vld1.s8    {d0[]}, [%5]!       \n"// d0(r11 r11 r11 r11 r11 r11 r11 r11)
                        "vld1.s8    {d1[]}, [%5]        \n"// d1(r12 r12 r12 r12 r12 r12 r12 r12)
                        "sub        %5, %5, #2          \n"

                        "vsli.64    d0, d1, #32         \n"// d0(r11 r11 r11 r11 r12 r12 r12 r12)

                        "vmlal.s16  q8, d20, d6         \n"// (r02) * (k02-k32)
                        "vmlal.s16  q9, d21, d7         \n"// (r10) * (k03-k33)
                        
                        "vmovl.s8   q0, d0              \n"// d0(r11 r11 r11 r11 ) d1(r12 r12 r12 r12)

                        "vld1.s8    {d2[]}, [%6]!       \n"// d2(r20 r20 r20 r20 r20 r20 r20 r20)
                        "vld1.s8    {d3[]}, [%6]!       \n"// d3(r21 r21 r21 r21 r21 r21 r21 r21)

                        "vsli.64    d2, d3, #32         \n"// d2(r20 r20 r20 r20 r21 r21 r21 r21)

                        "vmlal.s16  q8, d0, d8          \n"// (r11) * (k04-k34)
                        "vmlal.s16  q9, d1, d9          \n"// (r12) * (k05-k35)     

                        "vmovl.s8   q2, d2              \n"// d4(r20 r20 r20 r20) d5(r21 r21 r21 r21)

                        "vld1.s8    {d0[]}, [%6]        \n"// d0(r22 r22 r22 r22 r22 r22 r22 r22)
                        "sub        %6, %6, #2          \n"
                        "veor       d1, d1, d1          \n"// d1 = 0

                        "vld1.s8    {d6}, [%7]          \n"// d6 = k08-k38 xxxx
                        "sub        %7, #32             \n"

                        "vsli.64    d0, d1, #32         \n"// d0(r22 r22 r22 r22 0 0 0 0)
                        "vmovl.s8   q4, d6              \n"// d8(k08-k38)
                        "vmovl.s8   q0, d0              \n"// d0(r22 r22 r22 r22) d1(0 0 0 0)

                        "vmlal.s16  q8, d4, d10         \n"// (r20) * (k06-k36)
                        "vmlal.s16  q9, d5, d11         \n"// (r21) * (k07-k37)

                        "vld1.s32   {d20[0]}, [%0]      \n"

                        "vmlal.s16  q8, d0, d8          \n"// (r22) * (k08-k38)

                        "vld1.s32   {d20[1]}, [%1]      \n"

                        "vadd.s32   q8, q8, q9          \n"

                        "vld1.s32   {d21[0]}, [%2]      \n"
                        "vld1.s32   {d21[1]}, [%3]      \n"

                        "vadd.s32   q10, q10, q8        \n"

                        "vst1.s32   {d20[0]}, [%0]!     \n"
                        "vst1.s32   {d20[1]}, [%1]!     \n"
                        "vst1.s32   {d21[0]}, [%2]!     \n"
                        "vst1.s32   {d21[1]}, [%3]!     \n"

                        : "=r"(outptr0),    // %0
                          "=r"(outptr1),    // %1
                          "=r"(outptr2),    // %2
                          "=r"(outptr3),    // %3
                          "=r"(r0),         // %4
                          "=r"(r1),         // %5
                          "=r"(r2),         // %6
                          "=r"(ktmp)        // %7
                        : "0"(outptr0),
                          "1"(outptr1),
                          "2"(outptr2),
                          "3"(outptr3),
                          "4"(r0),
                          "5"(r1),
                          "6"(r2),
                          "7"(ktmp)
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q8", "q9", "q10"
                    );
#else
                    int sum0 = 0;
                    int sum1 = 0;
                    int sum2 = 0;
                    int sum3 = 0;

                    sum0 += r0[0] * ktmp[0];
                    sum1 += r0[0] * ktmp[1];
                    sum2 += r0[0] * ktmp[2];
                    sum3 += r0[0] * ktmp[3];

                    sum0 += r0[1] * ktmp[4];
                    sum1 += r0[1] * ktmp[5];
                    sum2 += r0[1] * ktmp[6];
                    sum3 += r0[1] * ktmp[7];
                    ktmp += 8;

                    sum0 += r0[2] * ktmp[0];
                    sum1 += r0[2] * ktmp[1];
                    sum2 += r0[2] * ktmp[2];
                    sum3 += r0[2] * ktmp[3];

                    sum0 += r1[0] * ktmp[4];
                    sum1 += r1[0] * ktmp[5];
                    sum2 += r1[0] * ktmp[6];
                    sum3 += r1[0] * ktmp[7];
                    ktmp += 8;

                    sum0 += r1[1] * ktmp[0];
                    sum1 += r1[1] * ktmp[1];
                    sum2 += r1[1] * ktmp[2];
                    sum3 += r1[1] * ktmp[3];

                    sum0 += r1[2] * ktmp[4];
                    sum1 += r1[2] * ktmp[5];
                    sum2 += r1[2] * ktmp[6];
                    sum3 += r1[2] * ktmp[7];
                    ktmp += 8;

                    sum0 += r2[0] * ktmp[0];
                    sum1 += r2[0] * ktmp[1];
                    sum2 += r2[0] * ktmp[2];
                    sum3 += r2[0] * ktmp[3];

                    sum0 += r2[1] * ktmp[4];
                    sum1 += r2[1] * ktmp[5];
                    sum2 += r2[1] * ktmp[6];
                    sum3 += r2[1] * ktmp[7];
                    ktmp += 8;

                    sum0 += r2[2] * ktmp[0];
                    sum1 += r2[2] * ktmp[1];
                    sum2 += r2[2] * ktmp[2];
                    sum3 += r2[2] * ktmp[3];
                    ktmp += 8;

                    *outptr0 += sum0;
                    *outptr1 += sum1;
                    *outptr2 += sum2;
                    *outptr3 += sum3;

                    ktmp -= 8*5;

                    outptr0++;
                    outptr1++;
                    outptr2++;
                    outptr3++;
#endif
                    r0++;
                    r1++;
                    r2++;
                }

                r0 += 2;
                r1 += 2;
                r2 += 2;
            }

            ktmp += 4*9;
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=remain_outch_start; p<outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        out0.fill(0);

        const signed char* ktmp = _kernel.channel(p/4 + p%4);

        for (int q=0; q<inch; q++)
        {
            int* outptr0 = out0;
            int* outptr0n = outptr0 + outw;

            const signed char* img0 = bottom_blob.channel(q);

            const signed char* r0 = img0;
            const signed char* r1 = img0 + w;
            const signed char* r2 = img0 + w * 2;
            const signed char* r3 = img0 + w * 3;

            int i = 0;

#if 0 //__ARM_NEON
            int8x16_t _k0123456789x = vld1q_s8(ktmp);
            int16x8_t _k_s16 = vmovl_s8(vget_low_s8(_k0123456789x));
            int16x8_t _kn_s16 = vmovl_s8(vget_high_s8(_k0123456789x));

            int16x4_t _k0123 = vget_low_s16(_k_s16);
            int16x4_t _k4567 = vget_high_s16(_k_s16);
            int16x4_t _k8xxx = vget_low_s16(_kn_s16);
#endif // __ARM_NEON

            for (; i+1 < outh; i+=2)
            {
#if 0 //__ARM_NEON
                int nn = outw >> 3;
                int remain = outw & 7;
#else
                int remain = outw;
#endif // __ARM_NEON

#if 0 //__ARM_NEON
                for (; nn >0; nn--)
                {
                    // r0
                    int8x8_t _r0 = vld1_s8(r0);
                    int8x8_t _r0n = vld1_s8(r0+8);
                    int8x8_t _r01 = vext_s8(_r0, _r0n, 1);
                    int8x8_t _r02 = vext_s8(_r0, _r0n, 2);
                    int16x8_t _r0_s16 = vmovl_s8(_r0);   // r00 - r07
                    int16x8_t _r01_s16 = vmovl_s8(_r01); // r01 - r08 
                    int16x8_t _r02_s16 = vmovl_s8(_r02); // r02 - r09

                    int32x4_t _sum0 = vmull_lane_s16(vget_low_s16(_r0_s16), _k0123, 0); // (r00 - r07) * k00
                    int32x4_t _sum0n = vmull_lane_s16(vget_high_s16(_r0_s16), _k0123, 0);

                    int32x4_t _sum1 = vmull_lane_s16(vget_low_s16(_r01_s16), _k0123, 1); // (r01 - r08) * k01
                    int32x4_t _sum1n = vmull_lane_s16(vget_high_s16(_r01_s16), _k0123, 1);

                    int32x4_t _sum2 = vmull_lane_s16(vget_low_s16(_r02_s16), _k0123, 2); // (r02 - r09) * k02
                    int32x4_t _sum2n = vmull_lane_s16(vget_high_s16(_r02_s16), _k0123, 2);

                    // r1
                    int8x8_t _r1 = vld1_s8(r1);
                    int8x8_t _r1n = vld1_s8(r1+8);
                    int8x8_t _r11 = vext_s8(_r1, _r1n, 1);
                    int8x8_t _r12 = vext_s8(_r1, _r1n, 2);
                    int16x8_t _r1_s16 = vmovl_s8(_r1);   // r10 - r17
                    int16x8_t _r11_s16 = vmovl_s8(_r11); // r11 - r18
                    int16x8_t _r12_s16 = vmovl_s8(_r12); // r12 - r19

                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_r1_s16), _k0123, 3); // (r10 - r17) * k03
                    _sum0n = vmlal_lane_s16(_sum0n, vget_high_s16(_r1_s16), _k0123, 3);

                    _sum1 = vmlal_lane_s16(_sum1, vget_low_s16(_r11_s16), _k4567, 0); // (r11 - r18) * k04
                    _sum1n = vmlal_lane_s16(_sum1n, vget_high_s16(_r11_s16), _k4567, 0);

                    _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_r12_s16), _k4567, 1); // (r12 - r19) * k05
                    _sum2n = vmlal_lane_s16(_sum2n, vget_high_s16(_r12_s16), _k4567, 1); 

                    int32x4_t _sum4 = vmull_lane_s16(vget_low_s16(_r1_s16), _k0123, 0); // (r10 - r17) * k00
                    int32x4_t _sum4n = vmull_lane_s16(vget_high_s16(_r1_s16), _k0123, 0);

                    int32x4_t _sum5 = vmull_lane_s16(vget_low_s16(_r11_s16), _k0123, 1); // (r11 - r18) * k01
                    int32x4_t _sum5n = vmull_lane_s16(vget_high_s16(_r11_s16), _k0123, 1);

                    int32x4_t _sum6 = vmull_lane_s16(vget_low_s16(_r12_s16), _k0123, 2); // (r12 - r19) * k02
                    int32x4_t _sum6n = vmull_lane_s16(vget_high_s16(_r12_s16), _k0123, 2);

                    // r2
                    int8x8_t _r2 = vld1_s8(r2);
                    int8x8_t _r2n = vld1_s8(r2+8);
                    int8x8_t _r21 = vext_s8(_r2, _r2n, 1);
                    int8x8_t _r22 = vext_s8(_r2, _r2n, 2);
                    int16x8_t _r2_s16 = vmovl_s8(_r2);   // r20 - r27
                    int16x8_t _r21_s16 = vmovl_s8(_r21); // r21 - r28
                    int16x8_t _r22_s16 = vmovl_s8(_r22); // r22 - r29

                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_r2_s16), _k4567, 2); // (r20 - r27) * k06
                    _sum0n = vmlal_lane_s16(_sum0n, vget_high_s16(_r2_s16), _k4567, 2);

                    _sum1 = vmlal_lane_s16(_sum1, vget_low_s16(_r21_s16), _k4567, 3); // (r21 - r28) * k07
                    _sum1n = vmlal_lane_s16(_sum1n, vget_high_s16(_r21_s16), _k4567, 3);

                    _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_r22_s16), _k8xxx, 0); // (r22 - r29) * k08
                    _sum2n = vmlal_lane_s16(_sum2n, vget_high_s16(_r22_s16), _k8xxx, 0);

                    _sum4 = vmlal_lane_s16(_sum4, vget_low_s16(_r2_s16), _k0123, 3); // (r20 - r27) * k03
                    _sum4n = vmlal_lane_s16(_sum4n, vget_high_s16(_r2_s16), _k0123, 3);

                    _sum5 = vmlal_lane_s16(_sum5, vget_low_s16(_r21_s16), _k4567, 0); // (r21 - r28) * k04
                    _sum5n = vmlal_lane_s16(_sum5n, vget_high_s16(_r21_s16), _k4567, 0);

                    _sum6 = vmlal_lane_s16(_sum6, vget_low_s16(_r22_s16), _k4567, 1); // (r22 - r29) * k05
                    _sum6n = vmlal_lane_s16(_sum6n, vget_high_s16(_r22_s16), _k4567, 1);

                    // load output sum0 sum0n
                    int32x4_t _out00 = vld1q_s32(outptr0);
                    int32x4_t _out01 = vld1q_s32(outptr0+4);
                    int32x4_t _out10 = vld1q_s32(outptr0n);
                    int32x4_t _out11 = vld1q_s32(outptr0n+4);

                    // r3
                    int8x8_t _r3 = vld1_s8(r3);
                    int8x8_t _r3n = vld1_s8(r3+8);
                    int8x8_t _r31 = vext_s8(_r3, _r3n, 1);
                    int8x8_t _r32 = vext_s8(_r3, _r3n, 2);
                    int16x8_t _r3_s16 = vmovl_s8(_r3);   // r30 - r37
                    int16x8_t _r31_s16 = vmovl_s8(_r31); // r31 - r38
                    int16x8_t _r32_s16 = vmovl_s8(_r32); // r32 - r39

                    _sum0 = vaddq_s32(_sum0, _sum1);
                    _sum0n = vaddq_s32(_sum0n, _sum1n);
                    _sum2 = vaddq_s32(_sum2, _sum0);
                    _sum2n = vaddq_s32(_sum2n, _sum0n);

                    _out00 = vaddq_s32(_out00, _sum2);
                    _out01 = vaddq_s32(_out01, _sum2n);

                    vst1q_s32(outptr0, _out00);
                    vst1q_s32(outptr0+4, _out01);

                    _sum4 = vmlal_lane_s16(_sum4, vget_low_s16(_r3_s16), _k4567, 2); // (r30 - r37) * k06
                    _sum4n = vmlal_lane_s16(_sum4n, vget_high_s16(_r3_s16), _k4567, 2);

                    _sum5 = vmlal_lane_s16(_sum5, vget_low_s16(_r31_s16), _k4567, 3); // (r31 - r38) * k07
                    _sum5n = vmlal_lane_s16(_sum5n, vget_high_s16(_r31_s16), _k4567, 3);

                    _sum6 = vmlal_lane_s16(_sum6, vget_low_s16(_r32_s16), _k8xxx, 0); // (r32 - r39) * k08
                    _sum6n = vmlal_lane_s16(_sum6n, vget_high_s16(_r32_s16), _k8xxx, 0); 

                    _sum4 = vaddq_s32(_sum4, _sum5);
                    _sum4n = vaddq_s32(_sum4n, _sum5n);
                    _sum6 = vaddq_s32(_sum6, _sum4);
                    _sum6n = vaddq_s32(_sum6n, _sum4n);

                    _out10 = vaddq_s32(_out10, _sum6);
                    _out11 = vaddq_s32(_out11, _sum6n);

                    vst1q_s32(outptr0n, _out10);
                    vst1q_s32(outptr0n+4, _out11);

                    r0 += 8;
                    r1 += 8;
                    r2 += 8;
                    r3 += 8;
                    outptr0 += 8;
                    outptr0n += 8;
                }
#endif
#if 0 //__ARM_NEON
                if (remain >= 4)
                {
                    remain -= 4;

                    // r0
                    int8x8_t _r0 = vld1_s8(r0);
                    int8x8_t _r0n = vld1_s8(r0+8);
                    int8x8_t _r01 = vext_s8(_r0, _r0n, 1);
                    int8x8_t _r02 = vext_s8(_r0, _r0n, 2);
                    int16x8_t _r0_s16 = vmovl_s8(_r0);   // r00 - r07
                    int16x8_t _r01_s16 = vmovl_s8(_r01); // r01 - r08
                    int16x8_t _r02_s16 = vmovl_s8(_r02); // r02 - r09

                    int32x4_t _sum0 = vmull_lane_s16(vget_low_s16(_r0_s16), _k0123, 0); // (r00 - r07) * k00
                    int32x4_t _sum1 = vmull_lane_s16(vget_low_s16(_r01_s16), _k0123, 1); // (r01 - r08) * k01
                    int32x4_t _sum2 = vmull_lane_s16(vget_low_s16(_r02_s16), _k0123, 2); // (r02 - r09) * k02

                    // r1
                    int8x8_t _r1 = vld1_s8(r1);
                    int8x8_t _r1n = vld1_s8(r1+8);
                    int8x8_t _r11 = vext_s8(_r1, _r1n, 1);
                    int8x8_t _r12 = vext_s8(_r1, _r1n, 2);
                    int16x8_t _r1_s16 = vmovl_s8(_r1);   // r10 - r17
                    int16x8_t _r11_s16 = vmovl_s8(_r11); // r11 - r18
                    int16x8_t _r12_s16 = vmovl_s8(_r12); // r12 - r19

                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_r1_s16), _k0123, 3); // (r10 - r17) * k03
                    _sum1 = vmlal_lane_s16(_sum1, vget_low_s16(_r11_s16), _k4567, 0); // (r11 - r18) * k04
                    _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_r12_s16), _k4567, 1); // (r12 - r19) * k05

                    int32x4_t _sum4 = vmull_lane_s16(vget_low_s16(_r1_s16), _k0123, 0); // (r10 - r17) * k00
                    int32x4_t _sum5 = vmull_lane_s16(vget_low_s16(_r11_s16), _k0123, 1); // (r11 - r18) * k01
                    int32x4_t _sum6 = vmull_lane_s16(vget_low_s16(_r12_s16), _k0123, 2); // (r12 - r19) * k02

                    // r2
                    int8x8_t _r2 = vld1_s8(r2);
                    int8x8_t _r2n = vld1_s8(r2+8);
                    int8x8_t _r21 = vext_s8(_r2, _r2n, 1);
                    int8x8_t _r22 = vext_s8(_r2, _r2n, 2);
                    int16x8_t _r2_s16 = vmovl_s8(_r2);   // r20 - r27
                    int16x8_t _r21_s16 = vmovl_s8(_r21); // r21 - r28
                    int16x8_t _r22_s16 = vmovl_s8(_r22); // r22 - r29

                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_r2_s16), _k4567, 2); // (r20 - r27) * k06
                    _sum1 = vmlal_lane_s16(_sum1, vget_low_s16(_r21_s16), _k4567, 3); // (r21 - r28) * k07
                    _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_r22_s16), _k8xxx, 0); // (r22 - r29) * k08

                    _sum4 = vmlal_lane_s16(_sum4, vget_low_s16(_r2_s16), _k0123, 3); // (r20 - r27) * k03
                    _sum5 = vmlal_lane_s16(_sum5, vget_low_s16(_r21_s16), _k4567, 0); // (r21 - r28) * k04
                    _sum6 = vmlal_lane_s16(_sum6, vget_low_s16(_r22_s16), _k4567, 1); // (r22 - r29) * k05

                    // load output sum0 sum0n
                    int32x4_t _out00 = vld1q_s32(outptr0);
                    int32x4_t _out10 = vld1q_s32(outptr0n);

                    // r3
                    int8x8_t _r3 = vld1_s8(r3);
                    int8x8_t _r3n = vld1_s8(r3+8);
                    int8x8_t _r31 = vext_s8(_r3, _r3n, 1);
                    int8x8_t _r32 = vext_s8(_r3, _r3n, 2);
                    int16x8_t _r3_s16 = vmovl_s8(_r3);   // r30 - r37
                    int16x8_t _r31_s16 = vmovl_s8(_r31); // r31 - r38
                    int16x8_t _r32_s16 = vmovl_s8(_r32); // r32 - r39

                    _sum0 = vaddq_s32(_sum0, _sum1);
                    _sum2 = vaddq_s32(_sum2, _sum0);
                    _out00 = vaddq_s32(_out00, _sum2);

                    vst1q_s32(outptr0, _out00);

                    _sum4 = vmlal_lane_s16(_sum4, vget_low_s16(_r3_s16), _k4567, 2); // (r30 - r37) * k06
                    _sum5 = vmlal_lane_s16(_sum5, vget_low_s16(_r31_s16), _k4567, 3); // (r31 - r38) * k07
                    _sum6 = vmlal_lane_s16(_sum6, vget_low_s16(_r32_s16), _k8xxx, 0); // (r32 - r39) * k08

                    _sum4 = vaddq_s32(_sum4, _sum5);
                    _sum6 = vaddq_s32(_sum6, _sum4);

                    _out10 = vaddq_s32(_out10, _sum6);

                    vst1q_s32(outptr0n, _out10);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    outptr0 += 4;
                    outptr0n += 4;
                }
#endif
                for (; remain>0; remain--)
                {
#if 0 //__ARM_NEON
                    asm volatile(
                        "vld1.s8    {d0[0]}, [%2]!  \n"
                        "vld1.s8    {d0[1]}, [%2]!  \n"
                        "vld1.s8    {d0[2]}, [%2]   \n"
                        "sub        %2, #2          \n"

                        "vld1.s8    {d0[3]}, [%3]!  \n"
                        "vld1.s8    {d0[4]}, [%3]!  \n"
                        "vld1.s8    {d0[5]}, [%3]   \n"
                        "sub        %3, #2          \n"

                        "vld1.s8    {d0[6]}, [%4]!  \n"
                        "vld1.s8    {d0[7]}, [%4]!  \n"// d0(r00 r01 r02 r10 r11 r12 r20 r21)

                        "vld1.s8    {d4[]}, [%4]    \n"// d4(r22 r22 r22 r22 r22 r22 r22 r22) 
                        "sub        %4, #2          \n"

                        "vext.s8    d1, d0, d4, #3  \n"// d1(r10 r11 r12 r22 r21 r22 r22 r22)

                        "vld1.s8    {d1[6]}, [%5]!  \n"
                        "vld1.s8    {d1[7]}, [%5]!  \n"// d1(r10 r11 r12 r22 r21 r22 r30 r31)

                        "vld1.s8    {d2}, [%6]!     \n"// d2(k00 k01 k02 k10 k11 k12 k20 k21)

                        "vld1.s8    {d5[]}, [%5]    \n"// d5(r32 r32 r32 r32 r32 r32 r32 r32)
                        "sub        %5, #2          \n"

                        "veor       d3, d1, d1      \n"// d3(00 00 00 00 00 00 00 00)

                        "vmull.s8   q8, d0, d2      \n"// sum0 = (r00 - r21) * (k00 - k21)
                        "vmull.s8   q9, d1, d2      \n"// sum1 = (r10 - r31) * (k00 - k21)

                        "vld1.s8    {d3[0]}, [%6]   \n"// d3(k22 00 00 00 00 00 00 00)
                        "sub        %6, #8          \n"

                        "vmull.s8   q10, d4, d3     \n"// r22 * k22
                        "vmull.s8   q11, d5, d3     \n"// r22 * k22

                        "vld1.s32   {d6[0]}, [%0]   \n"

                        "vaddl.s16  q10, d16, d18   \n"
                        "vaddl.s16  q11, d18, d22   \n"
                        "vaddw.s16  q10, q10, d17   \n"
                        "vaddw.s16  q11, q11, d19   \n"

                        "vld1.s32   {d6[1]}, [%1]   \n"

                        "vpadd.s32  d20, d20, d21   \n"
                        "vpadd.s32  d22, d22, d23   \n"
                        "vpadd.s32  d20, d20, d22   \n"
                        "vadd.s32   d6, d6, d20     \n"

                        "vst1.s32   {d6[0]}, [%0]!  \n"
                        "vst1.s32   {d6[1]}, [%1]!  \n"

                        : "=r"(outptr0),    // %0
                          "=r"(outptr0n),   // %1
                          "=r"(r0),         // %2
                          "=r"(r1),         // %3
                          "=r"(r2),         // %4
                          "=r"(r3),         // %5
                          "=r"(ktmp)        // %6
                        : "0"(outptr0),
                          "1"(outptr0n),
                          "2"(r0),
                          "3"(r1),
                          "4"(r2),
                          "5"(r3),
                          "6"(ktmp)
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11"
                    );
#else
                    int sum0 = 0;
                    int sum0n = 0;

                    sum0 += r0[0] * ktmp[0];
                    sum0 += r0[1] * ktmp[1];
                    sum0 += r0[2] * ktmp[2];
                    sum0 += r1[0] * ktmp[3];
                    sum0 += r1[1] * ktmp[4];
                    sum0 += r1[2] * ktmp[5];
                    sum0 += r2[0] * ktmp[6];
                    sum0 += r2[1] * ktmp[7];
                    sum0 += r2[2] * ktmp[8];

                    sum0n += r1[0] * ktmp[0];
                    sum0n += r1[1] * ktmp[1];
                    sum0n += r1[2] * ktmp[2];
                    sum0n += r2[0] * ktmp[3];
                    sum0n += r2[1] * ktmp[4];
                    sum0n += r2[2] * ktmp[5];
                    sum0n += r3[0] * ktmp[6];
                    sum0n += r3[1] * ktmp[7];
                    sum0n += r3[2] * ktmp[8];

                    *outptr0 += sum0;
                    *outptr0n += sum0n;

                    outptr0++;
                    outptr0n++;
#endif
                    r0++;
                    r1++;
                    r2++;
                    r3++;
                }

                r0 += 2 + w;
                r1 += 2 + w;
                r2 += 2 + w;
                r3 += 2 + w;

                outptr0 += outw;
                outptr0n += outw;
            }

            for (; i < outh; i++)
            {
#if 0 //__ARM_NEON
                int nn = outw >> 3;
                int remain = outw & 7;
#else
                int remain = outw;
#endif // __ARM_NEON

#if 0 //__ARM_NEON
                for (; nn >0; nn--)
                {
                    // r0
                    int8x8_t _r0 = vld1_s8(r0);
                    int8x8_t _r0n = vld1_s8(r0+8);
                    int8x8_t _r01 = vext_s8(_r0, _r0n, 1);
                    int8x8_t _r02 = vext_s8(_r0, _r0n, 2);
                    int16x8_t _r0_s16 = vmovl_s8(_r0);   // r00 - r07
                    int16x8_t _r01_s16 = vmovl_s8(_r01); // r01 - r08 
                    int16x8_t _r02_s16 = vmovl_s8(_r02); // r02 - r09

                    int32x4_t _sum0 = vmull_lane_s16(vget_low_s16(_r0_s16), _k0123, 0); // (r00 - r07) * k00
                    int32x4_t _sum0n = vmull_lane_s16(vget_high_s16(_r0_s16), _k0123, 0);

                    int32x4_t _sum1 = vmull_lane_s16(vget_low_s16(_r01_s16), _k0123, 1); // (r01 - r08) * k01
                    int32x4_t _sum1n = vmull_lane_s16(vget_high_s16(_r01_s16), _k0123, 1);

                    int32x4_t _sum2 = vmull_lane_s16(vget_low_s16(_r02_s16), _k0123, 2); // (r02 - r09) * k02
                    int32x4_t _sum2n = vmull_lane_s16(vget_high_s16(_r02_s16), _k0123, 2);

                    // r1
                    int8x8_t _r1 = vld1_s8(r1);
                    int8x8_t _r1n = vld1_s8(r1+8);
                    int8x8_t _r11 = vext_s8(_r1, _r1n, 1);
                    int8x8_t _r12 = vext_s8(_r1, _r1n, 2);
                    int16x8_t _r1_s16 = vmovl_s8(_r1);   // r10 - r17
                    int16x8_t _r11_s16 = vmovl_s8(_r11); // r11 - r18
                    int16x8_t _r12_s16 = vmovl_s8(_r12); // r12 - r19

                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_r1_s16), _k0123, 3); // (r10 - r17) * k03
                    _sum0n = vmlal_lane_s16(_sum0n, vget_high_s16(_r1_s16), _k0123, 3);

                    _sum1 = vmlal_lane_s16(_sum1, vget_low_s16(_r11_s16), _k4567, 0); // (r11 - r18) * k04
                    _sum1n = vmlal_lane_s16(_sum1n, vget_high_s16(_r11_s16), _k4567, 0);

                    _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_r12_s16), _k4567, 1); // (r12 - r19) * k05
                    _sum2n = vmlal_lane_s16(_sum2n, vget_high_s16(_r12_s16), _k4567, 1);

                    // r2
                    int8x8_t _r2 = vld1_s8(r2);
                    int8x8_t _r2n = vld1_s8(r2+8);
                    int8x8_t _r21 = vext_s8(_r2, _r2n, 1);
                    int8x8_t _r22 = vext_s8(_r2, _r2n, 2);
                    int16x8_t _r2_s16 = vmovl_s8(_r2);   // r20 - r27
                    int16x8_t _r21_s16 = vmovl_s8(_r21); // r21 - r28
                    int16x8_t _r22_s16 = vmovl_s8(_r22); // r22 - r29

                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_r2_s16), _k4567, 2); // (r20 - r27) * k06
                    _sum0n = vmlal_lane_s16(_sum0n, vget_high_s16(_r2_s16), _k4567, 2);

                    _sum1 = vmlal_lane_s16(_sum1, vget_low_s16(_r21_s16), _k4567, 3); // (r21 - r28) * k07
                    _sum1n = vmlal_lane_s16(_sum1n, vget_high_s16(_r21_s16), _k4567, 3);

                    _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_r22_s16), _k8xxx, 0); // (r22 - r29) * k08
                    _sum2n = vmlal_lane_s16(_sum2n, vget_high_s16(_r22_s16), _k8xxx, 0); 

                    // load output sum0 sum0n
                    int32x4_t _out00 = vld1q_s32(outptr0);
                    int32x4_t _out01 = vld1q_s32(outptr0+4);

                    _sum0 = vaddq_s32(_sum0, _sum1);
                    _sum0n = vaddq_s32(_sum0n, _sum1n);
                    _sum2 = vaddq_s32(_sum2, _sum0);
                    _sum2n = vaddq_s32(_sum2n, _sum0n);

                    _out00 = vaddq_s32(_out00, _sum2);
                    _out01 = vaddq_s32(_out01, _sum2n);

                    vst1q_s32(outptr0, _out00);
                    vst1q_s32(outptr0+4, _out01);

                    r0 += 8;
                    r1 += 8;
                    r2 += 8;
                    r3 += 8;
                    outptr0 += 8;
                    outptr0n += 8;
                }
#endif
                for (; remain>0; remain--)
                {
                    int sum0 = 0;

                    sum0 += r0[0] * ktmp[0];
                    sum0 += r0[1] * ktmp[1];
                    sum0 += r0[2] * ktmp[2];
                    sum0 += r1[0] * ktmp[3];
                    sum0 += r1[1] * ktmp[4];
                    sum0 += r1[2] * ktmp[5];
                    sum0 += r2[0] * ktmp[6];
                    sum0 += r2[1] * ktmp[7];
                    sum0 += r2[2] * ktmp[8];

                    *outptr0 += sum0;

                    r0++;
                    r1++;
                    r2++;
                    outptr0++;
                }

                r0 += 2;
                r1 += 2;
                r2 += 2;
            }

            ktmp += 9;
        }
    }
}

static void conv3x3s2_packed_int8_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2*outw + w;

    int nn_outch = outch >> 3;
    int remain_outch_start = nn_outch << 3;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp=0; pp<nn_outch; pp++)
    {
        int p = pp * 8;

        Mat out0 = top_blob.channel(p+0);
        Mat out1 = top_blob.channel(p+1);
        Mat out2 = top_blob.channel(p+2);
        Mat out3 = top_blob.channel(p+3);
        Mat out4 = top_blob.channel(p+4);
        Mat out5 = top_blob.channel(p+5);
        Mat out6 = top_blob.channel(p+6);
        Mat out7 = top_blob.channel(p+7);

        out0.fill(0);
        out1.fill(0);
        out2.fill(0);
        out3.fill(0);
        out4.fill(0);
        out5.fill(0);
        out6.fill(0);
        out7.fill(0);

        const signed char* ktmp = _kernel.channel(p/8);

        for (int q=0; q<inch; q++)
        {
            int* outptr0 = out0;
            int* outptr1 = out1;
            int* outptr2 = out2;
            int* outptr3 = out3;
            int* outptr4 = out4;
            int* outptr5 = out5;
            int* outptr6 = out6;
            int* outptr7 = out7;

            const signed char* img0 = bottom_blob.channel(q);

            const signed char* r0 = img0;
            const signed char* r1 = img0 + w;
            const signed char* r2 = img0 + w*2;

            int i = 0;

            for (; i < outh; i++)
            {
#if __ARM_NEON
                int nn = outw >> 2;
                int remain = outw & 3;
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
                for (; nn>0; nn--)
                {                  
                    // load output ch 0-7
                    int32x4_t _sum0 = vld1q_s32(outptr0);// out0
                    int32x4_t _sum1 = vld1q_s32(outptr1);// out1
                    int32x4_t _sum2 = vld1q_s32(outptr2);// out2
                    int32x4_t _sum3 = vld1q_s32(outptr3);// out3
                    int32x4_t _sum4 = vld1q_s32(outptr4);// out4
                    int32x4_t _sum5 = vld1q_s32(outptr5);// out5
                    int32x4_t _sum6 = vld1q_s32(outptr6);// out6
                    int32x4_t _sum7 = vld1q_s32(outptr7);// out7

                    // r0
                    int8x8x2_t _r0_s8 = vld2_s8(r0);
                    int8x8_t _r2_s8 = vext_s8(_r0_s8.val[0], _r0_s8.val[0], 1);
                    // k0 - k2
                    int8x8_t _k0_8 = vld1_s8(ktmp);    //(k00-k70)
                    int8x8_t _k1_8 = vld1_s8(ktmp+8);  //(k01-k71)
                    int8x8_t _k2_8 = vld1_s8(ktmp+16); //(k02-k72)

                    int16x8_t _r0 = vmovl_s8(_r0_s8.val[0]);
                    int16x8_t _r1 = vmovl_s8(_r0_s8.val[1]);
                    int16x8_t _r2 = vmovl_s8(_r2_s8);

                    int16x8_t _k0 = vmovl_s8(_k0_8);
                    int16x8_t _k1 = vmovl_s8(_k1_8);
                    int16x8_t _k2 = vmovl_s8(_k2_8);
                    // dot row 1 k0
                    _sum0 = vmlal_laneq_s16(_sum0, vget_low_s16(_r0), _k0, 0);
                    _sum1 = vmlal_laneq_s16(_sum1, vget_low_s16(_r0), _k0, 1);
                    _sum2 = vmlal_laneq_s16(_sum2, vget_low_s16(_r0), _k0, 2);
                    _sum3 = vmlal_laneq_s16(_sum3, vget_low_s16(_r0), _k0, 3);
                    _sum4 = vmlal_laneq_s16(_sum4, vget_low_s16(_r0), _k0, 4);
                    _sum5 = vmlal_laneq_s16(_sum5, vget_low_s16(_r0), _k0, 5);
                    _sum6 = vmlal_laneq_s16(_sum6, vget_low_s16(_r0), _k0, 6);
                    _sum7 = vmlal_laneq_s16(_sum7, vget_low_s16(_r0), _k0, 7);
                    // dot row 1 k1
                    _sum0 = vmlal_laneq_s16(_sum0, vget_low_s16(_r1), _k1, 0);
                    _sum1 = vmlal_laneq_s16(_sum1, vget_low_s16(_r1), _k1, 1);
                    _sum2 = vmlal_laneq_s16(_sum2, vget_low_s16(_r1), _k1, 2);
                    _sum3 = vmlal_laneq_s16(_sum3, vget_low_s16(_r1), _k1, 3);
                    _sum4 = vmlal_laneq_s16(_sum4, vget_low_s16(_r1), _k1, 4);
                    _sum5 = vmlal_laneq_s16(_sum5, vget_low_s16(_r1), _k1, 5);
                    _sum6 = vmlal_laneq_s16(_sum6, vget_low_s16(_r1), _k1, 6);
                    _sum7 = vmlal_laneq_s16(_sum7, vget_low_s16(_r1), _k1, 7);
                    // dot row 1 k2
                    _sum0 = vmlal_laneq_s16(_sum0, vget_low_s16(_r2), _k2, 0);
                    _sum1 = vmlal_laneq_s16(_sum1, vget_low_s16(_r2), _k2, 1);
                    _sum2 = vmlal_laneq_s16(_sum2, vget_low_s16(_r2), _k2, 2);
                    _sum3 = vmlal_laneq_s16(_sum3, vget_low_s16(_r2), _k2, 3);
                    _sum4 = vmlal_laneq_s16(_sum4, vget_low_s16(_r2), _k2, 4);
                    _sum5 = vmlal_laneq_s16(_sum5, vget_low_s16(_r2), _k2, 5);
                    _sum6 = vmlal_laneq_s16(_sum6, vget_low_s16(_r2), _k2, 6);
                    _sum7 = vmlal_laneq_s16(_sum7, vget_low_s16(_r2), _k2, 7);

                    // r1
                    _r0_s8 = vld2_s8(r1);
                    _r2_s8 = vext_s8(_r0_s8.val[0], _r0_s8.val[0], 1);
                    // k3 - k5
                    _k0_8 = vld1_s8(ktmp+24);    //(k03-k73)
                    _k1_8 = vld1_s8(ktmp+32);  //(k04-k74)
                    _k2_8 = vld1_s8(ktmp+40); //(k05-k75)

                    _r0 = vmovl_s8(_r0_s8.val[0]);
                    _r1 = vmovl_s8(_r0_s8.val[1]);
                    _r2 = vmovl_s8(_r2_s8);

                    _k0 = vmovl_s8(_k0_8);
                    _k1 = vmovl_s8(_k1_8);
                    _k2 = vmovl_s8(_k2_8);
                    // dot row 2 k3
                    _sum0 = vmlal_laneq_s16(_sum0, vget_low_s16(_r0), _k0, 0);
                    _sum1 = vmlal_laneq_s16(_sum1, vget_low_s16(_r0), _k0, 1);
                    _sum2 = vmlal_laneq_s16(_sum2, vget_low_s16(_r0), _k0, 2);
                    _sum3 = vmlal_laneq_s16(_sum3, vget_low_s16(_r0), _k0, 3);
                    _sum4 = vmlal_laneq_s16(_sum4, vget_low_s16(_r0), _k0, 4);
                    _sum5 = vmlal_laneq_s16(_sum5, vget_low_s16(_r0), _k0, 5);
                    _sum6 = vmlal_laneq_s16(_sum6, vget_low_s16(_r0), _k0, 6);
                    _sum7 = vmlal_laneq_s16(_sum7, vget_low_s16(_r0), _k0, 7);
                    // dot row 2 k4
                    _sum0 = vmlal_laneq_s16(_sum0, vget_low_s16(_r1), _k1, 0);
                    _sum1 = vmlal_laneq_s16(_sum1, vget_low_s16(_r1), _k1, 1);
                    _sum2 = vmlal_laneq_s16(_sum2, vget_low_s16(_r1), _k1, 2);
                    _sum3 = vmlal_laneq_s16(_sum3, vget_low_s16(_r1), _k1, 3);
                    _sum4 = vmlal_laneq_s16(_sum4, vget_low_s16(_r1), _k1, 4);
                    _sum5 = vmlal_laneq_s16(_sum5, vget_low_s16(_r1), _k1, 5);
                    _sum6 = vmlal_laneq_s16(_sum6, vget_low_s16(_r1), _k1, 6);
                    _sum7 = vmlal_laneq_s16(_sum7, vget_low_s16(_r1), _k1, 7);
                    // dot row 2 k5
                    _sum0 = vmlal_laneq_s16(_sum0, vget_low_s16(_r2), _k2, 0);
                    _sum1 = vmlal_laneq_s16(_sum1, vget_low_s16(_r2), _k2, 1);
                    _sum2 = vmlal_laneq_s16(_sum2, vget_low_s16(_r2), _k2, 2);
                    _sum3 = vmlal_laneq_s16(_sum3, vget_low_s16(_r2), _k2, 3);
                    _sum4 = vmlal_laneq_s16(_sum4, vget_low_s16(_r2), _k2, 4);
                    _sum5 = vmlal_laneq_s16(_sum5, vget_low_s16(_r2), _k2, 5);
                    _sum6 = vmlal_laneq_s16(_sum6, vget_low_s16(_r2), _k2, 6);
                    _sum7 = vmlal_laneq_s16(_sum7, vget_low_s16(_r2), _k2, 7);

                    // r2 
                    _r0_s8 = vld2_s8(r2);
                    _r2_s8 = vext_s8(_r0_s8.val[0], _r0_s8.val[0], 1);
                    // k6 - k8
                    _k0_8 = vld1_s8(ktmp+48); //(k06-k76)
                    _k1_8 = vld1_s8(ktmp+56); //(k07-k77)
                    _k2_8 = vld1_s8(ktmp+64); //(k08-k78)

                    _r0 = vmovl_s8(_r0_s8.val[0]);
                    _r1 = vmovl_s8(_r0_s8.val[1]);
                    _r2 = vmovl_s8(_r2_s8);

                    _k0 = vmovl_s8(_k0_8);
                    _k1 = vmovl_s8(_k1_8);
                    _k2 = vmovl_s8(_k2_8);
                    // dot row 2 k6
                    _sum0 = vmlal_laneq_s16(_sum0, vget_low_s16(_r0), _k0, 0);
                    _sum1 = vmlal_laneq_s16(_sum1, vget_low_s16(_r0), _k0, 1);
                    _sum2 = vmlal_laneq_s16(_sum2, vget_low_s16(_r0), _k0, 2);
                    _sum3 = vmlal_laneq_s16(_sum3, vget_low_s16(_r0), _k0, 3);
                    _sum4 = vmlal_laneq_s16(_sum4, vget_low_s16(_r0), _k0, 4);
                    _sum5 = vmlal_laneq_s16(_sum5, vget_low_s16(_r0), _k0, 5);
                    _sum6 = vmlal_laneq_s16(_sum6, vget_low_s16(_r0), _k0, 6);
                    _sum7 = vmlal_laneq_s16(_sum7, vget_low_s16(_r0), _k0, 7);
                    // dot row 2 k7
                    _sum0 = vmlal_laneq_s16(_sum0, vget_low_s16(_r1), _k1, 0);
                    _sum1 = vmlal_laneq_s16(_sum1, vget_low_s16(_r1), _k1, 1);
                    _sum2 = vmlal_laneq_s16(_sum2, vget_low_s16(_r1), _k1, 2);
                    _sum3 = vmlal_laneq_s16(_sum3, vget_low_s16(_r1), _k1, 3);
                    _sum4 = vmlal_laneq_s16(_sum4, vget_low_s16(_r1), _k1, 4);
                    _sum5 = vmlal_laneq_s16(_sum5, vget_low_s16(_r1), _k1, 5);
                    _sum6 = vmlal_laneq_s16(_sum6, vget_low_s16(_r1), _k1, 6);
                    _sum7 = vmlal_laneq_s16(_sum7, vget_low_s16(_r1), _k1, 7);
                    // dot row 2 k8
                    _sum0 = vmlal_laneq_s16(_sum0, vget_low_s16(_r2), _k2, 0);
                    _sum1 = vmlal_laneq_s16(_sum1, vget_low_s16(_r2), _k2, 1);
                    _sum2 = vmlal_laneq_s16(_sum2, vget_low_s16(_r2), _k2, 2);
                    _sum3 = vmlal_laneq_s16(_sum3, vget_low_s16(_r2), _k2, 3);
                    _sum4 = vmlal_laneq_s16(_sum4, vget_low_s16(_r2), _k2, 4);
                    _sum5 = vmlal_laneq_s16(_sum5, vget_low_s16(_r2), _k2, 5);
                    _sum6 = vmlal_laneq_s16(_sum6, vget_low_s16(_r2), _k2, 6);
                    _sum7 = vmlal_laneq_s16(_sum7, vget_low_s16(_r2), _k2, 7);

                    // save s32 to memory
                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr1, _sum1);
                    vst1q_s32(outptr2, _sum2);
                    vst1q_s32(outptr3, _sum3);
                    vst1q_s32(outptr4, _sum4);
                    vst1q_s32(outptr5, _sum5);
                    vst1q_s32(outptr6, _sum6);
                    vst1q_s32(outptr7, _sum7);
               
                    r0 += 8;
                    r1 += 8;
                    r2 += 8;

                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                    outptr4 += 4;
                    outptr5 += 4;
                    outptr6 += 4;
                    outptr7 += 4;
                }
                
#endif // __ARM_NEON
                for (; remain>0; remain--)
                {
#if __ARM_NEON
                    int8x8_t _r0_s8 = vld1_s8(r0);// (a00 a01 a02 ....)
                    int8x8_t _r1_s8 = vld1_s8(r1);// (a10 a11 a12 ....)
                    int8x8_t _r2_s8 = vld1_s8(r2);// (a20 a21 a22 ....)

                    int16x8_t _r0 = vmovl_s8(_r0_s8);
                    int16x8_t _r1 = vmovl_s8(_r1_s8);
                    int16x8_t _r2 = vmovl_s8(_r2_s8);

                    int32x4_t _sum03, _sum47;
                    _sum03 = vld1q_lane_s32(outptr0, _sum03, 0);// out0
                    _sum03 = vld1q_lane_s32(outptr1, _sum03, 1);// out1
                    _sum03 = vld1q_lane_s32(outptr2, _sum03, 2);// out2
                    _sum03 = vld1q_lane_s32(outptr3, _sum03, 3);// out3
                    _sum47 = vld1q_lane_s32(outptr4, _sum47, 0);// out4
                    _sum47 = vld1q_lane_s32(outptr5, _sum47, 1);// out5
                    _sum47 = vld1q_lane_s32(outptr6, _sum47, 2);// out6
                    _sum47 = vld1q_lane_s32(outptr7, _sum47, 3);// out7

                    // k0 - k2
                    int8x8_t _k0_8 = vld1_s8(ktmp);    //(k00-k70)
                    int8x8_t _k1_8 = vld1_s8(ktmp+8);  //(k01-k71)
                    int8x8_t _k2_8 = vld1_s8(ktmp+16); //(k02-k72)

                    int16x8_t _k0 = vmovl_s8(_k0_8);
                    int16x8_t _k1 = vmovl_s8(_k1_8);
                    int16x8_t _k2 = vmovl_s8(_k2_8);

                    int32x4_t _sum0 = vmull_laneq_s16(vget_low_s16(_k0), _r0, 0);
                    int32x4_t _sum0n = vmull_laneq_s16(vget_high_s16(_k0), _r0, 0);
                    int32x4_t _sum1 = vmull_laneq_s16(vget_low_s16(_k1), _r0, 1);
                    int32x4_t _sum1n = vmull_laneq_s16(vget_high_s16(_k1), _r0, 1);
                    _sum03 = vmlal_laneq_s16(_sum03, vget_low_s16(_k2), _r0, 2);
                    _sum47 = vmlal_laneq_s16(_sum47, vget_high_s16(_k2), _r0, 2);

                    // k3 - k5
                    _k0_8 = vld1_s8(ktmp+24); //(k03-k73)
                    _k1_8 = vld1_s8(ktmp+32); //(k04-k74)
                    _k2_8 = vld1_s8(ktmp+40); //(k05-k75)

                    _k0 = vmovl_s8(_k0_8);
                    _k1 = vmovl_s8(_k1_8);
                    _k2 = vmovl_s8(_k2_8);

                    _sum0 = vmlal_laneq_s16(_sum0, vget_low_s16(_k0), _r1, 0);
                    _sum0n = vmlal_laneq_s16(_sum0n, vget_high_s16(_k0), _r1, 0);
                    _sum1 = vmlal_laneq_s16(_sum1, vget_low_s16(_k1), _r1, 1);
                    _sum1n = vmlal_laneq_s16(_sum1n, vget_high_s16(_k1), _r1, 1);
                    _sum03 = vmlal_laneq_s16(_sum03, vget_low_s16(_k2), _r1, 2);
                    _sum47 = vmlal_laneq_s16(_sum47, vget_high_s16(_k2), _r1, 2);

                    // k6 - k8
                    _k0_8 = vld1_s8(ktmp+48); //(k06-k76)
                    _k1_8 = vld1_s8(ktmp+56); //(k07-k77)
                    _k2_8 = vld1_s8(ktmp+64); //(k08-k78)

                    _k0 = vmovl_s8(_k0_8);
                    _k1 = vmovl_s8(_k1_8);
                    _k2 = vmovl_s8(_k2_8);

                    _sum0 = vmlal_laneq_s16(_sum0, vget_low_s16(_k0), _r2, 0);
                    _sum0n = vmlal_laneq_s16(_sum0n, vget_high_s16(_k0), _r2, 0);
                    _sum1 = vmlal_laneq_s16(_sum1, vget_low_s16(_k1), _r2, 1);
                    _sum1n = vmlal_laneq_s16(_sum1n, vget_high_s16(_k1), _r2, 1);
                    _sum03 = vmlal_laneq_s16(_sum03, vget_low_s16(_k2), _r2, 2);
                    _sum47 = vmlal_laneq_s16(_sum47, vget_high_s16(_k2), _r2, 2);

                    _sum0 = vaddq_s32(_sum0, _sum1);
                    _sum0n = vaddq_s32(_sum0n, _sum1n);
                    _sum03 = vaddq_s32(_sum03, _sum0);
                    _sum47 = vaddq_s32(_sum47, _sum0n);

                    vst1q_lane_s32(outptr0, _sum03, 0);
                    vst1q_lane_s32(outptr1, _sum03, 1);
                    vst1q_lane_s32(outptr2, _sum03, 2);
                    vst1q_lane_s32(outptr3, _sum03, 3);
                    vst1q_lane_s32(outptr4, _sum47, 0);
                    vst1q_lane_s32(outptr5, _sum47, 1);
                    vst1q_lane_s32(outptr6, _sum47, 2);
                    vst1q_lane_s32(outptr7, _sum47, 3);

                    outptr0++;
                    outptr1++;
                    outptr2++;
                    outptr3++;
                    outptr4++;
                    outptr5++;
                    outptr6++;
                    outptr7++;
#else // __ARM_NEON
                    int sum0 = 0;
                    int sum1 = 0;
                    int sum2 = 0;
                    int sum3 = 0;
                    int sum4 = 0;
                    int sum5 = 0;
                    int sum6 = 0;
                    int sum7 = 0;

                    sum0 += (int)r0[0] * ktmp[0];
                    sum1 += (int)r0[0] * ktmp[1];
                    sum2 += (int)r0[0] * ktmp[2];
                    sum3 += (int)r0[0] * ktmp[3];
                    sum4 += (int)r0[0] * ktmp[4];
                    sum5 += (int)r0[0] * ktmp[5];
                    sum6 += (int)r0[0] * ktmp[6];
                    sum7 += (int)r0[0] * ktmp[7];
                    ktmp += 8;

                    sum0 += (int)r0[1] * ktmp[0];
                    sum1 += (int)r0[1] * ktmp[1];
                    sum2 += (int)r0[1] * ktmp[2];
                    sum3 += (int)r0[1] * ktmp[3];
                    sum4 += (int)r0[1] * ktmp[4];
                    sum5 += (int)r0[1] * ktmp[5];
                    sum6 += (int)r0[1] * ktmp[6];
                    sum7 += (int)r0[1] * ktmp[7];
                    ktmp += 8;

                    sum0 += (int)r0[2] * ktmp[0];
                    sum1 += (int)r0[2] * ktmp[1];
                    sum2 += (int)r0[2] * ktmp[2];
                    sum3 += (int)r0[2] * ktmp[3];
                    sum4 += (int)r0[2] * ktmp[4];
                    sum5 += (int)r0[2] * ktmp[5];
                    sum6 += (int)r0[2] * ktmp[6];
                    sum7 += (int)r0[2] * ktmp[7];
                    ktmp += 8;

                    sum0 += (int)r1[0] * ktmp[0];
                    sum1 += (int)r1[0] * ktmp[1];
                    sum2 += (int)r1[0] * ktmp[2];
                    sum3 += (int)r1[0] * ktmp[3];
                    sum4 += (int)r1[0] * ktmp[4];
                    sum5 += (int)r1[0] * ktmp[5];
                    sum6 += (int)r1[0] * ktmp[6];
                    sum7 += (int)r1[0] * ktmp[7];
                    ktmp += 8;

                    sum0 += (int)r1[1] * ktmp[0];
                    sum1 += (int)r1[1] * ktmp[1];
                    sum2 += (int)r1[1] * ktmp[2];
                    sum3 += (int)r1[1] * ktmp[3];
                    sum4 += (int)r1[1] * ktmp[4];
                    sum5 += (int)r1[1] * ktmp[5];
                    sum6 += (int)r1[1] * ktmp[6];
                    sum7 += (int)r1[1] * ktmp[7];
                    ktmp += 8;

                    sum0 += (int)r1[2] * ktmp[0];
                    sum1 += (int)r1[2] * ktmp[1];
                    sum2 += (int)r1[2] * ktmp[2];
                    sum3 += (int)r1[2] * ktmp[3];
                    sum4 += (int)r1[2] * ktmp[4];
                    sum5 += (int)r1[2] * ktmp[5];
                    sum6 += (int)r1[2] * ktmp[6];
                    sum7 += (int)r1[2] * ktmp[7];
                    ktmp += 8;

                    sum0 += (int)r2[0] * ktmp[0];
                    sum1 += (int)r2[0] * ktmp[1];
                    sum2 += (int)r2[0] * ktmp[2];
                    sum3 += (int)r2[0] * ktmp[3];
                    sum4 += (int)r2[0] * ktmp[4];
                    sum5 += (int)r2[0] * ktmp[5];
                    sum6 += (int)r2[0] * ktmp[6];
                    sum7 += (int)r2[0] * ktmp[7];
                    ktmp += 8;

                    sum0 += (int)r2[1] * ktmp[0];
                    sum1 += (int)r2[1] * ktmp[1];
                    sum2 += (int)r2[1] * ktmp[2];
                    sum3 += (int)r2[1] * ktmp[3];
                    sum4 += (int)r2[1] * ktmp[4];
                    sum5 += (int)r2[1] * ktmp[5];
                    sum6 += (int)r2[1] * ktmp[6];
                    sum7 += (int)r2[1] * ktmp[7];
                    ktmp += 8;

                    sum0 += (int)r2[2] * ktmp[0];
                    sum1 += (int)r2[2] * ktmp[1];
                    sum2 += (int)r2[2] * ktmp[2];
                    sum3 += (int)r2[2] * ktmp[3];
                    sum4 += (int)r2[2] * ktmp[4];
                    sum5 += (int)r2[2] * ktmp[5];
                    sum6 += (int)r2[2] * ktmp[6];
                    sum7 += (int)r2[2] * ktmp[7];
                    ktmp += 8;

                    *outptr0 += sum0;
                    *outptr1 += sum1;
                    *outptr2 += sum2;
                    *outptr3 += sum3;
                    *outptr4 += sum4;
                    *outptr5 += sum5;
                    *outptr6 += sum6;
                    *outptr7 += sum7;

                    ktmp -= 8*9;

                    outptr0++;
                    outptr1++;
                    outptr2++;
                    outptr3++;
                    outptr4++;
                    outptr5++;
                    outptr6++;
                    outptr7++;
#endif // __ARM_NEON
                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }

            ktmp += 8*9;
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=remain_outch_start; p<outch; p++)
    {
        Mat out = top_blob.channel(p);

        out.fill(0);

        const signed char* ktmp = _kernel.channel(p/8 + p%8);

        for (int q=0; q<inch; q++)
        {
            int* outptr = out;

            const signed char* img0 = bottom_blob.channel(q);

            const signed char* r0 = img0;
            const signed char* r1 = img0 + w;
            const signed char* r2 = img0 + w*2;

            int i = 0;

            for (; i < outh; i++)
            {
#if __ARM_NEON
                int nn = outw >> 3;
                int remain = outw & 7;
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
                for (; nn>0; nn--)
                {                  
                    // load output ch 0
                    int32x4_t _sum0 = vld1q_s32(outptr);// out0
                    int32x4_t _sum0n = vld1q_s32(outptr+4);// out0n

                    int8x8x2_t _r0_s8 = vld2_s8(r0); 
                    int8x8x2_t _r0n_s8 = vld2_s8(r0+16);

                    int8x8x2_t _r1_s8 = vld2_s8(r1);
                    int8x8x2_t _r1n_s8 = vld2_s8(r1+16);

                    int8x8x2_t _r2_s8 = vld2_s8(r2);
                    int8x8x2_t _r2n_s8 = vld2_s8(r2+16);

                    int8x8_t _r02_s8 = vext_s8(_r0_s8.val[0], _r0n_s8.val[0], 1);
                    int8x8_t _r12_s8 = vext_s8(_r1_s8.val[0], _r1n_s8.val[0], 1);
                    int8x8_t _r22_s8 = vext_s8(_r2_s8.val[0], _r2n_s8.val[0], 1);

                    int16x8_t _r00 = vmovl_s8(_r0_s8.val[0]); // r00
                    int16x8_t _r01 = vmovl_s8(_r0_s8.val[1]); // r01
                    int16x8_t _r02 = vmovl_s8(_r02_s8);       // r02

                    int16x8_t _r10 = vmovl_s8(_r1_s8.val[0]); // r10
                    int16x8_t _r11 = vmovl_s8(_r1_s8.val[1]); // r11
                    int16x8_t _r12 = vmovl_s8(_r12_s8);       // r12

                    int16x8_t _r20 = vmovl_s8(_r2_s8.val[0]); // r20
                    int16x8_t _r21 = vmovl_s8(_r2_s8.val[1]); // r21
                    int16x8_t _r22 = vmovl_s8(_r22_s8);       // r22

                    int8x16_t _k_s8 = vld1q_s8(ktmp);
                    int16x8_t _k_s16 = vmovl_s8(vget_low_s8(_k_s8)); // k0...k8
                    int16x8_t _kn_s16 = vmovl_s8(vget_high_s8(_k_s8));// k9... 

                    _sum0 = vmlal_laneq_s16(_sum0, vget_low_s16(_r00), _k_s16, 0);
                    _sum0n = vmlal_laneq_s16(_sum0n, vget_high_s16(_r00), _k_s16, 0);
                    int32x4_t _sum01 = vmull_laneq_s16(vget_low_s16(_r01), _k_s16, 1);
                    int32x4_t _sum01n = vmull_laneq_s16(vget_high_s16(_r01), _k_s16, 1);
                    _sum0 = vmlal_laneq_s16(_sum0, vget_low_s16(_r02), _k_s16, 2);
                    _sum0n = vmlal_laneq_s16(_sum0n, vget_high_s16(_r02), _k_s16, 2);

                    _sum0 = vmlal_laneq_s16(_sum0, vget_low_s16(_r10), _k_s16, 3);
                    _sum0n = vmlal_laneq_s16(_sum0n, vget_high_s16(_r10), _k_s16, 3);
                    _sum01 = vmlal_laneq_s16(_sum01, vget_low_s16(_r11), _k_s16, 4);
                    _sum01n = vmlal_laneq_s16(_sum01n, vget_high_s16(_r11), _k_s16, 4);
                    _sum0 = vmlal_laneq_s16(_sum0, vget_low_s16(_r12), _k_s16, 5);
                    _sum0n = vmlal_laneq_s16(_sum0n, vget_high_s16(_r12), _k_s16, 5);

                    _sum0 = vmlal_laneq_s16(_sum0, vget_low_s16(_r20), _k_s16, 6);
                    _sum0n = vmlal_laneq_s16(_sum0n, vget_high_s16(_r20), _k_s16, 6);
                    _sum01 = vmlal_laneq_s16(_sum01, vget_low_s16(_r21), _k_s16, 7);
                    _sum01n = vmlal_laneq_s16(_sum01n, vget_high_s16(_r21), _k_s16, 7);
                    _sum0 = vmlal_laneq_s16(_sum0, vget_low_s16(_r22), _kn_s16, 0);
                    _sum0n = vmlal_laneq_s16(_sum0n, vget_high_s16(_r22), _kn_s16, 0);

                    _sum0 = vaddq_s32(_sum0, _sum01);
                    _sum0n = vaddq_s32(_sum0n, _sum01n);

                    // save s32 to memory
                    vst1q_s32(outptr, _sum0);
                    vst1q_s32(outptr+4, _sum0n);
               
                    r0 += 16;
                    r1 += 16;
                    r2 += 16;

                    outptr += 8;
                }
#endif // __ARM_NEON

                if (remain > 0)
                {
#if __ARM_NEON
                    int8x8_t _k01234567s8 = vld1_s8(ktmp);
                    int8x8_t _k8xxxxxxxs8 = vld1_s8(ktmp+8);
                    int8x8_t _k34567xxxs8 = vext_s8(_k01234567s8, _k01234567s8, 3);
                    int8x8_t _k678xxxxxs8 = vext_s8(_k01234567s8, _k8xxxxxxxs8, 6);
                    int16x8_t _k0123_s16 = vmovl_s8(_k01234567s8);
                    int16x8_t _k3456_s16 = vmovl_s8(_k34567xxxs8);
                    int16x8_t _k678x_s16 = vmovl_s8(_k678xxxxxs8);
#endif
                    for (; remain>0; remain--)
                    {
#if __ARM_NEON
                        int8x8_t _r00s8 = vld1_s8(r0);
                        int8x8_t _r10s8 = vld1_s8(r1);
                        int8x8_t _r20s8 = vld1_s8(r2);

                        int16x8_t _r00s16 = vmovl_s8(_r00s8);
                        int16x8_t _r10s16 = vmovl_s8(_r10s8);
                        int16x8_t _r20s16 = vmovl_s8(_r20s8);

                        int32x4_t _sum = vmull_s16(vget_low_s16(_r00s16), vget_low_s16(_k0123_s16));
                        _sum = vmlal_s16(_sum, vget_low_s16(_r10s16), vget_low_s16(_k3456_s16));
                        _sum = vmlal_s16(_sum, vget_low_s16(_r20s16), vget_low_s16(_k678x_s16));

                        _sum = vsetq_lane_s32(*outptr, _sum, 3);

                        *outptr = vaddvq_s32(_sum);
#else
                        int sum = 0;

                        sum += (int)r0[0] * ktmp[0];
                        sum += (int)r0[1] * ktmp[1];
                        sum += (int)r0[2] * ktmp[2];
                        sum += (int)r1[0] * ktmp[3];
                        sum += (int)r1[1] * ktmp[4];
                        sum += (int)r1[2] * ktmp[5];
                        sum += (int)r2[0] * ktmp[6];
                        sum += (int)r2[1] * ktmp[7];
                        sum += (int)r2[2] * ktmp[8];

                        *outptr += sum;
#endif // __ARM_NEON
                        r0 += 2;
                        r1 += 2;
                        r2 += 2;
                        outptr++;
                    }
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }

            ktmp += 9;
        }
    }
}
#else // __aarch64__
static void conv3x3s1_winograd23_int8_neon(const Mat& bottom_blob, Mat& top_blob, const std::vector<Mat> &kernel_tm_test, const Option& opt)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    // pad to 2n+2, winograd F(2,3)
    Mat bottom_blob_bordered = bottom_blob;

    outw = (outw + 1) / 2 * 2;
    outh = (outh + 1) / 2 * 2;

    w = outw + 2;
    h = outh + 2;
    copy_make_border(bottom_blob, bottom_blob_bordered, 0, h - bottom_blob.h, 0, w - bottom_blob.w, 0, 0.f, opt.workspace_allocator, opt.num_threads);  

    // BEGIN transform input
    Mat bottom_blob_tm;
    {
        int w_tm = outw / 2 * 4;
        int h_tm = outh / 2 * 4;

        int nColBlocks = h_tm/4; // may be the block num in FeatherCNN
        int nRowBlocks = w_tm/4;

        const int tiles = nColBlocks * nRowBlocks;

        bottom_blob_tm.create(4, inch, tiles*4, 2u, opt.workspace_allocator);

        // BT
        // const float itm[4][4] = {
        //     {1.0f,  0.0f, -1.0f,  0.0f},
        //     {0.0f,  1.0f,  1.00f, 0.0f},
        //     {0.0f, -1.0f,  1.00f, 0.0f},
        //     {0.0f, -1.0f,  0.00f, 1.0f}
        // };        
        
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<inch; q++)
        {
            const signed char* img = bottom_blob_bordered.channel(q);

            for (int j=0; j<nColBlocks; j++)
            {
                const signed char* r0 = img + w * j * 2;
                const signed char* r1 = r0 + w;
                const signed char* r2 = r1 + w;
                const signed char* r3 = r2 + w;

                for (int i = 0; i<nRowBlocks; i++)
                {
                    short* out_tm0 = bottom_blob_tm.channel(tiles*0+j*nRowBlocks+i).row<short>(q);
                    short* out_tm1 = bottom_blob_tm.channel(tiles*1+j*nRowBlocks+i).row<short>(q);
                    short* out_tm2 = bottom_blob_tm.channel(tiles*2+j*nRowBlocks+i).row<short>(q);
                    short* out_tm3 = bottom_blob_tm.channel(tiles*3+j*nRowBlocks+i).row<short>(q);
#if __ARM_NEON
                    asm volatile(
                        // load
                        "pld         [%0, #64]     \n"
                        "vld1.s8     {d0}, [%0]    \n"
                        "pld         [%1, #64]     \n"
                        "vld1.s8     {d1}, [%1]    \n"
                        "pld         [%2, #64]     \n"
                        "vld1.s8     {d2}, [%2]    \n"
                        "pld         [%3, #64]     \n"
                        "vld1.s8     {d3}, [%3]    \n"
                        // w = B_t * d, trans int8 to int16
                        "vsubl.s8    q2, d0, d2    \n" // d4
                        "vaddl.s8    q3, d1, d2    \n" // d6
                        "vsubl.s8    q4, d2, d1    \n" // d8
                        "vsubl.s8    q5, d3, d1    \n" // d10
                        // transpose w to w_t
                        "vtrn.s16    d4, d6        \n"
                        "vtrn.s16    d8, d10       \n"
                        "vtrn.s32    d4, d8        \n"
                        "vtrn.s32    d6, d10       \n"
                        // U = B_t * d_t
                        "vsub.s16    d11, d4, d8   \n"
                        "vadd.s16    d12, d6, d8   \n"
                        "vsub.s16    d13, d8, d6   \n"
                        "vsub.s16    d14, d10, d6  \n"
                        // save
                        "vst1.s32    {d11}, [%4]   \n"
                        "vst1.s32    {d12}, [%5]   \n"
                        "vst1.s32    {d13}, [%6]   \n"
                        "vst1.s32    {d14}, [%7]   \n"
                        : "=r"(r0),      // %0
                          "=r"(r1),      // %1
                          "=r"(r2),      // %2
                          "=r"(r3),      // %3
                          "=r"(out_tm0), // %4
                          "=r"(out_tm1), // %5
                          "=r"(out_tm2), // %6
                          "=r"(out_tm3)  // %7
                        : "0"(r0),
                          "1"(r1),
                          "2"(r2),
                          "3"(r3),
                          "4"(out_tm0),
                          "5"(out_tm1),
                          "6"(out_tm2),
                          "7"(out_tm3)
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7"
                    );
#else
                    short d0[4],d1[4],d2[4],d3[4];
                    short w0[4],w1[4],w2[4],w3[4];
                    short t0[4],t1[4],t2[4],t3[4];
                    // load 
                    for (int n = 0; n < 4; n++)
                    {
                        d0[n] = r0[n];
                        d1[n] = r1[n];
                        d2[n] = r2[n];
                        d3[n] = r3[n];
                    }
                    // w = B_t * d
                    for (int n = 0; n < 4; n++)
                    {   
                        w0[n] = d0[n] - d2[n];
                        w1[n] = d1[n] + d2[n];
                        w2[n] = d2[n] - d1[n];
                        w3[n] = d3[n] - d1[n];
                    } 
                    // transpose d to d_t
                    {
                        t0[0]=w0[0]; t1[0]=w0[1]; t2[0]=w0[2]; t3[0]=w0[3];
                        t0[1]=w1[0]; t1[1]=w1[1]; t2[1]=w1[2]; t3[1]=w1[3];
                        t0[2]=w2[0]; t1[2]=w2[1]; t2[2]=w2[2]; t3[2]=w2[3];
                        t0[3]=w3[0]; t1[3]=w3[1]; t2[3]=w3[2]; t3[3]=w3[3];
                    }
                    // U = B_t * d_t
                    for (int n = 0; n < 4; n++)
                    {   
                        d0[n] = t0[n] - t2[n];
                        d1[n] = t1[n] + t2[n];
                        d2[n] = t2[n] - t1[n];
                        d3[n] = t3[n] - t1[n];
                    }                
                    // save to out_tm
                    for (int n = 0; n < 4; n++)
                    {
                        out_tm0[n] = d0[n];
                        out_tm1[n] = d1[n];
                        out_tm2[n] = d2[n];
                        out_tm3[n] = d3[n];
                    }
#endif                           
                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    r3 += 2;    
                }
            }
        }
    }
    bottom_blob_bordered = Mat();

    // BEGIN dot
    Mat top_blob_tm;
    {
        int w_tm = outw / 2 * 4;
        int h_tm = outh / 2 * 4;

        int nColBlocks = h_tm/4; // may be the block num in FeatherCNN
        int nRowBlocks = w_tm/4;

        const int tiles = nColBlocks * nRowBlocks; 

        top_blob_tm.create(16, tiles, outch, 4u, opt.workspace_allocator);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int r=0; r<4; r++)
        {
            int nn_outch = 0;
            int remain_outch_start = 0;

            nn_outch = outch >> 3;
            remain_outch_start = nn_outch << 3;

            for (int pp=0; pp<nn_outch; pp++)
            {
                int p = pp * 8;

                int* output0_tm = top_blob_tm.channel(p);
                int* output1_tm = top_blob_tm.channel(p+1);
                int* output2_tm = top_blob_tm.channel(p+2);
                int* output3_tm = top_blob_tm.channel(p+3);
                int* output4_tm = top_blob_tm.channel(p+4);
                int* output5_tm = top_blob_tm.channel(p+5);
                int* output6_tm = top_blob_tm.channel(p+6);
                int* output7_tm = top_blob_tm.channel(p+7);

                output0_tm = output0_tm + r*4;
                output1_tm = output1_tm + r*4;
                output2_tm = output2_tm + r*4;
                output3_tm = output3_tm + r*4;
                output4_tm = output4_tm + r*4;
                output5_tm = output5_tm + r*4;
                output6_tm = output6_tm + r*4;
                output7_tm = output7_tm + r*4;

                for (int i=0; i<tiles; i++)
                {
                    const short* kptr = kernel_tm_test[r].channel(p/8);
                    const short* r0 = bottom_blob_tm.channel(tiles*r+i);
#if __ARM_NEON
                    asm volatile(
                        // inch loop
                        "vmov.s32    q0, #0           \n"
                        "vmov.s32    q1, #0           \n"
                        "vmov.s32    q2, #0           \n"
                        "vmov.s32    q3, #0           \n"
                        "vmov.s32    q4, #0           \n"
                        "vmov.s32    q5, #0           \n"
                        "vmov.s32    q6, #0           \n"
                        "vmov.s32    q7, #0           \n"
                        "mov         r4, %20          \n"
                        
                        "0:                           \n" // for (int q=0; q<inch; q++)
                        "vld1.s16    {d16}, [%8]!     \n" // _r0 = vld1_s16(r0);  // input inch0
                        "vld1.s16    {d18-d19}, [%9]  \n" // _k0 = vld1q_s16(kptr);
                        "add         %9, #16          \n" 
                        "vld1.s16    {d20-d21}, [%9]  \n" // _k0n = vld1q_s16(kptr+8);
                        "add         %9, #16          \n"   
                        "vld1.s16    {d22-d23}, [%9]  \n" // _k1 = vld1q_s16(kptr+16);
                        "add         %9, #16          \n"  
                        "vld1.s16    {d24-d25}, [%9]  \n" // _k1n = vld1q_s16(kptr+24);
                        "add         %9, #16          \n"

                        "vmlal.s16   q0, d16, d18     \n" // sum0 += (a00-a03) * (k00-k03)
                        "vmlal.s16   q1, d16, d19     \n" // sum1 += (a00-a03) * (k10-k13)
                        "vmlal.s16   q2, d16, d20     \n" // sum2 += (a00-a03) * (k20-k23)
                        "vmlal.s16   q3, d16, d21     \n" // sum3 += (a00-a03) * (k30-k33)
                        "vmlal.s16   q4, d16, d22     \n" // sum4 += (a00-a03) * (k40-k43)
                        "vmlal.s16   q5, d16, d23     \n" // sum5 += (a00-a03) * (k50-k53)
                        "vmlal.s16   q6, d16, d24     \n" // sum6 += (a00-a03) * (k60-k63)
                        "vmlal.s16   q7, d16, d25     \n" // sum7 += (a00-a03) * (k70-k73)

                        "subs        r4, r4, #1       \n"
                        "bne         0b               \n" // end for

                        "vst1.s32    {d0-d1}, [%0]    \n" // store the result to memory
                        "vst1.s32    {d2-d3}, [%1]    \n"
                        "vst1.s32    {d4-d5}, [%2]    \n"
                        "vst1.s32    {d6-d7}, [%3]    \n"
                        "vst1.s32    {d8-d9}, [%4]    \n"
                        "vst1.s32    {d10-d11}, [%5]  \n"
                        "vst1.s32    {d12-d13}, [%6]  \n"
                        "vst1.s32    {d14-d15}, [%7]  \n"

                        : "=r"(output0_tm), // %0
                          "=r"(output1_tm), // %1
                          "=r"(output2_tm), // %2
                          "=r"(output3_tm), // %3
                          "=r"(output4_tm), // %4
                          "=r"(output5_tm), // %5
                          "=r"(output6_tm), // %6
                          "=r"(output7_tm), // %7
                          "=r"(r0),         // %8
                          "=r"(kptr)        // %9
                        : "0"(output0_tm),
                          "1"(output1_tm),
                          "2"(output2_tm),
                          "3"(output3_tm),
                          "4"(output4_tm),
                          "5"(output5_tm),
                          "6"(output6_tm),
                          "7"(output7_tm),
                          "8"(r0),
                          "9"(kptr),
                          "r"(inch)         // %20
                        : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12"
                    );
#else
                    int sum0[4] = {0};
                    int sum1[4] = {0};
                    int sum2[4] = {0};
                    int sum3[4] = {0};
                    int sum4[4] = {0};
                    int sum5[4] = {0};
                    int sum6[4] = {0};
                    int sum7[4] = {0};

                    for (int q=0; q<inch; q++)
                    {
                        for (int n=0; n<4; n++)
                        {
                            sum0[n] += (int)r0[n] * kptr[n];
                            sum1[n] += (int)r0[n] * kptr[n+4];
                            sum2[n] += (int)r0[n] * kptr[n+8];
                            sum3[n] += (int)r0[n] * kptr[n+12];
                            sum4[n] += (int)r0[n] * kptr[n+16];
                            sum5[n] += (int)r0[n] * kptr[n+20];
                            sum6[n] += (int)r0[n] * kptr[n+24];
                            sum7[n] += (int)r0[n] * kptr[n+28];
                        }
                        kptr += 32;
                        r0 += 4;
                    }

                    for (int n=0; n<4; n++)
                    {
                        output0_tm[n] = sum0[n];
                        output1_tm[n] = sum1[n];
                        output2_tm[n] = sum2[n];
                        output3_tm[n] = sum3[n];
                        output4_tm[n] = sum4[n];
                        output5_tm[n] = sum5[n];
                        output6_tm[n] = sum6[n];
                        output7_tm[n] = sum7[n];
                    }
#endif // __ARM_NEON
                    output0_tm += 16;
                    output1_tm += 16;
                    output2_tm += 16;
                    output3_tm += 16;
                    output4_tm += 16;
                    output5_tm += 16;
                    output6_tm += 16;
                    output7_tm += 16;
                }
            }

            nn_outch = (outch - remain_outch_start) >> 2;

            //#pragma omp parallel for num_threads(opt.num_threads)
            for (int pp=0; pp<nn_outch; pp++)
            {
                int p = remain_outch_start + pp * 4;

                int* output0_tm = top_blob_tm.channel(p);
                int* output1_tm = top_blob_tm.channel(p+1);
                int* output2_tm = top_blob_tm.channel(p+2);
                int* output3_tm = top_blob_tm.channel(p+3);

                output0_tm = output0_tm + r*4;
                output1_tm = output1_tm + r*4;
                output2_tm = output2_tm + r*4;
                output3_tm = output3_tm + r*4;

                for (int i=0; i<tiles; i++)
                {
                    const short* kptr = kernel_tm_test[r].channel(p/8 + (p%8)/4);
                    const short* r0 = bottom_blob_tm.channel(tiles*r+i);
#if __ARM_NEON
                    asm volatile(
                        // inch loop
                        "vmov.s32    q0, #0           \n"
                        "vmov.s32    q1, #0           \n"
                        "vmov.s32    q2, #0           \n"
                        "vmov.s32    q3, #0           \n"
                        "mov         r4, %12          \n"
                        
                        "0:                           \n" // for (int q=0; q<inch; q++)
                        "vld1.s16    {d16}, [%4]!     \n" // _r0 = vld1_s16(r0);  // input inch0
                        "vld1.s16    {d18-d19}, [%5]  \n" // _k0 = vld1q_s16(kptr);
                        "add         %5, #16          \n" 
                        "vld1.s16    {d20-d21}, [%5]  \n" // _k0n = vld1q_s16(kptr+8);
                        "add         %5, #16          \n"

                        "vmlal.s16   q0, d16, d18     \n" // sum0 += (a00-a03) * (k00-k03)
                        "vmlal.s16   q1, d16, d19     \n" // sum1 += (a00-a03) * (k10-k13)
                        "vmlal.s16   q2, d16, d20     \n" // sum2 += (a00-a03) * (k20-k23)
                        "vmlal.s16   q3, d16, d21     \n" // sum3 += (a00-a03) * (k30-k33)

                        "subs        r4, r4, #1       \n"
                        "bne         0b               \n" // end for

                        "vst1.s32    {d0-d1}, [%0]    \n" // store the result to memory
                        "vst1.s32    {d2-d3}, [%1]    \n"
                        "vst1.s32    {d4-d5}, [%2]    \n"
                        "vst1.s32    {d6-d7}, [%3]    \n"

                        : "=r"(output0_tm), // %0
                          "=r"(output1_tm), // %1
                          "=r"(output2_tm), // %2
                          "=r"(output3_tm), // %3
                          "=r"(r0),         // %4
                          "=r"(kptr)        // %5
                        : "0"(output0_tm),
                          "1"(output1_tm),
                          "2"(output2_tm),
                          "3"(output3_tm),
                          "4"(r0),
                          "5"(kptr),
                          "r"(inch)         // %12
                        : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q8", "q9", "q10"
                    );                   
#else
                    int sum0[4] = {0};
                    int sum1[4] = {0};
                    int sum2[4] = {0};
                    int sum3[4] = {0};

                    for (int q=0; q<inch; q++)
                    {   
                        for (int n=0; n<4; n++)
                        {
                            sum0[n] += (int)r0[n] * kptr[n];
                            sum1[n] += (int)r0[n] * kptr[n+4];
                            sum2[n] += (int)r0[n] * kptr[n+8];
                            sum3[n] += (int)r0[n] * kptr[n+12];
                        }
                        kptr += 16;
                        r0 += 4;
                    }

                    for (int n=0; n<4; n++)
                    {
                        output0_tm[n] = sum0[n];
                        output1_tm[n] = sum1[n];
                        output2_tm[n] = sum2[n];
                        output3_tm[n] = sum3[n];
                    }
#endif // __ARM_NEON
                    output0_tm += 16;
                    output1_tm += 16;
                    output2_tm += 16;
                    output3_tm += 16;
                }
            }

            remain_outch_start += nn_outch << 2;
            //#pragma omp parallel for num_threads(opt.num_threads)
            for (int p=remain_outch_start; p<outch; p++)
            {
                int* output0_tm = top_blob_tm.channel(p);

                output0_tm = output0_tm + r*4;

                for (int i=0; i<tiles; i++)
                {
                    const short* kptr = kernel_tm_test[r].channel(p/8 + (p%8)/4 + p%4);
                    const short* r0 = bottom_blob_tm.channel(tiles*r+i);
#if __ARM_NEON
                    asm volatile(
                        // inch loop
                        "vmov.s32    q0, #0           \n"
                        "mov         r4, %6           \n"
                        
                        "0:                           \n" // for (int q=0; q<inch; q++)
                        "vld1.s16    {d16}, [%1]      \n" // _r0 = vld1_s16(r0);  // input inch0
                        "add         %1, #8           \n"
                        "vld1.s16    {d18}, [%2]      \n" // _k0 = vld1q_s16(kptr);
                        "add         %2, #8           \n"
                        "vmlal.s16   q0, d16, d18     \n" // sum0 += (a00-a03) * (k00-k03)

                        "subs        r4, r4, #1       \n"
                        "bne         0b               \n" // end for

                        "vst1.s32    {d0-d1}, [%0]    \n" // store the result to memory

                        : "=r"(output0_tm), // %0
                          "=r"(r0),         // %1
                          "=r"(kptr)        // %2
                        : "0"(output0_tm),
                          "1"(r0),
                          "2"(kptr),
                          "r"(inch)         // %6
                        : "cc", "memory", "r4", "q0", "q8", "q9"
                    );               
#else
                    int sum0[4] = {0};

                    for (int q=0; q<inch; q++)
                    {
                        for (int n=0; n<4; n++)
                        {
                            sum0[n] += (int)r0[n] * kptr[n];
                        }
                        kptr += 4; 
                        r0 += 4;
                    }

                    for (int n=0; n<4; n++)
                    {
                        output0_tm[n] = sum0[n];
                    }           
#endif                           
                    output0_tm += 16;       
                }
            }
        }   
    }
    bottom_blob_tm = Mat();
    // END dot    

    // BEGIN transform output
    Mat top_blob_bordered;
    top_blob_bordered.create(outw, outh, outch, 4u, opt.workspace_allocator);
    {
        // AT
        // const float itm[2][4] = {
        //     {1.0f,  1.0f,  1.0f,  0.0f},
        //     {0.0f,  1.0f, -1.0f,  1.0f}
        // }; 

        int w_tm = outw / 2 * 4;
        int h_tm = outh / 2 * 4;

        int nColBlocks = h_tm/4; // may be the block num in FeatherCNN
        int nRowBlocks = w_tm/4;

        int32x2_t _shift = vdup_n_s32(-2);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p=0; p<outch; p++)
        {
            int* out_tile = top_blob_tm.channel(p);
            int* outRow0 = top_blob_bordered.channel(p);
            int* outRow1 = outRow0 + outw;     

            for (int j=0; j<nColBlocks; j++)
            {
                for(int i=0; i<nRowBlocks; i++)
                {
#if __ARM_NEON
                    asm volatile(
                        "pld        [%0, #512]      \n"
                        "vldm        %0!, {d0-d7}   \n"

                        "vaddq.s32    q0, q0, q1    \n" // s0 = s0 + s1 + s2;
                        "vsubq.s32    q1, q1, q2    \n"
                        "vaddq.s32    q0, q0, q2    \n" // s1 = s1 - s2 + s3;
                        "vaddq.s32    q1, q1, q3    \n"

                        "vtrn.s32    q0, q1         \n"
                        
                        "vadd.s32    d8, d0, d2     \n" // o0 = d0 + d1 + d2;
                        "vsub.s32    d9, d2, d1     \n"
                        "vadd.s32    d8, d8, d1     \n" // o1 = d1 - d2 + d3;
                        "vadd.s32    d9, d9, d3     \n"

                        "vshl.s32    d8, d8, %P6    \n" // o0 = o0 >> 2
                        "vshl.s32    d9, d9, %P6    \n" // o1 = o1 >> 2

                        "vst1.s32    {d8}, [%1]!    \n"
                        "vst1.s32    {d9}, [%2]!    \n"
                        : "=r"(out_tile), // %0
                          "=r"(outRow0),  // %1
                          "=r"(outRow1)   // %2
                        : "0"(out_tile),
                          "1"(outRow0),
                          "2"(outRow1),
                          "w"(_shift)     // %6
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4"
                    );
#else
                    int s0[4],s1[4],s2[4],s3[4];
                    int w0[4],w1[4];
                    int d0[2],d1[2],d2[2],d3[2];
                    int o0[2],o1[2];
                    // load
                    for (int n = 0; n < 4; n++)
                    {
                        s0[n] = out_tile[n];
                        s1[n] = out_tile[n+ 4];
                        s2[n] = out_tile[n+ 8];
                        s3[n] = out_tile[n+12];
                    }
                    // w = A_T * W
                    for (int n = 0; n < 4; n++)
                    {
                        w0[n] = s0[n] + s1[n] + s2[n];
                        w1[n] = s1[n] - s2[n] + s3[n];
                    }
                    // transpose w to w_t
                    {
                        d0[0] = w0[0]; d0[1] = w1[0];
                        d1[0] = w0[1]; d1[1] = w1[1];
                        d2[0] = w0[2]; d2[1] = w1[2];
                        d3[0] = w0[3]; d3[1] = w1[3];
                    }
                    // Y = A_T * w_t
                    for (int n = 0; n < 2; n++)
                    {
                        o0[n] = d0[n] + d1[n] + d2[n];
                        o1[n] = d1[n] - d2[n] + d3[n];
                    }
                    // save to top blob tm,why right 2,because the G' = G*2
                    outRow0[0] = o0[0] >> 2;
                    outRow0[1] = o0[1] >> 2;
                    outRow1[0] = o1[0] >> 2;
                    outRow1[1] = o1[1] >> 2;

                    out_tile += 16;

                    outRow0 += 2;
                    outRow1 += 2;
#endif // __ARM_NEON
                }

                outRow0 += outw;
                outRow1 += outw;
            }
        }        
    }
    // END transform output 
    
    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt.blob_allocator, opt.num_threads);  
}

static void conv3x3s1_packed_int8_neon(const Mat &bottom_blob, Mat &top_blob, const Mat &_kernel, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    int nn_outch = outch >> 2;
    int remain_outch_start = nn_outch << 2;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp=0; pp < nn_outch; pp++)
    {
        int p = pp * 4;

        Mat out0 = top_blob.channel(p+0);
        Mat out1 = top_blob.channel(p+1);
        Mat out2 = top_blob.channel(p+2);
        Mat out3 = top_blob.channel(p+3);

        out0.fill(0);
        out1.fill(0);
        out2.fill(0);
        out3.fill(0);

        const signed char* ktmp = _kernel.channel(p/4);

        for (int q = 0; q < inch; q++)
        {
            int* outptr0 = out0;
            int* outptr1 = out1;
            int* outptr2 = out2;
            int* outptr3 = out3;

            const signed char *img0 = bottom_blob.channel(q);

            const signed char *r0 = img0;
            const signed char *r1 = img0 + w;
            const signed char *r2 = img0 + w * 2;

            int i = 0;    

            for (; i < outh; i++)
            {
#if __ARM_NEON
                int nn = outw >> 3;
                int remain = outw & 7;
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
                if (nn > 0)
                {
                asm volatile(
                    "0:                         \n"
                    "vld1.s8    {d0-d3}, [%8]!  \n"// d0=(k00-k30 k01-k31) d1=(k02-k32 k03-k33) d2=(k04-k34 k05-k35) d3=(k06-k36 k07-k37)
                    // r0
                    "pld        [%5, #128]      \n"
                    "vld1.s8    {d8-d9}, [%5]   \n"// d8=r00-r07 d9=r08-r015 q4
                    "add        %5, #8          \n"
                    "pld        [%1, #128]      \n"
                    "vld1.s32   {d12-d15}, [%1] \n"// sum00-sum07 q6 q7
                    "pld        [%2, #128]      \n"
                    "vld1.s32   {d16-d19}, [%2] \n"// sum10-sum17 q8 q9
                    "pld        [%3, #128]      \n"
                    "vld1.s32   {d20-d23}, [%3] \n"// sum20-sum27 q10 q11
                    "pld        [%4, #128]      \n"
                    "vld1.s32   {d24-d27}, [%4] \n"// sum30-sum37 q12 q13
                    
                    "vmovl.s8   q3, d3          \n"// d6(k06-k36) d7(k07-k37) 
                    "vmovl.s8   q2, d2          \n"// d4(k04-k34) d5(k05-k35)
                    "vmovl.s8   q1, d1          \n"// d2(k02-k32) d3(k03-k33)
                    "vmovl.s8   q0, d0          \n"// d0(k00-k30) d1(k01-k31)
                    "vmovl.s8   q5, d8          \n"// d10(r00-r03) d11(r04-r07)
                    
                    "vmlal.s16  q6, d10, d0[0]  \n"// sum(00-07) += (r00-r07) * k00
                    "vmlal.s16  q7, d11, d0[0]  \n"
                    "vmlal.s16  q8, d10, d0[1]  \n"// sum(10-17) += (r00-r07) * k10
                    "vmlal.s16  q9, d11, d0[1]  \n"     
                    "vmlal.s16  q10, d10, d0[2] \n"// sum(20-27) += (r00-r07) * k20
                    "vmlal.s16  q11, d11, d0[2] \n"
                    "vmlal.s16  q12, d10, d0[3] \n"// sum(30-37) += (r00-r07) * k30
                    "vmlal.s16  q13, d11, d0[3] \n"

                    "vext.s8    q4, q4, #1      \n"// d8=r01-r08 q4
                    "vmovl.s8   q5, d8          \n"// d10(r01-r04) d11(r05-r08)
                    
                    "vmlal.s16  q6, d10, d1[0]  \n"// sum(00-07) += (r01-r08) * k01
                    "vmlal.s16  q7, d11, d1[0]  \n"
                    "vmlal.s16  q8, d10, d1[1]  \n"// sum(10-17) += (r01-r08) * k11
                    "vmlal.s16  q9, d11, d1[1]  \n"     
                    "vmlal.s16  q10, d10, d1[2] \n"// sum(20-27) += (r01-r08) * k21
                    "vmlal.s16  q11, d11, d1[2] \n"
                    "vmlal.s16  q12, d10, d1[3] \n"// sum(30-37) += (r01-r08) * k31
                    "vmlal.s16  q13, d11, d1[3] \n"
                    
                    "vext.s8    q4, q4, #1      \n"// d8=r02-r09 q4
                    "vmovl.s8   q5, d8          \n"// d10(r02-r05) d11(r06-r09)
                    
                    "vmlal.s16  q6, d10, d2[0]  \n"// sum(00-07) += (r02-r09) * k02
                    "vmlal.s16  q7, d11, d2[0]  \n"
                    "vmlal.s16  q8, d10, d2[1]  \n"// sum(10-17) += (r02-r09) * k12
                    "vmlal.s16  q9, d11, d2[1]  \n"     
                    "vmlal.s16  q10, d10, d2[2] \n"// sum(20-27) += (r02-r09) * k22
                    "vmlal.s16  q11, d11, d2[2] \n"
                    "vmlal.s16  q12, d10, d2[3] \n"// sum(30-37) += (r02-r09) * k32
                    "vmlal.s16  q13, d11, d2[3] \n"                 
                    
                    // r1
                    "pld        [%6, #128]      \n"
                    "vld1.s8    {d8-d9}, [%6]   \n"// d8=r10-r17 d9=r18-r115 q4
                    "add        %6, #8          \n"
                    "vmovl.s8   q5, d8          \n"// d10(r10-r13) d11(r14-r17)
                    
                    "vmlal.s16  q6, d10, d3[0]  \n"// sum(00-07) += (r10-r17) * k03
                    "vmlal.s16  q7, d11, d3[0]  \n"
                    "vmlal.s16  q8, d10, d3[1]  \n"// sum(10-17) += (r10-r17) * k13
                    "vmlal.s16  q9, d11, d3[1]  \n"     
                    "vmlal.s16  q10, d10, d3[2] \n"// sum(20-27) += (r10-r17) * k23
                    "vmlal.s16  q11, d11, d3[2] \n"
                    "vmlal.s16  q12, d10, d3[3] \n"// sum(30-37) += (r10-r17) * k33
                    "vmlal.s16  q13, d11, d3[3] \n"
                    
                    "vext.s8    q4, q4, #1      \n"// d8=r11-r18 q4
                    "vmovl.s8   q5, d8          \n"// d10(r11-r14) d11(r15-r18)

                    "vmlal.s16  q6, d10, d4[0]  \n"// sum(00-07) += (r11-r18) * k04
                    "vmlal.s16  q7, d11, d4[0]  \n"
                    "vmlal.s16  q8, d10, d4[1]  \n"// sum(10-17) += (r11-r18) * k14
                    "vmlal.s16  q9, d11, d4[1]  \n"     
                    "vmlal.s16  q10, d10, d4[2] \n"// sum(20-27) += (r11-r18) * k24
                    "vmlal.s16  q11, d11, d4[2] \n"
                    "vmlal.s16  q12, d10, d4[3] \n"// sum(30-37) += (r11-r18) * k34
                    "vmlal.s16  q13, d11, d4[3] \n"
                    
                    "vext.s8    q4, q4, #1      \n"// d8=r12-r19 q4
                    "vmovl.s8   q5, d8          \n"// d10(r12-r15) d11(r16-r19)

                    "vmlal.s16  q6, d10, d5[0]  \n"// sum(00-07) += (r12-r19) * k05
                    "vmlal.s16  q7, d11, d5[0]  \n"
                    "vmlal.s16  q8, d10, d5[1]  \n"// sum(10-17) += (r12-r19) * k15
                    "vmlal.s16  q9, d11, d5[1]  \n"     
                    "vmlal.s16  q10, d10, d5[2] \n"// sum(20-27) += (r12-r19) * k25
                    "vmlal.s16  q11, d11, d5[2] \n"
                    "vmlal.s16  q12, d10, d5[3] \n"// sum(30-37) += (r12-r19) * k35
                    "vmlal.s16  q13, d11, d5[3] \n"

                    // r2
                    "pld        [%7, #128]      \n"
                    "vld1.s8    {d8-d9}, [%7]   \n"// d8=r20-r27 d9=r28-r215 q4
                    "add        %7, #8          \n"
                    "vmovl.s8   q5, d8          \n"// d10(r20-r23) d11(r24-r27)

                    "vmlal.s16  q6, d10, d6[0]  \n"// sum(00-07) += (r20-r27) * k06
                    "vmlal.s16  q7, d11, d6[0]  \n"
                    "vmlal.s16  q8, d10, d6[1]  \n"// sum(10-17) += (r20-r27) * k16
                    "vmlal.s16  q9, d11, d6[1]  \n"     
                    "vmlal.s16  q10, d10, d6[2] \n"// sum(20-27) += (r20-r27) * k26
                    "vmlal.s16  q11, d11, d6[2] \n"
                    "vmlal.s16  q12, d10, d6[3] \n"// sum(30-37) += (r20-r27) * k36
                    "vmlal.s16  q13, d11, d6[3] \n"
                    
                    "vext.s8    q4, q4, #1      \n"// d8=r21-r28 q4
                    "vmovl.s8   q5, d8          \n"// d10(r21-r24) d11(r25-r28)

                    "vmlal.s16  q6, d10, d7[0]  \n"// sum(00-07) += (r21-r28) * k07
                    "vmlal.s16  q7, d11, d7[0]  \n"
                    "vmlal.s16  q8, d10, d7[1]  \n"// sum(10-17) += (r21-r28) * k17
                    "vmlal.s16  q9, d11, d7[1]  \n"     
                    "vmlal.s16  q10, d10, d7[2] \n"// sum(20-27) += (r21-r28) * k27
                    "vmlal.s16  q11, d11, d7[2] \n"
                    "vmlal.s16  q12, d10, d7[3] \n"// sum(30-37) += (r21-r28) * k37
                    "vmlal.s16  q13, d11, d7[3] \n"
                    
                    "vld1.s8    {d0}, [%8]      \n"// d0(k08-k38 xx-xx)
                    "add        %8, #4          \n"
                    "vmovl.s8   q0, d0          \n"// d0(k08-k38) d1(xx-xx)

                    "vext.s8    q4, q4, #1      \n"// d8=r22-r29 q4
                    "vmovl.s8   q5, d8          \n"// d10(r22-r25) d11(r26-r29)

                    "vmlal.s16  q6, d10, d0[0]  \n"// sum(00-07) += (r22-r29) * k08
                    "vmlal.s16  q7, d11, d0[0]  \n"
                    "vmlal.s16  q8, d10, d0[1]  \n"// sum(10-17) += (r22-r29) * k18
                    "vmlal.s16  q9, d11, d0[1]  \n"     
                    "vmlal.s16  q10, d10, d0[2] \n"// sum(20-27) += (r22-r29) * k28
                    "vmlal.s16  q11, d11, d0[2] \n"
                    "vmlal.s16  q12, d10, d0[3] \n"// sum(30-37) += (r22-r29) * k38
                    "vmlal.s16  q13, d11, d0[3] \n"
                    
                    "vst1.s32   {d12-d15}, [%1]! \n"// sum00-sum07 q6 q7
                    "vst1.s32   {d16-d19}, [%2]! \n"// sum10-sum17 q8 q9
                    "vst1.s32   {d20-d23}, [%3]! \n"// sum20-sum27 q10 q11
                    "vst1.s32   {d24-d27}, [%4]! \n"// sum30-sum37 q12 q13

                    "sub        %8, #36          \n"
                    "subs       %0, #1           \n"

                    "bne        0b               \n"

                    : "=r"(nn),         // %0
                      "=r"(outptr0),    // %1
                      "=r"(outptr1),    // %2
                      "=r"(outptr2),    // %3
                      "=r"(outptr3),    // %4
                      "=r"(r0),         // %5
                      "=r"(r1),         // %6
                      "=r"(r2),         // %7
                      "=r"(ktmp)        // %8
                    : "0"(nn),
                      "1"(outptr0),
                      "2"(outptr1),
                      "3"(outptr2),
                      "4"(outptr3),
                      "5"(r0),
                      "6"(r1),
                      "7"(r2),
                      "8"(ktmp)
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15" //q14 q15 not be used...
                );
                }
#endif
#if __ARM_NEON
                if (remain >= 4)
                {
                    remain -= 4;
                asm volatile(
                    "vld1.s8    {d0-d3}, [%7]!  \n"// d0=(k00-k30 k01-k31) d1=(k02-k32 k03-k33) d2=(k04-k34 k05-k35) d3=(k06-k36 k07-k37)
                    // r0
                    "vld1.s8    {d8}, [%4]      \n"// d8=r00-r07
                    "add        %4, #4          \n"
                    "vld1.s32   {d12-d13}, [%0] \n"// sum00-sum03 q6
                    "vld1.s32   {d16-d17}, [%1] \n"// sum10-sum13 q8
                    "vld1.s32   {d20-d21}, [%2] \n"// sum20-sum23 q10
                    "vld1.s32   {d24-d25}, [%3] \n"// sum30-sum33 q12
                    
                    "vmovl.s8   q3, d3          \n"// d6(k06-k36) d7(k07-k37) 
                    "vmovl.s8   q2, d2          \n"// d4(k04-k34) d5(k05-k35)
                    "vmovl.s8   q1, d1          \n"// d2(k02-k32) d3(k03-k33)
                    "vmovl.s8   q0, d0          \n"// d0(k00-k30) d1(k01-k31)
                    "vmovl.s8   q5, d8          \n"// d10(r00-r03)
                    
                    "vmlal.s16  q6, d10, d0[0]  \n"// sum(00-03) += (r00-r03) * k00
                    "vmlal.s16  q8, d10, d0[1]  \n"// sum(10-13) += (r00-r03) * k10
                    "vmlal.s16  q10, d10, d0[2] \n"// sum(20-23) += (r00-r03) * k20
                    "vmlal.s16  q12, d10, d0[3] \n"// sum(30-33) += (r00-r03) * k30

                    "vext.s8    d8, d8, #1      \n"// d8=r01-r08
                    "vmovl.s8   q5, d8          \n"// d10(r01-r04)
                    
                    "vmlal.s16  q6, d10, d1[0]  \n"// sum(00-03) += (r01-r04) * k01
                    "vmlal.s16  q8, d10, d1[1]  \n"// sum(10-13) += (r01-r04) * k11
                    "vmlal.s16  q10, d10, d1[2] \n"// sum(20-23) += (r01-r04) * k21
                    "vmlal.s16  q12, d10, d1[3] \n"// sum(30-33) += (r01-r04) * k31
                    
                    "vext.s8    d8, d8, #1      \n"// d8=r02-r09
                    "vmovl.s8   q5, d8          \n"// d10(r02-r05)
                    
                    "vmlal.s16  q6, d10, d2[0]  \n"// sum(00-03) += (r02-r05) * k02
                    "vmlal.s16  q8, d10, d2[1]  \n"// sum(10-13) += (r02-r05) * k12
                    "vmlal.s16  q10, d10, d2[2] \n"// sum(20-23) += (r02-r05) * k22
                    "vmlal.s16  q12, d10, d2[3] \n"// sum(30-33) += (r02-r05) * k32
                    
                    // r1
                    "vld1.s8    {d8}, [%5]      \n"// d8=r10-r17
                    "add        %5, #4          \n"
                    "vmovl.s8   q5, d8          \n"// d10(r10-r13)
                    
                    "vmlal.s16  q6, d10, d3[0]  \n"// sum(00-03) += (r10-r13) * k03
                    "vmlal.s16  q8, d10, d3[1]  \n"// sum(10-13) += (r10-r13) * k13
                    "vmlal.s16  q10, d10, d3[2] \n"// sum(20-23) += (r10-r13) * k23
                    "vmlal.s16  q12, d10, d3[3] \n"// sum(30-33) += (r10-r13) * k33
                    
                    "vext.s8    d8, d8, #1      \n"// d8=r11-r18
                    "vmovl.s8   q5, d8          \n"// d10(r11-r14)

                    "vmlal.s16  q6, d10, d4[0]  \n"// sum(00-03) += (r11-r14) * k04
                    "vmlal.s16  q8, d10, d4[1]  \n"// sum(10-13) += (r11-r14) * k14
                    "vmlal.s16  q10, d10, d4[2] \n"// sum(20-23) += (r11-r14) * k24
                    "vmlal.s16  q12, d10, d4[3] \n"// sum(30-33) += (r11-r14) * k34
                    
                    "vext.s8    d8, d8, #1      \n"// d8=r12-r19 q4
                    "vmovl.s8   q5, d8          \n"// d10(r12-r15)

                    "vmlal.s16  q6, d10, d5[0]  \n"// sum(00-03) += (r12-r15) * k05
                    "vmlal.s16  q8, d10, d5[1]  \n"// sum(10-13) += (r12-r15) * k15
                    "vmlal.s16  q10, d10, d5[2] \n"// sum(20-23) += (r12-r15) * k25
                    "vmlal.s16  q12, d10, d5[3] \n"// sum(30-33) += (r12-r15) * k35

                    // r2
                    "vld1.s8    {d8}, [%6]      \n"// d8=r20-r27
                    "add        %6, #4          \n"
                    "vmovl.s8   q5, d8          \n"// d10(r20-r23)

                    "vmlal.s16  q6, d10, d6[0]  \n"// sum(00-03) += (r20-r23) * k06
                    "vmlal.s16  q8, d10, d6[1]  \n"// sum(10-13) += (r20-r23) * k16
                    "vmlal.s16  q10, d10, d6[2] \n"// sum(20-23) += (r20-r23) * k26
                    "vmlal.s16  q12, d10, d6[3] \n"// sum(30-33) += (r20-r23) * k36
                    
                    "vext.s8    q4, q4, #1      \n"// d8=r21-r28 q4
                    "vmovl.s8   q5, d8          \n"// d10(r21-r24)

                    "vmlal.s16  q6, d10, d7[0]  \n"// sum(00-03) += (r21-r24) * k07
                    "vmlal.s16  q8, d10, d7[1]  \n"// sum(10-13) += (r21-r24) * k17
                    "vmlal.s16  q10, d10, d7[2] \n"// sum(20-23) += (r21-r24) * k27
                    "vmlal.s16  q12, d10, d7[3] \n"// sum(30-33) += (r21-r24) * k37
                    
                    "vld1.s8    {d0}, [%7]      \n"// d0(k08-k38 xx-xx)
                    "add        %7, #4          \n"
                    "vmovl.s8   q0, d0          \n"// d0(k08-k38) d1(xx-xx)

                    "vext.s8    d8, d8, #1      \n"// d8=r22-r25
                    "vmovl.s8   q5, d8          \n"// d10(r22-r25)

                    "vmlal.s16  q6, d10, d0[0]  \n"// sum(00-03) += (r22-r25) * k08
                    "vmlal.s16  q8, d10, d0[1]  \n"// sum(10-13) += (r22-r25) * k18
                    "vmlal.s16  q10, d10, d0[2] \n"// sum(20-23) += (r22-r25) * k28
                    "vmlal.s16  q12, d10, d0[3] \n"// sum(30-33) += (r22-r25) * k38
                    
                    "vst1.s32   {d12-d13}, [%0]! \n"// sum00-sum03 q6
                    "vst1.s32   {d16-d17}, [%1]! \n"// sum10-sum13 q8
                    "vst1.s32   {d20-d21}, [%2]! \n"// sum20-sum23 q10
                    "vst1.s32   {d24-d25}, [%3]! \n"// sum30-sum33 q12 

                    "sub        %7, #36          \n"

                    : "=r"(outptr0),    // %0
                      "=r"(outptr1),    // %1
                      "=r"(outptr2),    // %2
                      "=r"(outptr3),    // %3
                      "=r"(r0),         // %4
                      "=r"(r1),         // %5
                      "=r"(r2),         // %6
                      "=r"(ktmp)        // %7
                    : "0"(outptr0),
                      "1"(outptr1),
                      "2"(outptr2),
                      "3"(outptr3),
                      "4"(r0),
                      "5"(r1),
                      "6"(r2),
                      "7"(ktmp)
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13" //q14 q15 not be used...
                );
                }
#endif
                for (; remain>0; remain--)
                {
#if __ARM_NEON
                    asm volatile(
                        "vld1.s8    {d0[]}, [%4]!       \n"// d0(r00)
                        "vld1.s8    {d1[]}, [%4]!       \n"// d1(r01)

                        "vld1.s8    {d4-d7}, [%7]!      \n"// d4(k00-k30 k01-k31) d5(k02-k32 k03-k33) d6(k04-k34 k05-k35) d7(k06-k36 k07-k37)

                        "vsli.64    d0, d1, #32         \n"// d0(r00 r00 r00 r00 r01 r01 r01 r01)

                        "vld1.s8    {d2[]}, [%4]        \n"// d2(r02 r02 r02 r02 r02 r02 r02 r02)
                        "sub        %4, %4, #2          \n"
                        "vld1.s8    {d3[]}, [%5]!       \n"// d3(r10 r10 r10 r10 r10 r10 r10 r10)
                        
                        "vmovl.s8   q5, d7              \n"// d10(k06-k36) d11(k07-k37)
                        "vmovl.s8   q4, d6              \n"// d8(k04-k34) d9(k05-k35)
                        "vmovl.s8   q3, d5              \n"// d6(k02-k32) d7(k03-k33)
                        "vmovl.s8   q2, d4              \n"// d4(k00-k30) d5(k01-k31)
                        
                        "vmovl.s8   q0, d0              \n"// d0(r00 r00 r00 r00) d1(r01 r01 r01 r01)

                        "vsli.64    d2, d3, #32         \n"// d2(r02 r02 r02 r02 r10 r10 r10 r10)
                        
                        "vmull.s16  q8, d0, d4          \n"// (r00) * (k00-k30)
                        "vmull.s16  q9, d1, d5          \n"// (r01) * (k01-k31)
                        
                        "vmovl.s8   q10, d2             \n"// d20(r02 r02 r02 r02) d21(r10 r10 r10 r10)

                        "vld1.s8    {d0[]}, [%5]!       \n"// d0(r11 r11 r11 r11 r11 r11 r11 r11)
                        "vld1.s8    {d1[]}, [%5]        \n"// d1(r12 r12 r12 r12 r12 r12 r12 r12)
                        "sub        %5, %5, #2          \n"

                        "vsli.64    d0, d1, #32         \n"// d0(r11 r11 r11 r11 r12 r12 r12 r12)

                        "vmlal.s16  q8, d20, d6         \n"// (r02) * (k02-k32)
                        "vmlal.s16  q9, d21, d7         \n"// (r10) * (k03-k33)
                        
                        "vmovl.s8   q0, d0              \n"// d0(r11 r11 r11 r11 ) d1(r12 r12 r12 r12)

                        "vld1.s8    {d2[]}, [%6]!       \n"// d2(r20 r20 r20 r20 r20 r20 r20 r20)
                        "vld1.s8    {d3[]}, [%6]!       \n"// d3(r21 r21 r21 r21 r21 r21 r21 r21)

                        "vsli.64    d2, d3, #32         \n"// d2(r20 r20 r20 r20 r21 r21 r21 r21)

                        "vmlal.s16  q8, d0, d8          \n"// (r11) * (k04-k34)
                        "vmlal.s16  q9, d1, d9          \n"// (r12) * (k05-k35)     

                        "vmovl.s8   q2, d2              \n"// d4(r20 r20 r20 r20) d5(r21 r21 r21 r21)

                        "vld1.s8    {d0[]}, [%6]        \n"// d0(r22 r22 r22 r22 r22 r22 r22 r22)
                        "sub        %6, %6, #2          \n"
                        "veor       d1, d1, d1          \n"// d1 = 0

                        "vld1.s8    {d6}, [%7]          \n"// d6 = k08-k38 xxxx
                        "sub        %7, #32             \n"

                        "vsli.64    d0, d1, #32         \n"// d0(r22 r22 r22 r22 0 0 0 0)
                        "vmovl.s8   q4, d6              \n"// d8(k08-k38)
                        "vmovl.s8   q0, d0              \n"// d0(r22 r22 r22 r22) d1(0 0 0 0)

                        "vmlal.s16  q8, d4, d10         \n"// (r20) * (k06-k36)
                        "vmlal.s16  q9, d5, d11         \n"// (r21) * (k07-k37)

                        "vld1.s32   {d20[0]}, [%0]      \n"

                        "vmlal.s16  q8, d0, d8          \n"// (r22) * (k08-k38)

                        "vld1.s32   {d20[1]}, [%1]      \n"

                        "vadd.s32   q8, q8, q9          \n"

                        "vld1.s32   {d21[0]}, [%2]      \n"
                        "vld1.s32   {d21[1]}, [%3]      \n"

                        "vadd.s32   q10, q10, q8        \n"

                        "vst1.s32   {d20[0]}, [%0]!     \n"
                        "vst1.s32   {d20[1]}, [%1]!     \n"
                        "vst1.s32   {d21[0]}, [%2]!     \n"
                        "vst1.s32   {d21[1]}, [%3]!     \n"

                        : "=r"(outptr0),    // %0
                          "=r"(outptr1),    // %1
                          "=r"(outptr2),    // %2
                          "=r"(outptr3),    // %3
                          "=r"(r0),         // %4
                          "=r"(r1),         // %5
                          "=r"(r2),         // %6
                          "=r"(ktmp)        // %7
                        : "0"(outptr0),
                          "1"(outptr1),
                          "2"(outptr2),
                          "3"(outptr3),
                          "4"(r0),
                          "5"(r1),
                          "6"(r2),
                          "7"(ktmp)
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q8", "q9", "q10"
                    );
#else
                    int sum0 = 0;
                    int sum1 = 0;
                    int sum2 = 0;
                    int sum3 = 0;

                    sum0 += r0[0] * ktmp[0];
                    sum1 += r0[0] * ktmp[1];
                    sum2 += r0[0] * ktmp[2];
                    sum3 += r0[0] * ktmp[3];

                    sum0 += r0[1] * ktmp[4];
                    sum1 += r0[1] * ktmp[5];
                    sum2 += r0[1] * ktmp[6];
                    sum3 += r0[1] * ktmp[7];
                    ktmp += 8;

                    sum0 += r0[2] * ktmp[0];
                    sum1 += r0[2] * ktmp[1];
                    sum2 += r0[2] * ktmp[2];
                    sum3 += r0[2] * ktmp[3];

                    sum0 += r1[0] * ktmp[4];
                    sum1 += r1[0] * ktmp[5];
                    sum2 += r1[0] * ktmp[6];
                    sum3 += r1[0] * ktmp[7];
                    ktmp += 8;

                    sum0 += r1[1] * ktmp[0];
                    sum1 += r1[1] * ktmp[1];
                    sum2 += r1[1] * ktmp[2];
                    sum3 += r1[1] * ktmp[3];

                    sum0 += r1[2] * ktmp[4];
                    sum1 += r1[2] * ktmp[5];
                    sum2 += r1[2] * ktmp[6];
                    sum3 += r1[2] * ktmp[7];
                    ktmp += 8;

                    sum0 += r2[0] * ktmp[0];
                    sum1 += r2[0] * ktmp[1];
                    sum2 += r2[0] * ktmp[2];
                    sum3 += r2[0] * ktmp[3];

                    sum0 += r2[1] * ktmp[4];
                    sum1 += r2[1] * ktmp[5];
                    sum2 += r2[1] * ktmp[6];
                    sum3 += r2[1] * ktmp[7];
                    ktmp += 8;

                    sum0 += r2[2] * ktmp[0];
                    sum1 += r2[2] * ktmp[1];
                    sum2 += r2[2] * ktmp[2];
                    sum3 += r2[2] * ktmp[3];
                    ktmp += 8;

                    *outptr0 += sum0;
                    *outptr1 += sum1;
                    *outptr2 += sum2;
                    *outptr3 += sum3;

                    ktmp -= 8*5;

                    outptr0++;
                    outptr1++;
                    outptr2++;
                    outptr3++;
#endif
                    r0++;
                    r1++;
                    r2++;
                }

                r0 += 2;
                r1 += 2;
                r2 += 2;
            }

            ktmp += 4*9;
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=remain_outch_start; p<outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        out0.fill(0);

        const signed char* ktmp = _kernel.channel(p/4 + p%4);

        for (int q=0; q<inch; q++)
        {
            int* outptr0 = out0;
            int* outptr0n = outptr0 + outw;

            const signed char* img0 = bottom_blob.channel(q);

            const signed char* r0 = img0;
            const signed char* r1 = img0 + w;
            const signed char* r2 = img0 + w * 2;
            const signed char* r3 = img0 + w * 3;

            int i = 0;

#if __ARM_NEON
            int8x16_t _k0123456789x = vld1q_s8(ktmp);
            int16x8_t _k_s16 = vmovl_s8(vget_low_s8(_k0123456789x));
            int16x8_t _kn_s16 = vmovl_s8(vget_high_s8(_k0123456789x));

            int16x4_t _k0123 = vget_low_s16(_k_s16);
            int16x4_t _k4567 = vget_high_s16(_k_s16);
            int16x4_t _k8xxx = vget_low_s16(_kn_s16);
#endif // __ARM_NEON

            for (; i+1 < outh; i+=2)
            {
#if __ARM_NEON
                int nn = outw >> 3;
                int remain = outw & 7;
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
                for (; nn >0; nn--)
                {
                    // r0
                    int8x8_t _r0 = vld1_s8(r0);
                    int8x8_t _r0n = vld1_s8(r0+8);
                    int8x8_t _r01 = vext_s8(_r0, _r0n, 1);
                    int8x8_t _r02 = vext_s8(_r0, _r0n, 2);
                    int16x8_t _r0_s16 = vmovl_s8(_r0);   // r00 - r07
                    int16x8_t _r01_s16 = vmovl_s8(_r01); // r01 - r08 
                    int16x8_t _r02_s16 = vmovl_s8(_r02); // r02 - r09

                    int32x4_t _sum0 = vmull_lane_s16(vget_low_s16(_r0_s16), _k0123, 0); // (r00 - r07) * k00
                    int32x4_t _sum0n = vmull_lane_s16(vget_high_s16(_r0_s16), _k0123, 0);

                    int32x4_t _sum1 = vmull_lane_s16(vget_low_s16(_r01_s16), _k0123, 1); // (r01 - r08) * k01
                    int32x4_t _sum1n = vmull_lane_s16(vget_high_s16(_r01_s16), _k0123, 1);

                    int32x4_t _sum2 = vmull_lane_s16(vget_low_s16(_r02_s16), _k0123, 2); // (r02 - r09) * k02
                    int32x4_t _sum2n = vmull_lane_s16(vget_high_s16(_r02_s16), _k0123, 2);

                    // r1
                    int8x8_t _r1 = vld1_s8(r1);
                    int8x8_t _r1n = vld1_s8(r1+8);
                    int8x8_t _r11 = vext_s8(_r1, _r1n, 1);
                    int8x8_t _r12 = vext_s8(_r1, _r1n, 2);
                    int16x8_t _r1_s16 = vmovl_s8(_r1);   // r10 - r17
                    int16x8_t _r11_s16 = vmovl_s8(_r11); // r11 - r18
                    int16x8_t _r12_s16 = vmovl_s8(_r12); // r12 - r19

                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_r1_s16), _k0123, 3); // (r10 - r17) * k03
                    _sum0n = vmlal_lane_s16(_sum0n, vget_high_s16(_r1_s16), _k0123, 3);

                    _sum1 = vmlal_lane_s16(_sum1, vget_low_s16(_r11_s16), _k4567, 0); // (r11 - r18) * k04
                    _sum1n = vmlal_lane_s16(_sum1n, vget_high_s16(_r11_s16), _k4567, 0);

                    _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_r12_s16), _k4567, 1); // (r12 - r19) * k05
                    _sum2n = vmlal_lane_s16(_sum2n, vget_high_s16(_r12_s16), _k4567, 1); 

                    int32x4_t _sum4 = vmull_lane_s16(vget_low_s16(_r1_s16), _k0123, 0); // (r10 - r17) * k00
                    int32x4_t _sum4n = vmull_lane_s16(vget_high_s16(_r1_s16), _k0123, 0);

                    int32x4_t _sum5 = vmull_lane_s16(vget_low_s16(_r11_s16), _k0123, 1); // (r11 - r18) * k01
                    int32x4_t _sum5n = vmull_lane_s16(vget_high_s16(_r11_s16), _k0123, 1);

                    int32x4_t _sum6 = vmull_lane_s16(vget_low_s16(_r12_s16), _k0123, 2); // (r12 - r19) * k02
                    int32x4_t _sum6n = vmull_lane_s16(vget_high_s16(_r12_s16), _k0123, 2);

                    // r2
                    int8x8_t _r2 = vld1_s8(r2);
                    int8x8_t _r2n = vld1_s8(r2+8);
                    int8x8_t _r21 = vext_s8(_r2, _r2n, 1);
                    int8x8_t _r22 = vext_s8(_r2, _r2n, 2);
                    int16x8_t _r2_s16 = vmovl_s8(_r2);   // r20 - r27
                    int16x8_t _r21_s16 = vmovl_s8(_r21); // r21 - r28
                    int16x8_t _r22_s16 = vmovl_s8(_r22); // r22 - r29

                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_r2_s16), _k4567, 2); // (r20 - r27) * k06
                    _sum0n = vmlal_lane_s16(_sum0n, vget_high_s16(_r2_s16), _k4567, 2);

                    _sum1 = vmlal_lane_s16(_sum1, vget_low_s16(_r21_s16), _k4567, 3); // (r21 - r28) * k07
                    _sum1n = vmlal_lane_s16(_sum1n, vget_high_s16(_r21_s16), _k4567, 3);

                    _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_r22_s16), _k8xxx, 0); // (r22 - r29) * k08
                    _sum2n = vmlal_lane_s16(_sum2n, vget_high_s16(_r22_s16), _k8xxx, 0);

                    _sum4 = vmlal_lane_s16(_sum4, vget_low_s16(_r2_s16), _k0123, 3); // (r20 - r27) * k03
                    _sum4n = vmlal_lane_s16(_sum4n, vget_high_s16(_r2_s16), _k0123, 3);

                    _sum5 = vmlal_lane_s16(_sum5, vget_low_s16(_r21_s16), _k4567, 0); // (r21 - r28) * k04
                    _sum5n = vmlal_lane_s16(_sum5n, vget_high_s16(_r21_s16), _k4567, 0);

                    _sum6 = vmlal_lane_s16(_sum6, vget_low_s16(_r22_s16), _k4567, 1); // (r22 - r29) * k05
                    _sum6n = vmlal_lane_s16(_sum6n, vget_high_s16(_r22_s16), _k4567, 1);

                    // load output sum0 sum0n
                    int32x4_t _out00 = vld1q_s32(outptr0);
                    int32x4_t _out01 = vld1q_s32(outptr0+4);
                    int32x4_t _out10 = vld1q_s32(outptr0n);
                    int32x4_t _out11 = vld1q_s32(outptr0n+4);

                    // r3
                    int8x8_t _r3 = vld1_s8(r3);
                    int8x8_t _r3n = vld1_s8(r3+8);
                    int8x8_t _r31 = vext_s8(_r3, _r3n, 1);
                    int8x8_t _r32 = vext_s8(_r3, _r3n, 2);
                    int16x8_t _r3_s16 = vmovl_s8(_r3);   // r30 - r37
                    int16x8_t _r31_s16 = vmovl_s8(_r31); // r31 - r38
                    int16x8_t _r32_s16 = vmovl_s8(_r32); // r32 - r39

                    _sum0 = vaddq_s32(_sum0, _sum1);
                    _sum0n = vaddq_s32(_sum0n, _sum1n);
                    _sum2 = vaddq_s32(_sum2, _sum0);
                    _sum2n = vaddq_s32(_sum2n, _sum0n);

                    _out00 = vaddq_s32(_out00, _sum2);
                    _out01 = vaddq_s32(_out01, _sum2n);

                    vst1q_s32(outptr0, _out00);
                    vst1q_s32(outptr0+4, _out01);

                    _sum4 = vmlal_lane_s16(_sum4, vget_low_s16(_r3_s16), _k4567, 2); // (r30 - r37) * k06
                    _sum4n = vmlal_lane_s16(_sum4n, vget_high_s16(_r3_s16), _k4567, 2);

                    _sum5 = vmlal_lane_s16(_sum5, vget_low_s16(_r31_s16), _k4567, 3); // (r31 - r38) * k07
                    _sum5n = vmlal_lane_s16(_sum5n, vget_high_s16(_r31_s16), _k4567, 3);

                    _sum6 = vmlal_lane_s16(_sum6, vget_low_s16(_r32_s16), _k8xxx, 0); // (r32 - r39) * k08
                    _sum6n = vmlal_lane_s16(_sum6n, vget_high_s16(_r32_s16), _k8xxx, 0); 

                    _sum4 = vaddq_s32(_sum4, _sum5);
                    _sum4n = vaddq_s32(_sum4n, _sum5n);
                    _sum6 = vaddq_s32(_sum6, _sum4);
                    _sum6n = vaddq_s32(_sum6n, _sum4n);

                    _out10 = vaddq_s32(_out10, _sum6);
                    _out11 = vaddq_s32(_out11, _sum6n);

                    vst1q_s32(outptr0n, _out10);
                    vst1q_s32(outptr0n+4, _out11);

                    r0 += 8;
                    r1 += 8;
                    r2 += 8;
                    r3 += 8;
                    outptr0 += 8;
                    outptr0n += 8;
                }
#endif
#if __ARM_NEON
                if (remain >= 4)
                {
                    remain -= 4;

                    // r0
                    int8x8_t _r0 = vld1_s8(r0);
                    int8x8_t _r0n = vld1_s8(r0+8);
                    int8x8_t _r01 = vext_s8(_r0, _r0n, 1);
                    int8x8_t _r02 = vext_s8(_r0, _r0n, 2);
                    int16x8_t _r0_s16 = vmovl_s8(_r0);   // r00 - r07
                    int16x8_t _r01_s16 = vmovl_s8(_r01); // r01 - r08
                    int16x8_t _r02_s16 = vmovl_s8(_r02); // r02 - r09

                    int32x4_t _sum0 = vmull_lane_s16(vget_low_s16(_r0_s16), _k0123, 0); // (r00 - r07) * k00
                    int32x4_t _sum1 = vmull_lane_s16(vget_low_s16(_r01_s16), _k0123, 1); // (r01 - r08) * k01
                    int32x4_t _sum2 = vmull_lane_s16(vget_low_s16(_r02_s16), _k0123, 2); // (r02 - r09) * k02

                    // r1
                    int8x8_t _r1 = vld1_s8(r1);
                    int8x8_t _r1n = vld1_s8(r1+8);
                    int8x8_t _r11 = vext_s8(_r1, _r1n, 1);
                    int8x8_t _r12 = vext_s8(_r1, _r1n, 2);
                    int16x8_t _r1_s16 = vmovl_s8(_r1);   // r10 - r17
                    int16x8_t _r11_s16 = vmovl_s8(_r11); // r11 - r18
                    int16x8_t _r12_s16 = vmovl_s8(_r12); // r12 - r19

                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_r1_s16), _k0123, 3); // (r10 - r17) * k03
                    _sum1 = vmlal_lane_s16(_sum1, vget_low_s16(_r11_s16), _k4567, 0); // (r11 - r18) * k04
                    _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_r12_s16), _k4567, 1); // (r12 - r19) * k05

                    int32x4_t _sum4 = vmull_lane_s16(vget_low_s16(_r1_s16), _k0123, 0); // (r10 - r17) * k00
                    int32x4_t _sum5 = vmull_lane_s16(vget_low_s16(_r11_s16), _k0123, 1); // (r11 - r18) * k01
                    int32x4_t _sum6 = vmull_lane_s16(vget_low_s16(_r12_s16), _k0123, 2); // (r12 - r19) * k02

                    // r2
                    int8x8_t _r2 = vld1_s8(r2);
                    int8x8_t _r2n = vld1_s8(r2+8);
                    int8x8_t _r21 = vext_s8(_r2, _r2n, 1);
                    int8x8_t _r22 = vext_s8(_r2, _r2n, 2);
                    int16x8_t _r2_s16 = vmovl_s8(_r2);   // r20 - r27
                    int16x8_t _r21_s16 = vmovl_s8(_r21); // r21 - r28
                    int16x8_t _r22_s16 = vmovl_s8(_r22); // r22 - r29

                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_r2_s16), _k4567, 2); // (r20 - r27) * k06
                    _sum1 = vmlal_lane_s16(_sum1, vget_low_s16(_r21_s16), _k4567, 3); // (r21 - r28) * k07
                    _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_r22_s16), _k8xxx, 0); // (r22 - r29) * k08

                    _sum4 = vmlal_lane_s16(_sum4, vget_low_s16(_r2_s16), _k0123, 3); // (r20 - r27) * k03
                    _sum5 = vmlal_lane_s16(_sum5, vget_low_s16(_r21_s16), _k4567, 0); // (r21 - r28) * k04
                    _sum6 = vmlal_lane_s16(_sum6, vget_low_s16(_r22_s16), _k4567, 1); // (r22 - r29) * k05

                    // load output sum0 sum0n
                    int32x4_t _out00 = vld1q_s32(outptr0);
                    int32x4_t _out10 = vld1q_s32(outptr0n);

                    // r3
                    int8x8_t _r3 = vld1_s8(r3);
                    int8x8_t _r3n = vld1_s8(r3+8);
                    int8x8_t _r31 = vext_s8(_r3, _r3n, 1);
                    int8x8_t _r32 = vext_s8(_r3, _r3n, 2);
                    int16x8_t _r3_s16 = vmovl_s8(_r3);   // r30 - r37
                    int16x8_t _r31_s16 = vmovl_s8(_r31); // r31 - r38
                    int16x8_t _r32_s16 = vmovl_s8(_r32); // r32 - r39

                    _sum0 = vaddq_s32(_sum0, _sum1);
                    _sum2 = vaddq_s32(_sum2, _sum0);
                    _out00 = vaddq_s32(_out00, _sum2);

                    vst1q_s32(outptr0, _out00);

                    _sum4 = vmlal_lane_s16(_sum4, vget_low_s16(_r3_s16), _k4567, 2); // (r30 - r37) * k06
                    _sum5 = vmlal_lane_s16(_sum5, vget_low_s16(_r31_s16), _k4567, 3); // (r31 - r38) * k07
                    _sum6 = vmlal_lane_s16(_sum6, vget_low_s16(_r32_s16), _k8xxx, 0); // (r32 - r39) * k08

                    _sum4 = vaddq_s32(_sum4, _sum5);
                    _sum6 = vaddq_s32(_sum6, _sum4);

                    _out10 = vaddq_s32(_out10, _sum6);

                    vst1q_s32(outptr0n, _out10);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    outptr0 += 4;
                    outptr0n += 4;
                }
#endif
                for (; remain>0; remain--)
                {
#if __ARM_NEON
                    asm volatile(
                        "vld1.s8    {d0[0]}, [%2]!  \n"
                        "vld1.s8    {d0[1]}, [%2]!  \n"
                        "vld1.s8    {d0[2]}, [%2]   \n"
                        "sub        %2, #2          \n"

                        "vld1.s8    {d0[3]}, [%3]!  \n"
                        "vld1.s8    {d0[4]}, [%3]!  \n"
                        "vld1.s8    {d0[5]}, [%3]   \n"
                        "sub        %3, #2          \n"

                        "vld1.s8    {d0[6]}, [%4]!  \n"
                        "vld1.s8    {d0[7]}, [%4]!  \n"// d0(r00 r01 r02 r10 r11 r12 r20 r21)

                        "vld1.s8    {d4[]}, [%4]    \n"// d4(r22 r22 r22 r22 r22 r22 r22 r22) 
                        "sub        %4, #2          \n"

                        "vext.s8    d1, d0, d4, #3  \n"// d1(r10 r11 r12 r22 r21 r22 r22 r22)

                        "vld1.s8    {d1[6]}, [%5]!  \n"
                        "vld1.s8    {d1[7]}, [%5]!  \n"// d1(r10 r11 r12 r22 r21 r22 r30 r31)

                        "vld1.s8    {d2}, [%6]!     \n"// d2(k00 k01 k02 k10 k11 k12 k20 k21)

                        "vld1.s8    {d5[]}, [%5]    \n"// d5(r32 r32 r32 r32 r32 r32 r32 r32)
                        "sub        %5, #2          \n"

                        "veor       d3, d1, d1      \n"// d3(00 00 00 00 00 00 00 00)

                        "vmull.s8   q8, d0, d2      \n"// sum0 = (r00 - r21) * (k00 - k21)
                        "vmull.s8   q9, d1, d2      \n"// sum1 = (r10 - r31) * (k00 - k21)

                        "vld1.s8    {d3[0]}, [%6]   \n"// d3(k22 00 00 00 00 00 00 00)
                        "sub        %6, #8          \n"

                        "vmull.s8   q10, d4, d3     \n"// r22 * k22
                        "vmull.s8   q11, d5, d3     \n"// r22 * k22

                        "vld1.s32   {d6[0]}, [%0]   \n"

                        "vaddl.s16  q10, d16, d18   \n"
                        "vaddl.s16  q11, d18, d22   \n"
                        "vaddw.s16  q10, q10, d17   \n"
                        "vaddw.s16  q11, q11, d19   \n"

                        "vld1.s32   {d6[1]}, [%1]   \n"

                        "vpadd.s32  d20, d20, d21   \n"
                        "vpadd.s32  d22, d22, d23   \n"
                        "vpadd.s32  d20, d20, d22   \n"
                        "vadd.s32   d6, d6, d20     \n"

                        "vst1.s32   {d6[0]}, [%0]!  \n"
                        "vst1.s32   {d6[1]}, [%1]!  \n"

                        : "=r"(outptr0),    // %0
                          "=r"(outptr0n),   // %1
                          "=r"(r0),         // %2
                          "=r"(r1),         // %3
                          "=r"(r2),         // %4
                          "=r"(r3),         // %5
                          "=r"(ktmp)        // %6
                        : "0"(outptr0),
                          "1"(outptr0n),
                          "2"(r0),
                          "3"(r1),
                          "4"(r2),
                          "5"(r3),
                          "6"(ktmp)
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11"
                    );
#else
                    int sum0 = 0;
                    int sum0n = 0;

                    sum0 += r0[0] * ktmp[0];
                    sum0 += r0[1] * ktmp[1];
                    sum0 += r0[2] * ktmp[2];
                    sum0 += r1[0] * ktmp[3];
                    sum0 += r1[1] * ktmp[4];
                    sum0 += r1[2] * ktmp[5];
                    sum0 += r2[0] * ktmp[6];
                    sum0 += r2[1] * ktmp[7];
                    sum0 += r2[2] * ktmp[8];

                    sum0n += r1[0] * ktmp[0];
                    sum0n += r1[1] * ktmp[1];
                    sum0n += r1[2] * ktmp[2];
                    sum0n += r2[0] * ktmp[3];
                    sum0n += r2[1] * ktmp[4];
                    sum0n += r2[2] * ktmp[5];
                    sum0n += r3[0] * ktmp[6];
                    sum0n += r3[1] * ktmp[7];
                    sum0n += r3[2] * ktmp[8];

                    *outptr0 += sum0;
                    *outptr0n += sum0n;

                    outptr0++;
                    outptr0n++;
#endif
                    r0++;
                    r1++;
                    r2++;
                    r3++;
                }

                r0 += 2 + w;
                r1 += 2 + w;
                r2 += 2 + w;
                r3 += 2 + w;

                outptr0 += outw;
                outptr0n += outw;
            }

            for (; i < outh; i++)
            {
#if __ARM_NEON
                int nn = outw >> 3;
                int remain = outw & 7;
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
                for (; nn >0; nn--)
                {
                    // r0
                    int8x8_t _r0 = vld1_s8(r0);
                    int8x8_t _r0n = vld1_s8(r0+8);
                    int8x8_t _r01 = vext_s8(_r0, _r0n, 1);
                    int8x8_t _r02 = vext_s8(_r0, _r0n, 2);
                    int16x8_t _r0_s16 = vmovl_s8(_r0);   // r00 - r07
                    int16x8_t _r01_s16 = vmovl_s8(_r01); // r01 - r08 
                    int16x8_t _r02_s16 = vmovl_s8(_r02); // r02 - r09

                    int32x4_t _sum0 = vmull_lane_s16(vget_low_s16(_r0_s16), _k0123, 0); // (r00 - r07) * k00
                    int32x4_t _sum0n = vmull_lane_s16(vget_high_s16(_r0_s16), _k0123, 0);

                    int32x4_t _sum1 = vmull_lane_s16(vget_low_s16(_r01_s16), _k0123, 1); // (r01 - r08) * k01
                    int32x4_t _sum1n = vmull_lane_s16(vget_high_s16(_r01_s16), _k0123, 1);

                    int32x4_t _sum2 = vmull_lane_s16(vget_low_s16(_r02_s16), _k0123, 2); // (r02 - r09) * k02
                    int32x4_t _sum2n = vmull_lane_s16(vget_high_s16(_r02_s16), _k0123, 2);

                    // r1
                    int8x8_t _r1 = vld1_s8(r1);
                    int8x8_t _r1n = vld1_s8(r1+8);
                    int8x8_t _r11 = vext_s8(_r1, _r1n, 1);
                    int8x8_t _r12 = vext_s8(_r1, _r1n, 2);
                    int16x8_t _r1_s16 = vmovl_s8(_r1);   // r10 - r17
                    int16x8_t _r11_s16 = vmovl_s8(_r11); // r11 - r18
                    int16x8_t _r12_s16 = vmovl_s8(_r12); // r12 - r19

                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_r1_s16), _k0123, 3); // (r10 - r17) * k03
                    _sum0n = vmlal_lane_s16(_sum0n, vget_high_s16(_r1_s16), _k0123, 3);

                    _sum1 = vmlal_lane_s16(_sum1, vget_low_s16(_r11_s16), _k4567, 0); // (r11 - r18) * k04
                    _sum1n = vmlal_lane_s16(_sum1n, vget_high_s16(_r11_s16), _k4567, 0);

                    _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_r12_s16), _k4567, 1); // (r12 - r19) * k05
                    _sum2n = vmlal_lane_s16(_sum2n, vget_high_s16(_r12_s16), _k4567, 1);

                    // r2
                    int8x8_t _r2 = vld1_s8(r2);
                    int8x8_t _r2n = vld1_s8(r2+8);
                    int8x8_t _r21 = vext_s8(_r2, _r2n, 1);
                    int8x8_t _r22 = vext_s8(_r2, _r2n, 2);
                    int16x8_t _r2_s16 = vmovl_s8(_r2);   // r20 - r27
                    int16x8_t _r21_s16 = vmovl_s8(_r21); // r21 - r28
                    int16x8_t _r22_s16 = vmovl_s8(_r22); // r22 - r29

                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_r2_s16), _k4567, 2); // (r20 - r27) * k06
                    _sum0n = vmlal_lane_s16(_sum0n, vget_high_s16(_r2_s16), _k4567, 2);

                    _sum1 = vmlal_lane_s16(_sum1, vget_low_s16(_r21_s16), _k4567, 3); // (r21 - r28) * k07
                    _sum1n = vmlal_lane_s16(_sum1n, vget_high_s16(_r21_s16), _k4567, 3);

                    _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_r22_s16), _k8xxx, 0); // (r22 - r29) * k08
                    _sum2n = vmlal_lane_s16(_sum2n, vget_high_s16(_r22_s16), _k8xxx, 0); 

                    // load output sum0 sum0n
                    int32x4_t _out00 = vld1q_s32(outptr0);
                    int32x4_t _out01 = vld1q_s32(outptr0+4);

                    _sum0 = vaddq_s32(_sum0, _sum1);
                    _sum0n = vaddq_s32(_sum0n, _sum1n);
                    _sum2 = vaddq_s32(_sum2, _sum0);
                    _sum2n = vaddq_s32(_sum2n, _sum0n);

                    _out00 = vaddq_s32(_out00, _sum2);
                    _out01 = vaddq_s32(_out01, _sum2n);

                    vst1q_s32(outptr0, _out00);
                    vst1q_s32(outptr0+4, _out01);

                    r0 += 8;
                    r1 += 8;
                    r2 += 8;
                    r3 += 8;
                    outptr0 += 8;
                    outptr0n += 8;
                }
#endif
                for (; remain>0; remain--)
                {
                    int sum0 = 0;

                    sum0 += r0[0] * ktmp[0];
                    sum0 += r0[1] * ktmp[1];
                    sum0 += r0[2] * ktmp[2];
                    sum0 += r1[0] * ktmp[3];
                    sum0 += r1[1] * ktmp[4];
                    sum0 += r1[2] * ktmp[5];
                    sum0 += r2[0] * ktmp[6];
                    sum0 += r2[1] * ktmp[7];
                    sum0 += r2[2] * ktmp[8];

                    *outptr0 += sum0;

                    r0++;
                    r1++;
                    r2++;
                    outptr0++;
                }

                r0 += 2;
                r1 += 2;
                r2 += 2;
            }

            ktmp += 9;
        }
    }
}

static void conv3x3s2_packed_int8_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2*outw + w;

    int nn_outch = outch >> 3;
    int remain_outch_start = nn_outch << 3;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp=0; pp<nn_outch; pp++)
    {
        int p = pp * 8;

        Mat out0 = top_blob.channel(p+0);
        Mat out1 = top_blob.channel(p+1);
        Mat out2 = top_blob.channel(p+2);
        Mat out3 = top_blob.channel(p+3);
        Mat out4 = top_blob.channel(p+4);
        Mat out5 = top_blob.channel(p+5);
        Mat out6 = top_blob.channel(p+6);
        Mat out7 = top_blob.channel(p+7);

        out0.fill(0);
        out1.fill(0);
        out2.fill(0);
        out3.fill(0);
        out4.fill(0);
        out5.fill(0);
        out6.fill(0);
        out7.fill(0);

        const signed char* ktmp = _kernel.channel(p/8);

        for (int q=0; q<inch; q++)
        {
            int* outptr0 = out0;
            int* outptr1 = out1;
            int* outptr2 = out2;
            int* outptr3 = out3;
            int* outptr4 = out4;
            int* outptr5 = out5;
            int* outptr6 = out6;
            int* outptr7 = out7;

            const signed char* img0 = bottom_blob.channel(q);

            const signed char* r0 = img0;
            const signed char* r1 = img0 + w;
            const signed char* r2 = img0 + w*2;

            int i = 0;

            for (; i < outh; i++)
            {
#if __ARM_NEON
                int nn = outw >> 2;
                int remain = outw & 3;
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                if (nn > 0)
                {
                    // TODO
                }
#else // __aarch64__
                if (nn > 0)
                {
                asm volatile(
                    "0:                             \n"
                    "pld        [%1, #128]          \n"
                    "vld1.s32   {d16-d17}, [%1]     \n"// out0
                    "pld        [%2, #128]          \n"
                    "vld1.s32   {d18-d19}, [%2]     \n"// out1
                    "pld        [%3, #128]          \n"
                    "vld1.s32   {d20-d21}, [%3]     \n"// out2
                    "pld        [%4, #128]          \n"
                    "vld1.s32   {d22-d23}, [%4]     \n"// out3 

                    // r0
                    "pld        [%9, #64]          \n"
                    "vld2.s8    {d8-d9}, [%9]       \n"// d8(a00 a02 a04 a06 a08 a010 a012 a014), d9(a01 a03 a05 a07 a09 a011 a013 a015)
                    "add        %9, #8              \n"
                    "pld        [%12, #64]         \n"
                    "vld1.s8    {d0-d2}, [%12]!     \n"// d0(k00-k70) d1(k01-k71) d2(k02-k72)

                    "pld        [%5, #128]          \n"
                    "vld1.s32   {d24-d25}, [%5]     \n"// out4
                    "pld        [%6, #128]          \n"
                    "vld1.s32   {d26-d27}, [%6]     \n"// out5

                    "vmovl.s8   q2, d2              \n"// q2(k02-k72)
                    "vmovl.s8   q1, d1              \n"// q1(k01-k71)
                    "vmovl.s8   q0, d0              \n"// q0(k00-k70)
                    "vext.s8    d12, d8, d8, #1     \n"// d12(a02 a04 a06 a08 x x x x)

                    "pld        [%7, #128]          \n"
                    "vld1.s32   {d28-d29}, [%7]     \n"// out6

                    "vmovl.s8   q5, d9              \n"// q5(a01 a03 a05 a07 a09 a011 a013 a015) d11
                    "vmovl.s8   q4, d8              \n"// q4(a00 a02 a04 a06 a08 a010 a012 a014) d9
                    "vmovl.s8   q6, d12             \n"// q6(a02 a04 a06 a08 a010 a012 a014 a016) d13

                    "pld        [%8, #128]          \n"
                    "vld1.s32   {d30-d31}, [%8]     \n"// out7

                    "vmlal.s16  q8, d8, d0[0]       \n"// sum0 += (a00 a02 a04 a06) * k00
                    "vmlal.s16  q9, d8, d0[1]       \n"// sum1 += (a00 a02 a04 a06) * k10
                    "vmlal.s16  q10, d8, d0[2]      \n"// sum2 += (a00 a02 a04 a06) * k20
                    "vmlal.s16  q11, d8, d0[3]      \n"// sum3 += (a00 a02 a04 a06) * k30
                    "vmlal.s16  q12, d8, d1[0]      \n"// sum4 += (a00 a02 a04 a06) * k40
                    "vmlal.s16  q13, d8, d1[1]      \n"// sum5 += (a00 a02 a04 a06) * k50
                    "vmlal.s16  q14, d8, d1[2]      \n"// sum6 += (a00 a02 a04 a06) * k60
                    "vmlal.s16  q15, d8, d1[3]      \n"// sum7 += (a00 a02 a04 a06) * k70

                    "vmlal.s16  q8, d10, d2[0]      \n"// sum0 += (a01-a07) * k01
                    "vmlal.s16  q9, d10, d2[1]      \n"// sum1 += (a01-a07) * k11
                    "vmlal.s16  q10, d10, d2[2]     \n"// sum2 += (a01-a07) * k21
                    "vmlal.s16  q11, d10, d2[3]     \n"// sum3 += (a01-a07) * k31
                    "vmlal.s16  q12, d10, d3[0]     \n"// sum4 += (a01-a07) * k41
                    "vmlal.s16  q13, d10, d3[1]     \n"// sum5 += (a01-a07) * k51
                    "vmlal.s16  q14, d10, d3[2]     \n"// sum6 += (a01-a07) * k61
                    "vmlal.s16  q15, d10, d3[3]     \n"// sum7 += (a01-a07) * k71

                    "pld        [%10, #64]         \n"
                    "vld2.s8    {d8-d9}, [%10]      \n"// d8(a10 a12 a14 a16 a18 a110 a112 a114), d9(a11 a13 a15 a17 a19 a111 a113 a115)
                    "add        %10, #8             \n"

                    "vmlal.s16  q8, d12, d4[0]      \n"// sum0 += (a02-a08) * k02
                    "vmlal.s16  q9, d12, d4[1]      \n"// sum1 += (a02-a08) * k12
                    "vmlal.s16  q10, d12, d4[2]     \n"// sum2 += (a02-a08) * k22
                    "vmlal.s16  q11, d12, d4[3]     \n"// sum3 += (a02-a08) * k32

                    "pld        [%12, #64]         \n"
                    "vld1.s8    {d0-d2}, [%12]!     \n"// d0(k03-k73) d1(k04-k74) d2(k05-k75)

                    "vmlal.s16  q12, d12, d5[0]     \n"// sum4 += (a02-a08) * k42
                    "vmlal.s16  q13, d12, d5[1]     \n"// sum5 += (a02-a08) * k52
                    "vmlal.s16  q14, d12, d5[2]     \n"// sum6 += (a02-a08) * k62
                    "vmlal.s16  q15, d12, d5[3]     \n"// sum7 += (a02-a08) * k72

                    // r1
                    "vext.s8    d12, d8, d8, #1     \n"// d12(a12 a14 a16 a18 x x x x)

                    "vmovl.s8   q2, d2              \n"// q2(k05-k75)
                    "vmovl.s8   q1, d1              \n"// q1(k04-k74)
                    "vmovl.s8   q0, d0              \n"// q0(k03-k73)
                    "vmovl.s8   q5, d9              \n"// q5(a11-a115)
                    "vmovl.s8   q4, d8              \n"// q4(a10-a114)
                    "vmovl.s8   q6, d12             \n"// q6(a12-a116)

                    "vmlal.s16  q8, d8, d0[0]       \n"// sum0 += (a10-a16) * k03
                    "vmlal.s16  q9, d8, d0[1]       \n"// sum1 += (a10-a16) * k13
                    "vmlal.s16  q10, d8, d0[2]      \n"// sum2 += (a10-a16) * k23
                    "vmlal.s16  q11, d8, d0[3]      \n"// sum3 += (a10-a16) * k33
                    "vmlal.s16  q12, d8, d1[0]      \n"// sum4 += (a10-a16) * k43
                    "vmlal.s16  q13, d8, d1[1]      \n"// sum5 += (a10-a16) * k53
                    "vmlal.s16  q14, d8, d1[2]      \n"// sum6 += (a10-a16) * k63
                    "vmlal.s16  q15, d8, d1[3]      \n"// sum7 += (a10-a16) * k73

                    "vmlal.s16  q8, d10, d2[0]      \n"// sum0 += (a11-a17) * k04
                    "vmlal.s16  q9, d10, d2[1]      \n"// sum1 += (a11-a17) * k14
                    "vmlal.s16  q10, d10, d2[2]     \n"// sum2 += (a11-a17) * k24
                    "vmlal.s16  q11, d10, d2[3]     \n"// sum3 += (a11-a17) * k34
                    "vmlal.s16  q12, d10, d3[0]     \n"// sum4 += (a11-a17) * k44
                    "vmlal.s16  q13, d10, d3[1]     \n"// sum5 += (a11-a17) * k54
                    "vmlal.s16  q14, d10, d3[2]     \n"// sum6 += (a11-a17) * k64
                    "vmlal.s16  q15, d10, d3[3]     \n"// sum7 += (a11-a17) * k74

                    "pld        [%11, #64]         \n"
                    "vld2.s8    {d8-d9}, [%11]      \n"// d8(a20 a22 a24 a26 a28 a210 a212 a214), d9(a21 a23 a25 a27 a29 a211 a213 a215)
                    "add        %11, #8             \n"

                    "vmlal.s16  q8, d12, d4[0]      \n"// sum0 += (a12-a18) * k05
                    "vmlal.s16  q9, d12, d4[1]      \n"// sum1 += (a12-a18) * k15
                    "vmlal.s16  q10, d12, d4[2]     \n"// sum2 += (a12-a18) * k25
                    "vmlal.s16  q11, d12, d4[3]     \n"// sum3 += (a12-a18) * k35

                    "pld        [%12, #64]         \n"
                    "vld1.s8    {d0-d2}, [%12]!     \n"// d0(k06-k76) d1(k07-k77) d2(k08-k78)

                    "vmlal.s16  q12, d12, d5[0]     \n"// sum4 += (a12-a18) * k45
                    "vmlal.s16  q13, d12, d5[1]     \n"// sum5 += (a12-a18) * k55
                    "vmlal.s16  q14, d12, d5[2]     \n"// sum6 += (a12-a18) * k65
                    "vmlal.s16  q15, d12, d5[3]     \n"// sum7 += (a12-a18) * k75

                    // r2
                    "vext.s8    d12, d8, d8, #1     \n"// d12(a22 a24 a26 a28 x x x x)
                    
                    "vmovl.s8   q2, d2              \n"// q2(k08-k78)
                    "vmovl.s8   q1, d1              \n"// q1(k07-k77)
                    "vmovl.s8   q0, d0              \n"// q0(k06-k76) 
                    "vmovl.s8   q5, d9              \n"// q5(a21-a215)
                    "vmovl.s8   q4, d8              \n"// q4(a20-a214)
                    "vmovl.s8   q6, d12             \n"// q6(a22-a216)

                    "vmlal.s16  q8, d8, d0[0]       \n"// sum0 += (a20-a26) * k06
                    "vmlal.s16  q9, d8, d0[1]       \n"// sum1 += (a20-a26) * k16
                    "vmlal.s16  q10, d8, d0[2]      \n"// sum2 += (a20-a26) * k26
                    "vmlal.s16  q11, d8, d0[3]      \n"// sum3 += (a20-a26) * k36
                    "vmlal.s16  q12, d8, d1[0]      \n"// sum4 += (a20-a26) * k46
                    "vmlal.s16  q13, d8, d1[1]      \n"// sum5 += (a20-a26) * k56
                    "vmlal.s16  q14, d8, d1[2]      \n"// sum6 += (a20-a26) * k66
                    "vmlal.s16  q15, d8, d1[3]      \n"// sum7 += (a20-a26) * k76

                    "vmlal.s16  q8, d10, d2[0]      \n"// sum0 += (a21-a27) * k07
                    "vmlal.s16  q9, d10, d2[1]      \n"// sum1 += (a21-a27) * k17
                    "vmlal.s16  q10, d10, d2[2]     \n"// sum2 += (a21-a27) * k27
                    "vmlal.s16  q11, d10, d2[3]     \n"// sum3 += (a21-a27) * k37
                    "vmlal.s16  q12, d10, d3[0]     \n"// sum4 += (a21-a27) * k47
                    "vmlal.s16  q13, d10, d3[1]     \n"// sum5 += (a21-a27) * k57
                    "vmlal.s16  q14, d10, d3[2]     \n"// sum6 += (a21-a27) * k67
                    "vmlal.s16  q15, d10, d3[3]     \n"// sum7 += (a21-a27) * k77

                    "vmlal.s16  q8, d12, d4[0]      \n"// sum0 += (a22-a28) * k08
                    "vmlal.s16  q9, d12, d4[1]      \n"// sum1 += (a22-a28) * k18
                    "vmlal.s16  q10, d12, d4[2]     \n"// sum2 += (a22-a28) * k28
                    "vmlal.s16  q11, d12, d4[3]     \n"// sum3 += (a22-a28) * k38
                    "vmlal.s16  q12, d12, d5[0]     \n"// sum4 += (a22-a28) * k48
                    "vmlal.s16  q13, d12, d5[1]     \n"// sum5 += (a22-a28) * k58
                    "vmlal.s16  q14, d12, d5[2]     \n"// sum6 += (a22-a28) * k68
                    "vmlal.s16  q15, d12, d5[3]     \n"// sum7 += (a22-a28) * k78

                    // save s32 to memory
                    "sub        %12, %12, #72       \n"
                    "vst1.s32   {d16-d17}, [%1]!    \n"// out0
                    "vst1.s32   {d18-d19}, [%2]!    \n"// out1
                    "vst1.s32   {d20-d21}, [%3]!    \n"// out2
                    "vst1.s32   {d22-d23}, [%4]!    \n"// out3
                    "subs       %0, #1              \n"
                    "vst1.s32   {d24-d25}, [%5]!    \n"// out4
                    "vst1.s32   {d26-d27}, [%6]!    \n"// out5
                    "vst1.s32   {d28-d29}, [%7]!    \n"// out6
                    "vst1.s32   {d30-d31}, [%8]!    \n"// out7
                                                 
                    "bne        0b                  \n"
                    : "=r"(nn),         // %0
                      "=r"(outptr0),    // %1
                      "=r"(outptr1),    // %2
                      "=r"(outptr2),    // %3
                      "=r"(outptr3),    // %4
                      "=r"(outptr4),    // %5
                      "=r"(outptr5),    // %6
                      "=r"(outptr6),    // %7
                      "=r"(outptr7),    // %8
                      "=r"(r0),         // %9
                      "=r"(r1),         // %10
                      "=r"(r2),         // %11
                      "=r"(ktmp)        // %12
                    : "0"(nn),
                      "1"(outptr0),
                      "2"(outptr1),
                      "3"(outptr2),
                      "4"(outptr3),
                      "5"(outptr4),
                      "6"(outptr5),
                      "7"(outptr6),
                      "8"(outptr7),
                      "9"(r0),
                      "10"(r1),
                      "11"(r2),
                      "12"(ktmp)
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain>0; remain--)
                {
#if __ARM_NEON
#if __aarch64__
                    // TODO
#else // __aarch64__
                    asm volatile(
                        "pld        [%8, #64]          \n"
                        "vld1.s8    {d0}, [%8]         \n"// d0(a00 a01 a02 ....)
                        "pld        [%9, #64]          \n"
                        "vld1.s8    {d2}, [%9]         \n"// d2(a10 a11 a12 ....)
                        "pld        [%10, #64]         \n"
                        "vld1.s8    {d4}, [%10]        \n"// d4(a20 a21 a22 ....)

                        "pld        [%11, #64]         \n"
                        "vld1.s8    {d6-d8}, [%11]!    \n"// d6(k00-k70) d7(k01-k71) d8(k02-k72)

                        "vmovl.s8   q0, d0             \n"// d0(a00 a01 a02 x) 
                        "vmovl.s8   q1, d2             \n"// d2(a10 a11 a12 x)
                        "vmovl.s8   q2, d4             \n"// d4(a20 a21 a22 x)

                        "vmovl.s8   q5, d8             \n"// d10(k02-k32) d11(k42-k72)
                        "vmovl.s8   q4, d7             \n"// d8(k01-k31) d9(k41-k71)
                        "vmovl.s8   q3, d6             \n"// d6(k00-k30) d7(k40-k70)

                        "vld1.s32   {d20[0]}, [%0]     \n"// out0 q10
                        "vld1.s32   {d20[1]}, [%1]     \n"// out1
                        "vld1.s32   {d21[0]}, [%2]     \n"// out2 
                        "vld1.s32   {d21[1]}, [%3]     \n"// out3

                        "pld        [%11, #64]         \n"
                        "vld1.s8    {d24-d26}, [%11]!  \n"
                        "vmovl.s8   q14, d26           \n"// d28(k05-k35) d29(k45-k75)
                        "vmovl.s8   q13, d25           \n"// d26(k04-k34) d27(k44-k74)
                        "vmovl.s8   q12, d24           \n"// d24(k03-k33) d25(k43-k73)

                        "vld1.s32   {d22[0]}, [%4]     \n"// out4 q11
                        "vld1.s32   {d22[1]}, [%5]     \n"// out5
                        "vld1.s32   {d23[0]}, [%6]     \n"// out6
                        "vld1.s32   {d23[1]}, [%7]     \n"// out7

                        "vmull.s16  q6, d6, d0[0]      \n"// a00 x (k00-k30)
                        "vmull.s16  q7, d7, d0[0]      \n"// a00 x (k40-k70)
                        "vmull.s16  q8, d8, d0[1]      \n"// a01 x (k01-k31)
                        "vmull.s16  q9, d9, d0[1]      \n"// a01 x (k41-k71)
                        "vmlal.s16  q10, d10, d0[2]    \n"// a02 x (k02-k32)
                        "vmlal.s16  q11, d11, d0[2]    \n"// a02 x (k42-k72)

                        "pld        [%11, #64]         \n"
                        "vld1.s8    {d6-d8}, [%11]!    \n"
                        "vmovl.s8   q5, d8             \n"// d10(k08-k38) d11(k48-k78)
                        "vmovl.s8   q4, d7             \n"// d8(k07-k37) d9(k47-k77)
                        "vmovl.s8   q3, d6             \n"// d6(k06-k36) d7(k46-k76)

                        "vmlal.s16  q6, d24, d2[0]     \n"// a10 x (k03-k33)
                        "vmlal.s16  q7, d25, d2[0]     \n"// a10 x (k43-k73)
                        "vmlal.s16  q8, d26, d2[1]     \n"// a11 x (k04-k34)
                        "vmlal.s16  q9, d27, d2[1]     \n"// a11 x (k44-k74)
                        "vmlal.s16  q10, d28, d2[2]    \n"// a12 x (k05-k35)
                        "vmlal.s16  q11, d29, d2[2]    \n"// a12 x (k45-k75)

                        "vmlal.s16  q6, d6, d4[0]      \n"// a20 x (k06-k36)
                        "vmlal.s16  q7, d7, d4[0]      \n"// a20 x (k46-k76)
                        "vmlal.s16  q8, d8, d4[1]      \n"// a21 x (k07-k37)
                        "vmlal.s16  q9, d9, d4[1]      \n"// a21 x (k47-k77)
                        "vmlal.s16  q10, d10, d4[2]    \n"// a22 x (k08-k38)
                        "vmlal.s16  q11, d11, d4[2]    \n"// a22 x (k48-k78)

                        "vadd.s32   q8, q8, q6         \n"
                        "vadd.s32   q9, q9, q7         \n"

                        "sub        %11, %11, #72      \n"

                        "vadd.s32   q10, q10, q8       \n"
                        "vadd.s32   q11, q11, q9       \n"

                        "vst1.s32   {d20[0]}, [%0]!    \n"// out0
                        "vst1.s32   {d20[1]}, [%1]!    \n"// out1
                        "vst1.s32   {d21[0]}, [%2]!    \n"// out2
                        "vst1.s32   {d21[1]}, [%3]!    \n"// out3
                        "vst1.s32   {d22[0]}, [%4]!    \n"// out4
                        "vst1.s32   {d22[1]}, [%5]!    \n"// out5
                        "vst1.s32   {d23[0]}, [%6]!    \n"// out6
                        "vst1.s32   {d23[1]}, [%7]!    \n"// out7

                        : "=r"(outptr0),    // %0
                          "=r"(outptr1),    // %1
                          "=r"(outptr2),    // %2
                          "=r"(outptr3),    // %3
                          "=r"(outptr4),    // %4
                          "=r"(outptr5),    // %5
                          "=r"(outptr6),    // %6
                          "=r"(outptr7),    // %7
                          "=r"(r0),         // %8
                          "=r"(r1),         // %9
                          "=r"(r2),         // %10
                          "=r"(ktmp)        // %11
                        : "0"(outptr0),
                          "1"(outptr1),
                          "2"(outptr2),
                          "3"(outptr3),
                          "4"(outptr4),
                          "5"(outptr5),
                          "6"(outptr6),
                          "7"(outptr7),
                          "8"(r0),
                          "9"(r1),
                          "10"(r2),
                          "11"(ktmp)
                        : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                    );
#endif // __aarch64__
#else // __ARM_NEON
                    int sum0 = 0;
                    int sum1 = 0;
                    int sum2 = 0;
                    int sum3 = 0;
                    int sum4 = 0;
                    int sum5 = 0;
                    int sum6 = 0;
                    int sum7 = 0;

                    sum0 += (int)r0[0] * ktmp[0];
                    sum1 += (int)r0[0] * ktmp[1];
                    sum2 += (int)r0[0] * ktmp[2];
                    sum3 += (int)r0[0] * ktmp[3];
                    sum4 += (int)r0[0] * ktmp[4];
                    sum5 += (int)r0[0] * ktmp[5];
                    sum6 += (int)r0[0] * ktmp[6];
                    sum7 += (int)r0[0] * ktmp[7];
                    ktmp += 8;

                    sum0 += (int)r0[1] * ktmp[0];
                    sum1 += (int)r0[1] * ktmp[1];
                    sum2 += (int)r0[1] * ktmp[2];
                    sum3 += (int)r0[1] * ktmp[3];
                    sum4 += (int)r0[1] * ktmp[4];
                    sum5 += (int)r0[1] * ktmp[5];
                    sum6 += (int)r0[1] * ktmp[6];
                    sum7 += (int)r0[1] * ktmp[7];
                    ktmp += 8;

                    sum0 += (int)r0[2] * ktmp[0];
                    sum1 += (int)r0[2] * ktmp[1];
                    sum2 += (int)r0[2] * ktmp[2];
                    sum3 += (int)r0[2] * ktmp[3];
                    sum4 += (int)r0[2] * ktmp[4];
                    sum5 += (int)r0[2] * ktmp[5];
                    sum6 += (int)r0[2] * ktmp[6];
                    sum7 += (int)r0[2] * ktmp[7];
                    ktmp += 8;

                    sum0 += (int)r1[0] * ktmp[0];
                    sum1 += (int)r1[0] * ktmp[1];
                    sum2 += (int)r1[0] * ktmp[2];
                    sum3 += (int)r1[0] * ktmp[3];
                    sum4 += (int)r1[0] * ktmp[4];
                    sum5 += (int)r1[0] * ktmp[5];
                    sum6 += (int)r1[0] * ktmp[6];
                    sum7 += (int)r1[0] * ktmp[7];
                    ktmp += 8;

                    sum0 += (int)r1[1] * ktmp[0];
                    sum1 += (int)r1[1] * ktmp[1];
                    sum2 += (int)r1[1] * ktmp[2];
                    sum3 += (int)r1[1] * ktmp[3];
                    sum4 += (int)r1[1] * ktmp[4];
                    sum5 += (int)r1[1] * ktmp[5];
                    sum6 += (int)r1[1] * ktmp[6];
                    sum7 += (int)r1[1] * ktmp[7];
                    ktmp += 8;

                    sum0 += (int)r1[2] * ktmp[0];
                    sum1 += (int)r1[2] * ktmp[1];
                    sum2 += (int)r1[2] * ktmp[2];
                    sum3 += (int)r1[2] * ktmp[3];
                    sum4 += (int)r1[2] * ktmp[4];
                    sum5 += (int)r1[2] * ktmp[5];
                    sum6 += (int)r1[2] * ktmp[6];
                    sum7 += (int)r1[2] * ktmp[7];
                    ktmp += 8;

                    sum0 += (int)r2[0] * ktmp[0];
                    sum1 += (int)r2[0] * ktmp[1];
                    sum2 += (int)r2[0] * ktmp[2];
                    sum3 += (int)r2[0] * ktmp[3];
                    sum4 += (int)r2[0] * ktmp[4];
                    sum5 += (int)r2[0] * ktmp[5];
                    sum6 += (int)r2[0] * ktmp[6];
                    sum7 += (int)r2[0] * ktmp[7];
                    ktmp += 8;

                    sum0 += (int)r2[1] * ktmp[0];
                    sum1 += (int)r2[1] * ktmp[1];
                    sum2 += (int)r2[1] * ktmp[2];
                    sum3 += (int)r2[1] * ktmp[3];
                    sum4 += (int)r2[1] * ktmp[4];
                    sum5 += (int)r2[1] * ktmp[5];
                    sum6 += (int)r2[1] * ktmp[6];
                    sum7 += (int)r2[1] * ktmp[7];
                    ktmp += 8;

                    sum0 += (int)r2[2] * ktmp[0];
                    sum1 += (int)r2[2] * ktmp[1];
                    sum2 += (int)r2[2] * ktmp[2];
                    sum3 += (int)r2[2] * ktmp[3];
                    sum4 += (int)r2[2] * ktmp[4];
                    sum5 += (int)r2[2] * ktmp[5];
                    sum6 += (int)r2[2] * ktmp[6];
                    sum7 += (int)r2[2] * ktmp[7];
                    ktmp += 8;

                    *outptr0 += sum0;
                    *outptr1 += sum1;
                    *outptr2 += sum2;
                    *outptr3 += sum3;
                    *outptr4 += sum4;
                    *outptr5 += sum5;
                    *outptr6 += sum6;
                    *outptr7 += sum7;

                    ktmp -= 8*9;

                    outptr0++;
                    outptr1++;
                    outptr2++;
                    outptr3++;
                    outptr4++;
                    outptr5++;
                    outptr6++;
                    outptr7++;
#endif // __ARM_NEON
                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }

            ktmp += 8*9;
        }
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=remain_outch_start; p<outch; p++)
    {
        Mat out = top_blob.channel(p);

        out.fill(0);

        const signed char* ktmp = _kernel.channel(p/8 + p%8);

        for (int q=0; q<inch; q++)
        {
            int* outptr = out;

            const signed char* img0 = bottom_blob.channel(q);

            const signed char* r0 = img0;
            const signed char* r1 = img0 + w;
            const signed char* r2 = img0 + w*2;

            int i = 0;

            for (; i < outh; i++)
            {
#if __ARM_NEON
                int nn = outw >> 3;
                int remain = outw & 7;
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                // TODO
#else
                if (nn > 0)
                {
                asm volatile(
                    "vld1.s8    {d0-d1}, [%5]       \n"// d0(k0 - k7) d1(k8 ...)
                    "vmovl.s8   q1, d1              \n"// d2(k8 ...)
                    "vmovl.s8   q0, d0              \n"// d0(k0 - k3) d1(k4 - k7)
                    "0:                             \n"
                    "pld        [%2, #192]          \n"
                    "vld2.s8    {d4-d5}, [%2]!      \n"// r0 d4(a00 a02 ... a014) d5(a01 a03 ... a015)
                    "vld2.s8    {d8-d9}, [%2]       \n"//    d8(a016 ....)
                    "vld2.s8    {d10-d11}, [%3]!    \n"// r1 d10(a10 a12 ... a114) d11(a11 a13 ... a115)
                    "vld2.s8    {d14-d15}, [%3]     \n"//    d14(a116 ....)
                    "vld2.s8    {d16-d17}, [%4]!    \n"// r2 d16(a20 a22 ... a214) d17(a21 a23 ... a215)
                    "vld2.s8    {d20-d21}, [%4]     \n"//    d20(a216 ....)
                    "vld1.s32   {d22-d25}, [%1]     \n"// q11(out0 - out3) q12(out4 - out7)

                    "vext.s8    d8, d4, d8, #1      \n"//  d8(a02 a04 ... a016)
                    "vext.s8    d14, d10, d14, #1   \n"// d14(a12 a14 ... a116)
                    "vext.s8    d20, d16, d20, #1   \n"// d20(a22 a24 ... a216)

                    "vmovl.s8   q3, d5              \n"// q3(a01 a03 ... a015)
                    "vmovl.s8   q2, d4              \n"// q2(a00 a02 ... a014)
                    "vmovl.s8   q4, d8              \n"// q4(a02 a04 ... a016)

                    "vmovl.s8   q6, d11             \n"// q6(a11 a13 ... a115)
                    "vmovl.s8   q5, d10             \n"// q5(a10 a12 ... a114)
                    "vmovl.s8   q7, d14             \n"// q7(a12 a14 ... a116)

                    "vmovl.s8   q9, d17             \n"// q9(a21 a23 ... a215)
                    "vmovl.s8   q8, d16             \n"// q8(a20 a22 ... a214)
                    "vmovl.s8   q10, d20            \n"// q10(a22 a24 ... a216)
        
                    "vmlal.s16  q11, d4, d0[0]      \n"// k0
                    "vmlal.s16  q12, d5, d0[0]      \n"
                    "vmull.s16  q13, d6, d0[1]      \n"// k1
                    "vmull.s16  q14, d7, d0[1]      \n"
                    "vmlal.s16  q11, d8, d0[2]      \n"// k2
                    "vmlal.s16  q12, d9, d0[2]      \n"

                    "vmlal.s16  q13, d12, d1[0]     \n"// k4
                    "vmlal.s16  q14, d13, d1[0]     \n"
                    "vmlal.s16  q11, d10, d0[3]     \n"// k3
                    "vmlal.s16  q12, d11, d0[3]     \n"
                    "vmlal.s16  q13, d14, d1[1]     \n"// k5
                    "vmlal.s16  q14, d15, d1[1]     \n"

                    "vmlal.s16  q11, d16, d1[2]     \n"// k6
                    "vmlal.s16  q12, d17, d1[2]     \n"
                    "vmlal.s16  q13, d18, d1[3]     \n"// k7 
                    "vmlal.s16  q14, d19, d1[3]     \n"
                    "vmlal.s16  q11, d20, d2[0]     \n"// k8 
                    "vmlal.s16  q12, d21, d2[0]     \n"

                    "vadd.s32   q11, q11, q13       \n"
                    "vadd.s32   q12, q12, q14       \n"
                    
                    "vst1.32    {d22-d25}, [%1]!    \n"     

                    "subs       %0, #1              \n"
                    "bne        0b                  \n"
                    : "=r"(nn),     // %0
                      "=r"(outptr), // %1
                      "=r"(r0),     // %2
                      "=r"(r1),     // %3
                      "=r"(r2),     // %4
                      "=r"(ktmp)    // %5
                    : "0"(nn),
                      "1"(outptr),
                      "2"(r0),
                      "3"(r1),
                      "4"(r2),
                      "5"(ktmp)
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                if (remain > 0)
                {
#if __ARM_NEON
                    int8x8_t _k01234567s8 = vld1_s8(ktmp);
                    int8x8_t _k8xxxxxxxs8 = vld1_s8(ktmp+8);
                    int8x8_t _k34567xxxs8 = vext_s8(_k01234567s8, _k01234567s8, 3);
                    int8x8_t _k678xxxxxs8 = vext_s8(_k01234567s8, _k8xxxxxxxs8, 6);
                    int16x8_t _k0123_s16 = vmovl_s8(_k01234567s8);
                    int16x8_t _k3456_s16 = vmovl_s8(_k34567xxxs8);
                    int16x8_t _k678x_s16 = vmovl_s8(_k678xxxxxs8);
#endif
                    for (; remain>0; remain--)
                    {
#if __ARM_NEON
                        int8x8_t _r00s8 = vld1_s8(r0);
                        int8x8_t _r10s8 = vld1_s8(r1);
                        int8x8_t _r20s8 = vld1_s8(r2);

                        int16x8_t _r00s16 = vmovl_s8(_r00s8);
                        int16x8_t _r10s16 = vmovl_s8(_r10s8);
                        int16x8_t _r20s16 = vmovl_s8(_r20s8);

                        int32x4_t _sum = vmull_s16(vget_low_s16(_r00s16), vget_low_s16(_k0123_s16));
                        _sum = vmlal_s16(_sum, vget_low_s16(_r10s16), vget_low_s16(_k3456_s16));
                        _sum = vmlal_s16(_sum, vget_low_s16(_r20s16), vget_low_s16(_k678x_s16));

                        _sum = vsetq_lane_s32(*outptr, _sum, 3);

#if __aarch64__
                        *outptr = vaddvq_s32(_sum);
#else
                        int32x2_t _ss = vadd_s32(vget_low_s32(_sum), vget_high_s32(_sum));
                        _ss = vpadd_s32(_ss, _ss);

                        *outptr = vget_lane_s32(_ss, 0);
#endif // __aarch64__
#else
                        int sum = 0;

                        sum += (int)r0[0] * ktmp[0];
                        sum += (int)r0[1] * ktmp[1];
                        sum += (int)r0[2] * ktmp[2];
                        sum += (int)r1[0] * ktmp[3];
                        sum += (int)r1[1] * ktmp[4];
                        sum += (int)r1[2] * ktmp[5];
                        sum += (int)r2[0] * ktmp[6];
                        sum += (int)r2[1] * ktmp[7];
                        sum += (int)r2[2] * ktmp[8];

                        *outptr += sum;
#endif // __ARM_NEON
                        r0 += 2;
                        r1 += 2;
                        r2 += 2;
                        outptr++;
                    }
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }

            ktmp += 9;
        }
    }
}
#endif

static void conv3x3s1_int8_neon(const Mat &bottom_blob, Mat &top_blob, const Mat &_kernel, const Option& opt)
{
    int kernel_w = 3;
    int kernel_h = 3;

    int stride_w = 1;
    int stride_h = 1;

    conv_im2col_sgemm_int8_neon(bottom_blob, top_blob, _kernel, kernel_w, kernel_h, stride_w, stride_h, opt);
}

static void conv3x3s2_int8_neon(const Mat &bottom_blob, Mat &top_blob, const Mat &_kernel, const Option& opt)
{
    int kernel_w = 3;
    int kernel_h = 3;

    int stride_w = 2;
    int stride_h = 2;

    conv_im2col_sgemm_int8_neon(bottom_blob, top_blob, _kernel, kernel_w, kernel_h, stride_w, stride_h, opt);
}
