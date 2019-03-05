// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

static void conv3x3s1_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const float* kernel = _kernel;
    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=0; p<outch; p++)
    {
        Mat out = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;

        out.fill(bias0);

        for (int q=0; q<inch; q++)
        {
            float* outptr = out;
            float* outptr2 = outptr + outw;

            const float* img0 = bottom_blob.channel(q);

            const float* kernel0 = kernel + p*inch*9  + q*9;

            const float* r0 = img0;
            const float* r1 = img0 + w;
            const float* r2 = img0 + w*2;
            const float* r3 = img0 + w*3;

            const float* k0 = kernel0;
            const float* k1 = kernel0 + 3;
            const float* k2 = kernel0 + 6;

            int i = 0;

            for (; i+1 < outh; i+=2)
            {

                int remain = outw;

                for (; remain>0; remain--)
                {
                    float sum = 0;
                    float sum2 = 0;

                    sum += r0[0] * k0[0];
                    sum += r0[1] * k0[1];
                    sum += r0[2] * k0[2];
                    sum += r1[0] * k1[0];
                    sum += r1[1] * k1[1];
                    sum += r1[2] * k1[2];
                    sum += r2[0] * k2[0];
                    sum += r2[1] * k2[1];
                    sum += r2[2] * k2[2];

                    sum2 += r1[0] * k0[0];
                    sum2 += r1[1] * k0[1];
                    sum2 += r1[2] * k0[2];
                    sum2 += r2[0] * k1[0];
                    sum2 += r2[1] * k1[1];
                    sum2 += r2[2] * k1[2];
                    sum2 += r3[0] * k2[0];
                    sum2 += r3[1] * k2[1];
                    sum2 += r3[2] * k2[2];

                    *outptr += sum;
                    *outptr2 += sum2;

                    r0++;
                    r1++;
                    r2++;
                    r3++;
                    outptr++;
                    outptr2++;
                }

                r0 += 2 + w;
                r1 += 2 + w;
                r2 += 2 + w;
                r3 += 2 + w;

                outptr += outw;
                outptr2 += outw;
            }

            for (; i < outh; i++)
            {
                int remain = outw;

                for (; remain>0; remain--)
                {
                    float sum = 0;

                    sum += r0[0] * k0[0];
                    sum += r0[1] * k0[1];
                    sum += r0[2] * k0[2];
                    sum += r1[0] * k1[0];
                    sum += r1[1] * k1[1];
                    sum += r1[2] * k1[2];
                    sum += r2[0] * k2[0];
                    sum += r2[1] * k2[1];
                    sum += r2[2] * k2[2];

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

}

static void conv3x3s1_winograd23_transform_kernel_sse(const Mat& kernel, Mat& kernel_tm, int inch, int outch)
{
    kernel_tm.create(4*4, inch, outch);

    // G
    const float ktm[4][3] = {
        {   1.0f,     0.0f,     0.0f},
        { 1.0f/2,   1.0f/2,   1.0f/2},
        { 1.0f/2,  -1.0f/2,   1.0f/2},
        {   0.0f,     0.0f,     1.0f}
    };

    #pragma omp parallel for
    for (int p = 0; p<outch; p++)
    {
        for (int q = 0; q<inch; q++)
        {
            const float* kernel0 = (const float*)kernel + p*inch * 9 + q * 9;
            float* kernel_tm0 = kernel_tm.channel(p).row(q);

            // transform kernel
            const float* k0 = kernel0;
            const float* k1 = kernel0 + 3;
            const float* k2 = kernel0 + 6;

            // h
            float tmp[4][3];
            for (int i=0; i<4; i++)
            {
                tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
            }

            // U
            for (int j=0; j<4; j++)
            {
                float* tmpp = &tmp[j][0];

                for (int i=0; i<4; i++)
                {
                    kernel_tm0[j*4 + i] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                }
            }
        }
    }
}

static void conv3x3s1_winograd23_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel_tm, const Mat& _bias, const Option& opt)
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

    const float* bias = _bias;    

    // BEGIN transform input
    Mat bottom_blob_tm;
    {
        int w_tm = outw / 2 * 4;
        int h_tm = outh / 2 * 4;

        int nColBlocks = h_tm/4; // may be the block num in Feathercnn
        int nRowBlocks = w_tm/4;

        const int tiles = nColBlocks * nRowBlocks;

        bottom_blob_tm.create(4*4, tiles, inch, 4u, opt.workspace_allocator);

        // BT
        // const float itm[4][4] = {
        //     {1.0f,  0.0f, -1.0f,  0.0f},
        //     {0.0f,  1.0f,  1.00f, 0.0f},
        //     {0.0f, -1.0f,  1.00f, 0.0f},
        //     {0.0f, -1.0f,  0.00f, 1.0f}
        // };        

        for (int q=0; q<inch; q++)
        {
            const float* img = bottom_blob_bordered.channel(q);
            float* out_tm0 = bottom_blob_tm.channel(q);

            for (int j = 0; j < nColBlocks; j++)
            {
                const float* r0 = img + w * j * 2;
                const float* r1 = r0 + w;
                const float* r2 = r1 + w;
                const float* r3 = r2 + w;

                for (int i = 0; i < nRowBlocks; i++)
                {
                    float d0[4],d1[4],d2[4],d3[4];
                    float w0[4],w1[4],w2[4],w3[4];
                    float t0[4],t1[4],t2[4],t3[4];
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
                    // d = B_t * d_t
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
                        out_tm0[n   ] = d0[n];
                        out_tm0[n+ 4] = d1[n];
                        out_tm0[n+ 8] = d2[n];
                        out_tm0[n+12] = d3[n];
                    }                  

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    r3 += 2;

                    out_tm0 += 16;
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

        int nColBlocks = h_tm/4; // may be the block num in Feathercnn
        int nRowBlocks = w_tm/4;

        const int tiles = nColBlocks * nRowBlocks; 

        top_blob_tm.create(16, tiles, outch, 4u, opt.workspace_allocator);

        int nn_outch = outch >> 2;
        int remain_outch_start = nn_outch << 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp=0; pp<nn_outch; pp++)
        {
            int p = pp * 4;

            Mat out0_tm = top_blob_tm.channel(p);
            Mat out1_tm = top_blob_tm.channel(p+1);
            Mat out2_tm = top_blob_tm.channel(p+2);
            Mat out3_tm = top_blob_tm.channel(p+3);

            const Mat kernel0_tm = kernel_tm.channel(p);
            const Mat kernel1_tm = kernel_tm.channel(p+1);
            const Mat kernel2_tm = kernel_tm.channel(p+2);
            const Mat kernel3_tm = kernel_tm.channel(p+3);

            for (int i=0; i<tiles; i++)
            {
                float* output0_tm = out0_tm.row(i);
                float* output1_tm = out1_tm.row(i);
                float* output2_tm = out2_tm.row(i);
                float* output3_tm = out3_tm.row(i);

                float sum0[16] = {0.0f};
                float sum1[16] = {0.0f};
                float sum2[16] = {0.0f};
                float sum3[16] = {0.0f};

                int q = 0;
                for (; q+3<inch; q+=4)
                {   
                    const float* r0 = bottom_blob_tm.channel(q).row(i);
                    const float* r1 = bottom_blob_tm.channel(q+1).row(i);
                    const float* r2 = bottom_blob_tm.channel(q+2).row(i);
                    const float* r3 = bottom_blob_tm.channel(q+3).row(i);

                    const float* k0 = kernel0_tm.row(q);
                    const float* k1 = kernel1_tm.row(q);
                    const float* k2 = kernel2_tm.row(q);
                    const float* k3 = kernel3_tm.row(q);

                    for (int n=0; n<16; n++)
                    {
                        sum0[n] += r0[n] * k0[n];
                        k0 += 16;
                        sum0[n] += r1[n] * k0[n];
                        k0 += 16;
                        sum0[n] += r2[n] * k0[n];
                        k0 += 16;
                        sum0[n] += r3[n] * k0[n];
                        k0 -= 16 * 3;

                        sum1[n] += r0[n] * k1[n];
                        k1 += 16;
                        sum1[n] += r1[n] * k1[n];
                        k1 += 16;
                        sum1[n] += r2[n] * k1[n];
                        k1 += 16;
                        sum1[n] += r3[n] * k1[n];
                        k1 -= 16 * 3;

                        sum2[n] += r0[n] * k2[n];
                        k2 += 16;
                        sum2[n] += r1[n] * k2[n];
                        k2 += 16;
                        sum2[n] += r2[n] * k2[n];
                        k2 += 16;
                        sum2[n] += r3[n] * k2[n];
                        k2 -= 16 * 3;

                        sum3[n] += r0[n] * k3[n];
                        k3 += 16;
                        sum3[n] += r1[n] * k3[n];
                        k3 += 16;
                        sum3[n] += r2[n] * k3[n];
                        k3 += 16;
                        sum3[n] += r3[n] * k3[n];
                        k3 -= 16 * 3;
                    }
                }

                for (; q<inch; q++)
                {
                    const float* r0 = bottom_blob_tm.channel(q).row(i);

                    const float* k0 = kernel0_tm.row(q);
                    const float* k1 = kernel1_tm.row(q);
                    const float* k2 = kernel2_tm.row(q);
                    const float* k3 = kernel3_tm.row(q);

                    for (int n=0; n<16; n++)
                    {
                        sum0[n] += r0[n] * k0[n];
                        sum1[n] += r0[n] * k1[n];
                        sum2[n] += r0[n] * k2[n];
                        sum3[n] += r0[n] * k3[n];
                    }
                }

                for (int n=0; n<16; n++)
                {
                    output0_tm[n] = sum0[n];
                    output1_tm[n] = sum1[n];
                    output2_tm[n] = sum2[n];
                    output3_tm[n] = sum3[n];
                }
            }
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p=remain_outch_start; p<outch; p++)
        {
            Mat out0_tm = top_blob_tm.channel(p);
            const Mat kernel0_tm = kernel_tm.channel(p);

            for (int i=0; i<tiles; i++)
            {
                float* output0_tm = out0_tm.row(i);

                float sum0[16] = {0.0f};

                int q = 0;
                for (; q+3<inch; q+=4)
                {   
                    const float* r0 = bottom_blob_tm.channel(q).row(i);
                    const float* r1 = bottom_blob_tm.channel(q+1).row(i);
                    const float* r2 = bottom_blob_tm.channel(q+2).row(i);
                    const float* r3 = bottom_blob_tm.channel(q+3).row(i);

                    const float* k0 = kernel0_tm.row(q);
                    const float* k1 = kernel0_tm.row(q+1);
                    const float* k2 = kernel0_tm.row(q+2);
                    const float* k3 = kernel0_tm.row(q+3);

                    for (int n=0; n<16; n++)
                    {
                        sum0[n] += r0[n] * k0[n];
                        sum0[n] += r1[n] * k1[n];
                        sum0[n] += r2[n] * k2[n];
                        sum0[n] += r3[n] * k3[n];
                    }
                }

                for (; q<inch; q++)
                {
                    const float* r0 = bottom_blob_tm.channel(q).row(i);
                    const float* k0 = kernel0_tm.row(q);

                    for (int n=0; n<16; n++)
                    {
                        sum0[n] += r0[n] * k0[n];
                    }             
                }

                for (int n=0; n<16; n++)
                {
                    output0_tm[n] = sum0[n];
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

        int nColBlocks = h_tm/4; // may be the block num in Feathercnn
        int nRowBlocks = w_tm/4;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p=0; p<outch; p++)
        {
            Mat out_tm = top_blob_tm.channel(p);
            Mat out = top_blob_bordered.channel(p);

            const float bias0 = bias ? bias[p] : 0.f;

            for (int j=0; j<nColBlocks; j++)
            {
                float* outRow0 = out.row(j*2);
                float* outRow1 = out.row(j*2+1);

                for(int i=0; i<nRowBlocks; i++)
                {
                    float* out_tile = out_tm.row(j*nRowBlocks + i);

                    float s0[4],s1[4],s2[4],s3[4];
                    float w0[4],w1[4];
                    float d0[2],d1[2],d2[2],d3[2];
                    float o0[2],o1[2];
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
                        o0[n] = d0[n] + d1[n] + d2[n] + bias0;
                        o1[n] = d1[n] - d2[n] + d3[n] + bias0;
                    }
                    // save to top blob tm
                    outRow0[0] = o0[0];
                    outRow0[1] = o0[1];
                    outRow1[0] = o1[0];
                    outRow1[1] = o1[1];

                    outRow0 += 2;
                    outRow1 += 2;      
                }
            }
        }        
    }
    // END transform output 

    // cut result pad
    copy_cut_border(top_blob_bordered, top_blob, 0, top_blob_bordered.h - top_blob.h, 0, top_blob_bordered.w - top_blob.w, opt.blob_allocator, opt.num_threads);
}

static void conv3x3s2_sse(const Mat &bottom_blob, Mat &top_blob, const Mat &_kernel, const Mat& _bias, const Option& opt)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2 * outw + w;

    const float* kernel = _kernel;
    const float* bias = _bias;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        Mat out = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;

        out.fill(bias0);

        for (int q = 0; q < inch; q++)
        {
            float *outptr = out;

            const float *img = bottom_blob.channel(q);
            const float* kernel0 = kernel + p*inch*9  + q*9;

            const float *r0 = img;
            const float *r1 = img + w;
            const float *r2 = img + w * 2;

            const float* k0 = kernel0;
            const float* k1 = kernel0 + 3;
            const float* k2 = kernel0 + 6;

            for (int i = 0; i < outh; i++)
            {
                int remain = outw;

                for (; remain > 0; remain--)
                {
                    float sum = 0;

                    sum += r0[0] * k0[0];
                    sum += r0[1] * k0[1];
                    sum += r0[2] * k0[2];
                    sum += r1[0] * k1[0];
                    sum += r1[1] * k1[1];
                    sum += r1[2] * k1[2];
                    sum += r2[0] * k2[0];
                    sum += r2[1] * k2[1];
                    sum += r2[2] * k2[2];

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
}