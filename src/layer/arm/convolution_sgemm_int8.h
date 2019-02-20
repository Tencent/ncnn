// SenseNets is pleased to support the open source community by supporting ncnn available.
//
// Copyright (C) 2019 SenseNets Technology Ltd. All rights reserved.
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

static void conv_im2col_sgemm_int8_neon(const Mat &bottom_blob, Mat &top_blob, const Mat &_kernel, \
            const int kernel_w, const int kernel_h, const int stride_w, const int stride_h, const Option& opt)
{
    // printf("conv7x7s2_int8_neon compute with im2col sgemm\n");
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const signed char *kernel = _kernel;

    // double start = ncnn::get_current_time();
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
    // double end = ncnn::get_current_time();
    // printf("im2col : %8.3f ms\n", end - start);
    // start = ncnn::get_current_time();

    // kernel memory packed 4 x 8
    int kernel_size = kernel_w * kernel_h;
    Mat kernel_tm(4*kernel_size, inch, outch/4 + outch%4, (size_t)1u, opt.workspace_allocator);
    {
        int p=0;
        for (; p+3<outch; p+=4)
        {
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
        for (; p<outch; p++)
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

    // end = ncnn::get_current_time();
    // printf("k_pack : %8.3f ms\n", end - start);
    // start = ncnn::get_current_time();
    // sgemm(int M, int N, int L, float* A, float* B, float* C)
    {
        int M = outch;  // outch
        int N = outw * outh; // outsize or out stride
        int L = kernel_w * kernel_h * inch; // ksize * inch

        signed char* B = (signed char*)bottom_im2col;

        int i=0;
        for (; i+3<M; i=i+4)
        {
            int* output0 = top_blob.channel(i);
            int* output1 = top_blob.channel(i+1);
            int* output2 = top_blob.channel(i+2);
            int* output3 = top_blob.channel(i+3);

            int j=0;
            for (; j+7<N; j=j+8)
            {
                signed char* va = kernel_tm.channel(i/4);
                
#if __ARM_NEON
                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum0n = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);
                int32x4_t _sum1n = vdupq_n_s32(0);
                int32x4_t _sum2 = vdupq_n_s32(0);
                int32x4_t _sum2n = vdupq_n_s32(0);
                int32x4_t _sum3 = vdupq_n_s32(0);
                int32x4_t _sum3n = vdupq_n_s32(0);

                int k=0;
                for (; k+7<L; k=k+8)
                {
                    signed char* vb0 = B + k*N + j;
                    signed char* vb1 = B + (k+1)*N + j;
                    signed char* vb2 = B + (k+2)*N + j;
                    signed char* vb3 = B + (k+3)*N + j;
                    signed char* vb4 = B + (k+4)*N + j;
                    signed char* vb5 = B + (k+5)*N + j;
                    signed char* vb6 = B + (k+6)*N + j;
                    signed char* vb7 = B + (k+7)*N + j;

                    int8x8_t _vacc0_s8 = vld1_s8(va);
                    int8x8_t _vacc1_s8 = vld1_s8(va+8);
                    int8x8_t _vacc2_s8 = vld1_s8(va+16);
                    int8x8_t _vacc3_s8 = vld1_s8(va+24);
                    int16x8_t _vacc0 = vmovl_s8(_vacc0_s8);
                    int16x8_t _vacc1 = vmovl_s8(_vacc1_s8);
                    int16x8_t _vacc2 = vmovl_s8(_vacc2_s8);
                    int16x8_t _vacc3 = vmovl_s8(_vacc3_s8);

                    // k=0
                    int8x8_t _vb_s8 = vld1_s8(vb0);
                    int16x8_t _vb = vmovl_s8(_vb_s8);
                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_vb), vget_low_s16(_vacc0), 0);
                    _sum0n = vmlal_lane_s16(_sum0n, vget_high_s16(_vb), vget_low_s16(_vacc0), 0);
                    _sum1 = vmlal_lane_s16(_sum1, vget_low_s16(_vb), vget_low_s16(_vacc0), 1);
                    _sum1n = vmlal_lane_s16(_sum1n, vget_high_s16(_vb), vget_low_s16(_vacc0), 1);
                    _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_vb), vget_low_s16(_vacc0), 2);
                    _sum2n = vmlal_lane_s16(_sum2n, vget_high_s16(_vb), vget_low_s16(_vacc0), 2);
                    _sum3 = vmlal_lane_s16(_sum3, vget_low_s16(_vb), vget_low_s16(_vacc0), 3);
                    _sum3n = vmlal_lane_s16(_sum3n, vget_high_s16(_vb), vget_low_s16(_vacc0), 3);

                    // k=1
                    _vb_s8 = vld1_s8(vb1);
                    _vb = vmovl_s8(_vb_s8);
                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_vb), vget_high_s16(_vacc0), 0);
                    _sum0n = vmlal_lane_s16(_sum0n, vget_high_s16(_vb), vget_high_s16(_vacc0), 0);
                    _sum1 = vmlal_lane_s16(_sum1, vget_low_s16(_vb), vget_high_s16(_vacc0), 1);
                    _sum1n = vmlal_lane_s16(_sum1n, vget_high_s16(_vb), vget_high_s16(_vacc0), 1);
                    _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_vb), vget_high_s16(_vacc0), 2);
                    _sum2n = vmlal_lane_s16(_sum2n, vget_high_s16(_vb), vget_high_s16(_vacc0), 2);
                    _sum3 = vmlal_lane_s16(_sum3, vget_low_s16(_vb), vget_high_s16(_vacc0), 3);
                    _sum3n = vmlal_lane_s16(_sum3n, vget_high_s16(_vb), vget_high_s16(_vacc0), 3);

                    // k=2
                    _vb_s8 = vld1_s8(vb2);
                    _vb = vmovl_s8(_vb_s8);
                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_vb), vget_low_s16(_vacc1), 0);
                    _sum0n = vmlal_lane_s16(_sum0n, vget_high_s16(_vb), vget_low_s16(_vacc1), 0);
                    _sum1 = vmlal_lane_s16(_sum1, vget_low_s16(_vb), vget_low_s16(_vacc1), 1);
                    _sum1n = vmlal_lane_s16(_sum1n, vget_high_s16(_vb), vget_low_s16(_vacc1), 1);
                    _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_vb), vget_low_s16(_vacc1), 2);
                    _sum2n = vmlal_lane_s16(_sum2n, vget_high_s16(_vb), vget_low_s16(_vacc1), 2);
                    _sum3 = vmlal_lane_s16(_sum3, vget_low_s16(_vb), vget_low_s16(_vacc1), 3);
                    _sum3n = vmlal_lane_s16(_sum3n, vget_high_s16(_vb), vget_low_s16(_vacc1), 3);

                    // k=3
                    _vb_s8 = vld1_s8(vb3);
                    _vb = vmovl_s8(_vb_s8);
                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_vb), vget_high_s16(_vacc1), 0);
                    _sum0n = vmlal_lane_s16(_sum0n, vget_high_s16(_vb), vget_high_s16(_vacc1), 0);
                    _sum1 = vmlal_lane_s16(_sum1, vget_low_s16(_vb), vget_high_s16(_vacc1), 1);
                    _sum1n = vmlal_lane_s16(_sum1n, vget_high_s16(_vb), vget_high_s16(_vacc1), 1);
                    _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_vb), vget_high_s16(_vacc1), 2);
                    _sum2n = vmlal_lane_s16(_sum2n, vget_high_s16(_vb), vget_high_s16(_vacc1), 2);
                    _sum3 = vmlal_lane_s16(_sum3, vget_low_s16(_vb), vget_high_s16(_vacc1), 3);
                    _sum3n = vmlal_lane_s16(_sum3n, vget_high_s16(_vb), vget_high_s16(_vacc1), 3);

                    // k=4
                    _vb_s8 = vld1_s8(vb4);
                    _vb = vmovl_s8(_vb_s8);
                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_vb), vget_low_s16(_vacc2), 0);
                    _sum0n = vmlal_lane_s16(_sum0n, vget_high_s16(_vb), vget_low_s16(_vacc2), 0);
                    _sum1 = vmlal_lane_s16(_sum1, vget_low_s16(_vb), vget_low_s16(_vacc2), 1);
                    _sum1n = vmlal_lane_s16(_sum1n, vget_high_s16(_vb), vget_low_s16(_vacc2), 1);
                    _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_vb), vget_low_s16(_vacc2), 2);
                    _sum2n = vmlal_lane_s16(_sum2n, vget_high_s16(_vb), vget_low_s16(_vacc2), 2);
                    _sum3 = vmlal_lane_s16(_sum3, vget_low_s16(_vb), vget_low_s16(_vacc2), 3);
                    _sum3n = vmlal_lane_s16(_sum3n, vget_high_s16(_vb), vget_low_s16(_vacc2), 3);

                    // k=5
                    _vb_s8 = vld1_s8(vb5);
                    _vb = vmovl_s8(_vb_s8);
                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_vb), vget_high_s16(_vacc2), 0);
                    _sum0n = vmlal_lane_s16(_sum0n, vget_high_s16(_vb), vget_high_s16(_vacc2), 0);
                    _sum1 = vmlal_lane_s16(_sum1, vget_low_s16(_vb), vget_high_s16(_vacc2), 1);
                    _sum1n = vmlal_lane_s16(_sum1n, vget_high_s16(_vb), vget_high_s16(_vacc2), 1);
                    _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_vb), vget_high_s16(_vacc2), 2);
                    _sum2n = vmlal_lane_s16(_sum2n, vget_high_s16(_vb), vget_high_s16(_vacc2), 2);
                    _sum3 = vmlal_lane_s16(_sum3, vget_low_s16(_vb), vget_high_s16(_vacc2), 3);
                    _sum3n = vmlal_lane_s16(_sum3n, vget_high_s16(_vb), vget_high_s16(_vacc2), 3);

                    // k=6
                    _vb_s8 = vld1_s8(vb6);
                    _vb = vmovl_s8(_vb_s8);
                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_vb), vget_low_s16(_vacc3), 0);
                    _sum0n = vmlal_lane_s16(_sum0n, vget_high_s16(_vb), vget_low_s16(_vacc3), 0);
                    _sum1 = vmlal_lane_s16(_sum1, vget_low_s16(_vb), vget_low_s16(_vacc3), 1);
                    _sum1n = vmlal_lane_s16(_sum1n, vget_high_s16(_vb), vget_low_s16(_vacc3), 1);
                    _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_vb), vget_low_s16(_vacc3), 2);
                    _sum2n = vmlal_lane_s16(_sum2n, vget_high_s16(_vb), vget_low_s16(_vacc3), 2);
                    _sum3 = vmlal_lane_s16(_sum3, vget_low_s16(_vb), vget_low_s16(_vacc3), 3);
                    _sum3n = vmlal_lane_s16(_sum3n, vget_high_s16(_vb), vget_low_s16(_vacc3), 3);

                    // k=7
                    _vb_s8 = vld1_s8(vb7);
                    _vb = vmovl_s8(_vb_s8);
                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_vb), vget_high_s16(_vacc3), 0);
                    _sum0n = vmlal_lane_s16(_sum0n, vget_high_s16(_vb), vget_high_s16(_vacc3), 0);
                    _sum1 = vmlal_lane_s16(_sum1, vget_low_s16(_vb), vget_high_s16(_vacc3), 1);
                    _sum1n = vmlal_lane_s16(_sum1n, vget_high_s16(_vb), vget_high_s16(_vacc3), 1);
                    _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_vb), vget_high_s16(_vacc3), 2);
                    _sum2n = vmlal_lane_s16(_sum2n, vget_high_s16(_vb), vget_high_s16(_vacc3), 2);
                    _sum3 = vmlal_lane_s16(_sum3, vget_low_s16(_vb), vget_high_s16(_vacc3), 3);
                    _sum3n = vmlal_lane_s16(_sum3n, vget_high_s16(_vb), vget_high_s16(_vacc3), 3);

                    va += 32;
                }

                for (; k<L; k++)
                {
                    signed char* vb0 = B + k*N + j;

                    int8x8_t _vacc0_s8 = vld1_s8(va);
                    int16x8_t _vacc0 = vmovl_s8(_vacc0_s8);

                    // k=0
                    int8x8_t _vb_s8 = vld1_s8(vb0);
                    int16x8_t _vb = vmovl_s8(_vb_s8);
                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_vb), vget_low_s16(_vacc0), 0);
                    _sum0n = vmlal_lane_s16(_sum0n, vget_high_s16(_vb), vget_low_s16(_vacc0), 0);
                    _sum1 = vmlal_lane_s16(_sum1, vget_low_s16(_vb), vget_low_s16(_vacc0), 1);
                    _sum1n = vmlal_lane_s16(_sum1n, vget_high_s16(_vb), vget_low_s16(_vacc0), 1);
                    _sum2 = vmlal_lane_s16(_sum2, vget_low_s16(_vb), vget_low_s16(_vacc0), 2);
                    _sum2n = vmlal_lane_s16(_sum2n, vget_high_s16(_vb), vget_low_s16(_vacc0), 2);
                    _sum3 = vmlal_lane_s16(_sum3, vget_low_s16(_vb), vget_low_s16(_vacc0), 3);
                    _sum3n = vmlal_lane_s16(_sum3n, vget_high_s16(_vb), vget_low_s16(_vacc0), 3);

                    va += 4;
                }

                vst1q_s32(output0, _sum0);
                vst1q_s32(output0+4, _sum0n);
                vst1q_s32(output1, _sum1);
                vst1q_s32(output1+4, _sum1n);
                vst1q_s32(output2, _sum2);
                vst1q_s32(output2+4, _sum2n);
                vst1q_s32(output3, _sum3);
                vst1q_s32(output3+4, _sum3n);                           
#else
                int sum0[8] = {0};
                int sum1[8] = {0};
                int sum2[8] = {0};
                int sum3[8] = {0};
               
                int k=0;
                for (; k+7<L; k=k+8)
                {
                    signed char* vb0 = B + k*N + j;
                    signed char* vb1 = B + (k+1)*N + j;
                    signed char* vb2 = B + (k+2)*N + j;
                    signed char* vb3 = B + (k+3)*N + j;
                    signed char* vb4 = B + (k+4)*N + j;
                    signed char* vb5 = B + (k+5)*N + j;
                    signed char* vb6 = B + (k+6)*N + j;
                    signed char* vb7 = B + (k+7)*N + j;

                    for (int n=0; n<8; n++)
                    {
                        sum0[n] += (int)va[0] * vb0[n];
                        sum1[n] += (int)va[1] * vb0[n];
                        sum2[n] += (int)va[2] * vb0[n];
                        sum3[n] += (int)va[3] * vb0[n];
                        va += 4;

                        sum0[n] += (int)va[0] * vb1[n];
                        sum1[n] += (int)va[1] * vb1[n];
                        sum2[n] += (int)va[2] * vb1[n];
                        sum3[n] += (int)va[3] * vb1[n];
                        va += 4;

                        sum0[n] += (int)va[0] * vb2[n];
                        sum1[n] += (int)va[1] * vb2[n];
                        sum2[n] += (int)va[2] * vb2[n];
                        sum3[n] += (int)va[3] * vb2[n];
                        va += 4;

                        sum0[n] += (int)va[0] * vb3[n];
                        sum1[n] += (int)va[1] * vb3[n];
                        sum2[n] += (int)va[2] * vb3[n];
                        sum3[n] += (int)va[3] * vb3[n];
                        va += 4;

                        sum0[n] += (int)va[0] * vb4[n];
                        sum1[n] += (int)va[1] * vb4[n];
                        sum2[n] += (int)va[2] * vb4[n];
                        sum3[n] += (int)va[3] * vb4[n];
                        va += 4;

                        sum0[n] += (int)va[0] * vb5[n];
                        sum1[n] += (int)va[1] * vb5[n];
                        sum2[n] += (int)va[2] * vb5[n];
                        sum3[n] += (int)va[3] * vb5[n];
                        va += 4;

                        sum0[n] += (int)va[0] * vb6[n];
                        sum1[n] += (int)va[1] * vb6[n];
                        sum2[n] += (int)va[2] * vb6[n];
                        sum3[n] += (int)va[3] * vb6[n];
                        va += 4;

                        sum0[n] += (int)va[0] * vb7[n];
                        sum1[n] += (int)va[1] * vb7[n];
                        sum2[n] += (int)va[2] * vb7[n];
                        sum3[n] += (int)va[3] * vb7[n];
                        va -= 28;
                    }

                    va += 32;
                }

                for (; k<L; k++)
                {
                    signed char* vb0 = B + k*N + j;

                    for (int n=0; n<8; n++)
                    {
                        sum0[n] += (int)va[0] * vb0[n];
                        sum1[n] += (int)va[1] * vb0[n];
                        sum2[n] += (int)va[2] * vb0[n];
                        sum3[n] += (int)va[3] * vb0[n];
                    }
                    
                    va += 4;
                }

                for (int n=0; n<8; n++)
                {
                    output0[n] = sum0[n];
                    output1[n] = sum1[n];
                    output2[n] = sum2[n];
                    output3[n] = sum3[n];
                }
#endif
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

                signed char* va = kernel_tm.channel(i/4);

                for (int k=0; k<L; k++)
                {
                    signed char* vb0 = B + k*N + j;

                    sum0 += (int)va[0] * vb0[0];
                    sum1 += (int)va[1] * vb0[0];
                    sum2 += (int)va[2] * vb0[0];
                    sum3 += (int)va[3] * vb0[0];

                    va += 4;
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

        for (; i<M; i++)
        {
            int* output = top_blob.channel(i);

            int j=0;
            for (; j+7<N; j=j+8)
            {
                signed char* va = kernel_tm.channel(i/4 + i%4);
#if __ARM_NEON
                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum0n = vdupq_n_s32(0);
#else                
                int sum[8] = {0};
#endif
                int k=0;
                for (; k+7<L; k=k+8)
                {
                    signed char* vb0 = B + k*N + j;
                    signed char* vb1 = B + (k+1)*N + j;
                    signed char* vb2 = B + (k+2)*N + j;
                    signed char* vb3 = B + (k+3)*N + j;
                    signed char* vb4 = B + (k+4)*N + j;
                    signed char* vb5 = B + (k+5)*N + j;
                    signed char* vb6 = B + (k+6)*N + j;
                    signed char* vb7 = B + (k+7)*N + j;
#if __ARM_NEON
                    int8x8_t _vacc0_s8 = vld1_s8(va);
                    int16x8_t _vacc0 = vmovl_s8(_vacc0_s8);

                    // k=0
                    int8x8_t _vb_s8 = vld1_s8(vb0);
                    int16x8_t _vb = vmovl_s8(_vb_s8);
                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_vb), vget_low_s16(_vacc0), 0);
                    _sum0n = vmlal_lane_s16(_sum0n, vget_high_s16(_vb), vget_low_s16(_vacc0), 0);

                    // k=1
                    _vb_s8 = vld1_s8(vb1);
                    _vb = vmovl_s8(_vb_s8);
                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_vb), vget_low_s16(_vacc0), 1);
                    _sum0n = vmlal_lane_s16(_sum0n, vget_high_s16(_vb), vget_low_s16(_vacc0), 1);

                    // k=2
                    _vb_s8 = vld1_s8(vb2);
                    _vb = vmovl_s8(_vb_s8);
                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_vb), vget_low_s16(_vacc0), 2);
                    _sum0n = vmlal_lane_s16(_sum0n, vget_high_s16(_vb), vget_low_s16(_vacc0), 2);

                    // k=3
                    _vb_s8 = vld1_s8(vb3);
                    _vb = vmovl_s8(_vb_s8);
                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_vb), vget_low_s16(_vacc0), 3);
                    _sum0n = vmlal_lane_s16(_sum0n, vget_high_s16(_vb), vget_low_s16(_vacc0), 3);

                    // k=4
                    _vb_s8 = vld1_s8(vb4);
                    _vb = vmovl_s8(_vb_s8);
                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_vb), vget_high_s16(_vacc0), 0);
                    _sum0n = vmlal_lane_s16(_sum0n, vget_high_s16(_vb), vget_high_s16(_vacc0), 0);

                    // k=5
                    _vb_s8 = vld1_s8(vb5);
                    _vb = vmovl_s8(_vb_s8);
                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_vb), vget_high_s16(_vacc0), 1);
                    _sum0n = vmlal_lane_s16(_sum0n, vget_high_s16(_vb), vget_high_s16(_vacc0), 1);

                    // k=6
                    _vb_s8 = vld1_s8(vb6);
                    _vb = vmovl_s8(_vb_s8);
                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_vb), vget_high_s16(_vacc0), 2);
                    _sum0n = vmlal_lane_s16(_sum0n, vget_high_s16(_vb), vget_high_s16(_vacc0), 2);

                    // k=7
                    _vb_s8 = vld1_s8(vb7);
                    _vb = vmovl_s8(_vb_s8);
                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_vb), vget_high_s16(_vacc0), 3);
                    _sum0n = vmlal_lane_s16(_sum0n, vget_high_s16(_vb), vget_high_s16(_vacc0), 3);
#else
                    for (int n=0; n<8; n++)
                    {
                        sum[n] += (int)va[0] * vb0[n];
                        sum[n] += (int)va[1] * vb1[n];
                        sum[n] += (int)va[2] * vb2[n];
                        sum[n] += (int)va[3] * vb3[n];
                        sum[n] += (int)va[4] * vb4[n];
                        sum[n] += (int)va[5] * vb5[n];
                        sum[n] += (int)va[6] * vb6[n];
                        sum[n] += (int)va[7] * vb7[n];
                    }
                    va += 8;
#endif                    
                }

                for (; k<L; k++)
                {
                    signed char* vb0 = B + k*N + j;
#if __ARM_NEON
                    int8x8_t _vacc0_s8 = vld1_s8(va);
                    int16x8_t _vacc0 = vmovl_s8(_vacc0_s8);

                    // k=0
                    int8x8_t _vb_s8 = vld1_s8(vb0);
                    int16x8_t _vb = vmovl_s8(_vb_s8);
                    _sum0 = vmlal_lane_s16(_sum0, vget_low_s16(_vb), vget_low_s16(_vacc0), 0);
                    _sum0n = vmlal_lane_s16(_sum0n, vget_high_s16(_vb), vget_low_s16(_vacc0), 0);
#else
                    for (int n=0; n<8; n++)
                    {
                        sum[n] += (int)va[0] * vb0[n];
                    }
#endif
                    va += 1;
                }
#if __ARM_NEON
                vst1q_s32(output, _sum0);
                vst1q_s32(output+4, _sum0n);
#else
                for (int n=0; n<8; n++)
                {
                    output[n] = sum[n];
                }
#endif
                output += 8;
            }

            for (; j<N; j++)
            {
                int sum = 0;
                signed char* va = kernel_tm.channel(i/4 + i%4);

                for (int k=0; k<L; k++)
                {
                    signed char* vb0 = B + k*N + j;

                    sum += (int)va[0] * vb0[0];

                    va += 1;
                }
                output[0] = sum;

                output++;
            }
        }
    }
    // end = ncnn::get_current_time();
    // printf("sgemm  : %8.3f ms\n", end - start);
}
