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

static void conv1x1s1_sgemm_pack4_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt, int activation_type, const Mat& activation_params)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;
    int outch = top_blob.c;

    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int size = w * h;

    const float* bias = _bias;

    // interleave
    Mat tmp(8, inch, size/8 + (size%8)/4 + (size%4)/2 + size%2, elemsize, elempack, opt.workspace_allocator);
    {
        int nn_size = size >> 3;
        int remain_size_start = nn_size << 3;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii=0; ii<nn_size; ii++)
        {
            int i = ii * 8;

            const float* img0 = bottom_blob.channel(0);
            img0 += i*4;

            float* tmpptr = tmp.channel(i/8);

            for (int q=0; q<inch; q++)
            {
                float32x4_t _r0 = vld1q_f32(img0);
                float32x4_t _r1 = vld1q_f32(img0+4);
                float32x4_t _r2 = vld1q_f32(img0+8);
                float32x4_t _r3 = vld1q_f32(img0+12);
                float32x4_t _r4 = vld1q_f32(img0+16);
                float32x4_t _r5 = vld1q_f32(img0+20);
                float32x4_t _r6 = vld1q_f32(img0+24);
                float32x4_t _r7 = vld1q_f32(img0+28);
                vst1q_f32(tmpptr, _r0);
                vst1q_f32(tmpptr+4, _r1);
                vst1q_f32(tmpptr+8, _r2);
                vst1q_f32(tmpptr+12, _r3);
                vst1q_f32(tmpptr+16, _r4);
                vst1q_f32(tmpptr+20, _r5);
                vst1q_f32(tmpptr+24, _r6);
                vst1q_f32(tmpptr+28, _r7);

                tmpptr += 32;
                img0 += bottom_blob.cstep * 4;
            }
        }

        nn_size = (size - remain_size_start) >> 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii=0; ii<nn_size; ii++)
        {
            int i = remain_size_start + ii * 4;

            const float* img0 = bottom_blob.channel(0);
            img0 += i*4;

            float* tmpptr = tmp.channel(i/8 + (i%8)/4);

            for (int q=0; q<inch; q++)
            {
                float32x4_t _r0 = vld1q_f32(img0);
                float32x4_t _r1 = vld1q_f32(img0+4);
                float32x4_t _r2 = vld1q_f32(img0+8);
                float32x4_t _r3 = vld1q_f32(img0+12);
                vst1q_f32(tmpptr, _r0);
                vst1q_f32(tmpptr+4, _r1);
                vst1q_f32(tmpptr+8, _r2);
                vst1q_f32(tmpptr+12, _r3);

                tmpptr += 16;
                img0 += bottom_blob.cstep * 4;
            }
        }

        remain_size_start += nn_size << 2;
        nn_size = (size - remain_size_start) >> 1;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii=0; ii<nn_size; ii++)
        {
            int i = remain_size_start + ii * 2;

            const float* img0 = bottom_blob.channel(0);
            img0 += i*4;

            float* tmpptr = tmp.channel(i/8 + (i%8)/4 + (i%4)/2);

            for (int q=0; q<inch; q++)
            {
                float32x4_t _r0 = vld1q_f32(img0);
                float32x4_t _r1 = vld1q_f32(img0+4);
                vst1q_f32(tmpptr, _r0);
                vst1q_f32(tmpptr+4, _r1);

                tmpptr += 8;
                img0 += bottom_blob.cstep * 4;
            }
        }

        remain_size_start += nn_size << 1;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i=remain_size_start; i<size; i++)
        {
            const float* img0 = bottom_blob.channel(0);
            img0 += i*4;

            float* tmpptr = tmp.channel(i/8 + (i%8)/4 + (i%4)/2 + i%2);

            for (int q=0; q<inch; q++)
            {
                float32x4_t _r0 = vld1q_f32(img0);
                vst1q_f32(tmpptr, _r0);

                tmpptr += 4;
                img0 += bottom_blob.cstep * 4;
            }
        }
    }

    int nn_outch = 0;
    int remain_outch_start = 0;

#if __ARM_NEON && __aarch64__
    nn_outch = outch >> 1;
    remain_outch_start = nn_outch << 1;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int pp=0; pp<nn_outch; pp++)
    {
        int p = pp * 2;

        Mat out0 = top_blob.channel(p);
        Mat out1 = top_blob.channel(p+1);

        float32x4_t _bias0 = bias ? vld1q_f32(bias + p * 4) : vdupq_n_f32(0.f);
        float32x4_t _bias1 = bias ? vld1q_f32(bias + (p+1) * 4) : vdupq_n_f32(0.f);

        float* outptr0 = out0;
        float* outptr1 = out1;

        int i=0;
        for (; i+7<size; i+=8)
        {
            const float* tmpptr = tmp.channel(i/8);

            float32x4_t _sum0_0 = _bias0;
            float32x4_t _sum1_0 = _bias0;
            float32x4_t _sum2_0 = _bias0;
            float32x4_t _sum3_0 = _bias0;
            float32x4_t _sum4_0 = _bias0;
            float32x4_t _sum5_0 = _bias0;
            float32x4_t _sum6_0 = _bias0;
            float32x4_t _sum7_0 = _bias0;

            float32x4_t _sum0_1 = _bias1;
            float32x4_t _sum1_1 = _bias1;
            float32x4_t _sum2_1 = _bias1;
            float32x4_t _sum3_1 = _bias1;
            float32x4_t _sum4_1 = _bias1;
            float32x4_t _sum5_1 = _bias1;
            float32x4_t _sum6_1 = _bias1;
            float32x4_t _sum7_1 = _bias1;

            const float* kptr0 = (const float*)kernel + p * inch * 16;
            const float* kptr1 = (const float*)kernel + (p+1) * inch * 16;

            for (int q=0; q<inch; q++)
            {
//                 const float* r0 = bottom_blob.channel(q);

//                 float32x4_t _r0 = vld1q_f32(r0 + i*4);
//                 float32x4_t _r1 = vld1q_f32(r0 + (i+1)*4);
//                 float32x4_t _r2 = vld1q_f32(r0 + (i+2)*4);
//                 float32x4_t _r3 = vld1q_f32(r0 + (i+3)*4);

                float32x4_t _r0 = vld1q_f32( tmpptr );
                float32x4_t _r1 = vld1q_f32( tmpptr + 4 );
                float32x4_t _r2 = vld1q_f32( tmpptr + 8 );
                float32x4_t _r3 = vld1q_f32( tmpptr + 12 );
                float32x4_t _r4 = vld1q_f32( tmpptr + 16 );
                float32x4_t _r5 = vld1q_f32( tmpptr + 20 );
                float32x4_t _r6 = vld1q_f32( tmpptr + 24 );
                float32x4_t _r7 = vld1q_f32( tmpptr + 28 );

                float32x4_t _w0_0 = vld1q_f32( kptr0 );
                float32x4_t _w1_0 = vld1q_f32( kptr0 + 4 );
                float32x4_t _w2_0 = vld1q_f32( kptr0 + 8 );
                float32x4_t _w3_0 = vld1q_f32( kptr0 + 12 );

                float32x4_t _w0_1 = vld1q_f32( kptr1 );
                float32x4_t _w1_1 = vld1q_f32( kptr1 + 4 );
                float32x4_t _w2_1 = vld1q_f32( kptr1 + 8 );
                float32x4_t _w3_1 = vld1q_f32( kptr1 + 12 );

                _sum0_0 = vmlaq_laneq_f32(_sum0_0, _w0_0, _r0, 0);
                _sum0_0 = vmlaq_laneq_f32(_sum0_0, _w1_0, _r0, 1);
                _sum0_0 = vmlaq_laneq_f32(_sum0_0, _w2_0, _r0, 2);
                _sum0_0 = vmlaq_laneq_f32(_sum0_0, _w3_0, _r0, 3);
                _sum1_0 = vmlaq_laneq_f32(_sum1_0, _w0_0, _r1, 0);
                _sum1_0 = vmlaq_laneq_f32(_sum1_0, _w1_0, _r1, 1);
                _sum1_0 = vmlaq_laneq_f32(_sum1_0, _w2_0, _r1, 2);
                _sum1_0 = vmlaq_laneq_f32(_sum1_0, _w3_0, _r1, 3);
                _sum2_0 = vmlaq_laneq_f32(_sum2_0, _w0_0, _r2, 0);
                _sum2_0 = vmlaq_laneq_f32(_sum2_0, _w1_0, _r2, 1);
                _sum2_0 = vmlaq_laneq_f32(_sum2_0, _w2_0, _r2, 2);
                _sum2_0 = vmlaq_laneq_f32(_sum2_0, _w3_0, _r2, 3);
                _sum3_0 = vmlaq_laneq_f32(_sum3_0, _w0_0, _r3, 0);
                _sum3_0 = vmlaq_laneq_f32(_sum3_0, _w1_0, _r3, 1);
                _sum3_0 = vmlaq_laneq_f32(_sum3_0, _w2_0, _r3, 2);
                _sum3_0 = vmlaq_laneq_f32(_sum3_0, _w3_0, _r3, 3);

                _sum4_0 = vmlaq_laneq_f32(_sum4_0, _w0_0, _r4, 0);
                _sum4_0 = vmlaq_laneq_f32(_sum4_0, _w1_0, _r4, 1);
                _sum4_0 = vmlaq_laneq_f32(_sum4_0, _w2_0, _r4, 2);
                _sum4_0 = vmlaq_laneq_f32(_sum4_0, _w3_0, _r4, 3);
                _sum5_0 = vmlaq_laneq_f32(_sum5_0, _w0_0, _r5, 0);
                _sum5_0 = vmlaq_laneq_f32(_sum5_0, _w1_0, _r5, 1);
                _sum5_0 = vmlaq_laneq_f32(_sum5_0, _w2_0, _r5, 2);
                _sum5_0 = vmlaq_laneq_f32(_sum5_0, _w3_0, _r5, 3);
                _sum6_0 = vmlaq_laneq_f32(_sum6_0, _w0_0, _r6, 0);
                _sum6_0 = vmlaq_laneq_f32(_sum6_0, _w1_0, _r6, 1);
                _sum6_0 = vmlaq_laneq_f32(_sum6_0, _w2_0, _r6, 2);
                _sum6_0 = vmlaq_laneq_f32(_sum6_0, _w3_0, _r6, 3);
                _sum7_0 = vmlaq_laneq_f32(_sum7_0, _w0_0, _r7, 0);
                _sum7_0 = vmlaq_laneq_f32(_sum7_0, _w1_0, _r7, 1);
                _sum7_0 = vmlaq_laneq_f32(_sum7_0, _w2_0, _r7, 2);
                _sum7_0 = vmlaq_laneq_f32(_sum7_0, _w3_0, _r7, 3);

                _sum0_1 = vmlaq_laneq_f32(_sum0_1, _w0_1, _r0, 0);
                _sum0_1 = vmlaq_laneq_f32(_sum0_1, _w1_1, _r0, 1);
                _sum0_1 = vmlaq_laneq_f32(_sum0_1, _w2_1, _r0, 2);
                _sum0_1 = vmlaq_laneq_f32(_sum0_1, _w3_1, _r0, 3);
                _sum1_1 = vmlaq_laneq_f32(_sum1_1, _w0_1, _r1, 0);
                _sum1_1 = vmlaq_laneq_f32(_sum1_1, _w1_1, _r1, 1);
                _sum1_1 = vmlaq_laneq_f32(_sum1_1, _w2_1, _r1, 2);
                _sum1_1 = vmlaq_laneq_f32(_sum1_1, _w3_1, _r1, 3);
                _sum2_1 = vmlaq_laneq_f32(_sum2_1, _w0_1, _r2, 0);
                _sum2_1 = vmlaq_laneq_f32(_sum2_1, _w1_1, _r2, 1);
                _sum2_1 = vmlaq_laneq_f32(_sum2_1, _w2_1, _r2, 2);
                _sum2_1 = vmlaq_laneq_f32(_sum2_1, _w3_1, _r2, 3);
                _sum3_1 = vmlaq_laneq_f32(_sum3_1, _w0_1, _r3, 0);
                _sum3_1 = vmlaq_laneq_f32(_sum3_1, _w1_1, _r3, 1);
                _sum3_1 = vmlaq_laneq_f32(_sum3_1, _w2_1, _r3, 2);
                _sum3_1 = vmlaq_laneq_f32(_sum3_1, _w3_1, _r3, 3);

                _sum4_1 = vmlaq_laneq_f32(_sum4_1, _w0_1, _r4, 0);
                _sum4_1 = vmlaq_laneq_f32(_sum4_1, _w1_1, _r4, 1);
                _sum4_1 = vmlaq_laneq_f32(_sum4_1, _w2_1, _r4, 2);
                _sum4_1 = vmlaq_laneq_f32(_sum4_1, _w3_1, _r4, 3);
                _sum5_1 = vmlaq_laneq_f32(_sum5_1, _w0_1, _r5, 0);
                _sum5_1 = vmlaq_laneq_f32(_sum5_1, _w1_1, _r5, 1);
                _sum5_1 = vmlaq_laneq_f32(_sum5_1, _w2_1, _r5, 2);
                _sum5_1 = vmlaq_laneq_f32(_sum5_1, _w3_1, _r5, 3);
                _sum6_1 = vmlaq_laneq_f32(_sum6_1, _w0_1, _r6, 0);
                _sum6_1 = vmlaq_laneq_f32(_sum6_1, _w1_1, _r6, 1);
                _sum6_1 = vmlaq_laneq_f32(_sum6_1, _w2_1, _r6, 2);
                _sum6_1 = vmlaq_laneq_f32(_sum6_1, _w3_1, _r6, 3);
                _sum7_1 = vmlaq_laneq_f32(_sum7_1, _w0_1, _r7, 0);
                _sum7_1 = vmlaq_laneq_f32(_sum7_1, _w1_1, _r7, 1);
                _sum7_1 = vmlaq_laneq_f32(_sum7_1, _w2_1, _r7, 2);
                _sum7_1 = vmlaq_laneq_f32(_sum7_1, _w3_1, _r7, 3);

                tmpptr += 32;
                kptr0 += 16;
                kptr1 += 16;
            }

            if (activation_type == 1)
            {
                float32x4_t _zero = vdupq_n_f32(0.f);
                _sum0_0 = vmaxq_f32(_sum0_0, _zero);
                _sum1_0 = vmaxq_f32(_sum1_0, _zero);
                _sum2_0 = vmaxq_f32(_sum2_0, _zero);
                _sum3_0 = vmaxq_f32(_sum3_0, _zero);
                _sum4_0 = vmaxq_f32(_sum4_0, _zero);
                _sum5_0 = vmaxq_f32(_sum5_0, _zero);
                _sum6_0 = vmaxq_f32(_sum6_0, _zero);
                _sum7_0 = vmaxq_f32(_sum7_0, _zero);
                _sum0_1 = vmaxq_f32(_sum0_1, _zero);
                _sum1_1 = vmaxq_f32(_sum1_1, _zero);
                _sum2_1 = vmaxq_f32(_sum2_1, _zero);
                _sum3_1 = vmaxq_f32(_sum3_1, _zero);
                _sum4_1 = vmaxq_f32(_sum4_1, _zero);
                _sum5_1 = vmaxq_f32(_sum5_1, _zero);
                _sum6_1 = vmaxq_f32(_sum6_1, _zero);
                _sum7_1 = vmaxq_f32(_sum7_1, _zero);
            }
            else if (activation_type == 2)
            {
                float32x4_t _zero = vdupq_n_f32(0.f);
                float32x4_t _slope = vdupq_n_f32(activation_params[0]);
                _sum0_0 = vbslq_f32(vcleq_f32(_sum0_0, _zero), vmulq_f32(_sum0_0, _slope), _sum0_0);
                _sum1_0 = vbslq_f32(vcleq_f32(_sum1_0, _zero), vmulq_f32(_sum1_0, _slope), _sum1_0);
                _sum2_0 = vbslq_f32(vcleq_f32(_sum2_0, _zero), vmulq_f32(_sum2_0, _slope), _sum2_0);
                _sum3_0 = vbslq_f32(vcleq_f32(_sum3_0, _zero), vmulq_f32(_sum3_0, _slope), _sum3_0);
                _sum4_0 = vbslq_f32(vcleq_f32(_sum4_0, _zero), vmulq_f32(_sum4_0, _slope), _sum4_0);
                _sum5_0 = vbslq_f32(vcleq_f32(_sum5_0, _zero), vmulq_f32(_sum5_0, _slope), _sum5_0);
                _sum6_0 = vbslq_f32(vcleq_f32(_sum6_0, _zero), vmulq_f32(_sum6_0, _slope), _sum6_0);
                _sum7_0 = vbslq_f32(vcleq_f32(_sum7_0, _zero), vmulq_f32(_sum7_0, _slope), _sum7_0);
                _sum0_1 = vbslq_f32(vcleq_f32(_sum0_1, _zero), vmulq_f32(_sum0_1, _slope), _sum0_1);
                _sum1_1 = vbslq_f32(vcleq_f32(_sum1_1, _zero), vmulq_f32(_sum1_1, _slope), _sum1_1);
                _sum2_1 = vbslq_f32(vcleq_f32(_sum2_1, _zero), vmulq_f32(_sum2_1, _slope), _sum2_1);
                _sum3_1 = vbslq_f32(vcleq_f32(_sum3_1, _zero), vmulq_f32(_sum3_1, _slope), _sum3_1);
                _sum4_1 = vbslq_f32(vcleq_f32(_sum4_1, _zero), vmulq_f32(_sum4_1, _slope), _sum4_1);
                _sum5_1 = vbslq_f32(vcleq_f32(_sum5_1, _zero), vmulq_f32(_sum5_1, _slope), _sum5_1);
                _sum6_1 = vbslq_f32(vcleq_f32(_sum6_1, _zero), vmulq_f32(_sum6_1, _slope), _sum6_1);
                _sum7_1 = vbslq_f32(vcleq_f32(_sum7_1, _zero), vmulq_f32(_sum7_1, _slope), _sum7_1);
            }
            else if (activation_type == 3)
            {
                float32x4_t _min = vdupq_n_f32(activation_params[0]);
                float32x4_t _max = vdupq_n_f32(activation_params[1]);
                _sum0_0 = vmaxq_f32(_sum0_0, _min);
                _sum0_0 = vminq_f32(_sum0_0, _max);
                _sum1_0 = vmaxq_f32(_sum1_0, _min);
                _sum1_0 = vminq_f32(_sum1_0, _max);
                _sum2_0 = vmaxq_f32(_sum2_0, _min);
                _sum2_0 = vminq_f32(_sum2_0, _max);
                _sum3_0 = vmaxq_f32(_sum3_0, _min);
                _sum3_0 = vminq_f32(_sum3_0, _max);
                _sum4_0 = vmaxq_f32(_sum4_0, _min);
                _sum4_0 = vminq_f32(_sum4_0, _max);
                _sum5_0 = vmaxq_f32(_sum5_0, _min);
                _sum5_0 = vminq_f32(_sum5_0, _max);
                _sum6_0 = vmaxq_f32(_sum6_0, _min);
                _sum6_0 = vminq_f32(_sum6_0, _max);
                _sum7_0 = vmaxq_f32(_sum7_0, _min);
                _sum7_0 = vminq_f32(_sum7_0, _max);
                _sum0_1 = vmaxq_f32(_sum0_1, _min);
                _sum0_1 = vminq_f32(_sum0_1, _max);
                _sum1_1 = vmaxq_f32(_sum1_1, _min);
                _sum1_1 = vminq_f32(_sum1_1, _max);
                _sum2_1 = vmaxq_f32(_sum2_1, _min);
                _sum2_1 = vminq_f32(_sum2_1, _max);
                _sum3_1 = vmaxq_f32(_sum3_1, _min);
                _sum3_1 = vminq_f32(_sum3_1, _max);
                _sum4_1 = vmaxq_f32(_sum4_1, _min);
                _sum4_1 = vminq_f32(_sum4_1, _max);
                _sum5_1 = vmaxq_f32(_sum5_1, _min);
                _sum5_1 = vminq_f32(_sum5_1, _max);
                _sum6_1 = vmaxq_f32(_sum6_1, _min);
                _sum6_1 = vminq_f32(_sum6_1, _max);
                _sum7_1 = vmaxq_f32(_sum7_1, _min);
                _sum7_1 = vminq_f32(_sum7_1, _max);
            }
            else if (activation_type == 4)
            {
                float32x4_t _one = vdupq_n_f32(1.f);
                _sum0_0 = vnegq_f32(_sum0_0);
                _sum1_0 = vnegq_f32(_sum1_0);
                _sum2_0 = vnegq_f32(_sum2_0);
                _sum3_0 = vnegq_f32(_sum3_0);
                _sum4_1 = vnegq_f32(_sum4_1);
                _sum5_1 = vnegq_f32(_sum5_1);
                _sum6_1 = vnegq_f32(_sum6_1);
                _sum7_1 = vnegq_f32(_sum7_1);
                _sum0_0 = vnegq_f32(_sum0_0);
                _sum1_0 = vnegq_f32(_sum1_0);
                _sum2_0 = vnegq_f32(_sum2_0);
                _sum3_0 = vnegq_f32(_sum3_0);
                _sum4_1 = vnegq_f32(_sum4_1);
                _sum5_1 = vnegq_f32(_sum5_1);
                _sum6_1 = vnegq_f32(_sum6_1);
                _sum7_1 = vnegq_f32(_sum7_1);
                _sum0_0 = exp_ps(_sum0_0);
                _sum1_0 = exp_ps(_sum1_0);
                _sum2_0 = exp_ps(_sum2_0);
                _sum3_0 = exp_ps(_sum3_0);
                _sum4_1 = exp_ps(_sum4_1);
                _sum5_1 = exp_ps(_sum5_1);
                _sum6_1 = exp_ps(_sum6_1);
                _sum7_1 = exp_ps(_sum7_1);
                _sum0_0 = exp_ps(_sum0_0);
                _sum1_0 = exp_ps(_sum1_0);
                _sum2_0 = exp_ps(_sum2_0);
                _sum3_0 = exp_ps(_sum3_0);
                _sum4_1 = exp_ps(_sum4_1);
                _sum5_1 = exp_ps(_sum5_1);
                _sum6_1 = exp_ps(_sum6_1);
                _sum7_1 = exp_ps(_sum7_1);
                _sum0_0 = vaddq_f32(_sum0_0, _one);
                _sum1_0 = vaddq_f32(_sum1_0, _one);
                _sum2_0 = vaddq_f32(_sum2_0, _one);
                _sum3_0 = vaddq_f32(_sum3_0, _one);
                _sum4_0 = vaddq_f32(_sum4_0, _one);
                _sum5_0 = vaddq_f32(_sum5_0, _one);
                _sum6_0 = vaddq_f32(_sum6_0, _one);
                _sum7_0 = vaddq_f32(_sum7_0, _one);
                _sum0_1 = vaddq_f32(_sum0_1, _one);
                _sum1_1 = vaddq_f32(_sum1_1, _one);
                _sum2_1 = vaddq_f32(_sum2_1, _one);
                _sum3_1 = vaddq_f32(_sum3_1, _one);
                _sum4_1 = vaddq_f32(_sum4_1, _one);
                _sum5_1 = vaddq_f32(_sum5_1, _one);
                _sum6_1 = vaddq_f32(_sum6_1, _one);
                _sum7_1 = vaddq_f32(_sum7_1, _one);
                float32x4_t _outp0_0 = vrecpeq_f32(_sum0_0);
                float32x4_t _outp1_0 = vrecpeq_f32(_sum1_0);
                float32x4_t _outp2_0 = vrecpeq_f32(_sum2_0);
                float32x4_t _outp3_0 = vrecpeq_f32(_sum3_0);
                float32x4_t _outp4_0 = vrecpeq_f32(_sum4_0);
                float32x4_t _outp5_0 = vrecpeq_f32(_sum5_0);
                float32x4_t _outp6_0 = vrecpeq_f32(_sum6_0);
                float32x4_t _outp7_0 = vrecpeq_f32(_sum7_0);
                float32x4_t _outp0_1 = vrecpeq_f32(_sum0_1);
                float32x4_t _outp1_1 = vrecpeq_f32(_sum1_1);
                float32x4_t _outp2_1 = vrecpeq_f32(_sum2_1);
                float32x4_t _outp3_1 = vrecpeq_f32(_sum3_1);
                float32x4_t _outp4_1 = vrecpeq_f32(_sum4_1);
                float32x4_t _outp5_1 = vrecpeq_f32(_sum5_1);
                float32x4_t _outp6_1 = vrecpeq_f32(_sum6_1);
                float32x4_t _outp7_1 = vrecpeq_f32(_sum7_1);
                _outp0_0 = vmulq_f32(vrecpsq_f32(_sum0_0, _outp0_0), _outp0_0);
                _outp1_0 = vmulq_f32(vrecpsq_f32(_sum1_0, _outp1_0), _outp1_0);
                _outp2_0 = vmulq_f32(vrecpsq_f32(_sum2_0, _outp2_0), _outp2_0);
                _outp3_0 = vmulq_f32(vrecpsq_f32(_sum3_0, _outp3_0), _outp3_0);
                _outp4_0 = vmulq_f32(vrecpsq_f32(_sum4_0, _outp4_0), _outp4_0);
                _outp5_0 = vmulq_f32(vrecpsq_f32(_sum5_0, _outp5_0), _outp5_0);
                _outp6_0 = vmulq_f32(vrecpsq_f32(_sum6_0, _outp6_0), _outp6_0);
                _outp7_0 = vmulq_f32(vrecpsq_f32(_sum7_0, _outp7_0), _outp7_0);
                _outp0_1 = vmulq_f32(vrecpsq_f32(_sum0_1, _outp0_1), _outp0_1);
                _outp1_1 = vmulq_f32(vrecpsq_f32(_sum1_1, _outp1_1), _outp1_1);
                _outp2_1 = vmulq_f32(vrecpsq_f32(_sum2_1, _outp2_1), _outp2_1);
                _outp3_1 = vmulq_f32(vrecpsq_f32(_sum3_1, _outp3_1), _outp3_1);
                _outp4_1 = vmulq_f32(vrecpsq_f32(_sum4_1, _outp4_1), _outp4_1);
                _outp5_1 = vmulq_f32(vrecpsq_f32(_sum5_1, _outp5_1), _outp5_1);
                _outp6_1 = vmulq_f32(vrecpsq_f32(_sum6_1, _outp6_1), _outp6_1);
                _outp7_1 = vmulq_f32(vrecpsq_f32(_sum7_1, _outp7_1), _outp7_1);
//                 _outp0_0 = vmulq_f32(vrecpsq_f32(_sum0_0, _outp0_0), _outp0_0);
//                 _outp1_0 = vmulq_f32(vrecpsq_f32(_sum1_0, _outp1_0), _outp1_0);
//                 _outp2_0 = vmulq_f32(vrecpsq_f32(_sum2_0, _outp2_0), _outp2_0);
//                 _outp3_0 = vmulq_f32(vrecpsq_f32(_sum3_0, _outp3_0), _outp3_0);
//                 _outp4_0 = vmulq_f32(vrecpsq_f32(_sum4_0, _outp4_0), _outp4_0);
//                 _outp5_0 = vmulq_f32(vrecpsq_f32(_sum5_0, _outp5_0), _outp5_0);
//                 _outp6_0 = vmulq_f32(vrecpsq_f32(_sum6_0, _outp6_0), _outp6_0);
//                 _outp7_0 = vmulq_f32(vrecpsq_f32(_sum7_0, _outp7_0), _outp7_0);
//                 _outp0_1 = vmulq_f32(vrecpsq_f32(_sum0_1, _outp0_1), _outp0_1);
//                 _outp1_1 = vmulq_f32(vrecpsq_f32(_sum1_1, _outp1_1), _outp1_1);
//                 _outp2_1 = vmulq_f32(vrecpsq_f32(_sum2_1, _outp2_1), _outp2_1);
//                 _outp3_1 = vmulq_f32(vrecpsq_f32(_sum3_1, _outp3_1), _outp3_1);
//                 _outp4_1 = vmulq_f32(vrecpsq_f32(_sum4_1, _outp4_1), _outp4_1);
//                 _outp5_1 = vmulq_f32(vrecpsq_f32(_sum5_1, _outp5_1), _outp5_1);
//                 _outp6_1 = vmulq_f32(vrecpsq_f32(_sum6_1, _outp6_1), _outp6_1);
//                 _outp7_1 = vmulq_f32(vrecpsq_f32(_sum7_1, _outp7_1), _outp7_1);
                _sum0_0 = _outp0_0;
                _sum1_0 = _outp1_0;
                _sum2_0 = _outp2_0;
                _sum3_0 = _outp3_0;
                _sum4_0 = _outp4_0;
                _sum5_0 = _outp5_0;
                _sum6_0 = _outp6_0;
                _sum7_0 = _outp7_0;
                _sum0_1 = _outp0_1;
                _sum1_1 = _outp1_1;
                _sum2_1 = _outp2_1;
                _sum3_1 = _outp3_1;
                _sum4_1 = _outp4_1;
                _sum5_1 = _outp5_1;
                _sum6_1 = _outp6_1;
                _sum7_1 = _outp7_1;
            }

            vst1q_f32(outptr0, _sum0_0);
            vst1q_f32(outptr0 + 4, _sum1_0);
            vst1q_f32(outptr0 + 8, _sum2_0);
            vst1q_f32(outptr0 + 12, _sum3_0);
            vst1q_f32(outptr0 + 16, _sum4_0);
            vst1q_f32(outptr0 + 20, _sum5_0);
            vst1q_f32(outptr0 + 24, _sum6_0);
            vst1q_f32(outptr0 + 28, _sum7_0);
            vst1q_f32(outptr1, _sum0_1);
            vst1q_f32(outptr1 + 4, _sum1_1);
            vst1q_f32(outptr1 + 8, _sum2_1);
            vst1q_f32(outptr1 + 12, _sum3_1);
            vst1q_f32(outptr1 + 16, _sum4_1);
            vst1q_f32(outptr1 + 20, _sum5_1);
            vst1q_f32(outptr1 + 24, _sum6_1);
            vst1q_f32(outptr1 + 28, _sum7_1);

            outptr0 += 32;
            outptr1 += 32;
        }
        for (; i+3<size; i+=4)
        {
            const float* tmpptr = tmp.channel(i/8+(i%8)/4);

            float32x4_t _sum0_0 = _bias0;
            float32x4_t _sum1_0 = _bias0;
            float32x4_t _sum2_0 = _bias0;
            float32x4_t _sum3_0 = _bias0;
            float32x4_t _sum0_1 = _bias1;
            float32x4_t _sum1_1 = _bias1;
            float32x4_t _sum2_1 = _bias1;
            float32x4_t _sum3_1 = _bias1;

            const float* kptr0 = (const float*)kernel + p * inch * 16;
            const float* kptr1 = (const float*)kernel + (p+1) * inch * 16;

            for (int q=0; q<inch; q++)
            {
//                 const float* r0 = bottom_blob.channel(q);

//                 float32x4_t _r0 = vld1q_f32(r0 + i*4);
//                 float32x4_t _r1 = vld1q_f32(r0 + (i+1)*4);
//                 float32x4_t _r2 = vld1q_f32(r0 + (i+2)*4);
//                 float32x4_t _r3 = vld1q_f32(r0 + (i+3)*4);

                float32x4_t _r0 = vld1q_f32( tmpptr );
                float32x4_t _r1 = vld1q_f32( tmpptr + 4 );
                float32x4_t _r2 = vld1q_f32( tmpptr + 8 );
                float32x4_t _r3 = vld1q_f32( tmpptr + 12 );

                float32x4_t _w0_0 = vld1q_f32( kptr0 );
                float32x4_t _w1_0 = vld1q_f32( kptr0 + 4 );
                float32x4_t _w2_0 = vld1q_f32( kptr0 + 8 );
                float32x4_t _w3_0 = vld1q_f32( kptr0 + 12 );

                float32x4_t _w0_1 = vld1q_f32( kptr1 );
                float32x4_t _w1_1 = vld1q_f32( kptr1 + 4 );
                float32x4_t _w2_1 = vld1q_f32( kptr1 + 8 );
                float32x4_t _w3_1 = vld1q_f32( kptr1 + 12 );

                _sum0_0 = vmlaq_laneq_f32(_sum0_0, _w0_0, _r0, 0);
                _sum0_0 = vmlaq_laneq_f32(_sum0_0, _w1_0, _r0, 1);
                _sum0_0 = vmlaq_laneq_f32(_sum0_0, _w2_0, _r0, 2);
                _sum0_0 = vmlaq_laneq_f32(_sum0_0, _w3_0, _r0, 3);
                _sum1_0 = vmlaq_laneq_f32(_sum1_0, _w0_0, _r1, 0);
                _sum1_0 = vmlaq_laneq_f32(_sum1_0, _w1_0, _r1, 1);
                _sum1_0 = vmlaq_laneq_f32(_sum1_0, _w2_0, _r1, 2);
                _sum1_0 = vmlaq_laneq_f32(_sum1_0, _w3_0, _r1, 3);
                _sum2_0 = vmlaq_laneq_f32(_sum2_0, _w0_0, _r2, 0);
                _sum2_0 = vmlaq_laneq_f32(_sum2_0, _w1_0, _r2, 1);
                _sum2_0 = vmlaq_laneq_f32(_sum2_0, _w2_0, _r2, 2);
                _sum2_0 = vmlaq_laneq_f32(_sum2_0, _w3_0, _r2, 3);
                _sum3_0 = vmlaq_laneq_f32(_sum3_0, _w0_0, _r3, 0);
                _sum3_0 = vmlaq_laneq_f32(_sum3_0, _w1_0, _r3, 1);
                _sum3_0 = vmlaq_laneq_f32(_sum3_0, _w2_0, _r3, 2);
                _sum3_0 = vmlaq_laneq_f32(_sum3_0, _w3_0, _r3, 3);

                _sum0_1 = vmlaq_laneq_f32(_sum0_1, _w0_1, _r0, 0);
                _sum0_1 = vmlaq_laneq_f32(_sum0_1, _w1_1, _r0, 1);
                _sum0_1 = vmlaq_laneq_f32(_sum0_1, _w2_1, _r0, 2);
                _sum0_1 = vmlaq_laneq_f32(_sum0_1, _w3_1, _r0, 3);
                _sum1_1 = vmlaq_laneq_f32(_sum1_1, _w0_1, _r1, 0);
                _sum1_1 = vmlaq_laneq_f32(_sum1_1, _w1_1, _r1, 1);
                _sum1_1 = vmlaq_laneq_f32(_sum1_1, _w2_1, _r1, 2);
                _sum1_1 = vmlaq_laneq_f32(_sum1_1, _w3_1, _r1, 3);
                _sum2_1 = vmlaq_laneq_f32(_sum2_1, _w0_1, _r2, 0);
                _sum2_1 = vmlaq_laneq_f32(_sum2_1, _w1_1, _r2, 1);
                _sum2_1 = vmlaq_laneq_f32(_sum2_1, _w2_1, _r2, 2);
                _sum2_1 = vmlaq_laneq_f32(_sum2_1, _w3_1, _r2, 3);
                _sum3_1 = vmlaq_laneq_f32(_sum3_1, _w0_1, _r3, 0);
                _sum3_1 = vmlaq_laneq_f32(_sum3_1, _w1_1, _r3, 1);
                _sum3_1 = vmlaq_laneq_f32(_sum3_1, _w2_1, _r3, 2);
                _sum3_1 = vmlaq_laneq_f32(_sum3_1, _w3_1, _r3, 3);

                tmpptr += 16;
                kptr0 += 16;
                kptr1 += 16;
            }

            if (activation_type == 1)
            {
                float32x4_t _zero = vdupq_n_f32(0.f);
                _sum0_0 = vmaxq_f32(_sum0_0, _zero);
                _sum1_0 = vmaxq_f32(_sum1_0, _zero);
                _sum2_0 = vmaxq_f32(_sum2_0, _zero);
                _sum3_0 = vmaxq_f32(_sum3_0, _zero);
                _sum0_1 = vmaxq_f32(_sum0_1, _zero);
                _sum1_1 = vmaxq_f32(_sum1_1, _zero);
                _sum2_1 = vmaxq_f32(_sum2_1, _zero);
                _sum3_1 = vmaxq_f32(_sum3_1, _zero);
            }
            else if (activation_type == 2)
            {
                float32x4_t _zero = vdupq_n_f32(0.f);
                float32x4_t _slope = vdupq_n_f32(activation_params[0]);
                _sum0_0 = vbslq_f32(vcleq_f32(_sum0_0, _zero), vmulq_f32(_sum0_0, _slope), _sum0_0);
                _sum1_0 = vbslq_f32(vcleq_f32(_sum1_0, _zero), vmulq_f32(_sum1_0, _slope), _sum1_0);
                _sum2_0 = vbslq_f32(vcleq_f32(_sum2_0, _zero), vmulq_f32(_sum2_0, _slope), _sum2_0);
                _sum3_0 = vbslq_f32(vcleq_f32(_sum3_0, _zero), vmulq_f32(_sum3_0, _slope), _sum3_0);
                _sum0_1 = vbslq_f32(vcleq_f32(_sum0_1, _zero), vmulq_f32(_sum0_1, _slope), _sum0_1);
                _sum1_1 = vbslq_f32(vcleq_f32(_sum1_1, _zero), vmulq_f32(_sum1_1, _slope), _sum1_1);
                _sum2_1 = vbslq_f32(vcleq_f32(_sum2_1, _zero), vmulq_f32(_sum2_1, _slope), _sum2_1);
                _sum3_1 = vbslq_f32(vcleq_f32(_sum3_1, _zero), vmulq_f32(_sum3_1, _slope), _sum3_1);
            }
            else if (activation_type == 3)
            {
                float32x4_t _min = vdupq_n_f32(activation_params[0]);
                float32x4_t _max = vdupq_n_f32(activation_params[1]);
                _sum0_0 = vmaxq_f32(_sum0_0, _min);
                _sum0_0 = vminq_f32(_sum0_0, _max);
                _sum1_0 = vmaxq_f32(_sum1_0, _min);
                _sum1_0 = vminq_f32(_sum1_0, _max);
                _sum2_0 = vmaxq_f32(_sum2_0, _min);
                _sum2_0 = vminq_f32(_sum2_0, _max);
                _sum3_0 = vmaxq_f32(_sum3_0, _min);
                _sum3_0 = vminq_f32(_sum3_0, _max);
                _sum0_1 = vmaxq_f32(_sum0_1, _min);
                _sum0_1 = vminq_f32(_sum0_1, _max);
                _sum1_1 = vmaxq_f32(_sum1_1, _min);
                _sum1_1 = vminq_f32(_sum1_1, _max);
                _sum2_1 = vmaxq_f32(_sum2_1, _min);
                _sum2_1 = vminq_f32(_sum2_1, _max);
                _sum3_1 = vmaxq_f32(_sum3_1, _min);
                _sum3_1 = vminq_f32(_sum3_1, _max);
            }
            else if (activation_type == 4)
            {
                float32x4_t _one = vdupq_n_f32(1.f);
                _sum0_0 = vnegq_f32(_sum0_0);
                _sum1_0 = vnegq_f32(_sum1_0);
                _sum2_0 = vnegq_f32(_sum2_0);
                _sum3_0 = vnegq_f32(_sum3_0);
                _sum0_1 = vnegq_f32(_sum0_1);
                _sum1_1 = vnegq_f32(_sum1_1);
                _sum2_1 = vnegq_f32(_sum2_1);
                _sum3_1 = vnegq_f32(_sum3_1);
                _sum0_0 = exp_ps(_sum0_0);
                _sum1_0 = exp_ps(_sum1_0);
                _sum2_0 = exp_ps(_sum2_0);
                _sum3_0 = exp_ps(_sum3_0);
                _sum0_1 = exp_ps(_sum0_1);
                _sum1_1 = exp_ps(_sum1_1);
                _sum2_1 = exp_ps(_sum2_1);
                _sum3_1 = exp_ps(_sum3_1);
                _sum0_0 = vaddq_f32(_sum0_0, _one);
                _sum1_0 = vaddq_f32(_sum1_0, _one);
                _sum2_0 = vaddq_f32(_sum2_0, _one);
                _sum3_0 = vaddq_f32(_sum3_0, _one);
                _sum0_1 = vaddq_f32(_sum0_1, _one);
                _sum1_1 = vaddq_f32(_sum1_1, _one);
                _sum2_1 = vaddq_f32(_sum2_1, _one);
                _sum3_1 = vaddq_f32(_sum3_1, _one);
                float32x4_t _outp0_0 = vrecpeq_f32(_sum0_0);
                float32x4_t _outp1_0 = vrecpeq_f32(_sum1_0);
                float32x4_t _outp2_0 = vrecpeq_f32(_sum2_0);
                float32x4_t _outp3_0 = vrecpeq_f32(_sum3_0);
                float32x4_t _outp0_1 = vrecpeq_f32(_sum0_1);
                float32x4_t _outp1_1 = vrecpeq_f32(_sum1_1);
                float32x4_t _outp2_1 = vrecpeq_f32(_sum2_1);
                float32x4_t _outp3_1 = vrecpeq_f32(_sum3_1);
                _outp0_0 = vmulq_f32(vrecpsq_f32(_sum0_0, _outp0_0), _outp0_0);
                _outp1_0 = vmulq_f32(vrecpsq_f32(_sum1_0, _outp1_0), _outp1_0);
                _outp2_0 = vmulq_f32(vrecpsq_f32(_sum2_0, _outp0_0), _outp2_0);
                _outp3_0 = vmulq_f32(vrecpsq_f32(_sum3_0, _outp1_0), _outp3_0);
                _outp0_1 = vmulq_f32(vrecpsq_f32(_sum0_1, _outp0_1), _outp0_1);
                _outp1_1 = vmulq_f32(vrecpsq_f32(_sum1_1, _outp1_1), _outp1_1);
                _outp2_1 = vmulq_f32(vrecpsq_f32(_sum2_1, _outp0_1), _outp2_1);
                _outp3_1 = vmulq_f32(vrecpsq_f32(_sum3_1, _outp1_1), _outp3_1);
//                 _outp0_0 = vmulq_f32(vrecpsq_f32(_sum0_0, _outp0_0), _outp0_0);
//                 _outp1_0 = vmulq_f32(vrecpsq_f32(_sum1_0, _outp1_0), _outp1_0);
//                 _outp2_0 = vmulq_f32(vrecpsq_f32(_sum2_0, _outp0_0), _outp2_0);
//                 _outp3_0 = vmulq_f32(vrecpsq_f32(_sum3_0, _outp1_0), _outp3_0);
//                 _outp0_1 = vmulq_f32(vrecpsq_f32(_sum0_1, _outp0_1), _outp0_1);
//                 _outp1_1 = vmulq_f32(vrecpsq_f32(_sum1_1, _outp1_1), _outp1_1);
//                 _outp2_1 = vmulq_f32(vrecpsq_f32(_sum2_1, _outp0_1), _outp2_1);
//                 _outp3_1 = vmulq_f32(vrecpsq_f32(_sum3_1, _outp1_1), _outp3_1);
                _sum0_0 = _outp0_0;
                _sum1_0 = _outp1_0;
                _sum2_0 = _outp2_0;
                _sum3_0 = _outp3_0;
                _sum0_1 = _outp0_1;
                _sum1_1 = _outp1_1;
                _sum2_1 = _outp2_1;
                _sum3_1 = _outp3_1;
            }

            vst1q_f32(outptr0, _sum0_0);
            vst1q_f32(outptr0 + 4, _sum1_0);
            vst1q_f32(outptr0 + 8, _sum2_0);
            vst1q_f32(outptr0 + 12, _sum3_0);
            vst1q_f32(outptr1, _sum0_1);
            vst1q_f32(outptr1 + 4, _sum1_1);
            vst1q_f32(outptr1 + 8, _sum2_1);
            vst1q_f32(outptr1 + 12, _sum3_1);

            outptr0 += 16;
            outptr1 += 16;
        }
        for (; i+1<size; i+=2)
        {
            const float* tmpptr = tmp.channel(i/8+(i%8)/4 + (i%4)/2);

            float32x4_t _sum0_0 = _bias0;
            float32x4_t _sum1_0 = _bias0;
            float32x4_t _sum0_1 = _bias1;
            float32x4_t _sum1_1 = _bias1;

            const float* kptr0 = (const float*)kernel + p * inch * 16;
            const float* kptr1 = (const float*)kernel + (p+1) * inch * 16;

            for (int q=0; q<inch; q++)
            {
//                 const float* r0 = bottom_blob.channel(q);

//                 float32x4_t _r0 = vld1q_f32(r0 + i*4);
//                 float32x4_t _r1 = vld1q_f32(r0 + (i+1)*4);

                float32x4_t _r0 = vld1q_f32( tmpptr );
                float32x4_t _r1 = vld1q_f32( tmpptr + 4 );

                float32x4_t _w0_0 = vld1q_f32( kptr0 );
                float32x4_t _w1_0 = vld1q_f32( kptr0 + 4 );
                float32x4_t _w2_0 = vld1q_f32( kptr0 + 8 );
                float32x4_t _w3_0 = vld1q_f32( kptr0 + 12 );

                float32x4_t _w0_1 = vld1q_f32( kptr1 );
                float32x4_t _w1_1 = vld1q_f32( kptr1 + 4 );
                float32x4_t _w2_1 = vld1q_f32( kptr1 + 8 );
                float32x4_t _w3_1 = vld1q_f32( kptr1 + 12 );

                _sum0_0 = vmlaq_laneq_f32(_sum0_0, _w0_0, _r0, 0);
                _sum0_0 = vmlaq_laneq_f32(_sum0_0, _w1_0, _r0, 1);
                _sum0_0 = vmlaq_laneq_f32(_sum0_0, _w2_0, _r0, 2);
                _sum0_0 = vmlaq_laneq_f32(_sum0_0, _w3_0, _r0, 3);
                _sum1_0 = vmlaq_laneq_f32(_sum1_0, _w0_0, _r1, 0);
                _sum1_0 = vmlaq_laneq_f32(_sum1_0, _w1_0, _r1, 1);
                _sum1_0 = vmlaq_laneq_f32(_sum1_0, _w2_0, _r1, 2);
                _sum1_0 = vmlaq_laneq_f32(_sum1_0, _w3_0, _r1, 3);

                _sum0_1 = vmlaq_laneq_f32(_sum0_1, _w0_1, _r0, 0);
                _sum0_1 = vmlaq_laneq_f32(_sum0_1, _w1_1, _r0, 1);
                _sum0_1 = vmlaq_laneq_f32(_sum0_1, _w2_1, _r0, 2);
                _sum0_1 = vmlaq_laneq_f32(_sum0_1, _w3_1, _r0, 3);
                _sum1_1 = vmlaq_laneq_f32(_sum1_1, _w0_1, _r1, 0);
                _sum1_1 = vmlaq_laneq_f32(_sum1_1, _w1_1, _r1, 1);
                _sum1_1 = vmlaq_laneq_f32(_sum1_1, _w2_1, _r1, 2);
                _sum1_1 = vmlaq_laneq_f32(_sum1_1, _w3_1, _r1, 3);

                tmpptr += 8;
                kptr0 += 16;
                kptr1 += 16;
            }

            if (activation_type == 1)
            {
                float32x4_t _zero = vdupq_n_f32(0.f);
                _sum0_0 = vmaxq_f32(_sum0_0, _zero);
                _sum1_0 = vmaxq_f32(_sum1_0, _zero);
                _sum0_1 = vmaxq_f32(_sum0_1, _zero);
                _sum1_1 = vmaxq_f32(_sum1_1, _zero);
            }
            else if (activation_type == 2)
            {
                float32x4_t _zero = vdupq_n_f32(0.f);
                float32x4_t _slope = vdupq_n_f32(activation_params[0]);
                _sum0_0 = vbslq_f32(vcleq_f32(_sum0_0, _zero), vmulq_f32(_sum0_0, _slope), _sum0_0);
                _sum1_0 = vbslq_f32(vcleq_f32(_sum1_0, _zero), vmulq_f32(_sum1_0, _slope), _sum1_0);
                _sum0_1 = vbslq_f32(vcleq_f32(_sum0_1, _zero), vmulq_f32(_sum0_1, _slope), _sum0_1);
                _sum1_1 = vbslq_f32(vcleq_f32(_sum1_1, _zero), vmulq_f32(_sum1_1, _slope), _sum1_1);
            }
            else if (activation_type == 3)
            {
                float32x4_t _min = vdupq_n_f32(activation_params[0]);
                float32x4_t _max = vdupq_n_f32(activation_params[1]);
                _sum0_0 = vmaxq_f32(_sum0_0, _min);
                _sum0_0 = vminq_f32(_sum0_0, _max);
                _sum1_0 = vmaxq_f32(_sum1_0, _min);
                _sum1_0 = vminq_f32(_sum1_0, _max);
                _sum0_1 = vmaxq_f32(_sum0_1, _min);
                _sum0_1 = vminq_f32(_sum0_1, _max);
                _sum1_1 = vmaxq_f32(_sum1_1, _min);
                _sum1_1 = vminq_f32(_sum1_1, _max);
            }
            else if (activation_type == 4)
            {
                float32x4_t _one = vdupq_n_f32(1.f);
                _sum0_0 = vnegq_f32(_sum0_0);
                _sum1_0 = vnegq_f32(_sum1_0);
                _sum0_1 = vnegq_f32(_sum0_1);
                _sum1_1 = vnegq_f32(_sum1_1);
                _sum0_0 = exp_ps(_sum0_0);
                _sum1_0 = exp_ps(_sum1_0);
                _sum0_1 = exp_ps(_sum0_1);
                _sum1_1 = exp_ps(_sum1_1);
                _sum0_0 = vaddq_f32(_sum0_0, _one);
                _sum1_0 = vaddq_f32(_sum1_0, _one);
                _sum0_1 = vaddq_f32(_sum0_1, _one);
                _sum1_1 = vaddq_f32(_sum1_1, _one);
                float32x4_t _outp0_0 = vrecpeq_f32(_sum0_0);
                float32x4_t _outp1_0 = vrecpeq_f32(_sum1_0);
                float32x4_t _outp0_1 = vrecpeq_f32(_sum0_1);
                float32x4_t _outp1_1 = vrecpeq_f32(_sum1_1);
                _outp0_0 = vmulq_f32(vrecpsq_f32(_sum0_0, _outp0_0), _outp0_0);
                _outp1_0 = vmulq_f32(vrecpsq_f32(_sum1_0, _outp1_0), _outp1_0);
                _outp0_1 = vmulq_f32(vrecpsq_f32(_sum0_1, _outp0_1), _outp0_1);
                _outp1_1 = vmulq_f32(vrecpsq_f32(_sum1_1, _outp1_1), _outp1_1);
//                 _outp0_0 = vmulq_f32(vrecpsq_f32(_sum0_0, _outp0_0), _outp0_0);
//                 _outp1_0 = vmulq_f32(vrecpsq_f32(_sum1_0, _outp1_0), _outp1_0);
//                 _outp0_1 = vmulq_f32(vrecpsq_f32(_sum0_1, _outp0_1), _outp0_1);
//                 _outp1_1 = vmulq_f32(vrecpsq_f32(_sum1_1, _outp1_1), _outp1_1);
                _sum0_0 = _outp0_0;
                _sum1_0 = _outp1_0;
                _sum0_1 = _outp0_1;
                _sum1_1 = _outp1_1;
            }

            vst1q_f32(outptr0, _sum0_0);
            vst1q_f32(outptr0 + 4, _sum1_0);
            vst1q_f32(outptr1, _sum0_1);
            vst1q_f32(outptr1 + 4, _sum1_1);

            outptr0 += 8;
            outptr1 += 8;
        }
        for (; i<size; i++)
        {
            const float* tmpptr = tmp.channel(i/8+(i%8)/4 + (i%4)/2 + i%2);

            float32x4_t _sum_0 = _bias0;
            float32x4_t _sum_1 = _bias1;

            const float* kptr0 = (const float*)kernel + p * inch * 16;
            const float* kptr1 = (const float*)kernel + (p+1) * inch * 16;

            for (int q=0; q<inch; q++)
            {
//                 const float* r0 = bottom_blob.channel(q);

//                 float32x4_t _val = vld1q_f32(r0 + i*4);

                float32x4_t _val = vld1q_f32( tmpptr );

                float32x4_t _w0_0 = vld1q_f32( kptr0 );
                float32x4_t _w1_0 = vld1q_f32( kptr0 + 4 );
                float32x4_t _w2_0 = vld1q_f32( kptr0 + 8 );
                float32x4_t _w3_0 = vld1q_f32( kptr0 + 12 );

                float32x4_t _w0_1 = vld1q_f32( kptr1 );
                float32x4_t _w1_1 = vld1q_f32( kptr1 + 4 );
                float32x4_t _w2_1 = vld1q_f32( kptr1 + 8 );
                float32x4_t _w3_1 = vld1q_f32( kptr1 + 12 );

                _sum_0 = vmlaq_laneq_f32(_sum_0, _w0_0, _val, 0);
                _sum_0 = vmlaq_laneq_f32(_sum_0, _w1_0, _val, 1);
                _sum_0 = vmlaq_laneq_f32(_sum_0, _w2_0, _val, 2);
                _sum_0 = vmlaq_laneq_f32(_sum_0, _w3_0, _val, 3);
                _sum_1 = vmlaq_laneq_f32(_sum_1, _w0_1, _val, 0);
                _sum_1 = vmlaq_laneq_f32(_sum_1, _w1_1, _val, 1);
                _sum_1 = vmlaq_laneq_f32(_sum_1, _w2_1, _val, 2);
                _sum_1 = vmlaq_laneq_f32(_sum_1, _w3_1, _val, 3);

                tmpptr += 4;
                kptr0 += 16;
                kptr1 += 16;
            }

            if (activation_type == 1)
            {
                float32x4_t _zero = vdupq_n_f32(0.f);
                _sum_0 = vmaxq_f32(_sum_0, _zero);
                _sum_1 = vmaxq_f32(_sum_1, _zero);
            }
            else if (activation_type == 2)
            {
                float32x4_t _zero = vdupq_n_f32(0.f);
                float32x4_t _slope = vdupq_n_f32(activation_params[0]);
                _sum_0 = vbslq_f32(vcleq_f32(_sum_0, _zero), vmulq_f32(_sum_0, _slope), _sum_0);
                _sum_1 = vbslq_f32(vcleq_f32(_sum_1, _zero), vmulq_f32(_sum_1, _slope), _sum_1);
            }
            else if (activation_type == 3)
            {
                float32x4_t _min = vdupq_n_f32(activation_params[0]);
                float32x4_t _max = vdupq_n_f32(activation_params[1]);
                _sum_0 = vmaxq_f32(_sum_0, _min);
                _sum_0 = vminq_f32(_sum_0, _max);
                _sum_1 = vmaxq_f32(_sum_1, _min);
                _sum_1 = vminq_f32(_sum_1, _max);
            }
            else if (activation_type == 4)
            {
                float32x4_t _one = vdupq_n_f32(1.f);
                _sum_0 = vnegq_f32(_sum_0);
                _sum_1 = vnegq_f32(_sum_1);
                _sum_0 = exp_ps(_sum_0);
                _sum_1 = exp_ps(_sum_1);
                _sum_0 = vaddq_f32(_sum_0, _one);
                _sum_1 = vaddq_f32(_sum_1, _one);
                float32x4_t _outp_0 = vrecpeq_f32(_sum_0);
                float32x4_t _outp_1 = vrecpeq_f32(_sum_1);
                _outp_0 = vmulq_f32(vrecpsq_f32(_sum_0, _outp_0), _outp_0);
                _outp_1 = vmulq_f32(vrecpsq_f32(_sum_1, _outp_1), _outp_1);
//                 _outp_0 = vmulq_f32(vrecpsq_f32(_sum_0, _outp_0), _outp_0);
//                 _outp_1 = vmulq_f32(vrecpsq_f32(_sum_1, _outp_1), _outp_1);
                _sum_0 = _outp_0;
                _sum_1 = _outp_1;
            }

            vst1q_f32(outptr0, _sum_0);
            vst1q_f32(outptr1, _sum_1);

            outptr0 += 4;
            outptr1 += 4;
        }
    }

#endif // __ARM_NEON && __aarch64__

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p=remain_outch_start; p<outch; p++)
    {
        Mat out = top_blob.channel(p);

        float32x4_t _bias0 = bias ? vld1q_f32(bias + p * 4) : vdupq_n_f32(0.f);

        float* outptr = out;

        int i=0;
        for (; i+7<size; i+=8)
        {
            const float* tmpptr = tmp.channel(i/8);

            float32x4_t _sum0 = _bias0;
            float32x4_t _sum1 = _bias0;
            float32x4_t _sum2 = _bias0;
            float32x4_t _sum3 = _bias0;
            float32x4_t _sum4 = _bias0;
            float32x4_t _sum5 = _bias0;
            float32x4_t _sum6 = _bias0;
            float32x4_t _sum7 = _bias0;

            const float* kptr = (const float*)kernel + p * inch * 16;

            for (int q=0; q<inch; q++)
            {
//                 asm volatile("nop\nnop\nnop\n" : : :);
//                 const float* r0 = bottom_blob.channel(q);

//                 float32x4_t _r0 = vld1q_f32(r0 + i*4);
//                 float32x4_t _r1 = vld1q_f32(r0 + (i+1)*4);
//                 float32x4_t _r2 = vld1q_f32(r0 + (i+2)*4);
//                 float32x4_t _r3 = vld1q_f32(r0 + (i+3)*4);

                float32x4_t _r0 = vld1q_f32( tmpptr );
                float32x4_t _r1 = vld1q_f32( tmpptr + 4 );
                float32x4_t _r2 = vld1q_f32( tmpptr + 8 );
                float32x4_t _r3 = vld1q_f32( tmpptr + 12 );
                float32x4_t _r4 = vld1q_f32( tmpptr + 16 );
                float32x4_t _r5 = vld1q_f32( tmpptr + 20 );
                float32x4_t _r6 = vld1q_f32( tmpptr + 24 );
                float32x4_t _r7 = vld1q_f32( tmpptr + 28 );

                float32x4_t _w0 = vld1q_f32( kptr );
                float32x4_t _w1 = vld1q_f32( kptr + 4 );
                float32x4_t _w2 = vld1q_f32( kptr + 8 );
                float32x4_t _w3 = vld1q_f32( kptr + 12 );

#if __aarch64__
                _sum0 = vmlaq_laneq_f32(_sum0, _w0, _r0, 0);
                _sum0 = vmlaq_laneq_f32(_sum0, _w1, _r0, 1);
                _sum0 = vmlaq_laneq_f32(_sum0, _w2, _r0, 2);
                _sum0 = vmlaq_laneq_f32(_sum0, _w3, _r0, 3);
                _sum1 = vmlaq_laneq_f32(_sum1, _w0, _r1, 0);
                _sum1 = vmlaq_laneq_f32(_sum1, _w1, _r1, 1);
                _sum1 = vmlaq_laneq_f32(_sum1, _w2, _r1, 2);
                _sum1 = vmlaq_laneq_f32(_sum1, _w3, _r1, 3);
                _sum2 = vmlaq_laneq_f32(_sum2, _w0, _r2, 0);
                _sum2 = vmlaq_laneq_f32(_sum2, _w1, _r2, 1);
                _sum2 = vmlaq_laneq_f32(_sum2, _w2, _r2, 2);
                _sum2 = vmlaq_laneq_f32(_sum2, _w3, _r2, 3);
                _sum3 = vmlaq_laneq_f32(_sum3, _w0, _r3, 0);
                _sum3 = vmlaq_laneq_f32(_sum3, _w1, _r3, 1);
                _sum3 = vmlaq_laneq_f32(_sum3, _w2, _r3, 2);
                _sum3 = vmlaq_laneq_f32(_sum3, _w3, _r3, 3);
                _sum4 = vmlaq_laneq_f32(_sum4, _w0, _r4, 0);
                _sum4 = vmlaq_laneq_f32(_sum4, _w1, _r4, 1);
                _sum4 = vmlaq_laneq_f32(_sum4, _w2, _r4, 2);
                _sum4 = vmlaq_laneq_f32(_sum4, _w3, _r4, 3);
                _sum5 = vmlaq_laneq_f32(_sum5, _w0, _r5, 0);
                _sum5 = vmlaq_laneq_f32(_sum5, _w1, _r5, 1);
                _sum5 = vmlaq_laneq_f32(_sum5, _w2, _r5, 2);
                _sum5 = vmlaq_laneq_f32(_sum5, _w3, _r5, 3);
                _sum6 = vmlaq_laneq_f32(_sum6, _w0, _r6, 0);
                _sum6 = vmlaq_laneq_f32(_sum6, _w1, _r6, 1);
                _sum6 = vmlaq_laneq_f32(_sum6, _w2, _r6, 2);
                _sum6 = vmlaq_laneq_f32(_sum6, _w3, _r6, 3);
                _sum7 = vmlaq_laneq_f32(_sum7, _w0, _r7, 0);
                _sum7 = vmlaq_laneq_f32(_sum7, _w1, _r7, 1);
                _sum7 = vmlaq_laneq_f32(_sum7, _w2, _r7, 2);
                _sum7 = vmlaq_laneq_f32(_sum7, _w3, _r7, 3);
#else
                _sum0 = vmlaq_lane_f32(_sum0, _w0, vget_low_f32(_r0), 0);
                _sum0 = vmlaq_lane_f32(_sum0, _w1, vget_low_f32(_r0), 1);
                _sum0 = vmlaq_lane_f32(_sum0, _w2, vget_high_f32(_r0), 0);
                _sum0 = vmlaq_lane_f32(_sum0, _w3, vget_high_f32(_r0), 1);
                _sum1 = vmlaq_lane_f32(_sum1, _w0, vget_low_f32(_r1), 0);
                _sum1 = vmlaq_lane_f32(_sum1, _w1, vget_low_f32(_r1), 1);
                _sum1 = vmlaq_lane_f32(_sum1, _w2, vget_high_f32(_r1), 0);
                _sum1 = vmlaq_lane_f32(_sum1, _w3, vget_high_f32(_r1), 1);
                _sum2 = vmlaq_lane_f32(_sum2, _w0, vget_low_f32(_r2), 0);
                _sum2 = vmlaq_lane_f32(_sum2, _w1, vget_low_f32(_r2), 1);
                _sum2 = vmlaq_lane_f32(_sum2, _w2, vget_high_f32(_r2), 0);
                _sum2 = vmlaq_lane_f32(_sum2, _w3, vget_high_f32(_r2), 1);
                _sum3 = vmlaq_lane_f32(_sum3, _w0, vget_low_f32(_r3), 0);
                _sum3 = vmlaq_lane_f32(_sum3, _w1, vget_low_f32(_r3), 1);
                _sum3 = vmlaq_lane_f32(_sum3, _w2, vget_high_f32(_r3), 0);
                _sum3 = vmlaq_lane_f32(_sum3, _w3, vget_high_f32(_r3), 1);
                _sum4 = vmlaq_lane_f32(_sum4, _w0, vget_low_f32(_r4), 0);
                _sum4 = vmlaq_lane_f32(_sum4, _w1, vget_low_f32(_r4), 1);
                _sum4 = vmlaq_lane_f32(_sum4, _w2, vget_high_f32(_r4), 0);
                _sum4 = vmlaq_lane_f32(_sum4, _w3, vget_high_f32(_r4), 1);
                _sum5 = vmlaq_lane_f32(_sum5, _w0, vget_low_f32(_r5), 0);
                _sum5 = vmlaq_lane_f32(_sum5, _w1, vget_low_f32(_r5), 1);
                _sum5 = vmlaq_lane_f32(_sum5, _w2, vget_high_f32(_r5), 0);
                _sum5 = vmlaq_lane_f32(_sum5, _w3, vget_high_f32(_r5), 1);
                _sum6 = vmlaq_lane_f32(_sum6, _w0, vget_low_f32(_r6), 0);
                _sum6 = vmlaq_lane_f32(_sum6, _w1, vget_low_f32(_r6), 1);
                _sum6 = vmlaq_lane_f32(_sum6, _w2, vget_high_f32(_r6), 0);
                _sum6 = vmlaq_lane_f32(_sum6, _w3, vget_high_f32(_r6), 1);
                _sum7 = vmlaq_lane_f32(_sum7, _w0, vget_low_f32(_r7), 0);
                _sum7 = vmlaq_lane_f32(_sum7, _w1, vget_low_f32(_r7), 1);
                _sum7 = vmlaq_lane_f32(_sum7, _w2, vget_high_f32(_r7), 0);
                _sum7 = vmlaq_lane_f32(_sum7, _w3, vget_high_f32(_r7), 1);
#endif

                tmpptr += 32;
                kptr += 16;
            }

            if (activation_type == 1)
            {
                float32x4_t _zero = vdupq_n_f32(0.f);
                _sum0 = vmaxq_f32(_sum0, _zero);
                _sum1 = vmaxq_f32(_sum1, _zero);
                _sum2 = vmaxq_f32(_sum2, _zero);
                _sum3 = vmaxq_f32(_sum3, _zero);
                _sum4 = vmaxq_f32(_sum4, _zero);
                _sum5 = vmaxq_f32(_sum5, _zero);
                _sum6 = vmaxq_f32(_sum6, _zero);
                _sum7 = vmaxq_f32(_sum7, _zero);
            }
            else if (activation_type == 2)
            {
                float32x4_t _zero = vdupq_n_f32(0.f);
                float32x4_t _slope = vdupq_n_f32(activation_params[0]);
                _sum0 = vbslq_f32(vcleq_f32(_sum0, _zero), vmulq_f32(_sum0, _slope), _sum0);
                _sum1 = vbslq_f32(vcleq_f32(_sum1, _zero), vmulq_f32(_sum1, _slope), _sum1);
                _sum2 = vbslq_f32(vcleq_f32(_sum2, _zero), vmulq_f32(_sum2, _slope), _sum2);
                _sum3 = vbslq_f32(vcleq_f32(_sum3, _zero), vmulq_f32(_sum3, _slope), _sum3);
                _sum4 = vbslq_f32(vcleq_f32(_sum4, _zero), vmulq_f32(_sum4, _slope), _sum4);
                _sum5 = vbslq_f32(vcleq_f32(_sum5, _zero), vmulq_f32(_sum5, _slope), _sum5);
                _sum6 = vbslq_f32(vcleq_f32(_sum6, _zero), vmulq_f32(_sum6, _slope), _sum6);
                _sum7 = vbslq_f32(vcleq_f32(_sum7, _zero), vmulq_f32(_sum7, _slope), _sum7);
            }
            else if (activation_type == 3)
            {
                float32x4_t _min = vdupq_n_f32(activation_params[0]);
                float32x4_t _max = vdupq_n_f32(activation_params[1]);
                _sum0 = vmaxq_f32(_sum0, _min);
                _sum0 = vminq_f32(_sum0, _max);
                _sum1 = vmaxq_f32(_sum1, _min);
                _sum1 = vminq_f32(_sum1, _max);
                _sum2 = vmaxq_f32(_sum2, _min);
                _sum2 = vminq_f32(_sum2, _max);
                _sum3 = vmaxq_f32(_sum3, _min);
                _sum3 = vminq_f32(_sum3, _max);
                _sum4 = vmaxq_f32(_sum4, _min);
                _sum4 = vminq_f32(_sum4, _max);
                _sum5 = vmaxq_f32(_sum5, _min);
                _sum5 = vminq_f32(_sum5, _max);
                _sum6 = vmaxq_f32(_sum6, _min);
                _sum6 = vminq_f32(_sum6, _max);
                _sum7 = vmaxq_f32(_sum7, _min);
                _sum7 = vminq_f32(_sum7, _max);
            }
            else if (activation_type == 4)
            {
                float32x4_t _one = vdupq_n_f32(1.f);
                _sum0 = vnegq_f32(_sum0);
                _sum1 = vnegq_f32(_sum1);
                _sum2 = vnegq_f32(_sum2);
                _sum3 = vnegq_f32(_sum3);
                _sum4 = vnegq_f32(_sum4);
                _sum5 = vnegq_f32(_sum5);
                _sum6 = vnegq_f32(_sum6);
                _sum7 = vnegq_f32(_sum7);
                _sum0 = exp_ps(_sum0);
                _sum1 = exp_ps(_sum1);
                _sum2 = exp_ps(_sum2);
                _sum3 = exp_ps(_sum3);
                _sum4 = exp_ps(_sum4);
                _sum5 = exp_ps(_sum5);
                _sum6 = exp_ps(_sum6);
                _sum7 = exp_ps(_sum7);
                _sum0 = vaddq_f32(_sum0, _one);
                _sum1 = vaddq_f32(_sum1, _one);
                _sum2 = vaddq_f32(_sum2, _one);
                _sum3 = vaddq_f32(_sum3, _one);
                _sum4 = vaddq_f32(_sum4, _one);
                _sum5 = vaddq_f32(_sum5, _one);
                _sum6 = vaddq_f32(_sum6, _one);
                _sum7 = vaddq_f32(_sum7, _one);
                float32x4_t _outp0 = vrecpeq_f32(_sum0);
                float32x4_t _outp1 = vrecpeq_f32(_sum1);
                float32x4_t _outp2 = vrecpeq_f32(_sum2);
                float32x4_t _outp3 = vrecpeq_f32(_sum3);
                float32x4_t _outp4 = vrecpeq_f32(_sum4);
                float32x4_t _outp5 = vrecpeq_f32(_sum5);
                float32x4_t _outp6 = vrecpeq_f32(_sum6);
                float32x4_t _outp7 = vrecpeq_f32(_sum7);
                _outp0 = vmulq_f32(vrecpsq_f32(_sum0, _outp0), _outp0);
                _outp1 = vmulq_f32(vrecpsq_f32(_sum1, _outp1), _outp1);
                _outp2 = vmulq_f32(vrecpsq_f32(_sum2, _outp0), _outp2);
                _outp3 = vmulq_f32(vrecpsq_f32(_sum3, _outp1), _outp3);
                _outp4 = vmulq_f32(vrecpsq_f32(_sum4, _outp4), _outp4);
                _outp5 = vmulq_f32(vrecpsq_f32(_sum5, _outp5), _outp5);
                _outp6 = vmulq_f32(vrecpsq_f32(_sum6, _outp6), _outp6);
                _outp7 = vmulq_f32(vrecpsq_f32(_sum7, _outp7), _outp7);
//                 _outp0 = vmulq_f32(vrecpsq_f32(_sum0, _outp0), _outp0);
//                 _outp1 = vmulq_f32(vrecpsq_f32(_sum1, _outp1), _outp1);
//                 _outp2 = vmulq_f32(vrecpsq_f32(_sum2, _outp0), _outp2);
//                 _outp3 = vmulq_f32(vrecpsq_f32(_sum3, _outp1), _outp3);
//                 _outp4 = vmulq_f32(vrecpsq_f32(_sum4, _outp4), _outp4);
//                 _outp5 = vmulq_f32(vrecpsq_f32(_sum5, _outp5), _outp5);
//                 _outp6 = vmulq_f32(vrecpsq_f32(_sum6, _outp6), _outp6);
//                 _outp7 = vmulq_f32(vrecpsq_f32(_sum7, _outp7), _outp7);
                _sum0 = _outp0;
                _sum1 = _outp1;
                _sum2 = _outp2;
                _sum3 = _outp3;
                _sum4 = _outp4;
                _sum5 = _outp5;
                _sum6 = _outp6;
                _sum7 = _outp7;
            }

            vst1q_f32(outptr, _sum0);
            vst1q_f32(outptr + 4, _sum1);
            vst1q_f32(outptr + 8, _sum2);
            vst1q_f32(outptr + 12, _sum3);
            vst1q_f32(outptr + 16, _sum4);
            vst1q_f32(outptr + 20, _sum5);
            vst1q_f32(outptr + 24, _sum6);
            vst1q_f32(outptr + 28, _sum7);

            outptr += 32;
        }
        for (; i+3<size; i+=4)
        {
            const float* tmpptr = tmp.channel(i/8+(i%8)/4);

            float32x4_t _sum0 = _bias0;
            float32x4_t _sum1 = _bias0;
            float32x4_t _sum2 = _bias0;
            float32x4_t _sum3 = _bias0;

            const float* kptr = (const float*)kernel + p * inch * 16;

            for (int q=0; q<inch; q++)
            {
//                 const float* r0 = bottom_blob.channel(q);

//                 float32x4_t _r0 = vld1q_f32(r0 + i*4);
//                 float32x4_t _r1 = vld1q_f32(r0 + (i+1)*4);
//                 float32x4_t _r2 = vld1q_f32(r0 + (i+2)*4);
//                 float32x4_t _r3 = vld1q_f32(r0 + (i+3)*4);

                float32x4_t _r0 = vld1q_f32( tmpptr );
                float32x4_t _r1 = vld1q_f32( tmpptr + 4 );
                float32x4_t _r2 = vld1q_f32( tmpptr + 8 );
                float32x4_t _r3 = vld1q_f32( tmpptr + 12 );

                float32x4_t _w0 = vld1q_f32( kptr );
                float32x4_t _w1 = vld1q_f32( kptr + 4 );
                float32x4_t _w2 = vld1q_f32( kptr + 8 );
                float32x4_t _w3 = vld1q_f32( kptr + 12 );

#if __aarch64__
                _sum0 = vmlaq_laneq_f32(_sum0, _w0, _r0, 0);
                _sum0 = vmlaq_laneq_f32(_sum0, _w1, _r0, 1);
                _sum0 = vmlaq_laneq_f32(_sum0, _w2, _r0, 2);
                _sum0 = vmlaq_laneq_f32(_sum0, _w3, _r0, 3);
                _sum1 = vmlaq_laneq_f32(_sum1, _w0, _r1, 0);
                _sum1 = vmlaq_laneq_f32(_sum1, _w1, _r1, 1);
                _sum1 = vmlaq_laneq_f32(_sum1, _w2, _r1, 2);
                _sum1 = vmlaq_laneq_f32(_sum1, _w3, _r1, 3);
                _sum2 = vmlaq_laneq_f32(_sum2, _w0, _r2, 0);
                _sum2 = vmlaq_laneq_f32(_sum2, _w1, _r2, 1);
                _sum2 = vmlaq_laneq_f32(_sum2, _w2, _r2, 2);
                _sum2 = vmlaq_laneq_f32(_sum2, _w3, _r2, 3);
                _sum3 = vmlaq_laneq_f32(_sum3, _w0, _r3, 0);
                _sum3 = vmlaq_laneq_f32(_sum3, _w1, _r3, 1);
                _sum3 = vmlaq_laneq_f32(_sum3, _w2, _r3, 2);
                _sum3 = vmlaq_laneq_f32(_sum3, _w3, _r3, 3);
#else
                _sum0 = vmlaq_lane_f32(_sum0, _w0, vget_low_f32(_r0), 0);
                _sum0 = vmlaq_lane_f32(_sum0, _w1, vget_low_f32(_r0), 1);
                _sum0 = vmlaq_lane_f32(_sum0, _w2, vget_high_f32(_r0), 0);
                _sum0 = vmlaq_lane_f32(_sum0, _w3, vget_high_f32(_r0), 1);
                _sum1 = vmlaq_lane_f32(_sum1, _w0, vget_low_f32(_r1), 0);
                _sum1 = vmlaq_lane_f32(_sum1, _w1, vget_low_f32(_r1), 1);
                _sum1 = vmlaq_lane_f32(_sum1, _w2, vget_high_f32(_r1), 0);
                _sum1 = vmlaq_lane_f32(_sum1, _w3, vget_high_f32(_r1), 1);
                _sum2 = vmlaq_lane_f32(_sum2, _w0, vget_low_f32(_r2), 0);
                _sum2 = vmlaq_lane_f32(_sum2, _w1, vget_low_f32(_r2), 1);
                _sum2 = vmlaq_lane_f32(_sum2, _w2, vget_high_f32(_r2), 0);
                _sum2 = vmlaq_lane_f32(_sum2, _w3, vget_high_f32(_r2), 1);
                _sum3 = vmlaq_lane_f32(_sum3, _w0, vget_low_f32(_r3), 0);
                _sum3 = vmlaq_lane_f32(_sum3, _w1, vget_low_f32(_r3), 1);
                _sum3 = vmlaq_lane_f32(_sum3, _w2, vget_high_f32(_r3), 0);
                _sum3 = vmlaq_lane_f32(_sum3, _w3, vget_high_f32(_r3), 1);
#endif

                tmpptr += 16;
                kptr += 16;
            }

            if (activation_type == 1)
            {
                float32x4_t _zero = vdupq_n_f32(0.f);
                _sum0 = vmaxq_f32(_sum0, _zero);
                _sum1 = vmaxq_f32(_sum1, _zero);
                _sum2 = vmaxq_f32(_sum2, _zero);
                _sum3 = vmaxq_f32(_sum3, _zero);
            }
            else if (activation_type == 2)
            {
                float32x4_t _zero = vdupq_n_f32(0.f);
                float32x4_t _slope = vdupq_n_f32(activation_params[0]);
                _sum0 = vbslq_f32(vcleq_f32(_sum0, _zero), vmulq_f32(_sum0, _slope), _sum0);
                _sum1 = vbslq_f32(vcleq_f32(_sum1, _zero), vmulq_f32(_sum1, _slope), _sum1);
                _sum2 = vbslq_f32(vcleq_f32(_sum2, _zero), vmulq_f32(_sum2, _slope), _sum2);
                _sum3 = vbslq_f32(vcleq_f32(_sum3, _zero), vmulq_f32(_sum3, _slope), _sum3);
            }
            else if (activation_type == 3)
            {
                float32x4_t _min = vdupq_n_f32(activation_params[0]);
                float32x4_t _max = vdupq_n_f32(activation_params[1]);
                _sum0 = vmaxq_f32(_sum0, _min);
                _sum0 = vminq_f32(_sum0, _max);
                _sum1 = vmaxq_f32(_sum1, _min);
                _sum1 = vminq_f32(_sum1, _max);
                _sum2 = vmaxq_f32(_sum2, _min);
                _sum2 = vminq_f32(_sum2, _max);
                _sum3 = vmaxq_f32(_sum3, _min);
                _sum3 = vminq_f32(_sum3, _max);
            }
            else if (activation_type == 4)
            {
                float32x4_t _one = vdupq_n_f32(1.f);
                _sum0 = vnegq_f32(_sum0);
                _sum1 = vnegq_f32(_sum1);
                _sum2 = vnegq_f32(_sum2);
                _sum3 = vnegq_f32(_sum3);
                _sum0 = exp_ps(_sum0);
                _sum1 = exp_ps(_sum1);
                _sum2 = exp_ps(_sum2);
                _sum3 = exp_ps(_sum3);
                _sum0 = vaddq_f32(_sum0, _one);
                _sum1 = vaddq_f32(_sum1, _one);
                _sum2 = vaddq_f32(_sum2, _one);
                _sum3 = vaddq_f32(_sum3, _one);
                float32x4_t _outp0 = vrecpeq_f32(_sum0);
                float32x4_t _outp1 = vrecpeq_f32(_sum1);
                float32x4_t _outp2 = vrecpeq_f32(_sum2);
                float32x4_t _outp3 = vrecpeq_f32(_sum3);
                _outp0 = vmulq_f32(vrecpsq_f32(_sum0, _outp0), _outp0);
                _outp1 = vmulq_f32(vrecpsq_f32(_sum1, _outp1), _outp1);
                _outp2 = vmulq_f32(vrecpsq_f32(_sum2, _outp0), _outp2);
                _outp3 = vmulq_f32(vrecpsq_f32(_sum3, _outp1), _outp3);
//                 _outp0 = vmulq_f32(vrecpsq_f32(_sum0, _outp0), _outp0);
//                 _outp1 = vmulq_f32(vrecpsq_f32(_sum1, _outp1), _outp1);
//                 _outp2 = vmulq_f32(vrecpsq_f32(_sum2, _outp0), _outp2);
//                 _outp3 = vmulq_f32(vrecpsq_f32(_sum3, _outp1), _outp3);
                _sum0 = _outp0;
                _sum1 = _outp1;
                _sum2 = _outp2;
                _sum3 = _outp3;
            }

            vst1q_f32(outptr, _sum0);
            vst1q_f32(outptr + 4, _sum1);
            vst1q_f32(outptr + 8, _sum2);
            vst1q_f32(outptr + 12, _sum3);

            outptr += 16;
        }
        for (; i+1<size; i+=2)
        {
            const float* tmpptr = tmp.channel(i/8+(i%8)/4 + (i%4)/2);

            float32x4_t _sum0 = _bias0;
            float32x4_t _sum1 = _bias0;

            const float* kptr = (const float*)kernel + p * inch * 16;

            for (int q=0; q<inch; q++)
            {
//                 const float* r0 = bottom_blob.channel(q);

//                 float32x4_t _r0 = vld1q_f32(r0 + i*4);
//                 float32x4_t _r1 = vld1q_f32(r0 + (i+1)*4);

                float32x4_t _r0 = vld1q_f32( tmpptr );
                float32x4_t _r1 = vld1q_f32( tmpptr + 4 );

                float32x4_t _w0 = vld1q_f32( kptr );
                float32x4_t _w1 = vld1q_f32( kptr + 4 );
                float32x4_t _w2 = vld1q_f32( kptr + 8 );
                float32x4_t _w3 = vld1q_f32( kptr + 12 );

#if __aarch64__
                _sum0 = vmlaq_laneq_f32(_sum0, _w0, _r0, 0);
                _sum0 = vmlaq_laneq_f32(_sum0, _w1, _r0, 1);
                _sum0 = vmlaq_laneq_f32(_sum0, _w2, _r0, 2);
                _sum0 = vmlaq_laneq_f32(_sum0, _w3, _r0, 3);
                _sum1 = vmlaq_laneq_f32(_sum1, _w0, _r1, 0);
                _sum1 = vmlaq_laneq_f32(_sum1, _w1, _r1, 1);
                _sum1 = vmlaq_laneq_f32(_sum1, _w2, _r1, 2);
                _sum1 = vmlaq_laneq_f32(_sum1, _w3, _r1, 3);
#else
                _sum0 = vmlaq_lane_f32(_sum0, _w0, vget_low_f32(_r0), 0);
                _sum0 = vmlaq_lane_f32(_sum0, _w1, vget_low_f32(_r0), 1);
                _sum0 = vmlaq_lane_f32(_sum0, _w2, vget_high_f32(_r0), 0);
                _sum0 = vmlaq_lane_f32(_sum0, _w3, vget_high_f32(_r0), 1);
                _sum1 = vmlaq_lane_f32(_sum1, _w0, vget_low_f32(_r1), 0);
                _sum1 = vmlaq_lane_f32(_sum1, _w1, vget_low_f32(_r1), 1);
                _sum1 = vmlaq_lane_f32(_sum1, _w2, vget_high_f32(_r1), 0);
                _sum1 = vmlaq_lane_f32(_sum1, _w3, vget_high_f32(_r1), 1);
#endif

                tmpptr += 8;
                kptr += 16;
            }

            if (activation_type == 1)
            {
                float32x4_t _zero = vdupq_n_f32(0.f);
                _sum0 = vmaxq_f32(_sum0, _zero);
                _sum1 = vmaxq_f32(_sum1, _zero);
            }
            else if (activation_type == 2)
            {
                float32x4_t _zero = vdupq_n_f32(0.f);
                float32x4_t _slope = vdupq_n_f32(activation_params[0]);
                _sum0 = vbslq_f32(vcleq_f32(_sum0, _zero), vmulq_f32(_sum0, _slope), _sum0);
                _sum1 = vbslq_f32(vcleq_f32(_sum1, _zero), vmulq_f32(_sum1, _slope), _sum1);
            }
            else if (activation_type == 3)
            {
                float32x4_t _min = vdupq_n_f32(activation_params[0]);
                float32x4_t _max = vdupq_n_f32(activation_params[1]);
                _sum0 = vmaxq_f32(_sum0, _min);
                _sum0 = vminq_f32(_sum0, _max);
                _sum1 = vmaxq_f32(_sum1, _min);
                _sum1 = vminq_f32(_sum1, _max);
            }
            else if (activation_type == 4)
            {
                float32x4_t _one = vdupq_n_f32(1.f);
                _sum0 = vnegq_f32(_sum0);
                _sum1 = vnegq_f32(_sum1);
                _sum0 = exp_ps(_sum0);
                _sum1 = exp_ps(_sum1);
                _sum0 = vaddq_f32(_sum0, _one);
                _sum1 = vaddq_f32(_sum1, _one);
                float32x4_t _outp0 = vrecpeq_f32(_sum0);
                float32x4_t _outp1 = vrecpeq_f32(_sum1);
                _outp0 = vmulq_f32(vrecpsq_f32(_sum0, _outp0), _outp0);
                _outp1 = vmulq_f32(vrecpsq_f32(_sum1, _outp1), _outp1);
//                 _outp0 = vmulq_f32(vrecpsq_f32(_sum0, _outp0), _outp0);
//                 _outp1 = vmulq_f32(vrecpsq_f32(_sum1, _outp1), _outp1);
                _sum0 = _outp0;
                _sum1 = _outp1;
            }

            vst1q_f32(outptr, _sum0);
            vst1q_f32(outptr + 4, _sum1);

            outptr += 8;
        }
        for (; i<size; i++)
        {
            const float* tmpptr = tmp.channel(i/8+(i%8)/4 + (i%4)/2 + i%2);

            float32x4_t _sum = _bias0;

            const float* kptr = (const float*)kernel + p * inch * 16;

            for (int q=0; q<inch; q++)
            {
//                 const float* r0 = bottom_blob.channel(q);

//                 float32x4_t _val = vld1q_f32(r0 + i*4);

                float32x4_t _val = vld1q_f32( tmpptr );

                float32x4_t _w0 = vld1q_f32( kptr );
                float32x4_t _w1 = vld1q_f32( kptr + 4 );
                float32x4_t _w2 = vld1q_f32( kptr + 8 );
                float32x4_t _w3 = vld1q_f32( kptr + 12 );

#if __aarch64__
                _sum = vmlaq_laneq_f32(_sum, _w0, _val, 0);
                _sum = vmlaq_laneq_f32(_sum, _w1, _val, 1);
                _sum = vmlaq_laneq_f32(_sum, _w2, _val, 2);
                _sum = vmlaq_laneq_f32(_sum, _w3, _val, 3);
#else
                _sum = vmlaq_lane_f32(_sum, _w0, vget_low_f32(_val), 0);
                _sum = vmlaq_lane_f32(_sum, _w1, vget_low_f32(_val), 1);
                _sum = vmlaq_lane_f32(_sum, _w2, vget_high_f32(_val), 0);
                _sum = vmlaq_lane_f32(_sum, _w3, vget_high_f32(_val), 1);
#endif

                tmpptr += 4;
                kptr += 16;
            }

            if (activation_type == 1)
            {
                float32x4_t _zero = vdupq_n_f32(0.f);
                _sum = vmaxq_f32(_sum, _zero);
            }
            else if (activation_type == 2)
            {
                float32x4_t _zero = vdupq_n_f32(0.f);
                float32x4_t _slope = vdupq_n_f32(activation_params[0]);
                _sum = vbslq_f32(vcleq_f32(_sum, _zero), vmulq_f32(_sum, _slope), _sum);
            }
            else if (activation_type == 3)
            {
                float32x4_t _min = vdupq_n_f32(activation_params[0]);
                float32x4_t _max = vdupq_n_f32(activation_params[1]);
                _sum = vmaxq_f32(_sum, _min);
                _sum = vminq_f32(_sum, _max);
            }
            else if (activation_type == 4)
            {
                float32x4_t _one = vdupq_n_f32(1.f);
                _sum = vnegq_f32(_sum);
                _sum = exp_ps(_sum);
                _sum = vaddq_f32(_sum, _one);
                float32x4_t _outp = vrecpeq_f32(_sum);
                _outp = vmulq_f32(vrecpsq_f32(_sum, _outp), _outp);
//                 _outp = vmulq_f32(vrecpsq_f32(_sum, _outp), _outp);
                _sum = _outp;
            }

            vst1q_f32(outptr, _sum);

            outptr += 4;
        }

    }

//     // NOTE sgemm
//     for (; p<outch; p++)
//     {
//         Mat out0 = top_blob.channel(p);
//
//         const float bias0 = bias ? bias[p] : 0.f;
//
//         float* outptr0 = out0;
//
//         for (int i=0; i<size; i++)
//         {
//             float sum = bias0;
//
//             const float* kptr = _kernel.channel(p);
//
//             for (int q=0; q<inch; q++)
//             {
//                 const float* img0 = bottom_blob.channel(q);
//
//                 sum += img0[i] * kptr[0];
//                 kptr ++;
//             }
//
//             outptr0[i] = sum;
//         }
//     }
}
