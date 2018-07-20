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

#define M_Protect(a) ((a) > 127 ? (127) : ((a) < (-128) ? (-128) : (a)))

static void check_overflow(int sum, int &count, int &count_all)
{
    if (sum > 32767 || sum < -32768)
    {
        count++;
    }

    count_all++;
}

static void conv1x1_quantize_int8_transform_kernel(const Mat &_kernel, Mat &kernel_tm, int inch, int outch, const stQuantizeParams &scale)
{
    float ufKernelFactor = 0.f;

    //initial quantized kernel Mat
    kernel_tm.create(1 * inch * outch, 1);

    const float *kernel = _kernel;
    ufKernelFactor = scale.weightScale;

    //quantize the kernel weight
    signed char *kernel_s = (signed char *)kernel_tm.data;
    for (int p = 0; p < outch; p++)
    {
        for (int q = 0; q < inch; q++)
        {
            int tmp = p * inch * 1 + q * 1;
            float kernel_tmp;

            for (int idx = 0; idx < 1; idx++)
            {
                if (kernel[tmp + idx] >= 0)
                {
                    kernel_tmp = ufKernelFactor * kernel[tmp + idx] + 0.5;
                }
                else
                {
                    kernel_tmp = ufKernelFactor * kernel[tmp + idx] - 0.5;
                }

                kernel_s[tmp + idx] = (signed char)(M_Protect(kernel_tmp));
            }
        }
    }
}

static void conv3x3_quantize_int8_transform_kernel(const Mat &_kernel, Mat &kernel_tm, int inch, int outch, const stQuantizeParams &scale)
{
    float ufKernelFactor = 0.f;

    //initial quantized kernel Mat
    kernel_tm.create(9 * inch * outch, 1);

    const float *kernel = _kernel;
    ufKernelFactor = scale.weightScale;

    //quantize the kernel weight
    signed char *kernel_s = (signed char *)kernel_tm.data;
    for (int p = 0; p < outch; p++)
    {
        for (int q = 0; q < inch; q++)
        {
            int tmp = p * inch * 9 + q * 9;
            float kernel_tmp;

            for (int idx = 0; idx < 9; idx++)
            {
                if (kernel[tmp + idx] >= 0)
                {
                    kernel_tmp = ufKernelFactor * kernel[tmp + idx] + 0.5;
                }
                else
                {
                    kernel_tmp = ufKernelFactor * kernel[tmp + idx] - 0.5;
                }

                kernel_s[tmp + idx] = (signed char)(M_Protect(kernel_tmp));
            }
        }
    }
}

static void convdw3x3_quantize_int8_transform_kernel(const Mat &_kernel, Mat &kernel_tm, int group, const stQuantizeParams &scale)
{
    float ufKernelFactor = 0.f;

    //initial quantized kernel Mat
    kernel_tm.create(9 * group);

    const float *kernel = _kernel;
    ufKernelFactor = scale.weightScale;

    //quantize the kernel weight
    signed char *kernel_s = (signed char *)kernel_tm.data;
    for (int g = 0; g < group; g++)
    {
        int tmp = g * 9;
        float kernel_tmp;

        for (int idx = 0; idx < 9; idx++)
        {
            if (kernel[tmp + idx] >= 0)
            {
                kernel_tmp = ufKernelFactor * kernel[tmp + idx] + 0.5;
            }
            else
            {
                kernel_tmp = ufKernelFactor * kernel[tmp + idx] - 0.5;
            }

            kernel_s[tmp + idx] = (signed char)(M_Protect(kernel_tmp));
        }
    }
}

static void conv_quantize(const Mat &bottom_blob, Mat &bottom_blob_s8, const float dataScale)
{
    float ufDataFactor = dataScale;
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;
    int size = w * h;

#if NCNN_INT8_INFO
    fprintf(stderr, "scale %f\n", dataScale);
#endif    

    #pragma omp parallel for
    for (int qidx = 0; qidx < inch; qidx++)
    {
        const float *img0 = bottom_blob.channel(qidx);
        signed char *img0_s8 = bottom_blob_s8.channel(qidx);

        for (int i = 0; i < size; i++)
        {
            signed int tmp;

            if (img0[i] >= 0)
            {
                tmp = (int)(img0[i] * ufDataFactor + 0.5);
            }
            else
            {
                tmp = (int)(img0[i] * ufDataFactor - 0.5);
            }

            img0_s8[i] = (signed char)M_Protect(tmp);
        }
    }
}

static void conv_dequantize(Mat &top_blob, const Mat &_bias, const float dataScale, const float weightScale)
{
    //float ufDataFactor = 0.f;
    //float ufKernelFactor = 0.f;
    float ufReverseFactor = 0.f;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    int size = outh * outw;

    const float *bias = _bias;

    if (0 != dataScale * weightScale)
    {
        ufReverseFactor = 1 / (dataScale * weightScale);
    }

    #pragma omp parallel for
    for (int p = 0; p < outch; p++)
    {
        const float *img0 = top_blob.channel(p);
        signed int *img0_s32 = (signed int *)img0;
        float *img0_f32 = (float *)img0;

        float bias0 = bias ? bias[p] : 0.f;

        for (int i = 0; i < size; i++)
        {
            *img0_f32++ = ((float)(*img0_s32++)) * ufReverseFactor + bias0;
        }
    }
}

static void conv1x1s1_s8(const Mat &bottom_blob, Mat &top_blob, const Mat &_kernel)
{
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const float *kernel = _kernel;
#if NCNN_INT8_INFO
    int count_overflow = 0;
    int count_result = 0;
#endif    

    #pragma omp parallel for
    for (int p = 0; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        out0.fill(0.f);

        int q = 0;

        for (; q + 7 < inch; q += 8)
        {
            float *outptr0 = out0;

            int *outptr0_s32 = (int *)outptr0;

            const float *img0 = bottom_blob.channel(q);
            const float *img1 = bottom_blob.channel(q + 1);
            const float *img2 = bottom_blob.channel(q + 2);
            const float *img3 = bottom_blob.channel(q + 3);
            const float *img4 = bottom_blob.channel(q + 4);
            const float *img5 = bottom_blob.channel(q + 5);
            const float *img6 = bottom_blob.channel(q + 6);
            const float *img7 = bottom_blob.channel(q + 7);

            const signed char *kernel0 = (const signed char *)kernel + p * inch + q;

            const signed char *r0 = (signed char *)img0;
            const signed char *r1 = (signed char *)img1;
            const signed char *r2 = (signed char *)img2;
            const signed char *r3 = (signed char *)img3;
            const signed char *r4 = (signed char *)img4;
            const signed char *r5 = (signed char *)img5;
            const signed char *r6 = (signed char *)img6;
            const signed char *r7 = (signed char *)img7;

            int size = outw * outh;
            int remain = size;

            for (; remain > 0; remain--)
            {
                //ToDo Neon
                int sum0 = (int)*r0 * (int)kernel0[0] + (int)*r1 * (int)kernel0[1] +
                           (int)*r2 * (int)kernel0[2] + (int)*r3 * (int)kernel0[3] +
                           (int)*r4 * (int)kernel0[4] + (int)*r5 * (int)kernel0[5] +
                           (int)*r6 * (int)kernel0[6] + (int)*r7 * (int)kernel0[7];

                *outptr0_s32 += sum0;
#if NCNN_INT8_INFO
                check_overflow(sum0, count_overflow, count_result);
#endif                

                r0++;
                r1++;
                r2++;
                r3++;
                r4++;
                r5++;
                r6++;
                r7++;
                outptr0_s32++;
            }
        }

        for (; q < inch; q++)
        {
            float *outptr0 = out0;

            int *outptr0_s32 = (int *)outptr0;

            const float *img0 = bottom_blob.channel(q);
            const signed char *img0_s8 = (signed char *)img0;
            const signed char *r0 = img0_s8;

            const signed char *kernel0 = (const signed char *)kernel + p * inch + q;
            const signed char k0 = kernel0[0];

            int size = outw * outh;
            int remain = size;

            for (; remain > 0; remain--)
            {
                int sum0 = (int)(*r0) * (int)k0;

                *outptr0_s32 += sum0;
#if NCNN_INT8_INFO
                check_overflow(sum0, count_overflow, count_result);
#endif                

                r0++;
                outptr0_s32++;
            }
        }
    }
#if NCNN_INT8_INFO
    if (count_overflow)
        printf("overflow : %d, all : %d, error rate : %.3f\n", count_overflow, count_result, (float)count_overflow / count_result * 100);
#endif        
}

static void conv1x1s2_s8(const Mat &bottom_blob, Mat &top_blob, const Mat &_kernel)
{
    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2*outw + w;
    const signed char *kernel = _kernel;
#if NCNN_INT8_INFO
    int count_overflow = 0;
    int count_result = 0;
#endif    

    #pragma omp parallel for
    for (int p = 0; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        out0.fill(0.f);

        int q = 0;

        for (; q + 7 < inch; q += 8)
        {
            float *outptr0 = out0;

            int *outptr0_s32 = (int *)outptr0; 

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

                    *outptr0_s32 += sum0;
#if NCNN_INT8_INFO
                    check_overflow(sum0, count_overflow, count_result);
#endif                    

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    r3 += 2;
                    r4 += 2;
                    r5 += 2;
                    r6 += 2;
                    r7 += 2;
                    outptr0_s32++;
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

        for (; q < inch; q++)
        {
            float *outptr0 = out0;

            int *outptr0_s32 = (int *)outptr0;

            const signed char *r0 = bottom_blob.channel(q);

            const signed char *kernel0 = (const signed char *)kernel + p * inch + q;

            for(int i = 0; i < outh; i++)
            {
                int remain = outw;

                for (; remain > 0; remain--)
                {
                    //ToDo Neon
                    int sum0 = (int)*r0 * (int)kernel0[0];

                    *outptr0_s32 += sum0;
#if NCNN_INT8_INFO
                    check_overflow(sum0, count_overflow, count_result);
#endif                    

                    r0 += 2;
                    outptr0_s32++;
                }

                r0 += tailstep;
            }
        }
    }
#if NCNN_INT8_INFO
    if (count_overflow)
        printf("overflow : %d, all : %d, error rate : %.3f\n", count_overflow, count_result, (float)count_overflow / count_result * 100);
#endif
}

static void conv3x3s1_s8(const Mat &bottom_blob, Mat &top_blob, const Mat &_kernel)
{
    int w = bottom_blob.w;
    //int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const signed char *kernel = _kernel;

#if NCNN_INT8_INFO
    int count_overflow = 0;
    int count_result = 0;
#endif    

    #pragma omp parallel for
    for (int p = 0; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        out0.fill(0);

        const signed char *kernel0 = (const signed char *)kernel + p * inch * 9;

        for (int q = 0; q < inch; q++)
        {
            float *outptr0 = out0;
            //float *outptr0n = outptr0 + outw;

            int *outptr0_s32 = (int *)outptr0;
            //int *outptr0n_s32 = (int *)outptr0n;

            const float *img1 = bottom_blob.channel(q);
            const signed char *img0 = (signed char *)img1;

            const signed char *r0 = img0;
            const signed char *r1 = img0 + w;
            const signed char *r2 = img0 + w * 2;
            //const signed char *r3 = img0 + w * 3;

            //const signed char *k00 = kernel0;
            //const signed char *k03 = kernel0 + 3;
            //const signed char *k06 = kernel0 + 6;

            for (int i = 0; i < outh; i++)
            {
                int remain = outw;

                for (; remain > 0; remain--)
                {
                    int sum0 = 0;

                    sum0 += (int)r0[0] * kernel0[0];
                    sum0 += (int)r0[1] * kernel0[1];
                    sum0 += (int)r0[2] * kernel0[2];
                    sum0 += (int)r1[0] * kernel0[3];
                    sum0 += (int)r1[1] * kernel0[4];
                    sum0 += (int)r1[2] * kernel0[5];
                    sum0 += (int)r2[0] * kernel0[6];
                    sum0 += (int)r2[1] * kernel0[7];
                    sum0 += (int)r2[2] * kernel0[8];

#if NCNN_INT8_INFO
                    check_overflow(sum0, count_overflow, count_result);
#endif
                    *outptr0_s32 += sum0;

                    r0++;
                    r1++;
                    r2++;
                    outptr0_s32++;
                }

                r0 += 2;
                r1 += 2;
                r2 += 2;
            }

            kernel0 += 9;
        }
    }
#if NCNN_INT8_INFO
    if (count_overflow)
        printf("overflow : %d, all : %d, error rate : %.3f\n", count_overflow, count_result, (float)count_overflow / count_result * 100);
#endif        
}

static void conv3x3s2_s8(const Mat &bottom_blob, Mat &top_blob, const Mat &_kernel)
{
    int w = bottom_blob.w;
    //int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2 * outw + w;

    const signed char *kernel = _kernel;
#if NCNN_INT8_INFO
    int count_overflow = 0;
    int count_result = 0;
#endif    

    #pragma omp parallel for
    for (int p = 0; p < outch; p++)
    {
        Mat out0 = top_blob.channel(p);

        out0.fill(0.f);

        const signed char *kernel0 = (const signed char *)kernel + p * inch * 9;

        for (int q = 0; q < inch; q++)
        {
            float *outptr0 = out0;

            int *outptr0_s32 = (int *)outptr0;

            const float *img1 = bottom_blob.channel(q);
            const signed char *img0 = (signed char *)img1;

            const signed char *r0 = img0;
            const signed char *r1 = img0 + w;
            const signed char *r2 = img0 + w * 2;

            //const signed char *k00 = kernel0;
            //const signed char *k01 = kernel0 + 3;
            //const signed char *k02 = kernel0 + 6;

            for (int i = 0; i < outh; i++)
            {
                int remain = outw;

                for (; remain > 0; remain--)
                {
                    int sum0 = 0;

                    sum0 += (int)r0[0] * (int)kernel0[0];
                    sum0 += (int)r0[1] * (int)kernel0[1];
                    sum0 += (int)r0[2] * (int)kernel0[2];
                    sum0 += (int)r1[0] * (int)kernel0[3];
                    sum0 += (int)r1[1] * (int)kernel0[4];
                    sum0 += (int)r1[2] * (int)kernel0[5];
                    sum0 += (int)r2[0] * (int)kernel0[6];
                    sum0 += (int)r2[1] * (int)kernel0[7];
                    sum0 += (int)r2[2] * (int)kernel0[8];
#if NCNN_INT8_INFO
                    check_overflow(sum0, count_overflow, count_result);
#endif                    

                    *outptr0_s32 += sum0;

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    outptr0_s32++;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }

            kernel0 += 9;
        }
    }
#if NCNN_INT8_INFO
    if (count_overflow)
        printf("overflow : %d, all : %d, error rate : %.3f\n", count_overflow, count_result, (float)count_overflow / count_result * 100.0);
#endif        
}

static void convdw3x3s1_s8(const Mat &bottom_blob, Mat &top_blob, const Mat &_kernel)
{
    int w = bottom_blob.w;
    //int h = bottom_blob.h;
    //int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const signed char *kernel = _kernel;
#if NCNN_INT8_INFO
    int count_overflow = 0;
    int count_result = 0;
#endif    

    #pragma omp parallel for
    for (int p = 0; p < outch; p++)
    {
        Mat out = top_blob.channel(p);

        out.fill(0.f);

        const signed char *kernel0 = (const signed char *)kernel + p * 9;

        float *outptr = out;

        int *outptr_s32 = (int *)outptr;

        const float *img1 = bottom_blob.channel(p);
        const signed char *img0 = (signed char *)img1;

        const signed char *r0 = img0;
        const signed char *r1 = img0 + w;
        const signed char *r2 = img0 + w * 2;

        //const signed char *k0 = kernel0;
        //const signed char *k1 = kernel0 + 3;
        //const signed char *k2 = kernel0 + 6;

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

                *outptr_s32 += sum;
#if NCNN_INT8_INFO
                check_overflow(sum, count_overflow, count_result);
#endif                

                r0++;
                r1++;
                r2++;
                outptr_s32++;
            }

            r0 += 2;
            r1 += 2;
            r2 += 2;
        }
    }
#if NCNN_INT8_INFO
    if (count_overflow)
        printf("overflow : %d, all : %d, error rate : %.3f\n", count_overflow, count_result, (float)count_overflow / count_result * 100); 
#endif           
}

static void convdw3x3s2_s8(const Mat &bottom_blob, Mat &top_blob, const Mat &_kernel)
{
    int w = bottom_blob.w;
    //int h = bottom_blob.h;
    //int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2 * outw + w;

    const signed char *kernel = _kernel;
#if NCNN_INT8_INFO
    int count_overflow = 0;
    int count_result = 0;
#endif    

    #pragma omp parallel for
    for (int p = 0; p < outch; p++)
    {
        Mat out = top_blob.channel(p);
        out.fill(0.f);

        const signed char *kernel0 = (const signed char *)kernel + p * 9;

        float *outptr = out;
        int *outptr_s32 = (int *)outptr;

        const float *img1 = bottom_blob.channel(p);
        const signed char *img0 = (signed char *)img1;

        const signed char *r0 = img0;
        const signed char *r1 = img0 + w;
        const signed char *r2 = img0 + w * 2;

        //const signed char *k0 = kernel0;
        //const signed char *k1 = kernel0 + 3;
        //const signed char *k2 = kernel0 + 6;

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

                *outptr_s32 += sum;
#if NCNN_INT8_INFO
                check_overflow(sum, count_overflow, count_result);
#endif                

                r0 += 2;
                r1 += 2;
                r2 += 2;
                outptr_s32++;
            }

            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
        }
    }
#if NCNN_INT8_INFO
    if (count_overflow)
        printf("overflow : %d, all : %d, error rate : %.3f\n", count_overflow, count_result, (float)count_overflow / count_result * 100);   
#endif            
}
