// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

static void conv1x1s1_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const float* kernel = _kernel;
    const float* bias = _bias;

    #pragma omp parallel for
    for (int p=0; p<outch; p++)
    {
        Mat out = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;

        out.fill(bias0);

        int q = 0;

        for (; q+3<inch; q+=4)
        {
            float* outptr = out;

            const float* img0 = bottom_blob.channel(q);
            const float* img1 = bottom_blob.channel(q+1);
            const float* img2 = bottom_blob.channel(q+2);
            const float* img3 = bottom_blob.channel(q+3);

            const float* kernel0 = kernel + p*inch  + q;
            const float k0 = kernel0[0];
            const float k1 = kernel0[1];
            const float k2 = kernel0[2];
            const float k3 = kernel0[3];

            const float* r0 = img0;
            const float* r1 = img1;
            const float* r2 = img2;
            const float* r3 = img3;

            int size = outw * outh;

#if __ARM_NEON
            int nn = size >> 3;
            int remain = size & 7;
#else
            int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
            float32x4_t _k0 = vdupq_n_f32(k0);
            float32x4_t _k1 = vdupq_n_f32(k1);
            float32x4_t _k2 = vdupq_n_f32(k2);
            float32x4_t _k3 = vdupq_n_f32(k3);
#if __aarch64__
            for (; nn>0; nn--)
            {
                float32x4_t _p = vld1q_f32(r0);
                float32x4_t _pn = vld1q_f32(r0+4);

                float32x4_t _outp = vld1q_f32(outptr);
                float32x4_t _outpn = vld1q_f32(outptr+4);

                _outp = vfmaq_f32(_outp, _p, _k0);
                _outpn = vfmaq_f32(_outpn, _pn, _k0);

                float32x4_t _p1 = vld1q_f32(r1);
                float32x4_t _p1n = vld1q_f32(r1+4);

                _outp = vfmaq_f32(_outp, _p1, _k1);
                _outpn = vfmaq_f32(_outpn, _p1n, _k1);

                float32x4_t _p2 = vld1q_f32(r2);
                float32x4_t _p2n = vld1q_f32(r2+4);

                _outp = vfmaq_f32(_outp, _p2, _k2);
                _outpn = vfmaq_f32(_outpn, _p2n, _k2);

                float32x4_t _p3 = vld1q_f32(r3);
                float32x4_t _p3n = vld1q_f32(r3+4);

                _outp = vfmaq_f32(_outp, _p3, _k3);
                _outpn = vfmaq_f32(_outpn, _p3n, _k3);

                vst1q_f32(outptr, _outp);
                vst1q_f32(outptr+4, _outpn);

                r0 += 8;
                r1 += 8;
                r2 += 8;
                r3 += 8;
                outptr += 8;
            }
#else
            if (nn > 0)
            {
            asm volatile(
                "pld        [%2, #256]          \n"
                "vld1.f32   {d4-d7}, [%2 :128]! \n"
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "vld1.f32   {d0-d3}, [%1 :128]  \n"
                "vmla.f32   q0, q2, %q12        \n"
                "vmla.f32   q1, q3, %q12        \n"
                "pld        [%3, #256]          \n"
                "vld1.f32   {d4-d7}, [%3 :128]! \n"
                "vmla.f32   q0, q2, %q13        \n"
                "vmla.f32   q1, q3, %q13        \n"
                "pld        [%4, #256]          \n"
                "vld1.f32   {d4-d7}, [%4 :128]! \n"
                "vmla.f32   q0, q2, %q14        \n"
                "vmla.f32   q1, q3, %q14        \n"
                "pld        [%5, #256]          \n"
                "vld1.f32   {d4-d7}, [%5 :128]! \n"
                "vmla.f32   q0, q2, %q15        \n"
                "vmla.f32   q1, q3, %q15        \n"
                "pld        [%2, #256]          \n"
                "vld1.f32   {d4-d7}, [%2 :128]! \n"
                "subs       %0, #1              \n"
                "vst1.f32   {d0-d3}, [%1 :128]! \n"
                "bne        0b                  \n"
                "sub        %2, #32             \n"
                : "=r"(nn),     // %0
                  "=r"(outptr), // %1
                  "=r"(r0),     // %2
                  "=r"(r1),     // %3
                  "=r"(r2),     // %4
                  "=r"(r3)      // %5
                : "0"(nn),
                  "1"(outptr),
                  "2"(r0),
                  "3"(r1),
                  "4"(r2),
                  "5"(r3),
                  "w"(_k0),     // %12
                  "w"(_k1),     // %13
                  "w"(_k2),     // %14
                  "w"(_k3)      // %15
                : "cc", "memory", "q0", "q1", "q2", "q3"
            );
            }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain>0; remain--)
            {
                float sum = *r0 * k0;
                float sum1 = *r1 * k1;
                float sum2 = *r2 * k2;
                float sum3 = *r3 * k3;

                *outptr += sum + sum1 + sum2 + sum3;

                r0++;
                r1++;
                r2++;
                r3++;
                outptr++;
            }

        }

        for (; q<inch; q++)
        {
            float* outptr = out;

            const float* img0 = bottom_blob.channel(q);

            const float* kernel0 = kernel + p*inch  + q;
            const float k0 = kernel0[0];

            const float* r0 = img0;

            int size = outw * outh;

#if __ARM_NEON
            int nn = size >> 3;
            int remain = size & 7;
#else
            int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
            float32x4_t _k0 = vdupq_n_f32(k0);
#if __aarch64__
            for (; nn>0; nn--)
            {
                float32x4_t _p = vld1q_f32(r0);
                float32x4_t _outp = vld1q_f32(outptr);

                float32x4_t _pn = vld1q_f32(r0+4);
                float32x4_t _outpn = vld1q_f32(outptr+4);

                _outp = vfmaq_f32(_outp, _p, _k0);
                _outpn = vfmaq_f32(_outpn, _pn, _k0);

                vst1q_f32(outptr, _outp);
                vst1q_f32(outptr+4, _outpn);

                r0 += 8;
                outptr += 8;
            }
#else
            if (nn > 0)
            {
            asm volatile(
                "pld        [%2, #256]          \n"
                "vld1.f32   {d4-d7}, [%2 :128]! \n"
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "vld1.f32   {d0-d3}, [%1 :128]  \n"
                "vmla.f32   q0, q2, %q6         \n"
                "vmla.f32   q1, q3, %q6         \n"
                "pld        [%2, #256]          \n"
                "vld1.f32   {d4-d7}, [%2 :128]! \n"
                "subs       %0, #1              \n"
                "vst1.f32   {d0-d3}, [%1 :128]! \n"
                "bne        0b                  \n"
                "sub        %2, #32             \n"
                : "=r"(nn),     // %0
                  "=r"(outptr), // %1
                  "=r"(r0)      // %2
                : "0"(nn),
                  "1"(outptr),
                  "2"(r0),
                  "w"(_k0)      // %6
                : "cc", "memory", "q0", "q1", "q2", "q3"
            );
            }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain>0; remain--)
            {
                float sum = *r0 * k0;

                *outptr += sum;

                r0++;
                outptr++;
            }

        }
    }

}

static void conv1x1s2_neon(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2*outw + w;

    const float* kernel = _kernel;
    const float* bias = _bias;

    #pragma omp parallel for
    for (int p=0; p<outch; p++)
    {
        Mat out = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;

        out.fill(bias0);

        int q = 0;

        for (; q+3<inch; q+=4)
        {
            float* outptr = out;

            const float* img0 = bottom_blob.channel(q);
            const float* img1 = bottom_blob.channel(q+1);
            const float* img2 = bottom_blob.channel(q+2);
            const float* img3 = bottom_blob.channel(q+3);

            const float* kernel0 = kernel + p*inch  + q;
            const float k0 = kernel0[0];
            const float k1 = kernel0[1];
            const float k2 = kernel0[2];
            const float k3 = kernel0[3];

            const float* r0 = img0;
            const float* r1 = img1;
            const float* r2 = img2;
            const float* r3 = img3;

            for (int i = 0; i < outh; i++)
            {
#if __ARM_NEON
                int nn = outw >> 3;
                int remain = outw & 7;
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
                float32x4_t _k0 = vdupq_n_f32(k0);
                float32x4_t _k1 = vdupq_n_f32(k1);
                float32x4_t _k2 = vdupq_n_f32(k2);
                float32x4_t _k3 = vdupq_n_f32(k3);
#if __aarch64__
                for (; nn>0; nn--)
                {
                    float32x4x2_t _px2 = vld2q_f32(r0);
                    float32x4_t _p = _px2.val[0];
                    float32x4_t _outp = vld1q_f32(outptr);

                    float32x4x2_t _pnx2 = vld2q_f32(r0+8);
                    float32x4_t _pn = _pnx2.val[0];
                    float32x4_t _outpn = vld1q_f32(outptr+4);

                    _outp = vmlaq_f32(_outp, _p, _k0);
                    _outpn = vmlaq_f32(_outpn, _pn, _k0);

                    float32x4x2_t _p1x2 = vld2q_f32(r1);
                    float32x4_t _p1 = _p1x2.val[0];
                    float32x4x2_t _p1nx2 = vld2q_f32(r1+8);
                    float32x4_t _p1n = _p1nx2.val[0];

                    _outp = vmlaq_f32(_outp, _p1, _k1);
                    _outpn = vmlaq_f32(_outpn, _p1n, _k1);

                    float32x4x2_t _p2x2 = vld2q_f32(r2);
                    float32x4_t _p2 = _p2x2.val[0];
                    float32x4x2_t _p2nx2 = vld2q_f32(r2+8);
                    float32x4_t _p2n = _p2nx2.val[0];

                    _outp = vmlaq_f32(_outp, _p2, _k2);
                    _outpn = vmlaq_f32(_outpn, _p2n, _k2);

                    float32x4x2_t _p3x2 = vld2q_f32(r3);
                    float32x4_t _p3 = _p3x2.val[0];
                    float32x4x2_t _p3nx2 = vld2q_f32(r3+8);
                    float32x4_t _p3n = _p3nx2.val[0];

                    _outp = vmlaq_f32(_outp, _p3, _k3);
                    _outpn = vmlaq_f32(_outpn, _p3n, _k3);

                    vst1q_f32(outptr, _outp);
                    vst1q_f32(outptr+8, _outpn);

                    r0 += 16;
                    r1 += 16;
                    r2 += 16;
                    r3 += 16;
                    outptr += 8;
                }
#else
                if (nn > 0)
                {
                asm volatile(
                    "pld        [%2, #512]          \n"
                    "vld2.f32   {d4-d7}, [%2]!      \n"
                    "vld2.f32   {d16-d19}, [%2]!    \n"
                    "0:                             \n"
                    "pld        [%1, #256]          \n"
                    "vld1.f32   {d0-d3}, [%1]       \n"
                    "vmla.f32   q0, q2, %q12        \n"
                    "vmla.f32   q1, q8, %q12        \n"
                    "pld        [%3, #512]          \n"
                    "vld2.f32   {d4-d7}, [%3]!      \n"
                    "vld2.f32   {d16-d19}, [%3]!    \n"
                    "vmla.f32   q0, q2, %q13        \n"
                    "vmla.f32   q1, q8, %q13        \n"
                    "pld        [%4, #512]          \n"
                    "vld2.f32   {d4-d7}, [%4]!      \n"
                    "vld2.f32   {d16-d19}, [%4]!    \n"
                    "vmla.f32   q0, q2, %q14        \n"
                    "vmla.f32   q1, q8, %q14        \n"
                    "pld        [%5, #512]          \n"
                    "vld2.f32   {d4-d7}, [%5]!      \n"
                    "vld2.f32   {d16-d19}, [%5]!    \n"
                    "vmla.f32   q0, q2, %q15        \n"
                    "vmla.f32   q1, q8, %q15        \n"
                    "pld        [%2, #512]          \n"
                    "vld2.f32   {d4-d7}, [%2]!      \n"
                    "vld2.f32   {d16-d19}, [%2]!    \n"
                    "subs       %0, #1              \n"
                    "vst1.f32   {d0-d3}, [%1]!      \n"
                    "bne        0b                  \n"
                    "sub        %2, #64             \n"
                    : "=r"(nn),     // %0
                      "=r"(outptr), // %1
                      "=r"(r0),     // %2
                      "=r"(r1),     // %3
                      "=r"(r2),     // %4
                      "=r"(r3)      // %5
                    : "0"(nn),
                      "1"(outptr),
                      "2"(r0),
                      "3"(r1),
                      "4"(r2),
                      "5"(r3),
                      "w"(_k0),     // %12
                      "w"(_k1),     // %13
                      "w"(_k2),     // %14
                      "w"(_k3)      // %15
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain>0; remain--)
                {
                    float sum = *r0 * k0;
                    float sum1 = *r1 * k1;
                    float sum2 = *r2 * k2;
                    float sum3 = *r3 * k3;

                    *outptr += sum + sum1 + sum2 + sum3;

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    r3 += 2;
                    outptr++;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
                r3 += tailstep;
            }

        }

        for (; q<inch; q++)
        {
            float* outptr = out;

            const float* img0 = bottom_blob.channel(q);

            const float* kernel0 = kernel + p*inch  + q;
            const float k0 = kernel0[0];

            const float* r0 = img0;

            for (int i = 0; i < outh; i++)
            {
#if __ARM_NEON
                int nn = outw >> 3;
                int remain = outw & 7;
#else
                int remain = outw;
#endif // __ARM_NEON

#if __ARM_NEON
                float32x4_t _k0 = vdupq_n_f32(k0);
#if __aarch64__
                for (; nn>0; nn--)
                {
                    float32x4x2_t _px2 = vld2q_f32(r0);
                    float32x4_t _p = _px2.val[0];
                    float32x4_t _outp = vld1q_f32(outptr);

                    float32x4x2_t _pnx2 = vld2q_f32(r0+8);
                    float32x4_t _pn = _pnx2.val[0];
                    float32x4_t _outpn = vld1q_f32(outptr+4);

                    _outp = vmlaq_f32(_outp, _p, _k0);
                    _outpn = vmlaq_f32(_outpn, _pn, _k0);

                    vst1q_f32(outptr, _outp);
                    vst1q_f32(outptr+4, _outpn);

                    r0 += 16;
                    outptr += 8;
                }
#else
                if (nn > 0)
                {
                asm volatile(
                    "pld        [%2, #512]          \n"
                    "vld2.f32   {d4-d7}, [%2]!      \n"
                    "vld2.f32   {d16-d19}, [%2]!    \n"
                    "0:                             \n"
                    "pld        [%1, #256]          \n"
                    "vld1.f32   {d0-d3}, [%1]       \n"
                    "vmla.f32   q0, q2, %q6         \n"
                    "vmla.f32   q1, q8, %q6         \n"
                    "pld        [%2, #512]          \n"
                    "vld2.f32   {d4-d7}, [%2]!      \n"
                    "vld2.f32   {d16-d19}, [%2]!    \n"
                    "subs       %0, #1              \n"
                    "vst1.f32   {d0-d3}, [%1]!      \n"
                    "bne        0b                  \n"
                    "sub        %2, #64             \n"
                    : "=r"(nn),     // %0
                      "=r"(outptr), // %1
                      "=r"(r0)      // %2
                    : "0"(nn),
                      "1"(outptr),
                      "2"(r0),
                      "w"(_k0)      // %6
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain>0; remain--)
                {
                    float sum = *r0 * k0;

                    *outptr += sum;

                    r0 += 2;
                    outptr++;
                }

                r0 += tailstep;
            }

        }
    }

}
