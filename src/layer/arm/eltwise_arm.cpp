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

#include "eltwise_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn {

DEFINE_LAYER_CREATOR(Eltwise_arm)

int Eltwise_arm::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int size = w * h;

    Mat& top_blob = top_blobs[0];
    top_blob.create(w, h, channels);
    if (top_blob.empty())
        return -100;

    if (op_type == Operation_PROD)
    {
        // first blob
        const Mat& bottom_blob1 = bottom_blobs[1];
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            const float* ptr1 = bottom_blob1.channel(q);
            float* outptr = top_blob.channel(q);

#if __ARM_NEON
            int nn = size >> 2;
            int remain = size - (nn << 2);
#else
            int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
            for (; nn>0; nn--)
            {
                float32x4_t _ptr = vld1q_f32(ptr);
                float32x4_t _ptr1 = vld1q_f32(ptr1);
                float32x4_t _p = vmulq_f32(_ptr, _ptr1);
                vst1q_f32(outptr, _p);

                ptr += 4;
                ptr1 += 4;
                outptr += 4;
            }
#else
            if (nn > 0)
            {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #128]          \n"
                "pld        [%2, #128]          \n"
                "vld1.f32   {d0-d1}, [%1 :128]! \n"
                "vld1.f32   {d2-d3}, [%2 :128]! \n"
                "vmul.f32   q0, q0, q1          \n"
                "subs       %0, #1              \n"
                "vst1.f32   {d0-d1}, [%3 :128]! \n"
                "bne        0b                  \n"
                : "=r"(nn),     // %0
                  "=r"(ptr),    // %1
                  "=r"(ptr1),   // %2
                  "=r"(outptr)  // %3
                : "0"(nn),
                  "1"(ptr),
                  "2"(ptr1),
                  "3"(outptr)
                : "cc", "memory", "q0", "q1"
            );
            }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain>0; remain--)
            {
                *outptr = *ptr * *ptr1;

                ptr++;
                ptr1++;
                outptr++;
            }
        }

        for (size_t b=2; b<bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob1 = bottom_blobs[b];
            #pragma omp parallel for
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

#if __ARM_NEON
                int nn = size >> 2;
                int remain = size - (nn << 2);
#else
                int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                for (; nn>0; nn--)
                {
                    float32x4_t _ptr = vld1q_f32(ptr);
                    float32x4_t _p = vld1q_f32(outptr);
                    _p = vmulq_f32(_ptr, _p);
                    vst1q_f32(outptr, _p);

                    ptr += 4;
                    outptr += 4;
                }
#else
                if (nn > 0)
                {
                asm volatile(
                    "0:                             \n"
                    "pld        [%1, #128]          \n"
                    "pld        [%2, #128]          \n"
                    "vld1.f32   {d0-d1}, [%1 :128]! \n"
                    "vld1.f32   {d2-d3}, [%2 :128]  \n"
                    "vmul.f32   q0, q0, q1          \n"
                    "subs       %0, #1              \n"
                    "vst1.f32   {d0-d1}, [%2 :128]! \n"
                    "bne        0b                  \n"
                    : "=r"(nn),     // %0
                      "=r"(ptr),    // %1
                      "=r"(outptr)  // %2
                    : "0"(nn),
                      "1"(ptr),
                      "2"(outptr)
                    : "cc", "memory", "q0", "q1"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain>0; remain--)
                {
                    *outptr *= *ptr;

                    ptr++;
                    outptr++;
                }
            }
        }
    }
    else if (op_type == Operation_SUM)
    {
        if (num_coeff == 0)
        {
            // first blob
            const Mat& bottom_blob1 = bottom_blobs[1];
            #pragma omp parallel for
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                const float* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

#if __ARM_NEON
                int nn = size >> 2;
                int remain = size - (nn << 2);
#else
                int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                for (; nn>0; nn--)
                {
                    float32x4_t _ptr = vld1q_f32(ptr);
                    float32x4_t _ptr1 = vld1q_f32(ptr1);
                    float32x4_t _p = vaddq_f32(_ptr, _ptr1);
                    vst1q_f32(outptr, _p);

                    ptr += 4;
                    ptr1 += 4;
                    outptr += 4;
                }
#else
                if (nn > 0)
                {
                asm volatile(
                    "0:                             \n"
                    "pld        [%1, #128]          \n"
                    "pld        [%2, #128]          \n"
                    "vld1.f32   {d0-d1}, [%1 :128]! \n"
                    "vld1.f32   {d2-d3}, [%2 :128]! \n"
                    "vadd.f32   q0, q0, q1          \n"
                    "subs       %0, #1              \n"
                    "vst1.f32   {d0-d1}, [%3 :128]! \n"
                    "bne        0b                  \n"
                    : "=r"(nn),     // %0
                      "=r"(ptr),    // %1
                      "=r"(ptr1),   // %2
                      "=r"(outptr)  // %3
                    : "0"(nn),
                      "1"(ptr),
                      "2"(ptr1),
                      "3"(outptr)
                    : "cc", "memory", "q0", "q1"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain>0; remain--)
                {
                    *outptr = *ptr + *ptr1;

                    ptr++;
                    ptr1++;
                    outptr++;
                }
            }

            for (size_t b=2; b<bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
                #pragma omp parallel for
                for (int q=0; q<channels; q++)
                {
                    const float* ptr = bottom_blob1.channel(q);
                    float* outptr = top_blob.channel(q);

#if __ARM_NEON
                    int nn = size >> 2;
                    int remain = size - (nn << 2);
#else
                    int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                    for (; nn>0; nn--)
                    {
                        float32x4_t _ptr = vld1q_f32(ptr);
                        float32x4_t _p = vld1q_f32(outptr);
                        _p = vaddq_f32(_ptr, _p);
                        vst1q_f32(outptr, _p);

                        ptr += 4;
                        outptr += 4;
                    }
#else
                    if (nn > 0)
                    {
                    asm volatile(
                        "0:                             \n"
                        "pld        [%1, #128]          \n"
                        "pld        [%2, #128]          \n"
                        "vld1.f32   {d0-d1}, [%1 :128]! \n"
                        "vld1.f32   {d2-d3}, [%2 :128]  \n"
                        "vadd.f32   q0, q0, q1          \n"
                        "subs       %0, #1              \n"
                        "vst1.f32   {d0-d1}, [%2 :128]! \n"
                        "bne        0b                  \n"
                        : "=r"(nn),     // %0
                          "=r"(ptr),    // %1
                          "=r"(outptr)  // %2
                        : "0"(nn),
                          "1"(ptr),
                          "2"(outptr)
                        : "cc", "memory", "q0", "q1"
                    );
                    }
#endif // __aarch64__
#endif // __ARM_NEON
                    for (; remain>0; remain--)
                    {
                        *outptr += *ptr;

                        ptr++;
                        outptr++;
                    }
                }
            }
        }
        else
        {
            const float* coeffs_ptr = coeffs;

            // first blob
            const Mat& bottom_blob1 = bottom_blobs[1];
            float coeff0 = coeffs_ptr[0];
            float coeff1 = coeffs_ptr[1];
            #pragma omp parallel for
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                const float* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

#if __ARM_NEON
                int nn = size >> 2;
                int remain = size - (nn << 2);
#else
                int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
                float32x4_t _coeff0 = vdupq_n_f32(coeff0);
                float32x4_t _coeff1 = vdupq_n_f32(coeff1);
#if __aarch64__
                for (; nn>0; nn--)
                {
                    float32x4_t _ptr = vld1q_f32(ptr);
                    float32x4_t _ptr1 = vld1q_f32(ptr1);
                    float32x4_t _p = vmulq_f32(_ptr, _coeff0);
                    _p = vmlaq_f32(_p, _ptr1, _coeff1);
                    vst1q_f32(outptr, _p);

                    ptr += 4;
                    ptr1 += 4;
                    outptr += 4;
                }
#else
                if (nn > 0)
                {
                asm volatile(
                    "0:                             \n"
                    "pld        [%1, #128]          \n"
                    "pld        [%2, #128]          \n"
                    "vld1.f32   {d0-d1}, [%1 :128]! \n"
                    "vld1.f32   {d2-d3}, [%2 :128]! \n"
                    "vmul.f32   q0, q0, %q8         \n"
                    "vmla.f32   q0, q1, %q9         \n"
                    "subs       %0, #1              \n"
                    "vst1.f32   {d0-d1}, [%3 :128]! \n"
                    "bne        0b                  \n"
                    : "=r"(nn),     // %0
                      "=r"(ptr),    // %1
                      "=r"(ptr1),   // %2
                      "=r"(outptr)  // %3
                    : "0"(nn),
                      "1"(ptr),
                      "2"(ptr1),
                      "3"(outptr),
                      "w"(_coeff0), // %8
                      "w"(_coeff1)  // %9
                    : "cc", "memory", "q0", "q1"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain>0; remain--)
                {
                    *outptr = *ptr * coeff0 + *ptr1 * coeff1;

                    ptr++;
                    ptr1++;
                    outptr++;
                }
            }

            for (size_t b=2; b<bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
                float coeff = coeffs_ptr[b];
                #pragma omp parallel for
                for (int q=0; q<channels; q++)
                {
                    const float* ptr = bottom_blob1.channel(q);
                    float* outptr = top_blob.channel(q);

#if __ARM_NEON
                    int nn = size >> 2;
                    int remain = size - (nn << 2);
#else
                    int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
                    float32x4_t _coeff = vdupq_n_f32(coeff);
#if __aarch64__
                    for (; nn>0; nn--)
                    {
                        float32x4_t _ptr = vld1q_f32(ptr);
                        float32x4_t _p = vld1q_f32(outptr);
                        _p = vmlaq_f32(_p, _ptr, _coeff);
                        vst1q_f32(outptr, _p);

                        ptr += 4;
                        outptr += 4;
                    }
#else
                    if (nn > 0)
                    {
                    asm volatile(
                        "0:                             \n"
                        "pld        [%1, #128]          \n"
                        "pld        [%2, #128]          \n"
                        "vld1.f32   {d0-d1}, [%1 :128]! \n"
                        "vld1.f32   {d2-d3}, [%2 :128]  \n"
                        "vmla.f32   q1, q0, %q6         \n"
                        "subs       %0, #1              \n"
                        "vst1.f32   {d0-d1}, [%2 :128]! \n"
                        "bne        0b                  \n"
                        : "=r"(nn),     // %0
                          "=r"(ptr),    // %1
                          "=r"(outptr)  // %2
                        : "0"(nn),
                          "1"(ptr),
                          "2"(outptr),
                          "w"(_coeff)   // %6
                        : "cc", "memory", "q0", "q1"
                    );
                    }
#endif // __aarch64__
#endif // __ARM_NEON
                    for (; remain>0; remain--)
                    {
                        *outptr += *ptr * coeff;

                        ptr++;
                        outptr++;
                    }
                }
            }
        }
    }
    else if (op_type == Operation_MAX)
    {
        // first blob
        const Mat& bottom_blob1 = bottom_blobs[1];
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            const float* ptr1 = bottom_blob1.channel(q);
            float* outptr = top_blob.channel(q);

#if __ARM_NEON
            int nn = size >> 2;
            int remain = size - (nn << 2);
#else
            int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
            for (; nn>0; nn--)
            {
                float32x4_t _ptr = vld1q_f32(ptr);
                float32x4_t _ptr1 = vld1q_f32(ptr1);
                float32x4_t _p = vmaxq_f32(_ptr, _ptr1);
                vst1q_f32(outptr, _p);

                ptr += 4;
                ptr1 += 4;
                outptr += 4;
            }
#else
            if (nn > 0)
            {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #128]          \n"
                "pld        [%2, #128]          \n"
                "vld1.f32   {d0-d1}, [%1 :128]! \n"
                "vld1.f32   {d2-d3}, [%2 :128]! \n"
                "vmax.f32   q0, q0, q1          \n"
                "subs       %0, #1              \n"
                "vst1.f32   {d0-d1}, [%3 :128]! \n"
                "bne        0b                  \n"
                : "=r"(nn),     // %0
                  "=r"(ptr),    // %1
                  "=r"(ptr1),   // %2
                  "=r"(outptr)  // %3
                : "0"(nn),
                  "1"(ptr),
                  "2"(ptr1),
                  "3"(outptr)
                : "cc", "memory", "q0", "q1"
            );
            }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain>0; remain--)
            {
                *outptr = std::max(*ptr, *ptr1);

                ptr++;
                ptr1++;
                outptr++;
            }
        }

        for (size_t b=2; b<bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob1 = bottom_blobs[b];
            #pragma omp parallel for
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

#if __ARM_NEON
                int nn = size >> 2;
                int remain = size - (nn << 2);
#else
                int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
                for (; nn>0; nn--)
                {
                    float32x4_t _ptr = vld1q_f32(ptr);
                    float32x4_t _p = vld1q_f32(outptr);
                    _p = vmaxq_f32(_ptr, _p);
                    vst1q_f32(outptr, _p);

                    ptr += 4;
                    outptr += 4;
                }
#else
                if (nn > 0)
                {
                asm volatile(
                    "0:                             \n"
                    "pld        [%1, #128]          \n"
                    "pld        [%2, #128]          \n"
                    "vld1.f32   {d0-d1}, [%1 :128]! \n"
                    "vld1.f32   {d2-d3}, [%2 :128]  \n"
                    "vmax.f32   q0, q0, q1          \n"
                    "subs       %0, #1              \n"
                    "vst1.f32   {d0-d1}, [%2 :128]! \n"
                    "bne        0b                  \n"
                    : "=r"(nn),     // %0
                      "=r"(ptr),    // %1
                      "=r"(outptr)  // %2
                    : "0"(nn),
                      "1"(ptr),
                      "2"(outptr)
                    : "cc", "memory", "q0", "q1"
                );
                }
#endif // __aarch64__
#endif // __ARM_NEON
                for (; remain>0; remain--)
                {
                    *outptr = std::max(*ptr, *outptr);

                    ptr++;
                    outptr++;
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
