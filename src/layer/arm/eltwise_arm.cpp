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

Eltwise_arm::Eltwise_arm()
{
#if __ARM_NEON
    support_packing = true;
#endif // __ARM_NEON

    support_bf16_storage = true;
}

int Eltwise_arm::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    if (opt.use_bf16_storage)
        return forward_bf16s(bottom_blobs, top_blobs, opt);

    const Mat& bottom_blob = bottom_blobs[0];
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;
    int size = w * h;

    Mat& top_blob = top_blobs[0];
    top_blob.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

#if __ARM_NEON
    if (elempack == 4)
    {
        if (op_type == Operation_PROD)
        {
            // first blob
            const Mat& bottom_blob1 = bottom_blobs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                const float* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i=0; i<size; i++)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    float32x4_t _p1 = vld1q_f32(ptr1);
                    _p = vmulq_f32(_p, _p1);
                    vst1q_f32(outptr, _p);

                    ptr += 4;
                    ptr1 += 4;
                    outptr += 4;
                }
            }

            for (size_t b=2; b<bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q=0; q<channels; q++)
                {
                    const float* ptr = bottom_blob1.channel(q);
                    float* outptr = top_blob.channel(q);

                    for (int i=0; i<size; i++)
                    {
                        float32x4_t _p = vld1q_f32(outptr);
                        float32x4_t _p1 = vld1q_f32(ptr);
                        _p = vmulq_f32(_p, _p1);
                        vst1q_f32(outptr, _p);

                        ptr += 4;
                        outptr += 4;
                    }
                }
            }
        }
        if (op_type == Operation_SUM)
        {
            if (coeffs.w == 0)
            {
                // first blob
                const Mat& bottom_blob1 = bottom_blobs[1];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q=0; q<channels; q++)
                {
                    const float* ptr = bottom_blob.channel(q);
                    const float* ptr1 = bottom_blob1.channel(q);
                    float* outptr = top_blob.channel(q);

                    for (int i=0; i<size; i++)
                    {
                        float32x4_t _p = vld1q_f32(ptr);
                        float32x4_t _p1 = vld1q_f32(ptr1);
                        _p = vaddq_f32(_p, _p1);
                        vst1q_f32(outptr, _p);

                        ptr += 4;
                        ptr1 += 4;
                        outptr += 4;
                    }
                }

                for (size_t b=2; b<bottom_blobs.size(); b++)
                {
                    const Mat& bottom_blob1 = bottom_blobs[b];
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q=0; q<channels; q++)
                    {
                        const float* ptr = bottom_blob1.channel(q);
                        float* outptr = top_blob.channel(q);

                        for (int i=0; i<size; i++)
                        {
                            float32x4_t _p = vld1q_f32(outptr);
                            float32x4_t _p1 = vld1q_f32(ptr);
                            _p = vaddq_f32(_p, _p1);
                            vst1q_f32(outptr, _p);

                            ptr += 4;
                            outptr += 4;
                        }
                    }
                }
            }
            else
            {
                // first blob
                const Mat& bottom_blob1 = bottom_blobs[1];
                float32x4_t _coeff0 = vdupq_n_f32(coeffs[0]);
                float32x4_t _coeff1 = vdupq_n_f32(coeffs[1]);
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q=0; q<channels; q++)
                {
                    const float* ptr = bottom_blob.channel(q);
                    const float* ptr1 = bottom_blob1.channel(q);
                    float* outptr = top_blob.channel(q);

                    for (int i=0; i<size; i++)
                    {
                        float32x4_t _p = vld1q_f32(ptr);
                        float32x4_t _p1 = vld1q_f32(ptr1);
                        _p = vmulq_f32(_p, _coeff0);
                        _p = vmlaq_f32(_p, _p1, _coeff1);
                        vst1q_f32(outptr, _p);

                        ptr += 4;
                        ptr1 += 4;
                        outptr += 4;
                    }
                }

                for (size_t b=2; b<bottom_blobs.size(); b++)
                {
                    const Mat& bottom_blob1 = bottom_blobs[b];
                    float32x4_t _coeff = vdupq_n_f32(coeffs[b]);
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q=0; q<channels; q++)
                    {
                        const float* ptr = bottom_blob1.channel(q);
                        float* outptr = top_blob.channel(q);

                        for (int i=0; i<size; i++)
                        {
                            float32x4_t _p = vld1q_f32(outptr);
                            float32x4_t _p1 = vld1q_f32(ptr);
                            _p = vmlaq_f32(_p, _p1, _coeff);
                            vst1q_f32(outptr, _p);

                            ptr += 4;
                            outptr += 4;
                        }
                    }
                }
            }
        }
        if (op_type == Operation_MAX)
        {
            // first blob
            const Mat& bottom_blob1 = bottom_blobs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                const float* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i=0; i<size; i++)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    float32x4_t _p1 = vld1q_f32(ptr1);
                    _p = vmaxq_f32(_p, _p1);
                    vst1q_f32(outptr, _p);

                    ptr += 4;
                    ptr1 += 4;
                    outptr += 4;
                }
            }

            for (size_t b=2; b<bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q=0; q<channels; q++)
                {
                    const float* ptr = bottom_blob1.channel(q);
                    float* outptr = top_blob.channel(q);

                    for (int i=0; i<size; i++)
                    {
                        float32x4_t _p = vld1q_f32(outptr);
                        float32x4_t _p1 = vld1q_f32(ptr);
                        _p = vmaxq_f32(_p, _p1);
                        vst1q_f32(outptr, _p);

                        ptr += 4;
                        outptr += 4;
                    }
                }
            }
        }

        return 0;
    }
#endif // __ARM_NEON

    if (op_type == Operation_PROD)
    {
        // first blob
        const Mat& bottom_blob1 = bottom_blobs[1];
        #pragma omp parallel for num_threads(opt.num_threads)
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
            if (nn > 0)
            {
            asm volatile(
                "0:                               \n"
                "prfm       pldl1keep, [%1, #128] \n"
                "prfm       pldl1keep, [%2, #128] \n"
                "ld1        {v0.4s}, [%1], #16    \n"
                "ld1        {v1.4s}, [%2], #16    \n"
                "fmul       v0.4s, v0.4s, v1.4s   \n"
                "subs       %w0, %w0, #1          \n"
                "st1        {v0.4s}, [%3], #16    \n"
                "bne        0b                    \n"
                : "=r"(nn),     // %0
                  "=r"(ptr),    // %1
                  "=r"(ptr1),   // %2
                  "=r"(outptr)  // %3
                : "0"(nn),
                  "1"(ptr),
                  "2"(ptr1),
                  "3"(outptr)
                : "cc", "memory", "v0", "v1"
            );
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
            #pragma omp parallel for num_threads(opt.num_threads)
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
                if (nn > 0)
                {
                asm volatile(
                    "0:                               \n"
                    "prfm       pldl1keep, [%1, #128] \n"
                    "prfm       pldl1keep, [%2, #128] \n"
                    "ld1        {v0.4s}, [%1], #16    \n"
                    "ld1        {v1.4s}, [%2]         \n"
                    "fmul       v0.4s, v0.4s, v1.4s   \n"
                    "subs       %w0, %w0, #1          \n"
                    "st1        {v0.4s}, [%2], #16    \n"
                    "bne        0b                    \n"
                    : "=r"(nn),     // %0
                      "=r"(ptr),    // %1
                      "=r"(outptr)  // %2
                    : "0"(nn),
                      "1"(ptr),
                      "2"(outptr)
                    : "cc", "memory", "v0", "v1"
                );
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
    if (op_type == Operation_SUM)
    {
        if (coeffs.w == 0)
        {
            // first blob
            const Mat& bottom_blob1 = bottom_blobs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
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
                if (nn > 0)
                {
                asm volatile(
                    "0:                               \n"
                    "prfm       pldl1keep, [%1, #128] \n"
                    "prfm       pldl1keep, [%2, #128] \n"
                    "ld1        {v0.4s}, [%1], #16    \n"
                    "ld1        {v1.4s}, [%2], #16    \n"
                    "fadd       v0.4s, v0.4s, v1.4s   \n"
                    "subs       %w0, %w0, #1          \n"
                    "st1        {v0.4s}, [%3], #16    \n"
                    "bne        0b                  \n"
                    : "=r"(nn),     // %0
                      "=r"(ptr),    // %1
                      "=r"(ptr1),   // %2
                      "=r"(outptr)  // %3
                    : "0"(nn),
                      "1"(ptr),
                      "2"(ptr1),
                      "3"(outptr)
                    : "cc", "memory", "v0", "v1"
                );
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
                #pragma omp parallel for num_threads(opt.num_threads)
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
                    if (nn > 0)
                    {
                    asm volatile(
                        "0:                               \n"
                        "prfm       pldl1keep, [%1, #128] \n"
                        "prfm       pldl1keep, [%2, #128] \n"
                        "ld1        {v0.4s}, [%1], #16    \n"
                        "ld1        {v1.4s}, [%2]         \n"
                        "fadd       v0.4s, v0.4s, v1.4s   \n"
                        "subs       %w0, %w0, #1          \n"
                        "st1        {v0.4s}, [%2], #16    \n"
                        "bne        0b                    \n"
                        : "=r"(nn),     // %0
                          "=r"(ptr),    // %1
                          "=r"(outptr)  // %2
                        : "0"(nn),
                          "1"(ptr),
                          "2"(outptr)
                        : "cc", "memory", "v0", "v1"
                    );
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
            // first blob
            const Mat& bottom_blob1 = bottom_blobs[1];
            float coeff0 = coeffs[0];
            float coeff1 = coeffs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
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
                if (nn > 0)
                {
                asm volatile(
                    "0:                               \n"
                    "prfm       pldl1keep, [%1, #128] \n"
                    "prfm       pldl1keep, [%2, #128] \n"
                    "ld1        {v0.4s}, [%1], #16    \n"
                    "ld1        {v1.4s}, [%2], #16    \n"
                    "fmul       v0.4s, v0.4s, %8.4s   \n"
                    "fmla       v0.4s, v1.4s, %9.4s   \n"
                    "subs       %w0, %w0, #1          \n"
                    "st1        {v0.4s}, [%3], #16    \n"
                    "bne        0b                    \n"
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
                    : "cc", "memory", "v0", "v1"
                );
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
                float coeff = coeffs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
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
                    if (nn > 0)
                    {
                    asm volatile(
                        "0:                               \n"
                        "prfm       pldl1keep, [%1, #128] \n"
                        "prfm       pldl1keep, [%2, #128] \n"
                        "ld1        {v0.4s}, [%1], #16    \n"
                        "ld1        {v1.4s}, [%2]         \n"
                        "fmla       v1.4s, v0.4s, %6.4s   \n"
                        "subs       %w0, %w0, #1          \n"
                        "st1        {v1.4s}, [%2], #16    \n"
                        "bne        0b                    \n"
                        : "=r"(nn),     // %0
                          "=r"(ptr),    // %1
                          "=r"(outptr)  // %2
                        : "0"(nn),
                          "1"(ptr),
                          "2"(outptr),
                          "w"(_coeff)   // %6
                        : "cc", "memory", "v0", "v1"
                    );
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
                        "vst1.f32   {d2-d3}, [%2 :128]! \n"
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
    if (op_type == Operation_MAX)
    {
        // first blob
        const Mat& bottom_blob1 = bottom_blobs[1];
        #pragma omp parallel for num_threads(opt.num_threads)
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
            if (nn > 0)
            {
            asm volatile(
                "0:                               \n"
                "prfm       pldl1keep, [%1, #128] \n"
                "prfm       pldl1keep, [%2, #128] \n"
                "ld1        {v0.4s}, [%1], #16    \n"
                "ld1        {v1.4s}, [%2], #16    \n"
                "fmax       v0.4s, v0.4s, v1.4s   \n"
                "subs       %w0, %w0, #1          \n"
                "st1        {v0.4s}, [%3], #16    \n"
                "bne        0b                    \n"
                : "=r"(nn),     // %0
                  "=r"(ptr),    // %1
                  "=r"(ptr1),   // %2
                  "=r"(outptr)  // %3
                : "0"(nn),
                  "1"(ptr),
                  "2"(ptr1),
                  "3"(outptr)
                : "cc", "memory", "v0", "v1"
            );
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
            #pragma omp parallel for num_threads(opt.num_threads)
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
                if (nn > 0)
                {
                asm volatile(
                    "0:                               \n"
                    "prfm       pldl1keep, [%1, #128] \n"
                    "prfm       pldl1keep, [%2, #128] \n"
                    "ld1        {v0.4s}, [%1], #16    \n"
                    "ld1        {v1.4s}, [%2]         \n"
                    "fmax       v0.4s, v0.4s, v1.4s   \n"
                    "subs       %w0, %w0, #1          \n"
                    "st1        {v0.4s}, [%2], #16    \n"
                    "bne        0b                    \n"
                    : "=r"(nn),     // %0
                      "=r"(ptr),    // %1
                      "=r"(outptr)  // %2
                    : "0"(nn),
                      "1"(ptr),
                      "2"(outptr)
                    : "cc", "memory", "v0", "v1"
                );
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

int Eltwise_arm::forward_bf16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;
    int size = w * h;

    Mat& top_blob = top_blobs[0];
    top_blob.create(w, h, channels, elemsize, elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (bottom_blobs.size() == 2)
    {
        // fast path without fp32 accumulator
#if __ARM_NEON
        if (elempack == 4)
        {
            if (op_type == Operation_PROD)
            {
                const Mat& bottom_blob1 = bottom_blobs[1];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q=0; q<channels; q++)
                {
                    const unsigned short* ptr = bottom_blob.channel(q);
                    const unsigned short* ptr1 = bottom_blob1.channel(q);
                    unsigned short* outptr = top_blob.channel(q);

                    for (int i=0; i<size; i++)
                    {
                        float32x4_t _p = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(ptr), 16));
                        float32x4_t _p1 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(ptr1), 16));
                        _p = vmulq_f32(_p, _p1);
                        vst1_u16(outptr, vshrn_n_u32(vreinterpretq_u32_f32(_p), 16));

                        ptr += 4;
                        ptr1 += 4;
                        outptr += 4;
                    }
                }
            }
            if (op_type == Operation_SUM)
            {
                if (coeffs.w == 0)
                {
                    const Mat& bottom_blob1 = bottom_blobs[1];
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q=0; q<channels; q++)
                    {
                        const unsigned short* ptr = bottom_blob.channel(q);
                        const unsigned short* ptr1 = bottom_blob1.channel(q);
                        unsigned short* outptr = top_blob.channel(q);

                        for (int i=0; i<size; i++)
                        {
                            float32x4_t _p = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(ptr), 16));
                            float32x4_t _p1 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(ptr1), 16));
                            _p = vaddq_f32(_p, _p1);
                            vst1_u16(outptr, vshrn_n_u32(vreinterpretq_u32_f32(_p), 16));

                            ptr += 4;
                            ptr1 += 4;
                            outptr += 4;
                        }
                    }
                }
                else
                {
                    const Mat& bottom_blob1 = bottom_blobs[1];
                    float32x4_t _coeff0 = vdupq_n_f32(coeffs[0]);
                    float32x4_t _coeff1 = vdupq_n_f32(coeffs[1]);
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q=0; q<channels; q++)
                    {
                        const unsigned short* ptr = bottom_blob.channel(q);
                        const unsigned short* ptr1 = bottom_blob1.channel(q);
                        unsigned short* outptr = top_blob.channel(q);

                        for (int i=0; i<size; i++)
                        {
                            float32x4_t _p = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(ptr), 16));
                            float32x4_t _p1 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(ptr1), 16));
                            _p = vmulq_f32(_p, _coeff0);
                            _p = vmlaq_f32(_p, _p1, _coeff1);
                            vst1_u16(outptr, vshrn_n_u32(vreinterpretq_u32_f32(_p), 16));

                            ptr += 4;
                            ptr1 += 4;
                            outptr += 4;
                        }
                    }
                }
            }
            if (op_type == Operation_MAX)
            {
                const Mat& bottom_blob1 = bottom_blobs[1];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q=0; q<channels; q++)
                {
                    const unsigned short* ptr = bottom_blob.channel(q);
                    const unsigned short* ptr1 = bottom_blob1.channel(q);
                    unsigned short* outptr = top_blob.channel(q);

                    for (int i=0; i<size; i++)
                    {
                        float32x4_t _p = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(ptr), 16));
                        float32x4_t _p1 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(ptr1), 16));
                        _p = vmaxq_f32(_p, _p1);
                        vst1_u16(outptr, vshrn_n_u32(vreinterpretq_u32_f32(_p), 16));

                        ptr += 4;
                        ptr1 += 4;
                        outptr += 4;
                    }
                }
            }

            return 0;
        }
#endif // __ARM_NEON

        if (op_type == Operation_PROD)
        {
            const Mat& bottom_blob1 = bottom_blobs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const unsigned short* ptr = bottom_blob.channel(q);
                const unsigned short* ptr1 = bottom_blob1.channel(q);
                unsigned short* outptr = top_blob.channel(q);

                for (int i=0; i<size; i++)
                {
                    *outptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) * bfloat16_to_float32(*ptr1));

                    ptr++;
                    ptr1++;
                    outptr++;
                }
            }
        }
        if (op_type == Operation_SUM)
        {
            if (coeffs.w == 0)
            {
                const Mat& bottom_blob1 = bottom_blobs[1];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q=0; q<channels; q++)
                {
                    const unsigned short* ptr = bottom_blob.channel(q);
                    const unsigned short* ptr1 = bottom_blob1.channel(q);
                    unsigned short* outptr = top_blob.channel(q);

                    for (int i=0; i<size; i++)
                    {
                        *outptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) + bfloat16_to_float32(*ptr1));

                        ptr++;
                        ptr1++;
                        outptr++;
                    }
                }
            }
            else
            {
                const Mat& bottom_blob1 = bottom_blobs[1];
                float coeff0 = coeffs[0];
                float coeff1 = coeffs[1];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q=0; q<channels; q++)
                {
                    const unsigned short* ptr = bottom_blob.channel(q);
                    const unsigned short* ptr1 = bottom_blob1.channel(q);
                    unsigned short* outptr = top_blob.channel(q);

                    for (int i=0; i<size; i++)
                    {
                        *outptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) * coeff0 + bfloat16_to_float32(*ptr1) * coeff1);

                        ptr++;
                        ptr1++;
                        outptr++;
                    }
                }
            }
        }
        if (op_type == Operation_MAX)
        {
            // first blob
            const Mat& bottom_blob1 = bottom_blobs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const unsigned short* ptr = bottom_blob.channel(q);
                const unsigned short* ptr1 = bottom_blob1.channel(q);
                unsigned short* outptr = top_blob.channel(q);

                for (int i=0; i<size; i++)
                {
                    *outptr = float32_to_bfloat16(std::max(bfloat16_to_float32(*ptr), bfloat16_to_float32(*ptr1)));

                    ptr++;
                    ptr1++;
                    outptr++;
                }
            }
        }

        return 0;
    }

    Mat top_blob_fp32(w, h, channels, (size_t)4u * elempack, elempack, opt.workspace_allocator);
    if (top_blob_fp32.empty())
        return -100;

#if __ARM_NEON
    if (elempack == 4)
    {
        if (op_type == Operation_PROD)
        {
            // first blob
            const Mat& bottom_blob1 = bottom_blobs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const unsigned short* ptr = bottom_blob.channel(q);
                const unsigned short* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob_fp32.channel(q);

                for (int i=0; i<size; i++)
                {
                    float32x4_t _p = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(ptr), 16));
                    float32x4_t _p1 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(ptr1), 16));
                    _p = vmulq_f32(_p, _p1);
                    vst1q_f32(outptr, _p);

                    ptr += 4;
                    ptr1 += 4;
                    outptr += 4;
                }
            }

            size_t b=2;
            for (; b<bottom_blobs.size()-1; b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q=0; q<channels; q++)
                {
                    const unsigned short* ptr = bottom_blob1.channel(q);
                    float* outptr = top_blob_fp32.channel(q);

                    for (int i=0; i<size; i++)
                    {
                        float32x4_t _p = vld1q_f32(outptr);
                        float32x4_t _p1 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(ptr), 16));
                        _p = vmulq_f32(_p, _p1);
                        vst1q_f32(outptr, _p);

                        ptr += 4;
                        outptr += 4;
                    }
                }
            }
            for (; b<bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q=0; q<channels; q++)
                {
                    const unsigned short* ptr = bottom_blob1.channel(q);
                    const float* ptr0 = top_blob_fp32.channel(q);
                    unsigned short* outptr = top_blob.channel(q);

                    for (int i=0; i<size; i++)
                    {
                        float32x4_t _p = vld1q_f32(ptr0);
                        float32x4_t _p1 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(ptr), 16));
                        _p = vmulq_f32(_p, _p1);
                        vst1_u16(outptr, vshrn_n_u32(vreinterpretq_u32_f32(_p), 16));

                        ptr += 4;
                        ptr0 += 4;
                        outptr += 4;
                    }
                }
            }
        }
        if (op_type == Operation_SUM)
        {
            if (coeffs.w == 0)
            {
                // first blob
                const Mat& bottom_blob1 = bottom_blobs[1];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q=0; q<channels; q++)
                {
                    const unsigned short* ptr = bottom_blob.channel(q);
                    const unsigned short* ptr1 = bottom_blob1.channel(q);
                    float* outptr = top_blob_fp32.channel(q);

                    for (int i=0; i<size; i++)
                    {
                        float32x4_t _p = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(ptr), 16));
                        float32x4_t _p1 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(ptr1), 16));
                        _p = vaddq_f32(_p, _p1);
                        vst1q_f32(outptr, _p);

                        ptr += 4;
                        ptr1 += 4;
                        outptr += 4;
                    }
                }

                size_t b=2;
                for (; b<bottom_blobs.size()-1; b++)
                {
                    const Mat& bottom_blob1 = bottom_blobs[b];
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q=0; q<channels; q++)
                    {
                        const unsigned short* ptr = bottom_blob1.channel(q);
                        float* outptr = top_blob_fp32.channel(q);

                        for (int i=0; i<size; i++)
                        {
                            float32x4_t _p = vld1q_f32(outptr);
                            float32x4_t _p1 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(ptr), 16));
                            _p = vaddq_f32(_p, _p1);
                            vst1q_f32(outptr, _p);

                            ptr += 4;
                            outptr += 4;
                        }
                    }
                }
                for (; b<bottom_blobs.size(); b++)
                {
                    const Mat& bottom_blob1 = bottom_blobs[b];
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q=0; q<channels; q++)
                    {
                        const unsigned short* ptr = bottom_blob1.channel(q);
                        const float* ptr0 = top_blob_fp32.channel(q);
                        unsigned short* outptr = top_blob.channel(q);

                        for (int i=0; i<size; i++)
                        {
                            float32x4_t _p = vld1q_f32(ptr0);
                            float32x4_t _p1 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(ptr), 16));
                            _p = vaddq_f32(_p, _p1);
                            vst1_u16(outptr, vshrn_n_u32(vreinterpretq_u32_f32(_p), 16));

                            ptr += 4;
                            ptr0 += 4;
                            outptr += 4;
                        }
                    }
                }
            }
            else
            {
                // first blob
                const Mat& bottom_blob1 = bottom_blobs[1];
                float32x4_t _coeff0 = vdupq_n_f32(coeffs[0]);
                float32x4_t _coeff1 = vdupq_n_f32(coeffs[1]);
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q=0; q<channels; q++)
                {
                    const unsigned short* ptr = bottom_blob.channel(q);
                    const unsigned short* ptr1 = bottom_blob1.channel(q);
                    float* outptr = top_blob_fp32.channel(q);

                    for (int i=0; i<size; i++)
                    {
                        float32x4_t _p = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(ptr), 16));
                        float32x4_t _p1 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(ptr1), 16));
                        _p = vmulq_f32(_p, _coeff0);
                        _p = vmlaq_f32(_p, _p1, _coeff1);
                        vst1q_f32(outptr, _p);

                        ptr += 4;
                        ptr1 += 4;
                        outptr += 4;
                    }
                }

                size_t b=2;
                for (; b<bottom_blobs.size()-1; b++)
                {
                    const Mat& bottom_blob1 = bottom_blobs[b];
                    float32x4_t _coeff = vdupq_n_f32(coeffs[b]);
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q=0; q<channels; q++)
                    {
                        const unsigned short* ptr = bottom_blob1.channel(q);
                        float* outptr = top_blob_fp32.channel(q);

                        for (int i=0; i<size; i++)
                        {
                            float32x4_t _p = vld1q_f32(outptr);
                            float32x4_t _p1 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(ptr), 16));
                            _p = vmlaq_f32(_p, _p1, _coeff);
                            vst1q_f32(outptr, _p);

                            ptr += 4;
                            outptr += 4;
                        }
                    }
                }
                for (; b<bottom_blobs.size(); b++)
                {
                    const Mat& bottom_blob1 = bottom_blobs[b];
                    float32x4_t _coeff = vdupq_n_f32(coeffs[b]);
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int q=0; q<channels; q++)
                    {
                        const unsigned short* ptr = bottom_blob1.channel(q);
                        const float* ptr0 = top_blob_fp32.channel(q);
                        unsigned short* outptr = top_blob.channel(q);

                        for (int i=0; i<size; i++)
                        {
                            float32x4_t _p = vld1q_f32(ptr0);
                            float32x4_t _p1 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(ptr), 16));
                            _p = vmlaq_f32(_p, _p1, _coeff);
                            vst1_u16(outptr, vshrn_n_u32(vreinterpretq_u32_f32(_p), 16));

                            ptr += 4;
                            ptr0 += 4;
                            outptr += 4;
                        }
                    }
                }
            }
        }
        if (op_type == Operation_MAX)
        {
            // first blob
            const Mat& bottom_blob1 = bottom_blobs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const unsigned short* ptr = bottom_blob.channel(q);
                const unsigned short* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob_fp32.channel(q);

                for (int i=0; i<size; i++)
                {
                    float32x4_t _p = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(ptr), 16));
                    float32x4_t _p1 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(ptr1), 16));
                    _p = vmaxq_f32(_p, _p1);
                    vst1q_f32(outptr, _p);

                    ptr += 4;
                    ptr1 += 4;
                    outptr += 4;
                }
            }

            size_t b=2;
            for (; b<bottom_blobs.size()-1; b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q=0; q<channels; q++)
                {
                    const unsigned short* ptr = bottom_blob1.channel(q);
                    float* outptr = top_blob_fp32.channel(q);

                    for (int i=0; i<size; i++)
                    {
                        float32x4_t _p = vld1q_f32(outptr);
                        float32x4_t _p1 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(ptr), 16));
                        _p = vmaxq_f32(_p, _p1);
                        vst1q_f32(outptr, _p);

                        ptr += 4;
                        outptr += 4;
                    }
                }
            }
            for (; b<bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q=0; q<channels; q++)
                {
                    const unsigned short* ptr = bottom_blob1.channel(q);
                    const float* ptr0 = top_blob_fp32.channel(q);
                    unsigned short* outptr = top_blob.channel(q);

                    for (int i=0; i<size; i++)
                    {
                        float32x4_t _p = vld1q_f32(ptr0);
                        float32x4_t _p1 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(ptr), 16));
                        _p = vmaxq_f32(_p, _p1);
                        vst1_u16(outptr, vshrn_n_u32(vreinterpretq_u32_f32(_p), 16));

                        ptr += 4;
                        ptr0 += 4;
                        outptr += 4;
                    }
                }
            }
        }

        return 0;
    }
#endif // __ARM_NEON

    if (op_type == Operation_PROD)
    {
        // first blob
        const Mat& bottom_blob1 = bottom_blobs[1];
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            const unsigned short* ptr = bottom_blob.channel(q);
            const unsigned short* ptr1 = bottom_blob1.channel(q);
            float* outptr = top_blob_fp32.channel(q);

            for (int i=0; i<size; i++)
            {
                *outptr = bfloat16_to_float32(*ptr) * bfloat16_to_float32(*ptr1);

                ptr++;
                ptr1++;
                outptr++;
            }
        }

        size_t b=2;
        for (; b<bottom_blobs.size()-1; b++)
        {
            const Mat& bottom_blob1 = bottom_blobs[b];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const unsigned short* ptr = bottom_blob1.channel(q);
                float* outptr = top_blob_fp32.channel(q);

                for (int i=0; i<size; i++)
                {
                    *outptr *= bfloat16_to_float32(*ptr);

                    ptr++;
                    outptr++;
                }
            }
        }
        for (; b<bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob1 = bottom_blobs[b];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const unsigned short* ptr = bottom_blob1.channel(q);
                const float* ptr0 = top_blob_fp32.channel(q);
                unsigned short* outptr = top_blob.channel(q);

                for (int i=0; i<size; i++)
                {
                    *outptr = float32_to_bfloat16(*ptr0 * bfloat16_to_float32(*ptr));

                    ptr++;
                    ptr0++;
                    outptr++;
                }
            }
        }
    }
    if (op_type == Operation_SUM)
    {
        if (coeffs.w == 0)
        {
            // first blob
            const Mat& bottom_blob1 = bottom_blobs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const unsigned short* ptr = bottom_blob.channel(q);
                const unsigned short* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob_fp32.channel(q);

                for (int i=0; i<size; i++)
                {
                    *outptr = bfloat16_to_float32(*ptr) + bfloat16_to_float32(*ptr1);

                    ptr++;
                    ptr1++;
                    outptr++;
                }
            }

            size_t b=2;
            for (; b<bottom_blobs.size()-1; b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q=0; q<channels; q++)
                {
                    const unsigned short* ptr = bottom_blob1.channel(q);
                    float* outptr = top_blob_fp32.channel(q);

                    for (int i=0; i<size; i++)
                    {
                        *outptr += bfloat16_to_float32(*ptr);

                        ptr++;
                        outptr++;
                    }
                }
            }
            for (; b<bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q=0; q<channels; q++)
                {
                    const unsigned short* ptr = bottom_blob1.channel(q);
                    const float* ptr0 = top_blob_fp32.channel(q);
                    unsigned short* outptr = top_blob.channel(q);

                    for (int i=0; i<size; i++)
                    {
                        *outptr = float32_to_bfloat16(*ptr0 + bfloat16_to_float32(*ptr));

                        ptr++;
                        ptr0++;
                        outptr++;
                    }
                }
            }
        }
        else
        {
            // first blob
            const Mat& bottom_blob1 = bottom_blobs[1];
            float coeff0 = coeffs[0];
            float coeff1 = coeffs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const unsigned short* ptr = bottom_blob.channel(q);
                const unsigned short* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob_fp32.channel(q);

                for (int i=0; i<size; i++)
                {
                    *outptr = bfloat16_to_float32(*ptr) * coeff0 + bfloat16_to_float32(*ptr1) * coeff1;

                    ptr++;
                    ptr1++;
                    outptr++;
                }
            }

            size_t b=2;
            for (; b<bottom_blobs.size()-1; b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
                float coeff = coeffs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q=0; q<channels; q++)
                {
                    const unsigned short* ptr = bottom_blob1.channel(q);
                    float* outptr = top_blob_fp32.channel(q);

                    for (int i=0; i<size; i++)
                    {
                        *outptr += bfloat16_to_float32(*ptr) * coeff;

                        ptr++;
                        outptr++;
                    }
                }
            }
            for (; b<bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
                float coeff = coeffs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q=0; q<channels; q++)
                {
                    const unsigned short* ptr = bottom_blob1.channel(q);
                    const float* ptr0 = top_blob_fp32.channel(q);
                    unsigned short* outptr = top_blob.channel(q);

                    for (int i=0; i<size; i++)
                    {
                        *outptr = float32_to_bfloat16(*ptr0 + bfloat16_to_float32(*ptr) * coeff);

                        ptr++;
                        ptr0++;
                        outptr++;
                    }
                }
            }
        }
    }
    if (op_type == Operation_MAX)
    {
        // first blob
        const Mat& bottom_blob1 = bottom_blobs[1];
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            const unsigned short* ptr = bottom_blob.channel(q);
            const unsigned short* ptr1 = bottom_blob1.channel(q);
            float* outptr = top_blob_fp32.channel(q);

            for (int i=0; i<size; i++)
            {
                *outptr = std::max(bfloat16_to_float32(*ptr), bfloat16_to_float32(*ptr1));

                ptr++;
                ptr1++;
                outptr++;
            }
        }

        size_t b=2;
        for (; b<bottom_blobs.size()-1; b++)
        {
            const Mat& bottom_blob1 = bottom_blobs[b];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const unsigned short* ptr = bottom_blob1.channel(q);
                float* outptr = top_blob_fp32.channel(q);

                for (int i=0; i<size; i++)
                {
                    *outptr = std::max(bfloat16_to_float32(*ptr), *outptr);

                    ptr++;
                    outptr++;
                }
            }
        }
        for (; b<bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob1 = bottom_blobs[b];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const unsigned short* ptr = bottom_blob1.channel(q);
                const float* ptr0 = top_blob_fp32.channel(q);
                unsigned short* outptr = top_blob.channel(q);

                for (int i=0; i<size; i++)
                {
                    *outptr = float32_to_bfloat16(std::max(bfloat16_to_float32(*ptr), *ptr0));

                    ptr++;
                    ptr0++;
                    outptr++;
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
