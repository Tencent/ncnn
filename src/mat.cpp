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

#include "mat.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "cpu.h"

namespace ncnn {

void Mat::substract_mean_normalize(const float* mean_vals, const float* norm_vals)
{
    int size = w * h;

    if (mean_vals && !norm_vals)
    {
        // substract mean only
        #pragma omp parallel for
        for (int q=0; q<c; q++)
        {
            float* ptr = data + cstep * q;
            const float mean = mean_vals[q];

#if __ARM_NEON
            int nn = size >> 2;
            int remain = size - (nn << 2);
#else
            int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
            float32x4_t _mean = vdupq_n_f32(mean);
            for (; nn>0; nn--)
            {
                float32x4_t _ptr = vld1q_f32(ptr);
                _ptr = vsubq_f32(_ptr, _mean);
                vst1q_f32(ptr, _ptr);
                ptr += 4;
            }
#else
            if (nn > 0)
            {
            asm volatile(
                "vdup.f32   q1, %4              \n"
                "0:                             \n"
                "pld        [%1, #128]          \n"
                "vld1.f32   {d0-d1}, [%1 :128]  \n"
                "vsub.f32   q0, q0, q1          \n"
                "subs       %0, #1              \n"
                "vst1.f32   {d0-d1}, [%1 :128]! \n"
                "bne        0b                  \n"
                : "=r"(nn),     // %0
                  "=r"(ptr)     // %1
                : "0"(nn),
                  "1"(ptr),
                  "r"(mean)     // %4
                : "cc", "memory", "q0", "q1"
            );
            }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain>0; remain--)
            {
                *ptr -= mean;
                ptr++;
            }
        }
    }
    else if (!mean_vals && norm_vals)
    {
        // normalize only
        #pragma omp parallel for
        for (int q=0; q<c; q++)
        {
            float* ptr = data + cstep * q;
            const float norm = norm_vals[q];

#if __ARM_NEON
            int nn = size >> 2;
            int remain = size - (nn << 2);
#else
            int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
            float32x4_t _norm = vdupq_n_f32(norm);
            for (; nn>0; nn--)
            {
                float32x4_t _ptr = vld1q_f32(ptr);
                _ptr = vmulq_f32(_ptr, _norm);
                vst1q_f32(ptr, _ptr);
                ptr += 4;
            }
#else
            if (nn > 0)
            {
            asm volatile(
                "vdup.f32   q1, %4              \n"
                "0:                             \n"
                "pld        [%1, #128]          \n"
                "vld1.f32   {d0-d1}, [%1 :128]  \n"
                "vmul.f32   q0, q0, q1          \n"
                "subs       %0, #1              \n"
                "vst1.f32   {d0-d1}, [%1 :128]! \n"
                "bne        0b                  \n"
                : "=r"(nn),     // %0
                  "=r"(ptr)     // %1
                : "0"(nn),
                  "1"(ptr),
                  "r"(norm)     // %4
                : "cc", "memory", "q0", "q1"
            );
            }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain>0; remain--)
            {
                *ptr *= norm;
                ptr++;
            }
        }
    }
    else if (mean_vals && norm_vals)
    {
        // substract mean and normalize
        #pragma omp parallel for
        for (int q=0; q<c; q++)
        {
            float* ptr = data + cstep * q;
            const float mean = mean_vals[q];
            const float norm = norm_vals[q];

#if __ARM_NEON
            int nn = size >> 2;
            int remain = size - (nn << 2);
#else
            int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
            float32x4_t _mean = vdupq_n_f32(mean);
            float32x4_t _norm = vdupq_n_f32(norm);
            for (; nn>0; nn--)
            {
                float32x4_t _ptr = vld1q_f32(ptr);
                _ptr = vsubq_f32(_ptr, _mean);
                _ptr = vmulq_f32(_ptr, _norm);
                vst1q_f32(ptr, _ptr);
                ptr += 4;
            }
#else
            if (nn > 0)
            {
            asm volatile(
                "vdup.f32   q1, %4              \n"
                "vdup.f32   q2, %5              \n"
                "0:                             \n"
                "pld        [%1, #128]          \n"
                "vld1.f32   {d0-d1}, [%1 :128]  \n"
                "vsub.f32   q0, q0, q1          \n"
                "vmul.f32   q0, q0, q2          \n"
                "subs       %0, #1              \n"
                "vst1.f32   {d0-d1}, [%1 :128]! \n"
                "bne        0b                  \n"
                : "=r"(nn),     // %0
                  "=r"(ptr)     // %1
                : "0"(nn),
                  "1"(ptr),
                  "r"(mean),    // %4
                  "r"(norm)     // %5
                : "cc", "memory", "q0", "q1", "q2"
            );
            }
#endif // __aarch64__
#endif // __ARM_NEON
            for (; remain>0; remain--)
            {
                *ptr = (*ptr - mean) * norm;
                ptr++;
            }
        }
    }
}

// convert half precision floating point to float
static float half2float(unsigned short value)
{
    // 1 : 5 : 10
    unsigned short sign = (value & 0x8000) >> 15;
    unsigned short exponent = (value & 0x7c00) >> 10;
    unsigned short significand = value & 0x03FF;

//     fprintf(stderr, "%d %d %d\n", sign, exponent, significand);

    // 1 : 8 : 23
    union
    {
        unsigned int u;
        float f;
    } tmp;
    if (exponent == 0)
    {
        if (significand == 0)
        {
            // zero
            tmp.u = (sign << 31);
        }
        else
        {
            // denormal
            exponent = 0;
            // find non-zero bit
            while ((significand & 0x200) == 0)
            {
                significand <<= 1;
                exponent++;
            }
            significand <<= 1;
            significand &= 0x3FF;
            tmp.u = (sign << 31) | ((-exponent + (-15 + 127)) << 23) | (significand << 13);
        }
    }
    else if (exponent == 0x1F)
    {
        // infinity or NaN
        tmp.u = (sign << 31) | (0xFF << 23) | (significand << 13);
    }
    else
    {
        // normalized
        tmp.u = (sign << 31) | ((exponent + (-15 + 127)) << 23) | (significand << 13);
    }

    return tmp.f;
}

Mat Mat::from_float16(const unsigned short* data, int size)
{
    Mat m(size);
    if (m.empty())
        return m;

    float* ptr = m.data;

#if __ARM_NEON && (__ARM_FP & 2)
    int nn = cpu_support_arm_vfpv4() ? size >> 2 : 0;
    int remain = size - (nn << 2);
#else
    int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON && (__ARM_FP & 2)
#if __aarch64__
    if (nn > 0)
    {
    asm volatile(
        "0:                             \n"
        "ldr    d0, [%1], #8            \n"
        "fcvtl  v1.4s, v0.4h            \n"
        "subs   %w0, %w0, #1            \n"
        "str    q1, [%2], #16           \n"
        "bne    0b                      \n"
        : "=r"(nn),     // %0
          "=r"(data),   // %1
          "=r"(ptr)     // %2
        : "0"(nn),
          "1"(data),
          "2"(ptr)
        : "cc", "memory", "v0", "v1"
    );
    }
#else
    if (nn > 0)
    {
    asm volatile(
        "0:                             \n"
        "pld        [%1, #64]           \n"
        "vld1.s16   {d0}, [%1 :64]!     \n"
        "vcvt.f32.f16 q1, d0            \n"
        "subs       %0, #1              \n"
        "vst1.f32   {d2-d3}, [%2 :128]! \n"
        "bne        0b                  \n"
        : "=r"(nn),     // %0
          "=r"(data),   // %1
          "=r"(ptr)     // %2
        : "0"(nn),
          "1"(data),
          "2"(ptr)
        : "cc", "memory", "q0", "q1"
    );
    }
#endif // __aarch64__
#endif // __ARM_NEON
    for (; remain>0; remain--)
    {
        *ptr = half2float(*data);

        data++;
        ptr++;
    }

    return m;
}

static void copy_make_border_image(const Mat& src, Mat& dst, int top, int left, int type, float v)
{
    int w = dst.w;
    int h = dst.h;

    const float* ptr = src.data;
    float* outptr = dst.data;

    if (type == BORDER_CONSTANT)
    {
        int y = 0;
        // fill top
        for (; y < top; y++)
        {
            int x = 0;
            for (; x < w; x++)
            {
                outptr[x] = v;
            }
            outptr += w;
        }
        // fill center
        for (; y < (top + src.h); y++)
        {
            int x = 0;
            for (; x < left; x++)
            {
                outptr[x] = v;
            }
            for (; x < (left + src.w); x++)
            {
                outptr[x] = ptr[x - left];
            }
            for (; x < w; x++)
            {
                outptr[x] = v;
            }
            ptr += src.w;
            outptr += w;
        }
        // fill bottom
        for (; y < h; y++)
        {
            int x = 0;
            for (; x < w; x++)
            {
                outptr[x] = v;
            }
            outptr += w;
        }
    }
    else if (type == BORDER_REPLICATE)
    {
        int y = 0;
        // fill top
        for (; y < top; y++)
        {
            int x = 0;
            for (; x < left; x++)
            {
                outptr[x] = ptr[0];
            }
            for (; x < (left + src.w); x++)
            {
                outptr[x] = ptr[x - left];
            }
            for (; x < w; x++)
            {
                outptr[x] = ptr[src.w - 1];
            }
            outptr += w;
        }
        // fill center
        for (; y < (top + src.h); y++)
        {
            int x = 0;
            for (; x < left; x++)
            {
                outptr[x] = ptr[0];
            }
            for (; x < (left + src.w); x++)
            {
                outptr[x] = ptr[x - left];
            }
            for (; x < w; x++)
            {
                outptr[x] = ptr[src.w - 1];
            }
            ptr += src.w;
            outptr += w;
        }
        // fill bottom
        ptr -= src.w;
        for (; y < h; y++)
        {
            int x = 0;
            for (; x < left; x++)
            {
                outptr[x] = ptr[0];
            }
            for (; x < (left + src.w); x++)
            {
                outptr[x] = ptr[x - left];
            }
            for (; x < w; x++)
            {
                outptr[x] = ptr[src.w - 1];
            }
            outptr += w;
        }
    }
}

void copy_make_border(const Mat& src, Mat& dst, int top, int bottom, int left, int right, int type, float v)
{
    int w = src.w + left + right;
    int h = src.h + top + bottom;

    if (src.dims == 2)
    {
        dst.create(w, h);
        if (dst.empty())
            return;

        copy_make_border_image(src, dst, top, left, type, v);
    }
    else if (src.dims == 3)
    {
        int channels = src.c;

        dst.create(w, h, channels);
        if (dst.empty())
            return;

        // unroll image channel
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const Mat m = src.channel(q);
            Mat borderm = dst.channel(q);

            copy_make_border_image(m, borderm, top, left, type, v);
        }
    }
}

static void copy_cut_border_image(const Mat& src, Mat& dst, int top, int left)
{
    int w = dst.w;
    int h = dst.h;

    const float* ptr = src.data + src.w * top + left;
    float* outptr = dst.data;

    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            outptr[x] = ptr[x];
        }
        outptr += w;
        ptr += src.w;
    }
}

void copy_cut_border(const Mat& src, Mat& dst, int top, int bottom, int left, int right)
{
    int w = src.w - left - right;
    int h = src.h - top - bottom;

    if (src.dims == 2)
    {
        dst.create(w, h);
        if (dst.empty())
            return;

        copy_cut_border_image(src, dst, top, left);
    }
    else if (src.dims == 3)
    {
        int channels = src.c;

        dst.create(w, h, channels);
        if (dst.empty())
            return;

        // unroll image channel
        #pragma omp parallel for
        for (int q=0; q<channels; q++)
        {
            const Mat m = src.channel(q);
            Mat cutm = dst.channel(q);

            copy_cut_border_image(m, cutm, top, left);
        }
    }
}

} // namespace ncnn
