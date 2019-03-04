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
#include <math.h>

#include "cpu.h"

#include "layer_type.h"
#include "layer.h"

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
            float* ptr = channel(q);//data + cstep * q;
            const float mean = mean_vals[q];

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
                "dup        v1.4s, %w4            \n"
                "0:                               \n"
                "prfm       pldl1keep, [%1, #128] \n"
                "ld1        {v0.4s}, [%1]         \n"
                "fsub       v0.4s, v0.4s, v1.4s   \n"
                "subs       %w0, %w0, #1          \n"
                "st1        {v0.4s}, [%1], #16    \n"
                "bne        0b                    \n"
                : "=r"(nn),     // %0
                  "=r"(ptr)     // %1
                : "0"(nn),
                  "1"(ptr),
                  "r"(mean)     // %4
                : "cc", "memory", "v0", "v1"
            );
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
            float* ptr = channel(q);//data + cstep * q;
            const float norm = norm_vals[q];

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
                "dup        v1.4s, %w4            \n"
                "0:                               \n"
                "prfm       pldl1keep, [%1, #128] \n"
                "ld1        {v0.4s}, [%1]         \n"
                "fmul       v0.4s, v0.4s, v1.4s   \n"
                "subs       %w0, %w0, #1          \n"
                "st1        {v0.4s}, [%1], #16    \n"
                "bne        0b                    \n"
                : "=r"(nn),     // %0
                  "=r"(ptr)     // %1
                : "0"(nn),
                  "1"(ptr),
                  "r"(norm)     // %4
                : "cc", "memory", "v0", "v1"
            );
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
            float* ptr = channel(q);//data + cstep * q;
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
            if (nn > 0)
            {
            asm volatile(
                "dup        v1.4s, %w4            \n"
                "dup        v2.4s, %w5            \n"
                "0:                               \n"
                "prfm       pldl1keep, [%1, #128] \n"
                "ld1        {v0.4s}, [%1]         \n"
                "fsub       v0.4s, v0.4s, v1.4s   \n"
                "fmul       v0.4s, v0.4s, v2.4s   \n"
                "subs       %w0, %w0, #1          \n"
                "st1        {v0.4s}, [%1], #16    \n"
                "bne        0b                    \n"
                : "=r"(nn),     // %0
                  "=r"(ptr)     // %1
                : "0"(nn),
                  "1"(ptr),
                  "r"(mean),    // %4
                  "r"(norm)     // %5
                : "cc", "memory", "v0", "v1", "v2"
            );  
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

    float* ptr = m;//.data;

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
        "ld1    {v0.4h}, [%1], #8       \n"
        "fcvtl  v1.4s, v0.4h            \n"
        "subs   %w0, %w0, #1            \n"
        "st1    {v1.4s}, [%2], #16      \n"
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

void copy_make_border(const Mat& src, Mat& dst, int top, int bottom, int left, int right, int type, float v, Allocator* allocator, int num_threads)
{
    ncnn::Layer* padding = ncnn::create_layer(ncnn::LayerType::Padding);

    ncnn::ParamDict pd;
    pd.set(0, top);
    pd.set(1, bottom);
    pd.set(2, left);
    pd.set(3, right);
    pd.set(4, type);
    pd.set(5, v);

    padding->load_param(pd);

    ncnn::Option opt = ncnn::get_default_option();
    opt.num_threads = num_threads;
    opt.blob_allocator = allocator;

    padding->forward(src, dst, opt);

    delete padding;
}

static void copy_cut_border_image(const Mat& src, Mat& dst, int top, int left)
{
    int w = dst.w;
    int h = dst.h;

    const float* ptr = src.row(top) + left;//.data + src.w * top + left;
    float* outptr = dst;//.data;

    for (int y = 0; y < h; y++)
    {
        if(w < 12)
        {
            for (int x = 0; x < w; x++)
            {
                outptr[x] = ptr[x];
            }
        }
        else
        {
            memcpy(outptr, ptr, w*sizeof(float));
        }
        outptr += w;
        ptr += src.w;
    }
}

void copy_cut_border(const Mat& src, Mat& dst, int top, int bottom, int left, int right, Allocator* allocator, int num_threads)
{
    int w = src.w - left - right;
    int h = src.h - top - bottom;
    size_t elemsize = src.elemsize;

    if (w == src.w && h == src.h)
    {
        dst = src;
        return;
    }

    if (src.dims == 2)
    {
        dst.create(w, h, elemsize, allocator);
        if (dst.empty())
            return;

        copy_cut_border_image(src, dst, top, left);
    }
    else if (src.dims == 3)
    {
        int channels = src.c;

        dst.create(w, h, channels, elemsize, allocator);
        if (dst.empty())
            return;

        // unroll image channel
        #pragma omp parallel for num_threads(num_threads)
        for (int q=0; q<channels; q++)
        {
            const Mat m = src.channel(q);
            Mat cutm = dst.channel(q);

            copy_cut_border_image(m, cutm, top, left);
        }
    }
}

static void resize_bilinear_image(const Mat& src, Mat& dst, int w, int h)
{
    double scale_x = (double)src.w / w;
    double scale_y = (double)src.h / h;

    int* buf = new int[w + h + w*2 + h*2];

    int* xofs = buf;//new int[w];
    int* yofs = buf + w;//new int[h];

    float* alpha = (float*)(buf + w + h);//new float[w * 2];
    float* beta = (float*)(buf + w + h + w*2);//new float[h * 2];

    float fx;
    float fy;
    int sx;
    int sy;

    for (int dx = 0; dx < w; dx++)
    {
        fx = (float)((dx + 0.5) * scale_x - 0.5);
        sx = floor(fx);
        fx -= sx;

        if (sx < 0)
        {
            sx = 0;
            fx = 0.f;
        }
        if (sx >= src.w - 1)
        {
            sx = src.w - 2;
            fx = 1.f;
        }

        xofs[dx] = sx;

        alpha[dx*2    ] = 1.f - fx;
        alpha[dx*2 + 1] = fx;
    }

    for (int dy = 0; dy < h; dy++)
    {
        fy = (float)((dy + 0.5) * scale_y - 0.5);
        sy = floor(fy);
        fy -= sy;

        if (sy < 0)
        {
            sy = 0;
            fy = 0.f;
        }
        if (sy >= src.h - 1)
        {
            sy = src.h - 2;
            fy = 1.f;
        }

        yofs[dy] = sy;

        beta[dy*2    ] = 1.f - fy;
        beta[dy*2 + 1] = fy;
    }

    // loop body
    Mat rowsbuf0(w + 1);
    Mat rowsbuf1(w + 1);
    float* rows0 = rowsbuf0;
    float* rows1 = rowsbuf1;

    int prev_sy1 = -1;

    for (int dy = 0; dy < h; dy++ )
    {
        int sy = yofs[dy];

        if (sy == prev_sy1)
        {
            // hresize one row
            float* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows0_old;
            const float* S1 = src.row(sy+1);

            const float* alphap = alpha;
            float* rows1p = rows1;
            int dx = 0;
#if __ARM_NEON
            for ( ; dx+1 < w; dx += 2 )
            {
                int sx = xofs[dx];
                int sxn = xofs[dx+1];
                const float* S1p = S1 + sx;
                const float* S1np = S1 + sxn;

                float32x4_t _a = vld1q_f32(alphap);
                float32x2_t _S1 = vld1_f32(S1p);
                float32x2_t _S1n = vld1_f32(S1np);

                float32x4_t _S1S1n = vcombine_f32(_S1, _S1n);
                float32x4_t _ms1 = vmulq_f32(_S1S1n, _a);
                float32x2_t _rows1 = vpadd_f32(vget_low_f32(_ms1), vget_high_f32(_ms1));

                vst1_f32(rows1p + dx, _rows1);

                alphap += 4;
            }
#endif // __ARM_NEON
            for ( ; dx < w; dx++ )
            {
                int sx = xofs[dx];
                const float* S1p = S1 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                rows1p[dx] = S1p[0]*a0 + S1p[1]*a1;

                alphap += 2;
            }
        }
        else
        {
            // hresize two rows
            const float* S0 = src.row(sy);
            const float* S1 = src.row(sy+1);

            const float* alphap = alpha;
            float* rows0p = rows0;
            float* rows1p = rows1;
            int dx = 0;
#if __ARM_NEON
            for ( ; dx+1 < w; dx += 2 )
            {
                int sx = xofs[dx];
                int sxn = xofs[dx+1];
                const float* S0p = S0 + sx;
                const float* S1p = S1 + sx;
                const float* S0np = S0 + sxn;
                const float* S1np = S1 + sxn;

                float32x4_t _a = vld1q_f32(alphap);
                float32x2_t _S0 = vld1_f32(S0p);
                float32x2_t _S1 = vld1_f32(S1p);
                float32x2_t _S0n = vld1_f32(S0np);
                float32x2_t _S1n = vld1_f32(S1np);

                float32x4_t _S0S0n = vcombine_f32(_S0, _S0n);
                float32x4_t _S1S1n = vcombine_f32(_S1, _S1n);
                float32x4_t _ms0 = vmulq_f32(_S0S0n, _a);
                float32x4_t _ms1 = vmulq_f32(_S1S1n, _a);
                float32x2_t _rows0 = vpadd_f32(vget_low_f32(_ms0), vget_high_f32(_ms0));
                float32x2_t _rows1 = vpadd_f32(vget_low_f32(_ms1), vget_high_f32(_ms1));

                vst1_f32(rows0p + dx, _rows0);
                vst1_f32(rows1p + dx, _rows1);

                alphap += 4;
            }
#endif // __ARM_NEON
            for ( ; dx < w; dx++ )
            {
                int sx = xofs[dx];
                const float* S0p = S0 + sx;
                const float* S1p = S1 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                rows0p[dx] = S0p[0]*a0 + S0p[1]*a1;
                rows1p[dx] = S1p[0]*a0 + S1p[1]*a1;

                alphap += 2;
            }
        }

        prev_sy1 = sy + 1;

        // vresize
        float b0 = beta[0];
        float b1 = beta[1];

        float* rows0p = rows0;
        float* rows1p = rows1;
        float* Dp = dst.row(dy);

#if __ARM_NEON
        int nn = w >> 3;
#else
        int nn = 0;
#endif
        int remain = w - (nn << 3);

#if __ARM_NEON
        float32x4_t _b0 = vdupq_n_f32(b0);
        float32x4_t _b1 = vdupq_n_f32(b1);
        for (; nn>0; nn--)
        {
            float32x4_t _rows0 = vld1q_f32(rows0p);
            float32x4_t _rows1 = vld1q_f32(rows1p);

            float32x4_t _D = vmulq_f32(_rows0, _b0);
            _D = vmlaq_f32(_D, _rows1, _b1);

            vst1q_f32(Dp, _D);

            float32x4_t _rows0n = vld1q_f32(rows0p+4);
            float32x4_t _rows1n = vld1q_f32(rows1p+4);

            float32x4_t _Dn = vmulq_f32(_rows0n, _b0);
            _Dn = vmlaq_f32(_Dn, _rows1n, _b1);

            vst1q_f32(Dp+4, _Dn);

            Dp += 8;
            rows0p += 8;
            rows1p += 8;
        }
#endif // __ARM_NEON
        for ( ; remain; --remain )
        {
//             D[x] = rows0[x]*b0 + rows1[x]*b1;
            *Dp++ = *rows0p++ * b0 + *rows1p++ * b1;
        }

        beta += 2;
    }

    delete[] buf;
}

void resize_bilinear(const Mat& src, Mat& dst, int w, int h, Allocator* allocator, int num_threads)
{
    if (w == src.w && h == src.h)
    {
        dst = src;
        return;
    }

    size_t elemsize = src.elemsize;

    if (src.dims == 2)
    {
        dst.create(w, h, elemsize, allocator);
        if (dst.empty())
            return;

        resize_bilinear_image(src, dst, w, h);
    }
    else if (src.dims == 3)
    {
        int channels = src.c;

        dst.create(w, h, channels, elemsize, allocator);
        if (dst.empty())
            return;

        // unroll image channel
        #pragma omp parallel for num_threads(num_threads)
        for (int q=0; q<channels; q++)
        {
            const Mat m = src.channel(q);
            Mat resizem = dst.channel(q);

            resize_bilinear_image(m, resizem, w, h);
        }
    }
}

} // namespace ncnn
