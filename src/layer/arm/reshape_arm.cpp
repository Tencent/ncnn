// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "reshape_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "cpu.h"

#include <string.h>

namespace ncnn {

Reshape_arm::Reshape_arm()
{
#if __ARM_NEON
    support_packing = true;
#if NCNN_ARM82
    support_fp16_storage = cpu_support_arm_asimdhp();
#endif
#endif // __ARM_NEON

#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int Reshape_arm::forward_batch(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    Mat& top_blob = top_blobs[0];

    // resolve out shape
    int outw = w;
    int outh = h;
    int outd = d;
    int outc = c;

    if (!shape_expr.empty())
    {
        int er = eval_shape_expr(bottom_blobs, outw, outh, outd, outc);
        if (er != 0)
            return -1;
    }

    if (batch_mode == 2 && (outw == -1 || outh == -1 || outd == -1 || outc == -1))
        return -1;

    const int dims = bottom_blob.dims;
    const int elempack = bottom_blob.elempack;
    const size_t elemsize = bottom_blob.elemsize;
    const size_t scalar_elemsize = elemsize / elempack;

    int total = bottom_blob.w * bottom_blob.h * bottom_blob.d * bottom_blob.c * elempack;
    if (batch_mode == 1)
        total *= bottom_blob.n;

    if (ndim == 0)
        return -1;

    if (ndim == 1)
    {
        if (outw == 0)
            outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;

        if (outw == -1)
            outw = total;
    }
    if (ndim == 2)
    {
        if (outw == 0)
            outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
        if (outh == 0)
            outh = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;

        if (outw == -1)
            outw = total / outh;
        if (outh == -1)
            outh = total / outw;
    }
    if (ndim == 3)
    {
        if (outw == 0)
            outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
        if (outh == 0)
            outh = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;
        if (outc == 0)
            outc = dims == 3 ? bottom_blob.c * elempack : bottom_blob.c;

        if (outw == -1)
            outw = total / outc / outh;
        if (outh == -1)
            outh = total / outc / outw;
        if (outc == -1)
            outc = total / outh / outw;
    }
    if (ndim == 4)
    {
        if (outw == 0)
            outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
        if (outh == 0)
            outh = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;
        if (outc == 0)
            outc = (dims == 3 || dims == 4) ? bottom_blob.c * elempack : bottom_blob.c;
        if (outd == 0)
            outd = bottom_blob.d;

        if (outw == -1)
            outw = total / outc / outd / outh;
        if (outh == -1)
            outh = total / outc / outd / outw;
        if (outd == -1)
            outd = total / outc / outh / outw;
        if (outc == -1)
            outc = total / outd / outh / outw;
    }

    int out_elempack = 1;
    if (opt.use_packing_layout)
    {
        if (ndim == 1)
        {
#if NCNN_ARM82
            out_elempack = bottom_blob.elembits() == 16 && support_fp16_storage && opt.use_fp16_arithmetic && outw % 8 == 0 ? 8 : outw % 4 == 0 ? 4 : 1;
#else
            out_elempack = outw % 4 == 0 ? 4 : 1;
#endif
        }
        if (ndim == 2)
        {
#if NCNN_ARM82
            out_elempack = bottom_blob.elembits() == 16 && support_fp16_storage && opt.use_fp16_arithmetic && outh % 8 == 0 ? 8 : outh % 4 == 0 ? 4 : 1;
#else
            out_elempack = outh % 4 == 0 ? 4 : 1;
#endif
        }
        if (ndim == 3 || ndim == 4)
        {
#if NCNN_ARM82
            out_elempack = bottom_blob.elembits() == 16 && support_fp16_storage && opt.use_fp16_arithmetic && outc % 8 == 0 ? 8 : outc % 4 == 0 ? 4 : 1;
#else
            out_elempack = outc % 4 == 0 ? 4 : 1;
#endif
        }
    }
    const size_t out_elemsize = scalar_elemsize * out_elempack;

    int shape[4] = {0, 0, 0, 0};
    if (ndim == 1)
        shape[0] = outw;
    if (ndim == 2)
    {
        shape[0] = outh;
        shape[1] = outw;
    }
    if (ndim == 3)
    {
        shape[0] = outc;
        shape[1] = outh;
        shape[2] = outw;
    }
    if (ndim == 4)
    {
        shape[0] = outc;
        shape[1] = outd;
        shape[2] = outh;
        shape[3] = outw;
    }

    if (batch_mode == 1)
    {
        if (ndim == 1)
            top_blob.create(outw / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (ndim == 2)
            top_blob.create(outw, outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (ndim == 3)
            top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (ndim == 4)
            top_blob.create(outw, outh, outd, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);

        if (top_blob.empty())
            return -100;

        if (batch_axis == 0 && elempack == out_elempack && dims == 1 && ndim == 1 && top_blob.w == bottom_blob.w * bottom_blob.n)
        {
            const size_t size = (size_t)bottom_blob.w * elemsize;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int b = 0; b < bottom_blob.n; b++)
            {
                const unsigned char* ptr = (const unsigned char*)bottom_blob + (size_t)b * bottom_blob.nstep * elemsize;
                unsigned char* outptr = (unsigned char*)top_blob + (size_t)b * bottom_blob.w * out_elemsize;

                memcpy(outptr, ptr, size);
            }

            return 0;
        }

        if (batch_axis == 0 && elempack == out_elempack && dims == 2 && ndim == 2 && bottom_blob.w == top_blob.w && top_blob.h == bottom_blob.h * bottom_blob.n)
        {
            const size_t size = (size_t)bottom_blob.w * bottom_blob.h * elemsize;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int b = 0; b < bottom_blob.n; b++)
            {
                const unsigned char* ptr = (const unsigned char*)bottom_blob + (size_t)b * bottom_blob.nstep * elemsize;
                unsigned char* outptr = (unsigned char*)top_blob + (size_t)b * bottom_blob.w * bottom_blob.h * out_elemsize;

                memcpy(outptr, ptr, size);
            }

            return 0;
        }

        if (batch_axis == 0 && elempack == out_elempack && (dims == 3 || dims == 4) && (ndim == 3 || ndim == 4) && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == top_blob.d && top_blob.c == bottom_blob.c * bottom_blob.n)
        {
            const size_t size = (size_t)bottom_blob.w * bottom_blob.h * bottom_blob.d * elemsize;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bq = 0; bq < bottom_blob.n * bottom_blob.c; bq++)
            {
                const int b = bq / bottom_blob.c;
                const int q = bq - b * bottom_blob.c;

                const unsigned char* ptr = (const unsigned char*)bottom_blob + ((size_t)b * bottom_blob.nstep + (size_t)q * bottom_blob.cstep) * elemsize;
                unsigned char* outptr = (unsigned char*)top_blob + (size_t)bq * top_blob.cstep * out_elemsize;

                memcpy(outptr, ptr, size);
            }

            return 0;
        }

        if (batch_axis == 1 && elempack == out_elempack && dims == 2 && ndim == 3 && bottom_blob.w == top_blob.w && top_blob.h == bottom_blob.n && top_blob.c == bottom_blob.h)
        {
            const size_t size = (size_t)bottom_blob.w * elemsize;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bq = 0; bq < bottom_blob.n * bottom_blob.h; bq++)
            {
                const int b = bq / bottom_blob.h;
                const int q = bq - b * bottom_blob.h;

                const unsigned char* ptr = (const unsigned char*)bottom_blob + ((size_t)b * bottom_blob.nstep + (size_t)q * bottom_blob.w) * elemsize;
                unsigned char* outptr = (unsigned char*)top_blob + ((size_t)q * top_blob.cstep + (size_t)b * top_blob.w) * out_elemsize;

                memcpy(outptr, ptr, size);
            }

            return 0;
        }

        if (batch_axis == 1 && elempack == out_elempack && dims == 3 && ndim == 4 && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && top_blob.d == bottom_blob.n && top_blob.c == bottom_blob.c)
        {
            const size_t size = (size_t)bottom_blob.w * bottom_blob.h * elemsize;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bq = 0; bq < bottom_blob.n * bottom_blob.c; bq++)
            {
                const int b = bq / bottom_blob.c;
                const int q = bq - b * bottom_blob.c;

                const unsigned char* ptr = (const unsigned char*)bottom_blob + ((size_t)b * bottom_blob.nstep + (size_t)q * bottom_blob.cstep) * elemsize;
                unsigned char* outptr = (unsigned char*)top_blob + ((size_t)q * top_blob.cstep + (size_t)b * top_blob.w * top_blob.h) * out_elemsize;

                memcpy(outptr, ptr, size);
            }

            return 0;
        }

#if __ARM_NEON
        if (batch_axis == 0 && elempack == 1 && out_elempack == 4 && scalar_elemsize == 4 && dims == 2 && ndim == 2 && bottom_blob.w == top_blob.w && top_blob.h * 4 == bottom_blob.h * bottom_blob.n)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < top_blob.h; i++)
            {
                const int y = i * 4;
                const int b0 = y / bottom_blob.h;
                const int b1 = (y + 1) / bottom_blob.h;
                const int b2 = (y + 2) / bottom_blob.h;
                const int b3 = (y + 3) / bottom_blob.h;
                const int y0 = y - b0 * bottom_blob.h;
                const int y1 = y + 1 - b1 * bottom_blob.h;
                const int y2 = y + 2 - b2 * bottom_blob.h;
                const int y3 = y + 3 - b3 * bottom_blob.h;

                const float* ptr0 = (const float*)bottom_blob + (size_t)b0 * bottom_blob.nstep + (size_t)y0 * bottom_blob.w;
                const float* ptr1 = (const float*)bottom_blob + (size_t)b1 * bottom_blob.nstep + (size_t)y1 * bottom_blob.w;
                const float* ptr2 = (const float*)bottom_blob + (size_t)b2 * bottom_blob.nstep + (size_t)y2 * bottom_blob.w;
                const float* ptr3 = (const float*)bottom_blob + (size_t)b3 * bottom_blob.nstep + (size_t)y3 * bottom_blob.w;
                float* outptr = top_blob.row(i);

                int j = 0;
                for (; j + 3 < bottom_blob.w; j += 4)
                {
                    float32x4x4_t _v4;
                    _v4.val[0] = vld1q_f32(ptr0);
                    _v4.val[1] = vld1q_f32(ptr1);
                    _v4.val[2] = vld1q_f32(ptr2);
                    _v4.val[3] = vld1q_f32(ptr3);

                    vst4q_f32(outptr, _v4);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    ptr3 += 4;
                    outptr += 16;
                }
                for (; j < bottom_blob.w; j++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;

                    outptr += 4;
                }
            }

            return 0;
        }

        if (batch_axis == 0 && elempack == 1 && out_elempack == 4 && scalar_elemsize == 4 && (dims == 3 || dims == 4) && (ndim == 3 || ndim == 4) && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == top_blob.d && top_blob.c * 4 == bottom_blob.c * bottom_blob.n)
        {
            const int size = bottom_blob.w * bottom_blob.h * bottom_blob.d;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const int bq = q * 4;
                const int b0 = bq / bottom_blob.c;
                const int b1 = (bq + 1) / bottom_blob.c;
                const int b2 = (bq + 2) / bottom_blob.c;
                const int b3 = (bq + 3) / bottom_blob.c;
                const int q0 = bq - b0 * bottom_blob.c;
                const int q1 = bq + 1 - b1 * bottom_blob.c;
                const int q2 = bq + 2 - b2 * bottom_blob.c;
                const int q3 = bq + 3 - b3 * bottom_blob.c;

                const float* ptr0 = (const float*)bottom_blob + (size_t)b0 * bottom_blob.nstep + (size_t)q0 * bottom_blob.cstep;
                const float* ptr1 = (const float*)bottom_blob + (size_t)b1 * bottom_blob.nstep + (size_t)q1 * bottom_blob.cstep;
                const float* ptr2 = (const float*)bottom_blob + (size_t)b2 * bottom_blob.nstep + (size_t)q2 * bottom_blob.cstep;
                const float* ptr3 = (const float*)bottom_blob + (size_t)b3 * bottom_blob.nstep + (size_t)q3 * bottom_blob.cstep;
                float* outptr = top_blob.channel(q);

                int i = 0;
                for (; i + 3 < size; i += 4)
                {
                    float32x4x4_t _v4;
                    _v4.val[0] = vld1q_f32(ptr0);
                    _v4.val[1] = vld1q_f32(ptr1);
                    _v4.val[2] = vld1q_f32(ptr2);
                    _v4.val[3] = vld1q_f32(ptr3);

                    vst4q_f32(outptr, _v4);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    ptr3 += 4;
                    outptr += 16;
                }
                for (; i < size; i++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;

                    outptr += 4;
                }
            }

            return 0;
        }

        if (batch_axis == 1 && elempack == 1 && out_elempack == 4 && scalar_elemsize == 4 && dims == 3 && ndim == 4 && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && top_blob.d == bottom_blob.n && top_blob.c * 4 == bottom_blob.c)
        {
            const int total_bq = bottom_blob.n * top_blob.c;
            const int size = bottom_blob.w * bottom_blob.h;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bq = 0; bq < total_bq; bq++)
            {
                const int b = bq / top_blob.c;
                const int q = bq - b * top_blob.c;

                const Mat bottom_blob_b = bottom_blob.batch(b);
                const float* ptr0 = bottom_blob_b.channel(q * 4);
                const float* ptr1 = bottom_blob_b.channel(q * 4 + 1);
                const float* ptr2 = bottom_blob_b.channel(q * 4 + 2);
                const float* ptr3 = bottom_blob_b.channel(q * 4 + 3);
                float* outptr = top_blob.channel(q);
                outptr += (size_t)b * size * 4;

                int i = 0;
                for (; i + 3 < size; i += 4)
                {
                    float32x4x4_t _v4;
                    _v4.val[0] = vld1q_f32(ptr0);
                    _v4.val[1] = vld1q_f32(ptr1);
                    _v4.val[2] = vld1q_f32(ptr2);
                    _v4.val[3] = vld1q_f32(ptr3);

                    vst4q_f32(outptr, _v4);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    ptr3 += 4;
                    outptr += 16;
                }
                for (; i < size; i++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;

                    outptr += 4;
                }
            }

            return 0;
        }

        if (batch_axis == 0 && elempack == 1 && out_elempack == 4 && scalar_elemsize == 2 && dims == 2 && ndim == 2 && bottom_blob.w == top_blob.w && top_blob.h * 4 == bottom_blob.h * bottom_blob.n)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < top_blob.h; i++)
            {
                const int y = i * 4;
                const int b0 = y / bottom_blob.h;
                const int b1 = (y + 1) / bottom_blob.h;
                const int b2 = (y + 2) / bottom_blob.h;
                const int b3 = (y + 3) / bottom_blob.h;
                const int y0 = y - b0 * bottom_blob.h;
                const int y1 = y + 1 - b1 * bottom_blob.h;
                const int y2 = y + 2 - b2 * bottom_blob.h;
                const int y3 = y + 3 - b3 * bottom_blob.h;

                const unsigned short* ptr0 = (const unsigned short*)bottom_blob + (size_t)b0 * bottom_blob.nstep + (size_t)y0 * bottom_blob.w;
                const unsigned short* ptr1 = (const unsigned short*)bottom_blob + (size_t)b1 * bottom_blob.nstep + (size_t)y1 * bottom_blob.w;
                const unsigned short* ptr2 = (const unsigned short*)bottom_blob + (size_t)b2 * bottom_blob.nstep + (size_t)y2 * bottom_blob.w;
                const unsigned short* ptr3 = (const unsigned short*)bottom_blob + (size_t)b3 * bottom_blob.nstep + (size_t)y3 * bottom_blob.w;
                unsigned short* outptr = top_blob.row<unsigned short>(i);

                int j = 0;
                for (; j + 3 < bottom_blob.w; j += 4)
                {
                    uint16x4x4_t _v4;
                    _v4.val[0] = vld1_u16(ptr0);
                    _v4.val[1] = vld1_u16(ptr1);
                    _v4.val[2] = vld1_u16(ptr2);
                    _v4.val[3] = vld1_u16(ptr3);

                    vst4_u16(outptr, _v4);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    ptr3 += 4;
                    outptr += 16;
                }
                for (; j < bottom_blob.w; j++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;

                    outptr += 4;
                }
            }

            return 0;
        }

#if NCNN_ARM82
        if (batch_axis == 0 && elempack == 1 && out_elempack == 8 && scalar_elemsize == 2 && dims == 2 && ndim == 2 && bottom_blob.w == top_blob.w && top_blob.h * 8 == bottom_blob.h * bottom_blob.n)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < top_blob.h; i++)
            {
                const int y = i * 8;
                const int b0 = y / bottom_blob.h;
                const int b1 = (y + 1) / bottom_blob.h;
                const int b2 = (y + 2) / bottom_blob.h;
                const int b3 = (y + 3) / bottom_blob.h;
                const int b4 = (y + 4) / bottom_blob.h;
                const int b5 = (y + 5) / bottom_blob.h;
                const int b6 = (y + 6) / bottom_blob.h;
                const int b7 = (y + 7) / bottom_blob.h;
                const int y0 = y - b0 * bottom_blob.h;
                const int y1 = y + 1 - b1 * bottom_blob.h;
                const int y2 = y + 2 - b2 * bottom_blob.h;
                const int y3 = y + 3 - b3 * bottom_blob.h;
                const int y4 = y + 4 - b4 * bottom_blob.h;
                const int y5 = y + 5 - b5 * bottom_blob.h;
                const int y6 = y + 6 - b6 * bottom_blob.h;
                const int y7 = y + 7 - b7 * bottom_blob.h;

                const unsigned short* ptr0 = (const unsigned short*)bottom_blob + (size_t)b0 * bottom_blob.nstep + (size_t)y0 * bottom_blob.w;
                const unsigned short* ptr1 = (const unsigned short*)bottom_blob + (size_t)b1 * bottom_blob.nstep + (size_t)y1 * bottom_blob.w;
                const unsigned short* ptr2 = (const unsigned short*)bottom_blob + (size_t)b2 * bottom_blob.nstep + (size_t)y2 * bottom_blob.w;
                const unsigned short* ptr3 = (const unsigned short*)bottom_blob + (size_t)b3 * bottom_blob.nstep + (size_t)y3 * bottom_blob.w;
                const unsigned short* ptr4 = (const unsigned short*)bottom_blob + (size_t)b4 * bottom_blob.nstep + (size_t)y4 * bottom_blob.w;
                const unsigned short* ptr5 = (const unsigned short*)bottom_blob + (size_t)b5 * bottom_blob.nstep + (size_t)y5 * bottom_blob.w;
                const unsigned short* ptr6 = (const unsigned short*)bottom_blob + (size_t)b6 * bottom_blob.nstep + (size_t)y6 * bottom_blob.w;
                const unsigned short* ptr7 = (const unsigned short*)bottom_blob + (size_t)b7 * bottom_blob.nstep + (size_t)y7 * bottom_blob.w;
                unsigned short* outptr = top_blob.row<unsigned short>(i);

                int j = 0;
                for (; j + 3 < bottom_blob.w; j += 4)
                {
                    uint16x8_t _p01 = vcombine_u16(vld1_u16(ptr0), vld1_u16(ptr1));
                    uint16x8_t _p23 = vcombine_u16(vld1_u16(ptr2), vld1_u16(ptr3));
                    uint16x8_t _p45 = vcombine_u16(vld1_u16(ptr4), vld1_u16(ptr5));
                    uint16x8_t _p67 = vcombine_u16(vld1_u16(ptr6), vld1_u16(ptr7));

                    uint16x8x2_t _p0415 = vzipq_u16(_p01, _p45);
                    uint16x8x2_t _p2637 = vzipq_u16(_p23, _p67);

                    uint16x8x4_t _v4;
                    _v4.val[0] = _p0415.val[0];
                    _v4.val[1] = _p0415.val[1];
                    _v4.val[2] = _p2637.val[0];
                    _v4.val[3] = _p2637.val[1];

                    vst4q_u16(outptr, _v4);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    ptr3 += 4;
                    ptr4 += 4;
                    ptr5 += 4;
                    ptr6 += 4;
                    ptr7 += 4;
                    outptr += 32;
                }
                for (; j < bottom_blob.w; j++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;
                    outptr[4] = *ptr4++;
                    outptr[5] = *ptr5++;
                    outptr[6] = *ptr6++;
                    outptr[7] = *ptr7++;

                    outptr += 8;
                }
            }

            return 0;
        }
#endif // NCNN_ARM82

        if (batch_axis == 0 && elempack == 1 && out_elempack == 4 && scalar_elemsize == 2 && (dims == 3 || dims == 4) && (ndim == 3 || ndim == 4) && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == top_blob.d && top_blob.c * 4 == bottom_blob.c * bottom_blob.n)
        {
            const int size = bottom_blob.w * bottom_blob.h * bottom_blob.d;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const int bq = q * 4;
                const int b0 = bq / bottom_blob.c;
                const int b1 = (bq + 1) / bottom_blob.c;
                const int b2 = (bq + 2) / bottom_blob.c;
                const int b3 = (bq + 3) / bottom_blob.c;
                const int q0 = bq - b0 * bottom_blob.c;
                const int q1 = bq + 1 - b1 * bottom_blob.c;
                const int q2 = bq + 2 - b2 * bottom_blob.c;
                const int q3 = bq + 3 - b3 * bottom_blob.c;

                const unsigned short* ptr0 = (const unsigned short*)bottom_blob + (size_t)b0 * bottom_blob.nstep + (size_t)q0 * bottom_blob.cstep;
                const unsigned short* ptr1 = (const unsigned short*)bottom_blob + (size_t)b1 * bottom_blob.nstep + (size_t)q1 * bottom_blob.cstep;
                const unsigned short* ptr2 = (const unsigned short*)bottom_blob + (size_t)b2 * bottom_blob.nstep + (size_t)q2 * bottom_blob.cstep;
                const unsigned short* ptr3 = (const unsigned short*)bottom_blob + (size_t)b3 * bottom_blob.nstep + (size_t)q3 * bottom_blob.cstep;
                unsigned short* outptr = top_blob.channel(q);

                int i = 0;
                for (; i + 3 < size; i += 4)
                {
                    uint16x4x4_t _v4;
                    _v4.val[0] = vld1_u16(ptr0);
                    _v4.val[1] = vld1_u16(ptr1);
                    _v4.val[2] = vld1_u16(ptr2);
                    _v4.val[3] = vld1_u16(ptr3);

                    vst4_u16(outptr, _v4);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    ptr3 += 4;
                    outptr += 16;
                }
                for (; i < size; i++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;

                    outptr += 4;
                }
            }

            return 0;
        }

#if NCNN_ARM82
        if (batch_axis == 0 && elempack == 1 && out_elempack == 8 && scalar_elemsize == 2 && (dims == 3 || dims == 4) && (ndim == 3 || ndim == 4) && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == top_blob.d && top_blob.c * 8 == bottom_blob.c * bottom_blob.n)
        {
            const int size = bottom_blob.w * bottom_blob.h * bottom_blob.d;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const int bq = q * 8;
                const int b0 = bq / bottom_blob.c;
                const int b1 = (bq + 1) / bottom_blob.c;
                const int b2 = (bq + 2) / bottom_blob.c;
                const int b3 = (bq + 3) / bottom_blob.c;
                const int b4 = (bq + 4) / bottom_blob.c;
                const int b5 = (bq + 5) / bottom_blob.c;
                const int b6 = (bq + 6) / bottom_blob.c;
                const int b7 = (bq + 7) / bottom_blob.c;
                const int q0 = bq - b0 * bottom_blob.c;
                const int q1 = bq + 1 - b1 * bottom_blob.c;
                const int q2 = bq + 2 - b2 * bottom_blob.c;
                const int q3 = bq + 3 - b3 * bottom_blob.c;
                const int q4 = bq + 4 - b4 * bottom_blob.c;
                const int q5 = bq + 5 - b5 * bottom_blob.c;
                const int q6 = bq + 6 - b6 * bottom_blob.c;
                const int q7 = bq + 7 - b7 * bottom_blob.c;

                const unsigned short* ptr0 = (const unsigned short*)bottom_blob + (size_t)b0 * bottom_blob.nstep + (size_t)q0 * bottom_blob.cstep;
                const unsigned short* ptr1 = (const unsigned short*)bottom_blob + (size_t)b1 * bottom_blob.nstep + (size_t)q1 * bottom_blob.cstep;
                const unsigned short* ptr2 = (const unsigned short*)bottom_blob + (size_t)b2 * bottom_blob.nstep + (size_t)q2 * bottom_blob.cstep;
                const unsigned short* ptr3 = (const unsigned short*)bottom_blob + (size_t)b3 * bottom_blob.nstep + (size_t)q3 * bottom_blob.cstep;
                const unsigned short* ptr4 = (const unsigned short*)bottom_blob + (size_t)b4 * bottom_blob.nstep + (size_t)q4 * bottom_blob.cstep;
                const unsigned short* ptr5 = (const unsigned short*)bottom_blob + (size_t)b5 * bottom_blob.nstep + (size_t)q5 * bottom_blob.cstep;
                const unsigned short* ptr6 = (const unsigned short*)bottom_blob + (size_t)b6 * bottom_blob.nstep + (size_t)q6 * bottom_blob.cstep;
                const unsigned short* ptr7 = (const unsigned short*)bottom_blob + (size_t)b7 * bottom_blob.nstep + (size_t)q7 * bottom_blob.cstep;
                unsigned short* outptr = top_blob.channel(q);

                int i = 0;
                for (; i + 3 < size; i += 4)
                {
                    uint16x8_t _p01 = vcombine_u16(vld1_u16(ptr0), vld1_u16(ptr1));
                    uint16x8_t _p23 = vcombine_u16(vld1_u16(ptr2), vld1_u16(ptr3));
                    uint16x8_t _p45 = vcombine_u16(vld1_u16(ptr4), vld1_u16(ptr5));
                    uint16x8_t _p67 = vcombine_u16(vld1_u16(ptr6), vld1_u16(ptr7));

                    uint16x8x2_t _p0415 = vzipq_u16(_p01, _p45);
                    uint16x8x2_t _p2637 = vzipq_u16(_p23, _p67);

                    uint16x8x4_t _v4;
                    _v4.val[0] = _p0415.val[0];
                    _v4.val[1] = _p0415.val[1];
                    _v4.val[2] = _p2637.val[0];
                    _v4.val[3] = _p2637.val[1];

                    vst4q_u16(outptr, _v4);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    ptr3 += 4;
                    ptr4 += 4;
                    ptr5 += 4;
                    ptr6 += 4;
                    ptr7 += 4;
                    outptr += 32;
                }
                for (; i < size; i++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;
                    outptr[4] = *ptr4++;
                    outptr[5] = *ptr5++;
                    outptr[6] = *ptr6++;
                    outptr[7] = *ptr7++;

                    outptr += 8;
                }
            }

            return 0;
        }
#endif // NCNN_ARM82

        if (batch_axis == 1 && elempack == 1 && out_elempack == 4 && scalar_elemsize == 2 && dims == 3 && ndim == 4 && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && top_blob.d == bottom_blob.n && top_blob.c * 4 == bottom_blob.c)
        {
            const int total_bq = bottom_blob.n * top_blob.c;
            const int size = bottom_blob.w * bottom_blob.h;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bq = 0; bq < total_bq; bq++)
            {
                const int b = bq / top_blob.c;
                const int q = bq - b * top_blob.c;

                const Mat bottom_blob_b = bottom_blob.batch(b);
                const unsigned short* ptr0 = bottom_blob_b.channel(q * 4);
                const unsigned short* ptr1 = bottom_blob_b.channel(q * 4 + 1);
                const unsigned short* ptr2 = bottom_blob_b.channel(q * 4 + 2);
                const unsigned short* ptr3 = bottom_blob_b.channel(q * 4 + 3);
                unsigned short* outptr = top_blob.channel(q);
                outptr += (size_t)b * size * 4;

                int i = 0;
                for (; i + 3 < size; i += 4)
                {
                    uint16x4x4_t _v4;
                    _v4.val[0] = vld1_u16(ptr0);
                    _v4.val[1] = vld1_u16(ptr1);
                    _v4.val[2] = vld1_u16(ptr2);
                    _v4.val[3] = vld1_u16(ptr3);

                    vst4_u16(outptr, _v4);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    ptr3 += 4;
                    outptr += 16;
                }
                for (; i < size; i++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;

                    outptr += 4;
                }
            }

            return 0;
        }
#endif // __ARM_NEON

        size_t prefix = 1;
        for (int i = 0; i < batch_axis; i++)
            prefix *= shape[i];

        size_t suffix = 1;
        if (batch_axis == 0)
            suffix = (size_t)bottom_blob.w * bottom_blob.h * bottom_blob.d * bottom_blob.c * elempack;
        else
        {
            for (int i = batch_axis + 1; i < ndim; i++)
                suffix *= shape[i];
        }

        const size_t bottom_channel_size = (size_t)bottom_blob.w * bottom_blob.h * bottom_blob.d;
        const size_t top_channel_size = (size_t)top_blob.w * top_blob.h * top_blob.d;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp = 0; pp < (int)prefix; pp++)
        {
            for (int b = 0; b < bottom_blob.n; b++)
            {
                for (size_t s = 0; s < suffix; s++)
                {
                    const size_t srci = batch_axis == 0 ? s : (size_t)pp * suffix + s;
                    const unsigned char* ptr = (const unsigned char*)bottom_blob + (size_t)b * bottom_blob.nstep * elemsize;
                    if (dims == 1)
                    {
                        const int x = srci / elempack;
                        const int k = srci % elempack;
                        ptr += (size_t)x * elemsize + k * scalar_elemsize;
                    }
                    else if (dims == 2)
                    {
                        const int x = srci % bottom_blob.w;
                        const int y = srci / bottom_blob.w;
                        const int y0 = y / elempack;
                        const int k = y % elempack;
                        ptr += ((size_t)y0 * bottom_blob.w + x) * elemsize + k * scalar_elemsize;
                    }
                    else
                    {
                        const int q = srci / bottom_channel_size;
                        const size_t r = srci - (size_t)q * bottom_channel_size;
                        const int q0 = q / elempack;
                        const int k = q % elempack;
                        ptr += ((size_t)q0 * bottom_blob.cstep + r) * elemsize + k * scalar_elemsize;
                    }

                    const size_t dsti = batch_axis == 0 ? (size_t)b * suffix + s : ((size_t)pp * bottom_blob.n + b) * suffix + s;
                    unsigned char* outptr = (unsigned char*)top_blob;
                    if (top_blob.dims == 1)
                    {
                        const int x = dsti / out_elempack;
                        const int k = dsti % out_elempack;
                        outptr += (size_t)x * out_elemsize + k * scalar_elemsize;
                    }
                    else if (top_blob.dims == 2)
                    {
                        const int x = dsti % top_blob.w;
                        const int y = dsti / top_blob.w;
                        const int y0 = y / out_elempack;
                        const int k = y % out_elempack;
                        outptr += ((size_t)y0 * top_blob.w + x) * out_elemsize + k * scalar_elemsize;
                    }
                    else
                    {
                        const int q = dsti / top_channel_size;
                        const size_t r = dsti - (size_t)q * top_channel_size;
                        const int q0 = q / out_elempack;
                        const int k = q % out_elempack;
                        outptr += ((size_t)q0 * top_blob.cstep + r) * out_elemsize + k * scalar_elemsize;
                    }

                    memcpy(outptr, ptr, scalar_elemsize);
                }
            }
        }

        return 0;
    }

    if (batch_mode == 2)
    {
        if (bottom_blob.n != 1)
            return -1;

        size_t out_total = outw;
        if (ndim == 2)
            out_total *= outh;
        if (ndim == 3)
            out_total *= (size_t)outh * outc;
        if (ndim == 4)
            out_total *= (size_t)outh * outd * outc;

        if (out_total == 0)
            return -1;

        const size_t bottom_total = (size_t)bottom_blob.w * bottom_blob.h * bottom_blob.d * bottom_blob.c * elempack;
        const int batch = bottom_total / out_total;
        if ((size_t)batch * out_total != bottom_total)
            return -1;

        if (ndim == 1)
            top_blob.create_batch(outw / out_elempack, batch, out_elemsize, out_elempack, opt.blob_allocator);
        if (ndim == 2)
            top_blob.create_batch(outw, outh / out_elempack, batch, out_elemsize, out_elempack, opt.blob_allocator);
        if (ndim == 3)
            top_blob.create_batch(outw, outh, outc / out_elempack, batch, out_elemsize, out_elempack, opt.blob_allocator);
        if (ndim == 4)
            top_blob.create_batch(outw, outh, outd, outc / out_elempack, batch, out_elemsize, out_elempack, opt.blob_allocator);

        if (top_blob.empty())
            return -100;

        if (batch_axis == 0 && elempack == out_elempack && dims == 1 && ndim == 1 && bottom_blob.w == top_blob.w * batch)
        {
            const size_t size = (size_t)top_blob.w * out_elemsize;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int b = 0; b < batch; b++)
            {
                const unsigned char* ptr = (const unsigned char*)bottom_blob + (size_t)b * top_blob.w * elemsize;
                unsigned char* outptr = (unsigned char*)top_blob + (size_t)b * top_blob.nstep * out_elemsize;

                memcpy(outptr, ptr, size);
            }

            return 0;
        }

        if (batch_axis == 0 && elempack == out_elempack && dims == 2 && ndim == 2 && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h * batch)
        {
            const size_t size = (size_t)top_blob.w * top_blob.h * out_elemsize;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int b = 0; b < batch; b++)
            {
                const unsigned char* ptr = (const unsigned char*)bottom_blob + (size_t)b * top_blob.w * top_blob.h * elemsize;
                unsigned char* outptr = (unsigned char*)top_blob + (size_t)b * top_blob.nstep * out_elemsize;

                memcpy(outptr, ptr, size);
            }

            return 0;
        }

        if (batch_axis == 0 && elempack == out_elempack && (dims == 3 || dims == 4) && (ndim == 3 || ndim == 4) && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == top_blob.d && bottom_blob.c == top_blob.c * batch)
        {
            const size_t size = (size_t)top_blob.w * top_blob.h * top_blob.d * out_elemsize;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bq = 0; bq < batch * top_blob.c; bq++)
            {
                const int b = bq / top_blob.c;
                const int q = bq - b * top_blob.c;

                const unsigned char* ptr = (const unsigned char*)bottom_blob + (size_t)bq * bottom_blob.cstep * elemsize;
                unsigned char* outptr = (unsigned char*)top_blob + ((size_t)b * top_blob.nstep + (size_t)q * top_blob.cstep) * out_elemsize;

                memcpy(outptr, ptr, size);
            }

            return 0;
        }

        if (batch_axis == 1 && elempack == out_elempack && dims == 3 && ndim == 2 && bottom_blob.w == top_blob.w && bottom_blob.h == batch && bottom_blob.c == top_blob.h)
        {
            const size_t size = (size_t)top_blob.w * out_elemsize;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bq = 0; bq < batch * top_blob.h; bq++)
            {
                const int b = bq / top_blob.h;
                const int q = bq - b * top_blob.h;

                const unsigned char* ptr = (const unsigned char*)bottom_blob + ((size_t)q * bottom_blob.cstep + (size_t)b * bottom_blob.w) * elemsize;
                unsigned char* outptr = (unsigned char*)top_blob + ((size_t)b * top_blob.nstep + (size_t)q * top_blob.w) * out_elemsize;

                memcpy(outptr, ptr, size);
            }

            return 0;
        }

        if (batch_axis == 1 && elempack == out_elempack && dims == 4 && ndim == 3 && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == batch && bottom_blob.c == top_blob.c)
        {
            const size_t size = (size_t)top_blob.w * top_blob.h * out_elemsize;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bq = 0; bq < batch * top_blob.c; bq++)
            {
                const int b = bq / top_blob.c;
                const int q = bq - b * top_blob.c;

                const unsigned char* ptr = (const unsigned char*)bottom_blob + ((size_t)q * bottom_blob.cstep + (size_t)b * bottom_blob.w * bottom_blob.h) * elemsize;
                unsigned char* outptr = (unsigned char*)top_blob + ((size_t)b * top_blob.nstep + (size_t)q * top_blob.cstep) * out_elemsize;

                memcpy(outptr, ptr, size);
            }

            return 0;
        }

#if __ARM_NEON
        if (batch_axis == 0 && elempack == 1 && out_elempack == 4 && scalar_elemsize == 4 && dims == 2 && ndim == 2 && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h * batch * 4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bi = 0; bi < batch * top_blob.h; bi++)
            {
                const int b = bi / top_blob.h;
                const int i = bi - b * top_blob.h;
                const int y = bi * 4;

                const float* ptr0 = bottom_blob.row(y);
                const float* ptr1 = bottom_blob.row(y + 1);
                const float* ptr2 = bottom_blob.row(y + 2);
                const float* ptr3 = bottom_blob.row(y + 3);
                float* outptr = (float*)top_blob + ((size_t)b * top_blob.nstep + (size_t)i * top_blob.w) * 4;

                int j = 0;
                for (; j + 3 < bottom_blob.w; j += 4)
                {
                    float32x4x4_t _v4;
                    _v4.val[0] = vld1q_f32(ptr0);
                    _v4.val[1] = vld1q_f32(ptr1);
                    _v4.val[2] = vld1q_f32(ptr2);
                    _v4.val[3] = vld1q_f32(ptr3);

                    vst4q_f32(outptr, _v4);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    ptr3 += 4;
                    outptr += 16;
                }
                for (; j < bottom_blob.w; j++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;

                    outptr += 4;
                }
            }

            return 0;
        }

        if (batch_axis == 0 && elempack == 1 && out_elempack == 4 && scalar_elemsize == 2 && dims == 2 && ndim == 2 && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h * batch * 4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bi = 0; bi < batch * top_blob.h; bi++)
            {
                const int b = bi / top_blob.h;
                const int i = bi - b * top_blob.h;
                const int y = bi * 4;

                const unsigned short* ptr0 = bottom_blob.row<const unsigned short>(y);
                const unsigned short* ptr1 = bottom_blob.row<const unsigned short>(y + 1);
                const unsigned short* ptr2 = bottom_blob.row<const unsigned short>(y + 2);
                const unsigned short* ptr3 = bottom_blob.row<const unsigned short>(y + 3);
                unsigned short* outptr = (unsigned short*)top_blob + ((size_t)b * top_blob.nstep + (size_t)i * top_blob.w) * 4;

                int j = 0;
                for (; j + 3 < bottom_blob.w; j += 4)
                {
                    uint16x4x4_t _v4;
                    _v4.val[0] = vld1_u16(ptr0);
                    _v4.val[1] = vld1_u16(ptr1);
                    _v4.val[2] = vld1_u16(ptr2);
                    _v4.val[3] = vld1_u16(ptr3);

                    vst4_u16(outptr, _v4);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    ptr3 += 4;
                    outptr += 16;
                }
                for (; j < bottom_blob.w; j++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;

                    outptr += 4;
                }
            }

            return 0;
        }

#if NCNN_ARM82
        if (batch_axis == 0 && elempack == 1 && out_elempack == 8 && scalar_elemsize == 2 && dims == 2 && ndim == 2 && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h * batch * 8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bi = 0; bi < batch * top_blob.h; bi++)
            {
                const int b = bi / top_blob.h;
                const int i = bi - b * top_blob.h;
                const int y = bi * 8;

                const unsigned short* ptr0 = bottom_blob.row<const unsigned short>(y);
                const unsigned short* ptr1 = bottom_blob.row<const unsigned short>(y + 1);
                const unsigned short* ptr2 = bottom_blob.row<const unsigned short>(y + 2);
                const unsigned short* ptr3 = bottom_blob.row<const unsigned short>(y + 3);
                const unsigned short* ptr4 = bottom_blob.row<const unsigned short>(y + 4);
                const unsigned short* ptr5 = bottom_blob.row<const unsigned short>(y + 5);
                const unsigned short* ptr6 = bottom_blob.row<const unsigned short>(y + 6);
                const unsigned short* ptr7 = bottom_blob.row<const unsigned short>(y + 7);
                unsigned short* outptr = (unsigned short*)top_blob + ((size_t)b * top_blob.nstep + (size_t)i * top_blob.w) * 8;

                int j = 0;
                for (; j + 3 < bottom_blob.w; j += 4)
                {
                    uint16x8_t _p01 = vcombine_u16(vld1_u16(ptr0), vld1_u16(ptr1));
                    uint16x8_t _p23 = vcombine_u16(vld1_u16(ptr2), vld1_u16(ptr3));
                    uint16x8_t _p45 = vcombine_u16(vld1_u16(ptr4), vld1_u16(ptr5));
                    uint16x8_t _p67 = vcombine_u16(vld1_u16(ptr6), vld1_u16(ptr7));

                    uint16x8x2_t _p0415 = vzipq_u16(_p01, _p45);
                    uint16x8x2_t _p2637 = vzipq_u16(_p23, _p67);

                    uint16x8x4_t _v4;
                    _v4.val[0] = _p0415.val[0];
                    _v4.val[1] = _p0415.val[1];
                    _v4.val[2] = _p2637.val[0];
                    _v4.val[3] = _p2637.val[1];

                    vst4q_u16(outptr, _v4);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    ptr3 += 4;
                    ptr4 += 4;
                    ptr5 += 4;
                    ptr6 += 4;
                    ptr7 += 4;
                    outptr += 32;
                }
                for (; j < bottom_blob.w; j++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;
                    outptr[4] = *ptr4++;
                    outptr[5] = *ptr5++;
                    outptr[6] = *ptr6++;
                    outptr[7] = *ptr7++;

                    outptr += 8;
                }
            }

            return 0;
        }
#endif // NCNN_ARM82

        if (batch_axis == 0 && elempack == 1 && out_elempack == 4 && scalar_elemsize == 4 && (dims == 3 || dims == 4) && (ndim == 3 || ndim == 4) && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == top_blob.d && bottom_blob.c == top_blob.c * batch * 4)
        {
            const int size = top_blob.w * top_blob.h * top_blob.d;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bq = 0; bq < batch * top_blob.c; bq++)
            {
                const int b = bq / top_blob.c;
                const int q = bq - b * top_blob.c;
                const int sq = b * top_blob.c * 4 + q * 4;

                const float* ptr0 = bottom_blob.channel(sq);
                const float* ptr1 = bottom_blob.channel(sq + 1);
                const float* ptr2 = bottom_blob.channel(sq + 2);
                const float* ptr3 = bottom_blob.channel(sq + 3);
                float* outptr = (float*)top_blob + ((size_t)b * top_blob.nstep + (size_t)q * top_blob.cstep) * 4;

                int i = 0;
                for (; i + 3 < size; i += 4)
                {
                    float32x4x4_t _v4;
                    _v4.val[0] = vld1q_f32(ptr0);
                    _v4.val[1] = vld1q_f32(ptr1);
                    _v4.val[2] = vld1q_f32(ptr2);
                    _v4.val[3] = vld1q_f32(ptr3);

                    vst4q_f32(outptr, _v4);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    ptr3 += 4;
                    outptr += 16;
                }
                for (; i < size; i++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;

                    outptr += 4;
                }
            }

            return 0;
        }

        if (batch_axis == 0 && elempack == 1 && out_elempack == 4 && scalar_elemsize == 2 && (dims == 3 || dims == 4) && (ndim == 3 || ndim == 4) && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == top_blob.d && bottom_blob.c == top_blob.c * batch * 4)
        {
            const int size = top_blob.w * top_blob.h * top_blob.d;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bq = 0; bq < batch * top_blob.c; bq++)
            {
                const int b = bq / top_blob.c;
                const int q = bq - b * top_blob.c;
                const int sq = b * top_blob.c * 4 + q * 4;

                const unsigned short* ptr0 = bottom_blob.channel(sq);
                const unsigned short* ptr1 = bottom_blob.channel(sq + 1);
                const unsigned short* ptr2 = bottom_blob.channel(sq + 2);
                const unsigned short* ptr3 = bottom_blob.channel(sq + 3);
                unsigned short* outptr = (unsigned short*)top_blob + ((size_t)b * top_blob.nstep + (size_t)q * top_blob.cstep) * 4;

                int i = 0;
                for (; i + 3 < size; i += 4)
                {
                    uint16x4x4_t _v4;
                    _v4.val[0] = vld1_u16(ptr0);
                    _v4.val[1] = vld1_u16(ptr1);
                    _v4.val[2] = vld1_u16(ptr2);
                    _v4.val[3] = vld1_u16(ptr3);

                    vst4_u16(outptr, _v4);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    ptr3 += 4;
                    outptr += 16;
                }
                for (; i < size; i++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;

                    outptr += 4;
                }
            }

            return 0;
        }

#if NCNN_ARM82
        if (batch_axis == 0 && elempack == 1 && out_elempack == 8 && scalar_elemsize == 2 && (dims == 3 || dims == 4) && (ndim == 3 || ndim == 4) && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == top_blob.d && bottom_blob.c == top_blob.c * batch * 8)
        {
            const int size = top_blob.w * top_blob.h * top_blob.d;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bq = 0; bq < batch * top_blob.c; bq++)
            {
                const int b = bq / top_blob.c;
                const int q = bq - b * top_blob.c;
                const int sq = b * top_blob.c * 8 + q * 8;

                const unsigned short* ptr0 = bottom_blob.channel(sq);
                const unsigned short* ptr1 = bottom_blob.channel(sq + 1);
                const unsigned short* ptr2 = bottom_blob.channel(sq + 2);
                const unsigned short* ptr3 = bottom_blob.channel(sq + 3);
                const unsigned short* ptr4 = bottom_blob.channel(sq + 4);
                const unsigned short* ptr5 = bottom_blob.channel(sq + 5);
                const unsigned short* ptr6 = bottom_blob.channel(sq + 6);
                const unsigned short* ptr7 = bottom_blob.channel(sq + 7);
                unsigned short* outptr = (unsigned short*)top_blob + ((size_t)b * top_blob.nstep + (size_t)q * top_blob.cstep) * 8;

                int i = 0;
                for (; i + 3 < size; i += 4)
                {
                    uint16x8_t _p01 = vcombine_u16(vld1_u16(ptr0), vld1_u16(ptr1));
                    uint16x8_t _p23 = vcombine_u16(vld1_u16(ptr2), vld1_u16(ptr3));
                    uint16x8_t _p45 = vcombine_u16(vld1_u16(ptr4), vld1_u16(ptr5));
                    uint16x8_t _p67 = vcombine_u16(vld1_u16(ptr6), vld1_u16(ptr7));

                    uint16x8x2_t _p0415 = vzipq_u16(_p01, _p45);
                    uint16x8x2_t _p2637 = vzipq_u16(_p23, _p67);

                    uint16x8x4_t _v4;
                    _v4.val[0] = _p0415.val[0];
                    _v4.val[1] = _p0415.val[1];
                    _v4.val[2] = _p2637.val[0];
                    _v4.val[3] = _p2637.val[1];

                    vst4q_u16(outptr, _v4);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    ptr3 += 4;
                    ptr4 += 4;
                    ptr5 += 4;
                    ptr6 += 4;
                    ptr7 += 4;
                    outptr += 32;
                }
                for (; i < size; i++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;
                    outptr[4] = *ptr4++;
                    outptr[5] = *ptr5++;
                    outptr[6] = *ptr6++;
                    outptr[7] = *ptr7++;

                    outptr += 8;
                }
            }

            return 0;
        }
#endif // NCNN_ARM82

        if (batch_axis == 0 && elempack == 4 && out_elempack == 1 && scalar_elemsize == 4 && dims == 2 && ndim == 2 && bottom_blob.w == top_blob.w && bottom_blob.h * 4 == top_blob.h * batch)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < bottom_blob.h; i++)
            {
                const int y = i * 4;
                const int b0 = y / top_blob.h;
                const int b1 = (y + 1) / top_blob.h;
                const int b2 = (y + 2) / top_blob.h;
                const int b3 = (y + 3) / top_blob.h;
                const int y0 = y - b0 * top_blob.h;
                const int y1 = y + 1 - b1 * top_blob.h;
                const int y2 = y + 2 - b2 * top_blob.h;
                const int y3 = y + 3 - b3 * top_blob.h;

                const float* ptr = bottom_blob.row(i);
                float* outptr0 = (float*)top_blob + (size_t)b0 * top_blob.nstep + (size_t)y0 * top_blob.w;
                float* outptr1 = (float*)top_blob + (size_t)b1 * top_blob.nstep + (size_t)y1 * top_blob.w;
                float* outptr2 = (float*)top_blob + (size_t)b2 * top_blob.nstep + (size_t)y2 * top_blob.w;
                float* outptr3 = (float*)top_blob + (size_t)b3 * top_blob.nstep + (size_t)y3 * top_blob.w;

                int j = 0;
                for (; j + 3 < bottom_blob.w; j += 4)
                {
                    float32x4x4_t _v4 = vld4q_f32(ptr);
                    vst1q_f32(outptr0, _v4.val[0]);
                    vst1q_f32(outptr1, _v4.val[1]);
                    vst1q_f32(outptr2, _v4.val[2]);
                    vst1q_f32(outptr3, _v4.val[3]);

                    ptr += 16;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                }
                for (; j < bottom_blob.w; j++)
                {
                    *outptr0++ = ptr[0];
                    *outptr1++ = ptr[1];
                    *outptr2++ = ptr[2];
                    *outptr3++ = ptr[3];

                    ptr += 4;
                }
            }

            return 0;
        }

        if (batch_axis == 0 && elempack == 4 && out_elempack == 1 && scalar_elemsize == 4 && (dims == 3 || dims == 4) && (ndim == 3 || ndim == 4) && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == top_blob.d && bottom_blob.c * 4 == top_blob.c * batch)
        {
            const int size = top_blob.w * top_blob.h * top_blob.d;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < bottom_blob.c; q++)
            {
                const int bq = q * 4;
                const int b0 = bq / top_blob.c;
                const int b1 = (bq + 1) / top_blob.c;
                const int b2 = (bq + 2) / top_blob.c;
                const int b3 = (bq + 3) / top_blob.c;
                const int q0 = bq - b0 * top_blob.c;
                const int q1 = bq + 1 - b1 * top_blob.c;
                const int q2 = bq + 2 - b2 * top_blob.c;
                const int q3 = bq + 3 - b3 * top_blob.c;

                const float* ptr = bottom_blob.channel(q);
                float* outptr0 = (float*)top_blob + (size_t)b0 * top_blob.nstep + (size_t)q0 * top_blob.cstep;
                float* outptr1 = (float*)top_blob + (size_t)b1 * top_blob.nstep + (size_t)q1 * top_blob.cstep;
                float* outptr2 = (float*)top_blob + (size_t)b2 * top_blob.nstep + (size_t)q2 * top_blob.cstep;
                float* outptr3 = (float*)top_blob + (size_t)b3 * top_blob.nstep + (size_t)q3 * top_blob.cstep;

                int i = 0;
                for (; i + 3 < size; i += 4)
                {
                    float32x4x4_t _v4 = vld4q_f32(ptr);
                    vst1q_f32(outptr0, _v4.val[0]);
                    vst1q_f32(outptr1, _v4.val[1]);
                    vst1q_f32(outptr2, _v4.val[2]);
                    vst1q_f32(outptr3, _v4.val[3]);

                    ptr += 16;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                }
                for (; i < size; i++)
                {
                    *outptr0++ = ptr[0];
                    *outptr1++ = ptr[1];
                    *outptr2++ = ptr[2];
                    *outptr3++ = ptr[3];

                    ptr += 4;
                }
            }

            return 0;
        }

        if (batch_axis == 1 && elempack == 4 && out_elempack == 1 && scalar_elemsize == 4 && dims == 4 && ndim == 3 && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == batch && bottom_blob.c * 4 == top_blob.c)
        {
            const int total_bq = batch * bottom_blob.c;
            const int size = top_blob.w * top_blob.h;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bq = 0; bq < total_bq; bq++)
            {
                const int b = bq / bottom_blob.c;
                const int q = bq - b * bottom_blob.c;

                const float* ptr = bottom_blob.channel(q);
                ptr += (size_t)b * size * 4;
                Mat top_blob_b = top_blob.batch(b);
                float* outptr0 = top_blob_b.channel(q * 4);
                float* outptr1 = top_blob_b.channel(q * 4 + 1);
                float* outptr2 = top_blob_b.channel(q * 4 + 2);
                float* outptr3 = top_blob_b.channel(q * 4 + 3);

                int i = 0;
                for (; i + 3 < size; i += 4)
                {
                    float32x4x4_t _v4 = vld4q_f32(ptr);
                    vst1q_f32(outptr0, _v4.val[0]);
                    vst1q_f32(outptr1, _v4.val[1]);
                    vst1q_f32(outptr2, _v4.val[2]);
                    vst1q_f32(outptr3, _v4.val[3]);

                    ptr += 16;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                }
                for (; i < size; i++)
                {
                    *outptr0++ = ptr[0];
                    *outptr1++ = ptr[1];
                    *outptr2++ = ptr[2];
                    *outptr3++ = ptr[3];

                    ptr += 4;
                }
            }

            return 0;
        }

        if (batch_axis == 0 && elempack == 4 && out_elempack == 1 && scalar_elemsize == 2 && dims == 2 && ndim == 2 && bottom_blob.w == top_blob.w && bottom_blob.h * 4 == top_blob.h * batch)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < bottom_blob.h; i++)
            {
                const int y = i * 4;
                const int b0 = y / top_blob.h;
                const int b1 = (y + 1) / top_blob.h;
                const int b2 = (y + 2) / top_blob.h;
                const int b3 = (y + 3) / top_blob.h;
                const int y0 = y - b0 * top_blob.h;
                const int y1 = y + 1 - b1 * top_blob.h;
                const int y2 = y + 2 - b2 * top_blob.h;
                const int y3 = y + 3 - b3 * top_blob.h;

                const unsigned short* ptr = bottom_blob.row<const unsigned short>(i);
                unsigned short* outptr0 = (unsigned short*)top_blob + (size_t)b0 * top_blob.nstep + (size_t)y0 * top_blob.w;
                unsigned short* outptr1 = (unsigned short*)top_blob + (size_t)b1 * top_blob.nstep + (size_t)y1 * top_blob.w;
                unsigned short* outptr2 = (unsigned short*)top_blob + (size_t)b2 * top_blob.nstep + (size_t)y2 * top_blob.w;
                unsigned short* outptr3 = (unsigned short*)top_blob + (size_t)b3 * top_blob.nstep + (size_t)y3 * top_blob.w;

                int j = 0;
                for (; j + 3 < bottom_blob.w; j += 4)
                {
                    uint16x4x4_t _v4 = vld4_u16(ptr);
                    vst1_u16(outptr0, _v4.val[0]);
                    vst1_u16(outptr1, _v4.val[1]);
                    vst1_u16(outptr2, _v4.val[2]);
                    vst1_u16(outptr3, _v4.val[3]);

                    ptr += 16;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                }
                for (; j < bottom_blob.w; j++)
                {
                    *outptr0++ = ptr[0];
                    *outptr1++ = ptr[1];
                    *outptr2++ = ptr[2];
                    *outptr3++ = ptr[3];

                    ptr += 4;
                }
            }

            return 0;
        }

#if NCNN_ARM82
        if (batch_axis == 0 && elempack == 8 && out_elempack == 1 && scalar_elemsize == 2 && dims == 2 && ndim == 2 && bottom_blob.w == top_blob.w && bottom_blob.h * 8 == top_blob.h * batch)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < bottom_blob.h; i++)
            {
                const int y = i * 8;
                const int b0 = y / top_blob.h;
                const int b1 = (y + 1) / top_blob.h;
                const int b2 = (y + 2) / top_blob.h;
                const int b3 = (y + 3) / top_blob.h;
                const int b4 = (y + 4) / top_blob.h;
                const int b5 = (y + 5) / top_blob.h;
                const int b6 = (y + 6) / top_blob.h;
                const int b7 = (y + 7) / top_blob.h;
                const int y0 = y - b0 * top_blob.h;
                const int y1 = y + 1 - b1 * top_blob.h;
                const int y2 = y + 2 - b2 * top_blob.h;
                const int y3 = y + 3 - b3 * top_blob.h;
                const int y4 = y + 4 - b4 * top_blob.h;
                const int y5 = y + 5 - b5 * top_blob.h;
                const int y6 = y + 6 - b6 * top_blob.h;
                const int y7 = y + 7 - b7 * top_blob.h;

                const unsigned short* ptr = bottom_blob.row<const unsigned short>(i);
                unsigned short* outptr0 = (unsigned short*)top_blob + (size_t)b0 * top_blob.nstep + (size_t)y0 * top_blob.w;
                unsigned short* outptr1 = (unsigned short*)top_blob + (size_t)b1 * top_blob.nstep + (size_t)y1 * top_blob.w;
                unsigned short* outptr2 = (unsigned short*)top_blob + (size_t)b2 * top_blob.nstep + (size_t)y2 * top_blob.w;
                unsigned short* outptr3 = (unsigned short*)top_blob + (size_t)b3 * top_blob.nstep + (size_t)y3 * top_blob.w;
                unsigned short* outptr4 = (unsigned short*)top_blob + (size_t)b4 * top_blob.nstep + (size_t)y4 * top_blob.w;
                unsigned short* outptr5 = (unsigned short*)top_blob + (size_t)b5 * top_blob.nstep + (size_t)y5 * top_blob.w;
                unsigned short* outptr6 = (unsigned short*)top_blob + (size_t)b6 * top_blob.nstep + (size_t)y6 * top_blob.w;
                unsigned short* outptr7 = (unsigned short*)top_blob + (size_t)b7 * top_blob.nstep + (size_t)y7 * top_blob.w;

                int j = 0;
                for (; j + 3 < bottom_blob.w; j += 4)
                {
                    uint16x8x4_t _v4 = vld4q_u16(ptr);
                    uint16x8_t _v_01 = vuzp1q_u16(_v4.val[0], _v4.val[1]);
                    uint16x8_t _v_23 = vuzp1q_u16(_v4.val[2], _v4.val[3]);
                    uint16x8_t _v_45 = vuzp2q_u16(_v4.val[0], _v4.val[1]);
                    uint16x8_t _v_67 = vuzp2q_u16(_v4.val[2], _v4.val[3]);
                    vst1_u16(outptr0, vget_low_u16(_v_01));
                    vst1_u16(outptr1, vget_high_u16(_v_01));
                    vst1_u16(outptr2, vget_low_u16(_v_23));
                    vst1_u16(outptr3, vget_high_u16(_v_23));
                    vst1_u16(outptr4, vget_low_u16(_v_45));
                    vst1_u16(outptr5, vget_high_u16(_v_45));
                    vst1_u16(outptr6, vget_low_u16(_v_67));
                    vst1_u16(outptr7, vget_high_u16(_v_67));

                    ptr += 32;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                    outptr4 += 4;
                    outptr5 += 4;
                    outptr6 += 4;
                    outptr7 += 4;
                }
                for (; j < bottom_blob.w; j++)
                {
                    *outptr0++ = ptr[0];
                    *outptr1++ = ptr[1];
                    *outptr2++ = ptr[2];
                    *outptr3++ = ptr[3];
                    *outptr4++ = ptr[4];
                    *outptr5++ = ptr[5];
                    *outptr6++ = ptr[6];
                    *outptr7++ = ptr[7];

                    ptr += 8;
                }
            }

            return 0;
        }
#endif // NCNN_ARM82

        if (batch_axis == 0 && elempack == 4 && out_elempack == 1 && scalar_elemsize == 2 && (dims == 3 || dims == 4) && (ndim == 3 || ndim == 4) && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == top_blob.d && bottom_blob.c * 4 == top_blob.c * batch)
        {
            const int size = top_blob.w * top_blob.h * top_blob.d;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < bottom_blob.c; q++)
            {
                const int bq = q * 4;
                const int b0 = bq / top_blob.c;
                const int b1 = (bq + 1) / top_blob.c;
                const int b2 = (bq + 2) / top_blob.c;
                const int b3 = (bq + 3) / top_blob.c;
                const int q0 = bq - b0 * top_blob.c;
                const int q1 = bq + 1 - b1 * top_blob.c;
                const int q2 = bq + 2 - b2 * top_blob.c;
                const int q3 = bq + 3 - b3 * top_blob.c;

                const unsigned short* ptr = bottom_blob.channel(q);
                unsigned short* outptr0 = (unsigned short*)top_blob + (size_t)b0 * top_blob.nstep + (size_t)q0 * top_blob.cstep;
                unsigned short* outptr1 = (unsigned short*)top_blob + (size_t)b1 * top_blob.nstep + (size_t)q1 * top_blob.cstep;
                unsigned short* outptr2 = (unsigned short*)top_blob + (size_t)b2 * top_blob.nstep + (size_t)q2 * top_blob.cstep;
                unsigned short* outptr3 = (unsigned short*)top_blob + (size_t)b3 * top_blob.nstep + (size_t)q3 * top_blob.cstep;

                int i = 0;
                for (; i + 3 < size; i += 4)
                {
                    uint16x4x4_t _v4 = vld4_u16(ptr);
                    vst1_u16(outptr0, _v4.val[0]);
                    vst1_u16(outptr1, _v4.val[1]);
                    vst1_u16(outptr2, _v4.val[2]);
                    vst1_u16(outptr3, _v4.val[3]);

                    ptr += 16;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                }
                for (; i < size; i++)
                {
                    *outptr0++ = ptr[0];
                    *outptr1++ = ptr[1];
                    *outptr2++ = ptr[2];
                    *outptr3++ = ptr[3];

                    ptr += 4;
                }
            }

            return 0;
        }

#if NCNN_ARM82
        if (batch_axis == 0 && elempack == 8 && out_elempack == 1 && scalar_elemsize == 2 && (dims == 3 || dims == 4) && (ndim == 3 || ndim == 4) && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == top_blob.d && bottom_blob.c * 8 == top_blob.c * batch)
        {
            const int size = top_blob.w * top_blob.h * top_blob.d;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < bottom_blob.c; q++)
            {
                const int bq = q * 8;
                const int b0 = bq / top_blob.c;
                const int b1 = (bq + 1) / top_blob.c;
                const int b2 = (bq + 2) / top_blob.c;
                const int b3 = (bq + 3) / top_blob.c;
                const int b4 = (bq + 4) / top_blob.c;
                const int b5 = (bq + 5) / top_blob.c;
                const int b6 = (bq + 6) / top_blob.c;
                const int b7 = (bq + 7) / top_blob.c;
                const int q0 = bq - b0 * top_blob.c;
                const int q1 = bq + 1 - b1 * top_blob.c;
                const int q2 = bq + 2 - b2 * top_blob.c;
                const int q3 = bq + 3 - b3 * top_blob.c;
                const int q4 = bq + 4 - b4 * top_blob.c;
                const int q5 = bq + 5 - b5 * top_blob.c;
                const int q6 = bq + 6 - b6 * top_blob.c;
                const int q7 = bq + 7 - b7 * top_blob.c;

                const unsigned short* ptr = bottom_blob.channel(q);
                unsigned short* outptr0 = (unsigned short*)top_blob + (size_t)b0 * top_blob.nstep + (size_t)q0 * top_blob.cstep;
                unsigned short* outptr1 = (unsigned short*)top_blob + (size_t)b1 * top_blob.nstep + (size_t)q1 * top_blob.cstep;
                unsigned short* outptr2 = (unsigned short*)top_blob + (size_t)b2 * top_blob.nstep + (size_t)q2 * top_blob.cstep;
                unsigned short* outptr3 = (unsigned short*)top_blob + (size_t)b3 * top_blob.nstep + (size_t)q3 * top_blob.cstep;
                unsigned short* outptr4 = (unsigned short*)top_blob + (size_t)b4 * top_blob.nstep + (size_t)q4 * top_blob.cstep;
                unsigned short* outptr5 = (unsigned short*)top_blob + (size_t)b5 * top_blob.nstep + (size_t)q5 * top_blob.cstep;
                unsigned short* outptr6 = (unsigned short*)top_blob + (size_t)b6 * top_blob.nstep + (size_t)q6 * top_blob.cstep;
                unsigned short* outptr7 = (unsigned short*)top_blob + (size_t)b7 * top_blob.nstep + (size_t)q7 * top_blob.cstep;

                int i = 0;
                for (; i + 3 < size; i += 4)
                {
                    uint16x8x4_t _v4 = vld4q_u16(ptr);
                    uint16x8_t _v_01 = vuzp1q_u16(_v4.val[0], _v4.val[1]);
                    uint16x8_t _v_23 = vuzp1q_u16(_v4.val[2], _v4.val[3]);
                    uint16x8_t _v_45 = vuzp2q_u16(_v4.val[0], _v4.val[1]);
                    uint16x8_t _v_67 = vuzp2q_u16(_v4.val[2], _v4.val[3]);
                    vst1_u16(outptr0, vget_low_u16(_v_01));
                    vst1_u16(outptr1, vget_high_u16(_v_01));
                    vst1_u16(outptr2, vget_low_u16(_v_23));
                    vst1_u16(outptr3, vget_high_u16(_v_23));
                    vst1_u16(outptr4, vget_low_u16(_v_45));
                    vst1_u16(outptr5, vget_high_u16(_v_45));
                    vst1_u16(outptr6, vget_low_u16(_v_67));
                    vst1_u16(outptr7, vget_high_u16(_v_67));

                    ptr += 32;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                    outptr4 += 4;
                    outptr5 += 4;
                    outptr6 += 4;
                    outptr7 += 4;
                }
                for (; i < size; i++)
                {
                    *outptr0++ = ptr[0];
                    *outptr1++ = ptr[1];
                    *outptr2++ = ptr[2];
                    *outptr3++ = ptr[3];
                    *outptr4++ = ptr[4];
                    *outptr5++ = ptr[5];
                    *outptr6++ = ptr[6];
                    *outptr7++ = ptr[7];

                    ptr += 8;
                }
            }

            return 0;
        }
#endif // NCNN_ARM82

        if (batch_axis == 1 && elempack == 4 && out_elempack == 1 && scalar_elemsize == 2 && dims == 4 && ndim == 3 && bottom_blob.w == top_blob.w && bottom_blob.h == top_blob.h && bottom_blob.d == batch && bottom_blob.c * 4 == top_blob.c)
        {
            const int total_bq = batch * bottom_blob.c;
            const int size = top_blob.w * top_blob.h;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bq = 0; bq < total_bq; bq++)
            {
                const int b = bq / bottom_blob.c;
                const int q = bq - b * bottom_blob.c;

                const unsigned short* ptr = bottom_blob.channel(q);
                ptr += (size_t)b * size * 4;
                Mat top_blob_b = top_blob.batch(b);
                unsigned short* outptr0 = top_blob_b.channel(q * 4);
                unsigned short* outptr1 = top_blob_b.channel(q * 4 + 1);
                unsigned short* outptr2 = top_blob_b.channel(q * 4 + 2);
                unsigned short* outptr3 = top_blob_b.channel(q * 4 + 3);

                int i = 0;
                for (; i + 3 < size; i += 4)
                {
                    uint16x4x4_t _v4 = vld4_u16(ptr);
                    vst1_u16(outptr0, _v4.val[0]);
                    vst1_u16(outptr1, _v4.val[1]);
                    vst1_u16(outptr2, _v4.val[2]);
                    vst1_u16(outptr3, _v4.val[3]);

                    ptr += 16;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                }
                for (; i < size; i++)
                {
                    *outptr0++ = ptr[0];
                    *outptr1++ = ptr[1];
                    *outptr2++ = ptr[2];
                    *outptr3++ = ptr[3];

                    ptr += 4;
                }
            }

            return 0;
        }
#endif // __ARM_NEON

        size_t prefix = 1;
        for (int i = 0; i < batch_axis; i++)
            prefix *= shape[i];

        size_t suffix = 1;
        if (batch_axis == 0)
            suffix = out_total;
        else
        {
            for (int i = batch_axis; i < ndim; i++)
                suffix *= shape[i];
        }

        const size_t bottom_channel_size = (size_t)bottom_blob.w * bottom_blob.h * bottom_blob.d;
        const size_t top_channel_size = (size_t)top_blob.w * top_blob.h * top_blob.d;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int pp = 0; pp < (int)prefix; pp++)
        {
            for (int b = 0; b < batch; b++)
            {
                for (size_t s = 0; s < suffix; s++)
                {
                    const size_t srci = batch_axis == 0 ? (size_t)b * suffix + s : ((size_t)pp * batch + b) * suffix + s;
                    const unsigned char* ptr = (const unsigned char*)bottom_blob;
                    if (dims == 1)
                    {
                        const int x = srci / elempack;
                        const int k = srci % elempack;
                        ptr += (size_t)x * elemsize + k * scalar_elemsize;
                    }
                    else if (dims == 2)
                    {
                        const int x = srci % bottom_blob.w;
                        const int y = srci / bottom_blob.w;
                        const int y0 = y / elempack;
                        const int k = y % elempack;
                        ptr += ((size_t)y0 * bottom_blob.w + x) * elemsize + k * scalar_elemsize;
                    }
                    else
                    {
                        const int q = srci / bottom_channel_size;
                        const size_t r = srci - (size_t)q * bottom_channel_size;
                        const int q0 = q / elempack;
                        const int k = q % elempack;
                        ptr += ((size_t)q0 * bottom_blob.cstep + r) * elemsize + k * scalar_elemsize;
                    }

                    const size_t dsti = batch_axis == 0 ? s : (size_t)pp * suffix + s;
                    unsigned char* outptr = (unsigned char*)top_blob + (size_t)b * top_blob.nstep * out_elemsize;
                    if (top_blob.dims == 1)
                    {
                        const int x = dsti / out_elempack;
                        const int k = dsti % out_elempack;
                        outptr += (size_t)x * out_elemsize + k * scalar_elemsize;
                    }
                    else if (top_blob.dims == 2)
                    {
                        const int x = dsti % top_blob.w;
                        const int y = dsti / top_blob.w;
                        const int y0 = y / out_elempack;
                        const int k = y % out_elempack;
                        outptr += ((size_t)y0 * top_blob.w + x) * out_elemsize + k * scalar_elemsize;
                    }
                    else
                    {
                        const int q = dsti / top_channel_size;
                        const size_t r = dsti - (size_t)q * top_channel_size;
                        const int q0 = q / out_elempack;
                        const int k = q % out_elempack;
                        outptr += ((size_t)q0 * top_blob.cstep + r) * out_elemsize + k * scalar_elemsize;
                    }

                    memcpy(outptr, ptr, scalar_elemsize);
                }
            }
        }

        return 0;
    }

    return -1;
}

int Reshape_arm::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    Mat& top_blob = top_blobs[0];

    if (batch_mode != 0)
        return forward_batch(bottom_blobs, top_blobs, opt);

    int elembits = bottom_blob.elembits();

#if NCNN_ARM82
    if (support_fp16_storage && opt.use_fp16_storage && elembits == 16)
        return forward_bf16s_fp16s(bottom_blobs, top_blobs, opt);
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage && elembits == 16)
        return forward_bf16s_fp16s(bottom_blobs, top_blobs, opt);
#endif

    // resolve out shape
    int outw = w;
    int outh = h;
    int outd = d;
    int outc = c;

    if (!shape_expr.empty())
    {
        int er = eval_shape_expr(bottom_blobs, outw, outh, outd, outc);
        if (er != 0)
            return -1;
    }

    if (ndim == 1)
    {
        // flatten
        flatten(bottom_blob, top_blob, opt);
        if (top_blob.empty())
            return -100;

        return 0;
    }

    const int dims = bottom_blob.dims;
    const int elempack = bottom_blob.elempack;
    const size_t elemsize = bottom_blob.elemsize;

    const int total = bottom_blob.w * bottom_blob.h * bottom_blob.d * bottom_blob.c * elempack;

    if (ndim == 2)
    {
        if (outw == 0)
            outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
        if (outh == 0)
            outh = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;

        if (outw == -1)
            outw = total / outh;
        if (outh == -1)
            outh = total / outw;

        int out_elempack = opt.use_packing_layout && outh % 4 == 0 ? 4 : 1;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if (dims == 2 && bottom_blob.h * elempack == outh && elempack == out_elempack)
        {
            top_blob = bottom_blob;
            return 0;
        }

        if (out_elempack == 1)
        {
            // flatten
            flatten(bottom_blob, top_blob, opt);
            if (top_blob.empty())
                return -100;

            top_blob.dims = 2;
            top_blob.w = outw;
            top_blob.h = outh;
            top_blob.cstep = top_blob.cstep * top_blob.elempack;
            top_blob.elemsize = out_elemsize;
            top_blob.elempack = out_elempack;

            return 0;
        }

        // flatten
        Mat bottom_blob_flattened = bottom_blob;
        {
            Option opt_flatten = opt;
            opt_flatten.blob_allocator = opt.workspace_allocator;

            flatten(bottom_blob, bottom_blob_flattened, opt_flatten);
            if (bottom_blob_flattened.empty())
                return -100;
        }

        top_blob.create(outw, outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        // assert out_elempack == 4

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < top_blob.h; i++)
        {
            const float* ptr0 = (const float*)bottom_blob_flattened + outw * i * 4;
            const float* ptr1 = (const float*)bottom_blob_flattened + outw * (i * 4 + 1);
            const float* ptr2 = (const float*)bottom_blob_flattened + outw * (i * 4 + 2);
            const float* ptr3 = (const float*)bottom_blob_flattened + outw * (i * 4 + 3);
            float* outptr = top_blob.row(i);

            int j = 0;
#if __ARM_NEON
            for (; j + 3 < outw; j += 4)
            {
                float32x4x4_t _v4;
                _v4.val[0] = vld1q_f32(ptr0);
                _v4.val[1] = vld1q_f32(ptr1);
                _v4.val[2] = vld1q_f32(ptr2);
                _v4.val[3] = vld1q_f32(ptr3);

                vst4q_f32(outptr, _v4);

                ptr0 += 4;
                ptr1 += 4;
                ptr2 += 4;
                ptr3 += 4;
                outptr += 16;
            }
#endif
            for (; j < outw; j++)
            {
                outptr[0] = *ptr0++;
                outptr[1] = *ptr1++;
                outptr[2] = *ptr2++;
                outptr[3] = *ptr3++;

                outptr += 4;
            }
        }
    }

    if (ndim == 3 || ndim == 4)
    {
        if (ndim == 3)
        {
            if (outw == 0)
                outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
            if (outh == 0)
                outh = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;
            if (outc == 0)
                outc = dims == 3 ? bottom_blob.c * elempack : bottom_blob.c;

            if (outw == -1)
                outw = total / outc / outh;
            if (outh == -1)
                outh = total / outc / outw;
            if (outc == -1)
                outc = total / outh / outw;

            outd = 1;
        }
        else // if (ndim == 4)
        {
            if (outw == 0)
                outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
            if (outh == 0)
                outh = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;
            if (outd == 0)
                outd = bottom_blob.d;
            if (outc == 0)
                outc = (dims == 3 || dims == 4) ? bottom_blob.c * elempack : bottom_blob.c;

            if (outw == -1)
                outw = total / outc / outd / outh;
            if (outh == -1)
                outh = total / outc / outd / outw;
            if (outd == -1)
                outd = total / outc / outh / outw;
            if (outc == -1)
                outc = total / outd / outh / outw;
        }

        int out_elempack = opt.use_packing_layout && outc % 4 == 0 ? 4 : 1;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if ((dims == 3 || dims == 4) && bottom_blob.c * elempack == outc && elempack == out_elempack)
        {
            top_blob = bottom_blob;
            top_blob.dims = ndim;
            top_blob.w = outw;
            top_blob.h = outh;
            top_blob.d = outd;
            return 0;
        }

        // flatten
        Mat bottom_blob_flattened = bottom_blob;
        {
            Option opt_flatten = opt;
            opt_flatten.blob_allocator = opt.workspace_allocator;

            flatten(bottom_blob, bottom_blob_flattened, opt_flatten);
            if (bottom_blob_flattened.empty())
                return -100;
        }

        if (ndim == 3)
        {
            top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        }
        else // if (ndim == 4)
        {
            top_blob.create(outw, outh, outd, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        }
        if (top_blob.empty())
            return -100;

        int size = top_blob.w * top_blob.h * top_blob.d;

        if (out_elempack == 4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const float* ptr0 = (const float*)bottom_blob_flattened + size * q * 4;
                const float* ptr1 = (const float*)bottom_blob_flattened + size * (q * 4 + 1);
                const float* ptr2 = (const float*)bottom_blob_flattened + size * (q * 4 + 2);
                const float* ptr3 = (const float*)bottom_blob_flattened + size * (q * 4 + 3);
                float* outptr = top_blob.channel(q);

                int i = 0;
#if __ARM_NEON
                for (; i + 3 < size; i += 4)
                {
                    float32x4x4_t _v4;
                    _v4.val[0] = vld1q_f32(ptr0);
                    _v4.val[1] = vld1q_f32(ptr1);
                    _v4.val[2] = vld1q_f32(ptr2);
                    _v4.val[3] = vld1q_f32(ptr3);

                    vst4q_f32(outptr, _v4);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    ptr3 += 4;
                    outptr += 16;
                }
#endif
                for (; i < size; i++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;

                    outptr += 4;
                }
            }
        }

        if (out_elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const float* ptr = (const float*)bottom_blob_flattened + size * q;
                float* outptr = top_blob.channel(q);

                int i = 0;
#if __ARM_NEON
                for (; i + 3 < size; i += 4)
                {
                    float32x4_t _v = vld1q_f32(ptr);
                    vst1q_f32(outptr, _v);
                    ptr += 4;
                    outptr += 4;
                }
#endif
                for (; i < size; i++)
                {
                    *outptr++ = *ptr++;
                }
            }
        }
    }

    return 0;
}

int Reshape_arm::forward_bf16s_fp16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    if (batch_mode != 0)
        return forward_batch(bottom_blobs, top_blobs, opt);

    const Mat& bottom_blob = bottom_blobs[0];
    Mat& top_blob = top_blobs[0];

    // resolve out shape
    int outw = w;
    int outh = h;
    int outd = d;
    int outc = c;

    if (!shape_expr.empty())
    {
        int er = eval_shape_expr(bottom_blobs, outw, outh, outd, outc);
        if (er != 0)
            return -1;
    }

    if (ndim == 1)
    {
        // flatten
        flatten(bottom_blob, top_blob, opt);
        if (top_blob.empty())
            return -100;

        return 0;
    }

    const int dims = bottom_blob.dims;
    const int elempack = bottom_blob.elempack;
    const size_t elemsize = bottom_blob.elemsize;

    const int total = bottom_blob.w * bottom_blob.h * bottom_blob.d * bottom_blob.c * elempack;

    if (ndim == 2)
    {
        if (outw == 0)
            outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
        if (outh == 0)
            outh = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;

        if (outw == -1)
            outw = total / outh;
        if (outh == -1)
            outh = total / outw;

        int out_elempack = 1;
        if (opt.use_packing_layout)
        {
#if NCNN_ARM82
            out_elempack = support_fp16_storage && opt.use_fp16_arithmetic && outh % 8 == 0 ? 8 : outh % 4 == 0 ? 4 : 1;
#else
            out_elempack = outh % 4 == 0 ? 4 : 1;
#endif
        }
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if (dims == 2 && bottom_blob.h * elempack == outh && elempack == out_elempack)
        {
            top_blob = bottom_blob;
            return 0;
        }

        if (out_elempack == 1)
        {
            // flatten
            flatten(bottom_blob, top_blob, opt);
            if (top_blob.empty())
                return -100;

            top_blob.dims = 2;
            top_blob.w = outw;
            top_blob.h = outh;
            top_blob.cstep = top_blob.cstep * top_blob.elempack;
            top_blob.elemsize = out_elemsize;
            top_blob.elempack = out_elempack;

            return 0;
        }

        // flatten
        Mat bottom_blob_flattened = bottom_blob;
        {
            Option opt_flatten = opt;
            opt_flatten.blob_allocator = opt.workspace_allocator;

            flatten(bottom_blob, bottom_blob_flattened, opt_flatten);
            if (bottom_blob_flattened.empty())
                return -100;
        }

        top_blob.create(outw, outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#if NCNN_ARM82
        if (out_elempack == 8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < top_blob.h; i++)
            {
                const unsigned short* ptr0 = (const unsigned short*)bottom_blob_flattened + outw * i * 8;
                const unsigned short* ptr1 = (const unsigned short*)bottom_blob_flattened + outw * (i * 8 + 1);
                const unsigned short* ptr2 = (const unsigned short*)bottom_blob_flattened + outw * (i * 8 + 2);
                const unsigned short* ptr3 = (const unsigned short*)bottom_blob_flattened + outw * (i * 8 + 3);
                const unsigned short* ptr4 = (const unsigned short*)bottom_blob_flattened + outw * (i * 8 + 4);
                const unsigned short* ptr5 = (const unsigned short*)bottom_blob_flattened + outw * (i * 8 + 5);
                const unsigned short* ptr6 = (const unsigned short*)bottom_blob_flattened + outw * (i * 8 + 6);
                const unsigned short* ptr7 = (const unsigned short*)bottom_blob_flattened + outw * (i * 8 + 7);
                unsigned short* outptr = top_blob.row<unsigned short>(i);

                int j = 0;
                for (; j + 3 < outw; j += 4)
                {
                    uint16x8_t _p01 = vcombine_u16(vld1_u16(ptr0), vld1_u16(ptr1));
                    uint16x8_t _p23 = vcombine_u16(vld1_u16(ptr2), vld1_u16(ptr3));
                    uint16x8_t _p45 = vcombine_u16(vld1_u16(ptr4), vld1_u16(ptr5));
                    uint16x8_t _p67 = vcombine_u16(vld1_u16(ptr6), vld1_u16(ptr7));

                    uint16x8x2_t _p0415 = vzipq_u16(_p01, _p45);
                    uint16x8x2_t _p2637 = vzipq_u16(_p23, _p67);

                    uint16x8x4_t _v4;
                    _v4.val[0] = _p0415.val[0];
                    _v4.val[1] = _p0415.val[1];
                    _v4.val[2] = _p2637.val[0];
                    _v4.val[3] = _p2637.val[1];

                    vst4q_u16(outptr, _v4);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    ptr3 += 4;
                    ptr4 += 4;
                    ptr5 += 4;
                    ptr6 += 4;
                    ptr7 += 4;
                    outptr += 32;
                }
                for (; j < outw; j++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;
                    outptr[4] = *ptr4++;
                    outptr[5] = *ptr5++;
                    outptr[6] = *ptr6++;
                    outptr[7] = *ptr7++;

                    outptr += 8;
                }
            }
        }
#endif // NCNN_ARM82

        if (out_elempack == 4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < top_blob.h; i++)
            {
                const unsigned short* ptr0 = (const unsigned short*)bottom_blob_flattened + outw * i * 4;
                const unsigned short* ptr1 = (const unsigned short*)bottom_blob_flattened + outw * (i * 4 + 1);
                const unsigned short* ptr2 = (const unsigned short*)bottom_blob_flattened + outw * (i * 4 + 2);
                const unsigned short* ptr3 = (const unsigned short*)bottom_blob_flattened + outw * (i * 4 + 3);
                unsigned short* outptr = top_blob.row<unsigned short>(i);

                int j = 0;
#if __ARM_NEON
                for (; j + 3 < outw; j += 4)
                {
                    uint16x4x4_t _v4;
                    _v4.val[0] = vld1_u16(ptr0);
                    _v4.val[1] = vld1_u16(ptr1);
                    _v4.val[2] = vld1_u16(ptr2);
                    _v4.val[3] = vld1_u16(ptr3);

                    vst4_u16(outptr, _v4);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    ptr3 += 4;
                    outptr += 16;
                }
#endif
                for (; j < outw; j++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;

                    outptr += 4;
                }
            }
        }
    }

    if (ndim == 3 || ndim == 4)
    {
        if (ndim == 3)
        {
            if (outw == 0)
                outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
            if (outh == 0)
                outh = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;
            if (outc == 0)
                outc = dims == 3 ? bottom_blob.c * elempack : bottom_blob.c;

            if (outw == -1)
                outw = total / outc / outh;
            if (outh == -1)
                outh = total / outc / outw;
            if (outc == -1)
                outc = total / outh / outw;

            outd = 1;
        }
        else // if (ndim == 4)
        {
            if (outw == 0)
                outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
            if (outh == 0)
                outh = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;
            if (outd == 0)
                outd = bottom_blob.d;
            if (outc == 0)
                outc = (dims == 3 || dims == 4) ? bottom_blob.c * elempack : bottom_blob.c;

            if (outw == -1)
                outw = total / outc / outd / outh;
            if (outh == -1)
                outh = total / outc / outd / outw;
            if (outd == -1)
                outd = total / outc / outh / outw;
            if (outc == -1)
                outc = total / outd / outh / outw;
        }

        int out_elempack = 1;
        if (opt.use_packing_layout)
        {
#if NCNN_ARM82
            out_elempack = support_fp16_storage && opt.use_fp16_arithmetic && outc % 8 == 0 ? 8 : outc % 4 == 0 ? 4 : 1;
#else
            out_elempack = outc % 4 == 0 ? 4 : 1;
#endif
        }
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if ((dims == 3 || dims == 4) && bottom_blob.c * elempack == outc && elempack == out_elempack)
        {
            top_blob = bottom_blob;
            top_blob.dims = ndim;
            top_blob.w = outw;
            top_blob.h = outh;
            top_blob.d = outd;
            return 0;
        }

        // flatten
        Mat bottom_blob_flattened = bottom_blob;
        {
            Option opt_flatten = opt;
            opt_flatten.blob_allocator = opt.workspace_allocator;

            flatten(bottom_blob, bottom_blob_flattened, opt_flatten);
            if (bottom_blob_flattened.empty())
                return -100;
        }

        if (ndim == 3)
        {
            top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        }
        else // if (ndim == 4)
        {
            top_blob.create(outw, outh, outd, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        }
        if (top_blob.empty())
            return -100;

        int size = top_blob.w * top_blob.h * top_blob.d;

#if NCNN_ARM82
        if (out_elempack == 8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const unsigned short* ptr0 = (const unsigned short*)bottom_blob_flattened + size * q * 8;
                const unsigned short* ptr1 = (const unsigned short*)bottom_blob_flattened + size * (q * 8 + 1);
                const unsigned short* ptr2 = (const unsigned short*)bottom_blob_flattened + size * (q * 8 + 2);
                const unsigned short* ptr3 = (const unsigned short*)bottom_blob_flattened + size * (q * 8 + 3);
                const unsigned short* ptr4 = (const unsigned short*)bottom_blob_flattened + size * (q * 8 + 4);
                const unsigned short* ptr5 = (const unsigned short*)bottom_blob_flattened + size * (q * 8 + 5);
                const unsigned short* ptr6 = (const unsigned short*)bottom_blob_flattened + size * (q * 8 + 6);
                const unsigned short* ptr7 = (const unsigned short*)bottom_blob_flattened + size * (q * 8 + 7);
                unsigned short* outptr = top_blob.channel(q);

                int i = 0;
                for (; i + 3 < size; i += 4)
                {
                    uint16x8_t _p01 = vcombine_u16(vld1_u16(ptr0), vld1_u16(ptr1));
                    uint16x8_t _p23 = vcombine_u16(vld1_u16(ptr2), vld1_u16(ptr3));
                    uint16x8_t _p45 = vcombine_u16(vld1_u16(ptr4), vld1_u16(ptr5));
                    uint16x8_t _p67 = vcombine_u16(vld1_u16(ptr6), vld1_u16(ptr7));

                    uint16x8x2_t _p0415 = vzipq_u16(_p01, _p45);
                    uint16x8x2_t _p2637 = vzipq_u16(_p23, _p67);

                    uint16x8x4_t _v4;
                    _v4.val[0] = _p0415.val[0];
                    _v4.val[1] = _p0415.val[1];
                    _v4.val[2] = _p2637.val[0];
                    _v4.val[3] = _p2637.val[1];

                    vst4q_u16(outptr, _v4);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    ptr3 += 4;
                    ptr4 += 4;
                    ptr5 += 4;
                    ptr6 += 4;
                    ptr7 += 4;
                    outptr += 32;
                }
                for (; i < size; i++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;
                    outptr[4] = *ptr4++;
                    outptr[5] = *ptr5++;
                    outptr[6] = *ptr6++;
                    outptr[7] = *ptr7++;

                    outptr += 8;
                }
            }
        }
#endif // NCNN_ARM82

        if (out_elempack == 4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const unsigned short* ptr0 = (const unsigned short*)bottom_blob_flattened + size * q * 4;
                const unsigned short* ptr1 = (const unsigned short*)bottom_blob_flattened + size * (q * 4 + 1);
                const unsigned short* ptr2 = (const unsigned short*)bottom_blob_flattened + size * (q * 4 + 2);
                const unsigned short* ptr3 = (const unsigned short*)bottom_blob_flattened + size * (q * 4 + 3);
                unsigned short* outptr = top_blob.channel(q);

                int i = 0;
#if __ARM_NEON
                for (; i + 3 < size; i += 4)
                {
                    uint16x4x4_t _v4;
                    _v4.val[0] = vld1_u16(ptr0);
                    _v4.val[1] = vld1_u16(ptr1);
                    _v4.val[2] = vld1_u16(ptr2);
                    _v4.val[3] = vld1_u16(ptr3);

                    vst4_u16(outptr, _v4);

                    ptr0 += 4;
                    ptr1 += 4;
                    ptr2 += 4;
                    ptr3 += 4;
                    outptr += 16;
                }
#endif
                for (; i < size; i++)
                {
                    outptr[0] = *ptr0++;
                    outptr[1] = *ptr1++;
                    outptr[2] = *ptr2++;
                    outptr[3] = *ptr3++;

                    outptr += 4;
                }
            }
        }

        if (out_elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < top_blob.c; q++)
            {
                const unsigned short* ptr = (const unsigned short*)bottom_blob_flattened + size * q;
                unsigned short* outptr = top_blob.channel(q);

                int i = 0;
#if __ARM_NEON
                for (; i + 3 < size; i += 4)
                {
                    uint16x4_t _v = vld1_u16(ptr);
                    vst1_u16(outptr, _v);
                    ptr += 4;
                    outptr += 4;
                }
#endif
                for (; i < size; i++)
                {
                    *outptr++ = *ptr++;
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
