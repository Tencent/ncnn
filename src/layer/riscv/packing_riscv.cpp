// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "packing_riscv.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

#include "riscv_usability.h"

#include "cpu.h"

namespace ncnn {

Packing_riscv::Packing_riscv()
{
    support_packing = true;
#if NCNN_ZFH
#if __riscv_vector
    support_fp16_storage = cpu_support_riscv_zvfh();
#else
    support_fp16_storage = cpu_support_riscv_zfh();
#endif
#endif
    support_bf16_storage = true;
}

int Packing_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int elembits = bottom_blob.elembits();

    if (elembits == 8)
        return forward_int8(bottom_blob, top_blob, opt);

#if NCNN_ZFH
    if (opt.use_fp16_storage && elembits == 16)
        return forward_bf16s_fp16s(bottom_blob, top_blob, opt);
#endif

    if (opt.use_bf16_storage && elembits == 16)
        return forward_bf16s_fp16s(bottom_blob, top_blob, opt);

    if (use_padding)
    {
        return Packing::forward(bottom_blob, top_blob, opt);
    }

    if (elembits != 32)
    {
        // non-fp32 type
        return Packing::forward(bottom_blob, top_blob, opt);
    }

    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    if (elempack == out_elempack)
    {
        top_blob = bottom_blob;
        return 0;
    }

    bool pack1to4 = elempack == 1 && out_elempack == 4;
    bool pack4to1 = elempack == 4 && out_elempack == 1;
    bool pack1to8 = elempack == 1 && out_elempack == 8;
    bool pack8to1 = elempack == 8 && out_elempack == 1;
    bool pack4to8 = elempack == 4 && out_elempack == 8;
    bool pack8to4 = elempack == 8 && out_elempack == 4;

    if (!pack1to4 && !pack4to1 && !pack1to8 && !pack8to1 && !pack4to8 && !pack8to4)
    {
        return Packing::forward(bottom_blob, top_blob, opt);
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;

    if (!use_padding)
    {
        // identity if use_padding not allowed
        if (dims == 1 && w * elempack % out_elempack != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
        if (dims == 2 && h * elempack % out_elempack != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
        if ((dims == 3 || dims == 4) && channels * elempack % out_elempack != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
    }

    if (dims == 1)
    {
        top_blob = bottom_blob;
        top_blob.w = w * elempack / out_elempack;
        top_blob.cstep = bottom_blob.cstep * elempack / out_elempack;
        top_blob.elemsize = elemsize / elempack * out_elempack;
        top_blob.elempack = out_elempack;
        return 0;
    }

    if (dims == 2)
    {
        int outh = h * elempack / out_elempack;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        top_blob.create(w, outh, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (pack1to4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < outh; i++)
            {
                const float* r0 = bottom_blob.row(i * 4);
                const float* r1 = bottom_blob.row(i * 4 + 1);
                const float* r2 = bottom_blob.row(i * 4 + 2);
                const float* r3 = bottom_blob.row(i * 4 + 3);

                float* outptr = top_blob.row(i);

#if __riscv_vector
                int n = w;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m2(n);

                    vfloat32m2_t _p0 = __riscv_vle32_v_f32m2(r0, vl);
                    vfloat32m2_t _p1 = __riscv_vle32_v_f32m2(r1, vl);
                    vfloat32m2_t _p2 = __riscv_vle32_v_f32m2(r2, vl);
                    vfloat32m2_t _p3 = __riscv_vle32_v_f32m2(r3, vl);
                    __riscv_vsseg4e32_v_f32m2x4(outptr, __riscv_vcreate_v_f32m2x4(_p0, _p1, _p2, _p3), vl);

                    r0 += vl;
                    r1 += vl;
                    r2 += vl;
                    r3 += vl;
                    outptr += vl * 4;
                    n -= vl;
                }
#else  // __riscv_vector
                for (int j = 0; j < w; j++)
                {
                    outptr[0] = *r0++;
                    outptr[1] = *r1++;
                    outptr[2] = *r2++;
                    outptr[3] = *r3++;

                    outptr += 4;
                }
#endif // __riscv_vector
            }
        }
        if (pack4to1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const float* r0 = bottom_blob.row(i);

                float* outptr0 = top_blob.row(i * 4);
                float* outptr1 = top_blob.row(i * 4 + 1);
                float* outptr2 = top_blob.row(i * 4 + 2);
                float* outptr3 = top_blob.row(i * 4 + 3);

#if __riscv_vector
                int n = w;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m2(n);

                    vfloat32m2x4_t _p = __riscv_vlseg4e32_v_f32m2x4(r0, vl);

                    __riscv_vse32_v_f32m2(outptr0, __riscv_vget_v_f32m2x4_f32m2(_p, 0), vl);
                    __riscv_vse32_v_f32m2(outptr1, __riscv_vget_v_f32m2x4_f32m2(_p, 1), vl);
                    __riscv_vse32_v_f32m2(outptr2, __riscv_vget_v_f32m2x4_f32m2(_p, 2), vl);
                    __riscv_vse32_v_f32m2(outptr3, __riscv_vget_v_f32m2x4_f32m2(_p, 3), vl);

                    r0 += vl * 4;
                    outptr0 += vl;
                    outptr1 += vl;
                    outptr2 += vl;
                    outptr3 += vl;
                    n -= vl;
                }
#else  // __riscv_vector
                for (int j = 0; j < w; j++)
                {
                    *outptr0++ = r0[0];
                    *outptr1++ = r0[1];
                    *outptr2++ = r0[2];
                    *outptr3++ = r0[3];

                    r0 += 4;
                }
#endif // __riscv_vector
            }
        }
        if (pack1to8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < outh; i++)
            {
                const float* r0 = bottom_blob.row(i * 8);
                const float* r1 = bottom_blob.row(i * 8 + 1);
                const float* r2 = bottom_blob.row(i * 8 + 2);
                const float* r3 = bottom_blob.row(i * 8 + 3);
                const float* r4 = bottom_blob.row(i * 8 + 4);
                const float* r5 = bottom_blob.row(i * 8 + 5);
                const float* r6 = bottom_blob.row(i * 8 + 6);
                const float* r7 = bottom_blob.row(i * 8 + 7);

                float* outptr = top_blob.row(i);

#if __riscv_vector
                int n = w;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m1(n);

                    vfloat32m1_t _p0 = __riscv_vle32_v_f32m1(r0, vl);
                    vfloat32m1_t _p1 = __riscv_vle32_v_f32m1(r1, vl);
                    vfloat32m1_t _p2 = __riscv_vle32_v_f32m1(r2, vl);
                    vfloat32m1_t _p3 = __riscv_vle32_v_f32m1(r3, vl);
                    vfloat32m1_t _p4 = __riscv_vle32_v_f32m1(r4, vl);
                    vfloat32m1_t _p5 = __riscv_vle32_v_f32m1(r5, vl);
                    vfloat32m1_t _p6 = __riscv_vle32_v_f32m1(r6, vl);
                    vfloat32m1_t _p7 = __riscv_vle32_v_f32m1(r7, vl);
                    __riscv_vsseg8e32_v_f32m1x8(outptr, __riscv_vcreate_v_f32m1x8(_p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7), vl);

                    r0 += vl;
                    r1 += vl;
                    r2 += vl;
                    r3 += vl;
                    r4 += vl;
                    r5 += vl;
                    r6 += vl;
                    r7 += vl;
                    outptr += vl * 8;
                    n -= vl;
                }
#else  // __riscv_vector
                for (int j = 0; j < w; j++)
                {
                    outptr[0] = *r0++;
                    outptr[1] = *r1++;
                    outptr[2] = *r2++;
                    outptr[3] = *r3++;
                    outptr[4] = *r4++;
                    outptr[5] = *r5++;
                    outptr[6] = *r6++;
                    outptr[7] = *r7++;

                    outptr += 8;
                }
#endif // __riscv_vector
            }
        }
        if (pack8to1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const float* r0 = bottom_blob.row(i);

                float* outptr0 = top_blob.row(i * 8);
                float* outptr1 = top_blob.row(i * 8 + 1);
                float* outptr2 = top_blob.row(i * 8 + 2);
                float* outptr3 = top_blob.row(i * 8 + 3);
                float* outptr4 = top_blob.row(i * 8 + 4);
                float* outptr5 = top_blob.row(i * 8 + 5);
                float* outptr6 = top_blob.row(i * 8 + 6);
                float* outptr7 = top_blob.row(i * 8 + 7);

#if __riscv_vector
                int n = w;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m1(n);

                    vfloat32m1x8_t _p = __riscv_vlseg8e32_v_f32m1x8(r0, vl);
                    __riscv_vse32_v_f32m1(outptr0, __riscv_vget_v_f32m1x8_f32m1(_p, 0), vl);
                    __riscv_vse32_v_f32m1(outptr1, __riscv_vget_v_f32m1x8_f32m1(_p, 1), vl);
                    __riscv_vse32_v_f32m1(outptr2, __riscv_vget_v_f32m1x8_f32m1(_p, 2), vl);
                    __riscv_vse32_v_f32m1(outptr3, __riscv_vget_v_f32m1x8_f32m1(_p, 3), vl);
                    __riscv_vse32_v_f32m1(outptr4, __riscv_vget_v_f32m1x8_f32m1(_p, 4), vl);
                    __riscv_vse32_v_f32m1(outptr5, __riscv_vget_v_f32m1x8_f32m1(_p, 5), vl);
                    __riscv_vse32_v_f32m1(outptr6, __riscv_vget_v_f32m1x8_f32m1(_p, 6), vl);
                    __riscv_vse32_v_f32m1(outptr7, __riscv_vget_v_f32m1x8_f32m1(_p, 7), vl);

                    r0 += vl * 8;
                    outptr0 += vl;
                    outptr1 += vl;
                    outptr2 += vl;
                    outptr3 += vl;
                    outptr4 += vl;
                    outptr5 += vl;
                    outptr6 += vl;
                    outptr7 += vl;
                    n -= vl;
                }
#else  // __riscv_vector
                for (int j = 0; j < w; j++)
                {
                    *outptr0++ = r0[0];
                    *outptr1++ = r0[1];
                    *outptr2++ = r0[2];
                    *outptr3++ = r0[3];
                    *outptr4++ = r0[4];
                    *outptr5++ = r0[5];
                    *outptr6++ = r0[6];
                    *outptr7++ = r0[7];

                    r0 += 8;
                }
#endif // __riscv_vector
            }
        }
        if (pack4to8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < outh; i++)
            {
                const float* r0 = bottom_blob.row(i * 2);
                const float* r1 = bottom_blob.row(i * 2 + 1);

                float* outptr = top_blob.row(i);

#if __riscv_vector
                int n = w;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m1(n);

                    vfloat32m1x4_t _p0 = __riscv_vlseg4e32_v_f32m1x4(r0, vl);
                    vfloat32m1x4_t _p1 = __riscv_vlseg4e32_v_f32m1x4(r1, vl);

                    vfloat32m1_t _p00 = __riscv_vget_v_f32m1x4_f32m1(_p0, 0);
                    vfloat32m1_t _p01 = __riscv_vget_v_f32m1x4_f32m1(_p0, 1);
                    vfloat32m1_t _p02 = __riscv_vget_v_f32m1x4_f32m1(_p0, 2);
                    vfloat32m1_t _p03 = __riscv_vget_v_f32m1x4_f32m1(_p0, 3);
                    vfloat32m1_t _p10 = __riscv_vget_v_f32m1x4_f32m1(_p1, 0);
                    vfloat32m1_t _p11 = __riscv_vget_v_f32m1x4_f32m1(_p1, 1);
                    vfloat32m1_t _p12 = __riscv_vget_v_f32m1x4_f32m1(_p1, 2);
                    vfloat32m1_t _p13 = __riscv_vget_v_f32m1x4_f32m1(_p1, 3);

                    __riscv_vsseg8e32_v_f32m1x8(outptr, __riscv_vcreate_v_f32m1x8(_p00, _p01, _p02, _p03, _p10, _p11, _p12, _p13), vl);

                    r0 += vl * 4;
                    r1 += vl * 4;
                    outptr += vl * 8;
                    n -= vl;
                }
#else  // __riscv_vector
                for (int j = 0; j < w; j++)
                {
                    outptr[0] = r0[0];
                    outptr[1] = r0[1];
                    outptr[2] = r0[2];
                    outptr[3] = r0[3];
                    outptr[4] = r1[0];
                    outptr[5] = r1[1];
                    outptr[6] = r1[2];
                    outptr[7] = r1[3];

                    r0 += 4;
                    r1 += 4;
                    outptr += 8;
                }
#endif // __riscv_vector
            }
        }
        if (pack8to4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const float* r0 = bottom_blob.row(i);

                float* outptr0 = top_blob.row(i * 2);
                float* outptr1 = top_blob.row(i * 2 + 1);

#if __riscv_vector
                int n = w;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m1(n);

                    vfloat32m1x8_t _p = __riscv_vlseg8e32_v_f32m1x8(r0, vl);

                    vfloat32m1_t _p0 = __riscv_vget_v_f32m1x8_f32m1(_p, 0);
                    vfloat32m1_t _p1 = __riscv_vget_v_f32m1x8_f32m1(_p, 1);
                    vfloat32m1_t _p2 = __riscv_vget_v_f32m1x8_f32m1(_p, 2);
                    vfloat32m1_t _p3 = __riscv_vget_v_f32m1x8_f32m1(_p, 3);
                    vfloat32m1_t _p4 = __riscv_vget_v_f32m1x8_f32m1(_p, 4);
                    vfloat32m1_t _p5 = __riscv_vget_v_f32m1x8_f32m1(_p, 5);
                    vfloat32m1_t _p6 = __riscv_vget_v_f32m1x8_f32m1(_p, 6);
                    vfloat32m1_t _p7 = __riscv_vget_v_f32m1x8_f32m1(_p, 7);

                    __riscv_vsseg4e32_v_f32m1x4(outptr0, __riscv_vcreate_v_f32m1x4(_p0, _p1, _p2, _p3), vl);
                    __riscv_vsseg4e32_v_f32m1x4(outptr1, __riscv_vcreate_v_f32m1x4(_p4, _p5, _p6, _p7), vl);

                    r0 += vl * 8;
                    outptr0 += vl * 4;
                    outptr1 += vl * 4;
                    n -= vl;
                }
#else  // __riscv_vector
                for (int j = 0; j < w; j++)
                {
                    outptr0[0] = r0[0];
                    outptr0[1] = r0[1];
                    outptr0[2] = r0[2];
                    outptr0[3] = r0[3];
                    outptr1[0] = r0[4];
                    outptr1[1] = r0[5];
                    outptr1[2] = r0[6];
                    outptr1[3] = r0[7];

                    r0 += 8;
                    outptr0 += 4;
                    outptr1 += 4;
                }
#endif // __riscv_vector
            }
        }

        return 0;
    }

    if (dims == 3 || dims == 4)
    {
        int size = w * h * d;
        int outc = channels * elempack / out_elempack;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if (dims == 3)
            top_blob.create(w, h, outc, out_elemsize, out_elempack, opt.blob_allocator);
        else // if (dims == 4)
            top_blob.create(w, h, d, outc, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (pack1to4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const float* r0 = bottom_blob.channel(q * 4);
                const float* r1 = bottom_blob.channel(q * 4 + 1);
                const float* r2 = bottom_blob.channel(q * 4 + 2);
                const float* r3 = bottom_blob.channel(q * 4 + 3);

                float* outptr = top_blob.channel(q);

#if __riscv_vector
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m2(n);

                    vfloat32m2_t _p0 = __riscv_vle32_v_f32m2(r0, vl);
                    vfloat32m2_t _p1 = __riscv_vle32_v_f32m2(r1, vl);
                    vfloat32m2_t _p2 = __riscv_vle32_v_f32m2(r2, vl);
                    vfloat32m2_t _p3 = __riscv_vle32_v_f32m2(r3, vl);

                    __riscv_vsseg4e32_v_f32m2x4(outptr, __riscv_vcreate_v_f32m2x4(_p0, _p1, _p2, _p3), vl);

                    r0 += vl;
                    r1 += vl;
                    r2 += vl;
                    r3 += vl;
                    outptr += vl * 4;
                    n -= vl;
                }
#else  // __riscv_vector
                for (int i = 0; i < size; i++)
                {
                    outptr[0] = *r0++;
                    outptr[1] = *r1++;
                    outptr[2] = *r2++;
                    outptr[3] = *r3++;

                    outptr += 4;
                }
#endif // __riscv_vector
            }
        }
        if (pack4to1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* r0 = bottom_blob.channel(q);

                float* outptr0 = top_blob.channel(q * 4);
                float* outptr1 = top_blob.channel(q * 4 + 1);
                float* outptr2 = top_blob.channel(q * 4 + 2);
                float* outptr3 = top_blob.channel(q * 4 + 3);

#if __riscv_vector
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m2(n);

                    vfloat32m2x4_t _p = __riscv_vlseg4e32_v_f32m2x4(r0, vl);
                    __riscv_vse32_v_f32m2(outptr0, __riscv_vget_v_f32m2x4_f32m2(_p, 0), vl);
                    __riscv_vse32_v_f32m2(outptr1, __riscv_vget_v_f32m2x4_f32m2(_p, 1), vl);
                    __riscv_vse32_v_f32m2(outptr2, __riscv_vget_v_f32m2x4_f32m2(_p, 2), vl);
                    __riscv_vse32_v_f32m2(outptr3, __riscv_vget_v_f32m2x4_f32m2(_p, 3), vl);

                    r0 += vl * 4;
                    outptr0 += vl;
                    outptr1 += vl;
                    outptr2 += vl;
                    outptr3 += vl;
                    n -= vl;
                }
#else  // __riscv_vector
                for (int i = 0; i < size; i++)
                {
                    *outptr0++ = r0[0];
                    *outptr1++ = r0[1];
                    *outptr2++ = r0[2];
                    *outptr3++ = r0[3];

                    r0 += 4;
                }
#endif // __riscv_vector
            }
        }
        if (pack1to8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const float* r0 = bottom_blob.channel(q * 8);
                const float* r1 = bottom_blob.channel(q * 8 + 1);
                const float* r2 = bottom_blob.channel(q * 8 + 2);
                const float* r3 = bottom_blob.channel(q * 8 + 3);
                const float* r4 = bottom_blob.channel(q * 8 + 4);
                const float* r5 = bottom_blob.channel(q * 8 + 5);
                const float* r6 = bottom_blob.channel(q * 8 + 6);
                const float* r7 = bottom_blob.channel(q * 8 + 7);

                float* outptr = top_blob.channel(q);

#if __riscv_vector
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m1(n);

                    vfloat32m1_t _p0 = __riscv_vle32_v_f32m1(r0, vl);
                    vfloat32m1_t _p1 = __riscv_vle32_v_f32m1(r1, vl);
                    vfloat32m1_t _p2 = __riscv_vle32_v_f32m1(r2, vl);
                    vfloat32m1_t _p3 = __riscv_vle32_v_f32m1(r3, vl);
                    vfloat32m1_t _p4 = __riscv_vle32_v_f32m1(r4, vl);
                    vfloat32m1_t _p5 = __riscv_vle32_v_f32m1(r5, vl);
                    vfloat32m1_t _p6 = __riscv_vle32_v_f32m1(r6, vl);
                    vfloat32m1_t _p7 = __riscv_vle32_v_f32m1(r7, vl);
                    __riscv_vsseg8e32_v_f32m1x8(outptr, __riscv_vcreate_v_f32m1x8(_p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7), vl);

                    r0 += vl;
                    r1 += vl;
                    r2 += vl;
                    r3 += vl;
                    r4 += vl;
                    r5 += vl;
                    r6 += vl;
                    r7 += vl;
                    outptr += vl * 8;
                    n -= vl;
                }
#else  // __riscv_vector
                for (int i = 0; i < size; i++)
                {
                    outptr[0] = *r0++;
                    outptr[1] = *r1++;
                    outptr[2] = *r2++;
                    outptr[3] = *r3++;
                    outptr[4] = *r4++;
                    outptr[5] = *r5++;
                    outptr[6] = *r6++;
                    outptr[7] = *r7++;

                    outptr += 8;
                }
#endif // __riscv_vector
            }
        }
        if (pack8to1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* r0 = bottom_blob.channel(q);

                float* outptr0 = top_blob.channel(q * 8);
                float* outptr1 = top_blob.channel(q * 8 + 1);
                float* outptr2 = top_blob.channel(q * 8 + 2);
                float* outptr3 = top_blob.channel(q * 8 + 3);
                float* outptr4 = top_blob.channel(q * 8 + 4);
                float* outptr5 = top_blob.channel(q * 8 + 5);
                float* outptr6 = top_blob.channel(q * 8 + 6);
                float* outptr7 = top_blob.channel(q * 8 + 7);

#if __riscv_vector
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m1(n);

                    vfloat32m1x8_t _p = __riscv_vlseg8e32_v_f32m1x8(r0, vl);
                    __riscv_vse32_v_f32m1(outptr0, __riscv_vget_v_f32m1x8_f32m1(_p, 0), vl);
                    __riscv_vse32_v_f32m1(outptr1, __riscv_vget_v_f32m1x8_f32m1(_p, 1), vl);
                    __riscv_vse32_v_f32m1(outptr2, __riscv_vget_v_f32m1x8_f32m1(_p, 2), vl);
                    __riscv_vse32_v_f32m1(outptr3, __riscv_vget_v_f32m1x8_f32m1(_p, 3), vl);
                    __riscv_vse32_v_f32m1(outptr4, __riscv_vget_v_f32m1x8_f32m1(_p, 4), vl);
                    __riscv_vse32_v_f32m1(outptr5, __riscv_vget_v_f32m1x8_f32m1(_p, 5), vl);
                    __riscv_vse32_v_f32m1(outptr6, __riscv_vget_v_f32m1x8_f32m1(_p, 6), vl);
                    __riscv_vse32_v_f32m1(outptr7, __riscv_vget_v_f32m1x8_f32m1(_p, 7), vl);

                    r0 += vl * 8;
                    outptr0 += vl;
                    outptr1 += vl;
                    outptr2 += vl;
                    outptr3 += vl;
                    outptr4 += vl;
                    outptr5 += vl;
                    outptr6 += vl;
                    outptr7 += vl;
                    n -= vl;
                }
#else  // __riscv_vector
                for (int i = 0; i < size; i++)
                {
                    *outptr0++ = r0[0];
                    *outptr1++ = r0[1];
                    *outptr2++ = r0[2];
                    *outptr3++ = r0[3];
                    *outptr4++ = r0[4];
                    *outptr5++ = r0[5];
                    *outptr6++ = r0[6];
                    *outptr7++ = r0[7];

                    r0 += 8;
                }
#endif // __riscv_vector
            }
        }
        if (pack4to8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const float* r0 = bottom_blob.channel(q * 2);
                const float* r1 = bottom_blob.channel(q * 2 + 1);

                float* outptr = top_blob.channel(q);

#if __riscv_vector
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m1(n);

                    vfloat32m1x4_t _p0 = __riscv_vlseg4e32_v_f32m1x4(r0, vl);
                    vfloat32m1x4_t _p1 = __riscv_vlseg4e32_v_f32m1x4(r1, vl);

                    vfloat32m1_t _p00 = __riscv_vget_v_f32m1x4_f32m1(_p0, 0);
                    vfloat32m1_t _p01 = __riscv_vget_v_f32m1x4_f32m1(_p0, 1);
                    vfloat32m1_t _p02 = __riscv_vget_v_f32m1x4_f32m1(_p0, 2);
                    vfloat32m1_t _p03 = __riscv_vget_v_f32m1x4_f32m1(_p0, 3);
                    vfloat32m1_t _p10 = __riscv_vget_v_f32m1x4_f32m1(_p1, 0);
                    vfloat32m1_t _p11 = __riscv_vget_v_f32m1x4_f32m1(_p1, 1);
                    vfloat32m1_t _p12 = __riscv_vget_v_f32m1x4_f32m1(_p1, 2);
                    vfloat32m1_t _p13 = __riscv_vget_v_f32m1x4_f32m1(_p1, 3);

                    __riscv_vsseg8e32_v_f32m1x8(outptr, __riscv_vcreate_v_f32m1x8(_p00, _p01, _p02, _p03, _p10, _p11, _p12, _p13), vl);

                    r0 += vl * 4;
                    r1 += vl * 4;
                    outptr += vl * 8;
                    n -= vl;
                }
#else  // __riscv_vector
                for (int i = 0; i < size; i++)
                {
                    outptr[0] = r0[0];
                    outptr[1] = r0[1];
                    outptr[2] = r0[2];
                    outptr[3] = r0[3];
                    outptr[4] = r1[0];
                    outptr[5] = r1[1];
                    outptr[6] = r1[2];
                    outptr[7] = r1[3];

                    r0 += 4;
                    r1 += 4;
                    outptr += 8;
                }
#endif // __riscv_vector
            }
        }
        if (pack8to4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* r0 = bottom_blob.channel(q);

                float* outptr0 = top_blob.channel(q * 2);
                float* outptr1 = top_blob.channel(q * 2 + 1);

#if __riscv_vector
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e32m1(n);

                    vfloat32m1x8_t _p = __riscv_vlseg8e32_v_f32m1x8(r0, vl);

                    vfloat32m1_t _p0 = __riscv_vget_v_f32m1x8_f32m1(_p, 0);
                    vfloat32m1_t _p1 = __riscv_vget_v_f32m1x8_f32m1(_p, 1);
                    vfloat32m1_t _p2 = __riscv_vget_v_f32m1x8_f32m1(_p, 2);
                    vfloat32m1_t _p3 = __riscv_vget_v_f32m1x8_f32m1(_p, 3);
                    vfloat32m1_t _p4 = __riscv_vget_v_f32m1x8_f32m1(_p, 4);
                    vfloat32m1_t _p5 = __riscv_vget_v_f32m1x8_f32m1(_p, 5);
                    vfloat32m1_t _p6 = __riscv_vget_v_f32m1x8_f32m1(_p, 6);
                    vfloat32m1_t _p7 = __riscv_vget_v_f32m1x8_f32m1(_p, 7);

                    __riscv_vsseg4e32_v_f32m1x4(outptr0, __riscv_vcreate_v_f32m1x4(_p0, _p1, _p2, _p3), vl);
                    __riscv_vsseg4e32_v_f32m1x4(outptr1, __riscv_vcreate_v_f32m1x4(_p4, _p5, _p6, _p7), vl);

                    r0 += vl * 8;
                    outptr0 += vl * 4;
                    outptr1 += vl * 4;
                    n -= vl;
                }
#else  // __riscv_vector
                for (int i = 0; i < size; i++)
                {
                    outptr0[0] = r0[0];
                    outptr0[1] = r0[1];
                    outptr0[2] = r0[2];
                    outptr0[3] = r0[3];
                    outptr1[0] = r0[4];
                    outptr1[1] = r0[5];
                    outptr1[2] = r0[6];
                    outptr1[3] = r0[7];

                    r0 += 8;
                    outptr0 += 4;
                    outptr1 += 4;
                }
#endif // __riscv_vector
            }
        }

        return 0;
    }

    return 0;
}

int Packing_riscv::forward_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (use_padding)
    {
        return Packing::forward(bottom_blob, top_blob, opt);
    }

    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    if (elempack == out_elempack)
    {
        top_blob = bottom_blob;
        return 0;
    }

    bool pack1to4 = elempack == 1 && out_elempack == 4;
    bool pack4to1 = elempack == 4 && out_elempack == 1;
    bool pack1to8 = elempack == 1 && out_elempack == 8;
    bool pack8to1 = elempack == 8 && out_elempack == 1;
    bool pack4to8 = elempack == 4 && out_elempack == 8;
    bool pack8to4 = elempack == 8 && out_elempack == 4;

    if (!pack1to4 && !pack4to1 && !pack1to8 && !pack8to1 && !pack4to8 && !pack8to4)
    {
        return Packing::forward(bottom_blob, top_blob, opt);
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;

    if (!use_padding)
    {
        // identity if use_padding not allowed
        if (dims == 1 && w * elempack % out_elempack != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
        if (dims == 2 && h * elempack % out_elempack != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
        if ((dims == 3 || dims == 4) && channels * elempack % out_elempack != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
    }

    if (dims == 1)
    {
        top_blob = bottom_blob;
        top_blob.w = w * elempack / out_elempack;
        top_blob.cstep = bottom_blob.cstep * elempack / out_elempack;
        top_blob.elemsize = elemsize / elempack * out_elempack;
        top_blob.elempack = out_elempack;
        return 0;
    }

    if (dims == 2)
    {
        int outh = h * elempack / out_elempack;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        top_blob.create(w, outh, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (pack1to4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < outh; i++)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(i * 4);
                const unsigned short* r1 = bottom_blob.row<const unsigned short>(i * 4 + 1);
                const unsigned short* r2 = bottom_blob.row<const unsigned short>(i * 4 + 2);
                const unsigned short* r3 = bottom_blob.row<const unsigned short>(i * 4 + 3);

                unsigned short* outptr = top_blob.row<unsigned short>(i);

#if __riscv_vector
                int n = w;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m2(n);

                    vuint16m2_t _p0 = __riscv_vle16_v_u16m2(r0, vl);
                    vuint16m2_t _p1 = __riscv_vle16_v_u16m2(r1, vl);
                    vuint16m2_t _p2 = __riscv_vle16_v_u16m2(r2, vl);
                    vuint16m2_t _p3 = __riscv_vle16_v_u16m2(r3, vl);
                    __riscv_vsseg4e16_v_u16m2x4(outptr, __riscv_vcreate_v_u16m2x4(_p0, _p1, _p2, _p3), vl);

                    r0 += vl;
                    r1 += vl;
                    r2 += vl;
                    r3 += vl;
                    outptr += vl * 4;
                    n -= vl;
                }
#else  // __riscv_vector
                for (int j = 0; j < w; j++)
                {
                    outptr[0] = *r0++;
                    outptr[1] = *r1++;
                    outptr[2] = *r2++;
                    outptr[3] = *r3++;

                    outptr += 4;
                }
#endif // __riscv_vector
            }
        }
        if (pack4to1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(i);

                unsigned short* outptr0 = top_blob.row<unsigned short>(i * 4);
                unsigned short* outptr1 = top_blob.row<unsigned short>(i * 4 + 1);
                unsigned short* outptr2 = top_blob.row<unsigned short>(i * 4 + 2);
                unsigned short* outptr3 = top_blob.row<unsigned short>(i * 4 + 3);

#if __riscv_vector
                int n = w;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m2(n);

                    vuint16m2x4_t _p = __riscv_vlseg4e16_v_u16m2x4(r0, vl);
                    __riscv_vse16_v_u16m2(outptr0, __riscv_vget_v_u16m2x4_u16m2(_p, 0), vl);
                    __riscv_vse16_v_u16m2(outptr1, __riscv_vget_v_u16m2x4_u16m2(_p, 1), vl);
                    __riscv_vse16_v_u16m2(outptr2, __riscv_vget_v_u16m2x4_u16m2(_p, 2), vl);
                    __riscv_vse16_v_u16m2(outptr3, __riscv_vget_v_u16m2x4_u16m2(_p, 3), vl);

                    r0 += vl * 4;
                    outptr0 += vl;
                    outptr1 += vl;
                    outptr2 += vl;
                    outptr3 += vl;
                    n -= vl;
                }
#else  // __riscv_vector
                for (int j = 0; j < w; j++)
                {
                    *outptr0++ = r0[0];
                    *outptr1++ = r0[1];
                    *outptr2++ = r0[2];
                    *outptr3++ = r0[3];

                    r0 += 4;
                }
#endif // __riscv_vector
            }
        }
        if (pack1to8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < outh; i++)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(i * 8);
                const unsigned short* r1 = bottom_blob.row<const unsigned short>(i * 8 + 1);
                const unsigned short* r2 = bottom_blob.row<const unsigned short>(i * 8 + 2);
                const unsigned short* r3 = bottom_blob.row<const unsigned short>(i * 8 + 3);
                const unsigned short* r4 = bottom_blob.row<const unsigned short>(i * 8 + 4);
                const unsigned short* r5 = bottom_blob.row<const unsigned short>(i * 8 + 5);
                const unsigned short* r6 = bottom_blob.row<const unsigned short>(i * 8 + 6);
                const unsigned short* r7 = bottom_blob.row<const unsigned short>(i * 8 + 7);

                unsigned short* outptr = top_blob.row<unsigned short>(i);

#if __riscv_vector
                int n = w;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m1(n);

                    vuint16m1_t _p0 = __riscv_vle16_v_u16m1(r0, vl);
                    vuint16m1_t _p1 = __riscv_vle16_v_u16m1(r1, vl);
                    vuint16m1_t _p2 = __riscv_vle16_v_u16m1(r2, vl);
                    vuint16m1_t _p3 = __riscv_vle16_v_u16m1(r3, vl);
                    vuint16m1_t _p4 = __riscv_vle16_v_u16m1(r4, vl);
                    vuint16m1_t _p5 = __riscv_vle16_v_u16m1(r5, vl);
                    vuint16m1_t _p6 = __riscv_vle16_v_u16m1(r6, vl);
                    vuint16m1_t _p7 = __riscv_vle16_v_u16m1(r7, vl);
                    __riscv_vsseg8e16_v_u16m1x8(outptr, __riscv_vcreate_v_u16m1x8(_p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7), vl);

                    r0 += vl;
                    r1 += vl;
                    r2 += vl;
                    r3 += vl;
                    r4 += vl;
                    r5 += vl;
                    r6 += vl;
                    r7 += vl;
                    outptr += vl * 8;
                    n -= vl;
                }
#else  // __riscv_vector
                for (int j = 0; j < w; j++)
                {
                    outptr[0] = *r0++;
                    outptr[1] = *r1++;
                    outptr[2] = *r2++;
                    outptr[3] = *r3++;
                    outptr[4] = *r4++;
                    outptr[5] = *r5++;
                    outptr[6] = *r6++;
                    outptr[7] = *r7++;

                    outptr += 8;
                }
#endif // __riscv_vector
            }
        }
        if (pack8to1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(i);

                unsigned short* outptr0 = top_blob.row<unsigned short>(i * 8);
                unsigned short* outptr1 = top_blob.row<unsigned short>(i * 8 + 1);
                unsigned short* outptr2 = top_blob.row<unsigned short>(i * 8 + 2);
                unsigned short* outptr3 = top_blob.row<unsigned short>(i * 8 + 3);
                unsigned short* outptr4 = top_blob.row<unsigned short>(i * 8 + 4);
                unsigned short* outptr5 = top_blob.row<unsigned short>(i * 8 + 5);
                unsigned short* outptr6 = top_blob.row<unsigned short>(i * 8 + 6);
                unsigned short* outptr7 = top_blob.row<unsigned short>(i * 8 + 7);

#if __riscv_vector
                int n = w;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m1(n);

                    vuint16m1x8_t _p = __riscv_vlseg8e16_v_u16m1x8(r0, vl);
                    __riscv_vse16_v_u16m1(outptr0, __riscv_vget_v_u16m1x8_u16m1(_p, 0), vl);
                    __riscv_vse16_v_u16m1(outptr1, __riscv_vget_v_u16m1x8_u16m1(_p, 1), vl);
                    __riscv_vse16_v_u16m1(outptr2, __riscv_vget_v_u16m1x8_u16m1(_p, 2), vl);
                    __riscv_vse16_v_u16m1(outptr3, __riscv_vget_v_u16m1x8_u16m1(_p, 3), vl);
                    __riscv_vse16_v_u16m1(outptr4, __riscv_vget_v_u16m1x8_u16m1(_p, 4), vl);
                    __riscv_vse16_v_u16m1(outptr5, __riscv_vget_v_u16m1x8_u16m1(_p, 5), vl);
                    __riscv_vse16_v_u16m1(outptr6, __riscv_vget_v_u16m1x8_u16m1(_p, 6), vl);
                    __riscv_vse16_v_u16m1(outptr7, __riscv_vget_v_u16m1x8_u16m1(_p, 7), vl);

                    r0 += vl * 8;
                    outptr0 += vl;
                    outptr1 += vl;
                    outptr2 += vl;
                    outptr3 += vl;
                    outptr4 += vl;
                    outptr5 += vl;
                    outptr6 += vl;
                    outptr7 += vl;
                    n -= vl;
                }
#else  // __riscv_vector
                for (int j = 0; j < w; j++)
                {
                    *outptr0++ = r0[0];
                    *outptr1++ = r0[1];
                    *outptr2++ = r0[2];
                    *outptr3++ = r0[3];
                    *outptr4++ = r0[4];
                    *outptr5++ = r0[5];
                    *outptr6++ = r0[6];
                    *outptr7++ = r0[7];

                    r0 += 8;
                }
#endif // __riscv_vector
            }
        }
        if (pack4to8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < outh; i++)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(i * 2);
                const unsigned short* r1 = bottom_blob.row<const unsigned short>(i * 2 + 1);

                unsigned short* outptr = top_blob.row<unsigned short>(i);

#if __riscv_vector
                int n = w;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m1(n);

                    vuint16m1x4_t _p0 = __riscv_vlseg4e16_v_u16m1x4(r0, vl);
                    vuint16m1x4_t _p1 = __riscv_vlseg4e16_v_u16m1x4(r1, vl);

                    vuint16m1_t _p00 = __riscv_vget_v_u16m1x4_u16m1(_p0, 0);
                    vuint16m1_t _p01 = __riscv_vget_v_u16m1x4_u16m1(_p0, 1);
                    vuint16m1_t _p02 = __riscv_vget_v_u16m1x4_u16m1(_p0, 2);
                    vuint16m1_t _p03 = __riscv_vget_v_u16m1x4_u16m1(_p0, 3);
                    vuint16m1_t _p10 = __riscv_vget_v_u16m1x4_u16m1(_p1, 0);
                    vuint16m1_t _p11 = __riscv_vget_v_u16m1x4_u16m1(_p1, 1);
                    vuint16m1_t _p12 = __riscv_vget_v_u16m1x4_u16m1(_p1, 2);
                    vuint16m1_t _p13 = __riscv_vget_v_u16m1x4_u16m1(_p1, 3);

                    __riscv_vsseg8e16_v_u16m1x8(outptr, __riscv_vcreate_v_u16m1x8(_p00, _p01, _p02, _p03, _p10, _p11, _p12, _p13), vl);

                    r0 += vl * 4;
                    r1 += vl * 4;
                    outptr += vl * 8;
                    n -= vl;
                }
#else  // __riscv_vector
                for (int j = 0; j < w; j++)
                {
                    outptr[0] = r0[0];
                    outptr[1] = r0[1];
                    outptr[2] = r0[2];
                    outptr[3] = r0[3];
                    outptr[4] = r1[0];
                    outptr[5] = r1[1];
                    outptr[6] = r1[2];
                    outptr[7] = r1[3];

                    r0 += 4;
                    r1 += 4;
                    outptr += 8;
                }
#endif // __riscv_vector
            }
        }
        if (pack8to4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const unsigned short* r0 = bottom_blob.row<const unsigned short>(i);

                unsigned short* outptr0 = top_blob.row<unsigned short>(i * 2);
                unsigned short* outptr1 = top_blob.row<unsigned short>(i * 2 + 1);

#if __riscv_vector
                int n = w;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m1(n);

                    vuint16m1x8_t _p = __riscv_vlseg8e16_v_u16m1x8(r0, vl);

                    vuint16m1_t _p0 = __riscv_vget_v_u16m1x8_u16m1(_p, 0);
                    vuint16m1_t _p1 = __riscv_vget_v_u16m1x8_u16m1(_p, 1);
                    vuint16m1_t _p2 = __riscv_vget_v_u16m1x8_u16m1(_p, 2);
                    vuint16m1_t _p3 = __riscv_vget_v_u16m1x8_u16m1(_p, 3);
                    vuint16m1_t _p4 = __riscv_vget_v_u16m1x8_u16m1(_p, 4);
                    vuint16m1_t _p5 = __riscv_vget_v_u16m1x8_u16m1(_p, 5);
                    vuint16m1_t _p6 = __riscv_vget_v_u16m1x8_u16m1(_p, 6);
                    vuint16m1_t _p7 = __riscv_vget_v_u16m1x8_u16m1(_p, 7);

                    __riscv_vsseg4e16_v_u16m1x4(outptr0, __riscv_vcreate_v_u16m1x4(_p0, _p1, _p2, _p3), vl);
                    __riscv_vsseg4e16_v_u16m1x4(outptr1, __riscv_vcreate_v_u16m1x4(_p4, _p5, _p6, _p7), vl);

                    r0 += vl * 8;
                    outptr0 += vl * 4;
                    outptr1 += vl * 4;
                    n -= vl;
                }
#else  // __riscv_vector
                for (int j = 0; j < w; j++)
                {
                    outptr0[0] = r0[0];
                    outptr0[1] = r0[1];
                    outptr0[2] = r0[2];
                    outptr0[3] = r0[3];
                    outptr1[0] = r0[4];
                    outptr1[1] = r0[5];
                    outptr1[2] = r0[6];
                    outptr1[3] = r0[7];

                    r0 += 8;
                    outptr0 += 4;
                    outptr1 += 4;
                }
#endif // __riscv_vector
            }
        }

        return 0;
    }

    if (dims == 3 || dims == 4)
    {
        int size = w * h * d;
        int outc = channels * elempack / out_elempack;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if (dims == 3)
            top_blob.create(w, h, outc, out_elemsize, out_elempack, opt.blob_allocator);
        else // if (dims == 4)
            top_blob.create(w, h, d, outc, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (pack1to4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const unsigned short* r0 = bottom_blob.channel(q * 4);
                const unsigned short* r1 = bottom_blob.channel(q * 4 + 1);
                const unsigned short* r2 = bottom_blob.channel(q * 4 + 2);
                const unsigned short* r3 = bottom_blob.channel(q * 4 + 3);

                unsigned short* outptr = top_blob.channel(q);

#if __riscv_vector
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m2(n);

                    vuint16m2_t _p0 = __riscv_vle16_v_u16m2(r0, vl);
                    vuint16m2_t _p1 = __riscv_vle16_v_u16m2(r1, vl);
                    vuint16m2_t _p2 = __riscv_vle16_v_u16m2(r2, vl);
                    vuint16m2_t _p3 = __riscv_vle16_v_u16m2(r3, vl);
                    __riscv_vsseg4e16_v_u16m2x4(outptr, __riscv_vcreate_v_u16m2x4(_p0, _p1, _p2, _p3), vl);

                    r0 += vl;
                    r1 += vl;
                    r2 += vl;
                    r3 += vl;
                    outptr += vl * 4;
                    n -= vl;
                }
#else  // __riscv_vector
                for (int i = 0; i < size; i++)
                {
                    outptr[0] = *r0++;
                    outptr[1] = *r1++;
                    outptr[2] = *r2++;
                    outptr[3] = *r3++;

                    outptr += 4;
                }
#endif // __riscv_vector
            }
        }
        if (pack4to1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* r0 = bottom_blob.channel(q);

                unsigned short* outptr0 = top_blob.channel(q * 4);
                unsigned short* outptr1 = top_blob.channel(q * 4 + 1);
                unsigned short* outptr2 = top_blob.channel(q * 4 + 2);
                unsigned short* outptr3 = top_blob.channel(q * 4 + 3);

#if __riscv_vector
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m2(n);

                    vuint16m2x4_t _p = __riscv_vlseg4e16_v_u16m2x4(r0, vl);
                    __riscv_vse16_v_u16m2(outptr0, __riscv_vget_v_u16m2x4_u16m2(_p, 0), vl);
                    __riscv_vse16_v_u16m2(outptr1, __riscv_vget_v_u16m2x4_u16m2(_p, 1), vl);
                    __riscv_vse16_v_u16m2(outptr2, __riscv_vget_v_u16m2x4_u16m2(_p, 2), vl);
                    __riscv_vse16_v_u16m2(outptr3, __riscv_vget_v_u16m2x4_u16m2(_p, 3), vl);

                    r0 += vl * 4;
                    outptr0 += vl;
                    outptr1 += vl;
                    outptr2 += vl;
                    outptr3 += vl;
                    n -= vl;
                }
#else  // __riscv_vector
                for (int i = 0; i < size; i++)
                {
                    *outptr0++ = r0[0];
                    *outptr1++ = r0[1];
                    *outptr2++ = r0[2];
                    *outptr3++ = r0[3];

                    r0 += 4;
                }
#endif // __riscv_vector
            }
        }
        if (pack1to8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const unsigned short* r0 = bottom_blob.channel(q * 8);
                const unsigned short* r1 = bottom_blob.channel(q * 8 + 1);
                const unsigned short* r2 = bottom_blob.channel(q * 8 + 2);
                const unsigned short* r3 = bottom_blob.channel(q * 8 + 3);
                const unsigned short* r4 = bottom_blob.channel(q * 8 + 4);
                const unsigned short* r5 = bottom_blob.channel(q * 8 + 5);
                const unsigned short* r6 = bottom_blob.channel(q * 8 + 6);
                const unsigned short* r7 = bottom_blob.channel(q * 8 + 7);

                unsigned short* outptr = top_blob.channel(q);

#if __riscv_vector
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m1(n);

                    vuint16m1_t _p0 = __riscv_vle16_v_u16m1(r0, vl);
                    vuint16m1_t _p1 = __riscv_vle16_v_u16m1(r1, vl);
                    vuint16m1_t _p2 = __riscv_vle16_v_u16m1(r2, vl);
                    vuint16m1_t _p3 = __riscv_vle16_v_u16m1(r3, vl);
                    vuint16m1_t _p4 = __riscv_vle16_v_u16m1(r4, vl);
                    vuint16m1_t _p5 = __riscv_vle16_v_u16m1(r5, vl);
                    vuint16m1_t _p6 = __riscv_vle16_v_u16m1(r6, vl);
                    vuint16m1_t _p7 = __riscv_vle16_v_u16m1(r7, vl);
                    __riscv_vsseg8e16_v_u16m1x8(outptr, __riscv_vcreate_v_u16m1x8(_p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7), vl);

                    r0 += vl;
                    r1 += vl;
                    r2 += vl;
                    r3 += vl;
                    r4 += vl;
                    r5 += vl;
                    r6 += vl;
                    r7 += vl;
                    outptr += vl * 8;
                    n -= vl;
                }
#else  // __riscv_vector
                for (int i = 0; i < size; i++)
                {
                    outptr[0] = *r0++;
                    outptr[1] = *r1++;
                    outptr[2] = *r2++;
                    outptr[3] = *r3++;
                    outptr[4] = *r4++;
                    outptr[5] = *r5++;
                    outptr[6] = *r6++;
                    outptr[7] = *r7++;

                    outptr += 8;
                }
#endif // __riscv_vector
            }
        }
        if (pack8to1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* r0 = bottom_blob.channel(q);

                unsigned short* outptr0 = top_blob.channel(q * 8);
                unsigned short* outptr1 = top_blob.channel(q * 8 + 1);
                unsigned short* outptr2 = top_blob.channel(q * 8 + 2);
                unsigned short* outptr3 = top_blob.channel(q * 8 + 3);
                unsigned short* outptr4 = top_blob.channel(q * 8 + 4);
                unsigned short* outptr5 = top_blob.channel(q * 8 + 5);
                unsigned short* outptr6 = top_blob.channel(q * 8 + 6);
                unsigned short* outptr7 = top_blob.channel(q * 8 + 7);

#if __riscv_vector
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m1(n);

                    vuint16m1x8_t _p = __riscv_vlseg8e16_v_u16m1x8(r0, vl);
                    __riscv_vse16_v_u16m1(outptr0, __riscv_vget_v_u16m1x8_u16m1(_p, 0), vl);
                    __riscv_vse16_v_u16m1(outptr1, __riscv_vget_v_u16m1x8_u16m1(_p, 1), vl);
                    __riscv_vse16_v_u16m1(outptr2, __riscv_vget_v_u16m1x8_u16m1(_p, 2), vl);
                    __riscv_vse16_v_u16m1(outptr3, __riscv_vget_v_u16m1x8_u16m1(_p, 3), vl);
                    __riscv_vse16_v_u16m1(outptr4, __riscv_vget_v_u16m1x8_u16m1(_p, 4), vl);
                    __riscv_vse16_v_u16m1(outptr5, __riscv_vget_v_u16m1x8_u16m1(_p, 5), vl);
                    __riscv_vse16_v_u16m1(outptr6, __riscv_vget_v_u16m1x8_u16m1(_p, 6), vl);
                    __riscv_vse16_v_u16m1(outptr7, __riscv_vget_v_u16m1x8_u16m1(_p, 7), vl);

                    r0 += vl * 8;
                    outptr0 += vl;
                    outptr1 += vl;
                    outptr2 += vl;
                    outptr3 += vl;
                    outptr4 += vl;
                    outptr5 += vl;
                    outptr6 += vl;
                    outptr7 += vl;
                    n -= vl;
                }
#else  // __riscv_vector
                for (int i = 0; i < size; i++)
                {
                    *outptr0++ = r0[0];
                    *outptr1++ = r0[1];
                    *outptr2++ = r0[2];
                    *outptr3++ = r0[3];
                    *outptr4++ = r0[4];
                    *outptr5++ = r0[5];
                    *outptr6++ = r0[6];
                    *outptr7++ = r0[7];

                    r0 += 8;
                }
#endif // __riscv_vector
            }
        }
        if (pack4to8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const unsigned short* r0 = bottom_blob.channel(q * 2);
                const unsigned short* r1 = bottom_blob.channel(q * 2 + 1);

                unsigned short* outptr = top_blob.channel(q);

#if __riscv_vector
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m1(n);

                    vuint16m1x4_t _p0 = __riscv_vlseg4e16_v_u16m1x4(r0, vl);
                    vuint16m1x4_t _p1 = __riscv_vlseg4e16_v_u16m1x4(r1, vl);

                    vuint16m1_t _p00 = __riscv_vget_v_u16m1x4_u16m1(_p0, 0);
                    vuint16m1_t _p01 = __riscv_vget_v_u16m1x4_u16m1(_p0, 1);
                    vuint16m1_t _p02 = __riscv_vget_v_u16m1x4_u16m1(_p0, 2);
                    vuint16m1_t _p03 = __riscv_vget_v_u16m1x4_u16m1(_p0, 3);
                    vuint16m1_t _p10 = __riscv_vget_v_u16m1x4_u16m1(_p1, 0);
                    vuint16m1_t _p11 = __riscv_vget_v_u16m1x4_u16m1(_p1, 1);
                    vuint16m1_t _p12 = __riscv_vget_v_u16m1x4_u16m1(_p1, 2);
                    vuint16m1_t _p13 = __riscv_vget_v_u16m1x4_u16m1(_p1, 3);
                    __riscv_vsseg8e16_v_u16m1x8(outptr, __riscv_vcreate_v_u16m1x8(_p00, _p01, _p02, _p03, _p10, _p11, _p12, _p13), vl);

                    r0 += vl * 4;
                    r1 += vl * 4;
                    outptr += vl * 8;
                    n -= vl;
                }
#else  // __riscv_vector
                for (int i = 0; i < size; i++)
                {
                    outptr[0] = r0[0];
                    outptr[1] = r0[1];
                    outptr[2] = r0[2];
                    outptr[3] = r0[3];
                    outptr[4] = r1[0];
                    outptr[5] = r1[1];
                    outptr[6] = r1[2];
                    outptr[7] = r1[3];

                    r0 += 4;
                    r1 += 4;
                    outptr += 8;
                }
#endif // __riscv_vector
            }
        }
        if (pack8to4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const unsigned short* r0 = bottom_blob.channel(q);

                unsigned short* outptr0 = top_blob.channel(q * 2);
                unsigned short* outptr1 = top_blob.channel(q * 2 + 1);

#if __riscv_vector
                int n = size;
                while (n > 0)
                {
                    size_t vl = __riscv_vsetvl_e16m1(n);

                    vuint16m1x8_t _p = __riscv_vlseg8e16_v_u16m1x8(r0, vl);

                    vuint16m1_t _p0 = __riscv_vget_v_u16m1x8_u16m1(_p, 0);
                    vuint16m1_t _p1 = __riscv_vget_v_u16m1x8_u16m1(_p, 1);
                    vuint16m1_t _p2 = __riscv_vget_v_u16m1x8_u16m1(_p, 2);
                    vuint16m1_t _p3 = __riscv_vget_v_u16m1x8_u16m1(_p, 3);
                    vuint16m1_t _p4 = __riscv_vget_v_u16m1x8_u16m1(_p, 4);
                    vuint16m1_t _p5 = __riscv_vget_v_u16m1x8_u16m1(_p, 5);
                    vuint16m1_t _p6 = __riscv_vget_v_u16m1x8_u16m1(_p, 6);
                    vuint16m1_t _p7 = __riscv_vget_v_u16m1x8_u16m1(_p, 7);

                    __riscv_vsseg4e16_v_u16m1x4(outptr0, __riscv_vcreate_v_u16m1x4(_p0, _p1, _p2, _p3), vl);
                    __riscv_vsseg4e16_v_u16m1x4(outptr1, __riscv_vcreate_v_u16m1x4(_p4, _p5, _p6, _p7), vl);

                    r0 += vl * 8;
                    outptr0 += vl * 4;
                    outptr1 += vl * 4;
                    n -= vl;
                }
#else  // __riscv_vector
                for (int i = 0; i < size; i++)
                {
                    outptr0[0] = r0[0];
                    outptr0[1] = r0[1];
                    outptr0[2] = r0[2];
                    outptr0[3] = r0[3];
                    outptr1[0] = r0[4];
                    outptr1[1] = r0[5];
                    outptr1[2] = r0[6];
                    outptr1[3] = r0[7];

                    r0 += 8;
                    outptr0 += 4;
                    outptr1 += 4;
                }
#endif // __riscv_vector
            }
        }

        return 0;
    }

    return 0;
}

int Packing_riscv::forward_int8(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (use_padding)
    {
        return Packing::forward(bottom_blob, top_blob, opt);
    }

    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    if (elempack == out_elempack)
    {
        top_blob = bottom_blob;
        return 0;
    }

    bool pack1to8 = elempack == 1 && out_elempack == 8;
    bool pack8to1 = elempack == 8 && out_elempack == 1;

    if (!pack1to8 && !pack8to1)
    {
        return Packing::forward(bottom_blob, top_blob, opt);
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;

    if (!use_padding)
    {
        // identity if use_padding not allowed
        if (dims == 1 && w * elempack % out_elempack != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
        if (dims == 2 && h * elempack % out_elempack != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
        if ((dims == 3 || dims == 4) && channels * elempack % out_elempack != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
    }

    if (dims == 1)
    {
        top_blob = bottom_blob;
        top_blob.w = w * elempack / out_elempack;
        top_blob.cstep = bottom_blob.cstep * elempack / out_elempack;
        top_blob.elemsize = elemsize / elempack * out_elempack;
        top_blob.elempack = out_elempack;
        return 0;
    }

    if (dims == 2)
    {
        int outh = h * elempack / out_elempack;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        top_blob.create(w, outh, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (pack1to8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < outh; i++)
            {
                const signed char* r0 = bottom_blob.row<const signed char>(i * 8);
                const signed char* r1 = bottom_blob.row<const signed char>(i * 8 + 1);
                const signed char* r2 = bottom_blob.row<const signed char>(i * 8 + 2);
                const signed char* r3 = bottom_blob.row<const signed char>(i * 8 + 3);
                const signed char* r4 = bottom_blob.row<const signed char>(i * 8 + 4);
                const signed char* r5 = bottom_blob.row<const signed char>(i * 8 + 5);
                const signed char* r6 = bottom_blob.row<const signed char>(i * 8 + 6);
                const signed char* r7 = bottom_blob.row<const signed char>(i * 8 + 7);

                signed char* outptr = top_blob.row<signed char>(i);

                int j = 0;
                for (; j < w; j++)
                {
                    outptr[0] = *r0++;
                    outptr[1] = *r1++;
                    outptr[2] = *r2++;
                    outptr[3] = *r3++;
                    outptr[4] = *r4++;
                    outptr[5] = *r5++;
                    outptr[6] = *r6++;
                    outptr[7] = *r7++;

                    outptr += 8;
                }
            }
        }
        if (pack8to1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const signed char* r0 = bottom_blob.row<const signed char>(i);

                signed char* outptr0 = top_blob.row<signed char>(i * 8);
                signed char* outptr1 = top_blob.row<signed char>(i * 8 + 1);
                signed char* outptr2 = top_blob.row<signed char>(i * 8 + 2);
                signed char* outptr3 = top_blob.row<signed char>(i * 8 + 3);
                signed char* outptr4 = top_blob.row<signed char>(i * 8 + 4);
                signed char* outptr5 = top_blob.row<signed char>(i * 8 + 5);
                signed char* outptr6 = top_blob.row<signed char>(i * 8 + 6);
                signed char* outptr7 = top_blob.row<signed char>(i * 8 + 7);

                int j = 0;
                for (; j < w; j++)
                {
                    *outptr0++ = r0[0];
                    *outptr1++ = r0[1];
                    *outptr2++ = r0[2];
                    *outptr3++ = r0[3];
                    *outptr4++ = r0[4];
                    *outptr5++ = r0[5];
                    *outptr6++ = r0[6];
                    *outptr7++ = r0[7];

                    r0 += 8;
                }
            }
        }

        return 0;
    }

    if (dims == 3 || dims == 4)
    {
        int size = w * h * d;
        int outc = channels * elempack / out_elempack;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if (dims == 3)
            top_blob.create(w, h, outc, out_elemsize, out_elempack, opt.blob_allocator);
        else // if (dims == 4)
            top_blob.create(w, h, d, outc, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (pack1to8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < outc; q++)
            {
                const signed char* r0 = bottom_blob.channel(q * 8);
                const signed char* r1 = bottom_blob.channel(q * 8 + 1);
                const signed char* r2 = bottom_blob.channel(q * 8 + 2);
                const signed char* r3 = bottom_blob.channel(q * 8 + 3);
                const signed char* r4 = bottom_blob.channel(q * 8 + 4);
                const signed char* r5 = bottom_blob.channel(q * 8 + 5);
                const signed char* r6 = bottom_blob.channel(q * 8 + 6);
                const signed char* r7 = bottom_blob.channel(q * 8 + 7);

                signed char* outptr = top_blob.channel(q);

                int i = 0;
                for (; i < size; i++)
                {
                    outptr[0] = *r0++;
                    outptr[1] = *r1++;
                    outptr[2] = *r2++;
                    outptr[3] = *r3++;
                    outptr[4] = *r4++;
                    outptr[5] = *r5++;
                    outptr[6] = *r6++;
                    outptr[7] = *r7++;

                    outptr += 8;
                }
            }
        }
        if (pack8to1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const signed char* r0 = bottom_blob.channel(q);

                signed char* outptr0 = top_blob.channel(q * 8);
                signed char* outptr1 = top_blob.channel(q * 8 + 1);
                signed char* outptr2 = top_blob.channel(q * 8 + 2);
                signed char* outptr3 = top_blob.channel(q * 8 + 3);
                signed char* outptr4 = top_blob.channel(q * 8 + 4);
                signed char* outptr5 = top_blob.channel(q * 8 + 5);
                signed char* outptr6 = top_blob.channel(q * 8 + 6);
                signed char* outptr7 = top_blob.channel(q * 8 + 7);

                int i = 0;
                for (; i < size; i++)
                {
                    *outptr0++ = r0[0];
                    *outptr1++ = r0[1];
                    *outptr2++ = r0[2];
                    *outptr3++ = r0[3];
                    *outptr4++ = r0[4];
                    *outptr5++ = r0[5];
                    *outptr6++ = r0[6];
                    *outptr7++ = r0[7];

                    r0 += 8;
                }
            }
        }

        return 0;
    }

    return 0;
}

} // namespace ncnn
