// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "packing_powerpc.h"

#include <string.h>
#include <stdint.h>

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

namespace ncnn {

Packing_powerpc::Packing_powerpc()
{
    support_packing = true;
    support_any_packing = true;
}

int Packing_powerpc::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int elembits = bottom_blob.elembits();

    if (elembits == 8)
        return Packing::forward(bottom_blob, top_blob, opt);

    // fp16 stays a valid input when NCNN_BF16 is off. the kernels below
    // allocate by elemsize but read and write float, so any 16-bit tensor
    // has to take the generic path.
    if (elembits == 16)
        return Packing::forward(bottom_blob, top_blob, opt);

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
    int batch = bottom_blob.n;

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
#if NCNN_BATCH
        top_blob.nstep = bottom_blob.nstep * elempack / out_elempack;
#endif
        return 0;
    }

    if (dims == 2)
    {
        int outh = h * elempack / out_elempack;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        top_blob.create(w, outh, out_elemsize, out_elempack, batch, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (pack1to4)
        {
            const int total_bi = batch * outh;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bi = 0; bi < total_bi; bi++)
            {
                int b = bi / outh;
                int i = bi % outh;
                const float* r0 = bottom_blob.batch(b).row(i * 4);
                const float* r1 = bottom_blob.batch(b).row(i * 4 + 1);
                const float* r2 = bottom_blob.batch(b).row(i * 4 + 2);
                const float* r3 = bottom_blob.batch(b).row(i * 4 + 3);

                float* outptr = top_blob.batch(b).row(i);

                int j = 0;
#if __ALTIVEC__
                for (; j + 3 < w; j += 4)
                {
                    // transpose 4x4
                    v4f32 _r0 = __ncnn_altivec_load(r0);
                    v4f32 _r1 = __ncnn_altivec_load(r1);
                    v4f32 _r2 = __ncnn_altivec_load(r2);
                    v4f32 _r3 = __ncnn_altivec_load(r3);

                    transpose4x4_ps(_r0, _r1, _r2, _r3);

                    __ncnn_altivec_store(outptr, _r0);
                    __ncnn_altivec_store(outptr + 4, _r1);
                    __ncnn_altivec_store(outptr + 4 * 2, _r2);
                    __ncnn_altivec_store(outptr + 4 * 3, _r3);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    outptr += 16;
                }
#endif // __ALTIVEC__
                for (; j < w; j++)
                {
                    outptr[0] = *r0++;
                    outptr[1] = *r1++;
                    outptr[2] = *r2++;
                    outptr[3] = *r3++;

                    outptr += 4;
                }
            }
        }
        if (pack4to1)
        {
            const int total_bi = batch * h;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bi = 0; bi < total_bi; bi++)
            {
                int b = bi / h;
                int i = bi % h;
                const float* r0 = bottom_blob.batch(b).row(i);

                float* outptr0 = top_blob.batch(b).row(i * 4);
                float* outptr1 = top_blob.batch(b).row(i * 4 + 1);
                float* outptr2 = top_blob.batch(b).row(i * 4 + 2);
                float* outptr3 = top_blob.batch(b).row(i * 4 + 3);

                int j = 0;
#if __ALTIVEC__
                for (; j + 3 < w; j += 4)
                {
                    // transpose 4x4
                    v4f32 _r0 = __ncnn_altivec_load(r0);
                    v4f32 _r1 = __ncnn_altivec_load(r0 + 4);
                    v4f32 _r2 = __ncnn_altivec_load(r0 + 4 * 2);
                    v4f32 _r3 = __ncnn_altivec_load(r0 + 4 * 3);

                    transpose4x4_ps(_r0, _r1, _r2, _r3);

                    __ncnn_altivec_store(outptr0, _r0);
                    __ncnn_altivec_store(outptr1, _r1);
                    __ncnn_altivec_store(outptr2, _r2);
                    __ncnn_altivec_store(outptr3, _r3);

                    r0 += 16;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                }
#endif // __ALTIVEC__
                for (; j < w; j++)
                {
                    *outptr0++ = r0[0];
                    *outptr1++ = r0[1];
                    *outptr2++ = r0[2];
                    *outptr3++ = r0[3];

                    r0 += 4;
                }
            }
        }
        if (pack1to8)
        {
            const int total_bi = batch * outh;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bi = 0; bi < total_bi; bi++)
            {
                int b = bi / outh;
                int i = bi % outh;
                const float* r0 = bottom_blob.batch(b).row(i * 8);
                const float* r1 = bottom_blob.batch(b).row(i * 8 + 1);
                const float* r2 = bottom_blob.batch(b).row(i * 8 + 2);
                const float* r3 = bottom_blob.batch(b).row(i * 8 + 3);
                const float* r4 = bottom_blob.batch(b).row(i * 8 + 4);
                const float* r5 = bottom_blob.batch(b).row(i * 8 + 5);
                const float* r6 = bottom_blob.batch(b).row(i * 8 + 6);
                const float* r7 = bottom_blob.batch(b).row(i * 8 + 7);

                float* outptr = top_blob.batch(b).row(i);

                int j = 0;
#if __ALTIVEC__
                for (; j + 3 < w; j += 4)
                {
                    // transpose 8x4 as two 4x4 halves
                    v4f32 _r0 = __ncnn_altivec_load(r0);
                    v4f32 _r1 = __ncnn_altivec_load(r1);
                    v4f32 _r2 = __ncnn_altivec_load(r2);
                    v4f32 _r3 = __ncnn_altivec_load(r3);
                    v4f32 _r4 = __ncnn_altivec_load(r4);
                    v4f32 _r5 = __ncnn_altivec_load(r5);
                    v4f32 _r6 = __ncnn_altivec_load(r6);
                    v4f32 _r7 = __ncnn_altivec_load(r7);

                    transpose4x4_ps(_r0, _r1, _r2, _r3);
                    transpose4x4_ps(_r4, _r5, _r6, _r7);

                    // out element j = {r0[j], r1[j], ... r7[j]}
                    __ncnn_altivec_store(outptr, _r0);
                    __ncnn_altivec_store(outptr + 4, _r4);
                    __ncnn_altivec_store(outptr + 8, _r1);
                    __ncnn_altivec_store(outptr + 12, _r5);
                    __ncnn_altivec_store(outptr + 16, _r2);
                    __ncnn_altivec_store(outptr + 20, _r6);
                    __ncnn_altivec_store(outptr + 24, _r3);
                    __ncnn_altivec_store(outptr + 28, _r7);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    r4 += 4;
                    r5 += 4;
                    r6 += 4;
                    r7 += 4;
                    outptr += 32;
                }
#endif // __ALTIVEC__
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
            const int total_bi = batch * h;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bi = 0; bi < total_bi; bi++)
            {
                int b = bi / h;
                int i = bi % h;
                const float* r0 = bottom_blob.batch(b).row(i);

                float* outptr0 = top_blob.batch(b).row(i * 8);
                float* outptr1 = top_blob.batch(b).row(i * 8 + 1);
                float* outptr2 = top_blob.batch(b).row(i * 8 + 2);
                float* outptr3 = top_blob.batch(b).row(i * 8 + 3);
                float* outptr4 = top_blob.batch(b).row(i * 8 + 4);
                float* outptr5 = top_blob.batch(b).row(i * 8 + 5);
                float* outptr6 = top_blob.batch(b).row(i * 8 + 6);
                float* outptr7 = top_blob.batch(b).row(i * 8 + 7);

                int j = 0;
#if __ALTIVEC__
                for (; j + 3 < w; j += 4)
                {
                    // transpose 4x8 as two 4x4 halves
                    v4f32 _v0 = __ncnn_altivec_load(r0);
                    v4f32 _v1 = __ncnn_altivec_load(r0 + 4);
                    v4f32 _v2 = __ncnn_altivec_load(r0 + 8);
                    v4f32 _v3 = __ncnn_altivec_load(r0 + 12);
                    v4f32 _v4 = __ncnn_altivec_load(r0 + 16);
                    v4f32 _v5 = __ncnn_altivec_load(r0 + 20);
                    v4f32 _v6 = __ncnn_altivec_load(r0 + 24);
                    v4f32 _v7 = __ncnn_altivec_load(r0 + 28);

                    // rows 0-3 come from the low half of each element
                    transpose4x4_ps(_v0, _v2, _v4, _v6);
                    // rows 4-7 come from the high half of each element
                    transpose4x4_ps(_v1, _v3, _v5, _v7);

                    __ncnn_altivec_store(outptr0, _v0);
                    __ncnn_altivec_store(outptr1, _v2);
                    __ncnn_altivec_store(outptr2, _v4);
                    __ncnn_altivec_store(outptr3, _v6);
                    __ncnn_altivec_store(outptr4, _v1);
                    __ncnn_altivec_store(outptr5, _v3);
                    __ncnn_altivec_store(outptr6, _v5);
                    __ncnn_altivec_store(outptr7, _v7);

                    r0 += 32;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                    outptr4 += 4;
                    outptr5 += 4;
                    outptr6 += 4;
                    outptr7 += 4;
                }
#endif // __ALTIVEC__
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
        if (pack4to8)
        {
            const int total_bi = batch * outh;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bi = 0; bi < total_bi; bi++)
            {
                int b = bi / outh;
                int i = bi % outh;
                const float* r0 = bottom_blob.batch(b).row(i * 2);
                const float* r1 = bottom_blob.batch(b).row(i * 2 + 1);

                float* outptr = top_blob.batch(b).row(i);

                for (int j = 0; j < w; j++)
                {
#if __ALTIVEC__
                    __ncnn_altivec_store(outptr, __ncnn_altivec_load(r0));
                    __ncnn_altivec_store(outptr + 4, __ncnn_altivec_load(r1));
#else
                    outptr[0] = r0[0];
                    outptr[1] = r0[1];
                    outptr[2] = r0[2];
                    outptr[3] = r0[3];
                    outptr[4] = r1[0];
                    outptr[5] = r1[1];
                    outptr[6] = r1[2];
                    outptr[7] = r1[3];
#endif

                    r0 += 4;
                    r1 += 4;
                    outptr += 8;
                }
            }
        }
        if (pack8to4)
        {
            const int total_bi = batch * h;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bi = 0; bi < total_bi; bi++)
            {
                int b = bi / h;
                int i = bi % h;
                const float* r0 = bottom_blob.batch(b).row(i);

                float* outptr0 = top_blob.batch(b).row(i * 2);
                float* outptr1 = top_blob.batch(b).row(i * 2 + 1);

                for (int j = 0; j < w; j++)
                {
#if __ALTIVEC__
                    __ncnn_altivec_store(outptr0, __ncnn_altivec_load(r0));
                    __ncnn_altivec_store(outptr1, __ncnn_altivec_load(r0 + 4));
#else
                    outptr0[0] = r0[0];
                    outptr0[1] = r0[1];
                    outptr0[2] = r0[2];
                    outptr0[3] = r0[3];
                    outptr1[0] = r0[4];
                    outptr1[1] = r0[5];
                    outptr1[2] = r0[6];
                    outptr1[3] = r0[7];
#endif

                    r0 += 8;
                    outptr0 += 4;
                    outptr1 += 4;
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
            top_blob.create(w, h, outc, out_elemsize, out_elempack, batch, opt.blob_allocator);
        else
            top_blob.create(w, h, d, outc, out_elemsize, out_elempack, batch, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (pack1to4)
        {
            const int total_bq = batch * outc;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bq = 0; bq < total_bq; bq++)
            {
                int b = bq / outc;
                int q = bq % outc;
                const float* r0 = bottom_blob.batch(b).channel(q * 4);
                const float* r1 = bottom_blob.batch(b).channel(q * 4 + 1);
                const float* r2 = bottom_blob.batch(b).channel(q * 4 + 2);
                const float* r3 = bottom_blob.batch(b).channel(q * 4 + 3);

                float* outptr = top_blob.batch(b).channel(q);

                int i = 0;
#if __ALTIVEC__
                for (; i + 3 < size; i += 4)
                {
                    // transpose 4x4
                    v4f32 _r0 = __ncnn_altivec_load(r0);
                    v4f32 _r1 = __ncnn_altivec_load(r1);
                    v4f32 _r2 = __ncnn_altivec_load(r2);
                    v4f32 _r3 = __ncnn_altivec_load(r3);

                    transpose4x4_ps(_r0, _r1, _r2, _r3);

                    __ncnn_altivec_store(outptr, _r0);
                    __ncnn_altivec_store(outptr + 4, _r1);
                    __ncnn_altivec_store(outptr + 4 * 2, _r2);
                    __ncnn_altivec_store(outptr + 4 * 3, _r3);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    outptr += 16;
                }
#endif // __ALTIVEC__
                for (; i < size; i++)
                {
                    outptr[0] = *r0++;
                    outptr[1] = *r1++;
                    outptr[2] = *r2++;
                    outptr[3] = *r3++;

                    outptr += 4;
                }
            }
        }
        if (pack4to1)
        {
            const int total_bq = batch * channels;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bq = 0; bq < total_bq; bq++)
            {
                int b = bq / channels;
                int q = bq % channels;
                const float* r0 = bottom_blob.batch(b).channel(q);

                float* outptr0 = top_blob.batch(b).channel(q * 4);
                float* outptr1 = top_blob.batch(b).channel(q * 4 + 1);
                float* outptr2 = top_blob.batch(b).channel(q * 4 + 2);
                float* outptr3 = top_blob.batch(b).channel(q * 4 + 3);

                int i = 0;
#if __ALTIVEC__
                for (; i + 3 < size; i += 4)
                {
                    // transpose 4x4
                    v4f32 _r0 = __ncnn_altivec_load(r0);
                    v4f32 _r1 = __ncnn_altivec_load(r0 + 4);
                    v4f32 _r2 = __ncnn_altivec_load(r0 + 4 * 2);
                    v4f32 _r3 = __ncnn_altivec_load(r0 + 4 * 3);

                    transpose4x4_ps(_r0, _r1, _r2, _r3);

                    __ncnn_altivec_store(outptr0, _r0);
                    __ncnn_altivec_store(outptr1, _r1);
                    __ncnn_altivec_store(outptr2, _r2);
                    __ncnn_altivec_store(outptr3, _r3);

                    r0 += 16;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                }
#endif // __ALTIVEC__
                for (; i < size; i++)
                {
                    *outptr0++ = r0[0];
                    *outptr1++ = r0[1];
                    *outptr2++ = r0[2];
                    *outptr3++ = r0[3];

                    r0 += 4;
                }
            }
        }
        if (pack1to8)
        {
            const int total_bq = batch * outc;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bq = 0; bq < total_bq; bq++)
            {
                int b = bq / outc;
                int q = bq % outc;
                const float* r0 = bottom_blob.batch(b).channel(q * 8);
                const float* r1 = bottom_blob.batch(b).channel(q * 8 + 1);
                const float* r2 = bottom_blob.batch(b).channel(q * 8 + 2);
                const float* r3 = bottom_blob.batch(b).channel(q * 8 + 3);
                const float* r4 = bottom_blob.batch(b).channel(q * 8 + 4);
                const float* r5 = bottom_blob.batch(b).channel(q * 8 + 5);
                const float* r6 = bottom_blob.batch(b).channel(q * 8 + 6);
                const float* r7 = bottom_blob.batch(b).channel(q * 8 + 7);

                float* outptr = top_blob.batch(b).channel(q);

                int i = 0;
#if __ALTIVEC__
                for (; i + 3 < size; i += 4)
                {
                    // transpose 8x4 as two 4x4 halves
                    v4f32 _r0 = __ncnn_altivec_load(r0);
                    v4f32 _r1 = __ncnn_altivec_load(r1);
                    v4f32 _r2 = __ncnn_altivec_load(r2);
                    v4f32 _r3 = __ncnn_altivec_load(r3);
                    v4f32 _r4 = __ncnn_altivec_load(r4);
                    v4f32 _r5 = __ncnn_altivec_load(r5);
                    v4f32 _r6 = __ncnn_altivec_load(r6);
                    v4f32 _r7 = __ncnn_altivec_load(r7);

                    transpose4x4_ps(_r0, _r1, _r2, _r3);
                    transpose4x4_ps(_r4, _r5, _r6, _r7);

                    // out element i = {r0[i], r1[i], ... r7[i]}
                    __ncnn_altivec_store(outptr, _r0);
                    __ncnn_altivec_store(outptr + 4, _r4);
                    __ncnn_altivec_store(outptr + 8, _r1);
                    __ncnn_altivec_store(outptr + 12, _r5);
                    __ncnn_altivec_store(outptr + 16, _r2);
                    __ncnn_altivec_store(outptr + 20, _r6);
                    __ncnn_altivec_store(outptr + 24, _r3);
                    __ncnn_altivec_store(outptr + 28, _r7);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    r4 += 4;
                    r5 += 4;
                    r6 += 4;
                    r7 += 4;
                    outptr += 32;
                }
#endif // __ALTIVEC__
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
            const int total_bq = batch * channels;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bq = 0; bq < total_bq; bq++)
            {
                int b = bq / channels;
                int q = bq % channels;
                const float* r0 = bottom_blob.batch(b).channel(q);

                float* outptr0 = top_blob.batch(b).channel(q * 8);
                float* outptr1 = top_blob.batch(b).channel(q * 8 + 1);
                float* outptr2 = top_blob.batch(b).channel(q * 8 + 2);
                float* outptr3 = top_blob.batch(b).channel(q * 8 + 3);
                float* outptr4 = top_blob.batch(b).channel(q * 8 + 4);
                float* outptr5 = top_blob.batch(b).channel(q * 8 + 5);
                float* outptr6 = top_blob.batch(b).channel(q * 8 + 6);
                float* outptr7 = top_blob.batch(b).channel(q * 8 + 7);

                int i = 0;
#if __ALTIVEC__
                for (; i + 3 < size; i += 4)
                {
                    // transpose 4x8 as two 4x4 halves
                    v4f32 _v0 = __ncnn_altivec_load(r0);
                    v4f32 _v1 = __ncnn_altivec_load(r0 + 4);
                    v4f32 _v2 = __ncnn_altivec_load(r0 + 8);
                    v4f32 _v3 = __ncnn_altivec_load(r0 + 12);
                    v4f32 _v4 = __ncnn_altivec_load(r0 + 16);
                    v4f32 _v5 = __ncnn_altivec_load(r0 + 20);
                    v4f32 _v6 = __ncnn_altivec_load(r0 + 24);
                    v4f32 _v7 = __ncnn_altivec_load(r0 + 28);

                    // rows 0-3 come from the low half of each element
                    transpose4x4_ps(_v0, _v2, _v4, _v6);
                    // rows 4-7 come from the high half of each element
                    transpose4x4_ps(_v1, _v3, _v5, _v7);

                    __ncnn_altivec_store(outptr0, _v0);
                    __ncnn_altivec_store(outptr1, _v2);
                    __ncnn_altivec_store(outptr2, _v4);
                    __ncnn_altivec_store(outptr3, _v6);
                    __ncnn_altivec_store(outptr4, _v1);
                    __ncnn_altivec_store(outptr5, _v3);
                    __ncnn_altivec_store(outptr6, _v5);
                    __ncnn_altivec_store(outptr7, _v7);

                    r0 += 32;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                    outptr4 += 4;
                    outptr5 += 4;
                    outptr6 += 4;
                    outptr7 += 4;
                }
#endif // __ALTIVEC__
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
        if (pack4to8)
        {
            const int total_bq = batch * outc;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bq = 0; bq < total_bq; bq++)
            {
                int b = bq / outc;
                int q = bq % outc;
                const float* r0 = bottom_blob.batch(b).channel(q * 2);
                const float* r1 = bottom_blob.batch(b).channel(q * 2 + 1);

                float* outptr = top_blob.batch(b).channel(q);

                for (int i = 0; i < size; i++)
                {
#if __ALTIVEC__
                    __ncnn_altivec_store(outptr, __ncnn_altivec_load(r0));
                    __ncnn_altivec_store(outptr + 4, __ncnn_altivec_load(r1));
#else
                    outptr[0] = r0[0];
                    outptr[1] = r0[1];
                    outptr[2] = r0[2];
                    outptr[3] = r0[3];
                    outptr[4] = r1[0];
                    outptr[5] = r1[1];
                    outptr[6] = r1[2];
                    outptr[7] = r1[3];
#endif

                    r0 += 4;
                    r1 += 4;
                    outptr += 8;
                }
            }
        }
        if (pack8to4)
        {
            const int total_bq = batch * channels;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int bq = 0; bq < total_bq; bq++)
            {
                int b = bq / channels;
                int q = bq % channels;
                const float* r0 = bottom_blob.batch(b).channel(q);

                float* outptr0 = top_blob.batch(b).channel(q * 2);
                float* outptr1 = top_blob.batch(b).channel(q * 2 + 1);

                for (int i = 0; i < size; i++)
                {
#if __ALTIVEC__
                    __ncnn_altivec_store(outptr0, __ncnn_altivec_load(r0));
                    __ncnn_altivec_store(outptr1, __ncnn_altivec_load(r0 + 4));
#else
                    outptr0[0] = r0[0];
                    outptr0[1] = r0[1];
                    outptr0[2] = r0[2];
                    outptr0[3] = r0[3];
                    outptr1[0] = r0[4];
                    outptr1[1] = r0[5];
                    outptr1[2] = r0[6];
                    outptr1[3] = r0[7];
#endif

                    r0 += 8;
                    outptr0 += 4;
                    outptr1 += 4;
                }
            }
        }

        return 0;
    }

    return 0;
}

} // namespace ncnn
