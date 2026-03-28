// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "where.h"

namespace ncnn {

Where::Where()
{
    one_blob_only = false;
    support_inplace = false;
}

int Where::load_param(const ParamDict& pd)
{
    with_scalar_a = pd.get(0, 0);
    a = pd.get(1, 0.f);
    with_scalar_b = pd.get(2, 0);
    b = pd.get(3, 0.f);

    return 0;
}

static void where_broadcast_scalar_a_scalar_b(const Mat& cond, float a, float b, Mat& c, const Option& opt)
{
    const int dims = c.dims;
    const int w = c.w;
    const int h = c.h;
    const int d = c.d;
    const int channels = c.c;

    if (dims == 1)
    {
        const signed char* cond_ptr = (const signed char*)cond;
        float* outptr = (float*)c.data;

        for (int x = 0; x < w; x++)
        {
            outptr[x] = *cond_ptr != 0 ? a : b;
            cond_ptr++;
        }
    }

    if (dims == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int y = 0; y < h; y++)
        {
            const signed char* cond_ptr = (const signed char*)cond.row(std::min(y, cond.h - 1));
            float* outptr = (float*)c.row(y);

            for (int x = 0; x < w; x++)
            {
                outptr[x] = *cond_ptr != 0 ? a : b;
                cond_ptr++;
            }
        }
    }

    if (dims == 3 || dims == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* outptr = (float*)c.channel(q);

            for (int z = 0; z < d; z++)
            {
                for (int y = 0; y < h; y++)
                {
                    const signed char* cond_ptr = (const signed char*)cond.channel(std::min(q, cond.c - 1)).depth(std::min(z, cond.d - 1)).row(std::min(y, cond.h - 1));

                    for (int x = 0; x < w; x++)
                    {
                        outptr[x] = *cond_ptr != 0 ? a : b;
                        cond_ptr++;
                    }

                    outptr += w;
                }
            }
        }
    }
}

static void where_broadcast_scalar_a(const Mat& cond, float a, const Mat& b, Mat& c, const Option& opt)
{
    const int dims = c.dims;
    const int w = c.w;
    const int h = c.h;
    const int d = c.d;
    const int channels = c.c;

    if (dims == 1)
    {
        const signed char* cond_ptr = (const signed char*)cond;
        const float* b_ptr = (const float*)b;
        float* outptr = (float*)c.data;

        const int b_inc = b.w > 1 ? 1 : 0;

        for (int x = 0; x < w; x++)
        {
            outptr[x] = *cond_ptr != 0 ? a : *b_ptr;
            cond_ptr++;
            b_ptr += b_inc;
        }
    }

    if (dims == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int y = 0; y < h; y++)
        {
            const signed char* cond_ptr = (const signed char*)cond.row(std::min(y, cond.h - 1));
            const float* b_ptr = (const float*)b.row(std::min(y, b.h - 1));
            float* outptr = (float*)c.row(y);

            const int b_inc = b.w > 1 ? 1 : 0;

            for (int x = 0; x < w; x++)
            {
                outptr[x] = *cond_ptr != 0 ? a : *b_ptr;
                cond_ptr++;
                b_ptr += b_inc;
            }
        }
    }

    if (dims == 3 || dims == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* outptr = (float*)c.channel(q);

            const int b_inc = b.w > 1 ? 1 : 0;

            for (int z = 0; z < d; z++)
            {
                for (int y = 0; y < h; y++)
                {
                    const signed char* cond_ptr = (const signed char*)cond.channel(std::min(q, cond.c - 1)).depth(std::min(z, cond.d - 1)).row(std::min(y, cond.h - 1));
                    const float* b_ptr = (const float*)b.channel(std::min(q, b.c - 1)).depth(std::min(z, b.d - 1)).row(std::min(y, b.h - 1));

                    for (int x = 0; x < w; x++)
                    {
                        outptr[x] = *cond_ptr != 0 ? a : *b_ptr;
                        cond_ptr++;
                        b_ptr += b_inc;
                    }

                    outptr += w;
                }
            }
        }
    }
}

static void where_broadcast_scalar_b(const Mat& cond, const Mat& a, float b, Mat& c, const Option& opt)
{
    const int dims = c.dims;
    const int w = c.w;
    const int h = c.h;
    const int d = c.d;
    const int channels = c.c;

    if (dims == 1)
    {
        const signed char* cond_ptr = (const signed char*)cond;
        const float* a_ptr = (const float*)a;
        float* outptr = (float*)c.data;

        const int a_inc = a.w > 1 ? 1 : 0;

        for (int x = 0; x < w; x++)
        {
            outptr[x] = *cond_ptr != 0 ? *a_ptr : b;
            cond_ptr++;
            a_ptr += a_inc;
        }
    }

    if (dims == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int y = 0; y < h; y++)
        {
            const signed char* cond_ptr = (const signed char*)cond.row(std::min(y, cond.h - 1));
            const float* a_ptr = (const float*)a.row(std::min(y, a.h - 1));
            float* outptr = (float*)c.row(y);

            const int a_inc = a.w > 1 ? 1 : 0;

            for (int x = 0; x < w; x++)
            {
                outptr[x] = *cond_ptr != 0 ? *a_ptr : b;
                cond_ptr++;
                a_ptr += a_inc;
            }
        }
    }

    if (dims == 3 || dims == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* outptr = (float*)c.channel(q);

            const int a_inc = a.w > 1 ? 1 : 0;

            for (int z = 0; z < d; z++)
            {
                for (int y = 0; y < h; y++)
                {
                    const signed char* cond_ptr = (const signed char*)cond.channel(std::min(q, cond.c - 1)).depth(std::min(z, cond.d - 1)).row(std::min(y, cond.h - 1));
                    const float* a_ptr = (const float*)a.channel(std::min(q, a.c - 1)).depth(std::min(z, a.d - 1)).row(std::min(y, a.h - 1));

                    for (int x = 0; x < w; x++)
                    {
                        outptr[x] = *cond_ptr != 0 ? *a_ptr : b;
                        cond_ptr++;
                        a_ptr += a_inc;
                    }

                    outptr += w;
                }
            }
        }
    }
}

static void where_broadcast(const Mat& cond, const Mat& a, const Mat& b, Mat& c, const Option& opt)
{
    const int dims = c.dims;
    const int w = c.w;
    const int h = c.h;
    const int d = c.d;
    const int channels = c.c;

    if (dims == 1)
    {
        const signed char* cond_ptr = (const signed char*)cond;
        const float* a_ptr = (const float*)a;
        const float* b_ptr = (const float*)b;
        float* outptr = (float*)c.data;

        const int a_inc = a.w > 1 ? 1 : 0;
        const int b_inc = b.w > 1 ? 1 : 0;

        for (int x = 0; x < w; x++)
        {
            outptr[x] = *cond_ptr != 0 ? *a_ptr : *b_ptr;
            cond_ptr++;
            a_ptr += a_inc;
            b_ptr += b_inc;
        }
    }

    if (dims == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int y = 0; y < h; y++)
        {
            const signed char* cond_ptr = (const signed char*)cond.row(std::min(y, cond.h - 1));
            const float* a_ptr = (const float*)a.row(std::min(y, a.h - 1));
            const float* b_ptr = (const float*)b.row(std::min(y, b.h - 1));
            float* outptr = (float*)c.row(y);

            const int a_inc = a.w > 1 ? 1 : 0;
            const int b_inc = b.w > 1 ? 1 : 0;

            for (int x = 0; x < w; x++)
            {
                outptr[x] = *cond_ptr != 0 ? *a_ptr : *b_ptr;
                cond_ptr++;
                a_ptr += a_inc;
                b_ptr += b_inc;
            }
        }
    }

    if (dims == 3 || dims == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* outptr = (float*)c.channel(q);

            const int a_inc = a.w > 1 ? 1 : 0;
            const int b_inc = b.w > 1 ? 1 : 0;

            for (int z = 0; z < d; z++)
            {
                for (int y = 0; y < h; y++)
                {
                    const signed char* cond_ptr = (const signed char*)cond.channel(std::min(q, cond.c - 1)).depth(std::min(z, cond.d - 1)).row(std::min(y, cond.h - 1));
                    const float* a_ptr = (const float*)a.channel(std::min(q, a.c - 1)).depth(std::min(z, a.d - 1)).row(std::min(y, a.h - 1));
                    const float* b_ptr = (const float*)b.channel(std::min(q, b.c - 1)).depth(std::min(z, b.d - 1)).row(std::min(y, b.h - 1));

                    for (int x = 0; x < w; x++)
                    {
                        outptr[x] = *cond_ptr != 0 ? *a_ptr : *b_ptr;
                        cond_ptr++;
                        a_ptr += a_inc;
                        b_ptr += b_inc;
                    }

                    outptr += w;
                }
            }
        }
    }
}

int Where::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& cond = bottom_blobs[0];

    int outdims = cond.dims;

    Mat cond2 = cond;
    Mat A2;
    Mat B2;

    if (!with_scalar_a)
    {
        const Mat& A = bottom_blobs[1];
        A2 = A;
        outdims = std::max(outdims, A.dims);

        if (A.dims < outdims)
        {
            if (outdims == 2)
            {
                if (A.w == cond.h)
                    A2 = A.reshape(1, A.w);
                else
                    A2 = A.reshape(A.w, 1);
            }
            if (outdims == 3 && A.dims == 1)
            {
                if (A.w == cond.c)
                    A2 = A.reshape(1, 1, A.w);
                else
                    A2 = A.reshape(A.w, 1, 1);
            }
            if (outdims == 3 && A.dims == 2)
                A2 = A.reshape(1, A.w, A.h);
            if (outdims == 4 && A.dims == 1)
            {
                if (A.w == cond.c)
                    A2 = A.reshape(1, 1, 1, A.w);
                else
                    A2 = A.reshape(A.w, 1, 1, 1);
            }
            if (outdims == 4 && A.dims == 2)
                A2 = A.reshape(1, 1, A.w, A.h);
            if (outdims == 4 && A.dims == 3)
                A2 = A.reshape(1, A.w, A.h, A.c);
        }
    }

    if (!with_scalar_b)
    {
        const Mat& B = with_scalar_a ? bottom_blobs[1] : bottom_blobs[2];
        B2 = B;
        outdims = std::max(outdims, B.dims);

        if (B.dims < outdims)
        {
            if (outdims == 2)
            {
                if (B.w == cond.h || (!with_scalar_a && B.w == A2.h))
                    B2 = B.reshape(1, B.w);
                else
                    B2 = B.reshape(B.w, 1);
            }
            if (outdims == 3 && B.dims == 1)
            {
                if (B.w == cond.c || (!with_scalar_a && B.w == A2.c))
                    B2 = B.reshape(1, 1, B.w);
                else
                    B2 = B.reshape(B.w, 1, 1);
            }
            if (outdims == 3 && B.dims == 2)
                B2 = B.reshape(1, B.w, B.h);
            if (outdims == 4 && B.dims == 1)
            {
                if (B.w == cond.c || (!with_scalar_a && B.w == A2.c))
                    B2 = B.reshape(1, 1, 1, B.w);
                else
                    B2 = B.reshape(B.w, 1, 1, 1);
            }
            if (outdims == 4 && B.dims == 2)
                B2 = B.reshape(1, 1, B.w, B.h);
            if (outdims == 4 && B.dims == 3)
                B2 = B.reshape(1, B.w, B.h, B.c);
        }
    }

    if (cond.dims < outdims)
    {
        if (outdims == 2)
        {
            if (cond.w == (with_scalar_a ? B2.h : A2.h))
                cond2 = cond.reshape(1, cond.w);
            else
                cond2 = cond.reshape(cond.w, 1);
        }
        if (outdims == 3 && cond.dims == 1)
        {
            if (cond.w == (with_scalar_a ? B2.c : A2.c))
                cond2 = cond.reshape(1, 1, cond.w);
            else
                cond2 = cond.reshape(cond.w, 1, 1);
        }
        if (outdims == 3 && cond.dims == 2)
            cond2 = cond.reshape(1, cond.w, cond.h);
        if (outdims == 4 && cond.dims == 1)
        {
            if (cond.w == (with_scalar_a ? B2.c : A2.c))
                cond2 = cond.reshape(1, 1, 1, cond.w);
            else
                cond2 = cond.reshape(cond.w, 1, 1, 1);
        }
        if (outdims == 4 && cond.dims == 2)
            cond2 = cond.reshape(1, 1, cond.w, cond.h);
        if (outdims == 4 && cond.dims == 3)
            cond2 = cond.reshape(1, cond.w, cond.h, cond.c);
    }

    int outw, outh, outd, outc;
    outw = cond2.w;
    outh = cond2.h;
    outd = cond2.d;
    outc = cond2.c;

    if (!with_scalar_a)
    {
        outw = std::max(outw, A2.w);
        outh = std::max(outh, A2.h);
        outd = std::max(outd, A2.d);
        outc = std::max(outc, A2.c);
    }

    if (!with_scalar_b)
    {
        outw = std::max(outw, B2.w);
        outh = std::max(outh, B2.h);
        outd = std::max(outd, B2.d);
        outc = std::max(outc, B2.c);
    }

    Mat& top_blob = top_blobs[0];
    if (outdims == 1) top_blob.create(outw, (size_t)4u, opt.blob_allocator);
    if (outdims == 2) top_blob.create(outw, outh, (size_t)4u, opt.blob_allocator);
    if (outdims == 3) top_blob.create(outw, outh, outc, (size_t)4u, opt.blob_allocator);
    if (outdims == 4) top_blob.create(outw, outh, outd, outc, (size_t)4u, opt.blob_allocator);

    if (top_blob.empty())
        return -100;

    if (with_scalar_a && with_scalar_b)
    {
        where_broadcast_scalar_a_scalar_b(cond2, a, b, top_blob, opt);
    }
    else if (with_scalar_a)
    {
        where_broadcast_scalar_a(cond2, a, B2, top_blob, opt);
    }
    else if (with_scalar_b)
    {
        where_broadcast_scalar_b(cond2, A2, b, top_blob, opt);
    }
    else
    {
        where_broadcast(cond2, A2, B2, top_blob, opt);
    }

    return 0;
}

} // namespace ncnn
