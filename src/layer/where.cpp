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
    return 0;
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

        const int cond_inc = cond.w > 1 ? 1 : 0;
        const int a_inc = a.w > 1 ? 1 : 0;
        const int b_inc = b.w > 1 ? 1 : 0;

        for (int x = 0; x < w; x++)
        {
            outptr[x] = *cond_ptr != 0 ? *a_ptr : *b_ptr;
            cond_ptr += cond_inc;
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

            const int cond_inc = cond.w > 1 ? 1 : 0;
            const int a_inc = a.w > 1 ? 1 : 0;
            const int b_inc = b.w > 1 ? 1 : 0;

            for (int x = 0; x < w; x++)
            {
                outptr[x] = *cond_ptr != 0 ? *a_ptr : *b_ptr;
                cond_ptr += cond_inc;
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

            const int cond_inc = cond.w > 1 ? 1 : 0;
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
                        cond_ptr += cond_inc;
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
    const Mat& A = bottom_blobs[1];
    const Mat& B = bottom_blobs[2];

    const int outdims = std::max(std::max(cond.dims, A.dims), B.dims);

    Mat cond2 = cond;
    Mat A2 = A;
    Mat B2 = B;

    if (cond.dims < outdims)
    {
        if (outdims == 2)
        {
            if (cond.w == A.h || cond.w == B.h)
                cond2 = cond.reshape(1, cond.w);
            else
                cond2 = cond.reshape(cond.w, 1);
        }
        if (outdims == 3 && cond.dims == 1)
        {
            if (cond.w == A.c || cond.w == B.c)
                cond2 = cond.reshape(1, 1, cond.w);
            else
                cond2 = cond.reshape(cond.w, 1, 1);
        }
        if (outdims == 3 && cond.dims == 2)
            cond2 = cond.reshape(1, cond.w, cond.h);
        if (outdims == 4 && cond.dims == 1)
        {
            if (cond.w == A.c || cond.w == B.c)
                cond2 = cond.reshape(1, 1, 1, cond.w);
            else
                cond2 = cond.reshape(cond.w, 1, 1, 1);
        }
        if (outdims == 4 && cond.dims == 2)
            cond2 = cond.reshape(1, 1, cond.w, cond.h);
        if (outdims == 4 && cond.dims == 3)
            cond2 = cond.reshape(1, cond.w, cond.h, cond.c);
    }

    if (A.dims < outdims)
    {
        if (outdims == 2)
        {
            if (A.w == cond.h || A.w == B.h)
                A2 = A.reshape(1, A.w);
            else
                A2 = A.reshape(A.w, 1);
        }
        if (outdims == 3 && A.dims == 1)
        {
            if (A.w == cond.c || A.w == B.c)
                A2 = A.reshape(1, 1, A.w);
            else
                A2 = A.reshape(A.w, 1, 1);
        }
        if (outdims == 3 && A.dims == 2)
            A2 = A.reshape(1, A.w, A.h);
        if (outdims == 4 && A.dims == 1)
        {
            if (A.w == cond.c || A.w == B.c)
                A2 = A.reshape(1, 1, 1, A.w);
            else
                A2 = A.reshape(A.w, 1, 1, 1);
        }
        if (outdims == 4 && A.dims == 2)
            A2 = A.reshape(1, 1, A.w, A.h);
        if (outdims == 4 && A.dims == 3)
            A2 = A.reshape(1, A.w, A.h, A.c);
    }

    if (B.dims < outdims)
    {
        if (outdims == 2)
        {
            if (B.w == cond.h || B.w == A.h)
                B2 = B.reshape(1, B.w);
            else
                B2 = B.reshape(B.w, 1);
        }
        if (outdims == 3 && B.dims == 1)
        {
            if (B.w == cond.c || B.w == A.c)
                B2 = B.reshape(1, 1, B.w);
            else
                B2 = B.reshape(B.w, 1, 1);
        }
        if (outdims == 3 && B.dims == 2)
            B2 = B.reshape(1, B.w, B.h);
        if (outdims == 4 && B.dims == 1)
        {
            if (B.w == cond.c || B.w == A.c)
                B2 = B.reshape(1, 1, 1, B.w);
            else
                B2 = B.reshape(B.w, 1, 1, 1);
        }
        if (outdims == 4 && B.dims == 2)
            B2 = B.reshape(1, 1, B.w, B.h);
        if (outdims == 4 && B.dims == 3)
            B2 = B.reshape(1, B.w, B.h, B.c);
    }

    const int outw = std::max(std::max(cond2.w, A2.w), B2.w);
    const int outh = std::max(std::max(cond2.h, A2.h), B2.h);
    const int outd = std::max(std::max(cond2.d, A2.d), B2.d);
    const int outc = std::max(std::max(cond2.c, A2.c), B2.c);

    Mat& top_blob = top_blobs[0];
    if (outdims == 1) top_blob.create(outw, (size_t)4u, opt.blob_allocator);
    if (outdims == 2) top_blob.create(outw, outh, (size_t)4u, opt.blob_allocator);
    if (outdims == 3) top_blob.create(outw, outh, outc, (size_t)4u, opt.blob_allocator);
    if (outdims == 4) top_blob.create(outw, outh, outd, outc, (size_t)4u, opt.blob_allocator);

    if (top_blob.empty())
        return -100;

    where_broadcast(cond2, A2, B2, top_blob, opt);

    return 0;
}

} // namespace ncnn
