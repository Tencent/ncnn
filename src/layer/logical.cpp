// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "logical.h"

namespace ncnn {

Logical::Logical()
{
    one_blob_only = false;
    support_inplace = false;
}

int Logical::load_param(const ParamDict& pd)
{
    op_type = pd.get(0, 0);
    with_scalar = pd.get(1, 0);
    b = pd.get(2, 0);

    if (op_type == Operation_NOT || with_scalar != 0)
    {
        one_blob_only = true;
        support_inplace = true;
    }

    return 0;
}

template<typename Op>
static void logical_op_broadcast(const Mat& a, const Mat& b, Mat& c, const Option& opt)
{
    const Op op;

    const int dims = c.dims;
    const int w = c.w;
    const int h = c.h;
    const int d = c.d;
    const int channels = c.c;

    if (dims == 1)
    {
        const signed char* ptr = (const signed char*)a;
        const signed char* ptr1 = (const signed char*)b;
        signed char* outptr = (signed char*)c;

        const int ainc = a.w > 1 ? 1 : 0;
        const int binc = b.w > 1 ? 1 : 0;

        for (int x = 0; x < w; x++)
        {
            outptr[x] = op(*ptr != 0, *ptr1 != 0);
            ptr += ainc;
            ptr1 += binc;
        }
    }

    if (dims == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int y = 0; y < h; y++)
        {
            const signed char* ptr = (const signed char*)a.row(std::min(y, a.h - 1));
            const signed char* ptr1 = (const signed char*)b.row(std::min(y, b.h - 1));
            signed char* outptr = (signed char*)c.row(y);

            const int ainc = a.w > 1 ? 1 : 0;
            const int binc = b.w > 1 ? 1 : 0;

            for (int x = 0; x < w; x++)
            {
                outptr[x] = op(*ptr != 0, *ptr1 != 0);
                ptr += ainc;
                ptr1 += binc;
            }
        }
    }

    if (dims == 3 || dims == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            signed char* outptr = (signed char*)c.channel(q);

            const int ainc = a.w > 1 ? 1 : 0;
            const int binc = b.w > 1 ? 1 : 0;

            for (int z = 0; z < d; z++)
            {
                for (int y = 0; y < h; y++)
                {
                    const signed char* ptr = (const signed char*)a.channel(std::min(q, a.c - 1)).depth(std::min(z, a.d - 1)).row(std::min(y, a.h - 1));
                    const signed char* ptr1 = (const signed char*)b.channel(std::min(q, b.c - 1)).depth(std::min(z, b.d - 1)).row(std::min(y, b.h - 1));

                    for (int x = 0; x < w; x++)
                    {
                        outptr[x] = op(*ptr != 0, *ptr1 != 0);
                        ptr += ainc;
                        ptr1 += binc;
                    }

                    outptr += w;
                }
            }
        }
    }
}

template<typename Op>
static void logical_op_scalar_inplace(Mat& a, signed char b, const Option& opt)
{
    const Op op;

    const int dims = a.dims;
    const int w = a.w;
    const int h = a.h;
    const int d = a.d;
    const int channels = a.c;
    const int size = w * h * d;

    if (dims == 1)
    {
        signed char* ptr = (signed char*)a;
        for (int i = 0; i < w; i++)
        {
            ptr[i] = op(ptr[i] != 0, b != 0);
        }
    }

    if (dims == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int y = 0; y < h; y++)
        {
            signed char* ptr = (signed char*)a.row(y);
            for (int x = 0; x < w; x++)
            {
                ptr[x] = op(ptr[x] != 0, b != 0);
            }
        }
    }

    if (dims == 3 || dims == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            signed char* ptr = (signed char*)a.channel(q);
            for (int i = 0; i < size; i++)
            {
                ptr[i] = op(ptr[i] != 0, b != 0);
            }
        }
    }
}

struct logical_op_not
{
    signed char operator()(bool x, bool) const
    {
        return x ? 0 : 1;
    }
};

struct logical_op_and
{
    signed char operator()(bool x, bool y) const
    {
        return (x && y) ? 1 : 0;
    }
};

struct logical_op_or
{
    signed char operator()(bool x, bool y) const
    {
        return (x || y) ? 1 : 0;
    }
};

struct logical_op_xor
{
    signed char operator()(bool x, bool y) const
    {
        return (x != y) ? 1 : 0;
    }
};

static void logical_op_broadcast(const Mat& a, const Mat& b, Mat& c, int op_type, const Option& opt)
{
    if (op_type == Logical::Operation_AND) return logical_op_broadcast<logical_op_and>(a, b, c, opt);
    if (op_type == Logical::Operation_OR) return logical_op_broadcast<logical_op_or>(a, b, c, opt);
    if (op_type == Logical::Operation_XOR) return logical_op_broadcast<logical_op_xor>(a, b, c, opt);
}

static void logical_op_scalar_inplace(Mat& a, signed char b, int op_type, const Option& opt)
{
    if (op_type == Logical::Operation_NOT) return logical_op_scalar_inplace<logical_op_not>(a, b, opt);
    if (op_type == Logical::Operation_AND) return logical_op_scalar_inplace<logical_op_and>(a, b, opt);
    if (op_type == Logical::Operation_OR) return logical_op_scalar_inplace<logical_op_or>(a, b, opt);
    if (op_type == Logical::Operation_XOR) return logical_op_scalar_inplace<logical_op_xor>(a, b, opt);
}

int Logical::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& A = bottom_blobs[0];

    if (op_type == Operation_NOT)
    {
        Mat& top_blob = top_blobs[0];
        top_blob.create_like(A, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        top_blob = A.clone();
        logical_op_scalar_inplace(top_blob, b, op_type, opt);
        return 0;
    }

    int outdims = A.dims;

    Mat A2 = A;
    Mat B2;

    if (!with_scalar)
    {
        const Mat& B = bottom_blobs[1];
        B2 = B;
        outdims = std::max(outdims, B.dims);

        if (B.dims < outdims)
        {
            if (outdims == 2)
            {
                if (B.w == A.h)
                    B2 = B.reshape(1, B.w);
                else
                    B2 = B.reshape(B.w, 1);
            }
            if (outdims == 3 && B.dims == 1)
            {
                if (B.w == A.c)
                    B2 = B.reshape(1, 1, B.w);
                else
                    B2 = B.reshape(B.w, 1, 1);
            }
            if (outdims == 3 && B.dims == 2)
                B2 = B.reshape(1, B.w, B.h);
            if (outdims == 4 && B.dims == 1)
            {
                if (B.w == A.c)
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

    if (A.dims < outdims)
    {
        if (outdims == 2)
        {
            if (A.w == B2.h)
                A2 = A.reshape(1, A.w);
            else
                A2 = A.reshape(A.w, 1);
        }
        if (outdims == 3 && A.dims == 1)
        {
            if (A.w == B2.c)
                A2 = A.reshape(1, 1, A.w);
            else
                A2 = A.reshape(A.w, 1, 1);
        }
        if (outdims == 3 && A.dims == 2)
            A2 = A.reshape(1, A.w, A.h);
        if (outdims == 4 && A.dims == 1)
        {
            if (A.w == B2.c)
                A2 = A.reshape(1, 1, 1, A.w);
            else
                A2 = A.reshape(A.w, 1, 1, 1);
        }
        if (outdims == 4 && A.dims == 2)
            A2 = A.reshape(1, 1, A.w, A.h);
        if (outdims == 4 && A.dims == 3)
            A2 = A.reshape(1, A.w, A.h, A.c);
    }

    const int outw = std::max(A2.w, B2.w);
    const int outh = std::max(A2.h, B2.h);
    const int outd = std::max(A2.d, B2.d);
    const int outc = std::max(A2.c, B2.c);

    Mat& top_blob = top_blobs[0];
    if (outdims == 1)
    {
        top_blob.create(outw, (size_t)1u, opt.blob_allocator);
    }
    if (outdims == 2)
    {
        top_blob.create(outw, outh, (size_t)1u, opt.blob_allocator);
    }
    if (outdims == 3)
    {
        top_blob.create(outw, outh, outc, (size_t)1u, opt.blob_allocator);
    }
    if (outdims == 4)
    {
        top_blob.create(outw, outh, outd, outc, (size_t)1u, opt.blob_allocator);
    }

    if (top_blob.empty())
        return -100;

    if (with_scalar)
    {
        logical_op_scalar_inplace(A2, b, op_type, opt);
        top_blob = A2;
    }
    else
    {
        logical_op_broadcast(A2, B2, top_blob, op_type, opt);
    }

    return 0;
}

int Logical::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    logical_op_scalar_inplace(bottom_top_blob, b, op_type, opt);

    return 0;
}

} // namespace ncnn
