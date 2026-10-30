// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "compareop.h"

namespace ncnn {

CompareOp::CompareOp()
{
    one_blob_only = false;
    support_inplace = false;
}

int CompareOp::load_param(const ParamDict& pd)
{
    op_type = pd.get(0, 0);
    with_scalar = pd.get(1, 0);
    b = pd.get(2, 0.f);

    if (with_scalar != 0)
    {
        one_blob_only = true;
    }

    return 0;
}

template<typename Op>
static void compare_op_broadcast(const Mat& a, const Mat& b, Mat& c, const Option& opt)
{
    const Op op;

    const int dims = c.dims;
    const int w = c.w;
    const int h = c.h;
    const int d = c.d;
    const int channels = c.c;

    if (dims == 1)
    {
        const float* ptr = a;
        const float* ptr1 = b;
        signed char* outptr = (signed char*)c.data;

        const int ainc = a.w > 1 ? 1 : 0;
        const int binc = b.w > 1 ? 1 : 0;

        for (int x = 0; x < w; x++)
        {
            outptr[x] = op(*ptr, *ptr1);
            ptr += ainc;
            ptr1 += binc;
        }
    }

    if (dims == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int y = 0; y < h; y++)
        {
            const float* ptr = a.row(std::min(y, a.h - 1));
            const float* ptr1 = b.row(std::min(y, b.h - 1));
            signed char* outptr = (signed char*)c.row(y);

            const int ainc = a.w > 1 ? 1 : 0;
            const int binc = b.w > 1 ? 1 : 0;

            for (int x = 0; x < w; x++)
            {
                outptr[x] = op(*ptr, *ptr1);
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
                    const float* ptr = a.channel(std::min(q, a.c - 1)).depth(std::min(z, a.d - 1)).row(std::min(y, a.h - 1));
                    const float* ptr1 = b.channel(std::min(q, b.c - 1)).depth(std::min(z, b.d - 1)).row(std::min(y, b.h - 1));

                    for (int x = 0; x < w; x++)
                    {
                        outptr[x] = op(*ptr, *ptr1);
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
static void compare_op_scalar(const Mat& a, float b, Mat& c, const Option& opt)
{
    const Op op;

    const int channels = a.c;
    const int size = a.w * a.h * a.d;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* ptr = a.channel(q);
        signed char* outptr = (signed char*)c.channel(q);

        for (int i = 0; i < size; i++)
        {
            outptr[i] = op(ptr[i], b);
        }
    }
}

struct compare_op_lt
{
    signed char operator()(const float& x, const float& y) const
    {
        return x < y ? 1 : 0;
    }
};
struct compare_op_gt
{
    signed char operator()(const float& x, const float& y) const
    {
        return x > y ? 1 : 0;
    }
};
struct compare_op_le
{
    signed char operator()(const float& x, const float& y) const
    {
        return x <= y ? 1 : 0;
    }
};
struct compare_op_ge
{
    signed char operator()(const float& x, const float& y) const
    {
        return x >= y ? 1 : 0;
    }
};
struct compare_op_eq
{
    signed char operator()(const float& x, const float& y) const
    {
        return x == y ? 1 : 0;
    }
};
struct compare_op_ne
{
    signed char operator()(const float& x, const float& y) const
    {
        return x != y ? 1 : 0;
    }
};

static void compare_op_broadcast(const Mat& a, const Mat& b, Mat& c, int op_type, const Option& opt)
{
    if (op_type == CompareOp::Operation_LT) return compare_op_broadcast<compare_op_lt>(a, b, c, opt);
    if (op_type == CompareOp::Operation_GT) return compare_op_broadcast<compare_op_gt>(a, b, c, opt);
    if (op_type == CompareOp::Operation_LE) return compare_op_broadcast<compare_op_le>(a, b, c, opt);
    if (op_type == CompareOp::Operation_GE) return compare_op_broadcast<compare_op_ge>(a, b, c, opt);
    if (op_type == CompareOp::Operation_EQ) return compare_op_broadcast<compare_op_eq>(a, b, c, opt);
    if (op_type == CompareOp::Operation_NE) return compare_op_broadcast<compare_op_ne>(a, b, c, opt);
}

static void compare_op_scalar(const Mat& a, float b, Mat& c, int op_type, const Option& opt)
{
    if (op_type == CompareOp::Operation_LT) return compare_op_scalar<compare_op_lt>(a, b, c, opt);
    if (op_type == CompareOp::Operation_GT) return compare_op_scalar<compare_op_gt>(a, b, c, opt);
    if (op_type == CompareOp::Operation_LE) return compare_op_scalar<compare_op_le>(a, b, c, opt);
    if (op_type == CompareOp::Operation_GE) return compare_op_scalar<compare_op_ge>(a, b, c, opt);
    if (op_type == CompareOp::Operation_EQ) return compare_op_scalar<compare_op_eq>(a, b, c, opt);
    if (op_type == CompareOp::Operation_NE) return compare_op_scalar<compare_op_ne>(a, b, c, opt);
}

int CompareOp::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (bottom_blob.dims == 1) top_blob.create(bottom_blob.w, (size_t)1u, 1, opt.blob_allocator);
    if (bottom_blob.dims == 2) top_blob.create(bottom_blob.w, bottom_blob.h, (size_t)1u, 1, opt.blob_allocator);
    if (bottom_blob.dims == 3) top_blob.create(bottom_blob.w, bottom_blob.h, bottom_blob.c, (size_t)1u, 1, opt.blob_allocator);
    if (bottom_blob.dims == 4) top_blob.create(bottom_blob.w, bottom_blob.h, bottom_blob.d, bottom_blob.c, (size_t)1u, 1, opt.blob_allocator);

    if (top_blob.empty())
        return -100;

    compare_op_scalar(bottom_blob, b, top_blob, op_type, opt);
    return 0;
}

int CompareOp::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    if (with_scalar)
    {
        const Mat& A = bottom_blobs[0];
        Mat& top_blob = top_blobs[0];

        if (A.dims == 1) top_blob.create(A.w, (size_t)1u, 1, opt.blob_allocator);
        if (A.dims == 2) top_blob.create(A.w, A.h, (size_t)1u, 1, opt.blob_allocator);
        if (A.dims == 3) top_blob.create(A.w, A.h, A.c, (size_t)1u, 1, opt.blob_allocator);
        if (A.dims == 4) top_blob.create(A.w, A.h, A.d, A.c, (size_t)1u, 1, opt.blob_allocator);

        if (top_blob.empty())
            return -100;

        compare_op_scalar(A, b, top_blob, op_type, opt);
        return 0;
    }

    const Mat& A = bottom_blobs[0];
    const Mat& B = bottom_blobs[1];
    const int outdims = std::max(A.dims, B.dims);

    Mat A2 = A;
    Mat B2 = B;
    if (A.dims < outdims)
    {
        // expand inner axes
        if (outdims == 2)
        {
            if (A.w == B.h)
                A2 = A.reshape(1, A.w);
            else
                A2 = A.reshape(A.w, 1);
        }
        if (outdims == 3 && A.dims == 1)
        {
            if (A.w == B.c)
                A2 = A.reshape(1, 1, A.w);
            else
                A2 = A.reshape(A.w, 1, 1);
        }
        if (outdims == 3 && A.dims == 2)
            A2 = A.reshape(1, A.w, A.h);
        if (outdims == 4 && A.dims == 1)
        {
            if (A.w == B.c)
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
        // expand inner axes
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

    const int outw = std::max(A2.w, B2.w);
    const int outh = std::max(A2.h, B2.h);
    const int outd = std::max(A2.d, B2.d);
    const int outc = std::max(A2.c, B2.c);

    Mat& top_blob = top_blobs[0];
    if (outdims == 1) top_blob.create(outw, (size_t)1u, 1, opt.blob_allocator);
    if (outdims == 2) top_blob.create(outw, outh, (size_t)1u, 1, opt.blob_allocator);
    if (outdims == 3) top_blob.create(outw, outh, outc, (size_t)1u, 1, opt.blob_allocator);
    if (outdims == 4) top_blob.create(outw, outh, outd, outc, (size_t)1u, 1, opt.blob_allocator);

    if (top_blob.empty())
        return -100;

    compare_op_broadcast(A2, B2, top_blob, op_type, opt);

    return 0;
}

} // namespace ncnn