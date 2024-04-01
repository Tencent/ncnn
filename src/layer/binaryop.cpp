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

#include "binaryop.h"

namespace ncnn {

BinaryOp::BinaryOp()
{
    one_blob_only = false;
    support_inplace = false;
}

int BinaryOp::load_param(const ParamDict& pd)
{
    op_type = pd.get(0, 0);
    with_scalar = pd.get(1, 0);
    b = pd.get(2, 0.f);

    if (with_scalar != 0)
    {
        one_blob_only = true;
        support_inplace = true;
    }

    return 0;
}

// broadcasting rule
// https://github.com/Tencent/ncnn/wiki/binaryop-broadcasting

template<typename Op>
static void binary_op_broadcast(const Mat& a, const Mat& b, Mat& c, const Option& opt)
{
    // general broadcast
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
        float* outptr = c;

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
            float* outptr = c.row(y);

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
            float* outptr = c.channel(q);

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
static void binary_op_scalar_inplace(Mat& a, float b, const Option& opt)
{
    const Op op;

    const int channels = a.c;
    const int size = a.w * a.h * a.d;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = a.channel(q);

        for (int i = 0; i < size; i++)
        {
            ptr[i] = op(ptr[i], b);
        }
    }
}

struct binary_op_add
{
    float operator()(const float& x, const float& y) const
    {
        return x + y;
    }
};

struct binary_op_sub
{
    float operator()(const float& x, const float& y) const
    {
        return x - y;
    }
};

struct binary_op_mul
{
    float operator()(const float& x, const float& y) const
    {
        return x * y;
    }
};

struct binary_op_div
{
    float operator()(const float& x, const float& y) const
    {
        return x / y;
    }
};

struct binary_op_max
{
    float operator()(const float& x, const float& y) const
    {
        return std::max(x, y);
    }
};

struct binary_op_min
{
    float operator()(const float& x, const float& y) const
    {
        return std::min(x, y);
    }
};

struct binary_op_pow
{
    float operator()(const float& x, const float& y) const
    {
        return (float)powf(x, y);
    }
};

struct binary_op_rsub
{
    float operator()(const float& x, const float& y) const
    {
        return y - x;
    }
};

struct binary_op_rdiv
{
    float operator()(const float& x, const float& y) const
    {
        return y / x;
    }
};

struct binary_op_rpow
{
    float operator()(const float& x, const float& y) const
    {
        return (float)powf(y, x);
    }
};

struct binary_op_atan2
{
    float operator()(const float& x, const float& y) const
    {
        return (float)atan2f(x, y);
    }
};

struct binary_op_ratan2
{
    float operator()(const float& x, const float& y) const
    {
        return (float)atan2f(y, x);
    }
};

static void binary_op_broadcast(const Mat& a, const Mat& b, Mat& c, int op_type, const Option& opt)
{
    if (op_type == BinaryOp::Operation_ADD) return binary_op_broadcast<binary_op_add>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_SUB) return binary_op_broadcast<binary_op_sub>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_MUL) return binary_op_broadcast<binary_op_mul>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_DIV) return binary_op_broadcast<binary_op_div>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_MAX) return binary_op_broadcast<binary_op_max>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_MIN) return binary_op_broadcast<binary_op_min>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_POW) return binary_op_broadcast<binary_op_pow>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_RSUB) return binary_op_broadcast<binary_op_sub>(b, a, c, opt);
    if (op_type == BinaryOp::Operation_RDIV) return binary_op_broadcast<binary_op_div>(b, a, c, opt);
    if (op_type == BinaryOp::Operation_RPOW) return binary_op_broadcast<binary_op_pow>(b, a, c, opt);
    if (op_type == BinaryOp::Operation_ATAN2) return binary_op_broadcast<binary_op_atan2>(a, b, c, opt);
    if (op_type == BinaryOp::Operation_RATAN2) return binary_op_broadcast<binary_op_atan2>(b, a, c, opt);

    // should never reach here
}

static void binary_op_scalar_inplace(Mat& bottom_top_blob, float b, int op_type, const Option& opt)
{
    if (op_type == BinaryOp::Operation_ADD) return binary_op_scalar_inplace<binary_op_add>(bottom_top_blob, b, opt);
    if (op_type == BinaryOp::Operation_SUB) return binary_op_scalar_inplace<binary_op_sub>(bottom_top_blob, b, opt);
    if (op_type == BinaryOp::Operation_MUL) return binary_op_scalar_inplace<binary_op_mul>(bottom_top_blob, b, opt);
    if (op_type == BinaryOp::Operation_DIV) return binary_op_scalar_inplace<binary_op_div>(bottom_top_blob, b, opt);
    if (op_type == BinaryOp::Operation_MAX) return binary_op_scalar_inplace<binary_op_max>(bottom_top_blob, b, opt);
    if (op_type == BinaryOp::Operation_MIN) return binary_op_scalar_inplace<binary_op_min>(bottom_top_blob, b, opt);
    if (op_type == BinaryOp::Operation_POW) return binary_op_scalar_inplace<binary_op_pow>(bottom_top_blob, b, opt);
    if (op_type == BinaryOp::Operation_RSUB) return binary_op_scalar_inplace<binary_op_rsub>(bottom_top_blob, b, opt);
    if (op_type == BinaryOp::Operation_RDIV) return binary_op_scalar_inplace<binary_op_rdiv>(bottom_top_blob, b, opt);
    if (op_type == BinaryOp::Operation_RPOW) return binary_op_scalar_inplace<binary_op_rpow>(bottom_top_blob, b, opt);
    if (op_type == BinaryOp::Operation_ATAN2) return binary_op_scalar_inplace<binary_op_atan2>(bottom_top_blob, b, opt);
    if (op_type == BinaryOp::Operation_RATAN2) return binary_op_scalar_inplace<binary_op_ratan2>(bottom_top_blob, b, opt);

    // should never reach here
}

int BinaryOp::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
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
            else // if (A.w == B.w)
                A2 = A.reshape(A.w, 1);
        }
        if (outdims == 3 && A.dims == 1)
        {
            if (A.w == B.c)
                A2 = A.reshape(1, 1, A.w);
            else // if (A.w == B.w)
                A2 = A.reshape(A.w, 1, 1);
        }
        if (outdims == 3 && A.dims == 2)
            A2 = A.reshape(1, A.w, A.h);
        if (outdims == 4 && A.dims == 1)
        {
            if (A.w == B.c)
                A2 = A.reshape(1, 1, 1, A.w);
            else // if (A.w == B.w)
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
            else // if (B.w == A.w)
                B2 = B.reshape(B.w, 1);
        }
        if (outdims == 3 && B.dims == 1)
        {
            if (B.w == A.c)
                B2 = B.reshape(1, 1, B.w);
            else // if (B.w == A.w)
                B2 = B.reshape(B.w, 1, 1);
        }
        if (outdims == 3 && B.dims == 2)
            B2 = B.reshape(1, B.w, B.h);
        if (outdims == 4 && B.dims == 1)
        {
            if (B.w == A.c)
                B2 = B.reshape(1, 1, 1, B.w);
            else // if (B.w == A.w)
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
    if (outdims == 1)
    {
        top_blob.create(outw, 4u, opt.blob_allocator);
    }
    if (outdims == 2)
    {
        top_blob.create(outw, outh, 4u, opt.blob_allocator);
    }
    if (outdims == 3)
    {
        top_blob.create(outw, outh, outc, 4u, opt.blob_allocator);
    }
    if (outdims == 4)
    {
        top_blob.create(outw, outh, outd, outc, 4u, opt.blob_allocator);
    }
    if (top_blob.empty())
        return -100;

    binary_op_broadcast(A2, B2, top_blob, op_type, opt);

    return 0;
}

int BinaryOp::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    binary_op_scalar_inplace(bottom_top_blob, b, op_type, opt);

    return 0;
}

} // namespace ncnn
