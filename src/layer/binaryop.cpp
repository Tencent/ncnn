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
#include <math.h>
#include <algorithm>
#include <functional>

namespace ncnn {

DEFINE_LAYER_CREATOR(BinaryOp)

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

template<typename Op>
static int binary_op(const Mat& a, const Mat& b, Mat& c, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int channels = a.c;
    int size = w * h;
    size_t elemsize = a.elemsize;

    int w1 = b.w;
    int h1 = b.h;
    int channels1 = b.c;
    int size1 = w1 * h1;

    if (a.dims == 3)
    {
        c.create(w, h, channels, elemsize, opt.blob_allocator);
        if (c.empty())
            return -100;

        if (b.dims == 3)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const float* ptr = a.channel(q);
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);

                for (int i=0; i<size; i++)
                {
                    outptr[i] = op(ptr[i], ptr1[i]);
                }
            }

            return 0;
        }

        if (b.dims == 2)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const float* ptr = a.channel(q);
                const float* ptr1 = (const float*)b + h * q;
                float* outptr = c.channel(q);

                for (int y=0; y<h; y++)
                {
                    const float b0 = ptr1[y];
                    for (int x=0; x<w; x++)
                    {
                        outptr[x] = op(ptr[x], b0);
                    }

                    ptr += w;
                    outptr += w;
                }
            }

            return 0;
        }

        if (b.dims == 1)
        {
            if (b.w == 1)
            {
                const float b0 = b[0];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q=0; q<channels; q++)
                {
                    const float* ptr = a.channel(q);
                    float* outptr = c.channel(q);

                    for (int i=0; i<size; i++)
                    {
                        outptr[i] = op(ptr[i], b0);
                    }
                }

                return 0;
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels; q++)
            {
                const float* ptr = a.channel(q);
                const float b0 = b[q];
                float* outptr = c.channel(q);

                for (int i=0; i<size; i++)
                {
                    outptr[i] = op(ptr[i], b0);
                }
            }

            return 0;
        }
    }
    else if (a.dims == 2)
    {
        if (b.dims == 3)
        {
            c.create(w1, h1, channels1, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels1; q++)
            {
                const float* ptr = (const float*)a + h1 * q;
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);

                for (int y=0; y<h1; y++)
                {
                    const float a0 = ptr[y];
                    for (int x=0; x<w1; x++)
                    {
                        outptr[x] = op(a0, ptr1[x]);
                    }

                    ptr1 += w1;
                    outptr += w1;
                }
            }

            return 0;
        }

        c.create(w, h, elemsize, opt.blob_allocator);
        if (c.empty())
            return -100;

        if (b.dims == 2)
        {
            for (int i=0; i<size; i++)
            {
                c[i] = op(a[i], b[i]);
            }

            return 0;
        }

        if (b.dims == 1)
        {
            c.create(w, h, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

            if (b.w == 1)
            {
                const float b0 = b[0];
                for (int i=0; i<size; i++)
                {
                    c[i] = op(a[i], b0);
                }

                return 0;
            }

            const float* ptr = a;
            float* outptr = c;

            for (int y=0; y<h; y++)
            {
                const float b0 = b[y];
                for (int x=0; x<w; x++)
                {
                    outptr[x] = op(ptr[x], b0);
                }

                ptr += w;
                outptr += w;
            }

            return 0;
        }
    }
    else if (a.dims == 1)
    {
        if (a.w == 1)
        {
            if (b.dims == 3)
            {
                c.create(w1, h1, channels1, elemsize, opt.blob_allocator);
                if (c.empty())
                    return -100;

                const float a0 = a[0];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q=0; q<channels1; q++)
                {
                    const float* ptr1 = b.channel(q);
                    float* outptr = c.channel(q);

                    for (int i=0; i<size1; i++)
                    {
                        outptr[i] = op(a0, ptr1[i]);
                    }
                }

                return 0;
            }

            if (b.dims == 2)
            {
                c.create(w1, h1, elemsize, opt.blob_allocator);
                if (c.empty())
                    return -100;

                const float a0 = a[0];
                for (int i=0; i<size1; i++)
                {
                    c[i] = op(a0, b[i]);
                }

                return 0;
            }

            if (b.dims == 1)
            {
                c.create(w1, elemsize, opt.blob_allocator);
                if (c.empty())
                    return -100;

                const float a0 = a[0];
                for (int i=0; i<size1; i++)
                {
                    c[i] = op(a0, b[i]);
                }

                return 0;
            }
        }

        if (b.dims == 3)
        {
            c.create(w1, h1, channels1, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<channels1; q++)
            {
                const float a0 = a[q];
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);

                for (int i=0; i<size1; i++)
                {
                    outptr[i] = op(a0, ptr1[i]);
                }
            }

            return 0;
        }

        if (b.dims == 2)
        {
            c.create(w1, h1, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

            const float* ptr1 = b;
            float* outptr = c;

            for (int y=0; y<h1; y++)
            {
                const float a0 = a[y];
                for (int x=0; x<w1; x++)
                {
                    outptr[x] = op(a0, ptr1[x]);
                }

                ptr1 += w1;
                outptr += w1;
            }

            return 0;
        }

        if (b.dims == 1)
        {
            c.create(w, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

            if (b.w == 1)
            {
                const float b0 = b[0];
                for (int i=0; i<size; i++)
                {
                    c[i] = op(a[i], b0);
                }

                return 0;
            }

            for (int i=0; i<size; i++)
            {
                c[i] = op(a[i], b[i]);
            }
        }
    }

    return 0;
}

template<typename Op>
static int binary_op_scalar_inplace(Mat& a, float b, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int channels = a.c;
    int size = w * h;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q=0; q<channels; q++)
    {
        float* ptr = a.channel(q);

        for (int i=0; i<size; i++)
        {
            ptr[i] = op(ptr[i], b);
        }
    }

    return 0;
}

template<typename T>
struct binary_op_max : std::binary_function<T,T,T> {
    T operator() (const T& x, const T& y) const { return std::max(x, y); }
};

template<typename T>
struct binary_op_min : std::binary_function<T,T,T> {
    T operator() (const T& x, const T& y) const { return std::min(x, y); }
};

template<typename T>
struct binary_op_pow : std::binary_function<T,T,T> {
    T operator() (const T& x, const T& y) const { return pow(x, y); }
};

template<typename T>
struct binary_op_rsub : std::binary_function<T,T,T> {
    T operator() (const T& x, const T& y) const { return y - x; }
};

template<typename T>
struct binary_op_rdiv : std::binary_function<T,T,T> {
    T operator() (const T& x, const T& y) const { return y / x; }
};

int BinaryOp::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& bottom_blob1 = bottom_blobs[1];

    Mat& top_blob = top_blobs[0];

    if (op_type == Operation_ADD)
        return binary_op< std::plus<float> >(bottom_blob, bottom_blob1, top_blob, opt);

    if (op_type == Operation_SUB)
        return binary_op< std::minus<float> >(bottom_blob, bottom_blob1, top_blob, opt);

    if (op_type == Operation_MUL)
        return binary_op< std::multiplies<float> >(bottom_blob, bottom_blob1, top_blob, opt);

    if (op_type == Operation_DIV)
        return binary_op< std::divides<float> >(bottom_blob, bottom_blob1, top_blob, opt);

    if (op_type == Operation_MAX)
        return binary_op< binary_op_max<float> >(bottom_blob, bottom_blob1, top_blob, opt);

    if (op_type == Operation_MIN)
        return binary_op< binary_op_min<float> >(bottom_blob, bottom_blob1, top_blob, opt);

    if (op_type == Operation_POW)
        return binary_op< binary_op_pow<float> >(bottom_blob, bottom_blob1, top_blob, opt);

    if (op_type == Operation_RSUB)
        return binary_op< binary_op_rsub<float> >(bottom_blob, bottom_blob1, top_blob, opt);

    if (op_type == Operation_RDIV)
        return binary_op< binary_op_rdiv<float> >(bottom_blob, bottom_blob1, top_blob, opt);

    return 0;
}

int BinaryOp::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    if (op_type == Operation_ADD)
        return binary_op_scalar_inplace< std::plus<float> >(bottom_top_blob, b, opt);

    if (op_type == Operation_SUB)
        return binary_op_scalar_inplace< std::minus<float> >(bottom_top_blob, b, opt);

    if (op_type == Operation_MUL)
        return binary_op_scalar_inplace< std::multiplies<float> >(bottom_top_blob, b, opt);

    if (op_type == Operation_DIV)
        return binary_op_scalar_inplace< std::divides<float> >(bottom_top_blob, b, opt);

    if (op_type == Operation_MAX)
        return binary_op_scalar_inplace< binary_op_max<float> >(bottom_top_blob, b, opt);

    if (op_type == Operation_MIN)
        return binary_op_scalar_inplace< binary_op_min<float> >(bottom_top_blob, b, opt);

    if (op_type == Operation_POW)
        return binary_op_scalar_inplace< binary_op_pow<float> >(bottom_top_blob, b, opt);

    if (op_type == Operation_RSUB)
        return binary_op_scalar_inplace< binary_op_rsub<float> >(bottom_top_blob, b, opt);

    if (op_type == Operation_RDIV)
        return binary_op_scalar_inplace< binary_op_rdiv<float> >(bottom_top_blob, b, opt);

    return 0;
}

} // namespace ncnn
