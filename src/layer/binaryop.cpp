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
    layout = pd.get(3, 0);

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
static int binary_op_whc(const Mat& a, const Mat& b, Mat& c, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int channels = a.c;
    size_t elemsize = a.elemsize;

    int w1 = b.w;
    int h1 = b.h;
    int channels1 = b.c;

    if (a.dims == 3)
    {
        if (b.dims == 3)
        {
            // type 1
            int ow = std::max(w, w1);
            int oh = std::max(h, h1);
            int oc = std::max(channels, channels1);
            
            c.create(ow, oh, oc, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

            int wf = w == ow ? -1 : 0;
            int hf = h == oh ? -1 : 0;
            int cf = channels == oc ? -1 : 0;
            int wf1 = w1 == ow ? -1 : 0;
            int hf1 = h1 == oh ? -1 : 0;
            int cf1 = channels1 == oc ? -1 : 0;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<oc; q++)
            {
                const float* ptr = a.channel(q & cf);
                const float* ptr1 = b.channel(q & cf1);
                float* outptr = c.channel(q);

                for (int i=0; i<oh; i++)
                {
                    for (int j=0; j<ow; j++)
                    {
                        outptr[j] = op(ptr[j & wf], ptr1[j & wf1]);
                    }
                    ptr += w & hf;
                    ptr1 += w1 & hf1;
                    outptr += ow;
                }
            }
            return 0;
        }

        if (b.dims == 2)
        {
            // type 2
            int ow = w;
            int oh = std::max(h, w1);
            int oc = std::max(channels, h1);
            
            c.create(ow, oh, oc, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

            int hf = h == oh ? -1 : 0;
            int cf = channels == oc ? -1 : 0;
            int wf1 = w1 == oh ? -1 : 0;
            int hf1 = h1 == oc ? -1 : 0;
            
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<oc; q++)
            {
                const float* ptr = a.channel(q & cf);
                const float* ptr1 = b.row(q & hf1);
                float* outptr = c.channel(q);

                for (int i=0; i<oh; i++)
                {
                    const float b0 = ptr1[i & wf1];
                    for (int j=0; j<ow; j++)
                    {
                        outptr[j] = op(ptr[j], b0);
                    }
                    ptr += w & hf;
                    outptr += ow;
                }
            }
            return 0;
        }

        if (b.dims == 1)
        {
            // type 3
            int ow = w;
            int oh = h;
            int oc = std::max(channels, w1);
            int size = ow * oh;
            
            c.create(ow, oh, oc, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

            int cf = channels == oc ? -1 : 0;
            int wf1 = w1 == oc ? -1 : 0;
            
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<oc; q++)
            {
                const float* ptr = a.channel(q & cf);
                const float b0 = b[q & wf1];
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
            // type 4
            int ow = w1;
            int oh = std::max(w, h1);
            int oc = std::max(h, channels1);
            
            c.create(ow, oh, oc, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

            int wf = w == oh ? -1 : 0;
            int hf = h == oc ? -1 : 0;
            int hf1 = h1 == oh ? -1 : 0;
            int cf1 = channels1 == oc ? -1 : 0;
            
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<oc; q++)
            {
                const float* ptr = a.row(q & hf);
                const float* ptr1 = b.channel(q & cf1);
                float* outptr = c.channel(q);

                for (int i=0; i<oh; i++)
                {
                    const float a0 = ptr[i & wf];
                    for (int j=0; j<ow; j++)
                    {
                        outptr[j] = op(a0, ptr1[j]);
                    }
                    ptr1 += w1 & hf1;
                    outptr += ow;
                }
            }
            return 0;
        }

        if (b.dims == 2)
        {
            // type 5
            int ow = std::max(w, w1);
            int oh = std::max(h, h1);
            
            c.create(ow, oh, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

            int wf = w == ow ? -1 : 0;
            int hf = h == oh ? -1 : 0;
            int wf1 = w1 == ow ? -1 : 0;
            int hf1 = h1 == oh ? -1 : 0;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i=0; i<oh; i++)
            {
                const float* ptr = a.row(i & hf);
                const float* ptr1 = b.row(i & hf1);
                float* outptr = c.row(i);
                for (int j=0; j<ow; j++)
                {
                    outptr[j] = op(ptr[j & wf], ptr1[j & wf1]);
                }
            }
            return 0;
        }

        if (b.dims == 1)
        {
            // type 6
            int ow = w;
            int oh = std::max(h, w1);
            
            c.create(ow, oh, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

            int hf = h == oh ? -1 : 0;
            int wf1 = w1 == oh ? -1 : 0;

            for (int i=0; i<oh; i++)
            {
                const float* ptr = a.row(i & hf);
                const float b0 = b[i & wf1];
                float* outptr = c.row(i);

                for (int j=0; j<ow; j++)
                {
                    outptr[j] = op(ptr[j], b0);
                }
            }
            return 0;
        }
    }
    else if (a.dims == 1)
    {
        if (b.dims == 3)
        {
            // type 7
            int ow = w1;
            int oh = h1;
            int oc = std::max(w, channels1);
            int size = ow * oh;
            
            c.create(ow, oh, oc, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;
            
            int wf = w == oc ? -1 : 0;
            int cf1 = channels1 == oc ? -1 : 0;
            
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<oc; q++)
            {
                const float a0 = a[q & wf];
                const float* ptr1 = b.channel(q & cf1);
                float* outptr = c.channel(q);
                for (int i=0; i<size; i++)
                {
                    outptr[i] = op(a0, ptr1[i]);
                }
            }
            return 0;
        }

        if (b.dims == 2)
        {
            // type 8
            int ow = w1;
            int oh = std::max(w, h1);

            c.create(ow, oh, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;
            
            int wf = w == oh ? -1 : 0;
            int hf1 = h1 == oh ? -1 : 0;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i=0; i<oh; i++)
            {
                const float a0 = a[i & wf];
                const float* ptr1 = b.row(i & hf1);
                float* outptr = c.row(i);
                for (int j=0; j<ow; j++)
                {
                    outptr[j] = op(a0, ptr1[j]);
                }
            }
            return 0;
        }

        if (b.dims == 1)
        {
            // type 9
            int ow = std::max(w, w1);

            c.create(ow, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

            int wf = w == ow ? -1 : 0;
            int wf1 = w1 == ow ? -1 : 0;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i=0; i<ow; i++)
            {
                c[i] = op(a[i & wf], b[i & wf1]);
            }
            return 0;
        }
    }
    return 0;
}

template<typename Op>
static int binary_op_chw(const Mat& a, const Mat& b, Mat& c, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int channels = a.c;
    size_t elemsize = a.elemsize;

    int w1 = b.w;
    int h1 = b.h;
    int channels1 = b.c;

    if (a.dims == 3)
    {
        if (b.dims == 3)
        {
            // type 1
            int ow = std::max(w, w1);
            int oh = std::max(h, h1);
            int oc = std::max(channels, channels1);
            
            c.create(ow, oh, oc, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

            int wf = w == ow ? -1 : 0;
            int hf = h == oh ? -1 : 0;
            int cf = channels == oc ? -1 : 0;
            int wf1 = w1 == ow ? -1 : 0;
            int hf1 = h1 == oh ? -1 : 0;
            int cf1 = channels1 == oc ? -1 : 0;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<oc; q++)
            {
                const float* ptr = a.channel(q & cf);
                const float* ptr1 = b.channel(q & cf1);
                float* outptr = c.channel(q);

                for (int i=0; i<oh; i++)
                {
                    for (int j=0; j<ow; j++)
                    {
                        outptr[j] = op(ptr[j & wf], ptr1[j & wf1]);
                    }
                    ptr += w & hf;
                    ptr1 += w1 & hf1;
                    outptr += ow;
                }
            }
            return 0;
        }

        if (b.dims == 2)
        {
            // type 2
            int ow = std::max(w, w1);
            int oh = std::max(h, h1);
            int oc = channels;
            
            c.create(ow, oh, oc, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

            int wf = w == ow ? -1 : 0;
            int hf = h == oh ? -1 : 0;
            int wf1 = w1 == ow ? -1 : 0;
            int hf1 = h1 == oh ? -1 : 0;
            
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<oc; q++)
            {
                const float* ptr = a.channel(q);
                float* outptr = c.channel(q);

                for (int i=0; i<oh; i++)
                {
                    const float* ptr1 = b.row(i & hf1);
                    for (int j=0; j<ow; j++)
                    {
                        outptr[j] = op(ptr[j & wf], ptr1[j & wf1]);
                    }
                    ptr += w & hf;
                    outptr += ow;
                }
            }
            return 0;
        }

        if (b.dims == 1)
        {
            // type 3
            int ow = std::max(w, w1);
            int oh = h;
            int oc = channels;
            
            c.create(ow, oh, oc, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

            int wf = w == ow ? -1 : 0;
            int wf1 = w1 == ow ? -1 : 0;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<oc; q++)
            {
                const float* ptr = a.channel(q);
                float* outptr = c.channel(q);
                for (int i=0; i<oh; i++)
                {
                    for (int j=0; j<ow; j++)
                    {
                        outptr[j] = op(ptr[j & wf], b[j & wf1]);
                    }
                    ptr += w;
                    outptr += ow;
                }
            }
            return 0;
        }
    }
    else if (a.dims == 2)
    {
        if (b.dims == 3)
        {
            // type 4
            int ow = std::max(w, w1);
            int oh = std::max(h, h1);
            int oc = channels1;
            
            c.create(ow, oh, oc, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

            int wf = w == ow ? -1 : 0;
            int hf = h == oh ? -1 : 0;
            int wf1 = w1 == ow ? -1 : 0;
            int hf1 = h1 == oh ? -1 : 0;
            
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<oc; q++)
            {
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);

                for (int i=0; i<oh; i++)
                {
                    const float* ptr = a.row(i & hf);
                    for (int j=0; j<ow; j++)
                    {
                        outptr[j] = op(ptr[j & wf], ptr1[j & wf1]);
                    }
                    ptr1 += w1 & hf1;
                    outptr += ow;
                }
            }
            return 0;
        }

        if (b.dims == 2)
        {
            // type 5
            int ow = std::max(w, w1);
            int oh = std::max(h, h1);
            
            c.create(ow, oh, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

            int wf = w == ow ? -1 : 0;
            int hf = h == oh ? -1 : 0;
            int wf1 = w1 == ow ? -1 : 0;
            int hf1 = h1 == oh ? -1 : 0;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i=0; i<oh; i++)
            {
                const float* ptr = a.row(i & hf);
                const float* ptr1 = b.row(i & hf1);
                float* outptr = c.row(i);
                for (int j=0; j<ow; j++)
                {
                    outptr[j] = op(ptr[j & wf], ptr1[j & wf1]);
                }
            }
            return 0;
        }

        if (b.dims == 1)
        {
            // type 6
            int ow = std::max(w, w1);
            int oh = h;
            
            c.create(ow, oh, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

            int wf = w == ow ? -1 : 0;
            int wf1 = w1 == ow ? -1 : 0;

            for (int i=0; i<oh; i++)
            {
                const float* ptr = a.row(i);
                float* outptr = c.row(i);

                for (int j=0; j<ow; j++)
                {
                    outptr[j] = op(ptr[j & wf], b[j & wf1]);
                }
            }
            return 0;
        }
    }
    else if (a.dims == 1)
    {
        if (b.dims == 3)
        {
            // type 7
            int ow = std::max(w, w1);
            int oh = h1;
            int oc = channels1;
            
            c.create(ow, oh, oc, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;
            
            int wf = w == ow ? -1 : 0;
            int wf1 = w1 == ow ? -1 : 0;
            
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q=0; q<oc; q++)
            {
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);
                for (int i=0; i<oh; i++)
                {
                    for (int j=0; j<ow; j++)
                    {
                        outptr[j] = op(a[j & wf], ptr1[j & wf1]);
                    }
                    ptr1 += w1;
                    outptr += ow;
                }
            }
            return 0;
        }

        if (b.dims == 2)
        {
            // type 8
            int ow = std::max(w, w1);
            int oh = h1;

            c.create(ow, oh, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;
            
            int wf = w == ow ? -1 : 0;
            int wf1 = w1 == ow ? -1 : 0;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i=0; i<oh; i++)
            {
                const float* ptr1 = b.row(i);
                float* outptr = c.row(i);
                for (int j=0; j<ow; j++)
                {
                    outptr[j] = op(a[j & wf], ptr1[j & wf1]);
                }
            }
            return 0;
        }

        if (b.dims == 1)
        {
            // type 9
            int ow = std::max(w, w1);

            c.create(ow, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

            int wf = w == ow ? -1 : 0;
            int wf1 = w1 == ow ? -1 : 0;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i=0; i<ow; i++)
            {
                c[i] = op(a[i & wf], b[i & wf1]);
            }
            return 0;
        }
    }
    return 0;
}

template<typename Op>
static int binary_op(const Mat& a, const Mat& b, Mat& c, int layout, const Option& opt)
{
    if (layout)
        return binary_op_chw<Op>(a, b, c, opt);
    else
        return binary_op_whc<Op>(a, b, c, opt);
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
        return binary_op< std::plus<float> >(bottom_blob, bottom_blob1, top_blob, layout, opt);

    if (op_type == Operation_SUB)
        return binary_op< std::minus<float> >(bottom_blob, bottom_blob1, top_blob, layout, opt);

    if (op_type == Operation_MUL)
        return binary_op< std::multiplies<float> >(bottom_blob, bottom_blob1, top_blob, layout, opt);

    if (op_type == Operation_DIV)
        return binary_op< std::divides<float> >(bottom_blob, bottom_blob1, top_blob, layout, opt);

    if (op_type == Operation_MAX)
        return binary_op< binary_op_max<float> >(bottom_blob, bottom_blob1, top_blob, layout, opt);

    if (op_type == Operation_MIN)
        return binary_op< binary_op_min<float> >(bottom_blob, bottom_blob1, top_blob, layout, opt);

    if (op_type == Operation_POW)
        return binary_op< binary_op_pow<float> >(bottom_blob, bottom_blob1, top_blob, layout, opt);

    if (op_type == Operation_RSUB)
        return binary_op< binary_op_rsub<float> >(bottom_blob, bottom_blob1, top_blob, layout, opt);

    if (op_type == Operation_RDIV)
        return binary_op< binary_op_rdiv<float> >(bottom_blob, bottom_blob1, top_blob, layout, opt);

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
