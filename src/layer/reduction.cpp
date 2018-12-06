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

#include "reduction.h"
#include <float.h>
#include <limits.h>
#include <math.h>
#include <algorithm>
#include <functional>

namespace ncnn {

DEFINE_LAYER_CREATOR(Reduction)

Reduction::Reduction()
{
    one_blob_only = true;
    support_inplace = false;
}

int Reduction::load_param(const ParamDict& pd)
{
    operation = pd.get(0, 0);
    dim = pd.get(1, 0);
    coeff = pd.get(2, 1.f);

    return 0;
}

template<typename Op, typename Op2>
static int reduction_op(const Mat& a, Mat& b, float v0, int dim, float coeff, const Option& opt)
{
    Op op;
    Op2 op2;

    int w = a.w;
    int h = a.h;
    int channels = a.c;
    size_t elemsize = a.elemsize;
    int size = w * h;

    if (dim == 0)
    {
        // w h c -> X X X
        b.create(1, elemsize, opt.blob_allocator);
    }
    else if (dim == 1)
    {
        // w h c -> X X c
        b.create(channels, elemsize, opt.blob_allocator);
    }
    else if (dim == 2)
    {
        // w h c -> X h c
        b.create(h, channels, elemsize, opt.blob_allocator);
    }
    else if (dim == -1)
    {
        // w h c -> w X X
        b.create(w, elemsize, opt.blob_allocator);
    }
    else if (dim == -2)
    {
        // w h c -> w h X
        b.create(w, h, elemsize, opt.blob_allocator);
    }
    if (b.empty())
        return -100;

    if (dim == 0)
    {
        Mat sums(channels, elemsize, opt.workspace_allocator);
        if (sums.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            const float* ptr = a.channel(q);

            float sum = v0;
            for (int i=0; i<size; i++)
            {
                sum = op(sum, ptr[i]);
            }

            sums[q] = sum;
        }

        float sum = v0;
        for (int i=0; i<channels; i++)
        {
            sum = op2(sum, sums[i]);
        }

        b[0] = sum * coeff;
    }
    else if (dim == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            const float* ptr = a.channel(q);

            float sum = v0;
            for (int i=0; i<size; i++)
            {
                sum = op(sum, ptr[i]);
            }

            b[q] = sum * coeff;
        }
    }
    else if (dim == 2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            const float* ptr = a.channel(q);
            float* outptr = b.row(q);

            for (int i=0; i<h; i++)
            {
                float sum = v0;
                for (int j=0; j<w; j++)
                {
                    sum = op(sum, ptr[i]);
                }

                outptr[i] = sum * coeff;

                ptr += w;
            }
        }
    }
    else if (dim == -1)
    {
        Mat mins(w, 1, channels, elemsize, opt.workspace_allocator);
        if (mins.empty())
            return -100;

        mins.fill(v0);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<channels; q++)
        {
            const float* ptr = a.channel(q);
            float* mins_ptr = mins.channel(q);

            for (int i=0; i<h; i++)
            {
                for (int j=0; j<w; j++)
                {
                    mins_ptr[j] = op(mins_ptr[j], ptr[i]);
                }

                ptr += w;
            }
        }

        b.fill(v0);

        for (int q=0; q<channels; q++)
        {
            const float* mins_ptr = mins.channel(q);
            for (int j=0; j<w; j++)
            {
                b[j] = op2(b[j], mins_ptr[j]);
            }
        }

        for (int j=0; j<w; j++)
        {
            b[j] *= coeff;
        }
    }
    else if (dim == -2)
    {
        b.fill(v0);

        for (int q=0; q<channels; q++)
        {
            const float* ptr = a.channel(q);

            for (int i=0; i<size; i++)
            {
                b[i] = op(b[i], ptr[i]);
            }
        }

        for (int i=0; i<size; i++)
        {
            b[i] *= coeff;
        }
    }

    return 0;
}

template<typename T>
struct reduction_op_asum : std::binary_function<T,T,T> {
    T operator() (const T& x, const T& y) const { return x + fabs(y); }
};

template<typename T>
struct reduction_op_sumsq : std::binary_function<T,T,T> {
    T operator() (const T& x, const T& y) const { return x + y * y; }
};

template<typename T>
struct reduction_op_max : std::binary_function<T,T,T> {
    T operator() (const T& x, const T& y) const { return std::max(x, y); }
};

template<typename T>
struct reduction_op_min : std::binary_function<T,T,T> {
    T operator() (const T& x, const T& y) const { return std::min(x, y); }
};

int Reduction::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (operation == ReductionOp_SUM)
        return reduction_op< std::plus<float>, std::plus<float> >(bottom_blob, top_blob, 0.f, dim, coeff, opt);

    if (operation == ReductionOp_ASUM)
        return reduction_op< reduction_op_asum<float>, std::plus<float> >(bottom_blob, top_blob, 0.f, dim, coeff, opt);

    if (operation == ReductionOp_SUMSQ)
        return reduction_op< reduction_op_sumsq<float>, std::plus<float> >(bottom_blob, top_blob, 0.f, dim, coeff, opt);

    if (operation == ReductionOp_MEAN)
    {
        int ret = reduction_op< std::plus<float>, std::plus<float> >(bottom_blob, top_blob, 0.f, dim, coeff, opt);
        if (ret != 0)
            return -100;

        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;
        int size = w * h;

        if (dim == 0)
        {
            top_blob[0] /= channels * size;
        }
        else if (dim == 1)
        {
            for (int q=0; q<channels; q++)
            {
                top_blob[q] /= size;
            }
        }
        else if (dim == 2)
        {
            for (int q=0; q<channels; q++)
            {
                float* outptr = top_blob.channel(q);

                for (int i=0; i<h; i++)
                {
                    outptr[i] /= w;
                }
            }
        }
        else if (dim == -1)
        {
            for (int j=0; j<w; j++)
            {
                top_blob[j] /= h * channels;
            }
        }
        else if (dim == -2)
        {
            for (int i=0; i<size; i++)
            {
                top_blob[i] /= channels;
            }
        }
    }

    if (operation == ReductionOp_MAX)
        return reduction_op< reduction_op_max<float>, reduction_op_max<float> >(bottom_blob, top_blob, -FLT_MAX, dim, coeff, opt);

    if (operation == ReductionOp_MIN)
        return reduction_op< reduction_op_min<float>, reduction_op_min<float> >(bottom_blob, top_blob, FLT_MAX, dim, coeff, opt);

    if (operation == ReductionOp_PROD)
        return reduction_op< std::multiplies<float>, std::multiplies<float> >(bottom_blob, top_blob, 1.f, dim, coeff, opt);

    return 0;
}

} // namespace ncnn
