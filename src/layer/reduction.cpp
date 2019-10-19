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
    type = pd.get(1, 0);
    coeff = pd.get(2, 1.f);
    axes = pd.get(3, Mat());
    keepdims = pd.get(4, 0);

    return 0;
}

template<typename MathOp>
static int post_process(Mat& a, const Option& opt)
{
    MathOp mathop;

    int dims = a.dims;
    if (dims == 1)
    {
        int w = a.w;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i=0; i<w; i++)
            a[i] = mathop(a[i]);
    }
    else if (dims == 2)
    {
        int size = a.w * a.h;
        
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i=0; i<size; i++)
            a[i] = mathop(a[i]);
    }
    else if (dims == 3)
    {
        int c = a.c;
        int size = a.w * a.h;
        if(c == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i=0; i<size; i++)
                a[i] = mathop(a[i]);
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for(int q=0; q<c; q++)
            {
                float* outptr = a.channel(q);
                for (int i=0; i<size; i++)
                    outptr[i] = mathop(outptr[i]);
            }
        }
    }
    
    return 0;
}

template<typename Op, typename Op2>
static int reduction_op(const Mat& a, Mat& b, float v0, int reduction_type, float coeff, int keepdims, const Option& opt)
{
    Op op;
    Op2 op2;

    size_t elemsize = a.elemsize;
    int dims = a.dims;

    if (!keepdims)
    {
        if (dims == 1)  // reduction_type == 0
        {
            int w = a.w;
            b.create(1, elemsize, opt.blob_allocator);
            const float* ptr = a;

            float sum = v0;
            for (int i=0; i<w; i++)
            {
                sum = op(sum, ptr[i]);
            }
            b[0] = sum * coeff;

        }
        if (dims == 2)
        {
            int w = a.w;
            int h = a.h;

            if (reduction_type == 0)
            {
                // w h -> X X
                b.create(1, elemsize, opt.blob_allocator);

                Mat sums(h, elemsize, opt.workspace_allocator);
                if (sums.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i=0; i<h; i++)
                {
                    const float* ptr = a.row(i);

                    float sum = v0;
                    for (int j=0; j<w; j++)
                    {
                        sum = op(sum, ptr[j]);
                    }
                    sums[i] = sum;
                }

                float sum = v0;
                for (int i=0; i<h; i++)
                {
                    sum = op2(sum, sums[i]);
                }

                b[0] = sum * coeff;
            }
            else if (reduction_type == 1)
            {
                // w h -> X h
                b.create(h, elemsize, opt.blob_allocator);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i=0; i<h; i++)
                {
                    const float* ptr = a.row(i);

                    float sum = v0;
                    for (int j=0; j<w; j++)
                    {
                        sum = op(sum, ptr[j]);
                    }
                    b[i] = sum * coeff;
                }

            }
            else if (reduction_type == -1)
            {
                // w h -> w X
                b.create(w, elemsize, opt.blob_allocator);
                b.fill(v0);

                for (int i=0; i<h; i++)
                {
                    const float* ptr = a.row(i);
                    for (int j=0; j<w; j++)
                    {
                        b[j] = op(b[j], ptr[j]);
                    }
                }

                for (int j=0; j<w; j++)
                {
                    b[j] *= coeff;
                }
            }
        }
        if (dims == 3)
        {
            int w = a.w;
            int h = a.h;
            int channels = a.c;
            int size = w * h;

            if (reduction_type == 0)
            {
                // w h c -> X X X
                b.create(1, elemsize, opt.blob_allocator);
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
            else if (reduction_type == 1)
            {
                // w h c -> X X c
                b.create(channels, elemsize, opt.blob_allocator);

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
            else if (reduction_type == 2)
            {
                // w h c -> X h c
                b.create(h, channels, elemsize, opt.blob_allocator);

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
                            sum = op(sum, ptr[j]);
                        }

                        outptr[i] = sum * coeff;

                        ptr += w;
                    }
                }
            }
            else if (reduction_type == 3)
            {
                // w h c -> X h X
                b.create(h, elemsize, opt.blob_allocator);
                Mat mins(1, h, channels, elemsize, opt.workspace_allocator);
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
                        float sum = v0;
                        for (int j=0; j<w; j++)
                        {
                            sum = op(sum, ptr[j]);
                        }
                        mins_ptr[i] = sum;
                        ptr += w;
                    }
                }
                
                b.fill(v0);

                for (int q=0; q<channels; q++)
                {
                    const float* mins_ptr = mins.channel(q);
                    for (int i=0; i<h; i++)
                    {
                        b[i] = op2(b[i], mins_ptr[i]);
                    }
                }
                
                for (int j=0; j<h; j++)
                {
                    b[j] *= coeff;
                }

            }
            else if (reduction_type == -1)
            {
                // w h c -> w X X
                b.create(w, elemsize, opt.blob_allocator);

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
                            mins_ptr[j] = op(mins_ptr[j], ptr[j]);
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
            else if (reduction_type == -2)
            {
                // w h c -> w h X
                b.create(w, h, elemsize, opt.blob_allocator);

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
            else if (reduction_type == -3)
            {
                // w h c -> w X c
                b.create(w, channels, elemsize, opt.blob_allocator);
                
                b.fill(v0);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q=0; q<channels; q++)
                {
                    const float* ptr = a.channel(q);
                    float* outptr = b.row(q);

                    for (int i=0; i<h; i++)
                    {
                        for (int j=0; j<w; j++)
                        {
                            outptr[j] = op(outptr[j], ptr[j]);
                        }
                        ptr += w;
                    }

                    for (int j=0; j<w; j++)
                    {
                        outptr[j] *= coeff;
                    }
                }
            }
        }
    }
    else  // keepdims
    {
        if (dims == 1)  // reduction_type == 0
        {
            int w = a.w;
            b.create(1, elemsize, opt.blob_allocator);
            const float* ptr = a;

            float sum = v0;
            for (int i=0; i<w; i++)
            {
                sum = op(sum, ptr[i]);
            }
            b[0] = sum * coeff;

        }
        if (dims == 2)
        {
            int w = a.w;
            int h = a.h;

            if (reduction_type == 0)
            {
                // w h -> 1 1
                b.create(1, 1, elemsize, opt.blob_allocator);

                Mat sums(h, elemsize, opt.workspace_allocator);
                if (sums.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i=0; i<h; i++)
                {
                    const float* ptr = a.row(i);

                    float sum = v0;
                    for (int j=0; j<w; j++)
                    {
                        sum = op(sum, ptr[j]);
                    }
                    sums[i] = sum;
                }

                float sum = v0;
                for (int i=0; i<h; i++)
                {
                    sum = op2(sum, sums[i]);
                }

                b[0] = sum * coeff;
            }
            else if (reduction_type == 1)
            {
                // w h -> 1 h
                b.create(1, h, elemsize, opt.blob_allocator);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i=0; i<h; i++)
                {
                    const float* ptr = a.row(i);

                    float sum = v0;
                    for (int j=0; j<w; j++)
                    {
                        sum = op(sum, ptr[j]);
                    }
                    b[i] = sum * coeff;
                }

            }
            else if (reduction_type == -1)
            {
                // w h -> w 1
                b.create(w, 1, elemsize, opt.blob_allocator);
                b.fill(v0);

                for (int i=0; i<h; i++)
                {
                    const float* ptr = a.row(i);
                    for (int j=0; j<w; j++)
                    {
                        b[j] = op(b[j], ptr[j]);
                    }
                }

                for (int j=0; j<w; j++)
                {
                    b[j] *= coeff;
                }
            }
        }
        if (dims == 3)
        {
            int w = a.w;
            int h = a.h;
            int channels = a.c;
            int size = w * h;

            if (reduction_type == 0)
            {
                // w h c -> 1 1 1
                b.create(1, 1, 1, elemsize, opt.blob_allocator);
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
            else if (reduction_type == 1)
            {
                // w h c -> 1 1 c
                b.create(1, 1, channels, elemsize, opt.blob_allocator);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q=0; q<channels; q++)
                {
                    const float* ptr = a.channel(q);
                    float* outptr = b.channel(q);

                    float sum = v0;
                    for (int i=0; i<size; i++)
                    {
                        sum = op(sum, ptr[i]);
                    }

                    outptr[0] = sum * coeff;
                }
            }
            else if (reduction_type == 2)
            {
                // w h c -> 1 h c
                b.create(1, h, channels, elemsize, opt.blob_allocator);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q=0; q<channels; q++)
                {
                    const float* ptr = a.channel(q);
                    float* outptr = b.channel(q);

                    for (int i=0; i<h; i++)
                    {
                        float sum = v0;
                        for (int j=0; j<w; j++)
                        {
                            sum = op(sum, ptr[j]);
                        }
                        outptr[i] = sum * coeff;

                        ptr += w;
                    }
                }
            }
            else if (reduction_type == 3)
            {
                // w h c -> 1 h 1
                b.create(1, h, 1, elemsize, opt.blob_allocator);

                Mat mins(1, h, channels, elemsize, opt.workspace_allocator);
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
                        float sum = v0;
                        for (int j=0; j<w; j++)
                        {
                            sum = op(sum, ptr[j]);
                        }
                        mins_ptr[i] = sum;
                        ptr += w;
                    }
                }
                
                b.fill(v0);

                for (int q=0; q<channels; q++)
                {
                    const float* mins_ptr = mins.channel(q);
                    for (int i=0; i<h; i++)
                    {
                        b[i] = op2(b[i], mins_ptr[i]);
                    }
                }
                
                for (int j=0; j<h; j++)
                {
                    b[j] *= coeff;
                }

            }
            else if (reduction_type == -1)
            {
                // w h c -> w 1 1
                b.create(w, 1, 1, elemsize, opt.blob_allocator);

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
                            mins_ptr[j] = op(mins_ptr[j], ptr[j]);
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
            else if (reduction_type == -2)
            {
                // w h c -> w h 1
                b.create(w, h, 1, elemsize, opt.blob_allocator);

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
            else if (reduction_type == -3)
            {
                // w h c -> w 1 c
                b.create(w, 1, channels, elemsize, opt.blob_allocator);
                b.fill(v0);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q=0; q<channels; q++)
                {
                    const float* ptr = a.channel(q);
                    float* outptr = b.channel(q);

                    for (int i=0; i<h; i++)
                    {
                        for (int j=0; j<w; j++)
                        {
                            outptr[j] = op(outptr[j], ptr[j]);
                        }
                        ptr += w;
                    }

                    for (int j=0; j<w; j++)
                    {
                        outptr[j] *= coeff;
                    }
                }
            }
        }

    }

    return 0;
}

template<typename T>
struct post_process_sqrt : std::unary_function<T,T> {
    T operator() (const T& x) const { return sqrt(x); }
};

template<typename T>
struct post_process_log : std::unary_function<T,T> {
    T operator() (const T& x) const { return log(x); }
};

template<typename T>
struct reduction_op_asum : std::binary_function<T,T,T> {
    T operator() (const T& x, const T& y) const { return x + fabs(y); }
};

template<typename T>
struct reduction_op_sumsq : std::binary_function<T,T,T> {
    T operator() (const T& x, const T& y) const { return x + y * y; }
};

template<typename T>
struct reduction_op_sumsexp : std::binary_function<T,T,T> {
    T operator() (const T& x, const T& y) const { return x + exp(y); }
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
    /*
     * type = 0 reduce all axes by default
     * type = 1 reduce according to the axes
     *
     * reduction_type = 0 reduce all axes (supported dims rank: 1, 2, 3)
     * reduction_type = 1 reserve the first axis (supported dims rank: 2, 3)
     * reduction_type = -1 reserve the last axis (supported dims rank: 2, 3)
     * reduction_type = 2 reserve the first two axes (supported dims rank: 3)
     * reduction_type = -2 reserve the last two axes (supported dims rank: 3)
     * reduction_type = 3 reserve the median axis (supported dims rank: 3)
     * reduction_type = -3 reserve the first and last axes (supported dims rank: 3)
    */
    
    int dims = bottom_blob.dims;
    int axes_flag[3] = {0};
    int reduction_type = -233;

    if (type)
    {
        const int* axes_ptr = axes;
        int reduced_axes_num = axes.w;

        for (int i=0; i<reduced_axes_num; i++)
        {
            int axis = axes_ptr[i];
            // handle negative axis
            if (axis < 0)
                axis += dims + 1;
            axes_flag[axis - 1] = 1;
        }

        if (dims == 1)
        {
            reduction_type = 0;
        }
        else if (dims == 2)
        {
            if (axes_flag[0] == 1 && axes_flag[1] == 1)
                reduction_type = 0;
            else if (axes_flag[0] == 0 && axes_flag[1] == 1)
                reduction_type = 1;
            else if (axes_flag[0] == 1 && axes_flag[1] == 0)
                reduction_type = -1;
        }
        else if (dims == 3)
        {
            if (axes_flag[0] == 1 && axes_flag[1] == 1 && axes_flag[2] == 1)
                reduction_type = 0;
            else if (axes_flag[0] == 0 && axes_flag[1] == 1 && axes_flag[2] == 1)
                reduction_type = 1;
            else if (axes_flag[0] == 0 && axes_flag[1] == 0 && axes_flag[2] == 1)
                reduction_type = 2;
            else if (axes_flag[0] == 1 && axes_flag[1] == 0 && axes_flag[2] == 1)
                reduction_type = 3;
            else if (axes_flag[0] == 1 && axes_flag[1] == 1 && axes_flag[2] == 0)
                reduction_type = -1;
            else if (axes_flag[0] == 1 && axes_flag[1] == 0 && axes_flag[2] == 0)
                reduction_type = -2;
            else if (axes_flag[0] == 0 && axes_flag[1] == 1 && axes_flag[2] == 0)
                reduction_type = -3;
        }
    }
    else
    {
        reduction_type = 0;
    }

    if (operation == ReductionOp_SUM)
        return reduction_op< std::plus<float>, std::plus<float> >(bottom_blob, top_blob, 0.f, reduction_type, coeff, keepdims, opt);

    if (operation == ReductionOp_ASUM)
        return reduction_op< reduction_op_asum<float>, std::plus<float> >(bottom_blob, top_blob, 0.f, reduction_type, coeff, keepdims, opt);

    if (operation == ReductionOp_SUMSQ)
        return reduction_op< reduction_op_sumsq<float>, std::plus<float> >(bottom_blob, top_blob, 0.f, reduction_type, coeff, keepdims, opt);

    if (operation == ReductionOp_MEAN)
    {
        int scale = 1;
        int dims = bottom_blob.dims;
        if (dims == 1)
        {
            scale = bottom_blob.w;
        }
        else if (dims == 2)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;
            if(reduction_type == 0)
                scale = w * h;
            else if(reduction_type == 1)
                scale = w;
            else if(reduction_type == -1)
                scale = h;
        }
        else if (dims == 3)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int c = bottom_blob.c;
            if (reduction_type == 0)
                scale = w * h * c;
            else if(reduction_type == 1)
                scale = w * h;
            else if(reduction_type == 2)
                scale = w;
            else if(reduction_type == 3)
                scale = w * c;
            else if(reduction_type == -1)
                scale = h * c;
            else if(reduction_type == -2)
                scale = c;
            else if(reduction_type == -3)
                scale = h;
        }

        // Maybe lose some precision?
        float coeff_mean = coeff / scale;
        return reduction_op< std::plus<float>, std::plus<float> >(bottom_blob, top_blob, 0.f, reduction_type, coeff_mean, keepdims, opt);
    }

    if (operation == ReductionOp_MAX)
        return reduction_op< reduction_op_max<float>, reduction_op_max<float> >(bottom_blob, top_blob, -FLT_MAX, reduction_type, coeff, keepdims, opt);

    if (operation == ReductionOp_MIN)
        return reduction_op< reduction_op_min<float>, reduction_op_min<float> >(bottom_blob, top_blob, FLT_MAX, reduction_type, coeff, keepdims, opt);

    if (operation == ReductionOp_PROD)
        return reduction_op< std::multiplies<float>, std::multiplies<float> >(bottom_blob, top_blob, 1.f, reduction_type, coeff, keepdims, opt);
    
    if (operation == ReductionOp_L1)
        return reduction_op< reduction_op_asum<float>, std::plus<float> >(bottom_blob, top_blob, 0.f, reduction_type, 1.f, keepdims, opt);

    if (operation == ReductionOp_L2)
    {
        int ret = reduction_op< reduction_op_sumsq<float>, std::plus<float> >(bottom_blob, top_blob, 0.f, reduction_type, 1.f, keepdims, opt);
        if (ret != 0)
            return -100;

        ret = post_process< post_process_sqrt<float> >(top_blob, opt);
        if (ret != 0)
            return -100;
    }
    
    if (operation == ReductionOp_LogSum)
    {
        int ret = reduction_op< std::plus<float>, std::plus<float> >(bottom_blob, top_blob, 0.f, reduction_type, 1.f, keepdims, opt);
        if (ret != 0)
            return -100;

        ret = post_process< post_process_log<float> >(top_blob, opt);
        if (ret != 0)
            return -100;
    }
    
    if (operation == ReductionOp_LogSumExp)
    {
        int ret = reduction_op< reduction_op_sumsexp<float>, std::plus<float> >(bottom_blob, top_blob, 0.f, reduction_type, 1.f, keepdims, opt);
        if (ret != 0)
            return -100;

        ret = post_process< post_process_log<float> >(top_blob, opt);
        if (ret != 0)
            return -100;
    }
    return 0;
}

} // namespace ncnn
