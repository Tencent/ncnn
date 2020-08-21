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

namespace ncnn {

Reduction::Reduction()
{
    one_blob_only = true;
    support_inplace = false;
}

int Reduction::load_param(const ParamDict& pd)
{
    operation = pd.get(0, 0);
    reduce_all = pd.get(1, 1);
    coeff = pd.get(2, 1.f);
    axes = pd.get(3, Mat());
    keepdims = pd.get(4, 0);

    return 0;
}

template<typename Op, typename Op2>
static int reduction_op(const Mat& a, Mat& b, float v0, bool reduce_w, bool reduce_h, bool reduce_c, const Option& opt)
{
    Op op;
    Op2 op2;

    size_t elemsize = a.elemsize;
    int dims = a.dims;

    if (dims == 1)
    {
        int w = a.w;
        b.create(1, elemsize, opt.blob_allocator);
        const float* ptr = a;

        float sum = v0;
        for (int i = 0; i < w; i++)
        {
            sum = op(sum, ptr[i]);
        }
        b[0] = sum;

        return 0;
    }

    if (dims == 2)
    {
        int w = a.w;
        int h = a.h;

        if (reduce_w && reduce_h)
        {
            // w h -> X X
            b.create(1, elemsize, opt.blob_allocator);

            Mat sums(h, elemsize, opt.workspace_allocator);
            if (sums.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const float* ptr = a.row(i);

                float sum = v0;
                for (int j = 0; j < w; j++)
                {
                    sum = op(sum, ptr[j]);
                }
                sums[i] = sum;
            }

            float sum = v0;
            for (int i = 0; i < h; i++)
            {
                sum = op2(sum, sums[i]);
            }
            b[0] = sum;

            return 0;
        }

        if (reduce_w && !reduce_h)
        {
            // w h -> X h
            b.create(h, elemsize, opt.blob_allocator);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const float* ptr = a.row(i);

                float sum = v0;
                for (int j = 0; j < w; j++)
                {
                    sum = op(sum, ptr[j]);
                }
                b[i] = sum;
            }
            return 0;
        }

        if (!reduce_w && reduce_h)
        {
            // w h -> w X
            b.create(w, elemsize, opt.blob_allocator);
            b.fill(v0);

            for (int i = 0; i < h; i++)
            {
                const float* ptr = a.row(i);
                for (int j = 0; j < w; j++)
                {
                    b[j] = op(b[j], ptr[j]);
                }
            }
            return 0;
        }
    }

    if (dims == 3)
    {
        int w = a.w;
        int h = a.h;
        int channels = a.c;
        int size = w * h;

        if (reduce_w && reduce_h && reduce_c)
        {
            // w h c -> X X X
            b.create(1, elemsize, opt.blob_allocator);
            Mat sums(channels, elemsize, opt.workspace_allocator);
            if (sums.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);

                float sum = v0;
                for (int i = 0; i < size; i++)
                {
                    sum = op(sum, ptr[i]);
                }
                sums[q] = sum;
            }

            float sum = v0;
            for (int i = 0; i < channels; i++)
            {
                sum = op2(sum, sums[i]);
            }
            b[0] = sum;

            return 0;
        }

        if (reduce_w && reduce_h && !reduce_c)
        {
            // w h c -> X X c
            b.create(channels, elemsize, opt.blob_allocator);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);

                float sum = v0;
                for (int i = 0; i < size; i++)
                {
                    sum = op(sum, ptr[i]);
                }
                b[q] = sum;
            }

            return 0;
        }

        if (reduce_w && !reduce_h && !reduce_c)
        {
            // w h c -> X h c
            b.create(h, channels, elemsize, opt.blob_allocator);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                float* outptr = b.row(q);

                for (int i = 0; i < h; i++)
                {
                    float sum = v0;
                    for (int j = 0; j < w; j++)
                    {
                        sum = op(sum, ptr[j]);
                    }
                    outptr[i] = sum;
                    ptr += w;
                }
            }

            return 0;
        }

        if (reduce_w && !reduce_h && reduce_c)
        {
            // w h c -> X h X
            b.create(h, elemsize, opt.blob_allocator);
            Mat mins(1, h, channels, elemsize, opt.workspace_allocator);
            if (mins.empty())
                return -100;

            mins.fill(v0);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                float* mins_ptr = mins.channel(q);

                for (int i = 0; i < h; i++)
                {
                    float sum = v0;
                    for (int j = 0; j < w; j++)
                    {
                        sum = op(sum, ptr[j]);
                    }
                    mins_ptr[i] = sum;
                    ptr += w;
                }
            }

            b.fill(v0);

            for (int q = 0; q < channels; q++)
            {
                const float* mins_ptr = mins.channel(q);
                for (int i = 0; i < h; i++)
                {
                    b[i] = op2(b[i], mins_ptr[i]);
                }
            }

            return 0;
        }

        if (!reduce_w && reduce_h && reduce_c)
        {
            // w h c -> w X X
            b.create(w, elemsize, opt.blob_allocator);

            Mat mins(w, 1, channels, elemsize, opt.workspace_allocator);
            if (mins.empty())
                return -100;

            mins.fill(v0);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                float* mins_ptr = mins.channel(q);

                for (int i = 0; i < h; i++)
                {
                    for (int j = 0; j < w; j++)
                    {
                        mins_ptr[j] = op(mins_ptr[j], ptr[j]);
                    }
                    ptr += w;
                }
            }

            b.fill(v0);

            for (int q = 0; q < channels; q++)
            {
                const float* mins_ptr = mins.channel(q);
                for (int j = 0; j < w; j++)
                {
                    b[j] = op2(b[j], mins_ptr[j]);
                }
            }

            return 0;
        }

        if (!reduce_w && !reduce_h && reduce_c)
        {
            // w h c -> w h X
            b.create(w, h, elemsize, opt.blob_allocator);

            b.fill(v0);

            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);

                for (int i = 0; i < size; i++)
                {
                    b[i] = op(b[i], ptr[i]);
                }
            }

            return 0;
        }

        if (!reduce_w && reduce_h && !reduce_c)
        {
            // w h c -> w X c
            b.create(w, channels, elemsize, opt.blob_allocator);

            b.fill(v0);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                float* outptr = b.row(q);

                for (int i = 0; i < h; i++)
                {
                    for (int j = 0; j < w; j++)
                    {
                        outptr[j] = op(outptr[j], ptr[j]);
                    }
                    ptr += w;
                }
            }
            return 0;
        }
    }

    return 0;
}

template<typename Op, typename Op2>
static int reduction_op_keepdims(const Mat& a, Mat& b, float v0, bool reduce_w, bool reduce_h, bool reduce_c, const Option& opt)
{
    Op op;
    Op2 op2;

    size_t elemsize = a.elemsize;
    int dims = a.dims;

    if (dims == 1)
    {
        int w = a.w;
        b.create(1, elemsize, opt.blob_allocator);
        const float* ptr = a;

        float sum = v0;
        for (int i = 0; i < w; i++)
        {
            sum = op(sum, ptr[i]);
        }
        b[0] = sum;

        return 0;
    }

    if (dims == 2)
    {
        int w = a.w;
        int h = a.h;

        if (reduce_w && reduce_h)
        {
            // w h -> 1 1
            b.create(1, 1, elemsize, opt.blob_allocator);

            Mat sums(h, elemsize, opt.workspace_allocator);
            if (sums.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const float* ptr = a.row(i);

                float sum = v0;
                for (int j = 0; j < w; j++)
                {
                    sum = op(sum, ptr[j]);
                }
                sums[i] = sum;
            }

            float sum = v0;
            for (int i = 0; i < h; i++)
            {
                sum = op2(sum, sums[i]);
            }
            b[0] = sum;

            return 0;
        }

        if (reduce_w && !reduce_h)
        {
            // w h -> 1 h
            b.create(1, h, elemsize, opt.blob_allocator);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const float* ptr = a.row(i);

                float sum = v0;
                for (int j = 0; j < w; j++)
                {
                    sum = op(sum, ptr[j]);
                }
                b[i] = sum;
            }

            return 0;
        }

        if (!reduce_w && reduce_h)
        {
            // w h -> w 1
            b.create(w, 1, elemsize, opt.blob_allocator);
            b.fill(v0);

            for (int i = 0; i < h; i++)
            {
                const float* ptr = a.row(i);
                for (int j = 0; j < w; j++)
                {
                    b[j] = op(b[j], ptr[j]);
                }
            }

            return 0;
        }
    }

    if (dims == 3)
    {
        int w = a.w;
        int h = a.h;
        int channels = a.c;
        int size = w * h;

        if (reduce_w && reduce_h && reduce_c)
        {
            // w h c -> 1 1 1
            b.create(1, 1, 1, elemsize, opt.blob_allocator);
            Mat sums(channels, elemsize, opt.workspace_allocator);
            if (sums.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);

                float sum = v0;
                for (int i = 0; i < size; i++)
                {
                    sum = op(sum, ptr[i]);
                }
                sums[q] = sum;
            }

            float sum = v0;
            for (int i = 0; i < channels; i++)
            {
                sum = op2(sum, sums[i]);
            }
            b[0] = sum;

            return 0;
        }

        if (reduce_w && reduce_h && !reduce_c)
        {
            // w h c -> 1 1 c
            b.create(1, 1, channels, elemsize, opt.blob_allocator);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                float* outptr = b.channel(q);

                float sum = v0;
                for (int i = 0; i < size; i++)
                {
                    sum = op(sum, ptr[i]);
                }

                outptr[0] = sum;
            }

            return 0;
        }

        if (reduce_w && !reduce_h && !reduce_c)
        {
            // w h c -> 1 h c
            b.create(1, h, channels, elemsize, opt.blob_allocator);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                float* outptr = b.channel(q);

                for (int i = 0; i < h; i++)
                {
                    float sum = v0;
                    for (int j = 0; j < w; j++)
                    {
                        sum = op(sum, ptr[j]);
                    }
                    outptr[i] = sum;
                    ptr += w;
                }
            }

            return 0;
        }

        if (reduce_w && !reduce_h && reduce_c)
        {
            // w h c -> 1 h 1
            b.create(1, h, 1, elemsize, opt.blob_allocator);

            Mat mins(1, h, channels, elemsize, opt.workspace_allocator);
            if (mins.empty())
                return -100;

            mins.fill(v0);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                float* mins_ptr = mins.channel(q);

                for (int i = 0; i < h; i++)
                {
                    float sum = v0;
                    for (int j = 0; j < w; j++)
                    {
                        sum = op(sum, ptr[j]);
                    }
                    mins_ptr[i] = sum;
                    ptr += w;
                }
            }

            b.fill(v0);

            for (int q = 0; q < channels; q++)
            {
                const float* mins_ptr = mins.channel(q);
                for (int i = 0; i < h; i++)
                {
                    b[i] = op2(b[i], mins_ptr[i]);
                }
            }

            return 0;
        }

        if (!reduce_w && reduce_h && reduce_c)
        {
            // w h c -> w 1 1
            b.create(w, 1, 1, elemsize, opt.blob_allocator);

            Mat mins(w, 1, channels, elemsize, opt.workspace_allocator);
            if (mins.empty())
                return -100;

            mins.fill(v0);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                float* mins_ptr = mins.channel(q);

                for (int i = 0; i < h; i++)
                {
                    for (int j = 0; j < w; j++)
                    {
                        mins_ptr[j] = op(mins_ptr[j], ptr[j]);
                    }
                    ptr += w;
                }
            }

            b.fill(v0);

            for (int q = 0; q < channels; q++)
            {
                const float* mins_ptr = mins.channel(q);
                for (int j = 0; j < w; j++)
                {
                    b[j] = op2(b[j], mins_ptr[j]);
                }
            }

            return 0;
        }

        if (!reduce_w && !reduce_h && reduce_c)
        {
            // w h c -> w h 1
            b.create(w, h, 1, elemsize, opt.blob_allocator);

            b.fill(v0);

            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);

                for (int i = 0; i < size; i++)
                {
                    b[i] = op(b[i], ptr[i]);
                }
            }

            return 0;
        }

        if (!reduce_w && reduce_h && !reduce_c)
        {
            // w h c -> w 1 c
            b.create(w, 1, channels, elemsize, opt.blob_allocator);
            b.fill(v0);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                float* outptr = b.channel(q);

                for (int i = 0; i < h; i++)
                {
                    for (int j = 0; j < w; j++)
                    {
                        outptr[j] = op(outptr[j], ptr[j]);
                    }
                    ptr += w;
                }
            }

            return 0;
        }
    }

    return 0;
}

template<typename MathOp>
static int reduction_post_process(Mat& a, float coeff, const Option& opt)
{
    MathOp mathop;

    int dims = a.dims;
    if (dims == 1)
    {
        int w = a.w;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < w; i++)
            a[i] = mathop(a[i]) * coeff;
    }
    else if (dims == 2)
    {
        int size = a.w * a.h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < size; i++)
            a[i] = mathop(a[i]) * coeff;
    }
    else if (dims == 3)
    {
        int c = a.c;
        int size = a.w * a.h;
        if (c == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < size; i++)
                a[i] = mathop(a[i]) * coeff;
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < c; q++)
            {
                float* outptr = a.channel(q);
                for (int i = 0; i < size; i++)
                    outptr[i] = mathop(outptr[i]) * coeff;
            }
        }
    }

    return 0;
}

template<typename Op, typename Op2, typename Op3>
static int reduction(const Mat& a, Mat& b, float v0, bool reduce_w, bool reduce_h, bool reduce_c, bool post_process, float coeff, int keepdims, const Option& opt)
{
    int ret;
    if (keepdims)
        ret = reduction_op_keepdims<Op, Op2>(a, b, v0, reduce_w, reduce_h, reduce_c, opt);
    else
        ret = reduction_op<Op, Op2>(a, b, v0, reduce_w, reduce_h, reduce_c, opt);
    if (ret != 0)
        return -100;

    if (post_process || fabs(coeff - 1.f) > FLT_EPSILON)
    {
        ret = reduction_post_process<Op3>(b, coeff, opt);
        if (ret != 0)
            return -100;
    }
    return ret;
}

template<typename T>
struct post_process_identity
{
    T operator()(const T& x) const
    {
        return x;
    }
};

template<typename T>
struct post_process_sqrt
{
    T operator()(const T& x) const
    {
        return static_cast<T>(sqrt(x));
    }
};

template<typename T>
struct post_process_log
{
    T operator()(const T& x) const
    {
        return static_cast<T>(log(x));
    }
};

template<typename T>
struct reduction_op_add
{
    T operator()(const T& x, const T& y) const
    {
        return x + y;
    }
};

template<typename T>
struct reduction_op_mul
{
    T operator()(const T& x, const T& y) const
    {
        return x * y;
    }
};

template<typename T>
struct reduction_op_asum
{
    T operator()(const T& x, const T& y) const
    {
        return static_cast<T>(x + fabs(y));
    }
};

template<typename T>
struct reduction_op_sumsq
{
    T operator()(const T& x, const T& y) const
    {
        return x + y * y;
    }
};

template<typename T>
struct reduction_op_sumsexp
{
    T operator()(const T& x, const T& y) const
    {
        return static_cast<T>(x + exp(y));
    }
};

template<typename T>
struct reduction_op_max
{
    T operator()(const T& x, const T& y) const
    {
        return std::max(x, y);
    }
};

template<typename T>
struct reduction_op_min
{
    T operator()(const T& x, const T& y) const
    {
        return std::min(x, y);
    }
};

int Reduction::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int dims = bottom_blob.dims;
    int axes_flag[3] = {0};
    bool reduce_w = false;
    bool reduce_h = false;
    bool reduce_c = false;

    if (reduce_all)
    {
        reduce_w = true;
        reduce_h = true;
        reduce_c = true;
    }
    else
    {
        const int* axes_ptr = axes;
        int reduced_axes_num = axes.w;

        for (int i = 0; i < reduced_axes_num; i++)
        {
            int axis = axes_ptr[i];
            // handle negative axis
            if (axis < 0)
                axis += dims + 1;
            axes_flag[axis - 1] = 1;
        }

        if (dims == 1)
        {
            reduce_w = true;
        }
        else if (dims == 2)
        {
            if (axes_flag[0] == 1) reduce_h = true;
            if (axes_flag[1] == 1) reduce_w = true;
        }
        else if (dims == 3)
        {
            if (axes_flag[0] == 1) reduce_c = true;
            if (axes_flag[1] == 1) reduce_h = true;
            if (axes_flag[2] == 1) reduce_w = true;
        }
    }

    if (operation == ReductionOp_SUM)
        return reduction<reduction_op_add<float>, reduction_op_add<float>, post_process_identity<float> >(bottom_blob, top_blob, 0.f, reduce_w, reduce_h, reduce_c, false, coeff, keepdims, opt);

    if (operation == ReductionOp_ASUM)
        return reduction<reduction_op_asum<float>, reduction_op_add<float>, post_process_identity<float> >(bottom_blob, top_blob, 0.f, reduce_w, reduce_h, reduce_c, false, coeff, keepdims, opt);

    if (operation == ReductionOp_SUMSQ)
        return reduction<reduction_op_sumsq<float>, reduction_op_add<float>, post_process_identity<float> >(bottom_blob, top_blob, 0.f, reduce_w, reduce_h, reduce_c, false, coeff, keepdims, opt);

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
            if (reduce_w) scale *= bottom_blob.w;
            if (reduce_h) scale *= bottom_blob.h;
        }
        else if (dims == 3)
        {
            if (reduce_w) scale *= bottom_blob.w;
            if (reduce_h) scale *= bottom_blob.h;
            if (reduce_c) scale *= bottom_blob.c;
        }

        float coeff_mean = coeff / scale;
        return reduction<reduction_op_add<float>, reduction_op_add<float>, post_process_identity<float> >(bottom_blob, top_blob, 0.f, reduce_w, reduce_h, reduce_c, true, coeff_mean, keepdims, opt);
    }

    if (operation == ReductionOp_MAX)
        return reduction<reduction_op_max<float>, reduction_op_max<float>, post_process_identity<float> >(bottom_blob, top_blob, -FLT_MAX, reduce_w, reduce_h, reduce_c, false, coeff, keepdims, opt);

    if (operation == ReductionOp_MIN)
        return reduction<reduction_op_min<float>, reduction_op_min<float>, post_process_identity<float> >(bottom_blob, top_blob, FLT_MAX, reduce_w, reduce_h, reduce_c, false, coeff, keepdims, opt);

    if (operation == ReductionOp_PROD)
        return reduction<reduction_op_mul<float>, reduction_op_mul<float>, post_process_identity<float> >(bottom_blob, top_blob, 1.f, reduce_w, reduce_h, reduce_c, false, coeff, keepdims, opt);

    if (operation == ReductionOp_L1)
        return reduction<reduction_op_asum<float>, reduction_op_add<float>, post_process_identity<float> >(bottom_blob, top_blob, 0.f, reduce_w, reduce_h, reduce_c, false, 1.f, keepdims, opt);

    if (operation == ReductionOp_L2)
        return reduction<reduction_op_sumsq<float>, reduction_op_add<float>, post_process_sqrt<float> >(bottom_blob, top_blob, 0.f, reduce_w, reduce_h, reduce_c, true, 1.f, keepdims, opt);

    if (operation == ReductionOp_LogSum)
        return reduction<reduction_op_add<float>, reduction_op_add<float>, post_process_log<float> >(bottom_blob, top_blob, 0.f, reduce_w, reduce_h, reduce_c, true, 1.f, keepdims, opt);

    if (operation == ReductionOp_LogSumExp)
        return reduction<reduction_op_sumsexp<float>, reduction_op_add<float>, post_process_log<float> >(bottom_blob, top_blob, 0.f, reduce_w, reduce_h, reduce_c, true, 1.f, keepdims, opt);

    return 0;
}

} // namespace ncnn
