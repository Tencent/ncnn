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

    // the original reduction handle axes as blob with batch dimension
    // ask user to regenerate param instead of producing wrong result
    int fixbug0 = pd.get(5, 0);
    if (fixbug0 == 0 && !axes.empty())
    {
        NCNN_LOGE("param is too old, please regenerate!");
        return -1;
    }

    return 0;
}

template<typename Op>
static float reduction(float v0, const float* ptr, int size)
{
    Op op;

    float sum = v0;
    for (int i = 0; i < size; i++)
    {
        sum = op(sum, ptr[i]);
    }

    return sum;
}

template<typename Op>
static float reduction(float v0, const float* ptr, int size, int stride)
{
    Op op;

    float sum = v0;
    for (int i = 0; i < size; i++)
    {
        sum = op(sum, *ptr);
        ptr += stride;
    }

    return sum;
}

template<typename Op>
static float reduction(float v0, const float* ptr, int size0, int size1, int stride1)
{
    Op op;

    float sum = v0;
    for (int i = 0; i < size1; i++)
    {
        for (int j = 0; j < size0; j++)
        {
            sum = op(sum, ptr[j]);
        }
        ptr += stride1;
    }

    return sum;
}

template<typename Op>
static float reduction(float v0, const float* ptr, int size0, int stride0, int size1, int stride1)
{
    Op op;

    float sum = v0;
    for (int i = 0; i < size1; i++)
    {
        const float* ptr0 = ptr;
        for (int j = 0; j < size0; j++)
        {
            sum = op(sum, *ptr0);
            ptr0 += stride0;
        }
        ptr += stride1;
    }

    return sum;
}

struct reduction_op_add
{
    float operator()(const float& x, const float& y) const
    {
        return x + y;
    }
};

struct reduction_op_mul
{
    float operator()(const float& x, const float& y) const
    {
        return x * y;
    }
};

struct reduction_op_asum
{
    float operator()(const float& x, const float& y) const
    {
        return x + fabsf(y);
    }
};

struct reduction_op_sumsq
{
    float operator()(const float& x, const float& y) const
    {
        return x + y * y;
    }
};

struct reduction_op_sumexp
{
    float operator()(const float& x, const float& y) const
    {
        return x + expf(y);
    }
};

struct reduction_op_max
{
    float operator()(const float& x, const float& y) const
    {
        return std::max(x, y);
    }
};

struct reduction_op_min
{
    float operator()(const float& x, const float& y) const
    {
        return std::min(x, y);
    }
};

static float reduction(float v0, const float* ptr, int size, int op_type)
{
    if (op_type == Reduction::ReductionOp_SUM) return reduction<reduction_op_add>(v0, ptr, size);
    if (op_type == Reduction::ReductionOp_ASUM) return reduction<reduction_op_asum>(v0, ptr, size);
    if (op_type == Reduction::ReductionOp_SUMSQ) return reduction<reduction_op_sumsq>(v0, ptr, size);
    if (op_type == Reduction::ReductionOp_PROD) return reduction<reduction_op_mul>(v0, ptr, size);
    if (op_type == Reduction::ReductionOp_MAX) return reduction<reduction_op_max>(v0, ptr, size);
    if (op_type == Reduction::ReductionOp_MIN) return reduction<reduction_op_min>(v0, ptr, size);
    if (op_type == Reduction::ReductionOp_LogSumExp) return reduction<reduction_op_sumexp>(v0, ptr, size);

    // should never reach here
    return v0;
}

static float reduction(float v0, const float* ptr, int size, int stride, int op_type)
{
    if (op_type == Reduction::ReductionOp_SUM) return reduction<reduction_op_add>(v0, ptr, size, stride);
    if (op_type == Reduction::ReductionOp_ASUM) return reduction<reduction_op_asum>(v0, ptr, size, stride);
    if (op_type == Reduction::ReductionOp_SUMSQ) return reduction<reduction_op_sumsq>(v0, ptr, size, stride);
    if (op_type == Reduction::ReductionOp_PROD) return reduction<reduction_op_mul>(v0, ptr, size, stride);
    if (op_type == Reduction::ReductionOp_MAX) return reduction<reduction_op_max>(v0, ptr, size, stride);
    if (op_type == Reduction::ReductionOp_MIN) return reduction<reduction_op_min>(v0, ptr, size, stride);
    if (op_type == Reduction::ReductionOp_LogSumExp) return reduction<reduction_op_sumexp>(v0, ptr, size, stride);

    // should never reach here
    return v0;
}

static float reduction(float v0, const float* ptr, int size0, int size1, int stride1, int op_type)
{
    if (op_type == Reduction::ReductionOp_SUM) return reduction<reduction_op_add>(v0, ptr, size0, size1, stride1);
    if (op_type == Reduction::ReductionOp_ASUM) return reduction<reduction_op_asum>(v0, ptr, size0, size1, stride1);
    if (op_type == Reduction::ReductionOp_SUMSQ) return reduction<reduction_op_sumsq>(v0, ptr, size0, size1, stride1);
    if (op_type == Reduction::ReductionOp_PROD) return reduction<reduction_op_mul>(v0, ptr, size0, size1, stride1);
    if (op_type == Reduction::ReductionOp_MAX) return reduction<reduction_op_max>(v0, ptr, size0, size1, stride1);
    if (op_type == Reduction::ReductionOp_MIN) return reduction<reduction_op_min>(v0, ptr, size0, size1, stride1);
    if (op_type == Reduction::ReductionOp_LogSumExp) return reduction<reduction_op_sumexp>(v0, ptr, size0, size1, stride1);

    // should never reach here
    return v0;
}

static float reduction(float v0, const float* ptr, int size0, int stride0, int size1, int stride1, int op_type)
{
    if (op_type == Reduction::ReductionOp_SUM) return reduction<reduction_op_add>(v0, ptr, size0, stride0, size1, stride1);
    if (op_type == Reduction::ReductionOp_ASUM) return reduction<reduction_op_asum>(v0, ptr, size0, stride0, size1, stride1);
    if (op_type == Reduction::ReductionOp_SUMSQ) return reduction<reduction_op_sumsq>(v0, ptr, size0, stride0, size1, stride1);
    if (op_type == Reduction::ReductionOp_PROD) return reduction<reduction_op_mul>(v0, ptr, size0, stride0, size1, stride1);
    if (op_type == Reduction::ReductionOp_MAX) return reduction<reduction_op_max>(v0, ptr, size0, stride0, size1, stride1);
    if (op_type == Reduction::ReductionOp_MIN) return reduction<reduction_op_min>(v0, ptr, size0, stride0, size1, stride1);
    if (op_type == Reduction::ReductionOp_LogSumExp) return reduction<reduction_op_sumexp>(v0, ptr, size0, stride0, size1, stride1);

    // should never reach here
    return v0;
}

static int reduction_op(const Mat& a, Mat& b, bool reduce_w, bool reduce_h, bool reduce_d, bool reduce_c, int keepdims, int operation, float coeff, const Option& opt)
{
    int op_type = Reduction::ReductionOp_SUM;
    int op2_type = Reduction::ReductionOp_SUM;
    float v0 = 0.f;

    switch (operation)
    {
    case Reduction::ReductionOp_SUM:
    case Reduction::ReductionOp_MEAN:
    case Reduction::ReductionOp_LogSum:
    {
        break;
    }
    case Reduction::ReductionOp_ASUM:
    case Reduction::ReductionOp_L1:
    {
        op_type = Reduction::ReductionOp_ASUM;
        break;
    }
    case Reduction::ReductionOp_SUMSQ:
    case Reduction::ReductionOp_L2:
    {
        op_type = Reduction::ReductionOp_SUMSQ;
        break;
    }
    case Reduction::ReductionOp_MAX:
    {
        op_type = Reduction::ReductionOp_MAX;
        op2_type = Reduction::ReductionOp_MAX;
        v0 = -FLT_MAX;
        break;
    }
    case Reduction::ReductionOp_MIN:
    {
        op_type = Reduction::ReductionOp_MIN;
        op2_type = Reduction::ReductionOp_MIN;
        v0 = FLT_MAX;
        break;
    }
    case Reduction::ReductionOp_PROD:
    {
        op_type = Reduction::ReductionOp_PROD;
        op2_type = Reduction::ReductionOp_PROD;
        v0 = 1.f;
        break;
    }
    case Reduction::ReductionOp_LogSumExp:
    {
        op_type = Reduction::ReductionOp_LogSumExp;
        break;
    }
    default:
    {
        // should never reach here
        break;
    }
    }

    const size_t elemsize = a.elemsize;
    const int dims = a.dims;

    // NCNN_LOGE("%d  (%d %d %d %d)    %d %d %d %d", dims, a.w, a.h, a.d, a.c, reduce_w, reduce_h, reduce_d, reduce_c);

    if (dims == 1)
    {
        const int w = a.w;
        b.create(1, elemsize, opt.blob_allocator);

        b[0] = reduction(v0, a, w, op_type);
    }

    if (dims == 2)
    {
        const int w = a.w;
        const int h = a.h;

        if (reduce_w && reduce_h)
        {
            // w h -> X X
            if (keepdims)
                b.create(1, 1, elemsize, opt.blob_allocator);
            else
                b.create(1, elemsize, opt.blob_allocator);

            Mat sums(h, elemsize, opt.workspace_allocator);
            if (sums.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const float* ptr = a.row(i);

                sums[i] = reduction(v0, ptr, w, op_type);
            }

            b[0] = reduction(v0, sums, h, op2_type);
        }

        if (reduce_w && !reduce_h)
        {
            // w h -> X h
            if (keepdims)
                b.create(1, h, elemsize, opt.blob_allocator);
            else
                b.create(h, elemsize, opt.blob_allocator);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const float* ptr = a.row(i);

                b[i] = reduction(v0, ptr, w, op_type);
            }
        }

        if (!reduce_w && reduce_h)
        {
            // w h -> w X
            if (keepdims)
                b.create(w, 1, elemsize, opt.blob_allocator);
            else
                b.create(w, elemsize, opt.blob_allocator);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                b[i] = reduction(v0, (const float*)a + i, h, a.w, op_type);
            }
        }
    }

    if (dims == 3)
    {
        const int w = a.w;
        const int h = a.h;
        const int channels = a.c;
        const int size = w * h;

        if (reduce_w && reduce_h && reduce_c)
        {
            // w h c -> X X X
            if (keepdims)
                b.create(1, 1, 1, elemsize, opt.blob_allocator);
            else
                b.create(1, elemsize, opt.blob_allocator);
            Mat sums(channels, elemsize, opt.workspace_allocator);
            if (sums.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);

                sums[q] = reduction(v0, ptr, size, op_type);
            }

            b[0] = reduction(v0, sums, channels, op2_type);
        }

        if (reduce_w && reduce_h && !reduce_c)
        {
            // w h c -> X X c
            if (keepdims)
                b.create(1, 1, channels, elemsize, opt.blob_allocator);
            else
                b.create(channels, elemsize, opt.blob_allocator);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                float* outptr = keepdims ? b.channel(q) : (float*)b + q;

                outptr[0] = reduction(v0, ptr, size, op_type);
            }
        }

        if (reduce_w && !reduce_h && reduce_c)
        {
            // w h c -> X h X
            if (keepdims)
                b.create(1, h, 1, elemsize, opt.blob_allocator);
            else
                b.create(h, elemsize, opt.blob_allocator);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                b[i] = reduction(v0, (const float*)a.row(i), w, channels, a.cstep, op_type);
            }
        }

        if (!reduce_w && reduce_h && reduce_c)
        {
            // w h c -> w X X
            if (keepdims)
                b.create(w, 1, 1, elemsize, opt.blob_allocator);
            else
                b.create(w, elemsize, opt.blob_allocator);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int j = 0; j < w; j++)
            {
                b[j] = reduction(v0, (const float*)a + j, h, w, channels, a.cstep, op_type);
            }
        }

        if (reduce_w && !reduce_h && !reduce_c)
        {
            // w h c -> X h c
            if (keepdims)
                b.create(1, h, channels, elemsize, opt.blob_allocator);
            else
                b.create(h, channels, elemsize, opt.blob_allocator);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                float* outptr = keepdims ? b.channel(q) : b.row(q);

                for (int i = 0; i < h; i++)
                {
                    outptr[i] = reduction(v0, ptr, w, op_type);
                    ptr += w;
                }
            }
        }

        if (!reduce_w && !reduce_h && reduce_c)
        {
            // w h c -> w h X
            if (keepdims)
                b.create(w, h, 1, elemsize, opt.blob_allocator);
            else
                b.create(w, h, elemsize, opt.blob_allocator);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < size; i++)
            {
                b[i] = reduction(v0, (const float*)a + i, channels, a.cstep, op_type);
            }
        }

        if (!reduce_w && reduce_h && !reduce_c)
        {
            // w h c -> w X c
            if (keepdims)
                b.create(w, 1, channels, elemsize, opt.blob_allocator);
            else
                b.create(w, channels, elemsize, opt.blob_allocator);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                float* outptr = keepdims ? b.channel(q) : b.row(q);

                for (int j = 0; j < w; j++)
                {
                    outptr[j] = reduction(v0, ptr + j, h, w, op_type);
                }
            }
        }
    }

    if (dims == 4)
    {
        const int w = a.w;
        const int h = a.h;
        const int d = a.d;
        const int channels = a.c;
        const int size = w * h * d;

        if (reduce_w && reduce_h && reduce_d && reduce_c)
        {
            // w h d c -> X X X X
            if (keepdims)
                b.create(1, 1, 1, 1, elemsize, opt.blob_allocator);
            else
                b.create(1, elemsize, opt.blob_allocator);
            Mat sums(channels, elemsize, opt.workspace_allocator);
            if (sums.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);

                sums[q] = reduction(v0, ptr, size, op_type);
            }

            b[0] = reduction(v0, sums, channels, op2_type);
        }

        if (reduce_w && reduce_h && reduce_d && !reduce_c)
        {
            // w h d c -> X X X c
            if (keepdims)
                b.create(1, 1, 1, channels, elemsize, opt.blob_allocator);
            else
                b.create(channels, elemsize, opt.blob_allocator);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                float* outptr = keepdims ? b.channel(q) : (float*)b + q;

                outptr[0] = reduction(v0, ptr, size, op_type);
            }
        }

        if (reduce_w && reduce_h && !reduce_d && reduce_c)
        {
            // w h d c -> X X d X
            if (keepdims)
                b.create(1, 1, d, 1, elemsize, opt.blob_allocator);
            else
                b.create(d, elemsize, opt.blob_allocator);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < d; i++)
            {
                b[i] = reduction(v0, (const float*)a.depth(i), w * h, channels, a.cstep, op_type);
            }
        }

        if (reduce_w && !reduce_h && reduce_d && reduce_c)
        {
            // w h d c -> X h X X
            if (keepdims)
                b.create(1, h, 1, 1, elemsize, opt.blob_allocator);
            else
                b.create(h, elemsize, opt.blob_allocator);
            Mat mins(h, 1, channels, elemsize, opt.workspace_allocator);
            if (mins.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                float* mins_ptr = mins.channel(q);

                for (int j = 0; j < h; j++)
                {
                    mins_ptr[j] = reduction(v0, ptr, w, d, w * h, op_type);
                    ptr += w;
                }
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                b[i] = reduction(v0, (const float*)mins + i, channels, mins.cstep, op2_type);
            }
        }

        if (!reduce_w && reduce_h && reduce_d && reduce_c)
        {
            // w h d c -> w X X X
            if (keepdims)
                b.create(w, 1, 1, 1, elemsize, opt.blob_allocator);
            else
                b.create(w, elemsize, opt.blob_allocator);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                b[i] = reduction(v0, (const float*)a + i, h * d, w, channels, a.cstep, op_type);
            }
        }

        if (reduce_w && reduce_h && !reduce_d && !reduce_c)
        {
            // w h d c -> X X d c
            if (keepdims)
                b.create(1, 1, d, channels, elemsize, opt.blob_allocator);
            else
                b.create(d, channels, elemsize, opt.blob_allocator);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                float* outptr = keepdims ? b.channel(q) : b.row(q);

                for (int i = 0; i < d; i++)
                {
                    outptr[i] = reduction(v0, ptr, w * h, op_type);
                    ptr += w * h;
                }
            }
        }

        if (reduce_w && !reduce_h && !reduce_d && reduce_c)
        {
            // w h d c -> X h d X
            if (keepdims)
                b.create(1, h, d, 1, elemsize, opt.blob_allocator);
            else
                b.create(h, d, elemsize, opt.blob_allocator);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < d; i++)
            {
                float* bptr = keepdims ? b.depth(i) : b.row(i);

                for (int j = 0; j < h; j++)
                {
                    bptr[j] = reduction(v0, a.depth(i).row(j), w, channels, a.cstep, op_type);
                }
            }
        }

        if (!reduce_w && !reduce_h && reduce_d && reduce_c)
        {
            // w h d c -> w h X X
            if (keepdims)
                b.create(w, h, 1, 1, elemsize, opt.blob_allocator);
            else
                b.create(w, h, elemsize, opt.blob_allocator);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                float* bptr = b.row(i);

                for (int j = 0; j < w; j++)
                {
                    bptr[j] = reduction(v0, a.row(i) + j, d, w * h, channels, a.cstep, op_type);
                }
            }
        }

        if (reduce_w && !reduce_h && reduce_d && !reduce_c)
        {
            // w h d c -> X h X c
            if (keepdims)
                b.create(1, h, 1, channels, elemsize, opt.blob_allocator);
            else
                b.create(h, channels, elemsize, opt.blob_allocator);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                float* outptr = keepdims ? b.channel(q) : b.row(q);

                for (int i = 0; i < h; i++)
                {
                    outptr[i] = reduction(v0, ptr, w, d, w * h, op_type);
                    ptr += w;
                }
            }
        }

        if (!reduce_w && reduce_h && !reduce_d && reduce_c)
        {
            // w h d c -> w X d X
            if (keepdims)
                b.create(w, 1, d, 1, elemsize, opt.blob_allocator);
            else
                b.create(w, d, elemsize, opt.blob_allocator);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < d; i++)
            {
                float* bptr = b.row(i);

                for (int j = 0; j < w; j++)
                {
                    bptr[j] = reduction(v0, (const float*)a.depth(i) + j, h, w, channels, a.cstep, op_type);
                }
            }
        }

        if (!reduce_w && reduce_h && reduce_d && !reduce_c)
        {
            // w h d c -> w X X c
            if (keepdims)
                b.create(w, 1, 1, channels, elemsize, opt.blob_allocator);
            else
                b.create(w, channels, elemsize, opt.blob_allocator);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                float* outptr = keepdims ? b.channel(q) : b.row(q);

                for (int i = 0; i < w; i++)
                {
                    outptr[i] = reduction(v0, ptr + i, h * d, w, op_type);
                }
            }
        }

        if (reduce_w && !reduce_h && !reduce_d && !reduce_c)
        {
            // w h d c -> X h d c
            if (keepdims)
                b.create(1, h, d, channels, elemsize, opt.blob_allocator);
            else
                b.create(h, d, channels, elemsize, opt.blob_allocator);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                float* outptr = b.channel(q);

                for (int i = 0; i < d * h; i++)
                {
                    outptr[i] = reduction(v0, ptr, w, op_type);
                    ptr += w;
                }
            }
        }

        if (!reduce_w && !reduce_h && !reduce_d && reduce_c)
        {
            // w h d c -> w h d X
            if (keepdims)
                b.create(w, h, d, 1, elemsize, opt.blob_allocator);
            else
                b.create(w, h, d, elemsize, opt.blob_allocator);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < d; i++)
            {
                float* outptr = keepdims ? b.depth(i) : b.channel(i);

                for (int j = 0; j < w * h; j++)
                {
                    outptr[j] = reduction(v0, (const float*)a.depth(i) + j, channels, a.cstep, op_type);
                }
            }
        }

        if (!reduce_w && reduce_h && !reduce_d && !reduce_c)
        {
            // w h d c -> w X d c
            if (keepdims)
                b.create(w, 1, d, channels, elemsize, opt.blob_allocator);
            else
                b.create(w, d, channels, elemsize, opt.blob_allocator);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                Mat outm = b.channel(q);

                for (int i = 0; i < d; i++)
                {
                    const float* ptr = a.channel(q).depth(i);
                    float* outptr = outm.row(i);

                    for (int k = 0; k < w; k++)
                    {
                        outptr[k] = reduction(v0, ptr + k, h, w, op_type);
                    }
                }
            }
        }

        if (!reduce_w && !reduce_h && reduce_d && !reduce_c)
        {
            // w h d c -> w h X c
            if (keepdims)
                b.create(w, h, 1, channels, elemsize, opt.blob_allocator);
            else
                b.create(w, h, channels, elemsize, opt.blob_allocator);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                float* outptr = b.channel(q);

                for (int j = 0; j < w * h; j++)
                {
                    outptr[j] = reduction(v0, ptr + j, d, w * h, op_type);
                }
            }
        }
    }

    if (operation == Reduction::ReductionOp_LogSum || operation == Reduction::ReductionOp_LogSumExp)
    {
        const int size = b.total();

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < size; i++)
        {
            b[i] = logf(b[i]);
        }
    }

    if (operation == Reduction::ReductionOp_L2)
    {
        const int size = b.total();

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < size; i++)
        {
            // math optimization will probably generate rsqrt
            // that produce -inf on sse with subnormal input
            // flush subnormal input to zero as a workaround
            // TODO explicit use simd sqrt like unaryop     --- nihui
            b[i] = sqrtf(b[i] < FLT_MIN ? 0.f : b[i]);
        }
    }

    if (operation == Reduction::ReductionOp_MEAN)
    {
        int scale = 1;
        if (dims == 1)
        {
            scale = a.w;
        }
        if (dims == 2)
        {
            if (reduce_w) scale *= a.w;
            if (reduce_h) scale *= a.h;
        }
        if (dims == 3)
        {
            if (reduce_w) scale *= a.w;
            if (reduce_h) scale *= a.h;
            if (reduce_c) scale *= a.c;
        }
        if (dims == 4)
        {
            if (reduce_w) scale *= a.w;
            if (reduce_h) scale *= a.h;
            if (reduce_d) scale *= a.d;
            if (reduce_c) scale *= a.c;
        }

        coeff = coeff / scale;
    }

    if (coeff != 1.f)
    {
        const int size = b.total();

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < size; i++)
        {
            b[i] = b[i] * coeff;
        }
    }

    return 0;
}

int Reduction::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int dims = bottom_blob.dims;
    int axes_flag[4] = {0};
    bool reduce_w = false;
    bool reduce_h = false;
    bool reduce_d = false;
    bool reduce_c = false;

    if (reduce_all)
    {
        reduce_w = true;
        reduce_h = true;
        reduce_d = true;
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
                axis += dims;
            axes_flag[axis] = 1;
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
        else if (dims == 4)
        {
            if (axes_flag[0] == 1) reduce_c = true;
            if (axes_flag[1] == 1) reduce_d = true;
            if (axes_flag[2] == 1) reduce_h = true;
            if (axes_flag[3] == 1) reduce_w = true;
        }
    }

    return reduction_op(bottom_blob, top_blob, reduce_w, reduce_h, reduce_d, reduce_c, keepdims, operation, coeff, opt);
}

} // namespace ncnn
