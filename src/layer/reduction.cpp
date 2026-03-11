// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

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

    if (dims == 1)
    {
        const int w = a.w;

        b[0] = reduction(v0, a, w, op_type);
    }

    if (dims == 2)
    {
        const int w = a.w;
        const int h = a.h;

        if (reduce_w && reduce_h)
        {
            // w h -> X X
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
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                b[i] = reduction(v0, (const float*)a.row(i), w, channels, a.cstep, op_type);
            }
        }

        if (!reduce_w && reduce_h && reduce_c)
        {
            // w h c -> w X X
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int j = 0; j < w; j++)
            {
                b[j] = reduction(v0, (const float*)a + j, h, w, channels, a.cstep, op_type);
            }
        }

        if (reduce_w && !reduce_h && !reduce_c)
        {
            // w h c -> X h c
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
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < size; i++)
            {
                b[i] = reduction(v0, (const float*)a + i, channels, a.cstep, op_type);
            }
        }

        if (!reduce_w && reduce_h && !reduce_c)
        {
            // w h c -> w X c
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
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < d; i++)
            {
                b[i] = reduction(v0, (const float*)a.depth(i), w * h, channels, a.cstep, op_type);
            }
        }

        if (reduce_w && !reduce_h && reduce_d && reduce_c)
        {
            // w h d c -> X h X X
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
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                b[i] = reduction(v0, (const float*)a + i, h * d, w, channels, a.cstep, op_type);
            }
        }

        if (reduce_w && reduce_h && !reduce_d && !reduce_c)
        {
            // w h d c -> X X d c
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

void Reduction::resolve_reduce_flags_and_output_shape(const Mat& blob, bool& reduce_w, bool& reduce_h, bool& reduce_d, bool& reduce_c, int& outdims, int& outw, int& outh, int& outd, int& outc) const
{
    const int dims = blob.dims;

    // resolve reduce flags
    reduce_w = false;
    reduce_h = false;
    reduce_d = false;
    reduce_c = false;

    if (reduce_all)
    {
        reduce_w = true;
        reduce_h = true;
        reduce_d = true;
        reduce_c = true;
    }
    else
    {
        int axes_flag[4] = {0};

        const int* axes_ptr = axes;
        const int reduced_axes_num = axes.w;

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
        if (dims == 2)
        {
            if (axes_flag[0] == 1) reduce_h = true;
            if (axes_flag[1] == 1) reduce_w = true;
        }
        if (dims == 3)
        {
            if (axes_flag[0] == 1) reduce_c = true;
            if (axes_flag[1] == 1) reduce_h = true;
            if (axes_flag[2] == 1) reduce_w = true;
        }
        if (dims == 4)
        {
            if (axes_flag[0] == 1) reduce_c = true;
            if (axes_flag[1] == 1) reduce_d = true;
            if (axes_flag[2] == 1) reduce_h = true;
            if (axes_flag[3] == 1) reduce_w = true;
        }
    }

    // resolve output shape
    if (keepdims)
    {
        outdims = dims;
        outw = reduce_w ? 1 : blob.w;
        outh = reduce_h ? 1 : blob.h;
        outd = reduce_d ? 1 : blob.d;
        outc = reduce_c ? 1 : blob.c;
    }
    else
    {
        std::vector<int> out_shape;
        if (!reduce_w) out_shape.push_back(blob.w);
        if (dims >= 2 && !reduce_h) out_shape.push_back(blob.h);
        if (dims == 4 && !reduce_d) out_shape.push_back(blob.d);
        if (dims >= 3 && !reduce_c) out_shape.push_back(blob.c);

        outdims = (int)out_shape.size();
        outw = 1;
        outh = 1;
        outd = 1;
        outc = 1;

        if (outdims == 1)
        {
            outw = out_shape[0];
        }
        if (outdims == 2)
        {
            outw = out_shape[0];
            outh = out_shape[1];
        }
        if (outdims == 3)
        {
            outw = out_shape[0];
            outh = out_shape[1];
            outc = out_shape[2];
        }
        if (outdims == 4)
        {
            outw = out_shape[0];
            outh = out_shape[1];
            outd = out_shape[2];
            outc = out_shape[3];
        }
    }
}

int Reduction::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    bool reduce_w, reduce_h, reduce_d, reduce_c;
    int outdims, outw, outh, outd, outc;
    resolve_reduce_flags_and_output_shape(bottom_blob, reduce_w, reduce_h, reduce_d, reduce_c, outdims, outw, outh, outd, outc);

    const size_t elemsize = bottom_blob.elemsize;

    if (outdims == 0)
    {
        top_blob.create(1, elemsize, opt.blob_allocator);
    }
    if (outdims == 1)
    {
        top_blob.create(outw, elemsize, opt.blob_allocator);
    }
    if (outdims == 2)
    {
        top_blob.create(outw, outh, elemsize, opt.blob_allocator);
    }
    if (outdims == 3)
    {
        top_blob.create(outw, outh, outc, elemsize, opt.blob_allocator);
    }
    if (outdims == 4)
    {
        top_blob.create(outw, outh, outd, outc, elemsize, opt.blob_allocator);
    }
    if (top_blob.empty())
        return -100;

    return reduction_op(bottom_blob, top_blob, reduce_w, reduce_h, reduce_d, reduce_c, keepdims, operation, coeff, opt);
}

} // namespace ncnn
