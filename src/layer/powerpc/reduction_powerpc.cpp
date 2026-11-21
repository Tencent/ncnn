// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "reduction_powerpc.h"

#include <float.h>
#include <math.h>

#include <algorithm>

#if __ALTIVEC__
#include "powerpc_usability.h"
#endif // __ALTIVEC__

namespace ncnn {

Reduction_powerpc::Reduction_powerpc()
{
    // base defaults kept: one_blob_only = true, support_inplace = false
    // support_packing stays false, the generic index math and output shapes
    // are reused and only pack1 fp32 contiguous reductions are vectorized
}

#if __ALTIVEC__

template<typename Op>
static float reduction_scalar(float v0, const float* ptr, int size)
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
static float reduction_scalar(float v0, const float* ptr, int size, int stride)
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
static float reduction_scalar(float v0, const float* ptr, int size0, int size1, int stride1)
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
static float reduction_scalar(float v0, const float* ptr, int size0, int stride0, int size1, int stride1)
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

// contiguous memory, vectorized for SUM / ASUM / SUMSQ / MAX / MIN
static float reduction(float v0, const float* ptr, int size, int op_type)
{
    if (size >= 4)
    {
        int i = 0;

        if (op_type == Reduction::ReductionOp_SUM)
        {
            v4f32 _s = vec_splats(0.f);
            for (; i + 4 <= size; i += 4)
            {
                _s = vec_add(_s, __ncnn_altivec_load(ptr + i));
            }
            float sum = __ncnn_altivec_reduce_add(_s);
            for (; i < size; i++)
            {
                sum += ptr[i];
            }
            return sum;
        }

        if (op_type == Reduction::ReductionOp_ASUM)
        {
            v4f32 _s = vec_splats(0.f);
            for (; i + 4 <= size; i += 4)
            {
                _s = vec_add(_s, vec_abs(__ncnn_altivec_load(ptr + i)));
            }
            float sum = __ncnn_altivec_reduce_add(_s);
            for (; i < size; i++)
            {
                sum += fabsf(ptr[i]);
            }
            return sum;
        }

        if (op_type == Reduction::ReductionOp_SUMSQ)
        {
            v4f32 _s = vec_splats(0.f);
            for (; i + 4 <= size; i += 4)
            {
                v4f32 _v = __ncnn_altivec_load(ptr + i);
                _s = vec_add(_s, __ncnn_altivec_mul(_v, _v));
            }
            float sum = __ncnn_altivec_reduce_add(_s);
            for (; i < size; i++)
            {
                sum += ptr[i] * ptr[i];
            }
            return sum;
        }

        if (op_type == Reduction::ReductionOp_MAX)
        {
            v4f32 _s = vec_splats(-FLT_MAX);
            for (; i + 4 <= size; i += 4)
            {
                _s = vec_max(_s, __ncnn_altivec_load(ptr + i));
            }
            float maxv = __ncnn_altivec_reduce_max(_s);
            for (; i < size; i++)
            {
                maxv = std::max(maxv, ptr[i]);
            }
            return maxv;
        }

        if (op_type == Reduction::ReductionOp_MIN)
        {
            v4f32 _s = vec_splats(FLT_MAX);
            for (; i + 4 <= size; i += 4)
            {
                _s = vec_min(_s, __ncnn_altivec_load(ptr + i));
            }
            float minv = __ncnn_altivec_reduce_min(_s);
            for (; i < size; i++)
            {
                minv = std::min(minv, ptr[i]);
            }
            return minv;
        }
    }

    // PROD / LogSumExp and short inputs stay scalar
    if (op_type == Reduction::ReductionOp_SUM) return reduction_scalar<reduction_op_add>(v0, ptr, size);
    if (op_type == Reduction::ReductionOp_ASUM) return reduction_scalar<reduction_op_asum>(v0, ptr, size);
    if (op_type == Reduction::ReductionOp_SUMSQ) return reduction_scalar<reduction_op_sumsq>(v0, ptr, size);
    if (op_type == Reduction::ReductionOp_PROD) return reduction_scalar<reduction_op_mul>(v0, ptr, size);
    if (op_type == Reduction::ReductionOp_MAX) return reduction_scalar<reduction_op_max>(v0, ptr, size);
    if (op_type == Reduction::ReductionOp_MIN) return reduction_scalar<reduction_op_min>(v0, ptr, size);
    if (op_type == Reduction::ReductionOp_LogSumExp) return reduction_scalar<reduction_op_sumexp>(v0, ptr, size);

    // should never reach here
    return v0;
}

static float reduction(float v0, const float* ptr, int size, int stride, int op_type)
{
    if (op_type == Reduction::ReductionOp_SUM) return reduction_scalar<reduction_op_add>(v0, ptr, size, stride);
    if (op_type == Reduction::ReductionOp_ASUM) return reduction_scalar<reduction_op_asum>(v0, ptr, size, stride);
    if (op_type == Reduction::ReductionOp_SUMSQ) return reduction_scalar<reduction_op_sumsq>(v0, ptr, size, stride);
    if (op_type == Reduction::ReductionOp_PROD) return reduction_scalar<reduction_op_mul>(v0, ptr, size, stride);
    if (op_type == Reduction::ReductionOp_MAX) return reduction_scalar<reduction_op_max>(v0, ptr, size, stride);
    if (op_type == Reduction::ReductionOp_MIN) return reduction_scalar<reduction_op_min>(v0, ptr, size, stride);
    if (op_type == Reduction::ReductionOp_LogSumExp) return reduction_scalar<reduction_op_sumexp>(v0, ptr, size, stride);

    // should never reach here
    return v0;
}

static float reduction(float v0, const float* ptr, int size0, int size1, int stride1, int op_type)
{
    if (op_type == Reduction::ReductionOp_SUM) return reduction_scalar<reduction_op_add>(v0, ptr, size0, size1, stride1);
    if (op_type == Reduction::ReductionOp_ASUM) return reduction_scalar<reduction_op_asum>(v0, ptr, size0, size1, stride1);
    if (op_type == Reduction::ReductionOp_SUMSQ) return reduction_scalar<reduction_op_sumsq>(v0, ptr, size0, size1, stride1);
    if (op_type == Reduction::ReductionOp_PROD) return reduction_scalar<reduction_op_mul>(v0, ptr, size0, size1, stride1);
    if (op_type == Reduction::ReductionOp_MAX) return reduction_scalar<reduction_op_max>(v0, ptr, size0, size1, stride1);
    if (op_type == Reduction::ReductionOp_MIN) return reduction_scalar<reduction_op_min>(v0, ptr, size0, size1, stride1);
    if (op_type == Reduction::ReductionOp_LogSumExp) return reduction_scalar<reduction_op_sumexp>(v0, ptr, size0, size1, stride1);

    // should never reach here
    return v0;
}

static float reduction(float v0, const float* ptr, int size0, int stride0, int size1, int stride1, int op_type)
{
    if (op_type == Reduction::ReductionOp_SUM) return reduction_scalar<reduction_op_add>(v0, ptr, size0, stride0, size1, stride1);
    if (op_type == Reduction::ReductionOp_ASUM) return reduction_scalar<reduction_op_asum>(v0, ptr, size0, stride0, size1, stride1);
    if (op_type == Reduction::ReductionOp_SUMSQ) return reduction_scalar<reduction_op_sumsq>(v0, ptr, size0, stride0, size1, stride1);
    if (op_type == Reduction::ReductionOp_PROD) return reduction_scalar<reduction_op_mul>(v0, ptr, size0, stride0, size1, stride1);
    if (op_type == Reduction::ReductionOp_MAX) return reduction_scalar<reduction_op_max>(v0, ptr, size0, stride0, size1, stride1);
    if (op_type == Reduction::ReductionOp_MIN) return reduction_scalar<reduction_op_min>(v0, ptr, size0, stride0, size1, stride1);
    if (op_type == Reduction::ReductionOp_LogSumExp) return reduction_scalar<reduction_op_sumexp>(v0, ptr, size0, stride0, size1, stride1);

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

#endif // __ALTIVEC__

int Reduction_powerpc::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if __ALTIVEC__
    if (bottom_blob.elempack == 1 && bottom_blob.elemsize == 4u)
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
#endif // __ALTIVEC__

    return Reduction::forward(bottom_blob, top_blob, opt);
}

} // namespace ncnn
