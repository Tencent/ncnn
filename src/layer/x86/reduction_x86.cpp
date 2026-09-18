// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "reduction_x86.h"

#include <float.h>
#include <math.h>

#if __SSE2__
#include <emmintrin.h>
#include "sse_mathfun.h"
#if __AVX__
#include <immintrin.h>
#include "avx_mathfun.h"
#if __AVX512F__
#include "avx512_mathfun.h"
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__

#include "x86_usability.h"

namespace ncnn {

#include "reduction_fp32.h"

static int reduction_op(const Mat& a, Mat& b, bool reduce_w, bool reduce_h, bool reduce_d, bool reduce_c, float v0, int op_type, int op2_type, const Option& opt)
{
    int w = a.w;
    int h = a.h;
    int d = a.d;
    const int channels = a.c;

    // removing c from a 4d tensor makes d the output channel axis
    // keep these planes separate because the output may have channel padding
    if (a.dims == 4 && reduce_c && !reduce_w && !reduce_h && !reduce_d && b.dims == 3)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int z = 0; z < d; z++)
        {
            Mat out = b.channel(z);
            out.fill(v0);
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q).depth(z);
                reduction_vector(ptr, out, w * h, op_type);
            }
        }
        return 0;
    }

    // collapse adjacent spatial axes with the same reduction flag
    // channels are kept separate to preserve cstep
    if (reduce_w == reduce_h)
    {
        w *= h;
        h = d;
        reduce_h = reduce_d;
        d = 1;
        reduce_d = false;
    }
    if (reduce_h == reduce_d)
    {
        h *= d;
        d = 1;
        reduce_d = false;
    }
    if (reduce_w == reduce_h)
    {
        w *= h;
        h = 1;
        reduce_h = false;
    }

    if (reduce_w && h == 1 && d == 1)
    {
        Mat sums = b;
        if (reduce_c && channels > 1)
        {
            sums.create(channels, a.elemsize, opt.workspace_allocator);
            if (sums.empty())
                return -100;
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = a.channel(q);
            float* outptr = sums.dims >= 3 ? sums.channel(q) : (float*)sums + q;
            outptr[0] = reduction(v0, ptr, w, op_type);
        }

        if (reduce_c && channels > 1)
            b[0] = reduction(v0, sums, channels, op2_type);
        return 0;
    }

    // each contiguous row produces one output, with no partial sums to merge
    if (reduce_w && !reduce_h && !reduce_d && !reduce_c)
    {
        const int rows = h * d;
        const size_t out_cstep = b.dims >= 3 ? b.cstep : rows;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < channels * rows; i++)
        {
            const int q = i / rows;
            const int y = i % rows;
            const float* ptr = (const float*)a + q * a.cstep + y * w;
            float* outptr = (float*)b + q * out_cstep;
            outptr[y] = reduction(v0, ptr, w, op_type);
        }
        return 0;
    }

    const int outw = reduce_w ? 1 : w;
    const int outh = reduce_h ? 1 : h;
    const int outd = reduce_d ? 1 : d;
    const int outc = reduce_c ? 1 : channels;
    const int size = b.w * b.h * b.d;

    const int reduced_h = reduce_h ? h : 1;
    const int reduced_d = reduce_d ? d : 1;
    const int reduced_c = reduce_c ? channels : 1;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < outc * outd * outh; i++)
    {
        const int q = i / (outd * outh);
        const int z = i / outh % outd;
        const int y = i % outh;
        const size_t offset = (size_t)i * outw;
        float* outptr = (float*)b + offset / size * b.cstep + offset % size;
        const float* ptr = (const float*)a + q * a.cstep + (z * h + y) * w;

        if (reduce_w)
        {
            float sum = v0;
            for (int qc = 0; qc < reduced_c; qc++)
            {
                for (int zd = 0; zd < reduced_d; zd++)
                {
                    const float* ptr0 = ptr + qc * a.cstep + zd * w * h;
                    for (int yh = 0; yh < reduced_h; yh++)
                    {
                        float v = reduction(v0, ptr0, w, op_type);
                        sum = reduction(sum, v, op2_type);
                        ptr0 += w;
                    }
                }
            }
            outptr[0] = sum;
        }
        else
        {
            for (int j = 0; j < outw; j++)
                outptr[j] = v0;
            for (int qc = 0; qc < reduced_c; qc++)
            {
                for (int zd = 0; zd < reduced_d; zd++)
                {
                    const float* ptr0 = ptr + qc * a.cstep + zd * w * h;
                    for (int yh = 0; yh < reduced_h; yh++)
                    {
                        reduction_vector(ptr0, outptr, outw, op_type);
                        ptr0 += w;
                    }
                }
            }
        }
    }

    return 0;
}

int Reduction_x86::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    bool reduce_w, reduce_h, reduce_d, reduce_c;
    int outdims, outw, outh, outd, outc;
    resolve_reduce_flags_and_output_shape(bottom_blob, reduce_w, reduce_h, reduce_d, reduce_c, outdims, outw, outh, outd, outc);

    if (outdims == 0)
        top_blob.create(1, bottom_blob.elemsize, opt.blob_allocator);
    if (outdims == 1)
        top_blob.create(outw, bottom_blob.elemsize, opt.blob_allocator);
    if (outdims == 2)
        top_blob.create(outw, outh, bottom_blob.elemsize, opt.blob_allocator);
    if (outdims == 3)
        top_blob.create(outw, outh, outc, bottom_blob.elemsize, opt.blob_allocator);
    if (outdims == 4)
        top_blob.create(outw, outh, outd, outc, bottom_blob.elemsize, opt.blob_allocator);

    if (top_blob.empty())
        return -100;

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

    int ret = reduction_op(bottom_blob, top_blob, reduce_w, reduce_h, reduce_d, reduce_c, v0, op_type, op2_type, opt);
    if (ret != 0)
        return ret;

    float coeff = this->coeff;
    if (operation == ReductionOp_MEAN)
    {
        int scale = 1;
        if (reduce_w) scale *= bottom_blob.w;
        if (reduce_h) scale *= bottom_blob.h;
        if (reduce_d) scale *= bottom_blob.d;
        if (reduce_c) scale *= bottom_blob.c;
        coeff /= scale;
    }

    const int size = top_blob.w * top_blob.h * top_blob.d;
    if (operation == ReductionOp_LogSum || operation == ReductionOp_LogSumExp)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < top_blob.c; q++)
        {
            float* ptr = top_blob.channel(q);
            for (int i = 0; i < size; i++)
                ptr[i] = logf(ptr[i]);
        }
    }

    if (operation == ReductionOp_L2)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < top_blob.c; q++)
        {
            float* ptr = top_blob.channel(q);
            for (int i = 0; i < size; i++)
                ptr[i] = sqrtf(ptr[i] < FLT_MIN ? 0.f : ptr[i]);
        }
    }

    if (coeff != 1.f)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < top_blob.c; q++)
        {
            float* ptr = top_blob.channel(q);
            for (int i = 0; i < size; i++)
                ptr[i] *= coeff;
        }
    }

    return 0;
}

} // namespace ncnn
