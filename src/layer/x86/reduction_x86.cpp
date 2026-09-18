// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "reduction_x86.h"

#include "cpu.h"

#include <float.h>

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

    int ret = reduction_fp32(bottom_blob, top_blob, reduce_w, reduce_h, reduce_d, reduce_c, operation, opt);
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
