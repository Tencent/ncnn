// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

namespace reduction_x86_functor {

#include "reduction_functor.h"

} // namespace reduction_x86_functor

#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__AVX512F__ && !__FMA__ && !__FMA4__
int reduction_fp32_fma(const Mat& a, Mat& b, bool reduce_w, bool reduce_h, bool reduce_d, bool reduce_c, int operation, const Option& opt);
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA4 && __AVX__ && !__AVX512F__ && !__FMA__ && !__FMA4__
int reduction_fp32_fma4(const Mat& a, Mat& b, bool reduce_w, bool reduce_h, bool reduce_d, bool reduce_c, int operation, const Option& opt);
#endif

template<typename Op, typename Op2>
static float reduction(float v0, const float* ptr, int size)
{
    const Op op;
    float sum = v0;

    int i = 0;
#if __SSE2__
    const Op2 op2;
#if __AVX__
#if __AVX512F__
    __m512 _sum16_0 = _mm512_set1_ps(v0);
    __m512 _sum16_1 = _sum16_0;
    __m512 _sum16_2 = _sum16_0;
    __m512 _sum16_3 = _sum16_0;
    for (; i + 63 < size; i += 64)
    {
        __m512 _p0 = _mm512_loadu_ps(ptr);
        __m512 _p1 = _mm512_loadu_ps(ptr + 16);
        __m512 _p2 = _mm512_loadu_ps(ptr + 32);
        __m512 _p3 = _mm512_loadu_ps(ptr + 48);
        _sum16_0 = op.func_pack16(_sum16_0, _p0);
        _sum16_1 = op.func_pack16(_sum16_1, _p1);
        _sum16_2 = op.func_pack16(_sum16_2, _p2);
        _sum16_3 = op.func_pack16(_sum16_3, _p3);
        ptr += 64;
    }
    __m512 _sum16 = op2.func_pack16(op2.func_pack16(_sum16_0, _sum16_1), op2.func_pack16(_sum16_2, _sum16_3));
#endif // __AVX512F__
    __m256 _sum8_0 = _mm256_set1_ps(v0);
    __m256 _sum8_1 = _sum8_0;
    __m256 _sum8_2 = _sum8_0;
    __m256 _sum8_3 = _sum8_0;
    for (; i + 31 < size; i += 32)
    {
        __m256 _p0 = _mm256_loadu_ps(ptr);
        __m256 _p1 = _mm256_loadu_ps(ptr + 8);
        __m256 _p2 = _mm256_loadu_ps(ptr + 16);
        __m256 _p3 = _mm256_loadu_ps(ptr + 24);
        _sum8_0 = op.func_pack8(_sum8_0, _p0);
        _sum8_1 = op.func_pack8(_sum8_1, _p1);
        _sum8_2 = op.func_pack8(_sum8_2, _p2);
        _sum8_3 = op.func_pack8(_sum8_3, _p3);
        ptr += 32;
    }
    __m256 _sum8 = op2.func_pack8(op2.func_pack8(_sum8_0, _sum8_1), op2.func_pack8(_sum8_2, _sum8_3));
#endif // __AVX__
    __m128 _sum4_0 = _mm_set1_ps(v0);
    __m128 _sum4_1 = _sum4_0;
    __m128 _sum4_2 = _sum4_0;
    __m128 _sum4_3 = _sum4_0;
    for (; i + 15 < size; i += 16)
    {
        __m128 _p0 = _mm_loadu_ps(ptr);
        __m128 _p1 = _mm_loadu_ps(ptr + 4);
        __m128 _p2 = _mm_loadu_ps(ptr + 8);
        __m128 _p3 = _mm_loadu_ps(ptr + 12);
        _sum4_0 = op.func_pack4(_sum4_0, _p0);
        _sum4_1 = op.func_pack4(_sum4_1, _p1);
        _sum4_2 = op.func_pack4(_sum4_2, _p2);
        _sum4_3 = op.func_pack4(_sum4_3, _p3);
        ptr += 16;
    }
    for (; i + 3 < size; i += 4)
    {
        __m128 _p = _mm_loadu_ps(ptr);
        _sum4_0 = op.func_pack4(_sum4_0, _p);
        ptr += 4;
    }
    __m128 _sum4 = op2.func_pack4(op2.func_pack4(_sum4_0, _sum4_1), op2.func_pack4(_sum4_2, _sum4_3));
#if __AVX__
#if __AVX512F__
    _sum8 = op2.func_pack8(_sum8, op2.func_pack8(_mm512_castps512_ps256(_sum16), _mm512_extractf32x8_ps(_sum16, 1)));
#endif // __AVX512F__
    _sum4 = op2.func_pack4(_sum4, op2.func_pack4(_mm256_castps256_ps128(_sum8), _mm256_extractf128_ps(_sum8, 1)));
#endif // __AVX__
    _sum4 = op2.func_pack4(_sum4, _mm_movehl_ps(_sum4, _sum4));
    _sum4 = op2.func_pack4(_sum4, _mm_shuffle_ps(_sum4, _sum4, 0x55));
    sum = _mm_cvtss_f32(_sum4);
#endif // __SSE2__
    for (; i < size; i++)
    {
        sum = op.func(sum, *ptr);
        ptr++;
    }

    return sum;
}

template<typename Op>
static void reduction_vector(const float* ptr, float* outptr, int size)
{
    const Op op;

    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; i + 63 < size; i += 64)
    {
        __m512 _p0 = _mm512_loadu_ps(ptr);
        __m512 _p1 = _mm512_loadu_ps(ptr + 16);
        __m512 _p2 = _mm512_loadu_ps(ptr + 32);
        __m512 _p3 = _mm512_loadu_ps(ptr + 48);
        __m512 _outp0 = _mm512_loadu_ps(outptr);
        __m512 _outp1 = _mm512_loadu_ps(outptr + 16);
        __m512 _outp2 = _mm512_loadu_ps(outptr + 32);
        __m512 _outp3 = _mm512_loadu_ps(outptr + 48);
        _outp0 = op.func_pack16(_outp0, _p0);
        _outp1 = op.func_pack16(_outp1, _p1);
        _outp2 = op.func_pack16(_outp2, _p2);
        _outp3 = op.func_pack16(_outp3, _p3);
        _mm512_storeu_ps(outptr, _outp0);
        _mm512_storeu_ps(outptr + 16, _outp1);
        _mm512_storeu_ps(outptr + 32, _outp2);
        _mm512_storeu_ps(outptr + 48, _outp3);
        ptr += 64;
        outptr += 64;
    }
#endif // __AVX512F__
    for (; i + 31 < size; i += 32)
    {
        __m256 _p0 = _mm256_loadu_ps(ptr);
        __m256 _p1 = _mm256_loadu_ps(ptr + 8);
        __m256 _p2 = _mm256_loadu_ps(ptr + 16);
        __m256 _p3 = _mm256_loadu_ps(ptr + 24);
        __m256 _outp0 = _mm256_loadu_ps(outptr);
        __m256 _outp1 = _mm256_loadu_ps(outptr + 8);
        __m256 _outp2 = _mm256_loadu_ps(outptr + 16);
        __m256 _outp3 = _mm256_loadu_ps(outptr + 24);
        _outp0 = op.func_pack8(_outp0, _p0);
        _outp1 = op.func_pack8(_outp1, _p1);
        _outp2 = op.func_pack8(_outp2, _p2);
        _outp3 = op.func_pack8(_outp3, _p3);
        _mm256_storeu_ps(outptr, _outp0);
        _mm256_storeu_ps(outptr + 8, _outp1);
        _mm256_storeu_ps(outptr + 16, _outp2);
        _mm256_storeu_ps(outptr + 24, _outp3);
        ptr += 32;
        outptr += 32;
    }
#endif // __AVX__
    for (; i + 15 < size; i += 16)
    {
        __m128 _p0 = _mm_loadu_ps(ptr);
        __m128 _p1 = _mm_loadu_ps(ptr + 4);
        __m128 _p2 = _mm_loadu_ps(ptr + 8);
        __m128 _p3 = _mm_loadu_ps(ptr + 12);
        __m128 _outp0 = _mm_loadu_ps(outptr);
        __m128 _outp1 = _mm_loadu_ps(outptr + 4);
        __m128 _outp2 = _mm_loadu_ps(outptr + 8);
        __m128 _outp3 = _mm_loadu_ps(outptr + 12);
        _outp0 = op.func_pack4(_outp0, _p0);
        _outp1 = op.func_pack4(_outp1, _p1);
        _outp2 = op.func_pack4(_outp2, _p2);
        _outp3 = op.func_pack4(_outp3, _p3);
        _mm_storeu_ps(outptr, _outp0);
        _mm_storeu_ps(outptr + 4, _outp1);
        _mm_storeu_ps(outptr + 8, _outp2);
        _mm_storeu_ps(outptr + 12, _outp3);
        ptr += 16;
        outptr += 16;
    }
    for (; i + 3 < size; i += 4)
    {
        __m128 _p = _mm_loadu_ps(ptr);
        __m128 _outp = _mm_loadu_ps(outptr);
        _outp = op.func_pack4(_outp, _p);
        _mm_storeu_ps(outptr, _outp);
        ptr += 4;
        outptr += 4;
    }
#endif // __SSE2__
    for (; i < size; i++)
    {
        *outptr = op.func(*outptr, *ptr);
        ptr++;
        outptr++;
    }
}

template<typename Op, typename Op2>
static int reduction_op(const Mat& a, Mat& b, bool reduce_w, bool reduce_h, bool reduce_d, bool reduce_c, float v0, const Option& opt)
{
    const Op2 op2;

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
                reduction_vector<Op>(ptr, out, w * h);
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
            outptr[0] = reduction<Op, Op2>(v0, ptr, w);
        }

        if (reduce_c && channels > 1)
            b[0] = reduction<Op2, Op2>(v0, sums, channels);
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
            outptr[y] = reduction<Op, Op2>(v0, ptr, w);
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
                        float v = reduction<Op, Op2>(v0, ptr0, w);
                        sum = op2.func(sum, v);
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
                        reduction_vector<Op>(ptr0, outptr, outw);
                        ptr0 += w;
                    }
                }
            }
        }
    }

    return 0;
}

static int reduction_fp32(const Mat& a, Mat& b, bool reduce_w, bool reduce_h, bool reduce_d, bool reduce_c, int operation, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__AVX512F__ && !__FMA__ && !__FMA4__
    if (ncnn::cpu_support_x86_fma())
        return reduction_fp32_fma(a, b, reduce_w, reduce_h, reduce_d, reduce_c, operation, opt);
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA4 && __AVX__ && !__AVX512F__ && !__FMA__ && !__FMA4__
    if (ncnn::cpu_support_x86_fma4())
        return reduction_fp32_fma4(a, b, reduce_w, reduce_h, reduce_d, reduce_c, operation, opt);
#endif

    using namespace reduction_x86_functor;

    if (operation == Reduction::ReductionOp_SUM || operation == Reduction::ReductionOp_MEAN || operation == Reduction::ReductionOp_LogSum)
        return reduction_op<reduction_op_add, reduction_op_add>(a, b, reduce_w, reduce_h, reduce_d, reduce_c, 0.f, opt);

    if (operation == Reduction::ReductionOp_ASUM || operation == Reduction::ReductionOp_L1)
        return reduction_op<reduction_op_asum, reduction_op_add>(a, b, reduce_w, reduce_h, reduce_d, reduce_c, 0.f, opt);

    if (operation == Reduction::ReductionOp_SUMSQ || operation == Reduction::ReductionOp_L2)
        return reduction_op<reduction_op_sumsq, reduction_op_add>(a, b, reduce_w, reduce_h, reduce_d, reduce_c, 0.f, opt);

    if (operation == Reduction::ReductionOp_MAX)
        return reduction_op<reduction_op_max, reduction_op_max>(a, b, reduce_w, reduce_h, reduce_d, reduce_c, -FLT_MAX, opt);

    if (operation == Reduction::ReductionOp_MIN)
        return reduction_op<reduction_op_min, reduction_op_min>(a, b, reduce_w, reduce_h, reduce_d, reduce_c, FLT_MAX, opt);

    if (operation == Reduction::ReductionOp_PROD)
        return reduction_op<reduction_op_mul, reduction_op_mul>(a, b, reduce_w, reduce_h, reduce_d, reduce_c, 1.f, opt);

    if (operation == Reduction::ReductionOp_LogSumExp)
        return reduction_op<reduction_op_sumexp, reduction_op_add>(a, b, reduce_w, reduce_h, reduce_d, reduce_c, 0.f, opt);

    // should never reach here
    return -1;
}
