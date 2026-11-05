// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__FMA__ && !__FMA4__
void glu_fp32_avx2(const Mat& bottom_blob, Mat& top_blob, int axis, const Option& opt);
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__ && !__FMA4__
void glu_fp32_fma(const Mat& bottom_blob, Mat& top_blob, int axis, const Option& opt);
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA4 && __AVX__ && !__FMA__ && !__FMA4__
void glu_fp32_fma4(const Mat& bottom_blob, Mat& top_blob, int axis, const Option& opt);
#endif

static void glu_vector(const float* ptr, const float* gateptr, float* outptr, int size)
{
    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; i + 15 < size; i += 16)
    {
        __m512 _p = _mm512_loadu_ps(ptr);
        __m512 _g = _mm512_loadu_ps(gateptr);
        _mm512_storeu_ps(outptr, _mm512_mul_ps(_p, sigmoid_avx512(_g)));
        ptr += 16;
        gateptr += 16;
        outptr += 16;
    }
#endif // __AVX512F__
    for (; i + 7 < size; i += 8)
    {
        __m256 _p = _mm256_loadu_ps(ptr);
        __m256 _g = _mm256_loadu_ps(gateptr);
        _mm256_storeu_ps(outptr, _mm256_mul_ps(_p, sigmoid_avx(_g)));
        ptr += 8;
        gateptr += 8;
        outptr += 8;
    }
#endif // __AVX__
    for (; i + 3 < size; i += 4)
    {
        __m128 _p = _mm_loadu_ps(ptr);
        __m128 _g = _mm_loadu_ps(gateptr);
        _mm_storeu_ps(outptr, _mm_mul_ps(_p, sigmoid_sse(_g)));
        ptr += 4;
        gateptr += 4;
        outptr += 4;
    }
#endif // __SSE2__
    for (; i < size; i++)
    {
        *outptr = *ptr * (1.f / (1.f + expf(-*gateptr)));
        ptr++;
        gateptr++;
        outptr++;
    }
}

#if __AVX__
// ptr and gateptr point to the eight lanes of the selected half in each pack16 element
static void glu_pack16to8(const float* ptr, const float* gateptr, float* outptr, int size)
{
    int i = 0;
#if __AVX512F__
    for (; i + 1 < size; i += 2)
    {
        __m512 _p = combine8x2_ps(_mm256_loadu_ps(ptr), _mm256_loadu_ps(ptr + 16));
        __m512 _g = combine8x2_ps(_mm256_loadu_ps(gateptr), _mm256_loadu_ps(gateptr + 16));
        _mm512_storeu_ps(outptr, _mm512_mul_ps(_p, sigmoid_avx512(_g)));
        ptr += 32;
        gateptr += 32;
        outptr += 16;
    }
#endif // __AVX512F__
    for (; i < size; i++)
    {
        __m256 _p = _mm256_loadu_ps(ptr);
        __m256 _g = _mm256_loadu_ps(gateptr);
        _mm256_storeu_ps(outptr, _mm256_mul_ps(_p, sigmoid_avx(_g)));
        ptr += 16;
        gateptr += 16;
        outptr += 8;
    }
}
#endif // __AVX__

#if __SSE2__
// ptr and gateptr point to the four lanes of the selected half in each pack8 element
static void glu_pack8to4(const float* ptr, const float* gateptr, float* outptr, int size)
{
    int i = 0;
#if __AVX__
#if __AVX512F__
    for (; i + 3 < size; i += 4)
    {
        __m512 _p = combine4x4_ps(_mm_loadu_ps(ptr), _mm_loadu_ps(ptr + 8), _mm_loadu_ps(ptr + 16), _mm_loadu_ps(ptr + 24));
        __m512 _g = combine4x4_ps(_mm_loadu_ps(gateptr), _mm_loadu_ps(gateptr + 8), _mm_loadu_ps(gateptr + 16), _mm_loadu_ps(gateptr + 24));
        _mm512_storeu_ps(outptr, _mm512_mul_ps(_p, sigmoid_avx512(_g)));
        ptr += 32;
        gateptr += 32;
        outptr += 16;
    }
#endif // __AVX512F__
    for (; i + 1 < size; i += 2)
    {
        __m256 _p = combine4x2_ps(_mm_loadu_ps(ptr), _mm_loadu_ps(ptr + 8));
        __m256 _g = combine4x2_ps(_mm_loadu_ps(gateptr), _mm_loadu_ps(gateptr + 8));
        _mm256_storeu_ps(outptr, _mm256_mul_ps(_p, sigmoid_avx(_g)));
        ptr += 16;
        gateptr += 16;
        outptr += 8;
    }
#endif // __AVX__
    for (; i < size; i++)
    {
        __m128 _p = _mm_loadu_ps(ptr);
        __m128 _g = _mm_loadu_ps(gateptr);
        _mm_storeu_ps(outptr, _mm_mul_ps(_p, sigmoid_sse(_g)));
        ptr += 8;
        gateptr += 8;
        outptr += 4;
    }
}

// the four lanes of ptr are the values of four pack1 output channels
// their gates are lanes 2 3 of gateptr0 and lanes 0 1 of gateptr1
// only the first two output channels are stored when outch == 2
static void glu_pack4to1(const float* ptr, const float* gateptr0, const float* gateptr1, float* outptr, size_t out_cstep, int outch, int size)
{
    float* outptr0 = outptr;
    float* outptr1 = outptr + out_cstep;
    float* outptr2 = outptr + out_cstep * 2;
    float* outptr3 = outptr + out_cstep * 3;

    int i = 0;
#if __AVX__
#if __AVX512F__
    for (; i + 15 < size; i += 16)
    {
        __m512 _p0 = _mm512_loadu_ps(ptr);
        __m512 _p1 = _mm512_loadu_ps(ptr + 16);
        __m512 _p2 = _mm512_loadu_ps(ptr + 32);
        __m512 _p3 = _mm512_loadu_ps(ptr + 48);
        __m512 _g0 = _mm512_shuffle_ps(_mm512_loadu_ps(gateptr0), _mm512_loadu_ps(gateptr1), _MM_SHUFFLE(1, 0, 3, 2));
        __m512 _g1 = _mm512_shuffle_ps(_mm512_loadu_ps(gateptr0 + 16), _mm512_loadu_ps(gateptr1 + 16), _MM_SHUFFLE(1, 0, 3, 2));
        __m512 _g2 = _mm512_shuffle_ps(_mm512_loadu_ps(gateptr0 + 32), _mm512_loadu_ps(gateptr1 + 32), _MM_SHUFFLE(1, 0, 3, 2));
        __m512 _g3 = _mm512_shuffle_ps(_mm512_loadu_ps(gateptr0 + 48), _mm512_loadu_ps(gateptr1 + 48), _MM_SHUFFLE(1, 0, 3, 2));
        _p0 = _mm512_mul_ps(_p0, sigmoid_avx512(_g0));
        _p1 = _mm512_mul_ps(_p1, sigmoid_avx512(_g1));
        _p2 = _mm512_mul_ps(_p2, sigmoid_avx512(_g2));
        _p3 = _mm512_mul_ps(_p3, sigmoid_avx512(_g3));

        // transpose 16x4 to 4x16
        __m512 _tmp0 = _mm512_shuffle_f32x4(_p0, _p1, _MM_SHUFFLE(1, 0, 1, 0));
        __m512 _tmp1 = _mm512_shuffle_f32x4(_p2, _p3, _MM_SHUFFLE(1, 0, 1, 0));
        __m512 _tmp2 = _mm512_shuffle_f32x4(_p0, _p1, _MM_SHUFFLE(3, 2, 3, 2));
        __m512 _tmp3 = _mm512_shuffle_f32x4(_p2, _p3, _MM_SHUFFLE(3, 2, 3, 2));
        __m512 _tmp4 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(2, 0, 2, 0));
        __m512 _tmp5 = _mm512_shuffle_f32x4(_tmp0, _tmp1, _MM_SHUFFLE(3, 1, 3, 1));
        __m512 _tmp6 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(2, 0, 2, 0));
        __m512 _tmp7 = _mm512_shuffle_f32x4(_tmp2, _tmp3, _MM_SHUFFLE(3, 1, 3, 1));
        _tmp0 = _mm512_unpacklo_ps(_tmp4, _tmp5);
        _tmp1 = _mm512_unpackhi_ps(_tmp4, _tmp5);
        _tmp2 = _mm512_unpacklo_ps(_tmp6, _tmp7);
        _tmp3 = _mm512_unpackhi_ps(_tmp6, _tmp7);
        _p0 = _mm512_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(1, 0, 1, 0));
        _p1 = _mm512_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(3, 2, 3, 2));
        _p2 = _mm512_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(1, 0, 1, 0));
        _p3 = _mm512_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(3, 2, 3, 2));

        _mm512_storeu_ps(outptr0, _p0);
        _mm512_storeu_ps(outptr1, _p1);
        if (outch == 4)
        {
            _mm512_storeu_ps(outptr2, _p2);
            _mm512_storeu_ps(outptr3, _p3);
        }

        ptr += 64;
        gateptr0 += 64;
        gateptr1 += 64;
        outptr0 += 16;
        outptr1 += 16;
        outptr2 += 16;
        outptr3 += 16;
    }
#endif // __AVX512F__
    for (; i + 7 < size; i += 8)
    {
        __m256 _p0 = _mm256_loadu_ps(ptr);
        __m256 _p1 = _mm256_loadu_ps(ptr + 8);
        __m256 _p2 = _mm256_loadu_ps(ptr + 16);
        __m256 _p3 = _mm256_loadu_ps(ptr + 24);
        __m256 _g0 = _mm256_shuffle_ps(_mm256_loadu_ps(gateptr0), _mm256_loadu_ps(gateptr1), _MM_SHUFFLE(1, 0, 3, 2));
        __m256 _g1 = _mm256_shuffle_ps(_mm256_loadu_ps(gateptr0 + 8), _mm256_loadu_ps(gateptr1 + 8), _MM_SHUFFLE(1, 0, 3, 2));
        __m256 _g2 = _mm256_shuffle_ps(_mm256_loadu_ps(gateptr0 + 16), _mm256_loadu_ps(gateptr1 + 16), _MM_SHUFFLE(1, 0, 3, 2));
        __m256 _g3 = _mm256_shuffle_ps(_mm256_loadu_ps(gateptr0 + 24), _mm256_loadu_ps(gateptr1 + 24), _MM_SHUFFLE(1, 0, 3, 2));
        _p0 = _mm256_mul_ps(_p0, sigmoid_avx(_g0));
        _p1 = _mm256_mul_ps(_p1, sigmoid_avx(_g1));
        _p2 = _mm256_mul_ps(_p2, sigmoid_avx(_g2));
        _p3 = _mm256_mul_ps(_p3, sigmoid_avx(_g3));

        // transpose 8x4 to 4x8
        __m256 _tmp0 = _mm256_permute2f128_ps(_p0, _p2, _MM_SHUFFLE(0, 2, 0, 0));
        __m256 _tmp1 = _mm256_permute2f128_ps(_p0, _p2, _MM_SHUFFLE(0, 3, 0, 1));
        __m256 _tmp2 = _mm256_permute2f128_ps(_p1, _p3, _MM_SHUFFLE(0, 2, 0, 0));
        __m256 _tmp3 = _mm256_permute2f128_ps(_p1, _p3, _MM_SHUFFLE(0, 3, 0, 1));
        __m256 _tmp4 = _mm256_unpacklo_ps(_tmp0, _tmp1);
        __m256 _tmp5 = _mm256_unpackhi_ps(_tmp0, _tmp1);
        __m256 _tmp6 = _mm256_unpacklo_ps(_tmp2, _tmp3);
        __m256 _tmp7 = _mm256_unpackhi_ps(_tmp2, _tmp3);
        _p0 = _mm256_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(1, 0, 1, 0));
        _p1 = _mm256_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(3, 2, 3, 2));
        _p2 = _mm256_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(1, 0, 1, 0));
        _p3 = _mm256_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(3, 2, 3, 2));

        _mm256_storeu_ps(outptr0, _p0);
        _mm256_storeu_ps(outptr1, _p1);
        if (outch == 4)
        {
            _mm256_storeu_ps(outptr2, _p2);
            _mm256_storeu_ps(outptr3, _p3);
        }

        ptr += 32;
        gateptr0 += 32;
        gateptr1 += 32;
        outptr0 += 8;
        outptr1 += 8;
        outptr2 += 8;
        outptr3 += 8;
    }
#endif // __AVX__
    for (; i + 3 < size; i += 4)
    {
        __m128 _p0 = _mm_loadu_ps(ptr);
        __m128 _p1 = _mm_loadu_ps(ptr + 4);
        __m128 _p2 = _mm_loadu_ps(ptr + 8);
        __m128 _p3 = _mm_loadu_ps(ptr + 12);
        __m128 _g0 = _mm_shuffle_ps(_mm_loadu_ps(gateptr0), _mm_loadu_ps(gateptr1), _MM_SHUFFLE(1, 0, 3, 2));
        __m128 _g1 = _mm_shuffle_ps(_mm_loadu_ps(gateptr0 + 4), _mm_loadu_ps(gateptr1 + 4), _MM_SHUFFLE(1, 0, 3, 2));
        __m128 _g2 = _mm_shuffle_ps(_mm_loadu_ps(gateptr0 + 8), _mm_loadu_ps(gateptr1 + 8), _MM_SHUFFLE(1, 0, 3, 2));
        __m128 _g3 = _mm_shuffle_ps(_mm_loadu_ps(gateptr0 + 12), _mm_loadu_ps(gateptr1 + 12), _MM_SHUFFLE(1, 0, 3, 2));
        _p0 = _mm_mul_ps(_p0, sigmoid_sse(_g0));
        _p1 = _mm_mul_ps(_p1, sigmoid_sse(_g1));
        _p2 = _mm_mul_ps(_p2, sigmoid_sse(_g2));
        _p3 = _mm_mul_ps(_p3, sigmoid_sse(_g3));

        _MM_TRANSPOSE4_PS(_p0, _p1, _p2, _p3);

        _mm_storeu_ps(outptr0, _p0);
        _mm_storeu_ps(outptr1, _p1);
        if (outch == 4)
        {
            _mm_storeu_ps(outptr2, _p2);
            _mm_storeu_ps(outptr3, _p3);
        }

        ptr += 16;
        gateptr0 += 16;
        gateptr1 += 16;
        outptr0 += 4;
        outptr1 += 4;
        outptr2 += 4;
        outptr3 += 4;
    }
    for (; i < size; i++)
    {
        outptr0[0] = ptr[0] * (1.f / (1.f + expf(-gateptr0[2])));
        outptr1[0] = ptr[1] * (1.f / (1.f + expf(-gateptr0[3])));
        if (outch == 4)
        {
            outptr2[0] = ptr[2] * (1.f / (1.f + expf(-gateptr1[0])));
            outptr3[0] = ptr[3] * (1.f / (1.f + expf(-gateptr1[1])));
        }

        ptr += 4;
        gateptr0 += 4;
        gateptr1 += 4;
        outptr0++;
        outptr1++;
        outptr2++;
        outptr3++;
    }
}
#endif // __SSE2__

static void glu_fp32(const Mat& bottom_blob, Mat& top_blob, int axis, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__FMA__ && !__FMA4__
    if (ncnn::cpu_support_x86_avx2() && ncnn::cpu_support_x86_fma())
    {
        glu_fp32_avx2(bottom_blob, top_blob, axis, opt);
        return;
    }
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__ && !__FMA4__
    if (ncnn::cpu_support_x86_fma())
    {
        glu_fp32_fma(bottom_blob, top_blob, axis, opt);
        return;
    }
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA4 && __AVX__ && !__FMA__ && !__FMA4__
    if (ncnn::cpu_support_x86_fma4())
    {
        glu_fp32_fma4(bottom_blob, top_blob, axis, opt);
        return;
    }
#endif

    const int dims = bottom_blob.dims;
    const int elempack = bottom_blob.elempack;
    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int outd = top_blob.d;
    const int outc = top_blob.c;
    const int out_elempack = top_blob.elempack;

#if __SSE2__
    if (dims >= 2 && axis == 0 && out_elempack != elempack)
    {
        // the outermost axis is split inside packs, and out_elempack is half of elempack, or 1 when elempack is 4
        const int size = dims == 2 ? outw : outw * outh * outd;
        const int out_outer = dims == 2 ? outh : outc;
        const int out_size = out_outer * out_elempack;
        const size_t cstep = (dims == 2 ? (size_t)bottom_blob.w : bottom_blob.cstep) * elempack;
        const size_t out_cstep = (dims == 2 ? (size_t)outw : top_blob.cstep) * out_elempack;

        if (out_elempack == 1)
        {
            // out_size % 4 == 2, the gates of four output channels start at lane 2 of an input pack
            const int block_size = 256;
            const int nn_outer = (out_outer + 3) / 4;
            const int nn_block = (size + block_size - 1) / block_size;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int ii = 0; ii < nn_outer * nn_block; ii++)
            {
                const int qq = ii / nn_block;
                const int offset = ii % nn_block * block_size;
                const int q = qq * 4;
                const int outch = std::min(out_outer - q, 4);

                const float* ptr = (const float*)bottom_blob + qq * cstep + offset * 4;
                const float* gateptr0 = (const float*)bottom_blob + (q + out_size) / 4 * cstep + offset * 4;
                const float* gateptr1 = outch == 4 ? gateptr0 + cstep : gateptr0;
                float* outptr = (float*)top_blob + q * out_cstep + offset;

                glu_pack4to1(ptr, gateptr0, gateptr1, outptr, out_cstep, outch, std::min(size - offset, block_size));
            }
        }
        else
        {
            const int block_size = 1024 / out_elempack;
            const int nn_block = (size + block_size - 1) / block_size;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int ii = 0; ii < out_outer * nn_block; ii++)
            {
                const int q = ii / nn_block;
                const int offset = ii % nn_block * block_size;
                const int index = q * out_elempack;
                const int gate_index = index + out_size;

                const float* ptr = (const float*)bottom_blob + index / elempack * cstep + index % elempack + offset * elempack;
                const float* gateptr = (const float*)bottom_blob + gate_index / elempack * cstep + gate_index % elempack + offset * elempack;
                float* outptr = (float*)top_blob + q * out_cstep + offset * out_elempack;
                const int count = std::min(size - offset, block_size);

#if __AVX__
                if (elempack == 16)
                    glu_pack16to8(ptr, gateptr, outptr, count);
#endif // __AVX__
                if (elempack == 8)
                    glu_pack8to4(ptr, gateptr, outptr, count);
            }
        }

        return;
    }
#endif // __SSE2__

    // every output channel is a series of contiguous spans
    // dims == 1 is contiguous whatever the packing is
    const int inner_axis = dims - 1 - axis;
    int span_size = outw * out_elempack;
    if (inner_axis >= 1)
        span_size *= outh;
    if (inner_axis >= 2)
        span_size *= outd;

    int input_stride = bottom_blob.w * elempack;
    if (inner_axis >= 1)
        input_stride *= bottom_blob.h;
    if (inner_axis >= 2)
        input_stride *= bottom_blob.d;

    const int spans = outw * outh * outd * out_elempack / span_size;
    const size_t gate_offset = dims >= 3 && axis == 0 ? outc * bottom_blob.cstep * elempack : (size_t)span_size;

    // gather short spans and split long spans, so that each task handles about block_size elements
    const int block_size = 1024;
    const int span_count = span_size < block_size ? block_size / span_size : 1;
    const int nn_span = (spans + span_count - 1) / span_count;
    const int nn_block = (span_size + block_size - 1) / block_size;

    if (span_size < 4)
    {
        // spans shorter than one vector, plain scalar code avoids entering the simd kernel for every span
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < outc * nn_span; ii++)
        {
            const int q = ii / nn_span;
            const int span_start = ii % nn_span * span_count;
            const int span_end = std::min(span_start + span_count, spans);

            const float* ptr = (const float*)bottom_blob + q * bottom_blob.cstep * elempack + span_start * input_stride;
            float* outptr = (float*)top_blob + q * top_blob.cstep * out_elempack + span_start * span_size;

            for (int j = span_start; j < span_end; j++)
            {
                for (int k = 0; k < span_size; k++)
                {
                    outptr[k] = ptr[k] * (1.f / (1.f + expf(-ptr[k + gate_offset])));
                }
                ptr += input_stride;
                outptr += span_size;
            }
        }

        return;
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ii = 0; ii < outc * nn_span * nn_block; ii++)
    {
        const int q = ii / (nn_span * nn_block);
        const int span_start = ii % (nn_span * nn_block) / nn_block * span_count;
        const int span_end = std::min(span_start + span_count, spans);
        const int offset = ii % nn_block * block_size;
        const int size = std::min(span_size - offset, block_size);

        const float* ptr = (const float*)bottom_blob + q * bottom_blob.cstep * elempack + span_start * input_stride + offset;
        float* outptr = (float*)top_blob + q * top_blob.cstep * out_elempack + span_start * span_size + offset;

        for (int j = span_start; j < span_end; j++)
        {
            glu_vector(ptr, ptr + gate_offset, outptr, size);
            ptr += input_stride;
            outptr += span_size;
        }
    }
}
