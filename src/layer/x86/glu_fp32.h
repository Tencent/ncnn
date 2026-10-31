// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__FMA__ && !__FMA4__
int glu_fp32_avx2(const Mat& input, Mat& output, int axis, const Option& opt);
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__ && !__FMA4__
int glu_fp32_fma(const Mat& input, Mat& output, int axis, const Option& opt);
#endif

static void glu(const float* input, const float* gate, float* output, int size)
{
    int i = 0;
#if __AVX512F__
    const __m512 _one_avx512 = _mm512_set1_ps(1.f);
    const __m512 _zero_avx512 = _mm512_setzero_ps();
    for (; i + 15 < size; i += 16)
    {
        const __m512 _input = _mm512_loadu_ps(input + i);
        const __m512 _gate = _mm512_loadu_ps(gate + i);
        const __m512 _sigmoid = _mm512_div_ps(_one_avx512, _mm512_add_ps(_one_avx512, exp512_ps(_mm512_sub_ps(_zero_avx512, _gate))));
        _mm512_storeu_ps(output + i, _mm512_mul_ps(_input, _sigmoid));
    }
    if (i < size)
    {
        const __mmask16 _mask = (__mmask16)((1u << (size - i)) - 1);
        const __m512 _input = _mm512_maskz_loadu_ps(_mask, input + i);
        const __m512 _gate = _mm512_maskz_loadu_ps(_mask, gate + i);
        const __m512 _sigmoid = _mm512_div_ps(_one_avx512, _mm512_add_ps(_one_avx512, exp512_ps(_mm512_sub_ps(_zero_avx512, _gate))));
        _mm512_mask_storeu_ps(output + i, _mask, _mm512_mul_ps(_input, _sigmoid));
    }
#else // __AVX512F__
#if __AVX__
    const __m256 _one_avx = _mm256_set1_ps(1.f);
    const __m256 _zero_avx = _mm256_setzero_ps();
    for (; i + 7 < size; i += 8)
    {
        const __m256 _input = _mm256_loadu_ps(input + i);
        const __m256 _gate = _mm256_loadu_ps(gate + i);
        const __m256 _sigmoid = _mm256_div_ps(_one_avx, _mm256_add_ps(_one_avx, exp256_ps(_mm256_sub_ps(_zero_avx, _gate))));
        _mm256_storeu_ps(output + i, _mm256_mul_ps(_input, _sigmoid));
    }
#endif // __AVX__
#if __SSE2__
    const __m128 _one = _mm_set1_ps(1.f);
    const __m128 _zero = _mm_setzero_ps();
    for (; i + 3 < size; i += 4)
    {
        const __m128 _input = _mm_loadu_ps(input + i);
        const __m128 _gate = _mm_loadu_ps(gate + i);
        const __m128 _sigmoid = _mm_div_ps(_one, _mm_add_ps(_one, exp_ps(_mm_sub_ps(_zero, _gate))));
        _mm_storeu_ps(output + i, _mm_mul_ps(_input, _sigmoid));
    }
#endif // __SSE2__
    for (; i < size; i++)
    {
        output[i] = input[i] * (1.f / (1.f + expf(-gate[i])));
    }
#endif // __AVX512F__
}

static void glu_pack4to1(const float* input, const float* gate, float* output, int size)
{
    int i = 0;
#if __SSE2__
    const __m128 _one = _mm_set1_ps(1.f);
    const __m128 _zero = _mm_setzero_ps();
    for (; i + 3 < size; i += 4)
    {
        const __m128 _input = _mm_set_ps(input[12], input[8], input[4], input[0]);
        const __m128 _gate = _mm_set_ps(gate[12], gate[8], gate[4], gate[0]);
        const __m128 _sigmoid = _mm_div_ps(_one, _mm_add_ps(_one, exp_ps(_mm_sub_ps(_zero, _gate))));
        _mm_storeu_ps(output, _mm_mul_ps(_input, _sigmoid));
        input += 16;
        gate += 16;
        output += 4;
    }
#endif // __SSE2__
    for (; i < size; i++)
    {
        *output++ = *input * (1.f / (1.f + expf(-*gate)));
        input += 4;
        gate += 4;
    }
}

static void glu_pack(const float* input, const float* gate, float* output, int size, int elempack, int out_elempack)
{
    if (out_elempack == 1)
    {
        glu_pack4to1(input, gate, output, size);
        return;
    }

    // Each value/gate subvector stays within one input pack. Write the result
    // directly in the smaller output pack without an intermediate tensor.
    for (int i = 0; i < size; i++)
    {
        glu(input, gate, output, out_elempack);
        input += elempack;
        gate += elempack;
        output += out_elempack;
    }
}

static int glu_packed_fp32(const Mat& bottom_blob, Mat& top_blob, int axis, const Option& opt)
{
    const int dims = bottom_blob.dims;
    const int positive_axis = dims == 1 ? 0 : axis < 0 ? dims + axis
                                                       : axis;
    if (dims < 1 || dims > 4 || positive_axis < 0 || positive_axis >= dims)
        return -100;

    const int elempack = bottom_blob.elempack;
    if (elempack != 4 && elempack != 8 && elempack != 16)
        return -100;

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;

    if (dims == 1)
    {
        const int size = w * elempack / 2;
        int out_elempack = elempack;
        while (size % out_elempack != 0)
            out_elempack = out_elempack == 4 ? 1 : out_elempack / 2;
        top_blob.create(size / out_elempack, (size_t)4u * out_elempack, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const float* input = bottom_blob;
        float* output = top_blob;
        if (opt.num_threads == 1)
        {
            glu(input, input + size, output, size);
            return 0;
        }

        const int block_size = 1024;
        const int blocks = (size + block_size - 1) / block_size;
#pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < blocks; i++)
        {
            const int offset = i * block_size;
            const int count = std::min(size - offset, block_size);
            glu(input + offset, input + size + offset, output + offset, count);
        }
        return 0;
    }

    if (positive_axis == 0)
    {
        const int out_size = (dims == 2 ? h : channels) * elempack / 2;
        int out_elempack = elempack;
        while (out_size % out_elempack != 0)
            out_elempack = out_elempack == 4 ? 1 : out_elempack / 2;

        if (out_elempack != elempack)
        {
            if (dims == 2)
                top_blob.create(w, out_size / out_elempack, (size_t)4u * out_elempack, out_elempack, opt.blob_allocator);
            else if (dims == 3)
                top_blob.create(w, h, out_size / out_elempack, (size_t)4u * out_elempack, out_elempack, opt.blob_allocator);
            else
                top_blob.create(w, h, d, out_size / out_elempack, (size_t)4u * out_elempack, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            const int size = dims == 2 ? w : w * h * d;
            const size_t input_stride = (dims == 2 ? (size_t)w : bottom_blob.cstep) * elempack;
            const size_t output_stride = (dims == 2 ? (size_t)w : top_blob.cstep) * out_elempack;
            const int out_channels = out_size / out_elempack;
            if (opt.num_threads == 1)
            {
                for (int q = 0; q < out_channels; q++)
                {
                    const int value_index = q * out_elempack;
                    const int gate_index = value_index + out_size;
                    const float* input = (const float*)bottom_blob + value_index / elempack * input_stride + value_index % elempack;
                    const float* gate = (const float*)bottom_blob + gate_index / elempack * input_stride + gate_index % elempack;
                    float* output = (float*)top_blob + q * output_stride;
                    glu_pack(input, gate, output, size, elempack, out_elempack);
                }
                return 0;
            }

            const int block_size = 1024;
            const int blocks_per_channel = (size + block_size - 1) / block_size;
            const int block_count = out_channels * blocks_per_channel;
#pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < block_count; i++)
            {
                const int q = i / blocks_per_channel;
                const int offset = i % blocks_per_channel * block_size;
                const int count = std::min(size - offset, block_size);
                const int value_index = q * out_elempack;
                const int gate_index = value_index + out_size;
                const float* input = (const float*)bottom_blob + value_index / elempack * input_stride + value_index % elempack + offset * elempack;
                const float* gate = (const float*)bottom_blob + gate_index / elempack * input_stride + gate_index % elempack + offset * elempack;
                float* output = (float*)top_blob + q * output_stride + offset * out_elempack;
                glu_pack(input, gate, output, count, elempack, out_elempack);
            }
            return 0;
        }
    }

    if (positive_axis == dims - 1)
        w /= 2;
    else if (positive_axis == dims - 2)
        h /= 2;
    else if (dims == 4 && positive_axis == 1)
        d /= 2;
    else
        channels /= 2;

    if (dims == 2)
        top_blob.create(w, h, (size_t)4u * elempack, elempack, opt.blob_allocator);
    else if (dims == 3)
        top_blob.create(w, h, channels, (size_t)4u * elempack, elempack, opt.blob_allocator);
    else
        top_blob.create(w, h, d, channels, (size_t)4u * elempack, elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const int inner_axis = dims - 1 - positive_axis;
    int span_size = w * elempack;
    if (inner_axis >= 1)
        span_size *= h;
    if (inner_axis >= 2)
        span_size *= d;

    const bool split_channels = dims >= 3 && positive_axis == 0;
    const int spans_per_channel = w * h * d * elempack / span_size;
    const size_t gate_offset = split_channels ? channels * bottom_blob.cstep * elempack : (size_t)span_size;
    int input_stride = bottom_blob.w * elempack;
    if (inner_axis >= 1)
        input_stride *= bottom_blob.h;
    if (inner_axis >= 2)
        input_stride *= bottom_blob.d;

    if (opt.num_threads == 1)
    {
        for (int q = 0; q < channels; q++)
        {
            const float* input = (const float*)bottom_blob + q * bottom_blob.cstep * elempack;
            float* output = (float*)top_blob + q * top_blob.cstep * elempack;
            for (int span = 0; span < spans_per_channel; span++)
            {
                glu(input, input + gate_offset, output, span_size);
                input += input_stride;
                output += span_size;
            }
        }
        return 0;
    }

    const int block_size = 1024;
    const int blocks_per_span = (span_size + block_size - 1) / block_size;
    const int blocks_per_channel = spans_per_channel * blocks_per_span;
    const int block_count = channels * blocks_per_channel;
#pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < block_count; i++)
    {
        const int q = i / blocks_per_channel;
        const int span = i % blocks_per_channel / blocks_per_span;
        const int offset = i % blocks_per_span * block_size;
        const int size = std::min(span_size - offset, block_size);
        const float* input = (const float*)bottom_blob + q * bottom_blob.cstep * elempack + span * input_stride + offset;
        const float* gate = input + gate_offset;
        float* output = (float*)top_blob + q * top_blob.cstep * elempack + span * span_size + offset;
        glu(input, gate, output, size);
    }
    return 0;
}

static int glu_fp32(const Mat& bottom_blob, Mat& top_blob, int axis, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__FMA__ && !__FMA4__
    if (ncnn::cpu_support_x86_avx2() && ncnn::cpu_support_x86_fma())
        return glu_fp32_avx2(bottom_blob, top_blob, axis, opt);
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__ && !__FMA4__
    if (ncnn::cpu_support_x86_fma())
        return glu_fp32_fma(bottom_blob, top_blob, axis, opt);
#endif

    if (bottom_blob.elempack != 1)
        return glu_packed_fp32(bottom_blob, top_blob, axis, opt);

    const int dims = bottom_blob.dims;
    const int positive_axis = dims == 1 ? 0 : axis < 0 ? dims + axis
                                                       : axis;
    if (dims < 1 || dims > 4 || positive_axis < 0 || positive_axis >= dims)
        return -100;

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;

    if (positive_axis == dims - 1)
        w /= 2;
    else if (positive_axis == dims - 2)
        h /= 2;
    else if (dims == 4 && positive_axis == 1)
        d /= 2;
    else
        channels /= 2;

    if (dims == 1)
        top_blob.create(w, 4u, opt.blob_allocator);
    else if (dims == 2)
        top_blob.create(w, h, 4u, opt.blob_allocator);
    else if (dims == 3)
        top_blob.create(w, h, channels, 4u, opt.blob_allocator);
    else
        top_blob.create(w, h, d, channels, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const int inner_axis = dims - 1 - positive_axis;
    int span_size = w;
    if (inner_axis >= 1)
        span_size *= h;
    if (inner_axis >= 2)
        span_size *= d;

    const bool split_channels = dims >= 3 && positive_axis == 0;
    const int spans_per_channel = w * h * d / span_size;
    const size_t gate_offset = split_channels ? channels * bottom_blob.cstep : (size_t)span_size;
    int input_stride = bottom_blob.w;
    if (inner_axis >= 1)
        input_stride *= bottom_blob.h;
    if (inner_axis >= 2)
        input_stride *= bottom_blob.d;

    if (opt.num_threads == 1)
    {
        for (int q = 0; q < channels; q++)
        {
            const float* input = (const float*)bottom_blob + q * bottom_blob.cstep;
            float* output = (float*)top_blob + q * top_blob.cstep;
            for (int span = 0; span < spans_per_channel; span++)
            {
                glu(input, input + gate_offset, output, span_size);
                input += input_stride;
                output += span_size;
            }
        }
        return 0;
    }

    const int block_size = 1024;
    const int blocks_per_span = (span_size + block_size - 1) / block_size;
    const int blocks_per_channel = spans_per_channel * blocks_per_span;
    const int block_count = channels * blocks_per_channel;

#pragma omp parallel for num_threads(opt.num_threads)
    for (int i = 0; i < block_count; i++)
    {
        const int q = i / blocks_per_channel;
        const int span = i % blocks_per_channel / blocks_per_span;
        const int offset = i % blocks_per_span * block_size;
        const int size = span_size - offset < block_size ? span_size - offset : block_size;
        const float* input = (const float*)bottom_blob + q * bottom_blob.cstep + span * input_stride + offset;
        const float* gate = input + gate_offset;
        float* output = (float*)top_blob + q * top_blob.cstep + span * span_size + offset;
        glu(input, gate, output, size);
    }

    return 0;
}
