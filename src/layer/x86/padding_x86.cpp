// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "padding_x86.h"

#include "cpu.h"

#include <stdint.h>
#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE2__

namespace ncnn {

#if __SSE2__
#include "padding_pack4.h"
#include "padding_pack4_bf16s_fp16s.h"
#include "padding_pack8_int8.h"
#if __AVX__
#include "padding_pack8.h"
#include "padding_pack8_bf16s_fp16s.h"
#if __AVX512F__
#include "padding_pack16.h"
#include "padding_pack16_bf16s_fp16s.h"
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__

Padding_x86::Padding_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
    support_fp16_storage = cpu_support_x86_f16c();
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int Padding_x86::create_pipeline(const Option& opt)
{
    if (support_fp16_storage && opt.use_fp16_storage)
    {
        value_fp16 = float32_to_float16(value);

        ncnn::cast_float32_to_float16(per_channel_pad_data, per_channel_pad_data_fp16, opt);
    }

#if NCNN_BF16
    if (opt.use_bf16_storage)
    {
        value_bf16 = float32_to_bfloat16(value);

        ncnn::cast_float32_to_bfloat16(per_channel_pad_data, per_channel_pad_data_bf16, opt);
    }
#endif

    return 0;
}

int Padding_x86::destroy_pipeline(const Option& /*opt*/)
{
    per_channel_pad_data_fp16 = Mat();

#if NCNN_BF16
    per_channel_pad_data_bf16 = Mat();
#endif

    return 0;
}

int Padding_x86::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (top == 0 && bottom == 0 && left == 0 && right == 0 && front == 0 && behind == 0)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int elembits = bottom_blob.elembits();

    if (elembits == 8)
        return forward_int8(bottom_blob, top_blob, opt);

    if (elembits == 16)
        return forward_bf16s_fp16s(bottom_blob, top_blob, opt);

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (elempack == 16)
    {
        if (dims == 1)
        {
            int outw = w * elempack + left + right;

            int out_elempack = outw % 16 == 0 ? 16 : outw % 8 == 0 ? 8 : outw % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (left % 16 == 0 && out_elempack == 16 && type == 0)
            {
                top_blob.create(outw / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                __m512 pad_value = _mm512_set1_ps(value);
                padding_constant_pack16_avx512(bottom_blob, top_blob, 0, 0, left / 16, right / 16, pad_value);

                return 0;
            }
        }

        if (dims == 2)
        {
            int outw = w + left + right;
            int outh = h * elempack + top + bottom;

            int out_elempack = outh % 16 == 0 ? 16 : outh % 8 == 0 ? 8 : outh % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (top % 16 == 0 && out_elempack == 16 && type == 0)
            {
                top_blob.create(outw, outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                __m512 pad_value = _mm512_set1_ps(value);
                padding_constant_pack16_avx512(bottom_blob, top_blob, top / 16, bottom / 16, left, right, pad_value);

                return 0;
            }
        }

        if (dims == 3)
        {
            int outw = w + left + right;
            int outh = h + top + bottom;
            int outc = channels * elempack + front + behind;

            int out_elempack = outc % 16 == 0 ? 16 : outc % 8 == 0 ? 8 : outc % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (front % 16 == 0 && out_elempack == 16 && !(outc != channels * elempack && type != 0))
            {
                top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                int front_ = front / elempack;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < outc / out_elempack; q++)
                {
                    Mat borderm = top_blob.channel(q);

                    __m512 pad_value = per_channel_pad_data_size ? _mm512_loadu_ps((const float*)per_channel_pad_data + q * 16) : _mm512_set1_ps(value);
                    //Channel padding
                    if ((q - front_) < 0 || (q - front_) >= channels)
                    {
                        borderm.fill(pad_value);
                    }
                    else
                    {
                        const Mat m = bottom_blob.channel(q - front_);
                        if (type == 0)
                            padding_constant_pack16_avx512(m, borderm, top, bottom, left, right, pad_value);
                        if (type == 1)
                            padding_replicate_pack16_avx512(m, borderm, top, bottom, left, right);
                        if (type == 2)
                            padding_reflect_pack16_avx512(m, borderm, top, bottom, left, right);
                    }
                }

                return 0;
            }
        }

        if (dims == 4)
        {
            int outw = w + left + right;
            int outh = h + top + bottom;
            int outd = d + front + behind;

            if (type == 0)
            {
                top_blob.create(outw, outh, outd, channels, elemsize, elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    __m512 pad_value = per_channel_pad_data_size ? _mm512_loadu_ps((const float*)per_channel_pad_data + q * 16) : _mm512_set1_ps(value);

                    for (int z = 0; z < outd; z++)
                    {
                        Mat borderm = top_blob.channel(q).depth(z);

                        // depth padding
                        if ((z - front) < 0 || (z - front) >= d)
                        {
                            borderm.fill(pad_value);
                        }
                        else
                        {
                            const Mat m = bottom_blob.channel(q).depth(z - front);
                            padding_constant_pack16_avx512(m, borderm, top, bottom, left, right, pad_value);
                        }
                    }
                }

                return 0;
            }
        }
    }
#endif // __AVX512F__

    if (elempack == 8)
    {
        if (dims == 1)
        {
            int outw = w * elempack + left + right;

            int out_elempack = outw % 8 == 0 ? 8 : outw % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (left % 8 == 0 && out_elempack == 8 && type == 0)
            {
                top_blob.create(outw / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                __m256 pad_value = _mm256_set1_ps(value);
                padding_constant_pack8_avx(bottom_blob, top_blob, 0, 0, left / 8, right / 8, pad_value);

                return 0;
            }
        }

        if (dims == 2)
        {
            int outw = w + left + right;
            int outh = h * elempack + top + bottom;

            int out_elempack = outh % 8 == 0 ? 8 : outh % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (top % 8 == 0 && out_elempack == 8 && type == 0)
            {
                top_blob.create(outw, outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                __m256 pad_value = _mm256_set1_ps(value);
                padding_constant_pack8_avx(bottom_blob, top_blob, top / 8, bottom / 8, left, right, pad_value);

                return 0;
            }
        }

        if (dims == 3)
        {
            int outw = w + left + right;
            int outh = h + top + bottom;
            int outc = channels * elempack + front + behind;

            int out_elempack = outc % 8 == 0 ? 8 : outc % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (front % 8 == 0 && out_elempack == 8 && !(outc != channels * elempack && type != 0))
            {
                top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                int front_ = front / elempack;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < outc / out_elempack; q++)
                {
                    Mat borderm = top_blob.channel(q);

                    __m256 pad_value = per_channel_pad_data_size ? _mm256_loadu_ps((const float*)per_channel_pad_data + q * 8) : _mm256_set1_ps(value);
                    //Channel padding
                    if ((q - front_) < 0 || (q - front_) >= channels)
                    {
                        borderm.fill(pad_value);
                    }
                    else
                    {
                        const Mat m = bottom_blob.channel(q - front_);
                        if (type == 0)
                            padding_constant_pack8_avx(m, borderm, top, bottom, left, right, pad_value);
                        if (type == 1)
                            padding_replicate_pack8_avx(m, borderm, top, bottom, left, right);
                        if (type == 2)
                            padding_reflect_pack8_avx(m, borderm, top, bottom, left, right);
                    }
                }

                return 0;
            }
        }

        if (dims == 4)
        {
            int outw = w + left + right;
            int outh = h + top + bottom;
            int outd = d + front + behind;

            if (type == 0)
            {
                top_blob.create(outw, outh, outd, channels, elemsize, elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    __m256 pad_value = per_channel_pad_data_size ? _mm256_loadu_ps((const float*)per_channel_pad_data + q * 8) : _mm256_set1_ps(value);

                    for (int z = 0; z < outd; z++)
                    {
                        Mat borderm = top_blob.channel(q).depth(z);

                        // depth padding
                        if ((z - front) < 0 || (z - front) >= d)
                        {
                            borderm.fill(pad_value);
                        }
                        else
                        {
                            const Mat m = bottom_blob.channel(q).depth(z - front);
                            padding_constant_pack8_avx(m, borderm, top, bottom, left, right, pad_value);
                        }
                    }
                }

                return 0;
            }
        }
    }
#endif // __AVX__

    if (elempack == 4)
    {
        if (dims == 1)
        {
            int outw = w * elempack + left + right;

#if __AVX__
            int out_elempack = outw % 8 == 0 ? 8 : outw % 4 == 0 ? 4 : 1;
#else
            int out_elempack = outw % 4 == 0 ? 4 : 1;
#endif
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (left % 4 == 0 && out_elempack == 4 && type == 0)
            {
                top_blob.create(outw / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                __m128 pad_value = _mm_set1_ps(value);
                padding_constant_pack4_sse(bottom_blob, top_blob, 0, 0, left / 4, right / 4, pad_value);

                return 0;
            }
        }

        if (dims == 2)
        {
            int outw = w + left + right;
            int outh = h * elempack + top + bottom;

#if __AVX__
            int out_elempack = outh % 8 == 0 ? 8 : outh % 4 == 0 ? 4 : 1;
#else
            int out_elempack = outh % 4 == 0 ? 4 : 1;
#endif
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (top % 4 == 0 && out_elempack == 4 && type == 0)
            {
                top_blob.create(outw, outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                __m128 pad_value = _mm_set1_ps(value);
                padding_constant_pack4_sse(bottom_blob, top_blob, top / 4, bottom / 4, left, right, pad_value);

                return 0;
            }
        }

        if (dims == 3)
        {
            int outw = w + left + right;
            int outh = h + top + bottom;
            int outc = channels * elempack + front + behind;

#if __AVX__
            int out_elempack = outc % 8 == 0 ? 8 : outc % 4 == 0 ? 4 : 1;
#else
            int out_elempack = outc % 4 == 0 ? 4 : 1;
#endif
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (front % 4 == 0 && out_elempack == 4 && !(outc != channels * elempack && type != 0))
            {
                top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                int front_ = front / elempack;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < outc / out_elempack; q++)
                {
                    Mat borderm = top_blob.channel(q);

                    __m128 pad_value = per_channel_pad_data_size ? _mm_loadu_ps((const float*)per_channel_pad_data + q * 4) : _mm_set1_ps(value);
                    //Channel padding
                    if ((q - front_) < 0 || (q - front_) >= channels)
                    {
                        borderm.fill(pad_value);
                    }
                    else
                    {
                        const Mat m = bottom_blob.channel(q - front_);
                        if (type == 0)
                            padding_constant_pack4_sse(m, borderm, top, bottom, left, right, pad_value);
                        if (type == 1)
                            padding_replicate_pack4_sse(m, borderm, top, bottom, left, right);
                        if (type == 2)
                            padding_reflect_pack4_sse(m, borderm, top, bottom, left, right);
                    }
                }

                return 0;
            }
        }

        if (dims == 4)
        {
            int outw = w + left + right;
            int outh = h + top + bottom;
            int outd = d + front + behind;

            if (type == 0)
            {
                top_blob.create(outw, outh, outd, channels, elemsize, elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    __m128 pad_value = per_channel_pad_data_size ? _mm_loadu_ps((const float*)per_channel_pad_data + q * 4) : _mm_set1_ps(value);

                    for (int z = 0; z < outd; z++)
                    {
                        Mat borderm = top_blob.channel(q).depth(z);

                        // depth padding
                        if ((z - front) < 0 || (z - front) >= d)
                        {
                            borderm.fill(pad_value);
                        }
                        else
                        {
                            const Mat m = bottom_blob.channel(q).depth(z - front);
                            padding_constant_pack4_sse(m, borderm, top, bottom, left, right, pad_value);
                        }
                    }
                }

                return 0;
            }
        }
    }
#endif // __SSE2__

    Mat bottom_blob_unpacked = bottom_blob;
    if (elempack != 1)
    {
        Option opt_pack1 = opt;
        opt_pack1.blob_allocator = opt.workspace_allocator;

        convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack1);
        if (bottom_blob_unpacked.empty())
            return -100;
    }

    return Padding::forward(bottom_blob_unpacked, top_blob, opt);
}

int Padding_x86::forward_bf16s_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    unsigned short pad_value_bf16_fp16 = 0;
    if (support_fp16_storage && opt.use_fp16_storage)
    {
        pad_value_bf16_fp16 = value_fp16;
    }
    else
#if NCNN_BF16
    if (opt.use_bf16_storage)
    {
        pad_value_bf16_fp16 = value_bf16;
    }
    else
#endif
    {
        pad_value_bf16_fp16 = 0;
    }

    const Mat& per_channel_pad_data_bf16_fp16 =
        (support_fp16_storage && opt.use_fp16_storage) ? per_channel_pad_data_fp16 :
#if NCNN_BF16
        opt.use_bf16_storage ? per_channel_pad_data_bf16 :
#endif
        Mat();

#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (elempack == 16)
    {
        if (dims == 1)
        {
            int outw = w * elempack + left + right;

            int out_elempack = outw % 16 == 0 ? 16 : outw % 8 == 0 ? 8 : outw % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (left % 16 == 0 && out_elempack == 16 && type == 0)
            {
                top_blob.create(outw / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                __m256i pad_value = _mm256_set1_epi16(pad_value_bf16_fp16);
                padding_constant_pack16_bf16s_fp16s_avx512(bottom_blob, top_blob, 0, 0, left / 16, right / 16, pad_value);

                return 0;
            }
        }

        if (dims == 2)
        {
            int outw = w + left + right;
            int outh = h * elempack + top + bottom;

            int out_elempack = outh % 16 == 0 ? 16 : outh % 8 == 0 ? 8 : outh % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (top % 16 == 0 && out_elempack == 16 && type == 0)
            {
                top_blob.create(outw, outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                __m256i pad_value = _mm256_set1_epi16(pad_value_bf16_fp16);
                padding_constant_pack16_bf16s_fp16s_avx512(bottom_blob, top_blob, top / 16, bottom / 16, left, right, pad_value);

                return 0;
            }
        }

        if (dims == 3)
        {
            int outw = w + left + right;
            int outh = h + top + bottom;
            int outc = channels * elempack + front + behind;

            int out_elempack = outc % 16 == 0 ? 16 : outc % 8 == 0 ? 8 : outc % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (front % 16 == 0 && out_elempack == 16 && !(outc != channels * elempack && type != 0))
            {
                top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                int front_ = front / elempack;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < outc / out_elempack; q++)
                {
                    Mat borderm = top_blob.channel(q);

                    __m256i pad_value = per_channel_pad_data_size ? _mm256_loadu_si256((const __m256i*)((const unsigned short*)per_channel_pad_data_bf16_fp16 + q * 16)) : _mm256_set1_epi16(pad_value_bf16_fp16);
                    //Channel padding
                    if ((q - front_) < 0 || (q - front_) >= channels)
                    {
                        borderm.fill(pad_value);
                    }
                    else
                    {
                        const Mat m = bottom_blob.channel(q - front_);
                        if (type == 0)
                            padding_constant_pack16_bf16s_fp16s_avx512(m, borderm, top, bottom, left, right, pad_value);
                        if (type == 1)
                            padding_replicate_pack16_bf16s_fp16s_avx512(m, borderm, top, bottom, left, right);
                        if (type == 2)
                            padding_reflect_pack16_bf16s_fp16s_avx512(m, borderm, top, bottom, left, right);
                    }
                }

                return 0;
            }
        }

        if (dims == 4)
        {
            int outw = w + left + right;
            int outh = h + top + bottom;
            int outd = d + front + behind;

            if (type == 0)
            {
                top_blob.create(outw, outh, outd, channels, elemsize, elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    __m256i pad_value = per_channel_pad_data_size ? _mm256_loadu_si256((const __m256i*)((const unsigned short*)per_channel_pad_data_bf16_fp16 + q * 16)) : _mm256_set1_epi16(pad_value_bf16_fp16);

                    for (int z = 0; z < outd; z++)
                    {
                        Mat borderm = top_blob.channel(q).depth(z);

                        // depth padding
                        if ((z - front) < 0 || (z - front) >= d)
                        {
                            borderm.fill(pad_value);
                        }
                        else
                        {
                            const Mat m = bottom_blob.channel(q).depth(z - front);
                            padding_constant_pack16_bf16s_fp16s_avx512(m, borderm, top, bottom, left, right, pad_value);
                        }
                    }
                }

                return 0;
            }
        }
    }
#endif // __AVX512F__

    if (elempack == 8)
    {
        if (dims == 1)
        {
            int outw = w * elempack + left + right;

            int out_elempack = outw % 8 == 0 ? 8 : outw % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (left % 8 == 0 && out_elempack == 8 && type == 0)
            {
                top_blob.create(outw / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                __m128i pad_value = _mm_set1_epi16(pad_value_bf16_fp16);
                padding_constant_pack8_bf16s_fp16s_avx(bottom_blob, top_blob, 0, 0, left / 8, right / 8, pad_value);

                return 0;
            }
        }

        if (dims == 2)
        {
            int outw = w + left + right;
            int outh = h * elempack + top + bottom;

            int out_elempack = outh % 8 == 0 ? 8 : outh % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (top % 8 == 0 && out_elempack == 8 && type == 0)
            {
                top_blob.create(outw, outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                __m128i pad_value = _mm_set1_epi16(pad_value_bf16_fp16);
                padding_constant_pack8_bf16s_fp16s_avx(bottom_blob, top_blob, top / 8, bottom / 8, left, right, pad_value);

                return 0;
            }
        }

        if (dims == 3)
        {
            int outw = w + left + right;
            int outh = h + top + bottom;
            int outc = channels * elempack + front + behind;

            int out_elempack = outc % 8 == 0 ? 8 : outc % 4 == 0 ? 4 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (front % 8 == 0 && out_elempack == 8 && !(outc != channels * elempack && type != 0))
            {
                top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                int front_ = front / elempack;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < outc / out_elempack; q++)
                {
                    Mat borderm = top_blob.channel(q);

                    __m128i pad_value = per_channel_pad_data_size ? _mm_loadu_si128((const __m128i*)((const unsigned short*)per_channel_pad_data_bf16_fp16 + q * 8)) : _mm_set1_epi16(pad_value_bf16_fp16);
                    //Channel padding
                    if ((q - front_) < 0 || (q - front_) >= channels)
                    {
                        borderm.fill(pad_value);
                    }
                    else
                    {
                        const Mat m = bottom_blob.channel(q - front_);
                        if (type == 0)
                            padding_constant_pack8_bf16s_fp16s_avx(m, borderm, top, bottom, left, right, pad_value);
                        if (type == 1)
                            padding_replicate_pack8_bf16s_fp16s_avx(m, borderm, top, bottom, left, right);
                        if (type == 2)
                            padding_reflect_pack8_bf16s_fp16s_avx(m, borderm, top, bottom, left, right);
                    }
                }

                return 0;
            }
        }

        if (dims == 4)
        {
            int outw = w + left + right;
            int outh = h + top + bottom;
            int outd = d + front + behind;

            if (type == 0)
            {
                top_blob.create(outw, outh, outd, channels, elemsize, elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    __m128i pad_value = per_channel_pad_data_size ? _mm_loadu_si128((const __m128i*)((const unsigned short*)per_channel_pad_data_bf16_fp16 + q * 8)) : _mm_set1_epi16(pad_value_bf16_fp16);

                    for (int z = 0; z < outd; z++)
                    {
                        Mat borderm = top_blob.channel(q).depth(z);

                        // depth padding
                        if ((z - front) < 0 || (z - front) >= d)
                        {
                            borderm.fill(pad_value);
                        }
                        else
                        {
                            const Mat m = bottom_blob.channel(q).depth(z - front);
                            padding_constant_pack8_bf16s_fp16s_avx(m, borderm, top, bottom, left, right, pad_value);
                        }
                    }
                }

                return 0;
            }
        }
    }
#endif // __AVX__

    if (elempack == 4)
    {
        if (dims == 1)
        {
            int outw = w * elempack + left + right;

#if __AVX__
            int out_elempack = outw % 8 == 0 ? 8 : outw % 4 == 0 ? 4 : 1;
#else
            int out_elempack = outw % 4 == 0 ? 4 : 1;
#endif
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (left % 4 == 0 && out_elempack == 4 && type == 0)
            {
                top_blob.create(outw / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                int64_t v16 = (int64_t)(unsigned short)pad_value_bf16_fp16;
                int64_t pad_value_i64 = v16 | (v16 << 16) | (v16 << 32) | (v16 << 48);
                padding_constant_pack4_bf16s_fp16s_sse(bottom_blob, top_blob, 0, 0, left / 4, right / 4, pad_value_i64);

                return 0;
            }
        }

        if (dims == 2)
        {
            int outw = w + left + right;
            int outh = h * elempack + top + bottom;

#if __AVX__
            int out_elempack = outh % 8 == 0 ? 8 : outh % 4 == 0 ? 4 : 1;
#else
            int out_elempack = outh % 4 == 0 ? 4 : 1;
#endif
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (top % 4 == 0 && out_elempack == 4 && type == 0)
            {
                top_blob.create(outw, outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                int64_t v16 = (int64_t)(unsigned short)pad_value_bf16_fp16;
                int64_t pad_value_i64 = v16 | (v16 << 16) | (v16 << 32) | (v16 << 48);
                padding_constant_pack4_bf16s_fp16s_sse(bottom_blob, top_blob, top / 4, bottom / 4, left, right, pad_value_i64);

                return 0;
            }
        }

        if (dims == 3)
        {
            int outw = w + left + right;
            int outh = h + top + bottom;
            int outc = channels * elempack + front + behind;

#if __AVX__
            int out_elempack = outc % 8 == 0 ? 8 : outc % 4 == 0 ? 4 : 1;
#else
            int out_elempack = outc % 4 == 0 ? 4 : 1;
#endif
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (front % 4 == 0 && out_elempack == 4 && !(outc != channels * elempack && type != 0))
            {
                top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                int front_ = front / elempack;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < outc / out_elempack; q++)
                {
                    Mat borderm = top_blob.channel(q);

                    int64_t pad_value_i64;
                    if (per_channel_pad_data_size)
                    {
                        const unsigned short* p = (const unsigned short*)per_channel_pad_data_bf16_fp16 + q * 4;
                        pad_value_i64 = *(const int64_t*)p;
                    }
                    else
                    {
                        int64_t v16 = (int64_t)(unsigned short)pad_value_bf16_fp16;
                        pad_value_i64 = v16 | (v16 << 16) | (v16 << 32) | (v16 << 48);
                    }
                    //Channel padding
                    if ((q - front_) < 0 || (q - front_) >= channels)
                    {
                        borderm.fill<int64_t>(pad_value_i64);
                    }
                    else
                    {
                        const Mat m = bottom_blob.channel(q - front_);
                        if (type == 0)
                            padding_constant_pack4_bf16s_fp16s_sse(m, borderm, top, bottom, left, right, pad_value_i64);
                        if (type == 1)
                            padding_replicate_pack4_bf16s_fp16s_sse(m, borderm, top, bottom, left, right);
                        if (type == 2)
                            padding_reflect_pack4_bf16s_fp16s_sse(m, borderm, top, bottom, left, right);
                    }
                }

                return 0;
            }
        }

        if (dims == 4)
        {
            int outw = w + left + right;
            int outh = h + top + bottom;
            int outd = d + front + behind;

            if (type == 0)
            {
                top_blob.create(outw, outh, outd, channels, elemsize, elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    int64_t pad_value_i64;
                    if (per_channel_pad_data_size)
                    {
                        const unsigned short* p = (const unsigned short*)per_channel_pad_data_bf16_fp16 + q * 4;
                        pad_value_i64 = *(const int64_t*)p;
                    }
                    else
                    {
                        int64_t v16 = (int64_t)(unsigned short)pad_value_bf16_fp16;
                        pad_value_i64 = v16 | (v16 << 16) | (v16 << 32) | (v16 << 48);
                    }

                    for (int z = 0; z < outd; z++)
                    {
                        Mat borderm = top_blob.channel(q).depth(z);

                        // depth padding
                        if ((z - front) < 0 || (z - front) >= d)
                        {
                            borderm.fill<int64_t>(pad_value_i64);
                        }
                        else
                        {
                            const Mat m = bottom_blob.channel(q).depth(z - front);
                            padding_constant_pack4_bf16s_fp16s_sse(m, borderm, top, bottom, left, right, pad_value_i64);
                        }
                    }
                }

                return 0;
            }
        }
    }
#endif // __SSE2__

    Mat bottom_blob_unpacked = bottom_blob;
    if (elempack != 1)
    {
        Option opt_pack1 = opt;
        opt_pack1.blob_allocator = opt.workspace_allocator;

        convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack1);
        if (bottom_blob_unpacked.empty())
            return -100;
    }

    return Padding::forward(bottom_blob_unpacked, top_blob, opt);
}

int Padding_x86::forward_int8(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

#if __SSE2__
    if (elempack == 8)
    {
        if (dims == 1)
        {
            int outw = w * elempack + left + right;

            int out_elempack = outw % 8 == 0 ? 8 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (left % 8 == 0 && out_elempack == 8 && type == 0)
            {
                top_blob.create(outw / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                int64_t v8 = (int64_t)value;
                int64_t pad_value = v8 | (v8 << 8) | (v8 << 16) | (v8 << 24) | (v8 << 32) | (v8 << 40) | (v8 << 48) | (v8 << 56);
                padding_constant_pack8_int8_sse(bottom_blob, top_blob, 0, 0, left / 8, right / 8, pad_value);

                return 0;
            }
        }

        if (dims == 2)
        {
            int outw = w + left + right;
            int outh = h * elempack + top + bottom;

            int out_elempack = outh % 8 == 0 ? 8 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (top % 8 == 0 && out_elempack == 8 && type == 0)
            {
                top_blob.create(outw, outh / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                int64_t v8 = (int64_t)value;
                int64_t pad_value = v8 | (v8 << 8) | (v8 << 16) | (v8 << 24) | (v8 << 32) | (v8 << 40) | (v8 << 48) | (v8 << 56);
                padding_constant_pack8_int8_sse(bottom_blob, top_blob, top / 8, bottom / 8, left, right, pad_value);

                return 0;
            }
        }

        if (dims == 3)
        {
            int outw = w + left + right;
            int outh = h + top + bottom;
            int outc = channels * elempack + front + behind;

            int out_elempack = outc % 8 == 0 ? 8 : 1;
            size_t out_elemsize = elemsize / elempack * out_elempack;

            if (front % 8 == 0 && out_elempack == 8 && !(outc != channels * elempack && type != 0))
            {
                top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                int front_ = front / elempack;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < outc / out_elempack; q++)
                {
                    Mat borderm = top_blob.channel(q);

                    // TODO perchannel
                    //                     int64_t pad_value = per_channel_pad_data_size ? vld1_s8(per_channel_pad_data + q * 8) : vdup_n_s8((signed char)value);
                    int64_t v8 = (int64_t)value;
                    int64_t pad_value = v8 | (v8 << 8) | (v8 << 16) | (v8 << 24) | (v8 << 32) | (v8 << 40) | (v8 << 48) | (v8 << 56);

                    //Channel padding
                    if ((q - front_) < 0 || (q - front_) >= channels)
                    {
                        borderm.fill<int64_t>(pad_value);
                    }
                    else
                    {
                        const Mat m = bottom_blob.channel(q - front_);
                        if (type == 0)
                            padding_constant_pack8_int8_sse(m, borderm, top, bottom, left, right, pad_value);
                        if (type == 1)
                            padding_replicate_pack8_int8_sse(m, borderm, top, bottom, left, right);
                        if (type == 2)
                            padding_reflect_pack8_int8_sse(m, borderm, top, bottom, left, right);
                    }
                }

                return 0;
            }
        }

        if (dims == 4)
        {
            int outw = w + left + right;
            int outh = h + top + bottom;
            int outd = d + front + behind;

            if (type == 0)
            {
                top_blob.create(outw, outh, outd, channels, elemsize, elempack, opt.blob_allocator);
                if (top_blob.empty())
                    return -100;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    // TODO perchannel
                    //                     int64_t pad_value = per_channel_pad_data_size ? vld1_s8(per_channel_pad_data + q * 8) : vdup_n_s8((signed char)value);
                    int64_t v8 = (int64_t)value;
                    int64_t pad_value = v8 | (v8 << 8) | (v8 << 16) | (v8 << 24) | (v8 << 32) | (v8 << 40) | (v8 << 48) | (v8 << 56);

                    for (int z = 0; z < outd; z++)
                    {
                        Mat borderm = top_blob.channel(q).depth(z);

                        // depth padding
                        if ((z - front) < 0 || (z - front) >= d)
                        {
                            borderm.fill<int64_t>(pad_value);
                        }
                        else
                        {
                            const Mat m = bottom_blob.channel(q).depth(z - front);
                            padding_constant_pack8_int8_sse(m, borderm, top, bottom, left, right, pad_value);
                        }
                    }
                }

                return 0;
            }
        }
    }
#endif // __SSE2__

    Mat bottom_blob_unpacked = bottom_blob;
    if (elempack != 1)
    {
        Option opt_pack1 = opt;
        opt_pack1.blob_allocator = opt.workspace_allocator;

        convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_pack1);
        if (bottom_blob_unpacked.empty())
            return -100;
    }

    return Padding::forward(bottom_blob_unpacked, top_blob, opt);
}

} // namespace ncnn
